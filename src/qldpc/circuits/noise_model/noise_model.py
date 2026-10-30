# SPDX-License-Identifier: Apache-2.0

"""Noise models, which add noise to circuits.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

import collections
import math
from collections.abc import Callable, Collection, Iterable, Mapping

import stim

from qldpc._utils import format_docstring

from ..common import StimCircuitLike
from ._moments import (
    categorize_moment_qubits,
    compose_idle_channels,
    gate_qubits,
    get_immune_qubits,
    idle_barrier_qubits,
    quadratic_idle_channel,
    repeated_idle_channel,
    split_targets_if_needed,
)
from ._operations import (
    ANNOTATION,
    CLIFFORD_1Q,
    CLIFFORD_2Q,
    CLIFFORD_PP,
    GATE_OP_TYPES,
    IDLE_OPS,
    JUST_MEASURE_OPS,
    JUST_MEASURE_PP,
    JUST_RESET_OPS,
    MEASURE_AND_RESET_OPS,
    NOISE,
    canonical_rule_key,
    get_gate_aliases,
    involves_classical_bits,
    known_gate_shape,
    op_type,
)
from .channels import _ABSOLUTE_ERROR_TOLERANCE, PauliChannel, PauliChannelSequence
from .rules import (
    ErrorSpec,
    NoiseRule,
    _as_noise_rule,
    _normalize_clifford_nq_error,
    _rule_with_immunity,
    _validate_custom_rule,
    _validate_rule_for_arity,
)

# Tags that, by default, make an operation or repeat block noiseless, or declare that the qubits of
# a tagged QUBIT_COORDS annotation are immune to noise.
DEFAULT_IMMUNE_OP_TAG = "__IMMUNE_TO_NOISE__"
DEFAULT_IMMUNE_QUBIT_TAG = "__IMMUNE_QUBIT_TO_NOISE__"

####################################################################################################
# primary methods and classes: as_noiseless_circuit and NoiseModel


def as_noiseless_circuit(circuit: StimCircuitLike) -> StimCircuitLike:
    """Wrap a circuit in a one-repetition block tagged immune to :class:`NoiseModel`."""
    if (
        len(circuit) == 1
        and isinstance(circuit[0], stim.CircuitRepeatBlock)
        and DEFAULT_IMMUNE_OP_TAG in circuit[0].tag
    ):
        return circuit
    # Read the circuit's operations into a stim.Circuit body (the repeat block requires a
    # stim.Circuit).  Iterating by index works for a stim.Circuit and any stim-wrapping circuit.
    body = stim.Circuit()
    for index in range(len(circuit)):
        body.append(circuit[index])

    block = stim.CircuitRepeatBlock(repeat_count=1, body=body, tag=DEFAULT_IMMUNE_OP_TAG)
    noiseless_circuit = type(circuit)()
    noiseless_circuit.append(block)
    return noiseless_circuit


class NoiseModel:
    """A model that defines how to add noise to quantum circuits.

    This class provides a framework for adding various types of noise to quantum circuits, including
    gate errors, readout errors, reset errors, and idling errors.  Classically controlled operations
    are assumed to NOT occur, so the corresponding qubits pick up idling errors, if applicable.
    Explicit ``I`` and ``II`` markers are likewise treated as idle time rather than ordinary
    Clifford gates, but ``rules`` and ``rule_func`` may assign additional custom noise to them.

    Args:
        clifford_1q_error: Default noise applied after each one-qubit unitary Clifford gate.  A
            float ``p`` is a uniform 1-qubit depolarizing channel of total error probability
            ``p``.  Also accepts a ``Mapping[str, float]`` from one-qubit non-identity
            Pauli-strings to probabilities, a ``PauliChannel``, or a ``NoiseRule``.
        clifford_2q_error: Default noise applied after each two-qubit unitary Clifford gate.  A
            float ``p`` is a uniform 2-qubit depolarizing channel of total error probability
            ``p``.  Also accepts a ``Mapping[str, float]`` from two-qubit non-identity
            Pauli-strings to probabilities, a ``PauliChannel``, or a ``NoiseRule``.
        readout_error: Default probability of flipping measurement results.
        reset_error: Default probability of resetting qubits to the wrong state.
        clifford_nq_error: Optional mapping from a qubit count ``k`` to the noise applied after
            each ``k``-qubit unitary Clifford gate, similarly to ``clifford_?q_error``.
            Specifying both ``clifford_nq_error[1]`` and ``clifford_1q_error`` raises an
            ambiguity error; likewise with ``clifford_nq_error[2]`` and ``clifford_2q_error``.
            This dispatch intentionally excludes MXX/MYY/MZZ and MPP measurements.  SPP uses
            this dispatch when a matching arity is configured; there is no fallback for an
            unconfigured multi-qubit arity.  Configure basis-specific rules explicitly when
            needed.
        idle_error: Noise rule or depolarization probability applied to each idling qubit in any
            given moment.  If a NoiseRule is provided, its `after` channels are appended to the
            idle qubits (its readout_error/reset_error fields are ignored).  Consecutive
            Pauli-channel idle errors on a qubit are composed into one equivalent error; custom
            Stim noise fragments are applied once per idle moment.
        quadratic_idle_error: Error on each idling qubit in one moment, given as a float ``p``
            (a uniform channel of total error probability ``p``), a ``Mapping[str, float]``
            such as ``{"Z": p}``, or a one-qubit ``PauliChannel``, with total probability at
            most 1/2.  Over a run of ``k`` consecutive idle moments, this error has total
            probability ``(1 - (1 - 2 * p)**(k**2)) / 2``, split among Paulis as in one moment,
            and is combined with the ``idle_error`` of the same moments.  See
            :mod:`qldpc.circuits.noise_model` for its physical model and assumptions.
        additional_error_waiting_for_m_or_r: Additional noise rule or depolarization probability
            applied to every non-collapsing qubit while another qubit undergoes measurement or
            reset, including qubits undergoing a unitary operation.  Accepts the same NoiseRule
            fields as ``idle_error``, but is always applied once per moment.
        rules: Dictionary mapping specific gate names to their noise rules.  Overrides the
            arity-based defaults for unitary, measurement, and reset gates.  Explicit ``I`` and
            ``II`` markers accept one- and two-qubit ``after`` rules, respectively.  If
            ``idle_error`` is also configured, both kinds of noise are applied.
        rule_func: Optional callback function that maps a ``stim.CircuitInstruction`` to a
            ``NoiseRule``.  The callback is evaluated before named and default rules.  Broadcast
            instructions are split first, so each callback input represents one gate
            application: one target for a one-qubit gate, two for a two-qubit gate, or one Pauli
            product for an SPP/MPP instruction.  The callback receives unitary Cliffords,
            measurements, resets, and explicit ``I`` / ``II`` idle markers, but not annotations,
            pure-noise instructions, or classically controlled operations.  Returning ``None``
            continues with named and default rule lookup.  Identities can fall back to named
            rules, but not to Clifford defaults.

    Raises:
        ValueError: If readout_error or reset_error is not between 0 and 1, if a noise
            specification is ambiguous or duplicated, or if a rule does not fit the gates that
            it applies to.
    """

    def __init__(
        self,
        clifford_1q_error: ErrorSpec | None = None,
        clifford_2q_error: ErrorSpec | None = None,
        readout_error: float | None = None,
        reset_error: float | None = None,
        *,
        clifford_nq_error: Mapping[int, ErrorSpec] | None = None,
        idle_error: ErrorSpec | None = None,
        quadratic_idle_error: float | Mapping[str, float] | PauliChannel | None = None,
        additional_error_waiting_for_m_or_r: ErrorSpec | None = None,
        rules: Mapping[str, NoiseRule] | None = None,
        rule_func: Callable[[stim.CircuitInstruction], NoiseRule | None] | None = None,
    ) -> None:
        """Initializes a noise model with specified parameters."""
        self.rules = rules
        self.rule_func = rule_func
        if rules is not None:
            normalized_rules: dict[str, NoiseRule] = {}
            # Validate rules whose (arity, can_measure, can_reset) is known — fixed-arity stim gates
            # and basis-suffixed rule keys (``"MXYZ"`` for MPP X*Y*Z, ``"SXY_DAG"`` for SPP_DAG
            # X*Y).  Bare ``"MPP"`` / ``"SPP"`` / ``"SPP_DAG"`` are variable-arity and deferred to
            # emission-time checks via ``emit_after``.
            for op_name, rule in rules.items():
                canonical_name = canonical_rule_key(op_name)
                if canonical_name in normalized_rules:
                    raise ValueError(f"Duplicate noise rules after canonicalization: {op_name!r}")
                normalized_rules[canonical_name] = rule
                shape = known_gate_shape(canonical_name)
                if shape is not None:
                    arity, can_measure, can_reset = shape
                    _validate_rule_for_arity(
                        rule,
                        arity,
                        f"rules[{op_name!r}]",
                        can_measure=can_measure,
                        can_reset=can_reset,
                    )
            self.rules = normalized_rules
        for probability in (readout_error, reset_error):
            if probability is not None and not (0 <= probability <= 1):
                raise ValueError(f"{probability=} is not between 0 and 1")
        self.readout_error = readout_error or 0
        self.reset_error = reset_error or 0

        # `clifford_1q_error` / `clifford_2q_error` are syntactic sugar for `clifford_nq_error[1]` /
        # `clifford_nq_error[2]`; internally we normalize everything into a single
        # `clifford_nq_error` dict.  Ambiguity detection is symmetric on RAW inputs — an argument
        # counts as "user-specified" if the user passed anything other than None (including a zero
        # float, an empty NoiseRule, etc.), even if it normalizes to a no-op.
        for size, param, name in (
            (1, clifford_1q_error, "clifford_1q_error"),
            (2, clifford_2q_error, "clifford_2q_error"),
        ):
            if clifford_nq_error is not None and size in clifford_nq_error and param is not None:
                raise ValueError(
                    f"Ambiguous noise specification: both `clifford_nq_error[{size}]` and"
                    f"`{name}` are set.  Specify one or the other."
                )
        merged_nq_input: dict[int, ErrorSpec] = dict(clifford_nq_error) if clifford_nq_error else {}
        rule_1q = _as_noise_rule(clifford_1q_error, 1)
        rule_2q = _as_noise_rule(clifford_2q_error, 2)
        if rule_1q is not None:
            merged_nq_input[1] = rule_1q
        if rule_2q is not None:
            merged_nq_input[2] = rule_2q
        self.clifford_nq_error = _normalize_clifford_nq_error(merged_nq_input)

        self.idle_error = _as_noise_rule(idle_error, 1)
        self.additional_error_waiting_for_m_or_r = _as_noise_rule(
            additional_error_waiting_for_m_or_r, 1
        )
        # Idle noise is applied per-qubit, so anything with a declared arity != 1 has no natural
        # interpretation here.  Validate the declared arity BEFORE trivializing so a user typo like
        # ``PauliChannel({}, num_qubits=3)`` — trivial but wrongly-shaped — is surfaced.
        for field_name, idle_rule in (
            ("idle_error", self.idle_error),
            ("additional_error_waiting_for_m_or_r", self.additional_error_waiting_for_m_or_r),
        ):
            if idle_rule is None:
                continue
            if idle_rule.after is not None and idle_rule.after.num_qubits != 1:
                raise ValueError(
                    f"`{field_name}` does not support a multi-qubit `after` rule "
                    f"(arity={idle_rule.after.num_qubits})."
                )
        # A quadratic idle error is a one-qubit Pauli channel that flips a qubit at most half the
        # time in one moment.
        self.quadratic_idle_error: PauliChannel | None = None
        quadratic_idle_rule = _as_noise_rule(quadratic_idle_error, 1)
        if quadratic_idle_rule is not None and quadratic_idle_rule.after is not None:
            channel = quadratic_idle_rule.after
            if not isinstance(channel, PauliChannel) or channel.num_qubits != 1:
                raise ValueError("`quadratic_idle_error` must be a one-qubit PauliChannel")
            if math.fsum(channel.probabilities.values()) > 0.5 + _ABSOLUTE_ERROR_TOLERANCE:
                raise ValueError(
                    "`quadratic_idle_error` must have a total probability of at most 1/2"
                )
            self.quadratic_idle_error = channel or None
        # Guards done — now trivialize.
        self.idle_error = self.idle_error or None
        self.additional_error_waiting_for_m_or_r = self.additional_error_waiting_for_m_or_r or None

    @property
    def clifford_1q_error(self) -> NoiseRule | None:
        """Convenience view: ``clifford_nq_error[1]`` if set, else None."""
        return self.clifford_nq_error.get(1)

    @property
    def clifford_2q_error(self) -> NoiseRule | None:
        """Convenience view: ``clifford_nq_error[2]`` if set, else None."""
        return self.clifford_nq_error.get(2)

    def __bool__(self) -> bool:
        """Is this noise model nontrivial?"""
        return (
            bool(self.rules)
            or self.rule_func is not None
            or bool(self.clifford_nq_error)
            or bool(self.readout_error)
            or bool(self.reset_error)
            or bool(self.idle_error)
            or bool(self.quadratic_idle_error)
            or bool(self.additional_error_waiting_for_m_or_r)
        )

    def __str__(self) -> str:
        """A constructor-like description listing only the set fields.

        When ``rule_func`` is ``None`` this reads like a faithful ``repr`` — an eval-able
        ``NoiseModel(...)`` call.  A ``rule_func`` cannot be reproduced faithfully, so it is instead
        described by its qualified name and module (falling back to its type name for callables
        without a ``__qualname__``, e.g. a callable object or ``functools.partial``), which makes
        the result non-eval-able.
        """
        args = []
        if self.clifford_nq_error:
            args.append(f"clifford_nq_error={self.clifford_nq_error!r}")
        if self.readout_error:
            args.append(f"readout_error={self.readout_error!r}")
        if self.reset_error:
            args.append(f"reset_error={self.reset_error!r}")
        if self.idle_error is not None:
            args.append(f"idle_error={self.idle_error!r}")
        if self.quadratic_idle_error is not None:
            args.append(f"quadratic_idle_error={self.quadratic_idle_error!r}")
        if self.additional_error_waiting_for_m_or_r is not None:
            args.append(
                f"additional_error_waiting_for_m_or_r={self.additional_error_waiting_for_m_or_r!r}"
            )
        if self.rules:
            args.append(f"rules={dict(self.rules)!r}")
        if self.rule_func is not None:
            func = self.rule_func
            name = getattr(func, "__qualname__", None) or type(func).__name__
            args.append(f"rule_func=<{getattr(func, '__module__', '?')}.{name}>")
        return f"NoiseModel({', '.join(args)})"

    def __repr__(self) -> str:
        if self.rule_func is None:
            return str(self)
        return super().__repr__()

    def get_noise_rule(self, op: stim.CircuitInstruction) -> NoiseRule | None:
        """Determines the noise rule to apply to a specific operation.

        Noise rules are consulted in the following order of precedence:

        1. ``rule_func`` (stim.CircuitInstruction -> NoiseRule factory).
        2. ``rules`` (name-based NoiseRules).
        3. ``clifford_nq_error`` (arity-based NoiseRules for unitary Cliffords).
        4. ``readout_error`` and/or ``reset_error`` (per-gate defaults for measurement/reset ops).

        For explicit idle markers (``I`` and ``II``), ``rule_func`` is consulted first, followed by
        named ``rules``.  Clifford defaults do not apply.  Because automatic ``idle_error`` is
        evaluated separately, an identity can receive both kinds of noise.

        .. note::
            MPP / SPP / SPP_DAG instructions passed to this method must contain exactly one Pauli
            product (e.g. ``MPP X0*Y1*Z2``, not ``MPP X0*Y1 Z2*X3``).  Multi-product instructions
            are decomposed upstream by ``_split_targets_pp`` before this method is invoked;
            hand-calling with an unsplit multi-product op raises ``ValueError`` from
            ``get_gate_aliases``.

        Args:
            op: The circuit instruction to find a noise rule for.

        Returns:
            The NoiseRule to apply for the given operation, or None for no noise.
        """
        this_op_type = op_type(op.name)
        if this_op_type in (ANNOTATION, NOISE) or involves_classical_bits(op):
            return None
        # the number of qubits in the Pauli product of an MPP, SPP, or SPP_DAG instruction
        product_arity = (
            sum(1 for target in op.targets_copy() if not target.is_combiner)
            if this_op_type in (CLIFFORD_PP, JUST_MEASURE_PP)
            else 0
        )

        if self.rule_func is not None and this_op_type in GATE_OP_TYPES:
            rule = self.rule_func(op)
            if rule is not None:
                _validate_custom_rule(rule, op)
                return rule

        if self.rules is not None:
            for name in get_gate_aliases(op):
                rule = self.rules.get(name)
                if rule is not None:
                    if name in {"MPP", "SPP", "SPP_DAG"}:
                        if rule.after is not None and rule.after.num_qubits != product_arity:
                            raise ValueError(
                                f"This NoiseRule has arity {rule.after.num_qubits}, but"
                                f" {op.name!r} has {product_arity} qubit targets"
                            )
                        if rule.readout_error is not None and op.name not in JUST_MEASURE_OPS:
                            raise ValueError(f"`readout_error` is not valid on {op.name!r}")
                        if rule.reset_error is not None:
                            raise ValueError(f"`reset_error` is not valid on {op.name!r}")
                    return rule

        if op.name in IDLE_OPS:
            return None

        if this_op_type in (CLIFFORD_1Q, CLIFFORD_2Q, CLIFFORD_PP):
            if this_op_type == CLIFFORD_1Q:
                num_qubits = 1
            elif this_op_type == CLIFFORD_2Q:
                num_qubits = 2
            else:
                num_qubits = product_arity
            if num_qubits in self.clifford_nq_error:
                return self.clifford_nq_error[num_qubits]

        if self.readout_error and op.name in JUST_MEASURE_OPS:
            return NoiseRule(readout_error=self.readout_error)
        if self.reset_error and op.name in JUST_RESET_OPS:
            return NoiseRule(reset_error=self.reset_error)
        if (self.readout_error or self.reset_error) and op.name in MEASURE_AND_RESET_OPS:
            return NoiseRule(readout_error=self.readout_error, reset_error=self.reset_error)

        return None

    @format_docstring(
        DEFAULT_IMMUNE_OP_TAG=DEFAULT_IMMUNE_OP_TAG,
        DEFAULT_IMMUNE_QUBIT_TAG=DEFAULT_IMMUNE_QUBIT_TAG,
    )
    def noisy_circuit(
        self,
        circuit: StimCircuitLike,
        *,
        system_qubits: Iterable[int] | None = None,
        immune_qubits: Iterable[int] = (),
        immune_op_tag: str | None = DEFAULT_IMMUNE_OP_TAG,
        immune_qubit_tag: str | None = DEFAULT_IMMUNE_QUBIT_TAG,
        immunize_gates: bool = True,
        insert_ticks: bool = True,
    ) -> StimCircuitLike:
        """Construct a noisy version of the given circuit.

        This method first uses TICKs to split the input circuit into moments of operations that can
        be applied in parallel, thereby preventing qubit reuse conflicts.  Noise is then applied to
        each operation according to the rules of this NoiseModel.

        Args:
            circuit: The circuit to apply noise to.
            system_qubits: All qubits that are used by the circuit or are otherwise allowed to
                accumulate idling errors.  Defaults to set(range(circuit.num_qubits)).
            immune_qubits: Qubits that are declared to be immune to noise.  Defaults to none.
            immune_op_tag: If an operation contains this string in its tag, that operation is
                noiseless.  Set to None to disable tagged operation immunity.  Default:
                "{DEFAULT_IMMUNE_OP_TAG}".
            immune_qubit_tag: If a QUBIT_COORDS annotation contains this string in its tag, its
                target qubits are immune to noise.  Set to None to disable annotated immunity.
                Default: "{DEFAULT_IMMUNE_QUBIT_TAG}".
            immunize_gates: If True (the default), a gate that touches an immune qubit is treated as
                noiseless.  Otherwise, its Pauli noise is conditioned on the absence of errors on
                noise-immune qubits, keeping only strings that act as ``I`` on every immune qubit.
                An immune-tagged operation is still a real operation for moment accounting; it does
                not suppress idle noise on unrelated qubits in the same moment.
            insert_ticks: If True, automatically inserts TICK operations to prevent qubit reuse
                conflicts.  If False, assumes that this preprocessing is not necessary.  In either
                case, moments are identified as though all repeat blocks (other than those tagged
                with ``immune_op_tag``) were flattened.

        Returns:
            The input circuit with added noise.

        Raises:
            ValueError: If ``insert_ticks`` is False and a qubit is operated on more than once in a
                moment, including in different iterations of a repeat block without TICKs.
        """
        if immune_op_tag == "":
            raise ValueError("immune_op_tag must be non-empty or None")
        if immune_qubit_tag == "":
            raise ValueError("immune_qubit_tag must be non-empty or None")
        if not isinstance(circuit, stim.Circuit):
            # convert to stim, add noise, convert back
            stim_input = stim.Circuit()
            for index in range(len(circuit)):
                stim_input.append(circuit[index])
            stim_output = self.noisy_circuit(
                stim_input,
                system_qubits=system_qubits,
                immune_qubits=immune_qubits,
                immune_op_tag=immune_op_tag,
                immune_qubit_tag=immune_qubit_tag,
                immunize_gates=immunize_gates,
                insert_ticks=insert_ticks,
            )
            output = type(circuit)()
            output += stim_output
            return output

        system_qubits = frozenset(
            range(circuit.num_qubits) if system_qubits is None else system_qubits
        )
        immune_qubits = frozenset(immune_qubits) | (
            get_immune_qubits(circuit, immune_qubit_tag)
            if immune_qubit_tag is not None
            else frozenset()
        )

        noisy_circuit = stim.Circuit()
        builder = _NoisyCircuitBuilder(
            self,
            system_qubits=system_qubits,
            immune_qubits=immune_qubits,
            immune_op_tag=immune_op_tag,
            immunize_gates=immunize_gates,
            insert_ticks=insert_ticks,
        )
        builder.append(circuit, noisy_circuit)
        builder.finish(noisy_circuit)
        return noisy_circuit

    def _inplace_append_noisy_moment(
        self,
        *,
        circuit: stim.Circuit,
        moment: Collection[stim.CircuitInstruction | stim.CircuitRepeatBlock],
        system_qubits: frozenset[int],
        immune_qubits: frozenset[int],
        immune_op_tag: str | None,
        immunize_gates: bool,
        pending_idle_moments: dict[int, int],
    ) -> None:
        """Applies noise to a moment and appends it to a circuit (in-place).

        This method processes all operations in a moment, applies their respective noise rules, and
        adds the resulting noisy operations to the output circuit.

        Args:
            circuit: The circuit to append the noisy operations to.
            moment: Collection of operations happening during the moment in question.
            system_qubits: Set of all qubits in the system that may experience idle errors.
            immune_qubits: Qubits that are declared to be immune to noise.
            immune_op_tag: If an operation contains this string in its tag, that operation is
                noiseless.
            immunize_gates: If True (the default), a gate that touches an immune qubit is treated as
                noiseless.  Otherwise, its Pauli noise is conditioned on the absence of errors on
                noise-immune qubits, keeping only strings that act as ``I`` on every immune qubit.
                An immune-tagged operation does not suppress idle noise on unrelated qubits in the
                same moment.
            pending_idle_moments: Map from each qubit to its number of idle moments that have not
                yet been emitted as noise.  Updated in-place.

        Raises:
            ValueError: If qubits are operated on multiple times within the same moment without a
                TICK in between (violating the "each qubit at most once per moment" invariant that
                noise application relies on).
        """
        collapsed_qubits, operation_qubits = categorize_moment_qubits(moment)

        noise_after_moment = stim.Circuit()
        for op in moment:
            if (
                isinstance(op, stim.CircuitRepeatBlock)
                or (immune_op_tag is not None and immune_op_tag in op.tag)
                or (rule := self.get_noise_rule(op)) is None
            ):
                circuit.append(op)
                continue
            effective_rule = _rule_with_immunity(
                rule, op, immune_qubits, immunize_gates=immunize_gates
            )
            if effective_rule is None:
                circuit.append(op)
                continue
            noisy_op, after = effective_rule.noisy_operation(op)
            circuit.append(noisy_op)
            noise_after_moment += after
        circuit += noise_after_moment

        moment_was_noisy = any(
            isinstance(op, stim.CircuitInstruction) and op_type(op.name) in GATE_OP_TYPES
            for op in moment
        )
        if moment_was_noisy and (
            self.idle_error or self.quadratic_idle_error or self.additional_error_waiting_for_m_or_r
        ):
            self._inplace_append_idle_errors(
                circuit=circuit,
                collapsed_qubits=collapsed_qubits,
                operation_qubits=operation_qubits,
                system_qubits=system_qubits,
                immune_qubits=immune_qubits,
                pending_idle_moments=pending_idle_moments,
            )

    def _inplace_append_idle_errors(
        self,
        *,
        circuit: stim.Circuit,
        collapsed_qubits: list[int],
        operation_qubits: list[int],
        system_qubits: frozenset[int],
        immune_qubits: frozenset[int],
        pending_idle_moments: dict[int, int],
    ) -> None:
        """Append or count idling errors from the given moment.

        This method identifies which qubits are idle during a moment and applies noise to them
        according to the noise model parameters.  The qubit categorization is precomputed by
        ``categorize_moment_qubits`` and passed in.

        Args:
            circuit: The circuit to append idle error operations to.
            collapsed_qubits: Qubits acted on by measurement / reset ops in this moment.
            operation_qubits: Qubits acted on by non-collapsing (unitary) ops in this moment.
            system_qubits: Set of all qubits in the system that can experience idle errors.
            immune_qubits: Qubits that are declared to be immune to noise.
            pending_idle_moments: Map from each qubit to its number of idle moments that have not
                yet been emitted as noise.  Pauli and quadratic idle errors are counted here, to be
                composed and emitted by ``_flush_idle_errors``; other idle noise is emitted
                immediately.
        """
        non_collapse_qubits = system_qubits - immune_qubits - set(collapsed_qubits)
        idle_qubits = sorted(non_collapse_qubits - set(operation_qubits))

        count_idle_moments = self.quadratic_idle_error is not None
        if self.idle_error is not None:
            if isinstance(self.idle_error.after, (PauliChannel, PauliChannelSequence)):
                count_idle_moments = True
            elif idle_qubits:
                self.idle_error.emit_after(circuit, idle_qubits, context="idle qubits")
        if count_idle_moments:
            for qubit in idle_qubits:
                pending_idle_moments[qubit] = pending_idle_moments.get(qubit, 0) + 1
        if self.additional_error_waiting_for_m_or_r and collapsed_qubits and non_collapse_qubits:
            self.additional_error_waiting_for_m_or_r.emit_after(
                circuit, sorted(non_collapse_qubits), context="qubits waiting for m/r"
            )

    def _flush_idle_errors(
        self,
        circuit: stim.Circuit,
        pending_idle_moments: dict[int, int],
        qubits: Iterable[int] | None = None,
    ) -> None:
        """Emit and clear the pending idle errors on the given qubits (by default, on all qubits).

        Pending idle errors must be emitted before any instruction that depends on the state of
        their qubit, as identified by ``idle_barrier_qubits``.  They may be deferred past noise and
        classically controlled Paulis, since Pauli errors commute with these up to signs that do
        not affect detectors or observables.
        """
        if not pending_idle_moments:
            return
        qubits_by_count: dict[int, list[int]] = collections.defaultdict(list)
        for qubit in sorted(pending_idle_moments.keys() if qubits is None else qubits):
            if (count := pending_idle_moments.pop(qubit, None)) is not None:
                qubits_by_count[count].append(qubit)
        idle_channel = None if self.idle_error is None else self.idle_error.after
        for count, targets in qubits_by_count.items():
            channels: list[PauliChannel | PauliChannelSequence] = []
            if isinstance(idle_channel, (PauliChannel, PauliChannelSequence)):
                channels.append(
                    idle_channel if count == 1 else repeated_idle_channel(idle_channel, count)
                )
            if self.quadratic_idle_error is not None:
                channels.append(quadratic_idle_channel(self.quadratic_idle_error, count))
            if channels:
                after = channels[0] if len(channels) == 1 else compose_idle_channels(*channels)
                NoiseRule(after=after).emit_after(circuit, targets, context="idle qubits")


class _NoisyCircuitBuilder:
    """Adds noise to a circuit moment by moment, as though its repeat blocks were flattened.

    Repeat blocks are nonetheless kept compact.  The noise of an iteration of a repeat block is
    determined by the state of this builder at the start of the iteration: the open moment, whether
    a TICK must precede the next moment, and the pending idle moments of the qubits that the repeat
    block uses (the qubits that it never uses do not affect its noise).  Once an iteration ends in
    the same state that it started in, all remaining iterations are identical to it.
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        *,
        system_qubits: frozenset[int],
        immune_qubits: frozenset[int],
        immune_op_tag: str | None,
        immunize_gates: bool,
        insert_ticks: bool,
    ) -> None:
        self.noise_model = noise_model
        self.system_qubits = system_qubits
        self.immune_qubits = immune_qubits
        self.immune_op_tag = immune_op_tag
        self.immunize_gates = immunize_gates
        self.insert_ticks = insert_ticks

        self.moment: list[stim.CircuitInstruction | stim.CircuitRepeatBlock] = []  # the open moment
        self.used_qubits: set[int] = set()  # qubits that gates in the open moment act on
        self.idle_moments: dict[int, int] = {}  # idle moments not yet emitted as noise, by qubit
        self.needs_tick = False  # whether the next moment must be preceded by a TICK

    def append(self, circuit: stim.Circuit, output: stim.Circuit) -> None:
        """Add noise to the operations of a circuit, and append completed moments to the output."""
        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                if self.immune_op_tag is not None and self.immune_op_tag in op.tag:
                    self.close_moment(output)
                    self.noise_model._flush_idle_errors(
                        output, self.idle_moments, idle_barrier_qubits([op])
                    )
                    if self.needs_tick:
                        output.append("TICK")
                    output.append(op)
                    self.needs_tick = False
                elif not _is_gate_free(op.body_copy(), self.immune_op_tag):
                    self.append_repeat_block(op, output)
                elif not _is_empty(op.body_copy()):
                    self.moment.append(op)
            elif op.name == "TICK":
                self.close_moment(output)
            else:
                for gate in (
                    split_targets_if_needed(op, frozenset({-1}), self.immune_op_tag)
                    if self.insert_ticks
                    else [op]
                ):
                    qubits = gate_qubits(gate)
                    if qubits & self.used_qubits:
                        if not self.insert_ticks:
                            raise ValueError(
                                "Qubits were operated on multiple times without a TICK in between:"
                                f"\nmultiple uses: {sorted(qubits & self.used_qubits)}"
                                f"\nmoment:\n{[*self.moment, gate]}"
                            )
                        self.close_moment(output)
                    self.used_qubits |= qubits
                    self.moment.extend(
                        split_targets_if_needed(
                            gate,
                            self.immune_qubits,
                            self.immune_op_tag,
                            force_split=self.noise_model.rule_func is not None,
                        )
                    )

    def append_repeat_block(self, block: stim.CircuitRepeatBlock, output: stim.Circuit) -> None:
        """Add noise to a repeat block, and append its completed moments to the output."""
        body = block.body_copy()
        body_qubits = idle_barrier_qubits(body)

        def get_state() -> str:
            idle_moments = sorted(
                (qubit, count) for qubit, count in self.idle_moments.items() if qubit in body_qubits
            )
            return str((self.moment, self.needs_tick, idle_moments))

        state = get_state()
        for iteration in range(block.repeat_count):
            idle_moments = dict(self.idle_moments)
            noisy_body = stim.Circuit()
            self.append(body, noisy_body)
            previous_state, state = state, get_state()
            if state != previous_state:
                output += noisy_body
                continue

            # this iteration started and ended in the same state, so it repeats until the end
            num_iterations = block.repeat_count - iteration
            if num_iterations > 1 and noisy_body:
                output.append(stim.CircuitRepeatBlock(num_iterations, noisy_body, tag=block.tag))
            else:
                output += noisy_body
            for qubit, count in self.idle_moments.items():
                if qubit not in body_qubits:
                    increment = count - idle_moments.get(qubit, 0)
                    self.idle_moments[qubit] = count + increment * (num_iterations - 1)
            return

    def close_moment(self, output: stim.Circuit) -> None:
        """Add noise to the open moment (if nonempty), and append it to the output."""
        if not self.moment:
            return
        self.noise_model._flush_idle_errors(
            output, self.idle_moments, idle_barrier_qubits(self.moment)
        )
        if self.needs_tick:
            output.append("TICK")
        self.noise_model._inplace_append_noisy_moment(
            circuit=output,
            moment=self.moment,
            system_qubits=self.system_qubits,
            immune_qubits=self.immune_qubits,
            immune_op_tag=self.immune_op_tag,
            immunize_gates=self.immunize_gates,
            pending_idle_moments=self.idle_moments,
        )
        self.moment = []
        self.used_qubits = set()
        self.needs_tick = True

    def finish(self, output: stim.Circuit) -> None:
        """Close the open moment, and emit all pending idle errors."""
        self.close_moment(output)
        self.noise_model._flush_idle_errors(output, self.idle_moments)


def _is_gate_free(circuit: stim.Circuit, immune_op_tag: str | None) -> bool:
    """Does a circuit consist of annotations (other than TICKs) and noise, at any depth?"""
    return all(
        not (immune_op_tag is not None and immune_op_tag in op.tag)
        and _is_gate_free(op.body_copy(), immune_op_tag)
        if isinstance(op, stim.CircuitRepeatBlock)
        else op.name != "TICK" and op_type(op.name) in (ANNOTATION, NOISE)
        for op in circuit
    )


def _is_empty(circuit: stim.Circuit) -> bool:
    """Is a circuit empty when flattened?"""
    return all(
        isinstance(op, stim.CircuitRepeatBlock) and _is_empty(op.body_copy()) for op in circuit
    )


####################################################################################################
# built-in noise models


class DepolarizingNoiseModel(NoiseModel):
    """Creates a near-standard circuit depolarizing noise model.

    All operations have the same error probability p:

    - One-qubit Clifford gates get one-qubit depolarization.
    - Two-qubit Clifford gates get two-qubit depolarization.
    - Measurements have their outcomes flipped, including multi-qubit measurements such as MXX and
      MPP, which get no depolarization.
    - Reset gates probabilistically reset qubits to the wrong (orthogonal) state.
    - If include_idling_error is True, every idling qubit in a given moment gets depolarized.

    Unitary Clifford gates on more qubits, including Pauli-product rotations (SPP), can also be
    depolarized by increasing the max_gate_size; otherwise, they are noiseless.

    Explicit ``I`` and ``II`` instructions are treated as idle markers: they are preserved, do not
    receive gate noise, and their target qubits receive the configured idle noise.

    Args:
        p: The error probability of every noisy operation.
        include_idling_error: Whether to depolarize idling qubits.  Default: False.
        max_gate_size: The largest number of qubits of a unitary Clifford gate that gets
            depolarized.  Default: 2.

    Raises:
        ValueError: If max_gate_size is less than 1, or p is not between 0 and 1.
    """

    def __init__(
        self, p: float, *, include_idling_error: bool = False, max_gate_size: int = 2
    ) -> None:
        """Instantiate a depolarizing noise model."""
        if max_gate_size < 1:
            raise ValueError("max_gate_size must be at least 1")
        self.p = p
        self.include_idling_error = include_idling_error
        super().__init__(
            clifford_nq_error={size: p for size in range(1, max_gate_size + 1)},
            readout_error=p,
            reset_error=p,
            idle_error=p if include_idling_error else False,
        )


class SI1000NoiseModel(NoiseModel):
    """A superconducting-inspired noise model defined in "A Fault-Tolerant Honeycomb Memory".

    This noise model is defined by a two-qubit gate infidelity p that determines all error rates:

    - One-qubit Clifford gates get one-qubit depolarization with probability ``p/10``.
    - Two-qubit Clifford gates get two-qubit depolarization with probability ``p``.
    - Measurements have their outcomes flipped with probability ``5*p``.
    - Reset gates reset qubits to the wrong (orthogonal) state with probability ``2*p``.
    - Every idling qubit in a given moment gets depolarized with probability ``p/10``.

    See https://arxiv.org/abs/2108.10457.  A qubit waiting while another qubit is measured or reset
    receives both the ordinary idling rate ``p/10`` and the additional waiting rate ``2*p``; this
    matches the cited reference implementation.

    Args:
        p: The two-qubit gate error probability, which sets all other error rates.
    """

    def __init__(self, p: float) -> None:
        """Instantiate a superconducting-inspired noise model."""
        self.p = p
        super().__init__(
            clifford_1q_error=p / 10,
            clifford_2q_error=p,
            readout_error=p * 5,
            reset_error=p * 2,
            idle_error=p / 10,
            additional_error_waiting_for_m_or_r=2 * p,
        )
