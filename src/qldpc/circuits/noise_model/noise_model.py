# SPDX-License-Identifier: Apache-2.0

"""Noise models, which add noise to circuits.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

import bisect
import collections
import dataclasses
import functools
import math
from collections.abc import Callable, Iterable, Mapping

import stim

from qldpc import _deprecations
from qldpc._utils import format_docstring

from ..common import StimCircuitLike
from ..transforms import with_remapped_qubits
from ._moments import (
    compose_idle_channels,
    get_immune_qubits,
    quadratic_idle_channel,
    repeated_idle_channel,
)
from ._operations import (
    ANNOTATION,
    CLIFFORD_1Q,
    CLIFFORD_2Q,
    CLIFFORD_PP,
    GATE_OP_TYPES,
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
from .scheduling import (
    Schedule,
    Scheduler,
    _as_stim_circuit,
    _get_operations,
    _get_segments,
    _Operation,
    _Segment,
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
    Identity gates (``I`` and ``II``) are treated as ordinary one- and two-qubit Clifford gates.

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
            fields as ``idle_error``, and is likewise combined with the other idle errors of a
            qubit until its next operation.
        rules: Dictionary mapping specific gate names to their noise rules.  Overrides the
            arity-based defaults for unitary, measurement, and reset gates.
        rule_func: Optional callback function that maps a ``stim.CircuitInstruction`` to a
            ``NoiseRule``.  The callback is evaluated before named and default rules.  Broadcast
            instructions are split first, so each callback input represents one gate
            application: one target for a one-qubit gate, two for a two-qubit gate, or one Pauli
            product for an SPP/MPP instruction.  The callback receives unitary Cliffords,
            measurements, and resets, but not annotations, pure-noise instructions, or classically
            controlled operations.  Returning ``None`` continues with named and default rule
            lookup.
        scheduler: The :class:`~qldpc.circuits.Scheduler` that places the operations of circuits in
            moments, for idle noise.  Default: ``Scheduler.asap()``, which ignores the TICKs of a
            circuit and places each operation as early as possible (and each reset as late as
            possible), with single-qubit unitary gates taking no time.

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
        scheduler: Scheduler | None = None,
    ) -> None:
        """Initializes a noise model with specified parameters."""
        self.rules = rules
        self.rule_func = rule_func
        self.scheduler = Scheduler.asap() if scheduler is None else scheduler
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
        if self.scheduler != Scheduler.asap():
            args.append(f"scheduler={self.scheduler!r}")
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

        .. note::
            MPP / SPP / SPP_DAG instructions passed to this method must contain exactly one Pauli
            product (e.g. ``MPP X0*Y1*Z2``, not ``MPP X0*Y1 Z2*X3``).  ``noisy_circuit`` splits
            instructions into gate applications before it invokes this method; hand-calling with
            an unsplit multi-product op raises ``ValueError`` from ``get_gate_aliases``.

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
        insert_ticks: bool | None = None,  # DEPRECATED: see qldpc._deprecations
    ) -> StimCircuitLike:
        """Construct a noisy version of the given circuit.

        The noisy circuit is the input circuit with noise inserted: its instructions keep their
        order, and no TICKs are added.  The noise of each gate application follows it, and precedes
        the next instruction that acts on, or depends on the state of, its qubits.  If this model
        has idle noise, its ``scheduler`` places the operations of the circuit in moments, and each
        qubit gets the noise of each of its idle periods before its next operation.  See
        :mod:`qldpc.circuits.noise_model` for the details of idle noise.

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
                An immune-tagged operation still occupies its qubits for idle noise; it does not
                suppress idle noise on unrelated qubits.
            insert_ticks: Deprecated.  To lay out a noisy circuit in moments, add noise to the
                layout of a schedule, from ``Schedule.to_circuit``.

        Returns:
            The input circuit with added noise.

        Raises:
            ValueError: If the schedule of the circuit is invalid (see ``Scheduler.validate``).
        """
        if immune_op_tag == "":
            raise ValueError("immune_op_tag must be non-empty or None")
        if immune_qubit_tag == "":
            raise ValueError("immune_qubit_tag must be non-empty or None")
        if insert_ticks is not None:  # DEPRECATED: see qldpc._deprecations
            _deprecations.warn_deprecated(
                (
                    "noisy_circuit(..., insert_ticks=True) is deprecated and will be removed in a"
                    " future release.  To lay out a noisy circuit in moments, add noise to the"
                    " layout of a schedule, as in noise_model.noisy_circuit(layout) for layout ="
                    " ticks.get_schedule(circuit).to_circuit()[0] with ticks ="
                    " Scheduler.preserve_ticks(single_qubit_gates_take_time=True), and pass"
                    " scheduler=ticks to the noise model so that its idle noise uses those moments"
                )
                if insert_ticks
                else (
                    "noisy_circuit(..., insert_ticks=False) is deprecated, has no effect, and will"
                    " be removed in a future release, since noisy circuits no longer get TICKs."
                    "  Remove the argument"
                ),
                stacklevel=2,
            )
        stim_circuit = _as_stim_circuit(circuit)
        system_qubits = frozenset(
            range(stim_circuit.num_qubits) if system_qubits is None else system_qubits
        )
        immune_qubits = frozenset(immune_qubits) | (
            get_immune_qubits(stim_circuit, immune_qubit_tag)
            if immune_qubit_tag is not None
            else frozenset()
        )

        if insert_ticks:  # DEPRECATED: see qldpc._deprecations
            from qldpc._deprecations.noise_model import get_noisy_circuit_with_ticks

            noisy_circuit = get_noisy_circuit_with_ticks(
                self,
                stim_circuit,
                system_qubits=system_qubits,
                immune_qubits=immune_qubits,
                immune_op_tag=immune_op_tag,
                immunize_gates=immunize_gates,
            )
        else:
            schedule = None
            if (
                self.idle_error
                or self.quadratic_idle_error
                or self.additional_error_waiting_for_m_or_r
            ):
                schedule = self.scheduler.get_schedule(stim_circuit)
                self.scheduler.validate(stim_circuit, schedule)
            builder = _NoisyCircuitBuilder(
                self,
                idle_qubits=system_qubits - immune_qubits,
                immune_qubits=immune_qubits,
                immune_op_tag=immune_op_tag,
                immunize_gates=immunize_gates,
            )
            noisy_circuit = builder.get_noisy_circuit(stim_circuit, schedule)

        if isinstance(circuit, stim.Circuit):
            return noisy_circuit
        output = type(circuit)()
        output += noisy_circuit
        return output

    def _get_idle_noise(self, num_idle: int, num_waiting: int) -> stim.Circuit:
        """The noise on qubit 0 of an idle run, given its numbers of idle and waiting moments.

        Pauli channels are combined into one channel, and other noise is applied once per moment.
        """
        noise = stim.Circuit()
        channels: list[PauliChannel | PauliChannelSequence] = []
        for rule, count in (
            (self.idle_error, num_idle),
            (self.additional_error_waiting_for_m_or_r, num_waiting),
        ):
            if rule is None or not count:
                continue
            if isinstance(rule.after, (PauliChannel, PauliChannelSequence)):
                channels.append(
                    rule.after if count == 1 else repeated_idle_channel(rule.after, count)
                )
            else:
                for _ in range(count):
                    rule.emit_after(noise, [0], context="idle qubits")
        if self.quadratic_idle_error is not None and num_idle:
            channels.append(quadratic_idle_channel(self.quadratic_idle_error, num_idle))
        if channels:
            channel = (
                channels[0]
                if len(channels) == 1
                else functools.reduce(compose_idle_channels, channels)
            )
            NoiseRule(after=channel).emit_after(noise, [0], context="idle qubits")
        return noise


class _NoisyCircuitBuilder:
    """Inserts noise into a circuit, keeping the noise of each qubit until the qubit is next used.

    Pending noise is kept in a pool, and emitted before the next instruction that acts on, or
    depends on the state of, its qubits, or at the end of the circuit or of a REPEAT body.  Each
    qubit also has an open idle run: the numbers of moments in which it has idled and waited for a
    measurement or reset since its last operation.  The noise of an idle run joins the pool before
    the next operation on its qubit, and idle runs continue across the boundaries of segments and
    REPEAT blocks.
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        *,
        idle_qubits: frozenset[int],
        immune_qubits: frozenset[int],
        immune_op_tag: str | None,
        immunize_gates: bool,
    ) -> None:
        self.noise_model = noise_model
        self.immune_qubits = immune_qubits
        self.immune_op_tag = immune_op_tag
        self.immunize_gates = immunize_gates

        self.pool: dict[int, tuple[frozenset[int], stim.Circuit]] = {}  # pending noise, by order
        self.pool_by_qubit: dict[int, set[int]] = collections.defaultdict(set)  # pool keys
        self.num_pooled = 0
        self.idle_runs: dict[int, tuple[int, int]] = dict.fromkeys(sorted(idle_qubits), (0, 0))
        self.idle_noise: dict[tuple[int, int], stim.Circuit] = {}  # cache, by idle run

        # the state of the open segment
        self.times: _SegmentTimes | None = None
        self.last_moments: dict[int, int] = {}  # the moment of the last operation on each qubit

    def get_noisy_circuit(self, circuit: stim.Circuit, schedule: Schedule | None) -> stim.Circuit:
        """Add noise to a circuit, given a schedule of it if the noise model has idle noise."""
        output = stim.Circuit()
        self.append(circuit, schedule, output)
        for qubit, idle_run in self.idle_runs.items():
            self.add_idle_noise(qubit, idle_run)
        self.flush(output)
        return output

    def append(
        self, circuit: stim.Circuit, schedule: Schedule | None, output: stim.Circuit
    ) -> None:
        """Add noise to the instructions of a circuit, and append them to the output."""
        for item in _get_segments(circuit, schedule):
            if isinstance(item, _Segment):
                self.append_segment(item, schedule is not None, output)
            elif self.immune_op_tag is not None and self.immune_op_tag in item[0].tag:
                self.append_immune_block(item[0], output)
            else:
                self.append_block(item[0], item[1], output)

    def append_segment(self, segment: _Segment, timed: bool, output: stim.Circuit) -> None:
        """Add noise to the instructions of a segment, and append them to the output."""
        self.times = (
            _SegmentTimes.from_segment(segment, self.noise_model.scheduler.takes_time)
            if timed
            else None
        )
        self.last_moments = {}
        start = 0
        for instruction, num_operations in zip(segment.instructions, segment.num_operations):
            operations = segment.operations[start : start + num_operations]
            moments = segment.moments[start : start + num_operations]
            start += num_operations
            if any(operation.is_gate for operation in operations):
                self.append_gate(instruction, operations, moments, output)
            elif operations and operations[0].touched_qubits:
                # an observable with Pauli targets
                qubits = operations[0].touched_qubits
                for qubit in qubits:
                    self.add_idle_noise(qubit, self.end_idle_run(qubit, moments[0]))
                self.flush(output, qubits)
                output.append(instruction)
            else:
                output.append(instruction)
        if self.times is not None:
            for qubit, (idle, wait) in self.idle_runs.items():
                more_idle, more_wait = self.times.count(qubit, self.last_moments.get(qubit, 0))
                self.idle_runs[qubit] = (idle + more_idle, wait + more_wait)

    def append_gate(
        self,
        instruction: stim.CircuitInstruction,
        operations: list[_Operation],
        moments: list[int],
        output: stim.Circuit,
    ) -> None:
        """Add noise to the applications of a gate, and append them to the output.

        The gate is split only where the noise of an application must precede a later application.
        """
        noiseless = self.immune_op_tag is not None and self.immune_op_tag in instruction.tag
        chunk: list[tuple[stim.CircuitInstruction, stim.Circuit]] = []  # (gate, noise after)
        chunk_qubits: set[int] = set()  # qubits that the chunk acts on
        noisy_qubits: set[int] = set()  # qubits with noise from the chunk
        idle_noise: list[tuple[int, tuple[int, int]]] = []  # noise to precede the chunk

        def append_chunk() -> None:
            for qubit, idle_run in idle_noise:
                self.add_idle_noise(qubit, idle_run)
            self.flush(output, chunk_qubits)
            for gate, _ in chunk:
                output.append(gate)
            for _, after in chunk:
                self.add_noise(after)

        for operation, moment in zip(operations, moments):
            qubits = operation.touched_qubits
            idle_runs = [(qubit, self.end_idle_run(qubit, moment)) for qubit in qubits]
            if any(
                qubit in chunk_qubits and (qubit in noisy_qubits or any(idle_run))
                for qubit, idle_run in idle_runs
            ):
                append_chunk()
                chunk, chunk_qubits, noisy_qubits, idle_noise = [], set(), set(), []
            idle_noise.extend(idle_runs)
            gate, after = self.get_noisy_gate(operation.instruction, noiseless)
            chunk.append((gate, after))
            chunk_qubits.update(qubits)
            noisy_qubits.update(_get_qubits(after))
        append_chunk()

    def append_block(
        self, block: stim.CircuitRepeatBlock, schedule: Schedule | None, output: stim.Circuit
    ) -> None:
        """Add noise to a REPEAT block, and append it to the output.

        An iteration of the block gets the same noise as the next, unless an idle run of a qubit
        that the block acts on starts before the block.  If the first iteration gets other noise,
        it is written out on its own, so the remaining iterations stay in a REPEAT block.
        """
        self.flush(output)
        body = block.body_copy()
        qubits = _get_touched_qubits(body)
        idle_runs = [dict(self.idle_runs)]
        iterations: list[stim.Circuit] = []
        while len(iterations) < block.repeat_count:
            iteration = stim.Circuit()
            self.append(body, schedule, iteration)
            self.flush(iteration)
            iterations.append(iteration)
            idle_runs.append(dict(self.idle_runs))
            if all(
                idle_runs[-1][qubit] == idle_runs[-2][qubit]
                for qubit in qubits & self.idle_runs.keys()
            ):
                break

        # the remaining iterations repeat the last, and lengthen the idle runs of other qubits
        num_remaining = block.repeat_count - len(iterations)
        for qubit, (idle, wait) in self.idle_runs.items():
            if qubit not in qubits:
                last_idle, last_wait = idle_runs[-2][qubit]
                self.idle_runs[qubit] = (
                    idle + (idle - last_idle) * num_remaining,
                    wait + (wait - last_wait) * num_remaining,
                )
        for iteration in iterations[:-1]:
            output += iteration
        output.append(stim.CircuitRepeatBlock(num_remaining + 1, iterations[-1], tag=block.tag))

    def append_immune_block(self, block: stim.CircuitRepeatBlock, output: stim.Circuit) -> None:
        """Append a noiseless REPEAT block, which ends the idle runs of the qubits that it acts on.

        A noiseless REPEAT block takes no time.
        """
        qubits = _get_touched_qubits(block.body_copy())
        for qubit in sorted(qubits & self.idle_runs.keys()):
            self.add_idle_noise(qubit, self.idle_runs[qubit])
            self.idle_runs[qubit] = (0, 0)
        self.flush(output, qubits)
        output.append(block)

    def get_noisy_gate(
        self, gate: stim.CircuitInstruction, noiseless: bool
    ) -> tuple[stim.CircuitInstruction, stim.Circuit]:
        """A gate application with noise, and the noise that follows it."""
        rule = None if noiseless else self.noise_model.get_noise_rule(gate)
        if rule is not None:
            rule = _rule_with_immunity(
                rule, gate, self.immune_qubits, immunize_gates=self.immunize_gates
            )
        if rule is None:
            return gate, stim.Circuit()
        return rule.noisy_operation(gate)

    def end_idle_run(self, qubit: int, moment: int) -> tuple[int, int]:
        """End the idle run of a qubit at an operation in a moment of the open segment."""
        if self.times is None or qubit not in self.idle_runs:
            return 0, 0
        start = self.last_moments.get(qubit)
        self.last_moments[qubit] = moment
        if start is not None:
            return self.times.count(qubit, start, moment)
        idle, wait = self.idle_runs[qubit]
        more_idle, more_wait = self.times.count(qubit, 0, moment)
        self.idle_runs[qubit] = (0, 0)
        return idle + more_idle, wait + more_wait

    def add_idle_noise(self, qubit: int, idle_run: tuple[int, int]) -> None:
        """Add the noise of an idle run of a qubit to the pool."""
        if not any(idle_run):
            return
        if idle_run not in self.idle_noise:
            self.idle_noise[idle_run] = self.noise_model._get_idle_noise(*idle_run)
        if self.idle_noise[idle_run]:
            self.add_noise(with_remapped_qubits(self.idle_noise[idle_run], [qubit]))

    def add_noise(self, noise: stim.Circuit) -> None:
        """Add noise to the pool."""
        if not noise:
            return
        qubits = _get_qubits(noise)
        self.pool[self.num_pooled] = (qubits, noise)
        for qubit in qubits:
            self.pool_by_qubit[qubit].add(self.num_pooled)
        self.num_pooled += 1

    def flush(self, output: stim.Circuit, qubits: Iterable[int] | None = None) -> None:
        """Emit the pending noise on some qubits (by default, on all qubits), in order."""
        keys = (
            list(self.pool)
            if qubits is None
            else sorted({key for qubit in qubits for key in self.pool_by_qubit.get(qubit, ())})
        )
        for key in keys:
            noise_qubits, noise = self.pool.pop(key)
            for qubit in noise_qubits:
                self.pool_by_qubit[qubit].discard(key)
            output += noise


def _get_qubits(circuit: stim.Circuit) -> frozenset[int]:
    """The qubits that the instructions of a circuit (without REPEAT blocks) act on."""
    return frozenset(
        target.qubit_value
        for instruction in circuit
        for target in instruction.targets_copy()
        if target.qubit_value is not None
    )


def _get_touched_qubits(circuit: stim.Circuit) -> frozenset[int]:
    """The qubits whose state some instruction of a circuit depends on, at any depth."""
    qubits: set[int] = set()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            qubits.update(_get_touched_qubits(instruction.body_copy()))
        else:
            for operation in _get_operations([instruction]):
                qubits.update(operation.touched_qubits)
    return frozenset(qubits)


####################################################################################################
# idle times


@dataclasses.dataclass
class _SegmentTimes:
    """The moments of a segment in which qubits idle or wait for measurements and resets.

    A qubit idles in each moment that contains a gate, unless an operation that takes time acts on
    it, and waits in each moment that contains a measurement or reset, unless one acts on it.
    """

    timed_moments: list[int]  # the moments that contain gates
    occupied_moments: dict[int, list[int]]  # by qubit, the moments with operations that take time
    collapsing_moments: list[int]  # the moments that contain measurements or resets
    collapsed_moments: dict[int, list[int]]  # by qubit, the moments that measure or reset it

    @staticmethod
    def from_segment(
        segment: _Segment, takes_time: Callable[[stim.CircuitInstruction], bool]
    ) -> _SegmentTimes:
        timed: set[int] = set()
        occupied: dict[int, set[int]] = {}
        collapsing: set[int] = set()
        collapsed: dict[int, set[int]] = {}
        for operation, moment in zip(segment.operations, segment.moments):
            if not operation.is_gate:
                continue
            timed.add(moment)
            if takes_time(operation.instruction):
                for qubit in operation.qubits:
                    occupied.setdefault(qubit, set()).add(moment)
            if operation.is_collapsing:
                collapsing.add(moment)
                for qubit in operation.qubits:
                    collapsed.setdefault(qubit, set()).add(moment)
        return _SegmentTimes(
            sorted(timed),
            {qubit: sorted(moments) for qubit, moments in occupied.items()},
            sorted(collapsing),
            {qubit: sorted(moments) for qubit, moments in collapsed.items()},
        )

    def count(self, qubit: int, start: int, stop: int | None = None) -> tuple[int, int]:
        """The numbers of moments in which a qubit idles and waits, from ``start`` to ``stop``.

        The moments ``start <= moment < stop`` are counted, where ``stop=None`` counts to the end
        of the segment.
        """
        return (
            _count(self.timed_moments, start, stop)
            - _count(self.occupied_moments.get(qubit, []), start, stop),
            _count(self.collapsing_moments, start, stop)
            - _count(self.collapsed_moments.get(qubit, []), start, stop),
        )


def _count(moments: list[int], start: int, stop: int | None) -> int:
    """The number of sorted moments from ``start`` up to (but excluding) ``stop``."""
    end = len(moments) if stop is None else bisect.bisect_left(moments, stop)
    return max(0, end - bisect.bisect_left(moments, start))


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

    Args:
        p: The error probability of every noisy operation.
        include_idling_error: Whether to depolarize idling qubits.  Default: False.
        max_gate_size: The largest number of qubits of a unitary Clifford gate that gets
            depolarized.  Default: 2.
        scheduler: The :class:`~qldpc.circuits.Scheduler` that places the operations of circuits in
            moments, for idle noise.  Default: ``Scheduler.asap()``, as for
            :class:`~qldpc.circuits.NoiseModel`.

    Raises:
        ValueError: If max_gate_size is less than 1, or p is not between 0 and 1.
    """

    def __init__(
        self,
        p: float,
        *,
        include_idling_error: bool = False,
        max_gate_size: int = 2,
        scheduler: Scheduler | None = None,
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
            scheduler=scheduler,
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
        scheduler: The :class:`~qldpc.circuits.Scheduler` that places the operations of circuits in
            moments, for idle noise.  Default: ``Scheduler.preserve_ticks(
            single_qubit_gates_take_time=True)``, which keeps the moments of a circuit, as in the
            cited reference.
    """

    def __init__(self, p: float, *, scheduler: Scheduler | None = None) -> None:
        """Instantiate a superconducting-inspired noise model."""
        self.p = p
        super().__init__(
            clifford_1q_error=p / 10,
            clifford_2q_error=p,
            readout_error=p * 5,
            reset_error=p * 2,
            idle_error=p / 10,
            additional_error_waiting_for_m_or_r=2 * p,
            scheduler=(
                Scheduler.preserve_ticks(single_qubit_gates_take_time=True)
                if scheduler is None
                else scheduler
            ),
        )
