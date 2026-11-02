# SPDX-License-Identifier: Apache-2.0

"""Noise models, which add noise to circuits.

See :mod:`qldpc.circuits.noise_model` for attribution and license information.
"""

from __future__ import annotations

import bisect
import collections
import functools
import math
from collections.abc import Callable, Iterable, Mapping, Sequence

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
        # list the fields that are set, in the order of the arguments of the constructor
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
        # annotations, noise channels, and classically controlled gates get no noise
        this_op_type = op_type(op.name)
        if this_op_type in (ANNOTATION, NOISE) or involves_classical_bits(op):
            return None

        # the number of qubits in the Pauli product of an MPP, SPP, or SPP_DAG instruction
        product_arity = (
            sum(1 for target in op.targets_copy() if not target.is_combiner)
            if this_op_type in (CLIFFORD_PP, JUST_MEASURE_PP)
            else 0
        )

        # a rule from rule_func comes first
        if self.rule_func is not None and this_op_type in GATE_OP_TYPES:
            rule = self.rule_func(op)
            if rule is not None:
                _validate_custom_rule(rule, op)
                return rule

        # then a named rule, by the most specific name of the gate; a rule for every Pauli product
        # (named MPP, SPP, or SPP_DAG) must fit this one
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

        # then the default rule for unitary Cliffords on this number of qubits
        if this_op_type in (CLIFFORD_1Q, CLIFFORD_2Q, CLIFFORD_PP):
            if this_op_type == CLIFFORD_1Q:
                num_qubits = 1
            elif this_op_type == CLIFFORD_2Q:
                num_qubits = 2
            else:
                num_qubits = product_arity
            if num_qubits in self.clifford_nq_error:
                return self.clifford_nq_error[num_qubits]

        # then the default readout and reset errors
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
        has idle noise, its ``scheduler`` places the operations of the circuit in moments, each of
        which lasts one unit of time, and each qubit gets the noise of each of its idle periods
        before its next operation.  See :mod:`qldpc.circuits.noise_model` for the details of idle
        noise.

        Args:
            circuit: The circuit to apply noise to.
            system_qubits: All qubits that are used by the circuit or are otherwise allowed to
                accumulate idling errors.  Defaults to set(range(circuit.num_qubits)).
            immune_qubits: Qubits that are declared to be immune to noise.  Defaults to none.
            immune_op_tag: If an operation or REPEAT block contains this string in its tag, it is
                noiseless, but still takes its time.  Set to None to disable tagged operation
                immunity.  Default: "{DEFAULT_IMMUNE_OP_TAG}".
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
        # check the arguments
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
        # find the qubits that may idle, and those that are immune to noise
        stim_circuit = _as_stim_circuit(circuit)
        system_qubits = frozenset(
            range(stim_circuit.num_qubits) if system_qubits is None else system_qubits
        )
        immune_qubits = frozenset(immune_qubits) | (
            get_immune_qubits(stim_circuit, immune_qubit_tag)
            if immune_qubit_tag is not None
            else frozenset()
        )

        # with insert_ticks=True, add noise to the circuit laid out in the moments of its TICKs
        scheduler = self.scheduler
        if insert_ticks:  # DEPRECATED: see qldpc._deprecations
            scheduler = Scheduler.preserve_ticks(single_qubit_gates_take_time=True)
            stim_circuit = scheduler.get_schedule(stim_circuit).to_circuit()[0]

        # schedule the circuit if this model has idle noise, and add noise
        schedule = None
        if self.idle_error or self.quadratic_idle_error or self.additional_error_waiting_for_m_or_r:
            schedule = scheduler.get_schedule(stim_circuit)
            scheduler.validate(stim_circuit, schedule)
        builder = _NoisyCircuitBuilder(
            self,
            takes_time=scheduler.takes_time,
            idle_qubits=system_qubits - immune_qubits,
            immune_qubits=immune_qubits,
            immune_op_tag=immune_op_tag,
            immunize_gates=immunize_gates,
            timed=schedule is not None,
        )
        noisy_circuit = builder.get_noisy_circuit(stim_circuit, schedule)

        # return a circuit of the type of the input
        if isinstance(circuit, stim.Circuit):
            return noisy_circuit
        output = type(circuit)()
        output += noisy_circuit
        return output

    def _get_idle_noise(self, num_idle: int, num_waiting: int) -> stim.Circuit:
        """The noise on qubit 0 of an idle run, given its numbers of idle and waiting moments.

        Pauli channels are combined into one channel, and other noise is applied once per moment.
        """
        # collect the Pauli channels of idle and waiting moments, and emit other noise per moment
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

        # combine the Pauli channels into one
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
    depends on the state of, its qubits, or at the end of the circuit or of a REPEAT body.

    With a schedule, the builder also follows the idle run of each qubit: the time of its last
    operation, the number of moments before that time that contain a measurement or reset, and
    whether that operation took time or measured or reset the qubit.  Times are counted in moments
    of the schedule.  The noise of an idle run joins the pool before the next operation on its
    qubit.
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        *,
        takes_time: Callable[[stim.CircuitInstruction], bool],
        idle_qubits: frozenset[int],
        immune_qubits: frozenset[int],
        immune_op_tag: str | None,
        immunize_gates: bool,
        timed: bool,
    ) -> None:
        self.noise_model = noise_model
        self.takes_time = takes_time
        self.immune_qubits = immune_qubits
        self.immune_op_tag = immune_op_tag
        self.immunize_gates = immunize_gates
        self.timed = timed
        self.noiseless = False  # whether the current instructions are in a noiseless block

        self.pool: dict[int, tuple[Iterable[int], stim.Circuit]] = {}  # pending noise, in order
        self.pool_by_qubit: dict[int, set[int]] = collections.defaultdict(set)  # pool keys
        self.num_pooled = 0
        self.idle_noise: dict[tuple[int, int], stim.Circuit] = {}  # cache, by idle run

        # idle runs, by qubit: [time, measurement moments before it, timed, collapsing]
        self.runs: dict[int, list[int]] = {qubit: [0, 0, 0, 0] for qubit in sorted(idle_qubits)}
        self.time = 0  # the moment in which the current segment starts
        self.num_collapsing = 0  # the number of moments before it with measurements or resets
        self.collapsing_moments: list[int] = []  # those of the current segment

    def get_noisy_circuit(self, circuit: stim.Circuit, schedule: Schedule | None) -> stim.Circuit:
        """Add noise to a circuit, given a schedule of it if the noise model has idle noise."""
        output = stim.Circuit()
        self.append(circuit, schedule, output)
        if self.timed:
            for qubit in self.runs:
                self.add_idle_noise(qubit, self.end_idle_run(qubit, self.time, False, False))
        self.flush(output)
        return output

    def append(
        self, circuit: stim.Circuit, schedule: Schedule | None, output: stim.Circuit
    ) -> None:
        """Add noise to a circuit, and append it to the output."""
        for item in _get_segments(circuit, schedule):
            if isinstance(item, _Segment):
                self.append_segment(item, output)
            else:
                self.append_block(item[0], item[1], output)

    def append_segment(self, segment: _Segment, output: stim.Circuit) -> None:
        """Add noise to the instructions of a segment, and append them to the output."""
        # find the moments of the segment in which qubits wait for measurements or resets
        if self.timed:
            self.collapsing_moments = sorted(
                {
                    self.time + moment
                    for operation, moment in zip(segment.operations, segment.moments)
                    if operation.is_collapsing
                }
            )
        # add noise to each gate, and emit the noise before each observable with Pauli targets
        start = 0
        for instruction, num_operations in zip(segment.instructions, segment.num_operations):
            operations = segment.operations[start : start + num_operations]
            moments = segment.moments[start : start + num_operations]
            start += num_operations
            times = [self.time + moment for moment in moments] if self.timed else None
            if operations and operations[0].is_gate:
                self.append_gate(instruction, operations, times, output)
            elif operations and operations[0].touched_qubits:
                # an observable with Pauli targets
                qubits = operations[0].touched_qubits
                if times is not None:
                    for qubit in qubits:
                        self.add_idle_noise(qubit, self.end_idle_run(qubit, times[0], False, False))
                self.flush(output, qubits)
                output.append(instruction)
            else:
                output.append(instruction)

        # the next segment starts after this one
        if self.timed:
            self.time += segment.duration
            self.num_collapsing += len(self.collapsing_moments)
            self.collapsing_moments = []

    def append_gate(
        self,
        instruction: stim.CircuitInstruction,
        operations: list[_Operation],
        times: list[int] | None,
        output: stim.Circuit,
    ) -> None:
        """Add noise to the applications of a gate, and append them to the output.

        The gate is split only where the noise of an application must precede a later application.
        A noise rule is applied once to consecutive applications that share it.
        """
        # the applications are appended in chunks, which consecutive applications join unless they
        # must follow the noise of an application in the chunk
        rules = self.get_rules(instruction, operations)
        chunk: list[int] = []  # the indices of the applications in the chunk
        chunk_qubits: set[int] = set()  # qubits that the chunk acts on
        noisy_qubits: set[int] = set()  # qubits with noise from the chunk
        idle_noise: list[tuple[int, tuple[int, int]]] = []  # idle noise to precede the chunk

        def append_chunk() -> None:
            # emit the noise that precedes the chunk
            for qubit, idle_run in idle_noise:
                self.add_idle_noise(qubit, idle_run)
            self.flush(output, chunk_qubits)

            # append consecutive applications with the same rule as one instruction, and pool their
            # noise
            groups: list[tuple[NoiseRule | None, list[int]]] = []
            for index in chunk:
                if groups and groups[-1][0] is rules[index]:
                    groups[-1][1].append(index)
                else:
                    groups.append((rules[index], [index]))
            for rule, indices in groups:
                gate = _join_operations([operations[index] for index in indices])
                if rule is None:
                    output.append(gate)
                    continue
                noisy_gate, after = rule.noisy_operation(gate)
                output.append(noisy_gate)
                qubits = {qubit for index in indices for qubit in operations[index].qubits}
                self.add_noise(after, qubits)

        for index, operation in enumerate(operations):
            # end the idle runs of the qubits of the application
            qubits = operation.touched_qubits
            if times is None:
                idle_runs = []
            else:
                timed = operation.is_gate and self.takes_time(operation.instruction)
                run_qubits = operation.qubits if timed else qubits
                idle_runs = [
                    (qubit, self.end_idle_run(qubit, times[index], timed, operation.is_collapsing))
                    for qubit in run_qubits
                ]

            # start a new chunk if the application must follow noise on a qubit of the chunk
            if not chunk_qubits.isdisjoint(qubits) and (
                not noisy_qubits.isdisjoint(qubits)
                or any(any(idle_run) for qubit, idle_run in idle_runs if qubit in chunk_qubits)
            ):
                append_chunk()
                chunk, chunk_qubits, noisy_qubits, idle_noise = [], set(), set(), []
            idle_noise.extend(idle_runs)
            chunk.append(index)
            chunk_qubits.update(qubits)
            if rules[index] is not None:
                noisy_qubits.update(qubits)
        append_chunk()

    def get_rules(
        self, instruction: stim.CircuitInstruction, operations: list[_Operation]
    ) -> list[NoiseRule | None]:
        """The noise rule of each application of a gate, after accounting for immune qubits."""
        # noiseless gates get no rules
        if self.noiseless or (
            self.immune_op_tag is not None and self.immune_op_tag in instruction.tag
        ):
            return [None] * len(operations)

        # find the rule of one application where it is the same for all, or else of each
        if (
            self.noise_model.rule_func is None
            and op_type(instruction.name) not in (CLIFFORD_PP, JUST_MEASURE_PP)
            and not involves_classical_bits(instruction)
        ):
            # every application of this gate has the same rule
            rule = self.noise_model.get_noise_rule(operations[0].instruction)
            if rule is None or not self.immune_qubits:
                return [rule] * len(operations)
            return [
                _rule_with_immunity(
                    rule,
                    operation.instruction,
                    self.immune_qubits,
                    immunize_gates=self.immunize_gates,
                )
                for operation in operations
            ]
        rules: list[NoiseRule | None] = []
        for operation in operations:
            rule = self.noise_model.get_noise_rule(operation.instruction)
            if rule is not None:
                rule = _rule_with_immunity(
                    rule,
                    operation.instruction,
                    self.immune_qubits,
                    immunize_gates=self.immunize_gates,
                )
            rules.append(rule)
        return rules

    def append_block(
        self, block: stim.CircuitRepeatBlock, schedule: Schedule | None, output: stim.Circuit
    ) -> None:
        """Add noise to a REPEAT block, and append it to the output.

        The first iteration of the block starts from the idle runs before the block, and every later
        iteration starts from the idle runs that the iteration before it leaves, which are the same
        for every iteration.  So the noise of the second iteration is that of every later one: the
        first two iterations get noise, and the remaining iterations repeat the second, in one
        REPEAT block with it.

        A REPEAT block tagged with ``immune_op_tag`` is noiseless: it is appended unchanged, the
        qubits that it acts on get no noise while it runs, and other qubits idle as usual.
        """
        # emit the pending noise, and find the qubits whose idle runs the block ends
        self.flush(output)
        noiseless = self.immune_op_tag is not None and self.immune_op_tag in block.tag
        body = block.body_copy()
        qubits = sorted(_get_run_qubits(body, self.takes_time) & self.runs.keys())

        # a noiseless block ends the idle runs of its qubits before it starts
        was_noiseless = self.noiseless
        if noiseless:
            for qubit in qubits:
                self.add_idle_noise(qubit, self.end_idle_run(qubit, self.time, False, False))
            self.flush(output)
            self.noiseless = True

        # add noise to the first two iterations
        iterations: list[stim.Circuit] = []
        for _ in range(min(2, block.repeat_count)):
            start = (self.time, self.num_collapsing)
            iteration = stim.Circuit()
            self.append(body, schedule, iteration)
            self.flush(iteration)
            iterations.append(iteration)

        # move time and idle runs on by the remaining iterations, which repeat the last one
        num_remaining = block.repeat_count - len(iterations)
        duration, num_collapsing = self.time - start[0], self.num_collapsing - start[1]
        self.time += duration * num_remaining
        self.num_collapsing += num_collapsing * num_remaining
        for qubit in qubits:
            self.runs[qubit][0] += duration * num_remaining
            self.runs[qubit][1] += num_collapsing * num_remaining

        # a noiseless block is appended unchanged, and the idle runs of its qubits start at its end
        if noiseless:
            self.noiseless = was_noiseless
            for qubit in qubits:
                self.runs[qubit] = [self.time, self.num_collapsing, 0, 0]
            output.append(block)
            return

        # write out the first iteration if its noise differs from that of the others
        first, last = iterations[0], iterations[-1]
        if first != last:
            output += first
        num_repeats = block.repeat_count - (first != last)
        output.append(stim.CircuitRepeatBlock(num_repeats, last, tag=block.tag))

    def count_collapsing(self, moment: int) -> int:
        """The number of moments before a moment of the current segment that measure or reset."""
        return self.num_collapsing + bisect.bisect_left(self.collapsing_moments, moment)

    def end_idle_run(
        self, qubit: int, moment: int, timed: bool, collapsing: bool
    ) -> tuple[int, int]:
        """End the idle run of a qubit at an operation, and start a new one.

        Returns:
            The numbers of moments of the run in which the qubit idles, and in which it waits while
            another qubit is measured or reset.
        """
        run = self.runs.get(qubit)
        if run is None:
            return 0, 0
        if moment <= run[0]:
            # another operation in the moment of the last operation
            run[2] |= timed
            run[3] |= collapsing
            return 0, 0

        # the run lasts from the moment of the last operation, or from the next moment if that
        # operation took time, until this moment
        num_collapsing = self.count_collapsing(moment)
        idle_run = moment - run[0] - run[2], num_collapsing - run[1] - run[3]
        run[:] = [moment, num_collapsing, timed, collapsing]
        return idle_run

    def add_idle_noise(self, qubit: int, idle_run: tuple[int, int]) -> None:
        """Add the noise of an idle run of a qubit to the pool."""
        if not any(idle_run):
            return
        if idle_run not in self.idle_noise:
            self.idle_noise[idle_run] = self.noise_model._get_idle_noise(*idle_run)
        if self.idle_noise[idle_run]:
            self.add_noise(with_remapped_qubits(self.idle_noise[idle_run], [qubit]), (qubit,))

    def add_noise(self, noise: stim.Circuit, qubits: Iterable[int]) -> None:
        """Add noise on some qubits to the pool."""
        if not noise:
            return
        self.pool[self.num_pooled] = (qubits, noise)
        for qubit in qubits:
            self.pool_by_qubit[qubit].add(self.num_pooled)
        self.num_pooled += 1

    def flush(self, output: stim.Circuit, qubits: Iterable[int] | None = None) -> None:
        """Emit the pending noise on some qubits (by default, on all qubits), in order."""
        if qubits is None:
            keys: Iterable[int] = list(self.pool)
        else:
            keys = sorted({key for qubit in qubits for key in self.pool_by_qubit.get(qubit, ())})
        for key in keys:
            noise_qubits, noise = self.pool.pop(key)
            for qubit in noise_qubits:
                self.pool_by_qubit[qubit].discard(key)
            output += noise


def _join_operations(operations: Sequence[_Operation]) -> stim.CircuitInstruction:
    """Join applications of one gate into one instruction."""
    if len(operations) == 1:
        return operations[0].instruction
    first = operations[0].instruction
    targets = [
        target for operation in operations for target in operation.instruction.targets_copy()
    ]
    return stim.CircuitInstruction(first.name, targets, first.gate_args_copy(), tag=first.tag)


def _get_run_qubits(
    circuit: stim.Circuit, takes_time: Callable[[stim.CircuitInstruction], bool]
) -> frozenset[int]:
    """The qubits whose idle runs a circuit ends, which it touches or occupies at any depth."""
    qubits: set[int] = set()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            qubits.update(_get_run_qubits(instruction.body_copy(), takes_time))
        else:
            for operation in _get_operations([instruction]):
                qubits.update(operation.touched_qubits)
                if operation.is_gate and takes_time(operation.instruction):
                    qubits.update(operation.qubits)
    return frozenset(qubits)


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
