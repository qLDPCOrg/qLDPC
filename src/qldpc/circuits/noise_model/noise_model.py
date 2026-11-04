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
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence

import stim

from qldpc import _deprecations
from qldpc._utils import format_docstring

from ..common import (
    DEFAULT_IMMUNE_OP_TAG,
    DEFAULT_IMMUNE_QUBIT_TAG,
    StimCircuitLike,
    _get_immune_qubits,
)
from ..transforms import with_remapped_qubits
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
    MEASURE_OR_RESET_OPS,
    NOISE,
    canonical_rule_key,
    get_gate_aliases,
    involves_classical_bits,
    known_gate_shape,
    op_type,
)
from .channels import (
    _ABSOLUTE_ERROR_TOLERANCE,
    PauliChannel,
    PauliChannelSequence,
    _compose_idle_channels,
    _quadratic_idle_channel,
    _repeated_idle_channel,
)
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
    ASAPScheduler,
    Schedule,
    Scheduler,
    TickPreservingScheduler,
    _as_stim_circuit,
    _get_instruction_operations,
    _get_operations,
    _get_segments,
    _Operation,
    _Segment,
)

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
    # Read the circuit's instructions into a stim.Circuit body (the repeat block requires a
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
    gate errors, readout errors, reset errors, and idling errors.  Classically controlled gates get
    no noise.  Identity gates (``I`` and ``II``) are treated as ordinary one- and two-qubit Clifford
    gates.

    Idle errors apply per moment: a unit of time in the schedule that ``scheduler`` makes of a
    circuit by placing its operations, such as single gate applications, in moments (see
    :class:`~qldpc.circuits.Schedule`).  An idle run of a qubit lasts from the moment of its last
    gate, or from the start of the circuit, up to but not including the moment of its next gate, or
    to the end of the circuit (see :class:`~qldpc.circuits.Schedule` for REPEAT blocks).  Every gate
    on the qubit ends its idle run, even one that takes no time (see
    :meth:`~qldpc.circuits.Scheduler.takes_time`), except a classically controlled gate that takes
    no time.  The qubit idles in each moment of the run, except in the first if a gate on the qubit
    takes time in that moment.

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
        clifford_nq_error: Optional mapping from a qubit count ``k`` to the noise applied after each
            ``k``-qubit unitary Clifford gate, similarly to ``clifford_?q_error``.  Specifying both
            ``clifford_nq_error[1]`` and ``clifford_1q_error`` raises an ambiguity error; likewise
            with ``clifford_nq_error[2]`` and ``clifford_2q_error``.  Each Pauli product of an SPP
            or SPP_DAG gate gets the noise for its number of qubits, and a unitary Clifford gate
            whose number of qubits has no entry gets no default noise.
        idle_error: Noise rule or depolarization probability applied to each idling qubit in any
            given moment.  If a NoiseRule is provided, its `after` channels are appended to the
            idle qubits (its readout_error/reset_error fields are ignored).  Consecutive
            Pauli-channel idle errors on a qubit are composed into one equivalent error; custom
            Stim noise fragments are applied once per idle moment.
        quadratic_idle_error: Error on each idling qubit in one moment, given as a float ``p`` (a
            uniform channel of total error probability ``p``), a ``Mapping[str, float]`` such as
            ``{"Z": p}``, or a one-qubit ``PauliChannel``, with total probability at most 1/2.  Over
            an idle run with ``k`` moments in which the qubit idles, this error has total
            probability ``(1 - (1 - 2 * p)**(k**2)) / 2``, split among Paulis as in one moment, and
            is combined with the ``idle_error`` of the same moments.  An idle run ends at any gate
            on the qubit, even one that takes no time, other than a classically controlled gate that
            takes no time.  See :mod:`qldpc.circuits.noise_model` for the physical model and
            assumptions of this error.
        additional_error_waiting_for_m_or_r: Additional noise rule or depolarization probability
            applied to every qubit in each moment in which another qubit, but not this one, is
            measured or reset, including qubits undergoing a unitary gate.  Accepts the same
            NoiseRule fields as ``idle_error``, and is combined with the other idle errors of a
            qubit.
        rules: Dictionary mapping specific gate names to their noise rules.  Overrides the
            arity-based defaults for unitary, measurement, and reset gates.
        rule_func: Optional callback function that maps a ``stim.CircuitInstruction`` to a
            ``NoiseRule``.  The callback is evaluated before named and default rules.  Broadcast
            instructions are split first, so each callback input represents one gate application:
            one target for a one-qubit gate, two for a two-qubit gate, or one Pauli product for an
            SPP/MPP instruction.  The callback receives unitary Cliffords, measurements, and resets,
            but not annotations, pure-noise instructions, or classically controlled gates.
            Returning ``None`` continues with named and default rule lookup.
        scheduler: The :class:`~qldpc.circuits.Scheduler` that places the operations of circuits in
            moments, for idle noise.  Default: ``ASAPScheduler()``.

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
        self.scheduler = ASAPScheduler() if scheduler is None else scheduler
        if rules is not None:
            normalized_rules: dict[str, NoiseRule] = {}
            # Validate rules with fixed shapes now: fixed-arity stim gates and basis-suffixed rule
            # keys (``"MXYZ"`` for MPP X*Y*Z, ``"SXY_DAG"`` for SPP_DAG X*Y).  Bare Pauli-product
            # names (``"MPP"``, ``"SPP"``, ``"SPP_DAG"``) have variable arity, so validate them
            # against each concrete product when the rule is selected.
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
        # After checking the arities of idle rules, drop those that add no error.
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

        When ``rule_func`` is ``None`` and the scheduler has an eval-able repr, as the built-in
        schedulers do, this is an eval-able ``NoiseModel(...)`` call.  A ``rule_func`` cannot be
        reproduced faithfully, so it is instead described by its qualified name and module (falling
        back to its type name for callables without a ``__qualname__``, e.g. a callable object or
        ``functools.partial``), which makes the result non-eval-able.
        """
        # list the fields that are set, in a fixed order that reads like a constructor call
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
        if self.scheduler != ASAPScheduler():
            args.append(f"scheduler={self.scheduler!r}")
        return f"NoiseModel({', '.join(args)})"

    def __repr__(self) -> str:
        if self.rule_func is None:
            return str(self)
        return super().__repr__()

    def get_noise_rule(self, op: stim.CircuitInstruction) -> NoiseRule | None:
        """Determines the noise rule to apply to a specific instruction.

        Noise rules are consulted in the following order of precedence:

        1. ``rule_func`` (stim.CircuitInstruction -> NoiseRule factory).
        2. ``rules`` (name-based NoiseRules).
        3. ``clifford_nq_error`` (arity-based NoiseRules for unitary Cliffords).
        4. ``readout_error`` and/or ``reset_error`` (per-gate defaults for measurement/reset ops).

        .. note::
            ``noisy_circuit`` passes this method one gate application at a time, so an MPP, SPP, or
            SPP_DAG instruction has one Pauli product (e.g. ``MPP X0*Y1*Z2``, not ``MPP X0*Y1
            Z2*X3``).  Given several products, this method raises ``ValueError`` if it looks up a
            named rule in ``rules``, and otherwise treats them as one product.

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
        """Construct a noisy version of the given circuit by inserting noise into it.

        The noise of each gate application, and the noise of each idle run of a qubit if this model
        has idle noise (see :class:`NoiseModel`), is placed before the next gate application that
        depends on the state of its qubits or observable with a Pauli target on them, or else at the
        end of the circuit (see :class:`~qldpc.circuits.Schedule` for REPEAT blocks).

        Args:
            circuit: The circuit to apply noise to.
            system_qubits: All qubits that are used by the circuit or are otherwise allowed to
                accumulate idling errors.  Defaults to set(range(circuit.num_qubits)).
            immune_qubits: Qubits that are declared to be immune to noise.  Defaults to none.
            immune_op_tag: If an instruction or REPEAT block contains this string in its tag, it is
                noiseless.  Set to None to disable tagged instruction immunity.  Default:
                "{DEFAULT_IMMUNE_OP_TAG}".
            immune_qubit_tag: If a QUBIT_COORDS annotation contains this string in its tag, its
                target qubits are immune to noise.  Set to None to disable annotated immunity.
                Default: "{DEFAULT_IMMUNE_QUBIT_TAG}".
            immunize_gates: If True (the default), a gate that touches an immune qubit is treated as
                noiseless.  Otherwise, its Pauli noise is conditioned on the absence of errors on
                noise-immune qubits, keeping only strings that act as ``I`` on every immune qubit.
            insert_ticks: Deprecated.  To lay out a noisy circuit in moments, add noise to the
                layout of a schedule, from ``Schedule.to_circuit``.

        Returns:
            The input circuit with added noise.

        Raises:
            ValueError: If the schedule of the circuit is invalid (see ``Scheduler.validate``).
        """
        # check the immunity tags, and handle the deprecated insert_ticks option
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
                    " TickPreservingScheduler(single_qubit_gates_take_time=True), and pass"
                    " scheduler=ticks to the noise model so that its idle noise uses those moments"
                )
                if insert_ticks
                else (
                    "noisy_circuit(..., insert_ticks=False) is deprecated, has no effect, and will"
                    " be removed in a future release, since noisy_circuit keeps the TICKs of its"
                    " input and adds only noise.  Remove the argument"
                ),
                stacklevel=2,
            )
        # find the qubits that may idle, and those that are immune to noise
        stim_circuit = _as_stim_circuit(circuit)
        system_qubits = frozenset(
            range(stim_circuit.num_qubits) if system_qubits is None else system_qubits
        )
        immune_qubits = frozenset(immune_qubits) | _get_immune_qubits(
            stim_circuit, immune_qubit_tag
        )

        # for the deprecated insert_ticks=True, lay out the circuit with a TickPreservingScheduler
        # before adding noise
        scheduler = self.scheduler
        if insert_ticks:  # DEPRECATED: see qldpc._deprecations
            scheduler = TickPreservingScheduler(single_qubit_gates_take_time=True)
            stim_circuit = scheduler.get_schedule(stim_circuit).to_circuit()[0]

        # quadratic idle errors are split where REPEAT blocks and their iterations start and end
        if self.quadratic_idle_error is not None and any(
            isinstance(instruction, stim.CircuitRepeatBlock)
            and (immune_op_tag is None or immune_op_tag not in instruction.tag)
            for instruction in stim_circuit
        ):
            warnings.warn(
                "Quadratic idle errors are split at the boundaries of REPEAT blocks and of their"
                " iterations, which undercounts them for idle periods that cross those boundaries."
                "  Adding noise to circuit.flattened() avoids this",
                stacklevel=2,
            )

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
            has_idle_noise=schedule is not None,
        )
        noisy_circuit = builder.get_noisy_circuit(stim_circuit, schedule)

        # return a circuit of the type of the input
        if isinstance(circuit, stim.Circuit):
            return noisy_circuit
        output = type(circuit)()
        output += noisy_circuit
        return output

    def _get_idle_noise(self, num_idle: int, num_waiting_for_m_or_r: int) -> stim.Circuit:
        """The noise of an idle run of qubit 0.

        Pauli channels are combined into one channel, and other noise is applied once per moment.

        Args:
            num_idle: The number of moments of the run in which the qubit idles, which get
                ``idle_error`` and ``quadratic_idle_error``.
            num_waiting_for_m_or_r: The number of moments of the run in which another qubit, but
                not this one, is measured or reset, which get
                ``additional_error_waiting_for_m_or_r``.
        """
        # collect the Pauli channels of these moments, and append other noise once per moment
        noise = stim.Circuit()
        channels: list[PauliChannel | PauliChannelSequence] = []
        for rule, count in (
            (self.idle_error, num_idle),
            (self.additional_error_waiting_for_m_or_r, num_waiting_for_m_or_r),
        ):
            if rule is None or not count:
                continue
            if isinstance(rule.after, (PauliChannel, PauliChannelSequence)):
                channels.append(
                    rule.after if count == 1 else _repeated_idle_channel(rule.after, count)
                )
            else:
                for _ in range(count):
                    rule.emit_after(noise, [0], context="idle qubits")
        if self.quadratic_idle_error is not None and num_idle:
            channels.append(_quadratic_idle_channel(self.quadratic_idle_error, num_idle))

        # combine the Pauli channels into one
        if channels:
            channel = (
                channels[0]
                if len(channels) == 1
                else functools.reduce(_compose_idle_channels, channels)
            )
            NoiseRule(after=channel).emit_after(noise, [0], context="idle qubits")
        return noise


class _NoisyCircuitBuilder:
    """Inserts noise into a circuit, keeping the noise on each qubit pending until it is next used.

    An operation (``_Operation``) is a part of an instruction that acts independently on its qubits
    (see :class:`Schedule`), such as one application of a gate.  A gate takes time if the
    scheduler's ``takes_time`` says so: it then occupies its qubits for its moment, and otherwise
    leaves them idle in its moment.

    Pending noise is noise that the builder has made but not yet appended to its output.  The
    pending noise on some qubits is emitted, that is, appended to the output in the order in which
    it was made, before the next operation that depends on the state of those qubits.  All pending
    noise is emitted before each REPEAT block, at the end of each iteration of its body, and at the
    end of the circuit.

    If the noise model has idle noise, the builder also follows a schedule of the circuit, and the
    idle runs of qubits (see :class:`NoiseModel`).  A gate ends the idle runs of the qubits that it
    depends on and, if it takes time, of all its qubits.  When an idle run ends, its noise joins the
    pending noise of its qubit: the idle errors of the moments in which the qubit idles, and the
    additional error of the moments in which another qubit, but not this one, is measured or reset.
    Every idle run ends at the start and end of each REPEAT block and of each of its iterations, so
    every iteration gets the same noise.
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
        has_idle_noise: bool,
    ) -> None:
        self.noise_model = noise_model
        self.takes_time = takes_time
        self.immune_qubits = immune_qubits
        self.immune_op_tag = immune_op_tag
        self.immunize_gates = immunize_gates
        self.has_idle_noise = has_idle_noise
        self.idle_qubits = idle_qubits
        self.noiseless = False  # whether the current instructions are in a noiseless block

        # the pending noise and its qubits, by a key that numbers noise in the order it is added,
        # and the keys of the pending noise on each qubit
        self.pending_noise: dict[int, tuple[Iterable[int], stim.Circuit]] = {}
        self.pending_noise_keys_by_qubit: dict[int, set[int]] = collections.defaultdict(set)
        self.next_noise_key = 0

        # a cache of the noise of an idle run on qubit 0, by its numbers of idle moments and of
        # moments waiting for a measurement or reset
        self.idle_noise_by_num_moments: dict[tuple[int, int], stim.Circuit] = {}

        # the current idle run of each qubit, and the moments that contain a measurement or reset
        self.idle_runs: dict[int, _IdleRun] = {}
        self.measurement_or_reset_moments: list[int] = []

    def get_noisy_circuit(self, circuit: stim.Circuit, schedule: Schedule | None) -> stim.Circuit:
        """Add noise to a circuit, given a schedule of it if the noise model has idle noise."""
        output = stim.Circuit()
        self.append(circuit, schedule, output)
        self.emit_pending_noise(output)
        return output

    def append(
        self, circuit: stim.Circuit, schedule: Schedule | None, output: stim.Circuit
    ) -> None:
        """Add noise to a circuit, and append it to the output."""
        for segment_or_block in _get_segments(circuit, schedule):
            if isinstance(segment_or_block, _Segment):
                self.append_segment(segment_or_block, output)
            else:
                block, body_schedule = segment_or_block
                self.append_block(block, body_schedule, output)

    def append_segment(self, segment: _Segment, output: stim.Circuit) -> None:
        """Add noise to the instructions of a ``_Segment``, and append them to the output."""
        # start the idle runs, and find the moments with measurements or resets
        if self.has_idle_noise:
            self.idle_runs = {qubit: _IdleRun() for qubit in sorted(self.idle_qubits)}
            self.measurement_or_reset_moments = sorted(
                {
                    moment
                    for operation, moment in zip(segment.operations, segment.moments)
                    if operation.instruction.name in MEASURE_OR_RESET_OPS
                }
            )

        # add noise to each gate, and append each other instruction after the pending noise on the
        # qubits that it depends on (those of an observable), while their idle runs continue
        start = 0
        for instruction in segment.instructions:
            num_operations = len(_get_instruction_operations(instruction))
            operations = segment.operations[start : start + num_operations]
            moments = (
                segment.moments[start : start + num_operations] if self.has_idle_noise else None
            )
            start += num_operations
            if operations and operations[0].is_gate:
                self.append_gate(instruction, operations, moments, output)
            else:
                qubits_depended_on = [
                    qubit for operation in operations for qubit in operation.qubits_depended_on
                ]
                if qubits_depended_on:
                    self.emit_pending_noise(output, qubits_depended_on)
                output.append(instruction)

        # end the idle runs after the last moment
        if self.has_idle_noise:
            for qubit in self.idle_runs:
                self.add_idle_noise(qubit, self.end_idle_run(qubit, segment.duration, False, False))

    def append_gate(
        self,
        instruction: stim.CircuitInstruction,
        applications: list[_Operation],
        moments: list[int] | None,
        output: stim.Circuit,
    ) -> None:
        """Add noise to the applications of a gate, and append them to the output.

        The gate is split only where the noise of an application must precede a later application.
        A noise rule is applied once to consecutive applications that share it.
        """
        # the applications are appended in chunks: runs of consecutive applications, each appended
        # after the pending noise on the qubits that its applications depend on
        rules = self.get_rules(instruction, applications)
        chunk: list[int] = []  # the indices of the applications in the chunk
        chunk_qubits: set[int] = set()  # the qubits that the applications of the chunk depend on
        noisy_qubits: set[int] = set()  # the qubits with noise from the rules of the chunk
        chunk_idle_runs: list[tuple[int, tuple[int, int]]] = []  # idle runs that the chunk ends

        def append_chunk() -> None:
            # emit the noise that precedes the chunk
            for qubit, idle_counts in chunk_idle_runs:
                self.add_idle_noise(qubit, idle_counts)
            self.emit_pending_noise(output, chunk_qubits)

            # append consecutive applications with the same rule as one instruction, and add their
            # noise to the pending noise
            groups: list[tuple[NoiseRule | None, list[int]]] = []
            for index in chunk:
                if groups and groups[-1][0] is rules[index]:
                    groups[-1][1].append(index)
                else:
                    groups.append((rules[index], [index]))
            for rule, indices in groups:
                gate = _join_operations([applications[index] for index in indices])
                if rule is None:
                    output.append(gate)
                    continue
                noisy_gate, after = rule.noisy_operation(gate)
                output.append(noisy_gate)
                qubits = {qubit for index in indices for qubit in applications[index].qubits}
                self.add_pending_noise(after, qubits)

        for index, operation in enumerate(applications):
            # end the idle runs of the qubits that the application acts on, if it takes time, or
            # otherwise of those that it depends on
            qubits = operation.qubits_depended_on
            if moments is None:
                ended_idle_runs = []
            else:
                takes_time = operation.is_gate and self.takes_time(operation.instruction)
                ended_run_qubits = operation.qubits if takes_time else qubits
                ended_idle_runs = [
                    (
                        qubit,
                        self.end_idle_run(
                            qubit,
                            moments[index],
                            takes_time,
                            operation.instruction.name in MEASURE_OR_RESET_OPS,
                        ),
                    )
                    for qubit in ended_run_qubits
                ]

            # start a new chunk if the application must follow noise on a qubit of the chunk: the
            # noise of a rule of the chunk, or of an idle run that the application ends
            if not chunk_qubits.isdisjoint(qubits) and (
                not noisy_qubits.isdisjoint(qubits)
                or any(
                    any(idle_counts)
                    for qubit, idle_counts in ended_idle_runs
                    if qubit in chunk_qubits
                )
            ):
                append_chunk()
                chunk, chunk_qubits, noisy_qubits, chunk_idle_runs = [], set(), set(), []
            chunk_idle_runs.extend(ended_idle_runs)
            chunk.append(index)
            chunk_qubits.update(qubits)
            if rules[index] is not None:
                noisy_qubits.update(qubits)
        append_chunk()

    def get_rules(
        self, instruction: stim.CircuitInstruction, applications: list[_Operation]
    ) -> list[NoiseRule | None]:
        """The noise rule of each application of a gate, after accounting for immune qubits."""
        # noiseless gates get no rules
        if self.noiseless or (
            self.immune_op_tag is not None and self.immune_op_tag in instruction.tag
        ):
            return [None] * len(applications)

        # look up the rule once when the gate name determines it, then apply immunity to each
        # application
        if (
            self.noise_model.rule_func is None
            and op_type(instruction.name) not in (CLIFFORD_PP, JUST_MEASURE_PP)
            and not involves_classical_bits(instruction)
        ):
            rule = self.noise_model.get_noise_rule(applications[0].instruction)
            if rule is None or not self.immune_qubits:
                return [rule] * len(applications)
            return [
                _rule_with_immunity(
                    rule,
                    operation.instruction,
                    self.immune_qubits,
                    immunize_gates=self.immunize_gates,
                )
                for operation in applications
            ]
        rules: list[NoiseRule | None] = []
        for operation in applications:
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

        Idle runs end at the boundaries of the block and of its iterations, so every iteration gets
        the same noise, which is added once.  In a REPEAT block tagged with ``immune_op_tag``, gates
        get no noise, and the qubits whose idle runs those gates end get no idle noise.
        """
        # a noiseless block adds no noise to its gates, nor idle noise to the qubits whose idle runs
        # they end
        self.emit_pending_noise(output)
        body = block.body_copy()
        outer_state = self.noiseless, self.idle_qubits
        if self.immune_op_tag is not None and self.immune_op_tag in block.tag:
            self.noiseless = True
            self.idle_qubits = self.idle_qubits - _get_gate_qubits(body, self.takes_time)

        # add noise to one iteration, which every iteration repeats
        iteration = stim.Circuit()
        self.append(body, schedule, iteration)
        self.emit_pending_noise(iteration)
        self.noiseless, self.idle_qubits = outer_state
        output.append(stim.CircuitRepeatBlock(block.repeat_count, iteration, tag=block.tag))

    def end_idle_run(
        self, qubit: int, moment: int, takes_time: bool, measures_or_resets: bool
    ) -> tuple[int, int]:
        """End the idle run of a qubit at a gate or after the last moment, and start a new one.

        A gate in the moment in which the run starts ends no run, and joins that moment.

        Args:
            qubit: The qubit whose idle run ends.
            moment: The moment of the gate, or, after the last moment, the number of moments.
            takes_time: Whether the gate takes time.
            measures_or_resets: Whether the gate measures or resets the qubit.

        Returns:
            The numbers of moments of the run in which the qubit idles, and in which another qubit,
            but not this one, is measured or reset.
        """
        run = self.idle_runs.get(qubit)
        if run is None:
            return 0, 0
        if moment <= run.last_gate_moment:
            # another gate in the moment of the last gate
            run.last_gate_takes_time |= takes_time
            run.last_gate_measures_or_resets |= measures_or_resets
            return 0, 0

        # the qubit idles from the moment of its last gate, or from the next moment if that gate
        # took time, until this moment, and waits in the moments between them with a measurement or
        # reset, other than one of its own
        num_measurement_or_reset_moments = bisect.bisect_left(
            self.measurement_or_reset_moments, moment
        )
        idle_counts = (
            moment - run.last_gate_moment - run.last_gate_takes_time,
            num_measurement_or_reset_moments
            - run.num_measurement_or_reset_moments
            - run.last_gate_measures_or_resets,
        )
        run.last_gate_moment = moment
        run.num_measurement_or_reset_moments = num_measurement_or_reset_moments
        run.last_gate_takes_time = takes_time
        run.last_gate_measures_or_resets = measures_or_resets
        return idle_counts

    def add_idle_noise(self, qubit: int, idle_counts: tuple[int, int]) -> None:
        """Add the noise of an idle run of a qubit to the pending noise.

        Args:
            qubit: The qubit of the idle run.
            idle_counts: The numbers of moments of the run in which the qubit idles, and in which it
                waits while another qubit is measured or reset.
        """
        if not any(idle_counts):
            return
        if idle_counts not in self.idle_noise_by_num_moments:
            self.idle_noise_by_num_moments[idle_counts] = self.noise_model._get_idle_noise(
                *idle_counts
            )
        noise = self.idle_noise_by_num_moments[idle_counts]
        if noise:
            self.add_pending_noise(with_remapped_qubits(noise, [qubit]), (qubit,))

    def add_pending_noise(self, noise: stim.Circuit, qubits: Iterable[int]) -> None:
        """Add noise on some qubits to the pending noise."""
        if not noise:
            return
        self.pending_noise[self.next_noise_key] = (qubits, noise)
        for qubit in qubits:
            self.pending_noise_keys_by_qubit[qubit].add(self.next_noise_key)
        self.next_noise_key += 1

    def emit_pending_noise(self, output: stim.Circuit, qubits: Iterable[int] | None = None) -> None:
        """Emit the pending noise on some qubits (by default, on all qubits), in order."""
        if qubits is None:
            keys: Iterable[int] = list(self.pending_noise)
        else:
            keys = sorted(
                {key for qubit in qubits for key in self.pending_noise_keys_by_qubit.get(qubit, ())}
            )
        for key in keys:
            noise_qubits, noise = self.pending_noise.pop(key)
            for qubit in noise_qubits:
                self.pending_noise_keys_by_qubit[qubit].discard(key)
            output += noise


@dataclasses.dataclass(slots=True)
class _IdleRun:
    """The start of the current idle run of a qubit: the moment of its last gate, or zero."""

    last_gate_moment: int = 0
    # the number of moments before last_gate_moment that contain a measurement or reset
    num_measurement_or_reset_moments: int = 0
    # whether a gate on the qubit in last_gate_moment takes time, or measures or resets the qubit
    last_gate_takes_time: bool = False
    last_gate_measures_or_resets: bool = False


def _join_operations(operations: Sequence[_Operation]) -> stim.CircuitInstruction:
    """Join applications of one gate into one instruction."""
    if len(operations) == 1:
        return operations[0].instruction
    first = operations[0].instruction
    targets = [
        target for operation in operations for target in operation.instruction.targets_copy()
    ]
    return stim.CircuitInstruction(first.name, targets, first.gate_args_copy(), tag=first.tag)


def _get_gate_qubits(
    circuit: stim.Circuit, takes_time: Callable[[stim.CircuitInstruction], bool]
) -> frozenset[int]:
    """The qubits whose idle runs the gates of a circuit end, at any depth.

    These are the qubits that a gate depends on, and all qubits of a gate that takes time.
    """
    qubits: set[int] = set()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            qubits.update(_get_gate_qubits(instruction.body_copy(), takes_time))
        else:
            for operation in _get_operations([instruction]):
                if operation.is_gate:
                    qubits.update(operation.qubits_depended_on)
                if operation.is_gate and takes_time(operation.instruction):
                    qubits.update(operation.qubits)
    return frozenset(qubits)


####################################################################################################
# built-in noise models


class DepolarizingNoiseModel(NoiseModel):
    """Creates a near-standard circuit depolarizing noise model.

    Every error has the same probability p:

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
            moments, for idle noise.  Default: ``ASAPScheduler()``.

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

    See https://arxiv.org/abs/2108.10457.  In a moment in which another qubit is measured or reset,
    an idling qubit gets depolarized with probability ``p/10`` and also ``2*p``, as in the cited
    reference implementation, and a qubit occupied by a unitary gate that takes time gets ``2*p`` in
    addition to the noise of its gate.

    Args:
        p: The two-qubit gate error probability, which sets all other error rates.
        scheduler: The :class:`~qldpc.circuits.Scheduler` that places the operations of circuits in
            moments, for idle noise.  Default:
            ``TickPreservingScheduler(single_qubit_gates_take_time=True)``, as in the cited
            reference.
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
                TickPreservingScheduler(single_qubit_gates_take_time=True)
                if scheduler is None
                else scheduler
            ),
        )
