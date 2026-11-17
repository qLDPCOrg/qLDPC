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
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence

import stim

from qldpc._utils import format_docstring

from ..common import (
    DEFAULT_IMMUNE_OP_TAG,
    DEFAULT_IMMUNE_QUBIT_TAG,
    StimCircuitLike,
    _as_stim_circuit,
    _get_immune_qubits,
    _get_instructions,
    _new_instruction,
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
    get_gate_aliases,
    involves_classical_bits,
    op_type,
)
from .channels import (
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
    _normalize_rules,
    _rule_with_immunity,
    _validate_rule_for_arity,
    _validate_rule_for_op,
)
from .scheduling import (
    ASAPScheduler,
    Schedule,
    Scheduler,
    _get_segments,
    _match_schedule,
    _Operation,
    _Segment,
)

####################################################################################################
# primary methods and classes: as_noiseless_circuit and NoiseModel


def as_noiseless_circuit(circuit: StimCircuitLike) -> StimCircuitLike:
    """Wrap a circuit in a one-iteration REPEAT block tagged with ``DEFAULT_IMMUNE_OP_TAG``.

    With the default ``immune_op_tag``, :meth:`NoiseModel.noisy_circuit` adds no noise to the gates
    of this block, or to the qubits whose idle runs they end (see :class:`NoiseModel`).
    """
    # a wrapped circuit stays as it is
    if (
        len(circuit) == 1
        and isinstance(circuit[0], stim.CircuitRepeatBlock)
        and DEFAULT_IMMUNE_OP_TAG in circuit[0].tag
    ):
        return circuit
    body = _as_stim_circuit(circuit)
    block = stim.CircuitRepeatBlock(repeat_count=1, body=body, tag=DEFAULT_IMMUNE_OP_TAG)
    noiseless_circuit = type(circuit)()
    noiseless_circuit.append(block)
    return noiseless_circuit


class NoiseModel:
    """A model that defines how to add noise to circuits.

    A noise model adds errors after gates, to measurement results, after resets, and on idling
    qubits.  :meth:`get_noise_rule` gives the noise of each gate.  Classically controlled gates get
    no noise.

    Idle errors depend on how long qubits idle.  A schedule of a circuit, from the ``scheduler``,
    places its operations in moments, which are units of time, and says which gates take time.  An
    idle run of a qubit starts at the start of the circuit or at a gate on the qubit.  It ends at
    the next gate on the qubit, or at the end of the circuit.  A classically controlled gate ends no
    run.  The qubit idles in each moment of a run before the moment of the gate that ends it.  It
    does not idle in the first moment of the run if a gate on the qubit takes time there.
    :class:`~qldpc.circuits.Schedule` describes moments, and how REPEAT blocks split idle runs.

    Args:
        clifford_1q_error: The default noise after each one-qubit unitary Clifford gate.  A float
            ``p`` is depolarizing noise of total probability ``p``.  The noise may also be a Mapping
            from Pauli strings to probabilities, a Pauli channel, or a
            :class:`~qldpc.circuits.NoiseRule`.
        clifford_2q_error: The default noise after each two-qubit unitary Clifford gate, as for
            ``clifford_1q_error``.
        readout_error: The default probability of flipping each measurement result.
        reset_error: The default probability of resetting a qubit to the wrong state.
        clifford_nq_error: The default noise after each unitary Clifford gate on ``k`` qubits, by
            ``k``, as for ``clifford_1q_error``.  Each Pauli product of an SPP or SPP_DAG gate
            counts as a gate on its qubits.  A gate whose number of qubits has no entry gets no
            default noise.  Entries 1 and 2 replace ``clifford_1q_error`` and ``clifford_2q_error``,
            and cannot be given with them.
        idle_error: The noise on each idling qubit in each moment, as for ``clifford_1q_error`` on
            one qubit.  Pauli noise is applied once per idle run, with the probabilities of the
            whole run.  Noise given as a ``stim.Circuit`` in a :class:`~qldpc.circuits.NoiseRule` is
            applied once per idle moment.
        quadratic_idle_error: An error on each idling qubit, for slow noise, whose probability grows
            about quadratically with the length of an idle run.  It is given for one moment as a
            float ``p``, a Mapping such as ``{"Z": p}``, or a one-qubit ``PauliChannel``, with total
            probability ``p`` at most 1/2.  Over a run with ``k`` idle moments, it has total
            probability ``(1 - (1 - 2 * p)**(k**2)) / 2``, in the same proportions among Paulis.
            :mod:`qldpc.circuits.noise_model` describes this model.
        additional_error_waiting_for_m_or_r: Noise as for ``idle_error``, on each qubit in each
            moment in which another qubit, but not this one, is measured or reset.  A qubit gets
            this noise even when a gate acts on it in that moment.
        rules: Noise rules by gate name, which take precedence over the defaults above.  A key is a
            stim gate name, or the name of one Pauli product of an MPP, SPP, or SPP_DAG gate, as in
            ``"MXYZ"`` for ``MPP X*Y*Z`` or ``"SXY_DAG"`` for ``SPP_DAG X*Y``.
        rule_func: A function that gives the noise rule of one gate application, or None to fall
            back on ``rules`` and the defaults.  It takes precedence over ``rules``.  It gets one
            gate application at a time, as :class:`~qldpc.circuits.Schedule` defines it.  It is not
            called for classically controlled gates.
        scheduler: The :class:`~qldpc.circuits.Scheduler` that places the operations of circuits in
            moments, for idle errors, or a :class:`~qldpc.circuits.Schedule` to use for every
            circuit.  :mod:`qldpc.circuits.noise_model` describes both.  Default:
            ``ASAPScheduler()``.

    Raises:
        ValueError: If a noise specification is invalid, ambiguous, or duplicated, or does not fit
            the gates that it applies to.
        TypeError: If a noise specification or a key of ``rules`` has an unsupported type.
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
        scheduler: Scheduler | Schedule | None = None,
    ) -> None:
        """Initializes a noise model with specified parameters."""
        self.rules = None if rules is None else _normalize_rules(rules)
        self.rule_func = rule_func
        self.scheduler = ASAPScheduler() if scheduler is None else scheduler
        for probability in (readout_error, reset_error):
            if probability is not None and not (0 <= probability <= 1):
                raise ValueError(f"{probability=} is not between 0 and 1")
        self.readout_error = readout_error or 0
        self.reset_error = reset_error or 0

        # merge clifford_1q_error and clifford_2q_error into clifford_nq_error.  Giving both forms
        # for one gate size is an error, even if one of them adds no noise.
        for size, param, name in (
            (1, clifford_1q_error, "clifford_1q_error"),
            (2, clifford_2q_error, "clifford_2q_error"),
        ):
            if clifford_nq_error is not None and size in clifford_nq_error and param is not None:
                raise ValueError(
                    f"Ambiguous noise specification: both `clifford_nq_error[{size}]` and"
                    f" `{name}` are set.  Specify one or the other."
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
        # idle errors act on one qubit at a time, and neither measure nor reset it.  Check this
        # before dropping errors that add no noise, so that an empty channel on the wrong number of
        # qubits is still caught.
        for field_name, idle_rule in (
            ("idle_error", self.idle_error),
            ("additional_error_waiting_for_m_or_r", self.additional_error_waiting_for_m_or_r),
        ):
            if idle_rule is not None:
                _validate_rule_for_arity(idle_rule, 1, field_name)
        # a quadratic idle error must be a one-qubit Pauli channel of total probability at most 1/2
        self.quadratic_idle_error: PauliChannel | None = None
        quadratic_idle_rule = _as_noise_rule(quadratic_idle_error, 1)
        if quadratic_idle_rule is not None and quadratic_idle_rule.after is not None:
            channel = quadratic_idle_rule.after
            if not isinstance(channel, PauliChannel) or channel.num_qubits != 1:
                raise ValueError("`quadratic_idle_error` must be a one-qubit PauliChannel")
            # allow for rounding errors in probabilities that sum to 1/2
            if math.fsum(channel.probabilities.values()) > 0.5 + 1e-9:
                raise ValueError(
                    "`quadratic_idle_error` must have a total probability of at most 1/2"
                )
            self.quadratic_idle_error = channel or None
        # After checking the arities of idle rules, drop those that add no error.
        self.idle_error = self.idle_error or None
        self.additional_error_waiting_for_m_or_r = self.additional_error_waiting_for_m_or_r or None

    @property
    def clifford_1q_error(self) -> NoiseRule | None:
        """The default noise rule of one-qubit unitary Clifford gates, or None."""
        return self.clifford_nq_error.get(1)

    @property
    def clifford_2q_error(self) -> NoiseRule | None:
        """The default noise rule of two-qubit unitary Clifford gates, or None."""
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
        """A constructor-like description that lists only the fields that are set.

        Without a ``rule_func``, this evaluates to an equivalent ``NoiseModel`` if the scheduler's
        repr does, as for the built-in schedulers.  A ``rule_func`` is shown by its module and name,
        so the result then cannot be evaluated.
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
        """Find the noise rule of one gate application.

        Annotations, noise channels, and classically controlled gates get no noise.  Other
        instructions get the rule from the first of these that gives one:

        1. ``rule_func``.
        2. ``rules``, by the most specific name of the gate.
        3. ``clifford_nq_error``, for a unitary Clifford gate.
        4. ``readout_error`` and ``reset_error``, for a measurement or reset.

        Args:
            op: One gate application, as passed to ``rule_func``.

        Returns:
            The noise rule of the gate application, or None for no noise.

        Raises:
            ValueError: If a rule from ``rule_func`` or ``rules`` does not fit the gate application.
        """
        # annotations, noise channels, and classically controlled gates get no noise
        this_op_type = op_type(op.name)
        if this_op_type in (ANNOTATION, NOISE) or involves_classical_bits(op):
            return None

        # a rule from rule_func comes first
        if self.rule_func is not None and this_op_type in GATE_OP_TYPES:
            rule = self.rule_func(op)
            if rule is not None:
                _validate_rule_for_op(rule, op, f"rule_func returned a rule for {op.name!r}")
                return rule

        # then a rule by the most specific name of the gate.  A rule for bare MPP, SPP, or SPP_DAG
        # must fit this Pauli product.
        if self.rules is not None:
            for name in get_gate_aliases(op):
                rule = self.rules.get(name)
                if rule is not None:
                    if name in {"MPP", "SPP", "SPP_DAG"}:
                        _validate_rule_for_op(rule, op, f"rules[{name!r}]")
                    return rule

        # then the default rule for unitary Cliffords on this number of qubits
        if this_op_type in (CLIFFORD_1Q, CLIFFORD_2Q, CLIFFORD_PP):
            if this_op_type == CLIFFORD_PP:
                num_qubits = sum(1 for target in op.targets_copy() if not target.is_combiner)
            else:
                num_qubits = 1 if this_op_type == CLIFFORD_1Q else 2
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
        """Add noise to a circuit.

        The noise of each gate, and of each idle run of a qubit (see :class:`NoiseModel`), goes
        before the next gate or observable that depends on the state of its qubits, or else at the
        end of the circuit.  Noise does not cross the start or end of a REPEAT block or of one of
        its iterations (see :class:`~qldpc.circuits.Schedule`).

        Args:
            circuit: The circuit to add noise to.
            system_qubits: The qubits that can get idle errors.  Default: all qubits of the circuit,
                ``range(circuit.num_qubits)``.
            immune_qubits: Qubits that get no noise.  Default: none.
            immune_op_tag: An instruction whose tag contains this string gets no noise.  In a REPEAT
                block whose tag contains it, gates get no noise, and neither do the qubits whose
                idle runs they end.  None turns this off.  Default: "{DEFAULT_IMMUNE_OP_TAG}".
            immune_qubit_tag: A qubit gets no noise if a QUBIT_COORDS annotation of it has a tag
                that contains this string.  None turns this off.  Default:
                "{DEFAULT_IMMUNE_QUBIT_TAG}".
            immunize_gates: If True (the default), a gate on an immune qubit gets no noise.  If
                False, its Pauli noise keeps only the Pauli strings that act as ``I`` on immune
                qubits, with their probabilities.
            insert_ticks: Deprecated.  To lay out a noisy circuit in moments, add noise with
                ``scheduler=TickScheduler()`` to
                ``TickScheduler().get_schedule(circuit).to_circuit()``.

        Returns:
            The circuit with noise, of the same type as the input.

        Raises:
            ValueError: If an immunity tag is empty, the schedule does not match the circuit (see
                :mod:`qldpc.circuits.noise_model`), or a noise rule does not fit a gate.  With
                ``immunize_gates=False``, also if a gate on both immune and other qubits has noise
                other than a Pauli channel.
        """
        # check the immunity tags
        if immune_op_tag == "":
            raise ValueError("immune_op_tag must be non-empty or None")
        if immune_qubit_tag == "":
            raise ValueError("immune_qubit_tag must be non-empty or None")

        # find the qubits that may idle, and those that are immune to noise
        stim_circuit = _as_stim_circuit(circuit)
        system_qubits = frozenset(
            range(stim_circuit.num_qubits) if system_qubits is None else system_qubits
        )
        immune_qubits = frozenset(immune_qubits) | _get_immune_qubits(
            stim_circuit, immune_qubit_tag
        )

        scheduler = self.scheduler
        if insert_ticks is not None:  # DEPRECATED: see qldpc._deprecations
            from qldpc._deprecations.noise_model import apply_insert_ticks

            stim_circuit, scheduler = apply_insert_ticks(insert_ticks, stim_circuit, scheduler)

        # warn if REPEAT blocks that get noise split quadratic idle errors
        if self.quadratic_idle_error is not None and any(
            isinstance(instruction, stim.CircuitRepeatBlock)
            and (immune_op_tag is None or immune_op_tag not in instruction.tag)
            for instruction in _get_instructions(stim_circuit)
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
            schedule = (
                scheduler
                if isinstance(scheduler, Schedule)
                else scheduler.get_schedule(stim_circuit)
            )
            schedule = _match_schedule(schedule, stim_circuit)
        builder = _NoisyCircuitBuilder(
            self,
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

        Pauli noise is applied once, with the probabilities of the whole run, and other noise once
        per moment.

        Args:
            num_idle: The number of moments of the run in which the qubit idles, which get
                ``idle_error`` and ``quadratic_idle_error``.
            num_waiting_for_m_or_r: The number of moments of the run in which another qubit, but not
                this one, is measured or reset, which get ``additional_error_waiting_for_m_or_r``.
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
            channel = functools.reduce(_compose_idle_channels, channels)
            NoiseRule(after=channel).emit_after(noise, [0], context="idle qubits")
        return noise


class _NoisyCircuitBuilder:
    """Inserts noise into a circuit, keeping the noise on each qubit pending until it is next used.

    Pending noise is noise that the builder has made but not yet appended to its output.  The
    builder appends the pending noise on some qubits, in the order in which it was made, before the
    next operation that depends on the state of those qubits.  It appends all pending noise before
    each REPEAT block, at the end of each of its iterations, and at the end of the circuit.

    If the noise model has idle errors, the builder also follows the idle run of each qubit (see
    :class:`NoiseModel`) in a schedule of the circuit, as matched by ``_match_schedule``.  A gate
    ends the idle runs of the qubits that it depends on and, if it takes time, of all its qubits.
    When an idle run ends, its noise joins the pending noise of its qubit.
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        *,
        idle_qubits: frozenset[int],
        immune_qubits: frozenset[int],
        immune_op_tag: str | None,
        immunize_gates: bool,
        has_idle_noise: bool,
    ) -> None:
        self.noise_model = noise_model
        self.immune_qubits = immune_qubits
        self.immune_op_tag = immune_op_tag
        self.immunize_gates = immunize_gates
        self.has_idle_noise = has_idle_noise
        self.idle_qubits = idle_qubits
        self.noiseless = False  # whether the current instructions are in a noiseless block

        # the pending noise and its qubits, keyed by the order in which it was added, and the keys
        # of the pending noise on each qubit
        self.pending_noise: dict[int, tuple[Collection[int], stim.Circuit]] = {}
        self.pending_noise_keys_by_qubit: dict[int, set[int]] = collections.defaultdict(set)
        self.next_noise_key = 0

        # the noise of an idle run on qubit 0, cached by its two counts of moments (see
        # NoiseModel._get_idle_noise)
        self.idle_noise_by_num_moments: dict[tuple[int, int], stim.Circuit] = {}
        # that noise on each qubit, cached by the qubit and the counts.  Remapping qubits is slow,
        # and the idle runs of a qubit often repeat their counts of moments.
        self.idle_noise_by_qubit_and_num_moments: dict[tuple[int, int, int], stim.Circuit] = {}

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
        # qubits that it depends on, which only an observable has.  Only gates end idle runs.
        for instruction, span in zip(segment.instructions, segment.spans):
            operations = segment.operations[span]
            timing = None
            if self.has_idle_noise:
                timing = list(zip(segment.moments[span], segment.takes_time[span]))
            if operations and operations[0].is_gate:
                self.append_gate(instruction, operations, timing, output)
            else:
                qubits_depended_on = [
                    qubit for operation in operations for qubit in operation.qubits_depended_on
                ]
                if qubits_depended_on:
                    self.emit_pending_noise(output, qubits_depended_on)
                output.append(instruction)

        # end the idle runs after the last moment
        if self.has_idle_noise:
            num_moments = segment.num_moments
            for qubit in self.idle_runs:
                self.add_idle_noise(qubit, self.end_idle_run(qubit, num_moments, False, False))

    def append_gate(
        self,
        instruction: stim.CircuitInstruction,
        applications: list[_Operation],
        timing: list[tuple[int, bool]] | None,
        output: stim.Circuit,
    ) -> None:
        """Add noise to the applications of a gate, and append them to the output.

        ``timing`` gives the moment of each application, and whether it takes time, if the noise
        model has idle errors.

        The gate is split only where the noise of an application must precede a later application.
        A noise rule is applied once to consecutive applications that share it.
        """
        # append the applications in chunks of consecutive applications, each after the pending
        # noise on the qubits that its applications depend on
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
            if timing is None:
                ended_idle_runs = []
            else:
                moment, takes_time = timing[index]
                ended_run_qubits = operation.qubits if takes_time else qubits
                ended_idle_runs = [
                    (
                        qubit,
                        self.end_idle_run(
                            qubit,
                            moment,
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

        # otherwise, look up the rule of each application, and apply immunity to it
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

        Every iteration gets the same noise (see :class:`Schedule`), so the builder makes the noise
        of one iteration.
        """
        # emit the pending noise, and in a noiseless block, add no noise to gates or their qubits
        self.emit_pending_noise(output)
        body = block.body_copy()
        outer_state = self.noiseless, self.idle_qubits
        if self.immune_op_tag is not None and self.immune_op_tag in block.tag:
            self.noiseless = True
            if schedule is not None:
                self.idle_qubits = self.idle_qubits - _get_gate_qubits(body, schedule)

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

        # the qubit idles from the moment of its last gate, or the next moment if that gate took
        # time, until this moment.  It also waits in each moment of the run in which other qubits,
        # but not this one, are measured or reset.
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
            idle_counts: The counts of moments of the run, from :meth:`end_idle_run`.
        """
        if not any(idle_counts):
            return
        key = (qubit, *idle_counts)
        if key not in self.idle_noise_by_qubit_and_num_moments:
            if idle_counts not in self.idle_noise_by_num_moments:
                self.idle_noise_by_num_moments[idle_counts] = self.noise_model._get_idle_noise(
                    *idle_counts
                )
            noise = self.idle_noise_by_num_moments[idle_counts]
            self.idle_noise_by_qubit_and_num_moments[key] = with_remapped_qubits(noise, [qubit])
        self.add_pending_noise(self.idle_noise_by_qubit_and_num_moments[key], (qubit,))

    def add_pending_noise(self, noise: stim.Circuit, qubits: Collection[int]) -> None:
        """Add noise on some qubits to the pending noise."""
        if not noise:
            return
        self.pending_noise[self.next_noise_key] = (qubits, noise)
        for qubit in qubits:
            self.pending_noise_keys_by_qubit[qubit].add(self.next_noise_key)
        self.next_noise_key += 1

    def emit_pending_noise(self, output: stim.Circuit, qubits: Iterable[int] | None = None) -> None:
        """Emit the pending noise on some qubits, or on all if ``qubits`` is None, in order."""
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
    return _new_instruction(first.name, targets, first.gate_args_copy(), first.tag)


def _get_gate_qubits(circuit: stim.Circuit, schedule: Schedule) -> frozenset[int]:
    """The qubits whose idle runs the gates of a circuit end, including gates in REPEAT blocks."""
    qubits: set[int] = set()
    for segment_or_block in _get_segments(circuit, schedule):
        if isinstance(segment_or_block, _Segment):
            for operation, takes_time in zip(
                segment_or_block.operations, segment_or_block.takes_time
            ):
                if operation.is_gate:
                    qubits.update(operation.qubits if takes_time else operation.qubits_depended_on)
        else:
            block, body_schedule = segment_or_block
            assert body_schedule is not None
            qubits.update(_get_gate_qubits(block.body_copy(), body_schedule))
    return frozenset(qubits)


####################################################################################################
# built-in noise models


class DepolarizingNoiseModel(NoiseModel):
    """A circuit-level depolarizing noise model, in which every error has probability ``p``.

    - One-qubit Clifford gates get one-qubit depolarization.
    - Two-qubit Clifford gates get two-qubit depolarization.
    - Measurements, including MXX and MPP, have their results flipped, without depolarization.
    - Resets reset qubits to the orthogonal state.
    - If ``include_idling_error`` is True, each idling qubit gets depolarized in each moment.

    Args:
        p: The probability of every error.
        include_idling_error: Whether to depolarize idling qubits.  Default: False.
        max_gate_size: The largest number of qubits of a unitary Clifford gate, such as an SPP gate,
            that gets depolarized.  Larger gates get no noise.  Default: 2.
        scheduler: The scheduler or schedule for idle noise, as in :class:`NoiseModel`.  Default:
            ``ASAPScheduler()``.

    Raises:
        ValueError: If ``max_gate_size`` is less than 1, or ``p`` is not between 0 and 1.
    """

    def __init__(
        self,
        p: float,
        *,
        include_idling_error: bool = False,
        max_gate_size: int = 2,
        scheduler: Scheduler | Schedule | None = None,
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
            idle_error=p if include_idling_error else None,
            scheduler=scheduler,
        )


class SI1000NoiseModel(NoiseModel):
    """A superconducting-inspired noise model defined in "A Fault-Tolerant Honeycomb Memory".

    A two-qubit gate error probability ``p`` sets every error rate:

    - One-qubit Clifford gates get one-qubit depolarization with probability ``p/10``.
    - Two-qubit Clifford gates get two-qubit depolarization with probability ``p``.
    - Measurements have their results flipped with probability ``5*p``.
    - Resets reset qubits to the orthogonal state with probability ``2*p``.
    - Each idling qubit gets depolarized with probability ``p/10`` in each moment.
    - In each moment with a measurement or reset, every other qubit also gets depolarized with
      probability ``2*p``, even if a gate acts on it, as in the code that this package was adapted
      from (see :mod:`qldpc.circuits.noise_model`).

    See https://arxiv.org/abs/2108.10457.

    Args:
        p: The two-qubit gate error probability, which sets all other error rates.
        scheduler: The scheduler or schedule for idle noise, as in :class:`NoiseModel`.  Default:
            ``ASAPScheduler()``.
    """

    def __init__(self, p: float, *, scheduler: Scheduler | Schedule | None = None) -> None:
        """Instantiate a superconducting-inspired noise model."""
        self.p = p
        super().__init__(
            clifford_1q_error=p / 10,
            clifford_2q_error=p,
            readout_error=p * 5,
            reset_error=p * 2,
            idle_error=p / 10,
            additional_error_waiting_for_m_or_r=2 * p,
            scheduler=scheduler,
        )
