# SPDX-License-Identifier: Apache-2.0

"""Schedules, which place the operations of a circuit in moments, and schedulers that make them.

A moment is a unit of time in which a qubit is either occupied by an operation or idle.  A
:class:`~qldpc.circuits.NoiseModel` uses a schedule only to find how long each qubit idles between
its operations, and does not reorder or retime the circuit that it adds noise to.
"""

from __future__ import annotations

import abc
import dataclasses
import functools
import math
import operator
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import NoReturn, SupportsIndex

import stim

from ..common import StimCircuitLike
from ._operations import (
    CLIFFORD_1Q,
    CLIFFORD_PP,
    COLLAPSING_OPS,
    GATE_OP_TYPES,
    JUST_MEASURE_OPS,
    JUST_MEASURE_PP,
    JUST_RESET_OPS,
    NOISE,
    involves_classical_bits,
    op_type,
)

# annotations whose order sets the coordinates of detectors and qubits
_COORDINATE_ANNOTATIONS = frozenset({"DETECTOR", "QUBIT_COORDS", "SHIFT_COORDS"})

####################################################################################################
# schedules


class Schedule:
    """The moments of the operations of a circuit.

    The operations of a circuit are its gate applications (one target of a one-qubit gate, one pair
    of targets of a two-qubit gate, or one Pauli product of an MPP or SPP gate) and its annotations
    and noise channels other than TICKs, as listed by :meth:`get_operations`.  A schedule places
    each operation outside REPEAT blocks in a moment, numbered from zero, and each REPEAT block in
    the moments of its iterations, which all follow one schedule of its body.  Every moment lasts
    one unit of time, including a moment without operations, and a qubit idles in each moment
    unless an operation that takes time acts on it (see :meth:`Scheduler.takes_time`).

    Given the schedule of the body of a REPEAT block, a schedule places the first iteration of the
    block after all moments before it, and its iterations one after another.  The built-in
    schedulers may also let a block, and each of its iterations, share its first moment with the
    moment before it.

    A schedule is immutable.

    Args:
        circuit: The circuit to schedule.
        moments: In circuit order, the moment of each operation of the circuit outside REPEAT
            blocks, and the schedule of the body of each REPEAT block.

    Raises:
        ValueError: If ``moments`` does not have one nonnegative integer moment for each operation
            and one schedule of its body for each REPEAT block.
    """

    __slots__ = ("_circuit", "_duration", "_moments")

    _circuit: stim.Circuit
    _moments: tuple[int | _RepeatSchedule, ...]
    _duration: int

    def __init__(self, circuit: StimCircuitLike, moments: Sequence[int | Schedule]) -> None:
        self._initialize(circuit, moments)

    @staticmethod
    def get_operations(
        circuit: StimCircuitLike,
    ) -> list[stim.CircuitInstruction | stim.CircuitRepeatBlock]:
        """The operations of a circuit outside REPEAT blocks, and its REPEAT blocks, in order."""
        operations: list[stim.CircuitInstruction | stim.CircuitRepeatBlock] = []
        for index in range(len(circuit)):
            instruction = circuit[index]
            if isinstance(instruction, stim.CircuitRepeatBlock):
                operations.append(instruction)
            else:
                operations.extend(op.instruction for op in _get_instruction_operations(instruction))
        return operations

    @property
    def num_moments(self) -> int:
        """The number of moments of this schedule, which ends with its last nonempty moment."""
        return self._duration

    def get_moments(self) -> list[list[stim.CircuitInstruction | stim.CircuitRepeatBlock]]:
        """The operations in each moment, in circuit order.

        A REPEAT block is listed in the moment in which its first iteration starts, and
        :meth:`to_circuit` shows the moments of its iterations.  A REPEAT block whose body has no
        operations takes no time, and is not listed.
        """
        moments: list[list[stim.CircuitInstruction | stim.CircuitRepeatBlock]] = [
            [] for _ in range(self._duration)
        ]
        for item in _get_segments(self._circuit, self):
            if isinstance(item, _Segment):
                for operation, moment in zip(item.operations, item.moments):
                    moments[moment].append(operation.instruction)
            else:
                assert item[1] is not None
                if item[1].schedule._duration:
                    moments[item[1].start].append(item[0])
        return moments

    def to_circuit(self) -> tuple[stim.Circuit, tuple[int, ...]]:
        """The scheduled circuit, laid out in moments separated by TICKs.

        Operations are ordered by moment, and otherwise keep their order in the circuit, so a
        measurement moves only if this schedule puts it in an earlier moment than a measurement
        before it.  The detectors, observables, and classically controlled gates of the circuit read
        the same measurements as before.  A moment without operations is an extra TICK.  The first
        iterations of a REPEAT block are written out separately if they start in another way than
        later iterations, or if they read measurements before the block at other offsets than later
        iterations read those of earlier iterations.

        Returns:
            The scheduled circuit, and the index in the original circuit of each of its
            measurements.
        """
        layout = _Layout()
        circuit, _ = layout.lay_out(self, 0)
        return circuit, tuple(layout.new_to_old)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Schedule)
            and str(self._circuit) == str(other._circuit)
            and self._moments == other._moments
        )

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("A Schedule is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("A Schedule is immutable")

    @classmethod
    def _from_placements(
        cls, circuit: StimCircuitLike, moments: Sequence[int | Schedule | _RepeatSchedule]
    ) -> Schedule:
        """A schedule that may also place a REPEAT block with a :class:`_RepeatSchedule`."""
        schedule = object.__new__(cls)
        schedule._initialize(circuit, moments)
        return schedule

    def _initialize(
        self, circuit: StimCircuitLike, moments: Sequence[int | Schedule | _RepeatSchedule]
    ) -> None:
        stim_circuit = _as_stim_circuit(circuit)
        operations = Schedule.get_operations(stim_circuit)
        if len(operations) != len(moments):
            raise ValueError(
                f"A schedule needs {len(operations)} entries (one per operation or REPEAT block),"
                f" but got {len(moments)}"
            )
        entries: list[int | _RepeatSchedule] = []
        duration = 0  # one more than the last moment of the entries so far
        for operation, moment in zip(operations, moments):
            if isinstance(operation, stim.CircuitRepeatBlock):
                if isinstance(moment, Schedule):
                    moment = _RepeatSchedule(moment, duration, moment._duration)
                if not isinstance(moment, _RepeatSchedule) or str(moment.schedule._circuit) != str(
                    operation.body_copy()
                ):
                    raise ValueError("Each REPEAT block needs a schedule of its body")
                if moment.schedule._duration:
                    duration = max(
                        duration,
                        moment.start
                        + (operation.repeat_count - 1) * moment.period
                        + moment.schedule._duration,
                    )
            else:
                moment = _as_moment(moment, f"The moment of {operation}")
                duration = max(duration, moment + 1)
            entries.append(moment)
        object.__setattr__(self, "_circuit", stim_circuit)
        object.__setattr__(self, "_moments", tuple(entries))
        object.__setattr__(self, "_duration", duration)


@dataclasses.dataclass(frozen=True)
class _RepeatSchedule:
    """The placement of a REPEAT block in the moments of the circuit around it.

    Iteration ``k`` (counting from zero) of the block occupies the moments from ``start + k *
    period`` to ``start + k * period + schedule.num_moments - 1``.  If ``period`` is one less than
    ``schedule.num_moments``, consecutive iterations share a moment, so a moment continues across
    the boundary between them.  The first moment of the block may likewise be the last moment
    before the block, and its last moment the first moment after it.

    Args:
        schedule: The schedule of the body of the block.
        start: The moment in which the first iteration starts.
        period: The number of moments from the start of an iteration to the start of the next.
    """

    schedule: Schedule
    start: int
    period: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _as_moment(self.start, "The start of a REPEAT block"))
        object.__setattr__(self, "period", _as_moment(self.period, "The period of a REPEAT block"))


def _as_moment(value: object, description: str) -> int:
    """Check that a value is a nonnegative integer, and return it as an int."""
    if isinstance(value, bool) or not isinstance(value, SupportsIndex) or operator.index(value) < 0:
        raise ValueError(f"{description} must be a nonnegative integer, not {value!r}")
    return operator.index(value)


####################################################################################################
# schedulers


class Scheduler(abc.ABC):
    """A rule that places the operations of circuits in moments.

    A :class:`~qldpc.circuits.NoiseModel` with idle noise uses its scheduler to find how long each
    qubit idles.  The built-in schedulers are made by :meth:`preserve_ticks`, :meth:`asap`, and
    :meth:`alap`, and never make a moment without operations.  A custom scheduler implements
    :meth:`get_schedule`, and may override :meth:`takes_time` and :meth:`validate`.
    """

    @abc.abstractmethod
    def get_schedule(self, circuit: StimCircuitLike) -> Schedule:
        """Place the operations of a circuit in moments."""

    def takes_time(self, op: stim.CircuitInstruction) -> bool:
        """Does one gate application occupy its qubits for a moment?

        By default, every gate except a classically controlled Pauli gate takes time.  Annotations
        and noise channels never take time.  An operation that takes no time leaves its qubits idle
        in its moment, and may share its moment with other operations on its qubits.
        """
        return op_type(op.name) in GATE_OP_TYPES and not involves_classical_bits(op)

    def validate(self, circuit: StimCircuitLike, schedule: Schedule) -> None:
        """Check that a schedule of a circuit keeps the meaning of the circuit.

        The moments of the operations on each qubit, and those of detectors and coordinate
        annotations, may not decrease in circuit order.  An operation that reads a measurement may
        not precede that measurement, a correlated error chain stays in one moment, and no qubit has
        two operations that take time in one moment.  Consecutive iterations of a REPEAT block share
        at most one moment, and the operations and REPEAT blocks of a circuit (or of a REPEAT body)
        that are separated by REPEAT blocks do not overlap in time, apart from sharing a moment at
        their boundary.

        Raises:
            ValueError: If the schedule is not a valid schedule of the circuit.
        """
        if str(schedule._circuit) != str(_as_stim_circuit(circuit)):
            raise ValueError("The schedule is for a different circuit")
        _summarize(schedule, self.takes_time, check=True)

    @staticmethod
    def preserve_ticks(*, single_qubit_gates_take_time: bool = False) -> Scheduler:
        """A scheduler that keeps the moments of a circuit.

        Each TICK ends a moment that contains a gate, and a new moment also starts where an
        operation that takes time acts on a qubit that another operation that takes time already
        acts on in the moment.  Annotations and noise channels after a TICK join the next moment
        with a gate (or, at the end of a circuit outside REPEAT blocks, the moment before them).
        A moment continues across the boundary of a REPEAT block or of its iterations unless a TICK
        or the reuse of a qubit ends it there.

        All iterations of a REPEAT block follow one schedule of its body, so where the first
        iteration would join the moment before the block differently than later iterations join
        the iteration before them, some operations of the block may be placed later than in the
        circuit with its REPEAT blocks written out, which may add moments.  To keep the moments of
        that circuit exactly, schedule (or add noise to) ``circuit.flattened()`` instead.

        Args:
            single_qubit_gates_take_time: If False (the default), single-qubit unitary gates take no
                time, so a moment may contain any number of them on a qubit, which still idles.
        """
        return _TickScheduler(single_qubit_gates_take_time)

    @staticmethod
    def asap(*, late_resets: bool = True, single_qubit_gates_take_time: bool = False) -> Scheduler:
        """A scheduler that places every operation as early as possible, ignoring TICKs.

        The operations between REPEAT blocks, and the bodies of REPEAT blocks, are scheduled on
        their own.  The first moment of each part then shares the last moment of the part (or
        iteration) before it, if no qubit has an operation that takes time in both.

        Args:
            late_resets: If True (the default), place resets as late as possible instead, so that
                qubits do not idle between their reset and their next operation.
            single_qubit_gates_take_time: As in :meth:`preserve_ticks`.
        """
        return _LayerScheduler(False, late_resets, single_qubit_gates_take_time)

    @staticmethod
    def alap(
        *, early_measurements: bool = True, single_qubit_gates_take_time: bool = False
    ) -> Scheduler:
        """A scheduler that places every operation as late as possible, ignoring TICKs.

        REPEAT blocks are handled as in :meth:`asap`.

        Args:
            early_measurements: If True (the default), place measurements as early as possible
                instead, so that qubits do not idle between their last gate and measurement.
            single_qubit_gates_take_time: As in :meth:`preserve_ticks`.
        """
        return _LayerScheduler(True, early_measurements, single_qubit_gates_take_time)


@dataclasses.dataclass(frozen=True)
class _TickScheduler(Scheduler):
    """The scheduler made by :meth:`Scheduler.preserve_ticks`."""

    single_qubit_gates_take_time: bool

    def get_schedule(self, circuit: StimCircuitLike) -> Schedule:
        stim_circuit = _as_stim_circuit(circuit)
        schedule, _, _ = self._schedule(stim_circuit, _TickState(), at_end=True)
        return schedule

    def takes_time(self, op: stim.CircuitInstruction) -> bool:
        return _takes_time(op, self.single_qubit_gates_take_time)

    def __repr__(self) -> str:
        args = "single_qubit_gates_take_time=True" if self.single_qubit_gates_take_time else ""
        return f"Scheduler.preserve_ticks({args})"

    def _schedule(
        self, circuit: stim.Circuit, state: _TickState, *, at_end: bool = False
    ) -> tuple[Schedule, bool, _TickState]:
        """Schedule a circuit (or REPEAT body), given the state of the moment before it.

        Returns:
            The schedule, whether its first moment continues the moment before it, and the state of
            its last moment.
        """
        moments: list[int | _RepeatSchedule] = []
        moment = 0  # the current moment, which is the moment before the circuit if that is open
        used = state.is_open  # whether the current moment contains anything
        occupied = set(state.occupied) if state.is_open else set()
        has_gate = state.has_gate and state.is_open
        closed = False  # whether a TICK has ended the current moment
        gate_moments: set[int] = set()
        for instruction in circuit:
            if isinstance(instruction, stim.CircuitRepeatBlock):
                is_open = used and not closed
                incoming = (
                    _TickState(True, has_gate, frozenset(occupied)) if is_open else _TickState()
                )
                repeat, continues, state_out = self._schedule_block(instruction, incoming)
                duration = repeat.schedule._duration
                if duration:
                    start = moment if continues or not used else moment + 1
                    repeat = _RepeatSchedule(repeat.schedule, start, repeat.period)
                    end = start + (instruction.repeat_count - 1) * repeat.period + duration - 1
                    # the state after a block includes that of the moment before it, if continued
                    occupied, has_gate = set(state_out.occupied), state_out.has_gate
                    moment, closed, used = end, not state_out.is_open, True
                    gate_moments.add(moment)  # the moments of a block need not be checked below
                else:
                    repeat = _RepeatSchedule(repeat.schedule, moment, 0)
                    closed = closed or (has_gate and not state_out.is_open)
                moments.append(repeat)
                continue
            if instruction.name == "TICK":
                closed = closed or has_gate
                continue
            for operation in _get_operations([instruction]):
                timed = operation.is_gate and self.takes_time(operation.instruction)
                if closed or (timed and not occupied.isdisjoint(operation.qubits)):
                    moment, occupied, has_gate, closed = moment + 1, set(), False, False
                if timed:
                    occupied.update(operation.qubits)
                if operation.is_gate:
                    has_gate = True
                    gate_moments.add(moment)
                moments.append(moment)
                used = True
        state_out = _TickState(not closed, has_gate, frozenset(occupied))

        # the first moment continues the moment before the circuit only if it has operations in it
        continues = state.is_open and any(
            entry == 0 if isinstance(entry, int) else entry.start == 0 and entry.schedule._duration
            for entry in moments
        )
        if state.is_open and not continues and moments:
            moments = [
                entry - 1
                if isinstance(entry, int)
                else _RepeatSchedule(entry.schedule, max(entry.start - 1, 0), entry.period)
                for entry in moments
            ]
        if at_end and moment and moment not in gate_moments:
            # annotations after the last gate of a circuit join the moment before them, unless
            # they depend on the state of a qubit
            operations = Schedule.get_operations(circuit)
            last = [
                operation
                for operation, entry in zip(operations, moments)
                if isinstance(entry, int) and entry == moment
            ]
            if not any(_Operation.from_instruction(op).touched_qubits for op in last):
                moments = [moment - 1 if entry == moment else entry for entry in moments]
        return Schedule._from_placements(circuit, moments), continues, state_out

    def _schedule_block(
        self, block: stim.CircuitRepeatBlock, state: _TickState
    ) -> tuple[_RepeatSchedule, bool, _TickState]:
        """Schedule a REPEAT block, given the state of the moment before it.

        Returns:
            The placement of the block, with a start to be set, whether its first moment continues
            the moment before it, and the state of its last moment.
        """
        body = block.body_copy()
        first, continues, state_out = self._schedule(body, state)
        if block.repeat_count == 1:
            return _RepeatSchedule(first, 0, first._duration), continues, state_out

        # find the schedule of later iterations, which converges after a few iterations, and
        # whether every iteration continues the moment of the iteration before it
        schedule, last_state = first, state_out
        iterations_continue = True
        second: tuple[Schedule, bool] | None = None  # the schedule of the second iteration
        for _ in range(4):
            steady_in = last_state  # the state before the iteration with this schedule
            next_schedule, next_continues, next_state = self._schedule(body, last_state)
            second = second or (next_schedule, next_continues)
            iterations_continue = iterations_continue and next_continues
            converged = (next_schedule, next_state) == (schedule, last_state)
            schedule, last_state = next_schedule, next_state
            if converged:
                break
        if schedule != first:
            # later iterations would join the moment before them in another way than the first:
            # schedule all iterations alike, avoiding a first moment without gates of its own, and
            # continuing the moment before the block only if no TICK in the body would end it
            summary = _summarize(schedule, self.takes_time)
            steady_continues = (
                state.is_open
                and (steady_in.is_open and steady_in.has_gate or not state.has_gate)
                and state.occupied.isdisjoint(summary.first_timed)
            )
            if not (
                _starts_with_gate(schedule)
                or (steady_continues and state.has_gate)
                or not _has_gate(body)
            ):
                summary = _summarize(first, self.takes_time)
                schedule = first
                # later iterations continue the moment before them only if the second iteration
                # would do so with the schedule of the first; otherwise each starts a new moment
                iterations_continue = second == (first, True)
                last_state = _TickState(
                    state_out.is_open,
                    state_out.has_gate,
                    state_out.occupied | frozenset(summary.last_timed),
                )
            else:
                continues = steady_continues
        if not _has_gate(body):
            # iterations without gates take no time of their own, so they share one moment
            iterations_continue, last_state = True, state_out
        duration = schedule._duration
        period = duration - 1 if iterations_continue and duration else duration
        return _RepeatSchedule(schedule, 0, period), continues, last_state


def _starts_with_gate(schedule: Schedule) -> bool:
    """Does the first moment of a schedule have a gate?"""
    operations = Schedule.get_operations(schedule._circuit)
    return any(
        entry.start == 0 and _starts_with_gate(entry.schedule)
        if isinstance(operation, stim.CircuitRepeatBlock) and isinstance(entry, _RepeatSchedule)
        else entry == 0 and op_type(operation.name) in GATE_OP_TYPES
        for operation, entry in zip(operations, schedule._moments)
    )


def _has_gate(circuit: stim.Circuit) -> bool:
    """Does a circuit have a gate, at any depth?"""
    return any(
        _has_gate(instruction.body_copy())
        if isinstance(instruction, stim.CircuitRepeatBlock)
        else op_type(instruction.name) in GATE_OP_TYPES
        for instruction in circuit
    )


@dataclasses.dataclass(frozen=True)
class _TickState:
    """The state of the last moment before a part of a circuit, for :meth:`preserve_ticks`."""

    is_open: bool = False  # whether the next operation may join this moment
    has_gate: bool = False
    occupied: frozenset[int] = frozenset()  # the qubits of operations that take time


@dataclasses.dataclass(frozen=True)
class _LayerScheduler(Scheduler):
    """The schedulers made by :meth:`Scheduler.asap` and :meth:`Scheduler.alap`.

    Operations are placed in layers, which become moments, keeping the order of the operations on
    each qubit, of detectors and coordinate annotations, and of each measurement and the operations
    that read it.  The operations between REPEAT blocks, and the bodies of REPEAT blocks, are
    scheduled on their own, and their first moments share the moments before them when possible.
    """

    as_late_as_possible: bool
    shorten_lifetimes: bool  # place resets late in ASAP, and measurements early in ALAP
    single_qubit_gates_take_time: bool

    def get_schedule(self, circuit: StimCircuitLike) -> Schedule:
        stim_circuit = _as_stim_circuit(circuit)
        moments: list[int | _RepeatSchedule] = []
        last_moment = -1
        last_timed: set[int] = set()  # the qubits of timed operations in the last moment

        def place(first_timed: set[int]) -> int:
            """The moment in which a part starts, sharing the last moment if possible."""
            if last_moment >= 0 and first_timed.isdisjoint(last_timed):
                return last_moment
            return last_moment + 1

        segment: list[stim.CircuitInstruction] = []
        for instruction in [*stim_circuit, None]:
            if isinstance(instruction, stim.CircuitInstruction):
                segment.append(instruction)
                continue
            operations = _get_operations(segment)
            segment = []
            if operations:
                local = self._get_segment_moments(operations)
                timed = [
                    set(operation.qubits)
                    if operation.is_gate and self.takes_time(operation.instruction)
                    else set()
                    for operation in operations
                ]
                end = max(local)
                first = set().union(*(qubits for qubits, m in zip(timed, local) if m == 0))
                last = set().union(*(qubits for qubits, m in zip(timed, local) if m == end))
                offset = place(first)
                moments.extend(offset + m for m in local)
                if end == 0 and offset == last_moment:
                    last |= last_timed
                last_moment, last_timed = offset + end, last
            if instruction is None:
                break
            body = self.get_schedule(instruction.body_copy())
            summary = _summarize(body, self.takes_time)
            duration = summary.duration
            if not duration:
                moments.append(_RepeatSchedule(body, max(last_moment, 0), 0))
                continue
            shared = summary.first_timed.isdisjoint(summary.last_timed)
            period = duration - 1 if shared else duration
            start = place(summary.first_timed)
            end = start + (instruction.repeat_count - 1) * period + duration - 1
            last = set(summary.last_timed)
            if end == last_moment:
                last |= last_timed
            moments.append(_RepeatSchedule(body, start, period))
            last_moment, last_timed = end, last
        return Schedule._from_placements(stim_circuit, moments)

    def takes_time(self, op: stim.CircuitInstruction) -> bool:
        return _takes_time(op, self.single_qubit_gates_take_time)

    def __repr__(self) -> str:
        args = []
        if not self.shorten_lifetimes:
            args.append(
                "early_measurements=False" if self.as_late_as_possible else "late_resets=False"
            )
        if self.single_qubit_gates_take_time:
            args.append("single_qubit_gates_take_time=True")
        return f"Scheduler.{'alap' if self.as_late_as_possible else 'asap'}({', '.join(args)})"

    def _get_segment_moments(self, operations: Sequence[_Operation]) -> list[int]:
        """The moments of the operations of a segment."""
        # an ELSE_CORRELATED_ERROR joins the node of the error chain before it
        nodes: list[int] = []  # the node of each operation
        num_nodes = 0
        for operation in operations:
            if operation.instruction.name == "ELSE_CORRELATED_ERROR" and nodes:
                nodes.append(nodes[-1])
            else:
                nodes.append(num_nodes)
                num_nodes += 1
        edges = self._get_edges(operations, nodes)
        fixed = {}
        if self.shorten_lifetimes:
            # place measurements (or resets) first, as early (or late) as possible
            names = JUST_MEASURE_OPS if self.as_late_as_possible else JUST_RESET_OPS
            layers = _get_layers(num_nodes, edges, reverse=not self.as_late_as_possible)
            fixed = {
                node: layers[node]
                for node, operation in zip(nodes, operations)
                if operation.instruction.name in names
            }
        layers = _get_layers(num_nodes, edges, reverse=self.as_late_as_possible, fixed=fixed)
        return [layers[node] for node in nodes]

    def _get_edges(
        self, operations: Sequence[_Operation], nodes: Sequence[int]
    ) -> list[tuple[int, int, int]]:
        """Constraints ``(earlier, later, gap)``, which place a node ``gap`` or more layers later.

        A node follows the previous node on each of its qubits, and a node that takes time follows
        the previous node that takes time on each of its qubits by at least one layer.  A node that
        reads a measurement follows the node of that measurement, and a detector or coordinate
        annotation follows the previous one.
        """
        edges = []
        last_node: dict[int, int] = {}
        last_timed_node: dict[int, int] = {}
        record_nodes: list[int] = []
        last_coordinate_node: int | None = None
        for node, operation in zip(nodes, operations):
            timed = operation.is_gate and self.takes_time(operation.instruction)
            for qubit in operation.qubits:
                if last_node.get(qubit, node) != node:
                    edges.append((last_node[qubit], node, 0))
                if timed and qubit in last_timed_node:
                    edges.append((last_timed_node[qubit], node, 1))
                last_node[qubit] = node
                if timed:
                    last_timed_node[qubit] = node
            edges.extend(
                (record_nodes[record], node, 0) for record in operation.records if record >= 0
            )
            if operation.instruction.name in _COORDINATE_ANNOTATIONS:
                if last_coordinate_node is not None:
                    edges.append((last_coordinate_node, node, 0))
                last_coordinate_node = node
            record_nodes.extend([node] * operation.num_records)
        return edges


####################################################################################################
# operations and segments


@dataclasses.dataclass(frozen=True)
class _Operation:
    """An operation of a circuit, and the properties that schedules depend on."""

    instruction: stim.CircuitInstruction
    qubits: tuple[int, ...]  # qubits of a gate, observable, or noise, in fixed order
    touched_qubits: tuple[int, ...]  # the qubits whose state this operation depends on
    is_gate: bool
    is_collapsing: bool  # whether this is a measurement or reset
    num_records: int  # the number of measurement records that this operation produces
    records: tuple[int, ...]  # the records that this reads (relative to the operation, or as set)

    @staticmethod
    def from_instruction(instruction: stim.CircuitInstruction) -> _Operation:
        """The operation of an instruction with one gate application, annotation, or channel."""
        name = instruction.name
        targets = instruction.targets_copy()
        qubits = tuple(
            dict.fromkeys(
                target.qubit_value
                for target in targets
                if target.qubit_value is not None and not target.is_combiner
            )
        )
        is_gate = op_type(name) in GATE_OP_TYPES
        is_observable = name == "OBSERVABLE_INCLUDE"
        return _Operation(
            instruction,
            qubits if is_gate or is_observable or op_type(name) == NOISE else (),
            qubits
            if (is_gate and not involves_classical_bits(instruction)) or is_observable
            else (),
            is_gate,
            name in COLLAPSING_OPS,
            len(instruction.target_groups()) if _produces_measurements(name) else 0,
            tuple(target.value for target in targets if target.is_measurement_record_target),
        )


@dataclasses.dataclass
class _Segment:
    """The operations of a segment of a circuit (between REPEAT blocks), and their moments.

    The records that operations read are counted from the start of the segment, so records before
    the segment are negative.
    """

    instructions: list[stim.CircuitInstruction]
    num_operations: list[int]  # the number of operations of each instruction
    operations: list[_Operation]
    moments: list[int]


def _get_segments(
    circuit: stim.Circuit, schedule: Schedule | None
) -> Iterator[_Segment | tuple[stim.CircuitRepeatBlock, _RepeatSchedule | None]]:
    """The segments of a circuit, and its REPEAT blocks with their placements, in order.

    Without a schedule, every operation is in moment zero.
    """
    moments = iter(() if schedule is None else schedule._moments)
    segment = _Segment([], [], [], [])
    num_records = 0
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            yield segment
            repeat = next(moments, None)
            assert repeat is None or isinstance(repeat, _RepeatSchedule)
            yield instruction, repeat
            segment = _Segment([], [], [], [])
            num_records = 0
            continue
        operations = _get_operations([instruction], num_records)
        segment.instructions.append(instruction)
        segment.num_operations.append(len(operations))
        for info in operations:
            num_records += info.num_records
            moment = next(moments, 0)
            assert isinstance(moment, int)
            segment.operations.append(info)
            segment.moments.append(moment)
    yield segment


def _get_operations(
    instructions: Sequence[stim.CircuitInstruction], num_records: int = 0
) -> list[_Operation]:
    """The operations of instructions, given the number of records of their segment before them.

    The records that operations read are counted from the start of the segment.
    """
    operations = []
    for instruction in instructions:
        for operation in _get_instruction_operations(instruction):
            if operation.records:
                records = tuple(num_records + record for record in operation.records)
                operation = dataclasses.replace(operation, records=records)
            num_records += operation.num_records
            operations.append(operation)
    return operations


@functools.lru_cache(maxsize=4096)
def _get_instruction_operations(instruction: stim.CircuitInstruction) -> tuple[_Operation, ...]:
    """The operations of an instruction, with records relative to each operation.

    Instructions often recur, as in the rounds of an error-correction circuit, and stim builds
    instructions slowly, so the operations of recent instructions are cached.
    """
    return tuple(_Operation.from_instruction(op) for op in _split_operations(instruction))


def _split_operations(instruction: stim.CircuitInstruction) -> list[stim.CircuitInstruction]:
    """Split an instruction into gate applications, or keep an annotation or noise channel whole."""
    if instruction.name == "TICK":
        return []
    if op_type(instruction.name) not in GATE_OP_TYPES:
        return [instruction]
    if op_type(instruction.name) in (CLIFFORD_PP, JUST_MEASURE_PP):
        # restore the combiners of each Pauli product, which target_groups drops
        groups = [
            [target for pauli in product for target in (stim.target_combiner(), pauli)][1:]
            for product in instruction.target_groups()
        ]
    else:
        groups = instruction.target_groups()
    args = instruction.gate_args_copy()
    return [
        stim.CircuitInstruction(instruction.name, group, args, tag=instruction.tag)
        for group in groups
    ]


@functools.cache
def _produces_measurements(name: str) -> bool:
    """Does an instruction produce measurement records?"""
    return stim.gate_data(name).produces_measurements


def _takes_time(op: stim.CircuitInstruction, single_qubit_gates_take_time: bool) -> bool:
    """Does one gate application take time, for a built-in scheduler?"""
    return (
        op_type(op.name) in GATE_OP_TYPES
        and not involves_classical_bits(op)
        and (single_qubit_gates_take_time or op_type(op.name) != CLIFFORD_1Q)
    )


def _as_stim_circuit(circuit: StimCircuitLike) -> stim.Circuit:
    """Copy a circuit into a stim.Circuit."""
    stim_circuit = stim.Circuit()
    for index in range(len(circuit)):
        stim_circuit.append(circuit[index])
    return stim_circuit


####################################################################################################
# summaries of schedules


@dataclasses.dataclass
class _Summary:
    """The number of moments of a schedule, and the qubits of timed operations at its ends.

    The qubits of timed operations are those in the first and in the last moment of the schedule.
    """

    duration: int
    first_timed: set[int]
    last_timed: set[int]


def _summarize(
    schedule: Schedule,
    takes_time: Callable[[stim.CircuitInstruction], bool],
    *,
    check: bool = False,
) -> _Summary:
    """Summarize a schedule, and, if ``check``, check that it keeps the meaning of its circuit."""
    last_moment = -1  # the last moment of the parts of the circuit so far
    last_timed: set[int] = set()  # the qubits of timed operations in that moment
    first_timed: set[int] = set()
    for item in _get_segments(schedule._circuit, schedule):
        if isinstance(item, _Segment):
            if not item.operations:
                continue
            timed_at: dict[int, set[int]] = {}
            for operation, moment in zip(item.operations, item.moments):
                if operation.is_gate and takes_time(operation.instruction):
                    qubits = timed_at.setdefault(moment, set())
                    if check and not qubits.isdisjoint(operation.qubits):
                        qubit = min(qubits & set(operation.qubits))
                        raise ValueError(
                            f"Qubit {qubit} has two operations that take time in moment {moment}"
                        )
                    qubits.update(operation.qubits)
            if check:
                _check_segment(item)
            start, end = min(item.moments), max(item.moments)
            if check and start < last_moment:
                raise ValueError(
                    f"{item.instructions[0]} starts before the end of the REPEAT block before it"
                )
            if (
                start == last_moment
                and check
                and not last_timed.isdisjoint(timed_at.get(start, ()))
            ):
                qubit = min(last_timed & timed_at[start])
                raise ValueError(
                    f"Qubit {qubit} has two operations that take time in moment {start}"
                )
            first_timed |= timed_at.get(0, set())
            last = set(timed_at.get(end, ()))
            if end == last_moment:
                last |= last_timed
            last_moment, last_timed = max(last_moment, end), last
            continue

        block, repeat = item
        assert repeat is not None
        summary = _summarize(repeat.schedule, takes_time, check=check)
        if not summary.duration:
            if check and repeat.period:
                raise ValueError("A REPEAT block without operations must have a period of zero")
            continue
        end = repeat.start + (block.repeat_count - 1) * repeat.period + summary.duration - 1
        if check:
            if repeat.start < last_moment:
                raise ValueError(f"A REPEAT block starts before moment {last_moment}")
            if repeat.period < summary.duration - 1:
                raise ValueError("Consecutive iterations of a REPEAT block share several moments")
            shared = summary.first_timed & summary.last_timed
            if block.repeat_count > 1 and repeat.period == summary.duration - 1 and shared:
                raise ValueError(
                    f"Qubit {min(shared)} has two operations that take time in the moment shared"
                    " by two iterations of a REPEAT block"
                )
            shared = last_timed & summary.first_timed
            if repeat.start == last_moment and shared:
                raise ValueError(
                    f"Qubit {min(shared)} has two operations that take time in moment {last_moment}"
                )
        if repeat.start == 0:
            first_timed |= summary.first_timed
        last = set(summary.last_timed)
        if end == last_moment:
            last |= last_timed
        last_moment, last_timed = end, last
    return _Summary(last_moment + 1, first_timed, last_timed)


def _check_segment(segment: _Segment) -> None:
    """Check the order of the operations of a segment of a circuit."""
    last_moment: dict[int, int] = {}  # by qubit
    record_moments: list[int] = []  # the moment of each measurement record of the segment
    coordinate_moment = 0
    for index, (operation, moment) in enumerate(zip(segment.operations, segment.moments)):
        instruction = operation.instruction
        for qubit in operation.qubits:
            if moment < last_moment.get(qubit, 0):
                raise ValueError(f"Operations on qubit {qubit} change order at {instruction}")
            last_moment[qubit] = moment
        for record in operation.records:
            if record >= 0 and moment < record_moments[record]:
                raise ValueError(f"{instruction} precedes a measurement that it reads")
        if instruction.name in _COORDINATE_ANNOTATIONS:
            if moment < coordinate_moment:
                raise ValueError(f"{instruction} changes the coordinates of annotations")
            coordinate_moment = moment
        if (
            instruction.name == "ELSE_CORRELATED_ERROR"
            and index
            and moment != segment.moments[index - 1]
        ):
            raise ValueError(f"{instruction} must be in the moment of its error chain")
        record_moments.extend([moment] * operation.num_records)


####################################################################################################
# layers of operations


def _get_layers(
    num_nodes: int,
    edges: Sequence[tuple[int, int, int]],
    *,
    reverse: bool,
    fixed: Mapping[int, int] | None = None,
) -> list[int]:
    """Place nodes in layers as early (or, if ``reverse``, as late) as possible.

    Layers count moments from the start of a segment.  Fixed nodes keep their given layers, which
    must come from a valid schedule with the fewest layers.
    """
    incoming: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for earlier, later, gap in edges:
        if reverse:
            incoming[earlier].append((later, gap))
        else:
            incoming[later].append((earlier, gap))
    order = range(num_nodes - 1, -1, -1) if reverse else range(num_nodes)

    def get_depths(fixed_depths: Mapping[int, int]) -> list[int]:
        """Moments from the start (or, if reverse, the end) of the segment."""
        depths = [0] * num_nodes
        for index in order:
            if index in fixed_depths:
                depths[index] = fixed_depths[index]
            else:
                depths[index] = max(
                    (depths[other] + gap for other, gap in incoming[index]), default=0
                )
        return depths

    depths = get_depths({})
    num_layers = max(depths, default=0) + 1
    fixed = fixed or {}
    if reverse:
        depths = get_depths({index: num_layers - 1 - layer for index, layer in fixed.items()})
        return [num_layers - 1 - depth for depth in depths]
    return get_depths(fixed)


####################################################################################################
# laying out a schedule as a circuit


class _Layout:
    """Lays out scheduled circuits in moments, keeping track of the order of measurements."""

    def __init__(self) -> None:
        self.num_records = 0  # records laid out so far, which is the same in both orders
        self.new_to_old: list[int] = []
        self.old_to_new: dict[int, int] = {}

    def lay_out(self, schedule: Schedule, moment: int) -> tuple[stim.Circuit, int]:
        """Lay out a scheduled circuit, starting in a moment, and return its last moment.

        The layout starts after the records laid out so far.
        """
        circuit = stim.Circuit()
        for item in _get_segments(schedule._circuit, schedule):
            if isinstance(item, _Segment):
                moment = self.lay_out_segment(item, circuit, moment)
            else:
                assert item[1] is not None
                moment = self.lay_out_block(item[0], item[1], circuit, moment)
        return circuit, moment

    def lay_out_segment(self, segment: _Segment, circuit: stim.Circuit, moment: int) -> int:
        """Lay out the operations of a segment by moment, and return the last moment."""
        start = self.num_records
        old_starts = [start]
        for operation in segment.operations:
            old_starts.append(old_starts[-1] + operation.num_records)
        order = sorted(range(len(segment.operations)), key=lambda k: (segment.moments[k], k))
        for index in order:
            operation, old_start = segment.operations[index], old_starts[index]
            for _ in range(segment.moments[index] - moment):
                circuit.append("TICK")
            moment = segment.moments[index]
            records = iter(operation.records)
            targets = []
            for target in operation.instruction.targets_copy():
                if target.is_measurement_record_target:
                    new_record = self.old_to_new[start + next(records)]
                    target = stim.target_rec(new_record - self.num_records)
                targets.append(target)
            instruction = operation.instruction
            circuit.append(
                stim.CircuitInstruction(
                    instruction.name, targets, instruction.gate_args_copy(), tag=instruction.tag
                )
            )
            for old_record in range(old_start, old_start + operation.num_records):
                self.old_to_new[old_record] = self.num_records
                self.new_to_old.append(old_record)
                self.num_records += 1
        return moment

    def lay_out_block(
        self,
        block: stim.CircuitRepeatBlock,
        repeat: _RepeatSchedule,
        circuit: stim.Circuit,
        moment: int,
    ) -> int:
        """Lay out a REPEAT block, and return its last moment.

        Each iteration after the first starts with the TICKs that separate it from the iteration
        before it.  The first iterations are written out if they differ from later iterations.
        """
        duration = repeat.schedule._duration
        if not duration:
            circuit.append(
                stim.CircuitRepeatBlock(block.repeat_count, stim.Circuit(), tag=block.tag)
            )
            return moment
        body = block.body_copy()
        records_per_iteration = body.num_measurements
        reach = _get_reach(body)
        # iterations from this one on read only records of earlier iterations of the block
        num_distinct = (
            1
            if reach <= 0 or records_per_iteration == 0
            else 1 + math.ceil(reach / records_per_iteration)
        )
        num_distinct = min(max(num_distinct, 2), block.repeat_count)
        iterations: list[tuple[stim.Circuit, int]] = []  # the layout of iterations, and counts
        start = self.num_records
        for index in range(num_distinct):
            num_ticks = repeat.start - moment if index == 0 else repeat.period - duration + 1
            start = self.num_records
            iteration = stim.Circuit()
            for _ in range(num_ticks):
                iteration.append("TICK")
            iteration += self.lay_out(repeat.schedule, 0)[0]
            if iterations and iterations[-1][0] == iteration:
                iterations[-1] = (iteration, iterations[-1][1] + 1)
            else:
                iterations.append((iteration, 1))

        # the remaining iterations have the same layout as the last
        num_remaining = block.repeat_count - num_distinct
        iterations[-1] = (iterations[-1][0], iterations[-1][1] + num_remaining)
        offsets = [old - start for old in self.new_to_old[start:]]
        for _ in range(num_remaining):
            start = self.num_records
            for offset in offsets:
                self.old_to_new[start + offset] = self.num_records
                self.new_to_old.append(start + offset)
                self.num_records += 1

        # an iteration written out keeps the tag of the block in a block of its own, if it has one
        for index, (iteration, count) in enumerate(iterations):
            if index + 1 < len(iterations) and count == 1 and not block.tag:
                circuit += iteration
            else:
                circuit.append(stim.CircuitRepeatBlock(count, iteration, tag=block.tag))
        return repeat.start + (block.repeat_count - 1) * repeat.period + duration - 1


def _get_reach(circuit: stim.Circuit) -> int:
    """How many records before the start of a circuit its instructions read, at most."""
    reach = 0
    num_records = 0
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            reach = max(reach, _get_reach(instruction.body_copy()) - num_records)
            num_records += instruction.body_copy().num_measurements * instruction.repeat_count
            continue
        for target in instruction.targets_copy():
            if target.is_measurement_record_target:
                reach = max(reach, -target.value - num_records)
        if _produces_measurements(instruction.name):
            num_records += len(instruction.target_groups())
    return reach
