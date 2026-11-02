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
import itertools
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
    each operation in a moment, numbered from zero within each segment of the circuit: the part of
    the circuit, or of the body of a REPEAT block, that lies between REPEAT blocks.  The boundaries
    of REPEAT blocks and of their iterations are thus boundaries between moments, and the body of
    each REPEAT block has a schedule of its own, which each iteration follows.

    Every moment lasts one unit of time, including a moment without operations, and a qubit idles in
    each moment unless an operation that takes time acts on it (see :meth:`Scheduler.takes_time`).
    A REPEAT block lasts as long as all its iterations, and a segment lasts until the end of its
    last moment with a gate: the moments after it, which have only annotations and noise channels,
    take no time, and nor does a segment without gates.

    A schedule is immutable.

    Args:
        circuit: The circuit to schedule.
        moments: In circuit order, the moment of each operation of the circuit outside REPEAT
            blocks, and a schedule of the body of each REPEAT block.

    Raises:
        ValueError: If ``moments`` does not have one nonnegative integer moment for each operation
            and one schedule of its body for each REPEAT block.
    """

    __slots__ = ("_circuit", "_duration", "_moments")

    _circuit: stim.Circuit
    _moments: tuple[int | Schedule, ...]
    _duration: int

    def __init__(self, circuit: StimCircuitLike, moments: Sequence[int | Schedule]) -> None:
        stim_circuit = _as_stim_circuit(circuit)
        operations = Schedule.get_operations(stim_circuit)
        if len(operations) != len(moments):
            raise ValueError(
                f"A schedule needs {len(operations)} entries (one per operation or REPEAT block),"
                f" but got {len(moments)}"
            )
        entries: list[int | Schedule] = []
        for operation, moment in zip(operations, moments):
            if isinstance(operation, stim.CircuitRepeatBlock):
                if not isinstance(moment, Schedule) or str(moment._circuit) != str(
                    operation.body_copy()
                ):
                    raise ValueError("Each REPEAT block needs a schedule of its body")
            else:
                moment = _as_moment(moment, f"The moment of {operation}")
            entries.append(moment)
        object.__setattr__(self, "_circuit", stim_circuit)
        object.__setattr__(self, "_moments", tuple(entries))
        duration = 0
        for item in _get_segments(stim_circuit, self):
            if isinstance(item, _Segment):
                duration += item.duration
            else:
                assert item[1] is not None
                duration += item[0].repeat_count * item[1]._duration
        object.__setattr__(self, "_duration", duration)

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
        """The number of moments of this schedule that take time, in all iterations of blocks."""
        return self._duration

    def get_moments(self) -> list[list[stim.CircuitInstruction] | Schedule]:
        """The operations in each moment, and the schedule of each REPEAT block, in order.

        The moments of each segment are listed in order, followed by the schedule of the body of
        the REPEAT block after the segment (if any).  Operations in a moment keep their order in the
        circuit.
        """
        moments: list[list[stim.CircuitInstruction] | Schedule] = []
        for item in _get_segments(self._circuit, self):
            if isinstance(item, _Segment):
                segment_moments: list[list[stim.CircuitInstruction]] = [
                    [] for _ in range(max(item.moments, default=-1) + 1)
                ]
                for operation, moment in zip(item.operations, item.moments):
                    segment_moments[moment].append(operation.instruction)
                moments.extend(segment_moments)
            else:
                assert item[1] is not None
                moments.append(item[1])
        return moments

    def to_circuit(self) -> tuple[stim.Circuit, tuple[int, ...]]:
        """The scheduled circuit, laid out in moments separated by TICKs.

        Operations are ordered by moment, and otherwise keep their order in the circuit, so a
        measurement moves only if this schedule puts it in an earlier moment than a measurement
        before it.  The detectors, observables, and classically controlled gates of the circuit read
        the same measurements as before.  A moment without operations is an extra TICK, a TICK
        precedes each REPEAT block that follows a moment, and each iteration of a REPEAT block ends
        with a TICK.  The first iterations of a REPEAT block are written out separately if they read
        measurements before the block at other offsets than later iterations read those of earlier
        iterations.

        Returns:
            The scheduled circuit, and the index in the original circuit of each of its
            measurements.
        """
        layout = _Layout()
        circuit, _ = layout.lay_out(self)
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
        two operations that take time in one moment.

        Raises:
            ValueError: If the schedule is not a valid schedule of the circuit.
        """
        if str(schedule._circuit) != str(_as_stim_circuit(circuit)):
            raise ValueError("The schedule is for a different circuit")
        _check_schedule(schedule, self.takes_time)

    @staticmethod
    def preserve_ticks(*, single_qubit_gates_take_time: bool = False) -> Scheduler:
        """A scheduler that keeps the moments of a circuit.

        Each TICK ends a moment that contains a gate, and a new moment also starts where an
        operation that takes time acts on a qubit that another operation that takes time already
        acts on in the moment.  Annotations and noise channels after a TICK join the next moment
        with a gate, or else form the last moment of their segment (see :class:`Schedule`), which
        takes no time.  As in every schedule, the boundaries of REPEAT blocks and of their
        iterations are also boundaries between moments, which adding noise to
        ``circuit.flattened()`` avoids.

        Args:
            single_qubit_gates_take_time: If False (the default), single-qubit unitary gates take no
                time, so a moment may contain any number of them on a qubit, which still idles.
        """
        return _TickScheduler(single_qubit_gates_take_time)

    @staticmethod
    def asap(*, late_resets: bool = True, single_qubit_gates_take_time: bool = False) -> Scheduler:
        """A scheduler that places every operation as early as possible, ignoring TICKs.

        Each segment of a circuit (see :class:`Schedule`), and the body of each REPEAT block, is
        scheduled on its own.

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
        moments: list[int | Schedule] = []
        for item in _get_segments(stim_circuit, None):
            if isinstance(item, _Segment):
                moments.extend(self._get_segment_moments(item))
            else:
                moments.append(self.get_schedule(item[0].body_copy()))
        return Schedule(stim_circuit, moments)

    def takes_time(self, op: stim.CircuitInstruction) -> bool:
        return _takes_time(op, self.single_qubit_gates_take_time)

    def __repr__(self) -> str:
        args = "single_qubit_gates_take_time=True" if self.single_qubit_gates_take_time else ""
        return f"Scheduler.preserve_ticks({args})"

    def _get_segment_moments(self, segment: _Segment) -> list[int]:
        """The moments of the operations of a segment."""
        moments: list[int] = []
        moment = 0
        occupied: set[int] = set()  # the qubits of operations that take time in the moment
        has_gate = False  # whether the moment has a gate
        closed = False  # whether a TICK has ended the moment
        operations = iter(segment.operations)
        for instruction, num_operations in zip(segment.instructions, segment.num_operations):
            if instruction.name == "TICK":
                closed = closed or has_gate
            for operation in itertools.islice(operations, num_operations):
                timed = operation.is_gate and self.takes_time(operation.instruction)
                if closed or (timed and not occupied.isdisjoint(operation.qubits)):
                    moment, occupied, has_gate, closed = moment + 1, set(), False, False
                if timed:
                    occupied.update(operation.qubits)
                has_gate = has_gate or operation.is_gate
                moments.append(moment)
        return moments


@dataclasses.dataclass(frozen=True)
class _LayerScheduler(Scheduler):
    """The schedulers made by :meth:`Scheduler.asap` and :meth:`Scheduler.alap`.

    Operations are placed in layers, which become moments, keeping the order of the operations on
    each qubit, of detectors and coordinate annotations, and of each measurement and the operations
    that read it.  Each segment of a circuit, and the body of each REPEAT block, is scheduled on its
    own.
    """

    as_late_as_possible: bool
    shorten_lifetimes: bool  # place resets late in ASAP, and measurements early in ALAP
    single_qubit_gates_take_time: bool

    def get_schedule(self, circuit: StimCircuitLike) -> Schedule:
        stim_circuit = _as_stim_circuit(circuit)
        moments: list[int | Schedule] = []
        for item in _get_segments(stim_circuit, None):
            if isinstance(item, _Segment):
                moments.extend(self._get_segment_moments(item.operations))
            else:
                moments.append(self.get_schedule(item[0].body_copy()))
        return Schedule(stim_circuit, moments)

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

    @property
    def duration(self) -> int:
        """The number of moments of the segment, up to its last moment with a gate."""
        return max(
            (moment + 1 for op, moment in zip(self.operations, self.moments) if op.is_gate),
            default=0,
        )


def _get_segments(
    circuit: stim.Circuit, schedule: Schedule | None
) -> Iterator[_Segment | tuple[stim.CircuitRepeatBlock, Schedule | None]]:
    """The segments of a circuit, and its REPEAT blocks with the schedules of their bodies.

    Without a schedule, every operation is in moment zero.
    """
    moments = iter(() if schedule is None else schedule._moments)
    segment = _Segment([], [], [], [])
    num_records = 0
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            yield segment
            repeat = next(moments, None)
            assert repeat is None or isinstance(repeat, Schedule)
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
# checks of schedules


def _check_schedule(
    schedule: Schedule, takes_time: Callable[[stim.CircuitInstruction], bool]
) -> None:
    """Check that a schedule keeps the meaning of its circuit."""
    for item in _get_segments(schedule._circuit, schedule):
        if isinstance(item, _Segment):
            _check_segment(item, takes_time)
        else:
            assert item[1] is not None
            _check_schedule(item[1], takes_time)


def _check_segment(
    segment: _Segment, takes_time: Callable[[stim.CircuitInstruction], bool]
) -> None:
    """Check the order of the operations of a segment of a circuit, and the qubits they occupy."""
    last_moment: dict[int, int] = {}  # by qubit
    occupied: dict[int, set[int]] = {}  # by moment, the qubits of operations that take time
    record_moments: list[int] = []  # the moment of each measurement record of the segment
    coordinate_moment = 0
    for index, (operation, moment) in enumerate(zip(segment.operations, segment.moments)):
        instruction = operation.instruction
        for qubit in operation.qubits:
            if moment < last_moment.get(qubit, 0):
                raise ValueError(f"Operations on qubit {qubit} change order at {instruction}")
            last_moment[qubit] = moment
        if operation.is_gate and takes_time(instruction):
            qubits = occupied.setdefault(moment, set())
            if not qubits.isdisjoint(operation.qubits):
                qubit = min(qubits & set(operation.qubits))
                raise ValueError(
                    f"Qubit {qubit} has two operations that take time in moment {moment}"
                )
            qubits.update(operation.qubits)
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

    def lay_out(self, schedule: Schedule) -> tuple[stim.Circuit, bool]:
        """Lay out a scheduled circuit after the records laid out so far.

        Returns:
            The layout, and whether it ends with a moment that a TICK has not ended.
        """
        circuit = stim.Circuit()
        is_open = False
        for item in _get_segments(schedule._circuit, schedule):
            if isinstance(item, _Segment):
                if item.operations:
                    self.lay_out_segment(item, circuit)
                    is_open = True
            else:
                assert item[1] is not None
                if is_open:
                    circuit.append("TICK")
                self.lay_out_block(item[0], item[1], circuit)
                is_open = False
        return circuit, is_open

    def lay_out_segment(self, segment: _Segment, circuit: stim.Circuit) -> None:
        """Lay out the operations of a segment by moment."""
        start = self.num_records
        old_starts = [start]
        for operation in segment.operations:
            old_starts.append(old_starts[-1] + operation.num_records)
        order = sorted(range(len(segment.operations)), key=lambda k: (segment.moments[k], k))
        moment = 0
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

    def lay_out_block(
        self, block: stim.CircuitRepeatBlock, schedule: Schedule, circuit: stim.Circuit
    ) -> None:
        """Lay out a REPEAT block, ending each iteration with a TICK.

        The first iterations are written out if they read measurements before the block at other
        offsets than later iterations read those of earlier iterations.
        """
        body = block.body_copy()
        records_per_iteration = body.num_measurements
        reach = _get_reach(body)
        # iterations from this one on read only records of earlier iterations of the block
        num_distinct = (
            1
            if reach <= 0 or records_per_iteration == 0
            else 1 + math.ceil(reach / records_per_iteration)
        )
        num_distinct = min(num_distinct, block.repeat_count)
        iterations: list[tuple[stim.Circuit, int]] = []  # the layout of iterations, and counts
        start = self.num_records
        for _ in range(num_distinct):
            start = self.num_records
            iteration, is_open = self.lay_out(schedule)
            if is_open:
                iteration.append("TICK")
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
