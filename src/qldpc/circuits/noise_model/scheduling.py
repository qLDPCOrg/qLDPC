# SPDX-License-Identifier: Apache-2.0

"""Schedules, which place the operations of a circuit in moments, and schedulers that make them.

An operation is a part of an instruction that acts independently on its qubits, such as one
application of a gate (see :class:`Schedule`).  A moment is a unit of time, in which each qubit is
either occupied by an operation that takes time or idle.  A :class:`~qldpc.circuits.NoiseModel` with
idle noise uses a schedule to find how many moments each qubit idles between its gates.
"""

from __future__ import annotations

import abc
import dataclasses
import functools
import itertools
import math
import operator
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, NoReturn, SupportsIndex

import stim

from ..common import StimCircuitLike
from ._operations import (
    CLIFFORD_1Q,
    CLIFFORD_PP,
    GATE_OP_TYPES,
    JUST_MEASURE_OPS,
    JUST_MEASURE_PP,
    JUST_RESET_OPS,
    MEASURE_OR_RESET_OPS,
    NOISE,
    involves_classical_bits,
    op_type,
)

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

# coordinate annotations, whose order sets the coordinates of detectors and qubits
_COORDINATE_ANNOTATIONS = frozenset({"DETECTOR", "QUBIT_COORDS", "SHIFT_COORDS"})

####################################################################################################
# schedules


class Schedule:
    """The moments of the operations of a circuit.

    The operations of a circuit, as listed by :meth:`get_operations`, are the parts of its
    instructions other than TICKs that act independently on their qubits.  A gate or noise channel
    has one operation per application: one target of a one-qubit gate or channel, one pair of
    targets of a two-qubit gate or channel, one Pauli product of an MPP or SPP gate, or one link of
    a chain of correlated errors.  An observable has one operation for its measurement records and
    one for each of its Pauli targets, and other annotations are one operation each.

    A schedule places each operation in a moment, numbered from zero within each segment of the
    circuit.  A segment is a maximal stretch of a circuit, or of the body of a REPEAT block, that
    contains no REPEAT block.  The boundaries of REPEAT blocks and of their iterations are thus
    boundaries between moments, and the body of each REPEAT block has a schedule of its own, which
    each iteration follows.

    Every moment lasts one unit of time, and a qubit idles in each moment unless an operation that
    takes time acts on it (see :meth:`Scheduler.takes_time`).  A segment with gates lasts until the
    end of the latest moment of its operations, and a REPEAT block lasts as long as all its
    iterations.  A segment without gates has no moments and takes no time, and its operations are in
    moment zero, whatever moments are given for them.

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
        # check that there is a moment for each operation and a schedule for each REPEAT body
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

        # add up the moments of the segments and of all iterations of REPEAT blocks
        duration = 0
        for segment_or_block in _get_segments(stim_circuit, self):
            if isinstance(segment_or_block, _Segment):
                duration += segment_or_block.duration
            else:
                block, body_schedule = segment_or_block
                assert body_schedule is not None
                duration += block.repeat_count * body_schedule._duration
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
        """The number of moments of this schedule, in all iterations of REPEAT blocks."""
        return self._duration

    def get_moments(self) -> list[list[stim.CircuitInstruction] | Schedule]:
        """The operations in each moment, and the schedule of each REPEAT block, in order.

        The moments of each segment are listed in order, followed by the schedule of the body of
        the REPEAT block after the segment (if any).  Operations in a moment keep their order in the
        circuit.  A segment without gates has no moments, so its operations are not listed.
        """
        # list the moments of each segment, and then the schedule of the REPEAT block after it
        moments: list[list[stim.CircuitInstruction] | Schedule] = []
        for segment_or_block in _get_segments(self._circuit, self):
            if isinstance(segment_or_block, _Segment):
                segment = segment_or_block
                segment_moments: list[list[stim.CircuitInstruction]] = [
                    [] for _ in range(segment.duration)
                ]
                if segment.duration:
                    for operation, moment in zip(segment.operations, segment.moments):
                        segment_moments[moment].append(operation.instruction)
                moments.extend(segment_moments)
            else:
                _, body_schedule = segment_or_block
                assert body_schedule is not None
                moments.append(body_schedule)
        return moments

    def to_circuit(self) -> tuple[stim.Circuit, tuple[int, ...]]:
        """The scheduled circuit, laid out in moments separated by TICKs.

        Operations are ordered by moment, and otherwise keep their order in the circuit, so a
        measurement moves only if this schedule puts it in an earlier moment than a measurement
        before it.  The detectors, observables, and classically controlled gates of the circuit read
        the same measurements as before.  A moment without operations is an extra TICK, a TICK
        precedes each REPEAT block that follows a moment, and each iteration of a REPEAT block that
        ends with a moment ends with a TICK.  The first iterations of a REPEAT block are written
        out separately if they read measurements before the block at other offsets than later
        iterations read those of earlier iterations.

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
    qubit idles.  The built-in schedulers are :class:`TickPreservingScheduler`,
    :class:`ASAPScheduler`, and :class:`ALAPScheduler`.  They place the gates of a circuit in
    moments, and each annotation or noise channel in the moment of the operation before it on its
    qubits, or, for an annotation without qubits such as a detector, of the last gate before it (or
    in the first moment if there is none).  An annotation or noise channel joins a later moment only
    to follow a measurement that it reads, or, if it is a coordinate annotation (a DETECTOR,
    QUBIT_COORDS, or SHIFT_COORDS annotation), the coordinate annotation before it.

    A subclass of a built-in scheduler may override :meth:`takes_time` and :meth:`validate`.  A new
    scheduling algorithm subclasses :class:`Scheduler` and implements :meth:`get_schedule`.
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

        The moments of the operations on each qubit, and those of coordinate annotations (DETECTOR,
        QUBIT_COORDS, and SHIFT_COORDS), may not decrease in circuit order.  An operation that reads
        a measurement may not precede that measurement, a correlated error chain stays in one
        moment, and no qubit has two operations that take time in one moment.

        Raises:
            ValueError: If the schedule is not a valid schedule of the circuit.
        """
        if str(schedule._circuit) != str(_as_stim_circuit(circuit)):
            raise ValueError("The schedule is for a different circuit")
        _check_schedule(schedule, self.takes_time)


@dataclasses.dataclass(frozen=True, kw_only=True, repr=False)
class TickPreservingScheduler(Scheduler):
    """A scheduler that keeps the moments of a circuit.

    Each TICK ends a moment that contains a gate, and a new moment also starts where an operation
    that takes time acts on a qubit that another operation that takes time already acts on in the
    moment.  As in every schedule, the boundaries of REPEAT blocks and of their iterations are also
    boundaries between moments, which adding noise to ``circuit.flattened()`` avoids.

    Args:
        single_qubit_gates_take_time: If False (the default), single-qubit unitary gates take no
            time, so a moment may contain any number of them on a qubit, which still idles.
    """

    single_qubit_gates_take_time: bool = False

    def get_schedule(self, circuit: StimCircuitLike) -> Schedule:
        # schedule each segment (see Schedule), and the body of each REPEAT block recursively
        stim_circuit = _as_stim_circuit(circuit)
        moments: list[int | Schedule] = []
        for segment_or_block in _get_segments(stim_circuit, None):
            if isinstance(segment_or_block, _Segment):
                moments.extend(self._get_segment_moments(segment_or_block))
            else:
                block, _ = segment_or_block
                moments.append(self.get_schedule(block.body_copy()))
        return Schedule(stim_circuit, moments)

    def takes_time(self, op: stim.CircuitInstruction) -> bool:
        return _takes_time(op, self.single_qubit_gates_take_time)

    def __repr__(self) -> str:
        return _get_repr(self)

    def _get_segment_moments(self, segment: _Segment) -> list[int]:
        """The moments of the operations of a segment."""
        # each operation is a node, except that a chain of correlated errors is one node (see
        # _get_nodes); a gate starts a new moment after a TICK, or where it takes time on a qubit
        # that an operation that takes time already occupies in the moment
        nodes = _get_nodes(segment.operations)
        gate_moments: dict[int, int] = {}  # the moment of each node with a gate
        moment = 0
        occupied: set[int] = set()  # the qubits of operations that take time in the moment
        has_gate = False  # whether a gate precedes the next operation in the segment
        closed = False  # whether a TICK has ended the moment
        operations = iter(zip(nodes, segment.operations))
        for instruction, num_operations in zip(segment.instructions, segment.num_operations):
            if instruction.name == "TICK":
                closed = has_gate
            for node, operation in itertools.islice(operations, num_operations):
                if not operation.is_gate:
                    continue
                takes_time = self.takes_time(operation.instruction)
                if closed or (takes_time and not occupied.isdisjoint(operation.qubits)):
                    moment, occupied, closed = moment + 1, set(), False
                if takes_time:
                    occupied.update(operation.qubits)
                has_gate = True
                gate_moments[node] = moment

        # annotations and noise channels join the moments of operations before them
        return _place_annotations(segment.operations, nodes, gate_moments)


class _LayerScheduler(Scheduler):
    """The algorithm of :class:`ASAPScheduler` and :class:`ALAPScheduler`.

    Each segment of a circuit (see :class:`Schedule`), and the body of each REPEAT block, is
    scheduled on its own.  The operations of a segment are grouped into nodes (see ``_get_nodes``),
    and each node is placed in a layer: the layers of a segment are its moments, numbered from its
    start.  The layers keep the order of the operations on each qubit, of coordinate annotations,
    and of each measurement and the operations that read it.
    """

    _as_late_as_possible: ClassVar[bool]
    single_qubit_gates_take_time: bool

    @property
    @abc.abstractmethod
    def _gates_moved_back(self) -> frozenset[str]:
        """The gates to move as late (or early) as the other gates allow.

        Every gate is first placed as early (or late) as possible, and then these gates move.
        """

    def get_schedule(self, circuit: StimCircuitLike) -> Schedule:
        # schedule each segment, and the body of each REPEAT block recursively
        stim_circuit = _as_stim_circuit(circuit)
        moments: list[int | Schedule] = []
        for segment_or_block in _get_segments(stim_circuit, None):
            if isinstance(segment_or_block, _Segment):
                moments.extend(self._get_segment_moments(segment_or_block.operations))
            else:
                block, _ = segment_or_block
                moments.append(self.get_schedule(block.body_copy()))
        return Schedule(stim_circuit, moments)

    def takes_time(self, op: stim.CircuitInstruction) -> bool:
        return _takes_time(op, self.single_qubit_gates_take_time)

    def _get_segment_moments(self, operations: Sequence[_Operation]) -> list[int]:
        """The moments of the operations of a segment."""
        # each operation is a node, except that a chain of correlated errors is one node, and the
        # edges between nodes constrain their order
        nodes = _get_nodes(operations)
        num_nodes = nodes[-1] + 1 if nodes else 0
        edges = self._get_edges(operations, nodes)

        # place the gates as early (or late) as possible
        layers = _get_layers(num_nodes, edges, reverse=self._as_late_as_possible)
        if self._gates_moved_back:
            # keep the other gates in place, and move each gate that only resets (or only measures)
            # as late (or early) as they allow
            fixed_layers = {
                node: layers[node]
                for node, operation in zip(nodes, operations)
                if operation.is_gate and operation.instruction.name not in self._gates_moved_back
            }
            layers = _get_layers(
                num_nodes, edges, reverse=not self._as_late_as_possible, fixed=fixed_layers
            )

        # place annotations and noise channels in the moments of operations before them
        gate_moments = {
            node: layers[node] for node, operation in zip(nodes, operations) if operation.is_gate
        }
        return _place_annotations(operations, nodes, gate_moments)

    def _get_edges(
        self, operations: Sequence[_Operation], nodes: Sequence[int]
    ) -> list[tuple[int, int, int]]:
        """Constraints ``(earlier, later, gap)``, which place a node ``gap`` or more layers later.

        A node follows the previous node on each of its qubits, and a node that takes time follows
        the previous node that takes time on each of its qubits by at least one layer.  A node that
        reads a measurement follows the node of that measurement, a coordinate annotation follows
        the previous one, and an annotation without qubits follows the previous gate, whose moment
        it joins.
        """
        edges = []
        last_node: dict[int, int] = {}  # by qubit
        last_node_taking_time: dict[int, int] = {}  # by qubit
        record_nodes: list[int] = []  # the node of each measurement record of the segment
        last_coordinate_node: int | None = None
        last_gate_node: int | None = None
        for node, operation in zip(nodes, operations):
            # follow the previous nodes on the qubits of the operation
            takes_time = operation.is_gate and self.takes_time(operation.instruction)
            for qubit in operation.qubits:
                if last_node.get(qubit, node) != node:
                    edges.append((last_node[qubit], node, 0))
                if takes_time and qubit in last_node_taking_time:
                    edges.append((last_node_taking_time[qubit], node, 1))
                last_node[qubit] = node
                if takes_time:
                    last_node_taking_time[qubit] = node

            # follow the measurements that the operation reads, and the previous coordinate
            # annotation if it is one
            edges.extend(
                (record_nodes[record], node, 0) for record in operation.records if record >= 0
            )
            if operation.instruction.name in _COORDINATE_ANNOTATIONS:
                if last_coordinate_node is not None:
                    edges.append((last_coordinate_node, node, 0))
                last_coordinate_node = node

            # an annotation without qubits follows the previous gate
            if operation.is_gate:
                last_gate_node = node
            elif not operation.qubits and last_gate_node not in (None, node):
                edges.append((last_gate_node, node, 0))
            record_nodes.extend([node] * operation.num_records)
        return edges


@dataclasses.dataclass(frozen=True, kw_only=True, repr=False)
class ASAPScheduler(_LayerScheduler):
    """A scheduler that places every gate as early as possible, ignoring TICKs.

    Each segment of a circuit (see :class:`Schedule`), and the body of each REPEAT block, is
    scheduled on its own.

    Args:
        late_resets: If True (the default), then move each reset (R, RX, RY, or RZ) as late as the
            other gates allow, to shorten the time that qubits idle after a reset.  Gates that
            measure and then reset, such as MR, stay in place.
        single_qubit_gates_take_time: As in :class:`TickPreservingScheduler`.
    """

    _as_late_as_possible: ClassVar[bool] = False
    late_resets: bool = True
    single_qubit_gates_take_time: bool = False

    def __repr__(self) -> str:
        return _get_repr(self)

    @property
    def _gates_moved_back(self) -> frozenset[str]:
        return JUST_RESET_OPS if self.late_resets else frozenset()


@dataclasses.dataclass(frozen=True, kw_only=True, repr=False)
class ALAPScheduler(_LayerScheduler):
    """A scheduler that places every gate as late as possible, ignoring TICKs.

    REPEAT blocks are handled as in :class:`ASAPScheduler`.

    Args:
        early_measurements: If True (the default), then move each measurement (M, MX, MY, MZ, MPP,
            MXX, MYY, or MZZ) as early as the other gates allow, to shorten the time that qubits
            idle before a measurement.  Gates that measure and then reset, such as MR, stay in
            place.
        single_qubit_gates_take_time: As in :class:`TickPreservingScheduler`.
    """

    _as_late_as_possible: ClassVar[bool] = True
    early_measurements: bool = True
    single_qubit_gates_take_time: bool = False

    def __repr__(self) -> str:
        return _get_repr(self)

    @property
    def _gates_moved_back(self) -> frozenset[str]:
        return JUST_MEASURE_OPS if self.early_measurements else frozenset()


def _get_repr(scheduler: DataclassInstance) -> str:
    """A constructor-like repr of a built-in scheduler, which omits default arguments."""
    args = [
        f"{field.name}={getattr(scheduler, field.name)!r}"
        for field in dataclasses.fields(scheduler)
        if getattr(scheduler, field.name) != field.default
    ]
    return f"{type(scheduler).__name__}({', '.join(args)})"


def _get_nodes(operations: Sequence[_Operation]) -> list[int]:
    """The node of each operation of a segment, numbered from zero.

    A node is a group of operations that a scheduler places in one moment: each operation is a node
    of its own, except that a chain of correlated errors (an ``E`` and the ``ELSE_CORRELATED_ERROR``
    operations after it) is one node.
    """
    nodes: list[int] = []
    for operation in operations:
        if operation.instruction.name == "ELSE_CORRELATED_ERROR" and nodes:
            nodes.append(nodes[-1])
        else:
            nodes.append(nodes[-1] + 1 if nodes else 0)
    return nodes


def _place_annotations(
    operations: Sequence[_Operation], nodes: Sequence[int], gate_moments: Mapping[int, int]
) -> list[int]:
    """The moment of each operation of a segment, given the moment of each node with a gate.

    The other nodes (annotations, noise channels, and chains of correlated errors) join the latest
    moment of the operations before them on their qubits, or, without qubits, of the last gate
    before them, or else the first moment.  They also join a later moment to follow the
    measurements that they read, and a coordinate annotation follows the one before it.
    """
    moments: list[int] = []
    last_moment: dict[int, int] = {}  # by qubit
    record_moments: list[int] = []  # the moment of each measurement record of the segment
    gate_moment = 0  # the moment of the last gate
    coordinate_moment = 0  # the moment of the last coordinate annotation
    for node, group in itertools.groupby(zip(nodes, operations), key=operator.itemgetter(0)):
        # a node with a gate keeps its moment, and another node joins the latest moment of the
        # operations that it follows
        members = [operation for _, operation in group]
        moment = gate_moments.get(node)
        if moment is None:
            moment = 0
            for operation in members:
                if operation.qubits:
                    moment = max(moment, *(last_moment.get(q, 0) for q in operation.qubits))
                else:
                    moment = max(moment, gate_moment)
                for record in operation.records:
                    if record >= 0:
                        moment = max(moment, record_moments[record])
                if operation.instruction.name in _COORDINATE_ANNOTATIONS:
                    moment = max(moment, coordinate_moment)

        # record the moment of the node for the operations that follow it
        for operation in members:
            last_moment.update(dict.fromkeys(operation.qubits, moment))
            record_moments.extend([moment] * operation.num_records)
            if operation.is_gate:
                gate_moment = moment
            if operation.instruction.name in _COORDINATE_ANNOTATIONS:
                coordinate_moment = moment
            moments.append(moment)
    return moments


####################################################################################################
# operations and segments


@dataclasses.dataclass(frozen=True)
class _Operation:
    """An operation of a circuit (see :class:`Schedule`), with properties for scheduling and noise.

    An operation acts on its ``qubits``, which are those of a gate, observable, or noise channel.
    It depends on the state of its ``qubits_depended_on``, which are those of a gate without
    classical control or of an observable, so noise on these qubits must come before it.  A noise
    channel or a classically controlled Pauli gate acts on its qubits without depending on their
    state.

    The measurement records that an operation reads are counted back from the operation (so they
    are negative), or, in operations from ``_get_operations``, from the start of their segment.
    """

    instruction: stim.CircuitInstruction
    qubits: tuple[int, ...]  # the qubits that this operation acts on, in order
    qubits_depended_on: tuple[int, ...]  # the qubits whose state this operation depends on
    is_gate: bool
    is_measurement_or_reset: bool
    num_records: int  # the number of measurement records that this operation produces
    records: tuple[int, ...]  # the measurement records that this operation reads

    @staticmethod
    def from_instruction(instruction: stim.CircuitInstruction) -> _Operation:
        """The operation of an instruction with one gate application, annotation, or channel."""
        # the distinct qubits of the targets, in order
        name = instruction.name
        targets = instruction.targets_copy()
        qubits = tuple(
            dict.fromkeys(
                target.qubit_value
                for target in targets
                if target.qubit_value is not None and not target.is_combiner
            )
        )

        # gates, observables, and noise channels act on their qubits, and gates without classical
        # control and observables depend on their state
        is_gate = op_type(name) in GATE_OP_TYPES
        is_observable = name == "OBSERVABLE_INCLUDE"
        return _Operation(
            instruction,
            qubits if is_gate or is_observable or op_type(name) == NOISE else (),
            qubits
            if (is_gate and not involves_classical_bits(instruction)) or is_observable
            else (),
            is_gate,
            name in MEASURE_OR_RESET_OPS,
            len(instruction.target_groups()) if _produces_measurements(name) else 0,
            tuple(target.value for target in targets if target.is_measurement_record_target),
        )


@dataclasses.dataclass
class _Segment:
    """The operations of a segment of a circuit, and their moments.

    A segment is a maximal stretch of a circuit, or of the body of a REPEAT block, that contains no
    REPEAT block.  The records that operations read are counted from the start of the segment, so
    records before the segment are negative.
    """

    instructions: list[stim.CircuitInstruction]
    num_operations: list[int]  # the number of operations of each instruction
    operations: list[_Operation]
    moments: list[int]

    @property
    def duration(self) -> int:
        """The number of moments of the segment, which has none if it has no gates."""
        return max(self.moments) + 1 if any(op.is_gate for op in self.operations) else 0

    def close(self) -> _Segment:
        """Finish a segment: one without gates has no moments, so put its operations in moment 0."""
        if not self.duration:
            self.moments = [0] * len(self.moments)
        return self


def _get_segments(
    circuit: stim.Circuit, schedule: Schedule | None
) -> Iterator[_Segment | tuple[stim.CircuitRepeatBlock, Schedule | None]]:
    """The segments and REPEAT blocks of a circuit in order, each block with a schedule of its body.

    A segment (see ``_Segment``) precedes each REPEAT block and ends the circuit.  Without a
    schedule, every operation is in moment zero, and REPEAT blocks have no schedules.
    """
    moments = iter(() if schedule is None else schedule._moments)
    segment = _Segment([], [], [], [])
    num_records = 0
    for instruction in circuit:
        # a REPEAT block ends the current segment, and a new one starts after it
        if isinstance(instruction, stim.CircuitRepeatBlock):
            yield segment.close()
            body_schedule = next(moments, None)
            assert body_schedule is None or isinstance(body_schedule, Schedule)
            yield instruction, body_schedule
            segment = _Segment([], [], [], [])
            num_records = 0
            continue

        # other instructions join the segment with the moments of their operations
        operations = _get_operations([instruction], num_records)
        segment.instructions.append(instruction)
        segment.num_operations.append(len(operations))
        for operation in operations:
            num_records += operation.num_records
            moment = next(moments, 0)
            assert isinstance(moment, int)
            segment.operations.append(operation)
            segment.moments.append(moment)
    yield segment.close()


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
    """Split an instruction into the operations that act independently on their qubits.

    A gate or noise channel splits into its applications (such as one qubit of ``X_ERROR``, one
    pair of ``DEPOLARIZE2``, or one chain link of ``E``), and an observable into its measurement
    records and each of its Pauli targets.  Other annotations stay whole.
    """
    if instruction.name == "TICK":
        return []

    # the targets of each operation
    if instruction.name == "OBSERVABLE_INCLUDE":
        # an observable is the product of its parts, and one without targets still declares it
        targets = instruction.targets_copy()
        records = [target for target in targets if target.is_measurement_record_target]
        paulis = [[target] for target in targets if not target.is_measurement_record_target]
        groups = [records] + paulis if records or not paulis else paulis
    elif op_type(instruction.name) not in (*GATE_OP_TYPES, NOISE):
        return [instruction]
    elif op_type(instruction.name) in (CLIFFORD_PP, JUST_MEASURE_PP):
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
    for segment_or_block in _get_segments(schedule._circuit, schedule):
        if isinstance(segment_or_block, _Segment):
            _check_segment(segment_or_block, takes_time)
        else:
            _, body_schedule = segment_or_block
            assert body_schedule is not None
            _check_schedule(body_schedule, takes_time)


def _check_segment(
    segment: _Segment, takes_time: Callable[[stim.CircuitInstruction], bool]
) -> None:
    """Check the order of the operations of a segment of a circuit, and the qubits they occupy."""
    last_moment: dict[int, int] = {}  # by qubit
    occupied: dict[int, set[int]] = {}  # by moment, the qubits of operations that take time
    record_moments: list[int] = []  # the moment of each measurement record of the segment
    coordinate_moment = 0
    for index, (operation, moment) in enumerate(zip(segment.operations, segment.moments)):
        # operations on each qubit keep their order
        instruction = operation.instruction
        for qubit in operation.qubits:
            if moment < last_moment.get(qubit, 0):
                raise ValueError(f"Operations on qubit {qubit} change order at {instruction}")
            last_moment[qubit] = moment

        # no qubit has two operations that take time in one moment
        if operation.is_gate and takes_time(instruction):
            qubits = occupied.setdefault(moment, set())
            if not qubits.isdisjoint(operation.qubits):
                qubit = min(qubits & set(operation.qubits))
                raise ValueError(
                    f"Qubit {qubit} has two operations that take time in moment {moment}"
                )
            qubits.update(operation.qubits)

        # operations follow the measurements that they read, coordinate annotations keep their
        # order, and chains of correlated errors stay in one moment
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
    """Place the nodes of a segment in layers as early (or, if ``reverse``, as late) as possible.

    A layer is a moment of the segment, counted from its start.  Each edge ``(earlier, later,
    gap)`` places node ``later`` at least ``gap`` layers after node ``earlier``.  Fixed nodes keep
    their given layers, which must come from a valid placement with the fewest layers.
    """
    # the constraints on each node from the nodes before it (or, if reverse, after it)
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

    # find the number of layers, then place nodes with the fixed ones counted from the same end
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
        # a TICK ends the last moment of a segment before a REPEAT block
        circuit = stim.Circuit()
        is_open = False
        for segment_or_block in _get_segments(schedule._circuit, schedule):
            if isinstance(segment_or_block, _Segment):
                self.lay_out_segment(segment_or_block, circuit)
                is_open = segment_or_block.duration > 0
            else:
                block, body_schedule = segment_or_block
                assert body_schedule is not None
                if is_open:
                    circuit.append("TICK")
                self.lay_out_block(block, body_schedule, circuit)
                is_open = False
        return circuit, is_open

    def lay_out_segment(self, segment: _Segment, circuit: stim.Circuit) -> None:
        """Lay out the operations of a segment by moment."""
        # the index of the first record of each operation, in the order of the circuit
        start = self.num_records
        old_starts = [start]
        for operation in segment.operations:
            old_starts.append(old_starts[-1] + operation.num_records)

        # append the operations by moment, and in circuit order within a moment, with a TICK before
        # each new moment
        order = sorted(range(len(segment.operations)), key=lambda k: (segment.moments[k], k))
        moment = 0
        for index in order:
            operation, old_start = segment.operations[index], old_starts[index]
            for _ in range(segment.moments[index] - moment):
                circuit.append("TICK")
            moment = segment.moments[index]

            # point the records that the operation reads at their new positions
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

            # note the new positions of the records that the operation produces
            for old_record in range(old_start, old_start + operation.num_records):
                self.old_to_new[old_record] = self.num_records
                self.new_to_old.append(old_record)
                self.num_records += 1

    def lay_out_block(
        self, block: stim.CircuitRepeatBlock, schedule: Schedule, circuit: stim.Circuit
    ) -> None:
        """Lay out a REPEAT block, ending each iteration that ends with a moment with a TICK.

        The first iterations are written out if they read measurements before the block at other
        offsets than later iterations read those of earlier iterations.
        """
        # find how many first iterations may read records before the block
        body = block.body_copy()
        records_per_iteration = body.num_measurements
        reach = _get_reach(body)
        # include the first steady iteration: it and every later iteration read only records of
        # earlier iterations of the block
        num_distinct = (
            1
            if reach <= 0 or records_per_iteration == 0
            else 1 + math.ceil(reach / records_per_iteration)
        )
        num_distinct = min(num_distinct, block.repeat_count)

        # lay out those iterations, grouping equal layouts
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
    num_records = 0  # the records of the circuit so far
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
