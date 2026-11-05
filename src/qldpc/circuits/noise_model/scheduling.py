# SPDX-License-Identifier: Apache-2.0

"""Schedules, which place the operations of a circuit in moments, and schedulers that make them.

See :class:`Schedule` and :class:`Scheduler`.
"""

from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import itertools
import operator
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, ClassVar, NoReturn

import stim

from ..common import StimCircuitLike
from ._operations import (
    CLIFFORD_1Q,
    CLIFFORD_PP,
    GATE_OP_TYPES,
    JUST_MEASURE_OPS,
    JUST_MEASURE_PP,
    JUST_RESET_OPS,
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
    """The moments of the operations of a circuit, as made by a :class:`Scheduler`.

    An operation is a part of an instruction that acts independently on its qubits.  A gate or
    noise channel has one operation per application, such as one target of ``H``, one pair of
    targets of ``CX``, or one Pauli product of ``MPP``.  An observable has one operation for each of
    its Pauli targets, and one for its measurement records, unless it has Pauli targets but no
    records.  Other annotations are one operation each, and TICKs are not operations.

    A schedule places each operation in a moment, numbered from zero, and says whether each gate
    takes time.  Each moment is one unit of time.  A qubit idles in a moment unless a gate that
    takes time acts on it there.  A circuit with gates lasts until the end of the latest moment of
    its gates.  A circuit without gates has no moments.

    A circuit with REPEAT blocks is scheduled in parts.  The body of each block, and the
    instructions between blocks, are each scheduled in moments of their own.  Every iteration of a
    block follows the schedule of its body, so a noisy circuit keeps the block.  No moment crosses
    the start or end of a block or of an iteration, and neither does an idle run of a qubit (see
    :class:`~qldpc.circuits.NoiseModel`).  To let idle runs cross them, add noise to
    ``circuit.flattened()``.

    Schedulers make schedules, as in ``TickScheduler().get_schedule(circuit)``, and schedules are
    immutable.  A noise model can use one schedule for circuits that order their operations
    differently; :mod:`qldpc.circuits.noise_model` describes this and the other way to set the
    timing of circuits.
    """

    __slots__ = ("_circuit", "_moments", "_takes_time")

    _circuit: stim.Circuit
    # in circuit order, the moment of each operation outside REPEAT blocks, and a schedule of the
    # body of each REPEAT block
    _moments: tuple[int | Schedule, ...]
    # whether each operation outside REPEAT blocks is a gate that takes time
    _takes_time: tuple[bool, ...]

    def __init__(self) -> None:
        raise TypeError(
            "Schedules are made by schedulers, as in TickScheduler().get_schedule(circuit)"
        )

    @property
    def circuit(self) -> stim.Circuit:
        """The scheduled circuit."""
        return self._circuit.copy()

    @property
    def num_moments(self) -> int:
        """The number of moments of this schedule, in all iterations of REPEAT blocks."""
        num_moments = 0
        for segment_or_block in _get_segments(self._circuit, self):
            if isinstance(segment_or_block, _Segment):
                num_moments += segment_or_block.num_moments
            else:
                block, body_schedule = segment_or_block
                assert body_schedule is not None
                num_moments += block.repeat_count * body_schedule.num_moments
        return num_moments

    def get_moments(self) -> list[list[stim.CircuitInstruction] | tuple[int, Schedule]]:
        """The operations in each moment, in order, with the REPEAT blocks between them.

        Operations in a moment keep their order in the circuit.  A part of the circuit without gates
        has no moments, so its operations are not listed.  Each REPEAT block is listed as
        ``(repeat_count, body_schedule)``.
        """
        moments: list[list[stim.CircuitInstruction] | tuple[int, Schedule]] = []
        for segment_or_block in _get_segments(self._circuit, self):
            if isinstance(segment_or_block, _Segment):
                moments.extend(segment_or_block.get_moments())
            else:
                block, body_schedule = segment_or_block
                assert body_schedule is not None
                moments.append((block.repeat_count, body_schedule))
        return moments

    def to_circuit(self) -> stim.Circuit:
        """The scheduled circuit laid out in moments separated by TICKs.

        Operations are sorted by moment, and keep their circuit order within a moment.  An empty
        moment gives two TICKs in a row.  Each iteration of a REPEAT block repeats the layout of its
        body, and TICKs separate it from the moments around it.

        A :class:`TickScheduler` keeps measurements in circuit order.  So the layout of its schedule
        is a valid circuit with the same meaning as the scheduled circuit.  Other schedulers may
        reorder measurements, and record targets are kept as written, so their layouts are for
        display.
        """
        circuit = self._lay_out()
        if len(circuit) and circuit[-1] == stim.CircuitInstruction("TICK"):
            return circuit[:-1]
        return circuit

    def __eq__(self, other: object) -> bool:
        # compare circuits as text, since Stim's circuit equality ignores the tags of REPEAT blocks
        return (
            isinstance(other, Schedule)
            and str(self._circuit) == str(other._circuit)
            and self._moments == other._moments
            and self._takes_time == other._takes_time
        )

    def __repr__(self) -> str:
        return f"<Schedule with {self.num_moments} moments>"

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("A Schedule is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("A Schedule is immutable")

    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return Schedule._new, (self._circuit, self._moments, self._takes_time)

    @classmethod
    def _new(
        cls,
        circuit: stim.Circuit,
        moments: tuple[int | Schedule, ...],
        takes_time: tuple[bool, ...],
    ) -> Schedule:
        """A schedule of a circuit, from the entries of ``_moments`` and ``_takes_time``."""
        schedule = object.__new__(cls)
        object.__setattr__(schedule, "_circuit", circuit)
        object.__setattr__(schedule, "_moments", moments)
        object.__setattr__(schedule, "_takes_time", takes_time)
        return schedule

    def _lay_out(self) -> stim.Circuit:
        """The layout of :meth:`to_circuit`, with a TICK after every moment."""
        circuit = stim.Circuit()
        for segment_or_block in _get_segments(self._circuit, self):
            if isinstance(segment_or_block, _Segment):
                moments = segment_or_block.get_moments()
                if not moments:
                    for operation in segment_or_block.operations:
                        circuit.append(operation.instruction)
                for moment in moments:
                    for instruction in moment:
                        circuit.append(instruction)
                    circuit.append("TICK")
            else:
                block, body_schedule = segment_or_block
                assert body_schedule is not None
                body = body_schedule._lay_out()
                circuit.append(stim.CircuitRepeatBlock(block.repeat_count, body, tag=block.tag))
        return circuit


####################################################################################################
# schedulers


class Scheduler(abc.ABC):
    """A rule that places the operations of circuits in moments.

    A :class:`~qldpc.circuits.NoiseModel` with idle noise uses its scheduler to find how long each
    qubit idles.  The built-in schedulers are :class:`TickScheduler`, :class:`ASAPScheduler`,
    :class:`ALAPScheduler`, and :class:`MinIdleScheduler`.  Each places the gates of a circuit in
    its own way, and the other operations in the same way.  A noise channel, or a Pauli target of
    an observable, joins the latest moment of the operations before it on its qubits, or moment
    zero.  Other annotations, such as detectors and QUBIT_COORDS, join the moment of the last gate
    before them, or moment zero.  An annotation joins a later moment if it must follow a measurement
    that it reads, and a heralded noise channel, which records measurements, joins no earlier moment
    than the measurement before it.  The coordinate annotations DETECTOR, QUBIT_COORDS, and
    SHIFT_COORDS also keep their order, which sets the coordinates of detectors and qubits.

    A custom scheduler implements :meth:`get_schedule`, for example by laying a circuit out with
    TICKs and returning ``TickScheduler().get_schedule(timed_circuit)``.  A noise model can also
    use one schedule for many circuits (see :mod:`qldpc.circuits.noise_model`).
    """

    @abc.abstractmethod
    def get_schedule(self, circuit: StimCircuitLike) -> Schedule:
        """Place the operations of a circuit in moments.

        Returns:
            A schedule of the circuit, or of a circuit that it matches (see
            :mod:`qldpc.circuits.noise_model`).
        """


class _BuiltInScheduler(Scheduler):
    """A built-in scheduler, which schedules the parts of a circuit (see :class:`Schedule`).

    Single-qubit Clifford gates take time if ``single_qubit_gates_take_time`` is True.  Other gates
    always take time, except classically controlled Pauli gates.
    """

    single_qubit_gates_take_time: bool

    def get_schedule(self, circuit: StimCircuitLike) -> Schedule:
        return self._get_schedule(_as_stim_circuit(circuit), {})

    def _get_schedule(self, stim_circuit: stim.Circuit, next_gates: Mapping[int, bool]) -> Schedule:
        """Schedule a circuit, given the gates that follow it (see ``_get_moments``)."""
        # schedule the instructions between REPEAT blocks, and the body of each block recursively
        moments: list[int | Schedule] = []
        takes_time: list[bool] = []
        index = 0  # of the instruction after the current part
        for segment_or_block in _get_segments(stim_circuit, None):
            index += (
                len(segment_or_block.instructions) if isinstance(segment_or_block, _Segment) else 1
            )
            following = {**next_gates, **self._get_first_gates(stim_circuit[index:])}
            if isinstance(segment_or_block, _Segment):
                segment_takes_time = [self._takes_time(op) for op in segment_or_block.operations]
                moments.extend(self._get_moments(segment_or_block, segment_takes_time, following))
                takes_time.extend(segment_takes_time)
            else:
                # a body is followed by its next iteration, or by what follows the block
                block, _ = segment_or_block
                body = block.body_copy()
                repeated = self._get_first_gates(body).items()
                following |= {qubit: False for qubit, is_reset in repeated if not is_reset}
                moments.append(self._get_schedule(body, following))
        return Schedule._new(stim_circuit, tuple(moments), tuple(takes_time))

    def _takes_time(self, operation: _Operation) -> bool:
        """Whether an operation is a gate that takes time."""
        return (
            operation.is_gate
            and not involves_classical_bits(operation.instruction)
            and (
                self.single_qubit_gates_take_time
                or op_type(operation.instruction.name) != CLIFFORD_1Q
            )
        )

    def _get_first_gates(self, circuit: stim.Circuit) -> dict[int, bool]:
        """By qubit, whether the first gate that takes time on it is a reset (R, RX, RY, or RZ)."""
        first_gates: dict[int, bool] = {}
        for instruction in reversed(circuit):
            if isinstance(instruction, stim.CircuitRepeatBlock):
                first_gates |= self._get_first_gates(instruction.body_copy())
                continue
            for operation in _get_instruction_operations(instruction):
                if self._takes_time(operation):
                    is_reset = instruction.name in JUST_RESET_OPS
                    first_gates |= dict.fromkeys(operation.qubits, is_reset)
        return first_gates

    @abc.abstractmethod
    def _get_moments(
        self, segment: _Segment, takes_time: Sequence[bool], next_gates: Mapping[int, bool]
    ) -> list[int]:
        """The moment of each operation of a ``_Segment``, given whether each takes time.

        ``next_gates`` says, by qubit, whether the next gate that takes time after the ``_Segment``
        is a reset.  It omits qubits without such a gate.
        """


@dataclasses.dataclass(frozen=True, kw_only=True, repr=False)
class TickScheduler(_BuiltInScheduler):
    """A scheduler that takes the moments of a circuit from its TICKs.

    Each TICK between two gates ends a moment.  So the gates between two TICKs share a moment, even
    if several of them act on one qubit, and consecutive TICKs give empty moments, in which qubits
    idle.  TICKs before the first gate or after the last gate of a circuit or REPEAT body add no
    moments.

    Args:
        single_qubit_gates_take_time: If True (the default), every gate takes time, except a
            classically controlled Pauli gate.  If False, single-qubit Clifford gates such as H and
            S also take no time, so qubits idle in their moments.
    """

    single_qubit_gates_take_time: bool = True

    def __repr__(self) -> str:
        return _get_repr(self)

    def _get_moments(
        self, segment: _Segment, takes_time: Sequence[bool], next_gates: Mapping[int, bool]
    ) -> list[int]:
        # place each gate in the moment that the TICKs since the first gate have reached
        nodes = _get_nodes(segment.operations)
        gate_moments: dict[int, int] = {}  # the moment of each node with a gate
        moment = 0
        operations = iter(zip(nodes, segment.operations))
        for instruction in segment.instructions:
            if instruction.name == "TICK" and gate_moments:
                moment += 1
            num_operations = len(_get_instruction_operations(instruction))
            for node, operation in itertools.islice(operations, num_operations):
                if operation.is_gate:
                    gate_moments[node] = moment

        # annotations and noise channels join the moments of operations before them
        return _place_annotations(segment.operations, nodes, gate_moments)


class _LayerScheduler(_BuiltInScheduler):
    """The algorithm of :class:`ASAPScheduler` and :class:`ALAPScheduler`.

    The operations of a circuit are grouped into nodes (see ``_get_nodes``), and each node is placed
    in a layer, which is a moment.  The constraints from ``_get_edges`` keep the meaning of the
    circuit.
    """

    _as_late_as_possible: ClassVar[bool]

    @property
    @abc.abstractmethod
    def _gates_moved_back(self) -> frozenset[str]:
        """The gates to move back from their first placement, as far as the other gates allow.

        ASAP first places every gate as early as possible, and then moves these gates later.  ALAP
        does the reverse.
        """

    def _get_moments(
        self, segment: _Segment, takes_time: Sequence[bool], next_gates: Mapping[int, bool]
    ) -> list[int]:
        # group the operations into nodes, and find the constraints on their order
        operations = segment.operations
        nodes = _get_nodes(operations)
        num_nodes = nodes[-1] + 1 if nodes else 0
        edges = _get_edges(operations, nodes, takes_time)

        # place the nodes as early as possible, or as late for ALAP
        layers = _get_layers(num_nodes, edges, reverse=self._as_late_as_possible)
        if self._gates_moved_back:
            # keep the other gates in place, and move these gates back as far as they allow
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


@dataclasses.dataclass(frozen=True, kw_only=True, repr=False)
class ASAPScheduler(_LayerScheduler):
    """A scheduler that ignores TICKs and places each gate as early as possible.

    By default, it then moves each reset as late as the other gates allow.

    Args:
        late_resets: If True (the default), then move each reset (R, RX, RY, or RZ) as late as the
            other gates allow.  This shortens the time that qubits idle after a reset.  Gates that
            measure and then reset, such as MR, stay in place.
        single_qubit_gates_take_time: If False (the default), single-qubit Clifford gates such as H
            and S take no time.  A moment may then contain any number of them on a qubit, which
            still idles.  Other gates take time, except classically controlled Pauli gates.
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
    """A scheduler that ignores TICKs and places each gate as late as possible.

    By default, it then moves each measurement as early as the other gates allow.

    Args:
        early_measurements: If True (the default), then move each measurement (M, MX, MY, MZ, MPP,
            MXX, MYY, or MZZ) as early as the other gates allow.  This shortens the time that qubits
            idle before a measurement.  Gates that measure and then reset, such as MR, stay in
            place.
        single_qubit_gates_take_time: As in :class:`ASAPScheduler`.
    """

    _as_late_as_possible: ClassVar[bool] = True
    early_measurements: bool = True
    single_qubit_gates_take_time: bool = False

    def __repr__(self) -> str:
        return _get_repr(self)

    @property
    def _gates_moved_back(self) -> frozenset[str]:
        return JUST_MEASURE_OPS if self.early_measurements else frozenset()


@dataclasses.dataclass(frozen=True, kw_only=True, repr=False)
class MinIdleScheduler(_BuiltInScheduler):
    """A scheduler that ignores TICKs and minimizes the effective idling of qubits.

    It places gates to minimize the number of effective idle slots, which are the moments in which a
    qubit idles, except those of an idle run that ends at a reset (R, RX, RY, or RZ) or that follows
    the last gate on the qubit if that gate measures it.  Idle errors in the excluded moments are
    erased by the reset, or come after the last measurement of the qubit.  An idle run at the end
    of a part of a circuit (see :class:`Schedule`) ends at the next gate on its qubit, in the next
    iteration of a REPEAT block or after it, and counts if it counts in any iteration.  Each part
    keeps the number of moments that :class:`ASAPScheduler` gives it, which is the fewest possible.
    The placement comes from a linear program, which is solved exactly.

    Args:
        single_qubit_gates_take_time: As in :class:`ASAPScheduler`.
    """

    single_qubit_gates_take_time: bool = False

    def __repr__(self) -> str:
        return _get_repr(self)

    def _get_moments(
        self, segment: _Segment, takes_time: Sequence[bool], next_gates: Mapping[int, bool]
    ) -> list[int]:
        # group the operations into nodes, and find the constraints on their order and the depth
        operations = segment.operations
        nodes = _get_nodes(operations)
        num_nodes = nodes[-1] + 1 if nodes else 0
        edges = _get_edges(operations, nodes, takes_time)
        num_layers = max(_get_layers(num_nodes, edges, reverse=False), default=0) + 1

        # the effective idle slots are, up to a constant, the sum of costs[node] * layers[node]:
        # each counted run adds the layer of the gate that ends it, and subtracts the layer of the
        # gate that starts it
        costs = [0] * num_nodes
        last_gates: dict[int, tuple[int, _Operation]] = {}  # by qubit, the last that takes time
        for node, operation, operation_takes_time in zip(nodes, operations, takes_time):
            if operation_takes_time:
                for qubit in operation.qubits:
                    if operation.instruction.name not in JUST_RESET_OPS:
                        costs[node] += 1
                        if qubit in last_gates:
                            costs[last_gates[qubit][0]] -= 1
                    last_gates[qubit] = (node, operation)
        for qubit, (node, operation) in last_gates.items():
            # the run after the last gate counts unless a reset follows it, or nothing follows it
            # and the gate measures
            if not next_gates.get(qubit, operation.num_records > 0):
                costs[node] -= 1

        # place annotations and noise channels in the moments of operations before them
        layers = _get_min_cost_layers(costs, edges, num_layers)
        gate_moments = {
            node: layers[node] for node, operation in zip(nodes, operations) if operation.is_gate
        }
        return _place_annotations(operations, nodes, gate_moments)


def _get_repr(scheduler: DataclassInstance) -> str:
    """A constructor-like repr of a built-in scheduler, which omits default arguments."""
    args = [
        f"{field.name}={getattr(scheduler, field.name)!r}"
        for field in dataclasses.fields(scheduler)
        if getattr(scheduler, field.name) != field.default
    ]
    return f"{type(scheduler).__name__}({', '.join(args)})"


def _get_nodes(operations: Sequence[_Operation]) -> list[int]:
    """The node of each operation of a circuit, numbered from zero.

    A node is a group of operations that a scheduler places in one moment.  Each operation is a node
    of its own, except that an ``E`` and the ``ELSE_CORRELATED_ERROR`` operations after it form one
    node.
    """
    nodes: list[int] = []
    for operation in operations:
        if operation.instruction.name == "ELSE_CORRELATED_ERROR" and nodes:
            nodes.append(nodes[-1])
        else:
            nodes.append(nodes[-1] + 1 if nodes else 0)
    return nodes


def _get_edges(
    operations: Sequence[_Operation], nodes: Sequence[int], takes_time: Sequence[bool]
) -> list[tuple[int, int, int]]:
    """Constraints ``(earlier, later, gap)``: node ``later`` is ``gap`` or more layers later.

    A node follows the previous node on each of its qubits.  A node that takes time also follows the
    previous node that takes time on each of its qubits, by at least one layer.  A node follows the
    measurements that it reads.  An annotation without qubits follows the previous gate, so that a
    gate that reads the record of an MPAD follows the gates before the MPAD.
    """
    edges = []
    last_node: dict[int, int] = {}  # by qubit
    last_node_taking_time: dict[int, int] = {}  # by qubit
    record_nodes: list[int] = []  # the node of each measurement record of the circuit
    last_gate_node: int | None = None
    for node, operation, operation_takes_time in zip(nodes, operations, takes_time):
        # follow the previous nodes on the qubits of the operation
        for qubit in operation.qubits:
            if last_node.get(qubit, node) != node:
                edges.append((last_node[qubit], node, 0))
            if operation_takes_time and qubit in last_node_taking_time:
                edges.append((last_node_taking_time[qubit], node, 1))
            last_node[qubit] = node
            if operation_takes_time:
                last_node_taking_time[qubit] = node

        # follow the measurements that the operation reads
        edges.extend((record_nodes[record], node, 0) for record in operation.records if record >= 0)

        # an annotation without qubits follows the previous gate
        if operation.is_gate:
            last_gate_node = node
        elif not operation.qubits and last_gate_node not in (None, node):
            edges.append((last_gate_node, node, 0))
        record_nodes.extend([node] * operation.num_records)
    return edges


def _place_annotations(
    operations: Sequence[_Operation], nodes: Sequence[int], gate_moments: Mapping[int, int]
) -> list[int]:
    """The moment of each operation of a circuit, given the moment of each node with a gate.

    The other nodes join moments as the :class:`Scheduler` docstring describes.
    """
    moments: list[int] = []
    last_moment: dict[int, int] = {}  # by qubit
    record_moments: list[int] = []  # the moment of each measurement record of the circuit
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
                if operation.num_records and record_moments:
                    moment = max(moment, record_moments[-1])
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
# matching schedules to circuits


def _match_schedule(schedule: Schedule, circuit: stim.Circuit) -> Schedule:
    """A schedule of a circuit, with the moments of the matching operations of another schedule.

    A schedule matches a circuit if, for each qubit, the operations on that qubit, as written apart
    from their measurement-record targets, are the same in both, and their REPEAT blocks have the
    same repeat counts and bodies that match in the same way.  The k-th operation on a qubit in the
    circuit then gets the moment, and timing, of the k-th operation on that qubit in the schedule.
    Operations without qubits take no time, and get moment zero.  Record targets are ignored because
    only classically controlled Pauli gates have both qubits and record targets, and these take no
    time and end no idle runs.

    Raises:
        ValueError: If the schedule does not match the circuit.
    """
    if schedule._circuit == circuit:
        return schedule
    segments_and_blocks = list(_get_segments(schedule._circuit, schedule))
    circuit_segments_and_blocks = list(_get_segments(circuit, None))
    if len(segments_and_blocks) != len(circuit_segments_and_blocks):
        raise ValueError("The schedule is for a different circuit, with other REPEAT blocks")

    # segments and REPEAT blocks alternate, starting with a segment
    moments: list[int | Schedule] = []
    takes_time: list[bool] = []
    for segment_or_block, circuit_segment_or_block in zip(
        segments_and_blocks, circuit_segments_and_blocks
    ):
        if isinstance(segment_or_block, _Segment):
            assert isinstance(circuit_segment_or_block, _Segment)
            for index in _match_segment(segment_or_block, circuit_segment_or_block):
                moments.append(0 if index is None else segment_or_block.moments[index])
                takes_time.append(index is not None and segment_or_block.takes_time[index])
        else:
            assert not isinstance(circuit_segment_or_block, _Segment)
            (block, body_schedule), (circuit_block, _) = segment_or_block, circuit_segment_or_block
            assert body_schedule is not None
            if block.repeat_count != circuit_block.repeat_count:
                raise ValueError(
                    "The schedule is for a different circuit, with other REPEAT blocks"
                )
            moments.append(_match_schedule(body_schedule, circuit_block.body_copy()))
    return Schedule._new(circuit, tuple(moments), tuple(takes_time))


def _match_segment(segment: _Segment, circuit_segment: _Segment) -> list[int | None]:
    """For each operation of a ``_Segment`` of a circuit, the index of its match in a scheduled one.

    Operations without qubits have no match.

    Raises:
        ValueError: If the operations on a qubit differ.
    """
    # the indices of the scheduled operations on each qubit, in order
    indices_by_qubit: dict[int, list[int]] = collections.defaultdict(list)
    for index, operation in enumerate(segment.operations):
        for qubit in operation.qubits:
            indices_by_qubit[qubit].append(index)

    # match the k-th operation on each qubit of the circuit with the k-th scheduled one
    matches: list[int | None] = []
    num_matched: dict[int, int] = collections.defaultdict(int)  # by qubit
    for operation in circuit_segment.operations:
        match = None
        for qubit in operation.qubits:
            indices = indices_by_qubit[qubit]
            if num_matched[qubit] == len(indices) or not _match_operations(
                segment.operations[indices[num_matched[qubit]]], operation
            ):
                raise ValueError(f"The schedule is for a different circuit, on qubit {qubit}")
            match = indices[num_matched[qubit]]
            num_matched[qubit] += 1
        matches.append(match)
    for qubit, indices in indices_by_qubit.items():
        if num_matched[qubit] != len(indices):
            raise ValueError(f"The schedule is for a different circuit, on qubit {qubit}")
    return matches


def _match_operations(operation: _Operation, other: _Operation) -> bool:
    """Are two operations the same, as written apart from their measurement-record targets?"""
    if operation.instruction == other.instruction:
        return True
    return _without_records(operation.instruction) == _without_records(other.instruction)


def _without_records(
    instruction: stim.CircuitInstruction,
) -> tuple[str, list[stim.GateTarget], list[float], str]:
    """An instruction's name, targets other than measurement records, arguments, and tag."""
    targets = [
        target for target in instruction.targets_copy() if not target.is_measurement_record_target
    ]
    return instruction.name, targets, instruction.gate_args_copy(), instruction.tag


####################################################################################################
# operations


@dataclasses.dataclass(frozen=True)
class _Operation:
    """An operation of a circuit (see :class:`Schedule`), with properties for scheduling and noise.

    Only gates, noise channels, and observables have ``qubits``.  Gates without classical control
    and observables also depend on the state of their qubits, so noise on those qubits must come
    before them.  Noise channels and classically controlled Pauli gates do not.

    ``records`` are the measurement records that the operation reads, as negative offsets from the
    operation.  ``_get_segments`` indexes them differently, as described there.
    """

    instruction: stim.CircuitInstruction
    qubits: tuple[int, ...]  # the qubits that this operation acts on, in order
    qubits_depended_on: tuple[int, ...]  # the qubits whose state this operation depends on
    is_gate: bool
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
                target.qubit_value for target in targets if target.qubit_value is not None
            )
        )

        # find which of those qubits the operation acts on and depends on, and its records
        is_gate = op_type(name) in GATE_OP_TYPES
        is_observable = name == "OBSERVABLE_INCLUDE"
        return _Operation(
            instruction,
            qubits if is_gate or is_observable or op_type(name) == NOISE else (),
            qubits
            if (is_gate and not involves_classical_bits(instruction)) or is_observable
            else (),
            is_gate,
            len(instruction.target_groups()) if _produces_measurements(name) else 0,
            tuple(target.value for target in targets if target.is_measurement_record_target),
        )


@dataclasses.dataclass
class _Segment:
    """The instructions between REPEAT blocks of a circuit, with their operations and schedule.

    Records are indexed as in ``_get_segments``.
    """

    instructions: list[stim.CircuitInstruction]
    operations: list[_Operation]
    moments: list[int]
    takes_time: list[bool]  # whether each operation is a gate that takes time

    @property
    def num_moments(self) -> int:
        """The number of moments, which is zero without gates."""
        return 1 + max(
            (moment for op, moment in zip(self.operations, self.moments) if op.is_gate), default=-1
        )

    def get_moments(self) -> list[list[stim.CircuitInstruction]]:
        """The operations in each moment, which are none without gates."""
        moments: list[list[stim.CircuitInstruction]] = [[] for _ in range(self.num_moments)]
        if moments:
            for operation, moment in zip(self.operations, self.moments):
                moments[moment].append(operation.instruction)
        return moments


def _get_segments(
    circuit: stim.Circuit, schedule: Schedule | None
) -> Iterator[_Segment | tuple[stim.CircuitRepeatBlock, Schedule | None]]:
    """The ``_Segment``s and REPEAT blocks of a circuit in order, with a schedule of each body.

    A ``_Segment`` precedes each REPEAT block and ends the circuit.  The records that its operations
    read are indexed from its first measurement record, so earlier records have negative indices.
    Without a schedule, every operation is in moment zero and takes no time, and REPEAT blocks have
    no schedules.
    """
    moments = iter(() if schedule is None else schedule._moments)
    takes_time = iter(() if schedule is None else schedule._takes_time)
    segment = _Segment([], [], [], [])
    num_records = 0
    for instruction in circuit:
        # a REPEAT block ends the current _Segment, and a new one starts after it
        if isinstance(instruction, stim.CircuitRepeatBlock):
            yield segment
            body_schedule = next(moments, None)
            assert body_schedule is None or isinstance(body_schedule, Schedule)
            yield instruction, body_schedule
            segment = _Segment([], [], [], [])
            num_records = 0
            continue

        # other instructions join the _Segment with the moments of their operations
        segment.instructions.append(instruction)
        for operation in _get_instruction_operations(instruction):
            if operation.records:
                records = tuple(num_records + record for record in operation.records)
                operation = dataclasses.replace(operation, records=records)
            num_records += operation.num_records
            moment = next(moments, 0)
            assert isinstance(moment, int)
            segment.operations.append(operation)
            segment.moments.append(moment)
            segment.takes_time.append(next(takes_time, False))
    yield segment


@functools.lru_cache(maxsize=4096)
def _get_instruction_operations(instruction: stim.CircuitInstruction) -> tuple[_Operation, ...]:
    """The operations of an instruction, with records counted back from each operation.

    Instructions often recur, as in rounds of error correction, and stim builds instructions slowly.
    So the operations of recent instructions are cached.
    """
    return tuple(_Operation.from_instruction(op) for op in _split_operations(instruction))


def _split_operations(instruction: stim.CircuitInstruction) -> list[stim.CircuitInstruction]:
    """Split an instruction into its operations (see :class:`Schedule`)."""
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


def _as_stim_circuit(circuit: StimCircuitLike) -> stim.Circuit:
    """Copy a circuit into a stim.Circuit."""
    stim_circuit = stim.Circuit()
    for index in range(len(circuit)):
        stim_circuit.append(circuit[index])
    return stim_circuit


####################################################################################################
# layers of operations


def _get_layers(
    num_nodes: int,
    edges: Sequence[tuple[int, int, int]],
    *,
    reverse: bool,
    fixed: Mapping[int, int] | None = None,
) -> list[int]:
    """Place nodes in layers as early as possible, or as late if ``reverse``.

    A layer is a moment, counted from the start.  Each edge ``(earlier, later, gap)`` places node
    ``later`` at least ``gap`` layers after node ``earlier``.  Fixed nodes keep their given layers.
    These must come from a valid placement with the fewest layers.
    """
    # the constraints on each node from the nodes before it, or after it if reverse
    incoming: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for earlier, later, gap in edges:
        if reverse:
            incoming[earlier].append((later, gap))
        else:
            incoming[later].append((earlier, gap))
    order = range(num_nodes - 1, -1, -1) if reverse else range(num_nodes)

    def get_depths(fixed_depths: Mapping[int, int]) -> list[int]:
        """The layer of each node, counted from the start, or from the end if reverse."""
        depths = [0] * num_nodes
        for index in order:
            if index in fixed_depths:
                depths[index] = fixed_depths[index]
            else:
                depths[index] = max(
                    (depths[other] + gap for other, gap in incoming[index]), default=0
                )
        return depths

    # find the number of layers, then place the nodes, with fixed layers counted from the same end
    depths = get_depths({})
    num_layers = max(depths, default=0) + 1
    fixed = fixed or {}
    if reverse:
        depths = get_depths({index: num_layers - 1 - layer for index, layer in fixed.items()})
        return [num_layers - 1 - depth for depth in depths]
    return get_depths(fixed)


def _get_min_cost_layers(
    costs: Sequence[int], edges: Sequence[tuple[int, int, int]], num_layers: int
) -> list[int]:
    """Place nodes in ``num_layers`` layers to minimize the sum of ``costs[node] * layers[node]``.

    Each edge ``(earlier, later, gap)`` places node ``later`` at least ``gap`` layers after node
    ``earlier``, and such a placement must exist.  Each constraint bounds a difference of two
    layers, so the constraint matrix is totally unimodular, and the optimal vertex that the linear
    program finds is a placement in integer layers.

    Raises:
        RuntimeError: If the solver fails to find a placement.
    """
    # scipy.optimize is slow to import, so it is imported only when needed
    import scipy.optimize
    import scipy.sparse

    if not costs:
        return []
    # each edge is the constraint layers[earlier] - layers[later] <= -gap
    rows = [row for row in range(len(edges)) for _ in range(2)]
    columns = [node for earlier, later, _ in edges for node in (earlier, later)]
    constraints = scipy.sparse.coo_array(
        ([1, -1] * len(edges), (rows, columns)), shape=(len(edges), len(costs))
    )
    result = scipy.optimize.linprog(
        costs,
        A_ub=constraints,
        b_ub=[-gap for _, _, gap in edges],
        bounds=(0, num_layers - 1),
        method="highs-ds",
    )
    layers = [round(layer) for layer in result.x] if result.success else []
    if not result.success or any(
        layers[later] - layers[earlier] < gap for earlier, later, gap in edges
    ):
        raise RuntimeError(f"The linear program failed to place nodes in layers: {result.message}")
    return layers
