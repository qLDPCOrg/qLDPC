# SPDX-License-Identifier: Apache-2.0

"""Scheduling of circuit operations into moments, which set the idle errors of noise models."""

from __future__ import annotations

import abc
import dataclasses
from collections.abc import Callable, Mapping, Sequence

import stim

from ..common import StimCircuitLike
from ._operations import (
    ANNOTATION,
    CLIFFORD_1Q,
    CLIFFORD_PP,
    JUST_MEASURE_OPS,
    JUST_MEASURE_PP,
    JUST_RESET_OPS,
    NOISE,
    involves_classical_bits,
    op_type,
)

####################################################################################################
# schedulers


class Scheduler(abc.ABC):
    """A rule that divides a circuit into moments, during which idle qubits accumulate errors.

    A moment is the time between two TICKs.  A :class:`~qldpc.circuits.NoiseModel` uses its
    scheduler to divide a circuit into moments before adding noise, and to identify the operations
    that occupy their qubits for a moment.  Operations that take no time, such as classically
    controlled Pauli gates by default, may share a moment with another operation on their qubits,
    and leave their qubits idle.

    The built-in schedulers are made by :meth:`preserve_ticks`, :meth:`asap`, and :meth:`alap`.  A
    custom scheduler implements :meth:`schedule`, and may override :meth:`takes_time`.
    """

    @abc.abstractmethod
    def schedule(self, circuit: StimCircuitLike) -> StimCircuitLike:
        """Divide a noiseless circuit into moments separated by TICKs.

        The result may still use a qubit for more than one operation that takes time in a moment;
        a noise model then starts a new moment at each such reuse.
        """

    def takes_time(self, op: stim.CircuitInstruction) -> bool:
        """Does one application of a gate occupy its qubits for a moment?

        By default, every gate except a classically controlled Pauli gate takes time.  Annotations
        and noise channels never take time.
        """
        return not involves_classical_bits(op)

    @staticmethod
    def preserve_ticks(*, single_qubit_gates_take_time: bool = False) -> Scheduler:
        """A scheduler that keeps the moments of a circuit.

        A moment of a circuit ends at each TICK, and where an operation that takes time reuses a
        qubit.

        Args:
            single_qubit_gates_take_time: If False (the default), single-qubit gates take no time,
                so a moment may contain any number of them on a qubit in addition to one other
                operation.  Their qubits still idle, and a moment between two TICKs that contains
                only single-qubit gates still lasts a moment.
        """
        return _TickScheduler(single_qubit_gates_take_time)

    @staticmethod
    def asap(
        *,
        late_resets: bool = True,
        preserve_measurement_order: bool = False,
        single_qubit_gates_take_time: bool = False,
    ) -> Scheduler:
        """A scheduler that places every operation as early as possible, ignoring TICKs.

        Args:
            late_resets: If True (the default), place resets as late as possible instead, so that
                qubits do not idle between their reset and their next operation.
            preserve_measurement_order: If True, keep the order of measurements, and hence of the
                results of a circuit.  Otherwise (the default), measurements may change order, but
                detectors may not, and the circuit's own detectors, observables, and classically
                controlled gates keep reading the same measurements.  Measurement records read
                after the circuit, such as by detectors appended to it later, may then change.
            single_qubit_gates_take_time: As in :meth:`preserve_ticks`.
        """
        return _LayerScheduler(
            as_late_as_possible=False,
            shorten_lifetimes=late_resets,
            preserve_measurement_order=preserve_measurement_order,
            single_qubit_gates_take_time=single_qubit_gates_take_time,
        )

    @staticmethod
    def alap(
        *,
        early_measurements: bool = True,
        preserve_measurement_order: bool = False,
        single_qubit_gates_take_time: bool = False,
    ) -> Scheduler:
        """A scheduler that places every operation as late as possible, ignoring TICKs.

        Args:
            early_measurements: If True (the default), place measurements as early as possible
                instead, so that qubits do not idle between their last operation and measurement.
            preserve_measurement_order: As in :meth:`asap`.
            single_qubit_gates_take_time: As in :meth:`preserve_ticks`.
        """
        return _LayerScheduler(
            as_late_as_possible=True,
            shorten_lifetimes=early_measurements,
            preserve_measurement_order=preserve_measurement_order,
            single_qubit_gates_take_time=single_qubit_gates_take_time,
        )


@dataclasses.dataclass(frozen=True)
class _TickScheduler(Scheduler):
    """The scheduler made by :meth:`Scheduler.preserve_ticks`."""

    single_qubit_gates_take_time: bool

    def schedule(self, circuit: StimCircuitLike) -> StimCircuitLike:
        return circuit

    def takes_time(self, op: stim.CircuitInstruction) -> bool:
        return _takes_time(op, self.single_qubit_gates_take_time)

    def __repr__(self) -> str:
        args = "single_qubit_gates_take_time=True" if self.single_qubit_gates_take_time else ""
        return f"Scheduler.preserve_ticks({args})"


@dataclasses.dataclass(frozen=True)
class _LayerScheduler(Scheduler):
    """The schedulers made by :meth:`Scheduler.asap` and :meth:`Scheduler.alap`.

    Operations are placed in layers, which become moments, keeping the order of operations on each
    qubit.  Each detector or observable that reads measurement records follows the measurements
    that it reads, and an observable with Pauli targets keeps its place among the operations on its
    qubits, in a later moment than the gates before it, so that a noise model adds their noise
    before it.  Detectors keep their order, and hence their indices, and operations in a moment
    keep their order in the circuit.  Noise channels keep their place among the operations on their
    qubits.  ``SHIFT_COORDS`` instructions are folded into the coordinates of the annotations that
    follow them.

    Each REPEAT block is scheduled on its own, with its boundaries as boundaries between moments,
    so operations do not move into or out of a REPEAT block, or between its iterations.  If a
    measurement record is read across such a boundary, all measurements keep their order.
    Flattening a circuit's REPEAT blocks avoids these restrictions.
    """

    as_late_as_possible: bool
    shorten_lifetimes: bool
    preserve_measurement_order: bool
    single_qubit_gates_take_time: bool

    def schedule(self, circuit: StimCircuitLike) -> StimCircuitLike:
        stim_circuit = stim.Circuit()
        for index in range(len(circuit)):
            stim_circuit.append(circuit[index])
        preserve_measurement_order = self.preserve_measurement_order or _reads_across_blocks(
            stim_circuit
        )
        scheduled = self._schedule_block(stim_circuit, preserve_measurement_order)
        if isinstance(circuit, stim.Circuit):
            return scheduled
        output = type(circuit)()
        output += scheduled
        return output

    def takes_time(self, op: stim.CircuitInstruction) -> bool:
        return _takes_time(op, self.single_qubit_gates_take_time)

    def __repr__(self) -> str:
        args = []
        if not self.shorten_lifetimes:
            args.append(
                "early_measurements=False" if self.as_late_as_possible else "late_resets=False"
            )
        if self.preserve_measurement_order:
            args.append("preserve_measurement_order=True")
        if self.single_qubit_gates_take_time:
            args.append("single_qubit_gates_take_time=True")
        return f"Scheduler.{'alap' if self.as_late_as_possible else 'asap'}({', '.join(args)})"

    def _schedule_block(
        self, circuit: stim.Circuit, preserve_measurement_order: bool
    ) -> stim.Circuit:
        """Schedule the operations between REPEAT blocks, and the bodies of REPEAT blocks."""
        scheduled = stim.Circuit()
        segment = stim.Circuit()
        for op in circuit:
            if isinstance(op, stim.CircuitRepeatBlock):
                _append_moments(
                    scheduled, self._schedule_segment(segment, preserve_measurement_order)
                )
                segment = stim.Circuit()
                # end each iteration with a TICK, which a REPEAT block in the body already has
                body = self._schedule_block(op.body_copy(), preserve_measurement_order)
                if len(body) and body[-1].name != "REPEAT":
                    body.append("TICK")
                block = stim.Circuit()
                block.append(stim.CircuitRepeatBlock(op.repeat_count, body, tag=op.tag))
                _append_moments(scheduled, block)
            else:
                segment.append(op)
        _append_moments(scheduled, self._schedule_segment(segment, preserve_measurement_order))
        return scheduled

    def _schedule_segment(
        self, circuit: stim.Circuit, preserve_measurement_order: bool
    ) -> stim.Circuit:
        """Schedule a circuit without REPEAT blocks."""
        nodes, shift = _get_nodes(circuit, self.takes_time)
        edges = _get_edges(nodes, preserve_measurement_order)
        fixed = {}
        if self.shorten_lifetimes:
            # place measurements (or resets) first, as early (or late) as possible
            names = JUST_MEASURE_OPS if self.as_late_as_possible else JUST_RESET_OPS
            layers = _get_layers(len(nodes), edges, reverse=not self.as_late_as_possible)
            fixed = {
                index: layers[index]
                for index, node in enumerate(nodes)
                if node.instructions[0].name in names
            }
        layers = _get_layers(len(nodes), edges, reverse=self.as_late_as_possible, fixed=fixed)
        scheduled = _build_circuit(nodes, layers)
        if any(shift):
            scheduled.append("SHIFT_COORDS", [], shift)
        return scheduled


####################################################################################################
# helper functions


def _takes_time(op: stim.CircuitInstruction, single_qubit_gates_take_time: bool) -> bool:
    """Does one application of a gate take time, for a built-in scheduler?"""
    return not involves_classical_bits(op) and (
        single_qubit_gates_take_time or op_type(op.name) != CLIFFORD_1Q
    )


def _append_moments(circuit: stim.Circuit, moments: stim.Circuit) -> None:
    """Append moments to a circuit, with a boundary between them if both are nonempty.

    A scheduled REPEAT block ends with a TICK, so it needs no boundary after it.
    """
    if (
        len(circuit)
        and len(moments)
        and circuit[-1].name not in ("TICK", "REPEAT")
        and moments[0].name != "TICK"
    ):
        circuit.append("TICK")
    circuit += moments


def _reads_across_blocks(circuit: stim.Circuit) -> bool:
    """Does an instruction read a measurement record from outside its segment?

    A segment is the part of a circuit or REPEAT body that lies between REPEAT blocks.
    """
    num_records = 0
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            if _reads_across_blocks(op.body_copy()):
                return True
            num_records = 0
            continue
        for target in op.targets_copy():
            if target.is_measurement_record_target and -target.value > num_records:
                return True
        num_records += _count_records(op)
    return False


def _count_records(op: stim.CircuitInstruction) -> int:
    """The number of measurement records that an instruction produces."""
    circuit = stim.Circuit()
    circuit.append(op)
    return circuit.num_measurements


@dataclasses.dataclass
class _Node:
    """Instructions that a scheduler places together in a moment."""

    instructions: list[stim.CircuitInstruction]
    qubits: tuple[int, ...]
    is_gate: bool  # whether this node is a quantum operation, which a noise model makes noisy
    timed: bool  # whether this node takes a moment of time on its qubits
    num_records: int  # the number of measurement records that this node produces
    records: list[int]  # the indices of the measurement records that this node reads


def _get_nodes(
    circuit: stim.Circuit, takes_time: Callable[[stim.CircuitInstruction], bool]
) -> tuple[list[_Node], list[float]]:
    """Split a circuit into individual gate applications, annotations, and noise channels.

    Also return the total shift of coordinates by ``SHIFT_COORDS`` instructions, which are folded
    into the coordinates of the ``DETECTOR`` and ``QUBIT_COORDS`` annotations that follow them.
    """
    nodes: list[_Node] = []
    num_records = 0
    shift: list[float] = []
    for op in circuit:
        assert isinstance(op, stim.CircuitInstruction)
        args = op.gate_args_copy()
        if op.name == "TICK":
            continue
        if op.name == "SHIFT_COORDS":
            shift = [a + b for a, b in zip(_pad(shift, len(args)), _pad(args, len(shift)))]
            continue
        if op.name in ("DETECTOR", "QUBIT_COORDS"):
            args = [arg + offset for arg, offset in zip(args, _pad(shift, len(args)))]
        if op.name == "ELSE_CORRELATED_ERROR":
            # an ELSE_CORRELATED_ERROR belongs to the chain of correlated errors before it
            nodes[-1].instructions.append(op)
            nodes[-1].qubits = tuple(
                dict.fromkeys(nodes[-1].qubits + _get_qubits(op.targets_copy()))
            )
            continue

        if op.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            # split an observable into its Pauli targets, which act on qubits, and its records
            targets = op.targets_copy()
            paulis = [target for target in targets if not target.is_measurement_record_target]
            records = [target for target in targets if target.is_measurement_record_target]
            groups = [group for group in (paulis, records) if group] or [[]]
        elif op_type(op.name) in (ANNOTATION, NOISE):
            groups = [op.targets_copy()]
        elif op_type(op.name) in (CLIFFORD_PP, JUST_MEASURE_PP):
            # restore the combiners of each Pauli product, which target_groups drops
            groups = [
                [target for pauli in product for target in (stim.target_combiner(), pauli)][1:]
                for product in op.target_groups()
            ]
        else:
            groups = op.target_groups()

        for group in groups:
            instruction = stim.CircuitInstruction(op.name, group, args, tag=op.tag)
            is_gate = op_type(op.name) not in (ANNOTATION, NOISE)
            timed = is_gate and takes_time(instruction)
            is_gate = is_gate and not involves_classical_bits(instruction)
            records = [
                num_records + target.value
                for target in group
                if target.is_measurement_record_target
            ]
            node = _Node(
                [instruction],
                _get_qubits(group),
                is_gate,
                timed,
                _count_records(instruction),
                records,
            )
            nodes.append(node)
            num_records += node.num_records
    return nodes, shift


def _get_qubits(targets: Sequence[stim.GateTarget]) -> tuple[int, ...]:
    """The qubits of some targets, in order and without repetition."""
    return tuple(
        dict.fromkeys(
            target.qubit_value
            for target in targets
            if not target.is_combiner
            and not target.is_measurement_record_target
            and target.qubit_value is not None
        )
    )


def _pad(values: Sequence[float], length: int) -> list[float]:
    """Pad a list of numbers with zeros to a given length."""
    return list(values) + [0.0] * (length - len(values))


def _get_edges(
    nodes: Sequence[_Node], preserve_measurement_order: bool
) -> list[tuple[int, int, int]]:
    """Constraints ``(earlier, later, gap)``, which place a node ``gap`` or more moments later.

    A node follows the previous node on each of its qubits, and a timed node also follows the
    previous timed node on each of its qubits by at least one moment.  An observable with Pauli
    targets follows the previous gate on each of its qubits by at least one moment, since a noise
    model adds the noise of a gate at the end of its moment.  A node that reads a measurement record
    follows the node that produced it, and a detector follows the previous detector, which keeps
    the indices of detectors.  If ``preserve_measurement_order``, a node that produces measurement
    records also follows the previous such node.
    """
    edges = []
    last_node: dict[int, int] = {}
    last_gate: dict[int, int] = {}
    last_timed_node: dict[int, int] = {}
    record_nodes: list[int] = []
    last_detector: int | None = None
    last_measurement: int | None = None
    for index, node in enumerate(nodes):
        name = node.instructions[0].name
        for qubit in node.qubits:
            if qubit in last_node:
                edges.append((last_node[qubit], index, 0))
            if node.timed and qubit in last_timed_node:
                edges.append((last_timed_node[qubit], index, 1))
            if name == "OBSERVABLE_INCLUDE" and qubit in last_gate:
                edges.append((last_gate[qubit], index, 1))
            last_node[qubit] = index
            if node.is_gate:
                last_gate[qubit] = index
            if node.timed:
                last_timed_node[qubit] = index
        edges.extend((record_nodes[record], index, 0) for record in node.records if record >= 0)
        if name == "DETECTOR":
            if last_detector is not None:
                edges.append((last_detector, index, 0))
            last_detector = index
        if node.num_records and preserve_measurement_order:
            if last_measurement is not None:
                edges.append((last_measurement, index, 0))
            last_measurement = index
        record_nodes.extend([index] * node.num_records)
    return edges


def _get_layers(
    num_nodes: int,
    edges: Sequence[tuple[int, int, int]],
    *,
    reverse: bool,
    fixed: Mapping[int, int] | None = None,
) -> list[int]:
    """Place nodes in layers as early (or, if ``reverse``, as late) as possible.

    Layers count moments from the start of the circuit.  Fixed nodes keep their given layers, which
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
        """Moments from the start (or, if reverse, the end) of the circuit."""
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


def _build_circuit(nodes: Sequence[_Node], layers: Sequence[int]) -> stim.Circuit:
    """Build a circuit with TICKs between layers, updating the measurement records that nodes read.

    Nodes are ordered by layer, and otherwise keep their order in the input circuit.  Records that a
    node reads from before the circuit keep their index relative to the start of the circuit.
    """
    order = sorted(range(len(nodes)), key=lambda index: (layers[index], index))
    new_records: dict[int, int] = {}  # old record index -> new record index
    old_record_starts = [0]
    for node in nodes:
        old_record_starts.append(old_record_starts[-1] + node.num_records)
    circuit = stim.Circuit()
    num_records = 0
    for position, index in enumerate(order):
        if position and layers[index] != layers[order[position - 1]]:
            circuit.append("TICK")
        node = nodes[index]
        records = iter(node.records)
        for instruction in node.instructions:
            targets = []
            for target in instruction.targets_copy():
                if target.is_measurement_record_target:
                    record = next(records)
                    target = stim.target_rec(new_records.get(record, record) - num_records)
                targets.append(target)
            circuit.append(
                stim.CircuitInstruction(
                    instruction.name, targets, instruction.gate_args_copy(), tag=instruction.tag
                )
            )
        for offset in range(node.num_records):
            new_records[old_record_starts[index] + offset] = num_records + offset
        num_records += node.num_records
    return circuit
