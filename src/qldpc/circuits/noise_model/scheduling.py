# SPDX-License-Identifier: Apache-2.0

"""Scheduling of circuit operations into moments, which set the idle errors of noise models."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

import stim

from ..common import StimCircuitLike
from ._moments import is_instantaneous_1q_gate
from ._operations import (
    ANNOTATION,
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


@dataclasses.dataclass(frozen=True)
class Scheduler:
    """Rules that divide a circuit into moments, during which idle qubits accumulate errors.

    A moment is the time between two TICKs.  By default, a circuit keeps its own moments.  A
    rescheduling scheduler instead ignores the TICKs of a circuit, and places each operation in the
    earliest (or latest) moment that respects the order of operations on each qubit.

    Scheduling every operation as early as possible makes qubits wait after their preparation for
    the operations that follow, and scheduling them as late as possible makes qubits wait for
    measurement.  ``shorten_lifetimes`` avoids either wait.  Rescheduling keeps each ``I`` or ``II``
    gate in its place among the operations on its qubits, but not its alignment with idle markers on
    other qubits.

    Args:
        reschedule: If True, ignore the TICKs of a circuit and compute new moments.  Otherwise
            (the default), keep the moments of a circuit, which
            :class:`~qldpc.circuits.NoiseModel` splits wherever an operation reuses a qubit.
        alap: If True, schedule operations as late as possible, rather than as early as possible.
            Requires ``reschedule``.
        shorten_lifetimes: If True, shorten the time between the preparation and measurement of
            each qubit: resets are scheduled as late as possible when other operations are
            scheduled as early as possible, and measurements are scheduled as early as possible
            when other operations are scheduled as late as possible.  Requires ``reschedule``.
        instantaneous_1q_gates: If True, single-qubit gates other than ``I`` take no time on their
            qubits: a moment may contain any number of them on a qubit, in addition to one other
            operation, and a qubit that undergoes only single-qubit gates in a moment still idles
            during it.  Moments are still delimited by TICKs, so a moment that contains only
            single-qubit gates lasts a moment, during which every qubit idles.  An ``I`` gate marks
            a moment of idling on its qubit, as ``II`` does on two qubits, so both always take a
            moment.

    Raises:
        ValueError: If ``alap`` or ``shorten_lifetimes`` is set without ``reschedule``.
    """

    reschedule: bool = False
    alap: bool = False
    shorten_lifetimes: bool = False
    instantaneous_1q_gates: bool = False

    def __post_init__(self) -> None:
        if not self.reschedule and (self.alap or self.shorten_lifetimes):
            raise ValueError("`alap` and `shorten_lifetimes` require `reschedule=True`")

    def schedule(self, circuit: StimCircuitLike) -> StimCircuitLike:
        """Divide a noiseless circuit into moments separated by TICKs.

        Without ``reschedule``, the circuit is returned unchanged.  Otherwise, the circuit is
        rewritten with the same operations, in an order that preserves the order of operations on
        each qubit.  Each annotation keeps its meaning: a detector or observable that reads
        measurement records follows the measurements that it reads, and an observable with Pauli
        targets keeps its place among the operations on its qubits, in a later moment than the
        gates before it, so that a noise model adds their noise before it.  Detectors keep their
        order, and hence their indices.

        Args:
            circuit: The circuit to schedule.

        Returns:
            The scheduled circuit, of the same type as the input.

        Raises:
            ValueError: If rescheduling a circuit with REPEAT blocks, SHIFT_COORDS, or noise.
        """
        if not self.reschedule:
            return circuit
        stim_circuit = stim.Circuit()
        for index in range(len(circuit)):
            stim_circuit.append(circuit[index])
        nodes = _get_nodes(stim_circuit, self.instantaneous_1q_gates)
        edges = _get_edges(nodes)
        fixed = {}
        if self.shorten_lifetimes:
            # place measurements (or resets) first, as early (or late) as possible
            names = JUST_MEASURE_OPS if self.alap else JUST_RESET_OPS
            layers = _get_layers(len(nodes), edges, reverse=not self.alap)
            fixed = {
                index: layers[index]
                for index, node in enumerate(nodes)
                if node.instruction.name in names
            }
        layers = _get_layers(len(nodes), edges, reverse=self.alap, fixed=fixed)
        scheduled = _build_circuit(nodes, layers)
        if isinstance(circuit, stim.Circuit):
            return scheduled
        output = type(circuit)()
        output += scheduled
        return output


####################################################################################################
# helper functions


@dataclasses.dataclass
class _Node:
    """One gate application or annotation, which a scheduler places in a moment."""

    instruction: stim.CircuitInstruction
    qubits: tuple[int, ...]
    is_gate: bool  # whether this node is a quantum operation, which a noise model makes noisy
    timed: bool  # whether this node takes a moment of time on its qubits
    num_records: int  # the number of measurement records that this node produces
    records: list[int]  # the absolute indices of the measurement records that this node reads


def _get_nodes(circuit: stim.Circuit, instantaneous_1q_gates: bool) -> list[_Node]:
    """Split a circuit into individual gate applications and annotations."""
    nodes: list[_Node] = []
    num_records = 0
    for op in circuit:
        if op.name in ("REPEAT", "SHIFT_COORDS") or op_type(op.name) == NOISE:
            raise ValueError(f"Rescheduling does not support {op.name} instructions")
        assert isinstance(op, stim.CircuitInstruction)
        if op.name == "TICK":
            continue
        if op.name in ("DETECTOR", "OBSERVABLE_INCLUDE"):
            # split an observable into its Pauli targets, which act on qubits, and its records
            targets = op.targets_copy()
            paulis = [target for target in targets if not target.is_measurement_record_target]
            records = [target for target in targets if target.is_measurement_record_target]
            groups = [group for group in (paulis, records) if group] or [[]]
        elif op_type(op.name) == ANNOTATION:
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
            instruction = stim.CircuitInstruction(op.name, group, op.gate_args_copy(), tag=op.tag)
            qubits = tuple(
                dict.fromkeys(
                    target.qubit_value
                    for target in group
                    if not target.is_combiner
                    and not target.is_measurement_record_target
                    and target.qubit_value is not None
                )
            )
            # a noise model makes gates noisy, and gives each timed node a moment on its qubits
            is_gate = op_type(op.name) != ANNOTATION and not involves_classical_bits(instruction)
            timed = op_type(op.name) != ANNOTATION and not (
                instantaneous_1q_gates and is_instantaneous_1q_gate(instruction)
            )
            records = [
                num_records + target.value
                for target in group
                if target.is_measurement_record_target
            ]
            node_records = stim.Circuit()
            node_records.append(instruction)
            nodes.append(
                _Node(instruction, qubits, is_gate, timed, node_records.num_measurements, records)
            )
            num_records += node_records.num_measurements
    return nodes


def _get_edges(nodes: Sequence[_Node]) -> list[tuple[int, int, int]]:
    """Constraints ``(earlier, later, gap)``, which place a node ``gap`` or more moments later.

    A node follows the previous node on each of its qubits, and a timed node also follows the
    previous timed node on each of its qubits by at least one moment.  An observable with Pauli
    targets follows the previous gate on each of its qubits by at least one moment, since a noise
    model adds the noise of a gate at the end of its moment.  A node that reads a measurement record
    follows the node that produced it, and a detector follows the previous detector, which keeps
    the indices of detectors.
    """
    edges = []
    last_node: dict[int, int] = {}
    last_gate: dict[int, int] = {}
    last_timed_node: dict[int, int] = {}
    record_nodes: list[int] = []
    last_detector: int | None = None
    for index, node in enumerate(nodes):
        reads_qubits = node.instruction.name == "OBSERVABLE_INCLUDE"
        for qubit in node.qubits:
            if qubit in last_node:
                edges.append((last_node[qubit], index, 0))
            if node.timed and qubit in last_timed_node:
                edges.append((last_timed_node[qubit], index, 1))
            if reads_qubits and qubit in last_gate:
                edges.append((last_gate[qubit], index, 1))
            last_node[qubit] = index
            if node.is_gate:
                last_gate[qubit] = index
            if node.timed:
                last_timed_node[qubit] = index
        edges.extend((record_nodes[record], index, 0) for record in node.records)
        if node.instruction.name == "DETECTOR":
            if last_detector is not None:
                edges.append((last_detector, index, 0))
            last_detector = index
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

    Nodes are ordered by layer, and otherwise keep their order in the input circuit.
    """
    order = sorted(range(len(nodes)), key=lambda index: (layers[index], index))
    new_records: dict[int, int] = {}  # old absolute record index -> new absolute record index
    old_record_starts = [0]
    for node in nodes:
        old_record_starts.append(old_record_starts[-1] + node.num_records)
    circuit = stim.Circuit()
    num_records = 0
    for position, index in enumerate(order):
        if position and layers[index] != layers[order[position - 1]]:
            circuit.append("TICK")
        node = nodes[index]
        instruction = node.instruction
        if node.records:
            records = iter(node.records)
            targets = [
                stim.target_rec(new_records[next(records)] - num_records)
                if target.is_measurement_record_target
                else target
                for target in instruction.targets_copy()
            ]
            instruction = stim.CircuitInstruction(
                instruction.name, targets, instruction.gate_args_copy(), tag=instruction.tag
            )
        circuit.append(instruction)
        for offset in range(node.num_records):
            new_records[old_record_starts[index] + offset] = num_records + offset
        num_records += node.num_records
    return circuit
