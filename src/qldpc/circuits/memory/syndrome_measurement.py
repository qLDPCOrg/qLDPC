# SPDX-License-Identifier: Apache-2.0

"""Classes to define syndrome measurement strategies."""

from __future__ import annotations

import abc
import collections
from collections.abc import Sequence

import numpy as np
import stim

from qldpc import codes, math
from qldpc._utils import networkx as nx
from qldpc.objects import Pauli

from ..bookkeeping import MeasurementRecord, QubitIDs
from ..common import _restrict_to_qubits


class SyndromeMeasurementStrategy(abc.ABC):
    """Base class for a syndrome measurement strategy.

    Subclasses define how to measure the (mutually commuting) parity checks of a stabilizer code by
    implementing get_circuit.  The gauge checks of a subsystem code need not commute, so they cannot
    all be measured at once.  Instead, get_subsystem_circuit partitions the gauge checks of a code
    into layers of mutually commuting checks (see get_gauge_layers), and then measures one layer at
    a time with get_circuit.
    """

    @_restrict_to_qubits
    @abc.abstractmethod
    def get_circuit(
        self, code: codes.QuditCode, qubit_ids: QubitIDs | None = None
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Construct a circuit to measure the syndromes of a quantum error-correcting code.

        Args:
            code: The code whose syndromes we want to measure.
            qubit_ids: Integer indices for the data and check (syndrome readout) qubits.  Defaults
                to QubitIDs.from_code(code).

        Returns:
            stim.Circuit: A syndrome measurement circuit.
            circuits.MeasurementRecord: The record of measurements in the circuit.
        """

    def get_gauge_layers(self, code: codes.QuditCode) -> tuple[tuple[int, ...], ...]:
        """Partition the parity checks of a code into layers of mutually commuting checks.

        The layers are measured one at a time, in the returned order, by get_subsystem_circuit.  A
        memory experiment uses these layers to define detectors: any product of the checks in a
        single layer that is a stabilizer of the code should have the same value in consecutive
        rounds of syndrome measurement.  A subsystem-code memory experiment therefore requires that
        such single-layer products of checks generate all of the stabilizers that it tracks.

        By default, the checks of a CSS code are split into a layer of all X-type checks followed by
        a layer of all Z-type checks, as in the gauge measurement schedule for Bacon-Shor codes in
        arXiv:quant-ph/0610063.  The checks of a non-CSS code are split into layers by greedily
        coloring a graph whose edges connect non-commuting checks.  Subclasses may override this
        method to choose a different schedule.

        Args:
            code: The code whose checks we want to partition.

        Returns:
            tuple[tuple[int, ...], ...]: The indices (rows of code.matrix) of the checks in each
            layer.
        """
        if isinstance(code, codes.CSSCode):
            layers = (range(code.num_checks_x), range(code.num_checks_x, code.num_checks))
            return tuple(tuple(layer) for layer in layers if layer)

        # color a graph whose vertices are checks and whose edges connect non-commuting checks
        commutators = code.matrix @ math.symplectic_conjugate(code.matrix).T
        graph = nx.Graph()
        graph.add_nodes_from(range(code.num_checks))
        graph.add_edges_from(zip(*map(np.ndarray.tolist, np.nonzero(commutators))))
        coloring = nx.greedy_color(graph, "smallest_last")
        color_to_checks: dict[int, list[int]] = collections.defaultdict(list)
        for check, color in sorted(coloring.items()):
            color_to_checks[color].append(check)
        return tuple(tuple(color_to_checks[color]) for color in sorted(color_to_checks))

    @_restrict_to_qubits
    def get_subsystem_circuit(
        self,
        code: codes.QuditCode,
        qubit_ids: QubitIDs | None = None,
        *,
        layers: Sequence[Sequence[int]] | None = None,
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Construct a circuit that measures the checks of a code one commuting layer at a time.

        The checks in each layer define a stabilizer code, whose checks are measured with
        self.get_circuit.  The circuits for different layers are applied sequentially, so every
        operation of one layer on a qubit precedes every operation of the next layer on that qubit.
        This method thereby supports subsystem codes, whose (gauge) checks need not commute.

        Every operation of the combined circuit then moves to the earliest moment (block of
        operations between TICK instructions) that preserves the sequence of operations on each
        qubit and the order of all measurements.  Operations of consecutive layers can thereby share
        moments if they address disjoint qubits.  For example, the check qubits of one layer can be
        reset, and the first gates of that layer can be applied to data qubits that the previous
        layer is done with, while the previous layer is still in progress.  This rescheduling only
        applies if the circuits of all layers consist of unitary gates, resets, and measurements on
        qubits.

        Rescheduling does not change the action of the circuit, and Stim attaches no meaning to
        moments.  However, noise models that act on each moment, such as models with idle errors
        (see NoiseModel), add fewer errors to a circuit with fewer moments.  Treating moments as
        time steps assumes that every operation takes the same amount of time, and that resets and
        measurements can occur at the same time as gates on other qubits.

        Since self.get_circuit only sees the stabilizer code defined by one layer at a time, any
        code-specific scheduling (such as a code's custom get_syndrome_subgraphs) or optimization
        (such as the logical error rates targeted by AlphaSyndrome) is with respect to that layer
        code rather than the full subsystem code.  In particular, the layer code treats the gauge
        operators of the subsystem code as logical operators.

        Args:
            code: The code whose checks we want to measure.
            qubit_ids: Integer indices for the data and check (syndrome readout) qubits, with one
                check qubit per row of code.matrix.  Defaults to QubitIDs.from_code(code).
            layers: A partition of the checks of the code (identified by their rows in code.matrix)
                into layers of mutually commuting checks.  Defaults to self.get_gauge_layers(code).

        Returns:
            stim.Circuit: A circuit that measures all checks of the code.
            circuits.MeasurementRecord: The record of measurements in the circuit.
        """
        qubit_ids = QubitIDs.validated(qubit_ids or QubitIDs.from_code(code), code)
        layers = self.get_gauge_layers(code) if layers is None else layers
        _validate_gauge_layers(code, layers)

        circuit = stim.Circuit()
        measurement_record = MeasurementRecord()
        for layer in layers:
            checks = sorted(layer)
            layer_code: codes.QuditCode
            if isinstance(code, codes.CSSCode):
                checks_x = [check for check in checks if check < code.num_checks_x]
                checks_z = [check - code.num_checks_x for check in checks[len(checks_x) :]]
                layer_code = codes.CSSCode(
                    code.matrix_x[checks_x], code.matrix_z[checks_z], is_subsystem_code=False
                )
            else:
                layer_code = codes.QuditCode(code.matrix[checks], is_subsystem_code=False)
            layer_qubit_ids = QubitIDs(
                qubit_ids.data,
                [qubit_ids.check[check] for check in checks],
                qubit_ids.ancilla,
                reference=qubit_ids.reference,
            )
            layer_circuit, layer_record = self.get_circuit(layer_code, layer_qubit_ids)
            layer_record.validate_num_measurements(layer_circuit.num_measurements)
            circuit += layer_circuit
            measurement_record.append(layer_record)
        return _schedule_as_soon_as_possible(circuit), measurement_record


class EdgeColoring(SyndromeMeasurementStrategy):
    """Edge coloration strategy for constructing a syndrome measurement circuit.

    Every edge of a code's Tanner graph is associated with a two-qubit gate that needs to be applied
    to "write" parity checks onto ancilla qubits (i.e., for syndrome extraction).  This syndrome
    measurement strategy iterates over the subgraphs of a code's Tanner graph in the order specified
    by code.get_syndrome_subgraphs().  For each subgraph, this strategy colors the edges of that
    subgraph such that no pair of vertex-adjacent edges share the same color, and then applies the
    corresponding gates one color at a time.

    .. warning::
        This strategy is not guaranteed to be distance-preserving or fault-tolerant.
    """

    def __init__(self, strategy: str = "smallest_last", **subgraph_kwargs: object) -> None:
        """Initialize an EdgeColoring syndrome measurement strategy.

        Args:
            strategy: The graph coloration strategy passed to nx.greedy_color when coloring edges.
                Defaults to "smallest_last".
            subgraph_kwargs: Keyword arguments to pass to custom ``code.get_syndrome_subgraphs``
                overrides.  Built-in code families accept only their own ``strategy`` argument; this
                extension point is intentionally not used by them.
        """
        self.strategy = strategy
        self.subgraph_kwargs = subgraph_kwargs

    @_restrict_to_qubits
    def get_circuit(
        self, code: codes.QuditCode, qubit_ids: QubitIDs | None = None
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Construct a circuit to measure the syndromes of a quantum error-correcting code.

        Args:
            code: The code whose syndromes we want to measure.
            qubit_ids: Integer indices for the data and check (syndrome readout) qubits.  Defaults
                to QubitIDs.from_code(code).

        Returns:
            stim.Circuit: A syndrome measurement circuit.
            circuits.MeasurementRecord: The record of measurements in the circuit.
        """
        qubit_ids = _validate_syndrome_qubit_ids(code, qubit_ids)
        # subgraph_kwargs are meant for custom overrides, whose signatures mypy cannot know
        subgraphs = code.get_syndrome_subgraphs(**self.subgraph_kwargs)  # type:ignore[arg-type]
        return self._get_circuit_from_subgraphs(qubit_ids, subgraphs)

    @staticmethod
    def graph_to_circuit(graph: nx.DiGraph, qubit_ids: QubitIDs, strategy: str) -> stim.Circuit:
        """Convert a Tanner (sub)graph into a syndrome extraction circuit.

        Edges of the graph correspond to two-qubit controlled-pauli (i.e., CX, CY, or CZ) gates.
        This method colors all edges to identify subsets of qubit-disjoint gates, and then applies
        the corresponding gates one color at a time.

        Assumptions:

        - All two-qubit gates associated with edges in the graph commute.
        - Check qubits are initialized ``|+>``.

        The caller supplies the initial framing ``TICK``; each colored layer contributes one
        trailing ``TICK``.  This keeps ``num_ticks`` equal to the scheduled gate depth while making
        standalone rounds composable.
        """
        # color the edges of the Tanner graph
        coloring = nx.greedy_color(nx.line_graph(graph.to_undirected()), strategy)

        # collect operations by color, in (gate, qubit_1, qubit_2) format
        color_to_ops: dict[int, list[tuple[str, int, int]]] = collections.defaultdict(list)
        for edge, color in coloring.items():
            data_node, check_node = sorted(edge)
            data_id = qubit_ids.data[data_node.index]
            check_id = qubit_ids.check[check_node.index]
            pauli = graph[check_node][data_node][Pauli]
            color_to_ops[color].append((f"C{pauli}", check_id, data_id))

        # collect all gates into a circuit
        circuit = stim.Circuit()
        for gates in color_to_ops.values():
            for gate, check_id, data_id in sorted(gates):
                circuit.append(gate, [check_id, data_id])
            circuit.append("TICK")  # separate scheduled gate layers
        return circuit

    def _get_circuit_from_subgraphs(
        self,
        qubit_ids: QubitIDs,
        subgraphs: tuple[nx.DiGraph, ...],
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Build a circuit and record from an already selected subgraph sequence."""
        circuit = stim.Circuit()
        circuit.append("RX", qubit_ids.check)
        circuit.append("TICK")
        for subgraph in subgraphs:
            circuit += EdgeColoring.graph_to_circuit(subgraph, qubit_ids, self.strategy)
        circuit.append("MX", qubit_ids.check)
        circuit.append("TICK")

        measurement_record = MeasurementRecord(
            {qubit: [mm] for mm, qubit in enumerate(qubit_ids.check)}
        )
        return circuit, measurement_record


class EdgeColoringXZ(EdgeColoring):
    """Edge coloration syndrome measurement strategy in Algorithm 1 of arXiv:2109.14609.

    For a CSS code with Tanner graph T, this strategy is as follows:

    1. Construct the subgraphs ``T_X`` and ``T_Z`` of ``T`` restricted, respectively, to X and Z
       stabilizers.
    2. For each ``T_P`` in ``(T_X, T_Z)``, color the edges of ``T_P``, and then apply all
       corresponding gates one color at a time.

    .. warning::
        This strategy is not guaranteed to be distance-preserving or fault-tolerant.
    """

    def __init__(self, strategy: str = "smallest_last") -> None:
        """Initialize an EdgeColoringXZ syndrome measurement strategy.

        Args:
            strategy: The graph coloration strategy passed to nx.greedy_color when coloring edges.
                Defaults to "smallest_last".
        """
        self.strategy = strategy
        self.subgraph_kwargs: dict[str, object] = {}

    @_restrict_to_qubits
    def get_circuit(
        self, code: codes.QuditCode, qubit_ids: QubitIDs | None = None
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Construct a circuit to measure the syndromes of a quantum error-correcting code.

        Args:
            code: The code whose syndromes we want to measure.
            qubit_ids: Integer indices for the data and check (syndrome readout) qubits.  Defaults
                to QubitIDs.from_code(code).

        Returns:
            stim.Circuit: A syndrome measurement circuit.
            circuits.MeasurementRecord: The record of measurements in the circuit.
        """
        if not isinstance(code, codes.CSSCode):
            raise TypeError(
                "The EdgeColoringXZ strategy for syndrome measurement only supports CSS codes"
            )

        qubit_ids = _validate_syndrome_qubit_ids(code, qubit_ids)
        return self._get_circuit_from_subgraphs(qubit_ids, (code.graph_x, code.graph_z))


def _validate_syndrome_qubit_ids(
    code: codes.QuditCode, qubit_ids: QubitIDs | None = None
) -> QubitIDs:
    """Validate the qubit layout for a syndrome-measurement strategy.

    Args:
        code: The code whose syndromes will be measured.
        qubit_ids: Integer indices for the data and check qubits.  Defaults to
            ``QubitIDs.from_code(code)``.

    Returns:
        A validated qubit layout.

    Raises:
        ValueError: If the code is a subsystem code or the qubit layout is invalid.
    """
    if code.is_subsystem_code:
        raise ValueError(
            "Syndrome measurement strategies require a non-subsystem code (measure the gauge checks"
            " of a subsystem code with SyndromeMeasurementStrategy.get_subsystem_circuit)"
        )
    return QubitIDs.validated(qubit_ids or QubitIDs.from_code(code), code)


def _validate_gauge_layers(code: codes.QuditCode, layers: Sequence[Sequence[int]]) -> None:
    """Validate a partition of the checks of a code into layers of mutually commuting checks.

    Args:
        code: The code whose checks are partitioned.
        layers: The indices (rows of code.matrix) of the checks in each layer.

    Raises:
        ValueError: If the layers do not partition the checks of the code, or if a layer contains
            checks that do not commute.
    """
    if sorted(check for layer in layers for check in layer) != list(range(code.num_checks)):
        raise ValueError("Gauge layers must partition the parity checks of the code")
    for layer in layers:
        checks = code.matrix[list(layer)]
        if np.any(checks @ math.symplectic_conjugate(checks).T):
            raise ValueError(f"Gauge layer {tuple(layer)} contains checks that do not commute")


def _schedule_as_soon_as_possible(circuit: stim.Circuit) -> stim.Circuit:
    """Move every operation in a circuit to the earliest possible moment.

    Each operation (one application of a gate to its targets) moves to the moment after the last
    preceding operation that addresses one of its qubits.  The sequence of operations on each qubit
    is therefore unchanged.  Measurements additionally keep their order in the measurement record.
    Reordering operations on disjoint qubits does not change the action of a circuit, so the
    rescheduled circuit is equivalent to the original circuit, but may have fewer moments.

    Circuits that contain anything other than unitary gates, resets, and measurements on qubits (and
    TICK instructions that separate moments) are returned unchanged.
    """
    instructions = list(circuit)
    if not all(_can_reschedule(instruction) for instruction in instructions):
        return circuit.copy()

    moments: list[list[stim.CircuitInstruction]] = []
    next_free_moment: dict[int, int] = collections.defaultdict(int)
    last_measurement_moment = 0
    for instruction in instructions:
        assert isinstance(instruction, stim.CircuitInstruction)
        if instruction.name == "TICK":
            continue
        gate_data = stim.gate_data(instruction.name)
        for targets in _split_targets(instruction, gate_data):
            qubits = [target.qubit_value for target in targets if target.qubit_value is not None]
            moment = max((next_free_moment[qubit] for qubit in qubits), default=0)
            if gate_data.produces_measurements:
                moment = last_measurement_moment = max(moment, last_measurement_moment)
            moments.extend([] for _ in range(moment + 1 - len(moments)))
            moments[moment].append(
                stim.CircuitInstruction(
                    instruction.name, targets, instruction.gate_args_copy(), tag=instruction.tag
                )
            )
            for qubit in qubits:
                next_free_moment[qubit] = moment + 1

    scheduled_circuit = stim.Circuit()
    for mm, moment_instructions in enumerate(moments):
        if mm:
            scheduled_circuit.append("TICK")
        for instruction in moment_instructions:
            scheduled_circuit.append(instruction)
    if (
        instructions
        and isinstance(instructions[-1], stim.CircuitInstruction)
        and (instructions[-1].name == "TICK")
    ):
        scheduled_circuit.append("TICK")
    return scheduled_circuit


def _can_reschedule(instruction: stim.CircuitInstruction | stim.CircuitRepeatBlock) -> bool:
    """Can _schedule_as_soon_as_possible move this instruction without changing its action?"""
    if isinstance(instruction, stim.CircuitRepeatBlock):
        return False
    if instruction.name == "TICK":
        return True
    gate_data = stim.gate_data(instruction.name)
    return (
        (gate_data.is_unitary or gate_data.is_reset or gate_data.produces_measurements)
        and bool(instruction.targets_copy())
        and all(
            target.qubit_value is not None or target.is_combiner
            for target in instruction.targets_copy()
        )
    )


def _split_targets(
    instruction: stim.CircuitInstruction, gate_data: stim.GateData
) -> list[list[stim.GateTarget]]:
    """Split the targets of an instruction into the targets of individual gate applications."""
    targets = instruction.targets_copy()
    if gate_data.is_single_qubit_gate:
        return [[target] for target in targets]
    if gate_data.is_two_qubit_gate:
        return [targets[tt : tt + 2] for tt in range(0, len(targets), 2)]
    return [targets]
