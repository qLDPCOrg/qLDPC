"""Classes to define syndrome measurement strategies.

Copyright 2025 The qLDPC Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import abc
import collections

import stim

from qldpc import codes
from qldpc._util import networkx as nx
from qldpc.objects import Pauli

from ..bookkeeping import MeasurementRecord, QubitIDs
from ..common import restrict_to_qubits


def validate_syndrome_qubit_ids(
    code: codes.QuditCode, qubit_ids: QubitIDs | None = None
) -> QubitIDs:
    """Validate the qubit layout for a syndrome-measurement strategy.

    Args:
        code: The code whose syndromes will be measured.
        qubit_ids: Integer indices for the data and check qubits. Defaults to
            ``QubitIDs.from_code(code)``.

    Returns:
        A validated qubit layout.

    Raises:
        ValueError: If the code is a subsystem code or the qubit layout is invalid.
    """
    if code.is_subsystem_code:
        raise ValueError("Syndrome measurement strategies require a non-subsystem code")
    return QubitIDs.validated(qubit_ids or QubitIDs.from_code(code), code)


class SyndromeMeasurementStrategy(abc.ABC):
    """Base class for a syndrome measurement strategy."""

    @restrict_to_qubits
    @abc.abstractmethod
    def get_circuit(
        self, code: codes.QuditCode, qubit_ids: QubitIDs | None = None
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Construct a circuit to measure the syndromes of a quantum error-correcting code.

        Args:
            code: The code whose syndromes we want to measure.
            qubit_ids: Integer indices for the data and check (syndrome readout) qubits.
                Defaults to QubitIDs.from_code(code).

        Returns:
            stim.Circuit: A syndrome measurement circuit.
            circuits.MeasurementRecord: The record of measurements in the circuit.
        """


class EdgeColoring(SyndromeMeasurementStrategy):
    """Edge coloration strategy for constructing a syndrome measurement circuit.

    Every edge of a code's Tanner graph is associated with a two-qubit gate that needs to be applied
    to "write" parity checks onto ancilla qubits (i.e., for syndrome extraction).  This syndrome
    measurement strategy iterates over the subgraphs of a code's Tanner graph in the order specified
    by code.get_syndrome_subgraphs().  For each subgraph, this strategy colors the edges of that
    subgraph such that no pair of vertex-adjacent edges share the same color, and then applies the
    corresponding gates one color at a time.

    WARNING: This strategy is not guaranteed to be distance-preserving or fault-tolerant.
    """

    def __init__(self, strategy: str = "smallest_last", **subgraph_kwargs: object) -> None:
        """Initialize an EdgeColoring syndrome measurement strategy.

        Args:
            strategy: The graph coloration strategy passed to nx.greedy_color when coloring edges.
                Defaults to "smallest_last".
            subgraph_kwargs: Keyword arguments to pass to custom
                ``code.get_syndrome_subgraphs`` overrides. Built-in code families accept only their
                own ``strategy`` argument; this extension point is intentionally not used by them.
        """
        self.strategy = strategy
        self.subgraph_kwargs = subgraph_kwargs

    @restrict_to_qubits
    def get_circuit(
        self, code: codes.QuditCode, qubit_ids: QubitIDs | None = None
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Construct a circuit to measure the syndromes of a quantum error-correcting code.

        Args:
            code: The code whose syndromes we want to measure.
            qubit_ids: Integer indices for the data and check (syndrome readout) qubits.
                Defaults to QubitIDs.from_code(code).

        Returns:
            stim.Circuit: A syndrome measurement circuit.
            circuits.MeasurementRecord: The record of measurements in the circuit.
        """
        qubit_ids = validate_syndrome_qubit_ids(code, qubit_ids)
        subgraphs = code.get_syndrome_subgraphs(**self.subgraph_kwargs)  # type:ignore[arg-type]
        return self._get_circuit_from_subgraphs(qubit_ids, subgraphs)

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
        trailing ``TICK``. This keeps ``num_ticks`` equal to the scheduled gate depth while making
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


class EdgeColoringXZ(EdgeColoring):
    """Edge coloration syndrome measurement strategy in Algorithm 1 of arXiv:2109.14609.

    For a CSS code with Tanner graph T, this strategy is as follows:

    1. Construct the subgraphs T_X and T_Z of T restricted, respectively, to X and Z stabilizers.
    2. For each T_P in (T_X, T_Z), color the edges of T_P, and then apply all corresponding gates
        one color at a time.

    WARNING: This strategy is not guaranteed to be distance-preserving or fault-tolerant.
    In particular, the EdgeColoringXZ schedule can reduce the circuit-level distance of rotated
    surface codes; use it only when that trade-off is acceptable.
    """

    def __init__(self, strategy: str = "smallest_last") -> None:
        """Initialize an EdgeColoringXZ syndrome measurement strategy.

        Args:
            strategy: The graph coloration strategy passed to nx.greedy_color when coloring edges.
                Defaults to "smallest_last".
        """
        self.strategy = strategy
        self.subgraph_kwargs: dict[str, object] = {}

    @restrict_to_qubits
    def get_circuit(
        self, code: codes.QuditCode, qubit_ids: QubitIDs | None = None
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Construct a circuit to measure the syndromes of a quantum error-correcting code.

        Args:
            code: The code whose syndromes we want to measure.
            qubit_ids: Integer indices for the data and check (syndrome readout) qubits.
                Defaults to QubitIDs.from_code(code).

        Returns:
            stim.Circuit: A syndrome measurement circuit.
            circuits.MeasurementRecord: The record of measurements in the circuit.
        """
        if not isinstance(code, codes.CSSCode):
            raise TypeError(
                "The EdgeColoringXZ strategy for syndrome measurement only supports CSS codes"
            )

        qubit_ids = validate_syndrome_qubit_ids(code, qubit_ids)
        return self._get_circuit_from_subgraphs(qubit_ids, (code.graph_x, code.graph_z))
