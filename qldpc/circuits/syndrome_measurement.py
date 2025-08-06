"""Classes to define syndrome measurement strategies

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

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
import dataclasses
from collections.abc import Iterator

import networkx as nx
import stim

from qldpc import codes
from qldpc.objects import Node, Pauli

from .common import restrict_to_qubits


@dataclasses.dataclass
class QubitIDs:
    """Container for data qubit and check (syndrome readout) qubit indices."""

    data: list[int]  # data qubit indices
    check: list[int]  # check (syndrome readout) qubit indices

    @staticmethod
    def from_code(code: codes.QuditCode) -> QubitIDs:
        """Initialize from an error-correcting code with specific parity checks."""
        data = list(range(len(code)))
        check = list(range(len(code), len(code) + code.num_checks))
        return QubitIDs(data, check)


class MeasurementRecord:
    """Store a measurement record in a Stim circuit."""

    num_measurements: int
    qubit_to_measurement: dict[int, list[int]]

    def __init__(self, initial_record: dict[int, list[int]] | None = None) -> None:
        self.qubit_to_measurement = collections.defaultdict(
            list, initial_record if initial_record else {}
        )
        self.num_measurements = sum(
            len(measurements) for measurements in self.qubit_to_measurement.values()
        )

    def items(self) -> Iterator[tuple[int, list[int]]]:
        """Iterator over qubits and their measurements."""
        yield from self.qubit_to_measurement.items()

    def append(self, record: MeasurementRecord | dict[int, list[int]]) -> None:
        """Append the given record to this one."""
        for qubit, measurements in record.items():
            self.qubit_to_measurement[qubit].extend(
                [self.num_measurements + measurement for measurement in measurements]
            )
        self.num_measurements += sum(len(measurements) for _, measurements in record.items())

    def get_last_target_rec(self, qubit_index: int) -> stim.target_rec:
        """Get the most recent Stim measurement record target (by index) for the given qubit."""
        if qubit_index not in self.qubit_to_measurement:
            raise ValueError(f"Qubit {qubit_index} not found in measurement record")
        return stim.target_rec(self.qubit_to_measurement[qubit_index][-1] - self.num_measurements)


class SyndromeMeasurementStrategy(abc.ABC):
    """Base class for a syndrome measurement strategy."""

    @staticmethod
    @restrict_to_qubits
    @abc.abstractmethod
    def get_circuit(
        code: codes.QuditCode, qubit_ids: QubitIDs | None = None
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Construct a circuit to measure the syndromes of a quantum error-correcting code.

        Args:
            codes.QuditCode: The code whose syndromes we want to measure.
            circuits.QubitIDs: Integer indices for the data and check (syndrome readout) qubits.
                Defaults to QubitIDs.from_code(code).

        Returns:
            stim.Circuit: A syndrome measurement circuit.
            circuits.MeasurementRecord: The record of measurements in the circuit.
        """


class EdgeColoring(SyndromeMeasurementStrategy):
    """Coloration strategy for syndrome measurement in arXiv:2109.14609.

    WARNING: This strategy is not guaranteed to be distance-preserving or fault-tolerant.
    """

    @staticmethod
    @restrict_to_qubits
    def get_circuit(
        code: codes.QuditCode, qubit_ids: QubitIDs | None = None, *, strategy: str = "largest_first"
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Construct a syndrome measurement circuit using Algorithm 1 of arXiv:2109.14609.

        Args:
            codes.QuditCode: The code whose syndromes we want to measure.
            circuits.QubitIDs: Integer indices for the data and check (syndrome readout) qubits.
                Defaults to QubitIDs.from_code(code).
            strategy: The graph coloration stratepy passed to nx.greedy_color.
                Defaults to "largest_first".

        Returns:
            stim.Circuit: A syndrome measurement circuit.
            circuits.MeasurementRecord: The record of measurements in the circuit.
        """
        qubit_ids = qubit_ids or QubitIDs.from_code(code)

        circuit = stim.Circuit()
        circuit.append("RX", qubit_ids.check)
        circuit.append("TICK")

        data_nodes = [Node(index, is_data=True) for index in qubit_ids.data]
        check_nodes = [Node(index, is_data=False) for index in range(code.num_checks)]
        if isinstance(code, codes.CSSCode):
            check_nodes_x = check_nodes[: code.num_checks_x]
            check_nodes_z = check_nodes[code.num_checks_x :]
            graph_x = code.graph.subgraph(data_nodes + check_nodes_x)
            graph_z = code.graph.subgraph(data_nodes + check_nodes_z)
            circuit += EdgeColoring.graph_to_circuit(graph_x, qubit_ids, strategy)
            circuit += EdgeColoring.graph_to_circuit(graph_z, qubit_ids, strategy)
        else:
            circuit += EdgeColoring.graph_to_circuit(code.graph, qubit_ids, strategy)

        circuit.append("MX", qubit_ids.check)
        measurement_record = MeasurementRecord(
            {qubit: [num] for num, qubit in enumerate(qubit_ids.check)}
        )
        return circuit, measurement_record

    def graph_to_circuit(graph: nx.DiGraph, qubit_ids: QubitIDs, strategy: str) -> stim.Circuit:
        """Convert a Tanner graph into a syndrome extraction circuit.

        Assumes that check qubits are initialized |+>.
        """
        # color the edges of the Tanner graph
        coloring = nx.coloring.greedy_color(nx.line_graph(graph.to_undirected()), strategy)

        # collect operations by color, in (gate, qubit_1, qubit_2) format
        color_to_ops: dict[int, list[tuple[str, int, int]]] = collections.defaultdict(list)
        for edge, color in coloring.items():
            data_node, check_node = sorted(edge)
            data_qubit = qubit_ids.data[data_node.index]
            check_qubit = qubit_ids.check[check_node.index]
            pauli = graph[check_node][data_node][Pauli]
            color_to_ops[color].append((f"C{pauli}", check_qubit, data_qubit))

        # collect all gates into a circuit
        circuit = stim.Circuit()
        for gates in color_to_ops.values():
            for gate, check_qubit, data_qubit in sorted(gates):
                circuit.append(gate, [check_qubit, data_qubit])
            circuit.append("TICK")
        return circuit
