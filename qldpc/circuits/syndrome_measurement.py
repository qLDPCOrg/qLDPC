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
from qldpc.objects import Pauli

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

    def __iter__(self) -> Iterator[list[int]]:
        yield from (self.data, self.check)


class MeasurementRecord:
    """Store a measurement record in a Stim circuit."""

    num_measurements: int
    qubit_to_measurements: dict[int, list[int]]

    def __init__(self, initial_record: dict[int, list[int]] | None = None) -> None:
        self.qubit_to_measurements = collections.defaultdict(
            list, initial_record if initial_record else {}
        )
        self.num_measurements = sum(
            len(measurements) for measurements in self.qubit_to_measurements.values()
        )

    def items(self) -> Iterator[tuple[int, list[int]]]:
        """Iterator over qubits and their measurements."""
        yield from self.qubit_to_measurements.items()

    def append(self, record: MeasurementRecord | dict[int, list[int]]) -> None:
        """Append the given record to this one."""
        for qubit, measurements in record.items():
            self.qubit_to_measurements[qubit].extend(
                [self.num_measurements + measurement for measurement in measurements]
            )
        self.num_measurements += sum(len(measurements) for _, measurements in record.items())

    def get_target_rec(self, qubit: int, measurement_index: int = -1) -> stim.target_rec:
        """Retrieve a Stim measurement record target for the given qubit.

        Args:
            qubit: the qubit (by index) whose measurement record we want.
            measurement_index: an index specifying which measurement of the specified qubit we want.
                A measurement_index of 0 would be the first measurement of the qubit, while a
                measurement_index of -1 would be the most recent measurement.  Default value: -1.

        Returns:
            stim.target_rec: a Stim measurement record target.
        """
        if qubit not in self.qubit_to_measurements:
            raise ValueError(f"Qubit {qubit} not found in measurement record")
        measurements = self.qubit_to_measurements[qubit]
        if not -len(measurements) <= measurement_index < len(measurements):
            raise ValueError(
                f"Invalid measurement index {measurement_index} for qubit {qubit} with "
                f"{len(measurements)} measurements"
            )
        return stim.target_rec(measurements[measurement_index] - self.num_measurements)


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


class SerialExtraction(SyndromeMeasurementStrategy):
    """Serialize syndrome extraction according to a code's parity check matrix.

    WARNING: This strategy is not guaranteed to be distance-preserving or fault-tolerant.
    """

    @staticmethod
    @restrict_to_qubits
    def get_circuit(
        code: codes.QuditCode, qubit_ids: QubitIDs | None = None
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Construct a syndrome measurement circuit using Algorithm 1 of arXiv:2109.14609.

        Args:
            codes.QuditCode: The code whose syndromes we want to measure.
            circuits.QubitIDs: Integer indices for the data and check (syndrome readout) qubits.
                Defaults to QubitIDs.from_code(code).

        Returns:
            stim.Circuit: A syndrome measurement circuit.
            circuits.MeasurementRecord: The record of measurements in the circuit.
        """
        qubit_ids = qubit_ids or QubitIDs.from_code(code)

        circuit = stim.Circuit()
        circuit.append("RX", qubit_ids.check)

        # write syndromes to ancilla qubits one at a time
        for check_qubit, check in enumerate(code.matrix, start=len(code)):
            for data_qubit, pauli_xz in enumerate(check.reshape(2, len(code)).T):
                pauli = Pauli(tuple(pauli_xz))
                if pauli is not Pauli.I:
                    circuit.append(f"C{pauli}", [check_qubit, data_qubit])

        circuit.append("MX", qubit_ids.check)
        measurement_record = MeasurementRecord(
            {qubit: [num] for num, qubit in enumerate(qubit_ids.check)}
        )
        return circuit, measurement_record


class EdgeColoring(SyndromeMeasurementStrategy):
    """Edge coloration syndrome measurement strategy inspired by Algorithm 2 of arXiv:2109.14609.

    For a CSS code with Tanner graph T, Algorithm 2 of arXiv:2109.14609 proceeds as follows:
    1. Assign each edge in T a cardinal direction D in {E, N, S, W}.
    2. For each D in (E, N, S, W), consider the subgraph T_D, color the edges of T_D, and apply the
        corresponding gates one color at a time.

    The EdgeColoring syndrome measurement strategy here slightly generalizes that in Algorithm 2 of
    arXiv:2109.14609.  Specifically, this strategy delegates the division of the Tanner graph T into
    subgraphs T_D (i.e., step 1 above) to the error-correcting code, and in step 2 above iterates
    over each subgraph T_D in code.syndrome_subgraphs.

    WARNING: This strategy is not guaranteed to be distance-preserving or fault-tolerant.
    """

    @staticmethod
    @restrict_to_qubits
    def get_circuit(
        code: codes.QuditCode, qubit_ids: QubitIDs | None = None, *, strategy: str = "largest_first"
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        """Construct a syndrome measurement circuit using Algorithm 2 of arXiv:2109.14609.

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
        if code.syndrome_subgraphs is NotImplemented:
            raise ValueError(
                "The provided code is not equipped with a syndrome_subgraphs property, as required"
                " for the EdgeColoring syndrome measurement strategy"
            )

        qubit_ids = qubit_ids or QubitIDs.from_code(code)
        circuit = stim.Circuit()
        circuit.append("RX", qubit_ids.check)
        for graph in code.syndrome_subgraphs:
            circuit += EdgeColoring.graph_to_circuit(graph, qubit_ids, strategy)
        circuit.append("MX", qubit_ids.check)

        measurement_record = MeasurementRecord(
            {qubit: [num] for num, qubit in enumerate(qubit_ids.check)}
        )
        return circuit, measurement_record

    @staticmethod
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
        return circuit


class EdgeColoringXZ(SyndromeMeasurementStrategy):
    """Edge coloration syndrome measurement strategy in Algorithm 1 of arXiv:2109.14609.

    For a CSS code with Tanner graph T, this strategy is as follows:
    1. Construct the subgraphs T_X and T_Z of T restricted, respectively, to X and Z stabilizers.
    2. For each T_P in {T_X, T_Z}, color the edges of T_P, and then apply all corresponding gates
        one color at a time.

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
        if not isinstance(code, codes.CSSCode):
            raise ValueError(
                "The EdgeColoringXZ strategy for syndrome measurement only supports CSS codes"
            )

        qubit_ids = qubit_ids or QubitIDs.from_code(code)
        circuit = stim.Circuit()
        circuit.append("RX", qubit_ids.check)
        circuit += EdgeColoring.graph_to_circuit(code.graph_x, qubit_ids, strategy)
        circuit += EdgeColoring.graph_to_circuit(code.graph_z, qubit_ids, strategy)
        circuit.append("MX", qubit_ids.check)

        measurement_record = MeasurementRecord(
            {qubit: [num] for num, qubit in enumerate(qubit_ids.check)}
        )
        return circuit, measurement_record
