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
from collections.abc import Iterator, Sequence

import networkx as nx
import stim

from qldpc import codes


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

    @abc.abstractmethod
    def get_circuit(
        self, code: codes.QuditCode, qubit_ids: QubitIDs | None = None
    ) -> tuple[stim.Circuit, list[list[int]]]:
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

    WARNING: This scheme is not guaranteed to be fault-tolerant or distance-preserving.
    """

    def get_circuit(
        self,
        code: codes.QuditCode,
        qubit_ids: QubitIDs | None = None,
        *,
        strategy: str = "largest_first",
    ) -> tuple[stim.Circuit, list[list[int]]]:
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

        x_subcircuit = self._classical_subcode_to_subcircuit(
            code.code_x,
            qubit_ids.x_check,
            qubit_ids.data,
            "CX",
            strategy,
        )
        z_subcircuit = self._classical_subcode_to_subcircuit(
            code.code_z,
            qubit_ids.z_check,
            qubit_ids.data,
            "CZ",
            strategy,
        )

        circuit = stim.Circuit()
        check_qubits = qubit_ids.x_check + qubit_ids.z_check

        # Initialize check qubits
        circuit.append("RX", check_qubits)

        # "Write" Z and X stabilizers to check qubits
        circuit += x_subcircuit
        circuit += z_subcircuit

        # Measure the extracted stabilizers
        circuit.append("MX", check_qubits)

        measurements = [qubit_ids.x_check + qubit_ids.z_check]
        return circuit, measurements

    def _classical_subcode_to_subcircuit(
        self,
        subcode: codes.ClassicalCode,
        check_ids: Sequence[int],
        data_ids: Sequence[int],
        gate: str,
        strategy: str,
    ) -> stim.Circuit:
        coloring = nx.coloring.greedy_color(nx.line_graph(subcode.graph.to_undirected()), strategy)
        circuit = stim.Circuit()

        schedule: dict[int, list[tuple[int, int]]] = {}
        for edge, color in coloring.items():
            assert edge[0].is_data ^ edge[1].is_data  # Assert valid edge (data <-> check)
            if edge[0].is_data:
                check_op = (check_ids[edge[1].index], data_ids[edge[0].index])
            else:
                check_op = (check_ids[edge[0].index], data_ids[edge[1].index])
            schedule.setdefault(color, []).append(check_op)
        for color, moment in schedule.items():
            for check_qubit, data_qubit in moment:
                circuit.append(gate, [check_qubit, data_qubit])
            if moment:  # Only add TICK if there were operations in this moment
                circuit.append("TICK")

        return circuit
