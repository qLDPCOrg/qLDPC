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
import dataclasses
from collections.abc import Sequence

import networkx as nx
import stim

from qldpc import codes


@dataclasses.dataclass
class QubitIds:
    """Container for qubit indices."""

    data: list[int]
    x_check: list[int]
    z_check: list[int]

    def __init__(self, num_data: int, num_checks_x: int, num_checks_z: int) -> None:
        """Initialize from a number of data, X-check, and Z-check qubits."""
        self.data = list(range(num_data))
        self.x_check: list[int] = list(range(num_data, num_data + num_checks_x))
        self.z_check: list[int] = list(
            range(
                num_data + num_checks_x,
                num_data + num_checks_x + num_checks_z,
            )
        )


class SyndromeMeasurementStrategy(abc.ABC):
    """Base class for a syndrome measurement strategy."""

    @abc.abstractmethod
    def get_circuit(
        self, code: codes.CSSCode, stim_ids: QubitIds | None = None
    ) -> tuple[stim.Circuit, list[list[int]]]:
        """Compiles a syndrome measurement circuit for a given CSSCode and noise model.

        Args:
            codes.CSSCode:
                The quantum code to be compiled into a single round of syndrome measurements.
            QubitIds:
                Integer indices to be used for data qubits, X check qubits, and Z check qubits.

        Returns:
            stim.Circuit:
                Stim circuit containing the compiled syndrome measurement round
            list[list[int]]:
                The history of measurement rounds performed in the circuit.  Each round is a list of
                the stim ids measured that round, in the order they were passed to stim.
        """


class BareColorCircuit(SyndromeMeasurementStrategy):
    """A coloration circuit syndrome measurement scheme as defined in https://arxiv.org/abs/2109.14609 (Algorithm 1).

    WARNING: This scheme is not guaranteed to be fault-tolerant or distance-preserving.
    """

    def get_circuit(
        self,
        code: codes.CSSCode,
        qubit_ids: QubitIds | None = None,
        strategy: str = "largest_first",
    ) -> tuple[stim.Circuit, list[list[int]]]:
        """
        Compiles a coloration circuit. Not depth-optimal as no interleaving of opposite type checks is present. Z checks are performed first followed by X checks
        """
        if qubit_ids is not None:
            assert len(code) == len(qubit_ids.data)
            assert code.num_checks_x == len(qubit_ids.x_check)
            assert code.num_checks_z == len(qubit_ids.z_check)
        else:
            qubit_ids = QubitIds(len(code), code.num_checks_x, code.num_checks_z)

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
