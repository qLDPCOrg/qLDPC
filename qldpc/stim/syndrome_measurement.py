from __future__ import annotations

import abc
import dataclasses
from collections.abc import Sequence

import networkx as nx
import stim

from qldpc import codes


@dataclasses.dataclass
class StimIds:
    data_ids: list[int]
    z_check_ids: list[int]
    x_check_ids: list[int]


class SyndromeMeasurementStrategy(abc.ABC):
    """Base class for a syndrome measurement scheme."""

    @abc.abstractmethod
    def compile_sm_circuit(
        self, code: codes.CSSCode, stim_ids: StimIds
    ) -> tuple[stim.Circuit, list[list[int]]]:
        """Compiles a syndrome measurement circuit for a given CSSCode and noise model.

        Args:
            codes.CSSCode:
                The quantum code to be compiled into a single round of syndrome measurements.
            StimIDs:
                Stim circuit ids to be used for data qubits, Z check qubits, and X check qubits.
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

    def _classical_subcode_to_subcircuit(
        self,
        code: codes.ClassicalCode,
        check_ids: Sequence[int],
        data_ids: Sequence[int],
        gate: str,
        strategy: str,
    ) -> stim.Circuit:
        coloring = nx.coloring.greedy_color(nx.line_graph(code.graph.to_undirected()), strategy)
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

    def compile_sm_circuit(
        self,
        code: codes.CSSCode,
        stim_ids: StimIds,
        strategy: str = "largest_first",
    ) -> tuple[stim.Circuit, list[list[int]]]:
        """
        Compiles a coloration circuit. Not depth-optimal as no interleaving of opposite type checks is present. Z checks are performed first followed by X checks
        """
        assert len(code) == len(stim_ids.data_ids)
        assert code.num_checks_x == len(stim_ids.x_check_ids)
        assert code.num_checks_z == len(stim_ids.z_check_ids)

        z_subcircuit = self._classical_subcode_to_subcircuit(
            code.code_z,
            stim_ids.z_check_ids,
            stim_ids.data_ids,
            "CZ",
            strategy,
        )
        x_subcircuit = self._classical_subcode_to_subcircuit(
            code.code_x,
            stim_ids.x_check_ids,
            stim_ids.data_ids,
            "CX",
            strategy,
        )

        circuit = stim.Circuit()

        # Initialize check qubits
        reset_qubits = stim_ids.z_check_ids + stim_ids.x_check_ids
        circuit.append("RX", reset_qubits)

        # "Write" Z and X stabilizers to check qubits
        circuit += z_subcircuit
        circuit += x_subcircuit

        # Measure the extracted stabilizers
        circuit.append("MX", reset_qubits)

        measurements = [stim_ids.z_check_ids + stim_ids.x_check_ids]
        return circuit, measurements
