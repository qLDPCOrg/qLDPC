from __future__ import annotations

import abc
from dataclasses import dataclass

import networkx as nx
import stim

from qldpc.codes.common import ClassicalCode, CSSCode
from qldpc.objects import Pauli
from qldpc.stim.noise_model import NoiseModel


@dataclass
class StimIds:
    data_ids: list[int]
    z_check_ids: list[int]
    x_check_ids: list[int]


class SyndromeMeasurement(abc.ABC):
    """
    Base class for a syndrome measurement scheme
    """

    @abc.abstractmethod
    def compile_sm_circuit(
        self, code: CSSCode, noise_model: NoiseModel, stim_ids: StimIds
    ) -> tuple[stim.Circuit, list[list[int]]]:
        """
        Compiles a syndrome measurement circuit for a given CSSCode and noise model.

        args:
            code: CSSCode
                The quantum code to be compiled into a single round of syndrome measurements
            noise_model: NoiseModel
                The noise model to wrap stim operations (1q gate, 2q gate, meas, and reset)
            stim_ids: StimIds
                Stim circuit ids to be used for data qubits, z check qubits, and x check qubits
        returns:
            stim.Circuit:
                Stim circuit containg the compiled syndrome measurement round
            list[list[int]]:
                The history of measurement rounds performed in the circuit.
                Each round is a list of the stim ids measured that round, in the order they were passed to stim.
        """


class BareColorCircuit(SyndromeMeasurement):
    """
    A coloration circuit syndrome measurement scheme as defined in https://arxiv.org/abs/2109.14609 (Algorithm 1).

    NOTE: This is not guaranteed to be distance-preserving.
    """

    def _code_to_subcircuit(
        self,
        code: ClassicalCode,
        noise_model: NoiseModel,
        check_ids: list[int],
        data_ids: list[int],
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
            noise_model.apply_2q_gates(circuit, gate, moment)
        return circuit

    def compile_sm_circuit(
        self,
        code: CSSCode,
        noise_model: NoiseModel,
        stim_ids: StimIds,
        coloring_strategy: str = "largest_first",
    ) -> tuple[stim.Circuit, list[list[int]]]:
        """
        Compiles a coloration circuit. Not depth-optimal as no interleaving of opposite type checks is present. Z checks are performed first followed by X checks
        """
        z_subcircuit = self._code_to_subcircuit(
            code.code_z,
            noise_model,
            stim_ids.z_check_ids,
            stim_ids.data_ids,
            "CZ",
            coloring_strategy,
        )
        x_subcircuit = self._code_to_subcircuit(
            code.code_x,
            noise_model,
            stim_ids.x_check_ids,
            stim_ids.data_ids,
            "CX",
            coloring_strategy,
        )
        circuit = stim.Circuit()
        noise_model.apply_reset(circuit, Pauli.Z, stim_ids.z_check_ids + stim_ids.x_check_ids)
        noise_model.apply_1q_gates(circuit, "H", stim_ids.z_check_ids + stim_ids.x_check_ids)
        circuit.append(z_subcircuit)
        circuit.append(x_subcircuit)
        noise_model.apply_1q_gates(circuit, "H", stim_ids.z_check_ids + stim_ids.x_check_ids)
        noise_model.apply_meas(circuit, Pauli.Z, stim_ids.z_check_ids + stim_ids.x_check_ids)
        measurements = [stim_ids.z_check_ids + stim_ids.x_check_ids]
        return circuit, measurements
