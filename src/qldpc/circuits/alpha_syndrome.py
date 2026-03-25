"""Classes to define the AlphaSyndrome syndrome measurement strategies

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

import itertools
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import sinter
import stim

from qldpc import codes
from qldpc.objects import Node, Pauli, PauliXZ

from .bookkeeping import MeasurementRecord, QubitIDs
from .common import restrict_to_qubits, with_remapped_qubits
from .noise_model import NoiseModel
from .syndrome_measurement import SyndromeMeasurementStrategy


class AlphaSyndrome(SyndromeMeasurementStrategy):
    """AlphaSyndrome strategy for constructing a syndrome measurement circuit.

    Uses Monte-Carlo tree search (MCTS) to suppress hook errors.  Currently only supports CSS codes.

    For more information, see the paper at https://www.arxiv.org/abs/2601.12509.

    WARNING: This strategy is extremely SLOW due to unsolved problem with multiprocessing and MCTS.
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        decoder: str,
        custom_decoders: dict[str, sinter.Decoder] | None = None,
        iters_per_step: int = 8000,
        shots_per_iter: int = 10000,
    ) -> None:
        """Initialize an EdgeColoringXZ syndrome measurement strategy.

        Args:
            noise_model: The noise model append to the syndrome measurement circuit
            decoder: The decoder that Sinter should use to compute logical error rates.
            custom_decoder: Custom decoders to pass Sinter, if applicable
            iters_per_step: iterations per MCTS step, default is 8000
            shots_per_iter: number of sampling shots per iteration, default is 10000
        """
        self.noise_model = noise_model
        self.decoder = decoder
        self.custom_decoders = custom_decoders
        self.iters_per_step = iters_per_step
        self.shots_per_iter = shots_per_iter

    @restrict_to_qubits
    def get_circuit(
        self, code: codes.QuditCode, qubit_ids: QubitIDs | None = None
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
        if not isinstance(code, codes.CSSCode):
            raise ValueError(
                "The AlphaSyndrome strategy for syndrome measurement only supports CSS codes"
            )
        qubit_ids = qubit_ids or QubitIDs.from_code(code)
        x_ticks = self._get_schedule_for_basis(code, Pauli.X)
        z_ticks = self._get_schedule_for_basis(code, Pauli.Z)

        circuit = stim.Circuit()
        circuit.append("RX", range(len(code), len(code) + code.num_checks))
        circuit += self._get_scheduled_circuit_for_basis(code, Pauli.X, x_ticks)
        circuit += self._get_scheduled_circuit_for_basis(code, Pauli.Z, z_ticks)
        circuit.append("MX", range(len(code), len(code) + code.num_checks))

        record = MeasurementRecord({qubit: [mm] for mm, qubit in enumerate(qubit_ids.check)})
        return qubit_ids.with_remapped_qubits(circuit), record

    @staticmethod
    def _get_scheduled_circuit_for_basis(
        code: codes.CSSCode, basis: PauliXZ, schedule: Sequence[int]
    ) -> stim.Circuit:
        checks_of_basis = _get_checks(code, basis)
        zipped_schedule = zip(checks_of_basis, schedule)
        sorted_schedule = sorted(zipped_schedule, key=lambda x: x[1])

        circuit = stim.Circuit()
        for _, checks in itertools.groupby(sorted_schedule, key=lambda ct: ct[1]):
            for (data, ancilla), _ in checks:
                circuit.append(f"C{basis}", [ancilla, data])
                circuit.append("TICK", [])

        return circuit

    def _get_schedule_for_basis(self, code: codes.CSSCode, basis: PauliXZ) -> list[int]:
        checks = _get_checks(code, basis)
        node = TreeNode(TreeState.initial_state(len(checks), code.num_qubits + code.num_checks))
        while not node.is_terminal():
            node = self._schedule_step(code, basis, node, checks)
        return node.state.schedule

    def _schedule_step(
        self, code: codes.CSSCode, basis: PauliXZ, root: TreeNode, checks: Sequence[tuple[int, int]]
    ) -> TreeNode:
        iterations = max(0, self.iters_per_step - root.visits)
        for _ in range(iterations):
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()
            if not node.is_terminal():
                node = node.expand(checks)

            schedule = node.simulate_schedule(checks)
            circuit = self._get_evaluation_circuit(code, basis, schedule)
            noisy_circuit = self.noise_model.noisy_circuit(
                circuit, immune_qubits=range(code.num_qubits), insert_ticks=False
            )

            sampler = noisy_circuit.compile_detector_sampler()
            dets, observable_flips = sampler.sample(self.shots_per_iter, separate_observables=True)
            dem = noisy_circuit.detector_error_model(
                decompose_errors=True, ignore_decomposition_failures=True
            )

            predictions = sinter.predict_observables(
                dem=dem, dets=dets, decoder=self.decoder, custom_decoders=self.custom_decoders
            )
            result = np.sum(np.any(predictions != observable_flips, axis=1))
            node.backpropagate(self.shots_per_iter / (result + 1))

        return root.best_child(exploration_weight=0)

    def _get_evaluation_circuit(
        self, code: codes.CSSCode, basis: PauliXZ, schedule: list[int]
    ) -> stim.Circuit:

        # stabilizers and logical operators in the opposite basis
        opposite_basis = Pauli.swap_xz(basis)
        stabilizers = code.get_stabilizer_ops(opposite_basis, symplectic=True)
        logical_ops = code.get_logical_ops(opposite_basis, symplectic=True)
        opposite_basis_ops = np.vstack([stabilizers, logical_ops])
        opposite_basis_measurements = _get_pauli_product_measurements(opposite_basis_ops)

        circuit = stim.Circuit()
        circuit += opposite_basis_measurements
        circuit += self._get_scheduled_circuit_for_basis(code, basis, schedule)
        circuit += opposite_basis_measurements

        num_stabilizers = code.get_num_checks(opposite_basis)
        num_observables = code.dimension
        num_measurements = num_stabilizers + num_observables
        for ii in range(num_stabilizers):
            meas_index = -num_measurements + ii
            circuit.append(
                "DETECTOR",
                [stim.target_rec(meas_index), stim.target_rec(meas_index - num_measurements)],
                ii,
            )
        for ii in range(num_observables):
            meas_index = -num_measurements + num_stabilizers + ii
            circuit.append(
                "OBSERVABLE_INCLUDE",
                [stim.target_rec(meas_index), stim.target_rec(meas_index - num_measurements)],
                ii,
            )

        return circuit


@dataclass(slots=True)
class TreeState:
    schedule: list[int]
    maxticks: list[int]

    @staticmethod
    def initial_state(num_checks: int, num_qubits: int) -> TreeState:
        return TreeState([-1] * num_checks, [-1] * num_qubits)

    def shift(self, checks: Sequence[tuple[int, int]], meas_index: int) -> TreeState:
        check = checks[meas_index]
        new_tick = max(self.maxticks[check[0]], self.maxticks[check[1]]) + 1

        new_schedule = self.schedule.copy()
        new_maxticks = self.maxticks.copy()

        new_maxticks[check[0]] = new_tick
        new_maxticks[check[1]] = new_tick
        new_schedule[meas_index] = new_tick

        return TreeState(new_schedule, new_maxticks)

    def transitions(self) -> list[int]:
        states = []
        for meas_index, tick in enumerate(self.schedule):
            if tick == -1:  # unmeasured syndrome measurement
                states.append(meas_index)
        return states

    def is_terminal(self) -> bool:
        return min(self.schedule) != -1


class TreeNode:
    def __init__(self, state: TreeState, parent: TreeNode | None = None):
        self.state = state
        self.parent = parent

        self.children: list[TreeNode] = []

        self.visits = 0
        self.value = 0.0

        self.unvisited = state.transitions()

    def is_fully_expanded(self) -> bool:
        return len(self.unvisited) == 0

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def expand(self, checks: Sequence[tuple[int, int]]) -> TreeNode:
        next_state = self.state.shift(checks, self.unvisited.pop())
        child_node = TreeNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def best_child(self, exploration_weight: float = 1.4) -> TreeNode:
        def ucb_score(child: TreeNode) -> float:
            if child.visits == 0:
                return float("inf")  # pragma: no cover
            return child.value / child.visits + exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )

        return max(self.children, key=ucb_score)

    def backpropagate(self, result: float) -> None:
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(result)

    def simulate_schedule(self, checks: Sequence[tuple[int, int]]) -> list[int]:
        current_state = self.state
        while not current_state.is_terminal():
            current_state = current_state.shift(checks, random.choice(current_state.transitions()))
        return current_state.schedule


def _get_checks(code: codes.CSSCode, basis: PauliXZ) -> Sequence[tuple[int, int]]:
    graph = code.get_graph(basis)
    return [(data.index, check.index + len(code)) for data, check in map(sorted, graph.edges)]


def _get_pauli_product_measurements(op_matrix: npt.NDArray[np.int_]) -> stim.Circuit:
    op_graph = codes.QuditCode.matrix_to_graph(op_matrix)

    circuit = stim.Circuit()
    for node_index in range(len(op_matrix)):
        observable_node = Node(node_index, is_data=False)
        targets = [
            stim.target_pauli(data_node.index, str(edge_data[Pauli]))
            for _, data_node, edge_data in op_graph.edges(observable_node, data=True)
        ]
        circuit.append("MPP", stim.target_combined_paulis(targets))
        circuit.append("TICK")

    return circuit
