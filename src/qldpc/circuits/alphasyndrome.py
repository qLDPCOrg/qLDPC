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
from dataclasses import dataclass
from typing import Any

import numpy as np
import sinter
import stim

from qldpc import codes
from qldpc.circuits.syndrome_measurement import SyndromeMeasurementStrategy
from qldpc.objects import Node, Pauli, PauliXZ

from .bookkeeping import MeasurementRecord, QubitIDs
from .common import restrict_to_qubits
from .noise_model import NoiseModel


@dataclass(slots=True)
class TreeState:
    schedule: np.ndarray
    maxticks: np.ndarray

    @staticmethod
    def initial_state(nchecks: int, nqubits: int) -> "TreeState":
        return TreeState(np.repeat(-1, nchecks), np.repeat(-1, nqubits))

    def shift(self, checks: list[tuple[int, int]], meas_index: int) -> "TreeState":
        chk = checks[meas_index]
        new_tick = max(self.maxticks[chk[0]], self.maxticks[chk[1]]) + 1

        new_schedule = self.schedule.copy()
        new_maxticks = self.maxticks.copy()

        new_maxticks[chk[0]] = new_tick
        new_maxticks[chk[1]] = new_tick
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
    def __init__(self, state: TreeState, parent: "TreeNode | None" = None):
        self.state = state

        self.parent = parent
        self.children: list["TreeNode"] = []

        self.visits = 0
        self.value = 0.0

        self.unvisited = state.transitions()

    def is_fully_expanded(self) -> bool:
        return len(self.unvisited) == 0

    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def expand(self, checks: list[tuple[int, int]]) -> "TreeNode":
        next_state = self.state.shift(checks, self.unvisited.pop())
        child_node = TreeNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def best_child(self, exploration_weight: float = 1.4) -> "TreeNode":
        def ucb_score(child: "TreeNode") -> float:
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

    def simulate_schedule(self, checks: list[tuple[int, int]]) -> np.ndarray:
        current_state = self.state
        while not current_state.is_terminal():
            current_state = current_state.shift(checks, random.choice(current_state.transitions()))
        return current_state.schedule


def measure_as_product(circuit: stim.Circuit, pauli_targets: list[stim.GateTarget]) -> None:
    combined_targets = []
    for i, target in enumerate(pauli_targets):
        combined_targets.append(target)
        # Add a combiner between every target, but not after the last one
        if i < len(pauli_targets) - 1:
            combined_targets.append(stim.target_combiner())

    circuit.append_operation("MPP", combined_targets)


class WrapCSS:
    def __init__(self, code: codes.CSSCode, subgraph_kwargs: Any) -> None:
        syndrome_graphs = code.get_syndrome_subgraphs(**subgraph_kwargs)

        self.code = code

        self.x_checks: list[tuple[int, int]] = []
        self.z_checks: list[tuple[int, int]] = []

        for subgraph in syndrome_graphs:
            for edge in subgraph.edges:
                data_node, check_node = sorted(edge)
                pauli: Pauli = subgraph[check_node][data_node][Pauli]
                if pauli == Pauli.X:
                    self.x_checks.append((data_node.index, check_node.index + self.num_qubits))
                elif pauli == Pauli.Z:
                    self.z_checks.append((data_node.index, check_node.index + self.num_qubits))
                else:
                    assert False, "Unknown Pauli check for CSS code"  # pragma: no cover

        self.all_checks = self.x_checks + self.z_checks

    def checks(self, basis: Pauli) -> list[tuple[int, int]]:
        if basis == Pauli.X:
            return self.x_checks
        elif basis == Pauli.Z:
            return self.z_checks
        else:
            assert False, "Unknown Pauli check for CSS code"  # pragma: no cover

    @property
    def num_qubits(self) -> int:
        return self.code.num_qubits

    @property
    def num_ancillas(self) -> int:
        return self.code.num_checks

    def _measure_observable(self, circuit: stim.Circuit, basis: PauliXZ) -> int:
        num_observables = self.code.dimension
        logical_ops = self.code.get_logical_ops(basis, symplectic=True)
        logical_op_graph = codes.QuditCode.matrix_to_graph(logical_ops)

        for node_index in range(num_observables):
            observable_node = Node(node_index, is_data=False)
            targets = [
                stim.target_pauli(data_node.index, str(edge_data[Pauli]))
                for _, data_node, edge_data in logical_op_graph.edges(observable_node, data=True)
            ]
            measure_as_product(circuit, targets)
            circuit.append("TICK", [])

        return self.code.dimension

    def _measure_stabilizers(self, circuit: stim.Circuit, basis: PauliXZ) -> int:
        num_stabilizers = self.code.num_checks_x if basis == Pauli.X else self.code.num_checks_z
        stabilizer_ops = self.code.get_stabilizer_ops(basis, symplectic=True)
        stabilizer_op_graph = codes.QuditCode.matrix_to_graph(stabilizer_ops)

        for node_index in range(num_stabilizers):
            stabilizer_node = Node(node_index, is_data=False)
            targets = [
                stim.target_pauli(data_node.index, str(edge_data[Pauli]))
                for _, data_node, edge_data in stabilizer_op_graph.edges(stabilizer_node, data=True)
            ]
            measure_as_product(circuit, targets)
            circuit.append("TICK", [])

        return num_stabilizers

    def _ideal_measurement(self, circuit: stim.Circuit, basis: PauliXZ) -> tuple[int, int]:
        num_stabilizers = self._measure_stabilizers(circuit, basis)
        num_observables = self._measure_observable(circuit, basis)

        return num_stabilizers, num_observables

    def _syndrome_measurement(
        self,
        circuit: stim.Circuit,
        basis: Pauli,
        schedules: np.ndarray,
        qubit_ids: QubitIDs | None = None,
    ) -> None:
        checks_of_basis = self.checks(basis)
        zipped_schedule = zip(checks_of_basis, schedules)
        sorted_schedule = sorted(zipped_schedule, key=lambda x: x[1])

        for _, checks in itertools.groupby(sorted_schedule, key=lambda ct: ct[1]):
            for chk, _ in checks:
                if qubit_ids:
                    data, ancilla = (
                        qubit_ids.data[chk[0]],
                        qubit_ids.check[chk[1] - self.num_qubits],
                    )
                else:
                    data, ancilla = chk

                circuit.append(f"C{basis}", [ancilla, data])
                circuit.append("TICK", [])

    def evaluation_circuit(self, basis: Pauli, schedule: np.ndarray) -> stim.Circuit:
        oppsite_basis = Pauli.swap_xz(basis)

        circuit = stim.Circuit()
        num_stabilizers, num_observables = self._ideal_measurement(circuit, oppsite_basis)

        self._syndrome_measurement(circuit, basis, schedule)

        self._ideal_measurement(circuit, oppsite_basis)

        for i in range(num_observables):
            index = i + 1

            circuit.append(
                "OBSERVABLE_INCLUDE",
                [
                    stim.target_rec(-index),
                    stim.target_rec(-(index + num_stabilizers + num_observables)),
                ],
                i,
            )

        for i in range(num_stabilizers):
            index = i + 1 + num_observables

            circuit.append(
                "DETECTOR",
                [
                    stim.target_rec(-index),
                    stim.target_rec(-(index + num_stabilizers + num_observables)),
                ],
                i,
            )

        return circuit

    def measurement_circuit(
        self, x_ticks: np.ndarray, z_ticks: np.ndarray, qubit_ids: QubitIDs
    ) -> tuple[stim.Circuit, MeasurementRecord]:
        circuit = stim.Circuit()
        circuit.append("RX", qubit_ids.check)

        self._syndrome_measurement(circuit, Pauli.X, x_ticks, qubit_ids)
        self._syndrome_measurement(circuit, Pauli.Z, z_ticks, qubit_ids)

        circuit.append("MX", qubit_ids.check)
        measurement_record = MeasurementRecord(
            {qubit: [mm] for mm, qubit in enumerate(qubit_ids.check)}
        )
        return circuit, measurement_record


class AlphaSyndrome(SyndromeMeasurementStrategy):
    """AlphaSyndrome strategy for constructing a syndrome measurement circuit.

    Uses Monte-Carlo tree search to suppress hook error. For more information, find paper at
    https://www.arxiv.org/abs/2601.12509. Right now, only scheduling for CSS codes is implemented.

    WARNING: This strategy is extremely SLOW due to unsolved problem with multiprocessing and MCTS
    """

    def __init__(
        self,
        noise_model: NoiseModel,
        decoder: str,
        custome_decoders: dict[str, sinter.Decoder] | None = None,
        iters_per_step: int = 8000,
        shots_per_iter: int = 10000,
        **subgraph_kwargs: Any,
    ) -> None:
        """Initialize an EdgeColoringXZ syndrome measurement strategy.

        Args:
            noise_model: The noise model append to the syndrome measurement circuit
            iters_per_step: iterations per MCTS step, default is 8000
            shots_per_iter: number of sampling shots per iteration, default is 10000
        """

        super().__init__()

        self.decoder = decoder
        self.custom_decoders = custome_decoders
        self.noise_model = noise_model
        self.iters_per_step = iters_per_step
        self.shots_per_iter = shots_per_iter

        self.subgraph_kwargs = subgraph_kwargs

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

        wrap_code = WrapCSS(code, self.subgraph_kwargs)

        x_ticks = self._schedule_check_basis(Pauli.X, wrap_code)
        z_ticks = self._schedule_check_basis(Pauli.Z, wrap_code)

        return wrap_code.measurement_circuit(x_ticks, z_ticks, qubit_ids)

    def _schedule_step(
        self, root: TreeNode, basis: Pauli, code: WrapCSS, checks: list[tuple[int, int]]
    ) -> TreeNode:
        iterations = max(0, self.iters_per_step - root.visits)
        for _ in range(iterations):
            node = root

            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child()

            if not node.is_terminal():
                node = node.expand(checks)

            schedule = node.simulate_schedule(checks)

            circuit = code.evaluation_circuit(basis, schedule)

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

    def _schedule_check_basis(self, basis: Pauli, code: WrapCSS) -> np.ndarray:
        checks = code.checks(basis)

        node = TreeNode(TreeState.initial_state(len(checks), code.num_qubits + code.num_ancillas))

        while not node.is_terminal():
            node = self._schedule_step(node, basis, code, checks)
        return node.state.schedule
