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

import collections
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

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

T = TypeVar("T")


GateSchedule = list[list[tuple[int, int]]]


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
        exploration_weight: float = 1.4,
    ) -> None:
        """Initialize an EdgeColoringXZ syndrome measurement strategy.

        Args:
            noise_model: The noise model append to the syndrome measurement circuit.
            decoder: The decoder that Sinter should use to compute logical error rates.
            custom_decoder: Custom decoders to pass Sinter, if applicable.
            iters_per_step: iterations per MCTS step (default: 8000).
            shots_per_iter: number of sampling shots per iteration (default: 10000).
            exploration_weight: exploration parameter of MCTS (default: 1.4).
        """
        self.noise_model = noise_model
        self.decoder = decoder
        self.custom_decoders = custom_decoders
        self.iters_per_step = iters_per_step
        self.shots_per_iter = shots_per_iter
        self.exploration_weight = exploration_weight

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

        # the heavy lifting: schedule gates
        schedule_cx = self._get_schedule(code, Pauli.X)
        schedule_cz = self._get_schedule(code, Pauli.Z)

        # construct a circuit from the gate schedules
        circuit = stim.Circuit()
        circuit.append("RX", range(len(code), len(code) + code.num_checks))
        circuit += self._get_circuit_from_schedule(schedule_cx, Pauli.X)
        circuit += self._get_circuit_from_schedule(schedule_cz, Pauli.Z)
        circuit.append("MX", range(len(code), len(code) + code.num_checks))

        # remap qubits and return the circuit together with a measurement record
        circuit = with_remapped_qubits(circuit, qubit_ids.data + qubit_ids.check)
        record = MeasurementRecord({qubit: [mm] for mm, qubit in enumerate(qubit_ids.check)})
        return circuit, record

    @staticmethod
    def _get_circuit_from_schedule(
        schedule: Sequence[Sequence[tuple[int, int]]], basis: PauliXZ
    ) -> stim.Circuit:
        circuit = stim.Circuit("TICK")
        for gates in schedule:
            for gate in gates:
                circuit.append(f"C{basis}", gate)
            circuit.append("TICK")
        return circuit

    def _get_schedule(self, code: codes.CSSCode, basis: PauliXZ) -> GateSchedule:
        # identify gates that need to be cheduled
        graph = code.get_graph(basis)
        gates = [(check.index + len(code), data.index) for data, check in map(sorted, graph.edges)]

        # schedule gates with MCTS
        node = TreeNode(TreeState.head(len(gates), code.num_qubits + code.num_checks))
        while not node.is_terminal:
            node = self._schedule_step(code, basis, node, gates)
        return node.rollout(gates)

    def _schedule_step(
        self, code: codes.CSSCode, basis: PauliXZ, root: TreeNode, gates: Sequence[tuple[int, int]]
    ) -> TreeNode:
        iterations = max(0, self.iters_per_step - root.visits)
        for _ in range(iterations):
            node = root
            while not node.is_terminal and node.is_fully_expanded:
                node = node.best_child(self.exploration_weight)
            if not node.is_terminal:
                node = node.expand(gates)

            scheduled_gates = node.rollout(gates)
            circuit = self._get_evaluation_circuit(code, basis, scheduled_gates)
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
        self, code: codes.CSSCode, basis: PauliXZ, gates: Sequence[Sequence[tuple[int, int]]]
    ) -> stim.Circuit:

        # stabilizers and logical operators in the opposite basis
        opposite_basis = Pauli.swap_xz(basis)
        stabilizers = code.get_stabilizer_ops(opposite_basis, symplectic=True)
        logical_ops = code.get_logical_ops(opposite_basis, symplectic=True)
        opposite_basis_ops = np.vstack([stabilizers, logical_ops])
        opposite_basis_measurements = _get_pauli_product_measurements(opposite_basis_ops)

        circuit = stim.Circuit()
        circuit += opposite_basis_measurements
        circuit += self._get_circuit_from_schedule(gates, basis)
        circuit += opposite_basis_measurements

        num_stabilizers = len(stabilizers)
        num_observables = len(logical_ops)
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


class TreeNode:
    """Node of a tree for Monte Carlo tree search (MCTS)."""

    def __init__(self, state: TreeState, parent: TreeNode | None = None):
        self.state = state
        self.parent = parent

        self.children: list[TreeNode] = []
        self.unvisited = state.transitions()

        self.visits = 0
        self.value = 0.0

    @property
    def is_terminal(self) -> bool:
        """Is this a terminal node of the tree, which specifies a complete schedule?"""
        return self.state.is_terminal

    @property
    def is_fully_expanded(self) -> bool:
        """Have we constructed all children of this node?"""
        return len(self.unvisited) == 0

    def expand(self, gates: Sequence[tuple[int, int]]) -> TreeNode:
        """Construct a child of this node."""
        child_state = self.state.select(gates, self.unvisited.pop())
        child_node = TreeNode(child_state, self)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, reward: float) -> None:
        """Increase the value of this node and all of its parents."""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def rollout(self, gates: Sequence[tuple[int, int]]) -> GateSchedule:
        """Schedule any unscheduled gates at random, and return a complete gate schedule."""
        # select transitions at random to assign each unscheduled gate a time index
        current_state = self.state
        while not current_state.is_terminal:
            current_state = current_state.select(gates, random.choice(current_state.transitions()))
        gate_to_time = current_state.gate_to_time

        # collect gates according to their time index
        time_to_gates: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
        for gate, time in zip(gates, gate_to_time):
            time_to_gates[time].append(gate)

        # return a schedule of gates: a list whose t-th index is a list of gates to apply at time t
        return [time_to_gates[time] for time in sorted(time_to_gates.keys())]

    def best_child(self, exploration_weight: float) -> TreeNode:
        def ucb_score(child: TreeNode) -> float:
            if child.visits == 0:
                return float("inf")  # pragma: no cover
            return child.value / child.visits + exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )

        return max(self.children, key=ucb_score)


@dataclass(slots=True)
class TreeState:
    gate_to_time: list[int]  # time index for each gate.  -1 for unscheduled gates
    min_time_for_qubit: list[int]  # minimum time index for a new gate on a qubit

    @staticmethod
    def head(num_gates: int, num_qubits: int) -> TreeState:
        """The head node for a scheduling tree."""
        return TreeState([-1] * num_gates, [0] * num_qubits)

    @property
    def is_terminal(self) -> bool:
        """Have all gates been scheduled?"""
        return -1 not in self.gate_to_time

    def transitions(self) -> list[int]:
        """List of gates (by index) that still need to be scheduled."""
        return [
            gate_index
            for gate_index, time_index in enumerate(self.gate_to_time)
            if time_index == -1
        ]

    def select(self, gates: Sequence[tuple[int, int]], gate_index: int) -> TreeState:
        """Append the given gate (by index)."""
        pp, qq = gates[gate_index]  # gate targets
        time_index = max(self.min_time_for_qubit[pp], self.min_time_for_qubit[qq])

        gate_to_time = self.gate_to_time.copy()
        gate_to_time[gate_index] = time_index

        min_time_for_qubit = self.min_time_for_qubit.copy()
        min_time_for_qubit[pp] = time_index
        min_time_for_qubit[qq] = time_index

        return TreeState(gate_to_time, min_time_for_qubit)


def _group_items_by_sorted_values(items: Sequence[T], values: Sequence[int]) -> list[list[T]]:
    """Group items by associated values, and return groups of items in order of increasing value."""
    assert len(items) == len(values)
    value_to_items: dict[int, list[T]] = collections.defaultdict(list)
    for item, value in zip(items, values):
        value_to_items[value].append(item)
    return [value_to_items[value] for value in sorted(value_to_items.keys())]


def _get_pauli_product_measurements(op_matrix: npt.NDArray[np.int_]) -> stim.Circuit:
    """Construct a circuit that measures Pauli strings represented by the rows of a matrix.

    For example, passing the parity check matrix will measure stabilizers.
    """
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
