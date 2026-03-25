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
from .noise_model import NoiseModel, as_noiseless_circuit
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
        schedule_cx = self._build_schedule(code, Pauli.X)
        schedule_cz = self._build_schedule(code, Pauli.Z)

        # construct a circuit from the gate schedules
        circuit = stim.Circuit()
        circuit.append("RX", range(len(code), len(code) + code.num_checks))
        circuit += _schedule_to_circuit(schedule_cx, Pauli.X)
        circuit += _schedule_to_circuit(schedule_cz, Pauli.Z)
        circuit.append("MX", range(len(code), len(code) + code.num_checks))

        # remap qubits and return the circuit together with a measurement record
        circuit = with_remapped_qubits(circuit, qubit_ids.data + qubit_ids.check)
        record = MeasurementRecord({qubit: [mm] for mm, qubit in enumerate(qubit_ids.check)})
        return circuit, record

    def _build_schedule(self, code: codes.CSSCode, basis: PauliXZ) -> GateSchedule:
        """Schedule the gates that extract basis-type stabilizers of a CSS code."""
        # identify gates that need to be scheduled, as (control, target) pairs
        graph = code.get_graph(basis)
        gates = [(check.index + len(code), data.index) for data, check in map(sorted, graph.edges)]

        # schedule gates with MCTS
        node = TreeNode(TreeState.head(gates))
        while not node.is_terminal():
            node = self._schedule_step(code, basis, node)

        # convert the final tree node into a gate schedule
        return node.state.to_schedule()

    def _schedule_step(self, code: codes.CSSCode, basis: PauliXZ, root: TreeNode) -> TreeNode:
        iterations = max(0, self.iters_per_step - root.visits)
        for _ in range(iterations):
            node = root
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(self.exploration_weight)
            if not node.is_terminal():
                node = node.expand()

            scheduled_gates = node.rollout().to_schedule()
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
        self, code: codes.CSSCode, basis: PauliXZ, schedule: GateSchedule
    ) -> stim.Circuit:
        """Build the circuit used to evaluate a gate schedule.

        Assume without loss of generality that basis is Pauli.X.  The evaluation circuit penalizes
        Z-type logical operator flips when reading out X-type stabilizers.
        """

        # noiseless measurement of stabilizers and logical operators in the opposite basis
        opposite_basis = Pauli.swap_xz(basis)
        stabilizers = code.get_stabilizer_ops(opposite_basis, symplectic=True)
        logical_ops = code.get_logical_ops(opposite_basis, symplectic=True)
        opposite_basis_ops = np.vstack([stabilizers, logical_ops])
        opposite_basis_measurements = as_noiseless_circuit(
            _get_pauli_product_measurements(opposite_basis_ops)
        )
        num_stabilizers = len(stabilizers)
        num_observables = len(logical_ops)
        num_measurements = num_stabilizers + num_observables

        # if reading out (say) X-type stabilizers, detect Z-type stabilizer and observable flips
        circuit = stim.Circuit()
        circuit += opposite_basis_measurements
        circuit += _schedule_to_circuit(schedule, basis)
        circuit += opposite_basis_measurements
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
    """Node of a tree for Monte Carlo tree search (MCTS).

    TreeNode is agnostic to the problem being solved by MCTS.
    All problem data is handled by the TreeState.

    References:
    - https://en.wikipedia.org/wiki/Monte_Carlo_tree_search
    """

    def __init__(self, state: TreeState, parent: TreeNode | None = None):
        self.state = state
        self.parent = parent

        self.children: list[TreeNode] = []
        self.unvisited = state.transitions()

        self.visits = 0
        self.value = 0.0

    def is_terminal(self) -> bool:
        """Is this a terminal node of the tree, which specifies a complete schedule?"""
        return self.state.is_terminal()

    def is_fully_expanded(self) -> bool:
        """Have we constructed all children of this node?"""
        return len(self.unvisited) == 0

    def expand(self) -> TreeNode:
        """Construct a child of this node."""
        child_state = self.state.select(self.unvisited.pop())
        child_node = TreeNode(child_state, self)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, reward: float) -> None:
        """Increase the value of this node and all of its parents."""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.backpropagate(reward)

    def rollout(self) -> TreeState:
        """Select transitions at random until we reach a terminal node, and return its state."""
        state = self.state
        while not state.is_terminal():
            state = state.select(random.choice(state.transitions()))
        return state

    def best_child(self, exploration_weight: float) -> TreeNode:
        """Select a child of this node with the highest UCB score."""
        assert self.children

        def ucb_score(child: TreeNode) -> float:
            if child.visits == 0:
                return float("inf")  # pragma: no cover
            return child.value / child.visits + exploration_weight * math.sqrt(
                math.log(self.visits) / child.visits
            )

        return max(self.children, key=ucb_score)


@dataclass(slots=True)
class TreeState:
    """The state of an MCTS tree, representing a (possibly incomplete) gate schedule."""

    gates: list[tuple[int, int]]
    gate_to_time: list[int]  # time index for each gate.  -1 for unscheduled gates
    target_to_min_time: list[int]  # minimum time index for a new gate on a target

    @staticmethod
    def head(gates: Sequence[tuple[int, int]]) -> TreeState:
        """A TreeState in which no gates have been scheduled."""
        num_gates = len(gates)
        num_targets = max(target for gate in gates for target in gate) + 1
        return TreeState(list(gates), [-1] * num_gates, [0] * num_targets)

    def is_terminal(self) -> bool:
        """Have all gates been scheduled?"""
        return -1 not in self.gate_to_time

    def transitions(self) -> list[int]:
        """The indices of gates that still need to be scheduled."""
        return [
            gate_index
            for gate_index, time_index in enumerate(self.gate_to_time)
            if time_index == -1
        ]

    def select(self, gate_index: int) -> TreeState:
        """Append the gate at the given index to the gate schedule."""
        target_a, target_b = self.gates[gate_index]
        time_index = max(self.target_to_min_time[target_a], self.target_to_min_time[target_b])

        gate_to_time = self.gate_to_time.copy()
        gate_to_time[gate_index] = time_index

        min_time_for_target = self.target_to_min_time.copy()
        min_time_for_target[target_a] = time_index + 1
        min_time_for_target[target_b] = time_index + 1

        return TreeState(self.gates, gate_to_time, min_time_for_target)

    def to_schedule(self) -> GateSchedule:
        """Convert this TreeState into a gate schedule.

        The schedule is provided as a list whose t-th entry is a list of gates to apply at time t.
        Unscheduled gates are excluded from the returned schedule.
        """
        # collect gates according to their time index
        time_to_gates: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
        for gate, time in zip(self.gates, self.gate_to_time):
            time_to_gates[time].append(gate)

        # return a schedule of gates: a list whose t-th index is a list of gates to apply at time t
        return [time_to_gates[time] for time in sorted(time_to_gates.keys()) if time != -1]


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


def _schedule_to_circuit(schedule: GateSchedule, target_pauli: Pauli) -> stim.Circuit:
    """Convert a schedule of controlled-Pauli gates into a circuit."""
    circuit = stim.Circuit("TICK")
    for gates in schedule:
        for gate in gates:
            circuit.append(f"C{target_pauli}", gate)
        circuit.append("TICK")
    return circuit
