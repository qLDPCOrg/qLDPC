# SPDX-License-Identifier: Apache-2.0

"""Unit tests for alpha_syndrome.py."""

from __future__ import annotations

import random
from unittest import mock

import numpy as np
import pytest
import sinter
import stim

from qldpc import circuits, codes, decoders, math
from qldpc.circuits.memory import alpha_syndrome
from qldpc.circuits.memory.alpha_syndrome import _TreeNode, _TreeState
from qldpc.objects import Pauli

# default (fast, low-fidelity) strategy used by the validity-checking helper below
DEFAULT_STRATEGY = circuits.AlphaSyndrome(
    circuits.DepolarizingNoiseModel(0.001),
    iters_per_step=3,
    shots_per_iter=1,
    verbose=False,
)


def test_alpha_syndrome(pytestconfig: pytest.Config) -> None:
    """Verify that syndromes are read out correctly."""
    seed = pytestconfig.getoption("randomly_seed")

    # verify that AlphaSyndrome builds valid syndrome extraction circuits for a few codes
    assert alpha_syndrome_is_valid(codes.SteaneCode())
    assert alpha_syndrome_is_valid(codes.ToricCode(2, rotated=True))
    assert alpha_syndrome_is_valid(codes.SurfaceCode(2, rotated=True))

    code_a = codes.ClassicalCode.random(3, 2, seed=seed)
    code_b = codes.ClassicalCode.random(3, 2, seed=seed + 1)
    assert alpha_syndrome_is_valid(codes.HGPCode(code_a, code_b))

    # AlphaSyndrome does not support non-CSS codes
    with pytest.raises(TypeError, match="only supports CSS codes"):
        strategy = circuits.AlphaSyndrome(circuits.DepolarizingNoiseModel(0.001), "decoder_name")
        strategy.get_circuit(codes.FiveQubitCode())
    with pytest.raises(ValueError, match="invalid"):
        DEFAULT_STRATEGY.get_circuit(
            codes.SteaneCode(),
            circuits.QubitIDs(
                range(len(codes.SteaneCode())),
                range(20, 20 + codes.SteaneCode().num_checks - 1),
            ),
        )


def test_alpha_syndrome_qldpc_decoder() -> None:
    """AlphaSyndrome accepts qLDPC decoder inputs, such as decoder builders."""
    code = codes.CSSCode([[1, 1, 0]], [[1, 1, 0]])
    dems: list[stim.DetectorErrorModel] = []

    def build_decoder(dem: stim.DetectorErrorModel) -> decoders.ErrorDecoder:
        dems.append(dem)
        return decoders.bp_osd().build(dem)

    strategy = circuits.AlphaSyndrome(
        circuits.DepolarizingNoiseModel(0.01),
        build_decoder,
        iters_per_step=2,
        shots_per_iter=1,
        verbose=False,
        seed=0,
    )
    assert alpha_syndrome_is_valid(code, strategy)
    assert dems

    # a decoder with the decode_via_files method that Sinter calls is passed to Sinter as is
    class FileDecoder:
        called = False

        def decode_via_files(self, **kwargs: object) -> None:
            self.called = True
            sinter.BUILT_IN_DECODERS["pymatching"].decode_via_files(**kwargs)

    file_decoder = FileDecoder()
    strategy = circuits.AlphaSyndrome(
        circuits.DepolarizingNoiseModel(0.01),
        file_decoder,
        iters_per_step=2,
        shots_per_iter=1,
        verbose=False,
        seed=0,
    )
    assert alpha_syndrome_is_valid(code, strategy)
    assert file_decoder.called


def test_alpha_reproducibility_and_tree_edges() -> None:
    """Seeded searches are reproducible and cover the empty-search tree edges."""
    with pytest.warns(UserWarning, match="no fresh rollout"):
        circuits.AlphaSyndrome(circuits.DepolarizingNoiseModel(0.001), iters_per_step=1)
    with pytest.raises(ValueError, match="iters_per_step"):
        circuits.AlphaSyndrome(circuits.DepolarizingNoiseModel(0.001), iters_per_step=0)
    code = codes.CSSCode([[1, 1, 0]], [[1, 1, 0]])
    first = circuits.AlphaSyndrome(
        circuits.DepolarizingNoiseModel(0.001),
        iters_per_step=2,
        shots_per_iter=1,
        verbose=False,
        seed=123,
    )
    second = circuits.AlphaSyndrome(
        circuits.DepolarizingNoiseModel(0.001),
        iters_per_step=2,
        shots_per_iter=1,
        verbose=False,
        seed=123,
    )
    assert first.get_circuit(code)[0] == second.get_circuit(code)[0]
    verbose = circuits.AlphaSyndrome(
        circuits.DepolarizingNoiseModel(0.001), iters_per_step=2, verbose=True
    )
    verbose._build_schedule(code, Pauli.X)

    code = codes.SteaneCode()
    strategy = circuits.AlphaSyndrome(
        circuits.DepolarizingNoiseModel(0.001), iters_per_step=2, verbose=False
    )
    assert _TreeState.head([]).target_to_min_time == []
    gates = [(len(code), 0)]
    root = _TreeNode(_TreeState.head(gates))
    root.visits = strategy.iters_per_step
    measurements = strategy._get_evaluation_circuit(code, Pauli.X, [], None)
    assert measurements.num_measurements > 0
    strategy._schedule_one_gate(
        code,
        Pauli.X,
        root,
        step=0,
        evaluation_data=(stim.Circuit(), 0, 0),
    )
    child = _TreeNode(_TreeState.head(gates))
    child.expand()
    assert child.best_child(1).value == 0


def test_alpha_evaluation_data_and_empty_basis() -> None:
    """Precomputed evaluation data gives the same circuit, and an empty basis has no schedule."""
    code = codes.CSSCode([[1, 1, 0]], [[1, 1, 0]])
    strategy = circuits.AlphaSyndrome(
        circuits.DepolarizingNoiseModel(0.001),
        iters_per_step=2,
        shots_per_iter=1,
        verbose=False,
        seed=7,
    )
    schedule = strategy._build_schedule(code, Pauli.X)
    data = strategy._get_evaluation_data(code, Pauli.X)
    assert strategy._get_evaluation_circuit(
        code, Pauli.X, schedule, data
    ) == strategy._get_evaluation_circuit(code, Pauli.X, schedule)

    empty_x_code = codes.CSSCode(np.zeros((0, 2), dtype=int), [[1, 1]])
    assert strategy._build_schedule(empty_x_code, Pauli.X) == []


def test_alpha_reward_matches_paper() -> None:
    """The MCTS reward is the finite-sample inverse logical error rate of the AlphaSyndrome paper.

    See Section 4.4 of https://arxiv.org/abs/2601.12509.
    """
    code = codes.CSSCode([[1, 1, 0]], [[1, 1, 0]])
    shots = 20
    strategy = circuits.AlphaSyndrome(
        circuits.NoiseModel(),
        iters_per_step=3,
        shots_per_iter=shots,
        verbose=False,
        seed=11,
    )
    rewards: list[float] = []
    original_backpropagate = _TreeNode.backpropagate
    graph = code.get_graph(Pauli.X)
    gates = [(check.index + len(code), data.index) for data, check in map(sorted, graph.edges)]
    root = _TreeNode(_TreeState.head(gates))
    evaluation_data = strategy._get_evaluation_data(code, Pauli.X)

    def record_reward(node: _TreeNode, reward: float) -> None:
        rewards.append(reward)
        original_backpropagate(node, reward)

    # without noise, no observable flips, so every flipped prediction is a logical error
    num_observables = evaluation_data[2]
    predictions = []
    for num_errors in [0, 10, 0]:
        prediction = np.zeros((shots, num_observables), dtype=bool)
        prediction[:num_errors, 0] = True
        predictions.append(prediction)

    with (
        mock.patch.object(_TreeNode, "backpropagate", autospec=True, side_effect=record_reward),
        mock.patch.object(alpha_syndrome.sinter, "predict_observables", side_effect=predictions),
    ):
        strategy._schedule_one_gate(
            code,
            Pauli.X,
            root,
            step=0,
            evaluation_data=evaluation_data,
        )
    assert set(rewards) == {20.0, 20 / 11}


def alpha_syndrome_is_valid(
    code: codes.QuditCode,
    strategy: circuits.AlphaSyndrome = DEFAULT_STRATEGY,
) -> bool:
    """Check that an AlphaSyndrome circuit correctly reads out stabilizers."""
    # prepare a logical |0> state
    state_prep = circuits.get_encoding_circuit(code)

    # apply random Pauli errors to the data qubits
    errors = random.choices([Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], k=len(code))
    error_ops = stim.Circuit()
    for qubit, pauli in enumerate(errors):
        error_ops.append(f"{pauli}_error", [qubit], [1])

    # measure syndromes
    syndrome_extraction_circuit, measurement_record = strategy.get_circuit(code)
    for check in range(len(code), len(code) + code.num_checks):
        syndrome_extraction_circuit.append("DETECTOR", measurement_record.get_target_rec(check))

    # sample the circuit to obtain a syndrome vector
    circuit = state_prep + error_ops + syndrome_extraction_circuit
    syndrome = circuit.compile_detector_sampler().sample(1).ravel()

    # compare against the expected syndrome
    error_xz = code.field([pauli.value for pauli in errors]).T.ravel()
    expected_syndrome = code.matrix @ math.symplectic_conjugate(error_xz)
    return np.array_equal(expected_syndrome, syndrome)
