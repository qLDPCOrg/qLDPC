# SPDX-License-Identifier: Apache-2.0

"""Tests for exact minimum-weight bounds on Pauli error classes."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
import pytest
import stim

from ._bounded_weight_search import PauliCosetSearch

Metric = Literal["pauli", "css"]


def _graph_state(
    num_qubits: int, edges: Sequence[tuple[int, int]], phase: bool = False
) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] of a graph state, with S gates on every qubit if ``phase``.

    Generator j acts as X (or Y) on qubit j, so the X parts of the generators form the identity.
    """
    circuit = stim.Circuit()
    circuit.append("H", range(num_qubits))
    for edge in edges:
        circuit.append("CZ", edge)
    if phase:
        circuit.append("S", range(num_qubits))
    generators = stim.Tableau.from_circuit(circuit).to_stabilizers()
    return np.array([np.concatenate(gg.to_numpy()) for gg in generators], dtype=np.uint8)


def _class_weights(
    matrix: npt.NDArray[np.uint8], syndrome: int, metric: Metric
) -> npt.NDArray[np.int_]:
    """Weights of every error with this syndrome, for stabilizers whose X parts are the identity.

    Z on the set syndrome bits has the syndrome, and all other such errors are its products with the
    2**n stabilizers, so this enumeration is independent of the code under test.
    """
    n = len(matrix)
    assert np.array_equal(matrix[:, :n], np.eye(n))
    base = np.concatenate([np.zeros(n, dtype=int), (syndrome >> np.arange(n)) & 1])
    subsets = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    errors = (base + subsets @ matrix.astype(int)) % 2
    x, z = errors[:, :n], errors[:, n:]
    return (x | z).sum(axis=1) if metric == "pauli" else np.maximum(x.sum(axis=1), z.sum(axis=1))


def _pauli_weight(pauli: int, num_qubits: int, metric: Metric) -> int:
    """Weight of a packed Pauli, with X in the low bits and Z in the next block."""
    x, z = pauli & ((1 << num_qubits) - 1), pauli >> num_qubits
    return (x | z).bit_count() if metric == "pauli" else max(x.bit_count(), z.bit_count())


@pytest.mark.parametrize("metric", ["pauli", "css"])
def test_bounds_match_exhaustive_minima(metric: Metric, pytestconfig: pytest.Config) -> None:
    """Bounds agree with exhaustively computed minima for random graph states."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 7):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        matrix = _graph_state(num_qubits, edges, phase=num_qubits % 2 == 0)
        oracle = PauliCosetSearch(matrix, metric=metric)
        for syndrome in range(1 << num_qubits):
            minimum = int(_class_weights(matrix, syndrome, metric).min())
            for threshold in (0, 1):
                bound = oracle.get_min_weight_bound(syndrome, threshold)
                assert bound.within_max_weight == (minimum <= threshold)
                assert oracle.get_syndrome(bound.representative) == syndrome
                assert bound.lower_bound <= minimum
                if bound.within_max_weight:
                    assert _pauli_weight(bound.representative, num_qubits, metric) <= threshold
                else:
                    assert bound.lower_bound > threshold


def test_css_weight_counts_the_larger_sector() -> None:
    """X0 Z1 on |0+> has CSS weight one but physical weight two."""
    matrix = np.array([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)
    assert not PauliCosetSearch(matrix, metric="pauli").get_min_weight_bound(3, 1).within_max_weight
    css = PauliCosetSearch(matrix, metric="css")
    assert css.get_min_weight_bound(3, 1).within_max_weight
    assert not css.get_min_weight_bound(3, 0).within_max_weight


def test_css_weight_minimizes_both_sectors_jointly() -> None:
    """On a four-qubit path graph state, Z^4 is equivalent to pure X and to pure Z, but not to both.

    The X part and the Z part can each be made zero, yet no one error has CSS weight below two.
    """
    matrix = _graph_state(4, [(0, 1), (1, 2), (2, 3)])
    assert _class_weights(matrix, 15, "css").min() == 2
    oracle = PauliCosetSearch(matrix, metric="css")
    bound = oracle.get_min_weight_bound(15, 1)
    assert not bound.within_max_weight and bound.lower_bound == 2
    assert oracle.get_syndrome(bound.representative) == 15


def test_invalid_arguments_are_rejected() -> None:
    """Malformed states, metrics, queries, and thresholds other than zero and one raise."""
    # Not 2n columns, dependent rows, and anticommuting rows.
    for stabilizers in ([[1]], [[0, 0]], [[1, 0, 0, 0], [0, 0, 1, 0]]):
        with pytest.raises(ValueError):
            PauliCosetSearch(stabilizers)
    with pytest.raises(ValueError):
        PauliCosetSearch([[1, 0]], metric=cast(Metric, "invalid"))
    oracle = PauliCosetSearch([[1, 0]])
    with pytest.raises(ValueError):
        oracle.get_min_weight_bound(2, 1)
    with pytest.raises(ValueError):
        oracle.get_min_weight_bound(0, -1)
    with pytest.raises(ValueError):
        oracle.get_min_weight_bound(0, 2)
