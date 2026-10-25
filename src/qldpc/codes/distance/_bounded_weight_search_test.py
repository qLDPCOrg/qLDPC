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

from ._bounded_weight_search import (
    PauliCosetSearch,
    _iter_fixed_weight_supports,
    _stream_fixed_weight_supports,
)

Metric = Literal["pauli", "css"]

# A 10-qubit graph state whose syndrome 1019 has minimum weight 3, but for which a search with
# maximum weight 3 alone proves only a lower bound of 2.
GRAPH_10_EDGES = (
    (0, 2),
    (0, 3),
    (0, 6),
    (0, 7),
    (1, 8),
    (2, 5),
    (2, 8),
    (3, 4),
    (3, 6),
    (3, 7),
    (4, 6),
    (4, 7),
    (4, 9),
    (5, 6),
    (6, 8),
    (7, 8),
)


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


def _block_generators(block: Sequence[str], num_blocks: int) -> list[stim.PauliString]:
    """Generators of independent copies of a block state, given the generators of one block."""
    size = len(block[0])
    return [
        stim.PauliString("_" * (size * index) + local + "_" * (size * (num_blocks - index - 1)))
        for index in range(num_blocks)
        for local in block
    ]


def _stabilizer_matrix(generators: Sequence[stim.PauliString]) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] with the given Pauli strings as rows."""
    return np.array([np.concatenate(gg.to_numpy()) for gg in generators], dtype=np.uint8)


def _ghz_blocks(num_blocks: int) -> tuple[npt.NDArray[np.uint8], int]:
    """Stabilizers of independent GHZ blocks, and the syndrome of Z on each block's last qubit.

    That syndrome flips two X-type stabilizers in every block, which needs a Z component on the
    block, so every equivalent error has Z weight at least ``num_blocks``.  Weight ``num_blocks`` is
    achieved, so it is the exact minimum in both metrics.
    """
    generators = _block_generators(("ZZZ", "X_X", "_XX"), num_blocks)
    witness = stim.PauliString("__Z" * num_blocks)
    syndrome = sum((not witness.commutes(gg)) << jj for jj, gg in enumerate(generators))
    return _stabilizer_matrix(generators), syndrome


def _block_minimum(
    block: Sequence[str], local_syndromes: Sequence[Sequence[int]], metric: Metric
) -> int:
    """Minimum weight of an error on independent blocks with the given per-block syndromes.

    The syndrome bits of a block are the commutation with its generators, in order.  Every Pauli on
    one block is listed, and the blocks are combined by dynamic programming.
    """
    size = len(block[0])
    stabilizers = [stim.PauliString(local) for local in block]
    options: dict[tuple[int, ...], list[tuple[int, int, int]]] = {}
    for paulis in itertools.product("_XYZ", repeat=size):
        error = stim.PauliString("".join(paulis))
        syndrome = tuple(int(not error.commutes(ss)) for ss in stabilizers)
        x_weight = sum(pp in "XY" for pp in paulis)
        z_weight = sum(pp in "YZ" for pp in paulis)
        options.setdefault(syndrome, []).append((x_weight, z_weight, size - paulis.count("_")))
    if metric == "pauli":
        return sum(
            min(support for _, _, support in options[tuple(local)]) for local in local_syndromes
        )
    totals = {(0, 0)}
    for local in local_syndromes:
        totals = {(a + x, b + z) for a, b in totals for x, z, _ in options[tuple(local)]}
    return min(max(a, b) for a, b in totals)


def test_fixed_weight_supports_cover_all_combinations() -> None:
    """Supports are exactly the combinations, whatever the batch size."""
    for dimension, weight in itertools.product(range(1, 8), range(1, 8)):
        if weight <= dimension:
            batches = _iter_fixed_weight_supports(dimension, weight, batch_size=3)
            supports = [tuple(support) for batch in batches for support in batch]
            assert sorted(supports) == list(itertools.combinations(range(dimension), weight))


def test_fixed_weight_supports_stream_in_lexicographic_order() -> None:
    """Streaming, used when supports are too many to rank in 64 bits, yields every combination."""
    batches = _stream_fixed_weight_supports(5, 2, batch_size=3)
    supports = [tuple(support) for batch in batches for support in batch]
    assert supports == list(itertools.combinations(range(5), 2))
    # C(200, 20) exceeds 2**63, so the iterator streams; take only its first batch.
    first = next(_iter_fixed_weight_supports(200, 20, batch_size=3))
    expected = itertools.islice(itertools.combinations(range(200), 20), 3)
    assert first.tolist() == [list(support) for support in expected]


@pytest.mark.parametrize("metric", ["pauli", "css"])
def test_bounds_match_exhaustive_minima(metric: Metric, pytestconfig: pytest.Config) -> None:
    """Bounds agree with exhaustive minima for shuffled queries, a tiny cache, and small batches."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 7):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        matrix = _graph_state(num_qubits, edges, phase=num_qubits % 2 == 0)
        minima = [
            int(_class_weights(matrix, syndrome, metric).min())
            for syndrome in range(1 << num_qubits)
        ]
        queries = list(itertools.product(range(1 << num_qubits), range(num_qubits + 1)))
        oracle = PauliCosetSearch(matrix, metric=metric, cache_size=3, batch_size=2)
        for index in rng.permutation(len(queries)):
            syndrome, threshold = queries[index]
            bound = oracle.get_min_weight_bound(syndrome, threshold)
            assert bound.within_max_weight == (minima[syndrome] <= threshold)
            assert oracle.get_syndrome(bound.representative) == syndrome
            assert bound.lower_bound <= minima[syndrome]
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
    assert oracle.get_min_weight_bound(15, 2).within_max_weight


def test_global_bound_after_threshold_queries() -> None:
    """Four GHZ blocks need weight four, so a bound of five must never be reported or cached."""
    matrix, syndrome = _ghz_blocks(4)
    oracle = PauliCosetSearch(matrix)
    below = oracle.get_min_weight_bound(syndrome, 3)
    at_minimum = oracle.get_min_weight_bound(syndrome, 4)
    assert not below.within_max_weight and below.lower_bound == 4
    assert at_minimum.within_max_weight
    assert oracle.get_syndrome(at_minimum.representative) == syndrome


def test_cached_lower_bound_survives_a_larger_threshold() -> None:
    """A search that finds an error keeps the lower bound proved by an earlier search."""
    matrix = _graph_state(10, GRAPH_10_EDGES)
    assert _class_weights(matrix, 1019, "pauli").min() == 3
    oracle = PauliCosetSearch(matrix)
    assert oracle.get_min_weight_bound(1019, 2).lower_bound == 3
    bound = oracle.get_min_weight_bound(1019, 3)
    assert bound.within_max_weight and bound.lower_bound == 3


def test_rank_deficient_information_sets_certify_less() -> None:
    """Qubit 6 is isolated, so an information set is rank-deficient and certifies a weaker bound.

    Treating that set as full rank would overstate the bound and miss the weight-two error.
    """
    edges = [(0, 1), (0, 2), (0, 7), (1, 3), (1, 5), (2, 4), (3, 5), (4, 7), (5, 7)]
    matrix = _graph_state(8, edges)
    assert _class_weights(matrix, 44, "pauli").min() == 2
    assert PauliCosetSearch(matrix).get_min_weight_bound(44, 2).within_max_weight


def test_exhausted_search_gives_the_exact_minimum() -> None:
    """On |00000>, flipping every qubit has CSS weight five, which only exhaustion can prove."""
    zero_state = np.hstack([np.zeros((5, 5), dtype=np.uint8), np.eye(5, dtype=np.uint8)])
    oracle = PauliCosetSearch(zero_state, metric="css")
    bound = oracle.get_min_weight_bound(31, 4)
    assert not bound.within_max_weight and bound.lower_bound == 5
    assert oracle.get_min_weight_bound(31, 5).within_max_weight


@pytest.mark.parametrize("metric", ["pauli", "css"])
def test_wide_states_span_several_packed_words(metric: Metric) -> None:
    """Eleven GHZ blocks use 66 (CSS) or 99 (Pauli) bits, more than one 64-bit word."""
    num_blocks = 11
    matrix, syndrome = _ghz_blocks(num_blocks)
    oracle = PauliCosetSearch(matrix, metric=metric)
    if metric == "pauli":  # the CSS lower bound is too slow to prove at this width
        below = oracle.get_min_weight_bound(syndrome, num_blocks - 1)
        assert not below.within_max_weight and below.lower_bound == num_blocks
    found = oracle.get_min_weight_bound(syndrome, num_blocks)
    assert found.within_max_weight
    assert oracle.get_syndrome(found.representative) == syndrome
    assert _pauli_weight(found.representative, 3 * num_blocks, metric) == num_blocks


@pytest.mark.parametrize("metric", ["pauli", "css"])
def test_bounds_match_per_block_minima(metric: Metric, pytestconfig: pytest.Config) -> None:
    """Five four-qubit GHZ blocks match per-block minima for every threshold up to five.

    The stabilizers of these blocks have even weight, but their cosets can have odd weight.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    block, num_blocks = ("XXXX", "ZZ__", "_ZZ_", "__ZZ"), 5
    matrix = _stabilizer_matrix(_block_generators(block, num_blocks))
    oracle = PauliCosetSearch(matrix, metric=metric)
    for _ in range(8):
        local = rng.integers(2, size=(num_blocks, len(block)))
        syndrome = sum(int(bit) << jj for jj, bit in enumerate(local.flatten()))
        minimum = _block_minimum(block, local.tolist(), metric)
        for threshold in range(1, 6):
            bound = oracle.get_min_weight_bound(syndrome, threshold)
            assert bound.within_max_weight == (minimum <= threshold)
            assert bound.lower_bound <= minimum
            if not bound.within_max_weight:
                assert bound.lower_bound > threshold


@pytest.mark.parametrize("metric", ["pauli", "css"])
def test_candidate_weights_above_one_byte_are_exact(metric: Metric) -> None:
    """Weights above 255 must not wrap around in a one-byte sum."""
    n = 257
    zero_state = np.hstack([np.zeros((n, n), dtype=np.uint8), np.eye(n, dtype=np.uint8)])
    oracle = PauliCosetSearch(zero_state, metric=metric)
    width = (3 * n + 63) // 64 if metric == "pauli" else (2 * n + 63) // 64
    all_ones = np.full((1, width), np.iinfo(np.uint64).max, dtype=np.uint64)
    expected = 64 * width // 2 if metric == "pauli" else n
    assert oracle._candidate_weights(all_ones).tolist() == [expected]


def test_invalid_arguments_are_rejected() -> None:
    """Malformed states, parameters, queries, and thresholds raise."""
    # Not 2n columns, dependent rows, and anticommuting rows.
    for stabilizers in ([[1]], [[0, 0]], [[1, 0, 0, 0], [0, 0, 1, 0]]):
        with pytest.raises(ValueError):
            PauliCosetSearch(stabilizers)
    with pytest.raises(ValueError):
        PauliCosetSearch([[1, 0]], metric=cast(Metric, "invalid"))
    with pytest.raises(ValueError):
        PauliCosetSearch([[1, 0]], cache_size=0)
    with pytest.raises(ValueError):
        PauliCosetSearch([[1, 0]], batch_size=0)
    oracle = PauliCosetSearch([[1, 0]])
    with pytest.raises(ValueError):
        oracle.get_min_weight_bound(2, 1)
    with pytest.raises(ValueError):
        oracle.get_min_weight_bound(0, -1)
