# SPDX-License-Identifier: Apache-2.0

"""Tests for exact minimum-weight bounds on syndrome classes of Pauli errors."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import math

from . import _bounded_weight_search
from ._bounded_weight_search import PauliCosetSearch


@pytest.fixture(params=[False, True], ids=["compiled", "python"])
def kernels(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the compiled search, or its Python version for coverage."""
    if request.param:
        for name in (
            "_find_first_not_within",
            "_search_coset",
            "_visit_layer",
            "_keep_lighter",
            "_get_weight",
        ):
            function = getattr(_bounded_weight_search, name).py_func
            monkeypatch.setattr(_bounded_weight_search, name, function)


def _stabilizer_matrix(generators: Sequence[stim.PauliString]) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] with the given Pauli strings as rows."""
    return np.array([np.concatenate(gg.to_numpy()) for gg in generators], dtype=np.uint8)


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
    return _stabilizer_matrix(stim.Tableau.from_circuit(circuit).to_stabilizers())


def _class_weights(
    matrix: npt.NDArray[np.uint8], syndrome: int, css_weight: bool
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
    return (x | z).sum(axis=1) if not css_weight else np.maximum(x.sum(axis=1), z.sum(axis=1))


def _pauli_weight(pauli: int, num_qubits: int, css_weight: bool) -> int:
    """Weight of a packed Pauli, with X in the low bits and Z in the next block."""
    x, z = pauli & ((1 << num_qubits) - 1), pauli >> num_qubits
    return (x | z).bit_count() if not css_weight else max(x.bit_count(), z.bit_count())


def _get_syndrome(matrix: npt.NDArray[np.uint8], pauli: int) -> int:
    """Syndrome of a packed Pauli: its symplectic product with each row of the stabilizer matrix."""
    bits = np.array([(pauli >> jj) & 1 for jj in range(matrix.shape[1])])
    products = matrix @ math.symplectic_conjugate(bits) % 2
    return sum(int(bit) << jj for jj, bit in enumerate(products))


def _syndrome_words(syndromes: Sequence[int], num_qubits: int) -> npt.NDArray[np.uint64]:
    """Split packed syndromes into rows of words, with word ``j`` holding bits ``64 j`` and up."""
    width = (num_qubits + 63) // 64
    words = [[(syndrome >> (64 * jj)) % 2**64 for jj in range(width)] for syndrome in syndromes]
    return np.array(words, dtype=np.uint64).reshape(len(syndromes), width)


def _find_each_not_within(
    oracle: PauliCosetSearch, syndromes: npt.NDArray[np.uint64], max_weight: int
) -> list[int]:
    """Find each syndrome whose class has no error within ``max_weight``, one search at a time."""
    found: list[int] = []
    index = oracle.find_first_not_within(syndromes, max_weight)
    while index < len(syndromes):
        found.append(index)
        index += 1 + oracle.find_first_not_within(syndromes[index + 1 :], max_weight)
    return found


def _block_generators(block: Sequence[str], num_blocks: int) -> list[stim.PauliString]:
    """Generators of independent copies of a block state, given the generators of one block."""
    size = len(block[0])
    return [
        stim.PauliString("_" * (size * index) + local + "_" * (size * (num_blocks - index - 1)))
        for index in range(num_blocks)
        for local in block
    ]


def _ghz_blocks(num_blocks: int) -> tuple[npt.NDArray[np.uint8], int]:
    """Stabilizers of independent GHZ blocks, and the syndrome of Z on each block's last qubit.

    That syndrome flips two X-type stabilizers in every block, which needs a Z component on the
    block, so every equivalent error has Z weight at least ``num_blocks``.  Weight ``num_blocks`` is
    achieved, so it is the exact minimum by either measure of weight.
    """
    generators = _block_generators(("ZZZ", "X_X", "_XX"), num_blocks)
    witness = stim.PauliString("__Z" * num_blocks)
    syndrome = sum((not witness.commutes(gg)) << jj for jj, gg in enumerate(generators))
    return _stabilizer_matrix(generators), syndrome


def _block_minimum(
    block: Sequence[str], local_syndromes: Sequence[Sequence[int]], css_weight: bool
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
    if not css_weight:
        return sum(
            min(support for _, _, support in options[tuple(local)]) for local in local_syndromes
        )
    totals = {(0, 0)}
    for local in local_syndromes:
        totals = {(a + x, b + z) for a, b in totals for x, z, _ in options[tuple(local)]}
    return min(max(a, b) for a, b in totals)


@pytest.mark.usefixtures("kernels")
@pytest.mark.parametrize("css_weight", [False, True])
def test_bounds_match_exhaustive_minima(css_weight: bool, pytestconfig: pytest.Config) -> None:
    """Bounds agree with exhaustive minima.

    Each syndrome is queried at every threshold in random order.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 7):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        matrix = _graph_state(num_qubits, edges, phase=num_qubits % 2 == 0)
        oracle = PauliCosetSearch(matrix, css_weight=css_weight)
        for syndrome in rng.permutation(1 << num_qubits).tolist():
            minimum = int(_class_weights(matrix, syndrome, css_weight).min())
            for threshold in rng.permutation(num_qubits + 1).tolist():
                bound = oracle.get_min_weight_bound(syndrome, threshold)
                assert bound.within_max_weight == (minimum <= threshold)
                assert _get_syndrome(matrix, bound.representative) == syndrome
                assert bound.lower_bound <= minimum
                if bound.within_max_weight:
                    assert _pauli_weight(bound.representative, num_qubits, css_weight) <= threshold
                else:
                    assert bound.lower_bound > threshold


@pytest.mark.usefixtures("kernels")
@pytest.mark.parametrize("css_weight", [False, True])
def test_batches_find_each_class_beyond_the_threshold(
    css_weight: bool, pytestconfig: pytest.Config
) -> None:
    """Searching the rest of a batch after each class found finds exactly the classes beyond it.

    Every syndrome, in random order, is checked at every threshold against exhaustive minima.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 7):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        matrix = _graph_state(num_qubits, edges, phase=num_qubits % 2 == 0)
        oracle = PauliCosetSearch(matrix, css_weight=css_weight)
        syndromes = rng.permutation(1 << num_qubits).tolist()
        minima = np.array([_class_weights(matrix, ss, css_weight).min() for ss in syndromes])
        for threshold in range(num_qubits + 2):
            found = _find_each_not_within(oracle, _syndrome_words(syndromes, num_qubits), threshold)
            assert found == np.flatnonzero(minima > threshold).tolist()


@pytest.mark.parametrize("css_weight", [False, True])
def test_batches_read_syndromes_of_several_words(
    css_weight: bool, pytestconfig: pytest.Config
) -> None:
    """On 17 four-qubit GHZ blocks, batch answers agree with per-block minima.

    The 68 syndrome bits span two words.  Each error acts on at most three random qubits, and blocks
    that it does not touch add nothing to its minimum weight.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    block, num_blocks = ("XXXX", "ZZ__", "_ZZ_", "__ZZ"), 17
    generators = _block_generators(block, num_blocks)
    oracle = PauliCosetSearch(_stabilizer_matrix(generators), css_weight=css_weight)
    syndromes, minima = [], []
    for _ in range(30):
        error = stim.PauliString(4 * num_blocks)
        for qubit in rng.choice(4 * num_blocks, size=rng.integers(1, 4), replace=False):
            error[int(qubit)] = int(rng.integers(1, 4))
        syndromes.append(sum((not error.commutes(gg)) << jj for jj, gg in enumerate(generators)))
        bits = [(syndromes[-1] >> jj) & 1 for jj in range(4 * num_blocks)]
        local = [bits[4 * index : 4 * index + 4] for index in range(num_blocks)]
        minima.append(_block_minimum(block, [ss for ss in local if any(ss)], css_weight))
    for threshold in range(4):
        found = _find_each_not_within(oracle, _syndrome_words(syndromes, 68), threshold)
        assert found == np.flatnonzero(np.array(minima) > threshold).tolist()


def test_css_weight_counts_the_larger_sector() -> None:
    """X0 Z1 on |0+> has CSS weight one but physical weight two."""
    matrix = np.array([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)
    assert (
        not PauliCosetSearch(matrix, css_weight=False).get_min_weight_bound(3, 1).within_max_weight
    )
    css = PauliCosetSearch(matrix, css_weight=True)
    assert css.get_min_weight_bound(3, 1).within_max_weight
    assert not css.get_min_weight_bound(3, 0).within_max_weight


def test_css_weight_minimizes_both_sectors_jointly() -> None:
    """On a four-qubit path graph state, Z^4 is equivalent to pure X and to pure Z, but not to both.

    The X part and the Z part can each be made zero, yet no one error has CSS weight below two.
    """
    matrix = _graph_state(4, [(0, 1), (1, 2), (2, 3)])
    assert _class_weights(matrix, 15, True).min() == 2
    oracle = PauliCosetSearch(matrix, css_weight=True)
    bound = oracle.get_min_weight_bound(15, 1)
    assert not bound.within_max_weight and bound.lower_bound == 2
    assert _get_syndrome(matrix, bound.representative) == 15
    assert oracle.get_min_weight_bound(15, 2).within_max_weight


def test_syndrome_is_the_symplectic_product_with_each_stabilizer(
    pytestconfig: pytest.Config,
) -> None:
    """Bit ``j`` of a syndrome says whether the Pauli anticommutes with stabilizer ``j``."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    matrix = _graph_state(5, [(0, 1), (1, 2), (2, 3), (3, 4)], phase=True)
    oracle = PauliCosetSearch(matrix)
    for pauli in rng.integers(1 << 10, size=16).tolist():
        assert oracle._get_syndrome(pauli) == _get_syndrome(matrix, pauli)


@pytest.mark.usefixtures("kernels")
def test_layers_reach_their_last_combination() -> None:
    """A layer visits every combination of rows, including the last one.

    The rows are unit words and the offset is the sum of the last ``depth`` rows, so that only the
    last combination of layer ``depth`` gives the zero word.
    """
    for num_rows in range(1, 7):
        rows = (np.uint64(1) << np.arange(num_rows, dtype=np.uint64)).reshape(1, num_rows, 1)
        for depth in range(1, num_rows + 1):
            offset = np.bitwise_xor.reduce(rows[0, num_rows - depth :])
            best = np.empty(1, dtype=np.uint64)
            weight = _bounded_weight_search._visit_layer(
                offset,
                rows,
                np.ones(1, dtype=np.int64),
                np.empty(depth, dtype=np.int64),
                np.empty((depth, 1, 1), dtype=np.uint64),
                0,
                best,
                np.int64(num_rows),
                np.zeros(1, dtype=np.uint64),
                False,
            )
            assert weight == 0 and best[0] == 0


def test_rank_deficient_information_sets_certify_less() -> None:
    """Qubit 6 is isolated, so an information set is rank-deficient and certifies a weaker bound.

    Treating that set as full rank would overstate the bound and miss the weight-two error.
    """
    edges = [(0, 1), (0, 2), (0, 7), (1, 3), (1, 5), (2, 4), (3, 5), (4, 7), (5, 7)]
    matrix = _graph_state(8, edges)
    assert _class_weights(matrix, 44, False).min() == 2
    assert PauliCosetSearch(matrix).get_min_weight_bound(44, 2).within_max_weight


@pytest.mark.usefixtures("kernels")
def test_exhausted_search_gives_the_exact_minimum() -> None:
    """On |00000>, flipping every qubit has CSS weight five, which only exhaustion can prove."""
    zero_state = np.hstack([np.zeros((5, 5), dtype=np.uint8), np.eye(5, dtype=np.uint8)])
    oracle = PauliCosetSearch(zero_state, css_weight=True)
    bound = oracle.get_min_weight_bound(31, 4)
    assert not bound.within_max_weight and bound.lower_bound == 5
    assert oracle.get_min_weight_bound(31, 5).within_max_weight


@pytest.mark.parametrize("css_weight", [False, True])
def test_wide_states_span_several_packed_words(css_weight: bool) -> None:
    """Eleven GHZ blocks use 66 (CSS) or 99 (Pauli) bits, more than one 64-bit word."""
    num_blocks = 11
    matrix, syndrome = _ghz_blocks(num_blocks)
    oracle = PauliCosetSearch(matrix, css_weight=css_weight)
    if not css_weight:  # the CSS lower bound is too slow to prove at this width
        below = oracle.get_min_weight_bound(syndrome, num_blocks - 1)
        assert not below.within_max_weight and below.lower_bound == num_blocks
    found = oracle.get_min_weight_bound(syndrome, num_blocks)
    assert found.within_max_weight
    assert _get_syndrome(matrix, found.representative) == syndrome
    assert _pauli_weight(found.representative, 3 * num_blocks, css_weight) == num_blocks


@pytest.mark.parametrize("css_weight", [False, True])
def test_bounds_match_per_block_minima(css_weight: bool, pytestconfig: pytest.Config) -> None:
    """Five four-qubit GHZ blocks match per-block minima for every threshold up to five.

    The stabilizers of these blocks have even weight, but their cosets can have odd weight.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    block, num_blocks = ("XXXX", "ZZ__", "_ZZ_", "__ZZ"), 5
    matrix = _stabilizer_matrix(_block_generators(block, num_blocks))
    oracle = PauliCosetSearch(matrix, css_weight=css_weight)
    # For the first syndrome, the physical-weight search stops by its certificate while the lightest
    # error that it has visited is heavier than the minimum.
    syndromes = [443351, *rng.integers(1 << (num_blocks * len(block)), size=7).tolist()]
    for syndrome in syndromes:
        local = [
            [(syndrome >> (len(block) * index + jj)) & 1 for jj in range(len(block))]
            for index in range(num_blocks)
        ]
        minimum = _block_minimum(block, local, css_weight)
        for threshold in range(1, 6):
            bound = oracle.get_min_weight_bound(syndrome, threshold)
            assert bound.within_max_weight == (minimum <= threshold)
            assert bound.lower_bound <= minimum
            if not bound.within_max_weight:
                assert bound.lower_bound > threshold


def test_invalid_arguments_are_rejected() -> None:
    """Malformed states and queries raise."""
    # Not 2n columns, dependent rows, and anticommuting rows.
    for stabilizers in ([[1]], [[0, 0]], [[1, 0, 0, 0], [0, 0, 1, 0]]):
        with pytest.raises(ValueError):
            PauliCosetSearch(stabilizers)
    with pytest.raises(ValueError):
        PauliCosetSearch([[1, 0]]).get_min_weight_bound(2, 1)
