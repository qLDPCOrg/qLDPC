# SPDX-License-Identifier: Apache-2.0

"""Tests for exact minimum-weight bounds on syndrome classes of Pauli errors."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from math import comb
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import math
from qldpc._binary import get_kernel, pack_bits, row_reduce_binary, unpack_bits

from ...conftest import Tune
from .._tuning import DEFAULT_TUNING, use_tuning
from ..conftest import get_random_full_rank_matrix
from . import affine_instance, coset_leader_search, information_sets
from .coset_leader_search import CosetLeaderSearch, MinWeightBound


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


def _classical_syndrome(checks: npt.NDArray[np.uint8], words: npt.NDArray[np.int_]) -> Any:
    """The packed syndrome of a binary word for binary checks, or of each row of an array of words.

    Bit ``j`` of a syndrome is the parity of the word with row ``j`` of ``checks``.
    """
    return (words @ checks.T % 2) @ (1 << np.arange(len(checks)))


def _pack_syndromes(syndromes: Sequence[int], num_qubits: int) -> npt.NDArray[np.uint64]:
    """Split packed syndromes into rows of 64-bit words, with word ``j`` in bits ``64 j`` and up."""
    width = (num_qubits + 63) // 64
    words = [[(syndrome >> (64 * jj)) % 2**64 for jj in range(width)] for syndrome in syndromes]
    return np.array(words, dtype=np.uint64).reshape(len(syndromes), width)


# Four-qubit blocks: a GHZ state, which is CSS, and the same state after S on its first qubit, which
# is not.  For the CSS weight, the oracle splits its search only on the GHZ block, and for the
# physical weight it searches Pauli errors on both, so one block covers that path.
_GHZ_BLOCK = ("XXXX", "ZZ__", "_ZZ_", "__ZZ")
_PHASED_GHZ_BLOCK = ("YXXX", "ZZ__", "_ZZ_", "__ZZ")
_BLOCK_CASES = pytest.mark.parametrize(
    ("block", "css_weight"),
    [(_GHZ_BLOCK, False), (_GHZ_BLOCK, True), (_PHASED_GHZ_BLOCK, True)],
    ids=["ghz-physical", "ghz-css", "phased-ghz-css"],
)


def _find_each_not_within_max_weight(
    oracle: CosetLeaderSearch,
    matrix: npt.NDArray[np.uint8],
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
) -> list[int]:
    """Find each syndrome whose class has no error of weight at most ``max_weight``, in turn.

    The representative of each bound that a search returns has the syndrome for the stabilizers of
    ``matrix``, its lower bound exceeds ``max_weight``, and, for a maximum weight of at least two,
    the bound equals that of ``get_min_weight_bound``.  With one coset search,
    ``find_first_not_within_max_weight`` computes its bound separately from
    ``get_min_weight_bound``, so this checks one against the other.
    """
    found: list[int] = []
    while True:
        start = found[-1] + 1 if found else 0
        index, bound = oracle.find_first_not_within_max_weight(syndromes[start:], max_weight)
        if bound is None:
            assert start + index == len(syndromes)
            return found
        found.append(start + index)
        syndrome = int.from_bytes(syndromes[found[-1]].astype("<u8").tobytes(), "little")
        assert not bound.within_max_weight and bound.lower_bound > max_weight
        assert _get_syndrome(matrix, bound.representative) == syndrome
        if max_weight >= 2:
            assert bound == oracle.get_min_weight_bound(syndrome, max_weight)


def _random_css_state(
    num_qubits: int, num_x_stabilizers: int, rng: np.random.Generator
) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] of a random CSS state, with generators that mix pure-X and pure-Z.

    The X parts of the pure-X stabilizers are random independent rows, and the Z parts of the pure-Z
    stabilizers span their null space.  The generators are random invertible combinations of these
    stabilizers.
    """
    while True:
        x_rows = rng.integers(2, size=(num_x_stabilizers, num_qubits), dtype=np.uint8)
        if len(row_reduce_binary(x_rows)[2]) == num_x_stabilizers:
            break
    pure = np.zeros((num_qubits, 2 * num_qubits), dtype=np.uint8)
    pure[:num_x_stabilizers, :num_qubits] = x_rows
    pure[num_x_stabilizers:, num_qubits:] = get_kernel(x_rows)
    return _scramble(pure, rng)


def _scramble(matrix: npt.NDArray[np.uint8], rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """Random invertible combinations of the rows of a stabilizer matrix, which generate its group.

    The combinations are a random permutation of an upper triangular matrix with ones on its
    diagonal and random bits above it.
    """
    size = len(matrix)
    upper = np.triu(rng.integers(2, size=(size, size), dtype=np.uint8), 1) + np.eye(
        size, dtype=np.uint8
    )
    return (upper[rng.permutation(size)] @ matrix) % 2


def _css_minima(matrix: npt.NDArray[np.uint8], css_weight: bool = True) -> npt.NDArray[np.int_]:
    """The smallest CSS (or physical) weight of an error with each packed syndrome.

    The minima come from all ``4**n`` Paulis.
    """
    n = len(matrix)
    errors = (np.arange(1 << (2 * n))[:, None] >> np.arange(2 * n)) & 1
    checks = math.symplectic_conjugate(matrix.astype(int))
    syndromes = (errors @ checks.T % 2) @ (1 << np.arange(n))
    x, z = errors[:, :n], errors[:, n:]
    weights = np.maximum(x.sum(axis=1), z.sum(axis=1)) if css_weight else (x | z).sum(axis=1)
    minima = np.full(1 << n, 2 * n + 1)
    np.minimum.at(minima, syndromes, weights)
    return minima


def _classical_minima(checks: npt.NDArray[np.uint8]) -> npt.NDArray[np.int_]:
    """The least Hamming weight of the words with each packed syndrome for binary checks.

    The minima come from all ``2**n`` words of ``n`` bits.
    """
    num_columns = checks.shape[1]
    words = (np.arange(1 << num_columns)[:, None] >> np.arange(num_columns)) & 1
    minima = np.full(1 << len(checks), num_columns + 1)
    np.minimum.at(minima, _classical_syndrome(checks, words), words.sum(axis=1))
    return minima


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


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize(
    ("css_weight", "css_singles_table"), [(False, True), (True, True), (True, False)]
)
def test_bounds_match_exhaustive_minima(
    css_weight: bool,
    css_singles_table: bool,
    pytestconfig: pytest.Config,
    tune: Tune,
) -> None:
    """Bounds agree with exhaustive minima.

    Each syndrome is queried at every maximum weight in random order.  Graph states with phases on
    an even number of qubits are not CSS, so their queries by CSS weight look up errors of CSS
    weight one in a table, or, with no room for the table, match single-qubit X and Z errors.
    """
    if not css_singles_table:
        tune(max_css_single_pairs=0)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 7):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        matrix = _graph_state(num_qubits, edges, phase=num_qubits % 2 == 0)
        oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
        for syndrome in rng.permutation(1 << num_qubits).tolist():
            minimum = int(_class_weights(matrix, syndrome, css_weight).min())
            for max_weight in rng.permutation(num_qubits + 1).tolist():
                bound = oracle.get_min_weight_bound(syndrome, max_weight)
                assert bound.within_max_weight == (minimum <= max_weight)
                assert _get_syndrome(matrix, bound.representative) == syndrome
                assert bound.lower_bound <= minimum
                if bound.within_max_weight:
                    assert _pauli_weight(bound.representative, num_qubits, css_weight) <= max_weight
                else:
                    assert bound.lower_bound > max_weight


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("css_weight", [False, True])
def test_batches_find_each_class_above_max_weight(
    css_weight: bool, pytestconfig: pytest.Config
) -> None:
    """Repeated batch searches find exactly the classes of minimum weight above ``max_weight``.

    After each index that ``find_first_not_within_max_weight`` returns, it is called again on the
    syndromes after that index.  Every syndrome, in random order, is checked at every maximum weight
    against exhaustive minima.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 7):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        matrix = _graph_state(num_qubits, edges, phase=num_qubits % 2 == 0)
        oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
        syndromes = rng.permutation(1 << num_qubits).tolist()
        minima = np.array([_class_weights(matrix, ss, css_weight).min() for ss in syndromes])
        for max_weight in range(num_qubits + 2):
            words = _pack_syndromes(syndromes, num_qubits)
            found = _find_each_not_within_max_weight(oracle, matrix, words, max_weight)
            assert found == np.flatnonzero(minima > max_weight).tolist()


def test_batches_return_bounds_above_max_weight_plus_one() -> None:
    """On |0000> with CSS weight, the syndrome 1111 needs X on every qubit, so weight four.

    The search of X parts proves this at layer zero, since their code has no generators and each of
    its cosets is a single word.  A batch search at maximum weight two returns the same bound of
    four as ``get_min_weight_bound``, which ``_find_each_not_within_max_weight`` checks.
    """
    matrix = np.hstack([np.zeros((4, 4), dtype=np.uint8), np.eye(4, dtype=np.uint8)])
    oracle = CosetLeaderSearch(matrix, css_weight=True)
    words = _pack_syndromes([0b0011, 0b1111], 4)
    assert _find_each_not_within_max_weight(oracle, matrix, words, 2) == [1]
    assert oracle.get_min_weight_bound(0b1111, 2).lower_bound == 4


@_BLOCK_CASES
def test_batches_read_syndromes_of_several_words(
    block: tuple[str, ...], css_weight: bool, pytestconfig: pytest.Config
) -> None:
    """On 17 four-qubit blocks, batch answers agree with per-block minima.

    The 68 syndrome bits span two words.  Each error acts on at most three random qubits, and blocks
    that it does not touch add nothing to its minimum weight.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_blocks = 17
    generators = _block_generators(block, num_blocks)
    matrix = _stabilizer_matrix(generators)
    oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
    syndromes, minima = [], []
    for _ in range(30):
        error = stim.PauliString(4 * num_blocks)
        for qubit in rng.choice(4 * num_blocks, size=rng.integers(1, 4), replace=False):
            error[int(qubit)] = int(rng.integers(1, 4))
        syndromes.append(sum((not error.commutes(gg)) << jj for jj, gg in enumerate(generators)))
        bits = [(syndromes[-1] >> jj) & 1 for jj in range(4 * num_blocks)]
        local = [bits[4 * index : 4 * index + 4] for index in range(num_blocks)]
        minima.append(_block_minimum(block, [ss for ss in local if any(ss)], css_weight))
    for max_weight in range(4):
        words = _pack_syndromes(syndromes, 68)
        found = _find_each_not_within_max_weight(oracle, matrix, words, max_weight)
        assert found == np.flatnonzero(np.array(minima) > max_weight).tolist()


@_BLOCK_CASES
def test_batches_read_syndromes_of_fewer_words(
    block: tuple[str, ...], css_weight: bool, pytestconfig: pytest.Config
) -> None:
    """Syndromes in rows of one word suffice when they are small, also for more than 64 qubits.

    On 17 four-qubit blocks, errors on the first 16 blocks have syndromes below 2**64, and a batch
    of them in rows of one word, as the certification of fault propagation passes them, agrees with
    per-block minima.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_blocks = 17
    generators = _block_generators(block, num_blocks)
    matrix = _stabilizer_matrix(generators)
    oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
    syndromes, minima = [], []
    for _ in range(30):
        error = stim.PauliString(4 * num_blocks)
        for qubit in rng.choice(64, size=rng.integers(1, 4), replace=False):
            error[int(qubit)] = int(rng.integers(1, 4))
        syndromes.append(sum((not error.commutes(gg)) << jj for jj, gg in enumerate(generators)))
        bits = [(syndromes[-1] >> jj) & 1 for jj in range(64)]
        local = [bits[4 * index : 4 * index + 4] for index in range(16)]
        minima.append(_block_minimum(block, [ss for ss in local if any(ss)], css_weight))
    words = np.array(syndromes, dtype=np.uint64)[:, None]
    for max_weight in range(4):
        found = _find_each_not_within_max_weight(oracle, matrix, words, max_weight)
        assert found == np.flatnonzero(np.array(minima) > max_weight).tolist()


def test_x_and_z_syndromes_span_several_words(pytestconfig: pytest.Config) -> None:
    """On |0>^70 |+>^70, the syndromes of the X and Z parts have 70 bits, more than one word.

    An error is equivalent to one with X only on the first 70 qubits and Z only on the last 70, so
    its smallest CSS weight is the larger of the numbers of these.  The generators are scrambled,
    and the errors act on qubits whose syndrome bits fall in both words of each syndrome.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_qubits = 140
    pure = np.zeros((num_qubits, 2 * num_qubits), dtype=np.uint8)
    pure[:70, num_qubits:][np.arange(70), np.arange(70)] = 1  # Z on each of the first 70 qubits
    pure[70:, :num_qubits][np.arange(70), np.arange(70, num_qubits)] = 1  # X on each of the rest
    matrix = _scramble(pure, rng)
    oracle = CosetLeaderSearch(matrix, css_weight=True)
    syndromes, minima = [], []
    for _ in range(20):
        x_qubits = rng.choice([*range(50), *range(60, 70)], size=rng.integers(0, 4), replace=False)
        z_qubits = rng.choice(range(120, num_qubits), size=rng.integers(0, 4), replace=False)
        pauli = sum(1 << int(qq) for qq in x_qubits) + sum(
            1 << int(qq + num_qubits) for qq in z_qubits
        )
        syndromes.append(_get_syndrome(matrix, pauli))
        minima.append(max(len(x_qubits), len(z_qubits)))
    for max_weight in range(4):
        for syndrome, minimum in zip(syndromes, minima, strict=True):
            bound = oracle.get_min_weight_bound(syndrome, max_weight)
            assert bound.within_max_weight == (minimum <= max_weight)
            assert _get_syndrome(matrix, bound.representative) == syndrome
        words = _pack_syndromes(syndromes, num_qubits)
        found = _find_each_not_within_max_weight(oracle, matrix, words, max_weight)
        assert found == np.flatnonzero(np.array(minima) > max_weight).tolist()


def test_css_weight_counts_the_larger_sector() -> None:
    """X0 Z1 on |0+> has CSS weight one but physical weight two."""
    matrix = np.array([[0, 0, 1, 0], [0, 1, 0, 0]], dtype=np.uint8)
    assert (
        not CosetLeaderSearch(matrix, css_weight=False).get_min_weight_bound(3, 1).within_max_weight
    )
    css = CosetLeaderSearch(matrix, css_weight=True)
    assert css.get_min_weight_bound(3, 1).within_max_weight
    assert not css.get_min_weight_bound(3, 0).within_max_weight


def test_css_weight_minimizes_both_sectors_jointly() -> None:
    """On a four-qubit path graph state, Z^4 is equivalent to pure X and to pure Z, but not to both.

    The X part and the Z part can each be made zero, yet no one error has CSS weight below two.
    """
    matrix = _graph_state(4, [(0, 1), (1, 2), (2, 3)])
    assert _class_weights(matrix, 15, True).min() == 2
    oracle = CosetLeaderSearch(matrix, css_weight=True)
    bound = oracle.get_min_weight_bound(15, 1)
    assert not bound.within_max_weight and bound.lower_bound == 2
    assert _get_syndrome(matrix, bound.representative) == 15
    assert oracle.get_min_weight_bound(15, 2).within_max_weight


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize(("syndrome", "minimum"), [(44, 2), (108, 3)])
def test_rank_deficient_information_sets_give_valid_bounds(syndrome: int, minimum: int) -> None:
    """Rank-deficient sets give valid bounds whether a weight-two error exists or not.

    Qubit 6 is isolated, so the last information set has rank five rather than eight.  Syndrome 44
    has minimum weight two; treating the set as full rank would miss that error.  Syndrome 108 also
    flips the isolated qubit's stabilizer, so its minimum is three.  Certifying it at maximum weight
    two skips the deficient set, whose first two layers contribute nothing to the certificate.
    """
    edges = [(0, 1), (0, 2), (0, 7), (1, 3), (1, 5), (2, 4), (3, 5), (4, 7), (5, 7)]
    matrix = _graph_state(8, edges)
    assert _class_weights(matrix, syndrome, False).min() == minimum
    bound = CosetLeaderSearch(matrix).get_min_weight_bound(syndrome, 2)
    assert bound.within_max_weight == (minimum <= 2)
    assert bound.lower_bound <= minimum
    assert _get_syndrome(matrix, bound.representative) == syndrome
    if bound.within_max_weight:
        assert _pauli_weight(bound.representative, len(matrix), False) <= 2
    else:
        assert bound.lower_bound > 2


@pytest.mark.usefixtures("compilation_mode")
def test_exhausted_search_gives_the_exact_minimum() -> None:
    """On a state with stabilizers Y, Z, Z, Z, Z, the syndrome 11111 has CSS weight four.

    An error with that syndrome has X or Y on each of the last four qubits, so its X part has weight
    at least four, and Z X X X X reaches it.  The state is not CSS, so the oracle searches Pauli
    errors, and the bounds of its certificate stay below four, so only exhausting every layer proves
    the minimum.
    """
    matrix = _stabilizer_matrix(
        [stim.PauliString(pauli) for pauli in ("Y____", "_Z___", "__Z__", "___Z_", "____Z")]
    )
    oracle = CosetLeaderSearch(matrix, css_weight=True)
    bound = oracle.get_min_weight_bound(0b11111, 3)
    assert not bound.within_max_weight and bound.lower_bound == 4
    assert oracle.get_min_weight_bound(0b11111, 4).within_max_weight


@pytest.mark.parametrize("css_weight", [False, True])
def test_wide_states_span_several_packed_words(css_weight: bool) -> None:
    """Eleven GHZ blocks use 99 bits for the physical weight, more than one 64-bit word.

    For the CSS weight, the X and Z parts are searched separately, on 33 bits each.
    """
    num_blocks = 11
    matrix, syndrome = _ghz_blocks(num_blocks)
    oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
    below = oracle.get_min_weight_bound(syndrome, num_blocks - 1)
    assert not below.within_max_weight and below.lower_bound == num_blocks
    found = oracle.get_min_weight_bound(syndrome, num_blocks)
    assert found.within_max_weight
    assert _get_syndrome(matrix, found.representative) == syndrome
    assert _pauli_weight(found.representative, 3 * num_blocks, css_weight) == num_blocks


@_BLOCK_CASES
def test_bounds_match_per_block_minima(
    block: tuple[str, ...], css_weight: bool, pytestconfig: pytest.Config
) -> None:
    """Five four-qubit blocks match per-block minima for every maximum weight up to five.

    The stabilizers of these blocks have even weight, but their cosets can have odd weight.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_blocks = 5
    matrix = _stabilizer_matrix(_block_generators(block, num_blocks))
    oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
    # For the first syndrome, the physical-weight search stops by its certificate while the lightest
    # error that it has visited is heavier than the minimum.
    syndromes = [443351, *rng.integers(1 << (num_blocks * len(block)), size=7).tolist()]
    for syndrome in syndromes:
        local = [
            [(syndrome >> (len(block) * index + jj)) & 1 for jj in range(len(block))]
            for index in range(num_blocks)
        ]
        minimum = _block_minimum(block, local, css_weight)
        for max_weight in range(1, 6):
            bound = oracle.get_min_weight_bound(syndrome, max_weight)
            assert bound.within_max_weight == (minimum <= max_weight)
            assert bound.lower_bound <= minimum
            if not bound.within_max_weight:
                assert bound.lower_bound > max_weight


# Queries on three four-qubit blocks: the block, the weight (True for the CSS weight), the syndrome,
# the maximum weight, and the answer.  The representatives and bounds are those of the documented
# search order: layer zero (the word of the syndrome, then the offset of each information set),
# then each later layer set by set, keeping only strictly lighter words, until a word is light
# enough or the certificate stops the search.  In several cases an offset of layer zero weighs as
# much as the word of the syndrome, and the bound comes from the certificate, below the weight of
# the representative.  For the CSS weight, the X parts of the pure-X stabilizers of the GHZ block
# and the Z parts of its pure-Z stabilizers have even weight, so the bounds of its classical
# searches round up to the parity of each coset.  The X parts of the errors with syndrome 2635 have
# even weights above two, so its bound at maximum weight two is four.  Block minima (see
# ``_block_minimum``) independently check each answer.
_ORDERED_QUERIES = [
    (_GHZ_BLOCK, False, 742, 2, MinWeightBound(False, 338, 3)),
    (_GHZ_BLOCK, False, 742, 3, MinWeightBound(False, 338, 4)),
    (_GHZ_BLOCK, False, 350, 4, MinWeightBound(False, 1114165, 5)),
    (_GHZ_BLOCK, False, 734, 2, MinWeightBound(False, 282949, 3)),
    (_GHZ_BLOCK, False, 734, 4, MinWeightBound(True, 282949, 2)),
    (_PHASED_GHZ_BLOCK, True, 2635, 2, MinWeightBound(False, 1737, 3)),
    (_PHASED_GHZ_BLOCK, True, 2635, 4, MinWeightBound(False, 1737, 5)),
    (_PHASED_GHZ_BLOCK, True, 110, 2, MinWeightBound(False, 42, 3)),
    (_PHASED_GHZ_BLOCK, True, 1967, 5, MinWeightBound(True, 8421994, 2)),
    (_GHZ_BLOCK, True, 2635, 2, MinWeightBound(False, 5686, 4)),
    (_GHZ_BLOCK, True, 350, 3, MinWeightBound(False, 1114165, 4)),
]


@pytest.mark.usefixtures("compilation_mode")
def test_queries_return_the_word_and_bound_of_the_search_order() -> None:
    """Single and batch queries return the representatives and bounds of the search order.

    A batch of two syndromes of single-qubit errors, which every query accepts, and then the
    syndrome of the case, stops at the case if its answer is no, with the same bound.
    """
    for block, css_weight, syndrome, max_weight, expected in _ORDERED_QUERIES:
        num_qubits = 3 * len(block)
        matrix = _stabilizer_matrix(_block_generators(block, 3))
        oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
        bound = oracle.get_min_weight_bound(syndrome, max_weight)
        assert bound == expected

        # The answer is sound for the block minima.
        local = [
            [(syndrome >> (len(block) * index + jj)) & 1 for jj in range(len(block))]
            for index in range(3)
        ]
        minimum = _block_minimum(block, local, css_weight)
        assert _get_syndrome(matrix, bound.representative) == syndrome
        assert bound.within_max_weight == (minimum <= max_weight)
        assert (
            bound.lower_bound
            <= minimum
            <= _pauli_weight(bound.representative, num_qubits, css_weight)
        )

        words = _pack_syndromes(
            [_get_syndrome(matrix, 1), _get_syndrome(matrix, 1 << num_qubits), syndrome], num_qubits
        )
        index, batch_bound = oracle.find_first_not_within_max_weight(words, max_weight)
        if expected.within_max_weight:
            assert (index, batch_bound) == (3, None)
        else:
            assert (index, batch_bound) == (2, expected)


@pytest.mark.usefixtures("compilation_mode")
def test_css_bounds_match_exhaustive_minima(pytestconfig: pytest.Config) -> None:
    """On random CSS states, both queries for the CSS weight agree with exhaustive minima.

    The states have up to five qubits and any number of pure-X stabilizers, including none and all.
    Each syndrome is queried at every maximum weight in random order, and batch searches find
    exactly the classes of minimum weight above each maximum weight.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 6):
        for num_x_stabilizers in range(num_qubits + 1):
            matrix = _random_css_state(num_qubits, num_x_stabilizers, rng)
            oracle = CosetLeaderSearch(matrix, css_weight=True)
            minima = _css_minima(matrix)
            for syndrome in rng.permutation(1 << num_qubits).tolist():
                for max_weight in rng.permutation(num_qubits + 1).tolist():
                    bound = oracle.get_min_weight_bound(syndrome, max_weight)
                    assert bound.within_max_weight == (minima[syndrome] <= max_weight)
                    assert _get_syndrome(matrix, bound.representative) == syndrome
                    assert bound.lower_bound <= minima[syndrome]
                    if bound.within_max_weight:
                        weight = _pauli_weight(bound.representative, num_qubits, True)
                        assert weight <= max_weight
                    else:
                        assert bound.lower_bound > max_weight
            syndromes = rng.permutation(1 << num_qubits).tolist()
            words = _pack_syndromes(syndromes, num_qubits)
            for max_weight in range(num_qubits + 1):
                found = _find_each_not_within_max_weight(oracle, matrix, words, max_weight)
                assert found == np.flatnonzero(minima[syndromes] > max_weight).tolist()


@pytest.mark.usefixtures("compilation_mode")
def test_css_weight_searches_of_even_stabilizers_stop_at_bounds_of_each_class_parity() -> None:
    """Searches of Pauli errors by CSS weight stop at the bounds of each class, rounded to parity.

    The state is the graph state of a triangle on qubits 2, 5, and 6, with qubit 1 joined to qubit 2
    and qubits 0, 3, and 4 isolated, after S on each qubit of even degree.  Its stabilizers have
    even weight as binary words ``(x|z)``, and it is not CSS, so the oracle searches its Pauli
    errors.  The errors of a syndrome class share the parity of ``|x| + |z|``, which the test reads
    from every error.  Every query agrees with exhaustive minima.  A search that took the parity of
    the CSS weight instead would not: for syndrome 57, whose class has minimum two, the word of the
    syndrome has Hamming weight four but CSS weight three, and the bounds of odd classes stop the
    search at maximum weight two before it finds an error of weight two.

    Answers alone cannot show whether these searches round, since rounding mainly moves where an
    odd class stops (see Section 3.2 of ``brouwer_zimmermann.md``).  So the test also finds the
    stopping step of each query as ``_search_coset`` does, and checks it against the first step
    whose bound in ``AffineInstance.get_weight_bounds`` for the parity of the class exceeds the
    maximum weight.  ``information_sets_test.py`` and ``affine_instance_test.py`` check those bounds
    against their definition and against brute force.  Some odd classes stop sooner than they would
    without parity.
    """
    generators = ("Y______", "_XZ____", "_ZX__ZZ", "___Y___", "____Y__", "__Z__YZ", "__Z__ZY")
    matrix = _stabilizer_matrix([stim.PauliString(pauli) for pauli in generators])
    _assert_exact_answers(CosetLeaderSearch(matrix, css_weight=True), matrix, True)

    # Read the parity of each syndrome class from every error, and check that the class shares it.
    num_qubits = len(matrix)
    checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
    errors = (np.arange(1 << 2 * num_qubits)[:, None] >> np.arange(2 * num_qubits)) & 1
    syndromes, parities = _classical_syndrome(checks, errors), errors.sum(axis=1) % 2
    class_parities = np.zeros(1 << num_qubits, dtype=int)
    class_parities[syndromes] = parities
    assert np.array_equal(class_parities[syndromes], parities)

    # Find the stopping step of each query as ``_search_coset`` does.  Steps ``(layer,
    # num_completed)`` compare as tuples in the order of the search.
    instance = affine_instance.get_coset_instance(matrix, random_column_orders=0, weight_factor=2)
    inverse = coset_leader_search._get_inverse(checks)
    tables = coset_leader_search._get_search_tables(instance, inverse, num_qubits)
    unrounded = instance.get_weight_bounds()
    steps = list(itertools.product(range(len(unrounded)), range(1, unrounded.shape[1])))
    best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
    num_sooner = 0
    for syndrome in range(1 << num_qubits):
        _, parity = coset_leader_search._get_syndrome_word(
            _pack_syndromes([syndrome], num_qubits)[0], best, tables.inverse, tables.x_mask, 0
        )
        table = coset_leader_search._get_planning_table(tables.planning, parity)
        bounds = instance.get_weight_bounds(class_parities[syndrome])
        for max_weight in range(2, num_qubits):
            stopping_step = next(step for step in steps if bounds[step] > max_weight)
            assert coset_leader_search._get_stopping_step(table, max_weight)[:2] == stopping_step
            num_sooner += stopping_step < next(
                step for step in steps if unrounded[step] > max_weight
            )
    assert num_sooner


def test_cached_searches_answer_only_for_their_stabilizer_group(
    pytestconfig: pytest.Config,
) -> None:
    """Oracles built in turn for |000+++>, another generating set of it, and |+++000> are exact.

    The searches of a CSS state are cached by stabilizer group, so the second oracle reuses those of
    the first.  |+++000> has the same numbers of qubits and of pure-X stabilizers as |000+++>, but a
    different stabilizer group, so its oracle must not reuse them.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    eye, zeros = np.eye(3, dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)
    zero_plus = np.block([[zeros, eye, zeros, zeros], [zeros, zeros, eye, zeros]])
    plus_zero = np.block([[eye, zeros, zeros, zeros], [zeros, zeros, zeros, eye]])
    matrices = [_scramble(matrix, rng) for matrix in (zero_plus, zero_plus, plus_zero)]
    coset_leader_search._get_css_searches.cache_clear()
    oracles = [CosetLeaderSearch(matrix, css_weight=True) for matrix in matrices]
    for oracle, matrix in zip(oracles, matrices, strict=True):
        minima = _css_minima(matrix)
        for syndrome, max_weight in itertools.product(range(1 << 6), range(4)):
            bound = oracle.get_min_weight_bound(syndrome, max_weight)
            assert bound.within_max_weight == (minima[syndrome] <= max_weight)
            assert _get_syndrome(matrix, bound.representative) == syndrome


def test_oracles_use_the_tuning_with_which_they_are_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """An oracle builds its searches with the column orders of the tuning in force when it is built.

    The searches are cached, and the column orders key the cache.  So the first oracle built under
    some orders builds searches of its own with them, and a second oracle built under the same
    orders shares those searches.  The orders reach a search through
    ``information_sets.get_disjoint_family``, which the test wraps to record them.  The states are
    |000+++>, whose two classical searches are cached by stabilizer group, and a two-qubit graph
    state with phases, which is not CSS, so its search of Pauli errors is cached by stabilizers.
    Every oracle answers exactly.  The caches are cleared first, since earlier tests may have filled
    them.
    """
    orders: list[int] = []
    get_disjoint_family = information_sets.get_disjoint_family

    def record(*args: Any, random_column_orders: int) -> information_sets.BasisFamily:
        orders.append(random_column_orders)
        return get_disjoint_family(*args, random_column_orders=random_column_orders)

    monkeypatch.setattr(information_sets, "get_disjoint_family", record)
    eye, zeros = np.eye(3, dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)
    zero_plus = np.block([[zeros, eye, zeros, zeros], [zeros, zeros, eye, zeros]])
    phased = _graph_state(2, [(0, 1)], phase=True)
    coset_leader_search._get_css_searches.cache_clear()
    coset_leader_search._get_pauli_search.cache_clear()
    default = DEFAULT_TUNING.disjoint_random_column_orders
    for column_orders, matrix in itertools.product((default, 3), (zero_plus, phased)):
        with use_tuning(disjoint_random_column_orders=column_orders):
            first = CosetLeaderSearch(matrix, css_weight=True)
            assert orders and set(orders) == {column_orders}
            orders.clear()
            second = CosetLeaderSearch(matrix, css_weight=True)
            assert not orders
        for oracle in (first, second):
            _assert_exact_answers(oracle, matrix, True)


def test_cached_pauli_searches_answer_for_their_own_stabilizers_and_weight() -> None:
    """Oracles built in turn answer exactly for their own stabilizers and weight.

    The product states with stabilizers ``Y__``, ``_Z_``, ``__X`` and ``X__``, ``_Y_``, ``__Z`` are
    not CSS, so their oracles search Pauli errors by either weight, and both have three qubits.  The
    generators ``Y__``, ``YZ_``, ``__X`` of the first state give other syndromes to the same errors.
    In the first state, ``Z0 X1`` has the syndrome ``0b011`` and CSS weight one, but no error with
    that syndrome has physical weight below two.  Oracles are built for every state and weight under
    three tunings of the setup of their searches, and queried only once all are built, so that each
    answers after every search is cached.
    """
    states = [
        _product_state(generators)
        for generators in (("Y__", "_Z_", "__X"), ("Y__", "YZ_", "__X"), ("X__", "_Y_", "__Z"))
    ]
    coset_leader_search._get_pauli_search.cache_clear()
    oracles = []
    for changes in ({}, {"max_css_single_pairs": 0}, {"disjoint_random_column_orders": 3}):
        with use_tuning(**changes):
            for matrix, css_weight in itertools.product(states, (False, True)):
                oracles.append(
                    (matrix, css_weight, CosetLeaderSearch(matrix, css_weight=css_weight))
                )
    physical_bound = oracles[0][2].get_min_weight_bound(0b011, 1)
    assert not physical_bound.within_max_weight and physical_bound.lower_bound == 2
    css_bound = oracles[1][2].get_min_weight_bound(0b011, 1)
    assert css_bound.within_max_weight and css_bound.representative == 0b001010
    for matrix, css_weight, oracle in oracles:
        _assert_exact_answers(oracle, matrix, css_weight)


def test_cached_pauli_searches_ignore_later_changes_to_the_stabilizer_array() -> None:
    """Oracles answer for the stabilizers that they were built from, when an array changes later.

    Oracles of the state ``Y__``, ``_Z_``, ``__X`` are built from an array, which is then
    overwritten with the state ``X__``, ``_Y_``, ``__Z``, from which further oracles are built from
    the same array.  Oracles of the first state built again from a copy of it, and the oracles built
    before the change, still answer for the first state.
    """
    first, second = _product_state(("Y__", "_Z_", "__X")), _product_state(("X__", "_Y_", "__Z"))
    coset_leader_search._get_pauli_search.cache_clear()
    array = first.copy()
    oracles = [
        (first, css_weight, CosetLeaderSearch(array, css_weight=css_weight))
        for css_weight in (False, True)
    ]
    array[:] = second
    oracles += [
        (second, css_weight, CosetLeaderSearch(array, css_weight=css_weight))
        for css_weight in (False, True)
    ]
    oracles += [
        (first, css_weight, CosetLeaderSearch(first.copy(), css_weight=css_weight))
        for css_weight in (False, True)
    ]
    for matrix, css_weight, oracle in oracles:
        _assert_exact_answers(oracle, matrix, css_weight)


def test_tables_of_pairs_of_single_qubit_errors_respect_their_limit() -> None:
    """A table of products of X and Z errors keeps the first product of each syndrome.

    Three X errors (with the identity) and two Z errors make six pairs, which fit a limit of six but
    not of five.
    """
    x_singles = {0: 0, 0b01: 0b0001, 0b11: 0b0010}
    z_singles = {0: 0, 0b10: 0b0100}
    table = coset_leader_search._tabulate_css_singles(x_singles, z_singles, 6)
    assert table == {0: 0, 0b10: 0b0100, 0b01: 0b0001, 0b11: 0b0101}
    assert coset_leader_search._tabulate_css_singles(x_singles, z_singles, 5) is None


def test_invalid_arguments_are_rejected() -> None:
    """Malformed states and queries raise, for either measure of weight."""
    # Not 2n columns, dependent rows, anticommuting rows, and a syndrome out of range.
    for css_weight in (False, True):
        for stabilizers in ([[1]], [[0, 0]], [[1, 0, 0, 0], [0, 0, 1, 0]]):
            with pytest.raises(ValueError):
                CosetLeaderSearch(stabilizers, css_weight=css_weight)
        with pytest.raises(ValueError):
            CosetLeaderSearch([[1, 0]], css_weight=css_weight).get_min_weight_bound(2, 1)


def test_cached_stopping_steps_are_reused_exactly_while_they_stay_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For nonnegative weights, a cached stopping step is reused exactly while it stays first.

    The stopping step of a maximum weight is the first step whose bound exceeds it.
    ``_get_stopping_step`` reuses the step that a planning buffer caches if that step is the
    stopping step.  Otherwise, it finds the stopping step by binary search and caches it.  A reuse
    is observable only as a skipped binary search, so the test records the binary searches.

    Each step of the bounds is planted as the cached step.  A planted step may be stale, or pair the
    layer of one step with the number of completed sets of another, as interleaved queries might
    leave it.  Each maximum weight from zero to past the largest finite bound is then queried twice.
    The first query searches exactly if the planted step is not its stopping step.  The second
    query does not search, since the first cached its step.  Both queries return the stopping step,
    which a linear search finds, and its bound.  The bounds in the buffer never change.  A negative
    maximum weight always searches, since every bound is at least zero, so it is left out.

    The bounds are those of the empty code, of a rank-one code with 64 sets, and of sets of ranks
    zero, two, and four in a code of dimension four, for both weight factors.
    """
    searches: list[int] = []
    find_stopping_step = information_sets.find_stopping_step

    def record(*args: Any) -> tuple[int, int]:
        searches.append(args[1])
        return find_stopping_step(*args)

    monkeypatch.setattr(information_sets, "find_stopping_step", record)
    get_stopping_step = coset_leader_search._get_stopping_step.py_func
    certificates = [
        information_sets.Certificate(0, (0,)),
        information_sets.Certificate(1, (1,) * 64),
        information_sets.Certificate(4, (0, 2, 4, 0)),
    ]
    for certificate, factor in itertools.product(certificates, (1, 2)):
        bounds = certificate.get_weight_bounds(factor)
        num_rows, num_sets = bounds.shape[0] - 1, bounds.shape[1] - 1
        steps = list(itertools.product(range(num_rows + 1), range(1, num_sets + 1)))
        largest = int(bounds[bounds < information_sets.NO_WORD_BOUND].max())
        (table,) = coset_leader_search._get_planning([bounds])
        for planted, max_weight in itertools.product(steps, range(largest + 2)):
            stopping_step = next(step for step in steps if bounds[step] > max_weight)
            expected = (*stopping_step, int(bounds[stopping_step]))
            # Plant the cached step in the last row of the table (see ``_get_planning``).
            table[-1, :2] = planted
            searches.clear()
            assert get_stopping_step(table, max_weight) == expected
            assert len(searches) == (planted != stopping_step)
            assert get_stopping_step(table, max_weight) == expected
            assert len(searches) == (planted != stopping_step)
        assert np.array_equal(table[:-1], bounds)


# A code whose disjoint information sets have ranks five and four.
_RANK_DEFICIENT_CODE = np.array(
    [
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    ],
    dtype=np.uint8,
)


def _assert_classical_cosets_match_brute_force(generators: npt.NDArray[np.uint8]) -> None:
    """Check the classical search of the cosets of a code against the least weight of their words.

    Single queries and batches, at maximum weights from below zero to past the number of columns,
    agree with the least weight of the words of each syndrome.  From maximum weight two, they also
    agree with each other.  A coset without a word of weight at most the maximum weight gets a lower
    bound of at least the least weight above the maximum weight that a word of the coset can have.
    That weight has the parity of the coset's least weight if every row of ``generators`` has even
    weight.  Below maximum weight two, a single query of a coset without a word of weight at most
    one gets exactly that weight for maximum weight one, which it finds without a search.
    """
    num_columns = generators.shape[1]
    checks = get_kernel(generators)
    search = coset_leader_search._get_classical_search(generators, checks, 0)
    minima = _classical_minima(checks)
    even = not np.any(generators.sum(axis=1) % 2)

    def get_least_bound(syndrome: int, max_weight: int) -> int:
        """The least weight above the maximum weight that a word of the coset can have."""
        return max_weight + 1 + int(even and (max_weight + 1 - minima[syndrome]) % 2)

    syndromes = list(range(1 << len(checks)))
    packed_syndromes = _pack_syndromes(syndromes, len(checks))
    for max_weight in range(-1, num_columns + 2):
        bounds = {}
        for syndrome in syndromes:
            bound = search.get_min_weight_bound(syndrome, max_weight)
            representative = (bound.representative >> np.arange(num_columns)) & 1
            weight = int(representative.sum())
            assert bound.within_max_weight == (minima[syndrome] <= max_weight)
            assert _classical_syndrome(checks, representative) == syndrome
            assert bound.lower_bound <= minima[syndrome] <= weight
            if bound.within_max_weight:
                assert weight <= max_weight
            else:
                assert bound.lower_bound >= get_least_bound(syndrome, max_weight)
            if max_weight < 2 and minima[syndrome] >= 2:
                assert bound.lower_bound == get_least_bound(syndrome, 1)
            bounds[syndrome] = bound

        # Find each syndrome whose minimum exceeds the maximum weight with batches, each of which
        # starts after the last syndrome found.
        found, start = [], 0
        best = np.empty(search.tables.inverse.shape[1], dtype=np.uint64)
        while True:
            index, lower_bound = coset_leader_search._find_first_not_within_max_weight(
                packed_syndromes[start:], max_weight, best, *search.tables, search.scale
            )
            if start + index == len(syndromes):
                break
            syndrome = syndromes[start + index]
            representative = (search.unembed(best) >> np.arange(num_columns)) & 1
            assert _classical_syndrome(checks, representative) == syndrome
            assert get_least_bound(syndrome, max_weight) <= lower_bound <= minima[syndrome]
            if max_weight >= 2:
                assert bounds[syndrome] == MinWeightBound(False, search.unembed(best), lower_bound)
            found.append(syndrome)
            start += index + 1
        assert found == [syndrome for syndrome in syndromes if minima[syndrome] > max_weight]


@pytest.mark.usefixtures("compilation_mode")
def test_cosets_of_a_rank_deficient_code_match_brute_force() -> None:
    """The coset searches of a rank-deficient code match the minima of all words.

    The code has disjoint information sets of ranks five and four, so its searches skip the second
    set while it adds nothing to the bounds of their stopping steps.  Syndrome 19 has a word of
    weight two that the first set reaches only above layer one.  At maximum weight two, the stopping
    step is the one at which the second set completes layer one, which overcomes its deficiency of
    one, so the search visits that set.  It finds the word in layer zero of that set, its offset.
    Single queries and batches agree with the least weight of the words of each syndrome (see
    ``_assert_classical_cosets_match_brute_force``).  Below two, a single query of a coset without a
    word of weight at most one returns a bound of two without searching.
    """
    _assert_classical_cosets_match_brute_force(_RANK_DEFICIENT_CODE)


@pytest.mark.usefixtures("compilation_mode")
def test_cosets_of_an_even_code_have_bounds_of_their_parity() -> None:
    """Coset searches of an even code give each coset with no light word a bound of its parity.

    The code is ``_RANK_DEFICIENT_CODE`` with a last column that holds the parity of each row, so
    every word of the code has even weight, and the weights of the words of each coset share a
    parity.  A coset without words of weight at most ``m`` then has no word of weight ``m + 1`` if
    that integer has the other parity, so its lower bound is at least the least integer above ``m``
    of its parity (see ``_assert_classical_cosets_match_brute_force``).  The bounds of the searches,
    which round up to the parity of each coset, reach it.  So does the bound three of an odd coset
    of least weight three in a single query below maximum weight two, which does not search.  Bounds
    without parity would fall short of both.  The cosets have both parities, and least weights up to
    four.
    """
    parities = _RANK_DEFICIENT_CODE.sum(axis=1, keepdims=True) % 2
    _assert_classical_cosets_match_brute_force(np.hstack([_RANK_DEFICIENT_CODE, parities]))


@pytest.mark.usefixtures("compilation_mode")
def test_searches_visit_only_the_sets_that_add_to_their_bounds(pytestconfig: pytest.Config) -> None:
    """A search skips exactly the sets that add nothing to the bound of its stopping step.

    A set of rank ``r`` in a code of dimension ``k`` adds to the bound of a step only once it has
    completed layer ``k - r`` (see ``information_sets.get_step_lower_bounds``).  A search that stops
    at step ``(w, j)`` completes layer ``w`` on the first ``j`` sets and layer ``w - 1`` on the
    others, so it visits a set only if that layer is at least ``k - r``.  The sets are the disjoint
    information sets of random codes, whose last one is often rank deficient, and a set of rank zero
    with every row of the code.  A search whose answer is no weighs every word of the layers of the
    sets that it visits, and every answer is exact.  The exact number of words weighed guards the
    documented rule for skipping sets, so a legitimate future pruning of the coset scan would
    change it.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(12):
        num_columns = int(rng.integers(2, 9))
        num_rows = int(rng.integers(1, num_columns))
        generators = get_random_full_rank_matrix(rng, num_rows, num_columns)
        checks = get_kernel(generators)
        family = information_sets.get_disjoint_family(generators, None, random_column_orders=0)
        ranks = (*family.ranks.tolist(), 0)
        num_sets = len(ranks)
        rows = np.concatenate([family.generators, generators[None]])
        instance = affine_instance.AffineInstance(
            generators=pack_bits(rows.reshape(-1, num_columns)).reshape(num_sets, num_rows, -1),
            labels=np.zeros((num_sets, num_rows, 0), dtype=np.uint64),
            widths=np.ones(num_sets, dtype=np.int64),
            pivots=np.vstack([family.pivots, np.full(num_rows, -1)]),
            certificate=information_sets.Certificate(num_rows, ranks),
        )
        inverse = coset_leader_search._get_inverse(checks)
        tables = coset_leader_search._get_search_tables(instance, inverse, num_columns)
        minima = _classical_minima(checks)

        for syndrome, max_weight in itertools.product(
            range(1 << len(checks)), range(num_columns + 1)
        ):
            best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
            best_weight, parity = coset_leader_search._get_syndrome_word(
                _pack_syndromes([syndrome], len(checks))[0], best, tables.inverse, tables.x_mask, 1
            )
            if best_weight <= max_weight:
                continue
            table = coset_leader_search._get_planning_table(tables.planning, parity)
            stop_layer, stop_num_completed, stop_bound = coset_leader_search._get_stopping_step(
                table, max_weight
            )
            lower_bound, weighed = coset_leader_search._search_from_syndrome_word(
                max_weight,
                stop_layer,
                stop_num_completed,
                stop_bound,
                best,
                best_weight,
                tables.generators,
                tables.pivots,
                tables.x_mask,
                1,
            )
            representative = unpack_bits(best[None], num_columns)[0].astype(int)
            assert _classical_syndrome(checks, representative) == syndrome
            assert (lower_bound <= max_weight) == (minima[syndrome] <= max_weight)
            if lower_bound > max_weight:
                assert lower_bound <= minima[syndrome]
                last_layers = [
                    stop_layer if index < stop_num_completed else stop_layer - 1
                    for index in range(num_sets)
                ]
                assert weighed == sum(
                    comb(num_rows, layer)
                    for rank, last_layer in zip(ranks, last_layers, strict=True)
                    if last_layer >= num_rows - rank
                    for layer in range(last_layer + 1)
                )


def _product_state(generators: Sequence[str]) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] of the given Pauli strings."""
    return _stabilizer_matrix([stim.PauliString(pauli) for pauli in generators])


def _assert_exact_answers(
    oracle: CosetLeaderSearch, matrix: npt.NDArray[np.uint8], css_weight: bool
) -> None:
    """Check every query of an oracle against exhaustive minima of the state of ``matrix``."""
    num_qubits = len(matrix)
    minima = _css_minima(matrix, css_weight)
    for syndrome, max_weight in itertools.product(range(1 << num_qubits), range(num_qubits + 1)):
        bound = oracle.get_min_weight_bound(syndrome, max_weight)
        weight = _pauli_weight(bound.representative, num_qubits, css_weight)
        assert bound.within_max_weight == (minima[syndrome] <= max_weight)
        assert _get_syndrome(matrix, bound.representative) == syndrome
        assert bound.lower_bound <= minima[syndrome] <= weight
        assert weight <= max_weight if bound.within_max_weight else bound.lower_bound > max_weight


@pytest.mark.usefixtures("compilation_mode")
def test_connected_support_answers_match_exhaustive_minima(pytestconfig: pytest.Config) -> None:
    """The connected-support engine answers every query exactly, by either measure of weight.

    Graph states with phases are not CSS, and are weighed by physical weight.  CSS states with
    no pure-X stabilizer, with only pure-X stabilizers, and with both are weighed by both measures,
    and the CSS weight splits their searches.  Within ``max_weight``, the engine's bound is 1, or 0
    for an empty syndrome.  Otherwise it is ``max(max_weight + 1, int(syndrome != 0))``, since a
    negative ``max_weight`` admits no error.  These exact bounds imply the inequalities that
    ``_assert_exact_answers`` checks.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 6):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        graph_state = _graph_state(num_qubits, edges, phase=True)
        num_x_stabilizers = [0, num_qubits, num_qubits // 2][num_qubits % 3]
        css_state = _random_css_state(num_qubits, num_x_stabilizers, rng)
        for matrix, css_weight in [(graph_state, False), (css_state, False), (css_state, True)]:
            oracle = CosetLeaderSearch(matrix, css_weight=css_weight, method="connected_support")
            minima = _css_minima(matrix, css_weight)
            for syndrome, max_weight in itertools.product(
                range(1 << num_qubits), range(-1, num_qubits + 1)
            ):
                bound = oracle.get_min_weight_bound(syndrome, max_weight)
                weight = _pauli_weight(bound.representative, num_qubits, css_weight)
                assert _get_syndrome(matrix, bound.representative) == syndrome
                assert bound.within_max_weight == (minima[syndrome] <= max_weight)
                if bound.within_max_weight:
                    assert weight <= max_weight and bound.lower_bound == int(syndrome != 0)
                else:
                    assert bound.lower_bound == max(max_weight + 1, int(syndrome != 0))


def test_connected_support_batches_read_syndromes_of_several_words(
    pytestconfig: pytest.Config,
) -> None:
    """On 17 phased GHZ blocks, batches of the connected-support engine agree with block minima.

    The 68 syndrome bits span two words, and the physical weight takes one search of the Pauli
    errors.  Each error acts on at most three random qubits, and blocks that it does not touch add
    nothing to its minimum weight.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_blocks = 17
    generators = _block_generators(_PHASED_GHZ_BLOCK, num_blocks)
    matrix = _stabilizer_matrix(generators)
    oracle = CosetLeaderSearch(matrix, method="connected_support")
    syndromes, minima = [], []
    for _ in range(30):
        error = stim.PauliString(4 * num_blocks)
        for qubit in rng.choice(4 * num_blocks, size=rng.integers(1, 4), replace=False):
            error[int(qubit)] = int(rng.integers(1, 4))
        syndromes.append(sum((not error.commutes(gg)) << jj for jj, gg in enumerate(generators)))
        bits = [(syndromes[-1] >> jj) & 1 for jj in range(4 * num_blocks)]
        local = [bits[4 * index : 4 * index + 4] for index in range(num_blocks)]
        minima.append(_block_minimum(_PHASED_GHZ_BLOCK, [ss for ss in local if any(ss)], False))
    words = _pack_syndromes(syndromes, 68)
    for max_weight in range(4):
        found = _find_each_not_within_max_weight(oracle, matrix, words, max_weight)
        assert found == np.flatnonzero(np.array(minima) > max_weight).tolist()


def test_sparser_rows_span_the_same_space(pytestconfig: pytest.Config) -> None:
    """Adding rows to each other leaves rows of the same span, which no such addition makes lighter.

    The weight of a symplectic row counts the qubits on which it acts, whether by X, Y, or Z.
    """

    def get_weights(rows: npt.NDArray[np.uint8], homogeneous: bool) -> npt.NDArray[np.int_]:
        if homogeneous:
            return rows.sum(axis=1)
        half = rows.shape[1] // 2
        return (rows[:, :half] | rows[:, half:]).sum(axis=1)

    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    cases = []
    for homogeneous in (True, False):
        for _ in range(10):
            matrix = rng.integers(2, size=(6, 8), dtype=np.uint8)
            cases.append((matrix[row_reduce_binary(matrix.T.copy())[2]], homogeneous))
    for matrix, homogeneous in cases:
        rows, transform = coset_leader_search._sparsify(matrix, homogeneous=homogeneous)
        assert np.array_equal(transform @ matrix % 2, rows)
        assert len(row_reduce_binary(transform)[2]) == len(matrix)
        weights = get_weights(rows, homogeneous)
        for row, other in itertools.permutations(range(len(rows)), 2):
            assert get_weights((rows[row] ^ rows[other])[None], homogeneous)[0] >= weights[row]


def test_connected_support_rejects_what_it_cannot_search() -> None:
    """The engine rejects the CSS weight on a non-CSS state, and its batches reject large syndromes.

    Other methods are unknown.
    """
    with pytest.raises(ValueError, match="only on CSS states"):
        CosetLeaderSearch(
            _graph_state(2, [(0, 1)], phase=True), css_weight=True, method="connected_support"
        )
    method: Any = "brute_force"
    with pytest.raises(ValueError, match="Unknown coset-leader search method"):
        CosetLeaderSearch([[1, 0]], method=method)

    # A batch of the engine checks the range of its syndromes, through ``get_min_weight_bound``.
    oracle = CosetLeaderSearch([[1, 0]], method="connected_support")
    with pytest.raises(ValueError, match="out of range"):
        oracle.find_first_not_within_max_weight(np.array([[2]], dtype=np.uint64), 1)
