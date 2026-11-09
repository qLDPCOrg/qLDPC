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


def _syndrome_words(syndromes: Sequence[int], num_qubits: int) -> npt.NDArray[np.uint64]:
    """Split packed syndromes into rows of 64-bit words, with word ``j`` in bits ``64 j`` and up."""
    width = (num_qubits + 63) // 64
    words = [[(syndrome >> (64 * jj)) % 2**64 for jj in range(width)] for syndrome in syndromes]
    return np.array(words, dtype=np.uint64).reshape(len(syndromes), width)


# Four-qubit blocks: a GHZ state, which is CSS, and the same state after S on its first qubit, which
# is not.  For the CSS weight, the oracle splits its search only on the GHZ block, and for the
# physical weight it searches Pauli errors on both, so one block covers that path.
_GHZ_BLOCK = ("XXXX", "ZZ__", "_ZZ_", "__ZZ")
_PHASED_GHZ_BLOCK = ("YXXX", "ZZ__", "_ZZ_", "__ZZ")
_MAX = np.iinfo(np.int64).max
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

    Each syndrome is queried at every maximum weight in random order, with or without the table of
    errors of CSS weight one.
    """
    if not css_singles_table:
        tune(max_css_single_pairs=0)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    tables_checked = 0
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
        # Queries of a search of Pauli errors by CSS weight build the table only within the patched
        # limit.  Reading the instance dictionary, rather than the cached property, keeps this check
        # from building the table itself.
        for search in oracle._searches:
            if search._is_pauli and css_weight and "_css_singles" in vars(search):
                assert (vars(search)["_css_singles"] is not None) == css_singles_table
                tables_checked += 1
    # States with phases on an even number of qubits are not CSS, so some queries need the table.
    assert tables_checked or not css_weight


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
            words = _syndrome_words(syndromes, num_qubits)
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
    words = _syndrome_words([0b0011, 0b1111], 4)
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
        words = _syndrome_words(syndromes, 68)
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
        words = _syndrome_words(syndromes, num_qubits)
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
# the representative.  Block minima (see ``_block_minimum``) independently check each answer.
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
    (_GHZ_BLOCK, True, 2635, 2, MinWeightBound(False, 5686, 3)),
    (_GHZ_BLOCK, True, 2635, 3, MinWeightBound(False, 5686, 4)),
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

        words = _syndrome_words(
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
            words = _syndrome_words(syndromes, num_qubits)
            for max_weight in range(num_qubits + 1):
                found = _find_each_not_within_max_weight(oracle, matrix, words, max_weight)
                assert found == np.flatnonzero(minima[syndromes] > max_weight).tolist()


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


def test_oracles_use_the_tuning_with_which_they_are_built(
    tune: Tune, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An oracle packs its information sets with the column orders of the tuning that built it.

    The cached searches of a CSS state, and the cached search of Pauli errors of a state that is
    not CSS, as with phases on two qubits, are keyed by those orders too.  So an oracle built under
    other orders builds its own searches, and one built under the same orders reuses them.
    """
    orders: list[int] = []
    get_disjoint_sets = information_sets._get_disjoint_sets

    def record(*args: Any) -> Any:
        orders.append(args[2])
        return get_disjoint_sets(*args)

    monkeypatch.setattr(information_sets, "_get_disjoint_sets", record)
    eye, zeros = np.eye(3, dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)
    zero_plus = np.block([[zeros, eye, zeros, zeros], [zeros, zeros, eye, zeros]])
    phased = _graph_state(2, [(0, 1)], phase=True)
    coset_leader_search._get_css_searches.cache_clear()
    coset_leader_search._get_pauli_search.cache_clear()
    CosetLeaderSearch(zero_plus, css_weight=True)
    CosetLeaderSearch(phased, css_weight=True)
    tune(disjoint_random_column_orders=3)
    CosetLeaderSearch(zero_plus, css_weight=True)
    CosetLeaderSearch(zero_plus, css_weight=True)
    info = coset_leader_search._get_css_searches.cache_info()
    assert (info.misses, info.hits) == (2, 1)
    CosetLeaderSearch(phased, css_weight=True)
    CosetLeaderSearch(phased, css_weight=True)
    default = DEFAULT_TUNING.disjoint_random_column_orders
    assert orders == [default, default, default, 3, 3, 3]


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


def test_oracles_tabulate_pairs_within_the_limit_of_their_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An oracle tabulates pairs of single-qubit errors only within the limit of its own tuning.

    Oracles of a state that is not CSS, built under the default tuning and then with no room for
    pairs, each answer a query that needs the table, so each tabulates within its own limit.
    """
    limits: list[tuple[int, bool]] = []
    tabulate = coset_leader_search._tabulate_css_singles

    def record(*args: Any) -> Any:
        table = tabulate(*args)
        limits.append((args[2], table is None))
        return table

    monkeypatch.setattr(coset_leader_search, "_tabulate_css_singles", record)
    matrix = _product_state(("Y__", "_Z_", "__X"))
    coset_leader_search._get_pauli_search.cache_clear()
    assert (
        CosetLeaderSearch(matrix, css_weight=True).get_min_weight_bound(0b011, 1).within_max_weight
    )
    with use_tuning(max_css_single_pairs=0):
        oracle = CosetLeaderSearch(matrix, css_weight=True)
    assert oracle.get_min_weight_bound(0b011, 1).within_max_weight
    assert limits == [(DEFAULT_TUNING.max_css_single_pairs, False), (0, True)]


def test_invalid_arguments_are_rejected() -> None:
    """Malformed states and queries raise, for either measure of weight."""
    # Not 2n columns, dependent rows, anticommuting rows, and a syndrome out of range.
    for css_weight in (False, True):
        for stabilizers in ([[1]], [[0, 0]], [[1, 0, 0, 0], [0, 0, 1, 0]]):
            with pytest.raises(ValueError):
                CosetLeaderSearch(stabilizers, css_weight=css_weight)
        with pytest.raises(ValueError):
            CosetLeaderSearch([[1, 0]], css_weight=css_weight).get_min_weight_bound(2, 1)


@pytest.mark.usefixtures("compilation_mode")
def test_cached_stopping_steps_are_reused_only_while_they_stay_first(
    pytestconfig: pytest.Config,
) -> None:
    """A search stops at the first step whose bound exceeds the maximum weight, in any query order.

    The bounds are those of sets of every rank, including zero, of a rank-one code with 64 sets, and
    of the empty code, for both weight factors.  Each pair of maximum weights is queried in turn,
    past both ends of each table, to the largest native integer, and every step of the bounds is
    tried as the cached step, as a stale step or a mix of steps that searches cached at the same
    time might leave.  The answer is always that of a linear search of the bounds, the planning
    buffer then caches its step, and the bounds never change.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    certificates = [
        information_sets.Certificate(0, (0,)),
        information_sets.Certificate(1, (1,) * 64),
        information_sets.Certificate(4, (0, 2, 4, 0)),
    ]
    for _ in range(12):
        dimension = int(rng.integers(1, 6))
        ranks = rng.integers(0, dimension + 1, rng.integers(1, 5))
        certificates.append(information_sets.Certificate(dimension, tuple(ranks.tolist())))
    for certificate, factor in itertools.product(certificates, (1, 2)):
        instance = affine_instance.AffineInstance(
            *_empty_sets(certificate),
            np.array(certificate.ranks, dtype=np.int64),
            certificate,
            factor,
        )
        bounds = instance.get_weight_bounds()
        finite = bounds[bounds < _MAX]
        max_weights = [*range(-2, int(finite.max()) + 3), _MAX - 1, _MAX]
        expected = {weight: _get_first_step(bounds, weight) for weight in max_weights}
        for first, second in itertools.product(max_weights, repeat=2):
            planning = coset_leader_search._get_planning(bounds)
            _check_stopping_step(planning, first, expected[first], bounds)
            _check_stopping_step(planning, second, expected[second], bounds)
        planning = coset_leader_search._get_planning(bounds)
        num_rows, num_sets = bounds.shape[0] - 1, bounds.shape[1] - 1
        for layer, count, weight in itertools.product(
            range(num_rows + 1), range(1, num_sets + 1), max_weights
        ):
            # Cache the step in the last row of the buffer (see ``_get_planning``).
            planning[-1, :2] = layer, count
            _check_stopping_step(planning, weight, expected[weight], bounds)


@pytest.mark.parametrize("css_weight", [False, True])
def test_long_ghz_states_have_their_known_minima(
    css_weight: bool, pytestconfig: pytest.Config
) -> None:
    """The coset leaders of a 256-qubit GHZ state weigh what its structure says, by either weight.

    Multiplying an error ``X^a Z^b`` by stabilizers can complement its X part ``a``, and reduce its
    Z part to nothing if ``|b|`` is even, and to a single qubit, which may lie in the X part,
    otherwise.  So by either measure of weight, the least weight of the error's syndrome class is
    ``max(min(|a|, n - |a|), |b| mod 2)``.  For the CSS weight, the X parts are searched modulo the
    code of ``X...X``, which has rank one and an information set for each of the 256 columns.
    Queries run one by one and in batches, for maximum weights at both ends and around ``n / 2``.
    The physical weight skips the queries whose answer is no at ``n / 2 - 1``, which a search of
    Pauli errors could prove only by visiting a vast number of words.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_qubits = 256
    matrix = _product_state(
        ["X" * num_qubits]
        + ["_" * index + "ZZ" + "_" * (num_qubits - index - 2) for index in range(num_qubits - 1)]
    )
    oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
    syndromes, minima = [], []
    for size, parity in itertools.product((0, 1, 2, 3, 100, 127, 128, 129, 200, 255), (0, 1)):
        x = sum(1 << int(qubit) for qubit in rng.choice(num_qubits, size, replace=False))
        z = sum(
            1 << int(qubit)
            for qubit in rng.choice(num_qubits, 2 * size % 7 + parity, replace=False)
        )
        syndromes.append(_get_syndrome(matrix, x | (z << num_qubits)))
        minima.append(max(min(size, num_qubits - size), z.bit_count() % 2))
    for max_weight in (0, 1, 2, 127, 128, 2, 127, 0):
        cases = [
            (syndrome, minimum)
            for syndrome, minimum in zip(syndromes, minima, strict=True)
            if css_weight or max_weight != 127 or minimum <= max_weight
        ]
        for syndrome, minimum in cases:
            bound = oracle.get_min_weight_bound(syndrome, max_weight)
            weight = _pauli_weight(bound.representative, num_qubits, css_weight)
            assert bound.within_max_weight == (minimum <= max_weight)
            assert _get_syndrome(matrix, bound.representative) == syndrome
            assert bound.lower_bound <= minimum <= weight
            assert (
                weight <= max_weight if bound.within_max_weight else bound.lower_bound > max_weight
            )
        words = _syndrome_words([syndrome for syndrome, _ in cases], num_qubits)
        found = _find_each_not_within_max_weight(oracle, matrix, words, max_weight)
        assert found == [index for index, (_, minimum) in enumerate(cases) if minimum > max_weight]


@pytest.mark.usefixtures("compilation_mode")
def test_queries_in_any_order_of_maximum_weights_are_exact(pytestconfig: pytest.Config) -> None:
    """Interleaved queries on oracles that share searches match exhaustive minima.

    Two generating sets of a random CSS state share their X and Z searches, and a state that is not
    CSS has one search of Pauli errors.  Maximum weights rise, fall, and repeat, the oracles take
    turns, and each answers single queries and batches.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    css_state = _random_css_state(5, 2, rng)
    cases = [
        (css_state, True),
        (_scramble(css_state, rng), True),
        (css_state, False),
        (_stabilizer_matrix(_block_generators(_PHASED_GHZ_BLOCK, 1)), True),
    ]
    oracles = [CosetLeaderSearch(matrix, css_weight=css_weight) for matrix, css_weight in cases]
    minima = [_css_minima(matrix, css_weight) for matrix, css_weight in cases]
    for max_weight in (2, 3, 4, 5, 4, 3, 2, 2, 5, 0, 3):
        for oracle, (matrix, css_weight), case_minima in zip(oracles, cases, minima, strict=True):
            num_qubits = len(matrix)
            for syndrome in range(1 << num_qubits):
                bound = oracle.get_min_weight_bound(syndrome, max_weight)
                weight = _pauli_weight(bound.representative, num_qubits, css_weight)
                assert bound.within_max_weight == (case_minima[syndrome] <= max_weight)
                assert _get_syndrome(matrix, bound.representative) == syndrome
                assert bound.lower_bound <= case_minima[syndrome] <= weight
                assert (
                    weight <= max_weight
                    if bound.within_max_weight
                    else bound.lower_bound > max_weight
                )
            syndromes = rng.permutation(1 << num_qubits)[:12]
            found = _find_each_not_within_max_weight(
                oracle, matrix, _syndrome_words(syndromes.tolist(), num_qubits), max_weight
            )
            assert found == np.flatnonzero(case_minima[syndromes] > max_weight).tolist()


@pytest.mark.usefixtures("compilation_mode")
def test_cosets_of_random_codes_match_brute_force(pytestconfig: pytest.Config) -> None:
    """Coset searches of random codes, and of the empty code, match the minima of all words.

    The last information set of a random code is often rank deficient, and the one set of the empty
    code has rank zero, so the searches skip sets that add nothing to their stopping bounds.  Single
    queries and batches, at maximum weights from below zero to past the number of columns, agree
    with the least weight of the words of each syndrome, and with each other.

    The first code has disjoint information sets of ranks five and four.  Syndrome 19 has a word of
    weight two that the first set reaches only above layer one, so a search at maximum weight two
    finds it only in layer one of the second set, which has then overcome its deficiency of one.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    deficient_code = np.array(
        [
            [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    codes = [deficient_code]
    for num_columns in [*range(1, 9), *rng.integers(1, 9, size=8).tolist()]:
        num_rows = int(rng.integers(0, num_columns))
        codes.append(
            get_random_full_rank_matrix(rng, num_rows, num_columns)
            if num_rows
            else np.zeros((0, num_columns), dtype=np.uint8)
        )
    for generators in codes:
        num_columns = generators.shape[1]
        if len(generators):
            checks = get_kernel(generators)
        else:
            checks = np.eye(num_columns, dtype=np.uint8)
        search = coset_leader_search._get_classical_search(generators, checks, 0)

        # The least weight of the words with each syndrome.
        words = (np.arange(1 << num_columns)[:, None] >> np.arange(num_columns)) & 1
        word_syndromes = (words @ checks.T % 2) @ (1 << np.arange(len(checks)))
        minima = np.full(1 << len(checks), num_columns + 1)
        np.minimum.at(minima, word_syndromes, words.sum(axis=1))

        syndromes = rng.permutation(1 << len(checks)).tolist()
        syndrome_words = _syndrome_words(syndromes, len(checks))
        for max_weight in range(-2, num_columns + 2):
            bounds = {}
            for syndrome in syndromes:
                bound = search.get_min_weight_bound(syndrome, max_weight)
                representative = (bound.representative >> np.arange(num_columns)) & 1
                weight = int(representative.sum())
                assert bound.within_max_weight == (minima[syndrome] <= max_weight)
                assert (checks @ representative % 2) @ (1 << np.arange(len(checks))) == syndrome
                assert bound.lower_bound <= minima[syndrome] <= weight
                assert (
                    weight <= max_weight
                    if bound.within_max_weight
                    else bound.lower_bound > max_weight
                )
                bounds[syndrome] = bound

            # Find each syndrome above the maximum weight with batches, which start after the last.
            found, start = [], 0
            best = np.empty(search.tables.inverse.shape[1], dtype=np.uint64)
            while True:
                index, lower_bound = coset_leader_search._find_first_not_within_max_weight(
                    syndrome_words[start:], max_weight, best, *search.tables, search.scale
                )
                if start + index == len(syndromes):
                    break
                syndrome = syndromes[start + index]
                representative = (search.unembed(best) >> np.arange(num_columns)) & 1
                assert (checks @ representative % 2) @ (1 << np.arange(len(checks))) == syndrome
                assert max_weight < lower_bound <= minima[syndrome]
                if max_weight >= 2:
                    assert bounds[syndrome] == MinWeightBound(
                        False, search.unembed(best), lower_bound
                    )
                found.append(syndrome)
                start += index + 1
            assert found == [syndrome for syndrome in syndromes if minima[syndrome] > max_weight]


@pytest.mark.usefixtures("compilation_mode")
def test_searches_visit_only_the_sets_that_add_to_their_bounds(pytestconfig: pytest.Config) -> None:
    """A search skips exactly the sets that add nothing to the bound of its stopping step.

    A set of rank ``r`` in a code of dimension ``k`` adds to the bound of a step only once it has
    completed layer ``k - r`` (see ``information_sets.get_step_lower_bounds``).  A search that stops
    at step ``(w, j)`` completes layer ``w`` on the first ``j`` sets and layer ``w - 1`` on the
    others, so it visits a set only if that layer is at least ``k - r``.  The sets are the disjoint
    information sets of random codes, whose last one is often rank deficient, and a set of rank zero
    with every row of the code.  A search whose answer is no weighs every word of the layers of the
    sets that it visits, and every answer is exact.
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
            pack_bits(rows.reshape(-1, num_columns)).reshape(num_sets, num_rows, -1),
            np.zeros((num_sets, num_rows, 0), dtype=np.uint64),
            np.ones(num_sets, dtype=np.int64),
            np.zeros(num_sets, dtype=np.int64),
            np.vstack([family.pivots, np.full(num_rows, -1)]),
            np.array(ranks, dtype=np.int64),
            information_sets.Certificate(num_rows, ranks),
        )
        inverse = coset_leader_search._get_inverse(checks)
        tables = coset_leader_search._get_search_tables(instance, inverse, num_columns)

        # The least weight of the words with each syndrome.
        words = (np.arange(1 << num_columns)[:, None] >> np.arange(num_columns)) & 1
        word_syndromes = (words @ checks.T % 2) @ (1 << np.arange(len(checks)))
        minima = np.full(1 << len(checks), num_columns + 1)
        np.minimum.at(minima, word_syndromes, words.sum(axis=1))

        for syndrome, max_weight in itertools.product(
            range(1 << len(checks)), range(num_columns + 1)
        ):
            best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
            best_weight = coset_leader_search._get_syndrome_word(
                _syndrome_words([syndrome], len(checks))[0], best, tables.inverse, tables.x_mask, 1
            )
            if best_weight <= max_weight:
                continue
            stop_layer, stop_num_completed, stop_bound = coset_leader_search._get_stopping_step(
                tables.planning, max_weight
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
            assert (checks @ representative % 2) @ (1 << np.arange(len(checks))) == syndrome
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


@pytest.mark.parametrize("compilation_mode", ["python"], indirect=True)
@pytest.mark.usefixtures("compilation_mode")
def test_queries_plan_only_when_a_search_needs_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A query finds its stopping step only if the word of its syndrome is too heavy.

    Queries of weight at most one are answered from tables, and a syndrome whose word is light
    enough, as every word is at a maximum weight of the number of qubits, needs no search, so
    neither looks for a stopping step.  A batch looks for it once, when its first search starts,
    whatever the number of searches.  Answers stay exact.
    """
    calls: list[int] = []
    get_stopping_step = coset_leader_search._get_stopping_step

    def record(planning: npt.NDArray[np.int64], max_weight: int) -> tuple[int, int, int]:
        calls.append(max_weight)
        return get_stopping_step(planning, max_weight)

    monkeypatch.setattr(coset_leader_search, "_get_stopping_step", record)
    matrix, syndrome = _ghz_blocks(3)
    coset_leader_search._get_pauli_search.cache_clear()
    oracle = CosetLeaderSearch(matrix, css_weight=False)
    for max_weight in (0, 1):
        assert not oracle.get_min_weight_bound(syndrome, max_weight).within_max_weight
    assert oracle.get_min_weight_bound(0, 2).within_max_weight
    assert oracle.get_min_weight_bound(syndrome, 9).within_max_weight
    assert oracle.find_first_not_within_max_weight(_syndrome_words([0, 0], 9), 2) == (2, None)
    assert oracle.find_first_not_within_max_weight(_syndrome_words([syndrome], 9), 9) == (1, None)
    assert not calls

    assert not oracle.get_min_weight_bound(syndrome, 2).within_max_weight
    assert calls == [2]
    index, bound = oracle.find_first_not_within_max_weight(
        _syndrome_words([0, syndrome, syndrome], 9), 3
    )
    assert (index, bound) == (3, None)
    index, bound = oracle.find_first_not_within_max_weight(
        _syndrome_words([syndrome, 0, syndrome], 9), 2
    )
    assert index == 0 and bound is not None and bound.lower_bound == 3
    assert calls == [2, 3, 2]


def test_maximum_weights_may_reach_the_largest_native_integer() -> None:
    """Queries accept any maximum weight that fits in a native integer, without overflow."""
    matrix, syndrome = _ghz_blocks(3)
    for css_weight, max_weight in itertools.product((False, True), (_MAX - 1, _MAX)):
        oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
        bound = oracle.get_min_weight_bound(syndrome, max_weight)
        assert bound.within_max_weight and 2 <= bound.lower_bound <= 3
        assert _get_syndrome(matrix, bound.representative) == syndrome
        assert oracle.find_first_not_within_max_weight(
            _syndrome_words([syndrome], 9), max_weight
        ) == (1, None)


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


def _empty_sets(
    certificate: information_sets.Certificate,
) -> tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Packed rows, labels, widths, label widths, and pivots of sets with no columns."""
    num_sets, dimension = len(certificate.ranks), certificate.dimension
    return (
        np.zeros((num_sets, dimension, 0), dtype=np.uint64),
        np.zeros((num_sets, dimension, 0), dtype=np.uint64),
        np.zeros(num_sets, dtype=np.int64),
        np.zeros(num_sets, dtype=np.int64),
        np.full((num_sets, dimension), -1, dtype=np.int64),
    )


def _get_first_step(bounds: npt.NDArray[np.int64], max_weight: int) -> tuple[int, int, int]:
    """The first step, layer by layer and set by set, whose bound exceeds ``max_weight``."""
    num_rows, num_sets = bounds.shape[0] - 1, bounds.shape[1] - 1
    for layer, count in itertools.product(range(num_rows + 1), range(1, num_sets + 1)):
        if bounds[layer, count] > max_weight:
            return layer, count, int(bounds[layer, count])
    return num_rows, num_sets, int(bounds[num_rows, num_sets])


def _check_stopping_step(
    planning: npt.NDArray[np.int64],
    max_weight: int,
    expected: tuple[int, int, int],
    bounds: npt.NDArray[np.int64],
) -> None:
    """Check the stopping step that a planning buffer gives, and that it caches only that step.

    The buffer caches the layer and the number of completed sets of the step in its last row (see
    ``_get_planning``), and its bounds stay ``bounds``.
    """
    step = coset_leader_search._get_stopping_step(planning, max_weight)
    assert step == expected
    assert tuple(planning[-1, :2]) == step[:2]
    assert np.array_equal(coset_leader_search._get_step_bounds(planning), bounds)
