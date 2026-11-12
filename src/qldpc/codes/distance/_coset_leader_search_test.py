# SPDX-License-Identifier: Apache-2.0

"""Tests for exact minimum-weight bounds on syndrome classes of Pauli errors."""

from __future__ import annotations

import inspect
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

from ..conftest import Tune
from . import _coset_leader_search
from ._binary import symplectic_to_hamming
from ._brouwer_zimmermann import affine_instance
from ._coset_leader_search import CosetLeaderSearch, CosetSearchMethod, MinWeightBound
from ._information_sets import basis_families, completion_bounds
from ._information_sets import certificates as certificates_module
from ._tuning import DEFAULT_TUNING, MAX_VALUE, get_tuning, use_tuning
from .conftest import (
    EVEN_ROUTES_CODE,
    OVERLAPPING_ROUTES_CODE,
    TWO_FAMILIES_CODE,
    get_binary_matrix,
    get_random_full_rank_matrix,
)


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


# Graph states whose searches of Pauli errors, by physical and by CSS weight, have a maximum
# disjoint family with more information sets, or a larger partial set, than their greedy packing,
# which their costly queries take (see ``coset_routes``).  The family of the first state for the
# physical weight has more information sets, and those of the other two states have a larger partial
# set.  The two states for the physical weight differ in the answers that this family changes.  On
# the second, it returns other errors for some classes without a light enough error.
_ROUTED_PHYSICAL_STATES = (
    _graph_state(5, [(0, 3), (1, 2), (1, 3), (1, 4)], phase=True),
    _graph_state(5, [(0, 4), (2, 3)]),
)
_ROUTED_CSS_WEIGHT_STATE = _graph_state(5, [(0, 1), (0, 4), (1, 4)])


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
    # The phased GHZ block is slow to search as plain Python, so it gets fewer random syndromes.
    num_random = 3 if block == _PHASED_GHZ_BLOCK else 7
    syndromes = [443351, *rng.integers(1 << (num_blocks * len(block)), size=num_random).tolist()]
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
    inverse = _coset_leader_search._get_inverse(checks)
    tables = _coset_leader_search._get_search_tables(instance, inverse, num_qubits)
    unrounded = instance.get_weight_bounds()
    steps = list(itertools.product(range(len(unrounded)), range(1, unrounded.shape[1])))
    best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
    num_sooner = 0
    for syndrome in range(1 << num_qubits):
        _, parity = _coset_leader_search._get_syndrome_word(
            _pack_syndromes([syndrome], num_qubits)[0], best, tables.inverse, tables.x_mask, 0
        )
        table = _coset_leader_search._get_planning_table(tables.planning, parity)
        bounds = instance.get_weight_bounds(class_parities[syndrome])
        for max_weight in range(2, num_qubits):
            stopping_step = next(step for step in steps if bounds[step] > max_weight)
            assert _coset_leader_search._get_stopping_step(table, max_weight)[:2] == stopping_step
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
    _coset_leader_search._get_css_searches.cache_clear()
    oracles = [CosetLeaderSearch(matrix, css_weight=True) for matrix in matrices]
    for oracle, matrix in zip(oracles, matrices, strict=True):
        minima = _css_minima(matrix)
        for syndrome, max_weight in itertools.product(range(1 << 6), range(4)):
            bound = oracle.get_min_weight_bound(syndrome, max_weight)
            assert bound.within_max_weight == (minima[syndrome] <= max_weight)
            assert _get_syndrome(matrix, bound.representative) == syndrome


def test_oracles_use_the_tuning_with_which_they_are_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """An oracle builds its searches with the tuning in force, keyed by the fields that they read.

    The searches are cached, and the column orders key the cache.  So the first oracle built under
    some orders builds searches of its own with them, and a second oracle built under the same
    orders shares those searches.  The orders reach a search through
    ``information_sets.get_disjoint_family``, which the test wraps to record them.  The states are
    |000+++>, whose two classical searches are cached by stabilizer group, and a two-qubit graph
    state with phases, which is not CSS, so its search of Pauli errors is cached by stabilizers.
    Every oracle answers exactly.  The caches are cleared first, since earlier tests may have filled
    them.  Only a search of Pauli errors by CSS weight reads ``max_css_single_pairs``, so the
    classical searches, and a search of Pauli errors by physical weight, are shared by oracles built
    under different values of it, and a search of Pauli errors by CSS weight is not.  That search
    alone has no completion tables, so it alone is shared by oracles built under different values of
    ``coset_completion_block_bits``.
    """
    orders: list[int] = []
    get_disjoint_family = basis_families.get_disjoint_family

    def record(*args: Any, random_column_orders: int) -> basis_families.BasisFamily:
        orders.append(random_column_orders)
        return get_disjoint_family(*args, random_column_orders=random_column_orders)

    monkeypatch.setattr(basis_families, "get_disjoint_family", record)
    eye, zeros = np.eye(3, dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)
    zero_plus = np.block([[zeros, eye, zeros, zeros], [zeros, zeros, eye, zeros]])
    phased = _graph_state(2, [(0, 1)], phase=True)
    _coset_leader_search._get_css_searches.cache_clear()
    _coset_leader_search._get_pauli_search.cache_clear()
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
    for matrix, css_weight in ((zero_plus, True), (phased, False), (phased, True)):
        CosetLeaderSearch(matrix, css_weight=css_weight)
        paulis_by_css_weight = matrix is phased and css_weight
        changes: list[tuple[dict[str, Any], bool]] = [
            ({"max_css_single_pairs": 0}, paulis_by_css_weight),
            ({"coset_completion_block_bits": 4}, not paulis_by_css_weight),
        ]
        for change, reads in changes:
            orders.clear()
            with use_tuning(**change):
                CosetLeaderSearch(matrix, css_weight=css_weight)
            assert bool(orders) == reads


def test_oracles_build_other_families_with_the_tuning_they_are_built_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Costly queries build their families and tables with the tuning of the oracle that answers.

    With the threshold of the gate at zero, the first query that searches builds the other families
    of the search (see ``coset_routes``).  The most sets of the overlapping family and the column
    orders that it tries for each set reach ``information_sets.get_overlapping_family``, and the
    width of the blocks of completion tables and their limit of bytes reach
    ``completion_bounds.build_tables_within``, which the test wraps to record them.  They are those
    of the tuning in force when the oracle was built, not when it answers.
    """
    calls: list[tuple[int, int]] = []
    table_calls: set[tuple[int, int]] = set()
    get_overlapping_family = basis_families.get_overlapping_family
    build_tables_within = completion_bounds.build_tables_within
    signature = inspect.signature(get_overlapping_family)

    def record(*args: Any, **kwargs: Any) -> basis_families.BasisFamily:
        policy = signature.bind(*args, **kwargs).arguments["policy"]
        calls.append((policy.max_sets, policy.attempts_per_set))
        return get_overlapping_family(*args, **kwargs)

    def record_tables(
        rows: npt.NDArray[np.uint64], block_bits: int, max_bytes: int
    ) -> completion_bounds.CompletionTables | None:
        table_calls.add((block_bits, max_bytes))
        return build_tables_within(rows, block_bits, max_bytes)

    monkeypatch.setattr(basis_families, "get_overlapping_family", record)
    monkeypatch.setattr(completion_bounds, "build_tables_within", record_tables)
    _coset_leader_search._get_pauli_search.cache_clear()
    with use_tuning(
        overlapping_minimum_work=0,
        max_information_sets=5,
        overlapping_attempts_per_set=3,
        coset_completion_block_bits=4,
        completion_bound_max_table_bytes=12345,
    ):
        oracle = CosetLeaderSearch(_ROUTED_PHYSICAL_STATES[1])
    for syndrome, max_weight in itertools.product(range(1 << 5), range(2, 5)):
        oracle.get_min_weight_bound(syndrome, max_weight)
    assert calls == [(5, 3)]
    assert table_calls == {(4, 12345)}


def test_connected_support_oracles_make_checks_sparser_with_their_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connected-support oracle makes its checks sparser in the rounds of its tuning.

    The most rounds and the rounds without improvement of the tuning in force when the oracle is
    built reach ``_get_sparser_rows``, which the test wraps to record them.  The state is a graph
    state with phases, which is not CSS, so the oracle has one search of Pauli errors.
    """
    calls: list[tuple[int, int]] = []
    get_sparser_rows = _coset_leader_search._get_sparser_rows
    signature = inspect.signature(get_sparser_rows)

    def record(*args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        arguments = signature.bind(*args, **kwargs).arguments
        calls.append((arguments["max_rounds"], arguments["rounds_without_improvement"]))
        return get_sparser_rows(*args, **kwargs)

    monkeypatch.setattr(_coset_leader_search, "_get_sparser_rows", record)
    with use_tuning(max_sparsifying_rounds=5, sparsifying_rounds_without_improvement=2):
        CosetLeaderSearch(_graph_state(2, [(0, 1)], phase=True), method="connected_support")
    assert calls == [(5, 2)]


def test_oracles_under_other_gates_do_not_share_searches(tune: Tune) -> None:
    """An oracle answers as a fresh oracle under the threshold of its gate, whatever is cached.

    Oracles are built for a graph state, whose search of Pauli errors is cached by stabilizers, and
    for a CSS state whose pure-X stabilizers span a random code, whose two classical searches are
    cached by stabilizer group.  They are built under the threshold zero, then the largest value,
    then zero again.  Some answers differ between the thresholds, so an oracle that shared the
    search of another threshold would answer differently.
    """
    checks = get_kernel(OVERLAPPING_ROUTES_CODE)
    css_state = np.block(
        [
            [OVERLAPPING_ROUTES_CODE, np.zeros_like(OVERLAPPING_ROUTES_CODE)],
            [np.zeros_like(checks), checks],
        ]
    )
    for matrix in (_ROUTED_PHYSICAL_STATES[1], css_state):
        css_weight = matrix is css_state
        syndromes = range(0, 1 << len(matrix), max(1, (1 << len(matrix)) // 64))
        queries = list(itertools.product(syndromes, range(6)))
        answers = {}
        for minimum_work in (0, MAX_VALUE):
            _coset_leader_search._get_css_searches.cache_clear()
            _coset_leader_search._get_pauli_search.cache_clear()
            with use_tuning(overlapping_minimum_work=minimum_work):
                oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
                answers[minimum_work] = [oracle.get_min_weight_bound(*query) for query in queries]
        assert answers[0] != answers[MAX_VALUE]
        for minimum_work in (0, MAX_VALUE, 0):
            tune(overlapping_minimum_work=minimum_work)
            oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
            answers_now = [oracle.get_min_weight_bound(*query) for query in queries]
            assert answers_now == answers[minimum_work]


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
    _coset_leader_search._get_pauli_search.cache_clear()
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


def test_cached_pauli_searches_ignore_later_changes_to_the_stabilizer_array(tune: Tune) -> None:
    """Oracles answer for the stabilizers that they were built from, when an array changes later.

    Oracles of a graph state are built from an array, which is then overwritten with another graph
    state, from which further oracles are built from the same array.  Oracles of the first state
    built again from a copy of it, and the oracles built before the change, still answer for the
    first state.  With the threshold of the gate at zero, the searches build their other families
    at their first queries, after the change (see ``coset_routes``).  The first state has a larger
    disjoint family by CSS weight, and its search by CSS weight reads its generators from the
    stabilizers themselves.
    """
    tune(overlapping_minimum_work=0)
    first, second = _ROUTED_CSS_WEIGHT_STATE, _ROUTED_PHYSICAL_STATES[1]
    _coset_leader_search._get_pauli_search.cache_clear()
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
    table = _coset_leader_search._tabulate_css_singles(x_singles, z_singles, 6)
    assert table == {0: 0, 0b10: 0b0100, 0b01: 0b0001, 0b11: 0b0101}
    assert _coset_leader_search._tabulate_css_singles(x_singles, z_singles, 5) is None


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
    stopping step.  Otherwise, it plans the stopping step, which a binary search finds, and caches
    it.  A reuse is observable only as a skipped plan, so the test records the plans.

    Each step of the bounds is planted as the cached step.  A planted step may be stale, or pair the
    layer of one step with the number of completed sets of another, as interleaved queries might
    leave it.  Each maximum weight from zero to past the largest finite bound is then queried twice.
    The first query plans exactly if the planted step is not its stopping step.  The second query
    does not plan, since the first cached its step.  Both queries return the stopping step, which a
    linear search finds, and its bound.  The bounds in the buffer never change.  A negative maximum
    weight always plans, since every bound is at least zero, so it is left out.

    The bounds are those of the empty code, of a rank-one code with 64 sets, and of sets of ranks
    zero, two, and four in a code of dimension four, for both weight factors.
    """
    plans: list[int] = []
    find_stopping_step = certificates_module.find_stopping_step

    def record(*args: Any) -> tuple[int, int]:
        plans.append(args[1])
        return find_stopping_step(*args)

    monkeypatch.setattr(certificates_module, "find_stopping_step", record)
    # Take the Python version even when the function is compiled, since only it calls the recording
    # function.
    function = _coset_leader_search._get_stopping_step
    get_stopping_step = getattr(function, "py_func", function)
    certificates = [
        certificates_module.Certificate(0, (0,)),
        certificates_module.Certificate(1, (1,) * 64),
        certificates_module.Certificate(4, (0, 2, 4, 0)),
    ]
    for certificate, factor in itertools.product(certificates, (1, 2)):
        bounds = certificate.get_weight_bounds(factor)
        num_rows, num_sets = bounds.shape[0] - 1, bounds.shape[1] - 1
        steps = list(itertools.product(range(num_rows + 1), range(1, num_sets + 1)))
        largest = int(bounds[bounds < certificates_module.NO_WORD_BOUND].max())
        (table,) = _coset_leader_search._get_planning([bounds])
        for planted, max_weight in itertools.product(steps, range(largest + 2)):
            stopping_step = next(step for step in steps if bounds[step] > max_weight)
            expected = (*stopping_step, int(bounds[stopping_step]))
            # Plant the cached step in the last row of the table (see ``_get_planning``).
            table[-1, :2] = planted
            plans.clear()
            assert get_stopping_step(table, max_weight) == expected
            assert len(plans) == (planted != stopping_step)
            assert get_stopping_step(table, max_weight) == expected
            assert len(plans) == (planted != stopping_step)
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

# A code whose searches for minimum weights by tails skip completions by tables at the threshold
# zero, as do those of its even version.
_SKIPPING_CODE = np.array(
    [
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)


# A longer code with cosets of minimum weight six.  At maximum weight four and the threshold zero,
# its searches of them for minimum weights skip some completions by tables whose bound reaches the
# skip bound of that weight, but not the weight of the lightest word that the searches weigh.  The
# searches by tails of some of its cosets, such as those of syndromes 1126, 4578, and 6129, stop
# early within steps.
_HEAVY_COSETS_CODE = get_binary_matrix(
    [
        "11100011110000110001",
        "00001011001010001000",
        "10010110101010101100",
        "10001100101100100101",
        "01010100110001001100",
        "00010000001110010101",
        "00101100011111000001",
    ]
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
    tuning = _coset_leader_search._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    search = _coset_leader_search._get_classical_search(
        generators, checks, tuning._replace(disjoint_random_column_orders=0)
    )
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
        while True:
            index, batch_bound = search.find_first_not_within_max_weight(
                packed_syndromes[start:], max_weight
            )
            if batch_bound is None:
                assert start + index == len(syndromes)
                break
            syndrome = syndromes[start + index]
            representative = (batch_bound.representative >> np.arange(num_columns)) & 1
            assert _classical_syndrome(checks, representative) == syndrome
            assert not batch_bound.within_max_weight
            assert get_least_bound(syndrome, max_weight) <= batch_bound.lower_bound
            assert batch_bound.lower_bound <= minima[syndrome]
            if max_weight >= 2:
                assert bounds[syndrome] == batch_bound
            found.append(syndrome)
            start += index + 1
        assert found == [syndrome for syndrome in syndromes if minima[syndrome] > max_weight]


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
    parities = (_RANK_DEFICIENT_CODE.sum(axis=1, keepdims=True) % 2).astype(np.uint8)
    _assert_classical_cosets_match_brute_force(np.hstack([_RANK_DEFICIENT_CODE, parities]))


@pytest.mark.parametrize(
    "code", [EVEN_ROUTES_CODE, TWO_FAMILIES_CODE], ids=["even", "two-families"]
)
def test_cosets_match_brute_force_with_other_families(
    code: npt.NDArray[np.uint8], tune: Tune
) -> None:
    """Searches of X or Z parts answer exactly when costly queries take other families.

    With the threshold of the gate at zero, queries that the disjoint family would answer with more
    work take another family (see ``coset_routes``).  The even code takes a disjoint family with
    two information sets, for cosets of both parities.  The random code takes a disjoint family with
    a larger partial set, and a family of overlapping information sets.  Single queries and
    batches, at maximum weights from below zero to past the number of columns, and so just below
    and at the least weight of each coset, agree with the least weight of the words of each coset
    (see ``_assert_classical_cosets_match_brute_force``).  At the threshold zero, the queries also
    run the searches by tails.
    """
    tune(overlapping_minimum_work=0)
    _assert_classical_cosets_match_brute_force(code)


@pytest.mark.parametrize(
    "matrix, css_weight",
    [
        (_ROUTED_PHYSICAL_STATES[0], False),
        (_ROUTED_PHYSICAL_STATES[1], False),
        (_ROUTED_CSS_WEIGHT_STATE, True),
        (np.hstack([np.zeros((4, 4), np.uint8), np.eye(4, dtype=np.uint8)]), True),
    ],
    ids=["physical-0", "physical-1", "css-weight", "empty-x-code"],
)
def test_oracles_answer_exactly_with_other_families(
    matrix: npt.NDArray[np.uint8], css_weight: bool, tune: Tune
) -> None:
    """Oracles answer exactly when costly queries take other families.

    With the threshold of the gate at zero, the searches of Pauli errors of the graph states take
    their maximum disjoint families, which change different answers on the two states by physical
    weight.  On |0000>, the CSS weight splits the search, and the X parts have the empty code, which
    no other family improves.  At the threshold zero, the queries also run the searches by tails,
    except in the search of Pauli errors by CSS weight, which has none.  Every single query, and
    every batch, at every maximum weight, agrees with exhaustive minima.
    """
    tune(overlapping_minimum_work=0)
    oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
    _assert_exact_answers(oracle, matrix, css_weight)
    num_qubits = len(matrix)
    minima = _css_minima(matrix, css_weight)
    syndromes = _pack_syndromes(list(range(1 << num_qubits)), num_qubits)
    for max_weight in range(num_qubits + 1):
        found = _find_each_not_within_max_weight(oracle, matrix, syndromes, max_weight)
        assert found == np.flatnonzero(minima > max_weight).tolist()


def test_answers_do_not_depend_on_earlier_queries(pytestconfig: pytest.Config, tune: Tune) -> None:
    """The answers of a search depend only on its code, the syndrome, and the maximum weight.

    With the threshold of the gate at zero, fresh searches of the cosets of a random code answer
    every query in increasing, decreasing, and random order of the maximum weight.  Each order
    reaches the threshold at a different query, but the overlapping family of the code, which
    depends on the weight at which it is built, and so the answers, are the same.  With the largest
    threshold, some answers differ, so the other families answer some of the queries.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    checks = get_kernel(OVERLAPPING_ROUTES_CODE)

    def get_answers(order: str) -> dict[tuple[int, int], MinWeightBound]:
        """Every answer of a fresh search, with the maximum weights in the given order."""
        tuning = _coset_leader_search._CosetTuning.from_tuning(
            get_tuning(), paulis_by_css_weight=False
        )
        search = _coset_leader_search._get_classical_search(
            OVERLAPPING_ROUTES_CODE, checks, tuning._replace(disjoint_random_column_orders=0)
        )
        weights = list(range(15))
        if order != "up":
            weights = weights[::-1] if order == "down" else rng.permutation(weights).tolist()
        return {
            (syndrome, max_weight): search.get_min_weight_bound(syndrome, max_weight)
            for max_weight in weights
            for syndrome in range(1 << len(checks))
        }

    tune(overlapping_minimum_work=0)
    answers = [get_answers(order) for order in ("up", "down", "random")]
    assert answers[0] == answers[1] == answers[2]
    with use_tuning(overlapping_minimum_work=MAX_VALUE):
        assert get_answers("up") != answers[0]


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
    change it.  The work of the plan of the steps of the search, from layer zero and without the
    sets that it skips (see ``information_sets.plan_steps``), predicts this number.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(12):
        num_columns = int(rng.integers(2, 9))
        num_rows = int(rng.integers(1, num_columns))
        generators = get_random_full_rank_matrix(rng, num_rows, num_columns)
        checks = get_kernel(generators)
        family = basis_families.get_disjoint_family(generators, None, random_column_orders=0)
        ranks = (*family.ranks.tolist(), 0)
        num_sets = len(ranks)
        rows = np.concatenate([family.generators, generators[None]])
        instance = affine_instance.AffineInstance(
            generators=pack_bits(rows.reshape(-1, num_columns)).reshape(num_sets, num_rows, -1),
            labels=np.zeros((num_sets, num_rows, 0), dtype=np.uint64),
            widths=np.ones(num_sets, dtype=np.int64),
            pivots=np.vstack([family.pivots, np.full(num_rows, -1)]),
            certificate=certificates_module.Certificate(num_rows, ranks),
        )
        inverse = _coset_leader_search._get_inverse(checks)
        tables = _coset_leader_search._get_search_tables(instance, inverse, num_columns)
        minima = _classical_minima(checks)

        for syndrome, max_weight in itertools.product(
            range(1 << len(checks)), range(num_columns + 1)
        ):
            best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
            packed_syndrome = _pack_syndromes([syndrome], len(checks))[0]
            best_weight, parity = _coset_leader_search._get_syndrome_word(
                packed_syndrome, best, tables.inverse, tables.x_mask, 1
            )
            if best_weight <= max_weight:
                continue
            table = _coset_leader_search._get_planning_table(tables.planning, parity)
            stop_layer, stop_num_completed, stop_bound = _coset_leader_search._get_stopping_step(
                table, max_weight
            )
            lower_bound, weighed, _ = _coset_leader_search._search_from_syndrome_word(
                packed_syndrome,
                max_weight,
                stop_layer,
                stop_num_completed,
                stop_bound,
                best,
                best_weight,
                tables.generators,
                tables.pivots,
                tables.x_mask,
                None,
                1,
                parity if len(tables.planning) == 2 else -1,
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
                binomials = certificates_module.get_capped_binomials(num_rows)
                plan = certificates_module.plan_steps(
                    table[:-1], max_weight, 0, num_rows, instance.ranks, binomials, True
                )
                assert plan == (stop_layer, stop_num_completed, weighed)


def _assert_searches_by_tails_match_searches_of_words(
    generators: npt.NDArray[np.uint8],
    inverse: npt.NDArray[np.uint8],
    scale: int,
    block_sizes: Sequence[int],
    max_weights: Sequence[int],
    syndromes: Sequence[int],
    minima: npt.NDArray[np.int_] | None = None,
    *,
    block_bits: int | None = None,
) -> int:
    """Check searches by tails against searches of whole words, without or with completion tables.

    The family holds a set of the embedded generators reduced on each block of consecutive columns,
    whose sizes cycle through ``block_sizes``, and a set of rank zero with every generator.  The
    blocks are disjoint, so the certificate of the ranks of the sets bounds the words of each coset.
    Every column counts as the X part.  The lower bound of the search of whole words agrees with the
    minimum of the coset in ``minima``, if given.  Without ``block_bits``, for each syndrome and
    maximum weight, the two searches from the word of the syndrome give the same lower bound, word,
    and number of words weighed.  For each maximum weight, the two batch loops find the same
    syndromes, in turn, with the same bounds and words.

    With ``block_bits``, every set of at least two rows has completion tables on blocks of that many
    columns, and the searches by tails decide as the searches of whole words, and give the same word
    to a coset with a word within the maximum weight.  To another coset, they give the same lower
    bound, or the least weight beyond the maximum weight that a word of the coset can have if they
    skipped any word, and a word of at least that weight.  This least weight is one more than the
    maximum weight, or two more if the generators are even and the words of the coset have the
    parity of the maximum weight.  They weigh every word that the search of whole words weighs, but
    the words that they skip, which ``_count_skipped_supports`` counts.  The batch loops with tables
    find the same syndromes, with the bounds and words of single queries with tables.

    Returns:
        The number of supports that the searches by tails skipped.
    """
    num_rows, num_columns = generators.shape
    sets, pivots, start = [], [], 0
    for size in itertools.cycle(block_sizes):
        if start >= num_columns:
            break
        block = np.arange(start, min(start + size, num_columns))
        reduced, _, block_pivots = row_reduce_binary(generators, columns=block)
        sets.append(reduced)
        pivots.append(np.pad(block_pivots, (0, num_rows - len(block_pivots)), constant_values=-1))
        start += size
    sets.append(generators)
    pivots.append(np.full(num_rows, -1))
    instance = affine_instance.AffineInstance(
        generators=np.stack([pack_bits(rows) for rows in sets]),
        labels=np.zeros((len(sets), num_rows, 0), dtype=np.uint64),
        widths=np.ones(len(sets), dtype=np.int64),
        pivots=np.array(pivots, dtype=np.int64),
        certificate=certificates_module.Certificate(
            num_rows, tuple(int(np.sum(row >= 0)) for row in pivots)
        ),
        weight_factor=2 if scale == 2 else 1,
    )
    tables = _coset_leader_search._get_search_tables(instance, inverse, num_columns)
    tails = _coset_leader_search._get_coset_tails(
        instance,
        list(range(len(sets))) if block_bits else [],
        None,
        inverse=inverse,
        block_bits=block_bits or 8,
        max_table_bytes=MAX_VALUE,
    )
    even = scale == 1 and not np.any(generators.sum(axis=1) % 2)
    packed_syndromes = _pack_syndromes(syndromes, len(inverse))
    total_skipped = 0
    for max_weight in max_weights:
        answers = {}
        for syndrome, packed_syndrome in zip(syndromes, packed_syndromes, strict=True):
            words = np.empty((2, tables.inverse.shape[1]), dtype=np.uint64)
            best_weight, parity = _coset_leader_search._get_syndrome_word(
                packed_syndrome, words[0], tables.inverse, tables.x_mask, scale
            )
            if best_weight <= max_weight:
                continue
            words[1] = words[0]
            word = unpack_bits(words[:1], num_columns)[0]
            coset_parity = int(word.sum()) % 2
            stop_layer, stop_num_completed, stop_bound = _coset_leader_search._get_stopping_step(
                _coset_leader_search._get_planning_table(tables.planning, parity), max_weight
            )
            *answer, _ = _coset_leader_search._search_from_syndrome_word(
                packed_syndrome,
                max_weight,
                stop_layer,
                stop_num_completed,
                stop_bound,
                words[0],
                best_weight,
                tables.generators,
                tables.pivots,
                tables.x_mask,
                None,
                scale,
                parity if len(tables.planning) == 2 else -1,
            )
            *answer_by_tails, skipped = _coset_leader_search._search_from_syndrome_word(
                packed_syndrome,
                max_weight,
                stop_layer,
                stop_num_completed,
                stop_bound,
                words[1],
                best_weight,
                tables.generators,
                tables.pivots,
                tables.x_mask,
                tails,
                scale,
                parity if len(tables.planning) == 2 else -1,
            )
            if minima is not None:
                assert (answer[0] <= max_weight) == (minima[syndrome] <= max_weight)
                assert answer[0] <= minima[syndrome]
            answers[syndrome] = answer_by_tails[0], words[1]
            total_skipped += skipped
            if not block_bits:
                assert answer_by_tails == list(answer) and not skipped
                np.testing.assert_array_equal(words[0], words[1])
            elif answer[0] <= max_weight:
                assert answer_by_tails[0] == 0
                np.testing.assert_array_equal(words[0], words[1])
            else:
                least_beyond = max_weight + 1
                if even and (least_beyond + coset_parity) % 2:
                    least_beyond += 1
                assert answer_by_tails[0] == (least_beyond if skipped else answer[0])
                assert int(np.bitwise_count(words[1]).sum()) >= scale * answer_by_tails[0]
                # The skip bound is one more than the greatest Hamming weight at most
                # ``scale * max_weight`` that a word of the coset can have.
                skip_bound = scale * max_weight + 1
                if even and (skip_bound - coset_parity) % 2 == 0:
                    skip_bound -= 1
                assert answer_by_tails[1] + skipped == answer[1]
                assert skipped == sum(
                    _count_skipped_supports(rows, set_pivots, word, skip_bound, block_bits, last)
                    for index, (rows, set_pivots) in enumerate(zip(sets, pivots, strict=True))
                    if (last := stop_layer - (index >= stop_num_completed))
                    >= num_rows - np.sum(set_pivots >= 0)
                )

        # Find each syndrome without a light enough word with both batch loops, in turn.
        start = 0
        while True:
            words = np.empty((2, tables.inverse.shape[1]), dtype=np.uint64)
            found = _coset_leader_search._find_first_not_within_max_weight(
                packed_syndromes[start:], max_weight, words[0], *tables, None, scale
            )
            found_by_tails = _coset_leader_search._find_first_not_within_max_weight(
                packed_syndromes[start:], max_weight, words[1], *tables, tails, scale
            )
            assert found_by_tails[0] == found[0]
            if start + found[0] == len(packed_syndromes):
                break
            if block_bits:
                bound, word = answers[syndromes[start + found[0]]]
                assert found_by_tails[1] == bound
                np.testing.assert_array_equal(words[1], word)
            else:
                assert found_by_tails == found
                np.testing.assert_array_equal(words[0], words[1])
            start += found[0] + 1
    return total_skipped


def _count_skipped_supports(
    rows: npt.NDArray[np.uint8],
    pivots: npt.NDArray[np.int_],
    word: npt.NDArray[np.uint8],
    skip_bound: int,
    block_bits: int,
    last_layer: int,
) -> int:
    """Count the supports that tables skip in the layers from two to ``last_layer`` of a set.

    ``rows`` are the rows of the set, ``pivots`` their pivot columns, padded with -1 beyond the
    rank, and ``word`` the word of a syndrome, which the rows of its pivot ones reduce to the
    offset.  A support of a layer is skipped if the pivot ones of the rows of its prefix below the
    rank, the fewest pivot ones of a pair of rows from its row ``s`` on, and the least weight of the
    tail of such a completion on each block of columns add up to at least ``skip_bound``.  The tails
    hold the columns other than the pivots, and the blocks cover them up to the last on which a row
    has a one.  A set whose tails have no ones has no tables, and skips nothing.
    """
    rank = int(np.sum(pivots >= 0))
    offset = (word + rows[[row for row in range(rank) if word[pivots[row]]]].sum(axis=0)) % 2
    columns = np.delete(np.arange(rows.shape[1]), pivots[:rank])
    tails, offset_tail = rows[:, columns], offset[columns]
    num_bits = int(np.flatnonzero(tails.any(axis=0)).max(initial=-1)) + 1
    tails, offset_tail = tails[:, :num_bits], offset_tail[:num_bits]
    skipped = 0
    for layer in range(2, last_layer + 1):
        for support in itertools.combinations(range(len(rows)), layer):
            prefix = (offset_tail + tails[list(support[:-2])].sum(axis=0)) % 2
            pairs = list(itertools.combinations(range(support[-2], len(rows)), 2))
            bound = sum(row < rank for row in support[:-2])
            bound += min(sum(row < rank for row in pair) for pair in pairs)
            bound += sum(
                min(
                    int(((prefix + tails[i] + tails[j]) % 2)[start : start + block_bits].sum())
                    for i, j in pairs
                )
                for start in range(0, num_bits, block_bits)
            )
            if num_bits and bound >= skip_bound:
                skipped += 1
    return skipped


@pytest.mark.parametrize("block_bits", [None, 4])
def test_searches_by_tails_answer_as_searches_of_whole_words(
    block_bits: int | None, pytestconfig: pytest.Config
) -> None:
    """Coset searches by tails decide as searches of whole words, and match them without tables.

    The searches by tails weigh a word by its rows below the rank of its set and its tail, which
    holds every column but the pivot columns of the set.  Without completion tables, they visit the
    same words, and keep the same lightest word, as the searches of whole words (see
    ``_assert_searches_by_tails_match_searches_of_words``).  With tables on blocks of four columns,
    they skip words beyond the maximum weight, which changes only the answers to cosets without a
    word within it.  Blocks of as many columns as the generators have rows alternate with blocks of
    one column fewer, whose sets are rank deficient, so that the tails of the sets differ in their
    numbers of columns.  The random codes have a column of zeros, which a coset can still have ones
    on.  With tables, each is also made even by a column of the parities of its rows, so that the
    searches round their bounds to the parity of each coset.  A rank-deficient code made even in the
    same way is searched too.  The searches of every syndrome of these codes, at every maximum
    weight up to past their length, agree with the minima of the cosets.  A wide code begins with an
    information set, whose tails fit in one machine word, while the tails of its rank-deficient sets
    take two, and a sample of its syndromes is searched.  The words of the Pauli errors of a graph
    state that is not CSS are weighed by their physical weight, and with tables, its family has sets
    of ranks two and one.  With tables, the searches of the even rank-deficient code, the wide code,
    and the graph state skip words; those of small random codes need not.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The arguments of each check, and whether its searches with tables must skip words.
    searches: list[tuple[tuple[Any, ...], bool]] = []
    for _ in range(3):
        num_rows, num_columns = int(rng.integers(3, 5)), int(rng.integers(6, 10))
        generators = get_random_full_rank_matrix(rng, num_rows, num_columns)
        code = np.insert(generators, int(rng.integers(num_columns + 1)), 0, axis=1)
        codes = [code, np.hstack([code, code.sum(axis=1, keepdims=True) % 2]).astype(np.uint8)]
        for code in codes if block_bits else codes[:1]:
            checks = get_kernel(code)
            searches.append(
                (
                    (
                        code,
                        _coset_leader_search._get_inverse(checks),
                        1,
                        (num_rows, num_rows - 1),
                        range(-1, code.shape[1] + 2),
                        range(1 << len(checks)),
                        _classical_minima(checks),
                    ),
                    False,
                )
            )
    if block_bits:
        parities = (_RANK_DEFICIENT_CODE.sum(axis=1, keepdims=True) % 2).astype(np.uint8)
        even = np.hstack([_RANK_DEFICIENT_CODE, parities])
        checks = get_kernel(even)
        searches.append(
            (
                (
                    even,
                    _coset_leader_search._get_inverse(checks),
                    1,
                    (5, 4),
                    range(-1, even.shape[1] + 2),
                    range(1 << len(checks)),
                    _classical_minima(checks),
                ),
                True,
            )
        )

    wide = np.hstack([np.eye(4, dtype=np.uint8), rng.integers(0, 2, (4, 64), dtype=np.uint8)])
    checks = get_kernel(wide)
    syndromes = [int.from_bytes(rng.bytes(9), "little") % (1 << len(checks)) for _ in range(16)]
    searches.append(
        (
            (
                wide,
                _coset_leader_search._get_inverse(checks),
                1,
                (4, 3),
                range(-1, 40, 4),
                syndromes,
            ),
            True,
        )
    )

    matrix = _graph_state(4, [(0, 1), (1, 2)], phase=True)
    checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
    searches.append(
        (
            (
                symplectic_to_hamming(matrix),
                symplectic_to_hamming(_coset_leader_search._get_inverse(checks)),
                2,
                (2, 1) if block_bits else (4, 3),
                range(-1, 6),
                range(16),
                _css_minima(matrix, css_weight=False),
            ),
            True,
        )
    )
    for arguments, must_skip in searches:
        skipped = _assert_searches_by_tails_match_searches_of_words(
            *arguments, block_bits=block_bits
        )
        assert skipped or not (block_bits and must_skip)


def test_queries_of_routes_by_tails_weigh_the_tails_of_their_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single queries and batches of routes by tails weigh words by the tails of their record.

    At the threshold zero, every route of the classical search of the repetition code of length
    seven is by tails.  A record whose tails are all zero makes every offset weigh nothing, so the
    queries at maximum weight two find a light enough word in every coset, even in the cosets whose
    words weigh at least three, which a search of whole words would reject.
    """
    get_coset_tails = _coset_leader_search._get_coset_tails

    def get_zero_tails(*args: Any, **kwargs: Any) -> _coset_leader_search._CosetTails:
        """The record of the tails of a family, with every tail zero."""
        record = get_coset_tails(*args, **kwargs)
        return record._replace(
            tails=np.zeros_like(record.tails), inverse_tails=np.zeros_like(record.inverse_tails)
        )

    monkeypatch.setattr(_coset_leader_search, "_get_coset_tails", get_zero_tails)
    generators = np.ones((1, 7), dtype=np.uint8)
    checks = get_kernel(generators)
    tuning = _coset_leader_search._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    search = _coset_leader_search._get_classical_search(
        generators, checks, tuning._replace(overlapping_minimum_work=0)
    )
    syndromes = np.flatnonzero(_classical_minima(checks) > 2).tolist()
    assert syndromes
    for syndrome in syndromes:
        bound = search.get_min_weight_bound(syndrome, 2)
        assert (bound.within_max_weight, bound.lower_bound) == (True, 2)
    packed_syndromes = _pack_syndromes(syndromes, len(checks))
    assert search.find_first_not_within_max_weight(packed_syndromes, 2) == (len(syndromes), None)


def test_queries_of_routes_by_tails_skip_completions_by_the_tables_of_their_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single queries and batches of routes by tails skip completions by the tables of their record.

    At the threshold zero, every route of the classical search of a random code is by tails.  A
    record whose tables give every completion the bound 255 makes each query skip every completion
    in its layers of two or more rows.  At maximum weight three, some coset then gets no word within
    that weight, although it has one.  The searches skipped words, so single queries and batches
    bound such cosets by four.  The code is not even, so four is the least weight beyond the maximum
    weight that a word of such a coset can have.
    """
    get_coset_tails = _coset_leader_search._get_coset_tails

    def get_skipping_tails(*args: Any, **kwargs: Any) -> _coset_leader_search._CosetTails:
        """The record of the tails of a family, with tables that skip every completion."""
        record = get_coset_tails(*args, **kwargs)
        return record._replace(completion_tables=np.full_like(record.completion_tables, 255))

    monkeypatch.setattr(_coset_leader_search, "_get_coset_tails", get_skipping_tails)
    checks = get_kernel(OVERLAPPING_ROUTES_CODE)
    tuning = _coset_leader_search._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    search = _coset_leader_search._get_classical_search(
        OVERLAPPING_ROUTES_CODE, checks, tuning._replace(overlapping_minimum_work=0)
    )
    minima = _classical_minima(checks)
    missed = [
        syndrome
        for syndrome in np.flatnonzero(minima <= 3).tolist()
        if not search.get_min_weight_bound(syndrome, 3).within_max_weight
    ]
    assert missed
    assert {search.get_min_weight_bound(syndrome, 3).lower_bound for syndrome in missed} == {4}
    _, bound = search.find_first_not_within_max_weight(_pack_syndromes(missed, len(checks)), 3)
    assert bound is not None and bound.lower_bound == 4


def test_records_of_tails_add_the_tables_of_the_sets_that_fit() -> None:
    """A record of tails gains the tables of the given sets that fit its limit, once for each set.

    The rank-deficient code has disjoint information sets of ranks five and four, whose tails of
    five and six columns have tables of 128 and 256 bytes on blocks of eight columns.  Under a limit
    of 128 bytes, only the first set gets tables, in the layout of the record, and a second request
    for both sets keeps the record.  Under a larger limit, the second set gets its tables too in a
    new record, and the first set keeps its own, while the earlier record stays as it was.  A record
    built under that limit gets the tables of both sets at once.  Each record holds the tables of
    each set once.
    """
    checks = get_kernel(_RANK_DEFICIENT_CODE)
    instance = affine_instance.get_coset_instance(
        _RANK_DEFICIENT_CODE, random_column_orders=0, weight_factor=1
    )
    assert instance.ranks.tolist() == [5, 4]
    inverse = _coset_leader_search._get_inverse(checks)

    def get_record(
        sets: list[int], record: _coset_leader_search._CosetTails | None, max_table_bytes: int
    ) -> _coset_leader_search._CosetTails:
        """The record of the tails of the family, with the tables of the sets that fit."""
        return _coset_leader_search._get_coset_tails(
            instance, sets, record, inverse=inverse, block_bits=8, max_table_bytes=max_table_bytes
        )

    def get_set_tables(record: _coset_leader_search._CosetTails, set_index: int) -> list[int]:
        """The tables of a set, as its layout places them in the record."""
        start, num_blocks = record.layout[set_index, :2]
        offsets = record.layout[set_index, 2 : num_blocks + 3]
        size = (len(_RANK_DEFICIENT_CODE) - 1) * offsets[-1] if num_blocks else 0
        assert start + size <= len(record.completion_tables)
        return record.completion_tables[start : start + size].tolist()

    expected = [
        completion_bounds.CompletionTables.build(tails, 8)
        for tails in get_record([], None, 0).tails
    ]
    first = get_record([0, 1], None, 128)
    assert get_set_tables(first, 0) == expected[0].tables.tolist()
    assert get_set_tables(first, 1) == []
    assert get_record([0, 1], first, 128) is first
    for both in (get_record([1], first, 256), get_record([0, 1], None, 256)):
        assert [get_set_tables(both, index) for index in range(2)] == [
            tables.tables.tolist() for tables in expected
        ]
        assert len(both.completion_tables) == sum(len(tables.tables) for tables in expected)
    assert get_set_tables(first, 1) == []


def _coset_minima(
    generators: npt.NDArray[np.uint8], checks: npt.NDArray[np.uint8], syndromes: Sequence[int]
) -> npt.NDArray[np.int_]:
    """The least weight of the words of the coset of each syndrome, from every word of the coset.

    A word with the syndrome sums rows of a right inverse of the checks, from the module under test,
    so the helper checks the syndromes of these words.  The coset adds to such a word every word of
    the code of ``generators``.
    """
    code = np.array(list(itertools.product([0, 1], repeat=len(generators)))) @ generators % 2
    bits = (np.array(syndromes)[:, None] >> np.arange(len(checks))) & 1
    words = bits @ _coset_leader_search._get_inverse(checks) % 2
    assert np.array_equal(_classical_syndrome(checks, words), syndromes)
    return np.array([((word + code) % 2).sum(axis=1).min() for word in words])


def _assert_min_weights_match_brute_force(
    search: _coset_leader_search._CosetSearch,
    syndromes: Sequence[int],
    minima: npt.NDArray[np.int_],
    num_syndrome_bits: int,
    rng: np.random.Generator,
) -> None:
    """Check the minimum weights that a search finds for cosets against their least weights.

    One batch holds the syndromes, in random order, and ``minima`` holds the least weight of the
    coset of each.  At each maximum weight from zero to one past the largest least weight, the
    floors of the batch are zero, then random up to the maximum weight, and then that weight.  A
    coset whose least weight is at most its floor gets a weight from its least weight to its floor.
    Any other coset gets its least weight, or one more than the maximum weight if that is smaller.
    """
    order = rng.permutation(len(syndromes))
    packed_syndromes = _pack_syndromes([syndromes[index] for index in order], num_syndrome_bits)
    least = minima[order]
    for max_weight in range(int(least.max()) + 2):
        random_floors = rng.integers(0, max_weight + 1, len(least))
        for floors in (np.zeros_like(least), random_floors, np.full_like(least, max_weight)):
            weights = search.get_min_weights(packed_syndromes, max_weight, floors.astype(np.int64))
            exact = least > floors
            assert np.array_equal(weights[exact], np.minimum(least, max_weight + 1)[exact])
            assert np.all(exact | ((least <= weights) & (weights <= floors)))


@pytest.mark.parametrize(
    "tuning",
    [
        {},
        {"overlapping_minimum_work": 0},
        {"overlapping_minimum_work": 0, "completion_bound_max_table_bytes": 0},
    ],
    ids=["disjoint", "tables", "no-tables"],
)
def test_min_weights_of_cosets_match_brute_force(
    tuning: dict[str, int], pytestconfig: pytest.Config, tune: Tune
) -> None:
    """Searches for the minimum weights of cosets agree with the least weights of their words.

    The codes are the rank-deficient code, a code whose searches skip words by tables, and a random
    code, each also made even by a column of the parities of its rows.  The searches of the even
    codes round to the parity of each coset, and their incumbents are one below the Hamming weight
    of their best words.  The cosets have both parities.  Three cosets of a longer code are
    searched too.  Their searches by tails stop early within steps.  The searches take the disjoint
    family of each code, and at the threshold zero other families by tails, with completion tables
    and without them (see ``_assert_min_weights_match_brute_force``).  The least weights come from
    every word of each coset.
    """
    tune(**tuning)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    tuning_fields = _coset_leader_search._CosetTuning.from_tuning(
        get_tuning(), paulis_by_css_weight=False
    )
    num_rows, num_columns = int(rng.integers(3, 6)), int(rng.integers(8, 11))
    random_code = get_random_full_rank_matrix(rng, num_rows, num_columns)
    for code in (_RANK_DEFICIENT_CODE, _SKIPPING_CODE, random_code):
        even_code = np.hstack([code, code.sum(axis=1, keepdims=True) % 2]).astype(np.uint8)
        for generators in (code, even_code):
            checks = get_kernel(generators)
            search = _coset_leader_search._get_classical_search(generators, checks, tuning_fields)
            syndromes = range(1 << len(checks))
            minima = _coset_minima(generators, checks, syndromes)
            _assert_min_weights_match_brute_force(search, syndromes, minima, len(checks), rng)
    checks = get_kernel(_HEAVY_COSETS_CODE)
    search = _coset_leader_search._get_classical_search(_HEAVY_COSETS_CODE, checks, tuning_fields)
    early_stop_syndromes = [1126, 4578, 6129]
    minima = _coset_minima(_HEAVY_COSETS_CODE, checks, early_stop_syndromes)
    _assert_min_weights_match_brute_force(search, early_stop_syndromes, minima, len(checks), rng)


@pytest.mark.parametrize(
    "tuning",
    [
        {},
        {"overlapping_minimum_work": 0},
        {"overlapping_minimum_work": 0, "completion_bound_max_table_bytes": 0},
    ],
    ids=["disjoint", "tables", "no-tables"],
)
def test_min_weights_match_exhaustive_minima(
    tuning: dict[str, int], pytestconfig: pytest.Config, tune: Tune
) -> None:
    """The minimum weights of syndrome classes agree with exhaustive minima, by either method.

    Graph states with phases on two to five qubits and on eight qubits are not CSS, so one search
    covers their Pauli errors, by either weight.  By physical weight, the embedded words of the
    errors have even Hamming weights, so once a BZ search finds a word, its incumbent is one below
    the Hamming weight of its best word.  The searches of the state on eight qubits also stop early
    within steps and skip words by tables.  The stabilizers of the state on a cycle of five qubits
    have even Hamming weights as words [x|z] too, but CSS weights have no parity, so its searches by
    CSS weight do not lower their incumbents by one.  Random CSS states on two to five qubits, the
    logical zero of the Steane code, whose code of X parts is even and that of Z parts is not, and a
    GHZ block beside its conjugate by Hadamard gates, whose codes are both even, are weighed by CSS
    weight, which splits their searches.  The search of the Z part of a class then ends at a word
    within the weight of its X part.  Some classes of the GHZ blocks have an X part of weight two
    and a Z part of weight three, which a connected-support search of the Z part weighs exactly only
    from a weight limit of at most two.  The Steane state is also weighed by physical weight.  An
    oracle answers one batch of every syndrome, in random order, at every maximum weight from minus
    two, below every weight, to one past the number of qubits.  The BZ searches take the disjoint
    families, and at the threshold zero other families by tails, with completion tables and without
    them.  The connected-support method, which the tuning of these families does not configure,
    answers under the default tuning, but not by CSS weight on states that are not CSS.
    """
    tune(**tuning)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    cases = []
    for num_qubits in range(2, 6):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        graph_state = _graph_state(num_qubits, edges, phase=True)
        css_state = _random_css_state(num_qubits, int(rng.integers(num_qubits + 1)), rng)
        cases += [(graph_state, False, True), (graph_state, True, False), (css_state, True, True)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5)]
    edges += [(2, 6), (2, 7), (3, 4), (3, 5), (3, 7), (4, 5), (5, 6), (5, 7)]
    graph_state = _graph_state(8, edges, phase=True)
    cases += [(graph_state, False, True), (graph_state, True, False)]
    cycle_state = _graph_state(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)], phase=True)
    cases.append((cycle_state, True, False))
    steane_zero = ("XXXX___", "XX__XX_", "X_X_X_X", "ZZZZ___", "ZZ__ZZ_", "Z_Z_Z_Z", "ZZZZZZZ")
    cases += [(_product_state(steane_zero), css_weight, True) for css_weight in (False, True)]
    ghz_block = ("XXXX____", "ZZ______", "_ZZ_____", "__ZZ____")
    conjugate_block = ("____ZZZZ", "____XX__", "_____XX_", "______XX")
    cases.append((_product_state(ghz_block + conjugate_block), True, True))
    for matrix, css_weight, by_support in cases:
        num_qubits = len(matrix)
        minima = _css_minima(matrix, css_weight)
        syndromes = rng.permutation(1 << num_qubits)
        packed_syndromes = _pack_syndromes(syndromes.tolist(), num_qubits)
        methods: list[CosetSearchMethod] = ["brouwer_zimmermann"]
        if by_support and not tuning:
            methods.append("connected_support")
        for method in methods:
            oracle = CosetLeaderSearch(matrix, css_weight=css_weight, method=method)
            for max_weight in range(-2, num_qubits + 2):
                weights = oracle._get_min_weights(packed_syndromes, max_weight)
                assert np.array_equal(weights, np.minimum(minima[syndromes], max_weight + 1))


@pytest.mark.parametrize("tuning", [{}, {"overlapping_minimum_work": 0}], ids=["words", "tails"])
def test_min_weight_searches_above_max_weight_examine_as_queries(
    tuning: dict[str, int], tune: Tune
) -> None:
    """A search for the minimum weight of a coset without a word within a weight works as a query.

    Until it finds a word within ``max_weight``, the search keeps the skip bound of ``max_weight``
    as its incumbent, at which a query at ``max_weight`` skips completions too.  It stops at the
    stopping step of that query, so it weighs and skips the same words.  A code and its even version
    are searched at every syndrome, a longer code at three syndromes of minimum weight six, and the
    Pauli errors of a graph state by physical weight at every syndrome.  They are searched by whole
    words, and at the threshold zero by tails, with tables, which skip words of each of the three
    codes but need not skip any of the graph state.  The minimum weights come from every word of
    each coset.
    """
    tune(**tuning)
    tuning_fields = _coset_leader_search._CosetTuning.from_tuning(
        get_tuning(), paulis_by_css_weight=False
    )
    cases = []
    parities = (_SKIPPING_CODE.sum(axis=1, keepdims=True) % 2).astype(np.uint8)
    for generators, chosen_syndromes in [
        (_SKIPPING_CODE, None),
        (np.hstack([_SKIPPING_CODE, parities]), None),
        (_HEAVY_COSETS_CODE, [3571, 3722, 7870]),
    ]:
        checks = get_kernel(generators)
        syndromes = chosen_syndromes or list(range(1 << len(checks)))
        search = _coset_leader_search._get_classical_search(generators, checks, tuning_fields)
        minima = _coset_minima(generators, checks, syndromes)
        cases.append((search, len(checks), syndromes, minima, True))
    matrix = _ROUTED_PHYSICAL_STATES[1]
    checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
    search = _coset_leader_search._get_pauli_search(
        _coset_leader_search._StabilizerKey(matrix, checks), False, tuning_fields
    )
    syndromes = list(range(1 << len(matrix)))
    cases.append((search, len(matrix), syndromes, _css_minima(matrix, css_weight=False), False))

    for search, num_syndrome_bits, syndromes, minima, must_skip in cases:
        skipped = 0
        for max_weight in range(2, int(max(minima))):
            tables, tails = search._routes.get_route(max_weight)
            for syndrome, minimum in zip(syndromes, minima, strict=True):
                if minimum <= max_weight:
                    continue
                packed_syndrome = _pack_syndromes([syndrome], num_syndrome_bits)[0]
                best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
                best_weight, parity = _coset_leader_search._get_syndrome_word(
                    packed_syndrome, best, tables.inverse, tables.x_mask, search.scale
                )
                stop_layer, stop_num_completed, stop_bound = (
                    _coset_leader_search._get_stopping_step(
                        _coset_leader_search._get_planning_table(tables.planning, parity),
                        max_weight,
                    )
                )
                _, *expected = _coset_leader_search._search_from_syndrome_word(
                    packed_syndrome,
                    max_weight,
                    stop_layer,
                    stop_num_completed,
                    stop_bound,
                    best,
                    best_weight,
                    tables.generators,
                    tables.pivots,
                    tables.x_mask,
                    tails,
                    search.scale,
                    parity if len(tables.planning) == 2 else -1,
                )
                _, *found = _coset_leader_search._search_coset_for_min_weight(
                    packed_syndrome, max_weight, 0, best, *tables, tails, search.scale
                )
                skipped += expected[1]
                assert found == expected
        assert skipped or not (tuning and must_skip)


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


def test_connected_support_answers_match_exhaustive_minima(pytestconfig: pytest.Config) -> None:
    """The connected-support engine answers every query exactly, by either measure of weight.

    Graph states with phases are not CSS, and are weighed by physical weight.  CSS states with
    no pure-X stabilizer, with only pure-X stabilizers, and with both are weighed by both measures,
    and the CSS weight splits their searches.  A syndrome whose lightest errors weigh at most one
    gets their weight as its bound.  Otherwise a query within ``max_weight`` gets 2, and one with no
    error within ``max_weight`` gets ``max(2, max_weight + 1)``.  The queries of batches, which try
    increasing weight limits, instead get the exact minimum and a lightest error when they are
    within ``max_weight``.  These exact bounds imply the inequalities that ``_assert_exact_answers``
    checks.  A search of Pauli errors from a first weight limit above ``max_weight`` answers as a
    single query.
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
                minimum = int(minima[syndrome])
                single = oracle.get_min_weight_bound(syndrome, max_weight)
                batched = oracle._get_batch_support_bound(syndrome, max_weight)
                if not css_weight:
                    search = oracle._support_searches[0]
                    late = search.get_min_weight_bound(
                        syndrome, max_weight, first_limit=max_weight + 1
                    )
                    assert late == single
                for bound, exact in [(single, False), (batched, True)]:
                    weight = _pauli_weight(bound.representative, num_qubits, css_weight)
                    assert _get_syndrome(matrix, bound.representative) == syndrome
                    assert bound.within_max_weight == (minimum <= max_weight)
                    if not bound.within_max_weight:
                        expected_bound = minimum if minimum < 2 else max(2, max_weight + 1)
                        assert bound.lower_bound == expected_bound
                    elif exact or minimum < 2:
                        assert bound.lower_bound == weight == minimum
                    else:
                        assert bound.lower_bound == 2 and weight <= max_weight


def test_connected_support_batches_read_syndromes_of_several_words(
    pytestconfig: pytest.Config, tune: Tune
) -> None:
    """On 17 phased GHZ blocks, batches of the connected-support engine agree with block minima.

    The 68 syndrome bits span two words, and the physical weight takes one search of the Pauli
    errors.  The batch holds the syndromes of 30 errors.  The first acts on one qubit in each of
    blocks 0, 8, and 16.  Block 16 has its syndrome bits in the second word, so a search that read
    only the first word would find a lighter error than the minimum of three.  The other errors each
    act on one to three random qubits, and blocks that an error does not touch add nothing to its
    minimum weight.  The oracle looks for sparser checks for one round, rather than for the many
    rounds that are slow as plain Python.  On these blocks, one round gives the default checks.
    """
    tune(max_sparsifying_rounds=1)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_blocks = 17
    generators = _block_generators(_PHASED_GHZ_BLOCK, num_blocks)
    matrix = _stabilizer_matrix(generators)
    oracle = CosetLeaderSearch(matrix, method="connected_support")
    syndromes, minima = [], []
    for error_index in range(30):
        error = stim.PauliString(4 * num_blocks)
        qubits = (
            [0, 35, 67]
            if error_index == 0
            else rng.choice(4 * num_blocks, size=rng.integers(1, 4), replace=False)
        )
        for qubit in qubits:
            error[int(qubit)] = int(rng.integers(1, 4))
        syndromes.append(sum((not error.commutes(gg)) << jj for jj, gg in enumerate(generators)))
        bits = [(syndromes[-1] >> jj) & 1 for jj in range(4 * num_blocks)]
        local = [bits[4 * index : 4 * index + 4] for index in range(num_blocks)]
        minima.append(_block_minimum(_PHASED_GHZ_BLOCK, [ss for ss in local if any(ss)], False))
    assert syndromes[0] >> 64 and minima[0] == 3
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
        rows, transform = _coset_leader_search._sparsify(matrix, homogeneous=homogeneous)
        assert np.array_equal(transform @ matrix % 2, rows)
        assert len(row_reduce_binary(transform)[2]) == len(matrix)
        weights = get_weights(rows, homogeneous)
        for row, other in itertools.permutations(range(len(rows)), 2):
            assert get_weights((rows[row] ^ rows[other])[None], homogeneous)[0] >= weights[row]


def test_rounds_of_random_reductions_make_rows_sparser() -> None:
    """Rounds reach a lightest basis, which adding rows to each other misses.

    Three independent words of the span of these Paulis on five qubits act on three qubits each, so
    they form a lightest basis, of total weight 9.  Adding rows to each other replaces the first row
    with its sum with the second, which is one of them, and then no sum of two rows is lighter than
    a row, so zero rounds give total weight 10.  Another of the three words has two Ys, so it counts
    five bits, and rows weighed by bits would miss it.
    """
    matrix = np.array(
        [
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    for max_rounds, total_weight in [(0, 10), (20, 9)]:
        rows, transform = _coset_leader_search._get_sparser_rows(
            matrix, homogeneous=False, max_rounds=max_rounds, rounds_without_improvement=20
        )
        assert np.array_equal(transform @ matrix % 2, rows)
        assert len(row_reduce_binary(transform)[2]) == len(matrix)
        assert (rows[:, :5] | rows[:, 5:]).sum() == total_weight


def test_connected_support_rejects_what_it_cannot_search() -> None:
    """Oracles reject unknown methods, and the CSS weight on non-CSS states by connected support."""
    with pytest.raises(ValueError, match="only on CSS states"):
        CosetLeaderSearch(
            _graph_state(2, [(0, 1)], phase=True), css_weight=True, method="connected_support"
        )
    method: Any = "brute_force"
    with pytest.raises(ValueError, match="Unknown coset-leader search method"):
        CosetLeaderSearch([[1, 0]], method=method)


@pytest.mark.parametrize("method", ["brouwer_zimmermann", "connected_support"])
def test_batches_reject_syndromes_out_of_range(method: CosetSearchMethod) -> None:
    """Batches of either method reject a syndrome at or above ``2**num_qubits``.

    The syndromes of the state |+0> are below four.  A row with bit two set, or with a second
    machine word that is not zero, is out of range, in a batch of queries and in one of minimum
    weights.  A row with a zero second word is in range: the syndrome three needs Z on the first
    qubit and X on the second.
    """
    oracle = CosetLeaderSearch(_product_state(["X_", "_Z"]), method=method)
    for row in ([4], [3, 1]):
        syndromes = np.array([row], dtype=np.uint64)
        with pytest.raises(ValueError, match="out of range"):
            oracle.find_first_not_within_max_weight(syndromes, 1)
        with pytest.raises(ValueError, match="out of range"):
            oracle._get_min_weights(syndromes, 1)
    assert oracle._get_min_weights(np.array([[3, 0]], dtype=np.uint64), 1).tolist() == [2]
