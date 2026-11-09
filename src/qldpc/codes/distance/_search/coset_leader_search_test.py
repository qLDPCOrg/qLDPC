# SPDX-License-Identifier: Apache-2.0

"""Tests for exact minimum-weight bounds on syndrome classes of Pauli errors."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import math
from qldpc._binary import get_kernel, row_reduce_binary

from ...conftest import Tune
from .._tuning import DEFAULT_TUNING
from . import coset_leader_search, information_sets
from .coset_leader_search import CosetLeaderSearch


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


def _css_minima(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.int_]:
    """The smallest CSS weight of an error with each packed syndrome, from all ``4**n`` Paulis."""
    n = len(matrix)
    errors = (np.arange(1 << (2 * n))[:, None] >> np.arange(2 * n)) & 1
    checks = math.symplectic_conjugate(matrix.astype(int))
    syndromes = (errors @ checks.T % 2) @ (1 << np.arange(n))
    weights = np.maximum(errors[:, :n].sum(axis=1), errors[:, n:].sum(axis=1))
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
def test_layers_reach_their_last_support() -> None:
    """A layer visits every support, including the last one.

    The rows are unit words and the offset is the sum of the last ``layer`` rows, so that only the
    last support of the layer gives the zero word.
    """
    for num_rows in range(1, 7):
        rows = (np.uint64(1) << np.arange(num_rows, dtype=np.uint64)).reshape(1, num_rows, 1)
        for layer in range(1, num_rows + 1):
            offset = np.bitwise_xor.reduce(rows[0, num_rows - layer :])
            best = np.empty(1, dtype=np.uint64)
            weight = coset_leader_search._visit_layer(
                offset,
                rows,
                np.ones(1, dtype=np.int64),
                np.empty(layer, dtype=np.int64),
                np.empty((layer, 1, 1), dtype=np.uint64),
                0,
                best,
                np.int64(num_rows),
                np.zeros(1, dtype=np.uint64),
                False,
            )
            assert weight == 0 and best[0] == 0


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

    The cached searches of a CSS state are keyed by those orders too, so an oracle built under other
    orders builds its own, and one built under the same orders reuses them.
    """
    orders: list[int] = []
    get_disjoint_sets = information_sets._get_disjoint_sets

    def record(*args: Any) -> Any:
        orders.append(args[2])
        return get_disjoint_sets(*args)

    monkeypatch.setattr(information_sets, "_get_disjoint_sets", record)
    eye, zeros = np.eye(3, dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)
    zero_plus = np.block([[zeros, eye, zeros, zeros], [zeros, zeros, eye, zeros]])
    coset_leader_search._get_css_searches.cache_clear()
    CosetLeaderSearch(zero_plus, css_weight=True)
    tune(disjoint_random_column_orders=3)
    CosetLeaderSearch(zero_plus, css_weight=True)
    CosetLeaderSearch(zero_plus, css_weight=True)
    info = coset_leader_search._get_css_searches.cache_info()
    assert (info.misses, info.hits) == (2, 1)
    # A state that is not CSS, as with phases on two qubits, gets a search of Pauli errors.
    CosetLeaderSearch(_graph_state(2, [(0, 1)], phase=True), css_weight=True)
    default = DEFAULT_TUNING.disjoint_random_column_orders
    assert orders == [default, default, 3, 3, 3]


def test_invalid_arguments_are_rejected() -> None:
    """Malformed states and queries raise, for either measure of weight."""
    # Not 2n columns, dependent rows, anticommuting rows, and a syndrome out of range.
    for css_weight in (False, True):
        for stabilizers in ([[1]], [[0, 0]], [[1, 0, 0, 0], [0, 0, 1, 0]]):
            with pytest.raises(ValueError):
                CosetLeaderSearch(stabilizers, css_weight=css_weight)
        with pytest.raises(ValueError):
            CosetLeaderSearch([[1, 0]], css_weight=css_weight).get_min_weight_bound(2, 1)
