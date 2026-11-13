# SPDX-License-Identifier: Apache-2.0

"""Tests for exact minimum-weight bounds on syndrome classes of Pauli errors."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import Any, cast, get_args

import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse
import stim

from qldpc._binary import unpack_bits

from ..conftest import Tune
from .conftest import (
    ROUTED_CSS_WEIGHT_STATE,
    ROUTED_PHYSICAL_STATES,
    assert_exact_answers,
    get_exhaustive_minima,
    get_graph_state,
    get_pauli_weight,
    get_random_css_state,
    get_stabilizer_matrix,
    get_syndrome,
    scramble,
    split_syndromes_into_words,
)
from .core import DistanceMethod
from .coset_leaders import CosetLeaderSearch
from .min_weight_bounds import MinWeightBound


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


def _ghz_blocks(num_blocks: int) -> tuple[npt.NDArray[np.uint8], int]:
    """Stabilizers of independent GHZ blocks, and the syndrome of Z on each block's last qubit.

    That syndrome flips two X-type stabilizers in every block, which needs a Z component on the
    block, so every equivalent error has Z weight at least ``num_blocks``.  Weight ``num_blocks`` is
    achieved, so it is the exact minimum by either measure of weight.
    """
    generators = _block_generators(("ZZZ", "X_X", "_XX"), num_blocks)
    witness = stim.PauliString("__Z" * num_blocks)
    syndrome = sum((not witness.commutes(gg)) << jj for jj, gg in enumerate(generators))
    return get_stabilizer_matrix(generators), syndrome


def _block_generators(block: Sequence[str], num_blocks: int) -> list[stim.PauliString]:
    """Generators of independent copies of a block state, given the generators of one block."""
    size = len(block[0])
    return [
        stim.PauliString("_" * (size * index) + local + "_" * (size * (num_blocks - index - 1)))
        for index in range(num_blocks)
        for local in block
    ]


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
        assert get_syndrome(matrix, bound.representative) == syndrome
        if max_weight >= 2:
            assert bound == oracle.get_min_weight_bound(syndrome, max_weight)


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
        matrix = get_graph_state(num_qubits, edges, phase=num_qubits % 2 == 0)
        oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
        for syndrome in rng.permutation(1 << num_qubits).tolist():
            minimum = int(_class_weights(matrix, syndrome, css_weight).min())
            for max_weight in rng.permutation(num_qubits + 1).tolist():
                bound = oracle.get_min_weight_bound(syndrome, max_weight)
                assert bound.within_max_weight == (minimum <= max_weight)
                assert get_syndrome(matrix, bound.representative) == syndrome
                assert bound.lower_bound <= minimum
                if bound.within_max_weight:
                    assert (
                        get_pauli_weight(bound.representative, num_qubits, css_weight) <= max_weight
                    )
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
        matrix = get_graph_state(num_qubits, edges, phase=num_qubits % 2 == 0)
        oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
        syndromes = rng.permutation(1 << num_qubits).tolist()
        minima = np.array([_class_weights(matrix, ss, css_weight).min() for ss in syndromes])
        for max_weight in range(num_qubits + 2):
            words = split_syndromes_into_words(syndromes, num_qubits)
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
    words = split_syndromes_into_words([0b0011, 0b1111], 4)
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
    matrix = get_stabilizer_matrix(generators)
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
        words = split_syndromes_into_words(syndromes, 68)
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
    matrix = scramble(pure, rng)
    oracle = CosetLeaderSearch(matrix, css_weight=True)
    syndromes, minima = [], []
    for _ in range(20):
        x_qubits = rng.choice([*range(50), *range(60, 70)], size=rng.integers(0, 4), replace=False)
        z_qubits = rng.choice(range(120, num_qubits), size=rng.integers(0, 4), replace=False)
        pauli = sum(1 << int(qq) for qq in x_qubits) + sum(
            1 << int(qq + num_qubits) for qq in z_qubits
        )
        syndromes.append(get_syndrome(matrix, pauli))
        minima.append(max(len(x_qubits), len(z_qubits)))
    for max_weight in range(4):
        for syndrome, minimum in zip(syndromes, minima, strict=True):
            bound = oracle.get_min_weight_bound(syndrome, max_weight)
            assert bound.within_max_weight == (minimum <= max_weight)
            assert get_syndrome(matrix, bound.representative) == syndrome
        words = split_syndromes_into_words(syndromes, num_qubits)
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
    matrix = get_graph_state(4, [(0, 1), (1, 2), (2, 3)])
    assert _class_weights(matrix, 15, True).min() == 2
    oracle = CosetLeaderSearch(matrix, css_weight=True)
    bound = oracle.get_min_weight_bound(15, 1)
    assert not bound.within_max_weight and bound.lower_bound == 2
    assert get_syndrome(matrix, bound.representative) == 15
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
    matrix = get_graph_state(8, edges)
    assert _class_weights(matrix, syndrome, False).min() == minimum
    bound = CosetLeaderSearch(matrix).get_min_weight_bound(syndrome, 2)
    assert bound.within_max_weight == (minimum <= 2)
    assert bound.lower_bound <= minimum
    assert get_syndrome(matrix, bound.representative) == syndrome
    if bound.within_max_weight:
        assert get_pauli_weight(bound.representative, len(matrix), False) <= 2
    else:
        assert bound.lower_bound > 2


def test_exhausted_search_gives_the_exact_minimum() -> None:
    """On a state with stabilizers Y, Z, Z, Z, Z, the syndrome 11111 has CSS weight four.

    An error with that syndrome has X or Y on each of the last four qubits, so its X part has weight
    at least four, and Z X X X X reaches it.  The state is not CSS, so the oracle searches Pauli
    errors, and the bounds of its certificate stay below four, so only exhausting every layer proves
    the minimum.
    """
    matrix = get_stabilizer_matrix(("Y____", "_Z___", "__Z__", "___Z_", "____Z"))
    oracle = CosetLeaderSearch(matrix, css_weight=True)
    bound = oracle.get_min_weight_bound(0b11111, 3)
    assert not bound.within_max_weight and bound.lower_bound == 4
    assert oracle.get_min_weight_bound(0b11111, 4).within_max_weight


@pytest.mark.parametrize("css_weight", [False, True])
def test_wide_states_span_several_packed_words(css_weight: bool, tune: Tune) -> None:
    """Eleven GHZ blocks use 99 bits for the physical weight, more than one 64-bit word.

    For the CSS weight, the X and Z parts are searched separately, on 33 bits each.
    """
    # Eight-column blocks, rather than the default sixteen, keep the tables quick to build as plain
    # Python; this test is about packing, not tables.
    tune(coset_completion_block_bits=8)
    num_blocks = 11
    matrix, syndrome = _ghz_blocks(num_blocks)
    oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
    below = oracle.get_min_weight_bound(syndrome, num_blocks - 1)
    assert not below.within_max_weight and below.lower_bound == num_blocks
    found = oracle.get_min_weight_bound(syndrome, num_blocks)
    assert found.within_max_weight
    assert get_syndrome(matrix, found.representative) == syndrome
    assert get_pauli_weight(found.representative, 3 * num_blocks, css_weight) == num_blocks


@_BLOCK_CASES
def test_bounds_match_per_block_minima(
    block: tuple[str, ...], css_weight: bool, pytestconfig: pytest.Config
) -> None:
    """Five four-qubit blocks match per-block minima for every maximum weight up to five.

    The stabilizers of these blocks have even weight, but their cosets can have odd weight.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_blocks = 5
    matrix = get_stabilizer_matrix(_block_generators(block, num_blocks))
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
        matrix = get_stabilizer_matrix(_block_generators(block, 3))
        oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
        bound = oracle.get_min_weight_bound(syndrome, max_weight)
        assert bound == expected

        # The answer is sound for the block minima.
        local = [
            [(syndrome >> (len(block) * index + jj)) & 1 for jj in range(len(block))]
            for index in range(3)
        ]
        minimum = _block_minimum(block, local, css_weight)
        assert get_syndrome(matrix, bound.representative) == syndrome
        assert bound.within_max_weight == (minimum <= max_weight)
        assert (
            bound.lower_bound
            <= minimum
            <= get_pauli_weight(bound.representative, num_qubits, css_weight)
        )

        words = split_syndromes_into_words(
            [get_syndrome(matrix, 1), get_syndrome(matrix, 1 << num_qubits), syndrome], num_qubits
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
            matrix = get_random_css_state(num_qubits, num_x_stabilizers, rng)
            oracle = CosetLeaderSearch(matrix, css_weight=True)
            minima = get_exhaustive_minima(matrix, css_weight=True)
            for syndrome in rng.permutation(1 << num_qubits).tolist():
                for max_weight in rng.permutation(num_qubits + 1).tolist():
                    bound = oracle.get_min_weight_bound(syndrome, max_weight)
                    assert bound.within_max_weight == (minima[syndrome] <= max_weight)
                    assert get_syndrome(matrix, bound.representative) == syndrome
                    assert bound.lower_bound <= minima[syndrome]
                    if bound.within_max_weight:
                        weight = get_pauli_weight(bound.representative, num_qubits, True)
                        assert weight <= max_weight
                    else:
                        assert bound.lower_bound > max_weight
            syndromes = rng.permutation(1 << num_qubits).tolist()
            words = split_syndromes_into_words(syndromes, num_qubits)
            for max_weight in range(num_qubits + 1):
                found = _find_each_not_within_max_weight(oracle, matrix, words, max_weight)
                assert found == np.flatnonzero(minima[syndromes] > max_weight).tolist()


def test_stabilizers_must_define_a_pure_state() -> None:
    """Matrices that are not ``n`` independent commuting stabilizers on ``n`` qubits raise errors.

    One Z stabilizer on two qubits, or two copies of it, define a code that encodes a qubit, rather
    than a state.  A third stabilizer that is the product of two others is not independent, and X
    and Z on one qubit do not commute.
    """
    for stabilizers, message in [
        ([[1]], "columns"),
        ([[0, 0, 1, 0]], "not a state"),
        ([[0, 0, 1, 0], [0, 0, 1, 0]], "not a state"),
        ([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]], "independent"),
        ([[1, 0, 0, 0], [0, 0, 1, 0]], "commute"),
    ]:
        with pytest.raises(ValueError, match=message):
            CosetLeaderSearch(stabilizers)


def test_methods_take_the_names_of_distance_methods() -> None:
    """The alias ``"information_sets"`` selects the default method, and other names raise errors.

    The connected-support method weighs errors by CSS weight only on CSS states, so on a state that
    is not CSS, only the default method can answer queries by CSS weight.  ``"brute_force"`` has no
    coset searches.
    """
    matrix = get_graph_state(2, [(0, 1)], phase=True)
    with pytest.raises(ValueError, match="only on CSS states"):
        CosetLeaderSearch(matrix, css_weight=True, method="connected_support")
    assert_exact_answers(
        CosetLeaderSearch(matrix, css_weight=True, method="information_sets"), matrix, True
    )
    with pytest.raises(ValueError, match="no coset-leader search"):
        CosetLeaderSearch(matrix, method="brute_force")
    for unknown in ("gap", ["brouwer_zimmermann"]):
        with pytest.raises(ValueError, match="Unknown distance method") as error:
            CosetLeaderSearch(matrix, method=cast(Any, unknown))
        assert "brute_force" not in str(error.value)

    # Every other distance method has a coset-leader search.
    for method in get_args(DistanceMethod):
        if method != "brute_force":
            assert_exact_answers(CosetLeaderSearch(matrix, method=method), matrix, False)


def test_properties_are_read_only() -> None:
    """The number of qubits and the measure of weight of a search are fixed when it is built."""
    search = CosetLeaderSearch(get_stabilizer_matrix(["X_", "_Z"]), css_weight=cast(Any, 1))
    assert (search.num_qubits, search.css_weight) == (2, True)
    assert search.css_weight is True
    with pytest.raises(AttributeError):
        search.num_qubits = 3  # type: ignore[misc]
    with pytest.raises(AttributeError):
        search.css_weight = False  # type: ignore[misc]


def test_max_weight_must_be_a_nonnegative_integer() -> None:
    """Every query rejects a maximum weight that is negative or not an integer, such as a bool."""
    search = CosetLeaderSearch(get_stabilizer_matrix(["X_", "_Z"]))
    batch = search.pack_syndromes([1, 0])
    for max_weight, error in [(-1, ValueError), (1.0, TypeError), (True, TypeError)]:
        invalid: Any = max_weight
        with pytest.raises(error, match="max_weight"):
            search.get_min_weight_bound(1, invalid)
        with pytest.raises(error, match="max_weight"):
            search.find_first_not_within_max_weight(batch, invalid)
        with pytest.raises(error, match="max_weight"):
            search.get_min_weights(batch, invalid)

    # No error weighs more than the number of qubits, so larger maximum weights answer alike.
    assert search.get_min_weight_bound(1, 2**70) == search.get_min_weight_bound(1, 2)
    assert search.find_first_not_within_max_weight(batch, 2**70) == (1, None)
    np.testing.assert_array_equal(search.get_min_weights(batch, 2**70), [1])


def test_syndromes_of_queries_are_integers_or_binary_vectors() -> None:
    """A query gives the same answer for a packed syndrome as for its vector of syndrome bits.

    Packed syndromes out of range, bools, and vectors that are not binary, have the wrong length, or
    have two dimensions raise errors.
    """
    matrix = get_graph_state(5, [(0, 1), (0, 2), (2, 3), (3, 4)], phase=True)
    search = CosetLeaderSearch(matrix)
    for syndrome in (0b00110, 0b10011):
        bits = (syndrome >> np.arange(5)) & 1
        assert search.get_min_weight_bound(bits.tolist(), 2) == search.get_min_weight_bound(
            syndrome, 2
        )
    for packed in (-1, 1 << 5):
        with pytest.raises(ValueError, match="out of range"):
            search.get_min_weight_bound(packed, 2)
    for not_an_integer in (True, 1.0, None, "1"):
        with pytest.raises(TypeError, match="syndrome"):
            search.get_min_weight_bound(cast(Any, not_an_integer), 2)
    for vector in ([], [1, 0, 0, 0], [2, 0, 0, 0, 0], [[1, 0, 0, 0, 0]]):
        with pytest.raises(ValueError, match="syndrome"):
            search.get_min_weight_bound(vector, 2)


def test_packed_syndromes_unpack_to_their_bits(pytestconfig: pytest.Config) -> None:
    """Binary syndromes pack into a batch of their packed integers, and a vector into one row.

    On 70 qubits, each packed syndrome spans two words.  Syndromes that are not binary, or have the
    wrong length, raise errors.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_qubits = 70
    matrix = np.hstack(
        [np.zeros((num_qubits, num_qubits), np.uint8), np.eye(num_qubits, dtype=np.uint8)]
    )
    search = CosetLeaderSearch(matrix, css_weight=True)
    bits = rng.integers(2, size=(5, num_qubits), dtype=np.uint8)
    batch = search.pack_syndromes(bits)
    packed = [sum(int(bit) << jj for jj, bit in enumerate(row)) for row in bits]
    np.testing.assert_array_equal(batch, split_syndromes_into_words(packed, num_qubits))
    assert batch.dtype == np.uint64
    np.testing.assert_array_equal(unpack_bits(batch, num_qubits), bits)
    np.testing.assert_array_equal(search.pack_syndromes(bits[0]), batch[:1])
    np.testing.assert_array_equal(search.pack_syndromes(scipy.sparse.csr_array(bits)), batch)
    for invalid in (bits[:, 1:], np.full((1, num_qubits), 2), np.zeros((3, 0))):
        with pytest.raises(ValueError, match="syndromes"):
            search.pack_syndromes(invalid)


def test_representatives_unpack_to_binary_vectors_in_xz_order() -> None:
    """A representative unpacks to its X part and then its Z part, and other integers raise errors.

    On |0+>, the only error of CSS weight one with the syndrome 0b11 is X on qubit zero and Z on
    qubit one, which unpacks to X part [1, 0] and Z part [0, 1].  Integers outside the four bits of
    a Pauli error raise errors.
    """
    search = CosetLeaderSearch(get_stabilizer_matrix(["Z_", "_X"]), css_weight=True)
    bound = search.get_min_weight_bound(0b11, 1)
    assert bound.within_max_weight
    assert search.unpack_pauli(bound.representative).tolist() == [1, 0, 0, 1]
    for pauli in (-1, 1 << 4):
        with pytest.raises(ValueError, match="out of range"):
            search.unpack_pauli(pauli)

    # On six qubits, a packed error spans two bytes, and bit j of it is entry j of the vector.
    search = CosetLeaderSearch(get_graph_state(6, [(j, j + 1) for j in range(5)]))
    for pauli in (0b100000000001, 0b011010010110):
        assert search.unpack_pauli(pauli).tolist() == [(pauli >> j) & 1 for j in range(12)]


@pytest.mark.parametrize(
    "matrix, css_weight",
    [
        (ROUTED_PHYSICAL_STATES[0], False),
        (ROUTED_PHYSICAL_STATES[1], False),
        (ROUTED_CSS_WEIGHT_STATE, True),
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
    assert_exact_answers(oracle, matrix, css_weight)
    num_qubits = len(matrix)
    minima = get_exhaustive_minima(matrix, css_weight)
    syndromes = split_syndromes_into_words(list(range(1 << num_qubits)), num_qubits)
    for max_weight in range(num_qubits + 1):
        found = _find_each_not_within_max_weight(oracle, matrix, syndromes, max_weight)
        assert found == np.flatnonzero(minima > max_weight).tolist()


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
    oracle answers one batch of every syndrome, in random order, at every maximum weight from zero
    to one past the number of qubits.  The BZ searches take the disjoint families, and at the
    threshold zero other families by tails, with completion tables and without them.  The
    connected-support method, which the tuning of these families does not configure, answers under
    the default tuning, but not by CSS weight on states that are not CSS.
    """
    tune(**tuning)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    cases = []
    for num_qubits in range(2, 6):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        graph_state = get_graph_state(num_qubits, edges, phase=True)
        css_state = get_random_css_state(num_qubits, int(rng.integers(num_qubits + 1)), rng)
        cases += [(graph_state, False, True), (graph_state, True, False), (css_state, True, True)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5)]
    edges += [(2, 6), (2, 7), (3, 4), (3, 5), (3, 7), (4, 5), (5, 6), (5, 7)]
    graph_state = get_graph_state(8, edges, phase=True)
    cases += [(graph_state, False, True), (graph_state, True, False)]
    cycle_state = get_graph_state(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)], phase=True)
    cases.append((cycle_state, True, False))
    steane_zero = ("XXXX___", "XX__XX_", "X_X_X_X", "ZZZZ___", "ZZ__ZZ_", "Z_Z_Z_Z", "ZZZZZZZ")
    cases += [
        (get_stabilizer_matrix(steane_zero), css_weight, True) for css_weight in (False, True)
    ]
    ghz_block = ("XXXX____", "ZZ______", "_ZZ_____", "__ZZ____")
    conjugate_block = ("____ZZZZ", "____XX__", "_____XX_", "______XX")
    cases.append((get_stabilizer_matrix(ghz_block + conjugate_block), True, True))
    for matrix, css_weight, by_support in cases:
        num_qubits = len(matrix)
        minima = get_exhaustive_minima(matrix, css_weight)
        syndromes = rng.permutation(1 << num_qubits)
        packed_syndromes = split_syndromes_into_words(syndromes.tolist(), num_qubits)
        methods: list[DistanceMethod] = ["brouwer_zimmermann"]
        if by_support and not tuning:
            methods.append("connected_support")
        for method in methods:
            oracle = CosetLeaderSearch(matrix, css_weight=css_weight, method=method)
            for max_weight in range(num_qubits + 2):
                weights = oracle.get_min_weights(packed_syndromes, max_weight)
                assert np.array_equal(weights, np.minimum(minima[syndromes], max_weight + 1))


def test_batches_are_checked() -> None:
    """Batches must be ``uint64`` arrays with the words per row of ``pack_syndromes``, in range.

    The syndromes of the state |+0> are below four, so they take one word, and a row with bit two
    set is out of range.  The syndrome three needs Z on the first qubit and X on the second.
    """
    search = CosetLeaderSearch(get_stabilizer_matrix(["X_", "_Z"]))
    batch: Any
    for query in (search.find_first_not_within_max_weight, search.get_min_weights):
        for batch in ([[3]], np.array([[3]], dtype=np.int64)):
            with pytest.raises(TypeError, match="pack_syndromes"):
                query(batch, 1)
        for batch in (np.array([3], dtype=np.uint64), np.array([[3, 0]], dtype=np.uint64)):
            with pytest.raises(ValueError, match="pack_syndromes"):
                query(batch, 1)
        with pytest.raises(ValueError, match="out of range"):
            query(np.array([[4]], dtype=np.uint64), 1)
    assert search.get_min_weights(np.array([[3]], dtype=np.uint64), 1).tolist() == [2]


def test_connected_support_batches_by_css_weight_join_their_parts(
    pytestconfig: pytest.Config,
) -> None:
    """Batches of the connected-support method by CSS weight find the classes above ``max_weight``.

    On CSS states, the CSS weight splits the search of a class into searches of its X and Z parts.
    Every syndrome of random CSS states on one to five qubits, in random order, is checked at every
    maximum weight against exhaustive minima.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 6):
        matrix = get_random_css_state(num_qubits, num_qubits // 2, rng)
        oracle = CosetLeaderSearch(matrix, css_weight=True, method="connected_support")
        syndromes = rng.permutation(1 << num_qubits)
        minima = get_exhaustive_minima(matrix, css_weight=True)[syndromes]
        words = split_syndromes_into_words(syndromes.tolist(), num_qubits)
        for max_weight in range(num_qubits + 2):
            found = _find_each_not_within_max_weight(oracle, matrix, words, max_weight)
            assert found == np.flatnonzero(minima > max_weight).tolist()


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
    matrix = get_stabilizer_matrix(generators)
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
    words = split_syndromes_into_words(syndromes, 68)
    for max_weight in range(4):
        found = _find_each_not_within_max_weight(oracle, matrix, words, max_weight)
        assert found == np.flatnonzero(np.array(minima) > max_weight).tolist()
