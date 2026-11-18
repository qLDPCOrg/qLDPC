# SPDX-License-Identifier: Apache-2.0

"""Tests for exact minimum-weight bounds on syndrome classes of Pauli errors."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Sequence
from typing import Any, cast, get_args

import numba
import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse
import stim

from qldpc._binary import unpack_bits

from ..conftest import Tune
from . import _threads, coset_leaders
from ._brouwer_zimmermann.coset import BrouwerZimmermannCosetSearch
from ._tuning import MAX_VALUE, use_tuning
from .conftest import (
    ROUTED_CSS_WEIGHT_STATE,
    ROUTED_PHYSICAL_STATES,
    THREAD_TIMEOUT,
    assert_exact_answers,
    get_exhaustive_minima,
    get_graph_state,
    get_pauli_weight,
    get_random_css_state,
    get_stabilizer_matrix,
    get_syndrome,
    scramble,
    shared_arrays_read_only,
    split_syndromes_into_words,
)
from .core import DistanceMethod
from .coset_leaders import CosetLeaderSearch, _get_coset_search_method
from .min_weight_bounds import MinWeightBound


def _class_weights(
    matrix: npt.NDArray[np.uint8], syndrome: int, css_weight: bool
) -> npt.NDArray[np.int_]:
    """Weights of every error with the syndrome, for stabilizers whose X parts form the identity.

    Z on the set syndrome bits has the syndrome, and its products with the stabilizers are the
    other errors with it, so this enumeration is independent of the code under test.
    """
    n = len(matrix)
    assert np.array_equal(matrix[:, :n], np.eye(n))
    base = np.concatenate([np.zeros(n, dtype=int), (syndrome >> np.arange(n)) & 1])
    subsets = (np.arange(1 << n)[:, None] >> np.arange(n)) & 1
    errors = (base + subsets @ matrix.astype(int)) % 2
    x, z = errors[:, :n], errors[:, n:]
    return (x | z).sum(axis=1) if not css_weight else np.maximum(x.sum(axis=1), z.sum(axis=1))


# Four-qubit blocks: a GHZ state, which is CSS, and the same state after S on its first qubit, which
# is not.  By CSS weight, a search splits into X and Z parts only on the GHZ block, and by physical
# weight it searches Pauli errors on both, so one block covers that case.
_GHZ_BLOCK = ("XXXX", "ZZ__", "_ZZ_", "__ZZ")
_PHASED_GHZ_BLOCK = ("YXXX", "ZZ__", "_ZZ_", "__ZZ")
_BLOCK_CASES = pytest.mark.parametrize(
    ("block", "css_weight"),
    [(_GHZ_BLOCK, False), (_GHZ_BLOCK, True), (_PHASED_GHZ_BLOCK, True)],
    ids=["ghz-physical", "ghz-css", "phased-ghz-css"],
)

# A state whose class of syndrome 3 has minimum weight two, which connected-cluster queries at
# maximum weight two find by visiting states.  Some of its classes have minimum weight above two.
_THREADED_STATE = ROUTED_PHYSICAL_STATES[1]


def _ghz_blocks(num_blocks: int) -> tuple[npt.NDArray[np.uint8], int]:
    """Stabilizers of independent GHZ blocks, and the syndrome of Z on each block's last qubit.

    Every error with that syndrome needs a Z part on each block, so ``num_blocks`` is its exact
    minimum weight by either measure.
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

    Every Pauli on one block is listed, and the blocks are combined by dynamic programming.
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
    search: CosetLeaderSearch,
    matrix: npt.NDArray[np.uint8],
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
) -> list[int]:
    """Find each syndrome whose class has no error of weight at most ``max_weight``, in turn.

    Each bound must be valid, and, for maximum weights of two or more, equal to that of
    ``get_min_weight_bound``, an independent computation when there is one coset search.
    """
    found: list[int] = []
    while True:
        start = found[-1] + 1 if found else 0
        index, bound = search.find_first_not_within_max_weight(syndromes[start:], max_weight)
        if bound is None:
            assert start + index == len(syndromes)
            return found
        found.append(start + index)
        syndrome = int.from_bytes(syndromes[found[-1]].astype("<u8").tobytes(), "little")
        assert not bound.within_max_weight and bound.lower_bound > max_weight
        assert get_syndrome(matrix, bound.representative) == syndrome
        if max_weight >= 2:
            assert bound == search.get_min_weight_bound(syndrome, max_weight)


@pytest.mark.parametrize(
    ("css_weight", "css_singles_table"), [(False, True), (True, True), (True, False)]
)
def test_bounds_match_exhaustive_minima(
    css_weight: bool,
    css_singles_table: bool,
    pytestconfig: pytest.Config,
    tune: Tune,
) -> None:
    """Bounds agree with exhaustive minima."""
    # Graph states with phases on an even number of qubits are not CSS, so their queries by CSS
    # weight look up errors of CSS weight one in a table, or, with no room for the table, match
    # single-qubit X and Z errors.
    if not css_singles_table:
        tune(max_css_single_pairs=0)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 7):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        matrix = get_graph_state(num_qubits, edges, phase=num_qubits % 2 == 0)
        search = CosetLeaderSearch(matrix, css_weight=css_weight)
        for syndrome in rng.permutation(1 << num_qubits).tolist():
            minimum = int(_class_weights(matrix, syndrome, css_weight).min())
            for max_weight in rng.permutation(num_qubits + 1).tolist():
                bound = search.get_min_weight_bound(syndrome, max_weight)
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
    """Repeated batch searches find exactly the classes of minimum weight above ``max_weight``."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 7):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        matrix = get_graph_state(num_qubits, edges, phase=num_qubits % 2 == 0)
        search = CosetLeaderSearch(matrix, css_weight=css_weight)
        syndromes = rng.permutation(1 << num_qubits).tolist()
        minima = np.array([_class_weights(matrix, ss, css_weight).min() for ss in syndromes])
        for max_weight in range(num_qubits + 2):
            words = split_syndromes_into_words(syndromes, num_qubits)
            found = _find_each_not_within_max_weight(search, matrix, words, max_weight)
            assert found == np.flatnonzero(minima > max_weight).tolist()


def test_batches_return_bounds_above_max_weight_plus_one() -> None:
    """A batch can return a bound above ``max_weight + 1``, as ``get_min_weight_bound`` does."""
    # On |0000> with CSS weight, the syndrome 1111 needs X on every qubit.  The code of X parts has
    # no generators, so each of its cosets is one word, which proves this at layer zero.
    matrix = np.hstack([np.zeros((4, 4), dtype=np.uint8), np.eye(4, dtype=np.uint8)])
    search = CosetLeaderSearch(matrix, css_weight=True)
    words = split_syndromes_into_words([0b0011, 0b1111], 4)
    assert _find_each_not_within_max_weight(search, matrix, words, 2) == [1]
    assert search.get_min_weight_bound(0b1111, 2).lower_bound == 4


@_BLOCK_CASES
def test_batches_read_syndromes_of_several_words(
    block: tuple[str, ...], css_weight: bool, pytestconfig: pytest.Config
) -> None:
    """Batches of syndromes of two words, on 17 four-qubit blocks, agree with block minima."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_blocks = 17
    generators = _block_generators(block, num_blocks)
    matrix = get_stabilizer_matrix(generators)
    search = CosetLeaderSearch(matrix, css_weight=css_weight)
    syndromes, minima = [], []
    for _ in range(30):
        error = stim.PauliString(4 * num_blocks)
        for qubit in rng.choice(4 * num_blocks, size=rng.integers(1, 4), replace=False):
            error[int(qubit)] = int(rng.integers(1, 4))
        syndromes.append(sum((not error.commutes(gg)) << jj for jj, gg in enumerate(generators)))
        bits = [(syndromes[-1] >> jj) & 1 for jj in range(4 * num_blocks)]
        local = [bits[4 * index : 4 * index + 4] for index in range(num_blocks)]
        # Blocks that the error does not touch add nothing to its minimum weight.
        minima.append(_block_minimum(block, [ss for ss in local if any(ss)], css_weight))
    for max_weight in range(4):
        words = split_syndromes_into_words(syndromes, 68)
        found = _find_each_not_within_max_weight(search, matrix, words, max_weight)
        assert found == np.flatnonzero(np.array(minima) > max_weight).tolist()


def test_x_and_z_syndromes_span_several_words(pytestconfig: pytest.Config) -> None:
    """On |0>^70 |+>^70, syndromes of X and Z parts that span two words give exact answers."""
    # An error is equivalent to one with X only on the first 70 qubits and Z only on the last 70, so
    # its smallest CSS weight is the larger of the numbers of these.  The errors below act on qubits
    # whose syndrome bits fall in both words.
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_qubits = 140
    pure = np.zeros((num_qubits, 2 * num_qubits), dtype=np.uint8)
    pure[:70, num_qubits:][np.arange(70), np.arange(70)] = 1  # Z on each of the first 70 qubits
    pure[70:, :num_qubits][np.arange(70), np.arange(70, num_qubits)] = 1  # X on each of the rest
    matrix = scramble(pure, rng)
    search = CosetLeaderSearch(matrix, css_weight=True)
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
            bound = search.get_min_weight_bound(syndrome, max_weight)
            assert bound.within_max_weight == (minimum <= max_weight)
            assert get_syndrome(matrix, bound.representative) == syndrome
        words = split_syndromes_into_words(syndromes, num_qubits)
        found = _find_each_not_within_max_weight(search, matrix, words, max_weight)
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
    """The CSS weight is minimized over both parts of an error at once."""
    # On this path graph state, Z^4 is equivalent to an error without an X part and to one without a
    # Z part, but to no error of CSS weight below two.
    matrix = get_graph_state(4, [(0, 1), (1, 2), (2, 3)])
    assert _class_weights(matrix, 15, True).min() == 2
    search = CosetLeaderSearch(matrix, css_weight=True)
    bound = search.get_min_weight_bound(15, 1)
    assert not bound.within_max_weight and bound.lower_bound == 2
    assert get_syndrome(matrix, bound.representative) == 15
    assert search.get_min_weight_bound(15, 2).within_max_weight


@pytest.mark.parametrize(("syndrome", "minimum"), [(44, 2), (108, 3)])
def test_rank_deficient_information_sets_give_valid_bounds(syndrome: int, minimum: int) -> None:
    """Rank-deficient sets give valid bounds whether a weight-two error exists or not."""
    # Qubit 6 is isolated, so the last information set has rank five rather than eight, and a search
    # that took it as full rank would miss the error of weight two with syndrome 44.  Syndrome 108
    # also flips the stabilizer of qubit 6, and certifying it at maximum weight two skips this set,
    # whose first two layers add nothing.
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
    """A search that exhausts every layer proves the exact minimum."""
    # The syndrome 11111 needs X or Y on each of the last four qubits, and Z X X X X has it.  The
    # state is not CSS, so the Pauli errors are searched, and the bounds of the certificate stay
    # below four.
    matrix = get_stabilizer_matrix(("Y____", "_Z___", "__Z__", "___Z_", "____Z"))
    search = CosetLeaderSearch(matrix, css_weight=True)
    bound = search.get_min_weight_bound(0b11111, 3)
    assert not bound.within_max_weight and bound.lower_bound == 4
    assert search.get_min_weight_bound(0b11111, 4).within_max_weight


@pytest.mark.parametrize("css_weight", [False, True])
def test_wide_states_span_several_packed_words(css_weight: bool, tune: Tune) -> None:
    """On eleven GHZ blocks, queries by either weight find the exact minimum."""
    # The words of a search by physical weight have 99 bits, more than one 64-bit word, and those of
    # X and Z parts 33 bits.  Tables of eight-column blocks are quick to build as plain Python.
    tune(coset_completion_block_bits=8)
    num_blocks = 11
    matrix, syndrome = _ghz_blocks(num_blocks)
    search = CosetLeaderSearch(matrix, css_weight=css_weight)
    below = search.get_min_weight_bound(syndrome, num_blocks - 1)
    assert not below.within_max_weight and below.lower_bound == num_blocks
    found = search.get_min_weight_bound(syndrome, num_blocks)
    assert found.within_max_weight
    assert get_syndrome(matrix, found.representative) == syndrome
    assert get_pauli_weight(found.representative, 3 * num_blocks, css_weight) == num_blocks


@_BLOCK_CASES
def test_bounds_match_per_block_minima(
    block: tuple[str, ...], css_weight: bool, pytestconfig: pytest.Config, tune: Tune
) -> None:
    """Five blocks, whose stabilizers are even but cosets can be odd, match block minima.

    A query by physical weight can stop by its certificate with a representative heavier than the
    minimum.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_blocks = 5
    matrix = get_stabilizer_matrix(_block_generators(block, num_blocks))
    search = CosetLeaderSearch(matrix, css_weight=css_weight)

    # Bounds hold for the first syndrome and for random ones at each maximum weight.  The phased GHZ
    # block is slow to search as plain Python, so it gets fewer random syndromes.
    num_random = 3 if block == _PHASED_GHZ_BLOCK else 7
    syndromes = [443351, *rng.integers(1 << (num_blocks * len(block)), size=num_random).tolist()]
    for syndrome in syndromes:
        local = [
            [(syndrome >> (len(block) * index + jj)) & 1 for jj in range(len(block))]
            for index in range(num_blocks)
        ]
        minimum = _block_minimum(block, local, css_weight)
        if syndrome == syndromes[0]:
            first_minimum = minimum
        for max_weight in range(1, 6):
            bound = search.get_min_weight_bound(syndrome, max_weight)
            assert bound.within_max_weight == (minimum <= max_weight)
            assert bound.lower_bound <= minimum
            if not bound.within_max_weight:
                assert bound.lower_bound > max_weight

    # Under a tuning whose queries take the greedy packing of the columns in natural order, the
    # query of the first syndrome at maximum weight two by physical weight stops by its certificate
    # while the lightest error that it has visited, which it returns, is heavier than the minimum.
    # A search that ran past its certificate would return a minimal one.
    if not css_weight:
        tune(coset_route_minimum_work=MAX_VALUE, disjoint_random_column_orders=0)
        bound = CosetLeaderSearch(matrix).get_min_weight_bound(syndromes[0], 2)
        assert not bound.within_max_weight
        assert get_pauli_weight(bound.representative, len(matrix), False) > first_minimum


@pytest.mark.parametrize("block", [_PHASED_GHZ_BLOCK, _GHZ_BLOCK], ids=["phased-ghz", "ghz"])
def test_css_weight_queries_on_blocks_stop_below_their_representatives(
    block: tuple[str, ...], tune: Tune
) -> None:
    """Queries by CSS weight on three blocks can stop by certificates below their representatives.

    They do under a tuning whose queries take the greedy packing of the columns in natural order.
    On the GHZ block, whose codes of X and Z parts are even, the bound also rounds up to the parity
    of the X parts of the syndrome, which are even, and so exceeds ``max_weight + 1``.
    """
    tune(coset_route_minimum_work=MAX_VALUE, disjoint_random_column_orders=0)
    syndrome, max_weight = 2635, 2
    matrix = get_stabilizer_matrix(_block_generators(block, 3))
    bound = CosetLeaderSearch(matrix, css_weight=True).get_min_weight_bound(syndrome, max_weight)
    local = [[(syndrome >> (4 * index + jj)) & 1 for jj in range(4)] for index in range(3)]
    minimum = _block_minimum(block, local, True)
    weight = get_pauli_weight(bound.representative, len(matrix), True)
    assert get_syndrome(matrix, bound.representative) == syndrome
    assert not bound.within_max_weight
    assert max_weight < bound.lower_bound <= minimum <= weight
    assert bound.lower_bound < weight
    if block == _GHZ_BLOCK:
        assert bound.lower_bound > max_weight + 1


def test_css_bounds_match_exhaustive_minima(pytestconfig: pytest.Config) -> None:
    """On random CSS states, both queries for the CSS weight agree with exhaustive minima."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 6):
        for num_x_stabilizers in range(num_qubits + 1):
            matrix = get_random_css_state(num_qubits, num_x_stabilizers, rng)
            search = CosetLeaderSearch(matrix, css_weight=True)
            minima = get_exhaustive_minima(matrix, css_weight=True)
            for syndrome in rng.permutation(1 << num_qubits).tolist():
                for max_weight in rng.permutation(num_qubits + 1).tolist():
                    bound = search.get_min_weight_bound(syndrome, max_weight)
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
                found = _find_each_not_within_max_weight(search, matrix, words, max_weight)
                assert found == np.flatnonzero(minima[syndromes] > max_weight).tolist()


def test_stabilizers_must_define_a_pure_state() -> None:
    """Stabilizers that are not ``n`` independent commuting ones on ``n`` qubits raise errors."""
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
    """Coset-leader searches take each distance method but brute force, and reject invalid uses."""
    matrix = get_graph_state(2, [(0, 1)], phase=True)
    with pytest.raises(ValueError, match="only on CSS states"):
        CosetLeaderSearch(matrix, css_weight=True, method="connected_cluster")
    assert_exact_answers(
        CosetLeaderSearch(matrix, css_weight=True, method="information_sets"), matrix, True
    )
    with pytest.raises(ValueError, match="no coset-leader search"):
        CosetLeaderSearch(matrix, method="brute_force")
    for unknown in ("gap", ["brouwer_zimmermann"]):
        with pytest.raises(ValueError, match="Unknown distance method") as error:
            CosetLeaderSearch(matrix, method=cast(Any, unknown))
        assert "brute_force" not in str(error.value)

    for method in get_args(DistanceMethod):
        if method != "brute_force":
            assert_exact_answers(CosetLeaderSearch(matrix, method=method), matrix, False)

    # This state is not CSS, so only the Brouwer-Zimmermann method weighs its errors by CSS weight,
    # and "auto" takes that method.
    assert _get_coset_search_method("auto") == "brouwer_zimmermann"
    assert_exact_answers(CosetLeaderSearch(matrix, css_weight=True, method="auto"), matrix, True)


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
    """A query gives the same answer for a packed syndrome as for its vector, and checks both."""
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
    """Binary syndromes pack into a batch of their packed integers, and a vector into one row."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # On 70 qubits, each packed syndrome spans two words.
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
    """A representative unpacks to its X part and then its Z part; other integers raise errors."""
    # On |0+>, the only error of CSS weight one with the syndrome 0b11 is X on qubit zero and Z on
    # qubit one.
    search = CosetLeaderSearch(get_stabilizer_matrix(["Z_", "_X"]), css_weight=True)
    bound = search.get_min_weight_bound(0b11, 1)
    assert bound.within_max_weight
    assert search.unpack_pauli(bound.representative).tolist() == [1, 0, 0, 1]
    for pauli in (-1, 1 << 4):
        with pytest.raises(ValueError, match="out of range"):
            search.unpack_pauli(pauli)

    # On six qubits, a packed error spans two bytes.
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
def test_searches_answer_exactly_with_other_families(
    matrix: npt.NDArray[np.uint8], css_weight: bool, tune: Tune
) -> None:
    """Searches answer exactly when costly queries take other families."""
    # With coset_route_minimum_work at zero, costly queries take families that beat the greedy
    # packings of the graph states, and routes by tails, except in a search of Pauli errors by CSS
    # weight, as ``test_routed_states_take_families_that_beat_the_greedy_packing`` of
    # ``_brouwer_zimmermann/coset_test.py`` checks.  On |0000>, the CSS weight splits the search,
    # and the X parts have the empty code, which no other family improves.
    tune(coset_route_minimum_work=0)
    search = CosetLeaderSearch(matrix, css_weight=css_weight)
    assert_exact_answers(search, matrix, css_weight)
    num_qubits = len(matrix)
    minima = get_exhaustive_minima(matrix, css_weight)
    syndromes = split_syndromes_into_words(list(range(1 << num_qubits)), num_qubits)
    for max_weight in range(num_qubits + 1):
        found = _find_each_not_within_max_weight(search, matrix, syndromes, max_weight)
        assert found == np.flatnonzero(minima > max_weight).tolist()


@pytest.mark.parametrize(
    "tuning",
    [
        {},
        {"coset_route_minimum_work": 0},
        {"coset_route_minimum_work": 0, "coset_completion_max_table_bytes": 0},
    ],
    ids=["disjoint", "tables", "no-tables"],
)
def test_min_weights_match_exhaustive_minima(
    tuning: dict[str, int], pytestconfig: pytest.Config, tune: Tune
) -> None:
    """The minimum weights of syndrome classes agree with exhaustive minima, by either method."""
    tune(**tuning)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # Graph states with phases are not CSS, so one search covers their Pauli errors.  By physical
    # weight, the binary words of their errors are even, so a BZ search keeps its incumbent one
    # below the weight of its best word.
    cases = []
    for num_qubits in range(2, 6):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        graph_state = get_graph_state(num_qubits, edges, phase=True)
        css_state = get_random_css_state(num_qubits, int(rng.integers(num_qubits + 1)), rng)
        cases += [(graph_state, False, True), (graph_state, True, False), (css_state, True, True)]
    # The searches of this state on eight qubits also stop early within steps and skip words by
    # tables.
    edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (1, 6), (1, 7), (2, 3), (2, 4), (2, 5)]
    edges += [(2, 6), (2, 7), (3, 4), (3, 5), (3, 7), (4, 5), (5, 6), (5, 7)]
    graph_state = get_graph_state(8, edges, phase=True)
    cases += [(graph_state, False, True), (graph_state, True, False)]
    # The stabilizers of this cycle state are even as words [x|z] too, but CSS weights have no
    # parity, so its searches by CSS weight do not lower their incumbents.
    cycle_state = get_graph_state(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)], phase=True)
    cases.append((cycle_state, True, False))
    # The CSS weight splits the searches of CSS states, and the search of a Z part ends at a word
    # within the weight of its X part.  The code of X parts of the Steane state is even, and that of
    # Z parts is not.  Both codes of a GHZ block beside its conjugate by Hadamards are even.  Some
    # classes of that state have X parts of weight two and Z parts of weight three, and a
    # connected-cluster search weighs such a Z part exactly only from a first limit of at most two.
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
        # The tuning of these families does not configure the connected-cluster method, which
        # weighs by CSS weight only on CSS states.
        methods: list[DistanceMethod] = ["brouwer_zimmermann"]
        if by_support and not tuning:
            methods.append("connected_cluster")
        for method in methods:
            search = CosetLeaderSearch(matrix, css_weight=css_weight, method=method)
            for max_weight in range(num_qubits + 2):
                weights = search.get_min_weights(packed_syndromes, max_weight)
                assert np.array_equal(weights, np.minimum(minima[syndromes], max_weight + 1))


def test_batches_are_checked() -> None:
    """Batches must be ``uint64`` arrays with the words per row of ``pack_syndromes``, in range."""
    # The syndromes of |+0> are below four, and the syndrome three needs Z on the first qubit and X
    # on the second.
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


def test_connected_cluster_searches_take_the_fields_of_their_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connected-cluster coset searches are built with the fields of the tuning that configure them.

    This holds for the search of Pauli errors, and for the searches of the X and Z parts of a CSS
    state by CSS weight.
    """
    # Record the options with which each connected-cluster search is built.
    options: list[dict[str, Any]] = []
    build = coset_leaders.ConnectedClusterCosetSearch

    def record(matrix: npt.NDArray[np.uint8], **kwargs: Any) -> Any:
        options.append(kwargs)
        return build(matrix, **kwargs)

    monkeypatch.setattr(coset_leaders, "ConnectedClusterCosetSearch", record)

    # Build a Pauli search, and the two searches of a CSS state by CSS weight, under a tuning whose
    # fields differ from the defaults.
    css_state = get_random_css_state(4, 2, np.random.default_rng(0))
    with use_tuning(
        max_sparsifying_rounds=5,
        sparsifying_rounds_without_improvement=2,
        restrict_viable_symbols=False,
        stabilizer_overlap_cuts=False,
    ):
        CosetLeaderSearch(get_graph_state(2, [(0, 1)], phase=True), method="connected_cluster")
        CosetLeaderSearch(css_state, css_weight=True, method="connected_cluster")

    # Each of the three searches got the fields of that tuning.
    expected = {
        "max_rounds": 5,
        "rounds_without_improvement": 2,
        "restrict_viable_symbols": False,
        "stabilizer_overlap_cuts": False,
    }
    assert [{key: kwargs[key] for key in expected} for kwargs in options] == [expected] * 3


def test_connected_cluster_searches_get_the_stabilizers_in_their_kernel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each connected-cluster search gets the stabilizers in its kernel as overlap words.

    A Pauli search gets its own sparser stabilizers, and the search of X parts the X parts of the
    pure-X stabilizers, which the search of Z parts checks with, and vice versa (Section 3.8 of
    ``connected_cluster.md``).
    """
    # Record the overlap words that each search gets.
    received: dict[int, tuple[coset_leaders.ConnectedClusterCosetSearch, set[bytes]]] = {}
    use_words = coset_leaders.ConnectedClusterCosetSearch.use_query_overlap_words

    def record(self: coset_leaders.ConnectedClusterCosetSearch, words: npt.ArrayLike) -> None:
        """Record the overlap words that a search gets, and give them to it."""
        rows = {row.tobytes() for row in np.asarray(words, dtype=np.uint8)}
        received.setdefault(id(self), (self, set()))[1].update(rows)
        use_words(self, words)

    def get_rows(search: coset_leaders.ConnectedClusterCosetSearch) -> set[bytes]:
        """The checks of a search, as they compare with the recorded words."""
        return {np.asarray(row, dtype=np.uint8).tobytes() for row in search.checks}

    monkeypatch.setattr(
        coset_leaders.ConnectedClusterCosetSearch, "use_query_overlap_words", record
    )

    # A Pauli search gets its own checks.
    css_state = get_random_css_state(6, 3, np.random.default_rng(0))
    CosetLeaderSearch(css_state, method="connected_cluster")
    ((pauli_search, pauli_words),) = received.values()
    assert pauli_words == get_rows(pauli_search)

    # By CSS weight, the search of each part gets the checks of the other.  The two searches get
    # different words, so the assertion would fail if each search got its own checks.
    received.clear()
    CosetLeaderSearch(css_state, css_weight=True, method="connected_cluster")
    (first, first_words), (second, second_words) = received.values()
    assert first_words == get_rows(second) and second_words == get_rows(first)
    assert first_words != second_words


def test_connected_cluster_searches_by_css_weight_answer_exactly() -> None:
    """Connected-cluster queries by CSS weight on a CSS state agree with exhaustive minima."""
    css_state = get_random_css_state(4, 2, np.random.default_rng(0))
    search = CosetLeaderSearch(css_state, css_weight=True, method="connected_cluster")
    assert_exact_answers(search, css_state, True)


def test_connected_cluster_batches_by_css_weight_join_their_parts(
    pytestconfig: pytest.Config,
) -> None:
    """Connected-cluster batches by CSS weight find the classes above ``max_weight``."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 6):
        matrix = get_random_css_state(num_qubits, num_qubits // 2, rng)
        search = CosetLeaderSearch(matrix, css_weight=True, method="connected_cluster")
        syndromes = rng.permutation(1 << num_qubits)
        minima = get_exhaustive_minima(matrix, css_weight=True)[syndromes]
        words = split_syndromes_into_words(syndromes.tolist(), num_qubits)
        for max_weight in range(num_qubits + 2):
            found = _find_each_not_within_max_weight(search, matrix, words, max_weight)
            assert found == np.flatnonzero(minima > max_weight).tolist()


def test_connected_cluster_batches_read_syndromes_of_several_words(
    pytestconfig: pytest.Config, tune: Tune
) -> None:
    """Connected-cluster batches of syndromes of two words agree with block minima."""
    # One round of sparsifying gives the default checks on these blocks, without the many rounds
    # that are slow as plain Python.
    tune(max_sparsifying_rounds=1)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_blocks = 17
    generators = _block_generators(_PHASED_GHZ_BLOCK, num_blocks)
    matrix = get_stabilizer_matrix(generators)
    search = CosetLeaderSearch(matrix, method="connected_cluster")
    syndromes, minima = [], []
    for error_index in range(30):
        error = stim.PauliString(4 * num_blocks)
        # The first error acts on blocks 0, 8, and 16.  Block 16 has its syndrome bits in the second
        # word, so a search that read only the first word would find a lighter error.
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
        found = _find_each_not_within_max_weight(search, matrix, words, max_weight)
        assert found == np.flatnonzero(np.array(minima) > max_weight).tolist()


def test_batches_on_threads_answer_as_on_one_thread(
    pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Batch queries on threads answer as on one thread, on a CSS state part by part.

    By CSS weight, the batches of Z parts take the weights of the X parts within ``max_weight`` as
    floors.  With ``information_set_batch_exponent=0``, a batch moves to threads once its queries on
    the calling thread have done ``num_threads`` units of work.
    """
    # Record the floors and the weights of each batch query of a BZ search.
    calls: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = []
    get_min_weights = BrouwerZimmermannCosetSearch.get_min_weights

    def record(
        self: BrouwerZimmermannCosetSearch,
        syndromes: npt.NDArray[np.uint64],
        max_weight: int,
        floors: npt.NDArray[np.int64],
        **options: Any,
    ) -> npt.NDArray[np.int64]:
        """Record the floors and the weights of a batch query of a BZ search, and run it."""
        given_floors = floors.copy()
        weights = get_min_weights(self, syndromes, max_weight, floors, **options)
        calls.append((given_floors, weights.copy()))
        return weights

    monkeypatch.setattr(BrouwerZimmermannCosetSearch, "get_min_weights", record)

    # Build a search on one thread and a search on three, for random syndromes of a CSS state.
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    matrix = get_random_css_state(6, 3, rng)
    num_qubits = len(matrix)
    minima = get_exhaustive_minima(matrix, True)
    syndromes = rng.integers(1 << num_qubits, size=100)
    batch = split_syndromes_into_words(syndromes.tolist(), num_qubits)
    with use_tuning(num_threads=1):
        serial = CosetLeaderSearch(matrix, css_weight=True)
    with use_tuning(num_threads=3, information_set_batch_exponent=0):
        threaded = CosetLeaderSearch(matrix, css_weight=True)

    # At each maximum weight, the searches find the same classes, the batch of Z parts takes the
    # weights of the X parts within the maximum weight as floors, and the weights are exact.
    for max_weight in range(num_qubits + 1):
        found = _find_each_not_within_max_weight(threaded, matrix, batch, max_weight)
        assert found == np.flatnonzero(minima[syndromes] > max_weight).tolist()
        for start in [0, *found]:
            assert threaded.find_first_not_within_max_weight(
                batch[start:], max_weight
            ) == serial.find_first_not_within_max_weight(batch[start:], max_weight)
        calls.clear()
        weights = threaded.get_min_weights(batch, max_weight)
        (_, x_weights), (z_floors, _) = calls
        assert np.array_equal(z_floors, x_weights[x_weights <= max_weight])
        assert np.array_equal(weights, serial.get_min_weights(batch, max_weight))
        assert np.array_equal(weights, np.minimum(minima[syndromes], max_weight + 1))


def _record_thread_options(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, int]]:
    """Record the name, threads, and split work of each batch query of a BZ coset search."""
    calls: list[tuple[str, int, int]] = []
    for name in ("find_first_not_within_max_weight", "get_min_weights"):
        query = getattr(BrouwerZimmermannCosetSearch, name)

        def record(
            self: Any, *args: Any, name: str = name, query: Any = query, **options: Any
        ) -> Any:
            """Record the name and thread options of a batch query, and run it."""
            calls.append((name, options["num_threads"], options["split_work"]))
            return query(self, *args, **options)

        monkeypatch.setattr(BrouwerZimmermannCosetSearch, name, record)
    return calls


def test_searches_take_their_number_of_threads_when_they_are_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A search passes the threads of the tuning with which it was built to every batch query.

    Its split work grows with ``information_set_batch_exponent``, up to ``NO_WORK_LIMIT``.
    """
    calls = _record_thread_options(monkeypatch)
    matrix = ROUTED_PHYSICAL_STATES[0]
    batch = split_syndromes_into_words(list(range(32)), len(matrix))
    for num_threads, exponent in [(3, 4), (3, 5), (16, 62)]:
        with use_tuning(num_threads=num_threads, information_set_batch_exponent=exponent):
            search = CosetLeaderSearch(matrix)
        with use_tuning(num_threads=1):
            search.get_min_weights(batch, 2)
    assert [threads for _, threads, _ in calls] == [3, 3, 16]
    split_work = [work for _, _, work in calls]
    assert split_work[0] < split_work[1]
    assert split_work[2] == _threads.NO_WORK_LIMIT
    monkeypatch.setattr(_threads._num_threads, "NUM_THREADS", None)
    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 4)
    with use_tuning(num_threads=None):
        search = CosetLeaderSearch(matrix)
    search.find_first_not_within_max_weight(batch, len(matrix))
    assert calls[3][:2] == ("find_first_not_within_max_weight", 4)


def test_both_passes_of_css_states_take_the_threads_of_the_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For the CSS weight on a CSS state, the X and Z passes of batch queries take its threads.

    Every pass takes the threads and the split work that a search of Pauli errors takes.
    """
    calls = _record_thread_options(monkeypatch)
    matrix = get_stabilizer_matrix(
        ("XXXX___", "XX__XX_", "X_X_X_X", "ZZZZ___", "ZZ__ZZ_", "Z_Z_Z_Z", "ZZZZZZZ")
    )
    batch = split_syndromes_into_words(list(range(1 << len(matrix))), len(matrix))
    with use_tuning(num_threads=3):
        CosetLeaderSearch(matrix).get_min_weights(batch, 2)
        search = CosetLeaderSearch(matrix, css_weight=True)
    (single,) = calls
    search.get_min_weights(batch, 2)
    assert [name for name, _, _ in calls[1:]] == ["get_min_weights"] * 2
    search.find_first_not_within_max_weight(batch, 3)
    assert [name for name, _, _ in calls[3:]] == ["find_first_not_within_max_weight"] * 2
    assert {(threads, work) for _, threads, work in calls} == {(single[1], single[2])}
    assert single[1] == 3


def test_connected_cluster_batches_on_threads_answer_as_on_one_thread(
    pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Connected-cluster batch queries moved to threads answer as on one thread.

    The found index, its bound and error, and the weights are those of one thread, by physical
    weight and by CSS weight.  With chunks of one state, a batch moves to threads once its queries
    have visited a state for each thread.  The queries run with every array of their searches
    read-only: the threads of a batch share them, and so do the threads of callers that share a
    search.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    cases = [(get_random_css_state(6, 3, rng), True), (get_random_css_state(6, 2, rng), False)]
    thread_runs = 0
    run_work_items = _threads.run_work_items

    def run_counted(*args: Any, **kwargs: Any) -> None:
        """Count a run of work items on threads, and run it."""
        nonlocal thread_runs
        thread_runs += 1
        run_work_items(*args, **kwargs)

    monkeypatch.setattr(_threads, "run_work_items", run_counted)
    for matrix, css_weight in cases:
        num_qubits = len(matrix)
        minima = get_exhaustive_minima(matrix, css_weight)
        syndromes = rng.integers(1 << num_qubits, size=40)
        batch = split_syndromes_into_words(syndromes.tolist(), num_qubits)
        searches = []
        for num_threads in (1, 2, 3, 8):
            with use_tuning(num_threads=num_threads, information_set_batch_exponent=0):
                searches.append(
                    CosetLeaderSearch(matrix, css_weight=css_weight, method="connected_cluster")
                )
        # The arrays are marked before the queries start, so the slices that they take are
        # read-only too.
        with shared_arrays_read_only(*searches, batch) as arrays:
            for max_weight in range(num_qubits + 1):
                found = _find_each_not_within_max_weight(searches[-1], matrix, batch, max_weight)
                assert found == np.flatnonzero(minima[syndromes] > max_weight).tolist()
                for start in [0, *found]:
                    answers = [
                        search.find_first_not_within_max_weight(batch[start:], max_weight)
                        for search in searches
                    ]
                    assert all(answer == answers[0] for answer in answers)
                weights = [search.get_min_weights(batch, max_weight) for search in searches]
                assert all(np.array_equal(weight, weights[0]) for weight in weights)
                assert np.array_equal(weights[0], np.minimum(minima[syndromes], max_weight + 1))
        for index in range(len(searches)):
            assert any(each.path.startswith(f"roots[{index}].") for each in arrays)
    assert thread_runs


@pytest.mark.parametrize("later_first", [True, False])
def test_connected_cluster_batches_keep_the_least_index_whichever_block_ends_first(
    later_first: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch finds its least index without an error in reach, whichever block finds one first.

    With chunks of one state, the calling thread answers the first query, which visits states, and
    moves the rest of the batch to two threads, in blocks of one query each.  Every later class has
    no error of weight 2.  The range of the first block runs either after a later range has run, or
    while a later block that has started its queries waits for it.
    """
    later_ran, later_started, first_ran = (threading.Event() for _ in range(3))
    run_work_items = _threads.run_work_items
    get_bound = CosetLeaderSearch._get_batch_cluster_bound
    minima = get_exhaustive_minima(_THREADED_STATE, False)
    heavy = np.flatnonzero(minima > 2).tolist()
    syndromes = [3, heavy[0]] + [heavy[1 + row % (len(heavy) - 1)] for row in range(30)]
    assert minima[3] == 2

    def gated_run(num_items: int, num_threads: int, start_worker: Any, **kwargs: Any) -> None:
        """Run work items whose workers run their ranges through ``gated_range``."""

        def gated_start(stop: threading.Event) -> Any:
            """Start a worker whose ranges run through ``gated_range``."""
            run_range = start_worker(stop)

            def gated_range(start: int, end: int) -> None:
                """Run a range, the first after a later range ran or a later block started."""
                if start == 0:
                    assert (later_ran if later_first else later_started).wait(
                        timeout=THREAD_TIMEOUT
                    )
                run_range(start, end)
                (later_ran if start else first_ran).set()

            return gated_range

        run_work_items(num_items, num_threads, gated_start, **kwargs)

    def gated_bound(
        search: CosetLeaderSearch, syndrome: int, max_weight: int
    ) -> tuple[MinWeightBound, int]:
        """Get the bound of a query; later ones wait for the first range unless ``later_first``."""
        if not later_first and syndrome not in syndromes[:2]:
            later_started.set()
            assert first_ran.wait(timeout=THREAD_TIMEOUT)
        return get_bound(search, syndrome, max_weight)

    monkeypatch.setattr(_threads, "run_work_items", gated_run)
    monkeypatch.setattr(CosetLeaderSearch, "_get_batch_cluster_bound", gated_bound)
    batch = split_syndromes_into_words(syndromes, len(_THREADED_STATE))
    with use_tuning(num_threads=2, information_set_batch_exponent=0):
        search = CosetLeaderSearch(_THREADED_STATE, method="connected_cluster")
    index, bound = search.find_first_not_within_max_weight(batch, 2)
    assert later_ran.is_set() and first_ran.is_set()
    assert index == 1 and bound is not None and not bound.within_max_weight


def test_connected_cluster_batches_run_on_the_threads_of_their_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Costly connected-cluster batches run on the threads with which their search was built.

    With chunks of one state, a batch whose queries visit states moves to threads, and with the
    default chunks, this small batch stays on the calling thread.  So does a batch of syndromes of
    errors of weight at most one, whose queries look up their errors and visit no states.
    """
    calls: list[int] = []
    run_work_items = _threads.run_work_items

    def record(num_items: int, num_threads: int, *args: Any, **kwargs: Any) -> None:
        calls.append(num_threads)
        run_work_items(num_items, num_threads, *args, **kwargs)

    monkeypatch.setattr(_threads, "run_work_items", record)
    num_qubits = len(_THREADED_STATE)
    batch = split_syndromes_into_words([1 + row % 31 for row in range(32)], num_qubits)
    for exponent, splits in [(0, True), (15, False)]:
        calls.clear()
        with use_tuning(num_threads=3, information_set_batch_exponent=exponent):
            search = CosetLeaderSearch(_THREADED_STATE, method="connected_cluster")
        assert search.find_first_not_within_max_weight(batch, num_qubits) == (len(batch), None)
        num_calls = len(calls)
        assert bool(num_calls) == splits
        search.get_min_weights(batch, num_qubits)
        assert (len(calls) > num_calls) == splits
        assert set(calls) <= {3}

    # Syndromes of the single-qubit Paulis, which the queries look up.
    calls.clear()
    singles = [get_syndrome(_THREADED_STATE, 1 << bit) for bit in range(2 * num_qubits)]
    batch = split_syndromes_into_words(
        [singles[row % len(singles)] for row in range(32)], num_qubits
    )
    with use_tuning(num_threads=3, information_set_batch_exponent=0):
        search = CosetLeaderSearch(_THREADED_STATE, method="connected_cluster")
    assert search.find_first_not_within_max_weight(batch, 1) == (len(batch), None)
    search.get_min_weights(batch, 1)
    assert not calls


def test_connected_cluster_batches_by_css_weight_count_the_work_of_z_parts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A CSS-weight batch whose Z parts search moves to threads, though its X parts do no work.

    The errors of the batch have only Z parts, so the queries look up their X parts, whose
    syndromes are zero.
    """
    calls: list[int] = []
    run_work_items = _threads.run_work_items

    def record(num_items: int, num_threads: int, *args: Any, **kwargs: Any) -> None:
        calls.append(num_threads)
        run_work_items(num_items, num_threads, *args, **kwargs)

    monkeypatch.setattr(_threads, "run_work_items", record)
    matrix = get_random_css_state(8, 4, np.random.default_rng(0))
    num_qubits = len(matrix)
    rng = np.random.default_rng(1)
    z_errors = [
        sum(1 << int(bit) for bit in rng.choice(num_qubits, 3, replace=False)) for _ in range(32)
    ]
    syndromes = [get_syndrome(matrix, z_error << num_qubits) for z_error in z_errors]
    batch = split_syndromes_into_words(syndromes, num_qubits)
    with use_tuning(num_threads=2, information_set_batch_exponent=0):
        search = CosetLeaderSearch(matrix, css_weight=True, method="connected_cluster")
    assert search.find_first_not_within_max_weight(batch, num_qubits) == (len(batch), None)
    num_calls = len(calls)
    search.get_min_weights(batch, num_qubits)
    assert 0 < num_calls < len(calls) and set(calls) == {2}

    # Below the weights of some Z parts, their searches still do work.
    num_calls = len(calls)
    assert np.any(search.get_min_weights(batch, 2) == 3)
    assert num_calls < len(calls) and set(calls) == {2}
