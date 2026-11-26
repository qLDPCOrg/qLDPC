# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the connected-cluster coset searches of ``CosetLeaderSearch``."""

from __future__ import annotations

import inspect
import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import get_kernel, int_to_words

from .._cosets import xor_selected
from .._tuning import MAX_VALUE, use_tuning
from ..conftest import (
    get_binary_minima,
    get_binary_syndrome,
    get_exhaustive_minima,
    get_graph_state,
    get_pauli_weight,
    get_random_css_state,
    get_random_full_rank_matrix,
    get_syndrome,
    split_syndromes_into_words,
)
from ..coset_leaders import CosetLeaderSearch
from . import coset, search, sparse_checks


def test_connected_cluster_coset_searches_make_checks_sparser_with_their_rounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coset search passes its rounds of sparsifying to ``get_sparser_rows``."""
    calls: list[tuple[int, int]] = []
    get_sparser_rows = coset.get_sparser_rows
    signature = inspect.signature(get_sparser_rows)

    def record(*args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        arguments = signature.bind(*args, **kwargs).arguments
        calls.append((arguments["max_rounds"], arguments["rounds_without_improvement"]))
        return get_sparser_rows(*args, **kwargs)

    monkeypatch.setattr(coset, "get_sparser_rows", record)
    coset.ConnectedClusterCosetSearch(
        get_graph_state(2, [(0, 1)], phase=True),
        symplectic=True,
        max_rounds=5,
        rounds_without_improvement=2,
        restrict_viable_symbols=True,
        stabilizer_overlap_cuts=True,
    )
    assert calls == [(5, 2)]


def test_connected_cluster_queries_add_no_pair_sums(monkeypatch: pytest.MonkeyPatch) -> None:
    """The coset searches of ``CosetLeaderSearch`` search their sparser checks without pair sums.

    The checks of a repetition code have light sums of pairs, which distance searches would add,
    but queries search only the checks themselves (Sections 3.4 and 3.8 of
    ``connected_cluster.md``).
    """
    num_checks: list[int] = []
    search_supports = search._search_supports
    signature = inspect.signature(search_supports)

    def record(*args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.int64], ...]:
        num_checks.append(len(signature.bind(*args, **kwargs).arguments["check_starts"]) - 1)
        return search_supports(*args, **kwargs)

    monkeypatch.setattr(search, "_search_supports", record)
    repetition = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    coset_search = coset.ConnectedClusterCosetSearch(
        repetition,
        symplectic=False,
        max_rounds=0,
        rounds_without_improvement=1,
        restrict_viable_symbols=True,
        stabilizer_overlap_cuts=True,
    )
    assert len(sparse_checks.get_light_pair_sums(coset_search.checks, symplectic=False)[0])
    # The checks are a read-only view, since queries on several threads read them.
    assert not coset_search.checks.flags.writeable
    # The syndrome of the first and last checks has no word of weight one, so the query searches.
    assert coset_search.get_min_weight_bound(0b101, 2).within_max_weight
    assert num_checks == [len(coset_search.checks)]


def test_connected_cluster_queries_only_read_their_search(monkeypatch: pytest.MonkeyPatch) -> None:
    """A coset search tabulates the overlap cuts of its queries before its first query.

    It does so even before ``use_query_overlap_words`` gives it words, so that its queries, single
    or in batches, only read the search.
    """
    tabulated: list[object] = []
    get_overlap_cuts = search._get_overlap_cuts

    def record(*args: Any) -> Any:
        tabulated.append(args)
        return get_overlap_cuts(*args)

    monkeypatch.setattr(search, "_get_overlap_cuts", record)
    repetition = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    coset_search = coset.ConnectedClusterCosetSearch(
        repetition,
        symplectic=False,
        max_rounds=0,
        rounds_without_improvement=1,
        restrict_viable_symbols=True,
        stabilizer_overlap_cuts=True,
    )
    for words in ([], [[1, 1, 1, 1]]):
        if words:
            coset_search.use_query_overlap_words(words)
        num_tabulated = len(tabulated)
        assert num_tabulated
        for syndrome in (0b101, 0b111, 0b100):
            assert coset_search.get_min_weight_bound(syndrome, 2).within_max_weight
        batch = split_syndromes_into_words([0b101, 0b111, 0b100], 3)
        coset_search.find_first_not_within_max_weight(batch, 2)
        coset_search.get_min_weights(batch, 2, np.zeros(3, dtype=np.int64))
        assert len(tabulated) == num_tabulated


@pytest.mark.parametrize("restrict_viable_symbols", [True, False])
def test_connected_cluster_queries_read_their_restriction(
    monkeypatch: pytest.MonkeyPatch, restrict_viable_symbols: bool
) -> None:
    """A coset search passes ``restrict_viable_symbols`` to its single queries.

    ``test_connected_cluster_batches_search_as_single_queries`` checks its batch queries.
    """
    settings: list[bool] = []
    search_supports = search._search_supports
    signature = inspect.signature(search_supports)

    def record(*args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.int64], ...]:
        settings.append(signature.bind(*args, **kwargs).arguments["restrict_viable_symbols"])
        return search_supports(*args, **kwargs)

    # On this path graph state, the syndrome of X on the end qubits has no error of weight at most
    # one, so the query searches.
    monkeypatch.setattr(search, "_search_supports", record)
    matrix = get_graph_state(4, [(0, 1), (1, 2), (2, 3)])
    syndrome = get_syndrome(matrix, 0b1001)
    assert get_exhaustive_minima(matrix, False)[syndrome] == 2
    coset_search = coset.ConnectedClusterCosetSearch(
        matrix,
        symplectic=True,
        max_rounds=0,
        rounds_without_improvement=1,
        restrict_viable_symbols=restrict_viable_symbols,
        stabilizer_overlap_cuts=True,
    )
    assert coset_search.get_min_weight_bound(syndrome, 2).within_max_weight
    assert settings and set(settings) == {restrict_viable_symbols}


@pytest.mark.parametrize("cuts", [True, False])
def test_connected_cluster_queries_cut_by_their_overlap_words_if_switched_on(
    monkeypatch: pytest.MonkeyPatch, cuts: bool
) -> None:
    """A coset search given overlap words cuts its single queries exactly when its switch is on.

    ``test_connected_cluster_batches_search_as_single_queries`` checks that its batch queries cut
    as its single queries do.
    """
    num_cuts_per_query: list[int] = []
    search_supports = search._search_supports
    signature = inspect.signature(search_supports)

    def record(*args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.int64], ...]:
        num_cuts_per_query.append(len(signature.bind(*args, **kwargs).arguments["cut_capacities"]))
        return search_supports(*args, **kwargs)

    monkeypatch.setattr(search, "_search_supports", record)
    coset_search = coset.ConnectedClusterCosetSearch(
        get_random_css_state(6, 3, np.random.default_rng(0)),
        symplectic=True,
        max_rounds=0,
        rounds_without_improvement=1,
        restrict_viable_symbols=True,
        stabilizer_overlap_cuts=cuts,
    )
    coset_search.use_query_overlap_words(coset_search.checks)

    # Every syndrome that the table of light words lacks starts a query.
    for syndrome in range(1, 2 ** len(coset_search.checks)):
        coset_search.get_min_weight_bound(syndrome, 3)
    assert num_cuts_per_query
    assert all(bool(num_cuts) == cuts for num_cuts in num_cuts_per_query)


def test_connected_cluster_batches_search_as_single_queries() -> None:
    """Batch queries visit the states of single queries, with the restriction on and off.

    So they search with the restriction and the overlap cuts of the coset search, which are on.  On
    this CSS state, the lightest errors with the syndrome 57 weigh two, and the number of states
    that its query visits at limit two depends on the restriction.  Those with the syndrome 21 weigh
    three, so its batch query visits the states of single queries at the limits two and three.
    """
    matrix = get_random_css_state(6, 3, np.random.default_rng(0))
    syndrome = 57
    minima = get_exhaustive_minima(matrix, False)
    assert minima[syndrome] == 2 and minima[21] == 3
    counts = []
    for restrict_viable_symbols in (True, False):
        coset_search = coset.ConnectedClusterCosetSearch(
            matrix,
            symplectic=True,
            max_rounds=0,
            rounds_without_improvement=1,
            restrict_viable_symbols=restrict_viable_symbols,
            stabilizer_overlap_cuts=True,
        )
        coset_search.use_query_overlap_words(coset_search.checks)
        num_states = _count_single_states(coset_search, syndrome, range(2, 3))
        assert _count_batch_states(coset_search, syndrome, 2) == num_states
        counts.append(num_states)
    assert counts[0] != counts[1]
    num_states = _count_single_states(coset_search, 21, range(2, 4))
    assert _count_batch_states(coset_search, 21, 3) == num_states


def test_connected_cluster_batch_loop_answers_by_general_tables_within_its_work() -> None:
    """The compiled batch loop weighs cosets by fixed or general word tables, within its work.

    For the checks I64, a syndrome is the lightest word of its coset, so its bit count is its
    minimum weight, and a packed syndrome fills exactly one word.  An empty ``fixed_words`` selects
    the version of ``_search_supports`` for any number of words, which wider checks need, and the
    loop must not take its number of words from it.  The batch repeats a syndrome of one bit, so
    that each query must clear the mapped syndrome of the last.  The loop stops once its work
    reaches ``max_work``: three queries of the syndrome 3 that visit ``n`` states each stop after
    two at the work ``2 n``, and leave the weight of the third unwritten.
    """
    coset_search = coset.ConnectedClusterCosetSearch(
        np.eye(64, dtype=np.uint8),
        symplectic=False,
        max_rounds=0,
        rounds_without_improvement=1,
        restrict_viable_symbols=True,
        stabilizer_overlap_cuts=True,
    )
    tables = coset_search._get_batch_tables()
    syndromes = [1, 1, 3, 1 << 63, (1 << 63) | 1, 0, 7]
    batch = split_syndromes_into_words(syndromes, 64)
    floors = np.zeros(len(batch), dtype=np.int64)
    minima = np.array([syndrome.bit_count() for syndrome in syndromes])
    for batch_tables in (tables, tables[:4] + ((),) + tables[5:]):
        for max_weight in (1, 2):
            weights = np.full(len(batch), -1, dtype=np.int64)
            assert coset._find_min_weights(
                batch, max_weight, floors, MAX_VALUE, False, weights, batch_tables
            ) == (len(batch), False)
            assert np.array_equal(weights, np.minimum(minima, max_weight + 1))

    num_states = _count_single_states(coset_search, 3, range(2, 3))
    assert num_states
    batch = split_syndromes_into_words([3, 3, 3], 64)
    weights = np.full(3, -1, dtype=np.int64)
    assert coset._find_min_weights(
        batch, 2, floors[:3], 2 * num_states, False, weights, tables
    ) == (2, False)
    assert weights.tolist() == [2, 2, -1]


@pytest.mark.parametrize("cuts", [True, False])
@pytest.mark.parametrize("restrict_viable_symbols", [True, False])
def test_connected_cluster_answers_match_exhaustive_minima(
    pytestconfig: pytest.Config, restrict_viable_symbols: bool, cuts: bool
) -> None:
    """Queries get the bounds that ``ConnectedClusterCosetSearch`` states, with or without cuts.

    A single query searches only ``max_weight``.  The batch queries find the first coset without a
    word within ``max_weight``, with the bound of a single query, and the weight of the lightest
    words of each coset up to ``max_weight + 1``, or a weight within its floor if it has a word
    within the floor.
    """
    # The searches of Paulis check with the stabilizers of graph states with phases, which are not
    # CSS, and of CSS states, with no pure-X stabilizer, only pure-X stabilizers, or both.  The
    # searches of X or Z parts check with random independent rows.  The stabilizers, and a basis of
    # the kernel of the rows, are known words for the cuts.
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 6):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        graph_state = get_graph_state(num_qubits, edges, phase=True)
        num_x_stabilizers = [0, num_qubits, num_qubits // 2][num_qubits % 3]
        css_state = get_random_css_state(num_qubits, num_x_stabilizers, rng)
        rows = get_random_full_rank_matrix(rng, int(rng.integers(1, num_qubits + 1)), num_qubits)
        for checks, symplectic in [(graph_state, True), (css_state, True), (rows, False)]:
            coset_search = coset.ConnectedClusterCosetSearch(
                checks,
                symplectic=symplectic,
                max_rounds=10,
                rounds_without_improvement=2,
                restrict_viable_symbols=restrict_viable_symbols,
                stabilizer_overlap_cuts=cuts,
            )
            words = coset_search.checks if symplectic else get_kernel(checks)
            if len(words):
                coset_search.use_query_overlap_words(words)
            minima = (
                get_exhaustive_minima(checks, False) if symplectic else get_binary_minima(checks)
            )
            batch = split_syndromes_into_words(list(range(len(minima))), len(checks))
            for max_weight in range(num_qubits + 1):
                # Single queries.
                for syndrome, minimum in enumerate(minima.tolist()):
                    bound = coset_search.get_min_weight_bound(syndrome, max_weight)
                    word = bound.representative
                    if symplectic:
                        weight = get_pauli_weight(word, num_qubits, False)
                        assert get_syndrome(checks, word) == syndrome
                    else:
                        weight = word.bit_count()
                        bits = (word >> np.arange(num_qubits)) & 1
                        assert get_binary_syndrome(checks, bits) == syndrome
                    assert bound.within_max_weight == (minimum <= max_weight)
                    if not bound.within_max_weight:
                        expected_bound = minimum if minimum < 2 else max(2, max_weight + 1)
                        assert bound.lower_bound == expected_bound
                    elif minimum < 2:
                        assert bound.lower_bound == weight == minimum
                    else:
                        assert bound.lower_bound == 2 and weight <= max_weight

                # Batch queries, with random floors.
                floors = rng.integers(max_weight + 1, size=len(batch))
                weights = coset_search.get_min_weights(batch, max_weight, floors)
                within = minima <= floors
                assert np.all(weights[within] <= floors[within])
                assert np.array_equal(weights[~within], np.minimum(minima, max_weight + 1)[~within])
                above = np.flatnonzero(minima > max_weight).tolist()
                index, first_bound = coset_search.find_first_not_within_max_weight(
                    batch, max_weight
                )
                assert index == (above[0] if above else len(batch))
                if above:
                    assert first_bound == coset_search.get_min_weight_bound(index, max_weight)
                else:
                    assert first_bound is None


def test_connected_cluster_batches_in_blocks_answer_every_row_from_its_floor() -> None:
    """A batch that moves to blocks answers every row of each block from the row's own floor.

    For the checks I3, a syndrome is the lightest word of its coset, so its bit count is its minimum
    weight.  The search of the first syndrome of the batch does the work at which the rest of the
    batch moves to blocks, which up to two threads run.  The second syndrome starts a block, and its
    floor of zero gives its weight 2, where the floor 3 of the first or the last syndrome would give
    3.  With no split work, the rest of a batch moves to blocks after its first query, unless that
    query ends the batch, as that of the first syndrome does at ``max_weight`` two.
    """
    coset_search = coset.ConnectedClusterCosetSearch(
        np.eye(3, dtype=np.uint8),
        symplectic=False,
        max_rounds=0,
        rounds_without_improvement=1,
        restrict_viable_symbols=True,
        stabilizer_overlap_cuts=True,
    )
    batch = split_syndromes_into_words([0b111, 0b011, 0b111], 3)
    floors = np.array([3, 0, 3], dtype=np.int64)
    weights = coset_search.get_min_weights(batch, 3, floors, num_threads=2, split_work=1)
    assert weights[0] <= 3 and weights[1] == 2 and weights[2] <= 3

    index, bound = coset_search.find_first_not_within_max_weight(
        batch, 2, num_threads=2, split_work=0
    )
    assert index == 0 and bound is not None and bound.lower_bound == 3


def test_connected_cluster_batches_of_two_words_match_block_minima(
    pytestconfig: pytest.Config,
) -> None:
    """Batch queries of syndromes of two words agree with block minima and with single queries.

    The checks are 17 copies of a random graph state with phases on their own qubits, so the
    minimum weight of a coset is the sum of the minima of the syndromes of its copies, and the last
    copy has its four checks in the second 64-bit word.  Without rounds of sparsifying, each sparser
    check acts only on the qubits of the copy at its position.  The batch starts with each nonzero
    syndrome of the last copy alone, those of weight one first, so that at ``max_weight`` one the
    first syndrome without a word within it has weight two, and goes on with random syndromes of two
    copies.  The batches run on one to three threads in turn, with random split work and floors, and
    the first syndrome without a word within ``max_weight`` gets the bound of a single query.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    edges = [pair for pair in itertools.combinations(range(4), 2) if rng.integers(2)]
    block = get_graph_state(4, edges, phase=True)
    block_minima = get_exhaustive_minima(block, False)
    num_copies = 64 // len(block) + 1
    eye = np.eye(num_copies, dtype=np.uint8)
    checks = np.hstack([np.kron(eye, half) for half in np.split(block, 2, axis=1)])
    coset_search = coset.ConnectedClusterCosetSearch(
        checks,
        symplectic=True,
        max_rounds=0,
        rounds_without_improvement=1,
        restrict_viable_symbols=True,
        stabilizer_overlap_cuts=True,
    )
    copy_of_columns = np.tile(np.repeat(np.arange(num_copies), 4), 2)
    rows, columns = np.nonzero(coset_search.checks)
    assert np.array_equal(rows // len(block), copy_of_columns[columns])

    # Each query is a list of pairs of a copy and its syndrome.
    num_syndromes = 1 << len(block)
    by_weight = sorted(range(1, num_syndromes), key=lambda local: block_minima[local])
    queries = [[(num_copies - 1, local)] for local in by_weight]
    for _ in range(10):
        copies = rng.choice(num_copies, size=2, replace=False).tolist()
        queries.append(list(zip(copies, rng.integers(1, num_syndromes, size=2).tolist())))
    syndromes = [sum(local << (len(block) * copy) for copy, local in query) for query in queries]
    minima = np.array([sum(block_minima[local] for _, local in query) for query in queries])
    batch = split_syndromes_into_words(syndromes, len(checks))
    for max_weight in range(5):
        options = {"num_threads": 1 + max_weight % 3, "split_work": int(rng.integers(3))}
        floors = rng.integers(max_weight + 1, size=len(batch))
        weights = coset_search.get_min_weights(batch, max_weight, floors, **options)
        within = minima <= floors
        assert np.all(weights[within] <= floors[within])
        assert np.array_equal(weights[~within], np.minimum(minima, max_weight + 1)[~within])
        above = np.flatnonzero(minima > max_weight).tolist()
        index, bound = coset_search.find_first_not_within_max_weight(batch, max_weight, **options)
        assert index == (above[0] if above else len(batch))
        if above:
            assert bound == coset_search.get_min_weight_bound(syndromes[index], max_weight)
        else:
            assert bound is None


def test_connected_cluster_batches_match_exhaustive_minima() -> None:
    """Batch queries by one search, or by searches of X and Z parts, agree with exhaustive minima.

    Each batch holds every syndrome.  With chunks of one state, a batch on two threads moves to them
    once its queries have visited two states.
    """
    rng = np.random.default_rng(0)
    graph_state = get_graph_state(4, [(0, 1), (1, 2), (2, 3)], phase=True)
    for matrix, css_weight in [(graph_state, False), (get_random_css_state(5, 2, rng), True)]:
        num_qubits = len(matrix)
        minima = get_exhaustive_minima(matrix, css_weight)
        batch = split_syndromes_into_words(list(range(1 << num_qubits)), num_qubits)
        with use_tuning(num_threads=2, search_chunk_exponent=0):
            leaders = CosetLeaderSearch(matrix, css_weight=css_weight, method="connected_cluster")
        for max_weight in range(num_qubits + 1):
            weights = leaders.get_min_weights(batch, max_weight)
            assert np.array_equal(weights, np.minimum(minima, max_weight + 1))
            index, bound = leaders.find_first_not_within_max_weight(batch, max_weight)
            above = np.flatnonzero(minima > max_weight)
            assert index == (above[0] if len(above) else len(batch))
            assert (bound is None) == (not len(above))
            if bound is not None:
                assert not bound.within_max_weight and bound.lower_bound > max_weight
                assert get_syndrome(matrix, bound.representative) == index


####################################################################################################
# helpers


def _count_single_states(
    coset_search: coset.ConnectedClusterCosetSearch, syndrome: int, limits: range
) -> int:
    """The number of states that single searches of a syndrome visit at some weight limits.

    The syndrome must fit in one word, and its coset must have no word of weight at most one.
    """
    mapped = xor_selected(coset_search._syndrome_map_columns, syndrome)
    packed_syndrome = int_to_words(mapped, 1)
    return sum(
        coset_search._search.find_vector_with_packed_syndrome(packed_syndrome, limit)[1]
        for limit in limits
    )


def _count_batch_states(
    coset_search: coset.ConnectedClusterCosetSearch, syndrome: int, max_weight: int
) -> int:
    """The number of states that a batch query visits, as its compiled loop counts its work.

    The loop, ``_find_min_weights``, stops after a query once its work reaches ``max_work``, so of
    two copies of a syndrome that is not looked up and has a vector within ``max_weight``, it
    searches only the first exactly if the first visits at least ``max_work`` states.  A search for
    the largest such ``max_work`` finds that number.
    """
    batch = split_syndromes_into_words([syndrome, syndrome], len(coset_search.checks))
    floors = np.zeros(2, dtype=np.int64)
    weights = np.empty(2, dtype=np.int64)
    tables = coset_search._get_batch_tables()

    def searches_one(max_work: int) -> bool:
        """Whether the batch stops after the first copy within ``max_work``."""
        num_done, _ = coset._find_min_weights(
            batch, max_weight, floors, max_work, False, weights, tables
        )
        return bool(num_done == 1)

    low, high = 0, 1
    while searches_one(high):
        low, high = high, 2 * high
    while high - low > 1:
        middle = (low + high) // 2
        low, high = (middle, high) if searches_one(middle) else (low, middle)
    return low
