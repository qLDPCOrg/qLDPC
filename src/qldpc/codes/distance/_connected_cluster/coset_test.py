# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the connected-cluster coset searches of ``CosetLeaderSearch``."""

from __future__ import annotations

import inspect
import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import get_kernel

from .._tuning import use_tuning
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

    It does so even before ``use_query_overlap_words`` gives it words, so that its queries only read
    the search.
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
        assert len(tabulated) == num_tabulated


@pytest.mark.parametrize("restrict_viable_symbols", [True, False])
def test_connected_cluster_queries_read_their_restriction(
    monkeypatch: pytest.MonkeyPatch, restrict_viable_symbols: bool
) -> None:
    """A coset search passes ``restrict_viable_symbols`` to its queries."""
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
    """A coset search given overlap words cuts its queries exactly when its switch is on."""
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


@pytest.mark.parametrize("cuts", [True, False])
@pytest.mark.parametrize("restrict_viable_symbols", [True, False])
def test_connected_cluster_answers_match_exhaustive_minima(
    pytestconfig: pytest.Config, restrict_viable_symbols: bool, cuts: bool
) -> None:
    """Queries get the bounds that ``ConnectedClusterCosetSearch`` states, with or without cuts.

    A single query searches only ``max_weight``, as does a query whose first limit is above it.  A
    query whose first limit is two finds the weight of the lightest words of each coset that has a
    word within ``max_weight``.
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
            for syndrome, max_weight in itertools.product(
                range(1 << len(checks)), range(num_qubits + 1)
            ):
                minimum = minima[syndrome]
                single = coset_search.get_min_weight_bound(syndrome, max_weight)
                from_two, _ = coset_search._get_min_weight_bound_and_states(syndrome, max_weight, 2)
                late, _ = coset_search._get_min_weight_bound_and_states(
                    syndrome, max_weight, max_weight + 1
                )
                assert late == single
                for bound, exact in [(single, False), (from_two, True)]:
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
                    elif exact or minimum < 2:
                        # Queries from limit two, and words of weight at most one, are exact
                        # within max_weight.
                        assert bound.lower_bound == weight == minimum
                    else:
                        assert bound.lower_bound == 2 and weight <= max_weight


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
