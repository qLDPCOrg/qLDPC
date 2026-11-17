# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the connected-cluster coset searches of ``CosetLeaderSearch``."""

from __future__ import annotations

import inspect
import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from ...conftest import Tune
from .._tuning import use_tuning
from ..conftest import (
    get_exhaustive_minima,
    get_graph_state,
    get_pauli_weight,
    get_random_css_state,
    get_syndrome,
)
from ..coset_leaders import CosetLeaderSearch
from . import coset, search, sparse_checks


def test_connected_cluster_oracles_make_checks_sparser_with_their_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``CosetLeaderSearch`` passes the sparsifying rounds of its tuning to ``get_sparser_rows``."""
    calls: list[tuple[int, int]] = []
    get_sparser_rows = coset.get_sparser_rows
    signature = inspect.signature(get_sparser_rows)

    def record(*args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        arguments = signature.bind(*args, **kwargs).arguments
        calls.append((arguments["max_rounds"], arguments["rounds_without_improvement"]))
        return get_sparser_rows(*args, **kwargs)

    # Without ``css_weight``, the ``CosetLeaderSearch`` has one coset search.
    monkeypatch.setattr(coset, "get_sparser_rows", record)
    with use_tuning(max_sparsifying_rounds=5, sparsifying_rounds_without_improvement=2):
        CosetLeaderSearch(get_graph_state(2, [(0, 1)], phase=True), method="connected_cluster")
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

    def record(
        *args: Any, **kwargs: Any
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
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
    # The syndrome of the first and last checks has no word of weight one, so the query searches.
    assert coset_search.get_min_weight_bound(0b101, 2).within_max_weight
    assert num_checks == [len(coset_search.checks)]


@pytest.mark.parametrize(
    ("changes", "expected"), [({}, True), ({"restrict_viable_symbols": False}, False)]
)
def test_connected_cluster_queries_read_the_restriction_of_their_tuning(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, bool], expected: bool
) -> None:
    """A ``CosetLeaderSearch`` passes ``restrict_viable_symbols`` of its tuning to its queries."""
    settings: list[bool] = []
    search_supports = search._search_supports
    signature = inspect.signature(search_supports)

    def record(
        *args: Any, **kwargs: Any
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        settings.append(signature.bind(*args, **kwargs).arguments["restrict_viable_symbols"])
        return search_supports(*args, **kwargs)

    # On this path graph state, the syndrome of X on the end qubits has no error of weight at most
    # one, so the query searches.
    monkeypatch.setattr(search, "_search_supports", record)
    matrix = get_graph_state(4, [(0, 1), (1, 2), (2, 3)])
    syndrome = get_syndrome(matrix, 0b1001)
    assert get_exhaustive_minima(matrix, False)[syndrome] == 2
    with use_tuning(**changes):
        oracle = CosetLeaderSearch(matrix, method="connected_cluster")
    assert oracle.get_min_weight_bound(syndrome, 2).within_max_weight
    assert settings and set(settings) == {expected}


@pytest.mark.parametrize("cuts", [True, False])
def test_connected_cluster_queries_cut_by_the_stabilizers_of_their_state(
    monkeypatch: pytest.MonkeyPatch, cuts: bool
) -> None:
    """Each coset search gets the stabilizers in its kernel, which cut its queries if switched on.

    A Pauli search gets its own sparser stabilizers, and the search of X parts the X parts of the
    pure-X stabilizers, which the search of Z parts checks with, and vice versa (Section 3.8 of
    ``connected_cluster.md``).
    """
    received: dict[int, set[bytes]] = {}
    use_words = coset.ConnectedClusterCosetSearch.use_query_overlap_words

    def record_words(self: coset.ConnectedClusterCosetSearch, words: npt.ArrayLike) -> None:
        rows = np.asarray(words, dtype=np.uint8)
        received.setdefault(id(self), set()).update(row.tobytes() for row in rows)
        use_words(self, words)

    num_cuts_per_query: list[int] = []
    search_supports = search._search_supports
    signature = inspect.signature(search_supports)

    def record_cuts(
        *args: Any, **kwargs: Any
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        num_cuts_per_query.append(len(signature.bind(*args, **kwargs).arguments["cut_capacities"]))
        return search_supports(*args, **kwargs)

    monkeypatch.setattr(coset.ConnectedClusterCosetSearch, "use_query_overlap_words", record_words)
    monkeypatch.setattr(search, "_search_supports", record_cuts)
    css_state = get_random_css_state(6, 3, np.random.default_rng(0))
    with use_tuning(stabilizer_overlap_cuts=cuts):
        pauli_oracle = CosetLeaderSearch(css_state, method="connected_cluster")
        css_oracle = CosetLeaderSearch(css_state, css_weight=True, method="connected_cluster")
    (pauli_search,) = pauli_oracle._connected_cluster_searches
    x_part_search, z_part_search = css_oracle._connected_cluster_searches
    pairs = [
        (pauli_search, pauli_search),
        (x_part_search, z_part_search),
        (z_part_search, x_part_search),
    ]
    for searched, checking in pairs:
        expected = {np.asarray(row, dtype=np.uint8).tobytes() for row in checking.checks}
        assert received[id(searched)] == expected

        # Every syndrome that the table of light words lacks starts a query.
        num_cuts_per_query.clear()
        for syndrome in range(1, 2 ** len(searched.checks)):
            searched.get_min_weight_bound(syndrome, 3)
        assert num_cuts_per_query
        assert all(bool(num_cuts) == cuts for num_cuts in num_cuts_per_query)


@pytest.mark.parametrize("cuts", [True, False])
@pytest.mark.parametrize("restrict_viable_symbols", [True, False])
def test_connected_cluster_answers_match_exhaustive_minima(
    pytestconfig: pytest.Config, restrict_viable_symbols: bool, cuts: bool, tune: Tune
) -> None:
    """Queries get the bounds that ``ConnectedClusterCosetSearch`` states, with or without cuts."""
    tune(restrict_viable_symbols=restrict_viable_symbols, stabilizer_overlap_cuts=cuts)
    # Graph states with phases are not CSS, and the connected-cluster method weighs errors on them
    # only by physical weight.  CSS states, with no pure-X stabilizer, only pure-X stabilizers, or
    # both, are weighed by both measures.
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 6):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        graph_state = get_graph_state(num_qubits, edges, phase=True)
        num_x_stabilizers = [0, num_qubits, num_qubits // 2][num_qubits % 3]
        css_state = get_random_css_state(num_qubits, num_x_stabilizers, rng)
        for matrix, css_weight in [(graph_state, False), (css_state, False), (css_state, True)]:
            oracle = CosetLeaderSearch(matrix, css_weight=css_weight, method="connected_cluster")
            minima = get_exhaustive_minima(matrix, css_weight)
            for syndrome, max_weight in itertools.product(
                range(1 << num_qubits), range(num_qubits + 1)
            ):
                minimum = int(minima[syndrome])
                single = oracle.get_min_weight_bound(syndrome, max_weight)
                batched, _ = oracle._get_batch_cluster_bound(syndrome, max_weight)
                # A first limit above max_weight answers as a single query does.
                if not css_weight:
                    search = oracle._connected_cluster_searches[0]
                    late = search.get_min_weight_bound(
                        syndrome, max_weight, first_limit=max_weight + 1
                    )
                    assert late == single
                for bound, exact in [(single, False), (batched, True)]:
                    weight = get_pauli_weight(bound.representative, num_qubits, css_weight)
                    assert get_syndrome(matrix, bound.representative) == syndrome
                    assert bound.within_max_weight == (minimum <= max_weight)
                    if not bound.within_max_weight:
                        expected_bound = minimum if minimum < 2 else max(2, max_weight + 1)
                        assert bound.lower_bound == expected_bound
                    elif exact or minimum < 2:
                        # Batches, from limit two, and words of weight at most one are exact within
                        # max_weight.
                        assert bound.lower_bound == weight == minimum
                    else:
                        assert bound.lower_bound == 2 and weight <= max_weight
