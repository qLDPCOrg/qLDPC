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
from . import coset, search


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

    def record(*args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.int64], int]:
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


@pytest.mark.parametrize("restrict_viable_symbols", [True, False])
def test_connected_cluster_answers_match_exhaustive_minima(
    pytestconfig: pytest.Config, restrict_viable_symbols: bool, tune: Tune
) -> None:
    """Queries get the bounds that ``ConnectedClusterCosetSearch`` states."""
    tune(restrict_viable_symbols=restrict_viable_symbols)
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
                batched = oracle._get_batch_cluster_bound(syndrome, max_weight)
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
