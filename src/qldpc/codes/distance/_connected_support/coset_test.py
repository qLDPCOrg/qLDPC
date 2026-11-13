# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the connected-support coset searches of ``CosetLeaderSearch``."""

from __future__ import annotations

import inspect
import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from .._tuning import use_tuning
from ..conftest import (
    get_exhaustive_minima,
    get_graph_state,
    get_pauli_weight,
    get_random_css_state,
    get_syndrome,
)
from ..coset_leaders import CosetLeaderSearch
from . import coset


def test_connected_support_oracles_make_checks_sparser_with_their_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A connected-support oracle makes its checks sparser in the rounds of its tuning.

    The most rounds and the rounds without improvement of the tuning in force when the oracle is
    built reach ``get_sparser_rows``, which the test wraps to record them.  The state is a graph
    state with phases, which is not CSS, so the oracle has one search of Pauli errors.
    """
    calls: list[tuple[int, int]] = []
    get_sparser_rows = coset.get_sparser_rows
    signature = inspect.signature(get_sparser_rows)

    def record(*args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        arguments = signature.bind(*args, **kwargs).arguments
        calls.append((arguments["max_rounds"], arguments["rounds_without_improvement"]))
        return get_sparser_rows(*args, **kwargs)

    monkeypatch.setattr(coset, "get_sparser_rows", record)
    with use_tuning(max_sparsifying_rounds=5, sparsifying_rounds_without_improvement=2):
        CosetLeaderSearch(get_graph_state(2, [(0, 1)], phase=True), method="connected_support")
    assert calls == [(5, 2)]


def test_connected_support_answers_match_exhaustive_minima(pytestconfig: pytest.Config) -> None:
    """The connected-support engine answers every query exactly, by either measure of weight.

    Graph states with phases are not CSS, and are weighed by physical weight.  CSS states with
    no pure-X stabilizer, with only pure-X stabilizers, and with both are weighed by both measures,
    and the CSS weight splits their searches.  A syndrome whose lightest errors weigh at most one
    gets their weight as its bound.  Otherwise a query within ``max_weight`` gets 2, and one with no
    error within ``max_weight`` gets ``max(2, max_weight + 1)``.  The queries of batches, which try
    increasing weight limits, instead get the exact minimum and a lightest error when they are
    within ``max_weight``.  These exact bounds imply the inequalities that ``assert_exact_answers``
    checks.  A search of Pauli errors from a first weight limit above ``max_weight`` answers as a
    single query.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_qubits in range(1, 6):
        edges = [pair for pair in itertools.combinations(range(num_qubits), 2) if rng.integers(2)]
        graph_state = get_graph_state(num_qubits, edges, phase=True)
        num_x_stabilizers = [0, num_qubits, num_qubits // 2][num_qubits % 3]
        css_state = get_random_css_state(num_qubits, num_x_stabilizers, rng)
        for matrix, css_weight in [(graph_state, False), (css_state, False), (css_state, True)]:
            oracle = CosetLeaderSearch(matrix, css_weight=css_weight, method="connected_support")
            minima = get_exhaustive_minima(matrix, css_weight)
            for syndrome, max_weight in itertools.product(
                range(1 << num_qubits), range(num_qubits + 1)
            ):
                minimum = int(minima[syndrome])
                single = oracle.get_min_weight_bound(syndrome, max_weight)
                batched = oracle._get_batch_support_bound(syndrome, max_weight)
                if not css_weight:
                    search = oracle._connected_support_searches[0]
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
                        assert bound.lower_bound == weight == minimum
                    else:
                        assert bound.lower_bound == 2 and weight <= max_weight
