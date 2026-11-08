# SPDX-License-Identifier: Apache-2.0

"""Proposals are valid neighbors, and random initial codes are reproducible and filtered."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pytest

from qldpc import codes

from ..core import proposals as p
from .search_helpers import canonical_checks

####################################################################################################
# initial codes


@pytest.mark.parametrize("self_dual", [False, True])
def test_a_seed_reproduces_its_random_initial_code(self_dual: bool) -> None:
    first = p.random_initial_code(8, 2, seed=3, self_dual=self_dual)
    assert canonical_checks(first) == canonical_checks(
        p.random_initial_code(8, 2, seed=3, self_dual=self_dual)
    )
    assert canonical_checks(first) != canonical_checks(
        p.random_initial_code(8, 2, seed=4, self_dual=self_dual)
    )


@pytest.mark.parametrize("self_dual", [False, True])
def test_a_random_initial_code_has_the_requested_size_and_code_type(self_dual: bool) -> None:
    counter = p.random_initial_code(8, 2, seed=1, self_dual=self_dual)
    assert (counter.num_qubits, counter.dimension) == (8, 2)
    assert not (counter.checks_x @ counter.checks_z.T % 2).any()
    assert counter.is_self_dual == self_dual
    assert counter.code.is_swel or not self_dual


def test_a_random_initial_code_has_at_least_the_minimum_distance() -> None:
    def distance(counter: codes.CSSLogicalCounter) -> int:
        return counter.count_logical_ops(method="brute_force").weight

    # Find a seed whose first code is rejected, so that the search for another code is exercised.
    seeds = range(100)
    seed = next(
        seed for seed in seeds if distance(next(p.random_initial_codes(7, 1, seed=seed))) < 2
    )
    assert distance(p.random_initial_code(7, 1, seed=seed, min_distance=2)) >= 2


def test_the_stream_of_initial_codes_does_not_replay_the_random_numbers_of_the_search() -> None:
    # The search seeds its generator with the same integer as the stream of initial codes.
    replayed = p._scramble_css(
        *p._get_trivial_code(9, 1, self_dual=False), np.random.default_rng(4)
    )
    stream = p.random_initial_codes(9, 1, seed=4)
    assert canonical_checks(next(stream)) != canonical_checks(replayed)


####################################################################################################
# local Clifford proposals


@pytest.mark.parametrize("code_type,gate_size", [("css", 2), ("css", 3), ("css", 4), ("swel", 4)])
def test_a_proposal_is_a_neighbor_with_commuting_checks_and_the_same_dimension(
    code_type: Any, gate_size: int
) -> None:
    counter = p.random_initial_code(8, 2, seed=1, self_dual=code_type == "swel")
    before, rng = canonical_checks(counter), np.random.default_rng(0)
    for _ in range(25):
        neighbor = p.propose(counter, rng, code_type=code_type, gate_size=gate_size)
        assert (neighbor.num_qubits, neighbor.dimension) == (8, 2)
        assert not (neighbor.checks_x @ neighbor.checks_z.T % 2).any()
        assert code_type == "css" or (neighbor.is_self_dual and neighbor.code.is_swel)
    assert canonical_checks(counter) == before  # the code that is proposed from is not changed


def test_three_qubit_gates_have_six_classes_with_frequencies_2_2_2_1_1_1() -> None:
    # The paper proposes only CNOTs for CSS codes, and these frequencies are the specification in
    # the docstring of ``_sample_three_qubit_gate``.  Each class is drawn 1000 or 2000 times in
    # expectation, with a standard deviation of about 40.
    rng = np.random.default_rng(0)
    draws = Counter(p._sample_three_qubit_gate(rng).tobytes() for _ in range(9000))
    assert sorted(round(count / 1000) for count in draws.values()) == [1, 1, 1, 2, 2, 2]
