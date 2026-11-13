# SPDX-License-Identifier: Apache-2.0

"""Proposals are valid neighbors, and random initial codes are reproducible and valid."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ..core import proposals as p
from .search_helpers import canonical_checks

####################################################################################################
# initial codes


@pytest.mark.parametrize("self_dual", [False, True])
def test_a_seed_reproduces_its_random_initial_code(self_dual: bool) -> None:
    def first(seed: int) -> tuple[bytes, bytes]:
        return canonical_checks(next(p.random_initial_codes(8, 2, seed=seed, self_dual=self_dual)))

    assert first(3) == first(3) != first(4)


@pytest.mark.parametrize("self_dual", [False, True])
def test_a_random_initial_code_has_the_requested_size_and_code_type(self_dual: bool) -> None:
    counter = next(p.random_initial_codes(8, 2, seed=1, self_dual=self_dual))
    assert (counter.num_qubits, counter.dimension) == (8, 2)
    assert not (counter.checks_x @ counter.checks_z.T % 2).any()
    assert counter.is_self_dual == self_dual
    assert counter.code.is_swel or not self_dual


####################################################################################################
# local Clifford proposals


@pytest.mark.parametrize("code_type,gate_size", [("css", 2), ("css", 3), ("css", 4), ("swel", 4)])
def test_a_proposal_has_commuting_checks_the_same_size_and_the_code_type_of_its_search(
    code_type: Any, gate_size: int
) -> None:
    counter = next(p.random_initial_codes(8, 2, seed=1, self_dual=code_type == "swel"))
    before, rng = canonical_checks(counter), np.random.default_rng(0)
    for _ in range(25):
        neighbor = p.propose(counter, rng, code_type=code_type, gate_size=gate_size)
        assert (neighbor.num_qubits, neighbor.dimension) == (8, 2)
        assert not (neighbor.checks_x @ neighbor.checks_z.T % 2).any()
        assert code_type == "css" or (neighbor.is_self_dual and neighbor.code.is_swel)
    assert canonical_checks(counter) == before  # the code that is proposed from is not changed
