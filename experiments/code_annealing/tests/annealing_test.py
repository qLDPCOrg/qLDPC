# SPDX-License-Identifier: Apache-2.0

"""Exact scores and energies, scoring by guessing, the chain, and its configuration."""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Any
from unittest import mock

import numpy as np
import pytest

from qldpc import codes

from ..core import annealing as a
from ..core.proposals import propose, random_initial_codes
from .search_helpers import canonical_checks, counter_from_checks, steane_counter

####################################################################################################
# scores, energies, and ranks


def test_the_scores_of_the_steane_code_have_the_known_counts() -> None:
    # The minimum-weight X-only (and Z-only) logicals of the [[7,1,3]] Steane code are the 7 words
    # of weight 3 of the [7,4,3] Hamming code, which are not in its dual (they are the lines of the
    # Fano plane).
    assert a.score_code(steane_counter(), a.SearchConfig()) == a.Score(3, 7, 7)


def test_the_energy_of_the_steane_code_is_e_of_arxiv_2607_19574() -> None:
    # E = log(m p^d) / log(1/p) = log10(m) / 3 - 3 at p = 1e-3 and d = 3, where the multiplicity m
    # is the 7 + 7 = 14 X-only and Z-only operators.
    assert a.energy(a.Score(3, 7, 7), a.SearchConfig()) == pytest.approx(-2.6179573)


####################################################################################################
# scoring a proposal by guessing its distance


def test_counting_at_a_guess_gives_the_score_or_the_side_of_the_distance() -> None:
    steane = steane_counter()
    method: codes.LogicalCountMethod = "brouwer_zimmermann"
    assert a._score_at_guess(steane, 2, method) == "above"
    assert a._score_at_guess(steane, 3, method) == _brute_force_score(steane)
    assert a._score_at_guess(steane, 4, method) == "below"


def test_scoring_a_proposal_is_exact_whatever_the_guessed_distance_and_range() -> None:
    config = a.SearchConfig(method="brouwer_zimmermann", steps_per_epoch=1)
    steane = steane_counter()
    state = a.start_search(steane, config)
    for candidate in (steane, steane.deformed_cnot(0, 1)):  # of distances 3 and 2
        for guess, distance in itertools.product((1, 3, 6), repeat=2):
            state.recent_distances.clear()
            state.recent_distances.extend([guess] * 3)
            current_score = dataclasses.replace(state.current.score, distance=distance)
            state.current = dataclasses.replace(state.current, score=current_score)
            score = a._score_proposal(state, candidate)
            assert score == _brute_force_score(candidate)


def test_scoring_a_swel_proposal_whose_distance_moves_by_two_is_exact() -> None:
    config = a.SearchConfig(code_type="swel", gate_size=4, method="brouwer_zimmermann")
    steane, rng = steane_counter(), np.random.default_rng(0)
    proposals = (propose(steane, rng, code_type="swel", gate_size=4) for _ in range(100))
    neighbor = next(code for code in proposals if a.score_code(code, config).distance == 1)
    for current, candidate in ((steane, neighbor), (neighbor, steane)):
        state = a.start_search(current, dataclasses.replace(config, steps_per_epoch=1))
        assert a._score_proposal(state, candidate) == _brute_force_score(candidate)


####################################################################################################
# the scoring method


@pytest.mark.parametrize("code_type,gate_size", [("css", 2), ("css", 4), ("swel", 4)])
def test_the_scoring_method_never_changes_a_chain(code_type: Any, gate_size: int) -> None:
    outcomes = []
    for method in a.METHODS:
        config = a.SearchConfig(
            code_type=code_type,
            gate_size=gate_size,
            epochs=3,
            steps_per_epoch=16,
            method=method,
        )
        state = a.run_epochs(a.start_search(steane_counter(), config, seed=5))
        outcomes.append(_get_outcome(state))
    assert all(outcome == outcomes[0] for outcome in outcomes)


def test_switching_methods_between_epochs_never_changes_a_chain() -> None:
    config = a.SearchConfig(epochs=6, steps_per_epoch=16, method="auto")
    fixed = dataclasses.replace(config, method="brouwer_zimmermann")
    clock, methods = np.random.default_rng(1), []
    # A random clock makes the race of every epoch pick either method.
    with mock.patch.object(a, "_clock", side_effect=lambda: float(clock.random())):
        switched = a.start_search(steane_counter(), config, seed=9)
        a.run_epochs(switched, on_epoch=lambda state: methods.append(state.epoch_method))
    assert set(methods) == {"brouwer_zimmermann", "macwilliams"}
    unswitched = a.run_epochs(a.start_search(steane_counter(), fixed, seed=9))
    assert _get_outcome(switched) == _get_outcome(unswitched)


def _get_outcome(state: a.SearchState) -> tuple[Any, ...]:
    """The codes, scores, and history that a chain ends with."""
    roles = ("current", "lowest_energy", "greatest_distance")
    codes_and_scores = [
        (canonical_checks(getattr(state, role).counter), getattr(state, role).score)
        for role in roles
    ]
    return codes_and_scores, state.history


####################################################################################################
# the chain


def test_the_temperature_of_each_epoch_gives_its_target_acceptance_to_the_previous_jumps() -> None:
    # Without memory, the temperature T of an epoch is the one at which the uphill jumps of the
    # previous epoch (or of the pilot) have the mean acceptance exp(-jump / T) of the epoch's
    # target.  The targets fall geometrically from 0.04 to 0.0025 over three epochs: 0.04, 0.01,
    # and 0.0025.
    config = a.SearchConfig(
        epochs=3,
        steps_per_epoch=40,
        uphill_acceptance_start=0.04,
        uphill_acceptance_stop=0.0025,
        temperature_memory=0.0,
    )
    state = a.start_search(steane_counter().deformed_cnot(0, 1), config, seed=1)
    for target in (0.04, 0.01, 0.0025):
        jumps = state.energy_jumps[state.energy_jumps > 0]
        a.run_epochs(state, stop_after_epochs=1)
        temperature = state.history[-1].temperature
        assert a.LOG_TEMPERATURE_BOUNDS[0] < math.log(temperature) < a.LOG_TEMPERATURE_BOUNDS[1]
        assert np.exp(-jumps / temperature).mean() == pytest.approx(target, rel=1e-4)


def test_uphill_proposals_are_accepted_with_the_probability_of_the_metropolis_rule() -> None:
    # An uphill proposal of jump j is accepted with probability exp(-j / T), so the acceptance rate
    # of an epoch's uphill proposals is their mean probability, up to its binomial deviation.
    config = a.SearchConfig(
        method="brouwer_zimmermann",
        epochs=1,
        steps_per_epoch=200,
        uphill_acceptance_start=0.3,
        uphill_acceptance_stop=0.3,
    )
    state = a.run_epochs(a.start_search(next(random_initial_codes(12, 2, seed=0)), config))
    temperature, rate = state.history[-1].temperature, state.history[-1].uphill_acceptance_rate
    probabilities = np.exp(-state.energy_jumps[state.energy_jumps > 0] / temperature)
    deviation = np.sqrt(np.sum(probabilities * (1 - probabilities))) / len(probabilities)
    assert abs(rate - probabilities.mean()) < 4 * deviation


def test_the_pilot_records_the_best_of_all_its_proposals_and_keeps_the_first_of_equals() -> None:
    # At p = 1/4, a unit of distance is worth only four times as many operators, so the
    # lowest-energy code can have a lower distance than the greatest-distance code.
    config = a.SearchConfig(
        method="brouwer_zimmermann", epochs=1, steps_per_epoch=30, physical_error_rate=0.25
    )
    # Observe the codes that the pilot proposes, and score them by brute force.
    initial = steane_counter()
    candidates = [initial]

    def observe(*args: Any, **kwargs: Any) -> codes.CSSLogicalCounter:
        candidates.append(propose(*args, **kwargs))
        return candidates[-1]

    with mock.patch.object(a, "propose", side_effect=observe):
        state = a.start_search(initial, config, seed=5)
    assert len(candidates) == 31
    scores = [_brute_force_score(counter) for counter in candidates]
    lowest = min(range(31), key=lambda index: a.energy(scores[index], config))
    greatest = max(range(31), key=lambda index: _rank(scores[index]))
    assert lowest != greatest
    energies = [a.energy(score, config) for score in scores]
    np.testing.assert_allclose(state.energy_jumps, np.array(energies[1:]) - energies[0])
    for found, index in ((state.lowest_energy, lowest), (state.greatest_distance, greatest)):
        assert (canonical_checks(found.counter), found.score) == (
            canonical_checks(candidates[index]),
            scores[index],
        )


def test_a_rejected_proposal_can_set_a_record_and_is_reported() -> None:
    # Hardly any uphill move is accepted.  At p = 1/4, the starting code of distance 2, with 2
    # minimum-weight operators, has a lower energy than its neighbors of distance 1, with 1
    # operator, and of distance 3, with 14 operators.  So the chain stays at distance 2, and rejects
    # a neighbor of distance 3 that is nevertheless recorded.
    config = a.SearchConfig(
        method="brouwer_zimmermann",
        epochs=1,
        steps_per_epoch=400,
        uphill_acceptance_start=1e-6,
        uphill_acceptance_stop=1e-6,
        physical_error_rate=0.25,
    )
    state = a.start_search(steane_counter().deformed_cnot(0, 1), config, seed=1)
    state.greatest_distance = state.current
    reported: list[a.Score] = []
    a.run_epochs(state, on_greatest_distance=lambda s: reported.append(s.greatest_distance.score))
    assert state.current.score.distance == 2 and reported[-1].distance == 3
    jump = a.energy(reported[-1], config) - a.energy(state.current.score, config)
    assert np.isclose(state.energy_jumps, jump).any()  # the jump of the rejected proposal is kept
    assert state.greatest_distance.score == reported[-1]
    assert state.greatest_distance.score == _brute_force_score(state.greatest_distance.counter)


def test_a_record_is_reported_as_soon_as_a_proposal_improves_it() -> None:
    config = a.SearchConfig(
        method="brouwer_zimmermann",
        epochs=1,
        steps_per_epoch=100,
        uphill_acceptance_start=0.3,
        uphill_acceptance_stop=0.3,
    )
    state = a.start_search(next(random_initial_codes(12, 2, seed=0)), config, seed=0)
    state.greatest_distance = state.current  # so that the epoch, not the pilot, sets the record
    reported: list[a.Score] = []
    a.run_epochs(state, on_greatest_distance=lambda s: reported.append(s.greatest_distance.score))
    # A report at the end of the epoch alone would give at most one score.
    ranks = [_rank(score) for score in reported]
    assert len(ranks) >= 2 and ranks == sorted(set(ranks))
    assert reported[-1] == state.greatest_distance.score


####################################################################################################
# configuration


@pytest.mark.parametrize(
    "options",
    [
        {"physical_error_rate": 1.0},  # the energy would divide by log(1/p) = 0
        {"uphill_acceptance_stop": 1.0},  # a target acceptance that no temperature fits
        {"code_type": "swel", "gate_size": 2},  # a CNOT does not keep codes SWEL
    ],
    ids=["error rate", "target acceptance", "SWEL gate"],
)
def test_a_configuration_rejects_a_policy_that_would_corrupt_its_results(
    options: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        a.SearchConfig(**options)


@pytest.mark.parametrize(
    "checks_x,checks_z,code_type",
    [
        (np.eye(4), np.zeros((0, 4)), "css"),  # no logical qubit
        ([[1, 0, 0, 0]], [[1, 0, 0, 0]], "css"),  # a subsystem code
        ([[1, 0, 0, 0]], [[0, 1, 0, 0]], "swel"),  # not self-dual
        ([[1, 1, 1, 1]], [[1, 1, 1, 1]], "swel"),  # self-dual but not SWEL
    ],
)
def test_a_configuration_rejects_a_code_that_its_policy_cannot_use(
    checks_x: Any, checks_z: Any, code_type: Any
) -> None:
    config = a.SearchConfig(code_type=code_type, gate_size=2 + 2 * (code_type == "swel"))
    with pytest.raises(ValueError):
        config.validate_code(counter_from_checks(checks_x, checks_z))


####################################################################################################
# helper methods


def _brute_force_score(counter: codes.CSSLogicalCounter) -> a.Score:
    """The score of a code from the brute-force counts of the library."""
    counts = counter.count_logical_ops(method="brute_force")
    return a.Score.from_counts(counts)


def _rank(score: a.Score) -> tuple[int, int]:
    """The order of the greatest-distance record: a greater distance, then fewer operators."""
    return score.distance, -(score.count_x + score.count_z)
