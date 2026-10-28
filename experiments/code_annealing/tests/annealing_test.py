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
from ..core.proposals import propose, random_initial_code
from .search_helpers import canonical_checks, counter_from_checks, steane_counter

####################################################################################################
# scores, energies and ranks


def test_the_scores_of_the_steane_code_have_the_known_counts() -> None:
    # The [[7,1,3]] Steane code has 7 X-only and 7 Z-only logicals of weight 3, and 21 Paulis.
    assert a.score_code(steane_counter(), a.SearchConfig()) == a.Score(3, 7, 7, None)
    pauli = a.SearchConfig(objective="pauli")
    assert a.score_code(steane_counter(), pauli) == a.Score(3, 7, 7, 21)


@pytest.mark.parametrize("self_dual", [False, True])
@pytest.mark.parametrize("objective", a.OBJECTIVES)
def test_a_score_equals_the_brute_force_counts(objective: Any, self_dual: bool) -> None:
    counter = random_initial_code(9, 1 if self_dual else 2, seed=1, self_dual=self_dual)
    assert counter.is_self_dual == self_dual
    score = a.score_code(counter, a.SearchConfig(objective=objective))
    assert score == _brute_force_score(counter, objective)


@pytest.mark.parametrize(
    "objective,distance", [("homogeneous", 3), ("pauli", 4)], ids=["odd distance", "even distance"]
)
def test_the_energy_is_e_l_of_arxiv_2607_19574(objective: Any, distance: int) -> None:
    # E_L: the log of the probability that a majority of the support of a minimum-weight
    # operator has errors, over log(1/p), where the multiplicity counts the operators of the
    # objective.
    p, multiplicity = 1e-3, 9 if objective == "pauli" else 7
    weight = math.ceil((distance + 1) / 2)
    probability = multiplicity * math.comb(distance, weight) * p**weight
    score = a.Score(distance, 3, 4, 9 if objective == "pauli" else None)
    config = a.SearchConfig(objective=objective, physical_error_rate=p)
    assert a.energy(score, config) == pytest.approx(math.log(probability) / math.log(1 / p))


def test_the_energy_is_defined_when_the_error_probability_underflows() -> None:
    # With 14 operators of weight 3, the majority weight is 2 and binomial(3, 2) = 3.
    config = a.SearchConfig(physical_error_rate=1e-320)
    expected = -2 - math.log(14 * 3) / math.log(1e-320)
    assert a.energy(a.Score(3, 7, 7, None), config) == pytest.approx(expected)


@pytest.mark.parametrize(
    "objective,score",
    [
        ("homogeneous", a.Score(0, 1, 1, None)),
        ("homogeneous", a.Score(3, 0, 0, None)),
        ("pauli", a.Score(3, 7, 7, None)),
    ],
    ids=["no distance", "no operators", "no Pauli count"],
)
def test_a_score_without_a_distance_or_a_count_has_no_energy_or_rank(
    objective: Any, score: a.Score
) -> None:
    config = a.SearchConfig(objective=objective)
    with pytest.raises(ValueError):
        a.energy(score, config)
    with pytest.raises(ValueError):
        a.score_rank(score, config)


@pytest.mark.parametrize("objective", a.OBJECTIVES)
def test_codes_rank_by_distance_and_then_by_fewer_operators_of_the_objective(
    objective: Any,
) -> None:
    config = a.SearchConfig(objective=objective)
    fewer, more = a.Score(3, 2, 2, 20), a.Score(3, 3, 3, 10)  # fewer X and Z operators, more Paulis
    if objective == "pauli":
        fewer, more = more, fewer
    assert a.score_rank(a.Score(4, 90, 90, 90), config) > a.score_rank(fewer, config)
    assert a.score_rank(fewer, config) > a.score_rank(more, config)


def test_the_lowest_energy_code_of_a_sequence_is_found_and_the_first_of_ties_wins() -> None:
    poor, steane, twin = _poor_counter(), steane_counter(), steane_counter()
    assert a.get_lowest_energy_code([poor, steane, twin, poor], a.SearchConfig()) is steane


def test_there_is_no_lowest_energy_code_of_no_codes() -> None:
    with pytest.raises(ValueError):
        a.get_lowest_energy_code([], a.SearchConfig())


####################################################################################################
# scoring a proposal by guessing its distance


@pytest.mark.parametrize("objective", a.OBJECTIVES)
def test_counting_at_a_guess_gives_the_score_or_the_side_of_the_distance(objective: Any) -> None:
    steane, config = steane_counter(), a.SearchConfig(objective=objective)
    method: codes.LogicalCountMethod = "brouwer_zimmermann"
    assert a._score_at_guess(steane, 2, config, method) == "above"
    assert a._score_at_guess(steane, 3, config, method) == _brute_force_score(steane, objective)
    assert a._score_at_guess(steane, 4, config, method) == "below"


@pytest.mark.parametrize("objective", a.OBJECTIVES)
def test_scoring_a_proposal_is_exact_whatever_the_guessed_distance_and_range(
    objective: Any,
) -> None:
    config = a.SearchConfig(method="brouwer_zimmermann", objective=objective, steps_per_epoch=1)
    steane = steane_counter()
    state = a.start_search(steane, config)
    for candidate in (steane, steane.deformed_cnot(0, 1)):  # of distances 3 and 2
        for guess, distance in itertools.product((1, 3, 6), repeat=2):
            state.recent_distances.clear()
            state.recent_distances.extend([guess] * 3)
            score = dataclasses.replace(state.current.score, distance=distance)
            state.current = dataclasses.replace(state.current, score=score)
            score = a._score_proposal(state, candidate)
            assert score == _brute_force_score(candidate, objective)


def test_scoring_a_swel_proposal_whose_distance_moves_by_two_is_exact() -> None:
    config = a.SearchConfig(code_type="swel", gate_size=4, method="brouwer_zimmermann")
    steane, rng = steane_counter(), np.random.default_rng(0)
    proposals = (propose(steane, rng, code_type="swel", gate_size=4) for _ in range(100))
    neighbor = next(code for code in proposals if a.score_code(code, config).distance == 1)
    for current, candidate in ((steane, neighbor), (neighbor, steane)):
        state = a.start_search(current, dataclasses.replace(config, steps_per_epoch=1))
        assert a._score_proposal(state, candidate) == _brute_force_score(candidate)


@pytest.mark.parametrize("code_type,gate_size", [("css", 2), ("css", 3), ("css", 4), ("swel", 4)])
def test_one_proposal_changes_the_distance_by_at_most_its_reach(
    code_type: Any, gate_size: int
) -> None:
    config = a.SearchConfig(code_type=code_type, gate_size=gate_size)
    rng, steane, changes = np.random.default_rng(0), steane_counter(), set()
    for _ in range(20):
        first = propose(steane, rng, code_type=code_type, gate_size=gate_size)
        second = propose(first, rng, code_type=code_type, gate_size=gate_size)
        distances = [a.score_code(code, config).distance for code in (steane, first, second)]
        changes |= {abs(distances[1] - distances[0]), abs(distances[2] - distances[1])}
    assert 0 < max(changes) <= a._get_distance_reach(config)


####################################################################################################
# the scoring method


@pytest.mark.parametrize(
    "code_type,gate_size,objective",
    [("css", 2, "homogeneous"), ("css", 4, "pauli"), ("swel", 4, "pauli")],
)
def test_the_scoring_method_never_changes_a_chain(
    code_type: Any, gate_size: int, objective: Any
) -> None:
    outcomes = []
    for method in a.METHODS:
        if (code_type, objective, method) == ("css", "pauli", "macwilliams"):
            continue  # the MacWilliams method counts Paulis only of self-dual codes
        config = a.SearchConfig(
            code_type=code_type,
            gate_size=gate_size,
            objective=objective,
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


####################################################################################################
# the chain


def test_the_temperature_of_an_epoch_follows_the_fit_to_the_previous_epoch() -> None:
    config = a.SearchConfig(
        epochs=3,
        steps_per_epoch=40,
        uphill_acceptance_start=0.04,
        uphill_acceptance_stop=0.0025,
        temperature_memory=0.25,
    )
    state = a.start_search(steane_counter().deformed_cnot(0, 1), config, seed=1)
    log_temperature = state.log_temperature
    assert log_temperature == pytest.approx(a._fit_log_temperature(state.energy_jumps, 0.04))
    for epoch in range(3):
        target = 0.04 * (0.0025 / 0.04) ** (epoch / 2)  # the targets fall geometrically
        jumps = state.energy_jumps.copy()
        assert jumps.max() > 0 and a.LOG_TEMPERATURE_BOUNDS[0] < log_temperature
        a.run_epochs(state, stop_after_epochs=1)
        log_temperature = 0.25 * log_temperature + 0.75 * a._fit_log_temperature(jumps, target)
        assert state.log_temperature == pytest.approx(log_temperature)
        assert state.history[-1].temperature == pytest.approx(math.exp(log_temperature))


def test_the_fitted_temperature_gives_the_target_mean_uphill_acceptance() -> None:
    uphill = np.array([0.01, 0.02, 0.04, 0.1])
    log_temperature = a._fit_log_temperature([-0.5, 0, *uphill], 0.1)  # downhill jumps are ignored
    assert np.exp(-uphill / math.exp(log_temperature)).mean() == pytest.approx(0.1, abs=1e-7)


@pytest.mark.parametrize(
    "jumps,bound",
    [([-1, 0], 1), ([1e-12], 0), ([1e6], 1)],
    ids=["no uphill jump", "tiny jump", "huge jump"],
)
def test_the_fitted_log_temperature_is_bounded(jumps: list[float], bound: int) -> None:
    fit = a._fit_log_temperature(jumps, 0.5)
    assert fit == pytest.approx(a.LOG_TEMPERATURE_BOUNDS[bound], abs=1e-6)


def test_the_metropolis_rule_accepts_downhill_moves_and_uphill_moves_by_their_probability() -> None:
    assert a._accept_move(-1, 0.01, 0.99) and a._accept_move(0, 0.01, 0.99)
    probability = math.exp(-0.5)
    assert a._accept_move(0.5, 1, 0.99 * probability)
    assert not a._accept_move(0.5, 1, 1.01 * probability)
    assert a._accept_move(1000, 0.001, 0)  # a zero uniform accepts a probability that underflows
    assert not a._accept_move(1000, 0.001, 0.1)


def test_stopping_after_some_epochs_and_continuing_equals_one_run() -> None:
    config = a.SearchConfig(epochs=3, steps_per_epoch=8)
    full = a.run_epochs(a.start_search(steane_counter(), config, seed=71))
    split = a.run_epochs(a.start_search(steane_counter(), config, seed=71), stop_after_epochs=1)
    assert split.completed_epochs == 1
    assert a.run_epochs(split) is split
    assert _get_outcome(split) == _get_outcome(full)


def test_the_pilot_records_the_best_of_all_its_proposals_and_keeps_the_first_of_equals() -> None:
    config = a.SearchConfig(method="brouwer_zimmermann", epochs=1, steps_per_epoch=30)
    # The same random generator as the pilot of a search with seed 5.
    initial, rng = steane_counter(), np.random.default_rng(5)
    candidates = [initial]
    candidates += [propose(initial, rng, code_type="css", gate_size=2) for _ in range(30)]
    scores = [_brute_force_score(counter) for counter in candidates]
    lowest = min(range(31), key=lambda index: a.energy(scores[index], config))
    greatest = max(range(31), key=lambda index: a.score_rank(scores[index], config))
    state = a.start_search(initial, config, seed=5)
    assert lowest != greatest
    energies = [a.energy(score, config) for score in scores]
    np.testing.assert_allclose(state.energy_jumps, np.array(energies[1:]) - energies[0])
    for found, index in ((state.lowest_energy, lowest), (state.greatest_distance, greatest)):
        assert (canonical_checks(found.counter), found.score) == (
            canonical_checks(candidates[index]),
            scores[index],
        )


def test_a_record_changes_only_when_a_proposal_strictly_improves_it() -> None:
    config = a.SearchConfig(steps_per_epoch=1)
    steane, twin, lower = steane_counter(), steane_counter(), steane_counter().deformed_cnot(0, 1)
    state = a.start_search(steane, config)
    steane_score, lower_score = a.score_code(steane, config), a.score_code(lower, config)
    assert lower_score.distance < steane_score.distance
    assert a.energy(lower_score, config) < a.energy(steane_score, config)
    state.lowest_energy = state.greatest_distance = a.ScoredCode(lower, lower_score)
    assert a._update_optima(state, a.ScoredCode(steane, steane_score)) == (
        False,
        True,
    )  # a greater d
    assert a._update_optima(state, a.ScoredCode(twin, steane_score)) == (False, False)  # equal
    assert state.lowest_energy.counter is lower and state.greatest_distance.counter is steane
    best = a.ScoredCode(twin, a.Score(4, 1, 1, None))
    assert a._update_optima(state, best) == (True, True)
    assert state.lowest_energy is state.greatest_distance is best


def test_a_rejected_proposal_can_set_a_record_and_is_reported() -> None:
    # Hardly any uphill move is accepted, so the chain stays at a code of distance 2 and has a
    # neighbor of distance 3 that it rejects, since the energy of that neighbor is higher.
    config = a.SearchConfig(
        method="brouwer_zimmermann",
        epochs=1,
        steps_per_epoch=400,
        uphill_acceptance_start=1e-6,
        uphill_acceptance_stop=1e-6,
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


####################################################################################################
# configuration


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


def _brute_force_score(counter: codes.CSSLogicalCounter, objective: Any = "homogeneous") -> a.Score:
    """The score of a code from the brute-force counts of the library."""
    counts = counter.count_logical_ops(method="brute_force", count_pauli=objective == "pauli")
    return a.Score(counts.weight, counts.count_x, counts.count_z, counts.count_pauli)


def _poor_counter() -> codes.CSSLogicalCounter:
    """A code of seven qubits and distance 1."""
    return counter_from_checks(np.zeros((0, 7)), np.eye(7)[1:])


def _get_outcome(state: a.SearchState) -> tuple[Any, ...]:
    """The codes, scores and history that a chain ends with."""
    roles = ("current", "lowest_energy", "greatest_distance")
    codes_and_scores = [
        (canonical_checks(getattr(state, role).counter), getattr(state, role).score)
        for role in roles
    ]
    return codes_and_scores, state.history
