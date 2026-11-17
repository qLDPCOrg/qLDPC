# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the certificates of basis families and the plans of their steps."""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import numpy.typing as npt
import pytest
import scipy.optimize

from .._tuning import DEFAULT_TUNING
from . import certificates

DENOMINATOR = DEFAULT_TUNING.lp_dual_max_denominator


def test_certificate_follows_its_definition(pytestconfig: pytest.Config) -> None:
    """Bounds, work estimates, and numbers of sets that certify a bound follow their definitions."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(30):
        dimension = int(rng.integers(1, 13))
        ranks = tuple(int(rank) for rank in rng.integers(0, dimension + 1, rng.integers(1, 6)))
        weight_divisor, max_coverage = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        certificate = certificates.Certificate(dimension, ranks, weight_divisor, max_coverage)

        oracle = [
            _get_lower_bound_from_definition(dimension, ranks, weight_divisor, max_coverage, weight)
            for weight in range(dimension + 1)
        ]
        assert [certificate.lower_bound(weight) for weight in range(dimension + 1)] == oracle
        steps = [
            (weight, count)
            for weight in range(1, dimension + 1)
            for count in range(1, len(ranks) + 1)
        ]
        step_oracle = {
            step: _get_lower_bound_from_definition(
                dimension, ranks, weight_divisor, max_coverage, *step
            )
            for step in steps
        }
        assert {step: certificate.step_lower_bound(*step) for step in steps} == step_oracle
        ranks_array = np.array(ranks, dtype=np.int64)
        binomials = certificates.get_capped_binomials(dimension)
        for upper_bound in range(1, 2 * dimension + 2):
            # The work from a first layer counts the steps from that layer on, through the first
            # one that certifies the upper bound.
            works = []
            for first_layer in range(1, dimension + 1):
                candidates = 0
                for weight, count in steps:
                    if weight >= first_layer:
                        candidates += math.comb(dimension, weight)
                        if step_oracle[weight, count] >= upper_bound:
                            break
                works.append(candidates)
            assert certificate.work(upper_bound) == works[0]
            assert [
                certificate.work(upper_bound, first_layer)
                for first_layer in range(1, dimension + 1)
            ] == works
            assert (
                certificates.estimate_certificate_work(
                    dimension, ranks_array, weight_divisor, max_coverage, upper_bound, binomials
                )
                == works[0]
            )
            for step in steps:
                assert certificate.certifies(*step, upper_bound) == (
                    step_oracle[step] >= upper_bound
                )
            for weight in range(1, dimension + 1):
                num_sets = next(
                    (
                        count
                        for count in range(1, len(ranks))
                        if step_oracle[weight, count] >= upper_bound
                    ),
                    len(ranks),
                )
                assert certificate.get_num_sets_to_certify(weight, upper_bound) == num_sets

    # A family of no sets, which get_overlapping_family returns for no replacement, does no work.
    assert certificates.Certificate(3, ()).work(2) == 0


def test_weight_bounds_divide_the_step_bounds(pytestconfig: pytest.Config) -> None:
    """Weight bounds round the step bounds up to a parity, then divide them, rounding up."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(60):
        dimension = int(rng.integers(0, 7))
        ranks = tuple(int(rank) for rank in rng.integers(0, dimension + 1, rng.integers(1, 5)))
        divisor, coverage = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        certificate = certificates.Certificate(dimension, ranks, divisor, coverage)
        for factor, parity in itertools.product((1, 2), (None, 0, 1)):
            bounds = certificate.get_weight_bounds(factor, parity)
            assert bounds.shape == (dimension + 1, len(ranks) + 1)
            for layer, num_completed in itertools.product(
                range(dimension + 1), range(len(ranks) + 1)
            ):
                bound = _get_lower_bound_from_definition(
                    dimension, ranks, divisor, coverage, layer, num_completed
                )
                if parity is not None and (layer, num_completed) != (0, 0):
                    bound += (bound - parity) % 2
                expected = (
                    certificates.NO_WORD_BOUND
                    if layer == dimension and num_completed
                    else -(-bound // factor)
                )
                assert bounds[layer, num_completed] == expected
            expected_bounds = bounds.copy()
            bounds[:] = -1
            assert np.array_equal(certificate.get_weight_bounds(factor, parity), expected_bounds)


def test_plans_follow_their_definition(pytestconfig: pytest.Config) -> None:
    """Stopping steps and work of plans, and the sets that they use, follow their definitions."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    work_cap = certificates.WORK_CAP
    cases: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = [
        (np.zeros((3, 1), dtype=np.int64), np.empty(0, dtype=np.int64))
    ]
    for _ in range(20):
        dimension = int(rng.integers(0, 7))
        ranks = rng.integers(0, dimension + 1, rng.integers(1, 5))
        certificate = certificates.Certificate(
            dimension, tuple(ranks.tolist()), int(rng.integers(1, 3)), int(rng.integers(1, 3))
        )
        parity = [None, 0, 1][int(rng.integers(3))]
        cases.append((certificate._step_lower_bounds, ranks))
        cases.append((certificate.get_weight_bounds(int(rng.integers(1, 3)), parity), ranks))
    many_sets = np.full(3000, 2, dtype=np.int64)
    cases.append((certificates.get_step_lower_bounds(2, many_sets, 1, 1), many_sets))

    for bounds, ranks in cases:
        num_rows, num_sets = bounds.shape[0], bounds.shape[1] - 1
        no_word_bound = certificates.NO_WORD_BOUND
        finite = bounds[bounds < no_word_bound]
        limits = [*range(-2, int(finite.max(initial=0)) + 3), no_word_bound - 1, no_word_bound]
        binomials_choices = [
            np.empty(0, dtype=np.int64),
            certificates.get_capped_binomials(num_rows - 1),
            rng.choice([1, 5, work_cap // max(num_sets, 1) + 1, 2**60], num_rows),
        ]
        if num_sets == len(many_sets):
            # Thousands of sets tell the cap of a layer from that of the sum: the layers before the
            # stopping step (2, 1) sum below the cap, but its layer's binomial exceeds each set's
            # share of the cap.
            limits = [2 * num_sets]
            binomials_choices = [np.array([1, 1, 2**50])]
        for first_layer, binomials, skip_unused in itertools.product(
            range(num_rows), binomials_choices, (False, True)
        ):
            for last_layer, limit in itertools.product(range(first_layer, num_rows), limits):
                steps = [
                    (layer, count)
                    for layer in range(first_layer, last_layer + 1)
                    for count in range(1, num_sets + 1)
                ]
                layer, num_completed = next(
                    (step for step in steps if bounds[step] > limit), (last_layer, num_sets)
                )
                last_layers = [
                    layer if index < num_completed else layer - 1 for index in range(num_sets)
                ]
                work = 0
                if len(binomials):
                    work = sum(
                        int(binomials[weight])
                        for rank, last in zip(ranks, last_layers)
                        if not skip_unused or (last >= 0 and last + 1 > num_rows - 1 - rank)
                        for weight in range(first_layer, last + 1)
                    )
                    if work >= work_cap or any(
                        binomials[weight] > work_cap // max(num_sets, 1)
                        for weight in range(first_layer, layer + 1)
                    ):
                        work = work_cap
                expected = (layer, num_completed, work)
                plan = certificates.plan_steps(
                    bounds, limit, first_layer, last_layer, ranks, binomials, skip_unused
                )
                assert plan == expected
                dimension = num_rows - 1
                if first_layer or last_layer < dimension or skip_unused or len(binomials):
                    continue
                step = (layer, num_completed)
                pivots = np.where(np.arange(dimension) < ranks[:, np.newaxis], 0, -1)
                for index, (rank, last) in enumerate(zip(ranks, last_layers)):
                    assert certificates.get_completed_layer(index, *step) == last
                    if last >= 0:
                        unused = last + 1 <= dimension - rank
                        assert certificates.is_unused_set(pivots, index, *step) == unused


def test_sets_to_certify_a_bound_may_come_from_linear_programs() -> None:
    """With linear programs, fewer sets can certify a bound than the coverage bound needs."""
    # In the example of test_linear_programming_can_beat_the_coverage_bound, the coverage bound
    # needs all five sets to certify four in layer one, and cannot certify six in layer two.
    certificate = certificates.Certificate(
        3,
        (3,) * 5,
        max_coverage=3,
        pivot_blocks=((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (4, 5, 6)),
        lp_max_denominator=DENOMINATOR,
    )
    assert certificate.get_num_sets_to_certify(1, 4) == 3
    assert certificate.get_num_sets_to_certify(2, 6) == 2
    assert certificate.get_num_sets_to_certify(1, 5) == 5


def test_certificates_with_overlapping_blocks_of_different_sizes_follow_their_definition() -> None:
    """``certifies`` agrees with ``step_lower_bound`` when blocks overlap and differ in size.

    In the second certificate, a set of rank one in dimension three that has not completed layer
    one requires at least ``-1`` ones in its block, which asks for nothing.
    """
    for dimension, ranks, blocks in [
        (2, (1, 1, 2), ((2,), (2,), (1, 3))),
        (3, (1, 3, 1), ((0,), (0, 1, 2), (3,))),
    ]:
        certificate = certificates.Certificate(
            dimension,
            ranks,
            max_coverage=2,
            pivot_blocks=blocks,
            lp_max_denominator=DENOMINATOR,
        )
        for num_completed, upper_bound in itertools.product(range(4), range(1, 6)):
            bound = certificate.step_lower_bound(1, num_completed)
            assert certificate.certifies(1, num_completed, upper_bound) == (bound >= upper_bound)


def test_plans_with_linear_programs_solve_their_programs_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan with linear programs stops at the first step that certifies, and solves lazily."""
    blocks = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (4, 5, 6))

    def get_certificate() -> certificates.Certificate:
        return certificates.Certificate(
            3, (3,) * 5, max_coverage=3, pivot_blocks=blocks, lp_max_denominator=DENOMINATOR
        )

    reference = get_certificate()
    coverage = certificates.Certificate(3, (3,) * 5, max_coverage=3)
    # The requirements of a step's program identify the step.
    steps_by_requirements = {
        tuple((layer if index < count else layer - 1) + 1 for index in range(5)): (layer, count)
        for layer, count in itertools.product(range(1, 4), range(1, 6))
    }
    solved: list[tuple[int, int]] = []
    get_lp_lower_bound = certificates.get_lp_lower_bound

    def record(
        pivot_blocks: tuple[tuple[int, ...], ...], requirements: list[int], max_denominator: int
    ) -> int:
        solved.append(steps_by_requirements[tuple(requirements)])
        return get_lp_lower_bound(pivot_blocks, requirements, max_denominator)

    monkeypatch.setattr(certificates, "get_lp_lower_bound", record)
    num_solved = 0
    for layer, limit in itertools.product(range(1, 4), range(9)):
        steps = [(layer, count) for count in range(1, 6)]
        stop = next(
            (step for step in steps[:-1] if reference.step_lower_bound(*step) > limit), steps[-1]
        )
        solved.clear()
        assert get_certificate().plan(limit, layer) == stop[1]
        assert steps[-1] not in solved
        for step in solved:
            assert steps.index(step) <= steps.index(stop)
            assert coverage.step_lower_bound(*step) <= limit
        num_solved += len(solved)
    assert num_solved

    # The estimate of work solves no program.
    solved.clear()
    for upper_bound in range(9):
        assert get_certificate().work(upper_bound) == coverage.work(upper_bound)
    assert not solved


def test_linear_programming_can_beat_the_coverage_bound() -> None:
    """Linear programming certifies five in the example of Section 3.1 of brouwer_zimmermann.md."""
    # The columns are e1, e2, e3, e1 + e2 + e3, e1, e2, e3, and after layer one each block holds at
    # least two ones of an unvisited word, so the coverage bound, with coverage three, is
    # ceil(10 / 3) = 4.
    blocks = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (4, 5, 6))
    assert certificates.get_lp_lower_bound(blocks, [2] * 5, DENOMINATOR) == 5
    # Dual weights of thirds need a denominator of three; coarser ones round part of the bound away.
    bounds = [certificates.get_lp_lower_bound(blocks, [2] * 5, d) for d in (1, 2, 3)]
    assert bounds == [2, 4, 5]
    certificate = certificates.Certificate(3, (3,) * 5, max_coverage=3)
    assert certificate.lower_bound(1) == 4
    assert (
        dataclasses.replace(
            certificate, pivot_blocks=blocks, lp_max_denominator=DENOMINATOR
        ).lower_bound(1)
        == 5
    )

    # No word has three ones in a block of two columns, and requirements of zero ask for nothing.
    assert certificates.get_lp_lower_bound([(0, 1)], [3], DENOMINATOR) == certificates.NO_WORD_BOUND
    assert certificates.get_lp_lower_bound([(0, 1)], [0], DENOMINATOR) == 0

    # A word with every column of two blocks of three that share two columns has weight four.
    assert certificates.get_lp_lower_bound([(0, 1, 2), (0, 1, 3)], [3, 3], DENOMINATOR) == 4


def test_a_failed_linear_program_gives_no_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the solver reports a failure, the bound claims nothing rather than a solver's guess."""
    failure = scipy.optimize.OptimizeResult(status=4, message="numerical difficulties")
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *args, **kwargs: failure)
    blocks = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (4, 5, 6))
    assert certificates.get_lp_lower_bound(blocks, [2] * 5, DENOMINATOR) == 0


@pytest.mark.parametrize("dimension, num_sets", [(300, 16), (60, 8192)])
def test_work_estimates_saturate(dimension: int, num_sets: int) -> None:
    """Work estimates for huge searches stay large and positive, rather than overflowing."""
    certificate = certificates.Certificate(dimension, ranks=(dimension,) * num_sets)
    assert 2**50 < certificate.work(upper_bound=10**6) < 2**63


def test_binomials_are_exact_where_they_fit_and_cannot_be_modified() -> None:
    binomials = certificates.get_capped_binomials(70)
    assert not binomials.flags.writeable
    assert binomials[:5].tolist() == [math.comb(70, weight) for weight in range(5)]
    assert 0 < binomials[35] < math.comb(70, 35)


def _get_lower_bound_from_definition(
    dimension: int,
    ranks: tuple[int, ...],
    weight_divisor: int,
    max_coverage: int,
    weight: int,
    num_completed: int | None = None,
) -> int:
    """The certificate lower bound at step ``(weight, num_completed)``, from its definition.

    By default, every set has completed layer ``weight``.
    """
    num_completed = len(ranks) if num_completed is None else num_completed
    if weight == dimension and num_completed:
        # A set that has completed the last layer has visited every word.
        return certificates.NO_WORD_BOUND
    completed = [weight if index < num_completed else weight - 1 for index in range(len(ranks))]
    occupancy = sum(max(0, layer + 1 - (dimension - rank)) for layer, rank in zip(completed, ranks))
    bound = -(-occupancy // max_coverage)
    return bound + (-bound) % weight_divisor
