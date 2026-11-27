# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the certificates of basis families and the plans of their steps."""

from __future__ import annotations

import itertools
import math

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import row_reduce_binary

from ..conftest import assert_certificate_bounds_unvisited_words, get_random_full_rank_matrix
from . import basis_families, certificates


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


def test_certificates_bound_every_unvisited_word(pytestconfig: pytest.Config) -> None:
    """Certificates of random sets bound the weight of every word that no set has visited.

    The sets overlap unevenly, and some are partial.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(20):
        dimension = int(rng.integers(3, 6))
        basis = get_random_full_rank_matrix(
            rng, dimension, int(rng.integers(dimension + 2, 3 * dimension))
        )
        # Reduce the basis on random nonzero columns, so that each set has a pivot.
        nonzero = np.flatnonzero(basis.any(axis=0))
        num_sets = int(rng.integers(3, 6))
        sets = [
            row_reduce_binary(
                basis,
                None,
                columns=rng.permutation(nonzero)[: int(rng.integers(1, len(nonzero) + 1))],
            )
            for _ in range(num_sets)
        ]
        pivots = np.full((num_sets, dimension), -1, dtype=np.int64)
        for index, (_, _, found) in enumerate(sets):
            pivots[index, : len(found)] = found
        family = basis_families.BasisFamily(
            np.array([reduced for reduced, _, _ in sets]),
            np.zeros((num_sets, dimension, 0), dtype=np.uint8),
            pivots,
            np.array([len(found) for _, _, found in sets], dtype=np.int64),
            basis_families.infer_weight_divisor(basis),
            int(np.bincount(pivots[pivots >= 0]).max()),
        )
        assert_certificate_bounds_unvisited_words(basis, family)


def test_the_example_of_the_coverage_bound() -> None:
    """Pairs of columns of the code generated by ``101`` and ``011`` certify three after layer one.

    This is the example of ``brouwer_zimmermann.md#sec-overlapping-certificate``.
    """
    basis = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    words = [np.array(bits) @ basis % 2 for bits in itertools.product((0, 1), repeat=2)]
    # The code has distance two, and each pair of columns is an information set, since the four
    # words differ on it.  Each column lies in two of the pairs.
    assert min(int(word.sum()) for word in words[1:]) == 2
    for pair in itertools.combinations(range(3), 2):
        assert len({tuple(word[list(pair)]) for word in words}) == 4
    assert certificates.Certificate(2, (2, 2, 2), max_coverage=2).lower_bound(1) == 3


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


@pytest.mark.parametrize("dimension, num_sets", [(300, 16), (60, 8192)])
def test_work_estimates_saturate(dimension: int, num_sets: int) -> None:
    """Work estimates for huge searches stay large and positive, rather than overflowing."""
    certificate = certificates.Certificate(dimension, ranks=(dimension,) * num_sets)
    assert 2**50 < certificate.work(upper_bound=10**6) < 2**63


def test_binomials_are_exact_where_they_fit_and_cannot_be_modified() -> None:
    """Binomials are exact below the cap and capped above it, in a read-only shared array."""
    binomials = certificates.get_capped_binomials(70)
    assert certificates.get_capped_binomials(70) is binomials
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
