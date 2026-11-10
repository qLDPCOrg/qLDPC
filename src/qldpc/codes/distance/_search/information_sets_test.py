# SPDX-License-Identifier: Apache-2.0

"""Tests for the basis families and certificates shared by BZ distance and counting."""

from __future__ import annotations

import dataclasses
import functools
import itertools
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import scipy.optimize

import qldpc
from qldpc import _compilation
from qldpc._binary import get_independent_rows, row_reduce_binary

from .._tuning import DEFAULT_TUNING, SearchTuning
from ..conftest import get_random_full_rank_matrix
from . import information_sets

# The default numbers of column orders of disjoint families, and denominators of LP dual weights.
ORDERS = DEFAULT_TUNING.disjoint_random_column_orders
DENOMINATOR = DEFAULT_TUNING.lp_dual_max_denominator

# Two disjoint full blocks and a rank-two remainder.
PARTIAL_BASIS = np.hstack(
    [
        np.eye(3, dtype=np.uint8),
        np.eye(3, dtype=np.uint8),
        np.array([[1, 0], [0, 1], [0, 0]], dtype=np.uint8),
    ]
)
EVEN_BASIS = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 0, 1]], dtype=np.uint8)


def test_information_set_packing() -> None:
    """Information sets pack full blocks first and retain the residual rank."""
    get_family = functools.partial(
        information_sets.get_disjoint_family, random_column_orders=ORDERS
    )
    assert get_family(PARTIAL_BASIS, np.eye(3, dtype=np.uint8)).ranks.tolist() == [3, 3, 2]
    assert not len(get_family(np.zeros((1, 3), dtype=np.uint8), None).ranks)

    basis = np.array(
        [
            [1, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0],
            [1, 0, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert get_family(basis, None).ranks.tolist() == [3, 3]

    # Only one column carries the second row, so no column order packs a second full set.
    basis = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
    assert get_family(basis, None).ranks.tolist() == [2, 1]


def test_a_basis_family_stacks_its_sets() -> None:
    """The sets of a basis family are reduced on their pivots, which are padded for partial sets."""
    labels = np.eye(3, dtype=np.uint8)
    basis_family = information_sets.get_disjoint_family(
        PARTIAL_BASIS, labels, random_column_orders=ORDERS
    )
    assert basis_family.generators.shape == (3, 3, PARTIAL_BASIS.shape[1])
    assert basis_family.labels.shape == (3, 3, 3)
    assert basis_family.certificate.dimension == 3
    for generators, set_labels, pivots, rank in zip(*basis_family[:4], strict=True):
        # The labels record the row operations that reduce the basis.
        np.testing.assert_array_equal(generators, set_labels @ PARTIAL_BASIS % 2)
        np.testing.assert_array_equal(generators[:rank, pivots[:rank]], np.eye(rank))
        assert np.all(pivots[rank:] == -1)
    no_labels = information_sets.get_disjoint_family(
        PARTIAL_BASIS, None, random_column_orders=ORDERS
    )
    assert no_labels.labels.shape == (3, 3, 0)


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("basis", [PARTIAL_BASIS, EVEN_BASIS])
def test_family_certificates(basis: npt.NDArray[np.uint8]) -> None:
    """Rank-deficient blocks and inferred parity give certificates valid for every word."""
    # A target above every weight keeps the rank-deficient block.
    basis_family = information_sets.get_disjoint_family(basis, None, random_column_orders=ORDERS)
    basis_family = information_sets.prune_family(basis_family, basis.shape[1] + 1)
    _assert_certificate_bounds_unvisited_words(basis, basis_family)


def test_weight_divisor_inference() -> None:
    """Even rows span only even weights, and a divisor proven by the caller is retained."""
    odd = np.array([[1, 0, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    get_family = functools.partial(
        information_sets.get_disjoint_family, random_column_orders=ORDERS
    )
    assert get_family(EVEN_BASIS, None).certificate.weight_divisor == 2
    assert get_family(odd, None).certificate.weight_divisor == 1
    assert get_family(odd, None, weight_divisor=2).certificate.weight_divisor == 2


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize(
    "ranks, upper_bound, retained",
    [([10, 6, 2], 6, [10, 6]), ([10, 6, 2], 12, [10, 6, 2]), ([6, 2], 6, [6, 2])],
)
def test_prune_family(ranks: list[int], upper_bound: int, retained: list[int]) -> None:
    """Pruning keeps full sets, and the partial sets that can help certify the target.

    A set of ten rows with rank deficiency ``d`` adds to the bound only once layer ``d`` is
    complete.  The full set alone certifies 6 once layer 5 is complete, so the set of deficiency 8
    cannot help with 6, but no layer lets the full set alone certify 12.  Without a full set, every
    set stays.
    """
    no_columns = np.empty((len(ranks), 10, 0), dtype=np.uint8)
    pivots = np.full((len(ranks), 10), -1, dtype=np.int64)
    basis_family = information_sets.BasisFamily(
        no_columns, no_columns, pivots, np.array(ranks, dtype=np.int64), 1, 1
    )
    pruned = information_sets.prune_family(basis_family, upper_bound)
    assert pruned.ranks.tolist() == retained
    assert pruned.certificate.work(upper_bound) <= basis_family.certificate.work(upper_bound)


@pytest.mark.usefixtures("compilation_mode")
def test_overlapping_family_certificates() -> None:
    """Check pivot coverage and label transformations against exhaustive codeword weights."""

    # Label each row separately so its row operations can be checked.
    basis = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    basis = get_independent_rows(basis)
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = information_sets.get_disjoint_family(basis, labels, random_column_orders=ORDERS)
    # One full set certifies a target of one after its first layer, which no replacement improves.
    assert not len(_get_overlapping_family(basis, labels, disjoint, 1, minimum_work=0).ranks)
    basis_family = _get_overlapping_family(basis, labels, disjoint, 3, minimum_work=0)
    assert np.all(basis_family.ranks == len(basis))
    loads = np.bincount(basis_family.pivots.ravel(), minlength=basis.shape[1])
    assert basis_family.max_coverage >= loads.max() > 1
    for generators, set_labels, pivots in zip(*basis_family[:3], strict=True):
        np.testing.assert_array_equal(generators, set_labels @ basis % 2)
        np.testing.assert_array_equal(generators[:, pivots], np.eye(len(basis)))
    _assert_certificate_bounds_unvisited_words(basis, basis_family)


@pytest.mark.usefixtures("compilation_mode")
def test_a_cheap_search_keeps_its_disjoint_basis_family() -> None:
    """No overlapping basis family is built if the work of the disjoint one is below the minimum."""
    basis = get_independent_rows(np.asarray(qldpc.codes.HammingCode(4).generator))
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = information_sets.get_disjoint_family(basis, labels, random_column_orders=ORDERS)
    assert not len(_get_overlapping_family(basis, labels, disjoint, 3).ranks)
    assert len(_get_overlapping_family(basis, labels, disjoint, 3, minimum_work=0).ranks)
    # A limit of zero sets builds no family at all.
    assert not len(_get_overlapping_family(basis, labels, disjoint, 3, 0, minimum_work=0).ranks)


@pytest.mark.usefixtures("compilation_mode")
def test_overlapping_family_guards() -> None:
    """Reject unhelpful coverage and terminate repeated attempts at the same pivot set."""

    # Complete or nearly complete disjoint coverage leaves no cheaper family within the size budget.
    basis = np.hstack([np.eye(25, dtype=np.uint8)] * 4)
    complete = information_sets.get_disjoint_family(basis, None, random_column_orders=ORDERS)
    assert not len(_get_overlapping_family(basis, None, complete, 26).ranks)
    shortened = basis[:, :-1]
    disjoint = information_sets.get_disjoint_family(shortened, None, random_column_orders=ORDERS)
    assert not len(_get_overlapping_family(shortened, None, disjoint, 26).ranks)

    # A zero basis has no information sets to replace.
    zeros = np.zeros((1, 3), dtype=np.uint8)
    empty = information_sets.get_disjoint_family(zeros, None, random_column_orders=ORDERS)
    assert not len(empty.ranks)
    assert not len(_get_overlapping_family(zeros, None, empty, 2).ranks)

    # An information set holds one of the third to fifth columns, which are equal, and two of the
    # first, second, and sixth, so attempts rediscover the few such sets.  The search must terminate
    # without a replacement that beats the disjoint family.
    small = np.array([[1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 1, 0]], dtype=np.uint8)
    disjoint = information_sets.get_disjoint_family(small, None, random_column_orders=ORDERS)
    assert not len(_get_overlapping_family(small, None, disjoint, 3, minimum_work=0).ranks)


@pytest.mark.usefixtures("compilation_mode")
def test_certificate_follows_its_definition(pytestconfig: pytest.Config) -> None:
    """The bounds, work estimate, and number of sets that certify a bound equal their definitions.

    A search visits the sets of a layer in family order and stops after the first set whose
    completion of the layer certifies the upper bound, so the work counts the sets visited, and
    ``get_num_sets_to_certify`` gives the number of sets visited in each layer.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(30):
        dimension = int(rng.integers(1, 13))
        ranks = tuple(int(rank) for rank in rng.integers(0, dimension + 1, rng.integers(1, 6)))
        weight_divisor, max_coverage = int(rng.integers(1, 5)), int(rng.integers(1, 5))
        certificate = information_sets.Certificate(dimension, ranks, weight_divisor, max_coverage)

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
        for upper_bound in range(1, 2 * dimension + 2):
            candidates = 0
            for weight, count in steps:
                candidates += math.comb(dimension, weight)
                if step_oracle[weight, count] >= upper_bound:
                    break
            assert certificate.work(upper_bound) == candidates
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

    # A family without sets, which the overlapping stage returns for no replacement, does no work.
    assert information_sets.Certificate(3, ()).work(2) == 0


def test_weight_bounds_divide_the_step_bounds(pytestconfig: pytest.Config) -> None:
    """The weight bounds divide the certificate's bounds by the weight factor, rounding up.

    The certificate's bounds come from their definition, and the entries ``[dimension, j]`` for
    ``j >= 1`` are ``NO_WORD_BOUND``.  With a parity, each other entry is first rounded up to the
    least integer of that parity that is at least the entry, except the zero before the first step.
    This holds for sets of random ranks, including zero, both weight factors, and every parity.
    Each call returns a table of its own.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(60):
        dimension = int(rng.integers(0, 7))
        ranks = tuple(int(rank) for rank in rng.integers(0, dimension + 1, rng.integers(1, 5)))
        divisor, coverage = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        certificate = information_sets.Certificate(dimension, ranks, divisor, coverage)
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
                    information_sets.NO_WORD_BOUND
                    if layer == dimension and num_completed
                    else -(-bound // factor)
                )
                assert bounds[layer, num_completed] == expected
            expected_bounds = bounds.copy()
            bounds[:] = -1
            assert np.array_equal(certificate.get_weight_bounds(factor, parity), expected_bounds)


@pytest.mark.usefixtures("compilation_mode")
def test_plans_follow_their_definition(pytestconfig: pytest.Config) -> None:
    """The stopping step and the work of a plan follow their definitions.

    The stopping step is the first step of the given layers whose bound exceeds the limit, or the
    last step if none does.  At that step, each set has completed its last layer, and the work adds
    the binomial of each layer from the first layer to the last layer of each set, leaving out the
    sets whose rank keeps them out of the bound if unused sets are skipped.  The work saturates at
    the cap if a layer up to that of the step has a binomial above the cap divided by the number of
    sets, or the sum reaches the cap, and no binomials give no work.

    At the stopping step of a plan over every layer, ``get_completed_layer`` gives the last layer of
    each set.  ``is_unused_set`` tells from the pivots of a set, which are -1 beyond its rank,
    whether a plan that skips unused sets leaves the set out.

    The bounds are certificate tables of sets of every rank, including zero, a table without sets,
    and the weight bounds of the coset searches, which divide them by a weight factor after rounding
    them to a parity.  The limits run past both ends of each table, to the largest native integer.
    The binomials are random, and some exceed the share of the cap of each set.  Thousands of sets
    tell the cap of a layer from the cap of the sum, since the stopping step comes after the first
    set of a layer whose binomial exceeds the share, while the sum stays below the cap.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    work_cap = information_sets._WORK_CAP
    cases: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = [
        (np.zeros((3, 1), dtype=np.int64), np.empty(0, dtype=np.int64))
    ]
    for _ in range(20):
        dimension = int(rng.integers(0, 7))
        ranks = rng.integers(0, dimension + 1, rng.integers(1, 5))
        certificate = information_sets.Certificate(
            dimension, tuple(ranks.tolist()), int(rng.integers(1, 3)), int(rng.integers(1, 3))
        )
        parity = [None, 0, 1][int(rng.integers(3))]
        cases.append((certificate._step_lower_bounds, ranks))
        cases.append((certificate.get_weight_bounds(int(rng.integers(1, 3)), parity), ranks))
    many_sets = np.full(3000, 2, dtype=np.int64)
    cases.append((information_sets.get_step_lower_bounds(2, many_sets, 1, 1), many_sets))

    for bounds, ranks in cases:
        num_rows, num_sets = bounds.shape[0], bounds.shape[1] - 1
        no_word_bound = information_sets.NO_WORD_BOUND
        finite = bounds[bounds < no_word_bound]
        limits = [*range(-2, int(finite.max(initial=0)) + 3), no_word_bound - 1, no_word_bound]
        binomials_choices = [
            np.empty(0, dtype=np.int64),
            information_sets.get_capped_binomials(num_rows - 1),
            rng.choice([1, 5, work_cap // max(num_sets, 1) + 1, 2**60], num_rows),
        ]
        if num_sets == len(many_sets):
            # The layers before the stopping step (2, 1) sum below the cap, but its layer does not.
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
                plan = information_sets.plan_steps(
                    bounds, limit, first_layer, last_layer, ranks, binomials, skip_unused
                )
                assert plan == expected
                dimension = num_rows - 1
                if first_layer or last_layer < dimension or skip_unused or len(binomials):
                    continue
                step = (layer, num_completed)
                pivots = np.where(np.arange(dimension) < ranks[:, np.newaxis], 0, -1)
                for index, (rank, last) in enumerate(zip(ranks, last_layers)):
                    assert information_sets.get_completed_layer(index, *step) == last
                    if last >= 0:
                        unused = last + 1 <= dimension - rank
                        assert information_sets.is_unused_set(pivots, index, *step) == unused


def test_sets_to_certify_a_bound_may_come_from_linear_programs() -> None:
    """With linear programs, fewer sets can certify a bound than the coverage bound needs.

    In the [7, 3] example of ``test_linear_programming_can_beat_the_coverage_bound``, the coverage
    bound needs all five sets to certify four in layer one, and cannot certify six in layer two.
    Linear programs certify them after three sets and two sets.  Five needs all five sets in layer
    one either way.
    """
    certificate = information_sets.Certificate(
        3,
        (3,) * 5,
        max_coverage=3,
        pivot_blocks=((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (4, 5, 6)),
        lp_max_denominator=DENOMINATOR,
    )
    assert certificate.get_num_sets_to_certify(1, 4) == 3
    assert certificate.get_num_sets_to_certify(2, 6) == 2
    assert certificate.get_num_sets_to_certify(1, 5) == 5


def test_plans_with_linear_programs_solve_their_programs_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A plan with linear programs stops at the first step that certifies, and solves lazily.

    The certificate is that of ``test_sets_to_certify_a_bound_may_come_from_linear_programs``.  For
    every limit and layer, the plan stops at the first step of the layer whose bound, with linear
    programs, exceeds the limit, or at the last step.  It solves the linear program of a step only
    if that step is at or before the stopping step, and its coverage bound does not exceed the
    limit.  It never solves that of the last step, which stops the search anyway.  Some plans solve
    programs, and the work estimate of the certificate solves none, since it uses only the coverage
    bound.  The requirements of the program of a step identify the step.
    """
    blocks = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (4, 5, 6))

    def get_certificate() -> information_sets.Certificate:
        return information_sets.Certificate(
            3, (3,) * 5, max_coverage=3, pivot_blocks=blocks, lp_max_denominator=DENOMINATOR
        )

    reference = get_certificate()
    coverage = information_sets.Certificate(3, (3,) * 5, max_coverage=3)
    steps_by_requirements = {
        tuple((layer if index < count else layer - 1) + 1 for index in range(5)): (layer, count)
        for layer, count in itertools.product(range(1, 4), range(1, 6))
    }
    solved: list[tuple[int, int]] = []
    get_lp_lower_bound = information_sets.get_lp_lower_bound

    def record(
        pivot_blocks: tuple[tuple[int, ...], ...], requirements: list[int], max_denominator: int
    ) -> int:
        solved.append(steps_by_requirements[tuple(requirements)])
        return get_lp_lower_bound(pivot_blocks, requirements, max_denominator)

    monkeypatch.setattr(information_sets, "get_lp_lower_bound", record)
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


def test_partitioned_families_replace_costlier_ones() -> None:
    """Matroid partitioning replaces a disjoint family only in costly searches, and only if cheaper.

    This [11, 4] basis has disjoint information sets of ranks four, four, and two, and information
    sets that share columns certify the weight five with less work.
    """
    rows = ["10001011100", "01000001100", "00100000110", "00010011001"]
    basis = np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = information_sets.get_disjoint_family(basis, labels, random_column_orders=ORDERS)
    assert disjoint.ranks.tolist() == [4, 4, 2]

    def get_family(**changes: int) -> information_sets.BasisFamily:
        fields: dict[str, Any] = {"partition_work_ratio": 0, **changes}
        tuning = dataclasses.replace(DEFAULT_TUNING, **fields)
        return information_sets.get_partitioned_family(basis, labels, disjoint, 5, tuning)

    # A family needs a set to hold, a coverage of two or more, and partitioning that pays.
    assert not len(get_family(partition_work_ratio=2**62).ranks)
    assert not len(get_family(partition_cost_factor=2**40, partition_work_ratio=1).ranks)
    assert len(get_family(partition_cost_factor=0, partition_work_ratio=1).ranks)
    assert not len(get_family(max_information_sets=0).ranks)
    assert not len(get_family(max_partition_coverage=1).ranks)
    family = get_family()
    assert family.max_coverage > 1
    assert family.certificate.work(5) < disjoint.certificate.work(5)
    for generators, set_labels, pivots in zip(*family[:3], strict=True):
        np.testing.assert_array_equal(generators, set_labels @ basis % 2)
        np.testing.assert_array_equal(generators[:, pivots], np.eye(len(basis)))
    _assert_certificate_bounds_unvisited_words(basis, family)


def test_maximum_disjoint_families_replace_smaller_packings() -> None:
    """A maximum disjoint family has the most information sets, and then the largest partial set.

    In natural column order, a greedy packing of the columns ``e1, e2, e3, e1 + e2, e1 + e2`` takes
    the first three, and leaves a partial set of rank one, but ``e1 + e2, e2, e3`` leave ``e1`` and
    ``e1 + e2``, of rank two.  Likewise, ``e1, e2, e1 + e2, e1 + e2`` hold two disjoint information
    sets, but a greedy packing finds one.  The maximum families are disjoint sets, reduced on their
    pivots, with valid certificates.  No family replaces a packing whose sets use every nonzero
    column, or one of a basis whose packings all have the same ranks.
    """
    for columns, ranks in (([1, 2, 4, 3, 3], [3, 2]), ([1, 2, 3, 3], [2, 2])):
        dimension = max(columns).bit_length()
        basis = ((np.array(columns) >> np.arange(dimension)[:, None]) & 1).astype(np.uint8)
        disjoint = information_sets.get_disjoint_family(basis, None, random_column_orders=0)
        assert disjoint.ranks.tolist() == [dimension, 1]
        family = information_sets.get_maximum_disjoint_family(basis, disjoint)
        assert family.ranks.tolist() == ranks
        pivots = family.pivots[family.pivots >= 0]
        assert len(set(pivots.tolist())) == len(pivots)
        for generators, set_pivots, rank in zip(family.generators, family.pivots, family.ranks):
            np.testing.assert_array_equal(
                generators[:, set_pivots[:rank]], np.eye(dimension)[:, :rank]
            )
        _assert_certificate_bounds_unvisited_words(basis, family)

    one_full_set = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
    for basis in (PARTIAL_BASIS, one_full_set):
        disjoint = information_sets.get_disjoint_family(basis, None, random_column_orders=0)
        assert not len(information_sets.get_maximum_disjoint_family(basis, disjoint).ranks)


def test_linear_programming_can_beat_the_coverage_bound() -> None:
    """Five information sets of a [7, 3] code, where linear programming certifies five, not four.

    The columns are ``e1, e2, e3, e1 + e2 + e3, e1, e2, e3``.  Four blocks are the 3-subsets of the
    first four columns, and the fifth is the last three columns, so the maximum coverage is three.
    After layer one, each block holds at least two ones of an unvisited word.  The coverage bound is
    ``ceil(10 / 3) = 4``, but ``x = 2/3`` on every column and dual weights of ``1/3`` on the first
    four blocks and one on the fifth both have value ``14/3``, so the linear program certifies five.
    """
    blocks = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (4, 5, 6))
    assert information_sets.get_lp_lower_bound(blocks, [2] * 5, DENOMINATOR) == 5
    # Dual weights of thirds need a denominator of three; coarser ones round part of the bound away.
    bounds = [information_sets.get_lp_lower_bound(blocks, [2] * 5, d) for d in (1, 2, 3)]
    assert bounds == [2, 4, 5]
    certificate = information_sets.Certificate(3, (3,) * 5, max_coverage=3)
    assert certificate.lower_bound(1) == 4
    assert (
        dataclasses.replace(
            certificate, pivot_blocks=blocks, lp_max_denominator=DENOMINATOR
        ).lower_bound(1)
        == 5
    )

    # No word has three ones in a block of two columns, and requirements of zero ask for nothing.
    assert (
        information_sets.get_lp_lower_bound([(0, 1)], [3], DENOMINATOR)
        == information_sets.NO_WORD_BOUND
    )
    assert information_sets.get_lp_lower_bound([(0, 1)], [0], DENOMINATOR) == 0


def test_a_failed_linear_program_gives_no_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the solver reports a failure, the bound claims nothing rather than a solver's guess."""
    failure = scipy.optimize.OptimizeResult(status=4, message="numerical difficulties")
    monkeypatch.setattr(scipy.optimize, "linprog", lambda *args, **kwargs: failure)
    blocks = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (4, 5, 6))
    assert information_sets.get_lp_lower_bound(blocks, [2] * 5, DENOMINATOR) == 0


def test_linear_programming_certificates_bound_unvisited_words(pytestconfig: pytest.Config) -> None:
    """Families of random information sets, with uneven coverage, have valid certificates."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(20):
        dimension = int(rng.integers(3, 6))
        basis = get_random_full_rank_matrix(
            rng, dimension, int(rng.integers(dimension + 2, 3 * dimension))
        )
        num_sets = int(rng.integers(3, 6))
        sets = [
            row_reduce_binary(basis, None, columns=rng.permutation(basis.shape[1]))
            for _ in range(num_sets)
        ]
        pivots = np.array([found for _, _, found in sets], dtype=np.int64)
        family = information_sets.BasisFamily(
            np.array([reduced for reduced, _, _ in sets]),
            np.zeros((num_sets, dimension, 0), dtype=np.uint8),
            pivots,
            np.full(num_sets, dimension, dtype=np.int64),
            information_sets.infer_weight_divisor(basis),
            int(np.bincount(pivots.ravel()).max()),
        )
        _assert_certificate_bounds_unvisited_words(basis, family)


def test_searches_of_overlapping_sets_use_linear_programming_when_costly() -> None:
    """Only a costly search of overlapping sets gets the bound of linear programming."""
    basis = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    disjoint = information_sets.get_disjoint_family(basis, None, random_column_orders=ORDERS)
    overlapping = _get_overlapping_family(basis, None, disjoint, 3, minimum_work=0)
    assert overlapping.max_coverage > 1
    assert not overlapping.get_search_certificate(3, 2**62, DENOMINATOR).pivot_blocks
    assert len(overlapping.get_search_certificate(3, 0, DENOMINATOR).pivot_blocks) == len(
        overlapping.ranks
    )
    assert not disjoint.get_search_certificate(3, 0, DENOMINATOR).pivot_blocks


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("dimension, num_sets", [(300, 16), (60, 8192)])
def test_work_estimates_saturate(dimension: int, num_sets: int) -> None:
    """Work estimates for huge searches stay large and positive, rather than overflowing."""
    certificate = information_sets.Certificate(dimension, ranks=(dimension,) * num_sets)
    assert 2**50 < certificate.work(upper_bound=10**6) < 2**63


def test_binomials_are_exact_where_they_fit_and_cannot_be_modified() -> None:
    binomials = information_sets.get_capped_binomials(70)
    assert not binomials.flags.writeable
    assert binomials[:5].tolist() == [math.comb(70, weight) for weight in range(5)]
    assert 0 < binomials[35] < math.comb(70, 35)


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("max_sets, num_sets", [(1, 0), (3, 0), (4, 4), (16, 4)])
def test_the_size_limit_bounds_the_overlapping_basis_family(max_sets: int, num_sets: int) -> None:
    """A cheaper basis family of overlapping sets needs four sets, so a lower limit finds none."""
    basis = get_independent_rows(np.asarray(qldpc.codes.HammingCode(4).generator))
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = information_sets.get_disjoint_family(basis, labels, random_column_orders=ORDERS)
    overlapping = _get_overlapping_family(basis, labels, disjoint, 3, max_sets, minimum_work=0)
    assert len(overlapping.ranks) == num_sets


@pytest.mark.usefixtures("compilation_mode")
def test_layers_visit_each_support_once_with_its_sums(pytestconfig: pytest.Config) -> None:
    """A layer visits the combinations of rows in order, each with its sum in every set.

    The second set uses fewer words than the first, and the starting words are random.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    widths = np.array([3, 1], dtype=np.int64)
    for dimension in range(1, 7):
        rows = rng.integers(2**64, size=(2, dimension, 3), dtype=np.uint64)
        start = rng.integers(2**64, size=(2, 3), dtype=np.uint64)
        for weight in range(1, dimension + 1):
            support = np.zeros(weight, dtype=np.int64)
            prefixes = np.zeros((weight, 2, 3), dtype=np.uint64)
            prefixes[0] = start
            visited = []
            depth = information_sets.advance_support(rows, widths, support, prefixes, 0)
            while depth >= 0:
                sums = prefixes[depth] ^ rows[:, support[depth]]
                visited.append(
                    (support.tolist(), [sums[jj, : widths[jj]].tolist() for jj in range(2)])
                )
                support[depth] += 1
                depth = information_sets.advance_support(rows, widths, support, prefixes, depth)
            expected = []
            for indices in itertools.combinations(range(dimension), weight):
                sums = start ^ np.bitwise_xor.reduce(rows[:, list(indices)], axis=1)
                expected.append(
                    (list(indices), [sums[jj, : widths[jj]].tolist() for jj in range(2)])
                )
            assert visited == expected


def test_functions_agree_in_plain_python_and_compiled(pytestconfig: pytest.Config) -> None:
    """Pruned and replacement basis families are identical, bit for bit, plain and compiled."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The Hamming code has an overlapping basis family that does less work.  Random bases rarely do.
    hamming = get_independent_rows(np.asarray(qldpc.codes.HammingCode(4).generator))
    cases: list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None, int]] = [
        (hamming, np.eye(len(hamming), dtype=np.uint8), 3)
    ]
    for trial in range(12):
        rows = int(rng.integers(3, 10))
        basis = get_random_full_rank_matrix(rng, rows, int(rng.integers(2 * rows, 5 * rows)))
        labels = np.eye(rows, dtype=np.uint8) if trial % 2 else None
        cases.append((basis, labels, int(rng.integers(2, rows + 2))))
    for case, (basis, labels, upper_bound) in enumerate(cases):
        results = []
        for compiled in (False, True):
            _compilation.set_on_request_functions_compiled(compiled)
            try:
                disjoint = information_sets.get_disjoint_family(
                    basis, labels, random_column_orders=ORDERS
                )
                pruned = information_sets.prune_family(disjoint, upper_bound)
                replacement = _get_overlapping_family(
                    basis, labels, pruned, upper_bound, minimum_work=0
                )
            finally:
                _compilation.set_on_request_functions_compiled(False)
            results.append((pruned, replacement))
        assert case or len(results[0][1].ranks)
        for plain, compiled_result in zip(*results, strict=True):
            for field, value in zip(plain._fields, plain, strict=True):
                np.testing.assert_array_equal(value, getattr(compiled_result, field), field)


def _get_lower_bound_from_definition(
    dimension: int,
    ranks: tuple[int, ...],
    weight_divisor: int,
    max_coverage: int,
    weight: int,
    num_completed: int | None = None,
) -> int:
    """The certificate lower bound once sets have completed layers, from its definition.

    The first ``num_completed`` sets have completed layer ``weight``, and the others layer ``weight
    - 1``; by default, every set has completed layer ``weight``.
    """
    num_completed = len(ranks) if num_completed is None else num_completed
    if weight == dimension and num_completed:
        # A set that has completed the last layer has visited every word.
        return information_sets.NO_WORD_BOUND
    completed = [weight if index < num_completed else weight - 1 for index in range(len(ranks))]
    occupancy = sum(max(0, layer + 1 - (dimension - rank)) for layer, rank in zip(completed, ranks))
    bound = -(-occupancy // max_coverage)
    return bound + (-bound) % weight_divisor


def _get_overlapping_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    basis_family: information_sets.BasisFamily,
    upper_bound: int,
    max_sets: int = DEFAULT_TUNING.max_information_sets,
    *,
    minimum_work: int = DEFAULT_TUNING.overlapping_minimum_work,
) -> information_sets.BasisFamily:
    """Stage three, as a search whose tuning has these thresholds calls it."""
    tuning = SearchTuning(max_information_sets=max_sets, overlapping_minimum_work=minimum_work)
    return information_sets.get_overlapping_family(
        basis,
        np.empty((len(basis), 0), dtype=np.uint8) if labels is None else labels,
        basis_family,
        upper_bound,
        *information_sets.get_overlapping_family_policy(tuning, len(basis)),
    )


def _assert_certificate_bounds_unvisited_words(
    basis: npt.NDArray[np.uint8], family: information_sets.BasisFamily
) -> None:
    """Check a basis family's certificates against every word that its search bases span.

    A search basis visits a word after layer ``t`` if the word combines at most ``t`` of its rows.
    After the first ``j`` search bases have completed layer ``t`` and the others layer ``t - 1``,
    every word that none of them has visited must weigh at least the certificate of that step, with
    or without the bound of linear programming.
    """

    def get_coefficient_weights(generators: npt.NDArray[np.uint8]) -> dict[bytes, int]:
        return {
            (np.asarray(bits) @ generators % 2).astype(np.uint8).tobytes(): sum(bits)
            for bits in itertools.product((0, 1), repeat=len(generators))
        }

    words = get_coefficient_weights(basis)
    coefficient_weights = [get_coefficient_weights(generators) for generators in family.generators]
    certificate = family.certificate
    pivot_blocks = tuple(
        tuple(int(column) for column in pivots[:rank])
        for pivots, rank in zip(family.pivots, family.ranks, strict=True)
    )
    certificates = [
        certificate,
        dataclasses.replace(certificate, pivot_blocks=pivot_blocks, lp_max_denominator=DENOMINATOR),
    ]
    # Every search basis spans the same space as the original basis.
    assert all(weights.keys() == words.keys() for weights in coefficient_weights)
    weights = {word: np.count_nonzero(np.frombuffer(word, dtype=np.uint8)) for word in words}
    steps = [
        (layer, num_completed)
        for layer in range(1, len(basis))
        for num_completed in range(len(family.ranks) + 1)
    ]
    for word, weight in weights.items():
        assert weight % certificate.weight_divisor == 0
        for layer, num_completed in steps:
            completed = [
                layer if index < num_completed else layer - 1 for index in range(len(family.ranks))
            ]
            if all(item[word] > done for item, done in zip(coefficient_weights, completed)):
                for each in certificates:
                    assert weight >= each.step_lower_bound(layer, num_completed)

    # Deciding whether a step certifies a bound agrees with the bound of that step.
    for each, (layer, num_completed) in itertools.product(certificates, steps):
        bound = each.step_lower_bound(layer, num_completed)
        for upper_bound in range(1, basis.shape[1] + 2):
            assert each.certifies(layer, num_completed, upper_bound) == (bound >= upper_bound)
