# SPDX-License-Identifier: Apache-2.0

"""Tests for the basis families and certificates shared by BZ distance and counting."""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

import qldpc
from qldpc import _compilation
from qldpc._binary import get_independent_rows

from .._tuning import DEFAULT_TUNING, MAX_VALUE, SearchTuning
from ..conftest import assert_certificate_bounds_unvisited_words, get_random_full_rank_matrix
from . import basis_families, certificates, matroid_partition

# The default number of random column orders of disjoint families.
ORDERS = DEFAULT_TUNING.disjoint_random_column_orders

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
    get_family = functools.partial(basis_families.get_disjoint_family, random_column_orders=ORDERS)
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

    # A packing of k rows keeps at most 2**k - 1 full sets, so here 7 of the 8 that fit, and a
    # natural order that reaches that limit needs no random orders.
    three_rows = np.tile(np.eye(3, dtype=np.uint8), 8)
    packing = basis_families._get_disjoint_sets_for_columns
    with mock.patch.object(basis_families, "_get_disjoint_sets_for_columns", wraps=packing) as spy:
        assert get_family(three_rows, None).ranks.tolist() == [3] * 7
    assert spy.call_count == 1

    # Each set takes its pivots from the remaining columns in the given order, so a random column
    # order shapes every set of a packing.
    basis = np.array([[1, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 1]], dtype=np.uint8)
    columns = np.array([4, 5, 1, 0, 3, 2])
    sets = basis_families._get_disjoint_sets_for_columns(basis, np.eye(2, dtype=np.uint8), columns)
    assert [list(pivots) for *_, pivots in sets] == [[4, 1], [5, 0], [3, 2]]


def test_a_basis_family_stacks_its_sets() -> None:
    """The sets of a basis family are reduced on their pivots, which are padded for partial sets."""
    labels = np.eye(3, dtype=np.uint8)
    basis_family = basis_families.get_disjoint_family(
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
    no_labels = basis_families.get_disjoint_family(PARTIAL_BASIS, None, random_column_orders=ORDERS)
    assert no_labels.labels.shape == (3, 3, 0)


def test_a_dependent_set_of_columns_is_an_error() -> None:
    """A basis family rejects a dependent set of columns, which only a broken caller would pass.

    The check is not an ``assert``, so it also holds under ``python -O``.
    """
    basis = np.array([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=np.uint8)  # columns 0 and 1 are equal
    labels = np.eye(2, dtype=np.uint8)
    with pytest.raises(RuntimeError, match="not independent"):
        basis_families._get_family_of_sets(basis, labels, [[0, 1]], weight_divisor=1)


@pytest.mark.parametrize("basis", [PARTIAL_BASIS, EVEN_BASIS])
def test_family_certificates(basis: npt.NDArray[np.uint8]) -> None:
    """Rank-deficient blocks and inferred parity give certificates valid for every word."""
    # A target above every weight keeps the rank-deficient block.
    basis_family = basis_families.get_disjoint_family(basis, None, random_column_orders=ORDERS)
    basis_family = basis_families.prune_family(basis_family, basis.shape[1] + 1)
    assert_certificate_bounds_unvisited_words(basis, basis_family)


def test_weight_divisor_inference() -> None:
    """Even rows span only even weights, and a divisor proven by the caller is retained."""
    odd = np.array([[1, 0, 0, 0], [0, 1, 1, 0]], dtype=np.uint8)
    get_family = functools.partial(basis_families.get_disjoint_family, random_column_orders=ORDERS)
    assert get_family(EVEN_BASIS, None).certificate.weight_divisor == 2
    assert get_family(odd, None).certificate.weight_divisor == 1
    assert get_family(odd, None, weight_divisor=2).certificate.weight_divisor == 2


@pytest.mark.parametrize(
    "ranks, upper_bound, retained",
    [([10, 6, 2], 6, [10, 6]), ([10, 6, 2], 12, [10, 6, 2]), ([6, 2], 6, [6, 2])],
)
def test_prune_family(ranks: list[int], upper_bound: int, retained: list[int]) -> None:
    """Pruning keeps full sets, and the partial sets that can help certify the target."""
    # A set of ten rows and deficiency d adds to the bound only from layer d.  The full set alone
    # certifies 6 after layer 5, before the set of deficiency 8 adds anything, but never 12.
    no_columns = np.empty((len(ranks), 10, 0), dtype=np.uint8)
    pivots = np.full((len(ranks), 10), -1, dtype=np.int64)
    basis_family = basis_families.BasisFamily(
        no_columns, no_columns, pivots, np.array(ranks, dtype=np.int64), 1, 1
    )
    pruned = basis_families.prune_family(basis_family, upper_bound)
    assert pruned.ranks.tolist() == retained
    assert pruned.certificate.work(upper_bound) <= basis_family.certificate.work(upper_bound)


def test_overlapping_family_certificates() -> None:
    """Pivot coverage and label transformations agree with exhaustive codeword weights."""

    # Label each row separately so its row operations can be checked.
    basis = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    basis = get_independent_rows(basis)
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = basis_families.get_disjoint_family(basis, labels, random_column_orders=ORDERS)
    basis_family = _get_overlapping_family(basis, labels, disjoint, 3, minimum_work=0)
    assert np.all(basis_family.ranks == len(basis))
    loads = np.bincount(basis_family.pivots.ravel(), minlength=basis.shape[1])
    assert basis_family.max_coverage >= loads.max() > 1
    for generators, set_labels, pivots in zip(*basis_family[:3], strict=True):
        np.testing.assert_array_equal(generators, set_labels @ basis % 2)
        np.testing.assert_array_equal(generators[:, pivots], np.eye(len(basis)))
    assert_certificate_bounds_unvisited_words(basis, basis_family)


def test_overlapping_family_guards() -> None:
    """Unhelpful coverage is rejected, and repeated attempts at one pivot set terminate."""

    # Nearly complete disjoint coverage leaves no cheaper family within the size budget.  The basis
    # is C-ordered, as the searches pass it, so that numba compiles no version for another layout.
    basis = np.hstack([np.eye(25, dtype=np.uint8)] * 4)
    shortened = np.ascontiguousarray(basis[:, :-1])
    disjoint = basis_families.get_disjoint_family(shortened, None, random_column_orders=ORDERS)
    assert not len(_get_overlapping_family(shortened, None, disjoint, 26, minimum_work=0).ranks)

    # A zero basis has no information sets to replace.
    zeros = np.zeros((1, 3), dtype=np.uint8)
    empty = basis_families.get_disjoint_family(zeros, None, random_column_orders=ORDERS)
    assert not len(empty.ranks)
    assert not len(_get_overlapping_family(zeros, None, empty, 2).ranks)

    # An information set holds one of the third to fifth columns, which are equal, and two of the
    # first, second, and sixth, so attempts rediscover the few such sets.
    small = np.array([[1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 1, 0]], dtype=np.uint8)
    disjoint = basis_families.get_disjoint_family(small, None, random_column_orders=ORDERS)
    assert not len(_get_overlapping_family(small, None, disjoint, 3, minimum_work=0).ranks)


def test_the_checks_of_stage_three_say_which_keeps_a_family() -> None:
    """The checks of stage three say which keeps a family, and the works that they compare."""
    hamming = get_independent_rows(np.asarray(qldpc.codes.HammingCode(4).generator))
    disjoint = basis_families.get_disjoint_family(hamming, None, random_column_orders=ORDERS)
    complete = np.hstack([np.eye(4, dtype=np.uint8)] * 2)
    packed = basis_families.get_disjoint_family(complete, None, random_column_orders=ORDERS)
    check = basis_families._OverlappingCheck
    cases: list[
        tuple[npt.NDArray[np.uint8], basis_families.BasisFamily, int, dict[str, Any], Any]
    ] = [
        # Nothing is left to extend at a limit of zero sets, or when full sets use every column.
        (hamming, disjoint, 3, {"max_information_sets": 0}, check.NOTHING_TO_EXTEND),
        (complete, packed, 3, {"overlapping_minimum_work": 0}, check.NOTHING_TO_EXTEND),
        # The Hamming code's information set and partial set certify the weight three cheaply.
        (hamming, disjoint, 3, {}, check.BELOW_MINIMUM_WORK),
        # One full set certifies a target of one after its first layer, which no family improves.
        (hamming, disjoint, 1, {"overlapping_minimum_work": 0}, check.CANNOT_DO_LESS_WORK),
        # At the weight three, more full sets could do less work, and four sets could do less than
        # three.
        (
            hamming,
            disjoint,
            3,
            {"overlapping_minimum_work": 0, "max_information_sets": 4},
            check.NONE,
        ),
    ]
    for basis, family, upper_bound, changes, expected in cases:
        policy = basis_families.get_overlapping_family_policy(SearchTuning(**changes), len(basis))
        found, work, optimistic_work = basis_families._check_overlapping_family(
            basis, family, upper_bound, policy
        )
        assert found == expected
        if found == check.NOTHING_TO_EXTEND:
            assert work == optimistic_work == -1
            continue
        assert work == family.certificate.work(upper_bound)
        if found == check.BELOW_MINIMUM_WORK:
            assert optimistic_work == -1
            continue

        # The optimistic work reaches the work exactly when no family could do less.
        assert (optimistic_work >= work) == (found == check.CANNOT_DO_LESS_WORK)


def test_partitioned_families_replace_costlier_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    """Matroid partitioning replaces a disjoint family only in costly searches, if cheaper."""
    rows = ["10001011100", "01000001100", "00100000110", "00010011001"]
    basis = np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = basis_families.get_disjoint_family(basis, labels, random_column_orders=ORDERS)
    assert disjoint.ranks.tolist() == [4, 4, 2]

    # Record the coverages that partitioning tries.
    coverages: list[int] = []
    get_sets = matroid_partition.get_information_sets_with_coverage

    def record_coverage(basis: npt.NDArray[np.uint8], coverage: int, max_sets: int) -> Any:
        coverages.append(coverage)
        return get_sets(basis, coverage, max_sets)

    monkeypatch.setattr(matroid_partition, "get_information_sets_with_coverage", record_coverage)

    def get_family(**changes: int) -> basis_families.BasisFamily:
        fields: dict[str, Any] = {
            "partition_work_ratio": 0,
            "max_information_sets": 16,
            "partition_max_coverage": MAX_VALUE,
            **changes,
        }
        tuning = dataclasses.replace(DEFAULT_TUNING, **fields)
        coverages.clear()
        return basis_families.get_partitioned_family(basis, labels, disjoint, 5, tuning)

    # Without a limit on coverage, a family needs a set to hold, a coverage of two or more, and
    # partitioning that pays.
    assert not len(get_family(partition_work_ratio=2**62).ranks)
    assert not len(get_family(max_information_sets=0).ranks)

    # Partitioning may try each coverage from two up to the least at which the most sets allowed
    # could fit in the 10 nonzero columns, by counting their pivots: ceil(16 * 4 / 10) = 7, but
    # ceil(2 * 4 / 10) = 1 for two sets.  Here it tries each, since even two sets, fewer than fit
    # within any of these coverages, could do less work (14) than any family that it finds (16 or
    # more).
    assert not len(get_family(max_information_sets=2).ranks) and not coverages
    family = get_family()
    assert sorted(coverages) == list(range(2, 8))
    assert family.max_coverage > 1
    assert family.certificate.work(5) < disjoint.certificate.work(5)
    for generators, set_labels, pivots in zip(*family[:3], strict=True):
        np.testing.assert_array_equal(generators, set_labels @ basis % 2)
        np.testing.assert_array_equal(generators[:, pivots], np.eye(len(basis)))
    assert_certificate_bounds_unvisited_words(basis, family)

    # A limit on coverage caps the coverages that partitioning tries.
    assert 1 < get_family(partition_max_coverage=3).max_coverage <= 3
    assert sorted(coverages) == [2, 3]

    # A competing family that does no work rules partitioning out, and one whose work is MAX_VALUE
    # does not.  Partitioning asks for its work only if it may try some coverage and passes its
    # gate, and some partitioned family could do less work.
    tuning = dataclasses.replace(
        DEFAULT_TUNING,
        partition_work_ratio=0,
        max_information_sets=16,
        partition_max_coverage=MAX_VALUE,
    )
    for competing_work, partitions in ((0, False), (MAX_VALUE, True)):
        partitioned = basis_families.get_partitioned_family(
            basis,
            labels,
            disjoint,
            5,
            tuning,
            get_competing_work=functools.partial(int, competing_work),
        )
        assert bool(len(partitioned.ranks)) == partitions
    # For the weight four, the disjoint family already does the least work that any family could.
    gate_changes: list[tuple[dict[str, Any], int]] = [
        ({"partition_work_ratio": 0, "max_information_sets": 0}, 5),
        ({"partition_work_ratio": 0, "max_information_sets": 2}, 5),
        ({"partition_work_ratio": 0}, 4),
    ]
    for changes, upper_bound in gate_changes:
        get_competing_work = mock.Mock(return_value=0)
        basis_families.get_partitioned_family(
            basis,
            labels,
            disjoint,
            upper_bound,
            dataclasses.replace(tuning, **changes),
            get_competing_work=get_competing_work,
        )
        get_competing_work.assert_not_called()


def test_partitioning_runs_only_past_its_gate() -> None:
    """Matroid partitioning runs only once the work of a search reaches its ratio times its unit.

    It asks for competing work at a work ratio of zero, never at the largest, and at a ratio of one
    for work far above its unit but not far below.  Its plan tries every coverage from two up to the
    most that its information sets could reach.
    """
    # The coverages agree with their definition for largest coverages from zero to eight, on a basis
    # of 4 rows and 10 nonzero columns.
    basis = np.hstack([np.eye(4), np.eye(4), np.eye(4)[:, :2], np.zeros((4, 1))]).astype(np.uint8)
    for max_sets in range(20):
        plan = basis_families._get_partition_plan(
            basis, SearchTuning(max_information_sets=max_sets, partition_max_coverage=MAX_VALUE)
        )
        assert plan.coverages == range(2, math.ceil(max_sets * 4 / 10) + 1)

    basis = np.hstack([np.eye(20), np.eye(20), np.eye(20)[:, :5]]).astype(np.uint8)
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = basis_families.get_disjoint_family(basis, labels, random_column_orders=ORDERS)
    assert disjoint.ranks.tolist() == [20, 20, 5]
    # Five full sets of 20 rows could fit in the 45 nonzero columns from the coverage
    # ceil(5 * 20 / 45) = 3, so partitioning may try the coverages two and three, or only coverage
    # one at a limit of one.  Either way, it asks for the work of a competing family only if it
    # passes its gate, which a ratio of zero does, but not a ratio of MAX_VALUE.  At a ratio of one,
    # the work of the disjoint family passes the gate at the upper bound 45, at least 13 times the
    # unit of either plan, but not at the upper bound 7, at most a tenth of it.
    for coverage_limit in (MAX_VALUE, 1):
        unit = basis_families._get_partition_plan(
            basis, SearchTuning(max_information_sets=5, partition_max_coverage=coverage_limit)
        ).unit
        assert disjoint.certificate.work(7) * 10 <= unit <= disjoint.certificate.work(45) // 13
        for partition_work_ratio, upper_bound, passes in (
            (0, 19, True),
            (MAX_VALUE, 19, False),
            (1, 7, False),
            (1, 45, True),
        ):
            get_competing_work = mock.Mock(return_value=0)
            tuning = SearchTuning(
                max_information_sets=5,
                partition_work_ratio=partition_work_ratio,
                partition_max_coverage=coverage_limit,
            )
            basis_families.get_partitioned_family(
                basis, labels, disjoint, upper_bound, tuning, get_competing_work=get_competing_work
            )
            assert get_competing_work.called == passes


def test_partitioning_skips_coverages_at_which_no_family_could_replace_the_best() -> None:
    """Partitioning skips a coverage at which no family could do less work than the best so far.

    It takes the family that trying every coverage would.
    """
    rows = [
        "00101101010",
        "01011111101",
        "10000110010",
        "00010110001",
        "00001000101",
        "10001110001",
    ]
    basis = np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)
    labels = np.eye(len(basis), dtype=np.uint8)
    partition = matroid_partition.get_information_sets_with_coverage

    # The families that partitioning finds at each coverage that it may try, from two up to
    # ceil(16 * 6 / 11) = 9, and the work of each to certify the weight four.
    families = {
        coverage: basis_families._get_family_of_sets(
            basis, labels, partition(basis, coverage, 16), weight_divisor=1
        )
        for coverage in range(2, 10)
    }
    works = {coverage: family.certificate.work(4) for coverage, family in families.items()}

    # Trying every coverage, given the family of coverage two, takes the family of least work, of
    # the first coverage that reaches it, if that is less than the given family's work.
    expected = min((work, coverage) for coverage, work in works.items() if work < works[2])[1]
    assert expected == 3

    # The given family does the least work that any family of at most three sets could, the most
    # that fit within coverage two, so partitioning skips that coverage.  The family of coverage
    # three, which partitioning tries and takes, does the least work that any family of at most 16
    # sets could, so partitioning skips every coverage above three.
    tuning = SearchTuning(
        partition_work_ratio=0, max_information_sets=16, partition_max_coverage=MAX_VALUE
    )
    with mock.patch.object(
        matroid_partition, "get_information_sets_with_coverage", wraps=partition
    ) as spy:
        family = basis_families.get_partitioned_family(basis, labels, families[2], 4, tuning)
    assert [call.args[1] for call in spy.call_args_list] == [expected]
    np.testing.assert_array_equal(family.pivots, families[expected].pivots)

    # Here the bound at coverage two is exact.  The columns e1, e2, e3, e4, e1 + e3, and e2 + e4
    # hold three information sets that use each column exactly twice, floor(2 * 6 / 4) = 3, and the
    # rows are even, so the bound rounds to the weight divisor two.  These sets do less work than
    # the given set of the first four columns, so partitioning tries coverage two and takes them.
    columns = [1, 2, 4, 8, 5, 10]
    basis = ((np.array(columns) >> np.arange(4)[:, None]) & 1).astype(np.uint8)
    labels = np.eye(4, dtype=np.uint8)
    single = basis_families._get_family_of_sets(basis, labels, [[0, 1, 2, 3]], weight_divisor=2)
    family = basis_families.get_partitioned_family(basis, labels, single, 4, tuning)
    assert family.ranks.tolist() == [4, 4, 4] and family.max_coverage == 2


def test_partitioned_families_do_at_least_the_work_of_their_coverage_bound(
    pytestconfig: pytest.Config,
) -> None:
    """Each family that partitioning finds does at least the work of the bound for its coverage.

    The bound at a coverage ``b`` is that of ``_estimate_optimistic_work`` over the most sets that
    fit within ``b``.  Partitioning skips ``b`` if the bound is at least the work of the best family
    so far (Section 4.4 of ``brouwer_zimmermann_code.md``).
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    max_sets = 8
    for _ in range(8):
        dimension = int(rng.integers(2, 6))
        basis = get_random_full_rank_matrix(
            rng, dimension, int(rng.integers(dimension, 4 * dimension + 2))
        )
        num_columns = int(np.count_nonzero(np.any(basis, axis=0)))
        weight_divisor = basis_families.infer_weight_divisor(basis)
        binomials = certificates.get_capped_binomials(dimension)
        max_coverage = math.ceil(max_sets * dimension / num_columns)
        for coverage in range(2, max_coverage + 1):
            family = basis_families._get_family_of_sets(
                basis,
                np.empty((dimension, 0), dtype=np.uint8),
                matroid_partition.get_information_sets_with_coverage(basis, coverage, max_sets),
                weight_divisor,
            )
            most_sets = min(max_sets, coverage * num_columns // dimension)
            for upper_bound in range(1, num_columns + 2):
                bound = basis_families._estimate_optimistic_work(
                    dimension, num_columns, weight_divisor, upper_bound, most_sets, binomials
                )
                assert bound <= family.certificate.work(upper_bound)


def test_maximum_disjoint_families_replace_smaller_packings() -> None:
    """A maximum disjoint family has the most information sets, then the largest partial set."""
    # A greedy packing of e1, e2, e3, e1 + e2, e1 + e2 leaves a partial set of rank one, but
    # e1 + e2, e2, e3 leave rank two.  Of e1, e2, e1 + e2, e1 + e2, it finds one of two sets.
    for columns, ranks in (([1, 2, 4, 3, 3], [3, 2]), ([1, 2, 3, 3], [2, 2])):
        dimension = max(columns).bit_length()
        basis = ((np.array(columns) >> np.arange(dimension)[:, None]) & 1).astype(np.uint8)
        disjoint = basis_families.get_disjoint_family(basis, None, random_column_orders=0)
        assert disjoint.ranks.tolist() == [dimension, 1]
        family = basis_families.get_maximum_disjoint_family(basis, disjoint)
        assert family.ranks.tolist() == ranks
        pivots = family.pivots[family.pivots >= 0]
        assert len(set(pivots.tolist())) == len(pivots)
        for generators, set_pivots, rank in zip(family.generators, family.pivots, family.ranks):
            np.testing.assert_array_equal(
                generators[:, set_pivots[:rank]], np.eye(dimension)[:, :rank]
            )
        assert_certificate_bounds_unvisited_words(basis, family)

    # A packing whose sets use every nonzero column, as that of PARTIAL_BASIS does, skips
    # partitioning.  That of one_full_set leaves a column, so it partitions, but no family replaces
    # it.
    one_full_set = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
    partition = matroid_partition.get_information_sets_with_coverage
    for basis, num_partitions in ((PARTIAL_BASIS, 0), (one_full_set, 1)):
        disjoint = basis_families.get_disjoint_family(basis, None, random_column_orders=0)
        with mock.patch.object(
            matroid_partition, "get_information_sets_with_coverage", wraps=partition
        ) as spy:
            assert not len(basis_families.get_maximum_disjoint_family(basis, disjoint).ranks)
        assert spy.call_count == num_partitions

    # A maximum family keeps at most the 7 information sets of a packing, of the 8 that fit here.
    # So it replaces a family of 3, but a family of 7 skips partitioning.
    three_rows = np.tile(np.eye(3, dtype=np.uint8), 8)
    capped = basis_families.get_disjoint_family(three_rows, None, random_column_orders=0)
    smaller = basis_families.BasisFamily(
        capped.generators[:3],
        capped.labels[:3],
        capped.pivots[:3],
        capped.ranks[:3],
        capped.weight_divisor,
        1,
    )
    with mock.patch.object(
        matroid_partition, "get_information_sets_with_coverage", wraps=partition
    ) as spy:
        assert not len(basis_families.get_maximum_disjoint_family(three_rows, capped).ranks)
        spy.assert_not_called()
        family = basis_families.get_maximum_disjoint_family(three_rows, smaller)
        spy.assert_called_once()
    assert family.ranks.tolist() == [3] * 7


def test_a_coverage_limit_of_one_partitions_into_a_maximum_disjoint_family() -> None:
    """At a coverage limit of one, partitioning finds a maximum disjoint family, and prunes it.

    It does so whatever ``max_information_sets`` is, and only if some disjoint family could do less
    work.
    """
    # A greedy packing of e1, e2, e3, e1 + e2, e1 + e2, e3, e1 + e2 + e3 keeps ranks three and two,
    # but the columns hold two disjoint information sets and a partial set of the last column, which
    # adds nothing to the certificate before the two sets certify four.
    columns = [1, 2, 4, 3, 3, 4, 7]
    basis = ((np.array(columns) >> np.arange(3)[:, None]) & 1).astype(np.uint8)
    labels = np.eye(3, dtype=np.uint8)
    disjoint = basis_families.get_disjoint_family(basis, labels, random_column_orders=0)
    assert disjoint.ranks.tolist() == [3, 2]
    for max_sets in (16, 0):
        tuning = SearchTuning(
            partition_max_coverage=1, partition_work_ratio=0, max_information_sets=max_sets
        )
        family = basis_families.get_partitioned_family(basis, labels, disjoint, 4, tuning)
        assert family.ranks.tolist() == [3, 3] and family.max_coverage == 1
    assert family.certificate.work(4) < disjoint.certificate.work(4)
    assert len(set(family.pivots.ravel().tolist())) == 6
    for generators, set_labels, pivots in zip(*family[:3], strict=True):
        np.testing.assert_array_equal(generators, set_labels @ basis % 2)
        np.testing.assert_array_equal(generators[:, pivots], np.eye(3))
    assert_certificate_bounds_unvisited_words(basis, family)

    # No disjoint family does less work than a single information set that uses every column, so
    # partitioning returns before it asks for the work of a competing family.
    identity = np.eye(3, dtype=np.uint8)
    single = basis_families.get_disjoint_family(identity, labels, random_column_orders=0)
    get_competing_work = mock.Mock()
    partitioned = basis_families.get_partitioned_family(
        identity, labels, single, 3, tuning, get_competing_work=get_competing_work
    )
    assert not len(partitioned.ranks)
    get_competing_work.assert_not_called()


def test_a_coverage_limit_of_one_skips_coverage_one_at_the_pruned_disjoint_rank_bound() -> None:
    """At a coverage limit of one, partitioning is skipped at the pruned disjoint rank bound.

    A given family that is disjoint and has the pruned bound has the ranks of every pruned maximum
    disjoint family, so the skip leaves the result unchanged (Section 4.4 of
    ``brouwer_zimmermann_code.md``).
    """
    with_zero = np.hstack([PARTIAL_BASIS, np.zeros((3, 1), dtype=np.uint8)])
    three_rows = np.tile(np.eye(3, dtype=np.uint8), 8)
    one_extra = np.hstack([np.eye(3), np.eye(3)[:, :1]]).astype(np.uint8)
    # The columns e1, e2, e3, e1 + e2, e1 + e2.
    smaller_partial = ((np.array([1, 2, 4, 3, 3]) >> np.arange(3)[:, None]) & 1).astype(np.uint8)
    # The columns e1, e2, e1 + e2, e1 + e2, and a family of two sets that share e1.
    two_rows = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    overlapping = basis_families._get_family_of_sets(
        two_rows, np.eye(2, dtype=np.uint8), [[0, 1], [0, 2]], weight_divisor=1
    )
    assert overlapping.max_coverage == 2

    def get_packing(basis: npt.NDArray[np.uint8]) -> basis_families.BasisFamily:
        labels = np.eye(len(basis), dtype=np.uint8)
        return basis_families.get_disjoint_family(basis, labels, random_column_orders=0)

    tuning = SearchTuning(partition_max_coverage=1, partition_work_ratio=0)
    get_sets = basis_families._get_maximum_disjoint_sets
    cases = [
        # The pruned packing has two information sets and a partial set of rank two on the other two
        # nonzero columns.  The bound does not count the zero column.
        (with_zero, basis_families.prune_family(get_packing(with_zero), 5), 5, True),
        # The pruned packing has the limit of 7 information sets.
        (three_rows, basis_families.prune_family(get_packing(three_rows), 11), 11, True),
        # The packing has a full set and a partial set of rank one, which pruning for the bound
        # three drops.
        (one_extra, basis_families.prune_family(get_packing(one_extra), 3), 3, True),
        # The packing also has ranks three and one, but the rows are even, so pruning for the bound
        # four drops a partial set of rank one, and the pruned maximum disjoint family replaces the
        # unpruned packing.
        (EVEN_BASIS, get_packing(EVEN_BASIS), 4, False),
        # The pruned packing has a partial set of rank one, but a maximum disjoint family has one of
        # rank two, which pruning for the bound four keeps.
        (smaller_partial, basis_families.prune_family(get_packing(smaller_partial), 4), 4, False),
        # Two disjoint information sets replace two sets that share e1.
        (two_rows, overlapping, 3, False),
    ]
    for basis, given, upper_bound, skips in cases:
        labels = np.eye(len(basis), dtype=np.uint8)

        # Trying coverage one builds the pruned maximum disjoint family, which replaces the given
        # family exactly in the cases in which partitioning does not skip coverage one.
        maximum = basis_families._get_family_of_sets(
            basis, labels, get_sets(basis), given.weight_divisor
        )
        maximum = basis_families.prune_family(maximum, upper_bound)
        replaces = maximum.certificate.work(upper_bound) < given.certificate.work(upper_bound)
        assert replaces == (not skips)

        # That partitioning asks for the work of a competing family shows that it passes its gate,
        # and that some disjoint family could do less work.
        get_competing_work = mock.Mock(return_value=MAX_VALUE)
        with mock.patch.object(basis_families, "_get_maximum_disjoint_sets", wraps=get_sets) as spy:
            family = basis_families.get_partitioned_family(
                basis, labels, given, upper_bound, tuning, get_competing_work=get_competing_work
            )
        get_competing_work.assert_called_once()
        assert spy.called == (not skips)
        expected = maximum.pivots if replaces else np.empty((0, len(basis)))
        np.testing.assert_array_equal(family.pivots, expected)


def test_built_disjoint_families_do_at_least_the_disjoint_bound(
    pytestconfig: pytest.Config,
) -> None:
    """Built disjoint families do at least the work of ``_estimate_optimistic_disjoint_work``.

    The test checks maximum disjoint families and greedy packings, pruned or not, on fixed and
    random bases.

    The columns ``e1 e2 e3 e4 e1 e2 e3`` expose a bound over only ``floor(N / r)`` sets at the upper
    bound three, and ``e1 e1 e1 e1 e2`` one over only ``ceil(N / r)`` sets at six.  Four copies of
    ``e1 e2 e3``, whose rows are even, expose one that takes at most ``r`` sets at five, or ignores
    the weight divisor at six.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    bases = [
        ((np.array(columns) >> np.arange(rows)[:, None]) & 1).astype(np.uint8)
        for columns, rows in (([1, 2, 4, 8, 1, 2, 4], 4), ([1, 1, 1, 1, 2], 2))
    ]
    bases.append(np.tile(np.eye(3, dtype=np.uint8), 4))
    for _ in range(12):
        rows = int(rng.integers(1, 6))
        bases.append(get_random_full_rank_matrix(rng, rows, int(rng.integers(rows, 4 * rows + 2))))
    for basis in bases:
        dimension = len(basis)
        num_columns = int(np.count_nonzero(np.any(basis, axis=0)))
        weight_divisor = basis_families.infer_weight_divisor(basis)
        binomials = certificates.get_capped_binomials(dimension)
        maximum = basis_families._get_family_of_sets(
            basis,
            np.empty((dimension, 0), dtype=np.uint8),
            basis_families._get_maximum_disjoint_sets(basis),
            weight_divisor,
        )
        greedy = basis_families.get_disjoint_family(basis, None, random_column_orders=0)
        for upper_bound in range(1, num_columns + 2):
            bound = basis_families._estimate_optimistic_disjoint_work(
                dimension, num_columns, weight_divisor, upper_bound, binomials
            )
            for family in (maximum, greedy):
                for each in (family, basis_families.prune_family(family, upper_bound)):
                    assert bound <= each.certificate.work(upper_bound)


@pytest.mark.parametrize("max_sets, num_sets", [(3, 0), (4, 4), (16, 4)])
def test_the_size_limit_bounds_the_overlapping_basis_family(max_sets: int, num_sets: int) -> None:
    """A cheaper basis family of overlapping sets needs four sets, so a lower limit finds none."""
    basis = get_independent_rows(np.asarray(qldpc.codes.HammingCode(4).generator))
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = basis_families.get_disjoint_family(basis, labels, random_column_orders=ORDERS)
    overlapping = _get_overlapping_family(basis, labels, disjoint, 3, max_sets, minimum_work=0)
    assert len(overlapping.ranks) == num_sets


@pytest.mark.usefixtures("plain_search_setup")
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
                disjoint = basis_families.get_disjoint_family(
                    basis, labels, random_column_orders=ORDERS
                )
                pruned = basis_families.prune_family(disjoint, upper_bound)
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


def _get_overlapping_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    basis_family: basis_families.BasisFamily,
    upper_bound: int,
    max_sets: int = DEFAULT_TUNING.max_information_sets,
    *,
    minimum_work: int = DEFAULT_TUNING.overlapping_minimum_work,
) -> basis_families.BasisFamily:
    """Stage three, as a search whose tuning has these thresholds calls it."""
    tuning = SearchTuning(max_information_sets=max_sets, overlapping_minimum_work=minimum_work)
    return basis_families.get_overlapping_family(
        basis,
        np.empty((len(basis), 0), dtype=np.uint8) if labels is None else labels,
        basis_family,
        upper_bound,
        basis_families.get_overlapping_family_policy(tuning, len(basis)),
    )
