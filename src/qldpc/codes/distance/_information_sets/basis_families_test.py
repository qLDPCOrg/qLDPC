# SPDX-License-Identifier: Apache-2.0

"""Tests for the basis families and certificates shared by BZ distance and counting."""

from __future__ import annotations

import dataclasses
import functools
import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

import qldpc
from qldpc import _compilation
from qldpc._binary import get_independent_rows, row_reduce_binary

from .._tuning import DEFAULT_TUNING, SearchTuning
from ..conftest import get_random_full_rank_matrix
from . import basis_families

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


@pytest.mark.parametrize("basis", [PARTIAL_BASIS, EVEN_BASIS])
def test_family_certificates(basis: npt.NDArray[np.uint8]) -> None:
    """Rank-deficient blocks and inferred parity give certificates valid for every word."""
    # A target above every weight keeps the rank-deficient block.
    basis_family = basis_families.get_disjoint_family(basis, None, random_column_orders=ORDERS)
    basis_family = basis_families.prune_family(basis_family, basis.shape[1] + 1)
    _assert_certificate_bounds_unvisited_words(basis, basis_family)


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
    """Pruning keeps full sets, and the partial sets that can help certify the target.

    A set of ten rows with rank deficiency ``d`` adds to the bound only once layer ``d`` is
    complete.  The full set alone certifies 6 once layer 5 is complete, so the set of deficiency 8
    cannot help with 6, but no layer lets the full set alone certify 12.  Without a full set, every
    set stays.
    """
    no_columns = np.empty((len(ranks), 10, 0), dtype=np.uint8)
    pivots = np.full((len(ranks), 10), -1, dtype=np.int64)
    basis_family = basis_families.BasisFamily(
        no_columns, no_columns, pivots, np.array(ranks, dtype=np.int64), 1, 1
    )
    pruned = basis_families.prune_family(basis_family, upper_bound)
    assert pruned.ranks.tolist() == retained
    assert pruned.certificate.work(upper_bound) <= basis_family.certificate.work(upper_bound)


def test_overlapping_family_certificates() -> None:
    """Check pivot coverage and label transformations against exhaustive codeword weights."""

    # Label each row separately so its row operations can be checked.
    basis = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    basis = get_independent_rows(basis)
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = basis_families.get_disjoint_family(basis, labels, random_column_orders=ORDERS)
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


def test_a_cheap_search_keeps_its_disjoint_basis_family() -> None:
    """No overlapping basis family is built if the work of the disjoint one is below the minimum."""
    basis = get_independent_rows(np.asarray(qldpc.codes.HammingCode(4).generator))
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = basis_families.get_disjoint_family(basis, labels, random_column_orders=ORDERS)
    assert not len(_get_overlapping_family(basis, labels, disjoint, 3).ranks)
    assert len(_get_overlapping_family(basis, labels, disjoint, 3, minimum_work=0).ranks)
    # A limit of zero sets builds no family at all.
    assert not len(_get_overlapping_family(basis, labels, disjoint, 3, 0, minimum_work=0).ranks)


def test_overlapping_family_guards() -> None:
    """Reject unhelpful coverage and terminate repeated attempts at the same pivot set."""

    # Complete or nearly complete disjoint coverage leaves no cheaper family within the size budget.
    basis = np.hstack([np.eye(25, dtype=np.uint8)] * 4)
    complete = basis_families.get_disjoint_family(basis, None, random_column_orders=ORDERS)
    assert not len(_get_overlapping_family(basis, None, complete, 26).ranks)
    shortened = basis[:, :-1]
    disjoint = basis_families.get_disjoint_family(shortened, None, random_column_orders=ORDERS)
    assert not len(_get_overlapping_family(shortened, None, disjoint, 26).ranks)

    # A zero basis has no information sets to replace.
    zeros = np.zeros((1, 3), dtype=np.uint8)
    empty = basis_families.get_disjoint_family(zeros, None, random_column_orders=ORDERS)
    assert not len(empty.ranks)
    assert not len(_get_overlapping_family(zeros, None, empty, 2).ranks)

    # An information set holds one of the third to fifth columns, which are equal, and two of the
    # first, second, and sixth, so attempts rediscover the few such sets.  The search must terminate
    # without a replacement that beats the disjoint family.
    small = np.array([[1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 1], [0, 0, 1, 1, 1, 0]], dtype=np.uint8)
    disjoint = basis_families.get_disjoint_family(small, None, random_column_orders=ORDERS)
    assert not len(_get_overlapping_family(small, None, disjoint, 3, minimum_work=0).ranks)


def test_partitioned_families_replace_costlier_ones() -> None:
    """Matroid partitioning replaces a disjoint family only in costly searches, and only if cheaper.

    This [11, 4] basis has disjoint information sets of ranks four, four, and two, and information
    sets that share columns certify the weight five with less work.
    """
    rows = ["10001011100", "01000001100", "00100000110", "00010011001"]
    basis = np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)
    labels = np.eye(len(basis), dtype=np.uint8)
    disjoint = basis_families.get_disjoint_family(basis, labels, random_column_orders=ORDERS)
    assert disjoint.ranks.tolist() == [4, 4, 2]

    def get_family(**changes: int) -> basis_families.BasisFamily:
        fields: dict[str, Any] = {"partition_work_ratio": 0, **changes}
        tuning = dataclasses.replace(DEFAULT_TUNING, **fields)
        return basis_families.get_partitioned_family(basis, labels, disjoint, 5, tuning)

    # A family needs a set to hold, a coverage of two or more, and partitioning that pays.
    assert not len(get_family(partition_work_ratio=2**62).ranks)
    assert not len(get_family(partition_cost_factor=2**40, partition_work_ratio=1).ranks)
    assert len(get_family(partition_cost_factor=0, partition_work_ratio=1).ranks)
    assert not len(get_family(max_information_sets=0).ranks)
    assert not len(get_family(max_partition_coverage=1).ranks)

    # A family that the search would switch to rules partitioning out if it does no more work than
    # any partitioned family could, and only then.
    tuning = dataclasses.replace(DEFAULT_TUNING, partition_work_ratio=0)
    for competing_work, partitions in ((0, False), (disjoint.certificate.work(5), True)):
        competing = basis_families.get_partitioned_family(
            basis,
            labels,
            disjoint,
            5,
            tuning,
            get_competing_work=functools.partial(int, competing_work),
        )
        assert bool(len(competing.ranks)) == partitions
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
        _assert_certificate_bounds_unvisited_words(basis, family)

    one_full_set = np.array([[1, 1, 1, 0], [0, 0, 0, 1]], dtype=np.uint8)
    for basis in (PARTIAL_BASIS, one_full_set):
        disjoint = basis_families.get_disjoint_family(basis, None, random_column_orders=0)
        assert not len(basis_families.get_maximum_disjoint_family(basis, disjoint).ranks)


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
        family = basis_families.BasisFamily(
            np.array([reduced for reduced, _, _ in sets]),
            np.zeros((num_sets, dimension, 0), dtype=np.uint8),
            pivots,
            np.full(num_sets, dimension, dtype=np.int64),
            basis_families.infer_weight_divisor(basis),
            int(np.bincount(pivots.ravel()).max()),
        )
        _assert_certificate_bounds_unvisited_words(basis, family)


def test_searches_of_overlapping_sets_use_linear_programming_when_costly() -> None:
    """Only a costly search of overlapping sets gets the bound of linear programming."""
    basis = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    disjoint = basis_families.get_disjoint_family(basis, None, random_column_orders=ORDERS)
    overlapping = _get_overlapping_family(basis, None, disjoint, 3, minimum_work=0)
    assert overlapping.max_coverage > 1
    assert not overlapping.get_search_certificate(3, 2**62, DENOMINATOR).pivot_blocks
    assert len(overlapping.get_search_certificate(3, 0, DENOMINATOR).pivot_blocks) == len(
        overlapping.ranks
    )
    assert not disjoint.get_search_certificate(3, 0, DENOMINATOR).pivot_blocks


@pytest.mark.parametrize("max_sets, num_sets", [(1, 0), (3, 0), (4, 4), (16, 4)])
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


def _assert_certificate_bounds_unvisited_words(
    basis: npt.NDArray[np.uint8], family: basis_families.BasisFamily
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
