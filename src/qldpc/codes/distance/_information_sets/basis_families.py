# SPDX-License-Identifier: Apache-2.0

"""Basis families of information sets, shared by exact BZ and coset searches and by counts.

A basis family holds the information sets of one basis that a search scans and certifies together.
Distance searches and counts build families in stages, with their own exits between them:
``get_disjoint_family``, ``prune_family``, ``get_overlapping_family``, and
``get_partitioned_family``.  Each stage aims at a target ``upper_bound``: the best weight ``U`` of a
distance search, or ``t + 1`` for a count of the words of weight ``t``.  Sections cited in this
module are those of ``brouwer_zimmermann.md`` and ``brouwer_zimmermann_code.md``, whose Section 4.4
gives the rules of the stages and of the families of the coset searches.
"""

from __future__ import annotations

import dataclasses
import enum
import math
from collections.abc import Callable
from typing import NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits, row_reduce_binary, row_reduce_packed, unpack_bits
from qldpc._compilation import always_compiled, compiled_on_request

from .. import _tuning
from .._binary import get_nonzero_columns
from .._tuning import SearchTuning
from . import matroid_partition
from .certificates import (
    WORK_CAP,
    Certificate,
    estimate_certificate_work,
    get_capped_binomials,
    lower_bound_from_occupancy,
)

# One disjoint information set: its rows, its labels, and its pivot columns.
_DisjointSet = tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.int_]]

# The scalar type of an array.
_Scalar = TypeVar("_Scalar", bound=np.generic)


####################################################################################################
# basis families


class BasisFamily(NamedTuple):
    """The information sets of one binary basis that a search uses together, stacked into arrays.

    Each information set is the basis, row reduced on a block of pivot columns.  ``generators`` has
    shape ``(num_sets, dimension, num_columns)``, and ``labels``, the quotient labels that follow
    the row operations, has the same first two axes, with no columns for a basis without labels.
    ``pivots`` has shape ``(num_sets, dimension)``, padded with -1 beyond each set's rank.
    ``ranks``, ``weight_divisor``, and ``max_coverage`` are the parameters of its
    :attr:`certificate`.  Compiled code can take the family, since it holds only arrays and
    integers.
    """

    generators: npt.NDArray[np.uint8]
    labels: npt.NDArray[np.uint8]
    pivots: npt.NDArray[np.int64]
    ranks: npt.NDArray[np.int64]
    weight_divisor: int
    max_coverage: int

    @property
    def certificate(self) -> Certificate:
        """The certificate that the complete layers of this basis family give."""
        return Certificate(
            self.generators.shape[1],
            tuple(int(rank) for rank in self.ranks),
            self.weight_divisor,
            self.max_coverage,
        )

    def get_search_certificate(
        self, upper_bound: int, lp_minimum_work: int, lp_max_denominator: int
    ) -> Certificate:
        """The certificate with which to search this basis family for ``upper_bound``.

        A search of overlapping sets whose estimated work reaches ``lp_minimum_work`` also solves
        linear programs (Section 4.4), with dual weights of denominators at most
        ``lp_max_denominator``.
        """
        certificate = self.certificate
        if self.max_coverage == 1 or certificate.work(upper_bound) < lp_minimum_work:
            return certificate
        return dataclasses.replace(
            certificate,
            pivot_blocks=get_pivot_blocks(self.pivots, self.ranks),
            lp_max_denominator=lp_max_denominator,
        )


def get_pivot_blocks(
    pivots: npt.NDArray[np.int64], ranks: npt.NDArray[np.int64]
) -> tuple[tuple[int, ...], ...]:
    """The pivot columns of each set of a family, from its pivots padded beyond each rank."""
    return tuple(
        tuple(int(column) for column in set_pivots[:rank])
        for set_pivots, rank in zip(pivots, ranks, strict=True)
    )


####################################################################################################
# basis family stages


def get_disjoint_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    *,
    random_column_orders: int,
    weight_divisor: int = 1,
) -> BasisFamily:
    """Stage one: pack disjoint information sets, and certify them with any inferred parity.

    ``labels`` are the quotient labels of ``basis``, and None stands for labels with no columns.
    The packing may try up to ``random_column_orders`` random column orders (Section 4.4).
    """
    dimension = len(basis)
    set_labels = np.empty((dimension, 0), dtype=np.uint8) if labels is None else labels
    sets = _get_disjoint_sets(basis, set_labels, random_column_orders)
    generators = np.empty((len(sets), dimension, basis.shape[1]), dtype=np.uint8)
    reduced_labels = np.empty((len(sets), dimension, set_labels.shape[1]), dtype=np.uint8)
    pivots = np.full((len(sets), dimension), -1, dtype=np.int64)
    for index, (set_generators, set_reduced_labels, set_pivots) in enumerate(sets):
        generators[index], reduced_labels[index] = set_generators, set_reduced_labels
        pivots[index, : len(set_pivots)] = set_pivots
    ranks = np.array([len(set_pivots) for *_, set_pivots in sets], dtype=np.int64)
    # Disjoint sets share no pivot column, so no coordinate is covered more than once.
    weight_divisor = infer_weight_divisor(basis, weight_divisor)
    return BasisFamily(generators, reduced_labels, pivots, ranks, weight_divisor, 1)


@compiled_on_request
def prune_family(basis_family: BasisFamily, upper_bound: int) -> BasisFamily:
    """Stage two: keep the full sets and the partial sets that can help certify ``upper_bound``."""
    retained = np.flatnonzero(
        get_retained_set_mask(
            basis_family.generators.shape[1],
            basis_family.ranks,
            basis_family.weight_divisor,
            basis_family.max_coverage,
            upper_bound,
        )
    )
    return BasisFamily(
        basis_family.generators[retained],
        basis_family.labels[retained],
        basis_family.pivots[retained],
        basis_family.ranks[retained],
        basis_family.weight_divisor,
        basis_family.max_coverage,
    )


class OverlappingFamilyPolicy(NamedTuple):
    """How stage three builds a replacement basis family (see ``get_overlapping_family``).

    Attributes:
        max_sets: The most sets of a replacement.
        minimum_work: The estimated work below which a basis family is not replaced.
        attempts_per_set: The column orders that the construction tries for each set that it may
            add.
        binomials: ``get_capped_binomials(dimension)``, for the dimension of the basis.
        rng: The generator that breaks ties between equally used columns.
    """

    max_sets: int
    minimum_work: int
    attempts_per_set: int
    binomials: npt.NDArray[np.int64]
    rng: np.random.Generator


@compiled_on_request
def get_overlapping_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    basis_family: BasisFamily,
    upper_bound: int,
    policy: OverlappingFamilyPolicy,
) -> BasisFamily:
    """Stage three: build a replacement basis family of overlapping full sets that does less work.

    ``basis_family`` holds the sets of the independent binary ``basis``, whose quotient labels
    ``labels`` have no columns if it has none.  The replacement holds only full sets, with labels
    that follow their row operations, and is certified with its maximum coverage.  It has no sets if
    the construction does not run or finds no family that does less work (Section 4.4).
    """
    max_sets, _, attempts_per_set, binomials, rng = policy
    dimension = basis.shape[0]
    weight_divisor = basis_family.weight_divisor

    # Keep the family if a check rules a replacement out.
    check, best_work, optimistic_work = _check_overlapping_family(
        basis, basis_family, upper_bound, policy
    )
    if check != _OverlappingCheck.NONE:
        return _get_no_replacement(basis_family, basis, labels)

    # Find the nonzero columns, which the sets of the replacement can use, and the full sets of the
    # family, which the replacement starts from.
    columns = get_nonzero_columns(basis)
    num_columns = len(columns)
    full_indices = np.flatnonzero(basis_family.ranks == dimension)
    num_full = len(full_indices)

    # Track the coverage of each column and the distinct sets found so far, in arrays that grow on
    # demand, since the limit may be large.
    capacity = min(max_sets, 2 * num_full + 1)
    generators = np.empty((capacity, dimension, basis.shape[1]), dtype=np.uint8)
    set_labels = np.empty((capacity, dimension, labels.shape[1]), dtype=np.uint8)
    pivots = np.empty((capacity, dimension), dtype=np.int64)
    sorted_pivots = np.empty((capacity, dimension), dtype=np.int64)
    column_coverage = np.zeros(basis.shape[1], dtype=np.int64)
    for index in range(num_full):
        generators[index] = basis_family.generators[full_indices[index]]
        set_labels[index] = basis_family.labels[full_indices[index]]
        pivots[index] = basis_family.pivots[full_indices[index]]
        sorted_pivots[index] = np.sort(pivots[index])
        column_coverage[pivots[index]] += 1

    num_sets = num_full
    best_num_sets = 0
    best_coverage = 1
    packed_basis, packed_labels = pack_bits(basis), pack_bits(labels)
    # Saturate the number of attempts, which compiled code computes with 64-bit integers.  At least
    # one set is open, since the family has fewer full sets than the limit.
    num_open = max_sets - num_full
    num_attempts = (
        attempts_per_set * num_open
        if attempts_per_set <= _tuning.MAX_VALUE // num_open
        else _tuning.MAX_VALUE
    )
    for _ in range(num_attempts):
        # Order the columns by load, and then randomly, with two stable sorts, since numba has no
        # lexicographic sort.  Reduce the basis and its labels in that order.
        by_random = np.argsort(rng.random(num_columns), kind="mergesort")
        by_coverage = np.argsort(column_coverage[columns][by_random], kind="mergesort")
        reduced, reduced_labels, new_pivots = row_reduce_packed(
            packed_basis, packed_labels, columns[by_random[by_coverage]]
        )

        # Skip duplicate sets, whose row reductions still count against the attempt limit.
        new_sorted_pivots = np.sort(new_pivots)
        duplicate = False
        for index in range(num_sets):
            if np.all(sorted_pivots[index] == new_sorted_pivots):
                duplicate = True
                break
        if duplicate:
            continue
        if num_sets == capacity:
            capacity = min(2 * capacity, max_sets)
            generators = _copy_with_capacity(generators, capacity)
            set_labels = _copy_with_capacity(set_labels, capacity)
            pivots = _copy_with_capacity(pivots, capacity)
            sorted_pivots = _copy_with_capacity(sorted_pivots, capacity)
        generators[num_sets] = unpack_bits(reduced, basis.shape[1])
        set_labels[num_sets] = unpack_bits(reduced_labels, labels.shape[1])
        pivots[num_sets] = new_pivots
        sorted_pivots[num_sets] = new_sorted_pivots
        column_coverage[new_pivots] += 1
        num_sets += 1
        coverage = int(column_coverage.max())

        # Keep improvements, and stop at the set limit or once the optimistic work bound is met.
        full_ranks = np.full(num_sets, dimension, dtype=np.int64)
        work = estimate_certificate_work(
            dimension, full_ranks, weight_divisor, coverage, upper_bound, binomials
        )
        if work < best_work:
            best_num_sets, best_coverage, best_work = num_sets, coverage, work
        if num_sets == max_sets or best_work <= optimistic_work:
            break
    # Copy the sets, so that the result does not keep the larger arrays that they were built in.
    return BasisFamily(
        generators[:best_num_sets].copy(),
        set_labels[:best_num_sets].copy(),
        pivots[:best_num_sets].copy(),
        np.full(best_num_sets, dimension, dtype=np.int64),
        weight_divisor,
        best_coverage,
    )


def get_overlapping_family_policy(tuning: SearchTuning, dimension: int) -> OverlappingFamilyPolicy:
    """The policy of stage three under ``tuning``, for a basis of ``dimension`` rows."""
    return build_overlapping_family_policy(
        tuning.max_information_sets,
        tuning.overlapping_minimum_work,
        tuning.overlapping_attempts_per_set,
        dimension,
    )


def build_overlapping_family_policy(
    max_sets: int, minimum_work: int, attempts_per_set: int, dimension: int
) -> OverlappingFamilyPolicy:
    """The policy of stage three with given settings, for a basis of ``dimension`` rows.

    Its generator is new, with a fixed seed, so that every search chooses the same replacement for
    the same basis family and settings (Section 4.4).
    """
    return OverlappingFamilyPolicy(
        max_sets,
        minimum_work,
        attempts_per_set,
        get_capped_binomials(dimension),
        np.random.default_rng(0),
    )


def get_partitioned_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    basis_family: BasisFamily,
    upper_bound: int,
    tuning: SearchTuning,
    *,
    get_competing_work: Callable[[], int] | None = None,
) -> BasisFamily:
    """Stage four: a basis family of bounded coverage, if it does less work.

    For each coverage that it tries, from two and none above ``tuning.partition_max_coverage``,
    matroid partitioning finds the most information sets, up to ``tuning.max_information_sets``,
    within that coverage.  If ``tuning.partition_max_coverage`` is one, it finds instead the maximum
    disjoint family of ``get_maximum_disjoint_family`` (Section 4.4), and prunes it as stage two
    does.  It skips that family if ``basis_family`` is disjoint and has the pruned disjoint rank
    bound, since that family could not replace it.  ``basis``, ``labels``, and ``basis_family`` are
    as in ``get_overlapping_family``.  ``get_competing_work``, if given, returns the work of a
    family, such as an automorphism family, to which the caller would switch instead, discarding
    this stage's family (Section 3.7).  It is called only if partitioning may try some coverage and
    passes its gate, and some family of this stage could do less work than ``basis_family``
    (Section 4.4).  Partitioning then runs only if the competing work exceeds the least that a
    family of this stage could do.

    Returns the family that certifies ``upper_bound`` with the least estimated work, if partitioning
    runs and that is less than the work of ``basis_family``, and otherwise a family with no sets.
    """
    best_work = basis_family.certificate.work(upper_bound)
    best = _get_no_replacement(basis_family, basis, labels)

    # Skip partitioning if it would try no coverage, if it would cost too much for this search, or
    # if no family could do less work.
    plan = _get_partition_plan(basis, tuning)
    if not plan.coverages or best_work < tuning.partition_work_ratio * plan.unit:
        return best
    weight_divisor = basis_family.weight_divisor
    optimistic_work = plan.estimate_optimistic_work(weight_divisor, upper_bound)
    if best_work <= optimistic_work:
        return best

    # Skip partitioning if the search would switch to a family that does no more work than any
    # family of this stage could.
    if get_competing_work is not None and get_competing_work() <= optimistic_work:
        return best

    # Build a family for each coverage that could improve on the best so far, and keep the one that
    # does the least work.  A disjoint family can end with a partial set, which ``prune_family``
    # drops if it cannot help.
    for coverage in plan.coverages:
        if coverage == 1:
            # Skip coverage one if the given family is disjoint and has the pruned disjoint rank
            # bound, which leaves the result unchanged (Section 4.4).
            if _has_pruned_disjoint_rank_bound(basis_family, plan.num_columns, upper_bound):
                continue
            maximum_disjoint = _get_family_of_sets(
                basis, labels, _get_maximum_disjoint_sets(basis), weight_divisor
            )
            family = prune_family(maximum_disjoint, upper_bound)
        else:
            # Skip a coverage at which no family could do less work than the best so far, which
            # leaves the result unchanged (Section 4.4).
            bound = plan.estimate_optimistic_coverage_work(coverage, weight_divisor, upper_bound)
            if bound >= best_work:
                continue
            column_sets = matroid_partition.get_information_sets_with_coverage(
                basis, coverage, plan.max_sets
            )
            family = _get_family_of_sets(basis, labels, column_sets, weight_divisor)

        work = family.certificate.work(upper_bound)
        if work < best_work:
            best, best_work = family, work
    return best


def get_maximum_disjoint_family(
    basis: npt.NDArray[np.uint8], basis_family: BasisFamily
) -> BasisFamily:
    """A disjoint family with the most information sets, and then the largest partial set.

    ``basis_family`` is a disjoint family of the independent binary ``basis``, such as the greedy
    packing of ``get_disjoint_family``.  Matroid partitioning with coverage one finds the new
    family, which has no labels and at most as many information sets as a packing keeps
    (Section 4.4).

    Returns:
        The new family, if its tuple of ranks is lexicographically larger than that of
        ``basis_family``, and otherwise a family with no sets.
    """
    dimension = len(basis)
    num_columns = len(get_nonzero_columns(basis))
    no_labels = np.empty((dimension, 0), dtype=np.uint8)
    # No disjoint family improves on one whose pivots use every nonzero column, or that holds as
    # many information sets as a packing keeps.
    if int(basis_family.ranks.sum()) == num_columns:
        return _get_no_replacement(basis_family, basis, no_labels)
    full_set_limit = _get_full_set_limit(dimension)
    if int(np.count_nonzero(basis_family.ranks == dimension)) >= full_set_limit:
        return _get_no_replacement(basis_family, basis, no_labels)

    column_sets = _get_maximum_disjoint_sets(basis)
    ranks = tuple(len(columns) for columns in column_sets)
    if ranks <= tuple(basis_family.ranks.tolist()):
        return _get_no_replacement(basis_family, basis, no_labels)
    return _get_family_of_sets(basis, no_labels, column_sets, basis_family.weight_divisor)


def infer_weight_divisor(basis: npt.NDArray[np.uint8], weight_divisor: int = 1) -> int:
    """Combine a caller's proven Hamming-weight divisor with ambient even parity, if present."""
    # XOR preserves even parity, so an all-even basis spans only even Hamming weights.
    if np.all(np.count_nonzero(basis, axis=1) % 2 == 0):
        return math.lcm(weight_divisor, 2)
    return weight_divisor


@compiled_on_request
def _copy_with_capacity(array: npt.NDArray[_Scalar], capacity: int) -> npt.NDArray[_Scalar]:
    """A copy of an array with room for ``capacity`` entries along its first axis."""
    grown = np.empty((capacity,) + array.shape[1:], dtype=array.dtype)
    grown[: len(array)] = array
    return grown


@compiled_on_request
def _get_no_replacement(
    basis_family: BasisFamily, basis: npt.NDArray[np.uint8], labels: npt.NDArray[np.uint8]
) -> BasisFamily:
    """A family of no sets, for when no better family is found.

    Stages three and four and ``get_maximum_disjoint_family`` return it, and the distance search,
    the counts, and the routes of coset searches then keep the families that they have
    (Section 4.4).
    """
    dimension = basis.shape[0]
    return BasisFamily(
        np.empty((0, dimension, basis.shape[1]), dtype=np.uint8),
        np.empty((0, dimension, labels.shape[1]), dtype=np.uint8),
        np.empty((0, dimension), dtype=np.int64),
        np.empty(0, dtype=np.int64),
        basis_family.weight_divisor,
        1,
    )


def _get_disjoint_sets(
    basis: npt.NDArray[np.uint8], labels: npt.NDArray[np.uint8], random_column_orders: int
) -> list[_DisjointSet]:
    """Construct a large deterministic packing of disjoint information sets (Section 4.4).

    The natural column order comes first, and then up to ``random_column_orders`` random orders,
    until one packs ``min(N // r, 2**r - 1)`` full sets, which no order exceeds, for ``r`` rows and
    ``N`` nonzero columns.
    """
    # Only columns that are nonzero in the basis can be pivots.  Pack them in their natural order.
    nonzero_columns = get_nonzero_columns(basis)
    best = _get_disjoint_sets_for_columns(
        basis,
        labels,
        nonzero_columns,
    )

    # Return if the natural order packs as many full sets as any order could.
    max_full_sets = min(len(nonzero_columns) // len(basis), _get_full_set_limit(len(basis)))
    if _count_full_sets(best, len(basis)) == max_full_sets:
        return best

    # Otherwise, try random column orders, from a fixed seed for reproducible results.
    rng = np.random.default_rng(0)
    for _ in range(random_column_orders):
        # Keep the packing with the lexicographically larger tuple of ranks.
        candidate = _get_disjoint_sets_for_columns(
            basis,
            labels,
            rng.permutation(nonzero_columns),
        )
        candidate_ranks = tuple(len(pivots) for *_, pivots in candidate)
        best_ranks = tuple(len(pivots) for *_, pivots in best)
        if candidate_ranks > best_ranks:
            best = candidate

        # Stop early if the packing reaches the bound.
        if _count_full_sets(best, len(basis)) == max_full_sets:
            break
    return best


def _count_full_sets(information_sets: list[_DisjointSet], dimension: int) -> int:
    """The number of full sets, of ``dimension`` pivots, in a packing of disjoint sets."""
    return sum(len(pivots) == dimension for *_, pivots in information_sets)


def _get_full_set_limit(dimension: int) -> int:
    """The most full sets that a packing of ``dimension`` rows keeps (Section 4.4)."""
    return (1 << dimension) - 1


def _get_disjoint_sets_for_columns(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    columns: npt.NDArray[np.int_],
) -> list[_DisjointSet]:
    """Construct disjoint information sets, with pivots from the nonzero ``columns`` in order.

    It stops after ``_get_full_set_limit(len(basis))`` full sets.
    """
    # Start with every given column available as a pivot, and with the basis as it is.
    dimension = len(basis)
    full_set_limit = _get_full_set_limit(dimension)
    active_columns = columns
    working = basis
    working_labels = labels
    information_sets: list[_DisjointSet] = []

    while len(active_columns):
        # Row reduce the basis and its labels on the available columns, taking pivots in order.
        working, reduced_labels, pivots = row_reduce_binary(
            working, working_labels, columns=active_columns
        )
        assert reduced_labels is not None  # labels in give labels out
        working_labels = reduced_labels
        rank = len(pivots)
        assert rank  # The available columns are nonzero, so they give at least one pivot.
        information_sets.append((working, working_labels, pivots))

        # A partial set is the last, since the available columns have rank ``rank``, too little
        # for a later full set.  A packing also stops at its limit of full sets.
        if rank < dimension or len(information_sets) == full_set_limit:
            break

        # Make the pivot columns unavailable, so that later sets are disjoint from this one.  Row
        # operations keep the given nonzero columns nonzero, so each pass has a pivot.
        active_columns = active_columns[~np.isin(active_columns, pivots)]

    return information_sets


class _OverlappingCheck(enum.Enum):
    """The first check of ``_check_overlapping_family`` that rules a replacement out, or ``NONE``.

    The checks find nothing to extend (no full sets to start from, already the most full sets
    allowed, or full sets that already use every nonzero column), an estimated work below the
    minimum, or that no family could do less work.
    """

    NONE = 0
    NOTHING_TO_EXTEND = 1
    BELOW_MINIMUM_WORK = 2
    CANNOT_DO_LESS_WORK = 3


@compiled_on_request
def _check_overlapping_family(
    basis: npt.NDArray[np.uint8],
    basis_family: BasisFamily,
    upper_bound: int,
    policy: OverlappingFamilyPolicy,
) -> tuple[_OverlappingCheck, int, int]:
    """Find the first check that rules out a replacement in ``get_overlapping_family``, if any.

    The arguments are those of ``get_overlapping_family``.

    Returns:
        The check, the estimated work of ``basis_family`` to certify ``upper_bound``, and the least
        work that any family of at most ``policy.max_sets`` full sets could do.  A work is -1 if an
        earlier check rules a replacement out.
    """
    max_sets, minimum_work, _, binomials, _ = policy
    dimension = basis.shape[0]
    num_columns = len(get_nonzero_columns(basis))
    ranks = basis_family.ranks
    num_full = len(np.flatnonzero(ranks == dimension))

    # Replace only a family with full sets, which the replacement starts from.  Keep a family whose
    # full sets use every nonzero column, or that already has the most full sets allowed.
    if num_full == 0 or num_full >= max_sets or num_full * dimension == num_columns:
        return _OverlappingCheck.NOTHING_TO_EXTEND, -1, -1

    # Rule a replacement out when the existing search is already cheap.
    weight_divisor = basis_family.weight_divisor
    work = estimate_certificate_work(
        dimension, ranks, weight_divisor, basis_family.max_coverage, upper_bound, binomials
    )
    if work < minimum_work:
        return _OverlappingCheck.BELOW_MINIMUM_WORK, work, -1

    # Rule a replacement out if no family of at most max_sets full sets could do less work.
    optimistic_work = _estimate_optimistic_work(
        dimension, num_columns, weight_divisor, upper_bound, max_sets, binomials
    )
    if optimistic_work >= work:
        return _OverlappingCheck.CANNOT_DO_LESS_WORK, work, optimistic_work
    return _OverlappingCheck.NONE, work, optimistic_work


class _PartitionPlan(NamedTuple):
    """The coverages that ``get_partitioned_family`` may try for a basis, and the unit of its gate.

    Attributes:
        dimension: The rows ``r`` of the basis.
        num_columns: The nonzero columns ``N`` of the basis.
        coverages: The coverages that partitioning may try, which may be none.
        max_sets: The most information sets of a family of coverage two or more.
        unit: The sum of ``b**2 * r**2 * N`` over the coverages ``b``, which the gate multiplies
            by ``partition_work_ratio`` (Section 4.4).
    """

    dimension: int
    num_columns: int
    coverages: range
    max_sets: int
    unit: int

    def estimate_optimistic_work(self, weight_divisor: int, upper_bound: int) -> int:
        """A lower bound on the estimated work of any family that partitioning can build."""
        binomials = get_capped_binomials(self.dimension)
        if 1 in self.coverages:
            return _estimate_optimistic_disjoint_work(
                self.dimension, self.num_columns, weight_divisor, upper_bound, binomials
            )
        return _estimate_optimistic_work(
            self.dimension, self.num_columns, weight_divisor, upper_bound, self.max_sets, binomials
        )

    def estimate_optimistic_coverage_work(
        self, coverage: int, weight_divisor: int, upper_bound: int
    ) -> int:
        """A lower bound on the estimated work of a family that partitioning finds at a coverage.

        The coverage is two or more, and the bound is over the most sets that fit within it by
        counting their pivots (Section 4.4).
        """
        most_sets = min(self.max_sets, coverage * self.num_columns // self.dimension)
        return _estimate_optimistic_work(
            self.dimension,
            self.num_columns,
            weight_divisor,
            upper_bound,
            most_sets,
            get_capped_binomials(self.dimension),
        )


def _get_partition_plan(basis: npt.NDArray[np.uint8], tuning: SearchTuning) -> _PartitionPlan:
    """The plan of ``get_partitioned_family`` for a basis under a tuning (Section 4.4)."""
    # Choose the coverages to try and the unit of the gate.  Above a limit of one on coverage,
    # capping the sets at the most that fit within the limit, by counting their pivots, caps the
    # coverages.  Those limits leave coverage one to the disjoint packing of stage one, so that
    # searches at the default limit build no maximum disjoint family.
    dimension = len(basis)
    num_columns = len(get_nonzero_columns(basis))
    max_sets = tuning.max_information_sets
    coverage_limit = tuning.partition_max_coverage
    if coverage_limit == 1:
        unit = dimension**2 * num_columns
        return _PartitionPlan(dimension, num_columns, range(1, 2), max_sets, unit)
    max_sets = min(max_sets, coverage_limit * num_columns // dimension)

    # The largest coverage to try is the least that ``max_sets`` full sets need
    # (``_get_least_coverage``).  The sum of b**2 from 2 to B is B (B + 1) (2B + 1) / 6 - 1.
    max_coverage = _get_least_coverage(max_sets, dimension, num_columns)
    coverages = range(2, max_coverage + 1)
    if not coverages:
        return _PartitionPlan(dimension, num_columns, coverages, max_sets, 0)
    sum_of_squares = max_coverage * (max_coverage + 1) * (2 * max_coverage + 1) // 6 - 1
    unit = sum_of_squares * dimension**2 * num_columns
    return _PartitionPlan(dimension, num_columns, coverages, max_sets, unit)


def _get_maximum_disjoint_sets(basis: npt.NDArray[np.uint8]) -> list[list[int]]:
    """The column sets of the maximum disjoint family of ``get_maximum_disjoint_family``."""
    # A disjoint family has at most num_columns // dimension information sets and a partial set, but
    # a family at the limit of a packing holds as many information sets as a packing keeps, and no
    # partial set.
    dimension = len(basis)
    num_columns = len(get_nonzero_columns(basis))
    max_sets = min(num_columns // dimension + 1, _get_full_set_limit(dimension))
    return matroid_partition.get_information_sets_with_coverage(
        basis, 1, max_sets, keep_partial=True
    )


def _has_pruned_disjoint_rank_bound(
    basis_family: BasisFamily, num_columns: int, upper_bound: int
) -> bool:
    """Whether a family is disjoint and has the disjoint rank bound, pruned for ``upper_bound``.

    The disjoint rank bound is ``f = min(N // r, 2**r - 1)`` information sets, for ``r`` rows and
    the ``N = num_columns`` nonzero columns of the basis, and below the limit of a packing a partial
    set of rank ``N - f*r``, if that is positive.  It is pruned as ``prune_family`` prunes a family.
    If the family is disjoint and has these ranks, so does every pruned maximum disjoint family
    (Section 4.4).
    """
    # Bound the ranks of a maximum disjoint family: the most information sets that fit, up to the
    # limit of a packing, and below that limit a partial set of rank N - f*r, the number of other
    # columns, if positive.
    dimension = basis_family.generators.shape[1]
    full_set_limit = _get_full_set_limit(dimension)
    num_full = min(num_columns // dimension, full_set_limit)
    remainder = num_columns - num_full * dimension if num_full < full_set_limit else 0
    ranks = np.array([dimension] * num_full + ([remainder] if remainder else []), dtype=np.int64)

    # Prune these ranks as prune_family does, and compare them with those of the family, if it is
    # disjoint.
    weight_divisor = basis_family.weight_divisor
    retained = get_retained_set_mask(dimension, ranks, weight_divisor, 1, upper_bound)
    return (
        basis_family.max_coverage == 1 and basis_family.ranks.tolist() == ranks[retained].tolist()
    )


def _get_family_of_sets(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    column_sets: list[list[int]],
    weight_divisor: int,
) -> BasisFamily:
    """The basis family of one or more sets of independent columns of a basis.

    The basis's labels are reduced along with it, and the family is certified with the coverage of
    its pivots.

    Raises:
        RuntimeError: If a set of columns is not independent.
    """
    # Reduce the basis on each set, transforming its labels along with it.
    dimension = len(basis)
    num_sets = len(column_sets)
    generators = np.empty((num_sets, dimension, basis.shape[1]), dtype=np.uint8)
    set_labels = np.empty((num_sets, dimension, labels.shape[1]), dtype=np.uint8)
    pivots = np.full((num_sets, dimension), -1, dtype=np.int64)
    ranks = np.empty(num_sets, dtype=np.int64)
    for index, columns in enumerate(column_sets):
        reduced, reduced_labels, found = row_reduce_binary(
            basis, labels, columns=np.array(columns, dtype=int)
        )
        assert reduced_labels is not None  # labels in give labels out

        # A dependent set means that the caller, such as matroid partitioning, broke its contract.
        if len(found) != len(columns):
            raise RuntimeError(f"Column set {index} of a basis family is not independent")
        generators[index], set_labels[index] = reduced, reduced_labels
        pivots[index, : len(found)] = found
        ranks[index] = len(found)
    return BasisFamily(
        generators,
        set_labels,
        pivots,
        ranks,
        weight_divisor,
        int(np.bincount(pivots[pivots >= 0]).max()),
    )


####################################################################################################
# certificate estimates of the stages


@always_compiled
def get_retained_set_mask(
    dimension: int,
    ranks: npt.NDArray[np.int64],
    weight_divisor: int,
    max_coverage: int,
    upper_bound: int,
) -> npt.NDArray[np.bool_]:
    """Mark the sets that stage two retains: full sets, and partial sets that can help certify."""
    num_full = 0
    for rank in ranks:
        num_full += rank == dimension
    retained = np.ones(len(ranks), dtype=np.bool_)
    if num_full == 0:
        return retained

    # The completed weight from which the full sets alone certify the upper bound.
    certifying_weight = dimension + 1
    for weight in range(dimension + 1):
        occupancy = num_full * (weight + 1)
        if lower_bound_from_occupancy(occupancy, weight_divisor, max_coverage) >= upper_bound:
            certifying_weight = weight
            break

    # A dropped partial set adds zero lower-bound occupancy until full sets certify the upper bound.
    for index in range(len(ranks)):
        retained[index] = ranks[index] == dimension or dimension - ranks[index] < certifying_weight
    return retained


@always_compiled
def _estimate_optimistic_work(
    dimension: int,
    num_columns: int,
    weight_divisor: int,
    upper_bound: int,
    max_sets: int,
    binomials: npt.NDArray[np.int64],
) -> int:
    """Return a lower bound on the estimated work of a family of at most ``max_sets`` full sets.

    The family is of a basis of ``dimension`` rows and ``num_columns`` nonzero columns, and
    ``binomials`` are ``get_capped_binomials(dimension)``.  No family of ``m`` full sets has a
    maximum coverage below ``_get_least_coverage(m, dimension, num_columns)``, which the bound
    assumes for each ``m`` (Section 4.4).
    """
    optimistic_work = WORK_CAP
    for num_sets in range(1, max_sets + 1):
        coverage = _get_least_coverage(num_sets, dimension, num_columns)
        full_ranks = np.full(num_sets, dimension, dtype=np.int64)
        work = estimate_certificate_work(
            dimension, full_ranks, weight_divisor, coverage, upper_bound, binomials
        )
        optimistic_work = min(optimistic_work, work)
    return optimistic_work


def _estimate_optimistic_disjoint_work(
    dimension: int,
    num_columns: int,
    weight_divisor: int,
    upper_bound: int,
    binomials: npt.NDArray[np.int64],
) -> int:
    """Return a lower bound on the estimated work of a disjoint family that partitioning can build.

    Such a family, like a packing, holds at most ``k`` sets, counting a partial set, where
    ``k = min(ceil(num_columns / dimension), 2**dimension - 1)``.  The arguments are as for
    ``_estimate_optimistic_work``.  The bound is the least work of 1 to ``k`` full sets of coverage
    one (Section 4.4).
    """
    max_sets = min((num_columns + dimension - 1) // dimension, _get_full_set_limit(dimension))
    return min(
        estimate_certificate_work(
            dimension,
            np.full(num_sets, dimension, dtype=np.int64),
            weight_divisor,
            1,
            upper_bound,
            binomials,
        )
        for num_sets in range(1, max_sets + 1)
    )


@always_compiled
def _get_least_coverage(num_sets: int, dimension: int, num_columns: int) -> int:
    """A lower bound on the maximum coverage of ``num_sets`` full information sets of a basis.

    The basis has ``dimension`` rows and ``num_columns`` nonzero columns, so the sets' pivots put
    some column in at least ``ceil(num_sets * dimension / num_columns)`` of them (Section 4.4).  A
    light word can force more, since every information set meets its support.
    """
    return (num_sets * dimension + num_columns - 1) // num_columns
