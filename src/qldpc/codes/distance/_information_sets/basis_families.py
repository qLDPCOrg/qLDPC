# SPDX-License-Identifier: Apache-2.0

"""Basis families of information sets, shared by exact BZ and coset searches and by counts.

The Brouwer-Zimmermann distance search (``_brouwer_zimmermann/distance.py``) and fixed-weight
logical counting (``_counting/scan.py``) enumerate coefficient layers over the same basis families
(see ``brouwer_zimmermann.md`` for the terminology).  Layer ``w`` of an information set holds the
words that add ``w`` of its rows to an offset.  The offset is zero in the distance and counting
scans, which search a space.  The coset-leader oracle (``CosetLeaderSearch``) answers its queries
with coset searches, whose offset is a fixed word of the searched coset (see ``_search_coset`` in
``_brouwer_zimmermann/coset.py``).  This module owns how those basis families are built, pruned,
and replaced by overlapping ones.  The coset searches use the disjoint information sets, and for
costly queries the families of :func:`get_maximum_disjoint_family` and
:func:`get_overlapping_family` (see ``_brouwer_zimmermann/coset_routes.py``).  The certificate that
bounds the Hamming weight of the eligible words that a basis family has not yet visited, and the
plans of the steps of searches, are in ``certificates.py``.  The step from one support of a layer
to the next is in ``support_stepping.py``.

A :class:`BasisFamily` stacks its information sets into arrays, so that one set of functions serves
every caller, in the subset of Python that numba compiles, and some of them are compiled (see
``qldpc/_compilation.py``).  Its :class:`Certificate` gives the bound and the work estimate.

Basis family construction runs in stages, so that a caller can take its own exits between them:

1. :func:`get_disjoint_family` packs disjoint information sets and infers ambient even parity.
2. :func:`prune_family` drops partial sets that cannot help certify a target.
3. :func:`get_overlapping_family` may build a cheaper basis family of overlapping full sets.
4. :func:`get_partitioned_family` may build a cheaper one still, by matroid partitioning, for
   costly searches.

Each stage aims at a target ``upper_bound``.  Distance passes its current witness weight ``U`` and
stops once the certificate reaches ``U``.  Counting at weight ``w`` must also visit every word of
weight exactly ``w``, so it passes ``w + 1`` and stops only once the certificate exceeds ``w``.
Section 4.4 of ``brouwer_zimmermann.md`` describes the heuristics, and Sections 2.5 and 3.1 derive
the certificate.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits, row_reduce_binary, row_reduce_packed, unpack_bits
from qldpc._compilation import always_compiled, compiled_on_request

from .. import _tuning
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
    ``pivots`` has shape ``(num_sets, dimension)``, and the pivot columns of a set with rank
    deficiency are padded with -1.  ``ranks``, ``weight_divisor`` and ``max_coverage`` are the
    parameters of its :attr:`certificate`.
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

        A search of overlapping sets whose estimated work reaches ``lp_minimum_work`` also bounds
        unvisited words by linear programming, which can stop it sooner, with dual weights whose
        denominators are at most ``lp_max_denominator``.
        """
        certificate = self.certificate
        if self.max_coverage == 1 or certificate.work(upper_bound) < lp_minimum_work:
            return certificate
        pivot_blocks = tuple(
            tuple(int(column) for column in pivots[:rank])
            for pivots, rank in zip(self.pivots, self.ranks, strict=True)
        )
        return dataclasses.replace(
            certificate, pivot_blocks=pivot_blocks, lp_max_denominator=lp_max_denominator
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

    ``labels`` are the quotient labels of ``basis``, and None stands for labels with no columns.  If
    the natural column order packs too few full sets, up to ``random_column_orders`` random orders
    are tried (see ``_get_disjoint_sets``).
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
    """Stage two: retain partial sets that can strengthen certification of ``upper_bound``."""
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

    Searches take it from :func:`get_overlapping_family_policy`, or from
    :func:`build_overlapping_family_policy` for settings that their tuning does not hold.

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

    ``basis_family`` holds the full and partial sets of one independent binary basis, whose quotient
    labels ``labels`` have no columns if it has none.  A replacement holds only full sets, with
    their labels transformed alongside their rows, and is certified with its maximum coverage of a
    Hamming coordinate.  ``max_sets`` of ``policy`` bounds its size and the effort spent building
    it: more sets can avoid costly coefficient layers in hard searches, while fewer reduce set-up
    work.  No replacement is built if the estimated work of ``basis_family`` is below
    ``minimum_work``.  The construction tries ``attempts_per_set`` column orders for each set that
    it may add.

    Returns the replacement, which has no sets if no replacement does less work.
    """
    max_sets, minimum_work, attempts_per_set, binomials, rng = policy
    dimension = basis.shape[0]
    columns = np.flatnonzero(basis.sum(axis=0))
    num_columns = len(columns)
    ranks = basis_family.ranks
    full_indices = np.flatnonzero(ranks == dimension)
    num_full = len(full_indices)

    # Start from full blocks; skip complete packings and basis families already at the size limit.
    if num_full == 0 or num_full >= max_sets or num_full * dimension == num_columns:
        return _get_no_replacement(basis_family, basis, labels)

    # Skip basis family construction when the existing search is already cheap.
    weight_divisor = basis_family.weight_divisor
    best_work = estimate_certificate_work(
        dimension, ranks, weight_divisor, basis_family.max_coverage, upper_bound, binomials
    )
    if best_work < minimum_work:
        return _get_no_replacement(basis_family, basis, labels)

    # Skip basis family construction if no family of at most max_sets full sets could do less work.
    optimistic_work = _estimate_optimistic_work(
        dimension, num_columns, weight_divisor, upper_bound, max_sets, binomials
    )
    if optimistic_work >= best_work:
        return _get_no_replacement(basis_family, basis, labels)

    # Track coverage and distinct sets; fixed-seed tie-breaking keeps choices reproducible.  The
    # arrays hold the sets found so far and grow on demand, since the limit may be large.
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
        # lexicographic sort.  Transform the labels along with each candidate.
        by_random = np.argsort(rng.random(num_columns), kind="mergesort")
        by_coverage = np.argsort(column_coverage[columns][by_random], kind="mergesort")
        reduced, reduced_labels, new_pivots = row_reduce_packed(
            packed_basis, packed_labels, columns[by_random[by_coverage]]
        )

        # Do not add duplicate sets; their row reductions still count against the attempt limit.
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

        # Keep improvements; stop at the set limit or when the optimistic work bound is attained.
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
    """The policy of stage three for a search of a basis of a given dimension, under a tuning.

    It holds the most sets, the minimum work, and the attempts per set of ``tuning`` (see
    ``build_overlapping_family_policy``).
    """
    return build_overlapping_family_policy(
        tuning.max_information_sets,
        tuning.overlapping_minimum_work,
        tuning.overlapping_attempts_per_set,
        dimension,
    )


def build_overlapping_family_policy(
    max_sets: int, minimum_work: int, attempts_per_set: int, dimension: int
) -> OverlappingFamilyPolicy:
    """The policy of stage three with given settings, for a search of a basis of a given dimension.

    It holds the most sets, the minimum work, and the attempts per set, the capped binomials of the
    dimension, and a new generator with a fixed seed.  So every search, of the distance, of the
    counts, or of cosets (see ``_brouwer_zimmermann/coset_routes.py``), chooses the same replacement
    for the same basis family and settings.
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
) -> BasisFamily:
    """Stage four: a basis family of information sets with bounded coverage, if it does less work.

    ``basis_family`` is the family that a search would otherwise use for the independent binary
    ``basis``, whose quotient labels ``labels`` have no columns if it has none.  For each maximum
    coverage from two to ``tuning.max_partition_coverage``, matroid partitioning finds the most full
    information sets, up to ``tuning.max_information_sets``, that contain no column more often than
    that coverage (see ``matroid_partition``).  This follows Section 5.1 of Lisoněk and Trummer
    (2016), which shows that such families can do less work than those of the Brouwer-Zimmermann
    algorithm.

    Returns the family that certifies ``upper_bound`` with the least estimated work, if that is less
    than the work of ``basis_family``, and otherwise a family with no sets.  No family is built if
    the estimated work of ``basis_family`` is below ``tuning.partition_work_ratio`` times the
    estimated work of partitioning, ``tuning.partition_cost_factor * b * k**2 * n`` for each
    coverage ``b``, or if no family of at most ``tuning.max_information_sets`` full sets could do
    less work: some column of ``m`` such sets is in at least ``ceil(m * k / n)`` of them, for ``k``
    rows and ``n`` nonzero columns.
    """
    max_sets = tuning.max_information_sets
    max_coverage = tuning.max_partition_coverage
    best_work = basis_family.certificate.work(upper_bound)
    best = _get_no_replacement(basis_family, basis, labels)

    # Skip partitioning if it would cost too much for this search, or if no family of full sets
    # could do less work, which is certain if no sets are allowed.
    dimension = len(basis)
    num_columns = int(np.count_nonzero(np.any(basis, axis=0)))
    total_coverage = sum(range(2, max_coverage + 1))
    partition_work = tuning.partition_cost_factor * total_coverage * dimension**2 * num_columns
    if best_work < tuning.partition_work_ratio * partition_work:
        return best
    optimistic_work = _estimate_optimistic_work(
        dimension,
        num_columns,
        basis_family.weight_divisor,
        upper_bound,
        max_sets,
        get_capped_binomials(dimension),
    )
    if best_work <= optimistic_work:
        return best

    # Build a family for each coverage, and keep the one that does the least work.
    for coverage in range(2, max_coverage + 1):
        column_sets = matroid_partition.get_information_sets_with_coverage(
            basis, coverage, max_sets
        )

        # Reduce the basis on each set, transforming its labels along with it.
        num_sets = len(column_sets)
        generators = np.empty((num_sets, dimension, basis.shape[1]), dtype=np.uint8)
        set_labels = np.empty((num_sets, dimension, labels.shape[1]), dtype=np.uint8)
        pivots = np.empty((num_sets, dimension), dtype=np.int64)
        for index, columns in enumerate(column_sets):
            reduced, reduced_labels, found = row_reduce_binary(
                basis, labels, columns=np.array(columns, dtype=int)
            )
            assert reduced_labels is not None and len(found) == dimension
            generators[index], set_labels[index], pivots[index] = reduced, reduced_labels, found
        family = BasisFamily(
            generators,
            set_labels,
            pivots,
            np.full(num_sets, dimension, dtype=np.int64),
            basis_family.weight_divisor,
            int(np.bincount(pivots.ravel()).max()),
        )

        work = family.certificate.work(upper_bound)
        if work < best_work:
            best, best_work = family, work
    return best


def get_maximum_disjoint_family(
    basis: npt.NDArray[np.uint8], basis_family: BasisFamily
) -> BasisFamily:
    """A disjoint family with the most information sets, and then the largest partial set.

    ``basis_family`` is a disjoint family of the independent binary ``basis``, such as that of
    ``get_disjoint_family``, whose greedy packing can miss an information set, or leave a smaller
    partial set than another packing would.  Matroid partitioning with coverage one finds the most
    disjoint information sets, and the largest partial set on the columns that they leave (see
    ``matroid_partition``).  The sets of the family have no labels.  Like a greedy packing, each
    family holds information sets and then at most one partial set.  For such families, if the
    pivots of ``basis_family`` use every nonzero column, no other family has larger ranks, so this
    skips partitioning.

    Returns:
        That family, if its tuple of ranks is lexicographically larger than that of
        ``basis_family``, and otherwise a family with no sets.
    """
    dimension = len(basis)
    num_columns = int(np.count_nonzero(np.any(basis, axis=0)))
    no_labels = np.empty((dimension, 0), dtype=np.uint8)
    if int(basis_family.ranks.sum()) == num_columns:
        return _get_no_replacement(basis_family, basis, no_labels)

    # A disjoint family has at most num_columns // dimension information sets, and a partial set.
    column_sets = matroid_partition.get_information_sets_with_coverage(
        basis, 1, num_columns // dimension + 1, keep_partial=True
    )
    ranks = np.array([len(columns) for columns in column_sets], dtype=np.int64)
    if tuple(ranks.tolist()) <= tuple(basis_family.ranks.tolist()):
        return _get_no_replacement(basis_family, basis, no_labels)

    # Reduce the basis on each set, whose columns are independent.
    num_sets = len(column_sets)
    generators = np.empty((num_sets, dimension, basis.shape[1]), dtype=np.uint8)
    pivots = np.full((num_sets, dimension), -1, dtype=np.int64)
    for index, columns in enumerate(column_sets):
        reduced, _, found = row_reduce_binary(basis, None, columns=np.array(columns, dtype=int))
        generators[index], pivots[index, : len(found)] = reduced, found
    return BasisFamily(
        generators,
        np.empty((num_sets, dimension, 0), dtype=np.uint8),
        pivots,
        ranks,
        basis_family.weight_divisor,
        1,
    )


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

    Stages three and four return it, and so does ``get_maximum_disjoint_family``.  The distance
    search and the counts then keep their basis family, and the routes of coset searches (see
    ``_brouwer_zimmermann/coset_routes.py``) leave the family out of their candidates.
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
    """Construct a large deterministic packing of disjoint information sets.

    The natural column order is tried first, and then up to ``random_column_orders`` random orders,
    until one reaches the most full sets that the columns could allow.
    """
    # Only columns that are nonzero in the basis can be pivots.  Pack them in their natural order.
    nonzero_columns = np.flatnonzero(np.any(basis, axis=0))
    best = _get_disjoint_sets_for_columns(
        basis,
        labels,
        nonzero_columns,
    )

    # A full set has ``len(basis)`` pivots, and only the last set of a packing can have fewer.  Full
    # sets use disjoint nonzero columns, so there are at most this many, and possibly fewer.  Return
    # if the natural order already reaches this bound.
    max_full_sets = len(nonzero_columns) // len(basis)
    if sum(len(pivots) == len(basis) for *_, pivots in best) == max_full_sets:
        return best

    # Otherwise, try a few random column orders.  The fixed seed makes the result reproducible.
    rng = np.random.default_rng(0)
    for _ in range(random_column_orders):
        # Keep the packing with the lexicographically larger tuple of ranks: more full sets, or
        # equally many and a larger last set.
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
        if sum(len(pivots) == len(basis) for *_, pivots in best) == max_full_sets:
            break
    return best


def _get_disjoint_sets_for_columns(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    columns: npt.NDArray[np.int_],
) -> list[_DisjointSet]:
    """Construct disjoint information sets, with pivots from the nonzero ``columns`` in order."""
    # Start with every given column available as a pivot, and with the basis as it is.
    dimension = len(basis)
    active_columns = columns
    working = basis
    working_labels = labels
    information_sets: list[_DisjointSet] = []

    while len(active_columns):
        # Row-reduce on the available columns, taking pivots in their order, so that each pivot
        # column has a single one.  The labels follow the same row operations.
        working, reduced_labels, pivots = row_reduce_binary(
            working, working_labels, columns=active_columns
        )
        assert reduced_labels is not None  # labels in give labels out
        working_labels = reduced_labels
        rank = len(pivots)
        assert rank  # The available columns are nonzero, so they give at least one pivot.
        information_sets.append((working, working_labels, pivots))

        # A set with fewer pivots than rows is partial.  The columns that remain span at most
        # ``rank`` dimensions, so no later set could be full, and this is the last one.
        if rank < dimension:
            break

        # Make the pivot columns unavailable, so that later sets are disjoint from this one.  Also
        # drop columns that are zero in the reduced matrix, so that the next pass has a pivot.
        pivot_set = set(pivots)
        active_columns = np.asarray(
            [col for col in active_columns if col not in pivot_set and np.any(working[:, col])],
            dtype=int,
        )

    return information_sets


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

    The basis has ``dimension`` rows and ``num_columns`` nonzero columns, and ``binomials`` are
    ``get_capped_binomials(dimension)``.  Some column of ``m`` full sets is in at least ``ceil(m *
    dimension / num_columns)`` of them, so no such family is certified with a smaller maximum
    coverage.  Returns ``WORK_CAP`` if ``max_sets`` is zero.
    """
    optimistic_work = WORK_CAP
    for num_sets in range(1, max_sets + 1):
        coverage = (num_sets * dimension + num_columns - 1) // num_columns
        full_ranks = np.full(num_sets, dimension, dtype=np.int64)
        work = estimate_certificate_work(
            dimension, full_ranks, weight_divisor, coverage, upper_bound, binomials
        )
        optimistic_work = min(optimistic_work, work)
    return optimistic_work
