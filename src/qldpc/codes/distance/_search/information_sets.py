# SPDX-License-Identifier: Apache-2.0

"""Basis families of information sets and their certificates, shared by exact BZ and coset searches.

Exact distance search (``brouwer_zimmermann.py``) and fixed-weight logical counting
(``_counting/scan.py``) enumerate coefficient layers over the same basis families (see
``brouwer_zimmermann.md`` for the terminology).  Layer ``w`` of an information set holds the words
that add ``w`` of its rows to an offset.  The offset is zero in the distance and counting scans,
which search a space, and is a fixed word of the searched coset in the coset-leader oracle
(``CosetLeaderSearch``), which searches a coset (see ``_search_coset`` in
``coset_leader_search.py``).  This module owns how those basis families are built, pruned, and
replaced by overlapping ones, and the certificate that bounds the Hamming weight of eligible words
that a basis family has not yet visited.  The coset-leader oracle uses the disjoint information sets
for its own layered search.  All three enumerate a layer with :func:`advance_support`.

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
import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits, row_reduce_binary, row_reduce_packed, unpack_bits
from qldpc._compilation import always_compiled, compiled_on_request

from .. import _tuning
from .._tuning import SearchTuning
from . import matroid_partition

# Caps that keep work estimates within a native integer.  A binomial above _BINOMIAL_CAP counts as
# the cap, and a sum saturates at _WORK_CAP, so an estimate is exact only if it reaches neither cap.
# Searches that reach a cap are infeasible, so their order, against other basis families or
# exhaustive enumeration, does not matter.
_BINOMIAL_CAP = 1 << 50
_WORK_CAP = 1 << 61

# A bound on the weight of unvisited words when no word can remain unvisited.
_NO_WORD_BOUND = np.iinfo(np.int64).max

# One disjoint information set: its rows, its labels, and its pivot columns.
_DisjointSet = tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.int_]]

# The scalar type of an array.
_Scalar = TypeVar("_Scalar", bound=np.generic)


####################################################################################################
# basis families


@dataclass(frozen=True)
class Certificate:
    """Lower bound on the Hamming weight of eligible words that a family has not yet visited.

    ``lower_bound`` holds after every search basis of the family has completed each coefficient
    layer through ``completed_weight``.  ``step_lower_bound`` also holds part way through a layer,
    once the first sets in family order have completed it, in whatever order their words were
    visited.  A set that is interrupted within a layer does not advance either bound.  The bounds do
    not bound the weights of witnesses already found, and they do not transfer to another basis or
    CSS sector.  Each bound is the coverage bound: the occupancy of the sets (see
    ``get_step_lower_bounds``) divided by ``max_coverage``, rounded up, and then rounded up to a
    multiple of ``weight_divisor``.  With ``pivot_blocks``, each bound is the larger of the coverage
    bound and the bound of ``get_lp_lower_bound``, which can be larger when coverage is uneven.

    Attributes:
        dimension: Number of rows of the searched basis.
        ranks: Rank of the pivot block of each search basis, in family order.
        weight_divisor: A proven divisor of the Hamming weight of every eligible word.  It combines
            any divisor proven by the caller, such as two for the symplectic-to-Hamming map, with
            ambient even parity.  It is in the Hamming units of the searched basis, not Pauli units.
        max_coverage: An upper bound on the number of pivot blocks that contain any one coordinate.
            Overestimates are sound; underestimates are not.
        pivot_blocks: The pivot columns of each search basis, in family order, to bound unvisited
            words also by linear programming, or no blocks to use only the coverage bound.
        lp_max_denominator: The largest denominator of the rational dual weights of the linear
            program (see ``get_lp_lower_bound``), which ``pivot_blocks`` needs to be positive.
    """

    dimension: int
    ranks: tuple[int, ...]
    weight_divisor: int = 1
    max_coverage: int = 1
    pivot_blocks: tuple[tuple[int, ...], ...] = ()
    lp_max_denominator: int = 0

    def lower_bound(self, completed_weight: int) -> int:
        """Bound the Hamming weight of unvisited eligible words after the given complete layers."""
        return self.step_lower_bound(completed_weight, len(self.ranks))

    def step_lower_bound(self, weight: int, num_completed: int) -> int:
        """Bound the Hamming weight of unvisited eligible words part way through layer ``weight``.

        The first ``num_completed`` sets, in family order, have completed layer ``weight``, and the
        others have completed layer ``weight - 1``.
        """
        bound = int(self._step_lower_bounds[weight, num_completed])
        if not self.pivot_blocks or self.max_coverage == 1:
            # For disjoint blocks, the linear program separates by block, and its optimum is the
            # coverage bound.
            return bound
        step = (weight, num_completed)
        if step not in self._lp_lower_bounds:
            lp_bound = get_lp_lower_bound(
                self.pivot_blocks, self._get_requirements(*step), self.lp_max_denominator
            )
            if lp_bound < _NO_WORD_BOUND:
                lp_bound += (-lp_bound) % self.weight_divisor
            self._lp_lower_bounds[step] = lp_bound
        return max(bound, self._lp_lower_bounds[step])

    def certifies(self, weight: int, num_completed: int, upper_bound: int) -> bool:
        """Whether ``step_lower_bound(weight, num_completed)`` is at least ``upper_bound``.

        This solves a linear program only if the coverage bound falls short of the upper bound and
        the linear program could reach it: its optimum is at most the value of the feasible solution
        that sets each pivot column to the largest ``requirement / len(block)`` over the blocks that
        contain it.
        """
        if self._step_lower_bounds[weight, num_completed] >= upper_bound:
            return True
        if not self.pivot_blocks or self.max_coverage == 1:
            return False
        # No requirement exceeds its block, since a set that has completed the last layer makes the
        # coverage bound certify any upper bound.
        requirements = self._get_requirements(weight, num_completed)
        largest_fractions: dict[int, Fraction] = {}
        for block, requirement in zip(self.pivot_blocks, requirements):
            for column in block:
                fraction = Fraction(max(requirement, 0), len(block))
                largest_fractions[column] = max(fraction, largest_fractions.get(column, fraction))
        reachable = math.ceil(sum(largest_fractions.values(), Fraction(0)))
        reachable += (-reachable) % self.weight_divisor
        return (
            reachable >= upper_bound and self.step_lower_bound(weight, num_completed) >= upper_bound
        )

    def get_num_sets_to_certify(self, weight: int, upper_bound: int) -> int:
        """The fewest leading sets whose completion of layer ``weight`` certifies a bound.

        Once these sets have completed layer ``weight``, and the others layer ``weight - 1``, every
        unvisited eligible word weighs at least ``upper_bound``.  Returns the number of sets if no
        fewer sets suffice, in which case the bound may need later layers too.
        """
        num_sets = len(self.ranks)
        for num_completed in range(1, num_sets):
            if self.certifies(weight, num_completed, upper_bound):
                return num_completed
        return num_sets

    def work(self, upper_bound: int) -> int:
        """Estimate the candidate/set evaluations needed to certify ``upper_bound``.

        This pessimistic estimate includes systematic rows (coefficient weight one).  It assumes no
        lighter witness is found before the certificate reaches the upper bound, and it uses only
        the coverage bound.
        """
        arguments = self._as_array_arguments()
        return int(
            estimate_certificate_work(
                self.dimension, *arguments, upper_bound, get_capped_binomials(self.dimension)
            )
        )

    def _as_array_arguments(self) -> tuple[npt.NDArray[np.int64], int, int]:
        """The ranks, weight divisor and maximum coverage that the array functions take."""
        return np.array(self.ranks, dtype=np.int64), self.weight_divisor, self.max_coverage

    @functools.cached_property
    def _step_lower_bounds(self) -> npt.NDArray[np.int64]:
        """The table of ``get_step_lower_bounds`` for this certificate."""
        return get_step_lower_bounds(self.dimension, *self._as_array_arguments())

    def _get_requirements(self, weight: int, num_completed: int) -> list[int]:
        """The fewest ones that an unvisited word has in each block after the given step."""
        return [
            (weight if index < num_completed else weight - 1) + 1 - (self.dimension - rank)
            for index, rank in enumerate(self.ranks)
        ]

    @functools.cached_property
    def _lp_lower_bounds(self) -> dict[tuple[int, int], int]:
        """The rounded bounds of ``get_lp_lower_bound`` for the steps computed so far."""
        return {}


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


@compiled_on_request
def get_overlapping_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    basis_family: BasisFamily,
    upper_bound: int,
    max_sets: int,
    minimum_work: int,
    attempts_per_set: int,
    binomials: npt.NDArray[np.int64],
    rng: np.random.Generator,
) -> BasisFamily:
    """Stage three: build a replacement basis family of overlapping full sets that does less work.

    ``basis_family`` holds the full and partial sets of one independent binary basis, whose quotient
    labels ``labels`` have no columns if it has none.  A replacement holds only full sets, with
    their labels transformed alongside their rows, and is certified with its maximum coverage of a
    Hamming coordinate.  ``max_sets`` bounds its size and the effort spent building it: more sets
    can avoid costly coefficient layers in hard searches, while fewer reduce set-up work.  No
    replacement is built if the estimated work of ``basis_family`` is below ``minimum_work``.  The
    construction tries ``attempts_per_set`` column orders for each set that it may add.

    Returns the replacement, which has no sets if no replacement does less work.  ``binomials`` are
    ``get_capped_binomials(dimension)``, and ``rng`` breaks ties between equally used columns.
    Searches take ``max_sets``, ``minimum_work``, ``attempts_per_set``, ``binomials``, and ``rng``
    from :func:`get_overlapping_family_policy`.
    """
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
    # Saturate the number of attempts, which compiled code computes with 64-bit integers.
    num_open = max(max_sets - num_full, 1)
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


def get_overlapping_family_policy(
    tuning: SearchTuning, dimension: int
) -> tuple[int, int, int, npt.NDArray[np.int64], np.random.Generator]:
    """The arguments of stage three after ``upper_bound``, from ``max_sets`` on, for a search.

    They are the limit, the minimum work, and the attempts of ``tuning``, the capped binomials of
    the dimension, and a new generator with a fixed seed, so that every search, distance or count,
    chooses the same replacement for the same basis family and tuning.
    """
    return (
        tuning.max_information_sets,
        tuning.overlapping_minimum_work,
        tuning.overlapping_attempts_per_set,
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
    """The result of ``get_overlapping_family`` when no replacement does less work: no sets."""
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
# certificates


def get_lp_lower_bound(
    pivot_blocks: Sequence[Sequence[int]], requirements: Sequence[int], max_denominator: int
) -> int:
    """Bound the weight of binary words with given numbers of ones in blocks, by linear programming.

    Let ``B_i = pivot_blocks[i]`` and ``b_i = requirements[i]``.  Every binary word with at least
    ``b_i`` ones in the columns of ``B_i``, for each ``i``, has weight at least the optimum of the
    linear program

        minimize ``sum_c x_c`` subject to ``sum_{c in B_i} x_c >= b_i`` and ``0 <= x_c <= 1``.

    By weak duality, for every ``y >= 0``, such a word has weight at least the dual value

        ``sum_i b_i y_i - sum_c max(0, sum_{i : c in B_i} y_i - 1)``.

    The solver of ``scipy.optimize.linprog`` finds a nearly optimal ``y``, and the bound is the
    ceiling of its dual value in exact rational arithmetic, so rounding in the solver can only
    weaken the bound, never invalidate it.  The coverage bound is the dual value of ``y_i = 1/C``
    for the maximum coverage ``C``, so the optimum is at least that bound.

    Returns ``_NO_WORD_BOUND`` if a requirement exceeds the size of its block, since then no word
    meets the requirements, and zero if no requirement is positive or if the solver fails.
    """
    blocks = [
        list(block) for block, requirement in zip(pivot_blocks, requirements) if requirement > 0
    ]
    positive = [requirement for requirement in requirements if requirement > 0]
    if not blocks:
        return 0
    if any(requirement > len(block) for block, requirement in zip(blocks, positive)):
        return _NO_WORD_BOUND

    # Import the solver only here, since importing qLDPC does not otherwise load it.
    import scipy.optimize

    # Number the columns of the blocks, and solve the linear program in floating point.
    columns = sorted({column for block in blocks for column in block})
    index = {column: position for position, column in enumerate(columns)}
    incidence = np.zeros((len(blocks), len(columns)))
    for row, block in enumerate(blocks):
        incidence[row, [index[column] for column in block]] = 1
    result = scipy.optimize.linprog(
        np.ones(len(columns)),
        A_ub=-incidence,
        b_ub=-np.array(positive, dtype=float),
        bounds=(0, 1),
        method="highs",
    )
    if result.status != 0:
        return 0

    # Evaluate the dual value exactly.  The solver gives the dual of each constraint
    # ``-sum x >= -b`` as a nonpositive marginal.
    weights = [
        max(
            Fraction(0),
            Fraction(-float(marginal)).limit_denominator(max_denominator),
        )
        for marginal in result.ineqlin.marginals
    ]
    value = sum(
        (requirement * weight for requirement, weight in zip(positive, weights)), Fraction(0)
    )
    for position in range(len(columns)):
        load = sum(
            (weight for row, weight in enumerate(weights) if incidence[row, position]), Fraction(0)
        )
        value -= max(Fraction(0), load - 1)
    return max(0, math.ceil(value))


@functools.cache
def get_capped_binomials(dimension: int) -> npt.NDArray[np.int64]:
    """The binomial coefficients ``C(dimension, weight)`` for every weight, capped to fit.

    The array is shared between calls, so it is read-only.
    """
    binomials = np.array(
        [min(math.comb(dimension, weight), _BINOMIAL_CAP) for weight in range(dimension + 1)],
        dtype=np.int64,
    )
    binomials.flags.writeable = False
    return binomials


@always_compiled
def lower_bound_from_occupancy(occupancy: int, weight_divisor: int, max_coverage: int) -> int:
    """Turn the pivot occupancy of a certificate into a bound on the weight of unvisited words."""
    # Correct for repeated Hamming coordinates by dividing occupancy and rounding up.
    bound = (occupancy + max_coverage - 1) // max_coverage
    if weight_divisor > 1:
        # Every eligible Hamming weight is a multiple of weight_divisor, so the bound rounds up.
        bound += (-bound) % weight_divisor
    return bound


@always_compiled
def get_step_lower_bounds(
    dimension: int,
    ranks: npt.NDArray[np.int64],
    weight_divisor: int,
    max_coverage: int,
) -> npt.NDArray[np.int64]:
    """Certificate lower bounds after each step of a search, which completes one layer of one set.

    Entry ``[weight, num_completed]`` holds once the first ``num_completed`` sets, in family order,
    have completed layer ``weight``, and the others have completed layer ``weight - 1``, in whatever
    order their words were visited.  So column ``len(ranks)`` holds once every set has completed
    layer ``weight``, and column zero of row ``weight`` repeats column ``len(ranks)`` of row
    ``weight - 1``.  Rank deficiency limits how many selected rows can have zero pivot support: a
    set with deficiency ``s`` that has completed layer ``c`` adds ``c + 1 - s`` to the occupancy if
    positive, so it adds one more for each layer that it completes once its deficiency no longer
    keeps its contribution at zero.  Entries ``[dimension, j]`` for ``j >= 1`` hold
    ``_NO_WORD_BOUND``, since a set that has completed layer ``dimension`` has visited every word.
    """
    num_sets = len(ranks)
    bounds = np.empty((dimension + 1, num_sets + 1), dtype=np.int64)
    for weight in range(dimension + 1):
        # The occupancy once every set has completed layer ``weight - 1``.
        occupancy = 0
        for rank in ranks:
            occupancy += max(0, weight - (dimension - rank))
        bounds[weight, 0] = lower_bound_from_occupancy(occupancy, weight_divisor, max_coverage)

        # Completing layer ``weight`` adds one for each set whose deficiency it overcomes.
        for index in range(num_sets):
            if weight + 1 > dimension - ranks[index]:
                occupancy += 1
            bounds[weight, index + 1] = lower_bound_from_occupancy(
                occupancy, weight_divisor, max_coverage
            )
    bounds[dimension, 1:] = _NO_WORD_BOUND
    return bounds


@always_compiled
def estimate_certificate_work(
    dimension: int,
    ranks: npt.NDArray[np.int64],
    weight_divisor: int,
    max_coverage: int,
    upper_bound: int,
    binomials: npt.NDArray[np.int64],
) -> int:
    """Estimate the candidate/set evaluations needed to certify ``upper_bound``.

    The search completes its last layer on only as many leading sets, in family order, as certify
    the upper bound (see ``get_step_lower_bounds``).  This pessimistic estimate includes systematic
    rows (coefficient weight one).  It assumes no lighter witness is found before the certificate
    reaches the upper bound.  Estimates saturate.
    """
    bounds = get_step_lower_bounds(dimension, ranks, weight_divisor, max_coverage)
    candidates = 0
    for weight in range(1, dimension + 1):
        # Saturate before adding, so that no sum can overflow.
        if binomials[weight] > _WORK_CAP // max(len(ranks), 1):
            return _WORK_CAP
        for num_completed in range(1, len(ranks) + 1):
            candidates += binomials[weight]
            if bounds[weight, num_completed] >= upper_bound:
                return min(candidates, _WORK_CAP)
        if candidates >= _WORK_CAP:
            return _WORK_CAP
    return candidates


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
    coverage.  Returns ``_WORK_CAP`` if ``max_sets`` is zero.
    """
    optimistic_work = _WORK_CAP
    for num_sets in range(1, max_sets + 1):
        coverage = (num_sets * dimension + num_columns - 1) // num_columns
        full_ranks = np.full(num_sets, dimension, dtype=np.int64)
        work = estimate_certificate_work(
            dimension, full_ranks, weight_divisor, coverage, upper_bound, binomials
        )
        optimistic_work = min(optimistic_work, work)
    return optimistic_work


####################################################################################################
# coefficient layers


@always_compiled(inline=True)
def advance_support(
    rows: npt.NDArray[np.uint64],
    widths: npt.NDArray[np.int64],
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
) -> int:
    """Move to the next support of a layer of the search.

    Layer ``w = len(support) >= 1`` holds, for each information set, the sums of its offset and
    ``w`` of its rows.  A support lists the indices of such rows in increasing order, and the
    supports come in lexicographic order.  ``rows`` holds the packed rows of each information set,
    and set ``j`` uses the first ``widths[j]`` machine words of each row.  The rows chosen so far
    are ``support[:depth]``, and ``support[depth]`` is the next row to try at ``depth``.  Row
    ``level`` of ``prefixes`` holds, for each set, its offset plus its rows ``support[:level]``.
    Start with ``depth = 0``, ``support[0] = 0``, and the offsets in ``prefixes[0]``, which this
    function does not change.

    Return the last depth, ``len(support) - 1``, once ``support`` is complete.  The word of the
    support in set ``j`` is then ``prefixes[depth, j] ^ rows[j, support[depth]]``.  To continue,
    raise ``support[depth]``, by one or past the last row to skip the supports that differ only in
    their last row, and call again from that depth.  Return -1 once every support has been visited.
    """
    num_rows = rows.shape[1]
    weight = len(support)
    while depth >= 0:
        # Backtrack once the row at this depth leaves too few rows to complete the support.
        if support[depth] > num_rows - weight + depth:
            depth -= 1
            if depth >= 0:
                support[depth] += 1
            continue
        if depth + 1 == weight:
            return depth

        # Add the row at this depth to the prefixes, and try the next rows at the next depth.
        row = support[depth]
        for set_index in range(len(widths)):
            for word in range(widths[set_index]):
                prefixes[depth + 1, set_index, word] = (
                    prefixes[depth, set_index, word] ^ rows[set_index, row, word]
                )
        depth += 1
        support[depth] = row + 1
    return depth
