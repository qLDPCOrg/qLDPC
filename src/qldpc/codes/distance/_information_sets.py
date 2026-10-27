# SPDX-License-Identifier: Apache-2.0

"""Basis families of information sets and their certificates, shared by exact BZ and coset searches.

Exact distance search (``_brouwer_zimmermann.py``) and fixed-weight logical counting
(``_counting/scan.py``) enumerate coefficient layers over the same basis families.  Layer ``w`` of
an information set holds the words that add ``w`` of its rows to a starting word, which is zero
except in the coset search (see ``brouwer_zimmermann.md`` for the terminology).  This module owns
how those basis families are built, pruned, and replaced by overlapping ones, and the certificate
that bounds the Hamming weight of eligible words that a basis family has not yet visited.  The
Pauli-coset oracle (``_bounded_weight_search.py``) uses the disjoint information sets for its own
layered search.  All three enumerate a layer with :func:`advance_support`.

A :class:`BasisFamily` stacks its information sets into arrays, so that one set of functions serves
every caller, in the subset of Python that numba compiles, and some of them are compiled (see
``qldpc/_compilation.py``).  Its :class:`Certificate` gives the bound and the work estimate.

Basis family construction runs in stages, so that a caller can take its own exits between them:

1. :func:`get_disjoint_family` packs disjoint information sets and infers ambient even parity.
2. :func:`prune_family` drops partial sets that cannot help certify a target.
3. :func:`get_overlapping_family` may build a cheaper basis family of overlapping full sets.

Each stage aims at a target ``upper_bound``.  Distance passes its current witness weight ``U`` and
stops once the certificate reaches ``U``.  Counting at weight ``w`` must also visit every word of
weight exactly ``w``, so it passes ``w + 1`` and stops only once the certificate exceeds ``w``.
Section 4.4 of ``brouwer_zimmermann.md`` describes the heuristics, and Sections 2.5 and 3.1 derive
the certificate.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._compilation import always_compiled, compiled_on_request

from ._binary import row_reduce_binary, row_reduce_packed

# Default limit on the number of sets in a replacement basis family (see get_overlapping_family).
MAX_SETS = 16

# Estimated work below which a basis family is not replaced by one of overlapping sets.
MINIMUM_WORK = 100_000

# Caps that keep work estimates, which only compare candidate basis families, within a native
# integer.  Estimates above a cap compare as equal.  No search is large enough for the choice among
# basis families with such estimates to matter.
_BINOMIAL_CAP = 1 << 50
_WORK_CAP = 1 << 61

# One disjoint information set: its rows, its labels, and its pivot columns.
_DisjointSet = tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.int_]]


####################################################################################################
# basis families


@dataclass(frozen=True)
class Certificate:
    """Lower bound on the Hamming weight of eligible words that a family has not yet visited.

    The bound holds only after every search basis of the family has completed each coefficient layer
    through ``completed_weight``; an interrupted layer does not advance it.  It does not bound the
    weights of witnesses already found, and it does not transfer to another basis or CSS sector.

    Attributes:
        dimension: Number of rows of the searched basis.
        ranks: Rank of the pivot block of each search basis, in family order.
        weight_divisor: A proven divisor of the Hamming weight of every eligible word.  It combines
            any divisor proven by the caller, such as two for the symplectic-to-Hamming map, with
            ambient even parity.  It is in the Hamming units of the searched basis, not Pauli units.
        max_coverage: An upper bound on the number of pivot blocks that contain any one coordinate.
            Overestimates are sound; underestimates are not.
    """

    dimension: int
    ranks: tuple[int, ...]
    weight_divisor: int = 1
    max_coverage: int = 1

    def lower_bound(self, completed_weight: int) -> int:
        """Bound the Hamming weight of unvisited eligible words after the given complete layers."""
        bounds = get_certificate_lower_bounds(self.dimension, *self._as_array_arguments())
        return int(bounds[completed_weight])

    def work(self, upper_bound: int) -> int:
        """Estimate the candidate/set evaluations needed to certify ``upper_bound``.

        This pessimistic estimate includes systematic rows (coefficient weight one).  It assumes no
        lighter witness is found before the certificate reaches the upper bound.
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


####################################################################################################
# basis family stages


def get_disjoint_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    *,
    weight_divisor: int = 1,
) -> BasisFamily:
    """Stage one: pack disjoint information sets, and certify them with any inferred parity."""
    dimension = len(basis)
    set_labels = np.empty((dimension, 0), dtype=np.uint8) if labels is None else labels
    sets = _get_disjoint_sets(basis, set_labels)
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
    binomials: npt.NDArray[np.int64],
    rng: np.random.Generator,
) -> BasisFamily:
    """Stage three: build a replacement basis family of overlapping full sets that does less work.

    ``basis_family`` holds the full and partial sets of one independent binary basis, whose quotient
    labels ``labels`` have no columns if it has none.  A replacement holds only full sets, with
    their labels transformed alongside their rows, and is certified with its maximum coverage of a
    Hamming coordinate.  ``max_sets`` bounds its size and the effort spent building it: more sets
    can avoid costly coefficient layers in hard searches, while fewer reduce set-up work.  No
    replacement is built if the estimated work of ``basis_family`` is below ``minimum_work``.

    Returns the replacement, which has no sets if no replacement does less work.  ``binomials`` are
    ``get_capped_binomials(dimension)``.  ``rng`` breaks ties between equally used columns, so a
    caller that wants the same choice every time passes a generator with a fixed seed.
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

    # With m full sets, some column has coverage at least ceil(m * dimension / len(columns)).
    optimistic_work = _WORK_CAP  # work estimates saturate at this cap
    for count in range(1, max_sets + 1):
        coverage = (count * dimension + num_columns - 1) // num_columns
        full_ranks = np.full(count, dimension, dtype=np.int64)
        work = estimate_certificate_work(
            dimension, full_ranks, weight_divisor, coverage, upper_bound, binomials
        )
        optimistic_work = min(optimistic_work, work)
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
    for _ in range(2 * (max_sets - num_full)):
        # Order the columns by load, and then randomly, with two stable sorts, since numba has no
        # lexicographic sort.  Transform the labels along with each candidate.
        by_random = np.argsort(rng.random(num_columns), kind="mergesort")
        by_coverage = np.argsort(column_coverage[columns][by_random], kind="mergesort")
        reduced, reduced_labels, new_pivots = row_reduce_packed(
            basis, labels, columns[by_random[by_coverage]]
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
        generators[num_sets] = reduced
        set_labels[num_sets] = reduced_labels
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


def infer_weight_divisor(basis: npt.NDArray[np.uint8], weight_divisor: int = 1) -> int:
    """Combine a caller's proven Hamming-weight divisor with ambient even parity, if present."""
    # XOR preserves even parity, so an all-even basis spans only even Hamming weights.
    if np.all(np.count_nonzero(basis, axis=1) % 2 == 0):
        return math.lcm(weight_divisor, 2)
    return weight_divisor


####################################################################################################
# certificates


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
def get_certificate_lower_bounds(
    dimension: int,
    ranks: npt.NDArray[np.int64],
    weight_divisor: int,
    max_coverage: int,
) -> npt.NDArray[np.int64]:
    """Certificate lower bounds after each completed layer ``0, ..., dimension``.

    Rank deficiency limits how many selected rows can have zero pivot support: after ``weight``
    complete layers, a set with deficiency ``s`` adds ``weight + 1 - s`` to the occupancy if
    positive.
    """
    num_deficient = np.zeros(dimension + 1, dtype=np.int64)
    for rank in ranks:
        num_deficient[dimension - rank] += 1
    bounds = np.empty(dimension + 1, dtype=np.int64)
    num_contributing = 0
    deficiency_total = 0
    for weight in range(dimension + 1):
        num_contributing += num_deficient[weight]
        deficiency_total += weight * num_deficient[weight]
        occupancy = num_contributing * (weight + 1) - deficiency_total
        bounds[weight] = lower_bound_from_occupancy(occupancy, weight_divisor, max_coverage)
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

    This pessimistic estimate includes systematic rows (coefficient weight one).  It assumes no
    lighter witness is found before the certificate reaches the upper bound.  Estimates saturate.
    """
    bounds = get_certificate_lower_bounds(dimension, ranks, weight_divisor, max_coverage)
    candidates = 0
    for weight in range(1, dimension + 1):
        # Saturate before adding, so that no sum can overflow.
        if binomials[weight] > _WORK_CAP // max(len(ranks), 1):
            return _WORK_CAP
        candidates += len(ranks) * binomials[weight]
        if candidates >= _WORK_CAP:
            return _WORK_CAP
        if bounds[weight] >= upper_bound:
            break
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
    """Move to the next set of rows in a layer of the search.

    Layer ``w = len(support)`` of a basis family holds a starting word of each information set plus
    each sum of ``w`` of its rows.  A support lists the indices of such rows in increasing order,
    and the supports come in lexicographic order.  ``rows`` holds the packed rows of each
    information set, and set ``j`` uses its first ``widths[j]`` words.  The rows chosen so far are
    ``support[:depth]``, and ``support[depth]`` is the next row to try at ``depth``.  Row ``level``
    of ``prefixes`` holds, for each information set, its starting word plus its rows
    ``support[:level]``, and row zero holds the starting words.

    Return the last depth, ``len(support) - 1``, once ``support`` is complete: its sum in set ``j``
    is then ``prefixes[depth, j] ^ rows[j, support[depth]]`` at that depth.  To continue, increment
    ``support[depth]``, possibly several times to visit the supports that differ only in their last
    row, and call again from that depth.  Return -1 once every support has been visited.
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


####################################################################################################
# helper functions


@compiled_on_request
def _copy_with_capacity(array: Any, capacity: int) -> Any:
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
    basis: npt.NDArray[np.uint8], labels: npt.NDArray[np.uint8]
) -> list[_DisjointSet]:
    """Construct a large deterministic packing of disjoint information sets."""
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

    # Otherwise, try up to 8 random column orders.  The fixed seed makes the result reproducible.
    rng = np.random.default_rng(0)
    for _ in range(8):
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
