# SPDX-License-Identifier: Apache-2.0

"""Information-set families and lower certificates shared by exact BZ and Pauli-coset searches.

Exact distance search (``_brouwer_zimmermann.py``) and fixed-weight logical counting
(``_counting/scan.py``) enumerate coefficient layers over the same families of systematic bases.
This module owns how those families are built, pruned, and replaced by overlapping families, and the
certificate that bounds the Hamming weight of eligible words that a family has not yet visited.  The
Pauli-coset oracle (``_bounded_weight_search.py``) uses the disjoint information sets and the
certificate for its own layered search.

Family construction runs in stages, so that a caller can take its own exits between them:

1. :func:`get_disjoint_family` packs disjoint information sets and infers ambient even parity.
2. :func:`prune_family` drops partial sets that cannot help certify a target.
3. :func:`get_overlapping_family` may propose a cheaper family of overlapping full sets.

Each stage aims at a target ``upper_bound``.  Distance passes its current witness weight ``U`` and
stops once the certificate reaches ``U``.  Counting at weight ``w`` must also visit every word of
weight exactly ``w``, so it passes ``w + 1`` and stops only once the certificate exceeds ``w``.
:func:`prepare_family` runs every stage without exits.  Section 4.4 of ``brouwer_zimmermann.md``
describes the heuristics, and Sections 2.5 and 3.1 derive the certificate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from ._binary import row_reduce_binary

# Default limit on the number of sets in a replacement family (see get_overlapping_family).
MAX_SETS = 16


####################################################################################################
# information-set families


class InformationSet(NamedTuple):
    """Generators, labels, rank, and pivot columns for one systematic column block."""

    generators: npt.NDArray[np.uint8]
    labels: npt.NDArray[np.uint8] | None
    rank: int
    pivots: npt.NDArray[np.int_]


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
        # Rank deficiency limits how many selected rows can have zero pivot support.
        bound = sum(max(0, completed_weight + 1 - (self.dimension - rank)) for rank in self.ranks)

        # Correct for repeated Hamming coordinates by dividing occupancy and rounding up.
        bound = (bound + self.max_coverage - 1) // self.max_coverage
        if self.weight_divisor > 1:
            # Every eligible Hamming weight is a multiple of weight_divisor, so the bound rounds up.
            bound += (-bound) % self.weight_divisor
        return bound

    def work(self, upper_bound: int) -> int:
        """Estimate the candidate/set evaluations needed to certify ``upper_bound``.

        This pessimistic estimate includes systematic rows (coefficient weight one).  It assumes no
        lighter witness is found before the certificate reaches the upper bound.
        """
        candidates = 0
        for weight in range(1, self.dimension + 1):
            candidates += len(self.ranks) * math.comb(self.dimension, weight)
            if self.lower_bound(weight) >= upper_bound:
                break
        return candidates


@dataclass(frozen=True)
class BasisFamily:
    """Search bases for one binary basis, and the certificate that their complete layers give."""

    information_sets: tuple[InformationSet, ...]
    certificate: Certificate

    def __post_init__(self) -> None:
        assert self.certificate.ranks == tuple(item.rank for item in self.information_sets)
        assert self.certificate.max_coverage >= _get_max_coverage(self.information_sets)

    @classmethod
    def from_information_sets(
        cls,
        information_sets: Sequence[InformationSet],
        *,
        dimension: int,
        weight_divisor: int,
    ) -> BasisFamily:
        """Certify a family with its ranks and its actual maximum pivot coverage."""
        sets = tuple(information_sets)
        certificate = Certificate(
            dimension,
            tuple(item.rank for item in sets),
            weight_divisor,
            _get_max_coverage(sets),
        )
        return cls(sets, certificate)


def _get_max_coverage(information_sets: Sequence[InformationSet]) -> int:
    """Maximum number of pivot blocks containing any one coordinate, and at least one."""
    pivots = [item.pivots for item in information_sets]
    loads = np.bincount(np.concatenate(pivots)) if pivots else np.zeros(0, dtype=int)
    return max(1, int(loads.max(initial=0)))


####################################################################################################
# family stages


def prepare_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    *,
    upper_bound: int,
    weight_divisor: int = 1,
    disjoint: BasisFamily | None = None,
    max_sets: int = MAX_SETS,
) -> BasisFamily:
    """Run every family stage, aimed at a fixed ``upper_bound``, without intermediate exits.

    The first stage does not depend on ``upper_bound``, so a caller that prepares several families
    for one basis can pass its result as ``disjoint`` instead of computing it again.
    """
    family = (
        get_disjoint_family(basis, labels, weight_divisor=weight_divisor)
        if disjoint is None
        else disjoint
    )
    family = prune_family(family, upper_bound=upper_bound)
    overlapping = get_overlapping_family(
        basis, labels, family, upper_bound=upper_bound, max_sets=max_sets
    )
    return family if overlapping is None else overlapping


def get_disjoint_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    *,
    weight_divisor: int = 1,
) -> BasisFamily:
    """Stage one: pack disjoint information sets, and certify them with any inferred parity."""
    return BasisFamily.from_information_sets(
        get_information_set_generators(basis, labels),
        dimension=len(basis),
        weight_divisor=infer_weight_divisor(basis, weight_divisor),
    )


def prune_family(family: BasisFamily, *, upper_bound: int) -> BasisFamily:
    """Stage two: retain partial sets that can strengthen certification of ``upper_bound``."""
    certificate = family.certificate
    dimension = certificate.dimension
    num_full_sets = sum(item.rank == dimension for item in family.information_sets)
    if not num_full_sets:
        return family
    full_sets = replace(certificate, ranks=(dimension,) * num_full_sets)
    full_set_certifying_weight = next(
        (
            completed_weight
            for completed_weight in range(dimension + 1)
            if full_sets.lower_bound(completed_weight) >= upper_bound
        ),
        dimension + 1,
    )
    # A dropped partial set adds zero lower-bound units until full sets certify the upper bound.
    retained = tuple(
        item
        for item in family.information_sets
        if item.rank == dimension or dimension - item.rank < full_set_certifying_weight
    )
    return BasisFamily(retained, replace(certificate, ranks=tuple(item.rank for item in retained)))


def get_overlapping_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    family: BasisFamily,
    *,
    upper_bound: int,
    max_sets: int = MAX_SETS,
    minimum_work: int = 100_000,
) -> BasisFamily | None:
    """Stage three: look for a lower-work replacement for a disjoint family.

    ``family`` contains full or partial sets for one independent binary basis.  A replacement
    contains only full sets, with quotient labels transformed alongside rows.  ``max_sets`` bounds
    the size of a replacement family and the effort spent building it.  More sets can avoid costly
    coefficient layers in hard searches, while fewer reduce setup work.  Return a lower-work family
    certified with its maximum Hamming-coordinate coverage, or ``None``.
    """
    # Start from full blocks; skip complete packings and families already at the size limit.
    certificate = family.certificate
    dimension = certificate.dimension
    full_sets = [item for item in family.information_sets if item.rank == dimension]
    columns = np.flatnonzero(np.any(basis, axis=0))
    if not full_sets or len(full_sets) >= max_sets or len(full_sets) * dimension == len(columns):
        return None

    # Skip family construction when the existing search is already cheap.
    best_work = certificate.work(upper_bound)
    if best_work < minimum_work:
        return None

    # With m full sets, some column has coverage at least ceil(m * dimension / len(columns)).
    optimistic_work = min(
        replace(
            certificate,
            ranks=(dimension,) * count,
            max_coverage=(count * dimension + len(columns) - 1) // len(columns),
        ).work(upper_bound)
        for count in range(1, max_sets + 1)
    )
    if optimistic_work >= best_work:
        return None

    # Track coverage and distinct sets; fixed-seed tie-breaking keeps choices reproducible.
    loads = np.bincount(
        np.concatenate([item.pivots for item in full_sets]), minlength=basis.shape[1]
    )
    seen = {tuple(sorted(item.pivots.tolist())) for item in full_sets}
    rng = np.random.default_rng(0)
    best_sets = None
    best_coverage = 1
    for _ in range(2 * (max_sets - len(full_sets))):
        # Prefer less-used columns; transform quotient labels with each candidate basis.
        order = columns[np.lexsort((rng.random(len(columns)), loads[columns]))]
        generators, transformed_labels, pivots = row_reduce_binary(basis, labels, columns=order)
        assert len(pivots) == dimension

        # Do not add duplicate sets; their row reductions still count against the attempt limit.
        signature = tuple(sorted(pivots.tolist()))
        if signature in seen:
            continue
        seen.add(signature)
        full_sets.append(InformationSet(generators, transformed_labels, dimension, pivots))
        loads[pivots] += 1
        max_coverage = int(loads.max())

        # Keep improvements; stop at the set limit or when the optimistic work bound is attained.
        work = replace(
            certificate, ranks=(dimension,) * len(full_sets), max_coverage=max_coverage
        ).work(upper_bound)
        if work < best_work:
            best_sets, best_coverage, best_work = full_sets.copy(), max_coverage, work
        if len(full_sets) == max_sets or best_work <= optimistic_work:
            break
    if best_sets is None:
        return None
    return BasisFamily(
        tuple(best_sets),
        replace(certificate, ranks=(dimension,) * len(best_sets), max_coverage=best_coverage),
    )


def infer_weight_divisor(basis: npt.NDArray[np.uint8], weight_divisor: int = 1) -> int:
    """Combine a caller's proven Hamming-weight divisor with ambient even parity, if present."""
    # XOR preserves even parity, so an all-even basis spans only even Hamming weights.
    if np.all(np.count_nonzero(basis, axis=1) % 2 == 0):
        return math.lcm(weight_divisor, 2)
    return weight_divisor


####################################################################################################
# disjoint information sets


def get_information_set_generators(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
) -> list[InformationSet]:
    """Construct a large deterministic packing of disjoint information sets."""
    # Only columns that are nonzero in the basis can be pivots.  Pack them in their natural order.
    nonzero_columns = np.flatnonzero(np.any(basis, axis=0))
    best = _get_information_set_generators_for_columns(
        basis,
        labels,
        nonzero_columns,
    )

    # A full set has ``len(basis)`` pivots, and only the last set of a packing can have fewer.  Full
    # sets use disjoint nonzero columns, so there are at most this many, and possibly fewer.  Return
    # if the natural order already reaches this bound.
    max_full_sets = len(nonzero_columns) // len(basis)
    if sum(rank == len(basis) for _, _, rank, _ in best) == max_full_sets:
        return best

    # Otherwise, try up to 8 random column orders.  The fixed seed makes the result reproducible.
    rng = np.random.default_rng(0)
    for _ in range(8):
        # Keep the packing with the lexicographically larger tuple of ranks: more full sets, or
        # equally many and a larger last set.
        candidate = _get_information_set_generators_for_columns(
            basis,
            labels,
            rng.permutation(nonzero_columns),
        )
        candidate_ranks = tuple(rank for _, _, rank, _ in candidate)
        best_ranks = tuple(rank for _, _, rank, _ in best)
        if candidate_ranks > best_ranks:
            best = candidate

        # Stop early if the packing reaches the bound.
        if sum(rank == len(basis) for _, _, rank, _ in best) == max_full_sets:
            break
    return best


def _get_information_set_generators_for_columns(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    columns: npt.NDArray[np.int_],
) -> list[InformationSet]:
    """Construct disjoint information sets, with pivots from the nonzero ``columns`` in order."""
    # Start with every given column available as a pivot, and with the basis as it is.
    dimension = len(basis)
    active_columns = columns
    working = basis
    working_labels = labels
    information_sets: list[InformationSet] = []

    while len(active_columns):
        # Row-reduce on the available columns, taking pivots in their order, so that each pivot
        # column has a single one.  The labels follow the same row operations.
        working, working_labels, pivots = row_reduce_binary(
            working, working_labels, columns=active_columns
        )
        rank = len(pivots)
        assert rank  # The available columns are nonzero, so they give at least one pivot.
        information_sets.append(InformationSet(working, working_labels, rank, pivots))

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
