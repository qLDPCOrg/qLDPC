# SPDX-License-Identifier: Apache-2.0

"""Families of the translates of one information set by automorphisms of a search problem.

A permutation of columns that maps both the space ``V`` of a search and its excluded subspace ``S``
onto themselves maps each eligible word to an eligible word of the same weight.  Let ``G`` be a
group of such permutations, and ``I`` an information set.  A word with at most ``t`` ones on a
translate ``g(I)`` is the image of a word with the same weight and eligibility and at most ``t``
ones on ``I``.  So a BZ search can scan ``I`` alone, and certify with the family of all of its
translates, which completes each coefficient layer when ``I`` does.  Section 3.7 of
``brouwer_zimmermann.md`` derives the certificate of this family.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_rank_packed, pack_bits, row_reduce_packed, unpack_bits
from qldpc._compilation import always_compiled

from ._binary import get_eligible_rows
from ._information_sets import BasisFamily, Certificate, get_capped_binomials

####################################################################################################
# families of translates


def verify_automorphisms(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    automorphisms: npt.NDArray[np.int_],
) -> None:
    """Check that column permutations map a search problem onto itself.

    ``basis`` is an independent basis of the searched space.  Its rows with zero ``labels`` span the
    excluded subspace (see ``get_nested_code_basis``).  Row ``i`` of ``automorphisms`` maps column
    ``j`` to column ``automorphisms[i, j]``.

    Raises:
        ValueError: If a row of ``automorphisms`` is not a permutation of the columns of the basis,
            or does not map both the searched space and the excluded subspace onto themselves.
    """
    num_columns = basis.shape[1]
    columns = np.arange(num_columns)
    if (
        automorphisms.ndim != 2
        or automorphisms.shape[1] != num_columns
        or np.any(np.sort(automorphisms, axis=1) != columns)
    ):
        raise ValueError(f"Automorphisms must be permutations of {num_columns} columns")
    for permutation in automorphisms:
        for rows in (basis, basis[~get_eligible_rows(labels)]):
            # The rows are independent, so the permutation maps their span onto itself exactly if
            # their images add nothing to their rank.
            images = np.empty_like(rows)
            images[:, permutation] = rows
            if get_rank_packed(pack_bits(np.vstack([rows, images]))) > len(rows):
                raise ValueError("A permutation is not an automorphism of the search problem")


def get_translate_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    automorphisms: npt.NDArray[np.int_],
    weight_divisor: int,
) -> tuple[BasisFamily, Certificate]:
    """Choose an information set to scan, and certify it with its translates.

    ``basis``, ``labels``, and ``automorphisms`` are as in ``verify_automorphisms``, which must
    accept them, and ``weight_divisor`` is a proven divisor of the Hamming weight of every eligible
    word.  A column of an orbit ``O`` of the group lies in the same number of translates as every
    other column of ``O``, so the certificate is strongest when the scanned set ``I`` meets every
    orbit in an even share ``|I ∩ O| / |O|``.

    Returns:
        A basis family of ``I`` alone, and the certificate of all of its translates.
    """
    dimension, num_columns = basis.shape
    orbits = _get_orbits(automorphisms, num_columns)
    rows, packed_labels, pivots = row_reduce_packed(
        pack_bits(basis), pack_bits(labels), np.flatnonzero(np.any(basis, axis=0))
    )
    _balance_orbit_shares(rows, packed_labels, pivots, orbits)
    generators, set_labels = (
        unpack_bits(rows, num_columns),
        unpack_bits(packed_labels, labels.shape[1]),
    )

    # Each column of an orbit O lies in |I ∩ O| |G| / |O| of the |G| translates.  The coverage
    # bound of the family, ceil(|G| (t + 1) / C) for maximum coverage C, depends only on the ratio
    # |G| / C.  For the orbit with the largest share |I ∩ O| / |O|, this ratio is |O| / |I ∩ O|,
    # which the certificate represents in lowest terms as a number of sets and a coverage.
    sizes = np.bincount(orbits)
    counts = np.bincount(orbits[pivots], minlength=len(sizes))
    share = max(Fraction(int(count), int(size)) for count, size in zip(counts, sizes))
    certificate = Certificate(
        dimension, (dimension,) * share.denominator, weight_divisor, share.numerator
    )
    basis_family = BasisFamily(
        generators[None],
        set_labels[None],
        pivots[None],
        np.array([dimension], dtype=np.int64),
        weight_divisor,
        1,
    )
    return basis_family, certificate


def estimate_translate_work(certificate: Certificate, upper_bound: int) -> int:
    """Estimate the candidate/set evaluations of a search that certifies with translates.

    This is ``Certificate.work`` for a search that scans one set of the family of ``certificate``,
    whose other sets are its translates, and it is pessimistic in the same way.
    """
    binomials = get_capped_binomials(certificate.dimension)
    work = 0
    for weight in range(1, certificate.dimension + 1):
        work += int(binomials[weight])
        if certificate.lower_bound(weight) >= upper_bound:
            break
    return work


def _get_orbits(automorphisms: npt.NDArray[np.int_], num_columns: int) -> npt.NDArray[np.int64]:
    """Number the orbits of the group that the permutations generate, and label each column."""
    # Give each column the least column that it is known to share an orbit with, until no
    # permutation lowers a label.  Every label is a column of the same orbit, so the labels
    # settle on the least column of each orbit.
    least = np.arange(num_columns)
    while True:
        previous = least
        for permutation in automorphisms:
            least = np.minimum(least, least[permutation])
            np.minimum.at(least, permutation, least.copy())
        least = least[least]
        if np.array_equal(least, previous):
            return np.unique(least, return_inverse=True)[1].astype(np.int64)


@always_compiled
def _balance_orbit_shares(
    rows: npt.NDArray[np.uint64],
    labels: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    orbits: npt.NDArray[np.int64],
) -> None:
    """Exchange pivots between orbits, in place, while that lowers the largest orbit share.

    ``rows`` is a packed basis row reduced on the information set ``pivots``, and ``labels`` its
    packed quotient labels.  Pivoting row ``i`` on another column ``c`` where row ``i`` has a one
    replaces ``pivots[i]`` by ``c`` in the information set.  Each exchange moves a pivot from an
    orbit ``O`` with the largest share ``|I ∩ O| / |O|`` to an orbit whose share stays below it.

    For two orbits, the resulting largest share is the least of any information set: if no exchange
    moves a pivot out of ``O``, every column of the other orbit lies in the span of the pivots in
    it, so no information set has fewer pivots in ``O``.
    """
    sizes = np.bincount(orbits)
    counts = np.zeros(len(sizes), dtype=np.int64)
    is_pivot = np.zeros(len(orbits), dtype=np.bool_)
    for pivot in pivots:
        counts[orbits[pivot]] += 1
        is_pivot[pivot] = True
    while True:
        largest = 0
        for orbit in range(1, len(sizes)):
            if counts[orbit] * sizes[largest] > counts[largest] * sizes[orbit]:
                largest = orbit

        # Find a pivot of the fullest orbit, and a column of an orbit that can take it.
        row, column = -1, -1
        for candidate_row in range(len(pivots)):
            if orbits[pivots[candidate_row]] != largest:
                continue
            for candidate_column in range(len(orbits)):
                orbit = orbits[candidate_column]
                word = rows[candidate_row, candidate_column // 64]
                if (
                    not is_pivot[candidate_column]
                    and (counts[orbit] + 1) * sizes[largest] < counts[largest] * sizes[orbit]
                    and (word >> np.uint64(candidate_column % 64)) & np.uint64(1)
                ):
                    row, column = candidate_row, candidate_column
                    break
            if row >= 0:
                break
        if row < 0:
            return

        # Clear the new pivot column from the other rows, transforming the labels alongside.
        bit = np.uint64(1) << np.uint64(column % 64)
        for other in range(len(rows)):
            if other != row and rows[other, column // 64] & bit:
                for index in range(rows.shape[1]):
                    rows[other, index] ^= rows[row, index]
                for index in range(labels.shape[1]):
                    labels[other, index] ^= labels[row, index]
        counts[largest] -= 1
        counts[orbits[column]] += 1
        is_pivot[pivots[row]] = False
        is_pivot[column] = True
        pivots[row] = column
