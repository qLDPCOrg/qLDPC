# SPDX-License-Identifier: Apache-2.0

"""Automorphism families: the images of one information set under automorphisms of a problem.

A BZ search can scan one information set ``I`` and certify with all of its images under a group
``Γ`` of automorphisms.  Sections cited in this module are those of ``brouwer_zimmermann.md``, whose
Section 3.7 derives the certificate and the choice of ``I``.  The connected-cluster search also
uses ``verify_automorphisms`` and ``get_orbits``, and ranks its variables by the orbit numbers of
``get_orbits`` (Section 3.5 of ``connected_cluster.md``).
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_rank_packed, pack_bits, row_reduce_packed, unpack_bits
from qldpc._compilation import always_compiled

from .._code_matrices import get_eligible_rows, get_nonzero_columns
from .basis_families import BasisFamily
from .certificates import Certificate, get_capped_binomials, plan_steps

####################################################################################################
# automorphism families


def verify_automorphisms(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    automorphisms: npt.NDArray[np.int_],
) -> None:
    """Check that column permutations map a search problem onto itself.

    ``basis`` is an independent basis of the searched space, whose rows with zero ``labels`` span
    the excluded subspace (see ``get_nested_code_basis``).  Row ``i`` of ``automorphisms`` maps
    column ``j`` to column ``automorphisms[i, j]``.

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


def get_automorphism_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    automorphisms: npt.NDArray[np.int_],
    weight_divisor: int,
) -> tuple[BasisFamily, Certificate]:
    """Choose an information set ``I`` to scan, and certify it with its images (Section 3.7).

    ``basis``, ``labels``, and ``automorphisms`` are as in ``verify_automorphisms``, which must
    accept them, and ``weight_divisor`` is a proven divisor of the Hamming weight of every eligible
    word.  ``I`` is chosen to lower the largest share ``|I ∩ O| / |O|`` over the orbits ``O``, since
    the certificate depends on that share (see ``_balance_orbit_shares``).

    Returns:
        A basis family of ``I`` alone, and the certificate of all of its images.
    """
    dimension, num_columns = basis.shape
    orbits = get_orbits(automorphisms, num_columns)
    rows, packed_labels, pivots = row_reduce_packed(
        pack_bits(basis), pack_bits(labels), get_nonzero_columns(basis)
    )
    _balance_orbit_shares(rows, packed_labels, pivots, orbits)
    generators, set_labels = (
        unpack_bits(rows, num_columns),
        unpack_bits(packed_labels, labels.shape[1]),
    )

    # The certificate of the |Γ| images depends only on |Γ| / C = min |O| / |I ∩ O| (Section 3.7),
    # which ``certificate`` represents in lowest terms as a number of sets and a coverage.
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


def estimate_automorphism_family_work(certificate: Certificate, upper_bound: int) -> int:
    """Estimate the candidate/set evaluations that certify ``upper_bound`` with automorphisms.

    ``certificate`` is that of an automorphism family, whose search scans one set and advances its
    certificate once per layer (Section 3.7).  So the estimate plans one set whose bound after each
    layer is that of the whole family.
    """
    dimension, num_sets = certificate.dimension, len(certificate.ranks)
    layer_bounds = certificate.get_weight_bounds(1)[:, [0, num_sets]]
    # A plan that leaves out no sets reads no ranks, so the rank of the one set is a placeholder.
    _, _, work = plan_steps(
        layer_bounds,
        upper_bound - 1,
        1,
        dimension,
        np.array([dimension], dtype=np.int64),
        get_capped_binomials(dimension),
        False,
    )
    return work


def get_orbits(automorphisms: npt.NDArray[np.int_], num_columns: int) -> npt.NDArray[np.int64]:
    """Number the orbits of the group that the permutations generate, and label each column.

    The orbits are numbered in the order of their least columns.
    """
    # Lower each column's label to the least column known to share its orbit, until no permutation
    # lowers a label.  Labels stay within orbits, so each ends at the least column of its orbit.
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
    """Exchange pivots, in place, from orbits of the largest share to emptier ones while possible.

    ``rows`` is a packed basis row reduced on the information set ``I``, whose columns are
    ``pivots``, and ``labels`` its packed quotient labels.  Each exchange takes a row whose pivot
    lies in an orbit ``O`` of the largest share ``|I ∩ O| / |O|``, and pivots it on a column of
    another orbit where it has a one, if that orbit's share with the new pivot stays below that of
    ``O``.  For two orbits, the exchanges stop once the emptier orbit's own pivot columns span its
    columns, or once one more pivot would make it at least as full as ``O``, so they reach the least
    largest share of any information set (Section 3.7).
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
