# SPDX-License-Identifier: Apache-2.0

"""Families of the translates of one information set by automorphisms of a search problem.

A permutation of columns that maps both the space ``V`` of a search and its excluded subspace ``S``
onto themselves maps each eligible word to an eligible word of the same weight.  Given generators of
a group ``G`` of such permutations, a BZ search can scan one information set ``I`` and certify with
the family of all of its translates ``g(I)``: a word with at most ``t`` ones on ``g(I)`` is the
image of a word with at most ``t`` ones on ``I``, which has the same weight and eligibility.  So the
whole family completes each coefficient layer when ``I`` does.  Section 3.7 of
``brouwer_zimmermann.md`` derives the certificate of this family.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_rank_packed, pack_bits, row_reduce_binary

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

    ``basis`` is an independent basis of the searched space, whose rows with zero ``labels`` span
    its excluded subspace (see ``get_nested_code_basis``).  Row ``i`` of ``automorphisms`` maps
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

    Returns a basis family of ``I`` alone, and the certificate of all of its translates.
    """
    dimension = len(basis)
    orbits = _get_orbits(automorphisms, basis.shape[1])
    generators, set_labels, pivots = row_reduce_binary(
        basis, labels, columns=np.flatnonzero(np.any(basis, axis=0))
    )
    assert set_labels is not None  # labels in give labels out
    _balance_orbit_shares(generators, set_labels, pivots, orbits)

    # Each column of an orbit O lies in |I ∩ O| |G| / |O| of the |G| translates.  So the coverage
    # bound of the family, ceil(|G| (t + 1) / C) for maximum coverage C, is that of |O| sets with
    # coverage |I ∩ O| for the orbit with the largest share, which are reduced to lowest terms.
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


def _get_orbits(automorphisms: npt.NDArray[np.int_], num_columns: int) -> npt.NDArray[np.int_]:
    """Label each column by its orbit under the group that the permutations generate."""
    # Import the graph algorithms only here, since importing qLDPC does not otherwise load them.
    import scipy.sparse
    import scipy.sparse.csgraph

    sources = np.tile(np.arange(num_columns), len(automorphisms))
    graph = scipy.sparse.coo_array(
        (np.ones(len(sources)), (sources, automorphisms.ravel())), shape=(num_columns, num_columns)
    )
    return scipy.sparse.csgraph.connected_components(graph, directed=False)[1]


def _balance_orbit_shares(
    generators: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    pivots: npt.NDArray[np.int_],
    orbits: npt.NDArray[np.int_],
) -> None:
    """Exchange pivots between orbits, in place, while that lowers the largest orbit share.

    ``generators`` is a basis row reduced on the information set ``pivots``, and ``labels`` its
    quotient labels.  Pivoting row ``i`` on another column ``c`` with ``generators[i, c] == 1``
    replaces ``pivots[i]`` by ``c`` in the information set.  Each exchange moves a pivot from an
    orbit ``O`` with the largest share ``|I ∩ O| / |O|`` to an orbit whose share stays below it.

    For two orbits, the resulting largest share is the least of any information set: if no exchange
    moves a pivot out of ``O``, every column of the other orbit lies in the span of the pivots in
    it, so no information set has fewer pivots in ``O``.
    """
    sizes = np.bincount(orbits)
    is_pivot = np.zeros(len(orbits), dtype=bool)
    is_pivot[pivots] = True
    while True:
        counts = np.bincount(orbits[pivots], minlength=len(sizes))
        largest = int(np.argmax(counts / sizes))
        receivers = (counts + 1) * sizes[largest] < counts[largest] * sizes
        rows = np.flatnonzero(orbits[pivots] == largest)
        columns = np.flatnonzero(~is_pivot & receivers[orbits])
        candidates = generators[np.ix_(rows, columns)]
        exchangeable = np.any(candidates, axis=1)
        if not np.any(exchangeable):
            return
        index = int(np.argmax(exchangeable))
        row, column = rows[index], columns[np.argmax(candidates[index])]
        others = np.flatnonzero(generators[:, column])
        others = others[others != row]
        generators[others] ^= generators[row]
        labels[others] ^= labels[row]
        is_pivot[pivots[row]], is_pivot[column] = False, True
        pivots[row] = column
