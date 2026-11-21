# SPDX-License-Identifier: Apache-2.0

"""Qubit permutations that map the X sector of a CSS code onto its Z sector.

A distance search of both sectors of a CSS code needs to search only one of them if such a
permutation exists.  The section "Searching one sector of a CSS code" of ``problems.md`` proves
this.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_independent_rows, multiply_binary
from qldpc.external import _bliss

# How many random words of a row space _may_hold tests.
_NUM_RANDOM_WORDS = 16


def get_sector_map(
    gauge_x: npt.NDArray[np.uint8],
    gauge_z: npt.NDArray[np.uint8],
    basis_x: npt.NDArray[np.uint8],
) -> npt.NDArray[np.int_] | None:
    """A qubit permutation that swaps the row spaces of two binary matrices, or None.

    The permutation maps the row space of ``gauge_x`` onto that of ``gauge_z``, and that of
    ``gauge_z`` onto that of ``gauge_x``.  Entry ``j`` of the result is the image of qubit ``j``.
    ``basis_x`` is the basis of the row space of ``gauge_x`` that ``get_independent_rows`` gives,
    which the caller has already computed, and which a cheap test needs in reduced row echelon
    form.  The result is the identity if the row spaces are equal.  Otherwise, with the optional
    pybliss package, the result maps the nonzero rows of ``gauge_x``, counting repeats, one to one
    onto those of ``gauge_z``, and those of ``gauge_z`` onto those of ``gauge_x``.  It is the first
    of a few fixed permutations that does, or else one that bliss finds as an isomorphism of the
    Tanner graph of the gauge generators (the checks of a stabilizer code).  So, unless the row
    spaces are equal, the result is None if no permutation maps the two lists of nonzero rows onto
    each other in this way, even if some permutation swaps the row spaces.  This happens, for
    example, when only one list has a redundant row.  Every result passes ``maps_sectors``.
    """
    # Swapped row spaces have equal ranks.  Row spaces of equal ranks are equal if one holds the
    # other, which a few random words of the row space of gauge_z test cheaply before the exact
    # test, a rank of the rows of both.
    rank_x = len(basis_x)
    if len(get_independent_rows(gauge_z)) != rank_x:
        return None
    identity = np.arange(gauge_x.shape[1])
    if _may_hold(basis_x, gauge_z) and (
        len(get_independent_rows(np.vstack([gauge_x, gauge_z]))) == rank_x
    ):
        return identity
    if not _bliss.is_installed():
        return None

    # A permutation that maps the two lists of nonzero rows onto each other gives each qubit the
    # X-degree and Z-degree of its image swapped.  Before asking bliss, which would find any such
    # permutation, try a few that do for common codes (see _get_candidate_maps).
    rows_x = gauge_x[np.any(gauge_x, axis=1)].astype(int)
    rows_z = gauge_z[np.any(gauge_z, axis=1)].astype(int)
    degrees = np.column_stack([rows_x.sum(axis=0), rows_z.sum(axis=0)])
    for permutation in _get_candidate_maps(len(degrees)):
        if (
            np.array_equal(degrees[permutation][:, ::-1], degrees)
            and _maps_rows(rows_x, rows_z, permutation)
            and _maps_rows(rows_z, rows_x, permutation)
        ):
            return permutation

    # Such a permutation also maps the X-type checks to Z-type checks of the same weights.  Check
    # these necessary conditions, which are cheap, before asking bliss.
    if not np.array_equal(np.sort(rows_x.sum(axis=1)), np.sort(rows_z.sum(axis=1))) or (
        sorted(map(tuple, degrees.tolist())) != sorted(map(tuple, degrees[:, ::-1].tolist()))
    ):
        return None

    # The X-type checks hold ones and the Z-type checks twos in the first stack, and the other way
    # around in the second.  So an isomorphism of the stacks maps the rows holding ones in the
    # first, its X-type checks, to the rows holding ones in the second, its Z-type checks, and
    # likewise for twos.
    isomorphism = _bliss.get_matrix_isomorphism(
        np.vstack([rows_x, 2 * rows_z]), np.vstack([rows_z, 2 * rows_x])
    )
    if isomorphism is None:
        return None
    num_rows = len(rows_x) + len(rows_z)
    permutation = np.array(isomorphism[num_rows:], dtype=int) - num_rows
    return permutation if maps_sectors(gauge_x, gauge_z, permutation) else None


def maps_sectors(
    gauge_x: npt.NDArray[np.uint8],
    gauge_z: npt.NDArray[np.uint8],
    permutation: npt.NDArray[np.int_],
) -> bool:
    """Whether a qubit permutation swaps the row spaces of two binary matrices.

    The permutation maps qubit ``j`` to qubit ``permutation[j]``.  It swaps the row spaces if it
    maps the row space of ``gauge_x`` onto that of ``gauge_z``, and that of ``gauge_z`` onto that
    of ``gauge_x``, which rank tests over GF(2) check.
    """
    # A word w maps to the word whose entry permutation[j] is w[j], which takes the columns of w in
    # the order of the inverse permutation.  A permutation keeps the rank of a matrix, and row
    # spaces of equal ranks are equal if stacking them keeps the rank.
    inverse = np.argsort(permutation)
    rank = len(get_independent_rows(gauge_x))
    return len(get_independent_rows(gauge_z)) == rank and all(
        len(get_independent_rows(np.vstack([image, target]))) == rank
        for image, target in ((gauge_x[:, inverse], gauge_z), (gauge_z[:, inverse], gauge_x))
    )


def _may_hold(basis: npt.NDArray[np.uint8], rows: npt.NDArray[np.uint8]) -> bool:
    """False only if the row space of a basis does not hold every row, and otherwise True.

    The basis is in reduced row echelon form, as ``get_independent_rows`` gives it, so a word lies
    in its row space exactly if it equals the sum of the basis rows at whose pivots it has ones.
    If some row lies outside, then each random combination of the rows lies outside with
    probability at least one half, so all of ``_NUM_RANDOM_WORDS`` lie inside with probability at
    most ``2**-_NUM_RANDOM_WORDS``.  A fixed seed keeps the result reproducible.
    """
    pivots = np.argmax(basis, axis=1)
    combinations = np.random.default_rng(0).integers(
        2, size=(_NUM_RANDOM_WORDS, len(rows)), dtype=np.uint8
    )
    words = multiply_binary(combinations, rows)
    return np.array_equal(multiply_binary(words[:, pivots], basis), words)


def _get_candidate_maps(num_qubits: int) -> list[npt.NDArray[np.int_]]:
    """Qubit permutations that map the X sectors of some common codes onto their Z sectors.

    They are the reversal of the qubits, which does for bivariate bicycle and generalized bicycle
    codes, the cyclic shift by half of the qubits, which does for some toric codes, and, if the
    qubits fill a square grid row by row, a quarter turn of the grid, which does for rotated surface
    codes of odd distance.  Qubitserf tries the first two too (section "Searching one sector of a
    CSS code" of ``problems.md``).
    """
    qubits = np.arange(num_qubits)
    candidates = [qubits[::-1].copy(), (qubits + num_qubits // 2) % num_qubits]
    side = math.isqrt(num_qubits)
    if side**2 == num_qubits:
        rows, columns = divmod(qubits, side)
        candidates.append(columns * side + side - 1 - rows)
    return candidates


def _maps_rows(
    rows: npt.NDArray[np.int_], images: npt.NDArray[np.int_], permutation: npt.NDArray[np.int_]
) -> bool:
    """Whether a qubit permutation maps binary rows, counting repeats, one to one onto others."""

    def get_sorted_rows(matrix: npt.NDArray[np.int_]) -> npt.NDArray[np.void]:
        """The rows of a binary matrix, packed into byte strings, in sorted order."""
        packed = np.ascontiguousarray(np.packbits(matrix, axis=1))
        return np.sort(packed.view(np.dtype((np.void, packed.shape[1]))).ravel())

    # A word w maps to the word whose entry permutation[j] is w[j] (see maps_sectors).
    return np.array_equal(
        get_sorted_rows(rows[:, np.argsort(permutation)]), get_sorted_rows(images)
    )
