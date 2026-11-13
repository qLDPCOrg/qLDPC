# SPDX-License-Identifier: Apache-2.0

"""Sparser checks with the span of given checks, for the connected-support searches.

The searches are fast only if their checks are sparse.  ``get_sparser_rows`` makes checks sparser
for the coset searches of ``coset.py`` and for the distance searches of subsystem codes in
``qldpc.codes.distance.core`` (Sections 3.8 and 3.9 of ``connected_support.md``).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits, row_reduce_binary, unpack_bits

from .._binary import multiply_binary
from .._cosets import get_unit_syndrome_words

# The seed of the random orders of columns in which the connected-support searches row-reduce their
# checks to make them sparser (see ``get_sparser_rows``).  It makes the checks reproducible.
_SPARSIFYING_SEED = 0


def get_sparser_rows(
    matrix: npt.NDArray[np.uint8],
    *,
    homogeneous: bool,
    max_rounds: int,
    rounds_without_improvement: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Rows with the span of independent rows, made sparser by row reductions in random orders.

    The rows start as those of ``_sparsify``, and are weighed as it weighs them.  Each round
    row-reduces the given rows with pivots in a random order of the columns, which gives the words
    of the span that have exactly one pivot column in their support.  Those that are lighter than
    the heaviest current row join the rows as candidates, and the greedy algorithm for matroids
    keeps the lightest independent candidates: it takes them by increasing weight, and keeps each
    that is independent of those that it kept.  So the rows are always a lightest basis drawn from
    the candidates, and their total weight never rises.  The rounds stop after ``max_rounds``, or
    once ``rounds_without_improvement`` rounds in a row have left the total weight unchanged, and a
    last pass of ``_sparsify`` adds rows to each other where that helps.  The rows need not be the
    sparsest.  Rows with the same span give a search the same answers and bounds, though perhaps
    other representatives.  So the rounds may be random, and a fixed seed makes them reproducible.

    Args:
        matrix: Independent rows.
        homogeneous: False if the rows are symplectic, and True otherwise.
        max_rounds: The most rounds, or zero for the rows of ``_sparsify``.
        rounds_without_improvement: The number of rounds in a row without a lower total weight after
            which the rounds stop.

    Returns:
        The rows, and the invertible transform ``T``, so that the rows are ``T @ matrix % 2``.  The
        total weight of the rows is at most that of the rows of ``_sparsify``.
    """
    rows, transform = _sparsify(matrix, homogeneous=homogeneous)
    if not len(rows):
        return rows, transform

    # Each round adds the rows of a reduction that are lighter than the heaviest current row.  The
    # rows come first among candidates of equal weight, so a round that does not lower the total
    # weight leaves the rows as they are.  Row-reducing the transposed candidates, sorted by weight,
    # takes as pivots the first candidates that are independent of those before them.
    rng = np.random.default_rng(_SPARSIFYING_SEED)
    weights = _get_row_weights(rows, homogeneous=homogeneous)
    rounds_unchanged = 0
    for _ in range(max_rounds):
        reduced, _, _ = row_reduce_binary(matrix, columns=rng.permutation(matrix.shape[1]))
        reduced_weights = _get_row_weights(reduced, homogeneous=homogeneous)
        light = reduced_weights < weights.max()
        previous_total = weights.sum()
        if np.any(light):
            candidates = np.vstack([rows, reduced[light]])
            candidate_weights = np.concatenate([weights, reduced_weights[light]])
            order = np.argsort(candidate_weights, kind="stable")
            _, _, pivots = row_reduce_binary(np.ascontiguousarray(candidates[order].T))
            rows, weights = candidates[order[pivots]], candidate_weights[order[pivots]]
        rounds_unchanged = rounds_unchanged + 1 if weights.sum() == previous_total else 0
        if rounds_unchanged >= rounds_without_improvement:
            break

    # Add rows to each other where that helps, and find the transform from the given rows.  The rows
    # are in the span of ``matrix``, whose product with the transpose of ``get_unit_syndrome_words``
    # of ``matrix`` is the identity, so their product with it is the transform.
    rows, _ = _sparsify(rows, homogeneous=homogeneous)
    return rows, multiply_binary(rows, get_unit_syndrome_words(matrix).T)


def _sparsify(
    matrix: npt.NDArray[np.uint8], *, homogeneous: bool
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Rows with the span of the given rows, made sparser by adding rows to each other.

    The weight of a row is its number of nonzero entries, or, if the rows are symplectic, the number
    of qubits on which its X part ``[:n]`` or its Z part ``[n:]`` is nonzero.  While adding another
    row to a row lowers its weight, the row takes the sum that is lightest.  Every addition lowers
    the total weight, so this ends.

    Args:
        matrix: The rows.
        homogeneous: False if the rows are symplectic, and True otherwise.

    Returns:
        The rows, and the invertible transform ``T``, so that the rows are ``T @ matrix % 2``.
    """
    # Pack the X and Z parts, or the rows and zeros, so that a weight counts the nonzero qubits.
    num_rows, width = matrix.shape
    num_variables = width if homogeneous else width // 2
    part_x = pack_bits(np.ascontiguousarray(matrix[:, :num_variables]))
    part_z = (
        np.zeros_like(part_x)
        if homogeneous
        else pack_bits(np.ascontiguousarray(matrix[:, num_variables:]))
    )
    transform = np.eye(num_rows, dtype=np.uint8)
    weights = np.bitwise_count(part_x | part_z).sum(axis=1, dtype=np.int64)

    # Replace each row by its lightest sum with another row, while that lowers its weight.  A row
    # added to itself would vanish, so its own entry keeps its weight, which never wins.
    improved = True
    while improved:
        improved = False
        for row in range(num_rows):
            sums_x, sums_z = part_x ^ part_x[row], part_z ^ part_z[row]
            sum_weights = np.bitwise_count(sums_x | sums_z).sum(axis=1, dtype=np.int64)
            sum_weights[row] = weights[row]
            other = int(np.argmin(sum_weights))
            if sum_weights[other] < weights[row]:
                part_x[row], part_z[row] = sums_x[other], sums_z[other]
                transform[row] ^= transform[other]
                weights[row] = sum_weights[other]
                improved = True

    parts = [unpack_bits(part_x, num_variables)]
    if not homogeneous:
        parts.append(unpack_bits(part_z, num_variables))
    return np.hstack(parts), transform


def _get_row_weights(rows: npt.NDArray[np.uint8], *, homogeneous: bool) -> npt.NDArray[np.int64]:
    """The weights of rows, as ``_sparsify`` counts them: of nonzero entries, or of qubits."""
    if homogeneous:
        return rows.sum(axis=1, dtype=np.int64)
    half = rows.shape[1] // 2
    return (rows[:, :half] | rows[:, half:]).sum(axis=1, dtype=np.int64)
