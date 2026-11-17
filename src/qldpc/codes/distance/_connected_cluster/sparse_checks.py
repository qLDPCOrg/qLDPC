# SPDX-License-Identifier: Apache-2.0

"""Sparser checks with the span of given checks, and light sums of checks to add to them.

``get_sparser_rows`` makes checks sparser for the coset searches of ``coset.py`` and for the
distance searches of subsystem codes in ``qldpc.codes.distance.core`` (Sections 3.8 and 3.9 of
``connected_cluster.md``).  ``get_light_pair_sums`` finds light sums of pairs of rows, which the
distance searches of ``search.py`` add as redundant checks and known words (Section 3.4).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import scipy.sparse

from qldpc._binary import pack_bits, row_reduce_binary, unpack_bits

from .._binary import multiply_binary
from .._cosets import get_unit_syndrome_words

# The seed of the random column orders of ``get_sparser_rows``, which makes its rows reproducible.
_SPARSIFYING_SEED = 0


def get_sparser_rows(
    matrix: npt.NDArray[np.uint8],
    *,
    symplectic: bool,
    max_rounds: int,
    rounds_without_improvement: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Rows with the span of independent rows, made sparser by row reductions in random orders.

    Starting from the rows of ``_sparsify``, each round adds the rows of a row reduction in a random
    order of the columns as candidates, and keeps a lightest basis of the candidates, weighed by
    ``get_row_weights`` (Section 3.8 of ``connected_cluster.md``).  The rounds stop after
    ``max_rounds``, or at the end of a round once ``rounds_without_improvement`` rounds in a row
    have left the total weight unchanged, and ``_sparsify`` makes a last pass.

    Returns:
        The rows, and the invertible transform ``T`` for which the rows are ``T @ matrix % 2``.
    """
    rows, transform = _sparsify(matrix, symplectic=symplectic)
    if not len(rows):
        return rows, transform

    # Each round adds the rows of a reduction that are lighter than the heaviest current row.
    # Row-reducing the transposed candidates, sorted stably by weight, keeps each candidate that is
    # independent of those before it, and since the current rows come first among candidates of
    # equal weight, a round that does not lower the total weight keeps them.
    rng = np.random.default_rng(_SPARSIFYING_SEED)
    weights = get_row_weights(rows, symplectic=symplectic)
    rounds_unchanged = 0
    for _ in range(max_rounds):
        reduced, _, _ = row_reduce_binary(matrix, columns=rng.permutation(matrix.shape[1]))
        reduced_weights = get_row_weights(reduced, symplectic=symplectic)
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

    # Add rows to each other where that helps, and find the transform: ``matrix`` times the
    # transpose of ``get_unit_syndrome_words(matrix)`` is the identity, so the rows times it are T.
    rows, _ = _sparsify(rows, symplectic=symplectic)
    return rows, multiply_binary(rows, get_unit_syndrome_words(matrix).T)


def get_light_pair_sums(
    rows: npt.NDArray[np.uint8], *, symplectic: bool
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Light sums of pairs of rows, whose total weight is at most that of the rows.

    A sum of rows ``i < j`` is light if it is not zero and no heavier than the heavier of the two,
    as ``get_row_weights`` weighs them (Section 3.4 of ``connected_cluster.md``).  The distinct
    light sums that differ from every row are taken lightest first, with ties in the order of their
    pairs, until the next would make their total weight exceed that of the rows.

    Returns:
        The sums, and the pair ``(i, j)`` of rows of each.
    """
    # The sum of rows i and j acts on the variables of either row, less those on which the two
    # rows have equal values: it weighs w_i + w_j - s_ij - e_ij, where s_ij counts the variables
    # that the rows share, and e_ij those with equal values, which are all of them for binary rows.
    # Count s_ij + e_ij by products of indicator matrices.  Rows that share no variable sum to a row
    # heavier than the heavier of the two, so only pairs that share one can give a light sum.
    weights = get_row_weights(rows, symplectic=symplectic)
    if symplectic:
        num_variables = rows.shape[1] // 2
        part_x, part_z = rows[:, :num_variables], rows[:, num_variables:]
        overlaps = _count_overlaps(part_x | part_z)
        for values in (part_x & (1 - part_z), part_z & (1 - part_x), part_x & part_z):
            overlaps = overlaps + _count_overlaps(values)
    else:
        overlaps = 2 * _count_overlaps(rows)
    counts = scipy.sparse.triu(overlaps, k=1).tocoo()
    first, second = counts.row.astype(np.int64), counts.col.astype(np.int64)
    sum_weights = weights[first] + weights[second] - counts.data
    light = (sum_weights > 0) & (sum_weights <= np.maximum(weights[first], weights[second]))
    first, second, sum_weights = first[light], second[light], sum_weights[light]

    # Take the distinct new sums, lightest first, while their total weight stays within that of the
    # rows.
    order = np.lexsort((second, first, sum_weights))
    known = {row.tobytes() for row in rows}
    budget = weights.sum()
    kept = []
    for index in order.tolist():
        if sum_weights[index] > budget:
            break
        row_sum = rows[first[index]] ^ rows[second[index]]
        if row_sum.tobytes() not in known:
            known.add(row_sum.tobytes())
            kept.append(index)
            budget -= sum_weights[index]
    pairs = np.stack([first[kept], second[kept]], axis=1)
    return rows[pairs[:, 0]] ^ rows[pairs[:, 1]], pairs


def get_row_weights(rows: npt.NDArray[np.uint8], *, symplectic: bool) -> npt.NDArray[np.int64]:
    """The numbers of nonzero entries of rows, or of the qubits on which symplectic rows act."""
    if not symplectic:
        return rows.sum(axis=1, dtype=np.int64)
    half = rows.shape[1] // 2
    return (rows[:, :half] | rows[:, half:]).sum(axis=1, dtype=np.int64)


def _count_overlaps(matrix: npt.NDArray[np.uint8]) -> scipy.sparse.csr_array:
    """The number of columns in which both of each pair of rows of a binary matrix are one."""
    sparse = scipy.sparse.csr_array(matrix, dtype=np.int64)
    return sparse @ sparse.T


def _sparsify(
    matrix: npt.NDArray[np.uint8], *, symplectic: bool
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Rows with the span of the given rows, made sparser by adding rows to each other.

    While adding another row to a row lowers its weight, as ``get_row_weights`` counts it, the row
    takes its lightest such sum.  Every addition lowers the total weight, so this ends.

    Returns:
        The rows, and the invertible transform ``T`` for which the rows are ``T @ matrix % 2``.
    """
    # Pack the X and Z parts of symplectic rows, or binary rows with zero Z parts, so that a weight
    # counts the nonzero qubits or entries.
    num_rows, width = matrix.shape
    num_variables = width // 2 if symplectic else width
    part_x = pack_bits(np.ascontiguousarray(matrix[:, :num_variables]))
    part_z = (
        pack_bits(np.ascontiguousarray(matrix[:, num_variables:]))
        if symplectic
        else np.zeros_like(part_x)
    )
    transform = np.eye(num_rows, dtype=np.uint8)
    weights = np.bitwise_count(part_x | part_z).sum(axis=1, dtype=np.int64)

    # Replace each row by its lightest sum with another row, while that lowers its weight.  A row
    # added to itself would vanish, so that sum takes the weight of the row, which never wins.
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
    if symplectic:
        parts.append(unpack_bits(part_z, num_variables))
    return np.hstack(parts), transform
