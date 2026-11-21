# SPDX-License-Identifier: Apache-2.0

"""Sparser checks with the span of given checks, and light sums of checks to add to them.

``get_sparser_rows`` makes checks sparser for the coset searches of ``coset.py`` and for the
distance searches of subsystem codes in ``qldpc.codes.distance.distances`` (Sections 3.8 and 3.9 of
``connected_cluster.md``).  ``get_light_pair_sums`` finds light sums of pairs of rows, which
``SupportSearch`` of ``search.py`` adds as redundant checks, and, in distance searches, as known
words (Section 3.4).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qldpc._binary import (
    multiply_binary,
    pack_bits,
    popcount_word,
    row_reduce_packed,
    trailing_zeros_word,
    unpack_bits,
)
from qldpc._compilation import always_compiled

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

    # Each round row-reduces the matrix in a random order of its columns (see _take_lighter_rows).
    # The rounds work on rows packed as _pack_parts packs them, so a column of the matrix is a bit
    # at its position in the packed rows.
    rng = np.random.default_rng(_SPARSIFYING_SEED)
    packed_matrix, positions = _pack_parts(matrix, symplectic=symplectic)
    packed_rows, _ = _pack_parts(rows, symplectic=symplectic)
    weights = get_row_weights(rows, symplectic=symplectic)
    rounds_unchanged = 0
    for _ in range(max_rounds):
        previous_total = weights.sum()
        columns = positions[rng.permutation(matrix.shape[1])]
        packed_rows, weights = _take_lighter_rows(
            packed_matrix, columns, packed_rows, weights, symplectic
        )
        rounds_unchanged = rounds_unchanged + 1 if weights.sum() == previous_total else 0
        if rounds_unchanged >= rounds_without_improvement:
            break

    # Add rows to each other where that helps, and find the transform: ``matrix`` times the
    # transpose of ``get_unit_syndrome_words(matrix)`` is the identity, so the rows times it are T.
    rows = _unpack_parts(packed_rows, matrix.shape[1], symplectic=symplectic)
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
    # Rows that share no variable sum to a row heavier than the heavier of the two, so only pairs
    # that share one can give a light sum.  List the nonzero entries of the rows by row and by
    # variable, with the values 1, 2, and 3 for X, Z, and Y in symplectic rows, and count
    # s_ij + e_ij from these lists.
    weights = get_row_weights(rows, symplectic=symplectic)
    values = rows
    if symplectic:
        num_variables = rows.shape[1] // 2
        values = rows[:, :num_variables] + 2 * rows[:, num_variables:]
    entry_rows, entry_variables = np.nonzero(values)
    entry_values = values[entry_rows, entry_variables]
    by_variable = np.argsort(entry_variables, kind="stable")
    first, second, counts = _count_shared_entries(
        np.searchsorted(entry_rows, np.arange(len(rows) + 1)),
        entry_variables.astype(np.int64),
        entry_values.astype(np.int64),
        np.searchsorted(entry_variables[by_variable], np.arange(values.shape[1] + 1)),
        entry_rows[by_variable].astype(np.int64),
        entry_values[by_variable].astype(np.int64),
    )
    sum_weights = weights[first] + weights[second] - counts
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


@always_compiled
def _count_shared_entries(
    row_starts: npt.NDArray[np.int64],
    row_variables: npt.NDArray[np.int64],
    row_values: npt.NDArray[np.int64],
    variable_starts: npt.NDArray[np.int64],
    variable_rows: npt.NDArray[np.int64],
    variable_values: npt.NDArray[np.int64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Count, for the rows ``i < j`` of each pair that share a variable, ``s_ij + e_ij``.

    Here ``s_ij`` counts the variables on which both rows are nonzero, and ``e_ij`` those on which
    they have equal nonzero values.  Row ``i`` has the values ``row_values[k]`` on the variables
    ``row_variables[k]``, for ``k`` from ``row_starts[i]`` to ``row_starts[i + 1]``, and variable
    ``v`` lists its rows and values in the same way, from ``variable_starts``.

    Returns:
        The rows ``i`` and ``j`` of each such pair, and its count.
    """
    # Bound the number of pairs by the pairs of rows on each variable, and by the pairs of all rows.
    num_rows = len(row_starts) - 1
    max_pairs = 0
    for variable in range(len(variable_starts) - 1):
        degree = variable_starts[variable + 1] - variable_starts[variable]
        max_pairs += degree * (degree - 1) // 2
    max_pairs = min(max_pairs, num_rows * (num_rows - 1) // 2)
    first = np.empty(max_pairs, dtype=np.int64)
    second = np.empty(max_pairs, dtype=np.int64)
    counts = np.empty(max_pairs, dtype=np.int64)

    # Accumulate the counts of each row with the later rows that it meets, then list them.
    totals = np.zeros(num_rows, dtype=np.int64)
    met = np.empty(num_rows, dtype=np.int64)
    num_pairs = 0
    for row in range(num_rows):
        num_met = 0
        for entry in range(row_starts[row], row_starts[row + 1]):
            variable = row_variables[entry]
            for other_entry in range(variable_starts[variable], variable_starts[variable + 1]):
                other = variable_rows[other_entry]
                if other > row:
                    if totals[other] == 0:
                        met[num_met] = other
                        num_met += 1
                    totals[other] += 1 + (variable_values[other_entry] == row_values[entry])
        for index in range(num_met):
            other = met[index]
            first[num_pairs], second[num_pairs], counts[num_pairs] = row, other, totals[other]
            num_pairs += 1
            totals[other] = 0
    return first[:num_pairs], second[:num_pairs], counts[:num_pairs]


def _sparsify(
    matrix: npt.NDArray[np.uint8], *, symplectic: bool
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Rows with the span of the given rows, made sparser by adding rows to each other.

    While adding another row to a row lowers its weight, as ``get_row_weights`` counts it, the row
    takes its lightest such sum.  Every addition lowers the total weight, so this ends.

    Returns:
        The rows, and the invertible transform ``T`` for which the rows are ``T @ matrix % 2``.
    """
    rows, _ = _pack_parts(matrix, symplectic=symplectic)
    transform = np.eye(len(matrix), dtype=np.uint8)
    _add_rows_while_lighter(rows, transform, symplectic)
    return _unpack_parts(rows, matrix.shape[1], symplectic=symplectic), transform


def _pack_parts(
    matrix: npt.NDArray[np.uint8], *, symplectic: bool
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """Pack binary rows, or the X and Z parts of symplectic rows one after the other.

    Each part is packed as ``pack_bits`` packs a row, so that word ``k`` of the X part and word
    ``k`` of the Z part hold the same qubits.

    Returns:
        The packed rows, and the position of the bit of each column of the matrix in them.
    """
    if not symplectic:
        return pack_bits(np.ascontiguousarray(matrix)), np.arange(matrix.shape[1], dtype=np.int64)
    num_variables = matrix.shape[1] // 2
    part_x = pack_bits(np.ascontiguousarray(matrix[:, :num_variables]))
    part_z = pack_bits(np.ascontiguousarray(matrix[:, num_variables:]))
    variables = np.arange(num_variables, dtype=np.int64)
    positions = np.concatenate([variables, 64 * part_x.shape[1] + variables])
    return np.hstack([part_x, part_z]), positions


def _unpack_parts(
    rows: npt.NDArray[np.uint64], width: int, *, symplectic: bool
) -> npt.NDArray[np.uint8]:
    """Unpack rows of ``width`` columns that ``_pack_parts`` packed."""
    if not symplectic:
        return unpack_bits(rows, width)
    num_words = rows.shape[1] // 2
    num_variables = width // 2
    return np.hstack(
        [
            unpack_bits(np.ascontiguousarray(rows[:, :num_words]), num_variables),
            unpack_bits(np.ascontiguousarray(rows[:, num_words:]), num_variables),
        ]
    )


@always_compiled
def _get_packed_weights(rows: npt.NDArray[np.uint64], symplectic: bool) -> npt.NDArray[np.int64]:
    """The weights of rows packed by ``_pack_parts``, as ``get_row_weights`` counts them."""
    num_words = rows.shape[1] // 2 if symplectic else rows.shape[1]
    weights = np.zeros(len(rows), dtype=np.int64)
    for row in range(len(rows)):
        for word in range(num_words):
            value = rows[row, word]
            if symplectic:
                value |= rows[row, num_words + word]
            weights[row] += popcount_word(value)
    return weights


@always_compiled
def _add_rows_while_lighter(
    rows: npt.NDArray[np.uint64], transform: npt.NDArray[np.uint8], symplectic: bool
) -> None:
    """Make rows packed by ``_pack_parts`` sparser in place, as ``_sparsify`` describes.

    Each row added to a row is also added to the same row of ``transform``.
    """
    num_rows, num_words = rows.shape
    part_words = num_words // 2 if symplectic else num_words
    weights = _get_packed_weights(rows, symplectic)

    # Replace each row by its first lightest sum with another row, while that lowers its weight.
    improved = True
    while improved:
        improved = False
        for row in range(num_rows):
            best_weight, best_other = weights[row], -1
            for other in range(num_rows):
                if other == row:
                    continue
                weight = np.int64(0)
                for word in range(part_words):
                    value = rows[row, word] ^ rows[other, word]
                    if symplectic:
                        value |= rows[row, part_words + word] ^ rows[other, part_words + word]
                    weight += popcount_word(value)
                if weight < best_weight:
                    best_weight, best_other = weight, other
            if best_other >= 0:
                rows[row] ^= rows[best_other]
                transform[row] ^= transform[best_other]
                weights[row] = best_weight
                improved = True


@always_compiled
def _take_lighter_rows(
    matrix: npt.NDArray[np.uint64],
    columns: npt.NDArray[np.int64],
    rows: npt.NDArray[np.uint64],
    weights: npt.NDArray[np.int64],
    symplectic: bool,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """Run one round of ``get_sparser_rows`` on independent rows, packed by ``_pack_parts``.

    The round row-reduces ``matrix``, whose rows span the same space, over ``columns``, which give
    the positions of the bits of its columns in a random order.  It sorts the current rows and the
    rows of the reduction lighter than the heaviest current row stably by weight, the current rows
    first, and keeps each that is independent of those before it, so a round that does not lower
    the total weight keeps the current rows.

    Returns:
        The rows that the round keeps, and their weights.
    """
    reduced, _, _ = row_reduce_packed(matrix, np.zeros((len(matrix), 0), dtype=np.uint64), columns)
    reduced_weights = _get_packed_weights(reduced, symplectic)
    light = np.flatnonzero(reduced_weights < weights.max())
    if not len(light):
        return rows, weights

    # Sort the candidates by weight, with ties in their order, and keep the independent ones.
    candidates = np.vstack((rows, reduced[light]))
    candidate_weights = np.concatenate((weights, reduced_weights[light]))
    num_candidates = len(candidates)
    order = np.argsort(candidate_weights * num_candidates + np.arange(num_candidates))
    kept = order[_get_leading_independent_rows(candidates[order], len(rows))]
    return candidates[kept], candidate_weights[kept]


@always_compiled
def _get_leading_independent_rows(rows: npt.NDArray[np.uint64], rank: int) -> npt.NDArray[np.int64]:
    """The indices of the packed rows that are independent of the rows before them.

    The rows span a space of dimension ``rank``, so the indices end at the ``rank``-th such row.
    """
    # Reduce each row by the kept rows in order.  Each kept row has a pivot, its lowest set bit, on
    # which the rows kept after it are zero, so the reduction clears every pivot, and leaves a row
    # that vanishes exactly if it depends on the kept rows.
    basis = np.empty((rank, rows.shape[1]), dtype=np.uint64)
    pivot_words = np.empty(rank, dtype=np.int64)
    pivot_bits = np.empty(rank, dtype=np.uint64)
    kept = np.empty(rank, dtype=np.int64)
    num_kept = 0
    for index in range(len(rows)):
        if num_kept == rank:
            break
        row = rows[index].copy()
        for basis_index in range(num_kept):
            if row[pivot_words[basis_index]] & pivot_bits[basis_index]:
                row ^= basis[basis_index]
        for word in range(len(row)):
            if row[word]:
                basis[num_kept] = row
                pivot_words[num_kept] = word
                pivot_bits[num_kept] = np.uint64(1) << np.uint64(trailing_zeros_word(row[word]))
                kept[num_kept] = index
                num_kept += 1
                break
    return kept[:num_kept]
