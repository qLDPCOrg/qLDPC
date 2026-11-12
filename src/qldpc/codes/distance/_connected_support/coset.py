# SPDX-License-Identifier: Apache-2.0

"""Connected-support coset searches of ``CosetLeaderSearch``, with checks made sparser first."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits, row_reduce_binary, unpack_bits

from .. import _tuning
from .._cosets import (
    MinWeightBound,
    get_unit_syndrome_words,
    int_to_words,
    row_ints,
    tabulate_singles,
    xor_selected,
)
from .search import SupportSearch

# The seed of the random orders of columns in which connected-support coset searches row-reduce
# their checks to make them sparser (see ``_get_sparser_rows``).  It makes the checks reproducible.
_SPARSIFYING_SEED = 0


class ConnectedSupportCosetSearch:
    """A coset search of a ``CosetLeaderSearch`` by the connected-support search of its checks.

    The words, checks, and syndromes are those of a ``BrouwerZimmermannCosetSearch``
    (``_brouwer_zimmermann/coset.py``): packed Paulis, whose syndrome for the stabilizers is that of
    the symplectic search of ``SupportSearch``, or the X or Z parts of Paulis, whose syndrome is
    that of a binary search of the checks.  The search starts from the syndrome of a query, and
    grows supports through its unsatisfied checks (Section 3.8 of ``connected_support.md``).

    Like a ``BrouwerZimmermannCosetSearch``, it first looks up a word of weight zero or one by its
    syndrome: a single variable, or a single-qubit Pauli.  It searches only cosets with no such
    word, so a coset with such a word gets the weight of that word as a lower bound.  Any other
    coset gets two if it has a word within ``max_weight``.  With increasing weight limits from two
    (see ``get_min_weight_bound``), it instead gets the weight of its lightest words.  Without a
    word within ``max_weight``, it gets the larger of two and ``max_weight + 1``.  These bounds are
    often weaker than those of a ``BrouwerZimmermannCosetSearch``.

    The search is fast only if its checks are sparse, but the given checks, such as row-reduced
    stabilizers, can be dense.  So it searches the sparser checks of ``_get_sparser_rows``, which
    span the same space, and maps the syndrome of a query to them.

    Args:
        matrix: The checks, whose kernel is the code: the stabilizers, in ``[X|Z]`` order, for
            Paulis, or the rows that check the X or Z parts.  Bit ``j`` of the syndrome of a word
            is its symplectic product with row ``j`` for Paulis, and its dot product for parts.
        homogeneous: False for Paulis, and True for X or Z parts.
        max_rounds: The ``max_sparsifying_rounds`` of the tuning (see ``_get_sparser_rows``).
        rounds_without_improvement: The ``sparsifying_rounds_without_improvement`` of the tuning.
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.uint8],
        *,
        homogeneous: bool,
        max_rounds: int,
        rounds_without_improvement: int,
    ) -> None:
        # Map each syndrome of a word of weight zero or one to such a word.  The syndrome of a word
        # with one nonzero bit is a column of the checks, once the halves of symplectic checks are
        # swapped.
        num_variables = matrix.shape[1] if homogeneous else matrix.shape[1] // 2
        syndrome_matrix = matrix if homogeneous else np.roll(matrix, num_variables, axis=1)
        columns = row_ints(np.ascontiguousarray(syndrome_matrix.T))
        self._singles, _, _ = tabulate_singles(columns, num_variables, is_pauli=not homogeneous)

        rows, transform = _get_sparser_rows(
            matrix,
            homogeneous=homogeneous,
            max_rounds=max_rounds,
            rounds_without_improvement=rounds_without_improvement,
        )
        self._num_checks = len(rows)
        # Row ``i`` of the sparse rows sums the given rows that row ``i`` of the transform selects,
        # so bit ``i`` of a syndrome for them is the parity of those bits of the given syndrome.
        # Entry ``j`` of the map is column ``j`` of the transform, which a given syndrome bit adds.
        self._syndrome_map_columns = row_ints(np.ascontiguousarray(transform.T))
        # Without a function that finds automorphisms, the search never asks for them.
        self._search = SupportSearch(
            rows,
            np.zeros((0, matrix.shape[1]), dtype=np.uint8),
            homogeneous=homogeneous,
            automorphism_minimum_states=_tuning.MAX_VALUE,
        )
        # Word ``j`` of ``get_unit_syndrome_words`` of the sparser rows has a syndrome of only bit
        # ``j``, so the words of the set bits of a syndrome sum to a representative of its coset.
        syndrome_rows = rows if homogeneous else np.roll(rows, num_variables, axis=1)
        self._inverse_rows = row_ints(get_unit_syndrome_words(syndrome_rows))

    def get_min_weight_bound(
        self,
        syndrome: int,
        max_weight: int,
        *,
        first_limit: int | None = None,
    ) -> MinWeightBound:
        """Decide whether the coset of this syndrome has a word of weight at most ``max_weight``.

        A negative ``max_weight`` admits no word, as for a ``BrouwerZimmermannCosetSearch``.

        A search with weight limit ``max_weight`` can visit many heavy supports before it finds a
        word that is much lighter.  Given ``first_limit``, the search instead tries every limit from
        ``first_limit`` to ``max_weight``, and stops at the first limit within which it finds a
        word.  It tries no limit below two, which the lookup of words of weight at most one makes
        useless, and lowers a first limit above ``max_weight`` to it.  From the first limit two, it
        finds a lightest word.  This can save time on cosets with words much lighter than
        ``max_weight``, and repeats the work of the lower limits on cosets without a word within
        ``max_weight``.  A first limit above two suits callers to which any word within it gives the
        same answer.  A word that the search finds at that limit can be lighter than the limit, so
        its coset gets the lower bound two.
        """
        # Look up a word of weight zero or one.
        word = self._singles.get(syndrome)
        if word is not None:
            weight = int(word != 0)
            return MinWeightBound(weight <= max_weight, word, weight)

        # Map the syndrome to the sparse checks.  The map is invertible, and the lookup above
        # answers the zero syndrome, so the mapped syndrome is not zero.  The sparse checks are
        # independent, so the search keeps all of them, in order, and the mapped syndrome has one
        # bit for each check that it keeps.  So the search can start from the packed syndrome,
        # without the validation of ``find_vector_with_syndrome``, which would cost more than the
        # search itself in many queries.
        syndrome = xor_selected(self._syndrome_map_columns, syndrome)
        packed_syndrome = int_to_words(syndrome, (self._num_checks + 63) // 64)

        # The lookup leaves only cosets without a word below weight two, so no limit below two is
        # searched, and a single query searches only at ``max_weight``.  Each limit within which the
        # search finds no word raises the lower bound above that limit.
        lower_bound = 2
        first = max_weight if first_limit is None else min(first_limit, max_weight)
        for limit in range(max(2, first), max_weight + 1):
            symbols = self._search._find_vector_with_packed_syndrome(packed_syndrome, limit)
            if symbols is not None:
                return MinWeightBound(True, self._get_word(symbols), lower_bound)
            lower_bound = limit + 1
        representative = xor_selected(self._inverse_rows, syndrome)
        return MinWeightBound(False, representative, lower_bound)

    def _get_word(self, symbols: npt.NDArray[np.int64]) -> int:
        """Pack the symbols of a support into a word.

        A symbol gives the qubit ``symbol % n`` the Pauli X, Z, or Y for values 0, 1, and 2.  The
        symbols of an X or Z part have only the value 0.
        """
        num_variables = self._search.num_variables
        word = 0
        for symbol in symbols.tolist():
            value, variable = divmod(symbol, num_variables)
            if value != 1:
                word |= 1 << variable
            if value:
                word |= 1 << (num_variables + variable)
        return word


def _get_sparser_rows(
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
    # of ``matrix`` is the identity, so their product with it is the transform.  Products of binary
    # matrices in float32 are exact while their inner dimension is below ``2**24``, and faster than
    # in integers.
    rows, _ = _sparsify(rows, homogeneous=homogeneous)
    inverse = get_unit_syndrome_words(matrix)
    product = rows.astype(np.float32) @ inverse.T.astype(np.float32)
    return rows, (product % 2).astype(np.uint8)


def _sparsify(
    matrix: npt.NDArray[np.uint8], *, homogeneous: bool
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Rows with the span of the given rows, made sparser by adding rows to each other.

    The weight of a row is its number of nonzero entries, or, if the rows are symplectic, the
    number of qubits on which its X part ``[:n]`` or its Z part ``[n:]`` is nonzero.  While adding
    another row to a row lowers its weight, the row takes the sum that is lightest.  Every addition
    lowers the total weight, so this ends.

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
