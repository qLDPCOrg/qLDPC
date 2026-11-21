# SPDX-License-Identifier: Apache-2.0

"""Connected-cluster coset searches of ``CosetLeaderSearch``, with checks made sparser first.

The module also holds the batch queries of ``CosetLeaderSearch`` by its one or two connected-cluster
coset searches.  Several threads can run the queries of a coset search at once, as Section 3.8 of
``connected_cluster.md`` says, and a change must keep the requirements on threads of its Section 5.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_num_words, int_to_words, row_ints, words_to_int

from .. import _threads, _tuning
from .._code_matrices import swap_pauli_halves
from .._cosets import get_unit_syndrome_words, tabulate_singles, xor_selected
from ..min_weight_bounds import MinWeightBound
from .search import SupportSearch
from .sparse_checks import get_sparser_rows

# Maps the syndrome of a ``CosetLeaderSearch`` query to the syndromes of the X and Z parts of the
# errors in its class.
_SplitSyndrome = Callable[[int], tuple[int, int]]

# Joins the bounds of the cosets of X parts and of Z parts into a bound of a syndrome class.
_JoinPartBounds = Callable[[MinWeightBound, MinWeightBound], MinWeightBound]

####################################################################################################
# coset searches


class ConnectedClusterCosetSearch:
    """A coset search of a ``CosetLeaderSearch`` by the connected-cluster search of its checks.

    The words and syndromes are those of a ``BrouwerZimmermannCosetSearch``: packed Paulis, or X or
    Z parts of Paulis.  The coset search makes its checks sparser, with the same span, and searches
    from the syndrome of each query, mapped to them (Section 3.8 of ``connected_cluster.md``).
    ``get_min_weight_bound``, and the batch queries of this module through
    ``_get_min_weight_bound_and_states``, give these lower bounds on the weight of the words of a
    coset:

    - The weight of the word, for a coset with a word of weight zero or one, which a table finds.
    - Two, for any other coset whose word the search finds within its first weight limit
      (``max_weight`` for ``get_min_weight_bound``), since the word can be lighter than that limit.
    - The weight limit, for a coset whose word the search finds within a later limit, which is then
      the weight of the lightest words.
    - The larger of two and ``max_weight + 1``, for any other coset without a word within
      ``max_weight``.

    So with a first limit of two, every coset with a word within ``max_weight`` gets the weight of
    its lightest words.

    Args:
        matrix: Independent checks, whose kernel is the code: the stabilizers, in ``[X|Z]`` order,
            for Paulis, or the rows that check the X or Z parts.  Bit ``j`` of the syndrome of a
            word is its symplectic product with row ``j`` for Paulis, and its dot product for parts.
        symplectic: True for Paulis, and False for X or Z parts.
        max_rounds: The field ``max_coset_sparsifying_rounds`` of ``SearchTuning`` (see
            ``get_sparser_rows``).
        rounds_without_improvement: The field ``sparsifying_rounds_without_improvement`` of
            ``SearchTuning``.
        restrict_viable_symbols: The field ``restrict_viable_symbols`` of ``SearchTuning`` (see
            ``SupportSearch``).
        stabilizer_overlap_cuts: The field ``stabilizer_overlap_cuts`` of ``SearchTuning`` (see
            ``SupportSearch``).  The cuts need known words of the kernel of the checks, which
            ``use_query_overlap_words`` gives.
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.uint8],
        *,
        symplectic: bool,
        max_rounds: int,
        rounds_without_improvement: int,
        restrict_viable_symbols: bool,
        stabilizer_overlap_cuts: bool,
    ) -> None:
        # Tabulate the words of weight zero or one by syndrome.  The syndrome of a single bit is a
        # column of the checks, with the halves of symplectic checks swapped.
        num_variables = matrix.shape[1] // 2 if symplectic else matrix.shape[1]
        syndrome_matrix = swap_pauli_halves(matrix) if symplectic else matrix
        columns = row_ints(np.ascontiguousarray(syndrome_matrix.T))
        self._singles, _, _ = tabulate_singles(columns, num_variables, symplectic=symplectic)

        # Make the checks sparser, with the same span (Section 3.8 of connected_cluster.md).
        rows, transform = get_sparser_rows(
            matrix,
            symplectic=symplectic,
            max_rounds=max_rounds,
            rounds_without_improvement=rounds_without_improvement,
        )
        self._rows = rows
        # Row ``i`` of ``rows`` sums the given rows that row ``i`` of ``transform`` selects, so a
        # given syndrome maps to the sum of the columns of ``transform`` that its bits select.
        self._syndrome_map_columns = row_ints(np.ascontiguousarray(transform.T))
        # Without a function that finds automorphisms, the search never asks for them.
        self._search = SupportSearch(
            rows,
            np.zeros((0, matrix.shape[1]), dtype=np.uint8),
            symplectic=symplectic,
            automorphism_minimum_states=_tuning.MAX_VALUE,
            restrict_viable_symbols=restrict_viable_symbols,
            stabilizer_overlap_cuts=stabilizer_overlap_cuts,
            # Sums of pairs of checks made queries slower (Section 3.8 of connected_cluster.md).
            pair_sum_checks=False,
        )
        # Tabulate the cuts of queries now, so that queries only read the search.
        self._search.tabulate_query_cuts()
        # For ``rows``, word ``j`` of ``get_unit_syndrome_words`` has a syndrome of only bit ``j``,
        # so the words of the set bits of a syndrome sum to a representative of its coset.
        syndrome_rows = swap_pauli_halves(rows) if symplectic else rows
        self._inverse_rows = row_ints(get_unit_syndrome_words(syndrome_rows))

    @property
    def checks(self) -> npt.NDArray[np.uint8]:
        """The sparser checks that the search uses, which span the same space as the given ones.

        The array is a read-only view.
        """
        rows = self._rows.view()
        rows.flags.writeable = False
        return rows

    def use_query_overlap_words(self, words: npt.ArrayLike) -> None:
        """Let the overlap cuts of the queries use words of the kernel of the checks.

        Such words are stabilizers, for example.  The cuts do nothing until this method gives them
        words.  The words are in the representation of the checks (see
        ``SupportSearch.use_query_overlap_words``).  Call it only before any query runs, as the
        constructor of ``CosetLeaderSearch`` does, since queries on other threads read the cuts.
        """
        self._search.use_query_overlap_words(words)

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether the coset of this syndrome has a word of weight at most ``max_weight``.

        The search tries only the weight limit ``max_weight``, if it is at least two (Section 3.8 of
        ``connected_cluster.md``).  The class docstring states the lower bounds of the answers.
        """
        return self._get_min_weight_bound_and_states(syndrome, max_weight, None)[0]

    def _get_min_weight_bound_and_states(
        self, syndrome: int, max_weight: int, first_limit: int | None
    ) -> tuple[MinWeightBound, int]:
        """Run ``get_min_weight_bound`` from a first weight limit, and count the states it visits.

        The search tries each weight limit from ``max(min(first_limit, max_weight), 2)`` through
        ``max_weight``, and stops at the first within which it finds a word (Section 3.8 of
        ``connected_cluster.md``).  Without ``first_limit``, it tries only ``max_weight``, if that
        is at least two.  A first limit above two suits callers to which any word within it gives
        the same answer.  The class docstring states the lower bounds of the answers.
        """
        # Look up a word of weight zero or one.
        word = self._singles.get(syndrome)
        if word is not None:
            weight = int(word != 0)
            return MinWeightBound(weight <= max_weight, word, weight), 0

        # Map the syndrome to the sparser checks, which are independent, so the search keeps them
        # all, in order.  The map is invertible and the lookup answers the zero syndrome, so the
        # mapped syndrome is not zero, as ``find_vector_with_packed_syndrome`` requires.
        syndrome = xor_selected(self._syndrome_map_columns, syndrome)
        packed_syndrome = int_to_words(syndrome, get_num_words(len(self._rows)))

        # The lookup leaves only cosets without a word below weight two, so search no limit below
        # two, and raise the lower bound above each limit within which the search finds no word.
        lower_bound = 2
        num_states = 0
        first = max_weight if first_limit is None else min(first_limit, max_weight)
        for limit in range(max(2, first), max_weight + 1):
            symbols, states = self._search.find_vector_with_packed_syndrome(packed_syndrome, limit)
            num_states += states
            if symbols is not None:
                return MinWeightBound(True, self._get_word(symbols), lower_bound), num_states
            lower_bound = limit + 1
        representative = xor_selected(self._inverse_rows, syndrome)
        return MinWeightBound(False, representative, lower_bound), num_states

    def _get_word(self, symbols: npt.NDArray[np.int64]) -> int:
        """Pack symbols into a word, in which the values 0, 1, and 2 of a symbol are X, Z, and Y.

        The symbols are those of ``SupportSearch``.  Those of an X or Z part all have the value 0.
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


####################################################################################################
# batch queries


def find_first_cluster_bound(
    searches: tuple[ConnectedClusterCosetSearch, ...],
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    split_syndrome: _SplitSyndrome,
    join_part_bounds: _JoinPartBounds,
    num_threads: int,
    split_work: int,
) -> tuple[int, MinWeightBound | None]:
    """Run ``CosetLeaderSearch.find_first_not_within_max_weight`` with connected-cluster searches.

    ``searches`` holds the search of Pauli errors, or the searches of X parts and of Z parts, whose
    syndromes ``split_syndrome`` gives, and whose bounds ``join_part_bounds`` joins.
    ``_threads.run_blocks`` moves the rest of a costly batch to ``num_threads`` threads once its
    work reaches ``split_work``, and returns the least index of a syndrome whose class has no error
    within ``max_weight``, which, with its bound, is that of one thread.  The work of a query is the
    number of states that its searches visit (see ``SearchTuning.search_chunk_exponent``).  A query
    that looks up its error and visits no state does none, since it spends its time holding Python's
    global interpreter lock.  Threads run ``run_rows`` at once, so the code that it calls must keep
    the requirements on threads in Section 5 of ``connected_cluster.md``.
    """
    # The bound of each syndrome that ends a call of ``run_rows``.
    found: dict[int, MinWeightBound] = {}

    def run_rows(start: int, end: int, max_work: int) -> tuple[int, bool]:
        """Answer the queries of rows in turn, up to the first without an error in reach."""
        work = 0
        for index in range(start, end):
            bound, states = _get_batch_cluster_bound(
                searches,
                words_to_int(syndromes[index]),
                max_weight,
                split_syndrome,
                join_part_bounds,
            )
            if not bound.within_max_weight:
                found[index] = bound
                return index + 1, True
            work += states
            if work >= max_work:
                return index + 1, False
        return end, False

    index = _threads.run_blocks(len(syndromes), num_threads, split_work, lambda: run_rows)
    return index, found.get(index)


def get_min_cluster_weights(
    searches: tuple[ConnectedClusterCosetSearch, ...],
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    split_syndrome: _SplitSyndrome,
    num_threads: int,
    split_work: int,
) -> npt.NDArray[np.int64]:
    """Run ``CosetLeaderSearch.get_min_weights`` with connected-cluster searches.

    The searches, the split of syndromes, the threads, the work, and the requirements on threads are
    as in ``find_first_cluster_bound``.  Each weight depends only on its syndrome, so the weights
    are those of one thread.
    """
    weights = np.empty(len(syndromes), dtype=np.int64)

    def run_rows(start: int, end: int, max_work: int) -> tuple[int, bool]:
        """Write the weights of the syndromes of rows in turn."""
        work = 0
        for row in range(start, end):
            weights[row], states = _get_min_cluster_weight(
                searches, words_to_int(syndromes[row]), max_weight, split_syndrome
            )
            work += states
            if work >= max_work:
                return row + 1, False
        return end, False

    _threads.run_blocks(len(syndromes), num_threads, split_work, lambda: run_rows)
    return weights


def _get_min_cluster_weight(
    searches: tuple[ConnectedClusterCosetSearch, ...],
    syndrome: int,
    max_weight: int,
    split_syndrome: _SplitSyndrome,
) -> tuple[int, int]:
    """Find the minimum weight of the class of a syndrome with connected-cluster searches.

    The searches try increasing weight limits from two.  A Z-part search starts instead at the
    weight of the X part, if that is more than two, since any Z part within it gives the class that
    weight (Section 3.8 of ``connected_cluster.md``).

    Returns:
        The minimum weight of the class, or ``max_weight + 1`` if that is larger, and the number of
        states that the searches visited.
    """
    if len(searches) == 1:
        bound, states = searches[0]._get_min_weight_bound_and_states(syndrome, max_weight, 2)
        return (bound.lower_bound if bound.within_max_weight else max_weight + 1), states
    x_syndrome, z_syndrome = split_syndrome(syndrome)
    x_bound, x_states = searches[0]._get_min_weight_bound_and_states(x_syndrome, max_weight, 2)
    if not x_bound.within_max_weight:
        return max_weight + 1, x_states
    z_bound, z_states = searches[1]._get_min_weight_bound_and_states(
        z_syndrome, max_weight, x_bound.lower_bound
    )
    if not z_bound.within_max_weight:
        return max_weight + 1, x_states + z_states
    return max(x_bound.lower_bound, z_bound.lower_bound), x_states + z_states


def _get_batch_cluster_bound(
    searches: tuple[ConnectedClusterCosetSearch, ...],
    syndrome: int,
    max_weight: int,
    split_syndrome: _SplitSyndrome,
    join_part_bounds: _JoinPartBounds,
) -> tuple[MinWeightBound, int]:
    """Answer a query of a batch with connected-cluster searches, by increasing limits.

    The syndrome of the query must be in range (see ``CosetLeaderSearch._check_syndromes``).
    Section 3.8 of ``connected_cluster.md`` says why the limits increase.  Also returns the number
    of states that the searches visited.
    """
    if len(searches) == 1:
        return searches[0]._get_min_weight_bound_and_states(syndrome, max_weight, 2)
    x_syndrome, z_syndrome = split_syndrome(syndrome)
    x_bound, x_states = searches[0]._get_min_weight_bound_and_states(x_syndrome, max_weight, 2)
    z_bound, z_states = searches[1]._get_min_weight_bound_and_states(z_syndrome, max_weight, 2)
    return join_part_bounds(x_bound, z_bound), x_states + z_states
