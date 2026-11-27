# SPDX-License-Identifier: Apache-2.0

"""Connected-cluster coset searches of ``CosetLeaderSearch``, with checks made sparser first.

A ``ConnectedClusterCosetSearch`` makes its checks sparser with ``get_sparser_rows``, tabulates the
words of weight zero or one by syndrome, and searches the sparser checks with a ``SupportSearch``
(``connected_cluster.md#sec-syndromes``).  A single query, ``get_min_weight_bound``, looks up its
syndrome in the table, and otherwise searches the weight limit ``max_weight`` with
``SupportSearch.find_vector_with_packed_syndrome``.  The batch queries,
``find_first_not_within_max_weight`` and ``get_min_weights``, split a batch into blocks with
``_threads.run_blocks``, and answer each block in one call of the compiled loop
``_find_min_weights``.  For each syndrome of the block, the loop calls ``_get_min_weight``, which
maps the syndrome to the sparser checks, compares it with the checks of single symbols, and tries
increasing weight limits with ``_search_supports``.  Several threads can run these calls at once
(``connected_cluster.md#sec-threads``), so a change must keep the requirements on threads of
``connected_cluster.md#sec-requirements``.  ``problems.md#sec-coset-leader-search`` says how
``CosetLeaderSearch`` answers its queries with these searches.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qldpc._binary import (
    get_num_words,
    int_to_words,
    pack_bits,
    row_ints,
    trailing_zeros_word,
    words_to_int,
)
from qldpc._compilation import always_compiled

from .. import _threads
from .._code_matrices import swap_pauli_halves
from .._cosets import get_unit_syndrome_words, tabulate_singles, xor_selected, xor_selected_rows
from .._tuning import MAX_VALUE
from ..min_weight_bounds import MinWeightBound
from .search import SupportSearch, _CutTables, _get_fixed_words, _search_supports
from .sparse_checks import get_sparser_rows

# The tables of the compiled batch queries: the packed columns of the map of syndromes to the
# sparser checks, then the tables of their search that ``_search_supports`` reads, in its order,
# with those of the checks and of the cuts each a tuple.  They are a plain tuple, whose type Numba
# infers faster than that of a record at each call.
_BatchTables = tuple[
    npt.NDArray[np.uint64],
    tuple[
        npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.uint64], npt.NDArray[np.uint64]
    ],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    tuple[int, ...],
    bool,
    _CutTables,
]

####################################################################################################
# coset searches


class ConnectedClusterCosetSearch:
    """A coset search of a ``CosetLeaderSearch`` by the connected-cluster search of its checks.

    The words and syndromes are those of a ``BrouwerZimmermannCosetSearch``: packed Paulis, or X or
    Z parts of Paulis.  The coset search makes its checks sparser, with the same span, and searches
    from the syndrome of each query, mapped to them (``connected_cluster.md#sec-syndromes``).
    ``get_min_weight_bound`` searches only the weight limit ``max_weight``, and gives these lower
    bounds on the weight of the words of a coset:

    - The weight of the word, for a coset with a word of weight zero or one, which a table finds.
    - Two, for any other coset with a word within ``max_weight``, since the word can be lighter.
    - The larger of two and ``max_weight + 1``, for any other coset.

    The batch queries instead try increasing weight limits, and so find the weight of the lightest
    words of each coset that has a word within ``max_weight`` (see ``get_min_weights``).

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

        # Make the checks sparser, with the same span.
        rows, transform = get_sparser_rows(
            matrix,
            symplectic=symplectic,
            max_rounds=max_rounds,
            rounds_without_improvement=rounds_without_improvement,
        )
        self._rows = rows
        # Row ``i`` of ``rows`` sums the given rows that row ``i`` of ``transform`` selects, so a
        # given syndrome maps to the sum of the columns of ``transform`` that its bits select.
        # Single queries map their syndromes as integers, and the compiled batch queries as packed
        # words.
        self._syndrome_map_columns = row_ints(np.ascontiguousarray(transform.T))
        self._packed_map_columns = pack_bits(np.ascontiguousarray(transform.T))
        # Without a function that finds automorphisms, the search never asks for them.
        self._search = SupportSearch(
            rows,
            np.zeros((0, matrix.shape[1]), dtype=np.uint8),
            symplectic=symplectic,
            automorphism_minimum_states=MAX_VALUE,
            restrict_viable_symbols=restrict_viable_symbols,
            stabilizer_overlap_cuts=stabilizer_overlap_cuts,
            # Sums of pairs of checks made queries slower (connected_cluster.md#sec-syndromes).
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

        The search tries only the weight limit ``max_weight``, if it is at least two
        (``connected_cluster.md#sec-syndromes``).  The class docstring states the lower bounds of
        the answers.
        """
        # Look up a word of weight zero or one.
        word = self._singles.get(syndrome)
        if word is not None:
            weight = int(word != 0)
            return MinWeightBound(weight <= max_weight, word, weight)

        # Map the syndrome to the sparser checks, which are independent, so the search keeps them
        # all, in order.  The map is invertible and the lookup answers the zero syndrome, so the
        # mapped syndrome is not zero, as ``find_vector_with_packed_syndrome`` requires.
        syndrome = xor_selected(self._syndrome_map_columns, syndrome)

        # The lookup leaves only cosets without a word below weight two, so search no limit below
        # two.
        if max_weight >= 2:
            packed_syndrome = int_to_words(syndrome, get_num_words(len(self._rows)))
            symbols, _ = self._search.find_vector_with_packed_syndrome(packed_syndrome, max_weight)
            if symbols is not None:
                return MinWeightBound(True, self._get_word(symbols), 2)
        representative = xor_selected(self._inverse_rows, syndrome)
        return MinWeightBound(False, representative, max(2, max_weight + 1))

    def find_first_not_within_max_weight(
        self,
        syndromes: npt.NDArray[np.uint64],
        max_weight: int,
        *,
        num_threads: int = 1,
        split_work: int = 0,
    ) -> tuple[int, MinWeightBound | None]:
        """Find the first syndrome whose coset has no word of weight at most ``max_weight``.

        The syndromes are those of the given checks, packed as for
        ``CosetLeaderSearch.find_first_not_within_max_weight``.  The search of a syndrome tries the
        weight limits 2, 3, ..., ``max_weight`` in turn, unless it looks up a word of weight at most
        one (``_get_min_weight``).  On more than one thread, the calling thread searches the batch
        alone until its queries have done ``split_work`` work, one unit for each state that their
        searches visit, and ``num_threads`` threads then search the rest of it
        (``_threads.run_blocks``; ``connected_cluster.md#sec-threads``).  The result is the same on
        any number of threads.  ``CosetLeaderSearch`` passes the thread options of its tuning.  The
        defaults search on the calling thread alone.

        Returns:
            The index of the first such syndrome and the ``MinWeightBound`` that
            ``get_min_weight_bound`` gives for it; or the number of syndromes and None if there is
            none.
        """
        floors = np.zeros(len(syndromes), dtype=np.int64)
        index, _ = self._search_batch(syndromes, max_weight, floors, True, num_threads, split_work)
        if index == len(syndromes):
            return index, None
        return index, self.get_min_weight_bound(words_to_int(syndromes[index]), max_weight)

    def get_min_weights(
        self,
        syndromes: npt.NDArray[np.uint64],
        max_weight: int,
        floors: npt.NDArray[np.int64],
        *,
        num_threads: int = 1,
        split_work: int = 0,
    ) -> npt.NDArray[np.int64]:
        """Find the minimum weight of the coset of each syndrome, if it is at most ``max_weight``.

        The syndromes are packed as for ``find_first_not_within_max_weight``.  The search of a
        syndrome tries the weight limits from the larger of two and its floor in ``floors``, which
        must be at most ``max_weight``, unless it looks up a word of weight at most one
        (``_get_min_weight``).  A batch moves to ``num_threads`` threads as in
        ``find_first_not_within_max_weight``, and each weight is the same on any number of threads.

        Returns:
            For each syndrome, a weight at most its floor if its coset has a word within the floor,
            and otherwise the minimum weight of its coset, or ``max_weight + 1`` if that is larger.
            So the larger of the floor and this weight is the larger of the floor and the minimum
            weight, up to ``max_weight + 1``.
        """
        return self._search_batch(syndromes, max_weight, floors, False, num_threads, split_work)[1]

    def _search_batch(
        self,
        syndromes: npt.NDArray[np.uint64],
        max_weight: int,
        floors: npt.NDArray[np.int64],
        stop_above_max_weight: bool,
        num_threads: int,
        split_work: int,
    ) -> tuple[int, npt.NDArray[np.int64]]:
        """Search the blocks of a batch with ``_find_min_weights``, run by ``_threads.run_blocks``.

        Returns:
            The index that ``_threads.run_blocks`` returns, and the weights that the blocks wrote.
        """
        syndromes = np.ascontiguousarray(syndromes)
        floors = np.ascontiguousarray(floors)
        weights = np.empty(len(syndromes), dtype=np.int64)
        tables = self._get_batch_tables()

        def search_rows(start: int, end: int, max_work: int) -> tuple[int, bool]:
            """Search rows in order, and write their weights."""
            index, ends = _find_min_weights(
                syndromes[start:end],
                max_weight,
                floors[start:end],
                max_work,
                stop_above_max_weight,
                weights[start:end],
                tables,
            )
            return start + index + int(ends), bool(ends)

        index = _threads.run_blocks(len(syndromes), num_threads, split_work, lambda: search_rows)
        return index, weights

    def _get_batch_tables(self) -> _BatchTables:
        """The tables that the compiled batch queries read, as a plain tuple (see ``_BatchTables``).

        The search adds no sums of pairs of checks, so the syndrome of a query needs no map to them.
        The cuts of queries are tabulated (see ``SupportSearch.find_vector_with_packed_syndrome``).
        """
        search = self._search
        assert search._query_cuts is not None
        return (
            self._packed_map_columns,
            search._check_tables[:],
            search._ranks,
            search._inverse_overlaps,
            _get_fixed_words(search._check_tables.symbol_masks.shape[1]),
            search._restrict_viable_symbols,
            search._query_cuts,
        )

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
# compiled coset search


@always_compiled(nogil=True)
def _find_min_weights(
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    floors: npt.NDArray[np.int64],
    max_work: int,
    stop_above_max_weight: bool,
    weights: npt.NDArray[np.int64],
    tables: _BatchTables,
) -> tuple[int, bool]:
    """Search the coset of each syndrome in turn with ``_get_min_weight``, within a budget of work.

    The search of syndrome ``i`` starts from its floor ``floors[i]``, and writes its weight to
    ``weights[i]``.  The work of a query is the number of states that its searches visit.  The
    search stops after the last syndrome, or once the work of this call reaches ``max_work``, or, if
    ``stop_above_max_weight``, at the first syndrome whose coset has no word within ``max_weight``.
    It writes only ``weights`` and arrays that it allocates, so calls that run at once may share
    every argument but ``weights``.  It runs without the global interpreter lock, and keeps the
    rules on threads of ``connected_cluster.md#sec-requirements``.

    Returns:
        If the search stopped at a syndrome whose coset has no word within ``max_weight``, the index
        of that syndrome and True.  Otherwise the number of syndromes searched and False.
    """
    # A buffer for the packed syndrome of a query for the sparser checks (see ``_get_min_weight``).
    mapped = np.empty(tables[0].shape[1], dtype=np.uint64)
    work = 0
    for index in range(len(syndromes)):
        weight, num_states = _get_min_weight(
            syndromes[index], max_weight, floors[index], mapped, tables
        )
        weights[index] = weight
        if stop_above_max_weight and weight > max_weight:
            return index, True
        work += num_states
        if work >= max_work:
            return index + 1, False
    return len(syndromes), False


@always_compiled(inline=True)
def _get_min_weight(
    syndrome: npt.NDArray[np.uint64],
    max_weight: int,
    first_limit: int,
    mapped: npt.NDArray[np.uint64],
    tables: _BatchTables,
) -> tuple[int, int]:
    """Find the weight of a word of the coset of a syndrome, by increasing weight limits.

    ``syndrome`` is a packed syndrome of the given checks, and row ``j`` of ``map_columns``, the
    first of ``tables``, is the packed syndrome for the sparser checks of the word whose given
    syndrome has only bit ``j`` set (see ``ConnectedClusterCosetSearch``).  The search writes the
    mapped syndrome to ``mapped``.  It gives the weight zero to the zero syndrome, and one to the
    syndrome of a symbol, which flips the first check of the syndrome.  Otherwise it tries each
    weight limit from ``max(2, first_limit)`` through ``max_weight`` with ``_search_supports``, as
    ``SupportSearch.find_vector_with_packed_syndrome`` does, and stops at the first within which it
    finds a word (``connected_cluster.md#sec-syndromes``).  The other tables are those of
    ``_search_supports``, from the search of the sparser checks.

    Returns:
        A weight and the number of states that the searches visited.  The weight is the minimum
        weight ``w`` of the coset, or ``max_weight + 1`` if that is smaller, unless ``2 <= w <
        first_limit``, when it is ``first_limit``, which must then be at most ``max_weight``.
    """
    map_columns, check_tables, ranks, inverse_overlaps, fixed_words, restrict, cuts = tables
    check_starts, check_symbols, symbol_masks, test_columns = check_tables

    # Map the syndrome to the sparser checks.  The map is invertible, so only the zero syndrome maps
    # to zero.
    mapped[:] = 0
    xor_selected_rows(syndrome, map_columns, mapped)
    first_check = -1
    for word in range(len(mapped)):
        if mapped[word]:
            first_check = 64 * word + int(trailing_zeros_word(mapped[word]))
            break
    if first_check < 0:
        return 0, 0

    # A word of weight one is a symbol whose checks are those of the syndrome.
    for position in range(check_starts[first_check], check_starts[first_check + 1]):
        symbol = check_symbols[position]
        equal = True
        for word in range(len(mapped)):
            if symbol_masks[symbol, word] != mapped[word]:
                equal = False
                break
        if equal:
            return 1, 0

    # Every other word weighs at least two, so search no limit below two.
    roots = np.full(1, -1, dtype=np.int64)
    paths = np.empty((1, 0), dtype=np.int64)
    no_path = np.empty(0, dtype=np.int64)
    num_states = 0
    for limit in range(max(2, first_limit), max_weight + 1):
        # Name each table, since star-unpacking the tuples here makes queries a few percent slower.
        support, item_states, _, _, _ = _search_supports(
            check_starts,
            check_symbols,
            symbol_masks,
            test_columns,
            roots,
            paths,
            0,
            ranks,
            inverse_overlaps,
            min(limit, len(ranks)),
            no_path,
            no_path,
            MAX_VALUE,
            mapped,
            fixed_words,
            True,
            restrict,
            cuts[0],
            cuts[1],
            cuts[2],
            cuts[3],
            cuts[4],
            cuts[5],
        )
        num_states += item_states[0]
        if len(support):
            return limit, num_states
    return max_weight + 1, num_states
