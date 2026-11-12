# SPDX-License-Identifier: Apache-2.0

"""Brouwer-Zimmermann coset searches of ``CosetLeaderSearch``: the cosets of one binary code."""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._binary import popcount_word, unpack_bits
from qldpc._compilation import always_compiled

from .. import _tuning
from .._binary import symplectic_to_hamming
from .._cosets import (
    MinWeightBound,
    get_unit_syndrome_words,
    int_to_words,
    row_ints,
    tabulate_singles,
    words_to_int,
    xor_selected,
    xor_selected_rows,
)
from .._information_sets import certificates
from .._tuning import SearchTuning
from . import affine_instance, affine_search, coset_routes
from .coset_tables import (
    CosetSearchTables,
    CosetTails,
    PlanningBuffer,
    TailFields,
    get_coset_tails,
    get_first_step_reaching,
    get_planning_table,
    get_search_tables,
    get_stopping_step,
)

####################################################################################################
# coset searches


class BrouwerZimmermannCosetSearch:
    """A BZ coset search of a ``CosetLeaderSearch``: a search of the cosets of one binary code.

    The words of a search are either packed Paulis on ``n`` qubits, in which case the code is
    spanned by the packed stabilizers, or the X parts (or Z parts) of Paulis, in which case the code
    is spanned by the X parts of the pure-X stabilizers (or the Z parts of the pure-Z stabilizers).
    Each coset is labeled by the syndrome of its words, whose bit ``j`` is the parity of a word with
    row ``j`` of a matrix of checks.

    The search answers queries of each maximum weight with the tables of a family of information
    sets of the code, which ``routes`` choose (see ``coset_routes``).  The tables of every family
    have the same ``inverse`` and ``x_mask``.  The queries of a route by tails weigh the words of
    its family by their tails, and skip completions by the completion tables, from its record of
    tails.

    Args:
        routes: The routes of the search.
        css_weight: True to weigh words by their CSS weight, which is the Hamming weight of an X
            or Z part, or False to weigh them by their physical weight.
        num_qubits: The number of qubits ``n``.
        inverse_rows: Row ``j`` of ``get_unit_syndrome_words`` of the checks, not embedded, as an
            integer: a word whose syndrome has only bit ``j`` set.
        check_columns: Column ``c`` of the checks as an integer: the syndrome of the word with only
            bit ``c`` set.
        is_pauli: Whether the words are packed Paulis of ``2 n`` bits, rather than X or Z parts of
            ``n`` bits.
        max_css_single_pairs: For a search of Pauli errors by CSS weight, the most pairs of distinct
            syndromes of single-qubit X and Z errors that it tabulates (see ``_css_singles``).
    """

    def __init__(
        self,
        routes: coset_routes.CosetRoutes[CosetSearchTables, CosetTails],
        css_weight: bool,
        num_qubits: int,
        inverse_rows: tuple[int, ...],
        check_columns: tuple[int, ...],
        *,
        is_pauli: bool,
        max_css_single_pairs: int = 0,
    ) -> None:
        self._routes = routes
        self._max_css_single_pairs = max_css_single_pairs
        self.css_weight = css_weight
        # The compiled search weighs embedded words by their CSS weight only in a search of Pauli
        # errors by CSS weight, and otherwise by their Hamming weight, which is twice the physical
        # weight of an error, and the weight of an X or Z part (see ``_search_coset``).  The scale
        # chooses which of these weights a scan computes.  The weight factor of the coset instances
        # is separate: it is two for every search of Pauli errors, including those by CSS weight,
        # whose scale is zero, since an error weighs at least half the Hamming weight of its
        # embedded word (see ``get_pauli_search``).
        self.scale = 0 if css_weight and is_pauli else 1 if css_weight else 2
        self._num_qubits = num_qubits
        self._is_pauli = is_pauli
        self._inverse_rows = inverse_rows
        self._num_syndrome_bits = len(inverse_rows)
        # A query below maximum weight two bounds a coset without words of weight at most one by
        # two.  If the search is of X or Z parts and rounds to parity (see ``get_search_tables``),
        # every word of an odd coset has odd weight, so such an odd coset has no word of weight two
        # either, and the query bounds it by three.  An odd coset of a search of Pauli errors by
        # CSS weight can still hold an error of weight two, such as ``YX``, whose binary word has
        # three ones.
        num_tables = len(routes.disjoint_tables.planning)
        self._odd_coset_lookup_bound = 3 if not is_pauli and num_tables == 2 else 2

        # Map each syndrome of a word of weight zero or one to such a word.  For the CSS weight,
        # ``_weight_one_word`` also pairs the single-qubit X errors with the single-qubit Z errors,
        # to find ``X_i Z_j`` on distinct qubits.
        self._singles, self._x_singles, self._z_singles = tabulate_singles(
            check_columns, num_qubits, is_pauli=is_pauli
        )

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether the coset of this syndrome has a word of weight at most ``max_weight``.

        Returns:
            A ``MinWeightBound`` for the coset, whose representative is a word of the coset.
        """
        word = self._weight_one_word(syndrome)
        if word is not None:
            weight = self._weight(word)
            return MinWeightBound(weight <= max_weight, word, weight)

        # Otherwise the coset has no word of weight at most one, so two is a lower bound, and so is
        # the odd-coset lookup bound for an odd coset.
        if max_weight < 2:
            word = xor_selected(self._inverse_rows, syndrome)
            return MinWeightBound(
                False, word, self._odd_coset_lookup_bound if word.bit_count() % 2 else 2
            )

        # Search the coset with the tables of the route of the maximum weight, by tails on a route
        # by tails, and keep the lightest word that the search weighs in ``best``.
        tables, tails = self._get_route(max_weight)
        best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
        packed_syndrome = int_to_words(syndrome, max(1, (self._num_syndrome_bits + 63) // 64))
        lower_bound = _search_coset(packed_syndrome, max_weight, best, *tables, tails, self.scale)
        # A search that finds a light enough word returns zero, but every word of the coset weighs
        # at least two, since the coset has no word of weight at most one.
        return MinWeightBound(lower_bound <= max_weight, self.unembed(best), max(2, lower_bound))

    def find_first_not_within_max_weight(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> tuple[int, MinWeightBound | None]:
        """Find the first syndrome whose coset has no word of weight at most ``max_weight``.

        The syndromes are packed as for ``CosetLeaderSearch.find_first_not_within_max_weight``.
        The search answers them with the route of the maximum weight, which it takes once.

        Returns:
            The index of that syndrome, or the number of syndromes if there is none, and a
            ``MinWeightBound`` for its coset, with the lightest word that the search weighed, or
            None if there is none.
        """
        tables, tails = self._get_route(max_weight)
        best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
        index, lower_bound = _find_first_not_within_max_weight(
            syndromes, max_weight, best, *tables, tails, self.scale
        )
        if index == len(syndromes):
            return index, None
        return index, MinWeightBound(False, self.unembed(best), int(lower_bound))

    def get_min_weights(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int, floors: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        """Find the minimum weight of the coset of each syndrome, if it is at most ``max_weight``.

        The syndromes are packed as for ``CosetLeaderSearch.find_first_not_within_max_weight``.  The
        search of a syndrome also ends at a word of weight at most its floor in ``floors``, which
        must be at most ``max_weight``.  No coset has a minimum weight above the number ``n`` of
        qubits, so the search answers the syndromes with the route of the lesser of ``max_weight``
        and ``n``, which it takes once.  Unlike ``get_min_weight_bound``, it looks up no words of
        weight at most one, since the first layers of a search find them cheaply.

        Returns:
            For each syndrome, the weight of a word of its coset of weight at most its floor if
            there is one, and otherwise the minimum weight of its coset, or ``max_weight + 1`` if
            that is larger (see ``_search_coset_for_min_weight``).
        """
        max_weight = min(max_weight, self._num_qubits)
        tables, tails = self._get_route(max_weight)
        return _find_min_weights(syndromes, max_weight, floors, *tables, tails, self.scale)

    def unembed(self, embedded_word: npt.NDArray[np.uint64]) -> int:
        """Convert a packed embedded word into a word, without the columns that the embedding adds.

        The embedding for the physical weight appends ``x xor z`` to a packed Pauli (see
        ``CosetSearchTables``), and the other embeddings add nothing.
        """
        return words_to_int(embedded_word) & ((1 << self._num_word_bits) - 1)

    def _get_route(self, max_weight: int) -> tuple[CosetSearchTables, TailFields | None]:
        """The tables of the route of a maximum weight, and the fields of its tails, if any.

        The fields are a plain tuple, which numba types at each call of a compiled search several
        times faster than the named tuple ``CosetTails``.
        """
        tables, tails = self._routes.get_route(max_weight)
        return tables, None if tails is None else tails[:]

    @property
    def _num_word_bits(self) -> int:
        """The number of bits of a word: ``2 n`` for Paulis, and ``n`` for X or Z parts."""
        return 2 * self._num_qubits if self._is_pauli else self._num_qubits

    def _weight(self, word: int) -> int:
        """The weight of a word: of its Pauli, or the Hamming weight of an X or Z part."""
        if not self._is_pauli:
            return word.bit_count()
        mask = (1 << self._num_qubits) - 1
        x, z = word & mask, (word >> self._num_qubits) & mask
        return max(x.bit_count(), z.bit_count()) if self.css_weight else (x | z).bit_count()

    def _weight_one_word(self, syndrome: int) -> int | None:
        """Find a word of weight zero or one with this syndrome, if there is one."""
        if syndrome in self._singles:
            return self._singles[syndrome]
        if not (self._is_pauli and self.css_weight):
            return None
        # CSS weight one also allows X_i Z_j on distinct qubits.  Callers such as the certification
        # of fault propagation make a query for each of thousands of syndromes, so a table of these
        # errors, built on the first query that needs it, saves a scan of the qubits per query.
        if self._css_singles is not None:
            return self._css_singles.get(syndrome)
        # If that table would be too large, match the syndrome of each single-qubit X error against
        # those of the single-qubit Z errors, in time linear in the number of qubits.
        for sx, x in self._x_singles.items():
            z = self._z_singles.get(syndrome ^ sx)
            if z is not None:
                return x ^ z
        return None

    @functools.cached_property
    def _css_singles(self) -> dict[int, int] | None:
        """Errors X_i Z_j (with X_i or Z_j possibly the identity) by syndrome, if not too many."""
        return _tabulate_css_singles(self._x_singles, self._z_singles, self._max_css_single_pairs)


class StabilizerKey:
    """A stabilizer matrix and its checks, which a cache can hash and compare by the matrix.

    The checks follow from the stabilizers, so the stabilizers alone key the cache, and the key
    keeps both matrices to build a search that the cache has not stored yet.  Keys of packed bytes
    would not hold the matrices, but unpacking them would cost about 0.4% of the setup of a search.
    The matrices may be arrays of the caller.  The key compares a copy of the bytes of the
    stabilizers, and the search copies what it keeps of them, so later changes to the arrays of the
    caller reach neither.
    """

    def __init__(self, stabilizers: npt.NDArray[np.uint8], checks: npt.NDArray[np.uint8]) -> None:
        self.stabilizers = stabilizers
        self.checks = checks
        self._key = (stabilizers.shape, stabilizers.tobytes())
        self._hash = hash(self._key)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StabilizerKey) and self._key == other._key


class CosetTuning(NamedTuple):
    """The fields of the tuning that the setup of a BZ coset search reads, which key its cache.

    ``disjoint_random_column_orders`` configures the disjoint information sets of the search.  Only
    a search of Pauli errors by CSS weight reads ``max_css_single_pairs``, for its lookups.  The
    fields from ``overlapping_minimum_work`` to ``overlapping_attempts_per_set`` configure the
    families with which the search answers costly queries, and ``overlapping_minimum_work`` also
    chooses its routes by tails, whose queries weigh words by their tails and skip completions by
    tables, if it weighs Hamming weights (see ``coset_routes``).  The last two fields configure
    those tables, which the search of Pauli errors by CSS weight does not have.  The search builds
    the families, tails, and tables after the cache stores it.
    """

    disjoint_random_column_orders: int
    max_css_single_pairs: int
    overlapping_minimum_work: int
    max_information_sets: int
    overlapping_attempts_per_set: int
    completion_bound_max_table_bytes: int
    coset_completion_block_bits: int

    @classmethod
    def from_tuning(cls, tuning: SearchTuning, *, paulis_by_css_weight: bool) -> CosetTuning:
        """The fields of a tuning that the setup of a search reads.

        ``paulis_by_css_weight`` is True for a search of Pauli errors by CSS weight.  Only that
        search reads ``max_css_single_pairs``, and it alone has no completion tables.  The fields
        that a search does not read are zero, so that oracles built under tunings that differ only
        in those fields share cached searches.
        """
        fields = (getattr(tuning, name) for name in cls._fields)
        coset_tuning = cls(*fields)
        if paulis_by_css_weight:
            return coset_tuning._replace(
                completion_bound_max_table_bytes=0, coset_completion_block_bits=0
            )
        return coset_tuning._replace(max_css_single_pairs=0)


@functools.lru_cache(maxsize=_tuning.PAULI_SEARCH_CACHE_SIZE)
def get_pauli_search(
    key: StabilizerKey, css_weight: bool, tuning: CosetTuning
) -> BrouwerZimmermannCosetSearch:
    """The Pauli search, of Pauli errors modulo the stabilizers, with syndromes for the stabilizers.

    ``key`` holds the stabilizers and their checks (see ``CosetLeaderSearch``).  The syndromes
    depend on the rows of the stabilizer matrix, not only on the group that they generate, so the
    rows key the cache.  ``css_weight`` and the fields of the tuning that the setup reads key it
    too.
    """
    stabilizers, checks = key.stabilizers, key.checks
    num_qubits = len(stabilizers)
    inverse = get_unit_syndrome_words(checks)
    embedded_stabilizers = stabilizers if css_weight else symplectic_to_hamming(stabilizers)
    embedded_inverse = inverse if css_weight else symplectic_to_hamming(inverse)
    # An error weighs at least half of the Hamming weight of its embedded word, by either measure of
    # weight, so the certificate bounds are halved.  A search by CSS weight never weighs words by
    # tails, since that weight needs to know which deleted pivot columns are in the X part.
    routes = _get_routes(
        embedded_stabilizers,
        embedded_inverse,
        num_qubits,
        weight_factor=2,
        tuning=tuning,
        by_tails=not css_weight,
    )
    check_columns = row_ints(np.ascontiguousarray(checks.T))
    return BrouwerZimmermannCosetSearch(
        routes,
        css_weight,
        num_qubits,
        row_ints(inverse),
        check_columns,
        is_pauli=True,
        max_css_single_pairs=tuning.max_css_single_pairs,
    )


@functools.lru_cache(maxsize=_tuning.CSS_SEARCH_CACHE_SIZE)
def get_css_searches(
    num_qubits: int, num_x_rows: int, x_rows: bytes, z_rows: bytes, tuning: CosetTuning
) -> tuple[BrouwerZimmermannCosetSearch, BrouwerZimmermannCosetSearch]:
    """The classical searches of the X parts and of the Z parts of the errors on a CSS state.

    ``x_rows`` holds the X parts of ``num_x_rows`` pure-X stabilizers, and ``z_rows`` the Z parts of
    the other ``num_qubits - num_x_rows`` stabilizers, which are pure-Z.  Both are in reduced row
    echelon form, which depends only on the stabilizer group, and packed by ``pack_bits`` into
    bytes, which ``lru_cache`` can hash.  The X parts of errors are searched modulo the code of the
    X rows, with the Z rows as checks, and the Z parts the other way around (see
    ``CosetLeaderSearch``).  The fields of the tuning that the setup reads key the cache too.
    """
    num_words = max(1, (num_qubits + 63) // 64)
    x_matrix = unpack_bits(
        np.frombuffer(x_rows, dtype=np.uint64).reshape(num_x_rows, num_words), num_qubits
    )
    z_matrix = unpack_bits(
        np.frombuffer(z_rows, dtype=np.uint64).reshape(num_qubits - num_x_rows, num_words),
        num_qubits,
    )
    return (
        _get_classical_search(x_matrix, z_matrix, tuning),
        _get_classical_search(z_matrix, x_matrix, tuning),
    )


def _get_classical_search(
    generators: npt.NDArray[np.uint8], checks: npt.NDArray[np.uint8], tuning: CosetTuning
) -> BrouwerZimmermannCosetSearch:
    """The classical search of X or Z parts modulo the row space of independent generators.

    The row space of the generators is the null space of the independent checks, so each coset is
    labeled by its syndrome for the checks.  Words are weighed by their Hamming weight, which the BZ
    certificate bounds directly, so its bounds are not halved.
    """
    num_qubits = generators.shape[1]
    inverse = get_unit_syndrome_words(checks)
    # Every column counts as the X part, so the CSS weight of a word is its Hamming weight.
    routes = _get_routes(
        generators, inverse, num_qubits, weight_factor=1, tuning=tuning, by_tails=True
    )
    check_columns = row_ints(np.ascontiguousarray(checks.T))
    return BrouwerZimmermannCosetSearch(
        routes, True, num_qubits, row_ints(inverse), check_columns, is_pauli=False
    )


def _get_routes(
    generators: npt.NDArray[np.uint8],
    inverse: npt.NDArray[np.uint8],
    num_x_columns: int,
    *,
    weight_factor: int,
    tuning: CosetTuning,
    by_tails: bool,
) -> coset_routes.CosetRoutes[CosetSearchTables, CosetTails]:
    """The routes of a coset search of the code of independent embedded generators.

    The routes start from the coset instance of the disjoint information sets of the generators,
    with the weight factor ``weight_factor`` (see ``affine_instance.get_coset_instance``).  The
    tables of every family of the routes come from ``get_search_tables``, with ``inverse`` and
    ``num_x_columns``.  If ``by_tails`` is True, the routes whose queries are predicted to examine
    at least ``tuning.overlapping_minimum_work`` words with their own family are routes by tails,
    with records of tails and tables from ``get_coset_tails``, and otherwise no route is.
    """
    instance = affine_instance.get_coset_instance(
        generators,
        random_column_orders=tuning.disjoint_random_column_orders,
        weight_factor=weight_factor,
    )
    return coset_routes.CosetRoutes(
        generators,
        instance,
        functools.partial(get_search_tables, inverse=inverse, num_x_columns=num_x_columns),
        functools.partial(
            get_coset_tails,
            inverse=inverse,
            block_bits=tuning.coset_completion_block_bits,
            max_table_bytes=tuning.completion_bound_max_table_bytes,
        )
        if by_tails
        else None,
        minimum_work=tuning.overlapping_minimum_work,
        max_sets=tuning.max_information_sets,
        attempts_per_set=tuning.overlapping_attempts_per_set,
    )


def _tabulate_css_singles(
    x_singles: dict[int, int], z_singles: dict[int, int], max_pairs: int
) -> dict[int, int] | None:
    """Tabulate the products of single-qubit X and Z errors by syndrome, if not too many.

    Args:
        x_singles: Single-qubit X errors, and the identity, by syndrome.
        z_singles: Single-qubit Z errors, and the identity, by syndrome.
        max_pairs: The most pairs of an X and a Z error that the table may come from.

    Returns:
        None if there are more than ``max_pairs`` pairs, and otherwise the first product
        ``x ^ z`` of each syndrome ``sx ^ sz``, in the order of the two dictionaries.
    """
    if len(x_singles) * len(z_singles) > max_pairs:
        return None
    table: dict[int, int] = {}
    for sx, x in x_singles.items():
        for sz, z in z_singles.items():
            table.setdefault(sx ^ sz, x ^ z)
    return table


####################################################################################################
# compiled coset search


@always_compiled
def _find_first_not_within_max_weight(
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    best: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    planning: PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    tails: TailFields | None,
    scale: int,
) -> tuple[int, int]:
    """Search the coset of each syndrome in turn with ``_search_coset``.

    The tables, from ``inverse`` to ``x_mask``, and ``tails`` are those of ``_search_coset``.
    Return the index of the first syndrome whose coset has no word of weight at most
    ``max_weight``, and the lower bound that ``_search_coset`` returns for it, with the lightest
    word that it weighed left in ``best``.  If there is no such syndrome, return the number of
    syndromes and zero.
    """
    for index in range(len(syndromes)):
        lower_bound = _search_coset(
            syndromes[index],
            max_weight,
            best,
            inverse,
            generators,
            pivots,
            planning,
            x_mask,
            tails,
            scale,
        )
        if lower_bound > max_weight:
            return index, lower_bound
    return len(syndromes), 0


@always_compiled(inline=True)
def _search_coset(
    packed_syndrome: npt.NDArray[np.uint64],
    max_weight: int,
    best: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    planning: PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    tails: TailFields | None,
    scale: int,
) -> int:
    """Search the coset of a syndrome for a word of weight at most ``max_weight``.

    ``packed_syndrome`` is the syndrome split into 64-bit machine words, with machine word ``j``
    holding bits ``64 j`` and up.  The search runs on embedded words.  In the Pauli search, the
    words are Pauli errors, and the embedded word of an error is its packed Pauli when errors are
    weighed by CSS weight, and its packed Pauli followed by ``x xor z`` when they are weighed by
    physical weight, so that its Hamming weight is twice the physical weight (see
    ``symplectic_to_hamming``).  In a classical search, the words are X or Z parts, each is its own
    embedded word, and its weight is its Hamming weight.  The weight of an embedded word is that of
    its word: its CSS weight if ``scale`` is zero, with its X part selected by ``x_mask``, and
    otherwise its Hamming weight divided by ``scale``, which is two for the physical weight and one
    for X or Z parts.  The tables are those of ``CosetSearchTables``.  Row ``j`` of ``inverse`` is
    the embedded word of a word whose syndrome has only bit ``j`` set.  ``generators`` and
    ``pivots`` are the information sets of a family of the embedded generators of the code, with
    pivots padded with -1.  ``planning`` is the planning buffer of the search with this family,
    whose cached stopping steps a query may update.  ``tails`` is None, or, on a route by tails
    (see ``coset_routes``), the fields of the ``CosetTails`` of the family, and then ``scale`` is
    one or two, since the search weighs words by their Hamming weight.  Numba compiles the search
    separately for None and for tails, and leaves out the branches that the type of ``tails`` rules
    out, so choosing between them costs nothing for each word.

    The embedded words of the coset form a coset of the embedded code.  The search starts from the
    sum of the rows of ``inverse`` for the set bits of the syndrome, a word with that syndrome.  The
    offset of an information set is this word reduced by rows of the set until it is zero on the
    set's pivot columns.  Every embedded word of the coset is the offset plus a unique sum of rows
    of the set, and has a one on a pivot column exactly where it adds that pivot's row, so the BZ
    certificate applies to the coset as it does to a space.  Layer ``w`` of a set is its offset plus
    each sum of ``w`` of its rows, so layer zero is the offset itself.

    The search reads the bounds of its steps from the table of the planning buffer for the parity of
    the Hamming weight of the word of the syndrome (see ``get_planning_table``).

    The search visits layer zero of the information sets in order, then layer one, and so on.  It
    keeps the lightest word visited in ``best``, and stops at the first word of weight at most
    ``max_weight``.  Once the first ``j`` sets have completed layer ``w``, every unvisited word
    weighs at least the bound of step ``(w, j)`` in its table.  So the search also stops at its
    stopping step, the first step whose bound exceeds ``max_weight``, which ``get_stopping_step``
    finds.  It skips each set that adds nothing to the bound of that step (see
    ``_search_from_syndrome_word``).  A search by tails visits the same words in the same order,
    except those that the tables of ``tails`` prove to lie beyond ``max_weight``, which it skips.
    So it finds the same first word of weight at most ``max_weight``, if any.

    Returns:
        A lower bound on the weight of every word in the coset: zero, which is at most
        ``max_weight``, if ``best`` has weight at most ``max_weight``, and otherwise a bound that
        exceeds ``max_weight`` (see ``_search_from_syndrome_word``).
    """
    best_weight, parity = _get_syndrome_word(packed_syndrome, best, inverse, x_mask, scale)
    if best_weight <= max_weight:
        return 0
    table = get_planning_table(planning, parity)
    stop_layer, stop_num_completed, stop_bound = get_stopping_step(table, max_weight)
    # The search rounds to the parity of the coset if it weighs Hamming weights and its planning
    # buffer has a table for each parity (see ``get_planning_table``).
    lower_bound, _, _ = _search_from_syndrome_word(
        packed_syndrome,
        max_weight,
        stop_layer,
        stop_num_completed,
        stop_bound,
        best,
        best_weight,
        generators,
        pivots,
        x_mask,
        tails,
        scale,
        parity if scale != 0 and len(planning) == 2 else -1,
    )
    return lower_bound


@always_compiled
def _search_from_syndrome_word(
    packed_syndrome: npt.NDArray[np.uint64],
    max_weight: int,
    stop_layer: int,
    stop_num_completed: int,
    stop_bound: int,
    best: npt.NDArray[np.uint64],
    best_weight: int,
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    x_mask: npt.NDArray[np.uint64],
    tails: TailFields | None,
    scale: int,
    parity: int,
) -> tuple[int, int, int]:
    """Search a coset, as ``_search_coset`` does, from the word of its syndrome.

    ``best`` holds that word, of weight ``best_weight`` above ``max_weight``, and
    ``packed_syndrome`` is its syndrome.  The stopping step and its bound are those of
    ``get_stopping_step``.  ``parity`` is the parity of the Hamming weights of the words of the
    coset if the search weighs Hamming weights and rounds its bounds to that parity (see
    ``get_search_tables``), and otherwise -1.  A set of rank deficiency ``s`` adds to the bound of
    a step only once it has completed layer ``s``, so a set that would not have completed it by the
    stopping step adds nothing, and the search skips it.  These are the unused sets of
    ``certificates.plan_steps``, whose work for this step, from layer zero, counts the words that
    the search weighs after the word of the syndrome if it finds no light enough word.

    A search by tails (see ``_search_coset``) weighs a word by its rows below the rank of its set
    and its tail.  It keeps the set, the layer, and the support of the lightest word, and writes
    that word into ``best`` before it returns.  In each layer of two or more rows of a set with
    tables, it skips the completions of a prefix whose bound reaches the skip bound: the least bound
    that proves a word of the coset heavier than ``max_weight``, one more than the greatest Hamming
    weight at most ``scale * max_weight`` that a word of the coset can have (see
    ``affine_search.scan_coset_layer_by_tails_with_tables``).  So it finds the same first word of
    weight at most ``max_weight``, if any.  If it finds none and skipped any word, it bounds the
    coset by the least weight beyond ``max_weight`` that a word of the coset can have, which bounds
    the skipped words, the words that it weighed, and the words beyond the stopping step.

    Returns:
        A lower bound on the weight of every word in the coset: zero if the search finds a word of
        weight at most ``max_weight``, the least weight beyond ``max_weight`` that a word of the
        coset can have if it skipped any word, and otherwise the lesser of the weight of the
        lightest word that it weighed and the bound of the stopping step.  Also the numbers of words
        weighed after the word of the syndrome and of supports skipped.  Only tests read the
        numbers, to check which sets a search skips and that a search by tails weighs as many words.
    """
    # Reduce the word of the syndrome to the offset of each information set, or, by tails, find the
    # rank of each set and the tail of its offset.
    num_sets, num_rows, width = generators.shape
    if tails is None:
        offsets = _get_offsets(best, generators, pivots)
    else:
        tail_rows, inverse_tails, completion_tables, layout, block_bits = tails
        ranks, offsets = _get_offset_tails(packed_syndrome, best, pivots, tail_rows, inverse_tails)
        width = tail_rows.shape[2]

    # The scans weigh ``factor`` units for each unit of weight: Hamming weights, which are ``scale``
    # times the weights, or CSS weights if ``scale`` is zero.
    factor = max(1, scale)
    best_units, cutoff = factor * best_weight, factor * max_weight
    skip_bound = _get_skip_bound(cutoff, parity)

    # Visit each layer one information set at a time, up to the stopping step.  The scans take work
    # space for one set.  Those by tails keep the support of each lighter word in ``best_support``,
    # and the others keep the word itself in ``best``.
    best_set, best_layer = -1, 0
    weighed, skipped = 0, 0
    widths = np.full(1, width, dtype=np.int64)
    support = np.empty(num_rows, dtype=np.int64)
    best_support = np.empty(num_rows, dtype=np.int64)
    prefixes = np.empty((num_rows, 1, width), dtype=np.uint64)
    for layer in range(stop_layer + 1):
        for set_index in range(num_sets if layer < stop_layer else stop_num_completed):
            if certificates.is_unused_set(pivots, set_index, stop_layer, stop_num_completed):
                continue
            set_skipped = 0
            if layer == 0:
                # The offset is zero on the pivot columns, so its tail has its Hamming weight.
                weight, ones = _get_weight(offsets[set_index], x_mask, scale)
                found, count = weight if scale == 0 else ones, 1
                if tails is None and found < best_units:
                    best[:] = offsets[set_index]
            elif tails is None:
                if scale == 0:
                    found, count = affine_search.scan_coset_layer_by_css_weight(
                        offsets[set_index],
                        generators[set_index : set_index + 1],
                        widths,
                        support[:layer],
                        prefixes,
                        best_units,
                        cutoff,
                        x_mask,
                        best,
                    )
                else:
                    found, count = affine_search.scan_coset_layer(
                        offsets[set_index],
                        generators[set_index : set_index + 1],
                        widths,
                        support[:layer],
                        prefixes,
                        best_units,
                        cutoff,
                        best,
                    )
            elif layer >= 2 and layout[set_index, 1]:
                # The tables of the set start at ``layout[set_index, 0]``, with a row of tables for
                # each row of the set but the last.
                num_blocks = layout[set_index, 1]
                block_offsets = layout[set_index, 2 : num_blocks + 3]
                start = layout[set_index, 0]
                end = start + (num_rows - 1) * block_offsets[-1]
                found, count, set_skipped = affine_search.scan_coset_layer_by_tails_with_tables(
                    offsets[set_index],
                    tail_rows[set_index : set_index + 1],
                    ranks[set_index : set_index + 1],
                    widths,
                    support[:layer],
                    prefixes,
                    best_units,
                    cutoff,
                    skip_bound,
                    completion_tables[start:end],
                    block_offsets,
                    block_bits,
                    best_support,
                )
            else:
                found, count = affine_search.scan_coset_layer_by_tails(
                    offsets[set_index],
                    tail_rows[set_index : set_index + 1],
                    ranks[set_index : set_index + 1],
                    widths,
                    support[:layer],
                    prefixes,
                    best_units,
                    cutoff,
                    best_support,
                )
            weighed += count
            skipped += set_skipped
            if found < best_units:
                best_units, best_set, best_layer = found, set_index, layer
            if best_units <= cutoff:
                break
        if best_units <= cutoff:
            break

    # By tails, write the lightest word into ``best``, which holds the word of the syndrome until
    # then.
    if tails is not None and best_set >= 0:
        _rebuild_word(best, generators[best_set], pivots[best_set], best_support[:best_layer])

    # A search that finds no light enough word but skipped words bounds the coset by the least
    # weight beyond the maximum weight that a word of the coset can have: the skip bound rounded up
    # to the parity of the coset, then divided by the scale and rounded up, as the bounds of the
    # steps are.
    if best_units <= cutoff:
        lower_bound = 0
    elif skipped:
        rounded = skip_bound + (skip_bound - parity) % 2 if parity >= 0 else skip_bound
        lower_bound = -(-rounded // factor)
    else:
        lower_bound = min(best_units // factor, stop_bound)
    return lower_bound, weighed, skipped


@always_compiled
def _find_min_weights(
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    floors: npt.NDArray[np.int64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    planning: PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    tails: TailFields | None,
    scale: int,
) -> npt.NDArray[np.int64]:
    """Search the coset of each syndrome in turn with ``_search_coset_for_min_weight``.

    The arguments are those of ``_find_first_not_within_max_weight``, without ``best``, and with the
    floor of each syndrome in ``floors``.  Return the weight that ``_search_coset_for_min_weight``
    gives each syndrome.
    """
    weights = np.empty(len(syndromes), dtype=np.int64)
    best = np.empty(inverse.shape[1], dtype=np.uint64)
    for index in range(len(syndromes)):
        weight, _, _ = _search_coset_for_min_weight(
            syndromes[index],
            max_weight,
            floors[index],
            best,
            inverse,
            generators,
            pivots,
            planning,
            x_mask,
            tails,
            scale,
        )
        weights[index] = weight
    return weights


@always_compiled(inline=True)
def _search_coset_for_min_weight(
    packed_syndrome: npt.NDArray[np.uint64],
    max_weight: int,
    floor: int,
    best: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    planning: PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    tails: TailFields | None,
    scale: int,
) -> tuple[int, int, int]:
    """Search the coset of a syndrome for its minimum weight, if that is at most ``max_weight``.

    The arguments are those of ``_search_coset``, with a floor of at most ``max_weight``, and
    ``best`` is work space.  A word of weight at most ``floor`` ends the search, which saves work
    for callers to which any such word gives the same answer.

    The search visits the layers of the information sets in the order of ``_search_coset``.  Its
    best weight is that of the word of the syndrome, or of the lightest word within ``max_weight``
    that it has found since.  It stops at its stopping step, the first step whose bound reaches the
    lesser of its best weight and ``max_weight + 1``, since every word that it has not visited then
    weighs at least as much.  This step moves earlier whenever the best weight falls.  The search
    skips each set that adds nothing to the bound of its current stopping step, as
    ``_search_from_syndrome_word`` does.  Such a set adds nothing to the bound of an earlier step
    either, so the bound of the step at which the search stops holds.  Within a step, every word not
    yet visited weighs at least the bound of the step before it.  So the search also stops as soon
    as its best weight is at most that bound, or at most ``floor``.

    The scans weigh Hamming weights, which are ``scale`` times the weights, or CSS weights if
    ``scale`` is zero.  They keep only the words lighter than an incumbent: the skip bound of
    ``max_weight`` (see ``_get_skip_bound``) until the search finds a word within that weight, and
    then the best weight.  Where every word of the coset has the parity of the Hamming weight of the
    best word, no word is one lighter, so the incumbent is one less and the scans keep the same
    words.  This holds in a search of Pauli errors by physical weight, whose embedded words have
    even Hamming weights, and in a search of X or Z parts that rounds to the parity of each coset.
    The lower incumbent lets the tables of a search by tails skip more.

    A search by tails visits the same words in the same order, but weighs a word by its rows below
    the rank of its set and its tail, as ``_search_from_syndrome_word`` does.  In each layer of two
    or more rows of a set with tables, it skips the completions of a prefix whose bound reaches the
    best weight of the scan, which starts at its incumbent: the scan takes the skip bound of
    ``max_weight``, which is at least the incumbent, as its skip bound (see
    ``affine_search.scan_coset_layer_by_tails_with_tables``).  Until the search finds a word within
    ``max_weight``, the incumbent is that skip bound, at which a query by tails skips too.  Then it
    is the Hamming weight of the best word, or one less where every word of the coset has its
    parity, so the search skips no word lighter than the best word.  So the search gives the weight
    of a search of whole words.  For a coset without a word within ``max_weight``, it weighs and
    skips the words that ``_search_from_syndrome_word`` weighs and skips at ``max_weight``.

    Returns:
        The best weight when the search stops, or ``max_weight + 1`` if it found no word within
        ``max_weight``, and the numbers of words weighed after the word of the syndrome and of
        supports skipped.  The best weight is the minimum weight of the coset, unless it is at most
        ``floor``.  Only tests read the numbers.
    """
    # Start from the word of the syndrome, which ends the search if it is within the floor.  Reduce
    # it to the offset of each information set, or, by tails, find the rank of each set and the
    # tail of its offset.
    best_weight, parity = _get_syndrome_word(packed_syndrome, best, inverse, x_mask, scale)
    if best_weight <= floor:
        return best_weight, 0, 0
    bounds = get_planning_table(planning, parity)[:-1]
    num_sets, num_rows, width = generators.shape
    if tails is None:
        offsets = _get_offsets(best, generators, pivots)
    else:
        tail_rows, inverse_tails, completion_tables, layout, block_bits = tails
        ranks, offsets = _get_offset_tails(packed_syndrome, best, pivots, tail_rows, inverse_tails)
        width = tail_rows.shape[2]

    # Find the skip bound and the parity slack of the incumbent, in the units of the scans, which
    # weigh ``factor`` units for each unit of weight.  Only Hamming weights have a parity.  Find
    # also the upper bound ``U`` of the search, in units of weight (Section 3.10 of
    # ``brouwer_zimmermann.md``), and its first stopping step.
    factor = max(1, scale)
    rounds = scale != 0 and len(planning) == 2
    skip_bound = _get_skip_bound(factor * max_weight, parity if rounds else -1)
    slack = 1 if scale == 2 or rounds else 0
    best_units = factor * best_weight
    upper_bound = min(best_weight, max_weight + 1)
    stop_layer, stop_num_completed = get_first_step_reaching(bounds, upper_bound)

    # Visit each layer one information set at a time, up to the stopping step.  The scans take work
    # space for one set.  The search reaches a stopping step by the last layer, whose bounds are
    # ``certificates.NO_WORD_BOUND``, since its upper bound is at most ``max_weight + 1``, which
    # ``BrouwerZimmermannCosetSearch.get_min_weights`` keeps at most ``n + 1``.  The scans keep each
    # lighter word in ``best``, or its support in ``best_support`` by tails, which are only work
    # space, since the search returns no word.
    weighed, skipped = 0, 0
    widths = np.full(1, width, dtype=np.int64)
    support = np.empty(num_rows, dtype=np.int64)
    best_support = np.empty(num_rows, dtype=np.int64)
    prefixes = np.empty((num_rows, 1, width), dtype=np.uint64)
    layer = 0
    while True:
        for set_index in range(num_sets):
            # Weigh the words of this step that are lighter than the incumbent, unless the set adds
            # nothing to the bound of the stopping step, and return once the best weight is at most
            # the floor or the bound of the step before this one.  A scan returns once its best
            # weight, which starts at the incumbent, is at most its cutoff, so the cutoff stays
            # below the incumbent.
            if not certificates.is_unused_set(pivots, set_index, stop_layer, stop_num_completed):
                incumbent = min(best_units - slack, skip_bound)
                step_floor = max(floor, bounds[layer, set_index])
                cutoff = min(factor * step_floor, incumbent - 1)
                set_skipped = 0
                if layer == 0:
                    # The offset is zero on the pivot columns, so its tail has its Hamming weight.
                    weight, ones = _get_weight(offsets[set_index], x_mask, scale)
                    found, count = weight if scale == 0 else ones, 1
                elif tails is None:
                    if scale == 0:
                        found, count = affine_search.scan_coset_layer_by_css_weight(
                            offsets[set_index],
                            generators[set_index : set_index + 1],
                            widths,
                            support[:layer],
                            prefixes,
                            incumbent,
                            cutoff,
                            x_mask,
                            best,
                        )
                    else:
                        found, count = affine_search.scan_coset_layer(
                            offsets[set_index],
                            generators[set_index : set_index + 1],
                            widths,
                            support[:layer],
                            prefixes,
                            incumbent,
                            cutoff,
                            best,
                        )
                elif layer >= 2 and layout[set_index, 1]:
                    # The tables of the set start at ``layout[set_index, 0]`` (see
                    # ``_search_from_syndrome_word``).
                    num_blocks = layout[set_index, 1]
                    block_offsets = layout[set_index, 2 : num_blocks + 3]
                    start = layout[set_index, 0]
                    end = start + (num_rows - 1) * block_offsets[-1]
                    found, count, set_skipped = affine_search.scan_coset_layer_by_tails_with_tables(
                        offsets[set_index],
                        tail_rows[set_index : set_index + 1],
                        ranks[set_index : set_index + 1],
                        widths,
                        support[:layer],
                        prefixes,
                        incumbent,
                        cutoff,
                        skip_bound,
                        completion_tables[start:end],
                        block_offsets,
                        block_bits,
                        best_support,
                    )
                else:
                    found, count = affine_search.scan_coset_layer_by_tails(
                        offsets[set_index],
                        tail_rows[set_index : set_index + 1],
                        ranks[set_index : set_index + 1],
                        widths,
                        support[:layer],
                        prefixes,
                        incumbent,
                        cutoff,
                        best_support,
                    )
                weighed += count
                skipped += set_skipped
                if found < incumbent:
                    best_units, best_weight = found, found // factor
                if best_weight <= step_floor:
                    return best_weight, weighed, skipped

            # Stop at the stopping step, and move that step earlier if the best weight fell.
            new_upper_bound = min(best_weight, max_weight + 1)
            if bounds[layer, set_index + 1] >= new_upper_bound:
                return new_upper_bound, weighed, skipped
            if new_upper_bound < upper_bound:
                upper_bound = new_upper_bound
                stop_layer, stop_num_completed = get_first_step_reaching(bounds, upper_bound)
        layer += 1


@always_compiled(inline=True)
def _get_offsets(
    word: npt.NDArray[np.uint64], generators: npt.NDArray[np.uint64], pivots: npt.NDArray[np.int64]
) -> npt.NDArray[np.uint64]:
    """Reduce a word of a coset to the offset of each information set, which is zero on its pivots.

    ``generators`` and ``pivots`` are those of ``_search_coset``.
    """
    num_sets, _, width = generators.shape
    offsets = np.empty((num_sets, width), dtype=np.uint64)
    for set_index in range(num_sets):
        offsets[set_index] = word
        _reduce_to_offset(offsets[set_index], generators[set_index], pivots[set_index])
    return offsets


@always_compiled(inline=True)
def _get_offset_tails(
    packed_syndrome: npt.NDArray[np.uint64],
    word: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    tails: npt.NDArray[np.uint64],
    inverse_tails: npt.NDArray[np.uint64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint64]]:
    """Find the rank of each information set and the tail of its offset.

    ``word`` is the word of the syndrome ``packed_syndrome``, ``pivots`` are those of
    ``_search_coset``, and ``tails`` and ``inverse_tails`` are those of ``CosetTails``.  The rows of
    a set with pivots on which the word has ones reduce the word to the offset, since every row is
    zero on the pivots of the other rows.  The tail of the word sums the tails of the words with
    unit syndromes (``CosetTails.inverse_tails``) for the set bits of the syndrome.
    """
    num_sets, num_rows, tail_width = tails.shape
    ranks = np.zeros(num_sets, dtype=np.int64)
    offset_tails = np.zeros((num_sets, tail_width), dtype=np.uint64)
    for set_index in range(num_sets):
        xor_selected_rows(packed_syndrome, inverse_tails[set_index], offset_tails[set_index])
        rank = 0
        while rank < num_rows and pivots[set_index, rank] >= 0:
            pivot = pivots[set_index, rank]
            if word[pivot // 64] >> np.uint64(pivot % 64) & np.uint64(1):
                for index in range(tail_width):
                    offset_tails[set_index, index] ^= tails[set_index, rank, index]
            rank += 1
        ranks[set_index] = rank
    return ranks, offset_tails


@always_compiled(inline=True)
def _get_skip_bound(cutoff: int, parity: int) -> int:
    """The least Hamming weight that proves a word of a coset heavier than ``cutoff``.

    It is one more than the greatest Hamming weight at most the cutoff that a word of the coset can
    have.  ``parity`` is the parity of the coset if the search rounds to it, and -1 otherwise.  The
    bound is ``cutoff + 1``, or ``cutoff`` if the search rounds and the words have the parity of
    ``cutoff + 1``.
    """
    if parity >= 0 and (cutoff - parity) % 2:
        return cutoff
    return cutoff + 1


@always_compiled
def _rebuild_word(
    word: npt.NDArray[np.uint64],
    rows: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    support: npt.NDArray[np.int64],
) -> None:
    """Turn a word of a coset into the offset of an information set plus the rows of a support.

    ``rows`` and ``pivots`` are those of the set, as for ``_reduce_to_offset``.
    """
    _reduce_to_offset(word, rows, pivots)
    for row in support:
        for index in range(len(word)):
            word[index] ^= rows[row, index]


@always_compiled(inline=True)
def _reduce_to_offset(
    word: npt.NDArray[np.uint64], rows: npt.NDArray[np.uint64], pivots: npt.NDArray[np.int64]
) -> None:
    """Reduce a word of a coset to the offset of an information set, which is zero on its pivots.

    ``rows`` are the packed rows of the set, and ``pivots`` their pivot columns, padded with -1
    beyond its rank.  The word adds the row of each pivot that it has a one on, since every row is
    zero on the pivots of the other rows.
    """
    for row in range(len(pivots)):
        pivot = pivots[row]
        if pivot < 0:
            break
        if word[pivot // 64] >> np.uint64(pivot % 64) & np.uint64(1):
            for index in range(len(word)):
                word[index] ^= rows[row, index]


@always_compiled(inline=True)
def _get_syndrome_word(
    packed_syndrome: npt.NDArray[np.uint64],
    word: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    x_mask: npt.NDArray[np.uint64],
    scale: int,
) -> tuple[int, int]:
    """Write the sum of the rows of ``inverse`` for the set bits of a packed syndrome into ``word``.

    The sum is an embedded word with that syndrome.  Return its weight (see ``_search_coset``), and
    the parity of its Hamming weight.
    """
    word[:] = 0
    xor_selected_rows(packed_syndrome, inverse, word)
    weight, ones = _get_weight(word, x_mask, scale)
    return weight, ones % 2


@always_compiled
def _get_weight(
    word: npt.NDArray[np.uint64], x_mask: npt.NDArray[np.uint64], scale: int
) -> tuple[int, int]:
    """The weight of the error with an embedded word, and the Hamming weight of the embedded word.

    See ``_search_coset``.
    """
    ones = x_ones = np.int64(0)
    for index in range(len(word)):
        ones += popcount_word(word[index])
        x_ones += popcount_word(word[index] & x_mask[index])
    return int(max(x_ones, ones - x_ones) if scale == 0 else ones // scale), int(ones)
