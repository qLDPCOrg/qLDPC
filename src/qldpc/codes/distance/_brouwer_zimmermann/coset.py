# SPDX-License-Identifier: Apache-2.0

"""Brouwer-Zimmermann coset searches of ``CosetLeaderSearch``: the cosets of one binary code.

Sections cited in this module are those of ``brouwer_zimmermann.md`` and
``brouwer_zimmermann_code.md``, whose Sections 3.9, 3.10, and 4.1 explain these searches.  The batch
loops run on several threads at once, and so can the queries of callers that share a search.  A
change here must keep the requirements on threads under "Keep coset queries reproducible" in
Section 5, and Section 4.1 says which buffers each call writes.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._binary import (
    get_num_words,
    int_to_words,
    popcount_word,
    row_ints,
    words_to_int,
)
from qldpc._compilation import always_compiled

from .. import _threads, _tuning
from .._code_matrices import swap_pauli_halves, symplectic_to_hamming
from .._cosets import (
    get_unit_syndrome_words,
    tabulate_singles,
    xor_selected,
    xor_selected_rows,
)
from .._information_sets import certificates
from .._tuning import SearchTuning
from ..min_weight_bounds import MinWeightBound
from . import affine_instance, affine_search, coset_routes
from .coset_tables import (
    CosetSearchTables,
    PlanningBuffer,
    TailFields,
    get_first_step_reaching,
    get_planning_table,
    get_stopping_step,
)

####################################################################################################
# coset searches


class BrouwerZimmermannCosetSearch:
    """A BZ coset search of a ``CosetLeaderSearch``: a search of the cosets of one binary code.

    The words of a search are packed Paulis on ``n`` qubits, modulo the span of the stabilizers, or
    the X parts of Paulis modulo the span of the X parts of the pure-X stabilizers, or the Z parts
    modulo the span of the Z parts of the pure-Z stabilizers.  A coset is labeled by the syndrome of
    its words, whose bit ``j`` is the parity of a word with row ``j`` of a matrix of checks.  The
    search answers the queries of each maximum weight with the route that ``routes`` choose
    (Section 4.4).

    Args:
        routes: The routes of the search.
        num_qubits: The number of qubits ``n``.
        inverse_rows: Row ``j`` of ``get_unit_syndrome_words`` of the checks, not embedded, as an
            integer: a word whose syndrome has only bit ``j`` set.
        check_columns: Column ``c`` of the checks as an integer: the syndrome of the word with only
            bit ``c`` set.
        css_weight: True to weigh words by their CSS weight, ``max(|x|, |z|)`` for a Pauli error and
            the Hamming weight of an X or Z part, or False to weigh them by their physical weight.
        symplectic: Whether the words are packed Paulis of ``2 n`` bits, rather than X or Z parts of
            ``n`` bits.
        max_css_single_pairs: For a search of Pauli errors by CSS weight, the most pairs of distinct
            syndromes of single-qubit X and Z errors that it tabulates (see ``_css_singles``).
    """

    def __init__(
        self,
        routes: coset_routes.CosetRoutes,
        num_qubits: int,
        inverse_rows: tuple[int, ...],
        check_columns: tuple[int, ...],
        *,
        css_weight: bool,
        symplectic: bool,
        max_css_single_pairs: int = 0,
    ) -> None:
        self._routes = routes
        self._max_css_single_pairs = max_css_single_pairs
        self.css_weight = css_weight
        # The scale chooses the weight that ``_search_coset`` computes: the Hamming weight of an
        # embedded word divided by the scale, which is two for the physical weight of a Pauli error
        # and one for an X or Z part.  A scale of zero instead selects the CSS weight of a Pauli
        # error.
        self.scale = 0 if css_weight and symplectic else 1 if css_weight else 2
        self._num_qubits = num_qubits
        self._symplectic = symplectic
        self._inverse_rows = inverse_rows
        self._num_syndrome_bits = len(inverse_rows)
        # Below maximum weight two, a query bounds a coset without words of weight at most one by
        # two, or by three if the coset is odd in a search of X or Z parts that rounds to parity
        # (Section 4.1).  An odd coset of Pauli errors can hold one of CSS weight two, such as
        # ``YX``.
        num_tables = len(routes.disjoint_tables.planning)
        self._odd_coset_lookup_bound = 3 if not symplectic and num_tables == 2 else 2

        # Map each syndrome of a word of weight zero or one to such a word.  For the CSS weight,
        # ``_weight_one_word`` also pairs single-qubit X and Z errors, to find ``X_i Z_j``.
        self._singles, self._x_singles, self._z_singles = tabulate_singles(
            check_columns, num_qubits, symplectic=symplectic
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

        # Otherwise every word of the coset weighs at least two, or, for an odd coset,
        # ``_odd_coset_lookup_bound``.
        if max_weight < 2:
            word = xor_selected(self._inverse_rows, syndrome)
            return MinWeightBound(
                False, word, self._odd_coset_lookup_bound if word.bit_count() % 2 else 2
            )

        # Search the coset with the route of the maximum weight, keeping the lightest word that the
        # search weighs in ``best``.
        tables, tails = self._get_route(max_weight)
        best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
        packed_syndrome = int_to_words(syndrome, get_num_words(self._num_syndrome_bits))
        lower_bound, _ = _search_coset(
            packed_syndrome, max_weight, best, *tables, tails, self.scale
        )
        # A search that finds a light enough word returns zero, but every word of the coset weighs
        # at least two, since the coset has no word of weight at most one.
        return MinWeightBound(lower_bound <= max_weight, self.unembed(best), max(2, lower_bound))

    def find_first_not_within_max_weight(
        self,
        syndromes: npt.NDArray[np.uint64],
        max_weight: int,
        *,
        num_threads: int = 1,
        split_work: int = 0,
    ) -> tuple[int, MinWeightBound | None]:
        """Find the first syndrome whose coset has no word of weight at most ``max_weight``.

        The syndromes are packed as for ``CosetLeaderSearch.find_first_not_within_max_weight``, and
        the search takes the route of the maximum weight once for all of them.  On more than one
        thread, the calling thread searches the batch alone until its queries have done
        ``split_work`` work, about one unit for each query and each word that it weighs, and
        ``num_threads`` threads then search the rest of it (``_threads.run_blocks``; Section 4.5).
        The result is the same on any number of threads.  ``CosetLeaderSearch`` passes the thread
        options of its tuning.  The defaults search on the calling thread alone.

        Returns:
            The index of the first such syndrome and a ``MinWeightBound`` whose representative is
            the lightest word that the search of that syndrome weighed; or the number of syndromes
            and None if there is none.
        """
        tables, tails = self._get_route(max_weight)
        index, lower_bound, best = _find_first_on_threads(
            np.ascontiguousarray(syndromes),
            max_weight,
            tables,
            tails,
            self.scale,
            num_threads,
            split_work,
        )
        if best is None:
            return index, None
        return index, MinWeightBound(False, self.unembed(best), lower_bound)

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

        The syndromes are packed as for ``CosetLeaderSearch.find_first_not_within_max_weight``.  The
        search of a syndrome also ends at a word of weight at most its floor in ``floors``, which
        must be at most ``max_weight``.  For ``n`` qubits, the search takes the route of
        ``min(max_weight, n)`` once, since no coset has a minimum weight above ``n`` (Section 3.10).
        It looks up no words of weight at most one, which its first layers find.  A batch moves to
        ``num_threads`` threads as in ``find_first_not_within_max_weight``, and each weight is the
        same on any number of threads.

        Returns:
            For each syndrome, the weight of a word of its coset of weight at most its floor if
            there is one, and otherwise the minimum weight of its coset, or ``max_weight + 1`` if
            that is larger (see ``_search_coset_for_min_weight``).
        """
        max_weight = min(max_weight, self._num_qubits)
        tables, tails = self._get_route(max_weight)
        return _find_min_weights_on_threads(
            np.ascontiguousarray(syndromes),
            max_weight,
            np.ascontiguousarray(floors),
            tables,
            tails,
            self.scale,
            num_threads,
            split_work,
        )

    def unembed(self, embedded_word: npt.NDArray[np.uint64]) -> int:
        """Convert a packed embedded word into a word, without the columns that the embedding adds.

        The embedding for the physical weight appends ``x xor z`` to a packed Pauli (see
        ``CosetSearchTables``), and the other embeddings add nothing.
        """
        return words_to_int(embedded_word) & ((1 << self._num_word_bits) - 1)

    def _get_route(self, max_weight: int) -> tuple[CosetSearchTables, TailFields | None]:
        """The tables of the route of a maximum weight, and the fields of its tails, if any.

        The fields are a plain tuple, whose type Numba infers faster than that of ``CosetTails`` at
        each call of a compiled search.
        """
        tables, tails = self._routes.get_route(max_weight)
        return tables, None if tails is None else tails[:]

    @property
    def _num_word_bits(self) -> int:
        """The number of bits of a word, which is ``2 n`` for Paulis and ``n`` for X or Z parts."""
        return 2 * self._num_qubits if self._symplectic else self._num_qubits

    def _weight(self, word: int) -> int:
        """The weight of a word: of its Pauli, or the Hamming weight of an X or Z part."""
        if not self._symplectic:
            return word.bit_count()
        mask = (1 << self._num_qubits) - 1
        x, z = word & mask, (word >> self._num_qubits) & mask
        return max(x.bit_count(), z.bit_count()) if self.css_weight else (x | z).bit_count()

    def _weight_one_word(self, syndrome: int) -> int | None:
        """Find a word of weight zero or one with this syndrome, if there is one."""
        if syndrome in self._singles:
            return self._singles[syndrome]
        if not (self._symplectic and self.css_weight):
            return None
        # CSS weight one also allows X_i Z_j on distinct qubits.  Callers such as the certification
        # of fault propagation query many syndromes, so a table of these errors, built on the first
        # query that needs it, saves a scan of the qubits for each query.
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


def get_pauli_search(
    stabilizers: npt.NDArray[np.uint8], css_weight: bool, tuning: SearchTuning
) -> BrouwerZimmermannCosetSearch:
    """The search of Pauli errors modulo independent commuting stabilizers, from a cache.

    ``stabilizers`` is an ``n x 2n`` matrix in ``[X|Z]`` order of ``n`` independent commuting
    stabilizers on ``n`` qubits, and the syndromes are for its rows.  ``css_weight`` selects the CSS
    weight rather than the physical weight.  The cache is keyed by these rows, the measure of
    weight, and the tuning fields that the setup of the search reads.
    """
    return _get_pauli_search(
        _MatrixKey(stabilizers),
        css_weight,
        _CosetTuning.from_tuning(tuning, paulis_by_css_weight=css_weight),
    )


def get_css_searches(
    x_rows: npt.NDArray[np.uint8], z_rows: npt.NDArray[np.uint8], tuning: SearchTuning
) -> tuple[BrouwerZimmermannCosetSearch, BrouwerZimmermannCosetSearch]:
    """The searches of the X parts and of the Z parts of the errors on a CSS state, from a cache.

    ``x_rows`` holds the X parts of the pure-X stabilizers, and ``z_rows`` the Z parts of the pure-Z
    stabilizers, both in reduced row echelon form, which depends only on the stabilizer group.  The
    X parts of errors are searched modulo the code of the X rows, with the Z rows as checks, and the
    Z parts the other way around.  The cache is keyed by the rows and the tuning fields that the
    setup of the searches reads.
    """
    return _get_css_searches(
        _MatrixKey(x_rows),
        _MatrixKey(z_rows),
        _CosetTuning.from_tuning(tuning, paulis_by_css_weight=False),
    )


class _MatrixKey:
    """A matrix, which a cache can hash and compare by its shape and bytes.

    The key keeps the matrix, from which the cache builds a search that it has not stored yet.  It
    compares a copy of the bytes, and the search copies what it keeps, so later changes to the
    array of the caller reach neither.
    """

    def __init__(self, matrix: npt.NDArray[np.uint8]) -> None:
        self.matrix = matrix
        self._key = (matrix.shape, matrix.tobytes())
        self._hash = hash(self._key)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _MatrixKey) and self._key == other._key


class _CosetTuning(NamedTuple):
    """The fields of the tuning that the setup of a BZ coset search reads, which key its cache.

    The search builds its other families, its tails, and its completion tables after the cache
    stores it, so the fields that configure them key the cache too (Section 4.4).
    """

    disjoint_random_column_orders: int
    max_css_single_pairs: int
    coset_route_minimum_work: int
    max_information_sets: int
    overlapping_attempts_per_set: int
    coset_completion_block_bits: int
    coset_completion_max_table_bytes: int

    @classmethod
    def from_tuning(cls, tuning: SearchTuning, *, paulis_by_css_weight: bool) -> _CosetTuning:
        """The fields of a tuning that the setup of a search reads.

        ``paulis_by_css_weight`` is True for a search of Pauli errors by CSS weight, which alone
        reads ``max_css_single_pairs`` and alone has no completion tables.  The fields that a search
        does not read are zero, so that callers whose tunings differ only in those fields share
        cached searches.
        """
        fields = (getattr(tuning, name) for name in cls._fields)
        coset_tuning = cls(*fields)
        if paulis_by_css_weight:
            return coset_tuning._replace(
                coset_completion_block_bits=0, coset_completion_max_table_bytes=0
            )
        return coset_tuning._replace(max_css_single_pairs=0)


@functools.lru_cache(maxsize=_tuning.PAULI_SEARCH_CACHE_SIZE)
def _get_pauli_search(
    key: _MatrixKey, css_weight: bool, tuning: _CosetTuning
) -> BrouwerZimmermannCosetSearch:
    """The cached Pauli search of ``get_pauli_search``.

    The syndromes depend on the rows of the stabilizer matrix of ``key``, not only on the group that
    they generate, so the rows key the cache.
    """
    stabilizers = key.matrix
    num_qubits = len(stabilizers)
    # The syndrome of a packed Pauli ``e`` is ``checks @ e``, where ``checks`` swaps the X and Z
    # halves of the stabilizers.
    checks = swap_pauli_halves(stabilizers)
    inverse = get_unit_syndrome_words(checks)
    embedded_stabilizers = stabilizers if css_weight else symplectic_to_hamming(stabilizers)
    embedded_inverse = inverse if css_weight else symplectic_to_hamming(inverse)
    # The weight factor is two by either measure of weight, and a search by CSS weight has no routes
    # by tails (Section 4.1).
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
        num_qubits,
        row_ints(inverse),
        check_columns,
        css_weight=css_weight,
        symplectic=True,
        max_css_single_pairs=tuning.max_css_single_pairs,
    )


@functools.lru_cache(maxsize=_tuning.CSS_SEARCH_CACHE_SIZE)
def _get_css_searches(
    x_key: _MatrixKey, z_key: _MatrixKey, tuning: _CosetTuning
) -> tuple[BrouwerZimmermannCosetSearch, BrouwerZimmermannCosetSearch]:
    """The cached classical searches of ``get_css_searches``, of the rows that the keys hold."""
    return (
        _get_classical_search(x_key.matrix, z_key.matrix, tuning),
        _get_classical_search(z_key.matrix, x_key.matrix, tuning),
    )


def _get_classical_search(
    generators: npt.NDArray[np.uint8], checks: npt.NDArray[np.uint8], tuning: _CosetTuning
) -> BrouwerZimmermannCosetSearch:
    """The classical search of X or Z parts modulo the row space of independent generators.

    The row space is the null space of the independent checks, so the syndromes for the checks label
    the cosets.  The search weighs Hamming weights, with a weight factor of one.
    """
    num_qubits = generators.shape[1]
    inverse = get_unit_syndrome_words(checks)
    # Every column counts as the X part, so the CSS weight of a word is its Hamming weight.
    routes = _get_routes(
        generators, inverse, num_qubits, weight_factor=1, tuning=tuning, by_tails=True
    )
    check_columns = row_ints(np.ascontiguousarray(checks.T))
    return BrouwerZimmermannCosetSearch(
        routes, num_qubits, row_ints(inverse), check_columns, css_weight=True, symplectic=False
    )


def _get_routes(
    generators: npt.NDArray[np.uint8],
    inverse: npt.NDArray[np.uint8],
    num_x_columns: int,
    *,
    weight_factor: int,
    tuning: _CosetTuning,
    by_tails: bool,
) -> coset_routes.CosetRoutes:
    """The routes of a coset search of the code of independent embedded generators.

    The routes start from the disjoint information sets of the generators.  ``tuning`` supplies the
    fields that ``coset_routes.CosetRoutes`` and ``affine_instance.get_coset_instance`` take, and
    the other arguments are passed to them.
    """
    instance = affine_instance.get_coset_instance(
        generators,
        random_column_orders=tuning.disjoint_random_column_orders,
        weight_factor=weight_factor,
    )
    return coset_routes.CosetRoutes(
        generators,
        instance,
        inverse,
        num_x_columns,
        minimum_work=tuning.coset_route_minimum_work,
        max_sets=tuning.max_information_sets,
        attempts_per_set=tuning.overlapping_attempts_per_set,
        by_tails=by_tails,
        block_bits=tuning.coset_completion_block_bits,
        max_table_bytes=tuning.coset_completion_max_table_bytes,
    )


def _tabulate_css_singles(
    x_singles: dict[int, int], z_singles: dict[int, int], max_pairs: int
) -> dict[int, int] | None:
    """Tabulate the products of single-qubit X and Z errors by syndrome, if not too many.

    ``x_singles`` and ``z_singles`` map syndromes to single-qubit X and Z errors, and to the
    identity.  Returns None if they make more than ``max_pairs`` pairs, and otherwise the first
    product ``x ^ z`` of each syndrome ``sx ^ sz``, in the order of the two dictionaries.
    """
    if len(x_singles) * len(z_singles) > max_pairs:
        return None
    table: dict[int, int] = {}
    for sx, x in x_singles.items():
        for sz, z in z_singles.items():
            table.setdefault(sx ^ sz, x ^ z)
    return table


####################################################################################################
# batches on threads


def _find_first_on_threads(
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    tables: CosetSearchTables,
    tails: TailFields | None,
    scale: int,
    num_threads: int,
    split_work: int,
) -> tuple[int, int, npt.NDArray[np.uint64] | None]:
    """``_find_first_not_within_max_weight`` on a batch, run by ``_threads.run_blocks``.

    Each worker searches its rows with its own buffer for the lightest word and its own copy of the
    planning buffer (see ``_get_own_planning``).  A syndrome whose coset has no word within
    ``max_weight`` ends the batch, so the least such index, which ``_threads.run_blocks`` returns,
    is that of one thread.

    Returns:
        The index of the first such syndrome, the lower bound of ``_search_coset`` for it, and the
        lightest word of its search; or the number of syndromes, zero, and None if there is none.
    """
    # The lower bound and the lightest word of the query of each syndrome that ends a call of
    # ``search_rows``.
    found: dict[int, tuple[int, npt.NDArray[np.uint64]]] = {}
    width = tables.inverse.shape[1]

    def start_worker() -> _threads.RunRows:
        """Give a worker its own buffers, and return how it searches rows."""
        best = np.empty(width, dtype=np.uint64)
        own_tables = tables._replace(planning=_get_own_planning(tables.planning))

        def search_rows(start: int, end: int, max_work: int) -> tuple[int, bool]:
            """Search rows in order, and record the first without a light enough word."""
            index, lower_bound, ends = _find_first_not_within_max_weight(
                syndromes[start:end], max_weight, max_work, best, *own_tables, tails, scale
            )
            if not ends:
                return start + index, False
            found[start + index] = (int(lower_bound), best.copy())
            return start + index + 1, True

        return search_rows

    index = _threads.run_blocks(len(syndromes), num_threads, split_work, start_worker)
    if index == len(syndromes):
        return index, 0, None
    return index, *found[index]


def _find_min_weights_on_threads(
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    floors: npt.NDArray[np.int64],
    tables: CosetSearchTables,
    tails: TailFields | None,
    scale: int,
    num_threads: int,
    split_work: int,
) -> npt.NDArray[np.int64]:
    """``_find_min_weights`` on a batch, run by ``_threads.run_blocks``.

    Each weight depends only on its syndrome, its floor, and the route, so the weights are those of
    one thread.  Each call of ``_find_min_weights`` has its own buffer for the lightest word, and
    writes only its own rows of the weights.
    """
    weights = np.empty(len(syndromes), dtype=np.int64)

    def search_rows(start: int, end: int, max_work: int) -> tuple[int, bool]:
        """Search rows in order, and write their weights."""
        num_done = _find_min_weights(
            syndromes[start:end],
            max_weight,
            floors[start:end],
            max_work,
            weights[start:end],
            *tables,
            tails,
            scale,
        )
        return start + num_done, False

    _threads.run_blocks(len(syndromes), num_threads, split_work, lambda: search_rows)
    return weights


def _get_own_planning(planning: PlanningBuffer) -> PlanningBuffer:
    """A copy of a planning buffer whose cached stopping steps are the initial step ``(0, 1)``.

    ``_find_first_not_within_max_weight`` caches stopping steps in its planning buffer, and runs
    without the global interpreter lock, so no two of its calls that run at once may share one.
    Resetting the cached steps of the copy makes it independent of what another thread writes to
    the original while it is copied (Section 4.1).  The copy holds a few small tables.
    """
    own_planning = planning.copy()
    own_planning[:, -1, :2] = (0, 1)
    return own_planning


####################################################################################################
# compiled coset search


@always_compiled(nogil=True)
def _find_first_not_within_max_weight(
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    max_work: int,
    best: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    planning: PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    tails: TailFields | None,
    scale: int,
) -> tuple[int, int, bool]:
    """Search the coset of each syndrome in turn with ``_search_coset``, within a budget of work.

    The work of a query is one, plus the number of words that it weighs after the word of its
    syndrome.  Return the index of the first syndrome whose coset has no word within
    ``max_weight``, the lower bound that ``_search_coset`` returns for it, and True, with the
    lightest word that it weighed left in ``best``.  Otherwise return the number of syndromes
    searched, zero, and False: all of them, or those up to the first at which the work of this call
    reaches ``max_work``.  Threads run it at once, so no two calls that run at once
    may share ``best`` or ``planning``, whose cached stopping steps it writes (Section 4.1).
    """
    work = 0
    for index in range(len(syndromes)):
        lower_bound, weighed = _search_coset(
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
            return index, lower_bound, True
        work += 1 + weighed
        if work >= max_work:
            return index + 1, 0, False
    return len(syndromes), 0, False


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
) -> tuple[int, int]:
    """Search the coset of a syndrome for a word of weight at most ``max_weight``.

    Machine word ``j`` of ``packed_syndrome`` holds bits ``64 j`` and up of the syndrome.  The
    arguments from ``inverse`` to ``x_mask`` are the fields of a ``CosetSearchTables``, and a query
    may update the cached stopping steps of ``planning``.  A word weighs its CSS weight if ``scale``
    is zero, and otherwise the Hamming weight of its embedded word divided by ``scale``: two for the
    physical weight of a Pauli error, and one for an X or Z part.  ``tails`` is None, or the fields
    of the ``CosetTails`` of a route by tails (Section 4.4); with tails, ``scale`` is one or two.

    The search starts from the word of the syndrome, and visits layer zero of the sets that it uses,
    then layer one, and so on, up to its stopping step, until it finds a word within ``max_weight``
    (Sections 3.1 and 4.1).  By tails, it skips only words that tables prove beyond ``max_weight``,
    so it finds the same first word as without tails, if any (Section 3.9, "Coset searches").

    Returns:
        A lower bound on the weight of every word in the coset: zero if ``best``, which holds the
        lightest word that the search weighed, is within ``max_weight``, and otherwise a bound
        above ``max_weight`` (see ``_search_from_syndrome_word``).  Also the number of words that
        it weighed after the word of the syndrome, which measures its work.
    """
    best_weight, parity = _get_syndrome_word(packed_syndrome, best, inverse, x_mask, scale)
    if best_weight <= max_weight:
        return 0, 0
    table = get_planning_table(planning, parity)
    stop_layer, stop_num_completed, stop_bound = get_stopping_step(table, max_weight)
    lower_bound, weighed, _ = _search_from_syndrome_word(
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
        _get_rounding_parity(planning, scale, parity),
    )
    return lower_bound, weighed


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
    ``get_stopping_step``.  ``parity`` is the parity of the Hamming weights of the coset if the
    search weighs Hamming weights and rounds to it (see ``get_search_tables``), and otherwise -1.
    The search skips the unused sets of ``certificates.plan_steps``, which add nothing to the bound
    of the stopping step (Section 3.6).  By tails, it weighs words as
    ``affine_search.scan_coset_layer_by_tails`` does, skips completions at the skip bound that
    ``_get_skip_bound`` gives for ``scale * max_weight``, and writes the lightest word into ``best``
    before it returns.

    Returns:
        A lower bound on the weight of every word in the coset (Section 3.9, "Coset searches"):
        zero if the search finds a word within ``max_weight``; otherwise, if it skipped a word, the
        least weight beyond ``max_weight`` that a word of the coset can have; and otherwise the
        lesser of the weight of the lightest word that it weighed and the bound of the stopping
        step.  Also the number of words weighed after the word of the syndrome, which measures
        the work of the query for ``split_work``, and the number of supports skipped.  Only tests
        read the number of supports skipped, which they cannot count by wrapping compiled calls.
    """
    # Reduce the word of the syndrome to the offset of each set, or, by tails, find the rank of each
    # set and the tail of its offset.
    num_sets, num_rows, _ = generators.shape
    offsets, width, ranks = _get_set_offsets(packed_syndrome, best, generators, pivots, tails)

    # The scans weigh ``factor`` units for each unit of weight: Hamming weights, which are ``scale``
    # times the weights, or CSS weights if ``scale`` is zero.
    factor = max(1, scale)
    best_units, cutoff = factor * best_weight, factor * max_weight
    skip_bound = _get_skip_bound(cutoff, parity)

    # Visit each layer one set at a time, up to the stopping step.  The scans by tails keep the
    # support of each lighter word in ``best_support``, and the others the word in ``best``.
    best_set, best_layer = -1, 0
    weighed, skipped = 0, 0
    support = np.empty(num_rows, dtype=np.int64)
    best_support = np.empty(num_rows, dtype=np.int64)
    prefixes = np.empty((num_rows, width), dtype=np.uint64)
    for layer in range(stop_layer + 1):
        for set_index in range(num_sets if layer < stop_layer else stop_num_completed):
            if certificates.is_unused_set(pivots, set_index, stop_layer, stop_num_completed):
                continue
            found, count, set_skipped = _scan_step(
                set_index,
                layer,
                offsets,
                generators,
                ranks,
                width,
                support[:layer],
                prefixes,
                best_units,
                cutoff,
                skip_bound,
                x_mask,
                best,
                best_support,
                tails,
                scale,
            )
            if layer == 0 and tails is None and found < best_units:
                best[:] = offsets[set_index]
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

    # After a skip, the least weight beyond the maximum weight that a word of the coset can have is
    # the skip bound, rounded up to the parity of the coset, divided by ``factor``, and rounded up.
    if best_units <= cutoff:
        lower_bound = 0
    elif skipped:
        rounded = skip_bound + (skip_bound - parity) % 2 if parity >= 0 else skip_bound
        lower_bound = -(-rounded // factor)
    else:
        lower_bound = min(best_units // factor, stop_bound)
    return lower_bound, weighed, skipped


@always_compiled(nogil=True)
def _find_min_weights(
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    floors: npt.NDArray[np.int64],
    max_work: int,
    weights: npt.NDArray[np.int64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    planning: PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    tails: TailFields | None,
    scale: int,
) -> int:
    """Search the coset of each syndrome in turn with ``_search_coset_for_min_weight``.

    The arguments are those of ``_find_first_not_within_max_weight``, without ``best``, with the
    floor of each syndrome in ``floors``, and with ``weights``, into whose row ``i`` the search
    writes the weight of syndrome ``i``.  Returns the number of syndromes searched: all of them, or
    those up to the first at which the work of this call reaches ``max_work``.  Each call has its
    own ``best``, writes only ``weights``, and caches no stopping steps, so calls that run at once
    may share every argument but ``weights``.
    """
    work = 0
    best = np.empty(inverse.shape[1], dtype=np.uint64)
    for index in range(len(syndromes)):
        weight, weighed, _ = _search_coset_for_min_weight(
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
        work += 1 + weighed
        if work >= max_work:
            return index + 1
    return len(syndromes)


@always_compiled
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

    The arguments are those of ``_search_coset``, with ``best`` as work space, and a word of weight
    at most ``floor``, which must be at most ``max_weight``, ends the search.  The search keeps
    ``U``, the lesser of ``max_weight + 1`` and its best weight, and stops after the first step
    whose bound reaches ``U``, or within a step once ``U`` is at most ``floor`` or the bound of the
    step before.  Its scans keep only the words lighter than an incumbent, and its tables skip only
    at the incumbent or at the weight of a lighter word that a scan has found.  So it gives the
    weight that a search of whole words gives (Section 3.10).

    Returns:
        ``U`` when the search stops: the lesser of the minimum weight of the coset and
        ``max_weight + 1``, unless ``U`` is at most ``floor``.  Also the number of words weighed
        after the word of the syndrome, which measures the work of the search for ``split_work``,
        and the number of supports skipped, which only tests read, as for
        ``_search_from_syndrome_word``.
    """
    # Start from the word of the syndrome, which ends the search if it is within the floor, and
    # reduce it as ``_search_from_syndrome_word`` does.
    best_weight, parity = _get_syndrome_word(packed_syndrome, best, inverse, x_mask, scale)
    if best_weight <= floor:
        return best_weight, 0, 0
    bounds = get_planning_table(planning, parity)[:-1]
    num_sets, num_rows, _ = generators.shape
    offsets, width, ranks = _get_set_offsets(packed_syndrome, best, generators, pivots, tails)

    # Find, in the units of the scans, the skip bound and the slack, one where the words of the
    # coset share a parity, by which the incumbent falls below the best weight (Section 3.10).  Find
    # also the upper bound ``U`` of the search, in units of weight, with its first stopping step.
    factor = max(1, scale)
    rounding_parity = _get_rounding_parity(planning, scale, parity)
    skip_bound = _get_skip_bound(factor * max_weight, rounding_parity)
    slack = 1 if scale == 2 or rounding_parity >= 0 else 0
    best_units = factor * best_weight
    upper_bound = min(best_weight, max_weight + 1)
    stop_layer, stop_num_completed = get_first_step_reaching(bounds, upper_bound)

    # Visit each layer one set at a time, up to the stopping step, with ``best`` and
    # ``best_support`` as work space.  The loop ends by the last layer, whose bounds
    # ``certificates.NO_WORD_BOUND`` exceed ``U``, since
    # ``BrouwerZimmermannCosetSearch.get_min_weights`` keeps ``max_weight`` at most ``n``.
    weighed, skipped = 0, 0
    support = np.empty(num_rows, dtype=np.int64)
    best_support = np.empty(num_rows, dtype=np.int64)
    prefixes = np.empty((num_rows, width), dtype=np.uint64)
    layer = 0
    while True:
        for set_index in range(num_sets):
            # Weigh the words of this step lighter than the incumbent, unless the set is unused, and
            # return once the best weight is at most the floor or the bound of the step before.  A
            # scan returns once its best weight, which starts at the incumbent, is at most its
            # cutoff, so the cutoff must stay below the incumbent.
            if not certificates.is_unused_set(pivots, set_index, stop_layer, stop_num_completed):
                incumbent = min(best_units - slack, skip_bound)
                step_floor = max(floor, bounds[layer, set_index])
                cutoff = min(factor * step_floor, incumbent - 1)
                found, count, set_skipped = _scan_step(
                    set_index,
                    layer,
                    offsets,
                    generators,
                    ranks,
                    width,
                    support[:layer],
                    prefixes,
                    incumbent,
                    cutoff,
                    skip_bound,
                    x_mask,
                    best,
                    best_support,
                    tails,
                    scale,
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
def _get_rounding_parity(planning: PlanningBuffer, scale: int, parity: int) -> int:
    """The parity of the Hamming weights of a coset if its search rounds to it, and otherwise -1.

    A search whose buffer has a table for each parity reads the table of its coset's parity.  Only
    a search of Hamming weights (nonzero ``scale``) also rounds to that parity: for its skip bound,
    and for its bound after a skip (``_search_from_syndrome_word``) or its slack
    (``_search_coset_for_min_weight``).  The arguments are those of ``_search_coset``, and the
    parity of the coset.
    """
    return parity if scale != 0 and len(planning) == 2 else -1


@always_compiled(inline=True)
def _get_set_offsets(
    packed_syndrome: npt.NDArray[np.uint64],
    word: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    tails: TailFields | None,
) -> tuple[npt.NDArray[np.uint64], int, npt.NDArray[np.int64]]:
    """Reduce the word of a syndrome to the offset of each set, or, by tails, to each offset's tail.

    ``word`` is the word of the syndrome ``packed_syndrome``, and the other arguments are those of
    ``_search_coset``.  Returns the offsets or their tails, their number of packed words, and, by
    tails, the rank of each set.  Without tails, the scans read no ranks, so the pivots of the first
    set fill their place.
    """
    if tails is None:
        return _get_offsets(word, generators, pivots), generators.shape[2], pivots[0]
    tail_rows, inverse_tails, _, _, _ = tails
    ranks, offsets = _get_offset_tails(packed_syndrome, word, pivots, tail_rows, inverse_tails)
    return offsets, tail_rows.shape[2], ranks


# Inlined although it is not small: both coset searches call it for each set of each step, with many
# arguments, several of them arrays.
@always_compiled(inline=True)
def _scan_step(
    set_index: int,
    layer: int,
    offsets: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    ranks: npt.NDArray[np.int64],
    width: int,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    incumbent: int,
    cutoff: int,
    skip_bound: int,
    x_mask: npt.NDArray[np.uint64],
    representative: npt.NDArray[np.uint64],
    best_support: npt.NDArray[np.int64],
    tails: TailFields | None,
    scale: int,
) -> tuple[int, int, int]:
    """Weigh the words of a set in a layer, with the scan of ``affine_search`` that fits the route.

    ``offsets``, ``width``, and ``ranks`` are those of ``_get_set_offsets``, ``support`` has
    ``layer`` entries, and the other arguments are those of the scans and of ``_search_coset``.  In
    a layer above zero, a scan of whole words writes the lightest word that it finds into
    ``representative``, and a scan by tails its support into ``best_support``.

    Returns:
        In the units of the scans, the weight of the offset in layer zero, and in a later layer the
        weight of the lightest word that the scan found lighter than ``incumbent``, or ``incumbent``
        if there is none.  Also the numbers of words weighed and of supports skipped.
    """
    if layer == 0:
        # The offset is zero on the pivot columns, so its tail has its Hamming weight.
        weight, ones = _get_weight(offsets[set_index], x_mask, scale)
        return weight if scale == 0 else ones, 1, 0
    if tails is None:
        if scale == 0:
            found, count = affine_search.scan_coset_layer_by_css_weight(
                offsets[set_index],
                generators[set_index],
                width,
                support,
                prefixes,
                incumbent,
                cutoff,
                x_mask,
                representative,
            )
        else:
            found, count = affine_search.scan_coset_layer(
                offsets[set_index],
                generators[set_index],
                width,
                support,
                prefixes,
                incumbent,
                cutoff,
                representative,
            )
        return found, count, 0
    tail_rows, _, completion_tables, layout, block_bits = tails
    if layer >= 2 and layout[set_index, 1]:
        # The tables of the set start at ``layout[set_index, 0]``, with a row of tables for each row
        # of the set but the last.
        num_blocks = layout[set_index, 1]
        block_offsets = layout[set_index, 2 : num_blocks + 3]
        start = layout[set_index, 0]
        end = start + (generators.shape[1] - 1) * block_offsets[-1]
        return affine_search.scan_coset_layer_by_tails_with_tables(
            offsets[set_index],
            tail_rows[set_index],
            ranks[set_index],
            width,
            support,
            prefixes,
            incumbent,
            cutoff,
            skip_bound,
            completion_tables[start:end],
            block_offsets,
            block_bits,
            best_support,
        )
    found, count = affine_search.scan_coset_layer_by_tails(
        offsets[set_index],
        tail_rows[set_index],
        ranks[set_index],
        width,
        support,
        prefixes,
        incumbent,
        cutoff,
        best_support,
    )
    return found, count, 0


@always_compiled(inline=True)
def _get_offsets(
    word: npt.NDArray[np.uint64], generators: npt.NDArray[np.uint64], pivots: npt.NDArray[np.int64]
) -> npt.NDArray[np.uint64]:
    """Reduce a word of a coset to the offset of each set, which is zero on its pivots.

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
    """Find the rank of each set and the tail of its offset.

    ``word`` is the word of the syndrome ``packed_syndrome``, ``pivots`` are those of
    ``_search_coset``, and ``tails`` and ``inverse_tails`` are those of ``CosetTails``.  The tail of
    ``word`` is the sum of the inverse tails for the set bits of the syndrome.  The tail of an
    offset adds to it the tails of the rows of the set whose pivots carry ones of ``word``, since
    these rows reduce ``word`` to the offset (see ``_reduce_to_offset``).
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

    ``parity`` is the parity of the coset if the search rounds to it, and -1 otherwise.  The bound
    is ``cutoff + 1``, or ``cutoff`` if the search rounds and the words have the parity of
    ``cutoff + 1`` (``h(ω)`` of Section 3.9, "Coset searches").
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
    """Turn a word of a coset into the offset of a set plus the rows of a support.

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
    """Reduce a word of a coset to the offset of a set, which is zero on its pivots.

    ``rows`` are the packed rows of the set, and ``pivots`` their pivot columns, padded with -1
    beyond its rank.  The word adds the row of each pivot on which it has a one, since every row is
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

    Return the weight of this word of the syndrome (see ``_search_coset``), and the parity of its
    Hamming weight.
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
