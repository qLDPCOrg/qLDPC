# SPDX-License-Identifier: Apache-2.0

"""Brouwer-Zimmermann coset searches of ``CosetLeaderSearch``: the cosets of one binary code.

Sections cited in this module are those of ``brouwer_zimmermann.md``, whose Sections 3.9, 3.10,
and 4.1 explain these searches.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits, popcount_word, unpack_bits
from qldpc._compilation import always_compiled

from .. import _tuning
from .._binary import swap_pauli_halves, symplectic_to_hamming
from .._cosets import (
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
        css_weight: True to weigh words by their CSS weight, ``max(|x|, |z|)`` for a Pauli error and
            the Hamming weight of an X or Z part, or False to weigh them by their physical weight.
        num_qubits: The number of qubits ``n``.
        inverse_rows: Row ``j`` of ``get_unit_syndrome_words`` of the checks, not embedded, as an
            integer: a word whose syndrome has only bit ``j`` set.
        check_columns: Column ``c`` of the checks as an integer: the syndrome of the word with only
            bit ``c`` set.
        symplectic: Whether the words are packed Paulis of ``2 n`` bits, rather than X or Z parts of
            ``n`` bits.
        max_css_single_pairs: For a search of Pauli errors by CSS weight, the most pairs of distinct
            syndromes of single-qubit X and Z errors that it tabulates (see ``_css_singles``).
    """

    def __init__(
        self,
        routes: coset_routes.CosetRoutes,
        css_weight: bool,
        num_qubits: int,
        inverse_rows: tuple[int, ...],
        check_columns: tuple[int, ...],
        *,
        symplectic: bool,
        max_css_single_pairs: int = 0,
    ) -> None:
        self._routes = routes
        self._max_css_single_pairs = max_css_single_pairs
        self.css_weight = css_weight
        # The scale chooses the weight that ``_search_coset`` computes: zero for the CSS weight of a
        # Pauli error, two for its physical weight, and one for an X or Z part.
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
        packed_syndrome = int_to_words(syndrome, max(1, (self._num_syndrome_bits + 63) // 64))
        lower_bound = _search_coset(packed_syndrome, max_weight, best, *tables, tails, self.scale)
        # A search that finds a light enough word returns zero, but every word of the coset weighs
        # at least two, since the coset has no word of weight at most one.
        return MinWeightBound(lower_bound <= max_weight, self.unembed(best), max(2, lower_bound))

    def find_first_not_within_max_weight(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> tuple[int, MinWeightBound | None]:
        """Find the first syndrome whose coset has no word of weight at most ``max_weight``.

        The syndromes are packed as for ``CosetLeaderSearch.find_first_not_within_max_weight``, and
        the search takes the route of the maximum weight once for all of them.

        Returns:
            The index of the first such syndrome and a ``MinWeightBound`` whose representative is
            the lightest word that the search weighed; or the number of syndromes and None if there
            is none.
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
        must be at most ``max_weight``.  For ``n`` qubits, the search takes the route of
        ``min(max_weight, n)`` once, since no coset has a minimum weight above ``n`` (Section 3.10).
        It looks up no words of weight at most one, which its first layers find.

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

        The fields are a plain tuple, whose type Numba infers faster than that of ``CosetTails`` at
        each call of a compiled search.
        """
        tables, tails = self._routes.get_route(max_weight)
        return tables, None if tails is None else tails[:]

    @property
    def _num_word_bits(self) -> int:
        """The number of bits of a word: ``2 n`` for Paulis, and ``n`` for X or Z parts."""
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
        _StabilizerKey(stabilizers),
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
        x_rows.shape[1],
        len(x_rows),
        pack_bits(x_rows).tobytes(),
        pack_bits(z_rows).tobytes(),
        _CosetTuning.from_tuning(tuning, paulis_by_css_weight=False),
    )


class _StabilizerKey:
    """A stabilizer matrix, which a cache can hash and compare by its bytes.

    The key keeps the matrix, from which the cache builds a search that it has not stored yet.  It
    compares a copy of the bytes, and the search copies what it keeps, so later changes to the
    array of the caller reach neither.
    """

    def __init__(self, stabilizers: npt.NDArray[np.uint8]) -> None:
        self.stabilizers = stabilizers
        self._key = (stabilizers.shape, stabilizers.tobytes())
        self._hash = hash(self._key)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StabilizerKey) and self._key == other._key


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
    completion_bound_max_table_bytes: int
    coset_completion_block_bits: int

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
                completion_bound_max_table_bytes=0, coset_completion_block_bits=0
            )
        return coset_tuning._replace(max_css_single_pairs=0)


@functools.lru_cache(maxsize=_tuning.PAULI_SEARCH_CACHE_SIZE)
def _get_pauli_search(
    key: _StabilizerKey, css_weight: bool, tuning: _CosetTuning
) -> BrouwerZimmermannCosetSearch:
    """The cached Pauli search of ``get_pauli_search``.

    The syndromes depend on the rows of the stabilizer matrix of ``key``, not only on the group that
    they generate, so the rows key the cache.
    """
    stabilizers = key.stabilizers
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
        css_weight,
        num_qubits,
        row_ints(inverse),
        check_columns,
        symplectic=True,
        max_css_single_pairs=tuning.max_css_single_pairs,
    )


@functools.lru_cache(maxsize=_tuning.CSS_SEARCH_CACHE_SIZE)
def _get_css_searches(
    num_qubits: int, num_x_rows: int, x_rows: bytes, z_rows: bytes, tuning: _CosetTuning
) -> tuple[BrouwerZimmermannCosetSearch, BrouwerZimmermannCosetSearch]:
    """The cached classical searches of ``get_css_searches``.

    ``x_rows`` and ``z_rows`` are the rows of ``get_css_searches``, packed into bytes, which
    ``lru_cache`` can hash.  ``x_rows`` has ``num_x_rows`` rows, and ``z_rows`` has the other
    ``num_qubits - num_x_rows``.
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
        routes, True, num_qubits, row_ints(inverse), check_columns, symplectic=False
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
        max_table_bytes=tuning.completion_bound_max_table_bytes,
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

    Return the index of the first syndrome whose coset has no word within ``max_weight``, and the
    lower bound that ``_search_coset`` returns for it, with the lightest word that it weighed left
    in ``best``; or the number of syndromes and zero if there is none.
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
        above ``max_weight`` (see ``_search_from_syndrome_word``).
    """
    best_weight, parity = _get_syndrome_word(packed_syndrome, best, inverse, x_mask, scale)
    if best_weight <= max_weight:
        return 0
    table = get_planning_table(planning, parity)
    stop_layer, stop_num_completed, stop_bound = get_stopping_step(table, max_weight)
    # A search whose buffer has a table for each parity reads that of the parity of its coset, but
    # only a search of Hamming weights passes the parity on, for its skip bound and its bound after
    # a skip.
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
        step.  Also the numbers of words weighed after the word of the syndrome and of supports
        skipped, which only tests read.
    """
    # Reduce the word of the syndrome to the offset of each set, or, by tails, find the rank of each
    # set and the tail of its offset.
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
                        generators[set_index],
                        width,
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
                        generators[set_index],
                        width,
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
                    tail_rows[set_index],
                    ranks[set_index],
                    width,
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
                    tail_rows[set_index],
                    ranks[set_index],
                    width,
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
    floor of each syndrome in ``floors``.
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

    The arguments are those of ``_search_coset``, with ``best`` as work space, and a word of weight
    at most ``floor``, which must be at most ``max_weight``, ends the search.  The search keeps
    ``U``, the lesser of ``max_weight + 1`` and its best weight, and stops after the first step
    whose bound reaches ``U``, or within a step once ``U`` is at most ``floor`` or the bound of the
    step before.  Its scans keep only the words lighter than an incumbent, and its tables skip only
    at the incumbent or at the weight of a lighter word that a scan has found.  So it gives the
    weight that a search of whole words gives (Section 3.10).

    Returns:
        ``U`` when the search stops: the lesser of the minimum weight of the coset and
        ``max_weight + 1``, unless ``U`` is at most ``floor``.  Also the numbers of words weighed
        after the word of the syndrome and of supports skipped, which only tests read.
    """
    # Start from the word of the syndrome, which ends the search if it is within the floor, and
    # reduce it as ``_search_from_syndrome_word`` does.
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

    # Find, in the units of the scans, the skip bound and the slack, one where the words of the
    # coset share a parity, by which the incumbent falls below the best weight (Section 3.10).  Find
    # also the upper bound ``U`` of the search, in units of weight, with its first stopping step.
    factor = max(1, scale)
    rounds = scale != 0 and len(planning) == 2
    skip_bound = _get_skip_bound(factor * max_weight, parity if rounds else -1)
    slack = 1 if scale == 2 or rounds else 0
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
                set_skipped = 0
                if layer == 0:
                    # The offset is zero on the pivot columns, so its tail has its Hamming weight.
                    weight, ones = _get_weight(offsets[set_index], x_mask, scale)
                    found, count = weight if scale == 0 else ones, 1
                elif tails is None:
                    if scale == 0:
                        found, count = affine_search.scan_coset_layer_by_css_weight(
                            offsets[set_index],
                            generators[set_index],
                            width,
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
                            generators[set_index],
                            width,
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
                        tail_rows[set_index],
                        ranks[set_index],
                        width,
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
                        tail_rows[set_index],
                        ranks[set_index],
                        width,
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
