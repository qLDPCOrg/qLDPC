# SPDX-License-Identifier: Apache-2.0

"""Exact minimum-weight queries for syndrome classes of Pauli errors on a stabilizer state.

The Pauli errors with a given syndrome form a coset of the stabilizer group, and a lightest error in
it is a coset leader.  ``CosetLeaderSearch`` decides whether the coset leaders of a syndrome class
weigh at most a given maximum weight, with Brouwer-Zimmermann searches of cosets of binary codes, or
with connected-support searches that start from the syndrome (see ``connected_support.py``).
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, get_args

import numpy as np
import numpy.typing as npt

from qldpc import math
from qldpc._binary import (
    get_kernel,
    pack_bits,
    popcount_word,
    row_reduce_binary,
    trailing_zeros_word,
    unpack_bits,
)
from qldpc._compilation import always_compiled

from .. import _tuning
from .._binary import as_binary_matrix, assert_binary, symplectic_to_hamming
from .._tuning import get_tuning
from . import affine_instance, affine_search, connected_support, information_sets

# The method of a coset-leader search.
CosetSearchMethod = Literal["brouwer_zimmermann", "connected_support"]

# The planning buffer of a coset search: the bounds of its steps, and one more row that caches a
# stopping step (see ``_get_planning``).
_PlanningBuffer = npt.NDArray[np.int64]

####################################################################################################
# coset-leader minimum-weight bounds


@dataclass(frozen=True)
class MinWeightBound:
    """The answer to a query about the minimum weight of a coset.

    A query of ``CosetLeaderSearch`` asks whether a syndrome class of Pauli errors, which is a coset
    of the stabilizer group, has an error of weight at most ``max_weight``.  A query of one of its
    coset searches asks the same about a coset of the code that the search covers.

    Attributes:
        within_max_weight: Whether the coset has an element of weight at most ``max_weight``.
        representative: An element of the coset: a packed Pauli (see ``CosetLeaderSearch``) for a
            query of the oracle, and a word of the search (a packed Pauli, or an X or Z part) for a
            query of a coset search.  If ``within_max_weight`` is True, its weight is at most
            ``max_weight``.
        lower_bound: A lower bound on the minimum weight of the coset.  If ``within_max_weight`` is
            False, it exceeds ``max_weight``.
    """

    within_max_weight: bool
    representative: int
    lower_bound: int


class CosetLeaderSearch:
    """Minimum-weight oracle for Pauli errors modulo the stabilizers of a pure stabilizer state.

    The syndrome of a Pauli error on ``n`` qubits says which of the ``n`` given stabilizer
    generators, the rows of ``stabilizers``, it anticommutes with.  Two errors have the same
    syndrome exactly when they differ by a stabilizer, so the errors with a given syndrome form a
    syndrome class of equivalent errors, which is a coset of the stabilizer group.  A lightest error
    of a syndrome class is a coset leader, and the oracle determines whether the coset leaders of a
    syndrome class weigh at most a given maximum weight.

    A Pauli is packed into an integer with its X part ``x`` in the low ``n`` bits and its Z part
    ``z`` in the next ``n`` bits, and a syndrome is packed into an integer with bit ``j`` for
    stabilizer ``j``.  The weight of an error is either its physical weight, the number of qubits
    that it acts on, or its CSS weight ``max(|x|, |z|)``.

    The oracle answers a query with one or two coset searches.  By default, each searches the
    cosets of a binary code with the Brouwer-Zimmermann (BZ) algorithm (see
    ``brouwer_zimmermann.md``), as ``_search_coset`` describes, and ``get_min_weight_bound`` first
    looks up words of weight at most one in a table.

    - In general, one Pauli search covers the Pauli errors themselves, modulo the stabilizers.
      Its syndromes are for the given stabilizers, so it is cached for the most recently used
      stabilizer matrices (see ``_tuning.PAULI_SEARCH_CACHE_SIZE``).
    - For the CSS weight on a CSS state, two classical searches cover the X parts and the Z parts
      of the errors separately, on ``n`` columns each rather than ``2 n``.  A pure-X stabilizer has
      no Z part, a pure-Z stabilizer has no X part, and a stabilizer state is CSS if pure-X and
      pure-Z stabilizers generate its stabilizer group.  The X parts of the pure-X stabilizers span
      a binary code ``C_X``, and the Z parts of the pure-Z stabilizers a binary code ``C_Z``.
      Multiplying an error by a stabilizer adds a word of ``C_X`` to its X part and a word of
      ``C_Z`` to its Z part, independently, so the smallest CSS weight of an error with a given
      syndrome is the larger of the smallest Hamming weights in the cosets ``x + C_X`` and
      ``z + C_Z``.  The coset of an X part is labeled by its syndrome for the pure-Z stabilizers,
      and that of a Z part by its syndrome for the pure-X stabilizers, and the oracle maps a
      syndrome for the given stabilizers to these two syndromes.  The two searches depend only on
      the stabilizer group, so they are cached for the most recently used groups (see
      ``_tuning.CSS_SEARCH_CACHE_SIZE``), and oracles for different generating sets of one state
      share them.

    With ``method="connected_support"``, each coset search is instead the search of
    ``connected_support.SupportSearch`` from the syndrome of a query, through the unsatisfied
    stabilizers for Pauli errors, the pure-Z stabilizers for X parts, and the pure-X stabilizers
    for Z parts (Section 3.8 of ``connected_support.md``).  It is fast if its checks are sparse.
    Given stabilizers, such as row-reduced ones, can be dense, so it first makes its checks sparser
    by adding them to each other, which can fall short of the sparsest generators.  It uses no
    tuning field.  The CSS weight is not a sum over qubits, so it weighs errors by CSS weight only
    on CSS states, whose X and Z parts it searches separately.  Its bounds are weaker.  A query
    within ``max_weight`` gives a lower bound of one, or zero for an empty syndrome.  A query with
    no error within ``max_weight`` gives ``max_weight + 1``, or, for a negative ``max_weight``, one,
    or zero for an empty syndrome.  The BZ searches often give higher bounds, which spare queries
    to consumers that raise a threshold to each bound.

    Args:
        stabilizers: A binary ``n x 2n`` matrix in ``[X|Z]`` order whose rows are independent
            commuting stabilizers.
        css_weight: True to weigh an error by its CSS weight, or False to weigh it by its physical
            weight.
        method: ``"brouwer_zimmermann"`` (the default) or ``"connected_support"``, the search of
            the cosets.

    The oracle reads the active tuning (see ``_tuning``) when it is built.  Its field
    ``disjoint_random_column_orders`` configures every BZ search of the oracle, and
    ``max_css_single_pairs`` the BZ searches by CSS weight that are not split into two classical
    searches.

    Raises:
        ValueError: If the stabilizers are not ``n`` independent commuting stabilizers, if the
            method is unknown, or if the connected-support method would weigh errors on a state that
            is not CSS by their CSS weight.
    """

    def __init__(
        self,
        stabilizers: npt.ArrayLike,
        *,
        css_weight: bool = False,
        method: CosetSearchMethod = "brouwer_zimmermann",
    ) -> None:
        if method not in get_args(CosetSearchMethod):
            raise ValueError(
                f"Unknown coset-leader search method {method!r}; choose from 'brouwer_zimmermann'"
                " or 'connected_support'"
            )
        tuning = get_tuning()
        # Exactly one of the BZ searches and the connected-support searches is nonempty.
        self._searches: tuple[_CosetSearch, ...] = ()
        self._support_searches: tuple[_ConnectedSupportCosetSearch, ...] = ()
        self.css_weight = css_weight
        assert_binary(stabilizers, "stabilizers")
        matrix = as_binary_matrix(stabilizers)
        self.num_qubits = len(matrix)
        n = self.num_qubits
        if not n or matrix.shape[1] != 2 * n:
            raise ValueError("Require a complete nonempty n by 2n stabilizer matrix")
        # The syndrome of a packed Pauli ``e`` is ``checks @ e``, where ``checks`` swaps the X and Z
        # halves of the stabilizers, and the stabilizers commute exactly when ``checks @ matrix.T``
        # is zero.
        checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
        if len(row_reduce_binary(matrix)[2]) != n or np.any((checks @ matrix.T) % 2):
            raise ValueError("Stabilizers must be independent and commute")

        # Search the Pauli errors themselves, unless the CSS weight on a CSS state lets the X and Z
        # parts of the errors be searched separately.
        combinations = _get_css_combinations(matrix) if css_weight else None
        if combinations is None:
            if method == "brouwer_zimmermann":
                self._searches = (
                    _get_pauli_search(
                        _StabilizerKey(matrix, checks),
                        css_weight,
                        tuning.disjoint_random_column_orders,
                        tuning.max_css_single_pairs,
                    ),
                )
            elif css_weight:
                raise ValueError(
                    "The connected-support method weighs errors by CSS weight only on CSS states;"
                    " use css_weight=False or the default method"
                )
            else:
                self._support_searches = (_ConnectedSupportCosetSearch(matrix, homogeneous=False),)
            return

        # Combine the stabilizers into pure-X and pure-Z stabilizers, and row-reduce the X parts of
        # the pure-X stabilizers and the Z parts of the pure-Z stabilizers.  The reduced rows depend
        # only on the stabilizer group, so they key the cache of BZ searches.
        x_combinations, z_combinations = combinations
        x_rows, x_transform, _ = row_reduce_binary(
            (x_combinations @ matrix[:, :n]) % 2, np.eye(len(x_combinations), dtype=np.uint8)
        )
        z_rows, z_transform, _ = row_reduce_binary(
            (z_combinations @ matrix[:, n:]) % 2, np.eye(len(z_combinations), dtype=np.uint8)
        )
        assert x_transform is not None and z_transform is not None
        if method == "connected_support":
            # The Z parts of the pure-Z stabilizers check the X parts, and the X parts of the pure-X
            # stabilizers check the Z parts.
            self._support_searches = (
                _ConnectedSupportCosetSearch(z_rows, homogeneous=True),
                _ConnectedSupportCosetSearch(x_rows, homogeneous=True),
            )
        else:
            self._searches = _get_css_searches(
                n,
                len(x_rows),
                pack_bits(x_rows).tobytes(),
                pack_bits(z_rows).tobytes(),
                tuning.disjoint_random_column_orders,
            )

        # Row ``j`` of ``z_rows`` is the Z part of a pure-Z stabilizer, the product of the given
        # stabilizers with coefficients ``c = (z_transform @ z_combinations)[j]``.  An error's
        # syndrome bit for that stabilizer is the parity of its X part with the row, and also the
        # parity of its syndrome for the given stabilizers with ``c``.  So the rows ``c`` form the X
        # syndrome map, from a syndrome for the given stabilizers to the syndrome of the X part, and
        # likewise for Z.  Each map is stored as its packed columns, one per bit of a syndrome for
        # the given stabilizers.
        x_syndrome_map = (z_transform @ z_combinations) % 2
        z_syndrome_map = (x_transform @ x_combinations) % 2
        self._map_columns = (
            pack_bits(np.ascontiguousarray(x_syndrome_map.T)),
            pack_bits(np.ascontiguousarray(z_syndrome_map.T)),
        )

        # A single syndrome is mapped by XORing the columns of both maps for its set bits, which is
        # cheaper than a parity for each row, since most syndromes have few set bits.  Entry ``i``
        # holds column ``i`` of the X map in its low ``_num_x_syndrome_bits`` bits, and column ``i``
        # of the Z map above them.
        self._num_x_syndrome_bits = len(z_rows)
        self._joint_map_columns = _row_ints(
            np.ascontiguousarray(np.vstack([x_syndrome_map, z_syndrome_map]).T)
        )

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether some error with this syndrome has weight at most ``max_weight``.

        Args:
            syndrome: A packed syndrome (see ``CosetLeaderSearch``).
            max_weight: The largest weight that the query accepts.

        Returns:
            A ``MinWeightBound``.  With two searches, its lower bound is the larger of theirs.

        Raises:
            ValueError: If the syndrome is out of range.
        """
        if not 0 <= syndrome < 1 << self.num_qubits:
            raise ValueError("The syndrome is out of range")
        searches = self._support_searches or self._searches
        if len(searches) == 1:
            return searches[0].get_min_weight_bound(syndrome, max_weight)
        part_syndromes = _xor_selected(self._joint_map_columns, syndrome)
        x_syndrome = part_syndromes & ((1 << self._num_x_syndrome_bits) - 1)
        z_syndrome = part_syndromes >> self._num_x_syndrome_bits
        x_bound = searches[0].get_min_weight_bound(x_syndrome, max_weight)
        z_bound = searches[1].get_min_weight_bound(z_syndrome, max_weight)
        return MinWeightBound(
            x_bound.within_max_weight and z_bound.within_max_weight,
            x_bound.representative | (z_bound.representative << self.num_qubits),
            max(x_bound.lower_bound, z_bound.lower_bound),
        )

    def find_first_not_within_max_weight(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> tuple[int, MinWeightBound | None]:
        """Find the first syndrome whose class has no error of weight at most ``max_weight``.

        Args:
            syndromes: One packed syndrome (see ``CosetLeaderSearch``) per row, split into 64-bit
                machine words, with machine word ``j`` holding bits ``64 j`` and up.  Each syndrome
                must be below ``2**num_qubits``.  The BZ searches do not check this, and the
                connected-support searches raise a ``ValueError``.  The rows may have fewer words
                than ``num_qubits`` bits need, if every syndrome fits in them.
            max_weight: The largest weight that the query accepts.

        Returns:
            The index of that syndrome, or the number of syndromes if there is none, and a
            ``MinWeightBound`` for that syndrome, or None if there is none.  With one BZ search,
            the bound is the one that the batch search found, so the syndrome is not searched
            again.  With two BZ searches, or with the connected-support method, it is the one that
            ``get_min_weight_bound`` gives.
        """
        # The connected-support searches answer one syndrome at a time.
        if self._support_searches:
            for index, row in enumerate(syndromes):
                bound = self.get_min_weight_bound(_packed_row_int(row), max_weight)
                if not bound.within_max_weight:
                    return index, bound
            return len(syndromes), None

        # With one search, take its bound and the lightest error that it visited.
        if len(self._searches) == 1:
            (search,) = self._searches
            best = np.empty(search.tables.inverse.shape[1], dtype=np.uint64)
            index, lower_bound = _find_first_not_within_max_weight(
                syndromes, max_weight, best, *search.tables, search.scale
            )
            if index == len(syndromes):
                return index, None
            return index, MinWeightBound(False, search.unembed(best), int(lower_bound))

        # With two searches, a syndrome's class fails, that is, has no error of weight at most
        # ``max_weight``, exactly when the coset of its X part or that of its Z part does.  So find
        # the first syndrome whose X-part coset fails, and then search the Z-part cosets of the
        # syndromes before it.  The bound for the failing syndrome needs the representatives of both
        # cosets, which ``get_min_weight_bound`` finds.
        index = len(syndromes)
        for search, map_columns in zip(self._searches, self._map_columns):
            part_syndromes = _map_syndromes(syndromes[:index], map_columns)
            best = np.empty(search.tables.inverse.shape[1], dtype=np.uint64)
            index, _ = _find_first_not_within_max_weight(
                part_syndromes, max_weight, best, *search.tables, search.scale
            )
        if index == len(syndromes):
            return index, None
        return index, self.get_min_weight_bound(_packed_row_int(syndromes[index]), max_weight)


class _CosetSearchTables(NamedTuple):
    """The tables of ``_search_coset`` for one coset search of a ``CosetLeaderSearch``.

    The search runs on embedded words.  For a search of Pauli errors, the embedded word of an error
    is its packed Pauli for the CSS weight, and its packed Pauli followed by ``x xor z`` for the
    physical weight (see ``symplectic_to_hamming``).  For a search of X or Z parts, the embedded
    word of a part is the part itself.

    Attributes:
        inverse: Row ``j`` is the packed embedded word of a word whose syndrome has only bit ``j``
            set.
        generators: The packed rows of each disjoint information set of the embedded generators of
            the code.
        pivots: The pivot columns of each information set, padded with -1.
        planning: The planning buffer of the search (see ``_get_planning``): the bounds of its
            steps, and one more row that caches a stopping step.  For a nonnegative maximum weight,
            ``_get_stopping_step`` reuses the cached step exactly while it stays the first step
            whose bound exceeds that weight, and replaces it otherwise.  Queries write only this row
            of the tables, and oracles that share a cached search share it.
        x_mask: Selects the bits of a packed embedded word that count as its X part, with one
            machine word for each machine word of the embedded words.  Only searches of Pauli
            errors by CSS weight use the X part, and their layers scan with it, but every search
            reads the mask when it weighs the word of a syndrome and the offsets of layer zero.
    """

    inverse: npt.NDArray[np.uint64]
    generators: npt.NDArray[np.uint64]
    pivots: npt.NDArray[np.int64]
    planning: _PlanningBuffer
    x_mask: npt.NDArray[np.uint64]


class _CosetSearch:
    """One coset search of a ``CosetLeaderSearch``: a search of the cosets of one binary code.

    The words of a search are either packed Paulis on ``n`` qubits, in which case the code is
    spanned by the packed stabilizers, or the X parts (or Z parts) of Paulis, in which case the code
    is spanned by the X parts of the pure-X stabilizers (or the Z parts of the pure-Z stabilizers).
    Each coset is labeled by the syndrome of its words, whose bit ``j`` is the parity of a word with
    row ``j`` of a matrix of checks.

    Args:
        tables: The tables of ``_search_coset``.
        css_weight: True to weigh words by their CSS weight, which is the Hamming weight of an X
            or Z part, or False to weigh them by their physical weight.
        num_qubits: The number of qubits ``n``.
        inverse_rows: Row ``j`` of a right inverse of the checks, not embedded, as an integer: a
            word whose syndrome has only bit ``j`` set.
        check_columns: Column ``c`` of the checks as an integer: the syndrome of the word with only
            bit ``c`` set.
        is_pauli: Whether the words are packed Paulis of ``2 n`` bits, rather than X or Z parts of
            ``n`` bits.
        max_css_single_pairs: For a search of Pauli errors by CSS weight, the most pairs of distinct
            syndromes of single-qubit X and Z errors that it tabulates (see ``_css_singles``).
    """

    def __init__(
        self,
        tables: _CosetSearchTables,
        css_weight: bool,
        num_qubits: int,
        inverse_rows: tuple[int, ...],
        check_columns: tuple[int, ...],
        *,
        is_pauli: bool,
        max_css_single_pairs: int = 0,
    ) -> None:
        self.tables = tables
        self._max_css_single_pairs = max_css_single_pairs
        self.css_weight = css_weight
        # The compiled search weighs embedded words by their CSS weight only in a search of Pauli
        # errors by CSS weight, and otherwise by their Hamming weight, which is twice the physical
        # weight of an error, and the weight of an X or Z part (see ``_search_coset``).  The scale
        # chooses which of these weights a scan computes.  The weight factor of the coset instance
        # is separate: it is two for every search of Pauli errors, including those by CSS weight,
        # whose scale is zero, since an error weighs at least half the Hamming weight of its
        # embedded word (see ``_get_pauli_search``).
        self.scale = 0 if css_weight and is_pauli else 1 if css_weight else 2
        self._num_qubits = num_qubits
        self._is_pauli = is_pauli
        self._inverse_rows = inverse_rows
        self._num_syndrome_bits = len(inverse_rows)

        # Map each syndrome of a word of weight zero or one to such a word: a single bit, or a
        # single-qubit Pauli.  For the CSS weight, ``_weight_one_word`` also pairs the single-qubit
        # X errors with the single-qubit Z errors, to find ``X_i Z_j`` on distinct qubits.
        self._singles = {0: 0}
        self._x_singles = {0: 0}
        self._z_singles = {0: 0}
        if not is_pauli:
            for column, syndrome in enumerate(check_columns):
                self._singles.setdefault(syndrome, 1 << column)
            return
        for qubit in range(num_qubits):
            x, z = 1 << qubit, 1 << (qubit + num_qubits)
            sx, sz = check_columns[qubit], check_columns[qubit + num_qubits]
            self._x_singles.setdefault(sx, x)
            self._z_singles.setdefault(sz, z)
            self._singles.setdefault(sx, x)
            self._singles.setdefault(sz, z)
            self._singles.setdefault(sx ^ sz, x ^ z)

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether the coset of this syndrome has a word of weight at most ``max_weight``.

        Returns:
            A ``MinWeightBound`` for the coset, whose representative is a word of the coset.
        """
        word = self._weight_one_word(syndrome)
        if word is not None:
            weight = self._weight(word)
            return MinWeightBound(weight <= max_weight, word, weight)

        # Otherwise the coset has no word of weight at most one, so two is a lower bound.
        if max_weight < 2:
            return MinWeightBound(False, _xor_selected(self._inverse_rows, syndrome), 2)

        # Search the coset, and keep the lightest word that the search visits in ``best``.
        best = np.empty(self.tables.inverse.shape[1], dtype=np.uint64)
        syndrome_words = _int_to_words(syndrome, max(1, (self._num_syndrome_bits + 63) // 64))
        lower_bound = _search_coset(syndrome_words, max_weight, best, *self.tables, self.scale)
        return MinWeightBound(lower_bound <= max_weight, self.unembed(best), max(2, lower_bound))

    def unembed(self, embedded_word: npt.NDArray[np.uint64]) -> int:
        """Convert a packed embedded word into a word, without the columns that the embedding adds.

        The embedding for the physical weight appends ``x xor z`` to a packed Pauli (see
        ``_CosetSearchTables``), and the other embeddings add nothing.
        """
        return _packed_row_int(embedded_word) & ((1 << self._num_word_bits) - 1)

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


class _StabilizerKey:
    """A stabilizer matrix and its checks, which a cache can hash and compare by the matrix.

    The checks follow from the stabilizers, so the stabilizers alone key the cache, and the key
    keeps both matrices to build a search that the cache has not stored yet.  Keys of packed bytes
    would not hold the matrices, but unpacking them would cost about 0.4% of the setup of a search.
    """

    def __init__(self, stabilizers: npt.NDArray[np.uint8], checks: npt.NDArray[np.uint8]) -> None:
        self.stabilizers = stabilizers
        self.checks = checks
        self._key = (stabilizers.shape, stabilizers.tobytes())
        self._hash = hash(self._key)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StabilizerKey) and self._key == other._key


class _ConnectedSupportCosetSearch:
    """A coset search of a ``CosetLeaderSearch`` by the connected-support search of its checks.

    The words, checks, and syndromes are those of a ``_CosetSearch``: packed Paulis, whose syndrome
    for the stabilizers is that of the symplectic search of ``connected_support.SupportSearch``, or
    the X or Z parts of Paulis, whose syndrome is that of a binary search of the checks.  The search
    starts from the syndrome of a query, and grows supports through its unsatisfied checks (Section
    3.8 of ``connected_support.md``).  Its bounds are weaker than those of a ``_CosetSearch``.  A
    query within ``max_weight`` gives a lower bound of one, or zero for an empty syndrome.  A query
    with no word within ``max_weight`` gives ``max_weight + 1``, or, for a negative ``max_weight``,
    one, or zero for an empty syndrome.

    The search is fast only if its checks are sparse, but the given checks, such as row-reduced
    stabilizers, can be dense.  So it searches the sparser checks of ``_sparsify``, which span the
    same space, and maps the syndrome of a query to them.

    Args:
        matrix: The checks, whose kernel is the code: the stabilizers, in ``[X|Z]`` order, for
            Paulis, or the rows that check the X or Z parts.  Bit ``j`` of the syndrome of a word
            is its symplectic product with row ``j`` for Paulis, and its dot product for parts.
        homogeneous: False for Paulis, and True for X or Z parts.
    """

    def __init__(self, matrix: npt.NDArray[np.uint8], *, homogeneous: bool) -> None:
        rows, transform = _sparsify(matrix, homogeneous=homogeneous)
        self._num_checks = len(rows)
        # Row ``i`` of the sparse rows sums the given rows that row ``i`` of the transform selects,
        # so bit ``i`` of a syndrome for them is the parity of those bits of the given syndrome.
        # Entry ``j`` of the map is column ``j`` of the transform, which a given syndrome bit adds.
        self._syndrome_map_columns = _row_ints(np.ascontiguousarray(transform.T))
        # Without a function that finds automorphisms, the search never asks for them.
        self._search = connected_support.SupportSearch(
            rows,
            np.zeros((0, matrix.shape[1]), dtype=np.uint8),
            homogeneous=homogeneous,
            automorphism_minimum_states=_tuning.MAX_VALUE,
        )
        # Row ``j`` of the inverse is a word whose syndrome has only bit ``j`` set, so the words of
        # the set bits of a syndrome sum to a representative of its coset.  The syndrome of a Pauli
        # is its product with the rows whose halves are swapped.
        num_variables = self._search.num_variables
        syndrome_rows = rows if homogeneous else np.roll(rows, num_variables, axis=1)
        self._inverse_rows = _row_ints(_get_inverse(syndrome_rows))

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether the coset of this syndrome has a word of weight at most ``max_weight``.

        A negative ``max_weight`` admits no word, as for a ``_CosetSearch``.
        """
        # Map the syndrome to the sparse checks, and search from its bits, which the search packs
        # again.  Every word weighs at least one, unless the syndrome is empty.
        syndrome = _xor_selected(self._syndrome_map_columns, syndrome)
        least_weight = int(syndrome != 0)
        symbols = None
        if max_weight >= 0:
            num_words = (self._num_checks + 63) // 64
            bits = unpack_bits(_int_to_words(syndrome, num_words)[None], self._num_checks)[0]
            symbols = self._search.find_vector_with_syndrome(bits, max_weight)
        if symbols is None:
            representative = _xor_selected(self._inverse_rows, syndrome)
            return MinWeightBound(False, representative, max(max_weight + 1, least_weight))

        # A symbol gives the qubit ``symbol % n`` the Pauli X, Z, or Y for values 0, 1, and 2.  The
        # symbols of an X or Z part have only the value 0.
        num_variables = self._search.num_variables
        word = 0
        for symbol in symbols.tolist():
            value, variable = divmod(symbol, num_variables)
            if value != 1:
                word |= 1 << variable
            if value:
                word |= 1 << (num_variables + variable)
        return MinWeightBound(True, word, least_weight)


@functools.lru_cache(maxsize=_tuning.PAULI_SEARCH_CACHE_SIZE)
def _get_pauli_search(
    key: _StabilizerKey,
    css_weight: bool,
    random_column_orders: int,
    max_css_single_pairs: int,
) -> _CosetSearch:
    """The Pauli search, of Pauli errors modulo the stabilizers, with syndromes for the stabilizers.

    ``key`` holds the stabilizers and their checks (see ``CosetLeaderSearch``).  The syndromes
    depend on the rows of the stabilizer matrix, not only on the group that they generate, so the
    rows key the cache.  ``css_weight`` and the fields ``disjoint_random_column_orders`` and
    ``max_css_single_pairs`` of the tuning, which configure the search, key it too.
    """
    stabilizers, checks = key.stabilizers, key.checks
    num_qubits = len(stabilizers)
    inverse = _get_inverse(checks)
    embedded_stabilizers = stabilizers if css_weight else symplectic_to_hamming(stabilizers)
    embedded_inverse = inverse if css_weight else symplectic_to_hamming(inverse)
    # An error weighs at least half of the Hamming weight of its embedded word, by either measure of
    # weight, so the certificate bounds are halved.
    instance = affine_instance.get_coset_instance(
        embedded_stabilizers, random_column_orders=random_column_orders, weight_factor=2
    )
    tables = _get_search_tables(instance, embedded_inverse, num_qubits)
    check_columns = _row_ints(np.ascontiguousarray(checks.T))
    return _CosetSearch(
        tables,
        css_weight,
        num_qubits,
        _row_ints(inverse),
        check_columns,
        is_pauli=True,
        max_css_single_pairs=max_css_single_pairs,
    )


@functools.lru_cache(maxsize=_tuning.CSS_SEARCH_CACHE_SIZE)
def _get_css_searches(
    num_qubits: int, num_x_rows: int, x_rows: bytes, z_rows: bytes, random_column_orders: int
) -> tuple[_CosetSearch, _CosetSearch]:
    """The classical searches of the X parts and of the Z parts of the errors on a CSS state.

    ``x_rows`` holds the X parts of ``num_x_rows`` pure-X stabilizers, and ``z_rows`` the Z parts of
    the other ``num_qubits - num_x_rows`` stabilizers, which are pure-Z.  Both are in reduced row
    echelon form, which depends only on the stabilizer group, and packed by ``pack_bits`` into
    bytes, which ``lru_cache`` can hash.  The X parts of errors are searched modulo the code of the
    X rows, with the Z rows as checks, and the Z parts the other way around (see
    ``CosetLeaderSearch``).  ``random_column_orders`` is the ``disjoint_random_column_orders`` of
    the tuning, which keys the cache too.
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
        _get_classical_search(x_matrix, z_matrix, random_column_orders),
        _get_classical_search(z_matrix, x_matrix, random_column_orders),
    )


def _get_classical_search(
    generators: npt.NDArray[np.uint8], checks: npt.NDArray[np.uint8], random_column_orders: int
) -> _CosetSearch:
    """The classical search of X or Z parts modulo the row space of independent generators.

    The row space of the generators is the null space of the independent checks, so each coset is
    labeled by its syndrome for the checks.  Words are weighed by their Hamming weight, which the BZ
    certificate bounds directly, so its bounds are not halved.
    """
    num_qubits = generators.shape[1]
    inverse = _get_inverse(checks)
    instance = affine_instance.get_coset_instance(
        generators, random_column_orders=random_column_orders, weight_factor=1
    )
    # Every column counts as the X part, so the CSS weight of a word is its Hamming weight.
    tables = _get_search_tables(instance, inverse, num_qubits)
    check_columns = _row_ints(np.ascontiguousarray(checks.T))
    return _CosetSearch(tables, True, num_qubits, _row_ints(inverse), check_columns, is_pauli=False)


def _get_inverse(checks: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """A right inverse of independent checks: row ``j`` has a syndrome of only bit ``j``."""
    _, labels, pivots = row_reduce_binary(checks, np.eye(len(checks), dtype=np.uint8))
    assert labels is not None
    inverse = np.zeros(checks.shape, dtype=np.uint8)
    inverse[:, pivots] = labels.T
    return inverse


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


def _get_search_tables(
    instance: affine_instance.AffineInstance, inverse: npt.NDArray[np.uint8], num_x_columns: int
) -> _CosetSearchTables:
    """Build the tables of ``_search_coset`` for the cosets of the code of a coset instance.

    ``inverse`` holds embedded words whose syndromes have one bit set, and the first
    ``num_x_columns`` columns count as the X part of an embedded word.
    """
    width = instance.generators.shape[2]
    x_mask = _int_to_words((1 << num_x_columns) - 1, width)
    return _CosetSearchTables(
        pack_bits(inverse),
        instance.generators,
        instance.pivots,
        _get_planning(instance.get_weight_bounds()),
        x_mask,
    )


def _get_planning(bounds: npt.NDArray[np.int64]) -> _PlanningBuffer:
    """Build the planning buffer of a coset search from the bounds of its steps.

    The compiled search takes the bounds and its cached stopping step in this one array, since each
    argument of a compiled call adds to the time of every query.  The buffer has one more row than
    the bounds:

    - Every row but the last holds the bounds of ``AffineInstance.get_weight_bounds``: entry
      ``[w, j]`` bounds the weight of every word that the search has not visited once the first
      ``j`` sets have completed layer ``w`` and the others layer ``w - 1``.  So column zero of each
      row holds the bound before the first step of that row.  These rows never change, and code
      reads them as the view ``planning[:-1]``.
    - The last row holds no bounds.  Its first two entries are the layer and the number of
      completed sets of the stopping step that ``_get_stopping_step`` returned last, or of the first
      step, ``(0, 1)``, before its first call, and its other entries are zero.  Only
      ``_get_stopping_step`` reads and writes this row.
    """
    planning = np.zeros((len(bounds) + 1, bounds.shape[1]), dtype=np.int64)
    planning[:-1] = bounds
    planning[-1, 1] = 1
    return planning


def _get_css_combinations(
    stabilizers: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]] | None:
    """Combine stabilizers into pure-X and pure-Z generators, or return None if they are not CSS.

    The stabilizers must be independent.  The combinations of them whose Z parts cancel, which form
    the null space of the transposed Z parts, give the pure-X stabilizers, and likewise for pure-Z
    stabilizers.  A pure-X and a pure-Z stabilizer are equal only if both are the identity, so these
    stabilizers are independent, and they generate the group exactly when there are ``n`` of them.

    Returns:
        Matrices ``(x_combinations, z_combinations)``, whose products with ``stabilizers`` modulo 2
        are pure-X and pure-Z stabilizers that together generate the group, or None.
    """
    num_qubits = len(stabilizers)
    x_combinations = get_kernel(np.ascontiguousarray(stabilizers[:, num_qubits:].T))
    z_combinations = get_kernel(np.ascontiguousarray(stabilizers[:, :num_qubits].T))
    if len(x_combinations) + len(z_combinations) != num_qubits:
        return None
    return x_combinations, z_combinations


def _row_ints(matrix: npt.NDArray[np.uint8]) -> tuple[int, ...]:
    """Pack each row of a binary matrix into an integer, with entry ``j`` in bit ``j``."""
    return tuple(map(_packed_row_int, pack_bits(matrix)))


def _packed_row_int(row: npt.NDArray[np.uint64]) -> int:
    """Convert a row of 64-bit words into an integer, with word ``j`` in bits ``64 j`` and up."""
    return sum(int(word) << (64 * jj) for jj, word in enumerate(row))


def _int_to_words(value: int, width: int) -> npt.NDArray[np.uint64]:
    """Split an integer into ``width`` 64-bit words, with word ``j`` in bits ``64 j`` and up."""
    return np.frombuffer(value.to_bytes(8 * width, "little"), dtype="<u8").astype(np.uint64)


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


def _xor_selected(rows: Sequence[int], support: int) -> int:
    """XOR together the rows whose indices are the set bits of ``support``."""
    word = 0
    while support:
        # Add the row for the lowest set bit, and then clear that bit.
        low = support & -support
        word ^= rows[low.bit_length() - 1]
        support ^= low
    return word


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
    planning: _PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    scale: int,
) -> tuple[int, int]:
    """Search the coset of each syndrome in turn with ``_search_coset``.

    The tables, from ``inverse`` to ``x_mask``, are those of ``_search_coset``.  Return the index of
    the first syndrome whose coset has no word of weight at most ``max_weight``, and the lower bound
    that ``_search_coset`` returns for it, with its lightest word left in ``best``.  If there is no
    such syndrome, return the number of syndromes and zero.
    """
    for index in range(len(syndromes)):
        lower_bound = _search_coset(
            syndromes[index], max_weight, best, inverse, generators, pivots, planning, x_mask, scale
        )
        if lower_bound > max_weight:
            return index, lower_bound
    return len(syndromes), 0


@always_compiled(inline=True)
def _search_coset(
    syndrome: npt.NDArray[np.uint64],
    max_weight: int,
    best: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    planning: _PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    scale: int,
) -> int:
    """Search the coset of a syndrome for a word of weight at most ``max_weight``.

    The syndrome is split into 64-bit machine words, with machine word ``j`` holding bits ``64 j``
    and up.  The search runs on embedded words.  In the Pauli search, the words are Pauli errors,
    and the embedded word of an error is its packed Pauli when errors are weighed by CSS weight, and
    its packed Pauli followed by ``x xor z`` when they are weighed by physical weight, so that its
    Hamming weight is twice the physical weight (see ``symplectic_to_hamming``).  In a classical
    search, the words are X or Z parts, each is its own embedded word, and its weight is its Hamming
    weight.  The weight of an embedded word is that of its word: its CSS weight if ``scale`` is
    zero, with its X part selected by ``x_mask``, and otherwise its Hamming weight divided by
    ``scale``, which is two for the physical weight and one for X or Z parts.  The tables are those
    of ``_CosetSearchTables``.  Row ``j`` of ``inverse`` is the embedded word of a word whose
    syndrome has only bit ``j`` set.  ``generators`` and ``pivots`` are the disjoint information
    sets of the embedded generators of the code, with pivots padded with -1.  ``planning`` is the
    planning buffer of the search, whose cached stopping step a query may update.

    The embedded words of the coset form a coset of the embedded code.  The search starts from the
    sum of the rows of ``inverse`` for the set bits of the syndrome, a word with that syndrome.  The
    offset of an information set is this word reduced by rows of the set until it is zero on the
    set's pivot columns.  Every embedded word of the coset is the offset plus a unique sum of rows
    of the set, and has a one on a pivot column exactly where it adds that pivot's row, so the BZ
    certificate applies to the coset as it does to a space.  Layer ``w`` of a set is its offset plus
    each sum of ``w`` of its rows, so layer zero is the offset itself.

    The search visits layer zero of the information sets in order, then layer one, and so on.  It
    keeps the lightest word visited in ``best``, and stops at the first word of weight at most
    ``max_weight``.  Once the first ``j`` sets have completed layer ``w``, every unvisited word
    weighs at least the bound of step ``(w, j)`` in ``planning``.  So the search also stops at its
    stopping step, the first step whose bound exceeds ``max_weight``, which ``_get_stopping_step``
    finds.  It skips each set that adds nothing to the bound of that step (see
    ``_search_from_syndrome_word``).

    Returns:
        A lower bound on the weight of every word in the coset: zero, which is at most
        ``max_weight``, if ``best`` has weight at most ``max_weight``, and otherwise a bound that
        exceeds ``max_weight``.
    """
    best_weight = _get_syndrome_word(syndrome, best, inverse, x_mask, scale)
    if best_weight <= max_weight:
        return 0
    stop_layer, stop_num_completed, stop_bound = _get_stopping_step(planning, max_weight)
    lower_bound, _ = _search_from_syndrome_word(
        max_weight,
        stop_layer,
        stop_num_completed,
        stop_bound,
        best,
        best_weight,
        generators,
        pivots,
        x_mask,
        scale,
    )
    return lower_bound


@always_compiled
def _get_stopping_step(planning: _PlanningBuffer, max_weight: int) -> tuple[int, int, int]:
    """Find the step at which a coset search stops, and cache it in the planning buffer.

    The stopping step is the first step whose bound exceeds the maximum weight: the step at which
    ``stop_num_completed`` sets complete layer ``stop_layer``.  A search stops there, unless it
    finds a light enough word sooner.  The bounds of the steps do not decrease in their order.
    Column zero of each row of bounds holds the bound before the first step of the row, which is
    zero in row zero.  So the bound of the step before step ``(w, j)`` is entry ``[w, j - 1]``, and
    for a nonnegative maximum weight, step ``(w, j)`` is the stopping step exactly if its bound
    exceeds the maximum weight and entry ``[w, j - 1]`` does not.  The step cached in the planning
    buffer (see ``_get_planning``) is reused if it passes this check.  Otherwise, the stopping step
    is found by binary search and cached in its place.  A negative maximum weight fails the check,
    and the binary search returns the first step.  If no bound exceeds the maximum weight, which
    happens only when the maximum weight reaches the largest native integer, the binary search
    returns the last step.

    Oracles that share a cached search share its planning buffer.  Compiled calls hold the global
    interpreter lock, since numba compiles them without ``nogil``, so calls interleave only when
    these functions run as plain Python in several threads, or in free-threaded Python.  The layer
    and the number of completed sets of a step are written as two entries, so a read may pair the
    layer of one step with the number of completed sets of another.  Any such pair indexes the
    bounds, which never change, and it is checked against them before it is used.

    Returns:
        ``stop_layer``, ``stop_num_completed``, and the bound of that step.
    """
    bounds = planning[:-1]
    layer, num_completed = planning[-1, 0], planning[-1, 1]
    is_stopping_step = (
        max_weight < bounds[layer, num_completed] and bounds[layer, num_completed - 1] <= max_weight
    )
    if not is_stopping_step:
        layer, num_completed = information_sets.find_stopping_step(
            bounds, max_weight, 0, len(bounds) - 1
        )
        planning[-1, 0], planning[-1, 1] = layer, num_completed
    return layer, num_completed, int(bounds[layer, num_completed])


@always_compiled
def _search_from_syndrome_word(
    max_weight: int,
    stop_layer: int,
    stop_num_completed: int,
    stop_bound: int,
    best: npt.NDArray[np.uint64],
    best_weight: int,
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    x_mask: npt.NDArray[np.uint64],
    scale: int,
) -> tuple[int, int]:
    """Search a coset, as ``_search_coset`` does, from the word of its syndrome.

    ``best`` holds that word, of weight ``best_weight`` above ``max_weight``.  The stopping step and
    its bound are those of ``_get_stopping_step``.  A set of rank deficiency ``s`` adds to the bound
    of a step only once it has completed layer ``s``, so a set that would not have completed it by
    the stopping step adds nothing, and the search skips it.

    Returns:
        The lower bound of ``_search_coset``, and the number of words weighed after the word of the
        syndrome.  Only tests read the number, to check which sets a search skips.
    """
    num_sets, num_rows, width = generators.shape

    # Reduce the word on each information set to its offset, which is zero on its pivot columns.
    offsets = np.empty((num_sets, width), dtype=np.uint64)
    for set_index in range(num_sets):
        offsets[set_index] = best
        for row in range(num_rows):
            pivot = pivots[set_index, row]
            if pivot < 0:
                break
            if offsets[set_index, pivot // 64] >> np.uint64(pivot % 64) & np.uint64(1):
                for index in range(width):
                    offsets[set_index, index] ^= generators[set_index, row, index]

    # Visit each layer one information set at a time, up to the stopping step.  The scans of layers
    # take work space for one set, and weigh Hamming weights, which are ``scale`` times the weights,
    # or CSS weights if ``scale`` is zero.
    weighed = 0
    widths = np.full(1, width, dtype=np.int64)
    support = np.empty(num_rows, dtype=np.int64)
    prefixes = np.empty((num_rows, 1, width), dtype=np.uint64)
    for layer in range(stop_layer + 1):
        for set_index in range(num_sets if layer < stop_layer else stop_num_completed):
            # Skip a set that adds nothing to the bound of the stopping step, since its rank
            # deficiency exceeds the last layer that it completes by then.  This holds exactly if
            # its rank is at most ``position = num_rows - last_layer - 1``, that is, if its pivot
            # at ``position`` is padding.  The last layer is at least zero for every set visited
            # here, so ``position`` is below ``num_rows``.  It is negative only if the last layer
            # is ``num_rows``, where no set is skipped.
            last_layer = stop_layer if set_index < stop_num_completed else stop_layer - 1
            position = num_rows - last_layer - 1
            if position >= 0 and pivots[set_index, position] < 0:
                continue
            if layer == 0:
                best_weight = _keep_lighter(offsets[set_index], best, best_weight, x_mask, scale)
                weighed += 1
            elif scale == 0:
                best_weight, count = affine_search.scan_coset_layer_by_css_weight(
                    offsets[set_index],
                    generators[set_index : set_index + 1],
                    widths,
                    support[:layer],
                    prefixes,
                    best_weight,
                    max_weight,
                    x_mask,
                    best,
                )
                weighed += count
            else:
                best_ones, count = affine_search.scan_coset_layer(
                    offsets[set_index],
                    generators[set_index : set_index + 1],
                    widths,
                    support[:layer],
                    prefixes,
                    scale * best_weight,
                    scale * max_weight,
                    best,
                )
                best_weight = best_ones // scale
                weighed += count
            if best_weight <= max_weight:
                return 0, weighed
    return min(best_weight, stop_bound), weighed


@always_compiled(inline=True)
def _get_syndrome_word(
    syndrome: npt.NDArray[np.uint64],
    word: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    x_mask: npt.NDArray[np.uint64],
    scale: int,
) -> int:
    """Write the sum of the rows of ``inverse`` for the set bits of a syndrome into ``word``.

    The sum is an embedded word with that syndrome.  Return its weight (see ``_search_coset``).
    """
    word[:] = 0
    for syndrome_word in range(len(syndrome)):
        bits = syndrome[syndrome_word]
        while bits:
            bit = 64 * syndrome_word + trailing_zeros_word(bits)
            for index in range(len(word)):
                word[index] ^= inverse[bit, index]
            bits &= bits - np.uint64(1)
    return _get_weight(word, x_mask, scale)


@always_compiled
def _keep_lighter(
    word: npt.NDArray[np.uint64],
    best: npt.NDArray[np.uint64],
    best_weight: int,
    x_mask: npt.NDArray[np.uint64],
    scale: int,
) -> int:
    """Keep ``word`` in ``best`` if it is lighter, and return the weight of ``best``."""
    weight = _get_weight(word, x_mask, scale)
    if weight >= best_weight:
        return best_weight
    best[:] = word
    return weight


@always_compiled
def _get_weight(word: npt.NDArray[np.uint64], x_mask: npt.NDArray[np.uint64], scale: int) -> int:
    """The weight of the error with an embedded word (see ``_search_coset``)."""
    ones = x_ones = np.int64(0)
    for index in range(len(word)):
        ones += popcount_word(word[index])
        x_ones += popcount_word(word[index] & x_mask[index])
    return int(max(x_ones, ones - x_ones) if scale == 0 else ones // scale)


@always_compiled
def _map_syndromes(
    syndromes: npt.NDArray[np.uint64], map_columns: npt.NDArray[np.uint64]
) -> npt.NDArray[np.uint64]:
    """Map packed syndromes by a binary matrix, given by its packed columns.

    Row ``i`` of the result is the XOR of the columns ``map_columns[b]`` for the set bits ``b`` of
    row ``i`` of ``syndromes``, which is the product of the matrix with that syndrome.  The
    syndromes are split into 64-bit words, and each set bit must be below the number of columns;
    this is not checked.
    """
    width = map_columns.shape[1]
    mapped = np.zeros((len(syndromes), width), dtype=np.uint64)
    for row in range(len(syndromes)):
        for syndrome_word in range(syndromes.shape[1]):
            bits = syndromes[row, syndrome_word]
            while bits:
                bit = 64 * syndrome_word + trailing_zeros_word(bits)
                for index in range(width):
                    mapped[row, index] ^= map_columns[bit, index]
                bits &= bits - np.uint64(1)
    return mapped
