# SPDX-License-Identifier: Apache-2.0

"""Exact minimum-weight queries for syndrome classes of Pauli errors on a stabilizer state.

The Pauli errors with a given syndrome form a coset of the stabilizer group, and a lightest error in
it is a coset leader.  ``CosetLeaderSearch`` decides whether the coset leaders of a syndrome class
weigh at most a given maximum weight, with Brouwer-Zimmermann searches of cosets of binary codes.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

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
from . import information_sets
from .information_sets import advance_support

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

    The oracle answers a query with one or two coset searches.  Each searches the cosets of a binary
    code with the Brouwer-Zimmermann (BZ) algorithm (see ``brouwer_zimmermann.md``), as
    ``_search_coset`` describes.  ``get_min_weight_bound`` first looks up words of weight at most
    one in a table.

    - In general, one Pauli search covers the Pauli errors themselves, modulo the stabilizers.
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

    Args:
        stabilizers: A binary ``n x 2n`` matrix in ``[X|Z]`` order whose rows are independent
            commuting stabilizers.
        css_weight: True to weigh an error by its CSS weight, or False to weigh it by its physical
            weight.

    The oracle reads the active tuning (see ``_tuning``) when it is built.  Its field
    ``disjoint_random_column_orders`` configures every search of the oracle, and
    ``max_css_single_pairs`` the searches by CSS weight that are not split into two classical
    searches.

    Raises:
        ValueError: If the stabilizers are not ``n`` independent commuting stabilizers.
    """

    def __init__(self, stabilizers: npt.ArrayLike, *, css_weight: bool = False) -> None:
        tuning = get_tuning()
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
            self._searches: tuple[_CosetSearch, ...] = (
                _get_pauli_search(
                    matrix,
                    checks,
                    css_weight,
                    tuning.disjoint_random_column_orders,
                    tuning.max_css_single_pairs,
                ),
            )
            return

        # Combine the stabilizers into pure-X and pure-Z stabilizers, and row-reduce the X parts of
        # the pure-X stabilizers and the Z parts of the pure-Z stabilizers.  The reduced rows depend
        # only on the stabilizer group, so they key the cache of searches.
        x_combinations, z_combinations = combinations
        x_rows, x_transform, _ = row_reduce_binary(
            (x_combinations @ matrix[:, :n]) % 2, np.eye(len(x_combinations), dtype=np.uint8)
        )
        z_rows, z_transform, _ = row_reduce_binary(
            (z_combinations @ matrix[:, n:]) % 2, np.eye(len(z_combinations), dtype=np.uint8)
        )
        assert x_transform is not None and z_transform is not None
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
        if len(self._searches) == 1:
            return self._searches[0].get_min_weight_bound(syndrome, max_weight)
        part_syndromes = _xor_selected(self._joint_map_columns, syndrome)
        x_syndrome = part_syndromes & ((1 << self._num_x_syndrome_bits) - 1)
        z_syndrome = part_syndromes >> self._num_x_syndrome_bits
        x_bound = self._searches[0].get_min_weight_bound(x_syndrome, max_weight)
        z_bound = self._searches[1].get_min_weight_bound(z_syndrome, max_weight)
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
                must be below ``2**num_qubits``; this is not checked.  The rows may have fewer words
                than ``num_qubits`` bits need, if every syndrome fits in them.
            max_weight: The largest weight that the query accepts.

        Returns:
            The index of that syndrome, or the number of syndromes if there is none, and a
            ``MinWeightBound`` for that syndrome, or None if there is none.  With one search, the
            bound is the one that the batch search found, so the syndrome is not searched again.
            With two searches, it is the one that ``get_min_weight_bound`` gives.
        """
        # With one search, take its bound and the lightest error that it visited.
        if len(self._searches) == 1:
            (search,) = self._searches
            best = np.empty(search.tables.inverse.shape[1], dtype=np.uint64)
            index, lower_bound = _find_first_not_within_max_weight(
                syndromes, max_weight, best, *search.tables, search.css_weight
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
                part_syndromes, max_weight, best, *search.tables, search.css_weight
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
        unvisited_bounds: Entry ``[w, j]`` is a lower bound on the weight of every word that the
            search has not visited once the first ``j`` information sets have completed layer ``w``
            and the others have completed layer ``w - 1`` (see ``get_step_lower_bounds``).
        x_mask: Selects the bits of a packed embedded word that count as its X part.  A search of X
            or Z parts selects every column, so that the CSS weight of a word is its Hamming weight.
    """

    inverse: npt.NDArray[np.uint64]
    generators: npt.NDArray[np.uint64]
    pivots: npt.NDArray[np.int64]
    unvisited_bounds: npt.NDArray[np.int64]
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
        css_weight: The weight that ``_search_coset`` computes: True for the CSS weight of the
            embedded word, which is the Hamming weight of a word when ``x_mask`` selects every
            column, or False for the physical weight.
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
        lower_bound = int(
            _search_coset(syndrome_words, max_weight, best, *self.tables, self.css_weight)
        )
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
        if len(self._x_singles) * len(self._z_singles) > self._max_css_single_pairs:
            return None
        table: dict[int, int] = {}
        for sx, x in self._x_singles.items():
            for sz, z in self._z_singles.items():
                table.setdefault(sx ^ sz, x ^ z)
        return table


def _get_pauli_search(
    stabilizers: npt.NDArray[np.uint8],
    checks: npt.NDArray[np.uint8],
    css_weight: bool,
    random_column_orders: int,
    max_css_single_pairs: int,
) -> _CosetSearch:
    """The Pauli search, of Pauli errors modulo the stabilizers, with these checks."""
    num_qubits = len(stabilizers)
    inverse = _get_inverse(checks)
    embedded_stabilizers = stabilizers if css_weight else symplectic_to_hamming(stabilizers)
    embedded_inverse = inverse if css_weight else symplectic_to_hamming(inverse)
    # An error weighs at least half of the Hamming weight of its embedded word, by either measure of
    # weight, so the certificate bounds are halved.
    tables = _get_search_tables(
        embedded_stabilizers, embedded_inverse, num_qubits, random_column_orders, halve=True
    )
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
    # Every column counts as the X part, so the CSS weight of a word is its Hamming weight.
    tables = _get_search_tables(generators, inverse, num_qubits, random_column_orders, halve=False)
    check_columns = _row_ints(np.ascontiguousarray(checks.T))
    return _CosetSearch(tables, True, num_qubits, _row_ints(inverse), check_columns, is_pauli=False)


def _get_inverse(checks: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """A right inverse of independent checks: row ``j`` has a syndrome of only bit ``j``."""
    _, labels, pivots = row_reduce_binary(checks, np.eye(len(checks), dtype=np.uint8))
    assert labels is not None
    inverse = np.zeros(checks.shape, dtype=np.uint8)
    inverse[:, pivots] = labels.T
    return inverse


def _get_search_tables(
    generators: npt.NDArray[np.uint8],
    inverse: npt.NDArray[np.uint8],
    num_x_columns: int,
    random_column_orders: int,
    halve: bool,
) -> _CosetSearchTables:
    """Build the tables of ``_search_coset`` for the cosets of the row space of embedded generators.

    ``inverse`` holds embedded words whose syndromes have one bit set, and the first
    ``num_x_columns`` columns count as the X part of an embedded word.  ``random_column_orders`` is
    passed to ``information_sets.get_disjoint_family``.  Entry ``[w, j]`` of the certificate bounds
    the Hamming weight of every word that adds more than ``w`` rows on each of the first ``j``
    information sets and more than ``w - 1`` rows on the others, and its half bounds the weight of
    an error if ``halve``.  The certificate takes no weight divisor, since the words of a coset need
    not share the divisibility of the code, and a coverage of one, since the information sets are
    disjoint.  Without generators, each coset is a single word, which the search visits on layer
    zero of an empty information set.
    """
    num_generators, num_columns = generators.shape
    width = max(1, (num_columns + 63) // 64)
    if num_generators:
        family = information_sets.get_disjoint_family(
            generators, None, random_column_orders=random_column_orders
        )
        num_sets = family.generators.shape[0]
        packed_generators = pack_bits(family.generators.reshape(-1, num_columns)).reshape(
            num_sets, num_generators, -1
        )
        set_pivots = family.pivots
        certificate_bounds = information_sets.get_step_lower_bounds(
            num_generators, family.ranks, 1, 1
        )
        # Round half of each bound up, without overflowing the bounds after the last layer.
        unvisited_bounds = (
            certificate_bounds - certificate_bounds // 2 if halve else certificate_bounds
        ).astype(np.int64)
        unvisited_bounds[num_generators, 1:] = np.iinfo(np.int64).max
    else:
        packed_generators = np.zeros((1, 0, width), dtype=np.uint64)
        set_pivots = np.full((1, 0), -1, dtype=np.int64)
        # The one empty set visits the only word of each coset on layer zero.
        unvisited_bounds = np.array([[0, np.iinfo(np.int64).max]], dtype=np.int64)
    x_mask = _int_to_words((1 << num_x_columns) - 1, width)
    return _CosetSearchTables(
        pack_bits(inverse), packed_generators, set_pivots, unvisited_bounds, x_mask
    )


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
    unvisited_bounds: npt.NDArray[np.int64],
    x_mask: npt.NDArray[np.uint64],
    css_weight: bool,
) -> tuple[int, np.int64]:
    """Search the coset of each syndrome in turn, as ``_search_coset`` does.

    Return the index of the first syndrome whose coset has no word of weight at most ``max_weight``,
    and the lower bound that ``_search_coset`` returns for it, with its lightest word left in
    ``best``.  If there is no such syndrome, return the number of syndromes and zero.
    """
    for index in range(len(syndromes)):
        lower_bound = _search_coset(
            syndromes[index],
            max_weight,
            best,
            inverse,
            generators,
            pivots,
            unvisited_bounds,
            x_mask,
            css_weight,
        )
        if lower_bound > max_weight:
            return index, lower_bound
    return len(syndromes), np.int64(0)


@always_compiled
def _search_coset(
    syndrome: npt.NDArray[np.uint64],
    max_weight: int,
    best: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    unvisited_bounds: npt.NDArray[np.int64],
    x_mask: npt.NDArray[np.uint64],
    css_weight: bool,
) -> np.int64:
    """Search the coset of a syndrome for a word of weight at most ``max_weight``.

    The syndrome is split into 64-bit machine words, with machine word ``j`` holding bits ``64 j``
    and up.  The search runs on embedded words.  In the Pauli search, the words are Pauli errors,
    and the embedded word of an error is its packed Pauli when errors are weighed by CSS weight, and
    its packed Pauli followed by ``x xor z`` when they are weighed by physical weight, so that its
    Hamming weight is twice the physical weight (see ``symplectic_to_hamming``).  In a classical
    search, the words are X or Z parts, each is its own embedded word, and ``x_mask`` selects every
    column, so that the CSS weight of a word is its Hamming weight.  The weight of an embedded word
    is that of its word.  The tables are those of ``_CosetSearchTables``: row ``j`` of ``inverse``
    is the embedded word of a word whose syndrome has only bit ``j`` set, ``generators`` and
    ``pivots`` are the disjoint information sets of the embedded generators of the code, with pivots
    padded with -1, and ``x_mask`` selects the X part of an embedded word.

    The embedded words of the coset form a coset of the embedded code.  The search starts from the
    sum of the rows of ``inverse`` for the set bits of the syndrome, a word with that syndrome.  The
    offset of an information set is this word reduced by rows of the set until it is zero on the
    set's pivot columns.  Every embedded word of the coset is the offset plus a unique sum of rows
    of the set, and has a one on a pivot column exactly where it adds that pivot's row, so the BZ
    certificate applies to the coset as it does to a space.  Layer ``w`` of a set is its offset plus
    each sum of ``w`` of its rows, so layer zero is the offset itself.

    The search visits layer zero on every information set in order, then layer one, and so on.  It
    keeps the lightest word visited in ``best``, and stops at the first word of weight at most
    ``max_weight``.  Once the first ``j`` sets have completed layer ``w``, every unvisited word
    weighs at least ``unvisited_bounds[w, j]``, so the search also stops at the first such step
    whose bound exceeds ``max_weight``.  A set of rank deficiency ``s`` adds to that bound only once
    it has completed layer ``s``, so a set that would not have completed it by that step adds
    nothing, and the search skips it.

    Return a lower bound on the weight of every word in the coset: zero, which is at most
    ``max_weight``, if ``best`` has weight at most ``max_weight``, and otherwise a bound that
    exceeds ``max_weight``.
    """
    num_sets, num_rows, width = generators.shape
    zero = np.zeros(width, dtype=np.uint64)

    # Add the rows of the inverse for the set bits of the syndrome, to get an embedded error with
    # that syndrome.
    best[:] = 0
    for syndrome_word in range(len(syndrome)):
        bits = syndrome[syndrome_word]
        while bits:
            bit = 64 * syndrome_word + trailing_zeros_word(bits)
            for index in range(width):
                best[index] ^= inverse[bit, index]
            bits &= bits - np.uint64(1)
    best_weight = _get_weight(best, zero, x_mask, css_weight)
    if best_weight <= max_weight:
        return np.int64(0)

    # Reduce the error on each information set to its offset, which is zero on its pivot columns.
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

    # Find the first step whose bound exceeds the maximum weight: ``stop_num_completed`` sets
    # complete layer ``stop_layer``.  The search stops there, unless it finds a light enough word
    # sooner.
    stop_layer, stop_num_completed = num_rows, num_sets
    found = False
    for layer in range(num_rows + 1):
        for num_completed in range(1, num_sets + 1):
            if unvisited_bounds[layer, num_completed] > max_weight:
                stop_layer, stop_num_completed, found = layer, num_completed, True
                break
        if found:
            break

    # Skip the sets whose rank deficiency exceeds the last layer that they would complete, since
    # they add nothing to the bound of that step.
    skipped = np.zeros(num_sets, dtype=np.bool_)
    for set_index in range(num_sets):
        rank = 0
        while rank < num_rows and pivots[set_index, rank] >= 0:
            rank += 1
        last_layer = stop_layer if set_index < stop_num_completed else stop_layer - 1
        skipped[set_index] = last_layer + 1 <= num_rows - rank

    # The arguments of ``_visit_layer``, which visits one information set at a time.
    widths = np.full(1, width, dtype=np.int64)
    support = np.empty(num_rows, dtype=np.int64)
    prefixes = np.empty((num_rows, 1, width), dtype=np.uint64)
    for layer in range(stop_layer + 1):
        for set_index in range(num_sets if layer < stop_layer else stop_num_completed):
            if skipped[set_index]:
                continue
            if layer == 0:
                best_weight = _keep_lighter(
                    offsets[set_index], zero, best, best_weight, x_mask, css_weight
                )
            else:
                best_weight = _visit_layer(
                    offsets[set_index],
                    generators[set_index : set_index + 1],
                    widths,
                    support[:layer],
                    prefixes,
                    max_weight,
                    best,
                    best_weight,
                    x_mask,
                    css_weight,
                )
            if best_weight <= max_weight:
                return np.int64(0)
    return min(best_weight, np.int64(unvisited_bounds[stop_layer, stop_num_completed]))


@always_compiled
def _visit_layer(
    offset: npt.NDArray[np.uint64],
    rows: npt.NDArray[np.uint64],
    widths: npt.NDArray[np.int64],
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    max_weight: int,
    best: npt.NDArray[np.uint64],
    best_weight: np.int64,
    x_mask: npt.NDArray[np.uint64],
    css_weight: bool,
) -> np.int64:
    """Visit the offset plus each sum of ``len(support)`` rows of one information set.

    The offset, the rows, and the weights are those of ``_search_coset``.  ``rows`` has shape
    ``(1, num_rows, num_words)``, a stack of one set, because ``advance_support`` takes a stack of
    sets, and each row has ``num_words`` machine words.  ``widths`` is ``[num_words]``, and
    ``support`` and ``prefixes`` are work space for ``advance_support``.  ``best_weight``, the
    weight of ``best``, must exceed ``max_weight``.  Keep the lightest word in ``best``, as
    ``_keep_lighter`` does, stop at the first word of weight at most ``max_weight``, and return the
    weight of ``best``.

    This is the innermost loop of the search, so it weighs each word inline, and stops counting
    once the counts so far show that the word is no lighter than ``best``.  These counts are lower
    bounds on the full counts, so a word is kept exactly when ``_keep_lighter`` would keep it.
    """
    _, num_rows, num_words = rows.shape
    prefixes[0, 0] = offset
    support[0] = 0
    depth = advance_support(rows, widths, support, prefixes, 0)
    while depth >= 0:
        # Visit the remaining supports that differ from this one only in their last row.
        for row in range(support[depth], num_rows):
            # Weigh the word: its CSS weight is the larger of its X and Z counts, and its physical
            # weight is half the Hamming weight of its embedded word.
            if css_weight:
                x_ones = z_ones = np.int64(0)
                for index in range(num_words):
                    word = prefixes[depth, 0, index] ^ rows[0, row, index]
                    x_ones += popcount_word(word & x_mask[index])
                    z_ones += popcount_word(word & ~x_mask[index])
                    if x_ones >= best_weight or z_ones >= best_weight:
                        break
                weight = max(x_ones, z_ones)
            else:
                ones = np.int64(0)
                for index in range(num_words):
                    ones += popcount_word(prefixes[depth, 0, index] ^ rows[0, row, index])
                    if ones >= 2 * best_weight:
                        break
                weight = ones // 2

            # Keep the word if it is lighter, and stop if it is light enough.
            if weight < best_weight:
                best_weight = weight
                for index in range(num_words):
                    best[index] = prefixes[depth, 0, index] ^ rows[0, row, index]
                if best_weight <= max_weight:
                    return best_weight
        support[depth] = num_rows
        depth = advance_support(rows, widths, support, prefixes, depth)
    return best_weight


@always_compiled
def _keep_lighter(
    first: npt.NDArray[np.uint64],
    second: npt.NDArray[np.uint64],
    best: npt.NDArray[np.uint64],
    best_weight: np.int64,
    x_mask: npt.NDArray[np.uint64],
    css_weight: bool,
) -> np.int64:
    """Keep ``first ^ second`` in ``best`` if it is lighter, and return the weight of ``best``."""
    weight = _get_weight(first, second, x_mask, css_weight)
    if weight >= best_weight:
        return best_weight
    for index in range(len(best)):
        best[index] = first[index] ^ second[index]
    return weight


@always_compiled
def _get_weight(
    first: npt.NDArray[np.uint64],
    second: npt.NDArray[np.uint64],
    x_mask: npt.NDArray[np.uint64],
    css_weight: bool,
) -> np.int64:
    """The weight of the error with the embedded word ``first ^ second`` (see ``_search_coset``)."""
    ones = x_ones = np.int64(0)
    for index in range(len(first)):
        word = first[index] ^ second[index]
        ones += popcount_word(word)
        x_ones += popcount_word(word & x_mask[index])
    return max(x_ones, ones - x_ones) if css_weight else ones // 2


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
