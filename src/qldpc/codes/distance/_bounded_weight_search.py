# SPDX-License-Identifier: Apache-2.0

"""Exact minimum-weight queries for syndrome classes of Pauli errors on a stabilizer state."""

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

from . import _information_sets
from ._binary import as_binary_matrix, assert_binary, symplectic_to_hamming
from ._information_sets import advance_support

# The largest table of errors of CSS weight one that PauliCosetSearch builds, as a number of pairs
# of an X-single and a Z-single syndrome: about ten megabytes, or 255 qubits.
_MAX_CSS_SINGLES = 2**16

####################################################################################################
# Pauli-coset minimum-weight bounds


@dataclass(frozen=True)
class MinWeightBound:
    """The answer to a query about the minimum weight of a syndrome class of Pauli errors.

    The query asks whether the syndrome class has an error of weight at most ``max_weight``.

    Attributes:
        within_max_weight: Whether the syndrome class has such an error.
        representative: A packed Pauli (see ``PauliCosetSearch``) in the syndrome class.  If
            ``within_max_weight`` is True, its weight is at most ``max_weight``.
        lower_bound: A lower bound on the minimum weight of the syndrome class.  If
            ``within_max_weight`` is False, it exceeds ``max_weight``.
    """

    within_max_weight: bool
    representative: int
    lower_bound: int


class PauliCosetSearch:
    """Minimum-weight oracle for Pauli errors modulo the stabilizers of a pure stabilizer state.

    The syndrome of a Pauli error on ``n`` qubits says which of the ``n`` stabilizers it
    anticommutes with.  Two errors have the same syndrome exactly when they differ by a stabilizer,
    so the errors with a given syndrome form a syndrome class of equivalent errors.  The oracle
    determines whether a syndrome class has an error of weight at most a given maximum weight.

    The oracle searches a syndrome class, which is a coset of the stabilizers, with the
    Brouwer-Zimmermann (BZ) algorithm (see ``brouwer_zimmermann.md``) in compiled code, as
    ``_search_coset`` describes.  ``get_min_weight_bound`` first looks up errors of weight at most
    one in tables of single-qubit syndromes.

    A Pauli is packed into an integer with its X part in the low ``n`` bits and its Z part in the
    next ``n`` bits, and a syndrome is packed into an integer with bit ``j`` for stabilizer ``j``.

    Args:
        stabilizers: An independent commuting binary ``n x 2n`` matrix in ``[X|Z]`` order.
        css_weight: True to weigh an error by its CSS weight ``max(|x|, |z|)``, where ``x`` and
            ``z`` are its X and Z parts, or False to weigh it by its physical weight, the number of
            qubits that it acts on.
    """

    def __init__(self, stabilizers: npt.ArrayLike, *, css_weight: bool = False) -> None:
        self.css_weight = css_weight
        assert_binary(stabilizers, "stabilizers")
        matrix = as_binary_matrix(stabilizers)
        self.num_qubits = len(matrix)
        n = self.num_qubits
        if not n or matrix.shape[1] != 2 * n:
            raise ValueError("Require a complete nonempty n by 2n stabilizer matrix")
        checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
        _, labels, pivots = row_reduce_binary(checks, np.eye(n, dtype=np.uint8))
        if len(pivots) != n or np.any((checks @ matrix.T) % 2):
            raise ValueError("Stabilizers must be independent and commute")
        assert labels is not None

        # Row j of the inverse is an error whose syndrome has only bit j set.
        inverse = np.zeros((n, 2 * n), dtype=np.uint8)
        inverse[:, pivots] = labels.T
        self._inverse = _row_ints(inverse)
        self._checks = _row_ints(checks)

        # Map each syndrome of a single-qubit error, or of no error, to such an error.
        self._singles = {0: 0}
        self._x_singles = {0: 0}
        self._z_singles = {0: 0}
        for qubit in range(n):
            x, z = 1 << qubit, 1 << (qubit + n)
            sx, sz = self._get_syndrome(x), self._get_syndrome(z)
            self._x_singles.setdefault(sx, x)
            self._z_singles.setdefault(sz, z)
            self._singles.setdefault(sx, x)
            self._singles.setdefault(sz, z)
            self._singles.setdefault(sx ^ sz, x ^ z)

        # The tables of ``_search_coset``, which runs on embedded words (see ``_search_coset``).
        embedded_stabilizers = matrix if css_weight else symplectic_to_hamming(matrix)
        embedded_inverse = inverse if css_weight else symplectic_to_hamming(inverse)
        family = _information_sets.get_disjoint_family(embedded_stabilizers, None)
        num_sets, _, num_columns = family.generators.shape
        generators = pack_bits(family.generators.reshape(-1, num_columns)).reshape(num_sets, n, -1)
        # Entry ``w`` of the certificate bounds the Hamming weight of every word that adds more than
        # ``w`` rows on every information set, and an error weighs at least half of the Hamming
        # weight of its embedded word, by either measure of weight.  The certificate takes no weight
        # divisor, since the parity of the stabilizers need not hold on their cosets, and a coverage
        # of one, since the information sets are disjoint.  After layer ``n``, every word has been
        # visited.
        certificate_bounds = _information_sets.get_certificate_lower_bounds(n, family.ranks, 1, 1)
        unvisited_bounds = (certificate_bounds + 1) // 2
        unvisited_bounds[n] = np.iinfo(np.int64).max
        x_mask = _int_to_words((1 << n) - 1, generators.shape[2])
        self._tables = _CosetSearchTables(
            pack_bits(embedded_inverse), generators, family.pivots, unvisited_bounds, x_mask
        )

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether some error with this syndrome has weight at most ``max_weight``.

        Args:
            syndrome: A packed syndrome (see ``PauliCosetSearch``).
            max_weight: The largest weight that the query accepts.

        Returns:
            A ``MinWeightBound``.

        Raises:
            ValueError: If the syndrome is out of range.
        """
        if not 0 <= syndrome < 1 << self.num_qubits:
            raise ValueError("The syndrome is out of range")
        pauli = self._weight_one_representative(syndrome)
        if pauli is not None:
            return MinWeightBound(self._weight(pauli) <= max_weight, pauli, int(bool(syndrome)))
        if max_weight < 2:
            return MinWeightBound(False, self._representative(syndrome), 2)

        # The syndrome class has no error of weight at most one, so two is a lower bound.
        best = np.empty(self._tables.inverse.shape[1], dtype=np.uint64)
        words = _int_to_words(syndrome, (self.num_qubits + 63) // 64)
        lower_bound = int(_search_coset(words, max_weight, best, *self._tables, self.css_weight))
        return MinWeightBound(lower_bound <= max_weight, self._pauli(best), max(2, lower_bound))

    def find_first_not_within_max_weight(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> tuple[int, MinWeightBound | None]:
        """Find the first syndrome whose class has no error of weight at most ``max_weight``.

        Args:
            syndromes: One packed syndrome (see ``PauliCosetSearch``) per row, split into 64-bit
                machine words, with machine word ``j`` holding bits ``64 j`` and up.  Each syndrome
                must be below ``2**num_qubits``; this is not checked.
            max_weight: The largest weight that the query accepts.

        Returns:
            The index of that syndrome, or the number of syndromes if there is none, and the
            ``MinWeightBound`` of that syndrome, or None if there is none.  For ``max_weight`` of
            two or more, the bound is the one that ``get_min_weight_bound`` gives.
        """
        best = np.empty(self._tables.inverse.shape[1], dtype=np.uint64)
        index, lower_bound = _find_first_not_within_max_weight(
            syndromes, max_weight, best, *self._tables, self.css_weight
        )
        if index == len(syndromes):
            return index, None
        return index, MinWeightBound(False, self._pauli(best), int(lower_bound))

    def _get_syndrome(self, pauli: int) -> int:
        """Compute the syndrome of a packed Pauli."""
        return _get_parities(self._checks, pauli)

    def _representative(self, syndrome: int) -> int:
        """Find a packed Pauli with this syndrome."""
        return _xor_selected(self._inverse, syndrome)

    def _weight_one_representative(self, syndrome: int) -> int | None:
        """Find an error of weight zero or one with this syndrome, if there is one."""
        if syndrome in self._singles:
            return self._singles[syndrome]
        if not self.css_weight:
            return None
        # CSS weight one also allows X_i Z_j on distinct qubits.  Callers such as the certification
        # of fault propagation make a query for each of thousands of syndromes, so a table of these
        # errors, built on the first query that needs it, saves a scan of the qubits per query.
        if self._css_singles is not None:
            return self._css_singles.get(syndrome)
        # Without the table, matching an X-single syndrome with a Z-single syndrome takes linear
        # time.
        for sx, x in self._x_singles.items():
            z = self._z_singles.get(syndrome ^ sx)
            if z is not None:
                return x ^ z
        return None

    @functools.cached_property
    def _css_singles(self) -> dict[int, int] | None:
        """Errors X_i Z_j (with X_i or Z_j possibly the identity) by syndrome, if not too many."""
        if len(self._x_singles) * len(self._z_singles) > _MAX_CSS_SINGLES:
            return None
        table: dict[int, int] = {}
        for sx, x in self._x_singles.items():
            for sz, z in self._z_singles.items():
                table.setdefault(sx ^ sz, x ^ z)
        return table

    def _pauli(self, best: npt.NDArray[np.uint64]) -> int:
        """The packed Pauli of the embedded word that ``_search_coset`` left in ``best``."""
        return _packed_row_int(best) & ((1 << (2 * self.num_qubits)) - 1)

    def _weight(self, pauli: int) -> int:
        mask = (1 << self.num_qubits) - 1
        x, z = pauli & mask, (pauli >> self.num_qubits) & mask
        return max(x.bit_count(), z.bit_count()) if self.css_weight else (x | z).bit_count()


class CssCosetSearch:
    """Minimum-weight oracle for Pauli errors on a CSS state, weighed by their CSS weight.

    The X part ``x`` and Z part ``z`` of a packed Pauli are its low and high ``n`` bits (see
    ``PauliCosetSearch``).  A pure-X stabilizer has no Z part, and a pure-Z stabilizer no X part.  A
    stabilizer state is CSS if pure-X and pure-Z stabilizers generate its stabilizer group.

    This oracle answers the same queries as ``PauliCosetSearch`` with ``css_weight=True``, with the
    same packing of Paulis and syndromes, by two classical searches.  The X parts of the pure-X
    stabilizers span a binary code ``C_X``, and the Z parts of the pure-Z stabilizers a binary code
    ``C_Z``.  Multiplying an error by a stabilizer adds a word of ``C_X`` to its X part and a word
    of ``C_Z`` to its Z part, independently.  So the smallest CSS weight ``max(|x|, |z|)`` of an
    error with a given syndrome is the larger of two classical minima: the smallest Hamming weight
    in the coset ``x + C_X``, and the smallest in the coset ``z + C_Z``.  The oracle finds each with
    the search of ``_search_coset`` on ``n`` columns, rather than one search on ``2 n`` columns.

    The coset of an X part is labeled by its syndrome for the pure-Z stabilizers, and that of a Z
    part by its syndrome for the pure-X stabilizers.  The oracle maps a syndrome for the given
    stabilizers to these two classical syndromes.  The tables of the two searches depend only on the
    stabilizer group, so they are cached in-process for the last few groups, and certifying many
    circuits that prepare the same state builds them only once.

    Args:
        stabilizers: An independent commuting binary ``n x 2n`` matrix in ``[X|Z]`` order, whose
            stabilizer group is CSS.

    Raises:
        ValueError: If the stabilizers are not ``n`` independent commuting stabilizers of a CSS
            state.
    """

    def __init__(self, stabilizers: npt.ArrayLike) -> None:
        # Callers such as the certification of fault propagation read ``num_qubits`` and
        # ``css_weight`` from either oracle.
        self.css_weight = True
        assert_binary(stabilizers, "stabilizers")
        matrix = as_binary_matrix(stabilizers)
        self.num_qubits = len(matrix)
        n = self.num_qubits
        if not n or matrix.shape[1] != 2 * n:
            raise ValueError("Require a complete nonempty n by 2n stabilizer matrix")
        checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
        if len(row_reduce_binary(matrix)[2]) != n or np.any((checks @ matrix.T) % 2):
            raise ValueError("Stabilizers must be independent and commute")
        combinations = _get_css_combinations(matrix)
        if combinations is None:
            raise ValueError("The stabilizers must generate a CSS state")

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
        self._x_search, self._z_search = _get_css_searches(
            n, len(x_rows), pack_bits(x_rows).tobytes(), pack_bits(z_rows).tobytes()
        )

        # Row ``j`` of ``z_rows`` is the Z part of a pure-Z stabilizer, the product of the given
        # stabilizers with coefficients ``c = (z_transform @ z_combinations)[j]``.  An error's
        # syndrome bit for that stabilizer is the parity of its X part with the row, and also the
        # parity of its syndrome for the given stabilizers with ``c``.  So the rows ``c`` form the X
        # syndrome map, from a syndrome for the given stabilizers to the syndrome of the X part, and
        # likewise for Z.  Each map is stored as its packed columns, one per syndrome bit.
        x_syndrome_map = (z_transform @ z_combinations) % 2
        z_syndrome_map = (x_transform @ x_combinations) % 2
        self._x_map_columns = pack_bits(np.ascontiguousarray(x_syndrome_map.T))
        self._z_map_columns = pack_bits(np.ascontiguousarray(z_syndrome_map.T))

        # A single syndrome is mapped by XORing the columns of both maps for its set bits, which is
        # cheaper than a parity for each row, since most syndromes have few set bits.  Entry ``i``
        # holds column ``i`` of the X map in its low ``_num_x_bits`` bits, and column ``i`` of the Z
        # map above them.
        self._num_x_bits = len(z_rows)
        self._map_columns = _row_ints(
            np.ascontiguousarray(np.vstack([x_syndrome_map, z_syndrome_map]).T)
        )

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether some error with this syndrome has CSS weight at most ``max_weight``.

        Args:
            syndrome: A packed syndrome (see ``PauliCosetSearch``).
            max_weight: The largest weight that the query accepts.

        Returns:
            A ``MinWeightBound``, whose lower bound is the larger of those of the X and Z parts.

        Raises:
            ValueError: If the syndrome is out of range.
        """
        if not 0 <= syndrome < 1 << self.num_qubits:
            raise ValueError("The syndrome is out of range")
        part_syndromes = _xor_selected(self._map_columns, syndrome)
        x_syndrome = part_syndromes & ((1 << self._num_x_bits) - 1)
        z_syndrome = part_syndromes >> self._num_x_bits
        x_bound = self._x_search.get_min_weight_bound(x_syndrome, max_weight)
        z_bound = self._z_search.get_min_weight_bound(z_syndrome, max_weight)
        return MinWeightBound(
            x_bound.within_max_weight and z_bound.within_max_weight,
            x_bound.representative | (z_bound.representative << self.num_qubits),
            max(x_bound.lower_bound, z_bound.lower_bound),
        )

    def find_first_not_within_max_weight(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> tuple[int, MinWeightBound | None]:
        """Find the first syndrome whose class has no error of CSS weight at most ``max_weight``.

        Args:
            syndromes: One packed syndrome (see ``PauliCosetSearch``) per row, split into 64-bit
                machine words, with machine word ``j`` holding bits ``64 j`` and up.  Each syndrome
                must be below ``2**num_qubits``; this is not checked.  The rows may have fewer words
                than ``num_qubits`` bits need, if every syndrome fits in them.
            max_weight: The largest weight that the query accepts.

        Returns:
            The index of that syndrome, or the number of syndromes if there is none, and the
            ``MinWeightBound`` that ``get_min_weight_bound`` gives for that syndrome, or None if
            there is none.
        """
        # A syndrome's class fails, that is, has no error of CSS weight at most ``max_weight``,
        # exactly when the coset of its X part or that of its Z part has no word of Hamming weight
        # at most ``max_weight``.  So find the first syndrome whose X-part coset fails, and then
        # search the Z-part cosets of the syndromes before it.
        index = len(syndromes)
        for search, map_columns in (
            (self._x_search, self._x_map_columns),
            (self._z_search, self._z_map_columns),
        ):
            part_syndromes = _map_syndromes(syndromes[:index], map_columns)
            best = np.empty(search.tables.inverse.shape[1], dtype=np.uint64)
            # With an ``x_mask`` that selects every column, the CSS weight is the Hamming weight.
            index, _ = _find_first_not_within_max_weight(
                part_syndromes, max_weight, best, *search.tables, True
            )
        if index == len(syndromes):
            return index, None
        return index, self.get_min_weight_bound(_packed_row_int(syndromes[index]), max_weight)


def get_coset_search(
    stabilizers: npt.ArrayLike, *, css_weight: bool = False
) -> PauliCosetSearch | CssCosetSearch:
    """Build an exact minimum-weight oracle for the Pauli errors on a stabilizer state.

    For the CSS weight on a CSS state, this is a ``CssCosetSearch``, which runs two searches on
    ``n`` columns, and otherwise a ``PauliCosetSearch``, which runs one search on ``2 n`` columns.
    Both give the same ``within_max_weight`` for every query, but their representatives and lower
    bounds can differ.

    Args:
        stabilizers: An independent commuting binary ``n x 2n`` matrix in ``[X|Z]`` order.
        css_weight: True to weigh an error by its CSS weight, or False to weigh it by its physical
            weight (see ``PauliCosetSearch``).
    """
    assert_binary(stabilizers, "stabilizers")
    matrix = as_binary_matrix(stabilizers)
    if css_weight and len(matrix) and _get_css_combinations(matrix) is not None:
        return CssCosetSearch(matrix)
    return PauliCosetSearch(matrix, css_weight=css_weight)


class _CosetSearchTables(NamedTuple):
    """The tables of one search of ``_search_coset``.

    The search is that of a ``PauliCosetSearch``, on embedded words of errors, or one of the two
    classical searches of a ``CssCosetSearch``, on words of a binary code, which take the place of
    embedded words below.

    Attributes:
        inverse: Row ``j`` is the packed embedded word of an error whose syndrome has only bit ``j``
            set.
        generators: The packed rows of each disjoint information set of the embedded stabilizers.
        pivots: The pivot columns of each information set, padded with -1.
        unvisited_bounds: Entry ``w`` bounds the weight of every error that the search has not
            visited after layer ``w``.
        x_mask: Selects the bits of a packed embedded word that count as its X part.  A classical
            search selects every column, so that the CSS weight of a word is its Hamming weight.
    """

    inverse: npt.NDArray[np.uint64]
    generators: npt.NDArray[np.uint64]
    pivots: npt.NDArray[np.int64]
    unvisited_bounds: npt.NDArray[np.int64]
    x_mask: npt.NDArray[np.uint64]


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


class _ClassicalCosetSearch(NamedTuple):
    """The search for the lightest words in the cosets of one binary code, for ``CssCosetSearch``.

    The code is the row space of a matrix of generators.  Each coset is labeled by the syndrome of
    its words, whose bit ``j`` is the parity of a word with row ``j`` of a matrix of checks.  Words
    are weighed by their Hamming weight.

    Attributes:
        tables: The tables of ``_search_coset`` for the code, whose ``x_mask`` selects every column,
            so that the CSS weight that the search computes is the Hamming weight.
        inverse_rows: Row ``j`` of ``tables.inverse`` as an integer, a word whose syndrome has only
            bit ``j`` set.
        singles: A map from each syndrome of a word of weight zero or one to such a word.
    """

    tables: _CosetSearchTables
    inverse_rows: tuple[int, ...]
    singles: dict[int, int]

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether the coset of this syndrome has a word of weight at most ``max_weight``.

        Returns:
            A ``MinWeightBound`` for the coset, whose representative is a word of the coset.
        """
        word = self.singles.get(syndrome)
        if word is not None:
            weight = word.bit_count()
            return MinWeightBound(weight <= max_weight, word, weight)
        if max_weight < 2:
            return MinWeightBound(False, _xor_selected(self.inverse_rows, syndrome), 2)

        # The coset has no word of weight at most one, so two is a lower bound.  With an ``x_mask``
        # that selects every column, the CSS weight is the Hamming weight.
        best = np.empty(self.tables.inverse.shape[1], dtype=np.uint64)
        words = _int_to_words(syndrome, max(1, (len(self.inverse_rows) + 63) // 64))
        lower_bound = int(_search_coset(words, max_weight, best, *self.tables, True))
        return MinWeightBound(lower_bound <= max_weight, _packed_row_int(best), max(2, lower_bound))


@functools.lru_cache(maxsize=8)
def _get_css_searches(
    num_qubits: int, num_x_rows: int, x_rows: bytes, z_rows: bytes
) -> tuple[_ClassicalCosetSearch, _ClassicalCosetSearch]:
    """The searches of the X parts and of the Z parts of the errors on a CSS state.

    ``x_rows`` holds the X parts of ``num_x_rows`` pure-X stabilizers, and ``z_rows`` the Z parts of
    the other ``num_qubits - num_x_rows`` stabilizers, which are pure-Z.  Both are in reduced row
    echelon form, which depends only on the stabilizer group, and packed by ``pack_bits`` into
    bytes, which ``lru_cache`` can hash.  The X parts of errors are searched modulo the code of the
    X rows, with the Z rows as checks, and the Z parts the other way around (see
    ``CssCosetSearch``).
    """
    num_words = max(1, (num_qubits + 63) // 64)
    x_matrix = unpack_bits(
        np.frombuffer(x_rows, dtype=np.uint64).reshape(num_x_rows, num_words), num_qubits
    )
    z_matrix = unpack_bits(
        np.frombuffer(z_rows, dtype=np.uint64).reshape(num_qubits - num_x_rows, num_words),
        num_qubits,
    )
    return _get_classical_search(x_matrix, z_matrix), _get_classical_search(z_matrix, x_matrix)


def _get_classical_search(
    generators: npt.NDArray[np.uint8], checks: npt.NDArray[np.uint8]
) -> _ClassicalCosetSearch:
    """Build the search of the cosets of the row space of independent generators.

    The row space of the generators is the null space of the independent checks, so each coset is
    labeled by its syndrome for the checks.
    """
    num_generators, num_columns = generators.shape
    width = max(1, (num_columns + 63) // 64)

    # Row ``j`` of the inverse is a word whose syndrome has only bit ``j`` set.
    _, labels, pivots = row_reduce_binary(checks, np.eye(len(checks), dtype=np.uint8))
    assert labels is not None
    inverse = np.zeros((len(checks), num_columns), dtype=np.uint8)
    inverse[:, pivots] = labels.T

    # Words are weighed by their Hamming weight, so the certificate bounds the weight of every
    # unvisited word directly.  It takes no weight divisor, since the words of a coset need not
    # share the divisibility of the code, and a coverage of one, since the information sets are
    # disjoint.  Without generators, each coset is a single word, which the search visits on layer
    # zero of an empty information set.
    if num_generators:
        family = _information_sets.get_disjoint_family(generators, None)
        num_sets = family.generators.shape[0]
        packed_generators = pack_bits(family.generators.reshape(-1, num_columns)).reshape(
            num_sets, num_generators, -1
        )
        set_pivots = family.pivots
        unvisited_bounds = _information_sets.get_certificate_lower_bounds(
            num_generators, family.ranks, 1, 1
        ).astype(np.int64)
    else:
        packed_generators = np.zeros((1, 0, width), dtype=np.uint64)
        set_pivots = np.full((1, 0), -1, dtype=np.int64)
        unvisited_bounds = np.zeros(1, dtype=np.int64)
    unvisited_bounds[num_generators] = np.iinfo(np.int64).max
    x_mask = _int_to_words((1 << num_columns) - 1, width)
    tables = _CosetSearchTables(
        pack_bits(inverse), packed_generators, set_pivots, unvisited_bounds, x_mask
    )

    # Map the syndrome of each word of weight zero or one to such a word.
    check_columns = _row_ints(np.ascontiguousarray(checks.T))
    singles = {0: 0}
    for column, syndrome in enumerate(check_columns):
        singles.setdefault(syndrome, 1 << column)
    return _ClassicalCosetSearch(tables, _row_ints(inverse), singles)


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


def _get_parities(rows: Sequence[int], value: int) -> int:
    """The integer whose bit ``j`` is the parity of ``value`` with row ``j``."""
    return sum(((row & value).bit_count() & 1) << jj for jj, row in enumerate(rows))


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
    """Search the class of each syndrome in turn, as ``_search_coset`` does.

    Return the index of the first syndrome whose class has no error of weight at most
    ``max_weight``, and the lower bound that ``_search_coset`` returns for it, with its lightest
    error left in ``best``.  If there is no such syndrome, return the number of syndromes and zero.
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
    """Search the class of a syndrome for an error of weight at most ``max_weight``.

    The syndrome is split into 64-bit machine words, with machine word ``j`` holding bits ``64 j``
    and up.  The search runs on embedded words.  The embedded word of an error is its packed Pauli
    when errors are weighed by CSS weight, and its packed Pauli followed by ``x xor z`` when they
    are weighed by physical weight, so that its Hamming weight is twice the physical weight (see
    ``symplectic_to_hamming``).  The weight of an embedded word is that of its error.  The tables
    are those of ``_CosetSearchTables``: row ``j`` of ``inverse`` is the embedded word of an error
    whose syndrome has only bit ``j`` set, ``generators`` and ``pivots`` are the disjoint
    information sets of the embedded stabilizers, with pivots padded with -1, and ``x_mask`` selects
    the X part of an embedded word.

    The embedded words of the class form a coset of the embedded stabilizers.  The search starts
    from the sum of the rows of ``inverse`` for the set bits of the syndrome, an error with that
    syndrome.  The offset of an information set is this error reduced by rows of the set until it is
    zero on the set's pivot columns.  Every word of the class is the offset plus a unique sum of
    rows of the set, and has a one on a pivot column exactly where it adds that pivot's row, so the
    BZ certificate applies to the class as it does to a space.  Layer ``w`` of a set is its offset
    plus each sum of ``w`` of its rows, so layer zero is the offset itself.

    The search visits layer zero on every information set, then layer one, and so on.  It keeps the
    lightest word visited in ``best``, and stops at the first word of weight at most ``max_weight``.
    After layer ``w``, every unvisited word weighs at least ``unvisited_bounds[w]``, so the search
    also stops when that exceeds ``max_weight``.

    Return a lower bound on the weight of every error in the class: zero, which is at most
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
            for word in range(width):
                best[word] ^= inverse[bit, word]
            bits &= bits - np.uint64(1)
    best_weight = _get_weight(best, zero, x_mask, css_weight)
    if best_weight <= max_weight:
        return np.int64(0)

    # Reduce the error on each information set to its offset, which is zero on its pivot columns.
    offsets = np.empty((num_sets, width), dtype=np.uint64)
    for basis in range(num_sets):
        offsets[basis] = best
        for row in range(num_rows):
            pivot = pivots[basis, row]
            if pivot < 0:
                break
            if offsets[basis, pivot // 64] >> np.uint64(pivot % 64) & np.uint64(1):
                for word in range(width):
                    offsets[basis, word] ^= generators[basis, row, word]

    # The arguments of ``_visit_layer``, which visits one information set at a time.
    widths = np.full(1, width, dtype=np.int64)
    support = np.empty(num_rows, dtype=np.int64)
    prefixes = np.empty((num_rows, 1, width), dtype=np.uint64)
    for layer in range(num_rows + 1):
        for basis in range(num_sets):
            if layer == 0:
                best_weight = _keep_lighter(
                    offsets[basis], zero, best, best_weight, x_mask, css_weight
                )
            else:
                best_weight = _visit_layer(
                    offsets[basis],
                    generators[basis : basis + 1],
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
        if unvisited_bounds[layer] > max_weight:
            break
    return min(best_weight, unvisited_bounds[layer])


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
    ``support`` and ``prefixes`` are work space for ``advance_support``.  Keep the lightest word in
    ``best``, as ``_keep_lighter`` does, stop at the first word of weight at most ``max_weight``,
    and return the weight of ``best``.
    """
    num_rows = rows.shape[1]
    prefixes[0, 0] = offset
    support[0] = 0
    depth = advance_support(rows, widths, support, prefixes, 0)
    while depth >= 0:
        # Visit the remaining supports that differ from this one only in their last row.
        for row in range(support[depth], num_rows):
            best_weight = _keep_lighter(
                prefixes[depth, 0], rows[0, row], best, best_weight, x_mask, css_weight
            )
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
    for word in range(len(best)):
        best[word] = first[word] ^ second[word]
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
                for word in range(width):
                    mapped[row, word] ^= map_columns[bit, word]
                bits &= bits - np.uint64(1)
    return mapped
