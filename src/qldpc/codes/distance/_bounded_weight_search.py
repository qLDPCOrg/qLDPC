# SPDX-License-Identifier: Apache-2.0

"""Exact minimum-weight queries for syndrome classes of Pauli errors on a stabilizer state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc import math
from qldpc._compilation import always_compiled

from . import _information_sets
from ._binary import (
    as_binary_matrix,
    assert_binary,
    pack_bits,
    popcount_word,
    row_reduce_binary,
    symplectic_to_hamming,
    trailing_zeros_word,
)
from ._information_sets import advance_support

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
        pauli = _packed_row_int(best) & ((1 << (2 * self.num_qubits)) - 1)
        return MinWeightBound(lower_bound <= max_weight, pauli, max(2, lower_bound))

    def find_first_not_within_max_weight(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> int:
        """Find the first syndrome whose class has no error of weight at most ``max_weight``.

        Args:
            syndromes: One packed syndrome (see ``PauliCosetSearch``) per row, split into 64-bit
                machine words, with machine word ``j`` holding bits ``64 j`` and up.  Each syndrome
                must be below ``2**num_qubits``; this is not checked.
            max_weight: The largest weight that the query accepts.

        Returns:
            The index of that syndrome, or the number of syndromes if there is none.
        """
        return _find_first_not_within_max_weight(
            syndromes, max_weight, *self._tables, self.css_weight
        )

    def _get_syndrome(self, pauli: int) -> int:
        """Compute the syndrome of a packed Pauli."""
        return sum(((row & pauli).bit_count() & 1) << jj for jj, row in enumerate(self._checks))

    def _representative(self, syndrome: int) -> int:
        """Find a packed Pauli with this syndrome."""
        return _xor_selected(self._inverse, syndrome)

    def _weight_one_representative(self, syndrome: int) -> int | None:
        """Find an error of weight zero or one with this syndrome, if there is one."""
        if syndrome in self._singles:
            return self._singles[syndrome]
        if self.css_weight:
            # CSS weight one also allows X_i Z_j on distinct qubits.  Matching an X-single syndrome
            # with a Z-single syndrome finds these in linear time, without a quadratic table.
            for sx, x in self._x_singles.items():
                z = self._z_singles.get(syndrome ^ sx)
                if z is not None:
                    return x ^ z
        return None

    def _weight(self, pauli: int) -> int:
        mask = (1 << self.num_qubits) - 1
        x, z = pauli & mask, (pauli >> self.num_qubits) & mask
        return max(x.bit_count(), z.bit_count()) if self.css_weight else (x | z).bit_count()


class _CosetSearchTables(NamedTuple):
    """The tables of ``_search_coset`` for the stabilizers of a ``PauliCosetSearch``.

    Attributes:
        inverse: Row ``j`` is the packed embedded word of an error whose syndrome has only bit ``j``
            set.
        generators: The packed rows of each disjoint information set of the embedded stabilizers.
        pivots: The pivot columns of each information set, padded with -1.
        unvisited_bounds: Entry ``w`` bounds the weight of every error that the search has not
            visited after layer ``w``.
        x_mask: Selects the X part of a packed embedded word.
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


####################################################################################################
# compiled coset search


@always_compiled
def _find_first_not_within_max_weight(
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    unvisited_bounds: npt.NDArray[np.int64],
    x_mask: npt.NDArray[np.uint64],
    css_weight: bool,
) -> int:
    """Search the class of each syndrome in turn, as ``_search_coset`` does.

    Return the index of the first syndrome whose class has no error of weight at most
    ``max_weight``, or the number of syndromes if there is none.
    """
    best = np.empty(inverse.shape[1], dtype=np.uint64)
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
            return index
    return len(syndromes)


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
    syndrome.  The offset of an information set is this error reduced by rows of the set until it
    is zero on the set's pivot columns.  Every word of the class is the offset plus a unique sum of
    rows of the set, and has a one on a pivot column exactly where it adds that pivot's row, so the
    BZ certificate applies to the class as it does to a space.  Layer ``w`` of a set is its offset
    plus each sum of ``w`` of its rows, so layer zero is the offset itself.

    The search visits layer zero on every information set, then layer one, and so on.  It keeps the
    lightest word visited in ``best``, and stops at the first word of weight at most
    ``max_weight``.  After layer ``w``, every unvisited word weighs at least
    ``unvisited_bounds[w]``, so the search also stops when that exceeds ``max_weight``.

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
