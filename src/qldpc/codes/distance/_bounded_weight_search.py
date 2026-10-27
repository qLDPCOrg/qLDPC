# SPDX-License-Identifier: Apache-2.0

"""Exact minimum-weight queries for syndrome classes of Pauli errors on a stabilizer state."""

from __future__ import annotations

import functools
import itertools
import numbers
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import comb

import numpy as np
import numpy.typing as npt

from qldpc import math

from . import _information_sets
from ._binary import (
    as_binary_matrix,
    assert_binary,
    pack_bits,
    row_reduce_binary,
    symplectic_to_hamming,
)

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

    The oracle first looks up errors of weight at most one in tables of single-qubit syndromes.  A
    syndrome class with no such error has minimum weight at least two.  If the maximum weight is at
    least two, the oracle searches the syndrome class, which is a coset of the stabilizers, with the
    Brouwer-Zimmermann (BZ) algorithm (see ``brouwer_zimmermann.md``).

    A Pauli is packed into an integer with its X part in the low ``n`` bits and its Z part in the
    next ``n`` bits, and a syndrome is packed into an integer with bit ``j`` for stabilizer ``j``.

    Args:
        stabilizers: An independent commuting binary ``n x 2n`` matrix in ``[X|Z]`` order.
        css_weight: True to weigh an error by its CSS weight ``max(|x|, |z|)``, where ``x`` and
            ``z`` are its X and Z parts, or False to weigh it by its physical weight, the number of
            qubits that it acts on.
        batch_size: The largest number of candidates that a search evaluates at once.
        memory_limit: The approximate number of bytes that the oracle may use to keep the sums of
            rows that every search shares.  A smaller limit can make searches slower, but never
            changes their results.
    """

    def __init__(
        self,
        stabilizers: npt.ArrayLike,
        *,
        css_weight: bool = False,
        batch_size: int = 4096,
        memory_limit: int = 2**30,  # 1 GiB
    ) -> None:
        for value in (batch_size, memory_limit):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError("batch_size and memory_limit must be positive integers")
        self.css_weight = css_weight
        self.batch_size = batch_size
        self.memory_limit = memory_limit
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

        # The search runs on the embedded stabilizers (see ``_embed``).  Weighing a word by its CSS
        # weight needs a mask for each of its two sectors.
        self._embedded = matrix.copy() if css_weight else symplectic_to_hamming(matrix)
        if css_weight:
            width = (2 * n + 63) // 64
            self._sector_masks = tuple(
                _int_to_words(((1 << n) - 1) << shift, width) for shift in (0, n)
            )
        self._layers: dict[tuple[int, int], list[npt.NDArray[np.uint64]]] = {}
        self._layer_bytes = 0

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether some error with this syndrome has weight at most ``max_weight``.

        Args:
            syndrome: A packed syndrome (see ``PauliCosetSearch``).
            max_weight: The weight threshold.

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
        return self._search(syndrome, max_weight)

    def _search(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Search a syndrome class with no error of weight at most one.

        The search is for an error of weight at most ``max_weight``.
        """
        # Start from one error in the syndrome class, and reduce its embedded word on each basis to
        # get an offset for that basis.  Every word of the coset is that offset plus some rows of
        # that basis, and has a one on a pivot column exactly where it adds the row for that pivot.
        pauli = self._representative(syndrome)
        offsets = self._get_offsets(self._embed(pauli))

        # The low ``2 n`` bits of an embedded word are its packed Pauli, X part first.
        n = self.num_qubits
        mask = (1 << (2 * n)) - 1

        # Track the lightest error visited, starting with the one above.
        best_pauli, best_weight = pauli, self._weight(pauli)

        # Visit the coset in layers, where layer ``depth`` adds ``depth`` rows to the offset.
        for depth in range(n + 1):
            # Visit the layer on every basis in turn, since the certificate below needs all of them.
            for basis, offset in enumerate(offsets):
                # Visit the sums of ``depth`` rows of the basis in batches, add the offset to each
                # sum, and weigh the resulting words.
                for sums in self._iter_layer(basis, depth):
                    candidates = sums ^ offset
                    weights = self._candidate_weights(candidates)

                    # Remember the lightest word, and return it if its weight is at most
                    # ``max_weight``.  The lower bound is two, because the syndrome class has no
                    # error of weight at most one.
                    index = int(weights.argmin())
                    if weights[index] < best_weight:
                        best_weight = int(weights[index])
                        best_pauli = _packed_row_int(candidates[index]) & mask
                    if best_weight <= max_weight:
                        return MinWeightBound(True, best_pauli, 2)

            # Return if the unvisited words are too heavy.  Each of them adds more than ``depth``
            # rows on every basis, so the certificate bounds its Hamming weight from below, and its
            # weight is at least half of that, by either measure of weight.  The smaller of this
            # bound and the lowest weight visited is then a lower bound on every weight.
            if depth < n:
                unseen_lower = (self._certificate.lower_bound(depth) + 1) // 2
                if unseen_lower > max_weight:
                    return MinWeightBound(False, best_pauli, min(best_weight, unseen_lower))

        # Layer ``n`` can add every row, so every word of the coset has been visited.  None has
        # weight at most ``max_weight``, and the lowest weight visited is the minimum.
        return MinWeightBound(False, best_pauli, best_weight)

    def _iter_layer(self, basis: int, depth: int) -> Iterator[npt.NDArray[np.uint64]]:
        """Yield the sums of every set of ``depth`` rows of a basis, in batches.

        The sums do not depend on the syndrome, so every search shares them.  They are kept if they
        fit within the memory limit, and computed again for each search otherwise.
        """
        if (basis, depth) in self._layers:
            yield from self._layers[basis, depth]
            return
        packed = self._bases[basis][0]
        num_bytes = comb(len(packed), depth) * packed.shape[1] * 8
        batches = (
            np.bitwise_xor.reduce(packed[supports], axis=1)
            for supports in _iter_fixed_weight_supports(len(packed), depth, self.batch_size)
        )
        if self._layer_bytes + num_bytes > self.memory_limit:
            yield from batches
            return
        self._layer_bytes += num_bytes
        self._layers[basis, depth] = list(batches)
        yield from self._layers[basis, depth]

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

    def _candidate_weights(self, candidates: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
        """Compute the weights of packed embedded words."""
        if self.css_weight:
            return np.maximum(
                *(
                    np.bitwise_count(candidates & mask).sum(axis=-1, dtype=np.uint64)
                    for mask in self._sector_masks
                )
            )
        return np.bitwise_count(candidates).sum(axis=-1, dtype=np.uint64) // 2

    def _embed(self, pauli: int) -> int:
        """Embed a packed Pauli into the space in which the Hamming weight is searched."""
        if self.css_weight:
            return pauli
        n = self.num_qubits
        x, z = pauli & ((1 << n) - 1), pauli >> n
        return x | (z << n) | ((x ^ z) << (2 * n))

    def _get_offsets(self, embedded: int) -> list[npt.NDArray[np.uint64]]:
        """Reduce an embedded word on each basis, so that it is zero on the pivot columns."""
        offsets = []
        for packed, rows, pivots in self._bases:
            # Clear each pivot bit of the word by adding the row that has a one there.
            word = embedded
            for row, pivot in zip(rows, pivots):
                if (word >> pivot) & 1:
                    word ^= row
            offsets.append(_int_to_words(word, packed.shape[1]))
        return offsets

    @functools.cached_property
    def _bases(self) -> list[tuple[npt.NDArray[np.uint64], tuple[int, ...], list[int]]]:
        """The disjoint information sets of the embedded stabilizers, prepared on first use.

        Each has the packed generators, the generators as integers, and the pivots.
        """
        return [
            (pack_bits(item.generators), _row_ints(item.generators), item.pivots.tolist())
            for item in _information_sets.get_information_set_generators(self._embedded, None)
        ]

    @functools.cached_property
    def _certificate(self) -> _information_sets.Certificate:
        """The certificate for the information sets, prepared on first use."""
        ranks = tuple(len(pivots) for *_, pivots in self._bases)
        return _information_sets.Certificate(self.num_qubits, ranks)


####################################################################################################
# helper methods


def _iter_fixed_weight_supports(
    dimension: int, weight: int, batch_size: int
) -> Iterator[npt.NDArray[np.int_]]:
    """Yield every set of ``weight`` indices below ``dimension`` in lexicographic order, in batches.

    Each batch has at most ``batch_size`` sets, one per row.  Weight zero gives one empty set.
    """
    combinations = itertools.combinations(range(dimension), weight)
    while batch := list(itertools.islice(combinations, batch_size)):
        indices = itertools.chain.from_iterable(batch)
        yield np.fromiter(indices, dtype=np.min_scalar_type(dimension)).reshape(len(batch), weight)


def _row_ints(matrix: npt.NDArray[np.uint8]) -> tuple[int, ...]:
    """Pack each row of a binary matrix into an integer, with entry ``j`` in bit ``j``."""
    return tuple(map(_packed_row_int, pack_bits(matrix)))


def _packed_row_int(row: npt.NDArray[np.uint64]) -> int:
    """Convert a row of 64-bit words into an integer, with word ``j`` in bits ``64 j`` and up."""
    return sum(int(word) << (64 * jj) for jj, word in enumerate(row))


def _int_to_words(value: int, width: int) -> npt.NDArray[np.uint64]:
    """Split an integer into ``width`` 64-bit words, with word ``j`` in bits ``64 j`` and up."""
    return np.frombuffer(value.to_bytes(8 * width, "little"), dtype="<u8")


def _xor_selected(rows: Sequence[int], support: int) -> int:
    """XOR together the rows whose indices are the set bits of ``support``."""
    word = 0
    while support:
        # Add the row for the lowest set bit, and then clear that bit.
        low = support & -support
        word ^= rows[low.bit_length() - 1]
        support ^= low
    return word
