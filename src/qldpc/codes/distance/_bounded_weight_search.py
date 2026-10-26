# SPDX-License-Identifier: Apache-2.0

"""Exact minimum-weight queries for syndrome classes of Pauli errors on a stabilizer state."""

from __future__ import annotations

import itertools
import numbers
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from math import comb
from typing import Literal

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
        representative: An error in the syndrome class, packed into an integer with its X part in
            the low ``n`` bits and its Z part in the next ``n`` bits.  If ``within_max_weight`` is
            True, its weight is at most ``max_weight``.
        lower_bound: A lower bound on the minimum weight of the syndrome class.  If
            ``within_max_weight`` is False, it exceeds ``max_weight``.
    """

    within_max_weight: bool
    representative: int
    lower_bound: int


class PauliCosetSearch:
    """Minimum-weight oracle for Pauli errors modulo the stabilizers of a pure stabilizer state.

    The syndrome of a Pauli error says which of the ``n`` stabilizers it anticommutes with.  Two
    errors have the same syndrome exactly when they differ by a stabilizer, so the errors with a
    given syndrome form a syndrome class of equivalent errors.  The oracle determines whether a
    syndrome class has an error of weight at most a given maximum weight.

    The oracle first looks up errors of weight at most one in tables of single-qubit syndromes.  A
    syndrome class with no such error has minimum weight at least two.  If the maximum weight is at
    least two, the oracle searches the syndrome class, which is a coset of the stabilizers, with the
    Brouwer-Zimmermann (BZ) algorithm (see ``brouwer_zimmermann.md``).

    Paulis and syndromes are packed into integers: a Pauli has its X part in the low ``n`` bits and
    its Z part in the next ``n`` bits, and bit ``j`` of a syndrome is for stabilizer ``j``.

    Args:
        stabilizers: An independent commuting binary ``n x 2n`` matrix in ``[X|Z]`` order.
        metric: ``"pauli"`` counts the qubits that an error acts on, and ``"css"`` counts
            ``max(|x|, |z|)``, where ``x`` and ``z`` are the X and Z parts of the error.
        cache_size: The number of syndromes whose results are kept between queries, which saves the
            search when a caller repeats a syndrome.  The cache_size only affects runtime.
        batch_size: The largest number of candidates that a search evaluates at once, which limits
            its memory usage.
    """

    def __init__(
        self,
        stabilizers: npt.ArrayLike,
        *,
        metric: Literal["pauli", "css"] = "pauli",
        cache_size: int = 4096,
        batch_size: int = 4096,
    ) -> None:
        if metric not in ("pauli", "css"):
            raise ValueError("metric must be 'pauli' or 'css'")
        for value in (cache_size, batch_size):
            if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < 1:
                raise ValueError("cache_size and batch_size must be positive integers")
        self.metric = metric
        self.cache_size = int(cache_size)
        self.batch_size = int(batch_size)
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
        self._inverse = tuple(_binary_row_int(row) for row in inverse)
        self._checks = tuple(_binary_row_int(row) for row in checks)

        # Map each syndrome of a single-qubit error, or of no error, to such an error.
        self._singles = {0: 0}
        self._x_singles = {0: 0}
        self._z_singles = {0: 0}
        for qubit in range(n):
            x, z = 1 << qubit, 1 << (qubit + n)
            sx, sz = self.get_syndrome(x), self.get_syndrome(z)
            self._x_singles.setdefault(sx, x)
            self._z_singles.setdefault(sz, z)
            self._singles.setdefault(sx, x)
            self._singles.setdefault(sz, z)
            self._singles.setdefault(sx ^ sz, x ^ z)

        # Information sets of the embedded stabilizers are prepared on first use.  Each holds packed
        # generators, integer rows, and pivots.
        self._embedded = symplectic_to_hamming(matrix) if metric == "pauli" else matrix.copy()
        self._bases: list[tuple[npt.NDArray[np.uint64], tuple[int, ...], tuple[int, ...]]] = []
        self._certificate = _information_sets.Certificate(n, ())
        mask = (1 << n) - 1
        self._sector_masks = tuple(
            np.array(
                [(sector >> (64 * jj)) & ((1 << 64) - 1) for jj in range((2 * n + 63) // 64)],
                dtype=np.uint64,
            )
            for sector in (mask, mask << n)
        )
        self._cache: dict[int, MinWeightBound] = {}

    def get_syndrome(self, pauli: int) -> int:
        """Compute the syndrome of a Pauli error.

        Args:
            pauli: The error, packed into an integer with its X part in the low ``n`` bits and its Z
                part in the next ``n`` bits.
        """
        return sum(((row & pauli).bit_count() & 1) << jj for jj, row in enumerate(self._checks))

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether some error with this syndrome has weight at most ``max_weight``.

        Args:
            syndrome: A syndrome, packed into an integer with bit ``j`` for stabilizer ``j``.
            max_weight: A nonnegative integer.

        Returns:
            The answer, an error with this syndrome, and a lower bound on the minimum weight of the
            errors with this syndrome.

        Raises:
            ValueError: If the syndrome is out of range or ``max_weight`` is negative.
        """
        if not 0 <= syndrome < 1 << self.num_qubits:
            raise ValueError("The syndrome is out of range")
        if max_weight < 0:
            raise ValueError("The weight threshold must be nonnegative")
        known = self._cache.get(syndrome)
        if known is not None:
            weight = self._weight(known.representative)
            if weight <= max_weight or known.lower_bound > max_weight:
                return MinWeightBound(weight <= max_weight, known.representative, known.lower_bound)

        pauli = self._weight_one_representative(syndrome)
        if pauli is not None:
            bound = MinWeightBound(self._weight(pauli) <= max_weight, pauli, int(bool(syndrome)))
        elif max_weight < 2:
            bound = MinWeightBound(False, self._representative(syndrome), 2)
        else:
            bound = self._search(syndrome, max_weight)
        bound = self._merge_bounds(known, bound)
        self._cache[syndrome] = bound
        if len(self._cache) > self.cache_size:
            del self._cache[next(iter(self._cache))]
        return bound

    def _merge_bounds(self, known: MinWeightBound | None, found: MinWeightBound) -> MinWeightBound:
        """Combine two results into the lighter error and the larger lower bound."""
        if known is None:
            return found
        representative = min((known.representative, found.representative), key=self._weight)
        return MinWeightBound(
            found.within_max_weight, representative, max(known.lower_bound, found.lower_bound)
        )

    def _representative(self, syndrome: int) -> int:
        return _xor_selected(self._inverse, syndrome)

    def _weight_one_representative(self, syndrome: int) -> int | None:
        """Find an error of weight zero or one with this syndrome, if there is one."""
        if syndrome in self._singles:
            return self._singles[syndrome]
        if self.metric == "css":
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
        return max(x.bit_count(), z.bit_count()) if self.metric == "css" else (x | z).bit_count()

    def _candidate_weights(self, candidates: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
        """Compute the weights of packed embedded words, in the metric."""
        if self.metric == "pauli":
            return np.bitwise_count(candidates).sum(axis=-1, dtype=np.uint64) // 2
        return np.maximum(
            *(
                np.bitwise_count(candidates & mask).sum(axis=-1, dtype=np.uint64)
                for mask in self._sector_masks
            )
        )

    def _prepare_bases(self) -> None:
        """Pack the disjoint information sets of the embedded stabilizers, at most once."""
        if self._bases:
            return
        ranks = []
        for generators, _, rank, pivots in _information_sets.get_information_set_generators(
            self._embedded, None
        ):
            packed = pack_bits(generators)
            rows = tuple(_binary_row_int(row) for row in generators)
            self._bases.append((packed, rows, tuple(map(int, pivots))))
            ranks.append(rank)
        self._certificate = _information_sets.Certificate(self.num_qubits, tuple(ranks))

    def _embed(self, pauli: int) -> int:
        """Embed a packed Pauli into the space in which the Hamming weight is searched."""
        if self.metric == "css":
            return pauli
        n = self.num_qubits
        x, z = pauli & ((1 << n) - 1), pauli >> n
        return x | (z << n) | ((x ^ z) << (2 * n))

    def _get_offsets(self, embedded: int) -> list[npt.NDArray[np.uint64]]:
        """Reduce an embedded word on each basis, so that it is zero on the pivot columns.

        Every word of the coset is then this offset plus some rows of the basis, and the word has a
        one on a pivot column exactly where it adds the row for that pivot.
        """
        offsets = []
        for packed, rows, pivots in self._bases:
            # Clear each pivot bit of the word by adding the row that has a one there.
            word = embedded
            for row, pivot in zip(rows, pivots):
                if (word >> pivot) & 1:
                    word ^= row

            # Split the word into 64-bit words, as in the packed rows of the basis.
            offsets.append(
                np.array(
                    [(word >> (64 * jj)) & ((1 << 64) - 1) for jj in range(packed.shape[1])],
                    dtype=np.uint64,
                )
            )
        return offsets

    def _search(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Search a syndrome class with no error of weight at most one for one of ``max_weight``.

        The lower bounds that it returns are at least two.
        """
        # Start from one error in the syndrome class, and reduce its embedded word on each basis to
        # get an offset for that basis.  Every word of the coset is that offset plus some rows of
        # that basis.
        self._prepare_bases()
        pauli = self._representative(syndrome)
        offsets = self._get_offsets(self._embed(pauli))

        # Track the lowest weight visited and the error that has it.  Every weight is at most n, so
        # n + 1 means that no word has been visited.
        n = self.num_qubits
        visited_min = n + 1
        best_pauli = pauli

        # Visit the coset in layers, where layer ``depth`` adds ``depth`` rows to the offset.
        depth = 0
        while True:
            # Visit the layer on every basis in turn, since the certificate below needs all of them.
            for (packed, _, _), offset in zip(self._bases, offsets, strict=True):
                # Choose the sets of rows to add, in batches.  Depth zero has one empty set.
                supports_iter = (
                    [np.empty((1, 0), dtype=int)]
                    if depth == 0
                    else _iter_fixed_weight_supports(n, depth, self.batch_size)
                )
                for supports in supports_iter:
                    # Build the word for each set of rows in the batch, and weigh each word.
                    candidates = np.bitwise_xor.reduce(packed[supports], axis=1) ^ offset
                    weights = self._candidate_weights(candidates)

                    # Return the first word of weight at most ``max_weight``.  The lower bound is
                    # two, because the syndrome class has no error of weight at most one.
                    good = np.flatnonzero(weights <= max_weight)
                    if len(good):
                        # The low 2n bits of an embedded word are the packed Pauli, X part first.
                        word = _packed_row_int(candidates[good[0]])
                        return MinWeightBound(True, word & ((1 << (2 * n)) - 1), 2)

                    # Otherwise, remember the lightest word visited so far.
                    batch_min = int(weights.min())
                    if batch_min < visited_min:
                        visited_min = batch_min
                        word = _packed_row_int(candidates[int(weights.argmin())])
                        best_pauli = word & ((1 << (2 * n)) - 1)

            # After layer ``n``, which can add every row, every word of the coset has been visited.
            # None has weight at most ``max_weight``, and the lowest weight visited is the minimum.
            if depth == n:
                return MinWeightBound(False, best_pauli, visited_min)

            # Return if the unvisited words are too heavy.  Each of them adds more than ``depth``
            # rows on every basis, so the certificate bounds its Hamming weight from below, and its
            # weight is at least half of that in either metric.  The smaller of this bound and the
            # lowest weight visited is then a lower bound on every weight.
            unseen_lower = (self._certificate.lower_bound(depth) + 1) // 2
            if unseen_lower > max_weight:
                return MinWeightBound(False, best_pauli, min(visited_min, unseen_lower))

            # Otherwise, continue with the next layer.
            depth += 1


####################################################################################################
# helper methods


def _binary_row_int(row: npt.NDArray[np.uint8]) -> int:
    """Pack a binary row into an integer, with entry ``j`` in bit ``j``."""
    return int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")


def _iter_fixed_weight_supports(
    dimension: int, weight: int, batch_size: int
) -> Iterator[npt.NDArray[np.int_]]:
    """Yield every set of ``weight`` indices below ``dimension``, in batches of ``batch_size``."""
    # Rank the sets if the number of them fits in 63 bits, and stream them otherwise.
    num_supports = comb(dimension, weight)
    max_rank = np.iinfo(np.int64).max
    if num_supports <= max_rank:
        # For each position in a set, tabulate the binomial coefficients that unranking compares
        # against, capped so that they fit in 63 bits.
        binomial_tables = [
            np.asarray(
                [min(comb(value, index), max_rank) for value in range(dimension)],
                dtype=np.int64,
            )
            for index in range(1, weight + 1)
        ]

        # Take consecutive ranks one batch at a time, and turn each rank into a set of indices.
        for start in range(0, num_supports, batch_size):
            remainders = np.arange(
                start,
                min(start + batch_size, num_supports),
                dtype=np.int64,
            )
            supports = np.empty((len(remainders), weight), dtype=np.int_)

            # Unrank in the combinatorial number system.  From the last position ``index`` to the
            # first, the entry is the largest ``value`` with ``comb(value, index)`` at most the rank
            # that remains, which is then reduced by that coefficient.
            for index in range(weight, 0, -1):
                table = binomial_tables[index - 1]
                choices = np.searchsorted(table, remainders, side="right") - 1
                supports[:, index - 1] = choices
                remainders -= table[choices]
            yield supports
        return

    yield from _stream_fixed_weight_supports(dimension, weight, batch_size)


def _stream_fixed_weight_supports(
    dimension: int, weight: int, batch_size: int
) -> Iterator[npt.NDArray[np.int_]]:
    """Yield every set of ``weight`` indices below ``dimension`` in lexicographic order.

    This does not rank the sets, so it works when there are too many of them to rank in 63 bits.
    """
    # List the combinations lazily, as one flat stream of indices.
    flattened = itertools.chain.from_iterable(itertools.combinations(range(dimension), weight))

    # Take ``batch_size`` supports at a time from the stream, until it runs out.
    while True:
        flat_batch = np.fromiter(itertools.islice(flattened, batch_size * weight), dtype=np.int_)
        if not len(flat_batch):
            return

        # Put each support in its own row.
        yield flat_batch.reshape(-1, weight)


def _packed_row_int(row: npt.NDArray[np.uint64]) -> int:
    """Convert a row of 64-bit words into an integer, with word ``j`` in bits ``64 j`` and up."""
    return sum(int(word) << (64 * jj) for jj, word in enumerate(row))


def _xor_selected(rows: Sequence[int], support: int) -> int:
    """XOR together the rows whose indices are the set bits of ``support``."""
    word = 0
    while support:
        # Add the row for the lowest set bit, and then clear that bit.
        low = support & -support
        word ^= rows[low.bit_length() - 1]
        support ^= low
    return word
