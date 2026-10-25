# SPDX-License-Identifier: Apache-2.0

"""Exact minimum-weight queries for classes of equivalent Pauli errors on a stabilizer state."""

from __future__ import annotations

import itertools
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
    """What a query learned about the minimum weight of a class of equivalent Pauli errors.

    The query asks whether the class contains an error of weight at most ``max_weight``, where
    weight is measured in the oracle's metric.

    Attributes:
        within_max_weight: Whether the class contains such an error.
        representative: An error in the class, packed with its X part in the low ``n`` bits and
            its Z part in the next ``n``.  If ``within_max_weight`` is True, its weight is at most
            ``max_weight``.
        lower_bound: A proved lower bound on the minimum weight of the class.  If
            ``within_max_weight`` is False, it exceeds ``max_weight``.
    """

    within_max_weight: bool
    representative: int
    lower_bound: int


class PauliCosetSearch:
    """Minimum-weight oracle for Pauli errors modulo the stabilizers of a pure stabilizer state.

    The syndrome of a Pauli error lists whether it anticommutes with each of the ``n`` stabilizers.
    Two Pauli errors have the same syndrome exactly when they differ by a stabilizer, so a syndrome
    identifies a class of equivalent errors.  PauliCosetSearch determines whether the minimum weight
    of the Pauli errors in a syndrome class is no greater than any given maximum weight.

    Errors of weight at most one are found from tables of single-qubit syndromes, so a class without
    one has minimum weight at least two.  Otherwise, the oracle searches the class exactly, as in
    the Brouwer-Zimmermann distance algorithm.  It embeds the class as a coset of the stabilizers in
    a binary space, where the Hamming weight is twice the Pauli weight for ``metric="pauli"`` and at
    most twice the CSS weight for ``metric="css"``.  It then visits the words of the coset in layers
    of increasing coefficient weight on disjoint information sets, shifting the coset so that its
    offset is zero on the pivots of each set.  After each layer, the information sets certify a
    lower bound on the Hamming weight of every word that has not been visited.

    A search stops at the first word of weight at most the maximum weight.  Otherwise, it stops when
    no unvisited word can have such a weight, or when it has visited the whole coset.  If ``v`` is
    the lowest weight that it visited and ``L`` the certified Hamming weight of the unvisited words,
    ``min(v, ceil(L / 2))`` is a proved lower bound on the minimum weight of the class, and it is
    the minimum weight if the search visited the whole coset.  In the worst case, a search visits
    every word of the coset once for each information set.

    Args:
        stabilizers: An independent commuting binary ``n x 2n`` matrix in ``[X|Z]`` order.
        metric: ``"pauli"`` counts physical support, and ``"css"`` counts ``max(|x|, |z|)``, where
            ``x`` and ``z`` are the X and Z parts of a whole error.
        cache_size: The maximum number of syndromes whose best known error and proved lower bound
            are kept between queries, discarding the oldest first.  A kept result answers any
            query that it settles, and a later query never weakens it.  The cache size changes the
            run time of queries, not their correctness.
        batch_size: The maximum number of candidates that a search evaluates in one vectorized
            step, which bounds its working memory.
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
        if cache_size < 1 or batch_size < 1:
            raise ValueError("cache_size and batch_size must be positive")
        self.metric = metric
        self.cache_size = cache_size
        self.batch_size = batch_size
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

        # Information sets of the embedded stabilizers are prepared lazily.  Each holds packed
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
        """Compute the complete syndrome of a Pauli error.

        Args:
            pauli: A packed Pauli, with X in the low ``n`` bits and Z in the next ``n``.
        """
        return sum(((row & pauli).bit_count() & 1) << jj for jj, row in enumerate(self._checks))

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Bound the minimum weight of the errors with this syndrome against ``max_weight``.

        Args:
            syndrome: A packed syndrome, with bit ``j`` for stabilizer ``j``.
            max_weight: The weight threshold, which may be any nonnegative integer.

        Returns:
            Whether some error with this syndrome has weight at most ``max_weight``, together with
            an error that has this syndrome and a proved lower bound on the minimum weight.

        Raises:
            ValueError: If the syndrome is out of range or the threshold is negative.
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
        """Keep the lightest error and the strongest lower bound that have been found."""
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
        """Reduce an embedded word to its offset on each basis, with the pivots zeroed."""
        offsets = []
        for packed, rows, pivots in self._bases:
            word = embedded
            for row, pivot in zip(rows, pivots):
                if (word >> pivot) & 1:
                    word ^= row
            offsets.append(
                np.array(
                    [(word >> (64 * jj)) & ((1 << 64) - 1) for jj in range(packed.shape[1])],
                    dtype=np.uint64,
                )
            )
        return offsets

    def _search(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Search a class that has no error of weight at most one for one of weight ``max_weight``.

        Every lower bound that it returns is at least two.
        """
        self._prepare_bases()
        pauli = self._representative(syndrome)
        offsets = self._get_offsets(self._embed(pauli))
        n = self.num_qubits
        visited_min = n + 1
        best_pauli = pauli
        depth = 0
        while True:
            for (packed, _, _), offset in zip(self._bases, offsets, strict=True):
                supports_iter = (
                    [np.empty((1, 0), dtype=int)]
                    if depth == 0
                    else _iter_fixed_weight_supports(n, depth, self.batch_size)
                )
                for supports in supports_iter:
                    candidates = np.bitwise_xor.reduce(packed[supports], axis=1) ^ offset
                    weights = self._candidate_weights(candidates)
                    good = np.flatnonzero(weights <= max_weight)
                    if len(good):
                        # The low 2n bits of an embedded word are the packed (x|z) Pauli.
                        word = _packed_row_int(candidates[good[0]])
                        return MinWeightBound(True, word & ((1 << (2 * n)) - 1), 2)
                    batch_min = int(weights.min())
                    if batch_min < visited_min:
                        visited_min = batch_min
                        word = _packed_row_int(candidates[int(weights.argmin())])
                        best_pauli = word & ((1 << (2 * n)) - 1)
            if depth == n:
                return MinWeightBound(False, best_pauli, visited_min)
            # The weight is at least half of the Hamming weight in either metric.
            unseen_lower = (self._certificate.lower_bound(depth) + 1) // 2
            if unseen_lower > max_weight:
                return MinWeightBound(False, best_pauli, min(visited_min, unseen_lower))
            depth += 1


####################################################################################################
# helper methods


def _binary_row_int(row: npt.NDArray[np.uint8]) -> int:
    return int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")


def _iter_fixed_weight_supports(
    dimension: int, weight: int, batch_size: int
) -> Iterator[npt.NDArray[np.int_]]:
    """Yield every set of ``weight`` indices below ``dimension``, in batches of ``batch_size``."""
    num_supports = comb(dimension, weight)
    max_rank = np.iinfo(np.int64).max
    if num_supports <= max_rank:
        # Unrank each support from the combinatorial number system.
        binomial_tables = [
            np.asarray(
                [min(comb(value, index), max_rank) for value in range(dimension)],
                dtype=np.int64,
            )
            for index in range(1, weight + 1)
        ]
        for start in range(0, num_supports, batch_size):
            remainders = np.arange(
                start,
                min(start + batch_size, num_supports),
                dtype=np.int64,
            )
            supports = np.empty((len(remainders), weight), dtype=np.int_)
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
    """Yield the supports in lexicographic order, without ranking them, for more than 2**63."""
    flattened = itertools.chain.from_iterable(itertools.combinations(range(dimension), weight))
    while True:
        flat_batch = np.fromiter(itertools.islice(flattened, batch_size * weight), dtype=np.int_)
        if not len(flat_batch):
            return
        yield flat_batch.reshape(-1, weight)


def _packed_row_int(row: npt.NDArray[np.uint64]) -> int:
    return sum(int(word) << (64 * jj) for jj, word in enumerate(row))


def _xor_selected(rows: Sequence[int], support: int) -> int:
    word = 0
    while support:
        low = support & -support
        word ^= rows[low.bit_length() - 1]
        support ^= low
    return word
