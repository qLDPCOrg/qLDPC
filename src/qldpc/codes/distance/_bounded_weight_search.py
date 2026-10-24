# SPDX-License-Identifier: Apache-2.0

"""Exact internal searches for bounded kernel words and Pauli cosets.

Used by ``circuits.certify_state_prep_spread``; not public API.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import _information_sets
from ._binary import (
    as_binary_matrix,
    assert_binary,
    pack_bits,
    row_reduce_binary,
    symplectic_to_hamming,
)

####################################################################################################
# bounded kernel and affine Pauli searches


class BoundedKernelSearch:
    """Enumerate all nonzero binary kernel words through a weight bound with packed BZ coverage.

    Args:
        checks: Binary parity-check matrix; redundant and zero rows are allowed.
        max_weight: Hamming-weight limit.
        batch_size: Maximum coefficient supports per batch.

    Words are integers with column zero in the lowest bit; different information sets may yield
    duplicates.  Arbitrary filtering is safe: coverage uses no eligible-word upper bound or
    distance-search fallback.

    After coefficient weights through ``a`` in dimension ``k``, unseen words have weight at least
    ``sum(max(0, a + 1 - (k - r)) for r in ranks)``.  Each basis needs more than ``a`` coefficient
    bits, at most ``k-r`` outside its pivots; disjoint pivot sets let these bounds add.  Stop only
    when this bound exceeds ``max_weight`` or coefficients are exhausted.

    When dimension exceeds check rank, store only generator pivot coordinates to avoid a quadratic
    dense kernel matrix; coefficient support supplies free coordinates and their weight.  Otherwise
    reuse the distance search's disjoint information sets.
    """

    def __init__(self, checks: npt.ArrayLike, max_weight: int, *, batch_size: int = 4096) -> None:
        assert_binary(checks, "checks")
        matrix = as_binary_matrix(checks)
        if max_weight < 0 or batch_size < 1:
            raise ValueError("max_weight must be nonnegative and batch_size must be positive")
        reduced, _, pivots = row_reduce_binary(matrix)
        rank, length = len(pivots), matrix.shape[1]
        self.dimension = length - rank
        self.max_weight = min(max_weight, length)
        self.batch_size = batch_size
        self.generated = 0
        self._pivots = tuple(map(int, pivots))
        self._free = tuple(col for col in range(length) if col not in self._pivots)
        pivot_parts = reduced[:rank, self._free].T
        self._systematic = self.dimension > rank
        self.ranks: tuple[int, ...]
        if self._systematic:
            self.ranks = (self.dimension,)
            self._sets = [pack_bits(pivot_parts)]
        elif self.dimension:
            basis = np.zeros((self.dimension, length), dtype=np.uint8)
            basis[:, self._pivots] = pivot_parts
            basis[:, self._free] = np.eye(self.dimension, dtype=np.uint8)
            information_sets = _information_sets.get_information_set_generators(basis, None)
            self.ranks = tuple(rank for _, _, rank, _ in information_sets)
            self._sets = [pack_bits(rows) for rows, _, _, _ in information_sets]
        else:
            self.ranks = ()
            self._sets = []

    def words(self) -> Iterator[int]:
        """Yield bounded-weight words; exhaustion certifies coverage."""
        for depth in range(1, min(self.dimension, self.max_weight) + 1):
            for rows in self._sets:
                for supports in _iter_fixed_weight_supports(self.dimension, depth, self.batch_size):
                    self.generated += len(supports)
                    candidates = np.bitwise_xor.reduce(rows[supports], axis=1)
                    weights = np.bitwise_count(candidates).sum(axis=-1, dtype=np.uint64)
                    if self._systematic:
                        weights += depth
                    for index in np.flatnonzero(weights <= self.max_weight):
                        word = _packed_row_int(candidates[index])
                        if self._systematic:
                            word = sum(
                                ((word >> jj) & 1) << col for jj, col in enumerate(self._pivots)
                            ) | sum(1 << self._free[int(jj)] for jj in supports[index])
                        yield word
            if (
                _information_sets.Certificate(self.dimension, self.ranks).lower_bound(depth)
                > self.max_weight
            ):
                return


@dataclass(frozen=True)
class PauliWeightDecision:
    """A checked representative and global coset lower bound in the oracle's fixed metric.

    ``representative`` packs X in the low ``n`` bits and Z in the next ``n``; it need not minimize
    weight.  ``lower_bound`` bounds the entire coset at any threshold and exceeds the queried
    threshold when ``within_bound`` is False.
    """

    within_bound: bool
    representative: int
    lower_bound: int


class PauliCosetSearch:
    """Exact joint Pauli-weight oracle shared across queries for a pure stabilizer state.

    Args:
        stabilizers: An independent commuting binary ``n x 2n`` matrix in ``[X|Z]`` order.
        metric: ``"pauli"`` counts physical support; ``"css"`` counts ``max(|x|, |z|)`` in the
            physical frame, jointly over one representative even for non-CSS states.
        max_weight: The largest threshold that ``decide`` will be asked about, which only guides the
            choice of search bases; larger thresholds are still decided exactly.  Defaults to the
            number of qubits.
        cache_size: Maximum cached syndrome decisions; at least one.
        batch_size: Maximum coefficient supports per packed affine BZ batch.

    Full syndromes identify cosets: a maximal isotropic space equals its symplectic orthogonal.  The
    physical embedding ``(x|z) -> (x|z|x+z)`` has twice the Pauli weight, including Y.  CSS uses
    ``[X|Z]``, whose Hamming weight is at most twice ``max(|x|, |z|)``.

    Zero each affine offset on its basis pivots so pivot coordinates equal coefficient bits.  The
    information-set bound of the distance search then applies from coefficient weight zero, with
    overlapping bases when they certify the largest threshold with less work.
    A negative decision requires both visited weights and the unseen bound to exceed the threshold.
    For minimum visited weight ``v`` and unseen Hamming bound ``L``, the global metric bound is
    ``min(v, ceil(L / 2))``; exhaustion gives the exact visited minimum.  An unseen-word bound alone
    is never published as a global bound.

    Eliminate once and prepare information sets lazily, at most once.  Identity and single-qubit
    syndrome tables use linear space.  CSS weight one also includes ``X_i Z_j`` on distinct qubits:
    matching X-single and Z-single syndromes checks these in linear time without a quadratic table.
    Only this complete search, not physical singles alone, can prove a CSS lower bound of two.

    The bounded cache stores checked upper bounds and proved lower bounds.  Merging recomputed
    decisions never loses a proved bound; eviction affects runtime, not correctness.  Negative
    decisions are exact, never heuristic or probabilistic.
    """

    def __init__(
        self,
        stabilizers: npt.ArrayLike,
        *,
        metric: Literal["pauli", "css"] = "pauli",
        max_weight: int | None = None,
        cache_size: int = 4096,
        batch_size: int = 4096,
    ) -> None:
        if metric not in ("pauli", "css"):
            raise ValueError("metric must be 'pauli' or 'css'")
        self.metric = metric
        assert_binary(stabilizers, "stabilizers")
        matrix = as_binary_matrix(stabilizers)
        self.num_qubits = len(matrix)
        n = self.num_qubits
        if not n or matrix.shape[1] != 2 * n:
            raise ValueError("Require a complete nonempty n by 2n stabilizer matrix")
        checks = np.roll(matrix, n, axis=1)
        _, labels, pivots = row_reduce_binary(checks, np.eye(n, dtype=np.uint8))
        if len(pivots) != n or np.any((checks @ matrix.T) % 2):
            raise ValueError("Stabilizers must be independent and commute")
        if cache_size < 1 or batch_size < 1 or (max_weight is not None and max_weight < 0):
            raise ValueError(
                "cache_size and batch_size must be positive, and max_weight nonnegative"
            )
        assert labels is not None
        inverse = np.zeros((n, 2 * n), dtype=np.uint8)
        inverse[:, pivots] = labels.T
        self._inverse = tuple(_binary_row_int(row) for row in inverse)
        self._checks = tuple(_binary_row_int(row) for row in checks)
        self._embedded = symplectic_to_hamming(matrix) if metric == "pauli" else matrix.copy()
        mask = (1 << n) - 1
        self._sector_masks = tuple(
            np.array(
                [(sector >> (64 * jj)) & ((1 << 64) - 1) for jj in range((2 * n + 63) // 64)],
                dtype=np.uint64,
            )
            for sector in (mask, mask << n)
        )
        # Lazy bases hold packed generators, integer rows, and pivots.
        self._bases: list[tuple[npt.NDArray[np.uint64], tuple[int, ...], tuple[int, ...]]] = []
        self._certificate = _information_sets.Certificate(n, ())
        self._target = n if max_weight is None else min(max_weight, n)
        self.cache_size = cache_size
        self.batch_size = batch_size
        self._cache: dict[int, PauliWeightDecision] = {}
        self._singles = {0: 0}
        self._x_singles = {0: 0}
        self._z_singles = {0: 0}
        for qubit in range(n):
            x, z = 1 << qubit, 1 << (qubit + n)
            sx, sz = self.syndrome(x), self.syndrome(z)
            self._x_singles.setdefault(sx, x)
            self._z_singles.setdefault(sz, z)
            self._singles.setdefault(sx, x)
            self._singles.setdefault(sz, z)
            self._singles.setdefault(sx ^ sz, x ^ z)

    def syndrome(self, pauli: int) -> int:
        """Compute the complete syndrome of a packed symplectic Pauli."""
        return sum(((row & pauli).bit_count() & 1) << jj for jj, row in enumerate(self._checks))

    def decide(self, syndrome: int, max_weight: int) -> PauliWeightDecision:
        """Decide whether the coset has a representative within the weight threshold."""
        if not 0 <= syndrome < 1 << self.num_qubits or max_weight < 0:
            raise ValueError("Syndrome out of range or negative weight threshold")
        known = self._cache.get(syndrome)
        if known is not None:
            upper = self._weight(known.representative)
            if upper <= max_weight or known.lower_bound > max_weight:
                return PauliWeightDecision(
                    upper <= max_weight, known.representative, known.lower_bound
                )
        pauli = self._weight_one_representative(syndrome)
        if pauli is not None:
            decision = PauliWeightDecision(
                self._weight(pauli) <= max_weight, pauli, int(bool(syndrome))
            )
        elif max_weight < 2:
            decision = PauliWeightDecision(False, self._representative(syndrome), 2)
        else:
            decision = self._search(syndrome, max_weight)
        decision = self._merge_decisions(known, decision)
        self._cache[syndrome] = decision
        if len(self._cache) > self.cache_size:
            del self._cache[next(iter(self._cache))]
        return decision

    def _merge_decisions(
        self, known: PauliWeightDecision | None, found: PauliWeightDecision
    ) -> PauliWeightDecision:
        """Keep the lightest checked representative and strongest proved bound."""
        if known is None:
            return found
        representative = min((known.representative, found.representative), key=self._weight)
        return PauliWeightDecision(
            found.within_bound, representative, max(known.lower_bound, found.lower_bound)
        )

    def _representative(self, syndrome: int) -> int:
        return _xor_selected(self._inverse, syndrome)

    def _weight_one_representative(self, syndrome: int) -> int | None:
        if syndrome in self._singles:
            return self._singles[syndrome]
        if self.metric == "css":
            for sx, x in self._x_singles.items():
                z = self._z_singles.get(syndrome ^ sx)
                if z is not None:
                    return x ^ z
        return None

    def _weight(self, pauli: int) -> int:
        mask = (1 << self.num_qubits) - 1
        x, z = pauli & mask, (pauli >> self.num_qubits) & mask
        return max(x.bit_count(), z.bit_count()) if self.metric == "css" else (x | z).bit_count()

    def _candidate_weights(
        self,
        candidates: npt.NDArray[np.uint64],
    ) -> npt.NDArray[np.uint64]:
        if self.metric == "pauli":
            return np.bitwise_count(candidates).sum(axis=-1, dtype=np.uint64) // 2
        return np.maximum(
            *(
                np.bitwise_count(candidates & mask).sum(axis=-1, dtype=np.uint64)
                for mask in self._sector_masks
            )
        )

    def _prepare_bases(self) -> None:
        """Choose and pack search bases that certify the target weight quickly, at most once."""
        if self._bases:
            return
        # An embedded Hamming weight above twice the target bounds the metric weight above it.
        family = _information_sets.prepare_family(
            self._embedded, None, upper_bound=2 * self._target + 1
        )
        # Weights in a coset need not share the parity of its basis, so keep no weight divisor.
        self._certificate = replace(family.certificate, weight_divisor=1)
        for generators, _, _, pivots in family.information_sets:
            self._bases.append(
                (
                    pack_bits(generators),
                    tuple(_binary_row_int(row) for row in generators),
                    tuple(map(int, pivots)),
                )
            )

    def _embed(self, pauli: int) -> int:
        """Embed a Pauli for Hamming-weight bounds."""
        if self.metric == "css":
            return pauli
        n = self.num_qubits
        x, z = pauli & ((1 << n) - 1), pauli >> n
        return x | (z << n) | ((x ^ z) << (2 * n))

    def _get_offsets(self, embedded: int) -> list[npt.NDArray[np.uint64]]:
        """Pack affine offsets with each basis's pivots zeroed."""
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

    def _search(self, syndrome: int, threshold: int) -> PauliWeightDecision:
        """Scan affine layers for a witness or a proved global bound."""
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
                    good = np.flatnonzero(weights <= threshold)
                    if len(good):
                        # The low 2n bits of an embedded word are the packed (x|z) Pauli.
                        word = _packed_row_int(candidates[good[0]])
                        representative = word & ((1 << (2 * n)) - 1)
                        return PauliWeightDecision(True, representative, 2)
                    batch_min = int(weights.min())
                    if batch_min < visited_min:
                        visited_min = batch_min
                        word = _packed_row_int(candidates[int(weights.argmin())])
                        best_pauli = word & ((1 << (2 * n)) - 1)
            unseen_hamming = self._certificate.lower_bound(depth)
            # Either metric is at least half of the embedded Hamming weight.
            unseen_lower = (unseen_hamming + 1) // 2
            if depth == n:
                return PauliWeightDecision(False, best_pauli, visited_min)
            if unseen_lower > threshold:
                global_lower = min(visited_min, unseen_lower)
                return PauliWeightDecision(False, best_pauli, global_lower)
            depth += 1


####################################################################################################
# helper methods


def _iter_fixed_weight_supports(
    dimension: int, weight: int, batch_size: int
) -> Iterator[npt.NDArray[np.int_]]:
    """Yield batched coefficient supports.

    These searches need individual words, even those that do not improve a distance upper bound.
    """
    num_supports = math.comb(dimension, weight)
    max_rank = np.iinfo(np.int64).max
    if num_supports <= max_rank:
        binomial_tables = [
            np.asarray(
                [min(math.comb(value, index), max_rank) for value in range(dimension)],
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
    """Yield supports in lexicographic order, without ranking them, for counts beyond int64."""
    flattened = itertools.chain.from_iterable(itertools.combinations(range(dimension), weight))
    while True:
        flat_batch = np.fromiter(itertools.islice(flattened, batch_size * weight), dtype=np.int_)
        if not len(flat_batch):
            return
        yield flat_batch.reshape(-1, weight)


def _binary_row_int(row: npt.NDArray[np.uint8]) -> int:
    return int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")


def _packed_row_int(row: npt.NDArray[np.uint64]) -> int:
    return sum(int(word) << (64 * jj) for jj, word in enumerate(row))


def _xor_selected(rows: Sequence[int], support: int) -> int:
    word = 0
    while support:
        low = support & -support
        word ^= rows[low.bit_length() - 1]
        support ^= low
    return word
