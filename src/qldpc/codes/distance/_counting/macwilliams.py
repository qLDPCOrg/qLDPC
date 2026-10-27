# SPDX-License-Identifier: Apache-2.0

"""Exact logical counts of binary CSS stabilizer codes from the weight enumerators of their checks.

The X logicals of a CSS stabilizer code are the words of ``ker(H_Z)`` outside ``row(H_X)``.  The
kernel is the dual of ``row(H_Z)``, so the MacWilliams identity gives its number of words of weight
``w`` from the weight distribution ``A_Z`` of ``row(H_Z)``, and

    ``N_X(w) = 2**(-rank(H_Z)) * sum_j K_w(j) A_Z(j) - A_X(w)``,

where ``K_w`` is a binary Krawtchouk polynomial and ``A_X`` is the weight distribution of
``row(H_X)``.  The Z counts exchange X and Z.  Each weight distribution comes from a walk over all
``2**rank`` words of a check space, so the cost is exponential in the ranks of the checks rather
than in the number of logical qubits.  The transforms use 64-bit integers at the weights where a
bound rules out overflow, and exact Python integers at the others.

The walk visits the words of a span in reflected Gray-code order, as the brute-force distance search
in ``_brute_force.py`` does, but it reduces every word to a weight histogram in one compiled loop
instead of to a minimum weight over vectorized blocks.

References:

- https://en.wikipedia.org/wiki/MacWilliams_identity
- https://en.wikipedia.org/wiki/Kravchuk_polynomials
- https://arxiv.org/abs/quant-ph/9610040 (quantum MacWilliams identities)
"""

from __future__ import annotations

import functools
import math

import numpy as np
import numpy.typing as npt
from numba import njit

from .._binary import pack_bits, popcount_word, trailing_zeros_word

_BinaryMatrix = npt.NDArray[np.uint8]

# Gray-code steps visited per native call, which bounds the time between interruptions.
_CHUNK_SIZE = 1 << 16

# The walk indexes its steps, and counts the words of each weight, with unsigned 64-bit integers.
_MAX_RANK = 63


####################################################################################################
# logical counts


class CheckEnumerators:
    """Weight distributions of the X and Z check spaces of a binary CSS stabilizer code.

    Args:
        checks_x: Independent X checks.
        checks_z: Independent Z checks that commute with the X checks.  If they are the same array
            as ``checks_x``, one distribution serves both spaces.

    Raises:
        ValueError: If a check space has more than ``2**63`` words.
    """

    def __init__(self, checks_x: _BinaryMatrix, checks_z: _BinaryMatrix) -> None:
        for checks in (checks_x, checks_z):
            if len(checks) > _MAX_RANK:
                raise ValueError(
                    f"The MacWilliams method supports check ranks up to {_MAX_RANK}, not"
                    f" {len(checks)}; use 'brouwer_zimmermann' or 'brute_force'"
                )
        self.num_bits = checks_x.shape[1]
        self.rank_x, self.rank_z = len(checks_x), len(checks_z)
        self.weights_x = get_span_weights(checks_x)
        self.self_dual = checks_z is checks_x
        self.weights_z = self.weights_x if self.self_dual else get_span_weights(checks_z)
        # Counts of the light words of ker(H_Z) and ker(H_X), from one 64-bit transform each.
        self.kernel_x = _count_light_dual_words(self.weights_z, self.rank_z, self.num_bits)
        self.kernel_z = (
            self.kernel_x
            if self.self_dual
            else _count_light_dual_words(self.weights_x, self.rank_x, self.num_bits)
        )

    def count_logical_ops(self, weight: int) -> tuple[int, int]:
        """The numbers of nontrivial X-only and Z-only logical operators of the given weight."""
        count_x = self._count_sector(
            self.weights_z, self.rank_z, self.kernel_x, self.weights_x, weight
        )
        if self.self_dual:
            return count_x, count_x
        return count_x, self._count_sector(
            self.weights_x, self.rank_x, self.kernel_z, self.weights_z, weight
        )

    def _count_sector(
        self,
        dual_weights: npt.NDArray[np.uint64],
        dual_rank: int,
        light_counts: list[int],
        excluded_weights: npt.NDArray[np.uint64],
        weight: int,
    ) -> int:
        """Count the words of a kernel, minus those of the excluded checks, at one weight."""
        if weight < len(light_counts):
            count = light_counts[weight]
        else:
            count = _count_dual_words(dual_weights, dual_rank, self.num_bits, weight)
        count -= int(excluded_weights[weight])
        assert count >= 0
        return count


####################################################################################################
# weight distributions and transforms


def get_span_weights(generators: _BinaryMatrix) -> npt.NDArray[np.uint64]:
    """The number of words of each weight in the span of independent binary rows, as uint64s."""
    rank, num_bits = generators.shape
    weights = np.zeros(num_bits + 1, dtype=np.uint64)
    weights[0] = 1
    packed = pack_bits(generators)
    word = np.zeros(packed.shape[1], dtype=np.uint64)
    stop = 1 << rank
    for start in range(1, stop, _CHUNK_SIZE):
        chunk_stop = min(start + _CHUNK_SIZE, stop)
        _add_span_weights(packed, word, np.uint64(start), np.uint64(chunk_stop), weights)
    return weights


def _count_dual_words(
    span_weights: npt.NDArray[np.uint64], rank: int, num_bits: int, weight: int
) -> int:
    """The number of words of a given weight orthogonal to a span, by the MacWilliams identity."""
    krawtchouk = _get_krawtchouk_row(num_bits, weight)
    # Python integers keep the products and the sum exact at any length.
    total = sum(krawtchouk[jj] * int(count) for jj, count in enumerate(span_weights) if count)
    count, remainder = divmod(total, 1 << rank)
    assert not remainder and count >= 0
    return count


def _count_light_dual_words(
    span_weights: npt.NDArray[np.uint64], rank: int, num_bits: int
) -> list[int]:
    """The numbers of words of weights ``0, 1, ...`` orthogonal to a span, in 64-bit arithmetic.

    Since ``|K_w(j)| <= C(n, w)`` and the weight distribution ``A`` of the span sums to ``2**rank``,
    no product ``K_w(j) A(j)`` and no partial sum of ``sum_j K_w(j) A(j)`` exceeds
    ``2**rank * C(n, w)`` in magnitude.  Signed 64-bit arithmetic is therefore exact at the weights
    ``w`` below the first one at which this bound reaches ``2**63``, and the counts stop there.  At
    those weights, every Krawtchouk value is at most ``C(n, w) < 2**63`` in magnitude, and the rank
    is at most 62 because ``C(n, 0) = 1``, so the entries of the distribution, which are at most
    ``2**rank``, and the divisor ``2**rank`` also fit.  The counts are returned as Python integers.
    """
    krawtchouk = _get_light_krawtchouk_rows(num_bits, rank)
    counts = _transform_weights(krawtchouk, span_weights, rank)
    return counts.tolist()


@functools.lru_cache(maxsize=1024)
def _get_krawtchouk_row(num_bits: int, weight: int) -> tuple[int, ...]:
    """Binary Krawtchouk polynomial ``K_w(j)`` at every ``j``, for words of length ``n``.

    ``K_w(j) = sum_l (-1)**l C(j, l) C(n - j, w - l)`` is the sum of ``(-1)**(x.y)`` over the words
    ``x`` of weight ``w``, for any fixed ``y`` of weight ``j``.  The row satisfies the exact integer
    recurrence ``(n - j) K_w(j + 1) = (n - 2w) K_w(j) - j K_w(j - 1)``.
    """
    row = [math.comb(num_bits, weight)]
    if num_bits:
        lower = math.comb(num_bits - 1, weight - 1) if weight else 0
        row.append(math.comb(num_bits - 1, weight) - lower)
    for jj in range(1, num_bits):
        value, remainder = divmod(
            (num_bits - 2 * weight) * row[jj] - jj * row[jj - 1], num_bits - jj
        )
        assert not remainder
        row.append(value)
    return tuple(row)


@functools.lru_cache(maxsize=1024)
def _get_light_krawtchouk_rows(num_bits: int, rank: int) -> npt.NDArray[np.int64]:
    """Krawtchouk rows ``K_0, K_1, ...`` for a 64-bit transform of a span of the given rank.

    The rows stop at the first weight ``w`` with ``2**rank * C(n, w) >= 2**63``.  The returned array
    is read-only.
    """
    rows: list[tuple[int, ...]] = []
    for weight in range(num_bits + 1):
        if math.comb(num_bits, weight) << rank >= 1 << 63:
            break
        rows.append(_get_krawtchouk_row(num_bits, weight))
    matrix = np.array(rows, dtype=np.int64).reshape(len(rows), num_bits + 1)
    matrix.setflags(write=False)
    return matrix


@njit(cache=True)
def _add_span_weights(
    generators: npt.NDArray[np.uint64],
    word: npt.NDArray[np.uint64],
    start: np.uint64,
    stop: np.uint64,
    weights: npt.NDArray[np.uint64],
) -> None:
    """Continue a Gray-code walk over the span of packed generators, from step ``start``.

    Step ``s`` adds the generator indexed by the number of trailing zeros of ``s`` to ``word``, the
    word of the previous step, so steps ``1`` to ``2**rank - 1`` visit every nonzero word once.
    Each visited word increments the entry of ``weights`` for its weight.
    """
    if generators.shape[1] == 1:
        # Up to 64 bits fit in one word, which can stay in a register.
        value = word[0]
        for step in range(start, stop):
            value ^= generators[trailing_zeros_word(step), 0]
            weights[popcount_word(value)] += 1
        word[0] = value
        return
    for step in range(start, stop):
        index = trailing_zeros_word(step)
        weight = np.int64(0)
        for column in range(len(word)):
            word[column] ^= generators[index, column]
            weight += popcount_word(word[column])
        weights[weight] += 1


@njit(cache=True)
def _transform_weights(
    krawtchouk: npt.NDArray[np.int64], span_weights: npt.NDArray[np.uint64], rank: int
) -> npt.NDArray[np.int64]:
    """Apply Krawtchouk rows to a weight distribution, and divide by ``2**rank``, in int64.

    The caller guarantees that every Krawtchouk value, distribution entry, product and partial sum
    fits in int64, so converting the uint64 entries to int64 is exact.
    """
    counts = np.empty(len(krawtchouk), dtype=np.int64)
    for weight in range(len(krawtchouk)):
        total = np.int64(0)
        for ones in range(len(span_weights)):
            total += krawtchouk[weight, ones] * np.int64(span_weights[ones])
        count = total >> rank
        assert count << rank == total and count >= 0
        counts[weight] = count
    return counts
