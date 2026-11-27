# SPDX-License-Identifier: Apache-2.0

"""Exact logical counts of binary CSS stabilizer codes from the weight enumerators of their checks.

A Gray-code walk over each space of checks gives its weight distribution, and the MacWilliams
identity transforms it into the counts.  ``problems.md#sec-macwilliams`` derives the counts, and
says where 64-bit arithmetic is exact.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits, popcount_word, trailing_zeros_word
from qldpc._compilation import always_compiled

from .. import _tuning

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
        chunk_size: The largest number of steps that the walk of a check space takes in one
            compiled call (see ``get_span_weights``).

    Raises:
        ValueError: If a check space has more than ``2**63`` words.
    """

    def __init__(
        self, checks_x: npt.NDArray[np.uint8], checks_z: npt.NDArray[np.uint8], chunk_size: int
    ) -> None:
        for checks in (checks_x, checks_z):
            if len(checks) > _MAX_RANK:
                raise ValueError(
                    f"The MacWilliams method supports check ranks up to {_MAX_RANK}, not"
                    f" {len(checks)}; use 'brouwer_zimmermann' or 'brute_force'"
                )
        self.num_bits = checks_x.shape[1]
        self.rank_x, self.rank_z = len(checks_x), len(checks_z)
        self.weights_x = get_span_weights(checks_x, chunk_size)
        self.self_dual = checks_z is checks_x
        self.weights_z = (
            self.weights_x if self.self_dual else get_span_weights(checks_z, chunk_size)
        )
        # Counts of the words of ker(checks_z) and ker(checks_x) at the weights where one 64-bit
        # transform is exact.
        self.light_counts_x = _count_light_dual_words(self.weights_z, self.rank_z, self.num_bits)
        self.light_counts_z = (
            self.light_counts_x
            if self.self_dual
            else _count_light_dual_words(self.weights_x, self.rank_x, self.num_bits)
        )

    def count_logical_ops(self, weight: int) -> tuple[int, int]:
        """The numbers of nontrivial X-only and Z-only logical operators of the given weight."""
        count_x = self._count_sector(
            self.weights_z, self.rank_z, self.light_counts_x, self.weights_x, weight
        )
        if self.self_dual:
            return count_x, count_x
        return count_x, self._count_sector(
            self.weights_x, self.rank_x, self.light_counts_z, self.weights_z, weight
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


def get_span_weights(generators: npt.NDArray[np.uint8], chunk_size: int) -> npt.NDArray[np.uint64]:
    """The number of words of each weight in the span of independent binary rows, as uint64s.

    A compiled call cannot be interrupted, so the Gray-code walk of the span takes at most
    ``chunk_size`` steps in each call.
    """
    rank, num_bits = generators.shape
    weights = np.zeros(num_bits + 1, dtype=np.uint64)
    weights[0] = 1
    packed = pack_bits(generators)
    word = np.zeros(packed.shape[1], dtype=np.uint64)
    stop = 1 << rank
    for start in range(1, stop, chunk_size):
        chunk_stop = min(start + chunk_size, stop)
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

    The counts stop at the first weight ``w`` at which ``2**rank * C(n, w)`` reaches ``2**63``, and
    64-bit arithmetic is exact at the weights before it (``problems.md#sec-macwilliams``).  They are
    returned as Python integers.
    """
    krawtchouk = _get_light_krawtchouk_rows(num_bits, rank)
    counts = _transform_weights(krawtchouk, span_weights, rank)
    return counts.tolist()


@functools.lru_cache(maxsize=_tuning.KRAWTCHOUK_CACHE_SIZE)
def _get_krawtchouk_row(num_bits: int, weight: int) -> tuple[int, ...]:
    """Binary Krawtchouk polynomial ``K_w(j)`` at every ``j``, for words of length ``n``.

    ``K_w(j) = sum_l (-1)**l C(j, l) C(n - j, w - l)`` satisfies the exact integer recurrence
    ``(n - j) K_w(j + 1) = (n - 2w) K_w(j) - j K_w(j - 1)``, which computes the row.
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


@functools.lru_cache(maxsize=_tuning.KRAWTCHOUK_CACHE_SIZE)
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


@always_compiled
def _add_span_weights(
    generators: npt.NDArray[np.uint64],
    word: npt.NDArray[np.uint64],
    start: np.uint64,
    stop: np.uint64,
    weights: npt.NDArray[np.uint64],
) -> None:
    """Continue a Gray-code walk over the span of packed generators, from step ``start``.

    ``word`` holds the word of the previous step.  Step ``s`` adds to it the generator whose index
    is the number of trailing zeros of ``s``, so steps ``1`` to ``2**rank - 1`` visit every nonzero
    word once.  Each visited word increments the entry of ``weights`` for its weight.
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


@always_compiled
def _transform_weights(
    krawtchouk: npt.NDArray[np.int64], span_weights: npt.NDArray[np.uint64], rank: int
) -> npt.NDArray[np.int64]:
    """Apply Krawtchouk rows to a weight distribution, and divide by ``2**rank``, in int64.

    The caller guarantees that every Krawtchouk value, distribution entry, product, and partial sum
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
