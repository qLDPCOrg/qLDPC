# SPDX-License-Identifier: Apache-2.0

"""Tables that bound the weights of the words that complete a BZ support with two more rows.

A distance search scans the sums of ``w`` rows of a systematic basis ``[I | P]``, whose words weigh
``w`` plus the weight of their tail, the sum of the selected rows of ``P``.  A support whose first
``w - 2`` rows sum to a tail ``p``, and whose next row is at least ``s``, is completed by a pair of
rows ``s <= i < j``.  Partition the tail columns into disjoint blocks, and add, over the blocks, the
least weight of ``p + P_i + P_j`` on the block over all such pairs.  Every completion uses the same
pair on every block, so the sum bounds the weight of the tail of every completion.  If it reaches
``U - w``, for the best weight ``U`` found so far, none of these completions can lower ``U``, and
the search skips them.  Section 3.9 of ``brouwer_zimmermann.md`` derives this bound.

Tables of the least weights on each block, for every value of ``p`` on the block and every ``s``,
make each bound a few lookups.  They cost memory and time to build, so ``select_completion_tables``
estimates from a sample of supports how much of a layer the bound would skip, and uses tables only
if their predicted cost is within the limit of its policy.  The estimate only guides this choice: a
search skips completions only when a computed bound proves that they cannot lower ``U``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qldpc._binary import popcount_word
from qldpc._compilation import always_compiled

from .. import _tuning
from . import information_sets

####################################################################################################
# completion tables


@dataclass(frozen=True)
class CompletionTables:
    """Bounds on the completions by two more rows of the supports of every layer of one basis.

    The tables belong to the packed tails of one systematic basis, whose rows a search sums in their
    given order, and bound nothing for another basis or row order.

    Block ``b`` covers the tail columns from ``b * block_bits`` up to, but not including,
    ``(b + 1) * block_bits``, and the last block may be narrower.  Viewed with shape
    ``(num_rows - 1, offsets[-1])``, entry ``offsets[b] + x`` of row ``s`` of ``tables`` is the
    least weight of ``x + P_i + P_j`` on block ``b`` over the pairs ``s <= i < j``, where ``x`` is
    the value of a tail on the block.
    """

    tables: npt.NDArray[np.uint8]
    offsets: npt.NDArray[np.int64]
    block_bits: int

    @staticmethod
    def build(rows: npt.NDArray[np.uint64], block_bits: int) -> CompletionTables:
        """Build the tables of the packed tails of a basis, on blocks of columns.

        Raises:
            ValueError: If ``block_bits`` is neither ``_tuning.COMPLETION_BLOCK_BITS`` nor
                ``_tuning.COMPLETION_FALLBACK_BLOCK_BITS``, the widths of blocks that searches use.
        """
        if block_bits not in (
            _tuning.COMPLETION_BLOCK_BITS,
            _tuning.COMPLETION_FALLBACK_BLOCK_BITS,
        ):
            raise ValueError(f"Blocks of {block_bits} columns are not supported")
        offsets = _get_block_offsets(_get_num_tail_bits(rows), block_bits)
        tables = np.empty(_get_table_bytes(len(rows), offsets), dtype=np.uint8)
        _fill_tables(rows, block_bits, offsets, tables)
        return CompletionTables(tables, offsets, block_bits)


# Tables that bound nothing, for a scan that skips nothing.
NO_TABLES = CompletionTables(np.zeros(0, dtype=np.uint8), np.zeros(1, dtype=np.int64), 0)


@dataclass(frozen=True)
class CompletionPolicy:
    """The thresholds of a search's choice of completion tables, from its tuning.

    The fields are the ``completion_bound_*`` fields of ``_tuning.SearchTuning``, without that
    prefix.
    """

    minimum_layer_work: int
    max_cost_percent: int
    max_table_bytes: int


def select_completion_tables(
    rows: npt.NDArray[np.uint64],
    weight: int,
    upper_bound: int,
    current: CompletionTables | None,
    policy: CompletionPolicy,
) -> CompletionTables | None:
    """Choose the tables, if any, that a scan of a layer should use to skip completions.

    ``rows`` holds the packed tails of a systematic basis whose layer ``weight`` a search is about
    to scan, with the best weight ``upper_bound``.  For a layer of at least
    ``policy.minimum_layer_work`` supports, the function estimates, from random supports of the
    layer, a conservative lower bound on the share that tables would skip.  It predicts the cost of
    scanning the layer with tables, counting the cost of building them unless the search passes its
    tables of ``rows`` as ``current``.  Tables use blocks of sixteen columns, or of eight if those
    would need more than ``policy.max_table_bytes`` bytes.

    Returns:
        Tables, possibly ``current``, if the predicted cost is at most ``policy.max_cost_percent``
        percent of the cost of a scan without them, or None.
    """
    # Consider only the layers and tables that the policy allows.  Like the work estimates of
    # certificates, the work of a layer saturates at their cap, so that the largest threshold of a
    # tuning considers no layer.
    num_rows = len(rows)
    num_supports = math.comb(num_rows, weight)
    work = min(num_supports, information_sets._WORK_CAP)
    if (
        not 2 <= weight <= num_rows
        or work < policy.minimum_layer_work
        or not policy.max_cost_percent
    ):
        return None
    num_bits = _get_num_tail_bits(rows)
    block_bits = _get_block_bits(num_rows, num_bits, policy.max_table_bytes)
    if not block_bits:
        return None

    # Estimate a conservative share of the supports of the layer that tables would skip.
    share, num_samples = _estimate_skipped_share(rows, weight, upper_bound, block_bits, num_bits)
    share = _get_share_lower_bound(share, num_samples)

    # A search drops its tables when its basis changes, so tables that it passes belong to its
    # rows.  Compiled lookups do not check bounds, so check at least that their layout fits.
    offsets = _get_block_offsets(num_bits, block_bits)
    if current is not None:
        assert current.block_bits == block_bits and np.array_equal(current.offsets, offsets)
        assert len(current.tables) == _get_table_bytes(num_rows, offsets)

    # Predict the cost of the layer with tables, as a share of the cost of scanning it without them,
    # with the cost of building them unless they are reused, and use them if that is within the
    # limit.  Dividing the integer binomials first keeps their ratios small, even for layers whose
    # binomials are too large for floating point.
    kept_ratio = math.comb(num_rows, weight - 1) / num_supports
    prefix_ratio = math.comb(num_rows, weight - 2) / num_supports
    lookups = _tuning.COMPLETION_KEPT_FIRST_ROW_RATIO * (1 - share) * kept_ratio + prefix_ratio
    table_mib = _get_table_bytes(num_rows, offsets) / 2**20
    lookup_cost = (len(offsets) - 1) * (
        _tuning.COMPLETION_LOOKUP_BLOCK_COST
        + _tuning.COMPLETION_LOOKUP_BLOCK_COST_PER_MIB * table_mib
    )
    cost = 1 - share + lookup_cost * lookups
    if current is None:
        cost += _estimate_build_cost(num_rows, offsets, block_bits) / work
    if 100 * cost > policy.max_cost_percent:
        return None
    return CompletionTables.build(rows, block_bits) if current is None else current


@always_compiled(inline=True)
def get_completion_bound(
    tables: npt.NDArray[np.uint8],
    offsets: npt.NDArray[np.int64],
    block_bits: int,
    first: int,
    prefix: npt.NDArray[np.uint64],
    limit: int,
) -> np.int64:
    """Bound the tail weight of ``prefix`` plus any two rows from row ``first`` on.

    ``tables``, ``offsets``, and ``block_bits`` are those of ``CompletionTables``.  The sum stops
    once it reaches ``limit``, so the result is the bound or a lower value of at least ``limit``.
    """
    base = first * offsets[-1]
    # Bytes from the tables add up in int64, which no number of blocks can overflow.
    bound = np.int64(0)
    for block in range(len(offsets) - 1):
        start = block * block_bits
        # Mask each value by the size of its own table, so that ones of the prefix past the last
        # tail column with a one never index another table.
        mask = np.uint64(offsets[block + 1] - offsets[block] - 1)
        value = (prefix[start // 64] >> np.uint64(start % 64)) & mask
        bound += np.int64(tables[base + offsets[block] + np.int64(value)])
        if bound >= limit:
            break
    return bound


def _get_num_tail_bits(rows: npt.NDArray[np.uint64]) -> int:
    """Return one plus the last tail column on which a row has a one, or zero."""
    union = np.bitwise_or.reduce(rows, axis=0)
    for word in range(len(union) - 1, -1, -1):
        if union[word]:
            return 64 * word + int(union[word]).bit_length()
    return 0


def _get_block_offsets(num_bits: int, block_bits: int) -> npt.NDArray[np.int64]:
    """Return the offsets of the tables of blocks of columns in a row of tables.

    The last block may have fewer columns, and its table then has fewer entries.
    """
    sizes = [1 << min(block_bits, num_bits - start) for start in range(0, num_bits, block_bits)]
    return np.concatenate([[0], np.cumsum(sizes, dtype=np.int64)])


def _get_table_bytes(num_rows: int, offsets: npt.NDArray[np.int64]) -> int:
    """Return the size of the tables of a basis, one row of tables for each row but the last."""
    return (num_rows - 1) * int(offsets[-1])


def _get_block_bits(num_rows: int, num_bits: int, max_table_bytes: int) -> int:
    """Choose the widest supported blocks whose tables fit in memory, or 0 if none do.

    Tables of tails without ones would be empty and skip nothing, so none fit.
    """
    for block_bits in (_tuning.COMPLETION_BLOCK_BITS, _tuning.COMPLETION_FALLBACK_BLOCK_BITS):
        num_bytes = _get_table_bytes(num_rows, _get_block_offsets(num_bits, block_bits))
        if 0 < num_bytes <= max_table_bytes:
            return block_bits
    return 0


def _get_share_lower_bound(share: float, num_samples: int) -> float:
    """Return the lower end of the Wilson score interval for a share sampled from supports.

    The interval extends ``_tuning.COMPLETION_CONFIDENCE_Z`` standard errors on each side, and its
    lower end is zero without samples.
    """
    if not num_samples:
        return 0.0
    z2 = _tuning.COMPLETION_CONFIDENCE_Z**2 / num_samples
    center = share + z2 / 2
    spread = _tuning.COMPLETION_CONFIDENCE_Z * math.sqrt(
        share * (1 - share) / num_samples + z2 / (4 * num_samples)
    )
    return max(0.0, (center - spread) / (1 + z2))


def _estimate_build_cost(num_rows: int, offsets: npt.NDArray[np.int64], block_bits: int) -> float:
    """Estimate the cost of building tables, in units of the scan of a support without tables.

    ``offsets`` are those of the tables.  Building writes every table and adds the sums of pairs of
    rows to each block.  Each distance of a block falls at most ``block_bits + 1`` times, and then
    updates ``block_bits`` neighbors.
    """
    num_blocks = len(offsets) - 1
    steps = int(offsets[-1]) * (block_bits + 1) * block_bits + num_blocks * num_rows**2 // 2
    return (
        _tuning.COMPLETION_TABLE_BYTE_COST * _get_table_bytes(num_rows, offsets)
        + _tuning.COMPLETION_BUILD_STEP_COST * steps
    )


def _estimate_skipped_share(
    rows: npt.NDArray[np.uint64], weight: int, upper_bound: int, block_bits: int, num_bits: int
) -> tuple[float, int]:
    """Estimate the share of the supports of a layer that tables would skip.

    A support whose first row of its last pair is ``s`` is skipped exactly if the bound of its first
    ``weight - 2`` rows and ``s`` proves it cannot lower ``upper_bound``: the bound for a smaller
    ``s`` is weaker, and skips it only if this bound would.  ``_tuning.COMPLETION_SAMPLES`` random
    supports, or as many as fit the budget ``_tuning.COMPLETION_SAMPLING_WORK``, estimate this
    share.  The estimate assumes that ``upper_bound`` stays fixed.

    Returns:
        The sampled share and the number of samples, or zeros if fewer than
        ``_tuning.COMPLETION_MIN_SAMPLES`` supports fit the budget.
    """
    # Draw sorted random supports of the layer.
    num_rows = len(rows)
    rng = np.random.default_rng(0)
    supports = np.sort(
        rng.random((_tuning.COMPLETION_SAMPLES, num_rows)).argsort(axis=1)[:, :weight], axis=1
    )

    # Keep the first supports whose direct bounds fit the budget: each tries every pair of rows
    # from the first row of its last pair on, on every block, after summing its prefix.
    num_blocks = -(-num_bits // block_bits)
    remaining = num_rows - supports[:, weight - 2]
    costs = np.cumsum(remaining * (remaining - 1) // 2 * num_blocks + num_rows)
    num_samples = int(np.searchsorted(costs, _tuning.COMPLETION_SAMPLING_WORK, side="right"))
    if num_samples < _tuning.COMPLETION_MIN_SAMPLES:
        return 0.0, 0

    num_skipped = _count_skipped_samples(
        rows, supports[:num_samples], upper_bound - weight, block_bits, num_bits
    )
    return num_skipped / num_samples, num_samples


####################################################################################################
# compiled helpers


@always_compiled
def _fill_tables(
    rows: npt.NDArray[np.uint64],
    block_bits: int,
    offsets: npt.NDArray[np.int64],
    tables: npt.NDArray[np.uint8],
) -> None:
    """Fill the tables of ``CompletionTables`` for the packed tails of a basis.

    For each block, the distances to the sums of the pairs from row ``s`` on are the distances for
    row ``s + 1`` on, lowered by the new pairs of row ``s``.  A breadth-first search from the new
    sums lowers them.  It does not continue past a distance that it does not lower, since the old
    distances already bound every distance beyond it.
    """
    num_rows = rows.shape[0]
    row_bytes = offsets[-1]
    for block in range(len(offsets) - 1):
        start = block * block_bits
        size = offsets[block + 1] - offsets[block]
        # Extract the value of each row's tail on the block.
        bits = 0
        while (1 << bits) < size:
            bits += 1
        mask = np.uint64(size - 1)
        values = np.empty(num_rows, dtype=np.int64)
        for row in range(num_rows):
            values[row] = np.int64((rows[row, start // 64] >> np.uint64(start % 64)) & mask)

        # No pair is farther than ``bits`` from a value, so ``bits + 1`` stands for infinity.
        distances = np.full(size, bits + 1, dtype=np.uint8)
        queue = np.empty(size, dtype=np.int64)
        for first in range(num_rows - 2, -1, -1):
            # The new pair sums of row ``first`` are at distance zero.
            head, tail = 0, 0
            for other in range(first + 1, num_rows):
                point = values[first] ^ values[other]
                if distances[point]:
                    distances[point] = 0
                    queue[tail] = point
                    tail += 1

            # Lower the distances of their neighbors breadth-first, then store the row's table.
            while head < tail:
                point = queue[head]
                head += 1
                distance = distances[point] + 1
                for bit in range(bits):
                    neighbor = point ^ (1 << bit)
                    if distances[neighbor] > distance:
                        distances[neighbor] = distance
                        queue[tail] = neighbor
                        tail += 1
            begin = first * row_bytes + offsets[block]
            tables[begin : begin + size] = distances


@always_compiled
def _count_skipped_samples(
    rows: npt.NDArray[np.uint64],
    supports: npt.NDArray[np.int64],
    limit: int,
    block_bits: int,
    num_bits: int,
) -> int:
    """Count the supports whose bounds, computed directly from the rows, reach ``limit``.

    The bounds use blocks of ``block_bits`` columns, as tables would.
    """
    num_rows, width = rows.shape
    weight = supports.shape[1]
    prefix = np.empty(width, dtype=np.uint64)
    num_skipped = 0
    for support in supports:
        prefix[:] = 0
        for row in support[: weight - 2]:
            prefix ^= rows[row]
        first = support[weight - 2]

        # Add the least weights of the prefix plus a pair on each block of columns.
        bound = np.int64(0)
        for start in range(0, num_bits, block_bits):
            mask = (np.uint64(1) << np.uint64(min(block_bits, num_bits - start))) - np.uint64(1)
            shift = np.uint64(start % 64)
            value = (prefix[start // 64] >> shift) & mask
            least = np.int64(block_bits)
            for row in range(first, num_rows - 1):
                partial = value ^ ((rows[row, start // 64] >> shift) & mask)
                for other in range(row + 1, num_rows):
                    distance = popcount_word(partial ^ ((rows[other, start // 64] >> shift) & mask))
                    least = min(least, distance)
            bound += least
        if bound >= limit:
            num_skipped += 1
    return num_skipped
