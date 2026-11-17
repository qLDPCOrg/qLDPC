# SPDX-License-Identifier: Apache-2.0

"""Tables that bound the weights of the words that complete a BZ support with two more rows.

A BZ scan weighs the sums of sets of ``w`` rows of a systematic basis ``[I | P]``.  For the first
``w - 2`` rows of such a set and a row ``s``, ``CompletionTables`` bound, on the columns of ``P``,
the weight of the words that complete those rows with two rows from ``s`` on.  A scan adds the least
number of their pivot ones to this bound, and skips the words if the sum reaches its limit: the best
weight ``U`` in a distance search, ``t + 1`` in a count of the words of weight ``t``, the least
bound that proves a word heavier than the maximum weight in a coset query, and a limit that falls
as lighter words turn up in a search for a minimum weight.  Sections 3.9 and 3.10 of
``brouwer_zimmermann.md`` derive the bound and the limits.

The threads that share a layer of a distance search (Section 4.5 of that guide), and coset queries
that run at once, each read shared tables.  So a table must be complete before any scan reads it,
and unchanged while scans do, as Section 5 of that guide requires of the data that threads read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qldpc._binary import popcount_word
from qldpc._compilation import always_compiled

from .._tuning import SearchTuning
from . import certificates

####################################################################################################
# completion tables


@dataclass(frozen=True)
class CompletionTables:
    """Bounds on the completions by two more rows of the supports of every layer of one basis.

    The tables belong to the packed tails of one systematic basis, with its rows in the order in
    which a search sums them (Section 3.9).  Block ``b`` covers the tail columns from
    ``b * block_bits`` up to, but not including, ``(b + 1) * block_bits``, and the last block may be
    narrower.  Viewed with shape ``(num_rows - 1, offsets[-1])``, entry ``offsets[b] + x`` of row
    ``s`` of ``tables`` is the least weight of ``x + P_i + P_j`` on block ``b`` over the pairs
    ``s <= i < j``, where ``x`` is the value of a tail on the block.
    """

    tables: npt.NDArray[np.uint8]
    offsets: npt.NDArray[np.int64]
    block_bits: int

    @staticmethod
    def build(
        rows: npt.NDArray[np.uint64], block_bits: int, out: npt.NDArray[np.uint8] | None = None
    ) -> CompletionTables:
        """Build the tables of the packed tails of a basis, on blocks of columns.

        A block lies within one packed word, and its table has ``2**block_bits`` entries for each
        row, so ``block_bits`` divides 64 and is at most 16, as a tuning ensures.  The tables fill
        ``out``, if given, which must have their size (see ``get_table_bytes_within``).
        """
        assert block_bits <= 16 and 64 % block_bits == 0
        offsets = _get_block_offsets(_get_num_tail_bits(rows), block_bits)
        num_bytes = _get_table_bytes(len(rows), offsets)
        tables = np.empty(num_bytes, dtype=np.uint8) if out is None else out
        # The compiled fill does not check bounds.
        assert tables.shape == (num_bytes,)
        _fill_tables(rows, block_bits, offsets, tables)
        return CompletionTables(tables, offsets, block_bits)


# Tables that bound nothing, for a scan that skips nothing.
NO_TABLES = CompletionTables(np.zeros(0, dtype=np.uint8), np.zeros(1, dtype=np.int64), 0)


def get_table_bytes_within(rows: npt.NDArray[np.uint64], block_bits: int, max_bytes: int) -> int:
    """The bytes of the tables of the packed tails of a basis on blocks of columns, if they fit.

    Returns:
        The bytes, or zero if the tables would exceed ``max_bytes`` bytes, or be empty and skip
        nothing (see ``_tables_fit``).
    """
    num_bits = _get_num_tail_bits(rows)
    if not _tables_fit(len(rows), num_bits, block_bits, max_bytes):
        return 0
    return _get_table_bytes(len(rows), _get_block_offsets(num_bits, block_bits))


def select_completion_tables(
    rows: npt.NDArray[np.uint64],
    weight: int,
    upper_bound: int,
    current: CompletionTables | None,
    tuning: SearchTuning,
) -> CompletionTables | None:
    """Choose the tables, if any, with which to scan a layer, by the steps of Section 3.9.

    A search is about to scan layer ``weight`` of a systematic basis whose packed tails are
    ``rows``.  ``upper_bound`` is its best weight, or ``t + 1`` in a count of the words of weight
    ``t``, and ``current``, if given, are its tables of ``rows``, which need no build.

    Returns:
        Tables, possibly ``current``, if the predicted cost of the layer with them is at most
        ``tuning.completion_bound_max_cost_percent`` percent of the cost without them, or None.
    """
    cost, block_bits = predict_completion_cost(
        rows, weight, upper_bound, reuse=current is not None, tuning=tuning
    )

    # A search drops its tables when its basis changes, so tables that it passes belong to its
    # rows.  Compiled lookups do not check bounds, so check at least that their layout fits.
    if current is not None and block_bits:
        offsets = _get_block_offsets(_get_num_tail_bits(rows), block_bits)
        assert current.block_bits == block_bits and np.array_equal(current.offsets, offsets)
        assert len(current.tables) == _get_table_bytes(len(rows), offsets)

    if not is_within_cost_limit(cost, tuning):
        return None
    return CompletionTables.build(rows, block_bits) if current is None else current


def predict_completion_cost(
    rows: npt.NDArray[np.uint64],
    weight: int,
    upper_bound: int,
    *,
    reuse: bool,
    tuning: SearchTuning,
) -> tuple[float, int]:
    """Predict the cost of a scan of a layer with completion tables, and the width of their blocks.

    The arguments are those of ``select_completion_tables``, and ``reuse`` means that the tables
    exist already, so that the prediction leaves out their build.

    Returns:
        The predicted cost of the layer with tables, as a share of its cost without them, and the
        width of the blocks of the tables; or infinity and zero for a layer whose tables the tuning
        does not consider or that do not fit.
    """
    # Consider only the layers and tables that the tuning allows.
    num_rows = len(rows)
    if not considers_tables(num_rows, weight, tuning):
        return math.inf, 0
    num_bits = _get_num_tail_bits(rows)
    block_bits = tuning.completion_bound_block_bits
    if not _tables_fit(num_rows, num_bits, block_bits, tuning.completion_bound_max_table_bytes):
        return math.inf, 0

    # Estimate a conservative share of the supports of the layer that tables would skip.
    share, num_samples = _estimate_skipped_share(
        rows, weight, upper_bound, block_bits, num_bits, tuning
    )
    share = _get_share_lower_bound(share, num_samples, tuning.completion_bound_confidence_z)

    # Predict the cost of the layer with tables, and any build, as a share of the cost without
    # them.  Dividing the integer binomials works even where they exceed floating point.
    offsets = _get_block_offsets(num_bits, block_bits)
    num_supports = math.comb(num_rows, weight)
    kept_ratio = math.comb(num_rows, weight - 1) / num_supports
    prefix_ratio = math.comb(num_rows, weight - 2) / num_supports
    lookups = tuning.completion_bound_kept_first_row_ratio * (1 - share) * kept_ratio + prefix_ratio
    table_mib = _get_table_bytes(num_rows, offsets) / 2**20
    lookup_cost = (len(offsets) - 1) * (
        tuning.completion_bound_lookup_block_cost
        + tuning.completion_bound_lookup_block_cost_per_mib * table_mib
    )
    cost = 1 - share + lookup_cost * lookups
    if not reuse:
        cost += _estimate_build_cost(num_rows, offsets, block_bits, tuning) / _get_layer_work(
            num_rows, weight
        )
    return cost, block_bits


def is_within_cost_limit(cost: float, tuning: SearchTuning) -> bool:
    """Whether a predicted cost, as a share of the cost without tables, is within the limit.

    The limit is ``tuning.completion_bound_max_cost_percent`` percent.
    """
    return 100 * cost <= tuning.completion_bound_max_cost_percent


def considers_tables(num_rows: int, weight: int, tuning: SearchTuning) -> bool:
    """Whether a tuning considers completion tables for layer ``weight`` of a basis of ``num_rows``.

    The layer must have two to ``num_rows`` rows (tables bound completions by two rows) and at
    least ``tuning.completion_bound_minimum_layer_work`` supports.  That count saturates at
    ``WORK_CAP``, which lies below the largest threshold, so that threshold considers no layer.  A
    tuning that allows no cost or no bytes of tables considers none.
    """
    return (
        2 <= weight <= num_rows
        and _get_layer_work(num_rows, weight) >= tuning.completion_bound_minimum_layer_work
        and tuning.completion_bound_max_cost_percent > 0
        and tuning.completion_bound_max_table_bytes > 0
    )


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


def _get_layer_work(num_rows: int, weight: int) -> int:
    """The number of supports of a layer, saturated at the cap of work estimates."""
    return min(math.comb(num_rows, weight), certificates.WORK_CAP)


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


def _tables_fit(num_rows: int, num_bits: int, block_bits: int, max_bytes: int) -> bool:
    """Whether the tables of a basis on blocks of ``block_bits`` columns fit in ``max_bytes``.

    The basis has ``num_rows`` rows, and its tails have ``num_bits`` columns up to the last with a
    one.  Empty tables, of tails without ones or of fewer than two rows, skip nothing, so they never
    fit.
    """
    num_bytes = _get_table_bytes(num_rows, _get_block_offsets(num_bits, block_bits))
    return 0 < num_bytes <= max_bytes


def _get_share_lower_bound(share: float, num_samples: int, z: float) -> float:
    """Return the lower end of the Wilson score interval for a share sampled from supports.

    The interval extends ``z`` standard errors on each side, and its lower end is zero without
    samples.
    """
    if not num_samples:
        return 0.0
    z2 = z**2 / num_samples
    center = share + z2 / 2
    spread = z * math.sqrt(share * (1 - share) / num_samples + z2 / (4 * num_samples))
    return max(0.0, (center - spread) / (1 + z2))


def _estimate_build_cost(
    num_rows: int, offsets: npt.NDArray[np.int64], block_bits: int, tuning: SearchTuning
) -> float:
    """Estimate the cost of building tables, in units of the scan of a support without tables.

    Building writes every table, adds the sum of each pair of rows to each block, and lowers each
    distance at most ``block_bits + 1`` times, each time updating ``block_bits`` neighbors.
    """
    num_blocks = len(offsets) - 1
    steps = int(offsets[-1]) * (block_bits + 1) * block_bits + num_blocks * num_rows**2 // 2
    return (
        tuning.completion_bound_table_byte_cost * _get_table_bytes(num_rows, offsets)
        + tuning.completion_bound_build_step_cost * steps
    )


def _estimate_skipped_share(
    rows: npt.NDArray[np.uint64],
    weight: int,
    upper_bound: int,
    block_bits: int,
    num_bits: int,
    tuning: SearchTuning,
) -> tuple[float, int]:
    """Estimate the share of the supports of a layer that tables would skip, by sampling.

    A support is skipped exactly if the bound of its prefix and the first row of its last pair
    proves that it cannot lower ``upper_bound``, which the estimate takes as fixed (Section 3.9).

    Returns:
        The share in ``tuning.completion_bound_samples`` random supports, or as many as fit the
        budget ``tuning.completion_bound_sampling_work``, and their number; or zeros if fewer than
        ``tuning.completion_bound_min_samples`` fit.
    """
    # Draw sorted random supports of the layer, no more than the budget could afford, since each
    # costs at least one unit for each row.
    num_rows = len(rows)
    num_drawn = min(
        tuning.completion_bound_samples, tuning.completion_bound_sampling_work // num_rows
    )
    if num_drawn < tuning.completion_bound_min_samples:
        return 0.0, 0
    rng = np.random.default_rng(0)
    # The indices of the ``weight`` smallest of uniform random keys form a uniformly random
    # support, which argpartition finds in linear time.
    keys = rng.random((num_drawn, num_rows))
    supports = np.sort(np.argpartition(keys, weight - 1, axis=1)[:, :weight], axis=1)

    # Keep the first supports whose direct bounds fit the budget: each tries every pair of rows
    # from the first row of its last pair on, on every block, after summing its prefix.
    num_blocks = -(-num_bits // block_bits)
    remaining = num_rows - supports[:, weight - 2]
    costs = np.cumsum(remaining * (remaining - 1) // 2 * num_blocks + num_rows)
    num_samples = int(np.searchsorted(costs, tuning.completion_bound_sampling_work, side="right"))
    if num_samples < tuning.completion_bound_min_samples:
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
    """Fill the tables of ``CompletionTables`` for the packed tails of a basis (Section 3.9).

    On each block, the table of row ``s`` holds the distance from each value to the nearest sum of a
    pair of rows from ``s`` on.  A breadth-first search computes it from the table of row ``s + 1``.
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
