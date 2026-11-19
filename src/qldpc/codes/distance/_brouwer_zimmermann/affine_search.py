# SPDX-License-Identifier: Apache-2.0

"""Compiled scans of the layers of Brouwer-Zimmermann searches of binary spaces and their cosets.

Each scan covers one layer ``w`` of one search basis: the sums of its offset and ``w`` of its rows.
``scan_set`` and ``scan_set_items``, which scans a part of a layer on one of several threads, serve
the distance search, and the other scans the coset searches of ``CosetLeaderSearch``.  Every scan
runs one body, ``_scan``, which Numba compiles separately for each set of constant flags, so that no
flag costs a branch for each word.  Sections cited in this module are those of
``brouwer_zimmermann.md`` and ``brouwer_zimmermann_code.md``, whose Section 4.1 lists the scans.

Threads run ``_scan`` at once, through ``scan_set_items`` and the batch loops of coset searches in
``coset.py``.  So a change to it must keep the requirements on threads in Section 5, notably "Keep
the scans of a layer splittable across threads" and, for the coset scans, "Keep coset queries
reproducible".
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qldpc._binary import popcount_word
from qldpc._compilation import always_compiled

from .._information_sets.completion_bounds import get_completion_bound
from .._information_sets.support_stepping import advance_support

# Placeholders for the completion tables of scans without them, which never read them.
_NO_TABLE_BYTES = np.zeros(0, dtype=np.uint8)
_NO_TABLE_OFFSETS = np.zeros(1, dtype=np.int64)

####################################################################################################
# scans of distance searches


@always_compiled
def scan_set(
    generators: npt.NDArray[np.uint64],
    labels: npt.NDArray[np.uint64],
    rank: int,
    width: int,
    label_width: int,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
    best: int,
    cutoff: int,
    batch_size: int,
    tables: npt.NDArray[np.uint8],
    offsets: npt.NDArray[np.int64],
    block_bits: int,
) -> tuple[int, int, int, int, int]:
    """Scan one chunk of a layer of one set, skipping completions that tables prove too heavy.

    The rows have the pivot columns of the set deleted, so a word weighs the number of its rows
    below ``rank`` plus the weight of its packed tail (Section 3.5).  With tables, in a layer of two
    or more rows, before the scan tries each row ``s`` as the first of the last two rows of a
    support, it skips every completion of the prefix, the first ``len(support) - 2`` rows, by two
    rows from ``s`` on when their bound reaches the best weight (Section 3.9).  A chunk ends when
    the layer is complete, at the cutoff, or at the end of the batch.

    Args:
        generators: Packed rows of the set, with shape ``(num_rows, num_words)``.
        labels: Packed quotient labels of the rows, whose nonzero XOR marks eligible words, with
            shape ``(num_rows, num_label_words)``.
        rank: The rank of the block of the set.
        width: The number of packed words of the rows that the set uses.
        label_width: The number of packed words of the labels that it uses.
        support: The row indices of the current support, updated in place.
        prefixes: The XORs of the leading rows of the support, with shape
            ``(len(support) + 1, num_words)``, updated in place.
        depth: The depth of the support at which to resume, or ``-1`` once the layer is complete.
        best: The best weight; only strictly lighter eligible words replace it.
        cutoff: The chunk returns once ``best <= cutoff``.
        batch_size: The most supports weighed plus bounds looked up in one chunk.
        tables: The tables of a ``completion_bounds.CompletionTables`` of the rows, or of
            ``completion_bounds.NO_TABLES``, which skip nothing.
        offsets: The block offsets of the same tables.
        block_bits: The block width of the same tables.

    Returns:
        The updated ``best`` and ``depth``, and the numbers of supports weighed, of bounds looked
        up, and of supports skipped.
    """
    # The scan weighs no X parts, records nothing, and does not end after a work item, so
    # placeholders fill those arguments.  The skip bound is the best weight at the start, which the
    # best weight never exceeds, so the scan skips only completions that cannot lower the best
    # weight.  Every row of an information set has a pivot, so its scan needs no rank.
    if rank == generators.shape[0]:
        return _scan(
            generators,
            labels,
            rank,
            width,
            label_width,
            support,
            prefixes,
            depth,
            best,
            cutoff,
            batch_size,
            tables,
            offsets,
            block_bits,
            prefixes[0],
            prefixes[0],
            best,
            support,
            support,
            pair_steps=True,
            deleted_pivots=True,
            test_labels=True,
            record=False,
            css_weight=False,
            chunked=True,
            ranked=False,
            record_support=False,
            ends_after_item=False,
        )
    return _scan(
        generators,
        labels,
        rank,
        width,
        label_width,
        support,
        prefixes,
        depth,
        best,
        cutoff,
        batch_size,
        tables,
        offsets,
        block_bits,
        prefixes[0],
        prefixes[0],
        best,
        support,
        support,
        pair_steps=True,
        deleted_pivots=True,
        test_labels=True,
        record=False,
        css_weight=False,
        chunked=True,
        ranked=True,
        record_support=False,
        ends_after_item=False,
    )


def start_set_items(
    generators: npt.NDArray[np.uint64],
    first_item: npt.NDArray[np.int64],
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
) -> int:
    """Set ``support`` and ``prefixes`` to the first support of a work item, for ``scan_set_items``.

    ``first_item`` holds the ``k`` fixed rows of a work item of the layer ``len(support)``, with
    ``1 <= k <= len(support) - 2``, and ``prefixes[0]`` holds the offset of the words.  The support
    starts with the fixed rows and then the next row, and ``prefixes[1 : k + 1]`` get the XORs of
    the offset and its leading rows.

    Returns:
        The depth at which ``scan_set_items`` starts.
    """
    length = len(first_item)
    support[:length] = first_item
    support[length] = first_item[-1] + 1
    for depth in range(length):
        np.bitwise_xor(prefixes[depth], generators[first_item[depth]], out=prefixes[depth + 1])
    return length


@always_compiled(nogil=True)
def scan_set_items(
    generators: npt.NDArray[np.uint64],
    labels: npt.NDArray[np.uint64],
    rank: int,
    width: int,
    label_width: int,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
    best: int,
    cutoff: int,
    batch_size: int,
    tables: npt.NDArray[np.uint8],
    offsets: npt.NDArray[np.int64],
    block_bits: int,
    last_item: npt.NDArray[np.int64],
) -> tuple[int, int, int, int, int]:
    """Scan one chunk of a range of work items of a layer of one set, as ``scan_set`` scans it.

    A work item is the set of supports of the layer whose first ``k`` rows are fixed, and the items
    of a range are consecutive in lexicographic order of their fixed rows.  With
    ``1 <= k <= len(support) - 2``, every completion that tables skip lies in the item of its
    prefix, so the scans of the items of a layer, when they run in order, carrying the best weight,
    weigh, look up, and skip what the scan of the layer does.  Start the range with
    ``start_set_items`` at its first item.  The scan returns depth ``-1`` at the end of the range,
    once the first ``k`` rows of a support pass ``last_item``, as at the end of a layer.

    The scan runs without Python's global interpreter lock, so that threads can scan the ranges of
    a layer at once, each with its own ``support`` and ``prefixes``.  The other arguments and the
    results are those of ``scan_set``.
    """
    # Placeholders fill the arguments that the scan does not read, as in scan_set.
    if rank == generators.shape[0]:
        return _scan(
            generators,
            labels,
            rank,
            width,
            label_width,
            support,
            prefixes,
            depth,
            best,
            cutoff,
            batch_size,
            tables,
            offsets,
            block_bits,
            prefixes[0],
            prefixes[0],
            best,
            support,
            last_item,
            pair_steps=True,
            deleted_pivots=True,
            test_labels=True,
            record=False,
            css_weight=False,
            chunked=True,
            ranked=False,
            record_support=False,
            ends_after_item=True,
        )
    return _scan(
        generators,
        labels,
        rank,
        width,
        label_width,
        support,
        prefixes,
        depth,
        best,
        cutoff,
        batch_size,
        tables,
        offsets,
        block_bits,
        prefixes[0],
        prefixes[0],
        best,
        support,
        last_item,
        pair_steps=True,
        deleted_pivots=True,
        test_labels=True,
        record=False,
        css_weight=False,
        chunked=True,
        ranked=True,
        record_support=False,
        ends_after_item=True,
    )


####################################################################################################
# scans of coset searches


@always_compiled
def scan_coset_layer(
    offset: npt.NDArray[np.uint64],
    rows: npt.NDArray[np.uint64],
    width: int,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    best: int,
    cutoff: int,
    representative: npt.NDArray[np.uint64],
) -> tuple[int, int]:
    """Weigh the offset plus each sum of ``len(support)`` rows of one search basis of a coset.

    The words are packed with every column, and weigh their Hamming weight.  ``rows`` has shape
    ``(num_rows, width)``, and ``1 <= len(support) <= num_rows``.  ``prefixes``, of at least
    ``len(support)`` rows, and ``support`` are work space.  The scan copies each word lighter than
    the best weight so far, which starts at ``best``, into ``representative``, and returns once that
    weight is at most ``cutoff`` or the layer is complete.

    Returns:
        The weight of the lightest word found, or ``best`` if none is lighter, and the number of
        words weighed.
    """
    prefixes[0] = offset
    support[0] = 0
    best, _, weighed, _, _ = _scan(
        rows,
        rows,
        width,
        width,
        width,
        support,
        prefixes,
        0,
        best,
        cutoff,
        0,
        _NO_TABLE_BYTES,
        _NO_TABLE_OFFSETS,
        0,
        # The scan reads no labels, rank, X mask, skip bound, or best support, so placeholders fill
        # them.
        representative,
        representative,
        0,
        support,
        support,
        pair_steps=False,
        deleted_pivots=False,
        test_labels=False,
        record=True,
        css_weight=False,
        chunked=False,
        ranked=False,
        record_support=False,
        ends_after_item=False,
    )
    return best, weighed


@always_compiled
def scan_coset_layer_by_css_weight(
    offset: npt.NDArray[np.uint64],
    rows: npt.NDArray[np.uint64],
    width: int,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    best: int,
    cutoff: int,
    x_mask: npt.NDArray[np.uint64],
    representative: npt.NDArray[np.uint64],
) -> tuple[int, int]:
    """Scan a layer as ``scan_coset_layer`` does, but weigh each word by its CSS weight.

    The CSS weight of a word is the larger of the Hamming weights of its X part, the bits that
    ``x_mask`` selects, and of its Z part, the other bits.
    """
    prefixes[0] = offset
    support[0] = 0
    best, _, weighed, _, _ = _scan(
        rows,
        rows,
        width,
        width,
        width,
        support,
        prefixes,
        0,
        best,
        cutoff,
        0,
        _NO_TABLE_BYTES,
        _NO_TABLE_OFFSETS,
        0,
        # The scan reads no labels, rank, skip bound, or best support, so placeholders fill them.
        x_mask,
        representative,
        0,
        support,
        support,
        pair_steps=False,
        deleted_pivots=False,
        test_labels=False,
        record=True,
        css_weight=True,
        chunked=False,
        ranked=False,
        record_support=False,
        ends_after_item=False,
    )
    return best, weighed


@always_compiled
def scan_coset_layer_by_tails(
    offset_tail: npt.NDArray[np.uint64],
    tails: npt.NDArray[np.uint64],
    rank: int,
    width: int,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    best: int,
    cutoff: int,
    best_support: npt.NDArray[np.int64],
) -> tuple[int, int]:
    """Scan a layer as ``scan_coset_layer`` does, but weigh each word by its pivot rows and tail.

    The tail of a word is its packed columns other than the pivots of the basis.  ``tails`` holds
    the tails of the rows, with shape ``(num_rows, width)``, and ``offset_tail`` that of the offset.
    A word weighs the number of its rows below ``rank`` plus the weight of its tail (Section 3.5).
    The scan copies the support of each lighter word, rather than the word, into ``best_support``,
    of at least ``len(support)`` entries.  The routes by tails of coset searches weigh words this
    way (Section 4.4).

    Returns:
        The weight of the lightest word found, or ``best`` if none is lighter, and the number of
        words weighed.
    """
    prefixes[0] = offset_tail
    support[0] = 0
    best, _, weighed, _, _ = _scan(
        tails,
        tails,
        rank,
        width,
        width,
        support,
        prefixes,
        0,
        best,
        cutoff,
        0,
        _NO_TABLE_BYTES,
        _NO_TABLE_OFFSETS,
        0,
        # The scan reads no labels or X mask, records no word, and takes no skip bound, so
        # placeholders fill them.
        prefixes[0],
        prefixes[0],
        0,
        best_support,
        support,
        pair_steps=False,
        deleted_pivots=True,
        test_labels=False,
        record=False,
        css_weight=False,
        chunked=False,
        ranked=True,
        record_support=True,
        ends_after_item=False,
    )
    return best, weighed


@always_compiled
def scan_coset_layer_by_tails_with_tables(
    offset_tail: npt.NDArray[np.uint64],
    tails: npt.NDArray[np.uint64],
    rank: int,
    width: int,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    best: int,
    cutoff: int,
    skip_bound: int,
    tables: npt.NDArray[np.uint8],
    offsets: npt.NDArray[np.int64],
    block_bits: int,
    best_support: npt.NDArray[np.int64],
) -> tuple[int, int, int]:
    """Scan a layer of one search basis of a coset by tails, skipping completions by tables.

    The scan weighs words as ``scan_coset_layer_by_tails`` does, in a layer of
    ``2 <= len(support) <= num_rows`` rows, and ``scan_coset_layer_by_tails`` describes the shared
    arguments.  ``tables``, ``offsets``, and ``block_bits`` are the fields of a
    ``completion_bounds.CompletionTables`` of the tails of the rows, or of
    ``completion_bounds.NO_TABLES``, which skip nothing.  Before the scan tries row ``s`` as the
    first of the last two rows of a support, it skips every completion of the prefix of the support,
    its first ``len(support) - 2`` rows, by two rows from ``s`` on if their bound reaches the lesser
    of ``skip_bound`` and the best weight so far (Section 3.9, "Coset searches").  A coset query
    passes as ``skip_bound`` the least bound that proves a word of its coset heavier than its
    maximum weight; this bound is at most ``cutoff + 1``.  A search for a minimum weight passes at
    least ``best`` as ``skip_bound`` (Section 3.10).

    Returns:
        The weight of the lightest word found, or ``best`` if none is lighter, and the numbers of
        words weighed and of supports skipped.
    """
    prefixes[0] = offset_tail
    support[0] = 0
    best, _, weighed, _, skipped = _scan(
        tails,
        tails,
        rank,
        width,
        width,
        support,
        prefixes,
        0,
        best,
        cutoff,
        0,
        tables,
        offsets,
        block_bits,
        # The scan reads no labels or X mask, and records no word, so placeholders fill them.
        prefixes[0],
        prefixes[0],
        skip_bound,
        best_support,
        support,
        pair_steps=True,
        deleted_pivots=True,
        test_labels=False,
        record=False,
        css_weight=False,
        chunked=False,
        ranked=True,
        record_support=True,
        ends_after_item=False,
    )
    return best, weighed, skipped


####################################################################################################
# the shared scan


@always_compiled(inline=True)
def _scan(
    generators: npt.NDArray[np.uint64],
    labels: npt.NDArray[np.uint64],
    rank: int,
    width: int,
    label_width: int,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
    best: int,
    cutoff: int,
    batch_size: int,
    tables: npt.NDArray[np.uint8],
    offsets: npt.NDArray[np.int64],
    block_bits: int,
    x_mask: npt.NDArray[np.uint64],
    representative: npt.NDArray[np.uint64],
    skip_bound: int,
    best_support: npt.NDArray[np.int64],
    last_item: npt.NDArray[np.int64],
    pair_steps: bool,
    deleted_pivots: bool,
    test_labels: bool,
    record: bool,
    css_weight: bool,
    chunked: bool,
    ranked: bool,
    record_support: bool,
    ends_after_item: bool,
) -> tuple[int, int, int, int, int]:
    """Scan the supports of a layer of one search basis, starting from ``support``.

    The arguments are those of the entry points of this module.  The flags, which callers pass as
    constants, choose what the scan does:

    - ``pair_steps``: with ``deleted_pivots``, step through the first of the last two rows of each
      support separately, and skip completions by the tables, if any (see
      ``scan_coset_layer_by_tails_with_tables``).
    - ``deleted_pivots``: add to the weight of a word a pivot one for each of its rows
      (Section 3.5).
    - ``test_labels``: keep only words whose quotient labels make them eligible.
    - ``record``: copy each lighter word into ``representative``.
    - ``css_weight``: weigh words by their CSS weight, with ``x_mask``, rather than Hamming weight.
    - ``chunked``: return once the weighed supports and the lookups reach ``batch_size`` together.
    - ``ranked``: with ``deleted_pivots``, count pivot ones only for rows below ``rank``, in words
      and in bounds of completions (Section 3.9, "Coset searches").
    - ``record_support``: copy the support of each lighter word into ``best_support``.
    - ``ends_after_item``: with ``pair_steps``, end the layer once the first ``len(last_item)``
      rows of the head of a support pass ``last_item`` in lexicographic order (see
      ``scan_set_items``).

    Returns:
        Updated ``best`` and ``depth``, and the numbers of supports weighed, of bounds looked up,
        and of supports skipped.
    """
    num_rows = generators.shape[0]
    weight = len(support)
    # With pair steps, the scan steps through heads, the first ``weight - 1`` rows of each support,
    # and tries each last row in the inner loop.  No head ends at the last row of the basis, which
    # no row can follow.  A layer of one row has empty heads, so the scan weighs its words at once.
    # ``_count_chunk_skipping`` in ``_counting/scan.py`` skips completions the same way.
    head = support[: weight - 1]
    head_rows = generators[: num_rows - 1]
    pair_depth = weight - 2
    evaluated, lookups, skipped = 0, 0, 0
    while True:
        if not pair_steps:
            depth = advance_support(generators, width, support, prefixes, depth)
            if depth < 0:
                return best, depth, evaluated, lookups, skipped
        elif depth < weight - 1:
            depth = advance_support(head_rows, width, head, prefixes, depth)
            if depth < 0:
                return best, depth, evaluated, lookups, skipped
            # End the range once the fixed rows of the support pass ``last_item``, in
            # lexicographic order.
            if ends_after_item:
                for selected in range(len(last_item)):
                    if support[selected] != last_item[selected]:
                        if support[selected] > last_item[selected]:
                            return best, -1, evaluated, lookups, skipped
                        break
            first = support[pair_depth]

            # Skip the completions from row ``first`` on if their bound reaches the lesser of the
            # skip bound and the best weight.  The scan tests for tables at each prefix rather than
            # taking a constant flag, with which Numba would merge the lookup into the loop and
            # count references to the tables at each lookup.
            prefix = prefixes[pair_depth]
            if len(tables):
                lookups += 1
                # Count the pivot ones of every completion (Section 3.9, "Coset searches").
                pivot_ones = weight
                if ranked:
                    pivot_ones = max(0, 2 - (num_rows - rank))
                    for selected in range(pair_depth):
                        if support[selected] < rank:
                            pivot_ones += 1
                limit = min(skip_bound, best) - pivot_ones
                if get_completion_bound(tables, offsets, block_bits, first, prefix, limit) >= limit:
                    remaining = num_rows - first
                    skipped += remaining * (remaining - 1) // 2
                    support[pair_depth] = num_rows
                    if chunked and evaluated + lookups >= batch_size:
                        return best, depth, evaluated, lookups, skipped
                    continue

            # Otherwise add row ``first`` to the prefix, and try every later row as the last row.
            for word in range(width):
                prefixes[weight - 1, word] = prefix[word] ^ generators[first, word]
            depth = weight - 1
            support[depth] = first + 1
            if chunked and evaluated + lookups >= batch_size:
                return best, depth, evaluated, lookups, skipped

        # Weigh the words of the supports that differ from this one only in their last row,
        # returning at the cutoff or at the end of the chunk.  The last row enters the support only
        # where the support is read.
        row = support[depth]
        # Count the pivot rows of the head once, before trying the last rows.
        head_pivots = np.int64(0)
        if ranked:
            for selected in range(depth):
                if support[selected] < rank:
                    head_pivots += 1
        while row < num_rows:
            # Add the pivot ones of the rows, then the packed tail, until the word is no lighter
            # than the best.
            candidate = np.int64(0)
            if deleted_pivots:
                if ranked:
                    candidate = head_pivots + (1 if row < rank else 0)
                else:
                    candidate = np.int64(weight)
            z_ones = np.int64(0)
            for word in range(width):
                value = prefixes[depth, word] ^ generators[row, word]
                if css_weight:
                    candidate += popcount_word(value & x_mask[word])
                    z_ones += popcount_word(value & ~x_mask[word])
                    if candidate >= best or z_ones >= best:
                        break
                else:
                    candidate += popcount_word(value)
                    if candidate >= best:
                        break
            if css_weight:
                candidate = max(candidate, z_ones)

            # Keep a lighter word if its labels make it eligible.
            lighter = bool(candidate < best)
            if lighter and test_labels:
                support[depth] = row
                lighter = _is_eligible(labels, label_width, support)
            if lighter:
                best = int(candidate)
                if record:
                    for word in range(width):
                        representative[word] = prefixes[depth, word] ^ generators[row, word]
                if record_support:
                    for selected in range(depth):
                        best_support[selected] = support[selected]
                    best_support[depth] = row

            row += 1
            evaluated += 1
            if best <= cutoff or (chunked and evaluated + lookups >= batch_size):
                support[depth] = row
                return best, depth, evaluated, lookups, skipped
        if pair_steps:
            depth = pair_depth
            support[depth] += 1
        else:
            support[depth] = num_rows


@always_compiled(inline=True)
def _is_eligible(
    labels: npt.NDArray[np.uint64],
    label_width: int,
    support: npt.NDArray[np.int64],
) -> bool:
    """Decide whether the word of a support of a search basis lies outside the excluded subcode.

    It does if the basis has no label words, since then nothing is excluded, or if the XOR of the
    quotient labels of the rows of the support is nonzero (Section 3.4).
    """
    weight = len(support)
    eligible = label_width == 0
    for word in range(label_width):
        combined = labels[support[0], word]
        for selected in range(1, weight):
            combined ^= labels[support[selected], word]
        if combined:
            eligible = True
            break
    return eligible
