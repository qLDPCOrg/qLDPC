# SPDX-License-Identifier: Apache-2.0

"""Compiled scans of the layers of Brouwer-Zimmermann searches of binary spaces and their cosets.

A BZ search scans the words of an affine space ``u + V`` layer by layer: layer ``w`` of a search
basis holds the sums of its offset and ``w`` of its rows (see ``brouwer_zimmermann.md``).  Each scan
of this module covers a layer of one search basis.

- ``scan_set`` scans for the distance search, which searches ``V`` itself and keeps only the best
  weight.  Its offsets are zero, and quotient labels tell eligible words from excluded ones.
- ``scan_coset_layer`` and ``scan_coset_layer_by_css_weight`` scan for the coset searches of
  ``CosetLeaderSearch``, which add the offset of each basis and keep the lightest word.
- ``scan_coset_layer_by_tails`` and ``scan_coset_layer_by_tails_with_tables`` scan for the routes
  by tails of a coset search that weighs Hamming weights (see ``coset_routes``).  They weigh the
  tails of words, without the pivot columns of the basis, and keep the support of the lightest
  word.  The search rebuilds the word from it, unless it needs only the minimum weight of a coset.

In layers of two or more rows, the scans that take completion tables skip the completions of a
prefix that the tables show to be too heavy for the search.  Each scan says which words those are.

Every scan of this module runs one body, ``_scan``, which its compiled entry points call with
constant flags.  Numba includes the body in each entry point, which compiles it for those flags, so
no flag costs a branch for each word.  Counting has its own scan, in ``_counting/scan.py``, since it
must visit every word of its weight.
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
    """Scan one bounded chunk of a layer of one set of a family, skipping hopeless completions.

    The rows of the set have the pivot columns of its block deleted.  Only the first ``rank`` rows
    hold pivots, which is all of them if the block is an information set.  So a word weighs the
    number of its rows below ``rank`` plus the weight of its packed tail.

    Tables apply to layers of two or more rows, in which the first ``len(support) - 2`` rows of a
    support are its prefix.  With tables, before the scan tries row ``s`` as the first of the last
    two rows of a support, it bounds the weights of the completions of the prefix by two rows from
    row ``s`` on.  It counts their pivot ones as ``scan_coset_layer_by_tails_with_tables`` does.  If
    their words cannot weigh less than ``best``, it skips them all.

    A chunk ends when the layer is complete, at the cutoff, or once the supports that it weighed and
    the bounds that it looked up reach ``batch_size`` together.

    Args:
        generators: Packed rows of the set, with shape ``(num_rows, num_words)``.
        labels: Packed quotient labels of the rows, whose nonzero XOR marks eligible words, with
            shape ``(num_rows, num_label_words)``.
        rank: The rank of the block of the set.
        width: The number of packed words of the rows that the set uses.
        label_width: The number of packed words of the labels that it uses.
        support: Mutable row indices of the current support, as many as the rows of the layer.
        prefixes: Mutable prefix XORs for the depths of the support, with shape
            ``(len(support) + 1, num_words)``.
        depth: Current depth of the support, or ``-1`` once the layer is complete.
        best: Current upper bound; only strictly lighter eligible words replace it.
        cutoff: Weight that stops this chunk once ``best <= cutoff``.
        batch_size: Maximum number of supports weighed and bounds looked up by this chunk together.
        tables: The tables of a ``completion_bounds.CompletionTables`` of the rows, or of
            ``completion_bounds.NO_TABLES``, which skip nothing.
        offsets: The offsets of the blocks of the same tables.
        block_bits: The width of the blocks of the same tables.

    Returns:
        Updated ``best``; updated ``depth``, which is ``-1`` once the layer is complete; and the
        numbers of supports that the chunk weighed, of bounds that it looked up, and of supports
        that it skipped.
    """
    # This scan neither weighs X parts nor records words or supports.  Its skip bound is the best
    # weight at the start, which the best weight never exceeds, so it skips exactly the completions
    # whose bound shows that they cannot lower the best weight.  The scan of an information set,
    # each of whose rows has a pivot, counts one pivot per row without comparing the row with the
    # set's rank.
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
            pair_steps=True,
            deleted_pivots=True,
            test_labels=True,
            record=False,
            css_weight=False,
            chunked=True,
            ranked=False,
            record_support=False,
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
        pair_steps=True,
        deleted_pivots=True,
        test_labels=True,
        record=False,
        css_weight=False,
        chunked=True,
        ranked=True,
        record_support=False,
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

    The words are packed and full width, with no columns deleted, and weigh their Hamming weight.
    ``rows`` has shape ``(num_rows, width)``.  The layer must have ``1 <= len(support) <= num_rows``
    rows; the coset driver weighs the offset itself, layer zero, without a scan.  ``prefixes``, of
    at least ``len(support)`` rows, and ``support`` are work space.  The scan keeps each word
    lighter than ``best`` in ``representative``.  After each word, it returns if the best weight so
    far, which starts at ``best``, is at most ``cutoff``, and otherwise it returns once the layer is
    complete.

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
        # This scan neither reads labels nor ranks nor weighs X parts nor records supports, nor
        # takes a skip bound.
        representative,
        representative,
        0,
        support,
        pair_steps=False,
        deleted_pivots=False,
        test_labels=False,
        record=True,
        css_weight=False,
        chunked=False,
        ranked=False,
        record_support=False,
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

    The CSS weight of a word is the larger of the Hamming weights of the bits that ``x_mask``
    selects, its X part, and of the other bits, its Z part.  The layer must have
    ``1 <= len(support) <= num_rows`` rows, as for ``scan_coset_layer``.
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
        # This scan neither reads labels nor ranks nor records supports, nor takes a skip bound.
        x_mask,
        representative,
        0,
        support,
        pair_steps=False,
        deleted_pivots=False,
        test_labels=False,
        record=True,
        css_weight=True,
        chunked=False,
        ranked=False,
        record_support=False,
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

    The tail of a word holds its columns other than the pivot columns of the basis, packed.
    ``tails`` holds the tails of the rows, with shape ``(num_rows, width)``, and ``offset_tail``
    that of the offset.  The basis has rank ``rank``, and each of its first ``rank`` rows has a one
    on its own pivot column, where the other rows and the offset are zero.  So the Hamming weight
    of a word is the number of its rows below ``rank`` plus the weight of its tail.  The layer,
    ``support``, ``prefixes``, ``best``, and ``cutoff`` are as for ``scan_coset_layer``.  The scan
    keeps the support of each word lighter than ``best`` in ``best_support``, of at least
    ``len(support)`` entries, rather than the word.

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
        # This scan neither reads labels nor weighs X parts nor records words, nor takes a skip
        # bound.
        prefixes[0],
        prefixes[0],
        0,
        best_support,
        pair_steps=False,
        deleted_pivots=True,
        test_labels=False,
        record=False,
        css_weight=False,
        chunked=False,
        ranked=True,
        record_support=True,
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

    The scan weighs the words of the layer as ``scan_coset_layer_by_tails`` does, which describes
    the arguments that the two share, for a layer of ``2 <= len(support) <= num_rows`` rows of a
    basis of rank ``rank``.  ``tables``, ``offsets``, and ``block_bits`` are those of
    ``completion_bounds.CompletionTables`` of the tails of the rows, or
    ``completion_bounds.NO_TABLES``, which skip nothing.

    With tables, before the scan tries row ``s`` as the first of the last two rows of a support, it
    bounds the weights of the words that complete the first ``len(support) - 2`` rows, the prefix,
    with two rows from row ``s`` on.  Such a word has a pivot one for each row of the prefix below
    ``rank``, and none for the rows of the prefix at or above ``rank``.  Its last two rows have at
    least as many pivot ones as the last two rows of the basis, the pair with the fewest pivot rows,
    since the pivot rows come first: ``max(0, 2 - (num_rows - rank))``.  Its tail weighs at least
    the bound of the tables.  If the sum reaches the lesser of ``skip_bound`` and the best weight so
    far, which starts at ``best``, the scan skips these words.

    A coset query passes as ``skip_bound`` the least bound that proves a word of its coset heavier
    than its maximum weight: one more than the greatest Hamming weight, at most ``cutoff``, that a
    word of the coset can have, which is at most ``cutoff + 1``.  The scan returns once the best
    weight is at most ``cutoff``, so the best weight reaches the skip bound at every lookup, and the
    scan skips at the skip bound.  So it skips no word within that maximum weight, and every
    completion whose bound shows that it cannot lower the best weight, and more.  Skipped words can
    be lighter than the best weight, which is then the weight of the lightest word weighed.

    A search for the minimum weight of a coset passes a skip bound of at least ``best``, so the scan
    skips, as a distance search does, the completions whose bound reaches the best weight so far.
    So it skips no word lighter than the best weight at the time, and finds the lightest word of the
    layer if it is lighter than ``best``, unless it returns at the cutoff first (see
    ``_search_coset_for_min_weight`` in ``coset.py``).

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
        # This scan neither reads labels nor weighs X parts nor records words.
        prefixes[0],
        prefixes[0],
        skip_bound,
        best_support,
        pair_steps=True,
        deleted_pivots=True,
        test_labels=False,
        record=False,
        css_weight=False,
        chunked=False,
        ranked=True,
        record_support=True,
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
    pair_steps: bool,
    deleted_pivots: bool,
    test_labels: bool,
    record: bool,
    css_weight: bool,
    chunked: bool,
    ranked: bool,
    record_support: bool,
) -> tuple[int, int, int, int, int]:
    """Scan the supports of a layer of one search basis, starting from ``support``.

    The arguments are those of ``scan_set``, ``scan_coset_layer_by_css_weight``,
    ``scan_coset_layer_by_tails``, and ``scan_coset_layer_by_tails_with_tables``, and the flags,
    which callers pass as constants, choose what the scan does:

    - ``pair_steps``: with ``deleted_pivots``, step through the first of the last two rows of a
      support separately, and skip the completions whose bound by the completion tables, if any,
      reaches the lesser of ``skip_bound`` and the best weight so far (see ``scan_set``
      and ``scan_coset_layer_by_tails_with_tables``).
    - ``deleted_pivots``: the rows have their pivot columns deleted, so a word also weighs the
      number of its rows whose pivots are deleted, which is ``len(support)`` for an information set.
    - ``test_labels``: keep only words whose quotient labels make them eligible.
    - ``record``: copy each lighter word into ``representative``.
    - ``css_weight``: weigh words by their CSS weight, with ``x_mask``, rather than Hamming weight.
    - ``chunked``: return once the weighed supports and the lookups reach ``batch_size`` together.
    - ``ranked``: with ``deleted_pivots``, lift the assumption that the basis is an information
      set: only its first ``rank`` rows hold pivots.  A word then weighs the number of its rows
      below ``rank``, rather than ``len(support)``, plus the weight of its packed tail.  A bound of
      completions adds the rows of the prefix below ``rank`` and the fewest rows below ``rank`` of
      a completing pair, rather than ``len(support)``.
    - ``record_support``: copy the support of each lighter word into ``best_support``.

    Returns:
        Updated ``best`` and ``depth``, and the numbers of supports weighed, of bounds looked up,
        and of supports skipped.
    """
    num_rows = generators.shape[0]
    weight = len(support)
    # With pair steps, the scan steps through the heads of the supports: their first weight - 1
    # rows, the last of which is the first of the last two rows of the support.  It tries the last
    # row of each support in its own loop.  The first of the last two rows is never the last row of
    # the basis, which no row can follow.  A layer of one row has no such pair, so the scan weighs
    # its words, and then finds the layer complete.  ``_count_chunk_skipping`` in
    # ``_counting/scan.py`` skips completions the same way, but with a fixed limit, since it must
    # visit every word of its target weight.
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
            first = support[pair_depth]

            # Skip every completion from row first on if every one weighs at least the lesser of the
            # skip bound and the best weight.  The scan checks for tables once for each prefix,
            # rather than taking a constant flag.  With a flag, numba would merge the lookup into
            # the loop, and then count references to the tables at each lookup.
            prefix = prefixes[pair_depth]
            if len(tables):
                lookups += 1
                # Count the pivot ones that every completion has: one for each row of the support in
                # an information set.  In a ranked basis, each row of the prefix below the rank has
                # one.  The last two rows have at least as many as the last two rows of the basis,
                # which, since the pivot rows come first, is the pair with the fewest pivot rows:
                # max(0, 2 - (num_rows - rank)).
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

            # Otherwise add row first to the prefix, and try every later row as the last row.
            for word in range(width):
                prefixes[weight - 1, word] = prefix[word] ^ generators[first, word]
            depth = weight - 1
            support[depth] = first + 1
            if chunked and evaluated + lookups >= batch_size:
                return best, depth, evaluated, lookups, skipped

        # Weigh the words of the remaining supports that differ from this one only in their last
        # row, returning at the cutoff or at the end of this chunk.  The last row stays in a local
        # variable, and enters the support only where the support is read.
        row = support[depth]
        # In a ranked basis, count the pivot rows of the support but the last once, before trying
        # each last row.
        head_pivots = np.int64(0)
        if ranked:
            for selected in range(depth):
                if support[selected] < rank:
                    head_pivots += 1
        while row < num_rows:
            # Each selected row whose pivot column was deleted contributes one to the Hamming
            # weight; packed words hold the remaining columns.  The weighing stops once the counts
            # show that the word is no lighter than the best one.
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

    A word is eligible when no subcode is excluded, so that the basis has no label words, or when
    the XOR of the quotient labels of its rows is nonzero.
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
