# SPDX-License-Identifier: Apache-2.0

"""Compiled scans of the layers of Brouwer-Zimmermann searches of binary spaces and their cosets.

A BZ search scans the words of an affine space ``u + V`` layer by layer: layer ``w`` of a search
basis holds the sums of its offset and ``w`` of its rows (see ``brouwer_zimmermann.md``).  The
distance search scans ``V`` itself, with zero offsets and quotient labels that tell eligible words
from excluded ones, and keeps only the best weight.  The coset search of ``CosetLeaderSearch`` scans
a coset, with the offset of each basis, and keeps the lightest word.

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

from .completion_bounds import get_completion_bound
from .information_sets import advance_support

# Placeholders for the completion tables of scans without them, which never read them.
_NO_TABLE_BYTES = np.zeros(0, dtype=np.uint8)
_NO_TABLE_OFFSETS = np.zeros(1, dtype=np.int64)

####################################################################################################
# scans of distance searches


@always_compiled
def scan_family(
    generators: npt.NDArray[np.uint64],
    labels: npt.NDArray[np.uint64],
    ranks: npt.NDArray[np.int64],
    widths: npt.NDArray[np.int64],
    label_widths: npt.NDArray[np.int64],
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
    best: int,
    cutoff: int,
    batch_size: int,
) -> tuple[int, int, int]:
    """Scan one bounded chunk of a layer of every search basis of a family, in lockstep.

    At each support, the scan weighs the word of the support in each basis in turn.

    Args:
        generators: Packed rows of each search basis after its pivot columns have been deleted.
        labels: Packed quotient labels whose nonzero XOR marks eligible words.
        ranks: Rank of each search basis's block before its pivot columns were deleted.
        widths: Number of packed generator words used by each search basis.
        label_widths: Number of packed label words used by each search basis.
        support: Mutable row indices of the current support.
        prefixes: Mutable prefix XORs for the depths of the support and the search bases.
        depth: Current depth of the support, or ``-1`` once the layer is complete.
        best: Current upper bound; only strictly lighter eligible words replace it.
        cutoff: Weight that stops this chunk once ``best <= cutoff``.
        batch_size: Maximum number of complete supports visited by this chunk.

    Returns:
        Updated ``best``, updated ``depth``, and number of complete supports visited.
    """
    best, depth, visited, _, _ = _scan(
        generators,
        labels,
        ranks,
        widths,
        label_widths,
        support,
        prefixes,
        depth,
        best,
        cutoff,
        batch_size,
        _NO_TABLE_BYTES,
        _NO_TABLE_OFFSETS,
        0,
        # This scan neither weighs X parts nor records words.
        prefixes[0, 0],
        prefixes[0, 0],
        one_set=False,
        pair_steps=False,
        deleted_pivots=True,
        test_labels=True,
        record=False,
        css_weight=False,
        chunked=True,
    )
    return best, depth, visited


@always_compiled
def scan_information_set(
    generators: npt.NDArray[np.uint64],
    labels: npt.NDArray[np.uint64],
    widths: npt.NDArray[np.int64],
    label_widths: npt.NDArray[np.int64],
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
    """Scan one bounded chunk of a layer of one information set, skipping hopeless completions.

    The arguments are those of ``scan_family``, without ``ranks``, for one search basis whose block
    is an information set, so every word weighs its number of rows plus the weight of its packed
    tail, and for a layer of at least two rows.  ``tables``, ``offsets``, and ``block_bits`` are
    those of ``completion_bounds.CompletionTables``, such as ``completion_bounds.NO_TABLES``, which
    skip nothing.  With tables, before the scan tries row ``s`` as the first of the last two rows of
    a support, it bounds the tail weights of the completions of the first ``weight - 2`` rows, the
    prefix, by two rows from row ``s`` on.  If their words cannot weigh less than ``best``, it skips
    them all.

    A chunk ends when the layer is complete, at the cutoff, or once the supports that it weighed and
    the bounds that it looked up reach ``batch_size`` together.

    Returns:
        Updated ``best``; updated ``depth``, which is ``-1`` once the layer is complete; and the
        numbers of supports that the chunk weighed, of bounds that it looked up, and of supports
        that it skipped.
    """
    # This scan neither reads ranks nor weighs X parts nor records words.
    return _scan(
        generators,
        labels,
        widths,
        widths,
        label_widths,
        support,
        prefixes,
        depth,
        best,
        cutoff,
        batch_size,
        tables,
        offsets,
        block_bits,
        prefixes[0, 0],
        prefixes[0, 0],
        one_set=True,
        pair_steps=True,
        deleted_pivots=True,
        test_labels=True,
        record=False,
        css_weight=False,
        chunked=True,
    )


####################################################################################################
# scans of coset searches


@always_compiled
def scan_coset_layer(
    offset: npt.NDArray[np.uint64],
    rows: npt.NDArray[np.uint64],
    widths: npt.NDArray[np.int64],
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    best: int,
    cutoff: int,
    representative: npt.NDArray[np.uint64],
) -> tuple[int, int]:
    """Weigh the offset plus each sum of ``len(support)`` rows of one search basis of a coset.

    The words are packed and full width, with no columns deleted, and weigh their Hamming weight.
    ``rows`` has shape ``(1, num_rows, num_words)``, a stack of one search basis, and ``widths`` is
    ``[num_words]``.  The layer must have ``1 <= len(support) <= num_rows`` rows; the coset driver
    weighs the offset itself, layer zero, without a scan.  ``support`` and ``prefixes``, of at least
    ``len(support)`` rows, are work space.  The scan keeps each word lighter than ``best`` in
    ``representative``.  After each word, it returns if the best weight so far, which starts at
    ``best``, is at most ``cutoff``, and otherwise it returns once the layer is complete.

    Returns:
        The weight of the lightest word found, or ``best`` if none is lighter, and the number of
        words weighed.
    """
    prefixes[0, 0] = offset
    support[0] = 0
    best, _, weighed, _, _ = _scan(
        rows,
        rows,
        widths,
        widths,
        widths,
        support,
        prefixes,
        0,
        best,
        cutoff,
        0,
        _NO_TABLE_BYTES,
        _NO_TABLE_OFFSETS,
        0,
        # This scan neither reads labels nor ranks nor weighs X parts.
        representative,
        representative,
        one_set=True,
        pair_steps=False,
        deleted_pivots=False,
        test_labels=False,
        record=True,
        css_weight=False,
        chunked=False,
    )
    return best, weighed


@always_compiled
def scan_coset_layer_by_css_weight(
    offset: npt.NDArray[np.uint64],
    rows: npt.NDArray[np.uint64],
    widths: npt.NDArray[np.int64],
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
    prefixes[0, 0] = offset
    support[0] = 0
    best, _, weighed, _, _ = _scan(
        rows,
        rows,
        widths,
        widths,
        widths,
        support,
        prefixes,
        0,
        best,
        cutoff,
        0,
        _NO_TABLE_BYTES,
        _NO_TABLE_OFFSETS,
        0,
        # This scan neither reads labels nor ranks.
        x_mask,
        representative,
        one_set=True,
        pair_steps=False,
        deleted_pivots=False,
        test_labels=False,
        record=True,
        css_weight=True,
        chunked=False,
    )
    return best, weighed


####################################################################################################
# the shared scan


@always_compiled(inline=True)
def _scan(
    generators: npt.NDArray[np.uint64],
    labels: npt.NDArray[np.uint64],
    ranks: npt.NDArray[np.int64],
    widths: npt.NDArray[np.int64],
    label_widths: npt.NDArray[np.int64],
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
    one_set: bool,
    pair_steps: bool,
    deleted_pivots: bool,
    test_labels: bool,
    record: bool,
    css_weight: bool,
    chunked: bool,
) -> tuple[int, int, int, int, int]:
    """Scan the supports of a layer of a stack of search bases, starting from ``support``.

    The array arguments are those of ``scan_family``, ``scan_information_set``, and
    ``scan_coset_layer_by_css_weight``, and the flags, which callers pass as constants, choose what
    the scan does:

    - ``one_set``: scan only the first basis, rather than all bases in lockstep.
    - ``pair_steps``: with ``one_set``, step through the first of the last two rows of a support
      separately, and skip completions by the completion tables, if any, as
      ``scan_information_set`` describes.
    - ``deleted_pivots``: the rows have their pivot columns deleted, so a word also weighs the
      number of its rows whose pivots are deleted, which is ``len(support)`` for an information set.
    - ``test_labels``: keep only words whose quotient labels make them eligible.
    - ``record``: copy each lighter word into ``representative``.
    - ``css_weight``: weigh words by their CSS weight, with ``x_mask``, rather than Hamming weight.
    - ``chunked``: return once the weighed supports and the lookups reach ``batch_size`` together.

    Returns:
        Updated ``best`` and ``depth``, and the numbers of supports weighed, of bounds looked up,
        and of supports skipped.
    """
    num_rows = generators.shape[1]
    weight = len(support)
    num_sets = 1 if one_set else len(widths)
    first_width = widths[0]
    # With pair steps, the scan steps through the supports of the first weight - 1 rows, whose last
    # row is the first of the last two rows of a support, and visits the last row of each in its own
    # loop.  The first of the last two rows is never the last row, which no row can follow.
    head = support[: weight - 1]
    head_rows = generators[:, : num_rows - 1]
    pair_depth = weight - 2
    evaluated, lookups, skipped = 0, 0, 0
    while True:
        if not pair_steps:
            depth = advance_support(generators, widths, support, prefixes, depth)
            if depth < 0:
                return best, depth, evaluated, lookups, skipped
        elif depth < weight - 1:
            depth = advance_support(head_rows, widths, head, prefixes, depth)
            if depth < 0:
                return best, depth, evaluated, lookups, skipped
            first = support[pair_depth]

            # Skip every completion from row first on if none can be lighter than the best word.
            # The check for tables runs once for each prefix: as a constant, it would let numba
            # merge the lookup into the loop, and then count references to the tables for each
            # lookup.
            prefix = prefixes[pair_depth, 0]
            if len(tables):
                lookups += 1
                limit = best - weight
                if get_completion_bound(tables, offsets, block_bits, first, prefix, limit) >= limit:
                    remaining = num_rows - first
                    skipped += remaining * (remaining - 1) // 2
                    support[pair_depth] = num_rows
                    if chunked and evaluated + lookups >= batch_size:
                        return best, depth, evaluated, lookups, skipped
                    continue

            # Otherwise add row first to the prefix, and try every later row as the last row.
            for word in range(first_width):
                prefixes[weight - 1, 0, word] = prefix[word] ^ generators[0, first, word]
            depth = weight - 1
            support[depth] = first + 1
            if chunked and evaluated + lookups >= batch_size:
                return best, depth, evaluated, lookups, skipped

        # Weigh the words of the remaining supports that differ from this one only in their last
        # row, returning at the cutoff or at the end of this chunk.  The last row stays in a local
        # variable, and enters the support only where the support is read.
        row = support[depth]
        while row < num_rows:
            for set_index in range(num_sets):
                width = first_width if one_set else widths[set_index]

                # Each selected row whose pivot column was deleted contributes one to the Hamming
                # weight; packed words hold the remaining columns.  The weighing stops once the
                # counts show that the word is no lighter than the best one.
                candidate = np.int64(0)
                if deleted_pivots:
                    if one_set or ranks[set_index] == num_rows:
                        candidate = np.int64(weight)
                    else:
                        for selected in range(depth):
                            if support[selected] < ranks[set_index]:
                                candidate += 1
                        if row < ranks[set_index]:
                            candidate += 1
                z_ones = np.int64(0)
                for word in range(width):
                    value = prefixes[depth, set_index, word] ^ generators[set_index, row, word]
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
                if candidate >= best:
                    continue
                if test_labels:
                    support[depth] = row
                    if not _is_eligible(labels, label_widths, set_index, support):
                        continue

                best = int(candidate)
                if record:
                    for word in range(width):
                        representative[word] = (
                            prefixes[depth, set_index, word] ^ generators[set_index, row, word]
                        )

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
    label_widths: npt.NDArray[np.int64],
    set_index: int,
    support: npt.NDArray[np.int64],
) -> bool:
    """Decide whether the word of a support of a search basis lies outside the excluded subcode.

    A word is eligible when no subcode is excluded, so that the set has no label words, or when the
    XOR of the quotient labels of its rows is nonzero.
    """
    weight = len(support)
    eligible = label_widths[set_index] == 0
    for word in range(label_widths[set_index]):
        combined = labels[set_index, support[0], word]
        for selected in range(1, weight):
            combined ^= labels[set_index, support[selected], word]
        if combined:
            eligible = True
            break
    return eligible
