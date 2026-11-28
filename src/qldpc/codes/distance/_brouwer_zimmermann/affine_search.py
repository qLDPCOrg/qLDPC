# SPDX-License-Identifier: Apache-2.0

"""Compiled scans of the layers of Brouwer-Zimmermann searches of binary spaces and their cosets.

Each scan covers one layer ``w`` of one search basis: the sums of the basis's offset and ``w`` of
its rows.  Every scan runs one body, ``_scan``, which Numba compiles separately for each set of
constant flags, so that no flag costs a branch for each word.

- ``scan_set`` scans one set of a distance search, with or without completion tables.
- ``scan_set_items`` scans a range of work items of a layer of such a set, as ``scan_set`` does, on
  one of several threads, from the first support that ``start_set_items`` sets
  (``brouwer_zimmermann_code.md#sec-cross-file-behavior``).
- ``scan_coset_layer`` and ``scan_coset_layer_by_css_weight`` scan one layer of one set in a coset
  search, and keep its lightest word.
- ``scan_coset_layer_by_tails`` does so by the tails of the words, and keeps the support of the
  lightest word.
- ``scan_coset_layer_by_tails_with_tables`` also skips, in a layer of two or more rows, the
  completions whose bound reaches the lesser of a skip bound and the best weight so far.  A query
  skips at ``h(ω)`` (``brouwer_zimmermann.md#sec-completion-cosets``), and a search for minimum
  weights at its incumbent (``brouwer_zimmermann.md#sec-coset-min-weights``).

Threads run ``_scan`` at once, through ``scan_set_items`` and the batch loops of coset searches in
``coset.py``.  So a change to it must keep the requirements on threads of
``brouwer_zimmermann_code.md#sec-requirements``, notably "Keep the scans of a layer splittable
across threads" and, for the coset scans, "Keep coset queries reproducible".
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._binary import popcount_word
from qldpc._compilation import always_compiled

from .._information_sets.completion_bounds import (
    NO_TABLE_VIEW,
    CompletionTableView,
    get_completion_bound,
)
from .._information_sets.support_stepping import PackedBasis, advance_support

####################################################################################################
# scans of distance searches


@always_compiled
def scan_set(
    basis: PackedBasis,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
    best: int,
    cutoff: int,
    batch_size: int,
    tables: CompletionTableView,
) -> tuple[int, int, int, int, int]:
    """Scan one chunk of a layer of one set, skipping completions that tables prove too heavy.

    The rows of ``basis`` have the pivot columns of the set deleted, so a word weighs the number of
    its rows below ``basis.rank`` plus the weight of its packed tail
    (``brouwer_zimmermann.md#sec-packed-enumeration``).  With tables, in a layer of two or more
    rows, the scan skips the completions that tables prove no lighter than the best weight, as
    ``pair_steps`` of ``_scan`` describes (``brouwer_zimmermann.md#sec-completion-tables``).  A
    chunk ends when the layer is complete, at the cutoff, or at the end of the batch.

    Args:
        basis: The set, whose quotient labels mark its eligible words.
        support: The row indices of the current support, updated in place.
        prefixes: The XORs of the leading rows of the support, with shape
            ``(len(support) + 1, num_words)``, updated in place.
        depth: The depth of the support at which to resume, or ``-1`` once the layer is complete.
        best: The best weight; only strictly lighter eligible words replace it.
        cutoff: The chunk returns once ``best <= cutoff``.
        batch_size: The most supports weighed plus bounds looked up in one chunk.
        tables: The completion tables of the rows, or ``completion_bounds.NO_TABLE_VIEW``, which
            skips nothing.

    Returns:
        The updated ``best`` and ``depth``, and the numbers of supports weighed, of bounds looked
        up, and of supports skipped.
    """
    # The scan does not end after a work item, so the support fills ``last_item``.
    return _scan_set_chunk(
        basis,
        support,
        prefixes,
        depth,
        best,
        cutoff,
        batch_size,
        tables,
        support,
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
    basis: PackedBasis,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
    best: int,
    cutoff: int,
    batch_size: int,
    tables: CompletionTableView,
    last_item: npt.NDArray[np.int64],
) -> tuple[int, int, int, int, int]:
    """Scan one chunk of a range of work items of a layer of one set, as ``scan_set`` scans it.

    A work item is the set of supports of the layer whose first ``k`` rows are fixed, with
    ``1 <= k <= len(support) - 2``.  The items of a range are consecutive in lexicographic order of
    their fixed rows, and ``last_item`` holds the ``k`` fixed rows of the last item of the range.

    Suppose the items of a layer are scanned in order, with each scan starting from the best weight
    left by its predecessor.  Together, these scans weigh the same supports, look up the same
    bounds, and skip the same completions as one scan of the layer.
    ``brouwer_zimmermann_code.md#sec-cross-file-behavior`` ("Threads") says why, and ``pair_steps``
    of ``_scan`` defines completions.

    Start the range with ``start_set_items`` at its first item.  The scan returns depth ``-1`` at
    the end of the range, once the first ``k`` rows of a support pass ``last_item``, as at the end
    of a layer.

    The scan runs without Python's global interpreter lock, so that threads can scan the ranges of
    a layer at once, each with its own ``support`` and ``prefixes``.  The other arguments and the
    results are those of ``scan_set``.  The scan keeps the rules on threads of
    ``brouwer_zimmermann_code.md#sec-requirements``.
    """
    return _scan_set_chunk(
        basis,
        support,
        prefixes,
        depth,
        best,
        cutoff,
        batch_size,
        tables,
        last_item,
        ends_after_item=True,
    )


@always_compiled
def _scan_set_chunk(
    basis: PackedBasis,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
    best: int,
    cutoff: int,
    batch_size: int,
    tables: CompletionTableView,
    last_item: npt.NDArray[np.int64],
    ends_after_item: bool,
) -> tuple[int, int, int, int, int]:
    """Scan one chunk for ``scan_set`` or ``scan_set_items``, whose arguments it takes.

    ``ends_after_item``, which both pass as a constant, is True for ``scan_set_items``, and
    ``_scan`` reads ``last_item`` only then.  This function runs ``_scan`` with one of two sets of
    flags: one for a full set, whose every row has a pivot, and one for any other set.
    """
    # The scan weighs no X parts and records nothing, so placeholders fill those arguments.  The
    # skip bound is the best weight at the start, which the best weight never exceeds, so the scan
    # skips only completions that cannot lower the best weight.  Every row of a full set has a
    # pivot, so its scan needs no rank.
    workspace = ScanWorkspace(support, prefixes, prefixes[0], support)
    if basis.rank == basis.rows.shape[0]:
        return _scan(
            basis,
            workspace,
            tables,
            prefixes[0],
            last_item,
            depth,
            best,
            cutoff,
            batch_size,
            best,
            pair_steps=True,
            deleted_pivots=True,
            test_labels=True,
            record=False,
            css_weight=False,
            chunked=True,
            ranked=False,
            record_support=False,
            ends_after_item=ends_after_item,
        )
    return _scan(
        basis,
        workspace,
        tables,
        prefixes[0],
        last_item,
        depth,
        best,
        cutoff,
        batch_size,
        best,
        pair_steps=True,
        deleted_pivots=True,
        test_labels=True,
        record=False,
        css_weight=False,
        chunked=True,
        ranked=True,
        record_support=False,
        ends_after_item=ends_after_item,
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
    # Placeholders fill the arguments that ``_scan`` neither reads nor writes under the flags of
    # this call.
    best, _, weighed, _, _ = _scan(
        PackedBasis(rows, rows, width, width),
        ScanWorkspace(support, prefixes, representative, support),
        NO_TABLE_VIEW,
        representative,
        support,
        0,
        best,
        cutoff,
        0,
        0,
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
    # Placeholders fill the arguments that ``_scan`` neither reads nor writes under the flags of
    # this call.
    best, _, weighed, _, _ = _scan(
        PackedBasis(rows, rows, width, width),
        ScanWorkspace(support, prefixes, representative, support),
        NO_TABLE_VIEW,
        x_mask,
        support,
        0,
        best,
        cutoff,
        0,
        0,
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
    A word weighs the number of its rows below ``rank`` plus the weight of its tail
    (``brouwer_zimmermann.md#sec-packed-enumeration``).  The scan copies the support of each lighter
    word, rather than the word, into ``best_support``, of at least ``len(support)`` entries.  The
    routes by tails of coset searches weigh words this way
    (``brouwer_zimmermann_code.md#sec-heuristics``).

    Returns:
        The weight of the lightest word found, or ``best`` if none is lighter, and the number of
        words weighed.
    """
    prefixes[0] = offset_tail
    support[0] = 0
    # Placeholders fill the arguments that ``_scan`` neither reads nor writes under the flags of
    # this call.
    best, _, weighed, _, _ = _scan(
        PackedBasis(tails, tails, rank, width),
        ScanWorkspace(support, prefixes, prefixes[0], best_support),
        NO_TABLE_VIEW,
        prefixes[0],
        support,
        0,
        best,
        cutoff,
        0,
        0,
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
    tables: CompletionTableView,
    best_support: npt.NDArray[np.int64],
) -> tuple[int, int, int]:
    """Scan a layer of one search basis of a coset by tails, skipping completions by tables.

    The scan weighs words as ``scan_coset_layer_by_tails`` does, in a layer of
    ``2 <= len(support) <= num_rows`` rows, and ``scan_coset_layer_by_tails`` describes the shared
    arguments.  ``tables`` are the completion tables of the tails of the rows, or
    ``completion_bounds.NO_TABLE_VIEW``, which skips nothing.  The scan skips completions by the
    tables as ``pair_steps`` of ``_scan`` describes
    (``brouwer_zimmermann.md#sec-completion-cosets``).  A coset query passes as ``skip_bound`` the
    least Hamming weight that proves a word of its coset heavier than the query's maximum weight
    (see ``coset._get_skip_bound``).  A search for a minimum weight passes at least ``best`` as
    ``skip_bound`` (``brouwer_zimmermann.md#sec-coset-min-weights``).

    Returns:
        The weight of the lightest word found, or ``best`` if none is lighter, and the numbers of
        words weighed and of supports skipped.
    """
    prefixes[0] = offset_tail
    support[0] = 0
    # Placeholders fill the arguments that ``_scan`` neither reads nor writes under the flags of
    # this call.
    best, _, weighed, _, skipped = _scan(
        PackedBasis(tails, tails, rank, width),
        ScanWorkspace(support, prefixes, prefixes[0], best_support),
        tables,
        prefixes[0],
        support,
        0,
        best,
        cutoff,
        0,
        skip_bound,
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


class ScanWorkspace(NamedTuple):
    """The arrays that a scan writes: its position, and its records of the lightest word.

    Attributes:
        support: The row indices of the current support, one for each row of the supports of the
            layer.
        prefixes: Row ``level`` holds the offset plus the rows ``support[:level]``, as
            ``support_stepping.advance_support`` describes.
        representative: The lightest word found, which only scans that record words write.
        best_support: The support of the lightest word found, of at least ``len(support)`` entries,
            which only scans that record supports write.
    """

    support: npt.NDArray[np.int64]
    prefixes: npt.NDArray[np.uint64]
    representative: npt.NDArray[np.uint64]
    best_support: npt.NDArray[np.int64]


@always_compiled
def _scan(
    basis: PackedBasis,
    workspace: ScanWorkspace,
    tables: CompletionTableView,
    x_mask: npt.NDArray[np.uint64],
    last_item: npt.NDArray[np.int64],
    depth: int,
    best: int,
    cutoff: int,
    batch_size: int,
    skip_bound: int,
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
    """Scan the supports of a layer of one search basis, starting from ``workspace.support``.

    ``PackedBasis``, ``ScanWorkspace``, and ``CompletionTableView`` describe ``basis``,
    ``workspace``, and ``tables``, and the entry points of this module describe the other arguments.
    The flags, which callers pass as constants, choose what the scan does:

    - ``pair_steps``: with ``deleted_pivots``, step through the first of the last two rows of each
      support separately.  The prefix of a support is its first ``len(support) - 2`` rows.  The
      completions of a prefix from row ``s`` are the supports that add to it two rows
      ``s <= i < j``.  With tables, the scan bounds the weight of these completions before it tries
      row ``s`` as the first of the last two rows.  The bound is that of the tables plus the least
      number of pivot ones that such a completion can have (see ``deleted_pivots`` and ``ranked``).
      The scan skips the completions if this bound reaches the lesser of ``skip_bound`` and the best
      weight (``brouwer_zimmermann.md#sec-completion-bound``).
    - ``deleted_pivots``: add to the weight of a word a pivot one for each of its rows
      (``brouwer_zimmermann.md#sec-packed-enumeration``).
    - ``test_labels``: keep only words whose quotient labels make them eligible.
    - ``record``: copy each lighter word into ``workspace.representative``.
    - ``css_weight``: weigh words by their CSS weight, with ``x_mask``, rather than Hamming weight.
    - ``chunked``: return once the weighed supports and the lookups reach ``batch_size`` together.
    - ``ranked``: with ``deleted_pivots``, count pivot ones only for rows below ``basis.rank``, in
      words and in bounds of completions (``brouwer_zimmermann.md#sec-completion-cosets``).
    - ``record_support``: copy the support of each lighter word into ``workspace.best_support``.
    - ``ends_after_item``: with ``pair_steps``, end the layer once the first ``len(last_item)``
      rows of a support pass ``last_item`` in lexicographic order (see ``scan_set_items``).

    Returns:
        Updated ``best`` and ``depth``, and the numbers of supports weighed, of bounds looked up,
        and of supports skipped.
    """
    # Read the fields of the records once.  Reading them in the loops would count references to
    # their arrays at each read, and threads that share the arrays would contend for the counts.
    rows, labels, rank, width = basis
    support, prefixes, representative, best_support = workspace
    table_bytes, table_offsets, block_bits = tables
    num_rows = rows.shape[0]
    weight = len(support)
    # With pair steps, the scan steps through heads, the first ``weight - 1`` rows of each support,
    # and tries each last row in the inner loop.  No head ends at the last row of the basis, which
    # no row can follow.  A layer of one row has empty heads, so the scan weighs its words at once.
    # ``_count_chunk_skipping`` in ``_counting/scan.py`` skips completions the same way.
    head = support[: weight - 1]
    head_rows = rows[: num_rows - 1]
    pair_depth = weight - 2
    evaluated, lookups, skipped = 0, 0, 0
    while True:
        if not pair_steps:
            depth = advance_support(rows, width, support, prefixes, depth)
            if depth < 0:
                return best, depth, evaluated, lookups, skipped
        elif depth < weight - 1:
            depth = advance_support(head_rows, width, head, prefixes, depth)
            if depth < 0:
                return best, depth, evaluated, lookups, skipped
            # End the range once the fixed rows of the support pass ``last_item``, in
            # lexicographic order.  The test stays here rather than in an inlined helper, whose
            # arguments Numba counted references to at each head, which made scans on threads about
            # a fifth slower.
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
            if len(table_bytes):
                lookups += 1
                least_pivot_ones = _get_least_completion_pivot_ones(support, num_rows, rank, ranked)
                limit = min(skip_bound, best) - least_pivot_ones
                bound = get_completion_bound(
                    table_bytes, table_offsets, block_bits, first, prefix, limit
                )
                if bound >= limit:
                    remaining = num_rows - first
                    skipped += remaining * (remaining - 1) // 2
                    support[pair_depth] = num_rows
                    if chunked and evaluated + lookups >= batch_size:
                        return best, depth, evaluated, lookups, skipped
                    continue

            # Otherwise add row ``first`` to the prefix, and try every later row as the last row.
            for word in range(width):
                prefixes[weight - 1, word] = prefix[word] ^ rows[first, word]
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
            head_pivots = np.int64(_count_pivot_rows(support, depth, rank))
        while row < num_rows:
            # Add the pivot ones of the rows, then the packed tail, until the word is no lighter
            # than the best.
            pivot_ones = np.int64(0)
            if deleted_pivots:
                if ranked:
                    pivot_ones = head_pivots + (1 if row < rank else 0)
                else:
                    pivot_ones = np.int64(weight)
            candidate = _weigh(
                prefixes, depth, rows, row, width, pivot_ones, best, x_mask, css_weight
            )

            # Keep a lighter word if its labels make it eligible.
            lighter = bool(candidate < best)
            if lighter and test_labels:
                support[depth] = row
                lighter = _is_eligible(labels, support)
            if lighter:
                best = int(candidate)
                if record:
                    for word in range(width):
                        representative[word] = prefixes[depth, word] ^ rows[row, word]
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
def _get_least_completion_pivot_ones(
    support: npt.NDArray[np.int64], num_rows: int, rank: int, ranked: bool
) -> int:
    """The least number of pivot ones of a completion of the prefix of a support.

    The prefix is the first ``len(support) - 2`` rows of ``support``, and its completions add two
    rows from row ``support[len(support) - 2]`` on, as ``pair_steps`` of ``_scan`` describes.  Each
    row of a completion has a pivot one, or with ``ranked``, each row below ``rank``, the first
    ``rank`` of the ``num_rows`` rows of the basis (``brouwer_zimmermann.md#sec-completion-bound``
    and ``brouwer_zimmermann.md#sec-completion-cosets``).
    """
    weight = len(support)
    if not ranked:
        return weight
    # At most ``num_rows - rank`` rows of the pair lack pivots.
    return max(0, 2 - (num_rows - rank)) + _count_pivot_rows(support, weight - 2, rank)


@always_compiled(inline=True)
def _count_pivot_rows(support: npt.NDArray[np.int64], length: int, rank: int) -> int:
    """The number of the first ``length`` rows of a support that lie below ``rank``."""
    count = 0
    for selected in range(length):
        if support[selected] < rank:
            count += 1
    return count


@always_compiled(inline=True)
def _weigh(
    prefixes: npt.NDArray[np.uint64],
    depth: int,
    rows: npt.NDArray[np.uint64],
    row: int,
    width: int,
    pivot_ones: np.int64,
    best: int,
    x_mask: npt.NDArray[np.uint64],
    css_weight: bool,
) -> np.int64:
    """Weigh the word ``prefixes[depth] ^ rows[row]``, in its first ``width`` packed words.

    The weight is ``pivot_ones`` plus the Hamming weight of the word.  With ``css_weight``, let the
    X part of the word be the bits that ``x_mask`` selects, and its Z part the other bits; the
    weight is then the larger of ``pivot_ones`` plus the Hamming weight of the X part, and the
    Hamming weight of the Z part.  The sums stop once one reaches ``best``, so the result is either
    the weight or a value from ``best`` up to the weight.
    """
    candidate = pivot_ones
    z_ones = np.int64(0)
    for word in range(width):
        value = prefixes[depth, word] ^ rows[row, word]
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
    return candidate


@always_compiled(inline=True)
def _is_eligible(labels: npt.NDArray[np.uint64], support: npt.NDArray[np.int64]) -> bool:
    """Decide whether the word of a support of a search basis lies outside the excluded subcode.

    It does if the basis has no label words, since then nothing is excluded, or if the XOR of the
    quotient labels of the rows of the support is nonzero
    (``brouwer_zimmermann.md#sec-quotient-labels``).
    """
    weight = len(support)
    label_width = labels.shape[1]
    eligible = label_width == 0
    for word in range(label_width):
        combined = labels[support[0], word]
        for selected in range(1, weight):
            combined ^= labels[support[selected], word]
        if combined:
            eligible = True
            break
    return eligible
