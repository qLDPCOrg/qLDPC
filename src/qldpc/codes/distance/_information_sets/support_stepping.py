# SPDX-License-Identifier: Apache-2.0

"""Packed search bases, and the step of their scans from one support of a layer to the next.

``PackedBasis`` holds a search basis as compiled scans read it.  Splitting a layer into work items
for threads relies on the lexicographic order of the steps ("Keep the scans of a layer splittable
across threads" of ``brouwer_zimmermann_code.md#sec-requirements``).  ``advance_support`` also runs
within nogil calls of threaded searches, so it must keep the rules for nogil code in the docstring
of ``qldpc/_compilation.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._compilation import always_compiled


class PackedBasis(NamedTuple):
    """A search basis, packed as compiled scans read it.

    A word of the basis is an offset plus a sum of rows of the basis.  The record holds no offset:
    ``advance_support`` keeps it in row 0 of ``prefixes``.  In a search with deleted pivots, the
    packed rows omit the pivot columns, and a word weighs its packed tail plus one for each of its
    rows below ``rank``, the rows that have pivots
    (``brouwer_zimmermann.md#sec-packed-enumeration``).

    Attributes:
        rows: The packed rows, with shape ``(num_rows, num_words)``.
        labels: The packed quotient labels of the rows, with shape ``(num_rows, num_label_words)``.
            A word is eligible if the XOR of the labels of its rows is nonzero, or if the labels
            have no words (``brouwer_zimmermann.md#sec-quotient-labels``).
        rank: The number of leading rows that have pivots.
        width: The number of packed words of the rows that a scan reads.
    """

    rows: npt.NDArray[np.uint64]
    labels: npt.NDArray[np.uint64]
    rank: int
    width: int


@always_compiled(inline=True)
def advance_support(
    rows: npt.NDArray[np.uint64],
    width: int,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
) -> int:
    """Move to the next support of a layer, in lexicographic order.

    A support of layer ``len(support)`` lists the indices of rows of a search basis in increasing
    order, and its word adds those packed ``rows``, in their first ``width`` machine words, to an
    offset (``brouwer_zimmermann.md#sec-packed-enumeration``).  The state is the rows chosen so far,
    ``support[:depth]``; the next row to try, ``support[depth]``; and ``prefixes``, whose row
    ``level`` holds the offset plus the rows ``support[:level]``.  Start with ``depth = 0``,
    ``support[0] = 0``, and the offset in ``prefixes[0]``, which stays unchanged.  To continue after
    a complete support (see Returns), raise ``support[depth]`` at the returned ``depth``, and call
    again with that depth.  Raising it by one gives the next support.  Raising it past the last row
    skips the later supports that differ from the current one only in their last row.

    Returns:
        ``len(support) - 1`` once ``support`` is complete, when its word is
        ``prefixes[depth] ^ rows[support[depth]]``, or -1 after the last support.
    """
    num_rows = rows.shape[0]
    weight = len(support)
    while depth >= 0:
        # Backtrack once the row at this depth leaves too few rows to complete the support.
        if support[depth] > num_rows - weight + depth:
            depth -= 1
            if depth >= 0:
                support[depth] += 1
            continue
        if depth + 1 == weight:
            return depth

        # Add the row at this depth to the prefixes, and try the next rows at the next depth.
        row = support[depth]
        for word in range(width):
            prefixes[depth + 1, word] = prefixes[depth, word] ^ rows[row, word]
        depth += 1
        support[depth] = row + 1
    return depth
