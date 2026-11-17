# SPDX-License-Identifier: Apache-2.0

"""The step of the scans of search bases from one support of a layer to the next."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qldpc._compilation import always_compiled


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
    offset (Section 3.5 of ``brouwer_zimmermann.md``).  The state is the rows chosen so far,
    ``support[:depth]``; the next row to try, ``support[depth]``; and ``prefixes``, whose row
    ``level`` holds the offset plus the rows ``support[:level]``.  Start with ``depth = 0``,
    ``support[0] = 0``, and the offset in ``prefixes[0]``, which stays unchanged.

    Returns ``len(support) - 1`` once ``support`` is complete, when its word is ``prefixes[depth] ^
    rows[support[depth]]``, or -1 after the last support.  To continue, raise ``support[depth]``
    and call again with that depth: by one for the next support, or past the last row to skip the
    supports that differ from this one only in their last row.
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
