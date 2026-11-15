# SPDX-License-Identifier: Apache-2.0

"""The step of the scans of information sets from one support of a layer to the next."""

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
    """Move to the next support of a layer of the search.

    Layer ``w = len(support) >= 1`` holds the sums of an offset and ``w`` of the packed ``rows`` of
    an information set.  The sums use only the first ``width`` machine words of each row.  A support
    lists the indices of the rows of a sum in increasing order, and the supports come in
    lexicographic order.

    The state of the walk is ``support``, ``prefixes``, and ``depth``.  The rows chosen so far are
    ``support[:depth]``, and ``support[depth]`` is the next row to try at ``depth``.  Row ``level``
    of ``prefixes`` holds the offset plus the rows ``support[:level]``.  Start with ``depth = 0``,
    ``support[0] = 0``, and the offset in ``prefixes[0]``, which this function does not change.

    Returns the last depth, ``len(support) - 1``, once ``support`` is complete.  The word of the
    support is then ``prefixes[depth] ^ rows[support[depth]]``.  To continue, raise
    ``support[depth]`` and call again from that depth.  Raising it by one moves to the next support,
    and raising it past the last row skips the supports that differ from this one only in their last
    row.  Returns -1 once every support has been visited.
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
