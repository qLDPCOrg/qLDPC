# SPDX-License-Identifier: Apache-2.0

"""The step of the scans of information sets from one support of a layer to the next."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qldpc._compilation import always_compiled


@always_compiled(inline=True)
def advance_support(
    rows: npt.NDArray[np.uint64],
    widths: npt.NDArray[np.int64],
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
) -> int:
    """Move to the next support of a layer of the search.

    Layer ``w = len(support) >= 1`` holds, for each information set, the sums of its offset and
    ``w`` of its rows.  A support lists the indices of such rows in increasing order, and the
    supports come in lexicographic order.  ``rows`` holds the packed rows of each information set,
    and set ``j`` uses the first ``widths[j]`` machine words of each row.  The rows chosen so far
    are ``support[:depth]``, and ``support[depth]`` is the next row to try at ``depth``.  Row
    ``level`` of ``prefixes`` holds, for each set, its offset plus its rows ``support[:level]``.
    Start with ``depth = 0``, ``support[0] = 0``, and the offsets in ``prefixes[0]``, which this
    function does not change.

    Return the last depth, ``len(support) - 1``, once ``support`` is complete.  The word of the
    support in set ``j`` is then ``prefixes[depth, j] ^ rows[j, support[depth]]``.  To continue,
    raise ``support[depth]``, by one or past the last row to skip the supports that differ only in
    their last row, and call again from that depth.  Return -1 once every support has been visited.
    """
    num_rows = rows.shape[1]
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
        for set_index in range(len(widths)):
            for word in range(widths[set_index]):
                prefixes[depth + 1, set_index, word] = (
                    prefixes[depth, set_index, word] ^ rows[set_index, row, word]
                )
        depth += 1
        support[depth] = row + 1
    return depth
