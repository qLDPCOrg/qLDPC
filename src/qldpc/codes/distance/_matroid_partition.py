# SPDX-License-Identifier: Apache-2.0

"""Information sets with bounded column coverage, by matroid partitioning.

The columns of a binary basis with ``k`` independent rows form a linear matroid, whose independent
sets are the sets of linearly independent columns, and whose bases of size ``k`` are the information
sets.  Lisonek and Trummer, "Algorithms for the minimum weight of linear codes," *Advances in
Mathematics of Communications* 10(1), 195-207 (2016), Section 5.1, find the largest number of
information sets in which no column appears more than ``b`` times by partitioning ``b`` copies of
each column into independent sets, with Edmonds' matroid partitioning algorithm.

``get_information_sets_with_coverage`` does this.  It grows the sets one at a time: it fills a new
set with columns that keep it independent, and then moves columns between sets along augmenting
paths, as in Edmonds' algorithm, until the set holds ``k`` columns or no column can join.  An
augmenting path keeps the size of every set but the last, so full sets stay full.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import numpy.typing as npt


def get_information_sets_with_coverage(
    basis: npt.NDArray[np.uint8], max_coverage: int, max_sets: int
) -> list[list[int]]:
    """Find the most information sets, up to ``max_sets``, in which no column appears too often.

    Args:
        basis: A binary matrix with independent rows.
        max_coverage: The largest number of the sets that may contain any one column.
        max_sets: The largest number of sets to find.

    Returns:
        The column indices of each information set, at most ``max_sets`` of them, as many as the
        coverage allows.
    """
    dimension = len(basis)
    columns = [column for column in range(basis.shape[1]) if np.any(basis[:, column])]
    vectors = [_get_column_int(basis, column) for column in columns]
    # Element ``e`` is a copy of column ``columns[e // max_coverage]``.
    copies = [vector for vector in vectors for _ in range(max_coverage)]
    partition = _Partition(copies)
    while len(partition.sets) < max_sets:
        partition.add_set()
        if not partition.fill(dimension, max_coverage):
            partition.remove_last_set()
            break
    return [
        sorted(columns[element // max_coverage] for element in independent.elements)
        for independent in partition.sets
    ]


def _get_column_int(basis: npt.NDArray[np.uint8], column: int) -> int:
    """A column of a binary matrix as an integer, with row ``i`` in bit ``i``."""
    return sum(int(bit) << row for row, bit in enumerate(basis[:, column]))


class _IndependentSet:
    """Independent column vectors, row reduced so that circuits are cheap to find.

    Row ``(pivot, vector, mask)`` of the reduction is the sum of the members whose positions are the
    set bits of ``mask``, its highest set bit is ``pivot``, and no other row has that bit set.
    """

    def __init__(self) -> None:
        self.elements: list[int] = []
        self._rows: list[tuple[int, int, int]] = []

    def rebuild(self, vectors: list[int]) -> None:
        """Row reduce the vectors of the members again, after members have changed."""
        self._rows = []
        for position, element in enumerate(self.elements):
            vector, mask = self._reduce(vectors[element], 1 << position)
            assert vector, "a member depends on the others"
            pivot = vector.bit_length() - 1
            # Clear the new pivot from the other rows, so that every query reduces in one pass.
            self._rows = [
                (row_pivot, row ^ vector, row_mask ^ mask)
                if row >> pivot & 1
                else (row_pivot, row, row_mask)
                for row_pivot, row, row_mask in self._rows
            ]
            self._rows.append((pivot, vector, mask))

    def get_circuit(self, vector: int) -> list[int] | None:
        """The members that sum to ``vector``, or None if it is independent of them."""
        vector, mask = self._reduce(vector, 0)
        if vector:
            return None
        return [element for position, element in enumerate(self.elements) if mask >> position & 1]

    def _reduce(self, vector: int, mask: int) -> tuple[int, int]:
        """Reduce a vector by the rows, tracking which members the reduction adds."""
        for pivot, row, row_mask in self._rows:
            if vector >> pivot & 1:
                vector ^= row
                mask ^= row_mask
        return vector, mask


class _Partition:
    """Disjoint independent sets of elements, each element a column vector."""

    def __init__(self, vectors: list[int]) -> None:
        self.vectors = vectors
        self.sets: list[_IndependentSet] = []
        self.owner = [-1] * len(vectors)

    def add_set(self) -> None:
        self.sets.append(_IndependentSet())

    def remove_last_set(self) -> None:
        for element in self.sets.pop().elements:
            self.owner[element] = -1

    def fill(self, size: int, num_copies: int) -> bool:
        """Grow the last set to ``size`` elements if possible, and return whether it got there.

        Copies of one column are ``num_copies`` consecutive elements with equal vectors, so if one
        copy cannot join, neither can the others.
        """
        last = self.sets[-1]
        # Add elements that keep the last set independent.
        for element, vector in enumerate(self.vectors):
            if len(last.elements) == size:
                return True
            if self.owner[element] < 0 and last.get_circuit(vector) is None:
                last.elements.append(element)
                self.owner[element] = len(self.sets) - 1
                last.rebuild(self.vectors)

        # Then move elements between sets along augmenting paths.
        stuck_columns: set[int] = set()
        for element in range(len(self.vectors)):
            if len(last.elements) == size:
                return True
            if self.owner[element] >= 0 or element // num_copies in stuck_columns:
                continue
            if not self._augment(element, size):
                stuck_columns.add(element // num_copies)
        return len(last.elements) == size

    def _augment(self, element: int, size: int) -> bool:
        """Add an unassigned element along a shortest augmenting path, if there is one.

        In the exchange graph, an element ``x`` points to the members of a set that ``x`` could
        replace: the members of its circuit in that set.  The path ends at an element that is
        independent of a set, which then has room for it: every set but the last is full, and
        ``size`` columns of a basis of ``size`` rows span every column, so only the last set can be
        one.
        """
        parent: dict[int, tuple[int, int]] = {}
        queue = deque([element])
        seen = {element}
        end: tuple[int, int] | None = None
        while queue and end is None:
            current = queue.popleft()
            for index, independent in enumerate(self.sets):
                if self.owner[current] == index:
                    continue
                circuit = independent.get_circuit(self.vectors[current])
                if circuit is None:
                    assert len(independent.elements) < size
                    end = (current, index)
                    break
                for neighbor in circuit:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        parent[neighbor] = (current, index)
                        queue.append(neighbor)
        if end is None:
            return False

        # Move each element on the path into the set of the next, and the last into the open set.
        moves = [end]
        while moves[-1][0] != element:
            moves.append(parent[moves[-1][0]])
        changed = set()
        for moved, index in moves:
            if self.owner[moved] >= 0:
                self.sets[self.owner[moved]].elements.remove(moved)
                changed.add(self.owner[moved])
            self.sets[index].elements.append(moved)
            self.owner[moved] = index
            changed.add(index)
        for index in changed:
            self.sets[index].rebuild(self.vectors)
        return True
