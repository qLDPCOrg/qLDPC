# SPDX-License-Identifier: Apache-2.0

"""Information sets with bounded column coverage, by matroid partitioning.

The independent sets of columns of a binary basis form a linear matroid, whose bases are the
information sets.  As in Section 5.1 of Lisoněk and Trummer (References of
``brouwer_zimmermann.md``), ``get_information_sets_with_coverage`` finds the most information sets
that use no column more than ``b`` times, by partitioning ``b`` copies of each column into
independent sets with Edmonds' matroid partitioning algorithm.  Section 4.4 of
``brouwer_zimmermann.md`` says which searches use it.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator

import numpy as np
import numpy.typing as npt


def get_information_sets_with_coverage(
    basis: npt.NDArray[np.uint8], max_coverage: int, max_sets: int, *, keep_partial: bool = False
) -> list[list[int]]:
    """Find the most information sets, up to ``max_sets``, that use no column too often.

    With ``keep_partial``, a nonempty partial set, of fewer independent columns than the rows, may
    follow the information sets, and counts toward ``max_sets``.  With coverage one, no other choice
    of as many information sets leaves room for a larger one (Section 4.4 of
    ``brouwer_zimmermann.md``).

    Args:
        basis: A binary matrix with independent rows.
        max_coverage: The largest number of the sets that may contain any one column.
        max_sets: The largest number of sets to find.
        keep_partial: Whether to keep the last set if it could not be filled.

    Returns:
        The sorted column indices of each set.
    """
    dimension = len(basis)
    columns = np.flatnonzero(np.any(basis, axis=0)).tolist()
    vectors = _get_column_ints(basis[:, columns])
    # Element ``e`` is a copy of column ``columns[e // max_coverage]``.
    copies = [vector for vector in vectors for _ in range(max_coverage)]
    partition = _Partition(copies)
    while len(partition.sets) < max_sets:
        partition.add_set()
        if not partition.fill(dimension, max_coverage):
            if not (keep_partial and partition.sets[-1].elements):
                partition.remove_last_set()
            break
    return [
        sorted(columns[element // max_coverage] for element in independent.elements)
        for independent in partition.sets
    ]


def _get_column_ints(matrix: npt.NDArray[np.uint8]) -> list[int]:
    """The columns of a binary matrix as integers, with row ``i`` in bit ``i``."""
    packed = np.packbits(np.asarray(matrix, dtype=np.uint8).T, axis=1, bitorder="little")
    return [int.from_bytes(column.tobytes(), "little") for column in packed]


class _IndependentSet:
    """Independent column vectors, row reduced so that circuits are cheap to find.

    ``_rows`` maps each pivot to a pair ``(row, mask)``.  The row is the sum of the members whose
    positions are the set bits of ``mask``, and it is the only row with its pivot bit set.  So
    reducing a vector adds the row of each pivot bit that the vector has set.  Circuits are cached
    by vector until the members change.
    """

    def __init__(self) -> None:
        self.elements: list[int] = []
        self._rows: dict[int, tuple[int, int]] = {}
        self._pivots = 0
        self._circuits: dict[int, list[int] | None] = {}

    def add(self, element: int, vectors: list[int]) -> None:
        """Add an element whose vector is independent of the members, and reduce the rows."""
        vector, mask = self._reduce(vectors[element], 1 << len(self.elements))
        assert vector, "a member depends on the others"
        pivot = vector.bit_length() - 1
        # Clear the new pivot from the other rows.
        for row_pivot, (row, row_mask) in self._rows.items():
            if row >> pivot & 1:
                self._rows[row_pivot] = row ^ vector, row_mask ^ mask
        self._rows[pivot] = vector, mask
        self._pivots |= 1 << pivot
        self.elements.append(element)
        self._circuits.clear()

    def remove(self, element: int) -> None:
        """Remove a member, and keep the rows reduced."""
        # Drop a row whose mask has the member's position, after adding that row to the others whose
        # masks have it.  That row has no bit at the other pivots, so they stay pivots of theirs.
        position = self.elements.index(element)
        pivot = next(pivot for pivot, (_, mask) in self._rows.items() if mask >> position & 1)
        vector, mask = self._rows.pop(pivot)
        self._pivots ^= 1 << pivot
        low = (1 << position) - 1
        for row_pivot, (row, row_mask) in self._rows.items():
            if row_mask >> position & 1:
                row, row_mask = row ^ vector, row_mask ^ mask
            # Close the gap that the member leaves in the positions.
            self._rows[row_pivot] = row, (row_mask & low) | (row_mask >> (position + 1) << position)
        del self.elements[position]
        self._circuits.clear()

    def get_circuit(self, vector: int) -> list[int] | None:
        """The members that sum to ``vector``, or None if it is independent of them."""
        if vector in self._circuits:
            return self._circuits[vector]
        reduced, mask = self._reduce(vector, 0)
        circuit = None if reduced else [self.elements[position] for position in _get_bits(mask)]
        self._circuits[vector] = circuit
        return circuit

    def _reduce(self, vector: int, mask: int) -> tuple[int, int]:
        """Reduce a vector by the rows, tracking which members the reduction adds."""
        for pivot in _get_bits(vector & self._pivots):
            row, row_mask = self._rows[pivot]
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
                last.add(element, self.vectors)
                self.owner[element] = len(self.sets) - 1

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

        In the exchange graph, an element ``x`` points to the members of its circuits in the other
        sets, which ``x`` could replace.  The path ends at an element independent of the members of
        some set, and that set must be the last, since every other set is full and so spans every
        column.  So every set but the last keeps its size.
        """
        parent: dict[int, tuple[int, int]] = {}
        queue = deque([element])
        seen = {element}
        # An element skips each set in which an earlier element with the same vector already found a
        # circuit.  Equal vectors have equal circuits in a set, and that circuit's members are seen.
        searched: dict[int, set[int]] = {}
        end: tuple[int, int] | None = None
        while queue and end is None:
            current = queue.popleft()
            vector = self.vectors[current]
            searched_sets = searched.setdefault(vector, set())
            for index, independent in enumerate(self.sets):
                if self.owner[current] == index or index in searched_sets:
                    continue
                searched_sets.add(index)
                circuit = independent.get_circuit(vector)
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
        # Each set ends independent, so removing the moved elements first keeps it independent as
        # each one joins.
        moves = [end]
        while moves[-1][0] != element:
            moves.append(parent[moves[-1][0]])
        for moved, _ in moves:
            if self.owner[moved] >= 0:
                self.sets[self.owner[moved]].remove(moved)
        for moved, index in moves:
            self.sets[index].add(moved, self.vectors)
            self.owner[moved] = index
        return True


def _get_bits(value: int) -> Iterator[int]:
    """The positions of the set bits of a nonnegative integer, in increasing order."""
    while value:
        low = value & -value
        yield low.bit_length() - 1
        value ^= low
