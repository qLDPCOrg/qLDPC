# SPDX-License-Identifier: Apache-2.0

"""Tests for information sets with bounded column coverage."""

from __future__ import annotations

import functools
import itertools

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import row_reduce_binary

from ..conftest import get_random_full_rank_matrix
from .matroid_partition import get_information_sets_with_coverage


def _is_information_set(basis: npt.NDArray[np.uint8], columns: list[int]) -> bool:
    """Whether the columns are as many as the rows of the basis, and independent."""
    _, _, pivots = row_reduce_binary(basis[:, columns], None)
    return len(columns) == len(basis) == len(pivots)


def _get_max_num_sets(basis: npt.NDArray[np.uint8], max_coverage: int, max_sets: int) -> int:
    """The most information sets, up to ``max_sets``, with no column in more than ``max_coverage``.

    A depth-first search over the information sets, in a fixed order with repetition, which is
    feasible for bases of a few columns.
    """
    information_sets = [
        columns
        for columns in itertools.combinations(range(basis.shape[1]), len(basis))
        if _is_information_set(basis, list(columns))
    ]

    @functools.cache
    def search(start: int, loads: tuple[int, ...], count: int) -> int:
        best = count
        if count == max_sets:
            return best
        for index in range(start, len(information_sets)):
            new_loads = list(loads)
            for column in information_sets[index]:
                new_loads[column] += 1
            if max(new_loads) <= max_coverage:
                best = max(best, search(index, tuple(new_loads), count + 1))
        return best

    return search(0, (0,) * basis.shape[1], 0)


def test_a_basis_whose_sets_need_columns_that_could_not_join_earlier() -> None:
    """Three information sets of this [10, 4] basis use no column more than twice.

    Filling the third set, some columns cannot join it, and the search skips their other copies, but
    it must still try the columns after them.
    """
    rows = ["1100000100", "0010001000", "0000100100", "0000010100"]
    basis = np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)
    assert _get_max_num_sets(basis, 2, 16) == 3
    assert len(get_information_sets_with_coverage(basis, 2, 16)) == 3


def test_sets_are_information_sets_with_bounded_coverage(pytestconfig: pytest.Config) -> None:
    """On random bases, the sets are as many as brute force finds, and each is an information set.

    A zero column is in no information set, so the sets never use one.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(25):
        dimension = int(rng.integers(1, 4))
        num_columns = int(rng.integers(dimension, 7))
        basis = get_random_full_rank_matrix(rng, dimension, num_columns)
        for max_coverage, max_sets in itertools.product((1, 2, 3), (2, 16)):
            sets = get_information_sets_with_coverage(basis, max_coverage, max_sets)
            assert all(_is_information_set(basis, columns) for columns in sets)
            loads = np.bincount(
                [column for columns in sets for column in columns], minlength=num_columns
            )
            assert loads.max(initial=0) <= max_coverage
            assert len(sets) == _get_max_num_sets(basis, max_coverage, max_sets)


def test_partial_sets_are_the_largest_that_information_sets_leave_room_for(
    pytestconfig: pytest.Config,
) -> None:
    """With coverage one, a kept partial set is as large as any that disjoint sets leave room for.

    On random bases, the sets are disjoint, as many information sets as brute force finds come
    first, and then a partial set of independent columns, unless it would be empty.  Its size is
    the largest rank of the columns that any choice of that many disjoint information sets leaves.
    Without a partial set, those columns have rank zero.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(25):
        dimension = int(rng.integers(1, 4))
        num_columns = int(rng.integers(dimension, 8))
        basis = get_random_full_rank_matrix(rng, dimension, num_columns)
        sets = get_information_sets_with_coverage(
            basis, 1, num_columns // dimension + 1, keep_partial=True
        )
        num_full = _get_max_num_sets(basis, 1, num_columns)
        assert all(_is_information_set(basis, columns) for columns in sets[:num_full])
        partial = sets[num_full:]
        assert len(partial) <= 1 and all(partial)
        columns = [column for columns in sets for column in columns]
        assert len(columns) == len(set(columns))
        size = len(partial[0]) if partial else 0
        assert len(row_reduce_binary(basis[:, columns[num_full * dimension :]], None)[2]) == size
        assert size == _get_largest_remaining_rank(basis, num_full)


def _get_largest_remaining_rank(basis: npt.NDArray[np.uint8], num_sets: int) -> int:
    """The largest rank of the columns that any ``num_sets`` disjoint information sets leave.

    A search over the disjoint information sets, in a fixed order, which is feasible for bases of a
    few columns.
    """
    information_sets = [
        set(columns)
        for columns in itertools.combinations(range(basis.shape[1]), len(basis))
        if _is_information_set(basis, list(columns))
    ]

    def search(start: int, used: frozenset[int], count: int) -> int:
        if count == num_sets:
            remaining = [column for column in range(basis.shape[1]) if column not in used]
            return len(row_reduce_binary(basis[:, remaining], None)[2]) if remaining else 0
        return max(
            (
                search(index + 1, used | columns, count + 1)
                for index, columns in enumerate(information_sets[start:], start)
                if not used & columns
            ),
            default=-1,
        )

    return search(0, frozenset(), 0)
