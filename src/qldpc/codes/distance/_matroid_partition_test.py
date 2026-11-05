# SPDX-License-Identifier: Apache-2.0

"""Tests for information sets with bounded column coverage."""

from __future__ import annotations

import functools
import itertools

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import get_independent_rows, row_reduce_binary

from ._matroid_partition import get_information_sets_with_coverage


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


def test_three_pairs_of_three_columns() -> None:
    """The columns ``e1, e2, e1 + e2`` hold one information set alone, and three pairs at twice."""
    basis = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.uint8)
    assert len(get_information_sets_with_coverage(basis, 1, 16)) == 1
    sets = get_information_sets_with_coverage(basis, 2, 16)
    assert sorted(sets) == [[0, 1], [0, 2], [1, 2]]
    assert len(get_information_sets_with_coverage(basis, 2, 2)) == 2


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
        basis = get_independent_rows(
            rng.integers(0, 2, size=(dimension, num_columns), dtype=np.uint8)
        )
        if not len(basis):
            continue
        for max_coverage, max_sets in itertools.product((1, 2, 3), (2, 16)):
            sets = get_information_sets_with_coverage(basis, max_coverage, max_sets)
            assert all(_is_information_set(basis, columns) for columns in sets)
            loads = np.bincount(
                [column for columns in sets for column in columns], minlength=num_columns
            )
            assert loads.max(initial=0) <= max_coverage
            assert len(sets) == _get_max_num_sets(basis, max_coverage, max_sets)
