# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the sparser checks of the connected-support searches."""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import row_reduce_binary

from . import sparse_checks


def test_sparser_rows_span_the_same_space(pytestconfig: pytest.Config) -> None:
    """Adding rows to each other leaves rows of the same span, which no such addition makes lighter.

    The weight of a symplectic row counts the qubits on which it acts, whether by X, Y, or Z.
    """

    def get_weights(rows: npt.NDArray[np.uint8], homogeneous: bool) -> npt.NDArray[np.int_]:
        if homogeneous:
            return rows.sum(axis=1)
        half = rows.shape[1] // 2
        return (rows[:, :half] | rows[:, half:]).sum(axis=1)

    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    cases = []
    for homogeneous in (True, False):
        for _ in range(10):
            matrix = rng.integers(2, size=(6, 8), dtype=np.uint8)
            cases.append((matrix[row_reduce_binary(matrix.T.copy())[2]], homogeneous))
    for matrix, homogeneous in cases:
        rows, transform = sparse_checks._sparsify(matrix, homogeneous=homogeneous)
        assert np.array_equal(transform @ matrix % 2, rows)
        assert len(row_reduce_binary(transform)[2]) == len(matrix)
        weights = get_weights(rows, homogeneous)
        for row, other in itertools.permutations(range(len(rows)), 2):
            assert get_weights((rows[row] ^ rows[other])[None], homogeneous)[0] >= weights[row]


def test_rounds_of_random_reductions_make_rows_sparser() -> None:
    """Rounds reach a lightest basis, which adding rows to each other misses.

    Three independent words of the span of these Paulis on five qubits act on three qubits each, so
    they form a lightest basis, of total weight 9.  Adding rows to each other replaces the first row
    with its sum with the second, which is one of them, and then no sum of two rows is lighter than
    a row, so zero rounds give total weight 10.  Another of the three words has two Ys, so it counts
    five bits, and rows weighed by bits would miss it.
    """
    matrix = np.array(
        [
            [0, 1, 1, 0, 1, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    for max_rounds, total_weight in [(0, 10), (20, 9)]:
        rows, transform = sparse_checks.get_sparser_rows(
            matrix, homogeneous=False, max_rounds=max_rounds, rounds_without_improvement=20
        )
        assert np.array_equal(transform @ matrix % 2, rows)
        assert len(row_reduce_binary(transform)[2]) == len(matrix)
        assert (rows[:, :5] | rows[:, 5:]).sum() == total_weight


def test_rounds_stop_once_they_leave_the_weight_unchanged() -> None:
    """Rounds stop after the given number of rounds without a lighter total weight.

    The rows of an identity matrix weigh one each, so no round makes them lighter, and the first
    round ends the rounds, long before ``max_rounds`` would.  No rows give no rows.
    """
    for matrix in (np.eye(3, dtype=np.uint8), np.zeros((0, 3), dtype=np.uint8)):
        rows, transform = sparse_checks.get_sparser_rows(
            matrix, homogeneous=True, max_rounds=10**12, rounds_without_improvement=1
        )
        assert np.array_equal(rows, matrix)
        assert np.array_equal(transform, np.eye(len(matrix)))
