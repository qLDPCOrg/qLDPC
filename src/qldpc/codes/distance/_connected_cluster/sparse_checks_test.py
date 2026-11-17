# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the sparser checks of the connected-cluster searches."""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import row_reduce_binary

from . import sparse_checks


def test_sparser_rows_span_the_same_space(pytestconfig: pytest.Config) -> None:
    """``_sparsify`` keeps the span, and leaves no row that adding another row would lighten."""

    def get_weights(rows: npt.NDArray[np.uint8], symplectic: bool) -> npt.NDArray[np.int_]:
        if not symplectic:
            return rows.sum(axis=1)
        half = rows.shape[1] // 2
        return (rows[:, :half] | rows[:, half:]).sum(axis=1)

    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    cases = []
    for symplectic in (False, True):
        for _ in range(10):
            matrix = rng.integers(2, size=(6, 8), dtype=np.uint8)
            cases.append((matrix[row_reduce_binary(matrix.T.copy())[2]], symplectic))
    for matrix, symplectic in cases:
        rows, transform = sparse_checks._sparsify(matrix, symplectic=symplectic)
        assert np.array_equal(transform @ matrix % 2, rows)
        assert len(row_reduce_binary(transform)[2]) == len(matrix)
        weights = get_weights(rows, symplectic)
        for row, other in itertools.permutations(range(len(rows)), 2):
            assert get_weights((rows[row] ^ rows[other])[None], symplectic)[0] >= weights[row]


def test_rounds_of_random_reductions_make_rows_sparser() -> None:
    """Rounds reach a lightest basis, which adding rows to each other misses."""
    # Three independent words of the span act on three qubits each, a lightest basis of total weight
    # 9, and one of them has two Ys, so weighing bits would miss it.  Adding rows to each other
    # finds one of the words, and stops at total weight 10.
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
            matrix, symplectic=True, max_rounds=max_rounds, rounds_without_improvement=20
        )
        assert np.array_equal(transform @ matrix % 2, rows)
        assert len(row_reduce_binary(transform)[2]) == len(matrix)
        assert (rows[:, :5] | rows[:, 5:]).sum() == total_weight


def test_rounds_stop_once_they_leave_the_weight_unchanged() -> None:
    """With ``rounds_without_improvement=1``, a round without a lighter total ends the rounds."""
    # No round makes the rows of an identity lighter, and no rows give no rows.
    for matrix in (np.eye(3, dtype=np.uint8), np.zeros((0, 3), dtype=np.uint8)):
        rows, transform = sparse_checks.get_sparser_rows(
            matrix, symplectic=False, max_rounds=10**12, rounds_without_improvement=1
        )
        assert np.array_equal(rows, matrix)
        assert np.array_equal(transform, np.eye(len(matrix)))


def test_light_pair_sums_match_a_direct_choice(pytestconfig: pytest.Config) -> None:
    """``get_light_pair_sums`` takes the light new sums lightest first, within the weight budget.

    A direct choice weighs the sum of every pair of rows, keeps the distinct ones that are not zero,
    no heavier than the heavier row of the pair, and not a row, and takes them lightest first, ties
    by pair, while their total weight stays within that of the rows.  Random matrices with repeated
    rows, and qubit rows weighed by qubits, exercise the duplicates and the values of Paulis.  Three
    fixed cases sit at the bounds.  In the first, the 6 sums of pairs weigh 2 each, 12 in all, as
    much as the rows, so all are taken.  In the second, the 10 sums weigh 2 each, and the 7 taken
    leave 1 of the 15 of the rows, one less than the next sum.  In the third, XXI and IZZ share a
    qubit, as light sums must, but their sum XYZ weighs one more than either.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    cases = []
    for symplectic, _ in itertools.product((False, True), range(40)):
        num_rows, num_variables = rng.integers(0, 9), rng.integers(1, 9)
        width = 2 * num_variables if symplectic else num_variables
        rows = (rng.random((num_rows, width)) < rng.uniform(0.2, 0.8)).astype(np.uint8)
        if num_rows > 2:
            rows[1] = rows[0]
        cases.append((rows, symplectic))
    fixed_cases = [
        (np.hstack([np.ones((4, 2), dtype=np.uint8), np.eye(4, dtype=np.uint8)]), False),
        (np.hstack([np.ones((5, 2), dtype=np.uint8), np.eye(5, dtype=np.uint8)]), False),
        (np.array([[1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1]], dtype=np.uint8), True),
    ]

    num_sums = []
    for rows, symplectic in cases + fixed_cases:
        weights = sparse_checks.get_row_weights(rows, symplectic=symplectic)
        candidates = []
        for first, second in itertools.combinations(range(len(rows)), 2):
            row_sum = rows[first] ^ rows[second]
            weight = sparse_checks.get_row_weights(row_sum[None], symplectic=symplectic)[0]
            if 0 < weight <= max(weights[first], weights[second]):
                candidates.append((weight, first, second))
        known = {row.tobytes() for row in rows}
        budget, expected = weights.sum(), []
        for weight, first, second in sorted(candidates):
            row_sum = (rows[first] ^ rows[second]).tobytes()
            if row_sum not in known and weight <= budget:
                known.add(row_sum)
                expected.append((first, second))
                budget -= weight

        sums, pairs = sparse_checks.get_light_pair_sums(rows, symplectic=symplectic)
        assert pairs.tolist() == [list(pair) for pair in expected]
        assert np.array_equal(sums, rows[pairs[:, 0]] ^ rows[pairs[:, 1]])
        assert sums.shape == (len(expected), rows.shape[1])
        num_sums.append(len(expected))
    assert num_sums[-3:] == [6, 7, 0]
