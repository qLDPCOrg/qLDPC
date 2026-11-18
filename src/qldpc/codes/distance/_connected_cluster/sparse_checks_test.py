# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the sparser checks of the connected-cluster searches."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import row_reduce_binary

from ..conftest import get_random_full_rank_matrix
from . import sparse_checks


def test_sparser_rows_span_the_same_space(pytestconfig: pytest.Config) -> None:
    """``_sparsify`` keeps the span, and leaves no row that adding another row would lighten.

    Parts of 70 columns take two packed words each.
    """

    def get_weights(rows: npt.NDArray[np.uint8], symplectic: bool) -> npt.NDArray[np.int_]:
        if not symplectic:
            return rows.sum(axis=1)
        half = rows.shape[1] // 2
        return (rows[:, :half] | rows[:, half:]).sum(axis=1)

    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    cases = []
    for symplectic, num_variables in itertools.product((False, True), (4, 70)):
        for _ in range(5):
            width = 2 * num_variables if symplectic else num_variables
            matrix = rng.integers(2, size=(6, width), dtype=np.uint8)
            cases.append((matrix[row_reduce_binary(matrix.T.copy())[2]], symplectic))
    for matrix, symplectic in cases:
        rows, transform = sparse_checks._sparsify(matrix, symplectic=symplectic)
        assert np.array_equal(transform @ matrix % 2, rows)
        assert len(row_reduce_binary(transform)[2]) == len(matrix)
        weights = get_weights(rows, symplectic)
        for row, other in itertools.permutations(range(len(rows)), 2):
            assert get_weights((rows[row] ^ rows[other])[None], symplectic)[0] >= weights[row]


def test_rounds_of_random_reductions_make_rows_sparser() -> None:
    """Rounds reach a lightest basis, which adding rows to each other misses.

    The rounds reach it also with the X and Z parts of the rows swapped, which keeps the weights.
    """
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
    swapped = np.hstack([matrix[:, 5:], matrix[:, :5]])
    for (max_rounds, total_weight), checks in itertools.product(
        [(0, 10), (20, 9)], [matrix, swapped]
    ):
        rows, transform = sparse_checks.get_sparser_rows(
            checks, symplectic=True, max_rounds=max_rounds, rounds_without_improvement=20
        )
        assert np.array_equal(transform @ checks % 2, rows)
        assert len(row_reduce_binary(transform)[2]) == len(checks)
        assert (rows[:, :5] | rows[:, 5:]).sum() == total_weight


def test_a_round_on_packed_rows_matches_a_round_on_unpacked_rows(
    pytestconfig: pytest.Config,
) -> None:
    """``_take_lighter_rows`` keeps the rows that a round on unpacked rows keeps.

    The round row-reduces the matrix in an order of its columns, sorts the current rows and the rows
    of the reduction that are lighter than the heaviest current row stably by weight, the current
    rows first, and keeps each that is independent of those before it.  Binary rows and the parts
    of symplectic rows have 5, 19, or 70 columns, so that each takes one packed word or two, and the
    Z part of a packed row starts at a word of its own.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for symplectic, num_variables, _ in itertools.product((False, True), (5, 19, 70), range(6)):
        width = 2 * num_variables if symplectic else num_variables
        matrix = (rng.random((8, width)) < rng.uniform(0.2, 0.8)).astype(np.uint8)
        matrix[0, rng.integers(width)] = 1  # so that the round has a current row
        rows = matrix[row_reduce_binary(np.ascontiguousarray(matrix.T))[2]]
        weights = sparse_checks.get_row_weights(rows, symplectic=symplectic)
        columns = rng.permutation(width)

        expected, expected_weights = rows, weights
        reduced, _, _ = row_reduce_binary(rows, columns=columns)
        reduced_weights = sparse_checks.get_row_weights(reduced, symplectic=symplectic)
        light = reduced_weights < weights.max()
        if np.any(light):
            candidates = np.vstack([rows, reduced[light]])
            candidate_weights = np.concatenate([weights, reduced_weights[light]])
            order = np.argsort(candidate_weights, kind="stable")
            _, _, pivots = row_reduce_binary(np.ascontiguousarray(candidates[order].T))
            expected, expected_weights = candidates[order[pivots]], candidate_weights[order[pivots]]

        packed, positions = sparse_checks._pack_parts(rows, symplectic=symplectic)
        kept, kept_weights = sparse_checks._take_lighter_rows(
            packed, positions[columns], packed, weights, symplectic
        )
        unpacked = sparse_checks._unpack_parts(kept, width, symplectic=symplectic)
        assert np.array_equal(unpacked, expected)
        assert np.array_equal(kept_weights, expected_weights)


def test_rounds_stop_once_they_leave_the_weight_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rounds stop after ``rounds_without_improvement`` unchanged rounds in a row, or at a limit.

    The limit is ``max_rounds``, and a round is unchanged if it leaves the total weight of the rows
    unchanged.  The rounds on this matrix leave the total weight unchanged, then lower it, and then
    leave it unchanged, so that a round that improves resets the count of unchanged rounds.
    """
    totals: list[tuple[int, int]] = []
    take_lighter_rows = sparse_checks._take_lighter_rows

    def record_totals(*args: Any) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
        rows, weights = take_lighter_rows(*args)
        totals.append((int(args[3].sum()), int(weights.sum())))
        return rows, weights

    monkeypatch.setattr(sparse_checks, "_take_lighter_rows", record_totals)
    matrix = np.array(
        [
            [0, 0, 1, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 1, 1, 0, 1],
            [1, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    improvements = [False, True, False, False]
    cases = [
        # The first round leaves the weight unchanged, so the rounds end after it.
        (10, 1, 1),
        # The second round improves, so the count starts again, and two more rounds end it.
        (10, 2, 4),
        # The rounds stop at max_rounds.
        (2, 2, 2),
    ]
    for max_rounds, rounds_without_improvement, num_rounds in cases:
        totals.clear()
        sparse_checks.get_sparser_rows(
            matrix,
            symplectic=False,
            max_rounds=max_rounds,
            rounds_without_improvement=rounds_without_improvement,
        )
        assert [after < before for before, after in totals] == improvements[:num_rounds]

    # No rows give no rows, without rounds.
    totals.clear()
    rows, transform = sparse_checks.get_sparser_rows(
        np.zeros((0, 3), dtype=np.uint8),
        symplectic=False,
        max_rounds=10,
        rounds_without_improvement=1,
    )
    assert np.array_equal(rows, np.zeros((0, 3))) and np.array_equal(transform, np.eye(0))
    assert not totals


def test_sparser_rows_are_reproducible(monkeypatch: pytest.MonkeyPatch) -> None:
    """``get_sparser_rows`` gives the same rows and transform each time, after random rounds.

    The rows of rounds on this matrix, of many more columns than rows, depend on the random orders
    of the columns, which the fixed seed of the rounds makes the same each time.
    """
    matrix = get_random_full_rank_matrix(np.random.default_rng(0), 6, 20)

    def get_results() -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
        return sparse_checks.get_sparser_rows(
            matrix, symplectic=False, max_rounds=3, rounds_without_improvement=3
        )

    rows, transform = get_results()
    again = get_results()
    assert np.array_equal(again[0], rows) and np.array_equal(again[1], transform)
    monkeypatch.setattr(sparse_checks, "_SPARSIFYING_SEED", 1)
    assert not np.array_equal(get_results()[0], rows)


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
