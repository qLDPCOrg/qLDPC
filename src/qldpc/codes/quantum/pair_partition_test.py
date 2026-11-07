# SPDX-License-Identifier: Apache-2.0

"""Unit tests for pair_partition.py."""

from __future__ import annotations

import itertools

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes

# instance qc_276_98_14 in https://github.com/kasaikenta/pair-partition-cpm-css-codes
PAIR_PARTITION_ARRAY_276 = [
    [
        [[1, 7], [3, 9], [5, 11], [0, 6], [2, 8], [4, 10]],
        [[3, 11], [0, 10], [4, 9], [1, 8], [5, 7], [2, 6]],
        [[5, 10], [2, 11], [1, 9], [4, 8], [3, 6], [0, 7]],
        [[4, 6], [5, 9], [2, 7], [3, 10], [0, 8], [1, 11]],
    ],
    [
        [[3, 6], [1, 9], [5, 10], [0, 7], [4, 8], [2, 11]],
        [[2, 9], [4, 7], [5, 6], [0, 11], [1, 10], [3, 8]],
        [[0, 9], [1, 6], [2, 10], [3, 7], [4, 11], [5, 8]],
        [[4, 9], [5, 7], [3, 11], [2, 6], [0, 10], [1, 8]],
    ],
    [
        [[1, 8], [0, 10], [2, 6], [3, 11], [5, 7], [4, 9]],
        [[5, 8], [4, 11], [3, 7], [2, 10], [1, 6], [0, 9]],
        [[3, 8], [1, 10], [0, 11], [5, 6], [4, 7], [2, 9]],
        [[2, 11], [4, 8], [0, 7], [5, 10], [1, 9], [3, 6]],
    ],
    [
        [[1, 11], [0, 8], [3, 10], [2, 7], [5, 9], [4, 6]],
        [[0, 7], [3, 6], [4, 8], [1, 9], [2, 11], [5, 10]],
        [[2, 6], [5, 7], [1, 8], [4, 9], [0, 10], [3, 11]],
        [[4, 10], [2, 8], [0, 6], [5, 11], [3, 9], [1, 7]],
    ],
]


EXPONENTS_X_276 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 10, 16, 12, 20, 9, 6, 11, 2, 5, 21, 22],
    [0, 7, 20, 11, 3, 12, 2, 22, 14, 19, 13, 6],
    [0, 19, 17, 4, 2, 21, 12, 10, 18, 3, 11, 9],
]


EXPONENTS_Z_276 = [
    [0, 11, 18, 6, 13, 1, 0, 11, 18, 6, 13, 1],
    [0, 12, 7, 22, 19, 10, 7, 10, 12, 19, 0, 22],
    [0, 5, 6, 1, 4, 11, 1, 0, 4, 5, 11, 6],
    [0, 8, 22, 21, 12, 20, 12, 22, 0, 20, 21, 8],
]


def assert_pair_partition_cycle_properties(code: codes.PairPartitionCode) -> None:
    """Check PairPartitionCode.has_4_cycles and .is_mixed_distinct against their binary matrices."""
    matrix_x = code.matrix_x.view(np.ndarray).astype(int)
    matrix_z = code.matrix_z.view(np.ndarray).astype(int)
    has_4_cycles = False
    for matrix in [matrix_x, matrix_z]:
        overlaps = matrix @ matrix.T
        np.fill_diagonal(overlaps, 0)
        has_4_cycles |= bool(np.any(overlaps >= 2))
    assert code.has_4_cycles() == has_4_cycles
    assert code.is_mixed_distinct() == bool(np.all(matrix_x @ matrix_z.T <= 2))


def test_pair_partition_codes() -> None:
    """Pair-partition CSS codes from circulant permutation matrices."""
    # C(s) has the nonzero entry of its row r in column r - s, and block columns are not interleaved
    code = codes.PairPartitionCode([[1, 2]], [[0, 1]], lift_size=3)
    assert np.array_equal(
        code.matrix_x,
        [[0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1], [0, 1, 0, 1, 0, 0]],
    )

    # the [[276, 98, 14]] code of arXiv:2607.14091 and arXiv:2609.30069
    code = codes.PairPartitionCode(EXPONENTS_X_276, EXPONENTS_Z_276, lift_size=23)
    assert (code.num_qubits, code.dimension) == (276, 98)
    assert not code.has_4_cycles()
    assert code.is_mixed_distinct()
    assert_pair_partition_cycle_properties(code)

    # a code with (J, L, P) = (2, 4, 5), which has 4-cycles and is not mixed-distinct
    exponents_x = [[4, 3, 2, 1], [1, 0, 0, 4]]
    exponents_z = [[0, 4, 4, 3], [0, 4, 3, 2]]
    code = codes.PairPartitionCode(exponents_x, exponents_z, lift_size=5)
    assert code.has_4_cycles()
    assert not code.is_mixed_distinct()
    assert_pair_partition_cycle_properties(code)
    assert "lift size 5" in str(code)

    # translating every block column by the same cyclic shift preserves the checks of each type,
    # and moves each qubit through every position of its block column
    (translation,) = code._get_qudit_automorphisms()
    shifts = (translation.reshape(4, 5) - np.arange(5)) % 5
    assert np.all(shifts == shifts[0, 0]) and shifts[0, 0] != 0
    assert np.all(translation.reshape(4, 5) // 5 == np.arange(4)[:, None])
    for matrix in [code.matrix_x, code.matrix_z]:
        permuted = np.zeros_like(matrix)
        permuted[:, translation] = matrix
        assert {row.tobytes() for row in permuted} == {row.tobytes() for row in matrix}

    # exponents are stored modulo the lift size
    shifted = codes.PairPartitionCode(np.array(exponents_x) - 5, exponents_z, lift_size=5)
    assert np.array_equal(shifted.exponents_x, exponents_x)
    assert shifted == code

    # skip_validation builds a subsystem code from exponents that violate CSS orthogonality; here,
    # only one of the two Tanner graphs has a 4-cycle, from a single repeated difference
    for exponents_x, exponents_z in [
        ([[0, 0, 0], [0, 1, 1]], [[0, 0, 0], [0, 1, 2]]),
        ([[0, 0, 0], [0, 1, 2]], [[0, 0, 0], [0, 1, 1]]),
    ]:
        code = codes.PairPartitionCode(exponents_x, exponents_z, lift_size=3, skip_validation=True)
        assert code.is_subsystem_code
        assert code.has_4_cycles()
        assert_pair_partition_cycle_properties(code)


def test_pair_partition_code_validation() -> None:
    """Invalid inputs to the PairPartitionCode constructor are rejected."""
    with pytest.raises(ValueError, match="must be positive"):
        codes.PairPartitionCode([[0, 0]], [[0, 0]], lift_size=0)
    for exponents in [[0, 0], np.zeros((0, 2), dtype=int), [[0, 0.5]]]:
        with pytest.raises(ValueError, match="nonempty two-dimensional array of integers"):
            codes.PairPartitionCode(exponents, [[0, 0]], lift_size=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="same shape"):
        codes.PairPartitionCode([[0, 0]], [[0, 0], [0, 0]], lift_size=3)
    with pytest.raises(ValueError, match="CSS orthogonality"):
        codes.PairPartitionCode([[0, 0]], [[0, 1]], lift_size=3)


def test_pair_partition_from_array(pytestconfig: pytest.Config) -> None:
    """Building a pair-partition code from a pair-partition array by solving its constraints."""
    matchings = [
        [[[0, 1], [2, 3]], [[0, 2], [1, 3]]],
        [[[0, 3], [1, 2]], [[0, 1], [2, 3]]],
    ]

    def is_solution(exponents: npt.NDArray[np.int_], lift_size: int) -> npt.NDArray[np.bool_]:
        """Which stacked exponent arrays satisfy the paired-difference equations of matchings?"""
        satisfied = np.ones(len(exponents), dtype=bool)
        for i, j in itertools.product(range(2), repeat=2):
            diffs = exponents[:, 1, j] - exponents[:, 0, i]
            for uu, vv in matchings[i][j]:
                satisfied &= (diffs[:, uu] - diffs[:, vv]) % lift_size == 0
        return satisfied

    # the basis spans exactly the solutions that brute force finds for a lift size of 2
    basis = codes.PairPartitionCode.solve_pair_partition_array(matchings, 2)
    vectors = np.array(list(itertools.product(range(2), repeat=16))).reshape(-1, 2, 2, 4)
    assert np.all(is_solution(np.array(basis, dtype=int), 2))
    assert np.linalg.matrix_rank(basis.reshape(len(basis), -1)) == len(basis)
    assert 2 ** len(basis) == np.count_nonzero(is_solution(vectors, 2))

    # for a lift size of 5, the signs of the paired-difference equations matter
    lift_size = 5
    basis = codes.PairPartitionCode.solve_pair_partition_array(matchings, lift_size)
    assert np.all(is_solution(np.array(basis, dtype=int), lift_size))

    # members of the family are chosen by their coefficients in the basis, or at random
    coefficients = list(range(len(basis)))
    code = codes.PairPartitionCode.from_pair_partition_array(
        matchings, lift_size, coefficients=coefficients
    )
    exponents_x, exponents_z = np.tensordot(coefficients, np.array(basis, dtype=int), axes=1)
    assert np.array_equal(code.exponents_x, exponents_x % lift_size)
    assert np.array_equal(code.exponents_z, exponents_z % lift_size)
    seed = pytestconfig.getoption("randomly_seed")
    code = codes.PairPartitionCode.from_pair_partition_array(matchings, lift_size, seed=seed)
    assert code == codes.PairPartitionCode.from_pair_partition_array(
        matchings, lift_size, seed=seed
    )

    # a family can consist of a single member, with all exponents equal to zero
    code = codes.PairPartitionCode.from_pair_partition_array(
        [[[[0, 1]]]], lift_size, fix_gauge=True
    )
    assert not np.any(code.exponents_x) and not np.any(code.exponents_z)

    with pytest.raises(ValueError, match="coefficients"):
        codes.PairPartitionCode.from_pair_partition_array(matchings, lift_size, coefficients=[1])
    with pytest.raises(ValueError, match="must be prime"):
        codes.PairPartitionCode.solve_pair_partition_array(matchings, lift_size=4)

    # the array must be nonempty and square, and each entry must be a perfect matching of range(L)
    with pytest.raises(ValueError, match="nonempty and square"):
        codes.PairPartitionCode.solve_pair_partition_array([[[[0, 1]]], [[[0, 1]]]], lift_size=5)
    for bad_matching in [[], [[0, 2]], [[0, 1, 2], [3]]]:
        with pytest.raises(ValueError, match="perfect matching"):
            codes.PairPartitionCode.solve_pair_partition_array([[bad_matching]], lift_size=5)


def test_pair_partition_from_published_array() -> None:
    """The published [[276, 98, 14]] code belongs to the family of its pair-partition array."""
    lift_size = 23
    exponents = galois.GF(lift_size)([EXPONENTS_X_276, EXPONENTS_Z_276])

    # fixing the gauge removes the 2 * J + L - 1 trivial directions from the family
    basis = codes.PairPartitionCode.solve_pair_partition_array(PAIR_PARTITION_ARRAY_276, lift_size)
    fixed_basis = codes.PairPartitionCode.solve_pair_partition_array(
        PAIR_PARTITION_ARRAY_276, lift_size, fix_gauge=True
    )
    assert len(basis) - len(fixed_basis) == 2 * 4 + 12 - 1

    # the published exponents satisfy the gauge-fixing conditions, so they are in the fixed family
    vectors = np.vstack([fixed_basis, exponents[None]]).reshape(len(fixed_basis) + 1, -1)
    assert np.linalg.matrix_rank(vectors) == len(fixed_basis)
