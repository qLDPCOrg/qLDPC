# SPDX-License-Identifier: Apache-2.0

"""Tests for the local Clifford gates of deformed counters."""

from __future__ import annotations

import galois
import numpy as np
import numpy.typing as npt
import pytest

from . import local_cliffords

# The CNOT, with its control first, acts on X supports by this matrix, and on Z supports by its
# inverse transpose.  The SWEL-preserving Clifford acts by the second matrix on both.
CNOT_MATRIX = [[1, 0], [1, 1]]
SWEL_MATRIX = [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]


def test_the_named_gates_are_the_cnot_and_swel_matrices() -> None:
    """The packed constants agree with matrices that were validated and packed independently."""
    for gate, matrix in ((local_cliffords.CNOT, CNOT_MATRIX), (local_cliffords.SWEL, SWEL_MATRIX)):
        packed = local_cliffords.get_local_matrices(matrix, len(matrix))
        for constant, result in zip(gate, packed, strict=True):
            np.testing.assert_array_equal(constant, result)


@pytest.mark.parametrize("size", [3, 64])
def test_the_local_matrices_are_a_matrix_and_its_inverse_transpose(
    pytestconfig: pytest.Config, size: int
) -> None:
    """Packing puts entry (i, j) in bit j of word i, and the Z matrix is the X matrix's dual."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    field = galois.GF(2)
    matrix = rng.integers(0, 2, (size, size))
    while np.linalg.matrix_rank(field(matrix)) < size:
        matrix = rng.integers(0, 2, (size, size))

    local_x, local_z = local_cliffords.get_local_matrices(matrix, size)
    dense_x, dense_z = _unpack(local_x), _unpack(local_z)
    np.testing.assert_array_equal(dense_x, matrix)
    np.testing.assert_array_equal(dense_x @ dense_z.T % 2, np.eye(size, dtype=int))
    assert not local_x.flags.writeable and not local_z.flags.writeable


def test_the_local_matrix_may_be_any_array_of_zeros_and_ones() -> None:
    """Lists, integer, object and GF(2) arrays give the same packed matrices."""
    expected = local_cliffords.get_local_matrices(CNOT_MATRIX, 2)
    objects = np.array(CNOT_MATRIX, dtype=object)
    for matrix in (np.array(CNOT_MATRIX), objects, galois.GF(2)(CNOT_MATRIX)):
        for result, constant in zip(local_cliffords.get_local_matrices(matrix, 2), expected):
            np.testing.assert_array_equal(result, constant)


def test_invalid_local_matrices_are_rejected() -> None:
    identity = np.eye(2, dtype=np.uint8)
    # The size must match, even if the same bytes would fit another shape.
    for matrix in ([[1]], identity.reshape(1, 4)):
        with pytest.raises(ValueError, match="square"):
            local_cliffords.get_local_matrices(matrix, 2)
    with pytest.raises(ValueError, match="invertible"):
        local_cliffords.get_local_matrices(np.ones((2, 2)), 2)
    for invalid in (
        2 * identity,
        np.full((2, 2), np.nan),
        galois.GF(3)(identity),
        np.array([[2, 0], [0, 1]], dtype=object),
        galois.GF(2**100)(identity),  # an object array whose entries are 0 and 1, outside GF(2)
    ):
        with pytest.raises(ValueError, match="binary"):
            local_cliffords.get_local_matrices(invalid, 2)


@pytest.mark.parametrize("compiled", [True, False])
def test_apply_local_maps_the_target_columns_by_the_matrix(
    pytestconfig: pytest.Config, compiled: bool
) -> None:
    """The target entries of every row are multiplied by the matrix, and other columns are kept."""
    apply = local_cliffords.apply_local if compiled else local_cliffords.apply_local.py_func
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    positions = np.array([1, 4, 7], dtype=np.int64)
    matrix = np.array([[1, 1, 0], [0, 1, 1], [1, 1, 1]])
    rows = rng.integers(0, 2, (20, 9), dtype=np.uint8)
    rows[:3, positions] = 0  # rows that the gate does not touch
    original = rows.copy()

    expected = rows.copy()
    expected[:, positions] = rows[:, positions] @ matrix.T % 2
    local_x, _ = local_cliffords.get_local_matrices(matrix, 3)
    np.testing.assert_array_equal(apply(rows, positions, local_x), expected)
    np.testing.assert_array_equal(rows, original)


def _unpack(words: npt.NDArray[np.uint64]) -> npt.NDArray[np.int_]:
    """The matrix whose entry (i, j) is bit j of word i."""
    return np.array([[int(word) >> bit & 1 for bit in range(len(words))] for word in words])
