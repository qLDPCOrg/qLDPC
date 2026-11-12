# SPDX-License-Identifier: Apache-2.0

"""Unit tests for bit-packed linear algebra over GF(2)."""

from __future__ import annotations

import galois
import numpy as np
import numpy.typing as npt
import pytest
from llvmlite import ir
from numba import njit, types
from numba.core.registry import cpu_target

from qldpc import _binary as binary

####################################################################################################
# linear algebra of unpacked binary matrices


@pytest.mark.parametrize("height, width", [(5, 130), (0, 7), (3, 0)])
def test_row_reduction_and_rank_match_galois(height: int, width: int) -> None:
    """In the default column order, row reduction gives the reduced row echelon form."""
    matrix = np.random.default_rng(width).integers(0, 2, (height, width), dtype=np.uint8)
    if height:
        matrix[-1] = matrix[0] ^ matrix[1]
    expected = galois.GF2(matrix).row_reduce()
    reduced, labels, _ = binary.row_reduce_binary(matrix)
    np.testing.assert_array_equal(reduced, expected)
    assert labels is None
    rank = binary.get_rank_packed(binary.pack_bits(matrix))
    assert rank == np.count_nonzero(expected.any(axis=1))


@pytest.mark.parametrize("columns", ["subset", "repeated"])
def test_row_reduction_over_selected_columns(columns: str) -> None:
    """Pivots come from the selected columns, and labels undergo the same row operations."""
    rng = np.random.default_rng(65)
    matrix = rng.integers(0, 2, (7, 65), dtype=np.uint8)
    labels = np.eye(7, dtype=np.uint8)
    order = rng.permutation(65)[:30] if columns == "subset" else np.tile(np.arange(65), 2)
    original_matrix = matrix.copy()

    reduced, transform, pivots = binary.row_reduce_binary(matrix, labels, columns=order)
    assert transform is not None
    np.testing.assert_array_equal(transform.astype(int) @ matrix % 2, reduced)
    rank = len(pivots)
    assert rank == np.linalg.matrix_rank(galois.GF2(matrix[:, np.unique(order)]))
    assert len(set(pivots.tolist())) == rank and set(pivots.tolist()) <= set(order.tolist())
    np.testing.assert_array_equal(reduced[:rank][:, pivots], np.eye(rank, dtype=np.uint8))
    assert not np.any(reduced[rank:][:, order])
    np.testing.assert_array_equal(matrix, original_matrix)
    np.testing.assert_array_equal(labels, np.eye(7, dtype=np.uint8))


def test_row_reduction_rejects_out_of_range_columns() -> None:
    for columns in (np.array([-1]), np.array([2])):
        with pytest.raises(IndexError, match="out of range"):
            binary.row_reduce_binary(np.eye(2, dtype=np.uint8), columns=columns)


def test_get_independent_rows() -> None:
    matrix = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(binary.get_independent_rows(matrix), [[1, 0, 1], [0, 1, 1]])


@pytest.mark.parametrize("shape", [(0, 4), (4, 70)])
def test_get_kernel(shape: tuple[int, int]) -> None:
    """The kernel basis is independent, annihilated by the matrix, and of full nullity."""
    matrix = np.random.default_rng(shape[1]).integers(0, 2, shape, dtype=np.uint8)
    matrix[-1:] = matrix[:1]
    kernel = binary.get_kernel(matrix)
    rank = np.linalg.matrix_rank(galois.GF2(matrix)) if len(matrix) else 0
    assert kernel.dtype == np.uint8 and kernel.shape == (shape[1] - rank, shape[1])
    assert not np.any(matrix.astype(int) @ kernel.T % 2)
    assert np.linalg.matrix_rank(galois.GF2(kernel)) == len(kernel)


####################################################################################################
# compiled linear algebra


@pytest.mark.parametrize("width", [0, 64, 65, 130])
def test_pack_bits(width: int) -> None:
    """Column j goes to bit j % 64 of word j // 64, and unpacking inverts packing."""
    matrix = np.random.default_rng(width).integers(0, 2, (3, width), dtype=np.uint8)
    packed = binary.pack_bits(matrix)
    for row, words in zip(matrix, packed, strict=True):
        integer = sum(int(bit) << column for column, bit in enumerate(row))
        expected = [(integer >> (64 * index)) & (2**64 - 1) for index in range(len(words))]
        assert words.tolist() == expected
    np.testing.assert_array_equal(binary.unpack_bits(packed, width), matrix)


def test_get_inverse() -> None:
    """The inverse of a product of unitriangular matrices and a permutation inverts it."""
    rng = np.random.default_rng(70)
    size = 70
    lower = np.tril(rng.integers(0, 2, (size, size)), -1) + np.eye(size, dtype=int)
    upper = np.triu(rng.integers(0, 2, (size, size)), 1) + np.eye(size, dtype=int)
    matrix = (lower @ upper % 2)[rng.permutation(size)].astype(np.uint8)
    inverse = binary.get_inverse(matrix)
    assert inverse.dtype == np.uint8
    np.testing.assert_array_equal(inverse.astype(int) @ matrix % 2, np.eye(size))
    np.testing.assert_array_equal(matrix.astype(int) @ inverse % 2, np.eye(size))

    matrix[-1] = matrix[0] ^ matrix[1]
    with pytest.raises(ValueError, match="not invertible"):
        binary.get_inverse(matrix)
    with pytest.raises(ValueError, match="square"):
        binary.get_inverse(matrix[:-1])


####################################################################################################
# native word intrinsics


@njit
def _popcounts(values: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
    counts = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        counts[index] = binary.popcount_word(value)
    return counts


@njit
def _trailing_zeros(values: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
    counts = np.empty(len(values), dtype=np.int64)
    for index, value in enumerate(values):
        counts[index] = binary.trailing_zeros_word(value)
    return counts


def test_popcount_word() -> None:
    rng = np.random.default_rng(78)
    values = np.concatenate(
        [
            np.array([0, 1, 2**63, 2**64 - 1], dtype=np.uint64),
            rng.integers(0, 2**64, 100, np.uint64),
        ]
    )
    expected = [int(value).bit_count() for value in values]
    np.testing.assert_array_equal(_popcounts(values), expected)
    # The Python implementation calls the Python fallback, and lets coverage trace the helper.
    np.testing.assert_array_equal(_popcounts.py_func(values), expected)


def test_trailing_zeros_word() -> None:
    """Shift odd words by every amount, so the shift is the number of trailing zeros."""
    shifts = np.arange(64, dtype=np.uint64)
    odd = np.random.default_rng(79).integers(0, 2**64, 64, np.uint64) | np.uint64(1)
    values = np.left_shift(odd, shifts)
    np.testing.assert_array_equal(_trailing_zeros(values), shifts)
    np.testing.assert_array_equal(_trailing_zeros.py_func(values), shifts)


@pytest.mark.parametrize("name", ["popcount", "trailing_zeros"])
def test_word_lowering(name: str) -> None:
    """Exercise compiler hooks directly, since coverage cannot trace cached native calls."""
    typer = getattr(binary, f"_type_{name}_word")(cpu_target.typing_context)
    assert typer(types.uint64) == types.int64
    assert typer(types.int64) is None
    word = ir.IntType(64)
    function = ir.Function(ir.Module(), ir.FunctionType(word, [word]), name)
    builder = ir.IRBuilder(function.append_basic_block())
    arguments: list[ir.Value] = [function.args[0]]
    signature = types.int64(types.uint64)
    lowered = getattr(binary, f"_lower_{name}_word")(
        cpu_target.target_context, builder, signature, arguments
    )
    assert lowered.type == word
