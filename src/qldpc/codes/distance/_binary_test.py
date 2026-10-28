# SPDX-License-Identifier: Apache-2.0

"""Unit tests for shared binary algebra, packing and weight helpers."""

from __future__ import annotations

import itertools

import galois
import numpy as np
import numpy.typing as npt
import pytest
from llvmlite import ir
from numba import njit, types
from numba.core.registry import cpu_target

import qldpc

from . import _binary as binary
from .conftest import count_set_bits

####################################################################################################
# binary matrices, weights, and array helpers


def test_binary_matrices_and_nested_bases() -> None:
    """Check matrix conversion and basis ownership independently of distance dispatch."""
    assert binary.as_binary_matrix(np.empty((0, 3))).shape == (0, 3)
    assert binary.as_binary_matrix([], num_cols=3).shape == (0, 3)
    with pytest.raises(ValueError, match="infer the width"):
        binary.as_binary_matrix([])
    with pytest.raises(ValueError, match="two-dimensional"):
        binary.as_binary_matrix(np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="incompatible widths"):
        binary.as_binary_matrix([[1, 0]], num_cols=3)

    matrix = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.uint8)
    labels = np.eye(4, dtype=np.uint8)
    reduced, transformed, pivots = binary.row_reduce_binary(matrix, labels)
    np.testing.assert_array_equal(pivots, [0, 1])
    assert transformed is not None
    np.testing.assert_array_equal(transformed @ matrix % 2, reduced)
    np.testing.assert_array_equal(labels, np.eye(4, dtype=np.uint8))
    np.testing.assert_array_equal(binary.get_independent_rows(matrix), [[1, 0, 1], [0, 1, 1]])
    basis, tags = binary.get_nested_code_basis(matrix, np.array([[1, 0, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(basis, [[1, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(tags, [[0], [1]])


@pytest.mark.parametrize("shape", [(0, 4), (4, 70)])
def test_get_kernel(shape: tuple[int, int]) -> None:
    """The kernel basis is independent, annihilated by the matrix, and of full nullity."""
    matrix = np.random.default_rng(shape[1]).integers(0, 2, shape, dtype=np.uint8)
    matrix[-1:] = matrix[:1]
    kernel = binary.get_kernel(matrix)
    rank = len(binary.get_independent_rows(matrix))
    assert kernel.dtype == np.uint8 and kernel.shape == (shape[1] - rank, shape[1])
    assert not np.any(matrix.astype(int) @ kernel.T % 2)
    assert len(binary.get_independent_rows(kernel)) == len(kernel)


def test_riffle_and_symplectic_isometry() -> None:
    symplectic = np.array(list(itertools.product((0, 1), repeat=6)), dtype=np.uint8)
    expected = np.stack((symplectic[:, :3], symplectic[:, 3:]), axis=2).reshape(-1, 6)
    np.testing.assert_array_equal(binary.riffle(symplectic), expected)
    assert np.shape(binary.riffle(np.empty((0, 6)))) == (0, 6)
    with pytest.raises(AssertionError):
        binary.riffle(np.zeros((1, 3)))
    transformed = binary.symplectic_to_hamming(symplectic)
    np.testing.assert_array_equal(
        np.count_nonzero(transformed, axis=1),
        2 * np.count_nonzero(symplectic[:, :3] | symplectic[:, 3:], axis=1),
    )


def test_symplectic_weight() -> None:
    """Validate symplectic_weight against the reference bit-counting implementation."""
    rng = np.random.default_rng(64)
    vals = rng.integers(0, 2**64, size=(7, 11), dtype=np.uint64)
    expected_weights = count_set_bits((vals | (vals >> np.uint64(1))) & 0x5555555555555555)
    weights = binary.symplectic_weight(vals)
    assert weights.dtype == np.uint64
    np.testing.assert_array_equal(weights, expected_weights)

    out = rng.integers(0, 2**64, size=vals.shape, dtype=vals.dtype)
    weights = binary.symplectic_weight(vals, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert weights is out


@pytest.mark.parametrize("width", range(1, 8))
def test_inplace_rowsum(width: int) -> None:
    """inplace_rowsum reduces each row to its sum and stores the result in the first column."""
    arr = np.random.default_rng(width).integers(2**31, size=(10, width), dtype=np.uint64)
    expected = arr.sum(-1)
    actual = binary.inplace_rowsum(arr)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, arr[:, 0])


def test_distance_requires_binary_input() -> None:
    """Distance calculations reject non-binary input rather than silently truncating it.

    The enumeration packs each row into the bits of uint64 words, so a non-binary entry would
    otherwise be reinterpreted and yield a wrong distance with no indication of a problem.
    """
    # A GF(3) generator matrix is rejected because exact distance enumeration supports only GF(2).
    ternary_code = qldpc.codes.classical.HammingCode(3, 3)
    with pytest.raises(ValueError, match=r"only support binary codes.*GF\(3\)"):
        qldpc.codes.distance.get_distance_classical(ternary_code.generator)

    # A plain integer array with a nonbinary entry is rejected by value validation.
    with pytest.raises(ValueError, match="entries other than 0 and 1"):
        qldpc.codes.distance.get_distance_classical(np.array([[1, 2, 0], [0, 1, 1]]))
    with pytest.raises(ValueError, match="entries other than 0 and 1"):
        qldpc.codes.distance.get_distance_quantum(
            np.array([[1, 1, 0, 0]]), np.array([[0, 0, 2, 0]]), homogeneous=True
        )

    # A binary generator matrix is accepted with either a plain integer dtype or a GF(2) dtype.
    assert qldpc.codes.distance.get_distance_classical(np.array([[1, 1, 1]])) == 3
    assert qldpc.codes.distance.get_distance_classical(qldpc.codes.HammingCode(4).generator) == 3


####################################################################################################
# native bit counting, packing and row reduction


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


@pytest.fixture(params=[False, True], ids=["compiled", "python"])
def binary_kernels(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the compiled packing and row-reduction kernels, or their Python versions for coverage."""
    if request.param:
        for name in ("pack_bits", "_unpack_bits", "row_reduce_packed"):
            monkeypatch.setattr(binary, name, getattr(binary, name).py_func)


@pytest.mark.usefixtures("binary_kernels")
@pytest.mark.parametrize("width", [0, 64, 65, 130])
def test_pack_bits(width: int) -> None:
    """Column j goes to bit j % 64 of word j // 64, and unpacking inverts packing."""
    matrix = np.random.default_rng(width).integers(0, 2, (3, width), dtype=np.uint8)
    packed = binary.pack_bits(matrix)
    for row, words in zip(matrix, packed, strict=True):
        integer = sum(int(bit) << column for column, bit in enumerate(row))
        expected = [(integer >> (64 * index)) & (2**64 - 1) for index in range(len(words))]
        assert words.tolist() == expected
    np.testing.assert_array_equal(binary._unpack_bits(packed, width), matrix)


@pytest.mark.usefixtures("binary_kernels")
@pytest.mark.parametrize("height, width", [(5, 130), (0, 7), (3, 0)])
def test_row_reduction_matches_galois(height: int, width: int) -> None:
    """In the default column order, row reduction gives the reduced row echelon form."""
    matrix = np.random.default_rng(width).integers(0, 2, (height, width), dtype=np.uint8)
    reduced, _, _ = binary.row_reduce_binary(matrix)
    np.testing.assert_array_equal(reduced, galois.GF(2)(matrix).row_reduce())


@pytest.mark.usefixtures("binary_kernels")
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
    assert rank == np.linalg.matrix_rank(galois.GF(2)(matrix[:, np.unique(order)]))
    assert len(set(pivots.tolist())) == rank and set(pivots.tolist()) <= set(order.tolist())
    np.testing.assert_array_equal(reduced[:rank][:, pivots], np.eye(rank, dtype=np.uint8))
    assert not np.any(reduced[rank:][:, order])
    np.testing.assert_array_equal(matrix, original_matrix)


def test_row_reduction_rejects_out_of_range_columns() -> None:
    for columns in (np.array([-1]), np.array([2])):
        with pytest.raises(IndexError, match="out of range"):
            binary.row_reduce_binary(np.eye(2, dtype=np.uint8), columns=columns)
