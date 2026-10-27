# SPDX-License-Identifier: Apache-2.0

"""Shared binary algebra, packed weights and array conversion for distance searches."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from llvmlite import ir
from numba import types
from numba.extending import lower_builtin, type_callable

from ._compilation import always_compiled

if TYPE_CHECKING:
    from numba.core.base import BaseContext
    from numba.core.typing.context import BaseContext as TypingContext
    from numba.core.typing.templates import Signature

_MASK55 = np.uint64(0x5555555555555555)


####################################################################################################
# binary matrices


def assert_binary(vectors: npt.ArrayLike, name: str) -> None:
    """Assert that the given vectors are binary.

    The enumeration below packs each row into the bits of ``uint64`` words and combines rows by
    bitwise XOR, so it is only correct over GF(2).  Non-binary input would otherwise be truncated
    silently and yield a wrong distance rather than an error.
    """
    order = getattr(type(vectors), "order", None)
    if order is not None and order != 2:
        raise ValueError(
            f"Distance calculations only support binary codes, but {name} is defined over"
            f" GF({order})"
        )
    array = np.asarray(vectors)
    if array.size and not np.all((array == 0) | (array == 1)):
        raise ValueError(
            f"Distance calculations only support binary codes, but {name} has entries other"
            " than 0 and 1"
        )


def as_binary_matrix(
    vectors: npt.ArrayLike, *, num_cols: int | None = None
) -> npt.NDArray[np.uint8]:
    """Convert a collection of binary rows to a two-dimensional uint8 array."""
    matrix = np.asarray(vectors, dtype=np.uint8)
    if matrix.size == 0:
        if num_cols is None:
            if matrix.ndim != 2:
                raise ValueError("Cannot infer the width of an empty binary matrix")
            num_cols = matrix.shape[1]
        return np.empty((0, num_cols), dtype=np.uint8)
    matrix = np.atleast_2d(matrix)
    if matrix.ndim != 2:
        raise ValueError("Binary generators must be a two-dimensional matrix")
    if num_cols is not None and matrix.shape[1] != num_cols:
        raise ValueError(
            f"Binary matrices have incompatible widths {num_cols} and {matrix.shape[1]}"
        )
    return matrix


def row_reduce_binary(
    matrix: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None = None,
    *,
    columns: npt.NDArray[np.int_] | None = None,
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8] | None,
    npt.NDArray[np.int_],
]:
    """Row-reduce a binary matrix over selected columns, transforming labels in parallel.

    Pivots are taken in the order of ``columns`` (by default, every column in order), each from the
    first remaining row with a one in that column, and every other row is cleared in that column.
    The inputs are not modified, and the reduced matrix and labels are uint8 arrays.
    """
    width = matrix.shape[1]
    columns = np.arange(width) if columns is None else np.asarray(columns)
    if len(columns) and (columns.min() < 0 or columns.max() >= width):
        raise IndexError("Row-reduction columns are out of range")
    tags = np.empty((len(matrix), 0), dtype=np.uint8) if labels is None else labels
    reduced, reduced_tags, pivots = row_reduce_packed(matrix, tags, columns.astype(np.int64))
    return reduced, None if labels is None else reduced_tags, pivots.astype(int)


def get_independent_rows(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Return a row-reduced basis for the row space of a binary matrix."""
    reduced, _, pivots = row_reduce_binary(matrix)
    return reduced[: len(pivots)]


def get_kernel(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Return a basis of the null space of a binary matrix, from its reduced row echelon form."""
    reduced, _, pivots = row_reduce_binary(matrix)
    free = np.setdiff1d(np.arange(matrix.shape[1]), pivots)
    basis = np.zeros((len(free), matrix.shape[1]), dtype=np.uint8)
    basis[:, free] = np.eye(len(free), dtype=np.uint8)
    basis[:, pivots] = reduced[: len(pivots), free].T
    return basis


def get_nested_code_basis(
    logical_ops: npt.NDArray[np.uint8],
    stabilizers: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Build an adapted basis for ``span(stabilizers, logical_ops) / span(stabilizers)``."""
    stabilizer_basis = get_independent_rows(stabilizers)
    stabilizer_pivots = [int(np.flatnonzero(row)[0]) for row in stabilizer_basis]
    reduced_logical_ops = logical_ops.copy()
    for row, pivot in zip(stabilizer_basis, stabilizer_pivots, strict=True):
        reduced_logical_ops[reduced_logical_ops[:, pivot] == 1] ^= row

    logical_quotient = get_independent_rows(reduced_logical_ops)
    basis = np.vstack([stabilizer_basis, logical_quotient]).astype(np.uint8, copy=False)
    labels = np.zeros((len(basis), len(logical_quotient)), dtype=np.uint8)
    labels[len(stabilizer_basis) :] = np.eye(len(logical_quotient), dtype=np.uint8)
    return basis, labels


def symplectic_to_hamming(vectors: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Map ``(X | Z)`` to ``(X | Z | X xor Z)``, doubling symplectic weight.

    This is the ``Saved_isometry`` reduction from https://arxiv.org/abs/2408.10743.
    """
    half_width = vectors.shape[1] // 2
    vectors_x = vectors[:, :half_width]
    vectors_z = vectors[:, half_width:]
    return np.hstack([vectors_x, vectors_z, vectors_x ^ vectors_z])


####################################################################################################
# weight functions


def symplectic_weight(
    words: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64] | None = None
) -> npt.NDArray[np.uint64]:
    """Count the qubits in the support of each riffled word (see ``riffle``).

    Each qubit occupies one aligned bit pair ``(2m, 2m + 1)`` of a word, so a qubit is in the
    support if either bit of its pair is set.  The result is uint64, and ``out`` doubles as scratch
    space.
    """
    out = np.right_shift(words, 1, out=out)
    out |= words
    out &= _MASK55
    return np.bitwise_count(out, out=out)


####################################################################################################
# bit-packing and array helpers


def count_trailing_zeros(val: int) -> int:
    """Returns the position of the least significant 1 in the binary representation of `val`."""
    return (val & -val).bit_length() - 1


def inplace_rowsum(arr: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Destructively compute ``arr.sum(-1)``, placing the result in the first column of ``arr``.

    When complete, the returned sum will be stored in ``arr[..., 0]``, while other entries in
    ``arr[..., 1:]`` will be left in indeterminate states.  This permits a faster sum
    implementation.
    """
    width = arr.shape[-1]
    while width > 1:
        split = width // 2
        arr[..., :split] += arr[..., width - split : width]
        width -= split

    return arr[..., 0]


def riffle(array: npt.ArrayLike) -> npt.ArrayLike:
    """'Riffle' Pauli strings, putting X and Z support bits for each qubit next to each other."""
    arr = np.asarray(array)
    num_bits = arr.shape[-1]
    assert num_bits % 2 == 0
    if arr.size == 0:
        return np.empty((0, num_bits), dtype=arr.dtype)  # nothing to riffle (e.g. no stabilizers)
    return np.reshape(arr, (-1, 2, num_bits // 2)).transpose(0, 2, 1).reshape(-1, num_bits)


####################################################################################################
# native word intrinsics


def popcount_word(value: np.uint64) -> np.int64:
    """Count the set bits of one 64-bit word.

    Compiled code calls LLVM's population-count instruction instead of this Python fallback.
    """
    return np.int64(int(value).bit_count())


def trailing_zeros_word(value: np.uint64 | int) -> np.int64:
    """Count the trailing zero bits of one nonzero 64-bit word.

    Compiled code calls LLVM's count-trailing-zeros instruction instead of this Python fallback.
    """
    return np.int64(count_trailing_zeros(int(value)))


@type_callable(popcount_word)
def _type_popcount_word(context: TypingContext) -> Callable[[types.Type], types.Type | None]:
    """Declare that compiled code may call ``popcount_word`` on a uint64 word."""

    def typer(value: types.Type) -> types.Type | None:
        return types.int64 if value == types.uint64 else None

    return typer


@lower_builtin(popcount_word, types.uint64)
def _lower_popcount_word(
    context: BaseContext, builder: ir.IRBuilder, signature: Signature, arguments: list[ir.Value]
) -> ir.Value:
    """Compile ``popcount_word`` to LLVM's population-count instruction.

    The count is at most 64, so the 64-bit result is the same as a signed integer.
    """
    return builder.ctpop(arguments[0])


@type_callable(trailing_zeros_word)
def _type_trailing_zeros_word(
    context: TypingContext,
) -> Callable[[types.Type], types.Type | None]:
    """Declare that compiled code may call ``trailing_zeros_word`` on a uint64 word."""

    def typer(value: types.Type) -> types.Type | None:
        return types.int64 if value == types.uint64 else None

    return typer


@lower_builtin(trailing_zeros_word, types.uint64)
def _lower_trailing_zeros_word(
    context: BaseContext, builder: ir.IRBuilder, signature: Signature, arguments: list[ir.Value]
) -> ir.Value:
    """Compile ``trailing_zeros_word`` to LLVM's count-trailing-zeros instruction."""
    return builder.cttz(arguments[0], ir.Constant(ir.IntType(1), 0))


####################################################################################################
# native packing and row reduction


@always_compiled
def pack_bits(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint64]:
    """Pack binary rows into uint64 words, with column j in bit j % 64 of word j // 64.

    This is the only packed layout in the distance package.  Entries must be 0 or 1.
    """
    num_rows, width = matrix.shape
    num_words = (width + 63) // 64
    packed = np.empty((num_rows, num_words), dtype=np.uint64)
    for row in range(num_rows):
        for word in range(num_words):
            # Assemble each word in a register rather than updating memory once per bit.
            value = np.uint64(0)
            start = 64 * word
            for bit in range(min(64, width - start)):
                value |= np.uint64(matrix[row, start + bit]) << np.uint64(bit)
            packed[row, word] = value
    return packed


@always_compiled
def row_reduce_packed(
    matrix: npt.NDArray[np.uint8], labels: npt.NDArray[np.uint8], columns: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.int64]]:
    """Row-reduce a binary matrix over the given columns, transforming labels in parallel.

    This is ``row_reduce_binary`` on packed copies, without its checks and defaults: every column
    must be in range, and ``columns`` and ``labels`` must be arrays, with no columns of labels if
    there are none.  Returns the reduced matrix, the transformed labels, and the pivot columns.
    Compiled code calls this function, since it takes only arrays.
    """
    rows, tags = pack_bits(matrix), pack_bits(labels)
    num_rows = len(rows)
    pivots = np.empty(min(num_rows, len(columns)), dtype=np.int64)
    rank = 0
    for column in columns:
        if rank == num_rows:
            break
        word = column // 64
        bit = np.uint64(1) << np.uint64(column % 64)
        source = rank
        while source < num_rows and not rows[source, word] & bit:
            source += 1
        if source == num_rows:
            continue
        if source != rank:
            for index in range(rows.shape[1]):
                rows[rank, index], rows[source, index] = rows[source, index], rows[rank, index]
            for index in range(tags.shape[1]):
                tags[rank, index], tags[source, index] = tags[source, index], tags[rank, index]
        for row in range(num_rows):
            if row != rank and rows[row, word] & bit:
                for index in range(rows.shape[1]):
                    rows[row, index] ^= rows[rank, index]
                for index in range(tags.shape[1]):
                    tags[row, index] ^= tags[rank, index]
        pivots[rank] = column
        rank += 1
    return _unpack_bits(rows, matrix.shape[1]), _unpack_bits(tags, labels.shape[1]), pivots[:rank]


@always_compiled
def _unpack_bits(packed: npt.NDArray[np.uint64], width: int) -> npt.NDArray[np.uint8]:
    """Inverse of ``pack_bits`` for rows with ``width`` columns."""
    matrix = np.empty((len(packed), width), dtype=np.uint8)
    one = np.uint64(1)
    for row in range(len(packed)):
        for column in range(width):
            matrix[row, column] = (packed[row, column // 64] >> np.uint64(column % 64)) & one
    return matrix
