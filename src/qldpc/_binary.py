# SPDX-License-Identifier: Apache-2.0

"""Bit-packed linear algebra over GF(2), shared by qLDPC's subpackages.

A packed binary matrix stores each row in a row of uint64 words: bit ``j`` of the row is bit
``j % 64`` of word ``j // 64``, and the unused high bits of the last word are zero.  Row operations
then combine 64 entries at a time by bitwise XOR.

The packing functions, the functions on packed rows, and ``get_inverse``, which takes a binary
matrix with one uint8 entry per bit, are compiled with ``always_compiled`` from
``qldpc._compilation``, so they can be called from Python and from other compiled code.  In compiled
code, ``popcount_word`` and ``trailing_zeros_word`` become single LLVM instructions.  The functions
``row_reduce_binary``, ``get_independent_rows``, and ``get_kernel`` are Python-level wrappers for
binary matrices with one uint8 entry per bit, which pack and unpack around the compiled functions.

NumPy has no linear algebra over GF(2), and ``galois`` provides it only for unpacked arrays at the
Python level.  NumPy's ``packbits`` and ``bitwise_count`` resemble ``pack_bits`` and
``popcount_word``, but numba cannot compile them, so one compiled implementation of the packed
layout serves both compiled and Python callers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from llvmlite import ir
from numba import types
from numba.extending import lower_builtin, type_callable

from qldpc._compilation import always_compiled

if TYPE_CHECKING:
    from numba.core.base import BaseContext
    from numba.core.typing.context import BaseContext as TypingContext
    from numba.core.typing.templates import Signature

####################################################################################################
# linear algebra of unpacked binary matrices


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
    The inputs are not modified.  Returns the reduced matrix and the transformed labels, as uint8
    arrays with the shapes of the inputs, and the pivot columns.
    """
    width = matrix.shape[1]
    columns = np.arange(width) if columns is None else np.asarray(columns)
    if len(columns) and (columns.min() < 0 or columns.max() >= width):
        raise IndexError("Row-reduction columns are out of range")
    tags = np.empty((len(matrix), 0), dtype=np.uint8) if labels is None else labels
    rows, packed_tags, pivots = row_reduce_packed(
        pack_bits(matrix), pack_bits(tags), columns.astype(np.int64)
    )
    reduced_tags = None if labels is None else unpack_bits(packed_tags, tags.shape[1])
    return unpack_bits(rows, width), reduced_tags, pivots.astype(int)


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


####################################################################################################
# compiled linear algebra


@always_compiled
def pack_bits(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint64]:
    """Pack binary rows into uint64 words, with column j in bit j % 64 of word j // 64.

    Entries must be 0 or 1.
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
def unpack_bits(packed: npt.NDArray[np.uint64], width: int) -> npt.NDArray[np.uint8]:
    """Unpack binary rows with ``width`` columns from uint64 words, inverting ``pack_bits``."""
    matrix = np.empty((len(packed), width), dtype=np.uint8)
    one = np.uint64(1)
    for row in range(len(packed)):
        for column in range(width):
            matrix[row, column] = (packed[row, column // 64] >> np.uint64(column % 64)) & one
    return matrix


@always_compiled
def row_reduce_packed(
    rows: npt.NDArray[np.uint64], labels: npt.NDArray[np.uint64], columns: npt.NDArray[np.int64]
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """Row-reduce packed rows over the given columns, transforming packed labels in parallel.

    This is ``row_reduce_binary`` on packed rows, without its checks and defaults: every column must
    be in range, and ``labels`` must have one row per row of ``rows``, with no words if there are no
    labels.  The inputs are not modified.  Returns the reduced rows and labels, and the pivot
    columns.  Rows after the pivot rows vanish on the given columns, so if the columns are
    ``0, 1, ..., width - 1`` in order, ``reduced[: len(pivots)]`` is the reduced row echelon form
    without zero rows.
    """
    words, tags = rows.copy(), labels.copy()
    num_rows = len(words)
    pivots = np.empty(min(num_rows, len(columns)), dtype=np.int64)
    rank = 0
    for column in columns:
        if rank == num_rows:
            break
        word = column // 64
        bit = np.uint64(1) << np.uint64(column % 64)
        source = rank
        while source < num_rows and not words[source, word] & bit:
            source += 1
        if source == num_rows:
            continue
        if source != rank:
            for index in range(words.shape[1]):
                words[rank, index], words[source, index] = words[source, index], words[rank, index]
            for index in range(tags.shape[1]):
                tags[rank, index], tags[source, index] = tags[source, index], tags[rank, index]
        for row in range(num_rows):
            if row != rank and words[row, word] & bit:
                for index in range(words.shape[1]):
                    words[row, index] ^= words[rank, index]
                for index in range(tags.shape[1]):
                    tags[row, index] ^= tags[rank, index]
        pivots[rank] = column
        rank += 1
    return words, tags, pivots[:rank]


@always_compiled
def get_rank_packed(rows: npt.NDArray[np.uint64]) -> int:
    """Return the rank of packed rows over GF(2)."""
    labels = np.empty((len(rows), 0), dtype=np.uint64)
    _, _, pivots = row_reduce_packed(rows, labels, np.arange(64 * rows.shape[1]))
    return len(pivots)


@always_compiled
def get_inverse(matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Return the inverse of a square binary matrix over GF(2), with one uint8 entry per bit.

    Raises:
        ValueError: If the matrix is not square, or not invertible over GF(2).
    """
    size = len(matrix)
    if matrix.shape[1] != size:
        raise ValueError("Only square matrices have inverses")
    identity = pack_bits(np.eye(size, dtype=np.uint8))
    _, inverse, pivots = row_reduce_packed(pack_bits(matrix), identity, np.arange(size))
    if len(pivots) < size:
        raise ValueError("The matrix is not invertible over GF(2)")
    return unpack_bits(inverse, size)


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


def count_trailing_zeros(val: int) -> int:
    """Return the position of the least significant 1 in the binary representation of ``val``."""
    return (val & -val).bit_length() - 1


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
