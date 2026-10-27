# SPDX-License-Identifier: Apache-2.0

"""Local Clifford gates of a deformed counter: their packed matrices, and their action on rows.

A gate acts on at most 64 target qubits.  ``get_local_matrices`` validates the matrix for X
supports, and returns it with the matrix for Z supports, both packed into words.  ``apply_local``
applies a packed matrix to the target columns of binary rows.
"""

from __future__ import annotations

import functools

import numpy as np
import numpy.typing as npt
from numba import njit

from .._binary import pack_bits, popcount_word, row_reduce_binary

_BinaryMatrix = npt.NDArray[np.uint8]


####################################################################################################
# local Cliffords

# Packed local matrices, in which bit j of row i is entry (i, j).  The CNOT, with its control first,
# acts by [[1, 0], [1, 1]] on X supports and by [[1, 1], [0, 1]] on Z supports.  The SWEL-preserving
# Clifford acts by [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]] on both.
CNOT = (np.array([1, 3], dtype=np.uint64), np.array([3, 2], dtype=np.uint64))
SWEL = (np.array([7, 11, 13, 14], dtype=np.uint64),) * 2


def get_local_matrices(
    matrix_x: npt.ArrayLike, size: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Packed local matrices for X and Z supports, from a cache keyed by ``matrix_x``."""
    order = getattr(type(matrix_x), "order", None)
    local = np.asarray(matrix_x)
    if local.dtype.kind not in "biuf":
        # The bytes of other dtypes, such as objects, do not identify their entries.
        local, order = _as_local_matrix(matrix_x, size), None
    return _get_cached_local_matrices(size, order, local.shape, local.dtype.str, local.tobytes())


@functools.lru_cache(maxsize=1024)
def _get_cached_local_matrices(
    size: int, order: int | None, shape: tuple[int, ...], dtype: str, data: bytes
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Validate and pack ``matrix_x`` given by its bytes, and the matrix for Z supports.

    Searches apply the same local Cliffords repeatedly, so results are cached.  The key identifies
    the input completely, including its shape and field, so invalid input never matches a cached
    entry.  Results are read-only so that callers cannot alter the cache.
    """
    if order is not None and order != 2:
        raise ValueError("Local Clifford matrices must be binary")
    matrix = _as_local_matrix(np.frombuffer(data, dtype=dtype).reshape(shape), size)
    local_x, local_z = _pack_local(matrix), _pack_local(_get_dual_local_matrix(matrix))
    local_x.setflags(write=False)
    local_z.setflags(write=False)
    return local_x, local_z


def _as_local_matrix(matrix: npt.ArrayLike, size: int) -> _BinaryMatrix:
    """Validate a binary local Clifford matrix that acts on ``size`` targets."""
    order = getattr(type(matrix), "order", None)
    local = np.asarray(matrix)
    if local.shape != (size, size):
        raise ValueError("Local Clifford matrices must be square and match the target count")
    if (order is not None and order != 2) or not np.all((local == 0) | (local == 1)):
        raise ValueError("Local Clifford matrices must be binary")
    return local.astype(np.uint8)


def _get_dual_local_matrix(matrix_x: _BinaryMatrix) -> _BinaryMatrix:
    """Return the matrix for Z supports, ``matrix_z``, with ``matrix_x @ matrix_z.T = I``."""
    size = len(matrix_x)
    # Row reduction finds T with T @ matrix_x = I, so that matrix_z = T.T.
    _, inverse, pivots = row_reduce_binary(matrix_x, np.eye(size, dtype=np.uint8))
    if len(pivots) < size:
        raise ValueError("Local Clifford matrix_x must be invertible over GF(2)")
    assert inverse is not None
    return inverse.T.copy()


def _pack_local(matrix: _BinaryMatrix) -> npt.NDArray[np.uint64]:
    """Pack each row of a local matrix into one word, whose bit j is column j."""
    # There are at most 64 targets, so each row fits in a single word.
    return pack_bits(matrix)[:, 0]


####################################################################################################
# native functions


@njit(cache=True)
def apply_local(
    rows: _BinaryMatrix, positions: npt.NDArray[np.int64], local: npt.NDArray[np.uint64]
) -> _BinaryMatrix:
    """Apply a packed local matrix to the target columns of a copy of the given rows.

    Column ``positions[i]`` of a row becomes the parity of ``local[i]`` and the word whose bit ``j``
    is the row's entry at ``positions[j]``.
    """
    result = rows.copy()
    for row in range(rows.shape[0]):
        word = np.uint64(0)
        for index in range(len(positions)):
            word |= np.uint64(rows[row, positions[index]]) << np.uint64(index)
        if word:
            for index in range(len(positions)):
                result[row, positions[index]] = popcount_word(local[index] & word) & 1
    return result
