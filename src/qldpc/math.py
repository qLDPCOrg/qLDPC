"""Miscellaneous mathematical and linear algebra methods

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TypeVar, Union

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.special
import stim

DenseIntegerArray = Union[galois.FieldArray, npt.NDArray[np.int_]]
SparseIntegerArray = Union[scipy.sparse.spmatrix, scipy.sparse.sparray]
IntegerArray = Union[DenseIntegerArray, SparseIntegerArray]

DenseIntegerArrayType = TypeVar(
    "DenseIntegerArrayType",
    DenseIntegerArray,
    galois.FieldArray,
    npt.NDArray[np.int_],
)
GenericNumpyType = TypeVar("GenericNumpyType", bound=np.generic)


def op_to_string(op: npt.NDArray[np.int_]) -> stim.PauliString:
    """Convert an integer array that represents a Pauli string into a stim.PauliString.

    The (first, second) half the array indicates the support of (X, Z) Paulis.
    """
    support_xz = np.array(op, dtype=int).reshape(2, -1)
    paulis = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
    return stim.PauliString([paulis[xx, zz] for xx, zz in support_xz.T])


def string_to_op(string: stim.PauliString, num_qubits: int | None = None) -> npt.NDArray[np.int_]:
    """Convert a stim.PauliString into an integer array, inverting qldpc.math.op_to_string.

    The (first, second) half the array indicates the support of (X, Z) Paulis.
    """
    num_qubits = num_qubits or len(string)
    string *= stim.PauliString(f"I{num_qubits - 1}")
    return np.hstack(string.to_numpy()).astype(int)


def symplectic_conjugate(vectors: DenseIntegerArrayType) -> DenseIntegerArrayType:
    """Take symplectic vectors to their duals.

    The symplectic conjugate of a Pauli string swaps its X and Z support, and multiplies its X
    sector by -1, taking P = [P_x|P_z] -> [-P_z|P_x], such that the symplectic inner product between
    Pauli strings P and Q is ⟨P,Q⟩_s = P_x @ Q_z - P_z @ Q_x = symplectic_conjugate(P) @ Q.
    """
    assert vectors.shape[-1] % 2 == 0
    conjugated_vectors = vectors.copy().reshape(-1, 2, vectors.shape[-1] // 2)[:, ::-1, :]
    conjugated_vectors[:, 0, :] *= -1
    return conjugated_vectors.reshape(vectors.shape).view(type(vectors))


def symplectic_weight(vectors: npt.NDArray[np.int_]) -> int:
    """The symplectic weight of vectors.

    The symplectic weight of a Pauli string is the number of qudits that it addresses nontrivially.
    """
    assert vectors.shape[-1] % 2 == 0
    vectors_xz = vectors.reshape(-1, 2, vectors.shape[-1] // 2)
    vectors_x = np.asarray(vectors_xz[:, 0, :], dtype=int)
    vectors_z = np.asarray(vectors_xz[:, 1, :], dtype=int)
    return np.count_nonzero(vectors_x | vectors_z, axis=-1).reshape(vectors.shape[:-1])


def first_nonzero_cols(matrix: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
    """Get the first nonzero column for every row in a matrix."""
    if matrix.size == 0:
        return np.array([], dtype=int)
    assert matrix.ndim == 2
    return np.argmax(matrix.view(np.ndarray).astype(bool), axis=1)


def block_matrix(
    blocks: Sequence[Sequence[npt.NDArray[GenericNumpyType] | int | object]],
) -> npt.NDArray[GenericNumpyType]:
    """Build an integer block matrix.

    Literal 0 entries are replaced by zero matrices, and literal 1 entries are replaced by an
    identity matrix (padded below and to the right with zeros, if necessary).
    """
    assert len(set(len(row) for row in blocks)) == 1, "Inconsistent numbers of blocks in each row"

    # consistency checks
    row_sizes = np.array(
        [[bb.shape[0] if isinstance(bb, np.ndarray) else -1 for bb in row] for row in blocks]
    )
    assert all(len(set(row[row != -1])) == 1 for row in row_sizes), "Inconsistent row numbers"
    col_sizes = np.array(
        [[bb.shape[1] if isinstance(bb, np.ndarray) else -1 for bb in row] for row in blocks]
    )
    assert all(len(set(col[col != -1])) == 1 for col in col_sizes.T), "Inconsistent column numbers"
    dtypes = [block.dtype for row in blocks for block in row if hasattr(block, "dtype")]
    assert len(set(dtypes)) == 1, "Inconsistent block data types"

    # row numbers, column numbers, and data type
    row_nums = [next(size for size in row if size != -1) for row in row_sizes]
    col_nums = [next(size for size in col if size != -1) for col in col_sizes.T]
    dtype = dtypes[0]

    # initialize a zero matrix and populate blocks
    matrix = np.zeros((sum(row_nums), sum(col_nums)), dtype=dtype)
    for rr, row in enumerate(blocks):
        row_slice = slice(sum(row_nums[:rr]), sum(row_nums[: rr + 1]))
        for cc, block in enumerate(row):
            col_slice = slice(sum(col_nums[:cc]), sum(col_nums[: cc + 1]))
            if not isinstance(block, int):
                matrix[row_slice, col_slice] = block
            elif block == 1:
                matrix[row_slice, col_slice] = np.eye(row_nums[rr], col_nums[cc], dtype=dtype)
            else:
                assert block == 0, f"Unrecognized block: {block}"
    return matrix


@functools.cache
def log_choose(n: int, k: int) -> float:
    """Natural logarithm of (n choose k) = n! / ( k! * (n-k)! )."""
    return (
        scipy.special.gammaln(n + 1)
        - scipy.special.gammaln(k + 1)
        - scipy.special.gammaln(n - k + 1)
    )
