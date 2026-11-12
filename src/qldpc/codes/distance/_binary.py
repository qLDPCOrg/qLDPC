# SPDX-License-Identifier: Apache-2.0

"""Binary input checks, nested bases, packed weights, and array helpers for distance searches.

The generic bit-packed linear algebra over GF(2) that the searches use is in ``qldpc._binary``.
"""

from __future__ import annotations

import operator
from typing import SupportsIndex

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_independent_rows

_MASK55 = np.uint64(0x5555555555555555)


####################################################################################################
# input checks


def as_integer(value: SupportsIndex, name: str) -> int:
    """Return an integer argument as a Python integer, or raise a TypeError if it is not one.

    A NumPy integer becomes a Python integer, so that results are Python integers too.  A bool is
    rejected, although Python treats it as an integer.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not {value!r}")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, not {value!r}") from error


####################################################################################################
# binary matrices


def assert_binary(vectors: npt.ArrayLike, name: str) -> None:
    """Assert that the given vectors are binary.

    The distance searches pack rows into the bits of ``uint64`` words and combine them by bitwise
    XOR, so they are only correct over GF(2).  Non-binary input would otherwise be truncated
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
    vectors: npt.ArrayLike, name: str = "rows", *, num_cols: int | None = None
) -> npt.NDArray[np.uint8]:
    """Convert a collection of binary rows to a two-dimensional uint8 array.

    The dimension and width checks use ``name`` in their error messages, as ``assert_binary`` does.
    If NumPy cannot convert the rows to an array, such as when their lengths differ, it raises its
    own error.
    """
    matrix = np.asarray(vectors, dtype=np.uint8)
    if matrix.size == 0:
        if num_cols is None:
            if matrix.ndim != 2:
                raise ValueError(f"Cannot infer the width of {name}, which are empty")
            num_cols = matrix.shape[1]
        return np.empty((0, num_cols), dtype=np.uint8)
    matrix = np.atleast_2d(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"The {name} must be a two-dimensional matrix")
    if num_cols is not None and matrix.shape[1] != num_cols:
        raise ValueError(
            f"The {name} have {matrix.shape[1]} columns, which is incompatible with the width"
            f" {num_cols}"
        )
    return matrix


def get_nested_code_basis(
    logical_ops: npt.NDArray[np.uint8],
    stabilizers: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Build an adapted basis for ``span(stabilizers, logical_ops) / span(stabilizers)``.

    Returns:
        An independent basis whose first rows span the stabilizers, and the quotient labels of its
        rows: row ``i`` of the labels holds the coordinates of row ``i`` of the basis in the
        quotient, which are zero exactly for the stabilizer rows.  If the quotient is trivial, the
        labels have no columns.  Callers must handle that case first, since ``get_eligible_rows``
        and the scans read labels with no columns as excluding nothing.
    """
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


def get_eligible_rows(labels: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    """Mark the rows of a nested-code basis that are outside its excluded subcode.

    Row ``i`` of ``labels`` holds the quotient coordinates of row ``i`` of the basis (see
    ``get_nested_code_basis``), so a row is outside the excluded subcode if its labels are nonzero.
    Labels with no columns mean that nothing is excluded, so that every row is eligible.
    """
    if labels.shape[1] == 0:
        return np.ones(len(labels), dtype=bool)
    return np.any(labels, axis=1)


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
# array helpers


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
