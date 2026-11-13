# SPDX-License-Identifier: Apache-2.0

"""Binary input checks, nested bases, stabilizers, and the symplectic map for distance searches.

The generic bit-packed linear algebra over GF(2) that the searches use is in ``qldpc._binary``.
"""

from __future__ import annotations

import operator
from typing import SupportsIndex

import numpy as np
import numpy.typing as npt
import scipy.sparse

from qldpc._binary import get_independent_rows, get_kernel

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
            f"Exact distance searches and counts only support binary codes, but {name} is defined"
            f" over GF({order})"
        )
    array = np.asarray(_as_dense(vectors))
    if array.size and not np.all((array == 0) | (array == 1)):
        raise ValueError(
            f"Exact distance searches and counts only support binary codes, but {name} has entries"
            " other than 0 and 1"
        )


def as_binary_matrix(
    vectors: npt.ArrayLike, name: str = "rows", *, num_cols: int | None = None
) -> npt.NDArray[np.uint8]:
    """Convert a collection of binary rows to a two-dimensional uint8 array.

    The dimension and width checks use ``name`` in their error messages, as ``assert_binary`` does.
    If NumPy cannot convert the rows to an array, such as when their lengths differ, it raises its
    own error.
    """
    matrix = np.asarray(_as_dense(vectors), dtype=np.uint8)
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


def multiply_binary(
    left: npt.NDArray[np.uint8], right: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """The product of two binary matrices over GF(2).

    Products of matrices of zeros and ones in float32 are exact while their inner dimension is below
    ``2**24``, and faster than in integers.
    """
    assert left.shape[1] < 2**24
    return (left.astype(np.float32) @ right.astype(np.float32) % 2).astype(np.uint8)


def symplectic_to_hamming(vectors: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Map ``(X | Z)`` to ``(X | Z | X xor Z)``, doubling symplectic weight.

    This is the ``Saved_isometry`` reduction from https://arxiv.org/abs/2408.10743.
    """
    half_width = vectors.shape[1] // 2
    vectors_x = vectors[:, :half_width]
    vectors_z = vectors[:, half_width:]
    return np.hstack([vectors_x, vectors_z, vectors_x ^ vectors_z])


def _as_dense(vectors: npt.ArrayLike) -> npt.ArrayLike:
    """Convert a SciPy sparse matrix or array to a dense array, and return other input as is.

    NumPy would otherwise wrap a sparse matrix in an array of one object.
    """
    if isinstance(vectors, scipy.sparse.sparray | scipy.sparse.spmatrix):
        return vectors.toarray()
    return vectors


####################################################################################################
# stabilizers of subsystem codes


def get_css_stabilizers(
    gauge_x: npt.NDArray[np.uint8], gauge_z: npt.NDArray[np.uint8]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """The stabilizers among the X and Z gauge operators of a binary CSS subsystem code.

    The stabilizers are the gauge operators that commute with every gauge operator of the other
    type.  If all gauge generators commute, the given arrays are returned, with any redundant rows.
    Otherwise, the results are independent bases.
    """
    overlaps = multiply_binary(gauge_x, gauge_z.T)
    if not np.any(overlaps):
        return gauge_x, gauge_z
    return _get_commuting_span(gauge_x, overlaps), _get_commuting_span(gauge_z, overlaps.T)


def get_pauli_stabilizers(gauge: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """The stabilizers among the gauge operators of a binary subsystem code, in ``[X|Z]`` order.

    The stabilizers are the gauge operators that commute with every gauge operator.  If all gauge
    generators commute, the given array is returned, with any redundant rows.  Otherwise, the result
    is an independent basis.
    """
    # Swapping the X and Z halves of a Pauli turns its symplectic products into dot products.
    overlaps = multiply_binary(gauge, np.roll(gauge, gauge.shape[1] // 2, axis=1).T)
    if not np.any(overlaps):
        return gauge
    return _get_commuting_span(gauge, overlaps)


def _get_commuting_span(
    rows: npt.NDArray[np.uint8], overlaps: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """An independent basis of the combinations of rows that commute with a set of other rows.

    Entry ``(i, j)`` of ``overlaps`` is one if row ``i`` anticommutes with other row ``j``.  A
    combination ``c @ rows`` commutes with every other row exactly when ``c @ overlaps`` is zero.
    """
    return get_independent_rows(multiply_binary(get_kernel(overlaps.T), rows))
