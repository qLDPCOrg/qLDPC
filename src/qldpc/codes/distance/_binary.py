# SPDX-License-Identifier: Apache-2.0

"""Binary input checks, products, nested bases, stabilizers, and the symplectic map for searches.

The generic bit-packed linear algebra over GF(2) that the searches use is in ``qldpc._binary``.
"""

from __future__ import annotations

import numbers
import operator
from typing import SupportsIndex

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse

from qldpc._binary import (
    get_independent_rows,
    get_kernel,
    multiply_packed,
    pack_bits,
    row_reduce_binary,
    unpack_bits,
)

# Products with at least this many rows and columns use the Method of Four Russians.  Its tables pay
# off only over many rows, and each of its additions covers only as many columns as the product has.
# If either dimension is smaller, float32 BLAS through Apple's Accelerate can be several times
# faster.  Otherwise, the Method of Four Russians is usually faster under both Accelerate and
# OpenBLAS, by several times under OpenBLAS.
_MIN_FOUR_RUSSIANS_SIZE = 512

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


def infer_field(vectors: object) -> type[galois.FieldArray] | None:
    """Return the finite field of a field array, or of the field arrays in nested lists or tuples.

    NumPy drops the field of field arrays in a list or tuple, so a list of rows over GF(3) with
    entries 0 and 1 would otherwise pass as a binary matrix.  The function reads only the first
    item at each level of nesting, so its cost grows only with the depth of nesting, and it returns
    None if it finds no field.
    """
    while not isinstance(vectors, galois.FieldArray):
        if not isinstance(vectors, list | tuple) or not vectors:
            return None
        vectors = vectors[0]
    return type(vectors)


####################################################################################################
# binary matrices


def assert_binary(vectors: npt.ArrayLike, name: str) -> None:
    """Check that the given vectors are binary, and raise an error if they are not.

    The searches pack rows into bits and add them modulo two, so other input would silently give
    wrong results.  Input over another field, as ``infer_field`` finds it, raises a ``ValueError``
    even if its entries are 0 and 1, and a single object that is neither array-like nor a number,
    such as a code, raises a ``TypeError``.
    """
    field = infer_field(vectors)
    if field is not None and field.order != 2:
        raise ValueError(
            f"Exact searches only take binary input, but {name} is defined over GF({field.order})"
        )
    array = np.asarray(_as_dense(vectors))
    if array.dtype == object and array.ndim == 0:
        item = array.item()
        if not isinstance(item, numbers.Number | np.generic):
            raise TypeError(
                f"{name} must be an array of binary entries, not an object of type"
                f" {type(item).__name__}"
            )
    if array.size and not np.all((array == 0) | (array == 1)):
        raise ValueError(
            f"Exact searches only take binary input, but {name} has entries other than 0 and 1"
        )


def as_binary_matrix(
    vectors: npt.ArrayLike, name: str = "rows", *, num_cols: int | None = None
) -> npt.NDArray[np.uint8]:
    """Check that a collection of rows is binary, and return it as a two-dimensional uint8 array.

    A single vector is one row.  If ``num_cols`` is given, the matrix must have that many columns,
    except that an empty list, or a matrix of shape ``(0, 0)``, gets that width.  Otherwise, an
    empty list has no width, and raises an error.  Error messages call the argument ``name``, and
    NumPy raises its own error for rows that it cannot convert, such as rows of different lengths.
    """
    dense = _as_dense(vectors)
    assert_binary(dense, name)
    matrix = np.asarray(dense, dtype=np.uint8)
    if matrix.ndim == 1 and not matrix.size:
        if num_cols is None:
            raise ValueError(f"Cannot infer the width of {name}, which is empty")
        return np.empty((0, num_cols), dtype=np.uint8)
    matrix = np.atleast_2d(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a matrix or a single row, not a {matrix.ndim}-D array")
    if num_cols is not None and matrix.shape[1] != num_cols and matrix.shape != (0, 0):
        raise ValueError(
            f"{name} has {matrix.shape[1]} columns, which is incompatible with the width {num_cols}"
        )
    if not len(matrix):
        return np.empty((0, matrix.shape[1] if num_cols is None else num_cols), dtype=np.uint8)
    return matrix


def as_binary_matrix_pair(
    first: npt.ArrayLike, second: npt.ArrayLike, names: tuple[str, str]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Check two binary matrices with the same number of columns, and return them as uint8 arrays.

    Each is checked as by ``as_binary_matrix``, under its name in ``names``.  An empty list has no
    width, and a matrix of shape ``(0, 0)`` can take any width, so either takes the width of the
    other.  If both are empty lists, neither has a width, which raises an error.
    """
    first_shape, second_shape = np.shape(first), np.shape(second)
    if first_shape == second_shape == (0,):
        raise ValueError(
            f"Cannot infer the width of {names[0]} or {names[1]}, which are both empty"
        )
    if first_shape in ((0,), (0, 0)) and second_shape != (0,):
        second_matrix = as_binary_matrix(second, names[1])
        return as_binary_matrix(first, names[0], num_cols=second_matrix.shape[1]), second_matrix
    first_matrix = as_binary_matrix(first, names[0])
    return first_matrix, as_binary_matrix(second, names[1], num_cols=first_matrix.shape[1])


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
        and the scans of ``_brouwer_zimmermann/affine_search.py`` read labels with no columns as
        excluding nothing.
    """
    reduced_stabilizers, _, stabilizer_pivots = row_reduce_binary(stabilizers)
    stabilizer_basis = reduced_stabilizers[: len(stabilizer_pivots)]

    # Reduce the logical operators modulo the stabilizers by clearing the stabilizer pivots, which
    # leaves the rows of the reduced row echelon stabilizer basis unchanged.
    reduced, _, _ = row_reduce_binary(
        np.vstack([stabilizer_basis, logical_ops]), columns=stabilizer_pivots
    )
    logical_quotient = get_independent_rows(reduced[len(stabilizer_basis) :])

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
    """The product over GF(2) of two matrices of zeros and ones, as a uint8 array.

    A product with at least ``_MIN_FOUR_RUSSIANS_SIZE`` rows and columns adds packed rows by the
    Method of Four Russians (see ``multiply_packed``).  Any other product multiplies the matrices in
    float32 through BLAS, which is exact while their inner dimension is below ``2**24``, and faster
    than in integers.

    Raises:
        ValueError: If ``left`` has a different number of columns than ``right`` has rows.
    """
    if left.shape[1] != len(right):
        raise ValueError(
            f"Cannot multiply a matrix with {left.shape[1]} columns by one with {len(right)} rows"
        )
    if min(len(left), right.shape[1]) >= _MIN_FOUR_RUSSIANS_SIZE:
        # Pack C-ordered arrays, so that inputs in any memory layout share compiled specializations.
        packed = multiply_packed(
            pack_bits(np.ascontiguousarray(left)), pack_bits(np.ascontiguousarray(right))
        )
        return unpack_bits(packed, right.shape[1])
    assert left.shape[1] < 2**24
    # The float32 entries are exact integers, which reduce modulo two several times faster as int32
    # than as floats.
    product = left.astype(np.float32) @ right.astype(np.float32)
    return (product.astype(np.int32) & 1).astype(np.uint8)


def swap_pauli_halves(paulis: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Swap the X and Z halves of Pauli strings in ``[X|Z]`` order.

    The symplectic product of two Pauli strings is the dot product of one with the other swapped.
    """
    return np.roll(paulis, paulis.shape[1] // 2, axis=1)


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
    overlaps = multiply_binary(gauge, swap_pauli_halves(gauge).T)
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
