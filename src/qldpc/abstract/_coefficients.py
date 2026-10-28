# SPDX-License-Identifier: Apache-2.0

"""Arithmetic with RingArrays by way of arrays of their coefficients in the base field of a ring."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, NamedTuple

import galois
import numpy as np
import numpy.typing as npt

from . import ring_array
from .groups import Group, GroupMember
from .rings import GroupRing, RingMember

if TYPE_CHECKING:
    from .ring_array import RingArray

# Avoid replacing sparse object storage with coefficient tensors whose estimated working set is
# larger than 64 MiB.
_MAX_MATMUL_COEFFICIENT_BYTES = 64 << 20
# Keep tiny products on the object path, where coefficient conversion costs more than it saves.
_MIN_MATMUL_COEFFICIENT_WORK = 128


####################################################################################################
# matrix arithmetic with coefficient arrays


def coefficient_matmul(
    matrix_a: npt.NDArray[np.generic],
    matrix_b: npt.NDArray[np.generic],
    *,
    ring: GroupRing,
    right: bool = False,
) -> RingArray | RingMember | None:
    """Multiply matrices over a ring by way of arrays of their coefficients.

    Return None if the product should instead be computed with RingMember arithmetic, namely if an
    operand has no coefficient representation, the operand shapes are incompatible (so that numpy
    raises its usual error), or the product is too small or too large to benefit from coefficients.

    If ``right is True``, reverse the order of multiplication in the ring (see abstract.matmul).
    """
    if not (_is_coefficient_operand(matrix_a, ring) and _is_coefficient_operand(matrix_b, ring)):
        return None
    shape = _get_matmul_shape(matrix_a.shape, matrix_b.shape)
    if shape is None or not _use_coefficient_matmul(shape, ring):
        return None

    # coefficient arrays with shapes (*batch, rows, inner, |G|) and (*batch, inner, cols, |G|)
    batch, rows, inner, cols = shape.batch, shape.rows, shape.inner, shape.cols
    order = ring.group.order
    coefficients_a = _to_coefficients(matrix_a, ring)
    coefficients_b = _to_coefficients(matrix_b, ring)
    if shape.vector_a:
        coefficients_a = coefficients_a[np.newaxis, ...]
    if shape.vector_b:
        coefficients_b = coefficients_b[..., np.newaxis, :]
    coefficients_a = np.broadcast_to(coefficients_a, (*batch, rows, inner, order)).view(ring.field)
    coefficients_b = np.broadcast_to(coefficients_b, (*batch, inner, cols, order)).view(ring.field)

    # Writing A = sum_g A_g g and B = sum_h B_h h, where A_g and B_h are matrices over the base
    # field, the product A @ B = sum_{g,h} (A_g @ B_h) (g h), or sum_{g,h} (A_g @ B_h) (h g) if
    # right is True.  Loop over whichever of {g} or {h} has fewer members that appear in A or B,
    # computing all products with a fixed A_g (or B_h) in one matrix multiplication.
    entry_axes = tuple(range(coefficients_a.ndim - 1))
    indices_a = np.flatnonzero(np.any(coefficients_a, axis=entry_axes)).tolist()
    indices_b = np.flatnonzero(np.any(coefficients_b, axis=entry_axes)).tolist()
    coefficients_ab = ring.field.Zeros((*batch, rows, cols, order))
    if len(indices_a) <= len(indices_b):
        # all B_h side by side, with shape (*batch, inner, cols * |G|)
        stacked_b = coefficients_b.reshape(*batch, inner, cols * order)
        for index_g in indices_a:
            # products[..., h] = A_g @ B_h, which contributes to the coefficient of g h (or h g)
            products = (coefficients_a[..., index_g] @ stacked_b).reshape(*batch, rows, cols, order)
            coefficients_ab[..., _get_group_action(ring.group, index_g, on_left=not right)] += (
                products
            )
    else:
        # all A_g stacked vertically, with shape (*batch, |G| * rows, inner)
        batch_axes = tuple(range(len(batch)))
        stacked_a = coefficients_a.transpose(*batch_axes, -1, -3, -2).reshape(
            *batch, order * rows, inner
        )
        for index_h in indices_b:
            # products[..., g] = A_g @ B_h, which contributes to the coefficient of g h (or h g)
            products = (stacked_a @ coefficients_b[..., index_h]).reshape(*batch, order, rows, cols)
            products = np.moveaxis(products, -3, -1).view(ring.field)
            coefficients_ab[..., _get_group_action(ring.group, index_h, on_left=right)] += products

    if shape.vector_a:
        coefficients_ab = coefficients_ab[..., 0, :, :]
    if shape.vector_b:
        coefficients_ab = coefficients_ab[..., 0, :]
    if shape.vector_a and shape.vector_b:
        return _ring_member_from_vector(coefficients_ab, ring, list(ring.group._members))
    return ring_array_from_field_array(coefficients_ab, ring)


def ring_array_to_field_array(array: RingArray, ring: GroupRing) -> galois.FieldArray:
    """Convert a RingArray into coefficient vectors that are indexed by members of a ring's group.

    The provided ring must equal the ring of the array, but it may enumerate group members in a
    different order.
    """
    vectors = ring.field.Zeros((array.size, ring.group.order))
    for index, value in enumerate(array.ravel()):
        for coefficient, member in value:
            if coefficient:
                vectors[index, ring.group.index(member)] = coefficient
    return vectors.reshape(*array.shape, ring.group.order)


def ring_array_from_field_array(coefficients: galois.FieldArray, ring: GroupRing) -> RingArray:
    """Construct a RingArray from validated coefficient vectors in the last axis of an array."""
    members = list(ring.group._members)
    vectors = coefficients.reshape(-1, ring.group.order).view(ring.field)
    values = [_ring_member_from_vector(vector, ring, members) for vector in vectors]
    result = (
        np.array(values, dtype=object).reshape(coefficients.shape[:-1]).view(ring_array.RingArray)
    )
    result._ring = ring
    return result


class _MatmulShape(NamedTuple):
    """Dimensions of a matrix product, following the conventions of numpy.matmul."""

    batch: tuple[int, ...]  # broadcast shape of the leading axes that index stacked matrices
    rows: int
    inner: int
    cols: int
    vector_a: bool  # whether the left operand is a 1-D vector
    vector_b: bool  # whether the right operand is a 1-D vector


@functools.lru_cache(maxsize=512)
def _get_group_action(group: Group, index: int, *, on_left: bool) -> npt.NDArray[np.int_]:
    """Permutation of group-member indices induced by multiplying with the member at an index.

    Denoting the group member at index ``j`` by ``h_j``, entry ``j`` of the permutation is the index
    of ``h_index * h_j`` if ``on_left is True``, and the index of ``h_j * h_index`` otherwise.
    """
    members = list(group._members)
    factor = members[index]
    products = (factor * member if on_left else member * factor for member in members)
    permutation = np.fromiter(map(group.index, products), dtype=int, count=group.order)
    permutation.flags.writeable = False
    return permutation


def _ring_member_from_vector(
    vector: galois.FieldArray, ring: GroupRing, members: list[GroupMember]
) -> RingMember:
    """Construct a RingMember from its coefficients for the given (ordered) group members."""
    ring_member = RingMember(ring)
    for index in np.flatnonzero(vector):
        ring_member._vec[members[index]] = vector[index]
    return ring_member


def _get_matmul_shape(shape_a: tuple[int, ...], shape_b: tuple[int, ...]) -> _MatmulShape | None:
    """Dimensions of a matrix product, or None if numpy.matmul would reject the operand shapes."""
    if not shape_a or not shape_b:
        return None
    vector_a = len(shape_a) == 1
    vector_b = len(shape_b) == 1
    rows = 1 if vector_a else shape_a[-2]
    inner = shape_a[-1]
    cols = 1 if vector_b else shape_b[-1]
    if inner != (shape_b[0] if vector_b else shape_b[-2]):
        return None
    try:
        batch = np.broadcast_shapes(shape_a[:-2], shape_b[:-2])
    except ValueError:
        return None
    return _MatmulShape(batch, rows, inner, cols, vector_a, vector_b)


def _use_coefficient_matmul(shape: _MatmulShape, ring: GroupRing) -> bool:
    """Is a product large enough to benefit from coefficient arrays, and small enough to fit?"""
    batch_size = int(np.prod(shape.batch))
    group_order = ring.group.order
    work = batch_size * shape.rows * shape.inner * shape.cols * group_order
    if work < _MIN_MATMUL_COEFFICIENT_WORK:
        return False
    # coefficients of both (broadcast) operands, the product, and one contribution to the product
    num_entries = shape.inner * (shape.rows + shape.cols) + 2 * shape.rows * shape.cols
    num_coefficients = batch_size * num_entries * group_order
    item_size = np.dtype(ring.field.dtypes[0]).itemsize
    return num_coefficients * item_size <= _MAX_MATMUL_COEFFICIENT_BYTES


def _is_coefficient_operand(array: npt.NDArray[np.generic], ring: GroupRing) -> bool:
    """Does a matmul operand have a direct representation by coefficients in the given ring?

    Arrays over other finite fields are excluded, since their integer storage need not encode
    integers.
    """
    if isinstance(array, (ring_array.RingArray, ring.field)):
        return True
    return type(array) is np.ndarray and (
        np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_)
    )


def _to_coefficients(array: npt.NDArray[np.generic], ring: GroupRing) -> galois.FieldArray:
    """Represent a matmul operand by a coefficient vector for each entry."""
    if isinstance(array, ring_array.RingArray):
        return ring_array_to_field_array(array, ring)
    if isinstance(array, ring.field):
        scalars = array
    else:
        # interpret integers exactly as RingMember arithmetic does
        values, inverse = np.unique(array, return_inverse=True)
        values_in_field = ring.field([ring._eval_int(int(value)) for value in values])
        scalars = values_in_field[inverse].reshape(array.shape)
    coefficients = ring.field.Zeros((*array.shape, ring.group.order))
    coefficients[..., ring.group.index(ring.group.identity)] = scalars
    return coefficients
