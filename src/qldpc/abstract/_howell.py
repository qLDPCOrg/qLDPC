# SPDX-License-Identifier: Apache-2.0

"""Howell normal forms of RingArrays, by Wedderburn-Artin decomposition or by polynomial division.

The RingArray methods howell_normal_form_semisimple and howell_normal_form_poly document these
forms.  This module does not import ring_array, which builds on it: it builds RingArrays through
the class of its input.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import galois
import numpy as np

from qldpc.math import first_nonzero_cols

from .groups import CyclicGroup

if TYPE_CHECKING:
    from .ring_array import RingArray
    from .wedderburn_artin import WedderburnArtinTransformer


####################################################################################################
# Howell normal forms


def get_howell_normal_form_semisimple(
    ring_matrix: RingArray,
    transformer: WedderburnArtinTransformer | None = None,
    *,
    right: bool = False,
) -> RingArray:
    """Compute a Howell normal form of a RingArray over a semisimple ring.

    See RingArray.howell_normal_form_semisimple.
    """
    ring_matrix._require_ndim(2)
    if not ring_matrix.ring.is_semisimple:
        raise ValueError("The ordinary Howell normal form requires the base ring to be semisimple")
    transformer = transformer or ring_matrix.ring.get_transformer()
    num_components = len(transformer.transformers)

    # identify and row-reduce the components of this RingArray
    matrices = [
        _get_block_howell_form(component_transformer.project_array(ring_matrix), right=right)
        for component_transformer in transformer.transformers
    ]

    # pad zero rows to components that have fewer rows
    num_rows = max(len(matrix) for matrix in matrices)
    for mm, matrix in enumerate(matrices):
        if pad := num_rows - len(matrix):
            field = type(matrix)
            stack = [matrix, field.Zeros((pad, *matrix.shape[1:]))]
            matrices[mm] = np.concatenate(stack).view(field)

    pivot_row = 0
    pivot_col = 0
    num_rows, num_cols = matrices[0].shape[:2]
    while pivot_row < num_rows and pivot_col < num_cols - 1:
        # Identify:
        # 1. The column of the first nonzero value in the pivot_row of each component.
        # 2. The column that will contain the pivot when we recombine the components.
        pivot_rows_as_bools = [
            np.any(matrix[pivot_row].view(np.ndarray).astype(bool), axis=(1, 2))
            for matrix in matrices
        ]
        pivot_cols = first_nonzero_cols(pivot_rows_as_bools)
        pivot_col = min(pivot_cols)

        # Let π be a projector onto the components in which the pivot is nonzero.  If π != 1,
        # then (1-π) is a nontrivial annihilator of the pivot.  If, moreover, (1-π)·r is
        # nonzero, then (1-π)·r contains a "hidden" pivot in a later column.  In this case, we
        # in principle need to replace r -> π·r and add (1-π)·r as a new row to the matrix.  In
        # practice, this procedure messes up the reduced row echelon form of the matrix, so we
        # instead...
        # 1. In the (1-π) sector, insert a zero row at the pivot_row and shift down rows below.
        # 2. In the π sector, append a zero row to the matrix.
        components_with_hidden_pivots = [
            cc for cc in range(len(matrices)) if pivot_col < pivot_cols[cc] < num_cols
        ]
        if components_with_hidden_pivots:
            for cc in range(num_components):
                matrix = matrices[cc]
                size = transformer.transformers[cc].size
                field = type(matrix)
                zero_row = field.Zeros((1, num_cols, size, size))
                if cc in components_with_hidden_pivots:
                    stack = [matrix[:pivot_row], zero_row, matrix[pivot_row:]]
                else:
                    stack = [matrix, zero_row]
                matrices[cc] = np.concatenate(stack).view(field)
            num_rows += 1

        pivot_row += 1

    # remove rows that are zero in all components and return
    nonzero_rows = functools.reduce(
        np.bitwise_or,
        [np.any(matrix, axis=(1, 2, 3)) for matrix in matrices],
    )
    matrices = [matrix[nonzero_rows] for matrix in matrices]
    return transformer.recompose_array(matrices)._mark_as_hnf(transformer=transformer, right=right)


def get_howell_normal_form_poly(ring_matrix: RingArray) -> RingArray:
    """Compute a Howell normal form of a RingArray over a cyclic group algebra.

    See RingArray.howell_normal_form_poly.
    """
    ring_matrix._require_ndim(2)
    if not isinstance(ring_matrix.group, CyclicGroup):
        raise TypeError(
            "The Howell normal form induced by polynomial division requires an underlying"
            f" CyclicGroup, not {ring_matrix.group}"
        )

    # convert into 3-D, where the third dimension stores coefficients for group members
    field_array = ring_matrix.to_field_array()

    # The "modulus" of underlying polynomial ring for this RingArray: x^n - 1.  Analogous to N
    # in the ring of integers modulo N.
    modulus_poly = galois.Poly([1] + [0] * (ring_matrix.group.order - 1) + [-1], ring_matrix.field)

    def _multiply(poly: galois.Poly, vecs: galois.FieldArray) -> galois.FieldArray:
        """Multiply a member of a polynomial ring into a ring-valued matrix.

        The first argument represents a ring member by a polynomial, while the second argument
        represents a ``(vecs.ndim - 1)``-dimensional array of polynomials, such that
        ``vecs[*entry, c]`` is the coefficient of ``x**c`` in the given entry of ``vecs``.
        """
        new_vecs = vecs.Zeros(vecs.shape)
        for coeff, degree in zip(poly.nonzero_coeffs, poly.nonzero_degrees):
            new_vecs += coeff * np.roll(vecs, degree, axis=-1)
        return new_vecs

    pivot_row = 0
    pivot_col = 0
    while pivot_row < field_array.shape[0] and pivot_col < field_array.shape[1]:
        # look for a pivot in this column
        pivot_found = False
        for row in range(pivot_row, field_array.shape[0]):
            if np.any(field_array[row, pivot_col]):
                field_array[[pivot_row, row]] = field_array[[row, pivot_row]]
                pivot_found = True
                break

        if not pivot_found:
            pivot_col += 1
            continue

        # use invertible row operations to zero out all rows below at the pivot column
        for other_row in range(pivot_row + 1, field_array.shape[0]):
            aa_vec = field_array[pivot_row]
            bb_vec = field_array[other_row]
            if not np.any(bb_vec):
                continue
            # Let:
            #     aa = aa_vec[pivot_row]
            #     bb = bb_vec[other_row]
            # We will transform rows as
            #     [aa_vec, bb_vec] --> [[ss, tt], [uu, vv]] @ [aa_vec, bb_vec]
            # where
            #     (1) ss * aa + tt * bb = gcd(aa, bb) = gg
            #     (2) uu * aa + vv * bb = 0
            #     (3) det([[ss, tt], [uu, vv]]) = ss * vv - tt * uu = 1
            # Condition (3) ensures that this transformation is invertible.  Condition (2)
            # ensures that bb_vec gets zeroed out at the pivot column.
            aa_poly = galois.Poly(aa_vec[pivot_col, ::-1], field=ring_matrix.field)
            bb_poly = galois.Poly(bb_vec[pivot_col, ::-1], field=ring_matrix.field)

            # find gg, ss, tt, uu, vv; mypy sees only the integer signature of galois.egcd,
            # which also handles polynomials
            gg_poly: galois.Poly
            ss_poly: galois.Poly
            tt_poly: galois.Poly
            gg_poly, ss_poly, tt_poly = galois.egcd(aa_poly, bb_poly)  # type:ignore[assignment,arg-type]
            uu_poly = -bb_poly // gg_poly
            vv_poly = aa_poly // gg_poly

            new_aa_vec = _multiply(ss_poly, aa_vec) + _multiply(tt_poly, bb_vec)
            new_bb_vec = _multiply(uu_poly, aa_vec) + _multiply(vv_poly, bb_vec)
            field_array[pivot_row] = new_aa_vec
            field_array[other_row] = new_bb_vec

        # "Reduce" the pivot to gcd(pivot, modulus):
        #     (1) Find ff for which ff * pivot = gcd(pivot, modulus) = gg.
        #     (2) Replace the pivot row with ff * (pivot row), reducing the pivot to gg.
        # Multiplying the whole row by the non-unit ff would shrink the row-module span (it
        # scales every column, not just the pivot), so preserve the span by keeping the
        # residual: the original row minus its reconstruction quotient * (reduced row), where
        # quotient = pivot / gg (exact, and quotient * gg = pivot with no reduction modulo
        # x^n - 1 since deg(pivot) < n).  That residual is therefore zero in the pivot column
        # and, being a ring-multiple of the pivot row, adds no span; it is resolved later.
        pivot_poly = galois.Poly(field_array[pivot_row, pivot_col, ::-1], field=ring_matrix.field)
        gcd_poly: galois.Poly
        ff_poly: galois.Poly
        # mypy sees only the integer signature of galois.egcd, which also handles polynomials
        gcd_poly, ff_poly, _ = galois.egcd(pivot_poly, modulus_poly)  # type:ignore[assignment,arg-type]
        if pivot_poly != gcd_poly:
            quotient_poly = pivot_poly // gcd_poly
            reduced_row = _multiply(ff_poly, field_array[pivot_row])
            residual_row = field_array[pivot_row] - _multiply(quotient_poly, reduced_row)
            field_array[pivot_row] = reduced_row
            field_array = np.append(field_array, [residual_row], axis=0).view(ring_matrix.field)
            pivot_poly = gcd_poly

        # Reduce all rows above the pivot_row at the pivot_column.  If some value in the
        # pivot_col above the pivot_row can be written as a multiple of the pivot plus a
        # remainder, use row operations to subtract off that multiple of the pivot, leaving only
        # the remainder.
        for other_row in range(pivot_row):
            other_poly = galois.Poly(
                field_array[other_row, pivot_col, ::-1], field=ring_matrix.field
            )
            div_poly = other_poly // pivot_poly
            if div_poly != 0:
                field_array[other_row] -= _multiply(div_poly, field_array[pivot_row])

        # Check whether the pivot has a nontrivial annihilator, with annihilator * pivot = 0.
        # If a nontrivial annihilator is found, append a new row with the pivot annihilated.
        annihilator_poly = modulus_poly // pivot_poly
        if annihilator_poly != 0:
            new_row = _multiply(annihilator_poly, field_array[pivot_row])
            field_array = np.append(field_array, [new_row], axis=0).view(ring_matrix.field)

        pivot_row += 1
        pivot_col += 1

    # remove all-zero rows and return
    field_array = field_array[np.any(field_array, axis=(1, 2))]
    return (
        type(ring_matrix)
        .from_field_array(field_array, ring_matrix.ring)
        ._mark_as_hnf(transformer=None, right=False)
    )


def _get_block_howell_form(matrix: galois.FieldArray, *, right: bool = False) -> galois.FieldArray:
    """Compute a block-Howell normal form of the provided block matrix.

    The provided matrix should be 4-dimensional, with matrix[i, j] storing a square block at (i, j).
    The block-Howell form is essentially the same as the row-reduced echelon form when the matrix is
    expanded into a 2-dimensional array, except zero rows are inserted to shift pivots down so that
    they always lie on the diagonal of a block.
    """
    shape: tuple[int, ...]

    assert matrix.ndim == 4 and matrix.shape[-1] == matrix.shape[-2]
    field = type(matrix)
    num_block_rows, num_block_cols, size, _ = matrix.shape

    if right and size > 1:
        # transpose each block to turn right-side reduction into the existing left-side case
        matrix = matrix.transpose(0, 1, 3, 2)

    # row-reduce as an expanded 2-D matrix, keeping a basis of the row space (no all-zero rows)
    shape = (num_block_rows * size, num_block_cols * size)
    matrix = matrix.transpose(0, 2, 1, 3).reshape(shape).view(field).row_space()

    if size > 1:
        # insert zero rows to shift pivots down so that they always lie on the diagonal of a block
        pivot_row, pivot_col = 0, 0
        num_cols = matrix.shape[1]
        while pivot_row < matrix.shape[0]:
            pivot_col = first_nonzero_cols(matrix[pivot_row])[0]
            if pivot_row % size == 0:
                pivot_block_col = pivot_col // size
            if pad := pivot_col - pivot_block_col * size - pivot_row % size:
                zero_rows = np.zeros((pad, num_cols), dtype=int)
                matrix = np.vstack([matrix[:pivot_row], zero_rows, matrix[pivot_row:]]).view(field)
                pivot_row += pad
            pivot_row += 1

        # pad with zero rows on the bottom to ensure that all blocks have the correct size
        if tail := matrix.shape[0] % size:  # pragma: no cover
            zero_rows = np.zeros((size - tail, num_cols), dtype=int)
            matrix = np.vstack([matrix, zero_rows]).view(field)

    # re-collect into a 4-D array
    shape = (matrix.shape[0] // size, size, num_block_cols, size)
    matrix = matrix.reshape(shape).transpose(0, 2, 1, 3).view(field)

    if right and size > 1:
        matrix = matrix.transpose(0, 1, 3, 2)

    return matrix
