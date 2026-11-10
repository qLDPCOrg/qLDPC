# SPDX-License-Identifier: Apache-2.0

"""Howell normal forms of RingArrays, by Wedderburn-Artin decomposition or by polynomial division.

The RingArray methods howell_normal_form_semisimple and howell_normal_form_poly document these
forms.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import galois
import numpy as np

from qldpc.math import first_nonzero_cols

from ._coefficients import ring_array_from_field_array
from .groups import CyclicGroup
from .rings import GroupRing

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
    ring = ring_matrix.ring
    if not isinstance(ring, GroupRing):
        raise NotImplementedError(
            f"Semisimple Howell normal forms are implemented only for group algebras, not {ring}"
        )
    if not ring.is_semisimple:
        raise ValueError("The semisimple Howell normal form requires a semisimple base ring")
    transformer = transformer or ring.get_transformer()

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

    # split rows whose pivots hide later pivots in some components (see below).  A pivot in the last
    # column hides none.
    pivot_row = 0
    pivot_col = 0
    num_cols = matrices[0].shape[1]
    while pivot_row < num_rows and pivot_col < num_cols - 1:
        # Identify:
        # 1. The column of the first nonzero value in the pivot_row of each component.
        # 2. The column that will contain the pivot when we recombine the components.
        pivot_cols = first_nonzero_cols(
            [np.any(matrix[pivot_row], axis=(1, 2)) for matrix in matrices]
        )
        pivot_col = min(pivot_cols)

        # Let r be the pivot row, and π the projector onto the components in which its pivot is
        # nonzero.  If π != 1, then (1-π) is a nontrivial annihilator of the pivot.  If, moreover,
        # (1-π)·r is nonzero, then (1-π)·r contains a "hidden" pivot in a later column.  Keep π·r in
        # the pivot_row, and move (1-π)·r down a row, where it joins the next row of the other
        # components:
        # 1. In each component where r has a hidden pivot, insert a zero row at the pivot_row,
        #    shifting r and later rows down.
        # 2. In every other component, append a zero row, so that all components keep the same
        #    number of rows.
        # The components of a matrix are independent, so the rows span the same module, and each
        # component stays in echelon form.
        components_with_hidden_pivots = [
            cc for cc in range(len(matrices)) if pivot_col < pivot_cols[cc] < num_cols
        ]
        if components_with_hidden_pivots:
            for cc, matrix in enumerate(matrices):
                field = type(matrix)
                zero_row = field.Zeros((1, *matrix.shape[1:]))
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
    ring = ring_matrix.ring
    if not (isinstance(ring, GroupRing) and isinstance(ring.group, CyclicGroup)):
        raise TypeError(
            "The Howell normal form induced by polynomial division requires the group algebra of a"
            f" CyclicGroup, not {ring}"
        )

    # convert into 3-D, where the third dimension stores coefficients for group members
    field_array = ring_matrix.to_field_array()

    # The "modulus" of the underlying polynomial ring for this RingArray: x^n - 1.  Analogous to N
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
            #     aa = aa_vec[pivot_col]
            #     bb = bb_vec[pivot_col]
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

            # find gg, ss, tt, uu, vv
            gg_poly, ss_poly, tt_poly = _egcd_poly(aa_poly, bb_poly)
            uu_poly = -bb_poly // gg_poly
            vv_poly = aa_poly // gg_poly

            new_aa_vec = _multiply(ss_poly, aa_vec) + _multiply(tt_poly, bb_vec)
            new_bb_vec = _multiply(uu_poly, aa_vec) + _multiply(vv_poly, bb_vec)
            field_array[pivot_row] = new_aa_vec
            field_array[other_row] = new_bb_vec

        # "Reduce" the pivot to gcd(pivot, modulus) = gcd:
        #     (1) Find ff for which ff * pivot = gcd.
        #     (2) Replace the pivot row with ff * (pivot row), reducing the pivot to gcd.
        #     (3) Append the residual row (pivot row) - quotient * (reduced row), where
        #         quotient = pivot / gcd.
        # Step (2) alone could shrink the span of the rows, since ff need not be a unit.  The
        # residual is zero in the pivot column, and the reduced and residual rows together span the
        # original pivot row.  Later columns reduce the residual row.
        pivot_poly = galois.Poly(field_array[pivot_row, pivot_col, ::-1], field=ring_matrix.field)
        gcd_poly, ff_poly, _ = _egcd_poly(pivot_poly, modulus_poly)
        if pivot_poly != gcd_poly:
            quotient_poly = pivot_poly // gcd_poly
            reduced_row = _multiply(ff_poly, field_array[pivot_row])
            residual_row = field_array[pivot_row] - _multiply(quotient_poly, reduced_row)
            field_array[pivot_row] = reduced_row
            field_array = np.append(field_array, [residual_row], axis=0).view(ring_matrix.field)
            pivot_poly = gcd_poly

        # reduce each entry above the pivot modulo the pivot, by subtracting the pivot row times the
        # quotient of the entry by the pivot
        for other_row in range(pivot_row):
            other_poly = galois.Poly(
                field_array[other_row, pivot_col, ::-1], field=ring_matrix.field
            )
            div_poly = other_poly // pivot_poly
            if div_poly != 0:
                field_array[other_row] -= _multiply(div_poly, field_array[pivot_row])

        # The pivot divides the modulus, so a pivot that is not a unit has the nontrivial
        # annihilator modulus / pivot.  Append the pivot row times this annihilator.
        if pivot_poly.degree > 0:
            annihilator_poly = modulus_poly // pivot_poly
            new_row = _multiply(annihilator_poly, field_array[pivot_row])
            field_array = np.append(field_array, [new_row], axis=0).view(ring_matrix.field)

        pivot_row += 1
        pivot_col += 1

    # remove all-zero rows and return
    field_array = field_array[np.any(field_array, axis=(1, 2))]
    return ring_array_from_field_array(field_array, ring_matrix.ring)


def _egcd_poly(
    poly_a: galois.Poly, poly_b: galois.Poly
) -> tuple[galois.Poly, galois.Poly, galois.Poly]:
    """Return ``(gcd, s, t)`` with ``s·poly_a + t·poly_b = gcd`` (extended Euclidean algorithm)."""
    # mypy sees only the integer signature of galois.egcd, which also handles polynomials
    return galois.egcd(poly_a, poly_b)  # type:ignore[arg-type,return-value]


def _get_block_howell_form(matrix: galois.FieldArray, *, right: bool = False) -> galois.FieldArray:
    """Compute a block-Howell normal form of the provided block matrix.

    The matrix must be 4-dimensional, with ``matrix[i, j]`` the square block at ``(i, j)``.  The
    block-Howell form is the reduced row echelon form of the expanded 2-dimensional matrix, with
    zero rows inserted so that every pivot lies on the diagonal of a block, and all pivots in a
    block row lie in one block column.  If ``right=True``, the blocks are transposed before and
    after the reduction.
    """
    shape: tuple[int, ...]

    assert matrix.ndim == 4 and matrix.shape[-1] == matrix.shape[-2]
    field = type(matrix)
    num_block_rows, num_block_cols, size, _ = matrix.shape

    if right:
        # transpose each block to turn right-side reduction into the existing left-side case
        matrix = matrix.transpose(0, 1, 3, 2)

    # row-reduce as an expanded 2-D matrix, keeping a basis of the row space (no all-zero rows)
    shape = (num_block_rows * size, num_block_cols * size)
    matrix = matrix.transpose(0, 2, 1, 3).reshape(shape).view(field).row_space()

    if size > 1:
        # Insert zero rows to shift pivots down so that they always lie on the diagonal of a block.
        # All pivots in a block row lie in one block column, so a pivot in a later block column
        # starts a new block row.
        pivot_block_col = 0
        num_cols = matrix.shape[1]
        pivot_row = 0
        while pivot_row < matrix.shape[0]:
            pivot_col = first_nonzero_cols(matrix[pivot_row])[0]

            # move a pivot in a later block column to the next block row
            pad = 0
            if pivot_row % size and pivot_col // size != pivot_block_col:
                pad = size - pivot_row % size

            # record the block column of the pivots in a new block row
            if (pivot_row + pad) % size == 0:
                pivot_block_col = pivot_col // size

            # shift the pivot onto the diagonal of its block
            pad += pivot_col % size - (pivot_row + pad) % size
            if pad:
                zero_rows = np.zeros((pad, num_cols), dtype=int)
                matrix = np.vstack([matrix[:pivot_row], zero_rows, matrix[pivot_row:]]).view(field)
                pivot_row += pad
            pivot_row += 1

        # pad with zero rows on the bottom to ensure that all blocks have the correct size
        if tail := matrix.shape[0] % size:
            zero_rows = np.zeros((size - tail, num_cols), dtype=int)
            matrix = np.vstack([matrix, zero_rows]).view(field)

    # re-collect into a 4-D array
    shape = (matrix.shape[0] // size, size, num_block_cols, size)
    matrix = matrix.reshape(shape).transpose(0, 2, 1, 3).view(field)

    if right:
        matrix = matrix.transpose(0, 1, 3, 2)

    return matrix
