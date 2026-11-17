# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _howell.py."""

from __future__ import annotations

import galois
import numpy as np
import pytest

from qldpc import abstract, math
from qldpc.abstract._howell import _get_block_howell_form


def test_howell_form_standardization_and_errors(ring_dihedral3_gf5: abstract.GroupRing) -> None:
    """Howell forms standardize noncommutative pivots, and require suitable rings."""
    # matrix components of noncommutative rings get "standardized" to place pivots on the diagonal
    ring = ring_dihedral3_gf5
    transformer = ring.get_transformer()
    component_transformer = next(ct for ct in transformer.transformers if ct.size == 2)
    e2_12 = component_transformer.embed(component_transformer.extended_field([[0, 1], [0, 0]]))
    e2_21 = component_transformer.embed(component_transformer.extended_field([[0, 0], [1, 0]]))
    e2_22 = component_transformer.embed(component_transformer.extended_field([[0, 0], [0, 1]]))
    assert np.array_equal(
        abstract.RingArray.build([[e2_12]]).howell_normal_form_semisimple(),
        abstract.RingArray.build([[e2_22]]),
    )
    assert np.array_equal(
        abstract.RingArray.build([[e2_21]]).howell_normal_form_semisimple(right=True),
        abstract.RingArray.build([[e2_22]]),
    )

    # the semisimple Howell normal form requires a semisimple ring
    ring = abstract.GroupRing(abstract.AbelianGroup(2, 2), field=2)
    with pytest.raises(ValueError, match="requires a semisimple base ring"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).howell_normal_form()

    # the "polynomial" Howell normal form requires an underlying cyclic group
    with pytest.raises(TypeError, match="group algebra of a CyclicGroup"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).howell_normal_form(poly=True)


def test_howell_form(ring_cyclic3_gf2: abstract.GroupRing) -> None:
    """The Howell normal form can add rows to a RingArray, and merges compatible pivots."""
    ring = ring_cyclic3_gf2
    a, b = ring.get_primitive_central_idempotents()
    matrix = abstract.RingArray.build([[a, b]])
    assert np.array_equal(
        matrix.howell_normal_form(),
        abstract.RingArray.build([[a, 0], [0, b]]),
    )

    matrix = abstract.RingArray.build([[a, b], [0, a]])
    assert np.array_equal(
        matrix.howell_normal_form(),
        abstract.RingArray.build([[a, 0], [0, 1]]),
    )


def assert_howell_form(array: abstract.RingArray, *, right: bool = False) -> None:
    """Assert that a 2-D RingArray is in echelon form, and has the Howell property.

    The rows of the array span a left module over its ring (or a right module, if ``right=True``),
    whose elements are sums of ring multiples of rows.  The Howell property says that each element
    with at least j leading zeros is in the module spanned by rows with at least j leading zeros.
    This function compares the dimensions of these spaces over the base field, for every j.
    """
    pivots = math.first_nonzero_cols(array)
    assert all(pivots[:-1] < pivots[1:]), "pivot columns are not strictly increasing"

    num_cols, dim = array.shape[1], array.ring.dim
    span = get_module_span(array, right=right)
    rank = np.linalg.matrix_rank(span)
    for col in range(num_cols + 1):
        # the elements with at least col leading zeros, and the module of such rows
        dim_with_leading_zeros = rank - np.linalg.matrix_rank(span[:, : col * dim])
        rows = abstract.RingArray(array[pivots >= col])
        dim_spanned = np.linalg.matrix_rank(get_module_span(rows, right=right)) if len(rows) else 0
        assert dim_with_leading_zeros == dim_spanned


def get_module_span(rows: abstract.RingArray, *, right: bool = False) -> galois.FieldArray:
    """Vectors over the base field that span the left (or right) module of some rows."""
    num_cols, dim = rows.shape[1], rows.ring.dim
    products = [
        [row * basis_element if right else basis_element * row for row in rows]
        for basis_element in rows.ring.basis
    ]
    vectors = abstract.RingArray.build(np.array(products, dtype=object).reshape(-1, num_cols))
    return vectors.to_field_array().reshape(-1, num_cols * dim).view(rows.field)


def assert_same_module(
    array_a: abstract.RingArray, array_b: abstract.RingArray, *, right: bool = False
) -> None:
    """Assert that the rows of two RingArrays span the same left (or right) module."""
    span_a = get_module_span(array_a, right=right)
    span_b = get_module_span(array_b, right=right)
    both = np.vstack([span_a, span_b]).view(array_a.field)
    rank_a, rank_b = np.linalg.matrix_rank(span_a), np.linalg.matrix_rank(span_b)
    assert rank_a == rank_b == np.linalg.matrix_rank(both)


def test_block_howell_form_pads_block_rows() -> None:
    """Block-Howell forms start a new block row at each pivot in a later block column.

    With blocks of size 3, the padding that completes a block row can have one or two zero rows.
    """
    field = galois.GF(5)
    zero, identity = field.Zeros((3, 3)), field.Identity(3)
    for rank in [1, 2]:
        # this block matrix is already in block-Howell form
        partial_identity = field(np.diag([1] * rank + [0] * (3 - rank)))
        blocks = field([[partial_identity, zero], [zero, identity]])
        assert np.array_equal(_get_block_howell_form(blocks), blocks)

        # a block row with fewer pivots than the size of its blocks is padded with zero rows
        assert np.array_equal(_get_block_howell_form(blocks[:1, :1]), blocks[:1, :1])


@pytest.mark.parametrize("right", [False, True])
def test_howell_form_semisimple_is_howell(
    ring_dihedral3_gf5: abstract.GroupRing, right: bool
) -> None:
    """The semisimple Howell normal form spans the same module, and has the Howell property."""
    rng = np.random.default_rng(0)
    ring = ring_dihedral3_gf5
    for _ in range(4):
        coeffs = rng.integers(0, ring.field.order, size=(2, 3, ring.dim))
        array = abstract.RingArray.from_field_array(coeffs, ring)
        howell = array.howell_normal_form_semisimple(right=right)
        assert_howell_form(howell, right=right)
        assert_same_module(array, howell, right=right)


def test_howell_form_poly_preserves_module() -> None:
    """The polynomial Howell normal form spans the same module, and is unchanged when recomputed."""
    # a row whose pivot needs a non-unit reduction, coupled to a unit in a later column: naively
    # scaling the whole row by that non-unit would shrink the module (its dimension drops 5 -> 4).
    ring = abstract.GroupRing(abstract.CyclicGroup(5), field=2)
    coupled = abstract.RingArray.from_field_array(
        ring.field([[[1, 0, 1, 0, 0], [1, 0, 0, 0, 0]]]),
        ring,  # entries 1 + x^2 and 1
    )
    reduced = coupled.howell_normal_form_poly()
    assert_same_module(reduced, coupled)
    assert np.array_equal(reduced, reduced.howell_normal_form_poly())

    # the same invariants across several cyclic rings and fields, on random RingArrays
    rng = np.random.default_rng(0)
    for order, characteristic in [(3, 2), (4, 2), (6, 2), (4, 3), (5, 5)]:
        ring = abstract.GroupRing(abstract.CyclicGroup(order), field=characteristic)
        for _ in range(8):
            shape = (rng.integers(1, 5), rng.integers(1, 5), order)
            coeffs = rng.integers(0, characteristic, size=shape)
            array = abstract.RingArray.from_field_array(ring.field(coeffs), ring)
            howell = array.howell_normal_form_poly()
            assert_same_module(array, howell)
            assert np.array_equal(howell, howell.howell_normal_form_poly())
            assert_howell_form(howell)

            # entries above a pivot are remainders modulo the pivot
            coefficients = howell.to_field_array()
            for row, col in enumerate(math.first_nonzero_cols(howell)):
                pivot = galois.Poly(coefficients[row, col, ::-1])
                for other_row in range(row):
                    entry = galois.Poly(coefficients[other_row, col, ::-1])
                    assert entry.degree < pivot.degree or entry == 0


def test_howell_forms_require_group_algebra(ring_cyclic3_gf2: abstract.GroupRing) -> None:
    """Howell normal forms need a group algebra, and polynomial forms that of a cyclic group."""
    ring = abstract.MatrixRing(np.array([[0, 1], [1, 0]]))
    matrix = abstract.RingArray.build([[1, 0], [1, 1]], ring)
    with pytest.raises(TypeError, match="group algebra of a CyclicGroup"):
        matrix.howell_normal_form(poly=True)
    with pytest.raises(NotImplementedError, match="only for group algebras"):
        matrix.howell_normal_form()
    with pytest.raises(NotImplementedError, match="only for group algebras"):
        matrix.howell_normal_form_semisimple(ring_cyclic3_gf2.get_transformer())
