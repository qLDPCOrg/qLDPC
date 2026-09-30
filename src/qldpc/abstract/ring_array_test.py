# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ring_array.py."""

from __future__ import annotations

import galois
import numpy as np
import pytest

from qldpc import abstract


def test_printing() -> None:
    """Convert a RingArray into a human-readable string."""
    group = abstract.DihedralGroup(6)
    ring = abstract.GroupRing(group, 3)
    one = ring.one
    x, y = ring.generators
    vec = [one + y * x**2 * y, x + y]
    ring_array = abstract.RingArray.build(vec)
    assert str(ring_array) == "[1 + y x^2 y, x + y]"


def test_ring_array(pytestconfig: pytest.Config) -> None:
    """Construct and lift a RingArray."""
    seed = pytestconfig.getoption("randomly_seed")
    np.random.seed(seed)

    int_matrix = np.random.randint(2, size=(3, 3))
    matrix = abstract.RingArray.build(int_matrix)
    assert matrix.group.equiv(abstract.TrivialGroup())
    assert np.array_equal(matrix.lift(), int_matrix)
    assert np.array_equal(
        (matrix @ matrix).lift(),
        matrix.lift() @ matrix.lift(),
    )
    assert isinstance(np.kron(matrix, matrix), abstract.RingArray)

    # infer base ring automagically
    ring = abstract.GroupRing(abstract.TrivialGroup())
    assert np.array_equal(
        abstract.RingArray.build([1, ring.one]),
        abstract.RingArray([ring.one, ring.one]),
    )

    # fail to construct a valid ring array
    rings = [abstract.GroupRing(abstract.TrivialGroup(), field) for field in [2, 3]]
    with pytest.raises(TypeError, match="must be RingMember-valued"):
        abstract.RingArray([[0]])
    with pytest.raises(ValueError, match="Cannot determine the underlying ring"):
        abstract.RingArray([])
    with pytest.raises(ValueError, match="Inconsistent rings"):
        abstract.RingArray([ring.one for ring in rings])
    with pytest.raises(ValueError, match="Inconsistent rings"):
        abstract.RingArray.build([ring.one for ring in rings])

    new_matrix = abstract.RingArray.build([[1]], abstract.CyclicGroup(1))
    with pytest.raises(ValueError, match="different base rings"):
        matrix @ new_matrix
    with pytest.raises(ValueError, match="different base rings"):
        np.kron(matrix, new_matrix)

    # np.concatenate passes its arrays inside a *sequence* argument; ring mixing must still be
    # caught, and a same-ring concatenation must preserve the RingArray type and its base ring.
    one_c1 = abstract.RingArray.build([[1]], abstract.CyclicGroup(1))
    joined = np.concatenate([one_c1, new_matrix], 0)  # positional axis: a scalar arg survives too
    assert isinstance(joined, abstract.RingArray)
    assert joined.ring == new_matrix.ring
    with pytest.raises(ValueError, match="different base rings"):
        np.concatenate([one_c1, abstract.RingArray.build([[1]], abstract.CyclicGroup(2))])
    out_c2 = abstract.RingArray.build([[0], [0]], abstract.CyclicGroup(2))
    with pytest.raises(ValueError, match="different base rings"):
        np.concatenate([one_c1, new_matrix], out=out_c2)


def test_ring_array_explicit_ring_compatibility() -> None:
    """Explicit target rings accept valid embeddings and reject incompatible re-homing."""
    source_ring = abstract.GroupRing(abstract.DihedralGroup(3), field=2)
    bimodule_ring = abstract.GroupRing(source_ring.group * source_ring.group, field=2)
    matrix = abstract.RingArray.build([[source_ring.generators[0]]], bimodule_ring)
    assert matrix.ring == bimodule_ring
    assert all(member in bimodule_ring.group for _, member in matrix[0, 0])

    different_field = abstract.GroupRing(source_ring.group, field=3)
    with pytest.raises(ValueError, match="incompatible coefficient fields"):
        abstract.RingArray.build([[source_ring.one]], different_field)

    incompatible_group = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    with pytest.raises(ValueError, match="group support is not contained"):
        abstract.RingArray.build([[source_ring.generators[0]]], incompatible_group)

    cyclic_ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    malformed_member = abstract.RingMember(cyclic_ring, incompatible_group.group.generators[0])
    with pytest.raises(ValueError, match="group support is not contained"):
        abstract.RingArray.build([[malformed_member]], cyclic_ring)


def test_empty_lift() -> None:
    """Lifting 0-sized RingArrays still yields arrays of the correct shape."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    empty = abstract.RingArray.build(np.zeros((0, 2), dtype=int), ring)
    assert empty.regular_lift().shape == (0, 2 * ring.group.order)
    assert empty.lift().shape == (0, 2 * ring.group.lift_dim)


def test_transpose() -> None:
    """Transpose a RingArray."""
    group = abstract.CyclicGroup(4)
    x0, x1, x2, x3 = group.generate()
    matrix = abstract.RingArray.build([[x0, 0, x1], [x2, 0, abstract.RingMember(group, x3)]])
    assert np.array_equal(matrix.T.T, matrix)


@pytest.mark.parametrize(
    "ring",
    [
        abstract.GroupRing(abstract.DihedralGroup(3), field=2),
        abstract.GroupRing(abstract.AbelianGroup(2, 3), field=4),
    ],
)
def test_regular_rep(ring: abstract.GroupRing, pytestconfig: pytest.Config) -> None:
    """The regular representation enables straightforward linear algebra over group algebras."""
    seed = pytestconfig.getoption("randomly_seed")
    dense_vector = ring.field.Random(4 * ring.group.order, seed=seed)
    dense_array = ring.field.Random((3, 4, ring.group.order), seed=seed + 1)

    vector = abstract.RingArray.from_field_vector(dense_vector, ring)
    matrix = abstract.RingArray.from_field_array(dense_array, ring)
    assert np.array_equal(dense_vector, abstract.RingArray.to_field_vector(vector))
    assert np.array_equal(dense_array, abstract.RingArray.to_field_array(matrix))
    assert np.array_equal(
        (matrix @ vector).to_field_vector(),
        matrix.regular_lift() @ vector.to_field_vector(),
    )

    assert not np.any(matrix @ matrix.null_space().T)
    assert not np.any(matrix.regular_lift() @ matrix.null_space().regular_lift().T)
    assert not np.any(matrix.regular_lift() @ matrix.regular_lift().null_space().T)


def test_ring_row_reduction(
    ring_dihedral3_gf5: abstract.GroupRing, pytestconfig: pytest.Config
) -> None:
    """RingArrays can be row reduced in various ways."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))
    matrix: list[list[int | abstract.RingMember]] | abstract.RingArray

    # we can row-reduce a RingArray over a cyclic group algebra
    ring = abstract.GroupRing(abstract.CyclicGroup(5), field=3)
    one = ring.one
    gen = ring.generators[0]
    gen_inverse = gen.inverse()
    assert gen_inverse is not None

    matrix = abstract.RingArray.build(
        [
            [one + gen, 0, gen],
            [gen + gen**2, 0, gen**2],
            [0, 0, one + gen],
        ]
    )
    matrix_row_reduced = abstract.RingArray.build([[1, 0, 0], [0, 0, 1], [0, 0, 0]], ring)
    matrix_hnf = matrix_row_reduced[:2, :]  # without the all-zero row
    assert np.array_equal(matrix.row_reduce(), matrix_row_reduced)
    assert np.array_equal(matrix.howell_normal_form(), matrix_hnf)
    assert np.array_equal(matrix.howell_normal_form(poly=True), matrix_hnf)

    # matrix components of non-commutative rings get "standardized" to place pivots on the diagonal
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

    # RingArray.row_reduce requires semisimple rings
    ring = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    with pytest.raises(ValueError, match="only supports semisimple rings"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).row_reduce()

    # the ordinary Howell normal form requires a semisimple ring
    ring = abstract.GroupRing(abstract.AbelianGroup(2, 2), field=2)
    with pytest.raises(ValueError, match="requires the base ring to be semisimple"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).howell_normal_form()

    # the "polynomial" Howell normal form requires an underlying cyclic group
    with pytest.raises(TypeError, match="requires an underlying CyclicGroup"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).howell_normal_form(poly=True)

    # computing a reduced Groebner basis is the final boss
    ring = abstract.GroupRing(abstract.DihedralGroup(2), field=2)
    with pytest.raises(NotImplementedError, match="Here be dragons"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).reduced_groebner_basis()


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


def test_howell_form_poly_preserves_module() -> None:
    """The polynomial Howell normal form spans the same module and is canonical (idempotent).

    Over a cyclic group algebra F[x]/(x^n - 1), the rows of a RingArray span a left module.  A basis
    of that module over F is given by all cyclic shifts of all row entries, so two RingArrays span
    the same module exactly when their shift matrices have equal row spaces.  Reducing a pivot must
    preserve the module (not merely its dimension), and reducing twice must be a no-op.
    """

    def shift_matrix(array: abstract.RingArray) -> galois.FieldArray:
        """The F-vectors spanning the row module: every cyclic shift of every row entry."""
        field_array = array.to_field_array()
        rows, cols, order = field_array.shape
        shifts = [
            np.roll(field_array[row], shift, axis=-1).reshape(-1)
            for row in range(rows)
            for shift in range(order)
        ]
        data = np.array(shifts, dtype=int) if shifts else np.zeros((0, cols * order), dtype=int)
        return array.field(data % array.field.order)

    def spans_same_module(one: abstract.RingArray, two: abstract.RingArray) -> bool:
        rows_one, rows_two = shift_matrix(one), shift_matrix(two)
        both = np.vstack([rows_one, rows_two]).view(one.field)
        rank_one = np.linalg.matrix_rank(rows_one)
        rank_two = np.linalg.matrix_rank(rows_two)
        return rank_one == rank_two == np.linalg.matrix_rank(both)

    # a row whose pivot needs a non-unit reduction, coupled to a unit in a later column: naively
    # scaling the whole row by that non-unit would shrink the module (its dimension drops 5 -> 4).
    ring = abstract.GroupRing(abstract.CyclicGroup(5), field=2)
    coupled = abstract.RingArray.from_field_array(
        ring.field([[[1, 0, 1, 0, 0], [1, 0, 0, 0, 0]]]),
        ring,  # entries 1 + x^2 and 1
    )
    reduced = coupled.howell_normal_form_poly()
    assert spans_same_module(reduced, coupled)
    assert np.array_equal(reduced, reduced.howell_normal_form_poly())

    # the same invariants across several cyclic rings and fields, on random RingArrays
    rng = np.random.default_rng(0)
    for order, characteristic in [(3, 2), (4, 2), (6, 2), (4, 3), (5, 5)]:
        ring = abstract.GroupRing(abstract.CyclicGroup(order), field=characteristic)
        for _ in range(25):
            shape = (rng.integers(1, 5), rng.integers(1, 5), order)
            coeffs = rng.integers(0, characteristic, size=shape)
            array = abstract.RingArray.from_field_array(ring.field(coeffs), ring)
            howell = array.howell_normal_form_poly()
            assert spans_same_module(array, howell)
            assert np.array_equal(howell, howell.howell_normal_form_poly())


def test_deprecations() -> None:
    """Deprecated call signatures emit DeprecationWarning."""
    ring = abstract.GroupRing(abstract.TrivialGroup())

    vector = ring.field.Random(2 * ring.group.order)
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        ring_array = abstract.RingArray.from_field_vector(ring, vector)  # type:ignore[arg-type]
    assert np.array_equal(ring_array.to_field_vector(), vector)

    matrix = ring.field.Random((1, 2, ring.group.order))
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        ring_array = abstract.RingArray.from_field_array(ring, matrix)  # type:ignore[arg-type]
    assert np.array_equal(ring_array.to_field_array(), matrix)

    # the Protograph alias warns on use, and refers to RingArray itself
    with pytest.warns(DeprecationWarning, match="Protograph is deprecated; use RingArray"):
        assert abstract.Protograph is abstract.RingArray
    with pytest.warns(DeprecationWarning, match="Protograph is deprecated; use RingArray"):
        assert abstract.ring_array.Protograph is abstract.RingArray
