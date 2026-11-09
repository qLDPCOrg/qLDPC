# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ring_array.py."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import abstract
from qldpc.abstract import ring_array as ring_array_module


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
    with pytest.raises(ValueError, match="requires a 1 or 2-dimensional"):
        abstract.RingArray.build([[[1]]], rings[0]).lift()

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

    # the identity of a larger group is the identity of the target group
    assert abstract.RingArray.build([[bimodule_ring.one]], source_ring)[0, 0] == source_ring.one

    different_field = abstract.GroupRing(source_ring.group, field=3)
    with pytest.raises(ValueError, match="incompatible coefficient fields"):
        abstract.RingArray.build([[source_ring.one]], different_field)

    incompatible_group = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    with pytest.raises(ValueError, match="not a member"):
        abstract.RingArray.build([[source_ring.generators[0]]], incompatible_group)

    cyclic_ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    with pytest.raises(ValueError, match="not a member"):
        abstract.RingArray.build([[incompatible_group.group.generators[0]]], cyclic_ring)


def test_empty_ring_arrays() -> None:
    """Empty RingArrays keep their ring unless given another."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    empty = abstract.RingArray.build(np.zeros((0, 2), dtype=int), ring)
    assert abstract.RingArray(empty).ring == abstract.RingArray.build(empty).ring == ring
    other_ring = abstract.GroupRing(abstract.CyclicGroup(4), field=2)
    assert abstract.RingArray(empty, other_ring).ring == other_ring


def test_transpose() -> None:
    """Transpose a RingArray."""
    group = abstract.CyclicGroup(4)
    x0, x1, x2, x3 = group.generate()
    matrix = abstract.RingArray.build([[x0, 0, x1], [x2, 0, abstract.RingMember(group, x3)]])
    assert np.array_equal(matrix.T.T, matrix)

    # a RingArray without a base ring transposes its entries in their own rings
    ring = matrix.ring
    plain = np.asarray(abstract.RingArray.build([[x1, 1], [0, x1]], ring)).view(abstract.RingArray)
    expected = abstract.RingArray.build([[x3, 0], [1, x3]], ring)
    assert np.array_equal(np.asarray(plain.T), np.asarray(expected))


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


def test_from_field_array() -> None:
    """Any array of integer values is a valid array of coefficients."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3))
    identity = np.eye(3, dtype=int)
    for coefficients in [identity.astype(bool), identity.astype(float)]:
        matrix = abstract.RingArray.from_field_array(coefficients, ring)
        assert np.array_equal(matrix.to_field_array(), identity)

    with pytest.raises(ValueError, match="last axis"):
        abstract.RingArray.from_field_array(np.eye(4, dtype=int), ring)


def test_matmul_fallbacks() -> None:
    """Numpy multiplies RingArrays given keyword arguments, or with an unknown ring."""
    ring = abstract.GroupRing(abstract.CyclicGroup(7), field=4)
    matrix = abstract.RingArray.build(np.eye(4, dtype=int), ring)
    expected = np.asarray(matrix) @ np.asarray(matrix)
    assert np.array_equal(np.matmul(matrix, matrix, out=np.empty((4, 4), dtype=object)), expected)
    unknown_ring = np.asarray(matrix).view(abstract.RingArray)
    assert np.array_equal(unknown_ring @ unknown_ring, expected)

    # a RingArray without a base ring says so, rather than returning None
    with pytest.raises(ValueError, match="no base ring"):
        unknown_ring.ring  # noqa: B018


def test_ring_row_reduction(pytestconfig: pytest.Config) -> None:
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

    # RingArray.row_reduce requires semisimple rings
    ring = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    with pytest.raises(ValueError, match="only supports semisimple rings"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).row_reduce()

    # computing a reduced Groebner basis is the final boss
    ring = abstract.GroupRing(abstract.DihedralGroup(2), field=2)
    with pytest.raises(NotImplementedError, match="Here be dragons"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).reduced_groebner_basis()


def test_deprecated_argument_order() -> None:
    """Deprecated argument orders emit DeprecationWarning."""
    ring = abstract.GroupRing(abstract.TrivialGroup())

    vector = ring.field.Random(2 * ring.group.order)
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        # the deprecated argument order does not match the typed signature
        ring_array = abstract.RingArray.from_field_vector(ring, vector)  # type:ignore[arg-type]
    assert np.array_equal(ring_array.to_field_vector(), vector)

    matrix = ring.field.Random((1, 2, ring.group.order))
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        # the deprecated argument order does not match the typed signature
        ring_array = abstract.RingArray.from_field_array(ring, matrix)  # type:ignore[arg-type]
    assert np.array_equal(ring_array.to_field_array(), matrix)

    # the deprecated argument order also works for other rings
    other = abstract.MatrixRing(np.array([[0, 1], [1, 0]]))
    matrix = other.field([[[1, 0], [1, 1]]])
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        # the deprecated argument order does not match the typed signature
        ring_array = abstract.RingArray.from_field_array(other, matrix)  # type:ignore[arg-type]
    assert np.array_equal(ring_array.to_field_array(), matrix)
    vector = matrix.ravel()
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        # the deprecated argument order does not match the typed signature
        ring_array = abstract.RingArray.from_field_vector(other, vector)  # type:ignore[arg-type]
    assert np.array_equal(ring_array.to_field_vector(), vector)


def test_deprecated_attribute_lookup() -> None:
    """The module-level lookup of deprecated names rejects other names."""
    with pytest.raises(AttributeError, match="has no attribute 'not_a_real_export'"):
        ring_array_module.not_a_real_export  # type: ignore[attr-defined]  # noqa: B018


def test_ring_array_over_matrix_ring() -> None:
    """RingArrays over a matrix ring lift to block matrices of the matrices of their entries."""
    group = abstract.DihedralGroup(3)
    group_ring = abstract.GroupRing(group)
    mat_a, mat_b = [group_ring.regular_lift(member) for member in group.generators]
    ring = abstract.MatrixRing(mat_a, mat_b)
    a, b = ring.generators
    array = abstract.RingArray.build([[1, a + 1, 0], [b, 0, a * b]], ring)
    assert array.ring == ring and array.field is ring.field
    assert np.array_equal(abstract.RingArray([[a, b]]), abstract.RingArray.build([[a, b]], ring))
    assert abstract.RingArray(np.empty((0, 1), dtype=object), ring).ring == ring
    with pytest.raises(AttributeError, match="no base group"):
        _ = array.group

    # an array lifts to the block matrix of the matrices of its entries
    eye, zero = ring.field.Identity(6), ring.field.Zeros((6, 6))
    expected = np.block([[eye, mat_a + eye, zero], [mat_b, zero, mat_a @ mat_b]])
    assert np.array_equal(array.lift(), expected)

    # entries must be members of the ring of the array, or of an equal ring
    assert abstract.RingArray.build([[a]], abstract.MatrixRing(mat_a, mat_b))[0, 0] == a
    with pytest.raises(ValueError, match="different ring"):
        abstract.RingArray.build([[a]], abstract.MatrixRing(np.array([[0, 1], [1, 0]])))
    with pytest.raises(ValueError, match="different ring"):
        abstract.RingArray.build([[a]], group_ring)
    with pytest.raises(ValueError, match="different ring"):
        abstract.RingArray.build([[group_ring.one]], ring)
    with pytest.raises(
        TypeError, match="Group members combine only with members of group algebras"
    ):
        abstract.RingArray.build([[group.generators[0]]], ring)

    # integers are multiples of the identity, in the field of the ring
    ternary_ring = abstract.MatrixRing(np.array([[0, 1], [1, 0]]), field=3)
    integer_array = abstract.RingArray.build([[2]], ternary_ring)
    assert np.array_equal(integer_array.lift(), ternary_ring.field([[2, 0], [0, 2]]))


def test_ring_array_dimensions(pytestconfig: pytest.Config) -> None:
    """Lifts have blocks of size ``lift_dim``, and regular lifts and coefficients have ``dim``."""
    upper = np.array([[0, 1], [0, 0]])
    ring = abstract.MatrixRing(upper, upper.T)
    assert ring.dim == 4 and ring.lift_dim == 2
    upper_member, lower_member = ring.generators
    array = abstract.RingArray.build([[1, upper_member, 0], [lower_member, 0, 1]], ring)
    assert array.lift().shape == (2 * ring.lift_dim, 3 * ring.lift_dim)
    assert array.regular_lift().shape == (2 * ring.dim, 3 * ring.dim)
    empty = abstract.RingArray.build(np.zeros((0, 2), dtype=int), ring)
    assert empty.lift().shape == (0, 2 * ring.lift_dim)
    assert empty.regular_lift().shape == (0, 2 * ring.dim)

    # arrays are in correspondence with arrays of coefficients, one for each basis element
    seed = pytestconfig.getoption("randomly_seed")
    coefficients = ring.field.Random((2, 3, ring.dim), seed=seed)
    from_array = abstract.RingArray.from_field_array(coefficients, ring)
    assert np.array_equal(from_array.to_field_array(), coefficients)
    from_vector = abstract.RingArray.from_field_vector(coefficients.ravel(), ring)
    assert np.array_equal(from_vector.to_field_vector(), coefficients.ravel())
    with pytest.raises(ValueError, match="must have length 4"):
        abstract.RingArray.from_field_array(coefficients[..., :2], ring)


def test_kernel_over_matrix_rings() -> None:
    """The columns of a kernel are annihilated by an array, and span its null space."""
    upper = np.array([[0, 1], [0, 0]])
    ring_with_involution = abstract.MatrixRing(upper, upper.T)
    ring_without_involution = abstract.MatrixRing(upper.T)
    assert ring_with_involution.has_involution and not ring_without_involution.has_involution
    for ring in [ring_with_involution, ring_without_involution]:
        # the first two columns of this array form an invertible matrix, so the array maps R^3 onto
        # R^2 in either order of multiplication, and its null space is a free module of rank one
        array = abstract.RingArray.build([[1, ring.generators[-1], 0], [0, 1, 1]], ring)
        for right in [False, True]:
            kernel = array.kernel(right=right)
            assert kernel.shape[0] == array.shape[1]
            assert not np.any(abstract.matmul(array, kernel, right=right))
            vectors = kernel.transpose().to_field_array().reshape(-1, array.shape[1] * ring.dim)
            assert np.linalg.matrix_rank(vectors) == len(vectors) == ring.dim

        # the kernel of an invertible matrix has no columns
        assert abstract.RingArray.build([[1]], ring).kernel().shape == (1, 0)

    # with an involution, null-space rows are the conjugate transposes of kernel columns
    array = abstract.RingArray.build(
        [[1, ring_with_involution.generators[0]]], ring_with_involution
    )
    for right in [False, True]:
        assert np.array_equal(array.null_space(right=right), array.kernel(right=right).T)

    # kernels are defined only for matrices
    with pytest.raises(ValueError, match="2-dimensional"):
        abstract.RingArray.build([1, 0], ring_with_involution).kernel()


def test_ring_array_without_involution() -> None:
    """``null_space`` and transposes raise for a ring without an involution."""
    ring = abstract.MatrixRing(np.array([[0, 0], [1, 0]]))
    (nilpotent,) = ring.generators
    array = abstract.RingArray.build([[1, nilpotent]], ring)
    with pytest.raises(ValueError, match=r"use RingArray\.kernel"):
        array.null_space()
    with pytest.raises(ValueError, match="no involution"):
        _ = array.T

    # transposes raise even for an empty array
    with pytest.raises(ValueError, match="no involution"):
        _ = abstract.RingArray.build(np.zeros((0, 2), dtype=int), ring).T


def test_row_reduce_requires_group_algebra(ring_cyclic3_gf2: abstract.GroupRing) -> None:
    """Row reduction is implemented only for group algebras, even given a transformer."""
    ring = abstract.MatrixRing(np.array([[0, 1], [1, 0]]))
    matrix = abstract.RingArray.build([[1]], ring)
    with pytest.raises(NotImplementedError, match="only for group algebras"):
        matrix.row_reduce()
    with pytest.raises(NotImplementedError, match="only for group algebras"):
        matrix.row_reduce(ring_cyclic3_gf2.get_transformer())
