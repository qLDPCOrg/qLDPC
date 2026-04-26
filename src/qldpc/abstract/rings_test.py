"""Unit tests for rings.py

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

import unittest.mock

import numpy as np
import pytest
import sympy

from qldpc import abstract


def test_ring() -> None:
    """Construct elements of a group algebra."""
    group: abstract.Group

    group = abstract.TrivialGroup()
    ring = abstract.GroupRing(group, field=3)
    zero = ring.zero
    one = ring.one
    assert bool(one) and not bool(zero)
    assert zero.group == group
    assert one + 2 == group.identity + 2 * one == -one + 1 == one - 1 == zero
    assert group.identity * one == one * group.identity == one**2 == one
    assert np.array_equal(zero.lift(), np.array(0, ndmin=2))
    assert np.array_equal(one.lift(), np.array(1, ndmin=2))
    assert "GF(3)" in str(ring)
    assert ring.is_abelian
    assert ring.is_semisimple

    # test inverses
    for ring in [
        abstract.GroupRing(abstract.TrivialGroup(), field=3),
        abstract.GroupRing(abstract.AbelianGroup(2, 3), field=4),
        abstract.GroupRing(abstract.QuaternionGroup()),
    ]:
        for group_member in ring.group.generate():
            ring_member = abstract.RingMember(ring, group_member)
            ring_member_inverse = ring_member.inverse()
            assert ring_member_inverse is not None
            assert ring_member * ring_member_inverse == ring.one

    # nontrivial inverse
    group = abstract.CyclicGroup(2)
    ring = abstract.GroupRing(group, field=5)
    ring_member = abstract.RingMember(ring, group.identity, (3, group.generators[0]))
    assert ring_member.inverse() is not None
    assert (0 * ring_member).inverse() is None

    # nonexistent inverse
    group = abstract.CyclicGroup(2)
    ring_member = abstract.RingMember(group, group.identity, *group.generators)
    assert ring_member.inverse() is None

    # evaluate polynomials
    group = abstract.QuaternionGroup()
    ring = abstract.GroupRing(group, field=3)
    g_i, g_j = group.generators
    r_i, r_j = ring.generators
    x_i = sympy.Symbol("x_i")
    x_j = sympy.Symbol("x_j")
    symbols = {x_i: g_i, x_j: g_j}
    poly_r = 4 * r_i**2 - 2 * r_i * r_j + r_j
    poly_x = 4 * x_i**2 - 2 * x_i * x_j + x_j
    assert poly_r == ring.eval(poly_x, symbols)

    wrong_symbols = {x_i: r_i, x_j: r_j}
    with pytest.raises(ValueError, match="must be GroupMember-valued"):
        ring.eval(1, wrong_symbols)  # type:ignore[arg-type]

    # edge cases with non-prime number fields
    ring = abstract.GroupRing(group, field=4)
    assert ring.eval(-3, symbols) == -ring.eval(3, symbols)
    with pytest.raises(ValueError, match="The value of the coefficient .* is ambiguous"):
        ring.eval(5, symbols)


def test_printing() -> None:
    """Convert ring members and ring arrays into human-readable strings."""
    ring = abstract.GroupRing(abstract.AbelianGroup(2, 2))
    assert str(ring.zero) == "0"
    assert str(ring.one) == "1"
    assert [str(gg) for gg in ring.generators] == ["x", "y"]

    ring = abstract.GroupRing(abstract.AbelianGroup(2, 2, 2, 2))
    assert [str(gg) for gg in ring.generators] == ["w", "x", "y", "z"]

    # the order of generators for non-Abelian groups is preserved
    group = abstract.DihedralGroup(6)
    ring = abstract.GroupRing(group, 3)
    one = ring.one
    x, y = ring.generators
    vec = [one + y * x**2 * y, x + y]
    ring_array = abstract.RingArray.build(vec)
    assert str(ring_array) == "[1 + y x^2 y, x + y]"


def test_primitive_central_idempotents() -> None:
    """Convert external primitive central idempotents into RingMembers."""
    with pytest.raises(ValueError, match="Only semisimple rings"):
        abstract.GroupRing(abstract.CyclicGroup(2), 2).get_primitive_central_idempotents()

    group = abstract.CyclicGroup(3)
    x = group.generators[0]
    one = group.identity
    ring = abstract.GroupRing(group, 2)
    fake_output = (
        ((1, ((),)), (1, ((0, 1, 2),)), (1, ((0, 2, 1),))),
        ((1, ((0, 1, 2),)), (1, ((0, 2, 1),))),
    )
    expected_idempotents = (
        abstract.RingMember(ring, one, x, x**2),
        abstract.RingMember(ring, x, x**2),
    )
    with unittest.mock.patch(
        "qldpc.external.groups.get_primitive_central_idempotents", return_value=fake_output
    ):
        idempotents = ring.get_primitive_central_idempotents()
        assert idempotents == expected_idempotents
        assert all(idempotent == idempotent * idempotent for idempotent in idempotents)


def test_ring_array(pytestconfig: pytest.Config) -> None:
    """Construct and lift a RingArray."""
    seed = pytestconfig.getoption("randomly_seed")
    np.random.seed(seed)

    int_matrix = np.random.randint(2, size=(3, 3))
    matrix = abstract.RingArray.build(int_matrix)
    assert matrix.group == abstract.TrivialGroup()
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
    with pytest.raises(ValueError, match="must be RingMember-valued"):
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


def test_transpose() -> None:
    """Transpose various objects."""
    group = abstract.CyclicGroup(4)
    for member in group.generate():
        element = abstract.RingMember(group, member)
        assert element.T.T == element

    x0, x1, x2, x3 = group.generate()
    matrix = abstract.RingArray.build([[x0, 0, x1], [x2, 0, abstract.RingMember(group, x3)]])
    assert np.array_equal(matrix.T.T, matrix)


@pytest.mark.parametrize(
    "ring",
    [
        abstract.GroupRing(abstract.DihedralGroup(3)),
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

    assert not np.any(matrix @ matrix.null_space(row_reduce=False).T)
    assert not np.any(matrix.regular_lift() @ matrix.null_space(row_reduce=False).regular_lift().T)
    assert not np.any(matrix.regular_lift() @ matrix.regular_lift().null_space().T)

    with pytest.raises(NotImplementedError, match="Cannot row-reduce"):
        matrix.null_space(row_reduce=True)


def test_ring_row_reduce(pytestconfig: pytest.Config) -> None:
    """Row reduce a ring-valued matrix."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))
    matrix: list[list[int | abstract.RingMember]] | abstract.RingArray

    # we can row-reduce a RingArray over a cyclic group algebra
    ring = abstract.GroupRing(abstract.CyclicGroup(5), field=3)
    one = ring.one
    gen = ring.generators[0]
    gen_inverse = gen.inverse()
    assert gen_inverse is not None

    matrix = [
        [one + gen, 0, gen],
        [gen + gen**2, 0, gen**2],
        [0, 0, one + gen],
    ]
    assert np.array_equal(
        abstract.RingArray.build(matrix).row_reduce(cyclic=True),
        abstract.RingArray.build([[1, 0, 0], [0, 0, 1]], ring),
    )

    # cyclic row-reduction requires a cyclic group
    group = abstract.AbelianGroup(3, 3)
    with pytest.raises(ValueError, match="requires an underlying CyclicGroup"):
        abstract.RingArray.build([[1, 0], [1, 1]], group).row_reduce(cyclic=True)

    # the non-cyclic Howell normal form requires a semisimple group
    ring = abstract.GroupRing(abstract.AbelianGroup(2, 2), field=2)
    with pytest.raises(ValueError, match="not semisimple"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).howell_normal_form()

    # row-reduction for semisimple Abelian groups is not yet supported
    group = abstract.AbelianGroup(3, 3)
    with pytest.raises(NotImplementedError, match="Implementation .* still pending"):
        abstract.RingArray.build([[1, 0], [1, 1]], group).row_reduce()

    # row-reduction for semisimple non-Abelian groups is not yet supported
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    with pytest.raises(NotImplementedError, match="Implementation .* still pending"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).row_reduce()

    # computing a reduced Groebner basis is the final boss
    ring = abstract.GroupRing(abstract.DihedralGroup(2), field=2)
    with pytest.raises(NotImplementedError, match="Here be dragons"):
        abstract.RingArray.build([[1, 0], [1, 1]], ring).row_reduce()


def test_ring_linear_reduction() -> None:
    """Remove rows that are left-ring-linearly-dependent on others."""
    group = abstract.CyclicGroup(2)
    ring = abstract.GroupRing(group, field=2)
    x = ring.generators[0]
    matrix = abstract.RingArray.build([[x + 1, 1]])

    row_reduced_matrix = abstract.RingArray.build([[x + 1, 1], [0, x + 1]])
    assert np.array_equal(matrix.row_reduce(cyclic=True), row_reduced_matrix)
    assert np.array_equal(row_reduced_matrix.without_dependent_rows(), matrix * x)


def test_deprecations() -> None:
    """Throw warnings... for now."""
    ring = abstract.GroupRing(abstract.TrivialGroup())

    vector = ring.field.Random(ring.group.order)
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        ring_member = abstract.RingMember.from_vector(ring, vector)  # type:ignore[arg-type]
        assert np.array_equal(ring_member.to_vector(), vector)

    vector = ring.field.Random(2 * ring.group.order)
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        ring_array = abstract.RingArray.from_field_vector(ring, vector)  # type:ignore[arg-type]
        assert np.array_equal(ring_array.to_field_vector(), vector)

    matrix = ring.field.Random((1, 2, ring.group.order))
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        ring_array = abstract.RingArray.from_field_array(ring, matrix)  # type:ignore[arg-type]
        assert np.array_equal(ring_array.to_field_array(), matrix)
