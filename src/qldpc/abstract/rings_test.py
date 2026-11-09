# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rings.py."""

from __future__ import annotations

import operator

import numpy as np
import pytest
import sympy
import sympy.combinatorics

from qldpc import abstract
from qldpc.abstract import rings


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
    assert ring.is_commutative
    assert ring.is_semisimple

    with pytest.raises(ValueError, match="integer power >= 0"):
        one ** (-1)

    # inverses of group members in a noncommutative ring
    ring = abstract.GroupRing(abstract.QuaternionGroup())
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

    # the group trace projects any element of the ring into its center
    aa_vec = ring.group_trace_matrix @ ring.field.Random(group.order)
    aa = abstract.RingMember.from_vector(aa_vec, ring)
    bb = abstract.RingMember.from_vector(ring.field.Random(group.order), ring)
    assert aa * bb == bb * aa

    # a vector of coefficients has one entry for each group member
    for size in [group.order - 1, group.order + 1]:
        with pytest.raises(ValueError, match="has 8 coefficients"):
            abstract.RingMember.from_vector(ring.field.Zeros(size), ring)

    wrong_symbols = {x_i: r_i, x_j: r_j}
    with pytest.raises(ValueError, match="must be GroupMember-valued"):
        ring.eval(1, wrong_symbols)  # type:ignore[arg-type]

    # edge cases with non-prime number fields
    ring = abstract.GroupRing(group, field=4)
    assert ring.eval(-3, symbols) == -ring.eval(3, symbols)
    with pytest.raises(ValueError, match=r"The value .* is ambiguous"):
        ring.eval(5, symbols)


def test_ring_equality() -> None:
    """Group algebras compare by their fields, groups, and lifts."""
    group = abstract.CyclicGroup(3)
    other = abstract.CyclicGroup(3)  # the same group and lift, built separately
    ring, other_ring = abstract.GroupRing(group), abstract.GroupRing(other)
    assert ring == other_ring
    assert hash(ring) == hash(other_ring)
    assert ring != abstract.GroupRing(abstract.CyclicGroup(4))  # different group
    assert ring != abstract.GroupRing(group, field=4)  # different field
    assert ring != "not a ring"

    def algebras_equal(group_a: abstract.Group, group_b: abstract.Group) -> bool:
        """Do two groups give equal group algebras, whichever is compared with the other?"""
        ring_a, ring_b = abstract.GroupRing(group_a), abstract.GroupRing(group_b)
        assert (ring_a == ring_b) is (ring_b == ring_a)
        return ring_a == ring_b

    # equal group algebras lift group members to the same matrices
    symmetric = abstract.SymmetricGroup(3)
    natural = symmetric.with_natural_lift()
    assert algebras_equal(symmetric, abstract.SymmetricGroup(3))
    assert not algebras_equal(symmetric, natural)
    assert algebras_equal(
        natural, abstract.Group.from_sympy(symmetric.to_sympy(), lift=natural.lift)
    )

    # conjugating a lift by the lift of one generator of SymmetricGroup(3) changes the lift of the
    # other, which does not commute with it
    def conjugate_by(generator: abstract.GroupMember) -> abstract.Group:
        mat = natural.lift(generator)
        return abstract.Group.from_sympy(
            symmetric.to_sympy(), lift=lambda gg: mat @ natural.lift(gg) @ mat.T
        )

    for generator in symmetric.generators:
        conjugated = conjugate_by(generator)
        assert np.array_equal(conjugated.lift(generator), natural.lift(generator))
        assert not algebras_equal(natural, conjugated)

    # groups that list their members in the same order have the same default lifts
    members = list(symmetric.generate())

    def reorder(order: list[abstract.GroupMember]) -> abstract.Group:
        return abstract.Group(*symmetric.generators, generate_func=lambda: iter(order))

    assert algebras_equal(symmetric, reorder(members))

    # translating members by a member that is not central changes default lifts: translating them on
    # the right preserves lifts but changes right lifts, and vice versa on the left
    second = symmetric.generators[1]
    for order, right in [
        ([gg * second for gg in members], False),
        ([second * gg for gg in members], True),
    ]:
        translated = reorder(order)
        assert all(
            np.array_equal(symmetric.lift(gg, right=right), translated.lift(gg, right=right))
            for gg in symmetric.generators
        )
        assert not algebras_equal(symmetric, translated)

    # a noncommutative group has right lifts only with its default lift
    custom = abstract.Group.from_sympy(symmetric.to_sympy(), lift=symmetric.lift)
    assert not algebras_equal(symmetric, custom)

    # right lifts of a commutative group equal its lifts, so default and custom lifts can agree;
    # CyclicGroup has a custom lift, so build a cyclic group with the default lift
    cyclic = abstract.Group(*abstract.CyclicGroup(4).generators)
    assert algebras_equal(cyclic, abstract.Group.from_sympy(cyclic.to_sympy(), lift=cyclic.lift))


def test_printing() -> None:
    """Convert ring members into human-readable strings."""
    ring = abstract.GroupRing(abstract.AbelianGroup(2, 2))
    assert str(ring.zero) == "0"
    assert str(ring.one) == "1"
    assert [str(gg) for gg in ring.generators] == ["x", "y"]

    ring = abstract.GroupRing(abstract.AbelianGroup(2, 2, 2, 2))
    assert [str(gg) for gg in ring.generators] == ["w", "x", "y", "z"]

    # every member of a noncommutative group prints as a word that evaluates to that member, even
    # though the SymPy group of QuaternionGroup has more generators than the group
    group = abstract.QuaternionGroup()
    assert len(group.to_sympy().generators) > len(group.generators)
    ring = abstract.GroupRing(group, 3)
    symbols = sympy.symbols("x y", commutative=False)
    local_dict = {str(symbol): symbol for symbol in symbols}
    values = dict(zip(symbols, group.generators))
    for member in group.generate():
        ring_member = abstract.RingMember(ring, member)
        text = str(ring_member).replace("^", "**").replace(" ", "*")
        word = sympy.parse_expr(text, local_dict=local_dict)
        assert ring.eval(word, values) == ring_member


def test_field_scalars() -> None:
    """Field scalars add to, subtract from, and multiply ring members."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=3)
    x = ring.generators[0]
    scalar = ring.field(2)
    one = ring.one.to_vector()
    assert np.array_equal((x + scalar).to_vector(), x.to_vector() + scalar * one)
    assert np.array_equal((x - scalar).to_vector(), x.to_vector() - scalar * one)
    assert np.array_equal((x * scalar).to_vector(), scalar * x.to_vector())


def test_arithmetic_across_rings() -> None:
    """Members of a group algebra combine only with members of equal group algebras."""
    ring = abstract.GroupRing(abstract.SymmetricGroup(3))

    # members of a separately built, equal group algebra combine
    equal_ring = abstract.GroupRing(abstract.SymmetricGroup(3))
    assert ring.one + equal_ring.one == ring.zero
    assert ring.one * equal_ring.one == ring.one

    # members of group algebras with another group, field, or lift do not
    for different_ring in [
        abstract.GroupRing(abstract.CyclicGroup(3)),
        abstract.GroupRing(abstract.SymmetricGroup(3), field=3),
        abstract.GroupRing(abstract.SymmetricGroup(3).with_natural_lift()),
    ]:
        with pytest.raises(ValueError, match="different rings"):
            ring.one + different_ring.one
        with pytest.raises(ValueError, match="different rings"):
            ring.one * different_ring.one


def test_group_members_of_ring_members() -> None:
    """Ring members accept members of their group, padded to its degree, and reject others."""
    group = abstract.CyclicGroup(3)
    ring = abstract.GroupRing(group, field=3)
    transposition = abstract.GroupMember([1, 0, 2])
    assert transposition not in group
    for operation in [
        lambda: abstract.RingMember(ring, transposition),
        lambda: ring.one + transposition,
        lambda: ring.one - transposition,
        lambda: ring.one * transposition,
        lambda: transposition + ring.one,
        lambda: transposition - ring.one,
        lambda: transposition * ring.one,
    ]:
        with pytest.raises(ValueError, match="not a member"):
            operation()

    # in odd characteristic, subtracting a group member or ring member subtracts coefficients
    generator = group.generators[0]
    expected = ring.one.to_vector() - abstract.RingMember(ring, generator).to_vector()
    assert np.array_equal((ring.one - generator).to_vector(), expected)
    assert np.array_equal((1 - abstract.RingMember(ring, generator)).to_vector(), expected)
    assert np.array_equal((generator - ring.one).to_vector(), -expected)

    # a permutation that acts on fewer points than the group is padded with fixed points
    ring = abstract.GroupRing(abstract.SymmetricGroup(4))
    short, full = abstract.GroupMember(0, 1), abstract.GroupMember([1, 0, 2, 3])
    assert short.size < full.size
    assert abstract.RingMember(ring, short) == abstract.RingMember(ring, full)
    assert ring.zero + short == abstract.RingMember(ring, full)

    # a SymPy permutation becomes a group member, and a label that is not a permutation is rejected
    permutation = sympy.combinatorics.Permutation([1, 0, 2, 3])
    assert not isinstance(permutation, abstract.GroupMember)
    assert abstract.RingMember(ring, (1, permutation)) == abstract.RingMember(ring, full)
    with pytest.raises(ValueError, match="not a member"):
        abstract.RingMember(ring, (1, 0))

    # every identity permutation, of any size, is the identity of the group, even in a trivial
    # group, whose identity SymPy does not identify as a member of that group
    assert abstract.RingMember(ring, abstract.GroupMember(list(range(5)))) == ring.one
    trivial = abstract.TrivialGroup()
    assert trivial.identity not in trivial.to_sympy()
    trivial_ring = abstract.GroupRing(trivial)
    assert abstract.RingMember(trivial_ring, trivial.identity) == trivial_ring.one


def test_primitive_central_idempotents(ring_cyclic3_gf2: abstract.GroupRing) -> None:
    """Convert external primitive central idempotents into RingMembers."""
    ring = ring_cyclic3_gf2
    x = ring.generators[0]
    idempotents = ring.get_primitive_central_idempotents()
    assert idempotents == (x**2 + x + 1, x**2 + x)
    assert all(idempotent == idempotent * idempotent for idempotent in idempotents)

    with pytest.raises(ValueError, match="Only semisimple rings"):
        abstract.GroupRing(abstract.CyclicGroup(2), field=2).get_primitive_central_idempotents()


def test_get_transformer_reuse() -> None:
    """A seedless get_transformer call reuses an already-built decomposition."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    transformer = ring.get_transformer(seed=0)
    assert ring.get_transformer() is transformer


def test_transpose() -> None:
    """Transpose ring members, which inverts group members."""
    group = abstract.CyclicGroup(4)
    for member in group.generate():
        element = abstract.RingMember(group, member)
        assert element.T == abstract.RingMember(group, ~member)
        assert element.T.T == element


def test_deprecated_argument_order() -> None:
    """Deprecated argument orders emit DeprecationWarning."""
    ring = abstract.GroupRing(abstract.TrivialGroup())

    vector = ring.field.Random(ring.group.order)
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        # the deprecated argument order does not match the typed signature
        ring_member = abstract.RingMember.from_vector(ring, vector)  # type:ignore[arg-type]
    assert np.array_equal(ring_member.to_vector(), vector)

    # the deprecated argument order also works for other rings
    matrix_ring = abstract.MatrixRing(np.array([[0, 1], [1, 0]]))
    vector = matrix_ring.field([1, 1])
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        ring_member = abstract.RingMember.from_vector(matrix_ring, vector)  # type:ignore[arg-type]
    assert ring_member == abstract.RingMember.from_vector(vector, matrix_ring)


def test_deprecated_attribute_lookup() -> None:
    """The module-level lookup of deprecated names rejects other names."""
    with pytest.raises(AttributeError, match="has no attribute 'not_a_real_export'"):
        rings.not_a_real_export  # type: ignore[attr-defined]  # noqa: B018


def test_ring_interface() -> None:
    """Group algebras and matrix rings implement the ``Ring`` interface."""
    group = abstract.SymmetricGroup(3).with_natural_lift()
    group_ring = abstract.GroupRing(group)
    assert group_ring.dim == group.order == 6 and group_ring.lift_dim == group.lift_dim == 3
    assert group_ring.has_involution
    assert group_ring.basis == [
        abstract.RingMember(group_ring, member) for member in group.generate()
    ]

    ring = abstract.MatrixRing(np.array([[0, 1], [1, 0]]), field=3)
    one, swap = ring.basis
    assert ring.is_commutative and str(2 * swap + one) == "e_0 + 2 e_1" and str(ring.zero) == "0"
    shift = abstract.MatrixRing(np.roll(np.eye(3, dtype=int), 1, axis=0)).generators[0]
    assert shift.inverse() == shift**2
    with pytest.raises(NotImplementedError, match="only for group algebras"):
        ring.get_transformer()
    with pytest.raises(AttributeError, match="no base group"):
        _ = ring.one.group


def test_combining_members_of_different_rings() -> None:
    """Members of a matrix ring do not combine with members of other rings, or group members."""
    ring = abstract.MatrixRing(np.array([[0, 1], [1, 0]]))
    other_ring = abstract.MatrixRing(np.array([[0, 1], [1, 1]]))
    assert ring.field is other_ring.field and ring.dim == other_ring.dim
    with pytest.raises(ValueError, match="different rings"):
        ring.one + other_ring.one
    with pytest.raises(ValueError, match="different rings"):
        ring.one * other_ring.one

    # group members combine only with members of group algebras
    group_member = abstract.CyclicGroup(3).generators[0]
    with pytest.raises(TypeError):
        ring.one + group_member
    with pytest.raises(TypeError, match="only with members of group algebras"):
        ring.one * group_member
    with pytest.raises(TypeError):
        group_member * ring.one

    # operands that are not ring members are rejected
    for operation, operands in [
        (operator.add, (ring.one, "x")),
        (operator.mul, (ring.one, "x")),
        (operator.mul, ("x", ring.one)),
    ]:
        with pytest.raises(TypeError):
            operation(*operands)


def test_group_member_arithmetic() -> None:
    """Group members multiply members of a group algebra from the side on which they appear."""
    ring = abstract.GroupRing(abstract.SymmetricGroup(3), field=3)
    group_member = ring.group.generators[0]
    as_ring_member = abstract.RingMember(ring, group_member)
    member = ring.one + 2 * ring.generators[1]
    assert member * group_member != group_member * member
    assert group_member * member == as_ring_member * member
    assert member * group_member == member * as_ring_member
    assert member + group_member == group_member + member == member + as_ring_member
