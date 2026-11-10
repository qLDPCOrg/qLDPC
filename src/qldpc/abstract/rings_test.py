# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rings.py."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import galois
import numpy as np
import numpy.typing as npt
import pytest
import sympy
import sympy.combinatorics

from qldpc import abstract
from qldpc.abstract import rings
from qldpc.external import _groups


def test_ring() -> None:
    """Construct, invert, and evaluate members of group algebras."""
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
    x_i = sympy.Symbol("x_i", commutative=False)
    x_j = sympy.Symbol("x_j", commutative=False)
    symbols = {x_i: g_i, x_j: g_j}
    poly_r = 4 * r_i**2 - 2 * r_i * r_j + r_j
    poly_x = 4 * x_i**2 - 2 * x_i * x_j + x_j
    assert poly_r == ring.eval(poly_x, symbols)

    # the group trace is r -> sum_g g r g^-1, which projects into the center of the ring
    bb = 2 * r_i + r_j
    assert any(bb * gg != gg * bb for gg in ring.generators)
    aa = abstract.RingMember.from_vector(ring.group_trace_matrix @ bb.to_vector(), ring)
    assert aa == sum((ring.one * gg * bb * ~gg for gg in group.generate()), ring.zero)
    assert all(aa * gg == gg * aa for gg in ring.generators)
    with pytest.raises(ValueError, match="read-only"):
        ring.group_trace_matrix[0, 0] = 1

    # a vector of coefficients has one entry for each group member
    for size in [group.order - 1, group.order + 1]:
        with pytest.raises(ValueError, match="has 8 coefficients"):
            abstract.RingMember.from_vector(ring.field.Zeros(size), ring)

    wrong_symbols = {x_i: r_i, x_j: r_j}
    with pytest.raises(ValueError, match="must be GroupMember-valued"):
        ring.eval(1, wrong_symbols)  # type:ignore[arg-type]

    # terms must be integers times words in the given symbols
    assert ring.eval(1.0 * x_i, symbols) == ring.eval(x_i, symbols)
    assert ring.eval(x_i**2.0, symbols) == ring.eval(x_i**2, symbols)
    for expression in [x_i / 2, sympy.sin(x_i)]:
        with pytest.raises(ValueError, match="the given symbols"):
            ring.eval(expression, symbols)

    # words can be nested, and products and powers of sums are expanded
    a, b = sympy.symbols("a b", commutative=False)
    g, h = abstract.SymmetricGroup(3).generators
    word_ring = abstract.GroupRing(abstract.SymmetricGroup(3))
    word_symbols = {a: g, b: h}
    assert word_ring.eval(3 * (a * b) ** -1, word_symbols) == 3 * abstract.RingMember(
        word_ring, h**-1 * g**-1
    )
    ring_g, ring_h = abstract.RingMember(word_ring, g), abstract.RingMember(word_ring, h)
    assert word_ring.eval((a + b) ** 2, word_symbols) == (ring_g + ring_h) ** 2
    assert word_ring.eval((a + b) ** 2.0, word_symbols) == (ring_g + ring_h) ** 2
    with pytest.raises(ValueError, match="the given symbols"):
        word_ring.eval((a + b) ** -1, word_symbols)

    # products and powers in the unexpanded expression are checked, because expanding can cancel
    # reordered terms; sums are not checked
    x, y, z = sympy.symbols("x y z")
    assert word_ring.eval(x + y, {x: g, y: h}) == ring_g + ring_h
    assert word_ring.eval(x * (y + z), {x: g**0, y: g, z: h}) == ring_g + ring_h
    assert word_ring.eval((x + y) ** 1.0, {x: g, y: h}) == ring_g + ring_h
    for expression in [1 + z * (x + y) * (x - y), (x + y) ** 2 + (x - y) ** 2]:
        with pytest.raises(ValueError, match="does not commute"):
            word_ring.eval(expression, {x: g, y: h, z: g**0})
    with pytest.raises(ValueError, match="does not commute"):
        word_ring.eval(a * (x + b) + a * (b - x), {a: g, b: h, x: h})

    # edge cases with non-prime number fields
    ring = abstract.GroupRing(group, field=9)
    assert ring.eval(-3, symbols) == -ring.eval(3, symbols) != ring.eval(3, symbols)
    with pytest.raises(ValueError, match=r"The value .* is ambiguous"):
        ring.eval(10, symbols)


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

    # factors of order 1 get no symbols
    ring = abstract.GroupRing(abstract.AbelianGroup(2, 1, 2))
    assert [str(gg) for gg in ring.generators] == ["x", "y"]
    assert str(ring.one + sum(ring.generators, ring.zero)) == "x + y + 1"
    assert str(abstract.GroupRing(abstract.AbelianGroup()).one) == "1"

    # with more than 26 generators, members print as words in indexed symbols
    members = [gg for gg in abstract.SymmetricGroup(5).generate() if not gg.is_Identity]
    group = abstract.Group(*members[:27])
    assert len(group.generators) == 27
    ring = abstract.GroupRing(group)
    symbols = [sympy.Symbol(f"x_{index:02}", commutative=False) for index in range(27)]
    assert str(ring.generators[0]) == "x_00"
    local_dict = {str(symbol): symbol for symbol in symbols}
    member = ring.generators[0] * ring.generators[26]
    text = str(member).replace("^", "**").replace(" ", "*")
    word = sympy.parse_expr(text, local_dict=local_dict)
    assert ring.eval(word, dict(zip(symbols, group.generators))) == member

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

    # field arrays do not combine with ring members, but RingArrays of them do
    for combine in [operator.mul, operator.add, operator.sub]:
        with pytest.raises(TypeError, match="not with a field array"):
            combine(x, ring.field([2]))
        with pytest.raises(TypeError):
            combine(ring.field([2]), x)
    array = ring.field([1, 2])
    assert np.array_equal(
        x * abstract.RingArray.build(array, ring), abstract.RingArray.build([x, x * ring.field(2)])
    )

    # field scalars of other field classes are read as coefficients
    assert x * galois.GF(9)(2) == x * scalar
    for combine in [operator.mul, operator.add, operator.sub]:
        with pytest.raises(ValueError, match="cannot be read in"):
            combine(x, galois.GF(5)(2))


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

    # a permutation on fewer or more points than the group is read on the points of the group
    ring = abstract.GroupRing(abstract.SymmetricGroup(4))
    short, full = abstract.GroupMember(0, 1), abstract.GroupMember([1, 0, 2, 3])
    long = abstract.GroupMember([1, 0, 2, 3, 4])
    assert short.size < full.size < long.size
    for member in [short, long]:
        assert abstract.RingMember(ring, member) == abstract.RingMember(ring, full)
        assert ring.zero + member == abstract.RingMember(ring, full)

    # GroupRing.eval reads the members that symbols stand for in the same way
    x_sym = sympy.Symbol("x")
    assert ring.eval(1 + x_sym, {x_sym: long}) == ring.eval(1 + x_sym, {x_sym: full})

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
    """Get primitive central idempotents, sorted by their vectors of coefficients."""
    ring = ring_cyclic3_gf2
    x = ring.generators[0]
    assert ring.get_primitive_central_idempotents() == (x**2 + x, x**2 + x + 1)

    # GF(5)[C2] is not in the table of precomputed idempotents.  Its idempotents are
    # (1 ± x) / 2 = 3 ± 3 x.
    ring = abstract.GroupRing(abstract.CyclicGroup(2), field=5)
    assert (ring.group.to_gap_group(), 5) not in _groups.KNOWN_PRIMITIVE_CENTRAL_IDEMPOTENTS
    x = ring.generators[0]
    assert ring.get_primitive_central_idempotents() == (3 + 2 * x, 3 + 3 * x)

    # the group algebra of the trivial group has one primitive central idempotent: its identity
    ring = abstract.GroupRing(abstract.TrivialGroup())
    assert ring.get_primitive_central_idempotents() == (ring.one,)

    # idempotents from the table are sorted for a group that orders its members differently
    cyclic_group = abstract.CyclicGroup(4)
    group = abstract.Group(
        *cyclic_group.generators, generate_func=lambda: reversed(list(cyclic_group.generate()))
    )
    assert (group.to_gap_group(), 5) in _groups.KNOWN_PRIMITIVE_CENTRAL_IDEMPOTENTS
    vectors = [
        idempotent.to_vector().tolist()
        for idempotent in abstract.GroupRing(group, 5).get_primitive_central_idempotents()
    ]
    assert vectors == sorted(vectors)

    with pytest.raises(ValueError, match="Only semisimple rings"):
        abstract.GroupRing(abstract.CyclicGroup(2), field=2).get_primitive_central_idempotents()


def assert_primitive_central_idempotents(
    ring: abstract.GroupRing, idempotents: Sequence[abstract.RingMember]
) -> None:
    """Assert that ring members are the primitive central idempotents of a group algebra.

    The members must be nonzero, central, orthogonal idempotents that sum to one.  There must be as
    many of them as there are primitive central idempotents, which by the Witt-Berman theorem is the
    number of F_q-conjugacy classes.
    """
    assert sum(idempotents, ring.zero) == ring.one
    for index, idempotent in enumerate(idempotents):
        assert idempotent and idempotent * idempotent == idempotent
        assert all(
            idempotent * generator == generator * idempotent for generator in ring.generators
        )
        assert all(idempotent * other == ring.zero for other in idempotents[index + 1 :])

    # g and h are in the same F_q-conjugacy class if h = y·g^(q^i)·y^(-1) for some y in G and i ≥ 0
    members = list(ring.group.generate())
    order = ring.field.order
    classes = {
        frozenset(yy * gg ** (order**ii) * ~yy for yy in members for ii in range(len(members)))
        for gg in members
    }
    assert len(idempotents) == len(classes)


@pytest.mark.parametrize(
    "group, field",
    [
        (abstract.DihedralGroup(3), galois.GF(7)),
        (abstract.CyclicGroup(5), galois.GF(4)),
        (abstract.CyclicGroup(5), galois.GF(9, irreducible_poly="x^2 + 1")),
    ],
)
def test_computed_primitive_central_idempotents(
    group: abstract.Group, field: type[galois.FieldArray]
) -> None:
    """Computed idempotents are the primitive central idempotents, sorted by their coefficients.

    D3 is not abelian, so conjugation by every generator matters.  Over GF(4), the F_4-conjugacy
    classes of C5 differ from its F_2-conjugacy classes.  Over GF(9) with a non-default irreducible
    polynomial, the idempotents of C5 have coefficients outside the prime field.
    """
    ring = abstract.GroupRing(group, field)
    idempotents = rings._get_primitive_central_idempotents(ring)
    assert_primitive_central_idempotents(ring, idempotents)
    vectors = sorted(idempotent.to_vector().tolist() for idempotent in idempotents)
    assert [pci.to_vector().tolist() for pci in ring.get_primitive_central_idempotents()] == vectors


def test_precomputed_primitive_central_idempotents_need_default_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only prime fields and fields with the default irreducible polynomial use the table."""
    # pretend that the table holds the identity as the only idempotent of GF(9)[C2]
    group = abstract.CyclicGroup(2)
    table = {(group.to_gap_group(), 9): (((1, ()),),)}
    monkeypatch.setattr(_groups, "KNOWN_PRIMITIVE_CENTRAL_IDEMPOTENTS", table)
    ring = abstract.GroupRing(group, field=9)
    assert ring.get_primitive_central_idempotents() == (ring.one,)

    # a field with another irreducible polynomial computes its two idempotents instead
    ring = abstract.GroupRing(group, field=galois.GF(9, irreducible_poly="x^2 + 1"))
    assert len(ring.get_primitive_central_idempotents()) == 2

    # pretend the same for GF(5)[C2]: a prime field uses the table whatever its primitive element
    monkeypatch.setattr(
        _groups, "KNOWN_PRIMITIVE_CENTRAL_IDEMPOTENTS", {(group.to_gap_group(), 5): (((1, ()),),)}
    )
    ring = abstract.GroupRing(group, field=galois.GF(5, primitive_element=3))
    assert ring.get_primitive_central_idempotents() == (ring.one,)


def test_precomputed_primitive_central_idempotents() -> None:
    """The precomputed primitive central idempotents are correct, and equal the computed ones."""
    frobenius_group = abstract.Group(  # SmallGroup(21, 1)
        abstract.GroupMember([[1, 2, 4], [3, 6, 5]]), abstract.GroupMember([list(range(7))])
    )
    group_algebras = [
        *[(abstract.CyclicGroup(order), 2) for order in (3, 5, 7, 9)],
        *[(abstract.CyclicGroup(order), 3) for order in (2, 4, 5)],
        (abstract.CyclicGroup(3), 4),
        (abstract.CyclicGroup(4), 5),
        (abstract.DihedralGroup(3), 5),
        (abstract.AlternatingGroup(4), 5),
        (frobenius_group, 2),
        (abstract.DihedralGroup(5), 3),
    ]
    assert {(group.to_gap_group(), field) for group, field in group_algebras} == set(
        _groups.KNOWN_PRIMITIVE_CENTRAL_IDEMPOTENTS
    )
    for group, field in group_algebras:
        ring = abstract.GroupRing(group, field)
        assert_primitive_central_idempotents(ring, ring.get_primitive_central_idempotents())
        computed = rings._get_primitive_central_idempotents(ring)
        vectors = sorted(idempotent.to_vector().tolist() for idempotent in computed)
        assert [
            pci.to_vector().tolist() for pci in ring.get_primitive_central_idempotents()
        ] == vectors


def test_member_equality_inverses_and_powers() -> None:
    """Members compare by their rings and all of their terms, and have inverses and powers.

    A multiple of a group member has an inverse, and powers may be numpy integers.
    """
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    x, y = ring.generators
    assert ring.zero != ring.one != ring.zero
    assert x != x + y != x
    assert ring.one != abstract.GroupRing(abstract.DihedralGroup(3), field=3).one

    # a multiple of a group member has an inverse
    member = 2 * x
    inverse = member.inverse()
    assert inverse is not None and member * inverse == ring.one

    # powers may be numpy integers
    assert x ** np.int32(2) == x * x


def test_members_over_large_fields() -> None:
    """Members of algebras over fields that store their elements as Python objects can be lifted."""
    large_ring = abstract.GroupRing(abstract.CyclicGroup(2), field=2**70)
    (generator,) = large_ring.generators
    member = 1 + generator * large_ring.field(2**65)
    inverse = member.inverse()
    assert inverse is not None
    assert np.array_equal(member.lift() @ inverse.lift(), large_ring.field.Identity(2))
    assert not large_ring.group_trace_matrix.flags.writeable


def test_reading_coefficients() -> None:
    """Coefficients are integers or field elements, read as the same elements of the ring."""
    ring = abstract.GroupRing(abstract.CyclicGroup(2), field=4)
    x = ring.generators[0]
    vectors: list[npt.ArrayLike] = [
        [1, 2],
        range(1, 3),
        ring.field([1, 2]),
        list(ring.field([1, 2])),
    ]
    for vector in vectors:
        assert abstract.RingMember.from_vector(vector, ring) == 1 + x * ring.field(2)

    # integers are integer representations of field elements, so they are not reduced
    unreduced_vectors: list[npt.ArrayLike] = [[4, 0], np.array([4, 0])]
    for unreduced in unreduced_vectors:
        with pytest.raises(ValueError, match=r"must be in \[0, 4\)"):
            abstract.RingMember.from_vector(unreduced, ring)

    # empty sequences have no coefficients
    assert ring._read_coefficients([[], []]).shape == (2, 0)
    assert abstract.RingArray.build([], ring).shape == (0,)

    # elements of a prime field with the same characteristic denote the same elements
    assert abstract.RingMember.from_vector(galois.GF(2)([1, 1]), ring) == 1 + x

    # so do elements of a field class with the same irreducible polynomial
    same_poly_field = galois.GF(4, irreducible_poly="x^2 + x + 1", primitive_element="x + 1")
    assert same_poly_field is not ring.field
    assert abstract.RingMember.from_vector(same_poly_field([1, 2]), ring) == 1 + x * ring.field(2)

    # elements of another field that lie in its prime field denote the same elements, and others
    # cannot be read, also when they are given one by one
    assert abstract.RingMember.from_vector(galois.GF(16)([1, 0]), ring) == ring.one
    for other_field in [galois.GF(3), galois.GF(16)]:
        other_vectors: list[npt.ArrayLike] = [other_field([1, 2]), list(other_field([1, 2]))]
        for other_vector in other_vectors:
            with pytest.raises(ValueError, match="cannot be read in"):
                abstract.RingMember.from_vector(other_vector, ring)

    # integer-valued floats and booleans are read as integers, and are read exactly, even for fields
    # too large for fixed-width integers
    large_ring = abstract.GroupRing(abstract.CyclicGroup(2), field=2**70)
    large_vectors: list[npt.ArrayLike] = [[2**65, 1], large_ring.field([2**65, 1])]
    for large_vector in large_vectors:
        member = abstract.RingMember.from_vector(large_vector, large_ring)
        assert np.array_equal(member.to_vector(), large_ring.field([2**65, 1]))
    member = abstract.RingMember.from_vector(np.array([2.0**65, 1.0]), large_ring)
    assert np.array_equal(member.to_vector(), large_ring.field([2**65, 1]))
    member = abstract.RingMember.from_vector(np.array([np.True_, 1.0], dtype=object), ring)
    assert member == ring.one + x

    # other values are not coefficients
    bad_vectors: list[npt.ArrayLike] = [["a", "b"], [0.5, 1]]
    for bad_vector in bad_vectors:
        with pytest.raises(ValueError, match="must be integers"):
            abstract.RingMember.from_vector(bad_vector, ring)

    # the coefficients of terms are read in the same way
    (group_member,) = ring.group.generators
    assert abstract.RingMember(ring, (2.0, group_member)) == x * ring.field(2)
    assert abstract.RingMember(ring, (galois.GF(2)(1), group_member)) == x
    with pytest.raises(ValueError, match="cannot be read in"):
        abstract.RingMember(ring, (galois.GF(16)(2), group_member))
    with pytest.raises(ValueError, match="must be integers"):
        abstract.RingMember(ring, (0.5, group_member))
    with pytest.raises(ValueError, match="must be integers"):
        abstract.RingMember(ring, ("x", group_member))  # type:ignore[arg-type]
    with pytest.raises(ValueError, match="must be integers"):
        abstract.RingArray.build([["x"]], ring)
    for array in [[1, 1], [1], ring.field([1])]:
        with pytest.raises(ValueError, match="must be a scalar"):
            abstract.RingMember(ring, (array, group_member))  # type:ignore[arg-type]

    # integer coefficients must lie in [0, q), even -1, which is an additive inverse as an operand
    for integer in [4, -1]:
        with pytest.raises(ValueError, match=r"must be in \[0, 4\)"):
            abstract.RingMember(ring, (integer, group_member))

    # an error names the irreducible polynomial of an extension field
    ring_9 = abstract.GroupRing(abstract.CyclicGroup(2), field=9)
    field_9 = galois.GF(9, irreducible_poly="x^2 + 1")
    with pytest.raises(ValueError, match="irreducible polynomial x\\^2 \\+ 1 cannot be read in"):
        abstract.RingMember.from_vector(field_9([1, 3]), ring_9)


def test_get_transformer_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Seedless get_transformer calls use the default seed; decompositions are cached by seed."""
    monkeypatch.setattr(abstract.rings, "_DEFAULT_TRANSFORMER_SEED", 1)
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    transformer = ring.get_transformer(seed=0)
    assert ring.get_transformer() is ring.get_transformer(seed=1) is not transformer
    assert ring.get_transformer(seed=0) is transformer


def test_get_transformer_is_reproducible() -> None:
    """Without a seed, separately built rings have the same Wedderburn-Artin decomposition."""
    rings = [abstract.GroupRing(abstract.DihedralGroup(3), field=5) for _ in range(2)]
    transformer_a, transformer_b = [ring.get_transformer() for ring in rings]
    for component_a, component_b in zip(transformer_a.transformers, transformer_b.transformers):
        assert np.array_equal(component_a.matrix_basis, component_b.matrix_basis)


def test_transpose() -> None:
    """Transposing a member of a group algebra inverts its group members."""
    group = abstract.CyclicGroup(4)
    for member in group.generate():
        element = abstract.RingMember(group, member)
        assert element.T == abstract.RingMember(group, ~member)
        assert element.T.T == element


def test_deprecated_argument_order() -> None:
    """Deprecated argument orders emit DeprecationWarning."""
    ring = abstract.GroupRing(abstract.TrivialGroup())

    vector = ring.field.Random(ring.group.order, seed=0)
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
    assert ring.is_commutative and str(2 * swap + one) == "e_0 + 2 e_1" and str(swap - swap) == "0"
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


def test_lifts_over_other_fields() -> None:
    """Lifts over a finite field are read as field elements, as coefficients are."""
    group = abstract.QuaternionGroup()
    (generator, _) = group.generators
    lift = group.lift(generator)
    assert type(lift) is galois.GF(3)
    for field in [3, 9]:
        ring = abstract.GroupRing(group, field=field)
        ring_lift = ring.lift(generator)
        assert type(ring_lift) is ring.field and np.array_equal(ring_lift, lift)
    for field in [2, 5]:
        with pytest.raises(
            ValueError, match="Entries of group lifts in GF\\(3\\) cannot be read in"
        ):
            abstract.GroupRing(group, field=field).lift(generator)


def test_regular_lift_returns_independent_arrays() -> None:
    """Mutating a returned regular lift does not corrupt the cached lifts of the group."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3))
    ring.regular_lift(ring.group.identity)[:] = 0
    assert np.array_equal(ring.regular_lift(ring.group.identity), np.identity(3, dtype=int))
