# SPDX-License-Identifier: Apache-2.0

"""Unit tests for rings.py."""

from __future__ import annotations

import io
import pickle

import numpy as np
import pytest
import sympy

from qldpc import abstract
from qldpc.abstract import ring_array, rings


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
    assert ring.is_abelian
    assert ring.is_semisimple

    with pytest.raises(ValueError, match="integer power >= 0"):
        one ** (-1)

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

    # the group trace projects any element of the ring into its center
    aa_vec = ring.group_trace_matrix @ ring.field.Random(group.order)
    aa = abstract.RingMember.from_vector(aa_vec, ring)
    bb = abstract.RingMember.from_vector(ring.field.Random(group.order), ring)
    assert aa * bb == bb * aa

    wrong_symbols = {x_i: r_i, x_j: r_j}
    with pytest.raises(ValueError, match="must be GroupMember-valued"):
        ring.eval(1, wrong_symbols)  # type:ignore[arg-type]

    # edge cases with non-prime number fields
    ring = abstract.GroupRing(group, field=4)
    assert ring.eval(-3, symbols) == -ring.eval(3, symbols)
    with pytest.raises(ValueError, match=r"The value .* is ambiguous"):
        ring.eval(5, symbols)


def test_ring_equality() -> None:
    """Group algebras compare (and hash) by value, ignoring the group's representation."""
    group = abstract.CyclicGroup(3)
    other = abstract.CyclicGroup(3)  # the same group, but a separately built representation
    ring, other_ring = abstract.GroupRing(group), abstract.GroupRing(other)
    assert ring == other_ring
    assert hash(ring) == hash(other_ring)
    assert ring != abstract.GroupRing(abstract.CyclicGroup(4))  # different group
    assert ring != abstract.GroupRing(group, field=4)  # different field
    assert ring != "not a ring"

    with pytest.raises(ValueError, match="DEFUNCT"):
        abstract.TrivialGroup.to_ring_array([])


def test_printing() -> None:
    """Convert ring members into human-readable strings."""
    ring = abstract.GroupRing(abstract.AbelianGroup(2, 2))
    assert str(ring.zero) == "0"
    assert str(ring.one) == "1"
    assert [str(gg) for gg in ring.generators] == ["x", "y"]

    ring = abstract.GroupRing(abstract.AbelianGroup(2, 2, 2, 2))
    assert [str(gg) for gg in ring.generators] == ["w", "x", "y", "z"]

    # the order of generators for non-abelian groups is preserved
    group = abstract.DihedralGroup(6)
    ring = abstract.GroupRing(group, 3)
    one = ring.one
    x, y = ring.generators
    assert str(one + y * x**2 * y) == "1 + y x^2 y"
    assert str(x + y) == "x + y"


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
    """Transpose ring members."""
    group = abstract.CyclicGroup(4)
    for member in group.generate():
        element = abstract.RingMember(group, member)
        assert element.T.T == element


def test_deprecations() -> None:
    """Deprecated call signatures emit DeprecationWarning."""
    ring = abstract.GroupRing(abstract.TrivialGroup())

    vector = ring.field.Random(ring.group.order)
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        ring_member = abstract.RingMember.from_vector(ring, vector)  # type:ignore[arg-type]
    assert np.array_equal(ring_member.to_vector(), vector)

    # the Element alias warns on use
    with pytest.warns(DeprecationWarning, match="DEPRECATED"):
        abstract.Element(ring, ring.group.identity).to_vector()


def test_ring_array_compat_shim() -> None:
    """RingArray/Protograph remain importable from rings.py, the module they moved out of."""
    from qldpc.abstract.rings import Protograph, RingArray

    assert RingArray is ring_array.RingArray is abstract.RingArray
    assert Protograph is ring_array.Protograph is abstract.Protograph

    with pytest.raises(AttributeError, match=r"module .* has no attribute 'not_a_real_export'"):
        rings.not_a_real_export  # noqa: B018 (deliberately trigger the module __getattr__)


def test_ring_array_pickle_compat() -> None:
    """Objects pickled under the pre-split ``qldpc.abstract.rings.RingArray`` path still unpickle.

    Pickle locates a class by looking up its recorded ``(module, qualname)`` via
    ``Unpickler.find_class``, so exercising that lookup directly confirms compatibility without
    depending on whether the ring/group objects involved are themselves picklable.
    """
    unpickler = pickle.Unpickler(io.BytesIO())  # noqa: S301
    assert unpickler.find_class("qldpc.abstract.rings", "RingArray") is ring_array.RingArray
    assert unpickler.find_class("qldpc.abstract.rings", "Protograph") is ring_array.Protograph
