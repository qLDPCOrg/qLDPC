"""Unit tests for groups.py.

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

import copy
import functools
import itertools
import math
import operator
import random
import unittest.mock
from collections.abc import Callable

import galois
import numpy as np
import numpy.typing as npt
import pytest
import sympy
import sympy.core.random

from qldpc import abstract


def test_permutation_group(pytestconfig: pytest.Config) -> None:
    """Permutation members and group construction."""
    seed = pytestconfig.getoption("randomly_seed")
    random.seed(seed)

    gens = [abstract.GroupMember(seq) for seq in ([0, 1, 2], [1, 2, 0], [2, 0, 1])]
    assert gens[0] < gens[1] < gens[2]

    group = abstract.Group(*gens)
    assert all(perm in group for perm in gens)
    assert len(group.generators) == 2
    assert group.random() in group
    assert group.random(seed=0) == group.random(seed=0)
    assert group.to_sympy() == group._group
    assert group.is_commutative
    assert group.is_abelian
    assert group.to_gap_group() == "Group((1,2,3),(1,3,2))"

    gens = [abstract.GroupMember(seq) for seq in itertools.permutations([0, 1, 2])]
    group = abstract.Group(*gens)
    assert not group.is_commutative

    random.shuffle(gens)
    symbols = {sympy.Symbol(f"x_{ii}", commutative=False): gen for ii, gen in enumerate(gens)}
    exponents = [random.randint(-3, 3) for _ in range(len(gens))]
    monomial = functools.reduce(
        operator.mul, [symbol**exponent for symbol, exponent in zip(symbols, exponents)]
    )
    member = functools.reduce(
        operator.mul, [gen**exponent for gen, exponent in zip(gens, exponents)]
    )
    assert member == group.eval(monomial, symbols)
    with pytest.raises(ValueError, match="Only monomials with a coefficient of 1"):
        group.eval(5 * monomial, symbols)

    assert abstract.Group.from_generating_mats([[1]]).equiv(abstract.CyclicGroup(1))

    with pytest.raises(TypeError, match="not in group"):
        abstract.CyclicGroup(1).index(abstract.GroupMember(2, 1))

    assert isinstance(hash(group.hashable_generators()), int)
    assert isinstance(hash(group), int)


def test_trivial_group() -> None:
    """Trivial group tests."""
    group = abstract.TrivialGroup()
    group_squared = group**2
    assert group.equiv(group_squared) and group_squared.equiv(group * group)
    assert group.lift_dim == 1
    assert group_squared.lift_dim == 1
    assert group.random() == group.identity
    assert np.array_equal(group.lift(group.identity), np.array(1, ndmin=2))
    assert group.equiv(abstract.Group.from_generating_mats())
    assert str(group) == "TrivialGroup"


def test_group_equality_and_equivalence() -> None:
    """``==`` requires a matching representation; ``equiv`` compares underlying groups only."""
    group = abstract.CyclicGroup(3)
    other = abstract.CyclicGroup(3)  # same group, but a separately built lift

    # equality is representation-sensitive (consistent with __hash__), so distinct instances of
    # the same group are not equal, but they are equivalent.  A copy shares the representation.
    assert group == copy.copy(group)
    assert group != other
    assert group != "not a group"
    assert group.equiv(other) and other.equiv(group)
    assert not group.equiv(abstract.CyclicGroup(4))
    assert not group.equiv("not a group")


def test_natural_lift() -> None:
    """Select the natural permutation representation of a group."""
    group = abstract.SymmetricGroup(3)
    natural_group = group.with_natural_lift()

    assert group.lift_dim == group.order == 6
    assert natural_group.lift_dim == 3
    assert natural_group != group
    assert natural_group.equiv(group)
    assert natural_group.name == group.name
    assert list(natural_group.generate()) == list(group.generate())
    assert all(
        np.array_equal(natural_group.lift(member), member.to_matrix())
        for member in natural_group.generate()
    )
    assert_valid_lifts(natural_group)


def test_lifts() -> None:
    """Lift named group elements."""
    assert_valid_lifts(abstract.TrivialGroup())
    assert_valid_lifts(abstract.CyclicGroup(3))
    assert_valid_lifts(abstract.AbelianGroup(2, 3))
    assert_valid_lifts(abstract.AbelianGroup(2, 3, direct_sum=True))
    assert_valid_lifts(abstract.DihedralGroup(3))
    assert_valid_lifts(abstract.AlternatingGroup(3))
    assert_valid_lifts(abstract.SymmetricGroup(3))
    assert_valid_lifts(abstract.QuaternionGroup())

    # anti-representations for a non-commutative groups with a custom lift are not supported
    group = abstract.QuaternionGroup()
    member = next(iter(group.generate()))
    with pytest.raises(ValueError, match="Anti-representations.*not supported"):
        group.lift(member, right=True)


def assert_valid_lifts(group: abstract.Group) -> None:
    """Assert the faithfulness of various representations of group members."""
    group_members = list(group.generate())

    # permutation and regular representations
    lifts: list[Callable[[abstract.GroupMember], npt.NDArray[np.int_]]] = [
        abstract.GroupMember.to_matrix,
        group.lift,
    ]
    for lift in lifts:
        assert all(
            aa == bb or not np.array_equal(lift(aa), lift(bb))
            for aa, bb in itertools.product(group_members, repeat=2)
        )
        assert all(
            np.array_equal(lift(aa) @ lift(bb), lift(aa * bb))
            for aa, bb in itertools.product(group_members, repeat=2)
        )

    # regular-representation matrices are permutation matrices, hence orthogonal
    assert all(
        np.array_equal(
            group.regular_lift(gg).T @ group.regular_lift(gg), np.eye(group.order, dtype=int)
        )
        for gg in group_members
    )

    # invert elements: g -> g**(-1)
    assert all(
        np.array_equal(
            np.flatnonzero(group.inversion_matrix[:, group.index(gg)]),
            [group.index(~gg)],
        )
        for gg in group_members
    )

    # the inversion matrix converts between left- and right-regular representations
    assert all(
        np.array_equal(
            group.regular_lift(gg, right=True).T,
            group.inversion_matrix @ group.regular_lift(gg) @ group.inversion_matrix,
        )
        for gg in group_members
    )

    # adjoint representation
    if group.is_abelian:
        assert all(
            np.array_equal(group.adjoint_lift(aa), np.identity(group.order, dtype=int))
            for aa in group_members
        )
    else:
        assert all(
            np.array_equal(
                np.flatnonzero(group.adjoint_lift(aa)[:, group.index(bb)]),
                [group.index(aa * bb * ~aa)],
            )
            for aa, bb in itertools.product(group_members, repeat=2)
        )


def test_group_product() -> None:
    """Direct product of groups."""
    cycle = abstract.CyclicGroup(2)
    identity, shift = cycle.generate()
    table = [
        [0, 1, 2, 3],
        [1, 0, 3, 2],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]
    group = abstract.Group.product(cycle, cycle)
    assert_valid_lifts(group)
    assert group.generators == [shift @ identity, identity @ shift]
    assert np.array_equal(table, group.table)
    assert np.array_equal(table, abstract.Group.from_table(table).table)


def test_group_tensor_product() -> None:
    """Direct product of groups that preserves the selected factor lifts."""
    top = abstract.SymmetricGroup(3).with_natural_lift()
    bottom = abstract.CyclicGroup(5)
    group = abstract.Group.tensor_product(top, bottom)

    assert group.equiv(abstract.Group.product(top, bottom))
    assert group.order == 30
    assert group.lift_dim == 15
    assert all(
        np.array_equal(
            group.lift(top_member @ bottom_member),
            np.kron(top.lift(top_member), bottom.lift(bottom_member)),
        )
        for top_member, bottom_member in itertools.product(top.generate(), bottom.generate())
    )
    assert_valid_lifts(group)

    cycle = abstract.CyclicGroup(2)
    repeated_group = abstract.Group.tensor_product(cycle, repeat=2)
    assert repeated_group.lift_dim == 4


def test_wreath_product_group() -> None:
    """Permutational wreath product of top and bottom groups."""
    top = abstract.SymmetricGroup(3).with_natural_lift()
    bottom = abstract.CyclicGroup(2)
    group = abstract.WreathProductGroup(top, bottom)

    assert group.top is top and group.bottom is bottom
    assert group.top_degree == 3 and group.bottom_degree == 2
    assert group.order == top.order * bottom.order**group.top_degree == 48
    assert group.lift_dim == group.top_degree * group.bottom_degree == 6

    top_member = top.generators[0]
    bottom_identity, bottom_shift = bottom.generate()
    bottom_members = [bottom_shift, bottom_identity, bottom_shift]
    member = group.element(top_member, bottom_members)
    assert member in group
    assert all(
        member.apply(row * group.bottom_degree + col)
        == top_member.apply(row) * group.bottom_degree + bottom_members[row].apply(col)
        for row in range(group.top_degree)
        for col in range(group.bottom_degree)
    )
    assert np.array_equal(group.lift(member), member.to_matrix())
    assert_valid_lifts(group)

    direct_product = abstract.Group.tensor_product(top, bottom)
    direct_member = top_member @ bottom_shift
    wreath_member = group.element(top_member, [bottom_shift] * group.top_degree)
    assert np.array_equal(group.lift(wreath_member), direct_product.lift(direct_member))


def test_wreath_product_group_errors() -> None:
    """Reject invalid wreath-product factors and elements."""
    top = abstract.SymmetricGroup(3)
    bottom = abstract.CyclicGroup(2)
    group = abstract.WreathProductGroup(top, bottom)

    with pytest.raises(ValueError, match="top member"):
        group.element(abstract.CyclicGroup(4).generators[0], [bottom.identity] * 3)
    with pytest.raises(ValueError, match="Expected 3 bottom members"):
        group.element(top.identity, [bottom.identity] * 2)
    with pytest.raises(ValueError, match="bottom group"):
        group.element(top.identity, [bottom.identity] * 2 + abstract.CyclicGroup(3).generators)

    for factor_a, factor_b in [
        (abstract.TrivialGroup(), bottom),
        (bottom, abstract.TrivialGroup()),
    ]:
        with pytest.raises(ValueError, match="nonempty"):
            abstract.WreathProductGroup(factor_a, factor_b)


def test_random_symmetric_subset() -> None:
    """Group.random_symmetric_subset generates properly symmetric subsets of the requested size."""
    group = abstract.CyclicGroup(2) * abstract.CyclicGroup(3)
    for seed in [0, 1]:
        subset = group.random_symmetric_subset(size=2, seed=seed)
        assert subset == {~member for member in subset}

    subset = group.random_symmetric_subset(size=1, exclude_identity=False, seed=0)
    assert subset == {group.identity}

    with pytest.raises(ValueError, match="must have a size between"):
        group.random_symmetric_subset(size=0)


def test_seeded_random_leaves_global_rng_intact() -> None:
    """Passing a seed does not disturb SymPy's global RNG for other consumers.

    Seeding is still deterministic, but the reseed is confined to the seeded call: sampling the
    global SymPy RNG before and after a seeded call yields the same sequence as sampling it twice
    in a row.
    """
    group = abstract.CyclicGroup(2) * abstract.CyclicGroup(3)

    # sample both RNGs that sympy.core.random.seed touches: the main one and the assumptions one
    def sample_global_rngs() -> list[tuple[int, float]]:
        return [
            (sympy.core.random.randint(0, 10**9), sympy.core.random._assumptions_rng.random())
            for _ in range(3)
        ]

    for seeded_call in (
        lambda: group.random(seed=7),
        lambda: group.random_symmetric_subset(size=2, seed=7),
    ):
        sympy.core.random.seed(1234)
        baseline = sample_global_rngs()
        sympy.core.random.seed(1234)
        seeded_call()
        assert sample_global_rngs() == baseline

    # seeding remains deterministic across calls
    assert group.random(seed=3) == group.random(seed=3)


def test_quaternion_group() -> None:
    """Validate the multiplication table for the quaternion group."""
    group = abstract.QuaternionGroup()
    assert np.array_equal(group.table, group._table)

    one = group.identity
    ii, jj = group.generators
    kk = ii * jj
    minus_one = ii * ii
    members = [one, ii, jj, kk, minus_one, minus_one * ii, minus_one * jj, minus_one * kk]
    assert all(gg == hh for gg, hh in zip(group.generate(), members))

    # the 2-D lift is a faithful homomorphism but is NOT orthogonal over GF(3): lift(i) and lift(j)
    # satisfy M.T @ M == -I (= 2*I mod 3), so the transpose/inverse identity does not hold here.
    assert_lift_is_homomorphism(group)
    for generator in (ii, jj):
        lift = group.lift(generator)
        assert np.array_equal(lift.T @ lift, 2 * np.eye(2, dtype=int))


def assert_lift_is_homomorphism(group: abstract.Group) -> None:
    """The lift satisfies lift(g . h) == lift(g) @ lift(h) over the whole group."""
    members = list(group.generate())
    assert all(
        np.array_equal(group.lift(g * h), group.lift(g) @ group.lift(h))
        for g, h in itertools.product(members, members)
    )


@pytest.mark.parametrize("dimension,field,linear_rep", [(2, 4, True), (2, 2, False)])
def test_SL(dimension: int, field: int, linear_rep: bool) -> None:
    """Special linear group; its lift is a homomorphism (though not orthogonal)."""
    group = abstract.SL(dimension, field=field, linear_rep=linear_rep)
    order = np.prod([field**dimension - field**jj for jj in range(dimension)]) // (field - 1)
    mats = tuple(abstract.SL.iter_mats(dimension, field))
    assert group.order == len(mats) == order
    assert_lift_is_homomorphism(group)


@pytest.mark.parametrize(
    "dimension,field,linear_rep",
    [(2, 2, True), (2, 2, False), (2, 3, False), (2, 4, None), (2, 3, None)],
)
def test_PSL(dimension: int, field: int, linear_rep: bool | None) -> None:
    """Projective special linear group; its lift is a homomorphism (though not orthogonal).

    ``linear_rep=None`` (the default) uses the linear representation where it exists (gcd = 1, as in
    PSL(2,4)) and otherwise falls back to the permutation representation (gcd > 1, as in PSL(2,3)).
    """
    group = abstract.PSL(dimension, field, linear_rep=linear_rep)
    order_SL = np.prod([field**dimension - field**jj for jj in range(dimension)]) // (field - 1)
    order = order_SL // math.gcd(dimension, field - 1)
    mats = tuple(abstract.PSL.iter_mats(dimension, field))
    assert group.order == len(mats) == order
    assert_lift_is_homomorphism(group)


def test_psl_requires_trivial_center() -> None:
    """Asking for the linear representation raises an error when it does not exist.

    The linear representation of PSL(d, q) only exists when gcd(d, q - 1) == 1.  PSL(2, 5) has
    gcd(2, 4) == 2, so requesting the linear representation there raises an error.  (The fallback to
    a permutation representation is covered by test_PSL.)
    """
    with pytest.raises(ValueError, match="does not descend to PSL"):
        abstract.PSL(2, 5, linear_rep=True)


def test_resolve_field() -> None:
    """resolve_field accepts None (GF2 default), a field order, or a galois field type."""
    assert abstract.resolve_field(None) is galois.GF2
    assert abstract.resolve_field(3).order == 3
    field = galois.GF(4)
    assert abstract.resolve_field(field) is field


def test_small_group() -> None:
    """Groups indexed by the GAP computer algebra system."""
    order, index = 2, 1
    desired_group = abstract.CyclicGroup(order)

    # invalid group index
    with (
        pytest.raises(ValueError, match="Index for SmallGroup"),
        unittest.mock.patch("qldpc.external.groups.get_small_group_number", return_value=index),
    ):
        abstract.SmallGroup(order, 0)

    # everything works as expected
    generators = [tuple(gen.array_form) for gen in desired_group.generators]
    with (
        unittest.mock.patch("qldpc.external.groups.get_small_group_number", return_value=index),
        unittest.mock.patch("qldpc.external.groups.get_generators", return_value=generators),
    ):
        group = abstract.SmallGroup(order, index)
        assert group.generators == desired_group.generators
        generated = list(abstract.SmallGroup.generator(order))
        assert len(generated) == 1 and generated[0].equiv(desired_group)

        # retrieve group structure
        structure = "test"
        with unittest.mock.patch(
            "qldpc.external.groups.get_small_group_structure", return_value=structure
        ):
            assert group.structure == structure

    # cover a special case
    with unittest.mock.patch("qldpc.external.groups.get_small_group_number", return_value=1):
        group = abstract.SmallGroup(1, 1)
    assert group.equiv(abstract.TrivialGroup())
    assert group.random() == group.identity


def test_magma_group(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Retrieve a group from MAGMA."""
    name = "AutomorphismGroup(LinearCode(Matrix(GF(2),1,2,[[1,1]])));"

    # mock user inputs
    inputs = iter(
        ["Permutation group acting on a set of cardinality 2", "Order = 2", "    (1, 2)", ""]
    )
    monkeypatch.setattr("builtins.input", lambda: next(inputs))

    assert abstract.Group.from_name(name, from_magma=True).equiv(abstract.CyclicGroup(2))
    capsys.readouterr()  # intercept print statements
