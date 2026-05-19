"""Unit tests for groups.py

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

import functools
import itertools
import math
import operator
import random
import unittest.mock
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest
import sympy

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

    assert abstract.Group.from_generating_mats([[1]]) == abstract.CyclicGroup(1)

    with pytest.raises(ValueError, match="not in group"):
        abstract.CyclicGroup(1).index(abstract.GroupMember(2, 1))

    assert isinstance(hash(group.hashable_generators()), int)


def test_trivial_group() -> None:
    """Trivial group tests."""
    group = abstract.TrivialGroup()
    group_squared = group**2
    assert group == group_squared == group * group
    assert group.lift_dim == 1
    assert group_squared.lift_dim == 1
    assert group.random() == group.identity
    assert np.array_equal(group.lift(group.identity), np.array(1, ndmin=2))
    assert group == abstract.Group.from_generating_mats()
    assert str(group) == "TrivialGroup"

    with pytest.raises(ValueError, match="DEFUNCT"):
        abstract.TrivialGroup.to_ring_array([])


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

    # invert elements: g -> g**(-1)
    assert all(
        np.array_equal(
            np.where(group.inversion_matrix[:, group.index(gg)]),
            [[group.index(~gg)]],
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
                np.where(group.adjoint_lift(aa)[:, group.index(bb)]),
                [[group.index(aa * bb * ~aa)]],
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


@pytest.mark.parametrize("dimension,field,linear_rep", [(2, 4, True), (2, 2, False)])
def test_SL(dimension: int, field: int, linear_rep: bool) -> None:
    """Special linear group."""
    group = abstract.SL(dimension, field=field, linear_rep=linear_rep)
    order = np.prod([field**dimension - field**jj for jj in range(dimension)]) // (field - 1)
    mats = tuple(abstract.SL.iter_mats(dimension, field))
    assert group.order == len(mats) == order

    gens = group.generators
    gen_mats = group.get_generating_mats(dimension, field)
    assert np.array_equal(group.lift(gens[0]), gen_mats[0])
    assert np.array_equal(group.lift(gens[1]), gen_mats[1])


@pytest.mark.parametrize("dimension,field,linear_rep", [(2, 2, True), (2, 3, False)])
def test_PSL(dimension: int, field: int, linear_rep: bool) -> None:
    """Projective special linear group."""
    group = abstract.PSL(dimension, field, linear_rep=linear_rep)
    order_SL = np.prod([field**dimension - field**jj for jj in range(dimension)]) // (field - 1)
    order = order_SL // math.gcd(dimension, field - 1)
    mats = tuple(abstract.PSL.iter_mats(dimension, field))
    assert group.order == len(mats) == order

    if field == 2:
        gens = group.generators
        gen_mats = group.get_generating_mats(dimension, field)
        assert np.array_equal(group.lift(gens[0]), gen_mats[0])
        assert np.array_equal(group.lift(gens[1]), gen_mats[1])


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
        assert list(abstract.SmallGroup.generator(order)) == [desired_group]

        # retrieve group structure
        structure = "test"
        with unittest.mock.patch(
            "qldpc.external.groups.get_small_group_structure", return_value=structure
        ):
            assert group.structure == structure

    # cover a special case
    with unittest.mock.patch("qldpc.external.groups.get_small_group_number", return_value=1):
        group = abstract.SmallGroup(1, 1)
    assert group == abstract.TrivialGroup()
    assert group.random() == group.identity


def test_magma_group(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """Retrieve a group from MAGMA."""
    name = "AutomorphismGroup(LinearCode(Matrix(GF(2),1,2,[[1,1]])));"

    # mock user inputs
    inputs = iter(
        ["Permutation group acting on a set of cardinality 2", "Order = 2", "    (1, 2)", ""]
    )
    monkeypatch.setattr("builtins.input", lambda: next(inputs))

    assert abstract.Group.from_name(name, from_magma=True) == abstract.CyclicGroup(2)
    capsys.readouterr()  # intercept print statements


def test_sympy_parsing() -> None:
    """Parse SymPy monomial expressions."""
    x = sympy.abc.x
    y = sympy.abc.y
    assert abstract.get_coefficient_and_exponents(3) == (3, [])
    assert abstract.get_coefficient_and_exponents(x) == (1, [(x, 1)])
    assert abstract.get_coefficient_and_exponents(x**2) == (1, [(x, 2)])
    assert abstract.get_coefficient_and_exponents(3 * x * y**2) == (3, [(x, 1), (y, 2)])
