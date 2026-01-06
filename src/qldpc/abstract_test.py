"""Unit tests for abstract.py

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

import itertools
import math
import unittest.mock
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from qldpc import abstract


def test_permutation_group() -> None:
    """Permutation members and group construction."""
    gens = [abstract.GroupMember(seq) for seq in ([0, 1, 2], [1, 2, 0], [2, 0, 1])]
    assert gens[0] < gens[1] < gens[2]

    group = abstract.Group(*gens)
    assert all(perm in group for perm in gens)
    assert len(group.generators) == 2
    assert group.random() in group
    assert group.random(seed=0) == group.random(seed=0)
    assert group.to_sympy() == group._group
    assert group.is_abelian
    assert group.to_gap_group() == "Group((1,2,3),(1,3,2))"

    gens = [abstract.GroupMember(seq) for seq in itertools.permutations([0, 1, 2])]
    group = abstract.Group(*gens)
    assert not group.is_abelian

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


def test_lift() -> None:
    """Lift named group elements."""
    assert_valid_lift(abstract.TrivialGroup())
    assert_valid_lift(abstract.CyclicGroup(3))
    assert_valid_lift(abstract.AbelianGroup(2, 3))
    assert_valid_lift(abstract.AbelianGroup(2, 3, direct_sum=True))
    assert_valid_lift(abstract.DihedralGroup(3))
    assert_valid_lift(abstract.AlternatingGroup(3))
    assert_valid_lift(abstract.SymmetricGroup(3))
    assert_valid_lift(abstract.QuaternionGroup())


def assert_valid_lift(group: abstract.Group) -> None:
    """Assert faithfulness of the regular and permutation representations of group members."""
    lifts: list[Callable[[abstract.GroupMember], npt.NDArray[np.int_]]] = [
        group.lift,
        abstract.GroupMember.to_matrix,
    ]
    for lift in lifts:
        assert all(
            aa == bb or not np.array_equal(lift(aa), lift(bb))
            for aa in group.generate()
            for bb in group.generate()
        )
        assert all(
            np.array_equal(lift(aa) @ lift(bb), lift(aa * bb))
            for aa in group.generate()
            for bb in group.generate()
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
    assert_valid_lift(group)
    assert group.generators == [shift @ identity, identity @ shift]
    assert np.array_equal(table, group.table)
    assert np.array_equal(table, abstract.Group.from_table(table).table)


def test_random_symmetric_subset() -> None:
    """Cover Group.random_symmetric_subset."""
    group = abstract.CyclicGroup(2) * abstract.CyclicGroup(3)
    for seed in [0, 1]:
        subset = group.random_symmetric_subset(size=2, seed=seed)
        assert subset == {~member for member in subset}

    subset = group.random_symmetric_subset(size=1, exclude_identity=False, seed=0)
    assert subset == {group.identity}

    with pytest.raises(ValueError, match="must have a size between"):
        group.random_symmetric_subset(size=0)


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

    # nonexistent inverse
    group = abstract.CyclicGroup(2)
    ring_member = abstract.RingMember(group, group.identity, *group.generators)
    assert ring_member.inverse() is None


def test_primitive_central_idempotents() -> None:
    """Convert external primitive central idempotents into RingMembers."""
    with pytest.raises(ValueError, match="Only semisimple rings"):
        abstract.GroupRing(abstract.CyclicGroup(2), 2).get_primitive_central_idempotents()

    group = abstract.CyclicGroup(3)
    xx = group.generators[0]
    one = group.identity
    ring = abstract.GroupRing(group, 2)
    fake_output = (
        ((1, ((),)), (1, ((0, 1, 2),)), (1, ((0, 2, 1),))),
        ((1, ((0, 1, 2),)), (1, ((0, 2, 1),))),
    )
    expected_idempotents = (
        abstract.RingMember(ring, one, xx, xx**2),
        abstract.RingMember(ring, xx, xx**2),
    )
    with unittest.mock.patch(
        "qldpc.external.groups.get_primitive_central_idempotents", return_value=fake_output
    ):
        idempotents = ring.get_primitive_central_idempotents()
        assert idempotents == expected_idempotents
        assert all(idempotent == idempotent * idempotent for idempotent in idempotents)


def test_ring_array() -> None:
    """Construct and lift a RingArray."""
    int_matrix = np.random.randint(2, size=(3, 3))
    matrix = abstract.TrivialGroup.to_ring_array(int_matrix)
    assert matrix.group == abstract.TrivialGroup()
    assert np.array_equal(matrix.lift(), int_matrix)
    assert np.array_equal(
        (matrix @ matrix).lift(),
        matrix.lift() @ matrix.lift(),
    )
    assert isinstance(np.kron(matrix, matrix), abstract.RingArray)

    # fail to construct a valid ring array
    with pytest.raises(ValueError, match="must be RingMember-valued"):
        abstract.RingArray([[0]])
    with pytest.raises(ValueError, match="Cannot determine the underlying ring"):
        abstract.RingArray([])
    with pytest.raises(ValueError, match="Inconsistent base rings"):
        rings = [abstract.GroupRing(abstract.TrivialGroup(), field) for field in [2, 3]]
        abstract.RingArray([ring.one for ring in rings])

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
    matrix = abstract.RingArray.build([[x0, 0, x1], [x2, 0, abstract.RingMember(group, x3)]], group)
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

    vector = abstract.RingArray.from_field_vector(ring, dense_vector)
    matrix = abstract.RingArray.from_field_array(ring, dense_array)
    assert np.array_equal(dense_vector, abstract.RingArray.to_field_vector(vector))
    assert np.array_equal(dense_array, abstract.RingArray.to_field_array(matrix))
    assert np.array_equal(
        (matrix @ vector).to_field_vector(),
        matrix.regular_lift() @ vector.to_field_vector(),
    )

    assert not np.any(matrix @ matrix.null_space().T)
    assert not np.any(matrix.regular_lift() @ matrix.null_space().regular_lift().T)
    assert not np.any(matrix.regular_lift() @ matrix.regular_lift().null_space().T)


@pytest.mark.parametrize(
    "ring",
    [
        abstract.GroupRing(abstract.DihedralGroup(3)),
        abstract.GroupRing(abstract.AbelianGroup(2, 3), field=3),
    ],
)
def test_ring_row_reduce(ring: abstract.GroupRing, pytestconfig: pytest.Config) -> None:
    """Row reduce a ring-valued matrix."""
    seed = pytestconfig.getoption("randomly_seed")
    matrix: list[list[int | abstract.RingMember]] | abstract.RingArray

    one = ring.one
    gen = ring.group.generators[0] * one
    matrix = [
        [one + gen, gen],
        [gen + gen**2, gen**2],
        [0, one + gen],
    ]
    reduced_matrix = [
        [gen.inverse() + one, one],
        [-(one + gen) * (gen.inverse() + one), 0],
    ]
    assert np.array_equal(
        abstract.RingArray.build(matrix, ring).row_reduce(),
        abstract.RingArray.build(reduced_matrix, ring),
    )

    # RingArray.row_reduce and _remove_linearly_dependent_rows have the same left-ring-linear span
    num_rows, num_cols = 3, 5
    coefficients = ring.field.Random((num_rows, num_cols, ring.group.order), seed=seed)
    matrix = abstract.RingArray.from_field_array(ring, coefficients)
    matrix_1 = matrix.row_reduce().regular_lift().row_reduce()
    matrix_2 = matrix._remove_linearly_dependent_rows().regular_lift().row_reduce()
    assert np.array_equal(
        matrix_1[np.any(matrix_1, axis=1)],
        matrix_2[np.any(matrix_2, axis=1)],
    )


def test_row_reduce_errors() -> None:
    """Errors and warnings from RingArray.row_reduce."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3))
    coefficients = ring.field.Random((1, 2, ring.group.order))
    matrix = abstract.RingArray.from_field_array(ring, coefficients)
    with pytest.raises(NotImplementedError, match="We only aspire to perform exact row reduction"):
        matrix.row_reduce(force_heuristic=False)

    ring = abstract.GroupRing(abstract.CyclicGroup(2))
    coefficients = ring.field.Random((1, 2, ring.group.order))
    matrix = abstract.RingArray.from_field_array(ring, coefficients)
    with pytest.warns(UserWarning, match="Using heuristics"):
        matrix.row_reduce(force_heuristic=False)


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
