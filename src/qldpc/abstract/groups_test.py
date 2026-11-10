# SPDX-License-Identifier: Apache-2.0

"""Unit tests for groups.py."""

from __future__ import annotations

import copy
import functools
import itertools
import math
import operator
import random
import unittest.mock
from collections.abc import Callable, Iterator, Sequence

import galois
import numpy as np
import numpy.typing as npt
import pytest
import scipy.linalg
import sympy
import sympy.combinatorics
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
    assert group.order == group.degree == 3
    assert group.random() in group
    assert group.random(seed=0) == group.random(seed=0)
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
    symbol = next(iter(symbols))
    assert group.eval(1.0 * symbol, symbols) == group.eval(symbol, symbols)
    assert group.eval(symbol**2.0, symbols) == group.eval(symbol**2, symbols)
    for expression in [
        sympy.sin(symbol),
        sympy.Symbol("z") * symbol,
        symbol + symbol**2,
        symbol ** sympy.Rational(1, 2),
    ]:
        with pytest.raises(ValueError, match="as a word in the given symbols"):
            group.eval(expression, symbols)
    for expression in [1.5 * symbol, sympy.Rational(3, 2) * symbol, 5 * monomial]:
        with pytest.raises(ValueError, match="coefficient is not 1"):
            group.eval(expression, symbols)

    assert abstract.Group.from_generating_mats([[1]]).equiv(abstract.CyclicGroup(1))

    with pytest.raises(TypeError, match="not in group"):
        abstract.CyclicGroup(1).index(abstract.GroupMember(2, 1))

    assert isinstance(hash(group.hashable_generators()), int)
    assert hash(group) == hash(copy.copy(group))

    # group members combine with permutations, and defer other operands
    with pytest.raises(TypeError):
        gens[0] * 1
    with pytest.raises(TypeError):
        2 * gens[0]
    with pytest.raises(TypeError):
        gens[0] + 1

    # plain permutations on the left compose, with (p * q)(i) = q(p(i))
    product = sympy.combinatorics.Permutation([1, 0]) * abstract.GroupMember([0, 2, 1])
    assert isinstance(product, abstract.GroupMember) and product == abstract.GroupMember([2, 0, 1])

    # the identity is written in GAP as "()"
    assert abstract.GroupMember().to_gap_cycles() == "()"


def test_eval_words() -> None:
    """Group.eval maps words in symbols to products and powers of their members."""
    group = abstract.SymmetricGroup(3)
    g, h = group.generators
    a, b = sympy.symbols("a b", commutative=False)
    symbols = {a: g, b: h}
    words_and_members = [
        ((a * b) ** 3, g * h * g * h * g * h),
        ((a * b) ** -1, h**-1 * g**-1),
        (((a * b**-1) ** 2 * b) ** -2, ((g * h**-1) ** 2 * h) ** -2),
        (sympy.Integer(1), group.identity),
    ]
    for word, member in words_and_members:
        assert group.eval(word, symbols) == member
    with pytest.raises(TypeError, match="not a SymPy expression"):
        group.eval(True, symbols)
    with pytest.raises(ValueError, match="as a word in the given symbols"):
        group.eval(sympy.Mul(a, 2, evaluate=False), symbols)

    # symbols must stand for members of the group
    with pytest.raises(ValueError, match="not a member of"):
        group.eval(a * b, {a: g, b: abstract.GroupMember(0, 3)})
    with pytest.raises(ValueError, match="not a member of"):
        abstract.CyclicGroup(3).eval(b, symbols)
    assert abstract.CyclicGroup(3).eval(a, symbols) == g

    # SymPy reorders commutative symbols, so they must stand for members that commute with the
    # members of the other symbols
    x, y = sympy.symbols("x y")
    with pytest.raises(ValueError, match="does not commute"):
        group.eval(y * x, {x: g, y: h})
    with pytest.raises(ValueError, match="does not commute"):
        group.eval(a * x, {a: g, x: h})
    z = sympy.Symbol("z")
    with pytest.raises(ValueError, match="does not commute"):
        group.eval(x * y * z, {x: group.identity, y: g, z: h})
    assert group.eval(y * x, {x: g, y: g**2}) == group.identity
    assert group.eval(x, {x: g, y: h}) == g

    # malformed expressions get the word error, before commutation is checked
    with pytest.raises(ValueError, match="as a word in the given symbols"):
        group.eval(x * sympy.sin(y), {x: g, y: h})


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

    # every identity permutation is the identity of every group
    groups_to_check: list[abstract.Group] = [
        group,
        abstract.CyclicGroup(1),
        abstract.CyclicGroup(3),
        abstract.SymmetricGroup(3),
    ]
    for some_group in groups_to_check:
        identity_lifts = [
            some_group.lift(some_group.identity),
            some_group.regular_lift(some_group.identity),
            some_group.adjoint_lift(some_group.identity),
        ]
        for identity in [abstract.GroupMember(), abstract.GroupMember(list(range(6)))]:
            assert identity in some_group
            assert some_group.index(identity) == some_group.index(some_group.identity)
            lifts = [
                some_group.lift(identity),
                some_group.regular_lift(identity),
                some_group.adjoint_lift(identity),
            ]
            assert all(np.array_equal(aa, bb) for aa, bb in zip(lifts, identity_lifts))


def test_group_equality_and_equivalence() -> None:
    """``==`` requires a matching representation; ``equiv`` compares underlying groups only."""
    group = abstract.CyclicGroup(3)
    other = abstract.CyclicGroup(3)  # same group, but a separately built lift

    # equality is representation-sensitive (consistent with __hash__), so distinct instances of the
    # same group are not equal, but they are equivalent.  A copy shares the representation.
    assert group == copy.copy(group)
    assert group != other
    assert group != "not a group"
    assert group.equiv(other) and other.equiv(group)
    assert not group.equiv(abstract.CyclicGroup(4))

    # equiv compares SymPy groups, which compare their generators
    generator = abstract.GroupMember([1, 2, 0])
    assert not abstract.Group(generator).equiv(abstract.Group(~generator))
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
    assert_valid_lifts(abstract.AbelianGroup(1, 3))
    assert_valid_lifts(abstract.AbelianGroup(2, 1, 2))
    assert_valid_lifts(abstract.AbelianGroup(2, 3, direct_sum=True))
    assert_valid_lifts(abstract.DihedralGroup(3))
    assert_valid_lifts(abstract.AlternatingGroup(3))
    assert_valid_lifts(abstract.QuaternionGroup())

    # the lift of a cyclic group shifts standard basis vectors, as in L(g) |i> = |i + 1 mod R>
    cycles = [abstract.CyclicGroup(2), abstract.CyclicGroup(3)]
    shifts = [cycle.lift(cycle.generators[0]) for cycle in cycles]
    for cycle, shift in zip(cycles, shifts):
        assert np.array_equal(
            shift, np.eye(cycle.order, dtype=int)[:, np.roll(range(cycle.order), -1)]
        )
        assert np.array_equal(cycle.lift(cycle.generators[0], right=True), shift)

    # with direct_sum=True, an abelian group lifts to block-diagonal matrices
    abelian = abstract.AbelianGroup(2, 3, direct_sum=True)
    generator_2, generator_3 = abelian.generators
    assert abelian.lift_dim == 5
    assert np.array_equal(abelian.lift(generator_2 * generator_3), scipy.linalg.block_diag(*shifts))

    # anti-representations for a non-commutative group with a custom lift are not supported
    group = abstract.QuaternionGroup()
    member = next(iter(group.generate()))
    with pytest.raises(ValueError, match=r"Anti-representations.*not supported"):
        group.lift(member, right=True)


def test_generate_is_lazy() -> None:
    """Group.generate yields members without generating the rest of the group."""

    generated = []

    def generate_func() -> Iterator[abstract.GroupMember]:
        for member in abstract.SymmetricGroup(3).generate():
            generated.append(member)
            yield member

    group = abstract.Group(*abstract.SymmetricGroup(3).generators, generate_func=generate_func)
    assert next(group.generate()) == group.identity
    assert len(generated) == 1


def test_generate_after_sympy_changes_order() -> None:
    """Groups keep the order of their members when SymPy reorders the members of its groups."""
    members = list(abstract.SymmetricGroup(4).generate())
    subgroup = sympy.combinatorics.PermutationGroup(sympy.combinatorics.Permutation([0, 1, 3, 2]))

    # change the order of members in SymPy, before or after indexing the members of a group
    for index_first in [False, True]:
        group = abstract.SymmetricGroup(4)
        if index_first:
            group.index(members[0])
        group.to_sympy().coset_transversal(subgroup)
        assert list(map(abstract.GroupMember.from_sympy, group.to_sympy().generate())) != members

        # the group keeps its order of members, so regular lifts remain representations
        assert list(group.generate()) == members
        assert [group.index(member) for member in members] == list(range(len(members)))
        aa, bb = group.generators
        assert np.array_equal(group.lift(aa * bb), group.lift(aa) @ group.lift(bb))


def test_lifts_return_independent_arrays() -> None:
    """Writing into a lift, or into the matrices that a group was built from, does not change it."""
    group = abstract.CyclicGroup(3)
    member = group.identity
    expected = np.identity(group.order, dtype=int)
    for lift in [group.regular_lift, group.adjoint_lift]:
        lift(member)[:] = 0
        assert np.array_equal(lift(member), expected)

    # a group from matrices copies them, and lifts members to new arrays
    matrices: list[npt.NDArray[np.int_]] = [
        galois.GF(3)([[1, 1], [0, 1]]),
        np.array([[0, 1], [1, 0]]),
    ]
    for matrix in matrices:
        matrix_group = abstract.Group.from_generating_mats(matrix)
        ring = abstract.GroupRing(matrix_group, field=3)
        (generator,) = matrix_group.generators
        expected_lift = matrix.T.copy()
        matrix[:] = 0
        matrix_group.lift(generator)[:] = 0
        ring.lift(generator)[:] = 0
        assert np.array_equal(matrix_group.lift(generator), expected_lift)
        assert np.array_equal(ring.lift(generator), expected_lift)


def assert_valid_lifts(group: abstract.Group) -> None:
    """Assert that lifts are faithful homomorphisms, and check regular and adjoint lifts."""
    group_members = list(group.generate())

    # the permutation matrices and the selected lift
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


def test_members_on_other_numbers_of_points() -> None:
    """Lifts, membership, and indices accept members on fewer or more points than the group.

    SymPy truncates trailing fixed points, so a directly-constructed member (e.g. from cycle
    notation) can have a shorter array_form than the group's degree, and a member can also be built
    on more points.  Groups read members on as many points as they act on.
    """
    # these groups cover a custom lift keyed by table index (from_table with an integer_lift), the
    # to_matrix-based natural and wreath lifts, and the Kronecker-product tensor lift
    table = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    table_group = abstract.Group.from_table(
        table, integer_lift=lambda index: np.roll(np.eye(3, dtype=int), index, axis=0)
    )
    groups = [
        abstract.SymmetricGroup(3).with_natural_lift(),
        abstract.Group.tensor_product(
            abstract.SymmetricGroup(3).with_natural_lift(), abstract.CyclicGroup(5)
        ),
        abstract.WreathProductGroup(abstract.SymmetricGroup(3), abstract.CyclicGroup(2)),
        table_group,
    ]
    for group in groups:
        degree = group.degree
        truncated_members = 0
        for member in group.generate():
            # rebuild from cyclic form, which drops trailing fixed points as SymPy does
            truncated = abstract.GroupMember(member.cyclic_form)
            truncated_members += truncated.size < degree
            assert np.array_equal(group.lift(truncated), group.lift(member))
            assert truncated in group and group.index(truncated) == group.index(member)

            # a member with extra fixed points has the same lift, membership, and index
            extended = abstract.GroupMember(member, size=degree + 2)
            assert np.array_equal(group.lift(extended), group.lift(member))
            assert extended in group and group.index(extended) == group.index(member)
        # the group actually has members whose array_form is shorter than its degree
        assert truncated_members

    # products of members are read on the points of the group, too
    group = abstract.SymmetricGroup(3)
    symbol = sympy.Symbol("x")
    swap = abstract.GroupMember([1, 0, 2])
    assert group.eval(symbol, {symbol: abstract.GroupMember(swap, size=5)}) == swap

    # a member that moves a point beyond the degree of a group is not in the group
    member = abstract.GroupMember([1, 0, 3, 2])
    assert member not in abstract.SymmetricGroup(3)

    # plain SymPy permutations on fewer points are members too
    assert sympy.combinatorics.Permutation(0, 1) in abstract.SymmetricGroup(3)


def test_member_order() -> None:
    """Group members are ordered lexicographically, read on the same number of points."""
    # on the same number of points, the order is that of SymPy ranks
    members = [abstract.GroupMember(list(perm)) for perm in itertools.permutations(range(4))]
    random.Random(0).shuffle(members)
    assert sorted(members) == sorted(members, key=lambda member: member.rank())

    # a permutation on fewer points reads as one with extra fixed points
    swap, padded_swap = abstract.GroupMember([1, 0]), abstract.GroupMember([1, 0, 2])
    assert not swap < padded_swap and not padded_swap < swap
    assert abstract.GroupMember([0, 2, 1]) < swap < abstract.GroupMember([1, 2, 0])

    # the identity of the trivial group, on no points, is padded like other permutations
    identity = next(abstract.TrivialGroup().generate())
    padded_identity = abstract.GroupMember([0, 1])
    assert identity.size == 0
    assert not identity < padded_identity and not padded_identity < identity and identity < swap

    # SymPy permutations compare with members on either side, and other objects do not compare
    assert sympy.combinatorics.Permutation([0, 2, 1]) < swap
    for compare in [operator.lt, operator.gt]:
        with pytest.raises(TypeError):
            compare(swap, "swap")


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
    assert abstract.Group.product(cycle, repeat=2).equiv(group)
    assert (cycle**2).equiv(group)
    with pytest.raises(ValueError, match="positive power"):
        cycle**0
    with pytest.raises(ValueError, match="at least one factor"):
        abstract.Group.product()
    with pytest.raises(ValueError, match="at least one factor"):
        abstract.Group.product(cycle, repeat=0)

    # repeat repeats the entire sequence of factors
    other = abstract.CyclicGroup(3)
    assert abstract.Group.product(cycle, other, repeat=2).equiv(cycle * other * cycle * other)

    # an AbelianGroup without factors is trivial
    group = abstract.AbelianGroup()
    assert group.order == group.lift_dim == 1
    assert np.array_equal(group.lift(group.identity), [[1]])
    assert list(group.generate()) == [group.identity]
    assert group.equiv(abstract.TrivialGroup())


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

    # repeat repeats the entire sequence of factors
    repeated_group = abstract.Group.tensor_product(top, bottom, repeat=2)
    top_member, bottom_member = top.generators[0], bottom.generators[0]
    member = top_member @ bottom_member @ top.identity @ bottom_member
    lifts = [top.lift(top_member), bottom.lift(bottom_member), np.eye(3, dtype=int)]
    assert np.array_equal(
        repeated_group.lift(member), functools.reduce(np.kron, [*lifts, lifts[1]])
    )


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

    # members that act on fewer points than their groups are padded with fixed points
    short_member = abstract.GroupMember(0, 1)
    assert short_member.size < top.degree
    padded_member = abstract.GroupMember(short_member, size=top.degree)
    assert group.element(short_member, bottom_members) == group.element(
        padded_member, bottom_members
    )
    other_group = abstract.WreathProductGroup(abstract.CyclicGroup(2), abstract.SymmetricGroup(3))
    assert other_group.element(other_group.top.identity, [short_member] * 2) == other_group.element(
        other_group.top.identity, [padded_member] * 2
    )
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
    # with this seed, the sampled members overshoot the requested size by an odd count, so a
    # self-inverse member must be discarded
    subset = group.random_symmetric_subset(size=2, seed=1)
    assert len(subset) == 2
    assert subset == {~member for member in subset}

    subset = group.random_symmetric_subset(size=1, exclude_identity=False, seed=0)
    assert subset == {group.identity}

    # with this seed, the first sampled member is the identity, which is excluded
    subset = group.random_symmetric_subset(size=3, exclude_identity=True, seed=2)
    assert len(subset) == 3 and group.identity not in subset

    for size in [0, group.order + 1]:
        with pytest.raises(ValueError, match="must have a size between"):
            group.random_symmetric_subset(size=size)

    # groups of order 1 have random members
    for group in [abstract.CyclicGroup(1), abstract.Group.from_generating_mats([[1]])]:
        assert group.random() == group.identity
        assert group.random_symmetric_subset(1) == {group.identity}


def test_seeded_random_leaves_global_rng_intact() -> None:
    """Passing a seed does not disturb SymPy's global RNG for other consumers.

    A seeded call draws from its own random number generator, so sampling the global SymPy RNG
    before and after a seeded call yields the same sequence as sampling it twice in a row.
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
    """The lift satisfies lift(g . h) == lift(g) @ lift(h) over the whole group.

    It suffices to check that lift(identity) is the identity matrix and that the product rule holds
    whenever h is a generator s: every h is a product s_1 ... s_k of generators, so by induction on
    k, lift(g . h) == lift(g) @ lift(s_1) @ ... @ lift(s_k), which at g == identity equals lift(h).
    This avoids checking all |G|^2 pairs.
    """
    identity = group.lift(group.identity)
    assert np.array_equal(identity, np.eye(len(identity), dtype=int))
    generators = [(s, group.lift(s)) for s in group.generators]
    assert all(
        np.array_equal(group.lift(g * s), group.lift(g) @ lift_s)
        for g in group.generate()
        for s, lift_s in generators
    )


@pytest.mark.parametrize("dimension,field,linear_rep", [(2, 4, True), (2, 2, False), (3, 2, True)])
def test_SL(dimension: int, field: int, linear_rep: bool) -> None:
    """Special linear group; its lift is a homomorphism."""
    group = abstract.SL(dimension, field=field, linear_rep=linear_rep)
    assert group.dimension == dimension and group.field.order == field
    order = np.prod([field**dimension - field**jj for jj in range(dimension)]) // (field - 1)
    mats = tuple(abstract.SL.iter_mats(dimension, field))
    assert group.order == len(mats) == order
    assert all(mat.shape == (dimension, dimension) for mat in mats)
    assert_lift_is_homomorphism(group)
    assert_lifts_generating_mats(group, abstract.SL.get_generating_mats(dimension, field))


def test_matrix_groups_require_dimension_2() -> None:
    """Matrix groups are built from generating matrices of SL, so they need dimension 2 or more."""
    for group_type in [abstract.SL, abstract.PSL, abstract.GL, abstract.PGL]:
        with pytest.raises(ValueError, match="dimension of at least 2"):
            group_type(1, 2)


def assert_lifts_generating_mats(
    group: abstract.Group, generating_mats: Sequence[galois.FieldArray]
) -> None:
    """Assert that a matrix group's generators lift to transposes of its generating matrices."""
    assert len(group.generators) == len(generating_mats)
    for generator, matrix in zip(group.generators, generating_mats):
        assert np.array_equal(group.lift(generator), matrix.T)


@pytest.mark.parametrize(
    "dimension,field,linear_rep",
    [(2, 2, True), (2, 2, False), (2, 4, None), (2, 3, None), (3, 2, None)],
)
def test_PSL(dimension: int, field: int, linear_rep: bool | None) -> None:
    """Projective special linear group; its lift is a homomorphism.

    ``linear_rep=None`` (the default) uses the linear representation where it exists (``gcd = 1``,
    as in ``PSL(2, 4)`` and ``PSL(3, 2)``), and otherwise the representation by conjugation
    (``gcd > 1``, as in ``PSL(2, 3)``).
    """
    group = abstract.PSL(dimension, field, linear_rep=linear_rep)
    order_SL = np.prod([field**dimension - field**jj for jj in range(dimension)]) // (field - 1)
    order = order_SL // math.gcd(dimension, field - 1)
    mats = tuple(abstract.PSL.iter_mats(dimension, field))
    assert group.order == len(mats) == order
    assert all(mat.shape == (dimension, dimension) for mat in mats)
    assert_lift_is_homomorphism(group)

    assert_lift_dim(
        group, dimension, linear_rep, has_linear_rep=math.gcd(dimension, field - 1) == 1
    )

    # each matrix is the first of its orbit under roots of unity, and the matrices are sorted, both
    # by bytes
    roots = [root for root in group.field.elements if root and root**dimension == 1]
    assert_orbit_representatives(mats, roots)


def assert_lift_dim(
    group: abstract.Group, dimension: int, linear_rep: bool | None, *, has_linear_rep: bool
) -> None:
    """Assert that a projective matrix group lifts to d×d matrices, or to d²×d² by conjugation.

    The conjugation representation is replaced by the d×d one over GF(2).
    """
    uses_linear_rep = has_linear_rep if linear_rep is None else linear_rep
    assert isinstance(group, (abstract.PSL, abstract.PGL))
    if uses_linear_rep or group.field.order == 2:
        assert group.lift_dim == dimension
    else:
        assert group.lift_dim == dimension**2


def assert_orbit_representatives(
    mats: tuple[galois.FieldArray, ...], scalars: list[galois.FieldArray]
) -> None:
    """Assert that matrices are sorted by bytes, and are each the first of their orbits by bytes."""
    keys = [mat.tobytes() for mat in mats]
    assert keys == sorted(keys)
    assert all(mat.tobytes() == min((scalar * mat).tobytes() for scalar in scalars) for mat in mats)
    assert all(mat.flags.writeable for mat in mats)


def test_matrix_group_names() -> None:
    """Matrix groups are named after their classes."""
    assert abstract.SL(2, 3).name == "SpecialLinearGroup"
    assert abstract.SL(2, 2, linear_rep=False).name == "SpecialLinearGroup"


def test_psl_requires_trivial_center() -> None:
    """Asking for the linear representation raises an error when it does not exist.

    The linear representation of PSL(d, q) only exists when gcd(d, q - 1) == 1.  PSL(2, 5) has
    gcd(2, 4) == 2, so requesting the linear representation there raises an error.
    """
    with pytest.raises(ValueError, match="does not descend to PSL"):
        abstract.PSL(2, 5, linear_rep=True)


@pytest.mark.parametrize("dimension,field,linear_rep", [(2, 4, True), (2, 2, False), (3, 2, True)])
def test_GL(dimension: int, field: int, linear_rep: bool) -> None:
    """General linear group; its lift is a homomorphism."""
    group = abstract.GL(dimension, field=field, linear_rep=linear_rep)
    order = np.prod([field**dimension - field**jj for jj in range(dimension)])
    mats = tuple(abstract.GL.iter_mats(dimension, field))
    assert group.order == len(mats) == order
    assert all(mat.shape == (dimension, dimension) for mat in mats)
    assert_lift_is_homomorphism(group)
    assert_lifts_generating_mats(group, abstract.GL.get_generating_mats(dimension, field))


@pytest.mark.parametrize(
    "dimension,field,linear_rep",
    [(2, 2, True), (2, 2, False), (2, 3, None), (3, 2, None)],
)
def test_PGL(dimension: int, field: int, linear_rep: bool | None) -> None:
    """Projective general linear group; its lift is a homomorphism.

    ``linear_rep=None`` (the default) uses the linear representation where it exists (the center of
    ``GL`` is trivial only for ``q == 2``, as in ``PGL(2, 2)`` and ``PGL(3, 2)``), and otherwise the
    representation by conjugation (as in ``PGL(2, 3)``).
    """
    group = abstract.PGL(dimension, field, linear_rep=linear_rep)
    order_GL = np.prod([field**dimension - field**jj for jj in range(dimension)])
    order = order_GL // (field - 1)
    mats = tuple(abstract.PGL.iter_mats(dimension, field))
    assert group.order == len(mats) == order
    assert all(mat.shape == (dimension, dimension) for mat in mats)
    assert_lift_is_homomorphism(group)
    assert_lift_dim(group, dimension, linear_rep, has_linear_rep=field == 2)
    assert_orbit_representatives(mats, [scalar for scalar in group.field.elements if scalar])


def test_matrix_groups_over_wide_fields() -> None:
    """Matrix groups work over fields whose elements are stored in more than 8 bits."""
    field = galois.GF(257)
    assert field.dtypes[0] != np.uint8
    generators, lift = abstract.groups._get_linear_rep(field, 1, [field([[3]])])
    group = abstract.Group(*generators, lift=lift)
    assert group.order == 256
    assert np.array_equal(group.lift(group.generators[0]), field([[3]]))

    scalars = [field(1), field(256)]
    # the first representative found, diag(3, 1), is not the first in sorted order
    mats = [field(np.diag(values)) for values in [(3, 1), (256, 255), (1, 2)]]
    representatives = list(abstract.groups._iter_scalar_orbits(scalars, mats))
    assert len(representatives) == 2
    assert_orbit_representatives(tuple(representatives), scalars)

    # generating matrices of one field may be stored with different integer types
    gen_x, gen_w = abstract.SL.get_generating_mats(2, 3)
    group = abstract.Group.from_generating_mats(gen_x, gen_w.astype(np.int64).view(type(gen_w)))
    assert group.order == 24

    # integer matrices may also be stored with different integer types, and may repeat
    swap = np.array([[0, 1], [1, 0]], dtype=np.int32)
    flip = np.array([[1, 0], [0, -1]], dtype=np.int64)
    group = abstract.Group.from_generating_mats(swap, flip, flip.astype(np.int32), swap)
    assert group.order == 8 and len(group.generators) == 2

    # an array of Python integers as objects multiplies them exactly
    big = np.array([[2**40, 1 - 2**80], [1, -(2**40)]], dtype=object)
    assert abstract.Group.from_generating_mats(big).order == 2

    # matrices over fields whose elements are stored as Python objects are compared by value
    field = galois.GF(2**70)
    element = field(2**65)
    assert abstract.groups._array_key(field([[2**65]])) == abstract.groups._array_key(
        field([[2**64]]) * field(2)
    )
    assert abstract.groups._array_key(
        np.array([[np.int64(1)]], dtype=object)
    ) == abstract.groups._array_key(np.array([[1]], dtype=object))
    inverse = int(element**-1)
    assert abstract.Group.from_generating_mats(field([[0, 2**65], [inverse, 0]])).order == 2

    # a list of rows over such a field is a matrix over it
    rows = list(field([[0, 2**65], [inverse, 0]]))
    assert abstract.Group.from_generating_mats(rows).order == 2


def test_pgl_requires_trivial_center() -> None:
    """Asking for the linear representation raises an error when it does not exist.

    The linear representation of ``PGL(d, q)`` only exists when ``q == 2`` (a trivial center).
    ``PGL(2, 3)`` has a center of order 2, so requesting the linear representation there raises an
    error.
    """
    with pytest.raises(ValueError, match="does not descend to PGL"):
        abstract.PGL(2, 3, linear_rep=True)


def test_matrix_groups_keep_their_field() -> None:
    """Matrix groups lift into the field class that they are given, not the default of its order."""
    # a non-default primitive element (and irreducible polynomial x + 2) makes a field class that
    # differs from galois.GF(5)
    field = galois.GF(5, primitive_element=3)
    groups = [
        abstract.SL(2, field),
        abstract.PSL(2, field),  # the representation by conjugation, since gcd(2, 5 - 1) > 1
        abstract.GL(2, field),
        abstract.PGL(2, field),  # the representation by conjugation, since 5 > 2
    ]
    for group in groups:
        assert group.field is field
        assert all(type(group.lift(member)) is field for member in group.generators)

    # nested sequences of field elements give matrices over their field
    swap = field([[0, 1], [1, 0]])
    matrices: list[npt.NDArray[np.int_] | list[npt.NDArray[np.int_]] | list[list[int]]] = [
        list(swap),
        [list(row) for row in swap],
        np.array([list(row) for row in swap], dtype=object),
    ]
    for matrix in matrices:
        swap_group = abstract.Group.from_generating_mats(matrix)
        assert swap_group.order == 2 and type(swap_group.lift(swap_group.generators[0])) is field

    # matrices over different fields raise an error, even with equal entries
    with pytest.raises(ValueError, match="Inconsistent fields"):
        abstract.Group.from_generating_mats([swap[0], galois.GF(3)([1, 0])])
    with pytest.raises(ValueError, match="Inconsistent fields"):
        abstract.Group.from_generating_mats(swap, galois.GF(3)([[0, 1], [1, 0]]))


@pytest.mark.parametrize(
    "base_cls,quotient_cls,dimension,field",
    [
        (abstract.SL, abstract.PSL, 3, 3),
        (abstract.GL, abstract.PGL, 3, 3),
    ],
)
def test_quotient_generating_mats_are_homomorphic(
    base_cls: type[abstract.SL | abstract.GL],
    quotient_cls: type[abstract.PSL | abstract.PGL],
    dimension: int,
    field: int,
) -> None:
    """The generating matrices of PSL and PGL represent those of SL and GL by conjugation.

    Each generator ``g`` maps to ``kron(inv(g).T, g)``, the vectorized action of ``X -> g X g^-1``,
    which is a homomorphism.  This test checks the homomorphism on words in noncommuting generators.
    """
    field_type = galois.GF(field)
    gen_a, gen_b, *_ = base_cls.get_generating_mats(dimension, field_type)
    assert not np.array_equal(gen_a @ gen_b, gen_b @ gen_a)  # confirm noncommuting generators

    rep_a, rep_b, *_ = quotient_cls.get_generating_mats(dimension, field_type)
    assert rep_a.shape == rep_b.shape == (dimension**2, dimension**2)

    def conjugation_rep(mat: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return np.kron(np.linalg.inv(mat).T, mat).view(field_type)

    words = [gen_a @ gen_b, gen_b @ gen_a, gen_a @ gen_a @ gen_b, gen_b @ gen_a @ gen_b]
    reps = [rep_a @ rep_b, rep_b @ rep_a, rep_a @ rep_a @ rep_b, rep_b @ rep_a @ rep_b]
    for word, rep in zip(words, reps):
        assert np.array_equal(rep, conjugation_rep(word))


def test_small_group() -> None:
    """Groups indexed by the GAP computer algebra system."""
    order, index = 2, 1
    desired_group = abstract.CyclicGroup(order)

    # invalid group order or index
    with pytest.raises(ValueError, match="order must be positive"):
        abstract.SmallGroup(0, 1)
    with (
        pytest.raises(ValueError, match="Index for SmallGroup"),
        unittest.mock.patch("qldpc.external._groups.get_small_group_number", return_value=index),
    ):
        abstract.SmallGroup(order, 0)

    # everything works as expected
    generators = [tuple(gen.array_form) for gen in desired_group.generators]
    with (
        unittest.mock.patch("qldpc.external._groups.get_small_group_number", return_value=index),
        unittest.mock.patch("qldpc.external._groups.get_generators", return_value=generators),
    ):
        group = abstract.SmallGroup(order, index)
        assert group.generators == desired_group.generators
        generated = list(abstract.SmallGroup.iter_groups(order))
        assert len(generated) == 1 and generated[0].equiv(desired_group)

        # retrieve group structure
        structure = "test"
        with unittest.mock.patch(
            "qldpc.external._groups.get_small_group_structure", return_value=structure
        ):
            assert group.structure == structure

    # the trivial group, whose members SymPy cannot unrank
    with unittest.mock.patch("qldpc.external._groups.get_small_group_number", return_value=1):
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
