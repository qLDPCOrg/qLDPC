"""Module for abstract algebra: groups and representations thereof.

All groups in this module are finite, and represented under the hood as a SymPy PermutationGroup, or
a subgroup of the symmetric group.  Group members subclass the SymPy Permutation class.

!!! WARNINGS !!!

This module does not promise to be performant.  If you need to do heavy numerical abstract algebra,
you're probably better served by GAP or MAGMA (or maybe SageMath).

This module represents group members by matrices over a finite field via a lift L, a homomorphism
with L(g . h) = L(g) @ L(h).  The default lift is the regular representation, whose matrices are
permutation matrices and hence orthogonal; for an orthogonal lift the "transpose" of a group member
p satisfies L(p.T) = L(p).T, and p.T equals the inverse ~p = p**-1.  Some custom lifts
(SpecialLinearGroup, ProjectiveSpecialLinearGroup, QuaternionGroup) are generally not orthogonal, so
that transpose/inverse identity does not hold for them.


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

import contextlib
import functools
import itertools
import math
import operator
from collections.abc import Callable, Iterator, Sequence
from typing import Any, ClassVar

import galois
import numpy as np
import numpy.typing as npt
import scipy.linalg
import sympy.abc
import sympy.combinatorics as comb
import sympy.core

from qldpc import external

from ._monomials import get_coefficient_and_exponents


def resolve_field(
    field: int | type[galois.FieldArray] | None,
) -> type[galois.FieldArray]:
    """Parse a finite field argument to obtain an actual finite field."""
    if field is None:
        return galois.GF2
    if isinstance(field, int):
        return galois.GF(field)
    return field


NestedSequence = Sequence[object | Sequence["NestedSequence"]]


@contextlib.contextmanager
def _preserve_sympy_rng() -> Iterator[None]:
    """Restore SymPy's global RNG state on exit, so a local reseed leaves other consumers intact.

    ``sympy.core.random.seed`` reseeds both the main RNG and the separate "assumptions" RNG, so
    both are saved and restored.
    """
    rng_state = sympy.core.random.rng.getstate()
    assumptions_state = sympy.core.random._assumptions_rng.getstate()
    try:
        yield
    finally:
        sympy.core.random.rng.setstate(rng_state)
        sympy.core.random._assumptions_rng.setstate(assumptions_state)


################################################################################
# groups and group members


class GroupMember(comb.Permutation):
    """Wrapper for SymPy Permutation class.

    Supports sorting permutations (by their rank), and taking their tensor product.
    """

    @staticmethod
    def from_sympy(other: comb.Permutation) -> GroupMember:
        """Convert a SymPy Permutation into a GroupMember."""
        if isinstance(other, GroupMember):
            return other
        new = GroupMember()
        new.__dict__ = other.__dict__
        return new

    def __mul__(self, other: comb.Permutation) -> GroupMember:
        if isinstance(other, comb.Permutation):
            return GroupMember.from_sympy(super().__mul__(other))
        return NotImplemented  # pragma: no cover

    def __add__(self, other: object) -> Any:
        return NotImplemented  # pragma: no cover

    def __lt__(self, other: GroupMember) -> bool:
        return self.rank() < other.rank()

    def __matmul__(self, other: GroupMember) -> GroupMember:
        """Take the "tensor product" of two group members.

        If group members ``g_1`` and ``g_2`` are, respectively, elements of the groups ``G_1`` and
        ``G_2``, then the "tensor product" ``g_1 @ g_2`` is an element of the direct product of
        ``G_1`` and ``G_2``.
        """
        return GroupMember(self.array_form + [val + self.size for val in other.array_form])

    def to_matrix(self) -> npt.NDArray[np.int_]:
        """Lift this permutation object to a permutation matrix.

        For consistency with how SymPy composes permutations, the permutation matrix constructed
        here is right-acting, meaning that it acts on a vector v as ``v --> v @ p.to_matrix()``.
        This convention ensures that this lift is a homomorphism on SymPy Permutation objects,
        which is to say that ``(p * q).to_matrix() = p.to_matrix() @ q.to_matrix()``.
        """
        matrix = np.zeros([self.size] * 2, dtype=int)
        for ii in range(self.size):
            matrix[ii, self.apply(ii)] = 1
        return matrix

    def to_gap_cycles(self) -> str:
        """Convert a GroupMember into a GAP cycle."""

        def to_gap_cycle(cycle: tuple[int, ...]) -> str:
            """Convert a SymPy cycle into a GAP cycle."""
            shifted_cycle = [ii + 1 for ii in cycle]  # GAP indexes from 1
            return f"({','.join(map(str, shifted_cycle))})"

        cycles = [to_gap_cycle(cycle) for cycle in self.cyclic_form]
        return "".join(cycles) if cycles else "()"


Lift = Callable[[GroupMember], npt.NDArray[np.int_]]
IntegerLift = Callable[[int], npt.NDArray[np.int_]]
GenerateFunc = Callable[[], Iterator[comb.Permutation]]


class Group:
    """Base class for a finite group.

    Under the hood, a Group is represented by a SymPy PermutationGroup, and group members are
    represented by SymPy permutations.

    A group naturally comes equipped with a "regular lift" that maps each group member to a
    permutation matrix corresponding to the regular representation of the group.  The regular
    representation of a group represents group members by how they act on the group itself.
    See https://en.wikipedia.org/wiki/Regular_representation.

    A group may additionally be equipped with a custom lift to a matrix over a finite field, for
    which the group action corresponds to matrix multiplication.  Custom lifts are not required to
    be orthogonal (some, such as SpecialLinearGroup, are not).  If no lift is provided,
    group.lift(member) will default to the regular lift of the group.
    """

    _group: comb.PermutationGroup
    _name: str | None
    _generate_func: GenerateFunc | None
    _lift: Lift | None

    def __init__(
        self,
        *generators: comb.Permutation,
        name: str | None = None,
        generate_func: GenerateFunc | None = None,
        lift: Lift | None = None,
    ) -> None:
        self._init_from_group(comb.PermutationGroup(*generators), name, generate_func, lift)

    def _init_from_group(
        self,
        group: comb.PermutationGroup | Group,
        name: str | None = None,
        generate_func: GenerateFunc | None = None,
        lift: Lift | None = None,
    ) -> None:
        """Initialize from an existing group."""
        if isinstance(group, comb.PermutationGroup):
            self._group = group
            self._name = name
            self._generate_func = generate_func
            self._lift = lift
        else:
            assert isinstance(group, Group)
            self._group = group._group
            self._name = name or group._name
            self._generate_func = generate_func or group._generate_func
            self._lift = lift or group._lift

    def __eq__(self, other: object) -> bool:
        # Two groups are equal only if they share a representation as well as an underlying group:
        # equality is consistent with __hash__ (which also mixes in the generation function and
        # lift).  Use ``equiv`` to compare the underlying groups alone, ignoring the representation.
        return (
            isinstance(other, Group)
            and self._group == other._group
            and self._generate_func == other._generate_func
            and self._lift == other._lift
        )

    def __hash__(self) -> int:
        return hash((self._group, self._generate_func, self._lift))

    def equiv(self, other: object) -> bool:
        """Do these share the same underlying group, ignoring their representations (lifts)?

        Unlike ``==``, this compares only the underlying SymPy permutation groups, so groups that
        differ solely in their custom lift (e.g. the same group built twice, hence carrying
        distinct lift closures) compare as equivalent.
        """
        return isinstance(other, Group) and self._group == other._group

    @property
    def name(self) -> str:
        """A name for this group, which is not required to uniquely identify the group."""
        return self._name or f"{type(self).__name__}"

    def __str__(self) -> str:
        return self.name

    def to_sympy(self) -> comb.PermutationGroup:
        """The underlying SymPy permutation group of this Group."""
        return self._group

    @staticmethod
    def from_sympy(group: comb.PermutationGroup, *, lift: Lift | None = None) -> Group:
        """Instantiate a Group from a SymPy permutation group."""
        new_group = Group(lift=lift)
        new_group._group = group
        return new_group

    def with_natural_lift(self) -> Group:
        """Copy this group with its natural permutation representation as the lift.

        The natural lift has the dimension of the set acted on by the group, rather than the order
        of the group as in the regular representation.
        """
        group = Group()
        group._init_from_group(self, name=self.name, lift=GroupMember.to_matrix)
        return group

    def __contains__(self, member: GroupMember) -> bool:
        return self._pad(member) in self._group

    def _pad(self, member: GroupMember) -> GroupMember:
        """Pad a member to this group's full degree.

        A permutation on n points is stored as the list of images [f(0), ..., f(n-1)].  SymPy
        (which backs GroupMember) records only up to the largest moved point: any points fixed at
        the tail are omitted, so a member built directly (e.g. from cycle notation) can have a
        shorter image list than this group's degree.  For example, the permutation swapping 0 and 1
        in the symmetric group on 3 points (degree 3) is stored as [1, 0] rather than [1, 0, 2].

        Member-keyed operations (containment, indexing, custom lifts) assume a full-degree member,
        so re-append the omitted fixed tail points before use.
        """
        if member.size < self.degree:
            return GroupMember(member, size=self.degree)
        return member

    @property
    def order(self) -> int:
        """Number of members in this group."""
        return int(self._group.order())

    @property
    def degree(self) -> int:
        """Number of points that this group's permutation representation acts on."""
        return self._group.degree

    @property
    def is_commutative(self) -> bool:
        """Is this group commutative (abelian)?"""
        return isinstance(self, AbelianGroup) or self._group.is_abelian

    @property
    def is_abelian(self) -> bool:
        """Is this group abelian?

        Alias for Group.is_commutative.
        """
        return self.is_commutative

    @property
    def identity(self) -> GroupMember:
        """The identity element of this group."""
        return GroupMember.from_sympy(self._group.identity)

    @property
    def generators(self) -> list[GroupMember]:
        """Generators of this group."""
        return list(map(GroupMember.from_sympy, self._group.generators))

    def hashable_generators(self) -> tuple[tuple[int, ...], ...]:
        """Generators of this group in a hashable form."""
        return tuple(tuple(generator) for generator in self.generators)

    @functools.cached_property
    def _members(self) -> dict[GroupMember, int]:
        generate = self._generate_func or self._group.generate
        members = map(GroupMember.from_sympy, generate())
        return {member: idx for idx, member in enumerate(members)}

    def generate(self) -> Iterator[GroupMember]:
        """Iterate over all group members."""
        yield from self._members

    def index(self, member: GroupMember) -> int:
        """The index of a GroupMember in this group."""
        index = self._members.get(self._pad(member))
        if not isinstance(index, int):
            raise TypeError(f"Member {member} not in group {self}")
        return index

    def __mul__(self, other: Group) -> Group:
        """Direct product of two groups."""
        return Group.from_sympy(self._group * other._group)

    def __pow__(self, power: int) -> Group:
        """Direct product of self multiple times."""
        assert power > 0
        return functools.reduce(operator.mul, [self] * power)

    @staticmethod
    def product(*groups: Group, repeat: int = 1) -> Group:
        """Direct product of Groups."""
        return functools.reduce(operator.mul, groups * repeat)

    @staticmethod
    def tensor_product(*groups: Group, repeat: int = 1) -> Group:
        """Direct product of Groups with lifts combined by Kronecker products.

        The underlying group is the same as ``Group.product``, while the selected lift of a product
        element is the Kronecker product of its factor lifts.  The ``repeat`` argument repeats the
        entire sequence of factors.
        """
        groups = groups * repeat
        product = Group.product(*groups)
        degrees = [group.degree for group in groups]

        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            matrices = []
            offset = 0
            for group, degree in zip(groups, degrees):
                array_form = [
                    value - offset for value in member.array_form[offset : offset + degree]
                ]
                matrices.append(group.lift(GroupMember(array_form)))
                offset += degree
            return functools.reduce(np.kron, matrices)

        return Group.from_sympy(product.to_sympy(), lift=lift)

    def random(self, *, seed: int | None = None) -> GroupMember:
        """A random element this group."""
        with contextlib.ExitStack() as stack:
            # A seed reseeds SymPy's global RNG; confine that reseed to this call (SymPy exposes no
            # per-call random stream) by restoring the RNG state on the way out.
            if seed is not None:
                stack.enter_context(_preserve_sympy_rng())
                sympy.core.random.seed(seed)

            # HACK to circumvent an error thrown by sympy when "unranking" an empty Permutation
            if self.generators == [GroupMember()]:
                return self.identity

            return GroupMember.from_sympy(self._group.random())

    def regular_lift(self, member: GroupMember, *, right: bool = False) -> npt.NDArray[np.int_]:
        """Lift a group member to its regular representation.

        By default, this method lifts a group member to its left-regular representation.
        Specifically, if ``Vec : G -> F_2^{|G|}`` maps group members to standard basis vectors and
        ``g,h ∈ G``, then

            ``G.regular_lift(g) @ Vec(h) = Vec(g·h)``.

        If right is True, this method lifts a group member to its right-regular anti-representation,
        defined by

            ``G.regular_lift(g, right=True) @ Vec(h) = Vec(h·g)``.

        The right-regular representation of group member is then

            ``G.regular_lift(g, right=True).T = G.regular_lift(~g, right=True)``,

        where ``~g = g**-1`` is the inverse of g.

        See:

        - https://en.wikipedia.org/wiki/Regular_representation
        """
        if (member, right) not in self._regular_lift_cache:
            matrix = np.zeros([self.order] * 2, dtype=int)
            for ii, hh in enumerate(self.generate()):
                jj = self.index(hh * member) if right else self.index(member * hh)
                matrix[jj, ii] = 1
            self._regular_lift_cache[member, right] = matrix
        return self._regular_lift_cache[member, right]

    @functools.cached_property
    def _regular_lift_cache(self) -> dict[tuple[GroupMember, bool], npt.NDArray[np.int_]]:
        """Per-instance cache for regular_lift (avoids the memory leak of caching on methods)."""
        return {}

    def adjoint_lift(self, member: GroupMember) -> npt.NDArray[np.int_]:
        r"""Lift a group member to its adjoint representation.

        The adjoint representation captures how group members get transformed by conjugation.
        If ``Vec : G -> F_2^{|G|}`` lifts group members to standard basis vectors and
        ``g,h ∈ G``, then

            ``adjoint_lift(g) @ Vec(h) = Vec(g·h·~g)``,

        where ``~g = g**-1`` is the inverse of g.

        If the group is abelian, the adjoint lift of every group member is the identity matrix.
        """
        if member not in self._adjoint_lift_cache:
            self._adjoint_lift_cache[member] = (
                self.regular_lift(member) @ self.regular_lift(member, right=True).T
            )
        return self._adjoint_lift_cache[member]

    @functools.cached_property
    def _adjoint_lift_cache(self) -> dict[GroupMember, npt.NDArray[np.int_]]:
        """Per-instance cache for adjoint_lift (avoids the memory leak of caching on methods)."""
        return {}

    @functools.cached_property
    def inversion_matrix(self) -> npt.NDArray[np.int_]:
        """The matrix that maps any group member ``g ∈ G`` to its inverse ``~g = g**-1``.

        If ``Vec : G -> F_2^{|G|}`` lifts group members to standard basis vectors and ``g ∈ G``,
        then

            ``G.inversion_matrix @ Vec(g) = Vec(~g)``,

        where ``~g = g**-1`` is the inverse of g.
        """
        matrix = np.zeros([self.order] * 2, dtype=int)
        for ii, gg in enumerate(self.generate()):
            matrix[self.index(~gg), ii] = 1
        return matrix

    def lift(self, member: GroupMember, *, right: bool = False) -> npt.NDArray[np.int_]:
        """Lift a group member to a representation by a matrix.

        A representation satisfies

            ``self.lift(g·h) = self.lift(g) @ self.lift(h)``.

        If right=True, lift to an anti-representation, for which

            ``self.lift(g·h) = self.lift(h) @ self.lift(g)``.
        """
        if self._lift is None:
            return self.regular_lift(member, right=right)
        if not right or self.is_commutative:
            return self._lift(self._pad(member))
        raise ValueError(
            "Anti-representations for non-commutative groups with custom lifts are not supported"
        )

    @functools.cached_property
    def lift_dim(self) -> int:
        """Dimension of the representation for this group."""
        return self.order if self._lift is None else self.lift(self.generators[0]).shape[0]

    @functools.cached_property
    def table(self) -> npt.NDArray[np.int_]:
        """Multiplication (Cayley) table for this group."""
        return np.array(
            [self.index(aa * bb) for aa, bb in itertools.product(self.generate(), repeat=2)],
            dtype=int,
        ).reshape(self.order, self.order)

    @staticmethod
    def from_table(
        table: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        integer_lift: IntegerLift | None = None,
    ) -> Group:
        """Construct a group from a multiplication (Cayley) table."""
        members = {GroupMember(col): idx for idx, col in enumerate(np.asarray(table).T)}

        def generate_func() -> Iterator[comb.Permutation]:
            yield from members

        if integer_lift is None:
            return Group(*members, generate_func=generate_func)

        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            return integer_lift(members[member])

        return Group(*members, generate_func=generate_func, lift=lift)

    @staticmethod
    def from_generating_mats(*matrices: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> Group:
        """Construct a Group from a set of generating matrices.

        The matrices should be over a finite field (e.g. a ``galois.FieldArray``); their products
        then stay bounded and the generated group is finite.  Plain integer matrices multiply with
        unbounded integer arithmetic, so a group that is infinite as an integer matrix group will
        not terminate here.
        """
        if not matrices:
            return TrivialGroup()

        # keep track of group members and a multiplication table
        index_to_member = {
            idx: gen if isinstance(gen, np.ndarray) else np.asarray(gen, dtype=int)
            for idx, gen in enumerate(matrices)
        }
        hash_to_index = {hash(gen.data.tobytes()): idx for idx, gen in index_to_member.items()}
        table_as_dict = {}

        new_members: dict[int, npt.NDArray[np.int_]]

        def _account_for_product(aa_idx: int, bb_idx: int) -> None:
            """Account for the product of two matrices."""
            cc_mat = index_to_member[aa_idx] @ index_to_member[bb_idx]
            cc_hash = hash(cc_mat.data.tobytes())
            if cc_hash not in hash_to_index:
                hash_to_index[cc_hash] = cc_idx = len(hash_to_index)
                new_members[cc_idx] = cc_mat
            else:
                cc_idx = hash_to_index[cc_hash]
            table_as_dict[aa_idx, bb_idx] = cc_idx

        # generate all members of the group and build the group multiplication table
        members_to_add = index_to_member.copy()
        while members_to_add:
            new_members = {}
            for aa_idx, bb_idx in itertools.product(members_to_add, index_to_member):
                _account_for_product(aa_idx, bb_idx)
                _account_for_product(bb_idx, aa_idx)
            index_to_member |= new_members
            members_to_add = new_members

        # convert the multiplication table into a 2-D array
        table = np.zeros((len(index_to_member), len(index_to_member)), dtype=int)
        for (aa, bb), cc in table_as_dict.items():
            table[aa, bb] = cc

        # dictionary from a permutation to the index of a group member
        permutation_to_index = {tuple(row): idx for idx, row in enumerate(table)}

        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            """Lift a member to its (transposed) matrix representation.

            Each member g is identified with the matrix M_g that it came from.  If we returned M_g
            directly, the lift would not be a homomorphism: SymPy multiplies permutations in the
            order opposite to matrix multiplication, so the product g . h corresponds to M_h @ M_g
            rather than M_g @ M_h.  Transposing swaps the order back, since (M_h @ M_g).T equals
            M_g.T @ M_h.T, giving lift(g . h) = lift(g) @ lift(h) as desired.

            These matrices are generally not orthogonal, so lift(g.T) == lift(g).T may not hold.
            """
            return index_to_member[permutation_to_index[tuple(member.array_form)]].T

        # identify generating permutations and build the group itself
        generators = [GroupMember(table[row]) for row in range(len(matrices))]
        return Group(*generators, lift=lift)

    def random_symmetric_subset(
        self, size: int, *, exclude_identity: bool = False, seed: int | None = None
    ) -> set[GroupMember]:
        """Construct a random symmetric subset of a given size.

        Note: this is not a uniformly random subset, only a "sufficiently random" one.

        WARNING: if excluding the identity element, not all groups have symmetric subsets of
        arbitrary size.  If called with a poor choice of group and subset size, this method may
        never terminate.
        """
        if not 0 < size <= self.order:
            raise ValueError(
                "A random symmetric subset of this group must have a size between 1 and"
                f" {self.order} (provided: {size})"
            )
        with contextlib.ExitStack() as stack:
            # A seed reseeds SymPy's global RNG; confine that reseed to this call (SymPy exposes no
            # per-call random stream) by restoring the RNG state on the way out.
            if seed is not None:
                stack.enter_context(_preserve_sympy_rng())
                sympy.core.random.seed(seed)

            singles = set()  # group members equal to their own inverse
            doubles = set()  # pairs of group members and their inverses
            while True:  # sounds dangerous, but bear with me...
                member = self.random()
                if exclude_identity and member == self.identity:
                    continue  # pragma: no cover

                # always add group members and their inverses
                if member == ~member:
                    singles.add(member)
                else:
                    doubles.add(member)
                    doubles.add(~member)

                # count how many extra group members we have found
                num_extra = len(singles) + len(doubles) - size

                if not num_extra:
                    # if we have the correct number of group members, we are done
                    return singles | doubles

                elif num_extra > 0 and len(singles):
                    # we have overshot, so throw away members to get down to the right size
                    for _ in range(num_extra // 2):
                        member = sorted(doubles)[sympy.core.random.randint(0, len(doubles) - 1)]
                        doubles.remove(member)
                        doubles.remove(~member)
                    if num_extra % 2:
                        member = sorted(singles)[sympy.core.random.randint(0, len(singles) - 1)]
                        singles.remove(member)
                    return singles | doubles

    @staticmethod
    def from_name(
        name: str,
        *,
        from_magma: bool = False,
        warning_to_raise_if_calling_gap: str | None = None,
    ) -> Group:
        """Retrieve a group from the GAP computer algebra system (CAS).

        ... unless from_magma=True, in which case retrieve a group from the MAGMA CAS.
        """
        name = "".join(name.split())  # strip whitespace
        if from_magma:
            generators = external.groups.get_generators_from_magma(name)
        else:
            generators = external.groups.get_generators(
                name, warning_to_raise_if_calling_gap=warning_to_raise_if_calling_gap
            )
        return Group(*[GroupMember(generator) for generator in generators], name=name)

    def to_gap_group(self) -> str:
        """Convert a Group into a GAP group."""
        generators = [gen.to_gap_cycles() for gen in self.generators]
        return f"Group({','.join(generators)})"

    def eval(
        self,
        monomial: sympy.Integer | sympy.Symbol | sympy.Pow | sympy.Mul | int | np.int_,
        symbols: dict[sympy.Symbol, GroupMember],
    ) -> GroupMember:
        """Convert a SymPy monomial into a member of group."""
        coeff, exponents = get_coefficient_and_exponents(monomial)
        if coeff != 1:
            raise ValueError(
                "Only monomials with a coefficient of 1 can be converted into a GroupMember"
                f" (provided: {monomial})"
            )
        output = self.identity
        for base, exponent in exponents:
            output *= symbols[base] ** exponent
        return output


class WreathProductGroup(Group):
    """Permutational wreath product of a top group and copies of a bottom group.

    If the top group acts on ``k`` points and the bottom group acts on ``m`` points, an element
    ``(h; c_0, ..., c_{k-1})`` acts on ``k * m`` points by
    ``(i, j) -> (h(i), c_i(j))``.  The selected lift is the natural permutation representation on
    these ``k * m`` points.
    """

    top: Group
    bottom: Group
    top_degree: int
    bottom_degree: int

    def __init__(self, top: Group, bottom: Group) -> None:
        self.top = top
        self.bottom = bottom
        self.top_degree = top.degree
        self.bottom_degree = bottom.degree
        if not self.top_degree or not self.bottom_degree:
            raise ValueError("Wreath-product groups must act on nonempty sets")

        top_identity = self.top.identity
        bottom_identity = self.bottom.identity
        generators = [
            self.element(generator, [bottom_identity] * self.top_degree)
            for generator in self.top.generators
        ]
        generators += [
            self.element(
                top_identity,
                [
                    generator if index == block else bottom_identity
                    for index in range(self.top_degree)
                ],
            )
            for block in range(self.top_degree)
            for generator in self.bottom.generators
        ]
        super().__init__(*generators, lift=GroupMember.to_matrix)

    def element(
        self,
        top_member: GroupMember,
        bottom_members: Sequence[GroupMember],
    ) -> GroupMember:
        """Construct a wreath-product element from its top and bottom components.

        Args:
            top_member: An element of the top group.
            bottom_members: One bottom-group element for each point acted on by the top group.
        """
        bottom_members = tuple(bottom_members)
        if top_member not in self.top:
            raise ValueError("The top member must belong to the top group")
        if len(bottom_members) != self.top_degree:
            raise ValueError(
                f"Expected {self.top_degree} bottom members (provided: {len(bottom_members)})"
            )
        if any(member not in self.bottom for member in bottom_members):
            raise ValueError("All bottom members must belong to the bottom group")

        # pad members to their full degree, since apply below indexes over every point
        top_member = self.top._pad(top_member)
        bottom_members = tuple(self.bottom._pad(member) for member in bottom_members)

        return GroupMember(
            [
                top_member.apply(row) * self.bottom_degree + bottom_members[row].apply(col)
                for row in range(self.top_degree)
                for col in range(self.bottom_degree)
            ]
        )


################################################################################
# "simple" named groups


class TrivialGroup(Group):
    """The trivial group with one member: the identity."""

    def __init__(self) -> None:
        super().__init__(
            name=TrivialGroup.__name__,
            lift=TrivialGroup._trivial_lift,
        )

    @classmethod
    def _trivial_lift(cls, member: GroupMember) -> npt.NDArray[np.int_]:
        return np.array(1, ndmin=2, dtype=int)

    @staticmethod
    def to_ring_array(data: npt.NDArray[np.int_] | NestedSequence) -> None:
        """DEFUNCT alias for qldpc.abstract.RingArray.build(data)."""
        raise ValueError(
            "TrivialGroup.to_ring_array(data) is DEFUNCT; use"
            " qldpc.abstract.RingArray.build(data) instead"
        )


class AbelianGroup(Group):
    """Direct product of cyclic groups of the specified orders.

    See CyclicGroup for more info.  By default, an AbelianGroup member of the form
    ``∏_i g_i^{a_i}``, where ``{g_i}`` are the generators of the group, gets lifted to a Kronecker
    product ``⨂_i L(g_i)^{a_i}``.  If an AbelianGroup is initialized with direct_sum=True, the group
    members get lifted to a direct sum ``⨁_i L(g_i)^{a_i}``.
    """

    orders: tuple[int, ...]

    def __init__(self, *orders: int, direct_sum: bool = False) -> None:
        self.orders = orders
        group = comb.named_groups.AbelianGroup(*orders)
        order_text = ",".join(map(str, orders))
        name = f"AbelianGroup({order_text})"

        identity_mats = [np.eye(order, dtype=int) for order in orders]
        vals = [sum(orders[:idx]) for idx in range(len(orders))]

        # build lift manually, which is faster than the default lift
        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            shifts = [member.apply(val) - val for val in vals]
            mats = [
                np.roll(identity_mat, shift, axis=0)
                for identity_mat, shift in zip(identity_mats, shifts)
            ]
            if direct_sum:
                combined_mat = scipy.linalg.block_diag(*mats)
            else:
                combined_mat = functools.reduce(np.kron, mats)
            return combined_mat

        # override the default order in which SymPy iterates over group members
        def generate_func() -> Iterator[comb.Permutation]:
            for powers in itertools.product(*[range(order) for order in orders]):
                factors = [gen**power for gen, power in zip(group.generators, powers)]
                yield functools.reduce(operator.mul, factors)

        super()._init_from_group(group, name, generate_func, lift)


class CyclicGroup(AbelianGroup):
    """Cyclic group of a specified order.

    The cyclic group has one generator, g.  All members of the cyclic group of order R can be
    written as ``g^p`` for an integer power p in ``{0, 1, ..., R-1}``.  The member ``g^p`` can be
    represented by (that is, lifted to) an ``R×R`` "shift matrix", or the identity matrix with all
    rows shifted down (equivalently, all columns shifted right) by p.  That is, the lift ``L(g^p)``
    acts on a standard basis vector ``<i|`` as ``<i| L(g^p) = < i + p mod R |``.
    """

    def __init__(self, order: int) -> None:
        self.orders = (order,)
        identity_mat = np.eye(order, dtype=int)

        # build lift manually, which is faster than the default lift
        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            return np.roll(identity_mat, member.apply(0), axis=0)

        super()._init_from_group(comb.named_groups.CyclicGroup(order), lift=lift)


class DihedralGroup(Group):
    """Dihedral group: symmetries of a regular polygon with a given number of sides."""

    def __init__(self, sides: int) -> None:
        super()._init_from_group(comb.named_groups.DihedralGroup(sides))


class AlternatingGroup(Group):
    """Alternating group: even permutations of a set with a given number of elements."""

    def __init__(self, degree: int) -> None:
        super()._init_from_group(comb.named_groups.AlternatingGroup(degree))


class SymmetricGroup(Group):
    """Symmetric group: all permutations of a given number of symbols."""

    def __init__(self, symbols: int) -> None:
        super()._init_from_group(comb.named_groups.SymmetricGroup(symbols))


class QuaternionGroup(Group):
    """Quaternion group: 1, i, j, k, -1, -i, -j, -k.

    The 2-dimensional lift is a faithful homomorphism, but over GF(3) it is not orthogonal --
    lift(i) and lift(j) satisfy ``M.T @ M == -I`` -- so the transpose/inverse identity
    ``lift(g.T) == lift(g).T`` does not hold for this group.
    """

    # multiplication table for this group

    _table: ClassVar = [
        [0, 1, 2, 3, 4, 5, 6, 7],
        [1, 4, 3, 6, 5, 0, 7, 2],
        [2, 7, 4, 1, 6, 3, 0, 5],
        [3, 2, 5, 4, 7, 6, 1, 0],
        [4, 5, 6, 7, 0, 1, 2, 3],
        [5, 0, 7, 2, 1, 4, 3, 6],
        [6, 3, 0, 5, 2, 7, 4, 1],
        [7, 6, 1, 0, 3, 2, 5, 4],
    ]

    def __init__(self) -> None:

        def integer_lift(member: int) -> npt.NDArray[np.int_]:
            """Representation from https://en.wikipedia.org/wiki/Quaternion_group."""
            assert 0 <= member < 8
            sign = 1 if member < 4 else -1
            base = member % 4  # +/- 1, i, j, k
            if base == 0:  # +/- 1
                mat = [[1, 0], [0, 1]]
            elif base == 1:  # +/- i
                mat = [[1, 1], [1, -1]]
            elif base == 2:  # +/- j
                mat = [[-1, 1], [1, 1]]
            else:  # if base == 3; +/- k
                mat = [[0, -1], [1, 0]]
            return sign * (np.array(mat, dtype=int) % 3).view(galois.GF(3))

        group = Group.from_table(self._table, integer_lift=integer_lift)
        super()._init_from_group(group, name=QuaternionGroup.__name__)

    @property
    def generators(self) -> list[GroupMember]:
        """Generators of the quaternion group: [i, j]."""
        generator = self.generate()
        next(generator)
        ii = next(generator)
        jj = next(generator)
        return [ii, jj]


class SmallGroup(Group):
    """Group indexed by the GAP computer algebra system."""

    group_index: int

    def __init__(self, order: int, index: int) -> None:
        assert order > 0
        num_groups = SmallGroup.number(order)
        if not 1 <= index <= num_groups:
            raise ValueError(
                f"Index for SmallGroup of order {order} must be between 1 and {num_groups}"
                + f" (provided: {index})"
            )

        name = f"SmallGroup({order},{index})"
        super()._init_from_group(Group.from_name(name))
        self.group_index = index

    @functools.cached_property
    def structure(self) -> str:
        """A description of the structure of this group."""
        return self.get_structure(self.order, self.group_index)

    @staticmethod
    def number(order: int) -> int:
        """The number of groups of a given order."""
        return external.groups.get_small_group_number(order)

    @staticmethod
    def generator(order: int) -> Iterator[SmallGroup]:
        """Iterator over all groups of a given order."""
        for ii in range(SmallGroup.number(order)):
            yield SmallGroup(order, ii + 1)

    @staticmethod
    def get_structure(order: int, index: int) -> str:
        """Retrieve a description of the structure of a group."""
        return external.groups.get_small_group_structure(order, index)


################################################################################
# special linear (SL) and projective special linear (PSL) groups


class SpecialLinearGroup(Group):
    """Special linear group (SL): square matrices with determinant 1.

    The linear-representation lift is a homomorphism, but its matrices are generally not orthogonal,
    so the transpose/inverse identity ``lift(g.T) == lift(g).T`` does not hold here.
    """

    _dimension: int
    _field: type[galois.FieldArray]

    def __init__(
        self,
        dimension: int,
        field: int | type[galois.FieldArray] | None = None,
        linear_rep: bool = True,
    ) -> None:
        self._name = f"SL({dimension},{field})"
        self._dimension = dimension
        self._field = resolve_field(field)

        if linear_rep:
            # Construct a linear representation of this group, in which group elements permute
            # elements of the vector space that the generating matrices act on.

            # identify the target space that group members (as matrices) act on: all nonzero vectors
            target_space = [
                self.field(vec).tobytes()
                for vec in itertools.product(self.field.elements, repeat=self.dimension)
            ]
            del target_space[0]  # remove the zero vector

            # identify how the generators permute elements of the target space
            generators = []
            for member in self.get_generating_mats(self.dimension, self.field.order):
                perm = np.empty(len(target_space), dtype=int)
                for index, vec_bytes in enumerate(target_space):
                    next_vec = member @ np.frombuffer(vec_bytes, dtype=np.uint8).view(self.field)
                    next_index = target_space.index(next_vec.view(np.ndarray).tobytes())
                    perm[index] = next_index
                generators.append(GroupMember(perm))

            def lift(member: GroupMember) -> npt.NDArray[np.int_]:
                """Lift a group member to a square matrix.

                Each column is determined by how the matrix acts on a standard basis vector.
                """
                cols = []
                for entry in range(self.dimension):
                    inp_vec = np.zeros(self.dimension, dtype=np.uint8)
                    inp_vec[entry] = 1
                    inp_idx = target_space.index(inp_vec.tobytes())
                    out_idx = member(inp_idx)
                    out_vec = np.frombuffer(target_space[out_idx], dtype=np.uint8)
                    cols.append(out_vec)
                # stacking the images as rows yields the transpose of the acting matrix; that
                # transpose (not the matrix itself) is the homomorphism lift(g.h) = lift(g) lift(h)
                return self.field(np.vstack(cols, dtype=int))

            super()._init_from_group(comb.PermutationGroup(generators), lift=lift)

        else:
            # represent group members by how they permute elements of the group itself
            generating_mats = self.get_generating_mats(self.dimension, self.field.order)
            group = self.from_generating_mats(*generating_mats)
            super()._init_from_group(group)

    @property
    def dimension(self) -> int:
        """Dimension of the elements of this group."""
        return self._dimension

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this group."""
        return self._field

    @staticmethod
    def get_generating_mats(
        dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Generating matrices for the Special Linear group, based on https://arxiv.org/abs/2201.09155."""
        field = resolve_field(field)
        minus_one = -field(1)
        gen_w = minus_one * np.diag(np.ones(dimension - 1, dtype=int), k=-1).view(field)
        gen_w[0, -1] = 1
        gen_x = field.Identity(dimension)
        if field.order <= 3:
            gen_x[0, 1] = 1
        else:
            gen_x[0, 0] = field.primitive_element
            gen_x[1, 1] = field.primitive_element**-1
            gen_w[0, 0] = minus_one
        return gen_x, gen_w

    @staticmethod
    def iter_mats(
        dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> Iterator[galois.FieldArray]:
        """Iterate over all elements of SL(dimension, field)."""
        field = resolve_field(field)
        for vec in itertools.product(field.elements, repeat=dimension**2):
            mat = np.reshape(vec, (dimension, dimension)).view(field)
            if np.linalg.det(mat) == 1:
                yield mat


class ProjectiveSpecialLinearGroup(Group):
    """Projective special linear group (PSL = SL/center).

    Here "center" is the subgroup of SL that commutes with all elements of SL.  Specifically, every
    element in the center of SL is a scalar multiple of the identity matrix I.  In the case of
    ``SL(d,q)`` (``d×d`` matrices over ``F_q`` with determinant 1), the determinant of
    ``scalar*I`` is ``scalar**d``, which is only contained in ``SL(d,q)`` if
    ``scalar**d == 1``.

    Altogether, we construct ``PSL(d,q)`` by ``SL(d,q)`` mod [d-th roots of unity over ``F_q``].

    There are two ways to represent ``PSL`` by matrices.  The ``d``-dimensional linear
    representation (inherited from ``SL``) only works when ``gcd(d, q - 1) == 1`` (e.g. ``PSL(2,4)``
    but not ``PSL(2,5)``); otherwise we fall back to a permutation representation, which always
    works.  The ``linear_rep`` argument chooses between them: ``None`` (default) uses the linear
    representation when it exists and the permutation representation otherwise; ``True`` forces the
    linear representation, raising an error when it does not exist; ``False`` always uses the
    permutation representation.
    """

    _dimension: int
    _field: type[galois.FieldArray]

    def __init__(
        self,
        dimension: int,
        field: int | type[galois.FieldArray] | None = None,
        linear_rep: bool | None = None,
    ) -> None:
        self._name = f"PSL({dimension},{field})"
        self._dimension = dimension
        self._field = resolve_field(field)

        # The linear representation of PSL exists only when SL has a trivial center.  The center is
        # the scalar matrices in SL, of size gcd(d, q - 1), so this happens exactly when that gcd
        # is 1.  See the class docstring for how linear_rep selects the representation.
        num_roots = math.gcd(self.dimension, self.field.order - 1)
        has_linear_rep = num_roots == 1
        if linear_rep and not has_linear_rep:
            raise ValueError(
                f"PSL({self.dimension}, {self.field.order}) has a nontrivial center "
                f"(gcd(d, q - 1) = {num_roots} > 1), so the {self.dimension}-dimensional linear "
                "representation of SL does not descend to PSL; use linear_rep=False."
            )
        if linear_rep is None:
            linear_rep = has_linear_rep

        if linear_rep:
            # with a trivial center, PSL(d, q) = SL(d, q): represent members by their action on all
            # nonzero vectors, exactly as SpecialLinearGroup does
            target_space = [
                self.field(vec).tobytes()
                for vec in itertools.product(self.field.elements, repeat=self.dimension)
            ]
            del target_space[0]  # remove the zero vector

            # identify how the generators permute elements of the target space
            generators = []
            for member in SpecialLinearGroup.get_generating_mats(self.dimension, self.field.order):
                perm = np.empty(len(target_space), dtype=int)
                for index, vec_bytes in enumerate(target_space):
                    next_vec = member @ np.frombuffer(vec_bytes, dtype=np.uint8).view(self.field)
                    next_index = target_space.index(next_vec.view(np.ndarray).tobytes())
                    perm[index] = next_index
                generators.append(GroupMember(perm))

            def lift(member: GroupMember) -> npt.NDArray[np.int_]:
                """Lift a group member to a square matrix.

                Each column is determined by how the matrix acts on a standard basis vector.
                """
                cols = []
                for entry in range(self.dimension):
                    inp_vec = np.zeros(self.dimension, dtype=np.uint8)
                    inp_vec[entry] = 1
                    inp_idx = target_space.index(inp_vec.tobytes())
                    out_idx = member(inp_idx)
                    out_vec = np.frombuffer(target_space[out_idx], dtype=np.uint8)
                    cols.append(out_vec)
                # see SpecialLinearGroup: the transpose of the acting matrix is the homomorphism
                return self.field(np.vstack(cols, dtype=int))

            super()._init_from_group(comb.PermutationGroup(generators), lift=lift)

        else:
            # represent group members by how they permute elements of the group itself
            generating_mats = self.get_generating_mats(self.dimension, self.field.order)
            group = self.from_generating_mats(*generating_mats)
            super()._init_from_group(group)

    @property
    def dimension(self) -> int:
        """Dimension of the elements of this group."""
        return self._dimension

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this group."""
        return self._field

    @staticmethod
    def get_generating_mats(
        dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Generating matrices of PSL, constructed out of the generating matrices of SL."""
        field = resolve_field(field)
        gen_x, gen_w = SpecialLinearGroup.get_generating_mats(dimension, field)
        if field is galois.GF2:
            return gen_x, gen_w
        return (
            np.kron(np.linalg.inv(gen_x), gen_x).view(field),
            np.kron(np.linalg.inv(gen_w), gen_w).view(field),
        )

    @staticmethod
    def iter_mats(
        dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> Iterator[galois.FieldArray]:
        """Iterate over all elements of PSL(dimension, field)."""
        field = resolve_field(field)
        num_roots = math.gcd(dimension, field.order - 1)
        primitive_root = field.primitive_element ** ((field.order - 1) // num_roots)
        roots = [primitive_root**k for k in range(dimension)]
        orbits = [
            frozenset([(root * mat).tobytes() for root in roots])
            for mat in SpecialLinearGroup.iter_mats(dimension, field)
        ]
        for orbit in set(orbits):
            yield np.frombuffer(next(iter(orbit)), dtype=np.uint8).view(field)


SL = SpecialLinearGroup
PSL = ProjectiveSpecialLinearGroup
