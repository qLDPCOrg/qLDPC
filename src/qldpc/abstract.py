"""Module for abstract algebra: groups, algebras, and representations thereof

All groups in this module are finite, and represented under the hood as a SymPy PermutationGroup, or
a subgroup of the symmetric group.  Group members subclass the SymPy Permutation class.

!!! WARNINGS !!!

This module only supports representations of group members by orthogonal matrices over finite
fields.  The restriction to orthogonal representations allows identifying the "transpose" of a group
member p with respect to a representation (lift) L, which is defined by enforcing L(p.T) = L(p).T.
If the representation is orthogonal, then p.T is equal to the inverse ~p = p**-1.


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

import collections
import copy
import functools
import itertools
import math
import operator
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal, Union

import galois
import numpy as np
import numpy.typing as npt
import scipy.linalg
import sympy.combinatorics as comb
import sympy.core

from qldpc import external

DEFAULT_FIELD_ORDER = 2


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
        return NotImplemented

    def __add__(self, other: object) -> Any:
        return NotImplemented  # pragma: no cover

    def __lt__(self, other: GroupMember) -> bool:
        return self.rank() < other.rank()

    def __matmul__(self, other: GroupMember) -> GroupMember:
        """Take the "tensor product" of two group members.

        If group members g_1 and g_2 are, respectively, elements of the groups G_1 and G_2, then the
        "tensor product" g_1 @ g_2 is an element of the direct product of G_1 and G_2.
        """
        return GroupMember(self.array_form + [val + self.size for val in other.array_form])

    def to_matrix(self) -> npt.NDArray[np.int_]:
        """Lift this permutation object to a permutation matrix.

        For consistency with how SymPy composes permutations, the permutation matrix constructed
        here is right-acting, meaning that it acts on a vector v as v --> v @ p.to_matrix().  This
        convension ensures that this lift is a homomorphism on SymPy Permutation objects, which is
        to say that (p * q).to_matrix() = p.to_matrix() @ q.to_matrix().
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
    representation of a group represents group members by how they act on the group itself (from the
    left); see https://en.wikipedia.org/wiki/Regular_representation.

    A group may additionally be equipped with a custom lift to an orthogonal matrix over a finite
    field, for which the group action corresponds to matrix multiplication.  If no lift is provided,
    group.lift(member) will default to the regular lift of the group.
    """

    _group: comb.PermutationGroup
    _name: str | None
    _iterator: GenerateFunc | None
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
        self._name = name
        if isinstance(group, comb.PermutationGroup):
            self._group = group
            self._lift = lift
        else:
            assert isinstance(group, Group)
            self._group = group._group
            self._name = self._name or group._name  # explicitly provided name overrides group name
            self._lift = lift or group._lift
        self._generate_func = generate_func

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Group) and self._group == other._group

    def __hash__(self) -> int:
        return hash((self._group, self._generate_func, self._lift))

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

    def __contains__(self, member: GroupMember) -> bool:
        return member in self._group

    @property
    def order(self) -> int:
        """Number of members in this group."""
        return self._group.order()

    @property
    def is_abelian(self) -> bool:
        """Is this group Abelian?"""
        return isinstance(self, AbelianGroup) or self._group.is_abelian

    @property
    def generators(self) -> list[GroupMember]:
        """Generators of this group."""
        return list(map(GroupMember.from_sympy, self._group.generators))

    def hashable_generators(self) -> tuple[tuple[int, ...], ...]:
        """Generators of this group in a hashable form."""
        return tuple(tuple(generator) for generator in self.generators)

    def generate(self) -> Iterator[GroupMember]:
        """Iterate over all group members."""
        generate = self._generate_func or self._group.generate
        yield from map(GroupMember.from_sympy, generate())

    @property
    def identity(self) -> GroupMember:
        """The identity element of this group."""
        return GroupMember.from_sympy(self._group.identity)

    @functools.cached_property
    def _members(self) -> dict[GroupMember, int]:
        return {member: idx for idx, member in enumerate(self.generate())}

    def index(self, member: GroupMember) -> int:
        """The index of a GroupMember in this group."""
        index = self._members.get(member)
        if not isinstance(index, int):
            raise ValueError(f"Member {member} not in group {self}")
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

    def random(self, *, seed: int | None = None) -> GroupMember:
        """A random element this group."""
        if seed is not None:
            sympy.core.random.seed(seed)

        # HACK to circumvent an error thrown by sympy when "unranking" an empty Permutation
        if self.generators == [GroupMember()]:
            return self.identity

        return GroupMember.from_sympy(self._group.random())

    @functools.cache
    def regular_lift(self, member: GroupMember) -> npt.NDArray[np.int_]:
        """Lift a group member to its regular representation."""
        matrix = np.zeros([self.order] * 2, dtype=int)
        for ii, gg in enumerate(self.generate()):
            matrix[self.index(member * gg), ii] = 1
        return matrix

    def lift(self, member: GroupMember) -> npt.NDArray[np.int_]:
        """Lift a group member to a representation by an orthogonal matrix."""
        return self.regular_lift(member) if self._lift is None else self._lift(member)

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
        members = {GroupMember(row): idx for idx, row in enumerate(table)}

        if integer_lift is None:
            return Group(*members)

        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            return integer_lift(members[member])

        return Group(*members, lift=lift)

    @staticmethod
    def from_generating_mats(*matrices: npt.NDArray[np.int_] | Sequence[Sequence[int]]) -> Group:
        """Constructs a Group from a given set of generating matrices."""
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
            """Lift a member to its matrix representation."""
            return index_to_member[permutation_to_index[tuple(member.array_form)]]

        # identify generating permutations and build the group itself
        generators = [GroupMember(table[row]) for row in range(len(matrices))]
        return Group(*generators, lift=lift)

    def random_symmetric_subset(
        self, size: int, *, exclude_identity: bool = False, seed: int | None = None
    ) -> set[GroupMember]:
        """Construct a random symmetric subset of a given size.

        Note: this is not a uniformaly random subset, only a "sufficiently random" one.

        WARNING: if excluding the identity element, not all groups have symmetric subsets of
        arbitrary size.  If called with a poor choice of group and subset size, this method may
        never terminate.
        """
        if not 0 < size <= self.order:
            raise ValueError(
                "A random symmetric subset of this group must have a size between 1 and"
                f" {self.order} (provided: {size})"
            )
        if seed is not None:
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


################################################################################
# group algebra and elements thereof


class GroupRing:
    """A finite group algebra over a finite field.

    The base field is GF(2) by default.
    """

    _group: Group
    _field: type[galois.FieldArray]

    def __init__(self, group: Group, field: int | None = None) -> None:
        self._group = group
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)

    @property
    def group(self) -> Group:
        """Base group of this ring."""
        return self._group

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this ring."""
        return self._field

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GroupRing) and self.field is other.field and self.group == other.group
        )

    def __hash__(self) -> int:
        return hash((self.field.order, self.group))

    @property
    def name(self) -> str:
        """A name for this ring, which is not required to uniquely identify the ring."""
        return f"Group algebra of {self.group.name} over GF({self.field.order})"

    def __str__(self) -> str:
        return self.name

    @property
    def is_abelian(self) -> bool:
        """Is this ring Abelian?"""
        return isinstance(self, AbelianGroup) or self._group.is_abelian

    @property
    def is_semisimple(self) -> bool:
        """Is this ring semisimple?"""
        return bool(self.group.order % self.field.characteristic)

    def regular_lift(self, member: GroupMember) -> npt.NDArray[np.int_]:
        """Lift a group member to its regular representation."""
        return self.group.regular_lift(member)

    def lift(self, member: GroupMember) -> npt.NDArray[np.int_]:
        """Lift a group member to a representation by an orthogonal matrix."""
        return self.group.lift(member)

    @property
    def zero(self) -> RingMember:
        """Zero (additive identity) element."""
        return RingMember(self)

    @property
    def one(self) -> RingMember:
        """One (multiplicative identity) element."""
        return RingMember(self, self.group.identity)

    def get_primitive_central_idempotents(self) -> tuple[RingMember, ...]:
        """Get the primitive central idempotents of this ring.

        Primitive central idempotents of a ring are nonzero elements that:
        - square to themselves (they are idempotent),
        - commute with all other elements of the ring (they lie in the ring's center), and
        - cannot be decomposed into a sum of two nonzero orthogonal idempotents.
        Two idempotents g, h are orthogonal if g * h = h * g 0.

        Intuitively, primitive central idempotents idempotents act like projectors onto orthogonal
        simple components of a ring.

        See https://en.wikipedia.org/wiki/Idempotent_(ring_theory).
        """
        if not self.is_semisimple:
            raise ValueError("Only semisimple rings have primitive central idempotents")
        idempotents_as_tuples = external.groups.get_primitive_central_idempotents(
            self.group.to_gap_group(), self.field.order
        )
        idempotents = []
        for idempotent in idempotents_as_tuples:
            # collect terms, coercing cycles into elements of self.group
            terms = [
                (
                    self.field(coefficient),
                    GroupMember(cycles) * self.group.identity
                    if cycles != ((),)  # the empty cycle needs special treatment
                    else self.group.identity,
                )
                for coefficient, cycles in idempotent
            ]
            idempotents.append(RingMember(self, *terms))
        return tuple(idempotents)


class RingMember:
    """An element of the algebra of a group G over a finite field F_q.

    Each RingMember x is a sum of group members with coefficients in the field:
    x = sum_{g in G} x_g g, with each x_g in F_q.
    """

    _ring: GroupRing
    _vec: collections.defaultdict[GroupMember, galois.FieldArray]

    def __init__(
        self,
        ring: GroupRing | Group,
        *terms: GroupMember | tuple[int | galois.FieldArray, GroupMember],
    ) -> None:
        self._ring = ring if isinstance(ring, GroupRing) else GroupRing(ring)
        self._vec = collections.defaultdict(lambda: self.field(0))
        for term in terms:
            value, member = (1, term) if isinstance(term, GroupMember) else term
            self._vec[member] += self.field(value)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RingMember)
            and self._ring == other._ring
            and all(self._vec.get(member, 0) == other._vec.get(member, 0) for member in self._vec)
            and all(self._vec.get(member, 0) == other._vec.get(member, 0) for member in other._vec)
        )

    def __bool__(self) -> bool:
        return any(x_g for x_g in self._vec.values())

    def __iter__(self) -> Iterator[tuple[galois.FieldArray, GroupMember]]:
        for gg, x_g in self._vec.items():
            yield x_g, gg

    def __add__(self, other: int | GroupMember | RingMember) -> RingMember:
        if isinstance(other, int):
            return self + other * self.ring.one

        if isinstance(other, GroupMember):
            new_element = self.copy()
            new_element._vec[other] += self.field(1)
            return new_element

        if isinstance(other, RingMember):
            new_element = self.copy()
            for val, member in other:
                new_element._vec[member] += val
            return new_element

        return NotImplemented  # pragma: no cover

    def __sub__(self, other: RingMember | GroupMember | int) -> RingMember:
        return self + (-1) * other

    def __radd__(self, other: GroupMember) -> RingMember:
        return self + other

    def __mul__(self, other: int | GroupMember | RingMember) -> RingMember:
        if isinstance(other, int):
            # multiply coefficients by 'other'
            new_element = self.ring.zero
            for val, member in self:
                new_element._vec[member] = val * other
            return new_element

        if isinstance(other, GroupMember):
            # multiply group members by 'other'
            new_element = self.ring.zero
            for val, member in self:
                new_element._vec[member * other] = val
            return new_element

        if isinstance(other, RingMember):
            # collect and multiply pairs of terms from 'self' and 'other'
            new_element = self.ring.zero
            for (x_a, aa), (y_b, bb) in itertools.product(self, other):
                new_element._vec[aa * bb] += x_a * y_b
            return new_element

        return NotImplemented  # pragma: no cover

    def __rmul__(self, other: int | GroupMember) -> RingMember:
        if isinstance(other, int):
            return self * other

        if isinstance(other, GroupMember):
            new_element = self.ring.zero
            for val, member in self:
                new_element._vec[other * member] = val
            return new_element

        return NotImplemented  # pragma: no cover

    def __neg__(self) -> RingMember:
        return self * (-1)

    def __pow__(self, power: int) -> RingMember:
        return functools.reduce(operator.mul, [self] * power, self.ring.one)

    def copy(self) -> RingMember:
        """Copy of self."""
        element = self.ring.zero
        for val, member in self:
            element._vec[member] = copy.deepcopy(val)
        return element

    @property
    def ring(self) -> GroupRing:
        """Base ring of this algebra."""
        return self._ring

    @property
    def group(self) -> Group:
        """Base group of this algebra."""
        return self.ring.group

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this algebra."""
        return self.ring.field

    def lift(self) -> galois.FieldArray:
        """Lift this element using the underlying group representation."""
        return sum(
            (val * self.ring.lift(member) for val, member in self if val),
            start=self.field.Zeros([self.group.lift_dim] * 2),
        )

    def regular_lift(self) -> galois.FieldArray:
        """Lift this element using the regular representation of its base group."""
        return sum(
            (val * self.ring.regular_lift(member) for val, member in self if val),
            start=self.field.Zeros([self.group.order] * 2),
        )

    @property
    def T(self) -> RingMember:
        """Transpose of this element.

        If this element is x = sum_{g in G) x_g g, return x.T = sum_{g in G} x_g g.T, where g.T is
        the group member for which the lift L(g.T) = L(g).T.  The fact that group members get lifted
        to orthogonal matrices implies that g.T = ~g = g**-1.
        """
        new_element = self.ring.zero
        for val, member in self:
            new_element._vec[~member] = val
        return new_element

    def inverse(self) -> RingMember | None:
        """The inverse of this RingMember, if it exists."""
        self_vec = {gg: x_g for gg, x_g in self._vec.items() if x_g}
        if not self_vec:
            return None
        if len(self_vec) == 1:
            gg, x_g = next(iter(self_vec.items()))
            return RingMember(self.ring, (x_g**-1, gg**-1))
        try:
            matrix = self.regular_lift()
            matrix_inv = np.linalg.inv(matrix).view(self.field)
            return RingMember.from_vector(self.ring, matrix_inv[:, 0])
        except np.linalg.LinAlgError:
            return None

    @classmethod
    def from_vector(cls, ring: GroupRing | Group, vector: npt.NDArray[np.int_]) -> RingMember:
        """Construct a group algebra element from vector of coefficients, (x_g : g in G)."""
        group = ring.group if isinstance(ring, GroupRing) else ring
        terms = [(int(x_g), gg) for x_g, gg in zip(vector, group.generate()) if x_g]
        return RingMember(ring, *terms)

    def to_vector(self) -> galois.FieldArray:
        """Convert this group algebra element into a vector of coefficients, (x_g : g in G)."""
        vector = self.field.Zeros(self.group.order)
        for val, member in self:
            vector[self.group.index(member)] = val
        return vector


class Element(RingMember):  # pragma: no cover
    """Deprecated alias for RingMember."""

    def __getattribute__(self, name: str) -> Any:
        warnings.warn(
            f"{Element} is DEPRECATED; use {RingMember} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__getattribute__(name)


################################################################################
# RingArray: RingMember-valued array

NestedSequence = Sequence[Union[object, Sequence["NestedSequence"]]]


class RingArray(npt.NDArray[np.object_]):
    """Array whose entries are members of a GroupRing."""

    _ring: GroupRing

    def __new__(
        cls,
        data: npt.NDArray[np.object_] | NestedSequence,
        ring: GroupRing | Group | None = None,
    ) -> RingArray:
        array = np.asarray(data, dtype=object).view(cls)
        ring = GroupRing(ring) if isinstance(ring, Group) else ring

        # identify the base group for this RingArray
        for value in array.ravel():
            if not isinstance(value, RingMember):
                raise ValueError(
                    "Requirement failed: all entries of a RingArray must be RingMember-valued\n"
                    "Try building an array with RingArray.build(...)"
                )
            else:
                if not (ring is None or ring == value.ring):
                    raise ValueError("Inconsistent base rings provided for a RingArray")
                ring = value.ring

        if ring is None:
            raise ValueError("Cannot determine the underlying ring for a RingArray")
        array._ring = ring

        return array

    def __array_finalize__(self, obj: npt.NDArray[np.object_] | None) -> None:
        """Propagate metadata to newly constructed arrays."""
        setattr(self, "_ring", getattr(obj, "_ring", None))

    def __array_function__(
        self,
        func: Any,
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> RingArray | None:
        """Intercept array operations to ensure RingArray compatibility."""
        rings = {self._ring} | {x._ring for x in args if isinstance(x, RingArray)}
        if len(rings) > 1:
            raise ValueError("Cannot perform operations on RingArrays with different base rings")
        args = tuple(x.view(np.ndarray) if isinstance(x, RingArray) else x for x in args)
        result = super().__array_function__(func, types, args, kwargs)
        if isinstance(result, np.ndarray):
            result = result.view(RingArray)
            setattr(result, "_ring", next(iter(rings), None))
        return result

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *inputs: npt.NDArray[np.object_],
        **kwargs: object,
    ) -> RingArray | None:
        """Intercept array operations to ensure RingArray compatibility."""
        rings = {self._ring} | {x._ring for x in inputs if isinstance(x, RingArray)}
        if len(rings) > 1:
            raise ValueError("Cannot perform operations on RingArrays with different base rings")
        inputs = tuple(x.view(np.ndarray) if isinstance(x, RingArray) else x for x in inputs)
        result = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if isinstance(result, np.ndarray):
            result = result.view(RingArray)
            setattr(result, "_ring", next(iter(rings), None))
        return result

    @property
    def ring(self) -> GroupRing:
        """Base ring of this RingArray."""
        return self._ring

    @property
    def group(self) -> Group:
        """Base group of this RingArray."""
        return self.ring.group

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this RingArray."""
        return self.ring.field

    def regular_lift(self) -> galois.FieldArray:
        """Block matrix obtained by a regular lift of each entry of this RingArray."""
        assert self.ndim == 2
        blocks = [[val.regular_lift() for val in row] for row in self]
        return np.block(blocks).view(self.field)

    def lift(self) -> galois.FieldArray:
        """Block matrix obtained by lifting each entry of this RingArray."""
        assert self.ndim == 2
        blocks = [[val.lift() for val in row] for row in self]
        return np.block(blocks).view(self.field)

    def __invert__(self) -> RingArray:
        """Transpose the entries of this RingArray."""
        vals = [val.T for val in self.ravel()]
        return RingArray(np.array(vals, dtype=object).reshape(self.shape), self.ring)

    @property
    def T(self) -> RingArray:
        """Transpose of this RingArray, which also transposes every array entry."""
        return (~self).transpose()

    @staticmethod
    def build(
        data: npt.NDArray[np.object_ | np.int_] | NestedSequence, ring: GroupRing | Group
    ) -> RingArray:
        """Construct a RingArray.

        The constructed array is built from:
        - an array populated by
            (a) ring members,
            (b) group members, or
            (c) integers, and
        - a ring (or group, inducing a group algebra over GF(2)).
        Integers and group members are cast as members of the ring.
        """
        array = np.asarray(data)
        group = ring.group if isinstance(ring, GroupRing) else ring

        def as_ring_member(value: RingMember | GroupMember | int) -> RingMember:
            """Elevate a value to an element of the ring."""
            if isinstance(value, RingMember):
                return value
            if isinstance(value, GroupMember):
                return RingMember(ring, value)
            return RingMember(ring, (value, group.identity))

        vals = [as_ring_member(value) for value in array.ravel()]
        return RingArray(np.array(vals).reshape(array.shape), ring)

    @classmethod
    def from_field_array(cls, ring: GroupRing | Group, array: npt.NDArray[np.int_]) -> RingArray:
        """Construct a RingArray from an array of coefficients (in a finite field) for each entry.

        The input array should have shape (..., ring.group.order), such that if array.ndim == 3, for
        example, then array[a, b, :] is the vector of coefficients for the RingMember at the
        constructed ring_array[a, b].
        """
        group = ring.group if isinstance(ring, GroupRing) else ring
        assert array.shape[-1] == group.order
        vals = [RingMember.from_vector(ring, entry) for entry in array.reshape(-1, group.order)]
        return RingArray(np.array(vals, dtype=object).reshape(array.shape[:-1]))

    def to_field_array(self) -> galois.FieldArray:
        """Convert a RingArray into an array of coefficients (in a finite field) for each entry.

        This method is the inverse of RingArray.from_field_array.
        """
        vals = [val.to_vector() for val in self.ravel()]
        return np.asarray(vals).reshape(self.shape + (self.group.order,)).view(self.field)

    @classmethod
    def from_field_vector(cls, ring: GroupRing | Group, vector: npt.NDArray[np.int_]) -> RingArray:
        """Construct a RingArray from a vector of coefficients."""
        assert vector.ndim == 1
        group = ring.group if isinstance(ring, GroupRing) else ring
        assert vector.size % group.order == 0
        return RingArray.from_field_array(ring, vector.reshape(-1, group.order))

    def to_field_vector(self) -> galois.FieldArray:
        """Convert a RingArray into a vector of coefficients."""
        assert self.ndim == 1
        vals = [val.to_vector() for val in self.ravel()]
        return np.asarray(vals).ravel().view(self.field)

    def null_space(self, *, force_heuristic: bool = True) -> RingArray:
        """Construct a matrix of null-space row vectors for this RingArray.

        The transpose of the null-space matrix is annihilated by this RingArray, such that
        np.any(self @ self.null_space().T) is False
        """
        assert self.ndim == 2

        # field-valued null vectors of matrix.regular_lift() correspond to ring-valued null vectors
        # of the matrix via conversion with RingArray.from_field_vector <> RingArray.to_field_vector
        null_field_vectors = self.regular_lift().null_space()

        # collect ring-valued null row vectors (that is, transposed null column vectors)
        null_vectors = ~RingArray.from_field_array(
            self.ring,
            null_field_vectors.reshape(len(null_field_vectors), -1, self.group.order),
        )

        return null_vectors.row_reduce(force_heuristic=force_heuristic)

    def row_reduce(self, *, force_heuristic: bool = True) -> RingArray:
        """Compute the reduced row Echelon form of this RingArray, or the closest we can get to it.

        Use a series of heuristics for (possibly only partial) row reduction.  Every row in the
        returned RingArray is guaranteed to satisfy one the following:
        (a) there is some column at which the row is 1 and all other rows are 0, or
        (b) all entries of the row are non-invertible.
        Moreover, all rows of the returned RingArray are linearly independent and nonzero.
        """
        assert self.ndim == 2

        if self.ring.is_semisimple and not force_heuristic:
            return self._exact_row_reduce()

        if not force_heuristic:
            warnings.warn(
                "RingArray.row_reduce only supports exact row reduction for semisimple group"
                " algebras, for which the field.characteristic does not divide the group.order."
                "  Using heuristics for (possibly only partial) row reduction instead.",
            )
        return self._heuristic_row_reduce()

    def _exact_row_reduce(self) -> RingArray:
        """Perform exact row reduction based on the Wedderburn-Artin decomposition of the base ring.

        A semisimple ring can be decomposed into a direct product of simple rings, which are in turn
        isomorphic to matrix algebras over finite fields.  This method thereby row-reduces a
        RingArray over a semisimple ring by
        (a) decomposing the RingArray into its simple components,
        (b) "lifting" each component to a matrix over a finite field,
        (c) row-reducing these matrices, and
        (d) mapping back to a RingArray over the original semisimple ring.
        At least, that's the plan.  It has yet to be implemented.
        """
        assert self.ring.is_semisimple
        raise NotImplementedError(
            "We only aspire to perform exact row reduction over semisimple rings :("
        )

    def _heuristic_row_reduce(self) -> RingArray:
        """Use heuristics for (possibly only partial) row reduction.

        Warning: this method is unoptimized.  There is a lot of room for speeding things up.
        """

        # greedily convert invertible entries into "pivots" that are uniquely nonzero in some column
        matrix = self._reduce_rows_with_invertible_entries()

        # split matrix into rows with pivots, and all other rows
        pivot_matrix, non_pivot_matrix = matrix._split_by_pivots()

        if non_pivot_matrix.size:
            # identify a minimal basis for the span of the non-pivot rows
            non_pivot_matrix = non_pivot_matrix._remove_linearly_dependent_rows()

        return np.vstack([pivot_matrix, non_pivot_matrix]).view(RingArray)

    def _reduce_rows_with_invertible_entries(self, *, restart_call: bool = False) -> RingArray:
        """Row-reduce greedily using invertible entries.

        Loop over every row.  If that row contains an invertible entry, normalize the row by this
        entry's inverse, and zero out all other rows at the corresponding column by subtracting off
        an appropriate multiple of this row (as you would with ordinary Gaussian elimination).
        """
        assert self.ndim == 2
        rows: slice | list[int]

        matrix = self if restart_call else self.copy()
        num_rows, num_cols = self.shape

        row_reductions_made = False  # did we perform row reductions?
        noninvertible_rows = []  # which rows have only non-invertible entries?
        for row in range(num_rows):
            row_vector = matrix[row]

            # look for an invertible entry in this row
            for col, entry in enumerate(row_vector):
                if inverse := entry.inverse():
                    break

            """
            If we found an invertible entry,
            - multiply this row by the inverse, so that this entry is 1, and
            - zero out the corresponding column in all other rows.
            """
            if inverse:
                # "normalize" the entry to 1, and zero out its column in all other rows
                new_vector = inverse * row_vector
                matrix[row] = new_vector
                for rows in [slice(None, row), slice(row + 1, num_rows)]:
                    matrix[rows] = matrix[rows] - matrix[rows, col, None] * new_vector[None, :]
                row_reductions_made = True

            else:
                noninvertible_rows.append(row)

        if row_reductions_made and noninvertible_rows:
            # some non-invertible entries may have become invertible, so try row-reducing again
            rows = [row for row in noninvertible_rows if np.any(matrix[row])]
            matrix[rows].view(RingArray)._reduce_rows_with_invertible_entries(restart_call=True)

        return matrix

    def _split_by_pivots(self, *, allow_non_units: bool = True) -> tuple[RingArray, RingArray]:
        """Split into a matrix with all "pivot" rows, and a matrix with all other nonzero rows.

        "Pivot" rows are those that are uniquely nonzero in some column.  If allow_non_unit is
        False, this nonzero entry is also required to be a unit (invertible entry) of the base ring
        (group algebra) of this RingArray.
        """
        self_as_bool = self.astype(bool)
        pivot_rows = np.zeros(len(self), dtype=bool)
        pivot_cols = np.sum(self_as_bool, axis=0) == 1
        for col in np.argwhere(pivot_cols):
            row = np.argwhere(self_as_bool[:, col])[0][0]
            pivot_rows[row] = allow_non_units or self[row, col][0].inverse()
        pivot_matrix = self[pivot_rows].view(RingArray)
        non_pivot_matrix = self[~pivot_rows].view(RingArray)
        return pivot_matrix, non_pivot_matrix[np.any(non_pivot_matrix, axis=1)]

    def _remove_linearly_dependent_rows(self) -> RingArray:
        """Remove rows that can be expressed as ring-linear combinations of others.

        Due to peculiarities of working with modules (the generalization of a vector space when
        working over rings, as opposed to fields), we have to start by considering all rows in the
        RingArray, and checking individually whether each row lies in the span of the rest; if so,
        we remove that row.  "Trimming down" to a minimal basis, as opposed to "building one up"
        by accumulating linearly independent row vectors, is necessary because rows may have
        nontrivial annihilators, which is to say that a row v may have a ring element r for which
        r * v = 0 even though r and v are nonzero.  It is therefore possible, for examle, for a row
        v to lie in the left-ring-linear span of another row w (v = r * w for some r), but not the
        other way around (there is no r for which w = r * v).
        """
        assert self.ndim == 2

        # expand row vectors over the ring into row vectors over the field
        field_vectors = self.to_field_array().reshape(len(self), -1).view(self.field)

        # row-reduce over the field to find rows that are field-linearly independent
        field_vectors = field_vectors.row_reduce()
        field_vectors = field_vectors[np.any(field_vectors, axis=1)]  # remove all-zero rows
        ring_matrix = RingArray.from_field_array(
            self.ring, field_vectors.reshape(len(field_vectors), -1, self.group.order)
        )

        """
        Invert (transpose) the entries of the matrix to make them right-acting (that is, to
        transform coefficients that are to the left of each row vector), and lift to a matrix over
        the field.
        """
        field_matrix = (~ring_matrix).view(RingArray).regular_lift()

        # throw out rows that can be expressed as ring-linear combinations of other rows
        rows_to_keep = np.ones((len(ring_matrix), self.group.order), dtype=bool)
        for row in range(len(ring_matrix) - 1, -1, -1):
            rows_to_keep[row, :] = False
            linear_system = np.vstack([field_matrix[rows_to_keep.ravel()], field_vectors[row]])
            linear_system_rref = linear_system.T.view(self.field).row_reduce()
            for row_rref in np.argwhere(linear_system_rref[:, -1]):
                if not np.any(linear_system_rref[row_rref, :-1]):
                    rows_to_keep[row, :] = True
                    break

        return ring_matrix[rows_to_keep[:, 0]].view(RingArray)


class Protograph(RingArray):  # pragma: no cover
    """Deprecated alias for RingArray."""

    def __getattribute__(self, name: str) -> Any:
        warnings.warn(
            f"{Protograph} is DEPRECATED; use {RingArray} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__getattribute__(name)


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
    def to_ring_array(data: npt.NDArray[np.int_] | NestedSequence) -> RingArray:
        """Convert a matrix of 0s and 1s into a RingArray over the trivial group."""
        field = type(data).order if isinstance(data, galois.FieldArray) else None
        group = TrivialGroup()
        ring = GroupRing(group, field)
        zero = ring.zero
        one = ring.one

        array = np.asarray(data, dtype=int)
        terms = [val * one if val else zero for val in array.ravel()]
        return RingArray(np.array(terms, dtype=object).reshape(array.shape))


class AbelianGroup(Group):
    """Direct product of cyclic groups of the specified orders.  See CyclicGroup for more info.

    By default, an AbelianGroup member of the form ∏_i g_i^{a_i}, where {g_i} are the generators of
    the group, gets lifted to a Kronecker product ⨂_i L(g_i)^{a_i}.  If an AbelianGroup is
    initialized with direct_sum=True, the group members get lifted to a direct sum ⨁_i L(g_i)^{a_i}.
    """

    def __init__(self, *orders: int, direct_sum: bool = False) -> None:
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
    written as g^p for an integer power p in {0, 1, ..., R-1}.  The member g^p can be represented by
    (that is, lifted to) an R×R "shift matrix", or the identity matrix with all rows shifted down
    (equivalently, all columns shifted right) by p.  That is, the lift L(g^p) acts on a standard
    basis vector <i| as <i| L(g^p) = < i + p mod R |.
    """

    def __init__(self, order: int) -> None:
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
    """Quaternion group: 1, i, j, k, -1, -i, -j, -k."""

    def __init__(self) -> None:
        table = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 4, 3, 6, 5, 0, 7, 2],
            [2, 7, 4, 1, 6, 3, 0, 5],
            [3, 2, 5, 4, 7, 6, 1, 0],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [5, 0, 7, 2, 1, 4, 3, 6],
            [6, 3, 0, 5, 2, 7, 4, 1],
            [7, 6, 1, 0, 3, 2, 5, 4],
        ]

        def integer_lift(member: int) -> npt.NDArray[np.int_]:
            """Representation from https://en.wikipedia.org/wiki/Quaternion_group."""
            assert 0 <= member < 8
            sign = 1 if member < 4 else -1
            base = member % 4  # +/- 1, i, j, k
            zero = np.zeros((2, 2), dtype=int)
            unit = np.eye(2, dtype=int)
            imag = np.array([[0, -1], [1, 0]], dtype=int)
            if base == 0:  # +/- 1
                blocks = [[unit, zero], [zero, unit]]
            elif base == 1:  # +/- i
                blocks = [[imag, zero], [zero, -imag]]
            elif base == 2:  # +/- j
                blocks = [[zero, -unit], [unit, zero]]
            else:  # if base == 3; +/- k
                blocks = [[zero, -imag], [-imag, zero]]
            return sign * (np.block(blocks).T % 3).view(galois.GF(3))

        group = Group.from_table(table, integer_lift=integer_lift)
        super()._init_from_group(group, name=QuaternionGroup.__name__)


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
    """Special linear group (SL): square matrices with determinant 1."""

    _dimension: int
    _field: type[galois.FieldArray]

    def __init__(self, dimension: int, field: int | None = None, linear_rep: bool = True) -> None:
        self._name = f"SL({dimension},{field})"
        self._dimension = dimension
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)

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
                return np.vstack(cols, dtype=int).T

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
        dimension: int, field: int | None = None
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Generating matrices for the Special Linear group, based on arXiv:2201.09155."""
        base_field = galois.GF(field or DEFAULT_FIELD_ORDER)
        minus_one = -base_field(1)
        gen_w = minus_one * np.diag(np.ones(dimension - 1, dtype=int), k=-1).view(base_field)
        gen_w[0, -1] = 1
        gen_x = base_field.Identity(dimension)
        if base_field.order <= 3:
            gen_x[0, 1] = 1
        else:
            gen_x[0, 0] = base_field.primitive_element
            gen_x[1, 1] = base_field.primitive_element**-1
            gen_w[0, 0] = minus_one
        return gen_x, gen_w

    @staticmethod
    def iter_mats(dimension: int, field: int | None = None) -> Iterator[galois.FieldArray]:
        """Iterate over all elements of SL(dimension, field)."""
        base_field = galois.GF(field or DEFAULT_FIELD_ORDER)
        for vec in itertools.product(base_field.elements, repeat=dimension**2):
            mat = np.reshape(vec, (dimension, dimension)).view(base_field)
            if np.linalg.det(mat) == 1:
                yield mat


class ProjectiveSpecialLinearGroup(Group):
    """Projective special linear group (PSL = SL/center).

    Here "center" is the subgroup of SL that commutes with all elements of SL.  Specifically, every
    element in the center of SL is a scalar multiple of the identity matrix I.  In the case of
    SL(d,q) (d×d matrices over F_q with determinant 1), the determinant of scalar*I is scalar**d,
    which is only contained in SL(d,q) if scalar**d == 1.

    Altogether, we construct PSL(d,q) by SL(d,q) mod [d-th roots of unity over F_q].
    """

    _dimension: int
    _field: type[galois.FieldArray]

    def __init__(self, dimension: int, field: int | None = None, linear_rep: bool = True) -> None:
        self._name = f"PSL({dimension},{field})"
        self._dimension = dimension
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)

        if linear_rep:
            # Construct a linear representation of this group, in which group elements permute
            # elements of the vector space that the generating matrices act on.

            # identify multiplicative roots of unity
            num_roots = math.gcd(self.dimension, self.field.order - 1)
            primitive_root = self.field.primitive_element ** ((self.field.order - 1) // num_roots)
            roots = [primitive_root**kk for kk in range(num_roots)]

            # Identify the target space that group members (as matrices) act on:
            # all nonzero vectors, modded out by roots of unity.
            target_orbits = [
                frozenset([(root * self.field(vec)).tobytes() for root in roots])
                for vec in itertools.product(range(self.field.order), repeat=self.dimension)
            ]
            del target_orbits[0]  # remove the orbit of the zero vector
            target_space = [next(iter(orbit)) for orbit in set(target_orbits)]

            # identify how the generators permute elements of the target space
            generators = []
            for member in SpecialLinearGroup.get_generating_mats(self.dimension, self.field.order):
                perm = np.empty(len(target_space), dtype=int)
                for index, vec_bytes in enumerate(target_space):
                    vec = np.frombuffer(vec_bytes, dtype=np.uint8).view(self.field)
                    next_orbit = [root * member @ vec for root in roots]
                    next_vec = next((vec for vec in next_orbit if vec.tobytes() in target_space))
                    next_index = target_space.index(next_vec.tobytes())
                    perm[index] = next_index
                generators.append(GroupMember(perm))

            # construct a lift identical to that for the linear representation of SL
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
                return np.vstack(cols, dtype=int).T

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
        dimension: int, field: int | None = None
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Generating matrices of PSL, constructed out of the generating matrices of SL."""
        base_field = galois.GF(field or DEFAULT_FIELD_ORDER)
        gen_x, gen_w = SpecialLinearGroup.get_generating_mats(dimension, field)
        if base_field.order == 2:
            return gen_x, gen_w
        return (
            np.kron(np.linalg.inv(gen_x), gen_x).view(base_field),
            np.kron(np.linalg.inv(gen_w), gen_w).view(base_field),
        )

    @staticmethod
    def iter_mats(dimension: int, field: int | None = None) -> Iterator[galois.FieldArray]:
        """Iterate over all elements of PSL(dimension, field)."""
        field = field or DEFAULT_FIELD_ORDER
        base_field = galois.GF(field)
        num_roots = math.gcd(dimension, base_field.order - 1)
        primitive_root = base_field.primitive_element ** ((base_field.order - 1) // num_roots)
        roots = [primitive_root**k for k in range(dimension)]
        orbits = [
            frozenset([(root * mat).tobytes() for root in roots])
            for mat in SpecialLinearGroup.iter_mats(dimension, field)
        ]
        for orbit in set(orbits):
            yield np.frombuffer(next(iter(orbit)), dtype=np.uint8).view(base_field)


SL = SpecialLinearGroup
PSL = ProjectiveSpecialLinearGroup
