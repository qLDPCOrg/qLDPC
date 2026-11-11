# SPDX-License-Identifier: Apache-2.0

"""Module for abstract algebra: groups and representations thereof.

All groups in this module are finite and represented under the hood as a SymPy PermutationGroup, or
a subgroup of the symmetric group.  The class of group members, GroupMember, subclasses the SymPy
Permutation class.

.. warning::
    This module does not promise to be performant.  If you need to do heavy numerical abstract
    algebra, you're probably better served by GAP or MAGMA (or maybe SageMath).

This module represents group members by matrices over a finite field via a lift L, a homomorphism
with L(g . h) = L(g) @ L(h).  The default lift is the regular representation, whose matrices are
permutation matrices and hence orthogonal, so that L(~g) = L(g).T for the inverse ~g = g**-1 of a
group member g (see RingMember.T).  Some custom lifts, such as those of SpecialLinearGroup,
ProjectiveSpecialLinearGroup, GeneralLinearGroup, ProjectiveGeneralLinearGroup, and QuaternionGroup,
are not orthogonal, so this identity does not hold for them.
"""

from __future__ import annotations

import functools
import itertools
import math
import operator
from collections.abc import Callable, Iterable, Iterator, Sequence
from random import Random
from typing import Any, ClassVar, cast

import galois
import numpy as np
import numpy.typing as npt
import scipy.linalg
import sympy
import sympy.combinatorics as comb
import sympy.core

from qldpc.math import resolve_field

from .monomials import _is_integer

NestedSequence = Sequence[object | Sequence["NestedSequence"]]


####################################################################################################
# groups and group members


class GroupMember(comb.Permutation):
    """Wrapper for SymPy Permutation class.

    Supports sorting permutations (lexicographically, padding with fixed points), and taking their
    tensor product.
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
        """Compose permutations, and defer other products, such as with ring members."""
        if isinstance(other, comb.Permutation):
            return GroupMember.from_sympy(super().__mul__(other))
        return NotImplemented

    def __rmul__(self, other: object) -> Any:
        """Compose with permutations on the left, and defer other products, such as with integers.

        SymPy reads an integer n on the left of a permutation as the identity on n + 1 points, so
        without this method ``2 * g`` would be ``g``.
        """
        if isinstance(other, comb.Permutation):
            return GroupMember.from_sympy(comb.Permutation.__mul__(other, self))
        return NotImplemented

    def __add__(self, other: object) -> Any:
        """Defer sums, such as with ring members.

        A SymPy permutation plus an integer is the permutation whose rank is shifted by the integer,
        which this class does not support.
        """
        return NotImplemented

    def __lt__(self, other: comb.Permutation) -> bool:
        """Compare array forms lexicographically, padding the shorter one with fixed points.

        Permutations that differ only in trailing fixed points are therefore neither smaller nor
        larger than each other.  Sorting keeps them in their input order, although they are not
        equal (==).  On permutations of the same size, this is the order of SymPy ranks.  It also
        covers the identity of the trivial group, which acts on no points.
        """
        if not isinstance(other, comb.Permutation):
            return NotImplemented
        self_form, other_form = GroupMember._get_padded_array_forms(self, other)
        return self_form < other_form

    def __gt__(self, other: comb.Permutation) -> bool:
        """Compare permutations in the order of __lt__.

        Python also calls this method for ``p < member`` when p is a SymPy permutation.
        """
        if not isinstance(other, comb.Permutation):
            return NotImplemented
        self_form, other_form = GroupMember._get_padded_array_forms(self, other)
        return self_form > other_form

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
        This convention ensures that this lift is a homomorphism on SymPy Permutation objects, which
        is to say that ``(p * q).to_matrix() = p.to_matrix() @ q.to_matrix()``.
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

    @staticmethod
    def _get_padded_array_forms(
        first: comb.Permutation, second: comb.Permutation
    ) -> tuple[list[int], list[int]]:
        """Get the array forms of two permutations, padding the shorter one with fixed points."""
        size = max(first.size, second.size)
        return (
            first.array_form + list(range(first.size, size)),
            second.array_form + list(range(second.size, size)),
        )


Lift = Callable[[GroupMember], npt.NDArray[np.int_]]
IntegerLift = Callable[[int], npt.NDArray[np.int_]]
GenerateFunc = Callable[[], Iterator[comb.Permutation]]


class Group:
    """Base class for a finite group.

    Under the hood, a Group is represented by a SymPy PermutationGroup, and group members are
    represented by SymPy permutations.

    A group naturally comes equipped with a "regular lift" that maps each group member to a
    permutation matrix corresponding to the regular representation of the group.  The regular
    representation of a group represents group members by how they act on the group itself.  See
    https://en.wikipedia.org/wiki/Regular_representation.

    A group may additionally be equipped with a custom lift to a matrix over a finite field, for
    which the group action corresponds to matrix multiplication.  Custom lifts are not required to
    be orthogonal (some, such as SpecialLinearGroup, are not).  If no lift is provided,
    group.lift(member) will default to the regular lift of the group.

    Methods that take a group member, such as ``in``, ``index``, and lifts, read it on as many
    points as the group acts on: they pad a member on fewer points with fixed points, and drop fixed
    points beyond the degree of the group.
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
        """Do these groups have the same SymPy permutation group, ignoring their lifts?

        Unlike ``==``, this compares only the underlying SymPy permutation groups, so groups that
        differ only in their custom lifts, such as the same group built twice, are equivalent.
        SymPy compares permutation groups by their generators, so groups with the same members but
        different generators are not equivalent.
        """
        return isinstance(other, Group) and self._group == other._group

    @property
    def name(self) -> str:
        """A name for this group, which is not required to uniquely identify the group."""
        return self._name or type(self).__name__

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
        # every group contains the identity, which SymPy does not find in a group of order 1
        return member.is_Identity or self._at_degree(member) in self._group

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

    def generate(self) -> Iterator[GroupMember]:
        """Iterate over all group members, in the order of their indices (see ``Group.index``)."""
        # Keep iteration lazy.  ``index`` still materializes the lookup table when random access is
        # requested, but callers that stop early should not pay the memory cost of the full group.
        generate = self._generate_func or self._sympy_generate
        yield from map(GroupMember.from_sympy, generate())

    def index(self, member: GroupMember) -> int:
        """The index of a GroupMember in this group."""
        index = self._members.get(self._at_degree(member))
        if not isinstance(index, int):
            raise TypeError(f"Member {member} not in group {self}")
        return index

    def __mul__(self, other: Group) -> Group:
        """Direct product of two groups."""
        return Group.from_sympy(self._group * other._group)

    def __pow__(self, power: int) -> Group:
        """Direct product of self multiple times."""
        if power < 1:
            raise ValueError(f"A group can only be raised to a positive power (got {power})")
        return Group.product(self, repeat=power)

    @staticmethod
    def product(*groups: Group, repeat: int = 1) -> Group:
        """Direct product of Groups.

        The ``repeat`` argument repeats the entire sequence of factors.
        """
        if not groups or repeat < 1:
            raise ValueError("A product of groups needs at least one factor")
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
        """A random element of this group."""
        return self._random_member(_get_rng(seed))

    def regular_lift(self, member: GroupMember, *, right: bool = False) -> npt.NDArray[np.int_]:
        """Lift a group member to its regular representation.

        By default, this method lifts a group member to its left-regular representation.
        Specifically, if ``Vec : G -> F^{|G|}`` maps group members to standard basis vectors and
        ``g,h ∈ G``, then

            ``G.regular_lift(g) @ Vec(h) = Vec(g·h)``.

        If ``right is True``, this method lifts a group member to its right-regular
        anti-representation, defined by

            ``G.regular_lift(g, right=True) @ Vec(h) = Vec(h·g)``.

        The right-regular representation of a group member is then

            ``G.regular_lift(g, right=True).T = G.regular_lift(~g, right=True)``,

        where ``~g = g**-1`` is the inverse of g.

        See:

        - https://en.wikipedia.org/wiki/Regular_representation
        """
        member = self._at_degree(member)
        if (member, right) not in self._regular_lift_cache:
            matrix = np.zeros([self.order] * 2, dtype=int)
            for ii, hh in enumerate(self.generate()):
                jj = self.index(hh * member) if right else self.index(member * hh)
                matrix[jj, ii] = 1
            self._regular_lift_cache[member, right] = matrix
        return self._regular_lift_cache[member, right].copy()

    def adjoint_lift(self, member: GroupMember) -> npt.NDArray[np.int_]:
        r"""Lift a group member to its adjoint representation.

        The adjoint representation captures how group members get transformed by conjugation.  If
        ``Vec : G -> F^{|G|}`` lifts group members to standard basis vectors and ``g,h ∈ G``, then

            ``adjoint_lift(g) @ Vec(h) = Vec(g·h·~g)``,

        where ``~g = g**-1`` is the inverse of g.

        If the group is abelian, the adjoint lift of every group member is the identity matrix.
        """
        member = self._at_degree(member)
        if member not in self._adjoint_lift_cache:
            self._adjoint_lift_cache[member] = (
                self.regular_lift(member) @ self.regular_lift(member, right=True).T
            )
        return self._adjoint_lift_cache[member].copy()

    @functools.cached_property
    def inversion_matrix(self) -> npt.NDArray[np.int_]:
        """The matrix that maps any group member ``g ∈ G`` to its inverse ``~g = g**-1``.

        If ``Vec : G -> F^{|G|}`` lifts group members to standard basis vectors and ``g ∈ G``, then

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

        If ``right=True``, lift to an anti-representation, for which

            ``self.lift(g·h) = self.lift(h) @ self.lift(g)``.
        """
        if self._lift is None:
            return self.regular_lift(member, right=right)
        if not right or self.is_commutative:
            # custom lifts read the full image list of a member
            return self._lift(self._at_degree(member))
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
        """Construct a group from a multiplication (Cayley) table.

        Args:
            table: The table, whose entry ``[a, b]`` is the index of the product of the members with
                indices a and b.
            integer_lift: A function that lifts the member with a given index to a matrix.  By
                default, the group uses its regular representation.
        """
        # treat each table column as the permutation produced by right multiplication
        members = {GroupMember(col): idx for idx, col in enumerate(np.asarray(table).T)}

        def generate_func() -> Iterator[comb.Permutation]:
            yield from members

        if integer_lift is None:
            return Group(*members, generate_func=generate_func)

        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            return integer_lift(members[member])

        return Group(*members, generate_func=generate_func, lift=lift)

    @staticmethod
    def from_generating_mats(
        *matrices: npt.NDArray[np.integer]
        | Sequence[Sequence[int | galois.FieldArray] | npt.NDArray[np.integer]],
    ) -> Group:
        """Construct a Group from a set of generating matrices.

        The matrices should be over a finite field (e.g. a ``galois.FieldArray``), so that the group
        that they generate is finite.  A nested sequence or object array of field elements, such as
        a list of rows over a finite field, is a matrix over their field.  Matrices over different
        fields raise a ValueError.  Integer matrices must generate a finite group, and multiply as
        NumPy's fixed-width integers, which overflow silently.
        """
        if not matrices:
            return TrivialGroup()
        gens = [_read_generating_matrix(matrix) for matrix in matrices]
        if len({type(gen) for gen in gens if isinstance(gen, galois.FieldArray)}) > 1:
            raise ValueError("Inconsistent fields provided to Group.from_generating_mats")

        # keep track of distinct group members and a multiplication table
        index_to_member: dict[int, npt.NDArray[np.int_]] = {}
        key_to_index: dict[bytes, int] = {}
        generator_indices = []
        for gen in gens:
            if (key := _array_key(gen)) not in key_to_index:
                key_to_index[key] = len(key_to_index)
                index_to_member[key_to_index[key]] = gen
            generator_indices.append(key_to_index[key])
        table_as_dict = {}

        new_members: dict[int, npt.NDArray[np.int_]]

        def _account_for_product(aa_idx: int, bb_idx: int) -> None:
            """Account for the product of two matrices."""
            cc_mat = index_to_member[aa_idx] @ index_to_member[bb_idx]
            cc_key = _array_key(cc_mat)
            if cc_key not in key_to_index:
                key_to_index[cc_key] = cc_idx = len(key_to_index)
                new_members[cc_idx] = cc_mat
            else:
                cc_idx = key_to_index[cc_key]
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

            These matrices are generally not orthogonal (see the module docstring).
            """
            return index_to_member[permutation_to_index[tuple(member.array_form)]].T.copy()

        # identify generating permutations and build the group itself
        generators = [GroupMember(table[row]) for row in generator_indices]
        return Group(*generators, lift=lift)

    def random_symmetric_subset(
        self, size: int, *, exclude_identity: bool = False, seed: int | None = None
    ) -> set[GroupMember]:
        """Construct a random symmetric subset of a given size.

        .. note::
            This is not a uniformly random subset, only a "sufficiently random" one.

        If exclude_identity is True, the subset excludes the identity element.  Such a subset of a
        given size exists exactly if the size is at most the order of the group minus one, and the
        size or the order is even.  A symmetric subset of odd size needs a member other than the
        identity that is its own inverse, and only groups of even order have one.

        Raises:
            ValueError: If size is not positive, or no symmetric subset of that size exists.
        """
        max_size = self.order - 1 if exclude_identity else self.order
        if not 0 < size <= max_size:
            raise ValueError(
                "A random symmetric subset of this group must have a size between 1 and"
                f" {max_size} (provided: {size})"
            )
        if exclude_identity and size % 2 and self.order % 2:
            raise ValueError(
                "A group of odd order has no symmetric subset of odd size without the identity"
            )
        rng = _get_rng(seed)
        singles = set()  # group members equal to their own inverse
        doubles = set()  # pairs of group members and their inverses
        # loop until the members collected have the requested size, which the checks above allow
        while True:
            member = self._random_member(rng)
            if exclude_identity and member == self.identity:
                continue

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
                # We have overshot, so throw away members to get down to the right size.  Removing
                # an inverse pair removes two members, so an odd overshoot also removes a member
                # that is its own inverse.  Without one, keep sampling.
                for _ in range(num_extra // 2):
                    member = sorted(doubles)[rng.randint(0, len(doubles) - 1)]
                    doubles.remove(member)
                    doubles.remove(~member)
                if num_extra % 2:
                    member = sorted(singles)[rng.randint(0, len(singles) - 1)]
                    singles.remove(member)
                return singles | doubles

    @staticmethod
    def from_name(
        name: str,
        *,
        from_magma: bool = False,
        warning_to_raise_if_calling_gap: str | None = None,
    ) -> Group:
        """Retrieve a group from the GAP computer algebra system (CAS), or from MAGMA.

        If ``from_magma=True``, retrieve the group from the MAGMA CAS instead of GAP.

        A GAP group is retrieved from a small local database if possible, else from GAP if GAP can
        be called without user interaction, else from GroupNames.org (for a SmallGroup), and
        finally by a manual workflow.  Before falling back to the manual workflow, qLDPC asks on
        standard input whether GAP is available for copy/paste.  The manual workflow prints GAP
        commands, copies them to the system clipboard, and reads the output that the user pastes
        into standard input.  It blocks, and raises EOFError if standard input is closed.  Calling
        GAP may install the GAP GUAVA package, after asking for consent on standard input, and
        raises the warning warning_to_raise_if_calling_gap, if provided.  A MAGMA group is always
        retrieved by such a manual copy/paste workflow.  Retrieved generators are cached to disk.
        """
        from qldpc.external import _groups  # avoid circular import

        name = "".join(name.split())  # strip whitespace
        if from_magma:
            generators = _groups.get_generators_from_magma(name)
        else:
            generators = _groups.get_generators(
                name, warning_to_raise_if_calling_gap=warning_to_raise_if_calling_gap
            )
        return Group(*[GroupMember(generator) for generator in generators], name=name)

    def to_gap_group(self) -> str:
        """Convert a Group into a GAP group."""
        generators = [gen.to_gap_cycles() for gen in self.generators]
        return f"Group({','.join(generators)})"

    def eval(
        self,
        monomial: sympy.Expr | int | np.integer,
        symbols: dict[sympy.Symbol, GroupMember],
    ) -> GroupMember:
        """Evaluate a word in the given symbols as a member of this group.

        A word is built from the given symbols by products and integer powers, as in ``a * b**-1``
        or ``(a * b)**2``.  This method maps each symbol to its member, a product to the product of
        the members of its factors in order, a power ``w**n`` to the ``n``-th power of the member of
        ``w``, and ``1`` to the identity.  Integer-valued floats are read as integers.  Raise a
        ValueError for other expressions, such as sums and words times coefficients other than 1,
        and if a symbol in the word stands for a permutation that is not a member of this group.
        Raise a TypeError if SymPy reads the input as something other than an expression, such as an
        equation.

        Use noncommutative symbols for members that do not commute.  SymPy sorts commutative symbols
        into its own order, and cancels them, when it builds an expression: ``x * y * x**-1``
        becomes ``y``.  Raise a ValueError if a commutative symbol in the expression that this
        method receives stands for a member that does not commute with the member of another symbol
        in that expression.
        """
        expression = sympy.sympify(monomial)
        if not isinstance(expression, sympy.Expr):
            raise TypeError(f"Cannot evaluate {monomial}, because it is not a SymPy expression")
        coeff, _ = expression.as_coeff_Mul()
        if not (_is_integer(coeff) and int(coeff) == 1):
            raise ValueError(
                f"Cannot evaluate {monomial} as a word in the given symbols, because its"
                " coefficient is not 1"
            )
        member = self._eval_word(expression, symbols, monomial)
        for symbol in sorted(expression.free_symbols, key=str):
            if symbols[symbol] not in self:
                raise ValueError(
                    f"The symbol {symbol} stands for {symbols[symbol]}, which is not a member of"
                    f" {self} (degree {self.degree})"
                )
        self._check_commutative_symbols(expression.free_symbols, expression.free_symbols, symbols)
        return self._at_degree(member)

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

    def _at_degree(self, member: comb.Permutation) -> GroupMember:
        """A permutation as a group member on as many points as this group acts on.

        SymPy stores a permutation only up to its largest moved point, or on as many points as it
        was built with, so equal permutations can act on different numbers of points.  This method
        pads a permutation with fixed points, or drops fixed points beyond the degree of this group.
        A permutation that moves a point beyond the degree keeps its points.
        """
        image = member.array_form
        if len(image) < self.degree:
            return GroupMember(image, size=self.degree)
        if len(image) > self.degree and all(
            image[point] == point for point in range(self.degree, len(image))
        ):
            return GroupMember(image[: self.degree])
        return member if isinstance(member, GroupMember) else GroupMember.from_sympy(member)

    def _eval_word(
        self, word: sympy.Basic, symbols: dict[sympy.Symbol, GroupMember], original: object
    ) -> comb.Permutation:
        """Evaluate a part of a word, as in Group.eval, naming the original input in errors."""
        if isinstance(word, sympy.Symbol) and word in symbols:
            return symbols[word]
        if _is_integer(word) and int(word) == 1:
            return self.identity
        if isinstance(word, sympy.Mul):
            factors = (self._eval_word(factor, symbols, original) for factor in word.args)
            return functools.reduce(operator.mul, factors, self.identity)
        if isinstance(word, sympy.Pow):
            base, exponent = word.as_base_exp()
            if _is_integer(exponent):
                return self._eval_word(base, symbols, original) ** int(exponent)
        raise ValueError(f"Cannot evaluate {original} as a word in the given symbols")

    @staticmethod
    def _check_commutative_symbols(
        left: Iterable[sympy.Basic],
        right: Iterable[sympy.Basic],
        symbols: dict[sympy.Symbol, GroupMember],
    ) -> None:
        """Raise a ValueError if commutative symbols stand for members that do not commute.

        Compare the member of each symbol in ``left`` with the member of each symbol in ``right``,
        for every pair in which at least one symbol is commutative.  SymPy reorders commutative
        symbols in a product, which is safe only if those members commute.
        """
        pairs = itertools.product(symbols.keys() & left, symbols.keys() & right)
        # sort the pairs, so that the error names the same symbols in every run
        for symbol, other in sorted(pairs, key=lambda pair: (str(pair[0]), str(pair[1]))):
            if not symbol.is_commutative:
                symbol, other = other, symbol
            if symbol.is_commutative:
                member = symbols[symbol]
                if member * symbols[other] != symbols[other] * member:
                    raise ValueError(
                        f"The commutative symbol {symbol} stands for a member that does not"
                        f" commute with the member of {other}, but SymPy may reorder them; use"
                        " noncommutative symbols for members that do not commute"
                    )

    def _random_member(self, rng: Random) -> GroupMember:
        """A random element of this group, drawn with the given random number generator."""
        # SymPy cannot "unrank" the members of a group of order 1
        if self.order == 1:
            return self.identity
        return GroupMember.from_sympy(self._group.coset_unrank(rng.randrange(self.order)))

    @functools.cached_property
    def _members(self) -> dict[GroupMember, int]:
        return {member: idx for idx, member in enumerate(self.generate())}

    @functools.cached_property
    def _sympy_generate(self) -> GenerateFunc:
        """A function that generates members in an order set by the generators of the SymPy group.

        SymPy orders members according to the base of its group, which methods such as
        ``coset_transversal`` can change, so this function uses a new SymPy group with the same
        generators.
        """
        return comb.PermutationGroup(self._group.generators).generate

    @functools.cached_property
    def _regular_lift_cache(self) -> dict[tuple[GroupMember, bool], npt.NDArray[np.int_]]:
        """Per-instance cache for regular_lift (avoids the memory leak of caching on methods)."""
        return {}

    @functools.cached_property
    def _adjoint_lift_cache(self) -> dict[GroupMember, npt.NDArray[np.int_]]:
        """Per-instance cache for adjoint_lift (avoids the memory leak of caching on methods)."""
        return {}


class WreathProductGroup(Group):
    """Permutational wreath product of a top group and copies of a bottom group.

    If the top group acts on ``k`` points and the bottom group acts on ``m`` points, an element
    ``(h; c_0, ..., c_{k-1})`` acts on ``k * m`` points by ``(i, j) -> (h(i), c_i(j))``.  The
    selected lift is the natural permutation representation on these ``k * m`` points.
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

        Members are read on as many points as their groups act on (see ``Group``).

        Args:
            top_member: An element of the top group.
            bottom_members: One bottom-group element for each point acted on by the top group.
        """
        top_member = self.top._at_degree(top_member)
        bottom_members = tuple(self.bottom._at_degree(member) for member in bottom_members)
        if top_member not in self.top:
            raise ValueError("The top member must belong to the top group")
        if len(bottom_members) != self.top_degree:
            raise ValueError(
                f"Expected {self.top_degree} bottom members (provided: {len(bottom_members)})"
            )
        if any(member not in self.bottom for member in bottom_members):
            raise ValueError("All bottom members must belong to the bottom group")

        return GroupMember(
            [
                top_member.apply(row) * self.bottom_degree + bottom_members[row].apply(col)
                for row in range(self.top_degree)
                for col in range(self.bottom_degree)
            ]
        )


def _get_rng(seed: int | None) -> Random:
    """A new random number generator with the given seed, or SymPy's global one if seed is None."""
    return Random(seed) if seed is not None else sympy.core.random.rng


####################################################################################################
# "simple" named groups


class TrivialGroup(Group):
    """The trivial group with one member: the identity."""

    def __init__(self) -> None:
        super().__init__(
            name=TrivialGroup.__name__,
            lift=TrivialGroup._trivial_lift,
        )

    @staticmethod
    def _trivial_lift(member: GroupMember) -> npt.NDArray[np.int_]:
        return np.array(1, ndmin=2, dtype=int)


class AbelianGroup(Group):
    """Direct product of cyclic groups of the specified orders.

    See ``CyclicGroup`` for more info.  By default, an ``AbelianGroup`` member of the form
    ``∏_i g_i^{a_i}``, where ``g_i`` generates the i-th cyclic factor, gets lifted to a Kronecker
    product ``⨂_i L(g_i)^{a_i}``.  If an ``AbelianGroup`` is initialized with ``direct_sum=True``,
    the group members get lifted to a direct sum ``⨁_i L(g_i)^{a_i}``.
    """

    orders: tuple[int, ...]

    def __init__(self, *orders: int, direct_sum: bool = False) -> None:
        self.orders = orders
        group = comb.named_groups.AbelianGroup(*orders)
        order_text = ",".join(map(str, orders))
        name = f"AbelianGroup({order_text})"

        identity_mats = [np.eye(order, dtype=int) for order in orders]
        offsets = [sum(orders[:idx]) for idx in range(len(orders))]  # first point of each factor

        # SymPy omits the generators of factors of order 1, which are the identity
        identity = GroupMember(list(range(sum(orders))))
        sympy_generators = iter(group.generators)
        factor_generators = [next(sympy_generators) if order > 1 else identity for order in orders]

        # build the lift directly, which is faster than the default lift; with direct_sum=False, it
        # equals the regular lift
        def lift(member: GroupMember) -> npt.NDArray[np.int_]:
            shifts = [member.apply(offset) - offset for offset in offsets]
            mats = [
                np.roll(identity_mat, shift, axis=0)
                for identity_mat, shift in zip(identity_mats, shifts)
            ]
            # a group without factors has order 1
            if not mats:
                return np.ones((1, 1), dtype=int)
            if direct_sum:
                return scipy.linalg.block_diag(*mats)
            return functools.reduce(np.kron, mats)

        # override the default order in which SymPy iterates over group members
        def generate_func() -> Iterator[comb.Permutation]:
            for powers in itertools.product(*[range(order) for order in orders]):
                factors = [gen**power for gen, power in zip(factor_generators, powers)]
                yield functools.reduce(operator.mul, factors, identity)

        super()._init_from_group(group, name, generate_func, lift)


class CyclicGroup(AbelianGroup):
    """Cyclic group of a specified order.

    The cyclic group has one generator, g.  All members of the cyclic group of order R can be
    written as ``g^p`` for an integer power p in ``{0, 1, ..., R-1}``.  The member ``g^p`` can be
    represented by (that is, lifted to) an ``R×R`` "shift matrix", or the identity matrix with all
    rows shifted down by p.  That is, the lift ``L(g^p)`` acts on a standard basis vector ``|i>`` as
    ``L(g^p) |i> = |i + p mod R>``.
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

    The 2-dimensional lift over GF(3) is a faithful homomorphism, but it is not orthogonal (see the
    module docstring): lift(i) and lift(j) satisfy ``M.T @ M == -I``.
    """

    # multiplication table for this group

    _table: ClassVar[list[list[int]]] = [
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

        def integer_lift(index: int) -> npt.NDArray[np.int_]:
            """Representation from https://en.wikipedia.org/wiki/Quaternion_group."""
            assert 0 <= index < 8
            sign = 1 if index < 4 else -1
            base = index % 4  # +/- 1, i, j, k
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
        members = self.generate()
        next(members)
        ii = next(members)
        jj = next(members)
        return [ii, jj]


class SmallGroup(Group):
    """Group indexed by the GAP computer algebra system.

    Construction may run GAP, access GroupNames.org, or prompt on standard input; see
    SmallGroup.count_groups and Group.from_name.
    """

    group_index: int

    def __init__(self, order: int, index: int) -> None:
        if order < 1:
            raise ValueError(f"SmallGroup order must be positive (got {order})")
        num_groups = SmallGroup.count_groups(order)
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
    def count_groups(order: int) -> int:
        """The number of groups of a given order.

        This number is retrieved from the GAP SmallGrp package if GAP can be called without user
        interaction, else from GroupNames.org, and finally by the manual copy/paste GAP workflow
        described in Group.from_name.  Results are cached to disk.
        """
        from qldpc.external import _groups  # avoid circular import

        return _groups.get_small_group_number(order)

    @staticmethod
    def iter_groups(order: int) -> Iterator[SmallGroup]:
        """Iterator over all groups of a given order.

        Counting and constructing the groups may run GAP, access GroupNames.org, or prompt on
        standard input; see SmallGroup.count_groups and Group.from_name.
        """
        for ii in range(SmallGroup.count_groups(order)):
            yield SmallGroup(order, ii + 1)

    @staticmethod
    def get_structure(order: int, index: int) -> str:
        """Retrieve a description of the structure of a group.

        The structure is retrieved from GAP and cached to disk if GAP can be called without user
        interaction.  Otherwise, this method returns the name of the group,
        "SmallGroup(order,index)".
        """
        from qldpc.external import _groups  # avoid circular import

        return _groups.get_small_group_structure(order, index)


####################################################################################################
# matrix groups over finite fields


class _MatrixGroup(Group):
    """Group of square matrices over a finite field, built from the generating matrices of SL."""

    _dimension: int
    _field: type[galois.FieldArray]

    @property
    def dimension(self) -> int:
        """Dimension of the elements of this group."""
        return self._dimension

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this group."""
        return self._field

    def _init_matrix_group(
        self,
        dimension: int,
        field: int | type[galois.FieldArray] | None,
        get_generating_mats: Callable[[int, type[galois.FieldArray]], Sequence[galois.FieldArray]],
        linear_rep: bool,
    ) -> None:
        """Initialize the group generated by get_generating_mats(dimension, field).

        If linear_rep is True, represent group members by how they permute the nonzero vectors that
        the generating matrices act on.  Otherwise, represent group members by how they permute
        elements of the group itself.
        """
        self._dimension = dimension
        self._field = resolve_field(field)
        generating_mats = get_generating_mats(dimension, self.field)
        if linear_rep:
            generators, lift = _get_linear_rep(self.field, dimension, generating_mats)
            self._init_from_group(comb.PermutationGroup(generators), lift=lift)
        else:
            self._init_from_group(self.from_generating_mats(*generating_mats))


####################################################################################################
# special linear (SL) and projective special linear (PSL) groups


class SpecialLinearGroup(_MatrixGroup):
    """Special linear group (SL): square matrices with determinant 1, of dimension at least 2.

    The linear-representation lift is a homomorphism, but its matrices are generally not orthogonal
    (see the module docstring).
    """

    def __init__(
        self,
        dimension: int,
        field: int | type[galois.FieldArray] | None = None,
        linear_rep: bool = True,
    ) -> None:
        self._init_matrix_group(dimension, field, self.get_generating_mats, linear_rep)

    @staticmethod
    def get_generating_mats(
        dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Generating matrices for the Special Linear group, of dimension at least 2.

        Based on https://arxiv.org/abs/2201.09155.
        """
        if dimension < 2:
            raise ValueError(
                f"Matrix groups built from SL generators require a dimension of at least 2, not"
                f" {dimension}"
            )
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


class ProjectiveSpecialLinearGroup(_MatrixGroup):
    """Projective special linear group (PSL = SL/center), of dimension at least 2.

    Here "center" is the subgroup of SL that commutes with all elements of SL.  Specifically, every
    element in the center of SL is a scalar multiple of the identity matrix I.  In the case of
    ``SL(d, q)`` (``d × d`` matrices over ``F_q`` with determinant 1), the determinant of
    ``scalar*I`` is ``scalar**d``, so ``scalar*I`` lies in ``SL(d, q)`` only if ``scalar**d == 1``.

    Altogether, we construct ``PSL(d, q)`` by ``SL(d, q)`` mod [d-th roots of unity over ``F_q``].

    There are two ways to represent ``PSL`` by matrices.  The ``d``-dimensional linear
    representation inherited from ``SL`` only works when ``gcd(d, q - 1) == 1``, as for
    ``PSL(2, 4)`` but not ``PSL(2, 5)``.  The ``d²``-dimensional representation by conjugation,
    ``X -> g X g^-1`` on ``d × d`` matrices X, always works.  The ``linear_rep`` argument chooses
    between them:

    - ``None`` (default): use the linear representation if it exists, and otherwise conjugation.
    - ``True``: use the linear representation, and raise an error if it does not exist.
    - ``False``: use conjugation, except over GF(2), where ``PSL(d, 2) = SL(d, 2)`` and the
      ``d``-dimensional representation is used.
    """

    def __init__(
        self,
        dimension: int,
        field: int | type[galois.FieldArray] | None = None,
        linear_rep: bool | None = None,
    ) -> None:
        # The linear representation of PSL exists only when SL has a trivial center.  The center is
        # the scalar matrices in SL, of size gcd(d, q - 1), so this happens exactly when the size is
        # one.  See the class docstring for how linear_rep selects the representation.
        order = resolve_field(field).order
        num_roots = math.gcd(dimension, order - 1)
        has_linear_rep = num_roots == 1
        if linear_rep and not has_linear_rep:
            raise ValueError(
                f"PSL({dimension}, {order}) has a nontrivial center "
                f"(gcd(d, q - 1) = {num_roots} > 1), so the {dimension}-dimensional linear "
                "representation of SL does not descend to PSL; use linear_rep=False."
            )
        if linear_rep is None:
            linear_rep = has_linear_rep

        # with a trivial center, PSL(d, q) = SL(d, q), so the linear representation of PSL is that
        # of SpecialLinearGroup
        get_generating_mats = (
            SpecialLinearGroup.get_generating_mats if linear_rep else self.get_generating_mats
        )
        self._init_matrix_group(dimension, field, get_generating_mats, linear_rep)

    @staticmethod
    def get_generating_mats(
        dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Generating matrices of PSL, constructed out of the generating matrices of SL."""
        field = resolve_field(field)
        gen_x, gen_w = SpecialLinearGroup.get_generating_mats(dimension, field)
        if field.order == 2:
            # the center of SL(d, 2) is trivial, so PSL(d, 2) = SL(d, 2)
            return gen_x, gen_w
        # Represent ``g`` by its conjugation action ``X -> g X g^-1`` on ``d × d`` matrices ``X``.
        # Vectorized, this is ``vec(g X g^-1) = kron((g^-1).T, g) @ vec(X)``.  The kernel of this
        # representation is the center of SL, so it is well-defined and faithful on
        # ``SL/center = PSL``.
        return (
            np.kron(np.linalg.inv(gen_x).T, gen_x).view(field),
            np.kron(np.linalg.inv(gen_w).T, gen_w).view(field),
        )

    @staticmethod
    def iter_mats(
        dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> Iterator[galois.FieldArray]:
        """Iterate over all elements of PSL(dimension, field)."""
        field = resolve_field(field)
        num_roots = math.gcd(dimension, field.order - 1)
        primitive_root = field.primitive_element ** ((field.order - 1) // num_roots)
        roots = [primitive_root**k for k in range(num_roots)]
        yield from _iter_scalar_orbits(roots, SpecialLinearGroup.iter_mats(dimension, field))


####################################################################################################
# general linear (GL) and projective general linear (PGL) groups


class GeneralLinearGroup(_MatrixGroup):
    """General linear group (``GL``): invertible square matrices, of dimension at least 2.

    The linear-representation lift is a homomorphism, but its matrices are generally not orthogonal
    (see the module docstring).
    """

    def __init__(
        self,
        dimension: int,
        field: int | type[galois.FieldArray] | None = None,
        linear_rep: bool = True,
    ) -> None:
        self._init_matrix_group(dimension, field, self.get_generating_mats, linear_rep)

    @staticmethod
    def get_generating_mats(
        dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> tuple[galois.FieldArray, ...]:
        """Generating matrices for the General Linear group.

        ``GL(d, q)`` is generated by ``SL(d, q)`` together with any matrix whose determinant
        generates ``F_q^*``.  We add a diagonal matrix that scales one basis vector by a primitive
        element of the field, which is the determinant of that matrix.
        """
        field = resolve_field(field)
        gen_x, gen_w = SpecialLinearGroup.get_generating_mats(dimension, field)
        if field.order == 2:
            # ``F_2^*`` is trivial, so every invertible matrix already has determinant 1:
            # ``GL(d, 2) = SL(d, 2)``.
            return gen_x, gen_w
        gen_d = field.Identity(dimension)
        gen_d[0, 0] = field.primitive_element
        return gen_x, gen_w, gen_d

    @staticmethod
    def iter_mats(
        dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> Iterator[galois.FieldArray]:
        """Iterate over all elements of ``GL(dimension, field)``."""
        field = resolve_field(field)
        for vec in itertools.product(field.elements, repeat=dimension**2):
            mat = np.reshape(vec, (dimension, dimension)).view(field)
            if np.linalg.det(mat) != 0:
                yield mat


class ProjectiveGeneralLinearGroup(_MatrixGroup):
    """Projective general linear group (``PGL = GL/center``), of dimension at least 2.

    Here "center" is the subgroup of ``GL`` that commutes with all elements of ``GL``.
    Specifically, every element in the center of ``GL`` is a scalar multiple of the identity matrix
    ``I``.  Unlike ``SL``, ``GL`` places no constraint on the determinant, so *every* nonzero scalar
    in ``F_q`` times ``I`` lies in ``GL`` and hence in its center.

    Altogether, we construct ``PGL(d, q)`` by quotienting ``GL(d, q)`` by the subgroup of nonzero
    scalars in ``F_q``, which has order ``q - 1``.

    There are two ways to represent ``PGL`` by matrices.  The ``d``-dimensional linear
    representation inherited from ``GL`` only works when the center of ``GL`` is trivial, which is
    when ``q == 2``.  The ``d²``-dimensional representation by conjugation, ``X -> g X g^-1`` on
    ``d × d`` matrices X, always works.  The ``linear_rep`` argument chooses between them:

    - ``None`` (default): use the linear representation if it exists, and otherwise conjugation.
    - ``True``: use the linear representation, and raise an error if it does not exist.
    - ``False``: use conjugation, except over GF(2), where the center is trivial and the
      ``d``-dimensional representation is used.
    """

    def __init__(
        self,
        dimension: int,
        field: int | type[galois.FieldArray] | None = None,
        linear_rep: bool | None = None,
    ) -> None:
        # The linear representation of ``PGL`` exists only when ``GL`` has a trivial center.  The
        # center is all nonzero scalar matrices, of size ``q - 1``, so this happens exactly when
        # ``q == 2``.  See the class docstring for how ``linear_rep`` selects the representation.
        order = resolve_field(field).order
        has_linear_rep = order == 2
        if linear_rep and not has_linear_rep:
            raise ValueError(
                f"PGL({dimension}, {order}) has a nontrivial center "
                f"(order q - 1 = {order - 1} > 1), so the {dimension}-dimensional "
                "linear representation of GL does not descend to PGL; use linear_rep=False."
            )
        if linear_rep is None:
            linear_rep = has_linear_rep

        # With a trivial center, ``PGL(d, q) = GL(d, q)``, so the linear representation of ``PGL``
        # is that of ``GeneralLinearGroup``.
        get_generating_mats = (
            GeneralLinearGroup.get_generating_mats if linear_rep else self.get_generating_mats
        )
        self._init_matrix_group(dimension, field, get_generating_mats, linear_rep)

    @staticmethod
    def get_generating_mats(
        dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> tuple[galois.FieldArray, ...]:
        """Generating matrices of ``PGL``, constructed from the generating matrices of ``GL``."""
        field = resolve_field(field)
        generators = GeneralLinearGroup.get_generating_mats(dimension, field)
        if field.order == 2:
            # the center of GL(d, 2) is trivial, so PGL(d, 2) = GL(d, 2)
            return generators
        # The conjugation representation ``g -> kron(g^-1.T, g)`` (see
        # ``ProjectiveSpecialLinearGroup.get_generating_mats``) is invariant under rescaling ``g``
        # by any scalar, so it is well-defined on the quotient ``GL/center = PGL``.
        return tuple(np.kron(np.linalg.inv(gen).T, gen).view(field) for gen in generators)

    @staticmethod
    def iter_mats(
        dimension: int, field: int | type[galois.FieldArray] | None = None
    ) -> Iterator[galois.FieldArray]:
        """Iterate over all elements of ``PGL(dimension, field)``."""
        field = resolve_field(field)
        scalars = [elem for elem in field.elements if elem != 0]
        yield from _iter_scalar_orbits(scalars, GeneralLinearGroup.iter_mats(dimension, field))


# standard short names of the linear groups, as in the docstrings and the literature
SL = SpecialLinearGroup
PSL = ProjectiveSpecialLinearGroup
GL = GeneralLinearGroup
PGL = ProjectiveGeneralLinearGroup


####################################################################################################
# helper methods


def _iter_scalar_orbits(
    scalars: Sequence[galois.FieldArray], matrices: Iterable[galois.FieldArray]
) -> Iterator[galois.FieldArray]:
    """Iterate over one matrix in each orbit of the given matrices under multiplication by scalars.

    Each orbit is represented by its first matrix in the order of _array_key, and the
    representatives are yielded in that order.
    """
    representatives = {}
    for matrix in matrices:
        representative = min((scalar * matrix for scalar in scalars), key=_array_key)
        representatives[_array_key(representative)] = representative
    for key in sorted(representatives):
        yield representatives[key]


def _read_generating_matrix(
    matrix: npt.NDArray[np.integer]
    | Sequence[Sequence[int | galois.FieldArray] | npt.NDArray[np.integer]],
) -> npt.NDArray[np.int_]:
    """Read a matrix into a new array, keeping its type, or the field of its field elements.

    A nested sequence or an array of objects that holds field elements is a matrix over their field.
    Any other array is copied, and any other nested sequence is read as integers.
    """
    if isinstance(matrix, galois.FieldArray) or (
        isinstance(matrix, np.ndarray) and matrix.dtype != object
    ):
        return cast(npt.NDArray[np.int_], matrix.copy())
    fields = {
        type(value) for row in matrix for value in row if isinstance(value, galois.FieldArray)
    }
    if len(fields) > 1:
        raise ValueError("Inconsistent fields provided to Group.from_generating_mats")

    # read Python integers, which the elements of large fields can need
    integers = [[int(value) for value in row] for row in matrix]
    if fields:
        return cast(npt.NDArray[np.int_], fields.pop()(integers))
    if isinstance(matrix, np.ndarray):
        return cast(npt.NDArray[np.int_], matrix.copy())
    return np.array(integers, dtype=int)


def _array_key(matrix: npt.NDArray[np.integer]) -> bytes:
    """Bytes that identify an array among arrays of its shape and kind of values.

    Arrays of NumPy integers of any width get the same key, and so do arrays of Python objects, but
    an array of one kind and an array of the other do not.
    """
    array = np.asarray(matrix)
    if array.dtype == object:
        # arrays of Python objects, as over large fields
        values = [value.item() if isinstance(value, np.generic) else value for value in array.flat]
        return repr(values).encode()
    return (array.astype(np.int64) if array.dtype.kind in "iub" else array).tobytes()


def _get_linear_rep(
    field: type[galois.FieldArray],
    dimension: int,
    generating_mats: Iterable[galois.FieldArray],
) -> tuple[list[GroupMember], Lift]:
    """Represent a matrix group by how its members permute the nonzero vectors that they act on.

    Return the permutations of the generating matrices, and a lift that maps a group member to a
    matrix.  The lift stacks the images of standard basis vectors as rows, which yields the
    transpose of the acting matrix.  That transpose, not the matrix itself, is the homomorphism
    ``lift(g . h) = lift(g) @ lift(h)``.
    """
    # the nonzero vectors, which group members permute
    vectors = [field(vec) for vec in itertools.product(field.elements, repeat=dimension)][1:]
    vector_index = {_array_key(vec): index for index, vec in enumerate(vectors)}

    generators = [
        GroupMember([vector_index[_array_key(mat @ vec)] for vec in vectors])
        for mat in generating_mats
    ]

    def lift(member: GroupMember) -> npt.NDArray[np.int_]:
        """Lift a group member to the transpose of the matrix that it represents."""
        rows = [vectors[member(vector_index[_array_key(row)])] for row in field.Identity(dimension)]
        return field(np.vstack(rows))

    return generators, lift
