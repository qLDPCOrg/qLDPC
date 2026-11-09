# SPDX-License-Identifier: Apache-2.0

"""Module for abstract algebra: rings, group algebras, and their members.

.. warning::
    This module does not promise to be performant.  If you need to do heavy numerical abstract
    algebra, you're probably better served by GAP or MAGMA (or maybe SageMath).
"""

from __future__ import annotations

import abc
import collections
import copy
import functools
import itertools
import operator
from collections.abc import Hashable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

import galois
import numpy as np
import numpy.typing as npt
import sympy.abc
import sympy.combinatorics
import sympy.core

from qldpc import _deprecations
from qldpc.math import resolve_field

from .groups import AbelianGroup, Group, GroupMember
from .monomials import iter_monomial_terms

if TYPE_CHECKING:
    from .wedderburn_artin import WedderburnArtinTransformer


####################################################################################################
# rings


class Ring(abc.ABC):
    """A finite ring that is an algebra over a finite field.

    Every ``Ring`` is an associative algebra with a unit over a finite field ``F_q``, and has a
    finite dimension ``dim`` as a vector space over ``F_q``.  A basis of the ring is indexed by
    hashable labels, and every ``RingMember`` is a linear combination of basis elements.  The
    coefficient vector ``RingMember.to_vector`` lists the coefficients in the order of
    ``Ring.basis``.  The ring implementations are ``GroupRing``, whose basis labels are group
    members, and ``MatrixRing``, whose basis labels are positions in the basis.  ``Ring`` is not
    meant to be subclassed outside of qLDPC.

    A ring member acts on coefficient vectors by its regular representation, a matrix of size
    ``dim``.  It also lifts to a matrix over ``F_q`` of size ``lift_dim``.  For ring members ``a``
    and ``b``,

        ``(a * b).lift() = a.lift() @ b.lift()``.

    The opposite ring, in which the order of multiplication is reversed, is represented by lifts
    with ``right=True``, which are anti-representations:

        ``(a * b).lift(right=True) = b.lift(right=True) @ a.lift(right=True)``.
    """

    _field: type[galois.FieldArray]

    def __init__(self, field: int | type[galois.FieldArray] | None) -> None:
        self._field = resolve_field(field)

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this ring."""
        return self._field

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Dimension of this ring as a vector space over its base field."""

    @property
    @abc.abstractmethod
    def lift_dim(self) -> int:
        """Size of the matrices obtained by lifting ring members to a matrix representation."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """A name for this ring, which is not required to uniquely identify the ring."""

    def __str__(self) -> str:
        return self.name

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        """Do two rings have the same field, basis, multiplication, and lifts?"""

    @abc.abstractmethod
    def __hash__(self) -> int:
        """Hash that is consistent with equality."""

    @property
    @abc.abstractmethod
    def is_commutative(self) -> bool:
        """Is this ring commutative?"""

    @property
    @abc.abstractmethod
    def has_involution(self) -> bool:
        """Does ``RingMember.T`` apply an involution of this ring?

        An involution is an anti-automorphism of the ring that is its own inverse.  If this property
        is False, ``RingMember.T`` raises a ValueError.
        """

    @property
    def zero(self) -> RingMember:
        """Zero (additive identity) element."""
        return RingMember(self)

    @property
    @abc.abstractmethod
    def one(self) -> RingMember:
        """One (multiplicative identity) element."""

    @property
    @abc.abstractmethod
    def generators(self) -> list[RingMember]:
        """Ring members that generate this ring as an algebra with a unit."""

    @property
    def basis(self) -> list[RingMember]:
        """Basis of this ring as a vector space over its field.

        The basis is ordered like the coefficients of ``RingMember.to_vector``.
        """
        return [RingMember._from_trusted_terms(self, [(1, label)]) for label in self._labels]

    @abc.abstractmethod
    def lift(self, label: Any, /, *, right: bool = False) -> galois.FieldArray:
        """Lift the basis element with a given label to a matrix of size ``lift_dim``.

        If ``right=True``, lift to an anti-representation.  See ``help(qldpc.abstract.Ring)``.
        """

    @abc.abstractmethod
    def regular_lift(self, label: Any, /, *, right: bool = False) -> galois.FieldArray:
        """Lift the basis element with a given label to its regular representation.

        The regular representation is a matrix of size ``dim``.  By default, it is induced by
        multiplication from the left.  If ``right=True``, it instead acts by multiplication from the
        right, in the opposite ring.  See ``RingMember.regular_lift``.
        """

    def get_transformer(
        self, seed: int | None = None, *, skip_validation: bool = False
    ) -> WedderburnArtinTransformer:
        """Instrument for the Wedderburn-Artin decomposition of this ring.

        The decomposition is implemented only for group algebras, so this method raises a
        NotImplementedError for any other ring.
        """
        from .wedderburn_artin import WedderburnArtinTransformer  # avoid circular import

        # the transformer raises a NotImplementedError for rings other than group algebras
        return WedderburnArtinTransformer(self, seed=seed, skip_validation=skip_validation)

    @property
    @abc.abstractmethod
    def _labels(self) -> list[Hashable]:
        """Labels of the basis of this ring, in the order of the coefficient vectors."""

    @abc.abstractmethod
    def _index(self, label: Any) -> int:
        """Position of a basis label in the coefficient vector of a ring member."""

    @abc.abstractmethod
    def _multiply(self, left: RingMember, right: RingMember) -> RingMember:
        """Multiply two members of this ring."""

    @abc.abstractmethod
    def _apply_involution(self, member: RingMember) -> RingMember:
        """Apply the involution of this ring, or raise a ValueError if there is none."""

    @abc.abstractmethod
    def _as_label(self, label: Any) -> Hashable:
        """The canonical label of the basis element of this ring that a given label denotes.

        Raise a ValueError if the given label does not denote a basis element of this ring.
        """

    def _check_same_ring(self, other: Ring) -> None:
        """Raise a ValueError unless another ring equals this ring."""
        if other != self:
            raise ValueError("Cannot combine members of different rings")

    def _format(self, member: RingMember) -> str:
        """Write a ring member as a sum of basis elements, such as ``2 e_1 + e_3``."""
        terms = sorted((self._index(label), int(coeff)) for coeff, label in member if coeff)
        words = [f"e_{index}" if coeff == 1 else f"{coeff} e_{index}" for index, coeff in terms]
        return " + ".join(words) or "0"

    def _eval_int(self, value: int) -> galois.FieldArray:
        """Evaluate an integer as an element of the base field of this ring.

        Over an extension field the integer is interpreted via galois' integer encoding of field
        elements (e.g. over ``GF(4)`` the integer ``2`` is the element with integer representation
        2, the primitive element -- not ``2 * one``, which is ``0`` in characteristic 2).  Integers
        outside ``[0, field.order)`` are accepted only when unambiguous: over a prime field they
        reduce modulo the order, and over an extension field a negative integer in
        ``(-field.order, 0)`` is read as the additive inverse of its magnitude; any other
        out-of-range integer raises a ValueError.
        """
        if not 0 <= value < self.field.order:
            if self.field.degree == 1:
                # there is no ambiguity over prime number fields
                return self.field(int(value) % self.field.order)
            elif -self.field.order < value < 0:
                # negation corresponds to an additive inverse
                return -self.field(-value)
            else:
                raise ValueError(
                    f"The value of the integer {value} is ambiguous over GF({self.field.order})"
                )
        return self.field(value)


####################################################################################################
# group algebras


class GroupRing(Ring):
    """A finite group algebra over a finite field.

    The base field is ``GF(2)`` by default.  The basis of a group algebra is its group, so its
    dimension is the order of the group.
    """

    _group: Group
    _transformers: dict[int | None, WedderburnArtinTransformer]
    _idempotents: tuple[RingMember, ...] | None = None

    def __init__(self, group: Group, field: int | type[galois.FieldArray] | None = None) -> None:
        super().__init__(field)
        self._group = group
        self._transformers = {}

    @property
    def group(self) -> Group:
        """Base group of this ring."""
        return self._group

    @property
    def dim(self) -> int:
        """Dimension of this ring as a vector space over its base field: the order of the group."""
        return self.group.order

    @property
    def lift_dim(self) -> int:
        """Size of the matrices that lift ring members: the lift dimension of the group."""
        return self.group.lift_dim

    def get_transformer(
        self, seed: int | None = None, *, skip_validation: bool = False
    ) -> WedderburnArtinTransformer:
        """Instrument for the Wedderburn-Artin decomposition of this ring.

        By default each component's primitive central idempotent is validated; pass
        ``skip_validation=True`` to skip the check (see WedderburnArtinComponentTransformer).

        Unless the ring's primitive central idempotents are already known, this may run GAP, prompt
        on standard input, and use the system clipboard; see get_primitive_central_idempotents.
        """
        from .wedderburn_artin import WedderburnArtinTransformer  # avoid circular import

        # for seedless calls, reuse an existing decomposition instead of building another
        if seed not in self._transformers:
            if seed is None and self._transformers:
                return next(iter(self._transformers.values()))
            self._transformers[seed] = WedderburnArtinTransformer(
                self, seed=seed, skip_validation=skip_validation
            )
        return self._transformers[seed]

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, GroupRing)
            and self.field is other.field
            and self.group.equiv(other.group)
            and self._has_same_lifts(other.group)
        )

    def __hash__(self) -> int:
        return hash((self.field.order, self.group.to_sympy()))

    @property
    def name(self) -> str:
        """A name for this ring, which is not required to uniquely identify the ring."""
        return f"Group algebra of {self.group.name} over GF({self.field.order})"

    @property
    def is_commutative(self) -> bool:
        """Is this ring commutative?"""
        return self._group.is_abelian

    @functools.cached_property
    def is_semisimple(self) -> bool:
        """Is this ring semisimple?"""
        return bool(self.group.order % self.field.characteristic)

    @functools.cached_property
    def group_trace_matrix(self) -> galois.FieldArray:
        """Construct the matrix for a trace over the group: ``r -> sum_{g in G} g r g^{-1}``."""
        adjoints = [self.group.adjoint_lift(gg).view(self.field) for gg in self.group.generate()]
        return functools.reduce(operator.add, adjoints).view(self.field)

    @property
    def generators(self) -> list[RingMember]:
        """Generators of this ring's base group."""
        return [RingMember._from_trusted_terms(self, [(1, gen)]) for gen in self.group.generators]

    def regular_lift(self, member: GroupMember, *, right: bool = False) -> galois.FieldArray:
        """Lift a group member to its regular representation.

        See ``help(qldpc.abstract.Group.regular_lift)`` for more information.
        """
        return self.group.regular_lift(member, right=right).view(self.field)

    def lift(self, member: GroupMember, *, right: bool = False) -> galois.FieldArray:
        """Lift a group member to a representation by a matrix.

        A representation satisfies

            ``self.lift(g·h) = self.lift(g) @ self.lift(h)``.

        If ``right=True``, lift to an anti-representation, for which

            ``self.lift(g·h) = self.lift(h) @ self.lift(g)``.

        For a noncommutative group with a custom lift, ``right=True`` raises a ValueError.
        """
        return self.group.lift(member, right=right).view(self.field)

    @property
    def one(self) -> RingMember:
        """One (multiplicative identity) element."""
        return RingMember._from_trusted_terms(self, [(1, self.group.identity)])

    @property
    def has_involution(self) -> bool:
        """Always True, because the involution of a group algebra inverts group members.

        See ``RingMember.T``.
        """
        return True

    def get_primitive_central_idempotents(self) -> tuple[RingMember, ...]:
        """Get the primitive central idempotents of this ring.

        Primitive central idempotents of a ring are nonzero elements that:

        - square to themselves (they are idempotent),
        - commute with all other elements of the ring (they lie in the ring's center), and
        - cannot be decomposed into a sum of two nonzero orthogonal idempotents.

        Two idempotents g, h are orthogonal if ``g * h = h * g = 0``.

        Intuitively, primitive central idempotents act like projectors onto orthogonal simple
        components of a ring.

        See `Idempotent (ring theory)
        <https://en.wikipedia.org/wiki/Idempotent_%28ring_theory%29>`_.

        Unless they are already known, the idempotents are computed with the GAP Wedderga package
        and cached to disk.  If GAP can be called but lacks Wedderga, qLDPC asks for consent on
        standard input to install it.  If GAP cannot be called directly, qLDPC asks on standard
        input whether GAP is available for copy/paste, and then prints the GAP commands, copies
        them to the system clipboard, and reads the output that the user pastes into standard input.
        """
        if not self.is_semisimple:
            raise ValueError("Only semisimple rings have primitive central idempotents")
        if self._idempotents is None:
            from qldpc.external import _groups  # avoid circular import

            idempotents_as_tuples = _groups.get_primitive_central_idempotents(
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
                idempotents.append(RingMember._from_trusted_terms(self, terms))
            self._idempotents = tuple(idempotents)
        return self._idempotents

    def eval(
        self, expression: sympy.Basic | int | np.int_, symbols: dict[sympy.Symbol, GroupMember]
    ) -> RingMember:
        """Convert a SymPy expression (such as a polynomial) into a member of this ring."""
        # helpful error message for invalid symbols
        if any(not isinstance(value, GroupMember) for value in symbols.values()):
            raise ValueError("The symbols passed to GroupRing.eval must be GroupMember-valued")

        # sum the ring members obtained by evaluating each monomial term of the polynomial
        terms = (self._eval_monomial(term, symbols) for term in iter_monomial_terms(expression))
        return functools.reduce(operator.add, terms, self.zero)

    def _eval_monomial(
        self, monomial: sympy.Expr, symbols: dict[sympy.Symbol, GroupMember]
    ) -> RingMember:
        """Convert a single SymPy monomial into a member of this ring."""
        # split the monomial into its integer coefficient and its group-element content
        _coeff, group_content = monomial.as_coeff_Mul()
        coeff = self._eval_int(int(_coeff))
        group_member = self.group.eval(group_content, symbols)
        return RingMember(self, (coeff, group_member))

    @functools.cached_property
    def _labels(self) -> list[Hashable]:
        # cached, because many coefficient vectors are converted into ring members at a time
        return list(self.group._members)

    def _index(self, label: Any) -> int:
        return self.group.index(label)

    def _multiply(self, left: RingMember, right: RingMember) -> RingMember:
        product = self.zero
        for (x_a, aa), (y_b, bb) in itertools.product(left, right):
            product._vec[aa * bb] += x_a * y_b
        return product

    def _apply_involution(self, member: RingMember) -> RingMember:
        transpose = self.zero
        for val, label in member:
            transpose._vec[~label] = val
        return transpose

    def _as_label(self, label: Any) -> GroupMember:
        """A member of the group of this ring, at the degree of the group.

        A permutation can act on fewer points than the group, such as ``GroupMember(0, 1)`` in a
        group of degree 4.  This method pads such a permutation with fixed points, and raises a
        ValueError if the result is not in the group.  Every identity permutation, of any size,
        becomes the identity of the group, since SymPy rejects permutations with more points than
        the group, and does not identify the identity as a member of a trivial group.
        """
        if isinstance(label, sympy.combinatorics.Permutation):
            if label.is_Identity:
                return self.group.identity
            if label.size < self.group.degree:
                label = GroupMember(label.array_form, size=self.group.degree)
            if label in self.group:
                return GroupMember.from_sympy(label)
        raise ValueError(f"{label} is not a member of {self.group} (degree {self.group.degree})")

    def _has_same_lifts(self, other_group: Group) -> bool:
        """Does a group equivalent to this ring's group lift members to the same matrices?

        This method also compares lifts with ``right=True``, which Kronecker products use.
        """
        # equal groups share their lift and enumeration functions
        group = self.group
        if group == other_group:
            return True

        # lifts of different sizes differ, so avoid building them, which can be costly
        if group.lift_dim != other_group.lift_dim:
            return False

        # a noncommutative group has right lifts only with its default lift
        if not group.is_commutative and (group._lift is None) != (other_group._lift is None):
            return False

        # the lifts and right lifts of generators determine those of every member, and right lifts
        # of a commutative group equal its lifts
        compare_right_lifts = not group.is_commutative and group._lift is None
        sides = [False, True] if compare_right_lifts else [False]
        return all(
            np.array_equal(
                group.lift(generator, right=right), other_group.lift(generator, right=right)
            )
            for generator in group.generators
            for right in sides
        )

    def _format(self, member: RingMember) -> str:
        """Write a group algebra element as a polynomial in the generators of the group."""
        # identify symbols for the generators of the base group
        num_gens = len(self.group.generators)
        if num_gens <= 3:
            symbols = sympy.symbols("x:z", commutative=self.group.is_commutative)[:num_gens]
        elif num_gens <= 26:
            symbols = sympy.symbols("a:z", commutative=self.group.is_commutative)[-num_gens:]
        else:  # pragma: no cover
            index_length = int(np.ceil(np.log10(num_gens + 1)))
            symbols = [
                sympy.Symbol(f"x_{index:0{index_length}}", commutative=self.group.is_commutative)
                for index in range(num_gens)
            ]

        if isinstance(self.group, AbelianGroup):
            # abelian groups are an easy special case for building the polynomial
            monomials = []
            for powers in itertools.product(*[range(order) for order in self.group.orders]):
                factors = [symbol**power for symbol, power in zip(symbols, powers)]
                monomials.append(functools.reduce(operator.mul, factors))
            terms = [
                int(coeff) * monomial
                for coeff, monomial in zip(member.to_vector(), monomials)
                if coeff
            ]

        else:
            # general-purpose fallback: write members as words in Group.generators, which can differ
            # from the generators of the SymPy group
            sympy_group = sympy.combinatorics.PermutationGroup(self.group.generators)
            gen_to_symbol = {gen: symbol for gen, symbol in zip(sympy_group.generators, symbols)}
            gen_to_symbol |= {
                ~gen: 1 / symbol
                for gen, symbol in gen_to_symbol.items()
                if ~gen not in gen_to_symbol
            }

            terms = []
            for x_g, gg in member:
                # SymPy's factors [s_1, ..., s_n] give g = s_n * ... * s_1
                gens = sympy_group.generator_product(gg, original=True)[::-1]
                factors = [gen_to_symbol[gen] for gen in gens]
                monomial = functools.reduce(operator.mul, factors, 1)
                terms.append(int(x_g) * monomial)

        return str(sum(terms) + sympy.core.numbers.Zero()).replace("**", "^").replace("*", " ")


####################################################################################################
# members of rings


class RingMember:
    """An element of a finite ring, which is an algebra over a finite field ``F_q``.

    Each RingMember x is a sum of basis elements of the ring with coefficients in the field:
    ``x = sum_b x_b b``, with each ``x_b in F_q``.  The basis of a group algebra ``F_q[G]`` is the
    group ``G``, so each of its elements is a sum of group members, ``x = sum_{g in G} x_g g``.

    Terms are specified as ``(coefficient, label)`` pairs, where the ring defines the labels of its
    basis elements.  A group member as a term has coefficient one, and a group member that acts on
    fewer points than its group is padded with fixed points.

    A RingMember combines on either side with integers and members of an equal ring, and with field
    scalars on its right.  A member of a group algebra also combines on either side with members of
    its group.  Combining a RingMember with a member of a different ring raises a ValueError.  So
    does combining a member of a group algebra with a group member outside its group, or
    constructing a RingMember from a label that does not denote a basis element of its ring.
    Combining a member of any other ring with a group member raises a TypeError.
    """

    _ring: Ring
    _vec: collections.defaultdict[Any, galois.FieldArray]  # coefficient of each basis label

    def __init__(
        self,
        ring: Ring | Group,
        *terms: GroupMember | tuple[int | galois.FieldArray, Any],
    ) -> None:
        self._ring = ring if isinstance(ring, Ring) else GroupRing(ring)
        self._vec = collections.defaultdict(lambda: self.field(0))
        for term in terms:
            value, label = (1, term) if isinstance(term, GroupMember) else term
            self._vec[self.ring._as_label(label)] += self.field(value)

    def __str__(self) -> str:
        """Write a member of a group algebra as a polynomial in the generators of its group.

        Write a member of any other ring as a sum of basis elements, such as ``2 e_1 + e_3``, where
        ``e_i`` is ``Ring.basis[i]``.
        """
        return self.ring._format(self)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RingMember)
            and self._ring == other._ring
            and all(self._vec.get(label, 0) == other._vec.get(label, 0) for label in self._vec)
            and all(self._vec.get(label, 0) == other._vec.get(label, 0) for label in other._vec)
        )

    def __bool__(self) -> bool:
        return any(coefficient for coefficient in self._vec.values())

    def __iter__(self) -> Iterator[tuple[galois.FieldArray, Any]]:
        for label, coefficient in self._vec.items():
            yield coefficient, label

    def __add__(self, other: int | galois.FieldArray | GroupMember | RingMember) -> RingMember:
        if isinstance(other, (int, self.field, GroupMember)):
            # convert a scalar or group member into a ring member, with the scalar or group member
            # on the right, since galois raises an error for a field scalar on the left
            return self + self.ring.one * other

        if isinstance(other, RingMember):
            self.ring._check_same_ring(other.ring)
            new_element = self.copy()
            for val, label in other:
                new_element._vec[label] += val
            return new_element

        return NotImplemented

    def __sub__(self, other: int | galois.FieldArray | GroupMember | RingMember) -> RingMember:
        # a group member has no negative, so negate it as a ring member
        if isinstance(other, GroupMember):
            other = self.ring.one * other
        return self + -other

    def __radd__(self, other: int | GroupMember) -> RingMember:
        return self + other

    def __rsub__(self, other: int | GroupMember) -> RingMember:
        return -self + other

    def __mul__(self, other: int | galois.FieldArray | GroupMember | RingMember) -> RingMember:
        if isinstance(other, int):
            other = self.ring._eval_int(other)

        if isinstance(other, self.field):
            # multiply coefficients by 'other'
            new_element = self.ring.zero
            for val, label in self:
                new_element._vec[label] = val * other
            return new_element

        if isinstance(other, GroupMember):
            return self._translate(other, on_left=False)

        if isinstance(other, RingMember):
            self.ring._check_same_ring(other.ring)
            return self.ring._multiply(self, other)

        return NotImplemented

    def __rmul__(self, other: int | galois.FieldArray | GroupMember) -> RingMember:
        if isinstance(other, (int, self.field)):
            return self * other

        if isinstance(other, GroupMember):
            return self._translate(other, on_left=True)

        return NotImplemented

    def __neg__(self) -> RingMember:
        return self * (-1)

    def __pow__(self, power: int) -> RingMember:
        if not isinstance(power, (int, np.int_)) or power < 0:
            raise ValueError(
                "A RingMember can only be raised to an integer power >= 0."
                "\nTry ring_member.inverse() ** abs(power)"
            )
        return functools.reduce(operator.mul, [self] * power) if power > 0 else self.ring.one

    def copy(self) -> RingMember:
        """Copy of self."""
        element = self.ring.zero
        for val, label in self:
            element._vec[label] = copy.deepcopy(val)
        return element

    @property
    def ring(self) -> Ring:
        """Base ring of this member."""
        return self._ring

    @property
    def group(self) -> Group:
        """Base group of the group algebra that contains this member.

        Raise an AttributeError if the ring of this member is not a group algebra.
        """
        if isinstance(self.ring, GroupRing):
            return self.ring.group
        raise AttributeError(f"{self.ring} is not a group algebra, and has no base group")

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this member."""
        return self.ring.field

    def lift(self, *, right: bool = False) -> galois.FieldArray:
        """Lift this ring member to a matrix of size ``Ring.lift_dim`` over the base field.

        For ring members ``a`` and ``b``,

            ``(a * b).lift() = a.lift() @ b.lift()``.

        If ``right=True``, lift to an anti-representation, for which

            ``(a * b).lift(right=True) = b.lift(right=True) @ a.lift(right=True)``.
        """
        return sum(
            (val * self.ring.lift(label, right=right) for val, label in self if val),
            start=self.field.Zeros([self.ring.lift_dim] * 2),
        )

    def regular_lift(self, *, right: bool = False) -> galois.FieldArray:
        """Lift a ring member to its regular representation.

        By default, this method lifts a ring member to the regular representation induced by
        multiplication from the left.  Specifically, if ``r`` and ``s`` are ring members, then::

            r.regular_lift() @ s.to_vector() = (r * s).to_vector().

        If ``right is True``, this method lifts a ring member to its regular representation in the
        opposite ring, such that matrix multiplication corresponds to ring multiplication from the
        right::

            r.regular_lift(right=True) @ s.to_vector() = (s * r).to_vector().

        See `Opposite ring <https://en.wikipedia.org/wiki/Opposite_ring>`_.
        """
        terms = (val * self.ring.regular_lift(label, right=right) for val, label in self if val)
        return (
            functools.reduce(operator.add, terms)
            if bool(self)
            else self.field.Zeros([self.ring.dim] * 2)
        )

    @property
    def T(self) -> RingMember:
        """Transpose of this element, which is its image under the involution of the ring.

        The involution of a group algebra maps ``x = sum_{g in G} x_g g`` to
        ``x.T = sum_{g in G} x_g g**-1``.  If the lifts of group members are orthogonal, this
        transposes the lifted matrices, ``x.T.lift() = x.lift().T``.  The involution of a
        ``MatrixRing`` is the matrix transpose.

        Raise a ValueError if the ring has no involution (see ``Ring.has_involution``).
        """
        return self.ring._apply_involution(self)

    def inverse(self) -> RingMember | None:
        """The inverse of this RingMember, if it exists."""
        self_vec = {label: coefficient for label, coefficient in self._vec.items() if coefficient}
        if not self_vec:
            return None

        # a nonzero multiple of a group member is easy to invert
        if len(self_vec) == 1 and isinstance(self.ring, GroupRing):
            label, coefficient = next(iter(self_vec.items()))
            return RingMember._from_trusted_terms(self.ring, [(coefficient**-1, label**-1)])

        # otherwise, invert the matrix of the regular representation
        try:
            matrix = self.regular_lift()
            matrix_inv = np.linalg.inv(matrix).view(self.field)
            # ``regular_lift`` M satisfies M @ s.to_vector() == (self * s).to_vector(), so the
            # inverse element solves M @ x == one.to_vector(), i.e. x = M^{-1} @ one.to_vector().
            return RingMember.from_vector(matrix_inv @ self.ring.one.to_vector(), self.ring)
        except np.linalg.LinAlgError:
            return None

    @classmethod
    def from_vector(cls, vector: npt.NDArray[np.int_], ring: Ring | Group) -> RingMember:
        """Construct a ring member from its vector of coefficients, ``(x_b : b in basis)``.

        In a group algebra, the coefficients are indexed by group members, ``(x_g : g in G)``.
        """
        if isinstance(vector, (Ring, Group)):  # DEPRECATED: see qldpc._deprecations
            _deprecations.warn_deprecated(
                "Check argument order: it should be RingMember.from_vector(vector, ring)."
                "  The order (ring, vector) is DEPRECATED and will throw an error in the future!",
                stacklevel=2,
            )
            vector, ring = ring, vector
        ring = ring if isinstance(ring, Ring) else GroupRing(ring)
        if np.shape(vector) != (ring.dim,):
            raise ValueError(
                f"A member of a ring of dimension {ring.dim} has {ring.dim} coefficients, but the"
                f" vector has shape {np.shape(vector)}"
            )
        terms = [(int(x_b), label) for x_b, label in zip(vector, ring._labels) if x_b]
        return RingMember._from_trusted_terms(ring, terms)

    def to_vector(self) -> galois.FieldArray:
        """Convert this ring member into its vector of coefficients, ``(x_b : b in basis)``.

        In a group algebra, the coefficients are indexed by group members, ``(x_g : g in G)``.
        """
        vector = self.field.Zeros(self.ring.dim)
        for val, label in self:
            vector[self.ring._index(label)] = val
        return vector

    def _translate(self, group_member: GroupMember, *, on_left: bool) -> RingMember:
        """Multiply this member of a group algebra by a group member from the left or right."""
        if not isinstance(self.ring, GroupRing):
            raise TypeError(
                f"Group members combine only with members of group algebras, not {self.ring}"
            )
        group_member = self.ring._as_label(group_member)
        new_element = self.ring.zero
        for val, label in self:
            new_element._vec[group_member * label if on_left else label * group_member] = val
        return new_element

    @classmethod
    def _from_trusted_terms(
        cls, ring: Ring, terms: Iterable[tuple[int | galois.FieldArray, Hashable]]
    ) -> RingMember:
        """Construct a ring member from terms that the caller has checked.

        Each label must be the canonical label of a basis element, as returned by Ring._as_label.
        """
        ring_member = cls(ring)
        for value, label in terms:
            ring_member._vec[label] += ring_member.field(value)
        return ring_member


if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
