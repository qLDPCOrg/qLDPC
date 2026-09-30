# SPDX-License-Identifier: Apache-2.0

"""Module for abstract algebra: group rings and their members.

.. warning::
    This module does not promise to be performant.  If you need to do heavy numerical abstract
    algebra, you're probably better served by GAP or MAGMA (or maybe SageMath).

``RingArray`` and its deprecated ``Protograph`` alias now live in ``.ring_array``, but this module
still re-exports them lazily (see ``__getattr__`` below) for backward compatibility with existing
imports and pickled objects that reference ``qldpc.abstract.rings.RingArray``.

"""

from __future__ import annotations

import collections
import copy
import functools
import itertools
import operator
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import galois
import numpy as np
import numpy.typing as npt
import sympy.abc
import sympy.core

from qldpc import external
from qldpc._util import get_deprecated_alias

from ._monomials import iter_monomial_terms
from .groups import AbelianGroup, Group, GroupMember, resolve_field

if TYPE_CHECKING:
    # re-exported for type checkers only; see the module-level __getattr__ for the runtime shim
    from .ring_array import Protograph, RingArray  # noqa: F401
    from .wedderburn_artin import WedderburnArtinTransformer

################################################################################
# group algebra and elements thereof


class GroupRing:
    """A finite group algebra over a finite field.

    The base field is ``GF(2)`` by default.
    """

    _group: Group
    _field: type[galois.FieldArray]
    _transformers: dict[int | None, WedderburnArtinTransformer]
    _idempotents: tuple[RingMember, ...] | None = None

    def __init__(self, group: Group, field: int | type[galois.FieldArray] | None = None) -> None:
        self._group = group
        self._field = resolve_field(field)
        self._transformers = {}

    @property
    def group(self) -> Group:
        """Base group of this ring."""
        return self._group

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this ring."""
        return self._field

    def get_transformer(
        self, seed: int | None = None, *, skip_validation: bool = False
    ) -> WedderburnArtinTransformer:
        """Instrument for the Wedderburn-Artin decomposition of this ring.

        By default each component's primitive central idempotent is validated; pass
        ``skip_validation=True`` to skip the check (see WedderburnArtinComponentTransformer).
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
        # Two group algebras are equal when they share a base field and underlying group; the
        # group's representation (lift) is irrelevant here, so compare with ``Group.equiv``.  This
        # keeps equality value-based, and __hash__ (below) is keyed on the same stable data.
        return (
            isinstance(other, GroupRing)
            and self.field is other.field
            and self.group.equiv(other.group)
        )

    def __hash__(self) -> int:
        return hash((self.field.order, self.group.to_sympy()))

    @property
    def name(self) -> str:
        """A name for this ring, which is not required to uniquely identify the ring."""
        return f"Group algebra of {self.group.name} over GF({self.field.order})"

    def __str__(self) -> str:
        return self.name

    @property
    def is_commutative(self) -> bool:
        """Is this ring commutative?"""
        return self._group.is_abelian

    @property
    def is_abelian(self) -> bool:
        """Is this ring abelian?

        All rings are abelian with respect to addition, so this question concerns multiplication.
        ``GroupRing.is_abelian`` is therefore an alias for ``GroupRing.is_commutative``.
        """
        return self.is_commutative

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
        return [RingMember(self, gen) for gen in self.group.generators]

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
        """
        return self.group.lift(member, right=right).view(self.field)

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

        Two idempotents g, h are orthogonal if ``g * h = h * g = 0``.

        Intuitively, primitive central idempotents act like projectors onto orthogonal simple
        components of a ring.

        See `Idempotent (ring theory)
        <https://en.wikipedia.org/wiki/Idempotent_%28ring_theory%29>`_.
        """
        if not self.is_semisimple:
            raise ValueError("Only semisimple rings have primitive central idempotents")
        if self._idempotents is None:
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
            self._idempotents = tuple(idempotents)
        return self._idempotents

    def eval(
        self, expression: sympy.Basic | int | np.int_, symbols: dict[sympy.Symbol, GroupMember]
    ) -> RingMember:
        """Convert a SymPy expression (such as a polynomial) into a member of this ring."""
        # helpful error message for invalid symbols
        if any(not isinstance(value, GroupMember) for value in symbols.values()):
            raise ValueError("The symbols passed to Ring.eval must be GroupMember-valued")

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


class RingMember:
    """An element of the algebra of a group G over a finite field ``F_q``.

    Each RingMember x is a sum of group members with coefficients in the field:
    ``x = sum_{g in G} x_g g``, with each ``x_g in F_q``.
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

    def __str__(self) -> str:
        """Write this RingMember as a polynomial."""
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
                for coeff, monomial in zip(self.to_vector(), monomials)
                if coeff
            ]

        else:
            # general-purpose fallback
            sympy_group = self.group.to_sympy()
            gen_to_symbol = {gen: symbol for gen, symbol in zip(sympy_group.generators, symbols)}
            gen_to_symbol |= {
                ~gen: 1 / symbol
                for gen, symbol in gen_to_symbol.items()
                if ~gen not in gen_to_symbol
            }

            terms = []
            for x_g, gg in self:
                gens = sympy_group.generator_product(gg, original=True)
                factors = [gen_to_symbol[gen] for gen in gens]
                monomial = functools.reduce(operator.mul, factors, 1)
                terms.append(int(x_g) * monomial)

        return str(sum(terms) + sympy.core.numbers.Zero()).replace("**", "^").replace("*", " ")

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

    def __add__(self, other: int | galois.FieldArray | GroupMember | RingMember) -> RingMember:
        if isinstance(other, (int, self.field)):
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

    def __sub__(self, other: int | galois.FieldArray | GroupMember | RingMember) -> RingMember:
        return self + (-1) * other

    def __radd__(self, other: GroupMember) -> RingMember:
        return self + other

    def __mul__(self, other: int | galois.FieldArray | GroupMember | RingMember) -> RingMember:
        if isinstance(other, int):
            other = self.ring._eval_int(other)

        if isinstance(other, self.field):
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

    def __rmul__(self, other: int | galois.FieldArray | GroupMember) -> RingMember:
        if isinstance(other, (int, self.field)):
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
        if not isinstance(power, (int, np.int_)) or power < 0:
            raise ValueError(
                "A RingMember can only be raised to an integer power >= 0."
                "\nTry ring_member.inverse() ** abs(power)"
            )
        return functools.reduce(operator.mul, [self] * power) if power > 0 else self.ring.one

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

    def lift(self, *, right: bool = False) -> galois.FieldArray:
        """Lift this ring member to a representation by a matrix.

        A representation satisfies

            ``self.lift(g·h) = self.lift(g) @ self.lift(h)``.

        If ``right=True``, lift to an anti-representation, for which

            ``self.lift(g·h) = self.lift(h) @ self.lift(g)``.
        """
        return sum(
            (val * self.ring.lift(member, right=right) for val, member in self if val),
            start=self.field.Zeros([self.group.lift_dim] * 2),
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
        terms = (val * self.ring.regular_lift(member, right=right) for val, member in self if val)
        return (
            functools.reduce(operator.add, terms)
            if bool(self)
            else self.field.Zeros([self.group.order] * 2)
        )

    @property
    def T(self) -> RingMember:
        """Transpose of this element.

        If this element is ``x = sum_{g in G} x_g g``, return ``x.T = sum_{g in G} x_g g.T``,
        where ``g.T = ~g = g**-1``.  For an orthogonal lift this matches the matrix transpose,
        ``L(g.T) = L(g).T``; for non-orthogonal lifts the group inverse ``~g`` is still used, but it
        no longer corresponds to a matrix transpose.
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
            # ``regular_lift`` M satisfies M @ s.to_vector() == (self * s).to_vector(), so the
            # inverse element solves M @ x == one.to_vector(), i.e. x = M^{-1} @ one.to_vector().
            return RingMember.from_vector(matrix_inv @ self.ring.one.to_vector(), self.ring)
        except np.linalg.LinAlgError:
            return None

    @classmethod
    def from_vector(cls, vector: npt.NDArray[np.int_], ring: GroupRing | Group) -> RingMember:
        """Construct a group algebra element from vector of coefficients, ``(x_g : g in G)``."""
        if isinstance(vector, (GroupRing, Group)):
            warnings.warn(
                "Check argument order: it should be RingMember.from_vector(vector, ring)."
                "  The order (ring, vector) is DEPRECATED and will throw an error in the future!",
                DeprecationWarning,
                stacklevel=2,
            )
            vector, ring = ring, vector
        group = ring.group if isinstance(ring, GroupRing) else ring
        terms = [(int(x_g), gg) for x_g, gg in zip(vector, group.generate()) if x_g]
        return RingMember(ring, *terms)

    def to_vector(self) -> galois.FieldArray:
        """Convert this group algebra element into a vector of coefficients, ``(x_g : g in G)``."""
        vector = self.field.Zeros(self.group.order)
        for val, member in self:
            vector[self.group.index(member)] = val
        return vector


DEPRECATED_ALIASES = {"Element": RingMember}

# Deprecated names resolve at runtime through a module-level __getattr__ that warns when accessed.
# Type checkers instead see plain aliases, so that they still flag misspelled attributes.
if TYPE_CHECKING:
    Element = RingMember
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated names, and lazily re-export names that moved to ``.ring_array``.

        The lazy re-export preserves ``qldpc.abstract.rings.RingArray`` (and the deprecated
        ``Protograph`` alias) as a valid, if no longer canonical, import path -- including for
        unpickling objects saved before the split -- without importing ``.ring_array`` eagerly at
        module load time, which would recreate the import cycle (``.ring_array`` imports
        ``GroupRing``/``RingMember`` from here) that motivated splitting it out in the first place.
        """
        if name in ("RingArray", "Protograph"):
            from . import ring_array

            return getattr(ring_array, name)
        return get_deprecated_alias(__name__, name, DEPRECATED_ALIASES)
