"""Module for abstract algebra: rings and ring-valued numpy arrays

!!! WARNINGS !!!

This module does not promise to be performant.  If you need to do heavy numerical abstract algebra,
you're probably better served by GAP or MAGMA (or maybe SageMath).


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
import operator
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, Union

import galois
import numpy as np
import numpy.typing as npt
import sympy.abc
import sympy.core

import qldpc
from qldpc import external

from .groups import DEFAULT_FIELD_ORDER, AbelianGroup, CyclicGroup, Group, GroupMember, TrivialGroup

if TYPE_CHECKING:
    from .wedderburn_artin import WedderburnArtinTransformer

################################################################################
# group algebra and elements thereof


class GroupRing:
    """A finite group algebra over a finite field.

    The base field is GF(2) by default.
    """

    _group: Group
    _field: type[galois.FieldArray]
    _transformers: dict[int | None, WedderburnArtinTransformer]
    _idempotents: tuple[RingMember, ...] | None = None

    def __init__(self, group: Group, field: int | None = None) -> None:
        self._group = group
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        self._transformers = {}

    @property
    def group(self) -> Group:
        """Base group of this ring."""
        return self._group

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this ring."""
        return self._field

    def get_transformer(self, seed: int | None = None) -> WedderburnArtinTransformer:
        """Instrument for the Wedderburn-Artin decomposition of this ring."""
        from .wedderburn_artin import WedderburnArtinTransformer  # avoid circular import

        if seed not in self._transformers:
            if seed is None and self._transformers:
                return next(iter(self._transformers.values()))
            self._transformers[seed] = WedderburnArtinTransformer(self, seed=seed)
        return self._transformers[seed]

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
    def is_commutative(self) -> bool:
        """Is this ring commutative?"""
        return isinstance(self, AbelianGroup) or self._group.is_abelian

    @property
    def is_abelian(self) -> bool:
        """Is this ring abelian?

        All rings are abelian with respect to addition, so this question concerns multiplication.
        GroupRing.is_abelian method is therefore an alias for GroupRing.is_commutative.
        """
        return self.is_commutative

    @functools.cached_property
    def is_semisimple(self) -> bool:
        """Is this ring semisimple?"""
        return bool(self.group.order % self.field.characteristic)

    @functools.cached_property
    def group_trace_matrix(self) -> galois.FieldArray:
        """Construct the matrix for a trace over the group: r -> sum_{g in G} g r g^{-1}."""
        adjoints = [self.group.adjoint_lift(gg).view(self.field) for gg in self.group.generate()]
        return functools.reduce(operator.add, adjoints).view(self.field)

    @property
    def generators(self) -> list[RingMember]:
        """Generators of this ring's base group."""
        return [RingMember(self, gen) for gen in self.group.generators]

    def regular_lift(self, member: GroupMember, *, right: bool = False) -> galois.FieldArray:
        """Lift a group member to its regular representation.

        See help(qldpc.abstract.Group.regular_lift) for more information.
        """
        return self.group.regular_lift(member, right=right).view(self.field)

    def lift(self, member: GroupMember, *, right: bool = False) -> galois.FieldArray:
        """Lift a group member to a representation by an orthogonal matrix.

        A representation satisfies
            self.lift(g·h) = self.lift(g) @ self.lift(h).
        If right=True, lift to an anti-representation, for which
            self.lift(g·h) = self.lift(h) @ self.lift(g).
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
        Two idempotents g, h are orthogonal if g * h = h * g = 0.

        Intuitively, primitive central idempotents idempotents act like projectors onto orthogonal
        simple components of a ring.

        See https://en.wikipedia.org/wiki/Idempotent_(ring_theory).
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
        if isinstance(expression, (sympy.Poly, sympy.Add)):
            # evaluate this polynomial one monomial term at time
            terms = sympy.Add.make_args(expression.as_expr())
            evaluated_terms = [self.eval(term, symbols) for term in terms]
            return functools.reduce(operator.add, evaluated_terms)

        # helpful error message for invalid symbols
        if any(not isinstance(value, GroupMember) for value in symbols.values()):
            raise ValueError("The symbols passed to Ring.eval must be GroupMember-valued")

        # if applicable, convert python integers into SymPy integers
        if isinstance(expression, (int, np.int_)):
            expression = sympy.Integer(expression)

        # factor this term into its coefficient and variable content
        _coeff, monomial = expression.as_coeff_Mul()
        coeff = self._eval_int(int(_coeff))

        # construct and return a member of this ring
        group_member = self.group.eval(monomial, symbols)
        return RingMember(self, (coeff, group_member))

    def _eval_int(self, value: int) -> galois.FieldArray:
        """Evaluate an integer as an element of the base field of this ring.

        Some integers may have "invalid" but unambiguous interpretations as field members.
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
        """Lift this ring member to a representation by an orthogonal matrix.

        A representation satisfies
            self.lift(g·h) = self.lift(g) @ self.lift(h).
        If right=True, lift to an anti-representation, for which
            self.lift(g·h) = self.lift(h) @ self.lift(g).
        """
        return sum(
            (val * self.ring.lift(member, right=right) for val, member in self if val),
            start=self.field.Zeros([self.group.lift_dim] * 2),
        )

    def regular_lift(self, *, right: bool = False) -> galois.FieldArray:
        """Lift a ring member to its regular representation.

        By default, this method lifts a ring member to the regular representation induced by
        multiplication from the left.  Specifically, if r and s are ring members, then
            r.regular_lift() @ s.to_vector() = (r * s).to_vector().

        If right is True, this method lifts a ring member to its regular representation in the
        opposite ring, such that matrix multiplication corresponds to ring multiplication from the
        right:
            r.regular_lift(right=True) @ s.to_vector() = (s * r).to_vector().
        See https://en.wikipedia.org/wiki/Opposite_ring.
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
            return RingMember.from_vector(matrix_inv[:, 0], self.ring)
        except np.linalg.LinAlgError:
            return None

    @classmethod
    def from_vector(cls, vector: npt.NDArray[np.int_], ring: GroupRing | Group) -> RingMember:
        """Construct a group algebra element from vector of coefficients, (x_g : g in G)."""
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
                    "Requirement failed: all entries of a RingArray must be RingMember-valued."
                    "\nTry building an array with RingArray.build(...)"
                )
            else:
                if not (ring is None or ring == value.ring):
                    raise ValueError("Inconsistent rings provided for a RingArray")
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

    def __str__(self) -> str:
        return np.array2string(self, formatter={"object": str}, separator=", ")

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

    def regular_lift(self, *, right: bool = False) -> galois.FieldArray:
        """Block matrix obtained by a regular lift of each entry of this RingArray."""
        assert self.ndim == 1 or self.ndim == 2
        rows = 1 if self.ndim == 1 else self.shape[0]
        cols = self.shape[-1]
        block_size = self.group.order
        if 0 in (rows, cols):
            return self.field.Zeros((rows * block_size, cols * block_size))
        blocks = [
            [val.regular_lift(right=right) for val in row]
            for row in self.reshape(-1, self.shape[-1])
        ]
        return np.block(blocks).view(self.field)

    def lift(self, *, right: bool = False) -> galois.FieldArray:
        """Block matrix obtained by lifting each entry of this RingArray."""
        assert self.ndim == 1 or self.ndim == 2
        rows = 1 if self.ndim == 1 else self.shape[0]
        cols = self.shape[-1]
        block_size = self.group.lift_dim
        if 0 in (rows, cols):
            return self.field.Zeros((rows * block_size, cols * block_size))
        blocks = [
            [val.lift(right=right) for val in row] for row in self.reshape(-1, self.shape[-1])
        ]
        return np.block(blocks).view(self.field)

    def __invert__(self) -> RingArray:
        """Invert (transpose) the entries of this RingArray."""
        vals = [val.T for val in self.ravel()]
        array = np.array(vals, dtype=object).reshape(self.shape).view(RingArray)
        array._ring = self._ring
        return array

    @property
    def T(self) -> RingArray:
        """Conjugate-transpose of a matrix over a ring.

        In addition to transposing the first two indices of the array, this method "conjugates" or
        "transposes" each array element, which takes group members g -> ~g = g**-1.
        """
        return (~self).transpose(1, 0, *np.arange(2, self.ndim))

    @staticmethod
    def build(
        data: npt.NDArray[np.int_] | npt.NDArray[np.object_] | NestedSequence,
        ring: GroupRing | Group | None = None,
    ) -> RingArray:
        """Construct a RingArray.

        The constructed array is built from:
        1. An array populated by
            (a) ring members,
            (b) group members, or
            (c) integers.
        2. A ring (or group, inducing a group algebra over GF(2)).
        Integers and group members are cast into members of the ring.
        """
        array = np.asanyarray(data)

        # identify the base ring and group
        if ring is None:
            rings = {value.ring for value in array.ravel() if isinstance(value, RingMember)}
            if not len(set(rings)) <= 1:
                raise ValueError("Inconsistent rings provided to RingArray.build")
            if rings:
                ring = next(iter(rings))
            else:
                field = type(array).order if isinstance(array, galois.FieldArray) else None
                ring = GroupRing(TrivialGroup(), field)
        ring = ring if isinstance(ring, GroupRing) else GroupRing(ring)
        one = ring.group.identity

        def as_ring_member(value: RingMember | GroupMember | int) -> RingMember:
            """Elevate a value to an element of the ring."""
            if isinstance(value, RingMember):
                _value = value.copy() * one
                _value._ring = ring
                return _value
            if isinstance(value, GroupMember):
                return RingMember(ring, value * one)
            return RingMember(ring, (value, one))

        vals = [as_ring_member(value) for value in array.ravel()]
        result = np.array(vals, dtype=object).reshape(array.shape).view(RingArray)
        result._ring = ring
        return result

    def to_field_array(self) -> galois.FieldArray:
        """Convert a RingArray into an array of coefficients (in a finite field) for each entry.

        This method expands every entry of a RingArray into a vector of length ring.group.order.
        If ring_array is two-dimensional, for example, then ring_array.to_field_array()[a, b, :] is
        the vector of coefficients for the RingMember at ring_array[a, b].
        """
        vals = [val.to_vector() for val in self.ravel()]
        return np.asarray(vals, dtype=int).reshape(*self.shape, self.group.order).view(self.field)

    @classmethod
    def from_field_array(cls, array: npt.NDArray[np.int_], ring: GroupRing | Group) -> RingArray:
        """Construct a RingArray from an array of coefficients (in a finite field) for each entry.

        This method is the inverse of RingArray.to_field_array.
        """
        if isinstance(array, (GroupRing, Group)):
            warnings.warn(
                "Check argument order: it should be RingArray.from_field_array(array, ring)."
                "  The order (ring, array) is DEPRECATED and will throw an error in the future!",
                DeprecationWarning,
                stacklevel=2,
            )
            array, ring = ring, array
        array = np.asanyarray(array)
        group = ring.group if isinstance(ring, GroupRing) else ring
        vectors = array.reshape(array.size // group.order, group.order)
        vals = [RingMember.from_vector(vector, ring) for vector in vectors]
        result = np.array(vals, dtype=object).reshape(array.shape[:-1]).view(RingArray)
        result._ring = ring if isinstance(ring, GroupRing) else GroupRing(ring)
        return result

    def to_field_vector(self) -> galois.FieldArray:
        """Convert RingArray into a flattened 1-D vector of coefficients for each RingMember."""
        return self.to_field_array().ravel().view(self.field)

    @classmethod
    def from_field_vector(cls, vector: npt.NDArray[np.int_], ring: GroupRing | Group) -> RingArray:
        """Construct a 1-D RingArray from a vector of coefficients.

        This method is the inverse of RingArray.to_field_vector.
        """
        if isinstance(vector, (GroupRing, Group)):
            warnings.warn(
                "Check argument order: it should be RingArray.from_field_vector(vector, ring)."
                "  The order (ring, vector) is DEPRECATED and will throw an error in the future!",
                DeprecationWarning,
                stacklevel=2,
            )
            vector, ring = ring, vector
        vector = np.asanyarray(vector)
        group = ring.group if isinstance(ring, GroupRing) else ring
        entries_as_vecs = vector.reshape(vector.size // group.order, group.order)
        return RingArray.from_field_array(entries_as_vecs, ring)

    def null_space(self, *, right: bool = False) -> RingArray:
        """Construct a matrix of null-space row vectors for this RingArray.

        The transpose of the null-space matrix is annihilated by this RingArray, such that
        np.any(self @ self.null_space().T) is np.False_.

        If right is True, this method constructs a null space over the opposite ring, in which the
        order of multiplication is reversed.

        Due to the subtleties of defining row reduction for a matrix over a ring, this method does
        not row-reduce the matrix of null-space row vectors.  The rows of the matrix returned by
        this method are therefore generally an overcomplete basis for the null space of this
        RingArray.
        """
        assert self.ndim == 2

        # field-valued null vectors of self.regular_lift() provide an overcomplete basis for
        # the space of ring-valued null vectors
        null_field_vectors = self.regular_lift(right=right).null_space()

        # collect ring-valued null row vectors (that is, transposed null column vectors)
        field_array_shape = (len(null_field_vectors), self.shape[1], self.group.order)
        return ~RingArray.from_field_array(null_field_vectors.reshape(field_array_shape), self.ring)

    def row_reduce(self, transformer: WedderburnArtinTransformer | None = None) -> RingArray:
        """Compute a generalized reduced row echelon form of a RingArray over a semisimple ring.

        This method relies on the Wedderburn-Artin decomposition:
        1. Decompose the matrix over a ring into matrices over simple components.
        2. Put the matrices over simple components into RREF.
        3. Re-combine the simple components into a matrix over the original ring.

        The RREF of a RingArray over a commutative ring is unique.  For non-commutative rings, the
        RREF is only unique up to a choice of matrix basis for simple components of the ring.
        """
        assert self.ndim == 2
        if not self.ring.is_semisimple:
            raise ValueError("RingArray.row_reduce only supports semisimple rings")
        transformer = transformer or self.ring.get_transformer()
        matrices = [
            component.row_reduce()
            for component in transformer.decompose_array(self, merge_blocks=True)
        ]
        return transformer.recompose_array(matrices, from_blocks=True)

    def howell_normal_form(self, *, poly: bool = False) -> RingArray:
        """Compute a Howell normal form of this RingArray.

        Alias for:
            - RingArray.howell_normal_form_semisimple (if poly is False, the default), or
            - RingArray.howell_normal_form_poly (if poly is True).
        See the documentation of those methods for additional information.
        """
        if poly:
            return self.howell_normal_form_poly()
        return self.howell_normal_form_semisimple()

    def howell_normal_form_semisimple(
        self, transformer: WedderburnArtinTransformer | None = None, *, right: bool = False
    ) -> RingArray:
        """Compute a Howell normal form (HNF) of a RingArray over a semisimple ring.

        This method first puts a RingArray into a generalized reduced row echelon form (see
        RingArray.row_reduce), then further post-processes the rows to satisfy the Howell property,
        whereby an element v that is...
            - in the row span of the matrix, and
            - has j leading zeros, meaning = (0_1, 0_2, ..., 0_j, v_{j+1}, ...),
        can be written as a linear combinations of rows whose pivots are at position k >= j.

        The Howell property is enforecd as follows: if a row r has a pivot p with a nontrivial left
        annihilator α, meaning
              α != 0,
            α·p  = 0,
            α·r != 0,
        then the row r is replaced by (1-α)·r, and the row α·r is appended to the matrix.

        If right is True, the Howell property is instead enforced for nontrivial right annihilators:
              α != 0,
            p·α  = 0,
            r·α != 0,
        for which the row r is replaced by (1-α)·r, and the row α·r is appended to the matrix.
        The ordinary HNF and right-HNF are equal for a RingArray over a commutative ring.

        The HNF of a RingArray over a commutative ring is unique.  For non-commutative rings, the
        HNF is only unique up to a choice of matrix basis for simple components of the ring.

        References:
        - https://en.wikipedia.org/wiki/Howell_normal_form
        - https://github.com/m-webster/XPFpackage/blob/570ea89/Examples/A.1_howell_matrix.ipynb
        """
        assert self.ndim == 2
        if not self.ring.is_semisimple:
            raise ValueError(
                "The ordinary Howell normal form requires the base ring to be semisimple"
            )
        transformer = transformer or self.ring.get_transformer()
        num_components = len(transformer.transformers)

        # identify and row-reduce the components of this RingArray
        matrices = [
            _get_block_howell_form(component_transformer.project_array(self), right=right)
            for component_transformer in transformer.transformers
        ]

        # pad zero rows to components that have fewer rows
        num_rows = max(len(matrix) for matrix in matrices)
        for mm, matrix in enumerate(matrices):
            if pad := num_rows - len(matrix):
                field = type(matrix)
                stack = [matrix, field.Zeros((pad, *matrix.shape[1:]))]
                matrices[mm] = np.concatenate(stack).view(field)

        pivot_row = 0
        pivot_col = 0
        num_rows, num_cols = matrices[0].shape[:2]
        while pivot_row < num_rows and pivot_col < num_cols - 1:
            """
            Identify:
            1. The column of the first nonzero value in the pivot_row of each component.
            2. The column that will contain the pivot when we recombine the components.
            """
            pivot_rows_as_bools = [
                np.any(matrix[pivot_row].view(np.ndarray).astype(bool), axis=(1, 2))
                for matrix in matrices
            ]
            pivot_cols = qldpc.math.first_nonzero_cols(pivot_rows_as_bools)
            pivot_col = min(pivot_cols)

            """
            Let π be a projector onto the components in which the pivot is nonzero.  If π != 1, then
            (1-π) is a nontrivial annihilator of the pivot.  If, moreover, (1-π)·r is nonzero, then
            (1-π)·r contains a "hidden" pivot in a later column.  In this case, we in principle need
            to replace r -> π·r and add (1-π)·r as a new row to the matrix.  In practice, this
            procedure messes up the reduced row echelon form of the matrix, so we instead...
            1. In the (1-π) sector, insert a zero row at the pivot_row and shift down rows below.
            2. In the π sector, append a zero row to the matrix.
            """
            components_with_hidden_pivots = [
                cc for cc in range(len(matrices)) if pivot_col < pivot_cols[cc] < num_cols
            ]
            if components_with_hidden_pivots:
                for cc in range(num_components):
                    matrix = matrices[cc]
                    size = transformer.transformers[cc].size
                    field = type(matrix)
                    zero_row = field.Zeros((1, num_cols, size, size))
                    if cc in components_with_hidden_pivots:
                        stack = [matrix[:pivot_row], zero_row, matrix[pivot_row:]]
                    else:
                        stack = [matrix, zero_row]
                    matrices[cc] = np.concatenate(stack).view(field)
                num_rows += 1

            pivot_row += 1

        # remove rows that are zero in all components and return
        nonzero_rows = functools.reduce(
            np.bitwise_or,
            [np.any(matrix, axis=(1, 2, 3)) for matrix in matrices],
        )
        matrices = [matrix[nonzero_rows] for matrix in matrices]
        return transformer.recompose_array(matrices)

    def howell_normal_form_poly(self) -> RingArray:
        """Compute a Howell normal form of a RingArray using polynomial division.

        If the base ring of a RingArray is a cyclic group algebra, then it can be interpreted as a
        univariate polynomial ring, allowing us to compute greatest common divisors and perform row
        reduction with polynomial division.

        References:
        - https://en.wikipedia.org/wiki/Howell_normal_form
        - https://github.com/m-webster/XPFpackage/blob/570ea89/Examples/A.1_howell_matrix.ipynb
        """
        assert self.ndim == 2
        if not isinstance(self.group, CyclicGroup):
            raise ValueError(
                "The Howell normal form induced by polynomial division requires an underlying"
                f" CyclicGroup, not {self.group}"
            )

        # convert into 3-D, where the third dimension stores coefficients for group members
        field_array = self.to_field_array()

        # The "modulus" of underlying polynomial ring for this RingArray: x^n - 1.
        # Analogous to N in the ring of integers modulo N.
        modulus_poly = galois.Poly([1] + [0] * (self.group.order - 1) + [-1], self.field)

        def _multiply(poly: galois.Poly, vecs: galois.FieldArray) -> galois.FieldArray:
            """Multiply a member of a polynomial ring into a ring-valued matrix.

            The first argument represents a ring member by a polynomial, while the second argument
            represents a (vecs.ndim-1)-dimensional array of polynomials, such that
            vecs[*entry, c] is the coefficient of x^c in the given entry of vec.
            """
            new_vecs = vecs.Zeros(vecs.shape)
            for coeff, degree in zip(poly.nonzero_coeffs, poly.nonzero_degrees):
                new_vecs += coeff * np.roll(vecs, degree, axis=-1)
            return new_vecs

        pivot_row = 0
        pivot_col = 0
        while pivot_row < field_array.shape[0] and pivot_col < field_array.shape[1]:
            # look for a pivot in this column
            pivot_found = False
            for row in range(pivot_row, field_array.shape[0]):
                if np.any(field_array[row, pivot_col]):
                    field_array[[pivot_row, row]] = field_array[[row, pivot_row]]
                    pivot_found = True
                    break

            if not pivot_found:
                pivot_col += 1
                continue

            # use invertible row operations to zero out all rows below at the pivot column
            for other_row in range(pivot_row + 1, self.shape[0]):
                aa_vec = field_array[pivot_row]
                bb_vec = field_array[other_row]
                if not np.any(bb_vec):
                    continue
                """
                Let:
                    aa = aa_vec[pivot_row]
                    bb = bb_vec[other_row]
                We will transform rows as
                    [aa_vec, bb_vec] --> [[ss, tt], [uu, vv]] @ [aa_vec, bb_vec]
                where
                    (1) ss * aa + tt * bb = gcd(aa, bb) = gg
                    (2) uu * aa + vv * bb = 0
                    (3) det([[ss, tt], [uu, vv]]) = ss * vv - tt * uu = 1
                Condition (3) ensures that this transformation is invertible.
                Condition (2) ensures that bb_vec gets zeroed out at the pivot column.
                """
                aa_poly = galois.Poly(aa_vec[pivot_col, ::-1], field=self.field)
                bb_poly = galois.Poly(bb_vec[pivot_col, ::-1], field=self.field)

                # find gg, ss, tt, uu, vv, and work around some typing bugs/errors in galois/mypy
                gg_poly: galois.Poly
                ss_poly: galois.Poly
                tt_poly: galois.Poly
                gg_poly, ss_poly, tt_poly = galois.egcd(aa_poly, bb_poly)  # type:ignore[assignment,arg-type]
                uu_poly = -bb_poly // gg_poly
                vv_poly = aa_poly // gg_poly

                new_aa_vec = _multiply(ss_poly, aa_vec) + _multiply(tt_poly, bb_vec)
                new_bb_vec = _multiply(uu_poly, aa_vec) + _multiply(vv_poly, bb_vec)
                field_array[pivot_row] = new_aa_vec
                field_array[other_row] = new_bb_vec

            """
            "Reduce" the pivot:
            (1) Find ff for which ff * pivot = gcd(pivot, modulus).
            (2) Multiply the pivot row by ff, reducing the pivot to gcd(pivot, modulus).
            """
            pivot_poly = galois.Poly(field_array[pivot_row, pivot_col, ::-1], field=self.field)
            gcd_poly: galois.Poly
            ff_poly: galois.Poly
            gcd_poly, ff_poly, _ = galois.egcd(pivot_poly, modulus_poly)  # type:ignore[assignment,arg-type]
            if pivot_poly != gcd_poly:
                field_array[pivot_row] = _multiply(ff_poly, field_array[pivot_row])
                pivot_poly = gcd_poly

            """
            Reduce all rows above the pivot_row at the pivot_column.
            If some value in the pivot_col above the pivot_row can be written as a multiple of the
            pivot plus a remainder, use row operations to subtract off that multiple of the pivot,
            leaving only the remainder.
            """
            for other_row in range(pivot_row):
                other_poly = galois.Poly(field_array[other_row, pivot_col, ::-1], field=self.field)
                div_poly = other_poly // pivot_poly
                if div_poly != 0:
                    field_array[other_row] -= _multiply(div_poly, field_array[pivot_row])

            """
            Check whether the pivot has a nontrivial annihilator, with annihilator * pivot = 0.
            If a nontrivial annihilator is found, append a new row with the pivot annihilated.
            """
            annihilator_poly = modulus_poly // pivot_poly
            if annihilator_poly != 0:
                new_row = _multiply(annihilator_poly, field_array[pivot_row])
                field_array = np.append(field_array, [new_row], axis=0).view(self.field)

            pivot_row += 1
            pivot_col += 1

        # remove all-zero rows and return
        field_array = field_array[np.any(field_array, axis=(1, 2))]
        return RingArray.from_field_array(field_array, self.ring)

    def reduced_groebner_basis(self) -> RingArray:
        """Compute a reduced Groebner basis for this RingArray.

        At least, that the plan.  This method is not yet implemented.
        """
        assert self.ndim == 2
        raise NotImplementedError(
            "Computing a reduced Groebner basis is very mathematically involved.  Here be dragons."
        )


class Protograph(RingArray):  # pragma: no cover
    """Deprecated alias for RingArray."""

    def __getattribute__(self, name: str) -> Any:
        warnings.warn(
            f"{Protograph} is DEPRECATED; use {RingArray} instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return super().__getattribute__(name)


def _get_block_howell_form(matrix: galois.FieldArray, *, right: bool = False) -> galois.FieldArray:
    """Compute a block-Howell normal form of the provided block matrix.

    The provided matrix should be 4-dimensional, with matrix[i, j] storing a square block at (i, j).
    The block-Howell form is essentially the same as the row-reduced echelon form when the matrix
    is expanded into a 2-dimensional array, except zero rows are inserted to shift pivots down so
    that they always lie on the diagonal of a block.
    """
    shape: tuple[int, ...]

    assert matrix.ndim == 4 and matrix.shape[-1] == matrix.shape[-2]
    field = type(matrix)
    num_block_rows, num_block_cols, size, _ = matrix.shape

    if right and size > 1:
        matrix = matrix.transpose(0, 1, 3, 2)

    # row-reduce as an expanded 2-D matrix and remove all-zero rows
    shape = (num_block_rows * size, num_block_cols * size)
    matrix = matrix.transpose(0, 2, 1, 3).reshape(shape).view(field).row_reduce()
    matrix = matrix[qldpc.math.first_nonzero_cols(matrix) < matrix.shape[1]].view(field)

    if size > 1:
        # insert zero rows to shift pivots down so that they always lie on the diagonal of a block
        pivot_row, pivot_col = 0, 0
        num_cols = matrix.shape[1]
        while pivot_row < matrix.shape[0]:
            pivot_col = qldpc.math.first_nonzero_cols(matrix[pivot_row])[0]
            if pivot_row % size == 0:
                pivot_block_col = pivot_col // size
            if pad := pivot_col - pivot_block_col * size - pivot_row % size:
                zero_rows = np.zeros((pad, num_cols), dtype=int)
                matrix = np.vstack([matrix[:pivot_row], zero_rows, matrix[pivot_row:]]).view(field)
                pivot_row += pad
            pivot_row += 1

        # pad with zero rows on the bottom to ensure that all blocks have the correct size
        if tail := matrix.shape[0] % size:  # pragma: no cover
            zero_rows = np.zeros((size - tail, num_cols), dtype=int)
            matrix = np.vstack([matrix, zero_rows]).view(field)

    # re-collect into a 4-D array
    shape = (matrix.shape[0] // size, size, num_block_cols, size)
    matrix = matrix.reshape(shape).transpose(0, 2, 1, 3).view(field)

    if right and size > 1:
        matrix = matrix.transpose(0, 1, 3, 2)

    return matrix
