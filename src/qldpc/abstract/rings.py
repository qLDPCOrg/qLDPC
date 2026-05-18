"""Module for abstract algebra: rings and ring-valued numpy arrays

!!! WARNINGS !!!

First and foremost, this module does not promise to be performant.  If you need to do heavy
numerical abstract algebra, you're probably better served by GAP or MAGMA (or maybe SageMath).


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
import dataclasses
import functools
import itertools
import math
import operator
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal, Union

import galois
import numpy as np
import numpy.typing as npt
import sympy.abc
import sympy.core

import qldpc
from qldpc import external

from .groups import DEFAULT_FIELD_ORDER, AbelianGroup, CyclicGroup, Group, GroupMember, TrivialGroup

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

    def lift(self, member: GroupMember) -> galois.FieldArray:
        """Lift a group member to a representation by an orthogonal matrix."""
        return self.group.lift(member).view(self.field)

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

    def lift(self) -> galois.FieldArray:
        """Lift this ring member to a representation by an orthogonal matrix."""
        return sum(
            (val * self.ring.lift(member) for val, member in self if val),
            start=self.field.Zeros([self.group.lift_dim] * 2),
        )

    def regular_lift(self, *, right: bool = False) -> galois.FieldArray:
        """Construct a matrix that encodes multiplication in the ring by matrix multiplication.

        By default, the matrix constructed by this method represents multiplication from the left,
        meaning that if r and s are ring members, then
            r.regular_lift() @ s.to_vector() = (r * s).to_vector().
        If right is True, then matrix multiplication corresponds to ring multiplication from the
        right, meaning
            r.regular_lift(right=True) @ s.to_vector() = (s * r).to_vector().

        A potential point of confusion: right-multiplication in the ring should not be confused with
        the right-regular representation of group members, which also requires taking the inverse of
        group members.
        """
        if not right:
            terms = (val * self.ring.regular_lift(member) for val, member in self if val)
        else:
            terms = (
                val * self.ring.regular_lift(~member, right=True) for val, member in self if val
            )
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
        blocks = [
            [val.regular_lift(right=right) for val in row]
            for row in self.reshape(-1, self.shape[-1])
        ]
        return np.block(blocks).view(self.field)

    def lift(self) -> galois.FieldArray:
        """Block matrix obtained by lifting each entry of this RingArray."""
        assert self.ndim == 1 or self.ndim == 2
        blocks = [[val.lift() for val in row] for row in self.reshape(-1, self.shape[-1])]
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
        data: npt.NDArray[np.int_] | npt.NDArray[np.object_] | NestedSequence,
        ring: GroupRing | Group | None = None,
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
        array = np.asanyarray(data)

        # identify the base ring and group
        if ring is None:
            rings = [value.ring for value in array.ravel() if isinstance(value, RingMember)]
            if not len(set(rings)) <= 1:
                raise ValueError("Inconsistent rings provided to RingArray.build")
            if rings:
                ring = rings[0]
            else:
                field = type(array).order if isinstance(array, galois.FieldArray) else None
                ring = GroupRing(TrivialGroup(), field)
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
        return RingArray(np.array(vals, dtype=object).reshape(array.shape[:-1]), ring=ring)

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

    def null_space(self) -> RingArray:
        """Construct a matrix of null-space row vectors for this RingArray.

        The transpose of the null-space matrix is annihilated by this RingArray, such that
        np.any(self @ self.null_space().T) is np.False_.

        Due to the subtleties of defining row reduction for a matrix over a ring, this method does
        not row-reduce the matrix of null-space row vectors.  The rows of the matrix returned by
        this method are therefore generally an overcomplete basis for the null space of this
        RingArray.
        """
        assert self.ndim == 2

        # field-valued null vectors of self.regular_lift() provide an overcomplete basis for
        # the space of ring-valued null vectors
        null_field_vectors = self.regular_lift().null_space()

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
        self, transformer: WedderburnArtinTransformer | None = None
    ) -> RingArray:
        """Compute a Howell normal form (HNF) of a RingArray over a semisimple ring.

        This method first puts a RingArray into a generalized reduced row echelon form (see
        RingArray.row_reduce), then further post-processes the rows to satisfy the Howell property.
        Specifically, if a row r has a pivot p with a nontrivial annihilator α (meaning α != 0 and
        α·p = 0), then the row r is replaced by (1-α)·r, and the row α·r is appended to the matrix.

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
            _get_block_howell_form(component_transformer.project_array(self))
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
            (1-π)·r contains a "hidden" pivot in a later column.  In this caes, we in principle need
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


class WedderburnArtinTransformer:
    r"""Instrument for implementing the Wedderburn-Artin decomposition of semisimple rings.

    The Wedderburn-Artin theorem states that every semisimple ring R is isomorphic to a direct
    product of matrix algebras over division rings:
        R ≅ ⨂_i R_i
    where
        R_i = D_i^{n_i × n_i},
    and D^{n × n} denotes the space of n × n matrices over the division ring D.  By Wedderburn's
    little theorem, every finite division ring is a finite field, so if R is a group algebra over a
    finite field F then every division ring D_i is a field extension of F.  If R is a commutative
    ring, then all n_i = 1, so
        R = ⨂_i D_i  (if R is commutative).

    This class is an instrument for decomposing elements of r ∈ R into simple components, taking
        r -> (r_1, r_2, ...) ∈ ⨂_i R_i,
    and and embedding elements of ⨂_i R_i back into R.

    References:
    - https://en.wikipedia.org/wiki/Wedderburn%E2%80%93Artin_theorem
    - https://en.wikipedia.org/wiki/Wedderburn%27s_little_theorem
    """

    ring: GroupRing
    transformers: list[WedderburnArtinComponentTransformer]
    random_number_generator: np.random.Generator

    def __init__(self, ring: GroupRing, *, seed: np.random.Generator | int | None = None) -> None:
        if not ring.is_semisimple:
            raise ValueError("The Wedderburn-Artin decomposition only exists for semisimple rings")
        self.ring = ring
        self.random_number_generator = (
            seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
        )
        self.transformers = [
            WedderburnArtinComponentTransformer(pci, seed=self.random_number_generator)
            for pci in self.ring.get_primitive_central_idempotents()
        ]

    def decompose(self, element: RingMember) -> list[galois.FieldArray]:
        """Decompose an element of a ring into its Wedderburn-Artin components."""
        return [transformer.project(element) for transformer in self.transformers]

    def decompose_array(
        self, array: RingArray, *, merge_blocks: bool = False
    ) -> list[galois.FieldArray]:
        """Decompose a RingArray element-wise into Wedderburn-Artin components.

        Each component of N-dimensional RingArray is an (N+2)-dimensional galois.FieldArray.

        If merge_blocks is True, this method treats each projected element as a block matrix in the
        last two axes of the provided array, such that a projection with shape
            (..., r, c, rb, cb)
            is transposed and reshaped into an array with shape
            (..., r * rb, c * cb).
        """
        return [
            transformer.project_array(array, merge_blocks=merge_blocks)
            for transformer in self.transformers
        ]

    def recompose(self, components: Sequence[galois.FieldArray]) -> RingMember:
        """Invert WedderburnArtinTransformer.decompose."""
        if len(components) != len(self.transformers):
            raise ValueError(
                f"Provided {len(components)} WedderburnArtinTransformer components for a ring that"
                f" should have {len(self.transformers)}"
            )
        terms = [trans.embed(comp) for comp, trans in zip(components, self.transformers)]
        return functools.reduce(operator.add, terms)

    def recompose_array(
        self, components: Sequence[galois.FieldArray], *, from_blocks: bool = False
    ) -> RingArray:
        """Invert WedderburnArtinTransformer.decompose_array."""
        if len(components) != len(self.transformers):
            raise ValueError(
                f"Provided {len(components)} WedderburnArtinTransformer components for a ring that"
                f" should have {len(self.transformers)}"
            )
        if not len(set([array.shape[:-2] for array in components])) == 1:
            raise ValueError("Asked to combine arrays of inconsistent shapes")
        terms = [
            trans.embed_array(array, from_blocks=from_blocks)
            for array, trans in zip(components, self.transformers)
        ]
        return functools.reduce(operator.add, terms)


@dataclasses.dataclass
class WedderburnArtinComponentTransformer:
    r"""Transformer to map between a semisimple ring R and a simple component S of R.

    Let R = GF(q)[G] ≅ GF(q)^{|G|} be a finite group algebra, whose elements can be formally written
    as a GF(q)-linear combination of group members in G.  That is, if r ∈ R, then
        r = sum_{g in G} r_g g.
    By the Wedderburn-Artin theorem and Wedderburn's little theorem, every
    simple component S of R is isomorphic to the space of matrices over a field extension of GF(q):
        S ≅ GF(q^d)^{n × n}.
    We refer to d as the "degree" of S, and n as the "size" of S.
    If G is abelian, then its size is n = 1, so S ≅ GF(q^d).

    The simple component S can be identified by a primitive central idempotent (PCI) e ∈ R, where
        1. "Idempotent" means that e is a projector: e·e = e.
        2. "Central" means that e commutes with all of R.
        3. "Primitive" means that e cannot be decomposed into a sum of central idempotents in R.
    Given a PCI e of R, the corresponding simple component is S = R·e = { r·e : r ∈ R }.

    This class is an instrument for projecting elements of R onto a simple component S corresponding
    to a provided PCI e, and embedding elements of S back into R.

    References:
    - https://en.wikipedia.org/wiki/Wedderburn%E2%80%93Artin_theorem
    - https://en.wikipedia.org/wiki/Wedderburn%27s_little_theorem
    """

    ring: GroupRing  # base ring, R
    field: type[galois.FieldArray]  # base field of the ring, F = GF(q) = GF(p^k)

    pci: RingMember  # primitive central idempotent (PCI) e that projects onto S
    pci_vec: galois.FieldArray  # representation of the PCI as a vector in GF(q)^{|G|}
    pci_reg: galois.FieldArray  # PCI lifted to its regular representation in GF(q)^{|G| × |G|}

    center: galois.FieldArray  # basis for the center Z(S) of elements in S that commute with S
    degree: int  # degree d of the field extension GF(q^d) for S
    size: int  # size (n) of the matrices in the isomorphism S ≅ GF(q^d)^{n × n}

    power_basis: galois.FieldArray  # power basis B = (e, b, b^2, ..., b^{d-1}) for GF(q^d)
    power_basis_dual: galois.FieldArray  # dual A of the power basis, for which A @ B.T = 1_d

    extended_field: type[galois.FieldArray]  # field extension GF(p^(kd)) ≅ GF(q^d)
    embedded_scalars: galois.FieldArray  # embedding of GF(q) into GF(p^(kd))
    embedded_scalars_inverse: galois.FieldArray  # inverse of the embedding of GF(q) into GF(p^(kd))
    embedded_power_basis: galois.FieldArray  # embedding of B into GF(p^(kd))
    embedded_power_basis_dual: galois.FieldArray  # dual of embedded_power_basis w.r.t. field trace

    matrix_basis: galois.FieldArray  # matrix elements |i><j| for GF(q^d)^{n × n}

    # matrices to project R -> S ≅ GF(q^d)^{n × n} ≅ GF(q)^{n × n × d}, and embed back into R
    decomposition_coefficient_extractor: galois.FieldArray
    decomposition_coefficient_recombiner: galois.FieldArray

    def __init__(self, pci: RingMember, *, seed: np.random.Generator | None = None) -> None:
        """Initialize from a primitive central idempotent (PCI) of a ring.

        WARNING: This class assumes--and does not verify--that the provided RingMember is a PCI.
        """
        if not pci.ring.is_semisimple:
            raise ValueError("The Wedderburn-Artin decomposition only exists for semisimple rings")
        seed = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)

        self.ring = pci.ring
        self.field = self.ring.field

        self.pci = pci
        self.pci_vec = pci.to_vector()
        self.pci_reg = pci.regular_lift()

        self.center = self._get_center()
        self.degree = len(self.center)
        self.size = math.isqrt(np.linalg.matrix_rank(self.pci_reg) // self.degree)

        self.power_basis = self._get_power_basis(seed)
        self.power_basis_dual = qldpc.math.get_dual_basis(self.power_basis, validate=False)

        self.extended_field = galois.GF(self.field.order**self.degree)
        self.embedded_scalars, self.embedded_power_basis = self._get_center_embeddings()
        self.embedded_power_basis_dual = self._get_embedded_power_basis_dual()

        max_embedded_scalar = max(self.embedded_scalars.view(np.ndarray))
        self.embedded_scalars_inverse = self.field.Zeros(max_embedded_scalar + 1)
        for qq, pp in enumerate(self.embedded_scalars):
            self.embedded_scalars_inverse[int(pp)] = qq

        self.matrix_basis = self._get_matrix_basis(seed)
        self.decomposition_coefficient_extractor = self._get_decomposition_coefficient_extractor()
        self.decomposition_coefficient_recombiner = self._get_decomposition_coefficient_recombiner()

    def _get_center(self) -> galois.FieldArray:
        r"""Identify a basis for the center Z(S) of S.

        The center Z(S) is the subspace of "scalars" in S that commute with all elements of S:
            Z(S) = { z ∈ S : z·s = s·z for all s ∈ S }.

        We can decompose Z(S) = S ⋂ Z(R), where Z(R) is the center of R.  Letting L(r) and A(g)
        denote, respectively, the regular and adjoint representations of r ∈ R and g ∈ G, we then
        note that
            S ≅ ker(L(e) - 1)
        and
            Z(R) ≅ ⋂_{generators g of G} ker(A(g) - 1).
        We can therefore find a basis for Z(S) by intersecting the null space of L(e) - 1 with the
        null spaces of A(g) - 1 for the generators g of G.

        Returns:
            - A matrix in GF(q)^{d × |G|} whose rows form a basis for Z(S).
        """
        # identify the null space of L(e) - 1, which spans S
        center = self.pci_reg.column_space()  # equal to the null space of L(e) - 1

        if self.ring.is_commutative:
            # if R is commutative, then Z(S) = S, so we are done
            return center

        # intersect with the null spaces of A(g) - 1 for all generators g
        identity = self.field.Identity(self.ring.group.order)
        for generator in self.ring.group.generators:
            mat = self.ring.group.adjoint_lift(generator).view(self.field) - identity
            center = (mat @ center.T).null_space() @ center

        return center

    def _get_power_basis(self, seed: np.random.Generator) -> galois.FieldArray:
        r"""Construct a power basis for Z(S), used for the field extension GF(q) -> GF(q^d).

        Mathematically,
            GF(q^d) ≅ GF(q)[x] / f(x),
        where
        - GF(q)[x] is the set of univariate polynomials with coefficients in GF(q).
        - f(x) ∈ GF(q)[x] is an irreducible polynomial with degree d.  Here "irreducible"
            essentially means "prime": f(x) has no nontrivial factors of degree <= d.
        The monomial x is called the primitive element of GF(q)[x] / f(x), and its powers,
            (x^0, x^1, x^2, ..., x^{d-1}),
        form a "power basis" for GF(q)[x] / f(x).

        We need to find elements of Z(S) that act as GF(q^d) scalars when embedding S into
        GF(q^d)^{n × n}.  To this end, we identify a suitable generator b ∈ Z(S) that can serve as
        the primitive element of a field extension GF(q)[x] / f(x).  Crucially, the powers of this
        generator, collected into the power basis
            B = (b^0, b^1, b^2, ..., b^{d-1}),
        must be linearly independent.  Here b^0 = e is the identity element of S.

        To find a suitable power basis B, we...
            1. Pick a random element b ∈ Z(S).
            2. Collect powers of b into rows of the matrix B = (b^0, b^1, b^2, ..., b^{d-1}).
            3. Check whether the elements of B span GF(q)-linear vector space of dimension d.
                If so, return B.  Otherwise, go back to step 1.

        Returns:
            - A matrix in GF(q)^{d × |G|} whose j-th row is b^j ∈ Z(S).
        """
        if self.degree == 1:
            return self.pci_vec.reshape(1, -1).view(self.field)

        while True:
            generator_vec = self._random_nonzero_vec(len(self.center), seed) @ self.center
            generator = RingMember.from_vector(generator_vec, self.ring)
            generator_mat = generator.regular_lift()

            basis = [self.pci_vec, generator_vec]
            for _ in range(self.degree - 2):
                basis.append(generator_mat @ basis[-1])

            basis_in_field = self.field(basis)
            if np.linalg.matrix_rank(basis_in_field) == self.degree:
                return basis_in_field

    def _random_nonzero_vec(self, length: int, seed: np.random.Generator) -> galois.FieldArray:
        """Return a random nonzero vector over GF(q)."""
        while not np.any(vector := self.field.Random(len(self.center), seed=seed)):
            pass  # pragma: no cover
        return vector

    def _get_center_embeddings(self) -> tuple[galois.FieldArray, galois.FieldArray]:
        r"""Construct embeddings of elements in the center Z(S) into GF(p^{kd}) ≅ GF(q^d).

        There are two parts to this embedding:
        1. Embedding GF(q) scalars into GF(p^{kd}).
        2. Embedding the power basis B = (b^0, b^1, b^2, ..., b^{d-1}) for GF(q^d) into GF(p^{kd}).

        Returns:
            - A vector in GF(p^{kd})^d whose j-th element is the embedding of GF(q)(j).
            - A vector in GF(p^{kd})^d whose j-th element is the embedding of b^j.
        """
        if self.degree == 1:
            return self.field.elements, self.field.Ones([1])

        """
        PART 1
        ------
        To embed GF(q) scalars into GF(p^{kd}), we...
            1. Identify the generator α of GF(q) = GF(p^k).
            2. Find the minimal polynomial m(x) of α, which has α as a root in GF(p^k).
            3. Interpret the coefficients of m(x) as elements of GF(p^{kd}).
            4. Identify a root σ ∈ GF(p^{kd}) of m(x).
        A scalar in GF(q) = GF(p^k) = GF(p)[x] / m(x) can then be embedded into GF(p^{kd}) by...
            1. Expanding the scalar as a polynomial in α with coefficients in GF(p).
            2. Interpreting the coefficients as elements of GF(p^{kd}).
            3. Replacing α by σ.
        See:
        - https://mhostetter.github.io/galois/v0.1.1/api/galois.GF/
        - https://mhostetter.github.io/galois/v0.1.1/api/galois.FieldArray.minimal_poly/
        """
        minimal_poly = self.field.primitive_element.minimal_poly()  # this is m(x) ∈ GF(q)[x]
        extended_minimal_poly = galois.Poly(minimal_poly.coeffs, field=self.extended_field)
        embedded_root = extended_minimal_poly.roots()[0]  # this is σ ∈ GF(p^{kd})
        embedded_root_powers = [self.extended_field(1)]
        for _ in range(self.field.order - 1):
            embedded_root_powers.append(embedded_root_powers[-1] * embedded_root)
        embedded_scalars = []
        for scalar in self.field.elements:
            poly_coeffs = scalar.vector()[::-1]
            terms = [
                self.extended_field(coeff) * power
                for coeff, power in zip(poly_coeffs, embedded_root_powers)
            ]
            embedded_scalars.append(functools.reduce(operator.add, terms))

        """
        PART 2
        ------
        To embed elements of the power basis B into GF(p^{kd}) we...
            1. Identify the minimal polynomial f(x) of b, for which f(b) = 0.  This is the
                polynomial we use to construct GF(q^d) = GF(q)[x] / f(x) with f(b) = 0.
            2. Map the GF(q) coefficients of f(x) into GF(p^{kd}) to obtain the polynomial g(x).
            3. Use any root of g(x) as the generator of the embedded power basis.
        To find f(x), we seek coefficients f_j ∈ GF(q) for which
            f(b) = sum_{j=0}^d f_j b^j = 0,
        where we define b^0 = e.  We can set f_d = 1 without loss of generality, reducing the
        problem to
            sum_{j=0}^{d-1} f_j b^j = -b^d.
        The remaining coefficients can be found by solving a linear system of equations.
        """
        gen_to_dim_power = self._regular_lift(self.power_basis[1]) @ self.power_basis[-1]
        linear_system = np.column_stack([self.power_basis.T, -gen_to_dim_power])
        poly_coeffs = linear_system.view(self.field).row_reduce()[: self.degree, -1]
        poly_coeffs = np.append(poly_coeffs, self.field(1))
        irreducible_poly = galois.Poly(poly_coeffs[::-1], field=self.field)  # this is f(x)
        extended_irreducible_poly = galois.Poly(
            [embedded_scalars[cc] for cc in irreducible_poly.coeffs], field=self.extended_field
        )
        embedded_generator = extended_irreducible_poly.roots()[0]
        embedded_power_basis = [self.extended_field(1)]
        for _ in range(self.degree - 1):
            embedded_power_basis.append(embedded_power_basis[-1] * embedded_generator)

        return self.extended_field(embedded_scalars), self.extended_field(embedded_power_basis)

    def _regular_lift(self, vector: galois.FieldArray, *, right: bool = False) -> galois.FieldArray:
        """Lift a member of S from GF(q)^{|G|} to a matrix that encodes ring multiplication."""
        return RingMember.from_vector(vector, self.ring).regular_lift(right=right)

    def _get_embedded_power_basis_dual(self) -> galois.FieldArray:
        r"""Construct the dual of the embedded power basis E[B].

        Let E : GF(q)^{|G|} -> GF(p^{kd}) be the embedding of the center Z(S) into GF(p^{kd}).

        For an embedded power basis E[B] = (E[b^0], E[b^1], E[b^2], ..., E[b^{d-1}]) ∈ GF(p^{kd})^d,
        the dual basis
            E[A] = (E[a_0], E[a_1], E[a_2], ..., E[a_{d-1}])
        satisfies
            Tr_{GF(q^d)/GF(q)}[E[a_i] E[b^j]] = delta_{ij},
        where Tr_{GF(q^d)/GF(q)} denotes a field trace from GF(q^d) to GF(q); see self.field_trace.

        The dual basis allows us to "pick off" the coefficients of a polynomial in the center
            Z(S) ≅ GF(q^d) ≅ GF(q)[x] / f(x)
        that has been embedded into GF(p^{kd}), which is useful for mapping back into Z(S) by:
            1. Mapping z ∈ GF(p^{kd}) to (z_0, z_1, ..., z_{d-1}) ∈ GF(q)^d with z_j = Tr[E[a_j] z].
            2. Combining these coefficients to recover sum_j z_j b^j ∈ Z(S).

        Returns:
            - A vector in GF(p^{kd}) whose j-th entry is the embedding E[a_j].
        """
        if self.degree == 1:
            return self.extended_field.Ones([1])
        matrix = self.extended_field(
            [
                self.extended_field_trace(aa * bb)
                for aa, bb in itertools.product(self.embedded_power_basis, repeat=2)
            ]
        ).reshape([self.degree] * 2)
        return np.linalg.inv(matrix) @ self.embedded_power_basis

    def extended_field_trace(self, value: galois.FieldArray) -> galois.FieldArray:
        """Compute the field trace from GF(q^d) to GF(q).

        The field trace of z from GF(q^d) to GF(q) is defined by
            Tr_{GF(q^d)/GF(q)}[z] = sum_{i=0}^{d-1} z^{q^i}.

        See:
        - https://en.wikipedia.org/wiki/Field_trace
        """
        conjugates = [value ** (self.field.order**pow) for pow in range(self.degree)]
        values = self.extended_field(np.stack(conjugates, axis=0)).sum(axis=0)
        return values.reshape(value.shape).view(self.extended_field)

    def _get_matrix_basis(self, seed: np.random.Generator) -> galois.FieldArray:
        """Construct standard basis of matrix elements |i><j| ∈ S ≅ GF(q^d)^{n × n}.

        This method first decomposes the PCI e of S into primitive (possibly non-central)
        idempotents e_i that sum to the PCI: e = sum_i e_i.  The primitive idempotents e_i are the
        "diagonal" matrix elements |i><i|.  These idempotents are, in turn, used to construct
        off-diagonal matrix elements e_ij = |i><j| ∈ e_i S e_j.

        Returns:
            - A matrix in GF(q)^{n^2 × |G|} whose (ij, :) entry is |i><j| = e_ij ∈ S.
        """
        if self.ring.is_commutative:
            return self.pci_vec.reshape(1, -1).view(self.field)

        # collect primitive idempotents along the diagonal of the matrix_basis
        pids_as_vecs = self._get_primitive_idempotents(seed)
        basis_as_vecs = self.field.Zeros([self.size, self.size, self.ring.group.order])
        basis_as_vecs[np.arange(self.size), np.arange(self.size), :] = pids_as_vecs

        # construct matrices for left- and right-multiplication by the primitive idempotents
        pid_mats_l = [self._regular_lift(idempotent) for idempotent in pids_as_vecs]
        pid_mats_r = [self._regular_lift(idempotent, right=True) for idempotent in pids_as_vecs]

        # construct the off-diagonal matrix elements |0><i| and |i><0|
        for ii in range(1, self.size):
            # projections onto e_0 R e_i and e_i R e_0
            projection_0_i = pid_mats_l[0] @ pid_mats_r[ii]
            projection_i_0 = pid_mats_l[ii] @ pid_mats_r[0]

            # bases for e_0 R e_i and e_i R e_0 in GF(q)^{|G|}
            basis_0_i = projection_0_i.column_space()
            basis_i_0 = projection_i_0.column_space()

            # choose suitable elements of e_0 R e_i and e_i R e_0 as |0><i| and |i><0|
            basis_as_vecs[0, ii, :], basis_as_vecs[ii, 0, :] = self._get_off_diagonal_basis_vecs(
                basis_0_i, basis_i_0, seed
            )

        # build left-regular representations of |0><i| and |i><0|
        basis_as_mats = {}
        for ii in range(1, self.size):
            basis_as_mats[0, ii] = self._regular_lift(basis_as_vecs[0, ii, :])
            basis_as_mats[ii, 0] = self._regular_lift(basis_as_vecs[ii, 0, :])

        # construct the remaining matrix elements |i><j| = |i><0|·|0><j|
        for ii, jj in itertools.combinations(range(1, self.size), r=2):
            basis_as_vecs[ii, jj, :] = basis_as_mats[ii, 0] @ basis_as_vecs[0, jj, :]
            basis_as_vecs[jj, ii, :] = basis_as_mats[jj, 0] @ basis_as_vecs[0, ii, :]

        return basis_as_vecs.reshape(-1, basis_as_vecs.shape[-1])

    def _get_primitive_idempotents(
        self, seed: np.random.Generator, idempotent_vec: galois.FieldArray | None = None
    ) -> galois.FieldArray:
        """Decompose an idempotent of S into primitive idempotents.

        Recall that S ≅ GF(q^d)^{n × n}.  The PCI of S is the identity matrix of GF(q^d)^{n × n}.
        If n = 1 (or: when R is commutative), then the multiplicative identity in GF(q^d) is the
        only element of GF(q^d) that squares to itself, so the PCI of S is the only idempotent in S.
        If n > 1, however, then the PCI can be decomoposed into primitive (non-central) idempotents,
            e = sum_{i=1}^n e_i,
        where e_i is essentially the |i><i| matrix element of GF(q^d)^{n × n}.

        This method decomposes an idempotent of S into primitive idempotent recursively.  Given an
        idempotent e_start, this method proceeds as follows:
            1. Determine whether e_start is primitive.  If so, return {e_start}.
            2. Find two or more idempotents that sum to e_start.
            3. Decompose each of the idempotents found at step 2 by a recursive call to this method.
            4. Return the combined set of all idempotents found from decomposition at step 3.

        Returns:
            - A matrix in GF(q)^{n × |G|} whose j-th row the primitive idempotent e_i ∈ S.
        """
        assert not self.ring.is_commutative  # this method should not have been called
        if idempotent_vec is None:
            idempotent_vec = self.pci_vec

        """
        PART 1
        ------
        To determine whether the idempotent is primitive, we first identify the sub-algebra that it
        projects onto, or the image of the map r -> e_start r e_start.  If this image spans a
        d-dimensional vector space over GF(q), then it must be GF(q^d), which means that the
        idempotent must be primitive.
        """
        idempotent = RingMember.from_vector(idempotent_vec, self.ring)
        subalgebra_proj = idempotent.regular_lift() @ idempotent.regular_lift(right=True)
        subalgebra_basis = subalgebra_proj.column_space()
        if len(subalgebra_basis) == self.degree:  # pragma: no cover (we may not hit this in tests)
            return idempotent_vec.reshape(1, -1).view(self.field)

        """
        PART 2
        ------
        At this point, we know that the idempotent e_start is not primitive, so we seek to decompose
        e_start into a sum of two or more idempotents.  To this end, we proceed as follows:
            1. Pick a random element α in the sub-algebra stabilized by e_start.
            2. Find the minimal polynomial of α, or a minimal-degree polynomial m(x) with m(α) = 0.
            3. Identify the irreducible factors of m(x).  If m(x) is irreducible, return to step 1.
            4. Factor m(x) into irreducible ("prime") polynomials p_j(x) as
                    m(x) = prod_j p_j(x)^{k_j},
                and define
                    f_j(x) = p_j(x)^{k_j},
                    g_j(x) = m(x) / f_j(x).
        These polynomials will be used to construct idempotents that sum to e_start.
        """

        # find an element of the sub-algebra whose minimal polynomial has nontrivial factors
        while True:
            random_member = self.field.Random(len(subalgebra_basis), seed=seed) @ subalgebra_basis
            minimal_poly, powers = self._get_minimal_polynomial(random_member, idempotent_vec)
            factors, multiplicities = minimal_poly.factors()
            if len(factors) > 1:
                factors = [factor**mult for factor, mult in zip(factors, multiplicities)]
                break

        # above, we found factors f_j(x) of m(x); now, build the quotients g_j(x) = m(x) / f_j(x)
        quotients = [minimal_poly // factor for factor in factors]

        """
        PART 3
        ------
        We now use the minimal polynomial m(x) of α ∈ e_start R e_start, its factors f_j(x), and the
        quotients g_j(x) to construct idempotents that sum to e_start.  To this end, we use the
        extended Euclidean algorithm to find polynomials u_j and v_j for which
            u_j f_j + v_j g_j = gcd(f_j, g_j),
        and define
            F_j = u_j f_j / gcd(f_j, g_j),
            G_j = v_j g_j / gcd(f_j, g_j).
        Some observations:
            1. By construction, F_j + G_j = 1, where "1" is e_start in this sub-algebra of S.
            2. (F_j G_j)(x) contains a factor of (f_j g_j)(x) = m(x), so (F_j G_j)(α) = 0.
            3. G_j(α) = G_j(α) (F_j + G_j)(α) = (G_j F_j)(α) + G_j(α)^2 = G_j(α)^2.
            4. If i != j, then (G_i G_j)(x) contains a factor of m(x) so, (G_i G_j)(α) = 0.
        Observation 3 implies that G_j(α) is idempotent, and observation 4 implies that these
        idempotents are orthogonal, so {G_j(α)}_j is a set of orthogonal idempotents.
        """

        new_idempotents = []
        quotient_coeff: galois.Poly
        gcd: galois.Poly
        for factor, quotient in zip(factors, quotients):
            gcd, factor_coeff, quotient_coeff = galois.egcd(factor, quotient)  # type:ignore[assignment,arg-type]
            idempotent_poly = quotient_coeff * quotient // gcd  # <- G_j(x)
            new_idempotent = idempotent_poly.coeffs[::-1] @ powers[: len(idempotent_poly)]  # <- e_j
            new_idempotents.append(new_idempotent)

        # if sum_j G_j != e_start, then add the remainder to our set of new idempotents
        if np.any(remainder := functools.reduce(operator.add, new_idempotents) - idempotent_vec):
            new_idempotents.append(remainder)  # pragma: no cover

        if len(new_idempotents) == self.size:  # pragma: no cover (we may not hit this in tests)
            # the number of mutually orthogonal idempotents guarantees that they are primitive
            return self.field(new_idempotents)
        else:  # pragma: no cover (we may not hit this in tests)
            # recursively decompose the new idempotents to find primitive idempotents
            primitives = [self._get_primitive_idempotents(seed, id) for id in new_idempotents]
            return np.vstack(primitives).view(self.field)

    def _get_minimal_polynomial(
        self, element: galois.FieldArray, idempotent: galois.FieldArray
    ) -> tuple[galois.Poly, galois.FieldArray]:
        """Find the minimal polynomial an element within a sub-algebra stabilized by an idempotent.

        The minimal polynomial of α is the lowest-degree monic polynomial m(x) with m(α) = 0.
        Here "monic" means that the m(x) has a leading coefficient of one:
            m(x) = x^a + sum_{j=0}^{a-1} m_j x^j,
        where all coefficients m_j ∈ GF(q), and we define x^0 to be the idempotent.

        This method assumes--and does not verify---that:
            1. The provided idempotent is an idempotent of R.
            2. The provided element lives in the sub-algebra of R stabilized by the idempotent.

        Returns:
            - The minimal polynomial of RingMember(element, self.ring).
            - A matrix in GF(q)^{d × |G|} whose j-th row is element^j ∈ S.
        """

        """
        PART 1: Construct a matrix of linearly independent column vectors: [α^0, α^1, α^2, ...].

        We start with the one-column matrix [α^0], and repeatedly double the size of this matrix by
            [α^0] -> [α^0, α^1] -> [α^0, α^1, α^2, α^4] -> ...
        We stop when the number of columns exceeds the rank r of the matrix, at which point we save
        α^r for later use and throw out all but the first r columns, [α^0, α^1, α^2, ..., α^{r-1}].
        """
        element_mat = self._regular_lift(element)
        powers = idempotent.reshape(-1, 1).view(self.field)
        while True:
            new_powers = element_mat @ powers
            powers = np.column_stack([powers, new_powers]).view(self.field)
            rank = np.linalg.matrix_rank(powers)
            if powers.shape[1] > rank:
                break
            element_mat = element_mat @ element_mat
        extra = powers[:, rank]  # element^rank
        powers = powers[:, :rank]

        """
        PART 2: Construct the minimal polynomial m(x) of α.

        This polynomial is defined by coefficients m_j for which
            m(α) = α^r + sum_{j=0}^{r-1} m_j α^r = 0.
        These coefficients can be found by solving a linear system of equations.
        """
        linear_system = np.column_stack([powers, -extra]).view(self.field)
        poly_coeffs = linear_system.row_reduce()[:rank, -1]
        poly_coeffs = np.append(poly_coeffs, self.field(1))

        return galois.Poly(poly_coeffs[::-1], field=self.field), powers.T

    def _get_off_diagonal_basis_vecs(
        self, basis_ij: galois.FieldArray, basis_ji: galois.FieldArray, seed: np.random.Generator
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Construct standard-basis matrix elements |i><j| and |j><i| of S ≅ GF(q^d)^{n × n}.

        The strategy is as follows:
        1. Sample random vectors x_ij ∈ e_i R e_j and x_ji ∈ e_j R e_i.
        2. Construct y_i = x_ij x_ji = α e_i for some α ∈ GF(q^d).
        3. Compute z = n/|G| Tr_G[y_i] = α e, where Tr_G[y] = sum_{g in G} g y g^{-1}.
        3. Return e_ij = x_ij and e_ji = x_ji / α.

        A cautionary note: for representation-theoretic reasons, e_ji is the "dual" of e_ij in the
        sense that
            e_ij e_ji = e_i,
            e_ji e_ij = e_j,
        but it may not be the case that e_ji = e_ij.T in the sense of RingMember.T, because
        RingMember.T is an involution that inverts group members, which need not have anything to do
        with matrix transposition in GF(q^d)^{n × n}.

        Returns:
            - A vector in GF(q)^{|G|} representing e_ij ∈ S.
            - A vector in GF(q)^{|G|} representing e_ji ∈ S.
        """
        # sample x_ij ∈ e_i R e_j and x_ji ∈ e_j R e_i
        vec_ij = self._random_nonzero_vec(len(basis_ij), seed) @ basis_ij
        vec_ji = self._random_nonzero_vec(len(basis_ji), seed) @ basis_ji

        # construct y_i = α e_i, z = α e, and extract α as a scalar in GF(p^{kd})
        vec_i = self._regular_lift(vec_ij) @ vec_ji
        normalization = (self.field(1) * self.size) / (self.field(1) * self.ring.group.order)
        vec_z = self.ring.group_trace_matrix @ vec_i * normalization
        scalar = self._center_to_scalar(vec_z)

        # return e_ij = x_ij, e_ji = x_ji / α
        scalar_inv_as_mat = self._regular_lift(self._scalar_to_center(scalar ** (-1)))
        return vec_ij, scalar_inv_as_mat @ vec_ji

    def _center_to_scalar(self, vec_in_center: galois.FieldArray) -> galois.FieldArray:
        """Convert a "scalar' s ∈ Z(S) ≅ GF(q}^{|G|} into an element of GF(p^{kd}) ≅ GF(q^d)."""
        power_basis_coeffs = self.power_basis_dual @ vec_in_center
        embedded_power_basis_coeffs = self.embedded_scalars[power_basis_coeffs.view(np.ndarray)]
        return embedded_power_basis_coeffs @ self.embedded_power_basis

    def _scalar_to_center(self, scalar: galois.FieldArray) -> galois.FieldArray:
        """Embed a scalar in GF(p^{kd}) ≅ GF(q^d) back into the center Z(S) ≅ GF(q}^{|G|}."""
        embedded_coefficients = self.extended_field_trace(self.embedded_power_basis_dual * scalar)
        coefficients = self.embedded_scalars_inverse[embedded_coefficients.view(np.ndarray)]
        return coefficients @ self.power_basis

    def _get_decomposition_coefficient_extractor(self) -> galois.FieldArray:
        """Build a matrix that maps elements of S to their GF(q) coefficients in GF(q^d)^{n × n}.

        Consider a ring member r ∈ R ≅ GF(q)^{|G|} whose projection s = r·e ∈ S has the expansion
            s = sum_{i,j,k} s_ijk b^i e_jk ∈ GF(q)^{|G|},
        where each coefficient s_ijk ∈ GF(q).  This method constructs the linear map
            GF(q)^{|G|} -> GF(q)^{n × n × d}
        that takes a ring member r to the coefficients s_ijk.

        Returns:
            - A matrix in GF(q)^{n^2·d × |G|} that maps a ring member r to [s_ijk]_ijk.
        """
        # matrices representing the action of e_ij from multiplication on the left and right
        matrix_basis_as_mats_l = self.field(
            [self._regular_lift(vec) for vec in self.matrix_basis]
        ).reshape(self.size, self.size, self.ring.group.order, self.ring.group.order)
        matrix_basis_as_mats_r = self.field(
            [self._regular_lift(vec, right=True) for vec in self.matrix_basis]
        ).reshape(self.size, self.size, self.ring.group.order, self.ring.group.order)

        # construct a matrix that maps α e_i -> (α_0, α_1, ..., α_{d-1}), where α = sum_j α_j b^j
        normalization = (self.field(1) * self.size) / (self.field(1) * self.ring.group.order)
        get_diagonal_entry_scalar = (
            self.power_basis_dual @ self.ring.group_trace_matrix * normalization
        )

        # take r -> (e_j r e_k e_kj) = s_jk e_j -> s_ijk
        tensor = self.field(
            [
                get_diagonal_entry_scalar
                @ matrix_basis_as_mats_r[kk, jj]
                @ matrix_basis_as_mats_r[kk, kk]
                @ matrix_basis_as_mats_l[jj, jj]
                for jj, kk in itertools.product(range(self.size), repeat=2)
            ]
        )
        return tensor.reshape(self.size**2 * self.degree, self.ring.group.order).view(self.field)

    def _get_decomposition_coefficient_recombiner(self) -> galois.FieldArray:
        """Build a matrix that embeds GF(q) coefficients of GF(q^d)^{n × n} into S.

        The matrix built here is a left pseudo-inverse of that built in
        _get_decomposition_coefficient_extractor.

        Returns:
            - A matrix in GF(q)^{|G| × n^2·d} that maps [s_ijk]_ijk to s ∈ S ≅ GF(q)^{|G|}.
        """
        power_basis_mats = [self._regular_lift(bb) for bb in self.power_basis]
        return self.field(
            [
                power_basis_mats[ii] @ self.matrix_basis[jj_kk]
                for jj_kk in range(self.size**2)
                for ii in range(self.degree)
            ]
        ).T.view(self.field)

    def project(self, element: RingMember) -> galois.FieldArray:
        """Project an element of the parent ring into a simple component S ≅ GF(q^d)^{n × n}."""
        if element.ring is not self.ring:
            raise ValueError(
                "A Wedderburn-Artin transformer initialized for one ring was asked to decompose an"
                " element of a different ring"
            )
        coefficients = self.decomposition_coefficient_extractor @ element.to_vector()
        embedded_coefficients = self.embedded_scalars[coefficients.view(np.ndarray)]
        matrix_values = embedded_coefficients.reshape(-1, self.degree) @ self.embedded_power_basis
        return matrix_values.reshape(self.size, self.size)

    def project_array(self, array: RingArray, *, merge_blocks: bool = False) -> galois.FieldArray:
        """Project a RingArray element-wise into a simple component S ≅ GF(q^d)^{n × n}.

        An N-dimensional RingArray gets projected into an (N+2)-dimensional galois.FieldArray.

        If merge_blocks is True, this method treats each projected element as a block matrix in the
        last two axes of the provided array, such that a projection with shape
            (..., r, c, self.size, self.size)
            is transposed and reshaped into an array with shape
            (..., r * self.size, c * self.size).
        """
        if array.ring is not self.ring:
            raise ValueError(
                "A Wedderburn-Artin transformer initialized for one ring was asked to decompose an"
                " element of a different ring"
            )
        vectors = array.to_field_array().reshape(array.size, self.ring.group.order)
        coefficients = vectors @ self.decomposition_coefficient_extractor.T
        embedded_coefficients = self.embedded_scalars[coefficients.view(np.ndarray)]
        matrix_values = (
            embedded_coefficients.reshape(array.size * self.size**2, self.degree)
            @ self.embedded_power_basis
        )
        matrix_values = matrix_values.reshape(*array.shape, self.size, self.size)
        if merge_blocks:
            assert array.ndim >= 2
            repeat = array.size // (array.shape[-2] * array.shape[-1]) if array.size else 0
            old_shape = (repeat, array.shape[-2], array.shape[-1], self.size, self.size)
            new_shape = (
                *array.shape[:-2],
                array.shape[-2] * self.size,
                array.shape[-1] * self.size,
            )
            matrix_values = (
                matrix_values.reshape(old_shape).transpose(0, 1, 3, 2, 4).reshape(new_shape)
            )
        return matrix_values.view(self.extended_field)

    def embed(self, element: galois.FieldArray) -> RingMember:
        """Invert WedderburnArtinComponentTransformer.project."""
        if type(element) is not self.extended_field or element.shape != (self.size, self.size):
            raise ValueError(r"The provided element does not live in GF(q^d)^{n × n}")
        embedded_coefficients = self.extended_field_trace(
            np.outer(element.ravel(), self.embedded_power_basis_dual).view(self.extended_field)
        ).ravel()
        coefficients = self.embedded_scalars_inverse[embedded_coefficients.view(np.ndarray)]
        vector = self.decomposition_coefficient_recombiner @ coefficients
        return RingMember.from_vector(vector, self.ring)

    def embed_array(self, array: galois.FieldArray, *, from_blocks: bool = False) -> RingArray:
        """Invert WedderburnArtinComponentTransformer.project_array."""
        if from_blocks:
            block_rows, rem_rows = divmod(array.shape[-2], self.size)
            block_cols, rem_cols = divmod(array.shape[-1], self.size)
            assert rem_rows == rem_cols == 0
            repeat = array.size // (array.shape[-2] * array.shape[-1]) if array.size else 0
            old_shape = (repeat, block_rows, self.size, block_cols, self.size)
            new_shape = (*array.shape[:-2], block_rows, block_cols, self.size, self.size)
            array = (array.reshape(old_shape).transpose(0, 1, 3, 2, 4).reshape(new_shape)).view(
                type(array)
            )
        if type(array) is not self.extended_field or array.shape[-2:] != (self.size, self.size):
            raise ValueError(r"The provided array does not store matrices in GF(q^d)^{n × n}")
        embedded_coefficients = self.extended_field_trace(
            np.outer(array.ravel(), self.embedded_power_basis_dual).view(self.extended_field)
        ).reshape(*array.shape[:-2], self.size**2 * self.degree)
        coefficients = self.embedded_scalars_inverse[embedded_coefficients.view(np.ndarray)]
        new_array = coefficients @ self.decomposition_coefficient_recombiner.T
        return RingArray.from_field_array(
            new_array.reshape(*array.shape[:-2], self.ring.group.order), self.ring
        )


def _get_block_howell_form(matrix: galois.FieldArray) -> galois.FieldArray:
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
    return matrix.reshape(shape).transpose(0, 2, 1, 3).view(field)
