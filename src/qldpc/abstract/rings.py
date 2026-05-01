"""Module for abstract algebra: rings and ring-valued numpy arrays

!!! WARNINGS !!!

First and foremoest, this module does not promise to be performant.  If you need to do heavy
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
import operator
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any, Literal, Union

import galois
import numpy as np
import numpy.typing as npt
import sympy.abc
import sympy.core

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

    @property
    def generators(self) -> list[RingMember]:
        """Generators of this ring's base group."""
        return [RingMember(self, gen) for gen in self.group.generators]

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
        Two idempotents g, h are orthogonal if g * h = h * g = 0.

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
            symbols = sympy.symbols("x:z", commutative=self.group.is_abelian)[:num_gens]
        elif num_gens <= 26:
            symbols = sympy.symbols("a:z", commutative=self.group.is_abelian)[-num_gens:]
        else:  # pragma: no cover
            index_length = int(np.ceil(np.log10(num_gens + 1)))
            symbols = [
                sympy.Symbol(f"x_{index:0{index_length}}", commutative=self.group.is_abelian)
                for index in range(num_gens)
            ]

        if isinstance(self.group, AbelianGroup):
            # Abelian groups are an easy special case for building the polynomial
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

    def regular_lift(self) -> galois.FieldArray:
        """Block matrix obtained by a regular lift of each entry of this RingArray."""
        assert self.ndim == 1 or self.ndim == 2
        blocks = [[val.regular_lift() for val in row] for row in self.reshape(-1, self.shape[-1])]
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

    @classmethod
    def from_field_array(cls, array: npt.NDArray[np.int_], ring: GroupRing | Group) -> RingArray:
        """Construct a RingArray from an array of coefficients (in a finite field) for each entry.

        The input array should have shape (..., ring.group.order), such that if array.ndim == 3, for
        example, then array[a, b, :] is the vector of coefficients for the RingMember at the
        constructed ring_array[a, b].
        """
        if isinstance(array, (GroupRing, Group)):
            warnings.warn(
                "Check argument order: it should be RingArray.from_field_array(array, ring)."
                "  The order (ring, array) is DEPRECATED and will throw an error in the future!",
                DeprecationWarning,
                stacklevel=2,
            )
            array, ring = ring, array
        group = ring.group if isinstance(ring, GroupRing) else ring
        assert array.shape[-1] == group.order
        vals = [RingMember.from_vector(entry, ring) for entry in array.reshape(-1, group.order)]
        return RingArray(np.array(vals, dtype=object).reshape(array.shape[:-1]), ring=ring)

    def to_field_array(self) -> galois.FieldArray:
        """Convert a RingArray into an array of coefficients (in a finite field) for each entry.

        This method is the inverse of RingArray.from_field_array.
        """
        vals = [val.to_vector() for val in self.ravel()]
        return np.asarray(vals).reshape(self.shape + (self.group.order,)).view(self.field)

    @classmethod
    def from_field_vector(cls, vector: npt.NDArray[np.int_], ring: GroupRing | Group) -> RingArray:
        """Construct a 1-D RingArray from a vector of coefficients."""
        if isinstance(vector, (GroupRing, Group)):
            warnings.warn(
                "Check argument order: it should be RingArray.from_field_vector(vector, ring)."
                "  The order (ring, vector) is DEPRECATED and will throw an error in the future!",
                DeprecationWarning,
                stacklevel=2,
            )
            vector, ring = ring, vector
        assert vector.ndim == 1
        group = ring.group if isinstance(ring, GroupRing) else ring
        assert vector.size % group.order == 0
        return RingArray.from_field_array(vector.reshape(-1, group.order), ring)

    def to_field_vector(self) -> galois.FieldArray:
        """Convert a 1-D RingArray into a vector of coefficients."""
        assert self.ndim == 1
        vals = [val.to_vector() for val in self.ravel()]
        return np.asarray(vals).ravel().view(self.field)

    def null_space(self) -> RingArray:
        """Construct a matrix of null-space row vectors for this RingArray.

        The transpose of the null-space matrix is annihilated by this RingArray, such that
        np.any(self @ self.null_space().T) is np.False_.

        Unlike galois.FieldArray.row_reduce, this method does not perform any row reduction on the
        matrix of null-space row vectors.
        """
        assert self.ndim == 2

        # field-valued null vectors of matrix.regular_lift() correspond to ring-valued null vectors
        # of the matrix via conversion with RingArray.from_field_vector <> RingArray.to_field_vector
        null_field_vectors = self.regular_lift().null_space()

        # collect ring-valued null row vectors (that is, transposed null column vectors)
        field_array_shape = (len(null_field_vectors), self.shape[1], self.group.order)
        return ~RingArray.from_field_array(null_field_vectors.reshape(field_array_shape), self.ring)

    def row_reduce(self) -> RingArray:
        """Compute a generalized reduced row echelon form of a RingArray over a semisimple ring.

        This method relies on the Wedderburn-Artin decomposition:
        1. Decompose the matrix over a ring into matrices over simple components.
        2. Put the matrices over simple components into RREF.
        3. Re-combine the simple components into a matrix over the original ring.
        """
        assert self.ndim == 2
        if not self.ring.is_semisimple:
            raise ValueError("RingArray.row_reduce only supports semisimple rings")
        transformer = WedderburnArtinTransformer(self.ring)
        matrices = [component.row_reduce() for component in transformer.decompose_array(self)]
        return transformer.recompose_arrays(matrices)

    def howell_normal_form(self, *, poly: bool = False) -> RingArray:
        """Compute a Howell normal form of this RingArray.

        By default (if poly is False), this method first puts a RingArray into a generalized
        reduced row echelon form (see RingArray.row_reduce), then further post-processes the rows to
        satisfy the Howell property.  Specifically, if a row r has a pivot p with a nontrivial
        annihilator α (meaning α != 0 and α·p = 0), then the row r is replaced by (1-α)·r, and the
        row α·r is appended to the matrix.  This procedure requires the ring to be semisimple.

        If poly is True, then the base ring must be a cyclic group algebra.  In this case, this
        method interprets the base ring as a univariate polynomial ring, and computes a Howell
        normal form using a notion of row reduction that induced by polynomial division.

        References:
        - https://en.wikipedia.org/wiki/Howell_normal_form
        - https://github.com/m-webster/XPFpackage/blob/570ea89/Examples/A.1_howell_matrix.ipynb
        """
        assert self.ndim == 2
        if poly:
            return self._howell_normal_form_poly()
        if not self.ring.is_semisimple:
            raise ValueError(
                "The ordinary Howell normal form requires the base ring to be semisimple"
            )
        if self.group.is_abelian:
            return self._howell_normal_form_abelian()
        return self._howell_normal_form_non_abelian()

    def _howell_normal_form_abelian(self) -> RingArray:
        """Compute the Howell normal form of a RingArray over a semisimple Abelian ring."""
        assert self.ndim == 2 and self.ring.is_semisimple and self.group.is_abelian

        # identify the components of the reduced row echelon form of this RingArray
        transformer = WedderburnArtinTransformer(self.ring)
        matrices = [matrix.row_reduce() for matrix in transformer.decompose_array(self)]

        def _remove_zero_rows(matrices: list[galois.FieldArray]) -> list[galois.FieldArray]:
            """Remove rows that are zero in all components."""
            nonzero_rows = functools.reduce(
                np.bitwise_or, [np.any(matrix, axis=1) for matrix in matrices]
            )
            return [matrix[nonzero_rows] for matrix in matrices]

        matrices = _remove_zero_rows(matrices)

        pivot_row = 0
        pivot_col = 0
        num_rows, num_cols = self.shape
        while pivot_row < num_rows and pivot_col < num_cols - 1:
            """
            Identify:
            1. The column of the first nonzero value in the pivot_row of each component.
            2. The column that will contain the pivot when we recombine the components.
            """
            pivot_cols = [
                num_cols
                if not np.any(matrix[pivot_row])
                else int(np.argmax(matrix[pivot_row].view(np.ndarray).astype(bool)))
                for matrix in matrices
            ]
            pivot_col = min(pivot_cols)

            """
            Let π be a projector onto the components in which the pivot is nonzero.  If π != 1, then
            (1-π) is a nontrivial annihilator of the pivot.  In this case, in principle we need to
            replace the pivot row r -> π·r, and add (1-π)·r as a new row to the matrix.  In
            practice, this procedure messes up the reduced row echelon form of the matrix, so we
            instead...
            1. In the (1-π) sector, insert a zero row at the pivot_row and shift down rows below.
            2. In the π sector, append a zero row to the matrix.
            """
            annihilating_components = [
                cc for cc in range(len(matrices)) if pivot_col < pivot_cols[cc] < num_cols
            ]
            if annihilating_components:
                for cc, matrix in enumerate(matrices):
                    field = type(matrix)
                    if cc in annihilating_components:
                        stack = [matrix[:pivot_row], field.Zeros(num_cols), matrix[pivot_row:]]
                    else:
                        stack = [matrix, field.Zeros(num_cols)]
                    matrices[cc] = np.vstack(stack).view(field)
                num_rows += 1

            pivot_row += 1

        matrices = _remove_zero_rows(matrices)
        return transformer.recompose_arrays(matrices)

    def _howell_normal_form_non_abelian(self) -> RingArray:
        """Compute a Howell normal form of a RingArray over a semisimple non-Abelian ring."""
        assert self.ndim == 2 and self.ring.is_semisimple
        raise NotImplementedError(
            "RingArray.howell_normal_form does not yet support non-Abelian rings"
        )

    def _howell_normal_form_poly(self) -> RingArray:
        """Compute a Howell normal form of a RingArray using polynomial division.

        If the base ring of a RingArray is a cyclic group algebra, then it can be interpreted as a
        univariate polynomial ring, allowing us to compute greatest common divisors and perform row
        reduction with polynomial division.

        References:
        - https://en.wikipedia.org/wiki/Howell_normal_form
        - https://github.com/m-webster/XPFpackage/blob/570ea89/Examples/A.1_howell_matrix.ipynb
        """
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
    (Abelian) ring, then all n_i = 1, so
        R = ⨂_i D_i  (if R is Abelian).

    An instance of this class is a container for transformers that project elements of R onto simple
    components R_i, and embed elements of R_i back into R.
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
        """Decompose an element of the ring into its Wedderburn-Artin components."""
        return [transformer.project(element) for transformer in self.transformers]

    def decompose_array(self, array: RingArray) -> list[galois.FieldArray]:
        """Decompose an array over a ring into its Wedderburn-Artin components."""
        decomposed_arrays = []
        for transformer in self.transformers:
            values = transformer.extended_field([transformer.project(val) for val in array.ravel()])
            decomposed_arrays.append(values.reshape(array.shape).view(transformer.extended_field))
        return decomposed_arrays

    def recompose(self, components: Sequence[galois.FieldArray]) -> RingMember:
        """Invert WedderburnArtinTransformer.decompose."""
        if len(components) != len(self.transformers):
            raise ValueError(
                "Incorrect number of components provided to WedderburnArtinTransformer.recompose"
            )
        terms = [
            transformer.embed(component)
            for component, transformer in zip(components, self.transformers)
        ]
        return functools.reduce(operator.add, terms)

    def recompose_arrays(self, arrays: Sequence[galois.FieldArray]) -> RingArray:
        """Invert WedderburnArtinTransformer.decompose_array."""
        shapes = [array.shape for array in arrays]
        assert len(set(shapes)) == 1
        shape = shapes[0]
        values = [self.recompose([array[idx] for array in arrays]) for idx in np.ndindex(shape)]
        return RingArray(values, ring=self.ring).reshape(shape).view(RingArray)


@dataclasses.dataclass
class WedderburnArtinComponentTransformer:
    """Transformer to map between a semisimple ring R and a simple component R_i.

    Let R = F[G] be a group algebra, whose elements can be written in the form
        r = sum_{g in G} r_g g,
    where each r_g is an element of F.  Each primitive central idempotent (PCI) of R acts as a
    projector onto a simple component R_i that is isomorphic to set of matrices over a field
    extension of F.  That is,
        R ≅ ⨂_i R_i,
    where
        R_i = D_i^{n_i × n_i},
    with each D_i a field extension of F.  If R is Abelian, then all n_i = 1, so R_i ≅ D_i.

    This class is an instrument for projecting elements of R onto a simple component R_i, and
    embedding elements of R_i back into R.
    """

    ring: GroupRing  # base ring, R
    pci: RingMember  # primitive central idempotent (PCI) that projects onto this component of R
    lifted_pci: galois.FieldArray  # PCI lifted to a matrix over GF(q)

    field: type[galois.FieldArray]  # base field of the ring, F = GF(q) = GF(p^k)
    dimension: int  # dimension d of the field extension GF(q^d) for this component of R
    extended_field: type[galois.FieldArray]  # field extension GF(p^(kd)) ≅ GF(q^d)

    basis_in_ring: RingArray  # basis B for this component of R, represented by elements of R
    basis_in_field: galois.FieldArray  # same basis, lifted to field vectors over GF(q)

    embedded_scalars: galois.FieldArray  # embedding of GF(q) into GF(p^(kd))
    embedded_basis: galois.FieldArray  # embedding of B into GF(p^(kd))
    dual_basis: galois.FieldArray  # dual basis of the embedded_basis in GF(p^(kd))

    def __init__(self, pci: RingMember, *, seed: np.random.Generator | int | None = None) -> None:
        """Initialize from a primitive central idempotent (PCI) of a ring.

        WARNING: This class assumes that the provided RingMember is indeed a PCI of its parent ring.
        """
        self.pci = pci
        self.ring = pci.ring

        if not self.ring.is_semisimple:
            raise ValueError("The Wedderburn-Artin decomposition only exists for semisimple rings")
        if not self.ring.is_abelian:
            raise NotImplementedError(
                "WedderburnArtinTransformer does not yet support non-Abelian rings"
            )

        self.field = self.ring.field
        self.lifted_pci = pci.regular_lift()
        self.dimension = np.linalg.matrix_rank(self.lifted_pci)
        self.basis_in_ring, self.basis_in_field = self._get_basis_in_ring_and_field(seed)

        self.embedded_scalars, self.embedded_basis = self._get_embeddings()
        self.extended_field = type(self.embedded_scalars)
        self.dual_basis = self._get_dual_basis()

    def _get_basis_in_ring_and_field(
        self, seed: np.random.Generator | int | None
    ) -> tuple[RingArray, galois.FieldArray]:
        r"""Construct a basis for a simple component C of a semisimple ring R.

        Let e denote the primitive central idempotent (PCI) that projects onto C, and d = rank(e).

        If R is a finite Abelian group algebra, then C is isomorphic to a field extension of the
        base field GF(q) of R.  Mathematically,
            C ≅ GF(q^d) ≅ GF(q)[x] / f(x),
        where
        - GF(q)[x] denotes the set of univariate polynomials with coefficients in GF(q), and
        - f(x) ∈ GF(q)[x] is any irreducible polynomial with degree d.

        Below, we construct a power basis for the space of polynomials over GF(q) with degree < d,
            B = (b^0, b^1, b^2, ..., b^{d-1}),
        where
        - b ∈ C is called the generator of this power basis, and
        - b^0 is defined to be the multiplicative identity in C; that is, b^0 = e.

        The main requirement for B is that its elements are linearly independent over GF(q).
        To find a suitable basis, we...
        1. Pick a random element r of R.
        2. Multiply this element by e to project onto C.
        3. Check whether powers of the (r * e) span a GF(q)-linear vector space of sufficient
            dimension (namely, d).  If so, we set b = r * e.  Otherwise, we go back to step 1.
        """
        if self.dimension == 1:
            return RingArray([self.pci]), self.lifted_pci.reshape(1, -1).view(self.field)

        group_generators = self.ring.group.generators
        while True:
            coeffs = self.field.Random(len(group_generators), seed=seed)
            terms = [(coeff, gen) for coeff, gen in zip(coeffs, group_generators)]
            random_member = RingMember(self.ring, *terms)
            lifted_generator = self.lifted_pci @ random_member.regular_lift()
            vectorized_powers = [
                np.linalg.matrix_power(lifted_generator, power).ravel()
                for power in range(2, self.dimension)
            ]
            basis_in_field = np.vstack(
                [self.lifted_pci.ravel(), lifted_generator.ravel()] + vectorized_powers
            )
            if np.linalg.matrix_rank(basis_in_field) == self.dimension:
                break

        basis_in_ring = [self.pci]
        for _ in range(self.dimension - 1):
            basis_in_ring.append(basis_in_ring[-1] * random_member)

        return RingArray(basis_in_ring), basis_in_field.view(self.field)

    def _get_embeddings(self) -> tuple[galois.FieldArray, galois.FieldArray]:
        r"""Construct embeddings of elements of C into the extended field GF(q^d) ≅ GF(p^{kd}).

        If R is a finite Abelian group algebra, then C is isomorphic to a field extension of the
        base field GF(q) of R.  Mathematically,
            C ≅ GF(q^d) ≅ GF(q)[x] / f(x),
        where
        - GF(q)[x] denotes the set of univariate polynomials with coefficients in GF(q), and
        - f(x) ∈ GF(q)[x] is any irreducible polynomial with degree d.

        We previously found a power basis B = (b^0, b^1, b^2, ..., b^{d-1}) that spans a
        GF(q)-linear space of polynomials in some generator b ∈ C.  This basis allows us to write
        any element r ∈ C as a polynomial in GF(q)[b],
            r = sum_{j=0}^d r_j b^j,
        where all r_j ∈ GF(q).

        We now seek to embed elements of C into GF(p^{kd}).  There are four parts to this embedding:
        1. Constructing an extension GF(q^d) = GF(q)[x] / f(x) whose primitive element is b.
        2. Constructing an extension GF(p^{kd}) = GF(p)[x] / g(x) that naturally contains b.
        3. Embedding GF(q) scalars into GF(p^{kd}).
        4. Embedding elements of the power basis B into GF(p^{kd}).

        Returns:
            - A 1-dimensional galois.FieldArray whose j-th element is the embedding of GF(q)(j).
            - A 1-dimensional galois.FieldArray whose j-th element is the embedding of b^j.
        """
        if self.dimension == 1:
            return self.field.elements, self.field.Ones([1])

        """
        PART 1
        ------
        To construct an extension GF(q^d) = GF(q)[x] / f(x) whose primitive element is b, we need
        to construct an irreducible polynomial f(x) of degree d that has b as a root: f(b) = 0.
        That is, we seek coefficients c_j ∈ GF(q) for which
            f(b) = sum_{j=0}^d c_j b^j = 0.
        We can set c_d = 1 withous loss of generality, reducing the problem to
            sum_{j=0}^{d-1} c_j b^j = -b^d.
        The remaining coefficients can be found by solving a linear system of equations.
        """
        lifted_gen = self.basis_in_field[1].reshape([self.ring.group.order] * 2)
        gen_to_dim_power = np.linalg.matrix_power(lifted_gen, self.dimension)
        linear_system = np.vstack([self.basis_in_field, -gen_to_dim_power.reshape(1, -1)]).T
        poly_coeffs = linear_system.view(self.field).row_reduce()[: self.dimension, -1]
        poly_coeffs = np.append(poly_coeffs, self.field(1))
        irreducible_poly = galois.Poly(poly_coeffs[::-1], field=self.field)

        """
        PART 2
        ------
        To construct an extension GF(p^{kd}) = GF(p)[x] / g(x) that naturally contains b, we need
        to construct an irreducible polynomial g(x) of degree kd that has b as a root: g(b) = 0.
        We can build such a polynomial out of f(x) by multiplying all elements in the orbit of f(x)
        under the Frobenius map z -> z^p:
            g(x) = prod_{j=0}^{d-1} = f(x)^{p^j}.
        Due to the properties of the Frobenius map, f(x)^{p^j} is just f(x) with its coefficients
        raised to the power p^j.
        See:
        - https://en.wikipedia.org/wiki/Field_norm
        - https://en.wikipedia.org/wiki/Frobenius_endomorphism
        """
        orbit = [
            galois.Poly(poly_coeffs[::-1] ** (self.field.characteristic**power), field=self.field)
            for power in range(self.field.degree)
        ]
        product_of_orbit = functools.reduce(operator.mul, orbit)
        extension_irreducible_poly = galois.Poly(
            product_of_orbit.coeffs, galois.GF(self.field.characteristic)
        )
        extended_field = galois.GF(
            self.field.order**self.dimension, irreducible_poly=extension_irreducible_poly
        )

        """
        PART 3
        ------
        To embed GF(q) scalars into GF(p^{kd}), we...
            1. Identify the generator α of GF(q) = GF(p^k).
            2. Find the minimal polynomial m(x) of α, which has α as a root in GF(p^k).
            3. Interpret the coefficients of m(x) as elements of GF(p^{kd}).
            4. Identify a root σ ∈ GF(p^{kd}) of m(x).
        A scalar in GF(q) = GF(p^k) = GF(p) / m(x) can then be embedded into GF(p^{kd}) by...
            1. Expanding the scalar as a polynomial in α with coefficients in GF(p).
            2. Interpreting the coefficients as elements of GF(p^{kd}).
            3. Replacing α by σ.
        See:
        - https://mhostetter.github.io/galois/v0.1.1/api/galois.GF/
        - https://mhostetter.github.io/galois/v0.1.1/api/galois.FieldArray.minimal_poly/
        """
        minimal_poly = self.field.primitive_element.minimal_poly()
        extended_minimal_poly = galois.Poly(minimal_poly.coeffs, field=extended_field)
        embedded_root = extended_minimal_poly.roots()[0]
        embedded_root_powers = [extended_field(1)]
        for _ in range(self.field.order - 1):
            embedded_root_powers.append(embedded_root_powers[-1] * embedded_root)
        embedded_scalars = []
        for scalar in self.field.elements:
            poly_coeffs = scalar.vector()[::-1]
            terms = [
                extended_field(coeff) * power
                for coeff, power in zip(poly_coeffs, embedded_root_powers)
            ]
            embedded_scalars.append(functools.reduce(operator.add, terms))

        """
        PART 4
        ------
        Finally, to embed elements of the power basis B into GF(p^{kd}) we...
            1. Map the GF(q) coefficients of f(x) into GF(p^{kd}) to obtain the polynomial h(x).
            2. Use any root of h(x) as the generator of the embedded power basis.
        """
        extended_irreducible_poly = galois.Poly(
            [embedded_scalars[cc] for cc in irreducible_poly.coeffs], field=extended_field
        )
        embedded_generator = extended_irreducible_poly.roots()[0]
        embedded_basis = [extended_field(1)]
        for _ in range(self.dimension - 1):
            embedded_basis.append(embedded_basis[-1] * embedded_generator)

        return extended_field(embedded_scalars), extended_field(embedded_basis)

    def _get_dual_basis(self) -> galois.FieldArray:
        r"""Construct the dual of the power basis for GF(q^d) ≅ GF(p^{kd}).

        For the power basis B = (b^0, b^1, b^2, ..., b^{d-1}), the dual basis
            A = (a^0, a^1, a^2, ..., a^{d-1})
        satisfies
            Tr_{GF(q^d)/GF(q)}[a_i b^i] = delta_{ij},
        where Tr_{GF(q^d)/GF(q)} denotes a field trace from GF(q^d) to GF(q); see self._field_trace.

        The dual basis allows us to "pick off" the coefficients of a polynomial in GF(q)[x] / f(x),
        which is useful for embedding elements of GF(p^{kd}) ≅ GF(q^d) back into the ring R by:
        1. Mapping an element z ∈ GF(p^{kd}) to the vector (z_0, z_1, ...) with z_j = Tr[a_j z].
        2. Constructing the ring member r_z = sum_j z_j b^j ∈ R,
        Mechanically, this embedding procedure requires the dual basis to "live" in GF(p^{kd}).
        """
        if self.dimension == 1:
            return self.extended_field.Ones([1])
        matrix = self.extended_field(
            [
                self._field_trace(aa * bb)
                for aa, bb in itertools.product(self.embedded_basis, repeat=2)
            ]
        ).reshape([self.dimension] * 2)
        return np.linalg.inv(matrix) @ self.embedded_basis

    def _field_trace(self, value: galois.FieldArray) -> galois.FieldArray:
        """Compute the field trace from the extended field R_i into the base field of R.

        The field trace of z from GF(q^d) to GF(q) is defined by
            Tr_{GF(q^d)/GF(q)}[z] = sum_{i=0}^{d-1} z^{q^i}.

        See:
        - https://en.wikipedia.org/wiki/Field_trace
        """
        conjugates = [value ** (self.field.order**pow) for pow in range(self.dimension)]
        return functools.reduce(operator.add, conjugates)

    def project(self, element: RingMember) -> galois.FieldArray:
        """Project an element of the ring R ≅ ⨂_i R_i onto a component R_i."""
        if element.ring is not self.ring:
            raise ValueError(
                "A Wedderburn-Artin transformer initialized for one ring was asked to decompose an"
                " element of a different ring"
            )
        projection = self.lifted_pci @ element.regular_lift()
        linear_system = np.vstack([self.basis_in_field, projection.reshape(1, -1)]).T
        coeffs = linear_system.view(self.ring.field).row_reduce()[: self.dimension, -1]
        terms = [
            self.embedded_scalars[coeffs[ss]] * self.embedded_basis[ss]
            for ss in range(self.dimension)
        ]
        return functools.reduce(operator.add, terms)

    def embed(self, element: galois.FieldArray) -> RingMember:
        """Embed an element of a simple component R_i of R back into the ring R."""
        if type(element) is not self.extended_field:
            raise ValueError("Invalid field for an element of a simple component of a ring")
        if self.dimension == 1:
            return self.basis_in_ring[0] * element
        coefficients = [
            np.argmax(self.embedded_scalars == self._field_trace(dual * element))
            for dual in self.dual_basis
        ]
        terms = [vec * coeff for vec, coeff in zip(self.basis_in_ring, coefficients)]
        return functools.reduce(operator.add, terms)
