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
from collections.abc import Callable, Hashable, Iterable, Iterator
from typing import TYPE_CHECKING, Any

import galois
import numpy as np
import numpy.typing as npt
import scipy.sparse
import scipy.sparse.csgraph
import sympy
import sympy.combinatorics
import sympy.core

from qldpc import _deprecations
from qldpc.math import resolve_field

from .groups import AbelianGroup, Group, GroupMember
from .monomials import _is_integer, _read_polynomial, iter_monomial_terms

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

    The algebra of a noncommutative group has these anti-representations only if the group uses its
    default lift.  Otherwise, lifts with ``right=True`` raise a ValueError (see ``GroupRing.lift``).
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

        Integers are read as described in ``help(RingMember)``.
        """
        if not 0 <= value < self.field.order:
            if self.field.degree == 1:
                # there is no ambiguity over prime number fields
                return self.field(value % self.field.order)
            elif -self.field.order < value < 0:
                # negation corresponds to an additive inverse
                return -self.field(-value)
            else:
                raise ValueError(
                    f"The value of the integer {value} is ambiguous over GF({self.field.order})"
                )
        return self.field(value)

    def _read_coefficients(self, values: npt.ArrayLike) -> galois.FieldArray:
        """Read an array of coefficients as an array over the base field of this ring.

        The coefficients are read as described in ``RingMember.from_vector``, also when field
        elements are given in a nested sequence, as in a list of field arrays.  Raise a ValueError
        for other coefficients.
        """
        if isinstance(values, galois.FieldArray):
            return self._read_field_elements(values)

        # integer arrays are read directly
        if isinstance(values, np.ndarray) and values.dtype.kind in "iu":
            return self.field(self._check_integers(values))

        # read other values one by one, keeping the field of each field element, and reading the
        # rest as Python integers, which large fields can need
        def read(value: object) -> int:
            if isinstance(value, galois.FieldArray):
                return int(self._read_field_elements(value))
            try:
                return _read_integer(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Coefficients must be integers or field elements, not {value!r}"
                ) from error

        integers = np.vectorize(read, otypes=[object])(_as_object_array(values))
        return self.field(self._check_integers(integers))

    def _check_integers(self, integers: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Raise a ValueError unless integers are the integer representations of field elements."""
        if integers.size and not bool(np.all((integers >= 0) & (integers < self.field.order))):
            raise ValueError(f"Integer coefficients must be in [0, {self.field.order})")
        return integers

    def _read_field_elements(
        self, values: galois.FieldArray, description: str = "Coefficients"
    ) -> galois.FieldArray:
        """Read an array over a finite field as an array over the base field of this ring.

        Raise a ValueError unless the integer representations of the values denote the same elements
        in the base field of this ring.  The error message calls the values ``description``.
        """
        field = type(values)
        integers = values.view(np.ndarray)
        same_encoding = field.characteristic == self.field.characteristic and (
            field.degree == 1
            or field.irreducible_poly == self.field.irreducible_poly
            or bool(np.all(integers < field.characteristic))
        )
        if not same_encoding:
            raise ValueError(
                f"{description} in {_describe_field(field)} cannot be read in"
                f" {_describe_field(self.field)}"
            )
        return values if field is self.field else self.field(integers)


def _read_integer(value: Any) -> int:
    """Read a number that has an integer value, such as a boolean or a float, as an integer.

    Raise a ValueError for a float without an integer value, and a TypeError for other values that
    are not integers.
    """
    if isinstance(value, (float, np.floating)):
        if not float(value).is_integer():
            raise ValueError(f"{value} is not an integer")
        return int(value)
    if isinstance(value, np.bool_):
        return int(value)
    return operator.index(value)


def _as_object_array(values: object) -> npt.NDArray[np.object_]:
    """An array of the values in a nested sequence, which keeps field elements as field elements.

    Numpy reads field arrays in a sequence as integers, which loses their fields.
    """
    if isinstance(values, galois.FieldArray):
        array = np.empty(values.shape, dtype=object)
        for index in np.ndindex(values.shape):
            array[index] = values[index]
        return array
    if isinstance(values, (list, tuple)):
        if not values:
            return np.empty(0, dtype=object)
        return np.stack([_as_object_array(value) for value in values])
    return np.asarray(values, dtype=object)


def _describe_field(field: type[galois.FieldArray]) -> str:
    """Name a field class, with its irreducible polynomial if it is an extension field."""
    if field.degree == 1:
        return f"GF({field.order})"
    return f"GF({field.order}) with irreducible polynomial {field.irreducible_poly}"


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

        Without a ``seed``, this method reuses a decomposition of this ring that it already built,
        or else builds one from the seed 0, so that the decomposition is the same in every run.

        By default each component's primitive central idempotent is validated; pass
        ``skip_validation=True`` to skip the check (see WedderburnArtinComponentTransformer).
        """
        from .wedderburn_artin import WedderburnArtinTransformer  # avoid circular import

        # for seedless calls, reuse an existing decomposition, or else build a reproducible one
        if seed is None:
            if self._transformers:
                return next(iter(self._transformers.values()))
            seed = 0
        if seed not in self._transformers:
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
        """Construct the matrix for a trace over the group: ``r -> sum_{g in G} g r g^{-1}``.

        The matrix is read-only.
        """
        adjoints = [
            self._lift_in_field(self.group.adjoint_lift(gg)) for gg in self.group.generate()
        ]
        # copy, since for a group of order 1 the sum is the cached adjoint lift of the identity
        matrix = functools.reduce(operator.add, adjoints).copy()
        matrix.flags.writeable = False
        return matrix

    @property
    def generators(self) -> list[RingMember]:
        """The generators of this ring's base group, as ring members."""
        return [RingMember._from_trusted_terms(self, [(1, gen)]) for gen in self.group.generators]

    def regular_lift(self, member: GroupMember, *, right: bool = False) -> galois.FieldArray:
        """Lift a group member to its regular representation.

        See ``help(qldpc.abstract.Group.regular_lift)`` for more information.
        """
        return self._lift_in_field(self.group.regular_lift(member, right=right))

    def lift(self, member: GroupMember, *, right: bool = False) -> galois.FieldArray:
        """Lift a group member to a representation by a matrix.

        A representation satisfies

            ``self.lift(g·h) = self.lift(g) @ self.lift(h)``.

        If ``right=True``, lift to an anti-representation, for which

            ``self.lift(g·h) = self.lift(h) @ self.lift(g)``.

        Raise a ValueError if ``right=True`` for a noncommutative group with a custom lift, or if
        the group lifts its members to matrices over a finite field whose elements cannot be read in
        the base field of this ring (see ``RingMember.from_vector``).
        """
        return self._lift_in_field(self.group.lift(member, right=right))

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
        - are not sums of two nonzero orthogonal central idempotents.

        Two idempotents g, h are orthogonal if ``g * h = h * g = 0``.

        Intuitively, primitive central idempotents act like projectors onto orthogonal simple
        components of a ring.

        See `Idempotent (ring theory)
        <https://en.wikipedia.org/wiki/Idempotent_%28ring_theory%29>`_.

        This method looks up the idempotents in a table of precomputed ones, or else computes them.
        The cost of this computation grows with the order of the group, and most steeply with the
        number of simple components.  It also grows linearly with the order of the field.  The
        idempotents are sorted by their vectors of coefficients (see ``RingMember.to_vector``).

        Raise a ValueError if this ring is not semisimple.
        """
        if not self.is_semisimple:
            raise ValueError("Only semisimple rings have primitive central idempotents")
        if self._idempotents is None:
            from qldpc.external import _groups  # avoid circular import

            # Look up the idempotents, or else compute them.  The table stores coefficients as
            # integers in the default representation of the field, so only prime fields and fields
            # with the default irreducible polynomial can use it.
            key = (self.group.to_gap_group(), self.field.order)
            known_idempotents = _groups.KNOWN_PRIMITIVE_CENTRAL_IDEMPOTENTS.get(key)
            if known_idempotents is None or not (
                self.field.degree == 1
                or self.field.irreducible_poly == galois.GF(self.field.order).irreducible_poly
            ):
                idempotents = _get_primitive_central_idempotents(self)
            else:
                idempotents = tuple(
                    RingMember(self, *((coeff, GroupMember(cycles)) for coeff, cycles in terms))
                    for terms in known_idempotents
                )
            self._idempotents = tuple(
                sorted(idempotents, key=lambda idempotent: idempotent.to_vector().tolist())
            )
        return self._idempotents

    def eval(
        self,
        expression: sympy.Expr | sympy.Poly | int | np.integer,
        symbols: dict[sympy.Symbol, GroupMember],
    ) -> RingMember:
        """Evaluate a polynomial in words of the given symbols as a member of this ring.

        This method expands products and nonnegative integer powers of sums into terms.
        Integer-valued floats are read as integers.  It evaluates each term, an integer times a word
        in the given symbols (see ``Group.eval``), as its coefficient times the member of the group
        that its word evaluates to.  Raise a ValueError for any other term, and for any word that
        ``Group.eval`` rejects.  Raise a TypeError if SymPy reads the expression as something other
        than an expression or a polynomial, such as an equation.

        Use noncommutative symbols for members that do not commute.  SymPy sorts commutative symbols
        when it builds an expression, so terms can cancel, as in ``x * y - y * x == 0``.  Expanding
        a product can cancel terms in the same way, as in ``(x + y) * (x - y)``.  Raise a ValueError
        if, in the expression that this method receives, a commutative symbol stands for a member
        that does not commute with the member of a symbol in another factor of the same product, or
        in the base of the same power.
        """
        # helpful error message for invalid symbols
        if any(not isinstance(value, GroupMember) for value in symbols.values()):
            raise ValueError("The symbols passed to GroupRing.eval must be GroupMember-valued")

        # sum the ring members obtained by evaluating each term
        expression = _read_polynomial(expression)
        terms = (self._eval_term(term, symbols) for term in iter_monomial_terms(expression))
        result = functools.reduce(operator.add, terms, self.zero)

        # check the products and powers that expanding the expression may have cancelled
        for node in sympy.preorder_traversal(expression):
            if isinstance(node, sympy.Mul):
                factors = [factor.free_symbols for factor in node.args]
                for left, right in itertools.combinations(factors, 2):
                    self.group._check_commutative_symbols(left, right, symbols)
            elif isinstance(node, sympy.Pow):
                base = node.base.free_symbols
                self.group._check_commutative_symbols(base, base, symbols)
        return result

    def _eval_term(self, term: sympy.Expr, symbols: dict[sympy.Symbol, GroupMember]) -> RingMember:
        """Evaluate an integer times a word in the given symbols as a member of this ring."""
        # split the term into its integer coefficient and its word, which Group.eval checks
        sympy_coeff, word = term.as_coeff_Mul()
        if not _is_integer(sympy_coeff):
            raise ValueError(
                "GroupRing.eval requires terms that are integers times words in the given symbols,"
                f" not {term}"
            )
        coeff = self._eval_int(int(sympy_coeff))
        group_member = self.group.eval(word, symbols)
        return RingMember(self, (coeff, group_member))

    def _lift_in_field(self, matrix: npt.NDArray[np.int_] | galois.FieldArray) -> galois.FieldArray:
        """Read a matrix of a group lift as a matrix over the field of this ring.

        Read a matrix over a finite field as ``_read_field_elements`` does, and an integer matrix as
        integer representations of field elements.
        """
        if isinstance(matrix, galois.FieldArray):
            return self._read_field_elements(matrix, "Entries of group lifts")
        return matrix.view(self.field) if matrix.dtype in self.field.dtypes else self.field(matrix)

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
        """A member of the group of this ring, on as many points as the group acts on.

        A permutation can act on fewer or more points than the group (see ``Group``).  Raise a
        ValueError if it is not in the group.
        """
        if isinstance(label, sympy.combinatorics.Permutation) and label in self.group:
            return self.group._at_degree(label)
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
        else:
            index_length = int(np.ceil(np.log10(num_gens + 1)))
            symbols = [
                sympy.Symbol(f"x_{index:0{index_length}}", commutative=self.group.is_commutative)
                for index in range(num_gens)
            ]

        if isinstance(self.group, AbelianGroup):
            # abelian groups are an easy special case: each cyclic factor of order greater than 1
            # gets a symbol, and the other factors get 1
            nontrivial_symbols = iter(symbols)
            factor_symbols = [
                next(nontrivial_symbols) if order > 1 else sympy.Integer(1)
                for order in self.group.orders
            ]
            monomials = []
            for powers in itertools.product(*[range(order) for order in self.group.orders]):
                factors = [symbol**power for symbol, power in zip(factor_symbols, powers)]
                monomials.append(functools.reduce(operator.mul, factors, sympy.Integer(1)))
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


def _get_primitive_central_idempotents(ring: GroupRing) -> tuple[RingMember, ...]:
    """Compute the primitive central idempotents of a semisimple group algebra ``F_q[G]``.

    Members g and h of G are ``F_q``-conjugate if ``h = y·g^(q^i)·y^(-1)`` for some ``y ∈ G`` and
    integer ``i ≥ 0``.  By the Witt-Berman theorem, the sums over these ``F_q``-conjugacy classes
    form a basis of the subalgebra B of central members z with ``z^q = z``.  If there are n of these
    classes, then B is isomorphic to a product of n copies of ``F_q``.  Every primitive central
    idempotent (PCI) of ``F_q[G]`` lies in B, so the n PCIs of B are those of ``F_q[G]``.

    This function splits the identity of B into orthogonal idempotents until there are n of them.
    Each split takes a random member b of B and an idempotent e found so far.  The minimal
    polynomial m(x) of ``b·e`` in ``e·B`` has distinct roots ``λ ∈ F_q``, and e is the sum of the
    orthogonal idempotents

        ``e_λ = (m(x) / (x - λ)) / m'(λ)``,  evaluated at ``x = b·e``, where ``x^0 = e``.

    The PCIs do not depend on the random choices, but their order does.
    """
    assert ring.is_semisimple

    # This function computes with integer arrays of permutations, so that its largest arrays have
    # |G| × d and n × |G| entries, where d is the number of points that G acts on.  Group.table and
    # adjoint lifts would need |G| × |G| entries.

    # index members of G by their permutations, in the order of the coefficients of ring members
    group, field = ring.group, ring.field
    perms = np.array([member.array_form for member in group._members], dtype=int)
    inverses = np.argsort(perms, axis=1)
    member_index = {perm.tobytes(): index for index, perm in enumerate(perms)}

    def get_indices(perms_to_find: npt.NDArray[np.int_]) -> npt.NDArray[np.int_]:
        return np.array([member_index[perm.tobytes()] for perm in perms_to_find], dtype=int)

    # raise every member g to the power q by repeated squaring, composing members as
    # (a·b)[i] = b[a[i]]
    qth_powers = np.broadcast_to(np.arange(group.degree, dtype=int), perms.shape)
    squares, exponent = perms, field.order
    while exponent:
        if exponent % 2:
            qth_powers = np.take_along_axis(squares, qth_powers, axis=1)
        squares = np.take_along_axis(squares, squares, axis=1)
        exponent //= 2

    # identify F_q-conjugacy classes as the connected components of a graph that joins each member g
    # to g^q and to its conjugates by the generators of G
    neighbors = [get_indices(qth_powers)]
    for generator in group.generators:
        generator_array = np.array(generator.array_form, dtype=int)
        conjugates = np.argsort(generator_array)[perms[:, generator_array]]
        neighbors.append(get_indices(conjugates))
    graph = scipy.sparse.coo_array(
        (
            np.ones(len(neighbors) * group.order),
            (np.tile(np.arange(group.order), len(neighbors)), np.concatenate(neighbors)),
        ),
        shape=(group.order, group.order),
    )
    num_classes, class_of = scipy.sparse.csgraph.connected_components(graph, directed=False)
    representatives = [np.flatnonzero(class_of == label)[0] for label in range(num_classes)]

    # Members of B are vectors of coefficients on class sums.  The coefficient of a product x·y on
    # the class of a member r is the sum over g ∈ G of x(class of r·g^(-1))·y(class of g).
    product_classes = class_of[
        np.array([get_indices(inverses[:, perms[rep]]) for rep in representatives])
    ]

    def multiply(left: galois.FieldArray, right: galois.FieldArray) -> galois.FieldArray:
        return left[product_classes] @ right[class_of]

    # repeatedly split idempotents into orthogonal idempotents, starting from the identity of B
    identity = field.Zeros(num_classes)
    identity[class_of[member_index[np.arange(group.degree, dtype=int).tobytes()]]] = 1
    idempotents = [identity]
    rng = np.random.default_rng(0)
    while len(idempotents) < num_classes:
        random_member = field.Random(num_classes, seed=rng)
        new_idempotents = []
        for idempotent in idempotents:
            element = multiply(random_member, idempotent)
            minimal_poly, element_powers = _get_minimal_polynomial(element, idempotent, multiply)
            # B is a product of copies of F_q, so m(x) has distinct roots in F_q.  galois'
            # Poly.roots searches over F_q, which is faster than factorization for the small fields
            # that qLDPC typically uses.
            roots = minimal_poly.roots()
            assert len(roots) == minimal_poly.degree
            derivative = minimal_poly.derivative()
            for root in roots:
                quotient = minimal_poly // galois.Poly([1, -root], field=field)
                new_idempotents.append(quotient.coeffs[::-1] / derivative(root) @ element_powers)
        idempotents = new_idempotents

    # expand idempotents into coefficients on members of G
    return tuple(
        RingMember._from_trusted_vector(ring, idempotent[class_of]) for idempotent in idempotents
    )


def _get_minimal_polynomial(
    element: galois.FieldArray,
    identity: galois.FieldArray,
    multiply: Callable[[galois.FieldArray, galois.FieldArray], galois.FieldArray],
) -> tuple[galois.Poly, galois.FieldArray]:
    """Find the minimal polynomial of an element of an algebra with a given identity and product.

    The zeroth power of the element is the identity.

    WedderburnArtinComponentTransformer._get_minimal_polynomial does the same with regular lifts,
    whose size is the order of the group.  This function instead works with vectors of length n, the
    dimension of the subalgebra B of _get_primitive_central_idempotents, which is usually much
    smaller than the order of the group.

    Returns:
        - The minimal polynomial.
        - A matrix whose rows are the powers of the element below the degree of that polynomial.
    """
    field = type(element)
    powers = [identity]

    # Reduce each power against the earlier ones.  Keep each reduced power with its pivot, which is
    # its first nonzero entry, and with the combination of powers that it equals.  Divide the
    # reduced power and its combination by the pivot entry, so that the pivot entry becomes 1.  A
    # power that reduces to zero gives the minimal polynomial.
    reduced: list[tuple[int, galois.FieldArray, galois.FieldArray]] = []
    power = identity
    while True:
        degree = len(reduced)
        vector = power
        combination = field.Zeros(degree + 1)
        combination[degree] = 1
        for pivot, row, row_combination in reduced:
            if vector[pivot]:
                factor = vector[pivot]
                vector = vector - factor * row
                combination[: len(row_combination)] -= factor * row_combination
        if not np.any(vector):
            return galois.Poly(combination[::-1]), field(np.array(powers[:degree]))
        pivot = int(np.flatnonzero(vector)[0])
        reduced.append((pivot, vector / vector[pivot], combination / vector[pivot]))
        power = multiply(element, power)
        powers.append(power)


####################################################################################################
# members of rings


class RingMember:
    """An element of a finite ring, which is an algebra over a finite field ``F_q``.

    Each RingMember x is a sum of basis elements of the ring with coefficients in the field:
    ``x = sum_b x_b b``, with each ``x_b in F_q``.  The basis of a group algebra ``F_q[G]`` is the
    group ``G``, so each of its elements is a sum of group members, ``x = sum_{g in G} x_g g``.

    Terms are specified as ``(coefficient, label)`` pairs, where the ring defines the labels of its
    basis elements.  Their coefficients are read as in ``RingMember.from_vector``.  A group member
    as a term has coefficient one, and is read on as many points as the group of the ring acts on
    (see ``Group``).

    A RingMember combines on either side with integers and members of an equal ring, and with field
    scalars on its right, which are read as coefficients (see ``RingMember.from_vector``).  A member
    of a group algebra also combines on either side with members of its group.  Combining a
    RingMember with a member of a different ring raises a ValueError.  So does combining a member of
    a group algebra with a group member outside its group, or constructing a RingMember from a label
    that does not denote a basis element of its ring.  Combining a member of any other ring with a
    group member raises a TypeError.  So does combining a RingMember with a field array, whereas a
    RingMember combines with a ``RingArray`` entrywise.

    An integer operand in ``[0, q)`` is the field element with that integer representation in
    galois.  Over ``GF(4)``, for example, the integer ``2`` is a primitive element, whereas
    ``GF(4)(1) + GF(4)(1)`` is ``0``.  Over a prime field, other integers reduce modulo q.  Over an
    extension field, an integer in ``(-q, 0)`` is the additive inverse of its magnitude, and any
    other integer raises a ValueError.
    """

    _ring: Ring
    _vec: collections.defaultdict[Any, galois.FieldArray]  # coefficient of each basis label

    def __init__(
        self,
        ring: Ring | Group,
        *terms: GroupMember | tuple[int | float | np.integer | np.bool_ | galois.FieldArray, Any],
    ) -> None:
        self._ring = ring if isinstance(ring, Ring) else GroupRing(ring)
        self._vec = collections.defaultdict(lambda: self.field(0))
        for term in terms:
            value, label = (1, term) if isinstance(term, GroupMember) else term
            coefficient = self.ring._read_coefficients(value)
            if coefficient.ndim:
                raise ValueError(f"The coefficient of a term must be a scalar, not {value!r}")
            self._vec[self.ring._as_label(label)] += coefficient

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
            and all(
                self._vec.get(label, 0) == other._vec.get(label, 0)
                for label in self._vec.keys() | other._vec.keys()
            )
        )

    def __bool__(self) -> bool:
        return any(coefficient for coefficient in self._vec.values())

    def __iter__(self) -> Iterator[tuple[galois.FieldArray, Any]]:
        for label, coefficient in self._vec.items():
            yield coefficient, label

    def __add__(self, other: int | galois.FieldArray | GroupMember | RingMember) -> RingMember:
        if isinstance(other, (int, galois.FieldArray, GroupMember)):
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

        if isinstance(other, galois.FieldArray):
            if other.ndim:
                raise TypeError(
                    "A RingMember combines with field scalars, not with a field array of shape"
                    f" {other.shape}\nTry combining ring_member with RingArray.build(array, ring)"
                )

            # multiply coefficients by 'other', read in the field of this ring
            if type(other) is not self.field:
                other = self.ring._read_coefficients(other)
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
        if isinstance(other, (int, galois.FieldArray)):
            return self * other

        if isinstance(other, GroupMember):
            return self._translate(other, on_left=True)

        return NotImplemented

    def __neg__(self) -> RingMember:
        return self * (-1)

    def __pow__(self, power: int | np.integer) -> RingMember:
        if not isinstance(power, (int, np.integer)) or power < 0:
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

        For a group algebra, this method raises a ValueError in the cases that ``GroupRing.lift``
        describes.
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

        If ``right=True``, this method lifts a ring member to its regular representation in the
        opposite ring, such that matrix multiplication corresponds to ring multiplication from the
        right::

            r.regular_lift(right=True) @ s.to_vector() = (s * r).to_vector().

        See `Opposite ring <https://en.wikipedia.org/wiki/Opposite_ring>`_.
        """
        return sum(
            (val * self.ring.regular_lift(label, right=right) for val, label in self if val),
            start=self.field.Zeros([self.ring.dim] * 2),
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
        """The inverse of this RingMember if it exists, and None otherwise."""
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
            return RingMember._from_trusted_vector(
                self.ring, matrix_inv @ self.ring.one.to_vector()
            )
        except np.linalg.LinAlgError:
            return None

    @classmethod
    def from_vector(cls, vector: npt.ArrayLike, ring: Ring | Group) -> RingMember:
        """Construct a ring member from its vector of coefficients, ``(x_b : b in basis)``.

        In a group algebra, the coefficients are indexed by group members, ``(x_g : g in G)``.

        The coefficients may be integers in ``[0, q)``, including booleans and integer-valued
        floats.  Unlike integer operands of ring arithmetic, they are the integer representations of
        field elements, so other integers raise a ValueError.  Field elements may also be
        coefficients if their integer representations denote the same elements of the base field.
        These are the elements of any field class with the same irreducible polynomial as the base
        field, and the elements of the prime field in any field of the same characteristic.  Other
        coefficients raise a ValueError.
        """
        if isinstance(vector, (Ring, Group)):  # DEPRECATED: see qldpc._deprecations
            _deprecations.warn_deprecated(
                "Check argument order: it should be RingMember.from_vector(vector, ring)."
                "  The order (ring, vector) is DEPRECATED and will raise an error in a future"
                " release!",
                stacklevel=2,
            )
            vector, ring = ring, vector
        ring = ring if isinstance(ring, Ring) else GroupRing(ring)
        if np.shape(vector) != (ring.dim,):
            raise ValueError(
                f"A member of a ring of dimension {ring.dim} has {ring.dim} coefficients, but the"
                f" vector has shape {np.shape(vector)}"
            )
        return RingMember._from_trusted_vector(ring, ring._read_coefficients(vector))

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
    def _from_trusted_vector(cls, ring: Ring, vector: galois.FieldArray) -> RingMember:
        """Construct a ring member from a coefficient vector over the field of its ring.

        The caller must check that the vector has length ``ring.dim``.
        """
        ring_member = cls(ring)
        labels = ring._labels
        for index in np.flatnonzero(vector):
            ring_member._vec[labels[index]] = vector[index]
        return ring_member

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
