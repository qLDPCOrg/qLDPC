# SPDX-License-Identifier: Apache-2.0

"""Module for abstract algebra: ring-valued numpy arrays.

.. warning::
    This module does not promise to be performant.  If you need to do heavy numerical abstract
    algebra, you're probably better served by GAP or MAGMA (or maybe SageMath).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, Self

import galois
import numpy as np
import numpy.typing as npt

from qldpc import _deprecations

from ._coefficients import (
    coefficient_matmul,
    ring_array_from_field_array,
    ring_array_to_field_array,
)
from ._howell import get_howell_normal_form_poly, get_howell_normal_form_semisimple
from .groups import Group, GroupMember, NestedSequence, TrivialGroup
from .rings import GroupRing, RingMember

if TYPE_CHECKING:
    from .wedderburn_artin import WedderburnArtinTransformer


####################################################################################################
# RingArray: RingMember-valued array


class RingArray(np.ndarray[Any, np.dtype[np.object_]]):
    """Array whose entries are members of a GroupRing."""

    # None only for a view of a plain array that has not yet been assigned a ring
    _ring: GroupRing | None

    # Howell-form provenance, attached by howell_normal_form_* to their direct output only (via
    # _mark_as_hnf).  These deliberately do NOT propagate through numpy operations:
    # __array_finalize__ copies only _ring, so any slice, transpose, or arithmetic result falls back
    # to the class defaults below.  A True _in_hnf therefore reliably means "this exact object is a
    # fresh Howell normal form", which is what get_howell_dual relies on.
    _in_hnf: bool = False
    _hnf_transformer: WedderburnArtinTransformer | None = None
    _hnf_right: bool = False

    def __new__(
        cls,
        data: npt.NDArray[np.object_] | NestedSequence,
        ring: GroupRing | Group | None = None,
    ) -> Self:
        array = np.asarray(data, dtype=object).view(cls)
        ring = GroupRing(ring) if isinstance(ring, Group) else ring

        # identify the base group for this RingArray
        for value in array.ravel():
            if not isinstance(value, RingMember):
                raise TypeError(
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
        # A view of a plain array (obj without _ring) has no ring until its creator assigns one,
        # which every .view(RingArray) in this module does immediately.  Howell-form provenance is
        # intentionally NOT propagated here (see the class attributes).
        self._ring = getattr(obj, "_ring", None)

    def __array_function__(
        self,
        func: Any,
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Intercept array operations to ensure RingArray compatibility."""
        # ``_iter_ring_arrays`` descends into sequence arguments (e.g. the list passed to
        # np.concatenate/np.stack), so RingArrays nested one or more levels deep are still checked
        # for a ring mismatch -- a plain scan of the top-level args would miss them.
        rings = [
            self._ring,
            *(arr._ring for arr in _iter_ring_arrays(args)),
            *(arr._ring for arr in _iter_ring_arrays(tuple(kwargs.values()))),
        ]
        if any(ring != rings[0] for ring in rings[1:]):
            raise ValueError("Cannot perform operations on RingArrays with different base rings")
        args = tuple(_unwrap_ring_arrays(x) for x in args)
        result: object = super().__array_function__(func, types, args, kwargs)
        if isinstance(result, np.ndarray):
            result = result.view(RingArray)
            result._ring = rings[0]
        return result

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *inputs: npt.NDArray[np.object_],
        **kwargs: object,
    ) -> RingArray | RingMember | None:
        """Intercept array operations to ensure RingArray compatibility."""
        rings = {self._ring} | {x._ring for x in inputs if isinstance(x, RingArray)}
        if len(rings) > 1:
            raise ValueError("Cannot perform operations on RingArrays with different base rings")
        if (
            ufunc is np.matmul
            and method == "__call__"
            and not kwargs
            and self._ring is not None
            and (product := coefficient_matmul(*inputs, ring=self._ring)) is not None
        ):
            return product
        inputs = tuple(x.view(np.ndarray) if isinstance(x, RingArray) else x for x in inputs)
        result: RingArray | RingMember | None = super().__array_ufunc__(
            ufunc, method, *inputs, **kwargs
        )
        if isinstance(result, np.ndarray):
            result = result.view(RingArray)
            result._ring = next(iter(rings), None)
        return result

    def __str__(self) -> str:
        return np.array2string(self, formatter={"object": str}, separator=", ")

    @property
    def ring(self) -> GroupRing:
        """Base ring of this RingArray."""
        if self._ring is None:
            raise ValueError("This RingArray has no base ring (it may be a view of a plain array)")
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
        self._require_ndim(1, 2)
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
        self._require_ndim(1, 2)
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
        "transposes" each array element, which takes group members ``g -> ~g = g**-1``.
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

        Integers and group members are cast into members of the ring.  Ring members may be embedded
        into a different ring only when the coefficient fields match and their group support embeds
        into the target group.
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
            # validate and re-home explicit ring members
            if isinstance(value, RingMember):
                if value.field is not ring.field:
                    raise ValueError(
                        f"Cannot embed a ring member over GF({value.field.order}) into a ring over"
                        f" GF({ring.field.order}): incompatible coefficient fields"
                    )
                terms = [
                    (coefficient, member) for coefficient, member in value * one if coefficient
                ]
                if any(member != one and member not in ring.group for _, member in terms):
                    raise ValueError(
                        "Cannot embed a ring member whose group support is not contained in the"
                        " target group"
                    )
                return RingMember(ring, *terms)
            if isinstance(value, GroupMember):
                return RingMember(ring, value * one)
            return RingMember(ring, (value, one))

        vals = [as_ring_member(value) for value in array.ravel()]
        result = np.array(vals, dtype=object).reshape(array.shape).view(RingArray)
        result._ring = ring
        return result

    def to_field_array(self) -> galois.FieldArray:
        """Convert a RingArray into an array of coefficients (in a finite field) for each entry.

        This method expands every entry of a ``RingArray`` into a vector of length
        ``ring.group.order``.  If ``ring_array`` is two-dimensional, for example, then
        ``ring_array.to_field_array()[a, b, :]`` is the vector of coefficients for the
        ``RingMember`` at ``ring_array[a, b]``.
        """
        return ring_array_to_field_array(self, self.ring)

    @classmethod
    def from_field_array(cls, array: npt.NDArray[np.int_], ring: GroupRing | Group) -> RingArray:
        """Construct a RingArray from an array of coefficients (in a finite field) for each entry.

        This method is the inverse of ``RingArray.to_field_array``.
        """
        if isinstance(array, (GroupRing, Group)):  # DEPRECATED: see qldpc._deprecations
            _deprecations.warn_deprecated(
                "Check argument order: it should be RingArray.from_field_array(array, ring)."
                "  The order (ring, array) is DEPRECATED and will throw an error in the future!",
                stacklevel=2,
            )
            array, ring = ring, array
        ring = ring if isinstance(ring, GroupRing) else GroupRing(ring)

        # read values as integers first (like RingMember.from_vector) to accept any integer-valued
        # array, such as a boolean or floating-point array
        coefficients = ring.field(np.asarray(array).astype(int))
        if coefficients.ndim == 0 or coefficients.shape[-1] != ring.group.order:
            raise ValueError(
                "The last axis of a RingArray coefficient array must have length"
                f" {ring.group.order}"
            )
        return ring_array_from_field_array(coefficients, ring)

    def to_field_vector(self) -> galois.FieldArray:
        """Convert RingArray into a flattened 1-D vector of coefficients for each RingMember."""
        return self.to_field_array().ravel().view(self.field)

    @classmethod
    def from_field_vector(cls, vector: npt.NDArray[np.int_], ring: GroupRing | Group) -> RingArray:
        """Construct a 1-D RingArray from a vector of coefficients.

        This method is the inverse of ``RingArray.to_field_vector``.
        """
        if isinstance(vector, (GroupRing, Group)):  # DEPRECATED: see qldpc._deprecations
            _deprecations.warn_deprecated(
                "Check argument order: it should be RingArray.from_field_vector(vector, ring)."
                "  The order (ring, vector) is DEPRECATED and will throw an error in the future!",
                stacklevel=2,
            )
            vector, ring = ring, vector
        vector = np.asanyarray(vector)
        group = ring.group if isinstance(ring, GroupRing) else ring
        entries_as_vecs = vector.reshape(vector.size // group.order, group.order)
        return RingArray.from_field_array(entries_as_vecs, ring)

    def null_space(self, *, right: bool = False) -> RingArray:
        """Construct a matrix of null-space row vectors for this RingArray.

        The transpose of the null-space matrix is annihilated by this ``RingArray``, such that
        ``np.any(self @ self.null_space().T)`` is ``np.False_``.

        If ``right is True``, this method constructs a null space over the opposite ring, in which
        the order of multiplication is reversed.

        Due to the subtleties of defining row reduction for a matrix over a ring, this method does
        not row-reduce the matrix of null-space row vectors.  The rows of the matrix returned by
        this method are therefore generally an overcomplete basis for the null space of this
        ``RingArray``.
        """
        self._require_ndim(2)

        # field-valued null vectors of self.regular_lift() provide an overcomplete basis for the
        # space of ring-valued null vectors
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

        Without a transformer, this method may run GAP, prompt on standard input, and use the system
        clipboard; see GroupRing.get_transformer.
        """
        self._require_ndim(2)
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

        - ``RingArray.howell_normal_form_semisimple`` (if ``poly is False``, the default), or
        - ``RingArray.howell_normal_form_poly`` (if ``poly is True``).

        See the documentation of those methods for additional information, including the GAP
        computations, prompts, and clipboard use that ``howell_normal_form_semisimple`` may need.
        """
        if poly:
            return self.howell_normal_form_poly()
        return self.howell_normal_form_semisimple()

    def howell_normal_form_semisimple(
        self, transformer: WedderburnArtinTransformer | None = None, *, right: bool = False
    ) -> RingArray:
        """Compute a Howell normal form (HNF) of a RingArray over a semisimple ring.

        This method first puts a ``RingArray`` into a generalized reduced row echelon form (see
        ``RingArray.row_reduce``), then further post-processes the rows to satisfy the Howell
        property, whereby an element ``v`` that is...

            - in the row span of the matrix, and
            - has j leading zeros, meaning ``= (0_1, 0_2, ..., 0_j, v_{j+1}, ...)``,

        can be written as a linear combination of rows whose pivots are at position ``k >= j``.

        The Howell property is enforced as follows: if a row ``r`` has a pivot ``p`` with a
        nontrivial left annihilator ``α``, meaning::

              α != 0,
            α·p  = 0,
            α·r != 0,

        then the row ``r`` is replaced by ``(1-α)·r``, and the row ``α·r`` is appended to the
        matrix.

        If ``right is True``, the Howell property is instead enforced for nontrivial right
        annihilators::

              α != 0,
            p·α  = 0,
            r·α != 0,

        for which the row ``r`` is replaced by ``(1-α)·r``, and the row ``α·r`` is appended to the
        matrix.  The ordinary HNF and right-HNF are equal for a ``RingArray`` over a commutative
        ring.

        The HNF of a ``RingArray`` over a commutative ring is unique.  For non-commutative rings,
        the HNF is only unique up to a choice of matrix basis for simple components of the ring.

        References:

        - https://en.wikipedia.org/wiki/Howell_normal_form
        - https://github.com/m-webster/XPFpackage/blob/570ea89/Examples/A.1_howell_matrix.ipynb

        Without a transformer, this method may run GAP, prompt on standard input, and use the system
        clipboard; see GroupRing.get_transformer.
        """
        return get_howell_normal_form_semisimple(self, transformer, right=right)

    def howell_normal_form_poly(self) -> RingArray:
        """Compute a Howell normal form of a RingArray using polynomial division.

        If the base ring of a RingArray is a cyclic group algebra, then it can be interpreted as a
        univariate polynomial ring, allowing us to compute greatest common divisors and perform row
        reduction with polynomial division.

        References:

        - https://en.wikipedia.org/wiki/Howell_normal_form
        - https://github.com/m-webster/XPFpackage/blob/570ea89/Examples/A.1_howell_matrix.ipynb
        """
        return get_howell_normal_form_poly(self)

    def reduced_groebner_basis(self) -> RingArray:
        """Compute a reduced Groebner basis for this RingArray.

        At least, that's the plan.  This method is not yet implemented.
        """
        self._require_ndim(2)
        raise NotImplementedError(
            "Computing a reduced Groebner basis is very mathematically involved.  Here be dragons."
        )

    def _require_ndim(self, *ndims: int) -> None:
        """Raise a ValueError unless this RingArray has one of the given numbers of dimensions."""
        if self.ndim not in ndims:
            expected = " or ".join(map(str, ndims))
            raise ValueError(f"This method requires a {expected}-dimensional RingArray")

    def _mark_as_hnf(self, *, transformer: WedderburnArtinTransformer | None, right: bool) -> Self:
        """Record that this array is a Howell normal form, as provenance for get_howell_dual."""
        self._in_hnf = True
        self._hnf_transformer = transformer
        self._hnf_right = right
        return self


def _iter_ring_arrays(obj: object) -> Iterator[RingArray]:
    """Yield every RingArray in a (possibly nested) numpy-function argument.

    Functions such as ``np.concatenate`` and ``np.stack`` take a *sequence* of arrays as a single
    argument, so the RingArrays are nested one level (or more) inside a list/tuple rather than
    passed directly.
    """
    if isinstance(obj, RingArray):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _iter_ring_arrays(item)


def _unwrap_ring_arrays(obj: object) -> object:
    """Replace every RingArray in a (possibly nested) argument with its plain-ndarray view."""
    if isinstance(obj, RingArray):
        return obj.view(np.ndarray)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_unwrap_ring_arrays(item) for item in obj)
    return obj


if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
