# SPDX-License-Identifier: Apache-2.0

"""Module for abstract algebra: ring-valued numpy arrays.

.. warning::
    This module does not promise to be performant.  If you need to do heavy numerical abstract
    algebra, you're probably better served by GAP or MAGMA (or maybe SageMath).

"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Iterable, Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal

import galois
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

import qldpc
from qldpc._util import get_deprecated_alias

from .groups import CyclicGroup, Group, GroupMember, NestedSequence, TrivialGroup
from .rings import GroupRing, RingMember

if TYPE_CHECKING:
    from .wedderburn_artin import WedderburnArtinTransformer

################################################################################
# RingArray: RingMember-valued array


class RingArray(np.ndarray[Any, np.dtype[np.object_]]):
    """Array whose entries are members of a GroupRing."""

    _ring: GroupRing

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
        # obj may be None or lack _ring during numpy view construction; _ring is set before use.
        # Howell-form provenance is intentionally NOT propagated here (see the class attributes).
        self._ring = getattr(obj, "_ring", None)  # type:ignore[assignment]

    def _mark_as_hnf(self, *, transformer: WedderburnArtinTransformer | None, right: bool) -> Self:
        """Record that this array is a Howell normal form, as provenance for get_howell_dual."""
        self._in_hnf = True
        self._hnf_transformer = transformer
        self._hnf_right = right
        return self

    def __array_function__(
        self,
        func: Any,
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> RingArray | None:
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
        result = super().__array_function__(func, types, args, kwargs)
        if isinstance(result, np.ndarray):
            result = result.view(RingArray)
            result._ring = rings[0]  # type:ignore[attr-defined]
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
            result._ring = next(iter(rings), None)  # type:ignore[attr-defined]
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
        vals = [val.to_vector() for val in self.ravel()]
        return np.asarray(vals, dtype=int).reshape(*self.shape, self.group.order).view(self.field)

    @classmethod
    def from_field_array(cls, array: npt.NDArray[np.int_], ring: GroupRing | Group) -> RingArray:
        """Construct a RingArray from an array of coefficients (in a finite field) for each entry.

        This method is the inverse of ``RingArray.to_field_array``.
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

        This method is the inverse of ``RingArray.to_field_vector``.
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

        The transpose of the null-space matrix is annihilated by this ``RingArray``, such that
        ``np.any(self @ self.null_space().T)`` is ``np.False_``.

        If ``right is True``, this method constructs a null space over the opposite ring, in which
        the order of multiplication is reversed.

        Due to the subtleties of defining row reduction for a matrix over a ring, this method does
        not row-reduce the matrix of null-space row vectors.  The rows of the matrix returned by
        this method are therefore generally an overcomplete basis for the null space of this
        ``RingArray``.
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

        - ``RingArray.howell_normal_form_semisimple`` (if ``poly is False``, the default), or
        - ``RingArray.howell_normal_form_poly`` (if ``poly is True``).

        See the documentation of those methods for additional information.
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
        matrix.
        The ordinary HNF and right-HNF are equal for a ``RingArray`` over a commutative ring.

        The HNF of a ``RingArray`` over a commutative ring is unique.  For non-commutative rings,
        the HNF is only unique up to a choice of matrix basis for simple components of the ring.

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
            # Identify:
            # 1. The column of the first nonzero value in the pivot_row of each component.
            # 2. The column that will contain the pivot when we recombine the components.
            pivot_rows_as_bools = [
                np.any(matrix[pivot_row].view(np.ndarray).astype(bool), axis=(1, 2))
                for matrix in matrices
            ]
            pivot_cols = qldpc.math.first_nonzero_cols(pivot_rows_as_bools)
            pivot_col = min(pivot_cols)

            # Let π be a projector onto the components in which the pivot is nonzero.  If π != 1,
            # then (1-π) is a nontrivial annihilator of the pivot.  If, moreover, (1-π)·r is
            # nonzero, then (1-π)·r contains a "hidden" pivot in a later column.  In this case, we
            # in principle need to replace r -> π·r and add (1-π)·r as a new row to the matrix.  In
            # practice, this procedure messes up the reduced row echelon form of the matrix, so we
            # instead...
            # 1. In the (1-π) sector, insert a zero row at the pivot_row and shift down rows below.
            # 2. In the π sector, append a zero row to the matrix.
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
        return transformer.recompose_array(matrices)._mark_as_hnf(
            transformer=transformer, right=right
        )

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
            raise TypeError(
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
            represents a ``(vecs.ndim - 1)``-dimensional array of polynomials, such that
            ``vecs[*entry, c]`` is the coefficient of ``x**c`` in the given entry of ``vecs``.
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
            for other_row in range(pivot_row + 1, field_array.shape[0]):
                aa_vec = field_array[pivot_row]
                bb_vec = field_array[other_row]
                if not np.any(bb_vec):
                    continue
                # Let:
                #     aa = aa_vec[pivot_row]
                #     bb = bb_vec[other_row]
                # We will transform rows as
                #     [aa_vec, bb_vec] --> [[ss, tt], [uu, vv]] @ [aa_vec, bb_vec]
                # where
                #     (1) ss * aa + tt * bb = gcd(aa, bb) = gg
                #     (2) uu * aa + vv * bb = 0
                #     (3) det([[ss, tt], [uu, vv]]) = ss * vv - tt * uu = 1
                # Condition (3) ensures that this transformation is invertible.
                # Condition (2) ensures that bb_vec gets zeroed out at the pivot column.
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

            # "Reduce" the pivot to gcd(pivot, modulus):
            #     (1) Find ff for which ff * pivot = gcd(pivot, modulus) = gg.
            #     (2) Replace the pivot row with ff * (pivot row), reducing the pivot to gg.
            # Multiplying the whole row by the non-unit ff would shrink the row-module span (it
            # scales every column, not just the pivot), so preserve the span by keeping the
            # residual: the original row minus its reconstruction quotient * (reduced row), where
            # quotient = pivot / gg (exact, and quotient * gg = pivot with no reduction modulo
            # x^n - 1 since deg(pivot) < n).  That residual is therefore zero in the pivot column
            # and, being a ring-multiple of the pivot row, adds no span; it is resolved later.
            pivot_poly = galois.Poly(field_array[pivot_row, pivot_col, ::-1], field=self.field)
            gcd_poly: galois.Poly
            ff_poly: galois.Poly
            gcd_poly, ff_poly, _ = galois.egcd(pivot_poly, modulus_poly)  # type:ignore[assignment,arg-type]
            if pivot_poly != gcd_poly:
                quotient_poly = pivot_poly // gcd_poly
                reduced_row = _multiply(ff_poly, field_array[pivot_row])
                residual_row = field_array[pivot_row] - _multiply(quotient_poly, reduced_row)
                field_array[pivot_row] = reduced_row
                field_array = np.append(field_array, [residual_row], axis=0).view(self.field)
                pivot_poly = gcd_poly

            # Reduce all rows above the pivot_row at the pivot_column.
            # If some value in the pivot_col above the pivot_row can be written as a multiple of the
            # pivot plus a remainder, use row operations to subtract off that multiple of the pivot,
            # leaving only the remainder.
            for other_row in range(pivot_row):
                other_poly = galois.Poly(field_array[other_row, pivot_col, ::-1], field=self.field)
                div_poly = other_poly // pivot_poly
                if div_poly != 0:
                    field_array[other_row] -= _multiply(div_poly, field_array[pivot_row])

            # Check whether the pivot has a nontrivial annihilator, with annihilator * pivot = 0.
            # If a nontrivial annihilator is found, append a new row with the pivot annihilated.
            annihilator_poly = modulus_poly // pivot_poly
            if annihilator_poly != 0:
                new_row = _multiply(annihilator_poly, field_array[pivot_row])
                field_array = np.append(field_array, [new_row], axis=0).view(self.field)

            pivot_row += 1
            pivot_col += 1

        # remove all-zero rows and return
        field_array = field_array[np.any(field_array, axis=(1, 2))]
        return RingArray.from_field_array(field_array, self.ring)._mark_as_hnf(
            transformer=None, right=False
        )

    def reduced_groebner_basis(self) -> RingArray:
        """Compute a reduced Groebner basis for this RingArray.

        At least, that's the plan.  This method is not yet implemented.
        """
        assert self.ndim == 2
        raise NotImplementedError(
            "Computing a reduced Groebner basis is very mathematically involved.  Here be dragons."
        )


def _iter_ring_arrays(obj: Any) -> Iterator[RingArray]:
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


def _unwrap_ring_arrays(obj: Any) -> Any:
    """Replace every RingArray in a (possibly nested) argument with its plain-ndarray view."""
    if isinstance(obj, RingArray):
        return obj.view(np.ndarray)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_unwrap_ring_arrays(item) for item in obj)
    return obj


def _get_block_howell_form(matrix: galois.FieldArray, *, right: bool = False) -> galois.FieldArray:
    """Compute a block-Howell normal form of the provided block matrix.

    The provided matrix should be 4-dimensional, with matrix[i, j] storing a square block at (i, j).
    The block-Howell form is essentially the same as the row-reduced echelon form when the matrix is
    expanded into a 2-dimensional array, except zero rows are inserted to shift pivots down so that
    they always lie on the diagonal of a block.
    """
    shape: tuple[int, ...]

    assert matrix.ndim == 4 and matrix.shape[-1] == matrix.shape[-2]
    field = type(matrix)
    num_block_rows, num_block_cols, size, _ = matrix.shape

    if right and size > 1:
        # transpose each block to turn right-side reduction into the existing left-side case
        matrix = matrix.transpose(0, 1, 3, 2)

    # row-reduce as an expanded 2-D matrix, keeping a basis of the row space (no all-zero rows)
    shape = (num_block_rows * size, num_block_cols * size)
    matrix = matrix.transpose(0, 2, 1, 3).reshape(shape).view(field).row_space()

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


_DEPRECATED_ALIASES = {"Protograph": RingArray}

# Deprecated names resolve at runtime through a module-level __getattr__ that warns when accessed.
# Type checkers instead see plain aliases, so that they still flag misspelled attributes.
if TYPE_CHECKING:
    Protograph = RingArray
else:

    def __getattr__(name: str) -> Any:
        """Resolve deprecated names of ring arrays, with a DeprecationWarning."""
        return get_deprecated_alias(__name__, name, _DEPRECATED_ALIASES)
