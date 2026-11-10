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
from .rings import GroupRing, Ring, RingMember, _as_object_array

if TYPE_CHECKING:
    from .wedderburn_artin import WedderburnArtinTransformer

# numpy functions whose results are ring members, except for empty inputs, for which numpy returns
# the integer 0 or 1
_RING_VALUED_FUNCTIONS = {
    np.sum,
    np.prod,
    np.nansum,
    np.nanprod,
    np.trace,
    np.dot,
    np.inner,
    np.einsum,
}

# numpy functions that write values into their first argument, with the name of that argument, and
# the position and name of the argument for the values
_IN_PLACE_FUNCTIONS = {
    np.copyto: ("dst", 1, "src"),
    np.put: ("a", 2, "v"),
    np.place: ("arr", 2, "vals"),
    np.putmask: ("a", 2, "values"),
    np.put_along_axis: ("arr", 2, "values"),
    np.fill_diagonal: ("a", 1, "val"),
}


####################################################################################################
# RingArray: RingMember-valued array


class RingArray(np.ndarray[Any, np.dtype[np.object_]]):
    """Array whose entries are members of a Ring.

    Values written into a RingArray, by indexing, ``fill``, numpy functions such as ``np.copyto``,
    the ``out`` keyword argument of a numpy function, or the ``out`` argument of a ufunc, are read
    as in ``RingArray.build``.  Unlike in ``RingArray.build``, members of other rings raise a
    ValueError, as in ring arithmetic, and None, which ``np.empty_like`` gives, stays None.  To
    embed members of another group algebra, use ``RingArray.build`` explicitly.  A plain array of
    objects given as ``out`` is read in place, and returned as a RingArray view of it.

    Numpy functions and ufuncs read every field array among their inputs, including field scalars,
    as ring members, so arguments that serve as indices, masks, counts, or shifts must be plain
    integer or boolean arrays, such as ``np.asarray(field_array)``.  They also read the entries of
    the arrays of objects that they return, and read the integers that numpy returns for empty sums
    and products as ring members.

    Numpy operations with non-object results, such as comparisons, return plain arrays, and ufuncs
    that give a boolean for a 0-dimensional RingArray, such as ``np.logical_not``, return that
    boolean rather than a ring member.  Ufuncs from ``np.vectorize`` and ``np.frompyfunc`` return
    their results as computed, and those from ``np.frompyfunc`` raise a TypeError when given a
    RingArray as ``out``.  A ufunc or operator with a field array on the left, or a field array
    given as ``out``, raises an error from galois.  A numpy function whose first array argument is a
    field array is handled by galois and does not return a RingArray, as in
    ``np.concatenate([field_array, ring_array])``, so convert the field array with
    ``RingArray.build`` before the call.  A numpy function or ufunc that raises while writing into
    ``out``, as in ``ring_array += values``, may leave the values that it wrote, unread.

    Values written through ``flat``, the method ``put``, a plain view such as ``np.asarray(a)``, a
    positional ``out`` argument of a numpy function, or the method ``at`` of a ufunc are not read.
    The method ``resize`` fills new entries with the integer 0, ``astype(bool)`` returns a RingArray
    of booleans, and the method ``dot`` gives integer zeros for an empty inner dimension.
    """

    # None only for a view of a plain array that has not yet been assigned a ring
    _ring: Ring | None

    # Howell-form provenance, which howell_normal_form_semisimple records on its output (see
    # _mark_as_hnf): the transformer, the orientation, and a snapshot of the entry objects.
    # get_howell_dual trusts the output only while the array holds the same entry objects in the
    # same shape.  __array_finalize__ copies only _ring, so copies, slices, transposes, and
    # arithmetic results have no provenance.
    _hnf_provenance: tuple[WedderburnArtinTransformer, bool, npt.NDArray[np.object_]] | None = None

    def __new__(
        cls,
        data: npt.NDArray[np.object_] | NestedSequence,
        ring: Ring | Group | None = None,
    ) -> Self:
        array = np.asarray(data, dtype=object).view(cls)
        base_ring = GroupRing(ring) if isinstance(ring, Group) else ring

        # a RingArray given as data keeps its ring, which matters when it has no entries
        if base_ring is None and isinstance(data, RingArray):
            base_ring = data._ring

        # identify the base ring for this RingArray
        for value in array.ravel():
            if not isinstance(value, RingMember):
                raise TypeError(
                    "Requirement failed: all entries of a RingArray must be RingMember-valued."
                    "\nTry building an array with RingArray.build(...)"
                )
            else:
                if not (base_ring is None or base_ring == value.ring):
                    raise ValueError("Inconsistent rings provided for a RingArray")
                base_ring = value.ring

        if base_ring is None:
            raise ValueError("Cannot determine the underlying ring for a RingArray")
        array._ring = base_ring

        return array

    def __array_finalize__(self, obj: npt.NDArray[np.object_] | None) -> None:
        """Propagate metadata to newly constructed arrays."""
        # A view of a plain array (obj without _ring) has no ring until its creator assigns one,
        # which every .view(RingArray) in this module does immediately.  Howell-form provenance is
        # not copied (see the class attributes).
        self._ring = getattr(obj, "_ring", None)

    def __array_function__(
        self,
        func: Any,
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any:
        """Intercept array operations to ensure RingArray compatibility."""
        # _iter_ring_arrays also finds RingArrays inside sequence arguments, such as the list passed
        # to np.concatenate
        rings = [
            self._ring,
            *(arr._ring for arr in _iter_ring_arrays(args)),
            *(arr._ring for arr in _iter_ring_arrays(tuple(kwargs.values()))),
        ]
        if any(ring != rings[0] for ring in rings[1:]):
            raise ValueError("Cannot perform operations on RingArrays with different base rings")
        ring = rings[0]

        # read the values that numpy functions write into a RingArray before numpy writes them
        args = tuple(args)
        if func in _IN_PLACE_FUNCTIONS:
            target_name, position, name = _IN_PLACE_FUNCTIONS[func]
            target = args[0] if args else kwargs[target_name]
            if isinstance(target, RingArray):
                if len(args) > position:
                    value = target._read_value(args[position])
                    args = (*args[:position], value, *args[position + 1 :])
                else:
                    kwargs = {**kwargs, name: target._read_value(kwargs[name])}

        # numpy computes on plain arrays of ring members, and writes into the plain view of a
        # RingArray out=
        plain_args = tuple(_read_operand(x, ring) for x in args)
        plain_kwargs = {
            key: _read_operand(value, None if key == "out" else ring)
            for key, value in kwargs.items()
        }
        result = super().__array_function__(func, types, plain_args, plain_kwargs)
        if result is NotImplemented:
            return result

        # read the values that numpy writes into out=
        output = kwargs.get("out")
        if isinstance(output, np.ndarray) and output.dtype == object:
            return _read_in_place(output, ring)

        # read the integers 0 and 1 that numpy returns for empty sums and products, and wrap arrays,
        # also inside lists and tuples (as from np.split)
        if func in _RING_VALUED_FUNCTIONS and isinstance(result, (int, np.integer)):
            return result if ring is None else ring.one * int(result)
        if isinstance(result, (list, tuple)):
            items = [_wrap_result(item, ring) for item in result]
            result_type: Any = type(result)
            return result_type(*items) if hasattr(result, "_fields") else result_type(items)
        return _wrap_result(result, ring)

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: Literal["__call__", "reduce", "reduceat", "accumulate", "outer", "at"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any:
        """Intercept array operations to ensure RingArray compatibility."""
        outputs = kwargs.get("out", ())
        rings = {x._ring for x in (self, *inputs, *outputs) if isinstance(x, RingArray)}
        if len(rings) > 1:
            raise ValueError("Cannot perform operations on RingArrays with different base rings")
        ring = next(iter(rings))
        if (
            ufunc is np.matmul
            and method == "__call__"
            and not kwargs
            and ring is not None
            and (product := coefficient_matmul(*inputs, ring=ring)) is not None
        ):
            return product

        # ufuncs from np.vectorize and np.frompyfunc give values of any type, which are returned as
        # computed; they do not write into RingArrays given as out=
        user_defined = ufunc.__name__.endswith(" (vectorized)")

        # numpy computes on plain arrays of ring members, and writes into the plain views of
        # RingArray outputs
        arrays = tuple(_read_operand(x, ring) for x in inputs)
        if outputs and not user_defined:
            kwargs["out"] = tuple(_read_operand(x, None) for x in outputs)
        result = super().__array_ufunc__(ufunc, method, *arrays, **kwargs)
        if user_defined or result is NotImplemented:
            return result

        # read the values that numpy writes into out=
        if outputs:
            results = tuple(
                _read_in_place(output, ring)
                if isinstance(output, np.ndarray) and output.dtype == object
                else plain
                for output, plain in zip(outputs, result if ufunc.nout > 1 else (result,))
            )
            return results if ufunc.nout > 1 else results[0]

        # read the integers 0 and 1 that numpy returns for empty sums and products; logical ufuncs
        # give Python bools, which stay bools
        if isinstance(result, (int, np.integer)) and not isinstance(result, bool):
            return result if ring is None else ring.one * int(result)
        return _wrap_result(result, ring)

    def __setitem__(self, key: Any, value: Any) -> None:
        super().__setitem__(key, self._read_value(value))

    def fill(self, value: Any) -> None:
        """Set every entry of this RingArray to a value, read as for indexing (see RingArray)."""
        self[...] = value

    def __str__(self) -> str:
        return np.array2string(self, formatter={"object": str}, separator=", ")

    @property
    def ring(self) -> Ring:
        """Base ring of this RingArray."""
        if self._ring is None:
            raise ValueError("This RingArray has no base ring (it may be a view of a plain array)")
        return self._ring

    @property
    def group(self) -> Group:
        """Base group of this RingArray.

        Raise an AttributeError if this RingArray is over a ring other than a group algebra.
        """
        if isinstance(self.ring, GroupRing):
            return self.ring.group
        raise AttributeError(f"{self.ring} is not a group algebra, and has no base group")

    @property
    def field(self) -> type[galois.FieldArray]:
        """Base field of this RingArray."""
        return self.ring.field

    def regular_lift(self, *, right: bool = False) -> galois.FieldArray:
        """Block matrix obtained by a regular lift of each entry of this RingArray."""
        self._require_ndim(1, 2)
        rows = 1 if self.ndim == 1 else self.shape[0]
        cols = self.shape[-1]
        block_size = self.ring.dim
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
        block_size = self.ring.lift_dim
        if 0 in (rows, cols):
            return self.field.Zeros((rows * block_size, cols * block_size))
        blocks = [
            [val.lift(right=right) for val in row] for row in self.reshape(-1, self.shape[-1])
        ]
        return np.block(blocks).view(self.field)

    def __invert__(self) -> RingArray:
        """Transpose the entries of this RingArray, as in ``RingMember.T``."""
        # check the ring here, so that an empty array also raises an error; an array without a base
        # ring transposes its entries in their own rings
        if self._ring is not None and not self._ring.has_involution:
            raise ValueError(f"{self._ring} has no involution, so its members cannot be transposed")
        vals = [val.T for val in self.ravel()]
        array = np.array(vals, dtype=object).reshape(self.shape).view(RingArray)
        array._ring = self._ring
        return array

    @property
    def T(self) -> RingArray:
        """Conjugate-transpose of a matrix over a ring.

        In addition to transposing the first two indices of the array, this property "conjugates" or
        "transposes" each array element with the involution of the ring (see ``RingMember.T``).  For
        a group algebra, this takes group members ``g -> ~g = g**-1``.  This property raises a
        ValueError for an array over a ring without an involution (see ``Ring.has_involution``).
        """
        return (~self).transpose(1, 0, *np.arange(2, self.ndim))

    @staticmethod
    def build(
        data: npt.NDArray[np.int_] | npt.NDArray[np.object_] | NestedSequence,
        ring: Ring | Group | None = None,
    ) -> RingArray:
        """Construct a RingArray.

        The constructed array is built from:

        1. An array populated by

            (a) ring members,
            (b) group members (for a group algebra), or
            (c) integers, floats with integer values, or field elements.

        2. A ring (or group, inducing a group algebra over GF(2)).

        Without a ring, the ring is that of the ring members in the array.  A RingArray given as
        data with no entries keeps its ring.  Otherwise, the ring is the algebra of the trivial
        group over a field.  If the data is a field array, that is its field.  Otherwise, it is the
        field of the field elements in the data, or GF(2) if the data has no field elements.  Field
        elements over different fields raise a ValueError.

        Integers, floats, field elements, and members of the group of a group algebra are cast into
        members of the ring.  Integers are read as in ring arithmetic (see ``RingMember``), and
        floats and field elements as coefficients (see ``RingMember.from_vector``).  A group member
        that acts on fewer points than the group is padded with fixed points.  Members of a group
        algebra may be embedded into a different group algebra with the same field if the group of
        that algebra contains their group members.  Members of any other ring must already belong to
        the target ring.  An entry that is itself an array, such as a list or field array in an
        object array, raises a ValueError.
        """
        # read nested sequences as objects, which keeps the field of field elements
        array = data if isinstance(data, np.ndarray) else _as_object_array(data)

        # identify the base ring
        if ring is None:
            rings = {value.ring for value in array.ravel() if isinstance(value, RingMember)}
            if len(rings) > 1:
                raise ValueError("Inconsistent rings provided to RingArray.build")
            if rings:
                ring = next(iter(rings))
            elif isinstance(array, RingArray) and array._ring is not None:
                ring = array._ring  # a RingArray with no entries keeps its ring
            else:
                fields = (
                    {type(array)}
                    if isinstance(array, galois.FieldArray)
                    else {
                        type(value)
                        for value in array.ravel()
                        if isinstance(value, galois.FieldArray)
                    }
                )
                if len(fields) > 1:
                    raise ValueError("Inconsistent fields provided to RingArray.build")
                ring = GroupRing(TrivialGroup(), next(iter(fields), None))
        ring = ring if isinstance(ring, Ring) else GroupRing(ring)
        one = ring.one
        if isinstance(array, galois.FieldArray):
            # check all field elements at once, after which they are in the field of the ring
            array = ring._read_coefficients(array)

        def as_ring_member(
            value: RingMember | GroupMember | float | np.integer | np.bool_ | galois.FieldArray,
        ) -> RingMember:
            """Elevate a value to an element of the ring."""
            if isinstance(value, RingMember):
                # validate the field of the member
                if value.field is not ring.field:
                    raise ValueError(
                        f"Cannot embed a ring member over GF({value.field.order}) into a ring over"
                        f" GF({ring.field.order}): incompatible coefficient fields"
                    )

                # re-home a member of a group algebra into the target group algebra
                if isinstance(ring, GroupRing) and isinstance(value.ring, GroupRing):
                    return RingMember(ring, *[(coeff, member) for coeff, member in value if coeff])

                # a member of any other ring must belong to the target ring
                ring._check_same_ring(value.ring)
                return value

            # ring arithmetic pads and validates a group member, and raises an error for rings other
            # than group algebras
            if isinstance(value, GroupMember):
                return one * value

            if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype == object:
                return as_ring_member(value.item())
            if isinstance(value, (list, tuple)) or (isinstance(value, np.ndarray) and value.ndim):
                raise ValueError(
                    "The entries of a RingArray must be scalars, group members, or ring members,"
                    f" not {value!r}"
                )

            # an integer, float, or field scalar is a multiple of the identity
            if isinstance(value, (int, np.integer, np.bool_)):
                return one * int(value)
            return one * ring._read_coefficients(value)

        vals = [as_ring_member(value) for value in array.ravel()]
        result = np.array(vals, dtype=object).reshape(array.shape).view(RingArray)
        result._ring = ring
        return result

    def to_field_array(self) -> galois.FieldArray:
        """Convert a RingArray into an array of coefficients (in a finite field) for each entry.

        This method expands every entry of a ``RingArray`` into a vector whose length is the
        dimension of its ring (``Ring.dim``).  For a two-dimensional ``RingArray`` ``ring_array``,
        for example, ``ring_array.to_field_array()[a, b, :]`` is the vector of coefficients for the
        ``RingMember`` at ``ring_array[a, b]``.
        """
        return ring_array_to_field_array(self, self.ring)

    @classmethod
    def from_field_array(cls, array: npt.ArrayLike, ring: Ring | Group) -> RingArray:
        """Construct a RingArray from an array of coefficients (in a finite field) for each entry.

        This method is the inverse of ``RingArray.to_field_array``.  The coefficients are read as in
        ``RingMember.from_vector``.
        """
        if isinstance(array, (Ring, Group)):  # DEPRECATED: see qldpc._deprecations
            _deprecations.warn_deprecated(
                "Check argument order: it should be RingArray.from_field_array(array, ring)."
                "  The order (ring, array) is DEPRECATED and will raise an error in a future"
                " release!",
                stacklevel=2,
            )
            array, ring = ring, array
        ring = ring if isinstance(ring, Ring) else GroupRing(ring)

        coefficients = ring._read_coefficients(array)
        if coefficients.ndim == 0 or coefficients.shape[-1] != ring.dim:
            raise ValueError(
                f"The last axis of a RingArray coefficient array must have length {ring.dim}"
            )
        return ring_array_from_field_array(coefficients, ring)

    def to_field_vector(self) -> galois.FieldArray:
        """Convert RingArray into a flattened 1-D vector of coefficients for each RingMember."""
        return self.to_field_array().ravel().view(self.field)

    @classmethod
    def from_field_vector(cls, vector: npt.ArrayLike, ring: Ring | Group) -> RingArray:
        """Construct a 1-D RingArray from a vector of coefficients.

        This method is the inverse of ``RingArray.to_field_vector``.  The coefficients are read as
        in ``RingMember.from_vector``.  Raise a ValueError unless the vector is 1-D and its length
        is a multiple of the dimension of the ring.
        """
        if isinstance(vector, (Ring, Group)):  # DEPRECATED: see qldpc._deprecations
            _deprecations.warn_deprecated(
                "Check argument order: it should be RingArray.from_field_vector(vector, ring)."
                "  The order (ring, vector) is DEPRECATED and will raise an error in a future"
                " release!",
                stacklevel=2,
            )
            vector, ring = ring, vector
        ring = ring if isinstance(ring, Ring) else GroupRing(ring)
        vector = ring._read_coefficients(vector)
        if vector.ndim != 1:
            raise ValueError(
                "A RingArray coefficient vector must be 1-D, but the vector has shape"
                f" {vector.shape}"
            )
        if vector.size % ring.dim:
            raise ValueError(
                f"The length of a RingArray coefficient vector must be a multiple of {ring.dim}"
            )
        entries_as_vecs = vector.reshape(vector.size // ring.dim, ring.dim)
        return RingArray.from_field_array(entries_as_vecs, ring)

    def null_space(self, *, right: bool = False) -> RingArray:
        """Construct a matrix of null-space row vectors for this RingArray.

        The conjugate transpose (``.T``) of the null-space matrix is annihilated by this
        ``RingArray``, such that ``np.any(self @ self.null_space().T)`` is ``np.False_``.

        If ``right=True``, this method constructs a null space over the opposite ring, in which the
        order of multiplication is reversed.  Then
        ``abstract.matmul(self, self.null_space(right=True).T, right=True)`` is zero.

        This method returns the conjugate transpose of ``RingArray.kernel``, so it requires a ring
        with an involution (see ``Ring.has_involution``), and raises a ValueError for any other
        ring.
        """
        # check the ring here, to point to RingArray.kernel for a ring without an involution
        if not self.ring.has_involution:
            raise ValueError(
                f"{self.ring} has no involution, so null-space rows cannot be conjugated: use"
                " RingArray.kernel for null-space columns"
            )
        return self.kernel(right=right).T

    def kernel(self, *, right: bool = False) -> RingArray:
        """Construct a matrix of null-space column vectors for this RingArray.

        This ``RingArray`` annihilates the null-space matrix, such that ``self @ self.kernel()`` is
        zero.  If ``right=True``, this method constructs a null space over the opposite ring, in
        which the order of multiplication is reversed.  Then
        ``abstract.matmul(self, self.kernel(right=True), right=True)`` is zero.

        The columns of the returned matrix form a basis of the null space of this ``RingArray`` over
        the base field, so they span it over the ring.  As generators over the ring, they are
        generally redundant.
        """
        self._require_ndim(2)

        # field-valued null vectors of the regular lift span the space of ring-valued null vectors
        null_field_vectors = self.regular_lift(right=right).null_space()

        # collect ring-valued null vectors as rows, and transpose them into columns without applying
        # an involution of the ring
        field_array_shape = (len(null_field_vectors), self.shape[1], self.ring.dim)
        null_rows = RingArray.from_field_array(
            null_field_vectors.reshape(field_array_shape), self.ring
        )
        return null_rows.transpose()

    def row_reduce(self, transformer: WedderburnArtinTransformer | None = None) -> RingArray:
        """Compute a generalized reduced row echelon form of a RingArray over a semisimple ring.

        This method relies on the Wedderburn-Artin decomposition:

        1. Decompose the matrix over a ring into matrices over simple components.
        2. Put the matrices over simple components into reduced row echelon form (RREF).
        3. Re-combine the simple components into a matrix over the original ring.

        The RREF of a RingArray over a commutative ring is unique.  For noncommutative rings, the
        RREF is only unique up to a choice of matrix basis for simple components of the ring.

        This method is implemented only for group algebras, and raises a NotImplementedError for any
        other ring, and a ValueError for a group algebra that is not semisimple.  Without a
        ``transformer``, it uses the ring's transformer (see GroupRing.get_transformer).
        """
        self._require_ndim(2)
        if not isinstance(self.ring, GroupRing):
            raise NotImplementedError(
                f"Row reduction is implemented only for group algebras, not {self.ring}"
            )
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

        - ``RingArray.howell_normal_form_semisimple`` (if ``poly=False``, the default), or
        - ``RingArray.howell_normal_form_poly`` (if ``poly=True``).

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
        property: every element of the row span whose first ``j`` entries are zero is a linear
        combination of rows whose first ``j`` entries are zero.

        The Howell property is enforced as follows: if a row ``r`` has a pivot ``p`` with a
        nontrivial left annihilator ``α``, meaning::

              α != 0,
            α·p  = 0,
            α·r != 0,

        then the row ``r`` is replaced by ``(1-α)·r``, and the row ``α·r`` is added to the matrix.

        If ``right=True``, the Howell property is instead enforced for nontrivial right
        annihilators::

              α != 0,
            p·α  = 0,
            r·α != 0,

        for which the row ``r`` is replaced by ``r·(1-α)``, and the row ``r·α`` is added to the
        matrix.  The ordinary HNF and right-HNF are equal for a ``RingArray`` over a commutative
        ring.

        The HNF of a ``RingArray`` over a commutative ring is unique.  For noncommutative rings, the
        HNF is only unique up to a choice of matrix basis for simple components of the ring.

        This method is implemented only for group algebras, and raises a NotImplementedError for any
        other ring, and a ValueError for a group algebra that is not semisimple.  Without a
        ``transformer``, it uses the ring's transformer (see GroupRing.get_transformer).

        References:

        - https://en.wikipedia.org/wiki/Howell_normal_form
        - https://github.com/m-webster/XPFpackage/blob/570ea89/Examples/A.1_howell_matrix.ipynb
        """
        return get_howell_normal_form_semisimple(self, transformer, right=right)

    def howell_normal_form_poly(self) -> RingArray:
        """Compute a Howell normal form of a RingArray using polynomial division.

        If the base ring of a RingArray is the algebra of a ``CyclicGroup`` of order n, then it is
        the polynomial ring ``F[x]/(x^n - 1)``, so row reduction can use greatest common divisors
        and polynomial division.  This method raises a TypeError for any other ring.

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

    def _read_value(self, value: Any) -> Any:
        """Read a value to set as entries of this RingArray (see _read_entries)."""
        values = _read_entries(_as_object_array(value), self._ring)
        # numpy stores a 0-dimensional array as an entry, rather than its member
        return values[()] if values.ndim == 0 else values

    def _mark_as_hnf(self, *, transformer: WedderburnArtinTransformer, right: bool) -> Self:
        """Record that this array is a Howell normal form, as provenance for get_howell_dual."""
        self._hnf_provenance = (transformer, right, np.array(self, dtype=object))
        return self

    def _get_hnf_provenance(self) -> tuple[WedderburnArtinTransformer, bool, bool] | None:
        """The transformer and orientation that built this Howell normal form, if any.

        Return None if howell_normal_form_semisimple did not build this array.  Otherwise, return
        the transformer, the orientation, and whether every entry is the object that it recorded.
        """
        if self._hnf_provenance is None:
            return None
        transformer, right, entries = self._hnf_provenance
        unchanged = entries.shape == self.shape and all(
            aa is bb for aa, bb in zip(entries.flat, self.view(np.ndarray).flat)
        )
        return transformer, right, unchanged


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


def _read_operand(obj: object, ring: Ring | None) -> object:
    """Replace RingArrays by plain arrays, and field arrays by plain arrays of ring members.

    Lists and tuples are read item by item.  Without a ring, field arrays stay as they are.
    """
    if isinstance(obj, RingArray):
        return obj.view(np.ndarray)
    if ring is not None and isinstance(obj, galois.FieldArray):
        return RingArray.build(obj, ring).view(np.ndarray)
    if isinstance(obj, (list, tuple)):
        return type(obj)(_read_operand(item, ring) for item in obj)
    return obj


def _wrap_result(result: object, ring: Ring | None) -> object:
    """Wrap an array of objects that numpy returns as a RingArray over a ring (see _read_entries).

    Other results are returned as they are.
    """
    if not isinstance(result, np.ndarray) or result.dtype != object:
        return result
    array = _read_entries(result, ring).view(RingArray)
    array._ring = ring
    return array


def _read_in_place(array: npt.NDArray[np.object_], ring: Ring | None) -> RingArray:
    """Read the entries that numpy wrote into an array of objects, in place (see _read_entries).

    Return a RingArray itself, and a plain array as a RingArray view over the ring.
    """
    values = array.view(np.ndarray)
    values[...] = _read_entries(values, ring)
    if isinstance(array, RingArray):
        return array
    result = values.view(RingArray)
    result._ring = ring
    return result


def _read_entries(values: npt.NDArray[np.object_], ring: Ring | None) -> npt.NDArray[np.object_]:
    """Read the entries of an array of objects as members of a ring, as in RingArray.build.

    Members of the ring and None stay as they are, and members of other rings raise a ValueError.
    Return the array itself if no entry needs reading, and otherwise a copy with its entries read,
    so that arrays that numpy returns as views of other arrays leave those arrays unchanged.
    """
    if ring is None:
        return values
    # ring equality compares groups, so check each ring only once
    checked_rings = {id(ring)}
    unread = np.zeros(values.shape, dtype=bool)
    for index, value in np.ndenumerate(values):
        if isinstance(value, RingMember):
            if id(value.ring) not in checked_rings:
                _check_entry_ring(ring, value.ring)
                checked_rings.add(id(value.ring))
        else:
            unread[index] = value is not None
    if not unread.any():
        return values
    values = values.copy()
    values[unread] = RingArray.build(values[unread], ring).view(np.ndarray)
    return values


def _check_entry_ring(ring: Ring, other: Ring) -> None:
    """Raise a ValueError unless the ring of a value equals the ring of a RingArray.

    If the value is a member of another group algebra over the same field, the error points to
    RingArray.build.
    """
    try:
        ring._check_same_ring(other)
    except ValueError as error:
        if (
            isinstance(ring, GroupRing)
            and isinstance(other, GroupRing)
            and other.field is ring.field
        ):
            raise ValueError(
                f"{error}; embed members of another group algebra explicitly with RingArray.build"
            ) from None
        raise


if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
