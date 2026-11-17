# SPDX-License-Identifier: Apache-2.0

"""Module for linear algebra with matrices over rings and bimodules.

.. warning::
    This module does not promise to be performant.  If you need to do heavy numerical abstract
    algebra, you're probably better served by GAP or MAGMA (or maybe SageMath).
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import galois
import numpy as np
import numpy.typing as npt
import scipy.linalg

from qldpc import math

from .groups import Group, GroupMember
from .ring_array import RingArray
from .rings import GroupRing, Ring, RingMember

if TYPE_CHECKING:
    from .wedderburn_artin import WedderburnArtinComponentTransformer, WedderburnArtinTransformer


####################################################################################################
# matrices over rings


def matmul(
    matrix_a: RingArray | npt.NDArray[np.int_],
    matrix_b: RingArray | npt.NDArray[np.int_],
    *,
    right: bool = False,
) -> RingArray:
    """Multiply two matrices over a ring.

    By default, this function computes ``matrix_a @ matrix_b``.  If ``right=True``, it reverses the
    order of multiplication in the ring, so that entry ``[i, k]`` of the product is the sum over j
    of ``matrix_b[j, k] * matrix_a[i, j]``.  As in numpy, leading axes index stacks of matrices.
    """
    _get_ring(matrix_a, matrix_b)  # check that the matrices are over one ring
    if matrix_a.ndim < 2 or matrix_b.ndim < 2 or matrix_a.shape[-1] != matrix_b.shape[-2]:
        raise ValueError(
            f"Incompatible matrix shapes for abstract.matmul: {matrix_a.shape}, {matrix_b.shape}"
        )

    if not right:
        return (matrix_a @ matrix_b).view(RingArray)

    # transposing the matrices (but not their entries) reverses the order of products of entries
    product = np.swapaxes(matrix_b, -1, -2) @ np.swapaxes(matrix_a, -1, -2)
    return np.swapaxes(product, -1, -2).view(RingArray)


def kron(
    matrix_a: RingArray | npt.NDArray[np.int_], matrix_b: RingArray | npt.NDArray[np.int_]
) -> RingArray:
    """Take the Kronecker product of two matrices over a ring.

    If the base ring R is commutative, this is the ordinary Kronecker product.  Otherwise, this
    function is implemented only for a group algebra R = F[G], and raises a NotImplementedError for
    any other noncommutative ring.  Entries of the product then live in the bimodule
    F[G × G] ≅ R ⨂ R.  A basis element (g, h) of the bimodule, for group members g and h, acts on
    t ∈ R as (g, h)(t) = g·t·h⁻¹.  If entries of matrix_a and matrix_b are group members a and b,
    the corresponding entry of the product is the basis element (a, b⁻¹), which acts as t -> a·t·b.
    The product extends bilinearly to other entries.  Lifting the product needs anti-representations
    of G, which a noncommutative group has only with its default lift.
    """
    ring = _get_ring(matrix_a, matrix_b)

    if not ring.is_commutative:
        if not isinstance(ring, GroupRing):
            # An entry a ⨂ b of the product acts on t ∈ R as a·t·b.  Lifting it needs commuting left
            # and right actions of R, such as its left and right regular representations, which a
            # group algebra has with its default lift.  A MatrixRing's lift and its lift with
            # right=True need not commute.  For the checks of a lifted product to commute, its
            # involution would also need to transpose these lifts, which regular representations in
            # the basis of a MatrixRing need not do.
            raise NotImplementedError(
                "The Kronecker product over a noncommutative ring is implemented only for group"
                f" algebras, not {ring}"
            )
        # keep left and right multiplication in separate halves of the bimodule representation
        bimodule = _get_bimodule(ring)
        matrix_a = RingArray.build(matrix_a, bimodule)
        matrix_b = _swap_sectors(~RingArray.build(matrix_b, bimodule))

    return np.kron(matrix_a, matrix_b).view(RingArray)


def block_diag(
    matrix_a: RingArray | npt.NDArray[np.int_], matrix_b: RingArray | npt.NDArray[np.int_]
) -> RingArray:
    """Stack the two matrices into a block-diagonal matrix."""
    ring = _get_ring(matrix_a, matrix_b)
    matrix_ab = scipy.linalg.block_diag(matrix_a, matrix_b)
    return RingArray.build(matrix_ab, ring)


def get_howell_dual(
    matrix_hnf: RingArray,
    *,
    transformer: WedderburnArtinTransformer | None = None,
    right: bool | None = None,
    skip_validation: bool | None = None,
) -> RingArray:
    """Build the "dual" of a matrix in Howell normal form.

    The dual matrix provides a pseudoinverse of matrix_hnf in the following sense.  Below, ``@``
    denotes the product ``abstract.matmul(..., right=right)``.  If

        ``D = matrix_hnf @ dual_matrix.T``,

    then

    1. D is diagonal, with nonzero diagonal entries,
    2. ``D @ matrix_hnf = matrix_hnf``,
    3. ``D.T @ dual_matrix = dual_matrix``, and
    4. ``D = transformer.transpose_array(D)``.

    Note that we have two different notions of a transpose at play:

    1. ``D.T``...

        (a) swaps the matrix indices of D, and
        (b) takes group members ``g -> ~g = g**-1``, which transposes their regular representation.

    2. ``transformer.transpose_array(D)``...

        (a) swaps the matrix indices of D (identically to ``D.T``), and
        (b) for each entry of D, transposes its matrix representation within each Wedderburn-Artin
            component of the ring.

    Howell duals are implemented only for group algebras, so this function raises a
    NotImplementedError for any other ring.

    Over a commutative ring, the dual needs no transformer.  Over a noncommutative ring, the dual is
    built in the basis of a Wedderburn-Artin ``transformer``, which must be the one used to build
    ``matrix_hnf``.  Likewise, ``right`` must match the orientation used to build ``matrix_hnf``,
    since the dual for the other orientation need not satisfy properties 1-4.  By default, both are
    read from the provenance that ``howell_normal_form_semisimple`` records on its output.  Copies,
    slices, transposes, and arithmetic results have no provenance.  Without provenance, the default
    orientation is ``right=False``, and the default transformer is that of the ring (see
    GroupRing.get_transformer).

    ``matrix_hnf`` should be a Howell normal form with idempotent pivots, such as the output of
    ``howell_normal_form_semisimple``.  Over a commutative ring that is not semisimple, its pivots
    are 1.  The pivots of other matrices, such as polynomial Howell forms, need not be idempotent,
    so their duals generally fail properties 1-4.

    ``skip_validation`` controls whether the returned dual is checked.  Validation checks properties
    1 and 2.  The constructed dual satisfies properties 3 and 4 whenever it satisfies properties 1
    and 2.

    - ``False``: always check, raising a ValueError if the dual is not valid.
    - ``True``: never check.  A wrong ``transformer`` or ``right`` may then give a wrong dual.
    - ``None`` (default): trust the output of ``howell_normal_form_semisimple`` until its entries
      or shape change, if it is used with its recorded transformer.  For such an output over a
      noncommutative ring, raise a ValueError if ``right`` disagrees with its recorded orientation,
      whatever the transformer.  Check any other ``matrix_hnf``, as for ``False``.
    """
    matrix_hnf._require_ndim(2)
    ring = matrix_hnf.ring
    if not isinstance(ring, GroupRing):
        raise NotImplementedError(
            f"Howell duals are implemented only for group algebras, not {ring}"
        )

    recorded_transformer, recorded_right, unchanged = matrix_hnf._get_hnf_provenance() or (
        None,
        False,
        False,
    )
    if skip_validation is None:
        if unchanged and right is not None and not ring.is_commutative and right != recorded_right:
            raise ValueError(
                "get_howell_dual: `right` does not match the orientation used to build"
                f" matrix_hnf (built with right={recorded_right}, called with right={right})."
            )
        skip_validation = unchanged and (transformer is None or transformer is recorded_transformer)
    if right is None:
        right = recorded_right

    dual = RingArray.build(np.zeros(matrix_hnf.shape, dtype=int), ring)
    if ring.is_commutative:
        # Every simple component of a commutative ring is 1x1, and each idempotent pivot is the
        # projector onto the components where its row is nonzero.  So the construction for
        # noncommutative rings reduces to the transpose g -> g**-1 of each pivot.
        for row, col in enumerate(math.first_nonzero_cols(matrix_hnf)):
            if col < matrix_hnf.shape[1]:
                dual[row, col] = matrix_hnf[row, col].T
    else:
        # In each Wedderburn-Artin component, place a symmetric idempotent that projects onto the
        # span of each row (see _get_fixing_projector).  Place it at the leading column of that row
        # in that component.
        transformer = transformer or recorded_transformer or ring.get_transformer()
        for component_transformer in transformer.transformers:
            blocks = component_transformer.project_array(matrix_hnf)
            for row in range(matrix_hnf.shape[0]):
                nonzero_cols = np.nonzero(np.any(blocks[row], axis=(-2, -1)))[0]
                if not len(nonzero_cols):
                    continue
                projector = _get_fixing_projector(blocks[row], component_transformer, right=right)
                col = int(nonzero_cols[0])
                dual[row, col] += component_transformer.embed(projector).T

    if not skip_validation:
        _validate_howell_dual(matrix_hnf, dual, right)
    return dual


# cached, because kron builds the bimodule of the same ring repeatedly, and building it is costly
@functools.lru_cache(maxsize=16)
def _get_bimodule(ring: GroupRing) -> GroupRing:
    """Map a group algebra ``F[G]`` to its induced bimodule ``F[G × G]``.

    A basis element ``(g, h) ∈ G × G`` of the bimodule acts on elements of the base ring as::

        (g, h)(t) = g·t·h⁻¹.

    When lifting a RingArray over the bimodule, the left factor of each member of ``G × G`` lifts to
    a representation, and the right factor lifts to the anti-representation of its inverse, hence
    ``h⁻¹``.
    """
    size = ring.group.identity.size

    def lift(member: GroupMember) -> npt.NDArray[np.int_]:
        part_l = GroupMember(member.array_form[:size])
        part_r = GroupMember([val - size for val in member.array_form[size:]])
        return ring.group.lift(part_l) @ ring.group.lift(~part_r, right=True)

    bimodule_group = Group(ring.group.to_sympy() * ring.group.to_sympy(), lift=lift)
    return GroupRing(bimodule_group, ring.field)


def _swap_sectors(matrix: RingArray) -> RingArray:
    """Swap the left and right parts of the group member in each term of a matrix over a bimodule.

    The bimodule group acts on two sectors of points, one for each copy of the base group.  The
    permutation that swaps these sectors is a symmetry of the bimodule group rather than one of its
    members, so ring arithmetic rejects it, and this function conjugates group members by it.
    """
    sector_size = matrix.group.degree // 2
    swap = GroupMember(list(range(sector_size, 2 * sector_size)) + list(range(sector_size)))
    swapped = np.empty(matrix.shape, dtype=object)
    for index, entry in np.ndenumerate(matrix):
        terms = [(value, swap * member * swap) for value, member in entry]
        swapped[index] = RingMember._from_trusted_terms(matrix.ring, terms)
    return RingArray(swapped, matrix.ring)


def _validate_howell_dual(matrix_hnf: RingArray, dual: RingArray, right: bool) -> None:
    """Raise a ValueError if ``dual`` is not a valid dual of matrix_hnf.

    This function checks properties 1 and 2 in the docstring of get_howell_dual.  For the dual that
    get_howell_dual builds, they imply properties 3 and 4.  Let ``D = matrix_hnf @ dual.T``.  In a
    Wedderburn-Artin component, let B be the first nonzero block of a row, S the span of the columns
    of the blocks of the row, and E the projector onto S from _get_fixing_projector, which is a
    symmetric matrix.  The diagonal block of D for the row is ``B·E``.  Properties 1 and 2 make
    ``B·E`` fix S.  E vanishes on the orthogonal complement of S, so ``B·E`` does too.  S and its
    complement span the space, since the Gram matrix in _get_fixing_projector is invertible, so
    ``B·E = E``.  So D is a diagonal of symmetric idempotents, which gives properties 3 and 4.  If
    ``right=True``, the same argument applies with rows instead of columns, and with ``E·B`` instead
    of ``B·E``.

    Over a commutative ring, the diagonal entry of D for a row with pivot p is ``p·p``.  Property 2
    gives ``p·p·p = p``, which gives property 3.  Property 4 follows from property 1, since
    transpose_array then only transposes the array.
    """
    diag = matmul(matrix_hnf, dual.T, right=right)
    valid = (
        np.array_equal(diag.astype(bool), np.eye(len(diag), dtype=bool))  # 1
        and np.array_equal(matmul(diag, matrix_hnf, right=right), matrix_hnf)  # 2
    )
    if not valid:
        raise ValueError(
            "get_howell_dual could not validate the constructed dual: matrix_hnf may not be a"
            " Howell normal form with idempotent pivots, or `transformer`/`right` may not match those"
            " used to build it."
        )


def _get_fixing_projector(
    row_blocks: galois.FieldArray,
    component_transformer: WedderburnArtinComponentTransformer,
    *,
    right: bool,
) -> galois.FieldArray:
    """Build the projector onto the space spanned by one row's projected blocks.

    ``row_blocks`` is ``component_transformer.project_array(matrix_hnf)[row]``: the row's entries
    projected into one component of the ring and stacked, with shape ``(num_cols, size, size)``.

    The result E is a ``size x size`` matrix that leaves that space unchanged: for ``right=True`` it
    fixes the rows of the blocks (``v @ E == v``), and for ``right=False`` their columns
    (``E @ w == w``).  E is symmetric and ``E @ E == E``.  The dual places it as ``embed(E).T``.
    """
    field = component_transformer.extended_field
    size = component_transformer.size
    vectors = row_blocks if right else row_blocks.transpose(0, 2, 1)
    # `basis` holds independent vectors spanning that space.  It is never empty: the caller only
    # gets here when the row is nonzero in this component.  When `transformer` and `right` match
    # those used to build matrix_hnf, these are standard basis vectors, so `gram` is the identity
    # and inverting it does nothing.
    basis = vectors.reshape(-1, size).view(field).row_space().view(field)
    gram = (basis @ basis.T).view(field)
    try:
        gram_inv = np.linalg.inv(gram).view(field)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Cannot build a Howell dual: a row's projected blocks span a space with no projector"
            " (basis @ basis.T is not invertible).  This is not expected for a valid Howell normal"
            " form built with a matching `transformer` and `right`."
        ) from error
    return (basis.T @ gram_inv @ basis).view(field)


def _get_ring(
    matrix_a: RingArray | npt.NDArray[np.int_], matrix_b: RingArray | npt.NDArray[np.int_]
) -> Ring:
    """Identify the ring that at least one of the matrices is over."""
    ring_a = matrix_a._ring if isinstance(matrix_a, RingArray) else None
    ring_b = matrix_b._ring if isinstance(matrix_b, RingArray) else None
    if ring_a is None:
        if ring_b is None:
            raise ValueError("At least one of the provided matrices must be a RingArray")
        return ring_b
    if ring_b is not None and ring_a != ring_b:
        raise ValueError("The provided matrices are over different rings")
    return ring_a
