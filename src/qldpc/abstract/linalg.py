"""Module for linear algebra with matrices over rings and bimodules.

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

import functools
from typing import TYPE_CHECKING

import galois
import numpy as np
import numpy.typing as npt
import scipy.linalg

from qldpc import math

from .rings import Group, GroupMember, GroupRing, RingArray

if TYPE_CHECKING:
    from .wedderburn_artin import WedderburnArtinComponentTransformer, WedderburnArtinTransformer


def matmul(
    matrix_a: RingArray | npt.NDArray[np.int_],
    matrix_b: RingArray | npt.NDArray[np.int_],
    *,
    right: bool = False,
) -> RingArray:
    """Multiply two matrices over a ring.

    This method exists to handle two different cases:
    1. 'right is False' (default): Ordinary matrix multiplication, or simply 'matrix_a @ matrix_b'.
    2. 'right is True': matrix multiplication with a reversed order of multiplication in the ring.
    """
    ring = _get_ring(matrix_a, matrix_b)
    if (
        matrix_a.ndim != matrix_b.ndim
        or matrix_a.ndim < 2
        or (inner_dim := matrix_a.shape[-1]) != matrix_b.shape[-2]
    ):
        raise ValueError(
            f"Incompatible matrix shapes for abstract.matmul: {matrix_a.shape}, {matrix_b.shape}"
        )

    if ring.is_commutative or not right:
        return (matrix_a @ matrix_b).view(RingArray)

    final_shape = (*matrix_a.shape[:-1], matrix_b.shape[-1])
    matrix = RingArray.build(np.zeros(final_shape, dtype=int), ring)
    for idx in np.ndindex(final_shape):
        head, ii, jj = idx[:-2], idx[-2], idx[-1]
        matrix[idx] = sum(
            matrix_b[(*head, kk, jj)] * matrix_a[(*head, ii, kk)] for kk in range(inner_dim)
        )
    return matrix


def kron(
    matrix_a: RingArray | npt.NDArray[np.int_], matrix_b: RingArray | npt.NDArray[np.int_]
) -> RingArray:
    """Take the Kronecker product of two matrices over a ring.

    If the base ring R is commutative, this is the ordinary Kronecker product. Otherwise, matrix
    entries of the Kronecker product live in the bimodule of R. See get_bimodule for additional
    information.
    """
    ring = _get_ring(matrix_a, matrix_b)

    if not ring.is_commutative:
        bimodule = get_bimodule(ring)
        sector_size = ring.group.identity.size
        swap_sectors = GroupMember(
            list(range(sector_size, 2 * sector_size)) + list(range(sector_size))
        )
        matrix_a = RingArray.build(matrix_a, bimodule)
        matrix_b = swap_sectors * ~RingArray.build(matrix_b, bimodule) * swap_sectors

    return np.kron(matrix_a, matrix_b).view(RingArray)


@functools.cache
def get_bimodule(ring: GroupRing) -> GroupRing:
    """Map a group algebra ``F[G]`` to its induced bimodule ``F[G ⨂ G]``.

    Elements ``r ⨂ s ∈ F[G ⨂ G]`` of the bimodule act on elements of the base ring as::

        (r ⨂ s)(t) = r·t·s⁻¹.

    (The right sector lifts to an anti-representation of the inverse, hence ``s⁻¹`` rather than
    ``s``.)

    When lifting a RingArray matrix over a bimodule, the "left" entries of ``G ⨂ G`` get lifted to
    an ordinary representation, while the "right" entries get lifted to an anti-representation.
    """
    size = ring.group.identity.size

    def lift(member: GroupMember) -> npt.NDArray[np.int_]:
        part_l = GroupMember(member.array_form[:size])
        part_r = GroupMember([val - size for val in member.array_form[size:]])
        return ring.group.lift(part_l) @ ring.group.lift(~part_r, right=True)

    bimodule_group = Group(ring.group.to_sympy() * ring.group.to_sympy(), lift=lift)
    return GroupRing(bimodule_group, ring.field)


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
    right: bool = False,
    skip_validation: bool | None = None,
) -> RingArray:
    """Build the "dual" of a matrix in Howell normal form.

    The dual matrix provides a pseudoinverse of matrix_hnf in the following sense.  Below, ``@``
    denotes the orientation-aware product ``abstract.matmul(..., right=right)`` (ordinary matrix
    multiplication when ``right=False``).  If

        ``D = matrix_hnf @ dual_matrix.T``,

    then

    1. D is diagonal,
    2. ``D @ matrix_hnf = matrix_hnf``,
    3. ``D.T @ dual_matrix = dual_matrix``, and
    4. ``D = transformer.transpose_array(D)``.

    Note that we have two different notions of a transpose at play:

    1. D.T...

        (a) swaps the matrix indices of D, and
        (b) takes group members ``g -> ~g = g**-1``, which transposes their regular representation.

    2. transformer.transpose_array(D)...

        (a) swaps the matrix indices of D (identically to D.T), and
        (b) for each entry of D, transposes its matrix representation within each Wedderburn-Artin
            component of the ring.

    The ``transformer`` must be the one used to build matrix_hnf, since the dual is built in that
    transformer's basis.  By default it is read from matrix_hnf (the transformer recorded when the
    Howell form was built).

    Likewise, the ``right`` flag must match the orientation used to build matrix_hnf (the ``right``
    passed to ``null_space``/``howell_normal_form_semisimple``).  For a non-commutative ring the
    dual is orientation-specific: the dual built for the wrong orientation does not satisfy it.

    ``skip_validation`` controls whether the returned dual is checked against properties 1-4:

    - ``False``: always check, raising an error if the dual is not valid.
    - ``True``: never check (fastest; trusts that matrix_hnf is a Howell form and that ``right``
      matches its orientation).
    - ``None`` (default): if matrix_hnf carries Howell-form provenance (attached by
      ``howell_normal_form_semisimple``/``howell_normal_form_poly``), trust it -- but raise an error
      if its recorded orientation disagrees with ``right``.  Otherwise, check as for ``False``.

    WARNINGS: Any ``transformer`` passed in must be the one used to build matrix_hnf; otherwise the
    dual is built in the wrong basis and may be wrong.  By default the transformer recorded on
    matrix_hnf is used.  Separately, with ``skip_validation=True`` this method verifies nothing
    else: it trusts that matrix_hnf is in Howell normal form and that ``right`` matches the
    orientation used to build it, so a wrong ``right`` may raise an error but may also silently
    return a wrong dual.
    """
    ring = matrix_hnf.ring
    # prefer the transformer matrix_hnf was built with, so we build (and validate) in the same basis
    transformer = transformer or matrix_hnf._hnf_transformer or ring.get_transformer()

    if skip_validation is None:
        # trust Howell-form provenance when present, but reject an orientation that disagrees
        if matrix_hnf._in_hnf:
            if not ring.is_commutative and matrix_hnf._hnf_right != right:
                raise ValueError(
                    "get_howell_dual: `right` does not match the orientation used to build"
                    f" matrix_hnf (built with right={matrix_hnf._hnf_right}, called with"
                    f" right={right})."
                )
            skip_validation = True
        else:
            skip_validation = False

    if ring.is_commutative:
        # Every simple component of a commutative ring is 1x1, so the construction below reduces to
        # the pivot's group transpose g -> g**-1.
        dual_matrix = np.zeros(matrix_hnf.shape, dtype=object)
        for row, col in enumerate(math.first_nonzero_cols(matrix_hnf)):
            dual_matrix[row, col] = matrix_hnf[row, col].copy().T
        dual = RingArray.build(dual_matrix, ring)
    else:
        # For a non-commutative ring, the dual entry that "fixes" a row within a Wedderburn
        # component belongs at that component's own leading column, which on an augmented Howell
        # form can lie to the right of (and differ from) the ring-level pivot column.  In each
        # component we place a symmetric idempotent projecting onto the row's span (see
        # _get_fixing_projector).
        dual_matrix = np.empty(matrix_hnf.shape, dtype=object)
        for index in np.ndindex(matrix_hnf.shape):
            dual_matrix[index] = ring.zero
        for component_transformer in transformer.transformers:
            blocks = component_transformer.project_array(matrix_hnf)
            for row in range(matrix_hnf.shape[0]):
                nonzero_cols = np.nonzero(np.any(blocks[row], axis=(-2, -1)))[0]
                if not len(nonzero_cols):
                    continue
                projector = _get_fixing_projector(blocks[row], component_transformer, right=right)
                col = int(nonzero_cols[0])
                embedded = component_transformer.embed(projector).T
                dual_matrix[row, col] = dual_matrix[row, col] + embedded
        dual = RingArray.build(dual_matrix, ring)

    if not skip_validation:
        _validate_howell_dual(matrix_hnf, dual, transformer, right)
    return dual


def _validate_howell_dual(
    matrix_hnf: RingArray,
    dual: RingArray,
    transformer: WedderburnArtinTransformer,
    right: bool,
) -> None:
    """Raise an error if ``dual`` is not a valid dual of matrix_hnf (properties 1-4 above)."""
    diag = matmul(matrix_hnf, dual.T, right=right)
    valid = (
        np.array_equal(diag.astype(bool), np.eye(len(diag), dtype=bool))  # 1: diagonal
        and np.array_equal(matmul(diag, matrix_hnf, right=right), matrix_hnf)  # 2
        and np.array_equal(matmul(diag.T, dual, right=right), dual)  # 3
        and np.array_equal(diag, transformer.transpose_array(diag))  # 4
    )
    if not valid:
        raise ValueError(
            "get_howell_dual could not validate the constructed dual: matrix_hnf may not be in"
            " Howell normal form, or `transformer`/`right` may not match those used to build it."
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
    # gets here when the row is nonzero in this component.  When `right` matches the orientation
    # used to build matrix_hnf, these are just standard basis vectors, so `gram` is the identity and
    # inverting it does nothing.  A wrong `right` is a caller mistake: it may land on a `gram` that
    # cannot be inverted (raising below) or, worse, one that can (returning a wrong dual) -- so
    # callers must pass the same `right` used to build matrix_hnf.
    basis = vectors.reshape(-1, size).view(field).row_space().view(field)
    gram = (basis @ basis.T).view(field)
    try:
        gram_inv = np.linalg.inv(gram).view(field)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "Cannot build a Howell dual: a row's projected blocks span a space with no projector"
            " (basis @ basis.T is not invertible).  This is not expected for a valid Howell normal"
            " form built with a matching `right`."
        ) from error
    return (basis.T @ gram_inv @ basis).view(field)


def _get_ring(
    matrix_a: RingArray | npt.NDArray[np.int_], matrix_b: RingArray | npt.NDArray[np.int_]
) -> GroupRing:
    """Identify the ring that at least one of the matrices is over."""
    ring_a = matrix_a.ring if isinstance(matrix_a, RingArray) else None
    ring_b = matrix_b.ring if isinstance(matrix_b, RingArray) else None
    if ring_a is None and ring_b is None:
        raise ValueError("At least one of the provided matrices must be a RingArray")
    if ring_a is not None and ring_b is not None and ring_a != ring_b:
        raise ValueError("The provided matrices are over different rings")
    return ring_a if ring_a is not None else ring_b  # type:ignore[return-value]
