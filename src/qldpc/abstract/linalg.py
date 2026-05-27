"""Module for linear algebra with matrices over rings and bimodules

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

import numpy as np
import numpy.typing as npt
import scipy.linalg

from qldpc import math

from .rings import Group, GroupMember, GroupRing, RingArray

if TYPE_CHECKING:
    from .wedderburn_artin import WedderburnArtinTransformer


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

    If the base ring R is commutative, this is the ordinary Kronecker product.
    Otherwise, matrix entries of the Kronecker product live in the bimodule of R.
    See get_bimodule for additional information.
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
    """Map a group algebra F[G] to its induced bimodule F[G ⨂ G].

    Elements r ⨂ s ∈ F[G ⨂ G] of the bimodule act on elements of the base ring as
        (r ⨂ s)(t) = r·t·s.
    When lifting a RingArray matrix over a bimodule, the "left" entries of G ⨂ G get lifted to an
    ordinary representation, while the "right" entries get lifted to an anti-representation.
    """
    size = ring.group.identity.size

    def lift(member: GroupMember) -> npt.NDArray[np.int_]:
        part_l = GroupMember(member.array_form[:size])
        part_r = GroupMember([val - size for val in member.array_form[size:]])
        return ring.group.lift(part_l) @ ring.group.lift(~part_r, right=True)

    bimodule_group = Group(ring.group.to_sympy() * ring.group.to_sympy(), lift=lift)
    return GroupRing(bimodule_group, ring.field.order)


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
) -> RingArray:
    """Build the "dual" of a matrix in Howell normal form.

    The dual matrix provides a pseudoinverse of matrix_hnf in the following sense: if
        D = matrix_hnf @ dual_matrix.T,
    then
    1. D is diagonal,
    2. D @ matrix_hnf = matrix_hnf,
    3. D.T @ dual_matrix = dual_matrix, and
    4. D = transformer.transpose_array(D).

    Note that we have two different notions of a transpose at play:
    1. D.T...
        (a) swaps the matrix indices of D, and
        (b) takes group members g -> ~g = g**-1, which transposes their regular representation.
    2. transformer.transpose_array(D)...
        (a) swaps the matrix indices of D (identically to D.T), and
        (b) for each entry of D, transposes its matrix representation within each Wedderburn-Artin
            component of the ring.

    WARNING: This method assumes--and does not verify--that matrix_hnf is in Howell normal form.
    """
    ring = matrix_hnf.ring
    transformer = transformer = transformer or ring.get_transformer()
    dual_matrix = np.zeros(matrix_hnf.shape, dtype=object)
    for row, col in enumerate(math.first_nonzero_cols(matrix_hnf)):
        pivot = matrix_hnf[row, col].copy()
        if ring.is_commutative:
            new_matrix_entry = pivot
        else:
            new_matrix_entry = ring.zero
            for component_transformer in transformer.transformers:
                if np.any(component := component_transformer.project(pivot)):
                    field = component_transformer.extended_field
                    diags = np.diag(np.diag(component)).view(field)
                    new_matrix_entry += component_transformer.embed(diags).T
        dual_matrix[row, col] = new_matrix_entry
    return RingArray.build(dual_matrix, ring)


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
