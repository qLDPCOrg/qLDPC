"""Unit tests for linalg.py

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

import numpy as np
import pytest

from qldpc import abstract


def test_basics(rows: int = 2, cols: int = 3) -> None:
    """Basic linear algebra over commutative rings."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    values_a = [[ring.group.random() for _ in range(cols)] for _ in range(rows)]
    values_b = [[ring.group.random() for _ in range(cols)] for _ in range(rows)]
    matrix_a = abstract.RingArray.build(values_a, ring)
    matrix_b = abstract.RingArray.build(values_b, ring)

    # matrix multiplication and the Kronecker product work as expected
    assert np.array_equal(
        abstract.matmul(matrix_a, matrix_b.T),
        matrix_a @ matrix_b.T,
    )
    assert np.array_equal(
        abstract.kron(matrix_a, matrix_b),
        np.kron(matrix_a, matrix_b).view(abstract.RingArray),
    )

    # block_diag works as expected
    matrix_ab = abstract.block_diag(matrix_a, matrix_b)
    assert np.array_equal(matrix_ab[: matrix_a.shape[0], : matrix_a.shape[1]], matrix_a)
    assert np.array_equal(matrix_ab[matrix_a.shape[0] :, matrix_a.shape[1] :], matrix_b)
    assert not np.any(matrix_ab[: matrix_a.shape[0], matrix_a.shape[1] :])
    assert not np.any(matrix_ab[matrix_a.shape[0] :, : matrix_a.shape[1]])

    # we can still use these methods if only one of them is a RingArray
    assert np.array_equal(abstract.matmul(np.eye(matrix_a.shape[0], dtype=int), matrix_a), matrix_a)
    assert np.array_equal(abstract.matmul(matrix_a, np.eye(matrix_a.shape[1], dtype=int)), matrix_a)

    # ... but we need at least one of them to be a RingArray
    with pytest.raises(ValueError, match="At least one .* RingArray"):
        abstract.kron(np.eye(1, dtype=int), np.eye(1, dtype=int))

    # matrix dimensions must be compatible for matrix multiplication
    with pytest.raises(ValueError, match="Incompatible matrix shapes"):
        abstract.matmul(matrix_a, matrix_b)

    # we can't multiply matrices over different rings
    other_ring = abstract.GroupRing(ring.group, field=ring.field.order + 1)
    other_matrix = abstract.RingArray.build(np.eye(matrix_a.shape[0], dtype=int), other_ring)
    with pytest.raises(ValueError, match="different rings"):
        abstract.matmul(other_matrix, matrix_a)


def test_matmul_and_kron_interplay(ring: abstract.GroupRing, rows: int = 2, cols: int = 3) -> None:
    """Matrix multiplication and the Kronecker product work together as expected."""
    values_a = [[ring.group.random() for _ in range(cols)] for _ in range(rows)]
    values_b = [[ring.group.random() for _ in range(cols)] for _ in range(rows)]
    values_c = [[ring.group.random() for _ in range(cols)] for _ in range(rows)]
    values_d = [[ring.group.random() for _ in range(cols)] for _ in range(rows)]
    matrix_a = abstract.RingArray.build(values_a, ring)
    matrix_b = abstract.RingArray.build(values_b, ring)
    matrix_c = abstract.RingArray.build(values_c, ring).transpose()
    matrix_d = abstract.RingArray.build(values_d, ring).transpose()

    # (A ⨂ B) @ (C ⨂ D)
    matrix_ab = abstract.kron(matrix_a, matrix_b)
    matrix_cd = abstract.kron(matrix_c, matrix_d)
    matrix_ab_cd = abstract.matmul(matrix_ab, matrix_cd)

    # build (A @ C) ⨂ (B @ D), with reversed order of ring multiplication in second tensor factor
    matrix_ac = abstract.matmul(matrix_a, matrix_c)
    matrix_bd = abstract.matmul(matrix_b, matrix_d, right=True)
    matrix_ac_bd = abstract.kron(matrix_ac, matrix_bd)
    assert np.array_equal(matrix_ab_cd, matrix_ac_bd)

    if not ring.is_commutative:
        # assert that elements of the bimodule R ⨂ R transform elements of the ring R correctly
        val_a = matrix_a[0, 0]
        val_b = matrix_b[0, 0]
        val_ab = matrix_ab[0, 0]
        random_val = abstract.RingMember.from_vector(ring.field.Random(ring.group.order), ring)
        random_vec = random_val.to_vector()
        assert np.array_equal(
            val_ab.lift() @ random_vec,
            val_a.lift() @ (val_b.lift(right=True) @ random_vec),
        )


@pytest.mark.parametrize("right", [False, True])
def test_howell_dual(ring: abstract.GroupRing, right: bool, rows: int = 2, cols: int = 3) -> None:
    """Matrices in Howell normal form have a "dual" that acts like a pseudoinverse."""
    transformer = ring.get_transformer()
    values = [[ring.group.random() for _ in range(cols)] for _ in range(rows)]
    matrix = abstract.RingArray.build(values, ring)

    # find a null-space matrix in Howell normal form
    generator = matrix.null_space(right=right).howell_normal_form_semisimple(right=right)
    assert not np.any(abstract.matmul(matrix, generator.T, right=right))

    # find the "dual" of the null-space matrix, which acts like a pseudoinverse
    dual = abstract.get_howell_dual(generator)
    diag = abstract.matmul(generator, dual.T, right=right)
    assert np.array_equal(diag.astype(bool), np.eye(len(diag), dtype=bool))
    assert np.array_equal(abstract.matmul(diag, generator, right=right), generator)
    assert np.array_equal(abstract.matmul(diag.T, dual, right=right), dual)
    assert np.array_equal(diag, transformer.transpose_array(diag))
