"""Unit tests for math.py

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

import galois
import numpy as np
import pytest
import stim

import qldpc


def test_pauli_strings() -> None:
    """Conversion between symplectic vectors and stim.PauliString objects."""
    code = qldpc.codes.FiveQubitCode()
    for row, stabilizer in zip(code.matrix, code.get_strings()):
        string = qldpc.math.op_to_string(row)
        assert string == stim.PauliString(stabilizer.replace(" ", ""))
        assert np.array_equal(row, qldpc.math.string_to_op(string))

    string = stim.PauliString.random(5)
    sign = string.sign
    assert string == sign * qldpc.math.op_to_string(qldpc.math.string_to_op(string))


def test_vectors() -> None:
    """Methods that act on vectors."""
    vectors = np.array([[0, 1], [1, 2]], dtype=int)
    vectors_conj = np.array([[1, 0], [2, -1]], dtype=int)
    assert np.array_equal(qldpc.math.symplectic_weight(vectors), [1, 1])
    assert np.array_equal(qldpc.math.symplectic_conjugate(vectors), vectors_conj)
    assert np.array_equal(qldpc.math.first_nonzero_cols(vectors), [1, 0])
    assert np.array_equal(qldpc.math.first_nonzero_cols(vectors_conj), [0, 0])


def test_nonzero_cols() -> None:
    """Edge cases in finding the pivot columns."""
    empty_matrix = np.array([], ndmin=2, dtype=int)
    assert qldpc.math.first_nonzero_cols(empty_matrix).size == 0

    zero_matrix = np.zeros((1, 1), dtype=int)
    assert np.array_equal(qldpc.math.first_nonzero_cols(zero_matrix), [1])

    tensor = np.ones((1, 1, 1), dtype=int)
    assert np.array_equal(qldpc.math.first_nonzero_cols(tensor), [0])


def test_dual_basis() -> None:
    """Construct dual bases."""
    field = galois.GF(2)
    basis = field([[1, 1, 0, 0, 1], [0, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
    dual_basis = qldpc.math.get_dual_basis(basis)
    assert np.array_equal(dual_basis @ basis.T, field.Identity(len(basis)))

    with pytest.raises(ValueError, match="wide matrices of full rank"):
        qldpc.math.get_dual_basis(field([[1], [0]]))


def test_orthonormal_basis() -> None:
    """Orthonormal bases over finite fields (arXiv:2503.19790, Algorithm 1 and Lemma 2)."""
    # subspaces that admit an orthonormal basis: check that L @ L.T is the identity
    with_basis = [
        galois.GF(2).Zeros((0, 4)),  # the empty subspace
        galois.GF(2)([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1]]),  # Lemma 2
        galois.GF(4)([[2, 0]]),  # characteristic 2, a rescaled unit vector
        galois.GF(3)([[1, 0], [0, 1]]),  # odd characteristic, unit norms
        galois.GF(7)([[1, 1]]),  # odd characteristic, a rescaled square norm
        galois.GF(7)([[1, 2], [1, 3]]),  # odd characteristic, two non-square norms paired off
    ]
    # these inputs are full-rank bases crafted to exercise each branch of the orthonormalization
    for vectors in with_basis:
        field = type(vectors)
        basis = qldpc.math.get_orthonormal_basis(vectors, promise_full_rank=True)
        assert basis is not None and np.array_equal(basis @ basis.T, field.Identity(len(basis)))

    # subspaces with no orthonormal basis: get_orthonormal_basis returns None
    without_basis = [
        # characteristic 2, an alternating form (every vector has even weight)
        galois.GF(2)(
            [[1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0], [0, 1, 1, 1, 1, 0], [0, 0, 0, 1, 0, 1]]
        ),
        galois.GF(2)([[1, 1, 0, 0], [0, 0, 1, 1]]),  # a degenerate form
        galois.GF(7)([[1, 2]]),  # odd characteristic, a non-square discriminant
        galois.GF(7)([[1, 2, 3]]),  # odd characteristic, a degenerate (isotropic) form
        galois.GF(7)([[1, 2, 3], [1, 2, 4]]),  # odd characteristic, isotropic then non-square
    ]
    for vectors in without_basis:
        assert qldpc.math.get_orthonormal_basis(vectors, promise_full_rank=True) is None

    # by default (promise_full_rank=False) the rows are first reduced to a basis of their row space
    vectors = galois.GF(3)([[1, 0], [0, 1], [1, 1]])  # three dependent rows spanning all of GF(3)^2
    basis = qldpc.math.get_orthonormal_basis(vectors)
    assert basis is not None and basis.shape == (2, 2)
    assert np.array_equal(basis @ basis.T, galois.GF(3).Identity(2))


def test_block_matrix() -> None:
    """block_matrix assembles a nested block structure into a single NumPy array."""
    eye = np.eye(2, dtype=float)
    zero = np.zeros_like(eye)
    blocks = [[eye, 1], [0, eye]]
    matrix = np.block([[eye, eye], [zero, eye]])
    assert np.array_equal(qldpc.math.block_matrix(blocks), matrix)

    with pytest.raises(ValueError, match="Inconsistent numbers of blocks in each row"):
        qldpc.math.block_matrix([[0, 1], [1]])
    with pytest.raises(ValueError, match="Inconsistent row numbers"):
        qldpc.math.block_matrix([[np.eye(1), np.eye(2)]])
    with pytest.raises(ValueError, match="Inconsistent column numbers"):
        qldpc.math.block_matrix([[np.eye(1)], [np.eye(2)]])
    with pytest.raises(ValueError, match="Inconsistent block data types"):
        qldpc.math.block_matrix([[np.eye(1, dtype=int), np.eye(1, dtype=float)]])


def test_log() -> None:
    """Log choose function."""
    assert qldpc.math.log_choose(1, 1) == 0
    assert np.allclose(qldpc.math.log_choose(4, 1), np.log(4))
    assert np.allclose(qldpc.math.log_choose(5, 2), np.log(10))
