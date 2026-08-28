"""Unit tests for math.py.

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

    # string_to_op must not mutate its input, even when num_qubits exceeds len(string)
    unpadded = stim.PauliString("XZ")
    qldpc.math.string_to_op(unpadded, num_qubits=5)
    assert len(unpadded) == 2


def test_vectors() -> None:
    """Methods that act on vectors."""
    vectors = np.array([[0, 1], [1, 2]], dtype=int)
    vectors_conj = np.array([[1, 0], [2, -1]], dtype=int)
    assert np.array_equal(qldpc.math.symplectic_weight(vectors), [1, 1])
    assert np.array_equal(qldpc.math.symplectic_conjugate(vectors), vectors_conj)
    assert np.array_equal(qldpc.math.first_nonzero_cols(vectors), [1, 0])
    assert np.array_equal(qldpc.math.first_nonzero_cols(vectors_conj), [0, 0])


def test_symplectic_conjugate_identity() -> None:
    """The symplectic inner product satisfies ``<P, Q>_s = P @ symplectic_conjugate(Q)``.

    Anchors the identity documented on ``symplectic_conjugate``, including the ``-1`` on the Z
    sector that only matters in odd characteristic.
    """
    num_qubits = 6
    # GF(3)/GF(5) exercise the -1 on the Z sector, which is invisible over GF(2) (where -1 == 1).
    for field in [galois.GF(2), galois.GF(3), galois.GF(5)]:
        vectors_p = field(np.random.randint(field.order, size=(4, 2 * num_qubits)))
        vectors_q = field(np.random.randint(field.order, size=(7, 2 * num_qubits)))
        p_x, p_z = vectors_p[:, :num_qubits], vectors_p[:, num_qubits:]
        q_x, q_z = vectors_q[:, :num_qubits], vectors_q[:, num_qubits:]
        expected = p_x @ q_z.T - p_z @ q_x.T
        actual = vectors_p @ qldpc.math.symplectic_conjugate(vectors_q).T
        assert np.array_equal(actual, expected)


def test_nonzero_cols() -> None:
    """Edge cases in finding the pivot columns."""
    # a matrix with no columns: every row is all-zero, so its pivot is the column count (0)
    assert np.array_equal(qldpc.math.first_nonzero_cols(np.zeros((3, 0), dtype=int)), [0, 0, 0])
    assert np.array_equal(qldpc.math.first_nonzero_cols(np.array([], ndmin=2, dtype=int)), [0])

    # a matrix with no rows has no pivots
    assert qldpc.math.first_nonzero_cols(np.zeros((0, 4), dtype=int)).size == 0

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


def test_symplectic_gram_schmidt() -> None:
    """Symplectic Gram-Schmidt: hyperbolic pairs plus a symplectic radical."""
    conj = qldpc.math.symplectic_conjugate

    def check(vectors: galois.FieldArray, *, promise_full_rank: bool = False) -> tuple[int, int]:
        field = type(vectors)
        hyp, rad = qldpc.math.symplectic_gram_schmidt(vectors, promise_full_rank=promise_full_rank)
        num_pairs = len(hyp) // 2
        # the hyperbolic Gram matrix is the block anti-diagonal [[0, I], [-I, 0]]
        expected_gram = field.Zeros((2 * num_pairs, 2 * num_pairs))
        expected_gram[:num_pairs, num_pairs:] = field.Identity(num_pairs)
        expected_gram[num_pairs:, :num_pairs] = -field.Identity(num_pairs)
        assert np.array_equal(hyp @ conj(hyp).T, expected_gram)
        # the radical is isotropic and orthogonal to the entire space
        combined = field(np.vstack([hyp, rad]))
        assert not np.any(rad @ conj(combined).T)
        # the hyperbolic pairs and the radical together span the input row space
        assert np.array_equal(combined.row_space(), vectors.row_space())
        # the reduction is deterministic
        again = qldpc.math.symplectic_gram_schmidt(vectors, promise_full_rank=promise_full_rank)
        assert np.array_equal(again[0], hyp) and np.array_equal(again[1], rad)
        return num_pairs, len(rad)

    for field in [galois.GF(2), galois.GF(3), galois.GF(4)]:
        # a non-degenerate symplectic plane: one hyperbolic pair, empty radical
        assert check(field([[1, 0, 0, 0], [0, 0, 1, 0]]), promise_full_rank=True) == (1, 0)
        # a purely isotropic space (pure-Z rows mutually commute): no pairs, all radical
        assert check(field([[0, 0, 1, 0], [0, 0, 0, 1]]), promise_full_rank=True) == (0, 2)
        # a mix: an (X_0, Z_0) pair and a Z_2 radical vector
        rows = field([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1]])
        assert check(rows, promise_full_rank=True) == (1, 1)
        # the empty subspace
        assert check(field.Zeros((0, 4)), promise_full_rank=True) == (0, 0)
        # linearly dependent rows are reduced to a basis first (promise_full_rank=False)
        assert check(field([[1, 0, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0]])) == (1, 0)


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
    # a literal other than 0 or 1 is rejected (and does so under `python -O`, unlike an assert)
    with pytest.raises(ValueError, match="Unrecognized block: 2"):
        zero = np.zeros((2, 2), dtype=int)
        qldpc.math.block_matrix([[np.eye(2, dtype=int), zero], [zero, 2]])


def test_log() -> None:
    """Log choose function."""
    assert qldpc.math.log_choose(1, 1) == 0
    assert np.allclose(qldpc.math.log_choose(4, 1), np.log(4))
    assert np.allclose(qldpc.math.log_choose(5, 2), np.log(10))
