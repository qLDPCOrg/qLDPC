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

import numpy as np
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
    vectors_conj = np.array([[-1, 0], [-2, 1]], dtype=int)
    assert np.array_equal(qldpc.math.symplectic_weight(vectors), [1, 1])
    assert np.array_equal(qldpc.math.symplectic_conjugate(vectors), vectors_conj)

    assert np.array_equal(qldpc.math.first_nonzero_cols(np.empty(0, dtype=int)), [])
    assert np.array_equal(qldpc.math.first_nonzero_cols(vectors), [1, 0])
    assert np.array_equal(qldpc.math.first_nonzero_cols(vectors_conj), [0, 0])


def test_block_matrix() -> None:
    eye = np.eye(2, dtype=float)
    zero = np.zeros_like(eye)
    blocks = [[eye, 1], [0, eye]]
    matrix = np.block([[eye, eye], [zero, eye]])
    assert np.array_equal(qldpc.math.block_matrix(blocks), matrix)


def test_log() -> None:
    """Log choose function."""
    assert qldpc.math.log_choose(1, 1) == 0
    assert np.allclose(qldpc.math.log_choose(4, 1), np.log(4))
    assert np.allclose(qldpc.math.log_choose(5, 2), np.log(10))
