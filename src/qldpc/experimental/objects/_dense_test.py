# SPDX-License-Identifier: Apache-2.0

"""Tests for the dense reference matrices, against explicit matrices."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc.experimental.objects import _dense as dense

_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_T = np.diag([1, np.exp(1j * np.pi / 4)])


def test_pauli_and_rotation_matrices() -> None:
    np.testing.assert_allclose(
        dense.get_pauli_matrix(-stim.PauliString("XZ"), 3), -np.kron(np.kron(_X, _Z), np.eye(2))
    )
    t_gate = dense.get_rotation_matrix(stim.PauliString("Z"), np.pi / 4, 1)
    assert dense.equal_up_to_phase(t_gate, _T)
    with pytest.raises(ValueError, match="Hermitian"):
        dense.get_rotation_matrix(1j * stim.PauliString("Z"), np.pi / 4, 1)


def test_equality_up_to_phase() -> None:
    matrix = np.kron(_X, _T)
    assert dense.equal_up_to_phase(matrix, 1j * matrix)
    assert not dense.equal_up_to_phase(matrix, matrix.conj())
    assert not dense.equal_up_to_phase(matrix, np.zeros_like(matrix))
    assert not dense.equal_up_to_phase(matrix, _X)
    assert not dense.equal_up_to_phase(matrix, np.kron(_Z, _T))  # orthogonal
    assert dense.equal_up_to_phase(np.zeros((2, 2)), np.zeros((2, 2)))
    assert not dense.equal_up_to_phase(np.eye(2), np.diag([1, 1 + 1e-5]), atol=1e-6)  # no rtol
