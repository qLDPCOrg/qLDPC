# SPDX-License-Identifier: Apache-2.0

"""Dense matrices of Pauli strings and Pauli rotations, for tests only.

Library code must not use this module, since its cost in memory and time is exponential in the
number of qubits.  Its reference implementations check the scalable methods of qLDPC on a few
qubits, and share no code with them.

Matrices act on ``n`` qubits in big-endian order, in which qubit 0 is the most significant bit of a
basis-state index, and are complex128 arrays.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import stim

# The matrices of I, X, Y, and Z, indexed as in stim.PauliString.
_PAULI_MATRICES = (
    np.eye(2, dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
)

####################################################################################################
# dense matrices


def get_pauli_matrix(string: stim.PauliString, num_qubits: int) -> npt.NDArray[np.complex128]:
    """The matrix of a Pauli string, including its sign, padded with identities to ``num_qubits``.

    The matrix acts on ``num_qubits`` qubits, which must be at least the length of ``string``, and
    is exact.
    """
    matrix = np.array([[complex(string.sign)]])
    for qubit in range(num_qubits):
        letter = string[qubit] if qubit < len(string) else 0
        matrix = np.kron(matrix, _PAULI_MATRICES[letter])
    return matrix


def get_rotation_matrix(
    axis: stim.PauliString, angle: float, num_qubits: int
) -> npt.NDArray[np.complex128]:
    """The matrix of the rotation ``exp(-i angle axis/2)`` about a Hermitian Pauli ``axis``."""
    if axis.sign not in (1, -1):
        raise ValueError(f"Rotation axes must be Hermitian, but {axis} is not")
    pauli = get_pauli_matrix(axis, num_qubits)
    return np.cos(angle / 2) * np.eye(len(pauli)) - 1j * np.sin(angle / 2) * pauli


def equal_up_to_phase(first: npt.ArrayLike, second: npt.ArrayLike, atol: float = 1e-6) -> bool:
    """Whether two arrays are equal up to a global phase, entry by entry within ``atol``."""
    first = np.asarray(first, dtype=np.complex128)
    second = np.asarray(second, dtype=np.complex128)
    if first.shape != second.shape:
        return False
    norm = abs(np.vdot(first, first))
    if norm < atol or abs(np.vdot(second, second)) < atol:
        return bool(norm < atol and abs(np.vdot(second, second)) < atol)
    overlap = np.vdot(first, second)
    if abs(overlap) < atol:
        return False
    phase = overlap / abs(overlap)
    return bool(np.allclose(phase * first, second, rtol=0, atol=atol))
