# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for tests of experimental operators.

The dense matrices here cost memory and time exponential in the number of qubits, so they only check
the scalable methods of qLDPC on a few qubits.  Matrices act on qubits in big-endian order, in which
qubit 0 is the most significant bit of a basis-state index.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import stim

# The matrices of I, X, Y, and Z, indexed as in stim.PauliString.
_PAULI_MATRICES = (
    np.eye(2, dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    np.array([[1, 0], [0, -1]], dtype=np.complex128),
)
# Gates of random circuits, with the number of qubits that each addresses.
_CLIFFORD_GATES: tuple[tuple[str, int], ...] = (
    *(("H", 1), ("S", 1), ("SQRT_X_DAG", 1), ("C_XYZ", 1)),
    *(("CX", 2), ("CZ", 2), ("XCY", 2), ("ISWAP_DAG", 2), ("CXSWAP", 2)),
)


@pytest.fixture
def rng(pytestconfig: pytest.Config) -> np.random.Generator:
    """A random number generator seeded by the seed of pytest-randomly."""
    return np.random.default_rng(pytestconfig.getoption("randomly_seed"))


def get_random_pauli_string(
    rng: np.random.Generator, num_qubits: int, *, nontrivial: bool = False
) -> stim.PauliString:
    """A random Hermitian Pauli string with a random sign, not the identity if ``nontrivial``."""
    while True:
        letters = "".join(rng.choice(list("IXYZ"), size=num_qubits))
        string = (-1) ** int(rng.integers(2)) * stim.PauliString(letters)
        if string.weight or not nontrivial:
            return string


def get_random_clifford_t_circuit(
    rng: np.random.Generator, num_qubits: int, num_operations: int, *, t_fraction: float = 0.3
) -> stim.Circuit:
    """A random Clifford+T circuit with T gates tagged as in ``_operations``.

    The circuit mixes one- and two-qubit Clifford gates, π/2 and π/4 rotations about random Pauli
    products (``SPP``, ``SPP_DAG``, and their T-tagged versions), and T gates ``S[T]``/``S_DAG[T]``,
    with inverted targets and several targets per instruction.
    """
    circuit = stim.Circuit()
    for _ in range(num_operations):
        draw = rng.random()
        if draw < t_fraction / 2:
            name = ("S", "S_DAG")[rng.integers(2)]
            qubits = rng.choice(
                num_qubits, size=rng.integers(1, min(2, num_qubits) + 1), replace=False
            )
            circuit.append(name, qubits.tolist(), tag="T")
        elif draw < t_fraction or draw > 0.85:
            name = ("SPP", "SPP_DAG")[rng.integers(2)]
            targets: list[stim.GateTarget] = []
            for _ in range(rng.integers(1, 3)):
                axis = get_random_pauli_string(rng, num_qubits, nontrivial=True)
                product = stim.target_combined_paulis(axis * axis.sign, axis.sign == -1)
                targets.extend(product)
            circuit.append(name, targets, tag="T" if draw < t_fraction else "")
        else:
            name, size = _CLIFFORD_GATES[rng.integers(len(_CLIFFORD_GATES))]
            if size > num_qubits:
                continue
            qubits = rng.choice(num_qubits, size=size, replace=False)
            circuit.append(name, qubits.tolist())
    return circuit


def get_pauli_matrix(string: stim.PauliString, num_qubits: int) -> npt.NDArray[np.complex128]:
    """The matrix of a Pauli string, with its sign, padded with identities to ``num_qubits``."""
    matrix = np.array([[complex(string.sign)]])
    for qubit in range(num_qubits):
        letter = string[qubit] if qubit < len(string) else 0
        matrix = np.kron(matrix, _PAULI_MATRICES[letter])
    return matrix


def get_rotation_matrix(
    axis: stim.PauliString, angle: float, num_qubits: int
) -> npt.NDArray[np.complex128]:
    """The matrix of the rotation ``exp(-i angle axis/2)`` about a Hermitian Pauli ``axis``."""
    pauli = get_pauli_matrix(axis, num_qubits)
    return np.cos(angle / 2) * np.eye(len(pauli)) - 1j * np.sin(angle / 2) * pauli


def equal_up_to_phase(first: npt.ArrayLike, second: npt.ArrayLike, atol: float = 1e-6) -> bool:
    """Whether two arrays of the same shape are equal up to a global phase, within ``atol``."""
    first = np.asarray(first, dtype=np.complex128)
    second = np.asarray(second, dtype=np.complex128)
    overlap = np.vdot(first, second)
    phase = overlap / abs(overlap) if abs(overlap) > atol else 1
    return bool(np.allclose(phase * first, second, rtol=0, atol=atol))
