# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for tests of experimental tools: dense unitaries of circuits.

The dense matrices here cost memory and time exponential in the number of qubits, so tests use them
only on a few qubits, to check the scalable methods of qLDPC.  These helpers share no code with the
methods that they check, and parse π/4 rotations, encoded as described in the docstring of
``RotatedPauliString``, independently.  As in ``qldpc.experimental.objects.conftest``, matrices act
on qubits in big-endian order.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import stim

from qldpc.experimental.objects.conftest import get_rotation_matrix

# Instructions with no action on the state.
_ANNOTATIONS = frozenset({"TICK", "QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "MPAD"})


def get_circuit_unitary(circuit: stim.Circuit, num_qubits: int) -> npt.NDArray[np.complex128]:
    """The unitary matrix of a Clifford+T circuit, up to a global phase.

    Annotations such as ``TICK`` and ``DETECTOR`` are ignored, and other non-unitary instructions
    are rejected.  Stim provides the matrices of Clifford gates in single precision, with errors of
    about ``1e-7``.
    """
    unitary = np.eye(2**num_qubits, dtype=np.complex128)
    for instruction in circuit.flattened():
        if instruction.name in _ANNOTATIONS:
            continue
        unitary = _get_instruction_unitary(instruction, num_qubits) @ unitary
    return unitary


def _get_instruction_unitary(
    instruction: stim.CircuitInstruction, num_qubits: int
) -> npt.NDArray[np.complex128]:
    """The unitary matrix of an instruction, which may be a π/4 rotation, up to a global phase."""
    # a T-tagged instruction applies the π/4 rotations about its axes in order
    if instruction.tag == "T" or instruction.tag.startswith("T:"):
        unitary = np.eye(2**num_qubits, dtype=np.complex128)
        for axis in _get_t_axes(instruction):
            unitary = get_rotation_matrix(axis, np.pi / 4, num_qubits) @ unitary
        return unitary

    # pad the gate to num_qubits qubits, and take its big-endian matrix, which orders qubits as
    # get_pauli_matrix does
    padded_circuit = stim.Circuit()
    padded_circuit.append(
        instruction.name, instruction.targets_copy(), instruction.gate_args_copy()
    )
    padded_circuit.append("I", [num_qubits - 1])
    tableau = padded_circuit.to_tableau()
    return np.asarray(tableau.to_unitary_matrix(endian="big"), dtype=np.complex128)


def _get_t_axes(instruction: stim.CircuitInstruction) -> list[stim.PauliString]:
    """The Hermitian axes of the π/4 rotations of a T-tagged instruction, in order."""
    sign = -1 if instruction.name.endswith("_DAG") else 1
    if instruction.name in ("S", "S_DAG"):
        return [
            sign * stim.PauliString({target.value: "Z"}) for target in instruction.targets_copy()
        ]
    if instruction.name not in ("SPP", "SPP_DAG"):
        raise ValueError(f"Instruction cannot be a π/4 rotation: {instruction}")  # pragma: no cover
    axes = []
    for group in instruction.target_groups():
        axis = stim.PauliString({target.value: target.pauli_type for target in group})
        inversions = sum(target.is_inverted_result_target for target in group)
        axes.append((-1) ** inversions * sign * axis)
    return axes
