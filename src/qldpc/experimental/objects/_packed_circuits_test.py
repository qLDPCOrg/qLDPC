# SPDX-License-Identifier: Apache-2.0

"""Tests for packing Clifford+T circuits, against dense matrices."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc.experimental.conftest import get_circuit_unitary
from qldpc.experimental.objects import _packed_circuits as packed_circuits
from qldpc.experimental.objects import _packed_paulis as packed
from qldpc.experimental.objects.conftest import (
    equal_up_to_phase,
    get_random_clifford_t_circuit,
    get_rotation_matrix,
)

# The rotation angles of the types of rotations.
_ANGLES = {packed_circuits.S_ROTATION: np.pi / 2, packed_circuits.T_ROTATION: np.pi / 4}


def _get_matrices(
    packed_circuit: packed_circuits.PackedCircuit,
) -> list[npt.NDArray[np.complex128]]:
    """The dense matrices of the operations of a packed circuit, up to global phases.

    The matrix of a local gate is built from the images of the generators ``X_j`` and ``Z_j`` in its
    shared conjugation table, and the matrix of a rotation from its axis.
    """
    num_qubits = packed_circuit.num_qubits
    conjugation_tables = packed.get_conjugation_tables()
    matrices = []
    for operation_type, operation_qubits, operation_index in zip(
        packed_circuit.operation_types,
        packed_circuit.operation_qubits,
        packed_circuit.operation_indices,
    ):
        if operation_type == packed_circuits.LOCAL_GATE:
            # read the images of the generators from the conjugation table of the local gate, whose
            # row of qubits ends in -1 if it is a single-qubit gate
            qubits = operation_qubits[operation_qubits >= 0].tolist()
            images = conjugation_tables.images[operation_index]
            image_phases = conjugation_tables.image_phases[operation_index]
            generator_images = []
            for generator in range(2 * len(qubits)):
                image = int(images[1 << generator])
                words = [image & ((1 << len(qubits)) - 1), image >> len(qubits)]
                phase = int(image_phases[1 << generator])
                generator_images.append(packed.to_stim(words, phase, len(qubits)))

            # build the matrix of the Clifford gate with these images, on the qubits of the gate
            tableau = stim.Tableau.from_conjugated_generators(
                xs=generator_images[: len(qubits)], zs=generator_images[len(qubits) :]
            )
            circuit = stim.Circuit()
            for instruction in tableau.to_circuit():
                gate_targets = [qubits[target.value] for target in instruction.targets_copy()]
                circuit.append(instruction.name, gate_targets)
            matrices.append(get_circuit_unitary(circuit, num_qubits))
        else:
            axis = packed.to_stim(
                packed_circuit.rotation_axes[operation_index],
                int(packed_circuit.rotation_axis_phases[operation_index]),
                num_qubits,
            )
            matrices.append(get_rotation_matrix(axis, _ANGLES[int(operation_type)], num_qubits))
    return matrices


@pytest.mark.parametrize("inverse", [False, True])
def test_packed_circuits_reproduce_circuits(rng: np.random.Generator, inverse: bool) -> None:
    num_qubits = 3
    for _ in range(10):
        circuit = get_random_clifford_t_circuit(rng, num_qubits, 8)
        circuit.append("TICK")
        unitary = np.eye(2**num_qubits, dtype=complex)
        packed_circuit = packed_circuits.get_packed_circuit(circuit, num_qubits, inverse=inverse)
        for matrix in _get_matrices(packed_circuit):
            unitary = matrix @ unitary
        expected = get_circuit_unitary(circuit, num_qubits)
        assert equal_up_to_phase(unitary, expected.conj().T if inverse else expected)


@pytest.mark.parametrize("inverse", [False, True])
def test_local_gates_of_packed_circuits(inverse: bool) -> None:
    """Each local gate has its qubits and the shared index of its stim gate, or of its inverse."""
    circuit = stim.Circuit()
    for name, gate_data in stim.gate_data().items():
        if gate_data.is_unitary and (gate_data.is_single_qubit_gate or gate_data.is_two_qubit_gate):
            circuit.append(name, [0] if gate_data.is_single_qubit_gate else [0, 1])
    packed_circuit = packed_circuits.get_packed_circuit(circuit, 2, inverse=inverse)
    gate_indices = packed.get_conjugation_tables().gate_indices

    # each instruction is one local gate, and the inverse circuit applies the inverse gates in
    # reverse order
    instructions = list(circuit)[::-1] if inverse else list(circuit)
    assert packed_circuit.operation_types.tolist() == [packed_circuits.LOCAL_GATE] * len(circuit)
    for operation, instruction in enumerate(instructions):
        qubits = [target.value for target in instruction.targets_copy()]
        assert packed_circuit.operation_qubits[operation].tolist() == (qubits + [-1])[:2]
        gate_data = stim.gate_data(instruction.name)
        gate_name = gate_data.inverse.name if inverse else gate_data.name
        assert packed_circuit.operation_indices[operation] == gate_indices[gate_name]


def test_packed_circuits_of_tagged_instructions() -> None:
    """T-tagged instructions, labeled or not, are π/4 rotations, and a tag on H is ignored."""
    circuit = stim.Circuit("S[T] 0 1\nSPP_DAG[T:label] !X0*Z1\nSPP Y1\nH[other] 0")
    packed_circuit = packed_circuits.get_packed_circuit(circuit, 2)
    t_rotation, s_rotation = packed_circuits.T_ROTATION, packed_circuits.S_ROTATION
    local_gate = packed_circuits.LOCAL_GATE
    assert packed_circuit.operation_types.tolist() == [
        t_rotation,
        t_rotation,
        t_rotation,
        s_rotation,
        local_gate,
    ]
    axes = [
        packed.to_stim(words, int(phase), 2)
        for words, phase in zip(packed_circuit.rotation_axes, packed_circuit.rotation_axis_phases)
    ]
    assert axes == [stim.PauliString(axis) for axis in ("+Z_", "+_Z", "+XZ", "+_Y")]
    gate_index = packed.get_conjugation_tables().gate_indices["H"]
    assert packed_circuit.operation_qubits.tolist() == [[-1, -1]] * 4 + [[0, -1]]
    assert packed_circuit.operation_indices.tolist() == [0, 1, 2, 3, gate_index]


def test_packed_circuits_are_read_only_and_skip_annotations() -> None:
    packed_circuit = packed_circuits.get_packed_circuit(stim.Circuit("MPAD 0\nTICK\nH 0"), 1)
    assert packed_circuit.operation_types.tolist() == [packed_circuits.LOCAL_GATE]
    with pytest.raises(ValueError, match="read-only"):
        packed_circuit.operation_types[0] = packed_circuits.T_ROTATION


def test_invalid_circuits() -> None:
    with pytest.raises(ValueError, match="not unitary"):
        packed_circuits.get_packed_circuit(stim.Circuit("M 0"), 1)
    with pytest.raises(ValueError, match="cannot be a π/4 rotation"):
        packed_circuits.get_packed_circuit(stim.Circuit("H[T] 0"), 1)
    with pytest.raises(ValueError, match="Tags other than T"):
        packed_circuits.get_packed_circuit(stim.Circuit("I[R_Z(theta=0.25*pi)] 0"), 1)
    with pytest.raises(ValueError, match="Classically controlled"):
        packed_circuits.get_packed_circuit(stim.Circuit("CX sweep[0] 0"), 1)
    with pytest.raises(ValueError, match="repeat qubits"):
        packed_circuits.get_packed_circuit(stim.Circuit("SPP[T] X0*X0"), 1)
    with pytest.raises(ValueError, match="2 > 1"):
        packed_circuits.get_packed_circuit(stim.Circuit("CX 0 1"), 1)
