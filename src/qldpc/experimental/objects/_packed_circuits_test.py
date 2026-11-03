# SPDX-License-Identifier: Apache-2.0

"""Tests for packing Clifford+T circuits, against dense matrices."""

from __future__ import annotations

import itertools

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

# The rotation angles of the kinds of rotations.
_ANGLES = {packed_circuits.S_ROTATION: np.pi / 2, packed_circuits.T_ROTATION: np.pi / 4}


def _get_matrices(
    packed_circuit: packed_circuits.PackedCircuit,
) -> list[npt.NDArray[np.complex128]]:
    """The dense matrices of the operations of a packed circuit, up to global phases.

    The matrix of a local gate is built from the images of the generators ``X_j`` and ``Z_j`` in its
    conjugation table, and the matrix of a rotation from its axis.
    """
    num_qubits = packed_circuit.num_qubits
    matrices = []
    for kind, gate_or_rotation in zip(
        packed_circuit.operation_kinds, packed_circuit.gate_or_rotation_indices
    ):
        if kind == packed_circuits.LOCAL_GATE:
            # read the images of the generators from the conjugation table of the local gate, whose
            # row of qubits ends in -1 if it is a single-qubit gate
            gate_qubits = packed_circuit.gate_qubits[gate_or_rotation]
            qubits = gate_qubits[gate_qubits >= 0].tolist()
            conjugation_table = packed_circuit.gate_conjugation_table_indices[gate_or_rotation]
            images = packed_circuit.conjugation_table_images[conjugation_table]
            image_phases = packed_circuit.conjugation_table_image_phases[conjugation_table]
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
                packed_circuit.rotation_axes[gate_or_rotation],
                int(packed_circuit.rotation_axis_phases[gate_or_rotation]),
                num_qubits,
            )
            matrices.append(get_rotation_matrix(axis, _ANGLES[int(kind)], num_qubits))
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
def test_conjugation_tables_of_packed_circuits(inverse: bool) -> None:
    """The packed conjugation tables of all local gates map Pauli strings as stim does."""
    circuit = stim.Circuit()
    for name, gate_data in stim.gate_data().items():
        if gate_data.is_unitary and (gate_data.is_single_qubit_gate or gate_data.is_two_qubit_gate):
            circuit.append(name, [0] if gate_data.is_single_qubit_gate else [0, 1])
    packed_circuit = packed_circuits.get_packed_circuit(circuit, 2, inverse=inverse)
    assert len(packed_circuit.conjugation_table_images) == len(circuit)

    # each instruction is one local gate, and the inverse circuit applies the inverse gates in
    # reverse order, so the conjugation table of a gate maps a Pauli string as conjugation by its
    # instruction does, or for the inverse circuit as conjugation by the inverse of its instruction
    instructions = list(circuit)[::-1] if inverse else list(circuit)
    for gate, instruction in enumerate(instructions):
        qubits = np.array([target.value for target in instruction.targets_copy()], dtype=np.int64)
        assert packed_circuit.gate_qubits[gate, : len(qubits)].tolist() == qubits.tolist()
        conjugation_table = packed_circuit.gate_conjugation_table_indices[gate]
        for letters in itertools.product("IXYZ", repeat=len(qubits)):
            string = stim.PauliString("".join(letters))
            words, phase = packed.from_stim(string)
            phase = packed.conjugate_by_local_gate(
                words,
                phase,
                qubits,
                packed_circuit.conjugation_table_images[conjugation_table],
                packed_circuit.conjugation_table_image_phases[conjugation_table],
            )
            expected = string.before(instruction) if inverse else string.after(instruction)
            assert packed.to_stim(words, phase, len(qubits)) == expected, (instruction, string)


def test_packed_circuits_of_tagged_instructions() -> None:
    """T-tagged instructions, labeled or not, are π/4 rotations, and a tag on H is ignored."""
    circuit = stim.Circuit("S[T] 0 1\nSPP_DAG[T:label] !X0*Z1\nSPP Y1\nH[other] 0")
    packed_circuit = packed_circuits.get_packed_circuit(circuit, 2)
    t_rotation, s_rotation = packed_circuits.T_ROTATION, packed_circuits.S_ROTATION
    local_gate = packed_circuits.LOCAL_GATE
    assert packed_circuit.operation_kinds.tolist() == [
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
    images, image_phases = packed.get_conjugation_table("H")
    assert packed_circuit.gate_qubits.tolist() == [[0, -1]]
    assert packed_circuit.conjugation_table_images[:, : len(images)].tolist() == [images.tolist()]
    assert packed_circuit.conjugation_table_image_phases[:, : len(images)].tolist() == [
        image_phases.tolist()
    ]


def test_packed_circuits_are_read_only_and_skip_annotations() -> None:
    packed_circuit = packed_circuits.get_packed_circuit(stim.Circuit("MPAD 0\nTICK\nH 0"), 1)
    assert packed_circuit.operation_kinds.tolist() == [packed_circuits.LOCAL_GATE]
    with pytest.raises(ValueError, match="read-only"):
        packed_circuit.operation_kinds[0] = packed_circuits.T_ROTATION


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
