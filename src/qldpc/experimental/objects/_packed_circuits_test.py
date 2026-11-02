# SPDX-License-Identifier: Apache-2.0

"""Tests for packing Clifford+T circuits, against the dense reference matrices."""

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

# The rotation angles of the kinds of rotations.
_ANGLES = {packed_circuits.S_ROTATION: np.pi / 2, packed_circuits.T_ROTATION: np.pi / 4}


def _get_matrices(
    packed_circuit: packed_circuits.PackedCircuit,
) -> list[npt.NDArray[np.complex128]]:
    """The dense matrices of the operations of a packed circuit, built from gate names and axes."""
    num_qubits = packed_circuit.num_qubits
    matrices = []
    for kind, operand in zip(packed_circuit.kinds, packed_circuit.operands):
        if kind == packed_circuits.LOCAL_GATE:
            table = packed_circuit.gate_tables[operand]
            qubits = packed_circuit.gate_qubits[operand, : packed_circuit.table_sizes[table]]
            circuit = stim.Circuit()
            circuit.append(packed_circuit.table_names[table], qubits.tolist())
            matrices.append(get_circuit_unitary(circuit, num_qubits))
        else:
            words = packed_circuit.axes[operand]
            phase = int(packed_circuit.axis_phases[operand])
            axis = packed.to_stim(words, phase, num_qubits)
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


def test_packed_circuits_of_t_gates() -> None:
    circuit = stim.Circuit("S[T] 0 1\nSPP_DAG[T:label] !X0*Z1\nSPP Y1\nH[other] 0")
    packed_circuit = packed_circuits.get_packed_circuit(circuit, 2)
    t_rotation, s_rotation = packed_circuits.T_ROTATION, packed_circuits.S_ROTATION
    local_gate = packed_circuits.LOCAL_GATE
    assert packed_circuit.kinds.tolist() == [
        t_rotation,
        t_rotation,
        t_rotation,
        s_rotation,
        local_gate,
    ]
    axes = [
        packed.to_stim(words, int(phase), 2)
        for words, phase in zip(packed_circuit.axes, packed_circuit.axis_phases)
    ]
    assert axes == [stim.PauliString(s) for s in ("+Z_", "+_Z", "+XZ", "+_Y")]
    assert packed_circuit.table_names == ("H",)


def test_packed_circuits_are_read_only_and_skip_annotations() -> None:
    packed_circuit = packed_circuits.get_packed_circuit(stim.Circuit("MPAD 0\nTICK\nH 0"), 1)
    assert packed_circuit.kinds.tolist() == [packed_circuits.LOCAL_GATE]
    with pytest.raises(ValueError, match="read-only"):
        packed_circuit.kinds[0] = packed_circuits.T_ROTATION


def test_invalid_circuits() -> None:
    with pytest.raises(ValueError, match="not unitary"):
        packed_circuits.get_packed_circuit(stim.Circuit("M 0"), 1)
    with pytest.raises(ValueError, match="cannot be a T gate"):
        packed_circuits.get_packed_circuit(stim.Circuit("H[T] 0"), 1)
    with pytest.raises(ValueError, match="Tags other than T"):
        packed_circuits.get_packed_circuit(stim.Circuit("I[R_Z(theta=0.25*pi)] 0"), 1)
    with pytest.raises(ValueError, match="Classically controlled"):
        packed_circuits.get_packed_circuit(stim.Circuit("CX sweep[0] 0"), 1)
    with pytest.raises(ValueError, match="repeat qubits"):
        packed_circuits.get_packed_circuit(stim.Circuit("SPP[T] X0*X0"), 1)
    with pytest.raises(ValueError, match="acts on 2 > 1 qubits"):
        packed_circuits.get_packed_circuit(stim.Circuit("CX 0 1"), 1)
