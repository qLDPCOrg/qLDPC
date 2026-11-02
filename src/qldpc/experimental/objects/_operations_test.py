# SPDX-License-Identifier: Apache-2.0

"""Tests for parsing Clifford+T circuits, against the dense reference matrices."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc.experimental.conftest import get_circuit_unitary
from qldpc.experimental.objects import _operations as operations
from qldpc.experimental.objects import _packed_paulis as packed
from qldpc.experimental.objects.conftest import (
    equal_up_to_phase,
    get_random_clifford_t_circuit,
    get_rotation_matrix,
)


def _get_matrices(program: operations.Program) -> list[npt.NDArray[np.complex128]]:
    """The dense matrices of the operations of a program, built from gate names and axes."""
    num_qubits = program.num_qubits
    matrices = []
    for kind, operand in zip(program.kinds, program.operands):
        if kind == operations.GATE:
            table = program.gate_tables[operand]
            qubits = program.gate_qubits[operand, : program.table_sizes[table]]
            circuit = stim.Circuit()
            circuit.append(program.table_names[table], qubits.tolist())
            matrices.append(get_circuit_unitary(circuit, num_qubits))
        else:
            words, phase = program.axes[operand], int(program.axis_phases[operand])
            axis = packed.to_stim(words, phase, num_qubits)
            matrices.append(get_rotation_matrix(axis, np.pi * kind / 4, num_qubits))
    return matrices


@pytest.mark.parametrize("inverse", [False, True])
def test_programs_reproduce_circuits(rng: np.random.Generator, inverse: bool) -> None:
    num_qubits = 3
    for _ in range(10):
        circuit = get_random_clifford_t_circuit(rng, num_qubits, 8)
        circuit.append("TICK")
        unitary = np.eye(2**num_qubits, dtype=complex)
        program = operations.get_program(circuit, num_qubits, inverse=inverse)
        for matrix in _get_matrices(program):
            unitary = matrix @ unitary
        expected = get_circuit_unitary(circuit, num_qubits)
        assert equal_up_to_phase(unitary, expected.conj().T if inverse else expected)


def test_programs_of_t_gates() -> None:
    circuit = stim.Circuit("S[T] 0 1\nSPP_DAG[T:label] !X0*Z1\nSPP Y1\nH[other] 0")
    program = operations.get_program(circuit, 2)
    t_rotation, s_rotation, gate = operations.T_ROTATION, operations.S_ROTATION, operations.GATE
    assert program.kinds.tolist() == [t_rotation, t_rotation, t_rotation, s_rotation, gate]
    axes = [
        packed.to_stim(words, int(phase), 2)
        for words, phase in zip(program.axes, program.axis_phases)
    ]
    assert axes == [stim.PauliString(s) for s in ("+Z_", "+_Z", "+XZ", "+_Y")]
    assert program.table_names == ("H",)


def test_programs_are_read_only_and_skip_annotations() -> None:
    program = operations.get_program(stim.Circuit("MPAD 0\nTICK\nH 0"), 1)
    assert program.kinds.tolist() == [operations.GATE]
    with pytest.raises(ValueError, match="read-only"):
        program.kinds[0] = operations.T_ROTATION


def test_invalid_circuits() -> None:
    with pytest.raises(ValueError, match="not unitary"):
        operations.get_program(stim.Circuit("M 0"), 1)
    with pytest.raises(ValueError, match="cannot be a T gate"):
        operations.get_program(stim.Circuit("H[T] 0"), 1)
    with pytest.raises(ValueError, match="Tags other than T"):
        operations.get_program(stim.Circuit("I[R_Z(theta=0.25*pi)] 0"), 1)
    with pytest.raises(ValueError, match="Classically controlled"):
        operations.get_program(stim.Circuit("CX sweep[0] 0"), 1)
    with pytest.raises(ValueError, match="repeat qubits"):
        operations.get_program(stim.Circuit("SPP[T] X0*X0"), 1)
    with pytest.raises(ValueError, match="acts on 2 > 1 qubits"):
        operations.get_program(stim.Circuit("CX 0 1"), 1)
