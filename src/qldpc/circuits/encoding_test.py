"""Unit tests for encoding.py

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

import itertools

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import circuits, codes, math
from qldpc.objects import Pauli


def test_encoding_circuit(pytestconfig: pytest.Config) -> None:
    """Prepare logical Pauli states of qubit codes."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    codes_to_test = [
        codes.FiveQubitCode(),
        codes.SHPCode(codes.ClassicalCode.random(4, 2, seed=np.random.randint(2**31))),
    ]

    for code, only_zero in itertools.product(codes_to_test, [True, False]):
        simulator = stim.TableauSimulator()

        if not only_zero:
            # choose random logical and gauge Paulis that should be +1 after encoding
            bases = np.random.choice(
                [Pauli.Z, Pauli.X, Pauli.Y],  # type:ignore[arg-type]
                size=code.dimension + code.gauge_dimension,
            )
            basis_prep = stim.Circuit()
            basis_prep.append("H", np.where(bases != Pauli.Z)[0])
            basis_prep.append("S", np.where(bases == Pauli.Y)[0])
            simulator.do(basis_prep)

        encoder = circuits.get_encoding_circuit(code, only_zero=only_zero)
        simulator.do(encoder)

        # all stabilizers have expectation value +1
        for op in code.get_stabilizer_ops():
            string = math.op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 1

        if only_zero:
            # logical Z operators have expectation value +1
            for op in code.get_logical_ops(Pauli.Z, symplectic=True):
                string = math.op_to_string(op)
                assert simulator.peek_observable_expectation(string) == 1

        else:
            # examine logical operators that should have expectation value +1
            for qq in range(code.dimension):
                basis = bases[qq]
                op_x = code.get_logical_ops(Pauli.X, symplectic=True)[qq]
                op_z = code.get_logical_ops(Pauli.Z, symplectic=True)[qq]
                string = _get_logical_pauli_string(basis, op_x, op_z)
                assert simulator.peek_observable_expectation(string) == 1

            # examine gauge operators that should have expectation value +1
            for qq in range(code.gauge_dimension):
                basis = bases[code.dimension + qq]
                op_x = code.get_gauge_ops(Pauli.X, symplectic=True)[qq]
                op_z = code.get_gauge_ops(Pauli.Z, symplectic=True)[qq]
                string = _get_logical_pauli_string(basis, op_x, op_z)
                assert simulator.peek_observable_expectation(string) == 1


def _get_logical_pauli_string(  # pragma: no cover
    basis: Pauli, op_x: npt.NDArray[np.int_], op_z: npt.NDArray[np.int_]
) -> stim.PauliString:
    """Build a logical operator Pauli string from symplectic logical Pauli X and Z vectors."""
    if basis is Pauli.X:
        return math.op_to_string(op_x)
    if basis is Pauli.Z:
        return math.op_to_string(op_z)
    string_x = math.op_to_string(op_x)
    string_z = math.op_to_string(op_z)
    return 1j * string_x * string_z


def test_logical_tableau() -> None:
    """Reconstruct a logical tableau."""
    code = codes.FiveQubitCode()
    encoder, decoder = circuits.get_encoder_and_decoder(code, deformation=stim.Circuit())

    logical_circuit = stim.Circuit("H 0")
    extended_logical_circuit = logical_circuit + stim.Circuit(f"I {len(code) - 1}")
    physical_tableau = decoder.then(extended_logical_circuit.to_tableau()).then(encoder)
    physical_circuit = physical_tableau.to_circuit()

    reconstructed_logical_tableau = circuits.get_logical_tableau(code, physical_circuit)
    assert logical_circuit.to_tableau() == reconstructed_logical_tableau


def test_state_stabilizers(pytestconfig: pytest.Config) -> None:
    """Identify the stabilizers of state prepared by a circuit."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.SHPCode(codes.ClassicalCode.random(4, 2, seed=np.random.randint(2**31)))
    encoder = circuits.get_encoding_circuit(code)

    # prepare a random logical stabilizer state
    circuit = stim.Tableau.random(num_qubits=code.dimension).to_circuit() + encoder

    # the circuit can include detectors and observables
    circuit.append("X", [len(code)])
    circuit.append("M", [len(code)])
    circuit.append("DETECTOR", [stim.target_rec(-1)])
    simulator = stim.TableauSimulator()
    simulator.do(circuit)

    # all stabilizers of the code have expectation value +1
    for op in code.get_stabilizer_ops():
        string = math.op_to_string(op)
        assert simulator.peek_observable_expectation(string) == 1

    # all stabilizers of the state have expectation value +1
    state_stabs = circuits.get_state_stabilizers(circuit, len(code))
    assert len(state_stabs) == len(code)
    for stab in state_stabs:
        assert simulator.peek_observable_expectation(stab) == 1

    # all logical stabilizers of the state have expectation value +1
    logical_stabs = circuits.get_logical_state_stabilizers(circuit, code)
    assert len(logical_stabs) == code.dimension
    for stab in logical_stabs:
        assert simulator.peek_observable_expectation(stab) == 1

    # the logical stabilizers are "pure", and have no stabilizer content
    for stab in logical_stabs:
        decoded_stab = stab.before(encoder)
        xs, zs = decoded_stab.to_numpy()
        assert not np.any(xs[code.dimension :])
        assert not np.any(zs[code.dimension :])
