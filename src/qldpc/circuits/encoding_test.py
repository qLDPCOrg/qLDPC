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
import pytest
import stim

from qldpc import circuits, codes, math
from qldpc.objects import Pauli


def test_state_prep(pytestconfig: pytest.Config) -> None:
    """Prepare all-0 logical states of qubit codes."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    codes_to_test = [
        codes.FiveQubitCode(),
        codes.SHPCode(codes.ClassicalCode.random(4, 2, seed=np.random.randint(2**31))),
    ]

    for code, only_zero in itertools.product(codes_to_test, [True, False]):
        encoder = circuits.get_encoding_circuit(code, only_zero=only_zero)

        simulator = stim.TableauSimulator()
        simulator.do(encoder)

        # stabilizers have expectation value +1
        for op in code.get_stabilizer_ops():
            string = math.op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 1

        # logical Z operators have expectation value +1
        for op in code.get_logical_ops(Pauli.Z, symplectic=True):
            string = math.op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 1

        # logical X operators have expectation value 0
        for op in code.get_logical_ops(Pauli.X, symplectic=True):
            string = math.op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 0

        if only_zero is False:
            # gauge Z operators have expectation value +1
            for op in code.get_gauge_ops(Pauli.Z, symplectic=True):
                string = math.op_to_string(op)
                assert simulator.peek_observable_expectation(string) == 1

            # gauge X operators have expectation value 0
            for op in code.get_gauge_ops(Pauli.X, symplectic=True):
                string = math.op_to_string(op)
                assert simulator.peek_observable_expectation(string) == 0


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


def test_state_stabilizers() -> None:
    """Identify the stabilizers of state prepared by a circuit."""
    code = codes.SteaneCode()
    prep_z = circuits.get_encoding_circuit(code)
    prep_z += circuits.get_pauli_product_measurements(
        code.get_logical_ops(Pauli.Z, symplectic=True)
    )
    prep_z.append("DETECTOR", [stim.target_rec(-1)])

    # stabilizers in "decoded" (logical/gauge/stabilizer/destabilizer) basis
    decoded_state_stabs = circuits.get_state_stabilizers(code, prep_z, decoded=True)

    # stabilizers in physical-qubit bases
    state_stabs = [stab.after(prep_z) for stab in decoded_state_stabs]
    assert len(state_stabs) == len(code)

    # all stabilizers should have expectation value +1
    simulator = stim.TableauSimulator()
    simulator.do(prep_z)
    for stab in state_stabs:
        assert simulator.peek_observable_expectation(stab) == 1


def test_nontrivial_logical_stabilizers(pytestconfig: pytest.Config) -> None:
    """Identify logical stabilizers of a logical state."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.SHPCode(codes.ClassicalCode.random(4, 2, seed=np.random.randint(2**31)))

    # prepare a random logical stabilizer state
    circuit = stim.Tableau.random(num_qubits=code.dimension).to_circuit()
    circuit += circuits.get_encoding_circuit(code)

    logical_stabs = circuits.get_nontrivial_logical_stabilizers(code, circuit)
    assert len(logical_stabs) == code.dimension

    # All logical stabilizers should have a nonzero expectation value.
    # They might be -1 because we do not keep track of their sign.
    simulator = stim.TableauSimulator()
    simulator.do(circuit)
    for stab in logical_stabs:
        string = math.op_to_string(stab)
        assert simulator.peek_observable_expectation(string) != 0

    invalid_circuit = stim.Circuit(f"X {len(code) - 1}") + circuit
    with pytest.raises(ValueError, match="does not .* prepare a logical code state"):
        circuits.get_nontrivial_logical_stabilizers(code, invalid_circuit)
