"""Unit tests for bookkeeping.py

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
import sympy.combinatorics as comb

from qldpc import circuits, codes
from qldpc.math import op_to_string
from qldpc.objects import Pauli


def test_qubit_ids() -> None:
    """Default qubit indices."""
    code = codes.SteaneCode()
    qubit_ids = circuits.QubitIDs.from_code(code)
    data_ids, check_ids, ancilla_ids = qubit_ids
    assert data_ids == tuple(range(len(code)))
    assert check_ids == tuple(range(len(code), len(code) + code.num_checks))
    assert not ancilla_ids

    num_ancillas = 3
    qubit_ids.add_ancillas(num_ancillas)
    assert qubit_ids.ancilla == tuple(
        range(len(code) + code.num_checks, len(code) + code.num_checks + num_ancillas)
    )

    assert qubit_ids == circuits.QubitIDs.validated(qubit_ids, code)
    with pytest.raises(ValueError, match="invalid for the given code"):
        circuits.QubitIDs.validated(circuits.QubitIDs((), (), ()), code)


def test_records() -> None:
    """Measurement and detector records."""
    record = circuits.Record({0: [0]})
    assert record.num_events == 1
    record.append({0: [1], 2: [0]})
    assert record.num_events == 3
    record.append({1: [0, 1]}, repeat=3)
    assert record.key_to_events[1] == [3, 4, 5, 6, 7, 8]
    assert dict(record.items()) == record.key_to_events

    measurement_record = circuits.MeasurementRecord(record.key_to_events)
    assert measurement_record.num_events == 9
    assert measurement_record.get_target_rec(2) == stim.target_rec(-8)
    assert measurement_record.get_target_rec(0) == stim.target_rec(-7)
    assert measurement_record.get_target_rec(0, -2) == stim.target_rec(-9)

    with pytest.raises(ValueError, match="Invalid measurement index"):
        measurement_record.get_target_rec(3)
    with pytest.raises(ValueError, match="Invalid measurement index"):
        measurement_record.get_target_rec(0, 2)

    detector_record = circuits.DetectorRecord(record.key_to_events)
    assert detector_record.num_events == 9
    assert detector_record.get_detector(2) == 1
    assert detector_record.get_detector(0) == 2
    assert detector_record.get_detector(0, -2) == 0

    with pytest.raises(ValueError, match="Invalid detection index"):
        detector_record.get_detector(3)
    with pytest.raises(ValueError, match="Invalid detection index"):
        detector_record.get_detector(0, 2)


def test_restriction() -> None:
    """Raise an error for non-qubit codes."""
    code = codes.SurfaceCode(2, field=3)
    with pytest.raises(ValueError, match="only supported for qubit codes"):
        circuits.get_encoding_circuit(code)


def test_state_prep(pytestconfig: pytest.Config) -> None:
    """Prepare all-0 logical states of qubit codes."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    codes_to_test = [
        codes.FiveQubitCode(),
        codes.BaconShorCode(3, field=2),
        codes.HGPCode(codes.ClassicalCode.random(5, 3, field=2, seed=np.random.randint(2**32 - 1))),
    ]

    for code, only_zero in itertools.product(codes_to_test, [True, False]):
        encoder = circuits.get_encoding_circuit(code, only_zero=only_zero)
        simulator = stim.TableauSimulator()
        simulator.do(encoder)

        # stabilizers have expectation value +1
        for row in code.get_stabilizer_ops():
            string = op_to_string(row)
            assert simulator.peek_observable_expectation(string) == 1

        # logical Z operators have expectation value +1
        for op in code.get_logical_ops(Pauli.Z, symplectic=True):
            string = op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 1

        # logical Z operators have expectation value 0
        for op in code.get_logical_ops(Pauli.X, symplectic=True):
            string = op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 0

        if only_zero is False:
            # gauge Z operators have expectation value +1
            for op in code.get_gauge_ops(Pauli.Z, symplectic=True):
                string = op_to_string(op)
                assert simulator.peek_observable_expectation(string) == 1

            # gauge X operators have expectation value 0
            for op in code.get_gauge_ops(Pauli.X, symplectic=True):
                string = op_to_string(op)
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


def test_qubit_remap(num_qubits: int = 8) -> None:
    """Remap the qubits in a stim.Circuit."""
    # identify a qubit permutation
    permutation = comb.Permutation.random(num_qubits)
    qubit_map = permutation.array_form

    # build a random circuit
    circuit = stim.Tableau.random(num_qubits).to_circuit()
    circuit.append(
        stim.CircuitRepeatBlock(repeat_count=2, body=stim.Tableau.random(num_qubits).to_circuit())
    )

    # remap qubits using circuits.with_remapped_qubits
    circuit_a = circuits.with_remapped_qubits(circuit, qubit_map)

    # manually construct a permutation circuit to implement the remapping
    inverse_permutation_circuit = stim.Circuit()
    for cycle in permutation.cyclic_form:
        for qq in range(1, len(cycle)):
            inverse_permutation_circuit.append("SWAP", [cycle[0], cycle[qq]])

    # test that the two remapped circuits are equivalent
    circuit_b = inverse_permutation_circuit.inverse() + circuit + inverse_permutation_circuit
    assert circuit_a.to_tableau() == circuit_b.to_tableau()

    # cover an edge case
    circuit_a = circuits.with_remapped_qubits(stim.Circuit("MPP X1*!Y2 \n M !4"), {2: 3})
    circuit_b = stim.Circuit("MPP X1*!Y3 \n M !4")
    assert circuit_a == circuit_b
