"""Unit tests for common.py

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

import pytest
import stim

from qldpc import circuits, codes
from qldpc.math import op_to_string
from qldpc.objects import Pauli


def test_qubit_ids() -> None:
    """Default qubit indices."""
    code = codes.FiveQubitCode()
    qubit_ids = circuits.QubitIDs.from_code(code)
    data_ids, check_ids, ancilla_ids = qubit_ids
    assert data_ids == list(range(len(code)))
    assert check_ids == list(range(len(code), len(code) + code.num_checks))
    assert not ancilla_ids

    qubit_ids.add_ancilla(3)
    assert qubit_ids.ancilla == [9, 10, 11]


def test_records() -> None:
    """Measurement and detector records."""
    measurement_record = circuits.MeasurementRecord({0: [0]})
    assert measurement_record.num_events == 1
    measurement_record.append({0: [1], 2: [0]})
    assert measurement_record.num_events == 3
    assert dict(measurement_record.items()) == measurement_record.key_to_events
    assert measurement_record.get_target_rec(2) == stim.target_rec(-2)
    assert measurement_record.get_target_rec(0) == stim.target_rec(-1)
    assert measurement_record.get_target_rec(0, -2) == stim.target_rec(-3)
    with pytest.raises(ValueError, match="Qubit 1 not found"):
        measurement_record.get_target_rec(1)
    with pytest.raises(ValueError, match="Invalid measurement index"):
        measurement_record.get_target_rec(0, 2)

    detector_record = circuits.DetectorRecord(measurement_record.key_to_events)
    assert detector_record.num_events == 3
    assert dict(detector_record.items()) == detector_record.key_to_events
    assert detector_record.get_detector(2) == 1
    assert detector_record.get_detector(0) == 2
    assert detector_record.get_detector(0, -2) == 0
    with pytest.raises(ValueError, match="Parity check 1 not found"):
        detector_record.get_detector(1)
    with pytest.raises(ValueError, match="Invalid detection index"):
        detector_record.get_detector(0, 2)


def test_restriction() -> None:
    """Raise an error for non-qubit codes."""
    code = codes.SurfaceCode(2, field=3)
    with pytest.raises(ValueError, match="only supported for qubit codes"):
        circuits.get_encoding_circuit(code)


def test_state_prep() -> None:
    """Prepare all-0 logical states of qubit codes."""
    for code in [
        codes.FiveQubitCode(),
        codes.BaconShorCode(3, field=2),
        codes.HGPCode(codes.ClassicalCode.random(5, 3, field=2)),
    ]:
        encoder = circuits.get_encoding_circuit(code)
        simulator = stim.TableauSimulator()
        simulator.do(encoder)

        # stabilizers have expectation value +1
        for row in code.get_stabilizer_ops():
            string = op_to_string(row)
            assert simulator.peek_observable_expectation(string) == 1

        # logical Z operators have expectation value +1
        for op in codes.QuditCode.get_logical_ops(code, Pauli.Z):
            string = op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 1

        # logical Z operators have expectation value 0
        for op in codes.QuditCode.get_logical_ops(code, Pauli.X):
            string = op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 0

        # gauge Z operators have expectation value +1
        for op in codes.QuditCode.get_gauge_ops(code, Pauli.Z):
            string = op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 1

        # gauge X operators have expectation value 0
        for op in codes.QuditCode.get_gauge_ops(code, Pauli.X):
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
