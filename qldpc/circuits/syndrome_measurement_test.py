"""Unit tests for syndrome_measurement.py

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

import random

import numpy as np
import pytest
import stim

from qldpc import circuits, codes
from qldpc.math import symplectic_conjugate
from qldpc.objects import Pauli


def test_qubit_ids() -> None:
    """Default qubit indices."""
    code = codes.FiveQubitCode()
    qubit_ids = circuits.QubitIDs.from_code(code)
    data_ids, check_ids = qubit_ids
    assert data_ids == list(range(len(code)))
    assert check_ids == list(range(len(code), len(code) + code.num_checks))


def test_measurement_record() -> None:
    """Build and use a MeasurementRecord."""
    record = circuits.MeasurementRecord()
    record.append({0: [0, 1], 2: [2]})
    assert record.num_measurements == 3
    assert dict(record.items()) == record.qubit_to_measurements
    assert record.get_target_rec(2) == stim.target_rec(-1)
    assert record.get_target_rec(0) == stim.target_rec(-2)
    with pytest.raises(ValueError, match="Qubit 1 not found"):
        record.get_target_rec(1)
    with pytest.raises(ValueError, match="Invalid measurement index"):
        record.get_target_rec(0, 2)


def test_syndrome_measurement(pytestconfig: pytest.Config) -> None:
    """Syndrome extraction by Tanner graph edge coloring."""
    random.seed(pytestconfig.getoption("randomly_seed"))

    # classical seed codes for a random HGPCode
    code_a = codes.ClassicalCode.random(5, 3, seed=random.randint(0, 2**32))
    code_b = codes.ClassicalCode.random(3, 2, seed=random.randint(0, 2**32))

    for code, strategy in [
        (codes.FiveQubitCode(), circuits.SerialExtraction),
        (codes.SteaneCode(), circuits.EdgeColoring),
        (codes.SurfaceCode(2, rotated=True), circuits.CardinalEdgeColoring),
        (codes.ToricCode(2, rotated=True), circuits.CardinalEdgeColoring),
        (codes.HGPCode(code_a, code_b), circuits.CardinalEdgeColoring),
    ]:
        # prepare a logical |0> state
        state_prep = circuits.get_encoding_circuit(code)

        # apply random Pauli errors to the data qubits
        errors = random.choices([Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], k=len(code))
        error_ops = stim.Circuit()
        for qubit, pauli in enumerate(errors):
            if pauli is not Pauli.I:  # I_ERROR is only recognized in stim>=1.15.0
                error_ops.append(f"{pauli}_error", [qubit], [1])

        # measure syndromes
        syndrome_extraction, record = strategy.get_circuit(code)
        for check in range(len(code), len(code) + code.num_checks):
            syndrome_extraction.append("DETECTOR", record.get_target_rec(check))

        # sample the circuit to obtain a syndrome vector
        circuit = state_prep + error_ops + syndrome_extraction
        syndrome = circuit.compile_detector_sampler().sample(1).ravel()

        # compare against the expected syndrome
        error_xz = code.field([pauli.value for pauli in errors]).T.ravel()
        expected_syndrome = code.matrix @ symplectic_conjugate(error_xz)
        assert np.array_equal(expected_syndrome, syndrome)


def test_syndrome_errors() -> None:
    """Not all codes are supported by all syndrome extraction circuits."""
    with pytest.raises(ValueError, match="does not work for non-CSS codes"):
        circuits.EdgeColoring.get_circuit(codes.FiveQubitCode())

    with pytest.raises(ValueError, match="not equipped with a syndrome_subgraphs property"):
        circuits.CardinalEdgeColoring.get_circuit(codes.SteaneCode())
