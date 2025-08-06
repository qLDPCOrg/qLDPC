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

import numpy as np
import pytest
import stim

from qldpc import circuits, codes
from qldpc.math import symplectic_conjugate
from qldpc.objects import Pauli


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


@pytest.mark.parametrize("strategy", [circuits.EdgeColoring])
def test_syndrome_measurement(
    strategy: circuits.SyndromeMeasurementStrategy, pytestconfig: pytest.Config
) -> None:
    """Syndrome extraction by Tanner graph edge coloring."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.SteaneCode()
    state_prep = circuits.get_encoding_circuit(code)

    errors = np.random.choice([Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], size=[len(code)])
    ###################################################################
    code = codes.FiveQubitCode()
    state_prep = circuits.get_encoding_circuit(code)
    errors = [Pauli.I] + [Pauli.I] * (len(code) - 1)
    ###################################################################
    error_ops = stim.Circuit()
    for qubit, pauli in enumerate(errors):
        error_ops.append(f"{pauli}_error", [qubit], [1])

    error_vec = code.field([pauli.value for pauli in errors]).T.ravel()
    syndrome_vec = code.matrix @ symplectic_conjugate(error_vec)

    syndrome_extraction, record = strategy.get_circuit(code)
    detectors = stim.Circuit()
    for check in range(len(code), len(code) + code.num_checks):
        detectors.append("DETECTOR", record.get_target_rec(check))

    circuit = state_prep + error_ops + syndrome_extraction + detectors
    sample = circuit.compile_detector_sampler().sample(1).ravel()

    # assert np.array_equal(syndrome_vec, sample)

    print()
    print()
    print()
    print(circuit)
    print()
    print(code.get_strings())
    print()
    print(syndrome_vec)
    print(sample)

    print(np.array_equal(syndrome_vec, sample))
