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

import random

import numpy as np
import pytest
import stim
import sympy.combinatorics as comb

from qldpc import circuits, codes
from qldpc.math import symplectic_conjugate
from qldpc.objects import Pauli


def test_restriction() -> None:
    """restrict_to_qubits passes through for qubit codes and raises for qudit codes."""

    @circuits.restrict_to_qubits
    def get_empty_circuit(code: codes.QuditCode) -> stim.Circuit:
        return stim.Circuit()

    get_empty_circuit(codes.SurfaceCode(2, field=2))  # this does not raise an error

    with pytest.raises(ValueError, match="only supported for qubit codes"):
        get_empty_circuit(codes.SurfaceCode(2, field=3))


def test_pauli_product_measurements(pytestconfig: pytest.Config) -> None:
    """get_pauli_product_measurements correctly measures the syndrome of Pauli errors."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.FiveQubitCode()
    encoder = circuits.get_encoding_circuit(code)
    stabilizers = code.get_stabilizer_ops()

    error_vec = code.field.Random(len(code) * 2)
    error_ops = stim.Circuit()
    for qubit in range(len(code)):
        xx_zz = (error_vec[qubit], error_vec[qubit + len(code)])
        error_ops.append(str(Pauli(xx_zz)), qubit)

    measurements = circuits.get_pauli_product_measurements(stabilizers)
    outcomes = (encoder + error_ops + measurements).reference_sample()
    syndrome = stabilizers @ symplectic_conjugate(error_vec)
    assert np.array_equal(outcomes.astype(int), syndrome)


def test_qubit_remap(pytestconfig: pytest.Config, num_qubits: int = 8) -> None:
    """Remap the qubits in a stim.Circuit."""
    random.seed(pytestconfig.getoption("randomly_seed"))

    # build a random qubit permutation
    permutation = comb.Permutation.random(num_qubits)
    qubit_map = permutation.array_form

    # build a random circuit
    circuit = stim.Tableau.random(num_qubits).to_circuit()
    circuit.append(
        stim.CircuitRepeatBlock(repeat_count=2, body=stim.Tableau.random(num_qubits).to_circuit())
    )

    # remap qubits using circuits.with_remapped_qubits
    circuit_a = circuits.with_remapped_qubits(circuit, qubit_map)
    assert circuit == circuits.with_remapped_qubits(circuit_a, qubit_map, inverse=True)

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


def test_finding_unaddressed_measurements() -> None:
    """Identify measurements in a circuit that are not addressed by any detectors."""
    circuit = stim.Circuit("""
        M 0 1 2
        DETECTOR rec[-3] rec[-1]
    """)
    assert circuits.get_unaddressed_measurements(circuit) == [1]
