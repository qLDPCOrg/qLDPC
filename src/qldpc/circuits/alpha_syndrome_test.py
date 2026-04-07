"""Unit tests for alpha_syndrome.py

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

from qldpc import circuits, codes, math
from qldpc.objects import Pauli


def test_alpha_syndrome(pytestconfig: pytest.Config) -> None:
    """Verify that syndromes are read out correctly."""
    seed = pytestconfig.getoption("randomly_seed")

    # verify that AlphaSyndrome builds valid syndrome extraction circuits for a few codes
    assert alpha_syndrome_is_valid(codes.SteaneCode())
    assert alpha_syndrome_is_valid(codes.ToricCode(2, rotated=True))
    assert alpha_syndrome_is_valid(codes.SurfaceCode(2, rotated=True))

    code_a = codes.ClassicalCode.random(5, 3, seed=seed)
    code_b = codes.ClassicalCode.random(3, 2, seed=seed + 1)
    assert alpha_syndrome_is_valid(codes.HGPCode(code_a, code_b))

    # AlphaSyndrome does not support non-CSS codes
    with pytest.raises(ValueError, match="only supports CSS codes"):
        strategy = circuits.AlphaSyndrome(circuits.DepolarizingNoiseModel(0.001), "decoder_name")
        strategy.get_circuit(codes.FiveQubitCode())


def alpha_syndrome_is_valid(
    code: codes.QuditCode,
    strategy: circuits.AlphaSyndrome = circuits.AlphaSyndrome(
        circuits.DepolarizingNoiseModel(0.001),
        iters_per_step=3,
        shots_per_iter=1,
    ),
) -> bool:
    """Check that an AlphaSyndrome circuit correctly reads out stabilizers."""
    # prepare a logical |0> state
    state_prep = circuits.get_encoding_circuit(code)

    # apply random Pauli errors to the data qubits
    errors = random.choices([Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], k=len(code))
    error_ops = stim.Circuit()
    for qubit, pauli in enumerate(errors):
        error_ops.append(f"{pauli}_error", [qubit], [1])

    # measure syndromes
    syndrome_extraction_circuit, measurement_record = strategy.get_circuit(code)
    for check in range(len(code), len(code) + code.num_checks):
        syndrome_extraction_circuit.append("DETECTOR", measurement_record.get_target_rec(check))

    # sample the circuit to obtain a syndrome vector
    circuit = state_prep + error_ops + syndrome_extraction_circuit
    syndrome = circuit.compile_detector_sampler().sample(1).ravel()

    # compare against the expected syndrome
    error_xz = code.field([pauli.value for pauli in errors]).T.ravel()
    expected_syndrome = code.matrix @ math.symplectic_conjugate(error_xz)
    return np.array_equal(expected_syndrome, syndrome)
