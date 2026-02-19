"""Unit tests for alphasyndrome.py

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
import sinter
import stim

from qldpc import circuits, codes, math
from qldpc.objects import Pauli


class TrivialDecoder(sinter.Decoder):
    def compile_decoder_for_dem(
        self, *, dem: "stim.DetectorErrorModel"
    ) -> "TrivialCompiledDecoder":
        return TrivialCompiledDecoder(shape=(dem.num_observables + 7) // 8)


class TrivialCompiledDecoder(sinter.CompiledDecoder):
    def __init__(self, shape: int):
        self.shape = shape

    def decode_shots_bit_packed(
        self,
        *,
        bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        return np.zeros(
            shape=(bit_packed_detection_event_data.shape[0], self.shape), dtype=np.uint8
        )


def test_alpha_syndrome(pytestconfig: pytest.Config) -> None:
    """Verify that syndromes are read out correctly."""
    seed = pytestconfig.getoption("randomly_seed")

    # default strategies for non-CSS and CSS codes
    assert_valid_alphasyndrome(codes.SteaneCode())

    # special strategies for toric and surface codes
    assert_valid_alphasyndrome(codes.ToricCode(2, rotated=True))
    assert_valid_alphasyndrome(codes.SurfaceCode(2, rotated=True))

    # special strategy for HGPCodes
    code_a = codes.ClassicalCode.random(5, 3, seed=seed)
    code_b = codes.ClassicalCode.random(3, 2, seed=seed + 1)
    assert_valid_alphasyndrome(codes.HGPCode(code_a, code_b))

    # EdgeColoringXZ strategy
    with pytest.raises(ValueError, match="only supports CSS codes"):
        circuits.AlphaSyndrome(
            circuits.DepolarizingNoiseModel(0.001),
            "trivial",
            iters_per_step=2,
            shots_per_iter=5,
            custome_decoders={"trivial": TrivialDecoder()},
        ).get_circuit(codes.FiveQubitCode())


def assert_valid_alphasyndrome(
    code: codes.QuditCode,
) -> None:
    strategy = circuits.AlphaSyndrome(
        circuits.DepolarizingNoiseModel(0.001),
        "trivial",
        iters_per_step=5,
        shots_per_iter=5,
        custome_decoders={"trivial": TrivialDecoder()},
    )

    """Assert that the syndrome measurement of the given code with the given strategy is valid."""
    # prepare a logical |0> state
    state_prep = circuits.get_encoding_circuit(code)

    # apply random Pauli errors to the data qubits
    errors = random.choices([Pauli.I, Pauli.X, Pauli.Y, Pauli.Z], k=len(code))
    error_ops = stim.Circuit()
    for qubit, pauli in enumerate(errors):
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
    expected_syndrome = code.matrix @ math.symplectic_conjugate(error_xz)
    assert np.array_equal(expected_syndrome, syndrome)
