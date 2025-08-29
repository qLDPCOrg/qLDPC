"""Unit tests for memory.py

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

import pytest

from qldpc import circuits, codes
from qldpc.objects import Pauli


def test_memory_experiment() -> None:
    """Stim circuits for memory experiments."""
    num_rounds, shots = 5, 10
    noise_model = circuits.DepolarizingNoiseModel(1e-2)

    # try out a classical error correcting code
    rep_code = codes.RepetitionCode(3, field=2)
    circuit = circuits.get_memory_experiment(
        rep_code, basis=Pauli.Z, num_rounds=num_rounds, noise_model=noise_model
    )
    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)
    assert detectors.shape[0] == observables.shape[0] == shots
    assert detectors.shape[1] == circuit.num_detectors == rep_code.num_checks * (num_rounds + 1)
    assert observables.shape[1] == rep_code.dimension

    # try tracking both operators in a quantum code
    code = codes.RepetitionCode(2, field=2)
    circuit = circuits.get_memory_experiment(
        code, basis=None, num_rounds=num_rounds, noise_model=noise_model
    )
    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)
    assert detectors.shape[0] == observables.shape[0] == shots
    assert detectors.shape[1] == circuit.num_detectors == code.num_checks * (num_rounds + 1)
    assert observables.shape[1] == code.dimension * 2

    # we can also ask for a noiseless circuit, and inject noise afterwards
    noiseless_circuit = circuits.get_memory_experiment(code, basis=None, num_rounds=num_rounds)
    dem_1 = circuit.detector_error_model()
    dem_2 = noise_model.noisy_circuit(noiseless_circuit).detector_error_model()
    assert dem_1 == dem_2

    # Pauli.Y basis measurements are not supported
    with pytest.raises(ValueError, match="Pauli.X or Pauli.Z"):
        circuits.get_memory_experiment(rep_code, basis=Pauli.Y)  # type:ignore[arg-type]

    # non-CSS and subsystem codes are not always supported
    with pytest.raises(ValueError, match=r"only support stabilizer \(non-subsystem\) codes"):
        circuits.get_memory_experiment(codes.BaconShorCode(2))
    with pytest.raises(ValueError, match=r"only support CSS codes"):
        circuits.get_memory_experiment(codes.FiveQubitCode())
