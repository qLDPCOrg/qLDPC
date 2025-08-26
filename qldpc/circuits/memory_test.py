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
    # try out a classical error correcting code
    num_rounds, shots = 5, 10
    rep_code = codes.RepetitionCode(3)
    noise_model = circuits.DepolarizingNoiseModel(1e-2)
    circuit = circuits.get_memory_experiment(
        rep_code,
        num_rounds=num_rounds,
        basis=Pauli.Z,
        noise_model=noise_model,
    )
    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)
    assert detectors.shape[0] == observables.shape[0] == shots
    assert detectors.shape[1] == circuit.num_detectors == rep_code.num_checks * (num_rounds + 1)
    assert observables.shape[1] == rep_code.dimension

    # only Pauli.X and Pauli.Z basis measurements are supported
    with pytest.raises(ValueError, match="Pauli.X or Pauli.Z"):
        circuits.get_memory_experiment(rep_code, basis=Pauli.Y)

    # non-CSS and subsystem codes are not yet supported
    with pytest.raises(ValueError, match=r"only support stabilizer \(non-subsystem\) CSS codes"):
        circuits.get_memory_experiment(codes.FiveQubitCode())
    with pytest.raises(ValueError, match=r"only support stabilizer \(non-subsystem\) CSS codes"):
        circuits.get_memory_experiment(codes.BaconShorCode(2))
