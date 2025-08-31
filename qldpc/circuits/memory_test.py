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

import random

import pytest

from qldpc import circuits, codes
from qldpc.objects import PAULIS_XZ, Pauli


def test_memory_experiment() -> None:
    """Stim circuits for memory experiments."""
    num_rounds, shots = 5, 10
    noise_model = circuits.DepolarizingNoiseModel(1e-2)

    # try out a classical error correcting code
    rep_code = codes.RepetitionCode(3)
    circuit = circuits.get_memory_experiment(
        rep_code, basis=Pauli.Z, num_rounds=num_rounds, noise_model=noise_model
    )
    sampler = circuit.compile_detector_sampler()
    detectors, observables = sampler.sample(shots=shots, separate_observables=True)
    assert detectors.shape[0] == observables.shape[0] == shots
    assert detectors.shape[1] == circuit.num_detectors == rep_code.num_checks * (num_rounds + 1)
    assert observables.shape[1] == rep_code.dimension

    # try tracking both operators in a quantum code
    code = codes.RepetitionCode(2)
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


def test_qubit_ids(pytestconfig: pytest.Config) -> None:
    """We can construct memory experiments with different qubit IDs."""
    random.seed(pytestconfig.getoption("randomly_seed"))

    # pick a code, a number of "extra" unused qubits, and a number of QEC rounds
    code = codes.SurfaceCode(2, rotated=True)
    num_unused_qubits = 3
    num_qec_rounds = 3

    # assign random qubit indices
    qubits = list(range(len(code) + code.num_checks + code.dimension + num_unused_qubits))
    random.shuffle(qubits)
    qubit_ids = circuits.QubitIDs(
        data=qubits[: len(code)],
        check=qubits[len(code) : len(code) + code.num_checks],
        ancilla=qubits[len(code) + code.num_checks :],
    )

    for basis in PAULIS_XZ + [None]:
        # produce a memory experiment with the requested qubit IDs
        init, cycle, readout, *_, qubit_ids = circuits.get_memory_experiment_parts(
            code, basis=basis, num_rounds=num_qec_rounds, qubit_ids=qubit_ids
        )
        circuit_a = init + cycle + readout

        # produces a memory experiment with the default qubit IDs and remap manually
        qubit_map = qubit_ids.data + qubit_ids.check + qubit_ids.ancilla
        circuit_b = circuits.with_remapped_qubits(
            circuits.get_memory_experiment(code, basis=basis, num_rounds=num_qec_rounds),
            qubit_map,
        )

        assert circuit_a.flattened() == circuit_b.flattened()
