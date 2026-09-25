"""Unit tests for memory.py.

Copyright 2025 The qLDPC Authors

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
import stim

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
    parts = circuits.get_memory_experiment_parts(rep_code, basis=Pauli.X, num_rounds=1)
    assembled = parts.initialization + parts.qec_cycle + parts.readout
    assert circuits.get_memory_experiment(
        rep_code, basis=Pauli.X, num_rounds=1, noise_model=noise_model
    ) == noise_model.noisy_circuit(assembled)
    surface_code = codes.SurfaceCode(2)

    class AncillaStrategy(circuits.SyndromeMeasurementStrategy):
        def __init__(self, *, use_reference: bool = False) -> None:
            self.use_reference = use_reference

        def get_circuit(
            self, code: codes.QuditCode, qubit_ids: circuits.QubitIDs | None = None
        ) -> tuple[stim.Circuit, circuits.MeasurementRecord]:
            assert qubit_ids is not None
            circuit, record = circuits.EdgeColoring().get_circuit(code, qubit_ids)
            target = qubit_ids.reference[0] if self.use_reference else qubit_ids.ancilla[0]
            circuit.append("H", target)
            circuit.append("TICK")
            return circuit, record

    qubit_ids = circuits.QubitIDs(
        range(1, 5),
        range(10, 10 + surface_code.num_checks),
        [9],
        reference=[0],
    )
    combined = circuits.get_memory_experiment(
        surface_code,
        basis=None,
        noise_model=circuits.NoiseModel(idle_error=0.01),
        qubit_ids=qubit_ids,
        syndrome_measurement_strategy=AncillaStrategy(),
    )
    assert not any("DEPOLARIZE" in line and " 0" in line for line in str(combined).splitlines())
    assert any("DEPOLARIZE" in line and " 9" in line for line in str(combined).splitlines())
    with pytest.raises(ValueError, match="Bell-reference ancillas"):
        circuits.get_memory_experiment_parts(
            surface_code,
            basis=None,
            qubit_ids=qubit_ids,
            syndrome_measurement_strategy=AncillaStrategy(use_reference=True),
        )

    # Pauli.Y basis measurements are not supported
    with pytest.raises(ValueError, match="Pauli.X or Pauli.Z"):
        circuits.get_memory_experiment(rep_code, basis=Pauli.Y)  # type:ignore[arg-type]

    # non-CSS and subsystem codes are not always supported
    with pytest.raises(ValueError, match=r"only support stabilizer \(non-subsystem\) codes"):
        circuits.get_memory_experiment(codes.BaconShorCode(2))
    with pytest.raises(TypeError, match=r"only support CSS codes"):
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
        ancilla=qubits[len(code) + code.num_checks + code.dimension :],
        reference=qubits[
            len(code) + code.num_checks : len(code) + code.num_checks + code.dimension
        ],
    )

    for basis in PAULIS_XZ + [None]:
        # produce a memory experiment with the requested qubit IDs
        init, cycle, readout, *_, qubit_ids = circuits.get_memory_experiment_parts(
            code, basis=basis, num_rounds=num_qec_rounds, qubit_ids=qubit_ids
        )
        circuit_a = init + cycle + readout

        # produces a memory experiment with the default qubit IDs and remap manually
        qubit_map = (
            qubit_ids.data + qubit_ids.check + (qubit_ids.reference if basis is None else ())
        )
        circuit_b = circuits.with_remapped_qubits(
            circuits.get_memory_experiment(code, basis=basis, num_rounds=num_qec_rounds),
            qubit_map,
        )

        assert circuit_a.flattened() == circuit_b.flattened()


def test_errors() -> None:
    """Cover invalid options for observable annotations."""
    with pytest.raises(ValueError, match="CSS codes"):
        circuits.get_observables(codes.FiveQubitCode(), basis=Pauli.X, on_measurements=True)
    with pytest.raises(ValueError, match="fixed measurement basis"):
        circuits.get_observables(codes.SteaneCode(), basis=None, on_measurements=True)
    with pytest.raises(ValueError, match="basis must be"):
        circuits.get_observables(codes.SteaneCode(), basis="test", on_measurements=True)  # type:ignore[arg-type]
    with pytest.raises(ValueError, match="num_rounds"):
        circuits.get_memory_experiment_parts(codes.RepetitionCode(3), Pauli.X, num_rounds=0)
    with pytest.raises(ValueError, match="one target per data qubit"):
        circuits.get_observables(codes.SteaneCode(), data_qubits=[0], basis=Pauli.X)
    with pytest.raises(ValueError, match="one target per data qubit"):
        circuits.get_observables(
            codes.SteaneCode(), basis=Pauli.X, on_measurements=[stim.target_rec(-1)]
        )
    with pytest.raises(ValueError, match="observable indices"):
        circuits.get_observables(codes.SteaneCode(), basis=Pauli.X, observable_indices=[0, 1])
    with pytest.raises(ValueError, match="one target per data qubit"):
        circuits.get_logical_bell_prep(codes.SteaneCode(), data_qubits=[0])
    with pytest.raises(ValueError, match="one target per logical qubit"):
        circuits.get_logical_bell_prep(codes.SteaneCode(), ancilla_qubits=[0, 1])


def test_memory_rejects_unsynchronized_strategy_records() -> None:
    """A strategy that emits an unrecorded measurement fails at the first lookback."""

    class BadStrategy(circuits.SyndromeMeasurementStrategy):
        def get_circuit(
            self, code: codes.QuditCode, qubit_ids: circuits.QubitIDs | None = None
        ) -> tuple[stim.Circuit, circuits.MeasurementRecord]:
            circuit, record = circuits.EdgeColoring().get_circuit(code, qubit_ids)
            circuit.append("M", 0)
            return circuit, record

    with pytest.raises(ValueError, match="record contains"):
        circuits.get_memory_experiment_parts(
            codes.SurfaceCode(2),
            basis=Pauli.X,
            syndrome_measurement_strategy=BadStrategy(),
        )
