"""Unit tests for benchmarking.py

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

import numpy as np
import pytest
import stim

from qldpc import circuits, codes, decoders, math
from qldpc.objects import Pauli


def test_state_prep() -> None:
    """State preparation circuit benchmarks."""

    # construct a state prep circuit for the Steane code
    code = codes.SteaneCode()
    circuit = stim.Circuit("""
        # state prep
        H 0 2 4
        CX 0 3 2 1 4 5
        CX 0 1 2 6 4 3
        CX 2 5 3 6
        # flag a logical Z representative
        H 7
        CZ 7 1 7 3 7 5
        MX 7
    """)

    # invalid logical state preparation
    invalid_circuit = circuit + stim.Circuit("X 0")
    with pytest.raises(ValueError, match="does not deterministically prepare a logical code"):
        circuits.get_state_prep_diagnostic_circuit(code, invalid_circuit)
    invalid_circuit = stim.Circuit("H 0\nCX 0 7") + circuits.get_encoding_circuit(code)
    with pytest.raises(ValueError, match="unentangled from ancillas"):
        circuits.get_state_prep_diagnostic_circuit(code, invalid_circuit)

    noise_model_family = circuits.DepolarizingNoiseModel
    error_rates = np.logspace(-3, -1, 5)

    # observables can be specified by either a symplectic matrix or list of Pauli strings
    observables = code.get_logical_ops(Pauli.Z, symplectic=True)
    string_observables = [math.op_to_string(obs) for obs in observables]

    # build sinter tasks
    tasks = circuits.get_state_prep_diagnostic_tasks(
        code,
        circuit,
        error_rates,
        noise_model_family,
        observables=string_observables,
    )
    for error_rate, task in zip(error_rates, tasks):
        assert task.json_metadata["p"] == error_rate

    # find observables automatically and post-select on all measurements
    task = circuits.get_state_prep_diagnostic_tasks(
        code,
        circuit,
        error_rates[:1],
        noise_model_family,
        post_select=True,
    )[0]
    diagnostic_circuit, _ = circuits.get_state_prep_diagnostic_circuit(
        code, circuit, add_flags=True
    )
    postselection_array = np.zeros(task.circuit.num_detectors, dtype=int)
    postselection_array[: circuit.num_measurements] = 1
    postselection_mask = np.packbits(postselection_array, bitorder="little")
    assert task.circuit == noise_model_family(error_rates[0]).noisy_circuit(diagnostic_circuit)
    assert np.array_equal(task.postselection_mask, postselection_mask)

    # we can also post-select manually
    circuit.append("DETECTOR", stim.target_rec(-1))
    diagnostic_circuit, _ = circuits.get_state_prep_diagnostic_circuit(code, circuit)
    task = circuits.get_state_prep_diagnostic_tasks(
        code,
        circuit,
        error_rates[:1],
        noise_model_family,
        observables=string_observables,
        post_select=range(circuit.num_measurements),
    )[0]
    assert task.circuit == noise_model_family(error_rates[0]).noisy_circuit(diagnostic_circuit)
    assert np.array_equal(task.postselection_mask, postselection_mask)

    # we can only manually post-select on detectors that are present in the circuit
    with pytest.raises(ValueError, match="can only post-select on detectors with an index"):
        circuits.get_state_prep_diagnostic_tasks(
            code,
            circuit,
            error_rates[:1],
            noise_model_family,
            observables=string_observables,
            post_select=[circuit.num_measurements],
        )

    # bypass sinter to compute logical error rates
    logical_error_rate, discard_rate = circuits.get_logical_error_and_discard_rate(
        task.circuit,
        sinter_decoder=decoders.SinterDecoder(),
        num_samples=1,
        post_select=range(circuit.num_measurements),
    )
    assert logical_error_rate == 0
    assert discard_rate == 0

    # incompatible DEMs for sampling and decoding
    with pytest.raises(ValueError, match="Incompatible detector error models"):
        circuits.get_logical_error_and_discard_rate(
            task.circuit,
            sinter_decoder=decoders.SinterDecoder(),
            num_samples=1,
            dem_to_decode=stim.DetectorErrorModel(),
        )


def test_finding_unaddressed_measurements() -> None:
    """Identify measurements in a circuit that are not addressed by any detectors."""
    circuit = stim.Circuit("""
        M 0 1 2
        DETECTOR rec[-3] rec[-1]
    """)
    assert circuits.get_unaddressed_measurements(circuit) == [1]
