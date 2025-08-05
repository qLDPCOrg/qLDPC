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

import pytest
import stim

from qldpc import circuits, codes
from qldpc.math import op_to_string
from qldpc.objects import Pauli


def test_restriction() -> None:
    """Raise an error for non-qubit codes."""
    code = codes.SurfaceCode(2, field=3)
    with pytest.raises(ValueError, match="only supported for qubit codes"):
        circuits.get_encoding_circuit(code)


def test_state_prep() -> None:
    """Prepare all-0 logical states of qubit codes."""
    for code in [
        codes.FiveQubitCode(),
        codes.BaconShorCode(3, field=2),
        codes.HGPCode(codes.ClassicalCode.random(5, 3, field=2)),
    ]:
        encoder = circuits.get_encoding_circuit(code)
        simulator = stim.TableauSimulator()
        simulator.do(encoder)

        # the state of the simulator is a +1 eigenstate of code stabilizers
        for row in code.get_stabilizer_ops():
            string = op_to_string(row)
            assert simulator.peek_observable_expectation(string) == 1

        # the state of the simulator is a +1 eigenstate of all logical Z operators
        for op in codes.QuditCode.get_logical_ops(code, Pauli.Z):
            string = op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 1

        # the state of the simulator is a +1 eigenstate of all gauge Z operators
        for op in codes.QuditCode.get_gauge_ops(code, Pauli.Z):
            string = op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 1


def test_logical_tableau() -> None:
    """Reconstruct a logical tableau."""
    code = codes.FiveQubitCode()
    encoder, decoder = circuits.get_encoder_and_decoder(code, deformation=stim.Circuit())

    logical_circuit = stim.Circuit("H 0")
    extended_logical_circuit = logical_circuit + stim.Circuit(f"I {len(code) - 1}")
    physical_tableau = decoder.then(extended_logical_circuit.to_tableau()).then(encoder)
    physical_circuit = physical_tableau.to_circuit()

    reconstructed_logical_tableau = circuits.get_logical_tableau(code, physical_circuit)
    assert logical_circuit.to_tableau() == reconstructed_logical_tableau
