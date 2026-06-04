"""Unit tests for encoding.py

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

import itertools

import numpy as np
import pytest
import stim

from qldpc import circuits, codes
from qldpc.math import op_to_string
from qldpc.objects import Pauli


def test_state_prep(pytestconfig: pytest.Config) -> None:
    """Prepare all-0 logical states of qubit codes."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    codes_to_test = [
        codes.FiveQubitCode(),
        codes.SHPCode(codes.ClassicalCode.random(4, 2, seed=np.random.randint(2**31))),
    ]

    for code, only_zero in itertools.product(codes_to_test, [True, False]):
        encoder = circuits.get_encoding_circuit(code, only_zero=only_zero)

        simulator = stim.TableauSimulator()
        simulator.do(encoder)

        # stabilizers have expectation value +1
        for row in code.get_stabilizer_ops():
            string = op_to_string(row)
            assert simulator.peek_observable_expectation(string) == 1

        # logical Z operators have expectation value +1
        for op in code.get_logical_ops(Pauli.Z, symplectic=True):
            string = op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 1

        # logical X operators have expectation value 0
        for op in code.get_logical_ops(Pauli.X, symplectic=True):
            string = op_to_string(op)
            assert simulator.peek_observable_expectation(string) == 0

        if only_zero is False:
            # gauge Z operators have expectation value +1
            for op in code.get_gauge_ops(Pauli.Z, symplectic=True):
                string = op_to_string(op)
                assert simulator.peek_observable_expectation(string) == 1

            # gauge X operators have expectation value 0
            for op in code.get_gauge_ops(Pauli.X, symplectic=True):
                string = op_to_string(op)
                assert simulator.peek_observable_expectation(string) == 0


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
