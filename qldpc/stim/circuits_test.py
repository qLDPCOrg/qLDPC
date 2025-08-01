"""Unit tests for circuits.py

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

from qldpc import circuits, codes, external
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


def test_transversal_ops() -> None:
    """Construct SWAP-transversal logical Cliffords of a code."""
    code = codes.FiveQubitCode()

    for local_gates in [("SWAP", "S"), ("SWAP", "H"), ("SWAP", "SQRT_X"), ("SWAP", "H", "S")]:
        transversal_ops = circuits.get_transversal_ops(code, local_gates)
        assert len(transversal_ops) == len(local_gates) - 1

    with pytest.raises(ValueError, match="Local Clifford gates"):
        circuits.get_transversal_automorphism_group(code, ["SQRT_Y"])


def test_finding_circuit(pytestconfig: pytest.Config) -> None:
    """Find a physical circuit for a desired logical Clifford operation."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code: codes.QuditCode = codes.FiveQubitCode()

    if external.gap.is_installed():  # pragma: no cover
        # randomly permute the qubits to switch things up!
        new_matrix = code.matrix.reshape(-1, 5)[:, np.random.permutation(5)].reshape(-1, 10)
        code = codes.QuditCode(new_matrix)

    # logical circuit: random single-qubit Clifford recognized by Stim
    logical_op = np.random.choice(
        [
            "X",
            "Y",
            "Z",
            "C_XYZ",
            "C_ZYX",
            "H",
            "H_XY",
            "H_XZ",
            "H_YZ",
            "S",
            "SQRT_X",
            "SQRT_X_DAG",
            "SQRT_Y",
            "SQRT_Y_DAG",
            "SQRT_Z",
            "SQRT_Z_DAG",
            "S_DAG",
        ]
    )
    logical_circuit = stim.Circuit(f"{logical_op} 0")

    # construct physical circuit for the logical operation
    physical_circuit = circuits.get_transversal_circuit(code, logical_circuit)
    assert physical_circuit is not None

    # check that the physical circuit has the correct logical tableau
    reconstructed_logical_tableau = circuits.get_logical_tableau(code, physical_circuit)
    assert logical_circuit.to_tableau() == reconstructed_logical_tableau

    # there are no logical two-qubit gates in this code
    circuits.get_transversal_circuit(code, stim.Circuit("CX 0 1")) is None


def test_deformed_decoder() -> None:
    """Deform a code in such a way as to preserve its logicals, but change its stabilizers."""
    code = codes.CSSCode([[1] * 6], [[1] * 6], field=2)
    code.set_logical_ops_xz(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 1, 1],
        ],
        [
            [0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0],
        ],
    )
    deformation = stim.Circuit("H 0 1 2")
    encoder, decoder = circuits.get_encoder_and_decoder(code)
    deformation_encoder, deformation_decoder = circuits.get_encoder_and_decoder(code, deformation)
    assert encoder == deformation_encoder
    assert decoder == encoder.inverse()
    assert decoder != deformation_decoder
