"""Unit tests for transversal.py.

Copyright 2024 The qLDPC Authors

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

import contextlib

import numpy as np
import pytest
import stim

from qldpc import abstract, circuits, codes, external
from qldpc.circuits import transversal


def test_transversal_s() -> None:
    """Build a physical circuit for the transversal logical S gate of a SWEL code."""
    # the Steane code is SWEL, so its logical S gate is transversal
    code = codes.SteaneCode()
    circuit = circuits.get_transversal_s(code)

    # the circuit is built entirely from single-qubit S and S_DAG gates (no Pauli corrections)
    assert {instruction.name for instruction in circuit} <= {"S", "S_DAG"}

    # the circuit implements a logical S gate
    logical_tableau = circuits.get_logical_tableau(code, circuit)
    assert logical_tableau == stim.Circuit("S 0").to_tableau()

    # non-SWEL codes are rejected
    with pytest.raises(ValueError, match="SWEL"):
        circuits.get_transversal_s(codes.ToricCode(2))


def test_transversal_ops() -> None:
    """Construct SWAP-transversal logical Cliffords of a code."""
    code: codes.CSSCode

    code = codes.ToricCode(2)
    assert len(circuits.get_transversal_ops(code, [])) == 0
    assert len(circuits.get_transversal_ops(code, ["SWAP"])) == 2
    assert len(circuits.get_transversal_ops(code, ["SWAP", "H"])) == 3
    assert len(circuits.get_transversal_ops(code, ["SWAP", "S"])) == 3
    assert len(circuits.get_transversal_ops(code, ["SWAP", "SQRT_X"])) == 3
    assert len(circuits.get_transversal_ops(code, ["SWAP", "H", "S"])) == 4

    # non-self-dual [[8, 3, 2]] CSS code with nontrivial permutation automorphisms
    code = codes.CSSCode(
        [[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 1, 1, 0, 0], [0, 0, 1, 1, 0, 0, 1, 1]],
        [[1, 0, 1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1, 0, 1]],
    )
    assert len(circuits.get_transversal_ops(code, ["SWAP"])) >= 3

    with pytest.raises(ValueError, match="Local Clifford gates"):
        circuits.get_transversal_automorphism_group(code, ["SQRT_Y"])
    with pytest.raises(TypeError, match="single string"):
        circuits.get_transversal_automorphism_group(code, "SWAP")


def test_transversal_group_degree() -> None:
    """Construct transversal logical Cliffords of codes whose automorphisms fix their last columns.

    A permutation group acts on as many points as its generators move, so the automorphism group of
    a check matrix whose last columns every automorphism fixes acts on fewer points than the matrix
    has columns.  A transversal automorphism group nonetheless acts on every column, which is what
    lets it be evaluated on every qubit of the code.
    """
    code: codes.CSSCode

    # self-dual code with one qubit that no parity check addresses, so that its four addressed
    # qubits can be permuted arbitrarily
    code = codes.CSSCode([[1, 1, 1, 1, 0]], [[1, 1, 1, 1, 0]])
    group = circuits.get_transversal_automorphism_group(code, ["SWAP"])
    assert group.degree == len(code)
    assert group.order == 24
    assert len(circuits.get_transversal_ops(code, ["SWAP"])) == 2

    # non-self-dual code, whose X and Z sector automorphism groups are intersected with each other
    code = codes.CSSCode([[0, 0, 0, 0], [1, 1, 1, 0]], [[0, 1, 1, 0]])
    group = circuits.get_transversal_automorphism_group(code, ["SWAP"])
    assert group.degree == len(code)
    assert group.order == 2

    # a complete local Clifford gate set, which addresses three parity check sectors per qubit
    code = codes.CSSCode([[1, 1, 0, 0, 1], [1, 0, 0, 1, 0]], [[1, 0, 0, 1, 1], [0, 0, 1, 0, 0]])
    group = circuits.get_transversal_automorphism_group(code, ["SWAP", "H", "S"])
    assert group.degree == 3 * len(code)
    assert group.order == 16
    assert len(circuits.get_transversal_ops(code, ["SWAP", "H", "S"])) == 1

    # a code deformation, whose transversal operations are constrained by the logical operators
    code = codes.SurfaceCode(2)
    group = circuits.get_transversal_automorphism_group(code, ["SWAP", "H", "S"], deform_code=True)
    assert group.degree == 3 * len(code)
    assert group.order == 48
    deformations = circuits.get_transversal_ops(code, ["SWAP", "H", "S"], deform_code=True)
    assert len(deformations) == 1
    for _, physical_circuit in deformations:
        # the physical circuit deforms the code while preserving its logical operators
        code.deformed(physical_circuit, preserve_logicals=True)

    # a code whose instrumental automorphism group is trivial, admitting no transversal gate
    code = codes.CSSCode([[1]], [[0]])
    group = circuits.get_transversal_automorphism_group(code, ["H"])
    assert group.degree == 2 * len(code)
    assert group.order == 1
    assert len(circuits.get_transversal_ops(code, ["H"])) == 0


def test_finding_circuit(
    pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Find a physical circuit for a desired logical Clifford operation."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code: codes.QuditCode = codes.FiveQubitCode()

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

    monkeypatch.setattr("builtins.input", lambda: "n")  # user declines to pass around GAP commands
    if external.gap.is_installed() and external.gap.is_callable():  # pragma: no cover
        # randomly permute the qubits to switch things up!
        new_matrix = code.matrix.reshape(-1, 5)[:, np.random.permutation(5)].reshape(-1, 10)
        code = codes.QuditCode(new_matrix)
    capsys.readouterr()  # intercept printed text

    context = (
        contextlib.nullcontext()
        if code == codes.FiveQubitCode() or code.is_equiv_to(codes.FiveQubitCode())
        else pytest.warns(UserWarning, match="with_magma=True")
    )
    with context:
        # construct physical circuit for the logical operation
        physical_circuit = circuits.get_transversal_circuit(code, logical_circuit)
        assert physical_circuit is not None

        # A logical circuit must have the same number of qubits as the code's logical space.
        with pytest.raises(ValueError, match="at most 1 qubits"):
            circuits.get_transversal_circuit(code, stim.Circuit("CX 0 1"))
        with pytest.raises(ValueError, match="logical tableau on 1 qubits"):
            circuits.get_transversal_circuits(code, [stim.Tableau(2)])

        # There are no logical two-qubit gates in the SWAP-only group of this
        # two-logical-qubit CSS code.
        assert (
            circuits.get_transversal_circuit(
                codes.CSSCode([[1, 1, 0, 0]], [[0, 0, 1, 1]]),
                stim.Circuit("CX 0 1"),
                local_gates=[],
            )
            is None
        )

    zero_logical_code = codes.CSSCode([[1]], [[0]])
    monkeypatch.setattr(
        transversal,
        "get_transversal_automorphism_group",
        lambda *_args, **_kwargs: abstract.Group(abstract.GroupMember(range(3))),
    )
    assert circuits.get_transversal_circuits(zero_logical_code, [stim.Circuit()])[0] is not None
    monkeypatch.undo()
    assert circuits.get_transversal_circuit(codes.FiveQubitCode(), stim.Circuit("X 0")) is not None
    assert circuits.get_transversal_circuit(codes.FiveQubitCode(), stim.Circuit("Z 0")) is not None

    one_qubit_code = codes.CSSCode([[1]], [[0]])
    for permutation in ([2, 0, 1], [1, 2, 0]):
        circuit = transversal._get_pauli_permutation_circuit(
            one_qubit_code, abstract.GroupMember(permutation), ["H", "S"]
        )
        assert circuit.num_qubits == 1

    # check that the physical circuit has the correct logical tableau
    reconstructed_logical_tableau = circuits.get_logical_tableau(code, physical_circuit)
    assert logical_circuit.to_tableau() == reconstructed_logical_tableau


def test_deformed_decoder() -> None:
    """Deform a code in such a way as to preserve its logicals, but change its stabilizers."""
    code = codes.CSSCode([[1] * 6], [[1] * 6])
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
