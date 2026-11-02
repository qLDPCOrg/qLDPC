# SPDX-License-Identifier: Apache-2.0

"""Unit tests for qubit_remapping.py."""

from __future__ import annotations

import random

import pytest
import stim
import sympy.combinatorics as comb

from qldpc import circuits
from qldpc.circuits.conftest import StimCircuitWrapper


def test_qubit_remap(pytestconfig: pytest.Config, num_qubits: int = 8) -> None:
    """Remap the qubits in a stim.Circuit."""
    random.seed(pytestconfig.getoption("randomly_seed"))

    # build a random qubit permutation
    permutation = comb.Permutation.random(num_qubits)
    qubit_map = permutation.array_form

    # build a random circuit
    circuit = stim.Tableau.random(num_qubits).to_circuit()
    circuit.append(
        stim.CircuitRepeatBlock(repeat_count=2, body=stim.Tableau.random(num_qubits).to_circuit())
    )

    # remap qubits using circuits.with_remapped_qubits
    circuit_a = circuits.with_remapped_qubits(circuit, qubit_map)
    assert circuit == circuits.with_remapped_qubits(circuit_a, qubit_map, inverse=True)

    # manually construct a permutation circuit to implement the remapping
    inverse_permutation_circuit = stim.Circuit()
    for cycle in permutation.cyclic_form:
        for qq in range(1, len(cycle)):
            inverse_permutation_circuit.append("SWAP", [cycle[0], cycle[qq]])

    # test that the two remapped circuits are equivalent
    circuit_b = inverse_permutation_circuit.inverse() + circuit + inverse_permutation_circuit
    assert circuit_a.to_tableau() == circuit_b.to_tableau()

    # cover an edge case
    circuit_a = circuits.with_remapped_qubits(stim.Circuit("MPP X1*!Y2 \n M !4"), {2: 3})
    circuit_b = stim.Circuit("MPP X1*!Y3 \n M !4")
    assert circuit_a == circuit_b

    # a stim-wrapping circuit input is remapped and returned as the same wrapping type
    wrapped_remapped = circuits.with_remapped_qubits(StimCircuitWrapper(circuit), qubit_map)
    assert isinstance(wrapped_remapped, StimCircuitWrapper)
    assert wrapped_remapped.stim_circuit == circuits.with_remapped_qubits(circuit, qubit_map)


def test_qubit_remap_edge_cases() -> None:
    """MPAD padding bits are not qubits, and qubit maps must be injective."""

    mpad = stim.Circuit("MPAD 1 0")
    assert circuits.with_remapped_qubits(mpad, [1, 0]) == mpad
    with pytest.raises(ValueError, match="injective"):
        circuits.with_remapped_qubits(stim.Circuit("H 0 1"), [0, 0])
