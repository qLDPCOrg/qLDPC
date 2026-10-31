# SPDX-License-Identifier: Apache-2.0

"""Unit tests for common.py."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import circuits, codes
from qldpc.circuits import common
from qldpc.math import symplectic_conjugate
from qldpc.objects import Pauli


def test_pauli_product_measurements(pytestconfig: pytest.Config) -> None:
    """get_pauli_product_measurements correctly measures the syndrome of Pauli errors."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code = codes.FiveQubitCode()
    encoder = circuits.get_encoding_circuit(code)
    stabilizers = code.get_stabilizer_ops()

    error_vec = code.field.Random(len(code) * 2)
    error_ops = stim.Circuit()
    for qubit in range(len(code)):
        xx_zz = (error_vec[qubit], error_vec[qubit + len(code)])
        error_ops.append(str(Pauli(xx_zz)), qubit)

    measurements = circuits.get_pauli_product_measurements(stabilizers)
    outcomes = (encoder + error_ops + measurements).reference_sample()
    syndrome = stabilizers @ symplectic_conjugate(error_vec)
    assert np.array_equal(outcomes.astype(int), syndrome)


def test_pauli_product_measurement_edge_cases() -> None:
    """Pauli-product measurements of symplectic rows, with signs and qubit labels."""
    assert circuits.get_pauli_product_measurements([[1, 0, 0, 1]]) == stim.Circuit("MPP X0*Z1")

    # qubit labels place the measured Pauli strings on the given qubits
    strings = [stim.PauliString("-XIZ"), stim.PauliString("_Y_")]
    qubits = [7, 2, 4]
    unlabeled = circuits.get_pauli_product_measurements(strings)
    labeled = circuits.get_pauli_product_measurements(strings, qubits)
    assert labeled == circuits.with_remapped_qubits(unlabeled, qubits)

    with pytest.raises(ValueError, match="all-identity"):
        circuits.get_pauli_product_measurements([[0, 0, 0, 0]])
    with pytest.raises(ValueError, match="2 qubits, but 3 qubit labels"):
        circuits.get_pauli_product_measurements([[1, 0, 0, 1]], qubits=[3, 4, 5])
    with pytest.raises(ValueError, match="must be distinct"):
        circuits.get_pauli_product_measurements([[1, 0, 0, 1]], qubits=[3, 3])


def test_restriction() -> None:
    """_restrict_to_qubits passes through for qubit codes and raises for qudit codes."""

    @common._restrict_to_qubits
    def get_empty_circuit(code: codes.QuditCode) -> stim.Circuit:
        return stim.Circuit()

    get_empty_circuit(codes.SurfaceCode(2, field=2))  # this does not raise an error

    with pytest.raises(ValueError, match="only supported for qubit codes"):
        get_empty_circuit(codes.SurfaceCode(2, field=3))


def test_deprecated_attribute_lookup() -> None:
    """The module-level lookup of deprecated names rejects other names."""
    with pytest.raises(AttributeError, match="has no attribute 'not_a_real_export'"):
        common.not_a_real_export  # type: ignore[attr-defined]  # noqa: B018
