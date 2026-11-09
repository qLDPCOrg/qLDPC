# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact fault distance of annotated circuits."""

from __future__ import annotations

import numpy as np
import pytest
import stim

from qldpc import circuits
from qldpc.codes.distance import DistanceMethod


@pytest.mark.parametrize("method", ["brouwer_zimmermann", "brute_force"])
@pytest.mark.parametrize("measurement, expected_distance", [(-7, 3), (-4, 4)])
def test_fault_distance(measurement: int, expected_distance: int, method: DistanceMethod) -> None:
    """Undetected faults flip whole blocks of three or four qubits; only one block is logical."""
    circuit = stim.Circuit("""
        R 0 1 2 3 4 5 6
        X_ERROR(0.01) 0 1 2 3 4 5 6
        M 0 1 2 3 4 5 6
        DETECTOR rec[-7] rec[-6]
        DETECTOR rec[-6] rec[-5]
        DETECTOR rec[-4] rec[-3]
        DETECTOR rec[-3] rec[-2]
        DETECTOR rec[-2] rec[-1]
        DETECTOR rec[-4] rec[-2]
    """)
    circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(measurement)], 0)
    assert circuits.get_fault_distance(circuit, method=method) == expected_distance


def test_fault_distance_without_logical_faults() -> None:
    """No undetected observable flip has the code distance convention np.nan."""
    circuit = stim.Circuit("R 0\nX_ERROR(0.01) 0\nM 0\nDETECTOR rec[-1]")
    assert np.isnan(circuits.get_fault_distance(circuit))
