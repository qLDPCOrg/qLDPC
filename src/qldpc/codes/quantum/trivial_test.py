# SPDX-License-Identifier: Apache-2.0

"""Unit tests for trivial.py."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli


def test_trivial_code() -> None:
    """The trivial code is, well, trivial."""
    code = codes.TrivialCode(3, num_stabs_z=2)
    assert code.num_checks_x == 0
    assert code.num_checks_z == 2
    assert code.get_distance() == 1

    code = codes.TrivialCode(4, 2, 1)
    assert code.num_checks_x == 2
    assert code.num_checks_z == 1
    assert code.get_distance() == 1

    # A single stabilizer count is enough for a self-dual code; num_stabs_z mirrors num_stabs_x.
    code = codes.TrivialCode(6, 1, gauge_dimension=1, self_dual=True)
    assert np.array_equal(code.get_stabilizer_ops(Pauli.X), code.get_stabilizer_ops(Pauli.Z))
    assert code.gauge_dimension == 1
    assert code.get_distance() == 1

    # Qubits are laid out as [logical | gauge | X-stabilizer | Z-stabilizer].  With n=6, k=3, one
    # gauge qubit, and one stabilizer of each type, the gauge qubit must land at index 3 (after the
    # three logical qubits), not inside the logical block.
    assert code.dimension == 3
    expected_checks = [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 1]]
    assert np.array_equal(code.matrix_x, expected_checks)
    assert np.array_equal(code.matrix_z, expected_checks)

    with pytest.raises(ValueError, match="equal number"):
        codes.TrivialCode(4, 2, 1, self_dual=True)

    with pytest.raises(ValueError, match="too small"):
        codes.TrivialCode(1, num_stabs_z=2)
