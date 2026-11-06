# SPDX-License-Identifier: Apache-2.0

"""Unit tests for paulis.py."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import abstract, objects


def test_pauli() -> None:
    """Pauli operator capabilities."""
    for string in ["I", "X", "Y", "Z"]:
        assert str(objects.Pauli.from_string(string)) == string
    with pytest.raises(ValueError, match="Invalid Pauli operator"):
        objects.Pauli.from_string("Q")

    # from_string is case-sensitive, since QuditCode.from_strings relies on it
    with pytest.raises(ValueError, match="Invalid Pauli operator"):
        objects.Pauli.from_string("x")

    # Pauli.coerce_xz accepts Pauli.X, Pauli.Z, and case-insensitive "X" or "Z" strings
    xz_inputs: dict[objects.PauliXZ, tuple[objects.PauliXZLike, ...]] = {
        objects.Pauli.X: (objects.Pauli.X, "X", "x"),
        objects.Pauli.Z: (objects.Pauli.Z, "Z", "z"),
    }
    for expected_pauli, values in xz_inputs.items():
        for value in values:
            assert objects.Pauli.coerce_xz(value) is expected_pauli
    for invalid_value in [objects.Pauli.Y, "y", 0]:
        with pytest.raises(ValueError, match=r"Pauli\.X or Pauli\.Z"):
            objects.Pauli.coerce_xz(invalid_value)  # type: ignore[arg-type]

    assert ~objects.Pauli.Z == objects.Pauli.Z.swap_xz() == objects.Pauli.X
    assert ~objects.Pauli.X == objects.Pauli.X.swap_xz() == objects.Pauli.Z
    assert ~objects.Pauli.Y == objects.Pauli.Y
    assert ~objects.Pauli.I == objects.Pauli.I
    with pytest.raises(ValueError, match=r"Pauli\.X and Pauli\.Z"):
        objects.Pauli.Y.swap_xz()

    paulis = [objects.Pauli.I, objects.Pauli.Z, objects.Pauli.X, objects.Pauli.Y]
    table = [[paulis.index(pp * qq) for qq in paulis] for pp in paulis]
    group = abstract.Group.product(abstract.CyclicGroup(2), repeat=2)
    assert np.array_equal(table, group.table)

    assert objects.Pauli.X.index != objects.Pauli.Z.index
    for pauli in [objects.Pauli.I, objects.Pauli.Y]:
        with pytest.raises(TypeError, match="have an index"):
            assert pauli.index


def test_qudit_operator() -> None:
    """Qudit operator capabilities."""
    assert objects.QuditPauli((0, 0)) == objects.QuditPauli()
    assert objects.QuditPauli((0, 1)) == ~objects.QuditPauli((1, 0))
    assert -objects.QuditPauli((0, 1)) == objects.QuditPauli((0, -1))
    for op in ["I", "Y(1)", "X(1)*Z(2)"]:
        assert str(objects.QuditPauli.from_string(op)) == op
    for op in ["a*b*c", "a(1)", "*", "X(1)*", "", "X(²)"]:
        with pytest.raises(ValueError, match="Invalid qudit operator"):
            objects.QuditPauli.from_string(op)
