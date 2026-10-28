# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional pybliss integration."""

from __future__ import annotations

import builtins
import sys
import types
import unittest.mock

import numpy as np
import pytest

from qldpc import codes
from qldpc.external import _bliss


def test_get_matrix_automorphism_generators() -> None:
    """Automorphisms permute rows and columns, preserving the values of matrix entries."""
    pytest.importorskip("pybliss")
    matrix = codes.RingCode(6).matrix
    generators = _bliss.get_matrix_automorphism_generators(matrix)
    assert generators
    for perm in generators:
        perm_rows, perm_cols = perm[:6], [col - 6 for col in perm[6:]]
        assert np.array_equal(matrix[perm_rows][:, perm_cols], matrix)

    assert _bliss.get_matrix_automorphism_generators(np.array([[1, 1]])) == [[0, 2, 1]]
    assert _bliss.get_matrix_automorphism_generators(np.array([[1, 2]])) == []

    # the only nontrivial automorphism swaps rows 0, 1 and columns 1, 2; entry colors must differ
    # from the colors of rows and columns, or a spurious automorphism appears
    three_valued_matrix = np.array([[0, 2, 3], [0, 3, 2], [1, 0, 0]])
    assert _bliss.get_matrix_automorphism_generators(three_valued_matrix) == [[1, 0, 2, 3, 5, 4]]


def test_missing_pybliss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain how to install pybliss if it is missing, but not if a dependency of it is."""
    with (
        unittest.mock.patch.dict(sys.modules, {"pybliss": None}),
        pytest.raises(ModuleNotFoundError, match=r"qldpc\[bliss\]"),
    ):
        _bliss.get_matrix_automorphism_generators(np.array([[1, 1]]))

    original_import = builtins.__import__

    def import_pybliss(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> types.ModuleType:
        if name == "pybliss":
            raise ModuleNotFoundError("No module named 'other'", name="other")
        return original_import(name, globals_, locals_, fromlist, level)

    assert import_pybliss("types") is types
    monkeypatch.setattr(builtins, "__import__", import_pybliss)
    with pytest.raises(ModuleNotFoundError, match="other"):
        _bliss.get_matrix_automorphism_generators(np.array([[1, 1]]))
