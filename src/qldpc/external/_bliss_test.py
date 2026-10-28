# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional pybliss integration."""

from __future__ import annotations

import sys
import unittest.mock

import numpy as np
import pytest

from qldpc import codes
from qldpc.external import _bliss


def test_get_matrix_automorphisms() -> None:
    """Automorphisms permute rows and columns, preserving the values of matrix entries."""
    pytest.importorskip("pybliss")
    matrix = codes.RingCode(6).matrix
    generators = _bliss.get_matrix_automorphisms(matrix)
    assert generators
    for perm in generators:
        perm_rows, perm_cols = perm[:6], [col - 6 for col in perm[6:]]
        assert np.array_equal(matrix[perm_rows][:, perm_cols], matrix)

    assert _bliss.get_matrix_automorphisms(np.array([[1, 1]])) == [[0, 2, 1]]
    assert _bliss.get_matrix_automorphisms(np.array([[1, 2]])) == []


def test_missing_pybliss() -> None:
    """Explain how to install pybliss if it is missing."""
    with (
        unittest.mock.patch.dict(sys.modules, {"pybliss": None}),
        pytest.raises(ModuleNotFoundError, match=r"qldpc\[bliss\]"),
    ):
        _bliss.get_matrix_automorphisms(np.array([[1, 1]]))
