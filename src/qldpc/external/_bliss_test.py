# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional pybliss integration."""

from __future__ import annotations

import builtins
import importlib.util
import itertools
import sys
import types
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from qldpc.conftest import run_python
from qldpc.external import _bliss


class _FakeGraph:
    """Stand-in for a pybliss graph, which finds its automorphisms by brute force."""

    def __init__(self, colors: npt.NDArray[np.int32], edges: set[tuple[int, int]]) -> None:
        self.colors = colors
        self.edges = edges

    def find_automorphisms(self, stats: object, report: Callable[[int, np.ndarray], None]) -> None:
        """Report every nontrivial permutation within color classes that preserves edges."""
        size = len(self.colors)
        classes = [np.flatnonzero(self.colors == color) for color in np.unique(self.colors)]
        for images in itertools.product(*(itertools.permutations(cls) for cls in classes)):
            perm = np.empty(size, dtype=np.uint32)
            for cls, image in zip(classes, images):
                perm[cls] = image
            edges = {(min(perm[a], perm[b]), max(perm[a], perm[b])) for a, b in self.edges}
            if not np.array_equal(perm, np.arange(size)) and edges == self.edges:
                report(size, perm)


@pytest.fixture
def fake_pybliss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a controlled stand-in for the optional pybliss package."""

    def graph_from_numpy(
        num_vertices: int, edges: npt.NDArray[np.int32], colors: npt.NDArray[np.int32]
    ) -> _FakeGraph:
        assert len(colors) == num_vertices
        return _FakeGraph(colors, {(min(map(int, edge)), max(map(int, edge))) for edge in edges})

    monkeypatch.setitem(
        sys.modules,
        "pybliss",
        types.SimpleNamespace(graph_from_numpy=graph_from_numpy, Stats=lambda: None),
    )


def test_get_matrix_automorphism_generators(fake_pybliss: None) -> None:
    """Automorphisms permute rows and columns, preserving the values of matrix entries."""
    # the stand-in reports every automorphism, so these are full automorphism groups
    assert _bliss.get_matrix_automorphism_generators(np.array([[1, 1]])) == [[0, 2, 1]]
    assert _bliss.get_matrix_automorphism_generators(np.array([[1, 2]])) == []

    # the only nontrivial automorphism swaps rows 0, 1 and columns 1, 2; entry colors must differ
    # from the colors of rows and columns, or a spurious automorphism appears
    three_valued_matrix = np.array([[0, 2, 3], [0, 3, 2], [1, 0, 0]])
    assert _bliss.get_matrix_automorphism_generators(three_valued_matrix) == [[1, 0, 2, 3, 5, 4]]


def test_real_pybliss_package() -> None:
    """Check the published pybliss package whenever it is installed.

    The checks run in a subprocess, which runs an empty script if the package is absent, so the
    statement coverage of this module does not depend on the optional installation.
    """
    installed = importlib.util.find_spec("pybliss") is not None
    run_python(_REAL_PACKAGE_CHECKS if installed else "")


_REAL_PACKAGE_CHECKS = """
import numpy as np
from qldpc import codes
from qldpc.external import _bliss

matrix = codes.RingCode(6).matrix
generators = _bliss.get_matrix_automorphism_generators(matrix)
assert generators
for perm in generators:
    assert np.array_equal(matrix[perm[:6]][:, [col - 6 for col in perm[6:]]], matrix)
assert codes.RingCode(6).get_matrix_automorphism_group().order == 12
three_valued_matrix = np.array([[0, 2, 3], [0, 3, 2], [1, 0, 0]])
assert _bliss.get_matrix_automorphism_generators(three_valued_matrix) == [[1, 0, 2, 3, 5, 4]]
"""


def test_missing_pybliss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain how to install pybliss if it is missing, but not if a dependency of it is."""
    monkeypatch.setitem(sys.modules, "pybliss", None)
    with pytest.raises(ModuleNotFoundError, match=r"qldpc\[bliss\]"):
        _bliss.get_matrix_automorphism_generators(np.array([[1, 1]]))

    monkeypatch.delitem(sys.modules, "pybliss")
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
