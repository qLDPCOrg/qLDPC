# SPDX-License-Identifier: Apache-2.0

"""Tests for the optional pybliss integration."""

from __future__ import annotations

import importlib.util
import itertools
import sys
import types
import unittest.mock
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
import pytest

from qldpc.conftest import fail_import, run_python
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

    def get_permutation_to_canonical_form(self, stats: object) -> npt.NDArray[np.uint32]:
        """The relabeling, onto vertices sorted by color, whose edges are least."""
        sorted_colors = np.sort(self.colors)
        classes = [np.flatnonzero(self.colors == color) for color in np.unique(self.colors)]
        targets = [np.flatnonzero(sorted_colors == color) for color in np.unique(self.colors)]
        labelings = []
        for images in itertools.product(*(itertools.permutations(target) for target in targets)):
            perm = np.empty(len(self.colors), dtype=np.uint32)
            for cls, image in zip(classes, images):
                perm[cls] = image
            labelings.append(perm)
        return min(labelings, key=lambda perm: sorted(self.permute(perm).edges))

    def permute(self, perm: npt.NDArray[np.uint32]) -> _FakeGraph:
        """The graph whose vertex perm[v] has the color and edges of vertex v of this graph."""
        colors = np.empty_like(self.colors)
        colors[perm] = self.colors
        edges = {(min(perm[a], perm[b]), max(perm[a], perm[b])) for a, b in self.edges}
        return _FakeGraph(colors, {(int(a), int(b)) for a, b in edges})

    def cmp(self, other: _FakeGraph) -> int:
        """Zero if the graphs are equal, and one otherwise."""
        equal = np.array_equal(self.colors, other.colors) and self.edges == other.edges
        return 0 if equal else 1


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


def test_get_matrix_isomorphism(fake_pybliss: None) -> None:
    """Isomorphisms map rows to rows and columns to columns, preserving the values of entries.

    The second matrix permutes the rows of the first by a 3-cycle and its columns by a 4-cycle, and
    the first has no automorphisms but the identity, so the isomorphism is the inverse of these
    cycles, which differs from them.
    """
    matrix_a, matrix_b = _get_cycled_matrices()
    perm = _bliss.get_matrix_isomorphism(matrix_a, matrix_b)
    assert perm is not None
    for (row, col), value in np.ndenumerate(matrix_a):
        assert matrix_b[perm[row], perm[3 + col] - 3] == value
    assert _bliss.get_matrix_isomorphism(matrix_a, np.where(matrix_a == 2, 1, matrix_a)) is None
    assert _bliss.get_matrix_isomorphism(matrix_a, matrix_b[:2]) is None
    assert _bliss.get_matrix_isomorphism(np.array([[1]]), np.array([[2]])) is None


def test_is_installed() -> None:
    """Report whether the optional pybliss package is discoverable."""
    with unittest.mock.patch("importlib.util.find_spec", return_value=unittest.mock.Mock()):
        assert _bliss.is_installed()
    with unittest.mock.patch("importlib.util.find_spec", return_value=None):
        assert not _bliss.is_installed()


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
assert _bliss.is_installed()
matrix_a = np.array([[1, 2, 0, 0], [0, 1, 1, 0], [2, 0, 0, 0]])
matrix_b = matrix_a[[1, 2, 0]][:, [1, 2, 3, 0]]
perm = _bliss.get_matrix_isomorphism(matrix_a, matrix_b)
assert np.array_equal(matrix_b[perm[:3]][:, [col - 3 for col in perm[3:]]], matrix_a)
assert _bliss.get_matrix_isomorphism(np.array([[1]]), np.array([[2]])) is None
assert _bliss.get_matrix_isomorphism(three_valued_matrix, three_valued_matrix % 3) is None
"""


def test_missing_pybliss(monkeypatch: pytest.MonkeyPatch) -> None:
    """Explain how to install pybliss if it is missing, but not if a dependency of it is."""
    monkeypatch.setitem(sys.modules, "pybliss", None)
    with pytest.raises(ModuleNotFoundError, match=r"qldpc\[bliss\]"):
        _bliss.get_matrix_automorphism_generators(np.array([[1, 1]]))

    monkeypatch.delitem(sys.modules, "pybliss")
    fail_import(
        monkeypatch, "pybliss", ModuleNotFoundError("No module named 'other'", name="other")
    )
    with pytest.raises(ModuleNotFoundError, match="other"):
        _bliss.get_matrix_automorphism_generators(np.array([[1, 1]]))


def _get_cycled_matrices() -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    """A 3 x 4 matrix with no nontrivial automorphisms, and its rows and columns cycled."""
    matrix = np.array([[1, 2, 0, 0], [0, 1, 1, 0], [2, 0, 0, 0]])
    return matrix, matrix[[1, 2, 0]][:, [1, 2, 3, 0]]
