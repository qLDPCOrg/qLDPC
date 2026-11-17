# SPDX-License-Identifier: Apache-2.0

"""Optional integration with the bliss graph-automorphism tool, through the pybliss package."""

from __future__ import annotations

import importlib.util
import types
from typing import Any

import numpy as np
import numpy.typing as npt


def get_matrix_automorphism_generators(matrix: npt.NDArray[np.int_]) -> list[list[int]]:
    """Generators of the automorphism group of a matrix.

    An automorphism of an ``m × n`` matrix ``H`` is a permutation ``p`` of ``range(m + n)`` that
    maps rows to rows and columns to columns while leaving ``H`` invariant, in the sense that
    ``H[p_rows][:, p_cols] == H`` for ``p_rows = p[:m]`` and ``p_cols = p[m:] - m``.  These
    automorphisms are computed as automorphisms of a colored graph with a vertex for every row,
    column, and nonzero entry of ``H``, where the vertex of each entry is colored by its value.

    References:

    - https://users.aalto.fi/~tjunttil/bliss/
    """
    pybliss = _get_pybliss()
    matrix = np.asarray(matrix)
    num_rows, num_cols = matrix.shape
    graph = _get_matrix_graph(pybliss, matrix, np.unique(matrix[np.nonzero(matrix)]))

    generators: list[list[int]] = []
    graph.find_automorphisms(
        pybliss.Stats(),
        lambda _, automorphism: generators.append(automorphism[: num_rows + num_cols].tolist()),
    )
    return generators


def get_matrix_isomorphism(
    matrix_a: npt.NDArray[np.int_], matrix_b: npt.NDArray[np.int_]
) -> list[int] | None:
    """An isomorphism from one matrix to another, if there is one, and None otherwise.

    An isomorphism from an ``m × n`` matrix ``A`` to a matrix ``B`` of the same shape is a
    permutation ``p`` of ``range(m + n)`` that maps rows to rows and columns to columns, such that
    ``A[i, j] == B[p[i], p[m + j] - m]`` for every row ``i`` and column ``j``.  Bliss finds it by
    comparing the canonical forms of the colored graphs of ``get_matrix_automorphism_generators``,
    or, if the nonzero entries of each row of both matrices are equal, of smaller graphs that color
    each row by its value and join it to the columns of its nonzero entries.

    References:

    - https://users.aalto.fi/~tjunttil/bliss/
    """
    pybliss = _get_pybliss()
    matrix_a, matrix_b = np.asarray(matrix_a), np.asarray(matrix_b)
    if matrix_a.shape != matrix_b.shape:
        return None
    matrices = (matrix_a, matrix_b)

    # Each graph is relabeled into its canonical form, so equal canonical forms give an isomorphism
    # that relabels the first graph canonically and then undoes the relabeling of the second.  An
    # isomorphism maps each row to a row with the same nonzero entries, so if every row of both
    # matrices has one nonzero value, colors of rows can stand in for the vertices of the entries.
    row_values = [matrix.max(axis=1, initial=0) for matrix in matrices]
    if all(
        np.all((matrix == 0) | (matrix == matrix_values[:, None]))
        for matrix, matrix_values in zip(matrices, row_values)
    ):
        values = np.unique(np.concatenate(row_values))
        graphs = [
            _get_row_colored_graph(pybliss, matrix, np.searchsorted(values, matrix_values))
            for matrix, matrix_values in zip(matrices, row_values)
        ]
    else:
        values = np.unique(np.concatenate([matrix[np.nonzero(matrix)] for matrix in matrices]))
        graphs = [_get_matrix_graph(pybliss, matrix, values) for matrix in matrices]
    labelings = [
        np.asarray(graph.get_permutation_to_canonical_form(pybliss.Stats())) for graph in graphs
    ]
    if graphs[0].permute(labelings[0]).cmp(graphs[1].permute(labelings[1])) != 0:
        return None
    return np.argsort(labelings[1])[labelings[0]][: sum(matrix_a.shape)].tolist()


def is_installed() -> bool:
    """Return whether the optional pybliss package is discoverable."""
    return importlib.util.find_spec("pybliss") is not None


def _get_matrix_graph(
    pybliss: types.ModuleType, matrix: npt.NDArray[np.int_], values: npt.NDArray[np.int_]
) -> Any:
    """The colored graph of a matrix, whose automorphisms and isomorphisms are those of the matrix.

    The graph has a vertex for every row, column, and nonzero entry of the matrix, and colors the
    vertex of each entry by the index of its value in the sorted array ``values``.  Entry colors
    differ from those of rows and columns, or spurious automorphisms appear.
    """
    num_rows, num_cols = matrix.shape
    rows, cols = np.nonzero(matrix)
    entries = num_rows + num_cols + np.arange(rows.size)
    colors = np.concatenate(
        [
            np.zeros(num_rows, dtype=np.int32),
            np.ones(num_cols, dtype=np.int32),
            2 + np.searchsorted(values, matrix[rows, cols]).astype(np.int32),
        ]
    )
    edges = np.concatenate(
        [np.column_stack([rows, entries]), np.column_stack([entries, num_rows + cols])]
    )
    return _build_graph(pybliss, colors, edges)


def _get_row_colored_graph(
    pybliss: types.ModuleType, matrix: npt.NDArray[np.int_], row_colors: npt.NDArray[np.int_]
) -> Any:
    """The colored graph of a matrix whose rows are colored, as by the values of their entries.

    The graph has a vertex for every row and column, and an edge for every nonzero entry.  Row ``i``
    has the color ``1 + row_colors[i]``, and the columns have the color zero.
    """
    rows, cols = np.nonzero(matrix)
    colors = np.concatenate([1 + row_colors, np.zeros(matrix.shape[1], dtype=int)])
    return _build_graph(pybliss, colors, np.column_stack([rows, len(matrix) + cols]))


def _build_graph(
    pybliss: types.ModuleType, colors: npt.NDArray[np.int_], edges: npt.NDArray[np.int_]
) -> Any:
    """A pybliss graph with the given vertex colors and edges, each a row of two vertices.

    This builds the graph that ``pybliss.graph_from_numpy`` builds, from Python integers, which
    pybliss takes about 3 times faster than the NumPy integers that ``graph_from_numpy`` passes.
    """
    graph = pybliss.Graph(len(colors))
    for vertex, neighbor in edges.tolist():
        graph.add_edge(vertex, neighbor)
    for vertex, color in enumerate(colors.tolist()):
        graph.change_color(vertex, color)
    return graph


def _get_pybliss() -> types.ModuleType:
    """Import the optional pybliss package or raise an actionable error."""
    try:
        import pybliss
    except ModuleNotFoundError as error:
        if error.name != "pybliss":
            raise
        raise ModuleNotFoundError(
            "Matrix automorphisms and isomorphisms require the optional 'pybliss' package.  Install it with"
            " `pip install 'qldpc[bliss]'`."
        ) from error
    return pybliss
