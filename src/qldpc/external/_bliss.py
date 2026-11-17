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
    comparing the canonical forms of the colored graphs of ``get_matrix_automorphism_generators``.

    References:

    - https://users.aalto.fi/~tjunttil/bliss/
    """
    pybliss = _get_pybliss()
    matrix_a, matrix_b = np.asarray(matrix_a), np.asarray(matrix_b)
    if matrix_a.shape != matrix_b.shape:
        return None
    values = np.unique(
        np.concatenate([matrix[np.nonzero(matrix)] for matrix in (matrix_a, matrix_b)])
    )

    # Each graph is relabeled into its canonical form, so equal canonical forms give an isomorphism
    # that relabels the first graph canonically and then undoes the relabeling of the second.
    graphs = [_get_matrix_graph(pybliss, matrix, values) for matrix in (matrix_a, matrix_b)]
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
    ).astype(np.int32)
    return pybliss.graph_from_numpy(colors.size, edges, colors)


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
