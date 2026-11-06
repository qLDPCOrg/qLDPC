# SPDX-License-Identifier: Apache-2.0

"""Optional integration with the bliss graph-automorphism tool, through the pybliss package."""

from __future__ import annotations

import importlib.util
import types

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
    rows, cols = np.nonzero(matrix)
    values = matrix[rows, cols]
    entries = num_rows + num_cols + np.arange(values.size)

    colors = np.concatenate(
        [
            np.zeros(num_rows, dtype=np.int32),
            np.ones(num_cols, dtype=np.int32),
            2 + np.unique(values, return_inverse=True)[1].astype(np.int32),
        ]
    )
    edges = np.concatenate(
        [np.column_stack([rows, entries]), np.column_stack([entries, num_rows + cols])]
    ).astype(np.int32)
    graph = pybliss.graph_from_numpy(colors.size, edges, colors)

    generators: list[list[int]] = []
    graph.find_automorphisms(
        pybliss.Stats(),
        lambda _, automorphism: generators.append(automorphism[: num_rows + num_cols].tolist()),
    )
    return generators


def get_binary_matrix_automorphism_generators(
    matrix: npt.NDArray[np.int_], row_colors: npt.ArrayLike
) -> list[list[int]]:
    """Generators of the automorphism group of a binary matrix whose rows are colored.

    These automorphisms are those of ``get_matrix_automorphism_generators`` that also map each row
    to a row of the same color.  Bliss finds them as automorphisms of the bipartite graph of the
    matrix, which has a vertex for every row and column.  This graph is smaller than the graph of
    ``get_matrix_automorphism_generators``.

    Args:
        matrix: A binary matrix.
        row_colors: A color for each row of the matrix, such as an integer.

    Returns:
        Generators of the automorphism group, in the format of
        ``get_matrix_automorphism_generators``.

    References:

    - https://users.aalto.fi/~tjunttil/bliss/
    """
    pybliss = _get_pybliss()
    matrix = np.asarray(matrix)
    num_rows, num_cols = matrix.shape
    row_color_indices = np.unique(np.asarray(row_colors), return_inverse=True)[1]
    colors = np.concatenate(
        [row_color_indices, np.full(num_cols, row_color_indices.max(initial=-1) + 1)]
    ).astype(np.int32)
    rows, cols = np.nonzero(matrix)
    edges = np.column_stack([rows, num_rows + cols]).astype(np.int32)
    graph = pybliss.graph_from_numpy(colors.size, edges, colors)
    generators: list[list[int]] = []
    graph.find_automorphisms(
        pybliss.Stats(), lambda _, automorphism: generators.append(automorphism.tolist())
    )
    return generators


def is_installed() -> bool:
    """Return whether the optional pybliss package is discoverable."""
    return importlib.util.find_spec("pybliss") is not None


def _get_pybliss() -> types.ModuleType:
    """Import the optional pybliss package or raise an actionable error."""
    try:
        import pybliss
    except ModuleNotFoundError as error:
        if error.name != "pybliss":
            raise
        raise ModuleNotFoundError(
            "Matrix automorphisms require the optional 'pybliss' package.  Install it with"
            " `pip install 'qldpc[bliss]'`."
        ) from error
    return pybliss
