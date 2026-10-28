# SPDX-License-Identifier: Apache-2.0

"""Optional integration with the bliss graph-automorphism tool, through the pybliss package."""

from __future__ import annotations

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
