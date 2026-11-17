# SPDX-License-Identifier: Apache-2.0

"""The workloads keep the parity checks of quantum Tanner codes from earlier releases of qldpc."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np

from qldpc import abstract, codes

# Scripts in distance_tuning/ import each other as top-level modules.  mypy knows these files only
# as experiments.distance_tuning.*, and would silently treat a bare "import workloads" as untyped.
if TYPE_CHECKING:
    from experiments.distance_tuning import workloads
else:
    import workloads


def test_row_reduced_qt_code() -> None:
    """The row-reduced checks of a quantum Tanner code are those of earlier versions of qldpc.

    The expected supports of the checks were computed with QTCode before it built minimum-weight
    local checks.
    """
    shift = abstract.CyclicGroup(3).generators[0]
    code = codes.QTCode([shift, ~shift], [shift, ~shift, shift**3], [[1, 1]], [[1, 0, 1]])
    row_reduced_code = workloads.get_row_reduced_qt_code(code)
    supports_x = [tuple(np.flatnonzero(row)) for row in row_reduced_code.matrix_x.view(np.ndarray)]
    supports_z = [tuple(np.flatnonzero(row)) for row in row_reduced_code.matrix_z.view(np.ndarray)]
    assert supports_x == [
        (0, 2, 3, 5),
        (1, 4),
        (4, 6, 8, 9),
        (2, 7),
        (6, 10, 11, 13),
        (9, 12),
        (0, 12, 14, 15),
        (5, 11),
        (7, 8, 14, 16),
        (15, 17),
        (1, 3, 10, 17),
        (13, 16),
    ]
    assert supports_z == [
        (0, 3, 15, 17),
        (6, 8, 13, 16),
        (1, 4, 6, 10),
        (0, 2, 7, 14),
        (8, 9, 12, 14),
        (3, 5, 10, 11),
    ]


def test_qt7_workload() -> None:
    """The quantum Tanner code of the qt7 workload has the parity checks of earlier benchmarks.

    The expected hash is that of the parity check matrices of the code that QTCode built before it
    built minimum-weight local checks.
    """
    code = workloads.get_code("qt7")
    assert isinstance(code, codes.CSSCode)
    matrices = [np.asarray(matrix, dtype=np.uint8) for matrix in [code.matrix_x, code.matrix_z]]
    digest = hashlib.sha256(b"".join(matrix.tobytes() for matrix in matrices)).hexdigest()
    assert [matrix.shape for matrix in matrices] == [(168, 343), (168, 343)]
    assert digest == "80d732d32e4a7b0fdf2d9897ac07684cc8809b4fb629ae3e966654863bd62520"
