# SPDX-License-Identifier: Apache-2.0

"""Compile galois's GF(2) arithmetic in a task process before its measured call.

galois compiles its arithmetic the first time that a process uses it, and does not cache it on disk:
creating the field ``galois.GF(2)`` compiles ``numba.jit`` functions, and arithmetic such as
division and row reduction compiles ``numba.vectorize`` ufuncs.  So a measured call that reaches
this arithmetic compiles in every process, and the runner's retry cannot remove that compilation.  A
measured call that compiles galois's ``numba.jit`` functions gets a flagged row
(``compiled_in_timing``), since its retry compiles them again, but numba does not report the ufuncs
(see ``runner.py``).

The certification experiment's ``ground_truth`` uses galois in ``counting.residual_weight_table``,
so its task process first calls ``compile_arithmetic``.  This creates the field and does a small
calculation through galois's public API, untimed, which compiles the galois code that
``residual_weight_table`` uses.  The certification experiment's acceptance test AT2 checks, through
the runner, that no ``numba.jit`` compilation is then left in ``ground_truth``'s measured call.  The
ranking experiment's measured calls reach galois's ufuncs through qLDPC's residual-weight code and
through ``ranking/consumer.py``, and can compile them unreported.
"""

from __future__ import annotations

import galois
import numpy as np


def compile_arithmetic() -> None:
    """Row-reduce a small GF(2) matrix, find its null space and rank, and divide in GF(2)."""
    field = galois.GF(2)
    matrix = field([[1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 1]])
    matrix.row_reduce()
    matrix.null_space()
    np.linalg.matrix_rank(matrix)
    _ = field(1) / field(1)
