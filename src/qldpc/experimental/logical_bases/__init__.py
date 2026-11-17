# SPDX-License-Identifier: Apache-2.0

"""Light bases of logical operators that share few qubits, for binary CSS codes.

EXPERIMENTAL: this subpackage is under active development; its public API is unstable and may
change without notice or deprecation.

Logical operators that share no physical qubits can be acted on at the same time, for example by
parallel lattice surgery, so a basis whose operators are light and overlap little allows more
parallel logical operations.  This subpackage checks bases and measures their weights and overlaps
among operators of the same Pauli type, and finds the lightest bases of each Pauli type, which bound
the weights of every basis from below.

Public API:
    evaluate_logical_ops, LogicalOpsMetrics    — check a basis and measure weights and overlaps
    get_lightest_logical_basis, LightestLogicalBasis
        — the smallest weights that a basis of one Pauli type can have, with a basis that has them
"""

from __future__ import annotations

from .evaluation import LogicalOpsMetrics, evaluate_logical_ops
from .lightest_basis import LightestLogicalBasis, get_lightest_logical_basis

__all__ = [
    "LightestLogicalBasis",
    "LogicalOpsMetrics",
    "evaluate_logical_ops",
    "get_lightest_logical_basis",
]
