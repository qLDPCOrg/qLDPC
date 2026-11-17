# SPDX-License-Identifier: Apache-2.0

"""Check bases of logical operators of binary CSS codes, and find light ones.

EXPERIMENTAL: this subpackage is under active development; its public API is unstable and may
change without notice or deprecation.

Logical operators that share no physical qubits can be acted on at the same time, for example by
parallel lattice surgery, so a basis whose operators are light and overlap little allows more
parallel logical operations.  This subpackage checks bases, and measures the weights of their
operators and the overlaps between operators of the same Pauli type.  It also finds a lightest basis
of each Pauli type, whose weights bound those of every basis of that type (and, for a subsystem
code, of the same kind, dressed or bare) from below.  Finally, it searches for complete bases, whose
X-type and Z-type operators pair up, with low weights.

Public API:
    evaluate_logical_ops, LogicalBasisMetrics            — check a basis, measure its overlaps
    LogicalOpsMetrics                                    — the metrics of one Pauli type
    get_lightest_logical_basis, LightestLogicalBasis     — a lightest basis of one Pauli type
    optimize_logical_basis, OptimizedLogicalBasis        — a complete basis with low weights
"""

from __future__ import annotations

from .evaluation import LogicalBasisMetrics, LogicalOpsMetrics, evaluate_logical_ops
from .lightest_basis import LightestLogicalBasis, get_lightest_logical_basis
from .optimization import OptimizedLogicalBasis, optimize_logical_basis

__all__ = [
    "LightestLogicalBasis",
    "LogicalBasisMetrics",
    "LogicalOpsMetrics",
    "OptimizedLogicalBasis",
    "evaluate_logical_ops",
    "get_lightest_logical_basis",
    "optimize_logical_basis",
]
