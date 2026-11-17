# SPDX-License-Identifier: Apache-2.0

"""Check and find bases of logical operators of binary CSS codes for parallel logical operations.

EXPERIMENTAL: this subpackage is under active development; its public API is unstable and may change
without notice or deprecation.

Logical operators that share no physical qubits can be acted on at the same time, for example by
parallel lattice surgery, so a basis whose operators are light and overlap little allows more
parallel logical operations.  This subpackage checks bases, complete or of one Pauli type, and
measures the weights of their operators and the overlaps between operators of the same Pauli type.
It also finds a lightest basis of each Pauli type.  Sorted by weight, the weights of every basis of
that type are at least those of the lightest basis, entry by entry; for a subsystem code, this holds
among bases of the same kind, dressed or bare.  It searches for complete bases with low weights,
whose X-type and Z-type operators pair up as ``evaluate_logical_ops`` describes.  Finally, it
chooses an operator in each logical class of a basis, so that the chosen operators can be acted on
in few parallel rounds.

Public API:
    evaluate_logical_ops, LogicalBasisMetrics            — check and measure a complete basis
    LogicalOpsMetrics, LogicalOpsMetrics.from_ops        — check and measure one Pauli type
    get_lightest_logical_basis, LightestLogicalBasis     — a lightest basis of one Pauli type
    optimize_logical_basis, OptimizedLogicalBasis        — a complete basis with low weights
    optimize_logical_rounds, OptimizedLogicalRounds      — operators to act on in few rounds
    LogicalOpsRounds                                     — the rounds of one Pauli type
"""

from __future__ import annotations

from .evaluation import LogicalBasisMetrics, LogicalOpsMetrics, evaluate_logical_ops
from .lightest_basis import LightestLogicalBasis, get_lightest_logical_basis
from .optimization import OptimizedLogicalBasis, optimize_logical_basis
from .rounds import LogicalOpsRounds, OptimizedLogicalRounds, optimize_logical_rounds

__all__ = [
    "LightestLogicalBasis",
    "LogicalBasisMetrics",
    "LogicalOpsMetrics",
    "LogicalOpsRounds",
    "OptimizedLogicalBasis",
    "OptimizedLogicalRounds",
    "evaluate_logical_ops",
    "get_lightest_logical_basis",
    "optimize_logical_basis",
    "optimize_logical_rounds",
]
