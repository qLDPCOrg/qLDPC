# SPDX-License-Identifier: Apache-2.0

"""Bases of logical operators of binary CSS codes that are light and share few qubits.

EXPERIMENTAL: this subpackage is under active development; its public API is unstable and may
change without notice or deprecation.

Logical operators that share no physical qubits can be acted on at the same time, for example by
parallel lattice surgery, so a basis whose operators are light and overlap little allows more
parallel logical operations.  X-type and Z-type operators are considered separately.

Public API:
    evaluate_logical_ops, LogicalOpsMetrics       — check a basis and measure weights and overlaps
    get_logical_weight_spectrum, WeightSpectrum   — the smallest weights that any basis can have
"""

from __future__ import annotations

from .evaluation import LogicalOpsMetrics, evaluate_logical_ops
from .spectrum import WeightSpectrum, get_logical_weight_spectrum

__all__ = [
    "LogicalOpsMetrics",
    "WeightSpectrum",
    "evaluate_logical_ops",
    "get_logical_weight_spectrum",
]
