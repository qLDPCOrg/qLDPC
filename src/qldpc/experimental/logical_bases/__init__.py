# SPDX-License-Identifier: Apache-2.0

"""Check and find bases of logical operators of binary CSS codes for parallel logical operations.

EXPERIMENTAL: this subpackage is under active development; its public API is unstable and may change
without notice or deprecation.

Logical operators that share no physical qubits can be acted on at the same time, for example by
parallel lattice surgery, so a basis whose operators are light and overlap little allows more
parallel logical operations.  This subpackage checks bases, complete or of one Pauli type, and
measures the weights of their operators and the overlaps between operators of the same Pauli type.
It also finds a lightest basis of each Pauli type, whose sorted weights bound those of every basis
of that type and kind (dressed or bare), as ``LightestLogicalBasis`` describes.  It searches for
complete bases with low weights, whose X-type and Z-type operators pair up as
``evaluate_logical_ops`` describes.  Finally, it chooses an operator in each logical class of a
basis, so that the chosen operators can be acted on in few parallel rounds.

Public API:

- :func:`~qldpc.experimental.logical_bases.evaluate_logical_ops`,
  :class:`~qldpc.experimental.logical_bases.LogicalBasisMetrics`: check and measure a complete
  basis.
- :class:`~qldpc.experimental.logical_bases.LogicalOpsMetrics`,
  :meth:`LogicalOpsMetrics.from_ops <qldpc.experimental.logical_bases.LogicalOpsMetrics.from_ops>`:
  check and measure one Pauli type.
- :func:`~qldpc.experimental.logical_bases.get_lightest_logical_basis`,
  :class:`~qldpc.experimental.logical_bases.LightestLogicalBasis`: a lightest basis of one Pauli
  type.
- :func:`~qldpc.experimental.logical_bases.optimize_logical_basis`,
  :class:`~qldpc.experimental.logical_bases.OptimizedLogicalBasis`: a complete basis with low
  weights.
- :data:`~qldpc.experimental.logical_bases.WeightObjective`: the type of an objective of
  ``optimize_logical_basis`` given as a function of the weights.
- :func:`~qldpc.experimental.logical_bases.optimize_logical_rounds`,
  :class:`~qldpc.experimental.logical_bases.OptimizedLogicalRounds`: operators to act on in few
  rounds.
- :class:`~qldpc.experimental.logical_bases.LogicalOpsRounds`: the rounds of one Pauli type.
"""

from __future__ import annotations

from .evaluation import LogicalBasisMetrics, LogicalOpsMetrics, evaluate_logical_ops
from .lightest_basis import LightestLogicalBasis, get_lightest_logical_basis
from .optimization import OptimizedLogicalBasis, WeightObjective, optimize_logical_basis
from .rounds import LogicalOpsRounds, OptimizedLogicalRounds, optimize_logical_rounds

__all__ = [
    "LightestLogicalBasis",
    "LogicalBasisMetrics",
    "LogicalOpsMetrics",
    "LogicalOpsRounds",
    "OptimizedLogicalBasis",
    "OptimizedLogicalRounds",
    "WeightObjective",
    "evaluate_logical_ops",
    "get_lightest_logical_basis",
    "optimize_logical_basis",
    "optimize_logical_rounds",
]
