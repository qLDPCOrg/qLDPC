# SPDX-License-Identifier: Apache-2.0

"""Check bases of logical operators of binary CSS codes, and measure their weights and overlaps."""

from __future__ import annotations

import dataclasses
from typing import Literal, overload

import numpy as np
import numpy.typing as npt

from qldpc import codes
from qldpc._binary import get_independent_rows
from qldpc.codes.distance._binary import _as_dense
from qldpc.objects import Pauli, PauliXZLike

from ._sectors import Sector, get_sector

####################################################################################################
# evaluation


@overload
def evaluate_logical_ops(
    code: codes.CSSCode,
    logical_ops: npt.ArrayLike | None = None,
    *,
    pauli: None = None,
    logicals: Literal["dressed", "bare"] = "dressed",
) -> tuple[LogicalOpsMetrics, LogicalOpsMetrics]: ...


@overload
def evaluate_logical_ops(
    code: codes.CSSCode,
    logical_ops: npt.ArrayLike | None = None,
    *,
    pauli: PauliXZLike,
    logicals: Literal["dressed", "bare"] = "dressed",
) -> LogicalOpsMetrics: ...


def evaluate_logical_ops(
    code: codes.CSSCode,
    logical_ops: npt.ArrayLike | None = None,
    *,
    pauli: PauliXZLike | None = None,
    logicals: Literal["dressed", "bare"] = "dressed",
) -> tuple[LogicalOpsMetrics, LogicalOpsMetrics] | LogicalOpsMetrics:
    """Check a basis of logical operators of a CSS code, and measure its weights and overlaps.

    Without ``pauli``, ``logical_ops`` is a complete basis with shape ``(2k, 2n)``, laid out as by
    :meth:`qldpc.codes.CSSCode.get_logical_ops`: the first ``k`` rows are X-type operators and the
    last ``k`` rows are Z-type operators, with X-type operator ``j`` anticommuting with Z-type
    operator ``j`` and commuting with all others.  With ``pauli``, ``logical_ops`` is a basis of
    logical operators of that type only, with shape ``(k, n)``.

    For a subsystem code, these conditions apply to logical classes, since the gauge parts of
    dressed operators can change whether they commute.  :meth:`qldpc.codes.CSSCode.set_logical_ops`
    requires a complete basis to pair up physically, so give it bare operators, which pair up
    physically exactly when their classes do.

    Args:
        code: A binary CSS code, which may be a subsystem code.
        logical_ops: The basis to evaluate.  Defaults to the code's own logical operators.
        pauli: If given, evaluate a basis of logical operators of this type only.
        logicals: For a subsystem code, ``"dressed"`` (the default) accepts logical operators that
            commute with the stabilizers, while ``"bare"`` requires them to commute with every gauge
            operator.  The two are the same for a stabilizer code.

    Returns:
        The metrics of the X-type and Z-type operators, or only those of the requested type.

    Raises:
        TypeError: If ``code`` is not a CSS code.
        ValueError: If ``code`` is not binary, or ``logicals`` is invalid.  Also if ``logical_ops``
            has the wrong shape, has entries other than 0 and 1, or is not a basis as described
            above.
    """
    if pauli is not None:
        sector = get_sector(code, pauli, logicals)
        ops = code.get_logical_ops(pauli) if logical_ops is None else logical_ops
        matrix = _as_binary_matrix(ops, (sector.dimension, sector.num_qubits))
        _get_valid_classes(sector, matrix)
        return LogicalOpsMetrics._from_ops(matrix)

    sector_x = get_sector(code, Pauli.X, logicals)
    sector_z = get_sector(code, Pauli.Z, logicals)
    k, n = sector_x.dimension, sector_x.num_qubits
    ops = code.get_logical_ops() if logical_ops is None else logical_ops
    matrix = _as_binary_matrix(ops, (2 * k, 2 * n))

    # check that the X-type and Z-type halves are each a basis of logical operators of their type
    if np.any(matrix[:k, n:]) or np.any(matrix[k:, :n]):
        raise ValueError("The first k logical operators must be X-type, and the last k Z-type")
    ops_x, ops_z = matrix[:k, :n], matrix[k:, n:]
    classes_x = _get_valid_classes(sector_x, ops_x)
    classes_z = _get_valid_classes(sector_z, ops_z)

    # check that the classes of the two halves pair up
    pairings = classes_x.astype(np.int64) @ classes_z.T.astype(np.int64) % 2
    if np.any(pairings != np.eye(k, dtype=np.int64)):
        raise ValueError(
            "Logical X-type operator j must anticommute with logical Z-type operator j, and commute"
            " with all others"
        )
    return LogicalOpsMetrics._from_ops(ops_x), LogicalOpsMetrics._from_ops(ops_z)


@dataclasses.dataclass(frozen=True, eq=False)
class LogicalOpsMetrics:
    """The weights of logical operators of one Pauli type, and the qubits that they share.

    Operators that share no qubits can, for example, be measured by lattice surgery at the same
    time, so low loads and few overlapping pairs make for more parallel logical operations.

    Attributes:
        weights: The weight of each operator.
        loads: The number of operators that act on each qubit.
        overlaps: The number of qubits that each pair of operators share, as a symmetric matrix
            whose diagonal holds the weights.
    """

    weights: npt.NDArray[np.int_]
    loads: npt.NDArray[np.int_]
    overlaps: npt.NDArray[np.int_]

    @property
    def max_weight(self) -> int:
        """The largest weight of an operator, or zero if there are none."""
        return int(self.weights.max(initial=0))

    @property
    def total_weight(self) -> int:
        """The sum of the weights of the operators."""
        return int(self.weights.sum())

    @property
    def max_load(self) -> int:
        """The largest number of operators that act on one qubit."""
        return int(self.loads.max(initial=0))

    @property
    def num_overlapping_pairs(self) -> int:
        """The number of pairs of distinct operators that share at least one qubit."""
        return int(np.count_nonzero(np.triu(self.overlaps, k=1)))

    @classmethod
    def _from_ops(cls, ops: npt.NDArray[np.uint8]) -> LogicalOpsMetrics:
        """The metrics of the rows of a binary matrix, without checking that they are logical."""
        matrix = ops.astype(np.int_)
        return cls(matrix.sum(axis=1), matrix.sum(axis=0), matrix @ matrix.T)


def _as_binary_matrix(ops: npt.ArrayLike, shape: tuple[int, int]) -> npt.NDArray[np.uint8]:
    """A binary matrix with the given shape, such as from a ``galois`` or SciPy sparse array."""
    matrix = np.asarray(_as_dense(ops)).view(np.ndarray)
    if matrix.shape != shape:
        raise ValueError(f"Expected logical operators with shape {shape}, not {matrix.shape}")
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError("Logical operators must be binary")
    return matrix.astype(np.uint8)


def _get_valid_classes(sector: Sector, ops: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """The logical classes of a basis of logical operators of one type, or an error."""
    name = f"{sector.pauli}-type"
    if np.any(sector.checks.astype(np.int64) @ ops.T.astype(np.int64) % 2):
        raise ValueError(f"Some {name} operators are not logical operators of the code")
    classes = sector.get_classes(ops)
    if len(get_independent_rows(classes)) != sector.dimension:
        raise ValueError(f"The {name} operators do not have independent logical classes")
    return classes
