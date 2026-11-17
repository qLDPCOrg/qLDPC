# SPDX-License-Identifier: Apache-2.0

"""Check bases of logical operators of binary CSS codes, and measure their weights and overlaps."""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
import numpy.typing as npt

from qldpc import codes
from qldpc._binary import get_independent_rows
from qldpc.codes.distance._binary import _as_dense
from qldpc.objects import Pauli, PauliXZLike

from ._sectors import Sector, get_sector

####################################################################################################
# evaluation


def evaluate_logical_ops(
    code: codes.CSSCode,
    logical_ops: npt.ArrayLike | None = None,
    *,
    logicals: Literal["dressed", "bare"] = "dressed",
) -> LogicalBasisMetrics:
    """Check a complete basis of logical operators of a CSS code, and measure weights and overlaps.

    A complete basis has shape ``(2k, 2n)``, laid out as by
    :meth:`qldpc.codes.CSSCode.get_logical_ops`: the first ``k`` rows are X-type operators and the
    last ``k`` rows are Z-type operators, with X-type operator ``j`` anticommuting with Z-type
    operator ``j`` and commuting with all others.

    For a subsystem code, these conditions apply to logical classes, since the gauge parts of
    dressed operators can change whether they commute.  To check a basis for
    :meth:`qldpc.codes.CSSCode.set_logical_ops`, which accepts only bare operators, pass
    ``logicals="bare"``.  Bare operators pair up physically exactly when their classes do.

    For a basis of one Pauli type, use :meth:`LogicalOpsMetrics.from_ops`.

    Args:
        code: A binary CSS code, which may be a subsystem code.
        logical_ops: The basis to evaluate.  Defaults to the code's own logical operators.
        logicals: For a subsystem code, ``"dressed"`` (the default) accepts logical operators that
            commute with the stabilizers, while ``"bare"`` requires them to commute with every gauge
            operator.  The two are the same for a stabilizer code.

    Returns:
        The metrics of the X-type and Z-type operators.

    Raises:
        TypeError: If ``code`` is not a CSS code.
        ValueError: If ``code`` is not binary, or ``logicals`` is invalid.  Also if ``logical_ops``
            has the wrong shape, has entries other than 0 and 1, or is not a basis as described
            above.
    """
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
    return LogicalBasisMetrics(
        x=LogicalOpsMetrics._from_valid_ops(ops_x), z=LogicalOpsMetrics._from_valid_ops(ops_z)
    )


@dataclasses.dataclass(frozen=True, eq=False)
class LogicalBasisMetrics:
    """The metrics of the X-type and Z-type operators of a complete basis of logical operators.

    Attributes:
        x: The metrics of the X-type operators.
        z: The metrics of the Z-type operators.
    """

    x: LogicalOpsMetrics
    z: LogicalOpsMetrics


@dataclasses.dataclass(frozen=True, eq=False)
class LogicalOpsMetrics:
    """The weights of logical operators of one Pauli type, and the qubits that they share.

    Operators that share no qubits can, for example, be measured by lattice surgery at the same
    time, so low loads and few overlapping pairs make for more parallel logical operations.

    Attributes:
        loads: The number of operators that act on each qubit.
        overlaps: The number of qubits that each pair of operators share, as a symmetric matrix
            whose diagonal holds the weights of the operators.
    """

    loads: npt.NDArray[np.int_]
    overlaps: npt.NDArray[np.int_]

    @classmethod
    def from_ops(
        cls,
        code: codes.CSSCode,
        pauli: PauliXZLike,
        logical_ops: npt.ArrayLike | None = None,
        *,
        logicals: Literal["dressed", "bare"] = "dressed",
    ) -> LogicalOpsMetrics:
        """Check a basis of logical operators of one Pauli type, and measure weights and overlaps.

        For a complete basis, with operators of both Pauli types, use
        :func:`~qldpc.experimental.logical_bases.evaluate_logical_ops`.

        Args:
            code: A binary CSS code, which may be a subsystem code.
            pauli: The Pauli type of the operators.
            logical_ops: The basis to evaluate, with shape ``(k, n)``.  Defaults to the code's own
                logical operators of this type.
            logicals: For a subsystem code, ``"dressed"`` (the default) accepts logical operators
                that commute with the stabilizers, while ``"bare"`` requires them to commute with
                every gauge operator.  The two are the same for a stabilizer code.

        Returns:
            The metrics of the operators.

        Raises:
            TypeError: If ``code`` is not a CSS code.
            ValueError: If ``code`` is not binary, or ``pauli`` or ``logicals`` is invalid.  Also if
                ``logical_ops`` has the wrong shape, has entries other than 0 and 1, does not
                consist of logical operators of this type, or does not have independent logical
                classes.
        """
        sector = get_sector(code, pauli, logicals)
        ops = code.get_logical_ops(pauli) if logical_ops is None else logical_ops
        matrix = _as_binary_matrix(ops, (sector.dimension, sector.num_qubits))
        _get_valid_classes(sector, matrix)
        return cls._from_valid_ops(matrix)

    @property
    def weights(self) -> npt.NDArray[np.int_]:
        """The weight of each operator."""
        return np.diag(self.overlaps).copy()

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
    def _from_valid_ops(cls, ops: npt.NDArray[np.uint8]) -> LogicalOpsMetrics:
        """The metrics of the rows of a binary matrix, without checking that they are logical."""
        matrix = ops.astype(np.int_)
        return cls(matrix.sum(axis=0), matrix @ matrix.T)


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
