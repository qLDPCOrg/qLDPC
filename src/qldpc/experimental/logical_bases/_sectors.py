# SPDX-License-Identifier: Apache-2.0

"""The logical operators of one Pauli type of a binary CSS code, as binary matrices."""

from __future__ import annotations

import dataclasses
from typing import Literal

import numpy as np
import numpy.typing as npt

from qldpc import codes
from qldpc._binary import get_kernel
from qldpc.objects import Pauli, PauliXZ, PauliXZLike

LogicalOpsKind = Literal["dressed", "bare"]


####################################################################################################
# sectors


@dataclasses.dataclass(frozen=True, eq=False)
class Sector:
    """The logical operators of one Pauli type of a binary CSS code.

    The logical operators of this type are the words of ``ker(checks)``, and those in
    ``span(trivial)`` act trivially on the logical qubits.  The rows of ``dual_ops`` are the code's
    logical operators of the opposite type.  They are bare, so they commute with every row of
    ``trivial``, and the logical class ``dual_ops @ op`` of an operator ``op`` is unchanged by
    adding rows of ``trivial``.  The class of the code's own logical operator ``j`` of this type is
    the ``j``-th unit vector.

    Attributes:
        pauli: The Pauli type of the operators.
        checks: Rows of the opposite type that every logical operator of this type commutes with.
        trivial: Rows of this type that span the operators that act trivially on logical qubits.
        dual_ops: The code's logical operators of the opposite type, with shape ``(k, n)``.
    """

    pauli: PauliXZ
    checks: npt.NDArray[np.uint8]
    trivial: npt.NDArray[np.uint8]
    dual_ops: npt.NDArray[np.uint8]

    @property
    def num_qubits(self) -> int:
        """The number of physical qubits."""
        return self.dual_ops.shape[1]

    @property
    def dimension(self) -> int:
        """The number of logical qubits."""
        return len(self.dual_ops)

    def get_logical_space(self) -> npt.NDArray[np.uint8]:
        """Rows that span every logical operator of this type, trivial or not."""
        return get_kernel(self.checks)

    def get_classes(self, ops: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """The logical class of each row of ``ops``, as a row of length ``k``."""
        classes = ops.astype(np.int64) @ self.dual_ops.T.astype(np.int64) % 2
        return classes.astype(np.uint8)


def get_sector(code: codes.CSSCode, pauli: PauliXZLike, logicals: LogicalOpsKind) -> Sector:
    """The logical operators of one Pauli type of a binary CSS code.

    For a stabilizer code, dressed and bare logical operators coincide.  For a subsystem code, a
    bare logical operator commutes with every gauge operator and is trivial if it is a stabilizer,
    while a dressed logical operator need only commute with the stabilizers and is trivial if it is
    a gauge operator.  Every bare logical operator is also a dressed logical operator of the same
    class.
    """
    if not isinstance(code, codes.CSSCode):
        raise TypeError(f"Logical bases require a CSS code, not {type(code).__name__}")
    if code.field.order != 2:
        raise ValueError(
            f"Logical bases require a binary code, not one over GF({code.field.order})"
        )
    if logicals not in ("dressed", "bare"):
        raise ValueError(f"logicals must be 'dressed' or 'bare', not {logicals!r}")
    pauli = Pauli.coerce_xz(pauli)
    other = pauli.swap_xz()

    if logicals == "bare":
        checks = code.get_matrix(other)
        trivial = code.get_stabilizer_ops(pauli)
    else:
        checks = code.get_stabilizer_ops(other)
        trivial = code.get_matrix(pauli)
    return Sector(
        pauli, _as_uint8(checks), _as_uint8(trivial), _as_uint8(code.get_logical_ops(other))
    )


def _as_uint8(matrix: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """A binary matrix, such as a ``galois`` array, as a plain array of uint8 values."""
    return np.asarray(matrix).view(np.ndarray).astype(np.uint8)
