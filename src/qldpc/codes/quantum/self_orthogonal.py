# SPDX-License-Identifier: Apache-2.0

"""CSS codes whose X-type and Z-type parity checks come from one self-orthogonal classical code."""

from __future__ import annotations

from typing import Literal

import galois
import numpy as np
import scipy

from qldpc.objects import (
    PauliXZ,
)

from ..classical import (
    HammingCode,
    ReedMullerCode,
)
from ..common import CSSCode
from ..distance import DistanceMethod, NoDistance

####################################################################################################
# self-orthogonal CSS codes


class QuantumHammingCode(CSSCode):
    """Quantum Hamming code, whose parity check matrices are classical Hamming codes.

    References:

    - https://errorcorrectionzoo.org/c/quantum_hamming_css
    """

    def __init__(
        self,
        size: int,
        field: int | type[galois.FieldArray] | None = None,
        *,
        set_logicals: bool = True,
    ) -> None:
        code = HammingCode(size, field)
        super().__init__(code, code)
        self._is_subsystem_code = False  # known by construction
        self._distance_x = self._distance_z = 3

        if size == 4 and set_logicals and self.field is galois.GF2:
            """Make a "nice" choice of logical operators for the [15, 7, 3] quantum Hamming code.

            Pinning all but the last logical qubit to |0> results in the TetrahedralCode.  See the
            docstring of the TetrahedralCode for an explanation of the comments below.
            """
            support_x = [
                # red / green / blue 2-cells in the middle
                [8, 10, 12, 14],  # red
                [9, 10, 13, 14],  # green
                [11, 12, 13, 14],  # blue
                # 2-cells connecting the base to the middle
                [2, 6, 10, 14],  # red/green
                [4, 6, 12, 14],  # red/blue
                [5, 6, 13, 14],  # green/blue
                # all qubits
                range(len(self)),
            ]
            support_z = [
                # 2-cells connecting the base to the middle
                [5, 6, 13, 14],  # green/blue
                [4, 6, 12, 14],  # red/blue
                [2, 6, 10, 14],  # red/green
                # red / green / blue 2-cells in the middle
                [11, 12, 13, 14],  # blue
                [9, 10, 13, 14],  # green
                [8, 10, 12, 14],  # red
                # all qubits
                range(len(self)),
            ]
            logical_ops_x = np.zeros((len(support_x), len(self)), dtype=int)
            logical_ops_z = np.zeros((len(support_z), len(self)), dtype=int)
            for row in range(len(support_x)):
                logical_ops_x[row, support_x[row]] = 1
                logical_ops_z[row, support_z[row]] = 1
            self.set_logical_ops_xz(logical_ops_x, logical_ops_z)


class QuantumReedMullerCode(CSSCode):
    r"""Self-orthogonal CSS code from a classical Reed-Muller code.

    The code CSS(RM(r, m), RM(r, m)) is a [[2**m, 2**m - 2 * dim(RM(r, m)), 2**(r + 1)]] code for
    0 <= r < (m - 1) / 2, i.e. whenever RM(r, m) is strictly self-orthogonal::

        RM(r, m) ⊆ RM(r, m)⊥ = RM(m - r - 1, m).

    The stabilizer generators are the rows of the generator matrix of RM(r, m), whose pairwise
    orthogonality follows from self-orthogonality.  Both the X- and the Z-distance equal 2**(r + 1),
    the minimum weight of a vector in ``RM(m - r - 1, m) \ RM(r, m)``, attained by the indicator
    vector of an affine (r + 1)-flat in AG(m, 2) (MacWilliams & Sloane, Ch. 13).  This closed form
    makes distance evaluation O(1), independent of block length.

    References:

    - https://errorcorrectionzoo.org/c/quantum_reed_muller_css
    - https://feog.github.io/10-coding.pdf
    """

    def __init__(self, order: int, size: int) -> None:
        self._assert_valid_params(order, size)
        self._order = order
        self._size = size
        generator = np.atleast_2d(ReedMullerCode.get_generator(order, size))
        super().__init__(generator, generator, promise_equal_distance_xz=True)
        self._is_subsystem_code = False  # known by construction

    @property
    def order(self) -> int:
        """The order parameter of this code."""
        return self._order

    @property
    def size(self) -> int:
        """The size parameter of this code."""
        return self._size

    @staticmethod
    def _assert_valid_params(order: int, size: int) -> None:
        if not (size >= 1 and 0 <= order < size and 2 * order < size - 1):
            raise ValueError(
                "A self-orthogonal CSS(RM(r,m), RM(r,m)) code requires 0 <= r < m and 2r < m - 1\n"
                + f"Provided: (r, m) = ({order}, {size})"
            )

    def _get_distance_exact(
        self, pauli: PauliXZ | None, *, method: DistanceMethod
    ) -> int | Literal[NoDistance.NAN]:
        """Closed-form distance of CSS(RM(r, m), RM(r, m)) codes: 2**(r + 1)."""
        return 2 ** (self._order + 1)


class IcebergCode(CSSCode):
    """A quantum error detecting code: ``[[n, n - 2, 2]]``.

    References:

    - https://errorcorrectionzoo.org/c/iceberg
    """

    def __init__(self, size: int) -> None:
        if not size % 2 == 0:
            raise ValueError(
                f"The Iceberg code is only defined for even block lengths (provided: {size})"
            )
        checks = [[1] * size]
        super().__init__(checks, checks)
        self._is_subsystem_code = False  # known by construction
        self._dimension = size - 2
        self._distance_x = self._distance_z = 2


class ManyHypercubeCode(CSSCode):
    """The ``[[6**level, 4**level, 2**level]]`` concatenated many-hypercubes code.

    References:

    - https://arxiv.org/abs/2403.16054
    - https://errorcorrectionzoo.org/c/stab_6_4_2
    """

    def __init__(self, level: int = 1) -> None:
        if level < 1:
            raise ValueError(
                f"The many-hypercubes code requires a level of at least 1 (provided: {level})"
            )

        code: CSSCode
        if level == 1:
            # construct a [6, 4, 2] Iceberg code
            code = IcebergCode(6)
            super().__init__(code.code_x, code.code_z)
            self._is_subsystem_code = False  # known by construction

            # split the four logical qubits into pairs with disjoint support on the physical qubits
            sector_ops_x = [[1, 1, 0], [0, 1, 1]]
            sector_ops_z = sector_ops_x[::-1]
            ops_x = scipy.linalg.block_diag(sector_ops_x, sector_ops_x)
            ops_z = scipy.linalg.block_diag(sector_ops_z, sector_ops_z)
            self.set_logical_ops_xz(ops_x, ops_z)

        else:
            code = ManyHypercubeCode(1)
            base_code = ManyHypercubeCode(1)
            for _ in range(level - 1):
                code = CSSCode.concatenate(code, base_code)
            super().__init__(code.code_x, code.code_z)
            self._is_subsystem_code = False  # known by construction
            self._dimension = 4**level
            self._distance_x = self._distance_z = 2**level
