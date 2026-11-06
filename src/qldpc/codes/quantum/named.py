# SPDX-License-Identifier: Apache-2.0

"""Named quantum codes from the literature, each with a fixed length and dimension."""

from __future__ import annotations

import galois
import numpy as np
import scipy
import sympy

from qldpc.math import resolve_field

from ..classical import (
    GolayCode,
    ReedMullerCode,
)
from ..common import ClassicalCode, CSSCode, QuditCode
from .self_orthogonal import IcebergCode, QuantumHammingCode
from .two_block import BBCode

####################################################################################################
# named codes


class FiveQuditCode(QuditCode):
    """Smallest quantum error-correcting code.

    Generalizes the better-known FiveQubitCode.

    References:

    - https://errorcorrectionzoo.org/c/galois_5_1_3
    """

    def __init__(self, field: int | type[galois.FieldArray] | None = None) -> None:
        field = resolve_field(field)
        matrix = [
            [1, 0, 0, -1, 0, 0, 1, -1, 0, 0],
            [0, 1, 0, 0, -1, 0, 0, 1, -1, 0],
            [-1, 0, 1, 0, 0, 0, 0, 0, 1, -1],
            [0, -1, 0, 1, 0, -1, 0, 0, 0, 1],
        ]
        super().__init__(field(1) * np.array(matrix, dtype=int))
        self._is_subsystem_code = False  # known by construction
        self._dimension = 1
        self._distance = 3


class FiveQubitCode(FiveQuditCode):
    """Smallest quantum error-correcting code.

    References:

    - https://errorcorrectionzoo.org/c/stab_5_1_3
    """

    def __init__(self) -> None:
        super().__init__(field=2)


class SteaneCode(QuantumHammingCode):
    """Smallest quantum error-correcting CSS code.

    Also the smallest error-correcting color code.

    References:

    - https://errorcorrectionzoo.org/c/steane
    """

    def __init__(self) -> None:
        super().__init__(size=3)


class TetrahedralCode(CSSCode):
    r"""Smallest quantum error-correcting CSS code with a transversal non-Clifford (T) gate.

    Also:

    - The smallest quantum error-correcting 3-D color code.
    - Often referred to as the [15, 1, 3] quantum Reed-Muller code.

    Algebraically, a TetrahedralCode is a CSSCode built out of punctured Reed-Muller codes.
    Geometrically, a TetrahedralCode can be visualized with a tetrahedron (triangular pyramid).

    Consider a tessellation of a tetrahedron into four identical polyhedra, or 3-cells, where each
    polyhedron is the convex hull of (a) a vertex of the tetrahedron, (b) the centers of the edges
    and faces incident to that vertex, and (c) the centroid of the tetrahedron.  Qubits live on the
    vertices of these polyhedra.  The stabilizers of the TetrahedralCode can be defined as follows:

    - Every 3-cell (polyhedron) is associated with an X-type stabilizer.
    - Every 2-cell (face of a polyhedron) is associated with a Z-type stabilizer.

    Coloring the polyhedra red, green, blue, and yellow, the qubit layout for the TetrahedralCode
    can be visualized as follows::

                            red
                             0                           8
                            / \                         / \
                           /   \                       /   \
                          2     4                    10     12                7
                         / ‾‾6‾‾ \                   / ‾‾14‾ \
                        /    |    \                 /    |    \              top
                       1 --- 5 --- 3               9 --- 13 -- 11           vertex
                        green  blue                                        (yellow)

                    vertices on the base     vertices in the "middle"
                     of the tetrahedron     (edges, faces, and centroid)

    The ordering of qubits is fixed by enforcing consistency between geometric and algebraic
    constructions of the TetrahedralCode.

    See also Figure 2b of https://arxiv.org/pdf/2409.13465v2 for a nice picture, but note that

    (a) The tetrahedral code in arXiv:2409.13465 swaps all X and Z operators.
    (b) The TetrahedralCode defined here has a different qubit order.  Specifically, qubit jj of the
        code defined here gets mapped to qubit kk = qubit_map[jj] of the code in 2409.13465, where
        qubit_map = [0, 10, 3, 14, 7, 13, 6, 8, 1, 9, 2, 12, 4, 11, 5].

    A TetrahedralCode encodes one logical qubit.

    - The logical X operator can be defined on any _face_ of the tetrahedron.
    - The logical Z operator can be defined on any _edge_ of the tetrahedron.

    References:

    - https://arxiv.org/abs/1403.2734
    - https://arxiv.org/abs/2409.13465
    - https://errorcorrectionzoo.org/c/stab_15_1_3

    Args:
        algebraic: Choose Z-type stabilizers according to the algebraic (if True) or geometric
            (if False) definition of the TetrahedralCode, as described above.  The remaining
            stabilizers and logical operators are unaffected by this flag.  Default: False.
    """

    def __init__(self, *, algebraic: bool = False) -> None:
        """Construct an instance of the [15, 1, 3] tetrahedral code."""
        code_x = ReedMullerCode(2, 4).punctured([0])  # or HammingCode(4)

        if algebraic:
            code_z = ReedMullerCode(1, 4).punctured([0])

        else:
            # the stabilizers of Eq. 2 in arXiv:2409.13465v2, in a different order
            stabilizer_support_z = [
                # red / green / blue 2-cells on the base
                [0, 2, 4, 6],  # red
                [1, 2, 5, 6],  # green
                [3, 4, 5, 6],  # blue
                # red / green / blue 2-cells in the middle
                [8, 10, 12, 14],  # red
                [9, 10, 13, 14],  # green
                [11, 12, 13, 14],  # blue
                # 2-cells connecting the base to the middle
                [2, 6, 10, 14],  # red/green
                [4, 6, 12, 14],  # red/blue
                [5, 6, 13, 14],  # green/blue
                # 2-cell connecting the middle to the top vertex
                [7, 9, 11, 13],
            ]
            matrix_z = np.zeros((len(stabilizer_support_z), len(code_x)), dtype=int)
            for row, support in enumerate(stabilizer_support_z):
                matrix_z[row, support] = 1
            code_z = ClassicalCode(matrix_z)

        super().__init__(code_x, code_z)
        self._is_subsystem_code = False  # known by construction
        self.set_logical_ops_xz(
            [[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        )


class QuantumGolayCode(CSSCode):
    """Quantum [[23, 1, 7]] Golay code, built from the classical [23, 12, 7] Golay code.

    References:

    - https://errorcorrectionzoo.org/c/qubit_golay
    """

    def __init__(self) -> None:
        # by convention, keep only the first 11 of the classical code's (redundant) 23 parity checks
        checks = GolayCode().matrix[:11]
        super().__init__(checks, checks)
        self._is_subsystem_code = False  # known by construction
        self._distance_x = self._distance_z = 7


class C4Code(IcebergCode):
    """A [[4, 2, 2]] code, commonly known as the "C4" code.

    References:

    - https://errorcorrectionzoo.org/c/stab_4_2_2
    """

    def __init__(self) -> None:
        super().__init__(4)


class C6Code(CSSCode):
    """A [[6, 2, 2]] code, commonly known as the "C6" code.

    References:

    - https://errorcorrectionzoo.org/c/stab_6_2_2
    """

    def __init__(self) -> None:
        checks = [[1, 1, 0, 0, 1, 1], [0, 1, 1, 1, 1, 0]]
        super().__init__(checks, checks)
        self._is_subsystem_code = False  # known by construction
        logical_ops_xz = scipy.linalg.block_diag([1, 1, 1], [1, 1, 1])
        self.set_logical_ops_xz(logical_ops_xz, logical_ops_xz)


class GrossCode(BBCode):
    """The [[144, 12, 12]] gross bivariate bicycle code.

    References:

    - https://errorcorrectionzoo.org/c/gross
    - https://arxiv.org/abs/2308.07915
    - https://arxiv.org/abs/2506.03094
    """

    def __init__(self) -> None:
        x, y = sympy.symbols("x y")
        super().__init__({x: 12, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
        self._distance_x = self._distance_z = 12


class TwoGrossCode(BBCode):
    """The [[288, 12, 18]] two-gross (or double-gross) bivariate bicycle code.

    References:

    - https://errorcorrectionzoo.org/c/bb288
    - https://arxiv.org/abs/2308.07915
    - https://arxiv.org/abs/2506.03094
    """

    def __init__(self) -> None:
        x, y = sympy.symbols("x y")
        super().__init__({x: 12, y: 12}, x**3 + y**2 + y**7, y**3 + x + x**2)
        self._distance_x = self._distance_z = 18
