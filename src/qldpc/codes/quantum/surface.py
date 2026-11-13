# SPDX-License-Identifier: Apache-2.0

"""Surface and toric codes, and their generalizations to more dimensions."""

from __future__ import annotations

import collections
import itertools
from collections.abc import Iterator, Sequence

import galois
import numpy as np
import numpy.typing as npt
import sympy
from sympy.matrices.normalforms import hermite_normal_form

from qldpc import abstract
from qldpc._utils import networkx as nx
from qldpc.math import resolve_field
from qldpc.objects import (
    ChainComplex,
    Node,
    Pauli,
    PauliXZ,
)

from ..classical import (
    RepetitionCode,
    RingCode,
)
from ..common import CSSCode
from .hypergraph_product import HGPCode

####################################################################################################
# surface code and friends


class SurfaceCode(CSSCode):
    """The one and only!

    Actually, there are two variants: "ordinary" and "rotated" surface codes.  The rotated code is
    more qubit-efficient.

    References:

    - https://errorcorrectionzoo.org/c/toric
    - https://errorcorrectionzoo.org/c/surface
    - https://errorcorrectionzoo.org/c/rotated_surface
    """

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        field: int | type[galois.FieldArray] | None = None,
        *,
        rotated: bool = True,
    ) -> None:
        if cols is None:
            cols = rows
        self.rows = rows
        self.cols = cols
        self.rotated = rotated

        # save known distances and dimension
        self._distance_x = cols
        self._distance_z = rows
        self._dimension = 1

        if not rotated:
            # "original" surface code
            code_a = RepetitionCode(rows, field)
            code_b = RepetitionCode(cols, field)
            self.parent_code = HGPCode(code_a, code_b)
            matrix_x = self.parent_code.matrix_x.view(np.ndarray)
            matrix_z = self.parent_code.matrix_z.view(np.ndarray)
            self.bias_tailoring_qubits = self.parent_code.bias_tailoring_qubits

        else:
            # rotated surface code
            matrix_x, matrix_z = SurfaceCode.get_rotated_checks(rows, cols)
            self.bias_tailoring_qubits = [
                idx
                for idx, (row, col) in enumerate(np.ndindex(self.rows, self.cols))
                if (row + col) % 2
            ]

            # invert Z-type Pauli on every other qubit
            field = resolve_field(field)
            if field is not galois.GF2:
                matrix_z = matrix_z.view(field)
                matrix_z[:, self.bias_tailoring_qubits] *= -1

        super().__init__(matrix_x, matrix_z, field=field, promise_equal_distance_xz=rows == cols)

    @staticmethod
    def get_rotated_checks(
        rows: int, cols: int
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Build X-sector and Z-sector parity check matrices.

        Example 5x5 rotated surface code layout::

              ―――     ―――
             | ⋅ |   | ⋅ |
             ○―――○―――○―――○―――○―――
             | × | ⋅ | × | ⋅ | × |
          ―――○―――○―――○―――○―――○―――
         | × | ⋅ | × | ⋅ | × |
          ―――○―――○―――○―――○―――○―――
             | × | ⋅ | × | ⋅ | × |
          ―――○―――○―――○―――○―――○―――
         | × | ⋅ | × | ⋅ | × |
          ―――○―――○―――○―――○―――○
                 | ⋅ |   | ⋅ |
                  ―――     ―――

        Here:

        - Circles (○) denote data qubits (of which there are 5×5 = 25 total).
        - Tiles with a cross (×) denote X-type parity checks (12 total).
        - Tiles with a dot (⋅) denote Z-type parity checks (12 total).
        """

        def get_check(row: int, col: int) -> npt.NDArray[np.int_]:
            """Check on the qubits with the given indices, dropping any that are out of bounds."""
            row_indices = [row - 1, row, row - 1, row]
            col_indices = [col - 1, col - 1, col, col]
            check = np.zeros((rows, cols), dtype=int)
            for check_row, check_col in zip(row_indices, col_indices):
                if 0 <= check_row < rows and 0 <= check_col < cols:
                    check[check_row, check_col] = 1
            return check.ravel()

        checks_x = []
        checks_z = []
        for row, col in itertools.product(range(rows + 1), range(cols + 1)):
            if _rotated_check_is_used(row, col, rows, cols):
                check = get_check(row, col)
                if _get_check_pauli(row, col) is Pauli.X:
                    checks_x.append(check)
                else:
                    checks_z.append(check)

        return np.array(checks_x), np.array(checks_z)

    def get_syndrome_subgraphs(self, *, strategy: str = "smallest_last") -> tuple[nx.DiGraph, ...]:
        """Sequence of subgraphs of the Tanner graph that induces a syndrome extraction sequence.

        See help(qldpc.codes.QuditCode.get_syndrome_subgraphs) for additional information.

        If this is an unrotated surface code, return the syndrome subgraphs of the parent HGPCode.
        Otherwise, organize edges of the Tanner graph by an orientation in {NW, NE, SW, SE}, and by
        whether they are used for the readout of X-type or Z-type syndromes.  Interleave these edges
        in such a way as to minimize circuit depth and avoid hook errors.

        Args:
            strategy: Only used if self.rotated is False, in which case this argument is passed to
                HGPCode.get_syndrome_subgraphs to color the edges of the Tanner graph of this code.
                Default: "smallest_last".
        """
        if not self.rotated:
            return self.parent_code.get_syndrome_subgraphs(strategy=strategy)

        # identify all coordinates of check qubits, and a map from coordinates to a Node
        check_node_coords = sorted(
            [
                (row, col)
                for row, col in itertools.product(range(self.rows + 1), range(self.cols + 1))
                if _rotated_check_is_used(row, col, self.rows, self.cols)
            ],
            key=lambda row_col: (int(_get_check_pauli(*row_col)), *row_col),
        )
        node_map = {
            (row, col): Node(index, is_data=False)
            for index, (row, col) in enumerate(check_node_coords)
        }

        # collect edges of the Tanner graph by type: (pauli, orientation)
        edges: dict[tuple[Pauli, str], list[tuple[Node, Node]]] = collections.defaultdict(list)
        for qubit, (row, col) in enumerate(itertools.product(range(self.rows), range(self.cols))):
            data_node = Node(qubit, is_data=True)
            for orientation, (d_row, d_col) in _CHECK_OFFSETS.items():
                check_row, check_col = row + d_row, col + d_col
                if _rotated_check_is_used(check_row, check_col, self.rows, self.cols):
                    check_pauli = _get_check_pauli(check_row, check_col)
                    check_node = node_map[check_row, check_col]
                    edges[check_pauli, orientation].append((check_node, data_node))

        # return subgraphs in the order that minimizes hook errors
        subgraphs = {key: self.graph.edge_subgraph(edge_group) for key, edge_group in edges.items()}
        return (
            subgraphs[Pauli.X, "nw"],
            subgraphs[Pauli.Z, "nw"],
            subgraphs[Pauli.X, "sw"],
            subgraphs[Pauli.Z, "ne"],
            subgraphs[Pauli.X, "ne"],
            subgraphs[Pauli.Z, "sw"],
            subgraphs[Pauli.X, "se"],
            subgraphs[Pauli.Z, "se"],
        )


class ToricCode(CSSCode):
    """Surface code with periodic boundary conditions, encoding two logical qudits.

    References:

    - https://errorcorrectionzoo.org/c/toric
    - https://errorcorrectionzoo.org/c/surface
    - https://errorcorrectionzoo.org/c/rotated_surface
    """

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        field: int | type[galois.FieldArray] | None = None,
        *,
        rotated: bool = True,
    ) -> None:
        if cols is None:
            cols = rows
        self.rows = rows
        self.cols = cols
        self.rotated = rotated

        # save known distances and dimension
        self._distance_x = self._distance_z = min(rows, cols)
        self._dimension = 2

        if not rotated:
            # "original" toric code
            code_a = RingCode(rows, field)
            code_b = RingCode(cols, field)
            self.parent_code = HGPCode(code_a, code_b)
            matrix_x = self.parent_code.matrix_x.view(np.ndarray)
            matrix_z = self.parent_code.matrix_z.view(np.ndarray)
            self.bias_tailoring_qubits: list[int] | slice = slice(
                self.parent_code.sector_size[0, 0], None
            )

        else:
            # rotated toric code
            if rows % 2 or cols % 2:
                raise ValueError(
                    f"Rotated toric code must have even side lengths, not {rows} and {cols}"
                )

            matrix_x, matrix_z = ToricCode.get_rotated_checks(rows, cols)
            self.bias_tailoring_qubits = [
                idx
                for idx, (row, col) in enumerate(np.ndindex(self.rows, self.cols))
                if (row + col) % 2
            ]

            # invert Z-type Pauli on every other qubit
            field = resolve_field(field)
            if field is not galois.GF2:
                matrix_z = matrix_z.view(field)
                matrix_z[:, self.bias_tailoring_qubits] *= -1

            if rows == cols == 2 and rotated:
                # All Toric codes have redundant parity checks, but the case of the rotated 2x2
                # Toric code is particularly egregious: the two X/Z checks are *equal*.  So remove
                # the extra checks in this case.
                matrix_x = matrix_x[:-1]
                matrix_z = matrix_z[:-1]

        super().__init__(matrix_x, matrix_z, field=field, promise_equal_distance_xz=rows == cols)

    @staticmethod
    def get_rotated_checks(
        rows: int, cols: int
    ) -> tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Build X-sector and Z-sector parity check matrices.

        Same as in SurfaceCode.get_rotated_checks, but with periodic boundary conditions.
        """

        def get_check(row: int, col: int) -> npt.NDArray[np.int_]:
            """Check on the qubits with the given indices, dropping any that are out of bounds."""
            row_indices = np.array([row - 1, row, row - 1, row]) % rows
            col_indices = np.array([col - 1, col - 1, col, col]) % cols
            check = np.zeros((rows, cols), dtype=int)
            for check_row, check_col in zip(row_indices, col_indices):
                check[check_row, check_col] = 1
            return check.ravel()

        checks_x = []
        checks_z = []
        for row, col in itertools.product(range(rows), range(cols)):
            check = get_check(row, col)
            if _get_check_pauli(row, col) is Pauli.X:
                checks_x.append(check)
            else:
                checks_z.append(check)

        return np.array(checks_x), np.array(checks_z)

    def get_syndrome_subgraphs(self, *, strategy: str = "smallest_last") -> tuple[nx.DiGraph, ...]:
        """Sequence of subgraphs of the Tanner graph that induces a syndrome extraction sequence.

        If this is an unrotated toric code, return the syndrome subgraphs of the parent HGPCode.
        Otherwise, return the subgraphs of edges oriented along (NW, NE, SW, SE) directions.

        Args:
            strategy: Only used if self.rotated is False, in which case this argument is passed to
                HGPCode.get_syndrome_subgraphs to color the edges of the Tanner graph of this code.
                Default: "smallest_last".
        """
        if not self.rotated:
            return self.parent_code.get_syndrome_subgraphs(strategy=strategy)

        # identify all coordinates of check qubits, and a map from coordinates to a Node
        check_node_coords = sorted(
            [(row, col) for row, col in itertools.product(range(self.rows), range(self.cols))],
            key=lambda row_col: (int(_get_check_pauli(*row_col)), *row_col),
        )
        node_map = {
            (row, col): Node(index, is_data=False)
            for index, (row, col) in enumerate(check_node_coords)
        }

        # collect edges of the Tanner graph by type (orientation)
        edges: dict[str, list[tuple[Node, Node]]] = collections.defaultdict(list)
        for qubit, (row, col) in enumerate(itertools.product(range(self.rows), range(self.cols))):
            node_data = Node(qubit, is_data=True)
            edges["nw"].append((node_map[row, col], node_data))
            edges["ne"].append((node_map[row, (col + 1) % self.cols], node_data))
            edges["sw"].append((node_map[(row + 1) % self.rows, col], node_data))
            edges["se"].append((node_map[(row + 1) % self.rows, (col + 1) % self.cols], node_data))

        subgraphs = {key: self.graph.edge_subgraph(edge_group) for key, edge_group in edges.items()}
        return subgraphs["nw"], subgraphs["ne"], subgraphs["sw"], subgraphs["se"]


class GeneralizedSurfaceCode(CSSCode):
    """Surface or toric code defined on a multi-dimensional hypercubic lattice.

    References:

    - https://errorcorrectionzoo.org/c/higher_dimensional_surface
    """

    def __init__(
        self,
        size: int,
        dim: int,
        field: int | type[galois.FieldArray] | None = None,
        *,
        periodic: bool = False,
    ) -> None:
        if dim < 2:
            raise ValueError(
                f"The dimension of a generalized surface code should be >= 2 (provided: {dim})"
            )

        # save known distances
        self._distance_x = size ** (dim - 1)
        self._distance_z = size

        base_code = RingCode(size, field) if periodic else RepetitionCode(size, field)

        # build a chain complex one link at a time
        chain = ChainComplex([base_code.matrix])
        link = ChainComplex([base_code.matrix.T])
        for _ in range(dim - 1):
            chain = ChainComplex.tensor_product(chain, link)

            # to reduce computational overhead, remove chain links that we don't care about
            chain = ChainComplex(chain.ops[:2])

        matrix_x, matrix_z = chain.op(1), chain.op(2).T
        assert not isinstance(matrix_x, abstract.RingArray)
        assert not isinstance(matrix_z, abstract.RingArray)
        super().__init__(matrix_x, matrix_z, field)


class T4Code(CSSCode):
    """Four-dimensional (2, 2) toric code.

    A T4Code depends on a four-dimensional integer lattice.  The lattice acts on Euclidean
    four-space by translations.  The quotient manifold is a four-dimensional torus ``T^4``.  The
    number of physical qudits is 6 times the volume of the lattice.  The number of logical qudits is
    the dimension of the 2nd homology group of ``T^4``, which is 6.

    Homologically, a T4Code is the topological CSS code attached to the 3-term sub-complex in the
    middle of the 5-term chain-complex

        ``F[vertices] <- F[edges] <- F[faces] <- F[cubes] <- F[tesseracts]``

    of ``T^4`` tiled by integer unit tesseracts.  Cubes have 6 faces on the boundary, and edges are
    incident to 6 faces, so stabilizers have weight 6.

    The lattice must have at least two unit cells per period, that is ``abs(det(matrix)) >= 2``.  A
    unimodular basis tiles the torus with a single cell, for which every boundary operator vanishes
    identically and the resulting code has no parity checks at all.

    References:

    - https://arxiv.org/abs/2506.15130
    - https://arxiv.org/abs/2505.10403
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]],
        field: int | type[galois.FieldArray] | None = None,
        *,
        skip_validation: bool = False,
    ) -> None:
        """Construct a T4Code from a 4x4 integer matrix whose rows generate a 4d lattice."""
        self._field = resolve_field(field)

        determinant = int(sympy.Matrix(matrix).det())
        if abs(determinant) < 2:
            raise ValueError(
                "A T4Code requires a lattice basis with abs(determinant) >= 2"
                f" (provided a basis with determinant {determinant})"
            )

        # put the lattice basis in a standard form before enumerating one fundamental domain
        self.lattice_basis = hermite_normal_form(sympy.Matrix(matrix).T).T[::-1, ::-1]
        self.num_vertices = self.lattice_basis.det()
        self.num_edges = 4 * self.num_vertices
        self.num_faces = 6 * self.num_vertices
        self.num_cubes = 4 * self.num_vertices

        self.edges = list(self._iter_edges())
        self.face_basis = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        self.face_index = {face: index for index, face in enumerate(self.face_basis)}

        d_1 = np.vstack([self._d_1(edge) for edge in range(self.num_edges)]).T
        d_2 = np.vstack([self._d_2(face) for face in range(self.num_faces)]).T
        d_3 = np.vstack([self._d_3(cude) for cude in range(self.num_cubes)]).T
        self.chain = ChainComplex([d_1, d_2, d_3], skip_validation=skip_validation)

        matrix_x, matrix_z = self.chain.op(2), self.chain.op(3).T
        assert not isinstance(matrix_x, abstract.RingArray)
        assert not isinstance(matrix_z, abstract.RingArray)
        super().__init__(matrix_x, matrix_z, field)

    def _iter_edges(self) -> Iterator[tuple[int, int]]:
        """Identify edges in the standard integer lattice Z^4."""
        I4 = np.eye(4, dtype=np.int32)
        diag = self.lattice_basis.diagonal()
        vertices = list(itertools.product(*[range(n) for n in diag]))
        cosets = [self.lattice_basis.T.solve(sympy.Matrix(vertex)) % 1 for vertex in vertices]
        for source_vertex in range(self.num_vertices):
            for edge_direction in range(4):
                edge = sympy.Matrix(vertices[source_vertex]) + sympy.Matrix(I4[edge_direction])
                coset = self.lattice_basis.T.solve(edge).applyfunc(sympy.frac)
                k = [coset == c for c in cosets].index(True)
                yield (source_vertex, k)

    def _ones_vec(
        self, length: int, positive: Sequence[int], negative: Sequence[int]
    ) -> galois.FieldArray:
        """Construct a vector of +/- ones at the given locations."""
        output = self.field.Zeros(length)
        output[list(positive)] += self.field(1)
        output[list(negative)] -= self.field(1)
        return output

    def _get_edge_target(self, edge_index: int) -> int:
        """The "target" vertex of an edge."""
        return self.edges[edge_index][1]

    def _d_1(self, edge_index: int) -> npt.NDArray[np.int_]:
        """Boundary operator F[edges] -> F[vertices], evaluated at an edge."""
        source, target = self.edges[edge_index]
        return self._ones_vec(self.num_vertices, [target], [source])

    def _d_2(self, face_index: int) -> npt.NDArray[np.int_]:
        """Boundary operator F[faces] -> F[edges], evaluated at a face."""
        v, (i, j) = face_index // 6, self.face_basis[face_index % 6]
        bottom_edge = 4 * v + i
        left_edge = 4 * v + j
        right_edge = 4 * self._get_edge_target(bottom_edge) + j
        top_edge = 4 * self._get_edge_target(left_edge) + i
        return self._ones_vec(self.num_edges, [bottom_edge, right_edge], [top_edge, left_edge])

    def _d_3(self, cube_index: int) -> npt.NDArray[np.int_]:
        """Boundary operator F[cubes] -> F[faces], evaluated at a cube.

        See Section 2.2.3 of "Computational Homology" by T. Kaczynski, K. Mischaikow, and M. Mrozek.
        """
        v, (i, j, k) = cube_index // 4, (idx for idx in range(4) if idx != cube_index % 4)
        vi = self._get_edge_target(4 * v + i)
        vj = self._get_edge_target(4 * v + j)
        vk = self._get_edge_target(4 * v + k)
        front_face = 6 * v + self.face_index[(i, j)]
        left_face = 6 * v + self.face_index[(i, k)]
        bottom_face = 6 * v + self.face_index[(j, k)]
        top_face = 6 * vi + self.face_index[(j, k)]
        right_face = 6 * vj + self.face_index[(i, k)]
        back_face = 6 * vk + self.face_index[(i, j)]
        return self._ones_vec(
            self.num_faces, [top_face, back_face, left_face], [bottom_face, front_face, right_face]
        )


def _get_check_pauli(row: int, col: int) -> PauliXZ:
    """What type of stabilizer does the check at the given coordinates measure?

    Checks of a rotated surface or toric code alternate between X and Z type in a checkerboard
    pattern, so the type is fixed by the parity of the sum of the coordinates.
    """
    return Pauli.X if (row + col) % 2 == 0 else Pauli.Z


def _rotated_check_is_used(row: int, col: int, rows: int, cols: int) -> bool:
    """Is the check at these coordinates part of a rows x cols rotated surface code?

    Check coordinates range over a (rows + 1) x (cols + 1) grid.  The only boundary checks are
    Z-type checks on the top and bottom boundaries, and X-type checks on the left and right.
    """
    if row == 0 or row == rows:
        return 0 < col < cols and _get_check_pauli(row, col) is Pauli.Z
    if col == 0 or col == cols:
        return 0 < row < rows and _get_check_pauli(row, col) is Pauli.X
    return 0 < row < rows and 0 < col < cols


# offsets from a data qubit at (row, col) to its neighboring checks in a rotated surface code
_CHECK_OFFSETS = {"nw": (0, 0), "ne": (0, 1), "sw": (1, 0), "se": (1, 1)}
