# SPDX-License-Identifier: Apache-2.0

"""Hypergraph product codes, their cyclic and subsystem variants, and distance-balanced codes."""

from __future__ import annotations

import collections
import itertools
from collections.abc import Collection, Iterable, Sequence
from typing import TypeVar

import galois
import numpy as np
import numpy.typing as npt
import scipy
import sympy

import qldpc
from qldpc import abstract
from qldpc._utils import networkx as nx
from qldpc.objects import (
    ChainComplex,
    Node,
    Pauli,
    PauliXZ,
    PauliXZLike,
    QuditPauli,
    TannerGraph,
)

from ..classical import (
    CyclicCode,
    RepetitionCode,
    RingCode,
    SimplexCode,
)
from ..common import ClassicalCode, CSSCode
from ..distance import DistanceMethod
from ..distance.core import _get_classical_distance_method

####################################################################################################
# hypergraph product codes and their variants


FieldOrRingArray = TypeVar("FieldOrRingArray", galois.FieldArray, abstract.RingArray)


class HGPCode(CSSCode):
    """Hypergraph product code.

    A hypergraph product code AB is constructed from the parity check matrices of two classical
    codes, A and B.

    Consider the following:

    - Code A has 3 data and 2 check bits.
    - Code B has 4 data and 3 check bits.

    We represent data bits/qudits by circles (○) and check bits/qudits by squares (□).

    Denote the Tanner graph of code C by G_C.  The nodes of G_AB can be arranged into a matrix.  The
    rows of this matrix are labeled by nodes of G_A, and columns by nodes of G_B.  The matrix of
    nodes in G_AB can thus be organized into four sectors::

        ――――――――――――――――――――――――――――――――――
          | ○ ○ ○ ○ | □ □ □ ← nodes of G_B
        ――+―――――――――+――――――
        ○ | ○ ○ ○ ○ | □ □ □
        ○ | ○ ○ ○ ○ | □ □ □
        ○ | ○ ○ ○ ○ | □ □ □
        ――+―――――――――+――――――
        □ | □ □ □ □ | ○ ○ ○
        □ | □ □ □ □ | ○ ○ ○
        ↑ nodes of G_A
        ――――――――――――――――――――――――――――――――――

    We identify each sector by two bits.
    In the example above:

    - sector (0, 0) has 3×4=12 data qudits
    - sector (0, 1) has 3×3=9 check qudits
    - sector (1, 0) has 2×4=8 check qudits
    - sector (1, 1) has 2×3=6 data qudits

    Edges in G_AB are inherited across rows/columns from G_A and G_B.  For example, if rows r_1 and
    r_2 share an edge in G_A, then the same is true in every column of G_AB.

    By default, the check qudits in sectors...

    - (1, 0) of G_AB measure X-type operators, and
    - (0, 1) of G_AB measure Z-type operators.

    This class contains two equivalent constructions of an HGPCode:

    - A construction based on Tanner graphs (as discussed above).
    - A construction based on check matrices, as originally introduced in arXiv:0903.0566.

    The latter construction is less intuitive, but more efficient.

    References:

    - https://errorcorrectionzoo.org/c/hypergraph_product
    - https://arxiv.org/abs/0903.0566
    - https://arxiv.org/abs/1202.0928
    - https://arxiv.org/abs/2202.01702
    - https://www.youtube.com/watch?v=iehMcUr2saM

    Syndrome extraction, canonical logical operators, and exact distances additionally follow:

    - https://arxiv.org/abs/2109.14609
    - https://arxiv.org/abs/2204.10812
    - https://arxiv.org/abs/2502.07150
    - https://arxiv.org/abs/2308.15520
    """

    code_a: ClassicalCode
    code_b: ClassicalCode
    sector_size: npt.NDArray[np.int_]

    def __init__(
        self,
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | type[galois.FieldArray] | None = None,
        *,
        set_logicals: bool = True,
    ) -> None:
        """Hypergraph product of two classical codes, as in arXiv:0903.0566.

        The parity check matrices of the hypergraph product code are:

        ``matrix_x = [ H1 ⨂ In2, Im1 ⨂ H2.T]``
        ``matrix_z = [-In1 ⨂ H2, H1.T ⨂ Im2]``

        Here ``(H1, H2) == (matrix_a, matrix_b)``, and ``I[m/n][1/2]`` are identity matrices, with
        ``(m1, n1) = H1.shape`` and ``(m2, n2) = H2.shape``.

        A minus sign in one sector of matrix_x or matrix_z is necessary to satisfy CSS code
        requirements with nonbinary fields.  The placement of this sign is chosen for consistency
        with the tensor product of chain complexes.
        """
        if code_b is None:
            code_b = code_a
        self.code_a = ClassicalCode(code_a, field)
        self.code_b = ClassicalCode(code_b, field)
        field = self.code_a.field

        # use a matrix-based hypergraph product to identify X-sector and Z-sector parity checks
        matrix_x, matrix_z = HGPCode.get_matrix_product(self.code_a.matrix, self.code_b.matrix)

        # identify the number of qudits in each sector
        self.sector_size = np.outer(
            [self.code_a.num_bits, self.code_a.num_checks],
            [self.code_b.num_bits, self.code_b.num_checks],
        )

        # if Hadamard-transforming qudits, conjugate those in the (1, 1) sector by default
        self.bias_tailoring_qubits = slice(self.sector_size[0, 0], None)

        super().__init__(
            matrix_x.view(np.ndarray).astype(int),
            matrix_z.view(np.ndarray).astype(int),
            field,
        )
        self._is_subsystem_code = False  # known by construction
        if set_logicals:
            logical_ops_xz = HGPCode.get_canonical_logical_line_ops(
                self.code_a.matrix, self.code_b.matrix
            )
            self.set_logical_ops_xz(*logical_ops_xz, skip_validation=True)

    def get_syndrome_subgraphs(self, *, strategy: str = "smallest_last") -> tuple[nx.DiGraph, ...]:
        """Sequence of subgraphs of the Tanner graph that induces a syndrome extraction sequence.

        See help(qldpc.codes.QuditCode.get_syndrome_subgraphs) for additional information.

        The sequence here is essentially the sequence used for hypergraph product codes in Algorithm
        2 of arXiv:2109.14609, modified to obviate the need to find a balanced ordering of Tanner
        graph vertices, and to group horizontal edges more coarsely than vertical ones.

        More specifically, this method constructs Tanner subgraphs as follows:

        1. For the classical seed code that defines vertical edges of this ``HGPCode``
           (``self.code_a``), color the edges of its Tanner graph, and number these colors starting
           from zero.
        2. Even edges get assigned a "north" or "south" direction if they are associated,
           respectively, with X-type or Z-type parity checks.  Odd edges get assigned the opposite
           direction.
        3. Step 1 is repeated for the classical seed code that defines horizontal edges of this
           ``HGPCode`` (``self.code_b``), but the resulting edges get an "east" or "west" direction
           according to the parity of their color alone: even colors go east and odd colors go west,
           irrespective of parity check type.

        Any two overlapping X-type and Z-type parity checks of an HGPCode share exactly two data
        qubits, so the induced circuit measures the correct syndrome only if the X-type gate comes
        before the Z-type gate on both of those qubits, or after it on both.  The check-type term in
        step 2 is what enforces that, sending vertical edges of one color but opposite check type to
        opposite ends of the returned sequence; grouping vertical edges by color alone breaks it.
        Horizontal edges need no such term, and grouping them by color alone halves the number of
        subgraphs they require.  Each color class of an edge coloring is a matching, so every
        subgraph, horizontal or vertical, addresses each qubit at most once and is therefore
        realizable as a single layer of gates.

        Args:
            strategy: The strategy used by nx.greedy_color to color edges of the Tanner graph.
                Default: "smallest_last".
        """
        node_map = HGPCode.get_product_node_map(self.code_a.graph.nodes, self.code_b.graph.nodes)

        # collect subgraphs of North and South edges
        edges_n: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        edges_s: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        coloring_a = nx.greedy_color(nx.line_graph(self.code_a.graph.to_undirected()), strategy)
        for (check_a, data_a), color in coloring_a.items():
            for node_b in self.code_b.graph.nodes:
                node_0 = node_map[check_a, node_b]
                node_1 = node_map[data_a, node_b]
                data, check = sorted([node_0, node_1])
                # node_b.is_data selects whether this edge's parity check is X-type or Z-type
                edges_ns = edges_s if (color + node_b.is_data) % 2 == 0 else edges_n
                edges_ns[color].append((check, data))
        graphs_n = tuple(self.graph.edge_subgraph(edges) for edges in edges_n.values())
        graphs_s = tuple(self.graph.edge_subgraph(edges) for edges in edges_s.values())

        # collect subgraphs of East and West edges
        edges_e: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        edges_w: dict[int, list[tuple[Node, Node]]] = collections.defaultdict(list)
        coloring_b = nx.greedy_color(nx.line_graph(self.code_b.graph.to_undirected()), strategy)
        for (check_b, data_b), color in coloring_b.items():
            for node_a in self.code_a.graph.nodes:
                node_0 = node_map[node_a, check_b]
                node_1 = node_map[node_a, data_b]
                data, check = sorted([node_0, node_1])
                # the check-type term of the vertical edges is deliberately absent here
                edges_ew = edges_e if color % 2 == 0 else edges_w
                edges_ew[color].append((check, data))
        graphs_e = tuple(self.graph.edge_subgraph(edges) for edges in edges_e.values())
        graphs_w = tuple(self.graph.edge_subgraph(edges) for edges in edges_w.values())

        return graphs_n + graphs_e + graphs_w + graphs_s

    @staticmethod
    def get_matrix_product(
        matrix_a: FieldOrRingArray, matrix_b: FieldOrRingArray
    ) -> tuple[FieldOrRingArray, FieldOrRingArray]:
        """Hypergraph product of two parity check matrices."""
        _kron = abstract.kron if isinstance(matrix_a, abstract.RingArray) else np.kron

        # construct the nontrivial blocks of the final parity check matrices
        mat_H1_In2 = _kron(matrix_a, np.eye(matrix_b.shape[1], dtype=int))
        mat_In1_H2 = _kron(np.eye(matrix_a.shape[1], dtype=int), matrix_b)
        mat_H1_T_Im2 = _kron(matrix_a.T, np.eye(matrix_b.shape[0], dtype=int))
        mat_Im1_H2_T = _kron(np.eye(matrix_a.shape[0], dtype=int), matrix_b.T)

        # construct the X-sector and Z-sector parity check matrices
        matrix_x = np.hstack([mat_H1_In2, mat_Im1_H2_T])
        # numpy's stubs type this np.kron result as a boolean array, which they cannot negate
        matrix_z = np.hstack([-mat_In1_H2, mat_H1_T_Im2])  # type: ignore[misc]
        return matrix_x.view(type(matrix_a)), matrix_z.view(type(matrix_a))

    @staticmethod
    def get_graph_product(graph_a: nx.DiGraph, graph_b: nx.DiGraph) -> TannerGraph:
        """Hypergraph product of two Tanner graphs, as the Tanner graph of a qudit code."""
        graph = nx.DiGraph()
        field = getattr(graph_a, "field", galois.GF2)
        _Pauli = Pauli if field is galois.GF2 else QuditPauli

        # the keys of this map are the vertices of the product, before relabeling
        node_map = HGPCode.get_product_node_map(graph_a.nodes, graph_b.nodes)
        graph.add_nodes_from(node_map.keys())

        # start with a cartesian products of the input graphs
        graph_product = nx.cartesian_product(graph_a, graph_b)

        # fix edge orientation, and tag each edge with a QuditPauli
        for node_fst, node_snd, data in graph_product.edges(data=True):
            # identify the sectors of two nodes
            sector_fst = HGPCode.get_sector(*node_fst)
            sector_snd = HGPCode.get_sector(*node_snd)

            # identify data-qudit vs. check nodes, and their sectors
            if sector_fst in [(0, 0), (1, 1)]:
                node_qudit, sector_qudit = node_fst, sector_fst
                node_check, sector_check = node_snd, sector_snd
            else:
                node_check, sector_check = node_fst, sector_fst
                node_qudit, sector_qudit = node_snd, sector_snd

            # start with an X-type operator
            op = _Pauli((data.get("val", 0), 0))

            # switch to Z-type operator for check qudits in the (0, 1) sector
            if sector_check == (0, 1):
                op = ~op

            # account for the minus sign in the (0, 0) sector of the Z-type subcode
            if isinstance(op, QuditPauli) and sector_qudit == (0, 0) and op.value[Pauli.Z]:
                op = -op

            graph.add_edge(node_check, node_qudit)
            graph[node_check][node_qudit][Pauli] = op

        # relabel nodes, from (node_a, node_b) --> node_combined
        return TannerGraph(nx.relabel_nodes(graph, node_map), field=field, symplectic=True)

    @staticmethod
    def get_sector(node_a: Node, node_b: Node) -> tuple[int, int]:
        """Get the sector of a node in a graph product."""
        return int(not node_a.is_data), int(not node_b.is_data)

    @staticmethod
    def get_product_node_map(
        nodes_a: Collection[Node], nodes_b: Collection[Node]
    ) -> dict[tuple[Node, Node], Node]:
        """Map (dictionary) that re-labels nodes in the hypergraph product of two codes."""
        # identify the number of data/check nodes, and the total number of X/Z checks
        num_data_a = sum(node.is_data for node in nodes_a)
        num_data_b = sum(node.is_data for node in nodes_b)
        num_checks_a = len(nodes_a) - num_data_a
        num_checks_b = len(nodes_b) - num_data_b
        num_checks_x = num_checks_a * num_data_b
        num_checks_z = num_checks_b * num_data_a
        num_checks = num_checks_x + num_checks_z

        node_map = {}
        index_data, index_check = 0, 0
        for node_a, node_b in itertools.product(sorted(nodes_a), sorted(nodes_b)):
            if HGPCode.get_sector(node_a, node_b) in [(0, 0), (1, 1)]:
                node = Node(index=index_data, is_data=True)
                index_data += 1
            else:
                # shift check node indices for consistency with the CSSCode parity check matrix
                index = (index_check + num_checks_x) % num_checks
                node = Node(index=index, is_data=False)
                index_check += 1
            node_map[node_a, node_b] = node
        return node_map

    @staticmethod
    def get_canonical_logical_line_ops(
        matrix_a: galois.FieldArray, matrix_b: galois.FieldArray
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Canonical logical line operators for a hypergraph product code.

        These operators are essentially those in Lemma 1 of arXiv:2204.10812v3, modified using pivot
        matrices similarly to Theorem VIII.10 of arXiv:2502.07150v1 to ensure pair-wise
        anti-commutation relations.

        X-type logical operators are "horizontal" in sector (0, 0) and "vertical" in sector (1, 1).
        Vice versa for Z-type logical operators.
        """
        field = type(matrix_a)

        generator_a = matrix_a.null_space()
        generator_b = matrix_b.null_space()
        generator_a_T = matrix_a.T.null_space()
        generator_b_T = matrix_b.T.null_space()

        pivots_a = field.Zeros(generator_a.shape)
        pivots_b = field.Zeros(generator_b.shape)
        pivots_a[range(len(pivots_a)), qldpc.math.first_nonzero_cols(generator_a)] = 1
        pivots_b[range(len(pivots_b)), qldpc.math.first_nonzero_cols(generator_b)] = 1

        pivots_a_T = field.Zeros(generator_a_T.shape)
        pivots_b_T = field.Zeros(generator_b_T.shape)
        pivots_a_T[range(len(pivots_a_T)), qldpc.math.first_nonzero_cols(generator_a_T)] = 1
        pivots_b_T[range(len(pivots_b_T)), qldpc.math.first_nonzero_cols(generator_b_T)] = 1

        logical_ops_x_l = np.kron(pivots_a, generator_b)
        logical_ops_z_l = np.kron(generator_a, pivots_b)
        logical_ops_x_r = np.kron(generator_a_T, pivots_b_T)
        logical_ops_z_r = np.kron(pivots_a_T, generator_b_T)

        logical_ops_x = scipy.linalg.block_diag(logical_ops_x_l, logical_ops_x_r)
        logical_ops_z = scipy.linalg.block_diag(logical_ops_z_l, logical_ops_z_r)
        return logical_ops_x.view(field), logical_ops_z.view(field)

    def _get_distance_exact(self, pauli: PauliXZ | None, *, method: DistanceMethod) -> int | float:
        """Exact distance calculation for hypergraph product codes.

        These calculations are based on arXiv:2308.15520, but additionally allow for the separate
        calculation of X-distance and Z-distance.  The basic idea is to identify the size of
        minimum-weight string operators in the (0, 0) and (1, 1) sectors of the HGPCode.

        Each sector only carries logical operators when both of the codes whose code words build it
        are nontrivial: the (0, 0) sector needs code words in both seed codes, and the (1, 1) sector
        needs them in both transpose seed codes, which is to say a dependent parity check in each
        seed code.  A sector that carries none contributes no weight, and folding its weight in
        anyway would report a distance below the true one, so skip it.
        """
        if pauli is None:
            # this case is implicitly covered by the cases of Pauli.X and Pauli.Z below
            return NotImplemented
        method = _get_classical_distance_method(method)

        # a transpose seed code has code words exactly when that seed code has a dependent check
        dependent_a = self.code_a.rank < len(self.code_a.matrix)
        dependent_b = self.code_b.rank < len(self.code_b.matrix)

        if pauli is Pauli.X:
            transpose_a = ClassicalCode(self.code_a.matrix.T)
            dist_a = transpose_a.get_distance(method=method) if dependent_b else np.nan
            dist_b = self.code_b.get_distance(method=method) if self.code_a.dimension else np.nan
        else:
            assert pauli is Pauli.Z
            dist_a = self.code_a.get_distance(method=method) if self.code_b.dimension else np.nan
            transpose_b = ClassicalCode(self.code_b.matrix.T)
            dist_b = transpose_b.get_distance(method=method) if dependent_a else np.nan

        return dist_a if np.isnan(dist_b) else dist_b if np.isnan(dist_a) else min(dist_a, dist_b)


class CHGPCode(HGPCode):
    """Cyclic hypergraph product code, called a CxC code in arXiv:2511.09683.

    A CHGPCode is a hypergraph product of classical CyclicCodes.

    References:

    - https://arxiv.org/abs/2511.09683 (Definition 1)
    """

    def __init__(
        self,
        dims: tuple[int, int] | int,
        poly_a: sympy.Basic,
        poly_b: sympy.Basic | None = None,
        field: int | type[galois.FieldArray] | None = None,
    ) -> None:
        """Construct a CHGPCode from two classical block lengths and two univariate polynomials.

        If provided only one block length or one polynomial, use it for both underlying CyclicCodes.
        """
        if isinstance(dims, Iterable):
            bits_a, bits_b = dims
        else:
            bits_a = bits_b = int(dims)
        code_a = CyclicCode(bits_a, poly_a, field)
        code_b = CyclicCode(bits_b, poly_b or poly_a, field)
        super().__init__(code_a, code_b, field)


class CRCode(HGPCode):
    """Repeated cyclic hypergraph product code, called a CxR code in arXiv:2511.09683.

    A CRCode is a hypergraph product of a CyclicCode and a RingCode.

    References:

    - https://arxiv.org/abs/2511.09683 (Definition 3)
    """

    def __init__(
        self, bits: int, poly: sympy.Basic, field: int | type[galois.FieldArray] | None = None
    ) -> None:
        """Construct a CRCode from the block length and univariate polynomial of a CyclicCode.

        The block length of the RingCode used in the hypergraph product is set to the distance of
        the CyclicCode.
        """
        cyclic_code = CyclicCode(bits, poly, field)
        distance = cyclic_code.get_distance()
        if np.isnan(distance):
            raise ValueError(
                "Cannot construct a CRCode from a cyclic code without a defined distance"
            )
        ring_code = RingCode(int(distance), field)
        super().__init__(cyclic_code, ring_code, field)


class SHPCode(CSSCode):
    """Subsystem hypergraph product code.

    A subsystem hypergraph product code (SHPCode) is constructed from two classical codes.  Unlike
    the ordinary hypergraph product code, an SHPCode depends only on the actual classical codes it
    is built from; in particular, an SHPCode does not depend on the choice of parity check matrices
    for the underlying classical codes.

    If the classical generating codes of an SHPCode have code parameters ``[n1, k1, d1]`` and
    ``[n2, k2, d2]``, the SHPCode has parameters ``[n1 n2, k1 k2, min(d1, d2)]``.

    References:

    - https://errorcorrectionzoo.org/c/subsystem_quantum_parity
    - https://arxiv.org/abs/2002.06257
    - https://arxiv.org/abs/2502.07150
    """

    def __init__(
        self,
        code_a: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        code_b: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]] | None = None,
        field: int | type[galois.FieldArray] | None = None,
        *,
        set_logicals: bool = True,
    ) -> None:
        """Subsystem hypergraph product of two classical codes, as in arXiv:2002.06257."""
        if code_b is None:
            code_b = code_a
        self.code_a = ClassicalCode(code_a, field)
        self.code_b = ClassicalCode(code_b, field)
        field = self.code_a.field

        matrix_x, matrix_z = SHPCode.get_matrix_product(self.code_a.matrix, self.code_b.matrix)
        super().__init__(
            matrix_x.view(np.ndarray).astype(int),
            matrix_z.view(np.ndarray).astype(int),
            field,
            is_subsystem_code=True,
        )

        stab_ops_x = np.kron(self.code_a.matrix, self.code_b.generator)
        stab_ops_z = np.kron(-self.code_a.generator, self.code_b.matrix)
        self._stabilizer_ops = scipy.linalg.block_diag(stab_ops_x, stab_ops_z).view(field)

        if set_logicals:
            logical_ops_xz = SHPCode.get_canonical_logical_line_ops(
                self.code_a.matrix, self.code_b.matrix
            )
            self.set_logical_ops_xz(*logical_ops_xz, skip_validation=True)

    @staticmethod
    def get_matrix_product(
        matrix_a: FieldOrRingArray, matrix_b: FieldOrRingArray
    ) -> tuple[FieldOrRingArray, FieldOrRingArray]:
        """Subsystem hypergraph product of two parity check matrices."""
        _kron = abstract.kron if isinstance(matrix_a, abstract.RingArray) else np.kron
        matrix_x = _kron(matrix_a, np.eye(matrix_b.shape[1], dtype=int))
        matrix_z = _kron(np.eye(matrix_a.shape[1], dtype=int), matrix_b)
        return matrix_x.view(type(matrix_a)), matrix_z.view(type(matrix_a))

    @staticmethod
    def get_canonical_logical_line_ops(
        matrix_a: galois.FieldArray, matrix_b: galois.FieldArray
    ) -> tuple[galois.FieldArray, galois.FieldArray]:
        """Canonical logical line operators for a subsystem hypergraph product code.

        These operators are essentially those in Theorem VIII.10 of arXiv:2502.07150v1, generalized
        slightly to account for the possibility that code_a != code_b.
        """
        field = type(matrix_a)

        generator_a = matrix_a.null_space()
        generator_b = matrix_b.null_space()

        pivots_a = field.Zeros(generator_a.shape)
        pivots_b = field.Zeros(generator_b.shape)
        pivots_a[range(len(pivots_a)), qldpc.math.first_nonzero_cols(generator_a)] = 1
        pivots_b[range(len(pivots_b)), qldpc.math.first_nonzero_cols(generator_b)] = 1

        logical_ops_x = np.kron(pivots_a, generator_b)
        logical_ops_z = np.kron(generator_a, pivots_b)
        return logical_ops_x.view(field), logical_ops_z.view(field)

    def _get_distance_exact(self, pauli: PauliXZ | None, *, method: DistanceMethod) -> int | float:
        """Exact distance calculation for subsystem hypergraph product codes."""
        method = _get_classical_distance_method(method)
        match pauli:
            case Pauli.X:
                return self.code_b.get_distance(method=method)
            case Pauli.Z:
                return self.code_a.get_distance(method=method)
            case _:
                return min(
                    self.code_a.get_distance(method=method), self.code_b.get_distance(method=method)
                )


class BaconShorCode(SHPCode):
    """Bacon-Shor code on a square grid, implemented as a subsystem hypergraph product code.

    References:

    - https://errorcorrectionzoo.org/c/bacon_shor
    """

    def __init__(
        self,
        rows: int,
        cols: int | None = None,
        field: int | type[galois.FieldArray] | None = None,
        *,
        set_logicals: bool = True,
    ) -> None:
        code_x = RepetitionCode(rows, field)
        code_z = RepetitionCode(cols, field) if cols is not None else None
        super().__init__(code_x, code_z, field, set_logicals=set_logicals)

        self._distance_x = cols if cols is not None else rows
        self._distance_z = rows


class SHYPSCode(SHPCode):
    """Subsystem hypergraph product simplex (SHYPS) code.

    Subsystem hypergraph product codes naturally inherit the automorphisms (symmetries) of the
    classical codes that they are built from.  The SHYPSCode is built from classical SimplexCodes
    that have a very large automorphism group, which gives SHYPSCodes a large set of
    SWAP-transversal Clifford operations.

    References:

    - https://errorcorrectionzoo.org/c/shyps
    - https://arxiv.org/abs/2502.07150
    """

    def __init__(
        self,
        dim_x: int,
        dim_z: int | None = None,
        field: int | type[galois.FieldArray] | None = None,
        *,
        set_logicals: bool = True,
    ) -> None:
        dim_z = dim_z if dim_z is not None else dim_x

        code_x = SimplexCode(dim_x, field)
        code_z = SimplexCode(dim_z, field)
        super().__init__(code_x, code_z, set_logicals=set_logicals)

        self._dimension = dim_x * dim_z


####################################################################################################
# distance-balanced codes


class DistanceBalancedCode(CSSCode):
    """Distance-balanced code, obtained from a CSS code and a classical code.

    If the parity checks of a classical code are linearly independent, distance balancing with this
    code multiplies the X-distance or Z-distance of a CSS code by the distance of the classical
    code, while preserving the other distance of the CSS code.  If ``pauli`` is X, the CSS code
    defines a chain complex ``C_2 --[H_X.T]--> C_1 --[H_Z]--> C_0`` and the classical code defines a
    chain complex ``D_1 --[H]--> D_0``, where ``H_X``, ``H_Z``, and ``H`` are parity-check matrices.
    The distance-balanced code then has Z-type checks, qudits, and X-type checks in degrees 1, 2,
    and 3 of the tensor product ``C ⨂ D``.  If ``pauli`` is Z, the roles of X and Z are exchanged.

    Suppose that ``pauli`` is X, that the CSS code has parameters ``[[n, k, (d_X, d_Z)]]`` and
    ``m_X = len(H_X)`` X-type checks, and that the classical code has parameters ``[n_C, k_C, d_C]``
    and ``m_C = len(H)`` checks.  The distance-balanced code then has ``n * n_C + m_X * m_C``
    qudits.  The homology ``H_k`` of a chain complex in degree ``k`` consists of the cycles in
    degree ``k`` modulo boundaries.  The X-type logical operators of the CSS code, modulo
    stabilizers, form ``H_1(C)``, those of the distance-balanced code form ``H_2(C ⨂ D)``, and the
    code words of the classical code form ``H_1(D)``.  The Künneth formula identifies ``H_2(C ⨂ D)``
    with ``H_1(C) ⨂ H_1(D) ⊕ H_2(C) ⨂ H_0(D)``, where ``H_2(C)`` and ``H_0(D)`` have dimensions
    ``m_X - rank(H_X)`` and ``m_C - rank(H)``.  The distance-balanced code therefore has
    ``k * k_C + (m_X - rank(H_X)) * (m_C - rank(H))`` logical qudits.  If the rows of ``H`` are
    linearly independent, ``H_0(D)`` vanishes, so the distance-balanced code has ``k * k_C`` logical
    qudits, and if this number is nonzero, it has distances ``(d_X * d_C, d_Z)``.

    Args:
        code: The CSS code, which must not be a subsystem code.
        auxiliary_code: The classical code.
        pauli: The Pauli type, X or Z, whose distance to multiply.

    References:

    - Theorem 4.2 of https://arxiv.org/abs/2004.07935
    - Equation (4) of https://arxiv.org/abs/2505.13679
    - https://errorcorrectionzoo.org/c/distance_balanced
    """

    code: CSSCode
    auxiliary_code: ClassicalCode
    pauli: PauliXZ
    chain: ChainComplex

    def __init__(
        self,
        code: CSSCode,
        auxiliary_code: ClassicalCode | npt.NDArray[np.int_] | Sequence[Sequence[int]],
        pauli: PauliXZLike,
    ) -> None:
        if code.is_subsystem_code:
            raise ValueError("Distance balancing requires a stabilizer code, not a subsystem code")
        self.code = code
        self.auxiliary_code = ClassicalCode(auxiliary_code, code.field)
        self.pauli = Pauli.coerce_xz(pauli)

        code_x, code_z = code.matrix_x, code.matrix_z
        if self.pauli is Pauli.Z:
            code_x, code_z = code_z, code_x
        css_chain = ChainComplex([code_z, code_x.T])
        auxiliary_chain = ChainComplex([self.auxiliary_code.matrix])
        self.chain = ChainComplex.tensor_product(css_chain, auxiliary_chain)

        matrix_x, matrix_z = self.chain.op(3).T, self.chain.op(2)
        assert not isinstance(matrix_x, abstract.RingArray)
        assert not isinstance(matrix_z, abstract.RingArray)
        if self.pauli is Pauli.Z:
            matrix_x, matrix_z = matrix_z, matrix_x
        super().__init__(matrix_x, matrix_z, is_subsystem_code=False)
