"""Classical error-correcting codes

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import galois
import networkx as nx
import numpy as np
import numpy.typing as npt

from qldpc.abstract import DEFAULT_FIELD_ORDER

from .common import ClassicalCode


class RepetitionCode(ClassicalCode):
    """Classical repetition code."""

    def __init__(self, bits: int, field: int | None = None) -> None:
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        self._matrix = self.field.Zeros((bits - 1, bits))
        for row in range(bits - 1):
            self._matrix[row, row] = 1
            self._matrix[row, row + 1] = -self.field(1)

        self._dimension = 1
        self._distance = bits


class RingCode(ClassicalCode):
    """Classical ring code: repetition code with periodic boundary conditions."""

    def __init__(self, bits: int, field: int | None = None) -> None:
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        self._matrix = self.field.Zeros((bits, bits))
        for row in range(bits):
            self._matrix[row, row] = 1
            self._matrix[row, (row + 1) % bits] = -self.field(1)

        self._dimension = 1
        self._distance = bits


class HammingCode(ClassicalCode):
    """Classical Hamming code."""

    def __init__(self, rank: int, field: int | None = None) -> None:
        """Construct a Hamming code of a given rank."""
        self._distance = 3
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)
        if self.field.order == 2:
            # parity check matrix: columns = all nonzero bitstrings
            bitstrings = list(itertools.product([0, 1], repeat=rank))
            self._matrix = self.field(bitstrings[1:]).T

        else:
            # More generally, columns = [maximal set of linearly independent strings], so collect
            # together all strings whose first nonzero element is a 1.
            strings = [
                (0,) * top_row + (1,) + rest
                for top_row in range(rank - 1, -1, -1)
                for rest in itertools.product(range(self.field.order), repeat=rank - top_row - 1)
            ]
            self._matrix = self.field(strings).T

        self._dimension = len(self) - len(self._matrix)
        self._distance = 3


class ExtendedHammingCode(ClassicalCode):
    """Classical extended Hamming code: the ordinary Hamming code with an extra parity bit.

    The extended Hamming code of rank r is also equal to ReedMullerCode(r-2, r).
    """

    def __init__(self, rank: int) -> None:
        """Construct an extended Hamming code of a given rank."""
        matrix: npt.NDArray[np.int_] = HammingCode(rank, field=2).matrix
        matrix = np.column_stack([np.zeros(matrix.shape[0], dtype=int), matrix])
        matrix = np.vstack([np.ones(matrix.shape[1], dtype=int), matrix])
        matrix[0] += matrix[1]
        ClassicalCode.__init__(self, matrix)

        self._dimension = len(self) - len(self._matrix)
        self._distance = 4


class ReedSolomonCode(ClassicalCode):
    """Classical Reed-Solomon code.

    Source: https://galois.readthedocs.io/en/v0.3.8/api/galois.ReedSolomon/
    References:
    - https://errorcorrectionzoo.org/c/reed_solomon
    - https://www.cs.cmu.edu/~venkatg/teaching/codingtheory/notes/notes6.pdf
    """

    def __init__(self, bits: int, dimension: int) -> None:
        ClassicalCode.__init__(self, galois.ReedSolomon(bits, dimension).H)
        self._dimension = dimension


class BCHCode(ClassicalCode):
    """Classical binary BCH code.

    Source: https://galois.readthedocs.io/en/v0.3.8/api/galois.BCH/
    References:
    - https://errorcorrectionzoo.org/c/bch
    - https://www.cs.cmu.edu/~venkatg/teaching/codingtheory/notes/notes6.pdf
    """

    def __init__(self, length: int, dimension: int, field: int | None = None) -> None:
        field = field or DEFAULT_FIELD_ORDER
        length_in_base = np.base_repr(length, base=field)
        if not length_in_base == str(field - 1) * len(length_in_base):
            raise ValueError(
                f"BCH codes over F_{field} are only defined for block lengths {field}^m - 1 with"
                " integer m."
            )
        ClassicalCode.__init__(self, galois.BCH(length, dimension, field=galois.GF(field)).H)
        self._dimension = dimension


class ReedMullerCode(ClassicalCode):
    """Classical Reed-Muller code.

    References:
    - https://errorcorrectionzoo.org/c/reed_muller
    - https://feog.github.io/10-coding.pdf
    """

    def __init__(self, order: int, size: int, field: int | None = None) -> None:
        self._assert_valid_params(order, size)
        self._order = order
        self._size = size

        generator = ReedMullerCode.get_generator(order, size)
        self._matrix = ClassicalCode(generator, field).generator
        self._field = galois.GF(field or DEFAULT_FIELD_ORDER)

        self._dimension = len(generator)
        self._distance = 2 ** (size - order)

    @staticmethod
    def get_generator(order: int, size: int) -> npt.NDArray[np.int_]:
        """Get the generator matrix for the specified Reed-Muller code."""
        ReedMullerCode._assert_valid_params(order, size)

        if order == 0:
            return np.ones(2**size, dtype=int)
        if order == size:
            return np.identity(2**size, dtype=int)

        mat_a = ReedMullerCode.get_generator(order, size - 1)
        mat_b = ReedMullerCode.get_generator(order - 1, size - 1)
        mat_z = np.zeros_like(mat_b)
        return np.block([[mat_a, mat_a], [mat_z, mat_b]]).astype(int)

    @staticmethod
    def _assert_valid_params(order: int, size: int) -> None:
        if not (size >= 0 and 0 <= order <= size):
            raise ValueError(
                "Reed-Muller code R(r,m) must have m >= 0 and 0 <= r <= m\n"
                + f"Provided: (r,m) = ({order},{size})"
            )


class TannerCode(ClassicalCode):
    """Classical Tanner code, as described in DOI:10.1109/TIT.1981.1056404.

    A Tanner code T(G,C) is constructed from:
    [1] A bipartite "half-regular" graph G.  That is, a graph...
        ... with two sets of nodes, V and W.
        ... in which all nodes in V have degree n.
    [2] A classical code C on n bits.

    For convenience, we make G directed, with edges directed from V to W.  The node sets V and W can
    then be identified, respectively, by the sources and sinks of G.

    The Tanner code T(G,C) is defined on |W| bits.  A |W|-bit string x is a code word of T(G,C) iff,
    for every node v in V, the bits of x incident to v are a code word of C.

    This construction requires an ordering the edges E(v) adjacent to each vertex v.  This class
    sorts E(v) by the value of the "sort" attribute attached to each edge.  If there is no "sort"
    attribute, its value is treated as corresponding neighbor of v.

    Tanner codes can similarly be defined on regular (undirected) graphs G' = (V',E') by placing
    checks on V' and bits on E'.

    Notes:
    - If the subcode C has m checks, its parity matrix has shape (m,n).
    - The code T(G,C) has |W| bits and |V|m checks.
    """

    subgraph: nx.DiGraph
    subcode: ClassicalCode

    def __init__(self, subgraph: nx.Graph, subcode: ClassicalCode) -> None:
        """Construct a classical Tanner code."""
        if not isinstance(subgraph, nx.DiGraph):
            subgraph = TannerCode.as_directed_subgraph(subgraph)

        self.subgraph = subgraph
        self.subcode = subcode
        sources = [node for node in subgraph if subgraph.in_degree(node) == 0]
        sinks = [node for node in subgraph if subgraph.out_degree(node) == 0]
        sink_indices = {sink: idx for idx, sink in enumerate(sorted(sinks))}

        num_bits = len(sinks)
        num_checks = len(sources) * subcode.num_checks
        matrix = np.zeros((num_checks, num_bits), dtype=int)
        for idx, source in enumerate(sorted(sources)):
            checks = range(subcode.num_checks * idx, subcode.num_checks * (idx + 1))
            bits = [sink_indices[sink] for sink in self._get_sorted_neighbors(source)]
            matrix[np.ix_(checks, bits)] = subcode.matrix
        ClassicalCode.__init__(self, matrix, subcode.field.order)

    def _get_sorted_neighbors(self, node: object) -> Sequence[object]:
        """Sorted neighbors of the given node."""
        return sorted(
            self.subgraph.neighbors(node),
            key=lambda neighbor: self.subgraph[node][neighbor].get("sort", neighbor),
        )

    @staticmethod
    def as_directed_subgraph(subgraph: nx.Graph) -> nx.DiGraph:
        """Convert an undirected graph for a Tanner code into a directed graph for the same code."""
        directed_subgraph = nx.DiGraph()
        for node_a, node_b, edge_data in subgraph.edges(data=True):
            edge = frozenset([node_a, node_b])
            directed_subgraph.add_edge(node_a, edge)
            directed_subgraph.add_edge(node_b, edge)
            if (sort_data := edge_data.pop("sort", None)) is not None:
                directed_subgraph[node_a][edge]["sort"] = sort_data[node_a]
                directed_subgraph[node_b][edge]["sort"] = sort_data[node_b]
        return directed_subgraph


class SimplexCode(ClassicalCode):
    """Classical simplex codes, with code parameters [2**k - 1, k, 2 ** (k - 1)].

    The automorphism group of SimplexCode(k) is the general linear group GL(k).

    References:
    - https://errorcorrectionzoo.org/c/simplex
    - https://arxiv.org/abs/2502.07150
    """

    def __init__(self, dim: int) -> None:
        block_length = 2**dim - 1

        matrix = np.zeros([block_length] * 2, dtype=int)
        rows = np.arange(block_length, dtype=int)
        for shift in SimplexCode.get_polynomial_exponents(dim):
            matrix[rows, (rows + shift) % block_length] = 1
        ClassicalCode.__init__(self, matrix, field=2)

        self._dimension = dim
        self._distance = 2 ** (dim - 1)

    @staticmethod
    def get_polynomial_exponents(dim: int) -> tuple[int, ...]:
        """Exponents of the polynomial that defines a SimplexCode of a given dimension.

        The polynomial is a primitive polynomial (i.e., "prime", with no factors) of the form
            h(x) = 1 + x**c + x**dim,
        where x is the generator of a cyclic group of order 2**dim - 1, and c is an integer.
        """
        # QUESTION: would any polynomial in galois.primitive_polys(2, dim, terms=3) suffice here?
        if dim == 2:
            return 0, 1, 2
        if dim == 3:
            return 0, 2, 3
        if dim == 4:
            return 0, 3, 4
        if dim == 5:
            return 0, 3, 5
        if dim == 6:
            return 0, 5, 6
        if dim == 7:
            return 0, 6, 7
        raise ValueError(
            f"Classical simplex codes are only supported for dimensions >1, <=7 (provided: {dim})"
        )
