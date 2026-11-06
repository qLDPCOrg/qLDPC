# SPDX-License-Identifier: Apache-2.0

"""Tanner graphs of parity check matrices."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import galois
import numpy as np
import numpy.typing as npt

from qldpc._utils import networkx as nx

from .paulis import Pauli, QuditPauli


@dataclasses.dataclass(frozen=True)
class Node:
    """Node in a Tanner graph.

    A node is essentially an integer index, together with a boolean flag to distinguish "data" node
    from a "check" node in an error-correcting code.
    """

    index: int
    is_data: bool = True

    def __lt__(self, other: Node) -> bool:
        if self.is_data == other.is_data:
            return self.index < other.index
        return self.is_data  # data bits "precede" check bits

    def __str__(self) -> str:
        tag = "d" if self.is_data else "c"
        return f"{tag}_{self.index}"


class TannerGraph(nx.DiGraph):
    """Tanner graph of a parity check matrix, with edges directed from check nodes to data nodes.

    A Tanner graph is a bipartite graph whose nodes are the data (qu)dits and the checks of a code,
    identified by Node objects.  The check node c and the data node d share an edge iff check c
    addresses d.  Every row of the parity check matrix has a check node, so a check that addresses
    nothing is an isolated node, and the graph records the number of checks.

    A Tanner graph of a classical parity check matrix H (with ``symplectic=False``) has a data node
    for every column of H.  Edge (c, d) is in the graph iff ``H[c, d] != 0``, and its "val" data is
    the entry ``H[c, d]``.

    A Tanner graph of a symplectic parity check matrix (with ``symplectic=True``), whose rows have
    the form ``[X|Z]``, has a data node for every qudit, that is, for every pair of columns (j, j+n)
    of the matrix, where n is the number of qudits.  The data of edge (c, d) maps the Pauli class to
    the operator that check c applies to qudit d: a Pauli if the field is GF(2), and a QuditPauli
    otherwise.

    The field and symplectic attributes are stored in the graph attribute dictionary, ``graph``, so
    that copies, subgraph views, and other graphs that networkx derives from this one keep them.
    """

    @staticmethod
    def from_matrix(
        matrix: npt.NDArray[np.int_] | Sequence[Sequence[int]], *, symplectic: bool = False
    ) -> TannerGraph:
        """Build the Tanner graph of a parity check matrix.

        Args:
            matrix: The parity check matrix.  Its field is that of a galois.FieldArray, or GF(2)
                otherwise.
            symplectic: Whether the rows of the matrix are symplectic vectors of the form
                ``[X|Z]``.  Defaults to False.

        Returns:
            The Tanner graph of the matrix, with data nodes before check nodes.
        """
        matrix = np.asanyarray(matrix)
        field = type(matrix) if isinstance(matrix, galois.FieldArray) else galois.GF2
        graph = TannerGraph(field=field, symplectic=symplectic)
        if not symplectic:
            graph.add_nodes_from(Node(index=bit, is_data=True) for bit in range(matrix.shape[-1]))
            graph.add_nodes_from(Node(index=check, is_data=False) for check in range(len(matrix)))
            for row, col in zip(*np.nonzero(matrix)):
                node_check = Node(index=int(row), is_data=False)
                node_bit = Node(index=int(col), is_data=True)
                graph.add_edge(node_check, node_bit, val=matrix[row][col])
            return graph

        matrix = np.reshape(matrix, (len(matrix), 2, matrix.shape[-1] // 2))
        graph.add_nodes_from(Node(index=qudit, is_data=True) for qudit in range(matrix.shape[-1]))
        graph.add_nodes_from(Node(index=check, is_data=False) for check in range(len(matrix)))
        _Pauli = Pauli if field is galois.GF2 else QuditPauli
        for row, xz, col in zip(*np.nonzero(matrix)):
            node_check = Node(index=int(row), is_data=False)
            node_qudit = Node(index=int(col), is_data=True)
            graph.add_edge(node_check, node_qudit)
            qudit_op = graph[node_check][node_qudit].get(Pauli, _Pauli((0, 0)))
            vals_xz = list(qudit_op.value)
            vals_xz[xz] += int(matrix[row, xz, col])
            graph[node_check][node_qudit][Pauli] = _Pauli(tuple(vals_xz))
        return graph

    @property
    def field(self) -> type[galois.FieldArray]:
        """The field of the parity check matrix of this graph, GF(2) by default."""
        field: type[galois.FieldArray] = self.graph.get("field", galois.GF2)
        return field

    @field.setter
    def field(self, field: type[galois.FieldArray]) -> None:
        self.graph["field"] = field

    @property
    def symplectic(self) -> bool:
        """Whether this graph represents a symplectic parity check matrix, False by default."""
        return bool(self.graph.get("symplectic", False))

    def to_matrix(self) -> galois.FieldArray:
        """The parity check matrix of this Tanner graph."""
        num_data = sum(node.is_data for node in self.nodes())
        num_checks = len(self.nodes()) - num_data
        if not self.symplectic:
            matrix = self.field.Zeros((num_checks, num_data))
            for node_check, node_bit, data in self.edges(data=True):
                matrix[node_check.index, node_bit.index] = data.get("val", 1)
            return matrix

        int_matrix = np.zeros((num_checks, 2, num_data), dtype=int)
        for node_check, node_qudit, data in self.edges(data=True):
            int_matrix[node_check.index, :, node_qudit.index] = data.get(Pauli).value
        return self.field(int_matrix.reshape(num_checks, 2 * num_data))
