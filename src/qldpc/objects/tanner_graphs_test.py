# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tanner_graphs.py."""

from __future__ import annotations

import functools

import galois
import numpy as np

from qldpc import objects


def test_node() -> None:
    """Node properties."""
    node_d1 = objects.Node(1, is_data=True)
    node_d2 = objects.Node(2, is_data=True)
    node_c1 = objects.Node(1, is_data=False)
    assert node_d1 < node_d2 < node_c1
    assert str(node_d1) == "d_1"
    assert hash(node_c1) != hash(node_d1) != hash(node_d2)


def test_tanner_graph() -> None:
    """Tanner graphs of classical and symplectic parity check matrices, and their inverses."""
    data, check = objects.Node, functools.partial(objects.Node, is_data=False)
    pauli, qudit_pauli = objects.Pauli, objects.QuditPauli

    # a classical matrix, whose checks include one that addresses no bits
    matrix = galois.GF(3)([[0, 0, 0], [1, 2, 0]])
    graph = objects.TannerGraph.from_matrix(matrix)
    assert list(graph.nodes) == [data(0), data(1), data(2), check(0), check(1)]
    assert {(c_node, d_node): val for c_node, d_node, val in graph.edges(data="val")} == {
        (check(1), data(0)): 1,
        (check(1), data(1)): 2,
    }
    assert graph.field is type(matrix) and not graph.symplectic
    assert np.array_equal(graph.to_matrix(), matrix)
    assert np.array_equal(
        objects.TannerGraph.from_matrix([[1, 1]]).to_matrix(), galois.GF2([[1, 1]])
    )

    # graphs that networkx derives from a Tanner graph keep its field
    assert graph.copy().field is graph.subgraph([data(0), check(1)]).field is type(matrix)
    assert graph.to_undirected().graph == graph.graph

    # symplectic matrices, with edges labeled by the qubit or qudit operators of each check
    cases: list[tuple[galois.FieldArray, list[objects.Pauli] | list[objects.QuditPauli]]] = [
        (galois.GF2([[1, 1, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0]]), [pauli.X, pauli.Y, pauli.Z]),
        (
            galois.GF(3)([[1, 2, 0, 0, 1, 2], [0, 0, 0, 0, 0, 0]]),
            [qudit_pauli((1, 0)), qudit_pauli((2, 1)), qudit_pauli((0, 2))],
        ),
    ]
    for matrix, ops in cases:
        graph = objects.TannerGraph.from_matrix(matrix, symplectic=True)
        assert list(graph.nodes) == [data(0), data(1), data(2), check(0), check(1)]
        assert dict(graph[check(0)]) == {data(qudit): {pauli: op} for qudit, op in enumerate(ops)}
        assert graph.field is type(matrix) and graph.symplectic
        assert np.array_equal(graph.to_matrix(), matrix)

    # default and assigned attributes
    graph = objects.TannerGraph()
    assert graph.field is galois.GF2 and not graph.symplectic
    graph.field = galois.GF(3)
    assert graph.field is galois.GF(3)
