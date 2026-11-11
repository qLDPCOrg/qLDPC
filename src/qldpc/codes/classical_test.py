# SPDX-License-Identifier: Apache-2.0

"""Unit tests for classical.py."""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import networkx as nx
import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y

from qldpc import codes


def test_basic() -> None:
    """Repetition, ring, and Hamming codes."""
    num_bits = 4
    for code in [codes.RepetitionCode(num_bits), codes.RingCode(num_bits)]:
        params = (num_bits, 1, num_bits)
        assert (
            code.get_code_params() == codes.ClassicalCode(code.matrix).get_code_params() == params
        )
    ring_code = codes.RingCode(1)
    assert np.array_equal(ring_code.matrix, [[0]])
    assert ring_code.is_equiv_to(codes.RepetitionCode(1))
    assert codes.ClassicalCode(ring_code.matrix).get_code_params() == (1, 1, 1)

    # the rank of repetition and Hamming codes is independent of the field
    assert codes.RepetitionCode(3, 2).rank == codes.RepetitionCode(3, 3).rank
    assert codes.HammingCode(3, 2).rank == codes.HammingCode(3, 3).rank


def test_cyclic_codes() -> None:
    """Cyclic codes."""

    # the RingCode is a CyclicCode
    code_a = codes.CyclicCode(5, 1 - x, field=9)
    code_b = codes.RingCode(5, field=9)
    assert np.array_equal(code_a.matrix, code_b.matrix)

    # reproduce Table 2 from arxiv:2511.09683v2
    cyclic_codes = {
        (15, 1 + x + x**4): (15, 4, 8),
        (21, 1 + x + x**5): (21, 5, 10),
        (28, 1 + x**2 + x**4 + x**10): (28, 10, 6),
        (21, 1 + x + x**3 + x**8): (21, 7, 8),
        (30, 1 + x + x**2 + x**7): (30, 6, 14),
        (31, 1 + x + x**2 + x**6 + x**27): (31, 10, 10),
        (31, 1 + x + x**3 + x**9 + x**10): (31, 10, 12),
    }
    for (bits, poly), params in cyclic_codes.items():
        assert codes.CyclicCode(bits, poly).get_code_params() == params

    with pytest.raises(ValueError, match="not a univariate polynomial"):
        codes.CyclicCode(3, 4)

    with pytest.raises(ValueError, match="not a univariate polynomial"):
        codes.CyclicCode(5, x + y)


def test_special_codes() -> None:
    """More complicated classical codes."""
    code: codes.ClassicalCode

    bits, dimension = 3, 2
    code = codes.ReedSolomonCode(bits, dimension)
    assert code.dimension == codes.ClassicalCode(code.matrix).dimension == dimension

    bits, dimension, field = 7, 4, 2
    code = codes.BCHCode(bits, dimension, field)
    assert code.dimension == codes.ClassicalCode(code.matrix).dimension == dimension
    with pytest.raises(ValueError, match=rf"block lengths {field}\^m - 1"):
        codes.BCHCode(bits - 1, dimension, field)

    # with block length q - 1, a BCH code is a Reed-Solomon code, which is MDS: [q - 1, k, q - k]
    code = codes.BCHCode(2, 1, field=3)
    assert code.dimension == 1
    assert codes.ClassicalCode(code.matrix).get_code_params() == (2, 1, 2)

    order, size, field = 1, 3, 2
    code = codes.ReedMullerCode(order, size, field)
    assert (code.order, code.size) == (order, size)
    assert ~code == codes.ReedMullerCode(size - order - 1, size, field)
    assert code.dimension == len(code) - np.linalg.matrix_rank(code.matrix)

    with pytest.raises(ValueError, match="0 <= r <= m"):
        codes.ReedMullerCode(-1, 0)

    # the Hamming code can be recovered by puncturing the extended Hamming code
    assert codes.ClassicalCode.equiv(
        codes.HammingCode(4), codes.ExtendedHammingCode(4).punctured([0])
    )

    # Hamming and extended Hamming codes report the parameters that their parity checks bear out.
    # Rebuilding a code from its parity check matrix alone carries none of the parameters that its
    # constructor caches, so the rebuilt code has to compute them.
    for size in [2, 3, 4]:
        for code, params in [
            (codes.HammingCode(size), (2**size - 1, 2**size - 1 - size, 3)),
            (codes.ExtendedHammingCode(size), (2**size, 2**size - 1 - size, 4)),
        ]:
            assert code.get_code_params() == params
            assert codes.ClassicalCode(code.matrix).get_code_params() == params

    # the Golay code is a [23, 12, 7] code with minimum-weight (weight-8) parity checks; see
    # https://errorcorrectionzoo.org/c/golay
    code = codes.GolayCode()
    assert (
        code.get_code_params() == codes.ClassicalCode(code.matrix).get_code_params() == (23, 12, 7)
    )
    assert set(code.matrix.view(np.ndarray).sum(axis=1)) == {8}


def test_simplex_codes_over_fields() -> None:
    """A simplex code is the dual of a Hamming code: [(q**k - 1)/(q - 1), k, q**(k - 1)].

    Over F_2 the code has a cyclic presentation whose parity checks all have weight 3, which makes
    simplex codes useful as the building blocks of a SHYPSCode, so binary codes use it.  A cyclic
    code of length q**k - 1 over a larger field would instead be the (q - 1)-fold repetition of a
    simplex code, with (q - 1) times the block length and distance.
    """
    for field in [2, 3, 4]:
        for dim in [2, 3]:
            code = codes.SimplexCode(dim, field)
            params = ((field**dim - 1) // (field - 1), dim, field ** (dim - 1))

            # the parameters the constructor reports, and the same parameters computed by a code
            # rebuilt from the parity check matrix alone, which caches nothing
            assert code.get_code_params() == params
            assert codes.ClassicalCode(code.matrix).get_code_params() == params

            # over a larger field the code is built as, and equals, the dual of a Hamming code; the
            # binary cyclic presentation is the same code only up to a permutation of bits
            if field > 2:
                assert codes.ClassicalCode.equiv(
                    codes.SimplexCode(dim, field), ~codes.HammingCode(dim, field)
                )

    # the binary presentation has weight-3 parity checks, which SHYPS codes inherit.  Dimension 8
    # has no primitive trinomial, so its defining polynomial comes from a brute-force search.
    for dim in [2, 8]:
        matrix = codes.SimplexCode(dim).matrix
        assert np.all(np.count_nonzero(matrix.view(np.ndarray), axis=1) == 3)
        assert codes.ClassicalCode(matrix).get_code_params() == (2**dim - 1, dim, 2 ** (dim - 1))


def test_reed_muller_order_zero() -> None:
    """The order-zero Reed-Muller code RM(0, m) is the [2**m, 1, 2**m] repetition code."""
    for size in range(5):
        generator = codes.ReedMullerCode.get_generator(0, size)
        assert np.asarray(generator).ndim == 2  # a single row, not a flat vector
        code = codes.ReedMullerCode(0, size)
        params = (2**size, 1, 2**size)
        assert (
            code.get_code_params() == codes.ClassicalCode(code.matrix).get_code_params() == params
        )

    # the documented duality RM(r, m)^perp == RM(m - r - 1, m) reaches order zero at r == m - 1
    assert codes.ClassicalCode.equiv(~codes.ReedMullerCode(2, 3), codes.ReedMullerCode(0, 3))


def test_degenerate_code_sizes() -> None:
    """Code families reject parameters for which they are not defined."""
    size = 1
    with pytest.raises(ValueError, match="rank of at least 2"):
        codes.HammingCode(size)
    with pytest.raises(ValueError, match="rank of at least 2"):
        codes.ExtendedHammingCode(size)
    with pytest.raises(ValueError, match="dimension of at least 2"):
        codes.SimplexCode(size)
    with pytest.raises(ValueError, match="dimension of at least 2"):
        codes.SimplexCode.get_defining_polynomial(size)


def test_bch_block_lengths() -> None:
    """A BCH block length is valid exactly when it is field_order**m - 1 for an integer m >= 1."""
    # valid: q**m - 1.  Digits of these lengths are non-decimal in base q > 10, which a string-based
    # check on the base-q representation would reject.
    for length, order in [(1, 2), (15, 2), (8, 3), (120, 11), (168, 13), (16, 17)]:
        assert codes.BCHCode._is_valid_bch_length(length, order)

    # invalid: negative, or not one less than a power of the field order, or m == 0
    for length, order in [(-4, 2), (-1, 2), (0, 2), (6, 2), (14, 2), (7, 3), (119, 11)]:
        assert not codes.BCHCode._is_valid_bch_length(length, order)

    # a valid length over a field of order greater than 10 builds a code of the asked-for dimension,
    # which its parity checks have to agree with
    code = codes.BCHCode(10, 6, field=11)
    assert codes.ClassicalCode(code.matrix).dimension == 6


def test_tanner_code_preserves_input_graph() -> None:
    """Building a Tanner code leaves the given graph, and hence the resulting code, unchanged."""
    # a subcode whose automorphism group cannot absorb a relabeling of the subgraph edges
    subcode = codes.ClassicalCode([[0, 0, 1], [1, 1, 0]])
    subgraph = nx.complete_graph(4)
    for node_a, node_b in subgraph.edges:
        subgraph[node_a][node_b]["sort"] = {node_a: -node_b, node_b: -node_a}

    code = codes.TannerCode(subgraph, subcode)
    assert all("sort" in subgraph[node_a][node_b] for node_a, node_b in subgraph.edges)

    # a second code built from the same graph is the same code
    assert codes.ClassicalCode.equiv(code, codes.TannerCode(subgraph, subcode))


def test_tanner_code_requires_matching_degree() -> None:
    """A Tanner code requires every check node to have degree equal to the subcode block length."""
    subgraph = nx.Graph([(0, 3), (1, 3), (0, 4)])
    with pytest.raises(ValueError, match="but the subcode of this Tanner code"):
        codes.TannerCode(subgraph, codes.RepetitionCode(3))


def test_tanner_code_sorts_nodes_by_sort_attributes() -> None:
    """A Tanner code orders its checks and bits by "sort" node attributes, if they are given."""
    # Bits are frozensets, which are only partially ordered, so sorting them by their own order
    # would depend on the order in which they are added to the graph.
    bit_nodes = [frozenset([0]), frozenset([0, 1]), frozenset([2])]
    subcode = codes.ClassicalCode([[1, 1, 0]])

    def get_matrix(bit_order: Sequence[int], label_edges: bool) -> npt.NDArray[np.int_]:
        """Matrix of a Tanner code whose bits are added to its graph in the given order.

        If label_edges is True, edges have "sort" attributes, which differ from those of their bits.
        """
        subgraph = nx.DiGraph()
        for bit_index in bit_order:
            subgraph.add_node(bit_nodes[bit_index], sort=-bit_index)
        for source_index, source in enumerate(["x", "y"]):
            subgraph.add_node(source, sort=-source_index)
            for bit_index, bit_node in enumerate(bit_nodes):
                edge_data = {"sort": (bit_index + source_index) % 3} if label_edges else {}
                subgraph.add_edge(source, bit_node, **edge_data)
        return codes.TannerCode(subgraph, subcode).matrix.view(np.ndarray)

    # Checks are ordered as (y, x), and bits as (bit_nodes[2], bit_nodes[1], bit_nodes[0]).  The
    # edges of a check are ordered by their own "sort" attributes, if given, or else by those of
    # their bits.
    for label_edges, expected in [(True, [[1, 0, 1], [0, 1, 1]]), (False, [[1, 1, 0], [1, 1, 0]])]:
        for bit_order in itertools.permutations(range(len(bit_nodes))):
            np.testing.assert_array_equal(get_matrix(bit_order, label_edges), expected)


def test_tanner_code_on_undirected_graph_with_self_loops() -> None:
    """A Tanner code on an undirected graph orders its bits as the graph lists its edges.

    A self-loop adds 2 to the degree of its node, so its bit appears at two positions of the word
    that its node checks.  The columns of the parity check matrix of the subcode at those positions
    add.
    """
    # The graph lists its edges as (0,1), (0,0), (0,2), (1,1), (1,2), (2,2), so, for example, node 0
    # checks the word on bits (0, 1, 1, 2).  This order differs from the edges' own (partial)
    # order, in which the self-loop {0} precedes {0,1}.
    subgraph = nx.Graph([(0, 1), (0, 0), (0, 2), (1, 1), (1, 2), (2, 2)])
    assert list(subgraph.edges) == [(0, 1), (0, 0), (0, 2), (1, 1), (1, 2), (2, 2)]
    subcode = codes.ClassicalCode([[1, 2, 0, 1], [0, 1, 1, 0]], field=3)
    code = codes.TannerCode(subgraph, subcode)
    expected = [
        [1, 2, 1, 0, 0, 0],
        [0, 2, 0, 0, 0, 0],
        [1, 0, 0, 2, 1, 0],
        [0, 0, 0, 2, 0, 0],
        [0, 0, 1, 0, 2, 1],
        [0, 0, 0, 0, 1, 1],
    ]
    np.testing.assert_array_equal(code.matrix.view(np.ndarray), expected)


def test_tanner_code_on_graphs_with_parallel_edges() -> None:
    """Parallel edges are distinct bits of undirected graphs, and repeat bits of directed ones."""
    # each node of this undirected multigraph has two edges to each of the other nodes
    subgraph = nx.MultiGraph([(0, 1), (0, 1), (0, 2), (0, 2), (1, 2), (1, 2)])
    code = codes.TannerCode(subgraph, codes.ClassicalCode([[1, 0, 1, 0], [0, 1, 0, 1]]))
    supports = [tuple(np.flatnonzero(row)) for row in code.matrix.view(np.ndarray)]
    assert supports == [(0, 2), (1, 3), (0, 4), (1, 5), (2, 4), (3, 5)]

    # a directed multigraph whose check sees bit_0 at positions 0 and 2, as ordered by the "sort"
    # attributes of its edges
    directed_subgraph = nx.MultiDiGraph()
    directed_subgraph.add_edge("check", "bit_0", sort=2)
    directed_subgraph.add_edge("check", "bit_0", sort=0)
    directed_subgraph.add_edge("check", "bit_1", sort=1)
    code = codes.TannerCode(directed_subgraph, codes.ClassicalCode([[1, 2, 0]], field=3))
    np.testing.assert_array_equal(code.matrix.view(np.ndarray), [[1, 2]])


def test_tanner_code_isolated_nodes() -> None:
    """Isolated nodes of undirected graphs are checks, and those of directed graphs are bits."""
    subgraph = nx.cycle_graph(3)
    subgraph.add_node(3)
    with pytest.raises(ValueError, match="Node 3 has degree 0"):
        codes.TannerCode(subgraph, codes.RepetitionCode(2))

    directed_subgraph = nx.DiGraph([("check", "bit_0"), ("check", "bit_1")])
    directed_subgraph.add_node("bit_2")
    code = codes.TannerCode(directed_subgraph, codes.RepetitionCode(2))
    np.testing.assert_array_equal(code.matrix.view(np.ndarray), [[1, 1, 0]])

    # with a subcode on no bits, nodes have no edges, so the code has no checks
    subcode = codes.ClassicalCode(np.zeros((1, 0), dtype=int))
    assert codes.TannerCode(nx.empty_graph(3, create_using=nx.DiGraph), subcode).matrix.shape == (
        0,
        3,
    )
    assert codes.TannerCode(nx.empty_graph(3), subcode).matrix.shape == (0, 0)


def test_tanner_code_requires_sources_and_sinks() -> None:
    """Every node of a directed graph for a Tanner code is a check (source) or a bit (sink)."""
    subgraph = nx.DiGraph([("check", "middle"), ("middle", "bit"), ("check", "bit")])
    with pytest.raises(ValueError, match=r"Node middle .* both incoming and outgoing"):
        codes.TannerCode(subgraph, codes.RepetitionCode(2))


def test_tanner_code_directed_subgraph() -> None:
    """An undirected graph becomes a directed graph whose sinks are the undirected graph's edges."""
    assert type(codes.TannerCode.as_directed_subgraph(nx.path_graph(3))) is nx.DiGraph
    assert type(codes.TannerCode.as_directed_subgraph(nx.Graph([(0, 0)]))) is nx.MultiDiGraph
    assert type(codes.TannerCode.as_directed_subgraph(nx.MultiGraph([(0, 1)]))) is nx.MultiDiGraph

    # a multigraph with parallel edges, a self-loop, and an isolated node
    subgraph = nx.MultiGraph()
    subgraph.add_edge(0, 1, sort={0: 1, 1: 0})
    subgraph.add_edge(0, 1, sort={0: 0, 1: 1})
    subgraph.add_edge(0, 0, sort={0: 2})
    subgraph.add_node(2)
    directed_subgraph = codes.TannerCode.as_directed_subgraph(subgraph)
    assert type(directed_subgraph) is nx.MultiDiGraph

    edge_0, edge_1, loop = (frozenset([0, 1]), 0), (frozenset([0, 1]), 1), (frozenset([0]), 0)
    assert set(directed_subgraph.nodes) == {0, 1, edge_0, edge_1, loop}
    assert [directed_subgraph.nodes[node]["sort"] for node in [edge_0, edge_1, loop]] == [0, 1, 2]
    expected_edges = [
        (0, edge_0, 1),
        (0, edge_1, 0),
        (0, loop, 2),
        (0, loop, 2),
        (1, edge_0, 0),
        (1, edge_1, 1),
    ]
    assert sorted(directed_subgraph.edges(data="sort"), key=str) == sorted(expected_edges, key=str)


def test_tanner_code() -> None:
    """Classical Tanner codes on random regular graphs."""
    subcode = codes.ClassicalCode.random(5, 3)
    subgraph = nx.random_regular_graph(subcode.num_bits, subcode.num_bits * 2 + 2)

    tag = "sort_label"
    for node_a, node_b in subgraph.edges:
        subgraph[node_a][node_b]["sort"] = {node_a: tag, node_b: tag}

    code = codes.TannerCode(subgraph, subcode)
    assert code.num_bits == subgraph.number_of_edges()
    assert code.num_checks == subgraph.number_of_nodes() * code.subcode.num_checks
    assert all(code.subgraph.get_edge_data(*edge)["sort"] == tag for edge in code.subgraph.edges)
