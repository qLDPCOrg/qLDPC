# SPDX-License-Identifier: Apache-2.0

"""Unit tests for objects.py."""

from __future__ import annotations

import functools

import galois
import numpy as np
import pytest

from qldpc import abstract, objects


def test_pauli() -> None:
    """Pauli operator capabilities."""
    for string in ["I", "X", "Y", "Z"]:
        assert str(objects.Pauli.from_string(string)) == string
    with pytest.raises(ValueError, match="Invalid Pauli operator"):
        objects.Pauli.from_string("Q")

    # from_string is case-sensitive, since QuditCode.from_strings relies on it
    with pytest.raises(ValueError, match="Invalid Pauli operator"):
        objects.Pauli.from_string("x")

    # Pauli.coerce_xz accepts Pauli.X, Pauli.Z, and case-insensitive "X" or "Z" strings
    xz_inputs: dict[objects.PauliXZ, tuple[objects.PauliXZLike, ...]] = {
        objects.Pauli.X: (objects.Pauli.X, "X", "x"),
        objects.Pauli.Z: (objects.Pauli.Z, "Z", "z"),
    }
    for expected_pauli, values in xz_inputs.items():
        for value in values:
            assert objects.Pauli.coerce_xz(value) is expected_pauli
    for invalid_value in [objects.Pauli.Y, "y", 0]:
        with pytest.raises(ValueError, match=r"Pauli\.X or Pauli\.Z"):
            objects.Pauli.coerce_xz(invalid_value)  # type: ignore[arg-type]

    assert ~objects.Pauli.Z == objects.Pauli.Z.swap_xz() == objects.Pauli.X
    assert ~objects.Pauli.X == objects.Pauli.X.swap_xz() == objects.Pauli.Z
    assert ~objects.Pauli.Y == objects.Pauli.Y
    assert ~objects.Pauli.I == objects.Pauli.I
    with pytest.raises(ValueError, match=r"Pauli\.X and Pauli\.Z"):
        objects.Pauli.Y.swap_xz()

    paulis = [objects.Pauli.I, objects.Pauli.Z, objects.Pauli.X, objects.Pauli.Y]
    table = [[paulis.index(pp * qq) for qq in paulis] for pp in paulis]
    group = abstract.Group.product(abstract.CyclicGroup(2), repeat=2)
    assert np.array_equal(table, group.table)

    assert objects.Pauli.X.index != objects.Pauli.Z.index
    for pauli in [objects.Pauli.I, objects.Pauli.Y]:
        with pytest.raises(TypeError, match="have an index"):
            assert pauli.index


def test_qudit_operator() -> None:
    """Qudit operator capabilities."""
    assert objects.QuditPauli((0, 0)) == objects.QuditPauli()
    assert objects.QuditPauli((0, 1)) == ~objects.QuditPauli((1, 0))
    assert -objects.QuditPauli((0, 1)) == objects.QuditPauli((0, -1))
    for op in ["I", "Y(1)", "X(1)*Z(2)"]:
        assert str(objects.QuditPauli.from_string(op)) == op
    for op in ["a*b*c", "a(1)", "*", "X(1)*", "", "X(²)"]:
        with pytest.raises(ValueError, match="Invalid qudit operator"):
            objects.QuditPauli.from_string(op)


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


def test_cayley_complex() -> None:
    """Construct and test Cayley complexes."""
    group: abstract.Group
    subset_a: list[abstract.GroupMember]
    subset_b: list[abstract.GroupMember]

    # raise error when trying to build a complex from non-symmetric generating sets
    with pytest.raises(ValueError, match="not symmetric"):
        group = abstract.CyclicGroup(3)
        subset_a = [group.generators[0]]
        objects.CayleyComplex(subset_a, bipartite=True)

    # quadripartite complex
    group = abstract.CyclicGroup(3)
    shift = group.generators[0]
    subset_a = [shift, ~shift]
    cayplex = objects.CayleyComplex(subset_a, bipartite=False)
    assert_valid_complex(cayplex)
    with pytest.raises(ValueError, match="do not satisfy Total No Conjugacy"):
        objects.CayleyComplex(subset_a, bipartite=True)

    # complexes that may be bipartite or quadripartite
    for bipartite in [True, False]:
        group = abstract.CyclicGroup(6)
        shift = group.generators[0]
        subset_a = [shift, shift**2, ~shift, (~shift) ** 2]
        subset_b = [shift**3]
        cayplex = objects.CayleyComplex(subset_a, subset_b, bipartite=bipartite)
        assert_valid_complex(cayplex)

        group = abstract.Group.product(abstract.CyclicGroup(2), abstract.CyclicGroup(5))
        shift_x, shift_y = group.generators
        subset_a = [shift_x * shift_y, ~(shift_x * shift_y)]
        subset_b = [shift_x * shift_y**2, ~(shift_x * shift_y**2)]
        cayplex = objects.CayleyComplex(subset_a, subset_b, bipartite=bipartite)
        assert_valid_complex(cayplex)


def assert_valid_complex(cayplex: objects.CayleyComplex) -> None:
    """Sanity check on the number of edges in a Cayley complex."""
    size_a = len(cayplex.subset_a)
    size_b = len(cayplex.subset_b)
    size_g = cayplex.graph.number_of_nodes()
    assert cayplex.graph.number_of_edges() == size_g * (size_a + size_b) // 2


def test_chain_complex(field: int = 3) -> None:
    """Chain complex construction and errors."""

    # tensor product of one-complexes
    matrix = np.random.randint(field, size=(2, 3))
    two_chain = objects.ChainComplex.tensor_product(matrix, matrix, field)
    assert not np.any(two_chain.op(0))
    assert not np.any(two_chain.op(two_chain.num_links + 1))
    with pytest.raises(ValueError, match="out of range"):
        two_chain.op(-1)

    # tensor product of a two-complex and its dual
    four_chain = objects.ChainComplex.tensor_product(two_chain, two_chain.T, field)
    four_chain._validate_ops()

    # tensor product of one-complexes over a group algebra
    ring = abstract.GroupRing(abstract.TrivialGroup(), field)
    ring_matrix = abstract.RingArray.build(matrix, ring)
    two_chain = objects.ChainComplex.tensor_product(ring_matrix, ring_matrix)
    assert not np.any(two_chain.op(0))
    assert not np.any(two_chain.op(two_chain.num_links + 1))

    # a tensor product over a nontrivial commutative group algebra must yield boundary operators
    # whose entries are all ring members (and can therefore be lifted to matrices), including the
    # three-or-more-link case where some operator blocks are entirely zero
    cyclic_ring = abstract.GroupRing(abstract.CyclicGroup(3), field)
    cyclic_matrix = abstract.RingArray.build(matrix, cyclic_ring)
    ring_chain = objects.ChainComplex.tensor_product(cyclic_matrix, cyclic_matrix)
    ring_chain = objects.ChainComplex.tensor_product(ring_chain, cyclic_matrix)
    ring_chain._validate_ops()
    for ring_op in ring_chain.ops:
        # every op of a ring-valued chain is a RingArray; assert narrows the union type for .lift()
        assert isinstance(ring_op, abstract.RingArray)
        ring_op.lift()

    # invalid chain complex constructions
    with pytest.raises(ValueError, match="inconsistent operator types"):
        objects.ChainComplex([matrix, abstract.RingArray.build([[0]])])
    with pytest.raises(ValueError, match="Inconsistent base fields"):
        objects.ChainComplex([galois.GF(field)(matrix)], field=field**2)
    with pytest.raises(ValueError, match=r"boundary operators .* must compose to zero"):
        objects.ChainComplex([matrix] * 2, field=field)
    with pytest.raises(ValueError, match="different base fields"):
        objects.ChainComplex.tensor_product(galois.GF(field)(matrix), galois.GF(field**2)(matrix))

    # tensor products of chain complexes over non-commutative rings are not supported
    with pytest.raises(ValueError, match="non-commutative rings are not supported"):
        ring = abstract.GroupRing(abstract.DihedralGroup(3), field=2)
        ring_matrix = abstract.RingArray.build([[1]], ring)
        objects.ChainComplex.tensor_product(ring_matrix, ring_matrix)
