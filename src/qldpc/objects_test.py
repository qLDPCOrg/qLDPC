"""Unit tests for objects.py

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

    assert ~objects.Pauli.Z == objects.Pauli.Z.swap_xz() == objects.Pauli.X
    assert ~objects.Pauli.X == objects.Pauli.X.swap_xz() == objects.Pauli.Z
    assert ~objects.Pauli.Y == objects.Pauli.Y
    assert ~objects.Pauli.I == objects.Pauli.I
    with pytest.raises(ValueError, match="Pauli.X and Pauli.Z"):
        objects.Pauli.Y.swap_xz()

    paulis = [objects.Pauli.I, objects.Pauli.Z, objects.Pauli.X, objects.Pauli.Y]
    table = [[paulis.index(pp * qq) for qq in paulis] for pp in paulis]
    group = abstract.Group.product(abstract.CyclicGroup(2), repeat=2)
    assert np.array_equal(table, group.table)

    assert objects.Pauli.X.index != objects.Pauli.Z.index
    for pauli in [objects.Pauli.I, objects.Pauli.Y]:
        with pytest.raises(AttributeError, match="No index"):
            assert pauli.index


def test_qudit_operator() -> None:
    """Qudit operator capabilities."""
    assert objects.QuditPauli((0, 0)) == objects.QuditPauli()
    assert objects.QuditPauli((0, 1)) == ~objects.QuditPauli((1, 0))
    assert -objects.QuditPauli((0, 1)) == objects.QuditPauli((0, -1))
    for op in ["I", "Y(1)", "X(1)*Z(2)"]:
        assert str(objects.QuditPauli.from_string(op)) == op
    for op in ["a*b*c", "a(1)"]:
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

    # tensor product of a two-complex and its dual
    four_chain = objects.ChainComplex.tensor_product(two_chain, two_chain.T, field)
    four_chain._validate_ops()

    # tensor product of one-complexes over a group algebra
    ring = abstract.GroupRing(abstract.TrivialGroup(), field)
    ring_matrix = abstract.RingArray.build(matrix, ring)
    two_chain = objects.ChainComplex.tensor_product(ring_matrix, ring_matrix, field)
    assert not np.any(two_chain.op(0))
    assert not np.any(two_chain.op(two_chain.num_links + 1))

    # invalid chain complex constructions
    with pytest.raises(ValueError, match="inconsistent operator types"):
        objects.ChainComplex([matrix, abstract.TrivialGroup.to_ring_array([[0]])])
    with pytest.raises(ValueError, match="Inconsistent base fields"):
        objects.ChainComplex([galois.GF(field)(matrix)], field=field**2)
    with pytest.raises(ValueError, match="boundary operators .* must compose to zero"):
        objects.ChainComplex([matrix] * 2, field=field)
    with pytest.raises(ValueError, match="different base fields"):
        objects.ChainComplex.tensor_product(galois.GF(field)(matrix), galois.GF(field**2)(matrix))
