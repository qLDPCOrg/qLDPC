# SPDX-License-Identifier: Apache-2.0

"""Unit tests for complexes.py."""

from __future__ import annotations

import galois
import numpy as np
import pytest

from qldpc import abstract, objects


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
