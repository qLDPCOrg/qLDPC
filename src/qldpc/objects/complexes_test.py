# SPDX-License-Identifier: Apache-2.0

"""Unit tests for complexes.py."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import galois
import numpy as np
import pytest
import scipy.linalg
import sympy.combinatorics as comb

from qldpc import abstract, objects
from qldpc.conftest import run_python
from qldpc.objects import complexes


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

    # either generating subset may be empty, but not both
    shift = abstract.CyclicGroup(2).generators[0]
    for subsets in [([], [shift]), ([shift], [])]:
        assert objects.CayleyComplex(*subsets).graph.number_of_nodes() == 2
    with pytest.raises(ValueError, match="at least one generating member"):
        objects.CayleyComplex([], [])
    with pytest.raises(ValueError, match="at least one generating member"):
        objects.CayleyComplex.build_cayley_graph([])

    # subsets may be given as any iterables, such as arrays and iterators
    shift = abstract.CyclicGroup(3).generators[0]
    members = np.array([shift, ~shift], dtype=object)
    assert objects.CayleyComplex(members).subset_a == {shift, ~shift}
    assert objects.CayleyComplex(iter([shift, ~shift])).subset_b == {shift, ~shift}

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


def test_cayley_complex_reads_members_on_one_number_of_points() -> None:
    """Members that differ only in trailing fixed points are one member of a Cayley complex."""
    swap, padded_swap = abstract.GroupMember([1, 0]), abstract.GroupMember([1, 0, 2])
    reflection = abstract.GroupMember([0, 2, 1])
    sympy_swap = comb.Permutation(0, 1)  # in cycle notation, on two points
    for subsets, padded_subsets in [
        (([swap, reflection],), ([padded_swap, reflection],)),
        (([swap], [reflection]), ([padded_swap], [reflection])),
        (([sympy_swap, reflection],), ([padded_swap, reflection],)),
    ]:
        cayplex = objects.CayleyComplex(*subsets)
        padded_cayplex = objects.CayleyComplex(*padded_subsets)
        assert cayplex.graph.number_of_nodes() == padded_cayplex.graph.number_of_nodes()


def test_cayley_graph_is_independent_of_hash_seed() -> None:
    """Cayley graphs list their nodes and edges in an order that does not depend on the hash seed.

    The hash seed sets the iteration order of the sets of group members that graphs are built from.
    Child interpreters with fixed hash seeds must reproduce the graph built here, under this
    interpreter's hash seed.
    """
    script = (
        "from qldpc import abstract, objects;"
        "graph = objects.CayleyComplex.build_cayley_graph(set(abstract.SymmetricGroup(3).generators));"
        "print([tuple(node.array_form) for node in graph.nodes]);"
        "print([(tuple(u.array_form), tuple(v.array_form)) for u, v in graph.edges])"
    )
    outputs = {run_python(script, env={"PYTHONHASHSEED": hash_seed}) for hash_seed in ["0", "1"]}
    graph = objects.CayleyComplex.build_cayley_graph(set(abstract.SymmetricGroup(3).generators))
    nodes = [tuple(node.array_form) for node in graph.nodes]
    edges = [(tuple(u.array_form), tuple(v.array_form)) for u, v in graph.edges]
    assert outputs == {f"{nodes}\n{edges}\n"}


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


def test_balanced_product() -> None:
    """Balanced products of chain complexes are quotients of tensor products."""
    # non-free symmetries
    chain_a = objects.ChainComplex(
        [np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 0]])], field=3
    )
    chain_b = objects.ChainComplex(
        [np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])], field=3
    )
    swap_one, swap_two = [1, 0, 2, 3], [1, 0, 3, 2]
    symmetries_a = [(swap_one, swap_two)]
    symmetries_b = [(swap_one, comb.Permutation(3))]
    assert_valid_balanced_product(chain_a, chain_b, symmetries_a, symmetries_b, [10, 20, 8])

    # orbits that contribute to the same boundary operator entry, whose contributions cancel
    chain = objects.ChainComplex([np.ones((2, 1), dtype=int)])
    assert_valid_balanced_product(chain, chain, [([1, 0], [0])], [([1, 0], [0])], [2, 2, 1])

    # without symmetries, the balanced product is the tensor product, here of a two-link complex
    # (with an empty module) over a field in which signs matter
    chain_c = objects.ChainComplex([np.zeros((0, 2), dtype=int), np.array([[1], [2]])], field=3)
    chain_d = objects.ChainComplex([np.array([[1, 1]])], field=3)
    assert_valid_balanced_product(chain_c, chain_d, [], [], [0, 2, 5, 2])

    # a symmetry can also be one permutation of the bases of all modules, in order of degree
    member_a = abstract.GroupMember(swap_one) @ abstract.GroupMember(swap_two)
    balanced = objects.ChainComplex.balanced_product(chain_a, chain_b, symmetries_a, symmetries_b)
    balanced_from_members = objects.ChainComplex.balanced_product(
        chain_a, chain_b, [member_a], [comb.Permutation(0, 1)]
    )
    assert all(np.array_equal(*ops) for ops in zip(balanced.ops, balanced_from_members.ops))

    ring_chain = objects.ChainComplex(
        [abstract.RingArray.build([[1]], abstract.GroupRing(abstract.TrivialGroup()))]
    )
    with pytest.raises(ValueError, match="over rings are not supported"):
        objects.ChainComplex.balanced_product(chain_a, ring_chain, [], [])
    with pytest.raises(ValueError, match="same field"):
        objects.ChainComplex.balanced_product(chain_a, chain, [], [])
    with pytest.raises(ValueError, match="same number of symmetries"):
        objects.ChainComplex.balanced_product(chain_a, chain_b, symmetries_a, [])
    with pytest.raises(ValueError, match=r"symmetries_a\[0\]\[1\] must be a permutation"):
        objects.ChainComplex.balanced_product(chain_a, chain_b, [(swap_one, [0])], symmetries_b)
    with pytest.raises(ValueError, match=r"symmetries_a\[0\] is not a symmetry"):
        objects.ChainComplex.balanced_product(
            chain_a, chain_b, [(swap_two, swap_one)], symmetries_b
        )


def assert_valid_balanced_product(
    chain_a: objects.ChainComplex,
    chain_b: objects.ChainComplex,
    symmetries_a: Sequence[Sequence[Iterable[int]]],
    symmetries_b: Sequence[Sequence[Iterable[int]]],
    dims: list[int],
) -> None:
    """Check that projecting onto orbits maps a tensor product onto a balanced product."""
    tensor = objects.ChainComplex.tensor_product(chain_a, chain_b)
    balanced = objects.ChainComplex.balanced_product(chain_a, chain_b, symmetries_a, symmetries_b)
    assert balanced.num_links == tensor.num_links
    assert [balanced.dim(degree) for degree in range(len(dims))] == dims

    projections = []
    for degree in range(tensor.num_links + 1):
        blocks = []
        for deg_a, deg_b in complexes._get_degree_pairs(chain_a, chain_b, degree):
            shape = (chain_a.dim(deg_a), chain_b.dim(deg_b))
            perms = [
                comb.Permutation(
                    [
                        np.ravel_multi_index(
                            (list(perms_a[deg_a])[aa], list(perms_b[deg_b])[bb]), shape
                        )
                        for aa, bb in np.ndindex(shape)
                    ]
                )
                for perms_a, perms_b in zip(symmetries_a, symmetries_b)
            ]
            # without symmetries, use the identity on one extra point, whose orbit is dropped
            size = math.prod(shape)
            group = comb.PermutationGroup(perms or [comb.Permutation(size)])
            orbits = sorted((orbit for orbit in group.orbits() if min(orbit) < size), key=min)
            block = np.zeros((len(orbits), size), dtype=int)
            for index, orbit in enumerate(orbits):
                block[index, list(orbit)] = 1
            blocks.append(block)
        projections.append(chain_a.field(scipy.linalg.block_diag(*blocks)))

    for degree in range(1, tensor.num_links + 1):
        assert np.array_equal(
            balanced.op(degree) @ projections[degree], projections[degree - 1] @ tensor.op(degree)
        )


def test_chain_complex_over_matrix_ring() -> None:
    """A chain complex over a matrix ring matches that over an isomorphic group algebra."""
    group_ring = abstract.GroupRing(abstract.CyclicGroup(3), field=3)
    ring = abstract.MatrixRing(group_ring.generators[0].lift())

    def get_chain(base_ring: abstract.Ring) -> objects.ChainComplex:
        matrix = abstract.RingArray.build([[1, base_ring.generators[0]]], base_ring)
        return objects.ChainComplex.tensor_product(matrix, matrix)

    chain, group_chain = get_chain(ring), get_chain(group_ring)
    assert chain.ring == ring
    for op, group_op in zip(chain.ops, group_chain.ops):
        assert isinstance(op, abstract.RingArray) and isinstance(group_op, abstract.RingArray)
        assert np.array_equal(op.lift(), group_op.lift())
