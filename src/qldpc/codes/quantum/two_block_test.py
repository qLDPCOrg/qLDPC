# SPDX-License-Identifier: Apache-2.0

"""Unit tests for two_block.py."""

from __future__ import annotations

import itertools

import numpy as np
import numpy.typing as npt
import pytest
import sympy
from sympy.abc import x, y

from qldpc import abstract, codes
from qldpc.codes.distance._search import automorphism_families
from qldpc.objects import Node

from ..conftest import (
    assert_valid_subgraphs,
    assert_valid_translations,
    tanner_graphs_are_isomorphic,
)
from .two_block import _get_abelian_block_translations, _get_block_translations


def test_two_block_code_error() -> None:
    """Two-block codes need commuting matrices, unless validation is skipped."""
    matrix_a = [[1, 0], [0, 2]]
    matrix_b = [[0, 1], [1, 0]]
    assert not codes.TBCode(matrix_a, matrix_a, field=3).is_subsystem_code
    with pytest.raises(ValueError, match="do not commute"):
        codes.TBCode(matrix_a, matrix_b, field=3)

    # skipping validation builds the code, whose checks do not commute
    assert codes.TBCode(matrix_a, matrix_b, field=3, skip_validation=True).is_subsystem_code


def test_gala_code_construction() -> None:
    """Construct the block-circulant parent and active matrices of a GALA code."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3))
    one = ring.one
    xx = ring.generators[0]

    code = codes.GALACode(
        generators_f=[one, xx],
        generators_g=[xx**2, one],
        num_active_rows=1,
    )

    expected_f = abstract.RingArray([[one, xx], [xx, one]])
    expected_g = abstract.RingArray([[xx**2, one], [one, xx**2]])
    expected_parent_x = abstract.RingArray([[one, xx, xx**2, one], [xx, one, one, xx**2]])
    expected_parent_z = abstract.RingArray([[xx, one, one, xx**2], [one, xx, xx**2, one]])

    assert np.array_equal(code.matrix_f, expected_f)
    assert np.array_equal(code.matrix_g, expected_g)
    assert np.array_equal(code.parent_matrix_x, expected_parent_x)
    assert np.array_equal(code.parent_matrix_z, expected_parent_z)
    assert code.matrix_x.shape == (3, 12)
    assert code.matrix_z.shape == (3, 12)
    assert not np.any(code.matrix_x @ code.matrix_z.T)
    assert not code.is_subsystem_code
    assert code.num_blocks == 4
    assert code.num_active_rows == 1
    assert code.generators_f[0] is not one


def test_gala_code_active_orthogonality() -> None:
    """Validate aggregate rather than pairwise active orthogonality."""
    ring = abstract.GroupRing(abstract.SymmetricGroup(3))
    aa, bb = ring.generators
    assert aa * bb != bb * aa

    code = codes.GALACode(
        generators_f=[aa, aa],
        generators_g=[bb, bb],
        num_active_rows=1,
    )
    assert not np.any(code.matrix_x @ code.matrix_z.T)

    generators_f = [aa, ring.one]
    generators_g = [bb, ring.one]
    with pytest.raises(ValueError, match=r"active parity checks.*do not commute"):
        codes.GALACode(generators_f, generators_g, num_active_rows=1)

    code = codes.GALACode(generators_f, generators_g, num_active_rows=1, skip_validation=True)
    assert np.any(code.matrix_x @ code.matrix_z.T)


def test_gala_product_group_integration() -> None:
    """Construct GALA codes from direct- and wreath-product group elements."""
    top = abstract.SymmetricGroup(3).with_natural_lift()
    bottom = abstract.CyclicGroup(2)
    top_f, top_g = top.generators
    bottom_identity, bottom_shift = bottom.generate()

    direct_product = abstract.Group.tensor_product(top, bottom)
    direct_ring = abstract.GroupRing(direct_product)
    direct_generator_f = abstract.RingMember(direct_ring, top_f @ bottom_shift)
    direct_generator_g = abstract.RingMember(direct_ring, top_g @ bottom_identity)
    direct_code = codes.GALACode(
        [direct_generator_f] * 2, [direct_generator_g] * 2, num_active_rows=1
    )

    wreath_product = abstract.WreathProductGroup(top, bottom)
    wreath_element_f = wreath_product.element(top_f, [bottom_shift, bottom_identity, bottom_shift])
    wreath_element_g = wreath_product.element(
        top_g, [bottom_identity, bottom_shift, bottom_identity]
    )
    wreath_ring = abstract.GroupRing(wreath_product)
    wreath_generator_f = abstract.RingMember(wreath_ring, wreath_element_f)
    wreath_generator_g = abstract.RingMember(wreath_ring, wreath_element_g)
    wreath_code = codes.GALACode(
        [wreath_generator_f] * 2, [wreath_generator_g] * 2, num_active_rows=1
    )

    assert direct_code.group is direct_product
    assert direct_code.matrix_x.shape == direct_code.matrix_z.shape == (6, 24)
    assert direct_code.num_qubits == 24
    assert direct_code.get_weight() == 4
    assert not np.any(direct_code.matrix_x @ direct_code.matrix_z.T)

    assert wreath_code.group is wreath_product
    assert wreath_code.matrix_x.shape == wreath_code.matrix_z.shape == (6, 24)
    assert wreath_code.num_qubits == 24
    assert wreath_code.get_weight() == 4
    assert not np.any(wreath_code.matrix_x @ wreath_code.matrix_z.T)


def test_gala_code_errors() -> None:
    """Reject invalid GALA generator data."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3))
    one = ring.one

    with pytest.raises(ValueError, match="nonempty"):
        codes.GALACode([], [], num_active_rows=1)
    with pytest.raises(ValueError, match="equal lengths"):
        codes.GALACode([one], [one, one], num_active_rows=1)
    with pytest.raises(ValueError, match="RingMember"):
        codes.GALACode([one], [1], num_active_rows=1)  # type: ignore[list-item]

    other_ring = abstract.GroupRing(abstract.CyclicGroup(2))
    with pytest.raises(ValueError, match="same group algebra"):
        codes.GALACode([one], [other_ring.one], num_active_rows=1)

    qutrit_ring = abstract.GroupRing(abstract.CyclicGroup(3), field=3)
    with pytest.raises(ValueError, match=r"only over GF\(2\)"):
        codes.GALACode([qutrit_ring.one], [qutrit_ring.one], num_active_rows=1)

    with pytest.raises(TypeError, match="must be an integer"):
        codes.GALACode([one], [one], num_active_rows=1.5)  # type: ignore[arg-type]
    for num_active_rows in [0, 2]:
        with pytest.raises(ValueError, match="must lie between"):
            codes.GALACode([one], [one], num_active_rows)


def test_gala_code_from_paper() -> None:
    """Reproduce the construction-level parameters of the [[132, 30, 12]] GALA code.

    This code is from arXiv:2608.07431.
    """
    ring = abstract.GroupRing(abstract.CyclicGroup(11))
    xx = ring.generators[0]
    code = codes.GALACode(
        generators_f=[xx**power for power in [2, 4, 3, 6, 3, 9]],
        generators_g=[xx**power for power in [9, 2, 8, 5, 8, 7]],
        num_active_rows=5,
    )

    assert code.matrix_x.shape == (55, 132)
    assert code.matrix_z.shape == (55, 132)
    assert code.num_qubits == 132
    assert code.dimension == 30
    assert code.get_weight() == 12
    assert not np.any(code.matrix_x @ code.matrix_z.T)

    code_string = str(code)
    assert "132 qubits" in code_string
    assert "12 blocks" in code_string
    assert "5 active rows" in code_string
    assert code.group.name in code_string


def test_polynomial_gala_code() -> None:
    """Construct a GALA code with a polynomial group-ring generator."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3))
    one = ring.one
    xx = ring.generators[0]
    polynomial = one + xx
    code = codes.GALACode(
        generators_f=[polynomial, xx**2],
        generators_g=[one, xx],
        num_active_rows=1,
    )

    block_size = code.group.lift_dim
    assert np.count_nonzero(code.generators_f[0].to_vector()) == 2
    assert np.array_equal(code.matrix_x[:block_size, :block_size], polynomial.lift())
    assert code.get_weight() == 5


def test_bivariate_bicycle_codes() -> None:
    """Bivariate bicycle codes from arXiv:2308.07915 and arXiv:2311.16980."""
    from sympy.abc import x, y, z

    orders: tuple[int, int] | dict[sympy.Symbol, int]

    # last code in Table II of arXiv:2311.16980
    orders = {x: 12, y: 4}
    poly_a = 1 + y + x * y + x**9
    poly_b = 1 + x**2 + x**7 + x**9 * y**2
    code = codes.BBCode(orders, poly_a, poly_b)
    assert code.num_qudits == 96
    assert code.dimension == 10
    assert code.get_weight() == 8

    # print the generating data of the code in human-readable form
    code_string = str(code)
    assert str(orders) in code_string
    assert str(poly_a.as_expr()) in code_string
    assert str(poly_b.as_expr()) in code_string

    # [[144, 12, 12]] code in Table 3 and Figure 2 of arXiv:2308.07915
    orders = (12, 6)
    poly_a = x**3 + y + y**2
    poly_b = y**3 + x + x**2
    code = codes.BBCode(orders, poly_a, poly_b)
    assert code.num_qudits == 144
    assert code.dimension == 12
    assert code.get_weight() == 6

    # toric layouts of a qutrit BBCode
    code = codes.BBCode(orders, poly_a, poly_b, field=3)
    assert "GF(3)" in str(code)
    assert code.dimension == 8
    for orders, poly_a, poly_b in code.get_equivalent_toric_layout_code_data():
        # assert that the polynomials look like 1 + x + ... and 1 + y + ...
        exponents_a = [abstract.get_coefficient_and_exponents(term)[1] for term in poly_a.args]
        exponents_b = [abstract.get_coefficient_and_exponents(term)[1] for term in poly_b.args]
        assert [] in exponents_a and [(x, 1)] in exponents_a
        assert [] in exponents_b and [(y, 1)] in exponents_b

        # assert that the code has equivalent parameters
        equiv_code = codes.BBCode(orders, poly_a, poly_b, field=3)
        assert equiv_code.num_qudits == code.num_qudits
        assert equiv_code.dimension == code.dimension

    # check a code with no toric layouts
    orders = (6, 6)
    poly_a = 1 + y + y**2
    poly_b = y**3 + x**2 + x**4
    code = codes.BBCode(orders, poly_a, poly_b)
    assert not code.get_equivalent_toric_layout_code_data()

    # retrieve node labels
    assert code.get_node_label(Node(0, is_data=True)) == ("L", 0, 0)
    assert code.get_node_label(Node(len(code) // 2, is_data=True)) == ("R", 0, 0)
    assert code.get_node_label(Node(0, is_data=False)) == ("X", 0, 0)
    assert code.get_node_label(Node(len(code) // 2, is_data=False)) == ("Z", 0, 0)

    # codes with more than 2 symbols are unsupported
    with pytest.raises(ValueError, match="should have exactly two"):
        codes.BBCode({}, x, y + z)

    # fail to invert a system of equations
    orders = (3, 3)
    poly_a = 1 + x + x * y
    poly_b = 1 + y + x * y
    code = codes.BBCode(orders, poly_a, poly_b)
    with pytest.raises(ValueError, match="Uninvertible system of equations"):
        basis = (1, 0), (0, 0)
        code._get_basis_coefficients(basis, 0, 1)


@pytest.mark.parametrize("orders, field", [((3, 3), 2), ((6, 3), 3)])
def test_bivariate_bicycle_toric_layout_equivalence(orders: tuple[int, int], field: int) -> None:
    """Every reported toric layout is a relabeling of the original Pauli-labelled Tanner graph."""
    original = codes.BBCode(orders, 1 + x + x * y, 1 + y + x * y, field)
    layouts = original.get_equivalent_toric_layout_code_data()
    assert layouts

    for new_orders, poly_a, poly_b in layouts:
        assert tanner_graphs_are_isomorphic(
            codes.BBCode(new_orders, poly_a, poly_b, field), original
        )


def test_bivariate_bicycle_neighbors() -> None:
    """In a toric layout of a code, check qubits address their nearest neighbors."""
    from sympy.abc import x, y

    # [[72, 12, 6]] code in Table 3 and Figure 2 of arXiv:2308.07915
    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    for orders, poly_a, poly_b in code.get_equivalent_toric_layout_code_data():
        code = codes.BBCode(orders, poly_a, poly_b)
        break

    # identify the dimensions of the qubit array
    array_shape = tuple(2 * order for order in code.orders)

    # iterate over check qubit nodes of the Tanner graph
    for node in code.graph.nodes:
        if node.is_data:
            # this is a data qubit
            continue

        # get all L1 distances to the data qubits addressed by this check qubit when all qubits are
        # laid out canonically on a torus
        neighbor_dists = []
        node_pos = code.get_qubit_pos(node)
        for neighbor in code.graph.neighbors(node):
            neighbor_pos = code.get_qubit_pos(neighbor)
            dist = get_dist_l1(node_pos, neighbor_pos, array_shape)
            neighbor_dists.append(dist)

        # assert that this check qubit has 6 neighbors, and that 4 of them are nearby
        assert len(neighbor_dists) == 6
        assert neighbor_dists.count(1) == 4

        # get all L1 distances to the check qubit's neighbors with the "folded" layout of this code
        neighbor_dists = []
        node_pos = code.get_qubit_pos(node, folded_layout=True)
        for neighbor in code.graph.neighbors(node):
            neighbor_pos = code.get_qubit_pos(neighbor, folded_layout=True)
            dist = get_dist_l1(node_pos, neighbor_pos)
            neighbor_dists.append(dist)

        # assert that at least 4 of the neighbors are still pretty close by
        assert neighbor_dists.count(1) + neighbor_dists.count(2) >= 4


def get_dist_l1(
    pos_a: tuple[int, ...], pos_b: tuple[int, ...], torus_shape: tuple[int, ...] | None = None
) -> int:
    """Get the L1 distance between two points on a lattice.

    If provided a torus_shape, the lattice has periodic boundary conditions.
    """
    diffs = [abs(aa - bb) for aa, bb in zip(pos_a, pos_b)]
    if torus_shape is None:
        return sum(diffs)
    return sum(min(diff, length - diff) for diff, length in zip(diffs, torus_shape))


def test_multivariate_bicycle_codes() -> None:
    """Multivariate versions of the bicycle codes in arXiv:2308.07915 and arXiv:2311.16980."""

    # not enough orders provided
    with pytest.raises(ValueError, match=r"Provided .* symbols, but only .* orders"):
        codes.MBCode([], x, y)

    # add placeholder symbols if necessary
    code = codes.MBCode([2, 1, 3], x, x * y)
    assert len(code.symbols) == 3

    assert_valid_subgraphs(code)
    with pytest.raises(ValueError, match="does not use an edge coloration strategy"):
        code.get_syndrome_subgraphs(strategy="smallest_last")

    # a symbol whose cyclic group is trivial acts as the identity, so all of its powers agree
    assert np.array_equal(code.matrix, codes.MBCode([2, 1, 3], x, x * y**2).matrix)

    # distinct monomials can name the same group element, and are simplified into a single term
    assert_valid_subgraphs(codes.MBCode([3], 1 + x**3 + x**6, 1 + x))

    # the coefficients of such monomials are summed in the base field, each denoting a field element
    # rather than a multiplicity: over GF(4) the elements 1, 1 and -2 sum to 2, not to the 0 that
    # the same integers give
    assert codes.MBCode([3], 1 + x**3 - 2 * x**6, 1 + x, field=4).poly_a.as_expr() == 2

    # more than one placeholder symbol is needed when the orders outnumber the symbols by 2 or more
    for orders, poly_a, poly_b in [([3, 4, 5], 1 + x, 1 + x**2), ([3, 4, 5, 6], 1 + x, 1 + y)]:
        code = codes.MBCode(orders, poly_a, poly_b)
        assert len(code.symbols) == len(orders)
        assert len(set(code.symbols)) == len(orders)  # every placeholder is distinct
        assert len(code) == 2 * np.prod(orders)

    # placeholder names do not depend on the iteration order of a set of symbols
    assert [str(symbol) for symbol in codes.MBCode([3, 4, 5, 6], 1 + x, 1 + y).symbols] == [
        "x",
        "y",
        "~xy_2",
        "~xy_3",
    ]


def test_multivariate_bicycle_translations_preserve_checks() -> None:
    """Translations of a multivariate bicycle code map its checks of each type onto themselves.

    One cyclic factor of this code is trivial, so its group has fewer generators than symbols.
    """
    code = codes.MBCode([2, 1, 3], x + 1, x * y + 1)
    assert_valid_translations(code, code._get_qudit_automorphisms(), group_order=6)


def test_gala_translations_preserve_checks() -> None:
    """Translations of a GALA code over an abelian group map its checks onto themselves."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3))
    xx = ring.generators[0]
    code = codes.GALACode(
        generators_f=[ring.one, xx], generators_g=[xx**2, ring.one], num_active_rows=1
    )
    translations = code._get_qudit_automorphisms()
    assert translations is not None and len(translations) == 1
    assert_valid_translations(code, translations, group_order=3)


def test_nonabelian_groups_give_no_block_translations() -> None:
    """Translations by the generators of a nonabelian group need not preserve a code."""
    assert _get_abelian_block_translations(abstract.DihedralGroup(3), 2) is None


@pytest.mark.parametrize(
    "lifts",
    [
        # projections, which move two coordinates to one, or one coordinate to two
        [np.array([[1, 1], [0, 0]])] * 3,
        [np.array([[1, 0], [1, 0]])] * 3,
        # a projection whose rows and columns each sum to one, but which has a negative entry
        [np.array([[1, 0, 0], [1, 0, 0], [-1, 1, 1]])] * 3,
    ],
)
def test_abelian_block_translations_need_permutation_lifts(
    lifts: list[npt.NDArray[np.int_]],
) -> None:
    """Lifts of an abelian group that are not permutation matrices give no block translations."""
    table = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    for aa, bb in itertools.product(range(3), repeat=2):
        assert np.array_equal(lifts[table[aa][bb]], lifts[aa] @ lifts[bb])
    group = abstract.Group.from_table(table, integer_lift=lambda member: lifts[member])
    assert group.is_abelian
    assert _get_abelian_block_translations(group, 2) is None


def test_block_translations() -> None:
    """Block translations move every block of qudits alike, and act regularly on each block.

    Translations by the generators of an abelian group commute, and a commuting group of
    permutations that takes any qudit of a block to any other acts on it regularly.
    """
    group = abstract.AbelianGroup(2, 3)
    num_blocks, block_size = 3, group.order
    translations = _get_block_translations(group, num_blocks)
    assert len(translations) == len(group.generators)

    blocks = np.arange(num_blocks * block_size).reshape(num_blocks, block_size)
    for translation in translations:
        assert np.array_equal(translation[blocks] - blocks[:, :1], [translation[:block_size]] * 3)
    for first, second in itertools.combinations(translations, 2):
        assert np.array_equal(first[second], second[first])
    orbits = automorphism_families.get_orbits(translations, num_blocks * block_size)
    assert [len(set(orbits[block])) for block in blocks] == [1] * num_blocks
    assert len(set(orbits)) == num_blocks


@pytest.mark.parametrize("num_terms", [1, 3, 4])
def test_multivariate_bicycle_syndrome_subgraphs_by_term_count(num_terms: int) -> None:
    """Syndrome subgraphs partition every Tanner edge beyond the two-term documented example."""
    poly_a = sum(x**power for power in range(1, num_terms + 1))
    poly_b = sum(x**power for power in range(6, 6 + num_terms))
    code = codes.MBCode([11], poly_a, poly_b)

    assert len(code.poly_a.terms()) == num_terms
    assert len(code.poly_b.terms()) == num_terms
    assert len(code.get_syndrome_subgraphs()) == 4 * num_terms
    assert_valid_subgraphs(code)


def test_gala_requires_group_algebra() -> None:
    """GALA codes are built from members of a group algebra."""
    ring = abstract.MatrixRing(np.array([[0, 1], [1, 0]]))
    generators = (ring.one + ring.generators[0],)
    with pytest.raises(ValueError, match="same group algebra"):
        codes.GALACode(generators, generators, num_active_rows=1)
