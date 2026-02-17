"""Unit tests for quantum.py

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

import io
import unittest.mock

import networkx as nx
import numpy as np
import pytest
import sympy
from sympy.abc import x, y

from qldpc import abstract, codes
from qldpc.math import block_matrix
from qldpc.objects import ChainComplex, Node, Pauli

from .common_test import assert_valid_subgraphs


def test_small_codes() -> None:
    """Small named codes."""
    assert codes.FiveQubitCode().get_code_params() == (5, 1, 3)
    assert codes.SteaneCode().get_code_params() == (7, 1, 3)

    for size in [4, 6, 8]:
        code_params = (size, size - 2, 2)
        assert codes.IcebergCode(size).get_code_params() == code_params
    with pytest.raises(ValueError, match="even block lengths"):
        codes.IcebergCode(3)

    assert codes.CSSCode.equiv(codes.C4Code(), codes.IcebergCode(4))
    assert codes.C6Code().get_code_params() == (6, 2, 2)


def test_hamming_and_tetrahedral_codes() -> None:
    """Quantum Hamming and tetrahedral codes."""
    quantum_hamming_code = codes.QuantumHammingCode(4)
    assert quantum_hamming_code.get_code_params() == (15, 7, 3)

    tetrahedral_code = codes.TetrahedralCode(algebraic=False)
    assert tetrahedral_code.get_code_params() == (15, 1, 3)
    assert tetrahedral_code.is_equiv_to(codes.TetrahedralCode(algebraic=True))

    """
    The tetrahedral code (TC) can be constructed by concatenating the quantum Hamming code (QHC)
    with a classical code on the logical X operators of the QHC, as we show below.  To this end, we
    first decompose the logical X operator of the TC into a product of logical X operators of the
    QHC, which can be found by checking (anti-)commutation with the logical Zs of the QHC.  We then
    concatenate the QHC with a classical code that has only this combination of QHC logical Xs as a
    nontrivial code word.
    """
    decomposition = (
        tetrahedral_code.get_logical_ops(Pauli.X) @ quantum_hamming_code.get_logical_ops(Pauli.Z).T
    )
    code = codes.CSSCode.concatenate(
        quantum_hamming_code,
        codes.CSSCode.classical(codes.ClassicalCode.from_generator(decomposition), Pauli.Z),
        range(quantum_hamming_code.dimension),
    )
    assert code.is_equiv_to(tetrahedral_code)


def test_two_block_code_error() -> None:
    """Raise error when trying to construct incompatible two-block codes."""
    matrix_a = [[1, 0], [0, 2]]
    matrix_b = [[0, 1], [1, 0]]
    with pytest.raises(ValueError, match="do not commute"):
        codes.TBCode(matrix_a, matrix_b, field=3)


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
    assert str(poly_a.as_expr()) in code_string

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
        exponents_a = [code.get_coefficient_and_exponents(term)[1] for term in poly_a.args]
        exponents_b = [code.get_coefficient_and_exponents(term)[1] for term in poly_b.args]
        assert {} in exponents_a and {x: 1} in exponents_a
        assert {} in exponents_b and {y: 1} in exponents_b

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
        code.modular_inverse(basis, 0, 1)


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


def test_quasi_cyclic_codes() -> None:
    """Multivariave versions of the bicycle codes in arXiv:2308.07915 and arXiv:2311.16980."""

    # not enough orders provided
    with pytest.raises(ValueError, match="Provided .* symbols, but only .* orders"):
        codes.QCCode([], x, y)

    # invalid coefficients
    with pytest.raises(ValueError, match="Coefficient .* is invalid"):
        codes.QCCode([2], 2 * x, x)

    # add placeholder symbols if necessary
    code = codes.QCCode([1, 2, 3], x, x * y)
    assert len(code.symbols) == 3

    assert_valid_subgraphs(code)


@pytest.mark.parametrize("field", [2, 3])
def test_hypergraph_product(
    pytestconfig: pytest.Config,
    field: int,
    bits_checks_a: tuple[int, int] = (5, 3),
    bits_checks_b: tuple[int, int] = (3, 2),
) -> None:
    """Equivalency of matrix-based, graph-based, and chain-based hypergraph products."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code_a = codes.ClassicalCode.random(*bits_checks_a, field=field, seed=np.random.randint(2**31))
    code_b = codes.ClassicalCode.random(*bits_checks_b, field=field, seed=np.random.randint(2**31))

    code = codes.HGPCode(code_a, code_b, set_logicals=True)
    graph = codes.HGPCode.get_graph_product(code_a.graph, code_b.graph)
    chain = ChainComplex.tensor_product(code_a.matrix, code_b.matrix.T)
    matrix_x, matrix_z = chain.op(1), chain.op(2).T

    assert nx.utils.graphs_equal(code.graph, graph)
    assert np.array_equal(code.matrix, codes.QuditCode.graph_to_matrix(graph))
    assert np.array_equal(code.matrix_x, matrix_x)
    assert np.array_equal(code.matrix_z, matrix_z)
    assert_valid_subgraphs(code)

    # verify that the canonical logicals are valid
    code.set_logical_ops(code.get_logical_ops(), validate=True)


@pytest.mark.parametrize("field", [2, 3])
def test_subsystem_hypergraph_product(
    pytestconfig: pytest.Config,
    field: int,
    bits_checks_a: tuple[int, int] = (5, 3),
    bits_checks_b: tuple[int, int] = (3, 2),
) -> None:
    """Validity of the subsystem hypergraph product code."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code_a = codes.ClassicalCode.random(*bits_checks_a, field=field, seed=np.random.randint(2**31))
    code_b = codes.ClassicalCode.random(*bits_checks_b, field=field, seed=np.random.randint(2**31))
    code = codes.SHPCode(code_a, code_b, set_logicals=True)

    # assert validity of the the "natural" stabilizers that are set at initialization time
    assert np.array_equal(
        codes.ClassicalCode(code.get_stabilizer_ops()).canonicalized.matrix,
        code.get_stabilizer_ops(recompute=True, canonicalized=True),
    )

    # verify that the canonical logicals are valid
    code.set_logical_ops(code.get_logical_ops(), validate=True)


def test_trivial_lift(
    pytestconfig: pytest.Config,
    bits_checks_a: tuple[int, int] = (4, 3),
    bits_checks_b: tuple[int, int] = (3, 2),
    field: int = 3,
) -> None:
    """The lifted product code with a trivial lift reduces to the HGP code."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    code_a = codes.ClassicalCode.random(*bits_checks_a, field=field, seed=np.random.randint(2**31))
    code_b = codes.ClassicalCode.random(*bits_checks_b, field=field, seed=np.random.randint(2**31))
    code_HGP = codes.HGPCode(code_a, code_b, field)

    matrix_a = abstract.TrivialGroup.to_ring_array(code_a.matrix)
    matrix_b = abstract.TrivialGroup.to_ring_array(code_b.matrix)
    code_LP = codes.LPCode(matrix_a, matrix_b)

    assert np.array_equal(code_HGP.matrix_x, code_LP.matrix_x)
    assert np.array_equal(code_HGP.matrix_z, code_LP.matrix_z)
    assert nx.utils.graphs_equal(code_HGP.graph, code_LP.graph)
    assert np.array_equal(code_HGP.sector_size, code_LP.sector_size)

    chain = ChainComplex.tensor_product(matrix_a, matrix_b.T)
    matrix_x, matrix_z = chain.op(1), chain.op(2).T
    assert isinstance(matrix_x, abstract.RingArray)
    assert isinstance(matrix_z, abstract.RingArray)
    assert np.array_equal(matrix_x.lift(), code_HGP.matrix_x)
    assert np.array_equal(matrix_z.lift(), code_HGP.matrix_z)


def test_lift() -> None:
    """Verify lifting in Eqs. (8) and (10) of arXiv:2202.01702v3."""
    group = abstract.CyclicGroup(3)
    zero = abstract.RingMember(group)
    x0, x1, x2 = [abstract.RingMember(group, member) for member in group.generate()]
    base_matrix = [[x1 + x2, x0, zero], [zero, x0 + x1, x1]]

    ring_matrix = abstract.RingArray(base_matrix)
    lifted_matrix = [
        [0, 1, 1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 1, 0, 1, 0],
    ]
    assert np.array_equal(ring_matrix.lift(), lifted_matrix)

    # check that the lifted product code is indeed smaller than the HGP code!
    code_HP = codes.HGPCode(lifted_matrix)
    code_LP = codes.LPCode(ring_matrix)
    assert code_HP.num_qudits > code_LP.num_qudits
    assert code_HP.num_checks > code_LP.num_checks

    # check total number of qubits
    assert code_HP.sector_size.sum() == code_HP.num_qudits + code_HP.num_checks
    assert code_LP.sector_size.sum() == code_LP.num_qudits + code_LP.num_checks


def test_twisted_XZZX(width: int = 3) -> None:
    """Verify twisted XZZX code in Eqs.(29) and (32) of arXiv:2202.01702v3."""
    num_qudits = 2 * width**2
    code: codes.QuditCode

    # construct check matrix directly
    ring_code = codes.RingCode(width).matrix.T
    mat_1 = np.kron(ring_code, np.eye(width, dtype=int))
    mat_2 = codes.RingCode(num_qudits // 2).matrix.T
    matrix = block_matrix(
        [
            [mat_1, 0, 0, mat_2.T],
            [0, mat_1.T, -mat_2, 0],
        ]
    ).view(np.ndarray)

    # construct lifted product code
    group = abstract.CyclicGroup(num_qudits // 2)
    ring = abstract.GroupRing(group)
    shift = abstract.RingMember(ring, group.generators[0])
    element_a = ring.one - shift**width
    element_b = ring.one - shift
    code = codes.LPCode([[element_a]], [[element_b]])
    bias_tailoring_qubits = code.bias_tailoring_qubits
    assert np.array_equal(matrix, code.conjugated(bias_tailoring_qubits).matrix)

    # same construction with a chain complex
    matrix_a = abstract.RingArray([[element_a]])
    matrix_b = abstract.RingArray([[element_b]])
    chain = ChainComplex.tensor_product(matrix_a, matrix_b.T)
    matrix_x, matrix_z = chain.op(1), chain.op(2).T
    assert isinstance(matrix_x, abstract.RingArray)
    assert isinstance(matrix_z, abstract.RingArray)
    code = codes.CSSCode(matrix_x.lift(), matrix_z.lift()).conjugated(bias_tailoring_qubits)
    assert np.array_equal(matrix, code.matrix)


def test_lifted_product_codes() -> None:
    """Lifted product codes in Eq. (5) of arXiv:2308.08648."""
    for lift_dim, exponents in [
        (16, [[0, 0, 0, 0, 0], [0, 2, 4, 7, 11], [0, 3, 10, 14, 15]]),
        (21, [[0, 0, 0, 0, 0], [0, 4, 5, 7, 17], [0, 14, 18, 12, 11]]),
    ]:
        group = abstract.CyclicGroup(lift_dim)
        xx = group.generators[0]
        matrix = abstract.RingArray.build(
            [[xx**power for power in row] for row in exponents], group
        )
        code = codes.LPCode(matrix)
        rate = code.dimension / code.num_qudits
        assert rate >= 2 / 17

        # the subsystem version of this code has a highe encoding rate
        subsystem_code = codes.SLPCode(matrix)
        subsystem_rate = subsystem_code.dimension / subsystem_code.num_qudits
        assert subsystem_rate > rate


def test_quantum_tanner(pytestconfig: pytest.Config) -> None:
    """Quantum Tanner code."""
    np.random.seed(pytestconfig.getoption("randomly_seed"))

    # random quantum Tanner code
    group = abstract.CyclicGroup(12)
    subcode = codes.RepetitionCode(4, field=3)
    code = codes.QTCode.random(group, subcode, seed=np.random.randint(2**31))

    # assert that subgraphs have the right number of nodes, edges, and node degrees
    subgraph_x, subgraph_z = codes.QTCode.get_subgraphs(code.complex)
    group_size = code.complex.graph.number_of_nodes()
    num_faces = code.num_qudits
    for graph in [subgraph_x, subgraph_z]:
        assert graph.number_of_nodes() == num_faces + group_size / 2
        assert graph.number_of_edges() == num_faces * 2
        sources = [node for node in graph.nodes if graph.in_degree(node) == 0]
        assert {graph.out_degree(node) for node in sources} == {subcode.num_bits**2}

    # raise error if the generating data is underspecified
    subset_a = code.complex.subset_a
    subset_b = group.random_symmetric_subset(len(subset_a) - 1)
    subcode_a = codes.RepetitionCode(len(subset_a))
    with pytest.raises(ValueError, match="Underspecified generating data"):
        codes.QTCode(subset_a, subset_b, subcode_a)

    # raise error if seed codes are over different fields
    subcode_b = codes.RepetitionCode(2, field=3)
    with pytest.raises(ValueError, match="different fields"):
        codes.QTCode([], [], subcode_a, subcode_b)

    # saving and loading a QTCode
    contents = io.StringIO()
    with unittest.mock.patch("builtins.open", return_value=contents):
        # save to a file
        with (
            unittest.mock.patch("os.makedirs", return_value=None),
            unittest.mock.patch.object(contents, "close", lambda: None),
        ):
            code.save("path.txt", "header")

        # fail to load from a file
        with (
            unittest.mock.patch("os.path.isfile", return_value=False),
            pytest.raises(ValueError, match="Path does not exist"),
        ):
            codes.QTCode.load("path.txt")

        # load from a file
        with (
            unittest.mock.patch("os.path.isfile", return_value=True),
            unittest.mock.patch.object(contents, "read", lambda: contents.getvalue()),
        ):
            code_copy = codes.QTCode.load("path.txt")
        assert code_copy == code


def test_toric_tanner_code(size: int = 4) -> None:
    """Rotated toric code as a quantum Tanner code."""
    group = abstract.Group.product(abstract.CyclicGroup(size), repeat=2)
    shift_x, shift_y = group.generators
    subset_a = [shift_x, ~shift_x]
    subset_b = [shift_y, ~shift_y]
    subcode_a = codes.RepetitionCode(2)
    code = codes.QTCode(subset_a, subset_b, subcode_a, bipartite=False)
    assert code.get_code_params() == (size**2, 2, size)
    assert code.get_weight() == 4


def test_surface_codes(rows: int = 3, cols: int = 2) -> None:
    """Ordinary and rotated surface codes."""

    # "ordinary"/original surface code
    code = codes.SurfaceCode(rows, cols, rotated=False)
    assert code.dimension == 1
    assert code.num_qudits == rows * cols + (rows - 1) * (cols - 1)
    with (
        unittest.mock.patch("qldpc.codes.CSSCode.get_distance_if_known", return_value=None),
        unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False),
    ):
        assert cols <= code.get_distance(Pauli.X, bound=True) <= len(code)
        assert rows <= code.get_distance(Pauli.Z, bound=True) <= len(code)
    assert_valid_subgraphs(code)

    # un-rotated SurfaceCode = HGPCode
    rep_codes = (codes.RepetitionCode(rows), codes.RepetitionCode(cols))
    assert code.is_equiv_to(codes.HGPCode(*rep_codes))

    # rotated surface code
    code = codes.SurfaceCode(rows, cols, rotated=True, field=3)
    assert code.dimension == 1
    assert code.num_qudits == rows * cols
    assert codes.CSSCode.get_distance(code, Pauli.X) == cols
    assert codes.CSSCode.get_distance(code, Pauli.Z) == rows
    assert_valid_subgraphs(code)

    # the bais-tailored rotated surface code is an XZZX code
    code = codes.SurfaceCode(max(rows, cols), rotated=True)
    for row in code.conjugated(code.bias_tailoring_qubits).matrix:
        row_x, row_z = row[: code.num_qudits], row[-code.num_qudits :]
        assert np.count_nonzero(row_x) == np.count_nonzero(row_z)


def test_toric_codes() -> None:
    """Ordinary and rotated toric codes."""

    # "ordinary"/original toric code
    distance = 3
    code = codes.ToricCode(distance, rotated=False)
    assert code.dimension == 2
    assert code.num_qudits == 2 * distance**2
    assert_valid_subgraphs(code)

    # check minimal logical operator weights
    code.reduce_logical_ops(with_ILP=True)
    assert (
        {distance}
        == {sum(op) for op in code.get_logical_ops(Pauli.X).view(np.ndarray)}
        == {sum(op) for op in code.get_logical_ops(Pauli.Z).view(np.ndarray)}
    )

    # rotated toric code
    distance = 4
    code = codes.ToricCode(distance, rotated=True, field=3)
    assert code.dimension == 2
    assert code.num_qudits == distance**2
    assert codes.CSSCode.get_distance(code) == distance
    assert_valid_subgraphs(code)

    # rotated toric code must have even side lengths
    with pytest.raises(ValueError, match="even side lengths"):
        codes.ToricCode(3, rotated=True)

    # the rotated 2x2 Toric code is special -- we remove redundant checks
    code = codes.ToricCode(2, rotated=True)
    assert len(code.matrix_x) == len(code.matrix_z) == 1

    # the bais-tailored rotated toric code is an XZZX code
    rows, cols = 6, 4
    code = codes.ToricCode(rows, cols, rotated=True)
    for row in code.conjugated(code.bias_tailoring_qubits).matrix:
        row_x, row_z = row[: code.num_qudits], row[-code.num_qudits :]
        assert np.count_nonzero(row_x) == np.count_nonzero(row_z)


def test_generalized_surface_codes(size: int = 3) -> None:
    """Multi-dimensional surface and toric codes."""

    # recover ordinary surface code in 2D
    assert np.array_equal(
        codes.GeneralizedSurfaceCode(size, dim=2, periodic=False).matrix,
        codes.SurfaceCode(size, rotated=False).matrix,
    )

    # recover ordinary toric code in 2D
    assert np.array_equal(
        codes.GeneralizedSurfaceCode(size, dim=2, periodic=True).matrix,
        codes.ToricCode(size, rotated=False).matrix,
    )

    for dim in [3, 4]:
        # surface code
        code = codes.GeneralizedSurfaceCode(size, dim, periodic=False)
        assert code.dimension == 1
        assert code.num_qudits == size**dim + (dim - 1) * size ** (dim - 2) * (size - 1) ** 2

        # toric code
        code = codes.GeneralizedSurfaceCode(size, dim, periodic=True)
        assert code.dimension == dim
        assert code.num_qudits == dim * size**dim

    with pytest.raises(ValueError, match=">= 2"):
        codes.GeneralizedSurfaceCode(size, dim=1)


def test_many_hypercube_code() -> None:
    """Goto's many-hypercube code."""
    for level in range(1, 5):
        params = (6**level, 4**level, 2**level)
        assert codes.ManyHypercubeCode(level).get_code_params() == params


def test_bacon_shor_code() -> None:
    """Bacon-Shor code."""
    code = codes.BaconShorCode(2, 3)
    assert all(np.count_nonzero(row) == 2 for row in code.matrix)
    assert code.get_distance(Pauli.X) == 3
    assert code.get_distance(Pauli.Z) == 2


def test_shyps_code() -> None:
    """Sanity checks for the SHYPS code."""
    for dimension in [3, 4, 5]:
        code = codes.SHYPSCode(dimension)
        params = ((2**dimension - 1) ** 2, dimension**2, 2 ** (dimension - 1))
        assert code.get_code_params() == params
        assert np.all(np.count_nonzero(code.matrix.view(np.ndarray), axis=1) == 3)


def test_exact_distances() -> None:
    """Compute exact distances in special cases."""
    for code in [
        codes.HGPCode(codes.RingCode(2)),
        codes.SHPCode(codes.RepetitionCode(2)),
    ]:
        name = type(code).__name__
        dist = code.get_distance()
        dist_x = code.get_distance(Pauli.X)
        dist_z = code.get_distance(Pauli.Z)
        with (
            unittest.mock.patch("qldpc.codes.CSSCode.get_distance_if_known", return_value=None),
            unittest.mock.patch(
                f"qldpc.codes.{name}._get_distance_exact", return_value=NotImplemented
            ),
        ):
            assert dist == code.get_distance()
            assert dist_x == code.get_distance(Pauli.X)
            assert dist_z == code.get_distance(Pauli.Z)
