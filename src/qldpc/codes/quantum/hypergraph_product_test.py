# SPDX-License-Identifier: Apache-2.0

"""Unit tests for hypergraph_product.py."""

from __future__ import annotations

import unittest.mock

import networkx as nx
import numpy as np
import pytest
from sympy.abc import x, y

from qldpc import codes
from qldpc.codes.distance import DistanceMethod
from qldpc.math import resolve_field
from qldpc.objects import ChainComplex, Pauli, PauliXZ

from ..conftest import assert_cached_parameters_are_genuine, assert_valid_subgraphs


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
    assert np.array_equal(code.matrix, graph.to_matrix())
    assert np.array_equal(code.matrix_x, matrix_x)
    assert np.array_equal(code.matrix_z, matrix_z)
    assert_valid_subgraphs(code)

    # verify that the canonical logicals are valid
    code.set_logical_ops(code.get_logical_ops(), skip_validation=False)

    # the closed-form X and Z distances agree with those of a plain CSS code with the same checks
    if field == 2:  # the brute-force kernel behind the generic route is binary
        plain = codes.CSSCode(code.matrix_x, code.matrix_z)
        with unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False):
            assert code.get_distance(Pauli.X) == plain.get_distance(Pauli.X)
            assert code.get_distance(Pauli.Z) == plain.get_distance(Pauli.Z)

    # a random seed code addresses every bit.  Check a seed code that leaves one unaddressed, whose
    # product therefore has a data qudit that no check addresses.
    seed = codes.ClassicalCode([[1, 1, 0], [0, 0, 0]], field=field)
    graph = codes.HGPCode.get_graph_product(seed.graph, seed.graph)
    assert nx.utils.graphs_equal(codes.HGPCode(seed, seed).graph, graph)

    # without a second seed code, the product is of the first seed code with itself
    assert nx.utils.graphs_equal(codes.HGPCode(seed).graph, graph)


@pytest.mark.parametrize(
    "seed_a, seed_b",
    [
        # the same [4, 1, 4] code, presented with one parity check repeated: only its transpose code
        # gains code words, so the (1, 1) sector of the product carries no logical operator
        (codes.ClassicalCode(np.vstack([codes.RepetitionCode(4).matrix] * 2)[:5]), None),
        # a seed code of dimension zero, so that the (0, 0) sector carries no logical operator, in
        # each of the two positions
        (codes.ClassicalCode([[1, 0], [0, 1], [1, 1]]), codes.ClassicalCode([[1, 1], [1, 1]])),
        (codes.ClassicalCode([[1, 1], [1, 1]]), codes.ClassicalCode([[1, 0], [0, 1], [1, 1]])),
        # and a dependent check in only the second seed code, the mirror of the first case
        (
            codes.RepetitionCode(3),
            codes.ClassicalCode(np.vstack([codes.RepetitionCode(4).matrix] * 2)[:5]),
        ),
    ],
)
def test_hypergraph_product_distance_by_sector(
    seed_a: codes.ClassicalCode, seed_b: codes.ClassicalCode | None
) -> None:
    """A sector carrying no logical operator contributes no weight to a hypergraph product distance.

    Each seed code here either has a dependent parity check or has dimension zero, so the closed
    form has to decide, sector by sector, which candidate weight belongs to a logical operator.
    """
    seed_b = seed_b if seed_b is not None else codes.RepetitionCode(3)
    assert seed_a.rank < len(seed_a.matrix) or seed_b.rank < len(seed_b.matrix)

    code = codes.HGPCode(seed_a, seed_b)
    plain = codes.CSSCode(code.matrix_x, code.matrix_z)
    with unittest.mock.patch("qldpc.external.gap.is_installed", return_value=False):
        # the same distances that a code carrying no closed form of its own computes
        assert code.get_distance(Pauli.X) == plain.get_distance(Pauli.X)
        assert code.get_distance(Pauli.Z) == plain.get_distance(Pauli.Z)


@pytest.mark.parametrize("pauli", [None, Pauli.X, Pauli.Z])
@pytest.mark.parametrize(
    "method, seed_method",
    [("brute_force", "brute_force"), ("connected_support", "brouwer_zimmermann")],
)
def test_closed_forms_search_seed_codes_with_the_given_method(
    pauli: PauliXZ | None, method: DistanceMethod, seed_method: DistanceMethod
) -> None:
    """The closed-form distances of product codes search their seed codes with the given method.

    Classical codes do not accept the connected-support method, so the seed codes use the default.
    """
    for product in (codes.HGPCode, codes.SHPCode):
        # A generic copy of a ring code records no distance, and its dependent checks give its
        # transpose code words, so the closed form of HGPCode searches both.
        seed = codes.ClassicalCode(codes.RingCode(4).matrix)
        code = product(seed, seed)
        with unittest.mock.patch.object(
            codes.ClassicalCode,
            "get_distance",
            autospec=True,
            side_effect=codes.ClassicalCode.get_distance,
        ) as get_distance:
            distance = code.get_distance_exact(pauli, method=method)
        assert distance == product(seed, seed).get_distance_exact(pauli)
        assert get_distance.call_args_list
        assert all(call.kwargs["method"] == seed_method for call in get_distance.call_args_list)


def test_hypergraph_product_syndrome_subgraphs() -> None:
    """Horizontal syndrome subgraphs of an HGPCode merge X-type and Z-type parity checks."""
    # a seed code with more than two edge colors, so that merging color classes is detectable
    code = codes.HGPCode(codes.RepetitionCode(3), codes.HammingCode(3))
    subgraphs = code.get_syndrome_subgraphs()

    # each subgraph is a matching, so it is realizable as a single layer of gates
    assert all(subgraph.degree(node) == 1 for subgraph in subgraphs for node in subgraph.nodes)

    # the horizontal subgraphs address X-type and Z-type parity checks together, while the vertical
    # subgraphs that open and close the sequence keep the two check types apart
    addresses_both_check_types = [
        len({node in code.graph_x for node in subgraph.nodes if not node.is_data}) == 2
        for subgraph in subgraphs
    ]
    assert any(addresses_both_check_types)
    assert not addresses_both_check_types[0]
    assert not addresses_both_check_types[-1]

    # a seed code with no parity checks contributes no vertical edges at all
    assert_valid_subgraphs(codes.HGPCode(codes.RepetitionCode(1), codes.RepetitionCode(3)))

    # a check that addresses no bits of a seed code still addresses qudits of the product, so the
    # subgraphs have to cover its edges.  Horizontal and vertical edges are collected in separate
    # loops, so place such a check in each seed code in turn.
    assert_valid_subgraphs(
        codes.HGPCode(codes.ClassicalCode([[1, 1, 0], [0, 0, 0]]), codes.RepetitionCode(3))
    )
    assert_valid_subgraphs(
        codes.HGPCode(codes.RepetitionCode(3), codes.ClassicalCode([[1, 1, 0], [0, 0, 0]]))
    )


def test_cyclic_hypergraph_product_codes() -> None:
    """CHGPCode and CRCode."""

    # verify Remark 3 from arxiv:2511.09683v2
    code_a = codes.CHGPCode((4, 4), 1 - x, 1 - y, field=3)
    code_b = codes.ToricCode(4, rotated=False, field=3)
    assert np.array_equal(code_a.matrix_x, code_b.matrix_x)
    assert np.array_equal(code_a.matrix_z, code_b.matrix_z)

    # reproduce Table 3 from arxiv:2511.09683v2
    chgp_codes = {
        (15, 1 + x + x**4): ((450, 32, 8), (240, 8, 8)),
        (21, 1 + x + x**5): ((882, 50, 10), (420, 10, 10)),
        (28, 1 + x**2 + x**4 + x**10): ((1568, 200, 6), (336, 20, 6)),
        (21, 1 + x + x**3 + x**8): ((882, 98, 8), (336, 14, 8)),
        (30, 1 + x + x**2 + x**7): ((1800, 72, 14), (840, 12, 14)),
        (31, 1 + x + x**2 + x**6 + x**27): ((1922, 200, 10), (620, 20, 10)),
        (31, 1 + x + x**3 + x**9 + x**10): ((1922, 200, 12), (744, 20, 12)),
    }
    for (bits, poly), (c2_params, cr_params) in chgp_codes.items():
        assert codes.CHGPCode(bits, poly).get_code_params() == c2_params
        assert codes.CRCode(bits, poly).get_code_params() == cr_params

    with pytest.raises(ValueError, match="defined distance"):
        codes.CRCode(3, x)


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
    code.set_logical_ops(code.get_logical_ops(), skip_validation=False)


def test_bacon_shor_code() -> None:
    """Bacon-Shor code."""
    code = codes.BaconShorCode(2, 3)
    assert all(np.count_nonzero(row) == 2 for row in code.matrix)

    # a square Bacon-Shor code knows both of its distances without computing them
    code = codes.BaconShorCode(3)
    assert code.get_distance_if_known(Pauli.X) == 3
    assert code.get_distance_if_known(Pauli.Z) == 3


def test_shyps_code() -> None:
    """Sanity checks for the [[(2^r-1)^2, r^2, 2^(r-1)]] SHYPS codes of arXiv:2502.07150."""
    for dimension in [3, 4, 5]:
        code = codes.SHYPSCode(dimension)
        params = ((2**dimension - 1) ** 2, dimension**2, 2 ** (dimension - 1))
        assert code.get_code_params() == params
        assert np.all(np.count_nonzero(code.matrix.view(np.ndarray), axis=1) == 3)


def test_distance_balanced_code() -> None:
    """Distance-balanced codes."""
    # compare to Eq. (4) of arXiv:2505.13679v2 (with a sign that matters over non-binary fields),
    # up to the order of qudits and Z-type checks
    field = resolve_field(3)
    matrix_x = field([[1, 2, 0, 0], [0, 0, 1, 2]])
    matrix_z = field([[1, 1, 1, 1]])
    matrix = field([[1, 2, 0], [0, 1, 2]])
    balanced_code = codes.DistanceBalancedCode(
        codes.CSSCode(matrix_x, matrix_z), codes.ClassicalCode(matrix), Pauli.X
    )
    iden_n, iden_m = np.eye(3, dtype=int), np.eye(2, dtype=int)
    assert np.array_equal(
        balanced_code.matrix_x,
        np.hstack([np.kron(iden_m, matrix.T), np.kron(matrix_x, iden_n)]),
    )
    assert np.array_equal(
        balanced_code.matrix_z,
        np.block(
            [
                [np.kron(matrix_x.T, iden_m), -np.kron(field.Identity(4), matrix)],
                [field.Zeros((3, 4)), np.kron(matrix_z, iden_n)],
            ]
        ),
    )

    # distance balancing amplifies only the distance of the selected Pauli type
    code = codes.CSSCode(codes.RepetitionCode(3), np.zeros((0, 3), dtype=int))
    assert (code.get_distance_exact(Pauli.X), code.get_distance_exact(Pauli.Z)) == (1, 3)
    code_x = codes.DistanceBalancedCode(code, codes.RepetitionCode(2), "x")
    code_z = codes.DistanceBalancedCode(code, codes.RepetitionCode(2), "z")
    assert (code_x.get_distance_exact(Pauli.X), code_x.get_distance_exact(Pauli.Z)) == (2, 3)
    assert (code_z.get_distance_exact(Pauli.X), code_z.get_distance_exact(Pauli.Z)) == (1, 6)

    # redundant checks contribute (m_X - rank(H_X)) * (m_C - rank(H)) = 1 more logical qudit to
    # the k * k_C = 1 logical qudits of the distance-balanced code
    code = codes.CSSCode([[1, 1], [1, 1]], np.zeros((0, 2), dtype=int))
    assert codes.DistanceBalancedCode(code, [[1, 1], [1, 1]], "x").dimension == 2

    subsystem_code = codes.CSSCode([[1]], [[1]], is_subsystem_code=True)
    with pytest.raises(ValueError, match="not a subsystem code"):
        codes.DistanceBalancedCode(subsystem_code, codes.RepetitionCode(2), "x")


def test_cached_parameters_are_genuine() -> None:
    """Codes that cache their parameters agree with a computation that ignores the cache."""
    # the subsystem families additionally route their distance through a closed form, which is
    # itself expressed in terms of cached classical distances
    assert_cached_parameters_are_genuine(
        [
            (codes.BaconShorCode(2, 3), (6, 1, 2)),
            (codes.BaconShorCode(3, 5), (15, 1, 3)),
            (codes.SHYPSCode(2), (9, 4, 2)),
        ]
    )
