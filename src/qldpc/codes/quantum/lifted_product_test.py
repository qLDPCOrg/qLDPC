# SPDX-License-Identifier: Apache-2.0

"""Unit tests for lifted_product.py."""

from __future__ import annotations

from collections.abc import Callable

import galois
import networkx as nx
import numpy as np
import pytest
from sympy.abc import x

import qldpc
from qldpc import abstract, codes
from qldpc.objects import ChainComplex, Pauli

from ..conftest import assert_valid_translations
from .lifted_product import _get_paired_grid_line_ops, _get_split_logical_line_ops


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

    matrix_a = abstract.RingArray.build(code_a.matrix)
    matrix_b = abstract.RingArray.build(code_b.matrix)
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


def test_lift(ring_cyclic3_gf2: abstract.GroupRing) -> None:
    """Verify lifting in Equations (8) and (10) of arXiv:2202.01702v3."""
    ring = ring_cyclic3_gf2
    zero = abstract.RingMember(ring.group)
    x0, x1, x2 = [abstract.RingMember(ring, member) for member in ring.group.generate()]
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
    code_LP = codes.LPCode(ring_matrix, set_logicals=True)
    assert code_HP.num_qudits > code_LP.num_qudits
    assert code_HP.num_checks > code_LP.num_checks

    # check total number of qubits
    assert code_HP.sector_size.sum() == code_HP.num_qudits + code_HP.num_checks
    assert code_LP.sector_size.sum() == code_LP.num_qudits + code_LP.num_checks

    # line operators for the LPCode have lower weight than Gottesman-canonical logicals: sorted by
    # weight, no line operator is heavier than its counterpart, and their total weight is lower
    logical_line_ops = code_LP.get_logical_ops()
    logical_ops = code_LP.get_logical_ops(recompute=True)
    line_weights = np.count_nonzero(logical_line_ops.view(np.ndarray), axis=1)
    weights = np.count_nonzero(logical_ops.view(np.ndarray), axis=1)
    assert np.all(np.sort(line_weights) <= np.sort(weights))
    assert line_weights.sum() < weights.sum()

    # this seed also splits at trailing pivot blocks, but the split route declines it, since the
    # semisimple route takes precedence
    one = np.ones((1, 1), dtype=int)
    checks_a = abstract.kron(ring_matrix, one).lift()
    checks_b = abstract.kron(one, ring_matrix).lift()
    assert _get_paired_grid_line_ops(checks_a, checks_b, ring.lift_dim) is not None
    for with_second_block in [True, False]:
        assert (
            _get_split_logical_line_ops(
                ring_matrix, ring_matrix, with_second_block=with_second_block
            )
            is None
        )


def test_twisted_xzzx(width: int = 3) -> None:
    """Verify twisted XZZX code in Eqs.(29) and (32) of arXiv:2202.01702v3."""
    num_qudits = 2 * width**2
    code: codes.QuditCode

    # construct check matrix directly
    ring_code = codes.RingCode(width).matrix.T
    mat_1 = np.kron(ring_code, np.eye(width, dtype=int))
    mat_2 = codes.RingCode(num_qudits // 2).matrix.T
    matrix = qldpc.math.block_matrix(
        [
            [mat_1, 0, 0, mat_2.T],
            [0, mat_1.T, -mat_2, 0],
        ]
    )

    # construct lifted product code
    group = abstract.CyclicGroup(num_qudits // 2)
    ring = abstract.GroupRing(group)
    shift = ring.generators[0]
    element_a = ring.one - shift**width
    element_b = ring.one - shift
    code = codes.LPCode([[element_a]], [[element_b]], set_logicals=True)
    bias_tailoring_qubits = code.bias_tailoring_qubits
    assert np.array_equal(matrix, code.conjugated(bias_tailoring_qubits).matrix)

    # the canonical line operators for this code are disjoint!
    logical_ops = np.count_nonzero(code.get_logical_ops().view(np.ndarray), axis=0)
    assert np.all(logical_ops == 1)

    # ...which is not true of the Gottesman-canonical logicals
    logical_ops = np.count_nonzero(code.get_logical_ops(recompute=True).view(np.ndarray), axis=0)
    assert ~np.all(logical_ops == 1)

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
    """Lifted product codes in Equation (5) of arXiv:2308.08648."""
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

        # the subsystem version of this code has a higher encoding rate
        subsystem_code = codes.SLPCode(matrix)
        subsystem_rate = subsystem_code.dimension / subsystem_code.num_qudits
        assert subsystem_rate > rate


def test_one_by_one_lifted_product_is_multivariate_bicycle() -> None:
    """A one-by-one lifted product over an abelian group is a multivariate bicycle code."""
    ring = abstract.GroupRing(abstract.CyclicGroup(5), field=3)
    shift = ring.generators[0]
    element_a = ring.one + shift
    element_b = ring.one + shift**2

    lifted = codes.LPCode([[element_a.T]], [[element_b]])
    bicycle = codes.MBCode([5], 1 + x, 1 + x**2, field=3)

    assert np.array_equal(lifted.matrix_x, bicycle.matrix_x)
    assert np.array_equal(lifted.matrix_z, -bicycle.matrix_z)
    assert lifted.is_equiv_to(bicycle)


def test_subsystem_lifted_product_codes(ring_cyclic3_gf2: abstract.GroupRing) -> None:
    """Subsystem lifted product codes in arXiv:2404.18302v1."""

    # example 1 on page 6 of https://arxiv.org/pdf/2404.18302v1
    group = abstract.CyclicGroup(2)
    ring = abstract.GroupRing(group)
    xx = ring.generators[0]
    matrix = abstract.RingArray.build([[1, xx, xx], [xx, xx, 1]])  # Eq. 21
    code = codes.SLPCode(matrix)
    assert code.get_code_params() == (18, 4, 2)

    # example 2 on page 6 of https://arxiv.org/pdf/2404.18302v1
    ring = ring_cyclic3_gf2
    xx = ring.generators[0]
    matrix = abstract.RingArray.build([[xx**2 + xx + 1, xx + 1, xx]])  # Eq. 23
    code = codes.SLPCode(matrix, set_logicals=True)
    assert code.get_code_params() == (27, 12, 2)

    # line operators for this code have (on average) lower weight than Gottesman-canonical logicals
    logical_line_ops = code.get_logical_ops()
    logical_ops = code.get_logical_ops(recompute=True)
    line_weights = np.count_nonzero(logical_line_ops.view(np.ndarray), axis=1)
    weights = np.count_nonzero(logical_ops.view(np.ndarray), axis=1)
    assert np.sum(line_weights) < np.sum(weights)


def test_lifted_products_require_same_ring() -> None:
    """The two matrices of a lifted product must be over the same ring."""
    # these group algebras differ but lift their members to matrices of the same size, so only the
    # ring check rejects them.  Neither is semisimple, so canonical logical operators must check
    # rings before computing Howell normal forms.
    ring_a = abstract.GroupRing(abstract.CyclicGroup(4))
    ring_b = abstract.GroupRing(abstract.AbelianGroup(2, 2))
    assert ring_a != ring_b and ring_a.group.lift_dim == ring_b.group.lift_dim
    matrix_a = abstract.RingArray.build([[1, ring_a.generators[0]]], ring_a)
    matrix_b = abstract.RingArray.build([[1, ring_b.generators[0]]], ring_b)

    # an equal ring that is a separate object
    ring_c = abstract.GroupRing(abstract.CyclicGroup(4))
    assert ring_c == ring_a and ring_c is not ring_a
    matrix_c = abstract.RingArray.build([[1, ring_c.generators[0]]], ring_c)

    code_types: list[type[codes.LPCode | codes.SLPCode]] = [codes.LPCode, codes.SLPCode]
    for code_type in code_types:
        with pytest.raises(ValueError, match="different rings"):
            code_type(matrix_a, matrix_b)
        with pytest.raises(ValueError, match="different rings"):
            code_type.get_canonical_ring_logical_line_ops(matrix_a, matrix_b)
        with pytest.raises(ValueError, match="different rings"):
            code_type.get_canonical_logical_line_ops(matrix_a, matrix_b)
        assert np.array_equal(code_type(matrix_a, matrix_c).matrix, code_type(matrix_a).matrix)

    # a group algebra and a ring generated by the lifts of its members are different rings
    matrix_ring = abstract.MatrixRing(ring_a.generators[0].lift())
    matrix_d = abstract.RingArray.build([[1, matrix_ring.generators[0]]], matrix_ring)
    for code_type in code_types:
        for matrices in [(matrix_a, matrix_d), (matrix_d, matrix_a)]:
            with pytest.raises(ValueError, match="different rings"):
                code_type(*matrices)


def test_lifted_products_of_empty_matrices() -> None:
    """Lifted products accept matrices with no rows, which give codes with no checks."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3))
    empty = abstract.RingArray.build(np.zeros((0, 2), dtype=int), ring)
    for code_type in [codes.LPCode, codes.SLPCode]:
        code = code_type(empty)
        assert len(code) == code.dimension == 2 * 2 * ring.group.lift_dim

    # matrices with no columns give codes with no qudits, whose canonical logicals are empty
    empty = abstract.RingArray.build(np.zeros((0, 0), dtype=int), ring)
    for code_type in [codes.LPCode, codes.SLPCode]:
        code = code_type(empty, set_logicals=True)
        assert len(code) == code.dimension == 0
        assert code.get_logical_ops().shape == (0, 0)


def test_lifted_product_line_logicals_noncommutative(
    ring_dihedral3_gf5: abstract.GroupRing,
) -> None:
    """Line operators over a non-commutative ring, for a matrix that needs a basis change."""
    rot, ref = ring_dihedral3_gf5.group.generators
    values = [[ref * rot, ref * rot, rot**3], [ref, rot * ref, ref * rot]]
    matrix = abstract.RingArray.build(values, ring_dihedral3_gf5)
    for code in [
        codes.LPCode(matrix, set_logicals=True),
        codes.SLPCode(matrix, set_logicals=True),
    ]:
        logical_ops_x, logical_ops_z = code.get_logical_ops(Pauli.X), code.get_logical_ops(Pauli.Z)
        assert np.array_equal(logical_ops_x @ logical_ops_z.T, np.eye(code.dimension))


def test_lifted_product_valid_over_group_algebras() -> None:
    """LPCode and SLPCode set valid canonical logicals over commutative group algebras.

    Setting canonical logicals computes Howell duals, here of matrices whose entries have several
    terms.  The group algebras C3 over GF(4) and C4 over GF(5) have polynomial moduli, x^3 - 1 and
    x^4 - 1, that factor differently.
    """
    cases = [
        (
            abstract.CyclicGroup(3),
            4,
            ([[0, 2, 1], [2, 1, 0], [1, 0, 1]], [[0, 0, 1], [1, 1, 0], [1, 0, 0]]),
        ),
        (
            abstract.CyclicGroup(4),
            5,
            (
                [[0, 2, 1, 0], [2, 1, 0, 3], [1, 0, 1, 4]],
                [[0, 0, 1, 2], [1, 1, 0, 0], [1, 0, 0, 3]],
            ),
        ),
    ]
    for group, field_order, rows in cases:
        ring = abstract.GroupRing(group, field=field_order)
        assert ring.is_commutative
        matrix = abstract.RingArray.build(
            [
                [abstract.RingMember.from_vector(ring.field(coeffs), ring) for coeffs in row]
                for row in rows
            ],
            ring,
        )
        for code in (
            codes.LPCode(matrix, set_logicals=True),
            codes.SLPCode(matrix, set_logicals=True),
        ):
            # the code is nonzero and its canonical logicals are well-formed: Lx . Lz^T = I
            assert code.dimension > 0
            assert np.array_equal(
                code.get_logical_ops(Pauli.X) @ code.get_logical_ops(Pauli.Z).T,
                np.eye(code.dimension),
            )
            # an LPCode does not verify that its checks commute if the involution of its ring
            # transposes lifts, so check it here; the checks of an SLPCode need not commute
            if isinstance(code, codes.LPCode):
                assert not np.any(code.matrix_x @ code.matrix_z.T)


def test_unsupported_line_logicals() -> None:
    """Requesting line operators raises an error for seeds that fit neither route.

    The field characteristic divides the order of the group, so the semisimple route does not
    apply.  The trailing 2×2 block of the seed has determinant 1 + g, where g generates the group.
    This determinant is not a unit, so the seed does not split at trailing pivot blocks.
    """
    ring = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    gen = ring.generators[0]
    matrix = abstract.RingArray.build([[gen, 1, gen], [1, 1, 1]], ring)
    code_types: list[type[codes.LPCode | codes.SLPCode]] = [codes.LPCode, codes.SLPCode]
    for code_type in code_types:
        with pytest.raises(ValueError, match=r"semisimple.*do not fit the split route"):
            code_type(matrix, set_logicals=True)
        with pytest.raises(ValueError, match=r"semisimple.*do not fit the split route"):
            code_type.get_canonical_logical_line_ops(matrix, matrix)


def test_split_line_logicals() -> None:
    """Seeds that split at trailing pivot blocks have paired line operators.

    The split route follows Appendix B of arXiv:2607.28795v2.  The rings below are not semisimple
    group algebras, so their seeds avoid the semisimple route.
    """
    # dual numbers F_2[eps]/(eps^2), as the algebra of the cyclic group of order 2 over F_2
    ring_c2 = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    eps = ring_c2.one + ring_c2.generators[0]
    seed_c2 = abstract.RingArray.build([[eps, 1]], ring_c2)

    # a noncommutative ring, with a pivot block that is invertible in the left action that
    # seed_a_s3 needs, but not in the right action that seed_b_s3 needs.  The pivot block of
    # seed_b_s3 is its entrywise transpose (without the involution), and is invertible only in the
    # right action.
    ring_s3 = abstract.GroupRing(abstract.DihedralGroup(3), field=2)
    rot, ref = ring_s3.generators
    pivot = [[ring_s3.one, rot], [rot * ref, ring_s3.one + ref]]
    seed_a_s3 = abstract.RingArray.build([[rot, *pivot[0]], [ref, *pivot[1]]], ring_s3)
    seed_b_s3 = abstract.RingArray.build(
        [[ref, pivot[0][0], pivot[1][0]], [rot, pivot[0][1], pivot[1][1]]], ring_s3
    )
    one = np.ones((1, 1), dtype=int)
    pivot_a = abstract.RingArray.build(pivot, ring_s3)
    pivot_b = abstract.RingArray.build([list(row) for row in zip(*pivot)], ring_s3)
    for pivot_block, left_rank, right_rank in [(pivot_a, 12, 10), (pivot_b, 10, 12)]:
        assert np.linalg.matrix_rank(abstract.kron(pivot_block, one).lift()) == left_rank
        assert np.linalg.matrix_rank(abstract.kron(one, pivot_block).lift()) == right_rank

    # an extension field of odd characteristic, with coefficients outside its prime subfield, and
    # seeds with several free columns
    ring_gf9 = abstract.GroupRing(abstract.CyclicGroup(3), field=9)
    alpha = int(ring_gf9.field.primitive_element)
    member_u, member_v, member_w = [
        abstract.RingMember.from_vector(ring_gf9.field(coefficients), ring_gf9)
        for coefficients in ([1, alpha, 0], [0, 1, 1], [alpha, 0, 0])
    ]
    seed_a_gf9 = abstract.RingArray.build([[member_u, member_v, member_w]], ring_gf9)
    seed_b_gf9 = abstract.RingArray.build([[member_v, member_u, member_w]], ring_gf9)

    # a commutative matrix ring whose lifts are 3×3, although the ring has dimension 2
    ring_matrix = abstract.MatrixRing(galois.GF2([[0, 1, 0], [1, 0, 0], [0, 0, 1]]))
    assert ring_matrix.dim == 2 and ring_matrix.lift_dim == 3
    seed_matrix = abstract.RingArray.build(
        [[ring_matrix.one + ring_matrix.generators[0], 1]], ring_matrix
    )

    for seed_a, seed_b in [
        (seed_a_s3, seed_b_s3),
        (seed_a_gf9, seed_b_gf9),
        (seed_matrix, seed_matrix),
    ]:
        lp_code = codes.LPCode(seed_a, seed_b, set_logicals=True)
        slp_code = codes.SLPCode(seed_a, seed_b, set_logicals=True)
        assert_split_line_logicals(lp_code)
        assert_split_line_logicals(slp_code)

        # the operators of an SLPCode are those of the LPCode, restricted to its first block
        for pauli in (Pauli.X, Pauli.Z):
            assert np.array_equal(
                slp_code.get_logical_ops(pauli),
                lp_code.get_logical_ops(pauli)[:, : len(slp_code)],
            )

    # the dual-number seed has these line operators, as lifts of ring-valued operators
    code = codes.LPCode(seed_c2, set_logicals=True)
    assert len(code) == 10
    assert_split_line_logicals(code)
    expected_ops_x = abstract.RingArray.build([[1, eps, 0, 0, 0]], ring_c2).lift()
    expected_ops_z = abstract.RingArray.build([[1, 0, eps, 0, 0]], ring_c2).lift()
    assert np.array_equal(code.get_logical_ops(Pauli.X), expected_ops_x)
    assert np.array_equal(code.get_logical_ops(Pauli.Z), expected_ops_z)

    # SLPCode(seed, seed) uses each S3 seed in both actions, and its pivot block is invertible in
    # only one of them
    for seed in (seed_a_s3, seed_b_s3):
        with pytest.raises(ValueError, match="do not fit the split route"):
            codes.SLPCode(seed, seed, set_logicals=True)


def test_split_line_logicals_require_split_seeds() -> None:
    """The split route needs invertible trailing pivots and an involution that transposes lifts."""
    ring = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    eps = ring.one + ring.generators[0]
    code_types: list[type[codes.LPCode | codes.SLPCode]] = [codes.LPCode, codes.SLPCode]

    # square seeds with an invertible pivot block give codes with no logical qudits
    for code_type in code_types:
        code = code_type(abstract.RingArray.build([[1]], ring), set_logicals=True)
        assert code.dimension == 0
        assert_split_line_logicals(code)

    # pivot blocks that are not invertible, not square, or not trailing
    for values in ([[eps, eps]], [[1], [1]], [[1, eps]]):
        for code_type in code_types:
            with pytest.raises(ValueError, match="do not fit the split route"):
                code_type(abstract.RingArray.build(values, ring), set_logicals=True)

    # if the involution of the ring does not transpose lifts, a subsystem code from these seeds
    # encodes more qudits than the split route would give it
    field_2 = galois.GF(2)
    group = abstract.Group.from_sympy(
        abstract.CyclicGroup(2).to_sympy(),
        lift=lambda member: field_2([[1, 0], [0, 1]] if member.is_Identity else [[1, 0], [1, 1]]),
    )
    ring_with_shear = abstract.GroupRing(group, field=2)
    assert ring_with_shear.has_involution
    seed = abstract.RingArray.build([[ring_with_shear.generators[0], 1]], ring_with_shear)
    assert codes.SLPCode(seed).dimension == 4
    with pytest.raises(ValueError, match="do not fit the split route"):
        codes.SLPCode(seed, set_logicals=True)

    # likewise for a ring without an involution
    field = galois.GF(3)
    ring_without_involution = abstract.MatrixRing(field([[0, 1], [0, 0]]))
    assert not ring_without_involution.has_involution
    seed = abstract.RingArray.build(
        [[ring_without_involution.generators[0], 1]], ring_without_involution
    )
    assert codes.SLPCode(seed).dimension == 4
    with pytest.raises(NotImplementedError, match="only for group algebras"):
        codes.SLPCode.get_canonical_logical_line_ops(seed, seed)
    with pytest.raises(ValueError, match="do not fit the split route"):
        codes.SLPCode(seed, set_logicals=True)


def assert_split_line_logicals(code: codes.LPCode | codes.SLPCode) -> None:
    """Assert that a code has the logical line operators of the split route."""
    # by ranks of its checks, which are independent of the installed operators, the code encodes
    # (na - ma) * (nb - mb) * d logical qudits
    num_checks_a, num_bits_a = code.matrix_a.shape
    num_checks_b, num_bits_b = code.matrix_b.shape
    block_size = code.matrix_a.ring.lift_dim
    dimension = (num_bits_a - num_checks_a) * (num_bits_b - num_checks_b) * block_size
    rank_x = np.linalg.matrix_rank(code.matrix_x)
    rank_z = np.linalg.matrix_rank(code.matrix_z)
    num_gauge_pairs = np.linalg.matrix_rank(code.matrix_x @ code.matrix_z.T)
    assert len(code) - rank_x - rank_z + num_gauge_pairs == dimension
    if isinstance(code, codes.LPCode):
        assert num_gauge_pairs == 0

    # the logical operators commute with the checks and pair as the identity
    logical_ops_x = code.get_logical_ops(Pauli.X)
    logical_ops_z = code.get_logical_ops(Pauli.Z)
    assert np.array_equal(logical_ops_x @ logical_ops_z.T, np.eye(dimension, dtype=int))
    assert not np.any(code.matrix_z @ logical_ops_x.T)
    assert not np.any(code.matrix_x @ logical_ops_z.T)
    canonical_ops_x, canonical_ops_z = code.get_canonical_logical_line_ops(
        code.matrix_a, code.matrix_b
    )
    assert np.array_equal(canonical_ops_x, logical_ops_x)
    assert np.array_equal(canonical_ops_z, logical_ops_z)

    # X operators are supported on one row, and Z operators on one column, of the first block
    shape = (num_bits_a, num_bits_b, block_size)
    size = int(np.prod(shape))
    assert not np.any(logical_ops_x[:, size:]) and not np.any(logical_ops_z[:, size:])
    for op_x, op_z in zip(logical_ops_x, logical_ops_z):
        assert np.count_nonzero(np.any(op_x[:size].reshape(shape), axis=(1, 2))) == 1
        assert np.count_nonzero(np.any(op_z[:size].reshape(shape), axis=(0, 2))) == 1

    # on free cells, the operators are unit vectors in order of (row, column, block coordinate)
    for logical_ops in (logical_ops_x, logical_ops_z):
        grid = logical_ops[:, :size].reshape(dimension, *shape)
        free_values = grid[:, : num_bits_a - num_checks_a, : num_bits_b - num_checks_b]
        assert np.array_equal(
            free_values.reshape(dimension, dimension), np.eye(dimension, dtype=int)
        )


def test_lifted_product_translations() -> None:
    """Translations of lifted products over abelian groups preserve their checks."""
    ring = abstract.GroupRing(abstract.AbelianGroup(2, 3))
    gen_a, gen_b = ring.generators
    matrix_a = abstract.RingArray([[ring.one + gen_a, gen_b, gen_a * gen_b]])
    matrix_b = abstract.RingArray([[ring.one + gen_b], [gen_a]])
    for code in (codes.LPCode(matrix_a, matrix_b), codes.SLPCode(matrix_a, matrix_b)):
        translations = code._get_qudit_automorphisms()
        assert translations is not None and len(translations) == 2
        assert_valid_translations(code, translations, group_order=6)


def test_lifted_product_translations_require_group_algebra() -> None:
    """Block translations are known only for lifted products over group algebras."""
    ring = abstract.MatrixRing(np.array([[0, 1], [1, 0]]))
    matrix = abstract.RingArray([[ring.one + ring.generators[0]]])
    for code in (codes.LPCode(matrix), codes.SLPCode(matrix)):
        assert code._get_qudit_automorphisms() is None


def test_lifted_product_over_matrix_rings() -> None:
    """Lifted products over matrix rings match those over isomorphic group algebras."""

    def get_matrices(ring: abstract.Ring) -> tuple[abstract.RingArray, abstract.RingArray]:
        x = ring.generators[0]
        return (
            abstract.RingArray.build([[ring.one - x, x]], ring),
            abstract.RingArray.build([[x], [ring.one - x]], ring),
        )

    # the involution of this ring is not the identity, and the field has odd characteristic, so the
    # checks of an LPCode depend on the involution and on signs
    group_ring = abstract.GroupRing(abstract.CyclicGroup(3), field=3)
    ring = abstract.MatrixRing(group_ring.generators[0].lift())
    assert ring.generators[0].T != ring.generators[0]
    for code_type in [codes.LPCode, codes.SLPCode]:
        code = code_type(*get_matrices(ring))
        expected_code = code_type(*get_matrices(group_ring))
        assert np.array_equal(code.matrix_x, expected_code.matrix_x)
        assert np.array_equal(code.matrix_z, expected_code.matrix_z)


def test_lifted_product_ring_requirements() -> None:
    """Lifted products check the requirements on their rings."""
    # a noncommutative ring must be a group algebra.  This ring also has no involution, so these
    # errors show that the ring is checked before an involution is needed.
    upper, lower = np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]])
    noncommutative_ring = abstract.MatrixRing(np.diag([1, 0]), upper)
    assert not noncommutative_ring.is_commutative and not noncommutative_ring.has_involution
    matrix = abstract.RingArray.build([[1, noncommutative_ring.generators[1]]], noncommutative_ring)
    for code_type in [codes.LPCode, codes.SLPCode]:
        with pytest.raises(NotImplementedError, match="only for group algebras"):
            code_type(matrix)

    # a noncommutative group algebra must use the default lift
    group_ring = abstract.GroupRing(abstract.SymmetricGroup(3).with_natural_lift())
    matrix = abstract.RingArray.build([[1, group_ring.generators[0]]], group_ring)
    for code_type in [codes.LPCode, codes.SLPCode]:
        with pytest.raises(ValueError, match="Anti-representations"):
            code_type(matrix)

    # an LPCode needs an involution, but an SLPCode does not
    ring_without_involution = abstract.MatrixRing(lower)
    assert not ring_without_involution.has_involution and ring_without_involution.is_commutative
    matrix = abstract.RingArray.build(
        [[ring_without_involution.generators[0]]], ring_without_involution
    )
    with pytest.raises(ValueError, match="no involution"):
        codes.LPCode(matrix)
    subsystem_code = codes.SLPCode(matrix)
    assert np.array_equal(subsystem_code.matrix_x, matrix.lift())
    assert np.array_equal(subsystem_code.matrix_z, matrix.lift())

    # operators of the semisimple route need a group algebra, whether or not the ring has an
    # involution, and a group algebra paired with another ring raises this error rather than the
    # error for different rings.  These seeds do not fit the split route.
    ring_with_involution = abstract.MatrixRing(np.diag([1, 0]))
    assert ring_with_involution.is_commutative and ring_with_involution.has_involution
    get_lifted_ops: list[Callable[[abstract.RingArray, abstract.RingArray], object]] = [
        codes.LPCode.get_canonical_logical_line_ops,
        codes.SLPCode.get_canonical_logical_line_ops,
    ]
    get_ring_ops: list[Callable[[abstract.RingArray, abstract.RingArray], object]] = [
        codes.LPCode.get_canonical_ring_logical_line_ops,
        codes.SLPCode.get_canonical_ring_logical_line_ops,
    ]
    group_ring = abstract.GroupRing(abstract.CyclicGroup(2))
    group_matrix = abstract.RingArray.build([[1]], group_ring)
    for ring in [ring_with_involution, ring_without_involution]:
        matrix = abstract.RingArray.build([[ring.generators[0]]], ring)
        for get_canonical_logical_line_ops in get_lifted_ops + get_ring_ops:
            for matrices in [(matrix, matrix), (group_matrix, matrix), (matrix, group_matrix)]:
                with pytest.raises(NotImplementedError, match="only for group algebras"):
                    get_canonical_logical_line_ops(*matrices)
    matrix = abstract.RingArray.build([[ring_with_involution.generators[0]]], ring_with_involution)
    for code_type in [codes.LPCode, codes.SLPCode]:
        with pytest.raises(ValueError, match="Cannot set canonical logical operators"):
            code_type(matrix, set_logicals=True)

    # the split route declines a noncommutative ring that is not a group algebra, even if its
    # involution transposes lifts and its seeds have an invertible trailing pivot block
    full_matrix_ring = abstract.MatrixRing(upper, lower)
    assert not full_matrix_ring.is_commutative and full_matrix_ring.has_involution
    matrix = abstract.RingArray.build([[full_matrix_ring.generators[0], 1]], full_matrix_ring)
    for get_canonical_logical_line_ops in get_lifted_ops:
        with pytest.raises(NotImplementedError, match="do not fit the split route"):
            get_canonical_logical_line_ops(matrix, matrix)


def test_lifted_product_sector_sizes() -> None:
    """Lifted products use the size of lifted matrices, which can exceed the dimension of a ring."""
    ring = abstract.MatrixRing(np.diag([1, 0, 0]))
    assert ring.dim == 2 and ring.lift_dim == 3
    matrix = abstract.RingArray.build([[ring.generators[0]]], ring)
    code = codes.LPCode(matrix)
    assert code.num_qudits == code.sector_size[0, 0] + code.sector_size[1, 1] == 6


def test_lifted_product_translations_with_larger_lifts() -> None:
    """Translations of lifted products act on blocks of the size of lifted matrices."""
    group = abstract.Group.from_generating_mats(np.kron(np.eye(2, dtype=int), [[0, 1], [1, 0]]))
    assert group.order == 2 and group.lift_dim == 4
    ring = abstract.GroupRing(group)
    matrix = abstract.RingArray.build([[ring.one + ring.generators[0]]], ring)
    for code in (codes.LPCode(matrix), codes.SLPCode(matrix)):
        translations = code._get_qudit_automorphisms()
        assert translations is not None
        assert_valid_translations(code, translations, group_order=group.order)


def test_lifted_product_with_non_orthogonal_lifts() -> None:
    """LPCode verifies that its X and Z checks commute when lifts are not orthogonal."""
    # the first generator of this group lifts to an orthogonal matrix, and the second does not
    swap = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    shear = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]
    ring = abstract.GroupRing(
        abstract.Group.from_generating_mats(galois.GF2(swap), galois.GF2(shear))
    )
    first, second = ring.generators
    identity = np.eye(4, dtype=int)
    assert np.array_equal(first.lift() @ first.lift().T, identity)
    assert not np.array_equal(second.lift() @ second.lift().T, identity)
    with pytest.raises(ValueError, match="do not commute"):
        codes.LPCode(abstract.RingArray.build([[ring.one, second]], ring))

    # LPCode accepts commuting checks from lifts that are not orthogonal, here because the second
    # generator is its own inverse
    assert second * second == ring.one
    code = codes.LPCode(abstract.RingArray.build([[second]], ring))
    assert not np.any(code.matrix_x @ code.matrix_z.T)

    # this generator lifts to a symmetric matrix that is not orthogonal, and in odd characteristic,
    # the minus sign in the Z checks of lifted products keeps the checks of [[g]] from commuting
    group = abstract.Group.from_generating_mats(galois.GF(5)([[1, 1], [1, 4]]))
    ring = abstract.GroupRing(group, field=5)
    (generator,) = ring.generators
    assert np.array_equal(generator.lift(), generator.lift().T)
    assert not np.array_equal(generator.lift() @ generator.lift().T, np.eye(2, dtype=int))
    with pytest.raises(ValueError, match="do not commute"):
        codes.LPCode(abstract.RingArray.build([[generator]], ring))


def test_canonical_logicals_with_non_orthogonal_lifts() -> None:
    """With lifts that are not orthogonal, logical operators lift, and only valid ones are set."""
    # lift the generator of a cyclic group of order 2 to a matrix that is not orthogonal, over a
    # field for which qLDPC knows the primitive central idempotents of this group algebra
    field = galois.GF(3)
    lift = field([[1, 0], [1, 2]])
    group = abstract.Group.from_sympy(
        abstract.CyclicGroup(2).to_sympy(),
        lift=lambda member: field.Identity(2) if member.is_Identity else lift,
    )
    ring = abstract.GroupRing(group, field=3)
    (generator,) = ring.generators
    assert np.array_equal(generator.lift() @ generator.lift(), np.eye(2, dtype=int))
    assert not np.array_equal(generator.lift() @ generator.lift().T, np.eye(2, dtype=int))

    # the overlap matrices of these lifted operators are the lift of 1 - g and its transpose.  Their
    # top-left entries are zero, and their pivot rows and pivot columns differ.
    one = abstract.RingArray.build([[ring.one]], ring)
    one_minus_g = abstract.RingArray.build([[ring.one - generator]], ring)
    for ring_ops_x, ring_ops_z in [(one_minus_g, one), (one, one_minus_g)]:
        ops_x, ops_z = codes.LPCode.to_lifted_logical_ops(ring_ops_x, ring_ops_z)
        assert np.array_equal(ops_x @ ops_z.T, np.eye(1, dtype=int))

    # operators whose overlap matrix is zero lift to no operators
    zero = abstract.RingArray.build([[ring.zero]], ring)
    ops_x, ops_z = codes.LPCode.to_lifted_logical_ops(zero, one)
    assert ops_x.shape == ops_z.shape == (0, 2)

    # logical operators come in pairs
    with pytest.raises(ValueError, match="must come in pairs"):
        codes.LPCode.to_lifted_logical_ops(abstract.RingArray.build([[1], [1]], ring), one)

    code_types: list[type[codes.LPCode | codes.SLPCode]] = [codes.LPCode, codes.SLPCode]
    for code_type in code_types:
        # these canonical logical operators are valid
        code = code_type(
            abstract.RingArray.build([[generator, generator]], ring), set_logicals=True
        )
        logical_ops_x, logical_ops_z = code.get_canonical_logical_line_ops(
            code.matrix_a, code.matrix_b
        )
        assert code.dimension > 0
        assert np.array_equal(code.get_logical_ops(Pauli.X), logical_ops_x)
        assert np.array_equal(code.get_logical_ops(Pauli.Z), logical_ops_z)

        # the canonical logical operators for 1 - g fail validation
        with pytest.raises(ValueError, match=r"canonical logical operators.*does not transpose"):
            code_type(one_minus_g, set_logicals=True)
