# SPDX-License-Identifier: Apache-2.0

"""Unit tests for lifted_product.py."""

from __future__ import annotations

import galois
import networkx as nx
import numpy as np
import pytest
from sympy.abc import x

import qldpc
from qldpc import abstract, codes
from qldpc.objects import ChainComplex, Pauli

from ..conftest import assert_valid_translations


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

    # line operators for the LPCode have lower weight than Gottesman-canonical logicals
    logical_line_ops = code_LP.get_logical_ops()
    logical_ops = code_LP.get_logical_ops(recompute=True)
    line_weights = np.count_nonzero(logical_line_ops.view(np.ndarray), axis=1)
    weights = np.count_nonzero(logical_ops.view(np.ndarray), axis=1)
    assert np.all(line_weights <= weights)


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


def test_subsystem_lifted_product_requires_one_ring() -> None:
    """The two matrices of a subsystem lifted product must be over the same ring."""
    # these group algebras differ, but lift their members to matrices of the same size
    ring_a = abstract.GroupRing(abstract.CyclicGroup(4))
    ring_b = abstract.GroupRing(abstract.AbelianGroup(2, 2))
    assert ring_a != ring_b and ring_a.group.lift_dim == ring_b.group.lift_dim
    matrix_a = abstract.RingArray.build([[1, ring_a.generators[0]]], ring_a)
    matrix_b = abstract.RingArray.build([[1, ring_b.generators[0]]], ring_b)
    with pytest.raises(ValueError, match="same ring"):
        codes.SLPCode(matrix_a, matrix_b)


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
    for group, characteristic, rows in cases:
        ring = abstract.GroupRing(group, field=characteristic)
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
            # an ordinary CSS code's stabilizers commute; a subsystem code's need not
            if not code.is_subsystem_code:
                assert not np.any(code.matrix_x @ code.matrix_z.T)


def test_unsupported_line_logicals(rows: int = 2, cols: int = 3) -> None:
    """We do not support line operators in lifted product codes with non-semisimple rings."""
    ring = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    values = [[ring.group.random() for _ in range(cols)] for _ in range(rows)]
    matrix = abstract.RingArray.build(values, ring)
    with pytest.raises(ValueError, match="not yet supported"):
        codes.LPCode(matrix, set_logicals=True)
    with pytest.raises(ValueError, match="not yet supported"):
        codes.SLPCode(matrix, set_logicals=True)


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
        with pytest.raises(ValueError, match=r"canonical logical operators.*not orthogonal"):
            code_type(one_minus_g, set_logicals=True)
