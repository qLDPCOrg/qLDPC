# SPDX-License-Identifier: Apache-2.0

"""Unit tests for linalg.py."""

from __future__ import annotations

import itertools
import unittest.mock

import numpy as np
import pytest

from qldpc import abstract
from qldpc.abstract.linalg import get_howell_dual


def test_basics(pytestconfig: pytest.Config, rows: int = 2, cols: int = 3) -> None:
    """Basic linear algebra over commutative rings."""
    seed = pytestconfig.getoption("randomly_seed")
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    matrix_a = abstract.RingArray.from_field_array(
        ring.field.Random((rows, cols, ring.dim), seed=seed), ring
    )
    matrix_b = abstract.RingArray.from_field_array(
        ring.field.Random((rows, cols, ring.dim), seed=seed + 1), ring
    )

    # matrix multiplication and the Kronecker product agree with lifted matrices and entries
    assert np.array_equal(
        abstract.matmul(matrix_a, matrix_b.T).lift(),
        matrix_a.lift() @ matrix_b.T.lift(),
    )
    product = abstract.kron(matrix_a, matrix_b)
    for (ii, jj), (kk, ll) in itertools.product(np.ndindex(rows, cols), repeat=2):
        assert product[ii * rows + kk, jj * cols + ll] == matrix_a[ii, jj] * matrix_b[kk, ll]

    # block_diag places its matrices on the diagonal
    matrix_ab = abstract.block_diag(matrix_a, matrix_b)
    assert np.array_equal(matrix_ab[: matrix_a.shape[0], : matrix_a.shape[1]], matrix_a)
    assert np.array_equal(matrix_ab[matrix_a.shape[0] :, matrix_a.shape[1] :], matrix_b)
    assert not np.any(matrix_ab[: matrix_a.shape[0], matrix_a.shape[1] :])
    assert not np.any(matrix_ab[matrix_a.shape[0] :, : matrix_a.shape[1]])

    # these functions also work if only one matrix is a RingArray
    assert np.array_equal(abstract.matmul(np.eye(matrix_a.shape[0], dtype=int), matrix_a), matrix_a)
    assert np.array_equal(abstract.matmul(matrix_a, np.eye(matrix_a.shape[1], dtype=int)), matrix_a)

    # ... but we need at least one of them to be a RingArray
    with pytest.raises(ValueError, match=r"At least one .* RingArray"):
        abstract.kron(np.eye(1, dtype=int), np.eye(1, dtype=int))

    # matrix dimensions must be compatible for matrix multiplication
    with pytest.raises(ValueError, match="Incompatible matrix shapes"):
        abstract.matmul(matrix_a, matrix_b)
    with pytest.raises(ValueError, match="Incompatible matrix shapes"):
        abstract.matmul(matrix_a, matrix_a[0])
    with pytest.raises(ValueError, match="Incompatible matrix shapes"):
        abstract.matmul(matrix_a[0], matrix_a.T)

    # we can't multiply matrices over different rings
    other_ring = abstract.GroupRing(ring.group, field=ring.field.order + 1)
    other_matrix = abstract.RingArray.build(np.eye(matrix_a.shape[0], dtype=int), other_ring)
    with pytest.raises(ValueError, match="different rings"):
        abstract.matmul(other_matrix, matrix_a)


def random_monomial_matrix(
    ring: abstract.GroupRing, rows: int, cols: int, seed: int
) -> abstract.RingArray:
    """A matrix of random group members."""
    values = [
        [ring.group.random(seed=seed + cols * row + col) for col in range(cols)]
        for row in range(rows)
    ]
    return abstract.RingArray.build(values, ring)


def test_matmul_and_kron_interplay(
    ring: abstract.GroupRing, pytestconfig: pytest.Config, rows: int = 2, cols: int = 3
) -> None:
    """Matrix multiplication and the Kronecker product work together as expected."""
    seed = pytestconfig.getoption("randomly_seed")
    matrix_a, matrix_b, matrix_c, matrix_d = [
        random_monomial_matrix(ring, rows, cols, seed + index * rows * cols) for index in range(4)
    ]
    matrix_c = matrix_c.transpose()
    matrix_d = matrix_d.transpose()

    # (A ⨂ B) @ (C ⨂ D)
    matrix_ab = abstract.kron(matrix_a, matrix_b)
    matrix_cd = abstract.kron(matrix_c, matrix_d)
    matrix_ab_cd = abstract.matmul(matrix_ab, matrix_cd)

    # build (A @ C) ⨂ (B @ D), with reversed order of ring multiplication in second tensor factor
    matrix_ac = abstract.matmul(matrix_a, matrix_c)
    matrix_bd = abstract.matmul(matrix_b, matrix_d, right=True)
    matrix_ac_bd = abstract.kron(matrix_ac, matrix_bd)
    assert np.array_equal(matrix_ab_cd, matrix_ac_bd)

    if not ring.is_commutative:
        # assert that elements of the bimodule R ⨂ R transform elements of the ring R correctly
        val_a = matrix_a[0, 0]
        val_b = matrix_b[0, 0]
        val_ab = matrix_ab[0, 0]
        random_val = abstract.RingMember.from_vector(
            ring.field.Random(ring.group.order, seed=seed), ring
        )
        random_vec = random_val.to_vector()
        assert np.array_equal(
            val_ab.lift() @ random_vec,
            val_a.lift() @ (val_b.lift(right=True) @ random_vec),
        )


def test_matmul_right() -> None:
    """Matrix multiplication can reverse the order of multiplication in a noncommutative ring."""
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=3)
    dense = abstract.RingArray.from_field_array(
        ring.field.Random((4, 4, ring.group.order), seed=0), ring
    )
    monomial = abstract.RingArray.build(np.full((4, 4), ring.generators[0], dtype=object), ring)
    for matrix_a, matrix_b in [(dense, monomial), (monomial, dense)]:
        expected = abstract.RingArray.build(np.zeros((4, 4), dtype=int), ring)
        for row, col in np.ndindex(expected.shape):
            expected[row, col] = sum(matrix_b[kk, col] * matrix_a[row, kk] for kk in range(4))
        assert np.array_equal(abstract.matmul(matrix_a, matrix_b, right=True), expected)

    # stacks of matrices broadcast, as in numpy
    stack = abstract.RingArray.from_field_array(
        ring.field.Random((3, 2, 2, ring.dim), seed=1), ring
    )
    block = abstract.RingArray(dense[:2, :2])
    product = abstract.matmul(block, stack, right=True)
    assert product.shape == (3, 2, 2)
    for index, matrix in enumerate(stack):
        assert np.array_equal(product[index], abstract.matmul(block, matrix, right=True))


def assert_howell_pseudoinverse(
    matrix: abstract.RingArray, transformer: abstract.WedderburnArtinTransformer, right: bool
) -> None:
    """Assert the null-space Howell form of ``matrix`` has a dual that acts like a pseudoinverse."""
    # find a null-space matrix in Howell normal form
    generator = matrix.null_space(right=right).howell_normal_form_semisimple(right=right)
    assert not np.any(abstract.matmul(matrix, generator.T, right=right))

    # find the "dual" of the null-space matrix, which acts like a pseudoinverse.  `right` must match
    # the orientation used to build the Howell form.
    dual = get_howell_dual(generator, right=right)
    diag = abstract.matmul(generator, dual.T, right=right)
    assert np.array_equal(diag.astype(bool), np.eye(len(diag), dtype=bool))
    assert np.array_equal(abstract.matmul(diag, generator, right=right), generator)
    assert np.array_equal(abstract.matmul(diag.T, dual, right=right), dual)
    assert np.array_equal(diag, transformer.transpose_array(diag))


@pytest.mark.parametrize("right", [False, True])
def test_howell_dual(
    ring: abstract.GroupRing, right: bool, pytestconfig: pytest.Config, rows: int = 2, cols: int = 3
) -> None:
    """Matrices in Howell normal form have a "dual" that acts like a pseudoinverse."""
    matrix = random_monomial_matrix(ring, rows, cols, pytestconfig.getoption("randomly_seed"))
    assert_howell_pseudoinverse(matrix, ring.get_transformer(), right)


@pytest.mark.parametrize("right", [False, True])
def test_howell_dual_with_non_self_transpose_pivots(right: bool) -> None:
    """A non-self-transpose idempotent pivot distinguishes pivot.T from pivot in get_howell_dual.

    Multi-term entries make the null-space Howell pivots idempotents with several terms.  The
    degree-1 components of GF(4)[C3] form a conjugate pair of characters, so their idempotents E
    satisfy ``E.T != E``.
    """
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=4)
    multi_term = abstract.RingArray.build(
        [
            [abstract.RingMember.from_vector(ring.field(coeffs), ring) for coeffs in row]
            for row in ([[0, 2, 1], [2, 1, 0], [1, 0, 1]], [[0, 0, 1], [1, 1, 0], [1, 0, 0]])
        ],
        ring,
    )
    assert_howell_pseudoinverse(multi_term, ring.get_transformer(seed=0), right)


def build_right_howell_generator_without_left_dual(
    ring: abstract.GroupRing,
) -> tuple[abstract.WedderburnArtinTransformer, abstract.RingArray]:
    """Build a right-oriented Howell form over GF(5)[D₃] with no dual for right=False (a left dual).

    The Howell form is a single entry, which lies in the size-2 component of the ring.  Its block
    there, [[1, 0], [2, 0]], is a right-oriented pivot whose columns span the line through (1, 2).
    This line is orthogonal to itself, since 1² + 2² = 0 in GF(5), so it has no symmetric idempotent
    projector, which the dual for right=False needs.

    Args:
        ring: The ring GF(5)[D₃], as the ring_dihedral3_gf5 fixture builds it.

    Returns:
        The transformer that built the Howell form, and the Howell form.
    """
    transformer = ring.get_transformer()
    component_transformer = next(ct for ct in transformer.transformers if ct.size == 2)
    pivot = component_transformer.embed(component_transformer.extended_field([[1, 0], [2, 0]]))
    generator = abstract.RingArray.build([[pivot]]).howell_normal_form_semisimple(
        transformer=transformer, right=True
    )
    return transformer, generator


def test_howell_dual_requires_matching_right(ring_dihedral3_gf5: abstract.GroupRing) -> None:
    """The dual depends on the orientation used to build the Howell form.

    The ring has a size-2 component, so this tests the dual for blocks bigger than a single number.
    Passing the same ``right`` used to build the Howell form gives a working dual.  The Howell form
    records the orientation it was built with, so passing the wrong ``right`` is rejected.
    """
    transformer, generator = build_right_howell_generator_without_left_dual(ring_dihedral3_gf5)

    # with the matching orientation, the dual acts as a pseudoinverse: generator @ dual.T is a
    # diagonal that leaves both the generator and the dual unchanged, and is its own transpose
    dual = get_howell_dual(generator, transformer=transformer, right=True)
    diag = abstract.matmul(generator, dual.T, right=True)
    assert np.array_equal(diag.astype(bool), np.eye(len(diag), dtype=bool))
    assert np.array_equal(abstract.matmul(diag, generator, right=True), generator)
    assert np.array_equal(abstract.matmul(diag.T, dual, right=True), dual)
    assert np.array_equal(diag, transformer.transpose_array(diag))

    # omitting `right` infers it from the orientation the Howell form recorded; a wrong inference
    # would raise, since this generator has no dual for right=False
    assert np.array_equal(get_howell_dual(generator, transformer=transformer), dual)

    # a copy has no provenance, so it is validated, with the given orientation
    copy = generator.copy()
    assert np.array_equal(get_howell_dual(copy, transformer=transformer, right=True), dual)

    # skip_validation=False re-checks the constructed dual; it passes for the correct orientation
    rechecked = get_howell_dual(
        generator, transformer=transformer, right=True, skip_validation=False
    )
    assert np.array_equal(rechecked, dual)

    # the generator records right=True, so building the dual with right=False is rejected
    with pytest.raises(ValueError, match="does not match the orientation"):
        get_howell_dual(generator, transformer=transformer, right=False)

    # an edited Howell form keeps its recorded orientation as its default, and is validated with any
    # explicit orientation
    generator[0, 0] = generator[0, 0].copy()
    assert np.array_equal(get_howell_dual(generator, transformer=transformer), dual)
    with pytest.raises(ValueError, match="Cannot build a Howell dual"):
        get_howell_dual(generator, transformer=transformer, right=False)


@pytest.mark.parametrize("right", [False, True])
def test_howell_dual_uses_the_recorded_transformer(right: bool) -> None:
    """By default, get_howell_dual builds a dual in the basis that built the Howell form."""
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    default = ring.get_transformer(seed=0)
    transformer = ring.get_transformer(seed=1)

    # for this matrix, a dual in the basis of the default transformer cannot be built
    coeffs = np.random.default_rng(4).integers(0, ring.field.order, size=(1, 2, ring.dim))
    matrix = abstract.RingArray.from_field_array(coeffs, ring)
    generator = matrix.null_space(right=right).howell_normal_form_semisimple(
        transformer, right=right
    )
    with pytest.raises(ValueError, match="Cannot build a Howell dual"):
        get_howell_dual(generator, transformer=default)

    dual = get_howell_dual(generator)
    diag = abstract.matmul(generator, dual.T, right=right)
    assert np.array_equal(diag.astype(bool), np.eye(len(diag), dtype=bool))
    assert np.array_equal(abstract.matmul(diag, generator, right=right), generator)
    assert np.array_equal(abstract.matmul(diag.T, dual, right=right), dual)
    assert np.array_equal(diag, transformer.transpose_array(diag))

    # an edited Howell form keeps its recorded transformer as its default
    generator[0, 0] = generator[0, 0].copy()
    assert np.array_equal(get_howell_dual(generator), dual)


def test_howell_dual_trusts_unchanged_howell_forms(monkeypatch: pytest.MonkeyPatch) -> None:
    """By default, get_howell_dual validates all duals but those of unchanged Howell forms.

    A Howell form is trusted only while its entries are unchanged, and only in the basis that built
    it.
    """
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    transformer, other_transformer = ring.get_transformer(seed=0), ring.get_transformer(seed=1)
    coeffs = np.random.default_rng(0).integers(0, ring.field.order, size=(2, 3, ring.dim))
    howell = abstract.RingArray.from_field_array(coeffs, ring).howell_normal_form_semisimple(
        transformer
    )
    edited = abstract.RingArray.from_field_array(coeffs, ring).howell_normal_form_semisimple(
        transformer
    )
    edited[0, 0] = edited[0, 0].copy()

    # watch the validation of duals
    validate = unittest.mock.Mock(wraps=abstract.linalg._validate_howell_dual)
    monkeypatch.setattr(abstract.linalg, "_validate_howell_dual", validate)
    cases: list[tuple[abstract.RingArray, abstract.WedderburnArtinTransformer | None, bool]] = [
        (howell, None, False),
        (howell, transformer, False),
        (howell, other_transformer, True),
        (howell.copy(), None, True),
        (edited, None, True),
    ]
    for matrix, given_transformer, validated in cases:
        validate.reset_mock()
        get_howell_dual(matrix, transformer=given_transformer)
        assert validate.called == validated


def test_howell_dual_provenance(ring_cyclic3_gf2: abstract.GroupRing) -> None:
    """get_howell_dual trusts only the unmodified output of howell_normal_form_semisimple."""
    ring = ring_cyclic3_gf2
    idempotent_a, idempotent_b = ring.get_primitive_central_idempotents()

    def get_howell_form() -> abstract.RingArray:
        return abstract.RingArray.build([[idempotent_a, idempotent_b]]).howell_normal_form()

    howell = get_howell_form()
    get_howell_dual(howell)

    # the orientation of a Howell form over a commutative ring does not matter
    get_howell_dual(howell, right=True)

    # derived arrays, and arrays edited in place, are validated
    unit = abstract.RingArray.build([[0, 1], [0, 0]], ring)
    with pytest.raises(ValueError, match="could not validate"):
        get_howell_dual(howell + unit)
    howell = get_howell_form()
    howell[0][1] = ring.one
    with pytest.raises(ValueError, match="could not validate"):
        get_howell_dual(howell)

    # so are polynomial Howell forms, whose pivots need not be idempotent
    x = ring.generators[0]
    howell = abstract.RingArray.build([[1 + x, x]], ring).howell_normal_form_poly()
    with pytest.raises(ValueError, match="could not validate"):
        get_howell_dual(howell)

    # a zero row has no pivot, so the diagonal of its dual product is zero
    with pytest.raises(ValueError, match="could not validate"):
        get_howell_dual(abstract.RingArray.build([[idempotent_a], [0]], ring))

    with pytest.raises(ValueError, match="2-dimensional"):
        get_howell_dual(abstract.RingArray.build([idempotent_a], ring))


def test_commutative_howell_dual_needs_no_decomposition() -> None:
    """Over a commutative ring, get_howell_dual needs no Wedderburn-Artin decomposition.

    GF(2)[C2] is not semisimple, so it has no Wedderburn-Artin decomposition.
    """
    ring = abstract.GroupRing(abstract.CyclicGroup(2), field=2)
    x = ring.generators[0]
    dual = get_howell_dual(abstract.RingArray.build([[1, x]], ring))
    assert np.array_equal(dual, abstract.RingArray.build([[1, 0]], ring))


def test_get_howell_dual_edge_cases(ring_dihedral3_gf5: abstract.GroupRing) -> None:
    """Rejection of a non-Howell input, and of a mismatched orientation reaching the projector."""
    # a matrix that is not in Howell normal form (here a multi-term, non-idempotent pivot) has no
    # valid dual, so the default validation rejects it
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=4)
    pivot = abstract.RingMember.from_vector(ring.field([1, 1, 0]), ring)
    non_hnf = abstract.RingArray.build([[pivot]], ring)
    with pytest.raises(ValueError, match="could not validate"):
        get_howell_dual(non_hnf)

    # skip_validation bypasses the orientation check, so a mismatched `right` reaches the projector,
    # where a row's projected blocks span a space with no symmetric idempotent projector
    transformer, generator = build_right_howell_generator_without_left_dual(ring_dihedral3_gf5)
    with pytest.raises(ValueError, match="Cannot build a Howell dual"):
        get_howell_dual(generator, transformer=transformer, right=False, skip_validation=True)


def test_kron_over_matrix_rings() -> None:
    """The Kronecker product over a matrix ring needs a commutative ring."""
    matrix_units = [np.array([[0, 1], [0, 0]]), np.array([[0, 0], [1, 0]])]
    ring = abstract.MatrixRing(*matrix_units)
    matrix = abstract.RingArray.build([[1, ring.generators[0]]], ring)
    with pytest.raises(NotImplementedError, match="only for group algebras"):
        abstract.kron(matrix, matrix)

    # over a commutative matrix ring, kron is the ordinary Kronecker product
    ring = abstract.MatrixRing(matrix_units[0] + matrix_units[1])
    matrix = abstract.RingArray.build([[1, ring.generators[0]]], ring)
    product = abstract.kron(matrix, matrix)
    for index, (aa, bb) in enumerate(itertools.product(matrix[0], repeat=2)):
        assert product[0, index] == aa * bb


def test_howell_dual_requires_group_algebra(ring_cyclic3_gf2: abstract.GroupRing) -> None:
    """Howell duals are implemented only for group algebras, even given a transformer."""
    ring = abstract.MatrixRing(np.array([[0, 1], [1, 0]]))
    transformer = ring_cyclic3_gf2.get_transformer()
    with pytest.raises(NotImplementedError, match="only for group algebras"):
        get_howell_dual(
            abstract.RingArray.build([[1]], ring), transformer=transformer, skip_validation=True
        )


def test_kron_over_groups_with_different_lifts() -> None:
    """Kronecker products use the lifts of their own group algebras."""
    # these group algebras differ only in their right lifts
    group = abstract.SymmetricGroup(3)
    members = list(group.generate())
    translated = abstract.Group(
        *group.generators, generate_func=lambda: iter([gg * group.generators[1] for gg in members])
    )
    assert all(np.array_equal(group.lift(gg), translated.lift(gg)) for gg in group.generators)
    first = group.generators[0]
    assert not np.array_equal(group.lift(first, right=True), translated.lift(first, right=True))

    # for a generator g, the Kronecker product of [[g]] with itself maps each member t to g·t·g
    for ring in [abstract.GroupRing(group), abstract.GroupRing(translated)]:
        generator = ring.generators[0]
        matrix = abstract.RingArray.build([[generator]], ring)
        lifted = abstract.kron(matrix, matrix).lift()
        for member in ring.group.generate():
            element = abstract.RingMember(ring, member)
            product = generator * element * generator
            assert np.array_equal(lifted @ element.to_vector(), product.to_vector())
