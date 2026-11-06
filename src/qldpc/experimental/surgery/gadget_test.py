# SPDX-License-Identifier: Apache-2.0

"""Tests for src/qldpc/experimental/surgery/gadget.py."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from .conftest import (
    _webster_x_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)

WEBSTER_TABLE_1_BARE_GADGET_QUBITS = [(0, 19), (1, 31), (2, 49), (3, 79)]


def test_gadget_layout_is_frozen_dataclass() -> None:
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.basis = Pauli.Z  # type: ignore[misc]


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_restrict_checks_to_support_steane(basis: PauliXZ) -> None:
    """F restricts the complementary check matrix (H_Z for basis=X, H_X for basis=Z) to V_0."""
    from qldpc.experimental.surgery.gadget import _restrict_checks_to_support

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
    support, data_checks, incidence = _restrict_checks_to_support(code, x, basis=basis)
    H_comp = np.asarray(code.matrix_z if basis is Pauli.X else code.matrix_x).astype(np.uint8)
    # V_0 = supp(x), sorted ascending
    assert support == tuple(int(i) for i in np.where(x)[0])
    # C_0 = complementary checks touching V_0, sorted ascending
    touched = sorted({j for j in range(H_comp.shape[0]) for i in support if H_comp[j, i] == 1})
    assert data_checks == tuple(touched)
    # F = H_comp[C_0, V_0]
    assert np.array_equal(incidence, H_comp[np.ix_(data_checks, support)])
    # F @ 1_{V0} == 0 (invariant of the Webster §II A restriction)
    ones = np.ones(len(support), dtype=np.uint8)
    assert np.array_equal((incidence @ ones) % 2, np.zeros(len(data_checks), dtype=np.uint8))


def test_compute_gauge_basis() -> None:
    """G is a basis of ker(F^T): G F = 0, its rows are independent, and it has |C_0| - rank(F) rows.

    Webster §II A step 3 requires one independent gauge constraint per basis element, so redundant
    rows would inflate the gauge-check count without changing the gauge structure.  The fixtures are
    a rank-2 3x3 F (one gauge row) and the Webster code-0 X̄_1 gadget's F (four gauge rows).
    """
    import galois

    from qldpc.experimental.surgery.gadget import _compute_gauge_basis, build_gadget

    data = load_webster_seed_set(0)
    webster = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    webster_gadget = build_gadget(webster, _webster_x_bar_operator(data), basis=Pauli.X)
    incidences = [
        np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8),
        webster_gadget.incidence,
    ]
    for incidence, expected_rows in zip(incidences, [1, 4]):
        gauge = _compute_gauge_basis(incidence)
        rank_F = int(np.linalg.matrix_rank(galois.GF2(incidence.astype(np.int_))))
        assert gauge.shape == (incidence.shape[0] - rank_F, incidence.shape[0])
        assert gauge.shape[0] == expected_rows
        assert int(np.linalg.matrix_rank(galois.GF2(gauge.astype(np.int_)))) == gauge.shape[0]
        assert not ((gauge @ incidence) % 2).any()


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_gadget_merged_check_shapes_with_distinct_support_and_check_counts(
    basis: PauliXZ,
) -> None:
    """The merged checks add |V_0| χ rows, |C_0| - rank(F) gauge rows, and |C_0| κ columns.

    A 5-qubit CSS code (k = 1) whose measured logical has support size nV = 4 but touches only
    nC = 2 complementary checks pins the shapes, which a mix-up of nV and nC would break.  Each
    touched complementary check gains exactly one κ qubit (the indicator form of F̃).  The Z-basis
    case uses the dual code, with H_X and H_Z swapped.
    """
    import galois

    from qldpc.experimental.surgery.gadget import build_gadget

    H_same = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=np.int_)
    H_comp = np.array([[1, 1, 0, 0, 0], [1, 0, 1, 0, 0]], dtype=np.int_)
    if basis is Pauli.X:
        code = codes.CSSCode(H_same, H_comp, is_subsystem_code=False)
    else:
        code = codes.CSSCode(H_comp, H_same, is_subsystem_code=False)
    # Coordinate 3 forces the second H_same generator in and coordinate 4 forces it out, so x is a
    # logical operator rather than a stabilizer.
    x = np.array([1, 1, 1, 1, 0], dtype=np.uint8)
    g = build_gadget(code, x, basis=basis)

    n, num_support = len(code), int(x.sum())
    data_checks = np.flatnonzero((H_comp[:, x == 1]).any(axis=1))
    assert num_support != len(data_checks)
    F = H_comp[np.ix_(data_checks, np.flatnonzero(x))]
    num_gauge = len(data_checks) - int(np.linalg.matrix_rank(galois.GF2(F.astype(np.int_))))
    H_same_merged, H_comp_merged = (
        (g.HX_merged, g.HZ_merged) if basis is Pauli.X else (g.HZ_merged, g.HX_merged)
    )
    assert H_same_merged.shape == (len(H_same) + num_support, n + len(data_checks))
    assert H_comp_merged.shape == (len(H_comp) + num_gauge, n + len(data_checks))
    assert not ((g.HX_merged @ g.HZ_merged.T) % 2).any()
    ancilla_block = H_comp_merged[: len(H_comp), n:]
    assert np.array_equal(ancilla_block[data_checks].sum(axis=1), np.ones(len(data_checks)))


def test_build_gadget_steane_returns_valid_layout() -> None:
    from qldpc.experimental.surgery.gadget import GadgetLayout, build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    assert isinstance(g, GadgetLayout)
    assert g.code is code
    assert np.array_equal(g.x, x)
    # One κ qubit per row of F, indexed contiguously after the data qubits: the merged matrices
    # carry a column per data qubit plus a column per κ qubit, which is how the circuit layer sizes
    # the ancilla register.
    assert g.HX_merged.shape[1] == code.num_qudits + g.incidence.shape[0]
    assert g.HZ_merged.shape[1] == g.HX_merged.shape[1]


def test_webster_generalised_bicycle_codes_match_appendix_a() -> None:
    """The first two generalised bicycle codes have the [[n, k]] of Webster et al. Appendix A.

    Webster, Smith, Cohen, arXiv:2511.15989 Appendix A lists [[62, 10, 6]] and [[126, 12, 10]].
    """
    for code_index in [0, 1]:
        data = load_webster_seed_set(code_index)
        code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
        assert len(code) == data["n_data_qubits"]
        assert code.dimension == data["k_logical"]


@pytest.mark.parametrize("name,basis", [("X_bar_1", Pauli.X), ("Z_bar_1", Pauli.Z)])
@pytest.mark.parametrize("code_index,n_anc", WEBSTER_TABLE_1_BARE_GADGET_QUBITS)
def test_webster_table_1_bare_gadget_qubits_exact(
    code_index: int, n_anc: int, name: str, basis: PauliXZ
) -> None:
    """Bare-gadget qubit count matches Webster Table I for each of the 4 codes, in both bases.

    Webster's "Gadget Qubits" column counts qubits including ancillae, which for an L=1 gadget is
    |Q'| + |S'_meas| + |S'_comp|: κ ancillas, χ measurement checks, and gauge checks.  For codes 2
    and 3 the table writes that cell as a sum, (49+8) and (79+20), whose second term is the
    Cheeger-boost addition; the bare count checked here is the first term.  Seeds X̄_1 and Z̄_1.
    """
    from qldpc.experimental.surgery.gadget import build_gadget

    data = load_webster_seed_set(code_index)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    logical = _webster_x_bar_operator(data, name, pauli_type=name[0])
    g = build_gadget(code, logical, basis=basis)
    n_ancilla = g.incidence.shape[0]
    n_meas_checks = len(g.support)
    n_comp_checks = g.gauge.shape[0]
    assert n_ancilla + n_meas_checks + n_comp_checks == n_anc


def test_build_gadget_basis_is_required() -> None:
    """basis has no default: a CSS code's X-logical and Z-logical can coincide, as Steane's do."""
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    with pytest.raises(TypeError, match="basis"):
        build_gadget(code, x)  # type: ignore[call-arg]


def test_build_gadget_z_basis_dual_matches_x_basis_on_dual_code() -> None:
    """basis-symmetric invariant: build_gadget on the Z basis matches its dual on the X basis.

    build_gadget(code, z, basis=Z) gives the same merged matrices as build_gadget(dual_code, z,
    basis=X), where dual_code has HX/HZ swapped.  The swap labels swap too, so we compare HX_z vs
    HZ_dx_x and HZ_z vs HX_dx_x.
    """
    from qldpc.codes.common import CSSCode
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_z = build_gadget(code, z, basis=Pauli.Z)
    # Dual code: swap matrix_x and matrix_z
    dual = CSSCode(
        np.asarray(code.matrix_z).astype(np.int_),
        np.asarray(code.matrix_x).astype(np.int_),
        is_subsystem_code=False,
    )
    g_dual = build_gadget(dual, z, basis=Pauli.X)
    # In the dual construction, the basis-X chi rows end up in dual.HX_merged which corresponds to
    # original.HZ_merged in the basis-Z construction.
    assert np.array_equal(g_z.HZ_merged, g_dual.HX_merged), (
        "basis-Z chi (in HZ_merged) should equal basis-X chi (in HX_merged) on dual"
    )
    assert np.array_equal(g_z.HX_merged, g_dual.HZ_merged), (
        "basis-Z gauge-fix (in HX_merged) should equal basis-X gauge-fix (in HZ_merged) on dual"
    )


def test_with_added_ancillas_extends_incidence_and_recomputes_gauge() -> None:
    """Augmenting with one weight-2 row adds a column to merged matrices and recomputes G."""
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    # Pick two ports in V_0; create one extra weight-2 row connecting them
    support_a, support_b = g.support[0], g.support[1]
    extra_incidence = np.zeros((1, len(g.support)), dtype=np.uint8)
    idx_a = g.support.index(support_a)
    idx_b = g.support.index(support_b)
    extra_incidence[0, idx_a] = 1
    extra_incidence[0, idx_b] = 1
    assert not g.is_augmented
    g_aug = g.with_added_ancillas(extra_incidence)

    # incidence_aug = [incidence | extra_incidence] vertically stacked
    assert g_aug.is_augmented
    assert np.array_equal(g_aug.added_ancilla_incidence, extra_incidence)
    assert g_aug.incidence.shape == (g.incidence.shape[0] + 1, g.incidence.shape[1])
    assert np.array_equal(g_aug.incidence[: g.incidence.shape[0]], g.incidence)
    assert np.array_equal(g_aug.incidence[g.incidence.shape[0] :], extra_incidence)
    # HX_merged has one extra column (one extra κ qubit); same number of rows
    assert g_aug.HX_merged.shape == (g.HX_merged.shape[0], g.HX_merged.shape[1] + 1)
    # CSS commutation
    product = (g_aug.HX_merged @ g_aug.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_restrict_checks_to_support_rejects_x_shape_mismatch() -> None:
    """The support restriction validates x.shape == (n,)."""
    from qldpc.experimental.surgery.gadget import _restrict_checks_to_support

    code = codes.SteaneCode()
    bad_x = np.zeros(code.num_qudits + 1, dtype=np.uint8)
    with pytest.raises(ValueError, match="expected"):
        _restrict_checks_to_support(code, bad_x)


def test_build_gadget_rejects_non_logical_input() -> None:
    """build_gadget rejects x that isn't a logical operator support.

    For basis=X: HZ @ x must be 0; for basis=Z: HX @ x must be 0.  Single-qubit support [1,0,0,...]
    generally violates both (it's not in the codespace).
    """
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    bad = np.zeros(code.num_qudits, dtype=np.uint8)
    bad[0] = 1  # Single qubit support — not a logical operator on Steane.
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    assert ((HZ @ bad) % 2).any(), "fixture broken: single-qubit support should not be X-logical"
    assert ((HX @ bad) % 2).any(), "fixture broken: single-qubit support should not be Z-logical"
    with pytest.raises(ValueError, match="logical-X"):
        build_gadget(code, bad, basis=Pauli.X)
    with pytest.raises(ValueError, match="logical-Z"):
        build_gadget(code, bad, basis=Pauli.Z)


def test_build_gadget_rejects_invalid_basis() -> None:
    """build_gadget raises on basis that isn't Pauli.X or Pauli.Z."""
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    with pytest.raises(ValueError, match=r"Pauli\.X or Pauli\.Z"):
        build_gadget(code, x, basis=Pauli.Y)  # type: ignore[arg-type]


def test_with_added_ancillas_rejects_wrong_width() -> None:
    """GadgetLayout.with_added_ancillas rejects rows with the wrong width."""
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    gadget = build_gadget(code, x, basis=Pauli.X)
    # support has 3 columns (Steane X-logical weight 3); pass 2-column incidence_extra.
    bad_extra = np.array([[1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="columns"):
        gadget.with_added_ancillas(bad_extra)


def test_with_added_ancillas_rejects_non_weight_2_rows() -> None:
    """GadgetLayout.with_added_ancillas rejects rows with weight != 2."""
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    gadget = build_gadget(code, x, basis=Pauli.X)
    # Width 3 (Steane X-logical), but a row with weight 1 (not 2)
    bad_extra = np.array([[1, 0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="weight"):
        gadget.with_added_ancillas(bad_extra)


def test_build_gadget_rejects_zero_x() -> None:
    """build_gadget rejects x = 0, which satisfies H @ x == 0 but measures no logical operator.

    Left unchecked it yields an empty support and a 0x0 incidence, for which cheeger_constant
    reports inf -- an arbitrarily good Cheeger constant for a gadget that measures nothing.
    """
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    zero = np.zeros(code.num_qudits, dtype=np.uint8)
    with pytest.raises(ValueError, match="zero vector"):
        build_gadget(code, zero, basis=Pauli.X)


@pytest.mark.parametrize("entry", [2, 257])
def test_build_gadget_rejects_a_non_binary_x(entry: int) -> None:
    """x indicates a support, so an entry outside {0, 1} is rejected rather than reinterpreted.

    The cast to uint8 wraps 257 to 1, which would otherwise pass as a different vector entirely.  An
    even entry like 2 survives the cast and is taken at face value by the check equation, which then
    rejects x for the wrong reason.
    """
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.int_)
    x[np.flatnonzero(x)[0]] = entry
    with pytest.raises(ValueError, match="binary support vector"):
        build_gadget(code, x, basis=Pauli.X)


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_gadget_rejects_a_stabilizer_support(basis: PauliXZ) -> None:
    """A stabilizer of the measured basis satisfies the check equation but measures the identity.

    The surface code is not self-dual, so its measured-basis stabilizers do not lie in the
    complementary basis's row space, and only a guard reading the measured basis rejects them.
    """
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SurfaceCode(3)
    H_same = np.asarray(code.matrix_x if basis is Pauli.X else code.matrix_z).astype(np.uint8)
    with pytest.raises(ValueError, match="row space"):
        build_gadget(code, H_same[0], basis=basis)


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_build_gadget_accepts_a_code_with_no_measured_basis_checks(basis: PauliXZ) -> None:
    """An empty measured basis spans only the zero vector, so no nonzero x is a stabilizer.

    TrivialCode carries no checks in either basis, so every single-qubit Pauli is a logical operator
    and the restriction is a lone support vertex with no ancillas attached.
    """
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.TrivialCode(3)
    x = np.array([1, 0, 0], dtype=np.uint8)
    gadget = build_gadget(code, x, basis=basis)
    assert gadget.support == (0,)
    assert gadget.incidence.shape == (0, 1)
    assert gadget.gauge.shape == (0, 0)


def test_build_gadget_rejects_a_non_qubit_code() -> None:
    """Every stage of the gadget construction is mod 2, so a qudit code is rejected up front."""
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SurfaceCode(2, field=3)
    with pytest.raises(ValueError, match="qubit code"):
        build_gadget(code, np.zeros(len(code), dtype=np.uint8), basis=Pauli.X)


def test_build_gadget_rejects_a_subsystem_code() -> None:
    """Subsystem gauge generators do not define commuting stabilizers for the merged code."""
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.BaconShorCode(3)
    with pytest.raises(ValueError, match=r"only stabilizer \(non-subsystem\) CSS codes"):
        build_gadget(code, np.zeros(len(code), dtype=np.uint8), basis=Pauli.X)


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_single_ppm_merged_code_is_css_and_drops_one_logical(basis: PauliXZ) -> None:
    """The merged code commutes and carries one logical fewer than the data code.

    Measuring one logical operator of the data code fixes it, so the merged code's dimension is
    code.dimension - 1.  Steane has k = 1, so its merged code has k = 0.
    """
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=basis)
    assert g.basis is basis

    product = (g.HX_merged.astype(int) @ g.HZ_merged.astype(int).T) % 2
    assert np.array_equal(product, np.zeros_like(product)), "merged code violates CSS commutation"

    merged = codes.CSSCode(g.HX_merged, g.HZ_merged, is_subsystem_code=False)
    assert merged.dimension == code.dimension - 1


def test_gadget_cone_result_retains_stored_matrices_and_regions() -> None:
    """Derived provenance names the existing public matrix regions without replacing them."""
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    logical = np.asarray(code.get_logical_ops(Pauli.X)[0], dtype=np.uint8)
    gadget = build_gadget(code, logical, basis=Pauli.X)
    result = gadget._get_cone_result()

    assert np.array_equal(result.code.matrix_x, gadget.HX_merged)
    assert np.array_equal(result.code.matrix_z, gadget.HZ_merged)
    assert result.regions.data_qubits == tuple(range(len(code)))
    assert result.regions.ancilla_qubits == tuple(range(len(code), gadget.HX_merged.shape[1]))
    assert result.regions.measurement_checks == tuple(
        range(code.matrix_x.shape[0], gadget.HX_merged.shape[0])
    )


def test_gadget_cone_result_rejects_missing_base_incidence_rows() -> None:
    """Derived provenance fails loudly if a layout loses original ancilla rows."""
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    logical = np.asarray(code.get_logical_ops(Pauli.X)[0], dtype=np.uint8)
    gadget = build_gadget(code, logical, basis=Pauli.X)
    invalid = dataclasses.replace(gadget, incidence=gadget.incidence[:-1])

    with pytest.raises(ValueError, match="fewer rows"):
        invalid._get_cone_result()
