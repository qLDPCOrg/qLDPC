"""Tests for src/qldpc/experimental/surgery/cheeger.py (cheeger_constant + boost_gadget)."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from ._webster_fixture import (
    _webster_x_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)


def test_cheeger_constant_matches_boost_target() -> None:
    """cheeger_constant(g) reports the Webster boundary Cheeger; boost raises it."""
    from qldpc.experimental.surgery import boost_gadget, build_gadget, cheeger_constant

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    h0 = cheeger_constant(g)
    assert h0 >= 0
    # Boosting to a higher target raises h(F) to (at least) that target.
    g_aug = boost_gadget(g, method="combinatorial", target=2.0, max_extra_qubits=30, seed=7)
    h1 = cheeger_constant(g_aug)
    assert h1 >= 2.0 - 1e-9, f"boost to 2.0 produced h={h1}"
    # No-op contract: if h0 already meets target, boost adds no rows.
    g_noop = boost_gadget(g, method="combinatorial", target=h0, max_extra_qubits=30, seed=7)
    assert g_noop.incidence.shape[0] == g.incidence.shape[0], "boost to current h should be a no-op"


def test_boost_gadget_dispatches_to_two_methods() -> None:
    from qldpc.experimental.surgery.cheeger import boost_gadget
    from qldpc.experimental.surgery.gadget import (
        GadgetLayout,
        build_gadget,
    )

    # Use Webster code 0 (l=31, k>=2): Steane gadget has dimension 0 (Steane
    # k=1 minus 1 gadget-consumed logical), which causes the BP+OSD decoder
    # used by boost_gadget_distance to hang searching for nonexistent logicals.
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_operator(data)
    g = build_gadget(code, x, basis=Pauli.X)
    for method in ("combinatorial", "distance"):
        out = boost_gadget(g, method=method, target=1.0, seed=42)
        assert isinstance(out, GadgetLayout), f"method={method}"


def test_boost_gadget_seed_reproducible() -> None:
    from qldpc.experimental.surgery.cheeger import boost_gadget
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    a = boost_gadget(g, method="combinatorial", target=1.0, seed=42)
    b = boost_gadget(g, method="combinatorial", target=1.0, seed=42)
    assert np.array_equal(a.incidence, b.incidence)
    assert np.array_equal(a.HX_merged, b.HX_merged)


@pytest.mark.parametrize("method", ["combinatorial", "distance"])
def test_boost_gadget_preserves_css_commutation(method: str) -> None:
    from qldpc.experimental.surgery.cheeger import boost_gadget
    from qldpc.experimental.surgery.gadget import (
        build_gadget,
    )

    # Webster code 0 — Steane causes distance-boost decoder to hang on k=0 merged.
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_operator(data)
    g = build_gadget(code, x, basis=Pauli.X)
    boosted = boost_gadget(g, method=method, target=1.0, seed=0)
    product = (boosted.HX_merged @ boosted.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_boost_gadget_preserves_css_commutation_both_bases(basis: PauliXZ) -> None:
    """boost_gadget on a basis=X or basis=Z gadget preserves CSS commutation."""
    from qldpc.experimental.surgery.cheeger import boost_gadget
    from qldpc.experimental.surgery.gadget import (
        build_gadget,
    )

    from ._webster_fixture import _webster_z_bar_operator

    d = load_webster_seed_set(0)
    c = build_generalised_bicycle_code(d["l"], d["A"], d["B"])
    if basis is Pauli.X:
        op = _webster_x_bar_operator(d, "X_bar_1")
    else:
        op = _webster_z_bar_operator(d, "Z_bar_1")
    g = build_gadget(c, op, basis=basis)
    boosted = boost_gadget(g, method="combinatorial", target=1.0, seed=0)
    product = (boosted.HX_merged @ boosted.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))
    assert boosted.basis is basis  # boost preserves basis


def test_boost_gadget_combinatorial_basis_z_preserves_chi_carrier() -> None:
    """After basis=Z combinatorial boost, χ rows must live in HZ_merged.

    The legacy adapter handled basis=Z by swapping HX↔HZ on entry and back on
    exit; the GadgetLayout-native path delegates basis routing to
    build_gadget_augmented. This test catches a regression where χ rows end
    up in HX_merged instead of HZ_merged.

    Distance-strategy basis=Z is not tested here because the Webster JSON
    fixture only ships X̄ operators; the basis=X path of distance boost is
    covered by test_boost_gadget_preserves_css_commutation[distance].
    """
    from qldpc.experimental.surgery.cheeger import boost_gadget
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z_op = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z_op, basis=Pauli.Z)

    boosted = boost_gadget(g, method="combinatorial", target=1.0, seed=42)

    assert boosted.basis is Pauli.Z, (
        f"basis dropped through boost: got {boosted.basis!r}, expected Pauli.Z"
    )
    n_meas_checks = len(boosted.support)
    n_z_data = code.matrix_z.shape[0]
    chi_block = boosted.HZ_merged[n_z_data : n_z_data + n_meas_checks, :]
    assert chi_block.any(), (
        "χ rows missing from HZ_merged; basis=Z boost path likely swapped HX/HZ."
    )


def test_boost_combinatorial_above_initial_h_enters_loop_body() -> None:
    """Webster code 0 has h(F)=1; boosting to target=2.0 forces the augmentation
    loop to run (adds rows; cheeger constant increases)."""
    from qldpc.experimental.surgery import boost_gadget, build_gadget, cheeger_constant

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_operator(data)
    g = build_gadget(code, x, basis=Pauli.X)
    h0 = cheeger_constant(g)
    boosted = boost_gadget(g, method="combinatorial", target=h0 + 1.0, seed=0)
    assert boosted.incidence.shape[0] > g.incidence.shape[0], (
        f"boost target={h0 + 1.0} should add rows; got {boosted.incidence.shape[0]} == bare"
    )
    h_new = cheeger_constant(boosted)
    assert h_new >= h0 + 1.0 - 1e-9, f"boost target unmet: {h_new} < {h0 + 1.0}"


def test_boost_combinatorial_rejects_non_positive_target_h() -> None:
    """boost_gadget_cheeger_combinatorial rejects target_h <= 0."""
    from qldpc.experimental.surgery.cheeger import boost_gadget_cheeger_combinatorial

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    from qldpc.experimental.surgery import build_gadget

    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="target_h must be positive"):
        boost_gadget_cheeger_combinatorial(g, target_h=0.0)


def test_boost_combinatorial_rejects_negative_max_extra_qubits() -> None:
    """boost_gadget_cheeger_combinatorial rejects max_extra_qubits < 0."""
    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.cheeger import boost_gadget_cheeger_combinatorial

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="max_extra_qubits must be >= 0"):
        boost_gadget_cheeger_combinatorial(g, target_h=1.0, max_extra_qubits=-1)


def test_boost_distance_rejects_non_positive_target_distance() -> None:
    """boost_gadget_distance rejects target_distance <= 0."""
    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.cheeger import boost_gadget_distance

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="target_distance must be positive"):
        boost_gadget_distance(g, target_distance=0)


def test_boost_distance_rejects_negative_max_extra_qubits() -> None:
    """boost_gadget_distance rejects max_extra_qubits < 0."""
    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.cheeger import boost_gadget_distance

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="max_extra_qubits must be >= 0"):
        boost_gadget_distance(g, target_distance=2, max_extra_qubits=-1)


def test_boost_gadget_rejects_unknown_method() -> None:
    """boost_gadget(method='bogus') raises ValueError."""
    from qldpc.experimental.surgery import boost_gadget, build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="unknown method"):
        boost_gadget(g, method="bogus", target=1.0)


def test_exact_boundary_cheeger_n_V_below_2_returns_inf() -> None:
    """_exact_boundary_cheeger on a 1-column F returns (inf, [0])."""
    import galois

    from qldpc.experimental.surgery.cheeger import _exact_boundary_cheeger

    F = galois.GF(2)(np.array([[1]], dtype=np.int_))
    h, v_star = _exact_boundary_cheeger(F)
    assert h == float("inf")
    assert v_star.shape == (1,)
    assert int(v_star[0]) == 0


def test_exact_boundary_cheeger_rejects_n_V_above_26() -> None:
    """_exact_boundary_cheeger raises on |V| > 26 (would explode subset enumeration)."""
    import galois

    from qldpc.experimental.surgery.cheeger import _exact_boundary_cheeger

    F = galois.GF(2)(np.zeros((2, 27), dtype=np.int_))
    with pytest.raises(ValueError, match="requires \\|V\\| ≤ 26"):
        _exact_boundary_cheeger(F)


def test_spectral_cheeger_lower_bound_matches_lambda2_over_2() -> None:
    """_spectral_cheeger_lower_bound returns lambda_2(F F^T) / 2 for the given F."""
    import galois

    from qldpc.experimental.surgery.cheeger import _spectral_cheeger_lower_bound

    F = galois.GF(2)(np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int_))
    h = _spectral_cheeger_lower_bound(F)
    F_arr = np.asarray(F).astype(np.float64)
    expected_lambda2 = float(np.linalg.eigvalsh(F_arr @ F_arr.T)[1])
    assert abs(h - expected_lambda2 / 2.0) < 1e-9


def test_spectral_cheeger_lower_bound_degenerate_returns_zero() -> None:
    """_spectral_cheeger_lower_bound on a single-row F returns 0.0."""
    import galois

    from qldpc.experimental.surgery.cheeger import _spectral_cheeger_lower_bound

    F = galois.GF(2)(np.array([[1, 1, 0]], dtype=np.int_))
    assert _spectral_cheeger_lower_bound(F) == 0.0


def test_cheeger_constant_dispatches_to_spectral_for_n_V_above_26() -> None:
    """cheeger_constant uses _spectral_cheeger_lower_bound when |V_0| > 26."""
    import dataclasses

    from qldpc.experimental.surgery import build_gadget, cheeger_constant

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    # Synthesize a gadget with wide incidence (n_V = 27) to force the spectral path.
    # We don't run the merged code through validation here — we only check the dispatch
    # branch in cheeger_constant.
    wide_incidence = np.zeros((2, 27), dtype=np.uint8)
    wide_incidence[0, 0] = 1
    wide_incidence[0, 1] = 1
    wide_incidence[1, 0] = 1
    wide_incidence[1, 2] = 1
    g_wide = dataclasses.replace(g, incidence=wide_incidence)
    h = cheeger_constant(g_wide)
    assert h >= 0


def test_augment_incidence_with_random_edges_adds_rows_disjoint_from_existing() -> None:
    """_augment_incidence_with_random_edges adds degree-2 rows whose endpoint
    pairs are not already present in the base incidence."""
    from qldpc.experimental.surgery.cheeger import _augment_incidence_with_random_edges

    base = np.zeros((1, 5), dtype=np.int_)
    base[0, 0] = 1
    base[0, 1] = 1  # existing pair (0, 1)
    rng = np.random.default_rng(42)
    out = _augment_incidence_with_random_edges(base, n_new_edges=2, rng=rng)
    assert out is not None
    assert out.shape[0] == 3  # 1 base + 2 added
    new_rows = out[1:]
    for row in new_rows:
        ones = np.flatnonzero(row).tolist()
        assert len(ones) == 2, f"expected weight-2 row, got {row}"
        assert tuple(sorted(ones)) != (0, 1), "augmenter must skip already-present pairs"


def test_augment_incidence_with_random_edges_returns_none_when_too_few_columns() -> None:
    """Returns None if n_X < 2 (no valid degree-2 row exists)."""
    from qldpc.experimental.surgery.cheeger import _augment_incidence_with_random_edges

    base = np.zeros((1, 1), dtype=np.int_)
    out = _augment_incidence_with_random_edges(base, n_new_edges=1, rng=np.random.default_rng(0))
    assert out is None


def test_augment_incidence_with_random_edges_returns_base_when_no_new_edges_requested() -> None:
    """Returns base incidence unchanged if n_new_edges == 0."""
    from qldpc.experimental.surgery.cheeger import _augment_incidence_with_random_edges

    base = np.zeros((1, 3), dtype=np.int_)
    base[0, 0] = 1
    base[0, 2] = 1
    out = _augment_incidence_with_random_edges(base, n_new_edges=0, rng=np.random.default_rng(0))
    assert out is not None
    assert np.array_equal(out, base)


def test_augment_incidence_with_random_edges_returns_none_when_no_fresh_pair() -> None:
    """When all degree-2 pairs are already covered, the sampler exhausts and returns None."""
    from qldpc.experimental.surgery.cheeger import _augment_incidence_with_random_edges

    # 2 columns: only pair is (0,1), already present.
    base = np.zeros((1, 2), dtype=np.int_)
    base[0, 0] = 1
    base[0, 1] = 1
    out = _augment_incidence_with_random_edges(base, n_new_edges=1, rng=np.random.default_rng(0))
    assert out is None


def test_boost_combinatorial_rejects_synthetic_n_V_above_26() -> None:
    """Combinatorial boost raises on synthetic |V_0| > 26 (subset enumeration infeasible)."""
    import dataclasses

    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.cheeger import boost_gadget_cheeger_combinatorial

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    wide_incidence = np.zeros((2, 27), dtype=np.uint8)
    g_wide = dataclasses.replace(g, incidence=wide_incidence)
    with pytest.raises(ValueError, match="enumeration infeasible"):
        boost_gadget_cheeger_combinatorial(g_wide, target_h=1.0)


def test_boost_combinatorial_max_extra_qubits_saturation_returns_partial_augment() -> None:
    """When boost can't reach target_h within max_extra_qubits, it stops early
    and returns a partially-augmented gadget. Webster0 (h0=1) with target=10
    saturates at max_extra=2."""
    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.cheeger import boost_gadget_cheeger_combinatorial

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_operator(data)
    g = build_gadget(code, x, basis=Pauli.X)
    boosted = boost_gadget_cheeger_combinatorial(g, target_h=10.0, max_extra_qubits=2, seed=0)
    n_added = boosted.incidence.shape[0] - g.incidence.shape[0]
    assert n_added <= 2, f"expected at most 2 added rows, got {n_added}"
