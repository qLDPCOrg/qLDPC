"""Tests for src/qldpc/experimental/surgery/cheeger.py (cheeger_constant + boost_gadget).

Copyright 2026 The qLDPC Authors

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

import numpy as np
import pytest

from qldpc import codes
from qldpc.codes.common import CSSCode
from qldpc.experimental.surgery.gadget import GadgetLayout
from qldpc.objects import Pauli, PauliXZ

from .conftest import (
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
    # used by _boost_gadget_distance to hang searching for nonexistent logicals.
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

    from .conftest import _webster_z_bar_operator

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

    _build_gadget_augmented carries the basis through the rebuild, so this pins that χ rows land in
    HZ_merged and not HX_merged. Scope is the combinatorial path; the distance path's basis=X is
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
    """Webster code 0 has h(F)=1; boosting to target=2.0 forces the augmentation loop to run.

    Adds rows; cheeger constant increases.
    """
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
    """_boost_gadget_cheeger_combinatorial rejects target_h <= 0."""
    from qldpc.experimental.surgery.cheeger import _boost_gadget_cheeger_combinatorial

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    from qldpc.experimental.surgery import build_gadget

    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="target_h must be positive"):
        _boost_gadget_cheeger_combinatorial(g, target_h=0.0)


def test_boost_combinatorial_rejects_negative_max_extra_qubits() -> None:
    """_boost_gadget_cheeger_combinatorial rejects max_extra_qubits < 0."""
    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.cheeger import _boost_gadget_cheeger_combinatorial

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="max_extra_qubits must be >= 0"):
        _boost_gadget_cheeger_combinatorial(g, target_h=1.0, max_extra_qubits=-1)


def test_boost_distance_rejects_non_positive_target_distance() -> None:
    """_boost_gadget_distance rejects target_distance <= 0."""
    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.cheeger import _boost_gadget_distance

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="target_distance must be positive"):
        _boost_gadget_distance(g, target_distance=0)


def test_boost_distance_rejects_negative_max_extra_qubits() -> None:
    """_boost_gadget_distance rejects max_extra_qubits < 0."""
    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.cheeger import _boost_gadget_distance

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="max_extra_qubits must be >= 0"):
        _boost_gadget_distance(g, target_distance=2, max_extra_qubits=-1)


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

    F = galois.GF2(np.array([[1]], dtype=np.int_))
    h, v_star = _exact_boundary_cheeger(F)
    assert h == float("inf")
    assert v_star.shape == (1,)
    assert int(v_star[0]) == 0


def test_exact_boundary_cheeger_rejects_n_V_above_26() -> None:
    """_exact_boundary_cheeger raises on |V| > 26 (would explode subset enumeration)."""
    import galois

    from qldpc.experimental.surgery.cheeger import _exact_boundary_cheeger

    F = galois.GF2(np.zeros((2, 27), dtype=np.int_))
    with pytest.raises(ValueError, match="requires \\|V\\| ≤ 26"):
        _exact_boundary_cheeger(F)


def test_cheeger_constant_raises_for_n_V_above_26() -> None:
    """cheeger_constant raises for |V_0| > 26 instead of returning an unsound value."""
    import dataclasses

    from qldpc.experimental.surgery import build_gadget, cheeger_constant

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    # Synthesize a gadget with wide incidence (n_V = 27) to exceed the exact-enumeration limit;
    # cheeger_constant must raise rather than substitute an approximation for h.
    wide_incidence = np.zeros((2, 27), dtype=np.uint8)
    wide_incidence[0, 0] = 1
    wide_incidence[0, 1] = 1
    wide_incidence[1, 0] = 1
    wide_incidence[1, 2] = 1
    g_wide = dataclasses.replace(g, incidence=wide_incidence)
    with pytest.raises(ValueError, match="requires \\|V_0\\| ≤ 26 for an exact value"):
        cheeger_constant(g_wide)


def test_augment_incidence_with_random_edges_adds_rows_disjoint_from_existing() -> None:
    """_augment_incidence_with_random_edges adds degree-2 rows with new endpoint pairs.

    The endpoint pairs are not already present in the base incidence.
    """
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
    from qldpc.experimental.surgery.cheeger import _boost_gadget_cheeger_combinatorial

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    wide_incidence = np.zeros((2, 27), dtype=np.uint8)
    g_wide = dataclasses.replace(g, incidence=wide_incidence)
    with pytest.raises(ValueError, match="enumeration infeasible"):
        _boost_gadget_cheeger_combinatorial(g_wide, target_h=1.0)


def test_boost_combinatorial_raises_when_target_unreachable_in_budget() -> None:
    """When boost can't reach target_h within max_extra_qubits, it raises RuntimeError.

    It raises rather than silently returning an under-target (distance-degraded) gadget. Webster0
    (h0=1) with target=10 cannot be reached in max_extra=2.
    """
    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.cheeger import _boost_gadget_cheeger_combinatorial

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_operator(data)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(RuntimeError, match="could not reach target_h"):
        _boost_gadget_cheeger_combinatorial(g, target_h=10.0, max_extra_qubits=2, seed=0)


def _webster0_gadget() -> GadgetLayout:
    """Build the Webster code-0 basis=X gadget shared by the distance-boost tests."""
    from qldpc.experimental.surgery import build_gadget

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    return build_gadget(code, _webster_x_bar_operator(data), basis=Pauli.X)


def _stub_distance_bound(monkeypatch: pytest.MonkeyPatch, bounds: list[int]) -> None:
    """Replace the BP+OSD distance bound with a scripted sequence whose last value repeats."""
    calls = [0]

    def _bound(self: CSSCode, pauli: PauliXZ, **kwargs: object) -> int:
        index = min(calls[0], len(bounds) - 1)
        calls[0] += 1
        return bounds[index]

    monkeypatch.setattr(CSSCode, "get_distance_bound_with_decoder", _bound)


def test_boost_distance_raises_when_target_unreachable_in_budget() -> None:
    """_boost_gadget_distance raises RuntimeError when nothing clears the BP+OSD screen.

    A returned gadget has always met target_distance, so exhausting the budget is an error rather
    than a result. Webster0 cannot reach target_distance=999.
    """
    from qldpc.experimental.surgery.cheeger import _boost_gadget_distance

    with pytest.raises(RuntimeError, match="could not reach target_distance"):
        _boost_gadget_distance(
            _webster0_gadget(),
            target_distance=999,
            max_extra_qubits=2,
            num_trials_per_step=1,
            decoder_trials=2,
            seed=0,
        )


def test_boost_distance_returns_an_augmentation_that_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_boost_gadget_distance returns an augmented candidate that clears the screen.

    The bound is scripted so the bare gadget fails and the next candidate passes, which exercises
    the accept path without paying for BP+OSD.
    """
    from qldpc.experimental.surgery.cheeger import _boost_gadget_distance

    g = _webster0_gadget()
    _stub_distance_bound(monkeypatch, [0, 999])
    boosted = _boost_gadget_distance(
        g, target_distance=5, max_extra_qubits=1, num_trials_per_step=1, seed=0
    )
    assert boosted.incidence.shape[0] > g.incidence.shape[0]


def test_boost_distance_skips_unusable_augmentation_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sample yielding no fresh degree-2 rows is skipped, and the search moves to the next sample.

    The first sample returns None and the second delegates to the real sampler, so a boosted gadget
    comes back only if the unusable sample is skipped rather than ending the search.
    """
    from qldpc.experimental.surgery import cheeger as cheeger_module

    real_augment = cheeger_module._augment_incidence_with_random_edges
    calls = [0]

    def _augment(
        incidence_base: np.ndarray, n_new_edges: int, rng: np.random.Generator
    ) -> np.ndarray | None:
        calls[0] += 1
        if calls[0] == 1:
            return None
        return real_augment(incidence_base, n_new_edges, rng)

    g = _webster0_gadget()
    _stub_distance_bound(monkeypatch, [0, 999])
    monkeypatch.setattr(cheeger_module, "_augment_incidence_with_random_edges", _augment)
    boosted = cheeger_module._boost_gadget_distance(
        g,
        target_distance=5,
        max_extra_qubits=1,
        num_trials_per_step=2,
        seed=0,
    )
    assert calls[0] >= 2, "the unusable first sample must not end the search"
    assert boosted.incidence.shape[0] > g.incidence.shape[0]


def test_boost_distance_skips_augmentation_that_fails_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An augmentation rejected by _build_gadget_augmented's validation is skipped, not raised."""
    from qldpc.experimental.surgery import cheeger as cheeger_module
    from qldpc.experimental.surgery import gadget as gadget_module

    real_build = gadget_module._build_gadget_augmented

    def _build(
        code: CSSCode, x: np.ndarray, incidence_extra: np.ndarray, *, basis: PauliXZ = Pauli.X
    ) -> GadgetLayout:
        if incidence_extra.shape[0] == 0:
            return real_build(code, x, incidence_extra, basis=basis)
        raise ValueError("incidence_extra rows have weight != 2; required weight 2.")

    _stub_distance_bound(monkeypatch, [0])
    monkeypatch.setattr(gadget_module, "_build_gadget_augmented", _build)
    with pytest.raises(RuntimeError, match="could not reach target_distance"):
        cheeger_module._boost_gadget_distance(
            _webster0_gadget(),
            target_distance=5,
            max_extra_qubits=1,
            num_trials_per_step=1,
            seed=0,
        )


def test_boost_combinatorial_single_column_incidence_is_a_no_op() -> None:
    """A weight-1 support gives a single-column F, which admits no cut, so boost adds nothing."""
    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.cheeger import _boost_gadget_cheeger_combinatorial

    # A distance-1 code has a weight-1 logical X, which reaches |V_0| = 1.
    code = codes.CSSCode([[1, 1]], np.zeros((0, 2), dtype=int))
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    assert g.incidence.shape[1] == 1, f"expected |V_0| = 1, got {g.incidence.shape}"

    out = _boost_gadget_cheeger_combinatorial(g, target_h=1.0)
    assert out.incidence.shape == g.incidence.shape, "single-column boost must add no rows"


def test_boost_combinatorial_stops_when_every_cut_edge_is_already_a_row() -> None:
    """The greedy search ends when no degree-2 row can raise |∂v*|.

    Steane's F is the weight-2 complement of the identity on 3 columns, so all three cut pairs are
    already present. The budget is left generous to pin that the pair supply, not the budget, is
    what stops the search: h is reported unchanged at its bare value of 2.0.
    """
    from qldpc.experimental.surgery import build_gadget
    from qldpc.experimental.surgery.cheeger import _boost_gadget_cheeger_combinatorial

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)

    with pytest.raises(RuntimeError, match=r"reached h=2\.0"):
        _boost_gadget_cheeger_combinatorial(g, target_h=10.0, max_extra_qubits=50, seed=0)


def test_boost_distance_screen_consults_the_z_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """A candidate clearing the X screen but failing the Z screen is rejected.

    The bare gadget is scripted to pass on X and fail on Z with no augmentation budget, so the only
    way to reach the screen's verdict is through the Z bound.
    """
    from qldpc.experimental.surgery.cheeger import _boost_gadget_distance

    _stub_distance_bound(monkeypatch, [999, 0])
    with pytest.raises(RuntimeError, match="could not reach target_distance"):
        _boost_gadget_distance(
            _webster0_gadget(),
            target_distance=5,
            max_extra_qubits=0,
            num_trials_per_step=1,
            seed=0,
        )
