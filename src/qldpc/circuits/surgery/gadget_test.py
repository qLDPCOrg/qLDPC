"""Tests for src/qldpc/circuits/surgery/gadget.py."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli

from ._webster_fixture import (
    _webster_x_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)

WEBSTER_TABLE_I_ANCILLA_MEAS_COMP = [(0, 19), (1, 31), (2, 49), (3, 79)]


def test_gadget_layout_is_frozen_dataclass() -> None:
    from qldpc.circuits.surgery.gadget import GadgetLayout

    assert dataclasses.is_dataclass(GadgetLayout)
    # frozen
    fields = {f.name for f in dataclasses.fields(GadgetLayout)}
    assert fields == {
        "code",
        "x",
        "support",
        "data_checks",
        "incidence",
        "gauge",
        "HX_merged",
        "HZ_merged",
        "ancilla_qubits",
        "basis",
    }
    # Verify actually frozen: mutation must raise. None placeholders are fine here
    # — we only check FrozenInstanceError, never read the fields.
    inst = GadgetLayout(
        code=None,  # type: ignore[arg-type]
        x=None,  # type: ignore[arg-type]
        support=(),
        data_checks=(),
        incidence=None,  # type: ignore[arg-type]
        gauge=None,  # type: ignore[arg-type]
        HX_merged=None,  # type: ignore[arg-type]
        HZ_merged=None,  # type: ignore[arg-type]
        ancilla_qubits=(),
        basis=Pauli.X,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        inst.code = object()  # type: ignore[misc,assignment]


def test_step1_restriction_steane() -> None:
    from qldpc.circuits.surgery.gadget import _step1_restriction

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    support, data_checks, incidence = _step1_restriction(code, x)
    # V_0 = supp(x), sorted ascending
    assert support == tuple(int(i) for i in np.where(x)[0])
    assert list(support) == sorted(support)
    # C_0 = Z-checks touching V_0, sorted ascending
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    touched = sorted({j for j in range(HZ.shape[0]) for i in support if HZ[j, i] == 1})
    assert data_checks == tuple(touched)
    assert list(data_checks) == sorted(data_checks)
    # F = H_Z[C_0, V_0]
    assert incidence.shape == (len(data_checks), len(support))
    assert np.array_equal(incidence, HZ[np.ix_(data_checks, support)])
    # F @ 1_{V0} == 0 (Webster §II.A step 1 invariant)
    ones = np.ones(len(support), dtype=np.uint8)
    assert np.array_equal((incidence @ ones) % 2, np.zeros(len(data_checks), dtype=np.uint8))


def test_step2_gauge_fix_basis_property() -> None:
    from qldpc.circuits.surgery.gadget import _step1_restriction, _step2_gauge_fix

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    _, _, incidence = _step1_restriction(code, x)
    gauge = _step2_gauge_fix(incidence)
    # Webster §II.A step 2: G F = 0 over GF(2)
    assert gauge.shape[1] == incidence.shape[0]
    GF = (gauge @ incidence) % 2
    assert np.array_equal(GF, np.zeros_like(GF))
    # rank(G) = |C_0| - rank(F)
    import galois

    r_expected = incidence.shape[0] - int(np.linalg.matrix_rank(galois.GF(2)(incidence.tolist())))
    assert gauge.shape[0] == r_expected


def test_step2_gauge_fix_deterministic() -> None:
    """Same F twice → byte-identical G (non-trivial: rank-deficient F → non-empty G)."""
    from qldpc.circuits.surgery.gadget import _step2_gauge_fix

    # 3x3 matrix with rank 2 (row 0 + row 1 = row 2 over GF(2)), so G has 1 row.
    incidence = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    gauge1 = _step2_gauge_fix(incidence)
    gauge2 = _step2_gauge_fix(incidence)
    assert gauge1.shape == (1, 3), f"expected G shape (1,3), got {gauge1.shape}"
    assert np.array_equal(gauge1, gauge2)
    # And sanity-check the basis property holds on this F too.
    assert np.array_equal(
        (gauge1 @ incidence) % 2, np.zeros((1, incidence.shape[1]), dtype=np.uint8)
    )


def test_step3_assemble_basis_z_places_chi_in_HZ_merged_and_G_in_HX_merged() -> None:
    """basis=Pauli.Z: χ rows added to HZ_merged (Z-type); G added to HX_merged (X-type)."""
    from qldpc.circuits.surgery.gadget import (
        _step1_restriction,
        _step2_gauge_fix,
        _step3_assemble,
    )

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    support, data_checks, incidence = _step1_restriction(code, z, basis=Pauli.Z)
    gauge = _step2_gauge_fix(incidence)
    HX_m, HZ_m = _step3_assemble(code, support, data_checks, incidence, gauge, basis=Pauli.Z)

    n, mX, mZ = code.num_qudits, code.matrix_x.shape[0], code.matrix_z.shape[0]
    # For basis=Z: HX_merged grows by r rows (gauge-fix), HZ_merged by |V_0| rows (chi).
    assert HX_m.shape == (mX + gauge.shape[0], n + len(data_checks)), f"HX shape {HX_m.shape}"
    assert HZ_m.shape == (mZ + len(support), n + len(data_checks)), f"HZ shape {HZ_m.shape}"
    # CSS commutation
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_step3_assemble_steane_css_commutes() -> None:
    from qldpc.circuits.surgery.gadget import (
        _step1_restriction,
        _step2_gauge_fix,
        _step3_assemble,
    )

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    support, data_checks, incidence = _step1_restriction(code, x)
    gauge = _step2_gauge_fix(incidence)
    HX_m, HZ_m = _step3_assemble(code, support, data_checks, incidence, gauge)

    n, mX, mZ = code.num_qudits, code.matrix_x.shape[0], code.matrix_z.shape[0]
    assert HX_m.shape == (mX + len(support), n + len(data_checks))
    assert HZ_m.shape == (mZ + gauge.shape[0], n + len(data_checks))
    # Webster §II.A: H_X^merged @ H_Z^merged.T == 0 over GF(2) (CSS commutation)
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_step3_assemble_csscode_with_distinct_nV_nC() -> None:
    """Synthetic CSS code where nV != nC — catches F_tilde shape bug.

    Uses a 5-qubit CSS code with k=1, picking a logical-X representative
    whose support size (nV=4) differs from the number of Z-checks it
    touches (nC=2). With the buggy F_tilde[j] = F[k] form, numpy raises
    ValueError because F[k] has shape (nV=4,) but the row width is (nC=2).
    The fix (F_tilde[j, k] = 1) is the correct indicator/selection matrix.

    Verifies:
    1. CSS commutation: HX_merged @ HZ_merged.T == 0 over GF(2).
    2. Indicator form: each Z-check in data_checks attaches to EXACTLY ONE
       ancilla (row-sum == 1 in the ancilla block).
    """
    from qldpc.circuits.surgery.gadget import (
        _step1_restriction,
        _step2_gauge_fix,
        _step3_assemble,
    )

    # 5-qubit CSS code (k=1):
    #   HX = [[1,1,1,0,0],[0,0,0,1,1]]
    #   HZ = [[1,1,0,0,0],[1,0,1,0,0]]
    # Commutativity check (each pair of rows):
    #   row0(HX)·row0(HZ) = 1+1+0+0+0 = 0 mod 2 ✓
    #   row0(HX)·row1(HZ) = 1+0+1+0+0 = 0 mod 2 ✓
    #   row1(HX)·row0(HZ) = 0+0+0+0+0 = 0 mod 2 ✓
    #   row1(HX)·row1(HZ) = 0+0+0+0+0 = 0 mod 2 ✓
    HX_raw = np.array([[1, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=np.uint8)
    HZ_raw = np.array([[1, 1, 0, 0, 0], [1, 0, 1, 0, 0]], dtype=np.uint8)
    assert np.array_equal((HX_raw @ HZ_raw.T) % 2, np.zeros((2, 2), dtype=np.uint8)), (
        "CSS sanity failed"
    )

    code = codes.CSSCode(HX_raw, HZ_raw)  # type: ignore[arg-type]

    # Logical X rep: x = [1,1,1,1,0].
    #   HZ @ x = [1+1+0,1+0+1] = [0,0] mod 2  =>  x in ker(HZ) ✓
    #   row(HX) = span{[1,1,1,0,0],[0,0,0,1,1]}: cannot produce [1,1,1,1,0]
    #   because the last coord would require b=0 while 4th coord requires b=1 ✓ logical
    x_logical = np.array([1, 1, 1, 1, 0], dtype=np.uint8)
    assert np.array_equal((HZ_raw @ x_logical) % 2, np.zeros(2, dtype=np.uint8)), (
        "x_logical not in ker(HZ)"
    )

    support, data_checks, incidence = _step1_restriction(code, x_logical)
    # V0 = {0,1,2,3} (nV=4); HZ row0 touches {0,1}, HZ row1 touches {0,2} -> data_checks=(0,1) (nC=2)
    assert len(support) != len(data_checks), (
        f"nV={len(support)} == nC={len(data_checks)}: this test requires nV != nC to catch the bug"
    )

    gauge = _step2_gauge_fix(incidence)
    HX_m, HZ_m = _step3_assemble(code, support, data_checks, incidence, gauge)

    # 1. CSS commutation
    product = (HX_m @ HZ_m.T) % 2
    assert np.array_equal(product, np.zeros_like(product)), (
        "CSS commutation failed: HX_merged @ HZ_merged.T != 0"
    )

    # 2. Indicator form: each Z-check j in data_checks should attach to exactly
    #    one ancilla (column-slice after n data qubits in HZ_merged).
    n = code.num_qudits
    mZ = HZ_raw.shape[0]
    HZ_ancilla_block = HZ_m[:mZ, n:]
    for k, j in enumerate(data_checks):
        row_sum = int(HZ_ancilla_block[j].sum())
        assert row_sum == 1, (
            f"row j={j} of HZ ancilla-block should have exactly 1 one (indicator form), "
            f"got {row_sum} — F_tilde indicator form violated"
        )


def test_build_gadget_steane_returns_valid_layout() -> None:
    from qldpc.circuits.surgery.gadget import GadgetLayout, build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    assert isinstance(g, GadgetLayout)
    assert g.code is code
    assert np.array_equal(g.x, x)
    # κ qubits indexed contiguously after data qubits
    assert g.ancilla_qubits == tuple(range(code.num_qudits, code.num_qudits + len(g.data_checks)))


def test_build_gadget_deterministic() -> None:
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g1 = build_gadget(code, x, basis=Pauli.X)
    g2 = build_gadget(code, x, basis=Pauli.X)
    assert g1.support == g2.support
    assert g1.data_checks == g2.data_checks
    assert np.array_equal(g1.incidence, g2.incidence)
    assert np.array_equal(g1.gauge, g2.gauge)
    assert np.array_equal(g1.HX_merged, g2.HX_merged)
    assert np.array_equal(g1.HZ_merged, g2.HZ_merged)
    assert g1.ancilla_qubits == g2.ancilla_qubits


def test_build_gadget_rejects_non_x_logical() -> None:
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.zeros(code.num_qudits, dtype=np.uint8)
    x[0] = 1  # not a logical X (HZ @ x ≠ 0 in general)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    if ((HZ @ x) % 2).any():
        with pytest.raises(ValueError, match="logical"):
            build_gadget(code, x, basis=Pauli.X)


def test_load_webster_seed_set_returns_known_shape() -> None:
    data = load_webster_seed_set(0)
    assert "l" in data and "A" in data and "B" in data
    assert "seeds" in data


def test_build_generalised_bicycle_code_constructs_css() -> None:
    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    assert code.num_qudits == 2 * data["l"]
    # CSS commutation
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    assert np.array_equal((HX @ HZ.T) % 2, np.zeros((HX.shape[0], HZ.shape[0]), dtype=np.uint8))


@pytest.mark.parametrize("code_index,n_anc", WEBSTER_TABLE_I_ANCILLA_MEAS_COMP)
def test_webster_table_i_ancilla_meas_comp_exact(code_index: int, n_anc: int) -> None:
    """Webster Table I in Cain notation: |Q'| + |S'_meas| + |S'_comp| matches
    each of the 4 generalised-bicycle codes. Reproduces Webster Table I exactly."""
    from qldpc.circuits.surgery.gadget import (
        build_gadget,
    )

    data = load_webster_seed_set(code_index)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x1 = _webster_x_bar_operator(data)
    g1 = build_gadget(code, x1, basis=Pauli.X)
    n_ancilla = len(g1.ancilla_qubits)
    n_meas_checks = int(g1.x.sum())  # |support|
    n_comp_checks = g1.gauge.shape[0]
    assert n_ancilla + n_meas_checks + n_comp_checks == n_anc, (
        f"code {code_index}: |Q'|={n_ancilla}, |S'_meas|={n_meas_checks}, |S'_comp|={n_comp_checks}, "
        f"sum={n_ancilla + n_meas_checks + n_comp_checks}, expected {n_anc}"
    )


def test_build_gadget_basis_is_required() -> None:
    """basis has no default: a CSS code's X-logical and Z-logical can coincide
    (e.g. self-dual Steane), so the caller must declare intent explicitly.
    """
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    with pytest.raises(TypeError, match="basis"):
        build_gadget(code, x)  # type: ignore[call-arg]


def test_step1_restriction_basis_z_uses_HX() -> None:
    """For basis=Pauli.Z, F = H_X[C_0, V_0] (not H_Z)."""
    from qldpc.circuits.surgery.gadget import _step1_restriction

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    support, data_checks, incidence = _step1_restriction(code, z, basis=Pauli.Z)
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    # V_0 = supp(z)
    assert support == tuple(int(i) for i in np.where(z)[0])
    # C_0 = X-checks touching V_0
    touched = sorted({j for j in range(HX.shape[0]) for i in support if HX[j, i] == 1})
    assert data_checks == tuple(touched)
    # F = H_X[C_0, V_0]
    assert np.array_equal(incidence, HX[np.ix_(data_checks, support)])
    # Webster §II.A step 1 invariant: F @ 1_{V0} = 0 (since H_X @ z = 0 for a logical Z)
    ones = np.ones(len(support), dtype=np.uint8)
    assert np.array_equal((incidence @ ones) % 2, np.zeros(len(data_checks), dtype=np.uint8))


def test_build_gadget_z_basis_css_commutation() -> None:
    """build_gadget(code, z_logical, basis=Pauli.Z) yields a CSS-commuting merged code."""
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g = build_gadget(code, z, basis=Pauli.Z)
    assert g.basis is Pauli.Z
    product = (g.HX_merged @ g.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_build_gadget_z_basis_dual_matches_x_basis_on_dual_code() -> None:
    """basis-symmetric invariant: build_gadget(code, z, basis=Z) gives the same
    merged matrices as build_gadget(dual_code, z, basis=X), where dual_code has
    HX/HZ swapped. The swap labels swap too, so we compare HX_z vs HZ_dx_x and
    HZ_z vs HX_dx_x."""
    from qldpc.circuits.surgery.gadget import build_gadget
    from qldpc.codes.common import CSSCode

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
    # In the dual construction, the basis-X chi rows end up in dual.HX_merged
    # which corresponds to original.HZ_merged in the basis-Z construction.
    assert np.array_equal(g_z.HZ_merged, g_dual.HX_merged), (
        "basis-Z chi (in HZ_merged) should equal basis-X chi (in HX_merged) on dual"
    )
    assert np.array_equal(g_z.HX_merged, g_dual.HZ_merged), (
        "basis-Z gauge-fix (in HX_merged) should equal basis-X gauge-fix (in HZ_merged) on dual"
    )


def test_webster_table_i_z_basis_ancilla_meas_comp_exact() -> None:
    """Webster Z̄_1 seed in Cain notation: |Q'| + |S'_meas| + |S'_comp| matches
    (basis-symmetric dual; reproduces Webster Table I)."""
    from qldpc.circuits.surgery.gadget import (
        build_gadget,
    )

    from ._webster_fixture import _webster_z_bar_operator

    for code_index, expected in [(0, 19), (1, 31), (2, 49), (3, 79)]:
        d = load_webster_seed_set(code_index)
        c = build_generalised_bicycle_code(d["l"], d["A"], d["B"])
        z = _webster_z_bar_operator(d)
        g = build_gadget(c, z, basis=Pauli.Z)
        n_ancilla = len(g.ancilla_qubits)
        n_meas_checks = len(g.support)
        n_comp_checks = g.gauge.shape[0]
        assert n_ancilla + n_meas_checks + n_comp_checks == expected, (
            f"code {code_index}: Z-basis got |Q'|+|S'_meas|+|S'_comp|={n_ancilla + n_meas_checks + n_comp_checks}, expected {expected}"
        )


def test_build_gadget_augmented_extends_incidence_and_recomputes_gauge() -> None:
    """Augmenting with one weight-2 row adds a column to merged matrices and recomputes G."""
    from qldpc.circuits.surgery.gadget import build_gadget, build_gadget_augmented

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
    g_aug = build_gadget_augmented(code, x, extra_incidence, basis=Pauli.X)

    # incidence_aug = [incidence | extra_incidence] vertically stacked
    assert g_aug.incidence.shape == (g.incidence.shape[0] + 1, g.incidence.shape[1])
    assert np.array_equal(g_aug.incidence[: g.incidence.shape[0]], g.incidence)
    assert np.array_equal(g_aug.incidence[g.incidence.shape[0] :], extra_incidence)
    # HX_merged has one extra column (one extra κ qubit); same number of rows
    assert g_aug.HX_merged.shape == (g.HX_merged.shape[0], g.HX_merged.shape[1] + 1)
    # CSS commutation
    product = (g_aug.HX_merged @ g_aug.HZ_merged.T) % 2
    assert np.array_equal(product, np.zeros_like(product))


def test_step2_gauge_fix_rows_linearly_independent() -> None:
    """G rows from _step2_gauge_fix are linearly independent over GF(2).

    Webster §II.A step 3 requires |S_L| - wt(L) + 1 INDEPENDENT gauge
    constraints. The existing test verifies G @ F == 0 (i.e. G is in
    ker(F.T)) but not that G has full row rank.

    A degenerate F could let the gauge fix return redundant rows,
    inflating g.gauge.shape[0] without changing the actual gauge structure.
    The Cain Table III bb_18 G=20 reproduction would catch the final
    count but not the underlying rank degeneracy.
    """
    import galois
    import sympy

    from qldpc.circuits.surgery.gadget import build_gadget

    F2 = galois.GF(2)
    xs, ys = sympy.symbols("x y")

    cases: list[tuple[str, codes.CSSCode, np.ndarray]] = []

    # Case 1: Steane
    steane = codes.SteaneCode()
    x_steane = np.asarray(steane.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    cases.append(("Steane", steane, x_steane))

    # Case 2: Webster GB code 0
    data = load_webster_seed_set(0)
    webster = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x_webster = _webster_x_bar_operator(data, "X_bar_1")
    cases.append(("Webster GB 0", webster, x_webster))

    # Case 3: Cain bb_18 (cached Z̄ support — same as notebook §3.2)
    bb18 = codes.BBCode(
        (31, 4),
        1 + xs**6 * ys + xs**27,
        ys**2 + xs**15 * ys**3 + xs**24,
    )
    # Use the same cached wt-20 Z̄ rep used by the notebook §3.2 cell to
    # exercise the largest realistic gauge-fix case (G=20 rows). Treat
    # via swap (matrix_z ↔ matrix_x) so vec_20 acts as the X̄ on
    # target_code (matches notebook usage).
    z_bar_support = [
        8,
        9,
        14,
        18,
        24,
        34,
        40,
        56,
        75,
        76,
        97,
        111,
        122,
        171,
        202,
        208,
        213,
        218,
        228,
        238,
    ]
    from qldpc.codes.common import CSSCode

    vec_20 = np.zeros(bb18.num_qudits, dtype=np.uint8)
    vec_20[z_bar_support] = 1
    bb18_swapped = CSSCode(
        bb18.matrix_z,
        bb18.matrix_x,
        is_subsystem_code=False,
    )
    cases.append(("Cain bb_18 (swapped, wt-20)", bb18_swapped, vec_20))

    for label, code, seed_op in cases:
        g = build_gadget(code, seed_op, basis=Pauli.X)
        gauge = g.gauge
        # All three fixture cases have non-empty G in practice (Steane G is 1×3,
        # Webster's growing with code size); the row-rank invariant is what's
        # interesting. Empty-G is exercised by test_step3_assemble_steane_css_commutes
        # via _step2_gauge_fix on a synthetic full-rank F.
        assert gauge.shape[0] > 0, f"{label}: expected G to be non-empty in this fixture set"
        rank = int(np.linalg.matrix_rank(F2(gauge.astype(np.uint8).tolist())))
        assert rank == gauge.shape[0], (
            f"{label}: gauge-fix G has {gauge.shape[0]} rows but rank only "
            f"{rank}. _step2_gauge_fix returned redundant rows on this F."
        )
        # Re-assert the existing G @ F == 0 invariant alongside.
        # (G is a basis of ker(F.T), i.e. G F = 0 over GF(2);
        # see gadget._step2_gauge_fix and existing test_step2_gauge_fix.)
        incidence_mat = g.incidence.astype(np.uint8)
        commute = (gauge.astype(np.uint8) @ incidence_mat) % 2
        assert not commute.any(), f"{label}: G @ F != 0 (gauge-fix output failed commutation)."


def test_step1_restriction_rejects_x_shape_mismatch() -> None:
    """gadget._step1_restriction validates x.shape == (n,)."""
    from qldpc.circuits.surgery.gadget import _step1_restriction

    code = codes.SteaneCode()
    bad_x = np.zeros(code.num_qudits + 1, dtype=np.uint8)
    with pytest.raises(ValueError, match="expected"):
        _step1_restriction(code, bad_x)


def test_step2_gauge_fix_empty_incidence_returns_zero_rows() -> None:
    """_step2_gauge_fix on size-0 incidence returns shape (0, 0) gauge."""
    from qldpc.circuits.surgery.gadget import _step2_gauge_fix

    incidence = np.zeros((0, 0), dtype=np.uint8)
    gauge = _step2_gauge_fix(incidence)
    assert gauge.shape == (0, 0)


def test_build_gadget_rejects_non_logical_input() -> None:
    """build_gadget rejects x that isn't a logical operator support.

    For basis=X: HZ @ x must be 0; for basis=Z: HX @ x must be 0. Single-qubit
    support [1,0,0,...] generally violates both (it's not in the codespace).
    """
    from qldpc.circuits.surgery.gadget import build_gadget

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
    from qldpc.circuits.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    with pytest.raises(ValueError, match="basis must be"):
        build_gadget(code, x, basis=Pauli.Y)  # type: ignore[arg-type]


def test_build_gadget_augmented_rejects_wrong_width() -> None:
    """build_gadget_augmented rejects incidence_extra with wrong column count."""
    from qldpc.circuits.surgery.gadget import build_gadget_augmented

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    # support has 3 columns (Steane X-logical weight 3); pass 2-column incidence_extra.
    bad_extra = np.array([[1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="columns"):
        build_gadget_augmented(code, x, bad_extra, basis=Pauli.X)


def test_build_gadget_augmented_rejects_non_weight_2_rows() -> None:
    """build_gadget_augmented rejects incidence_extra rows with weight != 2."""
    from qldpc.circuits.surgery.gadget import build_gadget_augmented

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    # Width 3 (Steane X-logical), but a row with weight 1 (not 2)
    bad_extra = np.array([[1, 0, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match="weight"):
        build_gadget_augmented(code, x, bad_extra, basis=Pauli.X)
