"""L=1 gadget construction (Webster, Smith, Cohen arXiv:2511.15989 §II.A).

Three explicit named steps that map 1:1 to the paper:
    _step1_restriction  — Webster §II.A step 1 (restriction)
    _step2_gauge_fix    — Webster §II.A step 2 (gauge fix)
    _step3_assemble     — Webster §II.A step 3 (block assembly)

Notation (used throughout the surgery package; the symbols follow
Webster/Cohen/Cross, not Cain). For a logical measured on support V_0:
    V_0  — logical support (measured qubits)                 → ``support``
    C_0  — data-code checks touching V_0                     → ``data_checks``
    F    — restriction (incidence) matrix on (C_0, V_0)      → ``incidence``
    κ    — gadget ancilla qubits (one per touched check)     → ``ancilla_qubits``
    χ    — added meas-basis checks (one per support qubit)
    G    — a basis of ker(F^T); the gadget gauge checks      → ``gauge``
"""

from __future__ import annotations

import dataclasses

import galois
import numpy as np

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

GF2 = galois.GF(2)


@dataclasses.dataclass(frozen=True, eq=False)
class GadgetLayout:
    code: CSSCode
    x: np.ndarray
    support: tuple[int, ...]
    data_checks: tuple[int, ...]
    incidence: np.ndarray
    gauge: np.ndarray
    HX_merged: np.ndarray
    HZ_merged: np.ndarray
    ancilla_qubits: tuple[int, ...]
    basis: PauliXZ


def _step1_restriction(
    code: CSSCode,
    x: np.ndarray,
    *,
    basis: PauliXZ = Pauli.X,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """Webster §II.A step 1 — V_0 = supp(x); C_0 = checks touching V_0; F = H_complement[C_0, V_0].

    gadget notation: V_0 → support; C_0 → data_checks; F → incidence.

    For basis=Pauli.X: incidence = H_Z[data_checks, support] (the complementary
    basis to the measured logical). For basis=Pauli.Z: incidence = H_X[data_checks, support].
    """
    x = np.asarray(x).astype(np.uint8)
    if x.shape != (code.num_qudits,):
        raise ValueError(f"x has shape {x.shape}, expected ({code.num_qudits},)")
    support = tuple(int(i) for i in np.where(x)[0])
    # Use the COMPLEMENTARY check matrix to the measured logical type
    H_complement = (
        np.asarray(code.matrix_z).astype(np.uint8)
        if basis is Pauli.X
        else np.asarray(code.matrix_x).astype(np.uint8)
    )
    data_checks = tuple(
        int(j) for j in range(H_complement.shape[0]) if H_complement[j, list(support)].any()
    )
    incidence = (
        H_complement[np.ix_(data_checks, support)]
        if data_checks and support
        else np.zeros((len(data_checks), len(support)), dtype=np.uint8)
    )
    return support, data_checks, incidence.astype(np.uint8)


def _step2_gauge_fix(incidence: np.ndarray) -> np.ndarray:
    """Webster §II.A step 2 — G whose rows form a canonical basis of ker(F.T) over GF(2).

    gadget notation: F → incidence; G → gauge.

    Uses galois ``left_null_space`` (row-reduced) so the basis is deterministic.
    """
    if incidence.size == 0:
        return np.zeros((0, incidence.shape[0]), dtype=np.uint8)
    gauge = GF2(incidence.astype(np.int_).tolist()).left_null_space()
    return np.asarray(gauge).astype(np.uint8)


def _assemble_HX_L1(
    HX_data: np.ndarray,
    support_indices: np.ndarray,
    incidence: np.ndarray,
) -> np.ndarray:
    """L=1 HX-side block assembly: [[HX_data, 0], [E_V0, F^T]] over GF(2).

    gadget notation: V_0 → support; F → incidence.

    Used by _step3_assemble (initial gadget assembly) and
    build_gadget_augmented (post-boost rebuild). The Z-side
    assembly is NOT shared — the boost rebuild treats new κ' qubits as
    pure-gauge (no data-Z extension), unlike the initial assembly.

    Args:
        HX_data: original code's X-check matrix, shape (mX, n), uint8.
        support_indices: indices of V_0 within the n data qubits, shape (|V_0|,).
        incidence: restriction matrix, shape (|C_0|, |V_0|), uint8.

    Returns:
        HX_merged: shape (mX + |V_0|, n + |C_0|), uint8.
    """
    mX, n = HX_data.shape
    n_v0, n_c0 = int(incidence.shape[1]), int(incidence.shape[0])
    n_merged = n + n_c0
    top = np.hstack([HX_data, np.zeros((mX, n_c0), dtype=np.uint8)]).astype(np.uint8)
    bot = np.zeros((n_v0, n_merged), dtype=np.uint8)
    bot[np.arange(n_v0), np.asarray(support_indices)] = 1
    bot[:, n:] = incidence.T
    return np.vstack([top, bot]).astype(np.uint8)


def _step3_assemble(
    code: CSSCode,
    support: tuple[int, ...],
    data_checks: tuple[int, ...],
    incidence: np.ndarray,
    gauge: np.ndarray,
    *,
    basis: PauliXZ = Pauli.X,
) -> tuple[np.ndarray, np.ndarray]:
    """Webster §II.A step 3 — block assembly of HX_merged, HZ_merged.

    gadget notation: χ → S'_meas (meas-basis ancilla rows); G → gauge.

    basis=X (default): χ rows added to HX_merged, G to HZ_merged.
    basis=Z: χ rows added to HZ_merged, G to HX_merged (basis-symmetric dual).
    """
    HX = np.asarray(code.matrix_x).astype(np.uint8)
    HZ = np.asarray(code.matrix_z).astype(np.uint8)
    n = code.num_qudits
    mX, mZ = HX.shape[0], HZ.shape[0]
    nC = len(data_checks)
    r = gauge.shape[0]

    # incidence_tilde : (mZ_or_mX × nC) selection matrix — incidence_tilde[j, k] = 1 iff j == C_0[k]
    if basis is Pauli.X:
        incidence_tilde = np.zeros((mZ, nC), dtype=np.uint8)
    else:
        incidence_tilde = np.zeros((mX, nC), dtype=np.uint8)
    for k, j in enumerate(data_checks):
        if j < 0:
            continue  # sentinel for extra-κ rows from build_gadget_augmented
        incidence_tilde[j, k] = 1

    support_arr = np.asarray(support, dtype=np.int_)

    if basis is Pauli.X:
        # χ rows extend HX_merged; G rows extend HZ_merged
        HX_merged = _assemble_HX_L1(HX, support_arr, incidence)
        HZ_merged = np.block(
            [
                [HZ, incidence_tilde],
                [np.zeros((r, n), dtype=np.uint8), gauge.astype(np.uint8)],
            ]
        ).astype(np.uint8)
    else:
        # basis=Z (symmetric dual): χ rows extend HZ_merged; G rows extend HX_merged
        HZ_merged = _assemble_HX_L1(HZ, support_arr, incidence)
        HX_merged = np.block(
            [
                [HX, incidence_tilde],
                [np.zeros((r, n), dtype=np.uint8), gauge.astype(np.uint8)],
            ]
        ).astype(np.uint8)

    return HX_merged, HZ_merged


def build_gadget(
    code: CSSCode,
    x: np.ndarray,
    *,
    basis: PauliXZ,
) -> GadgetLayout:
    """Webster L=1 gadget = steps 1+2+3 composed. Deterministic in (code, x, basis).

    gadget notation: κ qubits → ancilla_qubits; G → gauge.

    basis=Pauli.X: measures a logical X (PPM of X̄). Validates H_Z @ x == 0.
    basis=Pauli.Z: measures a logical Z (PPM of Z̄). Validates H_X @ x == 0.
    """
    x = np.asarray(x).astype(np.uint8)
    if basis is Pauli.X:
        H_check = np.asarray(code.matrix_z).astype(np.uint8)
        if ((H_check @ x) % 2).any():
            raise ValueError("x is not a logical-X support (H_Z @ x != 0).")
    elif basis is Pauli.Z:
        H_check = np.asarray(code.matrix_x).astype(np.uint8)
        if ((H_check @ x) % 2).any():
            raise ValueError("x is not a logical-Z support (H_X @ x != 0).")
    else:
        raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {basis!r}")

    support, data_checks, incidence = _step1_restriction(code, x, basis=basis)
    gauge = _step2_gauge_fix(incidence)
    HX_m, HZ_m = _step3_assemble(code, support, data_checks, incidence, gauge, basis=basis)
    ancilla_qubits = tuple(range(code.num_qudits, code.num_qudits + len(data_checks)))
    return GadgetLayout(
        code=code,
        x=x,
        support=support,
        data_checks=data_checks,
        incidence=incidence,
        gauge=gauge,
        HX_merged=HX_m,
        HZ_merged=HZ_m,
        ancilla_qubits=ancilla_qubits,
        basis=basis,
    )


def build_gadget_augmented(
    code: CSSCode,
    x: np.ndarray,
    incidence_extra: np.ndarray,
    *,
    basis: PauliXZ,
) -> GadgetLayout:
    """Rebuild a GadgetLayout with incidence augmented by extra weight-2 rows.

    gadget notation: F → incidence; F_extra → incidence_extra; κ → ancilla qubits.

    Each row of ``incidence_extra`` has weight 2 and corresponds to a new κ qubit not
    backed by any original Z-check (basis=X) or X-check (basis=Z). The function:

    1. Stacks incidence_aug = [incidence; incidence_extra].
    2. Recomputes G_aug = ker(incidence_aug^T) via _step2_gauge_fix.
    3. Calls _step3_assemble with the original V_0 / C_0 plus the new κ rows.
       The extra columns of tilde_F are all zero (no original check sits on the
       new κ qubits).

    The returned ``GadgetLayout.data_checks`` and ``ancilla_qubits`` are extended to cover
    the new κ qubits; the new κ indices come after the original ones.
    """
    x = np.asarray(x).astype(np.uint8)
    support, data_checks, incidence = _step1_restriction(code, x, basis=basis)
    incidence_extra = np.asarray(incidence_extra).astype(np.uint8)
    if incidence_extra.shape[1] != len(support):
        raise ValueError(
            f"incidence_extra has {incidence_extra.shape[1]} columns; expected {len(support)} (= |support|)"
        )
    if incidence_extra.size and not np.all(incidence_extra.sum(axis=1) == 2):
        bad = np.flatnonzero(incidence_extra.sum(axis=1) != 2).tolist()
        raise ValueError(f"incidence_extra rows {bad} have weight != 2; required weight 2.")

    incidence_aug = np.vstack([incidence, incidence_extra]).astype(np.uint8)
    gauge_aug = _step2_gauge_fix(incidence_aug)

    # _step3_assemble computes tilde_F by indexing into C_0; we need an extended
    # C_0_aug that has the new rows as sentinels (their tilde_F columns must be 0).
    # Trick: pass C_0_aug = C_0 + (-1, -1, ...) sentinels which fall outside [0, mZ),
    # so the tilde_F loop sets nothing for those positions.
    n_extra = incidence_extra.shape[0]
    data_checks_aug = tuple(data_checks) + tuple([-1] * n_extra)
    HX_aug, HZ_aug = _step3_assemble(
        code,
        support,
        data_checks_aug,
        incidence_aug,
        gauge_aug,
        basis=basis,
    )
    ancilla_qubits_aug = tuple(range(code.num_qudits, code.num_qudits + len(data_checks_aug)))
    return GadgetLayout(
        code=code,
        x=x,
        support=support,
        data_checks=data_checks_aug,
        incidence=incidence_aug,
        gauge=gauge_aug,
        HX_merged=HX_aug,
        HZ_merged=HZ_aug,
        ancilla_qubits=ancilla_qubits_aug,
        basis=basis,
    )
