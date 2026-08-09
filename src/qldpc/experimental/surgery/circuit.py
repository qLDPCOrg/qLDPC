"""Stim surgery circuit construction (single-PPM and joint-PPM).

References:
    Cain et al. arXiv:2603.28627 §B.1  — single-PPM measurement protocol.
    Webster, Smith, Cohen arXiv:2511.15989  — gadget Eq. 1 observable.
"""

from __future__ import annotations

import numpy as np
import stim

from qldpc.circuits.bookkeeping import DetectorRecord, MeasurementRecord, QubitIDs
from qldpc.circuits.memory.syndrome_measurement import EdgeColoring
from qldpc.circuits.noise_model import NoiseModel
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli

from .bridge import Bridge
from .gadget import GadgetLayout


def _gadget_merged_csscode(g: GadgetLayout) -> CSSCode:
    return CSSCode(
        g.HX_merged.astype(np.int_),
        g.HZ_merged.astype(np.int_),
        is_subsystem_code=False,
    )


def keep_only_observable(circuit: stim.Circuit, keep_idx: int) -> stim.Circuit:
    """Return a copy of ``circuit`` with all OBSERVABLE_INCLUDE entries dropped
    except the one whose first argument equals ``keep_idx``. Recurses into
    REPEAT blocks so observables inside loops are filtered the same way.

    For surgery PPM circuits, pass ``keep_idx=0`` to retain only obs0
    (Webster Eq. 1, the physical syndrome-based readout). obs1 is an
    implementation cross-check that directly measures the data on V_0 and
    is NOT part of any physical protocol — keeping it for an LER run would
    sample the wrong distribution.

    Useful for sinter LER sweeps that compare one observable against a
    memory-experiment baseline — sinter expects exactly one observable per task.
    """
    out = stim.Circuit()
    for op in circuit:
        if isinstance(op, stim.CircuitRepeatBlock):
            out.append(
                stim.CircuitRepeatBlock(
                    op.repeat_count,
                    keep_only_observable(op.body_copy(), keep_idx),
                )
            )
            continue
        if op.name == "OBSERVABLE_INCLUDE" and int(op.gate_args_copy()[0]) != keep_idx:
            continue
        out.append(op)
    return out


def logical_state_init(code: CSSCode, state: str, *, log_idx: int) -> str:
    """Per-qubit ``data_init`` string preparing a Pauli logical state on
    logical qubit ``log_idx`` of a CSS code.

    ``state`` ∈ {"0", "1", "+", "-"}:
      * "0" → ``"0" * n``  — |0⟩^n projects to |0⟩_L^{⊗k} for any CSS code
      * "1" → "1" on supp(X̄_{log_idx}), "0" elsewhere — flips logical qubit
        ``log_idx`` from |0⟩_L to |1⟩_L; other logical qubits stay at |0⟩_L
      * "+" → ``"+" * n``  — |+⟩^n projects to |+⟩_L^{⊗k} for any CSS code
      * "-" → "-" on supp(Z̄_{log_idx}), "+" elsewhere — flips logical qubit
        ``log_idx`` from |+⟩_L to |-⟩_L; other logical qubits stay at |+⟩_L

    X̄_{log_idx} and Z̄_{log_idx} are taken from
    ``code.get_logical_ops(Pauli.X)[log_idx]`` and ``[Pauli.Z][log_idx]``;
    qldpc guarantees they form an anti-commuting symplectic pair on that
    logical qubit, so the prep is correct for ANY CSS code regardless of
    the parity of wt(X̄) / wt(Z̄). Naive broadcast ``data_init = "1" * n``
    is correct only when those weights are odd, and silently produces the
    wrong logical state on codes where they are even (e.g. BBCode [[36, 8]]
    with wt(Z̄_0) = 8).

    ``log_idx`` is REQUIRED (keyword-only, no default) — there is no
    universally "right" logical qubit choice on a k>1 code, so the
    caller must declare intent explicitly. Even for state="0" / "+"
    (which physically broadcast and don't depend on log_idx), supplying
    log_idx makes the targeted logical qubit unambiguous in the call
    site. To get a meaningful PPM truth-table check, ``log_idx`` MUST
    match the logical qubit chosen for the gadget's measured Z̄ (or X̄)
    — i.e. the gadget's seed operator should be
    ``code.get_logical_ops(Pauli.Z)[log_idx]`` (or ``[Pauli.X]`` for
    basis=X). The helper does NOT verify this; if indices disagree the
    prep targets a logical qubit that the gadget doesn't measure, and
    the obs0 outcome is silently random.

    The returned string has length ``code.num_qudits``. Plug it straight
    into ``build_single_ppm_circuit(..., data_init=...)`` or wrap with a
    tuple for ``build_joint_ppm_circuit(..., data_init=(s_l, s_r))``.

    Raises
    ------
    ValueError
        If ``state`` is not one of "0", "1", "+", "-".
    IndexError
        If ``log_idx`` is out of range for ``code.dimension``.
    """
    if state not in ("0", "1", "+", "-"):
        raise ValueError(f"state must be one of '0', '1', '+', '-'; got {state!r}")
    if not 0 <= log_idx < code.dimension:
        raise IndexError(f"log_idx={log_idx} out of range for code with dimension={code.dimension}")
    n = code.num_qudits
    if state in ("0", "+"):
        return state * n
    if state == "1":
        flip = np.asarray(code.get_logical_ops(Pauli.X)[log_idx]).astype(np.uint8)
        flip_char, base_char = "1", "0"
    else:  # state == "-"
        flip = np.asarray(code.get_logical_ops(Pauli.Z)[log_idx]).astype(np.uint8)
        flip_char, base_char = "-", "+"
    return "".join(flip_char if flip[i] else base_char for i in range(n))


def _surgery_qubit_coordinates(
    gadget: GadgetLayout,
    qubit_ids: QubitIDs,
    *,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
) -> stim.Circuit:
    """Emit QUBIT_COORDS in surgery's 6/7-lane semantic layout.

    gadget notation: V_0 → support; κ ancillas → ancilla qubits (Q');
    χ ancillas → S'_meas ancillas (= χ rows); G ancillas → S'_comp ancillas (= G rows).

    Lanes:
      y=0  data qubits         (originally data + κ + bridge in qubit_ids.data
                                slot; we split them across y=0/1/6 here).
      y=1  ancilla qubits (Q')
      y=2  data H_X ancillas   (checks_x[:m_X])
      y=3  S'_meas ancillas (= χ rows)
                               (basis=X: checks_x[m_X:]; basis=Z: checks_z[m_Z:])
      y=4  data H_Z ancillas   (checks_z[:m_Z])
      y=5  S'_comp ancillas (= G rows)
                               (basis=X: checks_z[m_Z:]; basis=Z: checks_x[m_X:])
      y=6  bridge data + bridge cycle ancillas (joint PPM only)

    For basis=X, y is monotonic in qubit ID order (ids 0..6→y=0, 7..9→y=1,
    10..12→y=2, 13..15→y=3, 16..18→y=4, 19→y=5), so QUBIT_COORDS lines in
    the stringified circuit dump appear in increasing y order. basis=Z
    breaks monotonicity because χ and G swap matrix slots, but the lane
    numbers remain stable: S'_meas always y=3, S'_comp always y=5.

    `joint=None` → single PPM. Otherwise pass (g_r, bridge, intercode).
    """
    circuit = stim.Circuit()

    if joint is None:
        g_l = gadget
        g_r = None
        bridge = None
        intercode = False
    else:
        g_l = gadget
        g_r, bridge, intercode = joint

    # Sizes for left side (always present).
    n_l = g_l.code.num_qudits
    m_X_l = g_l.code.matrix_x.shape[0]
    m_Z_l = g_l.code.matrix_z.shape[0]
    n_meas_l = len(g_l.support)
    n_gauge_l = g_l.gauge.shape[0]
    k_l = len(g_l.ancilla_qubits)

    # Sizes for right side (joint+intercode only — intracode shares data).
    if joint is not None and intercode:
        assert g_r is not None
        n_r = g_r.code.num_qudits
        m_X_r = g_r.code.matrix_x.shape[0]
        m_Z_r = g_r.code.matrix_z.shape[0]
        n_meas_r = len(g_r.support)
        n_gauge_r = g_r.gauge.shape[0]
    elif joint is not None:  # intracode: data shared, ancillas separate per gadget
        assert g_r is not None
        n_r = 0
        m_X_r = m_Z_r = 0  # data checks not duplicated for intracode
        n_meas_r = len(g_r.support)
        n_gauge_r = g_r.gauge.shape[0]
    else:
        n_r = 0
        m_X_r = m_Z_r = 0
        n_meas_r = n_gauge_r = k_r = 0

    # For joint PPM, the in-circuit κ count is the augmented value (bridge may
    # have added κ' ancillas during cellulation); use the bridge's augmented
    # gadgets as the source of truth.
    if joint is not None:
        assert bridge is not None
        k_l = bridge.g_l_aug.incidence.shape[0]
        k_r = bridge.g_r_aug.incidence.shape[0]

    n_data_total = n_l + n_r
    w = bridge.width if joint is not None and bridge is not None else 0

    # y=0 data
    for i in range(n_data_total):
        circuit.append("QUBIT_COORDS", qubit_ids.data[i], (i, 0))

    # y=1 κ
    for i in range(k_l + k_r):
        circuit.append("QUBIT_COORDS", qubit_ids.data[n_data_total + i], (i, 1))

    # y=6 bridge data (joint PPM only)
    for i in range(w):
        circuit.append(
            "QUBIT_COORDS",
            qubit_ids.data[n_data_total + k_l + k_r + i],
            (i, 6),
        )

    # X-check ancillas: data H_X on y=2, then either χ on y=3 (basis=X) or G on y=5 (basis=Z).
    is_basis_x = g_l.basis is Pauli.X
    m_X_total = m_X_l + m_X_r
    n_meas_total = n_meas_l + n_meas_r
    n_gauge_total = n_gauge_l + n_gauge_r

    for i in range(m_X_total):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_x[i], (i, 2))
    if is_basis_x:
        # χ rows on y=3 (within checks_x)
        for i in range(n_meas_total):
            circuit.append(
                "QUBIT_COORDS",
                qubit_ids.checks_x[m_X_total + i],
                (i, 3),
            )
    else:
        # G rows on y=5 (within checks_x for basis=Z)
        for i in range(n_gauge_total):
            circuit.append(
                "QUBIT_COORDS",
                qubit_ids.checks_x[m_X_total + i],
                (i, 5),
            )

    # Z-check ancillas: data H_Z on y=4, then either G on y=5 (basis=X) or χ on y=3 (basis=Z).
    m_Z_total = m_Z_l + m_Z_r
    for i in range(m_Z_total):
        circuit.append("QUBIT_COORDS", qubit_ids.checks_z[i], (i, 4))
    if is_basis_x:
        for i in range(n_gauge_total):
            circuit.append(
                "QUBIT_COORDS",
                qubit_ids.checks_z[m_Z_total + i],
                (i, 5),
            )
    else:
        for i in range(n_meas_total):
            circuit.append(
                "QUBIT_COORDS",
                qubit_ids.checks_z[m_Z_total + i],
                (i, 3),
            )

    # Joint PPM: bridge cycle ancillas on y=6 (sharing the row with bridge data).
    if joint is not None and w > 1:
        # The new cycle checks live at the end of checks_x (basis=Z) or
        # checks_z (basis=X). They're (w - 1) of them.
        if is_basis_x:
            cycle_check_ids = qubit_ids.checks_z[m_Z_total + n_gauge_total :]
        else:
            cycle_check_ids = qubit_ids.checks_x[m_X_total + n_gauge_total :]
        for i, cid in enumerate(cycle_check_ids):
            circuit.append("QUBIT_COORDS", cid, (i, 6))

    return circuit


def _check_lane_index_map(
    gadget: GadgetLayout,
    qubit_ids: QubitIDs,
    *,
    joint: tuple[GadgetLayout, Bridge, bool] | None = None,
) -> dict[int, tuple[int, int]]:
    """Build a {check_id: (lane, idx)} map matching the QUBIT_COORDS layout.

    Lanes for checks (idx is x position within lane):
      lane=2: data H_X check ancillas (checks_x[:m_X_total])
      lane=3: χ check ancillas (basis=X: checks_x[m_X:]; basis=Z: checks_z[m_Z:])
      lane=4: data H_Z check ancillas (checks_z[:m_Z_total])
      lane=5: G check ancillas (basis=X: checks_z[m_Z:]; basis=Z: checks_x[m_X:])
      lane=6: bridge cycle check ancillas (joint PPM only).
    """
    is_basis_x = gadget.basis is Pauli.X

    if joint is None:
        m_X_total = gadget.code.matrix_x.shape[0]
        m_Z_total = gadget.code.matrix_z.shape[0]
        n_meas_total = len(gadget.support)
        n_gauge_total = gadget.gauge.shape[0]
    else:
        g_r, _bridge, intercode = joint
        m_X_total = gadget.code.matrix_x.shape[0]
        m_Z_total = gadget.code.matrix_z.shape[0]
        if intercode:
            m_X_total += g_r.code.matrix_x.shape[0]
            m_Z_total += g_r.code.matrix_z.shape[0]
        n_meas_total = len(gadget.support) + len(g_r.support)
        n_gauge_total = gadget.gauge.shape[0] + g_r.gauge.shape[0]

    result: dict[int, tuple[int, int]] = {}

    # data H_X on lane=2
    for i in range(m_X_total):
        result[qubit_ids.checks_x[i]] = (2, i)
    # data H_Z on lane=4
    for i in range(m_Z_total):
        result[qubit_ids.checks_z[i]] = (4, i)

    if is_basis_x:
        # χ on lane=3 in checks_x[m_X:]; G on lane=5 in checks_z[m_Z:]
        for i in range(n_meas_total):
            result[qubit_ids.checks_x[m_X_total + i]] = (3, i)
        for i in range(n_gauge_total):
            result[qubit_ids.checks_z[m_Z_total + i]] = (5, i)
    else:
        # G on lane=5 in checks_x[m_X:]; χ on lane=3 in checks_z[m_Z:]
        for i in range(n_gauge_total):
            result[qubit_ids.checks_x[m_X_total + i]] = (5, i)
        for i in range(n_meas_total):
            result[qubit_ids.checks_z[m_Z_total + i]] = (3, i)

    # Joint PPM bridge cycle ancillas on lane=6.
    if joint is not None:
        if is_basis_x:
            cycle_ids = qubit_ids.checks_z[m_Z_total + n_gauge_total :]
        else:
            cycle_ids = qubit_ids.checks_x[m_X_total + n_gauge_total :]
        for i, cid in enumerate(cycle_ids):
            result[cid] = (6, i)

    return result


def build_single_ppm_circuit(
    gadget: GadgetLayout,
    *,
    rounds: int,
    noise_model: NoiseModel | None = None,
    data_init: str | None = None,
) -> stim.Circuit:
    """Single-PPM measurement circuit for `gadget`, assembled per Cain et al. arXiv:2603.28627.

    Emits two OBSERVABLE_INCLUDE entries (see ``_surgery_observable`` for
    full semantics):

      * obs0 — Single-round Z̄ = ∏_{v ∈ support} A_v readout (Webster, Smith,
        Cohen arXiv:2511.15989 §II.A, gadget Eq. 1). XOR of the **last** QEC
        round's meas-check outcomes. The repeated rounds give FT distance
        via the detector layer. Reading the eigenvalue at the final round
        should be decoding-equivalent to Cain et al.'s first-cycle readout
        (arXiv:2603.28627 App. D): the interface detectors telescope, so the
        observable's fault distance should be unchanged.
      * obs1 — Direct destructive M on ``support`` qubits; noiseless
        cross-check, not a physical protocol.

    For LER / noisy runs, use ``keep_only_observable(circuit, keep_idx=0)``.

    ``data_init`` (optional): per-data-qubit init override; see
    ``_surgery_state_prep`` for the character-to-state mapping.
    """
    merged_code = _gadget_merged_csscode(gadget)
    qubit_ids = QubitIDs.from_code(merged_code)
    n_data = gadget.code.num_qudits
    data_ids = qubit_ids.data[:n_data]
    ancilla_ids = qubit_ids.data[n_data:]
    bridge_ids: tuple[int, ...] = ()

    circuit = _surgery_qubit_coordinates(gadget, qubit_ids)
    circuit += _surgery_state_prep(
        gadget,
        data_ids,
        ancilla_ids,
        bridge_ids,
        data_init=data_init,
    )
    qec_cycle, measurement_record, _ = _surgery_qec_cycle(
        gadget,
        merged_code,
        num_rounds=rounds,
        qubit_ids=qubit_ids,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        gadget,
        data_ids=data_ids,
        ancilla_ids=ancilla_ids,
        bridge_ids=bridge_ids,
        measurement_record=measurement_record,
    )
    circuit += _surgery_final_detectors(
        gadget,
        merged_code,
        qubit_ids,
        measurement_record=measurement_record,
    )

    m_X, m_Z, n_V = (
        gadget.code.matrix_x.shape[0],
        gadget.code.matrix_z.shape[0],
        len(gadget.support),
    )
    if gadget.basis is Pauli.X:
        meas_check_ids = tuple(qubit_ids.checks_x[m_X : m_X + n_V])
    else:
        meas_check_ids = tuple(qubit_ids.checks_z[m_Z : m_Z + n_V])

    circuit += _surgery_observable(
        gadget,
        meas_check_ids=meas_check_ids,
        data_ids=data_ids,
        support_indices=gadget.support,
        measurement_record=measurement_record,
    )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)

    return circuit


def _stitch_intercode(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Inter-code joint stitch (g_l.code is not g_r.code). Handles both bases."""
    assert g_l.code is not g_r.code
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug

    # measured-basis abstraction: M_meas holds the new meas-basis check rows;
    # M_comp holds the dual cycle/comp-basis rows.
    if bridge.basis is Pauli.X:
        M_meas_l_src, M_comp_l_src = g_l_aug.HX_merged, g_l_aug.HZ_merged
        M_meas_r_src, M_comp_r_src = g_r_aug.HX_merged, g_r_aug.HZ_merged
        m_meas_l_data = g_l.code.matrix_x.shape[0]
        m_meas_r_data = g_r.code.matrix_x.shape[0]
        m_comp_l_data = g_l.code.matrix_z.shape[0]
        m_comp_r_data = g_r.code.matrix_z.shape[0]
    else:
        M_meas_l_src, M_comp_l_src = g_l_aug.HZ_merged, g_l_aug.HX_merged
        M_meas_r_src, M_comp_r_src = g_r_aug.HZ_merged, g_r_aug.HX_merged
        m_meas_l_data = g_l.code.matrix_z.shape[0]
        m_meas_r_data = g_r.code.matrix_z.shape[0]
        m_comp_l_data = g_l.code.matrix_x.shape[0]
        m_comp_r_data = g_r.code.matrix_x.shape[0]

    M_meas_l = np.asarray(M_meas_l_src).astype(np.int_)
    M_meas_r = np.asarray(M_meas_r_src).astype(np.int_)
    M_comp_l = np.asarray(M_comp_l_src).astype(np.int_)
    M_comp_r = np.asarray(M_comp_r_src).astype(np.int_)

    n_l, n_r = g_l.code.num_qudits, g_r.code.num_qudits
    k_l, k_r = g_l_aug.incidence.shape[0], g_r_aug.incidence.shape[0]
    w = bridge.width
    n_merged = n_l + n_r + k_l + k_r + w
    r_l, r_r = g_l_aug.gauge.shape[0], g_r_aug.gauge.shape[0]

    cl_data = slice(0, n_l)
    cr_data = slice(n_l, n_l + n_r)
    cl_ancilla = slice(n_l + n_r, n_l + n_r + k_l)
    cr_ancilla = slice(n_l + n_r + k_l, n_l + n_r + k_l + k_r)
    c_adapter = slice(n_l + n_r + k_l + k_r, n_merged)

    # Build M_meas: data χ-carrier rows (left & right) + χ rows + adapter Π labels.
    M_meas = np.zeros(
        (m_meas_l_data + m_meas_r_data + len(g_l.support) + len(g_r.support), n_merged),
        dtype=np.int_,
    )
    M_meas[:m_meas_l_data, cl_data] = M_meas_l[:m_meas_l_data, :n_l]
    M_meas[m_meas_l_data : m_meas_l_data + m_meas_r_data, cr_data] = M_meas_r[:m_meas_r_data, :n_r]
    meas_l_rows = M_meas_l[m_meas_l_data:, :]
    meas_r_rows = M_meas_r[m_meas_r_data:, :]
    meas_start = m_meas_l_data + m_meas_r_data
    M_meas[meas_start : meas_start + len(g_l.support), cl_data] = meas_l_rows[:, :n_l]
    M_meas[meas_start : meas_start + len(g_l.support), cl_ancilla] = meas_l_rows[:, n_l:]
    M_meas[meas_start + len(g_l.support) :, cr_data] = meas_r_rows[:, :n_r]
    M_meas[meas_start + len(g_l.support) :, cr_ancilla] = meas_r_rows[:, n_r:]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            M_meas[meas_start + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            M_meas[meas_start + len(g_l.support) + v_idx, c_adapter.start + lab] = 1

    # Build M_comp: co-carrier data rows (with κ extension) + G_aug + new cycle.
    M_comp = np.zeros(
        (m_comp_l_data + m_comp_r_data + r_l + r_r + (w - 1), n_merged),
        dtype=np.int_,
    )
    M_comp[:m_comp_l_data, cl_data] = M_comp_l[:m_comp_l_data, :n_l]
    M_comp[:m_comp_l_data, cl_ancilla] = M_comp_l[:m_comp_l_data, n_l:]
    M_comp[m_comp_l_data : m_comp_l_data + m_comp_r_data, cr_data] = M_comp_r[:m_comp_r_data, :n_r]
    M_comp[m_comp_l_data : m_comp_l_data + m_comp_r_data, cr_ancilla] = M_comp_r[
        :m_comp_r_data, n_r:
    ]
    g_start = m_comp_l_data + m_comp_r_data
    M_comp[g_start : g_start + r_l, cl_ancilla] = M_comp_l[m_comp_l_data:, n_l:]
    M_comp[g_start + r_l : g_start + r_l + r_r, cr_ancilla] = M_comp_r[m_comp_r_data:, n_r:]
    cyc_start = g_start + r_l + r_r
    M_comp[cyc_start:, cl_ancilla] = bridge.T_l
    M_comp[cyc_start:, cr_ancilla] = bridge.T_r
    M_comp[cyc_start:, c_adapter] = bridge.H_R

    if bridge.basis is Pauli.X:
        return CSSCode(field(M_meas), field(M_comp), is_subsystem_code=False)
    return CSSCode(field(M_comp), field(M_meas), is_subsystem_code=False)


def _stitch_intracode(g_l: GadgetLayout, g_r: GadgetLayout, bridge: Bridge) -> CSSCode:
    """Intra-code joint stitch (g_l.code is g_r.code). Handles both bases.

    Differences from _stitch_intercode:
      - Shared data check rows (count = m_meas/comp_data once, not l+r).
      - Shared data column block (n columns, not n_l + n_r).
      - χ rows from both sides write into the SAME data-column slice.
    """
    assert g_l.code is g_r.code
    field = g_l.code.field
    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug

    if bridge.basis is Pauli.X:
        M_meas_l_src, M_comp_l_src = g_l_aug.HX_merged, g_l_aug.HZ_merged
        M_meas_r_src, M_comp_r_src = g_r_aug.HX_merged, g_r_aug.HZ_merged
        m_meas_data = g_l.code.matrix_x.shape[0]
        m_comp_data = g_l.code.matrix_z.shape[0]
    else:
        M_meas_l_src, M_comp_l_src = g_l_aug.HZ_merged, g_l_aug.HX_merged
        M_meas_r_src, M_comp_r_src = g_r_aug.HZ_merged, g_r_aug.HX_merged
        m_meas_data = g_l.code.matrix_z.shape[0]
        m_comp_data = g_l.code.matrix_x.shape[0]

    M_meas_l = np.asarray(M_meas_l_src).astype(np.int_)
    M_meas_r = np.asarray(M_meas_r_src).astype(np.int_)
    M_comp_l = np.asarray(M_comp_l_src).astype(np.int_)
    M_comp_r = np.asarray(M_comp_r_src).astype(np.int_)

    n = g_l.code.num_qudits
    k_l, k_r = g_l_aug.incidence.shape[0], g_r_aug.incidence.shape[0]
    w = bridge.width
    n_merged = n + k_l + k_r + w
    r_l, r_r = g_l_aug.gauge.shape[0], g_r_aug.gauge.shape[0]

    c_data = slice(0, n)
    cl_ancilla = slice(n, n + k_l)
    cr_ancilla = slice(n + k_l, n + k_l + k_r)
    c_adapter = slice(n + k_l + k_r, n_merged)

    # Build M_meas: shared data check rows + χ rows (both sides into shared data).
    M_meas = np.zeros(
        (m_meas_data + len(g_l.support) + len(g_r.support), n_merged),
        dtype=np.int_,
    )
    M_meas[:m_meas_data, c_data] = M_meas_l[:m_meas_data, :n]  # shared
    meas_l_rows = M_meas_l[m_meas_data:, :]
    meas_r_rows = M_meas_r[m_meas_data:, :]
    M_meas[m_meas_data : m_meas_data + len(g_l.support), c_data] = meas_l_rows[:, :n]
    M_meas[m_meas_data : m_meas_data + len(g_l.support), cl_ancilla] = meas_l_rows[:, n:]
    M_meas[m_meas_data + len(g_l.support) :, c_data] = meas_r_rows[:, :n]
    M_meas[m_meas_data + len(g_l.support) :, cr_ancilla] = meas_r_rows[:, n:]
    for v_idx, lab in enumerate(bridge.label_l):
        if lab >= 0:
            M_meas[m_meas_data + v_idx, c_adapter.start + lab] = 1
    for v_idx, lab in enumerate(bridge.label_r):
        if lab >= 0:
            M_meas[m_meas_data + len(g_l.support) + v_idx, c_adapter.start + lab] = 1

    # Build M_comp: shared data co-carrier rows with κ extension on BOTH sides,
    # then G_l, G_r, then new cycle.
    M_comp = np.zeros(
        (m_comp_data + r_l + r_r + (w - 1), n_merged),
        dtype=np.int_,
    )
    M_comp[:m_comp_data, c_data] = M_comp_l[:m_comp_data, :n]
    M_comp[:m_comp_data, cl_ancilla] = M_comp_l[:m_comp_data, n:]
    M_comp[:m_comp_data, cr_ancilla] = M_comp_r[:m_comp_data, n:]
    M_comp[m_comp_data : m_comp_data + r_l, cl_ancilla] = M_comp_l[m_comp_data:, n:]
    M_comp[m_comp_data + r_l : m_comp_data + r_l + r_r, cr_ancilla] = M_comp_r[m_comp_data:, n:]
    cyc_start = m_comp_data + r_l + r_r
    M_comp[cyc_start:, cl_ancilla] = bridge.T_l
    M_comp[cyc_start:, cr_ancilla] = bridge.T_r
    M_comp[cyc_start:, c_adapter] = bridge.H_R

    if bridge.basis is Pauli.X:
        return CSSCode(field(M_meas), field(M_comp), is_subsystem_code=False)
    return CSSCode(field(M_comp), field(M_meas), is_subsystem_code=False)


def _stitch_to_joint_csscode(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
) -> CSSCode:
    """Assemble merged CSSCode for two-PPM surgery.

    Dispatches on the structural axis (g_l.code is g_r.code → intra-code
    shares data; otherwise inter-code).  Each branch handles both
    bridge.basis values internally via the χ-carrier abstraction.
    """
    if g_l.code is g_r.code:
        return _stitch_intracode(g_l, g_r, bridge)
    return _stitch_intercode(g_l, g_r, bridge)


def _expand_joint_data_init(
    data_init: str | tuple[str, ...] | list[str] | None,
    n_l: int,
    n_r: int,
    intercode: bool,
) -> str | None:
    """Normalize ``data_init`` to a per-physical-qubit string.

    Two accepted shapes:

      * ``str`` (or ``None``) — passed through verbatim to ``_surgery_state_prep``
        (length-1 broadcasts to all data qubits; length n_l + n_r is per-qubit).

      * ``tuple[str, str]`` (or list) — per-code logical-init spec. Each entry
        is a string that is itself per-code broadcast (length 1) or per-qubit
        (length n_code). Tuple form is only valid for intercode joint PPM
        (intracode has a single data set; use a plain string instead).
        Example: ``("0", "+")`` initializes c_l data to |0⟩^{n_l} and c_r data
        to |+⟩^{n_r} — which, after the first round of merged-code SE projects
        into the codespace, equals logical |0⟩_L ⊗ |+⟩_L for any CSS code.
    """
    if data_init is None or isinstance(data_init, str):
        return data_init
    if not isinstance(data_init, (tuple, list)):
        raise TypeError(
            f"data_init must be str, tuple, list, or None; got {type(data_init).__name__}"
        )
    if not intercode:
        raise ValueError(
            "tuple/list data_init only valid for intercode joint PPM; "
            "intracode joint has a single data set, pass a plain string instead"
        )
    if len(data_init) != 2:
        raise ValueError(
            f"data_init tuple must have 2 entries (one per code), got {len(data_init)}"
        )
    spec_l, spec_r = data_init
    if not isinstance(spec_l, str) or not isinstance(spec_r, str):
        raise TypeError(
            f"data_init tuple entries must be str, got "
            f"({type(spec_l).__name__}, {type(spec_r).__name__})"
        )
    if len(spec_l) == 1:
        spec_l = spec_l * n_l
    if len(spec_r) == 1:
        spec_r = spec_r * n_r
    if len(spec_l) != n_l:
        raise ValueError(f"data_init[0] length {len(spec_l)} does not match c_l data count {n_l}")
    if len(spec_r) != n_r:
        raise ValueError(f"data_init[1] length {len(spec_r)} does not match c_r data count {n_r}")
    return spec_l + spec_r


def build_joint_ppm_circuit(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    bridge: Bridge,
    *,
    rounds: int,
    noise_model: NoiseModel | None = None,
    data_init: str | tuple[str, ...] | list[str] | None = None,
) -> tuple[stim.Circuit, CSSCode]:
    """Joint-PPM circuit (universal adapter; no U_B in α*).

    Emits two OBSERVABLE_INCLUDE entries (see ``_surgery_observable`` for
    full semantics):

      * obs0 — Single-round joint readout via Webster's identity
        ∏_{v ∈ support_l ∪ support_r} A_v = X̄_l ⊗ X̄_r (or Z̄_l ⊗ Z̄_r for
        basis=Z). See Webster, Smith, Cohen arXiv:2511.15989 §II.A. XOR of
        the **last** QEC round's meas-check outcomes on both patches.
        Detectors carry the FT load. Reading at the final round should be
        decoding-equivalent to Cain et al.'s first-cycle readout
        (arXiv:2603.28627 App. D) because the interface detectors telescope,
        so the observable's fault distance should be unchanged.
      * obs1 — Direct destructive M on ``support_l ∪ support_r``; noiseless
        cross-check, not a physical protocol.

    For LER / noisy runs, use ``keep_only_observable(circuit, keep_idx=0)``.

    ``data_init`` (optional): override the per-code data init.

      * ``str`` — per-physical-qubit (or len-1 broadcast). For intercode,
        positions [0:n_l) are left, [n_l:n_l+n_r) are right; for intracode,
        length is n_l. See ``_surgery_state_prep`` for the char-to-state mapping.
      * ``tuple[str, str]`` (intercode only) — per-code logical-init spec.
        ``data_init=("0", "+")`` → c_l in |0⟩_L, c_r in |+⟩_L.
    """
    joint_code = _stitch_to_joint_csscode(g_l, g_r, bridge)
    qubit_ids = QubitIDs.from_code(joint_code)
    intercode = g_l.code is not g_r.code

    g_l_aug, g_r_aug = bridge.g_l_aug, bridge.g_r_aug
    n_l = g_l.code.num_qudits
    n_r = g_r.code.num_qudits if intercode else 0
    k_l = g_l_aug.incidence.shape[0]
    k_r = g_r_aug.incidence.shape[0]
    w = bridge.width

    if intercode:
        data_ids = qubit_ids.data[: n_l + n_r]
        support_combined = tuple(g_l.support) + tuple(n_l + i for i in g_r.support)
    else:
        data_ids = qubit_ids.data[:n_l]
        support_combined = tuple(g_l.support) + tuple(g_r.support)
    ancilla_ids = qubit_ids.data[n_l + n_r : n_l + n_r + k_l + k_r]
    bridge_ids = qubit_ids.data[n_l + n_r + k_l + k_r :]
    assert len(bridge_ids) == w

    circuit = _surgery_qubit_coordinates(
        g_l,
        qubit_ids,
        joint=(g_r, bridge, intercode),
    )
    expanded_data_init = _expand_joint_data_init(data_init, n_l, n_r, intercode)
    circuit += _surgery_state_prep(
        g_l,
        data_ids,
        ancilla_ids,
        bridge_ids,
        data_init=expanded_data_init,
    )
    qec_cycle, measurement_record, _ = _surgery_qec_cycle_joint(
        g_l,
        g_r,
        joint_code,
        bridge,
        num_rounds=rounds,
        qubit_ids=qubit_ids,
        intercode=intercode,
    )
    circuit += qec_cycle
    circuit += _surgery_detach_and_readout(
        g_l,
        data_ids=data_ids,
        ancilla_ids=ancilla_ids,
        bridge_ids=bridge_ids,
        measurement_record=measurement_record,
    )
    circuit += _surgery_final_detectors_joint(
        g_l,
        g_r,
        joint_code,
        bridge,
        qubit_ids,
        measurement_record=measurement_record,
        intercode=intercode,
    )

    # χ check IDs: data H_X^(l) rows occupy first mX_l indices in
    # qubit_ids.checks_x, then m_X_r (inter-code), then χ^(l), then χ^(r).
    if bridge.basis is Pauli.X:
        check_ids = qubit_ids.checks_x
        m_l = g_l.code.matrix_x.shape[0]
        m_r = g_r.code.matrix_x.shape[0] if intercode else 0
    else:
        check_ids = qubit_ids.checks_z
        m_l = g_l.code.matrix_z.shape[0]
        m_r = g_r.code.matrix_z.shape[0] if intercode else 0
    n_V_l = len(g_l.support)
    n_V_r = len(g_r.support)
    meas_l_offset = m_l + m_r
    meas_r_offset = meas_l_offset + n_V_l
    meas_l_ids = tuple(check_ids[meas_l_offset : meas_l_offset + n_V_l])
    meas_r_ids = tuple(check_ids[meas_r_offset : meas_r_offset + n_V_r])
    meas_check_ids = meas_l_ids + meas_r_ids  # NO U_B / no adapter cycle-check ids

    circuit += _surgery_observable(
        g_l,
        meas_check_ids=meas_check_ids,
        data_ids=data_ids,
        support_indices=support_combined,
        measurement_record=measurement_record,
    )

    if noise_model is not None:
        circuit = noise_model.noisy_circuit(circuit)
    return circuit, joint_code


def _classify_reliable_round1_checks_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    qubit_ids: QubitIDs,
    *,
    intercode: bool,
) -> tuple[int, ...]:
    """Joint-code variant: reliable checks across both gadgets + new cycle rows.

    basis=X (data |+⟩, ancilla + bridge |0⟩):
        H_X rows = [data S_X^(l), data S_X^(r), S'_meas^(l), S'_meas^(r)]
        H_Z rows = [data S_Z^(l) ext, data S_Z^(r) ext, S'_comp^(l)_aug,
                    S'_comp^(r)_aug, new cycle-Z]

      Reliable X: data S_X rows of both gadgets.
      Reliable Z: S'_comp_aug rows + new cycle-Z rows (all act on ancilla ∪ bridge, all |0⟩).

    basis=Z is the X↔Z dual.
    """
    m_X_l = g_l.code.matrix_x.shape[0]
    m_X_r = g_r.code.matrix_x.shape[0] if intercode else 0
    m_Z_l = g_l.code.matrix_z.shape[0]
    m_Z_r = g_r.code.matrix_z.shape[0] if intercode else 0
    if g_l.basis is Pauli.X:
        reliable_x = qubit_ids.checks_x[: m_X_l + m_X_r]  # data S_X^(l/r)
        reliable_z = qubit_ids.checks_z[m_Z_l + m_Z_r :]  # S'_comp_aug + new cycle-Z
    else:
        reliable_x = qubit_ids.checks_x[m_X_l + m_X_r :]  # S'_comp_aug + new cycle-X
        reliable_z = qubit_ids.checks_z[: m_Z_l + m_Z_r]  # data S_Z^(l/r)
    return tuple(reliable_x) + tuple(reliable_z)


def _surgery_qec_cycle_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    joint_code: CSSCode,
    bridge: Bridge,
    num_rounds: int,
    qubit_ids: QubitIDs,
    *,
    intercode: bool,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """Joint-code variant of _surgery_qec_cycle that classifies reliable checks
    across both gadgets + the bridge's new cycle-checks."""
    strategy = EdgeColoring()
    one_round, round_measurement_record = strategy.get_circuit(joint_code, qubit_ids)
    reliable = set(
        _classify_reliable_round1_checks_joint(
            g_l,
            g_r,
            qubit_ids,
            intercode=intercode,
        )
    )
    all_check_ids = qubit_ids.check
    lane_idx = _check_lane_index_map(
        g_l,
        qubit_ids,
        joint=(g_r, bridge, intercode),
    )

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    circuit += one_round
    measurement_record.append(round_measurement_record)
    for check_id in all_check_ids:
        if check_id in reliable:
            lane, idx = lane_idx[check_id]
            circuit.append(
                "DETECTOR", [measurement_record.get_target_rec(check_id)], (idx, lane, 0)
            )
    reliable_in_order = [cid for cid in all_check_ids if cid in reliable]
    detector_record.append({cid: dd for dd, cid in enumerate(reliable_in_order)})

    if num_rounds > 1:
        repeat_circuit = one_round.copy()
        measurement_record.append(round_measurement_record)
        repeat_circuit.append("SHIFT_COORDS", [], (0, 0, 1))
        for check_id in all_check_ids:
            lane, idx = lane_idx[check_id]
            repeat_circuit.append(
                "DETECTOR",
                [
                    measurement_record.get_target_rec(check_id, -1),
                    measurement_record.get_target_rec(check_id, -2),
                ],
                (idx, lane, 0),
            )
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))
        measurement_record.append(round_measurement_record, repeat=num_rounds - 2)
        detector_record.append(
            {cid: dd for dd, cid in enumerate(all_check_ids)},
            repeat=num_rounds - 1,
        )

    return circuit, measurement_record, detector_record


def _surgery_final_detectors_joint(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    joint_code: CSSCode,
    bridge: Bridge,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
    intercode: bool,
) -> stim.Circuit:
    """Joint-code variant of _surgery_final_detectors.

    Emits detectors for the same reliable stabilizers as the round-1 classifier:
    basis=X: data H_X rows from both gadgets + G_aug + new cycle-Z rows.
    basis=Z: data H_Z rows from both gadgets + G_aug + new cycle-X rows.
    """
    m_X_l = g_l.code.matrix_x.shape[0]
    m_X_r = g_r.code.matrix_x.shape[0] if intercode else 0
    m_Z_l = g_l.code.matrix_z.shape[0]
    m_Z_r = g_r.code.matrix_z.shape[0] if intercode else 0
    HX = np.asarray(joint_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(joint_code.matrix_z).astype(np.uint8)

    circuit = stim.Circuit()
    lane_idx = _check_lane_index_map(
        g_l,
        qubit_ids,
        joint=(g_r, bridge, intercode),
    )

    def _emit_detector(stab_row: np.ndarray, check_id: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        lane, idx = lane_idx[check_id]
        circuit.append("DETECTOR", targets, (idx, lane, 0))

    if g_l.basis is Pauli.X:
        for kk in range(m_X_l + m_X_r):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])
        for kk in range(m_Z_l + m_Z_r, HZ.shape[0]):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
    else:
        for kk in range(m_Z_l + m_Z_r):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
        for kk in range(m_X_l + m_X_r, HX.shape[0]):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])

    return circuit


def _classify_reliable_round1_checks(
    gadget: GadgetLayout,
    qubit_ids: QubitIDs,
) -> tuple[int, ...]:
    """Check ancillas with deterministic round-1 syndrome given surgery init state."""
    m_X, m_Z = gadget.code.matrix_x.shape[0], gadget.code.matrix_z.shape[0]
    if gadget.basis is Pauli.X:
        reliable_x = qubit_ids.checks_x[:m_X]  # data S_X rows (det. +1)
        reliable_z = qubit_ids.checks_z[m_Z:]  # gauge rows (= S'_comp) (det. +1)
    else:
        reliable_x = qubit_ids.checks_x[m_X:]  # gauge rows (= S'_comp)
        reliable_z = qubit_ids.checks_z[:m_Z]  # data S_Z rows

    return tuple(reliable_x) + tuple(reliable_z)


def _surgery_state_prep(
    gadget: GadgetLayout,
    data_ids: tuple[int, ...],
    ancilla_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...] = (),
    *,
    data_init: str | None = None,
) -> stim.Circuit:
    """Init data/ancilla/bridge qubits at the start of a surgery PPM circuit.

    Default (``data_init=None``):
      basis=X → data |+⟩ (RX), ancilla + bridge |0⟩ (R)
      basis=Z → data |0⟩ (R),  ancilla + bridge |+⟩ (RX)

    Optional ``data_init`` overrides per-data-qubit initial state. Each
    character selects a state for the data qubit at the same position:

      "0" → |0⟩  (R)
      "1" → |1⟩  (R + post-init X)
      "+" → |+⟩  (RX)
      "-" → |-⟩  (RX + post-init Z)

    A length-1 string broadcasts to all data qubits; otherwise length must
    equal ``len(data_ids)``.  ancilla + bridge init is independent of ``data_init``
    and always follows the protocol default (basis-complement +1 eigenstate).
    """
    if data_init is None:
        default_char = "+" if gadget.basis is Pauli.X else "0"
        per_qubit = default_char * len(data_ids)
    else:
        if len(data_init) == 1:
            data_init = data_init * len(data_ids)
        if len(data_init) != len(data_ids):
            raise ValueError(
                f"data_init length {len(data_init)} does not match num data "
                f"qubits {len(data_ids)}; pass a length-1 string to broadcast"
            )
        invalid = sorted(set(data_init) - set("01+-"))
        if invalid:
            raise ValueError(
                f"data_init must contain only '0', '1', '+', '-'; got invalid chars {invalid}"
            )
        per_qubit = data_init

    r_data: list[int] = []
    rx_data: list[int] = []
    x_after: list[int] = []
    z_after: list[int] = []
    for q, c in zip(data_ids, per_qubit):
        if c == "0":
            r_data.append(q)
        elif c == "1":
            r_data.append(q)
            x_after.append(q)
        elif c == "+":
            rx_data.append(q)
        else:  # "-"
            rx_data.append(q)
            z_after.append(q)

    circuit = stim.Circuit()
    if r_data:
        circuit.append("R", r_data)
    if rx_data:
        circuit.append("RX", rx_data)
    if x_after:
        circuit.append("X", x_after)
    if z_after:
        circuit.append("Z", z_after)

    anc_ids = list(ancilla_ids) + (list(bridge_ids) if bridge_ids else [])
    if anc_ids:
        anc_init = "R" if gadget.basis is Pauli.X else "RX"
        circuit.append(anc_init, anc_ids)

    return circuit


def _surgery_qec_cycle(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    num_rounds: int,
    qubit_ids: QubitIDs,
) -> tuple[stim.Circuit, MeasurementRecord, DetectorRecord]:
    """num_rounds of merged-code SE; round-1 detectors only for reliable checks."""
    strategy = EdgeColoring()
    one_round, round_measurement_record = strategy.get_circuit(merged_code, qubit_ids)
    reliable = set(_classify_reliable_round1_checks(gadget, qubit_ids))
    all_check_ids = qubit_ids.check
    lane_idx = _check_lane_index_map(gadget, qubit_ids)

    circuit = stim.Circuit()
    measurement_record = MeasurementRecord()
    detector_record = DetectorRecord()

    # Round 1: emit DETECTORs only for reliable checks.
    circuit += one_round
    measurement_record.append(round_measurement_record)
    for check_id in all_check_ids:
        if check_id in reliable:
            lane, idx = lane_idx[check_id]
            circuit.append(
                "DETECTOR", [measurement_record.get_target_rec(check_id)], (idx, lane, 0)
            )
    reliable_in_order = [cid for cid in all_check_ids if cid in reliable]
    detector_record.append({cid: dd for dd, cid in enumerate(reliable_in_order)})

    if num_rounds > 1:
        repeat_circuit = one_round.copy()
        measurement_record.append(round_measurement_record)
        repeat_circuit.append("SHIFT_COORDS", [], (0, 0, 1))
        for check_id in all_check_ids:
            lane, idx = lane_idx[check_id]
            repeat_circuit.append(
                "DETECTOR",
                [
                    measurement_record.get_target_rec(check_id, -1),
                    measurement_record.get_target_rec(check_id, -2),
                ],
                (idx, lane, 0),
            )
        circuit.append(stim.CircuitRepeatBlock(num_rounds - 1, repeat_circuit))
        measurement_record.append(round_measurement_record, repeat=num_rounds - 2)
        detector_record.append(
            {cid: dd for dd, cid in enumerate(all_check_ids)},
            repeat=num_rounds - 1,
        )

    return circuit, measurement_record, detector_record


def _surgery_observable(
    gadget: GadgetLayout,
    *,
    meas_check_ids: tuple[int, ...],
    data_ids: tuple[int, ...],
    support_indices: tuple[int, ...],
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Emit two OBSERVABLE_INCLUDE entries (obs0, obs1) for the surgery PPM.

    obs0 — physical readout of the logical Pauli. The merged stabilizer group
        satisfies the single-round identity Z̄ = ∏_{v ∈ support} A_v (Webster,
        Smith, Cohen arXiv:2511.15989 §II.A, gadget Eq. 1). We point
        ``OBSERVABLE_INCLUDE`` at the **last** QEC round's meas-check (S'_meas)
        outcomes — their XOR is the eigenvalue bit of Z̄ (or X̄ for basis=X).
        Detectors carry the FT load via round-to-round consistency. Reading at
        the final round should be decoding-equivalent to Cain et al.'s
        first-cycle readout (arXiv:2603.28627 App. D) because the interface
        detectors telescope, so the observable's fault distance should be
        unchanged (argued, not yet covered by a fault-distance test).

    obs1 — Direct stim measurement of the data qubits on ``support``. NOT a
        physical protocol — destructively projects the data — but a useful
        noiseless cross-check: in any noiseless shot ``obs0 == obs1``.

    For LER sweeps and any noisy run, keep ONLY obs0 via
    ``keep_only_observable(circuit, keep_idx=0)``.
    """
    # Precondition: every meas-check ancilla must have been measured during
    # the QEC cycle. Detach/readout only touches data + κ + bridge, never the
    # meas-check ancillas, so ``get_target_rec(cid)`` (default -1) is
    # guaranteed to resolve to the last QEC round. Fail loudly if a future
    # refactor breaks this.
    for cid in meas_check_ids:
        assert measurement_record[cid], (
            f"meas-check {cid} has no measurement record; "
            f"_surgery_observable expects the QEC cycle to have run first."
        )
    circuit = stim.Circuit()
    meas_targets = [measurement_record.get_target_rec(cid) for cid in meas_check_ids]
    circuit.append("OBSERVABLE_INCLUDE", meas_targets, 0)
    data_targets = [measurement_record.get_target_rec(data_ids[i]) for i in support_indices]
    circuit.append("OBSERVABLE_INCLUDE", data_targets, 1)
    return circuit


def _surgery_final_detectors(
    gadget: GadgetLayout,
    merged_code: CSSCode,
    qubit_ids: QubitIDs,
    *,
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Emit DETECTORs for reliable stabs inferable from final readouts.

    For basis=X: data H_X (from Mx data) + G (from Mz κ).
    For basis=Z: data H_Z (from Mz data) + G (from Mx κ).
    Each DETECTOR XORs ⊕(final M-record on stab support) ⊕ last-round syndrome.
    """
    m_X = gadget.code.matrix_x.shape[0]
    m_Z = gadget.code.matrix_z.shape[0]
    HX = np.asarray(merged_code.matrix_x).astype(np.uint8)
    HZ = np.asarray(merged_code.matrix_z).astype(np.uint8)

    circuit = stim.Circuit()
    lane_idx = _check_lane_index_map(gadget, qubit_ids)

    def _emit_detector(stab_row: np.ndarray, check_id: int) -> None:
        supp = np.where(stab_row)[0]
        targets = [measurement_record.get_target_rec(qubit_ids.data[q]) for q in supp]
        targets.append(measurement_record.get_target_rec(check_id, -1))
        lane, idx = lane_idx[check_id]
        circuit.append("DETECTOR", targets, (idx, lane, 0))

    if gadget.basis is Pauli.X:
        for kk in range(m_X):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])
        for kk in range(m_Z, HZ.shape[0]):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
    else:  # Pauli.Z (symmetric: chi in HZ, G in HX)
        for kk in range(m_Z):
            _emit_detector(HZ[kk], qubit_ids.checks_z[kk])
        for kk in range(m_X, HX.shape[0]):
            _emit_detector(HX[kk], qubit_ids.checks_x[kk])

    return circuit


def _surgery_detach_and_readout(
    gadget: GadgetLayout,
    *,
    data_ids: tuple[int, ...],
    ancilla_ids: tuple[int, ...],
    bridge_ids: tuple[int, ...],
    measurement_record: MeasurementRecord,
) -> stim.Circuit:
    """Cain step 3 + final data measure. Mκ then SHIFT_COORDS then Mdata."""
    circuit = stim.Circuit()
    detach_qubits = list(ancilla_ids) + list(bridge_ids)
    ancilla_op = "M" if gadget.basis is Pauli.X else "MX"
    data_op = "MX" if gadget.basis is Pauli.X else "M"
    circuit.append(ancilla_op, detach_qubits)
    measurement_record.append({q: i for i, q in enumerate(detach_qubits)})
    circuit.append("SHIFT_COORDS", [], (0, 0, 1))
    circuit.append(data_op, list(data_ids))
    measurement_record.append({q: i for i, q in enumerate(data_ids)})
    return circuit
