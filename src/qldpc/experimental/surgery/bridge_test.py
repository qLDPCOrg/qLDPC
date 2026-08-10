"""Tests for src/qldpc/experimental/surgery/bridge.py."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli

from ._webster_fixture import (
    _webster_z_bar_operator,
    build_generalised_bicycle_code,
    load_webster_seed_set,
)


def test_skip_tree_fullrank_on_K4_matches_H_R() -> None:
    """SkipTree full-rank: T_ind · G · P_ind = H_R for the complete graph K_4."""
    import networkx as nx

    from qldpc.experimental.surgery.bridge import _canonical_H_R, _skip_tree_fullrank

    G_nx = nx.complete_graph(4)
    n = 4
    edges = sorted(tuple(sorted(e)) for e in G_nx.edges())
    edge_index_verts = {e: i for i, e in enumerate(edges)}
    G_mat = np.zeros((len(edges), n), dtype=np.int_)
    for (u, v), i in edge_index_verts.items():
        G_mat[i, u] = 1
        G_mat[i, v] = 1

    T_ind, P_ind = _skip_tree_fullrank(G_nx, root=0, edge_index_verts=edge_index_verts)
    H_R = _canonical_H_R(n)

    assert T_ind.shape == (n - 1, len(edges))
    assert P_ind.shape == (n, n)
    # SkipTree key identity: T_ind · G · P_ind == H_R over GF(2)
    product = (T_ind @ G_mat @ P_ind) % 2
    assert np.array_equal(product, H_R), f"got\n{product}\nwant\n{H_R}"
    # Paper Theorem 7: (3,2)-sparsity is a general invariant of SkipTree.
    assert T_ind.sum(axis=1).max() <= 3
    assert T_ind.sum(axis=0).max() <= 2


def test_build_aux_graph_weight2_rows_become_edges() -> None:
    """F rows of weight 2 → graph edges; vertex set = {0, ..., |V_0|-1}."""
    from qldpc.experimental.surgery.bridge import _build_aux_graph_strict

    incidence = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=np.uint8)
    G_nx, edge_idx = _build_aux_graph_strict(incidence)
    assert set(G_nx.nodes) == {0, 1, 2, 3}
    assert {tuple(sorted(e)) for e in G_nx.edges} == {(0, 1), (1, 2), (2, 3)}
    assert edge_idx[(0, 1)] == 0
    assert edge_idx[(1, 2)] == 1
    assert edge_idx[(2, 3)] == 2


def test_build_aux_graph_filters_hyperedges() -> None:
    """F rows of weight >= 3 (hyperedges) are silently skipped; weight-2 rows survive."""
    from qldpc.experimental.surgery.bridge import _build_aux_graph_strict

    incidence = np.array(
        [
            [1, 1, 0, 0, 0],  # weight-2 → edge (0,1)
            [1, 1, 1, 1, 0],  # weight-4 hyperedge → skipped
            [0, 0, 1, 1, 0],  # weight-2 → edge (2,3)
            [0, 0, 0, 1, 1],  # weight-2 → edge (3,4)
        ],
        dtype=np.uint8,
    )
    G_nx, edge_idx = _build_aux_graph_strict(incidence)
    assert set(G_nx.nodes) == {0, 1, 2, 3, 4}
    # Three weight-2 rows → three edges; hyperedge row contributes nothing
    assert G_nx.number_of_edges() == 3
    assert (0, 1) in edge_idx
    assert (2, 3) in edge_idx
    assert (3, 4) in edge_idx
    # Hyperedge would have produced edges (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
    # but only edges from weight-2 rows are present
    assert (0, 2) not in edge_idx
    assert (0, 3) not in edge_idx
    assert (1, 3) not in edge_idx


def test_build_aux_graph_rejects_weight1_row() -> None:
    """F rows of weight 1 raise ValueError (dangling edge / no-op stabilizer)."""
    from qldpc.experimental.surgery.bridge import _build_aux_graph_strict

    incidence = np.array([[1, 1, 0, 0], [0, 0, 1, 0]], dtype=np.uint8)
    with pytest.raises(ValueError, match=r"weight 1"):
        _build_aux_graph_strict(incidence)


def test_connect_induced_subgraph_no_op_when_connected() -> None:
    """If induced subgraph is already connected, no edges are added."""
    import networkx as nx

    from qldpc.experimental.surgery.bridge import _connect_induced_subgraph

    G_aux = nx.path_graph(4)  # 0-1-2-3
    extra = _connect_induced_subgraph(G_aux, ports=(0, 1, 2, 3))
    assert extra == []
    assert {tuple(sorted(e)) for e in G_aux.edges} == {(0, 1), (1, 2), (2, 3)}


def test_connect_induced_subgraph_adds_edges_to_disconnected_components() -> None:
    """Disconnected induced subgraph gets one bridging edge per missing connection."""
    import networkx as nx

    from qldpc.experimental.surgery.bridge import _connect_induced_subgraph

    # G_aux: 0-1   2-3 (two separate components)
    G_aux = nx.Graph()
    G_aux.add_edges_from([(0, 1), (2, 3)])
    extra = _connect_induced_subgraph(G_aux, ports=(0, 1, 2, 3))
    assert len(extra) == 1  # exactly one bridge needed
    (u, v) = extra[0]
    # Endpoints must come from different original components
    assert {u, v} & {0, 1} and {u, v} & {2, 3}
    # G_aux mutated: induced subgraph now connected
    assert nx.is_connected(G_aux.subgraph((0, 1, 2, 3)))


def test_cellulate_caps_cycle_length() -> None:
    """After cellulation, every basis cycle has length <= cap."""
    import networkx as nx

    from qldpc.experimental.surgery.bridge import _cellulate_port_subgraph

    # 10-cycle: 0-1-2-...-9-0 has one length-10 basis cycle
    G_aux = nx.cycle_graph(10)
    added = _cellulate_port_subgraph(G_aux, ports=tuple(range(10)), max_len=6)
    assert len(added) >= 1
    # All basis cycles now bounded
    sub = G_aux.subgraph(tuple(range(10)))
    assert max((len(c) for c in nx.cycle_basis(sub)), default=0) <= 6


def test_cellulate_no_op_when_already_short() -> None:
    """If all basis cycles are short, no edges are added."""
    import networkx as nx

    from qldpc.experimental.surgery.bridge import _cellulate_port_subgraph

    G_aux = nx.cycle_graph(4)  # one 4-cycle
    added = _cellulate_port_subgraph(G_aux, ports=(0, 1, 2, 3), max_len=6)
    assert added == []


def test_cellulate_raises_when_port_cycle_has_no_available_chord() -> None:
    """RuntimeError when a port-subgraph cycle exists but every (i, j) pair is already an edge.

    I.e. the port subgraph is complete on those vertices.
    """
    import networkx as nx

    from qldpc.experimental.surgery.bridge import _cellulate_port_subgraph

    # 7-cycle 0-1-2-3-4-5-6-0 plus ALL chords among {0..6} → complete graph K_7.
    # cycle_basis still surfaces cycles of length > max_len in K_7 (basis cycles
    # are length-3 triangles), so no long cycle exists in this case.
    # Instead: make a 7-cycle without any extra edges, then call with max_len=2.
    G = nx.cycle_graph(7)
    ports = tuple(range(7))
    # Already a complete graph K_7? No — cycle_graph(7) has only 7 edges.
    # Pre-saturate with all possible chords so no chord can be added:
    for i in range(7):
        for j in range(i + 2, 7):
            if not G.has_edge(i, j) and (i, j) != (0, 6):
                G.add_edge(i, j)
    # Now every (i, j) with j >= i+2 in the 7-cycle is already an edge.
    # A length-7 basis cycle no longer exists (it's broken into triangles),
    # so max_len=6 finds no long cycle and returns []. Use max_len=2 to force
    # the failure path:
    with pytest.raises(RuntimeError, match=r"No chord found"):
        _cellulate_port_subgraph(G, ports, max_len=2)


def test_cellulate_port_subgraph_breaks_long_port_cycle() -> None:
    """Ports are a strict subset of vertices, with a long cycle on the port subgraph.

    Cellulation breaks the port cycle without inspecting non-port edges elsewhere in G_aux.
    """
    import networkx as nx

    from qldpc.experimental.surgery.bridge import _cellulate_port_subgraph

    G = nx.Graph()
    # 8-cycle on port vertices 0..7
    G.add_edges_from([(i, (i + 1) % 8) for i in range(8)])
    # Non-port "decoration": dangling vertex 100 attached to port 0
    G.add_edge(0, 100)
    ports = tuple(range(8))
    added = _cellulate_port_subgraph(G, ports, max_len=6)
    assert len(added) >= 1
    # All chord endpoints must be ports (cycle vertices are port vertices)
    for u, v in added:
        assert u in ports and v in ports
    # The non-port vertex 100 was not touched
    assert G.has_edge(0, 100)
    # All port-subgraph basis cycles now bounded
    sub = G.subgraph(ports)
    for c in nx.cycle_basis(sub):
        assert len(c) <= 6


def test_cellulate_port_subgraph_skips_non_port_cycle() -> None:
    """Long cycle entirely on non-port vertices is ignored; no edges added."""
    import networkx as nx

    from qldpc.experimental.surgery.bridge import _cellulate_port_subgraph

    G = nx.Graph()
    # Long non-port cycle: 10-11-12-...-17-10 (length 8)
    G.add_edges_from(
        [(10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 10)]
    )
    # Short port cycle: triangle on 0,1,2
    G.add_edges_from([(0, 1), (1, 2), (2, 0)])
    ports = (0, 1, 2)
    n_edges_before = G.number_of_edges()
    added = _cellulate_port_subgraph(G, ports, max_len=6)
    assert added == []
    assert G.number_of_edges() == n_edges_before


def test_bridge_dataclass_fields_universal_adapter() -> None:
    """Bridge dataclass exposes the universal-adapter fields.

    Swaroop et al. arXiv:2410.03628 §IV.
    """
    import dataclasses

    from qldpc.experimental.surgery.bridge import Bridge

    fields = {f.name for f in dataclasses.fields(Bridge)}
    assert fields == {
        "width",
        "basis",
        "port_l",
        "port_r",
        "label_l",
        "label_r",
        "extra_ancilla_l",
        "extra_ancilla_r",
        "T_l",
        "T_r",
        "H_R",
        "g_l_aug",
        "g_r_aug",
    }


def test_build_bridge_smoke_steane_intracode() -> None:
    """Steane × Steane intra-code joint X̄ X̄: build_bridge returns valid Bridge."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x1 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    x2 = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)  # same logical
    g_l = build_gadget(code, x1, basis=Pauli.X)
    g_r = build_gadget(code, x2, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)
    assert bridge.width == min(len(g_l.support), len(g_r.support))
    assert bridge.basis is Pauli.X
    assert len(bridge.port_l) == bridge.width
    assert len(bridge.port_r) == bridge.width
    assert bridge.T_l.shape == (bridge.width - 1, bridge.g_l_aug.incidence.shape[0])
    assert bridge.T_r.shape == (bridge.width - 1, bridge.g_r_aug.incidence.shape[0])
    assert bridge.H_R.shape == (bridge.width - 1, bridge.width)


def test_build_bridge_skiptree_invariant_holds() -> None:
    """T_s · G_s_aug · P_s = H_R for both sides on Steane × Steane."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, x, basis=Pauli.X)
    bridge = build_bridge(g_l, g_r)

    for side in ("l", "r"):
        T = getattr(bridge, f"T_{side}")
        g_aug = getattr(bridge, f"g_{side}_aug")
        label = getattr(bridge, f"label_{side}")
        # adjacency = incidence_aug (rows = edges = ancilla qubits, cols = support vertices)
        adjacency = g_aug.incidence.astype(np.int_)
        # P_s: |V_0^(s)| × w; P_s[v, k] = 1 iff v ∈ port AND label[v] == k
        P = np.zeros((adjacency.shape[1], bridge.width), dtype=np.int_)
        for v_idx, lab in enumerate(label):
            if lab >= 0:
                P[v_idx, lab] = 1
        lhs = (T @ adjacency @ P) % 2
        assert np.array_equal(lhs, bridge.H_R), f"side {side}:\n{lhs}\nvs\n{bridge.H_R}"


def test_build_bridge_rejects_basis_mismatch() -> None:
    """Bridge requires g_l.basis == g_r.basis."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    z = np.asarray(code.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code, x, basis=Pauli.X)
    g_r = build_gadget(code, z, basis=Pauli.Z)
    with pytest.raises(ValueError, match=r"basis"):
        build_bridge(g_l, g_r)


def test_build_bridge_bb18_hyperedge_and_long_cycle() -> None:
    """End-to-end: Cain bb_18 BBCode triggers both Bug 1 (hyperedge) and Bug 2 (long cycle).

    Bug 2 is a long port-subgraph cycle. build_bridge must succeed and produce a merged code with
    k_merged = k_orig - 1 (intra-code joint Z̄_1 ⊗ Z̄_2).

    Two *distinct* Z-logicals are used so that the joint measurement reduces k by exactly
    1.  Z-logical 0 has a weight-4 F row (triggers Bug 1); the pair together exercises the full
    _cellulate_port_subgraph path (Bug 2)."""
    import sympy

    from qldpc.experimental.surgery import build_bridge, build_gadget
    from qldpc.experimental.surgery.circuit import _stitch_to_joint_csscode

    x, y = sympy.symbols("x y")
    code = codes.BBCode(
        {x: 31, y: 4},
        1 + x**6 * y + x**27,
        y**2 + x**15 * y**3 + x**24,
    )
    z_ops = code.get_logical_ops(Pauli.Z)
    z0 = np.asarray(z_ops[0]).astype(np.uint8)  # hyperedge logical (Bug 1)
    z1 = np.asarray(z_ops[1]).astype(np.uint8)  # distinct second logical
    g_l = build_gadget(code, z0, basis=Pauli.Z)
    g_r = build_gadget(code, z1, basis=Pauli.Z)
    # Confirm we are actually exercising Bug 1 (hyperedge in left gadget):
    row_weights = np.asarray(g_l.incidence.sum(axis=1)).ravel().astype(int).tolist()
    assert max(row_weights) >= 4, "Test no longer triggers Bug 1 (no hyperedge)"
    # Build bridge (this used to raise NotImplementedError or RuntimeError)
    bridge = build_bridge(g_l, g_r)
    # Merged code construction must succeed
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    # Intra-code joint Z̄_1 ⊗ Z̄_2: k_merged == k_orig − 1
    assert merged.dimension == code.dimension - 1
    # CSS commutation on merged code
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    HZ = np.asarray(merged.matrix_z).astype(np.int_)
    assert not ((HX @ HZ.T) % 2).any(), "CSS commutation broken on merged code"


def test_adapter_cycle_check_weight_bounded() -> None:
    """Each new cycle-X row has weight <= 8 (SkipTree (3,2) + H_R weight 2). Basis=Z.

    For basis=Z, the new adapter cycle checks are placed in HX (the last w-1 rows).
    Each row has the form [T_l | H_R | T_r]:
      - T_l row: at most 3 entries on cl_ancilla (SkipTree (3,2)-sparsity)
      - H_R row: exactly 2 entries on c_adapter (canonical rep code)
      - T_r row: at most 3 entries on cr_ancilla (SkipTree (3,2)-sparsity)
    Total: weight <= 3 + 2 + 3 = 8.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import _stitch_to_joint_csscode
    from qldpc.experimental.surgery.gadget import (
        build_gadget,
    )

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    # Use Z̄_1 for both sides (intra-code, same logical) — bridge.width =
    # |V_0| = weight of Z̄_1, exercising the maximum-width cellulation path
    x = _webster_z_bar_operator(data, "Z_bar_1")
    g_l = build_gadget(code, x, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)
    merged = _stitch_to_joint_csscode(g_l, g_r, bridge)
    HX = np.asarray(merged.matrix_x).astype(np.int_)
    # basis=Z: new cycle-X-checks are the last (w-1) rows of HX
    new_x_rows = HX[-(bridge.width - 1) :, :]
    max_w = int(new_x_rows.sum(axis=1).max())
    assert max_w <= 8, f"max new cycle-X weight {max_w} > 8"


def test_cellulation_caps_aug_aux_cycle_length_on_webster() -> None:
    """After cellulation, every basis cycle in the augmented aux graph has length <= 6."""
    import networkx as nx

    from qldpc.experimental.surgery.bridge import _build_aux_graph_strict, build_bridge
    from qldpc.experimental.surgery.gadget import (
        build_gadget,
    )

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_z_bar_operator(data, "Z_bar_1")
    g_l = build_gadget(code, x, basis=Pauli.Z)
    g_r = build_gadget(code, x, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r, cellulate_max_len=6)
    # Cellulation is now scoped to the port subgraph (where SkipTree runs).
    # Inspect cycles on the induced port subgraph, not the full graph.
    G_aux, _ = _build_aux_graph_strict(bridge.g_l_aug.incidence)
    sub = G_aux.subgraph(bridge.port_l)
    cycles = nx.cycle_basis(sub)
    if cycles:
        assert max(len(c) for c in cycles) <= 6, (
            f"max port-subgraph cycle length {max(len(c) for c in cycles)} > 6"
        )


def test_canonical_H_R_rejects_w_below_2() -> None:
    """_canonical_H_R(w=1) raises (rep-code needs w >= 2)."""
    from qldpc.experimental.surgery.bridge import _canonical_H_R

    with pytest.raises(ValueError, match="w >= 2"):
        _canonical_H_R(1)


def test_skip_tree_fullrank_defaults_edge_index_when_omitted() -> None:
    """_skip_tree_fullrank with edge_index_verts=None builds the default index dict.

    Built from S.edges() order — matches the explicit-dict path.
    """
    import networkx as nx

    from qldpc.experimental.surgery.bridge import _skip_tree_fullrank

    G_nx = nx.complete_graph(4)
    T_explicit, P_explicit = _skip_tree_fullrank(
        G_nx,
        root=0,
        edge_index_verts={tuple(sorted(e)): i for i, e in enumerate(G_nx.edges())},
    )
    T_default, P_default = _skip_tree_fullrank(G_nx, root=0)
    assert np.array_equal(T_default, T_explicit)
    assert np.array_equal(P_default, P_explicit)


def test_build_bridge_rejects_width_below_2() -> None:
    """build_bridge rejects port subsets that intersect to width < 2."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="width must be >= 2"):
        build_bridge(g, g, port_subset_l=(0,), port_subset_r=(0,))


def test_build_bridge_rejects_spanning_tree_root_out_of_range_left() -> None:
    """build_bridge rejects spanning_tree_root_l outside [0, width)."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="spanning_tree_root_l=99"):
        build_bridge(g, g, spanning_tree_root_l=99)


def test_build_bridge_rejects_spanning_tree_root_out_of_range_right() -> None:
    """build_bridge rejects spanning_tree_root_r outside [0, width)."""
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.gadget import build_gadget

    code = codes.SteaneCode()
    x = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    g = build_gadget(code, x, basis=Pauli.X)
    with pytest.raises(ValueError, match="spanning_tree_root_r=99"):
        build_bridge(g, g, spanning_tree_root_r=99)


def _bb_72_12() -> codes.BBCode:
    """Cain et al. arXiv:2603.28627 Table I `[[72, 12]]` BB code (cheeger h<1)."""
    import sympy

    xs, ys = sympy.symbols("x y")
    return codes.BBCode({xs: 6, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)


def test_build_bridge_skiptree_invariant_holds_after_boost() -> None:
    """T_s · F_aug · P_s = H_R must hold even when g_l/g_r are boosted.

    Regression: build_bridge rebuilds g_l_aug via _step1_restriction on the ORIGINAL (un-boosted)
    code+x+basis, dropping boost-added κ' rows from g_l.incidence. SkipTree T_l is computed against
    the boosted G_aux but embedded into unboosted g_l_aug.incidence → tree edges through boost-κ'
    are silently zeroed in T_full → invariant fails → joint_code cycle stabilizers are bogus →
    non-deterministic detector in joint PPM DEM.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.cheeger import boost_gadget
    from qldpc.experimental.surgery.gadget import build_gadget

    z = np.asarray(_bb_72_12().get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l_raw = build_gadget(_bb_72_12(), z, basis=Pauli.Z)
    g_r_raw = build_gadget(_bb_72_12(), z, basis=Pauli.Z)
    g_l = boost_gadget(g_l_raw, method="combinatorial", target=1.0, max_extra_qubits=20, seed=3)
    g_r = boost_gadget(g_r_raw, method="combinatorial", target=1.0, max_extra_qubits=20, seed=3)
    assert g_l.incidence.shape[0] > g_l_raw.incidence.shape[0], "boost should add κ' rows"
    bridge = build_bridge(g_l, g_r)

    for side in ("l", "r"):
        T = getattr(bridge, f"T_{side}")
        g_aug = getattr(bridge, f"g_{side}_aug")
        label = getattr(bridge, f"label_{side}")
        adj = g_aug.incidence.astype(np.int_)
        P = np.zeros((adj.shape[1], bridge.width), dtype=np.int_)
        for v_idx, lab in enumerate(label):
            if lab >= 0:
                P[v_idx, lab] = 1
        lhs = (T @ adj @ P) % 2
        assert np.array_equal(lhs, bridge.H_R), (
            f"side {side}: T·F_aug·P ≠ H_R after boost — bridge dropped boost κ' rows"
        )


def _bb_36_8() -> codes.BBCode:
    """BBCode (l=3, m=6) [[36, 8]] — has *duplicate* weight-2 incidence rows.

    When restricted to Z̄_0, this exercises _run_skiptree_on_port_subgraph's duplicate-edge guard.
    """
    import sympy

    xs, ys = sympy.symbols("x y")
    return codes.BBCode({xs: 3, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)


def test_build_bridge_skiptree_invariant_holds_with_duplicate_incidence_rows() -> None:
    """T_s · F_aug · P_s = H_R must hold when F_aug has duplicate weight-2 rows.

    Regression: BBCode [[36, 8]] restricted to Z̄_0 has h(F)=1 (no boost needed) but the restricted
    incidence has two κ rows sharing the same (u, v) support — _build_aux_graph_strict dedups them
    to one G_aux edge. Pre-fix, _run_skiptree_on_port_subgraph assigned the *same* T_relab column to
    both duplicate κ rows, so their contributions to T · F_aug cancel mod 2 → invariant fails →
    joint_code cycle stabilizer non-trivially anti-commutes with the gauge → non-deterministic
    detector.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.gadget import build_gadget

    code_l = _bb_36_8()
    code_r = _bb_36_8()
    z = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, z, basis=Pauli.Z)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    # Test premise: restricted incidence has duplicate rows
    inc = g_l.incidence.astype(np.int_)
    assert inc.shape[0] > np.unique(inc, axis=0).shape[0], (
        "test premise broken: BB [[36, 8]] restricted incidence should have duplicates"
    )
    bridge = build_bridge(g_l, g_r)

    for side in ("l", "r"):
        T = getattr(bridge, f"T_{side}")
        g_aug = getattr(bridge, f"g_{side}_aug")
        label = getattr(bridge, f"label_{side}")
        adj = g_aug.incidence.astype(np.int_)
        P = np.zeros((adj.shape[1], bridge.width), dtype=np.int_)
        for v_idx, lab in enumerate(label):
            if lab >= 0:
                P[v_idx, lab] = 1
        lhs = (T @ adj @ P) % 2
        assert np.array_equal(lhs, bridge.H_R), (
            f"side {side}: T·F_aug·P ≠ H_R with duplicate κ rows — bridge "
            f"duplicate-edge guard missing"
        )


def test_build_joint_ppm_circuit_dem_deterministic_bb_36_8() -> None:
    """Joint PPM DEM constructs without non-deterministic detectors on BB [[36, 8]].

    End-to-end regression for the duplicate-edge bug: BB [[36, 8]] Z̄⊗Z̄ joint PPM (h=1, no boost)
    previously crashed stim DEM with non-deterministic detectors because the SkipTree invariant
    failed on duplicate incidence rows.
    """
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.circuit import (
        build_joint_ppm_circuit,
        keep_only_observable,
    )
    from qldpc.experimental.surgery.gadget import build_gadget

    code_l, code_r = _bb_36_8(), _bb_36_8()
    z = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, z, basis=Pauli.Z)
    g_r = build_gadget(code_r, z, basis=Pauli.Z)
    bridge = build_bridge(g_l, g_r)

    noise = DepolarizingNoiseModel(1e-3, include_idling_error=False)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, noise_model=noise)
    stripped = keep_only_observable(circuit, keep_idx=0)
    dem = stripped.detector_error_model(approximate_disjoint_errors=True)
    assert dem.num_detectors > 0


def test_build_joint_ppm_circuit_dem_deterministic_after_boost_bb() -> None:
    """Joint PPM DEM must construct without non-deterministic detectors after boost.

    End-to-end regression: BB Z̄⊗Z̄ joint PPM with boost (required to reach
    Webster threshold h(F)≥1). Before fix, stim raised
    ``ValueError: The circuit contains non-deterministic detectors``
    because cycle stabilizers in joint_code didn't actually commute with
    the round-1 initial state.
    """
    from qldpc.circuits.noise_model import DepolarizingNoiseModel
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.cheeger import boost_gadget
    from qldpc.experimental.surgery.circuit import (
        build_joint_ppm_circuit,
        keep_only_observable,
    )
    from qldpc.experimental.surgery.gadget import build_gadget

    bb_l, bb_r = _bb_72_12(), _bb_72_12()
    z = np.asarray(bb_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    g_l = boost_gadget(
        build_gadget(bb_l, z, basis=Pauli.Z),
        method="combinatorial",
        target=1.0,
        max_extra_qubits=20,
        seed=3,
    )
    g_r = boost_gadget(
        build_gadget(bb_r, z, basis=Pauli.Z),
        method="combinatorial",
        target=1.0,
        max_extra_qubits=20,
        seed=3,
    )
    bridge = build_bridge(g_l, g_r)

    noise = DepolarizingNoiseModel(1e-3, include_idling_error=False)
    circuit, _ = build_joint_ppm_circuit(g_l, g_r, bridge, rounds=3, noise_model=noise)
    stripped = keep_only_observable(circuit, keep_idx=0)
    # raises ValueError("non-deterministic detectors") if the bug regressed
    dem = stripped.detector_error_model(approximate_disjoint_errors=True)
    assert dem.num_detectors > 0
