"""Tests for src/qldpc/experimental/surgery/bridge.py.

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
from qldpc.objects import Pauli, PauliXZ

from .conftest import (
    _webster_x_bar_operator,
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
    # Row weight ≤ 3 follows from Swaroop et al. arXiv:2410.03628 Thm 7 even with full-graph paths;
    # column weight ≤ 2 does not (see _skip_tree_fullrank), so K_4 checks it directly.
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
    # Hyperedge would have produced edges (0,1), (0,2), (0,3), (1,2), (1,3), (2,3) but only edges
    # from weight-2 rows are present
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

    # A 7-cycle saturated with every chord -- i.e. K_7 -- so no chord can be added.
    G = nx.cycle_graph(7)
    ports = tuple(range(7))
    for i in range(7):
        for j in range(i + 2, 7):
            if not G.has_edge(i, j) and (i, j) != (0, 6):
                G.add_edge(i, j)
    # The saturated cycle basis is all triangles, so max_len=6 would find no long cycle; max_len=2
    # forces the failure path.
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
    """End-to-end: Cain bb_18 exercises both a hyperedge in F and a long port-subgraph cycle.

    build_bridge must succeed and produce a merged code with k_merged = k_orig - 1 (intra-code joint
    Z̄_1 ⊗ Z̄_2).

    Two *distinct* Z-logicals are used so that the joint measurement reduces k by exactly 1.
    Z-logical 0 has a weight-4 F row, which is the hyperedge case; the pair together drives the full
    _cellulate_port_subgraph path.
    """
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
    z0 = np.asarray(z_ops[0]).astype(np.uint8)  # hyperedge logical (weight-4 F row)
    z1 = np.asarray(z_ops[1]).astype(np.uint8)  # distinct second logical
    g_l = build_gadget(code, z0, basis=Pauli.Z)
    g_r = build_gadget(code, z1, basis=Pauli.Z)
    # Confirm the left gadget really does carry a hyperedge:
    row_weights = np.asarray(g_l.incidence.sum(axis=1)).ravel().astype(int).tolist()
    assert max(row_weights) >= 4, "fixture must carry a hyperedge (F row of weight >= 4)"
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
    """Each new cycle-X row has weight <= 8 (SkipTree (3,2) + H_R weight 2).  Basis=Z.

    For basis=Z the new adapter cycle checks are the last w-1 rows of HX, each of the form [T_l |
    T_r | H_R]: at most 3 entries from each SkipTree block and exactly 2 from H_R.
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
    # Cellulation is scoped to the port subgraph, where SkipTree runs, so the cap applies to cycles
    # of the induced port subgraph rather than of the full graph.
    G_aux, _ = _build_aux_graph_strict(bridge.g_l_aug.incidence)
    sub = G_aux.subgraph(bridge.port_l)
    cycles = nx.cycle_basis(sub)
    if cycles:
        assert max(len(c) for c in cycles) <= 6, (
            f"max port-subgraph cycle length {max(len(c) for c in cycles)} > 6"
        )


@pytest.mark.parametrize("basis", [Pauli.X, Pauli.Z])
def test_cellulate_max_len_defaults_to_the_max_basis_stabilizer_weight(
    basis: PauliXZ, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default cap is the larger of the two data codes' measured-basis max row weights.

    The hypergraph-product code is not self-dual -- 5 in X, 6 in Z -- while Steane is 4 in either,
    and the bases put the heavier code on opposite sides, so the resolved cap separates reading the
    complementary matrix, taking the smaller side, and reading one side alone.  The floor of 3 is
    inactive at these weights and has its own test.
    """
    from typing import Any

    from qldpc.experimental.surgery import bridge as bridge_module
    from qldpc.experimental.surgery.gadget import build_gadget

    hgp = codes.HGPCode(codes.RepetitionCode(3), codes.HammingCode(3))
    steane = codes.SteaneCode()
    assert bridge_module._max_basis_stabilizer_weight(hgp, Pauli.X) == 5
    assert bridge_module._max_basis_stabilizer_weight(hgp, Pauli.Z) == 6
    assert bridge_module._max_basis_stabilizer_weight(steane, basis) == 4

    # Heavier code on the left in the X basis, on the right in the Z basis.
    code_l, code_r = (hgp, steane) if basis is Pauli.X else (steane, hgp)
    x_l = np.asarray(code_l.get_logical_ops(basis)[0]).astype(np.uint8)
    x_r = np.asarray(code_r.get_logical_ops(basis)[0]).astype(np.uint8)
    g_l = build_gadget(code_l, x_l, basis=basis)
    g_r = build_gadget(code_r, x_r, basis=basis)

    real_cellulate = bridge_module._cellulate_port_subgraph
    seen: list[int] = []

    def _record(G_aux: Any, ports: tuple[int, ...], *, max_len: int) -> list[tuple[int, int]]:
        seen.append(max_len)
        return real_cellulate(G_aux, ports, max_len=max_len)

    monkeypatch.setattr(bridge_module, "_cellulate_port_subgraph", _record)

    expected = 5 if basis is Pauli.X else 6
    bridge_module.build_bridge(g_l, g_r)
    assert seen == [expected, expected]

    seen.clear()
    bridge_module.build_bridge(g_l, g_r, cellulate_max_len=7)
    assert seen == [7, 7]


def test_cellulate_max_len_default_is_floored_at_3(monkeypatch: pytest.MonkeyPatch) -> None:
    """The derived default never drops below 3, which every port subgraph can meet.

    Every H_X row of this [[6,1]] code has weight 2, so the cap read off the code is 2 -- shorter
    than the 3-cycle its port subgraph carries, and a 3-cycle has no chord to split.
    """
    from typing import Any

    from qldpc.experimental.surgery import bridge as bridge_module
    from qldpc.experimental.surgery.gadget import build_gadget

    matrix_x = np.array([[0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 1, 0]])
    matrix_z = np.array([[0, 1, 1, 1, 1, 0], [1, 1, 1, 0, 1, 1], [1, 0, 0, 1, 0, 1]])
    code = codes.CSSCode(matrix_x, matrix_z)
    assert bridge_module._max_basis_stabilizer_weight(code, Pauli.X) == 2

    x = np.array([0, 0, 0, 1, 1, 1], dtype=np.uint8)
    gadget = build_gadget(code, x, basis=Pauli.X)
    assert len(gadget.support) == 3

    real_cellulate = bridge_module._cellulate_port_subgraph
    seen: list[int] = []

    def _record(G_aux: Any, ports: tuple[int, ...], *, max_len: int) -> list[tuple[int, int]]:
        seen.append(max_len)
        return real_cellulate(G_aux, ports, max_len=max_len)

    monkeypatch.setattr(bridge_module, "_cellulate_port_subgraph", _record)

    bridge = bridge_module.build_bridge(gadget, gadget)
    assert seen == [3, 3]
    assert bridge.width == 3


def test_cellulate_max_len_default_reads_an_empty_measured_basis_as_zero() -> None:
    """A code with no measured-basis checks contributes 0 to the derived cap, not an error.

    Only the complementary basis feeds the auxiliary graph the cap governs, so such a code bridges
    normally -- on the floor of 3.
    """
    from qldpc.experimental.surgery import bridge as bridge_module
    from qldpc.experimental.surgery.gadget import build_gadget

    matrix_z = np.array([[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]])
    code = codes.CSSCode(np.zeros((0, 6), dtype=int), matrix_z)
    assert bridge_module._max_basis_stabilizer_weight(code, Pauli.X) == 0

    x = np.asarray(code.get_logical_ops(Pauli.X)[2]).astype(np.uint8)
    gadget = build_gadget(code, x, basis=Pauli.X)
    bridge = bridge_module.build_bridge(gadget, gadget)
    assert bridge.width == len(gadget.support)


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
    """Bravyi et al. arXiv:2308.07915 `[[72, 12]]` bivariate-bicycle code (cheeger h<1)."""
    import sympy

    xs, ys = sympy.symbols("x y")
    return codes.BBCode({xs: 6, ys: 6}, xs**3 + ys + ys**2, ys**3 + xs + xs**2)


def test_build_bridge_skiptree_invariant_holds_after_boost() -> None:
    """T_s · F_aug · P_s = H_R must hold even when g_l/g_r are boosted.

    The boost adds κ' rows to g.incidence, so g_l_aug must be rebuilt from the boosted incidence:
    SkipTree computes T_l against the boosted G_aux, and embedding it into an unboosted
    g_l_aug.incidence would zero the tree edges running through boost-κ'.  That breaks the
    invariant, which makes the joint code's cycle stabilizers wrong and leaves a non-deterministic
    detector in the joint PPM DEM.
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

    BBCode [[36, 8]] restricted to Z̄_0 has h(F)=1 (no boost needed) but the restricted incidence
    has two κ rows sharing the same (u, v) support, which _build_aux_graph_strict dedups to one
    G_aux edge.  Each duplicate κ row therefore needs its own T_relab column: sharing one column
    makes their contributions to T · F_aug cancel mod 2, which breaks the invariant, makes the joint
    code's cycle stabilizer anti-commute with the gauge, and leaves a non-deterministic detector.
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

    Duplicate incidence rows are the stressor: if the SkipTree invariant fails on them, stim reports
    non-deterministic detectors.  BB [[36, 8]] Z̄⊗Z̄ joint PPM at h=1 (no boost) has such rows.
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

    BB Z̄⊗Z̄ joint PPM with the boost needed to reach the Webster threshold h(F) >= 1.  If the cycle
    stabilizers in joint_code do not commute with the round-1 initial state, stim raises
    ``ValueError: The circuit contains non-deterministic detectors``.
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


@pytest.mark.parametrize("oversized_side", ["l", "r"])
def test_build_bridge_rejects_oversized_explicit_port_subset(oversized_side: str) -> None:
    """An explicit port subset longer than the bridge width is rejected, not silently truncated.

    Truncating to the narrower side is intended when both subsets are defaulted, but dropping ports
    the caller named is a silent change of request.  Each side carries its own check, so both are
    exercised.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.gadget import build_gadget

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    x = _webster_x_bar_operator(data)
    g = build_gadget(code, x, basis=Pauli.X)
    n_ports = len(g.support)
    assert n_ports >= 4, f"fixture needs >= 4 ports, got {n_ports}"

    full = tuple(range(n_ports))
    short = tuple(range(n_ports - 1))
    subset_l, subset_r = (full, short) if oversized_side == "l" else (short, full)
    with pytest.raises(ValueError, match=f"port_subset_{oversized_side} lists"):
        build_bridge(g, g, port_subset_l=subset_l, port_subset_r=subset_r)


def test_build_bridge_accepts_explicit_port_subsets_of_equal_length() -> None:
    """Explicit port subsets exactly as long as the width are accepted.

    Only a subset that would lose a named port is rejected, so the equal-length case — the remedy
    the error message recommends — has to go through.
    """
    from qldpc.experimental.surgery.bridge import build_bridge
    from qldpc.experimental.surgery.gadget import build_gadget

    data = load_webster_seed_set(0)
    code = build_generalised_bicycle_code(data["l"], data["A"], data["B"])
    g = build_gadget(code, _webster_x_bar_operator(data), basis=Pauli.X)
    ports = tuple(range(len(g.support)))

    bridge = build_bridge(g, g, port_subset_l=ports, port_subset_r=ports)
    assert bridge.width == len(ports)
