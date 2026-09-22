"""Standalone bridge adapter for two-PPM joint surgery.

Swaroop et al. arXiv:2410.03628 §IV / §VII. Handles both intra-code (g1.code is g2.code) and
inter-code joints.

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

import dataclasses
import itertools

import numpy as np

from qldpc._util import networkx as nx
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

from .gadget import GadgetLayout


@dataclasses.dataclass(frozen=True, eq=False)
class Bridge:
    """Universal adapter between two GadgetLayouts (Swaroop et al. arXiv:2410.03628 §IV / §VII).

    gadget notation: V_0 → support; F → incidence; κ → ancilla.

    Attributes follow the universal-adapter construction of Swaroop et al. arXiv:2410.03628.
    """

    width: int  # w = |𝒜| (adapter qubits)
    basis: PauliXZ  # X or Z (symmetric dual)
    port_l: tuple[int, ...]  # 𝒫_l* ⊆ V_0^(l), length w
    port_r: tuple[int, ...]  # 𝒫_r* ⊆ V_0^(r), length w
    label_l: tuple[int, ...]  # label_l[i] = SkipTree label of V_0^(l)[i]; -1 if i ∉ 𝒫_l*
    label_r: tuple[int, ...]
    extra_ancilla_l: np.ndarray  # (e_l, |support^(l)|) F_2; weight-2 rows added
    extra_ancilla_r: np.ndarray
    T_l: np.ndarray  # (w-1, |C_0^(l)| + e_l) F_2 (3,2)-sparse
    T_r: np.ndarray
    H_R: np.ndarray  # (w-1, w) canonical rep code parity
    g_l_aug: GadgetLayout  # gadget rebuilt over F_aug^(l)
    g_r_aug: GadgetLayout


def _skip_tree(
    S: nx.Graph,
    root: int = 0,
    edge_index_verts: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """SkipTree basis transform (Swaroop et al. arXiv:2410.03628 §III). Returns T, P."""
    n = S.number_of_nodes()
    index = 0
    label = [0] * n
    visited: set[int] = set()

    def label_first(v: int, skip: bool) -> None:
        nonlocal index
        visited.add(v)
        label[index] = v
        index = index + 1
        children = [nbr for nbr in S.neighbors(v) if nbr not in visited]
        for child_idx, child in enumerate(children):
            last_in_gen = child_idx == len(children) - 1
            if last_in_gen and not skip:
                label_first(child, skip=False)
            else:
                label_last(child)

    def label_last(v: int) -> None:
        nonlocal index
        visited.add(v)
        for child in S.neighbors(v):
            if child not in visited:
                label_first(child, skip=True)
        label[index] = v
        index = index + 1

    label_first(root, skip=False)

    P = np.zeros((n, n), dtype=np.int_)
    for l_idx, v in enumerate(label):
        P[v, l_idx] = 1

    if not edge_index_verts:
        edge_index_verts = {tuple(sorted(e)): i for i, e in enumerate(S.edges())}

    T = np.zeros((n - 1, len(edge_index_verts)), dtype=np.int_)
    for l_idx in range(n - 1):
        path = nx.shortest_path(S, source=label[l_idx], target=label[(l_idx + 1) % n])
        for u, v in itertools.pairwise(path):
            e = tuple(sorted((u, v)))
            T[l_idx, edge_index_verts[e]] = 1
    return T, P


def _canonical_H_R(w: int) -> np.ndarray:
    """Full-rank canonical rep-code parity check matrix, shape (w-1) × w.

    Row i has 1s in columns i and i+1. rank == w-1; column 0 and column w-1 have weight 1, other
    columns weight 2.
    """
    if w < 2:
        raise ValueError(f"H_R requires w >= 2, got {w}")
    H = np.zeros((w - 1, w), dtype=np.int_)
    for i in range(w - 1):
        H[i, i] = 1
        H[i, i + 1] = 1
    return H


def _skip_tree_fullrank(
    S: nx.Graph,
    root: int = 0,
    edge_index_verts: dict[tuple[int, int], int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SkipTree (T, P) satisfying T · G · P == H_R (full-rank rep code).

    Swaroop et al. arXiv:2410.03628 Algorithm 1 takes a connected graph and computes a spanning tree
    as its first step, then reads each T row off the tree path between consecutively labelled
    vertices. Here the spanning tree supplies the DFS vertex labeling, but each T row is the XOR of
    shortest-path edges in the full graph S. This lets S be any connected graph; the direct
    _skip_tree call would IndexError on cyclic inputs.

    Sparsity: row weight ≤ 3 still holds, since a shortest path in S is no longer than the tree path
    between the same endpoints, which Theorem 7 bounds at 3 edges. Theorem 7's column-weight-2 half
    does NOT carry over: its proof counts each *tree* edge's reuse via sub-tree exhaustion, and a
    full-graph path may route over non-tree edges for which the paper gives no reuse bound. Column
    weight ≤ 2 is checked empirically instead, in
    ``bridge_test.py::test_skip_tree_fullrank_on_K4_matches_H_R``.

    Returns (T, P) of shapes (n-1, |E|) and (n, n).
    """
    n = S.number_of_nodes()
    span = nx.minimum_spanning_tree(S)
    _, P = _skip_tree(span, root=root, edge_index_verts=None)
    # P is a permutation matrix from _skip_tree; recover label[ell] = v such that P[v, ell] = 1.
    label = [-1] * n
    for v in range(n):
        for ell in range(n):
            if P[v, ell] == 1:
                label[ell] = v
                break

    if edge_index_verts is None:
        edge_index_verts = {tuple(sorted(e)): i for i, e in enumerate(S.edges())}

    T = np.zeros((n - 1, len(edge_index_verts)), dtype=np.int_)
    for l_idx in range(n - 1):
        path = nx.shortest_path(S, source=label[l_idx], target=label[l_idx + 1])
        for u, v in itertools.pairwise(path):
            e = tuple(sorted((u, v)))
            T[l_idx, edge_index_verts[e]] ^= 1  # XOR cancels back-and-forth
    return T.astype(np.int_), P.astype(np.int_)


def _cellulate_port_subgraph(
    G_aux: nx.Graph,
    ports: tuple[int, ...],
    *,
    max_len: int,
) -> list[tuple[int, int]]:
    """Break port-subgraph cycles longer than ``max_len`` by adding chords.

    SkipTree runs on G_aux.subgraph(ports); cycles entirely outside the port subgraph never enter
    T_s, so we cellulate only there.

    T_s row weight is already ≤ 3 regardless of cycle length (see _skip_tree_fullrank), so this step
    is not load-bearing for correctness. Capping basis cycle length is the cellulation of Swaroop et
    al. arXiv:2410.03628 §II.3 — their desideratum 3a, distinct from the decongestion of their
    Lemma 6, which instead bounds how many basis cycles a single edge belongs to.

    The scope is the port subgraph because that is what SkipTree sees. When |V_0| > w, a long cycle
    of G_aux can thread non-port vertices; such a cycle contributes no basis cycle to the port
    subgraph, and it need not admit a port-port chord at all.

    Chords are added to ``G_aux`` (the full graph). For port-subgraph cycles, chord endpoints are
    necessarily ports (cycle vertices = port vertices), so no port-membership filter is needed in
    the chord-search loop.

    Returns the list of added (u, v) edges in insertion order. Idempotent once all port-subgraph
    basis cycles fit under the cap.
    """
    added: list[tuple[int, int]] = []
    while True:
        sub = G_aux.subgraph(ports)
        long_cycles = [c for c in nx.cycle_basis(sub) if len(c) > max_len]
        if not long_cycles:
            return added
        cycle = long_cycles[0]
        n = len(cycle)
        chord_found = False
        for i in range(n):
            if chord_found:
                break
            for j in range(i + 2, n):
                u, v = sorted((cycle[i], cycle[j]))
                if G_aux.has_edge(u, v):
                    continue
                G_aux.add_edge(u, v)
                added.append((u, v))
                chord_found = True
                break
        if not chord_found:
            raise RuntimeError(
                f"No chord found to cellulate port-subgraph cycle of length {n}; "
                f"ports={ports!r}, cycle={cycle!r}"
            )


def _build_aux_graph_strict(incidence: np.ndarray) -> tuple[nx.Graph, dict[tuple[int, int], int]]:
    """Build auxiliary graph from F; weight-2 rows become edges, hyperedges are skipped.

    Vertices: range(|V_0|) = range(F.shape[1]).
    Edges: one per weight-2 row of F, between the two columns where the row has 1s.

    Why skipping hyperedges is safe — paired with the guard at `_run_skiptree_on_port_subgraph`
    (search for `if len(cols) != 2: continue`), which assigns T_s zero columns on those same rows.
    So
        (T_s · F_aug)[c, v] = Σ_{k: weight-2 row} T_s[c,k] · F_aug[k, v]
                            = H_R[c, label(v)] · [v ∈ port]      (SkipTree identity)
    and the hyperedge rows contribute 0 regardless of F_aug[r, v]. χ_v · cycle_c on the κ side
    cancels the adapter side, CSS commutation holds. The hyperedge κ qubit itself stays in F_aug,
    so the gadget (G_aug = ker(F_aug^T), deformed check c → c · X(κ_r), χ_v) is untouched. That
    commutation conclusion is what
    ``bridge_test.py::test_build_bridge_bb18_hyperedge_and_long_cycle`` checks on a fixture
    carrying a weight-4 F row.

    Paper Eq. (9)'s perfect-matching decomposition (§II.3) is not applied, and no structural
    distance argument is claimed for the joint merge. Swaroop et al. Thm 11 (§IV) does give one for
    the adapter — the deformed code for the joint measurement keeps distance d — but only if the
    individual deformed codes are LDPC with distance d, which this library does not establish. The
    only empirical evidence here is the LER trend in
    ``circuit_test.py::test_joint_ppm_ler_monotone_steane_intercode``, whose Steane fixture has
    all-weight-2 F rows and so does not exercise the hyperedge path.

    Raises:
        ValueError: if any row of F has weight 1 (defensive — F · 1_{V_0} = 0 mod 2 forbids odd
        weights for any valid logical x with H · x = 0).
    """
    incidence_arr = np.asarray(incidence).astype(int)
    G = nx.Graph()
    G.add_nodes_from(range(incidence_arr.shape[1]))
    edge_index: dict[tuple[int, int], int] = {}
    for i, row in enumerate(incidence_arr):
        eps = np.flatnonzero(row).tolist()
        if len(eps) == 0 or len(eps) >= 3:
            continue
        if len(eps) == 1:
            raise ValueError(
                f"F row {i} has weight 1 (column {eps[0]}). "
                f"Auxiliary-graph edges require exactly 2 endpoints "
                f"(F · 1 = 0 mod 2 forbids odd weights — invalid logical?)."
            )
        u, v = sorted(eps)
        if (u, v) not in edge_index:
            edge_index[(u, v)] = len(edge_index)
            G.add_edge(u, v)
    return G, edge_index


def _connect_induced_subgraph(
    G_aux: nx.Graph,
    ports: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Add edges to G_aux so that G_aux.subgraph(ports) is connected.

    Mutates G_aux. Each added edge has both endpoints in ``ports`` so it contributes a weight-2 row
    to the augmented F matrix downstream.

    Loop invariant: u and v are drawn from different components of G_aux.subgraph(ports), so G_aux
    cannot already have a (u, v) edge — such an edge would put them in the same component.

    Returns the list of added edges in insertion order.
    """
    added: list[tuple[int, int]] = []
    while True:
        comps = list(nx.connected_components(G_aux.subgraph(ports)))
        if len(comps) <= 1:
            return added
        u, v = sorted((min(comps[0]), min(comps[1])))
        G_aux.add_edge(u, v)
        added.append((u, v))


def _edges_to_incidence_extra(edges: list[tuple[int, int]], n_V0: int) -> np.ndarray:
    """Convert a list of weight-2 (u, v) edges into a (|edges|, n_V0) F_2 matrix."""
    out = np.zeros((len(edges), n_V0), dtype=np.uint8)
    for r, (u, v) in enumerate(edges):
        out[r, u] = 1
        out[r, v] = 1
    return out


def _run_skiptree_on_port_subgraph(
    G_aux_full: nx.Graph,
    port: tuple[int, ...],
    root_port_idx: int,
    incidence_aug: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    """Run SkipTree on the induced port subgraph; embed result back onto F_aug rows.

    The induced subgraph's vertex IDs are relabeled to [0, |port|) so the n×n P allocation inside
    ``_skip_tree`` is square. The output T is then re-expressed onto the original F_aug edge
    ordering (rows of F_aug index the κ qubits = edges of G_aux_full). ``root_port_idx`` selects
    which entry of ``port`` is the SkipTree root.

    Returns (T_full, labels) where T_full has shape (w-1, F_aug.shape[0]) and labels[orig_v] = k
    iff orig_v ∈ port and got SkipTree label k (else -1).
    """
    sub_orig = G_aux_full.subgraph(port).copy()
    port_sorted = sorted(port)
    new_of_orig = {orig: new for new, orig in enumerate(port_sorted)}
    orig_of_new = {new: orig for orig, new in new_of_orig.items()}
    sub_relab = nx.relabel_nodes(sub_orig, new_of_orig, copy=True)
    # Take a spanning tree, as Algorithm 1 of the paper does at its first step. MST is
    # deterministic; for unweighted graphs nx returns a BFS-like tree.
    sub_tree = nx.minimum_spanning_tree(sub_relab)
    tree_edges = sorted(tuple(sorted(e)) for e in sub_tree.edges())
    edge_idx_tree = {e: i for i, e in enumerate(tree_edges)}
    root_orig = port[root_port_idx]
    root_relab = new_of_orig[root_orig]
    T_relab, P_relab = _skip_tree_fullrank(
        sub_tree,
        root=root_relab,
        edge_index_verts=edge_idx_tree,
    )
    labels = [-1] * incidence_aug.shape[1]
    for new_v in range(len(port)):
        orig_v = orig_of_new[new_v]
        nz = np.flatnonzero(P_relab[new_v])
        assert len(nz) == 1, f"vertex {orig_v} (relab {new_v}) has {len(nz)} labels"
        labels[orig_v] = int(nz[0])
    T_full = np.zeros((T_relab.shape[0], incidence_aug.shape[0]), dtype=np.int_)
    # Duplicate-edge guard: when two κ rows of F_aug share the same (u, v)
    # support (parallel edges in the *strict* aux graph), _build_aux_graph_strict
    # dedups them to one G_aux edge. Without this guard, both duplicate rows
    # would receive the same T_relab column → their contributions to T·F_aug
    # cancel mod 2 → SkipTree identity fails on codes like BB [[36, 8]] whose
    # restricted incidence has duplicate weight-2 rows. Assigning T only to the
    # FIRST matching row preserves T·F_aug = H_R (duplicate κ qubits remain in
    # the gauge group, untouched by the cycle).
    assigned_edges: set[tuple[int, int]] = set()
    for r in range(incidence_aug.shape[0]):
        cols = np.flatnonzero(incidence_aug[r])
        # Load-bearing skip: T_s gets zero columns on hyperedge rows (weight ≥ 3)
        # and on rows whose endpoints are outside the port subgraph. Paired with
        # _build_aux_graph_strict's matching skip; together they make hyperedge κ
        # qubits invisible to (T_s · F_aug), so χ_v · cycle_c commutation reduces
        # to the weight-2 sub-incidence SkipTree identity. See bridge.py docstring
        # for _build_aux_graph_strict for the proof sketch.
        if len(cols) != 2:
            continue
        u_orig, v_orig = sorted(int(x) for x in cols)
        if u_orig not in new_of_orig or v_orig not in new_of_orig:
            continue
        lo, hi = sorted((new_of_orig[u_orig], new_of_orig[v_orig]))
        e_relab = (lo, hi)
        if e_relab in edge_idx_tree and e_relab not in assigned_edges:
            T_full[:, r] = T_relab[:, edge_idx_tree[e_relab]]
            assigned_edges.add(e_relab)
    return T_full.astype(np.int_), labels


def _max_basis_stabilizer_weight(code: CSSCode, basis: PauliXZ) -> int:
    """Largest row weight of the data code's check matrix in the measured basis."""
    H = code.matrix_x if basis is Pauli.X else code.matrix_z
    return int(np.asarray(H).astype(int).sum(axis=1).max())


def build_bridge(
    g_l: GadgetLayout,
    g_r: GadgetLayout,
    *,
    port_subset_l: tuple[int, ...] | None = None,
    port_subset_r: tuple[int, ...] | None = None,
    spanning_tree_root_l: int = 0,
    spanning_tree_root_r: int = 0,
    cellulate_max_len: int | None = None,
) -> Bridge:
    """Universal-adapter bridge between two gadgets (Swaroop et al. arXiv:2410.03628 §IV).

    gadget notation: V_0^(l) → support^(l); F → incidence; extra_kappa → extra_ancilla.

    Implements the repetition-code adapter of Swaroop et al. arXiv:2410.03628 §IV, built on their
    SkipTree basis transform (§III).

    Args:
        g_l: left gadget.
        g_r: right gadget. Must share g_l's measurement basis.
        port_subset_l: indices into ``g_l.support`` to use as ports. Must be in range and distinct;
            out-of-range or repeated entries surface as an IndexError. Defaults to all of it.
        port_subset_r: indices into ``g_r.support`` to use as ports, same contract as port_subset_l.
            Defaults to all of it.
        spanning_tree_root_l: index INTO the left port tuple of the SkipTree root vertex.
        spanning_tree_root_r: index INTO the right port tuple of the SkipTree root vertex.
        cellulate_max_len: cap on port-subgraph cycle length, enforced by adding chords. Defaults to
            the larger of the two data codes' maximum measured-basis stabilizer row weight, so that
            the cap follows the code rather than a constant. Lowering it trades qubits for a sparser
            gauge: on the bb_18 intra-code pair, whose default is 6, capping at 6 cuts max gauge row
            weight from 34 to 13 at a cost of +314 qubits.

    Returns:
        A Bridge of width min(|ports_l|, |ports_r|).

    Raises:
        ValueError: the two gadgets disagree on basis; the resulting width is < 2; a spanning-tree
            root is out of range; or an explicitly supplied port subset is longer than the width, so
            that requested ports would be dropped.
        RuntimeError: a port-subgraph cycle has no chord available to meet cellulate_max_len, which
            a cap below 3 forces.
    """
    if g_l.basis is not g_r.basis:
        raise ValueError(
            f"build_bridge requires g_l.basis == g_r.basis, got {g_l.basis!r} vs {g_r.basis!r}"
        )
    basis = g_l.basis
    if cellulate_max_len is None:
        cellulate_max_len = max(
            _max_basis_stabilizer_weight(g_l.code, basis),
            _max_basis_stabilizer_weight(g_r.code, basis),
        )

    # Step 1: auxiliary graphs
    G_l_aux, _ = _build_aux_graph_strict(g_l.incidence)
    G_r_aux, _ = _build_aux_graph_strict(g_r.incidence)

    # Step 2: port subsets + width
    port_l_all = (
        tuple(port_subset_l) if port_subset_l is not None else tuple(range(len(g_l.support)))
    )
    port_r_all = (
        tuple(port_subset_r) if port_subset_r is not None else tuple(range(len(g_r.support)))
    )
    width = min(len(port_l_all), len(port_r_all))
    if width < 2:
        raise ValueError(f"bridge width must be >= 2, got {width}")
    # Truncating to the narrower side is the intended adapter behaviour when both port tuples are
    # defaulted — supports of unequal size bridge at their minimum. A named port that would be
    # truncated away is a request the caller made and cannot get, so it is an error.
    for side, subset, ports in (("l", port_subset_l, port_l_all), ("r", port_subset_r, port_r_all)):
        if subset is not None and len(ports) > width:
            raise ValueError(
                f"port_subset_{side} lists {len(ports)} ports but the bridge width is {width}, so "
                f"{len(ports) - width} requested port(s) would be dropped. Supply port subsets of "
                f"equal length, or leave both unset to bridge at min(|support_l|, |support_r|)."
            )
    port_l = port_l_all[:width]
    port_r = port_r_all[:width]
    if not (0 <= spanning_tree_root_l < width):
        raise ValueError(f"spanning_tree_root_l={spanning_tree_root_l} out of [0, {width})")
    if not (0 <= spanning_tree_root_r < width):
        raise ValueError(f"spanning_tree_root_r={spanning_tree_root_r} out of [0, {width})")

    # Step 3: induced-subgraph connectivity augmentation
    extras_l_conn = _connect_induced_subgraph(G_l_aux, port_l)
    extras_r_conn = _connect_induced_subgraph(G_r_aux, port_r)

    # Step 4: cellulation
    extras_l_cell = _cellulate_port_subgraph(G_l_aux, port_l, max_len=cellulate_max_len)
    extras_r_cell = _cellulate_port_subgraph(G_r_aux, port_r, max_len=cellulate_max_len)

    extras_l_edges = extras_l_conn + extras_l_cell
    extras_r_edges = extras_r_conn + extras_r_cell
    extra_ancilla_l = _edges_to_incidence_extra(extras_l_edges, len(g_l.support))
    extra_ancilla_r = _edges_to_incidence_extra(extras_r_edges, len(g_r.support))

    # Rebuild the augmented gadgets before SkipTree so that g_l_aug.incidence can be threaded in as
    # its column space. This ordering is safe because F_aug.shape[0] is fixed by extra_ancilla_*,
    # not by SkipTree.
    from .gadget import _build_gadget_augmented, _step1_restriction

    # boost_gadget appends weight-2 κ' rows to g_l.incidence beyond the original
    # _step1_restriction output. These rows must be preserved when assembling
    # g_l_aug — SkipTree runs against G_aux (built from boosted g_l.incidence)
    # but T_full is embedded into g_l_aug.incidence; dropping boost rows leaves
    # tree edges through boost-κ' silently zeroed and breaks the invariant
    # T_s · F_aug · P_s = H_R.
    _, _, _orig_inc_l = _step1_restriction(g_l.code, g_l.x, basis=basis)
    _, _, _orig_inc_r = _step1_restriction(g_r.code, g_r.x, basis=basis)
    boost_extras_l = g_l.incidence[_orig_inc_l.shape[0] :].astype(np.uint8)
    boost_extras_r = g_r.incidence[_orig_inc_r.shape[0] :].astype(np.uint8)
    combined_extras_l = np.vstack([boost_extras_l, extra_ancilla_l.astype(np.uint8)])
    combined_extras_r = np.vstack([boost_extras_r, extra_ancilla_r.astype(np.uint8)])

    g_l_aug = _build_gadget_augmented(g_l.code, g_l.x, combined_extras_l, basis=basis)
    g_r_aug = _build_gadget_augmented(g_r.code, g_r.x, combined_extras_r, basis=basis)

    # Step 5: SkipTree on induced port subgraph; embed back into full F_aug rows
    T_l, label_l = _run_skiptree_on_port_subgraph(
        G_l_aux,
        port_l,
        spanning_tree_root_l,
        g_l_aug.incidence,
    )
    T_r, label_r = _run_skiptree_on_port_subgraph(
        G_r_aux,
        port_r,
        spanning_tree_root_r,
        g_r_aug.incidence,
    )

    return Bridge(
        width=width,
        basis=basis,
        port_l=port_l,
        port_r=port_r,
        label_l=tuple(label_l),
        label_r=tuple(label_r),
        extra_ancilla_l=extra_ancilla_l.astype(np.uint8),
        extra_ancilla_r=extra_ancilla_r.astype(np.uint8),
        T_l=T_l,
        T_r=T_r,
        H_R=_canonical_H_R(width).astype(np.int_),
        g_l_aug=g_l_aug,
        g_r_aug=g_r_aug,
    )
