"""Cheeger and distance boost transformations for surgery gadgets.

References:
    Webster, Smith, Cohen arXiv:2511.15989 — boundary Cheeger constant, combinatorial boost (§II.1).
    Cross et al. arXiv:2407.18393 — Cheeger-based distance preservation (§3.3 Thm 6).
    Williamson & Yoder arXiv:2410.02213 — distance-verifying random augmentation boost.

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

from typing import Any

import galois
import numpy as np

from qldpc.codes.common import CSSCode

from .gadget import GadgetLayout


def _exact_boundary_cheeger(incidence: galois.FieldArray) -> tuple[float, np.ndarray]:
    """Exact boundary Cheeger constant of F per Webster §II.1 Definition 1.

    gadget notation: V → support; C → rows of incidence; F → incidence.

    Backs ``cheeger_constant``, additionally returning the cut that attains h(F).

    For bipartite incidence F: V -> C, the boundary ∂v of v ⊆ V is the subset of C with an odd
    number of neighbours in v, and h(F) = min |∂v| / |v| over the subsets with 1 ≤ |v| ≤ |V|/2.

    Computed by Gray-code enumeration over all subsets v, as a pure-Python loop over bit-packed
    columns; cost quadruples per additional 2 columns, which is what puts the ceiling at |V| = 26.

    Args:
        incidence: GF(2) restriction matrix of shape (|C|, |V|).

    Returns:
        (h, v_star_indicator) where v_star_indicator is a length-|V| binary numpy array marking the
        worst cut. If |V| < 2, returns (inf, zero vector) — boost is not applicable.

    Raises:
        ValueError: if |V| > 26 (exhaustive enumeration is infeasible).
    """
    incidence_arr = np.asarray(incidence).astype(np.int8)
    _n_C, n_V = incidence_arr.shape
    if n_V < 2:
        return float("inf"), np.zeros(n_V, dtype=np.int8)
    if n_V > 26:
        raise ValueError(
            f"_exact_boundary_cheeger requires |V| ≤ 26; got |V|={n_V}. "
            f"Exact boundary-Cheeger enumeration is infeasible beyond this size."
        )

    # Bit-pack F columns: incidence_col_ints[i] is a Python int with bit r set iff
    # F[r, i] = 1. Boundary as Python int allows O(1) XOR + popcount.
    incidence_col_ints = [
        int.from_bytes(np.packbits(incidence_arr[:, i][::-1]).tobytes()[::-1], "little")
        for i in range(n_V)
    ]
    boundary_int = 0
    subset_mask = 0
    half = n_V // 2
    best_h = float("inf")
    best_mask = 0
    total = 1 << n_V

    for k in range(1, total):
        bit = (k & -k).bit_length() - 1
        subset_mask ^= 1 << bit
        boundary_int ^= incidence_col_ints[bit]
        size = subset_mask.bit_count()
        if 1 <= size <= half:
            cut = boundary_int.bit_count()
            if cut < best_h * size:
                best_h = cut / size
                best_mask = subset_mask

    v_star = np.zeros(n_V, dtype=np.int8)
    for i in range(n_V):
        if best_mask & (1 << i):
            v_star[i] = 1
    return best_h, v_star


def cheeger_constant(g: GadgetLayout) -> float:
    """Exact boundary Cheeger constant h(F) of a gadget's F matrix.

    gadget notation: F → incidence; V_0 → support (Webster–Smith–Cohen
    arXiv:2511.15989 §II.1 Def 1 / Cross et al. arXiv:2407.18393 Def 3).

    Computed exactly by Gray-code subset enumeration, which is tractable only for ``|V_0|`` ≤ 26.

    Cross et al. arXiv:2407.18393 §3.3 Thm 6 concludes d_merged ≥ d_data when an L-layer ancilla
    system satisfies ceil(L/2) ≥ 1/h, which at the L=1 gadgets built here reduces to h ≥ 1. Below 1
    the gadget falls outside the theorem, where distance may degrade; consider
    ``boost_gadget(g, method="combinatorial", target=1.0)``.

    Reaching h ≥ 1 is a screen, not a distance guarantee: the theorem bounds the merged code
    distance rather than the circuit fault distance, it assumes the measured logical is irreducible
    — no other logical of the same type has support inside its support — which ``build_gadget``
    does not check, and nothing here verifies its conclusion. Nor does h track circuit fault
    distance, so a higher h is not a reason to expect a better circuit.

    Raises:
        ValueError: if |V_0| > 26, beyond which the exact enumeration is infeasible.
    """
    incidence = galois.GF2(np.asarray(g.incidence).astype(int))
    if incidence.shape[1] > 26:
        raise ValueError(
            f"cheeger_constant requires |V_0| ≤ 26 for an exact value; got "
            f"|V_0|={incidence.shape[1]}. Enumeration cost quadruples per additional 2 columns, and "
            f"no bound on h(F) is computed in its place, so no value can be returned; compute the "
            f"exact code distance instead."
        )
    h, _ = _exact_boundary_cheeger(incidence)
    return h


def _augment_incidence_with_random_edges(
    incidence_base: np.ndarray,
    n_new_edges: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Add n_new_edges random degree-2 rows to F.

    Each new row connects two distinct columns not already directly connected via another existing
    row. Returns None if a collision-free sample could not be drawn within the attempt budget.
    """
    incidence = incidence_base.copy()
    n_X = incidence.shape[1]
    if n_X < 2:
        return None

    def _existing_pairs(arr: np.ndarray) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for row in arr:
            ones = np.flatnonzero(row)
            for i in range(len(ones)):
                for j in range(i + 1, len(ones)):
                    pairs.add((int(ones[i]), int(ones[j])))
        return pairs

    pairs = _existing_pairs(incidence)
    new_rows: list[np.ndarray] = []
    for _ in range(n_new_edges):
        candidate = None
        for _attempt in range(n_X * 4):
            i, j = sorted(int(x) for x in rng.choice(n_X, 2, replace=False))
            if (i, j) not in pairs:
                candidate = (i, j)
                break
        if candidate is None:
            return None
        pairs.add(candidate)
        row = np.zeros(n_X, dtype=np.int_)
        row[candidate[0]] = 1
        row[candidate[1]] = 1
        new_rows.append(row)
    if not new_rows:
        return incidence
    return np.vstack([incidence, np.stack(new_rows)])


def _boost_gadget_cheeger_combinatorial(
    g: GadgetLayout,
    *,
    target_h: float = 1.0,
    max_extra_qubits: int = 50,
    seed: int | None = None,
) -> GadgetLayout:
    """Greedy combinatorial Cheeger boost toward a target h(F).

    gadget notation: F → incidence; V_0 → support; κ → ancilla.

    Computes the exact boundary Cheeger constant h(F) via subset enumeration (Webster Def 1 / Cross
    Def 3). When h < target_h, identifies the worst cut v* and adds a κ qubit (degree-2 row of F)
    with one endpoint in v* and one outside, which monotonically increases |∂v*| by 1 without
    decreasing any other |∂v|.

    Reaching target_h = 1.0 meets the h >= 1 condition of Cross et al. arXiv:2407.18393 §3.3 Thm 6
    at L=1; see ``cheeger_constant`` for what that does and does not establish. Unlike
    ``cheeger_constant``, this buffers every enumerated subset, so memory grows with |V_0| as well
    as time.

    Args:
        g: input gadget produced by build_gadget.
        target_h: Cheeger target. Default 1.0 (Cross Thm 6 threshold).
        max_extra_qubits: cap on additions. Default 50.
        seed: RNG seed for tie-breaking in edge selection.

    Returns:
        A new GadgetLayout with F augmented, rebuilt via _build_gadget_augmented (basis=X/Z handled
        symmetrically). Its h(F) >= target_h when ``g`` came from build_gadget.

    Raises:
        ValueError: |V_0| > 26 (enumeration infeasible) or target_h <= 0.
        RuntimeError: target_h could not be reached, either within max_extra_qubits or at all.
    """
    from .gadget import _build_gadget_augmented

    if target_h <= 0:
        raise ValueError(f"target_h must be positive, got {target_h}.")
    if max_extra_qubits < 0:
        raise ValueError(f"max_extra_qubits must be >= 0, got {max_extra_qubits}.")

    rng = np.random.default_rng(seed)
    incidence = np.asarray(g.incidence).astype(np.int_).copy()
    n_orig_rows = incidence.shape[0]
    n_V = incidence.shape[1]
    if n_V > 26:
        raise ValueError(
            f"|V_0| = {n_V} > 26; exact Cheeger enumeration infeasible. "
            f"Use boost_gadget(method='distance') (BP+OSD) instead."
        )
    if n_V < 2:
        # F has at most one column, so there is no cut to improve: rebuild the gadget unchanged.
        # Reached whenever the measured support has weight ≤ 1.
        return _build_gadget_augmented(
            g.code,
            g.x,
            np.zeros((0, n_V), dtype=np.uint8),
            basis=g.basis,
        )

    half = n_V // 2
    incidence_col_ints = [
        int.from_bytes(np.packbits(incidence[:, i][::-1]).tobytes()[::-1], "little")
        for i in range(n_V)
    ]
    total = 1 << n_V
    masks_buf: list[int] = []
    sizes_buf: list[int] = []
    cuts_buf: list[int] = []
    boundary_int = 0
    subset_mask = 0
    for k in range(1, total):
        bit = (k & -k).bit_length() - 1
        subset_mask ^= 1 << bit
        boundary_int ^= incidence_col_ints[bit]
        size = subset_mask.bit_count()
        if 1 <= size <= half:
            masks_buf.append(subset_mask)
            sizes_buf.append(size)
            cuts_buf.append(boundary_int.bit_count())

    masks = np.array(masks_buf, dtype=np.uint64)
    sizes = np.array(sizes_buf, dtype=np.int32)
    cuts = np.array(cuts_buf, dtype=np.int32)

    def _existing_pairs(arr: np.ndarray) -> set[tuple[int, int]]:
        pairs: set[tuple[int, int]] = set()
        for row in arr:
            ones = np.flatnonzero(row)
            for a in range(len(ones)):
                for b in range(a + 1, len(ones)):
                    pairs.add((int(ones[a]), int(ones[b])))
        return pairs

    extra = 0
    exhausted_pairs = False
    while True:
        h_num = cuts.astype(np.int64)
        h_den = sizes.astype(np.int64)
        idx = int(np.argmin(h_num / h_den))
        h = float(h_num[idx] / h_den[idx])
        worst_mask = int(masks[idx])

        if h >= target_h:
            break
        if extra >= max_extra_qubits:
            break

        # Both sides are non-empty: worst_mask comes from the enumeration above, which only kept
        # subsets with 1 <= |v| <= n_V // 2, and n_V >= 2 here.
        v_star_arr = np.array([(worst_mask >> i) & 1 for i in range(n_V)], dtype=np.int8)
        inside = np.flatnonzero(v_star_arr).tolist()
        outside = np.flatnonzero(1 - v_star_arr).tolist()

        rng.shuffle(inside)
        rng.shuffle(outside)
        pairs = _existing_pairs(incidence)
        chosen = None
        for i in inside:
            for j in outside:
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in pairs:
                    chosen = (a, b)
                    break
            if chosen is not None:
                break
        if chosen is None:
            # Every pair spanning the cut already shares a row of F -- one weight-4 row blocks all
            # six of its pairs.
            exhausted_pairs = True
            break

        new_row = np.zeros(n_V, dtype=np.int_)
        new_row[chosen[0]] = 1
        new_row[chosen[1]] = 1
        incidence = np.vstack([incidence, new_row])
        extra += 1

        bit_i = ((masks >> chosen[0]) & np.uint64(1)).astype(np.int32)
        bit_j = ((masks >> chosen[1]) & np.uint64(1)).astype(np.int32)
        cuts += bit_i ^ bit_j

    if h < target_h:
        if exhausted_pairs:
            reason = (
                "every pair spanning the worst cut already shares a row of F, and the search adds "
                "no second row on a pair it already spans, so a larger budget would not help. "
                "Lower the target"
            )
        else:
            reason = (
                f"the budget of max_extra_qubits={max_extra_qubits} is spent. Increase it or lower "
                f"the target"
            )
        raise RuntimeError(
            f"combinatorial boost could not reach target_h={target_h} (reached h={h}): {reason}."
        )
    incidence_extra = incidence[n_orig_rows:].astype(np.uint8)
    return _build_gadget_augmented(g.code, g.x, incidence_extra, basis=g.basis)


def _boost_gadget_distance(
    g: GadgetLayout,
    *,
    target_distance: int,
    max_extra_qubits: int = 30,
    num_trials_per_step: int = 20,
    decoder_trials: int = 10,
    seed: int | None = None,
) -> GadgetLayout:
    """Distance-verifying gadget boost (Williamson & Yoder arXiv:2410.02213).

    gadget notation: F → incidence; κ' → new ancilla qubits.

    Iteratively add small random batches of degree-2 edges to F, using a BP+OSD upper bound on
    merged code distance to fast-reject any augmentation that falls below target. Starts from
    n_extra = 0, so a bare gadget already meeting the target comes back unaugmented.

    Args:
        g: input gadget produced by build_gadget.
        target_distance: minimum X- and Z-distance required for acceptance (usually d_data, the data
            code's distance).
        max_extra_qubits: cap on number of new κ' qubits to consider.
        num_trials_per_step: random augmentations per n_extra value.
        decoder_trials: trials for each get_distance_bound_with_decoder call.
        seed: RNG seed for the edge sampling only. The BP+OSD screen is unseeded, so identical calls
            with the same seed can differ in outcome.

    Returns:
        A new GadgetLayout whose merged code passes the BP+OSD screen at target_distance.

    Raises:
        ValueError: target_distance <= 0 or max_extra_qubits < 0.
        RuntimeError: neither the bare gadget nor any augmentation within max_extra_qubits passed
            the screen. Retry, raise decoder_trials or num_trials_per_step, raise max_extra_qubits,
            or lower the target.
    """
    from qldpc.objects import Pauli as _Pauli

    from .gadget import _build_gadget_augmented

    if target_distance <= 0:
        raise ValueError(f"target_distance must be positive, got {target_distance}.")
    if max_extra_qubits < 0:
        raise ValueError(f"max_extra_qubits must be >= 0, got {max_extra_qubits}.")

    rng = np.random.default_rng(seed)
    incidence_base = np.asarray(g.incidence).astype(np.int_)
    n_V = incidence_base.shape[1]

    def _passes_decoder(layout: GadgetLayout) -> bool:
        # Reconstruct the merged CSSCode from layout.HX_merged / HZ_merged
        # to feed the existing decoder.
        merged = CSSCode(
            galois.GF2(np.asarray(layout.HX_merged).astype(np.int_).tolist()),
            galois.GF2(np.asarray(layout.HZ_merged).astype(np.int_).tolist()),
            is_subsystem_code=False,
        )
        bx = merged.get_distance_bound_with_decoder(_Pauli.X, num_trials=decoder_trials)
        if bx < target_distance:
            return False
        bz = merged.get_distance_bound_with_decoder(_Pauli.Z, num_trials=decoder_trials)
        return bz >= target_distance

    # n_extra = 0: bare gadget first.
    bare = _build_gadget_augmented(g.code, g.x, np.zeros((0, n_V), dtype=np.uint8), basis=g.basis)
    if _passes_decoder(bare):
        return bare

    # Augmentation loop: only runs when the bare gadget fails the BP+OSD screen.
    for n_extra in range(1, max_extra_qubits + 1):
        for _trial in range(num_trials_per_step):
            incidence_extra = _augment_incidence_with_random_edges(incidence_base, n_extra, rng)
            if incidence_extra is None:
                continue
            # _augment_incidence_with_random_edges returns F_aug = incidence_base + extra rows;
            # extract just the new rows for _build_gadget_augmented.
            incidence_extra_rows = np.asarray(incidence_extra[incidence_base.shape[0] :]).astype(
                np.uint8
            )
            # Best-effort heuristic search: skip augmentations that fail row-weight/shape
            # validation (the only failure _build_gadget_augmented raises); let anything
            # unexpected propagate rather than silently swallowing it.
            try:
                candidate = _build_gadget_augmented(
                    g.code,
                    g.x,
                    incidence_extra_rows,
                    basis=g.basis,
                )
            except ValueError:
                continue
            if _passes_decoder(candidate):
                return candidate

    raise RuntimeError(
        f"distance boost could not reach target_distance={target_distance} within "
        f"max_extra_qubits={max_extra_qubits}: neither the bare gadget nor any sampled "
        f"augmentation passed the BP+OSD screen. Increase max_extra_qubits or "
        f"num_trials_per_step, or lower the target."
    )


def boost_gadget(
    gadget: GadgetLayout,
    *,
    method: str,
    target: float,
    seed: int | None = None,
    **kwargs: Any,
) -> GadgetLayout:
    """Single entry point for Cheeger / distance boost.

    Args:
        gadget: a GadgetLayout from build_gadget.
        method: 'combinatorial' | 'distance'.
        target: target Cheeger constant (for combinatorial) or
            target distance (for distance method; cast via int(target)).
        seed: RNG seed.
        **kwargs: forwarded to the underlying boost function.

    Returns:
        A NEW GadgetLayout with boosted incidence, gauge, HX_merged, HZ_merged. method='distance'
        screens with a BP+OSD upper bound, so acceptance is a heuristic rather than a proof;
        confirm it by computing the exact distance of the code HX_merged / HZ_merged define.

    Raises:
        ValueError: method is neither 'combinatorial' nor 'distance', target is not positive, or
            the combinatorial method is used with |V_0| > 26.
        RuntimeError: the chosen method could not reach ``target``. Lower it; raising
            max_extra_qubits helps only when the budget is what ran out, which the message says.
            For method='distance', retrying can also succeed, since its screen is not seeded.
    """
    if method == "combinatorial":
        return _boost_gadget_cheeger_combinatorial(
            gadget,
            target_h=target,
            seed=seed,
            **kwargs,
        )
    if method == "distance":
        # Validate before the cast, so the rejection quotes the target the caller wrote rather than
        # the 0 that anything in (0, 1) casts to.
        if int(target) <= 0:
            raise ValueError(
                f"target must be at least 1 for method='distance', which casts it with int(), "
                f"got {target}."
            )
        return _boost_gadget_distance(
            gadget,
            target_distance=int(target),
            seed=seed,
            **kwargs,
        )
    raise ValueError(f"unknown method: {method!r}. Allowed: 'combinatorial', 'distance'.")
