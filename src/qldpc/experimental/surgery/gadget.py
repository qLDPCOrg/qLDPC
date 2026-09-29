"""L=1 gadget construction (Webster, Smith, Cohen arXiv:2511.15989 §II A).

The construction restricts complementary checks to the logical support, computes a gauge basis for
that incidence matrix, and assembles the resulting merged CSS checks. The paper writes the number
of gauge-fixing checks as ``|S_L| - wt(L) + 1``; the general form is ``|S_L| - rank(F)``, and the
two agree whenever dim ker(F) = 1, as they do for the paper's four codes.

Notation (used throughout the surgery package; the symbols follow Webster/Cohen/Cross, not Cain).
For a logical measured on support V_0::

    V_0  — logical support (measured qubits)                 → ``support``
    C_0  — data-code checks touching V_0                     → rows of ``incidence``
    F    — restriction (incidence) matrix on (C_0, V_0)      → ``incidence``
    κ    — gadget ancilla qubits (one per row of F)
    χ    — added meas-basis checks (one per support qubit)
    G    — a basis of ker(F^T); the gadget gauge checks      → ``gauge``

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
from typing import cast

import galois
import numpy as np

from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ, PauliXZLike

from .construction import _CSSConeMaps, _CSSConeResult


@dataclasses.dataclass(frozen=True, eq=False)
class GadgetLayout:
    """An L=1 surgery gadget for measuring one logical operator of a CSS code.

    Built by ``build_gadget``. The field names map onto Webster, Smith, Cohen arXiv:2511.15989 §II A
    as V_0 → support, F → incidence, G → gauge.

    ``incidence`` carries one row per κ ancilla qubit, and the κ qubits occupy the merged-code qubit
    indices from ``code.num_qudits`` onward, so ``incidence.shape[0]`` is the κ count. A boosted
    gadget appends one row per added κ qubit; those rows belong to no check of the data code, so
    ``incidence`` is then not a plain restriction of the complementary check matrix.
    """

    code: CSSCode  # the data code being operated on
    x: np.ndarray  # measured logical operator, a length-code.num_qudits binary support vector
    support: tuple[int, ...]  # V_0: data-qubit indices where x is 1
    incidence: np.ndarray  # F: complementary check matrix restricted to (checks on V_0, support)
    gauge: np.ndarray  # G: rows spanning ker(F^T) over GF(2), the gauge-fixing checks
    HX_merged: np.ndarray  # X checks of the merged code, over data qubits then ancilla qubits
    HZ_merged: np.ndarray  # Z checks of the merged code, same qubit ordering
    basis: PauliXZ  # Pauli.X to measure a logical X, Pauli.Z for a logical Z

    def _get_cone_result(self) -> _CSSConeResult:
        """Derive cone provenance while retaining the stored merged matrices as authoritative."""
        _, data_checks, base_incidence = _restrict_checks_to_support(
            self.code, self.x, basis=self.basis
        )
        num_extra = self.incidence.shape[0] - base_incidence.shape[0]
        if num_extra < 0:
            raise ValueError("layout incidence has fewer rows than the data-code restriction")
        data_checks_aug = tuple(data_checks) + (None,) * num_extra
        maps = _build_cone_maps(
            self.code,
            self.support,
            data_checks_aug,
            self.incidence,
            self.gauge,
            basis=self.basis,
        )
        derived = maps.build(self.code)
        stored_code = CSSCode(
            np.asarray(self.HX_merged, dtype=np.int_),
            np.asarray(self.HZ_merged, dtype=np.int_),
            is_subsystem_code=False,
        )
        return dataclasses.replace(derived, code=stored_code)

    def _base_incidence(self) -> np.ndarray:
        """Incidence matrix before any boost or bridge ancillas were added."""
        _, _, incidence = _restrict_checks_to_support(self.code, self.x, basis=self.basis)
        return incidence

    @property
    def is_augmented(self) -> bool:
        """Whether this layout contains ancillas beyond the data-code restriction."""
        return not np.array_equal(self.incidence, self._base_incidence())

    @property
    def added_ancilla_incidence(self) -> np.ndarray:
        """Rows added after the data-code restriction, in their current order."""
        base_rows = self._base_incidence().shape[0]
        return self.incidence[base_rows:].copy()

    def with_added_ancillas(self, incidence_rows: np.ndarray) -> GadgetLayout:
        """Return a layout with additional weight-2 ancilla rows, preserving existing additions."""
        added = np.asarray(incidence_rows).astype(np.uint8)
        if added.ndim != 2 or added.shape[1] != len(self.support):
            width = added.shape[1] if added.ndim == 2 else None
            raise ValueError(
                f"incidence_rows has {width} columns; expected {len(self.support)} (= |support|)"
            )
        combined = np.vstack([self.added_ancilla_incidence, added])
        return _rebuild_with_added_ancillas(self.code, self.x, combined, basis=self.basis)


def _restrict_checks_to_support(
    code: CSSCode,
    x: np.ndarray,
    *,
    basis: PauliXZ = Pauli.X,
) -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    """Webster §II A steps 1-2 — V_0 = supp(x); C_0 = checks on V_0; F = H_complement[C_0, V_0].

    For basis=Pauli.X: incidence = H_Z[data_checks, support] (the complementary basis to the
    measured logical). For basis=Pauli.Z: incidence = H_X[data_checks, support].
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


def _compute_gauge_basis(incidence: np.ndarray) -> np.ndarray:
    """Webster §II A step 3 — G whose rows form a canonical basis of ker(F.T) over GF(2).

    Uses galois ``left_null_space`` (row-reduced) so the basis is deterministic.
    """
    if incidence.size == 0:
        return np.zeros((0, incidence.shape[0]), dtype=np.uint8)
    gauge = galois.GF2(incidence.astype(np.int_).tolist()).left_null_space()
    return np.asarray(gauge).astype(np.uint8)


def _build_cone_maps(
    code: CSSCode,
    support: tuple[int, ...],
    data_checks: tuple[int | None, ...],
    incidence: np.ndarray,
    gauge: np.ndarray,
    *,
    basis: PauliXZ,
) -> _CSSConeMaps:
    """Translate the Webster pieces into the internal four-map cone contract."""
    num_ancillas, num_measurement_checks = incidence.shape
    measurement_to_data = np.zeros((num_measurement_checks, code.num_qudits), dtype=np.uint8)
    measurement_to_data[np.arange(num_measurement_checks), np.asarray(support)] = 1
    num_complement_checks = code.matrix_z.shape[0] if basis is Pauli.X else code.matrix_x.shape[0]
    complement_from_data = np.zeros((num_complement_checks, num_ancillas), dtype=np.uint8)
    for column, check in enumerate(data_checks):
        if check is not None:
            complement_from_data[check, column] = 1
    return _CSSConeMaps(
        basis=basis,
        measurement_to_data=measurement_to_data,
        measurement_boundary=np.asarray(incidence.T, dtype=np.uint8),
        complement_from_data=complement_from_data,
        complement_boundary=np.asarray(gauge, dtype=np.uint8),
        measurement_groups=np.ones((1, num_measurement_checks), dtype=np.uint8),
    )


def _assemble_merged_checks(
    code: CSSCode,
    support: tuple[int, ...],
    data_checks: tuple[int | None, ...],
    incidence: np.ndarray,
    gauge: np.ndarray,
    *,
    basis: PauliXZ = Pauli.X,
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble HX_merged and HZ_merged from the Webster §II A pieces.

    basis=X (default): χ rows added to HX_merged, G to HZ_merged.
    basis=Z: χ rows added to HZ_merged, G to HX_merged (basis-symmetric dual).
    """
    result = _build_cone_maps(
        code,
        support,
        data_checks,
        incidence,
        gauge,
        basis=basis,
    ).build(code)
    return (
        np.asarray(result.code.matrix_x, dtype=np.uint8),
        np.asarray(result.code.matrix_z, dtype=np.uint8),
    )


def build_gadget(
    code: CSSCode,
    x: np.ndarray,
    *,
    basis: PauliXZLike,
) -> GadgetLayout:
    """Webster §II A L=1 gadget: restriction, gauge fix, assembly. Deterministic in its arguments.

    gadget notation: κ qubits → rows of incidence; G → gauge.

    basis=Pauli.X: measures a logical X (PPM of X̄). Validates H_Z @ x == 0.
    basis=Pauli.Z: measures a logical Z (PPM of Z̄). Validates H_X @ x == 0.

    Raises:
        ValueError: code is a subsystem code or is not over GF(2); x has an entry outside {0, 1};
            basis is neither Pauli.X nor Pauli.Z; x fails the complementary check equation
            (H_Z @ x == 0 for basis=X, H_X @ x == 0 for basis=Z); x is the zero vector; or x lies
            in the row space of the measured basis's check matrix, making it a stabilizer rather
            than a logical operator.
    """
    if code.is_subsystem_code:
        raise ValueError(
            "build_gadget currently supports only stabilizer (non-subsystem) CSS codes."
        )
    if code.field.order != 2:
        raise ValueError(
            f"build_gadget requires a qubit code, got one over GF({code.field.order}). The gauge "
            f"fix, the Cheeger boost and the merged-code assembly are all mod 2."
        )
    if isinstance(basis, str):
        try:
            basis = cast(PauliXZ, Pauli.from_string(basis))
        except ValueError:
            pass  # fall through to the basis-validation error below
    x = np.asarray(x)
    # Check before the cast to uint8, which wraps 256 to 0 and 257 to 1 rather than complaining.
    if ((x != 0) & (x != 1)).any():
        raise ValueError(f"x must be a binary support vector, got entries outside {{0, 1}}: {x}.")
    x = x.astype(np.uint8)
    if basis is Pauli.X:
        H_check = np.asarray(code.matrix_z).astype(np.uint8)
        H_same = np.asarray(code.matrix_x).astype(np.uint8)
        if ((H_check @ x) % 2).any():
            raise ValueError("x is not a logical-X support (H_Z @ x != 0).")
    elif basis is Pauli.Z:
        H_check = np.asarray(code.matrix_x).astype(np.uint8)
        H_same = np.asarray(code.matrix_z).astype(np.uint8)
        if ((H_check @ x) % 2).any():
            raise ValueError("x is not a logical-Z support (H_X @ x != 0).")
    else:
        raise ValueError(f"basis must be Pauli.X or Pauli.Z, got {basis!r}")

    # The zero vector satisfies H @ x == 0 but measures nothing: it yields an empty support and a
    # 0x0 incidence, for which cheeger_constant reports inf.
    if not x.any():
        raise ValueError("x is the zero vector, which measures no logical operator.")

    # The check equation above admits the whole normalizer, so every stabilizer of the measured
    # basis passes it too. Such an x measures the identity, so reject it: x must not lie in the row
    # space of the measured basis's check matrix.
    H_same_gf2 = galois.GF2(H_same.astype(np.int_))
    x_gf2 = galois.GF2(x.astype(np.int_))
    if np.linalg.matrix_rank(np.vstack([H_same_gf2, x_gf2])) == np.linalg.matrix_rank(H_same_gf2):
        H_name = "H_X" if basis is Pauli.X else "H_Z"
        raise ValueError(
            f"x lies in the row space of {H_name}, so it is a stabilizer rather than a logical "
            f"operator, and the gadget would measure the identity."
        )

    support, data_checks, incidence = _restrict_checks_to_support(code, x, basis=basis)
    gauge = _compute_gauge_basis(incidence)
    maps = _build_cone_maps(code, support, data_checks, incidence, gauge, basis=basis)
    assert maps.measures_exact_span(code, x.reshape(1, -1)), (
        "internal cone maps do not measure the requested logical operator"
    )
    result = maps.build(code)
    return GadgetLayout(
        code=code,
        x=x,
        support=support,
        incidence=incidence,
        gauge=gauge,
        HX_merged=np.asarray(result.code.matrix_x, dtype=np.uint8),
        HZ_merged=np.asarray(result.code.matrix_z, dtype=np.uint8),
        basis=basis,
    )


def _rebuild_with_added_ancillas(
    code: CSSCode,
    x: np.ndarray,
    incidence_extra: np.ndarray,
    *,
    basis: PauliXZ,
) -> GadgetLayout:
    """Rebuild a GadgetLayout with incidence augmented by extra weight-2 rows.

    Each row of ``incidence_extra`` has weight 2 and corresponds to a new κ qubit not backed by any
    original Z-check (basis=X) or X-check (basis=Z). The function:

    1. Stacks incidence_aug = [incidence; incidence_extra].
    2. Recomputes G_aug = ker(incidence_aug^T).
    3. Assembles merged checks with the original V_0 / C_0 plus the new κ rows. The extra columns of
       tilde_F are all zero, since no original check sits on the new κ qubits.

    The returned ``incidence`` covers the new κ qubits, whose merged-code qubit indices come after
    the original ones.
    """
    x = np.asarray(x).astype(np.uint8)
    support, data_checks, incidence = _restrict_checks_to_support(code, x, basis=basis)
    incidence_extra = np.asarray(incidence_extra).astype(np.uint8)
    if incidence_extra.size and not np.all(incidence_extra.sum(axis=1) == 2):
        bad = np.flatnonzero(incidence_extra.sum(axis=1) != 2).tolist()
        raise ValueError(f"incidence_extra rows {bad} have weight != 2; required weight 2.")

    incidence_aug = np.vstack([incidence, incidence_extra]).astype(np.uint8)
    gauge_aug = _compute_gauge_basis(incidence_aug)

    # Added ancillas are not backed by data-code checks, so their tilde_F columns stay zero.
    n_extra = incidence_extra.shape[0]
    data_checks_aug = tuple(data_checks) + (None,) * n_extra
    maps = _build_cone_maps(
        code,
        support,
        data_checks_aug,
        incidence_aug,
        gauge_aug,
        basis=basis,
    )
    assert maps.measures_exact_span(code, x.reshape(1, -1)), (
        "augmented cone maps do not measure the requested logical operator"
    )
    result = maps.build(code)
    return GadgetLayout(
        code=code,
        x=x,
        support=support,
        incidence=incidence_aug,
        gauge=gauge_aug,
        HX_merged=np.asarray(result.code.matrix_x, dtype=np.uint8),
        HZ_merged=np.asarray(result.code.matrix_z, dtype=np.uint8),
        basis=basis,
    )
