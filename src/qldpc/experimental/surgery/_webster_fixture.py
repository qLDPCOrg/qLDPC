"""Webster (arXiv:2511.15989) Appendix A seed-set fixture + helpers.

Private to the surgery test suite and the `examples/lattice_surgery.ipynb`
demo. Not part of the public API. Pytest's default discovery skips this
file (no `def test_*` + leading underscore).

Provides:

* `load_webster_seed_set(code_index)` — read the JSON fixture for code
  index 0..3.
* `build_generalised_bicycle_code(l, A_set, B_set)` — build a CSS code
  from cyclic exponent sets per Kovalev-Pryadko arXiv:1212.6703.
* `_webster_x_bar_operator(data, name, pauli_type)` — extract a named
  logical operator from a seed-set dict as a length-2l bit-vector.
* `_webster_z_bar_operator(data, name)` — Z-type convenience wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from qldpc.codes.common import CSSCode

_WEBSTER_APP_A_PATH = Path(__file__).resolve().parent / "_webster_app_a.json"


def load_webster_seed_set(code_index: int) -> dict[str, Any]:
    """Load Webster (arXiv:2511.15989) Appendix A data for code index 0..3.

    Returns:
        A dict matching the JSON schema.

    Raises:
        IndexError: if code_index is not in 0..3.
        FileNotFoundError: if the JSON fixture is missing.
    """
    if not 0 <= code_index <= 3:
        raise IndexError(f"code_index must be in 0..3, got {code_index}")
    with _WEBSTER_APP_A_PATH.open() as fh:
        data = json.load(fh)
    return data["codes"][code_index]


def build_generalised_bicycle_code(ell: int, A_set: list[int], B_set: list[int]) -> CSSCode:
    """Build a generalised bicycle code from cyclic exponent sets A, B.

    Per Kovalev-Pryadko (arXiv:1212.6703) and Swaroop's reference
    implementation (https://github.com/eswaroop/adapters-LDPC-surgery,
    ext/bivariate_bicyclic.py): given subsets A, B of Z_ell, let A(x) =
    sum(x^a for a in A_set) and B(x) = sum(x^b for b in B_set) as cyclic
    matrices in F_2[Z_ell]. Then H_X = [A | B] and H_Z = [B^T | A^T] define
    the bicycle code on 2*ell data qubits.

    Args:
        ell: cyclic group order.
        A_set, B_set: subsets of {0, 1, ..., ell-1}.

    Returns:
        CSSCode on 2*ell data qubits with check matrices [A | B] and
        [B^T | A^T] over GF(2).
    """
    I_ell = np.eye(ell, dtype=np.int_)
    S = np.roll(I_ell, shift=-1, axis=0)
    A = np.zeros((ell, ell), dtype=np.int_)
    for a in A_set:
        A = (A + np.linalg.matrix_power(S, a)) % 2
    B = np.zeros((ell, ell), dtype=np.int_)
    for b in B_set:
        B = (B + np.linalg.matrix_power(S, b)) % 2

    H_X = np.hstack([A, B])
    H_Z = np.hstack([B.T, A.T])

    return CSSCode(H_X, H_Z, is_subsystem_code=False)


def _webster_x_bar_operator(
    data: dict[str, Any],
    name: str = "X_bar_1",
    pauli_type: str = "X",
) -> np.ndarray:
    """Extract the named logical operator from a Webster seed_set dict.

    L_support and R_support are sparse index lists (positions within each ell-half
    that are set to 1). Returns a dense binary vector of length 2*ell.

    Args:
        data: Webster seed set dict (from load_webster_seed_set).
        name: Seed name, e.g. "X_bar_1", "Z_bar_1".
        pauli_type: "X" or "Z"; filters seeds by pauli_type field.
    """
    ell = data["l"]
    for seed in data["seeds"]:
        if seed["name"] == name and seed["pauli_type"] == pauli_type:
            v_L = np.zeros(ell, dtype=np.uint8)
            v_L[seed["L_support"]] = 1
            v_R = np.zeros(ell, dtype=np.uint8)
            v_R[seed["R_support"]] = 1
            return np.concatenate([v_L, v_R])
    raise ValueError(f"{name!r} (pauli_type={pauli_type!r}) seed not found")


def _webster_z_bar_operator(data: dict[str, Any], name: str = "Z_bar_1") -> np.ndarray:
    """Extract the named Z-type logical operator from a Webster seed_set dict.

    Convenience wrapper around _webster_x_bar_operator with pauli_type="Z".
    """
    return _webster_x_bar_operator(data, name, pauli_type="Z")
