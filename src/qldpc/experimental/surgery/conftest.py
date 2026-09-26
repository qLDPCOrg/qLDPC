"""Shared reference data and helpers for the surgery tests.

The seed sets are from Webster, Smith, and Cohen, arXiv:2511.15989 Appendix A.
These helpers are private to the test suite; the example notebook carries its own inputs.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from qldpc.codes.common import CSSCode

_WEBSTER_APP_A: dict[str, Any] = {
    "source": "Webster, Smith, Cohen, arXiv:2511.15989 Appendix A",
    "codes": [
        {
            "name": "62_10_6",
            "l": 31,
            "A": [0, 6, 15],
            "B": [0, 5, 7],
            "n_data_qubits": 62,
            "k_logical": 10,
            "distance": 6,
            "expected_bare_gadget_qubits_per_seed": 19,
            "expected_bridge_qubits_per_pair": 11,
            "expected_cheeger_boost_qubits": 0,
            "seeds": [
                {
                    "name": "X_bar_1",
                    "pauli_type": "X",
                    "L_support": [1, 6, 8, 10],
                    "R_support": [11, 26],
                },
                {
                    "name": "Z_bar_1",
                    "pauli_type": "Z",
                    "L_support": [3, 12, 18, 19],
                    "R_support": [11, 18],
                },
                {
                    "name": "X_bar_k2p1",
                    "pauli_type": "X",
                    "L_support": [16, 23],
                    "R_support": [0, 15, 16, 22],
                },
                {
                    "name": "Z_bar_k2p1",
                    "pauli_type": "Z",
                    "L_support": [0, 16],
                    "R_support": [1, 3, 5, 10],
                },
            ],
        },
        {
            "name": "126_12_10",
            "l": 63,
            "A": [0, 4, 37],
            "B": [0, 29, 49],
            "n_data_qubits": 126,
            "k_logical": 12,
            "distance": 10,
            "expected_bare_gadget_qubits_per_seed": 31,
            "expected_bridge_qubits_per_pair": 19,
            "expected_cheeger_boost_qubits": 0,
            "seeds": [
                {
                    "name": "X_bar_1",
                    "pauli_type": "X",
                    "L_support": [7, 12, 36, 41, 56],
                    "R_support": [1, 27, 31, 38, 61],
                },
                {
                    "name": "Z_bar_1",
                    "pauli_type": "Z",
                    "L_support": [5, 15, 28, 35, 45, 61],
                    "R_support": [1, 11, 54, 57],
                },
                {
                    "name": "X_bar_k2p1",
                    "pauli_type": "X",
                    "L_support": [9, 19, 26, 29],
                    "R_support": [5, 15, 22, 38, 48, 55],
                },
                {
                    "name": "Z_bar_k2p1",
                    "pauli_type": "Z",
                    "L_support": [2, 25, 32, 36, 62],
                    "R_support": [7, 22, 27, 51, 56],
                },
            ],
        },
        {
            "name": "254_14_16",
            "l": 127,
            "A": [0, 32, 100],
            "B": [0, 28, 49],
            "n_data_qubits": 254,
            "k_logical": 14,
            "distance": 16,
            "expected_bare_gadget_qubits_per_seed": 49,
            "expected_bridge_qubits_per_pair": 31,
            "expected_cheeger_boost_qubits": 8,
            "seeds": [
                {
                    "name": "X_bar_1",
                    "pauli_type": "X",
                    "L_support": [28, 47, 55, 75, 103, 114, 124],
                    "R_support": [4, 14, 15, 23, 50, 77, 83, 109, 123],
                },
                {
                    "name": "Z_bar_1",
                    "pauli_type": "Z",
                    "L_support": [1, 24, 33, 51, 60, 65, 107, 119, 124],
                    "R_support": [7, 8, 36, 85, 106, 114, 124],
                },
                {
                    "name": "X_bar_k2p1",
                    "pauli_type": "X",
                    "L_support": [3, 31, 32, 42, 52, 60, 81],
                    "R_support": [6, 15, 38, 42, 47, 59, 101, 106, 115],
                },
                {
                    "name": "Z_bar_k2p1",
                    "pauli_type": "Z",
                    "L_support": [0, 8, 9, 19, 27, 41, 67, 73, 100],
                    "R_support": [26, 36, 47, 75, 95, 103, 122],
                },
            ],
        },
        {
            "name": "510_16_24",
            "l": 255,
            "A": [0, 39, 55],
            "B": [0, 70, 127],
            "n_data_qubits": 510,
            "k_logical": 16,
            "distance": 24,
            "expected_bare_gadget_qubits_per_seed": 79,
            "expected_bridge_qubits_per_pair": 51,
            "expected_cheeger_boost_qubits": 20,
            "seeds": [
                {
                    "name": "X_bar_1",
                    "pauli_type": "X",
                    "L_support": [18, 31, 35, 36, 91, 126, 146, 163, 164, 180, 196, 216, 233, 253],
                    "R_support": [48, 52, 87, 101, 103, 106, 107, 125, 140, 156, 179, 211],
                },
                {
                    "name": "Z_bar_1",
                    "pauli_type": "Z",
                    "L_support": [38, 54, 57, 93, 112, 148, 164, 185, 197, 203, 213, 238, 240, 252],
                    "R_support": [18, 55, 59, 73, 129, 130, 142, 182, 187, 199, 244, 252],
                },
                {
                    "name": "X_bar_k2p1",
                    "pauli_type": "X",
                    "L_support": [6, 27, 35, 80, 92, 97, 137, 149, 150, 206, 220, 224],
                    "R_support": [27, 39, 41, 66, 76, 82, 94, 115, 131, 167, 186, 222, 225, 241],
                },
                {
                    "name": "Z_bar_k2p1",
                    "pauli_type": "Z",
                    "L_support": [10, 11, 14, 16, 30, 65, 69, 161, 193, 216, 232, 247],
                    "R_support": [26, 81, 82, 86, 99, 119, 139, 156, 176, 192, 208, 209, 226, 246],
                },
            ],
        },
    ],
}


def load_webster_seed_set(code_index: int) -> dict[str, Any]:
    """Return a fresh copy of Webster Appendix A data for code index 0..3.

    Raises:
        IndexError: if code_index is not in 0..3.
    """
    if not 0 <= code_index < len(_WEBSTER_APP_A["codes"]):
        raise IndexError(f"code_index must be in 0..3, got {code_index}")
    return copy.deepcopy(_WEBSTER_APP_A["codes"][code_index])


def build_generalised_bicycle_code(ell: int, A_set: list[int], B_set: list[int]) -> CSSCode:
    """Build a generalised bicycle code from cyclic exponent sets A, B.

    Per Kovalev-Pryadko (arXiv:1212.6703) and Swaroop's reference implementation
    (https://github.com/eswaroop/adapters-LDPC-surgery, ext/bivariate_bicyclic.py): given subsets
    A, B of Z_ell, let A(x) = sum(x^a for a in A_set) and B(x) = sum(x^b for b in B_set) as cyclic
    matrices in F_2[Z_ell]. Then H_X = [A | B] and H_Z = [B^T | A^T] define the bicycle code on
    2*ell data qubits.

    Args:
        ell: cyclic group order.
        A_set: subset of {0, 1, ..., ell-1} defining A(x) = sum(x^a for a in A_set).
        B_set: subset of {0, 1, ..., ell-1} defining B(x) = sum(x^b for b in B_set).

    Returns:
        CSSCode on 2*ell data qubits with check matrices [A | B] and [B^T | A^T] over GF(2).
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

    L_support and R_support are sparse index lists (positions within each ell-half that are set to
    1). Returns a dense binary vector of length 2*ell.

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
