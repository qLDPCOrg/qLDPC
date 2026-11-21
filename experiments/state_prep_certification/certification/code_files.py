# SPDX-License-Identifier: Apache-2.0

"""The code files: each code's check matrices and logical operators, in output-qubit order.

A code file ``<code>_code.json`` holds ``distance``, ``Hx``, ``Hz``, ``Lx``, and ``Lz``, with one
column per output qubit, in the order of the circuit's output qubits.  ``circuits.py --check``
checks each input's state against these matrices, and the file says which Z-type rows are checks
and which are logical operators.  The signed stabilizers of a circuit's state do not say that: the
Z-type stabilizers of |0...0>_L span both Hz and Lz, and which rows are checks and which are logical
operators comes from the code's definition.

The code files follow MQT QECC's definitions of the codes.  Those of the A-ft codes lie in
``circuit_data/codes/``, and those of the codes of each family of synthesized circuits beside its
circuits (``circuits.SYNTHESES``).
``code_file`` gives each circuit of ``circuits.VERIFIED_FAMILIES`` its file, ``circuits.py
--build`` records the file and its SHA-256 in ``circuits.json``, and ``circuits.py --check`` checks
each file against its circuit (``check_code_file``).
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np

from experiments.state_prep_certification.certification import circuits, counting
from experiments.state_prep_certification.common import store

HERE = pathlib.Path(__file__).resolve().parent
CODES = circuits.DATA / "codes"


def code_file(entry: dict[str, Any]) -> str | None:
    """The code file of a circuit of ``circuits.json``, relative to ``certification/``, if any.

    A-ft circuits use the file of their code's folder, C-goto that of the Steane code, and
    synthesized circuits that of their synthesis.  Other families have none.
    """
    if entry["family"] not in circuits.VERIFIED_FAMILIES:
        return None
    name = "steane" if entry["family"] == "C-goto" else entry["id"].split(".")[1]
    synthesis = circuits.SYNTHESES.get(entry["family"])
    folder = CODES if synthesis is None else synthesis[1]
    return str((folder / f"{name}_code.json").relative_to(HERE))


def code_file_fields(entry: dict[str, Any]) -> dict[str, Any]:
    """``code_file`` and ``code_sha256`` of a circuit, or nothing if it has no code file."""
    name = code_file(entry)
    if name is None:
        return {}
    return {"code_file": name, "code_sha256": store.sha256_file(HERE / name)}


def check_code_file(entry: dict[str, Any]) -> list[str]:
    """The problems of a circuit's code file: its SHA-256, and those of ``check_code``."""
    path = HERE / entry["code_file"]
    if not path.exists():
        return [f"{entry['id']}: missing {entry['code_file']}"]
    if store.sha256_file(path) != entry["code_sha256"]:
        return [f"{entry['id']}: {entry['code_file']} does not match code_sha256"]
    return check_code(entry, json.loads(path.read_text()))


def check_code(entry: dict[str, Any], code: dict[str, Any]) -> list[str]:
    """The problems of a code against a circuit, each naming the failed check.

    With n the number of output qubits, the four matrices must be binary with n columns.  Each row
    of Hx, Hz, and Lz must have expectation +1 in the state that ``circuits.json`` records for the
    circuit, in output-qubit order (``circuits._check_code_state``), which implies that these rows
    commute.  The number k = n - rank(Hx) - rank(Hz) of logical qubits must equal the k of the
    circuit's [[n, k, d]], and Lx and Lz must have k rows each.  Over GF(2), Hz Lx^T must vanish,
    and Lx Lz^T must be nonsingular.  Together, these checks imply that the rows of Hx, Hz, and Lz
    generate the state's stabilizer group.  The ``distance`` of the file is not checked.
    """
    name = entry["id"]
    n = len(entry["output_qubits"])
    for key in ("Hx", "Hz", "Lx", "Lz"):
        rows = code[key]
        if any(len(row) != n or not set(row) <= {0, 1} for row in rows):
            return [f"{name}: {key} is not a binary matrix with n = {n} columns"]
    hx, hz, lx, lz = (
        np.array(code[key], dtype=np.int64).reshape(-1, n) for key in ("Hx", "Hz", "Lx", "Lz")
    )
    problems = circuits._check_code_state(entry, code)
    k = n - _rank(hx) - _rank(hz)
    expected_k = entry["code"][1]
    if k != expected_k:
        problems.append(
            f"{name}: k = n - rank(Hx) - rank(Hz) is {k}, not the code's k = {expected_k}"
        )
    for key, matrix in (("Lx", lx), ("Lz", lz)):
        if len(matrix) != expected_k:
            problems.append(f"{name}: {key} has {len(matrix)} rows, not k = {expected_k}")
    if (hz @ lx.T % 2).any():
        problems.append(f"{name}: Hz Lx^T is not zero")
    pairing = lx @ lz.T % 2
    if pairing.shape != (expected_k, expected_k) or _rank(pairing) != expected_k:
        problems.append(f"{name}: Lx Lz^T is not a nonsingular {expected_k} x {expected_k} matrix")
    return problems


def _rank(matrix: Any) -> int:
    """The GF(2) rank of a binary matrix."""
    return counting.gf2_rank(int("".join(map(str, row)) or "0", 2) for row in np.asarray(matrix))
