# SPDX-License-Identifier: Apache-2.0

"""Module for loading error-correcting codes from the GAP computer algebra system."""

from __future__ import annotations

import ast
import json
import re
import urllib.error
import urllib.request
from typing import Any

import galois
import numpy as np
import numpy.typing as npt

import qldpc
import qldpc.cache
import qldpc.external.gap


@qldpc.cache.use_disk_cache(
    "codes",
    key_func=lambda code: "".join(code.split()),  # strip whitespace
)
def get_classical_code(code: str) -> tuple[list[list[int]], int]:
    """Retrieve a classical code from GAP.

    Requires the GAP GUAVA package (https://www.gap-system.org/Packages/guava.html).
    """
    qldpc.external.gap.require_package("GUAVA")

    # Run GAP commands.  Each check-matrix entry is reported as its discrete logarithm base the
    # field's primitive root Z(q), or -1 for a zero entry.  GAP's Int() accepts only prime-field
    # elements, so a discrete logarithm is the field-agnostic way to bring an extension-field
    # element back; the entry is rebuilt below as primitive_element**log, which reproduces galois's
    # own encoding because GAP and galois share the primitive element (both use Conway polynomials).
    commands = [
        'LoadPackage("guava", false);;',
        f"code := {code};;",
        "mat := CheckMat(code);;",
        "z := PrimitiveRoot(LeftActingDomain(code));;",
        r'Print(LeftActingDomain(code), "\n");;',
        "for vec in mat do",
        r'  Print(List(vec, function(x) if IsZero(x) then return -1; else return LogFFE(x, z); fi; end), "\n");;',
        "od;;",
    ]
    code_str = qldpc.external.gap.get_output(*commands)

    # identify the base field, then rebuild the parity checks from the reported discrete logarithms
    field: int | None = None
    logarithms = []
    for line in code_str.splitlines():
        line = line.strip()
        if not line:
            continue

        if field is None and (match := re.search(r"GF\(([0-9]+(\^[0-9]+)?)\)", line)):
            base, exponent, *_ = (match.group(1) + "^1").split("^")
            field = int(base) ** int(exponent)
        elif line.startswith("["):
            logarithms.append(ast.literal_eval(line))

    if field is None:
        raise ValueError(f"Could not determine the base field of code: {code}")
    if not logarithms:
        raise ValueError(f"Code has no parity checks: {code}")

    primitive_element = galois.GF(field).primitive_element
    checks = [
        [0 if power < 0 else int(primitive_element**power) for power in row] for row in logarithms
    ]
    return checks, field


def _get_json(url: str) -> Any:
    """Retrieve JSON data from a URL."""
    try:
        return json.loads(urllib.request.urlopen(url, timeout=10).read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exception:
        raise RuntimeError(f"Cannot access {url}") from exception


@qldpc.cache.use_disk_cache("qecdb")
def get_quantum_code(code_id: str) -> tuple[list[str], int | None, bool]:
    """Retrieve a quantum code from qecdb.org.

    This function queries QECDB's JSON API at https://qecdb.org, so it requires network access.

    Return the stabilizers of the code, its distance, and whether it's CSS.
    """
    code_data = _get_json(f"https://qecdb.org/api/codes/{code_id}")
    stabilizers = code_data.get("H", "").split()
    if not stabilizers:
        raise ValueError(f"Could not find stabilizer data for code '{code_id}'")
    distance = code_data.get("d")
    return stabilizers, None if distance is None else int(distance), code_data.get("css") is True


@qldpc.cache.use_disk_cache("qldpc-challenge")
def get_qldpc_challenge_code(code_id: str) -> tuple[npt.NDArray[np.int_], int | None, bool]:
    """Retrieve a quantum code by ID from the Unitary Foundation qLDPC Challenge.

    This function fetches the JSON artifacts that the challenge publishes at
    https://unitaryfoundation.github.io/qldpc-challenge/codes/, so it requires network access.

    Return the symplectic parity check matrix of the code, its distance, and whether it's CSS.  A
    submitted distance is only a witness-certified upper bound, so the distance is None unless the
    challenge has certified it to be exact.
    """
    url = "https://unitaryfoundation.github.io/qldpc-challenge/codes"
    code_data = _get_json(f"{url}/{code_id}.json")
    index = _get_json(f"{url}/_index.json")
    try:
        num_qubits = code_data["n"]
        checks = code_data["checks"]
        is_css = code_data["code_type"] == "CSS"
        supports = (
            [(support, []) for support in checks["X"]] + [([], support) for support in checks["Z"]]
            if is_css
            else [(generator["X"], generator["Z"]) for generator in checks["S"]]
        )
        matrix = np.zeros((len(supports), 2, num_qubits), dtype=int)
        for row, (support_x, support_z) in enumerate(supports):
            matrix[row, 0, support_x] = 1
            matrix[row, 1, support_z] = 1
        exact = any(entry["id"] == code_id and entry["tier"] == "exact" for entry in index["codes"])
        distance = int(code_data["distance"]["d"]) if exact else None
    except (IndexError, KeyError, TypeError) as exception:
        raise ValueError(
            f"Could not parse qLDPC Challenge data for code '{code_id}'"
        ) from exception
    return matrix.reshape(len(supports), 2 * num_qubits), distance, is_css


def _gap_define_sparse_matrix(
    matrix_var: str, field_order: int, matrix: npt.NDArray[np.int_]
) -> list[str]:
    _, matrix_width = matrix.shape
    # Turn matrix into sparse representation where `nonzero_entries[i][j]` is a list of integers
    # where, for all values `l` in that list, `matrix[i,l] == j+1`.
    # Example:
    #     matrix_var: NDArray[F3] = [
    #         [0, 0, 0, 1, 2, 1],
    #         [1, 0, 0, 0, 0, 0],
    #     ]
    #     nonzero_entries: list[list[np.NDArray[np.int_]]] = [
    #         [ # Sparse definition of the first matrix row, `matrix_var[0]`
    #             [3, 5],  # Columns in this row that contain 1's: `matrix_var[0, 3] == 1`
    #             [4],  # Columns in this row that contain 2's: `matrix_var[0, 4] == 2`
    #         ],
    #         [ # Sparse definition of the second matrix row, `matrix_var[1]`
    #             [0],  # Columns in this row that contain 1's
    #             [],  # Columns in this row that contain 2's
    #         ],
    #     ]
    nonzero_entries = [
        [np.nonzero(row == val)[0] for val in range(1, field_order)] for row in matrix
    ]

    def nonzero_row_str(nonzeros: list[npt.NDArray[np.int_]]) -> str:
        all_field_vals = [f"[{','.join(str(int(val)) for val in columns)}]" for columns in nonzeros]
        return f"[{','.join(all_field_vals)}]"

    nonzero_str = ",".join(nonzero_row_str(nonzeros) for nonzeros in nonzero_entries)

    # Map each stored galois integer 1..field_order-1 to the matching GAP field element.  A galois
    # integer encodes an element's coordinates in the primitive-element basis, so it equals
    # Z(field_order)^log.  GAP and galois share that primitive element (both use Conway
    # polynomials), so the powers agree.
    field = galois.GF(field_order)
    field_elements = ",".join(
        f"Z({field_order})^{int(field(value).log())}" for value in range(1, field_order)
    )
    commands = [
        f"nz:=[{nonzero_str}];;",
        f"F:=GF({field_order});;",
        f"elements:=[{field_elements}];;",
        f"{matrix_var}:=[];",
        "for r in nz do",
        f"  v:=ListWithIdenticalEntries({matrix_width},Zero(F));;",
        f"  for f in [1..{field_order - 1}] do",
        "    for i in r[f] do",
        "      v[i+1]:=elements[f];;",
        "    od;;",
        "  od;;",
        f"  Append({matrix_var},[v]);;",
        "od;;",
    ]
    commands = [cmd.strip() for cmd in commands]
    return commands


def get_distance_bound(
    code: qldpc.codes.QuditCode,
    num_trials: int = 1,
    *,
    cutoff: int | None = None,
    maxav: str = "fail",
) -> int:
    """Estimate the distance of a quantum code using GAP's QDistRnd package.

    The estimate is a randomized upper bound (QDistRnd samples random codewords).  Requires the GAP
    GUAVA and QDistRnd packages.

    If given a CSSCode, estimate the Z-distance (minimum weight of a Z-type logical operator).
    See https://qec-pages.github.io/QDistRnd/doc/chap4.html.

    Note that QDistRnd does not support subsystem codes.  In the case of a CSS code, however, we
    can still compute the Z-distance by promoting all Z-type gauge group generators to stabilizers.
    ``maxav`` is passed through to QDistRnd as a raw GAP value (default "fail"); see its docs above.
    """
    qldpc.external.gap.require_package("GUAVA")
    qldpc.external.gap.require_package("QDistRnd", "https://github.com/QEC-pages/QDistRnd")

    field = f"GF({code.field.order})"
    one = f"One({field})"
    cutoff = cutoff or 0
    kwargs = ",".join([f"field:={field}", f"maxav:={maxav}"])

    if isinstance(code, qldpc.codes.CSSCode):
        code_x = qldpc.codes.ClassicalCode(code.get_stabilizer_ops(qldpc.objects.Pauli.X))
        code_z = code.code_z
        args = ",".join([f"{one}*matrix_x", f"{one}*matrix_z", f"{num_trials}", f"{cutoff}"])
        commands = [
            'LoadPackage("QDistRnd", false);;',
            *_gap_define_sparse_matrix("matrix_x", code.field.order, code_x.matrix),
            *_gap_define_sparse_matrix("matrix_z", code.field.order, code_z.matrix),
            f"Print(DistRandCSS({args}:{kwargs}));;",
        ]

    elif code.is_subsystem_code:
        raise ValueError("QDistRnd cannot estimate the distance of non-CSS subsystem codes.")

    else:
        # "riffle" the parity check matrix to put X and Z support bits each qudit next to each other
        matrix = code.matrix.reshape(-1, 2, len(code)).transpose(0, 2, 1).reshape(code.matrix.shape)
        riffled_code = qldpc.codes.ClassicalCode(matrix)
        args = ",".join([f"{one}*matrix", f"{num_trials}", f"{cutoff}"])
        commands = [
            'LoadPackage("QDistRnd", false);',
            *_gap_define_sparse_matrix("matrix", code.field.order, riffled_code.matrix),
            f"Print(DistRandStab({args}:{kwargs}));",
        ]

    # Issue: Piped input somehow causes extra terminal output.  Fix: Ignore all but last line.
    result_lines = qldpc.external.gap.get_output(*commands, use_pipe=True).strip().splitlines()
    if not result_lines:
        raise ValueError(f"QDistRnd returned no output for code:\n{code}")
    output = result_lines[-1]

    # strip whitespace and comments, and interpret the remaining text as the bound
    lines = [line.strip() for line in output.splitlines()]
    bound = "".join([line for line in lines if not line.startswith("#")])
    if not bound:
        raise ValueError(f"Could not parse a distance bound from QDistRnd output: {output!r}")
    return int(bound)
