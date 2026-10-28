# SPDX-License-Identifier: Apache-2.0

"""Loading codes from GAP, QECDB, and the qLDPC Challenge, and QDistRnd distance bounds in GAP."""

from __future__ import annotations

import ast
import json
import re
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import galois
import numpy as np
import numpy.typing as npt

import qldpc.cache
import qldpc.external.gap
from qldpc import _deprecations

####################################################################################################
# codes from databases


@qldpc.cache.use_disk_cache(
    "codes",
    key_func=lambda code: "".join(code.split()),  # strip whitespace
)
def get_guava_code(code: str) -> tuple[list[list[int]], int]:
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
        (
            "  Print(List(vec, function(x) if IsZero(x) then return -1;"
            r' else return LogFFE(x, z); fi; end), "\n");;'
        ),
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


@qldpc.cache.use_disk_cache("qecdb")
def get_qecdb_code(code_id: str) -> tuple[list[str], int | None, bool]:
    """Retrieve a quantum code from qecdb.org.

    This function queries QECDB's JSON API at https://qecdb.org, so it requires network access.

    Return the stabilizers of the code, its distance, and whether it's CSS.

    Results are cached to disk.  Use get_qecdb_code.refresh(code_id) to retrieve and cache the
    latest data from QECDB.
    """
    code_data = _get_json(f"https://qecdb.org/api/codes/{code_id}")
    stabilizers = code_data.get("H", "").split()
    if not stabilizers:
        raise ValueError(f"Could not find stabilizer data for code '{code_id}'")
    distance = code_data.get("d")
    return stabilizers, None if distance is None else int(distance), code_data.get("css") is True


@qldpc.cache.use_disk_cache("qldpc-challenge")
def get_qldpc_challenge_code(
    code_id: str,
) -> tuple[npt.NDArray[np.int_], int | None, bool, tuple[int, ...]]:
    """Retrieve a quantum code by ID from the Unitary Foundation qLDPC Challenge.

    This function fetches the JSON artifacts that the challenge publishes at
    https://unitaryfoundation.github.io/qldpc-challenge/codes/, so it requires network access.

    Return the symplectic parity check matrix of the code, its distance, whether it's CSS, and the
    witness-certified upper bounds on distance in its submission: ``(d_X, d_Z)`` for a CSS code, or
    ``(d,)`` otherwise.  A submitted distance is only an upper bound, so the distance is ``None``
    unless the challenge has certified it to be exact.

    Results are cached to disk.  Use get_qldpc_challenge_code.refresh(code_id) to retrieve and cache
    the latest data from the challenge.
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
        distance_bounds = (
            tuple(int(code_data["distance"][side]["value"]) for side in ("X", "Z"))
            if is_css
            else (int(code_data["distance"]["d"]),)
        )
    except (IndexError, KeyError, TypeError) as exception:
        raise ValueError(
            f"Could not parse qLDPC Challenge data for code '{code_id}'"
        ) from exception
    return matrix.reshape(len(supports), 2 * num_qubits), distance, is_css, distance_bounds


def _get_json(url: str) -> Any:
    """Retrieve JSON data from a URL."""
    try:
        return json.loads(urllib.request.urlopen(url, timeout=10).read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exception:
        raise RuntimeError(f"Cannot access {url}") from exception


####################################################################################################
# QDistRnd distance bounds


def get_qdistrnd_distance_bound(
    matrix: galois.FieldArray,
    num_trials: int = 1,
    *,
    matrix_z: galois.FieldArray | None = None,
    is_subsystem_code: bool = False,
    cutoff: int | None = None,
    maxav: str = "fail",
) -> int:
    """Estimate the distance of a quantum code using GAP's QDistRnd package.

    The estimate is a randomized upper bound (QDistRnd samples random codewords).  Requires the GAP
    GUAVA and QDistRnd packages.  See https://qec-pages.github.io/QDistRnd/doc/chap4.html.

    Note that QDistRnd does not support subsystem codes.  In the case of a CSS code, however, we can
    still compute the Z-distance by promoting all Z-type gauge group generators to stabilizers.
    ``maxav`` is passed through to QDistRnd as a raw GAP value (default "fail"); see the QDistRnd
    documentation linked above.

    Args:
        matrix: The symplectic parity check matrix of a stabilizer code, with rows of the form
            ``[X|Z]``, or the X-type stabilizers of a CSS code if matrix_z is given.  The field of
            this finite-field array is the field of the code.
        num_trials: The number of independent upper bounds to minimize over.
        matrix_z: The Z-type parity checks of a CSS code, which may generate a gauge group.  If
            given, estimate the Z-distance of the CSS code (the minimum weight of a Z-type logical
            operator).
        is_subsystem_code: Whether the code defined by a symplectic matrix is a subsystem code,
            which QDistRnd does not support.  Ignored if matrix_z is given.
        cutoff: Exit early once the upper bound falls to or below this cutoff.
        maxav: The maxav option of QDistRnd.

    Returns:
        An upper bound on the distance of the code.

    Raises:
        ValueError: If given a non-CSS subsystem code, or if QDistRnd output is missing or invalid.
    """
    qldpc.external.gap.require_package("GUAVA")
    qldpc.external.gap.require_package("QDistRnd", "https://github.com/QEC-pages/QDistRnd")

    field_order = type(matrix).order
    field = f"GF({field_order})"
    one = f"One({field})"
    cutoff = cutoff or 0
    kwargs = ",".join([f"field:={field}", f"maxav:={maxav}"])

    if matrix_z is not None:
        args = ",".join([f"{one}*matrix_x", f"{one}*matrix_z", f"{num_trials}", f"{cutoff}"])
        commands = [
            'LoadPackage("QDistRnd", false);;',
            *_get_gap_sparse_matrix_commands("matrix_x", field_order, matrix),
            *_get_gap_sparse_matrix_commands("matrix_z", field_order, matrix_z),
            f"Print(DistRandCSS({args}:{kwargs}));;",
        ]

    elif is_subsystem_code:
        raise ValueError("QDistRnd cannot estimate the distance of non-CSS subsystem codes.")

    else:
        # "riffle" the parity check matrix to put X and Z support bits each qudit next to each other
        num_qudits = matrix.shape[1] // 2
        riffled_matrix = matrix.reshape(-1, 2, num_qudits).transpose(0, 2, 1).reshape(matrix.shape)
        args = ",".join([f"{one}*matrix", f"{num_trials}", f"{cutoff}"])
        commands = [
            'LoadPackage("QDistRnd", false);',
            *_get_gap_sparse_matrix_commands("matrix", field_order, riffled_matrix),
            f"Print(DistRandStab({args}:{kwargs}));",
        ]

    # Issue: Piped input somehow causes extra terminal output.  Fix: Ignore all but last line.
    result_lines = qldpc.external.gap.get_output(*commands, use_pipe=True).strip().splitlines()
    if not result_lines:
        raise ValueError("QDistRnd returned no output")
    output = result_lines[-1]

    # strip whitespace and comments, and interpret the remaining text as the bound
    lines = [line.strip() for line in output.splitlines()]
    bound = "".join([line for line in lines if not line.startswith("#")])
    if not bound:
        raise ValueError(f"Could not parse a distance bound from QDistRnd output: {output!r}")
    return int(bound)


def _get_gap_sparse_matrix_commands(
    matrix_var: str, field_order: int, matrix: npt.NDArray[np.int_]
) -> list[str]:
    """Get GAP commands that define a variable as a matrix over a finite field.

    The commands transfer the matrix in a sparse representation and assign it to the GAP variable
    named matrix_var.  Matrix entries are integers that encode elements of GF(field_order).
    """
    _, matrix_width = matrix.shape
    # Turn matrix into sparse representation where `nonzero_entries[i][j]` is a list of integers
    # where, for all values `l` in that list, `matrix[i,l] == j+1`.  Example:
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


if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
