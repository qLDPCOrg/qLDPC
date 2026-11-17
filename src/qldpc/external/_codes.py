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

    QDistRnd does not support subsystem codes, but for a CSS code it bounds the dressed Z-distance,
    given the Z-type gauge generators as if they were stabilizers.  ``maxav`` is passed through to
    QDistRnd as a raw GAP value (default "fail"); see the QDistRnd documentation linked above.

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
        ValueError: If given the symplectic checks of a subsystem code, or if QDistRnd output is
            missing or invalid.
    """
    # Reject unsupported input before require_package, which can ask to install packages.
    if matrix_z is None and is_subsystem_code:
        raise ValueError(
            "QDistRnd cannot estimate the distance of a subsystem code from its symplectic checks;"
            " for a CSS subsystem code, convert it with code.to_css()"
        )
    qldpc.external.gap.require_package("GUAVA")
    qldpc.external.gap.require_package("QDistRnd", "https://github.com/QEC-pages/QDistRnd")

    field_order = type(matrix).order
    field = f"GF({field_order})"
    one = f"One({field})"
    cutoff = cutoff or 0
    kwargs = ",".join([f"field:={field}", f"maxav:={maxav}"])

    # QDistRnd fails on a matrix without rows, so give each such matrix a zero row, which keeps its
    # row space.
    if not len(matrix):
        matrix = type(matrix).Zeros((1, matrix.shape[1]))
    if matrix_z is not None and not len(matrix_z):
        matrix_z = type(matrix_z).Zeros((1, matrix_z.shape[1]))

    if matrix_z is not None:
        args = ",".join([f"{one}*matrix_x", f"{one}*matrix_z", f"{num_trials}", f"{cutoff}"])
        commands = [
            'LoadPackage("QDistRnd", false);;',
            *_get_gap_sparse_matrix_commands("matrix_x", type(matrix), matrix),
            *_get_gap_sparse_matrix_commands("matrix_z", type(matrix), matrix_z),
            f"Print(DistRandCSS({args}:{kwargs}));;",
        ]

    else:
        # "riffle" the parity check matrix to put X and Z support bits each qudit next to each other
        num_qudits = matrix.shape[1] // 2
        riffled_matrix = matrix.reshape(-1, 2, num_qudits).transpose(0, 2, 1).reshape(matrix.shape)
        args = ",".join([f"{one}*matrix", f"{num_trials}", f"{cutoff}"])
        commands = [
            'LoadPackage("QDistRnd", false);',
            *_get_gap_sparse_matrix_commands("matrix", type(matrix), riffled_matrix),
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

    # QDistRnd negates a bound at or below its argument mindist, the cutoff, to mark an early stop.
    return abs(int(bound))


def _get_gap_sparse_matrix_commands(
    matrix_var: str, field: type[galois.FieldArray], matrix: npt.NDArray[np.int_]
) -> list[str]:
    """Get GAP commands that define a variable as a matrix over a finite field.

    The commands transfer the matrix in a sparse representation and assign it to the GAP variable
    named matrix_var.  Matrix entries are the galois integers of elements of the field.
    """
    field_order = field.order
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

    field_elements = ",".join(get_gap_field_elements(field)[1:])
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


####################################################################################################
# finite-field elements in GAP and MAGMA


def get_gap_field_elements(field: type[galois.FieldArray]) -> list[str]:
    """GAP expressions for the elements of a finite field, indexed by their galois integers.

    GAP writes the elements of ``GF(q)`` as ``0*Z(q)`` and the powers of ``Z(q)``, a root of the
    Conway polynomial of ``GF(q)``.  GAP reads integers by its own conventions: ``i * One(GF(q))``
    is in the prime field, and GUAVA's ``CheckMatCode`` reads ``i`` as element ``i`` of GAP's sorted
    list of the elements of ``GF(q)``.  Neither agrees with the integers of galois for every field,
    such as ``GF(8)``, so these expressions write elements as GAP does.  galois builds ``GF(q)`` by
    default with the Conway polynomial, and then ``Z(q)`` is galois's primitive element: ``x``, or
    for a prime ``q``, the least primitive root modulo ``q``, which is the root of the Conway
    polynomial ``x - a``.  The element with galois integer ``v`` is then ``Z(q)^log(v)``.

    A field with another irreducible polynomial maps into the field with the Conway polynomial by
    sending its ``x`` to a root of its irreducible polynomial, which is a field isomorphism.  The
    choice among the roots, which are conjugate under the Frobenius automorphism, changes no weight
    of a word and no permutation automorphism of a code.
    """
    order = field.order
    conway_field = galois.GF(order)
    if field.irreducible_poly == conway_field.irreducible_poly:
        images = conway_field.Range(0, order)
    else:
        # evaluate the polynomial in x whose coefficients are the digits of each galois integer
        polynomial = galois.Poly(field.irreducible_poly.coeffs.view(np.ndarray), field=conway_field)
        root = polynomial.roots()[0]
        powers = root ** np.arange(field.degree)
        digits = field.Range(0, order).vector().view(np.ndarray)[:, ::-1]
        images = conway_field(digits) @ powers
    generator = conway_field.primitive_element
    return [f"0*Z({order})"] + [f"Z({order})^{int(power)}" for power in images[1:].log(generator)]


def get_magma_field(field: type[galois.FieldArray]) -> tuple[str, list[str]]:
    """MAGMA commands that define a finite field as ``F``, and its elements by galois integer.

    MAGMA maps an integer into a finite field through the prime field, so an integer cannot stand
    for an element outside the prime field.  For a prime field, the commands are empty and the
    elements are integers.  Otherwise, the commands define ``F`` as the extension of the prime field
    by the irreducible polynomial of the galois field, whose root is ``w``, and the element with
    galois integer ``v`` is the polynomial in ``w`` whose coefficients are the base-``p`` digits of
    ``v``.
    """
    if field.degree == 1:
        return "", [str(value) for value in range(field.order)]
    characteristic = field.characteristic
    coefficients = ",".join(map(str, field.irreducible_poly.coeffs[::-1].tolist()))
    definition = (
        f"F<w> := ext<GF({characteristic}) | Polynomial(GF({characteristic}), [{coefficients}])>; "
    )
    elements = ["0"]
    for digits in field.Range(1, field.order).vector().view(np.ndarray)[:, ::-1].tolist():
        terms = []
        for power, digit in enumerate(digits):
            if digit:
                coefficient = [str(digit)] if digit != 1 or power == 0 else []
                monomial = [] if power == 0 else ["w"] if power == 1 else [f"w^{power}"]
                terms.append("*".join(coefficient + monomial))
        elements.append("+".join(terms))
    return definition, elements


if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
