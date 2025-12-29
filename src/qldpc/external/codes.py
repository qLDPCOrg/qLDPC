"""Module for loading error-correcting codes from the GAP computer algebra system

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

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

import ast
import re
import urllib

import qldpc
import qldpc.cache
import qldpc.external.gap


@qldpc.cache.use_disk_cache(
    "codes",
    key_func=lambda code: "".join(code.split()),  # strip whitespace
)
def get_classical_code(code: str) -> tuple[list[list[int]], int | None]:
    """Retrieve a classical code from GAP."""
    qldpc.external.gap.require_package("GUAVA")

    # run GAP commands
    commands = [
        'LoadPackage("guava", false);;',
        f"code := {code};;",
        "mat := CheckMat(code);;",
        r'Print(LeftActingDomain(code), "\n");;',
        r'for vec in mat do Print(List(vec, x -> Int(x)), "\n");; od;;',
    ]
    code_str = qldpc.external.gap.get_output(*commands)

    # identify base field and retrieve parity checks
    field: int | None = None
    checks = []
    for line in code_str.splitlines():
        if not line.strip():
            continue

        if field is None and (match := re.search(r"GF\(([0-9]+(\^[0-9]+)?)\)", line)):
            base, exponent, *_ = (match.group(1) + "^1").split("^")
            field = int(base) ** int(exponent)
        else:
            checks.append(ast.literal_eval(line))

    if not checks:
        raise ValueError(f"Code has no parity checks: {code}")

    return checks, field


@qldpc.cache.use_disk_cache("qecdb")
def get_quantum_code(code_id: str) -> tuple[list[str], int, bool]:
    """Retrieve a quantum code from qecdb.org.

    Return the stabilizers of the code, its distance, and whether it's CSS.
    """
    url = f"https://qecdb.org/codes/{code_id}"
    try:
        lines = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
    except (urllib.error.URLError, urllib.error.HTTPError):
        print(f"ERROR: cannot access the webpage {url}")
        raise

    stab_line = next(line for line in lines if "<td>H</td>" in line)
    dist_line = next(line for line in lines if "<td>d</td>" in line)
    css_line = next(line for line in lines if "<td>css</td>" in line)

    stabilizers = re.findall("[IXYZ]+", stab_line)
    distance = int(re.findall(r"\d+", dist_line)[0])
    is_css = "True" in css_line
    return stabilizers, distance, is_css


def get_distance_bound(
    code: qldpc.codes.QuditCode,
    num_trials: int = 1,
    *,
    cutoff: int | None = None,
    maxav: str = "fail",
) -> int:
    """Estimate the distance of a quantum code using GAP's QDistRnd package.

    If given a CSSCode, estimate the Z-distance (minimum weight of a Z-type logical operator).
    See https://qec-pages.github.io/QDistRnd/doc/chap4.html.

    Note that QDistRnd does not support subsystem codes.  In the case of a CSS code, however, we
    can still compute the Z-distance by promoting all Z-type gauge group generators to stabilizers.
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
            f"matrix_x := {code_x.matrix_as_string()};;",
            f"matrix_z := {code_z.matrix_as_string()};;",
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
            f"matrix := {riffled_code.matrix_as_string()};",
            f"Print(DistRandStab({args}:{kwargs}));",
        ]

    output = qldpc.external.gap.get_output(*commands)

    # strip whitespace and comments, and interpret the remaining text as the bound
    lines = [line.strip() for line in output.splitlines()]
    bound = "".join([line for line in lines if not line.startswith("#")])
    return int(bound)
