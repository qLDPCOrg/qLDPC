#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Time the exact distance searches of qLDPC on pair-partition codes.

The benchmark times the connected-support method and Brouwer-Zimmermann (BZ).  The embedded
instances are ``qldpc.codes.PairPartitionCode`` codes, quasi-cyclic CSS codes with sparse parity
checks, mostly of high rate (arXiv:2607.14091 and arXiv:2609.30069).  They come from
https://github.com/ultra-high-rate-qec/design-principles-data (``codes/css/cpm_pp/instances``), and
https://github.com/kasaikenta/pair-partition-cpm-css-codes also publishes the ``276``, ``518``, and
``944`` instances.  They are in the ``.cpm`` format, a whitespace-separated token stream:
``CPM_CSS_V1``, a name, the lift size ``P``, the number ``L`` of block columns, the numbers ``J_X``
and ``J_Z`` of block rows, and then the X-type exponents and the Z-type exponents in row-major
order.  A path to another such file also names a code.

With ``--hadamards``, the script applies Hadamards to the second half of the qubits of each code,
which leaves a code with the same distance that is not CSS, and times it as a
``qldpc.codes.QuditCode``.

With ``--gauge K``, the script makes the first ``K`` logical qubits of each code gauge qubits, by
adding their logical operators to the checks.  The subsystem code that results has a dressed
distance of at least that of the code, and its connected-support search makes its stabilizers
sparser.

The script also builds a few other codes to contrast the two methods: ``gross`` (the [[144, 12, 12]]
bivariate bicycle code), and ``surface:D`` and ``toric:D`` (rotated, with distance ``D``, which is
even for a toric code).

It calls ``get_distance_exact`` once for each code, method, and sector, and prints the distance, and
the wall and CPU time.  It flags a distance of an embedded instance that contradicts its published
distance: a distance of both sectors that differs from it, or a distance of one sector that is
lower.  The searches run on one core.  Numba compiles on first use and caches the compiled code on
disk, so run the script twice for the times of a warm cache.  The script prints its
results, and saves nothing.

Some searches run for hours, so run them under a guard, such as a hard deadline.

Examples:

    python experiments/connected_support_distance/benchmark.py 110 232 276
    python experiments/connected_support_distance/benchmark.py 232 --methods brouwer_zimmermann
    python experiments/connected_support_distance/benchmark.py gross surface:11 surface:15
    python experiments/connected_support_distance/benchmark.py 110 232 --hadamards
"""

from __future__ import annotations

import argparse
import pathlib
import time
from collections.abc import Sequence
from typing import Literal

import numpy as np

from qldpc import codes
from qldpc.codes.distance import DistanceMethod
from qldpc.objects import Pauli

####################################################################################################
# command-line interface


def main() -> None:
    """Time the requested searches."""
    args = parse_args()
    for name in args.codes:
        css_code, parameters = get_code(name)
        if args.gauge:
            css_code = codes.CSSCode(
                np.vstack([css_code.matrix_x, css_code.get_logical_ops(Pauli.X)[: args.gauge]]),
                np.vstack([css_code.matrix_z, css_code.get_logical_ops(Pauli.Z)[: args.gauge]]),
            )
        print(f"{name}: n={len(css_code)}, k={css_code.dimension}", flush=True)
        code: codes.QuditCode = css_code
        if args.hadamards:
            # Hadamards on half of the qubits make a code that is not CSS, with the same distance.
            num_qubits = len(css_code)
            code = codes.QuditCode(css_code.conjugated(range(num_qubits // 2, num_qubits)).matrix)
        sectors: Sequence[Literal["X", "Z", "both"]] = ["both"] if args.hadamards else args.sectors
        for method in args.methods:
            for sector in sectors:
                distance, wall, cpu = time_distance(code, method, sector)
                note = ""
                if parameters is not None and (
                    distance < parameters[2]
                    or (sector == "both" and not args.gauge and distance != parameters[2])
                ):
                    note = f"  (differs from the published distance {parameters[2]})"
                print(
                    f"  {method:19s} {sector:4s} d={distance}  wall {wall:.2f} s  cpu {cpu:.2f} s"
                    + note,
                    flush=True,
                )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "codes",
        nargs="+",
        help="embedded instances (" + ", ".join(INSTANCES) + "), gross, surface:D, toric:D, or"
        " .cpm files",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["connected_support"],
        choices=["connected_support", "brouwer_zimmermann"],
        help="the exact-distance methods to time (default: %(default)s)",
    )
    parser.add_argument(
        "--sectors",
        nargs="+",
        default=["both"],
        choices=["X", "Z", "both"],
        help="the logical operators to search: X-type, Z-type, or both, whose joint search finds"
        " the minimum (default: %(default)s)",
    )
    parser.add_argument(
        "--hadamards",
        action="store_true",
        help="apply Hadamards to the second half of the qubits, and search the resulting code,"
        " which is not CSS, as a QuditCode; this searches both sectors, and ignores --sectors",
    )
    parser.add_argument(
        "--gauge",
        type=int,
        default=0,
        metavar="K",
        help="make the first K logical qubits gauge qubits (default: %(default)s)",
    )
    return parser.parse_args()


####################################################################################################
# embedded instances

# The parameters [[n, k, d]] and the ``.cpm`` text of each embedded instance.
INSTANCES: dict[str, tuple[tuple[int, int, int], str]] = {
    "110": (
        (110, 8, 12),
        """
        CPM_CSS_V1 j5l10_p11_c0002 11 10 5 5 0 0 0 0 0 0 0 0 0 0 0 1 6 3 10 9 5 7 8 2 0 8 4 2 3 1
        6 9 5 7 0 5 8 4 6 7 2 10 3 1 0 10 5 8 1 6 9 2 4 3 0 6 3 7 5 6 5 0 3 7 0 3 7 9 8 7 0 9 8 3
        0 9 10 5 2 0 9 10 5 2 0 4 2 1 7 1 2 7 4 0 0 2 1 6 9 9 6 2 0 1
        """,
    ),
    "170": (
        (170, 8, 20),
        """
        CPM_CSS_V1 j5l10_p17_c0000 17 10 5 5 0 0 0 0 0 0 0 0 0 0 0 15 4 3 8 7 6 5 10 2 0 16 12 5 3
        1 10 14 13 15 0 10 11 12 4 9 3 16 8 1 0 6 10 16 15 11 1 2 9 7 0 11 8 15 6 11 6 0 8 15 0 7
        9 14 10 9 0 14 10 7 0 1 16 13 2 0 1 16 13 2 0 9 3 1 5 1 3 5 9 0 0 2 1 10 7 7 10 2 0 1
        """,
    ),
    "232": (
        (232, 62, 12),
        """
        CPM_CSS_V1 j3l8_p29_g8_orbit_05025 29 8 3 3 0 0 0 0 0 0 0 0 0 5 4 18 2 1 8 16 0 10 24 15 3
        23 11 12 0 16 3 19 3 0 19 16 0 1 2 12 0 1 2 12 0 27 23 2 2 23 27 0
        """,
    ),
    "276": (
        (276, 98, 14),
        """
        CPM_CSS_V1 qc_276_98_14 23 12 4 4 9 3 11 13 14 19 5 22 10 15 2 21 16 20 11 9 18 12 18 17
        19 4 7 4 19 20 18 11 4 18 17 8 11 21 2 14 5 18 1 13 12 13 13 5 1 14 9 3 20 2 17 7 15 8 16
        21 16 9 3 10 2 8 11 5 3 22 5 2 15 4 18 13 22 21 7 4 8 20 19 12 4 10 3 17 12 14 13 14 6 19
        20 1 13 15 3 9
        """,
    ),
    "372": (
        (372, 130, 18),
        """
        CPM_CSS_V1 qc_372_130_18 31 12 4 4 21 3 24 1 7 23 17 30 24 4 19 23 5 7 27 23 9 28 24 5 2 1
        2 10 10 6 7 2 0 13 28 24 8 16 4 27 12 11 23 6 3 5 17 17 18 14 5 27 20 24 26 15 2 9 16 20
        26 18 14 9 2 5 3 16 11 0 27 7 26 8 0 7 13 8 0 9 29 10 25 22 15 9 6 30 19 27 27 29 14 25 24
        2 22 6 16 16
        """,
    ),
    "518": (
        (518, 228, 16),
        """
        CPM_CSS_V1 qc_518_228_16 37 14 4 4 3 3 4 5 6 7 8 9 10 11 12 13 14 15 12 13 16 31 29 24 13
        3 32 1 32 6 5 33 10 18 29 33 8 21 31 10 6 21 10 16 9 15 1 25 25 15 31 15 20 11 30 35 15 20
        4 28 13 8 27 13 10 36 23 13 33 19 27 18 24 7 2 21 15 6 9 9 29 20 31 13 11 14 32 18 2 24 12
        1 7 34 31 30 6 34 2 21 15 14 9 20 11 3 18 9 34 35 16 23 29 15 12 22
        """,
    ),
    "944": (
        (944, 478, 20),
        """
        CPM_CSS_V1 qc_944_478_20 59 16 4 4 13 53 38 20 16 56 9 15 54 11 15 32 4 20 58 27 17 2 2 11
        17 49 55 51 54 1 22 38 49 29 4 8 18 30 51 22 37 31 58 22 12 33 56 15 44 26 3 22 21 38 30
        48 15 35 19 58 19 14 54 58 29 8 26 57 34 17 25 46 14 37 37 42 18 57 13 53 31 46 39 55 10
        11 15 38 15 2 12 24 4 14 20 31 22 56 16 24 11 50 52 19 31 45 50 17 9 26 17 29 45 18 57 16
        10 13 40 20 14 55 15 40 1 10 15 57 1 18 1 46
        """,
    ),
}


####################################################################################################
# codes


def parse_cpm(text: str) -> codes.PairPartitionCode:
    """Build the pair-partition code of a ``.cpm`` token stream."""
    magic, _name, *numbers = text.split()
    if magic != "CPM_CSS_V1":
        raise ValueError(f"Expected a CPM_CSS_V1 file, not {magic!r}")
    lift_size, block_length, num_rows_x, num_rows_z, *exponents = map(int, numbers)
    values = np.array(exponents)
    split = num_rows_x * block_length
    exponents_x = values[:split].reshape(num_rows_x, block_length)
    exponents_z = values[split:].reshape(num_rows_z, block_length)
    return codes.PairPartitionCode(exponents_x, exponents_z, lift_size)


def get_code(name: str) -> tuple[codes.CSSCode, tuple[int, int, int] | None]:
    """Build a named code, and return it with its published parameters if it has any."""
    if name in INSTANCES:
        parameters, text = INSTANCES[name]
        return parse_cpm(text), parameters
    kind, _, size = name.partition(":")
    if kind == "gross":
        return codes.GrossCode(), None
    if kind == "surface":
        return codes.SurfaceCode(int(size)), None
    if kind == "toric":
        return codes.ToricCode(int(size), rotated=True), None
    if pathlib.Path(name).is_file():
        return parse_cpm(pathlib.Path(name).read_text(encoding="utf-8")), None
    raise ValueError(f"Unknown code {name!r}")


####################################################################################################
# timing


def time_distance(
    code: codes.QuditCode, method: DistanceMethod, sector: Literal["X", "Z", "both"]
) -> tuple[int | float, float, float]:
    """The distance of a code, and the wall and CPU seconds that it took.

    A code that is not CSS has no sectors, and is searched whole.
    """
    code.forget_distance()
    wall, cpu = time.perf_counter(), time.process_time()
    if isinstance(code, codes.CSSCode):
        pauli = None if sector == "both" else Pauli.coerce_xz(sector)
        distance = code.get_distance_exact(pauli, method=method)
    else:
        distance = code.get_distance_exact(method=method)
    return distance, time.perf_counter() - wall, time.process_time() - cpu


if __name__ == "__main__":
    main()
