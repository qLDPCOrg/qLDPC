#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Time the exact distance searches of qLDPC on pair-partition codes.

The benchmark times the connected-support method and Brouwer-Zimmermann (BZ).  The embedded
instances, in ``workloads.py``, are ``qldpc.codes.PairPartitionCode`` codes, quasi-cyclic CSS codes
with sparse parity checks, mostly of high rate (arXiv:2607.14091 and arXiv:2609.30069), whose
sources ``workloads.py`` names.  A path to another file in their ``.cpm`` format (see
``workloads.parse_cpm``) also names a code.

With ``--hadamards``, the script applies Hadamards to the second half of the qubits of each code,
which leaves a code with the same distance that is not CSS, and times it as a
``qldpc.codes.QuditCode``.

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

    python experiments/distance_tuning/benchmark.py 110 232 276
    python experiments/distance_tuning/benchmark.py 232 --methods brouwer_zimmermann
    python experiments/distance_tuning/benchmark.py gross surface:11 surface:15
    python experiments/distance_tuning/benchmark.py 110 232 --hadamards
"""

from __future__ import annotations

import argparse
import pathlib
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from qldpc import codes
from qldpc.codes.distance import DistanceMethod
from qldpc.objects import Pauli

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.distance_tuning.*, and would silently treat a bare "import workloads" as untyped.
if TYPE_CHECKING:
    from experiments.distance_tuning import workloads
else:
    import workloads

####################################################################################################
# command-line interface


def main() -> None:
    """Time the requested searches."""
    args = parse_args()
    for name in args.codes:
        css_code, parameters = get_code(name)
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
                    distance < parameters[2] or (sector == "both" and distance != parameters[2])
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
        help="embedded instances (" + ", ".join(workloads.INSTANCES) + "), gross, surface:D,"
        " toric:D, or"
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
    return parser.parse_args()


####################################################################################################
# codes


def get_code(name: str) -> tuple[codes.CSSCode, tuple[int, int, int] | None]:
    """Build a named code, and return it with its published parameters if it has any."""
    if name in workloads.INSTANCES:
        name = f"pp{name}"
    elif not (
        name == "gross" or name.startswith(("surface:", "toric:")) or pathlib.Path(name).is_file()
    ):
        raise ValueError(f"Unknown code {name!r}")
    return workloads.get_css_code(name), workloads.get_published_parameters(name)


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
