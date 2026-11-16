#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Time the exact distance searches of qLDPC on pair-partition codes.

The benchmark times the connected-cluster and Brouwer-Zimmermann (BZ) methods on known codes, to
compare them, and plays no part in the tuning of the searches.  Its codes are the pair-partition
codes of ``workloads.INSTANCES`` (arXiv:2607.14091 and arXiv:2609.30069) or of other ``.cpm`` files
(see ``workloads.parse_cpm``), and, for contrast, ``gross`` (the [[144, 12, 12]] bivariate bicycle
code) and the rotated ``surface:D`` and ``toric:D`` codes of distance ``D``, which is even for a
toric code.  The script calls ``get_distance_exact`` once for each code, method, and sector, on one
core, and prints the distance and the wall and CPU time.  It flags a distance of an embedded
instance below the published one, or, for both sectors, different from it; Hadamards from
``--hadamards`` keep the distance.  It saves nothing.  Numba caches compiled code on disk, so run
the script twice for the times of a warm cache.  Some searches run for hours, so run them with a
hard deadline.

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
        default=["connected_cluster"],
        choices=["connected_cluster", "brouwer_zimmermann"],
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
        " which is not CSS, as a QuditCode; this searches the whole code as one sector, and ignores"
        " --sectors",
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

    A code that is not CSS is searched whole, as one sector.
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
