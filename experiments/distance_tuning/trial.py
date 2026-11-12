# SPDX-License-Identifier: Apache-2.0

"""Time one workload under one tuning, in this process, after warming up its compiled code.

The script enters ``_tuning.use_tuning`` with the given changes, warms up the compiled code of the
workload's kind on small codes, and then times the workload, ``--repeat`` times in this process.
Codes and caches remember results, so only the first time is of a cold search; later repetitions
check whether the warm-up loaded all the compiled code that the search uses (they should not be
much faster).  It prints one JSON object: the workload, the tuning, the result of each repetition,
their wall and CPU seconds, and the peak resident memory after the warm-up and after the
repetitions.

``sweep.py`` runs this script in a new process for each trial.  To run it directly, limit it to
one thread, as ``sweep.py`` does, with ``NUMBA_NUM_THREADS=1`` and ``OMP_NUM_THREADS=1``.

Example:

    python experiments/distance_tuning/trial.py distance/gross/X \\
        --tuning '{"max_information_sets": 8}'
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import resource
import sys
import time
from typing import TYPE_CHECKING, Any

from qldpc import codes
from qldpc.codes.distance import CSSLogicalCounter, _tuning, compile_search_setup
from qldpc.codes.distance._search.coset_leader_search import CosetLeaderSearch
from qldpc.objects import Pauli

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.distance_tuning.*, and would silently treat a bare "import workloads" as untyped.
if TYPE_CHECKING:
    from experiments.distance_tuning import workloads
else:
    import workloads

# Changes that make small searches take every route of the BZ method, so that warming up on them
# loads the compiled code of every route that a timed search may take.
FORCE_EVERY_ROUTE = {
    "allow_enumeration_by_size": False,
    "allow_enumeration_by_work": False,
    "overlapping_minimum_work": 0,
    "partition_work_ratio": 0,
    "lp_minimum_work": 0,
    "automorphism_minimum_layer_work": 0,
    "completion_bound_minimum_layer_work": 0,
    "completion_bound_max_cost_percent": _tuning.MAX_VALUE,
}

# The codes of the warm-up.  Searches and caches remember results for them, so no timed workload
# may use them.
WARM_UP_CODES = ("classical:40:20:0", "surface:5", "toric:4")


def main() -> None:
    """Warm up, time the workload, and print the results."""
    args = parse_args()
    if args.workload.split("/")[1] in WARM_UP_CODES:
        raise ValueError(f"The warm-up uses the code of {args.workload!r}; choose another code")
    changes = json.loads(args.tuning)
    with _tuning.use_tuning(**changes) as tuning:
        if not args.cold:
            warm_up(args.workload.split("/")[0])
        maxrss_warm = get_maxrss()
        results, walls, cpus = [], [], []
        for _ in range(args.repeat):
            wall, cpu = time.perf_counter(), time.process_time()
            results.append(workloads.run_workload(args.workload))
            walls.append(time.perf_counter() - wall)
            cpus.append(time.process_time() - cpu)
    record: dict[str, Any] = {
        "workload": args.workload,
        "changes": changes,
        "tuning": dataclasses.asdict(tuning),
        "results": results,
        "wall": walls,
        "cpu": cpus,
        "maxrss_warm": maxrss_warm,
        "maxrss": get_maxrss(),
    }
    print(json.dumps(record), flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workload", help="a workload name (see workloads.py)")
    parser.add_argument(
        "--tuning",
        default="{}",
        help="a JSON object of changes to the fields of SearchTuning (default: no changes)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="how many times to run the workload in this process (default: %(default)s)",
    )
    parser.add_argument("--cold", action="store_true", help="skip the warm-up")
    return parser.parse_args()


def get_maxrss() -> int:
    """The peak resident memory of this process, in bytes."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss if sys.platform == "darwin" else 1024 * maxrss


####################################################################################################
# warm-up


def warm_up(kind: str) -> None:
    """Load the compiled code that the searches of one kind of workload may use.

    The warm-up runs small searches twice: under the active tuning, and with every route forced.
    Its codes, ``WARM_UP_CODES``, differ from the timed ones, so they leave no results for the
    timed search, and each of its distance searches first forgets the distances that the code
    remembers.  The searches of ``CosetLeaderSearch`` are cached for each tuning of their setup,
    so the two passes build their own.  The warm-up of ``ccount`` first compiles the setup of
    searches for this process.
    """
    if kind == "ccount":
        compile_search_setup()
        kind = "count"
    for changes in ({}, FORCE_EVERY_ROUTE):
        with _tuning.use_tuning(**changes):
            if kind == "distance":
                classical = workloads.get_code("classical:40:20:0")
                classical.forget_distance()
                classical.get_distance_exact()
                for name in ("surface:5", "toric:4"):
                    code = workloads.get_css_code(name)
                    for pauli in (Pauli.X, None):
                        code.forget_distance()
                        code.get_distance_exact(pauli)
            elif kind == "hadamard":
                hadamard_code = workloads.get_hadamard_code("surface:5")
                hadamard_code.forget_distance()
                hadamard_code.get_distance_exact()
            elif kind == "count":
                CSSLogicalCounter(codes.SurfaceCode(5)).count_logical_ops(5)
                CSSLogicalCounter(codes.ToricCode(4, rotated=True)).count_logical_ops(4)
            elif kind in ("coset", "first"):
                stabilizers = workloads.get_zero_state_stabilizers("surface:5")
                syndromes = workloads.get_random_syndromes(stabilizers, 3, 20, seed=1)
                for css_weight in (False, True):
                    search = CosetLeaderSearch(stabilizers, css_weight=css_weight)
                    search._get_min_weights(syndromes, 4)
                    search.find_first_not_within_max_weight(syndromes, 4)
                    search.get_min_weight_bound(int(syndromes[0, 0]), 4)
            else:
                raise ValueError(f"Unknown workload kind {kind!r}")


if __name__ == "__main__":
    main()
