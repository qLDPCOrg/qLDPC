#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Compute exact distances of saved quantum Tanner codes, within a time limit per code.

The distances that run_search.py saves are randomized upper bounds, which can be far from exact.
This script computes the exact distance of every distinct code saved by run_search.py that has at
least one logical qubit, with the connected-support search of qLDPC (method "connected_support" of
qldpc.codes.CSSCode.get_distance_exact).  Saved codes count as distinct if their generating data
(subsets, local codes, and complex; see saved_codes.get_code_key) differ.  The exact distance of
each distinct code is computed once.

The cost of an exact distance grows rapidly with the distance and the weights of parity checks, so
it is hard to predict from the number of qubits.  Each computation therefore runs in its own Python
process, which is stopped if it takes longer than --time-limit seconds, including a second or more
to start.  Codes whose computation is stopped keep only the upper bound on their distance from
run_search.py.  Codes are processed in order of increasing size.  Interrupting this script with
Ctrl-C also stops its child processes, whereas killing only the process of this script leaves them
running, without a time limit, until they finish.  To stop them too, kill the process group of this
script.

Results are appended to exact_distances.csv in --results-dir (by default,
results/exact_distances.csv in this directory), with one row per finished or stopped computation.
The columns of this file are saved_codes.EXACT_DISTANCE_FIELDS:
- code_key: the key of the code (see saved_codes.get_code_key),
- code_file: the name of the saved file from which the code was loaded,
- status: "exact" if the computation finished, or "timeout" if it was stopped,
- distance: the exact distance, or nothing if the computation was stopped,
- time_limit: the time limit of the computation, in seconds, and
- seconds: the duration of the computation, in seconds.

The script skips codes that already have an exact distance, and codes whose computation was stopped
under a time limit at least as long as --time-limit.  Rerunning the script therefore resumes where
it left off, and rerunning it with a longer time limit retries the codes that were stopped.
collect_results.py reports the exact distances in this file.

Example:
    python experiments/quantum_tanner/compute_distances.py --time-limit 60 --workers 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import os
import pathlib
import subprocess
import sys
import time
from typing import TYPE_CHECKING

from qldpc import codes

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.quantum_tanner.*, and would silently treat a bare "import saved_codes" as untyped.
if TYPE_CHECKING:
    from experiments.quantum_tanner import saved_codes
else:
    import saved_codes

# default settings
TIME_LIMIT = 60  # seconds per exact distance computation
NUM_WORKERS = max(1, (os.cpu_count() or 1) - 2)  # number of parallel computations

####################################################################################################
# command-line interface


def main() -> None:
    """Compute exact distances of saved quantum Tanner codes in parallel processes."""
    args = parse_args()
    if args.code_file is not None:
        print_exact_distance(args.code_file)
        return

    # find the codes to compute, and open the file of results, writing its header if it is new
    codes_to_compute = get_codes_to_compute(args.results_dir, args.time_limit)
    print(f"Computing exact distances of {len(codes_to_compute)} codes")
    num_failures = 0
    output_path = args.results_dir / saved_codes.EXACT_DISTANCES_NAME
    is_new_output = not output_path.is_file() or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as file:
        writer = csv.writer(file)
        if is_new_output:
            writer.writerow(saved_codes.EXACT_DISTANCE_FIELDS)
            file.flush()

        # compute distances in child processes, and record each result as soon as it is known; on
        # an interruption, cancel the computations that have not yet started
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)
        try:
            futures = {
                executor.submit(compute_exact_distance, path, args.time_limit): (key, path)
                for key, path in codes_to_compute
            }
            for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                key, path = futures[future]
                try:
                    distance, seconds = future.result()
                except RuntimeError as error:
                    print(f"{index}/{len(futures)} {path.name}: failed\n{error}")
                    num_failures += 1
                    continue
                status = "timeout" if distance is None else "exact"
                writer.writerow(
                    [key, path.name, status, distance, args.time_limit, f"{seconds:.2f}"]
                )
                file.flush()
                result = "timeout" if distance is None else f"d = {distance}"
                print(f"{index}/{len(futures)} {path.name}: {result} ({seconds:.1f} s)")
        finally:
            executor.shutdown(cancel_futures=True)

    if num_failures:
        raise RuntimeError(f"{num_failures} of {len(codes_to_compute)} computations failed")


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=TIME_LIMIT,
        help="time limit of each exact distance computation, in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help="number of parallel computations (default: %(default)s)",
    )
    saved_codes.add_results_dir_argument(parser)

    # used internally to compute the distance of one code in a child process
    parser.add_argument("--code-file", type=pathlib.Path, help=argparse.SUPPRESS)

    args = parser.parse_args()
    for name in ["time_limit", "workers"]:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


####################################################################################################
# exact distances


def get_codes_to_compute(
    results_dir: pathlib.Path, time_limit: float
) -> list[tuple[str, pathlib.Path]]:
    """Get the keys and files of the distinct saved codes whose exact distances to compute.

    Skips codes without logical qubits, codes that already have an exact distance, and codes whose
    computation was stopped under a time limit at least as long as the given one.  Returns the codes
    in order of increasing size, each with the first of its saved files.
    """
    # identify the longest time limit under which each code was computed or stopped, treating the
    # time limit of a finished computation as infinite
    longest_time_limits: dict[str, float] = {}
    for record in saved_codes.read_exact_distance_records(results_dir):
        key = record["code_key"]
        time_limit_used = math.inf if record["status"] == "exact" else float(record["time_limit"])
        longest_time_limits[key] = max(time_limit_used, longest_time_limits.get(key, 0))

    # collect the first saved file of each distinct code that needs an exact distance
    codes_to_compute: dict[str, tuple[int, pathlib.Path]] = {}
    for path in saved_codes.get_saved_code_paths(results_dir / saved_codes.CODES_NAME):
        key = saved_codes.get_code_key(path)
        if key in codes_to_compute or longest_time_limits.get(key, 0) >= time_limit:
            continue
        _, (num_qudits, dimension, _), _ = saved_codes.read_code_header(path)
        if dimension > 0:
            codes_to_compute[key] = (num_qudits, path)

    return [
        (key, path) for key, (_, path) in sorted(codes_to_compute.items(), key=lambda item: item[1])
    ]


def compute_exact_distance(path: pathlib.Path, time_limit: float) -> tuple[int | None, float]:
    """Compute the exact distance of a saved code in a child process, within a time limit.

    Returns a pair: the exact distance (or None if the computation was stopped at the time limit)
    and the duration of the computation in seconds.

    Raises:
        RuntimeError: If the child process fails or prints no distance, with its output.
    """
    start = time.perf_counter()
    command = [sys.executable, __file__, "--code-file", str(path)]
    try:
        process = subprocess.run(
            command, capture_output=True, text=True, timeout=time_limit, check=False
        )
    except subprocess.TimeoutExpired:
        return None, time.perf_counter() - start
    if process.returncode != 0:
        raise RuntimeError(
            f"Child process exited with code {process.returncode}:\n{process.stderr}"
        )
    try:
        distance = int(process.stdout.splitlines()[-1])
    except (IndexError, ValueError):
        output = process.stdout + process.stderr
        raise RuntimeError(f"Child process printed no distance:\n{output}") from None
    return distance, time.perf_counter() - start


def print_exact_distance(path: pathlib.Path) -> None:
    """Print the exact distance of a saved code on the last line of output."""
    code = codes.QTCode.load(str(path))
    print(int(code.get_distance_exact(method="connected_support")))


if __name__ == "__main__":
    main()
