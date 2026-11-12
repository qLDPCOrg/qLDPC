# SPDX-License-Identifier: Apache-2.0

"""Compare tunings of the exact distance searches on workloads, in paired, resumable trials.

A sweep compares arms, each a set of changes to the fields of ``SearchTuning``, on workloads (see
``workloads.py``).  Each trial runs ``trial.py`` in a new process, limited to one thread, which
warms up and then times one workload under one arm.  Each round runs every arm once on each
workload, in an order that alternates between rounds, so that drifts of the load of the machine
affect every arm alike.  With ``--jobs`` above one, groups of trials, each one round of one
workload, run at once, while the arms of a group run one after another on one worker, so that they
share the conditions of the machine as closely as possible.  Completion tables saved less when
several searches ran at once (see the Retuning section of ``qldpc.codes.distance._tuning``), so
compare their cost models with ``--jobs 1``.

Each trial appends one JSON line to ``trials.jsonl`` in the results directory, with the sweep's tag,
the arm, the round, the commit and a hash of the differences of the source from it, the load
average, and the record of ``trial.py``, or the error or timeout of the trial.  A sweep skips trials
that the file already holds, so an interrupted sweep resumes where it stopped.
``collect_results.py`` summarizes the file.

``--repeat`` times the workload several times in each trial's process, and ``collect_results.py``
takes the median of the repetitions.  The first repetition also does work that codes remember and
that does not depend on the tuning (see ``trial.py``), so with three or more repetitions the median
leaves it out.  Distance, hadamard, and count workloads repeat their searches; a workload of
``CosetLeaderSearch`` reuses its cached searches.

Arms are given either as ``--field`` and ``--values``, which compares values of one field, or as
``--arms``, a JSON list of objects of changes.  ``--base`` adds changes to every arm.  The first
arm is the reference of ``collect_results.py``, so list the current default first.

Run sweeps under a guard of memory and time that allows ``--jobs`` processes.

Examples:

    python experiments/distance_tuning/sweep.py --tag noise --rounds 10 \\
        --workloads distance/gross/X count/gross/12 --arms '[{}, {}]'
    python experiments/distance_tuning/sweep.py --tag max-sets --rounds 6 \\
        --workloads distance/pp232/X distance/lp13s5/X --field max_information_sets \\
        --values 16 8 24 32
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

DIRECTORY = pathlib.Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parent.parent
TRIALS_NAME = "trials.jsonl"  # name of the file of trials in the results directory

# Environment variables that limit a trial to one thread.
SINGLE_THREAD = {
    variable: "1"
    for variable in (
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
}


def main() -> None:
    """Run the trials of a sweep that the results directory lacks."""
    args = parse_args()
    arms = get_arms(args)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    path = args.results_dir / TRIALS_NAME
    source = get_source_fingerprint()
    done = get_done_keys(path)
    groups = [
        [
            {
                "tag": args.tag,
                "workload": workload,
                "round": round_index,
                "position": position,
                "arm_index": arm_index,
                "arm": arm,
                "source": source,
                "timeout": args.timeout,
                "repeat": args.repeat,
            }
            for position, (arm_index, arm) in enumerate(order_arms(arms, round_index))
        ]
        for round_index in range(args.rounds)
        for workload in args.workloads
    ]
    groups = [[trial for trial in group if get_key(trial) not in done] for group in groups]
    groups = [group for group in groups if group]
    num_trials = sum(map(len, groups))
    print(f"{num_trials} trials to run, in {len(groups)} groups", flush=True)

    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        # Each group's arms run one after another, in their alternating order, on one worker.
        futures = [executor.submit(run_group, group, path, lock) for group in groups]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tag", required=True, help="a name of the sweep, recorded with trials")
    parser.add_argument("--workloads", nargs="+", required=True, help="workload names")
    parser.add_argument("--field", help="a field of SearchTuning whose values to compare")
    parser.add_argument(
        "--values", nargs="+", help="values of --field, as JSON, such as 16 or 2.0 or true"
    )
    parser.add_argument("--arms", help="a JSON list of objects of changes, one for each arm")
    parser.add_argument(
        "--base", default="{}", help="a JSON object of changes for every arm (default: none)"
    )
    parser.add_argument(
        "--rounds", type=int, default=6, help="rounds of every arm (default: %(default)s)"
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="trials that run at once (default: %(default)s)"
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="times to run the workload in each trial's process (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=3600,
        help="seconds after which a trial is stopped and recorded as timed out"
        " (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=DIRECTORY / "results",
        help="the directory of trials.jsonl (default: results/ beside this script)",
    )
    args = parser.parse_args()
    if (args.field is None) == (args.arms is None) or (args.field is None) != (args.values is None):
        parser.error("give either --field and --values, or --arms")
    return args


def get_arms(args: argparse.Namespace) -> list[dict[str, Any]]:
    """The changes of each arm, with the base changes."""
    base = json.loads(args.base)
    if args.arms is not None:
        arms = json.loads(args.arms)
    else:
        arms = [{args.field: json.loads(value)} for value in args.values]
    return [base | arm for arm in arms]


def order_arms(arms: list[dict[str, Any]], round_index: int) -> list[tuple[int, dict[str, Any]]]:
    """The indices and changes of the arms in the order of a round.

    The order is forward in even rounds and backward in odd ones, rotated by half the round index,
    so that over twice as many rounds as arms, every arm takes every position.
    """
    indexed = list(enumerate(arms))
    shift = (round_index // 2) % len(arms)
    rotated = indexed[shift:] + indexed[:shift]
    return rotated if round_index % 2 == 0 else rotated[::-1]


####################################################################################################
# trials


def get_key(trial: dict[str, Any]) -> str:
    """A hash of what determines a trial's result.

    It leaves out the position and the timeout, and the number of repetitions if it is one, so that
    the trials of sweeps from before ``--repeat`` keep their keys.
    """
    names = ("tag", "workload", "round", "arm_index", "arm", "source")
    fields = {name: trial[name] for name in names}
    if trial.get("repeat", 1) != 1:
        fields["repeat"] = trial["repeat"]
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def get_done_keys(path: pathlib.Path) -> set[str]:
    """The keys of the trials that a file of trials holds."""
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as file:
        return {json.loads(line)["key"] for line in file if line.strip()}


def run_group(group: list[dict[str, Any]], path: pathlib.Path, lock: threading.Lock) -> None:
    """Run the trials of one round of one workload, one after another, and save each."""
    for trial in group:
        record = run_trial(trial)
        line = json.dumps(record)
        with lock:
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")
            status = record["status"]
            wall = f"{statistics.median(record['wall']):.3f} s" if status == "ok" else status
            arm = f"arm {trial['arm_index']} {trial['arm']}"
            print(f"{trial['workload']} round {trial['round']} {arm}: {wall}", flush=True)


def run_trial(trial: dict[str, Any]) -> dict[str, Any]:
    """Run one trial in a new process."""
    command = [
        sys.executable,
        str(DIRECTORY / "trial.py"),
        trial["workload"],
        "--tuning",
        json.dumps(trial["arm"]),
        "--repeat",
        str(trial.get("repeat", 1)),
    ]
    record = trial | {"key": get_key(trial), "load": os.getloadavg(), "start": time.time()}
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=trial["timeout"],
            env=os.environ | SINGLE_THREAD,
            cwd=REPOSITORY,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return record | {"status": "timeout"}
    if process.returncode != 0:
        return record | {"status": "error", "stderr": process.stderr[-4000:]}
    return record | {"status": "ok"} | json.loads(process.stdout.strip().splitlines()[-1])


def get_source_fingerprint() -> dict[str, str]:
    """The commit of the source of qLDPC, and a hash of its uncommitted differences."""
    commit = git("rev-parse", "HEAD").strip()
    diff = git("diff", "HEAD", "--", "src")
    return {"commit": commit, "diff": hashlib.sha256(diff.encode()).hexdigest()[:16]}


def git(*args: str) -> str:
    """The output of a git command in the repository."""
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=REPOSITORY
    ).stdout


def iter_trials(path: pathlib.Path) -> Iterator[dict[str, Any]]:
    """The trials of a file of trials."""
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


if __name__ == "__main__":
    main()
