# SPDX-License-Identifier: Apache-2.0

"""Compare the whole current tuning of the exact searches with a whole proposed one, by timing.

The comparison catches gross regressions or bugs when the tuning or the search code changes.  It
never sets or justifies a value: the Retuning section of the module docstring of
``qldpc.codes.distance._tuning`` says when a default changes.  The script times workloads (see
``workloads.py``) under two arms: the current tuning, the defaults of ``SearchTuning``, and the
proposed tuning, those defaults with the changes that ``--proposed`` gives.  With ``--worktrees``,
the arms are instead the defaults of two checkouts of qLDPC, the current one first.  Each checkout
runs its own ``trial.py`` and imports its own source, with the Python of this process.

Each trial runs ``trial.py`` in a new process, limited to one thread, which warms up and then times
one workload under one arm, and the trials run one at a time.  Each round runs both arms on each
workload, one after the other, in an order that alternates between rounds and between workloads, so
that drifts of the load of the machine affect both arms alike.  Each arm keeps the bytecode of its
source in its own directory of the results directory, so that, after its first trial, neither arm
compiles modules within its timed searches.  Before any trial, the script checks that each arm
imports qLDPC from the ``src`` of its checkout.

Each trial appends one JSON line to ``trials.jsonl`` in the results directory, with the tag of the
comparison, the arm, the round, the checkout, changes, and source of each arm (see
``get_source_fingerprint``), the load average, and the record of ``trial.py``, or the error or
timeout of the trial.  A comparison skips the trials that the file already holds, so an interrupted
comparison resumes where it stopped, and running a finished one again prints its summary.

The script summarizes a comparison by pairing the arms of each round, among the trials of the
arms, workloads, and repetitions that the command gives.  For each workload, the summary gives the
successful trials of each arm, the median wall seconds of each arm (of a trial, the median of its
repetitions), and the median, over rounds, of the difference and of the ratio of the proposed arm's
time to the current arm's.  It also gives the least and greatest ratio, a distribution-free 95%
confidence interval of the median ratio from six rounds (see ``get_median_interval``), and the
median peak resident memory of each arm, in MiB.  Ratios of one round share the conditions of the
machine, so their spread measures noise better than the spread of times.  The summary flags a
workload whose arms disagree on a result, which no tuning may change except a distance search with
an early exit.  The script prints the summary, and writes it to ``summary-<tag>.csv`` in the
results directory.

``--repeat`` times the workload several times in each trial's process.  The first repetition also
does work that codes remember and that does not depend on the tuning (see ``trial.py``), so with
three or more repetitions the median leaves it out.  But a code also remembers the automorphisms
that a search finds, and whether a search finds them depends on the tuning, so time a change of
when searches ask for automorphisms with one repetition.  Distance, hadamard, and count workloads
repeat their searches; a workload of ``CosetLeaderSearch`` reuses its cached searches.

Run comparisons under a guard of memory and time.

Examples, the first of which proposes a value of ``automorphism_minimum_states``, and the second of
which compares the defaults of two checkouts:

    python experiments/distance_tuning/sweep.py --tag automorphism-states --rounds 3 \\
        --workloads ccdistance/lp13s5/X ccdistance/surface:17/X \\
        --proposed '{"automorphism_minimum_states": 100000}'
    python experiments/distance_tuning/sweep.py --tag branch --rounds 3 \\
        --workloads distance/qt7/X count/gross/12 --worktrees ../qLDPC-main .
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any

DIRECTORY = pathlib.Path(__file__).resolve().parent
REPOSITORY = DIRECTORY.parent.parent
TRIALS_NAME = "trials.jsonl"  # name of the file of trials in the results directory
ARMS = ("current", "proposed")

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

# columns of the summary file
CSV_FIELDS = (
    "workload",
    "trials_current",
    "trials_proposed",
    "failed",
    "median_wall_current",
    "median_wall_proposed",
    "median_difference",
    "median_ratio",
    "ci_low",
    "ci_high",
    "min_ratio",
    "max_ratio",
    "median_maxrss_mib_current",
    "median_maxrss_mib_proposed",
)


def main() -> None:
    """Run the trials of a comparison that the results directory lacks, and summarize them."""
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    path = args.results_dir / TRIALS_NAME
    arms = get_arms(args)
    for name, settings in arms.items():
        check_import(settings, get_environment(settings, name, args.results_dir))

    # The trials of every round, whose order of arms alternates between rounds and workloads.
    trials = []
    for round_index in range(args.rounds):
        for workload_index, workload in enumerate(args.workloads):
            order = ARMS if (round_index + workload_index) % 2 == 0 else ARMS[::-1]
            for arm in order:
                trial = {
                    "tag": args.tag,
                    "workload": workload,
                    "round": round_index,
                    "arm": arm,
                    "arms": arms,
                    "repeat": args.repeat,
                }
                trials.append(trial | {"key": get_key(trial)})

    # Run the trials that the file lacks, and save each.
    done = {trial["key"] for trial in read_trials(path)}
    to_run = [trial for trial in trials if trial["key"] not in done]
    print(f"{len(to_run)} of {len(trials)} trials to run", flush=True)
    for trial in to_run:
        record = run_trial(trial, args.timeout, args.results_dir)
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
        status = record["status"]
        wall = f"{statistics.median(record['wall']):.3f} s" if status == "ok" else status
        print(f"{trial['workload']} round {trial['round']} {trial['arm']}: {wall}", flush=True)

    # Summarize the trials of this comparison.
    keys = {trial["key"] for trial in trials}
    rows = summarize([trial for trial in read_trials(path) if trial["key"] in keys])
    with (args.results_dir / f"summary-{args.tag}.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print_rows(rows)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--tag", required=True, help="a name of the comparison, recorded with trials"
    )
    parser.add_argument("--workloads", nargs="+", required=True, help="workload names")
    parser.add_argument(
        "--proposed", help="a JSON object of changes to the defaults, the proposed tuning"
    )
    parser.add_argument(
        "--worktrees",
        nargs=2,
        type=pathlib.Path,
        metavar=("CURRENT", "PROPOSED"),
        help="two checkouts of qLDPC whose defaults to compare, instead of --proposed",
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="rounds of both arms (default: %(default)s)"
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
    if (args.proposed is None) == (args.worktrees is None):
        parser.error("give either --proposed or --worktrees")
    # Trials run in the directories of their checkouts, so they need an absolute path.
    args.results_dir = args.results_dir.resolve()
    return args


def get_arms(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    """The checkout, the changes of the tuning, and the source of each arm."""
    if args.worktrees is not None:
        checkouts = [worktree.resolve() for worktree in args.worktrees]
        changes: list[dict[str, Any]] = [{}, {}]
    else:
        checkouts = [REPOSITORY, REPOSITORY]
        changes = [{}, json.loads(args.proposed)]
    return {
        arm: {
            "checkout": str(checkout),
            "changes": arm_changes,
            "source": get_source_fingerprint(checkout),
        }
        for arm, checkout, arm_changes in zip(ARMS, checkouts, changes)
    }


####################################################################################################
# trials


def get_key(trial: dict[str, Any]) -> str:
    """A hash of what determines a trial's result."""
    names = ("tag", "workload", "round", "arm", "arms", "repeat")
    fields = {name: trial[name] for name in names}
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()[:16]


def run_trial(trial: dict[str, Any], timeout: float, results_dir: pathlib.Path) -> dict[str, Any]:
    """Run one trial in a new process, and return its record."""
    arm = trial["arms"][trial["arm"]]
    checkout = pathlib.Path(arm["checkout"])
    command = [
        sys.executable,
        str(checkout / "experiments" / "distance_tuning" / "trial.py"),
        trial["workload"],
        "--tuning",
        json.dumps(arm["changes"]),
        "--repeat",
        str(trial["repeat"]),
    ]
    record = trial | {"load": os.getloadavg(), "start": time.time()}
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=get_environment(arm, trial["arm"], results_dir),
            cwd=checkout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return record | {"status": "timeout"}
    if process.returncode != 0:
        return record | {"status": "error", "stderr": process.stderr[-4000:]}
    return record | {"status": "ok"} | json.loads(process.stdout.strip().splitlines()[-1])


def get_environment(arm: dict[str, Any], name: str, results_dir: pathlib.Path) -> dict[str, str]:
    """The environment of the processes of an arm: one thread, its source, and its bytecode."""
    environment = os.environ | SINGLE_THREAD
    paths = [str(pathlib.Path(arm["checkout"]) / "src"), os.environ.get("PYTHONPATH")]
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, paths))
    environment["PYTHONPYCACHEPREFIX"] = str(results_dir / "pycache" / name)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    return environment


def check_import(arm: dict[str, Any], environment: dict[str, str]) -> None:
    """Check that the processes of an arm import qLDPC from the source of its checkout."""
    expected = (pathlib.Path(arm["checkout"]) / "src" / "qldpc").resolve()
    process = subprocess.run(
        [sys.executable, "-c", "import pathlib, qldpc; print(pathlib.Path(qldpc.__file__).parent)"],
        capture_output=True,
        text=True,
        env=environment,
        cwd=arm["checkout"],
        check=True,
    )
    imported = pathlib.Path(process.stdout.strip()).resolve()
    if imported != expected:
        raise RuntimeError(f"The arm of {arm['checkout']} imports qLDPC from {imported}")


def check_local_import() -> None:
    """Check that this process imports qLDPC from the ``src`` of the checkout of these scripts."""
    spec = importlib.util.find_spec("qldpc")
    imported = pathlib.Path(spec.origin).resolve().parent if spec and spec.origin else None
    expected = (REPOSITORY / "src" / "qldpc").resolve()
    if imported != expected:
        raise RuntimeError(
            f"This process imports qLDPC from {imported}, not from {expected}; run the script with"
            " the Python of its checkout"
        )


def get_source_fingerprint(repository: pathlib.Path = REPOSITORY) -> dict[str, str]:
    """The commit of a checkout of qLDPC, and a hash of the uncommitted changes of its source."""
    commit = git(repository, "rev-parse", "HEAD").strip()
    diff = git(repository, "diff", "HEAD", "--", "src")
    return {"commit": commit, "diff": hashlib.sha256(diff.encode()).hexdigest()[:16]}


def git(repository: pathlib.Path, *args: str) -> str:
    """The output of a git command in a repository."""
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True, cwd=repository
    ).stdout


def read_trials(path: pathlib.Path) -> list[dict[str, Any]]:
    """The trials of a file of trials, if it exists."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


####################################################################################################
# summary


def summarize(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row for each workload of a comparison."""
    by_workload: dict[str, list[dict[str, Any]]] = {}
    for trial in trials:
        by_workload.setdefault(trial["workload"], []).append(trial)
    rows = []
    for workload, group in by_workload.items():
        ok = {arm: [t for t in group if t["arm"] == arm and t["status"] == "ok"] for arm in ARMS}
        walls = {arm: {t["round"]: get_wall(t) for t in ok[arm]} for arm in ARMS}
        rounds = sorted(walls["current"].keys() & walls["proposed"].keys())
        ratios = sorted(walls["proposed"][r] / walls["current"][r] for r in rounds)
        differences = [walls["proposed"][r] - walls["current"][r] for r in rounds]
        ci_low, ci_high = get_median_interval(ratios)
        results = {result for arm in ARMS for t in ok[arm] for result in t["results"]}
        if len(results) > 1:
            print(f"WARNING: the arms disagree on {workload}: {sorted(results)}")
        rows.append(
            {
                "workload": workload,
                "trials_current": len(ok["current"]),
                "trials_proposed": len(ok["proposed"]),
                "failed": len(group) - len(ok["current"]) - len(ok["proposed"]),
                "median_wall_current": median(list(walls["current"].values())),
                "median_wall_proposed": median(list(walls["proposed"].values())),
                "median_difference": median(differences),
                "median_ratio": median(ratios),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "min_ratio": min(ratios, default=None),
                "max_ratio": max(ratios, default=None),
                "median_maxrss_mib_current": median([t["maxrss"] / 2**20 for t in ok["current"]]),
                "median_maxrss_mib_proposed": median([t["maxrss"] / 2**20 for t in ok["proposed"]]),
            }
        )
    return rows


def get_wall(trial: dict[str, Any]) -> float:
    """The wall seconds of a trial: the median of its repetitions."""
    return statistics.median(trial["wall"])


def get_median_interval(values: list[float]) -> tuple[float | None, float | None]:
    """A confidence interval of at least 95% of the median of sorted values, or Nones.

    The interval runs from the ``k``-th smallest to the ``k``-th largest value, for the largest
    ``k`` such that fewer than ``k`` of ``n`` independent values fall below the median with
    probability at most 2.5%.  It needs no assumption about the distribution of the values, but at
    least six of them.
    """
    num_values = len(values)
    rank, tail = 0, 0.0
    while True:
        tail += math.comb(num_values, rank) / 2**num_values
        if tail > 0.025:
            break
        rank += 1
    if rank == 0:
        return None, None
    return values[rank - 1], values[num_values - rank]


def median(values: list[float]) -> float | None:
    """The median of some values, or None if there are none."""
    return statistics.median(values) if values else None


def print_rows(rows: list[dict[str, Any]]) -> None:
    """Print the rows of the summary as a table."""

    def show(value: float | None, digits: int) -> str:
        return "-" if value is None else f"{value:.{digits}f}"

    for row in rows:
        print(
            f"{row['workload']:36s} n={row['trials_current']}/{row['trials_proposed']}"
            f" fail={row['failed']}"
            f" wall={show(row['median_wall_current'], 3)}/{show(row['median_wall_proposed'], 3)}"
            f" diff={show(row['median_difference'], 3)}"
            f" ratio={show(row['median_ratio'], 3)}"
            f" ci=[{show(row['ci_low'], 3)}, {show(row['ci_high'], 3)}]"
            f" [{show(row['min_ratio'], 3)}, {show(row['max_ratio'], 3)}]"
            f" rss={show(row['median_maxrss_mib_current'], 0)}"
            f"/{show(row['median_maxrss_mib_proposed'], 0)} MiB"
        )


if __name__ == "__main__":
    main()
