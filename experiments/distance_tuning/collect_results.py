# SPDX-License-Identifier: Apache-2.0

"""Summarize the trials of sweeps, comparing each arm with the first arm of its sweep.

For each sweep, workload, and arm, the summary gives the number of successful trials, the median
wall seconds (of a trial, the median of its repetitions), and the median, minimum, and maximum, over
rounds, of the ratio of the arm's wall time to that of the first arm, the reference, in the same
round and of the same source, with a distribution-free 95% confidence interval of the median ratio
(see ``get_median_interval``).  A sweep resumed after a change of the source reruns its trials, so
a round can hold trials of several sources; the summary warns of such sweeps, and pairs only trials
of one source.  Ratios of one round share the conditions of the machine, so their spread,
across rounds, measures noise better than the spread of times.  The summary also gives the median
peak resident memory in MiB after the timed search, and the trials that failed or timed out.  It
flags a workload whose arms disagree on a result, which no tuning may change except a distance
search with an early exit.

It prints the summary, and writes it to ``summary.csv`` in the results directory.

Example:

    python experiments/distance_tuning/collect_results.py --tags noise max-sets
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import pathlib
import statistics
from typing import TYPE_CHECKING, Any

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.distance_tuning.*, and would silently treat a bare "import sweep" as untyped.
if TYPE_CHECKING:
    from experiments.distance_tuning import sweep
else:
    import sweep

OUTPUT_NAME = "summary.csv"  # name of the summary file in the results directory

# columns of the summary file
CSV_FIELDS = (
    "tag",
    "workload",
    "arm_index",
    "arm",
    "trials",
    "failed",
    "median_wall",
    "median_ratio",
    "ci_low",
    "ci_high",
    "min_ratio",
    "max_ratio",
    "median_maxrss_mib",
)


def main() -> None:
    """Summarize the trials."""
    args = parse_args()
    trials = list(sweep.iter_trials(args.results_dir / sweep.TRIALS_NAME))
    if args.tags:
        trials = [trial for trial in trials if trial["tag"] in args.tags]
    rows = summarize(trials)
    with (args.results_dir / OUTPUT_NAME).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print_rows(rows)
    sources: dict[str, set[str]] = collections.defaultdict(set)
    for trial in trials:
        sources[trial["tag"]].add(get_source(trial))
    for tag, tag_sources in sorted(sources.items()):
        if len(tag_sources) > 1:
            print(
                f"WARNING: {tag} holds trials of {len(tag_sources)} sources: {sorted(tag_sources)}"
            )
    for (tag, workload), results in sorted(get_results(trials).items()):
        if len(results) > 1:
            print(f"WARNING: the arms of {tag} disagree on {workload}: {sorted(results)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tags", nargs="+", help="the sweeps to summarize (default: all)")
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=sweep.DIRECTORY / "results",
        help="the directory of trials.jsonl (default: results/ beside sweep.py)",
    )
    return parser.parse_args()


def summarize(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row for each sweep, workload, and arm."""
    by_arm: dict[tuple[str, str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for trial in trials:
        by_arm[trial["tag"], trial["workload"], trial["arm_index"]].append(trial)

    # the wall time of each round of the reference arm, by sweep, workload, and source
    reference = {
        (tag, workload, trial["round"], get_source(trial)): get_wall(trial)
        for (tag, workload, arm_index), group in by_arm.items()
        if arm_index == 0
        for trial in group
        if trial["status"] == "ok"
    }

    rows = []
    for (tag, workload, arm_index), group in sorted(by_arm.items()):
        ok = [trial for trial in group if trial["status"] == "ok"]
        ratios = sorted(
            get_wall(trial) / reference[tag, workload, trial["round"], get_source(trial)]
            for trial in ok
            if reference.get((tag, workload, trial["round"], get_source(trial)))
        )
        ci_low, ci_high = get_median_interval(ratios)
        rows.append(
            {
                "tag": tag,
                "workload": workload,
                "arm_index": arm_index,
                "arm": group[0]["arm"],
                "trials": len(ok),
                "failed": len(group) - len(ok),
                "median_wall": median([get_wall(trial) for trial in ok]),
                "median_ratio": median(ratios),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "min_ratio": min(ratios, default=None),
                "max_ratio": max(ratios, default=None),
                "median_maxrss_mib": median([trial["maxrss"] / 2**20 for trial in ok]),
            }
        )
    return rows


def get_source(trial: dict[str, Any]) -> str:
    """The commit of a trial, with the hash of its differences from the commit."""
    return f"{trial['source']['commit'][:8]}+{trial['source']['diff'][:8]}"


def get_wall(trial: dict[str, Any]) -> float:
    """The wall seconds of a trial: the median of its repetitions."""
    return statistics.median(trial["wall"])


def get_results(trials: list[dict[str, Any]]) -> dict[tuple[str, str], set[str]]:
    """The distinct results of the successful trials of each sweep and workload."""
    results: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for trial in trials:
        if trial["status"] == "ok":
            results[trial["tag"], trial["workload"]].update(trial["results"])
    return results


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
            f"{row['tag']:12s} {row['workload']:36s} {row['arm_index']:2d} {row['arm']!s:40s}"
            f" n={row['trials']:<3d} fail={row['failed']:<2d}"
            f" wall={show(row['median_wall'], 3):>9s}"
            f" ratio={show(row['median_ratio'], 3)}"
            f" ci=[{show(row['ci_low'], 3)}, {show(row['ci_high'], 3)}]"
            f" [{show(row['min_ratio'], 3)}, {show(row['max_ratio'], 3)}]"
            f" rss={show(row['median_maxrss_mib'], 0)} MiB"
        )


if __name__ == "__main__":
    main()
