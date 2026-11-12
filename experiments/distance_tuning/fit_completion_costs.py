# SPDX-License-Identifier: Apache-2.0

"""Refit the cost model of completion tables from traces of distance searches.

This script follows the retuning procedure of the module docstring of
``qldpc.codes.distance._tuning``.  ``collect`` traces each workload (see ``trace.py``) under four
tunings: tables disabled (``completion_bound_max_table_bytes=0``), tables forced
(``completion_bound_minimum_layer_work=0`` and ``completion_bound_max_cost_percent=MAX_VALUE``), the
same on blocks of 8 columns, whose tables are about 256 times smaller, and the defaults, each in a
new single-threaded process, in an order that alternates between repeats.  It saves each trace in
``traces/`` in the results directory, and skips the traces that are there.  Run it on a quiet
machine, one trace at a time, since lookups of tables are sensitive to the processor's caches.

``fit`` aligns the layers of the four traces of each workload and repeat, and expresses times in
scanned supports: the seconds per support of the layer with tables disabled.  For each layer of at
least ``--min-supports`` supports that a forced trace scanned with tables, it computes:

- the measured cost of the layer with tables, without choosing and building them, as a share of its
  cost with tables disabled, and the cost that ``predict_completion_cost`` predicted for it;
- the ratio of its lookups to the lookups of the cost model with a kept-first-row ratio of one,
  which estimates ``completion_bound_kept_first_row_ratio``;
- its excess cost over the scan of the supports that it kept, per lookup and block, which a fit
  of ``completion_bound_lookup_block_cost`` and ``completion_bound_lookup_block_cost_per_mib``
  models;
- the seconds of building its tables, which a fit of ``completion_bound_table_byte_cost`` and
  ``completion_bound_build_step_cost`` models, as ``_estimate_build_cost`` counts bytes and steps.
  Searches with tables forced build them in their first layers, and reuse them, so every layer
  that builds tables counts, with the seconds per support of the search's layers of at least
  ``--min-supports`` supports with tables disabled.

It prints these layers, with whether the defaults used tables on them, and the fitted coefficients
next to the current ones.  Medians over repeats replace single times.

Examples:

    python experiments/distance_tuning/fit_completion_costs.py collect --repeats 3 \\
        --workloads distance/qt7/X distance/pp232/X
    python experiments/distance_tuning/fit_completion_costs.py fit
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
from typing import TYPE_CHECKING, Any

import numpy as np

from qldpc.codes.distance import _tuning

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.distance_tuning.*, and would silently treat a bare "import sweep" as untyped.
if TYPE_CHECKING:
    from experiments.distance_tuning import sweep
else:
    import sweep

TRACES_NAME = "traces"  # name of the directory of traces in the results directory

# The tunings of the traces of each workload.
ARMS: dict[str, dict[str, Any]] = {
    "disabled": {"completion_bound_max_table_bytes": 0},
    "forced": {
        "completion_bound_minimum_layer_work": 0,
        "completion_bound_max_cost_percent": _tuning.MAX_VALUE,
    },
    "forced8": {
        "completion_bound_minimum_layer_work": 0,
        "completion_bound_max_cost_percent": _tuning.MAX_VALUE,
        "completion_bound_block_bits": 8,
    },
    "default": {},
}


def main() -> None:
    """Collect traces or fit the cost model."""
    args = parse_args()
    directory = args.results_dir / TRACES_NAME
    if args.command == "collect":
        collect(args.workloads, args.repeats, directory, args.timeout)
    else:
        fit(directory, args.min_supports)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=sweep.DIRECTORY / "results",
        help="the directory of traces/ (default: results/ beside this script)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    collect_parser = commands.add_parser("collect", help="trace workloads under the three tunings")
    collect_parser.add_argument("--workloads", nargs="+", required=True, help="workload names")
    collect_parser.add_argument(
        "--repeats", type=int, default=3, help="traces of each tuning (default: %(default)s)"
    )
    collect_parser.add_argument(
        "--timeout", type=float, default=3600, help="seconds per trace (default: %(default)s)"
    )
    fit_parser = commands.add_parser("fit", help="fit the cost model to the traces")
    fit_parser.add_argument(
        "--min-supports",
        type=float,
        default=1e7,
        help="the fewest supports of a fitted layer (default: %(default)g)",
    )
    return parser.parse_args()


####################################################################################################
# collection


def get_trace_path(directory: pathlib.Path, workload: str, arm: str, repeat: int) -> pathlib.Path:
    """The file of one trace."""
    slug = workload.replace("/", "_").replace(":", "-")
    return directory / f"{slug}__{arm}__{repeat}.json"


def collect(workloads: list[str], repeats: int, directory: pathlib.Path, timeout: float) -> None:
    """Trace each workload under each tuning, in new single-threaded processes."""
    directory.mkdir(parents=True, exist_ok=True)
    for repeat in range(repeats):
        for workload in workloads:
            arms = list(ARMS) if repeat % 2 == 0 else list(ARMS)[::-1]
            for arm in arms:
                path = get_trace_path(directory, workload, arm, repeat)
                if path.exists():
                    continue
                command = [
                    sys.executable,
                    str(sweep.DIRECTORY / "trace.py"),
                    workload,
                    "--tuning",
                    json.dumps(ARMS[arm]),
                    "--output",
                    str(path),
                ]
                try:
                    subprocess.run(
                        command,
                        check=True,
                        capture_output=True,
                        timeout=timeout,
                        env=os.environ | sweep.SINGLE_THREAD,
                        cwd=sweep.REPOSITORY,
                    )
                except subprocess.TimeoutExpired:
                    print(f"{workload} {arm} {repeat}: timeout", flush=True)
                    continue
                seconds = json.loads(path.read_text(encoding="utf-8"))["seconds"]
                print(f"{workload} {arm} {repeat}: {seconds:.2f} s", flush=True)


####################################################################################################
# fit


def fit(directory: pathlib.Path, min_supports: float) -> None:
    """Fit the cost model to the traces, and print the layers and the coefficients."""
    traces: dict[tuple[str, int], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        _, arm, repeat = path.stem.split("__")
        traces[record["workload"], int(repeat)][arm] = record

    # The measurements of each aligned layer, and of each build, over repeats.
    layers: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    builds: dict[tuple[str, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for (workload, _), arms in sorted(traces.items()):
        if set(arms) != set(ARMS):
            continue
        sequences = {arm: get_layers(record) for arm, record in arms.items()}
        if (
            len({tuple(layer["weight"] for layer in sequence) for sequence in sequences.values()})
            > 1
        ):
            print(f"WARNING: the layers of {workload} differ between tunings; skipping it")
            continue
        large = [layer for layer in sequences["disabled"] if layer["evaluated"] >= min_supports]
        for arm in ("forced", "forced8"):
            predictions = get_predictions(arms[arm])
            if large:
                seconds_per_support = sum(layer["seconds"] for layer in large) / sum(
                    layer["evaluated"] for layer in large
                )
                for index, layer in enumerate(sequences[arm]):
                    if layer["build_seconds"] > 0:
                        build = measure_build(layer, seconds_per_support)
                        builds[f"{workload} {arm}", index].append(build)
            for index, layer in enumerate(sequences[arm]):
                disabled, default = sequences["disabled"][index], sequences["default"][index]
                if layer["tables"] is None or disabled["evaluated"] < min_supports:
                    continue
                measured = measure(layer, disabled, default, predictions)
                layers[f"{workload} {arm}", index].append(measured)

    rows = [summarize(workload, index, group) for (workload, index), group in layers.items()]
    build_rows = [summarize(workload, index, group) for (workload, index), group in builds.items()]
    print_layers(rows)
    print_fits(rows, build_rows)


def get_layers(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The layers of a trace, in order."""
    return [event for event in record["events"] if event["event"] == "layer"]


def get_predictions(record: dict[str, Any]) -> dict[tuple[int, int], float]:
    """The first predicted cost of each number of rows and weight of a trace."""
    predictions: dict[tuple[int, int], float] = {}
    for event in record["events"]:
        if event["event"] == "predict":
            predictions.setdefault((event["rows"], event["weight"]), event["cost"])
    return predictions


def measure(
    forced: dict[str, Any],
    disabled: dict[str, Any],
    default: dict[str, Any],
    predictions: dict[tuple[int, int], float],
) -> dict[str, Any]:
    """The measurements of one layer, with tables forced, disabled, and at the defaults."""
    rows, weight = forced["dimension"], forced["weight"]
    scanned = disabled["evaluated"]
    seconds_per_support = disabled["seconds"] / scanned

    # A layer of one set builds its tables while choosing them, and a layer of several sets builds
    # them while scanning.
    overhead = forced["choose_seconds"]
    if not isinstance(forced["tables"], int):
        overhead += forced["build_seconds"]
    scan_cost = (forced["seconds"] - overhead) / seconds_per_support

    # The lookups of the cost model with a kept-first-row ratio of one, for the sets scanned.
    num_sets = scanned / math.comb(rows, weight)
    share = forced["skipped"] / max(1, forced["skipped"] + forced["evaluated"])
    kept_lookups = (1 - share) * math.comb(rows, weight - 1) * num_sets
    prefix_lookups = math.comb(rows, weight - 2) * num_sets

    return {
        "weight": weight,
        "rows": rows,
        "scanned": scanned,
        "share": share,
        "predicted": predictions.get((rows, weight), math.nan),
        "measured": scan_cost / scanned,
        "measured_with_overhead": forced["seconds"] / disabled["seconds"],
        "default_uses_tables": default["tables"] is not None,
        "lookup_ratio": (forced["lookups"] - prefix_lookups) / max(1.0, kept_lookups),
        "excess_per_lookup_block": (scan_cost - forced["evaluated"])
        / max(1, forced["lookups"] * forced["blocks"]),
        "table_mib": forced["table_bytes"] / 2**20,
        "lookups_blocks": forced["lookups"] * forced["blocks"],
        "excess": scan_cost - forced["evaluated"],
    }


def measure_build(layer: dict[str, Any], seconds_per_support: float) -> dict[str, Any]:
    """The cost of building the tables of a layer, and their bytes and steps.

    The bytes and steps are those that ``_estimate_build_cost`` counts, for the tables of each set
    of the layer, which all have the size of the largest.
    """
    rows = layer["dimension"]
    block_bits = layer["tables"] if isinstance(layer["tables"], int) else max(layer["tables"])
    num_built = 1 if isinstance(layer["tables"], int) else sum(map(bool, layer["tables"]))
    table_entries = layer["table_bytes"] // max(1, rows - 1)
    steps = table_entries * (block_bits + 1) * block_bits + layer["blocks"] * rows**2 // 2
    return {
        "build_cost": layer["build_seconds"] / seconds_per_support,
        "table_bytes": float(layer["table_bytes"] * num_built),
        "steps": float(steps * num_built),
    }


def summarize(workload: str, index: int, group: list[dict[str, Any]]) -> dict[str, Any]:
    """The medians over repeats of the measurements of one layer."""
    row: dict[str, Any] = {"workload": workload, "index": index, "repeats": len(group)}
    for key, value in group[0].items():
        row[key] = (
            statistics.median(entry[key] for entry in group)
            if isinstance(value, float) and not isinstance(value, bool)
            else value
        )
    return row


def print_layers(rows: list[dict[str, Any]]) -> None:
    """Print the measurements of each layer."""
    print(
        f"{'workload':42s} {'w':>2s} {'rows':>4s} {'scanned':>8s} {'skip':>5s} {'pred':>6s}"
        f" {'meas':>6s} {'+ovh':>6s} {'dflt':>4s} {'kept':>5s} {'xs/lkb':>6s} {'MiB':>5s}"
    )
    for row in rows:
        print(
            f"{row['workload']:42s} {row['weight']:2d} {row['rows']:4d} {row['scanned']:8.1e}"
            f" {row['share']:5.3f} {row['predicted']:6.3f} {row['measured']:6.3f}"
            f" {row['measured_with_overhead']:6.3f} {'yes' if row['default_uses_tables'] else 'no':>4s}"
            f" {row['lookup_ratio']:5.2f} {row['excess_per_lookup_block']:6.2f}"
            f" {row['table_mib']:5.0f}"
        )


def print_fits(rows: list[dict[str, Any]], build_rows: list[dict[str, Any]]) -> None:
    """Fit the coefficients of the cost model, and print them with the current ones."""
    tuning = _tuning.DEFAULT_TUNING
    if not rows:
        print("No layers to fit")
        return

    # Lookups: excess = lookups * blocks * (a + b * MiB), with each layer weighted equally.
    scale = np.array([1 / row["scanned"] for row in rows])
    design = np.array(
        [[row["lookups_blocks"], row["lookups_blocks"] * row["table_mib"]] for row in rows]
    )
    target = np.array([row["excess"] for row in rows])
    (lookup_cost, lookup_cost_per_mib), *_ = np.linalg.lstsq(
        design * scale[:, None], target * scale, rcond=None
    )

    # Builds: cost = byte_cost * bytes + step_cost * steps, with each build weighted equally.
    built = build_rows
    if built:
        scale = np.array([1 / row["build_cost"] for row in built])
        design = np.array([[row["table_bytes"], row["steps"]] for row in built])
        target = np.array([row["build_cost"] for row in built])
        (byte_cost, step_cost), *_ = np.linalg.lstsq(
            design * scale[:, None], target * scale, rcond=None
        )
    else:
        byte_cost = step_cost = math.nan

    ratios = [row["lookup_ratio"] for row in rows]
    predicted_builds = [
        tuning.completion_bound_table_byte_cost * row["table_bytes"]
        + tuning.completion_bound_build_step_cost * row["steps"]
        for row in built
    ]
    if built:
        build_ratios = [
            predicted / row["build_cost"] for predicted, row in zip(predicted_builds, built)
        ]
        print(
            f"\n{len(built)} builds: the current coefficients predict"
            f" {min(build_ratios):.2f} to {max(build_ratios):.2f} times their measured cost"
        )
    fits = {
        "completion_bound_lookup_block_cost": lookup_cost,
        "completion_bound_lookup_block_cost_per_mib": lookup_cost_per_mib,
        "completion_bound_kept_first_row_ratio": statistics.median(ratios),
        "completion_bound_table_byte_cost": byte_cost,
        "completion_bound_build_step_cost": step_cost,
    }
    print(f"\nkept-first-row ratios range from {min(ratios):.2f} to {max(ratios):.2f}")
    for name, value in fits.items():
        print(f"{name:45s} current {getattr(tuning, name):8.3f}  fitted {value:8.3f}")


if __name__ == "__main__":
    main()
