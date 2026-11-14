# SPDX-License-Identifier: Apache-2.0

"""Measure the machine constants of exact searches, and print the derived values of their tuning.

The derived values of ``qldpc.codes.distance._tuning`` follow from rules and from constants of the
machine.  The Retuning section of its module docstring states the rules, and "Rules and constants"
in the README of this directory tables the measurements.  The script measures every constant that a
derived value needs, except that of ``automorphism_minimum_layer_work``, whose comment says how it
was measured, and prints each derived value beside its default.  It measures:

- the seconds of one candidate/set evaluation of a BZ search: for each traced search, the seconds of
  its layers of at least ``10**7`` evaluations without completion tables, less the seconds of
  choosing tables and of the automorphism switch within them, divided by their evaluations, and the
  median over the searches.  It converts the constants of BZ searches into evaluations;
- matroid partitioning, per unit of the sum of ``b**2 * k**2 * n`` over the coverages ``b`` that a
  search tries, which ``partition_work_ratio`` multiplies: the costliest of the traced runs;
- the costliest prediction of the cost of completion tables, which samples supports, over the
  layers of the traced searches;
- building a family of overlapping information sets: the costliest of the families built;
- the linear programming of a search of a large basis, ``distance/qt7/X`` of 181 rows: the seconds
  that its certificates spend in their bounds (planning, the checks of ``Certificate.certifies``,
  and the programs), with automorphisms off so that the search keeps its certificate throughout;
- the seconds of one state of a connected-support search: for each traced search, the seconds of
  its weight limits of at least ``10**4`` states, less the seconds of asking for automorphisms,
  divided by their states, and the median over the searches.  It converts the cost of asking into
  states;
- asking for automorphisms in a connected-support search, which finds them, verifies them, and
  orders their orbits: the costliest of the traced searches, which ask before their first weight
  limit;
- the five coefficients of the cost model of completion tables, from the costs of the layers of the
  cost data set (see below).

Each gate breaks even with the costliest measurement of its step, in the unit of its field, rounded
by ratio to 1, 2, or 5 times a power of ten.

The cost model of completion tables (Section 3.9 of ``brouwer_zimmermann.md``), which
``predict_completion_cost`` and ``_estimate_build_cost`` in
``_information_sets/completion_bounds.py`` compute, has five coefficients, which are constants of
the machine.  The script measures them on the costs of the layers of the cost data set of the
Retuning section, ``COST_DATA_SET``: the X sectors of the [[288, 12, 18]] bivariate bicycle code and
of the pair-partition codes ``pp232``, ``pp170``, and ``pp276``.  It traces each of these searches
three times under each of four tunings: tables disabled (``completion_bound_max_table_bytes=0``),
tables forced (``completion_bound_minimum_layer_work=0`` and
``completion_bound_max_cost_percent=MAX_VALUE``), the same on blocks of 8 columns, whose tables are
about 128 times smaller, and the defaults, in an order that alternates between repeats.  It aligns
the layers of the four traces of each search and repeat, and expresses times in scanned supports,
the seconds per support of the layer with tables disabled.  For each layer of at least ``10**7``
supports that a forced trace scanned with tables, it computes:

- the measured cost of the layer with tables, without choosing and building them, as a share of its
  cost with tables disabled, and the cost that ``predict_completion_cost`` predicted for it;
- the ratio of its lookups to the lookups of the cost model with a kept-first-row ratio of one,
  which measures ``completion_bound_kept_first_row_ratio``, for a layer of one set;
- its excess cost over the scan of the supports that it kept, per lookup and block, to which it
  fits ``completion_bound_lookup_block_cost`` and ``completion_bound_lookup_block_cost_per_mib`` by
  least squares, for a layer of one set;
- the seconds of building its tables, which ``completion_bound_table_byte_cost`` and
  ``completion_bound_build_step_cost`` predict, as ``_estimate_build_cost`` counts bytes and steps.
  Bytes and steps grow together, so the builds measure one scale of both coefficients, the median
  ratio of measured to predicted builds.  The ratio of the two coefficients, one to two, was set
  by hand.  Searches with tables forced keep the tables of one information set at a time, and a set
  reuses its tables only if no other set has chosen tables since it did, so a layer that scans
  several information sets builds tables for each of them.  So every layer that builds tables
  counts, for each table that it builds, with the seconds per candidate/set evaluation of the
  search's layers of at least ``10**7`` supports with tables disabled.

The fits of lookups and of the kept-first-row ratio use only the layers that scan one set, as the
cost model assumes: such a layer chooses one width of tables, and its skipped supports and lookups
belong to that set.  The kept-first-row ratio is the median over the layers with blocks of
``completion_bound_block_bits`` columns that skip at most 99% of their supports, since the few
lookups of a layer that skips more, beyond those of prefixes, cannot resolve it.  Medians over
repeats replace single times.  The script prints these layers, with whether the defaults used tables
on them and how many sets they scan, and each measured coefficient beside its default.

Each measurement is a trace of ``trace.py``, or, for linear programming, a run of this script with
``--linear-programs-of``, in a new single-threaded process, which first warms up as ``trial.py``
does.  The script saves each measurement in ``constants/`` in the results directory, with the
source of qLDPC that it ran on (see ``sweep.get_source_fingerprint``), and skips those already
saved from the current source.  So it measures every constant again after a commit or a change to
``src/``, and it reports no value while a measurement of it from other source remains.  The quick
measurements take a few minutes, after which the script prints the derived values.  It then traces
the cost data set, which takes hours, and prints the coefficients of the cost model.  An
interrupted run resumes where it stopped.  It measures the qLDPC of its own checkout, and refuses to
run if this process imports qLDPC from elsewhere.  Run it on a quiet machine, under a guard of time
and memory.

Example:

    python experiments/distance_tuning/constants.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from qldpc.codes.distance import _tuning
from qldpc.codes.distance._information_sets import certificates

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.distance_tuning.*, and would silently treat a bare "import sweep" as untyped.
if TYPE_CHECKING:
    from experiments.distance_tuning import sweep, trial, workloads
else:
    import sweep
    import trial
    import workloads

CONSTANTS_NAME = "constants"  # name of the directory of measurements in the results directory
COST_MODEL_NAME = "cost-model"  # name of its directory of the traces of the cost data set

# Layers below this many candidate/set evaluations, or supports, are dominated by fixed costs.
LARGE_LAYER = 10**7

# Weight limits of connected-support searches below this many states are dominated by fixed costs.
LARGE_LIMIT = 10**4

# The traces of quick searches: a name, a workload, changes of the tuning, and whether to stop
# after setup.  Linear programs are off in BZ searches, so that layers time scans: ``trace.py``
# repeats the planning of a certificate that solves programs.  Connected-support searches ask for
# automorphisms before their first weight limit, so that each of their traces measures that step.
TRACES: list[tuple[str, str, dict[str, int], bool]] = [
    (name, workload, {"lp_minimum_work": _tuning.MAX_VALUE}, False)
    for name, workload in (
        ("qt7", "distance/qt7/X"),
        ("lp13s5", "distance/lp13s5/X"),
        ("pp232", "distance/pp232/X"),
        ("classical-150", "distance/classical:150:100:1/-"),
    )
] + [
    (
        "classical-128",
        "distance/classical:128:80:6/-",
        {"lp_minimum_work": _tuning.MAX_VALUE, "partition_work_ratio": 0},
        False,
    ),
    (
        "lp13s5-partition",
        "distance/lp13s5/X",
        {"partition_work_ratio": 0, "automorphism_minimum_layer_work": _tuning.MAX_VALUE},
        True,
    ),
]
TRACES += [
    (
        f"support-{code.replace(':', '-')}",
        f"csdistance/{code}/X",
        {"automorphism_minimum_states": 0},
        False,
    )
    for code in ("qt7", "pp276", "bb288", "gross", "lp13s5", "pp232", "surface:17")
]

# The search whose linear programming is timed, and its changes of the tuning.
LINEAR_PROGRAMS = (
    "distance/qt7/X",
    {"lp_minimum_work": 0, "automorphism_minimum_layer_work": _tuning.MAX_VALUE},
)

# The cost data set of the Retuning section of _tuning, on which the coefficients of the cost model
# of completion tables are measured, and how many times each search is traced under each tuning.
COST_DATA_SET = ("distance/bb288/X", "distance/pp232/X", "distance/pp170/X", "distance/pp276/X")
COST_MODEL_REPEATS = 3

# The tunings of the traces of the cost data set.
COST_MODEL_ARMS: dict[str, dict[str, Any]] = {
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

# The largest share of skipped supports of a layer whose kept-first-row ratio is measured.
MAX_KEPT_RATIO_SHARE = 0.99


@dataclasses.dataclass(frozen=True)
class Constant:
    """A measured cost, in seconds, and how it was measured."""

    seconds: float
    description: str


def main() -> None:
    """Measure the constants, or time the linear programming of one search."""
    args = parse_args()
    sweep.check_local_import()
    if args.linear_programs_of is not None:
        time_linear_programs(args.linear_programs_of, json.loads(args.tuning), args.output)
        return
    directory = args.results_dir / CONSTANTS_NAME
    measure(get_quick_commands(directory), directory, args.timeout)
    report(directory)
    measure(get_cost_model_commands(directory / COST_MODEL_NAME), directory, args.timeout)
    report_cost_model(directory / COST_MODEL_NAME)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        default=sweep.DIRECTORY / "results",
        help="the directory of constants/ (default: results/ beside this script)",
    )
    parser.add_argument(
        "--timeout", type=float, default=3600, help="seconds per measurement (default: %(default)s)"
    )
    parser.add_argument(
        "--linear-programs-of",
        metavar="WORKLOAD",
        help="only time the linear programming of one search, in this process, and save it",
    )
    parser.add_argument(
        "--tuning", default="{}", help="with --linear-programs-of, changes of the tuning"
    )
    parser.add_argument(
        "--output", type=pathlib.Path, help="with --linear-programs-of, a JSON file to save"
    )
    args = parser.parse_args()
    if args.linear_programs_of is not None and args.output is None:
        parser.error("--linear-programs-of needs --output")
    return args


####################################################################################################
# measurements


def get_quick_commands(directory: pathlib.Path) -> dict[pathlib.Path, list[str]]:
    """The file and the arguments, up to that file, of each quick measurement."""
    commands = {
        directory / f"{name}.json": get_trace_command(workload, changes, setup_only)
        for name, workload, changes, setup_only in TRACES
    }
    workload, changes = LINEAR_PROGRAMS
    commands[directory / "linear-programs.json"] = [
        str(pathlib.Path(__file__).resolve()),
        "--linear-programs-of",
        workload,
        "--tuning",
        json.dumps(changes),
        "--output",
    ]
    return commands


def get_cost_model_commands(directory: pathlib.Path) -> dict[pathlib.Path, list[str]]:
    """The file and the arguments of each trace of the cost data set, whose order alternates."""
    commands = {}
    for repeat in range(COST_MODEL_REPEATS):
        for workload in COST_DATA_SET:
            arms = list(COST_MODEL_ARMS) if repeat % 2 == 0 else list(COST_MODEL_ARMS)[::-1]
            for arm in arms:
                path = get_cost_model_path(directory, workload, arm, repeat)
                commands[path] = get_trace_command(workload, COST_MODEL_ARMS[arm], False)
    return commands


def measure(
    commands: dict[pathlib.Path, list[str]], directory: pathlib.Path, timeout: float
) -> None:
    """Run each measurement that lacks a file of the current source, in new processes.

    Each command gives the arguments of a measurement up to its file, which lies under
    ``directory``.  Each runs single-threaded, and replaces any measurement of other source.
    """
    source = sweep.get_source_fingerprint()
    for path, command in commands.items():
        if is_current(path, source):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        name = path.relative_to(directory)
        try:
            process = subprocess.run(
                [sys.executable, *command, str(path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ | sweep.SINGLE_THREAD,
                cwd=sweep.REPOSITORY,
            )
        except subprocess.TimeoutExpired:
            print(f"{name}: timeout", flush=True)
            continue
        if process.returncode:
            print(f"{name}: failed\n{process.stderr}", flush=True)
            continue
        seconds = json.loads(path.read_text(encoding="utf-8"))["seconds"]
        print(f"{name}: {seconds:.2f} s", flush=True)


def get_trace_command(workload: str, changes: dict[str, Any], setup_only: bool) -> list[str]:
    """The arguments of ``trace.py`` that trace a workload, up to the file of the trace."""
    return [
        str(sweep.DIRECTORY / "trace.py"),
        workload,
        "--tuning",
        json.dumps(changes),
        *(["--setup-only"] if setup_only else []),
        "--output",
    ]


def get_cost_model_path(
    directory: pathlib.Path, workload: str, arm: str, repeat: int
) -> pathlib.Path:
    """The file of one trace of the cost data set."""
    slug = workload.replace("/", "_").replace(":", "-")
    return directory / f"{slug}__{arm}__{repeat}.json"


def is_current(path: pathlib.Path, source: dict[str, str]) -> bool:
    """Whether a file holds a measurement of the current source."""
    return path.exists() and json.loads(path.read_text(encoding="utf-8")).get("source") == source


def time_linear_programs(workload: str, changes: dict[str, Any], output: pathlib.Path) -> None:
    """Time the bounds of the certificates of one search that solve linear programs, and save them.

    The bounds of such a certificate (``plan``, ``certifies``, and ``step_lower_bound``) plan its
    steps with exact arithmetic and solve its programs, so their seconds, outside nested calls, are
    the cost of linear programming in the search.
    """
    trial.check_workload(workload)
    source = sweep.get_source_fingerprint()
    record: dict[str, Any] = {"bound_seconds": 0.0, "programs": 0}
    depth = [0]

    def wrap_bound(function: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(certificate: certificates.Certificate, *args: Any) -> Any:
            if not certificate._solves_linear_programs:
                return function(certificate, *args)
            depth[0] += 1
            start = time.perf_counter()
            try:
                return function(certificate, *args)
            finally:
                depth[0] -= 1
                if not depth[0]:
                    record["bound_seconds"] += time.perf_counter() - start

        return wrapper

    def count_program(*args: Any, **kwargs: Any) -> Any:
        record["programs"] += 1
        return get_lp_lower_bound(*args, **kwargs)

    get_lp_lower_bound = certificates.get_lp_lower_bound
    with _tuning.use_tuning(**changes):
        trial.warm_up("distance")
        for name in ("plan", "certifies", "step_lower_bound"):
            setattr(
                certificates.Certificate,
                name,
                wrap_bound(getattr(certificates.Certificate, name)),
            )
        certificates.get_lp_lower_bound = count_program
        start = time.perf_counter()
        result = workloads.run_workload(workload)
        seconds = time.perf_counter() - start
    record |= {"workload": workload, "tuning": changes, "source": source}
    record |= {"result": result, "seconds": seconds}
    output.write_text(json.dumps(record, indent=1), encoding="utf-8")


####################################################################################################
# constants and derived values


def report(directory: pathlib.Path) -> None:
    """Print the constants and the derived values of gates, from the measurements in the directory.

    Print nothing else if a measurement is missing, or was made on other source code.
    """
    names = [name for name, *_ in TRACES] + ["linear-programs"]
    paths = {name: directory / f"{name}.json" for name in names}
    if not all(path.exists() for path in paths.values()):
        print("Some measurements of gates are missing; run the script again.")
        return
    traces = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    source = sweep.get_source_fingerprint()
    stale = [name for name, record in traces.items() if record.get("source") != source]
    if stale:
        print(f"Measurements of other source code: {', '.join(stale)}; run the script again.")
        return
    linear_programs = traces.pop("linear-programs")

    # Each constant, with the constant that converts it into the unit of its fields.
    evaluation = get_evaluation_seconds(traces)
    state = get_state_seconds(traces)
    units = {"evaluation": evaluation, "state": state}
    constants = {
        "partitioning, per unit": (get_partition_seconds(traces), "evaluation"),
        "costliest prediction": (get_prediction_seconds(traces), "evaluation"),
        "overlapping family": (get_overlapping_seconds(traces), "evaluation"),
        "linear programming of a search": (
            Constant(
                linear_programs["bound_seconds"],
                f"{linear_programs['workload']}: {linear_programs['programs']} programs, with"
                " planning",
            ),
            "evaluation",
        ),
        "asking for automorphisms": (get_ask_seconds(traces), "state"),
    }
    counts = {
        name: constant.seconds / units[unit_name].seconds
        for name, (constant, unit_name) in constants.items()
    }
    print(f"{'constant':<32} {'seconds':>10} {'in units':>10}  measured as")
    for name, unit in units.items():
        print(f"{name:<32} {unit.seconds:>10.3g} {1:>10} {name}s  {unit.description}")
    for name, (constant, unit_name) in constants.items():
        print(
            f"{name:<32} {constant.seconds:>10.3g} {counts[name]:>10.3g} {unit_name}s"
            f"  {constant.description}"
        )

    # Each gate breaks even with the costliest measurement of its step, rounded as the Retuning
    # section says (see round_to_grid).  The cost limit of completion tables compares two
    # estimates of one unit.
    gates = {
        "partition_work_ratio": "partitioning, per unit",
        "completion_bound_minimum_layer_work": "costliest prediction",
        "overlapping_minimum_work": "overlapping family",
        "lp_minimum_work": "linear programming of a search",
        "automorphism_minimum_states": "asking for automorphisms",
    }
    values = {field: round_to_grid(counts[name]) for field, name in gates.items()}
    values["completion_bound_max_cost_percent"] = 100
    print(f"\n{'field':<38} {'derived':>8} {'default':>8}")
    for field, value in values.items():
        default = getattr(_tuning.DEFAULT_TUNING, field)
        print(f"{field:<38} {value:>8.3g} {default:>8.3g}")


def round_to_grid(value: float) -> int:
    """Round a positive value to the nearest of 1, 2, or 5 times a power of ten, by ratio."""
    exponent = math.floor(math.log10(value))
    candidates = [
        mantissa * 10**power for power in (exponent, exponent + 1) for mantissa in (1, 2, 5)
    ]
    return min(candidates, key=lambda candidate: abs(math.log(candidate / value)))


def get_evaluation_seconds(traces: dict[str, dict[str, Any]]) -> Constant:
    """The median, over the traced searches, of their seconds per candidate/set evaluation.

    Each search counts its layers of at least ``LARGE_LAYER`` evaluations without completion tables,
    less the seconds of choosing tables and, in a layer that switches to automorphism images, of
    switching.
    """
    rates = {}
    for trace in traces.values():
        seconds = evaluations = switch_seconds = 0.0
        for event in trace["events"]:
            if event["event"] == "automorphisms":
                switch_seconds += event["seconds"]
            if event["event"] != "layer":
                continue
            if event["tables"] is None and event["evaluated"] >= LARGE_LAYER:
                seconds += event["seconds"] - event["choose_seconds"]
                seconds -= switch_seconds if event["switch_work"] is not None else 0.0
                evaluations += event["evaluated"]
            switch_seconds = 0.0
        if evaluations:
            rates[trace["workload"]] = seconds / evaluations
    description = "median of " + ", ".join(
        f"{workload}: {rate * 1e9:.2f} ns" for workload, rate in rates.items()
    )
    return Constant(statistics.median(rates.values()), description)


def get_partition_seconds(traces: dict[str, dict[str, Any]]) -> Constant:
    """The costliest seconds of matroid partitioning, per unit of ``partition_work_ratio``."""
    runs = [
        (trace["workload"], event)
        for trace in traces.values()
        for event in trace["events"]
        if event["event"] == "family" and event["kind"] == "partitioned" and event["partitioned"]
    ]
    description = "costliest of " + ", ".join(
        f"{workload} (coverages up to {event['max_coverage']}): "
        f"{event['seconds'] / event['unit'] * 1e9:.2f} ns"
        for workload, event in runs
    )
    return Constant(max(event["seconds"] / event["unit"] for _, event in runs), description)


def get_prediction_seconds(traces: dict[str, dict[str, Any]]) -> Constant:
    """The costliest seconds of a prediction that samples, over the layers of the traces."""
    seconds, workload, rows, weight = max(
        (event["seconds"], trace["workload"], event["rows"], event["weight"])
        for trace in traces.values()
        for event in trace["events"]
        if event["event"] == "predict" and event["block_bits"]
    )
    return Constant(seconds, f"{workload}: the layer of weight {weight}, of {rows} rows")


def get_overlapping_seconds(traces: dict[str, dict[str, Any]]) -> Constant:
    """The costliest seconds of building a family of overlapping information sets."""
    builds = [
        (trace["workload"], event["rows"], event["seconds"])
        for trace in traces.values()
        for event in trace["events"]
        if event["event"] == "family"
        and event["kind"] == "overlapping"
        and event.get("stage") == "built"
    ]
    description = "costliest of " + ", ".join(
        f"{workload} ({rows} rows): {seconds * 1e3:.2f} ms" for workload, rows, seconds in builds
    )
    return Constant(max(seconds for *_, seconds in builds), description)


def get_state_seconds(traces: dict[str, dict[str, Any]]) -> Constant:
    """The median, over traced connected-support searches, of their seconds per state.

    Each search counts its weight limits of at least ``LARGE_LIMIT`` states, less the seconds of
    asking for automorphisms before them.
    """
    rates = {}
    for trace in traces.values():
        seconds = states = 0.0
        for event in trace["events"]:
            if event["event"] == "weight limit" and event["states"] >= LARGE_LIMIT:
                seconds += event["seconds"] - get_seconds_of_ask(event)
                states += event["states"]
        if states:
            rates[trace["workload"]] = seconds / states
    description = "median of " + ", ".join(
        f"{workload}: {rate * 1e9:.0f} ns" for workload, rate in rates.items()
    )
    return Constant(statistics.median(rates.values()), description)


def get_ask_seconds(traces: dict[str, dict[str, Any]]) -> Constant:
    """The costliest seconds of asking for automorphisms in a connected-support search."""
    asks = [
        (trace["workload"], get_seconds_of_ask(event))
        for trace in traces.values()
        for event in trace["events"]
        if event["event"] == "weight limit" and event["asked"]
    ]
    description = "costliest of " + ", ".join(
        f"{workload}: {seconds * 1e3:.2f} ms" for workload, seconds in asks
    )
    return Constant(max(seconds for _, seconds in asks), description)


def get_seconds_of_ask(event: dict[str, Any]) -> float:
    """The seconds of asking for automorphisms before a weight limit, or zero if it did not ask."""
    return event.get("find_seconds", 0.0) + event.get("verify_seconds", 0.0)


####################################################################################################
# cost model of completion tables


def report_cost_model(directory: pathlib.Path) -> None:
    """Print the layers of the cost data set, and the coefficients that they measure.

    Print nothing else if a trace is missing, or was made on other source code.
    """
    source = sweep.get_source_fingerprint()
    traces: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    num_missing = 0
    for repeat in range(COST_MODEL_REPEATS):
        for workload in COST_DATA_SET:
            for arm in COST_MODEL_ARMS:
                path = get_cost_model_path(directory, workload, arm, repeat)
                if not is_current(path, source):
                    num_missing += 1
                    continue
                record = json.loads(path.read_text(encoding="utf-8"))
                traces.setdefault((workload, repeat), {})[arm] = record
    print()
    if num_missing:
        print(
            f"{num_missing} traces of the cost data set are missing or of other source code; run"
            " the script again."
        )
        return

    # The measurements of each aligned layer, and of each build, over repeats.
    layers: dict[tuple[str, int], list[dict[str, Any]]] = {}
    builds: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for (workload, _), arms in sorted(traces.items()):
        sequences = {arm: get_layers(record) for arm, record in arms.items()}
        if (
            len({tuple(layer["weight"] for layer in sequence) for sequence in sequences.values()})
            > 1
        ):
            print(f"WARNING: the layers of {workload} differ between tunings; skipping it")
            continue
        large = [layer for layer in sequences["disabled"] if layer["evaluated"] >= LARGE_LAYER]
        for arm in ("forced", "forced8"):
            predictions = get_predictions(arms[arm])
            if large:
                # The unit of builds is the candidate/set evaluation, a support that a scan
                # evaluates in its set.
                seconds_per_evaluation = sum(layer["seconds"] for layer in large) / sum(
                    layer["evaluated"] for layer in large
                )
                for index, layer in enumerate(sequences[arm]):
                    if layer["build_seconds"] > 0:
                        build = measure_build(layer, seconds_per_evaluation)
                        builds.setdefault((f"{workload} {arm}", index), []).append(build)
            for index, layer in enumerate(sequences[arm]):
                disabled, default = sequences["disabled"][index], sequences["default"][index]
                if layer["tables"] is None or disabled["evaluated"] < LARGE_LAYER:
                    continue
                measured = measure_layer(layer, disabled, default, predictions)
                layers.setdefault((f"{workload} {arm}", index), []).append(measured)

    rows = [summarize(workload, index, group) for (workload, index), group in layers.items()]
    build_rows = [summarize(workload, index, group) for (workload, index), group in builds.items()]
    print_layers(rows)
    print_coefficients(rows, build_rows)


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


def measure_layer(
    forced: dict[str, Any],
    disabled: dict[str, Any],
    default: dict[str, Any],
    predictions: dict[tuple[int, int], float],
) -> dict[str, Any]:
    """The measurements of one layer, with tables forced, disabled, and at the defaults."""
    rows, weight = forced["dimension"], forced["weight"]
    scanned = disabled["evaluated"]
    seconds_per_support = disabled["seconds"] / scanned

    # A layer builds its tables while choosing them.
    scan_cost = (forced["seconds"] - forced["choose_seconds"]) / seconds_per_support

    # The lookups of the cost model with a kept-first-row ratio of one, for a layer of one set, the
    # only kind whose ratio the script prints or measures, since its lookups and skipped supports
    # all belong to that set.
    num_sets = forced["scanned_sets"]
    share = forced["skipped"] / max(1, forced["skipped"] + forced["evaluated"])
    kept_lookups = (1 - share) * math.comb(rows, weight - 1)
    prefix_lookups = math.comb(rows, weight - 2)

    return {
        "weight": weight,
        "rows": rows,
        "scanned": scanned,
        "share": share,
        "predicted": predictions.get((rows, weight), math.nan),
        "measured": scan_cost / scanned,
        "measured_with_overhead": forced["seconds"] / disabled["seconds"],
        "default_uses_tables": default["tables"] is not None,
        "sets": num_sets,
        "block_bits": forced["tables"] if num_sets == 1 else 0,
        "lookup_ratio": (forced["lookups"] - prefix_lookups) / max(1.0, kept_lookups),
        "excess_per_lookup_block": (scan_cost - forced["evaluated"])
        / max(1, forced["lookups"] * forced["blocks"]),
        "table_mib": forced["table_bytes"] / 2**20,
    }


def measure_build(layer: dict[str, Any], seconds_per_evaluation: float) -> dict[str, Any]:
    """The cost of building the tables of a layer, and their bytes and steps.

    The bytes and steps are those that ``_estimate_build_cost`` counts, for each of the tables that
    the layer built, which all have the size of the largest.
    """
    rows = layer["dimension"]
    block_bits = layer["tables"] if isinstance(layer["tables"], int) else max(layer["tables"])
    num_built = layer["builds"]
    table_entries = layer["table_bytes"] // max(1, rows - 1)
    steps = table_entries * (block_bits + 1) * block_bits + layer["blocks"] * rows**2 // 2
    return {
        "build_cost": layer["build_seconds"] / seconds_per_evaluation,
        "table_bytes": float(layer["table_bytes"] * num_built),
        "steps": float(steps * num_built),
    }


def summarize(workload: str, index: int, group: list[dict[str, Any]]) -> dict[str, Any]:
    """The medians over repeats of the measurements of one layer."""
    row: dict[str, Any] = {"workload": workload, "index": index, "repeats": len(group)}
    for key, value in group[0].items():
        row[key] = (
            statistics.median(entry[key] for entry in group) if isinstance(value, float) else value
        )
    return row


def print_layers(rows: list[dict[str, Any]]) -> None:
    """Print the measurements of each layer."""
    print(
        f"{'workload':42s} {'w':>2s} {'rows':>4s} {'sets':>4s} {'scanned':>8s} {'skip':>5s}"
        f" {'pred':>6s} {'meas':>6s} {'+ovh':>6s} {'dflt':>4s} {'kept':>5s} {'xs/lkb':>6s}"
        f" {'MiB':>5s}"
    )
    for row in rows:
        # Some sets of a layer of several sets may look up no bounds, such as a set of deficient
        # rank, and the trace does not say which lookups and skipped supports belong to which set,
        # so the lookups of such a layer have no ratio to the cost model's.
        if row["sets"] == 1:
            lookups = f" {row['lookup_ratio']:5.2f} {row['excess_per_lookup_block']:6.2f}"
        else:
            lookups = f" {'-':>5s} {'-':>6s}"
        print(
            f"{row['workload']:42s} {row['weight']:2d} {row['rows']:4d} {row['sets']:4d}"
            f" {row['scanned']:8.1e}"
            f" {row['share']:5.3f} {row['predicted']:6.3f} {row['measured']:6.3f}"
            f" {row['measured_with_overhead']:6.3f}"
            f" {'yes' if row['default_uses_tables'] else 'no':>4s}"
            f"{lookups}"
            f" {row['table_mib']:5.0f}"
        )


def print_coefficients(rows: list[dict[str, Any]], build_rows: list[dict[str, Any]]) -> None:
    """Measure the coefficients of the cost model, and print them with their defaults."""
    tuning = _tuning.DEFAULT_TUNING
    rows = [row for row in rows if row["sets"] == 1]
    if not rows:
        print("No layers of one set to measure")
        return

    # Lookups: the excess cost per lookup and block is a + b * MiB, with each layer of one set
    # weighted equally.
    design = np.array([[1.0, row["table_mib"]] for row in rows])
    target = np.array([row["excess_per_lookup_block"] for row in rows])
    (lookup_cost, lookup_cost_per_mib), *_ = np.linalg.lstsq(design, target, rcond=None)

    # Builds: bytes and steps grow together, so scale both default coefficients by the median ratio
    # of the measured costs of builds to their predicted costs.
    build_ratios = [
        row["build_cost"]
        / (
            tuning.completion_bound_table_byte_cost * row["table_bytes"]
            + tuning.completion_bound_build_step_cost * row["steps"]
        )
        for row in build_rows
    ]
    build_scale = statistics.median(build_ratios) if build_ratios else math.nan

    ratios_by_width: dict[int, list[float]] = {}
    for row in rows:
        if row["share"] <= MAX_KEPT_RATIO_SHARE:
            ratios_by_width.setdefault(row["block_bits"], []).append(row["lookup_ratio"])
    for block_bits in sorted({row["block_bits"] for row in rows}):
        selected = [
            row["excess_per_lookup_block"] for row in rows if row["block_bits"] == block_bits
        ]
        mib = [row["table_mib"] for row in rows if row["block_bits"] == block_bits]
        print(
            f"blocks of {block_bits} columns: {len(selected)} layers of {min(mib):.0f} to"
            f" {max(mib):.0f} MiB, excess per lookup and block {min(selected):.2f} to"
            f" {max(selected):.2f}, median {statistics.median(selected):.2f}"
        )
    if build_ratios:
        print(
            f"\n{len(build_ratios)} builds: the default coefficients predict"
            f" {1 / max(build_ratios):.2f} to {1 / min(build_ratios):.2f} times their measured"
            f" cost, median {1 / build_scale:.2f}"
        )
    print()
    for block_bits, ratios in sorted(ratios_by_width.items()):
        print(
            f"kept-first-row ratios of blocks of {block_bits} columns: {len(ratios)} layers,"
            f" {min(ratios):.2f} to {max(ratios):.2f}, median {statistics.median(ratios):.2f}"
        )
    kept_ratios = ratios_by_width.get(tuning.completion_bound_block_bits, [])
    measured = {
        "completion_bound_lookup_block_cost": lookup_cost,
        "completion_bound_lookup_block_cost_per_mib": lookup_cost_per_mib,
        "completion_bound_kept_first_row_ratio": (
            statistics.median(kept_ratios) if kept_ratios else math.nan
        ),
        "completion_bound_table_byte_cost": build_scale * tuning.completion_bound_table_byte_cost,
        "completion_bound_build_step_cost": build_scale * tuning.completion_bound_build_step_cost,
    }
    for name, value in measured.items():
        print(f"{name:45s} default {getattr(tuning, name):8.3f}  measured {value:8.3f}")


if __name__ == "__main__":
    main()
