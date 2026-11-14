# SPDX-License-Identifier: Apache-2.0

"""Measure the machine constants of BZ searches, and print the derived values of their tuning.

The derived values of ``qldpc.codes.distance._tuning`` follow from rules and from a few constants of
the machine.  The Retuning section of its module docstring states the rule that turns the measured
cost of a step into the value of its gate, and "Derived values" in the README of this directory
tables the measurements.  The script re-derives ``partition_work_ratio``,
``completion_bound_minimum_layer_work``, ``overlapping_minimum_work``, ``lp_minimum_work``, and
``completion_bound_max_cost_percent``, but not yet ``automorphism_minimum_layer_work``, whose
comment says how it was measured, nor the coefficients of the cost model of completion tables,
which ``fit_completion_costs.py`` fits.  It measures:

- the seconds of one candidate/set evaluation: for each traced search, the seconds of its layers of
  at least ``10**7`` evaluations without completion tables, less the seconds of choosing tables and
  of the automorphism switch within them, divided by their evaluations, and the median over the
  searches.  It converts the other constants into evaluations;
- matroid partitioning, per unit of the sum of ``b**2 * k**2 * n`` over the coverages ``b`` that a
  search tries, which ``partition_work_ratio`` multiplies: the costliest of the traced runs;
- the costliest prediction of the cost of completion tables, which samples supports, over the
  layers of the traced searches;
- building a family of overlapping information sets: the costliest of the families built;
- the linear programming of a search of a large basis, ``distance/qt7/X`` of 181 rows: the seconds
  that its certificates spend in their bounds (planning, the checks of ``Certificate.certifies``,
  and the programs), with automorphisms off so that the search keeps its certificate throughout.

It traces five quick distance searches with ``trace.py``, with linear programs off, so that their
layers time scans, and partitioning forced on one of them.  It traces the setup of one more search,
of a large basis, with partitioning forced and automorphisms off, and times the linear programming
of another search, which ``--linear-programs-of`` runs.  Each measurement runs in a new
single-threaded process, which first warms up as ``trial.py`` does.  The script saves each
measurement in ``constants/`` in the results directory, with the source of qLDPC that it ran on
(see ``sweep.get_source_fingerprint``), and skips those already saved from the current source.  So
it measures every constant again after a commit or a change to ``src/``, and it reports nothing
while a measurement of other source remains.  It prints each constant, in seconds and in
evaluations, and each derived value beside its default.  Run it on a quiet machine.  It takes a few
minutes.

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

# Layers below this many candidate/set evaluations are dominated by fixed costs.
LARGE_LAYER = 10**7

# The traces of quick searches: a name, a workload, changes of the tuning, and whether to stop
# after setup.  Linear programs are off, so that layers time scans: ``trace.py`` repeats the
# planning of a certificate that solves programs.
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

# The search whose linear programming is timed, and its changes of the tuning.
LINEAR_PROGRAMS = (
    "distance/qt7/X",
    {"lp_minimum_work": 0, "automorphism_minimum_layer_work": _tuning.MAX_VALUE},
)


@dataclasses.dataclass(frozen=True)
class Constant:
    """A measured cost, in seconds, and how it was measured."""

    seconds: float
    description: str


def main() -> None:
    """Measure the constants, or time the linear programming of one search."""
    args = parse_args()
    if args.linear_programs_of is not None:
        time_linear_programs(args.linear_programs_of, json.loads(args.tuning), args.output)
        return
    directory = args.results_dir / CONSTANTS_NAME
    measure(directory, args.timeout)
    report(directory)


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
        "--timeout", type=float, default=600, help="seconds per measurement (default: %(default)s)"
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


def measure(directory: pathlib.Path, timeout: float) -> None:
    """Run each measurement that the directory lacks for the current source, in new processes.

    Each runs single-threaded, and replaces any measurement of other source.
    """
    directory.mkdir(parents=True, exist_ok=True)
    source = sweep.get_source_fingerprint()
    commands = {
        name: [
            str(sweep.DIRECTORY / "trace.py"),
            workload,
            "--tuning",
            json.dumps(changes),
            *(["--setup-only"] if setup_only else []),
            "--output",
        ]
        for name, workload, changes, setup_only in TRACES
    }
    workload, changes = LINEAR_PROGRAMS
    commands["linear-programs"] = [
        str(pathlib.Path(__file__).resolve()),
        "--linear-programs-of",
        workload,
        "--tuning",
        json.dumps(changes),
        "--output",
    ]
    for name, command in commands.items():
        path = directory / f"{name}.json"
        if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("source") == source:
            continue
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
    """Print the constants and the derived values, from the measurements in the directory.

    Print nothing else if a measurement is missing, or was made on other source code.
    """
    names = [name for name, *_ in TRACES] + ["linear-programs"]
    paths = {name: directory / f"{name}.json" for name in names}
    if not all(path.exists() for path in paths.values()):
        print("Some measurements are missing; run the script again.")
        return
    traces = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    source = sweep.get_source_fingerprint()
    stale = [name for name, record in traces.items() if record.get("source") != source]
    if stale:
        print(f"Measurements of other source code: {', '.join(stale)}; run the script again.")
        return
    linear_programs = traces.pop("linear-programs")

    evaluation = get_evaluation_seconds(traces)
    constants = {
        "partitioning, per unit": get_partition_seconds(traces),
        "costliest prediction": get_prediction_seconds(traces),
        "overlapping family": get_overlapping_seconds(traces),
        "linear programming of a search": Constant(
            linear_programs["bound_seconds"],
            f"{linear_programs['workload']}: {linear_programs['programs']} programs, with planning",
        ),
    }
    print(f"{'constant':<32} {'seconds':>10} {'evaluations':>12}  measured as")
    print(f"{'evaluation':<32} {evaluation.seconds:>10.3g} {1:>12}  {evaluation.description}")
    for name, constant in constants.items():
        evaluations = constant.seconds / evaluation.seconds
        print(f"{name:<32} {constant.seconds:>10.3g} {evaluations:>12.3g}  {constant.description}")

    # Each gate breaks even with the costliest measurement of its step, rounded as the Retuning
    # section says (see round_to_grid).  The cost limit of completion tables compares two
    # estimates of one unit.
    derived = {
        "partition_work_ratio": constants["partitioning, per unit"],
        "completion_bound_minimum_layer_work": constants["costliest prediction"],
        "overlapping_minimum_work": constants["overlapping family"],
        "lp_minimum_work": constants["linear programming of a search"],
    }
    values = {
        field: round_to_grid(constant.seconds / evaluation.seconds)
        for field, constant in derived.items()
    }
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


if __name__ == "__main__":
    main()
