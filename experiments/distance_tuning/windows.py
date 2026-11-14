# SPDX-License-Identifier: Apache-2.0

"""Find the interval of each threshold over which the decisions of traced searches stay the same.

The script reads traces of ``trace.py``, all under one tuning, and for each threshold of
``SearchTuning`` that the traces record a decision of, prints the decisions on either side of the
threshold that lie nearest to it: the values that the threshold would have to cross to change a
decision.  So every value between them makes the same decisions, as far as the traced searches go,
since a changed decision can change what the searches decide later.  It prints these windows for
the workloads named by ``--long`` and for all traces.  The thresholds and the quantities that they
are compared with are:

- ``completion_bound_minimum_layer_work``: the supports ``C(k, w)`` of a layer of a distance search,
  or of a step of a count, which considers completion tables if they reach it; the script also
  prints the smallest layer or step that used tables, below which raising the threshold changes no
  layer or step that uses them;
- ``completion_bound_max_cost_percent``: each predicted cost of completion tables, in percent, which
  a layer accepts if it is at most the threshold;
- the product of ``partition_work_ratio`` and ``partition_cost_factor``: the estimated work of a
  search divided by ``b * k**2 * n``.  A search tries matroid partitioning if this quotient reaches
  the product.  If it has automorphisms, it first asks for them, may skip partitioning, and
  considers the switch at the end of setup (see ``trace.py``);
- ``overlapping_minimum_work``: the estimated work of a family that an overlapping family may
  replace, if it reaches the threshold, after the checks that come before it;
- ``lp_minimum_work``: the estimated work of a certificate of overlapping sets, which may solve
  linear programs if it reaches the threshold;
- ``automorphism_minimum_layer_work``: the work of each layer before the switch to automorphism
  images, which the switch decides at the first layer that reaches the threshold; a search whose
  setup found automorphisms decides the switch at the end of setup, and has no such layer;
- ``enumeration_max_dimension`` and ``enumeration_setup_cost_factor``: the dimension ``k`` and the
  ratio ``2**k / (k * n)`` of each test of enumeration by size.

With ``--compare``, it instead compares the decisions of two traces of one workload under two
tunings: enumeration tests, families and their gates, certificates, switches, the tables of each
layer, and each comparison of a predicted cost with the limit.  Either way, it warns of traces that
do not record their source (see ``sweep.get_source_fingerprint``), or whose recorded sources differ.

Example:

    python experiments/distance_tuning/trace.py distance/bb288/X --output traces/bb288.json
    python experiments/distance_tuning/trace.py distance/qt7/X --output traces/qt7.json
    python experiments/distance_tuning/windows.py traces/*.json --long distance/bb288/X
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import pathlib
from typing import Any, NamedTuple

from qldpc.codes.distance import _tuning


class Gate(NamedTuple):
    """One decision of a threshold: the quantity compared with it, and where."""

    value: float
    workload: str
    detail: str


# Thresholds that decide when the quantity is at most the threshold, rather than at least.
AT_MOST = {
    "completion_bound_max_cost_percent",
    "enumeration_max_dimension",
    "enumeration_setup_cost_factor",
}


def main() -> None:
    """Print the windows of the thresholds, or compare two traces."""
    args = parse_args()
    traces = [json.loads(path.read_text(encoding="utf-8")) for path in args.traces]
    warn_of_sources(traces)
    if args.compare:
        if len(traces) != 2:
            raise ValueError("--compare takes two traces")
        compare(*traces)
        return
    tuning = get_tuning(traces)
    thresholds = get_thresholds(tuning)
    gates: dict[str, list[Gate]] = {field: [] for field in thresholds}
    for trace in traces:
        for field, gate in get_gates(trace, tuning):
            gates[field].append(gate)
    for field, threshold in thresholds.items():
        found = gates[field]
        num_workloads = len({g.workload for g in found})
        print(f"{field} = {threshold:g}: {len(found)} decisions in {num_workloads} workloads")
        for label, subset in (
            ("long", [g for g in found if g.workload in args.long]),
            ("all", found),
        ):
            if not subset:
                print(f"    {label}: no decisions")
                continue
            below, above = get_edges(subset, threshold, field in AT_MOST)
            print(f"    {label}: {format_gate(below)} | {format_gate(above)}")
    for label, names in (("long", set(args.long)), ("all", None)):
        used = [
            g
            for trace in traces
            for g in get_used_layers(trace, tuning)
            if names is None or g.workload in names
        ]
        if used:
            print(f"smallest layer with tables ({label}): {format_gate(min(used))}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("traces", nargs="+", type=pathlib.Path, help="JSON files of trace.py")
    parser.add_argument("--long", nargs="*", default=[], help="workloads of long runs")
    parser.add_argument(
        "--compare", action="store_true", help="compare the decisions of two traces"
    )
    return parser.parse_args()


def warn_of_sources(traces: list[dict[str, Any]]) -> None:
    """Warn of traces that do not record their source, or whose recorded sources differ."""
    if any(trace.get("source") is None for trace in traces):
        print("WARNING: some traces do not record their source")
    sources = {
        json.dumps(trace["source"], sort_keys=True)
        for trace in traces
        if trace.get("source") is not None
    }
    if len(sources) > 1:
        print(f"WARNING: the traces record {len(sources)} sources")


def get_tuning(traces: list[dict[str, Any]]) -> _tuning.SearchTuning:
    """The tuning that every trace ran under.

    ``trace.py`` records the whole tuning.  For a trace without it, the tuning is the defaults of
    this commit with the recorded changes, which the script warns of, since the trace may have run
    under other defaults.
    """
    if all("full_tuning" in trace for trace in traces):
        tunings = {json.dumps(trace["full_tuning"], sort_keys=True) for trace in traces}
        if len(tunings) != 1:
            raise ValueError("The traces must share one tuning")
        return dataclasses.replace(_tuning.SearchTuning(), **traces[0]["full_tuning"])
    changes = {json.dumps(trace["tuning"], sort_keys=True) for trace in traces}
    if len(changes) != 1:
        raise ValueError("The traces must share one tuning")
    print("WARNING: some traces lack their whole tuning; assuming the defaults of this commit")
    return dataclasses.replace(_tuning.SearchTuning(), **traces[0]["tuning"])


def get_thresholds(tuning: _tuning.SearchTuning) -> dict[str, float]:
    """The thresholds of a tuning, with the partition fields as their product."""
    return {
        "completion_bound_minimum_layer_work": tuning.completion_bound_minimum_layer_work,
        "completion_bound_max_cost_percent": tuning.completion_bound_max_cost_percent,
        "partition_work_ratio * partition_cost_factor": tuning.partition_work_ratio
        * tuning.partition_cost_factor,
        "overlapping_minimum_work": tuning.overlapping_minimum_work,
        "lp_minimum_work": tuning.lp_minimum_work,
        "automorphism_minimum_layer_work": tuning.automorphism_minimum_layer_work,
        "enumeration_max_dimension": tuning.enumeration_max_dimension,
        "enumeration_setup_cost_factor": tuning.enumeration_setup_cost_factor,
    }


def get_gates(trace: dict[str, Any], tuning: _tuning.SearchTuning) -> list[tuple[str, Gate]]:
    """The decisions of the thresholds in a trace, with the quantities compared with them."""
    workload = trace["workload"]
    gates = []
    for event in trace["events"]:
        kind = event["event"]
        if kind == "considers":
            detail = f"k={event['rows']} w={event['weight']}"
            gates.append(
                ("completion_bound_minimum_layer_work", Gate(event["layer_work"], workload, detail))
            )
        elif kind == "limit" and not math.isinf(event["cost"]):
            detail = f"w={event['weight']}"
            gates.append(
                ("completion_bound_max_cost_percent", Gate(100 * event["cost"], workload, detail))
            )
        elif kind == "family" and event["kind"] == "partitioned" and "estimated_work" in event:
            # Older traces record the cost of partitioning, but not its unit.
            factor = tuning.partition_cost_factor
            unit = event.get("unit", event["estimated_cost"] / factor if factor else 0)
            if not unit:
                continue
            detail = f"k={event['rows']} n={event['columns']}"
            field = "partition_work_ratio * partition_cost_factor"
            gates.append((field, Gate(event["estimated_work"] / unit, workload, detail)))
        elif (
            kind == "family"
            and event["kind"] == "overlapping"
            and event.get("estimated_work") is not None
        ):
            detail = f"k={event['rows']} {event['stage']}"
            gates.append(
                ("overlapping_minimum_work", Gate(event["estimated_work"], workload, detail))
            )
        elif kind == "certificate" and event.get("overlapping"):
            gates.append(("lp_minimum_work", Gate(event["work"], workload, "")))
        elif kind == "enumeration" and event["test"] == "size":
            dimension, ratio = event["dimension"], event["words_per_setup_unit"]
            detail = f"k={dimension} n={event['length']} ratio={ratio:.4g}"
            if dimension >= 63 or ratio > tuning.enumeration_setup_cost_factor:
                gates.append(("enumeration_max_dimension", Gate(dimension, workload, detail)))
            if tuning.enumeration_max_dimension < dimension < 63:
                gates.append(("enumeration_setup_cost_factor", Gate(ratio, workload, detail)))
        elif kind == "layer" and event.get("can_switch"):
            # trace.py numbers the layer of a switch as layer 2 of the new family.
            switched = event["automorphism_family"]
            detail = "at the switch" if switched else f"before layer {event['weight']}"
            gates.append(
                ("automorphism_minimum_layer_work", Gate(event["switch_work"], workload, detail))
            )
    return gates


def get_used_layers(trace: dict[str, Any], tuning: _tuning.SearchTuning) -> list[Gate]:
    """The supports of each layer of a distance search that used tables, or each step of a count.

    A layer of a distance search records the tables that it used.  A count records no layers, and
    a step of a count uses tables if their predicted cost is within the limit.
    """
    workload = trace["workload"]
    layers = [event for event in trace["events"] if event["event"] == "layer"]
    if layers:
        return [
            Gate(event["supports"], workload, f"k={event['dimension']} w={event['weight']}")
            for event in layers
            if any(event["tables"] if isinstance(event["tables"], list) else [event["tables"]])
        ]
    limit = tuning.completion_bound_max_cost_percent
    return [
        Gate(event["supports"], workload, f"k={event['rows']} w={event['weight']}")
        for event in trace["events"]
        if event["event"] == "predict" and 100 * event["cost"] <= limit
    ]


def get_edges(
    gates: list[Gate], threshold: float, at_most: bool
) -> tuple[Gate | None, Gate | None]:
    """The decisions nearest to a threshold on each side of it.

    A quantity at most the threshold decides one way if ``at_most``; otherwise, a quantity at
    least the threshold does.  The first result is the largest quantity on the lower side.
    """
    lower = [g for g in gates if (g.value <= threshold if at_most else g.value < threshold)]
    upper = [g for g in gates if (g.value > threshold if at_most else g.value >= threshold)]
    return (max(lower) if lower else None), (min(upper) if upper else None)


def format_gate(gate: Gate | None) -> str:
    """A decision as text."""
    if gate is None:
        return "none"
    detail = f" {gate.detail}" if gate.detail else ""
    return f"{gate.value:.4g} ({gate.workload}{detail})"


def get_decisions(trace: dict[str, Any]) -> list[tuple[Any, ...]]:
    """The decisions of a trace, in order, with predicted costs rounded to 9 digits."""
    decisions: list[tuple[Any, ...]] = []
    for event in trace["events"]:
        kind = event["event"]
        if kind == "enumeration":
            decisions.append((kind, event["test"], event["dimension"], event["enumerate"]))
        elif kind == "family":
            decisions.append(
                (kind, event["kind"], event["sets"], event.get("stage"), event.get("partitioned"))
            )
        elif kind == "certificate":
            decisions.append((kind, event.get("overlapping"), event.get("lp")))
        elif kind == "automorphisms":
            decisions.append((kind, event["switched"]))
        elif kind == "layer":
            decisions.append(
                (kind, event["weight"], event["dimension"], event["sets"], event["tables"])
            )
        elif kind == "limit" and not math.isinf(event["cost"]):
            cost = round(event["cost"], 9)
            decisions.append((kind, event.get("rows"), event["weight"], cost, event["within"]))
    return decisions


def compare(first: dict[str, Any], second: dict[str, Any]) -> None:
    """Print whether two traces of one workload make the same decisions.

    Layers that only one tuning considers for tables add comparisons with the limit to its trace;
    the comparison also reports the decisions without those.
    """
    print(f"{first['workload']}: results {first['result']} and {second['result']}")
    decisions = [get_decisions(first), get_decisions(second)]
    same = decisions[0] == decisions[1]
    print(f"    decisions: {len(decisions[0])} and {len(decisions[1])}, same: {same}")
    considered = [
        {
            (e["rows"], e["weight"])
            for e in trace["events"]
            if e["event"] == "considers" and e["considers"]
        }
        for trace in (first, second)
    ]
    for index, other in ((0, 1), (1, 0)):
        extra = considered[index] - considered[other]
        if not extra:
            continue
        # Older traces record only the weight of a comparison with the limit.
        weights = {weight for _, weight in extra}
        trimmed = [
            d
            for d in decisions[index]
            if not (
                d[0] == "limit" and ((d[1], d[2]) in extra or (d[1] is None and d[2] in weights))
            )
        ]
        print(f"    layers that only trace {index + 1} considers: {sorted(extra)}")
        print(f"    same decisions without their predictions: {trimmed == decisions[other]}")


if __name__ == "__main__":
    main()
