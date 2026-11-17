# SPDX-License-Identifier: Apache-2.0

"""Analyze a run of the distance experiment: ratios, family tables, the figure, and summary.csv.

    python analyze.py (--archive PATH | --label-dir DIR) [--out DIR]

The analysis is descriptive.  For each code, the time ratio R_T = T_prior / T_on of the BZ search
compares the prior arm with the arm with everything on, where T is the median of a timed run's
repeats, a repeat that did not finish counting as the limit L.  An arm counts as finished if its
first run finished, even if its repeats did not; ``summary.csv`` records the number of unfinished
repeats in ``repeats_unfinished``, and a median that is L is in effect a bound.  The cases, in
order:

1. **disagreement:** the arms, their repeats, or their counting runs report different distances,
   or a distance that differs from the code's reference distance; no ratio;
2. **failure:** either arm's first timed run or one of its repeats ended in an error or memout,
   or its first timed run has no final row although ``tasks.json`` lists it (or, for R_W, either
   counting run failed or is missing); no ratio.  A repeat without a final row only leaves the
   median with fewer runs, which ``repeats`` in ``summary.csv`` records;
3. **both finished:** R_T = T_prior / T_on;
4. **only on finished:** R_T >= L / T_on, a lower bound;
5. **only prior finished:** R_T <= T_prior / L, an upper bound;
6. **both timed out:** "both > L", no ratio.

A value above 1.2 is *faster* (the arm with everything on is faster), below 1/1.2 *slower*, and
otherwise *similar*; a lower bound is faster if it exceeds 1.2 and *inconclusive* otherwise, and an
upper bound is slower if it is below 1/1.2 and inconclusive otherwise.  The work ratio
R_W = W_prior / W_on of the counting runs follows the same cases, with the last partial count of a
counting run that timed out in place of L, if it reported one, and is not classified; a pair
without a value for an unfinished run is ``no work``.  The plug-in forwards no partial counts
until qLDPC's work counters can report them, so a counting run that timed out currently gives no
W.

For each family, the table gives the counts of each class of R_T and the median of R_T over the
codes of cases 3-5, with each bound at its value: marked ">=" if lower bounds are the only bounds,
"<=" if upper bounds are, and omitted if there are both.  It gives the median of R_W by the same
rule, with the number of codes in each median and the number of codes whose R_W is a failure or
disagreement.  The anchors' row gives counts only; anchors are reported per code and never pooled
with the families.

Outputs in ``--out`` (by default ``analysis/`` in the label directory, which is a temporary
directory for ``--archive``): ``summary.csv`` (one row per code and arm), ``families.csv``,
``anchors.csv``, ``distances.pdf`` (two log-log panels, prior against on, by time and work), and
``figure_notes.json`` (the number of codes that each panel omits, for lack of a value or for a
disagreement or failure).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import statistics
import sys
import tarfile
import tempfile
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.state_prep_certification.common import store, tasks

FASTER = 1.2
ON, PRIOR = "q_bz_on", "q_bz_prior"


def main() -> None:
    """Analyze the run, and write the outputs."""
    args = parse_args()
    label_dir = args.label_dir or _unpack(args.archive)
    out = args.out or label_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    final, problems = store.final_rows(label_dir)
    for problem in problems:
        print(f"note: {problem}")
    sample = json.loads((pathlib.Path(__file__).parent / "code_data" / "sample.json").read_text())
    expected = list(tasks.load_tasks(label_dir, pass2=False).values())
    missing = [key for key in tasks.load_tasks(label_dir) if key not in final]
    if missing:
        print(f"note: {len(missing)} tasks have no final row, such as {missing[0]}")
    per_code = summarize(list(final.values()), sample, expected)
    write_outputs(per_code, sample, out)
    print(f"wrote {out}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=pathlib.Path, help="the archive from the node")
    source.add_argument("--label-dir", type=pathlib.Path, help="a label directory")
    parser.add_argument("--out", type=pathlib.Path, help="the output directory")
    return parser.parse_args()


def _unpack(archive: pathlib.Path) -> pathlib.Path:
    """Unpack an archive into a temporary directory, and return its label directory."""
    target = pathlib.Path(tempfile.mkdtemp(prefix="spc-analysis-"))
    with tarfile.open(archive) as tar:
        tar.extractall(target, filter="data")
    (label_dir,) = [path for path in target.iterdir() if path.is_dir()]
    return label_dir


####################################################################################################
# per code


def summarize(
    rows: list[dict[str, Any]], sample: dict[str, Any], expected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One record per code and arm: status, median T, work, and distance.

    Args:
        rows: The final rows of the runs.
        sample: The frozen sample.
        expected: The tasks of pass 1, so that a task without a final row is reported as missing.
    """
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for task in expected:
        code_hash = task["key"].split(".")[2]
        grouped.setdefault((code_hash, task["arm"]), {}).setdefault(task["mode"], [])
    for row in rows:
        code_hash = row["task_key"].split(".")[2]
        grouped.setdefault((code_hash, row["arm"]), {}).setdefault(row["mode"], []).append(row)
    names = {
        entry["code_hash"][:12]: (entry, kind)
        for kind in ("anchors", "codes")
        for entry in sample[kind]
    }
    unknown = sorted({code_hash for code_hash, _ in grouped if code_hash not in names})
    if unknown:
        raise ValueError(
            f"{len(unknown)} codes of the run are not in sample.json, such as {unknown[0]}: "
            "analyze the run with the sample.json that it ran with"
        )
    records = []
    for (code_hash, arm), modes in sorted(grouped.items()):
        entry, kind = names[code_hash]
        timed = sorted(modes.get("timed", []), key=lambda row: row["repeat"])
        counted = modes.get("count")
        records.append(
            {
                "name": entry["name"],
                "kind": kind,
                "family": entry.get("family", "anchor"),
                "code_hash": code_hash,
                "arm": arm,
                **_timed_summary(timed),
                **_count_summary(counted),
                "reference_d": entry.get("reference_d"),
            }
        )
    return records


def ratio(on: dict[str, Any], prior: dict[str, Any], metric: str) -> dict[str, Any]:
    """The ratio of the two arms on one code, by the cases of the module docstring.

    The distances compared include those of the counting runs, of every repeat, and the code's
    reference distance, if it has one.
    """
    found = _as_list(on.get("reference_d"))
    for record in (on, prior):
        for key in ("distance", "count_distance"):
            found += _as_list(record.get(key))
    if len(set(found)) > 1:
        return {"case": "disagreement", "value": None, "bound": None}
    if {on["status"], prior["status"]} & {"error", "memout", "missing"}:
        return {"case": "failure", "value": None, "bound": None}
    count_statuses = {on.get("count_status"), prior.get("count_status")}
    if metric == "W" and count_statuses & {"error", "memout", "missing"}:
        return {"case": "failure", "value": None, "bound": None}
    if metric == "T":
        on_value, prior_value = on["T"], prior["T"]
        on_done, prior_done = on["status"] == "ok", prior["status"] == "ok"
        on_bound = prior_bound = on["limit_s"]
    else:
        on_value, prior_value = on["W"], prior["W"]
        on_done, prior_done = bool(on["W_complete"]), bool(prior["W_complete"])
        on_bound, prior_bound = on["W"], prior["W"]
        if (not on_done and on_bound is None) or (not prior_done and prior_bound is None):
            return {"case": "no work", "value": None, "bound": None}
    if on_done and prior_done:
        return {"case": "both", "value": prior_value / on_value, "bound": None}
    if on_done:
        return {"case": "only on", "value": prior_bound / on_value, "bound": ">="}
    if prior_done:
        return {"case": "only prior", "value": prior_value / on_bound, "bound": "<="}
    return {"case": "both > L", "value": None, "bound": None}


def classify(result: dict[str, Any]) -> str:
    """Faster, similar, slower, or inconclusive; or the case, when there is no ratio."""
    value, bound = result["value"], result["bound"]
    if value is None:
        return str(result["case"])
    if bound is None:
        return "faster" if value > FASTER else "slower" if value < 1 / FASTER else "similar"
    if bound == ">=":
        return "faster" if value > FASTER else "inconclusive"
    return "slower" if value < 1 / FASTER else "inconclusive"


def _timed_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The status, median T, and distance of an arm's timed runs on one code.

    The status is that of the first run, unless a repeat ended in an error or memout, whose status
    it then takes.  A repeat that timed out counts as the limit, which is larger than every finished
    value, and is counted in ``repeats_unfinished``.  Repeats that report different distances make
    the arm's distance a list, which ``ratio`` treats as a disagreement.
    """
    if not rows:
        return {"status": "missing", "T": None, "distance": None, "limit_s": None}
    first = rows[0]
    failed = [row["status"] for row in rows[1:] if row["status"] in ("error", "memout")]
    status = failed[0] if failed else first["status"]
    times = [row["T"] if row["status"] == "ok" else row["limit_s"] for row in rows]
    distances = sorted({row["result"].get("distance") for row in rows if row["status"] == "ok"})
    return {
        "status": status,
        "T": statistics.median(times) if status == "ok" else first["T"],
        "distance": distances[0] if len(distances) == 1 else (distances or None),
        "limit_s": first["limit_s"],
        "repeats": len(rows),
        "repeats_unfinished": sum(row["status"] != "ok" for row in rows[1:]),
    }


def _count_summary(rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The work W of an arm's counting run, whether it is complete, and the run's status.

    A counting run that timed out gives the last partial count that it reported, if any, as a
    bound; a counting run that failed gives none.  ``rows`` is None if there is no counting task,
    and empty if the task has no final row.
    """
    if not rows:
        status = None if rows is None else "missing"
        return {"W": None, "W_complete": None, "count_status": status, "count_distance": None}
    row = rows[0]
    work = row["result"].get("work") if row["status"] == "ok" else None
    partial = [entry["work"] for entry in row.get("partial", []) if "work" in entry]
    if work is not None:
        total = _total_work(work)
        if total is None:
            raise ValueError(f"Unreadable work count of {row['task_key']}: {work!r}")
    elif row["status"] == "timeout" and partial:
        total = _total_work(partial[-1])
    else:
        total = None
    return {
        "W": total,
        "W_complete": row["status"] == "ok" and work is not None,
        "count_status": row["status"],
        "count_distance": row["result"].get("distance") if row["status"] == "ok" else None,
    }


def _total_work(work: Any) -> int | None:
    """The total of words examined of a count record."""
    if isinstance(work, dict):
        for key in ("total", "words_examined"):
            if key in work:
                return int(work[key])
    return None


####################################################################################################
# families, tables, and the figure


def write_outputs(records: list[dict[str, Any]], sample: dict[str, Any], out: pathlib.Path) -> None:
    """Write summary.csv, families.csv, anchors.csv, and the figure."""
    fields = [
        "name",
        "kind",
        "family",
        "code_hash",
        "arm",
        "status",
        "T",
        "repeats",
        "repeats_unfinished",
        "W",
        "W_complete",
        "count_status",
        "distance",
        "count_distance",
        "reference_d",
        "limit_s",
    ]
    with open(out / "summary.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    by_code: dict[str, dict[str, dict[str, Any]]] = {}
    for record in records:
        by_code.setdefault(record["name"], {})[record["arm"]] = record
    pairs = []
    for name, arms in by_code.items():
        if ON in arms and PRIOR in arms:
            on, prior = arms[ON], arms[PRIOR]
            time_ratio = ratio(on, prior, "T")
            pairs.append(
                {
                    "name": name,
                    "kind": on["kind"],
                    "family": on["family"],
                    "time": time_ratio,
                    "class": classify(time_ratio),
                    "work": ratio(on, prior, "W"),
                    "on": on,
                    "prior": prior,
                }
            )
    _write_families(pairs, out)
    _write_anchors(by_code, sample, out)
    _plot(pairs, out)


def _median(results: list[dict[str, Any]]) -> str:
    """The agreement-only median of ratios, marked by the kinds of bounds among them."""
    values = [r for r in results if r["value"] is not None]
    if not values:
        return "-"
    bounds = {r["bound"] for r in values} - {None}
    if bounds == {">=", "<="}:
        return "-"
    median = statistics.median(r["value"] for r in values)
    return f"{bounds.pop() if bounds else ''}{median:.3g}"


def _write_families(pairs: list[dict[str, Any]], out: pathlib.Path) -> None:
    """Counts and medians per family, and one row for the anchors."""
    classes = ["faster", "similar", "slower", "inconclusive", "both > L", "failure", "disagreement"]
    rows = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        family = pair["family"] if pair["kind"] == "codes" else "anchors"
        groups.setdefault(family, []).append(pair)
    for family, members in sorted(groups.items()):
        times = [
            m["time"] for m in members if m["time"]["case"] in ("both", "only on", "only prior")
        ]
        works = [m["work"] for m in members if m["work"]["value"] is not None]
        work_failures = sum(m["work"]["case"] in ("failure", "disagreement") for m in members)
        rows.append(
            {
                "family": family,
                "codes": len(members),
                **{name: sum(m["class"] == name for m in members) for name in classes},
                "median_R_T": _median(times) if family != "anchors" else "-",
                "ratios_in_median": len(times),
                "lower_bounds": sum(r["bound"] == ">=" for r in times),
                "upper_bounds": sum(r["bound"] == "<=" for r in times),
                "median_R_W": _median(works) if family != "anchors" else "-",
                "work_ratios_in_median": len(works),
                "work_failures": work_failures,
            }
        )
    with open(out / "families.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["family"])
        writer.writeheader()
        writer.writerows(rows)


def _write_anchors(
    by_code: dict[str, dict[str, dict[str, Any]]], sample: dict[str, Any], out: pathlib.Path
) -> None:
    """One row per anchor: n, k, the reference d, the d found, and T and W of each arm."""
    arms = [ON, PRIOR]
    with open(out / "anchors.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["name", "n", "k", "reference_d", "d_found"]
            + [f"{arm}_{metric}" for arm in arms for metric in ("T", "W")]
        )
        for entry in sample["anchors"]:
            records = by_code.get(entry["name"], {})
            found = sorted(
                {
                    value
                    for record in records.values()
                    for key in ("distance", "count_distance")
                    for value in _as_list(record.get(key))
                }
            )
            cells = []
            for arm in arms:
                record = records.get(arm)
                if record is None:
                    cells += ["n/a", "n/a"]
                elif record["status"] != "ok":
                    cells += [
                        "> L" if record["status"] == "timeout" else record["status"],
                        record["W"],
                    ]
                else:
                    cells += [f"{record['T']:.3g}", record["W"]]
            writer.writerow(
                [
                    entry["name"],
                    entry["n"],
                    entry["k"],
                    entry["reference_d"],
                    found[0] if len(found) == 1 else found,
                ]
                + cells
            )


def _as_list(value: Any) -> list[Any]:
    """A distance or a list of distances as a list, without None."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _plot(pairs: list[dict[str, Any]], out: pathlib.Path) -> None:
    """Two log-log panels, by time and by work; prior on x, on on y."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))
    families = sorted({p["family"] for p in pairs})
    colors = {family: plt.cm.tab10(i % 10) for i, family in enumerate(families)}
    omitted = {}
    for axis, metric in zip(axes, ("T", "W")):
        omitted[metric] = 0
        for pair in pairs:
            on, prior = pair["on"][metric], pair["prior"][metric]
            case = pair["time" if metric == "T" else "work"]["case"]
            if on is None or prior is None or case in ("disagreement", "failure"):
                omitted[metric] += 1
                continue
            if metric == "T":
                unfinished = pair["on"]["status"] != "ok" or pair["prior"]["status"] != "ok"
            else:
                unfinished = not (pair["on"]["W_complete"] and pair["prior"]["W_complete"])
            axis.scatter(
                prior,
                on,
                marker="^" if unfinished else "o",
                facecolors="none" if pair["kind"] == "anchors" else colors[pair["family"]],
                edgecolors=colors[pair["family"]],
                s=18,
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        limits = axis.get_xlim()
        axis.plot(limits, limits, "k-", lw=0.6)
        if metric == "T":
            axis.plot(limits, [x / FASTER for x in limits], "k:", lw=0.6)
            axis.plot(limits, [x * FASTER for x in limits], "k:", lw=0.6)
        axis.set_title(f"by {'time (s)' if metric == 'T' else 'work'}")
        axis.set_xlabel("prior")
        axis.set_ylabel("on")
    figure.tight_layout()
    figure.savefig(out / "distances.pdf")
    (out / "figure_notes.json").write_text(json.dumps(omitted, indent=1) + "\n")


if __name__ == "__main__":
    main()
