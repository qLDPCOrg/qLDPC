# SPDX-License-Identifier: Apache-2.0

"""Analyze a run of the distance experiment: summary.csv, families.csv, anchors.csv, distances.pdf.

Everything in ``state_prep_certification/`` is a prototype, subject to change: its code, interfaces,
inputs, outputs, and documentation.

    python analyze.py (--archive PATH --out DIR | --label-dir DIR [--out DIR]) [--allow-local]

The analysis is descriptive.  For each code, the time ratio R_T = T_prior / T_on of the BZ search
compares the prior arm with the arm with everything on.  T is the median of the times of an arm's
first run and repeats on the code, with a repeat that timed out counted as the limit L.  If the
first run finished with complete timing, and a run that timed out determines the median, as the
middle one of an odd number of runs or either middle one of an even number, the median timed out:
the arm has no T, and its status in ``summary.csv`` and its T cell in ``anchors.csv`` are
``median_timed_out``.  If the first run did not finish, or a repeat ended in an error or memout, T
is the first run's time.  An arm counts as finished if its first run finished and its median did not
time out; ``summary.csv`` records the number of unfinished repeats in ``repeats_unfinished``.  A row
with ``compiled_in_timing`` set (see ``common/runner.py``) enters no T.  A repeat with it is left
out of the median, as a missing repeat is.  A first run with it leaves the arm without T, and its T
cell in ``anchors.csv`` is "compiled" unless the arm failed.  ``summary.csv`` counts the flagged
runs in ``runs_compiled``, and leaves them out of ``repeats`` and ``repeats_unfinished``.  The
cases, in order:

1. **disagreement:** the arms or their repeats report different distances, or a distance that
   differs from the code's reference distance; no ratio;
2. **failure:** either arm's first run or one of its repeats ended in an error or memout, or its
   first run has no final row although ``tasks.json`` lists it or a repeat has one; no ratio.  A
   repeat without a final row only leaves the median with fewer runs, which ``repeats`` in
   ``summary.csv`` records;
3. **incomplete timing:** the row of either arm's first run has ``compiled_in_timing`` set; no
   ratio;
4. **median timed out:** the median of either arm timed out; no ratio;
5. **both finished:** R_T = T_prior / T_on;
6. **only on finished:** R_T >= L / T_on, a lower bound;
7. **only prior finished:** R_T <= T_prior / L, an upper bound;
8. **both timed out:** "both > L", no ratio.

A value above 1.2 is *faster* (the arm with everything on is faster), below 1/1.2 *slower*, and
otherwise *similar*; a lower bound is faster if it exceeds 1.2 and *inconclusive* otherwise, and an
upper bound is slower if it is below 1/1.2 and inconclusive otherwise.

For each family, ``families.csv`` gives the counts of each class of R_T and the median of R_T over
the codes of cases 5-7, with each bound at its value: marked ">=" if lower bounds are the only
bounds, "<=" if upper bounds are, and omitted if there are both.  It also gives the number of these
ratios, and how many of them are lower bounds and how many upper bounds.  The anchors' row gives the
counts of each class only; anchors are reported per code and never pooled with the families.

The analysis reads the sample from the label directory, where ``experiment.prepare`` copies
``code_data/sample.json``, so that a later change to the sample, such as to a reference distance,
does not change the analysis of an earlier run.

The analysis checks an archive against its ``.sha256`` file and its inner ``SHA256SUMS``, and
extracts it into a temporary directory, which it removes when it ends, whether or not it succeeds
(``store.open_archive``).  So ``--archive`` requires ``--out``.  The outputs go to ``--out``, by
default ``analysis/`` in the label directory of ``--label-dir``.  They are ``summary.csv`` (one row
per code and arm), ``families.csv``, ``anchors.csv``, ``distances.pdf`` (a log-log plot of T, prior
against on), and ``figure_notes.json`` (the number of codes that ``distances.pdf`` omits, for lack
of a value, or for a disagreement, a failure, or an incomplete timing).

The analysis refuses a label whose final rows include rows of ``--local`` runs, unless given
``--allow-local``.  With ``--allow-local``, it marks every output with the number of these rows,
``local_rows``.  If a start of the label accepted a refused environment variable with
``--allow-env`` (see ``run_node.py``), or ran Python with -O, it marks every output with the names
of these variables, ``PYTHONOPTIMIZE`` for -O, ``allowed_environment``.  If the recorded environment
variables of the label's starts differ, such as those of a resume in another shell, it marks every
output with the names of the variables that differ, ``environment_changed``.  Each mark is a column
in each CSV file, a key in ``figure_notes.json``, and part of the title and of the Subject in the
metadata of ``distances.pdf``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import pathlib
import statistics
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.state_prep_certification.common import store, tasks

FASTER = 1.2
ON, PRIOR = "q_bz_on", "q_bz_prior"


def main() -> None:
    """Analyze the run, and write the outputs."""
    args = parse_args()
    with contextlib.ExitStack() as archives:
        label_dir = args.label_dir or archives.enter_context(store.open_archive(args.archive))
        analyze_label(label_dir, args.out or label_dir / "analysis", args.allow_local)


def analyze_label(label_dir: pathlib.Path, out: pathlib.Path, allow_local: bool) -> None:
    """Analyze a label directory, and write the outputs to ``out``."""
    label_tasks = tasks.load_tasks(label_dir)
    out.mkdir(parents=True, exist_ok=True)
    final, problems = store.final_rows(label_dir)
    for problem in problems:
        print(f"note: {problem}")
    marks: dict[str, Any] = {}
    local_rows = store.count_local_rows(final.values(), allow_local)
    if local_rows:
        marks["local_rows"] = local_rows
    allowed = store.allowed_environment(label_dir)
    if allowed:
        print(
            f"note: a start ran with, or accepted, settings that a node start refuses: "
            f"{', '.join(allowed)}; every output names them"
        )
        marks["allowed_environment"] = " ".join(allowed)
    changed = store.changed_environment(label_dir)
    if changed:
        print(f"note: the starts ran with different {', '.join(changed)}; every output says so")
        marks["environment_changed"] = " ".join(changed)
    sample = load_sample(label_dir)
    missing = [key for key in label_tasks if key not in final]
    if missing:
        print(f"note: {len(missing)} tasks have no final row, such as {missing[0]}")
    per_code = summarize(list(final.values()), sample, label_tasks)
    write_outputs(per_code, sample, out, marks)
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
    parser.add_argument(
        "--allow-local", action="store_true", help="analyze rows of --local runs, and say so"
    )
    args = parser.parse_args()
    if args.archive and args.out is None:
        parser.error("--archive requires --out, since the archive's extraction is removed")
    return args


####################################################################################################
# per code


def load_sample(label_dir: pathlib.Path) -> dict[str, Any]:
    """The run's copy of ``sample.json``, which ``experiment.prepare`` puts in the label directory.

    Raises:
        ValueError: If the label directory has no ``sample.json``.
    """
    path = label_dir / "sample.json"
    if not path.exists():
        raise ValueError(
            f"{label_dir} has no sample.json, which experiment.prepare copies into each label "
            "directory before its tasks run; analyze the run's archive, which holds it"
        )
    sample: dict[str, Any] = json.loads(path.read_text())
    return sample


def summarize(
    rows: list[dict[str, Any]], sample: dict[str, Any], label_tasks: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """One record per code and arm: status, median T, and distance.

    Args:
        rows: The final rows of the runs.
        sample: The run's copy of the sample (``load_sample``).
        label_tasks: Every task of the label by key, as ``tasks.load_tasks`` gives them.  Each row's
            code is the input of its task, and an arm whose first run has no final row is missing.

    Raises:
        ValueError: If a row's key is not the key of a task, or a code is not in ``sample``.
    """
    unknown_keys = sorted({row["task_key"] for row in rows} - label_tasks.keys())
    if unknown_keys:
        raise ValueError(
            f"{len(unknown_keys)} rows have no task in tasks.json, such as {unknown_keys[0]}"
        )
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = {}
    for task in label_tasks.values():
        code_hash = task["input"]["code_hash"][:12]
        grouped.setdefault((code_hash, task["arm"]), {}).setdefault(task["mode"], [])
    for row in rows:
        code_hash = label_tasks[row["task_key"]]["input"]["code_hash"][:12]
        grouped.setdefault((code_hash, row["arm"]), {}).setdefault(row["mode"], []).append(row)
    names = {
        entry["code_hash"][:12]: (entry, kind)
        for kind in ("anchors", "codes")
        for entry in sample[kind]
    }
    unknown = sorted({code_hash for code_hash, _ in grouped if code_hash not in names})
    if unknown:
        raise ValueError(
            f"{len(unknown)} codes of the run are not in the run's sample.json, such as "
            f"{unknown[0]}; the label directory's sample.json should be the one that the run used"
        )
    records = []
    for (code_hash, arm), modes in sorted(grouped.items()):
        entry, kind = names[code_hash]
        timed = sorted(modes.get("timed", []), key=lambda row: row["repeat"])
        records.append(
            {
                "name": entry["name"],
                "kind": kind,
                "family": entry.get("family", "anchor"),
                "code_hash": code_hash,
                "arm": arm,
                **_timed_summary(timed),
                "reference_d": entry.get("reference_d"),
            }
        )
    return records


def ratio(on: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    """The time ratio R_T of the two arms on one code, by the cases of the module docstring.

    The distances compared include those of every repeat, and the code's reference distance, if it
    has one.
    """
    found = _as_list(on.get("reference_d"))
    for record in (on, prior):
        found += _as_list(record.get("distance"))
    if len(set(found)) > 1:
        return {"case": "disagreement", "value": None, "bound": None}
    if {on["status"], prior["status"]} & {"error", "memout", "missing"}:
        return {"case": "failure", "value": None, "bound": None}
    if not (on["timing_complete"] and prior["timing_complete"]):
        return {"case": "incomplete timing", "value": None, "bound": None}
    if "median_timed_out" in (on["status"], prior["status"]):
        return {"case": "median timed out", "value": None, "bound": None}
    on_done, prior_done = on["status"] == "ok", prior["status"] == "ok"
    if on_done and prior_done:
        return {"case": "both", "value": prior["T"] / on["T"], "bound": None}
    if on_done:
        return {"case": "only on", "value": on["limit_s"] / on["T"], "bound": ">="}
    if prior_done:
        return {"case": "only prior", "value": prior["T"] / on["limit_s"], "bound": "<="}
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
    """The status, median T, and distance of an arm's runs on one code.

    The status is that of the first run, unless a repeat ended in an error or memout, whose status
    it then takes.  An arm without a first run is missing, even if a repeat has a row, and keeps the
    repeats' distances for the disagreement check.  A repeat that timed out counts as the limit in
    the median, and is counted in ``repeats_unfinished``.  If the first run finished with complete
    timing, and a run that timed out determines the median, as the middle one of an odd number of
    runs or either middle one of an even number, the status is ``median_timed_out`` and T is None.
    A row with ``compiled_in_timing`` set enters no T and is left out of ``repeats`` and
    ``repeats_unfinished``: a repeat with it is left out of the median, and a first run with it
    makes T None and the timing incomplete.  Repeats that report different distances make the arm's
    distance a list, which ``ratio`` treats as a disagreement.
    """
    distances = sorted({row["result"].get("distance") for row in rows if row["status"] == "ok"})
    distance = distances[0] if len(distances) == 1 else (distances or None)
    if not rows or rows[0]["repeat"] != 0:
        return {
            "status": "missing",
            "T": None,
            "timing_complete": False,
            "distance": distance,
            "limit_s": None,
        }
    first = rows[0]
    failed = [row["status"] for row in rows[1:] if row["status"] in ("error", "memout")]
    status = failed[0] if failed else first["status"]
    compiled = [row for row in rows if row.get("compiled_in_timing")]
    timing_complete = not first.get("compiled_in_timing")
    counted = sorted(
        (row["T"], False) if row["status"] == "ok" else (row["limit_s"], True)
        for row in rows
        if not row.get("compiled_in_timing")
    )
    middle = counted[(len(counted) - 1) // 2 : len(counted) // 2 + 1]
    if status == "ok" and timing_complete and any(timed_out for _, timed_out in middle):
        status = "median_timed_out"
    T = None
    if timing_complete and status != "median_timed_out":
        T = statistics.median(value for value, _ in counted) if status == "ok" else first["T"]
    return {
        "status": status,
        "T": T,
        "timing_complete": timing_complete,
        "distance": distance,
        "limit_s": first["limit_s"],
        "repeats": len(rows) - len(compiled),
        "repeats_unfinished": sum(
            row["status"] != "ok" for row in rows[1:] if not row.get("compiled_in_timing")
        ),
        "runs_compiled": len(compiled),
    }


####################################################################################################
# summary.csv, families.csv, anchors.csv, and distances.pdf


def write_outputs(
    records: list[dict[str, Any]],
    sample: dict[str, Any],
    out: pathlib.Path,
    marks: dict[str, Any] | None = None,
) -> None:
    """Write summary.csv, families.csv, anchors.csv, distances.pdf, and figure_notes.json.

    Every output carries ``marks``, if any, such as the number of rows of ``--local`` runs: as a
    column, a key, or part of the title and the metadata of ``distances.pdf``.  Other files in
    ``out`` are left as they are.
    """
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
        "runs_compiled",
        "distance",
        "reference_d",
        "limit_s",
    ]
    summary = out / "summary.csv"
    with open(summary, "w", newline="") as file:
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
            time_ratio = ratio(on, prior)
            pairs.append(
                {
                    "name": name,
                    "kind": on["kind"],
                    "family": on["family"],
                    "time": time_ratio,
                    "class": classify(time_ratio),
                    "on": on,
                    "prior": prior,
                }
            )
    paths = [summary, _write_families(pairs, out), _write_anchors(by_code, sample, out)]
    paths += _plot(pairs, out, marks or {})
    if marks:
        store.mark_outputs(paths, marks)


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


def _write_families(pairs: list[dict[str, Any]], out: pathlib.Path) -> pathlib.Path:
    """Write families.csv: counts and medians per family, and one row for the anchors."""
    classes = [
        "faster",
        "similar",
        "slower",
        "inconclusive",
        "both > L",
        "median timed out",
        "incomplete timing",
        "failure",
        "disagreement",
    ]
    rows = []
    groups: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        family = pair["family"] if pair["kind"] == "codes" else "anchors"
        groups.setdefault(family, []).append(pair)
    for family, members in sorted(groups.items()):
        times = [
            m["time"] for m in members if m["time"]["case"] in ("both", "only on", "only prior")
        ]
        rows.append(
            {
                "family": family,
                "codes": len(members),
                **{name: sum(m["class"] == name for m in members) for name in classes},
                "median_R_T": "-",
                "ratios_in_median": "-",
                "lower_bounds": "-",
                "upper_bounds": "-",
            }
            | (
                {}
                if family == "anchors"
                else {
                    "median_R_T": _median(times),
                    "ratios_in_median": len(times),
                    "lower_bounds": sum(r["bound"] == ">=" for r in times),
                    "upper_bounds": sum(r["bound"] == "<=" for r in times),
                }
            )
        )
    path = out / "families.csv"
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["family"])
        writer.writeheader()
        writer.writerows(rows)
    return path


def _write_anchors(
    by_code: dict[str, dict[str, dict[str, Any]]], sample: dict[str, Any], out: pathlib.Path
) -> pathlib.Path:
    """Write anchors.csv: one row per anchor, with n, k, the reference d, the d found, and Ts.

    ``d_found`` is the distance that the arms' runs on the anchor report, or the list of their
    distances if they report none or several.  An arm's T cell is its T to three significant digits,
    or else "n/a" where the arm has no runs on the anchor, "compiled" where its first run, finished
    or timed out, has ``compiled_in_timing`` set, "> L" where it timed out, and otherwise its
    status, such as "error", "memout", "missing", or "median_timed_out".
    """
    arms = [ON, PRIOR]
    path = out / "anchors.csv"
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["name", "n", "k", "reference_d", "d_found"] + [f"{arm}_T" for arm in arms])
        for entry in sample["anchors"]:
            records = by_code.get(entry["name"], {})
            found = sorted(
                {value for record in records.values() for value in _as_list(record.get("distance"))}
            )
            cells = []
            for arm in arms:
                record = records.get(arm)
                if record is None:
                    cells.append("n/a")
                elif record["status"] in ("ok", "timeout") and not record["timing_complete"]:
                    cells.append("compiled")
                elif record["status"] != "ok":
                    cells.append("> L" if record["status"] == "timeout" else record["status"])
                else:
                    cells.append(f"{record['T']:.3g}")
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
    return path


def _as_list(value: Any) -> list[Any]:
    """A distance or a list of distances as a list, without None."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _plot(
    pairs: list[dict[str, Any]], out: pathlib.Path, marks: dict[str, Any]
) -> list[pathlib.Path]:
    """A log-log plot of T, with the prior arm's T on the x axis and the on arm's on the y axis.

    Write distances.pdf and figure_notes.json, and return their paths.  The marks go into the
    title, and into the Subject of the PDF's metadata.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4, 4))
    families = sorted({p["family"] for p in pairs})
    colors = {family: plt.cm.tab10(i % 10) for i, family in enumerate(families)}
    omitted = 0
    for pair in pairs:
        on, prior = pair["on"]["T"], pair["prior"]["T"]
        if (
            on is None
            or prior is None
            or pair["time"]["case"] in ("disagreement", "failure", "incomplete timing")
        ):
            omitted += 1
            continue
        unfinished = pair["on"]["status"] != "ok" or pair["prior"]["status"] != "ok"
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
    axis.plot(limits, [x / FASTER for x in limits], "k:", lw=0.6)
    axis.plot(limits, [x * FASTER for x in limits], "k:", lw=0.6)
    marked = "; ".join(f"{name}: {value}" for name, value in marks.items())
    axis.set_title(f"time (s); {marked}" if marks else "time (s)")
    axis.set_xlabel("prior")
    axis.set_ylabel("on")
    figure.tight_layout()
    figure.savefig(out / "distances.pdf", metadata={"Subject": marked} if marks else None)
    (out / "figure_notes.json").write_text(json.dumps({"T": omitted}, indent=1) + "\n")
    return [out / "distances.pdf", out / "figure_notes.json"]


if __name__ == "__main__":
    main()
