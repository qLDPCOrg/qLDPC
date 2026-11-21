# SPDX-License-Identifier: Apache-2.0

"""Analyze a run of the distance experiment: ratios, family tables, the figure, and summary.csv.

    python analyze.py (--archive PATH | --label-dir DIR) [--out DIR] [--allow-local]

The analysis is descriptive.  For each code, the time ratio R_T = T_prior / T_on of the BZ search
compares the prior arm with the arm with everything on.  T is the median of the times of an arm's
first run and repeats on the code, with a repeat that timed out counted as the limit L.  If the
first run did not finish, or a repeat ended in an error or memout, T is the first run's time.  An
arm counts as finished if its first run finished, even if a repeat timed out; ``summary.csv``
records the number of unfinished repeats in ``repeats_unfinished``, and a median that is L is in
effect a bound.  A row with ``compiled_in_timing`` set (see ``common/runner.py``) enters no T.  A
repeat with it is left out of the median, as a missing repeat is.  A first run with it leaves the
arm without T, and its T cell in ``anchors.csv`` is "compiled" unless the arm failed.
``summary.csv`` counts the flagged runs in ``runs_compiled``, and leaves them out of ``repeats`` and
``repeats_unfinished``.  The cases, in order:

1. **disagreement:** the arms or their repeats report different distances, or a distance that
   differs from the code's reference distance; no ratio;
2. **failure:** either arm's first run or one of its repeats ended in an error or memout, or its
   first run has no final row although ``tasks.json`` lists it; no ratio.  A repeat without a final
   row only leaves the median with fewer runs, which ``repeats`` in ``summary.csv`` records;
3. **incomplete timing:** the row of either arm's first run has ``compiled_in_timing`` set; no
   ratio;
4. **both finished:** R_T = T_prior / T_on;
5. **only on finished:** R_T >= L / T_on, a lower bound;
6. **only prior finished:** R_T <= T_prior / L, an upper bound;
7. **both timed out:** "both > L", no ratio.

A value above 1.2 is *faster* (the arm with everything on is faster), below 1/1.2 *slower*, and
otherwise *similar*; a lower bound is faster if it exceeds 1.2 and *inconclusive* otherwise, and an
upper bound is slower if it is below 1/1.2 and inconclusive otherwise.

For each family, the table gives the counts of each class of R_T and the median of R_T over the
codes of cases 4-6, with each bound at its value: marked ">=" if lower bounds are the only bounds,
"<=" if upper bounds are, and omitted if there are both.  The table also gives the number of these
ratios, and how many of them are lower bounds and how many upper bounds.  The anchors' row gives the
counts of each class only; anchors are reported per code and never pooled with the families.

The analysis reads the sample from the label directory, where ``experiment.prepare`` copies
``code_data/sample.json``, so that a later change to the sample, such as to a reference distance,
does not change the analysis of an earlier run.

Outputs in ``--out`` (by default ``analysis/`` in the label directory, which is a temporary
directory for ``--archive``): ``summary.csv`` (one row per code and arm), ``families.csv``,
``anchors.csv``, ``distances.pdf`` (a log-log panel of T, prior against on), and
``figure_notes.json`` (the number of codes that the panel omits, for lack of a value, or for a
disagreement, a failure, or an incomplete timing).

The analysis refuses a label whose final rows include rows of ``--local`` runs, unless given
``--allow-local``.  It then marks every output with their number: a ``local_rows`` column in each
CSV file, a ``local_rows`` key in ``figure_notes.json``, and the panel's title.
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
    local_rows = store.local_rows(final.values(), args.allow_local)
    sample = load_sample(label_dir)
    expected = list(tasks.load_tasks(label_dir, pass2=False).values())
    missing = [key for key in tasks.load_tasks(label_dir) if key not in final]
    if missing:
        print(f"note: {len(missing)} tasks have no final row, such as {missing[0]}")
    per_code = summarize(list(final.values()), sample, expected)
    write_outputs(per_code, sample, out, local_rows)
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


def load_sample(label_dir: pathlib.Path) -> dict[str, Any]:
    """The run's copy of ``sample.json``, which ``experiment.prepare`` puts in the label directory.

    Raises:
        ValueError: If the label directory has no copy, as for a run made before runs kept one.
    """
    path = label_dir / "sample.json"
    if not path.exists():
        raise ValueError(
            f"{label_dir} has no sample.json, as for a run made before runs kept a copy of the"
            " sample; copy code_data/sample.json from the run's harness commit (in its manifest)"
            " into it"
        )
    sample: dict[str, Any] = json.loads(path.read_text())
    return sample


def summarize(
    rows: list[dict[str, Any]], sample: dict[str, Any], expected: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """One record per code and arm: status, median T, and distance.

    Args:
        rows: The final rows of the runs.
        sample: The run's copy of the sample (``load_sample``).
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
    it then takes.  A repeat that timed out counts as the limit, which is larger than every finished
    value, and is counted in ``repeats_unfinished``.  A row with ``compiled_in_timing`` set enters
    no T and is left out of ``repeats`` and ``repeats_unfinished``: a repeat with it is left out of
    the median, and a first run with it makes T None and the timing incomplete.  Repeats that report
    different distances make the arm's distance a list, which ``ratio`` treats as a disagreement.
    """
    if not rows:
        return {
            "status": "missing",
            "T": None,
            "timing_complete": False,
            "distance": None,
            "limit_s": None,
        }
    first = rows[0]
    failed = [row["status"] for row in rows[1:] if row["status"] in ("error", "memout")]
    status = failed[0] if failed else first["status"]
    compiled = [row for row in rows if row.get("compiled_in_timing")]
    timing_complete = not first.get("compiled_in_timing")
    times = [
        row["T"] if row["status"] == "ok" else row["limit_s"]
        for row in rows
        if not row.get("compiled_in_timing")
    ]
    distances = sorted({row["result"].get("distance") for row in rows if row["status"] == "ok"})
    T = None
    if timing_complete:
        T = statistics.median(times) if status == "ok" else first["T"]
    return {
        "status": status,
        "T": T,
        "timing_complete": timing_complete,
        "distance": distances[0] if len(distances) == 1 else (distances or None),
        "limit_s": first["limit_s"],
        "repeats": len(rows) - len(compiled),
        "repeats_unfinished": sum(
            row["status"] != "ok" for row in rows[1:] if not row.get("compiled_in_timing")
        ),
        "runs_compiled": len(compiled),
    }


####################################################################################################
# families, tables, and the figure


def write_outputs(
    records: list[dict[str, Any]],
    sample: dict[str, Any],
    out: pathlib.Path,
    local_rows: int = 0,
) -> None:
    """Write summary.csv, families.csv, anchors.csv, and the figure.

    With ``local_rows``, the number of rows of ``--local`` runs, every output says so.
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
    _write_families(pairs, out)
    _write_anchors(by_code, sample, out)
    _plot(pairs, out, local_rows)
    if local_rows:
        store.mark_local(out, local_rows)


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
    classes = [
        "faster",
        "similar",
        "slower",
        "inconclusive",
        "both > L",
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
    with open(out / "families.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["family"])
        writer.writeheader()
        writer.writerows(rows)


def _write_anchors(
    by_code: dict[str, dict[str, dict[str, Any]]], sample: dict[str, Any], out: pathlib.Path
) -> None:
    """One row per anchor: n, k, the reference d, the d found, and the T of each arm."""
    arms = [ON, PRIOR]
    with open(out / "anchors.csv", "w", newline="") as file:
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


def _as_list(value: Any) -> list[Any]:
    """A distance or a list of distances as a list, without None."""
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _plot(pairs: list[dict[str, Any]], out: pathlib.Path, local_rows: int) -> None:
    """A log-log panel of T, with the prior arm's T on the x axis and the on arm's on the y axis."""
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
    axis.set_title(f"time (s); {local_rows} local rows" if local_rows else "time (s)")
    axis.set_xlabel("prior")
    axis.set_ylabel("on")
    figure.tight_layout()
    figure.savefig(out / "distances.pdf")
    (out / "figure_notes.json").write_text(json.dumps({"T": omitted}, indent=1) + "\n")


if __name__ == "__main__":
    main()
