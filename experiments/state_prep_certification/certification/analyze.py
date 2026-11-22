# SPDX-License-Identifier: Apache-2.0

"""Analyze a run of the certification experiment: its CSV files and counterexamples.

    python analyze.py (--archive PATH --out DIR | --label-dir DIR [--out DIR])
        [--check-archive PATH | --check-label-dir DIR] [--allow-local]

The analysis is descriptive: counts and medians, with every timeout, memout, error, missing run,
and disagreement in its CSV files.  The ground truth runs in the subset ``check``, which has its own
label directory and archive; ``--check-label-dir`` or ``--check-archive`` adds its ``ground_truth``
rows to those of the analyzed run, if any.  The analysis extracts each archive into a temporary
directory, which it removes when it ends, whether or not it succeeds.

The circuits and their parameters come from the label's copy of the ``circuits.json`` that it ran
with.  If the label does not match that ``circuits.json`` and the working tree's ``config.json``
and ``pins.json``, the analysis prints a warning and analyzes the label anyway.  The label's hash
is recomputed from these files and from the harness commit and bundle that the label's
``manifest.json`` records.  The ``circuits.json`` of the label of ``check`` must equal that of the
analyzed label.  Every task of the label's ``tasks.json`` and ``tasks-pass2.json`` that has no row
is reported as ``missing``, with a printed note.

The analysis refuses rows of ``--local`` runs, of either label, unless given ``--allow-local``.
With ``--allow-local``, it marks every output with the number of these rows, ``local_rows``.  If a
start of either label accepted a refused environment variable with ``--allow-env`` (see
``run_node.py``), or ran Python with -O, it marks every output with the names of these variables,
``PYTHONOPTIMIZE`` for -O, ``allowed_environment``.  If the recorded environment variables of a
label's starts differ, such as those of a resume in another shell, it marks every output with the
names of the variables that differ, ``environment_changed``.  Each mark is a column in each CSV
file, a key in each JSON file, and a first comment line in each circuit of ``pruned/``; the
``final_sha256`` of ``deletion_table.csv`` is that of the circuit without these lines.

An arm's first run that timed out with ``compiled_in_timing`` set (see ``common/runner.py``) has the
status ``timeout_compiled`` in every output, not ``timeout``, since it may have timed out only
because it compiled.  A circuit's verdict is that of ``cert_bz``, or of ``cert_cc`` where
``cert_bz`` did not finish.  Where neither finished, it is ``undecided``, or ``timeout_compiled`` if
either arm's status is ``timeout_compiled``.  It is a disagreement where any two of its exact
results contradict each other, as ``check_consistency`` lists: its verdicts, smallest violating
weights, and margins from every run, repeats included, its counts, and its counterexamples.  The
column ``contradictions`` of summary.csv names them.  A disagreement enters no count of verdicts or
margins and no median, maximum, or ratio of ``families.csv``, nor ``listing_table.csv``,
``shadow.csv``, ``margin_violations.csv``, ``deletion_table.csv``, or the example counterexamples.
The status counts and agreements count it.  T of a certification arm is the median of its first run
and repeats.  Its timing is incomplete if the first run did not finish, if a repeat that the first
run calls for did not finish or is missing, or if the row of any of its runs has
``compiled_in_timing`` set; an incomplete timing enters no median or maximum.  The same holds for
the brute-force baseline ``brute_bz``.  If an arm's first run has ``compiled_in_timing`` set, its
per-weight times are left empty.

Omega_m is the number of rows that the fault search submits to the oracle at fault weight m
(``instrument.py``), and a weight is completed if a run finished it without a violation.  A circuit
whose enumeration size U exceeds the brute-force cap of ``config.json`` has no ``brute_bz`` task,
and its status is "not run (U > cap)", never a timeout.  An agreement is "a/b", a of b compared
circuits, or "not run" where no circuit of the family has rows of the arm compared.

Outputs, in ``--out``, which ``--archive`` requires (by default ``analysis/`` in the label directory
of ``--label-dir``):

* ``families.csv``, one row per family:

  - ``circuits``;
  - ``certified_at_t``: circuits certified at K = t, as their verdict at K = t + 1 is certified or
    a smallest violation at t + 1; ``violations_at_most_t``: circuits with a smallest violation at
    a weight of at most t;
  - ``certified_at_t_plus_1`` and ``violations_at_t_plus_1``: the outcome at K = t + 1;
  - ``undecided``: circuits that neither certification arm finished, except disagreements and
    those whose verdict is ``timeout_compiled``;
  - ``timeout_compiled``: circuits whose verdict is ``timeout_compiled``; ``disagreements``;
  - ``timeout_<arm>``, ``memout_<arm>``, ``error_<arm>``, and ``missing_<arm>`` for each arm of
    ``REPORTED_ARMS``, and ``timeout_compiled_<arm>``: first runs with status ``timeout_compiled``,
    which ``timeout_<arm>`` does not count;
  - ``largest_n_certified`` and ``largest_K_certified``, over the circuits certified at their K,
    with the cat width as n for cat circuits;
  - ``T_bz_median``, ``T_bz_max``, and ``T_bz_finished``, the median and largest T of ``cert_bz``
    over its finished runs with complete timing, and their number, and the same for ``cert_cc``;
  - ``brute_not_run_over_cap``, ``brute_finished``, and ``brute_timed_out``: circuits over the
    brute-force cap, and circuits whose ``brute_bz`` run has status ``ok`` or ``timeout``;
    ``brute_ratio_median`` and ``brute_ratio_circuits``: the median of T(``brute_bz``) /
    T(``cert_bz``) over the circuits where both finished with complete timing, and their number.
    T(``brute_bz``) includes the time of the baseline's count of distinct syndromes
    (``distinct_seconds``; see ``baseline.py``), which raises the ratio in the certifier's favor;
  - ``max_rss_gib``: the largest peak memory of a run of ``cert_bz`` or ``cert_cc``, of any
    status, as each ``rss_*`` of summary.csv is (see ``summarize_certification``);
  - ``agreement_bz_cc`` (verdicts), ``agreement_baseline`` (the verdict and smallest violating
    weight of a finished ``brute_bz`` against those of each finished certification, and, finished or
    not, its enumerated combinations against A_m of ``count_accepted``, or of the ground truth where
    ``count_accepted`` did not compute it, which they must equal on each weight that it completed
    and not exceed on one where it found a violation), and ``agreement_ground_truth`` (the smallest
    violating weight of ``ground_truth`` against that of every finished certification, and its Pi,
    E, Phi, and A against those of ``count_listing`` and ``count_accepted`` on the weights that both
    completed).

* ``summary.csv``, one row per circuit: its parameters and DEM counts, U and whether it is within
  the brute-force cap, the verdicts, T, timing completeness, and peak memory of both certification
  arms and of ``brute_bz``, the status of each arm of ``REPORTED_ARMS``, the post-processing status
  of each certification's first run, ``repeats_not_ok`` (the repeats that did not finish or are
  missing, and the runs whose rows have ``compiled_in_timing`` set), the oracle rows of ``cert_bz``
  (its threshold-1 queries and the sum of Omega_m) and of ``brute_bz`` (every row queried), the
  agreements, and the ``contradictions`` that make the circuit a disagreement.  The counts of a
  ``count_listing`` run that did not finish are those of its partial results: q, q_h, and h, and Pi,
  E, and Phi of the weights that it completed.
* ``per_weight.csv``, one row per circuit and weight m = 1, ..., K: A_m, C(N_0, m) (from N_0 of
  ``circuits.json`` where ``count_accepted`` did not finish), ``naive_prefixes_m_minus_1`` (Pi_(m-1)
  of ``counting.py``: the prefixes of a naive enumeration of class codewords, most of which qLDPC's
  walk by shared flags does not visit), E_m, Phi_m, Omega_m (marked partial on a weight that did not
  complete; at m = 1, the threshold-1 checks of zero-flag mechanisms and classifications of flagged
  ones instead), skipped = Phi_m - Omega_m (on completed weights only), T_m of BZ and CC, and the
  oracle time of BZ; and the baseline's combinations enumerated, rows queried, oracle calls,
  distinct syndromes, and oracle time, and whether it completed the weight.  For K' < m <= K,
  Phi_m = 0, and Omega_m = 0 where ``cert_bz`` certified the circuit.
* ``listing_table.csv``, the listing counts, one row per family and weight m >= 2: the median and
  maximum over circuits of Phi_m / A_m, and over the circuits that completed m with ``cert_bz``, of
  Omega_m / A_m.  Where A_m was not computed, C(N_0, m) takes its place if z >= m, which ensures
  A_m > 0: ``C_in_place_of_A`` counts these circuits, and ``ratio_is_lower_bound`` is true where
  there are any.  As A_m <= C(N_0, m), their ratios are at most the true ones, and so are the
  medians and maxima, which are then lower bounds on those of Phi_m / A_m and Omega_m / A_m.  A
  circuit whose A_m is 0, or was not computed while z < m, has no ratio, and ``ratio_unknown``
  counts it.  Beside them are the medians, over the circuits with a ratio, of the parameters that
  the listing depends on: the heavy share h / N_0, the unflagged share z / N_0, the start classes
  q_h, and the flag rank r.  ``listing_table.csv`` describes when
  Phi_m is small against A_m; it fits nothing.
* ``shadow.csv``, the cost of the two weight oracles on identical queries, one row per circuit and
  threshold m of the queries of its ``shadow`` run: the calls, their rows, and the seconds of BZ
  and CC (``shadow_table``).  A run that did not finish gives the calls that it wrote to
  ``oracle_calls/`` in the label directory, and no row if it wrote none; ``summary.csv`` gives its
  status.  Runs with ``compiled_in_timing`` set and circuits with a disagreement are left out.
* The margins and deletions (``analyze_applications.py``): the combined margin m* of each circuit
  and its agreement in ``summary.csv`` (``margin``, ``margin_lower_bound``, ``agree_margin``, and
  the arm and sector of one counterexample, ``counterexample_*``), with ``agree_prune`` for its
  deletion tasks; the margin counts of ``families.csv``; and ``margin_violations.csv``,
  ``deletion.csv``, ``deletion_table.csv``, ``deletion_tasks.csv``, ``deletion_notes.json``, and
  ``pruned/``.
* ``counterexamples.json``: the counterexamples of the first runs of ``cert_bz`` and ``cert_cc``
  (``all``), with ``disagreement`` true where its circuit is a disagreement, and up to three example
  counterexamples (``examples``), one for each group that has one, chosen by a fixed rule.  The
  groups are A-unv with the W variants of A-ft and C-goto, B-cat, and both D-synth families with
  their W variants, so that every W variant of ``circuits.json`` is in a group.  In each group, the
  example is the counterexample of ``cert_bz`` of largest smallest-violation weight on a circuit
  without a disagreement, ties to the smallest N_0 of ``circuits.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import statistics
import sys
import tarfile
import tempfile
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.state_prep_certification.certification import analyze_applications
from experiments.state_prep_certification.common import manifest, store
from experiments.state_prep_certification.common import tasks as task_store

HERE = pathlib.Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text())
REPEATS = CONFIG["repeats"]
BRUTE_CAP = int(CONFIG["brute_force"]["max_combinations"])
LIMIT_S = float(CONFIG["limits"]["cert_s"])

# The status of the brute-force baseline on a circuit over its cap, as in "not run (U > 1e9)".
OVER_CAP = "not run (U > {})".format(
    f"1e{len(str(BRUTE_CAP)) - 1}" if str(BRUTE_CAP).rstrip("0") == "1" else BRUTE_CAP
)

# The arms whose timeouts, memouts, and errors summary.csv and families.csv report.
REPORTED_ARMS = (
    "cert_bz",
    "cert_cc",
    "brute_bz",
    "shadow",
    "count_accepted",
    "count_listing",
    "ground_truth",
    "margin",
    "prune",
    "prune_sensitivity",
)

# The statuses of an arm's first run that families.csv counts per arm (see ``_status``).
COUNTED_STATUSES = ("timeout", "timeout_compiled", "memout", "error", "missing")

# The fields of ``circuits.json`` that hold the parameters of ``summary.csv`` by another name.
CIRCUIT_FIELDS = {"p": "num_post_selection_detectors", "N": "num_mechanisms"}

# The groups of the example counterexamples, as sets of (family, family of a W variant's parent).
EXAMPLE_GROUPS: dict[str, set[tuple[str, str | None]]] = {
    "A-unv/W": {("A-unv", None), ("W", "A-ft"), ("W", "C-goto")},
    "B-cat": {("B-cat", None)},
    "D-synth/W": {
        ("D-synth", None),
        ("D-synth (MQT 1.9.0)", None),
        ("W", "D-synth"),
        ("W", "D-synth (MQT 1.9.0)"),
    },
}


def main() -> None:
    """Analyze the run, and write the outputs."""
    args = parse_args()
    with tempfile.TemporaryDirectory(prefix="spc-analysis-") as scratch:
        label_dir = args.label_dir or _unpack(args.archive, pathlib.Path(scratch, "main"))
        check_dir = args.check_label_dir or (
            args.check_archive and _unpack(args.check_archive, pathlib.Path(scratch, "check"))
        )
        analyze_label(label_dir, check_dir, args.out or label_dir / "analysis", args.allow_local)


def analyze_label(
    label_dir: pathlib.Path,
    check_dir: pathlib.Path | None,
    out: pathlib.Path,
    allow_local: bool,
) -> None:
    """Analyze a label, with the ``ground_truth`` rows of the label of ``check``, if any."""
    out.mkdir(parents=True, exist_ok=True)
    check_frozen_files(label_dir)
    rows = _final_rows(label_dir)
    expected = _expected_tasks(label_dir)
    circuits_text = (label_dir / "circuits.json").read_text()
    if check_dir:
        check_frozen_files(check_dir)
        if (check_dir / "circuits.json").read_text() != circuits_text:
            raise SystemExit(
                f"{check_dir.name} ran with another circuits.json than {label_dir.name}"
            )
        rows += [row for row in _final_rows(check_dir) if row["arm"] == "ground_truth"]
        expected |= {
            key: task
            for key, task in _expected_tasks(check_dir).items()
            if task["arm"] == "ground_truth"
        }
    marks: dict[str, Any] = {}
    local_rows = store.count_local_rows(rows, allow_local)
    if local_rows:
        marks["local_rows"] = local_rows
    allowed = sorted(
        set(store.allowed_environment(label_dir))
        | set(store.allowed_environment(check_dir) if check_dir else [])
    )
    if allowed:
        print(
            f"note: a start ran with, or accepted, settings that a node start refuses: "
            f"{', '.join(allowed)}; every output names them"
        )
        marks["allowed_environment"] = " ".join(allowed)
    changed = sorted(
        set(store.changed_environment(label_dir))
        | set(store.changed_environment(check_dir) if check_dir else [])
    )
    if changed:
        print(f"note: the starts ran with different {', '.join(changed)}; every output says so")
        marks["environment_changed"] = " ".join(changed)
    circuits = {e["id"]: e for e in json.loads(circuits_text)["circuits"]}
    write_outputs(
        collect(rows, circuits, expected), circuits, out, marks, label_dir / "oracle_calls"
    )
    print(f"wrote {out}")


def write_outputs(
    per_circuit: dict[str, dict[str, Any]],
    circuits: dict[str, dict[str, Any]],
    out: pathlib.Path,
    marks: dict[str, Any] | None = None,
    calls_dir: pathlib.Path | None = None,
) -> None:
    """Write every output of the circuits that ``collect`` gives, as the module docstring lists.

    ``calls_dir``, if given, holds the files of oracle calls of the ``shadow`` runs, as a label's
    ``oracle_calls/`` does (see ``shadow_table``).  Every output takes each circuit's verdict from
    the rows of summary.csv.  Every output that this
    call writes carries ``marks``, if any, such as the number of rows of ``--local`` runs
    (``store.mark_outputs``).  The directory ``out/pruned/`` is deleted with everything in it and
    written again, so that it holds only this call's circuits; other files in ``out`` are left as
    they are.
    """
    summary_rows, weight_rows = summarize(per_circuit)
    paths = write_tables(summary_rows, weight_rows, out)
    verdicts = {row["id"]: row["verdict"] for row in summary_rows}
    paths.append(_write_csv(out / "shadow.csv", shadow_table(per_circuit, verdicts, calls_dir)))
    paths.append(write_counterexamples(per_circuit, verdicts, circuits, out))
    paths += analyze_applications.write_applications(per_circuit, summary_rows, out)
    if marks:
        store.mark_outputs(paths, marks)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=pathlib.Path, help="the archive from the node")
    source.add_argument("--label-dir", type=pathlib.Path, help="a label directory")
    check = parser.add_mutually_exclusive_group()
    check.add_argument("--check-archive", type=pathlib.Path, help="the archive of subset check")
    check.add_argument("--check-label-dir", type=pathlib.Path, help="the label directory of check")
    parser.add_argument("--out", type=pathlib.Path, help="the output directory")
    parser.add_argument(
        "--allow-local", action="store_true", help="analyze rows of --local runs, and say so"
    )
    args = parser.parse_args()
    if args.archive and args.out is None:
        parser.error("--archive requires --out, since the archive's extraction is removed")
    return args


def check_frozen_files(label_dir: pathlib.Path) -> None:
    """Print a warning if a label does not match its ``circuits.json`` and the working tree.

    The label's hash (``manifest.compute_label``) covers ``pins.json``, the frozen files of the
    experiment, the harness commit, and a bundle's requirements.  It is recomputed from the label's
    copy of ``circuits.json``, the working tree's other files, and the commit and bundle that the
    label's ``manifest.json`` records, and compared with the hash in the label's name.
    """
    from experiments.state_prep_certification.certification import experiment

    record_path = label_dir / "manifest.json"
    if not record_path.exists():
        print(f"warning: {label_dir.name} has no manifest.json; its frozen files are not checked")
        return
    record = json.loads(record_path.read_text())
    bundle = record.get("bundle")
    frozen = [
        label_dir / "circuits.json" if name.endswith("circuits.json") else HERE / name
        for name in experiment.FROZEN_FILES
    ]
    expected = manifest.compute_label(
        experiment.NAME,
        manifest.label_files(frozen),
        record["harness"]["commit"],
        bundle["requirements_sha256"] if bundle else None,
    )
    if not record["label"].startswith(expected):
        print(
            f"warning: the label {record['label']} does not match its circuits.json and the "
            f"working tree's config.json and pins.json ({expected}); its configuration may differ, "
            "and the analysis may be wrong"
        )


def _final_rows(label_dir: pathlib.Path) -> list[dict[str, Any]]:
    """The final row of each task of a label directory, with a note for each problem."""
    final, problems = store.final_rows(label_dir)
    for problem in problems:
        print(f"note: {label_dir.name}: {problem}")
    return list(final.values())


def _expected_tasks(label_dir: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Every task of a label, by key: pass 1 and, if written, pass 2."""
    if not (label_dir / task_store.TASKS_FILE).exists():
        print(f"note: {label_dir.name} has no {task_store.TASKS_FILE}; no task is missing")
        return {}
    return task_store.load_tasks(label_dir)


def _unpack(archive: pathlib.Path, directory: pathlib.Path) -> pathlib.Path:
    """Unpack an archive into a directory, and return its label directory."""
    directory.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(directory, filter="data")
    (label_dir,) = [path for path in directory.iterdir() if path.is_dir()]
    return label_dir


####################################################################################################
# collecting rows by circuit


def collect(
    rows: list[dict[str, Any]],
    circuits: dict[str, dict[str, Any]],
    expected: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """For each circuit: its record, and the rows of each arm (repeats in order).

    Each expected task without a row gets a row of status ``missing``, with a printed note.
    """
    present = {row["task_key"] for row in rows}
    missing = [task for key, task in sorted((expected or {}).items()) if key not in present]
    if missing:
        keys = [task["key"] for task in missing]
        more = f", and {len(keys) - 10} more" if len(keys) > 10 else ""
        print(f"note: {len(keys)} tasks have no row, reported as missing: {keys[:10]}{more}")
    per_circuit: dict[str, dict[str, Any]] = {}
    for row in rows + [_missing_row(task) for task in missing]:
        circuit_id = row["task_key"].split(".", 1)[1].rsplit(".r", 1)[0]
        entry = per_circuit.setdefault(circuit_id, {"circuit": circuits[circuit_id], "arms": {}})
        entry["arms"].setdefault(row["arm"], []).append(row)
    for entry in per_circuit.values():
        for arm_rows in entry["arms"].values():
            arm_rows.sort(key=lambda row: row["repeat"])
    return per_circuit


def listing_result(arm_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The counts of ``count_listing``: the result of its first run, or its partial results.

    A run that did not finish keeps q, q_h, and h, and Pi_(m-1), E_m, and Phi_m of each weight m
    that it completed, from its partial results.
    """
    if not arm_rows:
        return {}
    first = arm_rows[0]
    if first["status"] == "ok":
        result: dict[str, Any] = first["result"]
        return result
    partial: dict[str, Any] = {}
    for record in first.get("partial", []):
        if "m" not in record:
            partial |= record
            continue
        m = record["m"]
        partial.setdefault("Pi", {})[str(m - 1)] = record["Pi"]
        partial.setdefault("E", {})[str(m)] = record["E"]
        partial.setdefault("Phi", {})[str(m)] = record["Phi"]
    return partial


def summarize_certification(arm_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The status, T, timing completeness, peak memory, and result of a certification's runs.

    The status is the arm's ``_status``, and the result and partial results are those of the first
    run.  T is the median of the first run and its repeats, and None if the timing is incomplete: if
    the first run did not finish, if a repeat that it calls for did not finish or is missing, or if
    the row of any run has ``compiled_in_timing`` set.  ``repeats_not_ok`` lists the repeats that
    did not finish or are missing, and the runs whose rows have ``compiled_in_timing`` set, as in
    ``r1:timeout``, ``r2:missing``, or ``r0:compiled``.  The peak memory ``rss`` is the largest
    ``peak_rss_gib`` of the arm's runs, of any status, memouts above the cap included; each is the
    largest peak that the runner knows of its whole run, post-processing included (see
    ``common/runner.py``).
    """
    if not arm_rows:
        return {
            "status": "not run",
            "T": None,
            "timing_complete": False,
            "rss": None,
            "result": {},
            "partial": [],
            "post_status": None,
            "repeats_not_ok": [],
            "first_compiled": False,
        }
    first, repeats = arm_rows[0], arm_rows[1:]
    fast = first["status"] == "ok" and first["T"] is not None and first["T"] < REPEATS["below_s"]
    present = {row["repeat"] for row in repeats}
    not_ok = [f"r{row['repeat']}:{row['status']}" for row in repeats if row["status"] != "ok"]
    not_ok += [f"r{row['repeat']}:compiled" for row in arm_rows if row.get("compiled_in_timing")]
    if fast:
        not_ok += [f"r{r}:missing" for r in range(1, REPEATS["count"] + 1) if r not in present]
    complete = first["status"] == "ok" and not not_ok
    return {
        "status": _status(arm_rows),
        "T": statistics.median(row["T"] for row in arm_rows) if complete else None,
        "timing_complete": complete,
        "rss": max(
            (row["peak_rss_gib"] for row in arm_rows if row["peak_rss_gib"] is not None),
            default=None,
        ),
        "result": first["result"],
        "partial": first.get("partial", []),
        "post_status": first.get("post_status"),
        "repeats_not_ok": not_ok,
        "first_compiled": bool(first.get("compiled_in_timing")),
    }


def verdict(result: dict[str, Any]) -> str:
    """'certified', or 'violation at m'."""
    if result.get("certified"):
        return "certified"
    return f"violation at {result.get('violation_weight')}"


def _missing_row(task: dict[str, Any]) -> dict[str, Any]:
    """The row of an expected task that has none."""
    return {
        "task_key": task["key"],
        "arm": task["arm"],
        "repeat": task["repeat"],
        "status": "missing",
        "post_status": None,
        "T": None,
        "peak_rss_gib": None,
        "partial": [],
        "result": {},
    }


####################################################################################################
# summary.csv, per_weight.csv, families.csv, and listing_table.csv


def summarize(
    per_circuit: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The rows of summary.csv, one per circuit with its verdict, and those of per_weight.csv."""
    summary_rows = []
    weight_rows = []
    for circuit_id, entry in sorted(per_circuit.items()):
        circuit, arms = entry["circuit"], entry["arms"]
        bz = summarize_certification(arms.get("cert_bz"))
        cc = summarize_certification(arms.get("cert_cc"))
        brute = summarize_certification(arms.get("brute_bz"))
        listing = listing_result(arms.get("count_listing"))
        accepted = _first_result(arms.get("count_accepted"))
        finished = [run for run in (bz, cc) if run["status"] == "ok"]
        code = circuit.get("code") or [None, None, None]
        counts = _accepted_counts(accepted, arms.get("ground_truth"))
        agree_baseline = _agree_baseline(brute, finished, counts)
        agree_ground_truth = _agree_ground_truth(
            arms.get("ground_truth"), finished, listing, accepted
        )
        claims, contradictions = check_consistency(circuit, arms, listing, accepted)
        margin = analyze_applications.margin_fields(arms, claims)
        agree_prune = analyze_applications.agree_prune(bz, cc, arms)
        circuit_verdict = "disagreement" if contradictions else _circuit_verdict(bz, cc)
        summary_rows.append(
            {
                "id": circuit_id,
                "family": circuit["family"],
                "parent": circuit.get("parent"),
                "n": code[0] if code[0] is not None else circuit.get("cat_width"),
                "k": code[1],
                "d": code[2],
                "t": circuit["t"],
                "K": circuit["options"]["max_weight"],
                **{
                    key: listing.get(
                        key, accepted.get(key, circuit.get(CIRCUIT_FIELDS.get(key, key)))
                    )
                    for key in ("p", "N", "N0", "z", "r", "K_prime", "q", "q_h", "h")
                },
                "U": circuit.get("U"),
                "brute_within_cap": circuit.get("brute_force"),
                "verdict": circuit_verdict,
                "violation_weight": (
                    finished[0]["result"].get("violation_weight") if finished else None
                ),
                **{
                    field: value
                    for name, run in (("bz", bz), ("cc", cc))
                    for field, value in (
                        (f"verdict_{name}", _run_verdict(run)),
                        (f"T_{name}", run["T"]),
                        (f"timing_complete_{name}", run["timing_complete"]),
                        (f"rss_{name}", run["rss"]),
                        (f"post_status_{name}", run["post_status"]),
                        (f"repeats_not_ok_{name}", " ".join(run["repeats_not_ok"]) or None),
                    )
                },
                **{f"status_{arm}": _status(arms.get(arm)) for arm in REPORTED_ARMS},
                "status_brute_bz": OVER_CAP
                if circuit.get("brute_force") is False
                else _status(arms.get("brute_bz")),
                "T_brute_bz": brute["T"],
                "timing_complete_brute_bz": brute["timing_complete"],
                "rss_brute_bz": brute["rss"],
                "repeats_not_ok_brute_bz": " ".join(brute["repeats_not_ok"]) or None,
                "rows_bz": certifier_rows(bz),
                "rows_brute_bz": brute["result"].get("rows") if brute["status"] == "ok" else None,
                "agree_bz_cc": _agree(bz, cc),
                "agree_baseline": agree_baseline,
                "baseline_counts_match": _baseline_counts_match(brute["partial"], counts),
                "agree_ground_truth": agree_ground_truth,
                "ground_truth_counts_match": _ground_truth_counts_match(
                    arms.get("ground_truth"), listing, accepted
                ),
                **margin,
                "agree_prune": agree_prune,
                "contradictions": "; ".join(contradictions) or None,
            }
        )
        weight_rows += _per_weight(circuit_id, circuit, bz, cc, brute, listing, accepted)
    return summary_rows, weight_rows


def write_tables(
    summary_rows: list[dict[str, Any]], weight_rows: list[dict[str, Any]], out: pathlib.Path
) -> list[pathlib.Path]:
    """Write summary.csv, per_weight.csv, families.csv, and listing_table.csv; return the paths."""
    return [
        _write_csv(out / "summary.csv", summary_rows),
        _write_csv(out / "per_weight.csv", weight_rows),
        _write_csv(out / "families.csv", _family_rows(summary_rows)),
        _write_csv(out / "listing_table.csv", listing_table(summary_rows, weight_rows)),
    ]


def certifier_rows(run: dict[str, Any]) -> int | None:
    """The oracle rows of a finished certification: threshold-1 queries and the sum of Omega_m."""
    if run["status"] != "ok":
        return None
    result = run["result"]
    threshold_one = result["zero_flag_checks"]["calls"] + result["classification"]["calls"]
    return threshold_one + sum(batch["rows"] for batch in result["batches"].values())


def listing_table(
    summary_rows: list[dict[str, Any]], weight_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """The listing counts against all accepted combinations, per family and weight m >= 2.

    The ratios use A_m, or C(N_0, m) where A_m was not computed and z >= m, which makes them lower
    bounds (``ratio_is_lower_bound``); Omega_m / A_m uses only weights that ``cert_bz`` completed.
    A circuit whose A_m is 0, or was not computed while z < m, so that it may be 0, has no ratio;
    ``ratio_unknown`` counts these circuits.  The parameter medians are over the circuits with a
    ratio.  Circuits with a disagreement are left out.
    """
    circuits = {row["id"]: row for row in summary_rows}
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in weight_rows:
        circuit = circuits[row["id"]]
        if row["m"] >= 2 and row["Phi_m"] is not None and circuit["verdict"] != "disagreement":
            groups.setdefault((circuit["family"], row["m"]), []).append(row)
    table = []
    for (family, m), rows in sorted(groups.items()):
        phi, omega, parameters = [], [], []
        substituted = unknown = 0
        for row in rows:
            # z >= m ensures A_m > 0: m distinct unflagged mechanisms are an accepted combination
            denominator = row["A_m"]
            if denominator is None and (circuits[row["id"]]["z"] or 0) >= m:
                denominator = row["C_N0_m"]
            if not denominator:
                unknown += 1
                continue
            substituted += row["A_m"] is None
            phi.append(row["Phi_m"] / denominator)
            if row["Omega_m"] is not None and not row["Omega_partial"]:
                omega.append(row["Omega_m"] / denominator)
            parameters.append(circuits[row["id"]])
        table.append(
            {
                "family": family,
                "m": m,
                "circuits": len(phi),
                "ratio_unknown": unknown,
                "phi_over_A_median": statistics.median(phi) if phi else None,
                "phi_over_A_max": max(phi, default=None),
                "omega_circuits": len(omega),
                "omega_over_A_median": statistics.median(omega) if omega else None,
                "omega_over_A_max": max(omega, default=None),
                "C_in_place_of_A": substituted,
                "ratio_is_lower_bound": bool(substituted),
                **{
                    f"{name}_median": _median(values)
                    for name, values in (
                        ("h_over_N0", [_share(c["h"], c["N0"]) for c in parameters]),
                        ("z_over_N0", [_share(c["z"], c["N0"]) for c in parameters]),
                        ("q_h", [c["q_h"] for c in parameters]),
                        ("r", [c["r"] for c in parameters]),
                    )
                },
            }
        )
    return table


def _median(values: list[Any]) -> float | None:
    """The median of the known values, or None if there are none."""
    known = [value for value in values if value is not None]
    return statistics.median(known) if known else None


def _share(part: int | None, whole: int | None) -> float | None:
    """part / whole, or None if either is unknown or the whole is zero."""
    return None if part is None or not whole else part / whole


def _first_result(arm_rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    """The result of an arm's first run, empty if it did not run."""
    return arm_rows[0]["result"] if arm_rows else {}


def _status(arm_rows: list[dict[str, Any]] | None) -> str:
    """The status of an arm's first run, or 'not run'.

    A timeout whose row has ``compiled_in_timing`` set is ``timeout_compiled`` (see the module
    docstring).
    """
    if not arm_rows:
        return "not run"
    first = arm_rows[0]
    if first["status"] == "timeout" and first.get("compiled_in_timing"):
        return "timeout_compiled"
    return str(first["status"])


def _run_verdict(run: dict[str, Any]) -> str:
    """The verdict of a finished certification, or its status."""
    return verdict(run["result"]) if run["status"] == "ok" else run["status"]


def _circuit_verdict(bz: dict[str, Any], cc: dict[str, Any]) -> str:
    """BZ's verdict, or CC's where BZ did not finish; 'undecided' if neither finished.

    A circuit that neither arm finished is 'timeout_compiled', not 'undecided', if either arm's
    status is ``timeout_compiled`` (see the module docstring).  Whether the verdicts contradict
    each other is for ``check_consistency``.
    """
    verdicts = [verdict(run["result"]) for run in (bz, cc) if run["status"] == "ok"]
    if not verdicts:
        return (
            "timeout_compiled"
            if "timeout_compiled" in (bz["status"], cc["status"])
            else "undecided"
        )
    return verdicts[0]


def _agree(first: dict[str, Any], second: dict[str, Any]) -> bool | None:
    """Whether two finished certifications give the same verdict; None if one did not finish."""
    if first["status"] != "ok" or second["status"] != "ok":
        return None
    return verdict(first["result"]) == verdict(second["result"])


def _agree_baseline(
    brute: dict[str, Any], finished: list[dict[str, Any]], counts: dict[str, int]
) -> bool | None:
    """Whether brute_bz agrees with A_m and with the verdict of each finished certification.

    ``counts`` holds A_m by weight (``_accepted_counts``).  The counts are compared on the weights
    that brute_bz completed, where A_m is known, and a mismatch is a disagreement even if no
    certification finished.  Otherwise:

    1. If brute_bz and a certification finished, the result is whether their verdicts agree.
    2. Otherwise, if brute_bz recorded a violation before it was stopped and a certification
       finished, the result is whether every finished certification's smallest violation has the
       weight of brute_bz's smallest recorded violation, which is its smallest violation, since
       brute_bz checks the weights in order.
    3. Otherwise, a finished certification's violation at a weight that brute_bz completed without
       a violation is a disagreement.
    4. Otherwise, the result is that of the counts: True if they match, and None if none were
       compared.
    """
    counts_match = _baseline_counts_match(brute["partial"], counts)
    if counts_match is False:
        return False
    if brute["status"] == "ok" and finished:
        return all(verdict(brute["result"]) == verdict(run["result"]) for run in finished)
    records = [record for record in brute["partial"] if "m" in record]
    found = [record["m"] for record in records if record["violation"]]
    violations = [run["result"].get("violation_weight") for run in finished]
    if found and finished:
        return all(weight == min(found) for weight in violations)
    completed = [record["m"] for record in records if not record["violation"]]
    if any(weight is not None and weight <= max(completed, default=0) for weight in violations):
        return False
    return counts_match


def _baseline_counts_match(records: list[dict[str, Any]], counts: dict[str, int]) -> bool | None:
    """Whether a run of brute_bz enumerated as many combinations as A_m allows on each weight.

    ``records`` are the run's per-weight records, and ``counts`` holds A_m by weight
    (``_accepted_counts``).  A completed weight must have exactly A_m combinations, and a weight
    where the run found a violation, which stops its enumeration early, at most A_m.  None if no
    weight of the records has a known A_m.
    """
    compared = [
        record["combinations"] <= count if record["violation"] else record["combinations"] == count
        for record in records
        if "m" in record and (count := counts.get(str(record["m"]))) is not None
    ]
    return all(compared) if compared else None


def _accepted_counts(
    accepted: dict[str, Any], truth_rows: list[dict[str, Any]] | None
) -> dict[str, int]:
    """A_m by weight: of ``count_accepted``, or of the ground truth where that did not compute it.

    ``accepted`` is the result of ``count_accepted`` (``_first_result``), and ``truth_rows`` the
    rows of ``ground_truth``.
    """
    truth = (_ground_truth(truth_rows) or {}).get("A") or {}
    return truth | (accepted.get("A") or {})


def _agree_ground_truth(
    truth_rows: list[dict[str, Any]] | None,
    finished: list[dict[str, Any]],
    listing: dict[str, Any],
    accepted: dict[str, Any],
) -> bool | None:
    """Whether the ground truth agrees with the certifications and the counters.

    Every finished certification must have the ground truth's smallest violating weight, and the
    counters its counts (``_ground_truth_counts_match``).  None if nothing was compared.
    """
    truth = _ground_truth(truth_rows)
    if truth is None:
        return None
    compared = [
        run["result"].get("violation_weight") == truth["smallest_violation"] for run in finished
    ]
    counts = _ground_truth_counts_match(truth_rows, listing, accepted)
    if counts is not None:
        compared.append(counts)
    return all(compared) if compared else None


def _ground_truth_counts_match(
    truth_rows: list[dict[str, Any]] | None, listing: dict[str, Any], accepted: dict[str, Any]
) -> bool | None:
    """Whether the ground truth's Pi, E, Phi, and A equal the counters' on the weights of both.

    None if the ground truth has no result (``_ground_truth``) or no weight has both counts.
    """
    truth = _ground_truth(truth_rows)
    if truth is None:
        return None
    compared = []
    for key, counter in (("Pi", listing), ("E", listing), ("Phi", listing), ("A", accepted)):
        counts, truth_counts = counter.get(key) or {}, truth.get(key) or {}
        compared += [counts[m] == truth_counts[m] for m in counts.keys() & truth_counts.keys()]
    return all(compared) if compared else None


def _ground_truth(truth_rows: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    """The result of the first ground-truth run that has one, whatever its status, or None.

    A run that timed out or exceeded its memory cap after its call returned keeps its result
    (``common/runner.py``).
    """
    return next(
        (
            row["result"]
            for row in truth_rows or []
            if "smallest_violation" in (row.get("result") or {})
        ),
        None,
    )


def _per_weight(
    circuit_id: str,
    circuit: dict[str, Any],
    bz: dict[str, Any],
    cc: dict[str, Any],
    brute: dict[str, Any],
    listing: dict[str, Any],
    accepted: dict[str, Any],
) -> list[dict[str, Any]]:
    """One row per weight m = 1, ..., K of a circuit; skipped only on completed weights.

    The times of a first run whose row has ``compiled_in_timing`` set are left empty.
    """
    bz_weights = {record["m"]: record for record in bz["partial"] if "m" in record}
    cc_weights = {record["m"]: record for record in cc["partial"] if "m" in record}
    brute_weights = {record["m"]: record for record in brute["partial"] if "m" in record}
    max_weight = circuit["options"]["max_weight"]
    weight_limit = listing.get("K_prime", circuit.get("K_prime"))
    certified_bz = bz["status"] == "ok" and bool(bz["result"].get("certified"))
    rows = []
    for m in range(1, max_weight + 1):
        record = bz_weights.get(m, {})
        completed = bool(record) and not record.get("violation")
        phi = listing.get("Phi", {}).get(str(m))
        omega = record.get("rows")
        beyond = weight_limit is not None and m > weight_limit
        if beyond and "Phi" in listing:
            phi = 0
        if beyond and certified_bz:
            omega, completed = 0, True
        threshold_one = {}
        if m == 1 and bz["status"] == "ok":
            threshold_one = {
                "threshold1_zero_flag": bz["result"]["zero_flag_checks"]["calls"],
                "threshold1_flagged": bz["result"]["classification"]["calls"],
            }
        baseline = brute_weights.get(m, {})
        rows.append(
            {
                "id": circuit_id,
                "m": m,
                "A_m": (accepted.get("A") or {}).get(str(m)),
                "C_N0_m": (accepted.get("C") or {}).get(str(m), _binomial(circuit.get("N0"), m)),
                "naive_prefixes_m_minus_1": listing.get("Pi", {}).get(str(m - 1)),
                "E_m": listing.get("E", {}).get(str(m)),
                "Phi_m": phi,
                "Omega_m": omega,
                "Omega_partial": bool(record) and not completed,
                "threshold1_zero_flag": threshold_one.get("threshold1_zero_flag"),
                "threshold1_flagged": threshold_one.get("threshold1_flagged"),
                "skipped": phi - omega if completed and phi is not None else None,
                "T_m_bz": _first_run_time(bz, record.get("T_m")),
                "T_m_cc": _first_run_time(cc, cc_weights.get(m, {}).get("T_m")),
                "oracle_seconds_bz": _first_run_time(bz, record.get("oracle_seconds")),
                "brute_combinations": baseline.get("combinations"),
                "brute_rows": baseline.get("rows"),
                "brute_calls": baseline.get("calls"),
                "brute_distinct_syndromes": baseline.get("distinct_syndromes"),
                "brute_oracle_seconds": _first_run_time(brute, baseline.get("oracle_seconds")),
                "brute_completed": (not baseline["violation"]) if baseline else None,
            }
        )
    return rows


def _binomial(n: int | None, m: int) -> int | None:
    """C(n, m), or None if n is unknown."""
    return None if n is None else math.comb(n, m)


def _first_run_time(arm: dict[str, Any], seconds: float | None) -> float | None:
    """A time from an arm's first run, or None if that run's row has ``compiled_in_timing`` set."""
    return None if arm["first_compiled"] else seconds


def _family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per family."""
    table = []
    for family in sorted({row["family"] for row in rows}):
        members = [row for row in rows if row["family"] == family]
        kept = [row for row in members if row["verdict"] != "disagreement"]
        decided = [row for row in kept if row["verdict"] not in ("undecided", "timeout_compiled")]
        violations = [row for row in decided if row["verdict"] != "certified"]
        certified = [row for row in decided if row["verdict"] == "certified"]
        finished = {
            name: [
                row[f"T_{name}"]
                for row in kept
                if row[f"status_cert_{name}"] == "ok" and row[f"timing_complete_{name}"]
            ]
            for name in ("bz", "cc")
        }
        ratios = [
            row["T_brute_bz"] / row["T_bz"]
            for row in kept
            if row["T_brute_bz"] is not None and row["T_bz"]
        ]
        table.append(
            {
                "family": family,
                "circuits": len(members),
                "certified_at_t": len(certified)
                + sum(row["violation_weight"] == row["t"] + 1 for row in violations),
                "violations_at_most_t": sum(
                    row["violation_weight"] <= row["t"] for row in violations
                ),
                "certified_at_t_plus_1": len(certified),
                "violations_at_t_plus_1": sum(
                    row["violation_weight"] == row["t"] + 1 for row in violations
                ),
                "undecided": sum(row["verdict"] == "undecided" for row in members),
                "timeout_compiled": sum(row["verdict"] == "timeout_compiled" for row in members),
                "disagreements": sum(row["verdict"] == "disagreement" for row in members),
                **{
                    f"{status}_{arm}": sum(row[f"status_{arm}"] == status for row in members)
                    for arm in REPORTED_ARMS
                    for status in COUNTED_STATUSES
                },
                "largest_n_certified": max((row["n"] or 0 for row in certified), default=None),
                "largest_K_certified": max((row["K"] for row in certified), default=None),
                **{
                    field: value
                    for name, values in finished.items()
                    for field, value in (
                        (f"T_{name}_median", statistics.median(values) if values else None),
                        (f"T_{name}_max", max(values, default=None)),
                        (f"T_{name}_finished", len(values)),
                    )
                },
                "brute_not_run_over_cap": sum(
                    row["status_brute_bz"] == OVER_CAP for row in members
                ),
                "brute_finished": sum(row["status_brute_bz"] == "ok" for row in members),
                "brute_timed_out": sum(row["status_brute_bz"] == "timeout" for row in members),
                "brute_ratio_median": statistics.median(ratios) if ratios else None,
                "brute_ratio_circuits": len(ratios),
                "max_rss_gib": max(
                    (value for row in kept for value in (row["rss_bz"], row["rss_cc"]) if value),
                    default=None,
                ),
                "agreement_bz_cc": _agreement(members, "agree_bz_cc", ("cert_bz", "cert_cc")),
                "agreement_baseline": _agreement(members, "agree_baseline", ("brute_bz",)),
                "agreement_ground_truth": _agreement(
                    members, "agree_ground_truth", ("ground_truth",)
                ),
                **analyze_applications.margin_counts(members),
            }
        )
    return table


def _agreement(members: list[dict[str, Any]], key: str, arms: tuple[str, ...]) -> str:
    """'a/b' for a of b compared circuits that agree, or 'not run' if no arm has rows."""
    if all(row[f"status_{arm}"].startswith("not run") for row in members for arm in arms):
        return "not run"
    compared = [row[key] for row in members if row[key] is not None]
    return f"{sum(compared)}/{len(compared)}"


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> pathlib.Path:
    """Write rows to a CSV file, with the union of their keys as columns; return its path."""
    fields: list[str] = []
    for row in rows:
        fields += [key for key in row if key not in fields]
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields or ["empty"])
        writer.writeheader()
        writer.writerows(rows)
    return path


####################################################################################################
# shadow.csv


def shadow_table(
    per_circuit: dict[str, dict[str, Any]],
    verdicts: dict[str, str],
    calls_dir: pathlib.Path | None,
) -> list[dict[str, Any]]:
    """The rows of shadow.csv: both weight oracles on identical queries, by circuit and threshold.

    Each first run of ``shadow`` gives one row per threshold m of the fault search's queries, m = 1
    for the threshold-1 checks and classifications and m >= 2 for the batches: the calls, the rows
    that they submit, and the seconds of the BZ and the CC oracle on them.  A finished run
    gives the totals of its result (``instrument.py``), and a run that did not finish gives the
    sums of the calls that it wrote to its file in ``calls_dir`` before it stopped; ``status`` is
    the run's.  The file's last line, which a kill can cut off, is left out if it is not valid
    JSON, and any other such line raises a ValueError.  A run that wrote no calls, such as one
    killed before its first, has no row; ``summary.csv`` gives its status.  The runs whose rows
    have ``compiled_in_timing`` set, and the circuits with a disagreement, are left out.
    """
    table = []
    for circuit_id, entry in sorted(per_circuit.items()):
        runs = entry["arms"].get("shadow")
        if not runs or runs[0].get("compiled_in_timing") or verdicts[circuit_id] == "disagreement":
            continue
        run = runs[0]
        totals: dict[int, dict[str, Any]] = {}
        if run["status"] == "ok":
            calls = [
                {"m": int(key.split(":")[1])} | value
                for key, value in run["result"]["shadow"]["by_threshold"].items()
            ]
        else:
            calls = [
                call | {"calls": 1, "bz_seconds": call["bz"], "cc_seconds": call["cc"]}
                for call in _read_calls(calls_dir, run["task_key"])
            ]
        for call in calls:
            total = totals.setdefault(
                call["m"], {"calls": 0, "rows": 0, "bz_seconds": 0.0, "cc_seconds": 0.0}
            )
            for key in total:
                total[key] += call[key]
        table += [
            {"id": circuit_id, "family": entry["circuit"]["family"], "status": _status(runs)}
            | {"m": m}
            | total
            for m, total in sorted(totals.items())
        ]
    return table


def _read_calls(calls_dir: pathlib.Path | None, task_key: str) -> list[dict[str, Any]]:
    """The oracle calls that a shadow run wrote, without a last line cut off by a kill.

    Raises:
        ValueError: If a line other than the last is not valid JSON.
    """
    path = calls_dir / f"{task_key}.jsonl" if calls_dir else None
    if path is None or not path.exists():
        return []
    lines = path.read_text().splitlines()
    calls = []
    for number, line in enumerate(lines, start=1):
        try:
            calls.append(json.loads(line))
        except json.JSONDecodeError as error:
            if number < len(lines):
                raise ValueError(f"{path}: line {number} is not valid JSON") from error
    return calls


####################################################################################################
# consistency of exact results


def check_consistency(
    circuit: dict[str, Any],
    arms: dict[str, list[dict[str, Any]]],
    listing: dict[str, Any],
    accepted: dict[str, Any],
) -> tuple[dict[str, analyze_applications.Claim], list[str]]:
    """Every exact claim of a circuit's runs, and every contradiction between them.

    ``listing`` and ``accepted`` are the counts of ``count_listing`` and ``count_accepted``
    (``listing_result``, ``_first_result``).  The claims on the circuit's m* are those of
    ``analyze_applications.margin_claims``, which this returns: of every run of ``cert_bz``,
    ``cert_cc``, ``shadow``, ``brute_bz``, ``margin``, and ``ground_truth``, by its result and the
    fault weights that it completed or found a violation at, of each deletion task's parent at
    K = t, and of the final circuit of a deletion task that accepted no deletion, which is the
    parent.  These are the contradictions, each a disagreement:

    * two claims on m* that conflict (``analyze_applications.combine_claims``), such as verdicts or
      smallest violating weights that differ between runs, repeats included, or a violation at a
      fault weight that another run completed;
    * a ``shadow`` run that ended in ``ShadowDisagreement``, by its row's error, or by the error of
      the run that the runner discarded because it compiled (``discarded_run``), whose timing
      stays discarded;
    * the combinations that each run of ``brute_bz``, repeats included, enumerated on a weight,
      against A_m of ``count_accepted``, or of the ground truth where ``count_accepted`` did not
      compute it (``_accepted_counts``), which they must equal on a completed weight and not exceed
      on one where the run found a violation;
    * the ground truth's Pi, E, Phi, and A against the counters' on the weights of both, and its
      number of heavy mechanisms against h of ``count_listing``;
    * Phi_m of ``count_listing`` or of the ground truth above A_m (``_accepted_counts``), since each
      listed fault of weight m is an accepted combination, and Omega_m of a run of ``cert_bz``,
      ``cert_cc``, ``shadow``, or ``margin`` above Phi_m, of ``count_listing`` or else of the
      ground truth, since each row that the fault search submits is a listed fault, also on a
      weight where the run found a violation;
    * a counterexample whose exact residual weight does not exceed its fault weight, by its flag
      ``residual_exceeds_fault_weight`` or against the violating weight of its result, is below the
      certifier's lower bound on it, or, for CSS weight, is not the larger residual weight of its X
      and Z parts (``applications.sector_fields``);
    * a deletion task's final circuit that the CC oracle does not certify at K = t, or whose claim
      on its m* disagrees with itself, such as a final m* of at most t or one above a fault weight
      at which its search found a violation, or with that of another deletion task that ends at
      the same final circuit (``analyze_applications.final_contradictions``).
    """
    from experiments.state_prep_certification.certification import applications

    claims = analyze_applications.margin_claims(circuit, arms)
    contradictions = analyze_applications.combine_claims(claims)[2]

    # the shadow oracle's disagreements, which end their runs, also in a run that the runner
    # discarded because it compiled
    for row in arms.get("shadow") or []:
        source = f"shadow.r{row['repeat']}"
        runs = ((source, row), (f"{source} discarded run", row.get("discarded_run") or {}))
        for name, run in runs:
            if run.get("status") == "error" and "ShadowDisagreement" in (run.get("error") or ""):
                contradictions.append(f"{name}: BZ and CC answered a query differently")

    # the counts of every run of the baseline, and of the ground truth
    counts = _accepted_counts(accepted, arms.get("ground_truth"))
    for row in arms.get("brute_bz") or []:
        if _baseline_counts_match(row.get("partial") or [], counts) is False:
            contradictions.append(f"brute_bz.r{row['repeat']}: combinations against A_m")
    if _ground_truth_counts_match(arms.get("ground_truth"), listing, accepted) is False:
        contradictions.append("ground_truth: counts against count_listing or count_accepted")
    truth = _ground_truth(arms.get("ground_truth")) or {}
    heavy = truth.get("heavy")
    if heavy is not None and listing.get("h") is not None and sum(heavy) != listing["h"]:
        contradictions.append("ground_truth: heavy mechanisms against h of count_listing")

    # the listed faults against the accepted combinations, and the rows that each run submitted to
    # the oracle at a fault weight against the listed faults, also on a weight where it found a
    # violation, which stops the search early
    listed = {"count_listing": listing.get("Phi") or {}, "ground_truth": truth.get("Phi") or {}}
    for source, counted in listed.items():
        for m in sorted(counted.keys() & counts.keys(), key=int):
            if counted[m] > counts[m]:
                contradictions.append(f"{source}: Phi_{m} above A_{m}")
    phi = listed["ground_truth"] | listed["count_listing"]
    for arm in ("cert_bz", "cert_cc", "shadow", "margin"):
        for row in arms.get(arm) or []:
            for record in row.get("partial") or []:
                m = str(record.get("m"))
                if "rows" in record and m in phi and record["rows"] > phi[m]:
                    contradictions.append(f"{arm}.r{row['repeat']}: Omega_{m} above Phi_{m}")

    # each counterexample's exact residual weight, against its violating weight, its lower bound,
    # and, for CSS weight, the larger residual weight of its X and Z parts
    described = [
        (f"{arm}.r{row['repeat']}", row.get("result") or {}, "violation_weight")
        for arm in ("cert_bz", "cert_cc")
        for row in arms.get(arm) or []
    ]
    described += [
        (f"margin.r{row['repeat']}", row.get("result") or {}, "margin")
        for row in arms.get("margin") or []
    ]
    described += [
        (f"{arm} final", (row.get("result") or {}).get("final_margin") or {}, "margin")
        for arm in applications.PRUNE_ORDERS
        for row in arms.get(arm) or []
    ]
    for source, result, key in described:
        exact, bound = result.get("residual_weight"), result.get("residual_weight_lower_bound")
        weight = result.get(key)
        if result.get("residual_exceeds_fault_weight") is False or (
            exact is not None and isinstance(weight, int) and exact <= weight
        ):
            contradictions.append(f"{source}: residual weight does not exceed the fault weight")
        elif exact is not None and bound is not None and exact < bound:
            contradictions.append(f"{source}: residual weight below its lower bound")
        sectors = (result.get("sector_weight_x"), result.get("sector_weight_z"))
        if exact is not None and None not in sectors and exact != max(sectors):
            contradictions.append(f"{source}: residual weight against those of its X and Z parts")

    return claims, contradictions + analyze_applications.final_contradictions(circuit, arms)


####################################################################################################
# counterexamples


def write_counterexamples(
    per_circuit: dict[str, dict[str, Any]],
    verdicts: dict[str, str],
    circuits: dict[str, dict[str, Any]],
    out: pathlib.Path,
) -> pathlib.Path:
    """Write counterexamples.json, and return its path.

    It holds the counterexample of each first run of ``cert_bz`` and ``cert_cc`` that found one, and
    up to three example counterexamples, one for each group that has an eligible counterexample, by
    a fixed rule.  The counterexamples of the margins and of the deletions' final circuits are
    described in ``summary.csv``, ``margin_violations.csv``, and ``deletion_table.csv`` instead.
    ``verdicts`` holds each circuit's verdict, as in summary.csv.  A counterexample on a circuit
    with a disagreement is listed, marked as such, but is never an example.
    """
    counterexamples = []
    for circuit_id, entry in sorted(per_circuit.items()):
        for arm in ("cert_bz", "cert_cc"):
            run = summarize_certification(entry["arms"].get(arm))
            if run["status"] == "ok" and not run["result"].get("certified", True):
                counterexamples.append(
                    {
                        "id": circuit_id,
                        "family": entry["circuit"]["family"],
                        "parent_family": _parent_family(entry["circuit"], circuits),
                        "arm": arm,
                        "N0": entry["circuit"].get("N0"),
                        "disagreement": verdicts[circuit_id] == "disagreement",
                        "post_status": run["post_status"],
                        **{
                            key: run["result"].get(key)
                            for key in (
                                "violation_weight",
                                "residual_weight_lower_bound",
                                "residual_weight",
                                "certificate_text",
                            )
                        },
                    }
                )
    examples = {}
    for name, group in EXAMPLE_GROUPS.items():
        members = [
            counterexample
            for counterexample in counterexamples
            if counterexample["arm"] == "cert_bz"
            and not counterexample["disagreement"]
            and (counterexample["family"], counterexample["parent_family"]) in group
        ]
        if members:
            best = max(members, key=lambda e: (e["violation_weight"] or 0, -(e["N0"] or 0)))
            examples[name] = best["id"]
    path = out / "counterexamples.json"
    path.write_text(json.dumps({"examples": examples, "all": counterexamples}, indent=1) + "\n")
    return path


def _parent_family(circuit: dict[str, Any], circuits: dict[str, dict[str, Any]]) -> str | None:
    """The family of a weakened variant's parent, or None for another circuit."""
    parent = circuit.get("parent")
    return None if parent is None else circuits[parent]["family"]


if __name__ == "__main__":
    main()
