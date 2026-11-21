# SPDX-License-Identifier: Apache-2.0

"""Acceptance tests of the distance experiment.

    python acceptance.py [--local] [--only AT1 AT5 ...] [--results-dir R]

* **AT1:** both arms on fast anchors (bb72, pp110, the quantum Golay code, qr47, the five-qubit
  code concatenated with itself, the rotated surface code of distance 9) and on
  ``SurfaceCode(3, 5)`` as plain matrices return the reference distance (3 for the asymmetric
  code); the plain codes of Golay and qr47, which have H_X = H_Z, have ``_equal_distance_xz``, and
  the other CSS codes not.  Locally, runs are limited to ``LOCAL_LIMIT_S``, and arms that time out
  are reported rather than checked.
* **AT2:** ``q_bz_prior`` with a limit of 5 s on pp170 is a timeout with T >= 5 s and no process
  left, while a concurrent run on another code finishes unaffected, without ``compiled_in_timing``.
* **AT5:** regenerated codes match ``sample.json``, and ``prepare`` copies it into the label
  directory, where ``analyze.py`` reads it: changing golay's reference distance in the copy makes
  the analysis of a run on golay a disagreement.  The analysis refuses the rows of these local runs
  without ``--allow-local``, and with it, every output says how many it includes.
  ``compare_with_sample`` reports a sample in which one reference distance differs from that of
  ``sample.json``.  Each row's tuning is the shipped default, with ``BZ_PRIOR`` for ``q_bz_prior``.
* **AT6:** both arms agree on the AT1 codes and on the smallest drawn code of each family, and none
  fails; arms that time out are reported rather than compared.  Locally, runs are limited to
  ``LOCAL_LIMIT_S``.
* **AT9:** ``run_node.py --dry-run`` writes no task file without an approval that covers the
  ceiling, never rewrites its task files, and reports the ceiling that ``worst_case_s`` gives with
  the runner's retries (``tasks.ceiling_hours``).  The repeat policy gives two repeats exactly to
  first runs that finished in under 300 s.
* **AT10:** the analysis reads synthetic rows of every case of ``analyze.py`` (repeats that disagree
  or time out, arms that agree on a distance other than the reference, an arm without rows, a first
  run that ends in an error, a repeat that ends in a memout, both finished, only one arm finished,
  and both timed out), writes every output, classifies the codes, and marks the family medians as
  ``analyze.py``'s docstring says, with the number of these ratios, and how many of them are lower
  bounds and how many upper bounds.  Its rows with ``compiled_in_timing`` set are first runs and a
  timed-out repeat, some of them on codes that disagree or fail.  ``load_sample`` refuses a label
  directory without a copy of the sample.

GNU parallel's stop and resume (AT3) and job-slot pinning are tested locally by
``common/selftest.py`` with a stand-in for ``taskset``, and on Linux at the node.  The offline
install (AT8) is rehearsed with ``bundle.py``.  AT4 and AT7 are retired.  Each test prints PASS or
FAIL.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import json
import pathlib
import re
import shutil
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[2] / "distance_tuning"))

from experiments.state_prep_certification.common import acceptance, local_runs, store, tasks
from experiments.state_prep_certification.distances import arms, codes, experiment
from qldpc import codes as qldpc_codes
from qldpc.codes.distance import _tuning

ROOT = pathlib.Path(__file__).resolve().parents[3]
AT1_CODES = ("bb72", "pp110", "golay", "qr47", "five_five", "surface9")
EQUAL_SECTORS = ("golay", "qr47")
LOCAL_LIMIT_S = 60.0
# the prior tuning, written out here independently of config.json, which AT5 checks
EXPECTED_BZ_PRIOR = {
    "partition_max_coverage": 1,
    "overlapping_minimum_work": _tuning.MAX_VALUE,
    "completion_bound_minimum_layer_work": _tuning.MAX_VALUE,
    "completion_bound_max_table_bytes": 0,
    "automorphism_minimum_layer_work": _tuning.MAX_VALUE,
}


SUITE = acceptance.Suite("spc-dist-")
test = SUITE.test
run_all = SUITE.run_all


def _sample() -> dict[str, Any]:
    """The frozen sample."""
    sample: dict[str, Any] = json.loads(codes.SAMPLE.read_text())
    return sample


def _entries(names: tuple[str, ...]) -> list[dict[str, Any]]:
    """The sample's entries of named anchors."""
    anchors = {entry["name"]: entry for entry in _sample()["anchors"]}
    return [anchors[name] for name in names]


def _prepared(directory: pathlib.Path) -> pathlib.Path:
    """A label directory with the regenerated codes."""
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "codes").exists():
        experiment.prepare(directory)
    return directory


def _run_arms(
    directory: pathlib.Path,
    entries: list[dict[str, Any]],
    limit_s: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Run both arms on the entries, and return the rows by task key."""
    task_list = []
    for entry in entries:
        for arm in arms.ARMS:
            task = experiment._task(arm, entry, repeat=0)
            if limit_s is not None:
                task["limit_s"] = limit_s
            task_list.append(task)
    return local_runs.run_locally(directory, task_list)


####################################################################################################
# tests


@test
def at1_anchors(directory: pathlib.Path, node: bool) -> None:
    """Both arms return the reference distance on fast anchors, searching the same sectors.

    Both arms make the same code-method call on a plain code, so they search one sector exactly
    when the plain code has ``_equal_distance_xz``: on the codes with H_X = H_Z, and on no other.
    """
    label_dir = _prepared(directory)
    entries = _entries(AT1_CODES)
    asymmetric = qldpc_codes.SurfaceCode(3, 5)
    stored = codes.store_code(label_dir, asymmetric)
    entries.append({"name": "surface_3x5", "reference_d": 3, **stored})
    for entry in entries:
        arrays = codes.load_code(label_dir, entry["code_hash"])
        if "matrix" in arrays:
            continue
        plain = qldpc_codes.CSSCode(arrays["matrix_x"], arrays["matrix_z"], is_subsystem_code=False)
        equal = entry["name"] in EQUAL_SECTORS
        assert bool(plain._equal_distance_xz) == equal, entry["name"]
        assert entry["equal_distance_xz"] == equal, entry["name"]
    rows = _run_arms(label_dir, entries, limit_s=None if node else LOCAL_LIMIT_S)
    references = {entry["code_hash"][:12]: entry["reference_d"] for entry in entries}
    unfinished = []
    for key, row in rows.items():
        if row["status"] == "timeout" and not node:
            unfinished.append(key)
            continue
        assert row["status"] == "ok", (key, row["status"], row["error"])
        assert row["result"]["distance"] == references[key.split(".")[2]], (key, row)
    print(
        f"  {len(rows) - len(unfinished)} runs returned the reference distances; "
        f"not finished within the local limit: {unfinished or 'none'}"
    )


@test
def at2_timeout(directory: pathlib.Path, node: bool) -> None:
    """A run past 5 s is a timeout with no process left; a concurrent run is unaffected."""
    label_dir = _prepared(directory)
    (slow,) = _entries(("pp170",))
    (fast,) = _entries(("golay",))
    slow_task = dict(experiment._task("q_bz_prior", slow, 0), limit_s=5.0)
    fast_task = experiment._task("q_bz_on", fast, 0)
    tasks.write_pass1(label_dir, [slow_task, fast_task])
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(local_runs.RUN_TASK),
                "--label-dir",
                str(label_dir),
                "--local",
                task["key"],
            ]
        )
        for task in (slow_task, fast_task)
    ]
    assert [process.wait() for process in processes] == [0, 0]
    final, _ = store.final_rows(label_dir)
    slow_row, fast_row = final[slow_task["key"]], final[fast_task["key"]]
    assert slow_row["status"] == "timeout" and slow_row["T"] >= 5.0, slow_row
    assert slow_row["result"] == {} and "left after" not in (slow_row["error"] or ""), slow_row
    assert fast_row["status"] == "ok" and fast_row["result"]["distance"] == 7, fast_row
    assert not fast_row["compiled_in_timing"], fast_row["compiled_function"]


@test
def at5_sample_and_tuning(directory: pathlib.Path, node: bool) -> None:
    """Codes match the sample, the analysis reads the run's copy, and rows hold shipped tunings."""

    label_dir = directory
    label_dir.mkdir(parents=True, exist_ok=True)
    experiment.prepare(label_dir)
    assert (label_dir / "sample.json").read_bytes() == codes.SAMPLE.read_bytes()
    changed = copy.deepcopy(_sample())
    next(entry for entry in changed["anchors"] if entry["name"] == "qr79")["reference_d"] = 13
    assert codes.compare_with_sample(changed) == ["anchors: qr79 has another reference_d"]

    defaults = dataclasses.asdict(_tuning.SearchTuning())
    prior = dict(defaults, **EXPECTED_BZ_PRIOR)
    rows = _run_arms(label_dir, _entries(("golay",)))
    for row in rows.values():
        expected = prior if row["arm"] == "q_bz_prior" else defaults
        assert row["result"]["tuning"] == expected, row["arm"]

    # the analysis reads the run's copy of the sample, not code_data/sample.json: with golay's
    # reference distance changed in the copy, golay's arms disagree with it
    copy_of_sample = copy.deepcopy(_sample())
    next(e for e in copy_of_sample["anchors"] if e["name"] == "golay")["reference_d"] = 8
    (label_dir / "sample.json").write_text(json.dumps(copy_of_sample))
    out = label_dir / "analysis"
    script = pathlib.Path(__file__).with_name("analyze.py")
    command = [sys.executable, str(script), "--label-dir", str(label_dir), "--out", str(out)]
    # the rows are of --local runs, which the analysis refuses without --allow-local
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "--allow-local" in result.stderr, result.stderr
    result = subprocess.run(
        command + ["--allow-local"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    with open(out / "families.csv") as file:
        (anchors,) = [row for row in csv.DictReader(file) if row["family"] == "anchors"]
    assert (anchors["codes"], anchors["disagreement"]) == ("1", "1"), anchors

    # every output says how many rows of local runs it includes
    for path in out.iterdir():
        if path.suffix == ".csv":
            with open(path) as file:
                reader = csv.DictReader(file)
                assert "local_rows" in (reader.fieldnames or []), path.name
                assert all(row["local_rows"] == "2" for row in reader), path.name
        elif path.suffix == ".json":
            assert json.loads(path.read_text())["local_rows"] == 2, path.name
        else:
            assert path.name == "distances.pdf", path.name


@test
def at6_agreement(directory: pathlib.Path, node: bool) -> None:
    """Both arms agree on AT1's codes and a small code of each family."""
    label_dir = _prepared(directory)
    sample = _sample()
    smallest: dict[str, dict[str, Any]] = {}
    for entry in sample["codes"]:
        if entry["family"] not in smallest or entry["n"] < smallest[entry["family"]]["n"]:
            smallest[entry["family"]] = entry
    entries = _entries(AT1_CODES) + list(smallest.values())
    rows = _run_arms(label_dir, entries, limit_s=None if node else LOCAL_LIMIT_S)
    by_code: dict[str, dict[str, Any]] = {}
    for key, row in rows.items():
        by_code.setdefault(key.split(".")[2], {})[row["arm"]] = row
    timeouts = [key for key, row in rows.items() if row["status"] == "timeout"]
    failures = [
        (key, row["status"]) for key, row in rows.items() if row["status"] not in ("ok", "timeout")
    ]
    assert not failures, failures
    for code_hash, code_rows in by_code.items():
        distances = {
            arm: row["result"]["distance"]
            for arm, row in code_rows.items()
            if row["status"] == "ok"
        }
        assert len(set(distances.values())) <= 1, (code_hash, distances)
    print(f"  {len(rows)} runs; timed out within the limit: {timeouts or 'none'}")


@test
def at9_task_lists(directory: pathlib.Path, node: bool) -> None:
    """Task files need a sufficient approval and are never rewritten; the ceiling and repeats."""
    shutil.rmtree(directory, ignore_errors=True)
    command = [
        sys.executable,
        str(ROOT / "experiments" / "state_prep_certification" / "run_node.py"),
        "distances",
        "--local",
        "--dry-run",
        "--subset",
        "smoke",
        "--results-dir",
        str(directory),
    ]
    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode != 0 and "No task file was written" in refused.stderr, refused
    assert not list(directory.rglob("tasks.json"))
    low = subprocess.run(
        command + ["--approved-cpu-hours", "0.5"], capture_output=True, text=True, check=False
    )
    assert low.returncode != 0 and not list(directory.rglob("tasks.json")), low.stderr
    approved = subprocess.run(
        command + ["--approved-cpu-hours", "100"], check=True, capture_output=True, text=True
    )
    task_files = sorted(directory.rglob("tasks-pass1.txt")) + sorted(directory.rglob("tasks.json"))
    assert len(task_files) == 2, task_files
    before = [(path.read_bytes(), path.stat().st_mtime_ns) for path in task_files]
    subprocess.run(command + ["--approved-cpu-hours", "100"], check=True, capture_output=True)
    assert [(path.read_bytes(), path.stat().st_mtime_ns) for path in task_files] == before

    # the ceiling, by independent arithmetic: start-up cap, limit (or the repeat threshold for a
    # repeated run's first run), result window, and grace, for each run; then, for the runner's
    # retries, twice that and one more run at the first run's limit
    task_cost = (120 + 300 + 30 + 30) + 2 * (120 + 1800 + 30 + 30)
    run_cost = 120 + 1800 + 30 + 30  # one run at the first run's limit
    (golay,) = _entries(("golay",))
    for arm in arms.ARMS:
        assert experiment.worst_case_s(experiment._task(arm, golay, 0)) == task_cost
    # the smoke subset: four anchors, each with both arms
    hours = 8 * (run_cost + 2 * task_cost) / 3600
    match = re.search(r"ceiling ([\d,.]+) CPU-hours", approved.stdout)
    assert match and float(match.group(1).replace(",", "")) == round(hours, 1), approved.stdout

    # the repeat policy: two repeats of a first run that finished in under 300 s, and no others
    task = experiment._task("q_bz_on", golay, 0)
    for status, seconds, expected in (
        ("ok", 299.9, 2),
        ("ok", 300.0, 0),
        ("timeout", 1800.0, 0),
        ("error", 1.0, 0),
    ):
        repeats = experiment.repeat_tasks(task, {"status": status, "T": seconds})
        assert len(repeats) == expected, (status, seconds, repeats)
        for number, repeat in enumerate(repeats, 1):
            assert repeat["key"] == task["key"][:-1] + str(number), repeat["key"]
            assert repeat["limit_s"] == task["limit_s"] and repeat["repeat"] == number


@test
def at10_analysis(directory: pathlib.Path, node: bool) -> None:
    """The analysis classifies synthetic rows of every case, and needs the run's sample.json."""
    from experiments.state_prep_certification.distances import analyze

    sample = _sample()
    golay, bb72, qr47, qr71, qr79, pp110, surface9 = _entries(
        ("golay", "bb72", "qr47", "qr71", "qr79", "pp110", "surface9")
    )
    first, second, third, fourth, fifth = [e for e in sample["codes"] if e["family"] == "hgp"][:5]
    (gb,) = [e for e in sample["codes"] if e["family"] == "gb"][:1]
    lp_prior, lp_on = [e for e in sample["codes"] if e["family"] == "lp"][:2]
    rows = []

    def add(
        entry: dict[str, Any],
        arm: str,
        repeat: int,
        status: str,
        T: float,
        d: Any,
        compiled: bool = False,
    ) -> None:
        rows.append(
            {
                "task_key": f"{arm}.timed.{entry['code_hash'][:12]}.r{repeat}",
                "arm": arm,
                "mode": "timed",
                "repeat": repeat,
                "status": status,
                "T": T,
                "limit_s": 1800.0,
                "result": {"distance": d} if status == "ok" else {},
                "compiled_in_timing": compiled,
            }
        )

    # golay: the on arm's repeats disagree; bb72: the prior arm times out on its repeats
    for repeat, d in enumerate((7, 7, 8)):
        add(golay, "q_bz_on", repeat, "ok", 1.0, d)
        add(golay, "q_bz_prior", repeat, "ok", 2.0, 7)
    add(bb72, "q_bz_on", 0, "ok", 10.0, 6)
    add(bb72, "q_bz_prior", 0, "ok", 200.0, 6)
    add(bb72, "q_bz_prior", 1, "timeout", 1800.0, None)
    add(bb72, "q_bz_prior", 2, "timeout", 1800.0, None)
    for repeat in (1, 2):
        add(bb72, "q_bz_on", repeat, "ok", 10.0, 6)
    # rows with compiled_in_timing set (rows of retries that compiled too): pp110's first run of the
    # on arm, and surface9's timed-out repeat of the on arm; qr79's and third's flagged rows are
    # below
    for repeat in (0, 1, 2):
        add(pp110, "q_bz_on", repeat, "ok", 10.0, 12, compiled=repeat == 0)
        add(pp110, "q_bz_prior", repeat, "ok", 20.0, 12)
        add(surface9, "q_bz_prior", repeat, "ok", 3.0, 9)
    add(surface9, "q_bz_on", 0, "ok", 1.0, 9)
    add(surface9, "q_bz_on", 1, "ok", 2.0, 9)
    add(surface9, "q_bz_on", 2, "timeout", 1800.0, None, compiled=True)
    # qr47: the prior arm has no row; qr71: only the prior arm finishes; qr79: both arms agree on a
    # distance other than the reference, and the on arm has a flagged first run and a memout repeat
    add(qr47, "q_bz_on", 0, "ok", 5.0, 11)
    add(qr71, "q_bz_on", 0, "timeout", 1800.0, None)
    add(qr71, "q_bz_prior", 0, "ok", 900.0, 11)
    add(qr79, "q_bz_on", 0, "ok", 900.0, 13, compiled=True)
    add(qr79, "q_bz_on", 1, "memout", 900.0, None)
    add(qr79, "q_bz_prior", 0, "ok", 900.0, 13)
    # two hgp codes that fail: a first run ends in an error, and a repeat ends in a memout.  Third's
    # prior arm has a flagged first run; the code stays a failure.
    add(third, "q_bz_on", 0, "error", 3.0, None)
    add(third, "q_bz_prior", 0, "ok", 30.0, 4, compiled=True)
    add(fourth, "q_bz_on", 0, "ok", 20.0, 4)
    add(fourth, "q_bz_on", 1, "memout", 40.0, None)
    add(fourth, "q_bz_on", 2, "ok", 30.0, 4)
    add(fourth, "q_bz_prior", 0, "ok", 400.0, 4)
    # three hgp codes: only the on arm finishes; both time out; and both finish equally fast
    add(first, "q_bz_on", 0, "ok", 600.0, 4)
    add(first, "q_bz_prior", 0, "timeout", 1800.0, None)
    add(second, "q_bz_on", 0, "timeout", 1800.0, None)
    add(second, "q_bz_prior", 0, "timeout", 1800.0, None)
    add(fifth, "q_bz_on", 0, "ok", 400.0, 4)
    add(fifth, "q_bz_prior", 0, "ok", 400.0, 4)
    # gb: only the prior arm finishes, an inconclusive upper bound; lp: an upper bound and an
    # inconclusive lower bound, so that the family has both kinds of bound
    for entry, finished, T in (
        (gb, "q_bz_prior", 1600.0),
        (lp_prior, "q_bz_prior", 900.0),
        (lp_on, "q_bz_on", 1500.0),
    ):
        add(entry, finished, 0, "ok", T, 5)
        add(entry, ({"q_bz_on", "q_bz_prior"} - {finished}).pop(), 0, "timeout", 1800.0, None)
    expected = [
        experiment._task(arm, entry, repeat=0)
        for entry in (golay, bb72, qr47, qr71, qr79, pp110, surface9)
        + (first, second, third, fourth, fifth, gb, lp_prior, lp_on)
        for arm in ("q_bz_on", "q_bz_prior")
    ]
    records = analyze.summarize(rows, sample, expected)
    out = directory / "analysis"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    analyze.write_outputs(records, sample, out)
    by_name = {(r["name"], r["arm"]): r for r in records}

    def ratio(entry_name: str) -> dict[str, Any]:
        return analyze.ratio(by_name[(entry_name, "q_bz_on")], by_name[(entry_name, "q_bz_prior")])

    assert ratio("golay")["case"] == "disagreement"
    prior = by_name[("bb72", "q_bz_prior")]
    assert prior["T"] == 1800.0 and prior["repeats_unfinished"] == 2, prior
    assert analyze.classify(ratio("bb72")) == "faster"
    assert by_name[("qr47", "q_bz_prior")]["status"] == "missing"
    assert ratio("qr47")["case"] == "failure"
    only_prior = ratio("qr71")
    assert (only_prior["case"], only_prior["bound"], only_prior["value"]) == (
        "only prior",
        "<=",
        0.5,
    )
    assert analyze.classify(only_prior) == "slower"
    assert ratio("qr79")["case"] == "disagreement"
    only_on = ratio(first["name"])
    assert (only_on["case"], only_on["bound"], only_on["value"]) == ("only on", ">=", 3.0)
    assert analyze.classify(only_on) == "faster"
    assert ratio(second["name"])["case"] == "both > L"
    for entry in (third, fourth):
        assert ratio(entry["name"])["case"] == "failure", entry["name"]
    # fourth's first run finished, but a repeat ended in a memout, so T is the first run's time
    assert by_name[(fourth["name"], "q_bz_on")]["T"] == 20.0
    assert analyze.classify(ratio(fifth["name"])) == "similar"
    # lp_on's lower bound of 1800 / 1500 = 1.2 does not exceed 1.2, and gb's upper bound of
    # 1600 / 1800 is not below 1 / 1.2
    assert analyze.classify(ratio(lp_on["name"])) == "inconclusive"
    assert analyze.classify(ratio(gb["name"])) == "inconclusive"
    outputs = ("summary.csv", "families.csv", "anchors.csv", "distances.pdf", "figure_notes.json")
    for name in outputs:
        assert (out / name).exists(), name
    with open(out / "families.csv") as file:
        families = {row["family"]: row for row in csv.DictReader(file)}
    anchors = families["anchors"]
    assert (anchors["codes"], anchors["disagreement"], anchors["failure"]) == ("7", "2", "1")
    assert anchors["incomplete timing"] == "1", anchors
    hgp = families["hgp"]
    assert (hgp["both > L"], hgp["failure"]) == ("1", "2"), hgp
    # each family's median, the number of its ratios, and how many are lower bounds and how many
    # upper bounds; a family with both kinds of bound has no median, and the anchors have none
    medians = {
        name: tuple(families[name][key] for key in
                    ("median_R_T", "ratios_in_median", "lower_bounds", "upper_bounds"))
        for name in ("hgp", "gb", "lp", "anchors")
    }  # fmt: skip
    assert medians == {
        "hgp": (">=2", "2", "1", "0"),
        "gb": ("<=0.889", "1", "0", "1"),
        "lp": ("-", "2", "1", "1"),
        "anchors": ("-", "-", "-", "-"),
    }, medians
    # distances.pdf omits the disagreements, failures, and incomplete timings: golay, qr79, qr47,
    # third, fourth, and pp110
    assert json.loads((out / "figure_notes.json").read_text()) == {"T": 6}

    # pp110's flagged first run leaves R_T incomplete and its T cell "compiled"; surface9's flagged
    # timed-out repeat is left out of the median and of repeats_unfinished
    on = by_name[("pp110", "q_bz_on")]
    assert on["T"] is None and on["runs_compiled"] == 1, on
    assert ratio("pp110")["case"] == "incomplete timing"
    on = by_name[("surface9", "q_bz_on")]
    assert (on["T"], on["repeats"], on["repeats_unfinished"], on["runs_compiled"]) == (1.5, 2, 0, 1)
    with open(out / "anchors.csv") as file:
        cells = {row["name"]: row for row in csv.DictReader(file)}
    assert cells["pp110"]["q_bz_on_T"] == "compiled", cells["pp110"]
    # the T cell of a failed arm with a flagged first run shows the failure: qr79's memout repeat
    assert cells["qr79"]["q_bz_on_T"] == "memout", cells["qr79"]

    # a label directory without a copy of the sample, such as one from a run made before runs kept
    # a copy, is refused
    try:
        analyze.load_sample(directory)
    except ValueError:
        pass
    else:
        raise AssertionError("load_sample accepted a label directory without sample.json")


if __name__ == "__main__":
    SUITE.main(__doc__)
