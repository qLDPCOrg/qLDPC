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
* **AT5:** regenerated codes match ``sample.json``; ``compare_with_sample`` reports a sample whose
  reference distance differs from ``sample.json``; each row's tuning is the shipped default, with
  ``BZ_PRIOR`` for ``q_bz_prior``.
* **AT6:** both arms, timed and (with work counters) counting, agree on the AT1 codes and on the
  smallest drawn code of each family, and none fails; arms that time out are reported rather than
  compared.  Locally, runs are limited to ``LOCAL_LIMIT_S``.
* **AT7:** completed counting runs give identical counts.  It needs qLDPC's opt-in work counters;
  without them it is skipped locally and fails on the node, so that the node never runs the
  experiment without work counts.
* **AT9:** ``run_node.py --dry-run`` writes no task file without an approval that covers the
  ceiling, never rewrites its task files, and reports the ceiling that ``worst_case_s`` gives with
  the runner's retries (``tasks.ceiling_hours``).  The repeat policy gives two repeats exactly to
  timed runs that finished in under 300 s.
* **AT10:** the analysis reads synthetic rows of every case of ``analyze.py`` (repeats that
  disagree or time out, arms that agree on a distance other than the reference, an arm without
  rows, a first run and a counting run that end in an error, a repeat that ends in a memout, only
  one arm finished, both timed out, complete, partial, and missing work counts), writes every
  output, classifies them and marks the family medians as its docstring says, and refuses a work
  count that it cannot read.  Its rows with ``compiled_in_timing`` set are first timed runs, a
  timed-out repeat, a finished counting run, and counting timeouts, some of them on codes that
  disagree or fail.

GNU parallel's stop and resume (AT3) and job-slot pinning are tested locally by
``common/selftest.py`` with a stand-in for ``taskset``, and on Linux at the node.  The offline
install (AT8) is rehearsed with ``bundle.py``.  Each test prints PASS, FAIL, or SKIP.
"""

from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[2] / "distance_tuning"))

from experiments.state_prep_certification.common import local_runs, store, tasks
from experiments.state_prep_certification.distances import arms, codes, experiment
from qldpc import codes as qldpc_codes
from qldpc.codes.distance import _tuning

ROOT = pathlib.Path(__file__).resolve().parents[3]
TESTS: dict[str, Callable[[pathlib.Path, bool], None]] = {}
AT1_CODES = ("bb72", "pp110", "golay", "qr47", "five_five", "surface9")
EQUAL_SECTORS = ("golay", "qr47")
LOCAL_LIMIT_S = 60.0
# the prior tuning, written out here independently of config.json, which AT5 checks
PLAN_BZ_PRIOR = {
    "partition_max_coverage": 1,
    "overlapping_minimum_work": _tuning.MAX_VALUE,
    "lp_minimum_work": _tuning.MAX_VALUE,
    "completion_bound_minimum_layer_work": _tuning.MAX_VALUE,
    "completion_bound_max_table_bytes": 0,
    "automorphism_minimum_layer_work": _tuning.MAX_VALUE,
}


class Skipped(Exception):
    """A test that cannot run in this setting, with the reason."""


def main() -> None:
    """Run the selected tests."""
    args = parse_args()
    results_dir = args.results_dir or pathlib.Path(tempfile.mkdtemp(prefix="spc-dist-"))
    passed = run_all(node=not args.local, results_dir=results_dir, only=args.only)
    sys.exit(0 if passed else 1)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--local", action="store_true", help="the workstation's tests")
    parser.add_argument("--only", nargs="*", help="run only these tests, such as AT1")
    parser.add_argument("--results-dir", type=pathlib.Path, help="where the tests' runs go")
    return parser.parse_args()


def run_all(node: bool, results_dir: pathlib.Path, only: list[str] | None = None) -> bool:
    """Run the tests, print PASS, FAIL, or SKIP for each, and return whether none failed.

    On the node, a skipped test counts as a failure, since every test must pass there.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, test in TESTS.items():
        if only and name not in only:
            continue
        start = time.monotonic()
        try:
            test(results_dir / name, node)
            print(f"PASS {name} ({time.monotonic() - start:.1f} s)", flush=True)
        except Skipped as reason:
            print(f"SKIP {name}: {reason}", flush=True)
            if node:
                failed.append(name)
        except (
            ArithmeticError,
            AssertionError,
            AttributeError,
            ImportError,
            LookupError,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
            SystemExit,
            TypeError,
            ValueError,
        ):
            failed.append(name)
            print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
    print(f"{len(failed)} failed: {failed}" if failed else "none failed", flush=True)
    return not failed


def test(function: Callable[[pathlib.Path, bool], None]) -> Callable[[pathlib.Path, bool], None]:
    """Register a test by the ID at the start of its name, as in ``at1_...`` for AT1."""
    TESTS[function.__name__.split("_")[0].upper()] = function
    return function


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
    modes: tuple[str, ...] = ("timed",),
) -> dict[str, dict[str, Any]]:
    """Run both arms on the entries in the given modes, and return the rows by task key."""
    task_list = []
    for entry in entries:
        for arm in arms.ARMS:
            for mode in modes:
                task = experiment._task(arm, mode, entry, repeat=0)
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
    slow_task = dict(experiment._task("q_bz_prior", "timed", slow, 0), limit_s=5.0)
    fast_task = experiment._task("q_bz_on", "timed", fast, 0)
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
    """Regenerated codes match the sample; rows record the arms' tunings."""
    label_dir = directory
    label_dir.mkdir(parents=True, exist_ok=True)
    problems = codes.compare_with_sample(codes.generate(label_dir))
    assert not problems, problems
    changed = copy.deepcopy(_sample())
    next(entry for entry in changed["anchors"] if entry["name"] == "qr79")["reference_d"] = 13
    assert codes.compare_with_sample(changed) == ["anchors: qr79 has another reference_d"]
    defaults = dataclasses.asdict(_tuning.SearchTuning())
    prior = dict(defaults, **PLAN_BZ_PRIOR)
    rows = _run_arms(label_dir, _entries(("golay",)))
    for row in rows.values():
        expected = prior if row["arm"] == "q_bz_prior" else defaults
        assert row["result"]["tuning"] == expected, row["arm"]


@test
def at6_agreement(directory: pathlib.Path, node: bool) -> None:
    """Both arms agree, timed and counting, on AT1's codes and a small code of each family.

    Counting runs are made only if the installed qLDPC has work counters, as the task list does.
    """
    label_dir = _prepared(directory)
    sample = _sample()
    smallest: dict[str, dict[str, Any]] = {}
    for entry in sample["codes"]:
        if entry["family"] not in smallest or entry["n"] < smallest[entry["family"]]["n"]:
            smallest[entry["family"]] = entry
    entries = _entries(AT1_CODES) + list(smallest.values())
    modes = ("timed", "count") if arms.counters_available() else ("timed",)
    rows = _run_arms(label_dir, entries, limit_s=None if node else LOCAL_LIMIT_S, modes=modes)
    by_code: dict[str, dict[str, Any]] = {}
    for key, row in rows.items():
        by_code.setdefault(key.split(".")[2], {})[f"{row['arm']}.{row['mode']}"] = row
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
def at7_counters(directory: pathlib.Path, node: bool) -> None:
    """Completed counting runs give identical counts (needs qLDPC's counters)."""
    if not arms.counters_available():
        raise Skipped("the installed qLDPC has no opt-in work counters")
    label_dir = _prepared(directory)
    entries = _entries(("bb72", "pp110"))
    task_list = []
    for entry in entries:
        for arm in arms.ARMS:
            for repeat in (0, 1):
                task_list.append(experiment._task(arm, "count", entry, repeat))
    rows = local_runs.run_locally(label_dir, task_list)
    for arm in arms.ARMS:
        for entry in entries:
            pair = [rows[f"{arm}.count.{entry['code_hash'][:12]}.r{r}"] for r in (0, 1)]
            assert pair[0]["result"]["work"] == pair[1]["result"]["work"], (arm, entry["name"])


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
    timed_cost = (120 + 300 + 30 + 30) + 2 * (120 + 1800 + 30 + 30)
    count_cost = 120 + 1800 + 30 + 30
    run_cost = 120 + 1800 + 30 + 30  # S: one run at the first run's limit, in either mode
    (golay,) = _entries(("golay",))
    for mode, cost in (("timed", timed_cost), ("count", count_cost)):
        for arm in arms.ARMS:
            assert experiment.worst_case_s(experiment._task(arm, mode, golay, 0)) == cost
    smoke = experiment.tasks(directory, "smoke")
    hours = (
        sum(
            run_cost + 2 * (timed_cost if task["mode"] == "timed" else count_cost) for task in smoke
        )
        / 3600
    )
    match = re.search(r"ceiling ([\d,.]+) CPU-hours", approved.stdout)
    assert match and float(match.group(1).replace(",", "")) == round(hours, 1), approved.stdout

    # the repeat policy: two repeats of a timed run that finished in under 300 s, and no others
    task = experiment._task("q_bz_on", "timed", golay, 0)
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
    counting = experiment._task("q_bz_on", "count", golay, 0)
    assert experiment.repeat_tasks(counting, {"status": "ok", "T": 1.0}) == []


@test
def at10_analysis(directory: pathlib.Path, node: bool) -> None:
    """The analysis reads synthetic rows of every case without failing, and classifies them."""
    from experiments.state_prep_certification.distances import analyze

    sample = _sample()
    golay, bb72, qr47, qr71, qr79, pp110, surface9 = _entries(
        ("golay", "bb72", "qr47", "qr71", "qr79", "pp110", "surface9")
    )
    first, second, third, fourth, fifth = [e for e in sample["codes"] if e["family"] == "hgp"][:5]
    rows = []

    def add(
        entry: dict[str, Any],
        arm: str,
        repeat: int,
        status: str,
        T: float,
        d: Any,
        mode: str = "timed",
        work: Any = None,
        partial: tuple[Any, ...] = (),
        compiled: bool = False,
    ) -> None:
        result = {"distance": d, "work": work} if status == "ok" else {}
        rows.append(
            {
                "task_key": f"{arm}.{mode}.{entry['code_hash'][:12]}.r{repeat}",
                "arm": arm,
                "mode": mode,
                "repeat": repeat,
                "status": status,
                "T": T,
                "limit_s": 1800.0,
                "partial": [{"work": count} for count in partial],
                "result": result,
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
    # rows with compiled_in_timing set (rows of retries that compiled too): pp110's first timed run
    # and finished counting run of the on arm, and surface9's timed-out repeat and timed-out
    # counting run of the on arm, which reports a partial count that gives no W bound; qr79's,
    # third's, and fifth's flagged rows are below
    for repeat in (0, 1, 2):
        add(pp110, "q_bz_on", repeat, "ok", 10.0, 12, compiled=repeat == 0)
        add(pp110, "q_bz_prior", repeat, "ok", 20.0, 12)
        add(surface9, "q_bz_prior", repeat, "ok", 3.0, 9)
    add(surface9, "q_bz_on", 0, "ok", 1.0, 9)
    add(surface9, "q_bz_on", 1, "ok", 2.0, 9)
    add(surface9, "q_bz_on", 2, "timeout", 1800.0, None, compiled=True)
    for entry, d in ((pp110, 12), (surface9, 9)):
        add(entry, "q_bz_prior", 0, "ok", 20.0, d, mode="count", work={"total": 100})
    add(pp110, "q_bz_on", 0, "ok", 10.0, 12, "count", {"total": 50}, compiled=True)
    add(surface9, "q_bz_on", 0, "timeout", 1800.0, None, "count", None, ({"total": 10},), True)
    # qr47: the prior arm has no row; qr71: only the prior arm finishes; qr79: both arms agree on a
    # distance other than the reference, and the on arm has a flagged first run and a memout repeat
    add(qr47, "q_bz_on", 0, "ok", 5.0, 11)
    add(qr71, "q_bz_on", 0, "timeout", 1800.0, None)
    add(qr71, "q_bz_prior", 0, "ok", 900.0, 11)
    add(qr79, "q_bz_on", 0, "ok", 900.0, 13, compiled=True)
    add(qr79, "q_bz_on", 1, "memout", 900.0, None)
    add(qr79, "q_bz_prior", 0, "ok", 900.0, 13)
    # three hgp codes that fail: a first run ends in an error; a repeat ends in a memout; and a
    # counting run ends in an error.  Third's prior arm has a flagged first timed run, and fifth's a
    # flagged counting timeout; both codes stay failures.
    add(third, "q_bz_on", 0, "error", 3.0, None)
    add(third, "q_bz_prior", 0, "ok", 30.0, 4, compiled=True)
    add(fourth, "q_bz_on", 0, "ok", 20.0, 4)
    add(fourth, "q_bz_on", 1, "memout", 40.0, None)
    add(fourth, "q_bz_on", 2, "ok", 20.0, 4)
    add(fourth, "q_bz_prior", 0, "ok", 400.0, 4)
    add(fifth, "q_bz_on", 0, "ok", 400.0, 4)
    add(fifth, "q_bz_prior", 0, "ok", 400.0, 4)
    add(fifth, "q_bz_on", 0, "error", 5.0, None, mode="count")
    add(fifth, "q_bz_prior", 0, "timeout", 1800.0, None, "count", None, ({"total": 10},), True)
    # two hgp codes: only the on arm finishes, with complete counts and a partial count; and both
    # time out, with a counting run killed without partial counts
    add(first, "q_bz_on", 0, "ok", 600.0, 4)
    add(first, "q_bz_prior", 0, "timeout", 1800.0, None)
    add(first, "q_bz_on", 0, "ok", 600.0, 4, mode="count", work={"total": 100})
    add(first, "q_bz_prior", 0, "timeout", 1800.0, None, mode="count", partial=({"total": 4000},))
    add(second, "q_bz_on", 0, "timeout", 1800.0, None)
    add(second, "q_bz_prior", 0, "timeout", 1800.0, None)
    add(second, "q_bz_on", 0, "ok", 700.0, 4, mode="count", work={"total": 300})
    add(second, "q_bz_prior", 0, "timeout", 1800.0, None, mode="count")
    expected = [
        experiment._task(arm, "timed", entry, repeat=0)
        for entry in (golay, bb72, qr47, qr71, qr79, pp110, surface9)
        + (first, second, third, fourth, fifth)
        for arm in ("q_bz_on", "q_bz_prior")
    ] + [
        experiment._task(arm, "count", entry, repeat=0)
        for entry, arms_counted in (
            (bb72, ("q_bz_on",)),
            (first, arms.ARMS),
            (second, arms.ARMS),
            (fifth, arms.ARMS),
            (pp110, arms.ARMS),
            (surface9, arms.ARMS),
        )
        for arm in arms_counted
    ]
    records = analyze.summarize(rows, sample, expected)
    out = directory / "analysis"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    analyze.write_outputs(records, sample, out)
    by_name = {(r["name"], r["arm"]): r for r in records}
    on, prior = by_name[("golay", "q_bz_on")], by_name[("golay", "q_bz_prior")]
    assert analyze.ratio(on, prior, "T")["case"] == "disagreement"
    on, prior = by_name[("bb72", "q_bz_on")], by_name[("bb72", "q_bz_prior")]
    assert prior["T"] == 1800.0 and prior["repeats_unfinished"] == 2, prior
    assert analyze.classify(analyze.ratio(on, prior, "T")) == "faster"
    assert on["count_status"] == "missing" and analyze.ratio(on, prior, "W")["case"] == "failure"
    on, prior = by_name[("qr47", "q_bz_on")], by_name[("qr47", "q_bz_prior")]
    assert prior["status"] == "missing" and analyze.ratio(on, prior, "T")["case"] == "failure"
    on, prior = by_name[("qr71", "q_bz_on")], by_name[("qr71", "q_bz_prior")]
    only_prior = analyze.ratio(on, prior, "T")
    assert (only_prior["case"], only_prior["bound"], only_prior["value"]) == (
        "only prior",
        "<=",
        0.5,
    )
    assert analyze.classify(only_prior) == "slower"
    on, prior = by_name[("qr79", "q_bz_on")], by_name[("qr79", "q_bz_prior")]
    assert analyze.ratio(on, prior, "T")["case"] == "disagreement"
    on, prior = by_name[(first["name"], "q_bz_on")], by_name[(first["name"], "q_bz_prior")]
    only_on = analyze.ratio(on, prior, "T")
    assert (only_on["case"], only_on["bound"], only_on["value"]) == ("only on", ">=", 3.0)
    work = analyze.ratio(on, prior, "W")
    assert (work["case"], work["bound"], work["value"]) == ("only on", ">=", 40.0), work
    on, prior = by_name[(second["name"], "q_bz_on")], by_name[(second["name"], "q_bz_prior")]
    assert analyze.ratio(on, prior, "T")["case"] == "both > L"
    assert analyze.ratio(on, prior, "W")["case"] == "no work"
    for entry, metric in ((third, "T"), (fourth, "T"), (fifth, "W")):
        on, prior = by_name[(entry["name"], "q_bz_on")], by_name[(entry["name"], "q_bz_prior")]
        assert analyze.ratio(on, prior, metric)["case"] == "failure", (entry["name"], metric)
    outputs = ("summary.csv", "families.csv", "anchors.csv", "distances.pdf", "figure_notes.json")
    for name in outputs:
        assert (out / name).exists(), name
    with open(out / "families.csv") as file:
        (anchors,) = [row for row in csv.DictReader(file) if row["family"] == "anchors"]
    assert (anchors["codes"], anchors["disagreement"], anchors["failure"]) == ("7", "2", "1")
    assert anchors["work_failures"] == "4", anchors
    with open(out / "families.csv") as file:
        (hgp,) = [row for row in csv.DictReader(file) if row["family"] == "hgp"]
    assert (hgp["median_R_T"], hgp["both > L"], hgp["failure"]) == (">=2", "1", "2"), hgp
    assert (hgp["median_R_W"], hgp["work_ratios_in_median"], hgp["work_failures"]) == (
        ">=40",
        "1",
        "3",
    )

    # a work count without a known total is refused rather than read as missing
    unreadable = [dict(rows[-1], status="ok", result={"distance": 4, "work": {"other": 1}})]
    try:
        analyze.summarize(unreadable, sample, [])
    except ValueError:
        pass
    else:
        raise AssertionError("an unreadable work count was accepted")

    # pp110's flagged first timed run leaves R_T incomplete and its T cell "compiled", and its
    # flagged counting run, which finished, still gives R_W; surface9's flagged timed-out repeat is
    # left out of the median and of repeats_unfinished, and its flagged counting timeout leaves R_W
    # incomplete and its W cell "compiled"
    on, prior = by_name[("pp110", "q_bz_on")], by_name[("pp110", "q_bz_prior")]
    assert on["T"] is None and on["runs_compiled"] == 1, on
    assert analyze.ratio(on, prior, "T")["case"] == "incomplete timing"
    assert analyze.ratio(on, prior, "W")["value"] == 2.0
    on, prior = by_name[("surface9", "q_bz_on")], by_name[("surface9", "q_bz_prior")]
    assert (on["T"], on["repeats"], on["repeats_unfinished"], on["runs_compiled"]) == (1.5, 2, 0, 1)
    assert on["W"] is None and analyze.ratio(on, prior, "W")["case"] == "incomplete timing", on
    with open(out / "anchors.csv") as file:
        cells = {row["name"]: row for row in csv.DictReader(file)}
    assert (cells["pp110"]["q_bz_on_T"], cells["surface9"]["q_bz_on_W"]) == ("compiled",) * 2
    # the T cell of a failed arm with a flagged first run shows the failure: qr79's memout repeat
    assert cells["qr79"]["q_bz_on_T"] == "memout", cells["qr79"]
    assert anchors["incomplete timing"] == "1", anchors


if __name__ == "__main__":
    main()
