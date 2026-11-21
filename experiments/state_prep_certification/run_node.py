# SPDX-License-Identifier: Apache-2.0

"""Run one experiment: acceptance tests, task lists, the two passes, and the archive.

    python experiments/state_prep_certification/run_node.py EXPERIMENT --approved-cpu-hours H \
        [options]

On the node, ``node.sh`` calls this script with the bundle's interpreter.  For the label of the
experiment (``common/manifest.py``), it

1. refuses a bundle built without a pinned qLDPC commit, unless ``--allow-unfrozen`` is given;
2. runs the harness's node-mode self-tests and the experiment's node acceptance tests, once per
   label, and stops if they fail;
3. computes the ceiling of CPU-hours of the task list of pass 1, and refuses to go on unless
   ``--approved-cpu-hours`` covers it;
4. prepares the inputs (``prepare``), which checks them against the experiment's frozen files, and
   writes the task list;
5. writes ``cores.txt`` and the manifest: the number of job slots is the smallest of the number of
   free physical cores, the number of ``JOB_MEM_GIB`` slots that fit in memory, and ``--jobs``, if
   given, with ``tasks.RESERVED_CORES`` physical cores and ``tasks.RESERVED_MEM_GIB`` of memory
   reserved; it also pins this script, and so GNU parallel and the task runners, to the logical
   CPUs of the reserved cores;
6. runs pass 1 with GNU parallel, one task per job slot, whose child process pins itself to its
   slot's CPU, then the repeats of pass 2, and checks that every task has a final row;
7. writes the archive ``archive/<label>-<host>-<UTC>.tar.gz``: the one file to copy back.

Repeating the command resumes the run: GNU parallel's ``--resume-failed`` skips the tasks that
ended with status 0, and the task runner skips tasks that already have a final row.  The script
refuses to start while a run of an earlier start is still alive.  After a hard kill of this script
or of GNU parallel, run it once with ``--cleanup``, which kills every run still listed in the
label's ``active/`` directory, if its processes are still the ones recorded.  ``--status`` prints
the progress of the label.

``--local`` runs the tasks one at a time without GNU parallel, keeps every run in this session,
and skips the acceptance tests unless ``--subset acceptance`` asks for them; it serves development
on a workstation, whose times are never used as results.  ``--dry-run`` stops after step 5 and
prints the GNU parallel command.  ``--subset NAME`` selects a subset of the tasks, such as
``smoke``; a subset other than ``main`` or ``acceptance`` gets its own label directory (see
``common/manifest.py``).  ``--subset acceptance`` runs only the acceptance tests.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import signal
import socket
import subprocess
import sys
import time
import types
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.state_prep_certification.common import manifest, plugin, runner, store, tasks

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN_TASK = ROOT / "experiments" / "state_prep_certification" / "common" / "run_task.py"
SELFTEST = ROOT / "experiments" / "state_prep_certification" / "common" / "selftest.py"
DEFAULT_RESULTS = ROOT / "experiments" / "state_prep_certification" / "results"


def main() -> None:
    """Run the experiment, or report on or clean up its label."""
    args = parse_args()
    module = plugin.load_experiment(args.experiment)
    results_dir = (args.results_dir or DEFAULT_RESULTS).resolve()
    label_dir = get_label_dir(module, results_dir, args.subset, args.local)
    if args.cleanup:
        cleanup(label_dir)
        return
    if args.status:
        print_status(label_dir)
        return
    if not args.local and not sys.platform.startswith("linux"):
        sys.exit("Node mode needs Linux (lscpu, CPU affinity); pass --local on other systems")
    if not args.local and _unfrozen_bundle() and not args.allow_unfrozen:
        sys.exit("This bundle has no pinned qLDPC commit; pass --allow-unfrozen only for a trial")
    label_dir.mkdir(parents=True, exist_ok=True)
    live = live_runs(label_dir)
    if live:
        sys.exit(f"Runs of an earlier start are still alive ({live[:3]}...); run --cleanup first")
    print(f"label directory: {label_dir}", flush=True)

    # 2. self-tests and acceptance tests, once per label
    if args.subset == "acceptance" or not args.local:
        if not run_acceptance(module, label_dir, results_dir, node=not args.local):
            sys.exit("The acceptance tests failed; see above.  Nothing was run.")
        if args.subset == "acceptance":
            return

    # 3-4. task list and ceiling, then the inputs, checked against the frozen files
    task_list = module.tasks(label_dir, args.subset)
    try:
        tasks.check_memory_caps(task_list, module.JOB_MEM_GIB)
    except ValueError as error:
        sys.exit(f"{error}.  No task file was written.")
    ceiling = tasks.ceiling_hours(task_list, module.worst_case_s)
    print(f"{len(task_list)} tasks in pass 1; ceiling {ceiling:,.1f} CPU-hours", flush=True)
    try:
        tasks.check_approval(ceiling, args.approved_cpu_hours)
    except tasks.ApprovalError as error:
        sys.exit(f"{error}.  No task file was written.")
    module.prepare(label_dir)
    tasks.write_pass1(label_dir, task_list)

    # 5. job slots, with this process and so GNU parallel and the runners on the reserved cores
    jobs, cpus, reserved = choose_slots(module, label_dir, args.jobs, args.local)
    pin_runners(reserved)
    manifest.write_manifest(
        label_dir,
        results_dir,
        {
            "subset": args.subset,
            "jobs": jobs,
            "cpus": cpus,
            "runner_cpus": reserved,
            "ceiling_cpu_hours": ceiling,
            "approved_cpu_hours": args.approved_cpu_hours,
            "local": args.local,
        },
    )
    command = tasks.parallel_command(label_dir, pathlib.Path(sys.executable), "1", jobs)
    print("pass 1 command (from the qLDPC root):\n  " + shlex.join(command), flush=True)
    if args.dry_run:
        return

    # 6. the passes
    run_pass(label_dir, "1", jobs, args.local)
    final, _ = store.final_rows(label_dir)
    tasks.write_pass2(label_dir, final, module.repeat_tasks)
    run_pass(label_dir, "2", jobs, args.local)
    final, problems = store.final_rows(label_dir)
    if reserved:
        problems += [
            f"{key}: its runner ran on {row['runner_affinity']}, not on the reserved CPUs"
            for key, row in final.items()
            if row.get("runner_affinity") != sorted(reserved)
        ]
    for problem in problems:
        print(f"note: {problem}", flush=True)

    # 7. archive
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = store.make_archive(label_dir, f"{label_dir.name}-{socket.gethostname()}-{stamp}")
    print(f"archive: {archive}\n         {archive}.sha256", flush=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiment", choices=sorted(plugin.PLUGINS))
    parser.add_argument(
        "--results-dir", type=pathlib.Path, help="default: state_prep_certification/results"
    )
    parser.add_argument(
        "--jobs", default="auto", help="the number of job slots: auto (the default) or a bound"
    )
    parser.add_argument("--approved-cpu-hours", type=float, help="the approved CPU-hours")
    parser.add_argument("--local", action="store_true", help="run locally, one task at a time")
    parser.add_argument("--dry-run", action="store_true", help="write the task files and stop")
    parser.add_argument("--status", action="store_true", help="print the label's progress")
    parser.add_argument("--cleanup", action="store_true", help="kill runs left by a hard kill")
    parser.add_argument("--subset", default="main", help="the subset of tasks (default: main)")
    parser.add_argument(
        "--allow-unfrozen", action="store_true", help="run a bundle without a pinned qLDPC commit"
    )
    return parser.parse_args()


####################################################################################################
# label, slots, and acceptance


def get_label_dir(
    module: types.ModuleType, results_dir: pathlib.Path, subset: str, local: bool
) -> pathlib.Path:
    """The directory of the experiment's label, with the suffixes of ``common/manifest.py``."""
    commit = manifest.harness_commit()
    if commit["dirty"] and not local:
        sys.exit("The qLDPC tree has uncommitted changes; commit them, or pass --local")
    frozen = [plugin.experiment_dir(module.NAME) / name for name in module.FROZEN_FILES]
    label = manifest.compute_label(module.NAME, frozen, commit["commit"])
    label += "-dirty" if commit["dirty"] else ""
    label += "" if subset in ("main", "acceptance") else f"-{subset}"
    return results_dir / module.NAME / label


def choose_slots(
    module: types.ModuleType, label_dir: pathlib.Path, jobs: str, local: bool
) -> tuple[int, list[int], list[int]]:
    """The number of job slots J, the CPU of each slot, and the CPUs of the reserved cores.

    On the node, the CPUs of the slots go to ``cores.txt``.  Locally, there is one slot and no
    pinning.
    """
    if local:
        return 1, [], []
    lscpu = subprocess.run(
        ["lscpu", "-p=CPU,CORE,SOCKET"], capture_output=True, text=True, check=True
    ).stdout
    bound = None if jobs == "auto" else int(jobs)
    cpus, reserved = tasks.choose_cpus(lscpu, manifest.mem_total_bytes(), module.JOB_MEM_GIB, bound)
    tasks.write_cores(label_dir, cpus)
    return len(cpus), cpus, reserved


def run_acceptance(
    module: types.ModuleType, label_dir: pathlib.Path, results_dir: pathlib.Path, node: bool
) -> bool:
    """Run the self-tests (on the node) and the acceptance tests, unless they passed before.

    A pass in local mode does not count for node mode.
    """
    record = label_dir / "acceptance.json"
    if record.exists():
        previous = json.loads(record.read_text())
        if previous["passed"] and (previous["node"] or not node):
            print("acceptance tests: passed before for this label", flush=True)
            return True
    passed = True
    if node:
        selftest = [sys.executable, str(SELFTEST), "--node"]
        passed = subprocess.run(selftest, cwd=ROOT, check=False).returncode == 0
    passed = passed and bool(module.acceptance(node, results_dir / "acceptance"))
    record.write_text(json.dumps({"passed": passed, "node": node, "at": time.time()}) + "\n")
    return passed


def _unfrozen_bundle() -> bool:
    """Whether this tree comes from a bundle built without a frozen qLDPC commit."""
    if not manifest.BUNDLE_RECORD.exists():
        return False
    return bool(json.loads(manifest.BUNDLE_RECORD.read_text()).get("unfrozen"))


####################################################################################################
# passes


def run_pass(label_dir: pathlib.Path, name: str, jobs: int, local: bool) -> None:
    """Run one pass, and again the tasks that ended without a final row."""
    keys = tasks.pass_keys(label_dir, name)
    _run_keys(label_dir, name, keys, jobs, local)

    # a task that ended with status 0 but has no readable final row runs again, up to 3 times
    for retry in range(1, 4):
        final, _ = store.final_rows(label_dir)
        missing = [key for key in keys if key not in final]
        if not missing:
            return
        retry_name = f"{name}-retry{retry}-{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}"
        tasks.write_once(tasks.pass_file(label_dir, retry_name), "".join(k + "\n" for k in missing))
        print(f"rerunning {len(missing)} tasks without a final row", flush=True)
        _run_keys(label_dir, retry_name, missing, jobs, local)
    final, _ = store.final_rows(label_dir)
    missing = [key for key in keys if key not in final]
    if missing:
        sys.exit(f"{len(missing)} tasks of pass {name} have no final row, such as {missing[0]}")


def run_parallel(command: list[str]) -> int:
    """Run GNU parallel from the qLDPC root, passing SIGTERM on to it; return its exit status."""
    print("+ " + shlex.join(command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT)

    def forward(signum: int, frame: Any) -> None:
        process.send_signal(signum)

    previous = signal.signal(signal.SIGTERM, forward)
    try:
        return process.wait()
    finally:
        signal.signal(signal.SIGTERM, previous)


def _run_keys(label_dir: pathlib.Path, name: str, keys: list[str], jobs: int, local: bool) -> None:
    """Run the tasks of a pass file: one at a time locally, or with GNU parallel."""
    if local:
        for key in keys:
            command = [sys.executable, str(RUN_TASK), "--label-dir", str(label_dir), "--local", key]
            if subprocess.run(command, cwd=ROOT, check=False).returncode:
                sys.exit(f"Task {key} ended without a final row; rerun to resume")
        return
    command = tasks.parallel_command(label_dir, pathlib.Path(sys.executable), name, jobs)
    if run_parallel(command):
        sys.exit(f"Pass {name} did not finish; rerun the same command to resume")


####################################################################################################
# status and cleanup


def print_status(label_dir: pathlib.Path) -> None:
    """Print the final rows of the label by pass and status, and the active runs."""
    if not (label_dir / tasks.TASKS_FILE).exists():
        print(f"{label_dir}: no task list yet")
        return
    final, problems = store.final_rows(label_dir)
    for name in ("1", "2"):
        if not tasks.pass_file(label_dir, name).exists():
            continue
        keys = tasks.pass_keys(label_dir, name)
        counts: dict[str, int] = {}
        for key in keys:
            status = final[key]["status"] if key in final else "pending"
            counts[status] = counts.get(status, 0) + 1
        print(f"pass {name}: {len(keys)} tasks; {counts}")
    active = sorted(path.name for path in (label_dir / "active").glob("*"))
    print(f"active runs: {len(active)}" + "".join(f"\n  {key}" for key in active))
    for problem in problems:
        print(f"note: {problem}")


def pin_runners(reserved: list[int]) -> None:
    """Pin this process, and so GNU parallel and the task runners, to the reserved CPUs (Linux)."""
    if reserved and sys.platform.startswith("linux"):
        try:
            os.sched_setaffinity(0, reserved)
        except OSError as error:
            sys.exit(f"Could not pin to the reserved CPUs {reserved}: {error}")
        if os.sched_getaffinity(0) != set(reserved):
            sys.exit(f"Could not pin to the reserved CPUs {reserved}")


def live_runs(label_dir: pathlib.Path) -> list[str]:
    """The tasks of ``active/`` whose runner or run is still the recorded process.

    A run is also live if its process group's leader is gone but members of the group remain,
    which is when ``cleanup`` kills the group.
    """
    live = []
    table = runner.process_table()
    for path in sorted((label_dir / "active").glob("*")):
        record = json.loads(path.read_text())
        recorded = [
            (record["runner_pid"], record["runner_start"]),
            (record["pid"], record["pid_start"]),
        ]
        same = any(
            start is not None and pid in table and table[pid].start == start
            for pid, start in recorded
        )
        orphaned_group = (
            record.get("pgid") is not None
            and record["pid"] not in table
            and any(process.pgid == record["pid"] for process in table.values())
        )
        if same or orphaned_group:
            live.append(path.name)
    return live


def cleanup(label_dir: pathlib.Path) -> None:
    """Kill every run listed in the label's ``active/`` directory, and remove the files.

    Each task runner is killed first, so that it cannot write a row for a run that it lost.  A
    process is killed only if its start time is the recorded one.  A run's process group is killed
    only if its leader is that recorded process, or if no process has the leader's id any more but
    members of the group remain; a process that has reused the id is never killed.
    """
    for path in sorted((label_dir / "active").glob("*")):
        record = json.loads(path.read_text())
        _kill_if_same(record["runner_pid"], record["runner_start"], group=False)
        if record.get("pgid") is not None:
            killed = _kill_if_same(record["pid"], record["pid_start"], group=True)
        else:
            killed = _kill_tree_if_same(record["pid"], record["pid_start"])
        print(f"{path.name}: {'killed its run' if killed else 'its run had already ended'}")
        path.unlink()


def _kill_tree_if_same(pid: int, start: str | None) -> bool:
    """Kill a local run's process and its descendants, if it is still the recorded process."""
    table = runner.process_table()
    if start is None or pid not in table or table[pid].start != start:
        return False
    for member in runner.descendants(table, pid):
        try:
            os.kill(member, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return True


def _kill_if_same(pid: int, start: str | None, group: bool) -> bool:
    """Kill a process (or the group that it leads) if it is still the one recorded."""
    table = runner.process_table()
    if pid in table:
        if start is None or table[pid].start != start:
            return False
    elif not group or not any(process.pgid == pid for process in table.values()):
        return False
    try:
        if group:
            os.killpg(pid, signal.SIGKILL)
        else:
            os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return False
    return True


if __name__ == "__main__":
    main()
