# SPDX-License-Identifier: Apache-2.0

"""Task lists for GNU parallel: pass 1, pass 2, the cores of the job slots, and the ceiling.

A task is a JSON object with a unique ``key``, the ``experiment``, ``arm``, ``mode``, ``repeat``,
``limit_s``, and ``mem_cap_gib``.  It may also set ``post_s``, ``startup_cap_s``, and ``grace_s``,
which default to ``runner.FINISH_S``, ``runner.STARTUP_CAP_S``, and ``runner.GRACE_S``.  Beyond
these, it holds whatever the experiment needs to run it, such as an input.  ``write_pass1`` writes
every task of pass 1 to ``tasks.json`` and their keys, shuffled with a fixed seed, to
``tasks-pass1.txt``.  ``write_pass2`` writes the repeats that the experiment's repeat policy asks
for, once every task of pass 1 has a final row, to ``tasks-pass2.json`` and ``tasks-pass2.txt``, in
the order of pass 1.  A task file is written once and never changed, since GNU parallel's
resumption depends on its order; writing it again with other contents is an error, because a
changed input must give a new label.  ``cores.txt`` lists the logical CPU of each job slot.

The ceiling is the worst-case CPU time of the task list, with the runner's retries.  For each task
of pass 1, the experiment's ``worst_case_s`` W counts its first run and the repeats that it can
cause, without the runner's retries.  The runner runs each of these at most twice
(``runner.RUNS_PER_TASK``), so they cost at most 2 W.  One more run at the first run's limit, S
(``run_cost_s``), covers a first run that is discarded after its whole limit, while its retry
finishes fast and calls for repeats.  So the task costs at most S + 2 W.  The bound is loose for a
task without repeats, which costs at most 2 S.  ``check_approval`` refuses a list whose ceiling
exceeds the approved CPU-hours.
"""

from __future__ import annotations

import json
import pathlib
import random
import shlex
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from experiments.state_prep_certification.common import runner, store

TASKS_FILE = "tasks.json"
PASS2_TASKS_FILE = "tasks-pass2.json"
CORES_FILE = "cores.txt"
SHUFFLE_SEED = 20261115

# GNU parallel's signal sequence for a stopped task: TERM, then KILL after the runner's
# ``TERM_WAIT_S``, which leaves the task runner time to kill its run and wait out its grace.
TERMSEQ = f"TERM,{int(runner.TERM_WAIT_S * 1000)},KILL,25"

# Physical cores, and memory, that no job slot uses.
RESERVED_CORES = 2
RESERVED_MEM_GIB = 16.0


class ApprovalError(Exception):
    """The ceiling of a task list exceeds the approved CPU-hours, or no approval was given."""


####################################################################################################
# task files


def pass_file(label_dir: pathlib.Path, name: str) -> pathlib.Path:
    """The key list of a pass, ``1`` or ``2``."""
    return label_dir / f"tasks-pass{name}.txt"


def joblog_file(label_dir: pathlib.Path, name: str) -> pathlib.Path:
    """GNU parallel's job log of a pass."""
    return label_dir / f"joblog-pass{name}.tsv"


def write_pass1(label_dir: pathlib.Path, tasks: Sequence[dict[str, Any]]) -> list[str]:
    """Write ``tasks.json`` and the shuffled ``tasks-pass1.txt`` once; return the keys in order."""
    keys = [task["key"] for task in tasks]
    if len(set(keys)) != len(keys):
        raise ValueError("Task keys must be unique")
    for task in tasks:
        runner.check_task(task)
    order = list(keys)
    random.Random(SHUFFLE_SEED).shuffle(order)
    write_once(label_dir / TASKS_FILE, json.dumps(list(tasks), indent=1, sort_keys=True) + "\n")
    write_once(pass_file(label_dir, "1"), "".join(key + "\n" for key in order))
    return order


def write_pass2(
    label_dir: pathlib.Path,
    final: dict[str, dict[str, Any]],
    repeat_tasks: Callable[[dict[str, Any], dict[str, Any]], list[dict[str, Any]]],
) -> list[str]:
    """Write the repeats of pass 2 once, in the order of pass 1; return their keys.

    Args:
        label_dir: The label's directory.
        final: The first final row of each task, by key; every task of pass 1 must have one.
        repeat_tasks: The experiment's repeat policy, which maps a task of pass 1 and its final row
            to the tasks of its repeats.
    """
    tasks = load_tasks(label_dir, pass2=False)
    order = pass_keys(label_dir, "1")
    missing = [key for key in order if key not in final]
    if missing:
        raise ValueError(f"{len(missing)} tasks of pass 1 have no final row, such as {missing[0]}")
    repeats = [repeat for key in order for repeat in repeat_tasks(tasks[key], final[key])]
    for task in repeats:
        runner.check_task(task)
    write_once(label_dir / PASS2_TASKS_FILE, json.dumps(repeats, indent=1, sort_keys=True) + "\n")
    write_once(pass_file(label_dir, "2"), "".join(task["key"] + "\n" for task in repeats))
    return [task["key"] for task in repeats]


def load_tasks(label_dir: pathlib.Path, pass2: bool = True) -> dict[str, dict[str, Any]]:
    """Every task of the label by key: pass 1, and pass 2 if it has been written."""
    tasks = json.loads((label_dir / TASKS_FILE).read_text())
    if pass2 and (label_dir / PASS2_TASKS_FILE).exists():
        tasks += json.loads((label_dir / PASS2_TASKS_FILE).read_text())
    return {task["key"]: task for task in tasks}


def pass_keys(label_dir: pathlib.Path, name: str) -> list[str]:
    """The keys of one pass, in order."""
    return pass_file(label_dir, name).read_text().split()


####################################################################################################
# ceiling and approval


def ceiling_hours(
    tasks: Iterable[dict[str, Any]], worst_case_s: Callable[[dict[str, Any]], float]
) -> float:
    """The worst-case CPU-hours of the tasks of pass 1, with their repeats and retries.

    If the runner runs a task's first run and each of its repeats up to k = ``runner.RUNS_PER_TASK``
    times, the task costs at most (k - 1) S + k W, where S is ``run_cost_s(task)`` and W is
    ``worst_case_s(task)`` (see the module docstring).
    """
    runs = runner.RUNS_PER_TASK
    return sum((runs - 1) * run_cost_s(task) + runs * worst_case_s(task) for task in tasks) / 3600


def run_cost_s(task: dict[str, Any]) -> float:
    """The most seconds that one run of a task can take: start-up, limit, post-processing, grace."""
    return (
        float(task.get("startup_cap_s", runner.STARTUP_CAP_S))
        + float(task["limit_s"])
        + float(task.get("post_s", runner.FINISH_S))
        + float(task.get("grace_s", runner.GRACE_S))
    )


def check_memory_caps(tasks: Sequence[dict[str, Any]], job_mem_gib: float) -> None:
    """Raise ValueError if a task's memory cap exceeds the memory of a job slot."""
    too_big = [task["key"] for task in tasks if task["mem_cap_gib"] > job_mem_gib]
    if too_big:
        raise ValueError(
            f"Tasks with memory caps above the slots' {job_mem_gib:g} GiB: {too_big[:3]}"
        )


def check_approval(ceiling: float, approved: float | None) -> None:
    """Raise ``ApprovalError`` unless the approved CPU-hours cover the ceiling."""
    if approved is None:
        raise ApprovalError(
            f"The ceiling is {ceiling:,.1f} CPU-hours; pass --approved-cpu-hours to run it"
        )
    if approved < ceiling:
        raise ApprovalError(
            f"The ceiling is {ceiling:,.1f} CPU-hours, above the approved {approved:,.1f}"
        )


####################################################################################################
# job slots and cores


def choose_cpus(
    lscpu_output: str, mem_total_bytes: int, mem_per_job_gib: float, jobs: int | None = None
) -> tuple[list[int], list[int]]:
    """The CPU of each job slot, and the CPUs of the reserved cores, where the runners run.

    Each slot gets the first logical CPU of its own physical core, and the slots skip the first
    ``RESERVED_CORES`` physical cores, all of whose logical CPUs go to the runners, GNU parallel,
    and the system.  The number of slots J is the smallest of the number of free physical cores,
    the number of slots of ``mem_per_job_gib`` each that fit in the memory beyond
    ``RESERVED_MEM_GIB``, and ``jobs``, if given.

    Args:
        lscpu_output: The output of ``lscpu -p=CPU,CORE,SOCKET``.
        mem_total_bytes: The machine's total memory.
        mem_per_job_gib: The memory of each job slot (the experiment's ``JOB_MEM_GIB``).
        jobs: An upper bound on the number of slots, if any.
    """
    cpus_of_core: dict[tuple[int, int], list[int]] = {}
    for line in lscpu_output.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cpu, core, socket = (int(value) for value in line.split(",")[:3])
        cpus_of_core.setdefault((socket, core), []).append(cpu)
    cores = sorted(cpus_of_core.values(), key=min)
    by_memory = int((mem_total_bytes / 2**30 - RESERVED_MEM_GIB) // mem_per_job_gib)
    count = min(len(cores) - RESERVED_CORES, by_memory)
    if jobs is not None:
        count = min(count, jobs)
    if count < 1:
        raise ValueError("No job slot fits on this machine")
    slots = [min(core) for core in cores[RESERVED_CORES : RESERVED_CORES + count]]
    reserved = sorted(cpu for core in cores[:RESERVED_CORES] for cpu in core)
    return slots, reserved


def write_cores(label_dir: pathlib.Path, cpus: Sequence[int]) -> None:
    """Write ``cores.txt``: line j holds the logical CPU of job slot j.

    A resume may use another number of slots; each start's slots are in ``starts.jsonl``.
    """
    (label_dir / CORES_FILE).write_text("".join(f"{cpu}\n" for cpu in cpus))


def write_once(path: pathlib.Path, text: str) -> None:
    """Write a file that must never change: create it, or check that it already holds ``text``."""
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(
                f"{path} exists with other contents; a changed input needs a new label"
            )
        return
    store.write_atomic(path, text)


####################################################################################################
# GNU parallel


def parallel_command(
    label_dir: pathlib.Path, python: pathlib.Path, name: str, jobs: int, local: bool = False
) -> list[str]:
    """The GNU parallel command of one pass, to be run from the root of the qLDPC tree.

    Job slot ``{%}`` runs from 1 to J, and no two running jobs share a slot, so each task's child
    process pins itself to the CPU on line ``{%}`` of ``cores.txt``, and its descendants inherit
    that affinity.  ``local`` serves local tests, whose runs stay in the caller's session.
    """
    label = shlex.quote(str(label_dir))
    task_command = (
        f"{shlex.quote(str(python))} experiments/state_prep_certification/common/run_task.py "
        f'--label-dir {label} --job-slot {{%}} --cpu "$(sed -n {{%}}p {label}/{CORES_FILE})"'
        + (" --local" if local else "")
        + " {}"
    )
    return [
        "parallel",
        "--will-cite",
        "--jobs",
        str(jobs),
        "--arg-file",
        str(pass_file(label_dir, name)),
        "--joblog",
        str(joblog_file(label_dir, name)),
        "--resume-failed",
        "--termseq",
        TERMSEQ,
        "--line-buffer",
        "--tag",
        task_command,
    ]
