# SPDX-License-Identifier: Apache-2.0

"""A stand-in experiment with which ``selftest.py`` tests the shared code of the experiments.

Each task's ``input`` is a list of steps that ``run`` performs in order:

* ``["startup", s]``: sleep s seconds before timing starts;
* ``["sleep", s]``: the measured call sleeps s seconds;
* ``["alloc", gib]``: the measured call holds gib GiB of touched memory for 5 seconds;
* ``["flash", gib]``: the measured call touches gib GiB of memory and frees it at once, within a
  fraction of a second;
* ``["child_flash", gib]``: the measured call starts a grandchild that touches gib GiB of memory
  and exits at once, and waits for it;
* ``["spawn", s]``: the measured call starts a grandchild that sleeps s seconds, and waits for it;
* ``["orphan", s]``: the measured call starts a grandchild that sleeps s seconds, and does not wait;
* ``["partial", n]``: the measured call sends n partial results;
* ``["fail"]``: the measured call raises an exception;
* ``["post", s]``: after the measured call, send the result and post-process for s seconds;
* ``["value", v]``: the measured call returns v;
* ``["garbage", s]``: the measured call writes a line that is not a message to the runner, then
  sleeps s seconds;
* ``["compile", cache]``: the measured call compiles ``_add_one`` with numba, cached on disk if
  ``cache`` is true, and calls it;
* ``["compile_on_thread", cache]``: the same, on a thread that the measured call starts and joins;
* ``["vectorize"]``: the measured call compiles ``_add_one`` as a ufunc with ``numba.vectorize``,
  not cached on disk, and calls it;
* ``["compile_untimed"]``: ``_add_one`` is compiled, not cached, before the measured call, and again
  after it (after ``send_result``, if there is a ``post`` step);
* ``["die"]``: the measured call kills its own process with SIGKILL.

Every result also holds the child's process id and the wall-clock interval of its measured call,
with which the tests check that concurrent runs use different CPUs.
"""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np

from experiments.state_prep_certification.common import runner
from experiments.state_prep_certification.common.tasks import run_cost_s

NAME = "selftest"
EXTRA_WHEELS: list[str] = []
DATA: list[str] = []
FROZEN_FILES: list[str] = []
JOB_MEM_GIB = 1.0


def prepare(label_dir: pathlib.Path) -> None:
    """Nothing to prepare."""


def tasks(label_dir: pathlib.Path, subset: str) -> list[dict[str, Any]]:
    """Three short timed tasks, for tests of ``run_node.py``."""
    return [
        {
            "key": f"stand_in.timed.t{index}.r0",
            "experiment": NAME,
            "arm": "stand_in",
            "mode": "timed",
            "repeat": 0,
            "input": [["sleep", 0.1 * index]],
            "limit_s": 10.0,
            "mem_cap_gib": JOB_MEM_GIB,
            "grace_s": 5.0,
        }
        for index in range(3)
    ]


def worst_case_s(task: dict[str, Any]) -> float:
    """Three times the most seconds of one run: a first run and two repeats."""
    return 3 * run_cost_s(task)


def repeat_tasks(task: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    """Two repeats of every timed task that finished in under 1 s."""
    if task["mode"] != "timed" or row["status"] != "ok" or row["T"] >= 1:
        return []
    return [dict(task, key=f"{task['key']}.r{i}", repeat=i) for i in (1, 2)]


def run(task: dict[str, Any], label_dir: pathlib.Path) -> dict[str, Any]:
    """Perform the task's steps."""
    steps = [tuple(step) for step in task["input"]]
    compile_untimed = ("compile_untimed",) in steps
    for step in steps:
        if step[0] == "startup":
            time.sleep(step[1])
    if compile_untimed:
        _compile(False)
    start = time.time()
    value = runner.timed(_measured, steps)
    fields = {"value": value, "pid": os.getpid(), "wall": [start, time.time()]}
    for step in steps:
        if step[0] == "post":
            runner.send_result(fields)
            if compile_untimed:
                _compile(False)
            time.sleep(step[1])
            return {"post": "done"}
    if compile_untimed:
        _compile(False)
    return fields


def acceptance(node: bool, results_dir: pathlib.Path) -> bool:
    """The stand-in has no acceptance tests of its own."""
    return True


def _measured(steps: list[tuple[Any, ...]]) -> Any:
    """The measured call of the stand-in."""
    value = None
    for step in steps:
        if step[0] == "sleep":
            time.sleep(step[1])
        elif step[0] in ("alloc", "flash"):
            block = bytearray(int(step[1] * 2**30))
            for index in range(0, len(block), 4096):
                block[index] = 1
            if step[0] == "alloc":
                time.sleep(5)
            del block
        elif step[0] == "child_flash":
            size = int(step[1] * 2**30)
            script = f"block = bytearray({size})\nfor i in range(0, {size}, 4096): block[i] = 1"
            subprocess.run([sys.executable, "-c", script], check=True)
        elif step[0] in ("spawn", "orphan"):
            child = subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({step[1]})"])
            if step[0] == "spawn":
                child.wait()
        elif step[0] == "partial":
            for index in range(step[1]):
                runner.send_partial({"index": index})
        elif step[0] == "fail":
            raise RuntimeError("the stand-in failed on purpose")
        elif step[0] == "value":
            value = step[1]
        elif step[0] == "garbage":
            os.write(int(os.environ["FTSP_CHANNEL_FD"]), b"not a message\n")
            time.sleep(step[1])
        elif step[0] == "compile":
            _compile(step[1])
        elif step[0] == "compile_on_thread":
            thread = threading.Thread(target=_compile, args=(step[1],))
            thread.start()
            thread.join()
        elif step[0] == "vectorize":
            import numba

            numba.vectorize(["int64(int64)"])(_add_one)(np.arange(2))
        elif step[0] == "die":
            os.kill(os.getpid(), signal.SIGKILL)
    return value


def _compile(cache: bool) -> None:
    """Compile ``_add_one`` with numba, cached on disk if ``cache`` is true, and call it."""
    import numba

    numba.njit(cache=cache)(_add_one)(1)


def _add_one(number: int) -> int:
    """The function that the compiling steps compile."""
    return number + 1
