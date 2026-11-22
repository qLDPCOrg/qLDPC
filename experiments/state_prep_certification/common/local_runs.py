# SPDX-License-Identifier: Apache-2.0

"""Run tasks of an experiment locally through the task runner, for acceptance tests.

Each task runs as GNU parallel would run it, through ``run_task.py``, but in local mode: one at a
time, in the caller's session, and killed by process tree.  If an exception, such as one that the
handler of a test's time limit raises, interrupts ``run_locally`` while a task runs, ``run_locally``
sends the task's runner SIGTERM, as GNU parallel does.  The runner then kills its run and exits,
within ``runner.TERM_WAIT_S``, before ``run_locally`` re-raises the exception.

``run_locally`` fails if a row with status ``ok`` has ``compiled_in_timing`` set (see
``runner.py``), which checks that the disk caches and any untimed preparation keep every compilation
that numba reports out of the measured call.  Rows of other statuses are not checked, since the
retry of a run killed at its limit can compile code that the discarded run never reached.
"""

from __future__ import annotations

import pathlib
import subprocess
from collections.abc import Sequence
from typing import Any

from experiments.state_prep_certification.common import plugin, runner, store, tasks


def run_locally(
    label_dir: pathlib.Path, task_list: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Write the tasks as pass 1 of a fresh label directory, run them, and return their rows.

    Raises:
        AssertionError: If a row with status ``ok`` has ``compiled_in_timing`` set.
        subprocess.CalledProcessError: If a task's runner exits with a nonzero status.
    """
    label_dir.mkdir(parents=True, exist_ok=True)
    tasks.write_pass1(label_dir, task_list)
    for task in task_list:
        command = plugin.run_task_command(label_dir, task["key"], local=True)
        with subprocess.Popen(command) as process:
            try:
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(runner.TERM_WAIT_S)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    final, _ = store.final_rows(label_dir)
    compiled = [
        key for key, row in final.items() if row["status"] == "ok" and row["compiled_in_timing"]
    ]
    if compiled:
        raise AssertionError(f"retries compiled in their measured calls: {compiled}")
    return final
