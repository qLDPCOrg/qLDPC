# SPDX-License-Identifier: Apache-2.0

"""Run tasks of an experiment locally through the task runner, for acceptance tests.

Each task runs as GNU parallel would run it, through ``run_task.py``, but in local mode: one at a
time, in the caller's session, and killed by process tree.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
from collections.abc import Sequence
from typing import Any

from experiments.state_prep_certification.common import store, tasks

RUN_TASK = pathlib.Path(__file__).resolve().with_name("run_task.py")


def run_locally(
    label_dir: pathlib.Path, task_list: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Write the tasks as pass 1 of a fresh label directory, run them, and return their rows."""
    label_dir.mkdir(parents=True, exist_ok=True)
    tasks.write_pass1(label_dir, task_list)
    for task in task_list:
        command = [
            sys.executable,
            str(RUN_TASK),
            "--label-dir",
            str(label_dir),
            "--local",
            task["key"],
        ]
        subprocess.run(command, check=True)
    final, _ = store.final_rows(label_dir)
    return final
