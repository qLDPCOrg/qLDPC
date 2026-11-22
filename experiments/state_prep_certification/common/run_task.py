# SPDX-License-Identifier: Apache-2.0

"""Run one task of a label: the command that GNU parallel invokes once per task.

Everything in ``state_prep_certification/`` is a prototype, subject to change: its code, interfaces,
inputs, outputs, and documentation.

    python experiments/state_prep_certification/common/run_task.py --label-dir D [--local] \
        [--job-slot S] [--cpu C] KEY

It exits at once with status 0 if ``D/runs.jsonl`` already has a final row for the task.  Otherwise
it runs the task in one or two fresh child processes (see ``runner.py``), under the runner's
limits, appends one final row, and exits with status 0; it exits with a nonzero status only if it
wrote no final row, as after SIGTERM.  With ``--cpu``, each child pins itself to that logical CPU
(Linux only).  With ``--child``, this script is one of those child processes, which the runner
starts.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.state_prep_certification.common import plugin, runner, store, tasks


def main() -> None:
    """Run the task, or be its child process."""
    args = parse_args()
    label_dir = args.label_dir.resolve()
    task = tasks.load_tasks(label_dir)[args.task_key]
    if args.child:
        runner.pin_child()
        module = plugin.load_experiment(task["experiment"])
        sys.exit(runner.child_main(task, label_dir, module.run))
    final, _ = store.final_rows(label_dir)
    if args.task_key in final:
        sys.exit(0)
    sys.exit(
        runner.run_task(label_dir, task, local=args.local, job_slot=args.job_slot, cpu=args.cpu)
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("task_key", help="the key of the task, as in tasks.json")
    parser.add_argument("--label-dir", type=pathlib.Path, required=True, help="the label directory")
    parser.add_argument(
        "--local",
        action="store_true",
        help="keep the run in this session and kill it by process tree (workstation tests)",
    )
    parser.add_argument("--job-slot", type=int, help="GNU parallel's job slot of this task")
    parser.add_argument("--cpu", type=int, help="the logical CPU to pin the run to (Linux)")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    main()
