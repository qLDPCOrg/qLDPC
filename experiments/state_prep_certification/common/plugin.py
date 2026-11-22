# SPDX-License-Identifier: Apache-2.0

"""The plug-in interface of the experiments, and the loader of plug-ins.

Each experiment is a subdirectory with an ``experiment.py`` module that defines:

* ``NAME``: the experiment's name, which is also its subdirectory;
* ``EXTRA_WHEELS``: requirements beyond qLDPC's own, which ``bundle.py`` fetches;
* ``DATA``: directories of the experiment, relative to it, that are not committed but travel in the
  offline bundle, such as built input files;
* ``FROZEN_FILES``: files of the experiment, relative to it, that enter the label;
* ``JOB_MEM_GIB``: the memory of a job slot, which sets how many job slots fit on a machine, and
  which must be at least the ``mem_cap_gib`` of every task of pass 1, as ``run_node.py`` checks;
* ``prepare(label_dir)``: generate or load the inputs, and check them against the frozen files;
* ``tasks(label_dir, subset)``: the tasks of pass 1 of a subset, such as ``"main"`` or ``"smoke"``;
* ``worst_case_s(task)``: the worst-case CPU seconds of a task of pass 1, with its repeats, but
  without the runner's retries, which ``tasks.ceiling_hours`` adds;
* ``repeat_tasks(task, row)``: the tasks of pass 2 that a task of pass 1 and its row call for;
* ``run(task, label_dir)``: run one task in the child process, calling ``runner.timed`` once;
* ``acceptance(node, results_dir)``: run the acceptance tests, writing their runs into
  ``results_dir``, a fresh directory of each attempt, and return whether they passed.

The shared code knows nothing else about an experiment: its rows carry an ``experiment`` field and
a free-form ``result`` object beside the shared fields.
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

# The directory of the experiments, the root of the qLDPC tree that holds it, and the script that
# runs one task
EXPERIMENTS_DIR = pathlib.Path(__file__).resolve().parents[1]
ROOT = EXPERIMENTS_DIR.parents[1]
RUN_TASK = EXPERIMENTS_DIR / "common" / "run_task.py"

# The experiments, by name, and the module of each plug-in.  ``selftest`` is the stand-in plug-in
# with which ``common/selftest.py`` tests the shared code.
PLUGINS = {
    "certification": "experiments.state_prep_certification.certification.experiment",
    "distances": "experiments.state_prep_certification.distances.experiment",
    "ranking": "experiments.state_prep_certification.ranking.experiment",
    "selftest": "experiments.state_prep_certification.common.selftest_plugin",
}

# The experiments that run on the node; ``run_node.py`` also accepts ``selftest``, for tests.
EXPERIMENTS = sorted(name for name in PLUGINS if name != "selftest")


def load_experiment(name: str) -> types.ModuleType:
    """Import the plug-in of an experiment, and check that its ``NAME`` is ``name``."""
    if name not in PLUGINS:
        raise ValueError(f"Unknown experiment {name!r}; choose one of {sorted(PLUGINS)}")
    module = importlib.import_module(PLUGINS[name])
    if module.NAME != name:
        raise ValueError(f"The plug-in of {name!r} is named {module.NAME!r}")
    return module


def experiment_dir(name: str) -> pathlib.Path:
    """The directory of an experiment."""
    return EXPERIMENTS_DIR / name


def run_task_command(label_dir: pathlib.Path, key: str, local: bool) -> list[str]:
    """The command that runs one task of a label, in local mode if ``local``."""
    local_option = ["--local"] if local else []
    return [sys.executable, str(RUN_TASK), "--label-dir", str(label_dir), *local_option, key]
