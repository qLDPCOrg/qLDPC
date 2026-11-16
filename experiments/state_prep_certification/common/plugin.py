# SPDX-License-Identifier: Apache-2.0

"""The plug-in interface of the experiments, and the loader of plug-ins.

Each experiment is a subdirectory with an ``experiment.py`` module that defines:

* ``NAME``: the experiment's name, which is also its subdirectory;
* ``EXTRA_WHEELS``: requirements beyond qLDPC's own, which ``bundle.py`` fetches;
* ``DATA``: directories of the experiment, relative to it, that are not committed but travel in the
  offline bundle, such as built input files;
* ``FROZEN_FILES``: files of the experiment, relative to it, that enter the label;
* ``JOB_MEM_GIB``: the memory cap of a run, which sets how many job slots fit on a machine;
* ``prepare(label_dir)``: generate or load the inputs, and check them against the frozen files;
* ``tasks(label_dir, subset)``: the tasks of pass 1 of a subset, such as ``"main"`` or ``"smoke"``;
* ``worst_case_s(task)``: the worst-case CPU seconds of a task of pass 1, with its repeats;
* ``repeat_tasks(task, row)``: the tasks of pass 2 that a task of pass 1 and its row call for;
* ``run(task, label_dir)``: run one task in the child process, calling ``runner.timed`` once;
* ``acceptance(node, results_dir)``: run the acceptance tests, and return whether they passed.

An experiment may also give a ``qldpc_commit`` in its ``config.json``; ``bundle.py`` refuses a
bundle if it differs from the pinned commit of ``pins.json``.

The shared code knows nothing else about an experiment: its rows carry an ``experiment`` field and
a free-form ``result`` object beside the shared fields.
"""

from __future__ import annotations

import importlib
import pathlib
import types

EXPERIMENTS_DIR = pathlib.Path(__file__).resolve().parents[1]

# The experiments, by name, and the module of each plug-in.  ``selftest`` is the stand-in plug-in
# with which ``common/selftest.py`` tests the shared code.
PLUGINS = {
    "selftest": "experiments.state_prep_certification.common.selftest_plugin",
}

# The experiments that run on the node; ``run_node.py`` also accepts ``selftest``, for tests.
EXPERIMENTS = sorted(name for name in PLUGINS if name != "selftest")


def load_experiment(name: str) -> types.ModuleType:
    """Import the plug-in of an experiment."""
    if name not in PLUGINS:
        raise ValueError(f"Unknown experiment {name!r}; choose one of {sorted(PLUGINS)}")
    return importlib.import_module(PLUGINS[name])


def experiment_dir(name: str) -> pathlib.Path:
    """The directory of an experiment."""
    return EXPERIMENTS_DIR / name
