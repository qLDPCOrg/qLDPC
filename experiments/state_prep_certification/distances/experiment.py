# SPDX-License-Identifier: Apache-2.0

"""The plug-in of the distance experiment (see ``common/plugin.py`` for the interface).

Each task is one arm on one code, in mode ``timed`` or ``count``.  A timed run measures the wall
time T of the arm's distance call on its stored code, including building the code object; a
counting run makes the same call with qLDPC's opt-in work counters on, untimed for the analysis
but under the same limit.  Counting tasks exist only if the installed qLDPC has the counters.

Before timing, each run warms up: ``trial.warm_up`` for the ``distance`` and ``hadamard``
workloads, so that every route is compiled, and then the arm itself, untimed, on the
matrices of ``SurfaceCode(5)`` (or its Hadamard variant for a code that is not CSS), with its own
tuning.  A timed run that finishes in under ``repeat_below_s`` gets ``repeats`` more runs.
"""

from __future__ import annotations

import json
import pathlib
from typing import TYPE_CHECKING, Any

from experiments.state_prep_certification.common import runner
from experiments.state_prep_certification.distances import arms, codes

# Importing ``codes`` above puts ``experiments/distance_tuning`` on the path, whose scripts import
# each other as top-level modules; mypy knows them only as experiments.distance_tuning.*.
if TYPE_CHECKING:
    from experiments.distance_tuning import trial
else:
    import trial

NAME = "distances"
EXTRA_WHEELS = ["matplotlib"]
DATA: list[str] = []
FROZEN_FILES = ["config.json", "endpoints.json", "sample.json"]
HERE = pathlib.Path(__file__).resolve().parent
CONFIG = arms.CONFIG
JOB_MEM_GIB = float(CONFIG["mem_cap_gib"])

# The codes of the ``smoke`` subset: small anchors of each kind.
SMOKE = ("golay", "bb72", "surface9", "five_five")


def prepare(label_dir: pathlib.Path) -> None:
    """Regenerate the codes into the label's directory, and stop if any differs from the sample."""
    sample = codes.generate(label_dir)
    problems = codes.compare_with_sample(sample)
    if problems:
        raise SystemExit("The regenerated codes differ from sample.json:\n" + "\n".join(problems))


def tasks(label_dir: pathlib.Path, subset: str) -> list[dict[str, Any]]:
    """One timed task and, if counters exist, one counting task per arm and code."""
    sample = json.loads(codes.SAMPLE.read_text())
    entries = sample["anchors"] + sample["codes"]
    if subset == "smoke":
        entries = [entry for entry in sample["anchors"] if entry["name"] in SMOKE]
    elif subset != "main":
        raise ValueError(f"Unknown subset {subset!r}; choose main or smoke")
    modes = ["timed", "count"] if arms.counters_available() else ["timed"]
    task_list = []
    for entry in entries:
        for arm in arms.ARMS:
            for mode in modes:
                task_list.append(_task(arm, mode, entry, repeat=0))
    return task_list


def worst_case_s(task: dict[str, Any]) -> float:
    """The worst-case CPU seconds of a task with its repeats.

    A run costs at most its start-up cap, the limit L, the runner's window ``runner.FINISH_S`` for
    returning a result, and then the grace in which its processes must end.  The worst timed task
    has a first run just under the repeat threshold and ``repeats`` repeats that each cost the most
    that a run can.
    """
    startup, limit = CONFIG["startup_cap_s"], task["limit_s"]
    after = runner.FINISH_S + CONFIG["grace_s"]
    if task["mode"] == "count":
        return float(startup + limit + after)
    first = startup + CONFIG["repeat_below_s"] + after
    return float(first + CONFIG["repeats"] * (startup + limit + after))


def repeat_tasks(task: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    """The repeats of a timed run that finished in under ``repeat_below_s``."""
    if task["mode"] != "timed" or row["status"] != "ok" or row["T"] >= CONFIG["repeat_below_s"]:
        return []
    return [
        dict(task, key=task["key"].rsplit(".r", 1)[0] + f".r{repeat}", repeat=repeat)
        for repeat in range(1, CONFIG["repeats"] + 1)
    ]


def run(task: dict[str, Any], label_dir: pathlib.Path) -> dict[str, Any]:
    """Warm up, then run the arm's distance call once, timed (and counting in mode ``count``)."""
    from qldpc.codes.distance import _tuning

    arm, entry = task["arm"], task["input"]
    stored = codes.load_code(label_dir, entry["code_hash"])

    # warm up every route, then the arm itself on a small code, with its own tuning
    for kind in ("distance", "hadamard"):
        trial.warm_up(kind)
    with _tuning.use_tuning(**arms.tuning_changes(arm)):
        arms.distance_call(arms.warm_up_arrays(entry["css"]))()
        tuning = arms.active_tuning()
        call = arms.distance_call(stored)
        if task["mode"] == "count":
            with arms.counting() as counts:
                distance = runner.timed(call)
            work: Any = counts
        else:
            distance = runner.timed(call)
            work = None
    return {
        "name": entry["name"],
        "code_hash": entry["code_hash"],
        "n": entry["n"],
        "k": entry["k"],
        "css": entry["css"],
        "equal_distance_xz": entry["equal_distance_xz"],
        "distance": distance,
        "mode": task["mode"],
        "tuning": tuning,
        "work": work,
    }


def acceptance(node: bool, results_dir: pathlib.Path) -> bool:
    """Run the acceptance tests of ``acceptance.py``."""
    import experiments.state_prep_certification.distances.acceptance as tests

    return tests.run_all(node=node, results_dir=results_dir)


def _task(arm: str, mode: str, entry: dict[str, Any], repeat: int) -> dict[str, Any]:
    """A task of one arm on one code."""
    return {
        "key": f"{arm}.{mode}.{entry['code_hash'][:12]}.r{repeat}",
        "experiment": NAME,
        "arm": arm,
        "mode": mode,
        "repeat": repeat,
        "limit_s": float(CONFIG["limit_s"]),
        "mem_cap_gib": float(CONFIG["mem_cap_gib"]),
        "startup_cap_s": float(CONFIG["startup_cap_s"]),
        "grace_s": float(CONFIG["grace_s"]),
        "input": {
            key: entry[key] for key in ("name", "code_hash", "css", "equal_distance_xz", "n", "k")
        },
    }
