# SPDX-License-Identifier: Apache-2.0

"""The plug-in of the distance experiment (see ``common/plugin.py`` for the interface).

Each task runs one arm on one code; a run measures the wall time T of the arm's distance call on
its stored code, including building the code object.

No run warms up before timing; the runner retries any run whose measured call compiled
(``common/runner.py``).  A run that finishes in under ``repeat_below_s`` gets ``repeats`` more runs.
"""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any

from experiments.state_prep_certification.common import runner
from experiments.state_prep_certification.distances import arms, codes

NAME = "distances"
EXTRA_WHEELS = ["matplotlib"]
DATA = ["code_data"]
FROZEN_FILES = ["config.json", "code_data/endpoints.json", "code_data/sample.json"]
HERE = pathlib.Path(__file__).resolve().parent
CONFIG = arms.CONFIG
JOB_MEM_GIB = float(CONFIG["mem_cap_gib"])

# The codes of the ``smoke`` subset: small anchors of each kind.
SMOKE = ("golay", "bb72", "surface9", "five_five")


def prepare(label_dir: pathlib.Path) -> None:
    """Regenerate the codes into the label directory, and stop if any differs from the sample.

    Then copy the sample into the label directory, where ``analyze.py`` reads it, so that the run
    keeps the sample that it ran with.
    """
    sample = codes.generate(label_dir)
    problems = codes.compare_with_sample(sample)
    if problems:
        raise SystemExit("The regenerated codes differ from sample.json:\n" + "\n".join(problems))
    shutil.copyfile(codes.SAMPLE, label_dir / codes.SAMPLE.name)


def tasks(label_dir: pathlib.Path, subset: str) -> list[dict[str, Any]]:
    """One task per arm and code."""
    sample = json.loads(codes.SAMPLE.read_text())
    entries = sample["anchors"] + sample["codes"]
    if subset == "smoke":
        entries = [entry for entry in sample["anchors"] if entry["name"] in SMOKE]
    elif subset != "main":
        raise ValueError(f"Unknown subset {subset!r}; choose main or smoke")
    return [_task(arm, entry, repeat=0) for entry in entries for arm in arms.ARMS]


def worst_case_s(task: dict[str, Any]) -> float:
    """The worst-case CPU seconds of a task with its repeats.

    A run costs at most its start-up cap, the limit L, the runner's window ``runner.FINISH_S`` for
    returning a result, and then the grace in which its processes must end.  A task's worst case has
    a first run just under the repeat threshold and ``repeats`` repeats that each cost the most that
    a run can.
    """
    startup, limit = CONFIG["startup_cap_s"], task["limit_s"]
    after = runner.FINISH_S + CONFIG["grace_s"]
    first = startup + CONFIG["repeat_below_s"] + after
    return float(first + CONFIG["repeats"] * (startup + limit + after))


def repeat_tasks(task: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    """The repeats of a run that finished in under ``repeat_below_s``."""
    if row["status"] != "ok" or row["T"] >= CONFIG["repeat_below_s"]:
        return []
    return [
        dict(task, key=task["key"].rsplit(".r", 1)[0] + f".r{repeat}", repeat=repeat)
        for repeat in range(1, CONFIG["repeats"] + 1)
    ]


def run(task: dict[str, Any], label_dir: pathlib.Path) -> dict[str, Any]:
    """Run the arm's distance call once, timed."""
    from qldpc.codes.distance import _tuning

    arm, entry = task["arm"], task["input"]
    stored = codes.load_code(label_dir, entry["code_hash"])
    with _tuning.use_tuning(**arms.tuning_changes(arm)):
        tuning = arms.active_tuning()
        distance = runner.timed(arms.distance_call(stored))
    return {
        "name": entry["name"],
        "code_hash": entry["code_hash"],
        "n": entry["n"],
        "k": entry["k"],
        "css": entry["css"],
        "equal_distance_xz": entry["equal_distance_xz"],
        "distance": distance,
        "tuning": tuning,
    }


def acceptance(node: bool, results_dir: pathlib.Path) -> bool:
    """Run the acceptance tests of ``acceptance.py``."""
    import experiments.state_prep_certification.distances.acceptance as tests

    return tests.run_all(node=node, results_dir=results_dir)


def _task(arm: str, entry: dict[str, Any], repeat: int) -> dict[str, Any]:
    """A task of one arm on one code."""
    return {
        "key": f"{arm}.timed.{entry['code_hash'][:12]}.r{repeat}",
        "experiment": NAME,
        "arm": arm,
        "mode": "timed",
        "repeat": repeat,
        "limit_s": float(CONFIG["limit_s"]),
        "mem_cap_gib": float(CONFIG["mem_cap_gib"]),
        "startup_cap_s": float(CONFIG["startup_cap_s"]),
        "grace_s": float(CONFIG["grace_s"]),
        "input": {
            key: entry[key] for key in ("name", "code_hash", "css", "equal_distance_xz", "n", "k")
        },
    }
