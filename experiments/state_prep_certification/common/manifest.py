# SPDX-License-Identifier: Apache-2.0

"""The label of a run, and the version manifest of the machine and software that made it.

The label of an experiment is ``<experiment>-`` followed by the first 12 hexadecimal digits of the
SHA-256 of ``pins.json``, the experiment's frozen files, the harness commit, and, in a bundle, the
SHA-256 of the bundle's resolved requirements.  Any change to them gives a new label, and so a new
results directory.  ``run_node.py`` names the directory by the label, followed by ``-dirty`` if the
qLDPC tree has uncommitted changes to tracked files (allowed only in local runs), and by
``-<subset>`` for a subset other than ``main`` or ``acceptance``.

The harness commit is the Git commit of the qLDPC tree that holds the experiments.  On the node,
whose tree is a ``git archive`` without history, ``bundle.py`` records it in
``experiments/state_prep_certification/bundle.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

FT_STATE_PREP_DIR = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = FT_STATE_PREP_DIR.parents[1]
PINS_FILE = FT_STATE_PREP_DIR / "pins.json"
BUNDLE_RECORD = FT_STATE_PREP_DIR / "bundle.json"


####################################################################################################
# label


def harness_commit() -> dict[str, Any]:
    """The commit of the harness, and whether the tree differs from it (``dirty``)."""
    if BUNDLE_RECORD.exists():
        record = json.loads(BUNDLE_RECORD.read_text())
        return {"commit": record["harness_commit"], "dirty": False, "source": "bundle.json"}
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"commit": commit, "dirty": dirty, "source": "git"}


def compute_label(experiment: str, frozen_files: Sequence[pathlib.Path], commit: str) -> str:
    """The label of a run: its pins, frozen files, harness commit, and a bundle's requirements."""
    digest = hashlib.sha256()
    for path in (PINS_FILE, *frozen_files):
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    digest.update(commit.encode())
    if BUNDLE_RECORD.exists():
        digest.update(json.loads(BUNDLE_RECORD.read_text())["requirements_sha256"].encode())
    return f"{experiment}-{digest.hexdigest()[:12]}"


def load_pins() -> dict[str, Any]:
    """The contents of ``pins.json``."""
    pins: dict[str, Any] = json.loads(PINS_FILE.read_text())
    return pins


####################################################################################################
# manifest


def write_manifest(
    label_dir: pathlib.Path, results_dir: pathlib.Path, extra: dict[str, Any]
) -> None:
    """Write ``manifest.json`` at a label's first start, and record every start in ``starts.jsonl``.

    Args:
        label_dir: The label's directory.
        results_dir: The results directory, which may hold ``preflight.json`` from ``node.sh``.
        extra: Further fields, such as the number of job slots.
    """
    manifest = {
        "label": label_dir.name,
        "pins": load_pins(),
        "harness": harness_commit(),
        "bundle": json.loads(BUNDLE_RECORD.read_text()) if BUNDLE_RECORD.exists() else None,
        "preflight": _read_json(results_dir / "preflight.json"),
        "machine": machine_info(),
        "python": sys.version,
        "executable": sys.executable,
        "packages": package_freeze(),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    path = label_dir / "manifest.json"
    if not path.exists():
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    with open(label_dir / "starts.jsonl", "a") as file:
        file.write(json.dumps(manifest, sort_keys=True) + "\n")


def machine_info() -> dict[str, Any]:
    """The operating system, CPU model, physical cores, and memory of this machine."""
    info: dict[str, Any] = {
        "system": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
    }
    if sys.platform.startswith("linux"):
        info["lscpu"] = _run(["lscpu"])
        info["physical_cores"] = len(
            {
                tuple(line.split(",")[1:3])
                for line in _run(["lscpu", "-p=CPU,CORE,SOCKET"]).splitlines()
                if line and not line.startswith("#")
            }
        )
        info["mem_total_bytes"] = mem_total_bytes()
    elif sys.platform == "darwin":
        info["cpu"] = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        info["physical_cores"] = int(_run(["sysctl", "-n", "hw.physicalcpu"]) or 0)
        info["mem_total_bytes"] = mem_total_bytes()
    return info


def mem_total_bytes() -> int:
    """The machine's total memory, in bytes."""
    if sys.platform.startswith("linux"):
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return 1024 * int(line.split()[1])
    return int(_run(["sysctl", "-n", "hw.memsize"]) or 0)


def package_freeze() -> str:
    """The installed packages of this interpreter, from ``uv pip freeze`` or ``pip freeze``."""
    output = _run(["uv", "pip", "freeze", "--offline", "--python", sys.executable])
    return output or _run([sys.executable, "-m", "pip", "freeze"])


def _git(*args: str) -> str:
    """The output of a Git command in the repository."""
    return subprocess.run(
        ["git", "-C", str(REPOSITORY_ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _run(command: list[str]) -> str:
    """The output of a command, or an empty string if it fails."""
    try:
        return subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _read_json(path: pathlib.Path) -> Any:
    """The contents of a JSON file, or None if it does not exist."""
    return json.loads(path.read_text()) if path.exists() else None
