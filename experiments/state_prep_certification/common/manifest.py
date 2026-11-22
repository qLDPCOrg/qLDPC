# SPDX-License-Identifier: Apache-2.0

"""The label of a run, and the version manifest of the machine and software that made it.

The label of an experiment is ``<experiment>-`` followed by the first 12 hexadecimal digits of the
SHA-256 of ``pins.json``, the experiment's frozen files, the harness commit, and, in a bundle, the
SHA-256 of the bundle's resolved requirements.  Any change to them gives a new label, and so a new
results directory.  ``run_node.py`` names the directory by the label, followed by these suffixes,
in this order:

* ``-dirty-<hash>`` if the qLDPC tree has uncommitted changes to tracked files (allowed only in
  local runs), where ``<hash>`` is the first 8 hexadecimal digits of the SHA-256 of ``git diff HEAD
  --binary``, so that a local run of changed code never resumes in the directory of other changes
  to tracked files (untracked files do not enter the hash); the manifest records the full SHA-256
  of the diff;
* ``-local`` for a ``--local`` run, so that its rows never mix with those of the node;
* ``-<subset>`` for a subset other than ``main`` or ``acceptance``.

The harness commit is the Git commit of the qLDPC tree that holds the experiments.  On the node,
whose tree is a ``git archive`` without history, ``bundle.py`` records it in
``experiments/state_prep_certification/bundle.json``, with the SHA-256 of the bundle's
requirements; ``bundle_record`` reads that record.  ``compute_label`` reads no file: its callers
pass the contents of the files and the requirements' SHA-256.  An analysis that checks the label of
a results archive passes the requirements' SHA-256 that the archive's ``manifest.json`` records.
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
from collections.abc import Mapping, Sequence
from typing import Any

from experiments.state_prep_certification.common import store

FT_STATE_PREP_DIR = pathlib.Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = FT_STATE_PREP_DIR.parents[1]
PINS_FILE = FT_STATE_PREP_DIR / "pins.json"
BUNDLE_RECORD = FT_STATE_PREP_DIR / "bundle.json"

# The environment variables that the manifest records, with their values: those that can change
# what a run computes, how fast, or which code it imports.  The values enter manifest.json,
# starts.jsonl, and the archive as they are, so check them before sharing an archive.
RECORDED_PREFIXES = ("QLDPC_", "NUMBA_", "PYTHON", "OMP_", "MKL_", "OPENBLAS_", "VECLIB_")
RECORDED_NAMES = ("LD_PRELOAD", "LD_LIBRARY_PATH")


####################################################################################################
# label


def bundle_record() -> dict[str, Any] | None:
    """The record ``bundle.json`` of the bundle that holds this tree, or None outside a bundle."""
    if not BUNDLE_RECORD.exists():
        return None
    record: dict[str, Any] = json.loads(BUNDLE_RECORD.read_text())
    return record


def harness_commit() -> dict[str, Any]:
    """The commit of the harness, and whether the tree differs from it.

    ``dirty`` says whether tracked files have uncommitted changes, staged or not, and
    ``diff_sha256`` is then the SHA-256 of ``git diff HEAD --binary``, which covers them, without
    any external diff, textconv filter, or color that a user's Git configuration may set.
    """
    record = bundle_record()
    if record is not None:
        return {
            "commit": record["harness_commit"],
            "dirty": False,
            "diff_sha256": None,
            "source": "bundle.json",
        }
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    diff = None
    if dirty:
        diff = subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "diff",
                "HEAD",
                "--binary",
                "--no-ext-diff",
                "--no-textconv",
                "--no-color",
            ],
            capture_output=True,
            check=True,
        ).stdout
    return {
        "commit": commit,
        "dirty": dirty,
        "diff_sha256": None if diff is None else hashlib.sha256(diff).hexdigest(),
        "source": "git",
    }


def compute_label(
    experiment: str,
    files: Sequence[tuple[str, bytes]],
    commit: str,
    requirements_sha256: str | None,
) -> str:
    """The label of a run: its pins, frozen files, harness commit, and a bundle's requirements.

    Args:
        experiment: The experiment's name.
        files: The names and contents of ``pins.json`` and the experiment's frozen files, in order,
            as ``label_files`` gives them.
        commit: The harness commit.
        requirements_sha256: The SHA-256 of the bundle's resolved requirements, or None for a run
            outside a bundle.
    """
    digest = hashlib.sha256()
    for name, data in files:
        digest.update(name.encode() + b"\0" + data + b"\0")
    digest.update(commit.encode())
    if requirements_sha256 is not None:
        digest.update(requirements_sha256.encode())
    return f"{experiment}-{digest.hexdigest()[:12]}"


def label_files(frozen_files: Sequence[pathlib.Path]) -> list[tuple[str, bytes]]:
    """The names and contents of ``pins.json`` and the frozen files, for ``compute_label``."""
    return [(path.name, path.read_bytes()) for path in (PINS_FILE, *frozen_files)]


def load_pins() -> dict[str, Any]:
    """The contents of ``pins.json``."""
    pins: dict[str, Any] = json.loads(PINS_FILE.read_text())
    return pins


####################################################################################################
# manifest


def record_start(
    label_dir: pathlib.Path, results_dir: pathlib.Path, fields: dict[str, Any]
) -> None:
    """Append a start's line to ``starts.jsonl``, before the start runs the acceptance tests.

    Every start that passes the refusals of its first step records itself, whether it is a dry run,
    runs only the acceptance tests, or stops at a later check, so that the analyses see every
    environment in which the label's acceptance tests ran (see ``store.allowed_environment``).  The
    line holds the machine, the software, the recorded environment, the optimization level of this
    interpreter or of the Python processes that it starts, whichever is higher (``python_optimize``,
    ``optimize_level``), and ``fields``, such as the
    start's host, CPU model, subset, the names that ``--allow-env`` accepted, and whether it is a
    dry run (``dry_run``).  It is appended in one write.

    Args:
        label_dir: The label's directory.
        results_dir: The results directory, which may hold the record of ``node.sh``'s preflight,
            ``preflight-<bundle>.json``.
        fields: Further fields of the start.
    """
    bundle = bundle_record()
    preflight = _read_json(results_dir / f"preflight-{bundle['name']}.json") if bundle else None
    start = {
        "label": label_dir.name,
        "pins": load_pins(),
        "harness": harness_commit(),
        "bundle": bundle,
        "preflight": preflight,
        "machine": machine_info(),
        "python": sys.version,
        "executable": sys.executable,
        "python_optimize": optimize_level(),
        "packages": package_freeze(),
        "environment": recorded_environment(os.environ),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **fields,
    }
    with open(label_dir / store.STARTS_FILE, "ab", buffering=0) as file:
        file.write((json.dumps(start, sort_keys=True) + "\n").encode())  # one write


def write_manifest(label_dir: pathlib.Path, extra: dict[str, Any]) -> None:
    """Add ``extra``, such as the job slots, to this start's line, and write ``manifest.json``.

    This start's line is the last of ``starts.jsonl``, since the start holds the label's lock (see
    ``run_node.py``); ``starts.jsonl`` is rewritten whole or not at all.  ``manifest.json``, the
    completed line, is written whole or not at all at the label's first start that is not a dry
    run, so that it, and the CPU model that later starts must match, describe the first start that
    went on to run tasks.
    """
    path = label_dir / store.STARTS_FILE
    *earlier, last = path.read_text().splitlines()
    start = json.loads(last) | extra
    lines = [*earlier, json.dumps(start, sort_keys=True)]
    store.write_atomic(path, "".join(line + "\n" for line in lines))
    manifest = label_dir / "manifest.json"
    if not start.get("dry_run") and not manifest.exists():
        store.write_atomic(manifest, json.dumps(start, indent=1, sort_keys=True) + "\n")


def optimize_level(environ: Mapping[str, str] | None = None) -> int:
    """The larger of this interpreter's optimization level and that of a Python that it starts.

    A child started with ``environ`` (by default this process's) reads ``PYTHONOPTIMIZE`` even if
    this interpreter ignored it, as under -E, so the child is asked for its level.
    """
    environ = os.environ if environ is None else environ
    child = subprocess.run(
        [sys.executable, "-c", "import sys; print(sys.flags.optimize)"],
        env=dict(environ),
        capture_output=True,
        text=True,
        check=False,
    )
    if child.returncode != 0 or not child.stdout.split():
        sys.exit(f"a Python started with this environment fails:\n{child.stderr}")
    return max(sys.flags.optimize, int(child.stdout.split()[-1]))


def recorded_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """The variables of ``RECORDED_PREFIXES`` and ``RECORDED_NAMES`` in an environment."""
    return {
        name: value
        for name, value in sorted(environ.items())
        if name.startswith(RECORDED_PREFIXES) or name in RECORDED_NAMES
    }


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
