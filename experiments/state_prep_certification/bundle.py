# SPDX-License-Identifier: Apache-2.0

"""Build the offline bundle that carries the experiments of this directory to the node.

    python experiments/state_prep_certification/bundle.py EXPERIMENT... --out DIR [--unfrozen]

The node has no practical network access, so everything that it needs travels in one file,
``DIR/spc-bundle-<id>.tar``, with a ``.sha256`` file beside it, where ``<id>`` hashes the pins,
the harness commit, the experiments, the platforms, and the resolved requirements.  The tar holds
one directory of the same name, with:

* ``qLDPC/``: ``git archive`` of this worktree's HEAD, with the experiments' ``DATA`` directories
  (inputs built on the workstation and not committed) and
  ``experiments/state_prep_certification/bundle.json``;
* ``wheels/``: wheels for the pinned CPython and Linux platform of ``pins.json``.  They cover
  qLDPC's runtime requirements with the extras of ``qldpc_extras``, each experiment's
  ``EXTRA_WHEELS``, and qLDPC itself, built from the archive;
* ``requirements-node.txt``: the resolved requirements with exact versions and hashes, including
  qLDPC's own wheel;
* ``python/``: the pinned python-build-standalone CPython;
* ``parallel/``: the pinned GNU parallel release, whose ``src/parallel`` is a Perl script;
* ``SHA256SUMS`` over every file, and ``bundle.json``: the bundle's name, the commits, the platform,
  every command run, and every fallback taken.

``bundle.json`` records the SHA-256 of the resolved requirements, which enters the label of every
run on the node, so that a label fixes the versions of the third-party packages too.

A requirement without a wheel for the platform is built into a wheel here, from its sdist checked
against the resolution's hash, if it is pure Python (``py3-none-any`` or ``py2.py3-none-any``); the
lockfile's hash for it is then replaced by the hash of that wheel, since an install with
``--require-hashes`` would otherwise reject it.  Any other requirement without a wheel stops the
script.

The script refuses a named experiment whose frozen files (``FROZEN_FILES``) are not all in the
staged tree, before any download, so that an experiment whose frozen files lie in another
experiment's directory needs that experiment in the bundle too.  It refuses a worktree with
uncommitted changes to tracked files.  It refuses an unpinned qLDPC commit (``qldpc_commit`` of
``pins.json`` is null) unless ``--unfrozen`` asks for a trial, which ``bundle.json`` then records.
With a pinned commit, it checks that HEAD descends from that commit and that ``src/`` and
``pyproject.toml`` are unchanged since, so that the bundle's source, its qLDPC wheel, and the run's
manifest name one qLDPC commit.

Downloads are cached in ``DIR/cache``.  ``--uv-python-platform`` and ``--wheel-platform`` build a
bundle for another platform.  ``node.sh`` itself runs only on x86-64 Linux, but a bundle for the
workstation's platform can rehearse its install commands there.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from experiments.state_prep_certification.common import manifest, plugin, store

ROOT = pathlib.Path(__file__).resolve().parents[2]
COMMANDS: list[list[str]] = []
FALLBACKS: list[str] = []


def main() -> None:
    """Build the bundle."""
    args = parse_args()
    pins = manifest.load_pins()
    platform = args.uv_python_platform or pins["platform"]["uv_python_platform"]
    wheel_platforms = args.wheel_platform or pins["platform"]["wheel_platforms"]
    modules = [plugin.load_experiment(name) for name in args.experiments]
    commit = check_tree(pins, modules, args.unfrozen)

    # build in a staging directory, named once the requirements are resolved
    out = args.out.resolve()
    cache = out / "cache"
    staging = out / "staging"
    if staging.exists():
        shutil.rmtree(staging)
    for directory in (staging / "wheels", staging / "python", staging / "parallel", cache):
        directory.mkdir(parents=True, exist_ok=True)

    # qLDPC/: the archive of HEAD, the experiments' data, and the bundle record
    tree = staging / "qLDPC"
    tree.mkdir()
    archive = subprocess.run(
        ["git", "-C", str(ROOT), "archive", "HEAD"], capture_output=True, check=True
    )
    COMMANDS.append(["git", "archive", "HEAD"])
    with tarfile.open(fileobj=io.BytesIO(archive.stdout), mode="r|") as tar:
        tar.extractall(tree, filter="data")
    for module in modules:
        for data in module.DATA:
            source = plugin.experiment_dir(module.NAME) / data
            if not source.is_dir():
                sys.exit(f"{module.NAME}: its data directory {source} does not exist; build it")
            shutil.copytree(
                source, tree / "experiments" / "state_prep_certification" / module.NAME / data
            )
    check_frozen_files(tree, modules)

    # wheels and the hashed requirements
    qldpc_wheel = build_qldpc_wheel(tree, staging / "wheels")
    requirements = resolve(tree, modules, staging, platform, pins)
    requirements = download_wheels(
        requirements, staging / "wheels", cache, pins["python"]["version"], wheel_platforms
    )
    qldpc_hash = store.sha256_file(qldpc_wheel)
    requirements += f"qldpc=={_wheel_version(qldpc_wheel)} \\\n    --hash=sha256:{qldpc_hash}\n"
    (staging / "requirements-node.txt").write_text(requirements)

    # the bundle's name: a hash of the pins, the harness commit, the experiments, the platforms,
    # and the resolved requirements
    digest = hashlib.sha256(
        json.dumps(
            [pins, commit, args.experiments, platform, wheel_platforms, requirements]
        ).encode()
    ).hexdigest()
    name = f"spc-bundle-{digest[:12]}"

    # CPython and GNU parallel, checked against their pinned hashes
    fetch(
        pins["python"]["standalone_url"],
        pins["python"]["standalone_sha256"],
        staging / "python" / pins["python"]["standalone_asset"],
        cache,
    )
    parallel = pins["gnu_parallel"]
    fetch(
        parallel["url"],
        parallel["sha256"],
        staging / "parallel" / parallel["url"].rsplit("/", 1)[1],
        cache,
    )

    # the records, the checksums, and the tar
    record: dict[str, Any] = {
        "name": name,
        "harness_commit": commit,
        "qldpc_commit": pins["qldpc_commit"],
        "unfrozen": pins["qldpc_commit"] is None,
        "experiments": args.experiments,
        "python": pins["python"]["version"],
        "uv_python_platform": platform,
        "wheel_platforms": wheel_platforms,
        "qldpc_wheel": qldpc_wheel.name,
        "requirements_sha256": hashlib.sha256(requirements.encode()).hexdigest(),
        "commands": [" ".join(command) for command in COMMANDS],
        "fallbacks": FALLBACKS,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (tree / "experiments" / "state_prep_certification" / "bundle.json").write_text(
        json.dumps(record, indent=1) + "\n"
    )
    (staging / "bundle.json").write_text(json.dumps(record, indent=1) + "\n")
    files = sorted(path for path in staging.rglob("*") if path.is_file())
    (staging / "SHA256SUMS").write_text(
        "".join(
            f"{store.sha256_file(path)}  {path.relative_to(staging).as_posix()}\n" for path in files
        )
    )
    bundle = out / f"{name}.tar"
    with tarfile.open(bundle, "w") as tar:
        tar.add(staging, arcname=name)
    shutil.rmtree(staging)
    (out / f"{name}.tar.sha256").write_text(f"{store.sha256_file(bundle)}  {bundle.name}\n")
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "bytes": bundle.stat().st_size,
                "sha256": store.sha256_file(bundle),
                "fallbacks": FALLBACKS,
            },
            indent=1,
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("experiments", nargs="+", choices=plugin.EXPERIMENTS)
    parser.add_argument("--out", type=pathlib.Path, required=True, help="the output directory")
    parser.add_argument(
        "--unfrozen", action="store_true", help="build a trial bundle without a pinned qLDPC commit"
    )
    parser.add_argument("--uv-python-platform", help="another platform for uv, for a rehearsal")
    parser.add_argument(
        "--wheel-platform", action="append", help="another wheel platform tag, for a rehearsal"
    )
    return parser.parse_args()


####################################################################################################
# checks


def check_frozen_files(tree: pathlib.Path, modules: list[Any]) -> None:
    """Stop unless the staged tree holds every frozen file of the named experiments.

    A frozen file in another experiment's directory travels only in a bundle that names that
    experiment too.
    """
    experiments_dir = tree / "experiments" / "state_prep_certification"
    for module in modules:
        for name in module.FROZEN_FILES:
            path = pathlib.Path(os.path.normpath(experiments_dir / module.NAME / name))
            if path.is_file():
                continue
            owner = path.relative_to(experiments_dir).parts[0]
            if owner != module.NAME:
                sys.exit(f"{module.NAME} needs {owner}'s {path.name}: bundle {owner} too")
            sys.exit(f"{module.NAME}: its frozen file {name} is not in the bundle; build it")


def check_tree(pins: dict[str, Any], modules: list[Any], unfrozen: bool) -> str:
    """Check the worktree and the qLDPC pin; return the harness commit."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    if status:
        sys.exit(f"The worktree has uncommitted changes:\n{status}")
    head = _git("rev-parse", "HEAD")
    frozen = pins["qldpc_commit"]
    if frozen is None:
        if not unfrozen:
            sys.exit("pins.json has no pinned qldpc_commit; pass --unfrozen for a trial bundle")
        FALLBACKS.append("trial: no qldpc_commit in pins.json, so not for the node")
        return head
    if subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", frozen, head], check=False
    ).returncode:
        sys.exit(f"HEAD does not descend from the pinned qLDPC commit {frozen}")
    if _git("diff", "--name-only", frozen, head, "--", "src", "pyproject.toml"):
        sys.exit(f"src/ or pyproject.toml changed since the pinned qLDPC commit {frozen}")
    for module in modules:
        config = plugin.experiment_dir(module.NAME) / "config.json"
        if config.exists():
            own = json.loads(config.read_text()).get("qldpc_commit")
            if own not in (None, frozen):
                sys.exit(f"{module.NAME}/config.json pins qLDPC at {own}, pins.json at {frozen}")
    return head


####################################################################################################
# wheels


def build_qldpc_wheel(tree: pathlib.Path, wheels: pathlib.Path) -> pathlib.Path:
    """Build qLDPC's wheel from the archived tree."""
    command = ["uv", "build", "--wheel", "--out-dir", str(wheels), str(tree)]
    _run(command)
    (wheel,) = wheels.glob("qldpc-*.whl")
    return wheel


def resolve(
    tree: pathlib.Path,
    modules: list[Any],
    staging: pathlib.Path,
    platform: str,
    pins: dict[str, Any],
) -> str:
    """Resolve qLDPC's runtime requirements with its pinned extras and the experiments' wheels."""
    python = pins["python"]["version"]
    extras = sorted({wheel for module in modules for wheel in module.EXTRA_WHEELS})
    source = staging / "requirements-node.in"
    source.write_text(
        "# qLDPC's runtime requirements with its pinned extras, and the experiments' extra wheels\n"
        + "-r requirements-qldpc.txt\n"
        + "".join(f"{wheel}\n" for wheel in extras)
    )
    qldpc_requirements = staging / "requirements-qldpc.txt"
    _run(
        [
            "uv",
            "pip",
            "compile",
            str(tree / "pyproject.toml"),
            *[option for extra in pins["qldpc_extras"] for option in ("--extra", extra)],
            "--python-version",
            python,
            "--python-platform",
            platform,
            "--no-header",
            "--no-annotate",
            "-o",
            str(qldpc_requirements),
        ]
    )
    output = staging / "requirements-node.locked.txt"
    _run(
        [
            "uv",
            "pip",
            "compile",
            str(source),
            "--python-version",
            python,
            "--python-platform",
            platform,
            "--generate-hashes",
            "--no-header",
            "-o",
            str(output),
        ]
    )
    text = output.read_text()
    for path in (source, qldpc_requirements, output):
        path.unlink()
    return text


def download_wheels(
    requirements: str,
    wheels: pathlib.Path,
    cache: pathlib.Path,
    python: str,
    platforms: list[str],
) -> str:
    """Download a wheel of every requirement, or build a pure-Python one; return the requirements.

    The hash of a requirement whose wheel was built here is replaced by the hash of that wheel.
    """
    entries = _split_requirements(requirements)
    lines = []
    for name, version, entry in entries:
        single = cache / f"{name}-{version}.txt"
        single.write_text(entry)
        command = [
            "uvx",
            "pip",
            "download",
            "-r",
            str(single),
            "--no-deps",
            "--only-binary=:all:",
            "--implementation",
            "cp",
            "--python-version",
            python,
            *[option for tag in platforms for option in ("--platform", tag)],
            "-d",
            str(wheels),
            "--quiet",
        ]
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            COMMANDS.append(command)
            lines.append(entry)
            continue
        if "No matching distribution" not in result.stderr:
            sys.exit(f"failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}")
        wheel = build_pure_python_wheel(name, version, single, wheels, cache)
        FALLBACKS.append(
            f"{name}=={version}: no wheel for {platforms}; built {wheel.name} from its sdist"
        )
        lines.append(f"{name}=={version} \\\n    --hash=sha256:{store.sha256_file(wheel)}\n")
    return "".join(lines)


def build_pure_python_wheel(
    name: str, version: str, hashed: pathlib.Path, wheels: pathlib.Path, cache: pathlib.Path
) -> pathlib.Path:
    """Build a wheel from a requirement's sdist, and stop unless it is pure Python.

    The sdist is downloaded through the hashed requirement file ``hashed``, so that it must match
    a hash of the resolution.
    """
    sdists = cache / "sdists"
    sdists.mkdir(exist_ok=True)
    _run(
        [
            "uvx",
            "pip",
            "download",
            "-r",
            str(hashed),
            "--no-deps",
            "--no-binary=:all:",
            "-d",
            str(sdists),
            "--quiet",
        ]
    )
    normalized = re.sub(r"[-_.]+", "[-_.]", name)
    (sdist,) = [
        path
        for path in sdists.iterdir()
        if re.match(
            f"{normalized}-{re.escape(version)}\\.(tar\\.gz|zip)$", path.name, re.IGNORECASE
        )
    ]
    built = cache / "built"
    built.mkdir(exist_ok=True)
    _run(["uv", "build", "--wheel", "--out-dir", str(built), str(sdist)])
    (wheel,) = [
        path for path in built.glob("*.whl") if _wheel_name(path) == (_normalized(name), version)
    ]
    python, abi, platform = wheel.name.removesuffix(".whl").split("-")[-3:]
    if "py3" not in python.split(".") or (abi, platform) != ("none", "any"):
        sys.exit(
            f"{name}=={version} has no wheel for the node's platform, and its build "
            f"{wheel.name} is not pure Python: pin another version, or add a wheel platform"
        )
    target = wheels / wheel.name
    shutil.copy2(wheel, target)
    return target


def _split_requirements(requirements: str) -> list[tuple[str, str, str]]:
    """The name, version, and full hashed entry of each requirement of a compiled file."""
    entries: list[tuple[str, str, str]] = []
    current: list[str] = []
    for line in requirements.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if not line.startswith(" ") and current:
            entries.append(_entry(current))
            current = []
        current.append(line)
    if current:
        entries.append(_entry(current))
    return entries


def _entry(lines: list[str]) -> tuple[str, str, str]:
    """The name, version, and text of one compiled requirement."""
    match = re.match(r"([A-Za-z0-9_.-]+)==([^ ;\\]+)", lines[0])
    if match is None:
        raise ValueError(f"Cannot parse the requirement {lines[0]!r}")
    kept = [line for line in lines if not line.lstrip().startswith("#")]
    return match[1], match[2], "\n".join(kept) + "\n"


####################################################################################################
# downloads and helpers


def fetch(url: str, sha256: str, target: pathlib.Path, cache: pathlib.Path) -> None:
    """Download a file once into the cache, check its hash, and copy it to the target."""
    cached = cache / target.name
    if not cached.exists() or store.sha256_file(cached) != sha256:
        COMMANDS.append(["download", url])
        with urllib.request.urlopen(url, timeout=300) as response:
            cached.write_bytes(response.read())
    if store.sha256_file(cached) != sha256:
        sys.exit(f"{url} has SHA-256 {store.sha256_file(cached)}, not the pinned {sha256}")
    shutil.copy2(cached, target)


def _run(command: list[str]) -> None:
    """Run a command, record it, and stop on failure."""
    COMMANDS.append(command)
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        sys.exit(f"failed: {' '.join(command)}\n{result.stdout}\n{result.stderr}")


def _git(*args: str) -> str:
    """The output of a Git command in this worktree."""
    return subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def _normalized(name: str) -> str:
    """A distribution's name, normalized as in wheel file names."""
    return re.sub(r"[-_.]+", "_", name).lower()


def _wheel_name(wheel: pathlib.Path) -> tuple[str, str] | None:
    """The normalized name and the version of a wheel, from ``name-version[-build]-tags.whl``."""
    parts = wheel.name.removesuffix(".whl").split("-")
    return (_normalized(parts[0]), parts[1]) if len(parts) in (5, 6) else None


def _wheel_version(wheel: pathlib.Path) -> str:
    """The version in a wheel's file name."""
    return wheel.name.split("-")[1]


if __name__ == "__main__":
    main()
