# SPDX-License-Identifier: Apache-2.0

"""The JSONL result store of a label, and the archive that carries it back from the node.

Each task runner appends one final row per task to ``runs.jsonl`` with ``append_row``: one write of
a whole line under an exclusive ``flock``, followed by a flush and an fsync.  A runner that dies in
the middle of a write can leave an incomplete last line.  The next append, under the same lock,
moves that fragment to ``recovered.txt`` and truncates the file to its last complete line before it
writes, so that a new row never joins a fragment.  ``read_rows`` ignores an incomplete last line and
reports it, and ``final_rows`` keeps the first final row of each task and reports every duplicate,
such as a second row of a task that two runners ran at once.

``write_atomic`` writes the label's other records, such as ``acceptance.json``, ``manifest.json``,
and the runner's ``active/`` files, whole or not at all, so that a kill during a write never
leaves a partial record.  The temporary file that such a kill can leave, ``.<name>.<pid>.tmp``,
never enters the archive.

The rows of ``--local`` runs (``"local": true``) were not timed under the node's conditions.  An
analysis that refuses them, as those of certification and distances do, counts them with
``count_local_rows``, which exits unless the analysis is given ``--allow-local``.  Such an analysis
also reads, with ``allowed_environment``, the refused environment variables that a start accepted
with ``--allow-env`` (see ``run_node.py``), and, with ``changed_environment``, the recorded
environment variables whose values differ among the label's starts.  It marks each of its outputs
with these, and with the number of rows of local runs, whichever are nonempty, with
``mark_outputs``.
"""

from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import os
import pathlib
import sys
import tarfile
from collections.abc import Iterable
from typing import Any

RUNS_FILE = "runs.jsonl"
RECOVERED_FILE = "recovered.txt"
STARTS_FILE = "starts.jsonl"  # one line per start of the label (see manifest.record_start)
LOCK_FILE = "lock"  # the lock of the start that owns the label (see run_node.py)

# Files and directories of a label that never enter its archive.
_UNARCHIVED = ("archive", LOCK_FILE)


def append_row(label_dir: pathlib.Path, row: dict[str, Any]) -> None:
    """Append one row to the label's ``runs.jsonl``, atomically with respect to other runners."""
    line = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path = label_dir / RUNS_FILE
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        _truncate_incomplete_tail(fd, label_dir)
        os.lseek(fd, 0, os.SEEK_END)
        _write_all(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)  # closing releases the lock


def read_rows(label_dir: pathlib.Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The complete rows of ``runs.jsonl`` in file order, and a list of problems found reading it.

    An incomplete last line, or an interior line that is not a JSON object, is skipped and reported.
    """
    path = label_dir / RUNS_FILE
    if not path.exists():
        return [], []
    data = path.read_bytes()
    lines = data.split(b"\n")
    tail = lines.pop()  # empty if the file ends with a newline
    rows, problems = [], []
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except ValueError as error:
            row = error
        if isinstance(row, dict):
            rows.append(row)
        else:
            problems.append(f"line {number} of {RUNS_FILE} is not a row ({row!r}); skipped")
    if tail:
        problems.append(f"incomplete last line of {RUNS_FILE} ({len(tail)} bytes); ignored")
    return rows, problems


def final_rows(label_dir: pathlib.Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """The first final row of each task, by task key, and a list of problems found reading them."""
    rows, problems = read_rows(label_dir)
    first: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get("task_key")
        if not isinstance(key, str):
            problems.append(f"a row without a task key was skipped: run_id {row.get('run_id')}")
        elif key in first:
            problems.append(
                f"duplicate final row for {key}: run_id {row.get('run_id')} ignored, "
                f"run_id {first[key].get('run_id')} kept"
            )
        else:
            first[key] = row
    return first, problems


def count_local_rows(rows: Iterable[dict[str, Any]], allow_local: bool) -> int:
    """The number of rows of ``--local`` runs, which an analysis refuses unless ``allow_local``.

    Raises:
        SystemExit: If there are such rows and ``allow_local`` is false.
    """
    count = sum(bool(row.get("local")) for row in rows)
    if count and not allow_local:
        sys.exit(
            f"{count} final rows come from --local runs, which were not timed as on the node; "
            "pass --allow-local to analyze them anyway, and every output will say so"
        )
    return count


def allowed_environment(label_dir: pathlib.Path) -> list[str]:
    """The refused environment variables that a start of the label accepted and that were set.

    A node start refuses some environment variables unless ``--allow-env`` accepts them (see
    ``run_node.py``); ``starts.jsonl`` records, for each start, the names that it accepted and the
    environment.  Every start counts, dry runs too, since each records itself before it runs the
    acceptance tests, whose pass later starts reuse.
    """
    names: set[str] = set()
    for start in _starts(label_dir):
        environment = start.get("environment", {})
        names |= {name for name in start.get("allowed_environment") or [] if name in environment}
    return sorted(names)


def changed_environment(label_dir: pathlib.Path) -> list[str]:
    """The recorded environment variables whose values differ among the starts of the label.

    A variable that one start set and another did not counts too.  Every start counts, as in
    ``allowed_environment``.
    """
    environments = [start.get("environment", {}) for start in _starts(label_dir)]
    names = {name for environment in environments for name in environment}
    return sorted(
        name for name in names if len({environment.get(name) for environment in environments}) > 1
    )


def mark_outputs(out: pathlib.Path, marks: dict[str, Any]) -> None:
    """Mark the outputs in ``out`` with ``marks``, such as the number of rows of local runs.

    Each CSV file gets a column per mark, each JSON object a key per mark, and each Stim circuit a
    first comment line per mark, ``# <name>: <value>``.  Other files are left to the analysis.
    """
    header = "".join(f"# {name}: {value}\n" for name, value in marks.items())
    for path in sorted(out.rglob("*")):
        if path.suffix == ".csv":
            with open(path, newline="") as file:
                table = list(csv.reader(file))
            new = [name for name in marks if table and name not in table[0]]
            if new:
                values = [str(marks[name]) for name in new]
                table = [table[0] + new] + [row + values for row in table[1:]]
                with open(path, "w", newline="") as file:
                    csv.writer(file).writerows(table)
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
            data.update(marks)
            path.write_text(json.dumps(data, indent=1) + "\n")
        elif path.suffix == ".stim":
            text = path.read_text()
            if not text.startswith(header):
                path.write_text(header + text)


def write_atomic(path: pathlib.Path, text: str, directory: pathlib.Path | None = None) -> None:
    """Write a file whole or not at all: write a temporary file, then rename it to ``path``.

    The temporary file is in ``directory``, by default that of ``path``, on the same file system.
    """
    temporary = (directory or path.parent) / f".{path.name}.{os.getpid()}.tmp"
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def make_archive(label_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Pack a label's directory into ``archive/<name>.tar.gz``, with a ``.sha256`` file beside it.

    The archive holds the label's regular files, except earlier archives, the label's ``lock``, and
    the temporary files of ``write_atomic`` (see ``_archived_files``), and an inner ``SHA256SUMS``.
    """
    archive_dir = label_dir / "archive"
    archive_dir.mkdir(exist_ok=True)
    files = sorted(_archived_files(label_dir))
    sums = "".join(
        f"{sha256_file(path)}  {path.relative_to(label_dir).as_posix()}\n" for path in files
    )
    archive = archive_dir / f"{name}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in files:
            tar.add(path, arcname=f"{name}/{path.relative_to(label_dir).as_posix()}")
        info = tarfile.TarInfo(f"{name}/SHA256SUMS")
        info.size = len(sums.encode())
        tar.addfile(info, io.BytesIO(sums.encode()))
    archive.with_name(archive.name + ".sha256").write_text(
        f"{sha256_file(archive)}  {archive.name}\n"
    )
    return archive


def sha256_file(path: pathlib.Path) -> str:
    """The SHA-256 of a file, in hexadecimal."""
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _starts(label_dir: pathlib.Path) -> list[dict[str, Any]]:
    """The lines of the label's ``starts.jsonl``, one per start."""
    path = label_dir / STARTS_FILE
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _truncate_incomplete_tail(fd: int, label_dir: pathlib.Path) -> None:
    """Move an incomplete last line, if any, to ``recovered.txt``, and cut it from the file."""
    size = os.lseek(fd, 0, os.SEEK_END)
    if size == 0:
        return
    os.lseek(fd, size - 1, os.SEEK_SET)
    if os.read(fd, 1) == b"\n":
        return

    # find the end of the last complete line, reading backwards in blocks
    end = size
    cut = 0
    while end > 0:
        start = max(0, end - 65536)
        os.lseek(fd, start, os.SEEK_SET)
        block = os.read(fd, end - start)
        index = block.rfind(b"\n")
        if index >= 0:
            cut = start + index + 1
            break
        end = start

    # keep the fragment for the report, then truncate
    os.lseek(fd, cut, os.SEEK_SET)
    fragment = os.read(fd, size - cut)
    with open(label_dir / RECOVERED_FILE, "ab") as file:
        file.write(b"incomplete line removed before an append: " + fragment + b"\n")
    os.ftruncate(fd, cut)
    os.fsync(fd)


def _write_all(fd: int, data: bytes) -> None:
    """Write all of ``data`` to a file descriptor."""
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _archived_files(label_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    """Every regular file of a label's directory that belongs in its archive."""
    for path in label_dir.rglob("*"):
        parts = path.relative_to(label_dir).parts
        temporary = path.name.startswith(".") and path.name.endswith(".tmp")  # see write_atomic
        if (
            path.is_file()
            and not path.is_symlink()
            and parts[0] not in _UNARCHIVED
            and not temporary
        ):
            yield path
