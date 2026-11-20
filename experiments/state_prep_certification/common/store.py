# SPDX-License-Identifier: Apache-2.0

"""The JSONL result store of a label, and the archive that carries it back from the node.

Each task runner appends one final row per task to ``runs.jsonl`` with ``append_row``: one write of
a whole line under an exclusive ``flock``, followed by a flush and an fsync.  A runner that dies in
the middle of a write can leave an incomplete last line.  The next append, under the same lock,
moves that fragment to ``recovered.txt`` and truncates the file to its last complete line before it
writes, so that a new row never joins a fragment.  ``read_rows`` ignores an incomplete last line and
reports it, and ``final_rows`` keeps the first final row of each task and reports every duplicate,
such as a second row of a task that two runners ran at once.
"""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import pathlib
import tarfile
from collections.abc import Iterable
from typing import Any

RUNS_FILE = "runs.jsonl"
RECOVERED_FILE = "recovered.txt"

# Directories of a label that never enter its archive.
_UNARCHIVED = ("archive",)


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


def make_archive(label_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Pack a label's directory into ``archive/<name>.tar.gz``, with a ``.sha256`` file beside it.

    The archive holds every file of the label except earlier archives, and an inner ``SHA256SUMS``.
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
        if path.is_file() and not path.is_symlink() and parts[0] not in _UNARCHIVED:
            yield path
