# SPDX-License-Identifier: Apache-2.0

"""Files that parallel jobs share: replaced atomically, and locked against a second writer.

An atomic replacement means that a failure or an interruption never leaves half a file.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import stat
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO


def write_atomic(destination: Path, write: Callable[[IO[str]], None], *, newline: str = "") -> None:
    """Replace a file only after ``write`` has produced a complete, flushed file.

    Temporary files are removed on every exit, so a failure leaves the previous file untouched.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = _get_file_mode(destination)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline=newline,
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.chmod(temporary, mode)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def hold_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on a file while the block runs, creating the file if needed.

    The operating system releases the lock when the process ends, however it ends, so a killed
    process leaves no stale lock behind.

    Raises:
        BlockingIOError: If another process holds the lock.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def write_text_atomic(destination: str | Path, text: str) -> None:
    """Replace a file with text, atomically."""

    def write(stream: IO[str]) -> None:
        stream.write(text)

    write_atomic(Path(destination), write)


####################################################################################################
# helper methods


def _get_file_mode(destination: Path) -> int:
    """The permissions of the file to replace, or those that a new file gets with the umask.

    A temporary file is private to its owner, so the permissions are set explicitly instead.
    """
    if destination.exists():
        return stat.S_IMODE(destination.stat().st_mode)
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask
