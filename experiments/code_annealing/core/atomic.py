# SPDX-License-Identifier: Apache-2.0

"""Files that parallel jobs share: replaced atomically, and locked against a second writer.

An atomic replacement means that a failure or an interruption never leaves half a file.  Locks use
``fcntl``, which only Unix systems have.
"""

from __future__ import annotations

import contextlib
import io
import os
import stat
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO


def write_atomic(destination: Path, write: Callable[[IO[str]], None]) -> None:
    """Replace a file only after ``write`` has produced its complete contents.

    The stream translates no newlines, as the CSV writer of ``saved_runs`` requires.  The file is
    replaced as ``write_text_atomic`` replaces it.

    Args:
        destination: The file to replace, whose directory is created if needed.
        write: Writes the complete contents of the file to the stream that it is given.
            ``write_atomic`` calls it once, even if it writes the file again.
    """
    buffer = io.StringIO(newline="")
    write(buffer)
    write_text_atomic(destination, buffer.getvalue())


@contextlib.contextmanager
def hold_lock(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on a file while the block runs, creating the file if needed.

    The operating system releases the lock when the process ends, however it ends, so a killed
    process leaves no stale lock behind.

    Raises:
        BlockingIOError: If another process holds the lock.
    """
    # Imported here, so that the scripts that take no locks also run where fcntl is missing.
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def write_text_atomic(destination: str | Path, text: str) -> None:
    """Replace a file with text, atomically, after the text is flushed to disk.

    Temporary files are removed on every exit, so a failure leaves the previous file untouched.

    On a network file system, such as NFS, a client can send a rename again after the first attempt
    succeeded, and the second attempt then fails because the temporary file is gone.  So after such
    a failure, the text is written and renamed once more, which is harmless if the first rename took
    effect.
    """
    try:
        _replace_with_text(Path(destination), text)
    except FileNotFoundError:
        _replace_with_text(Path(destination), text)


####################################################################################################
# helper methods


def _replace_with_text(destination: Path, text: str) -> None:
    """Write text to a temporary file beside a file, flush it, and rename it over the file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    mode = _get_file_mode(destination)
    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.chmod(temporary, mode)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def _get_file_mode(destination: Path) -> int:
    """The permissions of the file to replace, or those that a new file gets with the umask.

    A temporary file is created private to its owner, so ``write_atomic`` gives it these permissions
    before it replaces the destination.
    """
    if destination.exists():
        return stat.S_IMODE(destination.stat().st_mode)
    umask = os.umask(0)
    os.umask(umask)
    return 0o666 & ~umask
