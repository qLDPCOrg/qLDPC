# SPDX-License-Identifier: Apache-2.0

"""Compiled functions, cached on disk by a hash of the source.

The distance and counting searches run functions that numba compiles.  They are written in the
subset of Python and NumPy that numba supports, and numba compiles them the first time that they
are used.

:func:`always_compiled` caches their compiled code on disk.  Numba checks only the source file of a
cached function for changes, not the functions that it calls in other files, so a cache that is
kept where numba puts it could serve outdated code.  :func:`always_compiled` instead keeps the
compiled code in a directory that is named by a hash of the source files of this package, so that
any change to them selects a new, empty directory.  Outdated code is never loaded, and old
directories can simply be deleted.
"""

from __future__ import annotations

import functools
import hashlib
import tempfile
import threading
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numba
from numba.core import config

import qldpc.cache

# Numba creates the attributes of its settings dynamically, which type checkers cannot see.
_numba_config: Any = config

# The package whose source files name the directory of its compiled code.
_PACKAGE_DIRECTORY = Path(__file__).parent

# Held while the cache directory of numba, which is a setting of the whole process, is changed.  A
# thread that caches another numba function meanwhile caches it in the same directory, which does
# not affect results.
_CACHE_DIRECTORY_LOCK = threading.Lock()


####################################################################################################
# functions cached on disk


def always_compiled(function: Callable[..., Any]) -> Any:
    """Compile a function with numba, and cache its compiled code in a directory of this source.

    The directory is named by a hash of the source files of this package, so that code compiled from
    outdated source files is never loaded.  It is ``qldpc-<hash>`` inside ``NUMBA_CACHE_DIR`` if
    that is set, and otherwise inside the ``qldpc/numba`` directory of the user's cache directory.
    If the directory cannot be used, qLDPC warns once, and every process compiles the function
    again.
    """
    if _numba_config.DISABLE_JIT:  # numba then runs every function as ordinary Python
        return numba.njit(function)
    cache_directory = _get_cache_directory(_PACKAGE_DIRECTORY)
    dispatcher = (
        None if cache_directory is None else _compile_with_cache_in(function, cache_directory)
    )
    return numba.njit(function) if dispatcher is None else dispatcher


####################################################################################################
# helper functions


def _compile_with_cache_in(function: Callable[..., Any], directory: Path) -> Any:
    """Compile a function with numba, cached in a directory, or return None if numba ignores it.

    Numba reads its ``CACHE_DIR`` setting, the only way to choose where it caches, when it compiles
    a function, and reports where it cached in ``stats.cache_path``.  Both are internal to numba,
    so the result is checked.
    """
    with _CACHE_DIRECTORY_LOCK:
        previous_directory = _numba_config.CACHE_DIR
        _numba_config.CACHE_DIR = str(directory)
        try:
            dispatcher = numba.njit(cache=True)(function)
        finally:
            _numba_config.CACHE_DIR = previous_directory

    # Numba moves on to its other cache locations if it cannot use the directory.
    cache_path = dispatcher.stats.cache_path
    if cache_path is not None and Path(cache_path).is_relative_to(directory):
        return dispatcher
    _warn_not_cached(f"numba does not use {directory}")
    return None


@functools.cache
def _get_cache_directory(package: Path) -> Path | None:
    """A directory that can be written for the compiled code of a package, or None with a warning.

    Numba needs to be able to write in the directory, and checks this itself only when it has
    already moved on to another directory.
    """
    source_hash = _get_source_hash(package)
    if source_hash is None:
        _warn_not_cached(f"{package} has no source files")
        return None
    base = _numba_config.CACHE_DIR or str(qldpc.cache.get_disk_cache_path("numba"))
    directory = Path(base) / f"qldpc-{source_hash}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        tempfile.TemporaryFile(dir=directory).close()
    except OSError as error:
        _warn_not_cached(f"{directory} cannot be written: {error}")
        return None
    return directory


def _warn_not_cached(reason: str) -> None:
    """Warn that compiled code is not cached, so that every process compiles it again."""
    warnings.warn(
        f"qLDPC cannot cache its compiled code ({reason}), so every process compiles it again.  "
        "Make the user cache directory writable, or set the NUMBA_CACHE_DIR environment variable "
        "to a writable directory.",
        RuntimeWarning,
        stacklevel=1,
    )


def _get_source_hash(directory: Path) -> str | None:
    """A hash of the paths and contents of the source files of a package, or None if it has none.

    Tests do not change the compiled code, so they are not included.
    """
    paths = [
        path
        for path in sorted(directory.rglob("*.py"))
        if not path.name.endswith("_test.py") and path.name != "conftest.py"
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16] if paths else None
