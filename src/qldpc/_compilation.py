# SPDX-License-Identifier: Apache-2.0

"""Compiled functions: cached on disk by a hash of the source, or compiled in memory on request.

qLDPC, for example in its distance and counting searches, runs two kinds of functions that numba
compiles.  They are written in the subset of Python and NumPy that numba supports.

Always-compiled functions, such as bit packing, row reduction and the scan of a chunk of
combinations, are compiled by numba the first time that they are used.  :func:`always_compiled`
caches their compiled code on disk.  Numba checks only the source file of a cached function for
changes, not the functions that it calls in other files, so a cache that is kept where numba puts it
could serve outdated code.  :func:`always_compiled` instead keeps the compiled code in a directory
that is named by a hash of the source files of qLDPC, so that any change to them selects a new,
empty directory.  Outdated code is never loaded, and old directories can simply be deleted.
Compiled code may therefore call always-compiled functions in any module of qLDPC, but no compiled
functions of other packages, whose source the hash does not cover.

Compiled-on-request functions, such as the building of overlapping basis families, are registered
with :func:`compiled_on_request`.  They run as ordinary Python until
:func:`~qldpc.codes.distance.compile_search_setup` replaces each registered name in its module by a
version compiled in memory, which is never cached on disk.  A function calls other functions by
their module-level names, which are looked up when it is compiled, so a compiled function calls
compiled functions.  This is why every function that a compiled-on-request function calls must be
always-compiled or compiled on request too.  Other modules must call a compiled-on-request function
through the module that defines it, as in ``_information_sets.get_overlapping_family(...)``, and
never through a name imported by ``from``, which would keep the version that was current at import
time.

To choose between them, use :func:`always_compiled` for the per-word loops and for the helper
functions that other compiled code calls: every search needs them, and the disk cache makes
compiling them a one-time cost.  Use :func:`compiled_on_request` for the larger functions that
orchestrate a search, which select and bound its information sets, build its basis families, and
drive its scan.  Compiling them pays off only in a program that makes very many searches.  A
function that is neither runs as ordinary Python.  A compiled-on-request function may call
always-compiled and compiled-on-request functions.  An always-compiled function may call only
always-compiled functions, since it is compiled and cached by itself.
"""

from __future__ import annotations

import functools
import hashlib
import tempfile
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

import numba
from numba.core import config

import qldpc.cache

if TYPE_CHECKING:
    from numba.core.dispatcher import Dispatcher

_Function = TypeVar("_Function", bound=Callable[..., Any])


class _NumbaConfig(Protocol):
    """The settings of numba that this module reads and writes."""

    CACHE_DIR: str
    DISABLE_JIT: bool


# Numba creates the attributes of its settings dynamically, which type checkers cannot see.
_numba_config = cast(_NumbaConfig, config)

# The directory of qLDPC, whose source files name the directory of its compiled code.
_PACKAGE_DIRECTORY = Path(__file__).parent

# Held while the cache directory of numba, which is a setting of the whole process, is changed.  A
# thread that caches another numba function meanwhile caches it in the same directory, which does
# not affect results.
_CACHE_DIRECTORY_LOCK = threading.Lock()


@dataclass
class _OnRequestFunction:
    """The module namespace and name of a registered function, with its plain and compiled forms."""

    namespace: dict[str, Any]
    name: str
    function: Callable[..., Any]
    compiled: Callable[..., Any] | None = None


# Every registered function, in registration order.
_ON_REQUEST_FUNCTIONS: list[_OnRequestFunction] = []


####################################################################################################
# functions cached on disk


def always_compiled(function: _Function) -> Dispatcher[_Function]:
    """Compile a function with numba, and cache its compiled code in a directory of qLDPC's source.

    The directory is named by a hash of the source files of qLDPC, so that code compiled from
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
# functions compiled in memory on request


def compiled_on_request(function: _Function) -> _Function:
    """Register a function that runs as plain Python, or is compiled by numba on request."""
    _ON_REQUEST_FUNCTIONS.append(
        _OnRequestFunction(function.__globals__, function.__name__, function)
    )
    return function


def set_on_request_functions_compiled(compiled: bool) -> None:
    """Replace every registered function by its compiled, or by its plain Python, version.

    A function is compiled once, when first called after the first request, and kept for later ones.
    """
    for item in _ON_REQUEST_FUNCTIONS:
        if compiled and item.compiled is None:
            item.compiled = numba.njit(item.function)
        item.namespace[item.name] = item.compiled if compiled else item.function


####################################################################################################
# helper functions


def _compile_with_cache_in(function: _Function, directory: Path) -> Dispatcher[_Function] | None:
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
