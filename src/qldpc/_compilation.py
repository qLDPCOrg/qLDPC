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
empty directory.  Outdated code is never loaded.  Compiled code may therefore call always-compiled
functions in any module of qLDPC, but no compiled functions of other packages, whose source the hash
does not cover.  Each directory names the package directory whose source files it was compiled from.
Each import of qLDPC updates the modification time of its directory, and deletes the outdated
directories of the same package directory, compiled from earlier versions of its source files, that
were last modified more than 7 days earlier.  Directories of other checkouts or installations are
kept.

Several processes can compile the same function at the same time, such as parallel test workers
after a change to the source.  Numba saves each compiled version of a function in a data file, and
keeps an index that maps the key of each compiled version, which includes its argument types, to its
data file.  It names the data file of a new compiled version by the lowest number that the index it
read does not use, and nothing stops another process from writing the index between its read and its
write.  Two processes that compile a function for different argument types at once can thus write
the same data file, and leave the index pointing at code compiled for other argument types (`numba
issue #10862 <https://github.com/numba/numba/issues/10862>`_).  Numba checks little more than the
size of the items of array arguments before it runs such code, which can then raise an error or
silently return wrong results.  :func:`always_compiled` therefore saves numba's version and the key
of each compiled version with its code, and loads the code only if both match.  Otherwise numba
compiles the function again for that key, and saves the new code in place of the code that did not
match.

Compiled-on-request functions, such as the building of overlapping basis families, are registered
with :func:`compiled_on_request`.  They run as ordinary Python until
:func:`~qldpc.codes.distance.compile_search_setup` replaces each registered name in its module by a
version compiled in memory, which is never cached on disk.  A function calls other functions by
their module-level names, which are looked up when it is compiled, so a compiled function calls
compiled functions.  This is why every function that a compiled-on-request function calls must be
always-compiled or compiled on request too.  Other modules must call a compiled-on-request function
through the module that defines it, as in ``information_sets.get_overlapping_family(...)``, and
never through a name imported by ``from``, which would keep the version that was current at import
time.

To choose between them, use :func:`always_compiled` for the per-word loops and for the helper
functions that other compiled code calls: every search needs them, and the disk cache makes
compiling them a one-time cost.  Use :func:`compiled_on_request` for the larger functions that
orchestrate a search, which select and bound its information sets, build its basis families, and
drive its scan.  Compiling them pays off only in a program that makes very many searches.  A
function that is neither runs as ordinary Python.  A small always-compiled helper that hot loops
call can be marked ``inline=True`` (see :func:`always_compiled`).  A compiled-on-request function
may call always-compiled and compiled-on-request functions.  An always-compiled function may call
only always-compiled functions, since it is compiled and cached by itself.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import os
import pickle
import re
import shutil
import tempfile
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, cast, overload

import numba
from numba.core import caching, config, serialize

import qldpc.cache

if TYPE_CHECKING:
    from numba.core.dispatcher import Dispatcher

_Function = TypeVar("_Function", bound=Callable[..., object])


class _NumbaConfig(Protocol):
    """The settings of numba that this module reads and writes."""

    CACHE_DIR: str
    DISABLE_JIT: bool


# Numba creates the attributes of its settings dynamically, which type checkers cannot see.
_numba_config = cast(_NumbaConfig, config)

# The directory of qLDPC, whose source files name the directory of its compiled code.
_PACKAGE_DIRECTORY = Path(__file__).parent

# The directory of compiled code is named by "qldpc-" and this many hexadecimal digits of the hash.
_HASH_LENGTH = 16
_CACHE_DIRECTORY_NAME = re.compile(f"qldpc-[0-9a-f]{{{_HASH_LENGTH}}}")

# Outdated directories of compiled code are deleted if they were last modified more than this many
# seconds before the one in use.
_MAX_UNUSED_SECONDS = 7 * 24 * 60 * 60

# The file in a directory of compiled code that names the package directory whose code it holds.
_SOURCE_MARKER = "source_directory.txt"

# What to do when compiled code cannot be cached in a directory.
_CACHE_DIRECTORY_ADVICE = (
    "Make the user cache directory writable, or set the NUMBA_CACHE_DIR environment variable to a "
    "writable directory."
)

# Held while the cache directory of numba, which is a setting of the whole process, is changed.  A
# thread that caches another numba function meanwhile caches it in the same directory, which does
# not affect results.
_CACHE_DIRECTORY_LOCK = threading.Lock()


@dataclass
class _OnRequestFunction:
    """The module namespace and name of a registered function, with its plain and compiled forms."""

    namespace: dict[str, object]
    name: str
    function: Callable[..., object]
    compiled: Callable[..., object] | None = None


# Every registered function, in registration order.
_ON_REQUEST_FUNCTIONS: list[_OnRequestFunction] = []


####################################################################################################
# functions cached on disk


@overload
def always_compiled(function: _Function, /, *, inline: bool = False) -> Dispatcher[_Function]: ...


@overload
def always_compiled(*, inline: bool) -> Callable[[_Function], Dispatcher[_Function]]: ...


def always_compiled(
    function: _Function | None = None, /, *, inline: bool = False
) -> Dispatcher[_Function] | Callable[[_Function], Dispatcher[_Function]]:
    """Compile a function with numba, and cache its compiled code in a directory of qLDPC's source.

    The directory is named by a hash of the source files of qLDPC, so that code compiled from
    outdated source files is never loaded.  It is ``qldpc-<hash>`` inside ``NUMBA_CACHE_DIR`` if
    that is set, and otherwise inside the ``qldpc/numba`` directory of the user's cache directory.
    Numba reads and writes the cache files of the function through numba's cache-file object.  If
    the directory cannot be used, qLDPC warns once.  If that object is not exactly of the class that
    qLDPC extends, qLDPC warns for each function.  Either way, every process compiles the function
    again.

    Used as ``@always_compiled(inline=True)``, it also makes compiled code that calls the function
    include its code rather than call it, which saves the cost of a call in a hot loop.
    """
    if function is None:
        return lambda function: always_compiled(function, inline=inline)
    mode: Literal["always", "never"] = "always" if inline else "never"
    if _numba_config.DISABLE_JIT:  # numba then runs every function as ordinary Python
        return numba.njit(inline=mode)(function)
    cache_directory = _get_cache_directory(_PACKAGE_DIRECTORY)
    dispatcher = (
        None if cache_directory is None else _compile_with_cache_in(function, cache_directory, mode)
    )
    return numba.njit(inline=mode)(function) if dispatcher is None else dispatcher


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


def _compile_with_cache_in(
    function: _Function, directory: Path, inline: Literal["always", "never"]
) -> Dispatcher[_Function] | None:
    """Compile a function with numba, cached safely in a directory, or return None if it cannot be.

    ``inline`` is numba's option of that name: ``"always"`` to include the function's code in
    compiled callers, or ``"never"``.  The ``CACHE_DIR`` setting of numba is the only way to choose
    where it caches.  Numba reads the setting when it sets up the cache of a function, and reports
    where it caches in ``stats.cache_path``.  This function also makes numba's cache-file object
    check numba's version and the key of each compiled version that it loads, by changing the class
    of the object to ``_KeyCheckedCacheFile``.  The setting, the report, and the object are internal
    to numba, so this function checks the report and the class of the object.
    """
    # Make numba cache the function in the directory.
    with _CACHE_DIRECTORY_LOCK:
        previous_directory = _numba_config.CACHE_DIR
        _numba_config.CACHE_DIR = str(directory)
        try:
            dispatcher = numba.njit(cache=True, inline=inline)(function)
        finally:
            _numba_config.CACHE_DIR = previous_directory

    # Use the directory only if numba caches there, since numba moves on to its other cache
    # locations if it cannot use the directory.
    cache_path = dispatcher.stats.cache_path
    if cache_path is None or not Path(cache_path).is_relative_to(directory):
        _warn_not_cached(f"numba does not use {directory}")
        return None

    # Make numba's cache-file object check what it loads, if it has exactly the class that qLDPC
    # extends.  Changing the class of an object of a subclass would drop the overrides of the
    # subclass.  The class is changed, rather than the object replaced, to keep the paths and source
    # stamp that the object holds.
    cache_file = getattr(getattr(dispatcher, "_cache", None), "_cache_file", None)
    if type(cache_file) is not caching.IndexDataCacheFile:
        _warn_not_cached(
            f"numba {numba.__version__} does not read and write its cache files through the class "
            "that qLDPC extends",
            "Install an earlier version of numba.",
        )
        return None
    cache_file.__class__ = _KeyCheckedCacheFile
    return dispatcher


class _KeyCheckedCacheFile(caching.IndexDataCacheFile):
    """Numba's cache-file object of a function, which loads only code saved for the same key.

    Each data file holds numba's version, and a pickle of the key of its compiled version with the
    code.  A load unpickles the pickle only if numba's version matches, since another version of
    numba may not be able to unpickle it.  It returns the code only if the key matches too.
    Otherwise numba compiles the function again for that key, and saves the new code in the same
    data file.  Environments with different versions of numba can share the directory: each reads
    only an index that its own version wrote, but any of them can write a data file.  The module
    docstring explains how a data file can hold code for another key.
    """

    def save(self, key: object, data: object) -> None:
        """Save the code compiled for a key, with the key and numba's version."""
        # Numba pickles its data with its own pickler, which handles more objects than pickle does.
        super().save(key, (numba.__version__, serialize.dumps((key, data))))

    def load(self, key: object) -> object:
        """Load the code for a key, or None unless this version of numba saved it for that key."""
        saved = super().load(key)
        if saved is None:
            return None
        numba_version, pickled = saved
        if numba_version != numba.__version__:
            return None
        # Numba has just unpickled this data file, as it does every file of its cache, so unpickling
        # the payload trusts the file no more than numba does.
        saved_key, data = pickle.loads(pickled)  # noqa: S301
        return data if saved_key == key else None


def _get_cache_directory(package: Path) -> Path | None:
    """A directory that can be written for the compiled code of a package, or None with a warning.

    The directory is inside ``NUMBA_CACHE_DIR`` if that is set, and otherwise inside the ``numba``
    directory of qLDPC's user cache directory.
    """
    base = _numba_config.CACHE_DIR or str(qldpc.cache.get_disk_cache_path("numba"))
    return _get_cache_directory_in(Path(base), package)


@functools.cache
def _get_cache_directory_in(base: Path, package: Path) -> Path | None:
    """A directory in a base directory that can be written for the compiled code of a package.

    Numba needs to be able to write in the directory, and checks this itself only when it has
    already moved on to another directory.  Outdated directories of the same package beside it are
    deleted.  If the directory cannot be used, warn and return None.
    """
    try:
        source_hash = _get_source_hash(package)
    except OSError as error:
        _warn_not_cached(
            f"the source files of {package} cannot be read: {error}",
            "Make the source files of qLDPC readable, or reinstall qLDPC.",
        )
        return None
    if source_hash is None:
        _warn_not_cached(
            f"{package} has no source files", "Install qLDPC with its .py source files."
        )
        return None
    directory = base / f"qldpc-{source_hash}"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        tempfile.TemporaryFile(dir=directory).close()
    except OSError as error:
        _warn_not_cached(f"{directory} cannot be written: {error}")
        return None
    _delete_outdated_cache_directories(directory, package)
    return directory


def _delete_outdated_cache_directories(directory: Path, package: Path) -> None:
    """Mark a directory of compiled code as used by a package, and delete its outdated directories.

    A directory beside this one is outdated if it holds compiled code of the same package directory,
    whose source files have since changed, and it was last modified, for example by an import of
    qLDPC that used it, more than 7 days before this one.  The grace period protects processes that
    still run the old source files.  Directories of other package directories, such as other
    checkouts or installations, are kept.  This is best effort, and errors are ignored.
    """
    with contextlib.suppress(OSError):
        source = str(package.resolve())
        marker = directory / _SOURCE_MARKER
        if not marker.is_file() or marker.read_text(encoding="utf-8") != source:
            marker.write_text(source, encoding="utf-8")
        os.utime(directory)
        cutoff = directory.stat().st_mtime - _MAX_UNUSED_SECONDS
        with os.scandir(directory.parent) as entries:
            for entry in entries:
                if (
                    entry.name != directory.name
                    and _CACHE_DIRECTORY_NAME.fullmatch(entry.name)
                    and entry.is_dir(follow_symlinks=False)
                    and entry.stat(follow_symlinks=False).st_mtime < cutoff
                    and _read_source_marker(Path(entry.path)) == source
                ):
                    shutil.rmtree(entry.path, ignore_errors=True)


def _read_source_marker(directory: Path) -> str | None:
    """The package directory whose compiled code a directory holds, or None if it is unknown."""
    try:
        return (directory / _SOURCE_MARKER).read_text(encoding="utf-8")
    except OSError:
        return None


def _warn_not_cached(reason: str, advice: str = _CACHE_DIRECTORY_ADVICE) -> None:
    """Warn that compiled code is not cached, so that every process compiles it again."""
    warnings.warn(
        f"qLDPC cannot cache its compiled code ({reason}), so every process compiles it again.  "
        + advice,
        RuntimeWarning,
        stacklevel=1,
    )


def _get_source_hash(directory: Path) -> str | None:
    """A hash of the paths and contents of the source files of a package, or None if it has none."""
    paths = _get_source_paths(directory)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:_HASH_LENGTH] if paths else None


def _get_source_paths(directory: Path) -> list[Path]:
    """The sorted paths of the source files of a package: its regular, non-hidden ``*.py`` files.

    Hidden files, such as the lock files of editors, are not imported, and tests do not change the
    compiled code, so neither is included.
    """
    return [
        path
        for path in sorted(directory.rglob("*.py"))
        if not path.name.startswith(".")
        and not path.name.endswith("_test.py")
        and path.name != "conftest.py"
        and path.is_file()
    ]
