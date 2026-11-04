# SPDX-License-Identifier: Apache-2.0

"""A pytest plugin that measures coverage, labeled by the test file responsible for each line.

The modular coverage check (``coverage_.py``) loads this plugin in every pytest process, including
each pytest-xdist worker.  It starts coverage before any ``conftest.py`` file imports the code under
test, and labels every line that runs with a coverage context:

* the empty context while pytest starts, which loads the ``conftest.py`` files of the directory
  that contains every selected test file and of its parents,
* ``<collector node ID>|collect`` while a test file or a directory, with its ``conftest.py``, is
  collected,
* ``<test node ID>|setup``, ``|run``, or ``|teardown`` while a test runs, and
* ``unattributed`` at other times, such as between tests and after the last one.

To make a test file run as it would in a fresh process, the plugin also undoes the state that a
test file leaves behind for the next one on the same worker: after the last test of a test file, it
tears down the session- and package-scoped fixtures of the repository that the file used, so that
the next test file sets them up anew, and it empties every ``functools`` cache of the packages in
``CACHED_PACKAGES``.  Other state that a test file leaves in modules is not reset.

Collection is restricted to the selected test files, so that other test files are not imported.
Lines that only a subprocess of a test runs are not measured.
"""

from __future__ import annotations

import os
import pathlib
import sys
from collections.abc import Generator

import coverage
import pytest

# The packages whose functools caches are emptied after each test file.
CACHED_PACKAGES = ("qldpc",)

# Environment variables set by coverage_.py.
DATA_FILE_VARIABLE = "MODULAR_COVERAGE_DATA_FILE"
INCLUDE_VARIABLE = "MODULAR_COVERAGE_INCLUDE"
TEST_FILES_VARIABLE = "MODULAR_COVERAGE_TEST_FILES"
BRANCH_VARIABLE = "MODULAR_COVERAGE_BRANCH"

# The context of lines that run outside of the collection and the tests of any one test file.
UNATTRIBUTED = "unattributed"

# Fixtures of these scopes can outlive the test file that set them up.
SHARED_SCOPES = ("session", "package")


####################################################################################################
# pytest hooks


def pytest_ignore_collect(collection_path: pathlib.Path, config: pytest.Config) -> bool | None:
    """Collect only the selected test files, and the directories that contain them."""
    if _selected_test_files is None:
        return None
    path = _normalize(str(collection_path))
    if collection_path.is_dir():
        prefix = path + os.sep
        return None if any(file.startswith(prefix) for file in _selected_test_files) else True
    return None if path in _selected_test_files else True


@pytest.hookimpl(tryfirst=True)
def pytest_collectstart(collector: pytest.Collector) -> None:
    """Label the lines that run while a test file or directory is collected."""
    _context_stack.append(_current_context[0])
    _switch_context(f"{collector.nodeid}|collect" if collector.nodeid else "")


@pytest.hookimpl(trylast=True)
def pytest_collectreport(report: pytest.CollectReport) -> None:
    """Restore the label of the enclosing collector."""
    if _context_stack:
        _switch_context(_context_stack.pop())


def pytest_collection_finish(session: pytest.Session) -> None:
    """Stop crediting lines to the startup of pytest."""
    _switch_context(UNATTRIBUTED)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Label the setup of a test, and note the shared fixtures that its test file uses."""
    fixture_info = getattr(item, "_fixtureinfo", None)
    for fixturedefs in fixture_info.name2fixturedefs.values() if fixture_info else ():
        for fixturedef in fixturedefs:
            # Fixtures of plugins, whose base ID is empty, are left alone.
            if fixturedef.scope in SHARED_SCOPES and fixturedef.baseid:
                _shared_fixtures[id(fixturedef)] = fixturedef
    _switch_context(f"{item.nodeid}|setup")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item: pytest.Item) -> None:
    """Label the body of a test."""
    _switch_context(f"{item.nodeid}|run")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> Generator[None]:
    """Label the teardown of a test, and reset shared state after the last test of a test file."""
    _switch_context(f"{item.nodeid}|teardown")
    try:
        yield
        if nextitem is None or _get_test_file(nextitem) != _get_test_file(item):
            _finish_shared_fixtures(item)
            _clear_package_caches()
    finally:
        _switch_context(UNATTRIBUTED)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """Save the coverage data of this process."""
    if _coverage is not None:
        _coverage.stop()
        _coverage.save()


####################################################################################################
# helper functions and state


def _start_coverage() -> coverage.Coverage | None:
    """Start measuring coverage in this process, if coverage_.py asked for it."""
    data_file = os.environ.get(DATA_FILE_VARIABLE)
    if not data_file:
        return None
    measurement = coverage.Coverage(data_file=data_file, data_suffix=True)
    # The sys.monitoring core reports each line once, so it would not credit a line to a test file
    # that runs it after another one did.
    measurement.set_option("run:core", "ctrace")
    measurement.set_option("run:include", os.environ[INCLUDE_VARIABLE].split(os.pathsep))
    # Lines that only subprocesses run are not attributed to a test, so they are not measured.
    measurement.set_option("run:patch", [])
    measurement.set_option("run:branch", bool(os.environ.get(BRANCH_VARIABLE)))
    measurement.start()
    return measurement


def _switch_context(context: str) -> None:
    """Label the lines that run from now on."""
    if _coverage is not None:
        _coverage.switch_context(context)
    _current_context[0] = context


def _get_test_file(item: pytest.Item) -> str:
    """The test file of a test."""
    return item.nodeid.split("::")[0]


def _finish_shared_fixtures(item: pytest.Item) -> None:
    """Tear down the shared fixtures that a test file used, so that the next one sets them up anew.

    Tearing down a fixture first tears down the fixtures that depend on it.
    """
    fixturedefs = list(_shared_fixtures.values())
    _shared_fixtures.clear()
    for fixturedef in reversed(fixturedefs):
        # FixtureDef.finish uses its request only to name the node in an error message.
        fixturedef.finish(item._request)  # type: ignore[attr-defined]


def _clear_package_caches() -> None:
    """Empty every functools cache of a function or method of a module in CACHED_PACKAGES."""
    for name, module in list(sys.modules.items()):
        if module is None or name.partition(".")[0] not in CACHED_PACKAGES:
            continue
        for value in list(vars(module).values()):
            candidates = [value]
            if isinstance(value, type) and value.__module__ == name:
                candidates += [
                    getattr(member, "__func__", member) for member in vars(value).values()
                ]
            for candidate in candidates:
                cache_clear = getattr(candidate, "cache_clear", None)
                if callable(cache_clear):
                    cache_clear()


def _normalize(path: str) -> str:
    """A path in the form that pytest and the operating system use, to compare it with others."""
    return os.path.normcase(os.path.normpath(path))


def _get_selected_test_files() -> set[str] | None:
    """The absolute paths of the test files to collect, or None to collect all of them."""
    test_files = os.environ.get(TEST_FILES_VARIABLE)
    return (
        None if test_files is None else {_normalize(file) for file in test_files.split(os.pathsep)}
    )


# This module is imported as a pytest plugin before pytest loads any conftest.py file.
_coverage = _start_coverage()
_selected_test_files = _get_selected_test_files()
_current_context = [""]
_context_stack: list[str] = []
_shared_fixtures: dict[int, pytest.FixtureDef[object]] = {}
