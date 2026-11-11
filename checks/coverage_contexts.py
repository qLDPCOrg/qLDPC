# SPDX-License-Identifier: Apache-2.0

"""A pytest plugin that measures coverage, labeled by the test file responsible for each line.

The modular coverage check (``coverage_.py``) loads this plugin in every pytest process, including
each pytest-xdist worker.  It starts coverage before any ``conftest.py`` file imports the code under
test, and labels every line that runs with a coverage context:

* the empty context while pytest starts, which loads the ``conftest.py`` files of the directory
  that contains every selected test file and of its parents, and while a module of the repository
  other than a test file first runs in the process, whenever that happens,
* ``<collector node ID>|collect`` while a test file or a directory, with its ``conftest.py``, is
  collected,
* ``<test node ID>|setup``, ``|run``, or ``|teardown`` while a test runs, and
* ``unattributed`` at other times, such as between tests and after the last one.

To make a test file run as it would in a fresh process, the plugin also undoes the state that a
test file leaves behind for the next one on the same worker: after the last test of a test file, it
tears down the session- and package-scoped fixtures of the repository that the file used, so that
the next test file sets them up anew.  After the plugin tears down these fixtures, qLDPC's
``conftest.py`` empties qLDPC's ``functools`` caches, because ``coverage_.py`` sets
``QLDPC_CLEAR_CACHES``.  Other state that a test file leaves in modules is not reset.

A module that first runs while a test file is collected or a test runs, such as one that qLDPC
imports lazily, would otherwise have the lines that it runs on import labeled with that test file,
rather than with its own test file, which would import it in a process of its own.

Collection is restricted to the selected test files, so that other test files are not imported.
Lines that only a subprocess of a test runs are not measured.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import pathlib
import sys
import types
from collections.abc import Generator, Sequence

import coverage
import pytest

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
    """Label the teardown of a test, and tear down shared fixtures after the last test of a file."""
    _switch_context(f"{item.nodeid}|teardown")
    try:
        yield
        if nextitem is None or _get_test_file(nextitem) != _get_test_file(item):
            _finish_shared_fixtures(item)
    finally:
        _switch_context(UNATTRIBUTED)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """Save the coverage data of this process."""
    if _coverage is not None:
        _coverage.stop()
        _coverage.save()


####################################################################################################
# first imports


class _FirstImportLoader(importlib.machinery.SourceFileLoader):
    """A source loader that runs a module in the empty context the first time that it runs.

    A module that runs again, as when it is reloaded, keeps the current context, since that run
    belongs to whatever ran it, such as a test.
    """

    def exec_module(self, module: types.ModuleType) -> None:
        """Run a module, in the empty context if this is its first run in the process."""
        file = _normalize(os.path.abspath(self.path))
        if file in _executed_files:
            super().exec_module(module)
            return
        _executed_files.add(file)
        previous_context = _current_context[0]
        _switch_context("")
        try:
            super().exec_module(module)
        finally:
            _switch_context(previous_context)


class _FirstImportFinder(importlib.abc.MetaPathFinder):
    """Load the modules of the repository, other than test files, with ``_FirstImportLoader``.

    Such modules include those that are not measured, such as ``__init__.py`` files, since their
    imports can run measured lines.
    """

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: types.ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Find a module with the finders after this one, and replace its loader if eligible."""
        finders = sys.meta_path
        start = finders.index(self) + 1 if self in finders else len(finders)
        for finder in finders[start:]:
            find_spec = getattr(finder, "find_spec", None)
            spec = find_spec(fullname, path, target) if find_spec is not None else None
            if spec is not None:
                break
        else:
            return None
        loader = spec.loader
        if (
            type(loader) is importlib.machinery.SourceFileLoader
            and _is_repository_file(_normalize(os.path.abspath(loader.path)))
            and not _is_test_file(loader.path)
        ):
            spec.loader = _FirstImportLoader(loader.name, loader.path)
        return spec


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


def _is_repository_file(file: str) -> bool:
    """Whether a normalized path is in the repository and outside ``site-packages``."""
    parts = set(pathlib.PurePath(file).parts)
    return file.startswith(_ROOT_DIRECTORY) and not {"site-packages", "dist-packages"} & parts


def _is_test_file(file: str) -> bool:
    """Whether a file is named as a test file or a ``conftest.py`` file."""
    name = os.path.basename(file)
    return name == "conftest.py" or name.endswith("_test.py") or name.startswith("test_")


def _normalize(path: str) -> str:
    """A path in the form that pytest and the operating system use, to compare it with others."""
    return os.path.normcase(os.path.normpath(path))


def _get_selected_test_files() -> set[str] | None:
    """The absolute paths of the test files to collect, or None to collect all of them."""
    test_files = os.environ.get(TEST_FILES_VARIABLE)
    return (
        None if test_files is None else {_normalize(file) for file in test_files.split(os.pathsep)}
    )


# The root directory of the repository, which contains the checks directory of this module.
_ROOT_DIRECTORY = _normalize(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + os.sep

# This module is imported as a pytest plugin before pytest loads any conftest.py file.
_coverage = _start_coverage()
_selected_test_files = _get_selected_test_files()
_current_context = [""]
_context_stack: list[str] = []
_shared_fixtures: dict[int, pytest.FixtureDef[object]] = {}
_executed_files: set[str] = set()
if _coverage is not None:
    sys.meta_path.insert(0, _FirstImportFinder())
