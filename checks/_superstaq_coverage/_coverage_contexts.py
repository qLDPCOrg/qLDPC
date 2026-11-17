# Copyright 2026 Infleqtion, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A pytest plugin that labels the lines that run with the test file responsible for them.

`coverage_.py --modular --single-run` loads this plugin with `-p` in one pytest-xdist run of the
selected test files, with `--dist=loadfile`.  The plugin starts coverage before any `conftest.py`
file is imported, and labels the lines that run with these coverage contexts:

* "" while pytest starts and collects (outside the collection of a directory or test file), and
  while a module of the repository runs for the first time in a process;
* "<directory or test file>|collect" while a directory or a test file (or its classes) is collected;
* "<test file>|setup", "|call", or "|teardown" while a test of the test file runs; and
* "unattributed" at other times, such as between tests.

`coverage_.py` credits a file only with the lines of "", of its test file, and of the collection of
the directories that contain its test file.  Unlike plain `--modular`, which ignores a file that its
test file never imports, `coverage_.py` checks such a file too (which fails at 0% if nothing runs
it).  After the last test of a test file on a worker, the plugin tears down the session- and
package-scoped fixtures defined in the repository that the test file used, so that the next test
file sets them up anew.

Known approximations of running each test file in a process of its own:

* The lines that run while a module of the repository is first imported, including its calls into
  other modules, are credited to every test file.  So a file can pass here but fail plain
  `--modular` if only the imports of other test files run some of its lines.
* Other state that a test file leaves behind, such as caches, persists (a repository can reset it
  in its own `conftest.py`).
* The `conftest.py` files that pytest loads at startup are credited to every test file.
* The first-import rule does not apply to modules that pytest rewrites for assertions.
* Lines that run in subprocesses are not measured, and those that run in other threads get whatever
  context is current.
"""

from __future__ import annotations

import importlib.machinery
import inspect
import os
import pathlib
import sys
from collections.abc import Generator, Sequence
from types import ModuleType

import coverage
import pytest

# Environment variables set by coverage_.py (which duplicates these names).
_DATA_FILE_ENV = "CHECKS_SUPERSTAQ_COVERAGE_DATA_FILE"
_INCLUDE_ENV = "CHECKS_SUPERSTAQ_COVERAGE_INCLUDE"
_BRANCH_ENV = "CHECKS_SUPERSTAQ_COVERAGE_BRANCH"


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector: pytest.Collector) -> Generator[None, object, None]:
    """Label the lines that run while a directory or test file is collected."""
    context = _context[0]
    path = collector.nodeid.split("::")[0]
    _switch_context(f"{path}|collect" if path else "")
    yield
    _switch_context(context)


def pytest_collection_finish() -> None:
    """Stop labeling lines as part of the startup of pytest."""
    _switch_context("unattributed")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Label the setup of a test, and note the shared fixtures that its test file uses."""
    for fixturedefs in item._fixtureinfo.name2fixturedefs.values():  # type: ignore[attr-defined]
        for fixturedef in fixturedefs:
            if fixturedef.scope in ("session", "package") and _is_repository_file(
                inspect.getfile(fixturedef.func)
            ):
                _shared_fixtures[id(fixturedef)] = fixturedef
    _switch_context(f"{_get_test_file(item)}|setup")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_call(item: pytest.Item) -> None:
    """Label the body of a test."""
    _switch_context(f"{_get_test_file(item)}|call")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(
    item: pytest.Item, nextitem: pytest.Item | None
) -> Generator[None, object, None]:
    """Label the teardown of a test, and tear down shared fixtures after a test file ends."""
    _switch_context(f"{_get_test_file(item)}|teardown")
    try:
        yield
        if nextitem is None or _get_test_file(nextitem) != _get_test_file(item):
            # Finish in the reverse order of first use.  Finishing a shared fixture first finishes
            # the fixtures that depend on it, such as module-scoped ones.
            for fixturedef in reversed(list(_shared_fixtures.values())):
                fixturedef.finish(item._request)  # type: ignore[attr-defined]
            _shared_fixtures.clear()
    finally:
        _switch_context("unattributed")


@pytest.hookimpl(trylast=True)
def pytest_unconfigure() -> None:
    """Save the coverage data of this process."""
    _coverage.stop()
    _coverage.save()


class _FirstImportLoader(importlib.machinery.SourceFileLoader):
    """A source loader that runs a module in the context "" the first time that it runs."""

    def exec_module(self, module: ModuleType) -> None:
        """Run a module, in the context "" if this is its first run in the process."""
        if self.path in _executed_files:
            super().exec_module(module)
            return
        _executed_files.add(self.path)
        context = _context[0]
        _switch_context("")
        try:
            super().exec_module(module)
        finally:
            _switch_context(context)


class _FirstImportFinder:
    """A meta path finder that loads the modules of the repository with `_FirstImportLoader`."""

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        """Find a module with the other finders, and replace its loader if it is eligible."""
        for finder in sys.meta_path[sys.meta_path.index(self) + 1 :]:
            find_spec = getattr(finder, "find_spec", None)
            if find_spec and (spec := find_spec(fullname, path, target)):
                break
        else:
            return None
        loader = spec.loader
        # Other loaders, such as that of pytest's assertion rewriting, are left alone.
        if type(loader) is importlib.machinery.SourceFileLoader and _is_repository_file(
            loader.path
        ):
            spec.loader = _FirstImportLoader(loader.name, loader.path)
        return spec


def _switch_context(context: str) -> None:
    """Label the lines that run from now on."""
    _coverage.switch_context(context)
    _context[0] = context


def _is_repository_file(path: str) -> bool:
    """Return whether a file is in the repository, other than in its `site-packages`."""
    return path.startswith(_ROOT_DIR) and "site-packages" not in pathlib.PurePath(path).parts


def _get_test_file(item: pytest.Item) -> str:
    """Return the test file of a test, as in its node ID."""
    return item.nodeid.split("::")[0]


# coverage_.py runs pytest in the root directory of the repository.
_ROOT_DIR = os.path.join(os.getcwd(), "")
_context = [""]
_shared_fixtures: dict[int, pytest.FixtureDef[object]] = {}
_executed_files: set[str] = set()

# pytest imports this plugin before any conftest.py file.
_coverage = coverage.Coverage(
    data_file=os.environ[_DATA_FILE_ENV],
    data_suffix=True,
    include=os.environ[_INCLUDE_ENV].split(os.pathsep),
    branch=bool(os.environ[_BRANCH_ENV]) or None,  # None defers to the configured run:branch.
)
# The sys.monitoring core records each line only once, so it can't label lines that run again.
_coverage.set_option("run:core", "ctrace")
# Lines that run in subprocesses would not be labeled, so they are not measured.
_coverage.set_option("run:patch", [])
# A process, such as the pytest-xdist controller, may run no measured lines.
_coverage.set_option("run:disable_warnings", ["no-data-collected"])
_coverage.start()
sys.meta_path.insert(0, _FirstImportFinder())
