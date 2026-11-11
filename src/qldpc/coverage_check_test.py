# SPDX-License-Identifier: Apache-2.0

"""Tests of the coverage check in checks/coverage_.py, mostly with its run of pytest stubbed."""

from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import types
from collections.abc import Iterator

import pytest

from . import conftest

_CHECKS = pathlib.Path(__file__).parents[2] / "checks"

# The module without a test file in the repository of the check fixture, as the check names it.
_UNTESTED = os.path.join("src", "x.py")


def test_untested_modules_are_found_from_the_root(check: types.ModuleType) -> None:
    """Modules under src/ without a test file are found, but not other files that need none.

    The values of -j and --shard are not taken as files, and --skip coverage finds no modules.
    """
    args = ["src/x.py", "src/y.py", "src/y_test.py", "src/conftest.py", "src/__init__.py"]
    assert check.get_untested_files([*args, "other/z.py"]) == [_UNTESTED]
    assert check.get_untested_files(["src/x.py", "-j2", "--shard", "1/2"]) == [_UNTESTED]
    assert check.get_untested_files(["src/x.py", "--skip", "coverage"]) == []


def test_untested_modules_are_found_from_a_subdirectory(
    check: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """The check finds the modules of relative paths from a subdirectory too."""
    monkeypatch.chdir(tmp_path / "src")
    assert check.get_untested_files(["x.py", "y.py"]) == [_UNTESTED]


@pytest.mark.parametrize(
    ("args", "returncode", "expected"),
    [
        (["src/y.py"], 0, 0),
        (["src/y.py"], 2, 2),
        (["src/x.py"], 0, 1),
        (["src/x.py"], 2, 2),
    ],
)
def test_run_reports_untested_modules(
    check: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    returncode: int,
    expected: int,
) -> None:
    """The check fails a module without a test file, after the run of checks-superstaq."""
    calls: list[tuple[tuple[str, ...], str | None]] = []

    def run(*run_args: str, exclude: object) -> int:
        assert exclude == check.EXCLUDE
        calls.append((run_args, os.environ.get(conftest.CLEAR_CACHES_VARIABLE)))
        return returncode

    monkeypatch.setattr(check.coverage_, "run", run)
    assert check.run(*args) == expected
    [(run_args, variable)] = calls
    assert run_args[-len(args) :] == tuple(args)
    assert {"--modular", "--single-run", "-pno:nbmake", "-mnot(compiled)"} <= set(run_args)
    assert variable == "1"
    output = capsys.readouterr().out
    assert (f"No test file for {_UNTESTED}" in output) == ("src/x.py" in args)
    assert ("COVERAGE FAILURE!" in output) == (expected == 1)


def test_help_is_that_of_checks_superstaq(
    check: types.ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    """-h prints the options of the check of checks-superstaq, such as --single-run."""
    with pytest.raises(SystemExit):
        check.run("-h")
    assert "--single-run" in capsys.readouterr().out


def test_pytest_processes_find_the_plugin(check: types.ModuleType) -> None:
    """The pytest processes of the check, which inherit its environment, find its plugin.

    They run in the root directory of the repository, as checks-superstaq needs.
    """
    code = f"import importlib.util; assert importlib.util.find_spec({check.coverage_._PLUGIN!r})"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=_CHECKS.parent)


@pytest.fixture
def check(monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path) -> Iterator[types.ModuleType]:
    """The module checks/coverage_.py, run in a repository in tmp_path.

    The repository has an untested module src/x.py, a tested module src/y.py, and other files that
    need no test file.
    """
    # Importing and running the check change the environment and sys.modules, which are restored.
    monkeypatch.setenv("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    monkeypatch.setenv(conftest.CLEAR_CACHES_VARIABLE, "0")
    monkeypatch.syspath_prepend(str(_CHECKS))
    modules = set(sys.modules)
    spec = importlib.util.spec_from_file_location("_coverage_check", _CHECKS / "coverage_.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The repository of the check.
    files = ["src/x.py", "src/y.py", "src/y_test.py", "src/conftest.py", "src/__init__.py"]
    for path in [*files, "other/z.py"]:
        (tmp_path / path).parent.mkdir(exist_ok=True)
        (tmp_path / path).write_text('"""A module."""\n')
    monkeypatch.setattr(module.check_utils, "root_dir", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    yield module
    for name in set(sys.modules) - modules:
        del sys.modules[name]
