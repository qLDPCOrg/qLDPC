# SPDX-License-Identifier: Apache-2.0

"""Tests of the cache clearing between test files that conftest.py provides."""

from __future__ import annotations

import functools
import pathlib
import types
from typing import cast

import pytest

from . import conftest


def test_hook_wraps_every_other_teardown(request: pytest.FixtureRequest) -> None:
    """Pytest runs the hook as the outermost wrapper of the teardown of each test."""
    hook_caller = request.config.pluginmanager.hook.pytest_runtest_teardown
    outermost = hook_caller.get_hookimpls()[-1]
    assert outermost.function is conftest.pytest_runtest_teardown
    assert outermost.wrapper


@pytest.mark.parametrize(
    ("next_file", "value", "cleared"),
    [
        ("a_test.py", "1", False),
        ("b_test.py", "1", True),
        (None, "1", True),
        ("b_test.py", None, False),
        ("b_test.py", "0", False),
    ],
)
def test_hook_clears_after_each_test_file(
    monkeypatch: pytest.MonkeyPatch, next_file: str | None, value: str | None, cleared: bool
) -> None:
    """After the rest of the teardown of the last test of a file, the hook empties the caches.

    It does so only if QLDPC_CLEAR_CACHES is "1".
    """
    calls: list[None] = []
    monkeypatch.setattr(conftest, "clear_caches", lambda: calls.append(None))
    if value is None:
        monkeypatch.delenv(conftest.CLEAR_CACHES_VARIABLE, raising=False)
    else:
        monkeypatch.setenv(conftest.CLEAR_CACHES_VARIABLE, value)

    nextitem = None if next_file is None else _make_item(next_file)
    teardown = conftest.pytest_runtest_teardown(_make_item("a_test.py"), nextitem)
    next(teardown)
    assert not calls
    with pytest.raises(StopIteration):
        next(teardown)
    assert calls == [None] * cleared


def test_caches_are_cleared() -> None:
    """The functools caches of the functions and static methods of a qLDPC module are emptied.

    Modules that are None, and attributes that are modules, are skipped.
    """
    module = _make_probe_module()
    module.function(1)
    module.Class.method(1)
    conftest.clear_caches({module.__name__: module, "qldpc.none": None})
    assert module.function.cache_info().currsize == 0
    assert module.Class.method.cache_info().currsize == 0


def test_caches_of_other_packages_are_kept() -> None:
    """The caches of modules outside of qLDPC are left alone."""
    module = _make_probe_module()
    module.function(1)
    conftest.clear_caches({"other._probe": module})
    assert module.function.cache_info().currsize == 1


def test_coverage_check_sets_the_variable() -> None:
    """The coverage check sets QLDPC_CLEAR_CACHES, which makes conftest.py empty caches, to "1"."""
    check = pathlib.Path(__file__).parents[2] / "checks" / "coverage_.py"
    text = check.read_text(encoding="utf-8")
    assert f'CLEAR_CACHES_VARIABLE = "{conftest.CLEAR_CACHES_VARIABLE}"' in text
    assert 'CLEAR_CACHES_VARIABLE: "1"' in text


def _make_probe_module() -> types.ModuleType:
    """A stand-in module of qLDPC, with a cached function and a cached static method.

    The module also holds a submodule whose __getattr__ fails the test, since clearing must not look
    up attributes of modules.
    """
    module = types.ModuleType("qldpc._cache_clearing_probe")

    @functools.cache
    def function(value: int) -> int:
        return value

    class Class:
        @staticmethod
        @functools.cache
        def method(value: int) -> int:
            return value

    Class.__module__ = module.__name__
    submodule = types.ModuleType("qldpc._cache_clearing_probe.submodule")
    vars(submodule)["__getattr__"] = lambda name: pytest.fail(f"looked up {name}")
    vars(module).update(function=function, Class=Class, submodule=submodule)
    return module


def _make_item(path: str) -> pytest.Item:
    """A stand-in for a test item of the test file at the given path."""
    return cast(pytest.Item, types.SimpleNamespace(path=pathlib.Path(path)))
