# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _utils.py."""

import importlib
import pathlib
import sys
import uuid
from types import FunctionType, ModuleType

import pytest

from qldpc._utils import (
    _lazy_import,
    format_docstring,
    get_external_caller_stacklevel,
)


def test_lazy_import() -> None:
    """A lazily imported module is only executed on first attribute access."""
    name = "colorsys"  # a small stdlib module not otherwise imported by the test suite
    sys.modules.pop(name, None)

    module = _lazy_import(name)  # uncached path
    assert isinstance(module, ModuleType)
    assert callable(module.rgb_to_hls)  # force the deferred execution

    assert _lazy_import(name) is module  # cached path returns the same module


def test_lazy_import_failure(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed lazy import cannot leave behind a usable partial module."""
    name = f"broken_lazy_module_{uuid.uuid4().hex}"
    module_path = tmp_path / f"{name}.py"
    module_path.write_text('value = 1\nraise RuntimeError("broken install")\n')
    monkeypatch.syspath_prepend(str(tmp_path))

    module = _lazy_import(name)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="broken install"):
            _ = module.value
        assert name not in sys.modules

    module_path.write_text("value = 2\n")
    importlib.invalidate_caches()
    assert module.value == 2
    assert _lazy_import(name) is module
    sys.modules.pop(name, None)

    with pytest.raises(ModuleNotFoundError, match="No module named"):
        _lazy_import(f"missing_{name}")


def test_format_docstring() -> None:
    """Named values are substituted into a docstring."""

    @format_docstring(value=1e-3, name="tag")
    def func() -> None:
        """A docstring with a {value} and a {name}."""

    assert func.__doc__ == "A docstring with a 0.001 and a tag."


def test_format_docstring_without_docstring() -> None:
    """A function without a docstring is left untouched."""

    @format_docstring(value=1)
    def func() -> None:
        return None

    assert func.__doc__ is None
    assert func() is None


def test_get_external_caller_stacklevel() -> None:
    """Test modules are external callers, so a call from here has stacklevel 1."""
    assert get_external_caller_stacklevel() == 1

    # a call from a qLDPC module is attributed to the caller of that module
    def get_stacklevel() -> int:
        return get_external_caller_stacklevel()

    namespace = {
        "__name__": "qldpc.module",
        "get_external_caller_stacklevel": get_external_caller_stacklevel,
    }
    assert FunctionType(get_stacklevel.__code__, namespace)() == 2
