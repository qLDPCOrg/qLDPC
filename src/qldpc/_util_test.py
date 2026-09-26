"""Unit tests for _util.py.

Copyright 2026 The qLDPC Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import importlib
import pathlib
import sys
import uuid
from types import ModuleType

import pytest

from qldpc._util import format_docstring, lazy_import


def test_lazy_import() -> None:
    """A lazily imported module is only executed on first attribute access."""
    name = "colorsys"  # a small stdlib module not otherwise imported by the test suite
    sys.modules.pop(name, None)

    module = lazy_import(name)  # uncached path
    assert isinstance(module, ModuleType)
    assert callable(module.rgb_to_hls)  # force the deferred execution

    assert lazy_import(name) is module  # cached path returns the same module


def test_lazy_import_failure(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed lazy import cannot leave behind a usable partial module."""
    name = f"broken_lazy_module_{uuid.uuid4().hex}"
    module_path = tmp_path / f"{name}.py"
    module_path.write_text('value = 1\nraise RuntimeError("broken install")\n')
    monkeypatch.syspath_prepend(str(tmp_path))

    module = lazy_import(name)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="broken install"):
            _ = module.value
        assert name not in sys.modules

    module_path.write_text("value = 2\n")
    importlib.invalidate_caches()
    assert module.value == 2
    assert lazy_import(name) is module
    sys.modules.pop(name, None)

    with pytest.raises(ModuleNotFoundError, match="No module named"):
        lazy_import(f"missing_{name}")


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
