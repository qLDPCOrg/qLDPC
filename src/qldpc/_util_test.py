# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _util.py."""

import importlib
import pathlib
import sys
import uuid
import warnings
from types import ModuleType

import pytest

from qldpc import decoders
from qldpc._util import (
    format_docstring,
    get_deprecated_alias,
    get_external_caller_stacklevel,
    lazy_import,
)


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


def test_get_deprecated_alias() -> None:
    """A deprecated alias resolves to its replacement, warning at the caller's line."""

    class NewName: ...

    aliases = {"OldName": NewName}
    with pytest.warns(DeprecationWarning, match="OldName is deprecated; use NewName instead") as w:
        assert get_deprecated_alias("some.module", "OldName", aliases) is NewName
    assert w[0].filename == __file__

    with pytest.raises(AttributeError, match=r"module 'some\.module' has no attribute 'Other'"):
        get_deprecated_alias("some.module", "Other", aliases)


def test_deprecated_alias_import_warns_once() -> None:
    """Importing a deprecated name from a package warns once, although Python looks it up twice."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        from qldpc.decoders import Decoder
    assert Decoder is decoders.ErrorDecoder
    assert [str(record.message) for record in records] == [
        "Decoder is deprecated; use ErrorDecoder instead"
    ]
    assert records[0].filename == __file__


def test_get_external_caller_stacklevel() -> None:
    """Test modules are external callers, so a call from here has stacklevel 1."""
    assert get_external_caller_stacklevel() == 1
