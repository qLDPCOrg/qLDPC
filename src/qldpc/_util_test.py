"""Unit tests for _util.py.

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

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

import sys
from types import ModuleType

from qldpc._util import format_docstring, lazy_import


def test_lazy_import() -> None:
    """A lazily imported module is only executed on first attribute access."""
    name = "colorsys"  # a small stdlib module not otherwise imported by the test suite
    sys.modules.pop(name, None)

    module = lazy_import(name)  # uncached path
    assert isinstance(module, ModuleType)
    assert callable(module.rgb_to_hls)  # force the deferred execution

    assert lazy_import(name) is module  # cached path returns the same module


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
