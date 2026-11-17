# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _utils.py."""

from types import FunctionType

from qldpc._utils import format_docstring, get_external_caller_stacklevel


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
