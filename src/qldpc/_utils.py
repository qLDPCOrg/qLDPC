# SPDX-License-Identifier: Apache-2.0

"""Python-level plumbing: warning stack levels and docstring formatting."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TypeVar

_CallableType = TypeVar("_CallableType", bound=Callable[..., object])


####################################################################################################
# docstrings and warnings


def format_docstring(**substitutions: object) -> Callable[[_CallableType], _CallableType]:
    """Substitute named values into a function's docstring via str.format.

    This lets a docstring reference values (such as module-level constants) by name, for example
    "Default: {error_rate}.", without making the docstring an f-string.  An f-string cannot be used
    as a docstring: Python evaluates it as an ordinary expression and leaves __doc__ set to None,
    silently discarding the documentation.
    """

    def decorator(func: _CallableType) -> _CallableType:
        if func.__doc__ is not None:
            func.__doc__ = func.__doc__.format(**substitutions)
        return func

    return decorator


def get_external_caller_stacklevel() -> int:
    """Find the stacklevel of the first caller outside qLDPC implementation modules.

    Passing this stacklevel to warnings.warn attributes a warning to the user code that called into
    qLDPC, however deeply nested the call that emits the warning.  Test modules are co-located with
    the modules that they test (as qldpc.<...>_test), so they are treated as external callers.
    """
    stacklevel = 1
    frame = inspect.currentframe()
    if frame is None:  # pragma: no cover
        return 2
    frame = frame.f_back
    while frame is not None:
        module = str(frame.f_globals.get("__name__", ""))
        if not module.startswith("qldpc.") or module.endswith("_test"):
            break
        stacklevel += 1
        frame = frame.f_back
    return stacklevel
