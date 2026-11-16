# SPDX-License-Identifier: Apache-2.0

"""Python-level plumbing: lazy imports, warning stack levels, and docstring formatting."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import sys
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, TypeVar

_CallableType = TypeVar("_CallableType", bound=Callable[..., object])


####################################################################################################
# lazy imports


def _lazy_import(name: str) -> ModuleType:
    """Import a module lazily, deferring its execution until its first attribute access.

    Uses importlib.util.LazyLoader so a heavy but rarely-used dependency stays out of ``import
    qldpc`` while call sites keep using ``module.attr`` exactly as if it had been imported eagerly.
    """
    if (module := sys.modules.get(name)) is not None:
        return module
    spec = importlib.util.find_spec(name)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    loader = _LoaderWithCleanup(spec.loader)
    spec.loader = importlib.util.LazyLoader(loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    loader.lazy_module_class = type(module)
    return module


class _LoaderWithCleanup(importlib.abc.Loader):
    """Loader proxy that restores lazy modules after failed execution."""

    def __init__(self, loader: importlib.abc.Loader) -> None:
        self.loader = loader
        self.lazy_module_class: type[ModuleType] | None = None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        """Create a module using the wrapped loader."""
        return self.loader.create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        """Execute a module and discard any partial state after failure."""
        spec = vars(module)["__spec__"]
        if spec.name not in sys.modules:
            sys.modules[spec.name] = module
        try:
            self.loader.exec_module(module)
        except BaseException:
            # restore the lazy-loading state so a failed import can be retried cleanly
            module_dict = vars(module)
            loader_state = spec.loader_state
            module_dict.clear()
            module_dict.update(loader_state["__dict__"])
            loader_state["is_loading"] = False
            if self.lazy_module_class is not None:
                object.__setattr__(module, "__class__", self.lazy_module_class)
            if sys.modules.get(spec.name) is module:
                del sys.modules[spec.name]
            raise


# networkx is loaded lazily, deferring its ~110 ms import until it is first used.  Call sites import
# it as ``from qldpc._utils import networkx as nx``.
if TYPE_CHECKING:
    import networkx
else:
    networkx = _lazy_import("networkx")


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
