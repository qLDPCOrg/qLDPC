# SPDX-License-Identifier: Apache-2.0

"""Miscellaneous internal utilities."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import inspect
import sys
import warnings
from collections.abc import Callable, Mapping
from types import FrameType, ModuleType
from typing import TYPE_CHECKING, Any, TypeVar, cast

CallableType = TypeVar("CallableType", bound=Callable[..., object])


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


def lazy_import(name: str) -> ModuleType:
    """Import a module lazily, deferring its execution until its first attribute access.

    Uses importlib.util.LazyLoader so a heavy but rarely-used dependency stays out of ``import
    qldpc`` while call sites keep using ``module.attr`` exactly as if it had been imported eagerly.
    """
    if (module := sys.modules.get(name)) is not None:
        return module
    spec = importlib.util.find_spec(name)
    if spec is None or spec.loader is None:
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    loader = _LoaderWithCleanup(cast(importlib.abc.Loader, spec.loader))
    spec.loader = importlib.util.LazyLoader(loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    loader.lazy_module_class = type(module)
    return module


# networkx is loaded lazily to keep it and its ~110 ms import out of ``import qldpc``.
# Call sites import it as ``from qldpc._util import networkx as nx``.
if TYPE_CHECKING:
    import networkx
else:
    networkx = lazy_import("networkx")


def format_docstring(**substitutions: object) -> Callable[[CallableType], CallableType]:
    """Substitute named values into a function's docstring via str.format.

    This lets a docstring reference values (such as module-level constants) by name, for example
    "Default: {error_rate}.", without making the docstring an f-string.  An f-string cannot be used
    as a docstring: Python evaluates it as an ordinary expression and leaves __doc__ set to None,
    silently discarding the documentation.
    """

    def decorator(func: CallableType) -> CallableType:
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


def get_deprecated_alias(module_name: str, name: str, aliases: Mapping[str, type]) -> Any:
    """Retrieve the replacement for a deprecated name, and warn that the name is deprecated.

    This function backs module-level __getattr__ functions (PEP 562), which Python calls only for
    names that a module does not define.  A deprecated name thereby refers to the same object as its
    replacement, which preserves isinstance checks, subclassing, and unpickling, while still warning
    whenever the deprecated name is accessed.  For example::

        DEPRECATED_ALIASES = {"OldName": NewName}

        if TYPE_CHECKING:
            OldName = NewName  # so that type checkers still flag misspelled names
        else:

            def __getattr__(name: str) -> Any:
                return get_deprecated_alias(__name__, name, DEPRECATED_ALIASES)

    A package that re-exports a deprecated name should define the same kind of __getattr__ in its
    __init__.py, rather than importing the deprecated name (which would warn at import time), and
    may list the deprecated name in __all__ so that star imports still define it.

    Args:
        module_name: The name of the module whose attribute is being retrieved.
        name: The name of the attribute being retrieved.
        aliases: A map from each deprecated name in the module to its replacement.

    Returns:
        The replacement for the deprecated name.

    Raises:
        AttributeError: If the name is not a deprecated alias.
    """
    if name not in aliases:
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    replacement = aliases[name]
    if not _is_import_probe(sys._getframe(2)):
        warnings.warn(
            f"{name} is deprecated; use {replacement.__name__} instead",
            DeprecationWarning,
            stacklevel=get_external_caller_stacklevel(),
        )
    return replacement


def _is_import_probe(frame: FrameType | None) -> bool:
    """Is a frame the probe that checks for names before a ``from package import ...`` statement?

    Python checks that a package provides each name in ``from package import name`` before it
    retrieves the name, so a module-level __getattr__ is called twice for one such statement.
    """
    return (
        frame is not None
        and frame.f_code.co_name == "_handle_fromlist"
        and frame.f_globals.get("__name__") == "importlib._bootstrap"
    )
