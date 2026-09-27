"""Miscellaneous internal utilities.

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

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import sys
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, TypeVar, cast

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
