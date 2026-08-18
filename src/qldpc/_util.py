"""Miscellaneous internal utilities.

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

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, TypeVar

CallableType = TypeVar("CallableType", bound=Callable[..., object])


def lazy_import(name: str) -> ModuleType:
    """Import a module lazily, deferring its execution until its first attribute access.

    Uses importlib.util.LazyLoader so a heavy but rarely-used dependency stays out of ``import
    qldpc`` while call sites keep using ``module.attr`` exactly as if it had been imported eagerly.
    """
    if (module := sys.modules.get(name)) is not None:
        return module
    spec = importlib.util.find_spec(name)
    assert spec is not None and spec.loader is not None
    spec.loader = importlib.util.LazyLoader(spec.loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
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
