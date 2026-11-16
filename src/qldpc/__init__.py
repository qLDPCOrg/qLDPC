# SPDX-License-Identifier: Apache-2.0

"""Tools for constructing and analyzing quantum low density parity check (qLDPC) codes."""

import importlib
import importlib.metadata
from typing import TYPE_CHECKING

from . import _deprecations, abstract, cache, codes, decoders, external, math, objects

__version__ = importlib.metadata.version("qldpc")

__all__ = [
    "__version__",
    "abstract",
    "cache",
    "circuits",
    "codes",
    "decoders",
    "experimental",
    "external",
    "math",
    "objects",
]

_deprecations.install_deprecated_methods()

# Subpackages that are imported on first access, to keep "import qldpc" fast.  No other subpackage
# imports them at import time (see qldpc/import_test.py).
_LAZY_SUBPACKAGES = ("circuits", "experimental")

if TYPE_CHECKING:
    from . import circuits, experimental
else:

    def __getattr__(name: str) -> object:
        if name in _LAZY_SUBPACKAGES:
            return importlib.import_module(f"{__name__}.{name}")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_SUBPACKAGES})
