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

# Subpackages that are imported on first access, to keep "import qldpc" fast.  The subpackages that
# "import qldpc" loads must not import them at import time, which import_test.py checks.
_LAZY_SUBPACKAGES = ("circuits", "experimental")

# Type checkers see the imports rather than __getattr__, so that they still reject unknown names.
if TYPE_CHECKING:
    from . import circuits, experimental
else:

    def __getattr__(name: str) -> object:
        if name in _LAZY_SUBPACKAGES:
            return importlib.import_module(f"{__name__}.{name}")
        return _deprecations.get_deprecated_attribute(__name__, name)


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_SUBPACKAGES})
