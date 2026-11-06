# SPDX-License-Identifier: Apache-2.0

"""Tools for constructing and analyzing quantum low density parity check (qLDPC) codes."""

import importlib.metadata

from . import (
    _deprecations,
    abstract,
    cache,
    circuits,
    codes,
    decoders,
    experimental,
    external,
    math,
    objects,
)

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
