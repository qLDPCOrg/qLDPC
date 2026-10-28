# SPDX-License-Identifier: Apache-2.0

"""Deprecated names, methods, arguments, and decoder construction of qLDPC.

All deprecation code lives in this private package.  The core module registers deprecated names,
defines deprecated methods, and provides the helpers that inline deprecated-argument blocks call.
The decoders module holds deprecated decoder construction, which qldpc.decoders serves through the
registry.  Modules with deprecated decoder arguments import it directly, and
install_deprecated_methods imports it when it runs, so this package does not import it when it is
imported.  The lookup module holds the deprecated direct construction of LookupDecoder and
WeightedLookupDecoder, and the lookup decoders that the deprecated predict_observable_flips=True
option builds.  It imports qldpc.decoders, so it is imported only within the functions that use it.
"""

from .core import (
    ALIASES,
    SILENT_ALIASES,
    defunct,
    get_deprecated_attribute,
    install_deprecated_methods,
    warn_deprecated,
)

__all__ = [
    "ALIASES",
    "SILENT_ALIASES",
    "defunct",
    "get_deprecated_attribute",
    "install_deprecated_methods",
    "warn_deprecated",
]
