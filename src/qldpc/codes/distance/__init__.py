# SPDX-License-Identifier: Apache-2.0

"""Exact binary distance calculations, logical-operator counts, and distance method selectors.

The inner loops of the exact searches are compiled by numba the first time that they are used, and
the compiled code is cached on disk.  The cache is a directory named ``qldpc-<hash>``, where the
hash is of the Python source files of this package.  It is inside the directory of the
``NUMBA_CACHE_DIR`` environment variable if that is set, and otherwise inside the ``qldpc/numba``
directory of the user's cache directory.  A change to those source files selects a new directory, so
outdated code is never loaded, and old directories can be deleted.  If the directory cannot be used,
qLDPC warns, and compiles again in every process.  Set ``NUMBA_CACHE_DIR`` to a writable directory,
such as one that many jobs share, to avoid this.

:func:`~qldpc.codes.distance.compile_search_setup` also compiles the code that sets those loops up,
in the current process only, for programs that count the logical operators of very many similar
codes.

Contributors can find a map of the modules in the README.md of this directory.
"""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from .core import (
    DistanceBackend,
    DistanceMethod,
    get_distance_classical,
    get_distance_quantum,
)
from .logical_counts import (
    CSSLogicalCounter,
    CSSLogicalCounts,
    LogicalCountMethod,
    compile_search_setup,
    count_logical_ops_css,
)

__all__ = [
    "CSSLogicalCounter",
    "CSSLogicalCounts",
    "DistanceBackend",
    "DistanceMethod",
    "LogicalCountMethod",
    "compile_search_setup",
    "count_logical_ops_css",
    "get_distance_classical",
    "get_distance_quantum",
]

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
