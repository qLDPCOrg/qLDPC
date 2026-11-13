# SPDX-License-Identifier: Apache-2.0

"""Exact distances of binary codes, counts of their logical operators, and searches of cosets.

The functions and classes of this package take plain binary arrays, so that programs can use them
without building code objects, and the distance methods of the code classes of :mod:`qldpc.codes`
call them.

.. list-table::
   :header-rows: 1

   * - Task
     - Entry point
   * - The distance of a classical code, from its generator or parity-check matrix
     - :func:`get_distance_from_generators` or :func:`get_distance_from_checks`
   * - The minimum weight of a nontrivial logical operator of a quantum code, from its logical
       operators and its stabilizers, or from checks and the rows to exclude
     - :func:`get_distance_from_generators` or :func:`get_distance_from_checks`, with
       ``symplectic=True`` for Pauli strings
   * - The distance, or the X- or Z-distance, of a CSS stabilizer or subsystem code, from its X and
       Z checks
     - :func:`get_distance_css`, with ``pauli`` choosing the type
   * - The distance of a stabilizer or subsystem code, from its checks as Pauli strings
     - :func:`get_distance_pauli`
   * - The number of logical operators of a given weight in a CSS code
     - :class:`CSSLogicalCounter`, or :func:`count_logical_ops_css` for one count
   * - Whether Pauli errors with given syndromes on a stabilizer state weigh at most a given weight,
       and the minimum weights of such errors
     - :class:`CosetLeaderSearch`
   * - Many counts of similar codes, faster
     - :func:`compile_search_setup`

Matrices are binary: NumPy arrays, nested lists, ``galois`` arrays over GF(2), or SciPy sparse
matrices.  A Pauli string on ``n`` qubits is a row ``[P_x|P_z]`` of ``2 n`` entries, the supports of
its X and Z parts.  The ``method`` argument of the distance searches takes a
:data:`DistanceMethod`, and a search without a nontrivial word returns :data:`NO_DISTANCE`.

The inner loops of the exact searches are compiled by numba the first time that they are used, and
the compiled code is cached on disk.  The cache is a directory named ``qldpc-<hash>``, where the
hash is of the Python source files of qLDPC.  It is inside the directory of the
``NUMBA_CACHE_DIR`` environment variable if that is set, and otherwise inside the ``qldpc/numba``
directory of the user's cache directory.  A change to those source files selects a new directory, so
outdated code is never loaded, and each import of qLDPC deletes the directories of earlier versions
of its own source files that no import has used for more than 7 days.  If the directory cannot be
used, qLDPC warns, and compiles again in every process.  Set ``NUMBA_CACHE_DIR`` to a writable
directory, such as one that many jobs share, to avoid this.

:func:`~qldpc.codes.distance.compile_search_setup` also compiles the code that sets those loops up,
in the current process only, for programs that count the logical operators of very many similar
codes.

Exact distance searches and logical-operator counts choose between routes, such as exhaustive
enumeration and searches of overlapping information sets, by cost models, and bound their batches,
memory, and caches.  The defaults of their settings suit typical searches.  For very costly
searches, the page :doc:`/exact_search_tuning` of the documentation leads from symptoms and routes
to the private settings that control them.  It also says how to change these settings, and what
changing them can and cannot affect.

Contributors can find a map of the modules in the README.md of this directory.
"""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from .core import (
    NO_DISTANCE,
    DistanceMethod,
    NoDistance,
    get_distance_css,
    get_distance_from_checks,
    get_distance_from_generators,
    get_distance_pauli,
)
from .logical_counts import (
    CSSLogicalCounter,
    CSSLogicalCounts,
    LogicalCountMethod,
    compile_search_setup,
    count_logical_ops_css,
)

__all__ = [
    "NO_DISTANCE",
    "CSSLogicalCounter",
    "CSSLogicalCounts",
    "DistanceMethod",
    "LogicalCountMethod",
    "NoDistance",
    "compile_search_setup",
    "count_logical_ops_css",
    "get_distance_css",
    "get_distance_from_checks",
    "get_distance_from_generators",
    "get_distance_pauli",
]

if not TYPE_CHECKING:  # DEPRECATED: see qldpc._deprecations

    def __getattr__(name: str) -> object:
        return _deprecations.get_deprecated_attribute(__name__, name)
