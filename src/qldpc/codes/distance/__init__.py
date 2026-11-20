# SPDX-License-Identifier: Apache-2.0

"""Exact distances of binary codes, counts of their logical operators, and searches of cosets.

The functions and classes of this package take plain binary arrays rather than code objects.  The
exact-distance methods of the code classes in :mod:`qldpc.codes` call its distance functions for
binary codes.

.. list-table::
   :header-rows: 1

   * - Task
     - Entry point
   * - The distance of a classical code, from its generator or parity check matrix
     - :func:`~qldpc.codes.distance.get_distance_from_generators` or
       :func:`~qldpc.codes.distance.get_distance_from_checks`
   * - The minimum weight of a nontrivial logical operator of a quantum code, from its logical
       operators and its stabilizers, or from checks and the rows to exclude
     - :func:`~qldpc.codes.distance.get_distance_from_generators` or
       :func:`~qldpc.codes.distance.get_distance_from_checks`, with ``symplectic=True`` for Pauli
       strings
   * - The distance, or the X- or Z-distance, of a CSS stabilizer code, or the dressed ones of a
       CSS subsystem code, from its X and Z checks
     - :func:`~qldpc.codes.distance.get_distance_css`, with ``pauli`` choosing the type
   * - The distance of a stabilizer code, or the dressed distance of a subsystem code, from its
       checks as Pauli strings
     - :func:`~qldpc.codes.distance.get_distance_pauli`
   * - The bare distance of a subsystem code, from its gauge generators and stabilizers
     - :func:`~qldpc.codes.distance.get_distance_from_checks`, whose description gives the recipe
   * - The number of logical operators of a given weight in a CSS code
     - :class:`~qldpc.codes.distance.CSSLogicalCounter`, or
       :func:`~qldpc.codes.distance.count_logical_ops_css` for one count
   * - Whether some Pauli error with a given syndrome on a pure stabilizer state weighs at most a
       given weight, and the weight of the lightest such errors
     - :class:`~qldpc.codes.distance.CosetLeaderSearch`
   * - Many counts of similar codes, faster
     - :func:`~qldpc.codes.distance.compile_search_setup`

Matrices are binary: NumPy arrays, nested lists, ``galois`` arrays over GF(2), or SciPy sparse
matrices, and a single vector is one row.  A Pauli string on ``n`` qubits is a row ``[P_x|P_z]`` of
``2 n`` entries, the supports of its X and Z parts.  The ``method`` argument takes a
:data:`~qldpc.codes.distance.DistanceMethod` in the distance searches, a
:data:`~qldpc.codes.distance.CosetMethod` in the coset-leader search, and a
:data:`~qldpc.codes.distance.LogicalCountMethod` in counts.  Each type lists only the methods of
its problem, and its ``"auto"`` chooses one of them.  A distance search without a nontrivial word
returns :data:`~qldpc.codes.distance.NO_DISTANCE`.

Numba compiles the inner loops of the searches on first use, and caches them on disk in a
directory named by a hash of qLDPC's source files, inside that of the ``NUMBA_CACHE_DIR``
environment variable if that is set (see "Disk caches" in the :doc:`library map </library_map>`).
If the cache cannot be used, qLDPC warns, and compiles again in every process; set
``NUMBA_CACHE_DIR`` to a writable directory, such as one that many jobs share, to avoid this.
:func:`~qldpc.codes.distance.compile_search_setup` also compiles the code that sets those loops up,
in the current process only, for programs that count the logical operators of very many similar
codes.

Settings, whose defaults suit typical searches, control how exact searches choose their routes by
cost models, and how they bound their batches, memory, and caches.  For very costly searches,
:doc:`/exact_search_tuning` leads from symptoms and routes to these private settings, and says how
to change them and what that can and cannot affect.

Contributors can find a map of the modules in the `README of the package
<https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/README.md>`_.
"""

from typing import TYPE_CHECKING

from qldpc import _deprecations

from .coset_leaders import CosetLeaderSearch, CosetMethod
from .distances import (
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
from .min_weight_bounds import MinWeightBound

__all__ = [
    "NO_DISTANCE",
    "CSSLogicalCounter",
    "CSSLogicalCounts",
    "CosetLeaderSearch",
    "CosetMethod",
    "DistanceMethod",
    "LogicalCountMethod",
    "MinWeightBound",
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
