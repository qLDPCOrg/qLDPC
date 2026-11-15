# SPDX-License-Identifier: Apache-2.0

"""The answer to a query about the lightest Pauli errors with a given syndrome.

The queries ``get_min_weight_bound`` and ``find_first_not_within_max_weight`` of
:class:`~qldpc.codes.distance.CosetLeaderSearch` answer with a :class:`MinWeightBound`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinWeightBound:
    """Whether a syndrome class has an error within a maximum weight, with an error and a bound.

    A query of :class:`~qldpc.codes.distance.CosetLeaderSearch` asks whether some Pauli error with a
    given syndrome has weight at most ``max_weight``.  Whether it has one never depends on the
    method or tuning of the search.  Which error and which lower bound the query returns can depend
    on the method, the tuning, and the version of qLDPC, within the guarantees below.

    Attributes:
        within_max_weight: Whether some error with the syndrome has weight at most ``max_weight``.
        representative: An error with the syndrome, packed into an integer with its X part in bits
            ``0`` to ``n - 1`` and its Z part in bits ``n`` to ``2 n - 1``, for ``n`` qubits
            (see :meth:`~qldpc.codes.distance.coset_leaders.CosetLeaderSearch.unpack_pauli`).  If
            ``within_max_weight`` is True, its weight is at most ``max_weight``, but it need not be
            a lightest error with the syndrome.
        lower_bound: A lower bound on the weight of every error with the syndrome.  If
            ``within_max_weight`` is False, it exceeds ``max_weight``.
    """

    within_max_weight: bool
    representative: int
    lower_bound: int
