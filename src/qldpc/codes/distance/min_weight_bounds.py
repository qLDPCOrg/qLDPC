# SPDX-License-Identifier: Apache-2.0

"""The answer of a query of :class:`~qldpc.codes.distance.CosetLeaderSearch` for one syndrome."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MinWeightBound:
    """Whether a syndrome class has an error within a maximum weight, with an error and a bound.

    The queries ``get_min_weight_bound`` and ``find_first_not_within_max_weight`` of
    :class:`~qldpc.codes.distance.CosetLeaderSearch` return it.  Which error and which lower bound
    they return can depend on the method, the tuning, and the version of qLDPC, within the
    guarantees below, but ``within_max_weight`` cannot.

    Attributes:
        within_max_weight: Whether some error with the syndrome has weight at most ``max_weight``.
        representative: An error with the syndrome, packed into an integer: for ``n`` qubits, bits
            ``0`` to ``n - 1`` hold its X part, and bits ``n`` to ``2 n - 1`` its Z part (see
            :meth:`~qldpc.codes.distance.CosetLeaderSearch.unpack_pauli`).  If ``within_max_weight``
            is True, its weight is at most ``max_weight``, but it need not be a lightest error with
            the syndrome.
        lower_bound: A lower bound on the weight of every error with the syndrome.  If
            ``within_max_weight`` is False, it exceeds ``max_weight``.
    """

    within_max_weight: bool
    representative: int
    lower_bound: int
