# SPDX-License-Identifier: Apache-2.0

"""The smallest weights that a basis of logical operators of one Pauli type can have."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import numpy.typing as npt

from qldpc import cache, codes
from qldpc._binary import get_independent_rows, get_kernel
from qldpc.codes.distance import NO_DISTANCE, get_distance_from_generators
from qldpc.objects import PauliXZLike

from ._sectors import LogicalOpsKind, get_sector

####################################################################################################
# weight spectra


@dataclasses.dataclass(frozen=True, eq=False)
class WeightSpectrum:
    """The smallest weights that a basis of logical operators of one Pauli type can have.

    The weights do not decrease, and bound every basis of logical operators of this type from below:
    the ``i``-th smallest weight of any basis is at least ``weights[i]``.  In particular, no basis
    has a smaller maximum weight or total weight than ``logical_ops``, which achieves these weights.

    These bounds hold for each Pauli type separately.  The X-type and Z-type operators of a complete
    basis must also pair up, as
    :func:`~qldpc.experimental.logical_bases.evaluation.evaluate_logical_ops` describes, and pairing
    can force some operators to be heavier than either spectrum suggests.

    Attributes:
        weights: The smallest weights, in nondecreasing order.
        logical_ops: A basis of logical operators with these weights, with shape ``(k, n)``.
    """

    weights: tuple[int, ...]
    logical_ops: npt.NDArray[np.uint8]

    @property
    def max_weight(self) -> int:
        """The smallest maximum weight of a basis, or zero if there are no logical qubits."""
        return max(self.weights, default=0)

    @property
    def total_weight(self) -> int:
        """The smallest total weight of a basis."""
        return sum(self.weights)

    @property
    def load_lower_bound(self) -> int:
        """A lower bound on the largest number of operators of any basis that act on one qubit.

        The operators of a basis have total weight at least ``total_weight``, spread over ``n``
        qubits, so some qubit is acted on by at least ``ceil(total_weight / n)`` of them.
        """
        num_qubits = self.logical_ops.shape[1]
        return math.ceil(self.total_weight / num_qubits) if num_qubits else 0


def get_logical_weight_spectrum(
    code: codes.CSSCode, pauli: PauliXZLike, *, logicals: LogicalOpsKind = "dressed"
) -> WeightSpectrum:
    """The smallest weights that a basis of logical operators of one Pauli type can have.

    Logical classes form a matroid, so a greedy search finds these weights exactly: it repeatedly
    adds a lightest logical operator whose class is independent of the classes of those already
    chosen.  Each step is an exact distance search (see :mod:`qldpc.codes.distance`), whose cost
    generally grows exponentially with the weight it finds, and many steps only confirm that the
    previous weight recurs, which is cheaper.  Results are cached on disk with :mod:`qldpc.cache`
    under the name ``"logical_weight_spectrum"``.

    Args:
        code: A binary CSS code, which may be a subsystem code.
        pauli: The Pauli type of the logical operators.
        logicals: For a subsystem code, ``"dressed"`` (the default) uses logical operators that
            commute with the stabilizers, while ``"bare"`` uses those that commute with every gauge
            operator.  The two are the same for a stabilizer code.

    Returns:
        The weight spectrum, with a basis of logical operators that achieves it.

    Raises:
        TypeError: If ``code`` is not a CSS code.
        ValueError: If ``code`` is not binary, or if ``logicals`` is not ``"dressed"`` or
            ``"bare"``.
    """
    sector = get_sector(code, pauli, logicals)
    space = get_independent_rows(sector.get_space())
    trivial = get_independent_rows(sector.trivial)
    weights, logical_ops = _get_greedy_basis(space, trivial)
    return WeightSpectrum(weights, logical_ops)


@cache.use_disk_cache(
    "logical_weight_spectrum",
    key_func=lambda space, trivial: (space.shape[1], space.tobytes(), trivial.tobytes()),
)
def _get_greedy_basis(
    space: npt.NDArray[np.uint8], trivial: npt.NDArray[np.uint8]
) -> tuple[tuple[int, ...], npt.NDArray[np.uint8]]:
    """Lightest operators of span(space) with independent classes modulo span(trivial).

    Both arguments are row-reduced, so that equal spaces give equal cache keys.
    """
    dimension = len(space) - len(trivial)
    chosen: list[npt.NDArray[np.uint8]] = []
    weights: list[int] = []
    for _ in range(dimension):
        excluded = np.vstack([trivial, *chosen])
        # The weights do not decrease, so first check whether the previous weight recurs.
        if weights and _has_light_word(space, excluded, weights[-1]):
            weight = weights[-1]
        else:
            weight = _get_min_weight(space, excluded)
        chosen.append(_get_min_weight_word(space, excluded, weight))
        weights.append(weight)
    logical_ops = np.array(chosen, dtype=np.uint8).reshape(dimension, space.shape[1])
    return tuple(weights), logical_ops


####################################################################################################
# minimum-weight words


def _get_min_weight(space: npt.NDArray[np.uint8], excluded: npt.NDArray[np.uint8]) -> int:
    """The minimum weight of a word of span(space) outside span(excluded), a proper subspace."""
    weight = get_distance_from_generators(space, excluded)
    assert weight is not NO_DISTANCE, "every word lies in span(excluded)"
    return weight


def _has_light_word(
    space: npt.NDArray[np.uint8], excluded: npt.NDArray[np.uint8], max_weight: int
) -> bool:
    """Whether a word of span(space) outside span(excluded) has weight at most ``max_weight``.

    The search stops at the first such word, or once it proves that there is none, so its cost
    depends on ``max_weight`` rather than on the minimum weight.
    """
    weight = get_distance_from_generators(space, excluded, cutoff=max_weight, max_weight=max_weight)
    return weight is not NO_DISTANCE and weight <= max_weight


def _get_min_weight_word(
    space: npt.NDArray[np.uint8], excluded: npt.NDArray[np.uint8], weight: int
) -> npt.NDArray[np.uint8]:
    """A word of span(space) outside span(excluded) with the given minimum weight.

    Qubits are deleted in blocks of decreasing size while some such word of weight ``weight`` is
    supported on the qubits that remain.  When no single qubit can be deleted, every remaining qubit
    lies in every such word, so the remaining qubits are exactly the support of one.  Each probe is
    a search limited to ``weight`` in a smaller code, and so is no harder than finding the weight.
    """
    num_qubits = space.shape[1]
    keep = np.ones(num_qubits, dtype=bool)
    block_size = max(num_qubits // 2, 1)
    while np.count_nonzero(keep) > weight:
        deleted = False
        remaining = np.flatnonzero(keep)
        for start in range(0, len(remaining), block_size):
            trial = keep.copy()
            trial[remaining[start : start + block_size]] = False
            if np.count_nonzero(trial) >= weight and _has_word_within(
                space, excluded, trial, weight
            ):
                keep, deleted = trial, True
        if not deleted:
            assert block_size > 1, "a word of the given weight must exist"
            block_size = max(block_size // 2, 1)
    return keep.astype(np.uint8)


def _has_word_within(
    space: npt.NDArray[np.uint8],
    excluded: npt.NDArray[np.uint8],
    keep: npt.NDArray[np.bool_],
    weight: int,
) -> bool:
    """Whether a word of span(space) outside span(excluded), supported on ``keep``, has ``weight``.

    No such word is lighter than ``weight``, which is the minimum over all supports.
    """
    return _has_light_word(_restrict(space, keep), _restrict(excluded, keep), weight)


####################################################################################################
# linear algebra


def _restrict(rows: npt.NDArray[np.uint8], keep: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint8]:
    """Rows spanning the words of span(rows) that vanish off ``keep``, restricted to ``keep``."""
    combinations = get_kernel(np.ascontiguousarray(rows[:, ~keep].T))
    words = combinations.astype(np.int64) @ rows.astype(np.int64) % 2
    return words[:, keep].astype(np.uint8)
