# SPDX-License-Identifier: Apache-2.0

"""The smallest weights that a basis of logical operators of one Pauli type can have."""

from __future__ import annotations

import dataclasses
import math
from typing import Literal

import numpy as np
import numpy.typing as npt

from qldpc import cache, codes
from qldpc._binary import get_independent_rows, get_kernel
from qldpc.codes.distance import NO_DISTANCE, get_distance_from_generators
from qldpc.objects import PauliXZLike

from ._sectors import get_sector
from .evaluation import LogicalOpsMetrics

####################################################################################################
# lightest bases


def get_lightest_logical_basis(
    code: codes.CSSCode, pauli: PauliXZLike, *, logicals: Literal["dressed", "bare"] = "dressed"
) -> LightestLogicalBasis:
    """A basis of logical operators of one Pauli type whose weights are as small as possible.

    The weights of this basis are the successive minima of the logical operators of this type: the
    ``i``-th weight is the smallest ``w`` for which ``i`` operators of weight at most ``w`` have
    independent logical classes.  Logical classes form a matroid, so a greedy search finds these
    weights exactly, by repeatedly adding a lightest operator whose class is independent of the
    classes of those already chosen.

    Each step after the first checks, with a search limited to the previous weight, whether that
    weight recurs.  If it does not, an exact distance search (see :mod:`qldpc.codes.distance`)
    finds the new weight, at a cost that generally grows exponentially with that weight.  Each step
    then recovers an operator of its weight with many searches limited to that weight.  Among
    operators of the lightest weight, the choice ignores how many qubits each shares with the
    operators already chosen; :attr:`LightestLogicalBasis.metrics` measures these overlaps.
    Results are cached on disk with :mod:`qldpc.cache` under the name
    ``"lightest_logical_basis_v1"``.

    Args:
        code: A binary CSS code, which may be a subsystem code.
        pauli: The Pauli type of the logical operators.
        logicals: For a subsystem code, ``"dressed"`` (the default) uses logical operators that
            commute with the stabilizers, while ``"bare"`` uses those that commute with every gauge
            operator.  The two are the same for a stabilizer code.

    Returns:
        The basis, with its weights.

    Raises:
        TypeError: If ``code`` is not a CSS code.
        ValueError: If ``code`` is not binary, or ``pauli`` or ``logicals`` is invalid.
    """
    sector = get_sector(code, pauli, logicals)
    space = get_independent_rows(sector.get_logical_space())
    trivial = get_independent_rows(sector.trivial)
    return LightestLogicalBasis(_get_greedy_basis(space, trivial))


@dataclasses.dataclass(frozen=True, eq=False)
class LightestLogicalBasis:
    """A basis of logical operators of one Pauli type whose weights are as small as possible.

    Its operators are in nondecreasing order of weight, and their weights bound the weights of every
    basis of logical operators of the same Pauli type and kind (dressed or bare) from below: sorted
    in nondecreasing order, the weights of every such basis are at least ``weights``, entry by
    entry.  In particular, no such basis has a smaller maximum weight or total weight.

    These bounds hold for each Pauli type separately.  The X-type and Z-type operators of a complete
    basis must also pair up, as :func:`~qldpc.experimental.logical_bases.evaluate_logical_ops`
    describes, so a complete basis may not reach the bounds of both types at once.

    Attributes:
        logical_ops: The operators, with shape ``(k, n)``.
    """

    logical_ops: npt.NDArray[np.uint8]

    @property
    def weights(self) -> npt.NDArray[np.int_]:
        """The weight of each operator, in nondecreasing order."""
        return self.logical_ops.astype(np.int_).sum(axis=1)

    @property
    def metrics(self) -> LogicalOpsMetrics:
        """The metrics of the operators: their weights, loads, and overlaps."""
        return LogicalOpsMetrics._from_valid_ops(self.logical_ops)

    @property
    def max_weight(self) -> int:
        """The smallest maximum weight of a basis, or zero if there are no logical qubits."""
        return int(self.weights.max(initial=0))

    @property
    def total_weight(self) -> int:
        """The smallest total weight of a basis."""
        return int(self.weights.sum())

    @property
    def load_lower_bound(self) -> int:
        """A lower bound on the ``max_load`` of every basis of the same type and kind.

        ``max_load`` is the largest number of operators that act on one qubit, as in
        :class:`~qldpc.experimental.logical_bases.LogicalOpsMetrics`.

        The operators of a basis have total weight at least ``total_weight``, spread over ``n``
        qubits, so some qubit is acted on by at least ``ceil(total_weight / n)`` of them.
        """
        num_qubits = self.logical_ops.shape[1]
        return math.ceil(self.total_weight / num_qubits) if num_qubits else 0


@cache.use_disk_cache(
    "lightest_logical_basis_v1",
    key_func=lambda space, trivial: (space.shape[1], space.tobytes(), trivial.tobytes()),
)
def _get_greedy_basis(
    space: npt.NDArray[np.uint8], trivial: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    """Lightest operators of span(space) with independent classes modulo span(trivial).

    The rows of ``trivial`` lie in span(space).  Both arguments are row-reduced, so that equal
    spaces give equal cache keys.
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
    return np.array(chosen, dtype=np.uint8).reshape(dimension, space.shape[1])


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

    Delete blocks of qubits, keeping each deletion that leaves a word of weight ``weight`` on the
    remaining qubits, and halve the block size when no block can be deleted.  While more than
    ``weight`` qubits remain, some qubit lies outside such a word and can be deleted, so the loop
    ends with exactly the support of one such word.  Each probe is a search limited to ``weight`` in
    a smaller code, which is typically no harder than finding the weight.
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


def _restrict(rows: npt.NDArray[np.uint8], keep: npt.NDArray[np.bool_]) -> npt.NDArray[np.uint8]:
    """Rows spanning the words of span(rows) that vanish off ``keep``, restricted to ``keep``."""
    combinations = get_kernel(np.ascontiguousarray(rows[:, ~keep].T))
    words = combinations.astype(np.int64) @ rows.astype(np.int64) % 2
    return words[:, keep].astype(np.uint8)
