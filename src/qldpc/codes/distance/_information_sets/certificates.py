# SPDX-License-Identifier: Apache-2.0

"""Certificates of the basis families of exact searches, and the plans of their steps.

A certificate bounds the weights of the eligible words that a search of a basis family has not
visited, and a plan says where a search stops above a limit, and the work of its steps up to there.

Compiled functions here, such as ``find_stopping_step`` and ``is_unused_set``, also run within nogil
calls of threaded searches, so they must keep the rules for such code in the docstring of
``qldpc/_compilation.py``.  The plans and certificates, through ``Certificate.certifies`` and
``Certificate.get_num_sets_to_certify``, feed the decisions of a distance search, which must keep
the requirement "Decide only from what a layer leaves the same on any number of threads" of
``brouwer_zimmermann_code.md#sec-requirements``.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qldpc._compilation import always_compiled

from .. import _tuning

# A bound on the weight of unvisited words when no word can remain unvisited.
NO_WORD_BOUND = np.iinfo(np.int64).max

# Caps that keep work estimates within a native integer: a binomial above _BINOMIAL_CAP counts as
# the cap, and a sum saturates at WORK_CAP.  A search whose estimate reaches a cap is infeasible, so
# its order against other basis families does not matter.
_BINOMIAL_CAP = 1 << 50
WORK_CAP = 1 << 61


@dataclass(frozen=True)
class Certificate:
    """Lower bounds on the Hamming weight of the eligible words that a family has not visited.

    ``lower_bound`` holds once every set of the family has completed each layer through
    ``completed_weight``.  ``step_lower_bound`` holds part way through a layer, once the first sets
    in family order have completed it (``brouwer_zimmermann.md#sec-stopping-certificate`` and
    ``brouwer_zimmermann.md#sec-overlapping-certificate``).  A set interrupted within a layer
    advances neither.  The bounds hold only for the eligible words of the family's sector that the
    family has not visited (``brouwer_zimmermann.md#sec-shared-witnesses``).  In a search that skips
    completions by tables, they hold only for those lighter than its best weight
    (``brouwer_zimmermann.md#sec-completion-contracts``).  A certificate of the images of an
    information set under automorphisms, as of an automorphism family, holds only for unvisited
    eligible words lighter than a best weight that every visited eligible word lowers, as in a
    distance search.  Counts and coset searches must not use one
    (``brouwer_zimmermann.md#sec-automorphism-certificate``).

    Each bound is the occupancy of the sets (see ``get_step_lower_bounds``) divided by
    ``max_coverage``, rounded up, and then rounded up to a multiple of ``weight_divisor``.

    Attributes:
        dimension: Number of rows of the searched basis.
        ranks: Rank of the pivot block of each search basis, in family order.
        weight_divisor: A proven divisor of the Hamming weight of every eligible word, such as two
            for general Pauli problems or an even basis
            (``brouwer_zimmermann.md#sec-weight-multiples``).
        max_coverage: An upper bound on the number of pivot blocks that contain any one column.
    """

    dimension: int
    ranks: tuple[int, ...]
    weight_divisor: int = 1
    max_coverage: int = 1

    def lower_bound(self, completed_weight: int) -> int:
        """Bound the Hamming weight of unvisited eligible words after the given complete layers."""
        return self.step_lower_bound(completed_weight, len(self.ranks))

    def step_lower_bound(self, weight: int, num_completed: int) -> int:
        """Bound the Hamming weight of unvisited eligible words part way through layer ``weight``.

        The first ``num_completed`` sets, in family order, have completed layer ``weight``, and the
        others have completed layer ``weight - 1``.
        """
        return int(self._step_lower_bounds[weight, num_completed])

    def certifies(self, weight: int, num_completed: int, upper_bound: int) -> bool:
        """Whether ``step_lower_bound(weight, num_completed)`` is at least ``upper_bound``."""
        return bool(self._step_lower_bounds[weight, num_completed] >= upper_bound)

    def get_num_sets_to_certify(self, weight: int, upper_bound: int) -> int:
        """The fewest leading sets whose completion of layer ``weight`` certifies ``upper_bound``.

        These are the sets completed at the first step of the layer whose ``step_lower_bound``
        reaches ``upper_bound``, or every set if no fewer suffice, in which case the bound may need
        later layers too (``brouwer_zimmermann.md#sec-stopping-certificate``).
        """
        _, num_completed = find_stopping_step(
            self._step_lower_bounds, upper_bound - 1, weight, weight
        )
        return num_completed

    def work(self, upper_bound: int, first_layer: int = 1) -> int:
        """Estimate the candidate/set evaluations that certify ``upper_bound``.

        The estimate is that of ``brouwer_zimmermann.md#sec-work-estimates``.  From a later
        ``first_layer``, it is the work left once every set has completed the layers before
        ``first_layer``.
        """
        ranks, *_ = self._as_array_arguments()
        binomials = get_capped_binomials(self.dimension)
        _, _, work = plan_steps(
            self._step_lower_bounds,
            upper_bound - 1,
            first_layer,
            self.dimension,
            ranks,
            binomials,
            False,
        )
        return work

    def get_weight_bounds(
        self, weight_factor: int, parity: int | None = None
    ) -> npt.NDArray[np.int64]:
        """Coverage bounds on the weight of the unvisited eligible words after each step.

        ``parity``, if given, is the parity of the Hamming weight of every eligible word, and every
        word weighs at least its Hamming weight divided by ``weight_factor``
        (``brouwer_zimmermann.md#sec-weight-multiples`` and
        ``brouwer_zimmermann_code.md#sec-coset-searches``).  So each entry of the table of
        ``get_step_lower_bounds`` is rounded up to ``parity``, if any, and then divided by
        ``weight_factor``, rounding up.  The entries do not decrease in the order of the steps,
        those that mean that no word remains stay ``NO_WORD_BOUND``, and entry ``[0, 0]``, before
        every step, stays zero.  Each call returns a new table.
        """
        bounds = self._step_lower_bounds
        if parity is not None:
            # Round each finite entry up to the parity, but keep the zero before the first step,
            # which every nonnegative maximum weight must pass in a check of a cached stopping step
            # (``get_stopping_step`` of ``_brouwer_zimmermann/coset_tables.py``).
            bounds = bounds + (bounds - parity) % 2 * (bounds < NO_WORD_BOUND)
            bounds[0, 0] = 0

        # Round up by negated floor division, which cannot overflow.  Division shrinks the entries
        # that mean that no word remains, so they are reset.
        bounds = -(-bounds // weight_factor)
        bounds[self.dimension, 1:] = NO_WORD_BOUND
        return bounds

    def _as_array_arguments(self) -> tuple[npt.NDArray[np.int64], int, int]:
        """The ranks, weight divisor, and maximum coverage that the array functions take."""
        return np.array(self.ranks, dtype=np.int64), self.weight_divisor, self.max_coverage

    @functools.cached_property
    def _step_lower_bounds(self) -> npt.NDArray[np.int64]:
        """The table of ``get_step_lower_bounds`` for this certificate."""
        return get_step_lower_bounds(self.dimension, *self._as_array_arguments())


@functools.lru_cache(maxsize=_tuning.BINOMIAL_CACHE_SIZE)
def get_capped_binomials(dimension: int) -> npt.NDArray[np.int64]:
    """The binomial coefficients ``C(dimension, weight)`` for every weight, capped to fit.

    The array is shared between calls, so it is read-only.
    """
    binomials = np.array(
        [min(math.comb(dimension, weight), _BINOMIAL_CAP) for weight in range(dimension + 1)],
        dtype=np.int64,
    )
    binomials.flags.writeable = False
    return binomials


@always_compiled
def lower_bound_from_occupancy(occupancy: int, weight_divisor: int, max_coverage: int) -> int:
    """Turn the pivot occupancy of a certificate into a bound on the weight of unvisited words."""
    # Correct for repeated Hamming coordinates by dividing occupancy and rounding up.
    bound = (occupancy + max_coverage - 1) // max_coverage
    if weight_divisor > 1:
        # Every eligible Hamming weight is a multiple of weight_divisor, so the bound rounds up.
        bound += (-bound) % weight_divisor
    return bound


@always_compiled
def get_step_lower_bounds(
    dimension: int,
    ranks: npt.NDArray[np.int64],
    weight_divisor: int,
    max_coverage: int,
) -> npt.NDArray[np.int64]:
    """Certificate lower bounds after each step of a search, which completes one layer of one set.

    Entry ``[w, j]`` holds once the first ``j`` sets, in family order, have completed layer ``w``
    and the others layer ``w - 1``, so column zero of row ``w`` repeats the last column of row
    ``w - 1``.  A set of rank deficiency ``s`` that has completed layer ``c`` adds
    ``max(0, c + 1 - s)`` to the occupancy from which each bound comes
    (``brouwer_zimmermann.md#sec-search-bases``).  Entries ``[dimension, j]`` for ``j >= 1`` hold
    ``NO_WORD_BOUND``, since a set that has completed layer ``dimension`` has visited every word.
    """
    num_sets = len(ranks)
    bounds = np.empty((dimension + 1, num_sets + 1), dtype=np.int64)
    for weight in range(dimension + 1):
        # The occupancy once every set has completed layer ``weight - 1``.
        occupancy = 0
        for rank in ranks:
            occupancy += max(0, weight - (dimension - rank))
        bounds[weight, 0] = lower_bound_from_occupancy(occupancy, weight_divisor, max_coverage)

        # Completing layer ``weight`` adds one for each set whose deficiency it overcomes.
        for index in range(num_sets):
            if weight + 1 > dimension - ranks[index]:
                occupancy += 1
            bounds[weight, index + 1] = lower_bound_from_occupancy(
                occupancy, weight_divisor, max_coverage
            )
    bounds[dimension, 1:] = NO_WORD_BOUND
    return bounds


@always_compiled
def plan_steps(
    step_bounds: npt.NDArray[np.int64],
    limit: int,
    first_layer: int,
    last_layer: int,
    ranks: npt.NDArray[np.int64],
    binomials: npt.NDArray[np.int64],
    skip_unused: bool,
) -> tuple[int, int, int]:
    """Plan a search of layers ``first_layer`` to ``last_layer`` that stops above a limit.

    ``step_bounds`` has the layout of ``get_step_lower_bounds`` for a family of ``num_sets`` sets:
    step ``(w, j)`` completes layer ``w`` of the ``j``-th set, and its bound is entry ``[w, j]``.
    The bounds must not decrease in step order: layer by layer, and set by set within a layer.

    The stopping step ``(w*, j*)`` is the first step whose bound exceeds ``limit``, or
    ``(last_layer, num_sets)`` if none does.  There, the first ``j*`` sets have completed layer
    ``w*``, and the others ``w* - 1``.  Call the last layer that set ``i`` has completed ``w_i``.

    The work adds ``binomials[w]`` candidate/set evaluations for each set ``i`` that the plan keeps
    (see ``skip_unused`` below) and each layer ``w`` from ``first_layer`` to ``w_i``.  It is zero
    for empty ``binomials``.  An unused set is one whose rank deficiency, read from ``ranks``,
    exceeds ``w_i``, so it adds nothing to the bound.  With ``skip_unused``, the plan leaves out the
    unused sets, as a coset search does.  Without ``skip_unused``, the plan reads no ranks.  The
    work saturates at ``WORK_CAP`` once a layer through ``w*`` has a binomial above
    ``WORK_CAP // num_sets``, or the sum reaches the cap.  From layer one, for a limit one below the
    best weight, the work is the estimate of a distance search.  From layer zero, for a maximum
    weight, and skipping unused sets, it is that of a coset query
    (``brouwer_zimmermann.md#sec-work-estimates``).

    Returns:
        The stopping step ``(w*, j*)``, and the work.
    """
    layer, num_completed = find_stopping_step(step_bounds, limit, first_layer, last_layer)
    work = _get_steps_work(
        step_bounds.shape[0] - 1,
        step_bounds.shape[1] - 1,
        ranks,
        layer,
        num_completed,
        first_layer,
        binomials,
        skip_unused,
    )
    return layer, num_completed, work


@always_compiled
def find_stopping_step(
    step_bounds: npt.NDArray[np.int64], limit: int, first_layer: int, last_layer: int
) -> tuple[int, int]:
    """Find the stopping step of ``plan_steps`` by a binary search, since the bounds never decrease.

    The steps are those of ``brouwer_zimmermann.md#sec-stopping-certificate``.

    Returns:
        The stopping step ``(w*, j*)``.
    """
    num_sets = step_bounds.shape[1] - 1
    low, high = 0, (last_layer - first_layer + 1) * num_sets
    while low < high:
        middle = (low + high) // 2
        if step_bounds[first_layer + middle // num_sets, 1 + middle % num_sets] > limit:
            high = middle
        else:
            low = middle + 1
    if low == (last_layer - first_layer + 1) * num_sets:
        return last_layer, num_sets
    return first_layer + low // num_sets, 1 + low % num_sets


@always_compiled(inline=True)
def get_completed_layer(set_index: int, layer: int, num_completed: int) -> int:
    """The last layer that set ``set_index`` has completed at step ``(layer, num_completed)``."""
    return layer if set_index < num_completed else layer - 1


@always_compiled(inline=True)
def is_unused_set(
    pivots: npt.NDArray[np.int64], set_index: int, layer: int, num_completed: int
) -> bool:
    """Whether a set is an unused set of step ``(layer, num_completed)`` (see ``plan_steps``).

    ``pivots`` holds the pivot columns of each set, padded with -1 beyond its rank, in rows of
    ``r = pivots.shape[1]`` entries.  A set that has completed layer ``c`` is unused if its rank
    deficiency exceeds ``c``, that is, if its pivot at index ``r - c - 1`` is padding.  A set that
    has completed layer ``r`` is used.  The completed layer ``c`` must be at least zero.
    """
    position = pivots.shape[1] - get_completed_layer(set_index, layer, num_completed) - 1
    return position >= 0 and pivots[set_index, position] < 0


@always_compiled
def estimate_certificate_work(
    dimension: int,
    ranks: npt.NDArray[np.int64],
    weight_divisor: int,
    max_coverage: int,
    upper_bound: int,
    binomials: npt.NDArray[np.int64],
) -> int:
    """``Certificate.work`` from layer one, for arrays that compiled code can pass.

    ``binomials`` are ``get_capped_binomials(dimension)``.
    """
    bounds = get_step_lower_bounds(dimension, ranks, weight_divisor, max_coverage)
    _, _, work = plan_steps(bounds, upper_bound - 1, 1, dimension, ranks, binomials, False)
    return work


@always_compiled
def _get_steps_work(
    dimension: int,
    num_sets: int,
    ranks: npt.NDArray[np.int64],
    layer: int,
    num_completed: int,
    first_layer: int,
    binomials: npt.NDArray[np.int64],
    skip_unused: bool,
) -> int:
    """The work of ``plan_steps`` for its stopping step ``(layer, num_completed)``."""
    if len(binomials) == 0:
        return 0
    work = 0
    for weight in range(first_layer, layer + 1):
        # Saturate before adding, so that no sum can overflow: a layer adds at most the cap.
        if binomials[weight] > WORK_CAP // max(num_sets, 1):
            return WORK_CAP
        for index in range(num_sets):
            # Skip the layers that the set has not completed, and the sets that the plan leaves out.
            completed_layer = get_completed_layer(index, layer, num_completed)
            if weight > completed_layer or (
                skip_unused and completed_layer < dimension - ranks[index]
            ):
                continue
            work += binomials[weight]
        if work >= WORK_CAP:
            return WORK_CAP
    return work
