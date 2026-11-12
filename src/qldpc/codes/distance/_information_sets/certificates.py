# SPDX-License-Identifier: Apache-2.0

"""Certificates of the basis families of exact searches, and the plans of their steps.

A certificate bounds the weights of the eligible words that a search of a basis family has not
visited, from the layers that its sets have completed.  It tells a search when it can stop, and a
plan of steps says where a search stops for a weight limit, and the work of the steps up to there.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import numpy.typing as npt

from qldpc._compilation import always_compiled

from .. import _tuning

# A bound on the weight of unvisited words when no word can remain unvisited.
NO_WORD_BOUND = np.iinfo(np.int64).max

# Caps that keep work estimates within a native integer.  A binomial above _BINOMIAL_CAP counts as
# the cap, and a sum saturates at _WORK_CAP, so an estimate is exact only if it reaches neither cap.
# Searches that reach a cap are infeasible, so their order, against other basis families or
# exhaustive enumeration, does not matter.
_BINOMIAL_CAP = 1 << 50
_WORK_CAP = 1 << 61


@dataclass(frozen=True)
class Certificate:
    """Lower bound on the Hamming weight of eligible words that a family has not yet visited.

    ``lower_bound`` holds after every search basis of the family has completed each coefficient
    layer through ``completed_weight``.  ``step_lower_bound`` also holds part way through a layer,
    once the first sets in family order have completed it, in whatever order their words were
    visited.  A set that is interrupted within a layer does not advance either bound.  The bounds do
    not bound the weights of witnesses already found, and they do not transfer to another basis or
    CSS sector.  A search that skips supports by completion tables completes a layer without
    visiting them, and then the bounds hold only for the eligible words lighter than its best weight
    (Section 3.9 of ``brouwer_zimmermann.md``).

    Each bound is the coverage bound: the occupancy of the sets (see ``get_step_lower_bounds``)
    divided by ``max_coverage``, rounded up, and then rounded up to a multiple of
    ``weight_divisor``.  With ``pivot_blocks``, each bound is the larger of the coverage
    bound and the bound of ``get_lp_lower_bound``, which can be larger when coverage is uneven.

    Attributes:
        dimension: Number of rows of the searched basis.
        ranks: Rank of the pivot block of each search basis, in family order.
        weight_divisor: A proven divisor of the Hamming weight of every eligible word.  It combines
            any divisor proven by the caller, such as two for the symplectic-to-Hamming map, with
            ambient even parity.  It is in the Hamming units of the searched basis, not Pauli units.
        max_coverage: An upper bound on the number of pivot blocks that contain any one coordinate.
            Overestimates are sound; underestimates are not.
        pivot_blocks: The pivot columns of each search basis, in family order, to bound unvisited
            words also by linear programming, or no blocks to use only the coverage bound.
        lp_max_denominator: The largest denominator of the rational dual weights of the linear
            program (see ``get_lp_lower_bound``), which ``pivot_blocks`` needs to be positive.
    """

    dimension: int
    ranks: tuple[int, ...]
    weight_divisor: int = 1
    max_coverage: int = 1
    pivot_blocks: tuple[tuple[int, ...], ...] = ()
    lp_max_denominator: int = 0

    def lower_bound(self, completed_weight: int) -> int:
        """Bound the Hamming weight of unvisited eligible words after the given complete layers."""
        return self.step_lower_bound(completed_weight, len(self.ranks))

    def step_lower_bound(self, weight: int, num_completed: int) -> int:
        """Bound the Hamming weight of unvisited eligible words part way through layer ``weight``.

        The first ``num_completed`` sets, in family order, have completed layer ``weight``, and the
        others have completed layer ``weight - 1``.
        """
        bound = int(self._step_lower_bounds[weight, num_completed])
        if not self._solves_linear_programs:
            return bound
        step = (weight, num_completed)
        if step not in self._lp_lower_bounds:
            lp_bound = get_lp_lower_bound(
                self.pivot_blocks, self._get_requirements(*step), self.lp_max_denominator
            )
            if lp_bound < NO_WORD_BOUND:
                lp_bound += (-lp_bound) % self.weight_divisor
            self._lp_lower_bounds[step] = lp_bound
        return max(bound, self._lp_lower_bounds[step])

    def certifies(self, weight: int, num_completed: int, upper_bound: int) -> bool:
        """Whether ``step_lower_bound(weight, num_completed)`` is at least ``upper_bound``.

        This solves a linear program only if the coverage bound falls short of the upper bound and
        the linear program could reach it: its optimum is at most the value of the feasible solution
        that sets each pivot column to the largest ``requirement / len(block)`` over the blocks that
        contain it.
        """
        if self._step_lower_bounds[weight, num_completed] >= upper_bound:
            return True
        if not self._solves_linear_programs:
            return False
        # No requirement exceeds its block, since a set that has completed the last layer makes the
        # coverage bound certify any upper bound.
        requirements = self._get_requirements(weight, num_completed)
        largest_fractions: dict[int, Fraction] = {}
        for block, requirement in zip(self.pivot_blocks, requirements):
            for column in block:
                fraction = Fraction(max(requirement, 0), len(block))
                largest_fractions[column] = max(fraction, largest_fractions.get(column, fraction))
        reachable = math.ceil(sum(largest_fractions.values(), Fraction(0)))
        reachable += (-reachable) % self.weight_divisor
        return (
            reachable >= upper_bound and self.step_lower_bound(weight, num_completed) >= upper_bound
        )

    def plan(self, limit: int, layer: int) -> int:
        """Find where a search of layer ``layer`` stops above a limit.

        The search stops at its stopping step: the first step of the layer whose
        ``step_lower_bound`` exceeds ``limit``, or the last step, ``(layer, len(ranks))``, if none
        does.  Returns the number of sets that have completed the layer at that step.  Without
        linear programs, ``find_stopping_step`` finds the step.  A certificate that solves linear
        programs checks the steps in order with ``certifies``, except the last step, which stops the
        search anyway.  So it solves a program only for a step before the stopping step or at it,
        never for the last step, and only when the coverage bound of the step does not exceed the
        limit.
        """
        num_sets = len(self.ranks)
        if not self._solves_linear_programs:
            _, num_completed = find_stopping_step(self._step_lower_bounds, limit, layer, layer)
            return num_completed
        for num_completed in range(1, num_sets):
            if self.certifies(layer, num_completed, limit + 1):
                return num_completed
        return num_sets

    def get_num_sets_to_certify(self, weight: int, upper_bound: int) -> int:
        """The fewest leading sets whose completion of layer ``weight`` certifies a bound.

        Once these sets have completed layer ``weight``, and the others layer ``weight - 1``, every
        unvisited eligible word weighs at least ``upper_bound``.  Returns the number of sets if no
        fewer sets suffice, in which case the bound may need later layers too.  The number is that
        of the stopping step of ``plan`` in the layer for the limit ``upper_bound - 1``.  Without
        linear programs, ``find_stopping_step`` finds that step, so the distance and coset searches
        find their stopping steps by one shared rule.
        """
        return self.plan(upper_bound - 1, weight)

    def work(self, upper_bound: int) -> int:
        """Estimate the candidate/set evaluations needed to certify ``upper_bound``.

        This is the work of ``plan_steps`` for every layer from one, every set, and the limit
        ``upper_bound - 1``, on the table of the coverage bound.  So the estimate solves no linear
        programs.  It is pessimistic: it includes systematic rows (coefficient weight one), and it
        assumes no lighter witness is found before the certificate reaches the upper bound.
        """
        ranks, *_ = self._as_array_arguments()
        binomials = get_capped_binomials(self.dimension)
        _, _, work = plan_steps(
            self._step_lower_bounds, upper_bound - 1, 1, self.dimension, ranks, binomials, False
        )
        return work

    def get_weight_bounds(
        self, weight_factor: int, parity: int | None = None
    ) -> npt.NDArray[np.int64]:
        """Coverage bounds on the weight of unvisited eligible words after each step.

        Every word must weigh at least its Hamming weight divided by ``weight_factor``.  Entry
        ``[w, j]`` is the entry of the table of ``get_step_lower_bounds`` for this certificate,
        divided by ``weight_factor`` and rounded up, except that the entries ``[dimension, j]`` for
        ``j >= 1`` stay ``NO_WORD_BOUND``, since no word remains unvisited.  Given ``parity``, zero
        or one, which the Hamming weight of every eligible word must have, the entries of that table
        are first rounded up to that parity: each becomes the smallest integer of that parity that
        is at least the entry.  The bounds do not decrease in the order of the steps.  Column zero
        of row zero, which precedes every step, stays zero, and column zero of each later row
        repeats the last column of the row before.  Each call returns a new table.
        """
        bounds = self._step_lower_bounds
        if parity is not None:
            # Round each finite entry up to the parity.  The entry before the first step bounds no
            # step, and stays zero so that every nonnegative maximum weight passes it, as a check of
            # a cached stopping step requires (see ``_get_stopping_step`` of a coset search).
            bounds = bounds + (bounds - parity) % 2 * (bounds < NO_WORD_BOUND)
            bounds[0, 0] = 0

        # Round up by negated floor division, which cannot overflow.  Division shrinks the entries
        # that mean that no word remains, so they are reset.
        bounds = -(-bounds // weight_factor)
        bounds[self.dimension, 1:] = NO_WORD_BOUND
        return bounds

    @property
    def _solves_linear_programs(self) -> bool:
        """Whether the certificate also bounds unvisited words by linear programs.

        It does so only with pivot blocks and a maximum coverage above one.  With a coverage of
        one, the blocks are disjoint, the linear program separates by block, and its optimum is the
        coverage bound.
        """
        return bool(self.pivot_blocks) and self.max_coverage != 1

    def _as_array_arguments(self) -> tuple[npt.NDArray[np.int64], int, int]:
        """The ranks, weight divisor, and maximum coverage that the array functions take."""
        return np.array(self.ranks, dtype=np.int64), self.weight_divisor, self.max_coverage

    @functools.cached_property
    def _step_lower_bounds(self) -> npt.NDArray[np.int64]:
        """The table of ``get_step_lower_bounds`` for this certificate."""
        return get_step_lower_bounds(self.dimension, *self._as_array_arguments())

    def _get_requirements(self, weight: int, num_completed: int) -> list[int]:
        """The fewest ones that an unvisited word has in each block after the given step."""
        return [
            (weight if index < num_completed else weight - 1) + 1 - (self.dimension - rank)
            for index, rank in enumerate(self.ranks)
        ]

    @functools.cached_property
    def _lp_lower_bounds(self) -> dict[tuple[int, int], int]:
        """The rounded bounds of ``get_lp_lower_bound`` for the steps computed so far."""
        return {}


def get_lp_lower_bound(
    pivot_blocks: Sequence[Sequence[int]], requirements: Sequence[int], max_denominator: int
) -> int:
    """Bound the weight of binary words with given numbers of ones in blocks, by linear programming.

    Let ``B_i = pivot_blocks[i]`` and ``b_i = requirements[i]``.  Every binary word with at least
    ``b_i`` ones in the columns of ``B_i``, for each ``i``, has weight at least the optimum of the
    linear program

        minimize ``sum_c x_c`` subject to ``sum_{c in B_i} x_c >= b_i`` and ``0 <= x_c <= 1``.

    By weak duality, for every ``y >= 0``, such a word has weight at least the dual value

        ``sum_i b_i y_i - sum_c max(0, sum_{i : c in B_i} y_i - 1)``.

    The solver of ``scipy.optimize.linprog`` finds a nearly optimal ``y``, and the bound is the
    ceiling of its dual value in exact rational arithmetic, so rounding in the solver can only
    weaken the bound, never invalidate it.  The coverage bound is the dual value of ``y_i = 1/C``
    for the maximum coverage ``C``, so the optimum is at least that bound.

    Returns ``NO_WORD_BOUND`` if a requirement exceeds the size of its block, since then no word
    meets the requirements, and zero if no requirement is positive or if the solver fails.
    """
    blocks = [
        list(block) for block, requirement in zip(pivot_blocks, requirements) if requirement > 0
    ]
    positive = [requirement for requirement in requirements if requirement > 0]
    if not blocks:
        return 0
    if any(requirement > len(block) for block, requirement in zip(blocks, positive)):
        return NO_WORD_BOUND

    # Import the solver only here, since importing qLDPC does not otherwise load it.
    import scipy.optimize

    # Number the columns of the blocks, and solve the linear program in floating point.
    columns = sorted({column for block in blocks for column in block})
    index = {column: position for position, column in enumerate(columns)}
    incidence = np.zeros((len(blocks), len(columns)))
    for row, block in enumerate(blocks):
        incidence[row, [index[column] for column in block]] = 1
    result = scipy.optimize.linprog(
        np.ones(len(columns)),
        A_ub=-incidence,
        b_ub=-np.array(positive, dtype=float),
        bounds=(0, 1),
        method="highs",
    )
    if result.status != 0:
        return 0

    # Evaluate the dual value exactly.  The solver gives the dual of each constraint
    # ``-sum x >= -b`` as a nonpositive marginal.
    weights = [
        max(
            Fraction(0),
            Fraction(-float(marginal)).limit_denominator(max_denominator),
        )
        for marginal in result.ineqlin.marginals
    ]
    value = sum(
        (requirement * weight for requirement, weight in zip(positive, weights)), Fraction(0)
    )
    for position in range(len(columns)):
        load = sum(
            (weight for row, weight in enumerate(weights) if incidence[row, position]), Fraction(0)
        )
        value -= max(Fraction(0), load - 1)
    return max(0, math.ceil(value))


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

    Entry ``[weight, num_completed]`` holds once the first ``num_completed`` sets, in family order,
    have completed layer ``weight``, and the others have completed layer ``weight - 1``, in whatever
    order their words were visited.  So column ``len(ranks)`` holds once every set has completed
    layer ``weight``, and column zero of row ``weight`` repeats column ``len(ranks)`` of row
    ``weight - 1``.  Rank deficiency limits how many selected rows can have zero pivot support: a
    set with deficiency ``s`` that has completed layer ``c`` adds ``c + 1 - s`` to the occupancy if
    positive, so it adds one more for each layer that it completes once its deficiency no longer
    keeps its contribution at zero.  Entries ``[dimension, j]`` for ``j >= 1`` hold
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

    The work estimates of the distance search and of the routes of the coset searches add up the
    work of the steps of a search with this function, and searches that need only where they stop
    call ``find_stopping_step``.  The table ``step_bounds`` has a row for each layer from zero to
    the dimension, and a column for each set after column zero.  The steps of layer ``w`` are
    ``(w, 1)`` to ``(w, num_sets)``, in that order and layer by layer, and the bound of step
    ``(w, j)`` is ``step_bounds[w, j]``, as in the table of ``get_step_lower_bounds``.  These bounds
    must not decrease in the order of the steps, which holds for that table and for any
    nondecreasing function of it, such as the tables of ``Certificate.get_weight_bounds``.

    The stopping step ``(w*, j*)`` is the first step of the layers whose bound exceeds ``limit``,
    or ``(last_layer, num_sets)`` if there is none, which ``find_stopping_step`` finds.  At that
    step, set ``i`` has completed layer ``w_i = w*`` if ``i < j*``, and ``w* - 1`` otherwise.  A set
    of rank deficiency ``s`` adds to the bound of the step only if ``w_i >= s`` (see
    ``get_step_lower_bounds``).  The other sets are the unused sets of the step.  If ``skip_unused``
    is True, the plan leaves them out, as a coset search skips them, and reads the ranks of the sets
    from ``ranks``.  Otherwise, it reads no ranks.

    The work adds ``binomials[w]`` candidate/set evaluations for each set that the plan keeps and
    each layer ``w`` from ``first_layer`` to ``w_i``.  Empty ``binomials`` give zero work.  The work
    saturates at ``_WORK_CAP``: it is the cap once a layer up to ``w*`` has a binomial above
    ``_WORK_CAP // num_sets``, or once the sum reaches the cap after a layer.  With the binomials of
    ``get_capped_binomials``, a plan from layer one for the limit ``U - 1`` gives the work estimate
    of a distance search (Section 3.6 of ``brouwer_zimmermann.md``).  A plan from layer zero for a
    maximum weight ``max_weight`` that skips unused sets gives the number of words that a coset
    search weighs, or skips by completion tables (Section 3.9 of ``brouwer_zimmermann.md``), after
    the word of its syndrome, if no word of the coset weighs at most ``max_weight``.

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
    """Find the stopping step of a search of layers ``first_layer`` to ``last_layer``.

    The steps and their bounds are those of ``plan_steps``, and so is the stopping step: the first
    step of the layers whose bound exceeds ``limit``, or ``(last_layer, num_sets)`` if there is
    none.  The bounds do not decrease in the order of the steps, so a binary search finds it.  The
    coset searches find where they stop with this function, and so does the distance search,
    through ``Certificate.plan``, when its certificate solves no linear programs.
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

    ``pivots`` holds the pivot columns of each set of ``num_rows = pivots.shape[1]`` rows, padded
    with -1 beyond its rank.  A set is unused if its rank deficiency exceeds the last layer that it
    has completed, which must be at least zero.  This holds exactly if its rank is at most
    ``position = num_rows - completed_layer - 1``, that is, if its pivot at ``position`` is padding.
    ``position`` is negative only if the set has completed layer ``num_rows``, and then it is used.
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
    """Estimate the candidate/set evaluations needed to certify ``upper_bound``.

    This is ``Certificate.work`` for arrays, which compiled code can pass.  It is the work of
    ``plan_steps`` for every layer from one, every set, and the limit ``upper_bound - 1``.
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
        if binomials[weight] > _WORK_CAP // max(num_sets, 1):
            return _WORK_CAP
        for index in range(num_sets):
            # Skip the layers that the set has not completed, and the sets that the plan leaves out.
            completed_layer = get_completed_layer(index, layer, num_completed)
            if weight > completed_layer or (
                skip_unused and completed_layer < dimension - ranks[index]
            ):
                continue
            work += binomials[weight]
        if work >= _WORK_CAP:
            return _WORK_CAP
    return work
