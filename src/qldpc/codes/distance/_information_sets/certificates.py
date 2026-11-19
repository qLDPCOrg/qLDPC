# SPDX-License-Identifier: Apache-2.0

"""Certificates of the basis families of exact searches, and the plans of their steps.

A certificate bounds the weights of the eligible words that a search of a basis family has not
visited, and a plan says where a search stops above a limit, and the work of its steps up to there.
Sections cited in this module are those of ``brouwer_zimmermann.md`` and
``brouwer_zimmermann_code.md``.

Compiled functions here, such as ``find_stopping_step`` and ``is_unused_set``, also run within nogil
calls of threaded searches, so they must keep the rules for such code in the docstring of
``qldpc/_compilation.py``.  The plans and certificates, through ``Certificate.certifies`` and
``Certificate.get_num_sets_to_certify``, feed the decisions of a distance search, which must keep
the requirement of Section 5, "Decide only from what a layer leaves the same on any number of
threads".
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

# Caps that keep work estimates within a native integer: a binomial above _BINOMIAL_CAP counts as
# the cap, and a sum saturates at WORK_CAP.  A search whose estimate reaches a cap is infeasible, so
# its order against other basis families does not matter.
_BINOMIAL_CAP = 1 << 50
WORK_CAP = 1 << 61


@dataclass(frozen=True)
class Certificate:
    """Lower bounds on the Hamming weight of the eligible words that a family has not visited.

    ``lower_bound`` holds once every set of the family has completed each layer through
    ``completed_weight``, and ``step_lower_bound`` part way through a layer, once the first sets in
    family order have completed it (Sections 2.5 and 3.1).  A set interrupted within a layer
    advances neither.  The bounds hold only for the eligible words of the family's sector that the
    family has not visited, and, in a search that skips completions by tables, only for those
    lighter than its best weight (Sections 3.3 and 3.9).  A certificate of the images of blocks
    under automorphisms, as of an automorphism family or with ``column_orbits``, holds only for
    unvisited eligible words lighter than a best weight that every visited eligible word lowers,
    as in a distance search; counts and coset searches must not use one (Section 3.7).

    Each bound is the occupancy of the sets (see ``get_step_lower_bounds``) divided by
    ``max_coverage``, rounded up, and then rounded up to a multiple of ``weight_divisor``; with
    ``pivot_blocks``, it is the larger of that and the bound of ``get_lp_lower_bound``, which with
    ``column_orbits`` covers the images of the blocks under a group of automorphisms too.

    Attributes:
        dimension: Number of rows of the searched basis.
        ranks: Rank of the pivot block of each search basis, in family order.
        weight_divisor: A proven divisor of the Hamming weight of every eligible word, such as two
            for general Pauli problems or an even basis (Section 3.2).
        max_coverage: An upper bound on the number of pivot blocks that contain any one column,
            which may overestimate but never underestimate.
        pivot_blocks: The pivot columns of each search basis, in family order, for the bounds of
            linear programming, or none.
        lp_max_denominator: The largest denominator of the dual weights of the linear programs,
            which must be positive with ``pivot_blocks``.
        column_orbits: For each column, the index of its orbit under a verified group of
            automorphisms of the searched problem, or none.  These indices take effect only with
            ``pivot_blocks``, and the linear program is then the program over orbits of Section 3.7,
            which bounds by the images of the blocks under the group too.
    """

    dimension: int
    ranks: tuple[int, ...]
    weight_divisor: int = 1
    max_coverage: int = 1
    pivot_blocks: tuple[tuple[int, ...], ...] = ()
    lp_max_denominator: int = 0
    column_orbits: tuple[int, ...] = ()

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
                self.pivot_blocks,
                self._get_requirements(*step),
                self.lp_max_denominator,
                self.column_orbits,
            )
            if lp_bound < NO_WORD_BOUND:
                lp_bound += (-lp_bound) % self.weight_divisor
            self._lp_lower_bounds[step] = lp_bound
        return max(bound, self._lp_lower_bounds[step])

    def certifies(self, weight: int, num_completed: int, upper_bound: int) -> bool:
        """Whether ``step_lower_bound(weight, num_completed)`` is at least ``upper_bound``.

        This solves a linear program only if the coverage bound falls short and, without
        ``column_orbits``, the feasible point that sets each pivot column to the largest positive
        ``requirement / len(block)`` of its blocks, or zero, does not show the optimum to fall short
        too (Section 3.1).
        """
        if self._step_lower_bounds[weight, num_completed] >= upper_bound:
            return True
        if not self._solves_linear_programs:
            return False
        if self.column_orbits:
            # The feasible point below is feasible only for the program of Section 3.1, so it cannot
            # show that the program over orbits falls short.
            return self.step_lower_bound(weight, num_completed) >= upper_bound
        # No requirement exceeds its block, since a set that has completed the last layer makes the
        # coverage bound certify any upper bound.  So only a nonempty block has a positive
        # requirement.  Let the share of a block be requirement / len(block) if the requirement is
        # positive, and zero if not.  The feasible point sets each pivot column to the largest share
        # of its blocks, which is the share of the first block that holds the column when the
        # blocks are sorted by decreasing share.  So its value is the sum of each share times the
        # number of columns that its block holds first.
        requirements = self._get_requirements(weight, num_completed)
        shares = [
            Fraction(requirement, len(block)) if requirement > 0 else Fraction(0)
            for block, requirement in zip(self.pivot_blocks, requirements)
        ]
        order = sorted(range(len(shares)), key=shares.__getitem__, reverse=True)
        columns = np.concatenate([self._pivot_block_arrays[index] for index in order])
        owners = np.repeat(order, [len(self.pivot_blocks[index]) for index in order])
        _, first_entries = np.unique(columns, return_index=True)
        counts = np.bincount(owners[first_entries], minlength=len(shares))
        value = sum((share * int(count) for share, count in zip(shares, counts)), Fraction(0))
        reachable = math.ceil(value)
        reachable += (-reachable) % self.weight_divisor
        return (
            reachable >= upper_bound and self.step_lower_bound(weight, num_completed) >= upper_bound
        )

    def plan(self, limit: int, layer: int) -> int:
        """Find the stopping step of layer ``layer`` for ``limit``, and the sets completed there.

        The stopping step is the first step of the layer whose ``step_lower_bound`` exceeds
        ``limit``, or the last step if none does.  Returns the number of sets that have completed
        the layer at that step.  With linear programs, this checks every step but the last, in
        order, with ``certifies`` (Section 4.1).
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
        """The fewest leading sets whose completion of layer ``weight`` certifies ``upper_bound``.

        This is ``plan(upper_bound - 1, weight)``: every set if no fewer suffice, in which case the
        bound may need later layers too.
        """
        return self.plan(upper_bound - 1, weight)

    def work(self, upper_bound: int, first_layer: int = 1) -> int:
        """Estimate the candidate/set evaluations that certify ``upper_bound`` (Section 3.6).

        The estimate uses the coverage bound alone.  From a later ``first_layer``, it is the work
        left once every set has completed the layers before ``first_layer``.
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
        word weighs at least its Hamming weight divided by ``weight_factor`` (Sections 3.2 and 4.1).
        So each entry of the table of ``get_step_lower_bounds`` is rounded up to ``parity``, if
        any, and then divided by ``weight_factor``, rounding up.  The entries do not decrease in
        the order of the steps, those that mean that no word remains stay ``NO_WORD_BOUND``, and
        entry ``[0, 0]``, before every step, stays zero.  Each call returns a new table.
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

    @property
    def _solves_linear_programs(self) -> bool:
        """Whether the certificate also solves linear programs.

        It does exactly with pivot blocks, and either a maximum coverage above one or column
        orbits.  With a coverage of one and no orbits, the blocks are disjoint, and the optimum is
        the coverage bound (Section 3.1).
        """
        return bool(self.pivot_blocks) and (self.max_coverage != 1 or bool(self.column_orbits))

    def _as_array_arguments(self) -> tuple[npt.NDArray[np.int64], int, int]:
        """The ranks, weight divisor, and maximum coverage that the array functions take."""
        return np.array(self.ranks, dtype=np.int64), self.weight_divisor, self.max_coverage

    @functools.cached_property
    def _step_lower_bounds(self) -> npt.NDArray[np.int64]:
        """The table of ``get_step_lower_bounds`` for this certificate."""
        return get_step_lower_bounds(self.dimension, *self._as_array_arguments())

    @functools.cached_property
    def _pivot_block_arrays(self) -> tuple[npt.NDArray[np.int64], ...]:
        """The pivot blocks as arrays, for ``certifies``."""
        return tuple(np.array(block, dtype=np.int64) for block in self.pivot_blocks)

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
    pivot_blocks: Sequence[Sequence[int]],
    requirements: Sequence[int],
    max_denominator: int,
    column_orbits: Sequence[int] = (),
) -> int:
    """Bound the weight of binary words with at least ``b_i`` ones in each block ``B_i``.

    ``B_i`` is ``pivot_blocks[i]``, and ``b_i`` is ``requirements[i]``.  With ``column_orbits``, the
    index of the orbit of each column under a group of column permutations, the words also have at
    least ``b_i`` ones in every image of ``B_i`` under the group.  The bound is the ceiling of the
    dual value of the linear program of Section 3.1, or of the program over orbits of Section 3.7,
    at the solver's dual weights, limited to denominators of at most ``max_denominator``, and
    evaluated exactly as Section 3.1 describes.  Returns ``NO_WORD_BOUND`` if a requirement exceeds
    its block, since then no word meets it, and zero if no requirement is positive or the solver
    fails.
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

    # Group the columns into classes, with one variable each: the orbits, for the program of
    # Section 3.7, or else the sets of columns that lie in the same blocks, which share a value in
    # an optimal solution of the program of Section 3.1.  Each block counts its columns in each
    # class.
    if column_orbits:
        classes = np.asarray(column_orbits, dtype=np.int64)
        sizes = np.bincount(classes)
    else:
        columns = np.array(sorted({column for block in blocks for column in block}))
        membership = np.zeros((columns[-1] + 1, len(blocks)), dtype=bool)
        for row, block in enumerate(blocks):
            membership[block, row] = True
        _, inverse, sizes = np.unique(
            membership[columns], axis=0, return_inverse=True, return_counts=True
        )
        classes = np.zeros(len(membership), dtype=np.int64)
        classes[columns] = inverse.reshape(-1)
    incidence = np.zeros((len(blocks), len(sizes)), dtype=np.int64)
    for row, block in enumerate(blocks):
        np.add.at(incidence[row], classes[block], 1)
    used = np.flatnonzero(incidence.any(axis=0))
    incidence, sizes = incidence[:, used], sizes[used]

    # Solve the linear program over the classes in floating point.
    result = scipy.optimize.linprog(
        sizes.astype(float),
        A_ub=-incidence.astype(float),
        b_ub=-np.array(positive, dtype=float),
        bounds=(0, 1),
        method="highs",
    )
    if result.status != 0:
        return 0

    # Evaluate the dual value exactly, as an integer numerator over the common denominator of the
    # weights.  The solver gives the dual of each constraint ``-sum x >= -b`` as a nonpositive
    # marginal.  Each class subtracts its dual term, ``max(0, sum_i count_i y_i - size)``, where
    # ``count_i`` is the number of its columns in block ``i`` (Sections 3.1 and 3.7).
    weights = [
        max(
            Fraction(0),
            Fraction(-float(marginal)).limit_denominator(max_denominator),
        )
        for marginal in result.ineqlin.marginals
    ]
    denominator = math.lcm(*(weight.denominator for weight in weights))
    numerators = [weight.numerator * (denominator // weight.denominator) for weight in weights]
    value = sum(
        int(requirement) * numerator for requirement, numerator in zip(positive, numerators)
    )
    for counts, size in zip(incidence.T, sizes):
        load = sum(int(count) * numerator for count, numerator in zip(counts, numerators))
        value -= max(0, load - int(size) * denominator)
    return max(0, -(-value // denominator))


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
    ``max(0, c + 1 - s)`` to the occupancy from which each bound comes (Section 2.4).  Entries
    ``[dimension, j]`` for ``j >= 1`` hold ``NO_WORD_BOUND``, since a set that has completed layer
    ``dimension`` has visited every word.
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

    ``step_bounds`` has a row for each layer from zero to the dimension, and column zero and then a
    column for each of ``num_sets`` sets.  Step ``(w, j)`` completes layer ``w`` of the ``j``-th
    set, and its bound is ``step_bounds[w, j]``, as in the table of ``get_step_lower_bounds``.  The
    steps run layer by layer, and their bounds must not decrease in that order, as holds for that
    table and the tables of ``Certificate.get_weight_bounds``.  The stopping step ``(w*, j*)`` is
    the first step whose bound exceeds ``limit``, or ``(last_layer, num_sets)`` if none does.
    There, the first ``j*`` sets have completed layer ``w*``, and the others ``w* - 1``; call the
    last layer that set ``i`` has completed ``w_i``.

    The work adds ``binomials[w]`` candidate/set evaluations for each set ``i`` that the plan keeps
    and each layer ``w`` from ``first_layer`` to ``w_i``, and is zero for empty ``binomials``.  With
    ``skip_unused``, the plan leaves out, as a coset search does, the unused sets, which add nothing
    to the bound since their rank deficiency, read from ``ranks``, exceeds ``w_i``.  Without it, the
    plan reads no ranks.  The work saturates at ``WORK_CAP`` once a layer through ``w*`` has a
    binomial above ``WORK_CAP // num_sets``, or the sum reaches the cap.  From layer one for a limit
    one below the best weight, the work is the estimate of a distance search, and from layer zero
    for a maximum weight, skipping unused sets, that of a coset query (Section 3.6).

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

    The coset searches and ``Certificate.plan`` use it (Section 4.1).
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

    ``pivots`` holds the pivot columns of each set of ``num_rows`` rows, padded with -1 beyond its
    rank.  A set that has completed layer ``c``, which must be at least zero, is unused if its rank
    deficiency exceeds ``c``, that is, if its pivot at index ``num_rows - c - 1`` is padding.  A set
    that has completed layer ``num_rows`` is used.
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
