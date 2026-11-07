# SPDX-License-Identifier: Apache-2.0

"""Exact binary distances by a depth-first search of the supports connected in a Tanner graph.

The search finds a nonzero vector ``x`` with ``H x = 0`` outside the row space of ``G``, for sparse
parity checks ``H``, by growing a support only through unsatisfied checks, and pruning with bounds
on the number of variables that a support still needs.  It is the quantum version of the searches
for small stopping sets and low-weight codewords in classical LDPC codes by G. Richter (Turbo Codes
2006), Rosnes and Ytrehus (IEEE Trans. Inf. Theory 55, 4167, 2009), and Khatami, Danjean, Nguyen,
and Vasić (ITA 2013), as Okada and Kasai describe in Section V of https://arxiv.org/abs/2607.14091.
The authors' implementation is in https://github.com/kasaikenta/pair-partition-cpm-css-codes.

``connected_support.md`` beside this module derives the search and states its correctness
requirements.  The Brouwer-Zimmermann search in ``brouwer_zimmermann.py`` solves the same problem
for any binary code.  This module reuses the weight divisors of ``information_sets.py``, and the
checks and orbits of automorphisms of ``automorphism_families.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse

from qldpc._binary import get_kernel, pack_bits, popcount_word, trailing_zeros_word
from qldpc._compilation import always_compiled

from .._binary import get_nested_code_basis
from .._tuning import get_tuning
from . import automorphism_families
from .information_sets import infer_weight_divisor

# The largest scale of the weights of checks, which keeps their sums far from overflowing 64 bits.
_MAX_SCALE = 2**40

# Added to the level of a variable while it is blocked (see _search_supports).
_BLOCKED = 2**40


####################################################################################################
# search entry points


def get_distance_connected_support(
    problems: Sequence[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]],
    *,
    cutoff: int,
    upper_bound: int | None = None,
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
) -> int:
    """Compute the minimum distance across problems by searching connected supports.

    Each problem is a pair ``(checks, excluded)`` of binary matrices with the same number of
    columns, whose distance is the least weight of a vector in the kernel of ``checks`` and outside
    the row space of ``excluded`` (see ``SupportSearch``).  The row space of ``excluded`` must lie
    in the kernel of ``checks``.

    The search of each problem raises its weight limit ``W`` through the weights that a witness may
    have, and each pass through a weight limit either finds a witness of weight at most ``W``,
    which is the distance if the lower limits found none, or certifies that every witness is
    heavier.  The first limit is ``cutoff`` if that is larger.  If the search finds a witness of
    weight at most ``cutoff``, the result is the weight of that witness, which can exceed the
    distance.

    Args:
        problems: The ``(checks, excluded)`` pairs to search.
        cutoff: Exit early and return once an upper bound on the distance falls to or below this
            cutoff.
        upper_bound: The weight of a known witness of a problem that is not in ``problems``, if
            any.  The result is the minimum of this bound and the distances of ``problems``.
        get_automorphisms: A function that returns an array of generators of a group of column
            permutations, or None.  Row ``i`` of the array maps column ``j`` to column ``row[j]``.
            Each generator must map the kernel of the checks of every problem, and the row space of
            its excluded rows, onto themselves.  The search of each problem calls the function at
            most once, before the first weight limit at which it expects to have visited
            ``automorphism_minimum_states`` states (a field of the tuning of ``_tuning.py``), and
            verifies the generators before it uses them.

    Returns:
        The minimum distance across the problems and the optional upper bound, or an upper bound
        that is at most ``cutoff``.  A problem without a word outside its excluded row space has no
        distance, and counts as one more than its width.

    Raises:
        ValueError: If there are neither problems nor an upper bound, if the matrices of a problem
            are not binary matrices of equal width, if the excluded rows of a problem are not in the
            kernel of its checks, or if a generator from ``get_automorphisms`` does not map the
            kernel of the checks of a searched problem, or the row space of its excluded rows, onto
            itself.
    """
    if not problems and upper_bound is None:
        raise ValueError("Require at least one problem or an upper bound")

    # A problem without witnesses only bounds the result.  A known bound within the cutoff ends the
    # search before it starts.
    automorphism_minimum_states = get_tuning().automorphism_minimum_states
    best_weight = upper_bound
    no_distance_bounds = []
    searches: list[SupportSearch] = []
    for checks, excluded in problems:
        search = SupportSearch(
            checks,
            excluded,
            get_automorphisms=get_automorphisms,
            automorphism_minimum_states=automorphism_minimum_states,
        )
        if search.has_witnesses:
            searches.append(search)
        else:
            no_distance_bounds.append(search.num_variables + 1)
    if not searches or (best_weight is not None and best_weight <= cutoff):
        return min([bound for bound in (best_weight, *no_distance_bounds) if bound is not None])

    # No witness of a problem has a weight below its lower bound, which is a multiple of the weight
    # divisor.  Search the problem with the lowest bound, until the best known witness is as light.
    lower_bounds = [search.weight_divisor for search in searches]
    while best_weight is None or best_weight > min(lower_bounds):
        index = lower_bounds.index(min(lower_bounds))
        search = searches[index]
        max_weight = max(lower_bounds[index], cutoff)
        witness = search.find_witness(max_weight)
        if witness is None:
            divisor = search.weight_divisor
            lower_bounds[index] = (max_weight // divisor + 1) * divisor
            continue
        best_weight = len(witness) if best_weight is None else min(best_weight, len(witness))
        if best_weight <= cutoff:
            break
    return min([best_weight, *no_distance_bounds])


class SupportSearch:
    """Search for light words of the kernel of ``checks`` outside the row space of ``excluded``.

    A word is a binary vector ``x`` with ``checks @ x = 0``, and its support is the set of its
    nonzero coordinates, which are its variables.  The row space of ``excluded`` must lie in the
    kernel of ``checks``, and the words outside it are the witnesses.  The search grows a support
    through the checks that it leaves unsatisfied (see ``connected_support.md``), so it is fast if
    ``checks`` is sparse.

    Args:
        checks: A binary matrix whose rows are the checks.  Redundant rows can strengthen the
            search, so the matrix need not have independent rows.
        excluded: A binary matrix whose row space is excluded.

    Keyword args:
        get_automorphisms: A function that returns an array of generators of a group of column
            permutations, or None, as in ``get_distance_connected_support``.  The search calls it
            at most once, before the first weight limit whose states, together with those visited
            so far, are expected to reach ``automorphism_minimum_states``.
        automorphism_minimum_states: The number of states from which the search uses
            automorphisms.  Zero, the default, asks for them before the first weight limit.

    Attributes:
        num_variables: The number of columns of ``checks``.
        weight_divisor: A divisor of the weight of every word of the kernel of ``checks``.
        num_states: The number of states that the search has visited so far, over all of its weight
            limits.

    Raises:
        ValueError: If the matrices are not binary matrices of equal width, or if the excluded rows
            are not in the kernel of the checks.
    """

    num_variables: int
    weight_divisor: int
    num_states: int

    def __init__(
        self,
        checks: npt.NDArray[np.uint8],
        excluded: npt.NDArray[np.uint8],
        *,
        get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
        automorphism_minimum_states: int = 0,
    ) -> None:
        if checks.ndim != 2 or excluded.ndim != 2 or checks.shape[1] != excluded.shape[1]:
            raise ValueError("Checks and excluded rows must be matrices with equal widths")
        if np.any(checks > 1) or np.any(excluded > 1):
            raise ValueError("Checks and excluded rows must be binary")
        if np.any(checks.astype(np.float32) @ excluded.astype(np.float32).T % 2):
            raise ValueError("The excluded rows must lie in the kernel of the checks")
        self.num_variables = checks.shape[1]
        self.num_states = 0
        self._last_two_num_states = (0, 0)
        self._excluded = excluded
        self._get_automorphisms = get_automorphisms
        self._automorphism_minimum_states = automorphism_minimum_states

        # Zero and duplicate checks add nothing to a search.
        nonzero_checks = checks[np.any(checks, axis=1)]
        _, first_rows = np.unique(nonzero_checks, axis=0, return_index=True)
        self._checks = nonzero_checks[np.sort(first_rows)]
        sparse_checks = scipy.sparse.csr_array(self._checks)
        self._check_starts = sparse_checks.indptr.astype(np.int64)
        self._check_variables = sparse_checks.indices.astype(np.int64)
        self._check_masks = pack_bits(np.ascontiguousarray(self._checks.T))
        self._test_columns = pack_bits(np.ascontiguousarray(self._get_tests().T))
        self._roots = np.arange(self.num_variables, dtype=np.int64)
        self._ranks = np.arange(self.num_variables, dtype=np.int64)
        self.weight_divisor = infer_weight_divisor(get_kernel(self._checks))

        # Weights of checks are multiples of 1 / scale, and an overlap is at most the largest column
        # weight.  Every overlap divides the scale if the scale is small enough, and the inverse of
        # an overlap is rounded down otherwise.
        max_column_weight = int(np.max(np.sum(self._checks, axis=0), initial=1))
        self._scale = min(math.lcm(*range(1, max_column_weight + 1)), _MAX_SCALE)
        self._inverse_overlaps = np.zeros(max_column_weight + 1, dtype=np.int64)
        self._inverse_overlaps[1:] = self._scale // np.arange(1, max_column_weight + 1)

    @property
    def has_witnesses(self) -> bool:
        """Whether any word of the kernel of the checks is outside the row space of ``excluded``."""
        return self._test_columns.shape[1] > 0

    def find_witness(self, max_weight: int) -> npt.NDArray[np.int64] | None:
        """Search for a witness with at most ``max_weight`` variables.

        The search is exhaustive: it finds a witness if there is one of weight at most
        ``max_weight``, and otherwise certifies that every witness is heavier.

        Args:
            max_weight: The weight limit, which is at least one.

        Returns:
            The sorted variables of a witness, or None if there is no witness of weight at most
            ``max_weight``.  A witness need not be a lightest one.

        Raises:
            ValueError: If this search asks for automorphisms, and a generator does not map the
                kernel of the checks, or the row space of the excluded rows, onto itself.
        """
        # Ask for automorphisms once, before the first weight limit that is expected to reach the
        # threshold.
        if (
            self._get_automorphisms is not None
            and self.num_states + self._predict_states() >= self._automorphism_minimum_states
        ):
            automorphisms, self._get_automorphisms = self._get_automorphisms(), None
            if automorphisms is not None:
                self._use_automorphisms(automorphisms)
        support, num_states = _search_supports(
            self._check_starts,
            self._check_variables,
            self._check_masks,
            self._test_columns,
            self._roots,
            self._ranks,
            self._scale,
            self._inverse_overlaps,
            int(min(max_weight, self.num_variables)),
        )
        self.num_states += num_states
        self._last_two_num_states = (self._last_two_num_states[1], num_states)
        return np.sort(support) if len(support) else None

    def _predict_states(self) -> int:
        """The number of states that the next weight limit is expected to visit.

        Each weight limit is expected to grow by the ratio between the last two, or not to grow if
        the limit before the last visited no states, as before the second limit (see Section 3.5
        of ``connected_support.md``).
        """
        previous, last = self._last_two_num_states
        return last * last // previous if previous else last

    def _get_tests(self) -> npt.NDArray[np.uint8]:
        """Rows whose products with a word all vanish exactly when the word is excluded.

        The row space of ``excluded`` is the set of vectors that are orthogonal to the kernel of
        ``excluded``.  That kernel contains the row space of ``checks``, which every word is
        orthogonal to, so it is enough to test against the rest of the kernel of ``excluded``.
        """
        basis, labels = get_nested_code_basis(get_kernel(self._excluded), self._checks)
        return basis[np.any(labels, axis=1)]

    def _use_automorphisms(self, automorphisms: npt.NDArray[np.int_]) -> None:
        """Verify automorphisms, and search only from the least variable of each of their orbits.

        The variables are ranked orbit by orbit, so that the search from each root excludes every
        variable of the earlier orbits.

        Raises:
            ValueError: If a generator does not map the kernel of the checks, or the row space of
                the excluded rows, onto itself.
        """
        basis, labels = get_nested_code_basis(get_kernel(self._checks), self._excluded)
        automorphism_families.verify_automorphisms(basis, labels, automorphisms)
        orbits = automorphism_families.get_orbits(automorphisms, self.num_variables)
        self._roots = np.unique(orbits, return_index=True)[1].astype(np.int64)
        self._ranks = np.empty(self.num_variables, dtype=np.int64)
        self._ranks[np.lexsort((np.arange(self.num_variables), orbits))] = np.arange(
            self.num_variables
        )


####################################################################################################
# native search


@always_compiled
def _search_supports(
    check_starts: npt.NDArray[np.int64],
    check_variables: npt.NDArray[np.int64],
    check_masks: npt.NDArray[np.uint64],
    test_columns: npt.NDArray[np.uint64],
    roots: npt.NDArray[np.int64],
    ranks: npt.NDArray[np.int64],
    scale: int,
    inverse_overlaps: npt.NDArray[np.int64],
    max_weight: int,
) -> tuple[npt.NDArray[np.int64], int]:
    """Search the supports that grow from each root through unsatisfied checks.

    The graph is the Tanner graph of the checks: check ``c`` has the variables
    ``check_variables[check_starts[c] : check_starts[c + 1]]``.  Row ``v`` of ``check_masks`` packs
    the checks of variable ``v`` into the bits of 64-bit blocks, and row ``v`` of ``test_columns``
    packs the tests that a word with the support ``{v}`` fails.  The search from a root uses only
    variables of higher rank.  The weights of checks are integers, in units of ``1 / scale``, and
    entry ``k`` of ``inverse_overlaps`` is ``scale // k``, up to the largest column weight (see
    ``_get_branches``).

    It is the depth-first search of Algorithm 2 of https://arxiv.org/abs/2607.14091, with an
    explicit stack.  A state is a support, whose variables are ``support[:depth]``, and its
    syndrome is the set of its unsatisfied checks.  A state branches on the viable variables of one
    unsatisfied check.  Each variable that it tries stays blocked until the search leaves the
    state, so that its later branches do not use it again.  A state also allows only its viable
    variables in its subtree, if it rules out any.  The state at depth zero is the empty support,
    which branches on the root.

    Returns:
        The variables of a witness of weight at most ``max_weight``, or an empty array, and the
        number of states that the search visited.
    """
    # The sizes of the stack, and of the branches of a state.
    num_variables, num_blocks = check_masks.shape
    max_column_weight = len(inverse_overlaps) - 1
    max_row_weight = 1
    for check in range(len(check_starts) - 1):
        max_row_weight = max(max_row_weight, check_starts[check + 1] - check_starts[check])

    # The level of a variable counts the states in the stack whose restrictions allow it, plus
    # _BLOCKED if it is blocked: if it is in the support, was tried by an earlier sibling of a state
    # in the stack, or has a lower rank than the root.  A variable is available if its level is the
    # number of active restrictions, which is never as large as _BLOCKED.
    levels = np.zeros(num_variables, dtype=np.int64)
    num_restrictions = 0

    # The state at each depth has its syndrome, its branch variables, and the cursor of the next
    # branch variable to try.  If it restricts its subtree, the variables that it allows are
    # allowances[allowance_starts[depth] : allowance_starts[depth + 1]].  A state allows each
    # variable at most once, and allowances grows if a state could overflow it.
    syndromes = np.zeros((max_weight + 2, num_blocks), dtype=np.uint64)
    support = np.empty(max_weight + 1, dtype=np.int64)
    branches = np.empty((max_weight + 1, max_row_weight), dtype=np.int64)
    num_branches = np.zeros(max_weight + 1, dtype=np.int64)
    cursors = np.zeros(max_weight + 1, dtype=np.int64)
    restricts = np.zeros(max_weight + 1, dtype=np.bool_)
    allowance_starts = np.zeros(max_weight + 2, dtype=np.int64)
    allowances = np.empty(num_variables, dtype=np.int64)

    # Scratch space for _get_branches.
    check_weights = np.zeros(len(check_starts) - 1, dtype=np.int64)
    branch_repairs = np.empty(max_row_weight, dtype=np.int64)

    num_states = 0
    for root in roots:
        for variable in range(num_variables):
            if ranks[variable] < ranks[root]:
                levels[variable] += _BLOCKED
        branches[0, 0] = root
        num_branches[0] = 1
        cursors[0] = 0
        depth = 0
        while depth >= 0:
            # Leave the state once it has tried every branch.  The variables that it tried are
            # available again to its later siblings, and so are the variables that it did not allow.
            if cursors[depth] == num_branches[depth]:
                for index in range(cursors[depth]):
                    levels[branches[depth, index]] -= _BLOCKED
                if restricts[depth]:
                    for index in range(allowance_starts[depth], allowance_starts[depth + 1]):
                        levels[allowances[index]] -= 1
                    num_restrictions -= 1
                depth -= 1
                continue

            # Take the next branch, and skip its state unless it can still meet the weight limit.
            # Every variable is in at most max_column_weight unsatisfied checks.
            variable = branches[depth, cursors[depth]]
            cursors[depth] += 1
            levels[variable] += _BLOCKED
            support[depth] = variable
            num_unsatisfied = _xor_and_count(syndromes, depth, check_masks, variable)
            needed = (num_unsatisfied + max_column_weight - 1) // max_column_weight
            if depth + 1 + needed > max_weight:
                continue

            # Enter the state: return it if it is a witness, and otherwise choose its branches.
            depth += 1
            num_states += 1
            num_branches[depth] = 0
            cursors[depth] = 0
            allowance_starts[depth + 1] = allowance_starts[depth]
            restricts[depth] = False
            if num_unsatisfied == 0:
                if _is_outside_excluded(test_columns, support, depth):
                    return support[:depth].copy(), num_states
                continue
            if allowance_starts[depth] + num_variables > len(allowances):
                grown = np.empty(2 * len(allowances) + num_variables, dtype=np.int64)
                grown[: allowance_starts[depth]] = allowances[: allowance_starts[depth]]
                allowances = grown
            num_branches[depth] = _get_branches(
                check_starts,
                check_variables,
                check_masks,
                levels,
                num_restrictions,
                syndromes,
                branches,
                restricts,
                allowance_starts,
                allowances,
                depth,
                max_weight - depth,
                scale,
                inverse_overlaps,
                check_weights,
                branch_repairs,
            )
            num_restrictions += restricts[depth]

        # Unblock the variables of lower rank than the root.
        for variable in range(num_variables):
            if ranks[variable] < ranks[root]:
                levels[variable] -= _BLOCKED
    return np.empty(0, dtype=np.int64), num_states


@always_compiled
def _get_branches(
    check_starts: npt.NDArray[np.int64],
    check_variables: npt.NDArray[np.int64],
    check_masks: npt.NDArray[np.uint64],
    levels: npt.NDArray[np.int64],
    available_level: int,
    syndromes: npt.NDArray[np.uint64],
    branches: npt.NDArray[np.int64],
    restricts: npt.NDArray[np.bool_],
    allowance_starts: npt.NDArray[np.int64],
    allowances: npt.NDArray[np.int64],
    depth: int,
    slack: int,
    scale: int,
    inverse_overlaps: npt.NDArray[np.int64],
    check_weights: npt.NDArray[np.int64],
    branch_repairs: npt.NDArray[np.int64],
) -> int:
    """Choose the variables that the state at a depth branches on, if its syndrome is not empty.

    A variable is available if its level is ``available_level``.  A support that has ``slack``
    variables to spare is completed by a set ``T`` of available variables that every unsatisfied
    check meets an odd number of times.  The overlap of an available variable is the number of
    unsatisfied checks that it is in.  Weigh each unsatisfied check by the inverse of the largest
    overlap of its available variables, in units of ``1 / scale``, rounded down, as
    ``inverse_overlaps`` lists.  The weights of the unsatisfied checks of any available variable
    then total at most one, and their total is its repair.  So ``T`` has at least as many variables
    as the total weight of the unsatisfied checks.  A variable ``v`` of ``T`` repairs its share,
    and the other variables of ``T`` the rest, so ``|T| >= 1 + total - repair(v)``.

    A variable for which this exceeds ``slack`` is in no completion, and the state allows only the
    other variables in its subtree.  It records them in ``allowances``, from
    ``allowance_starts[depth]``, and sets ``restricts[depth]``.  The state has no branches if an
    unsatisfied check has no viable variable, or if the total weight exceeds ``slack``.  Otherwise
    it branches on the viable variables of the unsatisfied check with the fewest, which it writes to
    ``branches[depth]``, most repairing first.  It returns their number.  ``check_weights`` and
    ``branch_repairs`` are scratch space for the weights of the unsatisfied checks and the repairs
    of the branch variables.
    """
    num_blocks = syndromes.shape[1]

    # Weigh each unsatisfied check, and choose the one with the fewest available variables.
    total_weight = 0
    branch_check = -1
    fewest = check_starts[-1] + 1
    for block in range(num_blocks):
        bits = syndromes[depth, block]
        while bits:
            check = 64 * block + int(trailing_zeros_word(bits))
            bits &= bits - np.uint64(1)
            largest_overlap = 0
            count = 0
            for position in range(check_starts[check], check_starts[check + 1]):
                variable = check_variables[position]
                if levels[variable] == available_level:
                    overlap = _get_overlap(syndromes, depth, check_masks, variable)
                    largest_overlap = max(largest_overlap, overlap)
                    count += 1
            if count == 0:
                return 0
            check_weights[check] = inverse_overlaps[largest_overlap]
            total_weight += check_weights[check]
            # The total only grows, so the state fails as soon as it exceeds the slack.
            if total_weight > slack * scale:
                return 0
            if count < fewest:
                branch_check, fewest = check, count

    # A variable outside the unsatisfied checks repairs nothing, so it is viable only if every
    # variable is.  Otherwise allow only the viable variables, and count them in each check.
    if total_weight > (slack - 1) * scale:
        restricts[depth] = True
        allowance_end = allowance_starts[depth]
        branch_check = -1
        fewest = check_starts[-1] + 1
        for block in range(num_blocks):
            bits = syndromes[depth, block]
            while bits:
                check = 64 * block + int(trailing_zeros_word(bits))
                bits &= bits - np.uint64(1)
                count = 0
                for position in range(check_starts[check], check_starts[check + 1]):
                    variable = check_variables[position]
                    if levels[variable] == available_level:
                        repair = _get_repair(syndromes, depth, check_masks, variable, check_weights)
                        if total_weight - repair <= (slack - 1) * scale:
                            levels[variable] += 1
                            allowances[allowance_end] = variable
                            allowance_end += 1
                    count += levels[variable] == available_level + 1
                if count == 0:
                    allowance_starts[depth + 1] = allowance_end
                    return 0
                if count < fewest:
                    branch_check, fewest = check, count
        allowance_starts[depth + 1] = allowance_end
        available_level += 1

    # Branch on the available variables of the chosen check, most repairing first.  On the
    # [[276, 98, 14]] code, this order visits a quarter of the states of the order of the variables
    # through the weight limit at which the search finds a witness (Section 3.3 of the guide).
    num_branches = 0
    for position in range(check_starts[branch_check], check_starts[branch_check + 1]):
        variable = check_variables[position]
        if levels[variable] != available_level:
            continue
        repair = _get_repair(syndromes, depth, check_masks, variable, check_weights)
        index = num_branches
        while index > 0 and branch_repairs[index - 1] < repair:
            branches[depth, index] = branches[depth, index - 1]
            branch_repairs[index] = branch_repairs[index - 1]
            index -= 1
        branches[depth, index] = variable
        branch_repairs[index] = repair
        num_branches += 1
    return num_branches


@always_compiled(inline=True)
def _get_overlap(
    syndromes: npt.NDArray[np.uint64],
    depth: int,
    check_masks: npt.NDArray[np.uint64],
    variable: int,
) -> int:
    """The number of unsatisfied checks of a variable, in the syndrome at a depth."""
    count = 0
    for block in range(syndromes.shape[1]):
        count += int(popcount_word(syndromes[depth, block] & check_masks[variable, block]))
    return count


@always_compiled(inline=True)
def _get_repair(
    syndromes: npt.NDArray[np.uint64],
    depth: int,
    check_masks: npt.NDArray[np.uint64],
    variable: int,
    check_weights: npt.NDArray[np.int64],
) -> int:
    """The total weight of the unsatisfied checks of a variable, in the syndrome at a depth."""
    repair = 0
    for block in range(syndromes.shape[1]):
        bits = syndromes[depth, block] & check_masks[variable, block]
        while bits:
            repair += check_weights[64 * block + int(trailing_zeros_word(bits))]
            bits &= bits - np.uint64(1)
    return repair


@always_compiled(inline=True)
def _xor_and_count(
    syndromes: npt.NDArray[np.uint64],
    depth: int,
    check_masks: npt.NDArray[np.uint64],
    variable: int,
) -> int:
    """Set the syndrome at ``depth + 1`` to that at ``depth`` with a variable added.

    Returns:
        The number of unsatisfied checks of the new syndrome.
    """
    count = 0
    for block in range(syndromes.shape[1]):
        syndromes[depth + 1, block] = syndromes[depth, block] ^ check_masks[variable, block]
        count += int(popcount_word(syndromes[depth + 1, block]))
    return count


@always_compiled
def _is_outside_excluded(
    test_columns: npt.NDArray[np.uint64], support: npt.NDArray[np.int64], size: int
) -> bool:
    """Whether the word that ``support[:size]`` supports fails a test."""
    for block in range(test_columns.shape[1]):
        combined = test_columns[support[0], block]
        for index in range(1, size):
            combined ^= test_columns[support[index], block]
        if combined:
            return True
    return False
