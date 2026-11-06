# SPDX-License-Identifier: Apache-2.0

"""Exact binary distances by a depth-first search of the supports connected in a Tanner graph.

The search finds a nonzero vector ``x`` with ``H x = 0`` outside the row space of ``G``, for sparse
parity checks ``H``, by growing a support only through unsatisfied checks, and pruning with bounds
on the number of variables that a support still needs.  It is the quantum version of the searches
for small stopping sets and low-weight codewords in classical LDPC codes by Richter (ISIT 2006),
Rosnes and Ytrehus (IEEE Trans. Inf. Theory 55, 4167, 2009), and Khatami, Danjean, Nguyen, and
Vasić (ITA 2013), as Okada and Kasai describe in Section V of https://arxiv.org/abs/2607.14091.  The
authors' implementation is in https://github.com/kasaikenta/pair-partition-cpm-css-codes.

``connected_support.md`` beside this module derives the search and states its correctness
requirements.  The Brouwer-Zimmermann search in ``brouwer_zimmermann.py`` solves the same problem
for any code.  This module reuses its weight divisors, and its checks of automorphisms.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_kernel, pack_bits, popcount_word, trailing_zeros_word
from qldpc._compilation import always_compiled

from .._binary import get_nested_code_basis
from . import automorphism_families
from .information_sets import infer_weight_divisor

# A search asks for automorphisms once it has visited this many states.  Finding automorphisms can
# take milliseconds, a small fraction of the time of a search that visits so many states.
MINIMUM_STATES_FOR_AUTOMORPHISMS = 100_000

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

    The search raises its weight limit ``W`` through the weights that a word may have, and each
    search through ``W`` either finds a word of weight at most ``W``, which is the distance if the
    searches through lower limits found none, or certifies that every word is heavier.  The first
    limit is ``cutoff`` if that is larger.  If a search finds a word of weight at most ``cutoff``,
    the result is the weight of that word, which can exceed the distance.

    Args:
        problems: The ``(checks, excluded)`` pairs to search.
        cutoff: Exit early and return once an upper bound on the distance falls to or below this
            cutoff.
        upper_bound: The weight of a known word that some problem omitted from ``problems`` is
            satisfied by, if any.  The result is the minimum of this bound and the distances of
            ``problems``.
        get_automorphisms: A function that returns an array of generators of a group of column
            permutations, or None.  Row ``i`` of the array maps column ``j`` to column ``row[j]``.
            Each generator must map the kernel of the checks of every problem, and the row space of
            its excluded rows, onto themselves.  A search calls the function once it has visited
            many states, and verifies the generators before it uses them.

    Returns:
        The minimum distance across the problems and the optional upper bound, or an upper bound
        that is at most ``cutoff``.  A problem without a word outside its excluded row space has no
        distance, and counts as one more than its width.

    Raises:
        ValueError: If the matrices of a problem are not binary matrices of equal width, if the
            excluded rows of a problem are not in the kernel of its checks, or if a generator from
            ``get_automorphisms`` does not map the kernel of the checks of a searched problem, or
            the row space of its excluded rows, onto itself.
    """
    best = upper_bound
    no_distance_bounds = []
    searches: list[SupportSearch] = []
    for checks, excluded in problems:
        search = SupportSearch(checks, excluded, get_automorphisms=get_automorphisms)
        if search.has_witnesses:
            searches.append(search)
        else:
            no_distance_bounds.append(search.num_variables + 1)
    if not searches or (best is not None and best <= cutoff):
        return min([bound for bound in (best, *no_distance_bounds) if bound is not None])

    # No word of a problem has a weight below its lower bound, which is a multiple of the weight
    # divisor.  Search the problem with the lowest bound, until the best known word is that light.
    lower_bounds = [search.weight_divisor for search in searches]
    while best is None or best > min(lower_bounds):
        index = lower_bounds.index(min(lower_bounds))
        search = searches[index]
        max_weight = min(max(lower_bounds[index], cutoff), search.num_variables)
        witness = search.find_witness(max_weight)
        if witness is None:
            divisor = search.weight_divisor
            lower_bounds[index] = (max_weight // divisor + 1) * divisor
            continue
        best = len(witness) if best is None else min(best, len(witness))
        if best <= cutoff:
            break
    return min([best, *no_distance_bounds])


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
            once, after it has visited ``MINIMUM_STATES_FOR_AUTOMORPHISMS`` states.

    Attributes:
        num_variables: The number of columns of ``checks``.
        weight_divisor: A divisor of the weight of every word of the kernel of ``checks``.
        num_states: The number of states that the searches have visited so far.
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
    ) -> None:
        if checks.ndim != 2 or excluded.ndim != 2 or checks.shape[1] != excluded.shape[1]:
            raise ValueError("Checks and excluded rows must be matrices with equal widths")
        if np.any(checks.astype(np.float32) @ excluded.astype(np.float32).T % 2):
            raise ValueError("The excluded rows must lie in the kernel of the checks")
        self.num_variables = checks.shape[1]
        self.num_states = 0
        self._excluded = excluded
        self._get_automorphisms = get_automorphisms

        # Zero and duplicate checks add nothing to a search.
        nonzero_checks = checks[np.any(checks, axis=1)]
        _, first_rows = np.unique(nonzero_checks, axis=0, return_index=True)
        self._checks = nonzero_checks[np.sort(first_rows)]
        self._graph = _TannerGraph(
            *_get_sparse_rows(self._checks), pack_bits(np.ascontiguousarray(self._checks.T))
        )
        self._test_columns = pack_bits(np.ascontiguousarray(self._get_tests().T))
        self._roots = np.arange(self.num_variables, dtype=np.int64)
        self.weight_divisor = infer_weight_divisor(get_kernel(self._checks))

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
        """
        if (
            self._get_automorphisms is not None
            and self.num_states >= MINIMUM_STATES_FOR_AUTOMORPHISMS
        ):
            automorphisms, self._get_automorphisms = self._get_automorphisms(), None
            if automorphisms is not None:
                self._use_automorphisms(automorphisms)
        support, num_states = _search_supports(
            *self._graph,
            self._test_columns,
            self._roots,
            int(min(max_weight, self.num_variables)),
        )
        self.num_states += num_states
        return np.sort(support) if len(support) else None

    def _get_tests(self) -> npt.NDArray[np.uint8]:
        """Rows whose products with a word of the kernel vanish only if the word is excluded.

        The row space of ``excluded`` is the set of vectors that are orthogonal to its kernel.  The
        kernel contains the row space of ``checks``, which every word of the kernel is orthogonal
        to, so it is enough to test against the rest of the kernel.
        """
        basis, labels = get_nested_code_basis(get_kernel(self._excluded), self._checks)
        return basis[np.any(labels, axis=1)]

    def _use_automorphisms(self, automorphisms: npt.NDArray[np.int_]) -> None:
        """Search only from the least variable of each orbit of the automorphisms."""
        basis, labels = get_nested_code_basis(get_kernel(self._checks), self._excluded)
        automorphism_families.verify_automorphisms(basis, labels, automorphisms)
        orbits = automorphism_families._get_orbits(automorphisms, self.num_variables)
        self._roots = np.unique(orbits, return_index=True)[1].astype(np.int64)


####################################################################################################
# helpers


class _TannerGraph(NamedTuple):
    """The Tanner graph of checks, in the arrays that ``_search_supports`` takes."""

    check_starts: npt.NDArray[np.int64]
    check_variables: npt.NDArray[np.int64]
    check_masks: npt.NDArray[np.uint64]


def _get_sparse_rows(
    matrix: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The nonzero columns of each row of a matrix, as the starts of its rows and their columns."""
    rows, columns = np.nonzero(matrix)
    starts = np.zeros(len(matrix) + 1, dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=len(matrix)), out=starts[1:])
    return starts, columns.astype(np.int64)


####################################################################################################
# native search


@always_compiled
def _search_supports(
    check_starts: npt.NDArray[np.int64],
    check_variables: npt.NDArray[np.int64],
    check_masks: npt.NDArray[np.uint64],
    test_columns: npt.NDArray[np.uint64],
    roots: npt.NDArray[np.int64],
    max_weight: int,
) -> tuple[npt.NDArray[np.int64], int]:
    """Search the supports that grow from each root through unsatisfied checks.

    The graph is the Tanner graph of the checks: check ``c`` has the variables
    ``check_variables[check_starts[c] : check_starts[c + 1]]``.  Row ``v`` of ``check_masks`` packs
    the checks of variable ``v`` into bits, and row ``v`` of ``test_columns`` packs the tests that a
    word with the support ``{v}`` fails.  The search from a root uses only variables above it.

    It is the depth-first search of Algorithm 2 of https://arxiv.org/abs/2607.14091, with an
    explicit stack.  A state is a support, whose variables are ``support[:depth]``, and its
    syndrome is the set of its unsatisfied checks.  A state branches on the available variables of
    one unsatisfied check, and blocks each variable for its later siblings, and for itself.  The
    state at depth zero is the empty support, which branches on the root.

    Returns:
        The variables of a witness of weight at most ``max_weight``, or none, and the number of
        states that the search visited.
    """
    num_variables = check_masks.shape[0]
    num_words = check_masks.shape[1]
    max_column_weight = 1
    for variable in range(num_variables):
        max_column_weight = max(max_column_weight, _count_bits(check_masks, variable))
    max_row_weight = 1
    for check in range(len(check_starts) - 1):
        max_row_weight = max(max_row_weight, check_starts[check + 1] - check_starts[check])

    # A variable is blocked if it is in the support, was tried by a sibling of a state in the
    # stack, or precedes the root.
    blocked = np.zeros(num_variables, dtype=np.bool_)

    # The state at each depth has its syndrome, and its branch variables.  A cursor marks the next
    # branch variable to try.  Its other entries are the total weight of its unsatisfied checks, and
    # how much of that weight each branch variable repairs (see ``_get_branches``).
    syndromes = np.zeros((max_weight + 2, num_words), dtype=np.uint64)
    support = np.empty(max_weight + 1, dtype=np.int64)
    branches = np.empty((max_weight + 1, max_row_weight), dtype=np.int64)
    repairs = np.empty((max_weight + 1, max_row_weight), dtype=np.float64)
    totals = np.zeros(max_weight + 1, dtype=np.float64)
    num_branches = np.zeros(max_weight + 1, dtype=np.int64)
    cursors = np.zeros(max_weight + 1, dtype=np.int64)
    widths = np.zeros(len(check_starts) - 1, dtype=np.int64)
    inverses = np.ones(max_column_weight + 1, dtype=np.float64)
    for width in range(1, max_column_weight + 1):
        inverses[width] = 1.0 / width

    num_states = 0
    for root in roots:
        blocked[:root] = True
        branches[0, 0] = root
        repairs[0, 0] = 0.0
        num_branches[0] = 1
        cursors[0] = 0
        totals[0] = 0.0
        depth = 0
        entering = False
        while depth >= 0:
            if entering:
                entering = False
                num_states += 1
                num_branches[depth] = 0
                cursors[depth] = 0
                if _is_empty(syndromes, depth):
                    if _is_outside_excluded(test_columns, support, depth):
                        return support[:depth].copy(), num_states
                else:
                    num_branches[depth] = _get_branches(
                        check_starts,
                        check_variables,
                        check_masks,
                        blocked,
                        syndromes,
                        branches,
                        repairs,
                        totals,
                        depth,
                        max_weight - depth,
                        widths,
                        inverses,
                    )

            if cursors[depth] < num_branches[depth]:
                # A variable that repairs too little of the total weight leaves too much for the
                # remaining weight limit, and so do the variables after it.
                if totals[depth] - repairs[depth, cursors[depth]] > max_weight - depth - 1 + 1e-9:
                    num_branches[depth] = cursors[depth]
                    continue
                variable = branches[depth, cursors[depth]]
                cursors[depth] += 1
                blocked[variable] = True
                support[depth] = variable
                num_unsatisfied = _xor_and_count(syndromes, depth, check_masks, variable)
                # Every variable repairs at most max_column_weight unsatisfied checks.
                needed = (num_unsatisfied + max_column_weight - 1) // max_column_weight
                if depth + 1 + needed <= max_weight:
                    depth += 1
                    entering = True
            else:
                # The variables that this state tried are available again to the later siblings of
                # its parent.
                for index in range(cursors[depth]):
                    blocked[branches[depth, index]] = False
                depth -= 1
        blocked[:root] = False
    return np.empty(0, dtype=np.int64), num_states


@always_compiled
def _get_branches(
    check_starts: npt.NDArray[np.int64],
    check_variables: npt.NDArray[np.int64],
    check_masks: npt.NDArray[np.uint64],
    blocked: npt.NDArray[np.bool_],
    syndromes: npt.NDArray[np.uint64],
    branches: npt.NDArray[np.int64],
    repairs: npt.NDArray[np.float64],
    totals: npt.NDArray[np.float64],
    depth: int,
    slack: int,
    widths: npt.NDArray[np.int64],
    inverses: npt.NDArray[np.float64],
) -> int:
    """Choose the variables that the state at a depth branches on, if its syndrome is not empty.

    A state has no branches if an unsatisfied check has no available variable.  Otherwise it
    branches on the available variables of the unsatisfied check with the fewest.  The variables are
    written to ``branches[depth]``, and their number is returned.

    The state also bounds the number of variables that it still needs.  An available variable is in
    as many unsatisfied checks as it repairs, which is its overlap.  Weigh each unsatisfied check by
    the inverse of the largest overlap of its available variables, which ``widths`` records.
    The weights of the unsatisfied checks of any available variable then total at most one, so the
    state needs at least as many variables as the total weight of its unsatisfied checks, which is
    written to ``totals[depth]``.  A state with more weight than its ``slack`` of variables to
    spare has no branches.  The branches are in order of how much of that weight they repair, which
    is written to ``repairs[depth]``.
    """
    num_words = syndromes.shape[1]
    best = -1
    best_count = int(check_starts[-1]) + 1
    total = 0.0
    for word in range(num_words):
        bits = syndromes[depth, word]
        while bits:
            check = 64 * word + int(trailing_zeros_word(bits))
            bits &= bits - np.uint64(1)
            count = 0
            largest = 0
            for position in range(check_starts[check], check_starts[check + 1]):
                variable = check_variables[position]
                if not blocked[variable]:
                    count += 1
                    overlap = 0
                    for other in range(num_words):
                        overlap += int(
                            popcount_word(syndromes[depth, other] & check_masks[variable, other])
                        )
                    largest = max(largest, overlap)
            if count == 0:
                return 0
            if count < best_count:
                best, best_count = check, count
            widths[check] = largest
            total += inverses[largest]
            if total > slack + 1e-9:
                return 0
    totals[depth] = total

    num_branches = 0
    for position in range(check_starts[best], check_starts[best + 1]):
        variable = check_variables[position]
        if blocked[variable]:
            continue
        repaired = 0.0
        for word in range(num_words):
            bits = syndromes[depth, word] & check_masks[variable, word]
            while bits:
                repaired += inverses[widths[64 * word + int(trailing_zeros_word(bits))]]
                bits &= bits - np.uint64(1)
        index = num_branches
        while index > 0 and repairs[depth, index - 1] < repaired:
            branches[depth, index] = branches[depth, index - 1]
            repairs[depth, index] = repairs[depth, index - 1]
            index -= 1
        branches[depth, index] = variable
        repairs[depth, index] = repaired
        num_branches += 1
    return num_branches


@always_compiled(inline=True)
def _count_bits(words: npt.NDArray[np.uint64], row: int) -> int:
    """The number of set bits in a row of packed words."""
    count = 0
    for word in range(words.shape[1]):
        count += int(popcount_word(words[row, word]))
    return count


@always_compiled(inline=True)
def _xor_and_count(
    syndromes: npt.NDArray[np.uint64],
    depth: int,
    check_masks: npt.NDArray[np.uint64],
    variable: int,
) -> int:
    """Set the syndrome below a depth to that of the variable added to it, and count its checks."""
    count = 0
    for word in range(syndromes.shape[1]):
        syndromes[depth + 1, word] = syndromes[depth, word] ^ check_masks[variable, word]
        count += int(popcount_word(syndromes[depth + 1, word]))
    return count


@always_compiled(inline=True)
def _is_empty(syndromes: npt.NDArray[np.uint64], depth: int) -> bool:
    """Whether the syndrome at a depth has no unsatisfied check."""
    for word in range(syndromes.shape[1]):
        if syndromes[depth, word]:
            return False
    return True


@always_compiled
def _is_outside_excluded(
    test_columns: npt.NDArray[np.uint64], support: npt.NDArray[np.int64], size: int
) -> bool:
    """Whether the word that ``support[:size]`` supports fails a test."""
    for word in range(test_columns.shape[1]):
        combined = test_columns[support[0], word]
        for index in range(1, size):
            combined ^= test_columns[support[index], word]
        if combined:
            return True
    return False
