# SPDX-License-Identifier: Apache-2.0

"""Exact distances by a depth-first search of the supports connected in a Tanner graph.

The search finds light binary vectors, or symplectic vectors of Pauli operators, that satisfy sparse
checks ``H`` and are outside the row space of ``G``, or that have a given syndrome.  It grows
supports through unsatisfied checks.  It is the connected-cluster algorithm of Dumer, Kovalev, and
Pryadko, with the branch and bound of Okada and Kasai.  ``connected_cluster.md`` in
``src/qldpc/codes/distance/`` derives the search, cites its sources, and states its correctness
requirements.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import scipy.sparse

from qldpc._binary import get_kernel, pack_bits, popcount_word, trailing_zeros_word
from qldpc._compilation import always_compiled

from .._binary import assert_binary, get_nested_code_basis, multiply_binary, swap_pauli_halves
from .._information_sets import automorphism_families
from .._information_sets.basis_families import infer_weight_divisor
from .._tuning import MAX_VALUE, SearchTuning, get_tuning

# The largest scale of the weights of checks, which keeps their sums far from overflowing 64 bits.
_MAX_SCALE = 2**40

# Added to the level of a symbol for each reason that blocks it (see _search_supports).
_BLOCKED = 2**40


####################################################################################################
# search entry points


def get_distance_connected_cluster(
    sectors: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
    *,
    cutoff: int,
    upper_bound: int | None = None,
    symplectic: bool = False,
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
    tuning: SearchTuning | None = None,
) -> int:
    """Compute the minimum distance across sectors by searching connected supports.

    Each sector is a pair ``(checks, excluded)``, whose distance is the least weight of a witness:
    a vector that satisfies every check and is outside the row space of ``excluded``.  The search of
    each sector raises its weight limit one weight divisor at a time, and each limit within which
    it finds no witness proves that every witness is heavier (Section 3.6 of
    ``connected_cluster.md``).

    Args:
        sectors: The pairs ``(checks, excluded)``, as ``SupportSearch`` takes them.
        cutoff: Return as soon as a witness, or ``upper_bound``, of weight at most ``cutoff`` is
            known.  The first weight limit of a sector is its divisor, or the largest multiple of
            the divisor at most ``cutoff`` if that is larger.
        upper_bound: If not None, a cap on the result, such as the weight of a known witness outside
            ``sectors``, or one more than a weight limit of the caller.
        symplectic: Whether the sectors are symplectic (see ``SupportSearch``).
        get_automorphisms: A function that returns an array of generators of a group of
            permutations of the variables, or None.  Each row ``p`` of the array maps variable ``j``
            to ``p[j]``, and must map the vectors that satisfy the checks of each sector, and the
            row space of its excluded rows, onto themselves.  The search of each sector calls the
            function at most once, as ``automorphism_minimum_states`` of ``tuning`` says (Section
            3.5), and verifies the generators.
        tuning: The tuning of the search (see ``_tuning``), by default the active one.

    Returns:
        The least of ``upper_bound`` and the distances of the sectors, in which a sector without
        witnesses counts as one more than its number of variables; or, after an early exit, a
        weight at most ``cutoff``, which can exceed the distance.

    Raises:
        ValueError: If there are neither sectors nor an upper bound, if a sector is not valid
            (see ``SupportSearch``), or if a generator fails its verification.
        RuntimeError: If the search takes more weight limits than a correct one can.
    """
    search = ConnectedClusterSearch(
        sectors,
        cutoff=cutoff,
        upper_bound=upper_bound,
        symplectic=symplectic,
        get_automorphisms=get_automorphisms,
        tuning=tuning,
    )
    while not search.finished:
        search.advance()
    return search.result


class ConnectedClusterSearch:
    """A connected-cluster search of sectors, which advances one weight limit at a time.

    The arguments are those of ``get_distance_connected_cluster``, which advances the search until
    it finishes.  Each advance searches the sector with the lowest lower bound on its witnesses
    through its next weight limit (Section 3.6 of ``connected_cluster.md``).  The searches of the
    sectors that have witnesses are numbered in the order of those sectors.

    Raises:
        ValueError: If there are neither sectors nor an upper bound, or if a sector is not valid
            (see ``SupportSearch``).
    """

    def __init__(
        self,
        sectors: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
        *,
        cutoff: int,
        upper_bound: int | None = None,
        symplectic: bool = False,
        get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
        tuning: SearchTuning | None = None,
    ) -> None:
        if not sectors and upper_bound is None:
            raise ValueError("Require at least one sector or an upper bound")

        # Keep a search of each sector with witnesses; a sector without witnesses only bounds the
        # result.
        tuning = get_tuning() if tuning is None else tuning
        self._cutoff = cutoff
        self._best_weight = upper_bound
        self._no_distance_bounds: list[int] = []
        self._searches: list[SupportSearch] = []
        for checks, excluded in sectors:
            search = SupportSearch(
                checks,
                excluded,
                symplectic=symplectic,
                get_automorphisms=get_automorphisms,
                automorphism_minimum_states=tuning.automorphism_minimum_states,
                restrict_viable_symbols=tuning.restrict_viable_symbols,
            )
            if search.has_witnesses:
                self._searches.append(search)
            else:
                self._no_distance_bounds.append(search.num_variables + 1)

        # Start each lower bound at its divisor.  A correct search takes at most n // d limits for a
        # sector of n variables and divisor d (Section 3.6 of connected_cluster.md), so a broken
        # one fails rather than loops.
        self._lower_bounds = [search.weight_divisor for search in self._searches]
        self._max_num_limits = sum(
            search.num_variables // search.weight_divisor for search in self._searches
        )
        self._num_limits = 0

        # The weight limit, states, and number of roots of each limit without a witness, per search.
        self._limits_without_witness: list[list[tuple[int, int, int]]] = [
            [] for _ in self._searches
        ]

    @property
    def finished(self) -> bool:
        """Whether no sector has witnesses, or the best weight is at most the cutoff or each bound.

        The best weight is that of the lightest witness found, or ``upper_bound`` if that is less,
        and the bounds are the lower bounds of the sectors on the weights of their witnesses.
        """
        if not self._searches:
            return True
        return self._best_weight is not None and self._best_weight <= max(
            self._cutoff, min(self._lower_bounds)
        )

    @property
    def result(self) -> int:
        """Once finished, the result that ``get_distance_connected_cluster`` returns."""
        bounds = (self._best_weight, *self._no_distance_bounds)
        return min(bound for bound in bounds if bound is not None)

    @property
    def searches(self) -> tuple[SupportSearch, ...]:
        """The search of each sector that has witnesses."""
        return tuple(self._searches)

    @property
    def num_states(self) -> int:
        """The states that the searches have visited, over all their weight limits."""
        return sum(search.num_states for search in self._searches)

    @property
    def next_limit(self) -> tuple[int, int]:
        """The search that the next advance runs, and the weight limit through which it searches."""
        index = self._lower_bounds.index(min(self._lower_bounds))
        return index, self.get_next_limit(index)

    def get_next_limit(self, index: int) -> int:
        """The next weight limit of a search: its lower bound, or the cutoff if that is higher.

        A cutoff raises the limit to the largest multiple of the search's divisor within the cutoff.
        """
        divisor = self._searches[index].weight_divisor
        return max(self._lower_bounds[index], self._cutoff - self._cutoff % divisor)

    def get_growth(self, index: int) -> float | None:
        """The factor by which the states of a search grow per unit of its weight limit.

        Each ratio compares the states of two successive limits that found no witness, both nonzero
        and from the same number of roots, per unit of weight between the limits.  The factor is the
        geometric mean of the last two ratios, which evens out the alternation of weight limits of
        odd and even parity, or the last ratio if there is one, or None if there is none.
        """
        limits = self._limits_without_witness[index]
        ratios = [
            (states / previous_states) ** (1 / (weight - previous_weight))
            for (previous_weight, previous_states, previous_roots), (weight, states, roots) in (
                itertools.pairwise(limits)
            )
            if previous_states and states and roots == previous_roots
        ]
        if not ratios:
            return None
        return math.sqrt(ratios[-1] * ratios[-2]) if len(ratios) > 1 else ratios[-1]

    def predict_states(self, index: int, max_weight: int) -> float:
        """The states that a search is expected to visit at one later weight limit.

        The states of its last limit without a witness grow by ``get_growth`` for each unit of
        weight, and a prediction too large for a float is infinite.  Without a growth factor, the
        prediction is that of the search itself for its next limit (Section 3.5 of
        ``connected_cluster.md``).
        """
        growth = self.get_growth(index)
        if growth is None:
            return self._searches[index]._predict_states(max_weight)
        last_weight, last_states, _ = self._limits_without_witness[index][-1]
        try:
            return last_states * growth ** (max_weight - last_weight)
        except OverflowError:
            return math.inf

    def update_upper_bound(self, upper_bound: int) -> None:
        """Lower the best weight to the weight of a witness, or to a cap, found elsewhere."""
        if self._best_weight is None or upper_bound < self._best_weight:
            self._best_weight = upper_bound

    def advance(self) -> None:
        """Search the sector with the lowest lower bound through its next limit, unless finished.

        If this raises, as at a keyboard interrupt, the search can be left inconsistent, so build a
        new one to search again.

        Raises:
            ValueError: If a generator of automorphisms fails its verification.
            RuntimeError: If the search takes more weight limits than a correct one can.
        """
        if self.finished:
            return
        if self._num_limits == self._max_num_limits:
            raise RuntimeError(
                f"The connected-cluster search needed more than {self._max_num_limits} weight"
                " limits, which only a bug can cause"
            )

        # Search the sector with the lowest lower bound through its next limit, which the cutoff
        # can raise (see ``get_next_limit``).
        index, max_weight = self.next_limit
        search = self._searches[index]
        divisor = search.weight_divisor
        num_states = search.num_states
        witness = search.find_witness(max_weight)
        self._num_limits += 1

        # A limit without a witness raises the lower bound past it, and records its states for
        # get_growth.  A witness may lower the best weight.
        if witness is None:
            self._limits_without_witness[index].append(
                (max_weight, search.num_states - num_states, len(search._roots))
            )
            self._lower_bounds[index] = (max_weight // divisor + 1) * divisor
        elif self._best_weight is None or len(witness) < self._best_weight:
            self._best_weight = len(witness)


class SupportSearch:
    """Search for light words that satisfy ``checks`` and are outside the row space of ``excluded``.

    A word is a vector that satisfies every check: a binary vector ``x`` satisfies a check ``h`` if
    ``h @ x = 0``, and, if ``symplectic`` is true, the vector ``(a | b)`` of a Pauli operator
    satisfies a check that commutes with it.  The rows of ``excluded`` must satisfy every check, and
    the witnesses are the words outside their row space.  The variables are the coordinates of a
    binary vector, or the qubits, and the weight of a vector counts the variables on which it is
    nonzero.  A symbol is a variable with one of its nonzero values: symbol ``k n + j`` gives
    variable ``j`` its ``k``-th value, counting from zero, for ``n`` variables.  A binary problem
    has one value, so its symbols are its variables, and a symplectic problem has the values ``X``,
    ``Z``, and ``Y``, in that order.

    The checks need not be independent.  The search grows supports through unsatisfied checks, so it
    is fast only if ``checks`` is sparse (Sections 2 and 3.3 of ``connected_cluster.md``).  It also
    finds light vectors with a given syndrome (see ``find_vector_with_syndrome``).

    Keyword args:
        get_automorphisms: As in ``get_distance_connected_cluster``.
        automorphism_minimum_states: The search calls ``get_automorphisms`` at most once, before
            the first weight limit at which it expects to have visited this many states over all
            its limits (Section 3.5).  Zero asks before the first limit.
        restrict_viable_symbols: Whether states restrict their subtrees to viable symbols, as
            Section 3.4 says.  The search is exact either way.

    Attributes:
        num_variables: The number of variables.
        weight_divisor: A divisor of the weight of every word.
        num_states: The number of states that ``find_witness`` has visited over all its weight
            limits, without the empty state of each root.

    Raises:
        ValueError: If the matrices are not binary matrices of equal width, if a symplectic matrix
            has an odd number of columns, or if the excluded rows do not satisfy the checks.
    """

    num_variables: int
    weight_divisor: int
    num_states: int

    def __init__(
        self,
        checks: npt.ArrayLike,
        excluded: npt.ArrayLike,
        *,
        symplectic: bool = False,
        get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
        automorphism_minimum_states: int,
        restrict_viable_symbols: bool,
    ) -> None:
        # The compiled search would read invalid input without an error, so validate it here.
        assert_binary(checks, "checks")
        assert_binary(excluded, "excluded")
        checks = np.asarray(checks, dtype=np.uint8)
        excluded = np.asarray(excluded, dtype=np.uint8)
        if checks.ndim != 2 or excluded.ndim != 2 or checks.shape[1] != excluded.shape[1]:
            raise ValueError("Checks and excluded rows must be matrices with equal widths")
        if symplectic and checks.shape[1] % 2:
            raise ValueError("Symplectic matrices must have an even number of columns")

        self.num_variables = checks.shape[1] // 2 if symplectic else checks.shape[1]
        self.num_states = 0
        self._symplectic = symplectic
        self._last_two_num_states: tuple[int, int] | None = None
        self._excluded = excluded
        self._get_automorphisms = get_automorphisms
        self._automorphism_minimum_states = automorphism_minimum_states
        self._restrict_viable_symbols = restrict_viable_symbols

        # A word satisfies a symplectic check if its dot product with the swapped check vanishes.
        if symplectic:
            checks = swap_pauli_halves(checks)
        if np.any(multiply_binary(checks, excluded.T)):
            raise ValueError("The excluded rows must satisfy the checks")
        # Keep the first copy of each nonzero check, in order, since zero and duplicate checks add
        # nothing.  Map each given check to the kept check that equals it, or to -1 if it is zero;
        # ``rank_of_unique`` maps each of the sorted unique rows to its position among kept checks.
        nonzero_rows = np.flatnonzero(np.any(checks, axis=1))
        _, first_rows, unique_index_of_row = np.unique(
            checks[nonzero_rows], axis=0, return_index=True, return_inverse=True
        )
        order = np.argsort(first_rows)
        rank_of_unique = np.argsort(order)
        self._checks = checks[nonzero_rows[first_rows[order]]]
        self._kept_index_of_check = np.full(len(checks), -1, dtype=np.int64)
        self._kept_index_of_check[nonzero_rows] = rank_of_unique[unique_index_of_row.ravel()]

        # Column ``s`` of ``flips`` marks the checks that symbol ``s`` flips, and column ``s`` of
        # ``tests`` the tests that it fails.
        flips = self._get_symbol_columns(self._checks)
        sparse_flips = scipy.sparse.csr_array(flips)
        self._check_starts = sparse_flips.indptr.astype(np.int64)
        self._check_symbols = sparse_flips.indices.astype(np.int64)
        self._symbol_masks = pack_bits(np.ascontiguousarray(flips.T))
        tests = self._get_symbol_columns(self._get_tests())
        self._test_columns = pack_bits(np.ascontiguousarray(tests.T))

        # Without symmetries, every variable is a root.  Symplectic words get no divisor, since
        # their weights count qubits rather than bits.
        self._roots = np.arange(self.num_variables, dtype=np.int64)
        self._ranks = np.arange(self.num_variables, dtype=np.int64)
        self.weight_divisor = 1 if symplectic else infer_weight_divisor(get_kernel(self._checks))

        # Weigh checks in units of 1 / scale, where scale is the least common multiple of the
        # possible overlaps, capped at _MAX_SCALE (Section 3.4 of connected_cluster.md).  The
        # largest overlap is the largest number of checks of a symbol, but at least one, so that the
        # scale exists without checks.
        max_column_weight = int(np.max(np.sum(flips, axis=0), initial=1))
        scale = min(math.lcm(*range(1, max_column_weight + 1)), _MAX_SCALE)
        self._inverse_overlaps = np.zeros(max_column_weight + 1, dtype=np.int64)
        self._inverse_overlaps[1:] = scale // np.arange(1, max_column_weight + 1)

    @property
    def has_witnesses(self) -> bool:
        """Whether any word is outside the row space of ``excluded``."""
        # Without tests, the packed test columns have no blocks.
        return self._test_columns.shape[1] > 0

    def find_witness(self, max_weight: int) -> npt.NDArray[np.int64] | None:
        """Search for a witness of weight at most ``max_weight``, a weight limit of at least one.

        The search is exhaustive: it returns None only if every witness is heavier than the limit.

        Returns:
            The sorted symbols of a witness within the limit, one for each variable of its
            support, or None.  The witness need not be a lightest one.

        Raises:
            ValueError: If the weight limit is less than one, or if a generator of automorphisms
                fails its verification.
        """
        if max_weight < 1:
            raise ValueError(f"The weight limit must be at least one, not {max_weight}")
        if (
            self._get_automorphisms is not None
            and self.num_states + self._predict_states(max_weight)
            >= self._automorphism_minimum_states
        ):
            automorphisms, self._get_automorphisms = self._get_automorphisms(), None
            if automorphisms is not None:
                self._use_automorphisms(automorphisms)
        support, num_states = _search_supports(
            self._check_starts,
            self._check_symbols,
            self._symbol_masks,
            self._test_columns,
            self._roots,
            self._ranks,
            self._inverse_overlaps,
            int(min(max_weight, self.num_variables)),
            # The searches from roots ignore the starting syndrome, and do not memoize.
            np.zeros(self._symbol_masks.shape[1], dtype=np.uint64),
            False,
            self._restrict_viable_symbols,
        )
        self.num_states += num_states
        last = self._last_two_num_states[1] if self._last_two_num_states else 0
        self._last_two_num_states = (last, num_states)
        return np.sort(support) if len(support) else None

    def find_vector_with_syndrome(
        self, syndrome: npt.ArrayLike, max_weight: int
    ) -> npt.NDArray[np.int64] | None:
        """Search for a vector with a given syndrome, of weight at most ``max_weight``.

        Entry ``j`` of the syndrome belongs to row ``j`` of ``checks``, and, for a symplectic
        problem, says whether a Pauli operator anticommutes with that row.  The vectors with a
        syndrome form a coset of the words, and ``excluded`` plays no part.  The search is
        exhaustive, but need not find a lightest vector (Section 3.8 of ``connected_cluster.md``).

        Returns:
            The sorted symbols of a vector with the syndrome, or None if every such vector weighs
            more than ``max_weight``.

        Raises:
            ValueError: If the syndrome is not a binary vector with one entry for each check, or if
                ``max_weight`` is negative.
        """
        assert_binary(syndrome, "syndrome")
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if syndrome.shape != self._kept_index_of_check.shape:
            num_checks = len(self._kept_index_of_check)
            raise ValueError(f"The syndrome must have one entry for each of {num_checks} checks")
        if max_weight < 0:
            raise ValueError(f"The weight limit must be at least zero, not {max_weight}")

        # The syndrome of every vector has equal entries for equal checks, and zero for zero checks.
        indices = self._kept_index_of_check
        kept = indices >= 0
        kept_syndrome = np.zeros(len(self._checks), dtype=np.uint8)
        kept_syndrome[indices[kept]] = syndrome[kept]
        expected = np.zeros_like(syndrome)
        expected[kept] = kept_syndrome[indices[kept]]
        if np.any(syndrome != expected):
            return None
        if not np.any(kept_syndrome):
            return np.empty(0, dtype=np.int64)
        return self._find_vector_with_packed_syndrome(pack_bits(kept_syndrome[None])[0], max_weight)

    def _find_vector_with_packed_syndrome(
        self, packed_syndrome: npt.NDArray[np.uint64], max_weight: int
    ) -> npt.NDArray[np.int64] | None:
        """Search like ``find_vector_with_syndrome``, from an unvalidated nonzero packed syndrome.

        ``packed_syndrome`` has one bit for each kept check, packed as ``pack_bits`` packs a row.
        The kept checks are the rows of ``checks`` without zero rows and later copies, in order, so
        independent checks are all kept.
        """
        support, _ = _search_supports(
            self._check_starts,
            self._check_symbols,
            self._symbol_masks,
            self._test_columns,
            np.array([-1], dtype=np.int64),
            self._ranks,
            self._inverse_overlaps,
            int(min(max_weight, self.num_variables)),
            packed_syndrome,
            True,
            self._restrict_viable_symbols,
        )
        return np.sort(support) if len(support) else None

    def _predict_states(self, max_weight: int) -> int:
        """The number of states that the next weight limit, ``max_weight``, is expected to visit.

        The number is expected to grow by the ratio between those of the last two limits, or not to
        grow if the earlier of them visited no states, as before the second limit (Section 3.5 of
        ``connected_cluster.md``).  A first limit is expected to visit no states if it is at most
        the weight divisor, and otherwise, with no limits below it to predict from, enough to reach
        any ``automorphism_minimum_states`` short of ``MAX_VALUE``.
        """
        if self._last_two_num_states is None:
            return MAX_VALUE - 1 if max_weight > self.weight_divisor else 0
        previous, last = self._last_two_num_states
        return last * last // previous if previous else last

    def _get_symbol_columns(self, matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """The products of the rows of a matrix with the word of each symbol, as columns.

        The word of a symbol has its value on its variable, so for a symplectic problem the columns
        are those of the ``X`` part, of the ``Z`` part, and of their sum.
        """
        if not self._symplectic:
            return matrix
        part_x, part_z = matrix[:, : self.num_variables], matrix[:, self.num_variables :]
        return np.hstack([part_x, part_z, part_x ^ part_z])

    def _get_tests(self) -> npt.NDArray[np.uint8]:
        """Rows whose products with a word all vanish exactly when the word is excluded.

        They are a basis of the kernel of ``excluded`` modulo the checks, whose halves are already
        swapped for a symplectic problem (Section 3.6 of ``connected_cluster.md``).
        """
        basis, labels = get_nested_code_basis(get_kernel(self._excluded), self._checks)
        return basis[np.any(labels, axis=1)]

    def _use_automorphisms(self, automorphisms: npt.NDArray[np.int_]) -> None:
        """Verify automorphisms, and search only from the least variable of each of their orbits.

        The variables are ranked orbit by orbit, and a permutation of the qubits of a symplectic
        problem moves both halves of each word (Sections 3.5 and 3.7 of ``connected_cluster.md``).

        Raises:
            ValueError: If a generator does not map the words, or the row space of the excluded
                rows, onto themselves.
        """
        basis, labels = get_nested_code_basis(get_kernel(self._checks), self._excluded)
        lifted = (
            np.hstack([automorphisms, automorphisms + self.num_variables])
            if self._symplectic
            else automorphisms
        )
        automorphism_families.verify_automorphisms(basis, labels, lifted)
        orbits = automorphism_families.get_orbits(automorphisms, self.num_variables)
        self._roots = np.unique(orbits, return_index=True)[1].astype(np.int64)
        self._ranks = np.empty(self.num_variables, dtype=np.int64)
        self._ranks[np.argsort(orbits, kind="stable")] = np.arange(self.num_variables)


####################################################################################################
# native search


@always_compiled
def _search_supports(
    check_starts: npt.NDArray[np.int64],
    check_symbols: npt.NDArray[np.int64],
    symbol_masks: npt.NDArray[np.uint64],
    test_columns: npt.NDArray[np.uint64],
    roots: npt.NDArray[np.int64],
    ranks: npt.NDArray[np.int64],
    inverse_overlaps: npt.NDArray[np.int64],
    max_weight: int,
    start_syndrome: npt.NDArray[np.uint64],
    memoize: bool,
    restrict_viable_symbols: bool,
) -> tuple[npt.NDArray[np.int64], int]:
    """Search the supports that grow from each root through unsatisfied checks.

    This is the depth-first search of Sections 3.3, 3.4, and 3.7 of ``connected_cluster.md``, in
    which symbol ``s`` gives variable ``s % n`` the value ``s // n``, for ``n = len(ranks)``.  Check
    ``c`` is flipped by the symbols ``check_symbols[check_starts[c] : check_starts[c + 1]]``, row
    ``s`` of ``symbol_masks`` packs the checks that symbol ``s`` flips, and row ``s`` of
    ``test_columns`` packs the tests that its word fails.  Entry ``k`` of ``inverse_overlaps`` is
    ``1 / k`` in units of ``1 / scale``, rounded down, so entry 1 is the scale.  The search from a
    root uses only variables of higher rank.  A root of -1 starts a syndrome query (Section 3.8)
    from ``start_syndrome``, which is not empty, is packed as a row of ``symbol_masks``, and is
    ignored by the other roots.  With ``memoize``, each state computes the overlap and repair of
    each symbol at most once.  ``restrict_viable_symbols`` is as for ``_get_branches``.

    Returns:
        The symbols of a witness, or of a vector with the syndrome, of weight at most
        ``max_weight``, or an empty array; and the number of states visited, without the empty
        state of each root.
    """
    # Size the stack, and the branches of a state, which include the values of a root.
    num_symbols, num_blocks = symbol_masks.shape
    num_variables = len(ranks)
    num_values = num_symbols // num_variables
    max_column_weight = len(inverse_overlaps) - 1
    max_branches = num_values
    for check in range(len(check_starts) - 1):
        max_branches = max(max_branches, check_starts[check + 1] - check_starts[check])

    # A symbol is available if its level is the number of states in the stack that restrict their
    # subtrees, which stays below _BLOCKED.  The level counts those that allow the symbol, plus
    # _BLOCKED for each reason that blocks it: a state in the stack tried it, another value of its
    # variable is the current branch of a state in the stack, or its variable ranks below the root.
    levels = np.zeros(num_symbols, dtype=np.int64)
    num_restrictions = 0

    # The state at each depth has a syndrome, branches, and the cursor of its next branch.  A state
    # that restricts its subtree sets restricts[depth], since it may allow no symbols, and allows
    # allowances[allowance_starts[depth] : allowance_starts[depth + 1]].  Room for every state to
    # allow every symbol would be large under a large cutoff, so allowances grows as needed.
    syndromes = np.zeros((max_weight + 1, num_blocks), dtype=np.uint64)
    support = np.empty(max_weight + 1, dtype=np.int64)
    branches = np.empty((max_weight + 1, max_branches), dtype=np.int64)
    num_branches = np.zeros(max_weight + 1, dtype=np.int64)
    cursors = np.zeros(max_weight + 1, dtype=np.int64)
    restricts = np.zeros(max_weight + 1, dtype=np.bool_)
    allowance_starts = np.zeros(max_weight + 2, dtype=np.int64)
    allowances = np.empty(num_symbols, dtype=np.int64)

    # Scratch space for _get_branches, whose docstring describes ``memos``.  Each call gets its own
    # stamp, the number of calls so far, including itself, so the zeros of ``memos`` match none.
    check_weights = np.zeros(len(check_starts) - 1, dtype=np.int64)
    branch_repairs = np.empty(max_branches, dtype=np.int64)
    memos = np.zeros((4, num_symbols if memoize else 0), dtype=np.int64)
    stamp = 0

    num_states = 0
    for root in roots:
        # The empty state has an empty syndrome, and no restriction, unless a query starts it.
        cursors[0] = 0
        depth = 0
        syndromes[0] = 0
        restricts[0] = False
        allowance_starts[1] = allowance_starts[0]
        if root < 0:
            # Branch on an unsatisfied check of the starting syndrome, as a state does.
            syndromes[0] = start_syndrome
            stamp += 1
            num_branches[0] = _get_branches(
                check_starts,
                check_symbols,
                symbol_masks,
                levels,
                num_restrictions,
                syndromes,
                branches,
                restricts,
                allowance_starts,
                allowances,
                0,
                max_weight,
                inverse_overlaps,
                check_weights,
                branch_repairs,
                memos,
                stamp,
                restrict_viable_symbols,
            )
            num_restrictions += restricts[0]
        else:
            # Block the variables of lower rank than the root, and branch on its values.
            for variable in range(num_variables):
                if ranks[variable] < ranks[root]:
                    for value in range(num_values):
                        levels[value * num_variables + variable] += _BLOCKED
            for value in range(num_values):
                branches[0, value] = value * num_variables + root
            num_branches[0] = num_values
        while depth >= 0:
            # Back at a state from a branch, unblock the other values of the variable of the branch.
            if cursors[depth]:
                _shift_other_values(
                    levels, branches[depth, cursors[depth] - 1], num_variables, -_BLOCKED
                )

            # Leave the state once it has tried every branch, and release the symbols that it tried
            # or did not allow.
            if cursors[depth] == num_branches[depth]:
                for index in range(cursors[depth]):
                    levels[branches[depth, index]] -= _BLOCKED
                if restricts[depth]:
                    for index in range(allowance_starts[depth], allowance_starts[depth + 1]):
                        levels[allowances[index]] -= 1
                    num_restrictions -= 1
                depth -= 1
                continue

            # Take the next branch, and skip its state if it cannot meet the weight limit, since a
            # symbol flips at most max_column_weight unsatisfied checks (b1 in Section 3.4).
            symbol = branches[depth, cursors[depth]]
            cursors[depth] += 1
            levels[symbol] += _BLOCKED
            _shift_other_values(levels, symbol, num_variables, _BLOCKED)
            support[depth] = symbol
            num_unsatisfied = _xor_and_count(syndromes, depth, symbol_masks, symbol)
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
                if root < 0 or _is_outside_excluded(test_columns, support, depth):
                    return support[:depth].copy(), num_states
                continue

            # Make room for the symbols that this state may allow, each at most once, and choose its
            # branches.
            if allowance_starts[depth] + num_symbols > len(allowances):
                grown = np.empty(2 * len(allowances) + num_symbols, dtype=np.int64)
                grown[: allowance_starts[depth]] = allowances[: allowance_starts[depth]]
                allowances = grown
            stamp += 1
            num_branches[depth] = _get_branches(
                check_starts,
                check_symbols,
                symbol_masks,
                levels,
                num_restrictions,
                syndromes,
                branches,
                restricts,
                allowance_starts,
                allowances,
                depth,
                max_weight - depth,
                inverse_overlaps,
                check_weights,
                branch_repairs,
                memos,
                stamp,
                restrict_viable_symbols,
            )
            num_restrictions += restricts[depth]

        # Unblock the variables of lower rank than the root.
        if root >= 0:
            for variable in range(num_variables):
                if ranks[variable] < ranks[root]:
                    for value in range(num_values):
                        levels[value * num_variables + variable] -= _BLOCKED
    return np.empty(0, dtype=np.int64), num_states


@always_compiled
def _get_branches(
    check_starts: npt.NDArray[np.int64],
    check_symbols: npt.NDArray[np.int64],
    symbol_masks: npt.NDArray[np.uint64],
    levels: npt.NDArray[np.int64],
    available_level: int,
    syndromes: npt.NDArray[np.uint64],
    branches: npt.NDArray[np.int64],
    restricts: npt.NDArray[np.bool_],
    allowance_starts: npt.NDArray[np.int64],
    allowances: npt.NDArray[np.int64],
    depth: int,
    slack: int,
    inverse_overlaps: npt.NDArray[np.int64],
    check_weights: npt.NDArray[np.int64],
    branch_repairs: npt.NDArray[np.int64],
    memos: npt.NDArray[np.int64],
    stamp: int,
    restrict_viable_symbols: bool,
) -> int:
    """Choose the branches of the state at ``depth``, whose syndrome is not empty, and count them.

    A symbol is available if its level is ``available_level``.  The support can take ``slack`` more
    variables.  The weights of the unsatisfied checks are in units of ``1 / inverse_overlaps[1]``.
    Section 3.4 of ``connected_cluster.md`` says when the state is pruned, and when it allows only
    viable symbols in its subtree, which it does only if ``restrict_viable_symbols``.  A state that
    restricts its subtree raises the levels of the symbols that it allows, lists them in
    ``allowances`` from ``allowance_starts[depth]``, and sets ``restricts[depth]``.  The branches
    are the available symbols of the unsatisfied check with the fewest, after any restriction.  They
    go to ``branches[depth]``, most repairing first, and their number is zero if the state is
    pruned.  ``check_weights`` and ``branch_repairs`` are scratch space.

    If ``memos`` has a column for each symbol, row 0 holds the stamp of the call that computed the
    overlap of each symbol, row 1 that overlap, and rows 2 and 3 the same for repairs.  ``stamp`` is
    the stamp of this call, which no other call uses.
    """
    num_blocks = syndromes.shape[1]
    scale = inverse_overlaps[1]
    memoize = memos.shape[1] > 0
    overlap_stamps, overlaps, repair_stamps, repairs = memos[0], memos[1], memos[2], memos[3]

    # Weigh each unsatisfied check, and choose the one with the fewest available symbols, starting
    # from a count above any possible count.
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
                symbol = check_symbols[position]
                if levels[symbol] == available_level:
                    if not memoize:
                        overlap = _get_overlap(syndromes, depth, symbol_masks, symbol)
                    elif overlap_stamps[symbol] != stamp:
                        overlap = _get_overlap(syndromes, depth, symbol_masks, symbol)
                        overlap_stamps[symbol] = stamp
                        overlaps[symbol] = overlap
                    else:
                        overlap = overlaps[symbol]
                    largest_overlap = max(largest_overlap, overlap)
                    count += 1
            if count == 0:
                return 0
            check_weights[check] = inverse_overlaps[largest_overlap]
            total_weight += check_weights[check]
            # The total weight only grows, so prune the state as soon as it exceeds the slack.
            if total_weight > slack * scale:
                return 0
            if count < fewest:
                branch_check, fewest = check, count

    # Symbols that flip no unsatisfied check repair nothing, so they are viable exactly when every
    # symbol is.  If they are not, and restrict_viable_symbols is set, allow only the viable symbols
    # of the unsatisfied checks, and count them in each check.
    if restrict_viable_symbols and total_weight > (slack - 1) * scale:
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
                    symbol = check_symbols[position]
                    if levels[symbol] == available_level:
                        if not memoize:
                            repair = _get_repair(
                                syndromes, depth, symbol_masks, symbol, check_weights
                            )
                        elif repair_stamps[symbol] != stamp:
                            repair = _get_repair(
                                syndromes, depth, symbol_masks, symbol, check_weights
                            )
                            repair_stamps[symbol] = stamp
                            repairs[symbol] = repair
                        else:
                            repair = repairs[symbol]
                        if total_weight - repair <= (slack - 1) * scale:
                            levels[symbol] += 1
                            allowances[allowance_end] = symbol
                            allowance_end += 1
                    count += levels[symbol] == available_level + 1
                if count == 0:
                    allowance_starts[depth + 1] = allowance_end
                    return 0
                if count < fewest:
                    branch_check, fewest = check, count
        allowance_starts[depth + 1] = allowance_end
        # The allowed symbols are now the available ones.
        available_level += 1

    # Branch on the available symbols of the chosen check, most repairing first (Section 3.3 of
    # connected_cluster.md).
    num_branches = 0
    for position in range(check_starts[branch_check], check_starts[branch_check + 1]):
        symbol = check_symbols[position]
        if levels[symbol] != available_level:
            continue
        if not memoize:
            repair = _get_repair(syndromes, depth, symbol_masks, symbol, check_weights)
        elif repair_stamps[symbol] != stamp:
            repair = _get_repair(syndromes, depth, symbol_masks, symbol, check_weights)
            repair_stamps[symbol] = stamp
            repairs[symbol] = repair
        else:
            repair = repairs[symbol]
        index = num_branches
        while index > 0 and branch_repairs[index - 1] < repair:
            branches[depth, index] = branches[depth, index - 1]
            branch_repairs[index] = branch_repairs[index - 1]
            index -= 1
        branches[depth, index] = symbol
        branch_repairs[index] = repair
        num_branches += 1
    return num_branches


@always_compiled(inline=True)
def _shift_other_values(
    levels: npt.NDArray[np.int64], symbol: int, num_variables: int, change: int
) -> None:
    """Add ``change`` to the level of every other symbol on the variable of ``symbol``."""
    num_symbols = len(levels)
    for value in range(1, num_symbols // num_variables):
        other = symbol + value * num_variables
        levels[other - num_symbols if other >= num_symbols else other] += change


@always_compiled(inline=True)
def _get_overlap(
    syndromes: npt.NDArray[np.uint64],
    depth: int,
    symbol_masks: npt.NDArray[np.uint64],
    symbol: int,
) -> int:
    """The number of unsatisfied checks that a symbol flips, in the syndrome at a depth."""
    count = 0
    for block in range(syndromes.shape[1]):
        count += int(popcount_word(syndromes[depth, block] & symbol_masks[symbol, block]))
    return count


@always_compiled(inline=True)
def _get_repair(
    syndromes: npt.NDArray[np.uint64],
    depth: int,
    symbol_masks: npt.NDArray[np.uint64],
    symbol: int,
    check_weights: npt.NDArray[np.int64],
) -> int:
    """The total weight of the unsatisfied checks that a symbol flips, at a depth."""
    repair = 0
    for block in range(syndromes.shape[1]):
        bits = syndromes[depth, block] & symbol_masks[symbol, block]
        while bits:
            repair += check_weights[64 * block + int(trailing_zeros_word(bits))]
            bits &= bits - np.uint64(1)
    return repair


@always_compiled(inline=True)
def _xor_and_count(
    syndromes: npt.NDArray[np.uint64],
    depth: int,
    symbol_masks: npt.NDArray[np.uint64],
    symbol: int,
) -> int:
    """Set the syndrome at ``depth + 1`` to that at ``depth`` plus a symbol, and return its size.

    The size of a syndrome is its number of unsatisfied checks.
    """
    count = 0
    for block in range(syndromes.shape[1]):
        syndromes[depth + 1, block] = syndromes[depth, block] ^ symbol_masks[symbol, block]
        count += int(popcount_word(syndromes[depth + 1, block]))
    return count


@always_compiled
def _is_outside_excluded(
    test_columns: npt.NDArray[np.uint64], support: npt.NDArray[np.int64], size: int
) -> bool:
    """Whether the word of the symbols ``support[:size]`` fails a test."""
    for block in range(test_columns.shape[1]):
        combined = test_columns[support[0], block]
        for index in range(1, size):
            combined ^= test_columns[support[index], block]
        if combined:
            return True
    return False
