# SPDX-License-Identifier: Apache-2.0

"""Exact distances by a depth-first search of the supports connected in a Tanner graph.

The search finds light binary vectors, or symplectic vectors of Pauli operators, that satisfy sparse
checks ``H`` and are outside the row space of ``G``, or that have a given syndrome.  It grows
supports through unsatisfied checks.  It is the connected-cluster algorithm of Dumer, Kovalev, and
Pryadko, with the branch and bound of Okada and Kasai.  ``connected_cluster.md`` in
``src/qldpc/codes/distance/`` derives the search, cites its sources, and states its correctness
requirements.  These include requirements on threads, since several threads run ``_search_supports``
at once, to search a costly weight limit or to answer several queries (Sections 3.11 and 5).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import scipy.sparse

from qldpc._binary import (
    get_kernel,
    multiply_binary,
    pack_bits,
    popcount_word,
    trailing_zeros_word,
    unpack_bits,
)
from qldpc._compilation import always_compiled

from .. import _threads
from .._code_matrices import (
    assert_binary,
    get_nested_code_basis,
    swap_pauli_halves,
    symplectic_to_hamming,
)
from .._information_sets import automorphism_families
from .._information_sets.basis_families import infer_weight_divisor
from .._tuning import MAX_VALUE, SearchTuning, get_tuning
from .sparse_checks import get_light_pair_sums, get_row_weights

# The largest scale of the weights of checks, which keeps their sums far from overflowing 64 bits.
_MAX_SCALE = 2**40

# Added to the level of a symbol for each reason that blocks it (see _search_supports).
_BLOCKED = 2**40

# The tables of overlap cuts pack an index ``i`` and a load ``l``, which is 1 or 2, into the entry
# ``_LOAD_BASE * i + l`` (see _get_overlap_cuts), so that each table is one array, which keeps the
# arguments of the compiled search fewer.
_LOAD_BASE = 4

# Numba compiles a version of _search_supports for each number of 64-bit words of a syndrome up to
# this one, and one version for all wider syndromes (see _get_fixed_words).  In each fixed version,
# the loops over the words have a fixed length that the compiler unrolls.  A version is compiled,
# and cached, only when a search first needs it, but each takes seconds to compile, so this bound
# keeps them few.
_MAX_FIXED_WORDS = 8


class _CheckTables(NamedTuple):
    """The tables of the checks and tests of a search, the first arguments of ``_search_supports``.

    The docstring of ``_search_supports`` describes them.
    """

    check_starts: npt.NDArray[np.int64]
    check_symbols: npt.NDArray[np.int64]
    symbol_masks: npt.NDArray[np.uint64]
    test_columns: npt.NDArray[np.uint64]


# The tables of the overlap cuts of a search, as ``_get_overlap_cuts`` returns them.
_CutTables = tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]


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
    through its next weight limit (Section 3.6 of ``connected_cluster.md``), on the threads of the
    tuning if the limit is costly, with the same results and counts of states as a search on one
    thread (Section 3.11).  The searches of the sectors that have witnesses are numbered in the
    order of those sectors.

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
        # result.  A weight limit runs on threads if it passes the gate of
        # ``connected_cluster_split_shift``, and runs in chunks of ``2**search_chunk_exponent``
        # states (Sections 3.6 and 3.11 of connected_cluster.md).
        tuning = get_tuning() if tuning is None else tuning
        num_threads = _threads.get_num_threads(tuning.num_threads)
        chunk_states = 1 << tuning.search_chunk_exponent
        split_minimum_states = (num_threads * chunk_states) >> tuning.connected_cluster_split_shift
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
                stabilizer_overlap_cuts=tuning.stabilizer_overlap_cuts,
                pair_sum_checks=tuning.pair_sum_checks,
                num_threads=num_threads,
                split_minimum_states=split_minimum_states,
                chunk_states=chunk_states,
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
        """The index of the search that the next advance runs, and that advance's weight limit."""
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

        A ratio compares the states of two successive limits without a witness, both nonzero and
        from the same number of roots, per unit of weight between them.  The factor is the geometric
        mean of the last two ratios, which evens out the alternation between limits of odd and even
        weight.  With only one ratio, the factor is that ratio, and with none it is None.
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
        prediction for any later limit is the search's own prediction for its next limit (Section
        3.5 of ``connected_cluster.md``).
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
        stabilizer_overlap_cuts: Whether the search forbids symbols that would make its support
            meet a known word in more than about half of it, as Section 3.4 says.  The known words
            of ``find_witness`` are the rows of ``excluded``.  Those of syndrome queries are these
            rows and the words that ``use_query_overlap_words`` adds.  The search is exact either
            way.
        pair_sum_checks: Whether the search adds, as redundant checks, the light sums of pairs
            of checks that ``get_light_pair_sums`` of ``sparse_checks.py`` finds.  With
            ``stabilizer_overlap_cuts``, ``find_witness`` also uses as known words the light sums
            of pairs of those excluded rows that are no heavier than the heaviest check (Section
            3.4).  The search is exact either way, and maps the syndrome of each query to the
            added checks.
        num_threads: The number of threads of ``find_witness``.  Syndrome queries run on the
            calling thread.
        split_minimum_states: ``find_witness`` splits a weight limit across threads only if it
            predicts that the limit visits more states than this.
        chunk_states: ``find_witness`` searches in compiled calls that each count at most this
            many states, so that it takes keyboard interrupts between them, and threads can stop a
            work item between them (Sections 3.6 and 3.11).  Must be at least one.

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
        stabilizer_overlap_cuts: bool,
        pair_sum_checks: bool,
        num_threads: int = 1,
        split_minimum_states: int = 0,
        chunk_states: int = MAX_VALUE,
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
        self._stabilizer_overlap_cuts = stabilizer_overlap_cuts
        self._num_threads = num_threads
        self._split_minimum_states = split_minimum_states
        self._chunk_states = chunk_states

        # A word satisfies a symplectic check if its dot product with the swapped check vanishes.
        if symplectic:
            checks = swap_pauli_halves(checks)
        if np.any(multiply_binary(checks, excluded.T)):
            raise ValueError("The excluded rows must satisfy the checks")
        # Keep the first copy of each nonzero check, in order, since zero and duplicate checks add
        # nothing.  Map each given check to the kept check that equals it, or to -1 if it is zero;
        # ``position_of_unique`` maps each of the sorted unique rows to its position among kept
        # checks.
        nonzero_rows = np.flatnonzero(np.any(checks, axis=1))
        first_rows, unique_index_of_row = _get_unique_rows(checks[nonzero_rows])
        order = np.argsort(first_rows)
        position_of_unique = np.argsort(order)
        self._checks = checks[nonzero_rows[first_rows[order]]]
        self._kept_index_of_check = np.full(len(checks), -1, dtype=np.int64)
        self._kept_index_of_check[nonzero_rows] = position_of_unique[unique_index_of_row]

        # The search also branches on and weighs the sums of the pairs of kept checks in
        # ``summed_pairs``, which follow the kept checks (Section 3.4 of connected_cluster.md).
        searched_checks = self._checks
        self._summed_pairs = np.zeros((0, 2), dtype=np.int64)
        if pair_sum_checks:
            sums, self._summed_pairs = get_light_pair_sums(self._checks, symplectic=symplectic)
            searched_checks = np.vstack([self._checks, sums])

        # Column ``s`` of ``flips`` marks the checks that symbol ``s`` flips, and column ``s`` of
        # ``tests`` the tests that it fails.
        flips = self._get_symbol_columns(searched_checks)
        sparse_flips = scipy.sparse.csr_array(flips)
        tests = self._get_symbol_columns(self._get_tests())
        self._check_tables = _CheckTables(
            sparse_flips.indptr.astype(np.int64),
            sparse_flips.indices.astype(np.int64),
            pack_bits(np.ascontiguousarray(flips.T)),
            pack_bits(np.ascontiguousarray(tests.T)),
        )

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

        # The excluded rows are known words for the overlap cuts, and queries may have more.  With
        # pair sums, so are the light sums of pairs of those excluded rows that are no heavier than
        # the heaviest check (Section 3.4 of connected_cluster.md).
        self._known_words = excluded
        if pair_sum_checks and stabilizer_overlap_cuts:
            max_check_weight = get_row_weights(self._checks, symplectic=symplectic).max(initial=0)
            word_weights = get_row_weights(excluded, symplectic=symplectic)
            light_words = excluded[word_weights <= max_check_weight]
            sums, _ = get_light_pair_sums(light_words, symplectic=symplectic)
            self._known_words = np.vstack([excluded, sums])
        self._query_words = np.zeros((0, checks.shape[1]), dtype=np.uint8)
        # find_witness tabulates its cuts at its first weight limit.  use_query_overlap_words and
        # tabulate_query_cuts tabulate the cuts of queries; otherwise find_vector_with_syndrome does
        # so at its first query.  Each search tabulates only the kind it uses.
        self._cuts: _CutTables | None = None
        self._query_cuts: _CutTables | None = None

    @property
    def has_witnesses(self) -> bool:
        """Whether any word is outside the row space of ``excluded``."""
        # Without tests, the packed test columns have no blocks.
        return self._check_tables.test_columns.shape[1] > 0

    def find_witness(self, max_weight: int) -> npt.NDArray[np.int64] | None:
        """Search for a witness of weight at most ``max_weight``, a weight limit of at least one.

        The search is exhaustive: it returns None only if every witness is heavier than the limit.
        A limit that is predicted to visit more than ``split_minimum_states`` states runs on
        ``num_threads`` threads, which give the same witness and count of states as a search on one
        thread (Section 3.11 of ``connected_cluster.md``).

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
        # Tabulate the cuts of the known words under the current ranks, if they are not tabulated,
        # before any thread starts, so that threads only read the search.
        if self._cuts is None:
            words = self._known_words
            if not self._stabilizer_overlap_cuts:
                words = words[:0]
            self._cuts = _get_overlap_cuts(words, self._ranks, self._symplectic)
        cuts = self._cuts
        limit = int(min(max_weight, self.num_variables))

        def search(
            roots: npt.NDArray[np.int64],
            paths: npt.NDArray[np.int64],
            frontier_depth: int,
            resume: npt.NDArray[np.int64],
            ends: npt.NDArray[np.int64],
            max_states: int,
        ) -> tuple[
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
        ]:
            """Search the work items of this limit, as ``_search_supports`` does."""
            return _search_supports(
                *self._check_tables,
                roots,
                paths,
                frontier_depth,
                self._ranks,
                self._inverse_overlaps,
                limit,
                resume,
                ends,
                max_states,
                # The searches from roots ignore the starting syndrome, and do not memoize.
                np.zeros(self._check_tables.symbol_masks.shape[1], dtype=np.uint64),
                _get_fixed_words(self._check_tables.symbol_masks.shape[1]),
                False,
                self._restrict_viable_symbols,
                *cuts,
            )

        predicted_states = self._predict_states(max_weight)
        if self._num_threads > 1 and predicted_states > self._split_minimum_states:
            support, num_states = _search_on_threads(
                search,
                self._roots,
                limit,
                self._num_threads,
                self._chunk_states,
                predicted_states,
            )
        else:
            support, num_states = _search_in_chunks(search, self._roots, self._chunk_states)
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
        syndrome form a coset of the words, and every vector of the coset counts, even if
        ``excluded`` has rows.  The search is exhaustive, but need not find a lightest vector
        (Section 3.8 of ``connected_cluster.md``).  The overlap cuts of the rows of ``excluded``
        and of the words that ``use_query_overlap_words`` adds can change which vector the search
        returns, but not whether it finds one within ``max_weight``.

        Coset searches call ``find_vector_with_packed_syndrome`` instead, which skips validation.
        Unless ``use_query_overlap_words`` or ``tabulate_query_cuts`` has tabulated the cuts of
        queries, the first call tabulates them and writes to the search, so no other query of the
        search may run during that call.

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
        if self._query_cuts is None:
            self.tabulate_query_cuts()
        packed_syndrome = pack_bits(kept_syndrome[None])[0]
        return self.find_vector_with_packed_syndrome(packed_syndrome, max_weight)[0]

    def find_vector_with_packed_syndrome(
        self, packed_syndrome: npt.NDArray[np.uint64], max_weight: int
    ) -> tuple[npt.NDArray[np.int64] | None, int]:
        """Search like ``find_vector_with_syndrome``, from an unvalidated nonzero packed syndrome.

        ``packed_syndrome`` has one bit for each kept check, packed as ``pack_bits`` packs a row.
        The kept checks are the rows of ``checks`` without zero rows and later copies, in order, so
        independent checks are all kept.  Also returns the number of states that the search visited.
        The cuts of queries must be tabulated already (see ``tabulate_query_cuts``), and the query
        only reads the search, so that queries can run on threads.
        """
        assert self._query_cuts is not None
        # The syndrome of a sum of two checks is the sum of their syndromes.
        if len(self._summed_pairs):
            bits = unpack_bits(packed_syndrome[None], len(self._checks))[0]
            sums = bits[self._summed_pairs[:, 0]] ^ bits[self._summed_pairs[:, 1]]
            packed_syndrome = pack_bits(np.concatenate([bits, sums])[None])[0]
        support, item_states, _, _, _ = _search_supports(
            *self._check_tables,
            np.array([-1], dtype=np.int64),
            np.empty((1, 0), dtype=np.int64),
            0,
            self._ranks,
            self._inverse_overlaps,
            int(min(max_weight, self.num_variables)),
            _NO_PATH,
            _NO_PATH,
            MAX_VALUE,
            packed_syndrome,
            _get_fixed_words(len(packed_syndrome)),
            True,
            self._restrict_viable_symbols,
            *self._query_cuts,
        )
        return (np.sort(support) if len(support) else None), int(item_states[0])

    def use_query_overlap_words(self, words: npt.ArrayLike) -> None:
        """Add known words, such as stabilizers, to those that the cuts of syndrome queries use.

        A word is a vector that satisfies every check, in the representation of ``excluded``.  The
        cuts of syndrome queries are exact for any words (Section 3.8 of ``connected_cluster.md``),
        and help most with light ones.  ``find_witness`` does not use these words, since its cuts
        are exact only for words in the row space of ``excluded`` (Section 3.4).  The cuts do
        nothing unless ``stabilizer_overlap_cuts`` is set.  Call this method only while no query or
        search of this object runs, as the constructor of ``CosetLeaderSearch`` does, since it
        rebuilds the cuts that threads read (Section 3.11).

        Raises:
            ValueError: If the words are not a binary matrix as wide as the checks, or do not
                satisfy every check.
        """
        assert_binary(words, "words")
        words = np.asarray(words, dtype=np.uint8)
        if words.ndim != 2 or words.shape[1] != self._query_words.shape[1]:
            raise ValueError("The words must be a matrix as wide as the checks")
        if np.any(multiply_binary(self._checks, words.T)):
            raise ValueError("The words must satisfy the checks")
        self._query_words = np.vstack([self._query_words, words])
        # Tabulate the cuts now, so that the queries after this call only read the search.
        self.tabulate_query_cuts()

    def tabulate_query_cuts(self) -> None:
        """Tabulate the overlap cuts of queries under the current ranks.

        The cuts use the rows of ``excluded`` and the words that ``use_query_overlap_words`` adds,
        and there are none without ``stabilizer_overlap_cuts``.
        """
        words = np.vstack([self._excluded, self._query_words])
        if not self._stabilizer_overlap_cuts:
            words = words[:0]
        self._query_cuts = _get_overlap_cuts(words, self._ranks, self._symplectic)

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
        return symplectic_to_hamming(matrix) if self._symplectic else matrix

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
        # A cut can leave out the lowest-ranked variable of its word, which the new ranks can move,
        # so find_witness tabulates its cuts again, and tabulated cuts of queries are replaced now.
        self._cuts = None
        if self._query_cuts is not None:
            self.tabulate_query_cuts()


def _get_overlap_cuts(
    words: npt.NDArray[np.uint8], ranks: npt.NDArray[np.int64], symplectic: bool
) -> _CutTables:
    """Tabulate the overlap cuts of known words for ``_search_supports``.

    The cut of a word ``g`` limits the load of a support on ``g``, which a support passes if the
    load is at most the capacity ``|g| - 1`` (Section 3.4 of ``connected_cluster.md``).  Each symbol
    on a variable of ``g`` adds 2 to the load if it has the value of ``g`` there and 1 otherwise,
    except that the lowest-ranked variable of ``g`` adds nothing, unless the problem is binary and
    ``|g|`` is odd.  Words of weight at most one are left out.  In a symplectic problem they cut
    nothing, since their one variable ranks lowest.  In a binary problem, such a word would forbid
    only its variable, which no lightest witness or solution holds, and the search stays exact
    without it.

    Returns:
        ``symbol_starts`` and ``symbol_entries``, which list ``_LOAD_BASE c + l`` for each cut
        ``c`` to which symbol ``s`` adds the load ``l``, at ``symbol_entries[symbol_starts[s] :
        symbol_starts[s + 1]]``; ``cut_starts`` and ``cut_entries``, which list ``_LOAD_BASE s +
        l`` for the same pairs by cut; the capacity of each cut; and the symbols whose load alone
        exceeds the capacity of a cut, once for each such cut.
    """
    # Find the support of each distinct word, and the value of the word on each variable, and keep
    # the words of weight at least two.
    num_variables = len(ranks)
    num_values = 3 if symplectic else 1
    words = words[_get_unique_rows(words)[0]]
    if symplectic:
        part_x, part_z = words[:, :num_variables], words[:, num_variables:]
        supports = part_x | part_z
        word_values = np.where(part_z == 0, 0, np.where(part_x == 0, 1, 2))
    else:
        supports, word_values = words, np.zeros_like(words)
    weights = np.sum(supports, axis=1, dtype=np.int64)
    supports = supports[weights >= 2]
    word_values = word_values[weights >= 2]
    weights = weights[weights >= 2]

    # Leave out the lowest-ranked variable of each word, except of an odd word of a binary problem.
    # np.nonzero lists the entries of the supports sorted by cut, so the entries of cut ``c`` start
    # at position ``np.searchsorted(cuts, c)`` both there and after sorting by cut and then rank.
    cuts, variables = np.nonzero(supports)
    lowest = np.lexsort((ranks[variables], cuts))[np.searchsorted(cuts, np.arange(len(weights)))]
    leaves_out = np.ones(len(weights), dtype=bool) if symplectic else weights % 2 == 0
    kept = np.ones(len(cuts), dtype=bool)
    kept[lowest[leaves_out]] = False
    cuts, variables = cuts[kept], variables[kept]

    # Give every value of each variable that a cut keeps its load on the cut.
    values = np.arange(num_values)
    symbols = (values[None, :] * num_variables + variables[:, None]).ravel()
    loads = np.where(values[None, :] == word_values[cuts, variables][:, None], 2, 1).ravel()
    cuts = np.repeat(cuts, num_values)
    capacities = weights - 1

    # List the pairs of a cut and a symbol by symbol and by cut.
    by_symbol = np.argsort(symbols, kind="stable")
    symbol_starts = np.searchsorted(symbols[by_symbol], np.arange(num_values * num_variables + 1))
    cut_starts = np.searchsorted(cuts, np.arange(len(weights) + 1))
    return (
        symbol_starts.astype(np.int64),
        (_LOAD_BASE * cuts[by_symbol] + loads[by_symbol]).astype(np.int64),
        cut_starts.astype(np.int64),
        (_LOAD_BASE * symbols + loads).astype(np.int64),
        capacities,
        symbols[loads > capacities[cuts]].astype(np.int64),
    )


def _get_unique_rows(
    matrix: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The distinct rows of a binary matrix, as ``np.unique(matrix, axis=0)`` finds them.

    Returns:
        The index of the first copy of each distinct row, with the distinct rows in increasing
        order, and the position of each row among the distinct rows.
    """
    # Rows packed with their first column in the highest bit, and compared as byte strings, sort
    # as the rows do, and tens of times faster than np.unique, which compares columns one by one.
    packed = np.ascontiguousarray(np.packbits(matrix, axis=1))
    keys = packed.view(np.dtype((np.void, packed.shape[1]))).ravel()
    _, first_rows, positions = np.unique(keys, return_index=True, return_inverse=True)
    return first_rows.astype(np.int64), positions.astype(np.int64)


def _get_fixed_words(num_words: int) -> tuple[int, ...]:
    """The argument ``fixed_words`` of ``_search_supports`` for syndromes of ``num_words`` words.

    It holds a zero for each word if there are at most ``_MAX_FIXED_WORDS`` words, and is empty
    otherwise.
    """
    return (0,) * num_words if num_words <= _MAX_FIXED_WORDS else ()


####################################################################################################
# weight limits on threads

# The most items that the frontier of a limit on threads targets, which bounds the cost of its
# passes and the memory of its paths to within the branching of one depth.
_MAX_ITEMS = 2**14

# A path of no states, for calls of _search_supports that resume nothing.
_NO_PATH = np.empty(0, dtype=np.int64)

# Searches the work items of a weight limit, as ``_search_supports`` does: from roots, along paths,
# with a frontier depth, and optionally with a path and ends to resume and a budget of states.
_ItemSearch = Callable[
    [
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        int,
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        int,
    ],
    tuple[
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
        npt.NDArray[np.int64],
    ],
]


def _search_in_chunks(
    search: _ItemSearch, roots: npt.NDArray[np.int64], chunk_states: int
) -> tuple[npt.NDArray[np.int64], int]:
    """Search the subtrees of the roots of a limit on the calling thread, in chunks of states.

    Each call of ``search`` counts at most ``chunk_states`` states, and the next call goes on where
    it stopped, so the search takes keyboard interrupts between calls, and finds the witness and
    counts the states of a search in one call (Sections 3.6 and 3.11 of ``connected_cluster.md``).

    Returns:
        The witness, or an empty array, and the number of states.
    """
    num_states = 0
    paths = np.empty((len(roots), 0), dtype=np.int64)
    first, resume, ends = 0, _NO_PATH, _NO_PATH
    while True:
        support, item_states, _, resume, ends = search(
            roots[first:], paths[first:], 0, resume, ends, chunk_states
        )
        num_states += int(np.sum(item_states))
        if len(support) or not len(resume):
            return support, num_states
        first += len(item_states) - 1


def _search_on_threads(
    search: _ItemSearch,
    roots: npt.NDArray[np.int64],
    max_weight: int,
    num_threads: int,
    chunk_states: int,
    predicted_states: int,
) -> tuple[npt.NDArray[np.int64], int]:
    """Search a weight limit on threads, which split it into the subtrees of a frontier.

    The frontier deepens from the roots, one depth at a time, until it has about as many states as
    the limit is predicted to visit chunks of ``chunk_states`` states, but at least
    ``_threads.ITEMS_PER_THREAD`` for each thread and at most ``_MAX_ITEMS``, or it reaches
    ``max_weight`` or a witness, or no state is that deep.  The threads claim its states, the work
    items, one at a time in DFS order, and search each in chunks of at most ``chunk_states`` states.
    Each item before the first with a witness runs in full, and once that witness is found, no item
    after it starts and a running one stops after its chunk (Section 3.11 of
    ``connected_cluster.md``).

    Returns:
        The witness and the number of states of a search of the limit on one thread.
    """
    target = max(
        _threads.ITEMS_PER_THREAD * num_threads, min(_MAX_ITEMS, predicted_states // chunk_states)
    )

    # Each pass lists the states one deeper than the current items, which become states above the
    # frontier, apart from the roots.  ``prefixes`` counts the states above the frontier that come
    # before each item in DFS order, and ``above`` counts them all.  A witness above the frontier,
    # with the states through it, ends the deepening, and comes after every listed item.
    item_roots = roots
    paths = np.empty((len(roots), 0), dtype=np.int64)
    prefixes = np.zeros(len(roots), dtype=np.int64)
    above = 0
    witness = np.empty(0, dtype=np.int64)
    best = item_roots, paths, prefixes, above
    while len(item_roots) < target and paths.shape[1] < max_weight:
        support, item_states, listed, _, _ = search(
            item_roots, paths, paths.shape[1] + 1, _NO_PATH, _NO_PATH, MAX_VALUE
        )
        if len(support):
            witness = support
            above = int(prefixes[len(item_states) - 1] + np.sum(item_states))
        else:
            above += int(np.sum(item_states))
        if not len(listed) and not len(support):
            return witness, above
        parents = listed[:, 0]
        item_roots, paths = item_roots[parents], np.ascontiguousarray(listed[:, 1:])
        # Each old item at depth one or more is one state now above the frontier, but the empty
        # states of the roots are not counted.
        prefixes = prefixes[parents] + (parents + 1 if paths.shape[1] > 1 else 0)
        if len(support) or len(item_roots) > len(best[0]):
            best = item_roots, paths, prefixes, above
        if len(support):
            break
    item_roots, paths, prefixes, above = best

    # Search the items one at a time, each in chunks, and record the states and the witness of each
    # item.
    num_items = len(item_roots)
    counts = np.zeros(num_items, dtype=np.int64)
    witnesses: dict[int, npt.NDArray[np.int64]] = {}

    def start_items(needed: Callable[[int], bool]) -> Callable[[int], bool]:
        """Return how a worker searches an item."""

        def run_item(item: int) -> bool:
            """Search an item in chunks until it ends, and return whether it has a witness.

            Between chunks, the item stops once it is no longer ``needed``: once an earlier item has
            a witness, or once the threads stop.
            """
            resume, ends = _NO_PATH, _NO_PATH
            while True:
                support, item_states, _, resume, ends = search(
                    item_roots[item : item + 1],
                    paths[item : item + 1],
                    0,
                    resume,
                    ends,
                    chunk_states,
                )
                counts[item] += item_states[0]
                if len(support):
                    witnesses[item] = support
                    return True
                if not len(resume) or not needed(item):
                    return False

        return run_item

    # Compile the search before the threads call it: a search of no items returns at once.
    search(item_roots[:0], paths[:0], 0, _NO_PATH, _NO_PATH, MAX_VALUE)
    first = _threads.run_items_until_first(num_items, num_threads, start_items, single_items=True)
    if first < num_items:
        return witnesses[first], int(prefixes[first] + np.sum(counts[: first + 1]))
    return witness, int(above + np.sum(counts))


####################################################################################################
# native search


@always_compiled(nogil=True)
def _search_supports(
    check_starts: npt.NDArray[np.int64],
    check_symbols: npt.NDArray[np.int64],
    symbol_masks: npt.NDArray[np.uint64],
    test_columns: npt.NDArray[np.uint64],
    roots: npt.NDArray[np.int64],
    paths: npt.NDArray[np.int64],
    frontier_depth: int,
    ranks: npt.NDArray[np.int64],
    inverse_overlaps: npt.NDArray[np.int64],
    max_weight: int,
    resume: npt.NDArray[np.int64],
    ends: npt.NDArray[np.int64],
    max_states: int,
    start_syndrome: npt.NDArray[np.uint64],
    fixed_words: tuple[int, ...],
    memoize: bool,
    restrict_viable_symbols: bool,
    cut_symbol_starts: npt.NDArray[np.int64],
    cut_symbol_entries: npt.NDArray[np.int64],
    cut_starts: npt.NDArray[np.int64],
    cut_entries: npt.NDArray[np.int64],
    cut_capacities: npt.NDArray[np.int64],
    cut_blocked_symbols: npt.NDArray[np.int64],
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    """Search the supports that grow from each root through unsatisfied checks.

    This is the depth-first search of Sections 3.3, 3.4, and 3.7 of ``connected_cluster.md``, in
    which symbol ``s`` gives variable ``s % n`` the value ``s // n``, for ``n = len(ranks)``.  Check
    ``c`` is flipped by the symbols ``check_symbols[check_starts[c] : check_starts[c + 1]]``, row
    ``s`` of ``symbol_masks`` packs the checks that symbol ``s`` flips, and row ``s`` of
    ``test_columns`` packs the tests that its word fails.  Entry ``k`` of ``inverse_overlaps`` is
    ``1 / k`` in units of ``1 / scale``, rounded down, so entry 1 is the scale.  The search from a
    root uses only variables of higher rank.  A root of -1 starts a syndrome query (Section 3.8)
    from ``start_syndrome``, which is not empty, is packed as a row of ``symbol_masks``, and is
    ignored by the other roots.  ``fixed_words`` is empty, or has an entry for each word of a row of
    ``symbol_masks``, as ``_get_fixed_words`` makes it, and numba compiles a version for each of its
    lengths (see ``_MAX_FIXED_WORDS``).  With ``memoize``, each state computes the overlap and
    repair of each symbol at most once.  ``restrict_viable_symbols`` is as for ``_get_branches``.
    The remaining arguments tabulate the overlap cuts, as ``_get_overlap_cuts`` returns them.

    The search runs work items in turn, as threads do (Section 3.11).  Item ``i`` is the subtree of
    a state at depth ``P = paths.shape[1]``, which holds ``P`` symbols: from root ``roots[i]``,
    state ``d < P`` of its path takes its branch ``paths[i, d]``, counting from zero.  The search
    replays the path, in which each earlier sibling only stays tried, and stops the item when it
    goes back above depth ``P``.  It counts only the states at depth ``P`` or more.  With ``P = 0``,
    the items are whole roots, as in a search of one thread.  A ``frontier_depth`` of ``P + 1``
    enters no state at that depth, and lists each such state that it would enter instead, in DFS
    order, by the index of its item and its branch at each depth below it.  A ``frontier_depth`` of
    zero lists nothing.  A path must lead to a state that the search enters.

    The search stops early once it has counted ``max_states`` states, which must be at least one, at
    the next state that does not replay a path, just before that state takes its next branch.  It
    then returns its stack as a path, ``resume``: the branch that each state on it takes, and, for
    its last state, the next branch.  It also returns, for each state of that path, its end in
    ``ends``: the branch before which the state leaves.  Given these, and the items from the one at
    which it stopped, the search replays ``resume`` instead of the path of its first item, counts
    no state that the replay enters, and goes on exactly where it stopped, so that a search in
    several calls visits and counts the states of a search in one call (Section 3.11).  A caller
    may split a call that resumes only the item at which the search stopped into two calls of that
    item, at a branch ``b`` above ``resume[d]`` and below ``ends[d]``.  The first call takes
    the ends ``resume[:d] + 1`` followed by ``b`` and the old ends below depth ``d``, so that it
    leaves the states above depth ``d`` after their branches on the path.  The second takes the
    path ``resume[:d]`` followed by ``b``, and the old ``ends[: d + 1]``, so that it searches what
    comes after the first call in DFS order.  The witness of the call that was split is that of the
    first call if it has one, and otherwise that of the second, and the call counts the states of
    the first, and also those of the second if the first has no witness.

    Returns:
        The symbols of a witness, or of a vector with the syndrome, of weight at most
        ``max_weight``, or an empty array; the number of states, never including the empty state
        of a root, that the search counted in each item, through the item of the witness or the
        item at which it stopped; the listed states, one per row, as ``[i, b_0, ...,
        b_{P}]``, for item ``i`` and the branches ``b_d``, through the witness if there is one;
        and, if the search stopped early, the path and ends with which its item goes on, and
        otherwise two empty arrays.

    Raises:
        ValueError: If ``fixed_words`` is not empty and its length is not the number of words of a
            row of ``symbol_masks``.
    """
    # Size the stack, and the branches of a state, which include the values of a root.
    num_symbols = len(symbol_masks)
    num_variables = len(ranks)
    num_values = num_symbols // num_variables
    max_column_weight = len(inverse_overlaps) - 1
    max_branches = num_values
    for check in range(len(check_starts) - 1):
        max_branches = max(max_branches, check_starts[check + 1] - check_starts[check])

    # Loops over the words of a syndrome run num_words times, which a version for a fixed number of
    # words knows when it is compiled.  Too many words would read past the rows of symbol_masks,
    # and too few would silently skip unsatisfied checks.
    num_words = len(fixed_words) if len(fixed_words) else symbol_masks.shape[1]
    if num_words != symbol_masks.shape[1]:
        raise ValueError("fixed_words must have an entry for each word of a syndrome, or none")

    # A symbol is available if its level is the number of states in the stack that restrict their
    # subtrees, which stays below _BLOCKED.  The level counts those that allow the symbol, plus
    # _BLOCKED for each reason that blocks it: a state in the stack tried it, another value of its
    # variable is the current branch of a state in the stack, its variable ranks below the root, or
    # an overlap cut forbids it.
    levels = np.zeros(num_symbols, dtype=np.int64)
    num_restrictions = 0

    # The load of the support on each overlap cut.  A symbol that would raise a load above its
    # capacity is blocked for as long as the load stays above the capacity minus the load of the
    # symbol, or from the start if the load of the symbol alone exceeds the capacity (Section 3.4 of
    # connected_cluster.md).
    cut_loads = np.zeros(len(cut_capacities), dtype=np.int64)
    for symbol in cut_blocked_symbols:
        levels[symbol] += _BLOCKED

    # The state at each depth has a syndrome, branches, and the cursor of its next branch.  A state
    # that restricts its subtree sets restricts[depth], since it may allow no symbols, and allows
    # allowances[allowance_starts[depth] : allowance_starts[depth + 1]].  Room for every state to
    # allow every symbol would be large under a large cutoff, so allowances grows as needed.
    syndromes = np.zeros((max_weight + 1, num_words), dtype=np.uint64)
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

    # The path of the current item, and the branch before which the state at each depth leaves; the
    # entry of a depth is MAX_VALUE unless the state at that depth is on the path.  States at
    # depths below replay_depth replay the path, and are not counted.
    route = np.empty(max_weight + 1, dtype=np.int64)
    limits = np.full(max_weight + 1, MAX_VALUE, dtype=np.int64)

    # The states at the frontier depth that the search lists, and its count of states.
    frontier = np.empty((len(roots) if frontier_depth else 0, frontier_depth + 1), dtype=np.int64)
    num_listed = 0
    item_states = np.zeros(len(roots), dtype=np.int64)
    num_counted = 0
    for item in range(len(roots)):
        root = roots[item]
        if item == 0 and len(resume):
            path_length = len(resume)
            route[:path_length] = resume
            limits[:path_length] = ends
        else:
            path_length = paths.shape[1]
            route[:path_length] = paths[item]
            limits[:path_length] = paths[item] + 1
        replay_depth = path_length
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
                num_words,
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
            # Block the variables of lower rank than the root, and branch on its available values.
            for variable in range(num_variables):
                if ranks[variable] < ranks[root]:
                    for value in range(num_values):
                        levels[value * num_variables + variable] += _BLOCKED
            num_branches[0] = 0
            for value in range(num_values):
                if levels[value * num_variables + root] == 0:
                    branches[0, num_branches[0]] = value * num_variables + root
                    num_branches[0] += 1
        while depth >= 0:
            # Back at a state from a branch, unblock the other values of the variable of the branch.
            if cursors[depth]:
                _shift_other_values(
                    levels, branches[depth, cursors[depth] - 1], num_variables, -_BLOCKED
                )
                _shift_cut_loads(
                    levels,
                    cut_loads,
                    branches[depth, cursors[depth] - 1],
                    -1,
                    max_weight,
                    cut_symbol_starts,
                    cut_symbol_entries,
                    cut_starts,
                    cut_entries,
                    cut_capacities,
                )

            # Leave the state once it has tried every branch, or every branch before its limit on
            # the path of the item, and release the symbols that it tried or did not allow.  The
            # states that its parent enters next do not replay the path, so they are counted.
            if cursors[depth] >= min(num_branches[depth], limits[depth]):
                for index in range(cursors[depth]):
                    levels[branches[depth, index]] -= _BLOCKED
                if restricts[depth]:
                    for index in range(allowance_starts[depth], allowance_starts[depth + 1]):
                        levels[allowances[index]] -= 1
                    num_restrictions -= 1
                limits[depth] = MAX_VALUE
                replay_depth = min(replay_depth, depth)
                depth -= 1
                continue

            # On the path of the item, skip to its branch.  An earlier sibling would leave only its
            # own symbol tried, once the search left its subtree.
            if depth < replay_depth:
                while cursors[depth] < route[depth]:
                    levels[branches[depth, cursors[depth]]] += _BLOCKED
                    cursors[depth] += 1

            # Stop once enough states are counted, outside the replay of the path, and return the
            # branch of each state of the stack and the next branch of the last state as a path.
            if num_counted >= max_states and depth + 1 >= replay_depth:
                stop_path = cursors[: depth + 1].copy()
                stop_path[:depth] -= 1
                stop_ends = np.minimum(limits[: depth + 1], num_branches[: depth + 1])
                return (
                    np.empty(0, dtype=np.int64),
                    item_states[: item + 1],
                    frontier[:num_listed],
                    stop_path,
                    stop_ends,
                )

            # Take the next branch, and skip its state if it cannot meet the weight limit, since a
            # symbol flips at most max_column_weight unsatisfied checks (b1 in Section 3.4).
            symbol = branches[depth, cursors[depth]]
            cursors[depth] += 1
            levels[symbol] += _BLOCKED
            _shift_other_values(levels, symbol, num_variables, _BLOCKED)
            _shift_cut_loads(
                levels,
                cut_loads,
                symbol,
                1,
                max_weight,
                cut_symbol_starts,
                cut_symbol_entries,
                cut_starts,
                cut_entries,
                cut_capacities,
            )
            support[depth] = symbol
            num_unsatisfied = _xor_and_count(syndromes, num_words, depth, symbol_masks, symbol)
            needed = (num_unsatisfied + max_column_weight - 1) // max_column_weight
            if depth + 1 + needed > max_weight:
                continue

            # At the frontier depth, list the state rather than enter it.
            if depth + 1 == frontier_depth:
                if num_listed == len(frontier):
                    larger = np.empty((2 * len(frontier) + 16, frontier.shape[1]), dtype=np.int64)
                    larger[:num_listed] = frontier[:num_listed]
                    frontier = larger
                frontier[num_listed, 0] = item
                frontier[num_listed, 1:] = cursors[:frontier_depth] - 1
                num_listed += 1
                continue

            # Enter the state: return it if it is a witness, and otherwise choose its branches.
            # Only the states of the item count, not those that replay its path.
            depth += 1
            if depth >= replay_depth:
                item_states[item] += 1
                num_counted += 1
            num_branches[depth] = 0
            cursors[depth] = 0
            allowance_starts[depth + 1] = allowance_starts[depth]
            restricts[depth] = False
            if num_unsatisfied == 0:
                if root < 0 or _is_outside_excluded(test_columns, support, depth):
                    return (
                        support[:depth].copy(),
                        item_states[: item + 1],
                        frontier[:num_listed],
                        np.empty(0, dtype=np.int64),
                        np.empty(0, dtype=np.int64),
                    )
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
                num_words,
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
    return (
        np.empty(0, dtype=np.int64),
        item_states,
        frontier[:num_listed],
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.int64),
    )


# Unlike the small helpers, _get_branches is long and has two call sites, but inlining it makes each
# state about 3 to 12 percent cheaper with a fixed number of words, and up to about 6 percent
# cheaper without, for the cost of compiling its code at both sites.
@always_compiled(inline=True)
def _get_branches(
    check_starts: npt.NDArray[np.int64],
    check_symbols: npt.NDArray[np.int64],
    symbol_masks: npt.NDArray[np.uint64],
    levels: npt.NDArray[np.int64],
    available_level: int,
    syndromes: npt.NDArray[np.uint64],
    num_words: int,
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

    A symbol is available if its level is ``available_level``.  A row of ``syndromes`` has
    ``num_words`` words.  The support can take ``slack`` more variables.  The weights of the
    unsatisfied checks are in units of ``1 / inverse_overlaps[1]``.
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
    scale = inverse_overlaps[1]
    memoize = memos.shape[1] > 0
    overlap_stamps, overlaps, repair_stamps, repairs = memos[0], memos[1], memos[2], memos[3]

    # Weigh each unsatisfied check, and choose the one with the fewest available symbols, starting
    # from a count above any possible count.
    total_weight = 0
    branch_check = -1
    fewest = check_starts[-1] + 1
    for word in range(num_words):
        bits = syndromes[depth, word]
        while bits:
            check = 64 * word + int(trailing_zeros_word(bits))
            bits &= bits - np.uint64(1)
            largest_overlap = 0
            count = 0
            if memoize:
                for position in range(check_starts[check], check_starts[check + 1]):
                    symbol = check_symbols[position]
                    if levels[symbol] == available_level:
                        if overlap_stamps[symbol] != stamp:
                            overlap = _get_overlap(
                                syndromes, num_words, depth, symbol_masks, symbol
                            )
                            overlap_stamps[symbol] = stamp
                            overlaps[symbol] = overlap
                        else:
                            overlap = overlaps[symbol]
                        largest_overlap = max(largest_overlap, overlap)
                        count += 1
            else:
                # Without memos, take the overlap of every symbol, and count only the available
                # ones by a select rather than a branch, which the processor would often mispredict.
                for position in range(check_starts[check], check_starts[check + 1]):
                    symbol = check_symbols[position]
                    available = levels[symbol] == available_level
                    overlap = _get_overlap(syndromes, num_words, depth, symbol_masks, symbol)
                    largest_overlap = max(largest_overlap, overlap if available else 0)
                    count += available
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
        for word in range(num_words):
            bits = syndromes[depth, word]
            while bits:
                check = 64 * word + int(trailing_zeros_word(bits))
                bits &= bits - np.uint64(1)
                count = 0
                for position in range(check_starts[check], check_starts[check + 1]):
                    symbol = check_symbols[position]
                    if levels[symbol] == available_level:
                        if not memoize:
                            repair = _get_repair(
                                syndromes, num_words, depth, symbol_masks, symbol, check_weights
                            )
                        elif repair_stamps[symbol] != stamp:
                            repair = _get_repair(
                                syndromes, num_words, depth, symbol_masks, symbol, check_weights
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
            repair = _get_repair(syndromes, num_words, depth, symbol_masks, symbol, check_weights)
        elif repair_stamps[symbol] != stamp:
            repair = _get_repair(syndromes, num_words, depth, symbol_masks, symbol, check_weights)
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
def _shift_cut_loads(
    levels: npt.NDArray[np.int64],
    cut_loads: npt.NDArray[np.int64],
    symbol: int,
    sign: int,
    max_weight: int,
    cut_symbol_starts: npt.NDArray[np.int64],
    cut_symbol_entries: npt.NDArray[np.int64],
    cut_starts: npt.NDArray[np.int64],
    cut_entries: npt.NDArray[np.int64],
    cut_capacities: npt.NDArray[np.int64],
) -> None:
    """Add (``sign`` 1) or remove (``sign`` -1) the loads of a symbol on its overlap cuts.

    A symbol of load ``l`` on a cut is blocked while the load of the support on the cut exceeds its
    capacity minus ``l``, so this blocks or unblocks the symbols of each cut whose load crosses that
    threshold.  A support within the weight limit has a load of at most ``2 max_weight`` on any cut,
    so a cut whose capacity is at least that blocks nothing, and is skipped.
    """
    for index in range(cut_symbol_starts[symbol], cut_symbol_starts[symbol + 1]):
        cut = cut_symbol_entries[index] // _LOAD_BASE
        capacity = cut_capacities[cut]
        if capacity >= 2 * max_weight:
            continue
        old = cut_loads[cut]
        new = old + sign * (cut_symbol_entries[index] % _LOAD_BASE)
        cut_loads[cut] = new
        low, high = min(old, new), max(old, new)
        crosses_one = low <= capacity - 1 < high
        crosses_two = low <= capacity - 2 < high
        if crosses_one or crosses_two:
            for position in range(cut_starts[cut], cut_starts[cut + 1]):
                load = cut_entries[position] % _LOAD_BASE
                if (crosses_one and load == 1) or (crosses_two and load == 2):
                    levels[cut_entries[position] // _LOAD_BASE] += sign * _BLOCKED


@always_compiled(inline=True)
def _get_overlap(
    syndromes: npt.NDArray[np.uint64],
    num_words: int,
    depth: int,
    symbol_masks: npt.NDArray[np.uint64],
    symbol: int,
) -> int:
    """The number of unsatisfied checks that a symbol flips, in the syndrome at a depth.

    A row of ``syndromes`` has ``num_words`` words.
    """
    count = 0
    for word in range(num_words):
        count += int(popcount_word(syndromes[depth, word] & symbol_masks[symbol, word]))
    return count


@always_compiled(inline=True)
def _get_repair(
    syndromes: npt.NDArray[np.uint64],
    num_words: int,
    depth: int,
    symbol_masks: npt.NDArray[np.uint64],
    symbol: int,
    check_weights: npt.NDArray[np.int64],
) -> int:
    """The total weight of the unsatisfied checks that a symbol flips, at a depth.

    A row of ``syndromes`` has ``num_words`` words.
    """
    repair = 0
    for word in range(num_words):
        bits = syndromes[depth, word] & symbol_masks[symbol, word]
        while bits:
            repair += check_weights[64 * word + int(trailing_zeros_word(bits))]
            bits &= bits - np.uint64(1)
    return repair


@always_compiled(inline=True)
def _xor_and_count(
    syndromes: npt.NDArray[np.uint64],
    num_words: int,
    depth: int,
    symbol_masks: npt.NDArray[np.uint64],
    symbol: int,
) -> int:
    """Set the syndrome at ``depth + 1`` to that at ``depth`` plus a symbol, and return its size.

    A row of ``syndromes`` has ``num_words`` words, and the size of a syndrome is its number of
    unsatisfied checks.
    """
    count = 0
    for word in range(num_words):
        syndromes[depth + 1, word] = syndromes[depth, word] ^ symbol_masks[symbol, word]
        count += int(popcount_word(syndromes[depth + 1, word]))
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
