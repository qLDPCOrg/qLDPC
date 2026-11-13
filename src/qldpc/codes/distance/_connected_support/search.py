# SPDX-License-Identifier: Apache-2.0

"""Exact distances by a depth-first search of the supports connected in a Tanner graph.

The search finds a nonzero vector ``x`` that satisfies sparse parity checks ``H`` and is outside the
row space of ``G``, by growing a support only through unsatisfied checks, and pruning with bounds on
the number of variables that a support still needs.  The vectors are binary, or the symplectic
vectors of Pauli operators, whose weights count qubits.  It follows Algorithm 2 in Section V of
Okada and Kasai, https://arxiv.org/abs/2607.14091, which searches CSS codes.  The idea is older:
searches of the Tanner graphs of classical LDPC codes, and the irreducible-cluster technique of
Dumer, Kovalev, and Pryadko for classical and quantum LDPC codes, use it too (see the references
of ``connected_support.md``).  The same search, started from a nonzero syndrome rather than from
a root, finds a light vector with that syndrome, such as a light Pauli error with a given syndrome
on a stabilizer state.

``connected_support.md`` in ``src/qldpc/codes/distance/`` derives the search and states its
correctness requirements.  The Brouwer-Zimmermann search in ``_brouwer_zimmermann/distance.py``
solves the same problem for any binary code.  This module reuses ``infer_weight_divisor`` of
``_information_sets/basis_families.py``, and ``verify_automorphisms`` and ``get_orbits`` of
``_information_sets/automorphism_families.py``.  The coset searches of ``CosetLeaderSearch`` that
start from a syndrome are in ``coset.py`` beside this module.
"""

from __future__ import annotations

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


def get_distance_connected_support(
    problems: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
    *,
    cutoff: int,
    upper_bound: int | None = None,
    homogeneous: bool = True,
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
    tuning: SearchTuning | None = None,
) -> int:
    """Compute the minimum distance across problems by searching connected supports.

    Each problem is a pair ``(checks, excluded)`` of binary matrices with the same number of
    columns, whose distance is the least weight of a word that satisfies every check and is outside
    the row space of ``excluded`` (see ``SupportSearch``).  The rows of ``excluded`` must satisfy
    every check.

    The search of each problem raises its weight limit ``W`` through the weights that a witness may
    have, one weight divisor at a time.  Each pass either finds a witness of weight at most ``W`` or
    certifies that every witness is heavier, so a witness within a limit that follows such passes
    is a lightest one.  The first limit of each problem is the larger of its weight divisor and
    the largest multiple of it that is at most ``cutoff``.  If the search finds a witness of weight
    at most ``cutoff``, the result is the weight of that witness, which can exceed the distance.

    Args:
        problems: The ``(checks, excluded)`` pairs to search.
        cutoff: Exit early and return once an upper bound on the distance falls to or below this
            cutoff.
        upper_bound: A cap on the result, if any, such as the weight of a known witness that is
            not in ``problems``, or ``max_weight + 1`` under a weight limit.  The result is the
            minimum of this bound and the distances of ``problems``.
        homogeneous: Whether the problems have binary words, rather than the symplectic words of
            Pauli operators (see ``SupportSearch``).  Default: True.
        get_automorphisms: A function that returns an array of generators of a group of
            permutations of the variables, or None.  Each row ``p`` of the array maps variable ``j``
            to variable ``p[j]``.  Each generator must map the words of every problem, and the row
            space of its excluded rows, onto themselves.  The search of each problem calls the
            function at most once, before the first weight limit at which it expects to have
            visited ``automorphism_minimum_states`` states (a field of ``tuning``), and verifies the
            generators before it uses them.  A first limit that ``cutoff`` raises has no estimate
            of its states, so the search asks before it, unless that field is ``MAX_VALUE``.
        tuning: The tuning of the search (see ``_tuning``), which defaults to the active tuning.

    Returns:
        The minimum distance across the problems and the optional upper bound, or an upper bound
        that is at most ``cutoff``.  A problem without a word outside its excluded row space has no
        distance, and counts as one more than its number of variables.

    Raises:
        ValueError: If there are neither problems nor an upper bound, if the matrices of a problem
            are not valid (see ``SupportSearch``), or if a generator from ``get_automorphisms`` does
            not map the words of a searched problem, or the row space of its excluded rows, onto
            themselves.
        RuntimeError: If the searches take more weight limits than a correct search can, which
            only a bug can cause.
    """
    if not problems and upper_bound is None:
        raise ValueError("Require at least one problem or an upper bound")

    # A problem without witnesses only bounds the result, unlike in
    # get_distance_brouwer_zimmermann_many, which needs a witness in every problem.  A known bound
    # within the cutoff ends the search before it starts, as it does there.
    tuning = get_tuning() if tuning is None else tuning
    automorphism_minimum_states = tuning.automorphism_minimum_states
    best_weight = upper_bound
    no_distance_bounds = []
    searches: list[SupportSearch] = []
    for checks, excluded in problems:
        search = SupportSearch(
            checks,
            excluded,
            homogeneous=homogeneous,
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
    # A correct search takes at most n // d limits for each problem of n variables and divisor d
    # (see "Termination" in connected_support.md), and a broken one fails instead of looping.
    lower_bounds = [search.weight_divisor for search in searches]
    max_num_limits = sum(search.num_variables // search.weight_divisor for search in searches)
    num_limits = 0
    while best_weight is None or best_weight > min(lower_bounds):
        if num_limits == max_num_limits:
            raise RuntimeError(
                f"The connected-support search needed more than {max_num_limits} weight limits,"
                " which only a bug can cause"
            )
        num_limits += 1
        index = lower_bounds.index(min(lower_bounds))
        search = searches[index]
        divisor = search.weight_divisor
        max_weight = max(lower_bounds[index], cutoff - cutoff % divisor)
        witness = search.find_witness(max_weight)
        if witness is None:
            lower_bounds[index] = (max_weight // divisor + 1) * divisor
            continue
        best_weight = len(witness) if best_weight is None else min(best_weight, len(witness))
        if best_weight <= cutoff:
            break
    return min([best_weight, *no_distance_bounds])


class SupportSearch:
    """Search for light words that satisfy ``checks`` and are outside the row space of ``excluded``.

    A word is a vector ``x`` that satisfies every check, and its support is the set of variables on
    which it is nonzero.  Its weight is the size of its support.  The row space of ``excluded`` must
    satisfy every check, and the words outside it are the witnesses.  The search grows a support
    through the checks that it leaves unsatisfied (see ``connected_support.md``), so it is fast if
    ``checks`` is sparse.  It also finds light vectors with a given syndrome, which need not
    satisfy the checks (see ``find_vector_with_syndrome``).

    A homogeneous problem has binary words, whose coordinates are the variables, and a word ``x``
    satisfies a check ``h`` if ``h @ x = 0``.  A problem that is not homogeneous has the symplectic
    words ``x = (a | b)`` of Pauli operators on ``n`` qubits, which are its variables, and the value
    of a qubit is the Pauli ``X``, ``Z``, or ``Y`` that ``x`` has there.  Such a word satisfies a
    check if they commute, which is if their symplectic product vanishes.

    Args:
        checks: A binary matrix whose rows are the checks.  Redundant rows can strengthen the
            search, so the matrix need not have independent rows.
        excluded: A binary matrix whose row space is excluded.

    Keyword args:
        homogeneous: Whether the problem is homogeneous, rather than symplectic.  Default: True.
        get_automorphisms: A function that returns an array of generators of a group of
            permutations of the variables, or None, as in ``get_distance_connected_support``.
        automorphism_minimum_states: The search calls ``get_automorphisms`` at most once, before
            the first weight limit at which it expects its visited states, over all weight limits,
            to reach this number, as the field of ``SearchTuning`` says.  Zero asks for them before
            the first weight limit.

    Attributes:
        num_variables: The number of variables: the number of columns of ``checks`` if the problem
            is homogeneous, and half of it otherwise.
        weight_divisor: A divisor of the weight of every word.
        num_states: The number of states that ``find_witness`` has visited so far, over all of its
            weight limits.  The empty support at the start of the search from each root is not
            counted, and neither are the states of syndrome queries.

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
        homogeneous: bool = True,
        get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
        automorphism_minimum_states: int,
    ) -> None:
        # The compiled search would read invalid input without an error, so the search validates its
        # input, which callers need not check.
        assert_binary(checks, "checks")
        assert_binary(excluded, "excluded")
        checks = np.asarray(checks, dtype=np.uint8)
        excluded = np.asarray(excluded, dtype=np.uint8)
        if checks.ndim != 2 or excluded.ndim != 2 or checks.shape[1] != excluded.shape[1]:
            raise ValueError("Checks and excluded rows must be matrices with equal widths")
        if not homogeneous and checks.shape[1] % 2:
            raise ValueError("Symplectic matrices must have an even number of columns")

        self.num_variables = checks.shape[1] if homogeneous else checks.shape[1] // 2
        self.num_states = 0
        self._homogeneous = homogeneous
        self._last_two_num_states: tuple[int, int] | None = None
        self._excluded = excluded
        self._get_automorphisms = get_automorphisms
        self._automorphism_minimum_states = automorphism_minimum_states

        # A symplectic check is satisfied if its dot product with a word vanishes once its halves
        # are swapped, and the excluded rows must satisfy every check.
        if not homogeneous:
            checks = swap_pauli_halves(checks)
        if np.any(multiply_binary(checks, excluded.T)):
            raise ValueError("The excluded rows must satisfy the checks")
        # Zero and duplicate checks add nothing to a search, which keeps the first copy of each
        # nonzero check, in order.  A syndrome query maps each given check to the index of the kept
        # check that equals it, or to -1 for a zero check.  The unique rows are sorted, and
        # ``rank_of_unique`` inverts the order that sorts their first copies.
        nonzero_rows = np.flatnonzero(np.any(checks, axis=1))
        _, first_rows, unique_index_of_row = np.unique(
            checks[nonzero_rows], axis=0, return_index=True, return_inverse=True
        )
        order = np.argsort(first_rows)
        rank_of_unique = np.argsort(order)
        self._checks = checks[nonzero_rows[first_rows[order]]]
        self._kept_index_of_check = np.full(len(checks), -1, dtype=np.int64)
        self._kept_index_of_check[nonzero_rows] = rank_of_unique[unique_index_of_row.ravel()]

        # The checks that each symbol flips, and the tests that it fails, are the columns of the
        # matrices of the checks and of the tests for its value.
        flips = self._get_symbol_columns(self._checks)
        sparse_flips = scipy.sparse.csr_array(flips)
        self._check_starts = sparse_flips.indptr.astype(np.int64)
        self._check_symbols = sparse_flips.indices.astype(np.int64)
        self._symbol_masks = pack_bits(np.ascontiguousarray(flips.T))
        tests = self._get_symbol_columns(self._get_tests())
        self._test_columns = pack_bits(np.ascontiguousarray(tests.T))

        # Without symmetries, every variable is a root.  Words of Pauli operators get no divisor,
        # since their weights count qubits rather than bits.
        self._roots = np.arange(self.num_variables, dtype=np.int64)
        self._ranks = np.arange(self.num_variables, dtype=np.int64)
        self.weight_divisor = infer_weight_divisor(get_kernel(self._checks)) if homogeneous else 1

        # Weights of checks are multiples of 1 / scale, and an overlap is at most the largest number
        # of checks of a symbol, which is taken to be at least one, so that the scale exists even
        # without checks.  Every overlap divides the scale if the scale is small enough, and the
        # inverse of an overlap is rounded down otherwise.
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
        """Search for a witness of weight at most ``max_weight``.

        The search is exhaustive: it finds a witness if there is one of weight at most
        ``max_weight``, and otherwise certifies that every witness is heavier.

        Args:
            max_weight: The weight limit, which is at least one.

        Returns:
            The sorted symbols of a witness, one for each variable of its support, or None if there
            is no witness of weight at most ``max_weight``.  A witness need not be a lightest one.
            Symbol ``k n + j`` gives variable ``j`` value ``k``, where ``n`` is the number of
            variables.  A homogeneous problem has one value, so its symbols are its variables, and
            a symplectic problem has the values ``X``, ``Z``, and ``Y``, in that order.

        Raises:
            ValueError: If the weight limit is less than one, or if this search asks for
                automorphisms, and a generator does not map the words, or the row space of the
                excluded rows, onto themselves.
        """
        if max_weight < 1:
            raise ValueError(f"The weight limit must be at least one, not {max_weight}")
        # Ask for automorphisms once, before the first weight limit whose states, with those visited
        # so far, are expected to reach the threshold.
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
        )
        self.num_states += num_states
        last = self._last_two_num_states[1] if self._last_two_num_states else 0
        self._last_two_num_states = (last, num_states)
        return np.sort(support) if len(support) else None

    def find_vector_with_syndrome(
        self, syndrome: npt.ArrayLike, max_weight: int
    ) -> npt.NDArray[np.int64] | None:
        """Search for a vector with a given syndrome, of weight at most ``max_weight``.

        The syndrome of a vector says which checks it fails, so the vectors with a syndrome are a
        coset of the vectors that satisfy every check, and the excluded rows play no part.  The
        search starts from the syndrome, and grows supports through its unsatisfied checks without
        roots, symmetries, or a weight divisor (see Section 3.8 of ``connected_support.md``).  Like
        ``find_witness``, it is exhaustive, but need not find a lightest vector.

        Args:
            syndrome: A binary vector with one entry for each row of the checks that the search was
                given, in their order.  For a symplectic problem, entry ``j`` says whether a Pauli
                operator anticommutes with check ``j``.
            max_weight: The weight limit, which is at least zero.

        Returns:
            The sorted symbols of a vector with the syndrome, one for each variable of its support,
            as ``find_witness`` returns them, or None if every such vector weighs more than
            ``max_weight``.

        Raises:
            ValueError: If the syndrome is not a binary vector with one entry for each check, or if
                the weight limit is negative.
        """
        assert_binary(syndrome, "syndrome")
        syndrome = np.asarray(syndrome, dtype=np.uint8)
        if syndrome.shape != self._kept_index_of_check.shape:
            num_checks = len(self._kept_index_of_check)
            raise ValueError(f"The syndrome must have one entry for each of {num_checks} checks")
        if max_weight < 0:
            raise ValueError(f"The weight limit must be at least zero, not {max_weight}")

        # Equal checks have equal entries, and a zero check has entry zero, for every vector.
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
        """Search as ``find_vector_with_syndrome`` does, from a packed syndrome of the kept checks.

        A caller that asks many queries, such as a coset search, can skip the validation, mapping,
        and packing of the syndrome that ``find_vector_with_syndrome`` does on every call.

        Args:
            packed_syndrome: A syndrome with one bit for each kept check, packed as ``pack_bits``
                packs a row.  The kept checks are the given checks without zero and duplicate
                rows, in order, so they are the given checks if those are independent.  The
                syndrome must not be zero; ``find_vector_with_syndrome`` answers a zero syndrome
                itself.
            max_weight: The weight limit, which is at least zero.

        Returns:
            The sorted symbols of a vector with the syndrome, or None, as
            ``find_vector_with_syndrome`` returns them.
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
        )
        return np.sort(support) if len(support) else None

    def _predict_states(self, max_weight: int) -> int:
        """The number of states that the next weight limit, ``max_weight``, is expected to visit.

        The number of states of each weight limit is expected to grow by the ratio between those of
        the last two limits.  It is expected not to grow if the limit before the last visited no
        states, as before the second limit (see Section 3.5 of ``connected_support.md``).  A first
        limit above the weight divisor, which a cutoff can raise, skips the cheap limits below it,
        so it has no estimate.  It is expected to reach any threshold short of ``MAX_VALUE``.
        """
        if self._last_two_num_states is None:
            return MAX_VALUE - 1 if max_weight > self.weight_divisor else 0
        previous, last = self._last_two_num_states
        return last * last // previous if previous else last

    def _get_symbol_columns(self, matrix: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """The products of the rows of a matrix with the word of each symbol, as its columns.

        The word of a symbol has the value of the symbol on its variable, and is zero elsewhere.
        For a symplectic problem, the columns are those of the ``X`` part, of the ``Z`` part, and
        of their sum, for the values ``X``, ``Z``, and ``Y``.
        """
        if self._homogeneous:
            return matrix
        part_x, part_z = matrix[:, : self.num_variables], matrix[:, self.num_variables :]
        return np.hstack([part_x, part_z, part_x ^ part_z])

    def _get_tests(self) -> npt.NDArray[np.uint8]:
        """Rows whose products with a word all vanish exactly when the word is excluded.

        The row space of ``excluded`` is the set of vectors that are orthogonal to the kernel of
        ``excluded``.  That kernel contains the rows of the checks, once their halves are swapped
        for a symplectic problem, which every word is orthogonal to, so it is enough to test against
        the rest of the kernel of ``excluded``.
        """
        basis, labels = get_nested_code_basis(get_kernel(self._excluded), self._checks)
        return basis[np.any(labels, axis=1)]

    def _use_automorphisms(self, automorphisms: npt.NDArray[np.int_]) -> None:
        """Verify automorphisms, and search only from the least variable of each of their orbits.

        The variables are ranked orbit by orbit, so that the search from each root excludes every
        variable of the earlier orbits.  A permutation of the qubits of a symplectic problem moves
        both halves of each word.

        Raises:
            ValueError: If a generator does not map the words, or the row space of the excluded
                rows, onto themselves.
        """
        basis, labels = get_nested_code_basis(get_kernel(self._checks), self._excluded)
        lifted = (
            automorphisms
            if self._homogeneous
            else np.hstack([automorphisms, automorphisms + self.num_variables])
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
) -> tuple[npt.NDArray[np.int64], int]:
    """Search the supports that grow from each root through unsatisfied checks.

    A symbol is a variable with one of its nonzero values.  Symbol ``s`` gives value ``s // n`` to
    variable ``s % n``, where ``n = len(ranks)`` is the number of variables, and every variable has
    the same number of values.  The graph is the Tanner graph of the checks and the symbols that
    flip them: check ``c`` is flipped by the symbols ``check_symbols[check_starts[c] :
    check_starts[c + 1]]``.  Row ``s`` of ``symbol_masks`` packs the checks that symbol ``s`` flips
    into the bits of 64-bit blocks, and row ``s`` of ``test_columns`` packs the tests that the word
    of symbol ``s`` alone fails.  The search from a root uses only variables of higher rank.  The
    weights of checks are integers, in units of ``1 / scale``, and entry ``k`` of
    ``inverse_overlaps`` is ``scale // k``, up to the largest number of checks of a symbol, so its
    entry 1 is the scale (see ``_get_branches``).

    It is the depth-first search of Algorithm 2 of https://arxiv.org/abs/2607.14091, with an
    explicit stack, and with values.  A state is a support, whose symbols are ``support[:depth]``,
    on distinct variables, and its syndrome is the set of its unsatisfied checks.  A state branches
    on the viable symbols that flip one unsatisfied check.  Each symbol that it tries stays blocked
    until the search leaves the state, so that its later branches do not use it again, and the other
    values of its variable are blocked while its branch is in the stack.  A state also allows only
    its viable symbols in its subtree, if it rules out any.  The state at depth zero is the empty
    support, which branches on the values of the root, and which the count of states omits.

    A root of -1 instead starts a search for a vector whose syndrome is ``start_syndrome``, packed
    as a row of ``symbol_masks`` is, rather than for a witness.  The syndrome must not be empty.
    Its empty state has that syndrome, and branches on one of its unsatisfied checks, as every
    other state does, without a root or ranks.  Every support that it reaches with no unsatisfied
    check is a solution, without tests.  The searches from other roots ignore ``start_syndrome``.

    With ``memoize``, each state computes the overlap and the repair of each symbol at most once
    (see ``_get_branches``).  This saves time when checks are dense, as those of syndrome queries
    can be (see Section 3.8 of ``connected_support.md``).  It costs time when they are sparse.

    Returns:
        The symbols of a witness, or of a solution, of weight at most ``max_weight``, or an empty
        array, and the number of states that the search visited.
    """
    # The sizes of the stack, and of the branches of a state, which are at least the values of the
    # root that the empty state branches on.
    num_symbols, num_blocks = symbol_masks.shape
    num_variables = len(ranks)
    num_values = num_symbols // num_variables
    max_column_weight = len(inverse_overlaps) - 1
    max_branches = num_values
    for check in range(len(check_starts) - 1):
        max_branches = max(max_branches, check_starts[check + 1] - check_starts[check])

    # The level of a symbol counts the states in the stack whose restrictions allow it, plus
    # _BLOCKED for each reason that blocks it: a state in the stack has tried it as a branch, which
    # includes every symbol of the support; another value of its variable is the current branch of a
    # state in the stack; or its variable ranks below the root.  A symbol is available if its level
    # is the number of active restrictions, which is never as large as _BLOCKED.
    levels = np.zeros(num_symbols, dtype=np.int64)
    num_restrictions = 0

    # The state at each depth has its syndrome, its branch symbols, and the cursor of the next
    # branch symbol to try.  If it restricts its subtree, the symbols that it allows are
    # allowances[allowance_starts[depth] : allowance_starts[depth + 1]], and restricts[depth] says
    # so, which the range alone would not for a state that allows nothing.  A state allows each
    # symbol at most once, and allowances grows if a state could overflow it, rather than start
    # with room for max_weight * num_symbols entries, which a large cutoff would make large.
    syndromes = np.zeros((max_weight + 1, num_blocks), dtype=np.uint64)
    support = np.empty(max_weight + 1, dtype=np.int64)
    branches = np.empty((max_weight + 1, max_branches), dtype=np.int64)
    num_branches = np.zeros(max_weight + 1, dtype=np.int64)
    cursors = np.zeros(max_weight + 1, dtype=np.int64)
    restricts = np.zeros(max_weight + 1, dtype=np.bool_)
    allowance_starts = np.zeros(max_weight + 2, dtype=np.int64)
    allowances = np.empty(num_symbols, dtype=np.int64)

    # Scratch space for _get_branches, whose docstring describes ``memos``.  Each call of
    # _get_branches gets a stamp of its own: the number of calls so far, including itself.  So the
    # zeros of ``memos`` match no stamp.
    check_weights = np.zeros(len(check_starts) - 1, dtype=np.int64)
    branch_repairs = np.empty(max_branches, dtype=np.int64)
    memos = np.zeros((4, num_symbols if memoize else 0), dtype=np.int64)
    stamp = 0

    num_states = 0
    for root in roots:
        # The empty state has the empty syndrome, and restricts nothing, unless the search starts
        # from a syndrome.
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
            # The search is back at a state after the branch that it tried last, so the other values
            # of the variable of that branch are available again.
            if cursors[depth]:
                _shift_other_values(
                    levels, branches[depth, cursors[depth] - 1], num_variables, -_BLOCKED
                )

            # Leave the state once it has tried every branch.  The symbols that it tried are
            # available again to its later siblings, and so are the symbols that it did not allow.
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
            # Every symbol is in at most max_column_weight unsatisfied checks.
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

            # Make room for the symbols that this state may allow, and choose its branches.
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
) -> int:
    """Choose the symbols that the state at a depth branches on, if its syndrome is not empty.

    A symbol is available if its level is ``available_level``.  A completion of a support that has
    ``slack`` variables to spare is a set ``T`` of available symbols, on distinct variables, that
    makes the support a witness, or a solution of a syndrome query, so ``T`` flips every
    unsatisfied check an odd number of times.  The overlap of an available symbol is the number of
    unsatisfied checks that it flips.  Weigh each unsatisfied check by the inverse of the largest
    overlap of its available symbols, in units of ``1 / scale``, rounded down, as
    ``inverse_overlaps`` lists, whose entry 1 is the scale.  The weights of the unsatisfied checks
    of any available symbol then total at most one, and their total is its repair.  So ``T`` has
    at least as many symbols as the total weight of the unsatisfied checks.  A symbol ``s`` of
    ``T`` repairs its share, and the other symbols of ``T`` the rest, so
    ``|T| >= 1 + total - repair(s)``.

    A symbol for which this exceeds ``slack`` is in no completion, and the state allows only the
    other symbols in its subtree.  It records them in ``allowances``, from
    ``allowance_starts[depth]``, and sets ``restricts[depth]``.  The state has no branches if an
    unsatisfied check has no viable symbol, or if the total weight exceeds ``slack``.  Otherwise it
    branches on the viable symbols of the unsatisfied check with the fewest, which it writes to
    ``branches[depth]``, most repairing first.  It returns their number.  ``check_weights`` and
    ``branch_repairs`` are scratch space for the weights of the unsatisfied checks and the repairs
    of the branch symbols.

    If ``memos`` has columns, one for each symbol, the state computes the overlap and the repair of
    each symbol at most once, and keeps them there.  Row 0 of ``memos`` holds the stamp of the
    call that computed the overlap of each symbol, and row 1 that overlap.  Rows 2 and 3 hold the
    same for repairs.  ``stamp`` is the stamp of this call, which no other call of the search uses.
    """
    num_blocks = syndromes.shape[1]
    scale = inverse_overlaps[1]
    memoize = memos.shape[1] > 0
    overlap_stamps, overlaps, repair_stamps, repairs = memos[0], memos[1], memos[2], memos[3]

    # Weigh each unsatisfied check, and choose the one with the fewest available symbols.  The
    # count starts above any possible count.
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
            # The total only grows, so the state fails as soon as it exceeds the slack.
            if total_weight > slack * scale:
                return 0
            if count < fewest:
                branch_check, fewest = check, count

    # A symbol that flips no unsatisfied check repairs nothing, so it is viable exactly when every
    # symbol is.  When the total weight leaves such symbols no room, allow only the viable symbols
    # of the unsatisfied checks, and count them in each check.
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

    # Branch on the available symbols of the chosen check, most repairing first, which Section 3.3
    # of connected_support.md measures.
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
    """Add ``change`` to the levels of the symbols that give the variable of a symbol other values.

    The loop is empty if every variable has one value, as for homogeneous problems.
    """
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
    """Set the syndrome at ``depth + 1`` to that at ``depth`` with a symbol added.

    Returns:
        The number of unsatisfied checks of the new syndrome.
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
