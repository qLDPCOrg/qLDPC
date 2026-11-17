# SPDX-License-Identifier: Apache-2.0

"""Connected-cluster coset searches of ``CosetLeaderSearch``, with checks made sparser first.

Several threads can run the queries of a coset search at once, as Section 3.8 of
``connected_cluster.md`` says, and a change must keep the requirements on threads of its Section 5.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .. import _tuning
from .._binary import swap_pauli_halves
from .._cosets import (
    get_unit_syndrome_words,
    int_to_words,
    row_ints,
    tabulate_singles,
    xor_selected,
)
from ..min_weight_bounds import MinWeightBound
from .search import SupportSearch
from .sparse_checks import get_sparser_rows


class ConnectedClusterCosetSearch:
    """A coset search of a ``CosetLeaderSearch`` by the connected-cluster search of its checks.

    The words and syndromes are those of a ``BrouwerZimmermannCosetSearch``: packed Paulis, or X or
    Z parts of Paulis.  The coset search makes its checks sparser, with the same span, and searches
    from the syndrome of each query, mapped to them (Section 3.8 of ``connected_cluster.md``).
    ``get_min_weight_bound`` gives these lower bounds on the weight of the words of a coset:

    - The weight of the word, for a coset with a word of weight zero or one, which a table finds.
    - Two, for any other coset whose word the search finds within its first weight limit
      (``max_weight`` for a single query), since the word can be lighter than that limit.
    - The weight limit, for a coset whose word the search finds within a later limit, which is then
      the weight of the lightest words.
    - The larger of two and ``max_weight + 1``, for any other coset without a word within
      ``max_weight``.

    So with a first limit of two, every coset with a word within ``max_weight`` gets the weight of
    its lightest words.

    Args:
        matrix: Independent checks, whose kernel is the code: the stabilizers, in ``[X|Z]`` order,
            for Paulis, or the rows that check the X or Z parts.  Bit ``j`` of the syndrome of a
            word is its symplectic product with row ``j`` for Paulis, and its dot product for parts.
        symplectic: True for Paulis, and False for X or Z parts.
        max_rounds: The field ``max_sparsifying_rounds`` of ``SearchTuning`` (see
            ``get_sparser_rows``).
        rounds_without_improvement: The field ``sparsifying_rounds_without_improvement`` of
            ``SearchTuning``.
        restrict_viable_symbols: The field ``restrict_viable_symbols`` of ``SearchTuning`` (see
            ``SupportSearch``).
        stabilizer_overlap_cuts: The field ``stabilizer_overlap_cuts`` of ``SearchTuning`` (see
            ``SupportSearch``).  The cuts need known words of the kernel of the checks, which
            ``use_query_overlap_words`` gives.
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.uint8],
        *,
        symplectic: bool,
        max_rounds: int,
        rounds_without_improvement: int,
        restrict_viable_symbols: bool,
        stabilizer_overlap_cuts: bool,
    ) -> None:
        # Tabulate the words of weight zero or one by syndrome.  The syndrome of a single bit is a
        # column of the checks, with the halves of symplectic checks swapped.
        num_variables = matrix.shape[1] // 2 if symplectic else matrix.shape[1]
        syndrome_matrix = swap_pauli_halves(matrix) if symplectic else matrix
        columns = row_ints(np.ascontiguousarray(syndrome_matrix.T))
        self._singles, _, _ = tabulate_singles(columns, num_variables, symplectic=symplectic)

        rows, transform = get_sparser_rows(
            matrix,
            symplectic=symplectic,
            max_rounds=max_rounds,
            rounds_without_improvement=rounds_without_improvement,
        )
        self._rows = rows
        # Row ``i`` of ``rows`` sums the given rows that row ``i`` of ``transform`` selects, so a
        # given syndrome maps to the sum of the columns of ``transform`` that its bits select.
        self._syndrome_map_columns = row_ints(np.ascontiguousarray(transform.T))
        # Without a function that finds automorphisms, the search never asks for them.
        self._search = SupportSearch(
            rows,
            np.zeros((0, matrix.shape[1]), dtype=np.uint8),
            symplectic=symplectic,
            automorphism_minimum_states=_tuning.MAX_VALUE,
            restrict_viable_symbols=restrict_viable_symbols,
            stabilizer_overlap_cuts=stabilizer_overlap_cuts,
            # Sums of pairs of checks made queries slower (Section 3.8 of connected_cluster.md).
            pair_sum_checks=False,
        )
        # For ``rows``, word ``j`` of ``get_unit_syndrome_words`` has a syndrome of only bit ``j``,
        # so the words of the set bits of a syndrome sum to a representative of its coset.
        syndrome_rows = swap_pauli_halves(rows) if symplectic else rows
        self._inverse_rows = row_ints(get_unit_syndrome_words(syndrome_rows))

    @property
    def checks(self) -> npt.NDArray[np.uint8]:
        """The sparser checks that the search uses, which span the same space as the given ones.

        The array is a read-only view.
        """
        rows = self._rows.view()
        rows.flags.writeable = False
        return rows

    def use_query_overlap_words(self, words: npt.ArrayLike) -> None:
        """Let the overlap cuts of the queries use words of the kernel of the checks.

        Such words are stabilizers, for example.  The cuts do nothing until this method gives them
        words.  The words are in the representation of the checks (see
        ``SupportSearch.use_query_overlap_words``).  Call it only before any query runs, as the
        constructor of ``CosetLeaderSearch`` does, since queries on other threads read the cuts.
        """
        self._search.use_query_overlap_words(words)

    def get_min_weight_bound(
        self,
        syndrome: int,
        max_weight: int,
        *,
        first_limit: int | None = None,
    ) -> MinWeightBound:
        """Decide whether the coset of this syndrome has a word of weight at most ``max_weight``.

        The search tries each weight limit from ``max(min(first_limit, max_weight), 2)`` through
        ``max_weight``, or, without ``first_limit``, only ``max_weight`` if it is at least two, and
        stops at the first within which it finds a word (Section 3.8 of ``connected_cluster.md``).
        A first limit above two suits callers to which any word within it gives the same answer.
        The class docstring states the lower bounds of the answers.
        """
        return self._get_min_weight_bound_and_states(syndrome, max_weight, first_limit)[0]

    def _get_min_weight_bound_and_states(
        self, syndrome: int, max_weight: int, first_limit: int | None
    ) -> tuple[MinWeightBound, int]:
        """Run ``get_min_weight_bound``, and count the states that its searches visit."""
        # Look up a word of weight zero or one.
        word = self._singles.get(syndrome)
        if word is not None:
            weight = int(word != 0)
            return MinWeightBound(weight <= max_weight, word, weight), 0

        # Map the syndrome to the sparser checks, which are independent, so the search keeps them
        # all, in order.  The map is invertible and the lookup answers the zero syndrome, so the
        # mapped syndrome is not zero, as ``_find_vector_with_packed_syndrome`` requires.
        syndrome = xor_selected(self._syndrome_map_columns, syndrome)
        packed_syndrome = int_to_words(syndrome, (len(self._rows) + 63) // 64)

        # The lookup leaves only cosets without a word below weight two, so search no limit below
        # two, and raise the lower bound above each limit within which the search finds no word.
        lower_bound = 2
        num_states = 0
        first = max_weight if first_limit is None else min(first_limit, max_weight)
        for limit in range(max(2, first), max_weight + 1):
            symbols, states = self._search._find_vector_with_packed_syndrome(packed_syndrome, limit)
            num_states += states
            if symbols is not None:
                return MinWeightBound(True, self._get_word(symbols), lower_bound), num_states
            lower_bound = limit + 1
        representative = xor_selected(self._inverse_rows, syndrome)
        return MinWeightBound(False, representative, lower_bound), num_states

    def _get_word(self, symbols: npt.NDArray[np.int64]) -> int:
        """Pack symbols into a word, in which the values 0, 1, and 2 of a symbol are X, Z, and Y.

        The symbols of an X or Z part all have the value 0.
        """
        num_variables = self._search.num_variables
        word = 0
        for symbol in symbols.tolist():
            value, variable = divmod(symbol, num_variables)
            if value != 1:
                word |= 1 << variable
            if value:
                word |= 1 << (num_variables + variable)
        return word
