# SPDX-License-Identifier: Apache-2.0

"""Connected-cluster coset searches of ``CosetLeaderSearch``, with checks made sparser first."""

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
    ``get_min_weight_bound`` gives these lower bounds on the weight of the words of a coset, which
    are often weaker than those of a ``BrouwerZimmermannCosetSearch``:

    - The weight of the word, for a coset with a word of weight zero or one, which a table finds.
    - Two, for any other coset whose word the search finds within its first weight limit, which can
      exceed the weight of the word.
    - The weight limit, for a coset whose word the search finds within a later limit, which is then
      the weight of the lightest words.
    - The larger of two and ``max_weight + 1``, for a coset without a word within ``max_weight``.

    So from the first limit two, every coset with a word within ``max_weight`` gets the weight of
    its lightest words.

    Args:
        matrix: The checks, whose kernel is the code: the stabilizers, in ``[X|Z]`` order, for
            Paulis, or the rows that check the X or Z parts.  Bit ``j`` of the syndrome of a word
            is its symplectic product with row ``j`` for Paulis, and its dot product for parts.
        symplectic: True for Paulis, and False for X or Z parts.
        max_rounds: The ``max_sparsifying_rounds`` of the tuning (see ``get_sparser_rows``).
        rounds_without_improvement: The ``sparsifying_rounds_without_improvement`` of the tuning.
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.uint8],
        *,
        symplectic: bool,
        max_rounds: int,
        rounds_without_improvement: int,
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
        self._num_checks = len(rows)
        # Row ``i`` of ``rows`` sums the given rows that row ``i`` of ``transform`` selects, so a
        # given syndrome maps to the sum of the columns of ``transform`` that its bits select.
        self._syndrome_map_columns = row_ints(np.ascontiguousarray(transform.T))
        # Without a function that finds automorphisms, the search never asks for them.
        self._search = SupportSearch(
            rows,
            np.zeros((0, matrix.shape[1]), dtype=np.uint8),
            symplectic=symplectic,
            automorphism_minimum_states=_tuning.MAX_VALUE,
        )
        # For ``rows``, word ``j`` of ``get_unit_syndrome_words`` has a syndrome of only bit ``j``,
        # so the words of the set bits of a syndrome sum to a representative of its coset.
        syndrome_rows = swap_pauli_halves(rows) if symplectic else rows
        self._inverse_rows = row_ints(get_unit_syndrome_words(syndrome_rows))

    def get_min_weight_bound(
        self,
        syndrome: int,
        max_weight: int,
        *,
        first_limit: int | None = None,
    ) -> MinWeightBound:
        """Decide whether the coset of this syndrome has a word of weight at most ``max_weight``.

        A negative ``max_weight`` admits no word, as for a ``BrouwerZimmermannCosetSearch``, and the
        class docstring states the lower bounds of the answers.  Without ``first_limit``, the search
        tries only the weight limit ``max_weight``.  Given ``first_limit``, it tries each limit from
        ``first_limit``, raised to two and lowered to ``max_weight``, through ``max_weight``, and
        stops at the first within which it finds a word.  Increasing limits can find light words
        sooner (Section 3.8 of ``connected_cluster.md``), and a first limit above two suits callers
        to which any word within it gives the same answer.
        """
        # Look up a word of weight zero or one.
        word = self._singles.get(syndrome)
        if word is not None:
            weight = int(word != 0)
            return MinWeightBound(weight <= max_weight, word, weight)

        # Map the syndrome to the sparser checks, which are independent, so the search keeps them
        # all, in order.  The map is invertible and the lookup answers the zero syndrome, so the
        # mapped syndrome is not zero, as ``_find_vector_with_packed_syndrome`` requires.
        syndrome = xor_selected(self._syndrome_map_columns, syndrome)
        packed_syndrome = int_to_words(syndrome, (self._num_checks + 63) // 64)

        # The lookup leaves only cosets without a word below weight two, so search no limit below
        # two, and raise the lower bound above each limit within which the search finds no word.
        lower_bound = 2
        first = max_weight if first_limit is None else min(first_limit, max_weight)
        for limit in range(max(2, first), max_weight + 1):
            symbols = self._search._find_vector_with_packed_syndrome(packed_syndrome, limit)
            if symbols is not None:
                return MinWeightBound(True, self._get_word(symbols), lower_bound)
            lower_bound = limit + 1
        representative = xor_selected(self._inverse_rows, syndrome)
        return MinWeightBound(False, representative, lower_bound)

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
