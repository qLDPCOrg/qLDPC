# SPDX-License-Identifier: Apache-2.0

"""Exact minimum-weight queries for syndrome classes of Pauli errors on a stabilizer state.

The Pauli errors with a given syndrome form a coset of the stabilizer group, and a lightest error in
it is a coset leader.  ``CosetLeaderSearch`` decides whether the coset leaders of a syndrome class
weigh at most a given maximum weight, with Brouwer-Zimmermann searches of cosets of binary codes, or
with connected-support searches that start from the syndrome.  For the residual-weight estimator of
``qldpc.experimental``, it also finds the weight of the coset leaders of each syndrome class up to a
maximum weight.

This module is the entry point of these queries.  It splits a query into coset searches, which live
with their methods.  The BZ coset searches are in ``_brouwer_zimmermann/coset.py``, and the tables
of their steps and tails in ``_brouwer_zimmermann/coset_tables.py``.  The connected-support coset
searches are in ``_connected_support/coset.py``.  What both methods share is in ``_cosets.py``.
"""

from __future__ import annotations

import functools
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_kernel, pack_bits, row_reduce_binary
from qldpc._compilation import always_compiled

from ._binary import as_binary_matrix, assert_binary
from ._brouwer_zimmermann.coset import (
    BrouwerZimmermannCosetSearch,
    get_css_searches,
    get_pauli_search,
)
from ._connected_support.coset import ConnectedSupportCosetSearch
from ._cosets import MinWeightBound, row_ints, words_to_int, xor_selected, xor_selected_rows
from ._tuning import get_tuning

# The method of a coset-leader search.
CosetSearchMethod = Literal["brouwer_zimmermann", "connected_support"]


####################################################################################################
# coset-leader minimum-weight bounds


class CosetLeaderSearch:
    """Minimum-weight oracle for Pauli errors modulo the stabilizers of a pure stabilizer state.

    The syndrome of a Pauli error on ``n`` qubits says which of the ``n`` given stabilizer
    generators, the rows of ``stabilizers``, it anticommutes with.  Two errors have the same
    syndrome exactly when they differ by a stabilizer, so the errors with a given syndrome form a
    syndrome class of equivalent errors, which is a coset of the stabilizer group.  A lightest error
    of a syndrome class is a coset leader, and the oracle determines whether the coset leaders of a
    syndrome class weigh at most a given maximum weight.

    A Pauli is packed into an integer with its X part ``x`` in the low ``n`` bits and its Z part
    ``z`` in the next ``n`` bits, and a syndrome is packed into an integer with bit ``j`` for
    stabilizer ``j``.  The weight of an error is either its physical weight, the number of qubits
    that it acts on, or its CSS weight ``max(|x|, |z|)``.

    The oracle answers a query with one or two coset searches.  By default, each searches the cosets
    of a binary code with the Brouwer-Zimmermann (BZ) algorithm (see ``brouwer_zimmermann.md``), as
    ``_search_coset`` in ``_brouwer_zimmermann/coset.py`` describes, and ``get_min_weight_bound`` of
    each coset search first looks up words of weight at most one in a table.

    - In general, one Pauli search covers the Pauli errors themselves, modulo the stabilizers.
      Its syndromes are for the given stabilizers, so it is cached for the most recently used
      stabilizer matrices (see ``_tuning.PAULI_SEARCH_CACHE_SIZE``).
    - For the CSS weight on a CSS state, two classical searches cover the X parts and the Z parts
      of the errors separately, on ``n`` columns each rather than ``2 n``.  A pure-X stabilizer has
      no Z part, a pure-Z stabilizer has no X part, and a stabilizer state is CSS if pure-X and
      pure-Z stabilizers generate its stabilizer group.  The X parts of the pure-X stabilizers span
      a binary code ``C_X``, and the Z parts of the pure-Z stabilizers a binary code ``C_Z``.
      Multiplying an error by a stabilizer adds a word of ``C_X`` to its X part and a word of
      ``C_Z`` to its Z part, independently, so the smallest CSS weight of an error with a given
      syndrome is the larger of the smallest Hamming weights in the cosets ``x + C_X`` and
      ``z + C_Z``.  The coset of an X part is labeled by its syndrome for the pure-Z stabilizers,
      and that of a Z part by its syndrome for the pure-X stabilizers, and the oracle maps a
      syndrome for the given stabilizers to these two syndromes.  The two searches depend only on
      the stabilizer group, so they are cached for the most recently used groups (see
      ``_tuning.CSS_SEARCH_CACHE_SIZE``), and oracles for different generating sets of one state
      share them.

    With ``method="connected_support"``, each coset search is instead the search of
    ``SupportSearch`` in ``_connected_support/search.py`` from the syndrome of a query, through the
    unsatisfied stabilizers for Pauli errors, the pure-Z stabilizers for X parts, and the pure-X
    stabilizers for Z parts (Section 3.8 of ``connected_support.md``).  It is fast if its checks are
    sparse.  Given stabilizers, such as row-reduced ones, can be dense, so it first replaces its
    checks with sparser ones that span the same space, which can fall short of the sparsest
    generators.  The CSS weight is not a sum over qubits, so it weighs errors by CSS weight only on
    CSS states, whose X and Z parts it searches separately.  Like the BZ searches, it first looks up
    errors of weight at most one, so a syndrome with such an error gets the weight of its lightest
    errors as a lower bound.  For any other syndrome, a query with an error within ``max_weight``
    gives two, and a query with none gives the larger of two and ``max_weight + 1``.
    ``find_first_not_within_max_weight`` searches each syndrome that the table lacks with increasing
    weight limits, which can find light errors sooner than one search at ``max_weight``.  The BZ
    searches often give higher bounds.

    The oracle reads the active tuning when it is built, and the module docstring of ``_tuning``
    says what a tuning can change about its answers.  The field ``disjoint_random_column_orders``
    configures every BZ search of the oracle.  The field ``max_css_single_pairs`` configures the BZ
    searches by CSS weight that are not split into two classical searches.  The fields
    ``overlapping_minimum_work``, ``max_information_sets``, and ``overlapping_attempts_per_set``
    configure the families of information sets with which a BZ search answers costly queries (see
    ``_brouwer_zimmermann/coset_routes.py``).  The threshold ``overlapping_minimum_work`` also
    decides which queries of the BZ searches by physical weight, and of the classical searches,
    weigh words without the pivot columns of each set of their family, and skip the words that
    tables of bounds prove to weigh more than ``max_weight``.  The fields
    ``coset_completion_block_bits`` and ``completion_bound_max_table_bytes`` configure these tables.
    The fields ``max_sparsifying_rounds`` and ``sparsifying_rounds_without_improvement`` configure
    how the connected-support searches make their checks sparser.

    Args:
        stabilizers: A binary ``n x 2n`` matrix in ``[X|Z]`` order whose rows are independent
            commuting stabilizers.
        css_weight: True to weigh an error by its CSS weight, or False to weigh it by its physical
            weight.
        method: ``"brouwer_zimmermann"`` (the default) or ``"connected_support"``, the search of
            the cosets.

    Raises:
        ValueError: If the stabilizers are not ``n`` independent commuting stabilizers, if the
            method is unknown, or if the connected-support method would weigh errors on a state that
            is not CSS by their CSS weight.
    """

    def __init__(
        self,
        stabilizers: npt.ArrayLike,
        *,
        css_weight: bool = False,
        method: CosetSearchMethod = "brouwer_zimmermann",
    ) -> None:
        if method not in get_args(CosetSearchMethod):
            raise ValueError(
                f"Unknown coset-leader search method {method!r}; choose from 'brouwer_zimmermann'"
                " or 'connected_support'"
            )
        tuning = get_tuning()
        # The connected-support searches make their checks sparser in rounds that the tuning bounds.
        support_search = functools.partial(
            ConnectedSupportCosetSearch,
            max_rounds=tuning.max_sparsifying_rounds,
            rounds_without_improvement=tuning.sparsifying_rounds_without_improvement,
        )
        # Exactly one of the BZ searches and the connected-support searches is nonempty.
        self._brouwer_zimmermann_searches: tuple[BrouwerZimmermannCosetSearch, ...] = ()
        self._connected_support_searches: tuple[ConnectedSupportCosetSearch, ...] = ()
        self.css_weight = css_weight
        assert_binary(stabilizers, "stabilizers")
        matrix = as_binary_matrix(stabilizers, "stabilizers")
        self.num_qubits = len(matrix)
        n = self.num_qubits
        if not n or matrix.shape[1] != 2 * n:
            raise ValueError("Require a complete nonempty n by 2n stabilizer matrix")
        # The syndrome of a packed Pauli ``e`` is ``checks @ e``, where ``checks`` swaps the X and Z
        # halves of the stabilizers, and the stabilizers commute exactly when ``checks @ matrix.T``
        # is zero.
        checks = np.roll(matrix, n, axis=1)
        if len(row_reduce_binary(matrix)[2]) != n or np.any((checks @ matrix.T) % 2):
            raise ValueError("Stabilizers must be independent and commute")

        # Search the Pauli errors themselves, unless the CSS weight on a CSS state lets the X and Z
        # parts of the errors be searched separately.
        combinations = _get_css_combinations(matrix) if css_weight else None
        if combinations is None:
            if method == "brouwer_zimmermann":
                self._brouwer_zimmermann_searches = (get_pauli_search(matrix, css_weight, tuning),)
            elif css_weight:
                raise ValueError(
                    "The connected-support method weighs errors by CSS weight only on CSS states;"
                    " use css_weight=False or the default method"
                )
            else:
                self._connected_support_searches = (support_search(matrix, homogeneous=False),)
            return

        # Combine the stabilizers into pure-X and pure-Z stabilizers, and row-reduce the X parts of
        # the pure-X stabilizers and the Z parts of the pure-Z stabilizers.  The reduced rows depend
        # only on the stabilizer group, so they key the cache of BZ searches.
        x_combinations, z_combinations = combinations
        x_rows, x_transform, _ = row_reduce_binary(
            (x_combinations @ matrix[:, :n]) % 2, np.eye(len(x_combinations), dtype=np.uint8)
        )
        z_rows, z_transform, _ = row_reduce_binary(
            (z_combinations @ matrix[:, n:]) % 2, np.eye(len(z_combinations), dtype=np.uint8)
        )
        assert x_transform is not None and z_transform is not None
        if method == "connected_support":
            # The Z parts of the pure-Z stabilizers check the X parts, and the X parts of the pure-X
            # stabilizers check the Z parts.
            self._connected_support_searches = (
                support_search(z_rows, homogeneous=True),
                support_search(x_rows, homogeneous=True),
            )
        else:
            self._brouwer_zimmermann_searches = get_css_searches(x_rows, z_rows, tuning)

        # Row ``j`` of ``z_rows`` is the Z part of a pure-Z stabilizer, the product of the given
        # stabilizers with coefficients ``c = (z_transform @ z_combinations)[j]``.  An error's
        # syndrome bit for that stabilizer is the parity of its X part with the row, and also the
        # parity of its syndrome for the given stabilizers with ``c``.  So the rows ``c`` form the X
        # syndrome map, from a syndrome for the given stabilizers to the syndrome of the X part, and
        # likewise for Z.  Each map is stored as its packed columns, one per bit of a syndrome for
        # the given stabilizers.
        x_syndrome_map = (z_transform @ z_combinations) % 2
        z_syndrome_map = (x_transform @ x_combinations) % 2
        self._map_columns = (
            pack_bits(np.ascontiguousarray(x_syndrome_map.T)),
            pack_bits(np.ascontiguousarray(z_syndrome_map.T)),
        )

        # A single syndrome is mapped by XORing the columns of both maps for its set bits, which is
        # cheaper than a parity for each row, since most syndromes have few set bits.  Entry ``i``
        # holds column ``i`` of the X map in its low ``_num_x_syndrome_bits`` bits, and column ``i``
        # of the Z map above them.
        self._num_x_syndrome_bits = len(z_rows)
        self._joint_map_columns = row_ints(
            np.ascontiguousarray(np.vstack([x_syndrome_map, z_syndrome_map]).T)
        )

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether some error with this syndrome has weight at most ``max_weight``.

        Args:
            syndrome: A packed syndrome (see ``CosetLeaderSearch``).
            max_weight: The largest weight that the query accepts.

        Returns:
            A ``MinWeightBound``.  With two searches, its lower bound is the larger of theirs.

        Raises:
            ValueError: If the syndrome is out of range.
        """
        if not 0 <= syndrome < 1 << self.num_qubits:
            raise ValueError("The syndrome is out of range")
        searches = self._connected_support_searches or self._brouwer_zimmermann_searches
        if len(searches) == 1:
            return searches[0].get_min_weight_bound(syndrome, max_weight)
        x_syndrome, z_syndrome = self._split_syndrome(syndrome)
        return self._join_part_bounds(
            searches[0].get_min_weight_bound(x_syndrome, max_weight),
            searches[1].get_min_weight_bound(z_syndrome, max_weight),
        )

    def find_first_not_within_max_weight(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> tuple[int, MinWeightBound | None]:
        """Find the first syndrome whose class has no error of weight at most ``max_weight``.

        Args:
            syndromes: One packed syndrome (see ``CosetLeaderSearch``) per row, split into 64-bit
                machine words, with machine word ``j`` holding bits ``64 j`` and up.  Each syndrome
                must be below ``2**num_qubits``.  The rows may have fewer words than ``num_qubits``
                bits need, if every syndrome fits in them.
            max_weight: The largest weight that the query accepts.

        Returns:
            The index of that syndrome, or the number of syndromes if there is none, and a
            ``MinWeightBound`` for that syndrome, or None if there is none.  With one BZ search,
            the bound is the one that the batch search found, so the syndrome is not searched
            again.  With two BZ searches, it is the one that ``get_min_weight_bound`` gives.  With
            the connected-support method, it has the lower bound that ``get_min_weight_bound``
            gives.  If the X and Z parts are searched separately, the representative of a part
            within ``max_weight`` can differ, since the batch searches with increasing weight
            limits (see ``_get_batch_support_bound``).

        Raises:
            ValueError: If a syndrome is out of range.
        """
        self._check_syndromes(syndromes)

        # The connected-support searches answer one syndrome at a time.
        if self._connected_support_searches:
            for index, row in enumerate(syndromes):
                bound = self._get_batch_support_bound(words_to_int(row), max_weight)
                if not bound.within_max_weight:
                    return index, bound
            return len(syndromes), None

        # With one search, take its bound and the lightest error that it weighed.
        if len(self._brouwer_zimmermann_searches) == 1:
            return self._brouwer_zimmermann_searches[0].find_first_not_within_max_weight(
                syndromes, max_weight
            )

        # With two searches, a syndrome's class fails, that is, has no error of weight at most
        # ``max_weight``, exactly when the coset of its X part or that of its Z part does.  So find
        # the first syndrome whose X-part coset fails, and then search the Z-part cosets of the
        # syndromes before it.  The bound for the failing syndrome needs the representatives of both
        # cosets, which ``get_min_weight_bound`` finds.
        index = len(syndromes)
        for search, map_columns in zip(self._brouwer_zimmermann_searches, self._map_columns):
            part_syndromes = _map_syndromes(syndromes[:index], map_columns)
            index, _ = search.find_first_not_within_max_weight(part_syndromes, max_weight)
        if index == len(syndromes):
            return index, None
        return index, self.get_min_weight_bound(words_to_int(syndromes[index]), max_weight)

    def _get_min_weights(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> npt.NDArray[np.int64]:
        """Find the minimum weight of the class of each syndrome, if it is at most ``max_weight``.

        The residual-weight estimator of ``qldpc.experimental`` asks for these weights.  The BZ
        searches find the minimum weight of each coset in one search, and answer the syndromes in
        one compiled loop for each coset search (see
        ``BrouwerZimmermannCosetSearch.get_min_weights``).  The connected-support searches answer
        one syndrome at a time, by trying increasing weight limits (see
        ``_get_min_support_weight``).

        Args:
            syndromes: Packed syndromes, as for ``find_first_not_within_max_weight``.
            max_weight: The largest minimum weight to find.

        Returns:
            The minimum weight of the class of each syndrome, or ``max_weight + 1`` if that is
            larger.

        Raises:
            ValueError: If a syndrome is out of range.
        """
        self._check_syndromes(syndromes)

        # Every class weighs more than a negative maximum weight.
        if max_weight < 0:
            return np.full(len(syndromes), max_weight + 1, dtype=np.int64)

        # The connected-support searches answer one syndrome at a time.
        if self._connected_support_searches:
            return np.array(
                [self._get_min_support_weight(words_to_int(row), max_weight) for row in syndromes],
                dtype=np.int64,
            )

        # With one search, take its weights.
        floors = np.zeros(len(syndromes), dtype=np.int64)
        if len(self._brouwer_zimmermann_searches) == 1:
            return self._brouwer_zimmermann_searches[0].get_min_weights(
                syndromes, max_weight, floors
            )

        # With two searches, the smallest CSS weight of an error in a class is the larger of the
        # minimum weights of its X part and its Z part.  So search the X parts, and then the Z parts
        # of the classes whose X parts are within ``max_weight``.  Any Z part within the weight of
        # its X part gives that weight, so that weight is the floor of its search.
        x_search, z_search = self._brouwer_zimmermann_searches
        x_syndromes = _map_syndromes(syndromes, self._map_columns[0])
        weights = x_search.get_min_weights(x_syndromes, max_weight, floors)
        rows = np.flatnonzero(weights <= max_weight)
        z_syndromes = _map_syndromes(syndromes[rows], self._map_columns[1])
        z_weights = z_search.get_min_weights(z_syndromes, max_weight, weights[rows])
        weights[rows] = np.maximum(weights[rows], z_weights)
        return weights

    def _get_min_support_weight(self, syndrome: int, max_weight: int) -> int:
        """Find the minimum weight of the class of a syndrome with the connected-support searches.

        The search of the Pauli errors or of the X part tries increasing weight limits from two, so
        it finds a lightest word of its coset, if within ``max_weight``, and gets its weight as its
        lower bound (see ``ConnectedSupportCosetSearch.get_min_weight_bound``).  With two searches,
        the search of the Z part starts at a weight limit equal to the weight of the X part, within
        which any Z part gives the CSS weight of the class.

        Returns:
            The minimum weight of the class, or ``max_weight + 1`` if that is larger.
        """
        searches = self._connected_support_searches
        if len(searches) == 1:
            bound = searches[0].get_min_weight_bound(syndrome, max_weight, first_limit=2)
            return bound.lower_bound if bound.within_max_weight else max_weight + 1
        x_syndrome, z_syndrome = self._split_syndrome(syndrome)
        x_bound = searches[0].get_min_weight_bound(x_syndrome, max_weight, first_limit=2)
        if not x_bound.within_max_weight:
            return max_weight + 1
        z_bound = searches[1].get_min_weight_bound(
            z_syndrome, max_weight, first_limit=x_bound.lower_bound
        )
        if not z_bound.within_max_weight:
            return max_weight + 1
        return max(x_bound.lower_bound, z_bound.lower_bound)

    def _get_batch_support_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Answer a query of a batch with the connected-support searches, by increasing limits.

        Most syndromes of a batch have an error within ``max_weight``, which a search at a lower
        weight limit finds sooner (see ``ConnectedSupportCosetSearch.get_min_weight_bound``).
        ``find_first_not_within_max_weight`` checks the range of every syndrome first
        (``_check_syndromes``).
        """
        searches = self._connected_support_searches
        if len(searches) == 1:
            return searches[0].get_min_weight_bound(syndrome, max_weight, first_limit=2)
        x_syndrome, z_syndrome = self._split_syndrome(syndrome)
        return self._join_part_bounds(
            searches[0].get_min_weight_bound(x_syndrome, max_weight, first_limit=2),
            searches[1].get_min_weight_bound(z_syndrome, max_weight, first_limit=2),
        )

    def _check_syndromes(self, syndromes: npt.NDArray[np.uint64]) -> None:
        """Raise a ``ValueError`` unless each packed syndrome of a batch is below ``2**num_qubits``.

        The compiled searches read a word with a unit syndrome for each set bit of a syndrome, and
        do not check the range of the bit.  So a batch checks its rows first, in the machine word
        that holds bit ``num_qubits``, if any, and in the machine words above it.
        """
        num_words, num_bits = divmod(self.num_qubits, 64)
        high_words = syndromes[:, num_words:]
        if high_words.size and (
            np.any(high_words[:, 0] >> np.uint64(num_bits)) or np.any(high_words[:, 1:])
        ):
            raise ValueError("A syndrome of the batch is out of range")

    def _split_syndrome(self, syndrome: int) -> tuple[int, int]:
        """Map a syndrome for the given stabilizers to the syndromes of the parts of its errors."""
        part_syndromes = xor_selected(self._joint_map_columns, syndrome)
        x_syndrome = part_syndromes & ((1 << self._num_x_syndrome_bits) - 1)
        return x_syndrome, part_syndromes >> self._num_x_syndrome_bits

    def _join_part_bounds(self, x_bound: MinWeightBound, z_bound: MinWeightBound) -> MinWeightBound:
        """Join the bounds of the cosets of X parts and of Z parts into a bound of a syndrome class.

        The smallest CSS weight of an error in the class is the larger of the smallest weights of
        the parts.
        """
        return MinWeightBound(
            x_bound.within_max_weight and z_bound.within_max_weight,
            x_bound.representative | (z_bound.representative << self.num_qubits),
            max(x_bound.lower_bound, z_bound.lower_bound),
        )


def _get_css_combinations(
    stabilizers: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]] | None:
    """Combine stabilizers into pure-X and pure-Z generators, or return None if they are not CSS.

    The stabilizers must be independent.  The combinations of them whose Z parts cancel, which form
    the null space of the transposed Z parts, give the pure-X stabilizers, and likewise for pure-Z
    stabilizers.  A pure-X and a pure-Z stabilizer are equal only if both are the identity, so these
    stabilizers are independent, and they generate the group exactly when there are ``n`` of them.

    Returns:
        Matrices ``(x_combinations, z_combinations)``, whose products with ``stabilizers`` modulo 2
        are pure-X and pure-Z stabilizers that together generate the group, or None.
    """
    num_qubits = len(stabilizers)
    x_combinations = get_kernel(np.ascontiguousarray(stabilizers[:, num_qubits:].T))
    z_combinations = get_kernel(np.ascontiguousarray(stabilizers[:, :num_qubits].T))
    if len(x_combinations) + len(z_combinations) != num_qubits:
        return None
    return x_combinations, z_combinations


####################################################################################################
# compiled syndrome maps


@always_compiled
def _map_syndromes(
    syndromes: npt.NDArray[np.uint64], map_columns: npt.NDArray[np.uint64]
) -> npt.NDArray[np.uint64]:
    """Map packed syndromes by a binary matrix, given by its packed columns.

    Row ``i`` of the result is the XOR of the columns ``map_columns[b]`` for the set bits ``b`` of
    row ``i`` of ``syndromes``, which is the product of the matrix with that syndrome.  The
    syndromes are split into 64-bit words, and each set bit must be below the number of columns;
    this is not checked.
    """
    mapped = np.zeros((len(syndromes), map_columns.shape[1]), dtype=np.uint64)
    for row in range(len(syndromes)):
        xor_selected_rows(syndromes[row], map_columns, mapped[row])
    return mapped
