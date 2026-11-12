# SPDX-License-Identifier: Apache-2.0

"""Exact minimum-weight queries for syndrome classes of Pauli errors on a stabilizer state.

The Pauli errors with a given syndrome form a coset of the stabilizer group, and a lightest error in
it is a coset leader.  ``CosetLeaderSearch`` decides whether the coset leaders of a syndrome class
weigh at most a given maximum weight, with Brouwer-Zimmermann searches of cosets of binary codes, or
with connected-support searches that start from the syndrome (see ``connected_support.py``).  For
the residual-weight estimator of ``qldpc.experimental``, it also finds the weight of the coset
leaders of each syndrome class up to a maximum weight.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, NamedTuple, get_args

import numpy as np
import numpy.typing as npt

from qldpc._binary import (
    get_kernel,
    pack_bits,
    popcount_word,
    row_reduce_binary,
    trailing_zeros_word,
    unpack_bits,
)
from qldpc._compilation import always_compiled

from .. import _tuning
from .._binary import as_binary_matrix, assert_binary, symplectic_to_hamming
from .._tuning import SearchTuning, get_tuning
from . import (
    affine_instance,
    affine_search,
    completion_bounds,
    connected_support,
    coset_routes,
    information_sets,
)

# The method of a coset-leader search.
CosetSearchMethod = Literal["brouwer_zimmermann", "connected_support"]

# The seed of the random orders of columns in which connected-support coset searches row-reduce
# their checks to make them sparser (see ``_get_sparser_rows``).  It makes the checks reproducible.
_SPARSIFYING_SEED = 0

# The planning buffer of a coset search: its tables of the bounds of its steps, each with one more
# row that caches a stopping step (see ``_get_planning``).
_PlanningBuffer = npt.NDArray[np.int64]

# The fields of a ``_CosetTails`` record, which the compiled searches take as the record itself or
# as a plain tuple.
_TailFields = tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint8],
    npt.NDArray[np.int64],
    int,
]

####################################################################################################
# coset-leader minimum-weight bounds


@dataclass(frozen=True)
class MinWeightBound:
    """The answer to a query about the minimum weight of a coset.

    A query of ``CosetLeaderSearch`` asks whether a syndrome class of Pauli errors, which is a coset
    of the stabilizer group, has an error of weight at most ``max_weight``.  A query of one of its
    coset searches asks the same about a coset of the code that the search covers.  Which element
    and which lower bound a query returns can depend on the method and tuning of the search, and on
    the version of qLDPC, within the guarantees below.

    Attributes:
        within_max_weight: Whether the coset has an element of weight at most ``max_weight``.
        representative: An element of the coset: a packed Pauli (see ``CosetLeaderSearch``) for a
            query of the oracle, and a word of the search (a packed Pauli, or an X or Z part) for a
            query of a coset search.  If ``within_max_weight`` is True, its weight is at most
            ``max_weight``.
        lower_bound: A lower bound on the minimum weight of the coset.  If ``within_max_weight`` is
            False, it exceeds ``max_weight``.
    """

    within_max_weight: bool
    representative: int
    lower_bound: int


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

    The oracle answers a query with one or two coset searches.  By default, each searches the
    cosets of a binary code with the Brouwer-Zimmermann (BZ) algorithm (see
    ``brouwer_zimmermann.md``), as ``_search_coset`` describes, and ``get_min_weight_bound`` first
    looks up words of weight at most one in a table.

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
    ``connected_support.SupportSearch`` from the syndrome of a query, through the unsatisfied
    stabilizers for Pauli errors, the pure-Z stabilizers for X parts, and the pure-X stabilizers
    for Z parts (Section 3.8 of ``connected_support.md``).  It is fast if its checks are sparse.
    Given stabilizers, such as row-reduced ones, can be dense, so it first replaces its checks with
    sparser ones that span the same space, which can fall short of the sparsest generators.  Which
    checks it finds depends on the tuning, and can change which error a query returns, but not its
    answer or its lower bound.  The CSS weight is not a sum over qubits, so it weighs errors by CSS
    weight only on CSS states, whose X and Z parts it searches separately.  Like the BZ
    searches, it first looks up errors of weight at most one, so a syndrome with such an error gets
    the weight of its lightest errors as a lower bound.  For any other syndrome, a query with an
    error within ``max_weight`` gives two, and a query with none gives the larger of two and
    ``max_weight + 1``.  ``find_first_not_within_max_weight`` searches each syndrome that the table
    lacks with increasing weight limits, which can find light errors sooner than one search at
    ``max_weight``.  The BZ searches often give higher bounds.

    The oracle reads the active tuning (see ``_tuning``) when it is built.  Its field
    ``disjoint_random_column_orders`` configures every BZ search of the oracle.  The field
    ``max_css_single_pairs`` configures the BZ searches by CSS weight that are not split into two
    classical searches.  The fields ``overlapping_minimum_work``, ``max_information_sets``, and
    ``overlapping_attempts_per_set`` configure the families of information sets with which a BZ
    search answers costly queries (see ``coset_routes``).  These families can change the
    representatives and lower bounds of the answers, but never whether an answer is within
    ``max_weight``.  The threshold ``overlapping_minimum_work`` also decides which queries of the BZ
    searches by physical weight, and of the classical searches, weigh words without the pivot
    columns of each set of their family, and skip the words that tables of bounds prove to weigh
    more than ``max_weight`` (see ``coset_routes``).  The fields ``coset_completion_block_bits`` and
    ``completion_bound_max_table_bytes`` configure these tables.  Skipping words can change the
    representatives and lower bounds of the answers without a word within ``max_weight``, but never
    whether an answer is within it.  The fields ``max_sparsifying_rounds`` and
    ``sparsifying_rounds_without_improvement`` configure how the connected-support searches make
    their checks sparser.

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
            _ConnectedSupportCosetSearch,
            max_rounds=tuning.max_sparsifying_rounds,
            rounds_without_improvement=tuning.sparsifying_rounds_without_improvement,
        )
        # Exactly one of the BZ searches and the connected-support searches is nonempty.
        self._searches: tuple[_CosetSearch, ...] = ()
        self._support_searches: tuple[_ConnectedSupportCosetSearch, ...] = ()
        self.css_weight = css_weight
        assert_binary(stabilizers, "stabilizers")
        matrix = as_binary_matrix(stabilizers)
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
                self._searches = (
                    _get_pauli_search(
                        _StabilizerKey(matrix, checks),
                        css_weight,
                        _CosetTuning.from_tuning(tuning, paulis_by_css_weight=css_weight),
                    ),
                )
            elif css_weight:
                raise ValueError(
                    "The connected-support method weighs errors by CSS weight only on CSS states;"
                    " use css_weight=False or the default method"
                )
            else:
                self._support_searches = (support_search(matrix, homogeneous=False),)
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
            self._support_searches = (
                support_search(z_rows, homogeneous=True),
                support_search(x_rows, homogeneous=True),
            )
        else:
            self._searches = _get_css_searches(
                n,
                len(x_rows),
                pack_bits(x_rows).tobytes(),
                pack_bits(z_rows).tobytes(),
                _CosetTuning.from_tuning(tuning, paulis_by_css_weight=False),
            )

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
        self._joint_map_columns = _row_ints(
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
        searches = self._support_searches or self._searches
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
        if self._support_searches:
            for index, row in enumerate(syndromes):
                bound = self._get_batch_support_bound(_packed_row_int(row), max_weight)
                if not bound.within_max_weight:
                    return index, bound
            return len(syndromes), None

        # With one search, take its bound and the lightest error that it weighed.
        if len(self._searches) == 1:
            return self._searches[0].find_first_not_within_max_weight(syndromes, max_weight)

        # With two searches, a syndrome's class fails, that is, has no error of weight at most
        # ``max_weight``, exactly when the coset of its X part or that of its Z part does.  So find
        # the first syndrome whose X-part coset fails, and then search the Z-part cosets of the
        # syndromes before it.  The bound for the failing syndrome needs the representatives of both
        # cosets, which ``get_min_weight_bound`` finds.
        index = len(syndromes)
        for search, map_columns in zip(self._searches, self._map_columns):
            part_syndromes = _map_syndromes(syndromes[:index], map_columns)
            index, _ = search.find_first_not_within_max_weight(part_syndromes, max_weight)
        if index == len(syndromes):
            return index, None
        return index, self.get_min_weight_bound(_packed_row_int(syndromes[index]), max_weight)

    def _get_min_weights(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> npt.NDArray[np.int64]:
        """Find the minimum weight of the class of each syndrome, if it is at most ``max_weight``.

        The residual-weight estimator of ``qldpc.experimental`` asks for these weights.  The BZ
        searches find the minimum weight of each coset in one search, and answer the syndromes in
        one compiled loop for each coset search (see ``_CosetSearch.get_min_weights``).  The
        connected-support searches answer one syndrome at a time, by trying increasing weight
        limits (see ``_get_min_support_weight``).

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
        if self._support_searches:
            return np.array(
                [
                    self._get_min_support_weight(_packed_row_int(row), max_weight)
                    for row in syndromes
                ],
                dtype=np.int64,
            )

        # With one search, take its weights.
        floors = np.zeros(len(syndromes), dtype=np.int64)
        if len(self._searches) == 1:
            return self._searches[0].get_min_weights(syndromes, max_weight, floors)

        # With two searches, the smallest CSS weight of an error in a class is the larger of the
        # minimum weights of its X part and its Z part.  So search the X parts, and then the Z parts
        # of the classes whose X parts are within ``max_weight``.  Any Z part within the weight of
        # its X part gives that weight, so that weight is the floor of its search.
        x_search, z_search = self._searches
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
        lower bound (see ``_ConnectedSupportCosetSearch.get_min_weight_bound``).  With two searches,
        the search of the Z part starts at a weight limit equal to the weight of the X part, within
        which any Z part gives the CSS weight of the class.

        Returns:
            The minimum weight of the class, or ``max_weight + 1`` if that is larger.
        """
        searches = self._support_searches
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
        weight limit finds sooner (see ``_ConnectedSupportCosetSearch.get_min_weight_bound``).
        ``find_first_not_within_max_weight`` checks the range of every syndrome first
        (``_check_syndromes``).
        """
        searches = self._support_searches
        if len(searches) == 1:
            return searches[0].get_min_weight_bound(syndrome, max_weight, first_limit=2)
        x_syndrome, z_syndrome = self._split_syndrome(syndrome)
        return self._join_part_bounds(
            searches[0].get_min_weight_bound(x_syndrome, max_weight, first_limit=2),
            searches[1].get_min_weight_bound(z_syndrome, max_weight, first_limit=2),
        )

    def _check_syndromes(self, syndromes: npt.NDArray[np.uint64]) -> None:
        """Raise a ``ValueError`` unless each packed syndrome of a batch is below ``2**num_qubits``.

        The compiled searches read a row of the right inverse of their checks for each set bit of a
        syndrome, and do not check the range of the bit.  So a batch checks its rows first, in the
        machine word that holds bit ``num_qubits``, if any, and in the words above it.
        """
        num_words, num_bits = divmod(self.num_qubits, 64)
        high_words = syndromes[:, num_words:]
        if high_words.size and (
            np.any(high_words[:, 0] >> np.uint64(num_bits)) or np.any(high_words[:, 1:])
        ):
            raise ValueError("A syndrome of the batch is out of range")

    def _split_syndrome(self, syndrome: int) -> tuple[int, int]:
        """Map a syndrome for the given stabilizers to the syndromes of the parts of its errors."""
        part_syndromes = _xor_selected(self._joint_map_columns, syndrome)
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


class _CosetSearchTables(NamedTuple):
    """The tables of ``_search_coset`` for one coset search of a ``CosetLeaderSearch``.

    The search runs on embedded words.  For a search of Pauli errors, the embedded word of an error
    is its packed Pauli for the CSS weight, and its packed Pauli followed by ``x xor z`` for the
    physical weight (see ``symplectic_to_hamming``).  For a search of X or Z parts, the embedded
    word of a part is the part itself.

    Attributes:
        inverse: Row ``j`` is the packed embedded word of a word whose syndrome has only bit ``j``
            set.
        generators: The packed rows of each set of a family of information sets of the embedded
            generators of the code (see ``coset_routes``).  The last set of a disjoint family may
            be partial, of rank below the dimension of the code.
        pivots: The pivot columns of each set, padded with -1 beyond its rank.
        planning: The planning buffer of the search with this family: its tables of the bounds of
            its steps, each with one more row that caches a stopping step (see ``_get_planning``).
            For a nonnegative maximum weight, ``_get_stopping_step`` reuses the cached step of a
            table exactly while it stays the first step whose bound exceeds that weight, and
            replaces it otherwise.  Queries write only the rows of cached steps, and oracles that
            share a cached search share them.
        x_mask: Selects the bits of a packed embedded word that count as its X part, with one
            machine word for each machine word of the embedded words.  Only searches of Pauli
            errors by CSS weight use the X part, and their layers scan with it, but every search
            reads the mask when it weighs the word of a syndrome, and a search of whole words also
            reads it when it weighs the offsets of layer zero.
    """

    inverse: npt.NDArray[np.uint64]
    generators: npt.NDArray[np.uint64]
    pivots: npt.NDArray[np.int64]
    planning: _PlanningBuffer
    x_mask: npt.NDArray[np.uint64]


class _CosetTails(NamedTuple):
    """The tails with which ``_search_coset`` weighs the words of a family by tails, and its tables.

    The tail of an embedded word on a set of the family holds every column of the word but the
    pivot columns of the set, including columns on which no row of the set has a one.  The tails are
    packed, and padded with zeros to one common width, that of the set of least rank.  A word of a
    coset is the offset of a set, which is zero on its pivot columns, plus rows of the set.  Each
    row below the rank of the set has a one on its own pivot column, where the other rows are zero.
    So the word weighs the number of its rows below the rank plus the weight of its tail (see
    Section 3.5 of ``brouwer_zimmermann.md``).

    Attributes:
        tails: The tails of the rows of each set, with shape ``(num_sets, num_rows, num_words)``.
        inverse_tails: The tails of the rows of ``inverse`` of ``_CosetSearchTables`` on each set,
            with shape ``(num_sets, num_syndrome_bits, num_words)``.  The tail of the word of a
            syndrome on a set is the sum of the rows for the set bits of the syndrome.  These rows
            can be derived from ``inverse``, but a query that sums them directly finds the tails of
            its offsets faster than one that gathers the columns of each tail from the full word.
        completion_tables: The completion tables of the tails of the sets that have them, one after
            another (see ``completion_bounds.CompletionTables``).
        layout: Where the tables of each set lie in ``completion_tables``, with a row for each set:
            the start of its tables, their number of blocks ``b``, and their ``b + 1`` block
            offsets, padded with zeros.  A set without tables has zero blocks.  The compiled search
            takes the layout of every set in one array, since each argument of a compiled call adds
            to the time of every query.
        block_bits: The number of columns of the blocks of the tables.
    """

    tails: npt.NDArray[np.uint64]
    inverse_tails: npt.NDArray[np.uint64]
    completion_tables: npt.NDArray[np.uint8]
    layout: npt.NDArray[np.int64]
    block_bits: int


class _CosetSearch:
    """One coset search of a ``CosetLeaderSearch``: a search of the cosets of one binary code.

    The words of a search are either packed Paulis on ``n`` qubits, in which case the code is
    spanned by the packed stabilizers, or the X parts (or Z parts) of Paulis, in which case the code
    is spanned by the X parts of the pure-X stabilizers (or the Z parts of the pure-Z stabilizers).
    Each coset is labeled by the syndrome of its words, whose bit ``j`` is the parity of a word with
    row ``j`` of a matrix of checks.

    The search answers queries of each maximum weight with the tables of a family of information
    sets of the code, which ``routes`` choose (see ``coset_routes``).  The tables of every family
    have the same ``inverse`` and ``x_mask``.  The queries of a route by tails weigh the words of
    its family by their tails, and skip completions by the completion tables, from its record of
    tails.

    Args:
        routes: The routes of the search.
        css_weight: True to weigh words by their CSS weight, which is the Hamming weight of an X
            or Z part, or False to weigh them by their physical weight.
        num_qubits: The number of qubits ``n``.
        inverse_rows: Row ``j`` of a right inverse of the checks, not embedded, as an integer: a
            word whose syndrome has only bit ``j`` set.
        check_columns: Column ``c`` of the checks as an integer: the syndrome of the word with only
            bit ``c`` set.
        is_pauli: Whether the words are packed Paulis of ``2 n`` bits, rather than X or Z parts of
            ``n`` bits.
        max_css_single_pairs: For a search of Pauli errors by CSS weight, the most pairs of distinct
            syndromes of single-qubit X and Z errors that it tabulates (see ``_css_singles``).
    """

    def __init__(
        self,
        routes: coset_routes.CosetRoutes[_CosetSearchTables, _CosetTails],
        css_weight: bool,
        num_qubits: int,
        inverse_rows: tuple[int, ...],
        check_columns: tuple[int, ...],
        *,
        is_pauli: bool,
        max_css_single_pairs: int = 0,
    ) -> None:
        self._routes = routes
        self._max_css_single_pairs = max_css_single_pairs
        self.css_weight = css_weight
        # The compiled search weighs embedded words by their CSS weight only in a search of Pauli
        # errors by CSS weight, and otherwise by their Hamming weight, which is twice the physical
        # weight of an error, and the weight of an X or Z part (see ``_search_coset``).  The scale
        # chooses which of these weights a scan computes.  The weight factor of the coset instances
        # is separate: it is two for every search of Pauli errors, including those by CSS weight,
        # whose scale is zero, since an error weighs at least half the Hamming weight of its
        # embedded word (see ``_get_pauli_search``).
        self.scale = 0 if css_weight and is_pauli else 1 if css_weight else 2
        self._num_qubits = num_qubits
        self._is_pauli = is_pauli
        self._inverse_rows = inverse_rows
        self._num_syndrome_bits = len(inverse_rows)
        # A query below maximum weight two bounds a coset without words of weight at most one by
        # two.  If the search is of X or Z parts and rounds to parity (see ``_get_search_tables``),
        # every word of an odd coset has odd weight, so such an odd coset has no word of weight two
        # either, and the query bounds it by three.  An odd coset of a search of Pauli errors by
        # CSS weight can still hold an error of weight two, such as ``YX``, whose binary word has
        # three ones.
        num_tables = len(routes.disjoint_tables.planning)
        self._odd_coset_lookup_bound = 3 if not is_pauli and num_tables == 2 else 2

        # Map each syndrome of a word of weight zero or one to such a word.  For the CSS weight,
        # ``_weight_one_word`` also pairs the single-qubit X errors with the single-qubit Z errors,
        # to find ``X_i Z_j`` on distinct qubits.
        self._singles, self._x_singles, self._z_singles = _tabulate_singles(
            check_columns, num_qubits, is_pauli=is_pauli
        )

    def get_min_weight_bound(self, syndrome: int, max_weight: int) -> MinWeightBound:
        """Decide whether the coset of this syndrome has a word of weight at most ``max_weight``.

        Returns:
            A ``MinWeightBound`` for the coset, whose representative is a word of the coset.
        """
        word = self._weight_one_word(syndrome)
        if word is not None:
            weight = self._weight(word)
            return MinWeightBound(weight <= max_weight, word, weight)

        # Otherwise the coset has no word of weight at most one, so two is a lower bound, and so is
        # the odd-coset lookup bound for an odd coset.
        if max_weight < 2:
            word = _xor_selected(self._inverse_rows, syndrome)
            return MinWeightBound(
                False, word, self._odd_coset_lookup_bound if word.bit_count() % 2 else 2
            )

        # Search the coset with the tables of the route of the maximum weight, by tails on a route
        # by tails, and keep the lightest word that the search weighs in ``best``.
        tables, tails = self._get_route(max_weight)
        best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
        packed_syndrome = _int_to_words(syndrome, max(1, (self._num_syndrome_bits + 63) // 64))
        lower_bound = _search_coset(packed_syndrome, max_weight, best, *tables, tails, self.scale)
        # A search that finds a light enough word returns zero, but every word of the coset weighs
        # at least two, since the coset has no word of weight at most one.
        return MinWeightBound(lower_bound <= max_weight, self.unembed(best), max(2, lower_bound))

    def find_first_not_within_max_weight(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> tuple[int, MinWeightBound | None]:
        """Find the first syndrome whose coset has no word of weight at most ``max_weight``.

        The syndromes are packed as for ``CosetLeaderSearch.find_first_not_within_max_weight``.
        The search answers them with the route of the maximum weight, which it takes once.

        Returns:
            The index of that syndrome, or the number of syndromes if there is none, and a
            ``MinWeightBound`` for its coset, with the lightest word that the search weighed, or
            None if there is none.
        """
        tables, tails = self._get_route(max_weight)
        best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
        index, lower_bound = _find_first_not_within_max_weight(
            syndromes, max_weight, best, *tables, tails, self.scale
        )
        if index == len(syndromes):
            return index, None
        return index, MinWeightBound(False, self.unembed(best), int(lower_bound))

    def get_min_weights(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int, floors: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        """Find the minimum weight of the coset of each syndrome, if it is at most ``max_weight``.

        The syndromes are packed as for ``CosetLeaderSearch.find_first_not_within_max_weight``.  The
        search of a syndrome also ends at a word of weight at most its floor in ``floors``, which
        must be at most ``max_weight``.  No coset has a minimum weight above the number ``n`` of
        qubits, so the search answers the syndromes with the route of the lesser of ``max_weight``
        and ``n``, which it takes once.  Unlike ``get_min_weight_bound``, it looks up no words of
        weight at most one, since the first layers of a search find them cheaply.

        Returns:
            For each syndrome, the weight of a word of its coset of weight at most its floor if
            there is one, and otherwise the minimum weight of its coset, or ``max_weight + 1`` if
            that is larger (see ``_search_coset_for_min_weight``).
        """
        max_weight = min(max_weight, self._num_qubits)
        tables, tails = self._get_route(max_weight)
        return _find_min_weights(syndromes, max_weight, floors, *tables, tails, self.scale)

    def unembed(self, embedded_word: npt.NDArray[np.uint64]) -> int:
        """Convert a packed embedded word into a word, without the columns that the embedding adds.

        The embedding for the physical weight appends ``x xor z`` to a packed Pauli (see
        ``_CosetSearchTables``), and the other embeddings add nothing.
        """
        return _packed_row_int(embedded_word) & ((1 << self._num_word_bits) - 1)

    def _get_route(self, max_weight: int) -> tuple[_CosetSearchTables, _TailFields | None]:
        """The tables of the route of a maximum weight, and the fields of its tails, if any.

        The fields are a plain tuple, which numba types at each call of a compiled search several
        times faster than the named tuple ``_CosetTails``.
        """
        tables, tails = self._routes.get_route(max_weight)
        return tables, None if tails is None else tails[:]

    @property
    def _num_word_bits(self) -> int:
        """The number of bits of a word: ``2 n`` for Paulis, and ``n`` for X or Z parts."""
        return 2 * self._num_qubits if self._is_pauli else self._num_qubits

    def _weight(self, word: int) -> int:
        """The weight of a word: of its Pauli, or the Hamming weight of an X or Z part."""
        if not self._is_pauli:
            return word.bit_count()
        mask = (1 << self._num_qubits) - 1
        x, z = word & mask, (word >> self._num_qubits) & mask
        return max(x.bit_count(), z.bit_count()) if self.css_weight else (x | z).bit_count()

    def _weight_one_word(self, syndrome: int) -> int | None:
        """Find a word of weight zero or one with this syndrome, if there is one."""
        if syndrome in self._singles:
            return self._singles[syndrome]
        if not (self._is_pauli and self.css_weight):
            return None
        # CSS weight one also allows X_i Z_j on distinct qubits.  Callers such as the certification
        # of fault propagation make a query for each of thousands of syndromes, so a table of these
        # errors, built on the first query that needs it, saves a scan of the qubits per query.
        if self._css_singles is not None:
            return self._css_singles.get(syndrome)
        # If that table would be too large, match the syndrome of each single-qubit X error against
        # those of the single-qubit Z errors, in time linear in the number of qubits.
        for sx, x in self._x_singles.items():
            z = self._z_singles.get(syndrome ^ sx)
            if z is not None:
                return x ^ z
        return None

    @functools.cached_property
    def _css_singles(self) -> dict[int, int] | None:
        """Errors X_i Z_j (with X_i or Z_j possibly the identity) by syndrome, if not too many."""
        return _tabulate_css_singles(self._x_singles, self._z_singles, self._max_css_single_pairs)


class _StabilizerKey:
    """A stabilizer matrix and its checks, which a cache can hash and compare by the matrix.

    The checks follow from the stabilizers, so the stabilizers alone key the cache, and the key
    keeps both matrices to build a search that the cache has not stored yet.  Keys of packed bytes
    would not hold the matrices, but unpacking them would cost about 0.4% of the setup of a search.
    The matrices may be arrays of the caller.  The key compares a copy of the bytes of the
    stabilizers, and the search copies what it keeps of them, so later changes to the arrays of the
    caller reach neither.
    """

    def __init__(self, stabilizers: npt.NDArray[np.uint8], checks: npt.NDArray[np.uint8]) -> None:
        self.stabilizers = stabilizers
        self.checks = checks
        self._key = (stabilizers.shape, stabilizers.tobytes())
        self._hash = hash(self._key)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _StabilizerKey) and self._key == other._key


class _CosetTuning(NamedTuple):
    """The fields of the tuning that the setup of a BZ coset search reads, which key its cache.

    ``disjoint_random_column_orders`` configures the disjoint information sets of the search.  Only
    a search of Pauli errors by CSS weight reads ``max_css_single_pairs``, for its lookups.  The
    fields from ``overlapping_minimum_work`` to ``overlapping_attempts_per_set`` configure the
    families with which the search answers costly queries, and ``overlapping_minimum_work`` also
    chooses its routes by tails, whose queries weigh words by their tails and skip completions by
    tables, if it weighs Hamming weights (see ``coset_routes``).  The last two fields configure
    those tables, which the search of Pauli errors by CSS weight does not have.  The search builds
    the families, tails, and tables after the cache stores it.
    """

    disjoint_random_column_orders: int
    max_css_single_pairs: int
    overlapping_minimum_work: int
    max_information_sets: int
    overlapping_attempts_per_set: int
    completion_bound_max_table_bytes: int
    coset_completion_block_bits: int

    @classmethod
    def from_tuning(cls, tuning: SearchTuning, *, paulis_by_css_weight: bool) -> _CosetTuning:
        """The fields of a tuning that the setup of a search reads.

        ``paulis_by_css_weight`` is True for a search of Pauli errors by CSS weight.  Only that
        search reads ``max_css_single_pairs``, and it alone has no completion tables.  The fields
        that a search does not read are zero, so that oracles built under tunings that differ only
        in those fields share cached searches.
        """
        fields = (getattr(tuning, name) for name in cls._fields)
        coset_tuning = cls(*fields)
        if paulis_by_css_weight:
            return coset_tuning._replace(
                completion_bound_max_table_bytes=0, coset_completion_block_bits=0
            )
        return coset_tuning._replace(max_css_single_pairs=0)


class _ConnectedSupportCosetSearch:
    """A coset search of a ``CosetLeaderSearch`` by the connected-support search of its checks.

    The words, checks, and syndromes are those of a ``_CosetSearch``: packed Paulis, whose syndrome
    for the stabilizers is that of the symplectic search of ``connected_support.SupportSearch``, or
    the X or Z parts of Paulis, whose syndrome is that of a binary search of the checks.  The search
    starts from the syndrome of a query, and grows supports through its unsatisfied checks (Section
    3.8 of ``connected_support.md``).

    Like a ``_CosetSearch``, it first looks up a word of weight zero or one by its syndrome: a
    single variable, or a single-qubit Pauli.  It searches only cosets with no such word, so a coset
    with such a word gets the weight of that word as a lower bound.  Any other coset gets two if it
    has a word within ``max_weight``.  With increasing weight limits from two (see
    ``get_min_weight_bound``), it instead gets the weight of its lightest words.  Without a word
    within ``max_weight``, it gets the larger of two and ``max_weight + 1``.  These bounds are often
    weaker than those of a ``_CosetSearch``.

    The search is fast only if its checks are sparse, but the given checks, such as row-reduced
    stabilizers, can be dense.  So it searches the sparser checks of ``_get_sparser_rows``, which
    span the same space, and maps the syndrome of a query to them.

    Args:
        matrix: The checks, whose kernel is the code: the stabilizers, in ``[X|Z]`` order, for
            Paulis, or the rows that check the X or Z parts.  Bit ``j`` of the syndrome of a word
            is its symplectic product with row ``j`` for Paulis, and its dot product for parts.
        homogeneous: False for Paulis, and True for X or Z parts.
        max_rounds: The ``max_sparsifying_rounds`` of the tuning (see ``_get_sparser_rows``).
        rounds_without_improvement: The ``sparsifying_rounds_without_improvement`` of the tuning.
    """

    def __init__(
        self,
        matrix: npt.NDArray[np.uint8],
        *,
        homogeneous: bool,
        max_rounds: int,
        rounds_without_improvement: int,
    ) -> None:
        # Map each syndrome of a word of weight zero or one to such a word.  The syndrome of a word
        # with one nonzero bit is a column of the checks, once the halves of symplectic checks are
        # swapped.
        num_variables = matrix.shape[1] if homogeneous else matrix.shape[1] // 2
        syndrome_matrix = matrix if homogeneous else np.roll(matrix, num_variables, axis=1)
        columns = _row_ints(np.ascontiguousarray(syndrome_matrix.T))
        self._singles, _, _ = _tabulate_singles(columns, num_variables, is_pauli=not homogeneous)

        rows, transform = _get_sparser_rows(
            matrix,
            homogeneous=homogeneous,
            max_rounds=max_rounds,
            rounds_without_improvement=rounds_without_improvement,
        )
        self._num_checks = len(rows)
        # Row ``i`` of the sparse rows sums the given rows that row ``i`` of the transform selects,
        # so bit ``i`` of a syndrome for them is the parity of those bits of the given syndrome.
        # Entry ``j`` of the map is column ``j`` of the transform, which a given syndrome bit adds.
        self._syndrome_map_columns = _row_ints(np.ascontiguousarray(transform.T))
        # Without a function that finds automorphisms, the search never asks for them.
        self._search = connected_support.SupportSearch(
            rows,
            np.zeros((0, matrix.shape[1]), dtype=np.uint8),
            homogeneous=homogeneous,
            automorphism_minimum_states=_tuning.MAX_VALUE,
        )
        # Row ``j`` of the inverse is a word whose syndrome has only bit ``j`` set, so the words of
        # the set bits of a syndrome sum to a representative of its coset.
        syndrome_rows = rows if homogeneous else np.roll(rows, num_variables, axis=1)
        self._inverse_rows = _row_ints(_get_inverse(syndrome_rows))

    def get_min_weight_bound(
        self,
        syndrome: int,
        max_weight: int,
        *,
        first_limit: int | None = None,
    ) -> MinWeightBound:
        """Decide whether the coset of this syndrome has a word of weight at most ``max_weight``.

        A negative ``max_weight`` admits no word, as for a ``_CosetSearch``.

        A search with weight limit ``max_weight`` can visit many heavy supports before it finds a
        word that is much lighter.  Given ``first_limit``, the search instead tries every limit from
        ``first_limit`` to ``max_weight``, and stops at the first limit within which it finds a
        word.  It tries no limit below two, which the lookup of words of weight at most one makes
        useless, and lowers a first limit above ``max_weight`` to it.  From the first limit two, it
        finds a lightest word.  This can save time on cosets with words much lighter than
        ``max_weight``, and repeats the work of the lower limits on cosets without a word within
        ``max_weight``.  A first limit above two suits callers to which any word within it gives the
        same answer.  A word that the search finds at that limit can be lighter than the limit, so
        its coset gets the lower bound two.
        """
        # Look up a word of weight zero or one.
        word = self._singles.get(syndrome)
        if word is not None:
            weight = int(word != 0)
            return MinWeightBound(weight <= max_weight, word, weight)

        # Map the syndrome to the sparse checks.  The map is invertible, and the lookup above
        # answers the zero syndrome, so the mapped syndrome is not zero.  The sparse checks are
        # independent, so the search keeps all of them, in order, and the mapped syndrome has one
        # bit for each check that it keeps.  So the search can start from the packed syndrome,
        # without the validation of ``find_vector_with_syndrome``, which would cost more than the
        # search itself in many queries.
        syndrome = _xor_selected(self._syndrome_map_columns, syndrome)
        packed_syndrome = _int_to_words(syndrome, (self._num_checks + 63) // 64)

        # The lookup leaves only cosets without a word below weight two, so no limit below two is
        # searched, and a single query searches only at ``max_weight``.  Each limit within which the
        # search finds no word raises the lower bound above that limit.
        lower_bound = 2
        first = max_weight if first_limit is None else min(first_limit, max_weight)
        for limit in range(max(2, first), max_weight + 1):
            symbols = self._search._find_vector_with_packed_syndrome(packed_syndrome, limit)
            if symbols is not None:
                return MinWeightBound(True, self._get_word(symbols), lower_bound)
            lower_bound = limit + 1
        representative = _xor_selected(self._inverse_rows, syndrome)
        return MinWeightBound(False, representative, lower_bound)

    def _get_word(self, symbols: npt.NDArray[np.int64]) -> int:
        """Pack the symbols of a support into a word.

        A symbol gives the qubit ``symbol % n`` the Pauli X, Z, or Y for values 0, 1, and 2.  The
        symbols of an X or Z part have only the value 0.
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


@functools.lru_cache(maxsize=_tuning.PAULI_SEARCH_CACHE_SIZE)
def _get_pauli_search(key: _StabilizerKey, css_weight: bool, tuning: _CosetTuning) -> _CosetSearch:
    """The Pauli search, of Pauli errors modulo the stabilizers, with syndromes for the stabilizers.

    ``key`` holds the stabilizers and their checks (see ``CosetLeaderSearch``).  The syndromes
    depend on the rows of the stabilizer matrix, not only on the group that they generate, so the
    rows key the cache.  ``css_weight`` and the fields of the tuning that the setup reads key it
    too.
    """
    stabilizers, checks = key.stabilizers, key.checks
    num_qubits = len(stabilizers)
    inverse = _get_inverse(checks)
    embedded_stabilizers = stabilizers if css_weight else symplectic_to_hamming(stabilizers)
    embedded_inverse = inverse if css_weight else symplectic_to_hamming(inverse)
    # An error weighs at least half of the Hamming weight of its embedded word, by either measure of
    # weight, so the certificate bounds are halved.  A search by CSS weight never weighs words by
    # tails, since that weight needs to know which deleted pivot columns are in the X part.
    routes = _get_routes(
        embedded_stabilizers,
        embedded_inverse,
        num_qubits,
        weight_factor=2,
        tuning=tuning,
        by_tails=not css_weight,
    )
    check_columns = _row_ints(np.ascontiguousarray(checks.T))
    return _CosetSearch(
        routes,
        css_weight,
        num_qubits,
        _row_ints(inverse),
        check_columns,
        is_pauli=True,
        max_css_single_pairs=tuning.max_css_single_pairs,
    )


@functools.lru_cache(maxsize=_tuning.CSS_SEARCH_CACHE_SIZE)
def _get_css_searches(
    num_qubits: int, num_x_rows: int, x_rows: bytes, z_rows: bytes, tuning: _CosetTuning
) -> tuple[_CosetSearch, _CosetSearch]:
    """The classical searches of the X parts and of the Z parts of the errors on a CSS state.

    ``x_rows`` holds the X parts of ``num_x_rows`` pure-X stabilizers, and ``z_rows`` the Z parts of
    the other ``num_qubits - num_x_rows`` stabilizers, which are pure-Z.  Both are in reduced row
    echelon form, which depends only on the stabilizer group, and packed by ``pack_bits`` into
    bytes, which ``lru_cache`` can hash.  The X parts of errors are searched modulo the code of the
    X rows, with the Z rows as checks, and the Z parts the other way around (see
    ``CosetLeaderSearch``).  The fields of the tuning that the setup reads key the cache too.
    """
    num_words = max(1, (num_qubits + 63) // 64)
    x_matrix = unpack_bits(
        np.frombuffer(x_rows, dtype=np.uint64).reshape(num_x_rows, num_words), num_qubits
    )
    z_matrix = unpack_bits(
        np.frombuffer(z_rows, dtype=np.uint64).reshape(num_qubits - num_x_rows, num_words),
        num_qubits,
    )
    return (
        _get_classical_search(x_matrix, z_matrix, tuning),
        _get_classical_search(z_matrix, x_matrix, tuning),
    )


def _get_classical_search(
    generators: npt.NDArray[np.uint8], checks: npt.NDArray[np.uint8], tuning: _CosetTuning
) -> _CosetSearch:
    """The classical search of X or Z parts modulo the row space of independent generators.

    The row space of the generators is the null space of the independent checks, so each coset is
    labeled by its syndrome for the checks.  Words are weighed by their Hamming weight, which the BZ
    certificate bounds directly, so its bounds are not halved.
    """
    num_qubits = generators.shape[1]
    inverse = _get_inverse(checks)
    # Every column counts as the X part, so the CSS weight of a word is its Hamming weight.
    routes = _get_routes(
        generators, inverse, num_qubits, weight_factor=1, tuning=tuning, by_tails=True
    )
    check_columns = _row_ints(np.ascontiguousarray(checks.T))
    return _CosetSearch(routes, True, num_qubits, _row_ints(inverse), check_columns, is_pauli=False)


def _get_routes(
    generators: npt.NDArray[np.uint8],
    inverse: npt.NDArray[np.uint8],
    num_x_columns: int,
    *,
    weight_factor: int,
    tuning: _CosetTuning,
    by_tails: bool,
) -> coset_routes.CosetRoutes[_CosetSearchTables, _CosetTails]:
    """The routes of a coset search of the code of independent embedded generators.

    The routes start from the coset instance of the disjoint information sets of the generators,
    with the weight factor ``weight_factor`` (see ``affine_instance.get_coset_instance``).  The
    tables of every family of the routes come from ``_get_search_tables``, with ``inverse`` and
    ``num_x_columns``.  If ``by_tails`` is True, the routes whose queries are predicted to examine
    at least ``tuning.overlapping_minimum_work`` words with their own family are routes by tails,
    with records of tails and tables from ``_get_coset_tails``, and otherwise no route is.
    """
    instance = affine_instance.get_coset_instance(
        generators,
        random_column_orders=tuning.disjoint_random_column_orders,
        weight_factor=weight_factor,
    )
    return coset_routes.CosetRoutes(
        generators,
        instance,
        functools.partial(_get_search_tables, inverse=inverse, num_x_columns=num_x_columns),
        functools.partial(
            _get_coset_tails,
            inverse=inverse,
            block_bits=tuning.coset_completion_block_bits,
            max_table_bytes=tuning.completion_bound_max_table_bytes,
        )
        if by_tails
        else None,
        minimum_work=tuning.overlapping_minimum_work,
        max_sets=tuning.max_information_sets,
        attempts_per_set=tuning.overlapping_attempts_per_set,
    )


def _get_inverse(checks: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """A right inverse of independent checks: row ``j`` has a syndrome of only bit ``j``."""
    _, labels, pivots = row_reduce_binary(checks, np.eye(len(checks), dtype=np.uint8))
    assert labels is not None
    inverse = np.zeros(checks.shape, dtype=np.uint8)
    inverse[:, pivots] = labels.T
    return inverse


def _get_sparser_rows(
    matrix: npt.NDArray[np.uint8],
    *,
    homogeneous: bool,
    max_rounds: int,
    rounds_without_improvement: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Rows with the span of independent rows, made sparser by row reductions in random orders.

    The rows start as those of ``_sparsify``, and are weighed as it weighs them.  Each round
    row-reduces the given rows with pivots in a random order of the columns, which gives the words
    of the span that have exactly one pivot column in their support.  Those that are lighter than
    the heaviest current row join the rows as candidates, and the greedy algorithm for matroids
    keeps the lightest independent candidates: it takes them by increasing weight, and keeps each
    that is independent of those that it kept.  So the rows are always a lightest basis drawn from
    the candidates, and their total weight never rises.  The rounds stop after ``max_rounds``, or
    once ``rounds_without_improvement`` rounds in a row have left the total weight unchanged, and a
    last pass of ``_sparsify`` adds rows to each other where that helps.  The rows need not be the
    sparsest.  Rows with the same span give a search the same answers and bounds, though perhaps
    other representatives.  So the rounds may be random, and a fixed seed makes them reproducible.

    Args:
        matrix: Independent rows.
        homogeneous: False if the rows are symplectic, and True otherwise.
        max_rounds: The most rounds, or zero for the rows of ``_sparsify``.
        rounds_without_improvement: The number of rounds in a row without a lower total weight after
            which the rounds stop.

    Returns:
        The rows, and the invertible transform ``T``, so that the rows are ``T @ matrix % 2``.  The
        total weight of the rows is at most that of the rows of ``_sparsify``.
    """
    rows, transform = _sparsify(matrix, homogeneous=homogeneous)
    if not len(rows):
        return rows, transform

    # Each round adds the rows of a reduction that are lighter than the heaviest current row.  The
    # rows come first among candidates of equal weight, so a round that does not lower the total
    # weight leaves the rows as they are.  Row-reducing the transposed candidates, sorted by weight,
    # takes as pivots the first candidates that are independent of those before them.
    rng = np.random.default_rng(_SPARSIFYING_SEED)
    weights = _get_row_weights(rows, homogeneous=homogeneous)
    rounds_unchanged = 0
    for _ in range(max_rounds):
        reduced, _, _ = row_reduce_binary(matrix, columns=rng.permutation(matrix.shape[1]))
        reduced_weights = _get_row_weights(reduced, homogeneous=homogeneous)
        light = reduced_weights < weights.max()
        previous_total = weights.sum()
        if np.any(light):
            candidates = np.vstack([rows, reduced[light]])
            candidate_weights = np.concatenate([weights, reduced_weights[light]])
            order = np.argsort(candidate_weights, kind="stable")
            _, _, pivots = row_reduce_binary(np.ascontiguousarray(candidates[order].T))
            rows, weights = candidates[order[pivots]], candidate_weights[order[pivots]]
        rounds_unchanged = rounds_unchanged + 1 if weights.sum() == previous_total else 0
        if rounds_unchanged >= rounds_without_improvement:
            break

    # Add rows to each other where that helps, and find the transform from the given rows.  The
    # rows are in the span of ``matrix``, whose product with the transposed inverse is the identity,
    # so their product with it is the transform.  Products of binary matrices in float32 are exact
    # while their inner dimension is below ``2**24``, and faster than in integers.
    rows, _ = _sparsify(rows, homogeneous=homogeneous)
    inverse = _get_inverse(matrix)
    product = rows.astype(np.float32) @ inverse.T.astype(np.float32)
    return rows, (product % 2).astype(np.uint8)


def _sparsify(
    matrix: npt.NDArray[np.uint8], *, homogeneous: bool
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Rows with the span of the given rows, made sparser by adding rows to each other.

    The weight of a row is its number of nonzero entries, or, if the rows are symplectic, the
    number of qubits on which its X part ``[:n]`` or its Z part ``[n:]`` is nonzero.  While adding
    another row to a row lowers its weight, the row takes the sum that is lightest.  Every addition
    lowers the total weight, so this ends.

    Args:
        matrix: The rows.
        homogeneous: False if the rows are symplectic, and True otherwise.

    Returns:
        The rows, and the invertible transform ``T``, so that the rows are ``T @ matrix % 2``.
    """
    # Pack the X and Z parts, or the rows and zeros, so that a weight counts the nonzero qubits.
    num_rows, width = matrix.shape
    num_variables = width if homogeneous else width // 2
    part_x = pack_bits(np.ascontiguousarray(matrix[:, :num_variables]))
    part_z = (
        np.zeros_like(part_x)
        if homogeneous
        else pack_bits(np.ascontiguousarray(matrix[:, num_variables:]))
    )
    transform = np.eye(num_rows, dtype=np.uint8)
    weights = np.bitwise_count(part_x | part_z).sum(axis=1, dtype=np.int64)

    # Replace each row by its lightest sum with another row, while that lowers its weight.  A row
    # added to itself would vanish, so its own entry keeps its weight, which never wins.
    improved = True
    while improved:
        improved = False
        for row in range(num_rows):
            sums_x, sums_z = part_x ^ part_x[row], part_z ^ part_z[row]
            sum_weights = np.bitwise_count(sums_x | sums_z).sum(axis=1, dtype=np.int64)
            sum_weights[row] = weights[row]
            other = int(np.argmin(sum_weights))
            if sum_weights[other] < weights[row]:
                part_x[row], part_z[row] = sums_x[other], sums_z[other]
                transform[row] ^= transform[other]
                weights[row] = sum_weights[other]
                improved = True

    parts = [unpack_bits(part_x, num_variables)]
    if not homogeneous:
        parts.append(unpack_bits(part_z, num_variables))
    return np.hstack(parts), transform


def _get_row_weights(rows: npt.NDArray[np.uint8], *, homogeneous: bool) -> npt.NDArray[np.int64]:
    """The weights of rows, as ``_sparsify`` counts them: of nonzero entries, or of qubits."""
    if homogeneous:
        return rows.sum(axis=1, dtype=np.int64)
    half = rows.shape[1] // 2
    return (rows[:, :half] | rows[:, half:]).sum(axis=1, dtype=np.int64)


def _get_search_tables(
    instance: affine_instance.AffineInstance, inverse: npt.NDArray[np.uint8], num_x_columns: int
) -> _CosetSearchTables:
    """Build the tables of ``_search_coset`` for the cosets of the code of a coset instance.

    ``inverse`` holds embedded words whose syndromes have one bit set, and the first
    ``num_x_columns`` columns count as the X part of an embedded word.

    The search rounds its bounds to the parity of each coset if every word of the code has even
    Hamming weight and some coset is odd.  Then the words of each coset share a parity (see
    ``get_coset_instance``).  The word of a syndrome sums rows of ``inverse``, so some coset is odd
    exactly when some row is.  If the code and every row of ``inverse`` are even, every coset is
    even.  This happens only in the search of Pauli errors by physical weight, whose bounds are
    halved, so that rounding them up to even would change none.
    """
    width = instance.generators.shape[2]
    x_mask = _int_to_words((1 << num_x_columns) - 1, width)

    # Round to the parity of each coset if the code is even and some coset is odd.  The rows of any
    # one information set span the code.
    even_code = not np.any(np.bitwise_count(instance.generators[0]).sum(axis=1) % 2)
    has_odd_coset = bool(np.any(np.count_nonzero(inverse, axis=1) % 2))
    parities = (0, 1) if even_code and has_odd_coset else (None,)
    return _CosetSearchTables(
        pack_bits(inverse),
        instance.generators,
        instance.pivots,
        _get_planning([instance.get_weight_bounds(parity) for parity in parities]),
        x_mask,
    )


def _get_coset_tails(
    instance: affine_instance.AffineInstance,
    sets: list[int],
    record: _CosetTails | None,
    *,
    inverse: npt.NDArray[np.uint8],
    block_bits: int,
    max_table_bytes: int,
) -> _CosetTails:
    """Build the record of the tails of the family of a coset instance, with tables of some sets.

    ``inverse`` holds embedded words whose syndromes have one bit set, with every column of the
    embedded words.  The tails of each set keep every column but its pivots, and are zero-padded to
    the width of the set of least rank, which keeps the most columns.  A new record has tables on
    blocks of ``block_bits`` columns.  Given the ``record`` of the family, the function keeps its
    tails, tables, and blocks rather than build them again.  The record that it returns also has the
    completion tables of each set of ``sets``, unless they would exceed ``max_table_bytes`` bytes or
    be empty (see ``completion_bounds.build_tables_within``).  It holds the tables of each set once,
    and is ``record`` itself if it adds no tables.
    """
    # Build the tails of the family once, with a layout of no tables.
    if record is None:
        num_sets, num_rows, width = instance.generators.shape
        num_columns = inverse.shape[1]
        rows = unpack_bits(instance.generators.reshape(-1, width), num_columns)
        rows = rows.reshape(num_sets, num_rows, num_columns)
        tail_width = (num_columns - int(instance.ranks.min()) + 63) // 64
        tails = np.zeros((num_sets, num_rows, tail_width), dtype=np.uint64)
        inverse_tails = np.zeros((num_sets, len(inverse), tail_width), dtype=np.uint64)
        for set_index, (pivots, rank) in enumerate(
            zip(instance.pivots, instance.ranks, strict=True)
        ):
            columns = np.delete(np.arange(num_columns), pivots[:rank])
            set_width = (len(columns) + 63) // 64
            tails[set_index, :, :set_width] = pack_bits(rows[set_index][:, columns])
            inverse_tails[set_index, :, :set_width] = pack_bits(inverse[:, columns])
        # The tables of a set have at most one block for each block_bits columns of its tails, and
        # block_bits divides the 64 bits of a word.
        max_blocks = 64 * tail_width // block_bits
        layout = np.zeros((num_sets, max_blocks + 3), dtype=np.int64)
        record = _CosetTails(tails, inverse_tails, np.zeros(0, np.uint8), layout, block_bits)

    # Append the tables of each set that has none yet, if they fit, and lay them out.  The routes
    # keep only the latest record of a family, so the earlier tables are held once.
    layout = record.layout.copy()
    added = []
    start = len(record.completion_tables)
    for set_index in sets:
        if layout[set_index, 1]:
            continue
        tables = completion_bounds.build_tables_within(
            record.tails[set_index], record.block_bits, max_table_bytes
        )
        if tables is not None:
            num_blocks = len(tables.offsets) - 1
            layout[set_index, : num_blocks + 3] = [start, num_blocks, *tables.offsets]
            added.append(tables.tables)
            start += len(tables.tables)
    if not added:
        return record
    return record._replace(
        completion_tables=np.concatenate([record.completion_tables, *added]), layout=layout
    )


def _get_planning(bound_tables: Sequence[npt.NDArray[np.int64]]) -> _PlanningBuffer:
    """Build the planning buffer of a coset search from tables of the bounds of its steps.

    The compiled search takes the bounds and its cached stopping steps in this one array, since each
    argument of a compiled call adds to the time of every query.  The buffer stacks one table for
    each given table of bounds.  A search that rounds its bounds to parity (see
    ``_get_search_tables``) gives two tables: table ``p`` holds the bounds of the cosets whose words
    have Hamming weights of parity ``p`` (see ``_get_planning_table``).  Other searches give one
    table, for every coset.  Each table of the buffer has one more row than the bounds:

    - Every row but the last holds the bounds of ``AffineInstance.get_weight_bounds``: entry
      ``[w, j]`` bounds the weight of every word that the search has not visited once the first
      ``j`` sets have completed layer ``w`` and the others layer ``w - 1``.  So column zero of each
      row holds the bound before the first step of that row.  These rows never change, and code
      reads them as the view ``table[:-1]``.
    - The last row holds no bounds.  Its first two entries are the layer and the number of completed
      sets of the stopping step that ``_get_stopping_step`` returned last for this table, or of the
      first step, ``(0, 1)``, before its first call, and its other entries are zero.  Only
      ``_get_stopping_step`` reads and writes this row.
    """
    planning = np.zeros(
        (len(bound_tables), len(bound_tables[0]) + 1, bound_tables[0].shape[1]), dtype=np.int64
    )
    planning[:, :-1] = bound_tables
    planning[:, -1, 1] = 1
    return planning


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


def _row_ints(matrix: npt.NDArray[np.uint8]) -> tuple[int, ...]:
    """Pack each row of a binary matrix into an integer, with entry ``j`` in bit ``j``."""
    return tuple(map(_packed_row_int, pack_bits(matrix)))


def _packed_row_int(row: npt.NDArray[np.uint64]) -> int:
    """Convert a row of 64-bit words into an integer, with word ``j`` in bits ``64 j`` and up."""
    return sum(int(word) << (64 * jj) for jj, word in enumerate(row))


def _int_to_words(value: int, width: int) -> npt.NDArray[np.uint64]:
    """Split an integer into ``width`` 64-bit words, with word ``j`` in bits ``64 j`` and up."""
    return np.frombuffer(value.to_bytes(8 * width, "little"), dtype="<u8").astype(np.uint64)


def _tabulate_singles(
    check_columns: Sequence[int], num_qubits: int, *, is_pauli: bool
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Tabulate the words of weight zero or one by syndrome, and the single-qubit X and Z errors.

    The words are the identity and, for packed Paulis, X, Z, and Y on each qubit, and for X or Z
    parts, the single bits.  The syndrome of Y on a qubit is the sum of those of X and Z there.
    Each table maps each syndrome of its words to one of them.

    Args:
        check_columns: Column ``c`` of the checks as an integer: the syndrome of the word with only
            bit ``c`` set.
        num_qubits: The number of qubits ``n`` of packed Paulis, which have ``2 n`` bits; unused for
            X or Z parts.
        is_pauli: Whether the words are packed Paulis, rather than X or Z parts.

    Returns:
        The table of these words, and the tables of the identity and the single-qubit X errors, and
        of the identity and the single-qubit Z errors.  For X or Z parts, the last two hold only the
        identity.
    """
    singles, x_singles, z_singles = {0: 0}, {0: 0}, {0: 0}
    if not is_pauli:
        for column, syndrome in enumerate(check_columns):
            singles.setdefault(syndrome, 1 << column)
        return singles, x_singles, z_singles
    for qubit in range(num_qubits):
        x, z = 1 << qubit, 1 << (qubit + num_qubits)
        sx, sz = check_columns[qubit], check_columns[qubit + num_qubits]
        x_singles.setdefault(sx, x)
        z_singles.setdefault(sz, z)
        singles.setdefault(sx, x)
        singles.setdefault(sz, z)
        singles.setdefault(sx ^ sz, x ^ z)
    return singles, x_singles, z_singles


def _tabulate_css_singles(
    x_singles: dict[int, int], z_singles: dict[int, int], max_pairs: int
) -> dict[int, int] | None:
    """Tabulate the products of single-qubit X and Z errors by syndrome, if not too many.

    Args:
        x_singles: Single-qubit X errors, and the identity, by syndrome.
        z_singles: Single-qubit Z errors, and the identity, by syndrome.
        max_pairs: The most pairs of an X and a Z error that the table may come from.

    Returns:
        None if there are more than ``max_pairs`` pairs, and otherwise the first product
        ``x ^ z`` of each syndrome ``sx ^ sz``, in the order of the two dictionaries.
    """
    if len(x_singles) * len(z_singles) > max_pairs:
        return None
    table: dict[int, int] = {}
    for sx, x in x_singles.items():
        for sz, z in z_singles.items():
            table.setdefault(sx ^ sz, x ^ z)
    return table


def _xor_selected(rows: Sequence[int], support: int) -> int:
    """XOR together the rows whose indices are the set bits of ``support``."""
    word = 0
    while support:
        # Add the row for the lowest set bit, and then clear that bit.
        low = support & -support
        word ^= rows[low.bit_length() - 1]
        support ^= low
    return word


####################################################################################################
# compiled coset search


@always_compiled
def _find_first_not_within_max_weight(
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    best: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    planning: _PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    tails: _TailFields | None,
    scale: int,
) -> tuple[int, int]:
    """Search the coset of each syndrome in turn with ``_search_coset``.

    The tables, from ``inverse`` to ``x_mask``, and ``tails`` are those of ``_search_coset``.
    Return the index of the first syndrome whose coset has no word of weight at most
    ``max_weight``, and the lower bound that ``_search_coset`` returns for it, with the lightest
    word that it weighed left in ``best``.  If there is no such syndrome, return the number of
    syndromes and zero.
    """
    for index in range(len(syndromes)):
        lower_bound = _search_coset(
            syndromes[index],
            max_weight,
            best,
            inverse,
            generators,
            pivots,
            planning,
            x_mask,
            tails,
            scale,
        )
        if lower_bound > max_weight:
            return index, lower_bound
    return len(syndromes), 0


@always_compiled(inline=True)
def _search_coset(
    packed_syndrome: npt.NDArray[np.uint64],
    max_weight: int,
    best: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    planning: _PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    tails: _TailFields | None,
    scale: int,
) -> int:
    """Search the coset of a syndrome for a word of weight at most ``max_weight``.

    ``packed_syndrome`` is the syndrome split into 64-bit machine words, with machine word ``j``
    holding bits ``64 j`` and up.  The search runs on embedded words.  In the Pauli search, the
    words are Pauli errors, and the embedded word of an error is its packed Pauli when errors are
    weighed by CSS weight, and its packed Pauli followed by ``x xor z`` when they are weighed by
    physical weight, so that its Hamming weight is twice the physical weight (see
    ``symplectic_to_hamming``).  In a classical search, the words are X or Z parts, each is its own
    embedded word, and its weight is its Hamming weight.  The weight of an embedded word is that of
    its word: its CSS weight if ``scale`` is zero, with its X part selected by ``x_mask``, and
    otherwise its Hamming weight divided by ``scale``, which is two for the physical weight and one
    for X or Z parts.  The tables are those of ``_CosetSearchTables``.  Row ``j`` of ``inverse`` is
    the embedded word of a word whose syndrome has only bit ``j`` set.  ``generators`` and
    ``pivots`` are the information sets of a family of the embedded generators of the code, with
    pivots padded with -1.  ``planning`` is the planning buffer of the search with this family,
    whose cached stopping steps a query may update.  ``tails`` is None, or, on a route by tails
    (see ``coset_routes``), the fields of the ``_CosetTails`` of the family, and then ``scale`` is
    one or two, since the search weighs words by their Hamming weight.  Numba compiles the search
    separately for None and for tails, and leaves out the branches that the type of ``tails`` rules
    out, so choosing between them costs nothing for each word.

    The embedded words of the coset form a coset of the embedded code.  The search starts from the
    sum of the rows of ``inverse`` for the set bits of the syndrome, a word with that syndrome.  The
    offset of an information set is this word reduced by rows of the set until it is zero on the
    set's pivot columns.  Every embedded word of the coset is the offset plus a unique sum of rows
    of the set, and has a one on a pivot column exactly where it adds that pivot's row, so the BZ
    certificate applies to the coset as it does to a space.  Layer ``w`` of a set is its offset plus
    each sum of ``w`` of its rows, so layer zero is the offset itself.

    The search reads the bounds of its steps from the table of the planning buffer for the parity of
    the Hamming weight of the word of the syndrome (see ``_get_planning_table``).

    The search visits layer zero of the information sets in order, then layer one, and so on.  It
    keeps the lightest word visited in ``best``, and stops at the first word of weight at most
    ``max_weight``.  Once the first ``j`` sets have completed layer ``w``, every unvisited word
    weighs at least the bound of step ``(w, j)`` in its table.  So the search also stops at its
    stopping step, the first step whose bound exceeds ``max_weight``, which ``_get_stopping_step``
    finds.  It skips each set that adds nothing to the bound of that step (see
    ``_search_from_syndrome_word``).  A search by tails visits the same words in the same order,
    except those that the tables of ``tails`` prove to lie beyond ``max_weight``, which it skips.
    So it finds the same first word of weight at most ``max_weight``, if any.

    Returns:
        A lower bound on the weight of every word in the coset: zero, which is at most
        ``max_weight``, if ``best`` has weight at most ``max_weight``, and otherwise a bound that
        exceeds ``max_weight`` (see ``_search_from_syndrome_word``).
    """
    best_weight, parity = _get_syndrome_word(packed_syndrome, best, inverse, x_mask, scale)
    if best_weight <= max_weight:
        return 0
    table = _get_planning_table(planning, parity)
    stop_layer, stop_num_completed, stop_bound = _get_stopping_step(table, max_weight)
    # The search rounds to the parity of the coset if it weighs Hamming weights and its planning
    # buffer has a table for each parity (see ``_get_planning_table``).
    lower_bound, _, _ = _search_from_syndrome_word(
        packed_syndrome,
        max_weight,
        stop_layer,
        stop_num_completed,
        stop_bound,
        best,
        best_weight,
        generators,
        pivots,
        x_mask,
        tails,
        scale,
        parity if scale != 0 and len(planning) == 2 else -1,
    )
    return lower_bound


@always_compiled
def _get_planning_table(planning: _PlanningBuffer, parity: int) -> npt.NDArray[np.int64]:
    """The table of a planning buffer for a coset whose Hamming weights have parity ``parity``.

    The Hamming weights are those of the embedded words of the coset.  A buffer with a table for
    each parity belongs to an even code, whose cosets each have Hamming weights of one parity, and
    gives the table of the parity (see ``_get_planning``).  A buffer with one table gives it for
    every coset.
    """
    return planning[parity] if len(planning) == 2 else planning[0]


@always_compiled
def _get_stopping_step(table: npt.NDArray[np.int64], max_weight: int) -> tuple[int, int, int]:
    """Find the step at which a coset query stops, and cache it in a table of the planning buffer.

    The stopping step is the first step whose bound in the table exceeds the maximum weight: the
    step at which ``stop_num_completed`` sets complete layer ``stop_layer``.  A query stops there,
    unless it finds a light enough word sooner.  The bounds of the steps do not decrease in their
    order.  Column zero of each row of bounds holds the bound before the first step of the row,
    which is zero in row zero.  So the bound of the step before step ``(w, j)`` is entry
    ``[w, j - 1]``, and for a nonnegative maximum weight, step ``(w, j)`` is the stopping step
    exactly if its bound exceeds the maximum weight and entry ``[w, j - 1]`` does not.  The step
    cached in the table (see ``_get_planning``) is reused if it passes this check.  Otherwise,
    ``information_sets.find_stopping_step`` finds the stopping step by binary search, and it is
    cached in its place.  A negative maximum weight fails the check, and the binary search returns
    the first step.  If no bound exceeds the maximum weight, which happens only when the maximum
    weight reaches the largest native integer, the binary search returns the last step.

    Oracles that share a cached search share its planning buffer.  Compiled calls hold the global
    interpreter lock, since numba compiles them without ``nogil``, so calls interleave only when
    these functions run as plain Python in several threads, or in free-threaded Python.  The layer
    and the number of completed sets of a step are written as two entries, so a read may pair the
    layer of one step with the number of completed sets of another.  Any such pair indexes the
    bounds, which never change, and it is checked against them before it is used.

    Returns:
        ``stop_layer``, ``stop_num_completed``, and the bound of that step.
    """
    bounds = table[:-1]
    layer, num_completed = table[-1, 0], table[-1, 1]
    is_stopping_step = (
        max_weight < bounds[layer, num_completed] and bounds[layer, num_completed - 1] <= max_weight
    )
    if not is_stopping_step:
        layer, num_completed = information_sets.find_stopping_step(
            bounds, max_weight, 0, len(bounds) - 1
        )
        table[-1, 0], table[-1, 1] = layer, num_completed
    return layer, num_completed, int(bounds[layer, num_completed])


@always_compiled
def _search_from_syndrome_word(
    packed_syndrome: npt.NDArray[np.uint64],
    max_weight: int,
    stop_layer: int,
    stop_num_completed: int,
    stop_bound: int,
    best: npt.NDArray[np.uint64],
    best_weight: int,
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    x_mask: npt.NDArray[np.uint64],
    tails: _TailFields | None,
    scale: int,
    parity: int,
) -> tuple[int, int, int]:
    """Search a coset, as ``_search_coset`` does, from the word of its syndrome.

    ``best`` holds that word, of weight ``best_weight`` above ``max_weight``, and
    ``packed_syndrome`` is its syndrome.  The stopping step and its bound are those of
    ``_get_stopping_step``.  ``parity`` is the parity of the Hamming weights of the words of the
    coset if the search weighs Hamming weights and rounds its bounds to that parity (see
    ``_get_search_tables``), and otherwise -1.  A set of rank deficiency ``s`` adds to the bound of
    a step only once it has completed layer ``s``, so a set that would not have completed it by the
    stopping step adds nothing, and the search skips it.  These are the unused sets of
    ``information_sets.plan_steps``, whose work for this step, from layer zero, counts the words
    that the search weighs after the word of the syndrome if it finds no light enough word.

    A search by tails (see ``_search_coset``) weighs a word by its rows below the rank of its set
    and its tail.  It keeps the set, the layer, and the support of the lightest word, and writes
    that word into ``best`` before it returns.  In each layer of two or more rows of a set with
    tables, it skips the completions of a prefix whose bound reaches the skip bound: the least bound
    that proves a word of the coset heavier than ``max_weight``, one more than the greatest Hamming
    weight at most ``scale * max_weight`` that a word of the coset can have (see
    ``affine_search.scan_coset_layer_by_tails_with_tables``).  So it finds the same first word of
    weight at most ``max_weight``, if any.  If it finds none and skipped any word, it bounds the
    coset by the least weight beyond ``max_weight`` that a word of the coset can have, which bounds
    the skipped words, the words that it weighed, and the words beyond the stopping step.

    Returns:
        A lower bound on the weight of every word in the coset: zero if the search finds a word of
        weight at most ``max_weight``, the least weight beyond ``max_weight`` that a word of the
        coset can have if it skipped any word, and otherwise the lesser of the weight of the
        lightest word that it weighed and the bound of the stopping step.  Also the numbers of words
        weighed after the word of the syndrome and of supports skipped.  Only tests read the
        numbers, to check which sets a search skips and that a search by tails weighs as many words.
    """
    # Reduce the word of the syndrome to the offset of each information set, or, by tails, find the
    # rank of each set and the tail of its offset.
    num_sets, num_rows, width = generators.shape
    if tails is None:
        offsets = _get_offsets(best, generators, pivots)
    else:
        tail_rows, inverse_tails, completion_tables, layout, block_bits = tails
        ranks, offsets = _get_offset_tails(packed_syndrome, best, pivots, tail_rows, inverse_tails)
        width = tail_rows.shape[2]

    # The scans weigh ``factor`` units for each unit of weight: Hamming weights, which are ``scale``
    # times the weights, or CSS weights if ``scale`` is zero.
    factor = max(1, scale)
    best_units, cutoff = factor * best_weight, factor * max_weight
    skip_bound = _get_skip_bound(cutoff, parity)

    # Visit each layer one information set at a time, up to the stopping step.  The scans take work
    # space for one set.  Those by tails keep the support of each lighter word in ``best_support``,
    # and the others keep the word itself in ``best``.
    best_set, best_layer = -1, 0
    weighed, skipped = 0, 0
    widths = np.full(1, width, dtype=np.int64)
    support = np.empty(num_rows, dtype=np.int64)
    best_support = np.empty(num_rows, dtype=np.int64)
    prefixes = np.empty((num_rows, 1, width), dtype=np.uint64)
    for layer in range(stop_layer + 1):
        for set_index in range(num_sets if layer < stop_layer else stop_num_completed):
            if information_sets.is_unused_set(pivots, set_index, stop_layer, stop_num_completed):
                continue
            set_skipped = 0
            if layer == 0:
                # The offset is zero on the pivot columns, so its tail has its Hamming weight.
                weight, ones = _get_weight(offsets[set_index], x_mask, scale)
                found, count = weight if scale == 0 else ones, 1
                if tails is None and found < best_units:
                    best[:] = offsets[set_index]
            elif tails is None:
                if scale == 0:
                    found, count = affine_search.scan_coset_layer_by_css_weight(
                        offsets[set_index],
                        generators[set_index : set_index + 1],
                        widths,
                        support[:layer],
                        prefixes,
                        best_units,
                        cutoff,
                        x_mask,
                        best,
                    )
                else:
                    found, count = affine_search.scan_coset_layer(
                        offsets[set_index],
                        generators[set_index : set_index + 1],
                        widths,
                        support[:layer],
                        prefixes,
                        best_units,
                        cutoff,
                        best,
                    )
            elif layer >= 2 and layout[set_index, 1]:
                # The tables of the set start at ``layout[set_index, 0]``, with a row of tables for
                # each row of the set but the last.
                num_blocks = layout[set_index, 1]
                block_offsets = layout[set_index, 2 : num_blocks + 3]
                start = layout[set_index, 0]
                end = start + (num_rows - 1) * block_offsets[-1]
                found, count, set_skipped = affine_search.scan_coset_layer_by_tails_with_tables(
                    offsets[set_index],
                    tail_rows[set_index : set_index + 1],
                    ranks[set_index : set_index + 1],
                    widths,
                    support[:layer],
                    prefixes,
                    best_units,
                    cutoff,
                    skip_bound,
                    completion_tables[start:end],
                    block_offsets,
                    block_bits,
                    best_support,
                )
            else:
                found, count = affine_search.scan_coset_layer_by_tails(
                    offsets[set_index],
                    tail_rows[set_index : set_index + 1],
                    ranks[set_index : set_index + 1],
                    widths,
                    support[:layer],
                    prefixes,
                    best_units,
                    cutoff,
                    best_support,
                )
            weighed += count
            skipped += set_skipped
            if found < best_units:
                best_units, best_set, best_layer = found, set_index, layer
            if best_units <= cutoff:
                break
        if best_units <= cutoff:
            break

    # By tails, write the lightest word into ``best``, which holds the word of the syndrome until
    # then.
    if tails is not None and best_set >= 0:
        _rebuild_word(best, generators[best_set], pivots[best_set], best_support[:best_layer])

    # A search that finds no light enough word but skipped words bounds the coset by the least
    # weight beyond the maximum weight that a word of the coset can have: the skip bound rounded up
    # to the parity of the coset, then divided by the scale and rounded up, as the bounds of the
    # steps are.
    if best_units <= cutoff:
        lower_bound = 0
    elif skipped:
        rounded = skip_bound + (skip_bound - parity) % 2 if parity >= 0 else skip_bound
        lower_bound = -(-rounded // factor)
    else:
        lower_bound = min(best_units // factor, stop_bound)
    return lower_bound, weighed, skipped


@always_compiled
def _find_min_weights(
    syndromes: npt.NDArray[np.uint64],
    max_weight: int,
    floors: npt.NDArray[np.int64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    planning: _PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    tails: _TailFields | None,
    scale: int,
) -> npt.NDArray[np.int64]:
    """Search the coset of each syndrome in turn with ``_search_coset_for_min_weight``.

    The arguments are those of ``_find_first_not_within_max_weight``, without ``best``, and with the
    floor of each syndrome in ``floors``.  Return the weight that ``_search_coset_for_min_weight``
    gives each syndrome.
    """
    weights = np.empty(len(syndromes), dtype=np.int64)
    best = np.empty(inverse.shape[1], dtype=np.uint64)
    for index in range(len(syndromes)):
        weight, _, _ = _search_coset_for_min_weight(
            syndromes[index],
            max_weight,
            floors[index],
            best,
            inverse,
            generators,
            pivots,
            planning,
            x_mask,
            tails,
            scale,
        )
        weights[index] = weight
    return weights


@always_compiled(inline=True)
def _search_coset_for_min_weight(
    packed_syndrome: npt.NDArray[np.uint64],
    max_weight: int,
    floor: int,
    best: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    generators: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    planning: _PlanningBuffer,
    x_mask: npt.NDArray[np.uint64],
    tails: _TailFields | None,
    scale: int,
) -> tuple[int, int, int]:
    """Search the coset of a syndrome for its minimum weight, if that is at most ``max_weight``.

    The arguments are those of ``_search_coset``, with a floor of at most ``max_weight``, and
    ``best`` is work space.  A word of weight at most ``floor`` ends the search, which saves work
    for callers to which any such word gives the same answer.

    The search visits the layers of the information sets in the order of ``_search_coset``.  Its
    best weight is that of the word of the syndrome, or of the lightest word within ``max_weight``
    that it has found since.  It stops at its stopping step, the first step whose bound reaches the
    lesser of its best weight and ``max_weight + 1``, since every word that it has not visited then
    weighs at least as much.  This step moves earlier whenever the best weight falls.  The search
    skips each set that adds nothing to the bound of its current stopping step, as
    ``_search_from_syndrome_word`` does.  Such a set adds nothing to the bound of an earlier step
    either, so the bound of the step at which the search stops holds.  Within a step, every word not
    yet visited weighs at least the bound of the step before it.  So the search also stops as soon
    as its best weight is at most that bound, or at most ``floor``.

    The scans weigh Hamming weights, which are ``scale`` times the weights, or CSS weights if
    ``scale`` is zero.  They keep only the words lighter than an incumbent: the skip bound of
    ``max_weight`` (see ``_get_skip_bound``) until the search finds a word within that weight, and
    then the best weight.  Where every word of the coset has the parity of the Hamming weight of the
    best word, no word is one lighter, so the incumbent is one less and the scans keep the same
    words.  This holds in a search of Pauli errors by physical weight, whose embedded words have
    even Hamming weights, and in a search of X or Z parts that rounds to the parity of each coset.
    The lower incumbent lets the tables of a search by tails skip more.

    A search by tails visits the same words in the same order, but weighs a word by its rows below
    the rank of its set and its tail, as ``_search_from_syndrome_word`` does.  In each layer of two
    or more rows of a set with tables, it skips the completions of a prefix whose bound reaches the
    best weight of the scan, which starts at its incumbent: the scan takes the skip bound of
    ``max_weight``, which is at least the incumbent, as its skip bound (see
    ``affine_search.scan_coset_layer_by_tails_with_tables``).  Until the search finds a word within
    ``max_weight``, the incumbent is that skip bound, at which a query by tails skips too.  Then it
    is the Hamming weight of the best word, or one less where every word of the coset has its
    parity, so the search skips no word lighter than the best word.  So the search gives the weight
    of a search of whole words.  For a coset without a word within ``max_weight``, it weighs and
    skips the words that ``_search_from_syndrome_word`` weighs and skips at ``max_weight``.

    Returns:
        The best weight when the search stops, or ``max_weight + 1`` if it found no word within
        ``max_weight``, and the numbers of words weighed after the word of the syndrome and of
        supports skipped.  The best weight is the minimum weight of the coset, unless it is at most
        ``floor``.  Only tests read the numbers.
    """
    # Start from the word of the syndrome, which ends the search if it is within the floor.  Reduce
    # it to the offset of each information set, or, by tails, find the rank of each set and the
    # tail of its offset.
    best_weight, parity = _get_syndrome_word(packed_syndrome, best, inverse, x_mask, scale)
    if best_weight <= floor:
        return best_weight, 0, 0
    bounds = _get_planning_table(planning, parity)[:-1]
    num_sets, num_rows, width = generators.shape
    if tails is None:
        offsets = _get_offsets(best, generators, pivots)
    else:
        tail_rows, inverse_tails, completion_tables, layout, block_bits = tails
        ranks, offsets = _get_offset_tails(packed_syndrome, best, pivots, tail_rows, inverse_tails)
        width = tail_rows.shape[2]

    # Find the skip bound and the parity slack of the incumbent, in the units of the scans, which
    # weigh ``factor`` units for each unit of weight.  Only Hamming weights have a parity.  Find
    # also the upper bound ``U`` of the search, in units of weight (Section 3.10 of
    # ``brouwer_zimmermann.md``), and its first stopping step.
    factor = max(1, scale)
    rounds = scale != 0 and len(planning) == 2
    skip_bound = _get_skip_bound(factor * max_weight, parity if rounds else -1)
    slack = 1 if scale == 2 or rounds else 0
    best_units = factor * best_weight
    upper_bound = min(best_weight, max_weight + 1)
    stop_layer, stop_num_completed = _get_first_step_reaching(bounds, upper_bound)

    # Visit each layer one information set at a time, up to the stopping step.  The scans take work
    # space for one set.  The search reaches a stopping step by the last layer, whose bounds are
    # ``NO_WORD_BOUND``, since its upper bound is at most ``max_weight + 1``, which
    # ``_CosetSearch.get_min_weights`` keeps at most ``n + 1``.  The scans keep each lighter word in
    # ``best``, or its support in ``best_support`` by tails, which are only work space, since the
    # search returns no word.
    weighed, skipped = 0, 0
    widths = np.full(1, width, dtype=np.int64)
    support = np.empty(num_rows, dtype=np.int64)
    best_support = np.empty(num_rows, dtype=np.int64)
    prefixes = np.empty((num_rows, 1, width), dtype=np.uint64)
    layer = 0
    while True:
        for set_index in range(num_sets):
            # Weigh the words of this step that are lighter than the incumbent, unless the set adds
            # nothing to the bound of the stopping step, and return once the best weight is at most
            # the floor or the bound of the step before this one.  A scan returns once its best
            # weight, which starts at the incumbent, is at most its cutoff, so the cutoff stays
            # below the incumbent.
            if not information_sets.is_unused_set(
                pivots, set_index, stop_layer, stop_num_completed
            ):
                incumbent = min(best_units - slack, skip_bound)
                step_floor = max(floor, bounds[layer, set_index])
                cutoff = min(factor * step_floor, incumbent - 1)
                set_skipped = 0
                if layer == 0:
                    # The offset is zero on the pivot columns, so its tail has its Hamming weight.
                    weight, ones = _get_weight(offsets[set_index], x_mask, scale)
                    found, count = weight if scale == 0 else ones, 1
                elif tails is None:
                    if scale == 0:
                        found, count = affine_search.scan_coset_layer_by_css_weight(
                            offsets[set_index],
                            generators[set_index : set_index + 1],
                            widths,
                            support[:layer],
                            prefixes,
                            incumbent,
                            cutoff,
                            x_mask,
                            best,
                        )
                    else:
                        found, count = affine_search.scan_coset_layer(
                            offsets[set_index],
                            generators[set_index : set_index + 1],
                            widths,
                            support[:layer],
                            prefixes,
                            incumbent,
                            cutoff,
                            best,
                        )
                elif layer >= 2 and layout[set_index, 1]:
                    # The tables of the set start at ``layout[set_index, 0]`` (see
                    # ``_search_from_syndrome_word``).
                    num_blocks = layout[set_index, 1]
                    block_offsets = layout[set_index, 2 : num_blocks + 3]
                    start = layout[set_index, 0]
                    end = start + (num_rows - 1) * block_offsets[-1]
                    found, count, set_skipped = affine_search.scan_coset_layer_by_tails_with_tables(
                        offsets[set_index],
                        tail_rows[set_index : set_index + 1],
                        ranks[set_index : set_index + 1],
                        widths,
                        support[:layer],
                        prefixes,
                        incumbent,
                        cutoff,
                        skip_bound,
                        completion_tables[start:end],
                        block_offsets,
                        block_bits,
                        best_support,
                    )
                else:
                    found, count = affine_search.scan_coset_layer_by_tails(
                        offsets[set_index],
                        tail_rows[set_index : set_index + 1],
                        ranks[set_index : set_index + 1],
                        widths,
                        support[:layer],
                        prefixes,
                        incumbent,
                        cutoff,
                        best_support,
                    )
                weighed += count
                skipped += set_skipped
                if found < incumbent:
                    best_units, best_weight = found, found // factor
                if best_weight <= step_floor:
                    return best_weight, weighed, skipped

            # Stop at the stopping step, and move that step earlier if the best weight fell.
            new_upper_bound = min(best_weight, max_weight + 1)
            if bounds[layer, set_index + 1] >= new_upper_bound:
                return new_upper_bound, weighed, skipped
            if new_upper_bound < upper_bound:
                upper_bound = new_upper_bound
                stop_layer, stop_num_completed = _get_first_step_reaching(bounds, upper_bound)
        layer += 1


@always_compiled(inline=True)
def _get_first_step_reaching(bounds: npt.NDArray[np.int64], target: int) -> tuple[int, int]:
    """The first step whose bound in a table of bounds of steps reaches ``target``.

    It is the stopping step of ``information_sets.find_stopping_step`` for the limit
    ``target - 1``.  Unlike ``_get_stopping_step``, this function caches no step, so the cached
    steps of the planning buffer serve only the queries of ``_search_coset``.
    """
    return information_sets.find_stopping_step(bounds, target - 1, 0, len(bounds) - 1)


@always_compiled(inline=True)
def _get_offsets(
    word: npt.NDArray[np.uint64], generators: npt.NDArray[np.uint64], pivots: npt.NDArray[np.int64]
) -> npt.NDArray[np.uint64]:
    """Reduce a word of a coset to the offset of each information set, which is zero on its pivots.

    ``generators`` and ``pivots`` are those of ``_search_coset``.
    """
    num_sets, _, width = generators.shape
    offsets = np.empty((num_sets, width), dtype=np.uint64)
    for set_index in range(num_sets):
        offsets[set_index] = word
        _reduce_to_offset(offsets[set_index], generators[set_index], pivots[set_index])
    return offsets


@always_compiled(inline=True)
def _get_offset_tails(
    packed_syndrome: npt.NDArray[np.uint64],
    word: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    tails: npt.NDArray[np.uint64],
    inverse_tails: npt.NDArray[np.uint64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint64]]:
    """Find the rank of each information set and the tail of its offset.

    ``word`` is the word of the syndrome ``packed_syndrome``, ``pivots`` are those of
    ``_search_coset``, and ``tails`` and ``inverse_tails`` are those of ``_CosetTails``.  The rows
    of a set with pivots on which the word has ones reduce the word to the offset, since every row
    is zero on the pivots of the other rows.  The tail of the word sums the tails of the rows of the
    inverse for the set bits of the syndrome.
    """
    num_sets, num_rows, tail_width = tails.shape
    ranks = np.zeros(num_sets, dtype=np.int64)
    offset_tails = np.zeros((num_sets, tail_width), dtype=np.uint64)
    for set_index in range(num_sets):
        _xor_selected_rows(packed_syndrome, inverse_tails[set_index], offset_tails[set_index])
        rank = 0
        while rank < num_rows and pivots[set_index, rank] >= 0:
            pivot = pivots[set_index, rank]
            if word[pivot // 64] >> np.uint64(pivot % 64) & np.uint64(1):
                for index in range(tail_width):
                    offset_tails[set_index, index] ^= tails[set_index, rank, index]
            rank += 1
        ranks[set_index] = rank
    return ranks, offset_tails


@always_compiled(inline=True)
def _get_skip_bound(cutoff: int, parity: int) -> int:
    """The least Hamming weight that proves a word of a coset heavier than ``cutoff``.

    It is one more than the greatest Hamming weight at most the cutoff that a word of the coset can
    have.  ``parity`` is the parity of the coset if the search rounds to it, and -1 otherwise.  The
    bound is ``cutoff + 1``, or ``cutoff`` if the search rounds and the words have the parity of
    ``cutoff + 1``.
    """
    if parity >= 0 and (cutoff - parity) % 2:
        return cutoff
    return cutoff + 1


@always_compiled
def _rebuild_word(
    word: npt.NDArray[np.uint64],
    rows: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    support: npt.NDArray[np.int64],
) -> None:
    """Turn a word of a coset into the offset of an information set plus the rows of a support.

    ``rows`` and ``pivots`` are those of the set, as for ``_reduce_to_offset``.
    """
    _reduce_to_offset(word, rows, pivots)
    for row in support:
        for index in range(len(word)):
            word[index] ^= rows[row, index]


@always_compiled(inline=True)
def _reduce_to_offset(
    word: npt.NDArray[np.uint64], rows: npt.NDArray[np.uint64], pivots: npt.NDArray[np.int64]
) -> None:
    """Reduce a word of a coset to the offset of an information set, which is zero on its pivots.

    ``rows`` are the packed rows of the set, and ``pivots`` their pivot columns, padded with -1
    beyond its rank.  The word adds the row of each pivot that it has a one on, since every row is
    zero on the pivots of the other rows.
    """
    for row in range(len(pivots)):
        pivot = pivots[row]
        if pivot < 0:
            break
        if word[pivot // 64] >> np.uint64(pivot % 64) & np.uint64(1):
            for index in range(len(word)):
                word[index] ^= rows[row, index]


@always_compiled(inline=True)
def _get_syndrome_word(
    packed_syndrome: npt.NDArray[np.uint64],
    word: npt.NDArray[np.uint64],
    inverse: npt.NDArray[np.uint64],
    x_mask: npt.NDArray[np.uint64],
    scale: int,
) -> tuple[int, int]:
    """Write the sum of the rows of ``inverse`` for the set bits of a packed syndrome into ``word``.

    The sum is an embedded word with that syndrome.  Return its weight (see ``_search_coset``), and
    the parity of its Hamming weight.
    """
    word[:] = 0
    _xor_selected_rows(packed_syndrome, inverse, word)
    weight, ones = _get_weight(word, x_mask, scale)
    return weight, ones % 2


@always_compiled(inline=True)
def _xor_selected_rows(
    packed_bits: npt.NDArray[np.uint64], rows: npt.NDArray[np.uint64], word: npt.NDArray[np.uint64]
) -> None:
    """Add to ``word`` row ``b`` of ``rows`` for each set bit ``b`` of ``packed_bits``.

    ``packed_bits`` is split into 64-bit machine words, with machine word ``j`` holding bits
    ``64 j`` and up, and each set bit must be below the number of rows; this is not checked.
    """
    for machine_word in range(len(packed_bits)):
        bits = packed_bits[machine_word]
        while bits:
            bit = 64 * machine_word + trailing_zeros_word(bits)
            for index in range(len(word)):
                word[index] ^= rows[bit, index]
            bits &= bits - np.uint64(1)


@always_compiled
def _get_weight(
    word: npt.NDArray[np.uint64], x_mask: npt.NDArray[np.uint64], scale: int
) -> tuple[int, int]:
    """The weight of the error with an embedded word, and the Hamming weight of the embedded word.

    See ``_search_coset``.
    """
    ones = x_ones = np.int64(0)
    for index in range(len(word)):
        ones += popcount_word(word[index])
        x_ones += popcount_word(word[index] & x_mask[index])
    return int(max(x_ones, ones - x_ones) if scale == 0 else ones // scale), int(ones)


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
        _xor_selected_rows(syndromes[row], map_columns, mapped[row])
    return mapped
