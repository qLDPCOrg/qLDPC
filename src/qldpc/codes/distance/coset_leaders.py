# SPDX-License-Identifier: Apache-2.0

"""Exact queries of the lightest Pauli errors with a given syndrome on a stabilizer state.

:class:`CosetLeaderSearch` decides whether the lightest Pauli errors with a given syndrome weigh at
most a given weight, and finds their weight up to such a limit.
:func:`qldpc.circuits.certify_bounded_fault_propagation` uses it to check whether the faults of a
circuit leave errors that are too heavy.
"""

from __future__ import annotations

import functools
from typing import Literal, SupportsIndex, cast, get_args

import numpy as np
import numpy.typing as npt

from qldpc._binary import (
    get_kernel,
    get_num_words,
    multiply_binary,
    pack_bits,
    row_ints,
    row_reduce_binary,
    words_to_int,
)
from qldpc._compilation import always_compiled

from . import _threads
from ._brouwer_zimmermann.coset import (
    BrouwerZimmermannCosetSearch,
    get_css_searches,
    get_pauli_search,
)
from ._code_matrices import as_binary_matrix, as_integer, swap_pauli_halves
from ._connected_cluster.coset import (
    ConnectedClusterCosetSearch,
    find_first_cluster_bound,
    get_min_cluster_weights,
)
from ._cosets import xor_selected, xor_selected_rows
from ._tuning import get_tuning
from .min_weight_bounds import MinWeightBound

CosetMethod = Literal["auto", "brouwer_zimmermann", "connected_cluster"]
"""The methods of exact searches of coset leaders, such as those of :class:`CosetLeaderSearch`.

- ``"auto"`` (the default): currently ``"brouwer_zimmermann"``, but this can change between versions
  of qLDPC.  For searches by physical weight, ``"connected_cluster"`` can be much faster on states
  whose stabilizer groups have sparse generators, such as encoded states of surface codes.  On other
  states, it can be much slower.
- ``"brouwer_zimmermann"``: searches the cosets of binary codes by information sets (see the
  `Brouwer-Zimmermann guide
  <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/brouwer_zimmermann.md>`_).
- ``"connected_cluster"``: makes generators of the stabilizer group sparser, and grows errors from
  the syndrome through them, so it can be fast only if those generators are sparse (see the
  `connected-cluster guide
  <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/connected_cluster.md>`_).
  It weighs errors by CSS weight only on CSS states.  For a syndrome whose errors all weigh more
  than a positive maximum weight, the lower bound that it gives is that maximum weight plus one,
  whereas ``"brouwer_zimmermann"`` often proves a higher one.
"""

####################################################################################################
# coset-leader minimum-weight bounds


class CosetLeaderSearch:
    """Exact search of the lightest Pauli errors with a given syndrome on a pure stabilizer state.

    The search takes ``n`` independent commuting stabilizers on ``n`` qubits, which define a pure
    stabilizer state rather than a code that encodes qubits.  The syndrome of a Pauli error says
    which of these stabilizers it anticommutes with.  The errors with a given syndrome form a
    syndrome class, which is a coset of the stabilizer group, and its lightest errors are its coset
    leaders.  The weight of an error is either its physical weight, the number of qubits that it
    acts on, or its CSS weight ``max(|x|, |z|)``, where ``x`` and ``z`` are its X and Z parts.

    The queries :meth:`get_min_weight_bound`, :meth:`find_first_not_within_max_weight`, and
    :meth:`get_min_weights` represent syndromes and errors by packed integers and arrays, which the
    search uses directly:

    - A syndrome is an integer with bit ``j`` set if the error anticommutes with stabilizer ``j``.
      :meth:`get_min_weight_bound` also takes a syndrome as a binary vector.
    - A Pauli error is an integer with its X part in bits ``0`` to ``n - 1`` and its Z part in bits
      ``n`` to ``2 n - 1``.  :meth:`unpack_pauli` turns it into a binary vector.
    - A batch of syndromes is a two-dimensional ``uint64`` array with one row per syndrome and
      ``ceil(n / 64)`` words per row.  Bit ``k`` of word ``j`` of a row is bit ``64 j + k`` of its
      syndrome.  :meth:`pack_syndromes` makes a batch from binary syndromes.

    Whether a syndrome class has an error within a maximum weight never depends on the method or the
    tuning of the search, but the error and the lower bound that a query returns can (see
    :class:`~qldpc.codes.distance.MinWeightBound`).  The search reads the active tuning when it is
    built, and :doc:`/exact_search_tuning` says how to change it.

    :meth:`find_first_not_within_max_weight` and :meth:`get_min_weights` move the rest of a costly
    batch to several threads.  The calling thread starts a batch alone, and a batch of cheap queries
    stays there.  The search fixes its number of threads when it is built, from the ``num_threads``
    field of the tuning.  The results of the queries do not depend on the number of threads.
    :doc:`/exact_search_tuning` says how many threads a search uses by default, and how to cap them,
    as programs that run queries in several processes or threads at once should.

    The Brouwer-Zimmermann method caches its searches for recently used stabilizers, so that
    another ``CosetLeaderSearch`` for the same stabilizers, measure of weight, and tuning reuses
    them.  A CSS state is one whose stabilizer group is generated by stabilizers without Z parts and
    stabilizers without X parts.  For the CSS weight on a CSS state, the search covers the X and Z
    parts of errors separately, and the cache is for the stabilizer group, whichever stabilizers
    generate it.

    Args:
        stabilizers: A binary matrix with ``2 n`` columns in ``[X|Z]`` order, whose rows are ``n``
            independent commuting stabilizers on ``n`` qubits.
        css_weight: True to weigh an error by its CSS weight, or False (the default) to weigh it by
            its physical weight.
        method: ``"auto"`` (the default), ``"brouwer_zimmermann"``, or ``"connected_cluster"``
            (see :data:`CosetMethod`).  ``"connected_cluster"`` weighs errors by CSS weight only on
            CSS states.

    Raises:
        TypeError: If ``stabilizers`` is not array-like, as when a code is given instead of a
            matrix.
        ValueError: If the stabilizers are not a binary matrix with ``2 n`` columns, or are not
            ``n`` independent commuting stabilizers, as when they define a code that encodes
            qubits; if the method is unknown; or if the connected-cluster method would weigh errors
            on a state that is not CSS by their CSS weight.
    """

    def __init__(
        self,
        stabilizers: npt.ArrayLike,
        *,
        css_weight: bool = False,
        method: CosetMethod = "auto",
    ) -> None:
        # Check the arguments, and keep those that the properties return.
        coset_search_method = _get_coset_search_method(method)
        css_weight = bool(css_weight)
        self._css_weight = css_weight
        matrix = _as_state_stabilizers(stabilizers)
        self._num_qubits = n = len(matrix)

        # Read the active tuning, and prepare the searches.
        tuning = get_tuning()
        # A batch query moves to threads once its queries on the calling thread have done the split
        # work (Section 4.5 of brouwer_zimmermann_code.md), in the units that the comment of
        # ``search_chunk_exponent`` in ``_tuning.py`` defines.
        num_threads = _threads.get_num_threads(tuning.num_threads)
        split_work = min(num_threads << tuning.search_chunk_exponent, _threads.NO_WORK_LIMIT)
        self._thread_options = {"num_threads": num_threads, "split_work": split_work}
        cluster_search = functools.partial(
            ConnectedClusterCosetSearch,
            max_rounds=tuning.max_coset_sparsifying_rounds,
            rounds_without_improvement=tuning.sparsifying_rounds_without_improvement,
            restrict_viable_symbols=tuning.restrict_viable_symbols,
            stabilizer_overlap_cuts=tuning.stabilizer_overlap_cuts,
        )
        # Exactly one of the BZ searches and the connected-cluster searches is nonempty.
        self._brouwer_zimmermann_searches: tuple[BrouwerZimmermannCosetSearch, ...] = ()
        self._connected_cluster_searches: tuple[ConnectedClusterCosetSearch, ...] = ()

        # Search the Pauli errors themselves, unless the CSS weight on a CSS state lets the X and Z
        # parts of the errors be searched separately.
        combinations = _get_css_combinations(matrix) if css_weight else None
        if combinations is None:
            if coset_search_method == "brouwer_zimmermann":
                self._brouwer_zimmermann_searches = (get_pauli_search(matrix, css_weight, tuning),)
            elif css_weight:
                raise ValueError(
                    "The connected-cluster method weighs errors by CSS weight only on CSS states;"
                    " use css_weight=False or the default method"
                )
            else:
                # The stabilizers commute with each other, so the sparser stabilizers have a zero
                # syndrome and are known words for the overlap cuts (Section 3.8 of
                # connected_cluster.md).
                search = cluster_search(matrix, symplectic=True)
                search.use_query_overlap_words(search.checks)
                self._connected_cluster_searches = (search,)
            return

        # Row-reduce the X parts of the pure-X stabilizers and the Z parts of the pure-Z ones.  The
        # reduced rows depend only on the stabilizer group, so they key the cache of BZ searches.
        x_combinations, z_combinations = combinations
        x_rows, x_transform, _ = row_reduce_binary(
            multiply_binary(x_combinations, matrix[:, :n]),
            np.eye(len(x_combinations), dtype=np.uint8),
        )
        z_rows, z_transform, _ = row_reduce_binary(
            multiply_binary(z_combinations, matrix[:, n:]),
            np.eye(len(z_combinations), dtype=np.uint8),
        )
        assert x_transform is not None and z_transform is not None
        if coset_search_method == "connected_cluster":
            # The Z parts of the pure-Z stabilizers check X parts, and vice versa.
            x_part_search = cluster_search(z_rows, symplectic=False)
            z_part_search = cluster_search(x_rows, symplectic=False)
            # The X parts of the pure-X stabilizers, which the other search checks with, commute
            # with the pure-Z stabilizers, so they have a zero syndrome and are known words for the
            # overlap cuts, and likewise for Z parts (Section 3.8 of connected_cluster.md).
            x_part_search.use_query_overlap_words(z_part_search.checks)
            z_part_search.use_query_overlap_words(x_part_search.checks)
            self._connected_cluster_searches = (x_part_search, z_part_search)
        else:
            self._brouwer_zimmermann_searches = get_css_searches(x_rows, z_rows, tuning)

        # Row ``j`` of ``x_syndrome_map`` selects the given stabilizers whose product is the pure-Z
        # stabilizer with Z part ``z_rows[j]``, so the map takes the syndrome of an error to that of
        # its X part (problems.md, "The coset-leader search"), and likewise for Z.
        x_syndrome_map = multiply_binary(z_transform, z_combinations)
        z_syndrome_map = multiply_binary(x_transform, x_combinations)
        # Only the batch queries of the BZ searches read the packed columns of the maps.
        if self._brouwer_zimmermann_searches:
            self._map_columns = (
                pack_bits(np.ascontiguousarray(x_syndrome_map.T)),
                pack_bits(np.ascontiguousarray(z_syndrome_map.T)),
            )

        # Entry ``i`` of ``_joint_map_columns`` holds column ``i`` of the X map in its low
        # ``_num_x_syndrome_bits`` bits, and that of the Z map above them.  So ``_split_syndrome``
        # maps a syndrome by XORing the entries of its set bits, which is cheap for few set bits.
        self._num_x_syndrome_bits = len(z_rows)
        self._joint_map_columns = row_ints(
            np.ascontiguousarray(np.vstack([x_syndrome_map, z_syndrome_map]).T)
        )

    @property
    def num_qubits(self) -> int:
        """The number of qubits ``n`` of the state, which is also its number of stabilizers."""
        return self._num_qubits

    @property
    def css_weight(self) -> bool:
        """Whether the search weighs an error by its CSS weight, rather than its physical weight."""
        return self._css_weight

    def get_min_weight_bound(
        self, syndrome: int | npt.ArrayLike, max_weight: int
    ) -> MinWeightBound:
        """Decide whether some error with a syndrome has weight at most ``max_weight``.

        Args:
            syndrome: The syndrome, packed into an integer, or a binary vector with one entry for
                each stabilizer.
            max_weight: The largest weight that the query accepts, a nonnegative integer.

        Returns:
            A ``MinWeightBound`` for the class of the syndrome.

        Raises:
            TypeError: If ``max_weight``, or a syndrome that is not a vector, is not an integer, or
                is a bool.
            ValueError: If ``max_weight`` is negative, if a packed syndrome is negative or has a
                bit at or above ``num_qubits``, or if a syndrome vector is not binary or does not
                have one entry for each stabilizer.
        """
        packed_syndrome = self._pack_syndrome(syndrome)
        max_weight = self._as_max_weight(max_weight)
        searches = self._connected_cluster_searches or self._brouwer_zimmermann_searches
        if len(searches) == 1:
            return searches[0].get_min_weight_bound(packed_syndrome, max_weight)
        x_syndrome, z_syndrome = self._split_syndrome(packed_syndrome)
        return self._join_part_bounds(
            searches[0].get_min_weight_bound(x_syndrome, max_weight),
            searches[1].get_min_weight_bound(z_syndrome, max_weight),
        )

    def find_first_not_within_max_weight(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> tuple[int, MinWeightBound | None]:
        """Find the first syndrome whose class has no error of weight at most ``max_weight``.

        Args:
            syndromes: A batch of syndromes, such as ``pack_syndromes`` makes.
            max_weight: The largest weight that the queries accept, a nonnegative integer.

        Returns:
            The index of the first syndrome whose class has no error of weight at most
            ``max_weight``, and a ``MinWeightBound`` for that syndrome, or the number of syndromes
            and None if there is no such syndrome.

        Raises:
            TypeError: If ``max_weight`` is not an integer, or is a bool, or if ``syndromes`` is not
                a ``uint64`` array.
            ValueError: If ``max_weight`` is negative, if the batch does not have
                ``ceil(num_qubits / 64)`` words per row, or if a syndrome has a bit at or above
                ``num_qubits``.
        """
        self._check_syndromes(syndromes)
        max_weight = self._as_max_weight(max_weight)

        # The connected-cluster searches answer one syndrome at a time.
        if self._connected_cluster_searches:
            return find_first_cluster_bound(
                self._connected_cluster_searches,
                syndromes,
                max_weight,
                self._split_syndrome,
                self._join_part_bounds,
                **self._thread_options,
            )

        # With one search, take its bound and the lightest error that it weighed.
        if len(self._brouwer_zimmermann_searches) == 1:
            return self._brouwer_zimmermann_searches[0].find_first_not_within_max_weight(
                syndromes, max_weight, **self._thread_options
            )

        # With two searches, a class has no error within ``max_weight`` exactly if its X part or its
        # Z part has none (problems.md, "The coset-leader search").  So the Z search covers only the
        # syndromes before the first whose X part has none, and ``get_min_weight_bound`` joins the
        # bounds of both parts.
        index = len(syndromes)
        for search, map_columns in zip(self._brouwer_zimmermann_searches, self._map_columns):
            part_syndromes = _map_syndromes(syndromes[:index], map_columns)
            index, _ = search.find_first_not_within_max_weight(
                part_syndromes, max_weight, **self._thread_options
            )
        if index == len(syndromes):
            return index, None
        return index, self.get_min_weight_bound(words_to_int(syndromes[index]), max_weight)

    def get_min_weights(
        self, syndromes: npt.NDArray[np.uint64], max_weight: int
    ) -> npt.NDArray[np.int64]:
        """Find the weight of the lightest errors with each syndrome, up to ``max_weight + 1``.

        Args:
            syndromes: A batch of syndromes, such as ``pack_syndromes`` makes.
            max_weight: The largest weight to find, a nonnegative integer.

        Returns:
            For each syndrome, the weight of the lightest errors with the syndrome, or
            ``max_weight + 1`` if that is smaller.

        Raises:
            TypeError: If ``max_weight`` is not an integer, or is a bool, or if ``syndromes`` is not
                a ``uint64`` array.
            ValueError: If ``max_weight`` is negative, if the batch does not have
                ``ceil(num_qubits / 64)`` words per row, or if a syndrome has a bit at or above
                ``num_qubits``.
        """
        self._check_syndromes(syndromes)
        max_weight = self._as_max_weight(max_weight)

        # The connected-cluster searches answer one syndrome at a time.
        if self._connected_cluster_searches:
            return get_min_cluster_weights(
                self._connected_cluster_searches,
                syndromes,
                max_weight,
                self._split_syndrome,
                **self._thread_options,
            )

        # Every search starts from weight zero, except those of the Z parts below.
        floors = np.zeros(len(syndromes), dtype=np.int64)

        # With one search, take its weights.
        if len(self._brouwer_zimmermann_searches) == 1:
            return self._brouwer_zimmermann_searches[0].get_min_weights(
                syndromes, max_weight, floors, **self._thread_options
            )

        # With two searches, search the Z parts only of the classes whose X parts are within
        # ``max_weight``, with the weight of the X part as the floor (problems.md, "The coset-leader
        # search").
        x_search, z_search = self._brouwer_zimmermann_searches
        x_syndromes = _map_syndromes(syndromes, self._map_columns[0])
        weights = x_search.get_min_weights(x_syndromes, max_weight, floors, **self._thread_options)
        rows = np.flatnonzero(weights <= max_weight)
        z_syndromes = _map_syndromes(syndromes[rows], self._map_columns[1])
        z_weights = z_search.get_min_weights(
            z_syndromes, max_weight, weights[rows], **self._thread_options
        )
        weights[rows] = np.maximum(weights[rows], z_weights)
        return weights

    def pack_syndromes(self, syndromes: npt.ArrayLike) -> npt.NDArray[np.uint64]:
        """Pack binary syndromes into a batch of syndromes, the input of the batch queries.

        Args:
            syndromes: Binary syndromes, one per row, each with one entry for each stabilizer, or a
                single syndrome as a vector.

        Returns:
            A batch of the syndromes, a ``uint64`` array with one row per syndrome and
            ``ceil(num_qubits / 64)`` words per row, in which bit ``k`` of word ``j`` of a row is
            entry ``64 j + k`` of its syndrome.

        Raises:
            TypeError: If ``syndromes`` is not array-like.
            ValueError: If the syndromes are not binary, or do not have one entry for each
                stabilizer.
        """
        matrix = as_binary_matrix(syndromes, "syndromes", num_cols=self.num_qubits)
        return pack_bits(np.ascontiguousarray(matrix))

    def unpack_pauli(self, pauli: int) -> npt.NDArray[np.uint8]:
        """Unpack a packed Pauli error, such as a representative of a ``MinWeightBound``.

        Args:
            pauli: A Pauli error, packed into an integer with its X part in bits ``0`` to ``n - 1``
                and its Z part in bits ``n`` to ``2 n - 1``, for ``n = num_qubits``.

        Returns:
            The error as a binary vector of length ``2 n`` in ``[X|Z]`` order.

        Raises:
            TypeError: If ``pauli`` is not an integer, or is a bool.
            ValueError: If ``pauli`` is negative or has a bit at or above ``2 n``.
        """
        value = as_integer(pauli, "pauli")
        num_bits = 2 * self.num_qubits
        if not 0 <= value < 1 << num_bits:
            raise ValueError(f"The packed Pauli error {value} is out of range for {num_bits} bits")
        data = np.frombuffer(value.to_bytes((num_bits + 7) // 8, "little"), dtype=np.uint8)
        return np.unpackbits(data, count=num_bits, bitorder="little")

    def _pack_syndrome(self, syndrome: int | npt.ArrayLike) -> int:
        """Return the syndrome of a query packed into an integer, after checking it."""
        if np.ndim(syndrome) == 0:
            packed_syndrome = as_integer(cast(SupportsIndex, syndrome), "syndrome")
            if not 0 <= packed_syndrome < 1 << self.num_qubits:
                raise ValueError(
                    f"The packed syndrome {packed_syndrome} is out of range for"
                    f" {self.num_qubits} stabilizers"
                )
            return packed_syndrome
        if np.shape(syndrome) != (self.num_qubits,):
            raise ValueError(
                "A syndrome must be an integer, or a binary vector with one entry for each of the"
                f" {self.num_qubits} stabilizers"
            )
        matrix = as_binary_matrix(syndrome, "syndrome", num_cols=self.num_qubits)
        return words_to_int(pack_bits(np.ascontiguousarray(matrix))[0])

    def _as_max_weight(self, max_weight: int) -> int:
        """Return the maximum weight of a query as a Python integer, at most ``num_qubits``.

        No error is heavier than ``num_qubits``, so lowering a larger maximum weight to it changes
        no answer.
        """
        max_weight = as_integer(max_weight, "max_weight")
        if max_weight < 0:
            raise ValueError(f"max_weight must be nonnegative, not {max_weight}")
        return min(max_weight, self.num_qubits)

    def _check_syndromes(self, syndromes: npt.NDArray[np.uint64]) -> None:
        """Raise an error unless a batch holds packed syndromes of this search.

        The compiled searches do not check that the set bits of a syndrome are below ``num_qubits``.
        """
        num_words, num_bits = get_num_words(self.num_qubits), self.num_qubits % 64
        if not isinstance(syndromes, np.ndarray) or syndromes.dtype != np.uint64:
            raise TypeError(
                "A batch of syndromes must be a uint64 array, such as pack_syndromes makes"
            )
        if syndromes.ndim != 2 or syndromes.shape[1] != num_words:
            raise ValueError(
                f"A batch of syndromes must have {num_words} words per row, not shape"
                f" {syndromes.shape}, as pack_syndromes makes it"
            )
        if num_bits and np.any(syndromes[:, -1] >> np.uint64(num_bits)):
            raise ValueError("A syndrome of the batch is out of range")

    def _split_syndrome(self, syndrome: int) -> tuple[int, int]:
        """Map a syndrome for the given stabilizers to the syndromes of the parts of its errors."""
        part_syndromes = xor_selected(self._joint_map_columns, syndrome)
        x_syndrome = part_syndromes & ((1 << self._num_x_syndrome_bits) - 1)
        return x_syndrome, part_syndromes >> self._num_x_syndrome_bits

    def _join_part_bounds(self, x_bound: MinWeightBound, z_bound: MinWeightBound) -> MinWeightBound:
        """Join the bounds of the cosets of X parts and of Z parts into a bound of a syndrome class.

        The smallest CSS weight of a class is the larger of the smallest weights of its X and Z
        parts (problems.md, "The coset-leader search").
        """
        return MinWeightBound(
            x_bound.within_max_weight and z_bound.within_max_weight,
            x_bound.representative | (z_bound.representative << self.num_qubits),
            max(x_bound.lower_bound, z_bound.lower_bound),
        )


def _get_coset_search_method(
    method: CosetMethod,
) -> Literal["brouwer_zimmermann", "connected_cluster"]:
    """Validate a coset method, and return the method that the coset searches use."""
    if not isinstance(method, str) or method not in get_args(CosetMethod):
        raise ValueError(
            f"Unknown coset method {method!r}; choose from"
            f" {', '.join(map(repr, get_args(CosetMethod)))}."
        )
    return "brouwer_zimmermann" if method == "auto" else method


def _as_state_stabilizers(stabilizers: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Check that stabilizers define a pure stabilizer state, and return them as a binary matrix."""
    matrix = as_binary_matrix(stabilizers, "stabilizers")
    num_qubits, odd = divmod(matrix.shape[1], 2)
    if not num_qubits or odd:
        raise ValueError(
            "The stabilizers must have 2 n columns in [X|Z] order, for n qubits, not"
            f" {matrix.shape[1]}"
        )

    # Swapping the halves of the stabilizers turns their symplectic products into dot products.
    checks = swap_pauli_halves(matrix)
    if np.any(multiply_binary(checks, matrix.T)):
        raise ValueError(
            "The stabilizers must commute, unlike the gauge generators of a subsystem code: pass the"
            " stabilizers of a state, such as a state of the code with its gauge and logical qubits"
            " fixed"
        )

    # Commuting stabilizers of rank ``n`` define a state, and of a lower rank a code.
    rank = len(row_reduce_binary(matrix)[2])
    if rank < num_qubits:
        raise ValueError(
            f"The stabilizers have rank {rank} on {num_qubits} qubits, so they define a code that"
            f" encodes qubits, not a state: pass {num_qubits} independent commuting stabilizers,"
            " which define a pure stabilizer state"
        )
    if len(matrix) > num_qubits:
        raise ValueError(
            f"The stabilizers must be independent, but {len(matrix)} of them have rank {rank}"
        )
    return matrix


def _get_css_combinations(
    stabilizers: npt.NDArray[np.uint8],
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]] | None:
    """Combine stabilizers into pure-X and pure-Z generators, or return None if they are not CSS.

    The stabilizers must be independent.  The state is CSS exactly if the dimensions of its pure-X
    and pure-Z stabilizers add up to ``n`` (problems.md, "The coset-leader search").

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

    Each set bit of a syndrome must index a column, but nothing checks that.
    """
    mapped = np.zeros((len(syndromes), map_columns.shape[1]), dtype=np.uint64)
    for row in range(len(syndromes)):
        xor_selected_rows(syndromes[row], map_columns, mapped[row])
    return mapped
