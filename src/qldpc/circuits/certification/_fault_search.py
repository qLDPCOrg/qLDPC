# SPDX-License-Identifier: Apache-2.0

"""Exact search for an accepted fault of the smallest fault weight that violates the bound.

The search sees only the flags and the syndrome of each primitive fault, never a circuit.
``fault_propagation.md`` defines the search's terms, describes the search, and proves that it is
exact, and ``FaultLister`` defines the terms of its implementation.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Callable, Collection, Generator, Iterator, Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._binary import (
    get_num_words,
    get_rank_packed,
    popcount_word,
    trailing_zeros_word,
    words_to_int,
)
from qldpc._compilation import always_compiled
from qldpc.codes.distance import CosetLeaderSearch, CosetMethod, MinWeightBound
from qldpc.codes.distance._threads import get_num_threads, iterate_in_order
from qldpc.codes.distance._tuning import get_tuning

# The number of choices of members of a class tuple whose syndromes ``_expand_class_tuples``
# remembers, to skip the faults that complete a later choice with a remembered syndrome.
_MAX_REMEMBERED_CHOICES = 2**12

# The number of faults in the first batch of a call of ``FaultLister.find_first``.  Each batch
# doubles in size, so that an early violation is found quickly, and larger later batches spread the
# overhead of each call of the oracle over more faults.
_INITIAL_BATCH_SIZE = 2**6

# The largest number of faults in a batch.
_BATCH_SIZE = 2**12

# The number of class codewords (see ``FaultLister``) after which ``_list_by_shared_flags`` returns
# once it has finished a pass over a root, to be resumed by its next call.  A thread hands on its
# class codewords in batches of at most this many, so that the batches that ``iterate_in_order``
# holds take bounded memory.
_CODEWORD_BATCH_SIZE = 2**10

# The work (see ``_PROBES_PER_REDUCTION``) after which ``_list_by_shared_flags`` also returns on
# threads, so that a thread checks often whether to stop (see ``fault_propagation.md#sec-threads``).
_PROBES_PER_CALL = 2**20

# The walk by shared flags counts its work in probes of the map of classes by hash, and counts a
# reduction of flags as this many probes (``fault_propagation.md#sec-listing-on-threads`` lists what
# the walk is charged for).
_PROBES_PER_REDUCTION = 64

# The number of rows of the array into which the walk by shared flags lists the class codewords of
# a pass over a root, which takes at most ``8 m`` MiB for fault weight ``m``.  Sorting a full array
# takes about as much again and 16 MiB more, so about ``16 m + 16`` MiB in all.  A root with more
# class codewords takes several passes, each listing the next of them in lexicographic order
# (``fault_propagation.md#sec-shared-flags``).
_MAX_ROOT_CODEWORDS = 2**20

# A listing of class codewords moves to threads once the calling thread, listing alone, has done
# this much work for each thread, in probes, and the work left is estimated to be at least as much,
# so that small listings never start threads.
_SPLIT_PROBES_PER_THREAD = 2**18

# The number of syndromes that the table of listed syndromes starts with room for.
_INITIAL_SYNDROME_TABLE_SIZE = 2**10

# An odd 32-bit constant, near 2**32 divided by the golden ratio.  The slot of a syndrome in a table
# comes from the product of this and the low 32 bits of its hash, which mixes these bits.  The
# hashes are linear, so their bits alone can place the syndromes of a table in long runs of slots,
# which makes lookups slow.
_HASH_MULTIPLIER = np.uint64(0x9E3779B9)

####################################################################################################
# exact search for a violating fault


class Violation(NamedTuple):
    """An accepted fault that violates the bound.

    Attributes:
        fault_indices: The DEM indices of the primitive faults that make up the fault.
        syndrome: The packed syndrome of the fault's output error.
        bound: The oracle's answer for this syndrome at the fault weight: a Pauli error with the
            syndrome, and a lower bound on the residual weight that exceeds the fault weight.
    """

    fault_indices: tuple[int, ...]
    syndrome: int
    bound: MinWeightBound


def find_smallest_violation(
    flags: Sequence[int],
    syndromes: Sequence[int],
    stabilizers: npt.ArrayLike,
    max_weight: int | None,
    css_weight: bool,
    memory_limit: int = 2**30,  # 1 GiB
    method: CosetMethod = "auto",
) -> Violation | None:
    """Find a violating accepted fault of the smallest fault weight, or None if there is none.

    ``flags`` and ``syndromes`` hold, for each primitive fault in DEM order, its flags or its
    syndrome packed into an integer.  ``max_weight``, a positive integer or None, limits the fault
    weight.  ``stabilizers``, ``css_weight``, and ``method`` configure the oracle of the search, a
    ``CosetLeaderSearch``.  ``fault_propagation.md#sec-first-violation`` proves that the search
    finds a violating accepted fault of the smallest fault weight, up to ``max_weight``, if there is
    one.
    """
    # Keep the first primitive fault of each distinct nonzero signature (see
    # ``fault_propagation.md#sec-dropped-signatures``).
    signatures: dict[tuple[int, int], int] = {}
    for index, signature in enumerate(zip(flags, syndromes, strict=True)):
        if any(signature):
            signatures.setdefault(signature, index)

    # Repeated flags do not change the rank of the matrix P of the flags, so each value enters it
    # once.  Faults are checked only up to fault weight min(max_weight, rank(P) + 1), so with
    # ``max_weight`` the rank is computed only up to ``max_weight - 1`` (see
    # ``fault_propagation.md#sec-fault-weight-bound``).  The limit is also at most
    # ``len(flag_rows)``, which bounds the rank, so that compiled code receives a 64-bit integer.
    distinct_flags = list(dict.fromkeys(flag for flag, _ in signatures))
    flag_rows = _to_words(distinct_flags, _get_num_words_for_values(distinct_flags))
    if max_weight is None:
        max_fault_weight = get_rank_packed(flag_rows) + 1
    else:
        max_fault_weight = get_rank_packed(flag_rows, min(max_weight - 1, len(flag_rows))) + 1

    oracle = CosetLeaderSearch(stabilizers, css_weight=css_weight, method=method)
    return _search_irreducible_faults(signatures, max_fault_weight, oracle, memory_limit)


def _search_irreducible_faults(
    signatures: dict[tuple[int, int], int],
    max_fault_weight: int,
    oracle: CosetLeaderSearch,
    memory_limit: int = 2**30,  # 1 GiB
) -> Violation | None:
    """Check the irreducible accepted faults that can violate the bound, by increasing fault weight.

    The signatures map the (flags, syndrome) of each kept primitive fault to its DEM index, and
    ``max_fault_weight`` is the largest fault weight to check.  Return a violating fault of the
    smallest fault weight, or None if there is none.
    """
    # Check each primitive fault with zero flags alone (see
    # ``fault_propagation.md#sec-listing-once``), and keep the others for ``FaultLister``.
    flagged: list[tuple[int, int, int]] = []
    for (flag, syndrome), index in signatures.items():
        if flag:
            flagged.append((flag, syndrome, index))
            continue
        failure = _check_fault((index,), syndrome, oracle)
        if failure is not None:
            return failure

    # Split the other primitive faults into heavy and light ones.  A fault that contains no heavy
    # primitive fault satisfies the bound (see ``fault_propagation.md#sec-heavy-primitive-fault``),
    # so the search stops if none is heavy.  Heaviness depends only on the syndrome, and any error
    # with the syndrome serves as a representative (see
    # ``fault_propagation.md#sec-checking-one-fault``), so the search queries the oracle once for
    # each distinct syndrome.
    bounds_by_syndrome = {
        distinct_syndrome: oracle.get_min_weight_bound(distinct_syndrome, 1)
        for distinct_syndrome in dict.fromkeys(syndrome for _, syndrome, _ in flagged)
    }
    heavy: list[tuple[int, int, int, int]] = []
    light: list[tuple[int, int, int, int]] = []
    for flag, syndrome, index in flagged:
        bound = bounds_by_syndrome[syndrome]
        primitive = (flag, syndrome, index, bound.representative)
        (light if bound.within_max_weight else heavy).append(primitive)
    if not heavy:
        return None

    # Give each primitive fault a representative of low Pauli weight, so that ``FaultLister`` skips
    # more faults (see ``fault_propagation.md#sec-checking-one-fault``).  A light primitive fault's
    # representative has Pauli weight at most one.  A heavy primitive fault whose syndrome is the
    # sum of the syndromes of two light ones gets the product of their representatives.  The pair
    # depends only on the syndrome, so the search runs once for each distinct heavy syndrome.
    # Compiled code does it, since its time is proportional to the number of distinct heavy
    # syndromes times the number of light ones.  Any other heavy primitive fault keeps the error
    # that the oracle returned for it.
    light_errors = {syndrome: error for _, syndrome, _, error in light}
    light_syndromes = list(light_errors)
    heavy_syndromes = list(dict.fromkeys(syndrome for _, syndrome, _, _ in heavy))
    num_words = _get_num_words_for_values(light_syndromes + heavy_syndromes)
    pairs = _get_light_pairs(
        _hashed_rows(heavy_syndromes, num_words), _hashed_rows(light_syndromes, num_words)
    )
    pair_errors = {
        syndrome: light_errors[light_syndromes[first]] ^ light_errors[light_syndromes[second]]
        for syndrome, (first, second) in zip(heavy_syndromes, pairs.tolist())
        if first >= 0
    }
    heavy = [
        (flag, syndrome, index, pair_errors.get(syndrome, error))
        for flag, syndrome, index, error in heavy
    ]

    # Order the primitive faults with the heavy ones first, and each group in DEM order, so that the
    # faults to check are the ones that contain one of the first ``len(heavy)`` primitive faults.
    primitives = heavy + light

    # The oracle needs only small tables, so the lister may use the whole memory limit.  It lists on
    # as many threads as the oracle's searches use, a number read here, on the calling thread, since
    # the threads of a listing must not read the tuning
    # (``fault_propagation.md#sec-thread-requirements``).
    lister = FaultLister(
        [flag for flag, _, _, _ in primitives],
        [syndrome for _, syndrome, _, _ in primitives],
        [representative for _, _, _, representative in primitives],
        oracle.num_qubits,
        oracle.css_weight,
        len(heavy),
        memory_limit,
        num_threads=get_num_threads(get_tuning().num_threads),
    )

    # Check the faults by increasing fault weight, and keep the oracle's bound for a violating
    # syndrome, so that it is not searched again.
    bounds: list[MinWeightBound] = []

    def find_violation(syndromes: npt.NDArray[np.uint64], max_weight: int) -> int:
        index, bound = oracle.find_first_not_within_max_weight(syndromes, max_weight)
        if bound is not None:
            bounds.append(bound)
        return index

    for fault_weight in range(2, max_fault_weight + 1):
        violation = lister.find_first(fault_weight, find_violation)
        if violation is not None:
            positions, syndrome = violation
            indices = tuple(primitives[jj][2] for jj in positions)
            return Violation(indices, syndrome, bounds[-1])
    return None


def _check_fault(
    indices: tuple[int, ...], syndrome: int, oracle: CosetLeaderSearch
) -> Violation | None:
    """Return the fault with these indices and syndrome if it violates the bound."""
    bound = oracle.get_min_weight_bound(syndrome, len(indices))
    if bound.within_max_weight:
        return None
    return Violation(indices, syndrome, bound)


####################################################################################################
# listing irreducible accepted faults


class FaultLister:
    """Lists each irreducible accepted fault of given primitive faults that has a start member.

    The lister refers to each primitive fault by its position in the arguments.  Every primitive
    fault must have nonzero flags, and the first ``num_starts`` are the start members.  The search
    passes its heavy primitive faults first, as the start members.
    ``fault_propagation.md#sec-listing-once`` defines flag classes, start classes, and class
    codewords, and proves that each such fault is listed once.  The classes are numbered in the
    order of their first members.  A class tuple is a row of classes from which the lister expands
    faults: a start class twice for fault weight two, and a class codeword that starts with a start
    class for larger fault weights.

    Each class is a row of 64-bit words: the hash of its flags, then its flags.  Each primitive
    fault is a row of the hash of its syndrome, its syndrome, and the X and Z parts of its
    representative.  The hashes and the other parts are linear, so the row of a set of classes or
    of primitive faults is the sum of their rows.

    For fault weights of three or more, the lister lists the class codewords of each start class,
    the root, in the order of the roots, by the walk by shared flags (``_list_by_shared_flags``).
    The walk adds only classes that share a flag bit with the sum of the flags of the classes chosen
    so far, and finds the class that completes each class codeword in the map of classes by hash:
    the classes sorted stably by the hash of their flags, with a hash map from each hash to its
    first entry (``fault_propagation.md#sec-shared-flags``).  It lists the class codewords of each
    root in lexicographic order, in one or more passes over the root.  The table of syndromes of
    listed faults (see ``find_first``) grows within the memory limit, up to 2**31 syndromes.

    On several threads, a costly listing of class codewords moves to threads, while the calling
    thread expands the class codewords into faults in the serial order, that of a listing on a
    single thread, so that the lister checks the same faults as on one thread
    (``fault_propagation.md#sec-threads``).  Each thread allocates arrays for its listing as large
    as the calling thread's.

    Args:
        flags: The flags of each primitive fault, packed into an integer.
        syndromes: The syndrome of each primitive fault, packed into an integer.
        representatives: For each primitive fault, a Pauli error with its syndrome, packed as in
            ``CosetLeaderSearch``.  Lighter Pauli errors let ``find_first`` skip more faults.
        memory_limit: The approximate number of bytes for the table of syndromes.  The map of
            classes by hash, the index of their flag bits, and the arrays into which each thread
            lists class codewords are extra (see ``_MAX_ROOT_CODEWORDS``).
        num_threads: The number of threads of a costly listing, counting the calling thread; with
            fewer than two, the calling thread lists alone.

    Raises:
        ValueError: If a primitive fault has zero flags.
    """

    def __init__(
        self,
        flags: Sequence[int],
        syndromes: Sequence[int],
        representatives: Sequence[int],
        num_qubits: int,
        css_weight: bool,
        num_starts: int,
        memory_limit: int,
        num_threads: int = 1,
    ) -> None:
        # Group the primitive faults into classes by their flags, each with its members in order,
        # and order the classes by their first members, so that the start classes come first.  The
        # walk by shared flags needs the flags of the classes to be nonzero, which is checked here,
        # and pairwise distinct, which grouping ensures (``fault_propagation.md#sec-shared-flags``).
        if not all(flags):
            raise ValueError("Every primitive fault of a fault lister must have nonzero flags")
        member_lists_by_flags: dict[int, list[int]] = {}
        for position, flag in enumerate(flags):
            member_lists_by_flags.setdefault(flag, []).append(position)
        member_lists = list(member_lists_by_flags.values())
        self._num_start_classes = sum(members[0] < num_starts for members in member_lists)
        self._class_offsets = np.cumsum(
            [0] + [len(members) for members in member_lists], dtype=np.int64
        )
        self._class_members = np.array(
            [position for members in member_lists for position in members], dtype=np.int64
        )

        # Build the rows of the classes and of the primitive faults, as the class docstring
        # describes, and the walk's index from each flag bit to its classes and map of classes by
        # hash.
        class_flags = [flags[members[0]] for members in member_lists]
        flag_words = _to_words(class_flags, _get_num_words_for_values(class_flags))
        self._class_rows = np.hstack([_hash_words(flag_words)[:, None], flag_words])
        self._bit_index = _get_bit_index(flag_words)
        self._class_map = _get_class_map(self._class_rows[:, 0])
        # The syndromes, and the X and Z parts of the representatives, take a word for every 64
        # output qubits, as the batches of ``CosetLeaderSearch`` need.
        mask = (1 << num_qubits) - 1
        num_qubit_words = _get_num_words_for_values([mask])
        syndrome_words = _to_words(syndromes, num_qubit_words)
        x_parts = [representative & mask for representative in representatives]
        z_parts = [representative >> num_qubits for representative in representatives]
        self._primitive_rows = np.hstack(
            [
                _hash_words(syndrome_words)[:, None],
                syndrome_words,
                _to_words(x_parts, num_qubit_words),
                _to_words(z_parts, num_qubit_words),
            ]
        )
        self._num_syndrome_words = syndrome_words.shape[1]
        self._css_weight = css_weight
        self._num_starts = num_starts
        self._num_threads = num_threads

        # Start with an empty table of syndromes.  Each syndrome takes a row of its hash and its
        # words, and two slots, and the table has room for at most 2**31 syndromes, so that its
        # slots have at most 2**32 entries, as ``_find_syndrome_slot`` needs.
        syndrome_bytes = 8 * (self._num_syndrome_words + 3)
        self._max_syndromes = min(memory_limit // syndrome_bytes, 2**31)
        self._syndrome_rows, self._syndrome_slots = _get_empty_syndrome_table(
            self._num_syndrome_words
        )
        self._num_syndromes = 0

    def find_first(
        self, fault_weight: int, find_violation: Callable[[npt.NDArray[np.uint64], int], int]
    ) -> tuple[tuple[int, ...], int] | None:
        """Find the first listed fault of a fault weight that violates the bound.

        The listed faults that are not left out (see below) are passed to
        ``find_violation(syndromes, fault_weight)`` in batches of growing size (see
        ``_INITIAL_BATCH_SIZE``), one syndrome per row, with word ``j`` of a row holding bits
        ``64 j`` and up.  ``find_violation`` returns the index of the first syndrome that violates
        the bound at this fault weight, or the number of rows if there is none.

        A listed fault is left out if the product of the representatives of its primitive faults has
        Pauli weight at most its fault weight, or if its syndrome is in the table of syndromes.  The
        table holds the syndromes of faults listed earlier, by this call or by an earlier call that
        found no violation, as far as it had room.  A call that finds a violation clears the table,
        since the table can hold syndromes of faults listed after that violation, which were never
        checked.  Leaving these faults out does not change the result if the fault weight never
        decreases from one call to the next, every call uses the same ``find_violation``, and a
        syndrome that satisfies the bound at one fault weight satisfies it at every larger one.
        ``fault_propagation.md#sec-checking-one-fault`` shows why.

        Returns:
            The positions of the primitive faults of the first violating fault that was passed, in
            increasing order, and its syndrome, or None if there is none.
        """
        # A batch holds, for each fault to check, the positions of its primitive faults and its
        # syndrome.
        batch_members = np.empty((_BATCH_SIZE, fault_weight), dtype=np.int64)
        batch_syndromes = np.empty((_BATCH_SIZE, self._num_syndrome_words), dtype=np.uint64)
        batch_size = _INITIAL_BATCH_SIZE

        def check_batch(num_faults: int) -> tuple[tuple[int, ...], int] | None:
            """Check the first ``num_faults`` faults of the batch.

            Return the positions and the syndrome of the first violating fault, or None if there is
            none.
            """
            if num_faults == 0:
                return None
            index = find_violation(batch_syndromes[:num_faults], fault_weight)
            if index == num_faults:
                return None
            self._syndrome_rows, self._syndrome_slots = _get_empty_syndrome_table(
                self._num_syndrome_words
            )
            self._num_syndromes = 0
            # A row of the batch holds the primitive faults of a fault in the order of its classes,
            # so sort them.
            positions = tuple(sorted(batch_members[index].tolist()))
            syndrome = words_to_int(batch_syndromes[index])
            return positions, syndrome

        # The state of the expansion of class tuples into faults (see ``_expand_class_tuples``).
        next_member_entries = np.empty(fault_weight, dtype=np.int64)
        chosen_row_sums = np.zeros(
            (fault_weight + 1, self._primitive_rows.shape[1]), dtype=np.uint64
        )
        chosen_start_counts = np.zeros(fault_weight + 1, dtype=np.int64)
        choice_rows = np.empty(
            (_MAX_REMEMBERED_CHOICES, 2 + self._num_syndrome_words), dtype=np.uint64
        )
        choice_slots = np.full(2 * _MAX_REMEMBERED_CHOICES, -1, dtype=np.int64)
        choice_slots_used = np.empty(_MAX_REMEMBERED_CHOICES, dtype=np.int64)
        expansion_state = np.array([0, -1, 0, self._num_syndromes, 0, 0], dtype=np.int64)

        # Expand each batch of class tuples into faults.  Whenever the expansion pauses, grow the
        # table of syndromes if it should grow, and check the batch of faults if it is full.  At the
        # end, check the last batch.  Closing the batches of class tuples stops the threads that
        # list them, if any, when a violation ends the listing early or an exception is raised
        # (``fault_propagation.md#sec-thread-requirements``).
        with contextlib.closing(self._iter_class_tuples(fault_weight)) as class_tuple_batches:
            for class_tuples in class_tuple_batches:
                expansion_state[0], expansion_state[1] = 0, -1
                while expansion_state[0] < len(class_tuples):
                    _expand_class_tuples(
                        class_tuples,
                        self._class_offsets,
                        self._class_members,
                        self._num_start_classes,
                        self._num_starts,
                        self._primitive_rows,
                        self._css_weight,
                        batch_size,
                        batch_members,
                        batch_syndromes,
                        self._syndrome_rows,
                        self._syndrome_slots,
                        self._max_syndromes,
                        next_member_entries,
                        chosen_row_sums,
                        chosen_start_counts,
                        choice_rows,
                        choice_slots,
                        choice_slots_used,
                        expansion_state,
                    )
                    self._num_syndromes = int(expansion_state[3])
                    num_rows = len(self._syndrome_rows)
                    if _table_should_grow(self._num_syndromes, num_rows, self._max_syndromes):
                        self._syndrome_rows, self._syndrome_slots = _grow_syndrome_table(
                            self._syndrome_rows, self._num_syndromes
                        )
                    if expansion_state[2] == batch_size:
                        violation = check_batch(batch_size)
                        if violation is not None:
                            return violation
                        expansion_state[2] = 0
                        batch_size = min(2 * batch_size, _BATCH_SIZE)
        return check_batch(int(expansion_state[2]))

    def _iter_class_tuples(self, fault_weight: int) -> Generator[npt.NDArray[np.int64], None, None]:
        """Yield the class tuples of a fault weight in batches, one class tuple per row.

        For fault weight two, each class tuple is ``(c, c)`` for a start class ``c`` with at least
        two members.  For larger fault weights, the class tuples are the class codewords that start
        with a start class, in the serial order.  Each batch is a view of an array that a later
        batch can overwrite.  On several threads, closing the iterator stops its threads.
        """
        if fault_weight == 2:
            class_sizes = np.diff(self._class_offsets[: self._num_start_classes + 1])
            yield np.repeat(np.flatnonzero(class_sizes > 1), 2).reshape(-1, 2)
            return

        # List the class codewords of each root by the walk by shared flags.  On one thread, list
        # every root with one walk.
        def make_walk() -> _SharedFlagWalk:
            """A walk of the roots by shared flags."""
            return _SharedFlagWalk(self._class_rows, self._bit_index, self._class_map, fault_weight)

        walk = make_walk()
        num_roots = self._num_start_classes
        if self._num_threads < 2:
            yield from walk.list_range(0, num_roots)
            return

        # On several threads, the work items are the roots.  List the first roots alone, in ranges
        # of 1, 2, 4, and more roots, until the work shows that the listing is costly (see
        # ``fault_propagation.md#sec-threads``).  If the work left, estimated in proportion to the
        # weights of the roots (see ``_get_root_weights``), is less than
        # ``_SPLIT_PROBES_PER_THREAD`` for each thread, list the rest alone too.
        cumulative_weights = np.cumsum(
            _get_root_weights(len(self._class_rows), num_roots, fault_weight)
        )
        split_work = self._num_threads * _SPLIT_PROBES_PER_THREAD
        num_listed, num_next = 0, 1
        while num_listed < num_roots and walk.work < split_work:
            end = min(num_roots, num_listed + num_next)
            yield from walk.list_range(num_listed, end)
            num_listed, num_next = end, 2 * num_next
        listed_weight = cumulative_weights[num_listed - 1] if num_listed else 0.0
        weight_left = cumulative_weights[-1] - listed_weight if num_roots else 0.0
        if num_listed < num_roots and walk.work * weight_left < split_work * listed_weight:
            yield from walk.list_range(num_listed, num_roots)
            num_listed = num_roots

        # List the other roots on threads, each with its own walk, and yield their class codewords
        # in the order of the roots, while the threads list later roots.

        def start_walk(
            stop: threading.Event,
        ) -> Callable[[int, int], Iterator[npt.NDArray[np.int64]]]:
            """Start the walk of a thread, and return how it lists a range of the other roots."""
            thread_walk = make_walk()

            def list_range(start: int, end: int) -> Iterator[npt.NDArray[np.int64]]:
                """List the class codewords of a range of the other roots, a batch at a time.

                Stop early, leaving the walk unfinished, once the stop event is set, after which
                the thread lists no other range.
                """
                # Copy each batch in slices of at most ``_CODEWORD_BATCH_SIZE`` class codewords,
                # since the walk overwrites its array while the calling thread reads earlier ones,
                # and the batches that ``iterate_in_order`` holds should take little memory.  Hand
                # on no empty batch, since empty batches would count toward its cap on held batches
                # and stall the threads that list ahead; check the stop event after each call of the
                # walk instead (``fault_propagation.md#sec-thread-requirements``).
                for class_codewords in thread_walk.list_range(
                    num_listed + start, num_listed + end, _PROBES_PER_CALL
                ):
                    for first in range(0, len(class_codewords), _CODEWORD_BATCH_SIZE):
                        yield class_codewords[first : first + _CODEWORD_BATCH_SIZE].copy()
                    if stop.is_set():
                        return

            return list_range

        if num_listed < num_roots:
            yield from iterate_in_order(
                num_roots - num_listed,
                self._num_threads,
                start_walk,
                cumulative_weights=cumulative_weights[num_listed:] - listed_weight,
            )


class _SharedFlagWalk:
    """A walk by shared flags on one thread, with the arrays of ``_list_by_shared_flags``.

    ``class_rows`` are the rows of the classes of a ``FaultLister``, ``bit_index`` the index from
    each flag bit to its classes (see ``_get_bit_index``), and ``class_map`` the map of classes by
    hash (see ``_get_class_map``).  A walk lists ranges of roots, one range at a time, and counts
    its work over all of them.
    """

    def __init__(
        self,
        class_rows: npt.NDArray[np.uint64],
        bit_index: tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]],
        class_map: tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64], npt.NDArray[np.int64]],
        fault_weight: int,
    ) -> None:
        self._class_rows = class_rows
        self._bit_index = bit_index
        self._class_map = class_map

        # Make the arrays of ``_list_by_shared_flags``, of which ``levels``, ``chosen_row_sums``,
        # ``basis``, and ``pivots`` have a column, row, or entry for each class of a state, at most
        # ``m - 2`` for fault weight ``m``.  The array of class codewords takes memory only for the
        # rows that the walk writes.
        num_classes, num_words = class_rows.shape
        num_levels = fault_weight - 2
        self._levels = np.zeros((4, num_levels), dtype=np.int64)
        self._chosen_row_sums = np.zeros((num_levels, num_words), dtype=np.uint64)
        self._basis = np.zeros((num_levels, num_words - 1), dtype=np.uint64)
        self._pivots = np.zeros(num_levels, dtype=np.int64)
        self._unavailable = np.zeros(num_classes, dtype=np.bool_)
        self._marks = np.empty(num_classes, dtype=np.int64)
        self._bounds = np.zeros((2, fault_weight), dtype=np.int64)
        self._walk_state = np.zeros(10, dtype=np.int64)
        self._codewords = np.empty((_MAX_ROOT_CODEWORDS, fault_weight), dtype=np.int64)

    @property
    def work(self) -> int:
        """The work of the walk for all of its roots, in probes."""
        return int(self._walk_state[4])

    def list_range(
        self, start: int, end: int, probes_per_call: int = 1 << 62
    ) -> Iterator[npt.NDArray[np.int64]]:
        """Yield the class codewords of the roots from ``start`` to ``end - 1``, in batches.

        The class codewords of each root come in lexicographic order.  Each batch is a view of an
        array that a later batch can overwrite.  A call of ``_list_by_shared_flags`` returns once
        it has done ``probes_per_call`` work, so a batch can be empty.  Each range must be listed to
        its end before the next starts, since the walk then leaves no class unavailable.
        """
        # Start before the first pass over the root ``start``, with no rows held and no floor or
        # ceiling.
        walk_state = self._walk_state
        walk_state[0], walk_state[1], walk_state[2] = start, end, -1
        walk_state[6:] = 0
        bit_offsets, bit_classes = self._bit_index
        map_keys, map_classes, map_slots = self._class_map
        while walk_state[0] < end:
            walk_state[5] = min(int(walk_state[4]) + probes_per_call, 1 << 62)
            num_codewords = _list_by_shared_flags(
                self._class_rows,
                bit_offsets,
                bit_classes,
                map_keys,
                map_classes,
                map_slots,
                _PROBES_PER_REDUCTION,
                _CODEWORD_BATCH_SIZE,
                self._levels,
                self._chosen_row_sums,
                self._basis,
                self._pivots,
                self._unavailable,
                self._marks,
                self._bounds,
                walk_state,
                self._codewords,
            )
            yield self._codewords[:num_codewords]


def _get_root_weights(
    num_classes: int, num_roots: int, fault_weight: int
) -> npt.NDArray[np.float64]:
    """Weigh the roots of a listing of class codewords by an estimate of the work of each.

    The weights decide whether a listing moves to threads and how its roots are split between them,
    never what is listed.  A class codeword of fault weight ``m`` whose root is ``r`` has its other
    ``m - 1`` classes after ``r``, and the walk chooses or tries all of them but one, which it
    finds by a probe.  So each root is weighed by ``C(q - 1 - r, m - 2)``, for ``q`` classes, the
    number of these choices.  The candidates of every state of the walk by shared flags come after
    ``r``, so the work of a root falls with ``r`` about as fast as this weight does if the classes
    that share each flag bit are spread over the order of the classes.  The weights are divided by
    ``C(q - 1, m - 2)``, so that each is the product of the factors
    ``(q - 1 - r - j) / (q - 1 - j)`` for ``j < m - 2``, each from 0 to 1, and never overflows.  A
    weight is zero if ``q - 1 - r < m - 2``.  ``fault_propagation.md#sec-listing-on-threads``
    explains how the weights split a listing.
    """
    remaining = num_classes - 1 - np.arange(num_roots, dtype=np.float64)
    offsets = np.arange(fault_weight - 2, dtype=np.float64)
    numerators = np.maximum(remaining[:, None] - offsets, 0.0)
    denominators = np.maximum(num_classes - 1 - offsets, 1.0)
    return np.prod(numerators / denominators, axis=1)


@always_compiled
def _get_bit_index(
    flag_words: npt.NDArray[np.uint64],
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Index classes by their flag bits, for the walk by shared flags.

    ``flag_words`` holds the flags of each class.  Return offsets and classes such that the classes
    whose flags have bit ``b`` are ``classes[offsets[b]:offsets[b + 1]]``, in increasing order.
    """
    num_classes, num_words = flag_words.shape
    offsets = np.zeros(64 * num_words + 1, dtype=np.int64)
    for each in range(num_classes):
        for word in range(num_words):
            remaining = flag_words[each, word]
            while remaining:
                offsets[64 * word + trailing_zeros_word(remaining) + 1] += 1
                remaining &= remaining - np.uint64(1)
    for entry in range(64 * num_words):
        offsets[entry + 1] += offsets[entry]
    classes = np.empty(offsets[-1], dtype=np.int64)
    next_entries = offsets[:-1].copy()
    for each in range(num_classes):
        for word in range(num_words):
            remaining = flag_words[each, word]
            while remaining:
                bit = 64 * word + trailing_zeros_word(remaining)
                classes[next_entries[bit]] = each
                next_entries[bit] += 1
                remaining &= remaining - np.uint64(1)
    return offsets, classes


def _get_class_map(
    hashes: npt.NDArray[np.uint64],
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Build the map of classes by hash (see ``FaultLister``).

    ``hashes`` holds the hash of the flags of each class.  Return the hashes in increasing order,
    the classes in the same order, those with each hash in increasing order, and the slots of a hash
    map from each hash to its first entry (see ``_get_key_slots``).
    """
    classes = np.argsort(hashes, kind="stable")
    keys = hashes[classes]
    return keys, classes, _get_key_slots(keys)


def _get_empty_syndrome_table(
    num_words: int,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """An empty table of syndromes of this many words (see ``_find_syndrome``)."""
    size = _INITIAL_SYNDROME_TABLE_SIZE
    return np.empty((size, 1 + num_words), dtype=np.uint64), np.full(2 * size, -1, dtype=np.int64)


def _hash_words(words: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Hash each row of words to 64 bits with a fixed random linear map.

    The hash of a sum of rows is the sum of their hashes.  Rows that differ can share a hash.
    """
    return _hash_bytes(np.ascontiguousarray(words), _get_byte_hash_tables(words.shape[1]))


# Each certification hashes rows of a few widths, and building their tables can take longer than
# hashing the rows of a small circuit, so keep the tables of the widths used last.
@functools.lru_cache(maxsize=8)
def _get_byte_hash_tables(num_words: int) -> npt.NDArray[np.uint64]:
    """The read-only tables of ``_hash_words`` for rows of this many words (see ``_hash_bytes``)."""
    # Bit ``b`` of word ``c`` adds ``images[c, b]`` to the hash.  Tabulate the hash of each value of
    # each byte of each word, so that a row takes one lookup per byte rather than steps per bit.
    images = np.random.default_rng(0).integers(2**64, size=(num_words, 64), dtype=np.uint64)
    byte_images = images.reshape(num_words, 8, 8)
    byte_values = np.arange(256)
    tables = np.zeros((num_words, 8, 256), dtype=np.uint64)
    for bit in range(8):
        has_bit = ((byte_values >> bit) & 1).astype(np.uint64)
        tables ^= byte_images[:, :, bit, None] * has_bit
    tables.flags.writeable = False
    return tables


def _get_num_words_for_values(values: Collection[int]) -> int:
    """The number of 64-bit words that hold each of these integers, at least one."""
    return max(1, get_num_words(max((value.bit_length() for value in values), default=0)))


def _to_words(values: Sequence[int], num_words: int) -> npt.NDArray[np.uint64]:
    """Split each integer into words, with word ``j`` holding bits ``64 j`` and up."""
    data = b"".join(value.to_bytes(8 * num_words, "little") for value in values)
    words = np.frombuffer(data, dtype="<u8").reshape(len(values), num_words)
    return words.astype(np.uint64)


def _hashed_rows(values: Sequence[int], num_words: int) -> npt.NDArray[np.uint64]:
    """For each integer, a row of its linear hash and then its words, as in a table of syndromes."""
    words = _to_words(values, num_words)
    return np.hstack([_hash_words(words)[:, None], words])


@always_compiled
def _hash_bytes(
    words: npt.NDArray[np.uint64], tables: npt.NDArray[np.uint64]
) -> npt.NDArray[np.uint64]:
    """Hash each row of words with tables of the hashes of bytes.

    A row hashes to the XOR, over each byte ``k`` of each word ``c``, of ``tables[c, k, v]`` for the
    value ``v`` of that byte.
    """
    hashes = np.zeros(len(words), dtype=np.uint64)
    byte_mask = np.uint64(0xFF)
    for row in range(len(words)):
        hashed = np.uint64(0)
        for column in range(words.shape[1]):
            word = words[row, column]
            for byte in range(8):
                value = np.int64((word >> np.uint64(8 * byte)) & byte_mask)
                hashed ^= tables[column, byte, value]
        hashes[row] = hashed
    return hashes


####################################################################################################
# compiled listing


@always_compiled(nogil=True)
def _list_by_shared_flags(
    class_rows: npt.NDArray[np.uint64],
    bit_offsets: npt.NDArray[np.int64],
    bit_classes: npt.NDArray[np.int64],
    map_keys: npt.NDArray[np.uint64],
    map_classes: npt.NDArray[np.int64],
    map_slots: npt.NDArray[np.int64],
    reduction_cost: int,
    batch_size: int,
    levels: npt.NDArray[np.int64],
    chosen_row_sums: npt.NDArray[np.uint64],
    basis: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    unavailable: npt.NDArray[np.bool_],
    marks: npt.NDArray[np.int64],
    bounds: npt.NDArray[np.int64],
    walk_state: npt.NDArray[np.int64],
    codewords: npt.NDArray[np.int64],
) -> int:
    """List class codewords of roots by shared flags into an array, and return how many are ready.

    ``class_rows`` holds the row of each class (see ``FaultLister``), ``bit_offsets`` and
    ``bit_classes`` index the classes by their flag bits (see ``_get_bit_index``), and ``map_keys``,
    ``map_classes``, and ``map_slots`` are the map of classes by hash from ``_get_class_map``.  The
    classes must have nonzero and pairwise distinct flags, as those of ``FaultLister`` do.  For each
    root in turn, the listing lists the class codewords of ``m = codewords.shape[1]`` classes whose
    first class is the root, as ``fault_propagation.md#sec-shared-flags`` describes.  Each class
    codeword is a row of classes in increasing order.  The listing makes one or more passes over
    each root.  The array, of at least two rows, holds the class codewords that the listing keeps in
    the pass, which it sorts in lexicographic order once the pass is done, after any rows of earlier
    passes that it has not yet returned.  If the kept class codewords fill the array when the
    listing examines another class at the last depth, it keeps the first ``⌊D/2⌋`` of them in
    lexicographic order, for ``D`` rows, and makes another pass for the others.  The listing counts
    its work in probes, as ``fault_propagation.md#sec-listing-on-threads`` describes, with
    ``reduction_cost`` probes for each reduction of flags.

    The listing keeps its position between calls in the other arrays.  The depth is the index, among
    the chosen classes, of the last one chosen, and the chosen classes are the state ``A`` of the
    guide.  Column ``d`` of ``levels`` holds the class chosen at depth ``d``, the entry of
    ``bit_classes`` of its next candidate and the entry after its last, and the number of marks when
    it was chosen.  Row ``d`` of ``chosen_row_sums`` is the sum of the rows of the classes chosen up
    to depth ``d``, and row ``d`` of ``basis`` and entry ``d`` of ``pivots`` hold the reduced flags
    of the class chosen at depth ``d``, as in ``_reduce_flags``.  ``marks`` lists, in the order in
    which they were forbidden, the classes of ``Φ(A)`` and, if ``|A| < m - 2``, the candidates of
    ``A`` that the listing has tried, which are forbidden in the states of later candidates.
    ``unavailable`` is true for the classes of ``marks`` and of ``A``.  ``bounds`` holds the floor,
    the last class codeword released by an earlier pass over the root, and the ceiling, the least
    class codeword that this pass has dropped.  ``walk_state`` holds (0) the root, (1) the end of
    the roots, (2) the depth, -1 between passes, (3) the number of marks, (4) the work so far, (5)
    the work at which to pause, (6) the number of rows that the last call returned, (7) the number
    of rows of the unfinished pass, which follow them, (8) 1 if an earlier pass released class
    codewords of the root, so that the floor holds, and (9) 1 if this pass has dropped class
    codewords, so that the ceiling holds.  The state starts with entry 0 at the first root, entry 1
    at the end of the roots, entry 2 at -1, ``unavailable`` false, and the other entries of
    ``walk_state`` zero, apart from entries 4 and 5.

    The listing returns once no root before the end is left, once it has listed at least
    ``batch_size`` class codewords and finished a pass, once its work reaches the work at which to
    pause, or once the array is full and holds rows of earlier passes.  It returns the rows of the
    passes that it has finished, and its next call moves the rows of an unfinished pass to the
    start of the array.  It writes only to ``levels``, ``chosen_row_sums``, ``basis``, ``pivots``,
    ``unavailable``, ``marks``, ``bounds``, ``walk_state``, and ``codewords``, and only reads the
    other arrays, so that calls with their own such arrays can run at once without the global
    interpreter lock, as ``fault_propagation.md#sec-thread-requirements`` requires.
    """
    # Column 0 of the class rows holds the hash of the flags, and the other columns the flags.  The
    # keys of the map of classes serve as rows of one column, a hash and no syndrome words, so that
    # ``_find_syndrome_slot`` compares the keys alone (see ``_get_key_slots``).  Make these arrays
    # once, rather than for each probe, which would slow the listing.
    num_words = class_rows.shape[1]
    fault_weight = codewords.shape[1]
    last_depth = fault_weight - 3
    reduced = np.empty(num_words - 1, dtype=np.uint64)
    row = np.empty(fault_weight, dtype=np.int64)
    key_rows = map_keys.reshape((len(map_keys), 1))
    no_words = map_keys[:0]

    # Move the rows of an unfinished pass to the start of the array.
    root, end_root, depth, num_marks = walk_state[0], walk_state[1], walk_state[2], walk_state[3]
    work, max_work, num_pending = walk_state[4], walk_state[5], walk_state[7]
    has_floor, has_ceiling = walk_state[8], walk_state[9]
    for index in range(num_pending):
        codewords[index] = codewords[walk_state[6] + index]
    num_ready = 0
    while work < max_work:
        # Start the next pass, if a root is left and the batch is not done, by choosing the root
        # and the bit whose candidates come next.
        if depth < 0:
            if root >= end_root or num_ready >= batch_size:
                break
            depth, num_marks = 0, 0
            unavailable[root] = True
            levels[0, 0], levels[3, 0] = root, 0
            levels[1, 0], levels[2, 0], num_bits = _choose_bit(
                class_rows[root, 1:], root, bit_offsets, bit_classes
            )
            work += reduction_cost + num_bits
            chosen_row_sums[0] = class_rows[root]
            pivots[0] = _reduce_flags(class_rows[root, 1:], basis[:0], pivots, basis[0])
            continue

        # Once every candidate after a chosen class is tried, allow again the classes forbidden
        # since it was chosen, and forbid it in the candidates after it, or, for the root, allow it
        # again for the next pass.  Once the pass over the root is done, sort its class codewords,
        # and make another pass for those after the last of them if the pass dropped any.
        if levels[1, depth] == levels[2, depth]:
            for index in range(levels[3, depth], num_marks):
                unavailable[marks[index]] = False
            num_marks = levels[3, depth]
            chosen_class = levels[0, depth]
            unavailable[chosen_class] = depth > 0
            if depth > 0:
                marks[num_marks] = chosen_class
                num_marks += 1
            depth -= 1
            if depth < 0:
                _sort_rows(codewords[num_ready : num_ready + num_pending])
                num_ready += num_pending
                num_pending = 0
                has_floor = has_ceiling
                if has_ceiling:
                    bounds[0] = codewords[num_ready - 1]
                else:
                    root += 1
                has_ceiling = 0
            continue

        # At the last depth, a class examined adds at most one class codeword.  If the array has no
        # room for it, return the rows of earlier passes, or else keep the first half, rounded down,
        # of the rows of this pass in lexicographic order, and from now on keep only class codewords
        # before the first row dropped.
        if depth == last_depth and num_ready + num_pending == len(codewords):
            if num_ready:
                break
            _sort_rows(codewords[:num_pending])
            num_pending //= 2
            bounds[1] = codewords[num_pending]
            has_ceiling = 1
            continue

        # Take the next candidate, unless it is chosen or forbidden.
        candidate = bit_classes[levels[1, depth]]
        levels[1, depth] += 1
        work += 1
        if unavailable[candidate]:
            continue

        # With all classes but one of a class codeword chosen, probe the map of classes with the
        # hash of the summed flags, and check every class with this hash, comparing its flags in
        # full, since different flags can share a hash.  A class with exactly these flags that comes
        # after the root and is neither chosen nor forbidden completes a class codeword if the
        # candidate keeps the chosen classes independent.  Keep the class codeword if it comes after
        # the floor, if the pass has one, and before the ceiling, if it has one.  The classes have
        # distinct flags, so at most one class completes a class codeword.
        if depth == last_depth:
            key = chosen_row_sums[depth, 0] ^ class_rows[candidate, 0]
            entry = map_slots[_find_syndrome_slot(key, no_words, key_rows, map_slots)]
            while entry >= 0 and entry < len(map_keys) and map_keys[entry] == key:
                closing_class = map_classes[entry]
                entry += 1
                closes = closing_class > root and not unavailable[closing_class]
                for word in range(1, num_words):
                    summed = chosen_row_sums[depth, word] ^ class_rows[candidate, word]
                    closes = closes and summed == class_rows[closing_class, word]
                if not closes:
                    continue
                work += reduction_cost
                pivot = _reduce_flags(
                    class_rows[candidate, 1:], basis[: depth + 1], pivots, reduced
                )
                if pivot < 0:
                    continue
                for level in range(depth + 1):
                    row[level] = levels[0, level]
                row[depth + 1] = candidate
                row[depth + 2] = closing_class
                row.sort()
                if (not has_floor or _comes_before(bounds[0], row)) and (
                    not has_ceiling or _comes_before(row, bounds[1])
                ):
                    codewords[num_ready + num_pending] = row
                    num_pending += 1
            continue

        # Otherwise choose the candidate, and the bit whose candidates come next, if it keeps the
        # chosen classes independent, and else forbid it in the candidates after it.
        work += reduction_cost
        pivot = _reduce_flags(class_rows[candidate, 1:], basis[: depth + 1], pivots, reduced)
        if pivot < 0:
            unavailable[candidate] = True
            marks[num_marks] = candidate
            num_marks += 1
            continue
        depth += 1
        unavailable[candidate] = True
        levels[0, depth], levels[3, depth] = candidate, num_marks
        basis[depth] = reduced
        pivots[depth] = pivot
        for word in range(num_words):
            chosen_row_sums[depth, word] = (
                chosen_row_sums[depth - 1, word] ^ class_rows[candidate, word]
            )
        levels[1, depth], levels[2, depth], num_bits = _choose_bit(
            chosen_row_sums[depth, 1:], root, bit_offsets, bit_classes
        )
        work += num_bits

    walk_state[0], walk_state[2], walk_state[3], walk_state[4] = root, depth, num_marks, work
    walk_state[6], walk_state[7], walk_state[8], walk_state[9] = (
        num_ready,
        num_pending,
        has_floor,
        has_ceiling,
    )
    return num_ready


@always_compiled
def _choose_bit(
    flags: npt.NDArray[np.uint64],
    root: int,
    bit_offsets: npt.NDArray[np.int64],
    bit_classes: npt.NDArray[np.int64],
) -> tuple[int, int, int]:
    """Choose the bit of nonzero flags with the fewest classes after the root, the lowest of these.

    Return the entry of ``bit_classes`` of its first class after the root, the entry after its last
    class (see ``_get_bit_index``), and the number of bits of the flags.
    """
    best_start, best_end, num_bits = 0, -1, 0
    for word in range(len(flags)):
        remaining = flags[word]
        while remaining:
            bit = 64 * word + trailing_zeros_word(remaining)
            remaining &= remaining - np.uint64(1)
            num_bits += 1
            start, end = bit_offsets[bit], bit_offsets[bit + 1]
            start += np.searchsorted(bit_classes[start:end], root, side="right")
            if best_end < 0 or end - start < best_end - best_start:
                best_start, best_end = start, end
    return best_start, best_end, num_bits


@always_compiled(inline=True)
def _comes_before(first: npt.NDArray[np.int64], second: npt.NDArray[np.int64]) -> bool:
    """Whether a row of classes comes before another of as many in lexicographic order."""
    for index in range(len(first)):
        if first[index] != second[index]:
            return bool(first[index] < second[index])
    return False


@always_compiled
def _sort_rows(rows: npt.NDArray[np.int64]) -> None:
    """Sort rows in lexicographic order, by a stable sort by each column from the last."""
    for column in range(rows.shape[1] - 1, -1, -1):
        order = np.argsort(rows[:, column].copy(), kind="mergesort")
        rows[:] = rows[order]


@always_compiled(nogil=True)
def _expand_class_tuples(
    class_tuples: npt.NDArray[np.int64],
    class_offsets: npt.NDArray[np.int64],
    class_members: npt.NDArray[np.int64],
    num_start_classes: int,
    num_starts: int,
    primitive_rows: npt.NDArray[np.uint64],
    css_weight: bool,
    batch_size: int,
    batch_members: npt.NDArray[np.int64],
    batch_syndromes: npt.NDArray[np.uint64],
    syndrome_rows: npt.NDArray[np.uint64],
    syndrome_slots: npt.NDArray[np.int64],
    max_syndromes: int,
    next_member_entries: npt.NDArray[np.int64],
    chosen_row_sums: npt.NDArray[np.uint64],
    chosen_start_counts: npt.NDArray[np.int64],
    choice_rows: npt.NDArray[np.uint64],
    choice_slots: npt.NDArray[np.int64],
    choice_slots_used: npt.NDArray[np.int64],
    expansion_state: npt.NDArray[np.int64],
) -> None:
    """Expand class tuples into faults, and add each fault to a batch unless it is left out.

    A fault of a class tuple (see ``FaultLister``) takes one member of each of its classes, with
    increasing members where a class repeats, and must contain a start member, one of the first
    ``num_starts`` primitive faults.  The start members come first in their classes, and a class
    tuple lists its start classes, the first ``num_start_classes`` classes, before its other
    classes.  The members of class ``c`` are
    ``class_members[class_offsets[c]:class_offsets[c + 1]]``, and ``primitive_rows`` holds the row
    of each primitive fault (see ``FaultLister``).  The first ``expansion_state[3]`` rows of
    ``syndrome_rows``, with ``syndrome_slots``, form the table of syndromes (see
    ``_find_syndrome``).  Each fault that is not left out (see ``FaultLister.find_first``) gets the
    next row of the batch arrays: its primitive faults, in the order of its classes, and its
    syndrome.  The expansion skips, without looking them up, the faults that the table would leave
    out because an earlier choice of members had the same syndrome.

    The level is the index, in a class tuple, of the class whose member is being chosen.  The
    expansion keeps its state between calls in ``expansion_state`` (the index of the class tuple to
    expand, the level or -1 before the class tuple starts, the number of faults in the batch, the
    number of syndromes in the table, the number of remembered choices, and the first level whose
    class has more than one member), ``next_member_entries`` (at each level, the entry of
    ``class_members`` after that of the chosen member), ``chosen_row_sums`` (row ``i`` is the sum
    of the primitive rows of the members chosen at the first ``i`` levels), ``chosen_start_counts``
    (entry ``i`` is the number of start members among them), and ``choice_rows``,
    ``choice_slots``, and ``choice_slots_used``.  These form a hash map as in
    ``_find_syndrome_slot`` of the choices of members of the current class tuple whose completions
    were listed: each row holds a hash, then the level of the choice and whether it has a start
    member, then its syndrome.  ``choice_slots_used`` lists the slots in use, to clear them for the
    next class tuple.  The expansion returns once the batch has ``batch_size`` faults, once the
    table should grow (see ``_table_should_grow``), or once every class tuple is expanded.  A full
    table takes no more syndromes.  It runs without the global interpreter lock, writes only to
    the arrays of the batch, of the table of syndromes, and of its state, and only reads
    ``class_tuples``, ``class_offsets``, ``class_members``, and ``primitive_rows``, as
    ``fault_propagation.md#sec-thread-requirements`` requires.
    """
    # Column 0 of the primitive rows holds the hash of the syndrome, the next columns hold the
    # syndrome, and the rest hold the X and then the Z part of the representative.
    fault_weight = class_tuples.shape[1]
    num_words = primitive_rows.shape[1]
    x_start = 1 + batch_syndromes.shape[1]
    z_start = x_start + (num_words - x_start) // 2

    # The key of a choice of members in the map of remembered choices: its level and whether it has
    # a start member, then its syndrome.
    choice_key = np.empty((1, choice_rows.shape[1]), dtype=np.uint64)

    tuple_index, level = expansion_state[0], expansion_state[1]
    num_batched, num_syndromes = expansion_state[2], expansion_state[3]
    num_choices, first_branching_level = expansion_state[4], expansion_state[5]
    while tuple_index < len(class_tuples):
        # Start a class tuple with no remembered choices.  Up to the first level whose class has
        # more than one member, no two choices have the same syndrome, so none is remembered.
        # Before that level, there is only one choice at each level.  At that level, the choices
        # differ only in one member of one class.  The members of a class have the same flags, and
        # the search passes primitive faults of distinct signatures, so they have distinct
        # syndromes.  Primitive faults of repeated signatures would only miss some skips.
        if level < 0:
            level = 0
            next_member_entries[0] = class_offsets[class_tuples[tuple_index, 0]]
            for choice in range(num_choices):
                choice_slots[choice_slots_used[choice]] = -1
            num_choices = 0
            first_branching_level = 0
            while first_branching_level < fault_weight:
                branching_class = class_tuples[tuple_index, first_branching_level]
                if class_offsets[branching_class + 1] - class_offsets[branching_class] > 1:
                    break
                first_branching_level += 1
        while level >= 0:
            # Choose the next member of the class at this level, or go back a level if there is
            # none.
            current_class = class_tuples[tuple_index, level]
            member_entry = next_member_entries[level]
            if member_entry == class_offsets[current_class + 1]:
                level -= 1
                continue
            next_member_entries[level] = member_entry + 1
            member = class_members[member_entry]
            is_last = level + 1 == fault_weight

            # Skip this member and the rest of its class if no member chosen so far, this one
            # included, is a start member, and no member chosen later can be one: at the last level,
            # when the next level repeats this class, or when the next class is not a start class
            # (see ``fault_propagation.md#sec-listing-once``).
            num_chosen_starts = chosen_start_counts[level] + (member < num_starts)
            if num_chosen_starts == 0 and (
                is_last
                or class_tuples[tuple_index, level + 1] == current_class
                or class_tuples[tuple_index, level + 1] >= num_start_classes
            ):
                next_member_entries[level] = class_offsets[current_class + 1]
                continue

            # Add the member to the sums, and move on to the next level, if any.  A class repeats
            # only for fault weight two, and then its second member comes after its first.
            chosen_start_counts[level + 1] = num_chosen_starts
            for word in range(num_words):
                chosen_row_sums[level + 1, word] = (
                    chosen_row_sums[level, word] ^ primitive_rows[member, word]
                )
            if not is_last:
                # Skip every fault that completes this choice if an earlier choice of members of the
                # same classes of this class tuple had the same syndrome, and a start member if and
                # only if this one has, since the table then holds the syndromes of all these faults
                # (see ``fault_propagation.md#sec-skipping-listed-syndromes``).  This needs a table
                # that has had room for every syndrome so far in this call of
                # ``FaultLister.find_first``, as that section says.  A table that is not full now
                # has had that room.  A table grows when half full whenever it can, so a full table
                # never grows.  A table is cleared only when a violation ends the listing.  This
                # applies only past the first level whose class has more than one member (see
                # above).  Otherwise remember the choice, if there is room.
                if num_syndromes < len(syndrome_rows) and level > first_branching_level:
                    choice_key[0, 1] = np.uint64(2 * level + (1 if num_chosen_starts > 0 else 0))
                    choice_key[0, 0] = chosen_row_sums[level + 1, 0] ^ (
                        (choice_key[0, 1] + np.uint64(1)) * _HASH_MULTIPLIER
                    )
                    for word in range(1, x_start):
                        choice_key[0, 1 + word] = chosen_row_sums[level + 1, word]
                    slot = _find_row(choice_key, 0, choice_rows, choice_slots)
                    if slot < 0:
                        continue
                    if num_choices < len(choice_rows):
                        for word in range(choice_rows.shape[1]):
                            choice_rows[num_choices, word] = choice_key[0, word]
                        choice_slots[slot] = num_choices
                        choice_slots_used[num_choices] = slot
                        num_choices += 1
                level += 1
                next_class = class_tuples[tuple_index, level]
                next_member_entries[level] = (
                    member_entry + 1 if next_class == current_class else class_offsets[next_class]
                )
                continue

            # Leave out the fault if its syndrome is in the table, and otherwise add the syndrome to
            # the table if there is room, even if the representative test below leaves the fault out
            # (see ``fault_propagation.md#sec-checking-one-fault``).
            slot = _find_row(chosen_row_sums, fault_weight, syndrome_rows, syndrome_slots)
            if slot < 0:
                continue
            fault_row = chosen_row_sums[fault_weight]
            if num_syndromes < len(syndrome_rows):
                syndrome_rows[num_syndromes] = fault_row[:x_start]
                syndrome_slots[slot] = num_syndromes
                num_syndromes += 1

            # Leave out the fault if the product of the representatives has Pauli weight at most the
            # fault weight (see ``fault_propagation.md#sec-checking-one-fault``), and otherwise add
            # it to the batch.
            weight = _pauli_weight(fault_row[x_start:z_start], fault_row[z_start:], css_weight)
            if weight > fault_weight:
                batch_syndromes[num_batched] = fault_row[1:x_start]
                for index in range(fault_weight):
                    batch_members[num_batched, index] = class_members[
                        next_member_entries[index] - 1
                    ]
                num_batched += 1

            # Pause when the batch is full or the table should grow.
            if num_batched == batch_size or _table_should_grow(
                num_syndromes, len(syndrome_rows), max_syndromes
            ):
                expansion_state[0], expansion_state[1] = tuple_index, level
                expansion_state[2], expansion_state[3] = num_batched, num_syndromes
                expansion_state[4], expansion_state[5] = num_choices, first_branching_level
                return
        tuple_index += 1
        level = -1
    expansion_state[0], expansion_state[1] = tuple_index, level
    expansion_state[2], expansion_state[3] = num_batched, num_syndromes
    expansion_state[4], expansion_state[5] = num_choices, first_branching_level


@always_compiled
def _reduce_flags(
    flags: npt.NDArray[np.uint64],
    basis: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    reduced: npt.NDArray[np.uint64],
) -> int:
    """Reduce flags by a basis, and return the lowest set bit of the result, or -1 if it is zero.

    Row ``r`` of the basis has bit ``pivots[r]`` set, which every later row has clear.  Adding the
    rows in order clears these bits, so the result is zero exactly when the flags are a sum of rows.
    """
    reduced[:] = flags
    for row in range(len(basis)):
        pivot = pivots[row]
        if reduced[pivot // 64] >> np.uint64(pivot % 64) & np.uint64(1):
            for word in range(len(reduced)):
                reduced[word] ^= basis[row, word]
    for word in range(len(reduced)):
        if reduced[word]:
            return int(64 * word + trailing_zeros_word(reduced[word]))
    return -1


@always_compiled
def _pauli_weight(
    x: npt.NDArray[np.uint64], z: npt.NDArray[np.uint64], css_weight: bool
) -> np.int64:
    """The Pauli weight of the error with these packed X and Z parts."""
    x_weight = z_weight = support = np.int64(0)
    for word in range(len(x)):
        x_weight += popcount_word(x[word])
        z_weight += popcount_word(z[word])
        support += popcount_word(x[word] | z[word])
    return max(x_weight, z_weight) if css_weight else support


@always_compiled
def _find_syndrome(
    hashed: np.uint64,
    syndrome: npt.NDArray[np.uint64],
    syndrome_rows: npt.NDArray[np.uint64],
    syndrome_slots: npt.NDArray[np.int64],
) -> int:
    """Look up a syndrome with this hash in a table of syndromes.

    A table of syndromes is a hash map as in ``_find_syndrome_slot``, with twice as many slots as
    ``syndrome_rows`` has rows, so that the rows fill at most half of the slots.  Return -1 if the
    syndrome is in the table, and otherwise the empty slot for it.
    """
    slot = _find_syndrome_slot(hashed, syndrome, syndrome_rows, syndrome_slots)
    return -1 if syndrome_slots[slot] >= 0 else slot


@always_compiled
def _find_syndrome_slot(
    hashed: np.uint64,
    syndrome: npt.NDArray[np.uint64],
    syndrome_rows: npt.NDArray[np.uint64],
    syndrome_slots: npt.NDArray[np.int64],
) -> int:
    """The slot that holds a syndrome in a hash map of syndromes, or else the empty slot for it.

    Each slot holds the index of a row of ``syndrome_rows``, or -1 if it is empty, and each row
    that a slot refers to holds the hash of a syndrome and then its words.  A lookup starts at the
    slot that the hash picks, and steps forward, wrapping around, to the first slot that is empty or
    refers to the syndrome.  The number of slots must be a power of two, at most 2**32, and at least
    one slot must be empty, so that the lookup ends.
    """
    slot = _get_first_slot(hashed, len(syndrome_slots))
    while syndrome_slots[slot] >= 0:
        row = syndrome_slots[slot]
        equal = syndrome_rows[row, 0] == hashed
        for word in range(len(syndrome)):
            equal = equal and syndrome_rows[row, 1 + word] == syndrome[word]
        if equal:
            break
        slot = (slot + 1) % len(syndrome_slots)
    return int(slot)


@always_compiled(inline=True)
def _find_row(
    keys: npt.NDArray[np.uint64],
    key: int,
    rows: npt.NDArray[np.uint64],
    slots: npt.NDArray[np.int64],
) -> int:
    """Look up row ``key`` of ``keys`` in a hash map as in ``_find_syndrome_slot``.

    The key is the first ``rows.shape[1]`` columns of the row, the first of which holds the hash.
    Return -1 if the key is in the map, and otherwise the empty slot for it.  Unlike
    ``_find_syndrome``, this takes no views of arrays, since numba compiles a view with reference
    counting, which would make each lookup of ``_expand_class_tuples`` several times slower.
    """
    mask = len(slots) - 1
    slot = _get_first_slot(keys[key, 0], len(slots))
    while True:
        row = slots[slot]
        if row < 0:
            return int(slot)
        equal = True
        for column in range(rows.shape[1]):
            if rows[row, column] != keys[key, column]:
                equal = False
                break
        if equal:
            return -1
        slot = (slot + 1) & mask


@always_compiled(inline=True)
def _get_first_slot(hashed: np.uint64, num_slots: int) -> np.int64:
    """The slot at which a lookup of a hash starts, in a hash map of ``num_slots`` slots.

    The number of slots must be a power of two, at most 2**32.
    """
    # The product of the hash's low 32 bits and ``_HASH_MULTIPLIER`` fits in 64 bits, and the slot
    # is the top bits of its low 32 bits.
    low_bits = np.uint64(0xFFFFFFFF)
    shift = np.uint64(32 - trailing_zeros_word(np.uint64(num_slots)))
    return np.int64((((hashed & low_bits) * _HASH_MULTIPLIER) & low_bits) >> shift)


@always_compiled
def _get_key_slots(keys: npt.NDArray[np.uint64]) -> npt.NDArray[np.int64]:
    """The slots of a hash map from each distinct key of sorted keys to its first entry.

    The slots form a hash map as in ``_find_syndrome_slot`` whose rows are the keys, each a row of
    one column that holds a hash and no syndrome words, so that a lookup compares the keys alone.
    The first entry of a key is
    ``slots[_find_syndrome_slot(key, keys[:0], keys.reshape((len(keys), 1)), slots)]``, or -1 if no
    entry has the key.  The number of slots is the smallest power of two that is at least twice the
    number of distinct keys, and at least two, so that at least one slot is empty.
    """
    num_keys = 0
    for entry in range(len(keys)):
        if entry == 0 or keys[entry] != keys[entry - 1]:
            num_keys += 1
    num_slots = 2
    while num_slots < 2 * num_keys:
        num_slots *= 2
    slots = np.full(num_slots, -1, dtype=np.int64)
    rows = keys.reshape((len(keys), 1))
    for entry in range(len(keys)):
        if entry == 0 or keys[entry] != keys[entry - 1]:
            slots[_find_syndrome_slot(keys[entry], keys[:0], rows, slots)] = entry
    return slots


@always_compiled
def _get_light_pairs(
    heavy_rows: npt.NDArray[np.uint64], light_rows: npt.NDArray[np.uint64]
) -> npt.NDArray[np.int64]:
    """For each heavy syndrome, two light syndromes that sum to it, if there are any.

    Each row holds the linear hash of a syndrome and then its words, and the light syndromes are
    distinct.  Row ``h`` of the result holds the first light syndrome ``i``, in order, such that the
    sum of heavy syndrome ``h`` and light syndrome ``i`` is a light syndrome ``j``, and then ``j``,
    or -1 twice if there is none.
    """
    # Put the light syndromes in a hash map as in ``_find_syndrome_slot``, with at least eight times
    # as many slots, so that most lookups of a sum that is not light end at once at an empty slot.
    # This makes the search about five times faster than with half-full slots.
    num_slots = 2
    while num_slots < 8 * len(light_rows):
        num_slots *= 2
    slots = np.full(num_slots, -1, dtype=np.int64)
    for row in range(len(light_rows)):
        slots[_find_syndrome_slot(light_rows[row, 0], light_rows[row, 1:], light_rows, slots)] = row

    # Most sums are not light, so look each sum up by its hash first, as ``_find_syndrome_slot``
    # does, and compare its words only with those of a light syndrome with the same hash.  The
    # hashes are copied out of the rows, which are long for many qubits, so that the loop reads them
    # in order.
    light_hashes = light_rows[:, 0].copy()
    num_words = heavy_rows.shape[1] - 1
    pairs = np.full((len(heavy_rows), 2), -1, dtype=np.int64)
    for heavy in range(len(heavy_rows)):
        heavy_hash = heavy_rows[heavy, 0]
        for first in range(len(light_rows)):
            hashed = heavy_hash ^ light_hashes[first]
            slot = _get_first_slot(hashed, num_slots)
            second = slots[slot]
            while second >= 0:
                if light_hashes[second] == hashed:
                    equal = True
                    for word in range(1, num_words + 1):
                        sum_word = heavy_rows[heavy, word] ^ light_rows[first, word]
                        if sum_word != light_rows[second, word]:
                            equal = False
                            break
                    if equal:
                        break
                slot = (slot + 1) & (num_slots - 1)
                second = slots[slot]
            if second >= 0:
                pairs[heavy, 0] = first
                pairs[heavy, 1] = second
                break
    return pairs


@always_compiled(inline=True)
def _table_should_grow(num_syndromes: int, num_rows: int, max_syndromes: int) -> bool:
    """Whether a table of syndromes should double in size.

    It should if it is half full, and if the old and the new table together have at most
    ``max_syndromes`` rows.
    """
    return 2 * num_syndromes >= num_rows and 3 * num_rows <= max_syndromes


@always_compiled
def _grow_syndrome_table(
    syndrome_rows: npt.NDArray[np.uint64], num_syndromes: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
    """Copy a table of ``num_syndromes`` syndromes (see ``_find_syndrome``) into a larger one."""
    larger_rows = np.empty((2 * len(syndrome_rows), syndrome_rows.shape[1]), dtype=np.uint64)
    larger_rows[:num_syndromes] = syndrome_rows[:num_syndromes]
    larger_slots = np.full(2 * len(larger_rows), -1, dtype=np.int64)
    for row in range(num_syndromes):
        hashed, syndrome = larger_rows[row, 0], larger_rows[row, 1:]
        larger_slots[_find_syndrome(hashed, syndrome, larger_rows, larger_slots)] = row
    return larger_rows, larger_slots
