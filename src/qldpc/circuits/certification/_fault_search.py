# SPDX-License-Identifier: Apache-2.0

"""Exact search for an accepted fault of the smallest fault weight that violates the bound.

The search sees only the flags and the syndrome of each primitive fault, never a circuit.
``fault_propagation.md`` defines the search's terms, describes the search, and proves that it is
exact, and ``FaultLister`` defines the terms of its implementation.  Comments and docstrings cite
the guide's sections by title.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterator, Sequence
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
from qldpc.codes.distance import CosetLeaderSearch, DistanceMethod, MinWeightBound

# The peak memory, in bytes per set, of building a closing map of sets of two or more classes (see
# ``FaultLister``), apart from 4 bytes for each class of a set.  The finished map takes 8 bytes per
# set for its hash, 4 for each class, and at most 32 for the slots of its hash map.  While its
# closing table is built, before the slots, each set takes at most 8 more for a bucket count, so the
# peak is 40 bytes plus 4 for each class: 48 for a pair.  This leaves out the extra memory of merge
# sorting a bucket (a range of the table whose keys share their top bits) of more than 32 sets that
# is out of order (see ``_sort_bucket``), and the array into which ``_list_class_codewords`` lists
# class codewords, whose rows have room for all the class codewords of one lookup (see
# ``FaultLister._iter_class_tuples``).
_BYTES_PER_SET = 40

# The largest number of classes in a closing set (see ``FaultLister``).  Each class that a closing
# set adds removes about one factor of the number of classes from the time of listing class
# codewords, at the cost of another factor in the size of its closing map.
_MAX_CLOSING_SIZE = 4

# A closing map of sets of more than two classes is built only once the listing of class codewords
# at the last fault weight made at least this many lookups for each set of the map, so that building
# the map takes time of the order of that listing.  A map is wasted if the next fault weight finds a
# violation at once, which costs about one or two such listings.
_LOOKUPS_PER_CLOSING_SET = 1

# The number of choices of members of a class tuple whose syndromes ``_expand_class_tuples``
# remembers, to skip the faults that complete a later choice with a remembered syndrome.
_MAX_REMEMBERED_CHOICES = 2**12

# The number of faults in the first batch of a call of ``FaultLister.find_first``.  Each batch
# doubles in size, so that an early violation is found quickly, and larger later batches spread the
# overhead of each call of the oracle over more faults.
_INITIAL_BATCH_SIZE = 2**6

# The largest number of faults in a batch.
_BATCH_SIZE = 2**12

# The number of class codewords (see ``FaultLister``) after which ``_list_class_codewords`` returns,
# to be resumed by its next call, so that memory does not grow with their number.
_CODEWORD_BATCH_SIZE = 2**10

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
    method: DistanceMethod = "brouwer_zimmermann",
) -> Violation | None:
    """Find a violating accepted fault of the smallest fault weight, or None if there is none.

    ``flags`` and ``syndromes`` hold, for each primitive fault in DEM order, its flags or its
    syndrome packed into an integer.  ``max_weight``, if not None, limits the fault weight.
    ``stabilizers``, ``css_weight``, and ``method`` configure the oracle of the search, a
    ``CosetLeaderSearch``.
    """
    # Keep the first primitive fault of each distinct nonzero signature (see "Duplicate and zero
    # signatures can be dropped").
    signatures: dict[tuple[int, int], int] = {}
    for index, signature in enumerate(zip(flags, syndromes, strict=True)):
        if any(signature):
            signatures.setdefault(signature, index)

    # Check faults only up to fault weight rank(P) + 1, for the matrix P of the flags (see
    # "Irreducible accepted faults have fault weight at most rank(P) + 1").
    # Repeated flags do not change the rank, so each value enters the matrix once.
    distinct_flags = list(dict.fromkeys(flag for flag, _ in signatures))
    flag_rank = get_rank_packed(
        _to_words(distinct_flags, _get_num_words_for_values(distinct_flags))
    )
    max_fault_weight = flag_rank + 1 if max_weight is None else min(max_weight, flag_rank + 1)

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
    # Check each primitive fault with zero flags alone (see "Listing each irreducible accepted fault
    # once"), and keep the others for ``FaultLister``.
    flagged: list[tuple[int, int, int]] = []
    for (flag, syndrome), index in signatures.items():
        if flag:
            flagged.append((flag, syndrome, index))
            continue
        failure = _check_fault((index,), syndrome, oracle)
        if failure is not None:
            return failure

    # Split the other primitive faults into heavy and light ones.  A fault that contains no heavy
    # primitive fault satisfies the bound (see "A violating fault contains a heavy primitive
    # fault"), so the search stops if none is heavy.
    heavy: list[tuple[int, int, int, int]] = []
    light: list[tuple[int, int, int, int]] = []
    for flag, syndrome, index in flagged:
        bound = oracle.get_min_weight_bound(syndrome, 1)
        primitive = (flag, syndrome, index, bound.representative)
        (light if bound.within_max_weight else heavy).append(primitive)
    if not heavy:
        return None

    # Give each primitive fault a representative of low Pauli weight, so that ``FaultLister`` skips
    # more faults (see "Checking one fault").  A light primitive fault's representative has Pauli
    # weight at most one.  A heavy primitive fault whose syndrome is the sum of the syndromes of two
    # light ones gets the product of their representatives.  The pair depends only on the syndrome,
    # so the search runs once for each distinct heavy syndrome.  Compiled code does it, since its
    # time is proportional to the number of distinct heavy syndromes times the number of light
    # ones.  Any other heavy primitive fault keeps the error that the oracle returned for it.
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

    # The oracle needs only small tables, so the lister may use the whole memory limit.
    lister = FaultLister(
        [flag for flag, _, _, _ in primitives],
        [syndrome for _, syndrome, _, _ in primitives],
        [representative for _, _, _, representative in primitives],
        oracle.num_qubits,
        oracle.css_weight,
        len(heavy),
        memory_limit,
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
    passes its heavy primitive faults first, as the start members.  "Listing each irreducible
    accepted fault once" in ``fault_propagation.md`` defines flag classes, start classes, class
    codewords, and closing sets, and proves that each such fault is listed once.
    The classes are numbered in the order of their first members.  A class tuple is a row of classes
    from which the lister expands faults: a start class twice for fault weight two, and a class
    codeword that starts with a start class for larger fault weights.

    Each class is a row of 64-bit words: the hash of its flags, then its flags.  Each primitive
    fault is a row of the hash of its syndrome, its syndrome, and the X and Z parts of its
    representative.  The hashes and the other parts are linear, so the row of a set of classes or
    of primitive faults is the sum of their rows.

    A closing map of a size has two parts: a closing table, which lists the sets of that many
    classes sorted by the hash of their summed flags, and a hash map from each hash to its first
    entry in the table.  Fault weight three uses the closing map of single classes.  Larger fault
    weights use the closing map of pairs if it fits within the memory limit and has at most 2**31
    pairs, and otherwise that of single classes.  The table of syndromes of listed faults (see
    ``find_first``) grows within the rest of the memory limit, up to 2**31 syndromes.  With the map
    of pairs, a fault weight ``m`` may use a closing map of up to ``min(m - 2, _MAX_CLOSING_SIZE)``
    classes instead: one that was built for an earlier fault weight, or one that is built now
    because the listing of class codewords at the last fault weight made enough lookups (see
    ``_LOOKUPS_PER_CLOSING_SET``), if the table of syndromes can give up the room for it and still
    grow to the same largest size (see ``_take_table_room``).

    Args:
        flags: The flags of each primitive fault, packed into an integer.
        syndromes: The syndrome of each primitive fault, packed into an integer.
        representatives: For each primitive fault, a Pauli error with its syndrome, packed as in
            ``CosetLeaderSearch``.  Lighter Pauli errors let ``find_first`` skip more faults.
        memory_limit: The approximate number of bytes for the closing maps of two or more classes
            and the table of syndromes.  The closing map of pairs has first claim, since it saves
            more time.
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
    ) -> None:
        # Group the primitive faults into classes by their flags, each with its members in order,
        # and order the classes by their first members, so that the start classes come first.
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
        # describes.
        class_flags = [flags[members[0]] for members in member_lists]
        flag_words = _to_words(class_flags, _get_num_words_for_values(class_flags))
        self._class_rows = np.hstack([_hash_words(flag_words)[:, None], flag_words])
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

        # Split the memory limit, and start with no closing maps, which are built when first
        # needed, and an empty table of syndromes.
        self._should_build_closing_pairs, self._max_syndromes = _split_memory_limit(
            num_classes=len(member_lists),
            num_syndrome_words=self._num_syndrome_words,
            memory_limit=memory_limit,
        )
        self._closing_maps: dict[
            int, tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.int64]]
        ] = {}
        self._num_lookups = 0
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

        A listed fault is left out if the product of the representatives of its primitive faults
        has Pauli weight at most its fault weight, or if its syndrome is in the table of syndromes.
        The table holds the syndromes of faults listed earlier, by this call or by an earlier call
        that found no violation, as far as it had room.  A call that finds a violation clears the
        table, since the table can hold syndromes of faults listed after that violation, which were
        never checked.  Leaving these faults out does not change the result if the fault weight
        never decreases from one call to the next, every call uses the same ``find_violation``, and
        a syndrome that satisfies the bound at one fault weight satisfies it at every larger one.
        "Checking one fault" in ``fault_propagation.md`` shows why.

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
        # end, check the last batch.
        for class_tuples in self._iter_class_tuples(fault_weight):
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

    def _iter_class_tuples(self, fault_weight: int) -> Iterator[npt.NDArray[np.int64]]:
        """Yield the class tuples of a fault weight in batches, one class tuple per row.

        For fault weight two, each class tuple is ``(c, c)`` for a start class ``c`` with at least
        two members.  For larger fault weights, the class tuples are the class codewords that start
        with a start class.  Each batch is a view of an array that the next batch overwrites.
        """
        if fault_weight == 2:
            class_sizes = np.diff(self._class_offsets[: self._num_start_classes + 1])
            yield np.repeat(np.flatnonzero(class_sizes > 1), 2).reshape(-1, 2)
            return

        # Find the last classes of a class codeword together, as a closing set, in a closing map,
        # after choosing the classes before them.  List the class codewords in batches, into an
        # array of ``_CODEWORD_BATCH_SIZE`` rows plus as many as one lookup can add, as
        # ``_list_class_codewords`` needs.
        closing_map = self._get_closing_map(fault_weight)
        closing_size = closing_map[1].shape[1]
        num_chosen = fault_weight - closing_size
        num_classes, num_words = self._class_rows.shape
        max_lookup_codewords = math.comb(num_classes, closing_size - 1)
        codewords = np.empty(
            (_CODEWORD_BATCH_SIZE + max_lookup_codewords, fault_weight), dtype=np.int64
        )

        # The state of the listing (see ``_list_class_codewords``).  The basis has room for the
        # chosen classes and every closing class but the last.
        next_classes = np.zeros(num_chosen, dtype=np.int64)
        chosen_row_sums = np.zeros((num_chosen + 1, num_words), dtype=np.uint64)
        basis = np.zeros((fault_weight - 1, num_words - 1), dtype=np.uint64)
        pivots = np.zeros(fault_weight - 1, dtype=np.int64)
        walk_state = np.zeros(2, dtype=np.int64)
        while walk_state[0] >= 0:
            num_codewords = _list_class_codewords(
                self._class_rows,
                self._num_start_classes,
                *closing_map,
                next_classes,
                chosen_row_sums,
                basis,
                pivots,
                walk_state,
                codewords,
            )
            yield codewords[:num_codewords]
        self._num_lookups = int(walk_state[1])

    def _get_closing_map(
        self, fault_weight: int
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        """The closing map for a fault weight of three or more, as the class docstring describes."""
        hashes = self._class_rows[:, 0]
        size = 1 if fault_weight == 3 or not self._should_build_closing_pairs else 2
        while 2 <= size < min(fault_weight - 2, _MAX_CLOSING_SIZE):
            # Build the next map if the last listing made enough lookups, and if the table of
            # syndromes can give up the room for it without changing the largest size it can reach.
            if size + 1 not in self._closing_maps:
                num_sets = math.comb(len(hashes), size + 1)
                max_syndromes = _take_table_room(
                    num_sets * (_BYTES_PER_SET + 4 * (size + 1)),
                    self._max_syndromes,
                    len(self._syndrome_rows),
                    self._num_syndrome_words,
                )
                if (
                    self._num_lookups < num_sets * _LOOKUPS_PER_CLOSING_SET
                    or max_syndromes < 0
                    or num_sets > 2**31
                ):
                    break
                self._closing_maps[size + 1] = _get_closing_map(hashes, size + 1)
                self._max_syndromes = max_syndromes
            size += 1
        if size not in self._closing_maps:
            self._closing_maps[size] = _get_closing_map(hashes, size)
        return self._closing_maps[size]


def _split_memory_limit(
    num_classes: int, num_syndrome_words: int, memory_limit: int
) -> tuple[bool, int]:
    """Split a memory limit between the closing map of pairs and the table of syndromes.

    Return whether to build the closing map of the pairs of ``num_classes`` classes, and how many
    syndromes of ``num_syndrome_words`` words the rest of the limit has room for, at most 2**31.
    ``_table_should_grow`` takes this number as ``max_syndromes``.
    """
    # The closing map of pairs takes ``_BYTES_PER_SET`` bytes, and 4 for each class, per pair.  It
    # is built only if it fits and has at most 2**31 pairs, so that its hash map has at most 2**32
    # slots, as slot numbers of 32 bits allow (see ``_find_syndrome_slot``).  The table of
    # syndromes has room for at most 2**31 syndromes for the same reason.
    num_pairs = num_classes * (num_classes - 1) // 2
    pair_bytes = num_pairs * (_BYTES_PER_SET + 8)
    should_build_pairs = pair_bytes <= memory_limit and num_pairs <= 2**31
    free_bytes = memory_limit - pair_bytes if should_build_pairs else memory_limit
    max_syndromes = free_bytes // _get_syndrome_bytes(num_syndrome_words)
    return should_build_pairs, min(max_syndromes, 2**31)


def _take_table_room(
    num_bytes: int, max_syndromes: int, num_rows: int, num_syndrome_words: int
) -> int:
    """Take room for a number of bytes from a table of syndromes, if it still reaches the same size.

    The table has room for ``max_syndromes`` syndromes of ``num_syndrome_words`` words, and has
    ``num_rows`` rows.  Return how many syndromes it has room for after giving up the bytes.  Return
    -1 instead if it would then have room for fewer than its rows.  Also return -1 if an empty
    table, which starts with ``_INITIAL_SYNDROME_TABLE_SIZE`` rows, could then not grow to as many
    rows as before (see ``_get_largest_table_size``), since it would take fewer syndromes.  The
    table grew from that size by the same doublings, so it also still grows as far as before, and
    so does each table after a violation empties it.
    """
    num_syndromes = -(-num_bytes // _get_syndrome_bytes(num_syndrome_words))
    rest = max_syndromes - num_syndromes
    largest_size = _get_largest_table_size(_INITIAL_SYNDROME_TABLE_SIZE, max_syndromes)
    if (
        rest < num_rows
        or _get_largest_table_size(_INITIAL_SYNDROME_TABLE_SIZE, rest) < largest_size
    ):
        return -1
    return rest


def _get_largest_table_size(num_rows: int, max_syndromes: int) -> int:
    """The number of rows to which a table of syndromes with this many rows can grow.

    It doubles as long as ``_table_should_grow`` allows, with as many syndromes as rows.
    """
    while _table_should_grow(num_rows, num_rows, max_syndromes):
        num_rows *= 2
    return num_rows


def _get_syndrome_bytes(num_syndrome_words: int) -> int:
    """The bytes that a syndrome of this many words takes in a table of syndromes.

    Each syndrome takes a row of its hash and its words, and two slots.
    """
    return 8 * (num_syndrome_words + 3)


def _get_closing_map(
    hashes: npt.NDArray[np.uint64], size: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.int64]]:
    """Build a closing map of the sets of ``size`` classes.

    ``hashes`` holds the hash of the flags of each class.  Return the keys of the closing table, the
    hashes of the summed flags of its sets in increasing order, its sets of classes (see
    ``_get_closing_table``), and the slots of its hash map (see ``_get_key_slots``).
    """
    keys, sets = _get_closing_table(hashes, size)
    return keys, sets, _get_key_slots(keys)


@always_compiled
def _get_closing_table(
    hashes: npt.NDArray[np.uint64], size: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32]]:
    """Build the table that finds closing sets of ``size`` classes.

    The table lists the sets of ``size`` classes, each in increasing order, sorted by the hash of
    their summed flags.  The sets are listed in lexicographic order and sorted stably, so the sets
    with each hash stay in that order.  Return the sorted hashes and the sets.
    """
    num_classes = len(hashes)
    num_sets = _get_num_sets(num_classes, size)

    # Use 2**bucket_bits buckets, the largest power of two that is at most the number of sets (and
    # at least two), indexed by the top bits of each key.
    bucket_bits = 1
    while 2 << bucket_bits <= num_sets:
        bucket_bits += 1
    shift = np.uint64(64 - bucket_bits)

    # Count the sets in each bucket, and turn the counts into the start of each bucket.
    bucket_starts = np.zeros((1 << bucket_bits) + 1, dtype=np.int64)
    keys = np.empty(0, dtype=np.uint64)
    sets = np.empty((0, size), dtype=np.int32)
    _place_sets(hashes, size, shift, bucket_starts, keys, sets, False)
    for bucket in range(1 << bucket_bits):
        bucket_starts[bucket + 1] += bucket_starts[bucket]

    # Fill the buckets in the order of the sets, which moves each start to the end of its bucket.
    keys = np.empty(num_sets, dtype=np.uint64)
    sets = np.empty((num_sets, size), dtype=np.int32)
    _place_sets(hashes, size, shift, bucket_starts, keys, sets, True)

    # Sort each bucket stably (see ``_sort_bucket``).
    for bucket in range(1 << bucket_bits):
        start = bucket_starts[bucket - 1] if bucket else 0
        if bucket_starts[bucket] - start > 1:
            _sort_bucket(keys, sets, start, bucket_starts[bucket])
    return keys, sets


@always_compiled
def _place_sets(
    hashes: npt.NDArray[np.uint64],
    size: int,
    shift: np.uint64,
    bucket_starts: npt.NDArray[np.int64],
    keys: npt.NDArray[np.uint64],
    sets: npt.NDArray[np.int32],
    fill: bool,
) -> None:
    """Count, or place, the sets of ``size`` classes in the buckets of a closing table.

    The sets are visited in lexicographic order, and the key of a set is the sum of the hashes of
    its classes, whose top bits pick its bucket.  Without ``fill``, add one to
    ``bucket_starts[b + 1]`` for each set in bucket ``b``.  With ``fill``, ``bucket_starts[b]`` is
    the next free entry of bucket ``b``: place the set there, and move the entry on.
    """
    num_classes = len(hashes)

    # Single classes and pairs, the sizes that most listings use, take a faster loop of their own.
    # For single classes, the inner loop runs only for ``second == first``.
    if size <= 2:
        for first in range(num_classes):
            for second in range(first + size - 1, num_classes if size == 2 else first + 1):
                key = hashes[first] ^ hashes[second] if size == 2 else hashes[first]
                entry = _place_set(key, shift, bucket_starts, keys, fill)
                if fill:
                    sets[entry, 0] = first
                    sets[entry, size - 1] = second
        return

    # Larger sets step through the sets of classes in lexicographic order.
    classes = np.arange(size)
    for _ in range(_get_num_sets(num_classes, size)):
        key = np.uint64(0)
        for index in range(size):
            key ^= hashes[classes[index]]
        entry = _place_set(key, shift, bucket_starts, keys, fill)
        if fill:
            for index in range(size):
                sets[entry, index] = classes[index]
        _advance_set(classes, num_classes)


@always_compiled(inline=True)
def _place_set(
    key: np.uint64,
    shift: np.uint64,
    bucket_starts: npt.NDArray[np.int64],
    keys: npt.NDArray[np.uint64],
    fill: bool,
) -> np.int64:
    """Count, or place, the key of one set as ``_place_sets`` describes, and return its entry.

    The entry is -1 when counting.
    """
    bucket = np.int64(key >> shift)
    if not fill:
        bucket_starts[bucket + 1] += 1
        return np.int64(-1)
    entry = bucket_starts[bucket]
    bucket_starts[bucket] = entry + 1
    keys[entry] = key
    return entry


@always_compiled
def _get_num_sets(num_classes: int, size: int) -> int:
    """The number of sets of ``size`` classes out of ``num_classes``, the binomial coefficient."""
    if size < 0 or size > num_classes:
        return 0
    num_sets = 1
    for index in range(size):
        num_sets = num_sets * (num_classes - index) // (index + 1)
    return num_sets


@always_compiled(inline=True)
def _advance_set(classes: npt.NDArray[np.int64], num_classes: int) -> None:
    """Replace one or more increasing classes by the next set of as many in lexicographic order.

    The last set, which ends with the last class, has no next set, and becomes an invalid set.
    """
    size = len(classes)
    position = size - 1
    while position > 0 and classes[position] == num_classes - size + position:
        position -= 1
    classes[position] += 1
    for later in range(position + 1, size):
        classes[later] = classes[later - 1] + 1


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
    # Bit ``b`` of word ``c`` adds ``images[c, b]`` to the hash.  Tabulate the hash of each value of
    # each byte of each word, so that a row takes one lookup per byte rather than steps per bit.
    images = np.random.default_rng(0).integers(2**64, size=(words.shape[1], 64), dtype=np.uint64)
    byte_images = images.reshape(words.shape[1], 8, 8)
    byte_values = np.arange(256)
    tables = np.zeros((words.shape[1], 8, 256), dtype=np.uint64)
    for bit in range(8):
        has_bit = ((byte_values >> bit) & 1).astype(np.uint64)
        tables ^= byte_images[:, :, bit, None] * has_bit
    return _hash_bytes(np.ascontiguousarray(words), tables)


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


@always_compiled
def _sort_bucket(
    keys: npt.NDArray[np.uint64], sets: npt.NDArray[np.int32], start: int, stop: int
) -> None:
    """Sort a bucket, the entries from ``start`` to ``stop`` of a closing table, stably by key.

    A bucket of at most 32 entries, or one in order, such as a bucket of many sets with one hash, is
    sorted by insertion, which takes linear time on entries in order.  A larger bucket out of order
    is merge sorted, which takes memory in proportion to its size but avoids the quadratic time of
    insertion.
    """
    if stop - start > 32 and np.any(keys[start + 1 : stop] < keys[start : stop - 1]):
        order = start + np.argsort(keys[start:stop], kind="mergesort")
        keys[start:stop] = keys[order]
        sets[start:stop] = sets[order]
        return
    for inserted in range(start + 1, stop):
        entry = inserted
        while entry > start and keys[entry - 1] > keys[entry]:
            keys[entry - 1], keys[entry] = keys[entry], keys[entry - 1]
            for column in range(sets.shape[1]):
                sets[entry - 1, column], sets[entry, column] = (
                    sets[entry, column],
                    sets[entry - 1, column],
                )
            entry -= 1


####################################################################################################
# compiled listing


@always_compiled
def _list_class_codewords(
    class_rows: npt.NDArray[np.uint64],
    num_start_classes: int,
    closing_keys: npt.NDArray[np.uint64],
    closing_sets: npt.NDArray[np.int32],
    closing_slots: npt.NDArray[np.int64],
    next_classes: npt.NDArray[np.int64],
    chosen_row_sums: npt.NDArray[np.uint64],
    basis: npt.NDArray[np.uint64],
    pivots: npt.NDArray[np.int64],
    walk_state: npt.NDArray[np.int64],
    codewords: npt.NDArray[np.int64],
) -> int:
    """List class codewords into an array, and return their number.

    ``class_rows`` holds the row of each class (see ``FaultLister``), and ``closing_keys``,
    ``closing_sets``, and ``closing_slots`` are a closing map of the sets of ``s`` classes from
    ``_get_closing_map``.  The listing chooses independent classes, starting with one of the first
    ``num_start_classes`` classes, and looks up the closing sets of each choice, as "Listing each
    irreducible accepted fault once" in ``fault_propagation.md`` describes.  Each class codeword is
    a row of classes in increasing order.

    The level is the index, among the chosen classes, of the class being chosen.  The listing keeps
    its state between calls in ``next_classes`` (at each level, the class after the chosen one),
    ``chosen_row_sums`` (row ``i`` is the sum of the class rows of the classes chosen at the first
    ``i`` levels), ``basis`` and ``pivots`` (the reduced flags of the chosen classes, as in
    ``_reduce_flags``, in their first ``len(next_classes)`` rows), and ``walk_state`` (the level of
    the next choice, and the number of lookups made so far).  They start as zeros, and the level is
    -1 once every class codeword is listed.  The other ``s - 1`` rows of ``basis`` and ``pivots``
    hold the closing classes being checked.

    The listing pauses once at most ``C(num_classes, s - 1)`` rows of the array are free, since one
    lookup adds at most that many class codewords (see "Listing each irreducible accepted fault
    once").
    """
    # Column 0 of the class rows holds the hash of the flags, and the other columns the flags.
    num_classes, num_words = class_rows.shape
    num_closing = closing_sets.shape[1]
    num_chosen = len(next_classes)
    max_lookup_codewords = _get_num_sets(num_classes, num_closing - 1)
    reduced = np.empty(num_words - 1, dtype=np.uint64)

    # The closing keys serve as rows of one column, a hash and no syndrome words, so that
    # ``_find_syndrome_slot`` compares the keys alone (see ``_get_key_slots``).  Make these arrays
    # once, rather than for each lookup, which would slow the listing.
    key_rows = closing_keys.reshape((len(closing_keys), 1))
    no_words = closing_keys[:0]

    num_codewords = 0
    current_level, num_lookups = walk_state[0], walk_state[1]
    while current_level >= 0 and num_codewords < len(codewords) - max_lookup_codewords:
        # Choose the next candidate class at this level, or go back a level if there is none.  Only
        # a start class can be chosen first.
        chosen_class = next_classes[current_level]
        if chosen_class == (num_start_classes if current_level == 0 else num_classes):
            current_level -= 1
            continue
        next_classes[current_level] = chosen_class + 1

        # The chosen classes must stay independent.  Then choose a later class at the next level,
        # unless all ``num_chosen`` classes are chosen.
        pivot = _reduce_flags(class_rows[chosen_class, 1:], basis[:current_level], pivots, reduced)
        if pivot < 0:
            continue
        basis[current_level] = reduced
        pivots[current_level] = pivot
        for word in range(num_words):
            chosen_row_sums[current_level + 1, word] = (
                chosen_row_sums[current_level, word] ^ class_rows[chosen_class, word]
            )
        if current_level + 1 < num_chosen:
            current_level += 1
            next_classes[current_level] = chosen_class + 1
            continue

        # Look up the closing sets by the hash of their summed flags, which equals the hash of the
        # flags of the chosen classes.  Record a class codeword for each entry with this hash whose
        # classes come after the last chosen class, whose flags sum to exactly those of the chosen
        # classes, since different flags can share a hash, and whose classes other than the last
        # keep the chosen classes independent.
        num_lookups += 1
        key = chosen_row_sums[num_chosen, 0]
        table_entry = closing_slots[_find_syndrome_slot(key, no_words, key_rows, closing_slots)]
        while (
            table_entry >= 0
            and table_entry < len(closing_keys)
            and closing_keys[table_entry] == key
        ):
            table_entry += 1
            if closing_sets[table_entry - 1, 0] <= chosen_class:
                continue
            closing_classes = closing_sets[table_entry - 1]
            closes = True
            for word in range(1, num_words):
                remainder = chosen_row_sums[num_chosen, word]
                for closing_class in closing_classes:
                    remainder ^= class_rows[closing_class, word]
                closes = closes and not remainder
            if not closes:
                continue
            for index in range(num_closing - 1):
                pivot = _reduce_flags(
                    class_rows[closing_classes[index], 1:],
                    basis[: num_chosen + index],
                    pivots,
                    reduced,
                )
                if pivot < 0:
                    closes = False
                    break
                basis[num_chosen + index] = reduced
                pivots[num_chosen + index] = pivot
            if not closes:
                continue
            for index in range(num_chosen):
                codewords[num_codewords, index] = next_classes[index] - 1
            for index in range(num_closing):
                codewords[num_codewords, num_chosen + index] = closing_classes[index]
            num_codewords += 1
    walk_state[0], walk_state[1] = current_level, num_lookups
    return num_codewords


@always_compiled
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
    out because an earlier choice of members had the same syndrome (see "Skipping faults whose
    syndromes are listed" in ``fault_propagation.md``).

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
    table takes no more syndromes.
    """
    # Column 0 of the primitive rows holds the hash of the syndrome, the next columns hold the
    # syndrome, and the rest hold the X and then the Z part of the representative.
    fault_weight = class_tuples.shape[1]
    num_words = primitive_rows.shape[1]
    x_start = 1 + batch_syndromes.shape[1]
    z_start = x_start + (num_words - x_start) // 2

    # The key of a choice of members in the map of remembered choices: its level and whether it has
    # a start member, then its syndrome.
    choice_key = np.empty(choice_rows.shape[1] - 1, dtype=np.uint64)

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
            # (see "Listing each irreducible accepted fault once").
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
                # (see "Skipping faults whose syndromes are listed").  This needs a table that has
                # had room for every syndrome so far in this call of ``FaultLister.find_first``, as
                # that section says.  A table that is not full now has had that room.  A table grows
                # when half full whenever it can, so a full table never grows.  A table is cleared
                # only when a violation ends the listing.  This applies only past the first level
                # whose class has more than one member (see above).  Otherwise remember the choice,
                # if there is room.
                if num_syndromes < len(syndrome_rows) and level > first_branching_level:
                    choice_key[0] = np.uint64(2 * level + (1 if num_chosen_starts > 0 else 0))
                    choice_key[1:] = chosen_row_sums[level + 1, 1:x_start]
                    choice_hash = chosen_row_sums[level + 1, 0] ^ (
                        (choice_key[0] + np.uint64(1)) * _HASH_MULTIPLIER
                    )
                    slot = _find_syndrome(choice_hash, choice_key, choice_rows, choice_slots)
                    if slot < 0:
                        continue
                    if num_choices < len(choice_rows):
                        choice_rows[num_choices, 0] = choice_hash
                        choice_rows[num_choices, 1:] = choice_key
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
            # (see "Checking one fault").
            fault_row = chosen_row_sums[fault_weight]
            slot = _find_syndrome(fault_row[0], fault_row[1:x_start], syndrome_rows, syndrome_slots)
            if slot < 0:
                continue
            if num_syndromes < len(syndrome_rows):
                syndrome_rows[num_syndromes] = fault_row[:x_start]
                syndrome_slots[slot] = num_syndromes
                num_syndromes += 1

            # Leave out the fault if the product of the representatives has Pauli weight at most
            # the fault weight (see "Checking one fault"), and otherwise add it to the batch.
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
