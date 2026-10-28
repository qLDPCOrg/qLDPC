# SPDX-License-Identifier: Apache-2.0

"""Exact search for an accepted fault of the smallest fault weight that violates the bound.

The search works on the flags and syndrome of each primitive fault, and never sees a circuit.  It
lists the irreducible accepted faults with compiled code.  ``fault_propagation.md`` proves that the
search is exact.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Collection, Iterator, Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._compilation import always_compiled
from qldpc.codes.distance._binary import popcount_word, trailing_zeros_word
from qldpc.codes.distance._bounded_weight_search import MinWeightBound, PauliCosetSearch

# The peak number of bytes per pair of primitive faults while building a table of pairs: the hash
# and the two primitive faults of each pair, and at most one bucket count per pair.  A bucket of
# more than 32 sets that is out of order is merge sorted, with extra memory in proportion to its
# size, which this leaves out.  That can happen when a key that many pairs share, such as the zero
# key of the pairs with equal flags, shares its bucket with another key, which then adds extra
# memory for each pair of that bucket.
_BYTES_PER_PAIR = 24

# The largest number of faults in a batch of the compiled listing.
_BATCH_SIZE = 2**12

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
        bound: The oracle's answer for that syndrome at the fault's fault weight, which gives an
            error with the syndrome and a lower bound above the fault weight.
    """

    fault_indices: tuple[int, ...]
    syndrome: int
    bound: MinWeightBound


def find_smallest_violation(
    flags: Sequence[int],
    syndromes: Sequence[int],
    stabilizers: npt.ArrayLike,
    css_weight: bool,
    max_weight: int | None,
    memory_limit: int = 2**30,  # 1 GiB
) -> Violation | None:
    """Find a violating accepted fault of the smallest fault weight, or None if there is none.

    The search keeps one primitive fault for each distinct nonzero signature, and limits the fault
    weight to ``max_weight`` and to what an irreducible accepted fault can have.  Then it runs
    ``_search_irreducible_faults``.
    """
    # A primitive fault with zero flags and zero syndrome does nothing, and two primitive faults
    # with equal (flags, syndrome) signatures cancel.  Keeping one primitive fault for each distinct
    # nonzero signature cannot hide a violation, and each signature keeps the index of its first
    # primitive fault.
    signatures: dict[tuple[int, int], int] = {}
    for index, signature in enumerate(zip(flags, syndromes, strict=True)):
        if any(signature):
            signatures.setdefault(signature, index)

    # An irreducible accepted fault has fault weight at most one more than the rank of the flags.
    flag_rank = _get_binary_rank([flag for flag, _ in signatures])
    max_fault_weight = flag_rank + 1 if max_weight is None else min(max_weight, flag_rank + 1)

    oracle = PauliCosetSearch(stabilizers, css_weight=css_weight)
    return _search_irreducible_faults(signatures, max_fault_weight, oracle, memory_limit)


def _search_irreducible_faults(
    signatures: dict[tuple[int, int], int],
    max_fault_weight: int,
    oracle: PauliCosetSearch,
    memory_limit: int = 2**30,  # 1 GiB
) -> Violation | None:
    """Check the irreducible accepted faults that can violate the bound, by increasing fault weight.

    The signatures map the (flags, syndrome) of each kept primitive fault to its DEM index, and
    ``max_fault_weight`` is the largest fault weight to check.  Return a violating fault of the
    smallest fault weight, or None if there is none.
    """
    # A primitive fault that fires no post-selection detector is an irreducible accepted fault by
    # itself, and no larger irreducible accepted fault contains it.  Check these first.
    flagged: list[tuple[int, int, int]] = []
    for (flag, syndrome), index in signatures.items():
        if flag:
            flagged.append((flag, syndrome, index))
            continue
        failure = _check_fault((index,), syndrome, oracle)
        if failure is not None:
            return failure

    # A primitive fault is heavy if its own error has no equivalent error of Pauli weight at most
    # one, and light otherwise.  A fault that contains only light primitive faults cannot violate
    # the bound, so without a heavy primitive fault there is nothing more to check.
    heavy: list[tuple[int, int, int, int]] = []
    light: list[tuple[int, int, int, int]] = []
    for flag, syndrome, index in flagged:
        bound = oracle.get_min_weight_bound(syndrome, 1)
        primitive = (flag, syndrome, index, bound.representative)
        (light if bound.within_max_weight else heavy).append(primitive)
    if not heavy:
        return None

    # Give each primitive fault an error with its syndrome, of low Pauli weight so that the search
    # can skip more faults.  The error of a light primitive fault has Pauli weight at most one.  A
    # heavy primitive fault whose syndrome is the sum of those of two light primitive faults gets
    # the product of their errors, which has Pauli weight at most two.
    light_errors = {syndrome: error for _, syndrome, _, error in light}
    for jj, (flag, syndrome, index, _) in enumerate(heavy):
        for other, error in light_errors.items():
            if syndrome ^ other in light_errors:
                heavy[jj] = (flag, syndrome, index, error ^ light_errors[syndrome ^ other])
                break

    # Order the primitive faults with the heavy ones first, and each group in DEM order, so that the
    # faults to check are the ones that start with a heavy primitive fault.
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
    indices: tuple[int, ...], syndrome: int, oracle: PauliCosetSearch
) -> Violation | None:
    """Return the fault with these indices and syndrome if it violates the bound."""
    bound = oracle.get_min_weight_bound(syndrome, len(indices))
    if bound.within_max_weight:
        return None
    return Violation(indices, syndrome, bound)


def _get_binary_rank(columns: Sequence[int]) -> int:
    """Get the GF(2) rank of binary columns that are packed into integers."""
    # The basis holds columns with distinct leading bits, largest first.  Reducing a column by each
    # basis column in turn clears these leading bits from it, so a nonzero result is independent of
    # the basis.
    basis: list[int] = []
    for column in columns:
        for row in basis:
            column = min(column, column ^ row)
        if column:
            basis.append(column)
            basis.sort(reverse=True)
    return len(basis)


####################################################################################################
# listing irreducible accepted faults


class FaultLister:
    """Lists the irreducible accepted faults that are made of given primitive faults.

    The primitive faults have nonzero flags and come in search order.  A fault of fault weight ``m``
    is listed as the positions ``j_1 < ... < j_m`` of its primitive faults, made of an independent
    prefix ``j_1, ..., j_{m-1}`` and a closing primitive fault ``j_m`` whose flags are the sum of
    the prefix flags, as ``fault_propagation.md`` explains.  Only faults whose ``j_1`` is among the
    first ``num_starts`` primitive faults are listed, in lexicographic order.

    The closing primitive fault is looked up in a table by the hash of its flags.  For fault weight
    four or more, the last two primitive faults are looked up together in a table of pairs, if it
    fits within the memory limit.  The lister also keeps a table of the syndromes of the faults that
    it has listed, which grows within the rest of the memory limit.

    Args:
        flags: The flags of each primitive fault, packed into an integer.
        syndromes: The syndrome of each primitive fault, packed into an integer.
        representatives: For each primitive fault, an error with its syndrome, packed as in
            ``PauliCosetSearch``.  Lighter errors let ``find_first`` skip more faults.
        num_qubits: The number of output qubits.
        css_weight: Whether the Pauli weight is the CSS weight, or else the physical weight.
        num_starts: The number of primitive faults that a listed fault can start with.
        memory_limit: The approximate number of bytes for the table of pairs of primitive faults,
            which has first claim on them if it fits, since it saves more time, and the table of
            syndromes.
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
        # Each primitive fault is a row of words: the hashes of its flags and of its syndrome, then
        # its flags, its syndrome, and the X and Z parts of its representative.  Every part is
        # linear, so the row of a fault is the sum of the rows of its primitive faults.
        flag_words = _to_words(flags, _get_num_words(flags))
        syndrome_words = _to_words(syndromes, _get_num_words(syndromes))
        mask = (1 << num_qubits) - 1
        num_qubit_words = _get_num_words([mask])
        x_parts = [representative & mask for representative in representatives]
        z_parts = [representative >> num_qubits for representative in representatives]
        self._rows = np.hstack(
            [
                _hash_words(flag_words)[:, None],
                _hash_words(syndrome_words)[:, None],
                flag_words,
                syndrome_words,
                _to_words(x_parts, num_qubit_words),
                _to_words(z_parts, num_qubit_words),
            ]
        )
        self._num_flag_words = flag_words.shape[1]
        self._num_syndrome_words = syndrome_words.shape[1]
        self._css_weight = css_weight
        self._num_starts = num_starts
        self._singles = _get_closing_table(self._rows[:, 0], 1)

        # Each syndrome in the table takes a row of its hash and its words, and two slots, and the
        # slots are numbered by 32 bits (see ``_find_syndrome``).
        num_primitives = len(self._rows)
        pair_bytes = num_primitives * (num_primitives - 1) // 2 * _BYTES_PER_PAIR
        self._pairs_fit = pair_bytes <= memory_limit
        free_bytes = memory_limit - pair_bytes if self._pairs_fit else memory_limit
        bytes_per_syndrome = 8 * (self._num_syndrome_words + 3)
        self._max_syndromes = min(free_bytes // bytes_per_syndrome, 2**31)
        self._syndrome_rows, self._syndrome_slots = _get_empty_syndrome_table(
            self._num_syndrome_words
        )
        self._num_syndromes = 0

    def find_first(
        self, fault_weight: int, find_violation: Callable[[npt.NDArray[np.uint64], int], int]
    ) -> tuple[tuple[int, ...], int] | None:
        """Find the first listed fault of a fault weight that violates the bound.

        The faults are passed to ``find_violation(syndromes, fault_weight)`` in batches, one
        syndrome per row, split into 64-bit words with word ``j`` holding bits ``64 j`` and up.  It
        returns the index of the first syndrome that violates the bound at this fault weight, or the
        number of rows if there is none.  A fault is left out if the product of the representatives
        of its primitive faults has Pauli weight at most its fault weight, since then it satisfies
        the bound.

        A fault is also left out if a fault listed earlier, by this call or by an earlier call
        without a violation, has its syndrome, unless the table of syndromes had no room for it.
        That fault had the same or a smaller fault weight, so this is exact if the fault weight
        never decreases from one call to the next, if every call uses the same ``find_violation``,
        and if a syndrome that does not violate the bound at one fault weight does not violate it at
        any larger one, as for the bound ``μ(s) ≤ fault_weight``.  A call that finds a violation
        clears the table, since it holds syndromes of faults that were not checked.

        Returns:
            The positions of the primitive faults of the first violating fault and its syndrome, or
            None if there is none.
        """
        pairs = self._pairs if fault_weight >= 4 else None
        keys, members = self._singles if pairs is None else pairs
        batch_members = np.empty((_BATCH_SIZE, fault_weight), dtype=np.int64)
        batch_syndromes = np.empty((_BATCH_SIZE, self._num_syndrome_words), dtype=np.uint64)
        for count, syndrome_rows, syndrome_slots, num_syndromes in _iter_irreducible_fault_batches(
            self._rows,
            self._num_flag_words,
            keys,
            members,
            self._num_starts,
            self._css_weight,
            batch_members,
            batch_syndromes,
            self._syndrome_rows,
            self._syndrome_slots,
            self._num_syndromes,
            self._max_syndromes,
        ):
            self._syndrome_rows, self._syndrome_slots, self._num_syndromes = (
                syndrome_rows,
                syndrome_slots,
                num_syndromes,
            )
            index = find_violation(batch_syndromes[:count], fault_weight)
            if index < count:
                self._syndrome_rows, self._syndrome_slots = _get_empty_syndrome_table(
                    self._num_syndrome_words
                )
                self._num_syndromes = 0
                positions = tuple(batch_members[index].tolist())
                syndrome = int.from_bytes(batch_syndromes[index].astype("<u8").tobytes(), "little")
                return positions, syndrome
        return None

    @functools.cached_property
    def _pairs(self) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32]] | None:
        """The table of pairs of primitive faults, if it fits within the memory limit."""
        return _get_closing_table(self._rows[:, 0], 2) if self._pairs_fit else None


@always_compiled
def _get_closing_table(
    hashes: npt.NDArray[np.uint64], size: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32]]:
    """Build the table that finds closing primitive faults.

    The table lists the sets of ``size`` primitive faults (one, or a pair), sorted by the hash of
    their summed flags.  The sets are listed in lexicographic order and sorted stably, so the sets
    with each hash stay in that order.  Return the sorted hashes and the sets.

    The sort is a counting sort into buckets by the high bits of the hash, with about one bucket per
    set, which keeps the sets of each bucket in order, followed by a stable sort of each bucket.
    """
    num_primitives = len(hashes)
    num_sets = num_primitives if size == 1 else num_primitives * (num_primitives - 1) // 2

    # Use 2**bucket_bits buckets, the largest power of two that is at most the number of sets (and
    # at least two), indexed by the top bits of each key.
    bucket_bits = 1
    while 2 << bucket_bits <= num_sets:
        bucket_bits += 1
    shift = np.uint64(64 - bucket_bits)

    # Count the sets in each bucket, and turn the counts into the start of each bucket.  The sets
    # are the single primitive faults ``first``, with the hash of ``first`` as their key, or the
    # pairs ``first < second``, with the sum of their hashes as their key.  For single faults, the
    # inner loop runs only for ``second == first``.
    starts = np.zeros((1 << bucket_bits) + 1, dtype=np.int64)
    for first in range(num_primitives):
        for second in range(first + size - 1, num_primitives if size == 2 else first + 1):
            key = hashes[first] ^ hashes[second] if size == 2 else hashes[first]
            starts[np.int64(key >> shift) + 1] += 1
    for bucket in range(1 << bucket_bits):
        starts[bucket + 1] += starts[bucket]

    # Fill the buckets in the order of the sets, which moves each start to the end of its bucket.
    keys = np.empty(num_sets, dtype=np.uint64)
    members = np.empty((num_sets, size), dtype=np.int32)
    for first in range(num_primitives):
        for second in range(first + size - 1, num_primitives if size == 2 else first + 1):
            key = hashes[first] ^ hashes[second] if size == 2 else hashes[first]
            set_bucket = np.int64(key >> shift)
            slot = starts[set_bucket]
            starts[set_bucket] = slot + 1
            keys[slot] = key
            members[slot, 0] = first
            if size == 2:
                members[slot, 1] = second
    for bucket in range(1 << bucket_bits):
        start = starts[bucket - 1] if bucket else 0
        if starts[bucket] - start > 1:
            _sort_bucket(keys, members, start, starts[bucket])
    return keys, members


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
    images = np.random.default_rng(0).integers(2**64, size=(words.shape[1], 64), dtype=np.uint64)
    hashes = np.zeros(len(words), dtype=np.uint64)
    for column, column_images in enumerate(images):
        bits = (words[:, column, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)
        hashes ^= np.bitwise_xor.reduce(bits * column_images, axis=1)
    return hashes


def _get_num_words(values: Collection[int]) -> int:
    """The number of 64-bit words that hold each of these integers, at least one."""
    return max(1, (max((value.bit_length() for value in values), default=0) + 63) // 64)


def _to_words(values: Sequence[int], num_words: int) -> npt.NDArray[np.uint64]:
    """Split each integer into words, with word ``j`` holding bits ``64 j`` and up."""
    data = b"".join(value.to_bytes(8 * num_words, "little") for value in values)
    words = np.frombuffer(data, dtype="<u8").reshape(len(values), num_words)
    return words.astype(np.uint64)


@always_compiled
def _sort_bucket(
    keys: npt.NDArray[np.uint64], members: npt.NDArray[np.int32], start: int, stop: int
) -> None:
    """Sort the entries from ``start`` to ``stop`` of a closing table stably by their keys.

    A bucket usually holds few sets, or many sets with one hash (such as the pairs with equal
    flags), which are in order unless another hash shares the bucket, and then it is sorted by
    insertion.  A larger bucket that is out of order, which holds several hashes, is merge sorted
    instead, which takes memory in proportion to its size but avoids the quadratic time of
    insertion.
    """
    if stop - start > 32 and np.any(keys[start + 1 : stop] < keys[start : stop - 1]):
        order = start + np.argsort(keys[start:stop], kind="mergesort")
        keys[start:stop] = keys[order]
        members[start:stop] = members[order]
        return
    for entry in range(start + 1, stop):
        slot = entry
        while slot > start and keys[slot - 1] > keys[slot]:
            keys[slot - 1], keys[slot] = keys[slot], keys[slot - 1]
            for column in range(members.shape[1]):
                members[slot - 1, column], members[slot, column] = (
                    members[slot, column],
                    members[slot - 1, column],
                )
            slot -= 1


####################################################################################################
# compiled listing


@always_compiled
def _iter_irreducible_fault_batches(
    rows: npt.NDArray[np.uint64],
    num_flag_words: int,
    keys: npt.NDArray[np.uint64],
    members: npt.NDArray[np.int32],
    num_starts: int,
    css_weight: bool,
    batch_members: npt.NDArray[np.int64],
    batch_syndromes: npt.NDArray[np.uint64],
    syndrome_rows: npt.NDArray[np.uint64],
    syndrome_slots: npt.NDArray[np.int64],
    num_syndromes: int,
    max_syndromes: int,
) -> Iterator[tuple[int, npt.NDArray[np.uint64], npt.NDArray[np.int64], int]]:
    """List irreducible accepted faults into batches, as ``FaultLister.find_first`` describes.

    The rows describe the primitive faults, as in ``FaultLister``, and the keys and members are a
    table from ``_get_closing_table``.  Each listed fault that is not left out gets a row of the
    batch arrays: its primitive faults and its syndrome.  The first ``num_syndromes`` rows of
    ``syndrome_rows``, with ``syndrome_slots``, form the table of syndromes of ``_find_syndrome``.
    Each time that the batch is full, and once at the end, yield the number of faults in the batch
    and the table.  The caller must read a batch before resuming.

    Once the table is half full, the next fault that is added to the batch yields it early, and the
    table then doubles in size, while the old and the new table together have at most
    ``max_syndromes`` rows.  A full table takes no more syndromes.
    """
    # Columns 0 and 1 of the rows hold the hashes of the flags and of the syndrome.
    num_primitives, num_words = rows.shape
    fault_weight = batch_members.shape[1]
    num_closing = members.shape[1]
    depth = fault_weight - num_closing
    flag_start = 2
    syndrome_start = flag_start + num_flag_words
    x_start = syndrome_start + batch_syndromes.shape[1]
    z_start = x_start + (num_words - x_start) // 2

    # The loops choose the first ``depth`` primitive faults, and ``candidates`` holds the next
    # candidate for each, one past the current choice.  Row ``level`` of ``sums`` is the sum of the
    # rows of the first ``level`` choices, and their reduced flags form the basis.
    candidates = np.zeros(depth, dtype=np.int64)
    sums = np.zeros((depth + 1, num_words), dtype=np.uint64)
    basis = np.zeros((depth, num_flag_words), dtype=np.uint64)
    pivots = np.zeros(depth, dtype=np.int64)
    fault = np.empty(num_words, dtype=np.uint64)
    syndrome = np.empty(x_start - syndrome_start, dtype=np.uint64)
    reduced = np.empty(num_flag_words, dtype=np.uint64)
    count = 0

    level = 0
    while level >= 0:
        choice = candidates[level]
        if choice == (num_starts if level == 0 else num_primitives):
            level -= 1
            continue
        candidates[level] = choice + 1

        # The choice must keep the flags of the chosen primitive faults independent.
        pivot = _reduce_flags(
            rows[choice, flag_start:syndrome_start], basis[:level], pivots, reduced
        )
        if pivot < 0:
            continue
        basis[level] = reduced
        pivots[level] = pivot
        for word in range(num_words):
            sums[level + 1, word] = sums[level, word] ^ rows[choice, word]
        if level + 1 < depth:
            level += 1
            candidates[level] = choice + 1
            continue

        # Look up the remaining one or two primitive faults by the hash of their flags, which must
        # equal that of the chosen ones.  Keep those that come after the last choice and whose flags
        # sum to exactly those of the chosen ones, since different flags can share a hash.  Of two,
        # the first must also keep the flags of the prefix independent.
        low = np.searchsorted(keys, sums[depth, 0], "left")
        high = np.searchsorted(keys, sums[depth, 0], "right")
        for entry in range(low, high):
            if members[entry, 0] <= choice:
                continue
            closes = True
            for word in range(flag_start, syndrome_start):
                flags = sums[depth, word]
                for member in members[entry]:
                    flags ^= rows[member, word]
                closes = closes and not flags
            if not closes:
                continue

            # Leave out the fault if a fault listed earlier has its syndrome.  This comes before the
            # costlier test of independence, which a fault left out here does not need.
            hashed = sums[depth, 1]
            for member in members[entry]:
                hashed ^= rows[member, 1]
            for word in range(len(syndrome)):
                syndrome[word] = sums[depth, syndrome_start + word]
                for member in members[entry]:
                    syndrome[word] ^= rows[member, syndrome_start + word]
            slot = _find_syndrome(hashed, syndrome, syndrome_rows, syndrome_slots)
            if slot < 0:
                continue
            first_flags = rows[members[entry, 0], flag_start:syndrome_start]
            if num_closing == 2 and _reduce_flags(first_flags, basis, pivots, reduced) < 0:
                continue

            # The fault is listed, so add its syndrome to the table if there is room, also if the
            # fault is left out below, since it satisfies the bound then.
            if num_syndromes < len(syndrome_rows):
                syndrome_rows[num_syndromes, 0] = hashed
                syndrome_rows[num_syndromes, 1:] = syndrome
                syndrome_slots[slot] = num_syndromes
                num_syndromes += 1

            # The product of the representatives is an error with the fault's syndrome, so if its
            # Pauli weight is at most the fault weight, the fault satisfies the bound.  Otherwise
            # add the fault to the batch.
            for word in range(x_start, num_words):
                fault[word] = sums[depth, word]
                for member in members[entry]:
                    fault[word] ^= rows[member, word]
            weight = _pauli_weight(fault[x_start:z_start], fault[z_start:], css_weight)
            if weight <= fault_weight:
                continue
            batch_syndromes[count] = syndrome
            for index in range(depth):
                batch_members[count, index] = candidates[index] - 1
            batch_members[count, depth:] = members[entry]
            count += 1

            # Yield the batch when it is full, or early when the table should grow, and then grow
            # the table.  Growing it anywhere else, or keeping a flag across the yield, makes numba
            # compile this loop to slower code.
            num_rows = len(syndrome_rows)
            grow = 2 * num_syndromes >= num_rows and 3 * num_rows <= max_syndromes
            if count == len(batch_members) or grow:
                yield count, syndrome_rows, syndrome_slots, num_syndromes
                count = 0
                num_rows = len(syndrome_rows)
                if 2 * num_syndromes >= num_rows and 3 * num_rows <= max_syndromes:
                    syndrome_rows, syndrome_slots = _grow_syndrome_table(
                        syndrome_rows, num_syndromes
                    )
    yield count, syndrome_rows, syndrome_slots, num_syndromes


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

    Each row of ``syndrome_rows`` that is in the table holds the hash of a syndrome and then its
    words.  ``syndrome_slots`` has twice as many entries as ``syndrome_rows`` has rows, each a row
    or -1 for none, and a syndrome is in the first slot, from the one that its hash picks, that is
    empty or holds it.  Return -1 if the syndrome is in the table, and otherwise the empty slot for
    it.
    """
    # The number of slots is a power of two, at most 2**32, and the slot is the top bits of the low
    # 32 bits of the product, which fits in 64 bits.
    low_bits = np.uint64(0xFFFFFFFF)
    shift = np.uint64(32 - trailing_zeros_word(np.uint64(len(syndrome_slots))))
    slot = np.int64((((hashed & low_bits) * _HASH_MULTIPLIER) & low_bits) >> shift)
    while syndrome_slots[slot] >= 0:
        row = syndrome_slots[slot]
        equal = syndrome_rows[row, 0] == hashed
        for word in range(len(syndrome)):
            equal = equal and syndrome_rows[row, 1 + word] == syndrome[word]
        if equal:
            return -1
        slot = (slot + 1) % len(syndrome_slots)
    return int(slot)


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
