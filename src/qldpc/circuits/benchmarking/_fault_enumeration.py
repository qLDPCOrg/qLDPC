# SPDX-License-Identifier: Apache-2.0

"""Compiled listing of the irreducible accepted faults for the fault-propagation certifier."""

from __future__ import annotations

import functools
from collections.abc import Callable, Collection, Iterator, Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from qldpc._compilation import always_compiled
from qldpc.codes.distance._binary import popcount_word, trailing_zeros_word

# The peak number of bytes per pair of primitive faults while building a table of pairs.
BYTES_PER_PAIR = 32

# The largest number of faults in a batch of the compiled listing.
BATCH_SIZE = 2**12

Result = TypeVar("Result")

####################################################################################################
# listing irreducible accepted faults


class FaultLister:
    """Lists the irreducible accepted faults that are made of given primitive faults.

    The primitive faults have nonzero flags and come in search order.  A fault of fault weight ``m``
    is listed as its primitive faults ``j_1 < ... < j_m`` in this order, where ``j_1, ..., j_{m-1}``
    have independent flags and ``j_m`` has flags equal to their sum (see Section 4.2 of
    ``fault_propagation.md``).  Only faults whose ``j_1`` is among the first ``num_starts``
    primitive faults are listed, in lexicographic order.

    The last primitive fault, or for fault weight four or more the last two if a table of their
    pairs fits within the memory limit, are looked up in a table by the hash of their flag sum.

    Args:
        flags: The flags of each primitive fault, packed into an integer.
        syndromes: The syndrome of each primitive fault, packed into an integer.
        representatives: For each primitive fault, an error with its syndrome, packed as in
            ``PauliCosetSearch``.  Lighter errors let the search skip more faults.
        num_qubits: The number of output qubits.
        css_weight: Whether the Pauli weight is the CSS weight, or else the physical weight.
        num_starts: The number of primitive faults that a listed fault can start with.
        memory_limit: The approximate number of bytes for a table of pairs of primitive faults.
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
        flag_words = to_words(flags, get_num_words(flags))
        syndrome_words = to_words(syndromes, get_num_words(syndromes))
        mask = (1 << num_qubits) - 1
        x_parts = [representative & mask for representative in representatives]
        z_parts = [representative >> num_qubits for representative in representatives]
        self._rows = np.hstack(
            [
                hash_words(flag_words)[:, None],
                hash_words(syndrome_words)[:, None],
                flag_words,
                syndrome_words,
                to_words(x_parts, get_num_words([mask])),
                to_words(z_parts, get_num_words([mask])),
            ]
        )
        self._num_flag_words = flag_words.shape[1]
        self._num_syndrome_words = syndrome_words.shape[1]
        self._css_weight = css_weight
        self._num_starts = num_starts
        self._memory_limit = memory_limit
        self._singles = get_closing_table(self._rows[:, 0], 1)

    def find_first(
        self, fault_weight: int, check: Callable[[tuple[int, ...], int], Result | None]
    ) -> tuple[Result | None, int]:
        """Check the listed faults of a fault weight in order, and stop at the first result.

        The check gets the positions in search order of the primitive faults of a fault, and the
        fault's syndrome.  It returns None if the fault satisfies the bound, and must depend only on
        the syndrome.  So a fault is not checked if its syndrome was already checked in the same
        batch, or if the product of the representatives of its primitive faults has Pauli weight at
        most its fault weight, since then it satisfies the bound.

        Returns:
            The first result of the check that is not None, or None if there is none, and the
            number of faults listed up to and including the fault with that result, or of all
            listed faults if there is none.
        """
        pairs = self._pairs if fault_weight >= 4 else None
        keys, members = self._singles if pairs is None else pairs
        batch_members = np.empty((BATCH_SIZE, fault_weight), dtype=np.int64)
        batch_syndromes = np.empty((BATCH_SIZE, self._num_syndrome_words), dtype=np.uint64)
        batch_positions = np.empty(BATCH_SIZE, dtype=np.int64)
        num_listed = 0
        for batch in iter_batches(
            self._rows,
            self._num_flag_words,
            self._num_syndrome_words,
            keys,
            members,
            self._num_starts,
            self._css_weight,
            batch_members,
            batch_syndromes,
            batch_positions,
        ):
            count, num_listed = batch
            faults = batch_members[:count].tolist()
            syndromes = [
                int.from_bytes(row.tobytes(), "little")
                for row in batch_syndromes[:count].astype("<u8")
            ]
            positions = batch_positions[:count].tolist()
            for fault, syndrome, position in zip(faults, syndromes, positions, strict=True):
                result = check(tuple(fault), syndrome)
                if result is not None:
                    return result, position + 1
        return None, num_listed

    @functools.cached_property
    def _pairs(self) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32]] | None:
        """The table of pairs of primitive faults, if it fits within the memory limit."""
        num_primitives = len(self._rows)
        if num_primitives * (num_primitives - 1) // 2 * BYTES_PER_PAIR > self._memory_limit:
            return None
        return get_closing_table(self._rows[:, 0], 2)


def get_closing_table(
    hashes: npt.NDArray[np.uint64], size: int
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32]]:
    """List the sets of ``size`` primitive faults, one or two, sorted by the hash of their flags.

    The sets are listed in lexicographic order and sorted stably, so the sets with each hash stay in
    that order.  Return the sorted hashes and the sets.
    """
    num_primitives = len(hashes)
    if size == 1:
        members = np.arange(num_primitives, dtype=np.int32)[:, None]
    else:
        members = np.empty((num_primitives * (num_primitives - 1) // 2, 2), dtype=np.int32)
        members[:, 0], members[:, 1] = np.triu_indices(num_primitives, 1)
    keys = np.bitwise_xor.reduce(hashes[members], axis=1)
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    return keys, members[order]


def hash_words(words: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Hash each row of words to 64 bits with a fixed random linear map.

    The hash of a sum of rows is the sum of their hashes.  Rows that differ can share a hash.
    """
    images = np.random.default_rng(0).integers(2**64, size=(words.shape[1], 64), dtype=np.uint64)
    hashes = np.zeros(len(words), dtype=np.uint64)
    for column, column_images in enumerate(images):
        bits = (words[:, column, None] >> np.arange(64, dtype=np.uint64)) & np.uint64(1)
        hashes ^= np.bitwise_xor.reduce(bits * column_images, axis=1)
    return hashes


def get_num_words(values: Collection[int]) -> int:
    """The number of 64-bit words that hold each of these integers, at least one."""
    return max(1, (max((value.bit_length() for value in values), default=0) + 63) // 64)


def to_words(values: Sequence[int], num_words: int) -> npt.NDArray[np.uint64]:
    """Split each integer into words, with word ``j`` holding bits ``64 j`` and up."""
    data = b"".join(value.to_bytes(8 * num_words, "little") for value in values)
    words = np.frombuffer(data, dtype="<u8").reshape(len(values), num_words)
    return words.astype(np.uint64)


####################################################################################################
# compiled listing


@always_compiled
def iter_batches(
    rows: npt.NDArray[np.uint64],
    num_flag_words: int,
    num_syndrome_words: int,
    keys: npt.NDArray[np.uint64],
    members: npt.NDArray[np.int32],
    num_starts: int,
    css_weight: bool,
    batch_members: npt.NDArray[np.int64],
    batch_syndromes: npt.NDArray[np.uint64],
    batch_positions: npt.NDArray[np.int64],
) -> Iterator[tuple[int, int]]:
    """List faults into batches, as ``FaultLister.find_first`` describes.

    The rows describe the primitive faults, as in ``FaultLister``, and the keys and members are a
    table from ``get_closing_table``.  Each listed fault that needs a check gets a row of the batch
    arrays: its primitive faults, its syndrome, and its position in the listing.  Yield the number
    of faults in the batch and the number of faults listed so far each time that the batch is full,
    and once at the end.  The caller must read a batch before resuming.
    """
    # Columns 0 and 1 of the rows hold the hashes of the flags and of the syndrome.
    num_primitives = len(rows)
    fault_weight = batch_members.shape[1]
    num_closing = members.shape[1]
    depth = fault_weight - num_closing
    flag_start = 2
    syndrome_start = flag_start + num_flag_words
    x_start = syndrome_start + num_syndrome_words
    z_start = x_start + (rows.shape[1] - x_start) // 2

    # The loops choose the first ``depth`` primitive faults, and ``candidates`` holds the next
    # candidate for each, one past the current choice.  Row ``level`` of ``sums`` is the sum of the
    # rows of the first ``level`` choices, and their reduced flags form the basis.
    candidates = np.zeros(depth, dtype=np.int64)
    sums = np.zeros((depth + 1, rows.shape[1]), dtype=np.uint64)
    basis = np.zeros((depth, num_flag_words), dtype=np.uint64)
    pivots = np.zeros(depth, dtype=np.int64)
    fault = np.empty(rows.shape[1], dtype=np.uint64)
    reduced = np.empty(num_flag_words, dtype=np.uint64)

    # A hash table of the syndromes in the batch, holding their rows or -1 in each slot.
    slots = np.full(2 * len(batch_positions), -1, dtype=np.int64)
    count = num_listed = 0

    level = 0
    while level >= 0:
        choice = candidates[level]
        if choice == (num_starts if level == 0 else num_primitives):
            level -= 1
            continue
        candidates[level] = choice + 1
        pivot = reduce_flags(
            rows[choice, flag_start:syndrome_start], basis[:level], pivots, reduced
        )
        if pivot < 0:
            continue
        basis[level] = reduced
        pivots[level] = pivot
        for word in range(rows.shape[1]):
            sums[level + 1, word] = sums[level, word] ^ rows[choice, word]
        if level + 1 < depth:
            level += 1
            candidates[level] = choice + 1
            continue

        # Look up the closing primitive faults by the hash of their flags, which must be that of the
        # chosen ones.  Keep those that come after the last choice, whose flags sum to exactly those
        # of the chosen ones, and of which all but the last keep the flags independent.
        low = np.searchsorted(keys, sums[depth, 0], "left")
        high = np.searchsorted(keys, sums[depth, 0], "right")
        for entry in range(low, high):
            if members[entry, 0] <= choice:
                continue
            fault[:] = sums[depth]
            for member in members[entry]:
                for word in range(rows.shape[1]):
                    fault[word] ^= rows[member, word]
            if fault[flag_start:syndrome_start].any():
                continue
            first_flags = rows[members[entry, 0], flag_start:syndrome_start]
            if num_closing == 2 and reduce_flags(first_flags, basis, pivots, reduced) < 0:
                continue
            num_listed += 1

            # The product of the representatives is an error with the fault's syndrome, so if its
            # Pauli weight is at most the fault weight, the fault satisfies the bound.
            weight = pauli_weight(fault[x_start:z_start], fault[z_start:], css_weight)
            if weight <= fault_weight:
                continue
            batch_syndromes[count] = fault[syndrome_start:x_start]
            if not add_syndrome(batch_syndromes, count, slots, fault[1]):
                continue
            for index in range(depth):
                batch_members[count, index] = candidates[index] - 1
            batch_members[count, depth:] = members[entry]
            batch_positions[count] = num_listed - 1
            count += 1
            if count == len(batch_positions):
                yield count, num_listed
                count = 0
                slots[:] = -1
    yield count, num_listed


@always_compiled
def reduce_flags(
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
def pauli_weight(
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
def add_syndrome(
    syndromes: npt.NDArray[np.uint64], row: int, slots: npt.NDArray[np.int64], hashed: np.uint64
) -> bool:
    """Add a row of syndromes to a hash table of the earlier rows, unless an equal row is there.

    The hash table has more slots than rows, and finds a row in the first empty or matching slot
    from the slot that its hash picks.  Return whether the row was added.
    """
    slot = np.int64(hashed % np.uint64(len(slots)))
    while slots[slot] >= 0:
        other = syndromes[slots[slot]]
        equal = True
        for word in range(len(other)):
            equal = equal and other[word] == syndromes[row, word]
        if equal:
            return False
        slot = (slot + 1) % len(slots)
    slots[slot] = row
    return True
