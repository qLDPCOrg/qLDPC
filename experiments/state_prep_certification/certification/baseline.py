# SPDX-License-Identifier: Apache-2.0

"""The brute-force baseline of the certification experiment, with the certifier's own oracle.

The baseline measures what the listing of irreducible accepted faults saves over checking every
accepted fault combination with the same oracle.  ``brute_force_smallest_violation`` takes the
arguments of qLDPC's ``find_smallest_violation`` (``qldpc.circuits.certification._fault_search``)
and returns its result, a ``Violation`` of the smallest fault weight or None.  Like it, the
baseline keeps one mechanism per distinct nonzero (flags, syndrome) signature, the N_0 reduced
mechanisms, and builds its oracle as ``CosetLeaderSearch(stabilizers, css_weight=css_weight,
method=method)``, by the name that the fault search uses.  Its search uses no theory of the
certifier, such as listing, the classification of heavy mechanisms, the cutoff K', or skipping:

* for each fault weight m = 1, ..., K, it enumerates every accepted combination of weight m, a
  set of m reduced mechanisms whose flags sum to zero.  A compiled depth-first search runs over
  increasing tuples of m - 1 mechanisms with their running sums, and finds each m-th mechanism by
  a lookup of the later mechanisms whose flags equal the prefix's sum;
* it submits the packed syndromes of the combinations, in enumeration order, to
  ``oracle.find_first_not_within_max_weight(rows, m)`` in batches of ``batch_size`` rows;
* it stops at the first weight with a violation, as the certifier does, and returns the first
  violating combination of that weight.

``BruteForce.install`` replaces ``fault_propagation.find_smallest_violation`` by the baseline in
this process, so that ``certify_bounded_fault_propagation`` builds the DEM and the oracle as for
``cert_bz``, and records for each weight m the combinations enumerated, the rows queried, the
oracle calls and their time, and the number of distinct syndromes among the rows.  The queries
are not deduplicated: the distinct syndromes are a count only, kept in a compiled hash set of at
most ``max_distinct`` syndromes per weight, beyond which the count is None.  The count's time is
part of T, and each weight records it as ``distinct_seconds``.  By default, ``batch_size`` and
``max_distinct`` are the values of ``brute_force`` in ``config.json``.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numba
import numpy as np
import numpy.typing as npt

from experiments.state_prep_certification.certification.counting import (
    pack_words,
    reduce_signatures,
)
from qldpc.circuits.certification import _fault_search, fault_propagation
from qldpc.circuits.certification._fault_search import Violation
from qldpc.codes.distance import CosetMethod

CONFIG = json.loads((pathlib.Path(__file__).resolve().parent / "config.json").read_text())

# The number of rows in each batch of oracle queries.
BATCH_SIZE = int(CONFIG["brute_force"]["batch_size"])

# The most distinct syndromes per weight that the hash set counts.
MAX_DISTINCT = int(CONFIG["brute_force"]["max_distinct_syndromes"])

# The number of slots of the hash set of distinct syndromes at the start of each weight.
_INITIAL_SLOTS = 2**12

# An odd 64-bit constant, near 2**64 divided by the golden ratio, that mixes the words of a hash.
_HASH_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)

# Compiled with numba.njit(cache=False): qLDPC's always_compiled keys its cache to qLDPC's source.


####################################################################################################
# brute force


class BruteForce:
    """The brute-force search, with its counters, for one certification.

    Attributes:
        arguments: The arguments of the last search, by name.
        search_seconds: The time of the last search, T_search.
        oracle_setup_seconds: The time to construct the oracle of the last search.
        weights: Per-weight records: m, T_m, the combinations enumerated, the rows queried, the
            oracle calls and their time, the distinct syndromes (None beyond ``max_distinct``),
            whether the weight has a violation, and the cumulative T of the runner's clock.
    """

    def __init__(
        self,
        *,
        batch_size: int = BATCH_SIZE,
        max_distinct: int = MAX_DISTINCT,
        progress: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float | None] | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.max_distinct = max_distinct
        self.progress = progress
        self.clock = clock
        self.arguments: dict[str, Any] | None = None
        self.search_seconds: float | None = None
        self.oracle_setup_seconds: float | None = None
        self.weights: list[dict[str, Any]] = []

    def summary(self) -> dict[str, Any]:
        """The recorded fields, for a result row."""
        return {
            "T_search": self.search_seconds,
            "oracle_setup_seconds": self.oracle_setup_seconds,
            "per_weight": self.weights,
            "rows": sum(record["rows"] for record in self.weights),
            "batch_size": self.batch_size,
        }

    @contextmanager
    def install(self) -> Iterator[BruteForce]:
        """Replace ``fault_propagation.find_smallest_violation`` for the duration of the block."""
        original = fault_propagation.find_smallest_violation
        fault_propagation.find_smallest_violation = self.find_smallest_violation
        try:
            yield self
        finally:
            fault_propagation.find_smallest_violation = original

    def find_smallest_violation(
        self,
        flags: Sequence[int],
        syndromes: Sequence[int],
        stabilizers: npt.ArrayLike,
        max_weight: int | None,
        css_weight: bool,
        memory_limit: int = 2**30,
        method: CosetMethod = "auto",
    ) -> Violation | None:
        """Find a violating accepted fault of the smallest fault weight by brute force, or None.

        The arguments and the result are those of qLDPC's ``find_smallest_violation``.  The
        search checks the weights 1, ..., ``max_weight``, or up to one more than the rank of the
        flags if ``max_weight`` is None.  ``memory_limit`` bounds qLDPC's table of syndromes,
        which the baseline does not use.
        """
        self.arguments = {
            "flags": list(flags),
            "syndromes": list(syndromes),
            "stabilizers": np.array(stabilizers, dtype=np.uint8),
            "max_weight": max_weight,
            "css_weight": css_weight,
            "memory_limit": memory_limit,
            "method": method,
        }
        self.weights = []
        start = time.perf_counter()
        try:
            return self._search(flags, syndromes, stabilizers, max_weight, css_weight, method)
        finally:
            self.search_seconds = time.perf_counter() - start

    def _search(
        self,
        flags: Sequence[int],
        syndromes: Sequence[int],
        stabilizers: npt.ArrayLike,
        max_weight: int | None,
        css_weight: bool,
        method: CosetMethod,
    ) -> Violation | None:
        """Reduce the signatures, build the oracle, and check each weight in turn."""
        reduced = reduce_signatures(flags, syndromes)
        if max_weight is None:
            max_weight = reduced.rank + 1

        start = time.perf_counter()
        oracle = _fault_search.CosetLeaderSearch(stabilizers, css_weight=css_weight, method=method)
        self.oracle_setup_seconds = time.perf_counter() - start

        flag_words = pack_words(reduced.flags)
        syndrome_words = pack_words(reduced.syndromes, (oracle.num_qubits + 63) // 64)
        order = np.lexsort(
            (
                np.arange(reduced.num_reduced),
                *(flag_words[:, word] for word in range(flag_words.shape[1])),
            )
        ).astype(np.int64)
        sorted_flags = np.ascontiguousarray(flag_words[order])

        for m in range(1, max_weight + 1):
            violation = self._check_weight(
                m, flag_words, syndrome_words, sorted_flags, order, oracle
            )
            if violation is not None:
                positions, row, bound = violation
                syndrome = int.from_bytes(row.astype("<u8").tobytes(), "little")
                fault_indices = tuple(sorted(reduced.indices[position] for position in positions))
                return Violation(fault_indices, syndrome, bound)
        return None

    def _check_weight(
        self,
        m: int,
        flag_words: npt.NDArray[np.uint64],
        syndrome_words: npt.NDArray[np.uint64],
        sorted_flags: npt.NDArray[np.uint64],
        order: npt.NDArray[np.int64],
        oracle: Any,
    ) -> tuple[list[int], npt.NDArray[np.uint64], Any] | None:
        """Enumerate and query the accepted combinations of weight m; return the first violation."""
        start = time.perf_counter()
        state = _new_state(m, flag_words.shape[1], syndrome_words.shape[1])
        combinations = np.zeros((self.batch_size, m), dtype=np.int64)
        rows = np.zeros((self.batch_size, syndrome_words.shape[1]), dtype=np.uint64)
        distinct = _DistinctSyndromes(syndrome_words.shape[1], self.max_distinct)
        record: dict[str, Any] = {
            "m": m,
            "combinations": 0,
            "rows": 0,
            "calls": 0,
            "oracle_seconds": 0.0,
            "distinct_seconds": 0.0,
        }
        violation = None
        while True:
            written, done = _enumerate_combinations(
                flag_words, syndrome_words, sorted_flags, order, combinations, rows, *state
            )
            if written:
                batch = rows[:written]
                record["combinations"] += written
                count_start = time.perf_counter()
                distinct.add(batch)
                record["distinct_seconds"] += time.perf_counter() - count_start
                query_start = time.perf_counter()
                index, bound = oracle.find_first_not_within_max_weight(batch, m)
                record["oracle_seconds"] += time.perf_counter() - query_start
                record["calls"] += 1
                record["rows"] += written
                if bound is not None:
                    violation = ([int(i) for i in combinations[index]], batch[index].copy(), bound)
                    break
            if done:
                break
        record |= {
            "distinct_syndromes": distinct.count,
            "violation": violation is not None,
            "T_m": time.perf_counter() - start,
            "T_cumulative": self.clock() if self.clock else None,
        }
        self.weights.append(record)
        if self.progress is not None:
            self.progress(record)
        return violation


def brute_force_smallest_violation(
    flags: Sequence[int],
    syndromes: Sequence[int],
    stabilizers: npt.ArrayLike,
    max_weight: int | None,
    css_weight: bool,
    memory_limit: int = 2**30,
    method: CosetMethod = "auto",
) -> Violation | None:
    """Find a violating accepted fault of the smallest fault weight by brute force, or None.

    The arguments and the result are those of qLDPC's ``find_smallest_violation``; see
    ``BruteForce.find_smallest_violation``.
    """
    return BruteForce().find_smallest_violation(
        flags, syndromes, stabilizers, max_weight, css_weight, memory_limit, method=method
    )


def enumeration_size(num_reduced: int, max_weight: int) -> int:
    """U, the sum over 1 <= m <= K of C(N_0, m), which bounds the combinations and rows."""
    return sum(math.comb(num_reduced, m) for m in range(1, max_weight + 1))


def _new_state(m: int, num_flag_words: int, num_syndrome_words: int) -> tuple[Any, ...]:
    """The resumable state of ``_enumerate_combinations`` for weight m.

    It holds the depth and the closing position (-1 outside a closing scan), the next candidate at
    each depth, the chosen prefix, and the running sums of flags and syndromes.
    """
    prefix_length = m - 1
    counters = np.array([0, -1, 0], dtype=np.int64)  # depth, closing position, done
    next_candidate = np.zeros(prefix_length + 1, dtype=np.int64)
    chosen = np.zeros(prefix_length + 1, dtype=np.int64)
    flag_sums = np.zeros((prefix_length + 1, num_flag_words), dtype=np.uint64)
    syndrome_sums = np.zeros((prefix_length + 1, num_syndrome_words), dtype=np.uint64)
    return counters, next_candidate, chosen, flag_sums, syndrome_sums


@numba.njit(cache=False)
def _compare_key(
    sorted_flags: npt.NDArray[np.uint64],
    order: npt.NDArray[np.int64],
    position: int,
    key: npt.NDArray[np.uint64],
    key_index: int,
) -> int:
    """-1, 0, or 1 as the mechanism at a sorted position is below, at, or above (key, index)."""
    for word in range(key.shape[0] - 1, -1, -1):
        if sorted_flags[position, word] != key[word]:
            return -1 if sorted_flags[position, word] < key[word] else 1
    if order[position] != key_index:
        return -1 if order[position] < key_index else 1
    return 0


@numba.njit(cache=False)
def _enumerate_combinations(
    flag_words: npt.NDArray[np.uint64],
    syndrome_words: npt.NDArray[np.uint64],
    sorted_flags: npt.NDArray[np.uint64],
    order: npt.NDArray[np.int64],
    combinations: npt.NDArray[np.int64],
    rows: npt.NDArray[np.uint64],
    counters: npt.NDArray[np.int64],
    next_candidate: npt.NDArray[np.int64],
    chosen: npt.NDArray[np.int64],
    flag_sums: npt.NDArray[np.uint64],
    syndrome_sums: npt.NDArray[np.uint64],
) -> tuple[int, bool]:
    """Write the next accepted combinations of weight m and their syndromes to the buffers.

    The weight m is the number of columns of ``combinations``.  The search is depth first over
    increasing prefixes of m - 1 mechanisms, with an explicit stack kept in the state arrays, so
    that a call that fills the buffers returns and the next call resumes.  For each prefix, a
    closing scan writes every later mechanism whose flags equal the prefix's sum: the mechanisms
    sorted by (flags, index) are searched for the first at or above (sum, last index + 1).  Return
    the combinations written and whether the enumeration is done.
    """
    num_mechanisms = flag_words.shape[0]
    m = combinations.shape[1]
    prefix_length = m - 1
    written = 0
    depth = counters[0]
    closing = counters[1]
    if counters[2]:
        return 0, True
    if prefix_length == 0 and closing < 0 and next_candidate[0] == 0:
        # weight 1: the closing scan of the empty prefix, from the first mechanism of zero flags
        next_candidate[0] = 1
        closing = _lower_bound(sorted_flags, order, flag_sums[0], 0)
    while written < combinations.shape[0]:
        if closing >= 0:
            # the closing scan of the prefix of length ``prefix_length``
            if closing < num_mechanisms:
                matches = True
                for word in range(flag_words.shape[1]):
                    if sorted_flags[closing, word] != flag_sums[prefix_length, word]:
                        matches = False
                        break
                if matches:
                    last = order[closing]
                    for level in range(prefix_length):
                        combinations[written, level] = chosen[level]
                    combinations[written, prefix_length] = last
                    for word in range(syndrome_words.shape[1]):
                        rows[written, word] = (
                            syndrome_sums[prefix_length, word] ^ syndrome_words[last, word]
                        )
                    written += 1
                    closing += 1
                    continue
            closing = -1
            if prefix_length == 0:
                counters[2] = 1
                break
            depth = prefix_length - 1
            continue

        # the next candidate at this depth, or back up a level
        candidate = next_candidate[depth]
        if candidate > num_mechanisms - m + depth:
            if depth == 0:
                counters[2] = 1
                break
            depth -= 1
            continue
        next_candidate[depth] = candidate + 1
        chosen[depth] = candidate
        for word in range(flag_words.shape[1]):
            flag_sums[depth + 1, word] = flag_sums[depth, word] ^ flag_words[candidate, word]
        for word in range(syndrome_words.shape[1]):
            syndrome_sums[depth + 1, word] = (
                syndrome_sums[depth, word] ^ syndrome_words[candidate, word]
            )
        depth += 1
        if depth < prefix_length:
            next_candidate[depth] = candidate + 1
            continue
        closing = _lower_bound(sorted_flags, order, flag_sums[prefix_length], candidate + 1)
    counters[0] = depth
    counters[1] = closing
    return written, bool(counters[2])


@numba.njit(cache=False)
def _lower_bound(
    sorted_flags: npt.NDArray[np.uint64],
    order: npt.NDArray[np.int64],
    key: npt.NDArray[np.uint64],
    key_index: int,
) -> int:
    """The first sorted position whose (flags, index) is at or above (key, key_index)."""
    low, high = 0, sorted_flags.shape[0]
    while low < high:
        middle = (low + high) // 2
        if _compare_key(sorted_flags, order, middle, key, key_index) < 0:
            low = middle + 1
        else:
            high = middle
    return low


####################################################################################################
# distinct syndromes


class _DistinctSyndromes:
    """A count of distinct packed syndromes, in a hash set of at most ``max_count`` syndromes.

    The count is None once the set would exceed ``max_count``.
    """

    def __init__(self, num_words: int, max_count: int) -> None:
        self.max_count = max_count
        self.count: int | None = 0
        self._keys = np.zeros((_INITIAL_SLOTS, num_words), dtype=np.uint64)
        self._used = np.zeros(_INITIAL_SLOTS, dtype=np.bool_)

    def add(self, rows: npt.NDArray[np.uint64]) -> None:
        """Add a batch of packed syndromes."""
        if self.count is None:
            return
        while 2 * (self.count + len(rows)) > len(self._used):
            if len(self._used) >= 2 * self.max_count:
                break
            self._grow()
        count = _insert_rows(rows, self._keys, self._used, self.count, self.max_count)
        if count < 0:
            self.count = None
            self._keys = self._keys[:0]
            self._used = self._used[:0]
        else:
            self.count = count

    def _grow(self) -> None:
        """Double the slots and reinsert the syndromes."""
        keys, used = self._keys[self._used], len(self._used) * 2
        self._keys = np.zeros((used, keys.shape[1]), dtype=np.uint64)
        self._used = np.zeros(used, dtype=np.bool_)
        _insert_rows(keys, self._keys, self._used, 0, used)


@numba.njit(cache=False)
def _insert_rows(
    rows: npt.NDArray[np.uint64],
    keys: npt.NDArray[np.uint64],
    used: npt.NDArray[np.bool_],
    count: int,
    max_count: int,
) -> int:
    """Insert rows into an open-addressing hash set; return the new count, or -1 beyond the cap.

    The set must have more slots than the syndromes that it will hold, a power of two of them.
    """
    mask = np.uint64(used.shape[0] - 1)
    for row in range(rows.shape[0]):
        value = np.uint64(0)
        for word in range(rows.shape[1]):
            value = (value ^ rows[row, word]) * _HASH_MULTIPLIER
            value ^= value >> np.uint64(29)
        slot = value & mask
        while True:
            if not used[slot]:
                if count >= max_count or count + 1 >= used.shape[0]:
                    return -1
                used[slot] = True
                for word in range(rows.shape[1]):
                    keys[slot, word] = rows[row, word]
                count += 1
                break
            equal = True
            for word in range(rows.shape[1]):
                if keys[slot, word] != rows[row, word]:
                    equal = False
                    break
            if equal:
                break
            slot = (slot + np.uint64(1)) & mask
    return count
