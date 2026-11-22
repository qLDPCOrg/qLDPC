# SPDX-License-Identifier: Apache-2.0

"""Deterministic counts of the certification experiment, and brute force for small circuits.

These counts describe the listing of irreducible accepted faults in ``fault_propagation.md`` of
``qldpc.circuits.certification``, independently of qLDPC's data structures:

* ``reduce_signatures`` keeps one mechanism per distinct nonzero (flags, syndrome) signature of
  the DEM, in DEM order, as qLDPC does: N_0 mechanisms, z of them with zero flags.
* ``listing_counts`` counts, for each weight, the class codewords E_m whose first class is a
  start class, the listed faults Phi_m, and the prefixes Pi_l of the naive enumeration of these
  class codewords: the increasing tuples of l classes with independent flags whose first class is
  a start class.  ``fault_propagation.md`` contrasts this enumeration with qLDPC's walk by shared
  flags, which visits far fewer sets of classes, so Pi_l counts the work of the naive enumeration,
  not of qLDPC's listing.  Classes are the flagged mechanisms with equal flags, numbered as qLDPC
  numbers them: heavy mechanisms first, then the rest, each in DEM order, and classes in the order
  of their first members, so that start classes (those with a heavy member) come first.  Phi_2 is
  a sum over start classes, and Phi_m for m >= 3 a sum over class codewords of
  prod(s_c) - prod(s_c - h_c), with s_c members and h_c heavy members in class c.  The naive
  enumeration, a compiled depth-first search, emits the class codewords, and Python integers hold
  the products and totals, which can exceed 64 bits.
* ``accepted_counts`` counts the accepted fault combinations A_m of weight m <= K (sets of
  reduced mechanisms whose flags sum to zero) by dynamic programming over classes, with C(N_0, m)
  beside them.  It counts up to K, not K', as the brute-force baseline (``baseline.py``)
  enumerates up to K; for K' < m <= K nothing is listed, and Phi_m = 0.
* ``ground_truth`` enumerates every accepted fault of weight at most K of a small reduced DEM, and
  gives the smallest violating fault weight, A by its definition up to K, and Pi, E, and Phi by
  their definitions up to K'.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import stim

from qldpc._compilation import always_compiled

# The number of class codewords that the compiled search returns at a time.
_BUFFER = 4096

# The compiled functions of this file and of baseline.py are cached on disk by qLDPC's
# always_compiled, in a directory named by a hash of qLDPC's source files.  numba loads a function's
# cached code unless the function's own source file has changed, and freezes the values of the
# globals that the function reads.  Neither the hash nor numba's check covers the experiment's other
# files or config.json.  So each of these functions calls only functions of its own file and NumPy,
# and reads only NumPy and globals that its file's text fixes, such as literal constants; otherwise,
# a change elsewhere would leave stale code or values in the cache.  AT2 checks that they call by
# name no compiled function of another file.


####################################################################################################
# signatures


def dem_signatures(
    dem: stim.DetectorErrorModel, selected: Sequence[int], num_stabilizers: int
) -> tuple[list[int], list[int]]:
    """The flags and syndrome of each error mechanism of a certificate's DEM, in DEM order.

    Flag bit j is the j-th post-selection detector, and syndrome bit j is the detector of
    stabilizer j, which are the last ``num_stabilizers`` detectors of the DEM.
    """
    flag_bit = {detector: bit for bit, detector in enumerate(selected)}
    first_stabilizer = dem.num_detectors - num_stabilizers
    flags, syndromes = [], []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        flag = syndrome = 0
        for target in instruction.targets_copy():
            if not target.is_relative_detector_id():
                continue
            detector = target.val
            if detector in flag_bit:
                flag ^= 1 << flag_bit[detector]
            if detector >= first_stabilizer:
                syndrome ^= 1 << (detector - first_stabilizer)
        flags.append(flag)
        syndromes.append(syndrome)
    return flags, syndromes


@dataclass
class ReducedDem:
    """One mechanism per distinct nonzero signature, in DEM order of first occurrence.

    Attributes:
        num_mechanisms: N, the error mechanisms of the DEM.
        flags: The flags of each reduced mechanism.
        syndromes: The syndrome of each reduced mechanism.
        indices: The DEM index of each reduced mechanism's first occurrence.
        rank: r, the GF(2) rank of the flags.
    """

    num_mechanisms: int
    flags: list[int]
    syndromes: list[int]
    indices: list[int]
    rank: int

    @property
    def num_reduced(self) -> int:
        """N_0."""
        return len(self.flags)

    @property
    def num_zero_flag(self) -> int:
        """z, the reduced mechanisms with zero flags."""
        return sum(flag == 0 for flag in self.flags)

    def max_fault_weight(self, max_weight: int) -> int:
        """K' = min(K, r + 1)."""
        return min(max_weight, self.rank + 1)


def reduce_signatures(flags: Sequence[int], syndromes: Sequence[int]) -> ReducedDem:
    """Keep the first mechanism of each distinct nonzero (flags, syndrome) signature."""
    first: dict[tuple[int, int], int] = {}
    for index, signature in enumerate(zip(flags, syndromes, strict=True)):
        if any(signature):
            first.setdefault(signature, index)
    kept = list(first.items())
    return ReducedDem(
        num_mechanisms=len(flags),
        flags=[flag for (flag, _), _ in kept],
        syndromes=[syndrome for (_, syndrome), _ in kept],
        indices=[index for _, index in kept],
        rank=gf2_rank(flag for (flag, _), _ in kept),
    )


def gf2_rank(vectors: Any) -> int:
    """The GF(2) rank of integers read as bit vectors."""
    pivots: dict[int, int] = {}
    for vector in vectors:
        while vector:
            top = vector.bit_length() - 1
            if top not in pivots:
                pivots[top] = vector
                break
            vector ^= pivots[top]
    return len(pivots)


####################################################################################################
# classes and listing counts


@dataclass
class Classes:
    """The flag classes of the flagged mechanisms, in qLDPC's order.

    Attributes:
        flags: The flags of each class.
        sizes: s_c, the members of each class.
        heavy: h_c, the heavy members of each class.
        num_start: q_h, the start classes, which come first.
    """

    flags: list[int]
    sizes: list[int]
    heavy: list[int]
    num_start: int


def get_classes(flags: Sequence[int], heavy: Sequence[bool]) -> Classes:
    """Group the flagged mechanisms with equal flags, heavy members first, as qLDPC numbers them.

    Args:
        flags: The flags of each flagged reduced mechanism, in DEM order, all nonzero.
        heavy: Whether each of them is heavy.
    """
    order = [i for i in range(len(flags)) if heavy[i]] + [
        i for i in range(len(flags)) if not heavy[i]
    ]
    members: dict[int, list[int]] = {}
    for i in order:
        members.setdefault(flags[i], []).append(i)
    class_flags = list(members)
    return Classes(
        flags=class_flags,
        sizes=[len(members[flag]) for flag in class_flags],
        heavy=[sum(heavy[i] for i in members[flag]) for flag in class_flags],
        num_start=sum(heavy[members[flag][0]] for flag in class_flags),
    )


def listing_counts(
    classes: Classes,
    max_fault_weight: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
    buffer_size: int = _BUFFER,
) -> dict[str, Any]:
    """Pi_l for l < K', and E_m and Phi_m for 2 <= m <= K', weight by weight.

    Pi_l counts the prefixes of the naive enumeration of class codewords (see the module docstring),
    which this function runs.  The search for each weight m >= 3 visits the prefixes of length
    m - 1 from scratch, so that a weight's counts are final once it is reported to ``progress``.
    Counts are Python integers, keyed by weight in JSON-friendly strings.

    Args:
        classes: The flag classes of the flagged mechanisms.
        max_fault_weight: K'.
        progress: A function that receives the counts of each weight as it completes.
        buffer_size: The number of class codewords that the compiled search returns at a time.
    """
    q = len(classes.flags)
    words = pack_words(classes.flags)
    sizes = np.array(classes.sizes, dtype=object)
    lights = np.array([s - h for s, h in zip(classes.sizes, classes.heavy)], dtype=object)
    order = np.lexsort(words.T) if q else np.zeros(0, dtype=np.int64)
    sorted_words = words[order]
    counts: dict[str, Any] = {"Pi": {}, "E": {}, "Phi": {}}

    # weight 2: two members of one start class, at least one of them heavy
    if max_fault_weight >= 2:
        counts["Pi"]["1"] = classes.num_start
        counts["E"]["2"] = 0
        counts["Phi"]["2"] = sum(
            math.comb(s, 2) - math.comb(s - h, 2)
            for s, h in zip(classes.sizes[: classes.num_start], classes.heavy[: classes.num_start])
        )
        if progress is not None:
            progress({"m": 2, "Pi": counts["Pi"]["1"], "E": 0, "Phi": counts["Phi"]["2"]})

    # weights 3 and more: class codewords from prefixes of length m - 1
    for m in range(3, max_fault_weight + 1):
        prefix_length = m - 1
        state = _new_state(prefix_length, words.shape[1])
        buffer = np.zeros((buffer_size, m), dtype=np.int64)
        visited = codewords = 0
        phi = 0
        while True:
            written, done, visited_now = _search_codewords(
                words, sorted_words, order, classes.num_start, prefix_length, buffer, *state
            )
            visited += visited_now
            codewords += written
            if written:
                chosen = buffer[:written]
                phi += int(np.sum(np.prod(sizes[chosen], axis=1) - np.prod(lights[chosen], axis=1)))
            if done:
                break
        counts["Pi"][str(prefix_length)] = visited
        counts["E"][str(m)] = codewords
        counts["Phi"][str(m)] = phi
        if progress is not None:
            progress({"m": m, "Pi": visited, "E": codewords, "Phi": phi})
    return counts


def pack_words(values: Sequence[int], num_words: int | None = None) -> npt.NDArray[np.uint64]:
    """Integers as rows of 64-bit words, with word ``j`` holding bits ``64 j`` and up.

    Args:
        values: Nonnegative integers, such as flags or syndromes.
        num_words: The words per row; by default the fewest that hold every value, at least one.
    """
    if num_words is None:
        num_words = max(1, (max((value.bit_length() for value in values), default=0) + 63) // 64)
    data = b"".join(value.to_bytes(8 * num_words, "little") for value in values)
    return np.frombuffer(data, dtype="<u8").reshape(len(values), num_words).astype(np.uint64)


def _new_state(prefix_length: int, num_words: int) -> tuple[npt.NDArray[Any], ...]:
    """The resumable state of ``_search_codewords``: depth, next candidates, basis, sums."""
    depth = np.zeros(1, dtype=np.int64)
    next_class = np.zeros(prefix_length + 1, dtype=np.int64)
    chosen = np.zeros(prefix_length + 1, dtype=np.int64)
    basis = np.zeros((prefix_length + 1, num_words), dtype=np.uint64)
    pivot_word = np.zeros(prefix_length + 1, dtype=np.int64)
    pivot_bit = np.zeros(prefix_length + 1, dtype=np.uint64)
    sums = np.zeros((prefix_length + 2, num_words), dtype=np.uint64)
    return depth, next_class, chosen, basis, pivot_word, pivot_bit, sums


@always_compiled
def _search_codewords(
    words: npt.NDArray[np.uint64],
    sorted_words: npt.NDArray[np.uint64],
    order: npt.NDArray[np.int64],
    num_start: int,
    prefix_length: int,
    buffer: npt.NDArray[np.int64],
    depth_box: npt.NDArray[np.int64],
    next_class: npt.NDArray[np.int64],
    chosen: npt.NDArray[np.int64],
    basis: npt.NDArray[np.uint64],
    pivot_word: npt.NDArray[np.int64],
    pivot_bit: npt.NDArray[np.uint64],
    sums: npt.NDArray[np.uint64],
) -> tuple[int, bool, int]:
    """Visit prefixes of ``prefix_length`` classes and write their class codewords to the buffer.

    A prefix is an increasing tuple of classes with independent flags whose first class is a start
    class.  Its codeword closes it with the later class whose flags equal the prefix's sum, if any.
    The search is depth first with an explicit stack, kept in the state arrays, so that a call that
    fills the buffer returns and the next call resumes.  Return the codewords written, whether the
    search is done, and the prefixes of full length visited in this call.
    """
    num_classes, num_words = words.shape
    written = 0
    visited = 0
    depth = depth_box[0]
    residual = np.zeros(num_words, dtype=np.uint64)
    while written < buffer.shape[0]:
        # the next candidate at this depth, or back up a level
        limit = num_start if depth == 0 else num_classes
        candidate = next_class[depth]
        if candidate >= limit:
            if depth == 0:
                depth_box[0] = 0
                return written, True, visited
            depth -= 1
            continue
        next_class[depth] = candidate + 1

        # reduce the candidate's flags by the basis of the prefix; dependent flags are skipped
        for word in range(num_words):
            residual[word] = words[candidate, word]
        for level in range(depth):
            if residual[pivot_word[level]] & pivot_bit[level]:
                for word in range(num_words):
                    residual[word] ^= basis[level, word]
        nonzero = -1
        for word in range(num_words):
            if residual[word]:
                nonzero = word
                break
        if nonzero < 0:
            continue

        # extend the prefix
        chosen[depth] = candidate
        for word in range(num_words):
            basis[depth, word] = residual[word]
            sums[depth + 1, word] = sums[depth, word] ^ words[candidate, word]
        lowest = residual[nonzero] & (~residual[nonzero] + np.uint64(1))
        pivot_word[depth] = nonzero
        pivot_bit[depth] = lowest
        depth += 1
        if depth < prefix_length:
            next_class[depth] = candidate + 1
            continue

        # a full prefix: count it, and look up its closing class among the later classes
        visited += 1
        if prefix_length >= 2:
            low, high = 0, num_classes
            while low < high:
                middle = (low + high) // 2
                less = False
                for word in range(num_words - 1, -1, -1):
                    a = sorted_words[middle, word]
                    b = sums[depth, word]
                    if a != b:
                        less = a < b
                        break
                if less:
                    low = middle + 1
                else:
                    high = middle
            if low < num_classes:
                equal = True
                for word in range(num_words):
                    if sorted_words[low, word] != sums[depth, word]:
                        equal = False
                        break
                if equal and order[low] > candidate:
                    for level in range(prefix_length):
                        buffer[written, level] = chosen[level]
                    buffer[written, prefix_length] = order[low]
                    written += 1
        depth -= 1
    depth_box[0] = depth
    return written, False, visited


####################################################################################################
# accepted fault combinations


def accepted_counts(flags: Sequence[int], max_weight: int, max_entries: int) -> dict[str, Any]:
    """A_m and C(N_0, m) for 1 <= m <= K, over the reduced mechanisms with these flags.

    The zero-flag mechanisms are free: A_m = sum_j C(z, m - j) B_j, where B_j counts the sets of j
    flagged mechanisms whose flags sum to zero.  B_j comes from a dynamic program over classes: a
    map from flag sums to polynomials of degree at most K.  A class of size s multiplies each
    entry by sum_i C(s, i) x^i, moving the terms of odd i to the sum plus the class's flags.  If
    the map would exceed ``max_entries``, A is not computed and is None.
    """
    total = len(flags)
    zero = sum(flag == 0 for flag in flags)
    sizes: dict[int, int] = {}
    for flag in flags:
        if flag:
            sizes[flag] = sizes.get(flag, 0) + 1
    degree = max_weight
    binomials = {"C": {str(m): math.comb(total, m) for m in range(1, degree + 1)}}

    table: dict[int, list[int]] = {0: [1] + [0] * degree}
    if len(table) > max_entries:
        return {"A": None, **binomials, "dp_entries": len(table)}
    for flag, size in sizes.items():
        even = [math.comb(size, i) if i % 2 == 0 else 0 for i in range(degree + 1)]
        odd = [math.comb(size, i) if i % 2 == 1 else 0 for i in range(degree + 1)]
        updated: dict[int, list[int]] = {}
        for key, poly in table.items():
            for target, factor in ((key, even), (key ^ flag, odd)):
                product = _multiply(poly, factor, degree)
                if any(product):
                    entry = updated.setdefault(target, [0] * (degree + 1))
                    for i, value in enumerate(product):
                        entry[i] += value
            if len(updated) > max_entries:
                return {"A": None, **binomials, "dp_entries": len(updated)}
        table = updated
    flagged = table.get(0, [1] + [0] * degree)
    accepted = {
        str(m): sum(math.comb(zero, m - j) * flagged[j] for j in range(m + 1))
        for m in range(1, degree + 1)
    }
    return {"A": accepted, **binomials, "dp_entries": len(table)}


def _multiply(a: list[int], b: list[int], degree: int) -> list[int]:
    """The product of two polynomials, truncated at the degree."""
    product = [0] * (degree + 1)
    for i, x in enumerate(a):
        if x:
            for j in range(degree + 1 - i):
                if b[j]:
                    product[i + j] += x * b[j]
    return product


####################################################################################################
# brute force


def _compile_galois() -> None:
    """Compile the galois arithmetic of ``residual_weight_table``, untimed, before a measured call.

    galois compiles its GF(2) arithmetic in each process that uses it, and does not cache it on
    disk: creating the field compiles ``numba.jit`` functions, and row reduction compiles
    ``numba.vectorize`` ufuncs.  The runner's retry cannot remove these compilations from a measured
    call, so ``ground_truth`` calls this first.
    """
    import galois

    galois.GF(2)([[1, 1, 0], [0, 1, 1]]).null_space()


def residual_weight_table(stabilizers: npt.NDArray[np.uint8]) -> npt.NDArray[np.int64]:
    """The CSS residual weight of every syndrome of a CSS stabilizer state on at most 20 qubits.

    ``stabilizers`` has one row ``[x | z]`` per generator, and syndrome bit j of an error is whether
    it anticommutes with generator j.  The generators need not be pure X or pure Z: this function
    first finds pure ones, T = G S over GF(2), whose syndrome is G times the original one.  The
    residual weight of a syndrome is then the larger of the least weights of an X error and a Z
    error with its parts on the pure-Z and pure-X generators.
    """
    import galois

    num_qubits = stabilizers.shape[1] // 2
    if num_qubits > 20:
        raise ValueError("Brute force needs at most 20 output qubits")
    gf2 = galois.GF(2)
    matrix = gf2(stabilizers.astype(np.uint8))
    x_part, z_part = matrix[:, :num_qubits], matrix[:, num_qubits:]

    # combinations G of the generators whose products are pure X or pure Z
    pure_x = gf2(z_part.T).null_space()
    pure_z = gf2(x_part.T).null_space()
    combination = np.vstack([np.asarray(pure_x), np.asarray(pure_z)]).astype(np.int64)
    if combination.shape[0] != stabilizers.shape[0]:
        raise ValueError("The state is not CSS")
    pure = (combination @ stabilizers.astype(np.int64)) % 2
    num_x = pure_x.shape[0]

    # least weights of X errors by their syndrome on the pure-Z generators, and of Z errors
    errors = ((np.arange(2**num_qubits)[:, None] >> np.arange(num_qubits)) & 1).astype(np.int64)
    weights = errors.sum(axis=1)
    least = []
    for checks in (pure[num_x:, num_qubits:], pure[:num_x, :num_qubits]):
        keys = ((errors @ checks.T) % 2) @ (1 << np.arange(checks.shape[0], dtype=np.int64))
        table = np.full(2 ** checks.shape[0], np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(table, keys, weights)
        least.append(table)

    # map every original syndrome to its parts on the pure generators
    syndromes = np.arange(2 ** stabilizers.shape[0], dtype=np.int64)
    bits = (syndromes[:, None] >> np.arange(stabilizers.shape[0])) & 1
    new_bits = (bits @ combination.T) % 2
    x_keys = new_bits[:, :num_x] @ (1 << np.arange(num_x, dtype=np.int64))
    z_keys = new_bits[:, num_x:] @ (1 << np.arange(new_bits.shape[1] - num_x, dtype=np.int64))
    return np.maximum(least[0][z_keys], least[1][x_keys])


def ground_truth(
    reduced: ReducedDem, stabilizers: npt.NDArray[np.uint8], max_weight: int
) -> dict[str, Any]:
    """Every accepted fault of weight at most K of a small reduced DEM, by enumeration.

    Return the smallest violating fault weight (or None), whether each flagged mechanism is heavy,
    and A up to K and Pi, E, and Phi up to K' by their definitions (``brute_force_counts``), for
    comparison with the counters and the baseline.  The flags must fit in 64 bits.
    """
    if any(flag >> 64 for flag in reduced.flags):
        raise ValueError("Brute force needs at most 64 post-selection detectors")
    table = residual_weight_table(stabilizers)
    flags = np.array(reduced.flags, dtype=np.uint64)
    syndromes = np.array(reduced.syndromes, dtype=np.int64)

    # the smallest violation, over all accepted faults of the reduced mechanisms
    smallest = None
    for m in range(1, max_weight + 1):
        _, violated = _enumerate_accepted(flags, syndromes, table, m)
        if violated:
            smallest = m
            break
    heavy = [bool(table[syndrome] > 1) for flag, syndrome in zip(flags, syndromes) if flag]
    return {
        "smallest_violation": smallest,
        "heavy": heavy,
        **brute_force_counts(reduced.flags, heavy, max_weight),
    }


def brute_force_counts(
    flags: Sequence[int], heavy: Sequence[bool], max_weight: int
) -> dict[str, Any]:
    """A up to K, and Pi, E, and Phi up to K' = min(K, r + 1), by their definitions.

    Args:
        flags: The flags of every reduced mechanism, at most 64 bits each.
        heavy: Whether each flagged mechanism, in order, is heavy.
        max_weight: K.
    """
    flag_array = np.array(flags, dtype=np.uint64)
    no_syndromes = np.zeros(len(flags), dtype=np.int64)
    no_table = np.zeros(1, dtype=np.int64)
    accepted = {
        str(m): _enumerate_accepted(flag_array, no_syndromes, no_table, m)[0]
        for m in range(1, max_weight + 1)
    }
    weight_limit = min(max_weight, gf2_rank(flags) + 1)
    member_flags = flag_array[flag_array != 0]
    classes = get_classes([int(flag) for flag in member_flags], list(heavy))
    class_flags = np.array(classes.flags, dtype=np.uint64)
    heavy_array = np.array(heavy, dtype=np.bool_)
    counts: dict[str, Any] = {"A": accepted, "Pi": {}, "E": {}, "Phi": {}}
    for m in range(2, weight_limit + 1):
        counts["Pi"][str(m - 1)] = _count_independent(class_flags, classes.num_start, m - 1)
        counts["E"][str(m)] = _count_circuits(class_flags, classes.num_start, m)
        counts["Phi"][str(m)] = _count_heavy_circuits(member_flags, heavy_array, m)
    return counts


@always_compiled
def _next_combination(indices: npt.NDArray[np.int64], n: int) -> bool:
    """Advance an increasing index tuple to the next one; return False after the last."""
    k = indices.shape[0]
    i = k - 1
    while i >= 0 and indices[i] == n - k + i:
        i -= 1
    if i < 0:
        return False
    indices[i] += 1
    for j in range(i + 1, k):
        indices[j] = indices[j - 1] + 1
    return True


@always_compiled
def _rank64(vectors: npt.NDArray[np.uint64]) -> int:
    """The GF(2) rank of 64-bit vectors."""
    basis = np.zeros(64, dtype=np.uint64)
    rank = 0
    for vector in vectors:
        v = vector
        for bit in range(63, -1, -1):
            if not (v >> np.uint64(bit)) & np.uint64(1):
                continue
            if basis[bit] == 0:
                basis[bit] = v
                rank += 1
                break
            v ^= basis[bit]
    return rank


@always_compiled
def _enumerate_accepted(
    flags: npt.NDArray[np.uint64],
    syndromes: npt.NDArray[np.int64],
    table: npt.NDArray[np.int64],
    m: int,
) -> tuple[int, bool]:
    """The accepted faults of weight m, and whether one of them violates the bound."""
    n = flags.shape[0]
    if m > n:
        return 0, False
    indices = np.arange(m)
    count = 0
    violated = False
    while True:
        flag = np.uint64(0)
        syndrome = 0
        for i in indices:
            flag ^= flags[i]
            syndrome ^= syndromes[i]
        if flag == 0:
            count += 1
            if table[syndrome] > m:
                violated = True
        if not _next_combination(indices, n):
            return count, violated


@always_compiled
def _count_independent(class_flags: npt.NDArray[np.uint64], num_start: int, length: int) -> int:
    """Increasing tuples of classes with independent flags whose first class is a start class."""
    n = class_flags.shape[0]
    if length > n or length < 1:
        return 0
    indices = np.arange(length)
    count = 0
    while True:
        if indices[0] >= num_start:
            return count
        if _rank64(class_flags[indices]) == length:
            count += 1
        if not _next_combination(indices, n):
            return count


@always_compiled
def _count_circuits(class_flags: npt.NDArray[np.uint64], num_start: int, size: int) -> int:
    """Minimal sets of classes whose flags sum to zero, with a start class first."""
    n = class_flags.shape[0]
    if size > n:
        return 0
    indices = np.arange(size)
    count = 0
    while True:
        if indices[0] >= num_start:
            return count
        total = np.uint64(0)
        for i in indices:
            total ^= class_flags[i]
        if total == 0 and _rank64(class_flags[indices]) == size - 1:
            count += 1
        if not _next_combination(indices, n):
            return count


@always_compiled
def _count_heavy_circuits(
    flags: npt.NDArray[np.uint64], heavy: npt.NDArray[np.bool_], size: int
) -> int:
    """Minimal sets of flagged mechanisms whose flags sum to zero and that contain a heavy one.

    A set of vectors that sums to zero is minimal exactly when its rank is one less than its size.
    """
    n = flags.shape[0]
    if size > n:
        return 0
    indices = np.arange(size)
    count = 0
    while True:
        total = np.uint64(0)
        has_heavy = False
        for i in indices:
            total ^= flags[i]
            has_heavy = has_heavy or heavy[i]
        if total == 0 and has_heavy and _rank64(flags[indices]) == size - 1:
            count += 1
        if not _next_combination(indices, n):
            return count
