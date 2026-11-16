# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact search for a violating fault, and its compiled listing of faults."""

from __future__ import annotations

import functools
import itertools
import operator
import tracemalloc
from collections.abc import Callable, Collection, Sequence
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
import stim

from . import _fault_search


def _irreducible_faults(
    flags: Sequence[int], fault_weight: int, num_starts: int
) -> list[tuple[int, ...]]:
    """By brute force, in lexicographic order, the irreducible accepted faults of a fault weight.

    Only the faults that contain one of the first ``num_starts`` primitive faults are returned.
    """
    accepted: list[int] = []
    faults = []
    for weight in range(1, fault_weight + 1):
        for indices in itertools.combinations(range(len(flags)), weight):
            if functools.reduce(operator.xor, (flags[jj] for jj in indices)) == 0:
                word = sum(1 << jj for jj in indices)
                irreducible = not any(previous & word == previous for previous in accepted)
                if irreducible and weight == fault_weight and indices[0] < num_starts:
                    faults.append(indices)
                accepted.append(word)
    return faults


def _xor(values: Sequence[int], indices: Sequence[int]) -> int:
    return functools.reduce(operator.xor, (values[jj] for jj in indices))


def _check_faults(
    lister: _fault_search.FaultLister, fault_weight: int, violating: Collection[int]
) -> tuple[list[int], tuple[tuple[int, ...], int] | None]:
    """List faults, with a search that checks each syndrome in order and stops at one of these.

    Return the checked syndromes, and the result of the listing.
    """
    checked: list[int] = []

    def find_violation(syndromes: npt.NDArray[np.uint64], weight: int) -> int:
        assert weight == fault_weight
        for index, row in enumerate(syndromes.astype("<u8")):
            checked.append(int.from_bytes(row.tobytes(), "little"))
            if checked[-1] in violating:
                return index
        return len(syndromes)

    return checked, lister.find_first(fault_weight, find_violation)


@pytest.mark.parametrize("collide", [False, True])
def test_listing_matches_brute_force(
    collide: bool, pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each irreducible accepted fault with a start member is listed once, and no other fault is."""
    # With zero hashes, lookups find every entry of a table and must compare flags.  Small batches
    # make a listing pause and resume often.
    if collide:
        monkeypatch.setattr(
            _fault_search, "_hash_words", lambda words: np.zeros(len(words), np.uint64)
        )
    monkeypatch.setattr(_fault_search, "_INITIAL_BATCH_SIZE", 1)
    monkeypatch.setattr(_fault_search, "_BATCH_SIZE", 3)
    monkeypatch.setattr(_fault_search, "_CODEWORD_BATCH_SIZE", 1)
    # Record the memory limit of each listing that builds a closing map of pairs of two or more
    # classes, to check that only the listings with memory for one build one.
    limits_with_pairs: set[int] = set()
    get_closing_map = _fault_search._get_closing_map

    def record_closing_map(
        hashes: npt.NDArray[np.uint64], size: int
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        if size == 2 and len(hashes) > 1:
            limits_with_pairs.add(memory_limit)
        return get_closing_map(hashes, size)

    monkeypatch.setattr(_fault_search, "_get_closing_map", record_closing_map)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The first flags make an accepted fault {0, 1, 2, 3} that contains the accepted fault {0, 2}.
    # In the second, the classes with flags 1, 2, 4, and 8 have the same sum of flags, 15, as the
    # pair of classes with flags 3 and 12, but 3 = 1 + 2, so these six classes are not a class
    # codeword.  In the third, with two start members, 0 and 1, the class with flags 1 has members 0
    # and 3, and the class with flags 2 is a start class.  So the fault {1, 2, 3} takes member 3,
    # which is not a start member, before member 1, which is, and must not be skipped.  In the
    # fourth, the classes with flags 1 and 2 have two closing pairs, with flags 4 and 7 and with
    # flags 5 and 6, so the order of the pairs that one lookup finds decides the order of the
    # faults.
    flag_sets = [
        [1, 2, 1, 2, 3, 3],
        [1, 2, 4, 8, 3, 12, 16, 31],
        [1, 2, 3, 1],
        [1, 2, 3, 4, 5, 6, 7],
    ]
    flag_sets += [list(rng.integers(1, 16, size=rng.integers(1, 10))) for _ in range(10)]
    for small_flags in flag_sets:
        # Spread the five bits of the flags over five words.
        flags = [
            sum(((int(ff) >> bit) & 1) << (70 * bit) for bit in range(5)) for ff in small_flags
        ]
        num_primitives = len(flags)
        # Syndromes of distinct bits, 13 apart so that those of primitive faults 5 and up are in the
        # second word, give each fault its own syndrome.  Errors on disjoint sets of seven qubits
        # leave out no fault.
        syndromes = [1 << (13 * jj) for jj in range(num_primitives)]
        errors = [0b1111111 << (7 * jj) for jj in range(num_primitives)]
        # Without memory, the lister builds no closing map of pairs, except an empty one when there
        # are fewer than two classes.  The memory limit must not change the order in which faults
        # are listed, since the first violation in that order is the result.
        orders: dict[tuple[int, int], list[int]] = {}
        for memory_limit, num_starts in itertools.product((0, 2**20), (num_primitives, 2, 1)):
            args = (flags, syndromes, errors, 70, False, num_starts, memory_limit)
            lister = _fault_search.FaultLister(*args)
            for fault_weight in range(2, 7):
                expected = _irreducible_faults(flags, fault_weight, num_starts)
                checked, result = _check_faults(lister, fault_weight, ())
                assert result is None
                assert sorted(checked) == sorted(_xor(syndromes, fault) for fault in expected)
                assert orders.setdefault((num_starts, fault_weight), checked) == checked
                if expected:
                    stop = int(rng.integers(len(expected)))
                    violation = (expected[stop], _xor(syndromes, expected[stop]))
                    # Use a new lister, since a lister skips the syndromes that it has listed.
                    _, result = _check_faults(
                        _fault_search.FaultLister(*args), fault_weight, {violation[1]}
                    )
                    assert result == violation
    assert limits_with_pairs == {2**20}


@pytest.mark.parametrize("size", [1, 2])
def test_closing_tables_sort_sets_stably_by_hash(size: int, pytestconfig: pytest.Config) -> None:
    """A closing table sorts its sets by hash, keeping lexicographic order among equal hashes."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_classes in (0, 1, 2, 5, 40):
        # Random hashes, hashes from four values so that many sets share one, and small hashes so
        # that different hashes share their high bits.
        for hashes in (
            rng.integers(2**64, size=num_classes, dtype=np.uint64),
            rng.integers(2**64, size=4, dtype=np.uint64)[rng.integers(4, size=num_classes)],
            rng.integers(2**8, size=num_classes, dtype=np.uint64),
        ):
            expected_sets = sorted(
                itertools.combinations(range(num_classes), size),
                key=lambda classes: _xor(hashes.tolist(), classes),
            )
            keys, sets = _fault_search._get_closing_table(hashes, size)
            assert keys.tolist() == [_xor(hashes.tolist(), classes) for classes in expected_sets]
            assert sets.tolist() == [list(classes) for classes in expected_sets]


def test_repeated_syndromes_are_checked_once() -> None:
    """A fault with no error, or whose syndrome is already in the table, is not checked."""
    # On |000>, the faults {0, 1} and {0, 2} leave X on all qubits, and {1, 2} leaves no error.
    lister = _fault_search.FaultLister([1, 1, 1], [7, 0, 0], [7, 0, 0], 3, False, 3, 0)
    assert _check_faults(lister, 2, ()) == ([7], None)


@pytest.mark.parametrize("collide", [False, True])
def test_light_pairs_match_brute_force(
    collide: bool, pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each heavy syndrome h gets the first light syndrome i such that h + i is a light syndrome."""
    # With zero hashes, lookups must compare syndromes.  Syndromes of bits 62 to 65 span two words.
    if collide:
        monkeypatch.setattr(
            _fault_search, "_hash_words", lambda words: np.zeros(len(words), np.uint64)
        )
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(20):
        light = list(dict.fromkeys(int(ss) << 62 for ss in rng.integers(16, size=6)))
        heavy = [int(ss) << 62 for ss in rng.integers(16, size=8)]
        num_words = _fault_search._get_num_words(light + heavy)
        pairs = _fault_search._get_light_pairs(
            _fault_search._hashed_rows(heavy, num_words),
            _fault_search._hashed_rows(light, num_words),
        )
        for syndrome, pair in zip(heavy, pairs.tolist(), strict=True):
            first = next((ii for ii, ss in enumerate(light) if syndrome ^ ss in light), -1)
            expected = [first, light.index(syndrome ^ light[first])] if first >= 0 else [-1, -1]
            assert pair == expected


@pytest.mark.parametrize("css_weight", [False, True])
def test_skipped_faults_satisfy_the_bound(css_weight: bool, pytestconfig: pytest.Config) -> None:
    """Calls of rising fault weight check no syndrome twice, skip no violating one, and find one."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    n = 3
    paulis = (np.arange(4**n)[:, None] >> np.arange(2 * n)) & 1
    x, z = paulis[:, :n], paulis[:, n:]
    weights = np.maximum(x.sum(axis=1), z.sum(axis=1)) if css_weight else (x | z).sum(axis=1)
    for _ in range(20):
        circuit = stim.Circuit()
        circuit.append("I", range(n))
        for gate in rng.integers(3, size=12):
            qubits = rng.permutation(n)[: 2 if gate == 2 else 1]
            circuit.append(("H", "S", "CX")[gate], qubits.tolist())
        state = stim.Tableau.from_circuit(circuit).to_stabilizers()
        rows = np.array([np.concatenate(stabilizer.to_numpy()) for stabilizer in state], dtype=int)
        bits = (x @ rows[:, n:].T + z @ rows[:, :n].T) % 2
        syndrome_of = bits @ (1 << np.arange(n))
        minimum = {
            int(syndrome): int(weights[syndrome_of == syndrome].min()) for syndrome in range(2**n)
        }
        lightest = {
            int(syndrome_of[pauli]): pauli
            for pauli in sorted(range(4**n), key=lambda pauli: -weights[pauli])
        }

        flags = list(map(int, rng.integers(1, 8, size=8)))
        syndromes = [int(syndrome_of[pauli]) for pauli in rng.integers(4**n, size=8)]
        representatives = [lightest[syndrome] for syndrome in syndromes]
        lister, searcher = (
            _fault_search.FaultLister(flags, syndromes, representatives, n, css_weight, 8, 2**20)
            for _ in range(2)
        )
        passed: set[int] = set()
        for fault_weight in range(2, 5):
            faults = _irreducible_faults(flags, fault_weight, 8)
            fault_syndromes = [_xor(syndromes, fault) for fault in faults]
            violating = [jj for jj, ss in enumerate(fault_syndromes) if minimum[ss] > fault_weight]

            checked, _ = _check_faults(lister, fault_weight, ())
            assert len(set(checked)) == len(checked)
            assert not passed & set(checked)
            assert {fault_syndromes[jj] for jj in violating} - passed <= set(checked)
            passed |= set(checked)

            violating_syndromes = {ss for ss, weight in minimum.items() if weight > fault_weight}
            _, result = _check_faults(searcher, fault_weight, violating_syndromes)
            expected = [(faults[jj], fault_syndromes[jj]) for jj in violating]
            assert (result is None) == (not expected)
            assert result is None or result in expected


def test_syndromes_of_listed_faults_are_not_checked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fault is not checked if the table holds the syndrome of a smaller listed fault."""
    # The table of syndromes starts with room for one syndrome, so that it grows if there is memory.
    monkeypatch.setattr(_fault_search, "_INITIAL_SYNDROME_TABLE_SIZE", 1)
    # On |0...0>, the minimum Pauli weight of a syndrome is its number of ones.  Primitive faults
    # 0-3 have flags 1 and flip qubits 0-3, one each, and 4 and 5 have flags 2 and 3 and flip qubits
    # {4} and {0, 4}.  So the faults of fault weight two are the pairs of 0-3, and those of fault
    # weight three add 4 and 5 to one of 0-3.  Each representative also has Z on two qubits of its
    # own, so that the weight of the representatives leaves out no fault.
    flags = [1, 1, 1, 1, 2, 3]
    syndromes = [1, 2, 4, 8, 16, 17]
    num_qubits = 5 + 2 * len(flags)
    representatives = [
        syndrome | (0b11 << (num_qubits + 5 + 2 * jj)) for jj, syndrome in enumerate(syndromes)
    ]
    pairs = [syndromes[aa] ^ syndromes[bb] for aa, bb in itertools.combinations(range(4), 2)]
    for memory_limit in (2**20, 0):
        lister = _fault_search.FaultLister(
            flags, syndromes, representatives, num_qubits, False, 6, memory_limit
        )
        assert _check_faults(lister, 2, ()) == (pairs, None)
        checked, _ = _check_faults(lister, 3, ())
        # Fault {0, 4, 5} has no syndrome, and {jj, 4, 5} has that of the pair {0, jj}.  Without
        # memory, the table keeps only the syndrome of the first pair, {0, 1}.
        assert checked == ([0] if memory_limit else [0, 1 ^ 4, 1 ^ 8])

    # A search that finds a violation clears the table, so a second search finds the violation
    # again.
    searcher = _fault_search.FaultLister(
        flags, syndromes, representatives, num_qubits, False, 6, 2**20
    )
    for _ in range(2):
        assert _check_faults(searcher, 2, {1 ^ 4})[1] == ((0, 2), 1 ^ 4)


def test_pair_table_peak_memory_is_within_its_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    """A table of pairs and its slots need at most about ``_BYTES_PER_PAIR`` bytes for each pair."""
    # tracemalloc traces only allocations by Python, so build the table with the Python versions of
    # the compiled functions.
    for name in ("_get_closing_table", "_sort_bucket", "_get_key_slots", "_find_syndrome_slot"):
        function = getattr(_fault_search, name)
        monkeypatch.setattr(_fault_search, name, getattr(function, "py_func", function))
    # Distinct random hashes leave no bucket to merge sort, and give the pairs distinct keys, which
    # need the most slots.
    num_classes = 200
    hashes = np.random.default_rng(0).integers(2**64, size=num_classes, dtype=np.uint64)
    # Build a small table first, so that loading the compiled code that the Python versions call
    # adds no allocations to the measurement.
    _fault_search._get_closing_map(hashes[:3], 2)
    was_tracing = tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        _fault_search._get_closing_map(hashes, 2)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if not was_tracing:
            tracemalloc.stop()
    # Allow a few kilobytes that do not grow with the number of pairs.
    num_pairs = num_classes * (num_classes - 1) // 2
    assert peak - before <= num_pairs * _fault_search._BYTES_PER_PAIR + 2**14


def test_memory_limit_is_split_within_the_caps() -> None:
    """The memory limit goes to the map of pairs first, and then to the table of syndromes.

    The map of pairs holds at most 2**31 pairs, and the table of syndromes gets room for at most
    2**31, however large the limit.
    """
    split = _fault_search._split_memory_limit
    # With 2**16 classes there are 2**31 - 2**15 pairs, which the map may hold.  With one more
    # class there are more than 2**31, which it may not.
    assert split(2**16, 1, 2**50) == (True, 2**31)
    assert split(2**16 + 1, 1, 2**50) == (False, 2**31)
    # Three classes have three pairs.  A syndrome of two words takes 40 bytes: 8 for its hash, 16
    # for its words, and 16 for its two slots.  The table gets what the map leaves, or the whole
    # limit if the map misses by a byte.
    pair_bytes = 3 * _fault_search._BYTES_PER_PAIR
    syndrome_bytes = 40
    assert split(3, 2, pair_bytes) == (True, 0)
    assert split(3, 2, pair_bytes + 5 * syndrome_bytes) == (True, 5)
    assert split(3, 2, pair_bytes - 1) == (False, (pair_bytes - 1) // syndrome_bytes)


def test_key_slots_find_the_first_entry_of_each_key(pytestconfig: pytest.Config) -> None:
    """A lookup in sorted keys finds the first entry of a key, or none for a key that is absent."""
    # The keys repeat, and different keys share their low 32 bits, which pick their slots, so that
    # lookups must pass slots of other keys.
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    values = [low + (high << 32) for low in (0, 1, 2**31) for high in (0, 1, 2**31)]
    for num_keys in (0, 1, 20):
        keys = np.sort(rng.choice(values[:-1], size=num_keys).astype(np.uint64))
        slots = _fault_search._get_key_slots(keys)
        rows = keys.reshape((len(keys), 1))
        for value in values:
            slot = _fault_search._find_syndrome_slot(np.uint64(value), keys[:0], rows, slots)
            assert slots[slot] == (keys.tolist().index(value) if value in keys else -1)


def _z_state(num_qubits: int) -> npt.NDArray[np.int_]:
    """Stabilizers of |0...0>, on which a syndrome's smallest Pauli weight is its number of ones."""
    return np.hstack([np.zeros((num_qubits, num_qubits), dtype=int), np.eye(num_qubits, dtype=int)])


def _smallest_violation(
    flags: Sequence[int],
    syndromes: Sequence[int],
    max_weight: int | None,
    residual_weight: Callable[[int], int] = int.bit_count,
) -> int | None:
    """By brute force, the smallest fault weight of an accepted fault that violates the bound.

    ``residual_weight`` maps a syndrome to its smallest Pauli weight, by default as on |0...0>.
    """
    limit = len(flags) if max_weight is None else min(max_weight, len(flags))
    for fault_weight in range(1, limit + 1):
        for indices in itertools.combinations(range(len(flags)), fault_weight):
            flag = functools.reduce(operator.xor, (flags[jj] for jj in indices))
            syndrome = functools.reduce(operator.xor, (syndromes[jj] for jj in indices))
            if flag == 0 and residual_weight(syndrome) > fault_weight:
                return fault_weight
    return None


def _certify_map(
    flags: Sequence[int], syndromes: Sequence[int], max_weight: int | None, num_qubits: int
) -> _fault_search.Violation | None:
    """Search primitive faults with these flags and syndromes on |0...0> against brute force."""
    violation = _fault_search.find_smallest_violation(
        flags, syndromes, _z_state(num_qubits), max_weight, False
    )
    smallest = _smallest_violation(flags, syndromes, max_weight)
    assert (violation is None) == (smallest is None)
    if violation is not None:
        indices = violation.fault_indices
        flag = functools.reduce(operator.xor, (flags[jj] for jj in indices))
        syndrome = functools.reduce(operator.xor, (syndromes[jj] for jj in indices))
        assert len(indices) == len(set(indices)) == smallest
        assert flag == 0
        assert violation.syndrome == syndrome
        assert len(indices) < violation.bound.lower_bound <= syndrome.bit_count()
        # X on a qubit anticommutes with that qubit's stabilizer Z.
        assert violation.bound.representative & ((1 << num_qubits) - 1) == syndrome
    return violation


def test_search_without_flags_checks_primitive_faults() -> None:
    """With zero flags, a primitive fault whose residual weight exceeds one fails."""
    assert _certify_map([0] * 5, [0, 1, 2, 4, 2], None, 3) is None
    assert _certify_map([0] * 5, [1, 2, 7, 4, 7], None, 3) is not None


def test_search_with_flags_matches_exhaustive_search(pytestconfig: pytest.Config) -> None:
    """Random maps with repeated flags and flag rank above ``max_weight`` match brute force."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for n, rank in itertools.product((3, 5), range(5)):
        for _ in range(12):
            flags = list(map(int, rng.integers(1 << rank, size=8)))
            syndromes = list(map(int, rng.integers(1 << n, size=8)))
            for max_weight in (1, 2, 3, 5, None):
                _certify_map(flags, syndromes, max_weight, n)


def test_css_weight_search_matches_exhaustive_search(pytestconfig: pytest.Config) -> None:
    """On |00++>, the search matches brute force by CSS weight for random flags and syndromes."""
    # The stabilizers Z0, Z1, X2, and X3 make syndrome bits 0 and 1 need X on qubits 0 and 1, and
    # bits 2 and 3 need Z on qubits 2 and 3, so the oracle searches both X parts and Z parts.
    stabilizers = np.zeros((4, 8), dtype=int)
    stabilizers[[0, 1], [4, 5]] = 1
    stabilizers[[2, 3], [2, 3]] = 1

    def css_weight(syndrome: int) -> int:
        return max((syndrome & 0b0011).bit_count(), (syndrome & 0b1100).bit_count())

    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for rank in range(5):
        for _ in range(12):
            flags = list(map(int, rng.integers(1 << rank, size=8)))
            syndromes = list(map(int, rng.integers(1 << 4, size=8)))
            for max_weight in (1, 2, None):
                violation = _fault_search.find_smallest_violation(
                    flags, syndromes, stabilizers, max_weight, True
                )
                smallest = _smallest_violation(flags, syndromes, max_weight, css_weight)
                assert (violation is None) == (smallest is None)
                if violation is not None:
                    syndrome = violation.syndrome
                    assert len(violation.fault_indices) == smallest
                    assert smallest < violation.bound.lower_bound <= css_weight(syndrome)
                    # The X part of the representative flips bits 0 and 1, and its Z part 2 and 3.
                    representative = violation.bound.representative
                    assert (representative & 0b0011) | ((representative >> 4) & 0b1100) == syndrome


def test_hand_derived_dependency_cases() -> None:
    """Small faults, repeated and zero signatures, and ``max_weight`` give hand-derived results."""
    for flags, syndromes, max_weight, certified in (
        ([0, 0, 0], [1, 2, 4], 4, True),  # zero flags; each has residual weight one
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 4, 8, 12], 4, False),
        # the accepted faults have fault weight two and residual weight two
        ([1, 1, 2, 2, 4, 4], [1, 2, 1, 2, 1, 2], 4, True),
        # the heavy primitive faults 4 and 5 are in no violating fault
        ([1, 1, 2, 2, 3, 3], [1, 2, 1, 2, 5, 6], None, True),
        ([1, 1, 1, 0], [1, 1, 1, 0], 4, True),  # equal signatures cancel each other
        ([0, 0], [0, 0], 4, True),  # primitive faults with no effect
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], 3, True),
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], 4, False),  # fault weight 4, residual weight 6
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], None, False),
    ):
        failure = _certify_map(flags, syndromes, max_weight, 6)
        assert (failure is None) == certified


def test_batches_have_a_word_for_every_64_output_qubits() -> None:
    """On 70 qubits, syndromes below ``2**64`` have two words in batches, as the oracle needs."""
    # Together, the two primitive faults pass post-selection and leave X on three qubits.
    failure = _certify_map([1, 1], [0b011, 0b100], None, 70)
    assert failure is not None and failure.fault_indices == (0, 1)


def test_failures_report_the_smallest_faults() -> None:
    """A violating fault of two primitive faults is reported before a violating fault of three."""
    # The primitive faults 0-2 form an accepted fault of residual weight four, and the primitive
    # faults 3 and 4 the only accepted fault of fault weight two, which has residual weight three.
    failure = _certify_map([1, 2, 3, 4, 4], [1, 2, 12, 1, 6], None, 6)
    assert failure is not None and set(failure.fault_indices) == {3, 4}


def test_light_primitive_faults_are_never_searched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a heavy primitive fault, no fault can violate the bound, so none is listed."""
    flags = [3, 5, 6, 9, 10, 12, 1, 2, 4, 8, 7, 11]
    syndromes = [1, 2, 4, 8, 0, 1, 2, 4, 8, 0, 1, 2]

    with monkeypatch.context() as patch:
        lister = mock.Mock(wraps=_fault_search.FaultLister)
        patch.setattr(_fault_search, "FaultLister", lister)
        assert _certify_map(flags, syndromes, None, 4) is None
    lister.assert_not_called()

    # One primitive fault of residual weight three brings back the search, and it fails
    # together with the primitive fault that has the same flags.
    failure = _certify_map(flags + [3], syndromes + [14], None, 4)
    assert failure is not None and len(failure.fault_indices) == 2


def test_max_weight_limits_the_fault_weight() -> None:
    """Only faults of fault weight at most ``max_weight`` are checked."""
    # The only accepted fault is that of all three primitive faults.
    for max_weight in (None, 30, 2):
        failure = _certify_map([1, 2, 3], [1, 2, 12], max_weight, 4)
        assert (failure is None) == (max_weight == 2)


def test_memory_limit_does_not_change_results(pytestconfig: pytest.Config) -> None:
    """With almost no memory, the search finds the same counterexample as with ample memory."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The first map has flags of rank four, and its only accepted fault has fault weight five and
    # residual weight six, so that the search finds it with the table of pairs if it fits.
    maps = [([1, 2, 4, 8, 15], [3, 12, 48, 0, 0])]
    for _ in range(10):
        num_primitives = int(rng.integers(4, 10))
        flags = list(map(int, rng.integers(1, 16, size=num_primitives)))
        maps.append((flags, list(map(int, rng.integers(64, size=num_primitives)))))
    for flags, syndromes in maps:
        results = [
            _fault_search.find_smallest_violation(flags, syndromes, _z_state(6), None, False, limit)
            for limit in (1, 2**20)
        ]
        assert results[0] == results[1] == _certify_map(flags, syndromes, None, 6)
