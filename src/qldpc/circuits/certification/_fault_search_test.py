# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact search for a violating fault, and its compiled listing of faults."""

from __future__ import annotations

import functools
import itertools
import math
import operator
import pathlib
import threading
import tracemalloc
from collections.abc import Callable, Collection, Generator, Iterator, Sequence
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import circuits, conftest
from qldpc.codes.distance import _threads, _tuning
from qldpc.codes.distance import conftest as distance_conftest
from qldpc.codes.distance.conftest import THREAD_TIMEOUT

from . import _fault_search

# The timeout of a test that runs the lister on a thread of its own, long enough to compile it on a
# cold cache of compiled code, such as its versions for read-only arrays.
_COMPILE_TIMEOUT = 6 * THREAD_TIMEOUT


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


def _listed_faults(
    flags: Sequence[int], fault_weight: int, num_starts: int
) -> list[tuple[int, ...]]:
    """By brute force, the faults that a lister lists at a fault weight, in the order it lists them.

    The lister numbers its classes in the order of their first members, lists its class tuples in
    lexicographic order, and lists the faults of each class tuple in lexicographic order of their
    members, in the order of their classes.
    """
    class_numbers: dict[int, int] = {}
    for flag in flags:
        class_numbers.setdefault(flag, len(class_numbers))

    def position(fault: tuple[int, ...]) -> tuple[list[int], list[int]]:
        members = sorted(fault, key=lambda jj: (class_numbers[flags[jj]], jj))
        return [class_numbers[flags[jj]] for jj in members], members

    return sorted(_irreducible_faults(flags, fault_weight, num_starts), key=position)


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
    """Each irreducible accepted fault with a start member is listed once, and no other fault is.

    The faults are listed in the same order with closing maps of any size, and only listers with
    memory for a closing map of two or more classes build it.
    """
    # With zero hashes, lookups find every entry of a table and must compare flags.  Small batches
    # make a listing pause and resume often.
    if collide:
        monkeypatch.setattr(
            _fault_search, "_hash_words", lambda words: np.zeros(len(words), np.uint64)
        )
    monkeypatch.setattr(_fault_search, "_INITIAL_BATCH_SIZE", 1)
    monkeypatch.setattr(_fault_search, "_BATCH_SIZE", 3)
    monkeypatch.setattr(_fault_search, "_CODEWORD_BATCH_SIZE", 1)
    # Build every closing map that fits, whatever the number of lookups.  Record the sizes of the
    # nonempty closing maps that the listers with each memory limit (from the loop below) build, to
    # check that some listers with memory for maps of each size build them, and that none without
    # memory does.
    monkeypatch.setattr(_fault_search, "_LOOKUPS_PER_CLOSING_SET", 0)
    sizes_built: set[tuple[int, int]] = set()
    get_closing_map = _fault_search._get_closing_map

    def record_closing_map(
        hashes: npt.NDArray[np.uint64], size: int
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        if size > 1 and len(hashes) >= size:
            sizes_built.add((memory_limit, size))
        return get_closing_map(hashes, size)

    monkeypatch.setattr(_fault_search, "_get_closing_map", record_closing_map)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The first flags make an accepted fault {0, 1, 2, 3} that contains the accepted fault {0, 2}.
    # In the second, the classes with flags 1, 2, 4, 8, 3, and 12 have flags that sum to zero, but
    # 3 = 1 + 2, so these six classes are not a class codeword.  After choosing the classes with
    # flags 1 and 2, a listing must reject the closing set with flags 4, 8, 3, and 12, and after
    # also choosing 4 and 8, the closing pair with flags 3 and 12.  In the third, with two start
    # members, 0 and 1, the class with flags 1 has members 0 and 3, and the class with flags 2 is a
    # start class.  So the fault {1, 2, 3} takes member 3, which is not a start member, before
    # member 1, which is, and must not be skipped.  In the fourth, the classes with flags 1 and 2
    # have two closing pairs, with flags 4 and 7 and with flags 5 and 6, so the order of the pairs
    # that one lookup finds decides the order of the faults.  In the fifth, after choosing the
    # classes with flags 1 and 2, a listing must reject the closing set with flags 4, 5, 8, and 10.
    # Unlike in the second, each class of that set is independent of the chosen classes, but the
    # classes with flags 4 and 5 together are not, since 5 = 1 + 4, nor are those with flags 8 and
    # 10, since 10 = 2 + 8.
    flag_sets = [
        [1, 2, 1, 2, 3, 3],
        [1, 2, 4, 8, 3, 12, 16, 31],
        [1, 2, 3, 1],
        [1, 2, 3, 4, 5, 6, 7],
        [1, 2, 4, 5, 8, 10],
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
        # Without memory, the lister closes class codewords with single classes.  With 2**15 bytes,
        # it has room for the map of pairs, but its table of syndromes, which starts with room for
        # 1024 syndromes of 40 bytes, leaves no room for larger maps.  With 2**20 bytes, it builds
        # maps of up to four classes.  The memory limit must not change the order in which faults
        # are listed, since the first violation in that order is the result.
        orders: dict[tuple[int, int], list[int]] = {}
        memory_limits = (0, 2**15, 2**20)
        for memory_limit, num_starts in itertools.product(memory_limits, (num_primitives, 2, 1)):
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
    assert sizes_built == {(2**15, 2), (2**20, 2), (2**20, 3), (2**20, 4)}


@pytest.mark.parametrize("size", [1, 2, 3, 4])
def test_closing_tables_sort_sets_stably_by_hash(size: int, pytestconfig: pytest.Config) -> None:
    """A closing table sorts its sets by hash, keeping lexicographic order among equal hashes."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_classes in (0, 1, 2, 5, 40 if size <= 2 else 16):
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
        num_words = _fault_search._get_num_words_for_values(light + heavy)
        pairs = _fault_search._get_light_pairs(
            _fault_search._hashed_rows(heavy, num_words),
            _fault_search._hashed_rows(light, num_words),
        )
        for syndrome, pair in zip(heavy, pairs.tolist(), strict=True):
            first = next((ii for ii, ss in enumerate(light) if syndrome ^ ss in light), -1)
            expected = [first, light.index(syndrome ^ light[first])] if first >= 0 else [-1, -1]
            assert pair == expected


def test_heavy_primitive_faults_get_products_of_light_representatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A heavy primitive fault gets the product of the representatives of its first light pair.

    The pair starts with the first light syndrome, in the order of the primitive faults, whose sum
    with the heavy syndrome is also light.  A heavy primitive fault without a pair keeps the
    oracle's representative.
    """
    # The stabilizers ZXX, X on qubit 2, and XZX make the syndromes 1, 4, 5, and 7 light.  With the
    # light syndromes 5, 7, and 4 of these faults, the heavy syndrome 2, which repeats, is 5 + 7,
    # the heavy syndrome 3 is 7 + 4, and the heavy syndrome 6 is no such sum.  The oracle's
    # representatives of 2 and 3 differ from these products.
    stabilizers = np.array([[0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 0, 0], [1, 0, 1, 0, 1, 0]])
    syndromes = [5, 7, 4, 2, 2, 3, 6]
    lister = mock.Mock(wraps=_fault_search.FaultLister)
    monkeypatch.setattr(_fault_search, "FaultLister", lister)
    flags = [1 << jj for jj in range(len(syndromes))]
    assert _fault_search.find_smallest_violation(flags, syndromes, stabilizers, None, False) is None
    _, primitive_syndromes, representatives, *_, num_starts, _ = lister.call_args.args

    def syndrome_of(pauli: int) -> int:
        x = [(pauli >> qubit) & 1 for qubit in range(3)]
        z = [(pauli >> (3 + qubit)) & 1 for qubit in range(3)]
        return sum(
            int(np.dot(x, row[3:]) + np.dot(z, row[:3])) % 2 << jj
            for jj, row in enumerate(stabilizers)
        )

    for syndrome, representative in zip(primitive_syndromes, representatives, strict=True):
        assert syndrome_of(representative) == syndrome

    oracle = _fault_search.CosetLeaderSearch(stabilizers)

    def oracle_representative(syndrome: int) -> int:
        return oracle.get_min_weight_bound(syndrome, 1).representative

    light = [5, 7, 4]
    expected = []
    for syndrome in primitive_syndromes[:num_starts]:
        first = next((ss for ss in light if syndrome ^ ss in light), None)
        expected.append(
            oracle_representative(syndrome)
            if first is None
            else oracle_representative(first) ^ oracle_representative(syndrome ^ first)
        )
    assert primitive_syndromes[:num_starts] == [2, 2, 3, 6]
    assert list(representatives[:num_starts]) == expected
    assert oracle_representative(2) != expected[0] and oracle_representative(3) != expected[2]


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


def test_checked_faults_follow_the_table_of_syndromes(
    pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lister checks, in the order it lists them, the faults that the table does not leave out.

    It leaves out a fault if its syndrome is in the table of syndromes, which takes each new
    syndrome while it has room, or if its representatives have Pauli weight at most its fault
    weight.  This holds whether or not the table fills up, though the lister skips many of the
    faults that the table leaves out without looking them up.
    """
    # With no memory, the table of syndromes keeps room for its first two syndromes.
    monkeypatch.setattr(_fault_search, "_INITIAL_SYNDROME_TABLE_SIZE", 2)
    default_remembered = _fault_search._MAX_REMEMBERED_CHOICES

    def check(
        flags: list[int],
        syndromes: list[int],
        representatives: list[int],
        num_qubits: int,
        num_starts: int,
    ) -> None:
        """Check the listing against a model of the table, with and without room in it.

        Also check it when the lister remembers only one choice of members of each class tuple.
        """
        cases = itertools.product(((0, 2), (2**20, None)), (default_remembered, 1))
        for (memory_limit, room), remembered in cases:
            monkeypatch.setattr(_fault_search, "_MAX_REMEMBERED_CHOICES", remembered)
            lister = _fault_search.FaultLister(
                flags, syndromes, representatives, num_qubits, False, num_starts, memory_limit
            )
            table: set[int] = set()
            for fault_weight in range(2, 6):
                expected = []
                for fault in _listed_faults(flags, fault_weight, num_starts):
                    syndrome = _xor(syndromes, fault)
                    if syndrome in table:
                        continue
                    if room is None or len(table) < room:
                        table.add(syndrome)
                    error = _xor(representatives, fault)
                    error_weight = (
                        error & ((1 << num_qubits) - 1) | error >> num_qubits
                    ).bit_count()
                    if error_weight > fault_weight:
                        expected.append(syndrome)
                assert _check_faults(lister, fault_weight, ()) == (expected, None)

    # Syndromes of two bits repeat often, for faults and for partial choices of their members, and
    # errors on six qubits leave out some faults but not all.  Then the same with zero hashes, so
    # that every remembered choice shares its hash, and with the two bits of each syndrome in
    # different words, so that a remembered choice must match every word of its syndrome.
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for shift, num_qubits in ((0, 6), (63, 70)):
        if shift:
            monkeypatch.setattr(
                _fault_search, "_hash_words", lambda words: np.zeros(len(words), np.uint64)
            )
        for _ in range(20):
            num_primitives = int(rng.integers(6, 11))
            flags = [int(flag) for flag in rng.integers(1, 8, size=num_primitives)]
            syndromes = [
                int(syndrome) << shift for syndrome in rng.integers(4, size=num_primitives)
            ]
            representatives = [int(error) for error in rng.integers(2**12, size=num_primitives)]
            num_starts = int(rng.integers(1, num_primitives + 1))
            check(flags, syndromes, representatives, num_qubits, num_starts)

    # The start members 0, 1, and 2 come first in the classes with flags 1, 2, and 4, which, with
    # the class with flags 7, form a class codeword.  The choice of members 3 and 5 of the first two
    # classes has no start member, and the later choice of 4 and 1 has one, with the same syndrome.
    # Only the latter is completed by member 6, which is not a start member, and its fault {1, 4, 6,
    # 7} has a syndrome of its own.  Each representative has weight three.
    flags = [1, 2, 4, 1, 1, 2, 4, 7]
    syndromes = [16, 3, 32, 1, 0, 2, 64, 128]
    representatives = [0b111 << (3 * jj) for jj in range(len(flags))]
    check(flags, syndromes, representatives, 24, 3)

    # Primitive fault 3 has flags but no syndrome, so adding it to a choice keeps the syndrome of
    # the choice.  In the class codeword of the classes with flags 1, 2, 4, and 7, the choice of
    # members 0, 2, and 3 of the first three classes has the syndrome of the choice of members 0 and
    # 2 of the first two, but other completions, so a remembered choice must also match the level.
    flags = [1, 1, 2, 4, 7]
    syndromes = [1, 2, 4, 0, 8]
    representatives = [0b111 << (3 * jj) for jj in range(len(flags))]
    check(flags, syndromes, representatives, 15, 1)

    # Members 0 to 3, the start members, come first in the classes with flags 1, 2, 4, and 7, and
    # members 4 to 8 come after them.  The syndrome of member 6 makes the choice of members 4, 6,
    # and 7 of the first three classes, which has no start member, share its syndrome with a later
    # choice that has one.  With the first syndromes below, that is the choice of members 5, 1,
    # and 7, whose start member was chosen before the last of its levels, so a remembered choice
    # must record a start member chosen at any level.  With the second, it is the choice of members
    # 5 and 1 of the first two classes, so it must also record the level, besides whether it has a
    # start member.  Member 8, which is not a start member, completes only the later choices, so
    # skipping them after the earlier choice would lose faults.
    flags = [1, 2, 4, 7, 1, 1, 2, 4, 7]
    representatives = [0b111 << (3 * jj) for jj in range(len(flags))]
    check(flags, [1, 2, 4, 8, 16, 32, 50, 64, 128], representatives, 27, 4)
    check(flags, [1, 2, 4, 8, 16, 32, 114, 64, 128], representatives, 27, 4)


@pytest.mark.parametrize("num_threads", [2, 3, 8])
@pytest.mark.parametrize("items_per_thread", [1e-9, 2**20])
@pytest.mark.parametrize("split_lookups", [0, 1])
def test_listing_on_threads_checks_the_faults_of_one_thread(
    num_threads: int,
    items_per_thread: float,
    split_lookups: int,
    pytestconfig: pytest.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On threads, a lister checks the faults of one thread, in order, with the same lookups.

    Listings move to threads at once, or after the calling thread has made a lookup for each thread
    alone, in work items of one start class each (with few items for each thread) or of one pair
    of classes, with batches of one class codeword, calls of one lookup, and one held batch for
    each thread, so that the threads pause and wait often.  This holds with a full table of
    syndromes too, and when a violation stops a listing, which leaves no thread running.  The
    threads run with the arrays of the lister in use that they read read-only, and leave them
    unchanged.  The calling thread consumes the listing on a thread of its own, so that a deadlock
    fails the test.
    """
    monkeypatch.setattr(_fault_search, "_SPLIT_LOOKUPS_PER_THREAD", split_lookups)
    monkeypatch.setattr(_fault_search, "_WALK_ITEMS_PER_THREAD", items_per_thread)
    monkeypatch.setattr(_fault_search, "_CODEWORD_BATCH_SIZE", 1)
    monkeypatch.setattr(_fault_search, "_LOOKUPS_PER_CALL", 1)
    monkeypatch.setattr(_fault_search, "_LOOKUPS_PER_CLOSING_SET", 0)
    monkeypatch.setattr(_threads, "OUTPUTS_PER_THREAD", 1)
    # With no memory, the table of syndromes keeps room for its first two syndromes.
    monkeypatch.setattr(_fault_search, "_INITIAL_SYNDROME_TABLE_SIZE", 2)

    # Record the lookups and the class codewords of each call that lists class codewords, and,
    # for a call on a thread of the listing, whether the rows of the classes were writeable and
    # whether it made at most one lookup.
    listed: list[tuple[int, int]] = []
    thread_calls: list[tuple[bool, bool]] = []
    list_class_codewords = _fault_search._list_class_codewords

    def record_listing(*args: Any) -> int:
        walk_state = args[-2]
        lookups = int(walk_state[1])
        num_codewords = list_class_codewords(*args)
        listed.append((int(walk_state[1]) - lookups, num_codewords))
        if threading.current_thread().name == "qldpc-search":
            thread_calls.append((args[0].flags.writeable, int(walk_state[1]) - lookups <= 1))
        return num_codewords

    monkeypatch.setattr(_fault_search, "_list_class_codewords", record_listing)
    in_use: list[_fault_search.FaultLister] = []
    marked = distance_conftest.check_shared_arrays_on_threads(
        monkeypatch,
        lambda *args, **kwargs: [
            vars(in_use[0])[name]
            for name in (
                "_class_rows",
                "_closing_maps",
                "_class_offsets",
                "_class_members",
                "_primitive_rows",
            )
        ],
    )

    def check_listing(
        listers: Sequence[_fault_search.FaultLister], fault_weight: int, violating: Collection[int]
    ) -> list[int]:
        """Check that the listers check the same faults, and return the syndromes checked."""
        results = []
        for lister in listers:
            listed.clear()
            in_use[:] = [lister]
            checked, result = distance_conftest.run_on_a_thread(
                functools.partial(_check_faults, lister, fault_weight, violating),
                timeout=_COMPILE_TIMEOUT,
            )
            totals = [sum(column) for column in zip(*listed, strict=True)] if listed else []
            results.append((checked, result, totals))
            assert not [each for each in threading.enumerate() if each.name == "qldpc-search"]
        assert results[0] == results[1]
        return results[0][0]

    # Syndromes of three bits repeat often, and errors on six qubits leave out some faults.
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(6):
        num_primitives = int(rng.integers(6, 12))
        flags = [int(flag) for flag in rng.integers(1, 16, size=num_primitives)]
        syndromes = [int(syndrome) for syndrome in rng.integers(8, size=num_primitives)]
        representatives = [int(error) for error in rng.integers(2**12, size=num_primitives)]
        num_starts = int(rng.integers(1, num_primitives + 1))
        for memory_limit in (0, 2**20):
            args = (flags, syndromes, representatives, 6, False, num_starts, memory_limit)
            listers = [
                _fault_search.FaultLister(*args),
                _fault_search.FaultLister(*args, num_threads),
            ]
            for fault_weight in range(2, 7):
                checked = check_listing(listers, fault_weight, ())
                if checked:
                    # Stop new listers at a violation, since a lister leaves out the syndromes
                    # that it has checked.
                    violating = {checked[int(rng.integers(len(checked)))]}
                    pair = [
                        _fault_search.FaultLister(*args),
                        _fault_search.FaultLister(*args, num_threads),
                    ]
                    check_listing(pair, fault_weight, violating)
    if split_lookups == 0:
        assert any(marked) and thread_calls
    assert thread_calls == [(False, True)] * len(thread_calls)


def test_walk_items_split_the_walk_in_order_into_few_items(pytestconfig: pytest.Config) -> None:
    """Work items cover the pairs of a start class and a later class in order, with their weights.

    The weight of an item is the number of ways to choose the other chosen classes after the second,
    summed over its range of second classes, and the items are at most the start classes plus
    ``_WALK_ITEMS_PER_THREAD`` for each thread, however many classes there are.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    cases = [(7, 7, 2, 3), (48, 16, 8, 8), (10, 3, 3, 2), (5, 5, 4, 2), (6, 1, 6, 2)]
    cases += [
        (int(q), int(rng.integers(1, q + 1)), int(rng.integers(2, 6)), int(rng.integers(2, 9)))
        for q in rng.integers(1, 40, size=10)
    ]
    for num_classes, num_starts, num_chosen, num_threads in cases:
        firsts, seconds, cumulative = _fault_search._get_walk_items(
            num_classes, num_starts, num_chosen, num_threads
        )
        assert len(firsts) <= num_starts + _fault_search._WALK_ITEMS_PER_THREAD * num_threads
        assert sorted(set(firsts.tolist())) == list(range(num_starts))
        items = list(zip(firsts.tolist(), seconds.tolist(), strict=True))
        assert items == sorted(set(items))
        weights = []
        for index, (first, second) in enumerate(items):
            starts_class = index == 0 or items[index - 1][0] != first
            assert (second == first + 1) == starts_class and second <= num_classes
            later = [other for each, other in items[index + 1 :] if each == first]
            stop = later[0] if later else num_classes
            weights.append(
                sum(
                    math.comb(num_classes - 1 - other, num_chosen - 2)
                    for other in range(second, stop)
                )
            )
        assert np.allclose(np.diff(cumulative, prepend=0.0), weights)

    # A walk with very many classes still has few items.
    firsts, _, _ = _fault_search._get_walk_items(131894, 127, 6, 8)
    assert len(firsts) <= 127 + _fault_search._WALK_ITEMS_PER_THREAD * 8


def test_entry_points_without_the_lock_are_named_in_the_requirements_on_threads() -> None:
    """The requirements on threads of ``fault_propagation.md`` name each entry point without lock.

    The entry points are the functions of this package compiled with ``nogil=True``, which threads
    run, and whose rules those requirements state.
    """
    guide = pathlib.Path(__file__).with_name("fault_propagation.md").read_text(encoding="utf-8")
    requirements = guide.split("\n### Requirements on threads\n", 1)[1].split("\n## ", 1)[0]
    entry_points = conftest.get_entry_points_without_the_lock(pathlib.Path(__file__).parent)
    assert entry_points
    for module, name in entry_points:
        assert f"`{name}`" in requirements, f"The requirements on threads omit {module}.{name}"


def test_listing_on_threads_keeps_the_order_whichever_range_ends_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lister checks the faults of one thread when a later range is listed before the first.

    The calling thread lists the first seven work items alone, and the thread of the first range of
    the others lists it only once a later range is listed.  Of two violations, one in each of these
    ranges, the lister finds the first, as one thread does.
    """
    monkeypatch.setattr(_fault_search, "_SPLIT_LOOKUPS_PER_THREAD", 2)
    monkeypatch.setattr(_threads, "RANGES_PER_THREAD", 2)
    later_listed = threading.Event()
    iterate_in_order = _fault_search.iterate_in_order

    def gated_iterate(
        num_items: int, num_threads: int, start_worker: Callable[..., Any], **kwargs: Any
    ) -> Iterator[npt.NDArray[np.int64]]:
        """Iterate in order, with the first range listed only once a later range is listed."""

        def gated_start(stop: threading.Event) -> Callable[[int, int], Any]:
            list_range = start_worker(stop)

            def gated_range(start: int, end: int) -> Iterator[npt.NDArray[np.int64]]:
                if start == 0:
                    assert later_listed.wait(timeout=THREAD_TIMEOUT)
                yield from list_range(start, end)
                if start:
                    later_listed.set()

            return gated_range

        return iterate_in_order(num_items, num_threads, gated_start, **kwargs)

    monkeypatch.setattr(_fault_search, "iterate_in_order", gated_iterate)
    # Seven classes of one member each, with flags 1 to 7 and distinct syndromes, make seven class
    # codewords of fault weight three.  On three threads, the work items are the pairs of a class
    # and a later one, and a last item after the last class.  The calling thread lists seven items,
    # with as many lookups, and the threads list the other 15, in ranges of four.  The class
    # codeword of classes 1, 3, and 5 is in the first range, and that of classes 4, 5, and 6, the
    # last, in the item of classes 4 and 5, the twelfth of the threads.  Errors on disjoint sets
    # of seven qubits leave out no fault.
    flags = [4, 5, 6, 7, 1, 2, 3]
    syndromes = [1 << jj for jj in range(7)]
    errors = [0b1111111 << (7 * jj) for jj in range(7)]
    args = (flags, syndromes, errors, 49, False, 7, 0)

    def check() -> None:
        checked, _ = _check_faults(_fault_search.FaultLister(*args), 3, ())
        first, last = 2 | 8 | 32, 16 | 32 | 64
        assert len(checked) == 7 and checked[-1] == last and first in checked
        for violating in ((), {first, last}):
            later_listed.clear()
            expected = _check_faults(_fault_search.FaultLister(*args), 3, violating)
            assert _check_faults(_fault_search.FaultLister(*args, 3), 3, violating) == expected
            assert later_listed.is_set()

    distance_conftest.run_on_a_thread(check, timeout=_COMPILE_TIMEOUT)


@pytest.mark.parametrize(("lookups_per_set", "expected_sizes"), [(0.59, [1, 2, 3]), (0.61, [1, 2])])
def test_listing_on_threads_counts_the_lookups_of_one_thread(
    lookups_per_set: float, expected_sizes: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    """On threads, a lister counts every lookup once, so it builds the closing maps of one thread.

    With classes of flags 1 to 7, a listing of fault weight four makes 21 lookups, one for each pair
    of classes.  The closing map of the 35 sets of three classes waits for 20.65 lookups, or for
    21.35, as ``_LOOKUPS_PER_CLOSING_SET`` is 0.59 or 0.61.  The calling thread makes seven of them
    alone.  So the map is built for fault weight five in the first case only if the threads' lookups
    are counted, and in the second only if lookups are counted more than once.
    """
    monkeypatch.setattr(_fault_search, "_SPLIT_LOOKUPS_PER_THREAD", 2)
    monkeypatch.setattr(_fault_search, "_LOOKUPS_PER_CLOSING_SET", lookups_per_set)
    sizes: list[int] = []
    get_closing_map = _fault_search._get_closing_map

    def record_closing_map(
        hashes: npt.NDArray[np.uint64], size: int
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        sizes.append(size)
        return get_closing_map(hashes, size)

    monkeypatch.setattr(_fault_search, "_get_closing_map", record_closing_map)
    flags = [1, 2, 3, 4, 5, 6, 7]
    syndromes = [1 << jj for jj in range(7)]
    errors = [0b1111111 << (7 * jj) for jj in range(7)]
    results = []

    def check() -> None:
        for num_threads in (1, 3):
            sizes.clear()
            lister = _fault_search.FaultLister(
                flags, syndromes, errors, 49, False, 7, 2**20, num_threads
            )
            checked = [_check_faults(lister, fault_weight, ()) for fault_weight in range(2, 6)]
            results.append((checked, list(sizes)))

    distance_conftest.run_on_a_thread(check, timeout=_COMPILE_TIMEOUT)
    assert results[0] == results[1]
    assert results[0][1] == expected_sizes


@pytest.mark.parametrize("split_lookups", [1, 5])
def test_only_costly_listings_move_to_threads(
    split_lookups: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listing starts threads only once the calling thread has made the lookups that split it.

    It also needs as many lookups left, as estimated from the weights of its work items.  With
    classes of flags 1 to 7, fault weight three, and three threads, the 21 pairs of classes take a
    lookup each.  A split of three lookups comes after three pairs, and 18 are left, so the listing
    starts threads.  A split of 15 comes after 15 pairs, in ranges of 1, 2, 4, and 8, and only six
    are left, so the calling thread lists them too.  Either way the lister checks the faults of one
    thread.
    """
    monkeypatch.setattr(_fault_search, "_SPLIT_LOOKUPS_PER_THREAD", split_lookups)
    iterations: list[int] = []
    iterate_in_order = _fault_search.iterate_in_order

    def count_iterations(*args: Any, **kwargs: Any) -> Iterator[npt.NDArray[np.int64]]:
        iterations.append(args[0])
        return iterate_in_order(*args, **kwargs)

    monkeypatch.setattr(_fault_search, "iterate_in_order", count_iterations)
    flags = [1, 2, 3, 4, 5, 6, 7]
    syndromes = [1 << jj for jj in range(7)]
    errors = [0b1111111 << (7 * jj) for jj in range(7)]
    args = (flags, syndromes, errors, 49, False, 7, 0)

    def check() -> None:
        expected = _check_faults(_fault_search.FaultLister(*args), 3, ())
        assert _check_faults(_fault_search.FaultLister(*args, 3), 3, ()) == expected

    distance_conftest.run_on_a_thread(check, timeout=_COMPILE_TIMEOUT)
    assert bool(iterations) == (split_lookups == 1)


def test_listing_on_threads_stops_its_threads_when_an_exception_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception raised while threads list, here by the oracle, stops them before it propagates.

    With 35 class codewords of fault weight three, in batches of one, and one held batch for each
    thread, the threads cannot finish while the calling thread takes no more, so only closing the
    listing stops them.
    """
    monkeypatch.setattr(_fault_search, "_SPLIT_LOOKUPS_PER_THREAD", 0)
    monkeypatch.setattr(_fault_search, "_CODEWORD_BATCH_SIZE", 1)
    monkeypatch.setattr(_fault_search, "_INITIAL_BATCH_SIZE", 1)
    monkeypatch.setattr(_threads, "OUTPUTS_PER_THREAD", 1)
    flags = list(range(1, 16))
    syndromes = [1 << jj for jj in range(15)]
    errors = [0b1111111 << (7 * jj) for jj in range(15)]
    lister = _fault_search.FaultLister(flags, syndromes, errors, 105, False, 15, 0, 3)

    def find_violation(syndromes: npt.NDArray[np.uint64], max_weight: int) -> int:
        raise ValueError("the oracle failed")

    # Record the batches of class tuples, to close them after the check if it fails.
    batches: list[Generator[npt.NDArray[np.int64], None, None]] = []
    iter_class_tuples = lister._iter_class_tuples

    def record_batches(fault_weight: int) -> Generator[npt.NDArray[np.int64], None, None]:
        batches.append(iter_class_tuples(fault_weight))
        return batches[-1]

    monkeypatch.setattr(lister, "_iter_class_tuples", record_batches)

    def list_faults() -> None:
        with pytest.raises(ValueError, match="the oracle failed"):
            try:
                lister.find_first(3, find_violation)
            finally:
                running = [each for each in threading.enumerate() if each.name == "qldpc-search"]
                for each in batches:
                    each.close()
                assert not running

    distance_conftest.run_on_a_thread(list_faults, timeout=_COMPILE_TIMEOUT)


def test_threads_of_a_listing_run_with_the_checks_of_the_rules_on_threads() -> None:
    """A thread of a listing that reads the tuning through ``get_tuning`` fails.

    ``conftest.py`` here gives the tests of this package the fixture ``check_code_on_threads`` of
    the distance package, which makes it fail.
    """

    def start_worker(stop: threading.Event) -> Callable[[int, int], Iterator[object]]:
        return lambda start, end: iter([_tuning.get_tuning()])

    def iterate() -> None:
        with pytest.raises(AssertionError, match="get_tuning"):
            list(_threads.iterate_in_order(2, 2, start_worker))

    distance_conftest.run_on_a_thread(iterate)


def test_certifications_list_on_the_threads_of_the_search_tuning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A certification lists faults on the number of threads of the search tuning.

    The circuit of ``test_flagged_circuit_with_permuted_signed_outputs`` in
    ``fault_propagation_test.py``, without its damaging fault, lists faults up to fault weight
    three.  With every listing moved to threads, the certificate on three threads is that of one.
    """
    monkeypatch.setattr(_fault_search, "_SPLIT_LOOKUPS_PER_THREAD", 0)
    thread_counts: list[int] = []
    iterator_thread_counts: list[int] = []
    fault_lister = _fault_search.FaultLister
    iterate_in_order = _fault_search.iterate_in_order

    def record_lister(*args: Any, **kwargs: Any) -> _fault_search.FaultLister:
        thread_counts.append(kwargs["num_threads"])
        return fault_lister(*args, **kwargs)

    def record_iterator_threads(*args: Any, **kwargs: Any) -> Iterator[npt.NDArray[np.int64]]:
        iterator_thread_counts.append(args[1])
        return iterate_in_order(*args, **kwargs)

    monkeypatch.setattr(_fault_search, "FaultLister", record_lister)
    monkeypatch.setattr(_fault_search, "iterate_in_order", record_iterator_threads)
    circuit = stim.Circuit()
    circuit.append("I", range(9))
    for flag in range(1, 16):
        for syndrome in (3, 5, 6):
            targets = [stim.target_x(jj) for jj in range(3) if (syndrome >> jj) & 1]
            targets += [stim.target_x(5 + jj) for jj in range(4) if (flag >> jj) & 1]
            circuit.append("CORRELATED_ERROR", targets, 0.001)
    circuit += stim.Circuit("H 0 1 2\nS 0 2\nX 0")
    circuit.append("M", range(5, 9))
    for jj in range(4):
        circuit.append("DETECTOR", [stim.target_rec(jj - 4)])

    def certify() -> list[circuits.BoundedFaultPropagationCertificate]:
        certificates = []
        for num_threads in (1, 3):
            with _tuning.use_tuning(num_threads=num_threads):
                certificates.append(
                    circuits.certify_bounded_fault_propagation(
                        circuit, output_qubits=[2, 0, 4, 1, 3], post_select=range(4), max_weight=3
                    )
                )
        return certificates

    one, three = distance_conftest.run_on_a_thread(certify, timeout=_COMPILE_TIMEOUT)
    assert thread_counts == [1, 3]
    assert iterator_thread_counts and set(iterator_thread_counts) == {3}
    assert one.certified and one == three


@pytest.mark.parametrize("size", [2, 3, 4])
def test_closing_map_peak_memory_is_within_its_estimate(
    size: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closing map needs at most about ``_BYTES_PER_SET``, and 4 per class, bytes for each set."""
    # tracemalloc traces only allocations by Python, so build the table with the Python versions of
    # the compiled functions.
    for name in (
        "_get_closing_table",
        "_place_sets",
        "_place_set",
        "_get_num_sets",
        "_advance_set",
        "_sort_bucket",
        "_get_key_slots",
        "_find_syndrome_slot",
    ):
        function = getattr(_fault_search, name)
        monkeypatch.setattr(_fault_search, name, getattr(function, "py_func", function))
    # Distinct random hashes leave no bucket to merge sort, and give the sets distinct keys, which
    # need the most slots.  Each size has about 20,000 sets.
    num_classes = {2: 200, 3: 50, 4: 25}[size]
    hashes = np.random.default_rng(0).integers(2**64, size=num_classes, dtype=np.uint64)
    # Build a small table first, so that loading the compiled code that the Python versions call
    # adds no allocations to the measurement.
    _fault_search._get_closing_map(hashes[: size + 1], size)
    was_tracing = tracemalloc.is_tracing()
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        _fault_search._get_closing_map(hashes, size)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if not was_tracing:
            tracemalloc.stop()
    # Allow a few kilobytes that do not grow with the number of sets.
    num_sets = math.comb(num_classes, size)
    assert peak - before <= num_sets * (_fault_search._BYTES_PER_SET + 4 * size) + 2**14


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
    pair_bytes = 3 * (_fault_search._BYTES_PER_SET + 8)
    syndrome_bytes = 40
    assert split(3, 2, pair_bytes) == (True, 0)
    assert split(3, 2, pair_bytes + 5 * syndrome_bytes) == (True, 5)
    assert split(3, 2, pair_bytes - 1) == (False, (pair_bytes - 1) // syndrome_bytes)


def test_closing_maps_take_room_that_the_table_does_not_need(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closing map takes room from the table of syndromes only if the table can grow as far."""
    take = _fault_search._take_table_room
    # Empty tables start with four rows.  A syndrome of one word takes 32 bytes, so 33 bytes take
    # the room of two.  A table of four rows with room for ten syndromes cannot grow, so it can give
    # up the room of six, and no more.
    monkeypatch.setattr(_fault_search, "_INITIAL_SYNDROME_TABLE_SIZE", 4)
    assert take(33, 10, 4, 1) == 8
    assert take(6 * 32, 10, 4, 1) == 4
    assert take(6 * 32 + 1, 10, 4, 1) == -1
    # With room for 100, a table of four rows can grow to 64 rows, since the old and the new table
    # of the last step take 96.  It can give up the room of four syndromes, but not of five.
    assert take(4 * 32, 100, 4, 1) == 96
    assert take(5 * 32, 100, 4, 1) == -1
    # A table that has grown to 64 rows can still give up only the room of four, since a table
    # emptied by a violation grows again from four rows.
    assert take(5 * 32, 100, 64, 1) == -1


def _listing_lister(num_classes: int, seed: int, memory_limit: int) -> _fault_search.FaultLister:
    """A lister of two members in each of ``num_classes`` classes with random flags of six bits.

    The syndromes are random and of 12 bits, so that the faults share some syndromes but fill a
    table of syndromes, and each representative has weight three, so that each fault with a new
    syndrome is checked.
    """
    rng = np.random.default_rng(seed)
    class_flags = rng.choice(np.arange(1, 64), size=num_classes, replace=False).tolist()
    flags = [int(flag) for flag in class_flags for _ in range(2)]
    syndromes = [int(syndrome) for syndrome in rng.integers(2**12, size=len(flags))]
    representatives = [0b111 << (3 * jj) for jj in range(len(flags))]
    lister = _fault_search.FaultLister(
        flags, syndromes, representatives, 3 * len(flags), False, len(flags), memory_limit
    )
    return lister


def test_closing_maps_do_not_change_the_checked_faults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Closing maps of three or four classes leave the checked faults as they are without them.

    In particular, a map does not take room that the table of syndromes, which grows from two rows
    here, would need to take each new syndrome as it does without the map, also after a violation
    empties the table.
    """
    monkeypatch.setattr(_fault_search, "_INITIAL_SYNDROME_TABLE_SIZE", 2)
    monkeypatch.setattr(_fault_search, "_LOOKUPS_PER_CLOSING_SET", 0)
    # With twelve classes, the map of pairs takes 3168 bytes, and those of three and four classes
    # take 11,440 and 27,720.  Each syndrome takes 40 bytes, and the listing has 1362 syndromes.
    # With room for 1536 or 3072 syndromes, the table can just grow to 1024 or 2048 rows, so that
    # any room that a map took from it would stop it one doubling short.  With 3072, the table
    # holds every syndrome only if it reaches 2048 rows.  With 1536, it fills up at 1024 rows, and
    # after a violation at fault weight six empties it, it must grow again to 1024 rows when that
    # weight is listed again, which a map that took room while the table had 1024 rows would stop.
    violating = set(range(0, 2**12, 31))
    for room in (1536, 3072):
        memory_limit = 3168 + 40 * room
        checked = []
        for max_closing_size in (2, 4):
            monkeypatch.setattr(_fault_search, "_MAX_CLOSING_SIZE", max_closing_size)
            lister = _listing_lister(12, 5, memory_limit)
            listings = [_check_faults(lister, weight, ()) for weight in range(2, 7)]
            listings.append(_check_faults(lister, 6, violating))
            listings.append(_check_faults(lister, 6, ()))
            checked.append(listings)
        assert checked[0] == checked[1], memory_limit


def test_one_lookup_can_list_many_class_codewords(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lookup in a closing map of three classes can find more class codewords than classes.

    The faults are then listed as with closing pairs.
    """
    # One class for each nonzero flag of five bits.  After the classes with flags 1 and 2, one
    # lookup finds the 112 triples of later classes that close class codewords of fault weight five,
    # more than the 31 classes.  Distinct syndromes, and representatives of weight three, check
    # every listed fault.
    monkeypatch.setattr(_fault_search, "_CODEWORD_BATCH_SIZE", 1)
    monkeypatch.setattr(_fault_search, "_LOOKUPS_PER_CLOSING_SET", 0)
    get_closing_map = _fault_search._get_closing_map
    sizes_built: set[int] = set()

    def record_closing_map(
        hashes: npt.NDArray[np.uint64], size: int
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        sizes_built.add(size)
        return get_closing_map(hashes, size)

    monkeypatch.setattr(_fault_search, "_get_closing_map", record_closing_map)
    flags = list(range(1, 32))
    syndromes = [1 << jj for jj in range(len(flags))]
    representatives = [0b111 << (3 * jj) for jj in range(len(flags))]
    checked = []
    for max_closing_size in (2, 3):
        monkeypatch.setattr(_fault_search, "_MAX_CLOSING_SIZE", max_closing_size)
        lister = _fault_search.FaultLister(
            flags, syndromes, representatives, 3 * len(flags), False, len(flags), 2**24
        )
        checked.append(_check_faults(lister, 5, ())[0])
    assert checked[0] == checked[1]
    assert 3 in sizes_built


def test_larger_closing_maps_wait_for_listings_with_many_lookups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closing map of three classes is built only after a listing with as many lookups as sets."""
    get_closing_map = _fault_search._get_closing_map
    sizes_built: list[int] = []

    def record_closing_map(
        hashes: npt.NDArray[np.uint64], size: int
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        sizes_built.append(size)
        return get_closing_map(hashes, size)

    monkeypatch.setattr(_fault_search, "_get_closing_map", record_closing_map)

    # Fault weight five may use the map of three of the twelve classes, which has 220 sets, if the
    # listing at the last fault weight made at least 220 lookups.
    for num_lookups, expected in ((2, [2]), (22_000, [3])):
        sizes_built.clear()
        lister = _listing_lister(12, 5, 2**20)
        lister._num_lookups = num_lookups
        _check_faults(lister, 5, ())
        assert sizes_built == expected, num_lookups

    # With sixteen classes, the map of three has 560 sets.  The listings with pairs look up at most
    # 120 choices of two classes at fault weight four.  At fault weight five, they look up at most
    # 560 choices of three, 553 here, since 7 triples are dependent, and at fault weight six 1697
    # choices of four.  So fault weight seven is the first to build the map of three.  Batches of
    # one class codeword make each listing pause and resume often, which must keep the count.
    monkeypatch.setattr(_fault_search, "_CODEWORD_BATCH_SIZE", 1)
    sizes_built.clear()
    lister = _listing_lister(16, 5, 2**20)
    for weight in range(2, 7):
        _check_faults(lister, weight, ())
    assert 3 not in sizes_built
    _check_faults(lister, 7, ())
    assert 3 in sizes_built


def test_closing_maps_and_the_table_of_syndromes_stay_within_the_memory_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The closing maps and the table of syndromes, also while it grows, fit the memory limit."""
    monkeypatch.setattr(_fault_search, "_INITIAL_SYNDROME_TABLE_SIZE", 2)
    monkeypatch.setattr(_fault_search, "_LOOKUPS_PER_CLOSING_SET", 0)
    get_closing_map = _fault_search._get_closing_map
    grow_syndrome_table = _fault_search._grow_syndrome_table

    # Record the memory of the closing maps of two or more classes as they are built, and the rows
    # of the table of syndromes, and check the memory in use after each step against the limit.  A
    # syndrome takes 40 bytes, and while the table grows, its old and new rows are both in use.
    def record_closing_map(
        hashes: npt.NDArray[np.uint64], size: int
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int32], npt.NDArray[np.int64]]:
        if size > 1:
            set_bytes = _fault_search._BYTES_PER_SET + 4 * size
            usage["maps"] += math.comb(len(hashes), size) * set_bytes
            assert usage["maps"] + 40 * usage["rows"] <= memory_limit
            sizes_built.add(size)
        return get_closing_map(hashes, size)

    def record_growth(
        syndrome_rows: npt.NDArray[np.uint64], num_syndromes: int
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
        assert usage["maps"] + 40 * 3 * len(syndrome_rows) <= memory_limit
        usage["rows"] = 2 * len(syndrome_rows)
        return grow_syndrome_table(syndrome_rows, num_syndromes)

    monkeypatch.setattr(_fault_search, "_get_closing_map", record_closing_map)
    monkeypatch.setattr(_fault_search, "_grow_syndrome_table", record_growth)

    # With twelve classes, the maps of two, three, and four classes take 3168, 11,440, and 27,720
    # bytes, the room of 286 and 693 syndromes of 40 bytes for the last two.  With room for 2480 or
    # 2515 syndromes beyond the map of pairs, the table grows to 1024 rows if the maps leave room
    # for 1536, since growing to 1024 rows also holds the old 512.  So the map of three fits in
    # both, and the map of four only in 2515, where the table, emptied by a violation at fault
    # weight six, grows again to 1024 rows with every map held, exactly at the limit.  At 2480, a
    # map of four is built only if the maps under-charge by at least 35 syndromes (1400 bytes) in
    # all, and the table's growth then exceeds the limit.  Leaving out 4 bytes per set of four
    # (about 50 syndromes), 4 bytes per set of either map (about 72), all 4 bytes per class (66 and
    # 198), or all the room (979) does so, and so does forgetting the room that an earlier map took.
    sizes_built: set[int] = set()
    for room in (2480, 2515):
        memory_limit = 3168 + 40 * room
        usage = {"maps": 0, "rows": 2}
        lister = _listing_lister(12, 5, memory_limit)
        for weight in range(2, 7):
            _check_faults(lister, weight, ())
        _check_faults(lister, 6, set(range(0, 2**12, 31)))
        _check_faults(lister, 6, ())
    assert sizes_built == {2, 3, 4}


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
