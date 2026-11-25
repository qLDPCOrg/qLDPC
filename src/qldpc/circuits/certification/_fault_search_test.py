# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact search for a violating fault, and its compiled listing of faults."""

from __future__ import annotations

import contextlib
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


def _class_codewords(
    class_flags: Sequence[int], fault_weight: int, num_start_classes: int
) -> list[tuple[int, ...]]:
    """By brute force, in lexicographic order, the class codewords that start with a start class.

    The classes have the distinct flags ``class_flags``, and the first ``num_start_classes`` of them
    are the start classes.  The class codewords are the sets of ``fault_weight`` classes whose flags
    sum to zero and have rank ``fault_weight - 1``.
    """
    class_codewords = []
    for classes in itertools.combinations(range(len(class_flags)), fault_weight):
        if classes[0] >= num_start_classes:
            break
        basis: list[int] = []
        for flag in (class_flags[cc] for cc in classes):
            for row in basis:
                flag = min(flag, flag ^ row)
            if flag:
                basis.append(flag)
        if _xor(class_flags, classes) == 0 and len(basis) == fault_weight - 1:
            class_codewords.append(classes)
    return class_codewords


def _take_class_tuples(
    lister: _fault_search.FaultLister, fault_weight: int, max_rows: int
) -> list[tuple[int, ...]]:
    """The first class tuples, at most ``max_rows`` of them, that a lister lists at a fault weight.

    Taking a bounded number of them makes a listing that does not end fail a test, not hang it.
    """
    rows = (
        tuple(row) for batch in lister._iter_class_tuples(fault_weight) for row in batch.tolist()
    )
    return list(itertools.islice(rows, max_rows))


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

    The faults are listed in the order of brute force, by class tuple and then by members.
    """
    # With zero hashes, each probe finds every class and must compare flags.  Small batches make a
    # listing pause and resume often.
    if collide:
        monkeypatch.setattr(
            _fault_search, "_hash_words", lambda words: np.zeros(len(words), np.uint64)
        )
    monkeypatch.setattr(_fault_search, "_INITIAL_BATCH_SIZE", 1)
    monkeypatch.setattr(_fault_search, "_BATCH_SIZE", 3)
    monkeypatch.setattr(_fault_search, "_CODEWORD_BATCH_SIZE", 1)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The first flags make an accepted fault {0, 1, 2, 3} that contains the accepted fault {0, 2}.
    # In the second, the classes with flags 1, 2, 4, 8, 3, and 12 have flags that sum to zero, but
    # 3 = 1 + 2, so these six classes are not a class codeword.  In the third, with two start
    # members, 0 and 1, the class with flags 1 has members 0 and 3, and the class with flags 2 is a
    # start class.  So the fault {1, 2, 3} takes member 3, which is not a start member, before
    # member 1, which is, and must not be skipped.  In the fourth, the classes with flags 1 and 2
    # are in two class codewords of fault weight four, with the classes with flags 4 and 7 and with
    # those with flags 5 and 6, whose order decides the order of the faults.  In the fifth, the six
    # classes have flags that sum to zero, but 5 = 1 + 4 and 10 = 2 + 8, so they are not a class
    # codeword, although each class is independent of the classes with flags 1 and 2.
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
        # The order matters, since the first violation in it is the result.
        for num_starts in (num_primitives, 2, 1):
            args = (flags, syndromes, errors, 70, False, num_starts, 0)
            lister = _fault_search.FaultLister(*args)
            for fault_weight in range(2, 7):
                expected = _listed_faults(flags, fault_weight, num_starts)
                checked, result = _check_faults(lister, fault_weight, ())
                assert result is None
                assert checked == [_xor(syndromes, fault) for fault in expected]
                if expected:
                    stop = int(rng.integers(len(expected)))
                    violation = (expected[stop], _xor(syndromes, expected[stop]))
                    # Use a new lister, since a lister skips the syndromes that it has listed.
                    _, result = _check_faults(
                        _fault_search.FaultLister(*args), fault_weight, {violation[1]}
                    )
                    assert result == violation


def test_the_walk_lists_each_class_codeword_once_in_lexicographic_order(
    pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk by shared flags lists the class codewords of brute force once, in order.

    The flags of the classes are dense, or sparse, each with one or two bits of up to ten, so that
    the walk has few candidates and forbids classes at several depths.  The 15 classes of every
    nonzero flag of four bits have class codewords with several classes that have the bit that the
    walk chooses at their root.  A walk that ignores forbidden classes lists these twice.  They also
    have sets of six classes whose flags sum to zero but have rank four, which a walk that skips its
    last test of independence lists.  In the other cases, the walk's array has two rows.  So a full
    array makes it return the class codewords of earlier roots before it finishes a root, and then
    move those of the root to the start of the array, and it makes several passes over a root with
    more than two class codewords.  Each case is listed by a walk that pauses after each probe of
    work, as on threads, and resumes, and then by the lister on one thread.  The paused walk may
    return a bounded number of batches and class codewords, and both listings run on a thread of
    their own with a timeout, so that a listing that does not end fails rather than hangs.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    cases = [(list(range(1, 16)), 1, 6, _fault_search._MAX_ROOT_CODEWORDS)]
    for dense in [False, True] * 4:
        num_bits = int(rng.integers(4, 11))
        values = [value for value in range(1, 2**num_bits) if dense or value.bit_count() <= 2]
        num_classes = min(len(values), int(rng.integers(5, 13)))
        flags = [int(flag) for flag in rng.choice(values, size=num_classes, replace=False)]
        cases.append((flags, int(rng.integers(num_classes + 1)), 6, 2))
    for flags, num_starts, max_fault_weight, max_root_codewords in cases:
        monkeypatch.setattr(_fault_search, "_MAX_ROOT_CODEWORDS", max_root_codewords)
        lister = _class_lister(flags, num_starts)
        for fault_weight in range(3, max_fault_weight + 1):
            expected = _class_codewords(flags, fault_weight, num_starts)
            walk = _fault_search._SharedFlagWalk(
                lister._class_rows, lister._bit_index, lister._class_map, fault_weight
            )
            listed = distance_conftest.run_on_a_thread(
                functools.partial(_list_paused, walk, lister._num_start_classes, len(expected)),
                timeout=_COMPILE_TIMEOUT,
            )
            assert listed == expected
            listed = distance_conftest.run_on_a_thread(
                functools.partial(_take_class_tuples, lister, fault_weight, len(expected) + 1),
                timeout=_COMPILE_TIMEOUT,
            )
            assert listed == expected


def _list_paused(
    walk: _fault_search._SharedFlagWalk, num_roots: int, max_rows: int
) -> list[tuple[int, ...]]:
    """List the first roots with a walk that pauses after each probe of work, and resumes.

    The walk may return at most ``10**5`` batches and ``max_rows`` class codewords.
    """
    rows: list[tuple[int, ...]] = []
    for num_batches, batch in enumerate(walk.list_range(0, num_roots, 1)):
        rows.extend(tuple(row) for row in batch.tolist())
        assert num_batches < 10**5 and len(rows) <= max_rows
    return rows


def _class_lister(flags: Sequence[int], num_starts: int) -> _fault_search.FaultLister:
    """A lister of faults with these flags, distinct syndromes, and representatives of weight 3."""
    syndromes = [1 << jj for jj in range(len(flags))]
    errors = [0b111 << (3 * jj) for jj in range(len(flags))]
    return _fault_search.FaultLister(flags, syndromes, errors, 3 * len(flags), False, num_starts, 0)


def _list_class_tuples(
    flags: Sequence[int], num_starts: int, fault_weight: int, max_rows: int
) -> list[tuple[int, ...]]:
    """The first class tuples, at most ``max_rows``, of a lister of faults with these flags."""
    return _take_class_tuples(_class_lister(flags, num_starts), fault_weight, max_rows)


@pytest.mark.parametrize("collide", [False, True])
def test_a_class_codeword_that_two_states_could_find_is_listed_once(
    collide: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk by shared flags lists a class codeword once, also with members and hashes shared.

    Of the classes 0 to 3 with flags 4, 5, 6, and 7, the walk from root 0 finds the class codeword
    of all four at the state of classes 0 and 1, and could find it again at the state of classes 0
    and 2, where class 1 is forbidden ("Listing class codewords by shared flags" in
    ``fault_propagation.md``).  Classes of several members, and zero hashes, which make every class
    share each probe's hash, change nothing, and with no start class nothing is listed.
    """
    if collide:
        monkeypatch.setattr(
            _fault_search, "_hash_words", lambda words: np.zeros(len(words), np.uint64)
        )
    assert _list_class_tuples([4, 5, 6, 7], 1, 4, 2) == [(0, 1, 2, 3)]
    assert _list_class_tuples([4, 5, 4, 6, 7, 5, 6], 1, 4, 2) == [(0, 1, 2, 3)]
    assert _list_class_tuples([4, 5, 6, 7], 0, 4, 1) == []


@pytest.mark.parametrize(
    ("max_root_codewords", "more_passes"), [(2, True), (27, True), (29, False)]
)
def test_a_root_whose_class_codewords_fill_its_array_takes_more_passes(
    max_root_codewords: int, more_passes: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The walk by shared flags makes another pass over a root whose class codewords fill its array.

    Among the classes of every nonzero flag of four bits, the roots of flags 1, 2, and 3 have 28,
    22, and 16 class codewords of fault weight four.  An array of 27 rows fills with class codewords
    of the first root, so the walk keeps the first 13 of them, releases those before its ceiling,
    and makes another pass for the rest, while an array of 29 rows holds them all, and makes the
    walk return those of the first root once the first class codeword of the second fills it.  The
    listing is that of brute force either way.
    """
    # Record the work of the walk, which another pass over a root adds to.
    works: list[int] = []
    list_by_shared_flags = _fault_search._list_by_shared_flags

    def record_work(*args: Any) -> int:
        num_codewords = list_by_shared_flags(*args)
        works.append(int(args[-2][4]))
        return num_codewords

    monkeypatch.setattr(_fault_search, "_list_by_shared_flags", record_work)
    flags = list(range(1, 16))
    expected = _class_codewords(flags, 4, 3)
    assert _list_class_tuples(flags, 3, 4, len(expected) + 1) == expected
    one_pass_work = works[-1]
    monkeypatch.setattr(_fault_search, "_MAX_ROOT_CODEWORDS", max_root_codewords)
    assert _list_class_tuples(flags, 3, 4, len(expected) + 1) == expected
    assert (works[-1] > one_pass_work) == more_passes


def test_listers_reject_zero_flags() -> None:
    """A lister rejects a primitive fault with zero flags, which no flag class can hold."""
    with pytest.raises(ValueError, match="nonzero flags"):
        _fault_search.FaultLister([1, 0], [1, 2], [1, 2], 2, False, 1, 0)


def test_the_map_of_classes_sorts_the_classes_stably_by_hash(pytestconfig: pytest.Config) -> None:
    """The map of classes by hash lists the classes by hash, those of each hash in order."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_classes in (0, 1, 2, 5, 40):
        # Random hashes, and hashes from four values, so that many classes share one.
        for hashes in (
            rng.integers(2**64, size=num_classes, dtype=np.uint64),
            rng.integers(2**64, size=4, dtype=np.uint64)[rng.integers(4, size=num_classes)],
        ):
            expected = sorted(range(num_classes), key=lambda each: (int(hashes[each]), each))
            keys, classes, slots = _fault_search._get_class_map(hashes)
            assert classes.tolist() == expected
            assert keys.tolist() == [int(hashes[each]) for each in expected]
            assert slots.tolist() == _fault_search._get_key_slots(keys).tolist()


def test_repeated_syndromes_are_checked_once() -> None:
    """A fault with no error, or whose syndrome is already in the table, is not checked."""
    # On |000>, the faults {0, 1} and {0, 2} leave X on all qubits, and {1, 2} leaves no error.
    lister = _fault_search.FaultLister([1, 1, 1], [7, 0, 0], [7, 0, 0], 3, False, 3, 0)
    assert _check_faults(lister, 2, ()) == ([7], None)


@pytest.mark.parametrize("css_weight", [False, True])
@pytest.mark.parametrize("z_part", [False, True])
def test_representatives_are_weighed_over_every_word(css_weight: bool, z_part: bool) -> None:
    """A fault is checked if the product of its representatives, over all words, is too heavy."""
    # On |0>^65 with X representatives, or on |+>^65 with Z representatives, the product of the two
    # representatives acts on qubits 0, 63, and 64, so the fault of weight two violates the bound.
    num_qubits = 65
    syndromes = [1 << 63 | 1 << 64, 1]
    representatives = [syndrome << (num_qubits if z_part else 0) for syndrome in syndromes]
    lister = _fault_search.FaultLister(
        [1, 1], syndromes, representatives, num_qubits, css_weight, 2, 2**20
    )
    syndrome = syndromes[0] ^ syndromes[1]
    assert _check_faults(lister, 2, {syndrome}) == ([syndrome], ((0, 1), syndrome))


@pytest.mark.parametrize("css_weight", [False, True])
def test_representatives_are_weighed_by_the_pauli_weight_of_the_search(css_weight: bool) -> None:
    """The product of the representatives has the CSS weight or the physical weight, as asked."""
    # On |0>^64 |+>, X on qubits 0 and 63 times Z on qubit 64 has X weight two in the first word and
    # Z weight one in the second, so CSS weight two, at most the fault weight, and physical weight
    # three.
    num_qubits = 65
    syndromes = [1 | 1 << 63, 1 << 64]
    representatives = [1 | 1 << 63, 1 << (64 + num_qubits)]
    lister = _fault_search.FaultLister(
        [1, 1], syndromes, representatives, num_qubits, css_weight, 2, 2**20
    )
    syndrome = syndromes[0] ^ syndromes[1]
    assert _check_faults(lister, 2, ()) == ([] if css_weight else [syndrome], None)


def test_representative_weight_is_the_pauli_weight_of_the_product(
    pytestconfig: pytest.Config,
) -> None:
    """The weight of the chosen members is that of the product of their representatives."""
    # Random rows, with two or three words for each of the X and Z parts, often set bit 63 of a
    # word and several bits past the first word.  The first primitive fault sets them in every word
    # of its X and Z parts, and is also chosen alone.  Columns 0 and 1, the hash and syndrome, are
    # not read.
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_primitives = 6
    members = np.arange(num_primitives, dtype=np.int64)
    for num_qubit_words in (2, 3):
        rows = rng.integers(2**64, size=(num_primitives, 2 + 2 * num_qubit_words), dtype=np.uint64)
        rows[0, 2:] |= np.uint64(1 << 63 | 0b111)
        x_parts, z_parts = (
            [int.from_bytes(words.astype("<u8").tobytes(), "little") for words in rows[:, columns]]
            for columns in (slice(2, 2 + num_qubit_words), slice(2 + num_qubit_words, None))
        )

        choices = [[0]] + [
            rng.choice(num_primitives, size=int(rng.integers(1, 5)), replace=False).tolist()
            for _ in range(10)
        ]
        for chosen in choices:
            x = z = 0
            for member in chosen:
                x ^= x_parts[member]
                z ^= z_parts[member]
            entries = np.array(chosen, dtype=np.int64) + 1
            for css_weight in (False, True):
                expected = max(x.bit_count(), z.bit_count()) if css_weight else (x | z).bit_count()
                weight = _fault_search._get_representative_weight(
                    rows, members, entries, 2, css_weight
                )
                assert weight == expected


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
@pytest.mark.parametrize("max_root_codewords", [2, 2**20])
@pytest.mark.parametrize("split_probes", [0, 1])
def test_listing_on_threads_checks_the_faults_of_one_thread(
    num_threads: int,
    max_root_codewords: int,
    split_probes: int,
    pytestconfig: pytest.Config,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On threads, a lister checks the faults of one thread, in order, with the same work.

    Listings move to threads at once, or after the calling thread has listed alone a probe of work
    for each thread, with batches of one class codeword, calls of about one probe of work, and one
    held batch for each thread, so that the threads pause within passes over roots and wait often.
    The threads hand on no empty batch.  With an array of two class codewords, the walk makes
    several passes over some roots.  Fixed instances check that a thread pauses within a pass and
    makes another pass over a root, that the calling thread lists alone first when the work splits
    after a probe, and that a violation found while threads list closes the listing as on one
    thread and leaves no thread running.  The same holds with a full table of syndromes.  The
    threads run with read-only views of the arrays of the lister that they read, and leave them
    unchanged.  The calling thread consumes the listing on a thread of its own, so that a deadlock
    fails the test.
    """
    monkeypatch.setattr(_fault_search, "_MAX_ROOT_CODEWORDS", max_root_codewords)
    monkeypatch.setattr(_fault_search, "_SPLIT_PROBES_PER_THREAD", split_probes)
    monkeypatch.setattr(_fault_search, "_CODEWORD_BATCH_SIZE", 1)
    monkeypatch.setattr(_fault_search, "_PROBES_PER_CALL", 1)
    monkeypatch.setattr(_threads, "OUTPUTS_PER_THREAD", 1)
    # With no memory, the table of syndromes keeps room for its first two syndromes.
    monkeypatch.setattr(_fault_search, "_INITIAL_SYNDROME_TABLE_SIZE", 2)

    # Record the work and the class codewords of each call of the walk, and, for a call on a thread
    # of the listing, whether the rows of the classes were writeable and whether it did at most
    # about one probe of work, which a call passes by at most a step: a candidate, a reduction, and
    # the bits of a word of flags.  For a lister on several threads, also record whether the
    # calling thread or a thread listed, and whether a thread paused within a pass over a root or
    # made another pass over a root.
    listed: list[tuple[int, int]] = []
    thread_calls: list[tuple[bool, bool]] = []
    events: set[str] = set()
    list_by_shared_flags = _fault_search._list_by_shared_flags
    max_work = 1 + _fault_search._PROBES_PER_REDUCTION + 64

    def record_listing(*args: Any) -> int:
        walk_state = args[-2]
        work = int(walk_state[4])
        num_codewords = list_by_shared_flags(*args)
        work = int(walk_state[4]) - work
        listed.append((work, num_codewords))
        on_thread = threading.current_thread().name == "qldpc-search"
        if on_thread:
            thread_calls.append((args[0].flags.writeable, work <= max_work))
        if in_use[0]._num_threads > 1:
            events.add("listed on a thread" if on_thread else "listed alone")
            if on_thread and walk_state[2] >= 0:
                events.add("paused within a pass")
            if on_thread and walk_state[8]:
                events.add("another pass on a thread")
        return num_codewords

    monkeypatch.setattr(_fault_search, "_list_by_shared_flags", record_listing)

    # Check that the threads pass on batches of at most ``_CODEWORD_BATCH_SIZE`` class codewords,
    # although a call of the walk by shared flags can return the two of a root, and no empty batch,
    # although most calls of one probe of work return none.
    iterate_in_order = _fault_search.iterate_in_order

    def check_batches(*args: Any, **kwargs: Any) -> Iterator[npt.NDArray[np.int64]]:
        with contextlib.closing(iterate_in_order(*args, **kwargs)) as batches:
            for batch in batches:
                assert len(batch) == 1
                try:
                    yield batch
                except GeneratorExit:
                    if any(each.name == "qldpc-search" for each in threading.enumerate()):
                        events.add("closed while threads list")
                    raise

    monkeypatch.setattr(_fault_search, "iterate_in_order", check_batches)

    # Mark the arrays of the lister in use that the threads read read-only while they list.
    def get_shared_arrays(*args: Any, **kwargs: Any) -> list[object]:
        names = (
            "_class_rows",
            "_bit_index",
            "_class_map",
            "_class_offsets",
            "_class_members",
            "_primitive_rows",
        )
        return [vars(in_use[0])[name] for name in names]

    in_use: list[_fault_search.FaultLister] = []
    marked = distance_conftest.check_shared_arrays_on_threads(monkeypatch, get_shared_arrays)

    def check_listing(
        listers: Sequence[_fault_search.FaultLister],
        fault_weight: int,
        violating: Collection[int],
        same_work: bool = True,
    ) -> list[int]:
        """Check that the listers check the same faults, and return the syndromes checked.

        With ``same_work``, also check that they list the same class codewords with the same work.
        Threads can list past a violation that ends a listing early, so then they need not.
        """
        results = []
        for lister in listers:
            listed.clear()
            in_use[:] = [lister]
            checked, result = distance_conftest.run_on_a_thread(
                functools.partial(_check_faults, lister, fault_weight, violating),
                timeout=_COMPILE_TIMEOUT,
            )
            totals = (sum(work for work, _ in listed), sum(number for _, number in listed))
            results.append((checked, result, totals if same_work else None))
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
    if split_probes == 0:
        assert any(marked) and thread_calls

    # List a fixed instance: the classes of flags 4, 5, 6, 7, 1, 2, and 3, of one member each, at
    # fault weight three, whose seven class codewords each make one fault with a syndrome of its
    # own.  The first root has three class codewords, more than an array of two rows holds.  When
    # the work splits at once, threads list every root, and so make another pass over the first
    # root, pausing within it.  When it splits after a probe, the calling thread lists the first
    # root alone, and threads list the others.
    flags = [4, 5, 6, 7, 1, 2, 3]
    errors = [0b1111111 << (7 * jj) for jj in range(7)]
    args = (flags, [1 << jj for jj in range(7)], errors, 49, False, 7, 0)
    events.clear()
    checked = check_listing(
        [_fault_search.FaultLister(*args), _fault_search.FaultLister(*args, num_threads)], 3, ()
    )
    assert len(checked) == 7 and "listed on a thread" in events
    if split_probes == 0 and max_root_codewords == 2:
        assert {"another pass on a thread", "paused within a pass"} <= events
    assert ("listed alone" in events) == (split_probes == 1)

    # The classes of every nonzero flag of four bits, of one member each, make 35 class codewords
    # of fault weight three, seven of them of the first root.  Check each fault on its own, and stop
    # at the tenth, a class codeword of the second root, so that the calling thread closes the
    # listing while threads list.  They cannot finish: the 25 class codewords after it are batches
    # of their own, more than the at most eight batches that the threads may hold while the calling
    # thread takes no more (see ``iterate_in_order``).
    flags = list(range(1, 16))
    errors = [0b1111111 << (7 * jj) for jj in range(15)]
    args = (flags, [1 << jj for jj in range(15)], errors, 105, False, 15, 0)
    listers = [_fault_search.FaultLister(*args), _fault_search.FaultLister(*args, num_threads)]
    checked = check_listing(listers, 3, ())
    assert len(checked) == 35
    monkeypatch.setattr(_fault_search, "_INITIAL_BATCH_SIZE", 1)
    monkeypatch.setattr(_fault_search, "_BATCH_SIZE", 1)
    events.clear()
    pair = [_fault_search.FaultLister(*args), _fault_search.FaultLister(*args, num_threads)]
    assert len(check_listing(pair, 3, {checked[9]}, same_work=False)) == 10
    assert "closed while threads list" in events
    assert thread_calls == [(False, True)] * len(thread_calls)


def test_a_thread_of_a_listing_hands_on_no_empty_batch_and_stops_when_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thread of a listing hands on only nonempty batches, and returns once its stop event is set.

    With calls of one probe of work, most calls of the walk list no class codeword.  The function
    that a listing on threads gives ``iterate_in_order`` for each thread lists the range of all the
    roots, as one thread would, in batches that are never empty.  With its stop event set, it
    returns after one call of the walk, which lists no class codeword.
    """
    monkeypatch.setattr(_fault_search, "_SPLIT_PROBES_PER_THREAD", 0)
    monkeypatch.setattr(_fault_search, "_PROBES_PER_CALL", 1)
    start_workers: list[Callable[..., Any]] = []
    iterate_in_order = _fault_search.iterate_in_order

    def record_start_worker(
        num_items: int, num_threads: int, start_worker: Callable[..., Any], **kwargs: Any
    ) -> Iterator[npt.NDArray[np.int64]]:
        start_workers.append(start_worker)
        return iterate_in_order(num_items, num_threads, start_worker, **kwargs)

    calls: list[int] = []
    list_by_shared_flags = _fault_search._list_by_shared_flags

    def count_calls(*args: Any) -> int:
        calls.append(1)
        return list_by_shared_flags(*args)

    monkeypatch.setattr(_fault_search, "iterate_in_order", record_start_worker)
    monkeypatch.setattr(_fault_search, "_list_by_shared_flags", count_calls)
    flags = list(range(1, 16))
    syndromes = [1 << jj for jj in range(15)]
    errors = [0b111 << (3 * jj) for jj in range(15)]
    lister = _fault_search.FaultLister(flags, syndromes, errors, 45, False, 15, 0, 3)
    expected = _class_codewords(flags, 3, 15)
    listed = distance_conftest.run_on_a_thread(
        functools.partial(_take_class_tuples, lister, 3, len(expected) + 1),
        timeout=_COMPILE_TIMEOUT,
    )
    assert listed == expected and len(start_workers) == 1

    # Run the function of a thread on this thread, as a thread would.
    batches = list(start_workers[0](threading.Event())(0, lister._num_start_classes))
    assert all(len(batch) for batch in batches)
    assert [tuple(row) for batch in batches for row in batch.tolist()] == expected
    stop = threading.Event()
    stop.set()
    calls.clear()
    assert not list(start_workers[0](stop)(0, lister._num_start_classes))
    assert len(calls) == 1


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

    The calling thread lists the first root alone, and the thread of the first range of the other
    roots lists it only once a later range is listed.  Of two violations, one in each of these
    ranges, the lister finds the first, as one thread does.
    """
    monkeypatch.setattr(_fault_search, "_get_root_weights", lambda q, s, m: np.full(s, 1.0))
    monkeypatch.setattr(_fault_search, "_PROBES_PER_REDUCTION", 4)
    monkeypatch.setattr(_fault_search, "_SPLIT_PROBES_PER_THREAD", 2)
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
    # Seven classes of one member each, with flags 4, 5, 6, 7, 1, 2, and 3 and distinct syndromes,
    # make seven class codewords of fault weight three.  On three threads, a split comes after six
    # probes.  The root 0, of flags 4, takes four probes for its reduction, a probe for its one bit,
    # and a probe and a reduction of four probes for each of the three classes after it with bit 2,
    # so the calling thread lists it alone.  The threads list the six roots of equal weights left in
    # ranges of two.  The class codeword of classes 1, 3, and 5 is in the first range, and that of
    # classes 4, 5, and 6, the last, in the second.  Errors on disjoint sets of seven qubits leave
    # out no fault.
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


@pytest.mark.parametrize("split_probes", [1, 200])
def test_only_costly_listings_move_to_threads(
    split_probes: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A listing starts threads only once the calling thread has done the work that splits it.

    It also needs as much work left, as estimated from the weights of its roots.  With classes of
    flags 1 to 7, fault weight three, and three threads, the roots 0 to 6 weigh in proportion to 6,
    5, 4, 3, 2, 1, and 0, and take 260, 196, 196, 68, 67, 67, and 67 probes of work: 64 for the
    root, one for each bit of the root and each candidate, and 64 for each class codeword.  A split
    of the work of three probes comes after the first root, whose work, times 15 for the weight
    left over 6 for its own, exceeds the split, so the listing starts threads.  A split of 600
    comes after the first three roots, in ranges of 1 and 2, whose 652 probes, times 6 left over
    their 15, fall far short of it, so the calling thread lists the other roots too.  Either way,
    the lister checks the faults of one thread.
    """
    monkeypatch.setattr(_fault_search, "_SPLIT_PROBES_PER_THREAD", split_probes)
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
    assert bool(iterations) == (split_probes == 1)


def test_listing_on_threads_stops_its_threads_when_an_exception_is_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception raised while threads list, here by the oracle, stops them before it propagates.

    With 35 class codewords of fault weight three, of seven roots, in batches of one class codeword,
    and one held batch for each thread, the threads cannot finish while the calling thread takes no
    more, so only closing the listing stops them.
    """
    monkeypatch.setattr(_fault_search, "_SPLIT_PROBES_PER_THREAD", 0)
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
    monkeypatch.setattr(_fault_search, "_SPLIT_PROBES_PER_THREAD", 0)
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


def _listing_lister(memory_limit: int) -> _fault_search.FaultLister:
    """A lister of two members in each of 12 classes with random flags of six bits.

    The syndromes are random and of 12 bits, so that the faults share some syndromes but fill a
    table of syndromes, and each representative has weight three, so that each fault with a new
    syndrome is checked.
    """
    rng = np.random.default_rng(5)
    class_flags = rng.choice(np.arange(1, 64), size=12, replace=False).tolist()
    flags = [int(flag) for flag in class_flags for _ in range(2)]
    syndromes = [int(syndrome) for syndrome in rng.integers(2**12, size=len(flags))]
    representatives = [0b111 << (3 * jj) for jj in range(len(flags))]
    lister = _fault_search.FaultLister(
        flags, syndromes, representatives, 3 * len(flags), False, len(flags), memory_limit
    )
    return lister


def test_the_table_of_syndromes_stays_within_the_memory_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The table of syndromes, also while it grows, fits the memory limit, and grows up to it.

    Twelve classes of two members each, with random flags of six bits and random syndromes of 12
    bits, list 1362 distinct syndromes up to fault weight six.  A syndrome of the 72 output qubits
    takes two words, and so 40 bytes in the table (see ``FaultLister``), and
    while the table grows, its old and new rows are both in use.  So, from two rows, the table
    grows to 1024 rows with room for 1536 syndromes, which also holds the old 512, but not with room
    for 1535.  After a violation empties the table, it grows again as far.
    """
    monkeypatch.setattr(_fault_search, "_INITIAL_SYNDROME_TABLE_SIZE", 2)
    grow_syndrome_table = _fault_search._grow_syndrome_table
    sizes: list[int] = []

    def record_growth(
        syndrome_rows: npt.NDArray[np.uint64], num_syndromes: int
    ) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.int64]]:
        assert 40 * 3 * len(syndrome_rows) <= memory_limit
        sizes.append(2 * len(syndrome_rows))
        return grow_syndrome_table(syndrome_rows, num_syndromes)

    monkeypatch.setattr(_fault_search, "_grow_syndrome_table", record_growth)
    for room, largest_size in ((1535, 512), (1536, 1024)):
        memory_limit = 40 * room
        lister = _listing_lister(memory_limit)
        for weight in range(2, 7):
            _check_faults(lister, weight, ())
        assert max(sizes) == largest_size
        sizes.clear()
        _check_faults(lister, 6, set(range(0, 2**12, 31)))
        _check_faults(lister, 6, ())
        assert max(sizes) == largest_size
        sizes.clear()


def test_sorting_a_full_array_of_class_codewords_takes_about_as_much_memory_again() -> None:
    """Sorting ``n`` rows of ``m`` classes takes at most ``(8 m + 16) n`` bytes beyond the rows.

    This is the memory that ``_MAX_ROOT_CODEWORDS`` and the docstring of ``memory_limit`` in
    ``fault_propagation.py`` state for each thread that lists: an array of ``2**20`` rows of
    ``8 m`` bytes, and as much again and 16 MiB more to sort it.  tracemalloc does not see the
    allocations of compiled code, so the test sorts with the Python version of the compiled
    function.
    """
    sort_rows = getattr(_fault_search._sort_rows, "py_func", _fault_search._sort_rows)
    num_rows = 2**14
    rng = np.random.default_rng(0)
    for fault_weight in (3, 6):
        rows = rng.integers(num_rows, size=(num_rows, fault_weight))
        expected = sorted(map(tuple, rows.tolist()))
        was_tracing = tracemalloc.is_tracing()
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            sort_rows(rows)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            if not was_tracing:
                tracemalloc.stop()
        assert list(map(tuple, rows.tolist())) == expected
        # Allow a few kilobytes that do not grow with the number of rows.
        assert peak - before <= num_rows * (8 * fault_weight + 16) + 2**14


def test_the_table_of_syndromes_has_room_for_at_most_2_to_the_31_syndromes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The room of the table of syndromes is the memory limit over 40 bytes, at most 2**31.

    A syndrome of two words takes 40 bytes.  The table's slots, twice as many as its rows, must
    stay within 2**32, as ``_find_syndrome_slot`` needs.
    """
    rooms: list[int] = []
    table_should_grow = _fault_search._table_should_grow

    def record_room(num_syndromes: int, num_rows: int, max_syndromes: int) -> bool:
        rooms.append(max_syndromes)
        return table_should_grow(num_syndromes, num_rows, max_syndromes)

    # List once before wrapping, so that numba compiles the expansion, which also calls
    # _table_should_grow, with the compiled function rather than the wrapper.
    _check_faults(_listing_lister(2**50), 3, ())
    monkeypatch.setattr(_fault_search, "_table_should_grow", record_room)
    for memory_limit, room in (
        (40 * 2**31 - 1, 2**31 - 1),
        (40 * 2**31, 2**31),
        (40 * 2**31 + 40, 2**31),
        (2**50, 2**31),
    ):
        rooms.clear()
        _check_faults(_listing_lister(memory_limit), 3, ())
        assert rooms and set(rooms) == {room}


def test_roots_are_weighed_by_the_choices_of_their_other_classes() -> None:
    """The weights of the roots are in proportion to ``C(q - 1 - r, m - 2)``, and stay finite."""
    for num_classes, fault_weight in itertools.product(range(12), range(3, 9)):
        for num_roots in range(num_classes + 1):
            weights = _fault_search._get_root_weights(num_classes, num_roots, fault_weight)
            expected = [
                math.comb(num_classes - 1 - root, fault_weight - 2) for root in range(num_roots)
            ]
            scale = math.comb(num_classes - 1, fault_weight - 2) if num_classes else 0
            if scale:
                np.testing.assert_allclose(weights * scale, expected, rtol=1e-12)
            else:
                assert not weights.any()
    # About 10**420 choices of the classes of the first root, more than a float holds.
    weights = _fault_search._get_root_weights(10**4, 3, 200)
    assert np.isfinite(weights).all() and weights[0] == 1 and 0 < weights[2] < weights[1] < 1


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
    """``max_weight``, even one that exceeds 64 bits, limits the fault weight of checked faults."""
    # The only accepted fault is that of all three primitive faults.
    failures = {
        max_weight: _certify_map([1, 2, 3], [1, 2, 12], max_weight, 4)
        for max_weight in (None, 2**70, 2)
    }
    assert failures[2] is None and failures[None] is not None
    assert failures[2**70] == failures[None]


def test_memory_limit_does_not_change_results(
    pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With almost no memory, the search finds the same counterexample as with ample memory.

    The table of syndromes starts with room for two, so that it grows with ample memory, and fills
    without.
    """
    monkeypatch.setattr(_fault_search, "_INITIAL_SYNDROME_TABLE_SIZE", 2)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The first map has flags of rank four, and its only accepted fault has fault weight five and
    # residual weight six.
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
