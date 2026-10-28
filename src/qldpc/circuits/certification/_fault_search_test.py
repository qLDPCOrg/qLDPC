# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact search for a violating fault, and its compiled listing of faults."""

from __future__ import annotations

import functools
import itertools
import operator
import tracemalloc
from collections.abc import Collection, Sequence
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
import stim

from . import _fault_search


@pytest.fixture(params=[False, True], ids=["compiled", "python"])
def kernels(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the compiled listing, or its Python version for coverage."""
    if request.param:
        names = (
            "_iter_irreducible_fault_batches",
            "_reduce_flags",
            "_pauli_weight",
            "_find_syndrome",
            "_grow_syndrome_table",
            "_get_closing_table",
            "_sort_bucket",
        )
        for name in names:
            monkeypatch.setattr(_fault_search, name, getattr(_fault_search, name).py_func)


def _irreducible_faults(
    flags: Sequence[int], fault_weight: int, num_starts: int
) -> list[tuple[int, ...]]:
    """Irreducible accepted faults of a fault weight that start among the first ``num_starts``.

    They are found by brute force, in lexicographic order.
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


@pytest.mark.usefixtures("kernels")
@pytest.mark.parametrize("collide", [False, True])
def test_listing_matches_brute_force(
    collide: bool, pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each fault is listed once and in lexicographic order, with or without a table of pairs.

    Flags and syndromes span several words.  If ``collide``, every hash is zero, so that lookups
    find every entry of a table and must compare flags.  The batches are small, so that a listing
    fills several.  A search for a violation uses a new lister, since a lister skips the syndromes
    that it has listed.
    """
    if collide:
        monkeypatch.setattr(
            _fault_search, "_hash_words", lambda words: np.zeros(len(words), np.uint64)
        )
    monkeypatch.setattr(_fault_search, "_BATCH_SIZE", 3)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The first flags make an accepted fault {0, 1, 2, 3} that contains the accepted fault {0, 2}.
    flag_sets = [[1, 2, 1, 2, 3, 3]]
    flag_sets += [list(rng.integers(1, 16, size=rng.integers(1, 10))) for _ in range(10)]
    for small_flags in flag_sets:
        # Spread the four bits of the flags over four words.
        flags = [
            sum(((int(ff) >> bit) & 1) << (70 * bit) for bit in range(4)) for ff in small_flags
        ]
        num_primitives = len(flags)
        # Syndromes of distinct bits, so that each fault has its own syndrome, and errors on
        # disjoint sets of seven qubits, so that no fault is skipped.
        syndromes = [1 << (13 * jj) for jj in range(num_primitives)]
        errors = [0b1111111 << (7 * jj) for jj in range(num_primitives)]
        for memory_limit, num_starts in itertools.product((0, 2**20), (num_primitives, 1)):
            args = (flags, syndromes, errors, 70, False, num_starts, memory_limit)
            lister = _fault_search.FaultLister(*args)
            for fault_weight in range(2, 6):
                expected = _irreducible_faults(flags, fault_weight, num_starts)
                checked, result = _check_faults(lister, fault_weight, ())
                assert result is None
                assert checked == [_xor(syndromes, fault) for fault in expected]
                if expected:
                    stop = int(rng.integers(len(expected)))
                    violation = (expected[stop], _xor(syndromes, expected[stop]))
                    _, result = _check_faults(
                        _fault_search.FaultLister(*args), fault_weight, {violation[1]}
                    )
                    assert result == violation


@pytest.mark.usefixtures("kernels")
@pytest.mark.parametrize("size", [1, 2])
def test_closing_tables_sort_sets_stably_by_hash(size: int, pytestconfig: pytest.Config) -> None:
    """A table lists the sets in lexicographic order, sorted stably by hash, as Python sorts them.

    The hashes are random, drawn from a few values so that many sets share a hash, or small so that
    the sets with different hashes share their high bits.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_primitives in (0, 1, 2, 5, 40):
        for hashes in (
            rng.integers(2**64, size=num_primitives, dtype=np.uint64),
            rng.integers(2**64, size=4, dtype=np.uint64)[rng.integers(4, size=num_primitives)],
            rng.integers(2**8, size=num_primitives, dtype=np.uint64),
        ):
            sets = sorted(
                itertools.combinations(range(num_primitives), size),
                key=lambda members: _xor(hashes.tolist(), members),
            )
            keys, members = _fault_search._get_closing_table(hashes, size)
            assert keys.tolist() == [_xor(hashes.tolist(), members) for members in sets]
            assert members.tolist() == [list(members) for members in sets]


@pytest.mark.usefixtures("kernels")
def test_repeated_syndromes_are_checked_once() -> None:
    """The faults {0, 1} and {0, 2} on |000> leave X on all qubits, so only the first is checked.

    The fault {1, 2} leaves no error, so it is not checked either.
    """
    lister = _fault_search.FaultLister([1, 1, 1], [7, 0, 0], [7, 0, 0], 3, False, 3, 0)
    assert _check_faults(lister, 2, ()) == ([7], None)


@pytest.mark.usefixtures("kernels")
@pytest.mark.parametrize("css_weight", [False, True])
def test_skipped_faults_satisfy_the_bound(css_weight: bool, pytestconfig: pytest.Config) -> None:
    """Calls with increasing fault weights check each syndrome once, and find the first violation.

    The primitive faults have random errors on a random three-qubit stabilizer state, and minimum
    Pauli weight representatives.  Brute force over all Paulis gives the minimum Pauli weights.  One
    lister checks every syndrome that it passes, and another searches for violations.
    """
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
            if violating:
                first = violating[0]
                assert result == (faults[first], fault_syndromes[first])
            else:
                assert result is None


@pytest.mark.usefixtures("kernels")
def test_syndromes_of_listed_faults_are_not_checked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fault is not checked if an earlier listed fault of smaller fault weight has its syndrome.

    On |0...0>, the minimum Pauli weight of a syndrome is its number of ones.  Primitive faults 0-3
    have flags 1 and flip one qubit each, and 4 and 5 have flags 2 and 3 and flip qubits {4} and
    {0, 4}.  So the faults of fault weight two are the pairs of 0-3, with distinct syndromes, and
    those of fault weight three add 4 and 5 to one of 0-3, which flips the qubits of a pair with 0.
    Each representative also has Z on two qubits of its own, so that no fault is left out for the
    weight of its representatives.

    The table of syndromes starts with room for one syndrome, so that it grows, unless there is no
    memory for it.  A search that finds a violation finds it again, since it clears the table.
    """
    monkeypatch.setattr(_fault_search, "_INITIAL_SYNDROME_TABLE_SIZE", 1)
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

    searcher = _fault_search.FaultLister(
        flags, syndromes, representatives, num_qubits, False, 6, 2**20
    )
    for _ in range(2):
        assert _check_faults(searcher, 2, {1 ^ 4})[1] == ((0, 2), 1 ^ 4)


def test_pair_table_peak_memory_is_within_its_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building a table of pairs needs at most about ``_BYTES_PER_PAIR`` bytes for each pair.

    The primitive faults have distinct random hashes, so no bucket of the table is merge sorted.

    The table is built by the Python versions of its compiled functions, since only allocations by
    Python are traced, and loading compiled code would add allocations of its own.
    """
    monkeypatch.setattr(_fault_search, "_sort_bucket", _fault_search._sort_bucket.py_func)
    num_primitives = 200
    hashes = np.random.default_rng(0).integers(2**64, size=num_primitives, dtype=np.uint64)
    tracemalloc.start()
    try:
        _fault_search._get_closing_table.py_func(hashes, 2)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Allow a few kilobytes that do not grow with the number of pairs.
    num_pairs = num_primitives * (num_primitives - 1) // 2
    assert peak <= num_pairs * _fault_search._BYTES_PER_PAIR + 2**14


def _z_state(num_qubits: int) -> npt.NDArray[np.int_]:
    """Stabilizers of |0...0>.

    The smallest Pauli weight of an error with a given syndrome is the syndrome's number of ones.
    """
    return np.hstack([np.zeros((num_qubits, num_qubits), dtype=int), np.eye(num_qubits, dtype=int)])


def _smallest_violation(
    flags: Sequence[int], syndromes: Sequence[int], max_weight: int | None
) -> int | None:
    """The smallest fault weight of an accepted fault that violates the bound on |0...0>."""
    limit = len(flags) if max_weight is None else min(max_weight, len(flags))
    for fault_weight in range(1, limit + 1):
        for indices in itertools.combinations(range(len(flags)), fault_weight):
            flag = functools.reduce(operator.xor, (flags[jj] for jj in indices))
            syndrome = functools.reduce(operator.xor, (syndromes[jj] for jj in indices))
            if flag == 0 and syndrome.bit_count() > fault_weight:
                return fault_weight
    return None


def _certify_map(
    flags: Sequence[int], syndromes: Sequence[int], max_weight: int | None, num_qubits: int
) -> _fault_search.Violation | None:
    """Search primitive faults with these flags and syndromes on |0...0> against brute force."""
    violation = _fault_search.find_smallest_violation(
        flags, syndromes, _z_state(num_qubits), False, max_weight
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
    """With zero flags, a primitive fault whose error has Pauli weight above one fails."""
    # Primitive faults whose errors have Pauli weight at most one certify, and those with Pauli
    # weight three do not.
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


def test_hand_derived_dependency_cases() -> None:
    """Faults of two or three primitive faults, repeated and zero signatures, and limits.

    The limits are values of ``max_weight``.
    """
    for flags, syndromes, max_weight, certified in (
        ([0, 0, 0], [1, 2, 4], 4, True),  # zero flags; each error has Pauli weight one
        ([1, 2, 3, 1, 2, 3], [1, 2, 3, 4, 8, 12], 4, False),
        # the accepted faults have fault weight two and errors of Pauli weight two
        ([1, 1, 2, 2, 4, 4], [1, 2, 1, 2, 1, 2], 4, True),
        # the heavy primitive faults 4 and 5 are in no violating fault
        ([1, 1, 2, 2, 3, 3], [1, 2, 1, 2, 5, 6], None, True),
        ([1, 1, 1, 0], [1, 1, 1, 0], 4, True),  # equal signatures cancel each other
        ([0, 0], [0, 0], 4, True),  # primitive faults with no effect
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], 3, True),
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], 4, False),  # fault weight 4, error of Pauli weight 6
        ([1, 2, 4, 3, 5], [0, 0, 0, 7, 56], None, False),
    ):
        failure = _certify_map(flags, syndromes, max_weight, 6)
        assert (failure is None) == certified


def test_failures_report_the_smallest_faults() -> None:
    """The smallest fault weight is reported.

    A fault of two primitive faults is reported although a fault of three primitive faults also
    violates the bound.
    """
    # The primitive faults 0-2 form an accepted fault whose error has Pauli weight four, and the
    # primitive faults 3 and 4 the only accepted fault of fault weight two, whose error has Pauli
    # weight three.
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

    # One primitive fault whose error has Pauli weight three brings back the search, and it fails
    # together with the primitive fault that has the same flags.
    failure = _certify_map(flags + [3], syndromes + [14], None, 4)
    assert failure is not None and len(failure.fault_indices) == 2


def test_max_weight_limits_the_fault_weight() -> None:
    """Only faults of fault weight at most ``max_weight`` are checked.

    The flags 1, 2, and 3 have rank two, and only the fault of all three primitive faults is
    accepted.
    """
    for max_weight in (None, 30, 2):
        failure = _certify_map([1, 2, 3], [1, 2, 12], max_weight, 4)
        assert (failure is None) == (max_weight == 2)


def test_memory_limit_does_not_change_results(pytestconfig: pytest.Config) -> None:
    """With almost no memory, the search is slower but finds the same counterexample."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The first map has flags of rank four, and its only accepted fault has fault weight five and an
    # error of Pauli weight six, so that the search finds it with the table of pairs if it fits.
    maps = [([1, 2, 4, 8, 15], [3, 12, 48, 0, 0])]
    for _ in range(10):
        num_primitives = int(rng.integers(4, 10))
        flags = list(map(int, rng.integers(1, 16, size=num_primitives)))
        maps.append((flags, list(map(int, rng.integers(64, size=num_primitives)))))
    for flags, syndromes in maps:
        results = [
            _fault_search.find_smallest_violation(flags, syndromes, _z_state(6), False, None, limit)
            for limit in (1, 2**20)
        ]
        assert results[0] == results[1] == _certify_map(flags, syndromes, None, 6)
