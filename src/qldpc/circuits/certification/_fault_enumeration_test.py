# SPDX-License-Identifier: Apache-2.0

"""Tests for the compiled listing of irreducible accepted faults."""

from __future__ import annotations

import functools
import itertools
import operator
import tracemalloc
from collections.abc import Collection, Sequence

import numpy as np
import numpy.typing as npt
import pytest
import stim

from . import _fault_enumeration as enumeration


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
            monkeypatch.setattr(enumeration, name, getattr(enumeration, name).py_func)


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
    lister: enumeration.FaultLister, fault_weight: int, violating: Collection[int]
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
            enumeration, "_hash_words", lambda words: np.zeros(len(words), np.uint64)
        )
    monkeypatch.setattr(enumeration, "_BATCH_SIZE", 3)
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
            lister = enumeration.FaultLister(*args)
            for fault_weight in range(2, 6):
                expected = _irreducible_faults(flags, fault_weight, num_starts)
                checked, result = _check_faults(lister, fault_weight, ())
                assert result is None
                assert checked == [_xor(syndromes, fault) for fault in expected]
                if expected:
                    stop = int(rng.integers(len(expected)))
                    violation = (expected[stop], _xor(syndromes, expected[stop]))
                    _, result = _check_faults(
                        enumeration.FaultLister(*args), fault_weight, {violation[1]}
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
            keys, members = enumeration._get_closing_table(hashes, size)
            assert keys.tolist() == [_xor(hashes.tolist(), members) for members in sets]
            assert members.tolist() == [list(members) for members in sets]


@pytest.mark.usefixtures("kernels")
def test_repeated_syndromes_are_checked_once() -> None:
    """The faults {0, 1} and {0, 2} on |000> leave X on all qubits, so only the first is checked.

    The fault {1, 2} leaves no error, so it is not checked either.
    """
    lister = enumeration.FaultLister([1, 1, 1], [7, 0, 0], [7, 0, 0], 3, False, 3, 0)
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
            enumeration.FaultLister(flags, syndromes, representatives, n, css_weight, 8, 2**20)
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
    monkeypatch.setattr(enumeration, "_INITIAL_SYNDROME_TABLE_SIZE", 1)
    flags = [1, 1, 1, 1, 2, 3]
    syndromes = [1, 2, 4, 8, 16, 17]
    num_qubits = 5 + 2 * len(flags)
    representatives = [
        syndrome | (0b11 << (num_qubits + 5 + 2 * jj)) for jj, syndrome in enumerate(syndromes)
    ]
    pairs = [syndromes[aa] ^ syndromes[bb] for aa, bb in itertools.combinations(range(4), 2)]
    for memory_limit in (2**20, 0):
        lister = enumeration.FaultLister(
            flags, syndromes, representatives, num_qubits, False, 6, memory_limit
        )
        assert _check_faults(lister, 2, ()) == (pairs, None)
        checked, _ = _check_faults(lister, 3, ())
        # Fault {0, 4, 5} has no syndrome, and {jj, 4, 5} has that of the pair {0, jj}.  Without
        # memory, the table keeps only the syndrome of the first pair, {0, 1}.
        assert checked == ([0] if memory_limit else [0, 1 ^ 4, 1 ^ 8])

    searcher = enumeration.FaultLister(
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
    monkeypatch.setattr(enumeration, "_sort_bucket", enumeration._sort_bucket.py_func)
    num_primitives = 200
    hashes = np.random.default_rng(0).integers(2**64, size=num_primitives, dtype=np.uint64)
    tracemalloc.start()
    try:
        enumeration._get_closing_table.py_func(hashes, 2)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Allow a few kilobytes that do not grow with the number of pairs.
    num_pairs = num_primitives * (num_primitives - 1) // 2
    assert peak <= num_pairs * enumeration._BYTES_PER_PAIR + 2**14
