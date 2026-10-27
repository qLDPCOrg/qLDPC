# SPDX-License-Identifier: Apache-2.0

"""Tests for the compiled listing of irreducible accepted faults."""

from __future__ import annotations

import functools
import itertools
import operator
from collections.abc import Collection, Sequence

import numpy as np
import pytest
import stim

from . import _fault_enumeration as enumeration


@pytest.fixture(params=[False, True], ids=["compiled", "python"])
def kernels(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the compiled listing, or its Python version for coverage."""
    if request.param:
        for name in ("iter_batches", "reduce_flags", "pauli_weight", "add_syndrome"):
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
) -> tuple[list[tuple[tuple[int, ...], int]], tuple[tuple[int, ...] | None, int]]:
    """List faults with a check that records each fault, and fails on these syndromes.

    Return the checked faults with their syndromes, and the result of the listing.
    """
    checked = []

    def check(fault: tuple[int, ...], syndrome: int) -> tuple[int, ...] | None:
        checked.append((fault, syndrome))
        return fault if syndrome in violating else None

    return checked, lister.find_first(fault_weight, check)


@pytest.mark.usefixtures("kernels")
@pytest.mark.parametrize("collide", [False, True])
def test_listing_matches_brute_force(
    collide: bool, pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each fault is listed once and in lexicographic order, with or without a table of pairs.

    Flags and syndromes span several words.  If ``collide``, every hash is zero, so that lookups
    find every entry of a table and must compare flags.  The batches are small, so that a listing
    fills several.
    """
    if collide:
        monkeypatch.setattr(
            enumeration, "hash_words", lambda words: np.zeros(len(words), np.uint64)
        )
    monkeypatch.setattr(enumeration, "BATCH_SIZE", 3)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The first flags make an accepted fault {0, 1, 2, 3} that contains the accepted fault {0, 2}.
    flag_sets = [[1, 2, 1, 2, 3, 3]]
    flag_sets += [list(rng.integers(1, 16, size=rng.integers(1, 10))) for _ in range(10)]
    for small_flags in flag_sets:
        # Spread the four bits of the flags over three words.
        flags = [
            sum(((int(ff) >> bit) & 1) << (70 * bit) for bit in range(4)) for ff in small_flags
        ]
        num_primitives = len(flags)
        # Distinct syndromes, and errors on disjoint sets of seven qubits, so that no fault is
        # skipped.
        syndromes = [1 << (13 * jj) for jj in range(num_primitives)]
        errors = [0b1111111 << (7 * jj) for jj in range(num_primitives)]
        for memory_limit, num_starts in itertools.product((0, 2**20), (num_primitives, 1)):
            lister = enumeration.FaultLister(
                flags, syndromes, errors, 70, False, num_starts, memory_limit
            )
            for fault_weight in range(2, 6):
                expected = _irreducible_faults(flags, fault_weight, num_starts)
                checked, result = _check_faults(lister, fault_weight, ())
                assert result == (None, len(expected))
                assert checked == [(fault, _xor(syndromes, fault)) for fault in expected]
                if expected:
                    stop = int(rng.integers(len(expected)))
                    violating = {_xor(syndromes, expected[stop])}
                    _, result = _check_faults(lister, fault_weight, violating)
                    assert result == (expected[stop], stop + 1)


@pytest.mark.usefixtures("kernels")
def test_repeated_syndromes_are_checked_once() -> None:
    """The faults {0, 1} and {0, 2} on |000> leave X on all qubits, so only the first is checked.

    The fault {1, 2} leaves no error, so it is not checked either, but all three faults count.
    """
    lister = enumeration.FaultLister([1, 1, 1], [7, 0, 0], [7, 0, 0], 3, False, 3, 0)
    assert _check_faults(lister, 2, ()) == ([((0, 1), 7)], (None, 3))


@pytest.mark.usefixtures("kernels")
@pytest.mark.parametrize("css_weight", [False, True])
def test_skipped_faults_satisfy_the_bound(css_weight: bool, pytestconfig: pytest.Config) -> None:
    """Each syndrome of a violating fault is checked once, and the first violation is found.

    The primitive faults have random errors on a random three-qubit stabilizer state, and minimum
    Pauli weight representatives.  Brute force over all Paulis gives the minimum Pauli weights.
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
        lister = enumeration.FaultLister(flags, syndromes, representatives, n, css_weight, 8, 2**20)
        for fault_weight in range(2, 5):
            faults = _irreducible_faults(flags, fault_weight, 8)
            fault_syndromes = [_xor(syndromes, fault) for fault in faults]
            violating = [jj for jj, ss in enumerate(fault_syndromes) if minimum[ss] > fault_weight]

            checked, _ = _check_faults(lister, fault_weight, ())
            checked_syndromes = [syndrome for _, syndrome in checked]
            assert len(set(checked_syndromes)) == len(checked_syndromes)
            assert {fault_syndromes[jj] for jj in violating} <= set(checked_syndromes)

            violating_syndromes = {ss for ss, weight in minimum.items() if weight > fault_weight}
            _, result = _check_faults(lister, fault_weight, violating_syndromes)
            if violating:
                assert result == (faults[violating[0]], violating[0] + 1)
            else:
                assert result == (None, len(faults))
