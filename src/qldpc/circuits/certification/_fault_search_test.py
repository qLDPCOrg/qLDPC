# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact search for a violating fault, against brute force on |0...0>."""

from __future__ import annotations

import functools
import itertools
import operator
from collections.abc import Sequence
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

from . import _fault_search


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
