# SPDX-License-Identifier: Apache-2.0

"""Tests for the scan of the words of one weight, through the counts that it gives."""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
import pytest

from qldpc import codes

from .. import _information_sets
from .. import logical_counts as lc
from ..conftest import (
    HAND_X,
    HAND_Z,
    METHODS,
    Counts,
    append_frozen_qubits,
    get_counter,
    get_expected_counts,
    get_random_css_code,
)
from . import local_cliffords, scan


def test_overlapping_information_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Overlapping information sets, which large searches use, still count every word once."""
    select = functools.partial(_information_sets.get_overlapping_family, minimum_work=0)
    monkeypatch.setattr(_information_sets, "get_overlapping_family", select)
    hamming = np.asarray(codes.HammingCode(4).matrix)
    code = get_counter(*append_frozen_qubits(hamming, hamming, 15)).code
    counter = lc.CSSLogicalCounter(code)
    for weight in range(len(code) + 1):
        expected = counter.count_logical_ops(weight, method="brute_force")
        assert counter.count_logical_ops(weight) == expected


def test_information_set_limit_bounds_overlapping_families(monkeypatch: pytest.MonkeyPatch) -> None:
    """The limit shrinks the overlapping families that a count may use, not the counts."""
    sizes: list[int] = []
    original = _information_sets.get_overlapping_family

    def select(*args: Any, **kwargs: Any) -> _information_sets.BasisFamily | None:
        family = original(*args, minimum_work=0, **kwargs)
        if family is not None:
            sizes.append(len(family.information_sets))
        return family

    monkeypatch.setattr(_information_sets, "get_overlapping_family", select)
    code = get_random_css_code(np.random.default_rng(5), 24, 8, 8)
    counter = lc.CSSLogicalCounter(code)
    expected = [counter.count_logical_ops(weight, method="brute_force") for weight in range(25)]
    assert [counter.count_logical_ops(weight) for weight in range(25)] == expected
    assert max(sizes) > 4

    sizes.clear()
    limited = [
        lc.count_logical_ops_css(code, weight, max_information_sets=4) for weight in range(25)
    ]
    assert limited == expected
    assert sizes and max(sizes) <= 4


def test_python_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the Python versions of the compiled kernels, which coverage can trace."""
    for name in ("_count_chunk", "_is_owned", "_get_masked_rank"):
        monkeypatch.setattr(scan, name, getattr(scan, name).py_func)
    monkeypatch.setattr(local_cliffords, "apply_local", local_cliffords.apply_local.py_func)
    # Tiny chunks make every scan stop and resume repeatedly.
    monkeypatch.setattr(scan, "_CHUNK_SIZE", 1)
    for method in METHODS:
        counter = get_counter(HAND_X, HAND_Z)
        assert counter.count_logical_ops(count_pauli=True, method=method) == Counts(2, 2, 3, 6)
        with pytest.raises(ValueError, match="less than weight 3"):
            counter.count_logical_ops(3, method=method, count_pauli=True)
        bacon_shor = codes.BaconShorCode(3)
        expected = get_expected_counts(bacon_shor)
        bacon_shor_counter = lc.CSSLogicalCounter(bacon_shor)
        assert bacon_shor_counter.count_logical_ops(count_pauli=True, method=method) == expected
        stacked = lc.CSSLogicalCounter(codes.CSSCode.stack([codes.SteaneCode()] * 2))
        assert stacked.count_logical_ops(method=method) == Counts(3, 14, 14, None)
        deformed = counter.deformed_cnot(0, 3)
        fresh = get_counter(deformed.checks_x, deformed.checks_z)
        assert deformed.count_logical_ops(
            count_pauli=True, method=method
        ) == fresh.count_logical_ops(count_pauli=True, method=method)
