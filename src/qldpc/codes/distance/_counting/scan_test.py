# SPDX-License-Identifier: Apache-2.0

"""Tests for the scan of the words of one weight, through the counts that it gives."""

from __future__ import annotations

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
    assert_counts_match_enumeration,
    get_counter,
    get_expected_counts,
    get_random_css_code,
)
from ..core import DistanceMethod
from . import scan


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("max_sets", [1, 3, 16])
def test_counts_against_enumeration(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config, max_sets: int
) -> None:
    """Counts match an independent enumeration, as do those of codes deformed after a count."""
    # Build overlapping basis families even for cheap counts.
    monkeypatch.setattr(_information_sets, "MINIMUM_WORK", 0)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    counter = lc.CSSLogicalCounter(get_random_css_code(rng, 12, 4, 4))
    assert_counts_match_enumeration(counter, max_information_sets=max_sets)

    # The counts above prepared basis families, which the deformed counters transport.
    for deformed in (counter.deformed_cnot(0, 5), counter.deformed_cnot(2, 7)):
        assert_counts_match_enumeration(deformed, max_information_sets=max_sets)


@pytest.mark.usefixtures("compilation_mode")
def test_a_deformed_self_dual_counter_shares_its_basis_family() -> None:
    """A self-dual code stays self-dual under a SWEL-preserving gate, and keeps one basis family."""
    counter = lc.CSSLogicalCounter(codes.SteaneCode())
    assert_counts_match_enumeration(counter)
    deformed = counter.deformed_swel(0, 1, 2, 3)
    assert deformed.is_self_dual
    assert_counts_match_enumeration(deformed)


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("method", METHODS)
def test_scans_resume_after_every_chunk(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config, method: DistanceMethod
) -> None:
    """A scan that stops after each combination, and after each call, still visits every word."""
    monkeypatch.setattr(_information_sets, "MINIMUM_WORK", 0)
    monkeypatch.setattr(scan, "_CHUNK_SIZE", 1)
    monkeypatch.setattr(scan, "_CHUNKS_PER_SCAN_CALL", 1)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    counter = lc.CSSLogicalCounter(get_random_css_code(rng, 10, 3, 3))
    assert_counts_match_enumeration(counter, method=method)


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("max_sets", [4, 16])
def test_overlapping_basis_families_count_every_word_once(
    monkeypatch: pytest.MonkeyPatch, max_sets: int
) -> None:
    """Large basis families of overlapping sets give the counts of a brute-force scan."""
    monkeypatch.setattr(_information_sets, "MINIMUM_WORK", 0)
    # This code makes the basis family stages choose basis families of over four overlapping sets.
    code = get_random_css_code(np.random.default_rng(5), 24, 8, 8)
    counter = lc.CSSLogicalCounter(code)
    weights = range(len(code) + 1)
    expected = [counter.count_logical_ops(weight, method="brute_force") for weight in weights]
    assert [
        counter.count_logical_ops(w, max_information_sets=max_sets) for w in weights
    ] == expected


def test_the_information_set_limit_bounds_the_basis_families_of_a_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The limit shrinks the overlapping basis families that a count may use, not the counts."""
    sizes: list[int] = []
    original = _information_sets.get_overlapping_family

    def record(*args: Any) -> _information_sets.BasisFamily:
        replacement = original(*args)
        sizes.append(len(replacement.ranks))
        return replacement

    monkeypatch.setattr(_information_sets, "get_overlapping_family", record)
    monkeypatch.setattr(_information_sets, "MINIMUM_WORK", 0)
    code = get_random_css_code(np.random.default_rng(5), 24, 8, 8)
    weights = range(len(code) + 1)
    for weight in weights:
        lc.count_logical_ops_css(code, weight)
    assert max(sizes) > 4

    sizes.clear()
    for weight in weights:
        lc.count_logical_ops_css(code, weight, max_information_sets=4)
    assert sizes and max(sizes) <= 4


def test_python_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the Python versions of the compiled kernels, which coverage can trace."""
    for name in ("_count_chunk", "_is_owned", "_get_masked_rank"):
        monkeypatch.setattr(scan, name, getattr(scan, name).py_func)
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
