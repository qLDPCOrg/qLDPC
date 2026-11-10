# SPDX-License-Identifier: Apache-2.0

"""Tests for the scan of the words of one weight, through the counts that it gives."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes
from qldpc._binary import pack_bits
from qldpc.objects import Pauli

from ...conftest import Tune
from .. import logical_counts as lc
from .._search import completion_bounds, information_sets
from .._tuning import MAX_VALUE
from ..conftest import (
    HAND_X,
    HAND_Z,
    METHODS,
    CostlyStages,
    Counts,
    assert_counts_match_enumeration,
    get_counter,
    get_expected_counts,
    get_random_css_code,
    get_sector_problem,
)
from ..core import DistanceMethod
from . import scan


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("max_sets", [1, 3, 16])
def test_counts_against_enumeration(tune: Tune, pytestconfig: pytest.Config, max_sets: int) -> None:
    """Counts match an independent enumeration, as do those of codes deformed after a count."""
    # Build overlapping basis families even for cheap counts.
    tune(overlapping_minimum_work=0, max_information_sets=max_sets)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    counter = lc.CSSLogicalCounter(get_random_css_code(rng, 12, 4, 4))
    assert_counts_match_enumeration(counter)

    # The counts above prepared basis families, which the deformed counters transport.
    for deformed in (counter.deformed_cnot(0, 5), counter.deformed_cnot(2, 7)):
        assert_counts_match_enumeration(deformed)


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
    tune: Tune, pytestconfig: pytest.Config, method: DistanceMethod
) -> None:
    """A scan that stops after each combination, and after each call, still visits every word."""
    tune(overlapping_minimum_work=0, scan_chunk_size=1, scan_chunks_per_call=1)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    counter = lc.CSSLogicalCounter(get_random_css_code(rng, 10, 3, 3))
    assert_counts_match_enumeration(counter, method=method)


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("max_sets", [4, 16])
def test_overlapping_basis_families_count_every_word_once(tune: Tune, max_sets: int) -> None:
    """Large basis families of overlapping sets give the counts of a brute-force scan."""
    tune(overlapping_minimum_work=0, max_information_sets=max_sets)
    # This code makes the basis family stages choose basis families of over four overlapping sets.
    code = get_random_css_code(np.random.default_rng(4), 12, 4, 4)
    counter = lc.CSSLogicalCounter(code)
    weights = range(len(code) + 1)
    expected = [counter.count_logical_ops(weight, method="brute_force") for weight in weights]
    assert [counter.count_logical_ops(weight) for weight in weights] == expected


def test_counts_through_the_costly_stages_count_every_word_once(
    tune: Tune, costly_stages: CostlyStages
) -> None:
    """Counts with partitioned families and the certificate of linear programs are exact."""
    tune(overlapping_minimum_work=0, lp_minimum_work=0, partition_work_ratio=0)
    code = get_random_css_code(np.random.default_rng(4), 12, 4, 4)
    counter = lc.CSSLogicalCounter(code)
    weights = range(len(code) + 1)
    expected = [counter.count_logical_ops(weight, method="brute_force") for weight in weights]
    assert [counter.count_logical_ops(weight) for weight in weights] == expected
    assert any(
        len(family.ranks) and family.max_coverage > 1
        for family in costly_stages.partitioned_families
    )
    assert any(certificate.pivot_blocks for certificate in costly_stages.search_certificates)


def test_the_information_set_limit_bounds_the_basis_families_of_a_count(
    monkeypatch: pytest.MonkeyPatch, tune: Tune
) -> None:
    """The limit shrinks the overlapping basis families that a count may use, not the counts."""
    sizes: list[int] = []
    original = information_sets.get_overlapping_family

    def record(*args: Any) -> information_sets.BasisFamily:
        replacement = original(*args)
        sizes.append(len(replacement.ranks))
        return replacement

    monkeypatch.setattr(information_sets, "get_overlapping_family", record)
    tune(overlapping_minimum_work=0)
    code = get_random_css_code(np.random.default_rng(4), 12, 4, 4)
    weights = range(len(code) + 1)
    for weight in weights:
        lc.count_logical_ops_css(code, weight)
    assert max(sizes) > 4

    sizes.clear()
    tune(max_information_sets=4)
    for weight in weights:
        lc.count_logical_ops_css(code, weight)
    assert sizes and max(sizes) <= 4


def test_python_kernels(monkeypatch: pytest.MonkeyPatch, tune: Tune) -> None:
    """Run the Python versions of the compiled kernels, which coverage can trace.

    The compiled helpers that the scan imports by name run as Python too.
    """
    for name in (
        "_count_chunk",
        "_count_chunk_skipping",
        "_visit_last_rows",
        "_is_owned",
        "_get_masked_rank",
        "advance_support",
        "get_completion_bound",
    ):
        monkeypatch.setattr(scan, name, getattr(scan, name).py_func)
    # Tiny chunks make every scan stop and resume repeatedly, and forced completion tables make the
    # steps of two or more rows skip completions.
    tune(scan_chunk_size=1, **FORCED_TABLES)
    code = get_random_css_code(np.random.default_rng(3), 14, 4, 4)
    assert_counts_match_enumeration(lc.CSSLogicalCounter(code))
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


####################################################################################################
# skipping completions with completion tables

# Tuning that makes every step of two or more rows of an information set use completion tables.
FORCED_TABLES: dict[str, Any] = {
    "completion_bound_minimum_layer_work": 0,
    "completion_bound_max_cost_percent": MAX_VALUE,
}


@pytest.mark.parametrize("chunk_size", [1, 3, 1 << 16])
def test_counts_with_completion_tables_match_enumeration(
    monkeypatch: pytest.MonkeyPatch, tune: Tune, chunk_size: int
) -> None:
    """Steps that skip completions still count every word, also when they stop at every lookup.

    Some completions of these codes have bounds equal to the target weight, so a scan that skipped
    them, rather than only heavier ones, would lose words that this enumeration counts.
    """
    selected: list[completion_bounds.CompletionTables] = []
    select = completion_bounds.select_completion_tables

    def record(*args: Any) -> completion_bounds.CompletionTables | None:
        tables = select(*args)
        if tables is not None:
            selected.append(tables)
        return tables

    monkeypatch.setattr(completion_bounds, "select_completion_tables", record)
    tune(overlapping_minimum_work=0, scan_chunk_size=chunk_size, **FORCED_TABLES)
    for seed in range(3):
        code = get_random_css_code(np.random.default_rng(seed), 14, 4, 4)
        assert_counts_match_enumeration(lc.CSSLogicalCounter(code))
    assert selected


@pytest.mark.parametrize("max_sets", [4, 16])
def test_overlapping_families_with_completion_tables_count_every_word_once(
    monkeypatch: pytest.MonkeyPatch, tune: Tune, max_sets: int
) -> None:
    """Every set of an overlapping family skips completions with its own tables and tails."""
    sets_with_tables: set[int] = set()
    select = scan._StepTables.select

    def record(
        step_tables: scan._StepTables, layer: int, set_index: int
    ) -> tuple[completion_bounds.CompletionTables, npt.NDArray[np.uint64]]:
        tables, rows = select(step_tables, layer, set_index)
        if len(tables.tables):
            sets_with_tables.add(set_index)
        return tables, rows

    monkeypatch.setattr(scan._StepTables, "select", record)
    tune(overlapping_minimum_work=0, max_information_sets=max_sets, **FORCED_TABLES)
    code = get_random_css_code(np.random.default_rng(5), 14, 5, 5)
    counter = lc.CSSLogicalCounter(code)
    weights = range(len(code) + 1)
    expected = [counter.count_logical_ops(weight, method="brute_force") for weight in weights]
    assert [counter.count_logical_ops(weight) for weight in weights] == expected
    assert len(sets_with_tables) > 1


def test_counts_return_to_python_only_around_steps_that_may_use_tables(
    monkeypatch: pytest.MonkeyPatch, tune: Tune
) -> None:
    """A count whose steps are too small for tables scans them all in one call."""
    calls: list[int] = []
    scan_layers = scan._scan_coefficient_layers

    def record(*args: Any) -> int:
        calls.append(1)
        return scan_layers(*args)

    monkeypatch.setattr(scan, "_scan_coefficient_layers", record)
    code = get_random_css_code(np.random.default_rng(5), 24, 8, 8)
    lc.CSSLogicalCounter(code).count_logical_ops(6)
    assert len(calls) == 2

    # With tables considered for every step, a count calls once for each step that may use them.
    calls.clear()
    tune(**FORCED_TABLES)
    lc.CSSLogicalCounter(code).count_logical_ops(6)
    assert len(calls) > 2


def test_steps_run_in_order() -> None:
    """Steps run through the sets of each layer, and only the completed sets of the last layer."""
    steps = scan._Steps(num_sets=2, last_layer=3, last_num_completed=1)
    visited = [(1, 0)]
    while (step := steps.get_next(*visited[-1])) != (0, 0):
        visited.append(step)
    assert visited == [(1, 0), (1, 1), (2, 0), (2, 1), (3, 0)]
    assert steps.find((1, 0), lambda layer, set_index: set_index == 1) == (1, 1)
    assert steps.find((1, 1), lambda layer, set_index: set_index == 1) == (2, 1)
    assert steps.find((2, 1), lambda layer, set_index: set_index == 1) == (0, 0)


def test_step_tables_choose_and_keep_tables(tune: Tune) -> None:
    """Steps get tables only for layers of two or more rows of full sets, within a memory limit."""
    code = get_random_css_code(np.random.default_rng(1), 24, 8, 8)
    basis, labels = get_sector_problem(code, Pauli.X)
    disjoint = information_sets.get_disjoint_family(basis, labels, random_column_orders=0)
    deficient = len(disjoint.ranks) - 1
    assert disjoint.ranks[0] == len(basis) and disjoint.ranks[deficient] < len(basis)

    # A family of two full sets: the first set, and the same set with its rows in reverse order.
    order = np.arange(len(basis))[::-1]
    family = disjoint._replace(
        generators=np.stack([disjoint.generators[0], disjoint.generators[0][order]]),
        labels=np.stack([disjoint.labels[0], disjoint.labels[0][order]]),
        pivots=np.stack([disjoint.pivots[0], disjoint.pivots[0][order]]),
        ranks=disjoint.ranks[[0, 0]],
    )
    rows = scan._pack_basis_family(family, basis.shape[1], labels.shape[1])[0]
    rows_and_labels = np.zeros((2, len(basis), rows.shape[2] + 1), dtype=np.uint64)
    rows_and_labels[:, :, : rows.shape[2]] = rows
    tuning = tune(**FORCED_TABLES)

    # No step of a brute-force scan, of layer one, of a set without a full block, or below the
    # minimum work, may use tables.
    rows_of_disjoint = scan._pack_basis_family(disjoint, basis.shape[1], labels.shape[1])[0]
    assert not scan._StepTables(None, rows_and_labels, 4, tuning).may_use_tables(2, 0)
    assert not scan._StepTables(family, rows_and_labels, 4, tuning).may_use_tables(1, 0)
    assert not scan._StepTables(disjoint, rows_of_disjoint, 4, tuning).may_use_tables(2, deficient)
    small = tune(completion_bound_minimum_layer_work=math.comb(len(basis), 3) + 1)
    assert not scan._StepTables(family, rows_and_labels, 4, small).may_use_tables(3, 0)
    assert scan._StepTables(family, rows_and_labels, 4, small).may_use_tables(4, 0)

    # Each full set's step gets tables, and its rows carry that set's own tails after the labels.
    step_tables = scan._StepTables(family, rows_and_labels, 4, tuning)
    assert step_tables.may_use_tables(2, 0) and step_tables.may_use_tables(2, 1)
    for set_index in range(2):
        tables, step_rows = step_tables.select(2, set_index)
        tails = pack_bits(np.delete(family.generators[set_index], family.pivots[set_index], axis=1))
        assert len(tables.tables)
        assert step_rows.shape == (1, len(basis), rows_and_labels.shape[2] + tails.shape[1])
        assert step_rows.shape[2] == step_tables.max_row_width
        np.testing.assert_array_equal(
            step_rows[0, :, : rows_and_labels.shape[2]], rows_and_labels[set_index]
        )
        np.testing.assert_array_equal(step_rows[0, :, rows_and_labels.shape[2] :], tails)
        assert step_tables.select(2, set_index)[0] is tables
        assert step_tables.select(3, set_index)[0] is tables

    # A limit that holds one set's tables keeps the latest set's tables, and rebuilds the others.
    tuning = tune(completion_bound_max_table_bytes=len(tables.tables), **FORCED_TABLES)
    step_tables = scan._StepTables(family, rows_and_labels, 4, tuning)
    first = step_tables.select(2, 0)[0]
    second = step_tables.select(2, 1)[0]
    assert step_tables.select(3, 1)[0] is second
    assert step_tables.select(3, 0)[0] is not first

    # A step whose tables the gate declines scans without them.
    declining = tune(completion_bound_minimum_layer_work=0, completion_bound_max_cost_percent=1)
    step_tables = scan._StepTables(family, rows_and_labels, 4, declining)
    assert step_tables.may_use_tables(2, 0)
    assert step_tables.select(2, 0)[0] is completion_bounds.NO_TABLES


@pytest.mark.parametrize("compiled", [False, True])
def test_skipping_chunks_stop_at_a_lighter_nontrivial_word(compiled: bool) -> None:
    """A step with tables stops at a nontrivial word lighter than the target, if asked to."""
    # Three systematic rows [I | P], each followed by a distinct label and by its tail P.
    tails = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 1]], dtype=np.uint8)
    words = np.hstack([np.eye(3, dtype=np.uint8), tails])
    labels = np.eye(3, dtype=np.uint8)
    step_rows = np.concatenate([pack_bits(words), pack_bits(labels), pack_bits(tails)], axis=1)
    tables = completion_bounds.CompletionTables.build(pack_bits(tails), 8)
    count_chunk = scan._count_chunk_skipping if compiled else scan._count_chunk_skipping.py_func
    support = np.zeros(2, dtype=np.int64)
    prefixes = np.zeros((3, 1, step_rows.shape[1]), dtype=np.uint64)
    histogram = np.zeros(1, dtype=np.int64)
    arguments = (step_rows[np.newaxis], 1, 2, np.zeros((1, 0, 1), dtype=np.uint64), 0, support)
    rank_rows = np.zeros((0, 1), dtype=np.uint64)
    options = (rank_rows, False, 0, histogram)
    table_arguments = (1 << 10, tables.tables, tables.offsets, tables.block_bits)

    # Every word of two rows weighs at most six, so it is lighter than ten and nontrivial.
    depth, lighter = count_chunk(*arguments, prefixes, 0, 10, *options, True, *table_arguments)
    assert lighter and depth == 1
    assert histogram.sum() == 0

    # Without rejecting lighter words, the step counts none of them and finishes its layer.
    support[:] = 0
    depth, lighter = count_chunk(*arguments, prefixes, 0, 10, *options, False, *table_arguments)
    assert not lighter and depth == -1

    # The words of two rows weigh 4, 5, and 5.  At target 3, the first lookup rules out every pair
    # from row 0 on, so a chunk of one lookup ends having skipped every first row.
    support[:] = 0
    small_chunk = (1, *table_arguments[1:])
    depth, lighter = count_chunk(*arguments, prefixes, 0, 3, *options, False, *small_chunk)
    assert (depth, lighter) == (0, False)
    assert support[0] == 3
    assert histogram.sum() == 0

    # At target 4, the bound from row 0 on is below the limit, so the step weighs those pairs and
    # counts the word of weight 4; from row 1 on, the only pair weighs 5, which it skips.
    support[:] = 0
    depth, lighter = count_chunk(*arguments, prefixes, 0, 4, *options, False, *table_arguments)
    assert (depth, lighter) == (-1, False)
    assert histogram.sum() == 1
