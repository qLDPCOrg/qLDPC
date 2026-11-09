# SPDX-License-Identifier: Apache-2.0

"""Tests for the scan of the words of one weight, through the counts that it gives."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sympy.abc import x, y

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
    code = get_random_css_code(np.random.default_rng(5), 24, 8, 8)
    counter = lc.CSSLogicalCounter(code)
    weights = range(len(code) + 1)
    expected = [counter.count_logical_ops(weight, method="brute_force") for weight in weights]
    assert [counter.count_logical_ops(weight) for weight in weights] == expected


def test_counts_through_the_costly_stages_count_every_word_once(
    tune: Tune, costly_stages: CostlyStages
) -> None:
    """Counts with partitioned families and the certificate of linear programs are exact."""
    tune(overlapping_minimum_work=0, lp_minimum_work=0, partition_work_ratio=0)
    code = get_random_css_code(np.random.default_rng(5), 24, 8, 8)
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
    code = get_random_css_code(np.random.default_rng(5), 24, 8, 8)
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
    """Run the Python versions of the compiled kernels, which coverage can trace."""
    for name in (
        "_count_chunk",
        "_count_chunk_skipping",
        "_visit_last_rows",
        "_is_owned",
        "_get_masked_rank",
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


def test_counts_of_a_larger_code_with_completion_tables(tune: Tune) -> None:
    """Counts with completion tables equal those without, for a code with overlapping families."""
    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    tune(completion_bound_max_table_bytes=0)
    expected = [lc.CSSLogicalCounter(code).count_logical_ops(weight) for weight in (6, 7, 8)]
    assert expected[0] == Counts(6, 84, 84, None)
    tune(**FORCED_TABLES)
    assert [
        lc.CSSLogicalCounter(code).count_logical_ops(weight) for weight in (6, 7, 8)
    ] == expected
    assert lc.CSSLogicalCounter(code).count_logical_ops(count_pauli=True) == Counts(6, 84, 84, 180)


def test_step_tables_choose_and_keep_tables(tune: Tune) -> None:
    """Steps get tables only for layers of two or more rows of full sets, within a memory limit."""
    code = get_random_css_code(np.random.default_rng(1), 24, 8, 8)
    basis, labels = get_sector_problem(code, Pauli.X)
    family = information_sets.get_disjoint_family(basis, labels, random_column_orders=0)
    deficient = len(family.ranks) - 1
    assert family.ranks[0] == len(basis) and family.ranks[deficient] < len(basis)
    rows = scan._pack_basis_family(family, basis.shape[1], labels.shape[1])[0]
    rows_and_labels = np.zeros((len(family.ranks), len(basis), rows.shape[2] + 1), dtype=np.uint64)
    rows_and_labels[:, :, : rows.shape[2]] = rows
    tuning = tune(**FORCED_TABLES)

    # No step of a brute-force scan, of layer one, or of a set without a full block has tables.
    for basis_family, layer, set_index in ((None, 2, 0), (family, 1, 0), (family, 2, deficient)):
        step_tables = scan._StepTables(basis_family, rows_and_labels, 4, tuning)
        tables, _ = step_tables.select(layer, set_index)
        assert tables is completion_bounds.NO_TABLES

    # A full set's step gets tables, and its rows carry the set's tails after the labels.
    step_tables = scan._StepTables(family, rows_and_labels, 4, tuning)
    tables, step_rows = step_tables.select(2, 0)
    tails = step_tables._get_tails(0)
    assert len(tables.tables)
    assert step_rows.shape == (1, len(basis), rows_and_labels.shape[2] + tails.shape[1])
    assert step_rows.shape[2] == step_tables.max_row_width
    np.testing.assert_array_equal(step_rows[0, :, rows_and_labels.shape[2] :], tails)
    assert step_tables.select(2, 0)[0] is tables
    assert step_tables.select(3, 0)[0] is tables

    # Tables of another set replace kept ones that the limit cannot hold too.
    two_full_sets = family._replace(
        generators=family.generators[[0, 0]],
        labels=family.labels[[0, 0]],
        pivots=family.pivots[[0, 0]],
        ranks=family.ranks[[0, 0]],
    )
    tuning = tune(completion_bound_max_table_bytes=len(tables.tables), **FORCED_TABLES)
    step_tables = scan._StepTables(two_full_sets, rows_and_labels[[0, 0]], 4, tuning)
    step_tables.select(2, 0)
    step_tables.select(2, 1)
    assert list(step_tables._kept) == [1]


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
