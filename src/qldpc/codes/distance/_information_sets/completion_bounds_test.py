# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tables that bound the completions of BZ supports by two more rows."""

from __future__ import annotations

import dataclasses
import itertools
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import pack_bits

from .._tuning import DEFAULT_TUNING, MAX_VALUE, SearchTuning
from . import completion_bounds as cb

# A tuning that considers every layer, uses tables whatever they cost, and allows any size.
Z = DEFAULT_TUNING.completion_bound_confidence_z
MAX_SAMPLES = 2**12
FORCED = dataclasses.replace(
    DEFAULT_TUNING,
    completion_bound_minimum_layer_work=0,
    completion_bound_max_cost_percent=MAX_VALUE,
    completion_bound_max_table_bytes=MAX_VALUE,
)


def _with_table_bytes(max_table_bytes: int) -> SearchTuning:
    """The default tuning with a memory limit for completion tables."""
    return dataclasses.replace(DEFAULT_TUNING, completion_bound_max_table_bytes=max_table_bytes)


def get_random_tails(
    rng: np.random.Generator, num_rows: int, num_columns: int
) -> npt.NDArray[np.uint8]:
    """Random tails with all-zero, all-one, and repeated columns, and a repeated row."""
    tails = (rng.random((num_rows, num_columns)) < rng.uniform(0.1, 0.6)).astype(np.uint8)
    tails[:, 0] = 0
    tails[:, 1 % num_columns] = 1
    tails[:, -1] = tails[:, 2 % num_columns]
    if num_rows > 2:
        tails[-1] = tails[0]
    return tails


def get_completion_minimum(
    tails: npt.NDArray[np.uint8], prefix: npt.NDArray[np.uint8], first: int
) -> int:
    """The least weight of the prefix plus two rows from row ``first`` on, by enumeration."""
    return min(
        int(np.sum(prefix ^ tails[i] ^ tails[j]))
        for i, j in itertools.combinations(range(first, len(tails)), 2)
    )


def get_bound_oracle(
    tails: npt.NDArray[np.uint8], prefix: npt.NDArray[np.uint8], first: int, block_bits: int
) -> int:
    """The sum over blocks of columns of the least weight on the block, by enumeration."""
    pairs = list(itertools.combinations(range(first, len(tails)), 2))
    return sum(
        min(
            int(np.sum((prefix ^ tails[i] ^ tails[j])[start : start + block_bits]))
            for i, j in pairs
        )
        for start in range(0, tails.shape[1], block_bits)
    )


def test_bounds_are_exact_relaxations(pytestconfig: pytest.Config) -> None:
    """Bounds equal their definition on blocks, and never exceed the lightest completion.

    The prefixes are sums of rows, the only prefixes that a scan bounds.  The cases include a
    partial last block, blocks in the second packed word, and tails with constant or repeated
    columns.  Merging blocks never weakens a bound.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    get_bound = cb.get_completion_bound
    for num_rows, num_columns in [(2, 1), (3, 5), (6, 21), (7, 70)]:
        tails = get_random_tails(rng, num_rows, num_columns)
        rows = pack_bits(tails)
        # The default width of 16 columns, and narrower ones.  16-column tables are slow to build
        # as plain Python for 64 or more columns.
        widths = (4, 8, 16) if num_columns < 64 else (4, 8)
        all_tables = {bits: cb.CompletionTables.build(rows, bits) for bits in widths}
        for first, _ in itertools.product(range(num_rows - 1), range(6)):
            selection = rng.integers(0, 2, num_rows).astype(bool)
            prefix = np.bitwise_xor.reduce(tails[selection], axis=0, initial=0).astype(np.uint8)
            packed = pack_bits(prefix[None])[0]
            minimum = get_completion_minimum(tails, prefix, first)
            bounds = {}
            for bits, tables in all_tables.items():
                bound = get_bound(tables.tables, tables.offsets, bits, first, packed, 10**9)
                assert bound == get_bound_oracle(tails, prefix, first, bits) <= minimum
                bounds[bits] = bound

                # A bound that stops at a limit still decides whether it reaches the limit.
                for limit in range(bound + 2):
                    stopped = get_bound(tables.tables, tables.offsets, bits, first, packed, limit)
                    assert (stopped >= limit) == (bound >= limit)
            assert all(
                bounds[narrow] <= bounds[wide] for narrow, wide in itertools.pairwise(widths)
            )


def test_bounds_ignore_ones_past_the_tails() -> None:
    """A prefix with ones past the last tail column with a one reads only its own tables.

    Such ones cannot come from sums of rows, but a bound must not read the table of another block or
    row for them.  Dropping columns only weakens a bound.
    """
    tails = np.eye(3, dtype=np.uint8)
    tables = cb.CompletionTables.build(pack_bits(tails), 8)
    prefix = pack_bits(np.array([[1, 1, 1, 0, 0, 1, 1, 1]], dtype=np.uint8))[0]
    bound = cb.get_completion_bound(tables.tables, tables.offsets, 8, 0, prefix, 10**9)
    assert bound == get_bound_oracle(tails, np.ones(3, dtype=np.uint8), 0, 8) == 1


def test_tables_fit_their_memory_limit() -> None:
    """Tables use the blocks of the tuning that fit, or else its fallback blocks.

    Tables of tails without ones skip nothing, so a search builds none.
    """
    # Three rows of tables of five eight-column blocks take 3 * 5 * 256 bytes.
    assert cb._get_block_bits(4, 40, _with_table_bytes(3 * 5 * 256)) == 8
    assert cb._get_block_bits(4, 40, _with_table_bytes(3 * 5 * 256 - 1)) == 0
    assert cb._get_block_bits(4, 40, _with_table_bytes(2**30)) == 16
    # With blocks of four columns, and two as the fallback, the tables of the same tails take
    # 3 * 10 * 16 and 3 * 20 * 4 bytes.
    narrow = dataclasses.replace(
        DEFAULT_TUNING, completion_bound_block_bits=4, completion_bound_fallback_block_bits=2
    )
    for max_table_bytes, block_bits in ((480, 4), (479, 2), (239, 0)):
        tuning = dataclasses.replace(narrow, completion_bound_max_table_bytes=max_table_bytes)
        assert cb._get_block_bits(4, 40, tuning) == block_bits
    assert cb._get_num_tail_bits(np.zeros((4, 2), dtype=np.uint64)) == 0
    assert cb._get_block_bits(4, 0, _with_table_bytes(2**30)) == 0


def test_tables_within_a_limit_are_built_only_if_they_fit_and_skip_something() -> None:
    """Tables are built only within their limit, and not for tails without ones or a single row."""
    rows = pack_bits(get_random_tails(np.random.default_rng(0), 4, 40))
    tables = cb.CompletionTables.build(rows, 8)
    built = cb.build_tables_within(rows, 8, len(tables.tables))
    assert built is not None
    np.testing.assert_array_equal(built.tables, tables.tables)
    assert cb.build_tables_within(rows, 8, len(tables.tables) - 1) is None
    assert cb.build_tables_within(np.zeros_like(rows), 8, MAX_VALUE) is None
    assert cb.build_tables_within(rows[:1], 8, MAX_VALUE) is None


def test_sampled_supports_are_counted_exactly(pytestconfig: pytest.Config) -> None:
    """The count of skipped sampled supports matches bounds computed independently."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    tails = get_random_tails(rng, 9, 37)
    rows = pack_bits(tails)
    for weight in (2, 3, 5):
        supports = np.array(list(itertools.combinations(range(9), weight)), dtype=np.int64)
        for block_bits in (8, 16):
            bounds = []
            for support in supports:
                prefix = np.bitwise_xor.reduce(tails[support[: weight - 2]], axis=0, initial=0)
                bounds.append(
                    get_bound_oracle(
                        tails, prefix.astype(np.uint8), support[weight - 2], block_bits
                    )
                )
            for limit in range(4):
                expected = sum(bound >= limit for bound in bounds)
                assert cb._count_skipped_samples(rows, supports, limit, block_bits, 37) == expected


def test_selection_follows_its_policy() -> None:
    """Tables are used if their predicted cost is within the limit, and reused for the same rows.

    With a best weight equal to the coefficient weight, every bound reaches its limit of zero, so
    tables skip every sampled support; with a best weight far above every word, they skip none.  In
    layer six of 60 rows, building tables costs a small part of the layer.  In layer two, building
    them costs far more than the layer, while looking up bounds in tables already built does not.
    """
    rows = pack_bits(get_random_tails(np.random.default_rng(0), 60, 24))
    assert cb.select_completion_tables(rows, 1, 12, None, FORCED) is None
    default = DEFAULT_TUNING
    # A layer with fewer supports than the minimum layer work is not considered.
    minimum_layer_work = math.comb(60, 6) + 1
    gated = dataclasses.replace(default, completion_bound_minimum_layer_work=minimum_layer_work)
    assert cb.select_completion_tables(rows, 6, 6, None, gated) is None

    # Tables that skip every support repay their lookups and their construction.
    finite = dataclasses.replace(default, completion_bound_minimum_layer_work=0)
    tables = cb.select_completion_tables(rows, 6, 6, None, finite)
    assert tables is not None
    # Tables that skip nothing never repay their lookups, even without building them.
    assert cb.select_completion_tables(rows, 6, 1000, tables, finite) is None
    # A small layer repays tables only if they need not be built.
    assert cb.select_completion_tables(rows, 2, 2, None, finite) is None
    assert cb.select_completion_tables(rows, 2, 2, tables, finite) is tables

    # Each coefficient of the cost model, made large enough, makes the same tables too costly, and
    # the most cautious estimate of the share that they skip still accepts them.
    for name in (
        "completion_bound_lookup_block_cost",
        "completion_bound_lookup_block_cost_per_mib",
        "completion_bound_kept_first_row_ratio",
        "completion_bound_table_byte_cost",
        "completion_bound_build_step_cost",
    ):
        change: dict[str, Any] = {name: 1e6}
        costly = dataclasses.replace(finite, **change)
        assert cb.select_completion_tables(rows, 6, 6, None, costly) is None, name
    # Tables that skip every sampled support predict about a fifth of the cost of the layer, so a
    # limit of 20% accepts them at the usual confidence, but not at the most cautious one.
    tight = dataclasses.replace(finite, completion_bound_max_cost_percent=20)
    assert cb.select_completion_tables(rows, 6, 6, None, tight) is not None
    cautious = dataclasses.replace(tight, completion_bound_confidence_z=10.0)
    assert cb.select_completion_tables(rows, 6, 6, None, cautious) is None

    # Tables laid out for other rows are never read.
    with pytest.raises(AssertionError):
        cb.select_completion_tables(rows[:30], 6, 6, tables, FORCED)

    # A limit of one byte fits no tables, and no layer has more rows than its basis.
    no_memory = dataclasses.replace(FORCED, completion_bound_max_table_bytes=1)
    assert cb.select_completion_tables(rows, 6, 6, None, no_memory) is None
    assert cb.select_completion_tables(rows[:5], 7, 7, None, FORCED) is None


def test_tunings_consider_tables_for_large_enough_layers_of_two_or_more_rows() -> None:
    """A tuning considers tables for layers from two rows to every row, with enough supports.

    Layers two and three of five rows have ten supports, and layers four and five fewer.  A tuning
    that allows no cost or no bytes of tables considers no layer.
    """

    def get_considered(**changes: Any) -> list[int]:
        """The layers of five rows for which the forced tuning, with changes, considers tables."""
        tuning = dataclasses.replace(FORCED, **changes)
        return [weight for weight in range(7) if cb.considers_tables(5, weight, tuning)]

    assert get_considered() == [2, 3, 4, 5]
    assert get_considered(completion_bound_minimum_layer_work=10) == [2, 3]
    assert get_considered(completion_bound_minimum_layer_work=11) == []
    assert get_considered(completion_bound_max_cost_percent=0) == []
    assert get_considered(completion_bound_max_table_bytes=0) == []


def test_the_largest_layer_threshold_considers_no_layer() -> None:
    """The work of a layer saturates at the cap of work estimates, below the largest threshold.

    Layer 35 of 70 rows has more supports than ``MAX_VALUE``, so only a smaller threshold considers
    tables for it.
    """
    rows = pack_bits(get_random_tails(np.random.default_rng(0), 70, 6))
    assert math.comb(70, 35) > MAX_VALUE
    largest = dataclasses.replace(FORCED, completion_bound_minimum_layer_work=MAX_VALUE)
    assert cb.select_completion_tables(rows, 35, 35, None, largest) is None
    smaller = dataclasses.replace(FORCED, completion_bound_minimum_layer_work=2**61)
    assert cb.select_completion_tables(rows, 35, 35, None, smaller) is not None


def test_selection_handles_layers_too_large_for_floating_point() -> None:
    """The cost of layers whose numbers of supports exceed floating point is still predicted.

    Layers 400 and 550 of 1100 rows have more than ``10**308`` supports.  With tails of one block of
    eight columns, the tables stay small, and the policy decides as for any other layer.
    """
    rows = pack_bits(get_random_tails(np.random.default_rng(0), 1100, 8))
    policy = dataclasses.replace(DEFAULT_TUNING, completion_bound_minimum_layer_work=0)
    for weight in (400, 550):
        assert math.comb(1100, weight) > 10**308
        cb.select_completion_tables(rows, weight, weight + 4, None, policy)


def test_estimates_need_enough_affordable_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """Estimates are exact shares of the sampled supports, or zero if too few are affordable."""
    # Every pair of these tails sums to a word of weight two, which the bound on one block proves.
    rows = pack_bits(np.eye(6, dtype=np.uint8))
    num_samples = DEFAULT_TUNING.completion_bound_samples
    assert cb._estimate_skipped_share(rows, 2, 4, 8, 6, DEFAULT_TUNING) == (1.0, num_samples)
    # The tuning sets the number of samples, the fewest that suffice, and the budget.
    fewer = dataclasses.replace(DEFAULT_TUNING, completion_bound_samples=100)
    assert cb._estimate_skipped_share(rows, 2, 4, 8, 6, fewer) == (1.0, 100)
    too_few = dataclasses.replace(fewer, completion_bound_min_samples=101)
    assert cb._estimate_skipped_share(rows, 2, 4, 8, 6, too_few) == (0.0, 0)
    no_budget = dataclasses.replace(DEFAULT_TUNING, completion_bound_sampling_work=0)
    assert cb._estimate_skipped_share(rows, 2, 4, 8, 6, no_budget) == (0.0, 0)
    # The budget bounds the samples drawn, not only those bounded, so without a budget, even the
    # most samples that a tuning allows draw no random supports.
    most = dataclasses.replace(no_budget, completion_bound_samples=MAX_SAMPLES)
    with monkeypatch.context() as patch:
        patch.setattr(np.random, "default_rng", None)
        assert cb._estimate_skipped_share(rows, 2, 4, 8, 6, most) == (0.0, 0)

    # The supports of layer two of 2000 rows each leave about a million pairs to bound directly,
    # so fewer than the minimum number of samples fit the budget.
    many_rows = np.ones((2000, 1), dtype=np.uint64)
    assert cb._estimate_skipped_share(many_rows, 2, 4, 8, 1, DEFAULT_TUNING) == (0.0, 0)


def test_share_lower_bounds_are_conservative() -> None:
    """The policy trusts a sampled share less when it has fewer samples.

    For 1005 skipped supports in 1024 samples, the Wilson interval that extends two standard errors
    on each side starts at 97.09%.
    """
    assert cb._get_share_lower_bound(0.5, 0, Z) == 0
    assert cb._get_share_lower_bound(0.0, 100, Z) == 0
    assert 0 < cb._get_share_lower_bound(1.0, 64, Z) < cb._get_share_lower_bound(1.0, 256, Z) < 1
    assert cb._get_share_lower_bound(1005 / 1024, 1024, Z) == pytest.approx(0.97095, abs=1e-4)
