# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tables that bound the completions of BZ supports by two more rows."""

from __future__ import annotations

import dataclasses
import itertools
import math

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import pack_bits

from .._tuning import COMPLETION_SAMPLES, MAX_VALUE
from . import completion_bounds as cb

# A policy that considers every layer, uses tables whatever they cost, and allows any size.
FORCED = cb.CompletionPolicy(0, MAX_VALUE, MAX_VALUE)


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


@pytest.mark.parametrize("interpreted", [False, True])
def test_bounds_are_exact_relaxations(
    interpreted: bool, pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bounds equal their definition on blocks, and never exceed the lightest completion.

    The prefixes are sums of rows, the only prefixes that a scan bounds.  The cases include a
    partial last block, blocks in the second packed word, and tails with constant or repeated
    columns.  Merging blocks never weakens a bound.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    get_bound = cb.get_completion_bound
    if interpreted:
        monkeypatch.setattr(
            cb, "_fill_tables", getattr(cb._fill_tables, "py_func", cb._fill_tables)
        )
        get_bound = getattr(get_bound, "py_func", get_bound)
    for num_rows, num_columns in [(2, 1), (3, 5), (6, 21), (7, 70)]:
        tails = get_random_tails(rng, num_rows, num_columns)
        rows = pack_bits(tails)
        all_tables = {bits: cb.CompletionTables.build(rows, bits) for bits in (8, 16)}
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
            assert bounds[8] <= bounds[16]


def test_bounds_ignore_ones_past_the_tails() -> None:
    """A prefix with ones past the last tail column with a one reads only its own tables.

    Such ones cannot come from sums of rows, but a bound must not read the table of another block or
    row for them.  Dropping columns only weakens a bound.
    """
    tails = np.eye(3, dtype=np.uint8)
    tables = cb.CompletionTables.build(pack_bits(tails), 8)
    prefix = pack_bits(np.array([[1, 1, 1, 0, 0, 1, 1, 1]], dtype=np.uint8))[0]
    bound = cb.get_completion_bound.py_func(tables.tables, tables.offsets, 8, 0, prefix, 10**9)
    assert bound == get_bound_oracle(tails, np.ones(3, dtype=np.uint8), 0, 8) == 1


def test_tables_fit_their_memory_limit() -> None:
    """Tables use the widest blocks that fit, and only supported block widths.

    Tables of tails without ones skip nothing, so a search builds none.
    """
    with pytest.raises(ValueError, match="not supported"):
        cb.CompletionTables.build(pack_bits(np.eye(4, 40, 3, dtype=np.uint8)), 12)
    # Three rows of tables of five eight-column blocks take 3 * 5 * 256 bytes.
    assert cb._get_block_bits(4, 40, 3 * 5 * 256) == 8
    assert cb._get_block_bits(4, 40, 3 * 5 * 256 - 1) == 0
    assert cb._get_block_bits(4, 40, 2**30) == 16
    assert cb._get_num_tail_bits(np.zeros((4, 2), dtype=np.uint64)) == 0
    assert cb._get_block_bits(4, 0, 2**30) == 0


@pytest.mark.parametrize("interpreted", [False, True])
def test_sampled_supports_are_counted_exactly(
    interpreted: bool, pytestconfig: pytest.Config
) -> None:
    """The count of skipped sampled supports matches bounds computed independently."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    count = cb._count_skipped_samples
    if interpreted:
        count = getattr(count, "py_func", count)
    tails = get_random_tails(rng, 9, 37)
    rows = pack_bits(tails)
    for weight in (2, 3, 5):
        supports = np.array(list(itertools.combinations(range(9), weight)), dtype=np.int64)
        for limit, block_bits in itertools.product(range(4), (8, 16)):
            expected = 0
            for support in supports:
                prefix = np.bitwise_xor.reduce(tails[support[: weight - 2]], axis=0, initial=0)
                bound = get_bound_oracle(
                    tails, prefix.astype(np.uint8), support[weight - 2], block_bits
                )
                expected += bound >= limit
            assert count(rows, supports, limit, block_bits, 37) == expected


def test_selection_follows_its_policy() -> None:
    """Tables are used if their predicted cost is within the limit, and reused for the same rows.

    With a best weight equal to the coefficient weight, every bound reaches its limit of zero, so
    tables skip every sampled support; with a best weight far above every word, they skip none.  In
    layer six of 60 rows, building tables costs a small part of the layer.  In layer two, building
    them costs far more than the layer, while looking up bounds in tables already built does not.
    """
    rows = pack_bits(get_random_tails(np.random.default_rng(0), 60, 24))
    assert cb.select_completion_tables(rows, 1, 12, None, FORCED) is None
    default = cb.CompletionPolicy(100_000_000, 90, 2**28)
    assert cb.select_completion_tables(rows, 6, 6, None, default) is None

    # Tables that skip every support repay their lookups and their construction.
    finite = dataclasses.replace(default, minimum_layer_work=0)
    tables = cb.select_completion_tables(rows, 6, 6, None, finite)
    assert tables is not None
    # Tables that skip nothing never repay their lookups, even without building them.
    assert cb.select_completion_tables(rows, 6, 1000, tables, finite) is None
    # A small layer repays tables only if they need not be built.
    assert cb.select_completion_tables(rows, 2, 2, None, finite) is None
    assert cb.select_completion_tables(rows, 2, 2, tables, finite) is tables

    # Tables laid out for other rows are never read.
    with pytest.raises(AssertionError):
        cb.select_completion_tables(rows[:30], 6, 6, tables, FORCED)

    # A limit of zero bytes allows no tables, and no layer has more rows than its basis.
    no_memory = dataclasses.replace(FORCED, max_table_bytes=0)
    assert cb.select_completion_tables(rows, 6, 6, None, no_memory) is None
    assert cb.select_completion_tables(rows[:5], 7, 7, None, FORCED) is None


def test_the_largest_layer_threshold_considers_no_layer() -> None:
    """The work of a layer saturates at the cap of work estimates, below the largest threshold.

    Layer 35 of 70 rows has more supports than ``MAX_VALUE``, so only a smaller threshold considers
    tables for it.
    """
    rows = pack_bits(get_random_tails(np.random.default_rng(0), 70, 6))
    assert math.comb(70, 35) > MAX_VALUE
    largest = dataclasses.replace(FORCED, minimum_layer_work=MAX_VALUE)
    assert cb.select_completion_tables(rows, 35, 35, None, largest) is None
    smaller = dataclasses.replace(FORCED, minimum_layer_work=2**61)
    assert cb.select_completion_tables(rows, 35, 35, None, smaller) is not None


def test_selection_handles_layers_too_large_for_floating_point() -> None:
    """The cost of layers whose numbers of supports exceed floating point is still predicted.

    Layers 400 and 550 of 1100 rows have more than ``10**308`` supports.  With tails of one block of
    eight columns, the tables stay small, and the policy decides as for any other layer.
    """
    rows = pack_bits(get_random_tails(np.random.default_rng(0), 1100, 8))
    policy = cb.CompletionPolicy(0, 90, 2**28)
    for weight in (400, 550):
        assert math.comb(1100, weight) > 10**308
        cb.select_completion_tables(rows, weight, weight + 4, None, policy)


def test_estimates_need_enough_affordable_samples() -> None:
    """Estimates are exact shares of the sampled supports, or zero if too few are affordable."""
    # Every pair of these tails sums to a word of weight two, which the bound on one block proves.
    rows = pack_bits(np.eye(6, dtype=np.uint8))
    assert cb._estimate_skipped_share(rows, 2, 4, 8, 6) == (1.0, COMPLETION_SAMPLES)

    # The supports of layer two of 2000 rows each leave about a million pairs to bound directly,
    # so fewer than the minimum number of samples fit the budget.
    many_rows = np.ones((2000, 1), dtype=np.uint64)
    assert cb._estimate_skipped_share(many_rows, 2, 4, 8, 1) == (0.0, 0)


def test_share_lower_bounds_are_conservative() -> None:
    """The policy trusts a sampled share less when it has fewer samples.

    For 1005 skipped supports in 1024 samples, the Wilson interval that extends two standard errors
    on each side starts at 97.09%.
    """
    assert cb._get_share_lower_bound(0.5, 0) == 0
    assert cb._get_share_lower_bound(0.0, 100) == 0
    assert 0 < cb._get_share_lower_bound(1.0, 64) < cb._get_share_lower_bound(1.0, 256) < 1
    assert cb._get_share_lower_bound(1005 / 1024, 1024) == pytest.approx(0.97095, abs=1e-4)
