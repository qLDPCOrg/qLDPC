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

from .._tuning import DEFAULT_TUNING, MAX_VALUE
from . import completion_bounds as cb

Z = DEFAULT_TUNING.completion_bound_confidence_z
MAX_SAMPLES = 2**12
# A tuning that considers every layer, uses tables whatever they cost, and allows any size.  The
# fewest samples that the cost model trusts keep its predictions quick as plain Python.
FORCED = dataclasses.replace(
    DEFAULT_TUNING,
    completion_bound_minimum_layer_work=0,
    completion_bound_max_cost_percent=MAX_VALUE,
    completion_bound_max_table_bytes=MAX_VALUE,
    completion_bound_samples=DEFAULT_TUNING.completion_bound_min_samples,
)


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
    """Bounds on sums of rows equal their definition and never exceed the lightest completion."""
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

                for limit in range(bound + 2):
                    stopped = get_bound(tables.tables, tables.offsets, bits, first, packed, limit)
                    assert (stopped >= limit) == (bound >= limit)
            assert all(
                bounds[narrow] <= bounds[wide] for narrow, wide in itertools.pairwise(widths)
            )


def get_value(bits: npt.NDArray[np.uint8]) -> int:
    """The value of a block of columns, with its first column in the lowest bit."""
    return int(bits.astype(np.int64) @ (1 << np.arange(len(bits))))


def get_block_minima(
    tails: npt.NDArray[np.uint8], start: int, block_bits: int
) -> npt.NDArray[np.int64]:
    """The least weights on a block of its values plus two rows from each row on, by enumeration.

    The block has the columns from ``start`` on, at most ``block_bits`` of them.  Entry ``[s, x]``
    is the least weight on the block of ``x`` plus two rows from row ``s`` on.
    """
    block = tails[:, start : start + block_bits]
    values = np.array([get_value(row) for row in block])
    points = np.arange(1 << block.shape[1])
    weights = np.array([point.bit_count() for point in range(len(points))])
    num_rows = len(tails)
    minima = np.empty((num_rows - 1, len(points)), dtype=np.int64)
    for first in range(num_rows - 1):
        sums = (values[:, None] ^ values[None, :])[first:, first:]
        pairs = np.unique(sums[np.triu_indices(num_rows - first, 1)])
        minima[first] = weights[points[:, None] ^ pairs[None, :]].min(axis=1)
    return minima


def test_every_entry_of_the_tables_matches_its_definition(pytestconfig: pytest.Config) -> None:
    """Every entry of the tables of random tails matches its definition.

    The fill transposes the tables of 64 rows at once, four words of each at a time.  The first
    tails have fewer rows of tables, on blocks of fewer than eight values.  The second have more, on
    blocks of 256 values.  Where blocks have more than one column, the last block is narrower.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_rows, block_bits in [(63, int(rng.choice([1, 2]))), (66, 8)]:
        last_bits = int(rng.integers(1, block_bits)) if block_bits > 1 else 1
        num_columns = block_bits * int(rng.integers(1, 3)) + last_bits
        tails = get_random_tails(rng, num_rows, num_columns)
        # A one in the last column keeps the last block as wide as the columns make it.
        tails[rng.integers(num_rows), -1] = 1
        tables = cb.CompletionTables.build(pack_bits(tails), block_bits)
        offsets = tables.offsets
        entries = tables.tables.reshape(offsets[-1], num_rows - 1)
        for block, start in enumerate(range(0, num_columns, block_bits)):
            block_entries = entries[offsets[block] : offsets[block + 1]]
            np.testing.assert_array_equal(
                block_entries.T, get_block_minima(tails, start, block_bits)
            )


def test_tables_without_blocks_bound_nothing() -> None:
    """Tables of tails without ones, and ``NO_TABLES``, give the bound zero."""
    rows = np.zeros((2, 1), dtype=np.uint64)
    prefix = np.zeros(1, dtype=np.uint64)
    for tables in (cb.CompletionTables.build(rows, 4), cb.NO_TABLES):
        bound = cb.get_completion_bound(tables.tables, tables.offsets, 4, 0, prefix, 10**9)
        assert bound == 0


def test_bounds_ignore_ones_past_the_tails() -> None:
    """A prefix with ones past the last tail column with a one reads only its own tables."""
    tails = np.eye(3, dtype=np.uint8)
    tables = cb.CompletionTables.build(pack_bits(tails), 8)
    prefix = pack_bits(np.array([[1, 1, 1, 0, 0, 1, 1, 1]], dtype=np.uint8))[0]
    bound = cb.get_completion_bound(tables.tables, tables.offsets, 8, 0, prefix, 10**9)
    assert bound == get_bound_oracle(tails, np.ones(3, dtype=np.uint8), 0, 8) == 1


def test_tables_fit_their_memory_limit() -> None:
    """Tables fit only within their memory limit."""
    # Three rows of tables of three sixteen-column blocks (the last of eight columns) take
    # 3 * (2 * 2**16 + 2**8) bytes.
    fitting_bytes = 3 * (2 * 2**16 + 2**8)
    assert cb._tables_fit(4, 40, 16, fitting_bytes)
    assert not cb._tables_fit(4, 40, 16, fitting_bytes - 1)
    # With blocks of four columns, the tables of the same tails take 3 * 10 * 16 bytes.
    assert cb._tables_fit(4, 40, 4, 480)
    assert not cb._tables_fit(4, 40, 4, 479)


def test_tables_within_a_limit_fit_only_if_they_skip_something_and_build_in_place() -> None:
    """Tables that skip something fit only within their limit, and fill exactly a given array."""
    rows = pack_bits(get_random_tails(np.random.default_rng(0), 4, 40))
    tables = cb.CompletionTables.build(rows, 8)
    num_bytes = len(tables.tables)
    assert cb.get_table_bytes_within(rows, 8, num_bytes) == num_bytes
    assert cb.get_table_bytes_within(rows, 8, MAX_VALUE) == num_bytes
    assert cb.get_table_bytes_within(rows, 8, num_bytes - 1) == 0
    assert cb.get_table_bytes_within(np.zeros_like(rows), 8, MAX_VALUE) == 0
    assert cb.get_table_bytes_within(rows[:1], 8, MAX_VALUE) == 0

    array = np.full(num_bytes + 2, 255, dtype=np.uint8)
    cb.CompletionTables.build(rows, 8, out=array[1:-1])
    np.testing.assert_array_equal(array[1:-1], tables.tables)
    assert array[0] == array[-1] == 255
    for size in (num_bytes - 1, num_bytes + 1):
        with pytest.raises(AssertionError):
            cb.CompletionTables.build(rows, 8, out=np.empty(size, dtype=np.uint8))


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
    """Tables are used if their predicted cost is within the limit, and reused for the same rows."""
    rows = pack_bits(get_random_tails(np.random.default_rng(0), 60, 24))
    assert cb.select_completion_tables(rows, 1, 12, None, FORCED) is None

    # With the best weight six, every bound in layer six reaches its limit of zero, so tables skip
    # every support, and repay their lookups and their build.  The tuning sets every setting of the
    # cost model, so that retuning the defaults does not change these cases.
    pinned = dataclasses.replace(
        FORCED,
        completion_bound_max_cost_percent=100,
        completion_bound_block_bits=8,
        completion_bound_samples=64,
        completion_bound_min_samples=64,
        completion_bound_sampling_work=5 * 10**7,
        completion_bound_confidence_z=2.0,
        completion_bound_lookup_block_cost=2.0,
        completion_bound_lookup_block_cost_per_mib=0.04,
        completion_bound_kept_first_row_ratio=0.35,
        completion_bound_table_byte_cost=0.1,
        completion_bound_build_step_cost=0.2,
    )
    tables = cb.select_completion_tables(rows, 6, 6, None, pinned)
    assert tables is not None
    # A layer with fewer supports than the minimum layer work is not considered.
    gated = dataclasses.replace(pinned, completion_bound_minimum_layer_work=10**12)
    assert cb.select_completion_tables(rows, 6, 6, None, gated) is None
    # Tables that skip nothing, as for a best weight of 1000, never repay their lookups, even when
    # already built.
    assert cb.select_completion_tables(rows, 6, 1000, tables, pinned) is None
    # In layer two, a build costs more than the layer, but lookups in built tables do not.
    assert cb.select_completion_tables(rows, 2, 2, None, pinned) is None
    assert cb.select_completion_tables(rows, 2, 2, tables, pinned) is tables

    # Each coefficient of the cost model, made large enough, makes the same tables too costly.
    for name in (
        "completion_bound_lookup_block_cost",
        "completion_bound_lookup_block_cost_per_mib",
        "completion_bound_kept_first_row_ratio",
        "completion_bound_table_byte_cost",
        "completion_bound_build_step_cost",
    ):
        change: dict[str, Any] = {name: 1e6}
        costly = dataclasses.replace(pinned, **change)
        assert cb.select_completion_tables(rows, 6, 6, None, costly) is None, name
    # With free lookups, built tables that skip every one of the 64 sampled supports predict the
    # share 1 / (1 + 64 / z**2) of the cost of the layer: about 6 % at the usual confidence, z = 2,
    # but about 61 % at the most cautious, z = 10, so the limit of 10 % accepts the first and
    # rejects the second.
    free_lookups = dataclasses.replace(
        pinned,
        completion_bound_max_cost_percent=10,
        completion_bound_lookup_block_cost=0.0,
        completion_bound_lookup_block_cost_per_mib=0.0,
    )
    assert cb.select_completion_tables(rows, 6, 6, tables, free_lookups) is tables
    cautious = dataclasses.replace(free_lookups, completion_bound_confidence_z=10.0)
    assert cb.select_completion_tables(rows, 6, 6, tables, cautious) is None

    # Tables laid out for other rows are never read.
    with pytest.raises(AssertionError):
        cb.select_completion_tables(rows[:30], 6, 6, tables, FORCED)

    # No layer has more rows than its basis.
    assert cb.select_completion_tables(rows[:5], 7, 7, None, FORCED) is None

    # Tables use blocks of the width of the tuning: four-column tables of these rows take
    # 59 * 6 * 16 bytes, 6 blocks of 16 entries for each row but the last; sixteen-column tables
    # would not fit.
    small = dataclasses.replace(FORCED, completion_bound_max_table_bytes=59 * 6 * 16)
    narrow = dataclasses.replace(small, completion_bound_block_bits=4)
    narrow_tables = cb.select_completion_tables(rows, 6, 6, None, narrow)
    assert narrow_tables is not None and narrow_tables.block_bits == 4
    wide = dataclasses.replace(small, completion_bound_block_bits=16)
    assert cb.select_completion_tables(rows, 6, 6, None, wide) is None


def test_tunings_consider_tables_for_large_enough_layers_of_two_or_more_rows() -> None:
    """Tunings consider tables for layers of two or more rows with enough supports."""

    def get_considered(**changes: Any) -> list[int]:
        """The layers of five rows for which the forced tuning, with changes, considers tables."""
        tuning = dataclasses.replace(FORCED, **changes)
        return [weight for weight in range(7) if cb.considers_tables(5, weight, tuning)]

    assert get_considered() == [2, 3, 4, 5]
    # Of 30 rows, layer two has 435 supports, and layer 15 about 1.6 * 10**8.
    gated = dataclasses.replace(FORCED, completion_bound_minimum_layer_work=10**5)
    assert cb.considers_tables(30, 15, gated) and not cb.considers_tables(30, 2, gated)
    assert get_considered(completion_bound_max_cost_percent=0) == []
    assert get_considered(completion_bound_max_table_bytes=0) == []


def test_the_largest_layer_threshold_considers_no_layer() -> None:
    """The work of a layer saturates at the cap of work estimates, below the largest threshold."""
    rows = pack_bits(get_random_tails(np.random.default_rng(0), 70, 6))
    assert math.comb(70, 35) > MAX_VALUE
    largest = dataclasses.replace(FORCED, completion_bound_minimum_layer_work=MAX_VALUE)
    assert cb.select_completion_tables(rows, 35, 35, None, largest) is None
    smaller = dataclasses.replace(FORCED, completion_bound_minimum_layer_work=2**61)
    assert cb.select_completion_tables(rows, 35, 35, None, smaller) is not None


def test_the_cost_of_layers_too_large_for_floating_point_is_predicted() -> None:
    """The cost of layers whose numbers of supports exceed floating point is still predicted."""
    rows = pack_bits(get_random_tails(np.random.default_rng(0), 1100, 8))
    for weight in (400, 550):
        assert math.comb(1100, weight) > 10**308
        cost, _ = cb.predict_completion_cost(rows, weight, weight + 4, reuse=False, tuning=FORCED)
        assert math.isfinite(cost)


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
    default_rng = np.random.default_rng
    generators: list[np.random.Generator] = []

    def record(*args: Any, **kwargs: Any) -> np.random.Generator:
        generators.append(default_rng(*args, **kwargs))
        return generators[-1]

    with monkeypatch.context() as patch:
        patch.setattr(np.random, "default_rng", record)
        assert cb._estimate_skipped_share(rows, 2, 4, 8, 6, most) == (0.0, 0)
        assert not generators
        # Under a tuning with a budget, the call draws random supports.
        assert cb._estimate_skipped_share(rows, 2, 4, 8, 6, fewer) == (1.0, 100)
        assert generators

    # The supports of layer two of 2000 rows each leave about a million pairs to bound directly,
    # so fewer than the minimum number of samples fit the budget.
    many_rows = np.ones((2000, 1), dtype=np.uint64)
    assert cb._estimate_skipped_share(many_rows, 2, 4, 8, 1, DEFAULT_TUNING) == (0.0, 0)


def test_share_lower_bounds_are_conservative() -> None:
    """The policy trusts a sampled share less when it has fewer samples."""
    assert cb._get_share_lower_bound(0.5, 0, Z) == 0
    assert cb._get_share_lower_bound(0.0, 100, Z) == 0
    assert 0 < cb._get_share_lower_bound(1.0, 64, Z) < cb._get_share_lower_bound(1.0, 256, Z) < 1
    assert cb._get_share_lower_bound(1005 / 1024, 1024, Z) == pytest.approx(0.97095, abs=1e-4)
