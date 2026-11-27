# SPDX-License-Identifier: Apache-2.0

"""Tests for the compiled scans of layers that distance and coset searches share."""

from __future__ import annotations

import itertools
import math
import threading
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import pack_bits
from qldpc._compilation import is_compilation_disabled

from .. import _threads
from .._information_sets import completion_bounds
from . import affine_search


def test_scans_of_sets_of_any_rank() -> None:
    """Chunked scans of a set weigh each combination of its rows once, whatever its rank."""
    rng = np.random.default_rng(120)
    no_tables = completion_bounds.NO_TABLES

    # The sets have full rank, partial rank, and rank zero.  Scans of the last two read fewer packed
    # words of the rows than the sets store, and these sets have labels only in their second word,
    # while the first set has none.
    for dimension in range(1, 7):
        generators = rng.integers(0, 2**64, (3, dimension, 5), dtype=np.uint64)
        labels = np.zeros((3, dimension, 2), dtype=np.uint64)
        labels[1:, :, 1] = rng.integers(0, 2, (2, dimension), dtype=np.uint64)
        ranks = np.array([dimension, dimension // 2, 0], dtype=np.int64)
        widths = np.array([5, 2, 1], dtype=np.int64)
        label_widths = np.array([0, 2, 2], dtype=np.int64)
        for index, weight in itertools.product(range(3), range(1, dimension + 1)):
            support = np.zeros(weight, dtype=np.int64)
            arguments = (
                generators[index],
                np.ascontiguousarray(labels[index, :, : label_widths[index]]),
                ranks[index],
                widths[index],
                support,
                np.zeros((weight + 1, 5), dtype=np.uint64),
            )
            depth, best, expected = 0, 1000, 1000
            for indices in itertools.combinations(range(dimension), weight):
                # Update the expected minimum with the current combination of rows.
                words = np.bitwise_xor.reduce(generators[index, indices, : widths[index]])
                eligible = not label_widths[index] or np.any(
                    np.bitwise_xor.reduce(labels[index, indices], axis=0)
                )
                candidate = sum(int(word).bit_count() for word in words)
                candidate += sum(row < ranks[index] for row in indices)
                if eligible:
                    expected = min(expected, candidate)

                # Scan one combination of rows.
                best, depth, *visited = affine_search.scan_set(
                    *arguments,
                    depth,
                    best,
                    0,
                    1,
                    no_tables.tables,
                    no_tables.offsets,
                    no_tables.block_bits,
                )
                assert visited == [1, 0, 0] and best == expected
                np.testing.assert_array_equal(support[:-1], indices[:-1])
                assert support[-1] == indices[-1] + 1

            # One more call finds the layer complete.
            best, depth, *visited = affine_search.scan_set(
                *arguments,
                depth,
                best,
                0,
                1,
                no_tables.tables,
                no_tables.offsets,
                no_tables.block_bits,
            )
            assert (best, depth, visited) == (expected, -1, [0, 0, 0])


def test_scan_of_one_information_set(pytestconfig: pytest.Config) -> None:
    """Chunked scans of one information set's search basis find a layer's lightest eligible word.

    It checks "Skip only words that a bound proves heavy enough" of
    ``brouwer_zimmermann_code.md#sec-requirements``.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    no_tables = completion_bounds.NO_TABLES
    for dimension, num_label_words in itertools.product(range(2, 7), (0, 2)):
        tails = (rng.random((dimension, 70)) < 0.3).astype(np.uint8)
        generators = pack_bits(tails)
        labels = np.zeros((dimension, num_label_words), dtype=np.uint64)
        if num_label_words:
            labels[:, 1] = rng.integers(0, 4, dimension, dtype=np.uint64)
        width = generators.shape[1]
        # Tables on blocks of 4 and 8 columns, which leave a narrower last block, build quickly as
        # plain Python.
        all_tables = [no_tables] + [
            completion_bounds.CompletionTables.build(generators, bits) for bits in (4, 8)
        ]
        for weight in range(2, dimension + 1):
            # The weight of each support in order, or None if its word is ineligible.
            weights = [
                weight + int(np.sum(np.bitwise_xor.reduce(tails[list(rows)], axis=0)))
                if not num_label_words or np.any(np.bitwise_xor.reduce(labels[list(rows)]))
                else None
                for rows in itertools.combinations(range(dimension), weight)
            ]
            eligible = [value for value in weights if value is not None]
            for initial_best, tables, batch_size in itertools.product(
                (weight + 1, min(eligible, default=1000) + 1, 1000), all_tables, (1, 5, 10**6)
            ):
                expected = min([initial_best, *eligible])
                for cutoff in (-1, expected):
                    support = np.zeros(weight, dtype=np.int64)
                    prefixes = np.zeros((weight + 1, width), dtype=np.uint64)
                    depth, best, evaluated, skipped = 0, initial_best, 0, 0
                    while depth >= 0 and best > cutoff:
                        best, depth, num_evaluated, num_lookups, num_skipped = (
                            affine_search.scan_set(
                                generators,
                                labels,
                                dimension,
                                width,
                                support,
                                prefixes,
                                depth,
                                best,
                                cutoff,
                                batch_size,
                                tables.tables,
                                tables.offsets,
                                tables.block_bits,
                            )
                        )
                        assert num_evaluated + num_lookups <= batch_size
                        evaluated += num_evaluated
                        skipped += num_skipped
                    assert best == expected
                    if cutoff < 0:
                        assert evaluated + skipped == math.comb(dimension, weight)
                        if tables is not no_tables and initial_best == expected:
                            limit = initial_best - weight
                            assert skipped == sum(
                                _get_completion_bound(tails, rows, tables.block_bits) >= limit
                                for rows in itertools.combinations(range(dimension), weight)
                            )
                    elif tables is no_tables and initial_best > expected:
                        running = itertools.accumulate(
                            (value if value is not None else initial_best for value in weights),
                            min,
                            initial=initial_best,
                        )
                        assert evaluated == next(
                            index for index, value in enumerate(running) if value <= cutoff
                        )


def test_scans_of_the_work_items_of_a_layer_add_up_to_the_scan_of_the_layer() -> None:
    """Scans of ranges of the work items of a layer, in order, find and count what its scan does.

    The items fix the first one to ``weight - 2`` rows of the supports, and the ranges hold one item
    or up to four.  The sets have full and partial rank, and the scans of the set of full rank also
    run with tables, which skip some completions.  Scans of the ranges on three threads at once,
    from the best weight of the layer, which none of them lowers, count what the scan of the layer
    from that weight does.

    It checks "Keep the scans of a layer splittable across threads" of
    ``brouwer_zimmermann_code.md#sec-requirements``.
    """
    rng = np.random.default_rng(7)
    num_rows, weight = 8, 5
    tails = (rng.random((num_rows, 40)) < 0.3).astype(np.uint8)
    generators = pack_bits(tails)
    labels = rng.integers(0, 2, (num_rows, 1), dtype=np.uint64)
    no_tables = completion_bounds.NO_TABLES
    tables = completion_bounds.CompletionTables.build(generators, 4)
    for rank, set_tables, initial_best, batch_size in itertools.product(
        (num_rows, 3), (no_tables, tables), (weight + 3, 1000), (1, 7)
    ):
        if rank < num_rows and set_tables is tables:
            continue
        arguments = (generators, labels, rank, generators.shape[1])
        table_arguments = (set_tables.tables, set_tables.offsets, set_tables.block_bits)

        # Scan the layer.
        scan_arguments = (*arguments, weight, batch_size, table_arguments)
        expected_best, expected_counts = _scan_layer(*scan_arguments, initial_best)
        assert expected_counts[2] > 0 or set_tables is no_tables
        support = np.zeros(weight, dtype=np.int64)
        prefixes = np.zeros((weight + 1, generators.shape[1]), dtype=np.uint64)

        # Scan the ranges of work items of the layer in order, each from its first support.
        for length, range_size in itertools.product(range(1, weight - 1), (1, 4)):
            items = [
                np.array(rows, dtype=np.int64)
                for rows in itertools.combinations(range(num_rows - weight + length), length)
            ]
            best, item_counts = initial_best, np.zeros(3, dtype=np.int64)
            for start in range(0, len(items), range_size):
                last_item = items[min(start + range_size, len(items)) - 1]
                depth = affine_search.start_set_items(generators, items[start], support, prefixes)
                while depth >= 0:
                    best, depth, *counts = affine_search.scan_set_items(
                        *arguments,
                        support,
                        prefixes,
                        depth,
                        best,
                        -1,
                        batch_size,
                        *table_arguments,
                        last_item,
                    )
                    assert counts[0] + counts[1] <= batch_size
                    item_counts += counts
            assert best == expected_best
            np.testing.assert_array_equal(item_counts, expected_counts)

        # Scan the items that fix one row on threads, from the best weight of the layer.
        _, final_counts = _scan_layer(*scan_arguments, expected_best)
        thread_counts = _scan_items_on_threads(*scan_arguments, expected_best)
        np.testing.assert_array_equal(thread_counts, final_counts)


def test_scans_of_work_items_run_without_the_global_interpreter_lock() -> None:
    """Threads can run the compiled scans of work items at once, unless they run as plain Python."""
    assert is_compilation_disabled() or affine_search.scan_set_items.targetoptions["nogil"]


@pytest.mark.parametrize("css_weight", [False, True])
def test_coset_layer_scans_keep_the_first_lightest_word(
    css_weight: bool, pytestconfig: pytest.Config
) -> None:
    """A coset layer scan keeps each word lighter than the best so far, and stops at the cutoff."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_rows, num_bits = 6, 100
    row_bits = (rng.random((num_rows, num_bits)) < 0.3).astype(np.uint8)
    offset_bits = rng.integers(0, 2, num_bits, dtype=np.uint8)
    x_bits = rng.integers(0, 2, num_bits, dtype=np.uint8)
    rows = pack_bits(row_bits)
    offset, x_mask = pack_bits(np.stack([offset_bits, x_bits]))
    width = rows.shape[1]

    def get_weight(bits: npt.NDArray[np.uint8]) -> int:
        if not css_weight:
            return int(bits.sum())
        return int(max((bits & x_bits).sum(), (bits & (1 - x_bits)).sum()))

    for layer in range(1, num_rows + 1):
        words = [
            offset_bits ^ np.bitwise_xor.reduce(row_bits[list(rows_of_support)], axis=0)
            for rows_of_support in itertools.combinations(range(num_rows), layer)
        ]
        weights = [get_weight(word) for word in words]
        for initial_best, cutoff in itertools.product(
            (min(weights), min(weights) + 1, max(weights) + 1), (-1, min(weights), max(weights))
        ):
            # The expected best weight, the last word kept, and the number of words weighed.
            expected_best, expected_word, expected_weighed = initial_best, None, len(words)
            for index, (word, weight) in enumerate(zip(words, weights, strict=True)):
                if weight < expected_best:
                    expected_best, expected_word = weight, pack_bits(word[None])[0]
                if expected_best <= cutoff:
                    expected_weighed = index + 1
                    break

            representative = np.full(width, 7, dtype=np.uint64)
            support = np.empty(layer, dtype=np.int64)
            prefixes = np.empty((layer, width), dtype=np.uint64)
            if css_weight:
                best, weighed = affine_search.scan_coset_layer_by_css_weight(
                    offset,
                    rows,
                    width,
                    support,
                    prefixes,
                    initial_best,
                    cutoff,
                    x_mask,
                    representative,
                )
            else:
                best, weighed = affine_search.scan_coset_layer(
                    offset, rows, width, support, prefixes, initial_best, cutoff, representative
                )
            assert (best, weighed) == (expected_best, expected_weighed)
            if expected_word is None:
                assert np.all(representative == 7)
            else:
                np.testing.assert_array_equal(representative, expected_word)


@pytest.mark.parametrize("css_weight", [False, True])
def test_coset_layer_scans_keep_the_first_of_equally_light_words(css_weight: bool) -> None:
    """A scan keeps the first of equally light words."""
    rows = np.array([[0b001], [0b010], [0b100]], dtype=np.uint64)
    arguments = (
        np.zeros(1, dtype=np.uint64),
        rows,
        1,
        np.empty(1, dtype=np.int64),
        np.empty((1, 1), dtype=np.uint64),
        5,
        -1,
    )
    representative = np.zeros(1, dtype=np.uint64)
    if css_weight:
        result = affine_search.scan_coset_layer_by_css_weight(
            *arguments, np.full(1, 0b111, dtype=np.uint64), representative
        )
    else:
        result = affine_search.scan_coset_layer(*arguments, representative)
    assert result == (1, 3)
    assert representative[0] == 0b001


def test_coset_layer_scans_reach_their_last_support() -> None:
    """A scan of a layer of a coset weighs its last support."""
    for num_rows in range(1, 7):
        rows = (np.uint64(1) << np.arange(num_rows, dtype=np.uint64)).reshape(num_rows, 1)
        for layer in range(1, num_rows + 1):
            offset = np.bitwise_xor.reduce(rows[num_rows - layer :])
            representative = np.ones(1, dtype=np.uint64)
            best, weighed = affine_search.scan_coset_layer(
                offset,
                rows,
                1,
                np.empty(layer, dtype=np.int64),
                np.empty((layer, 1), dtype=np.uint64),
                num_rows + 1,
                0,
                representative,
            )
            assert (best, weighed) == (0, math.comb(num_rows, layer))
            assert representative[0] == 0


def test_coset_tail_scans_weigh_words_by_their_pivot_rows_and_tails(
    pytestconfig: pytest.Config,
) -> None:
    """A coset tail scan gives each word its full Hamming weight, and keeps the first lightest."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_rows = 5
    tails = (rng.random((num_rows, 70)) < 0.3).astype(np.uint8)
    offset_tail = rng.integers(0, 2, 70, dtype=np.uint8)
    packed_tails, packed_offset = pack_bits(tails), pack_bits(offset_tail[None])[0]
    width = packed_tails.shape[1]
    no_tables = completion_bounds.NO_TABLES
    for rank, layer in itertools.product((num_rows, num_rows - 1, 0), range(1, num_rows + 1)):
        weights = _get_tail_weights(offset_tail, tails, rank, layer)
        least = min(weight for _, weight in weights)
        for initial_best, cutoff in itertools.product((least, least + 1, 100), (-1, least)):
            # The expected best weight, the support of the last word kept, and the words weighed.
            expected_best, expected_support, expected_weighed = initial_best, None, len(weights)
            for index, (rows, weight) in enumerate(weights):
                if weight < expected_best:
                    expected_best, expected_support = weight, rows
                if expected_best <= cutoff:
                    expected_weighed = index + 1
                    break

            # Scan the layer without pair steps, and with them, without tables, from two rows.
            for pair_steps in (False, True) if layer >= 2 else (False,):
                best_support = np.full(layer, -1, dtype=np.int64)
                arguments = (
                    packed_offset,
                    packed_tails,
                    rank,
                    width,
                    np.empty(layer, dtype=np.int64),
                    np.empty((layer, width), dtype=np.uint64),
                    initial_best,
                    cutoff,
                )
                if pair_steps:
                    best, weighed, skipped = affine_search.scan_coset_layer_by_tails_with_tables(
                        *arguments,
                        0,
                        no_tables.tables,
                        no_tables.offsets,
                        no_tables.block_bits,
                        best_support,
                    )
                    assert not skipped
                else:
                    best, weighed = affine_search.scan_coset_layer_by_tails(
                        *arguments, best_support
                    )
                assert (best, weighed) == (expected_best, expected_weighed)
                if expected_support is None:
                    assert np.all(best_support == -1)
                else:
                    assert tuple(best_support) == expected_support


@pytest.mark.parametrize("deficiency", [0, 1, 3])
def test_coset_tail_scans_with_tables_skip_only_completions_that_reach_the_skip_bound(
    deficiency: int,
) -> None:
    """A tail scan with tables skips completions only if their bound reaches the skip bound.

    It checks "Skip only words that a bound proves heavy enough" of
    ``brouwer_zimmermann_code.md#sec-requirements``.
    """
    # A fixed seed, since whether the tables skip anything depends on the rows.
    rng = np.random.default_rng(0)
    num_rows, num_columns, block_bits = 6, 24, 4
    # A one in the last column makes the tables cover every column.
    tails = (rng.random((num_rows, num_columns)) < 0.25).astype(np.uint8)
    tails[0, -1] = 1
    packed_tails = pack_bits(tails)
    width = packed_tails.shape[1]
    tables = completion_bounds.CompletionTables.build(packed_tails, block_bits)
    rank = num_rows - deficiency
    total_skipped = 0
    for _ in range(4):
        offset_tail = (rng.random(num_columns) < 0.25).astype(np.uint8)
        packed_offset = pack_bits(offset_tail[None])[0]
        for layer in range(2, num_rows + 1):
            weights = _get_tail_weights(offset_tail, tails, rank, layer)
            least = min(weight for _, weight in weights)
            for skip_bound in range(least - 1, least + 2):
                best_support = np.full(layer, -1, dtype=np.int64)
                best, weighed, skipped = affine_search.scan_coset_layer_by_tails_with_tables(
                    packed_offset,
                    packed_tails,
                    rank,
                    width,
                    np.empty(layer, dtype=np.int64),
                    np.empty((layer, width), dtype=np.uint64),
                    num_columns + num_rows + 1,
                    skip_bound - 1,
                    skip_bound,
                    tables.tables,
                    tables.offsets,
                    tables.block_bits,
                    best_support,
                )
                assert (best < skip_bound) == (least < skip_bound)
                if best <= num_columns + num_rows:
                    assert dict(weights)[tuple(best_support)] == best
                if skip_bound <= least:
                    bounds = [
                        sum(row < rank for row in rows[:-2])
                        + min(
                            sum(row < rank for row in pair)
                            for pair in itertools.combinations(range(rows[-2], num_rows), 2)
                        )
                        + _get_completion_bound(tails, rows, block_bits, offset_tail)
                        for rows, _ in weights
                    ]
                    assert skipped == sum(bound >= skip_bound for bound in bounds)
                    assert weighed + skipped == math.comb(num_rows, layer)
                    total_skipped += skipped
    assert total_skipped


def test_coset_tail_scans_below_the_best_skip_only_completions_that_cannot_lower_it() -> None:
    """A tail scan whose skip bound is its best weight skips no completion that can lower it."""
    # A fixed seed, since whether the tables skip anything depends on the rows.
    rng = np.random.default_rng(0)
    num_rows, num_columns, block_bits = 6, 24, 4
    tails = (rng.random((num_rows, num_columns)) < 0.25).astype(np.uint8)
    tails[0, -1] = 1
    packed_tails = pack_bits(tails)
    width = packed_tails.shape[1]
    tables = completion_bounds.CompletionTables.build(packed_tails, block_bits)
    table_arguments = (tables.tables, tables.offsets, tables.block_bits)
    total_skipped = 0
    for rank, _ in itertools.product((num_rows, num_rows - 3), range(4)):
        offset_tail = (rng.random(num_columns) < 0.25).astype(np.uint8)
        packed_offset = pack_bits(offset_tail[None])[0]
        for layer in range(2, num_rows + 1):
            weights = dict(_get_tail_weights(offset_tail, tails, rank, layer))
            least = min(weights.values())
            for best in (least - 1, least, num_columns + num_rows + 1):
                arguments = (
                    packed_offset,
                    packed_tails,
                    rank,
                    width,
                    np.empty(layer, dtype=np.int64),
                    np.empty((layer, width), dtype=np.uint64),
                )
                best_support = np.full(layer, -1, dtype=np.int64)
                found, weighed, skipped = affine_search.scan_coset_layer_by_tails_with_tables(
                    *arguments, best, -1, best, *table_arguments, best_support
                )
                assert weighed + skipped == math.comb(num_rows, layer)
                if best > least:
                    assert found == least == weights[tuple(best_support)]
                else:
                    assert found == best
                    _, *expected = affine_search.scan_coset_layer_by_tails_with_tables(
                        *arguments, best, best - 1, best, *table_arguments, best_support
                    )
                    assert [weighed, skipped] == expected
                total_skipped += skipped
    assert total_skipped


def _get_tail_weights(
    offset_tail: npt.NDArray[np.uint8], tails: npt.NDArray[np.uint8], rank: int, layer: int
) -> list[tuple[tuple[int, ...], int]]:
    """Each support of a layer of a coset basis by tails, in order, with the weight of its word.

    The word of a support has a one on the pivot of each of its rows below ``rank``, and its tail on
    the other columns.
    """
    return [
        (
            rows,
            sum(row < rank for row in rows)
            + int(np.sum(offset_tail ^ tails[list(rows)].sum(0) % 2)),
        )
        for rows in itertools.combinations(range(len(tails)), layer)
    ]


def _get_completion_bound(
    tails: npt.NDArray[np.uint8],
    support: Sequence[int],
    block_bits: int,
    offset: npt.NDArray[np.uint8] | None = None,
) -> int:
    """The block bound on the completions of a support by pairs from its next-to-last row on.

    It sums, over blocks of columns, the least weight on the block of the offset, if any, plus the
    support's rows before that row plus such a pair.
    """
    prefix = np.bitwise_xor.reduce(tails[list(support[:-2])], axis=0, initial=0)
    if offset is not None:
        prefix ^= offset
    pairs = list(itertools.combinations(range(support[-2], len(tails)), 2))
    return sum(
        min(
            int(np.sum((prefix ^ tails[i] ^ tails[j])[start : start + block_bits]))
            for i, j in pairs
        )
        for start in range(0, tails.shape[1], block_bits)
    )


def _scan_layer(
    generators: npt.NDArray[np.uint64],
    labels: npt.NDArray[np.uint64],
    rank: int,
    width: int,
    weight: int,
    batch_size: int,
    table_arguments: tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64], int],
    best: int,
) -> tuple[int, npt.NDArray[np.int64]]:
    """Scan a layer of a set with ``scan_set``, and return its best weight and its total counts."""
    support = np.zeros(weight, dtype=np.int64)
    prefixes = np.zeros((weight + 1, generators.shape[1]), dtype=np.uint64)
    depth, layer_counts = 0, np.zeros(3, dtype=np.int64)
    while depth >= 0:
        best, depth, *counts = affine_search.scan_set(
            generators,
            labels,
            rank,
            width,
            support,
            prefixes,
            depth,
            best,
            -1,
            batch_size,
            *table_arguments,
        )
        layer_counts += counts
    return best, layer_counts


def _scan_items_on_threads(
    generators: npt.NDArray[np.uint64],
    labels: npt.NDArray[np.uint64],
    rank: int,
    width: int,
    weight: int,
    batch_size: int,
    table_arguments: tuple[npt.NDArray[np.uint8], npt.NDArray[np.int64], int],
    best: int,
) -> npt.NDArray[np.int64]:
    """Scan the items of a layer that fix one row, on three threads, all from one best weight.

    Returns the total counts of the scans.
    """
    num_rows = generators.shape[0]
    items = [np.array([row], dtype=np.int64) for row in range(num_rows - weight + 1)]
    lock = threading.Lock()
    total_counts = np.zeros(3, dtype=np.int64)

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        support = np.zeros(weight, dtype=np.int64)
        prefixes = np.zeros((weight + 1, generators.shape[1]), dtype=np.uint64)

        def scan_range(start: int, end: int) -> None:
            depth = affine_search.start_set_items(generators, items[start], support, prefixes)
            while depth >= 0:
                _, depth, *counts = affine_search.scan_set_items(
                    generators,
                    labels,
                    rank,
                    width,
                    support,
                    prefixes,
                    depth,
                    best,
                    -1,
                    batch_size,
                    *table_arguments,
                    items[end - 1],
                )
                with lock:
                    total_counts[:] += counts

        return scan_range

    _threads.run_work_items(len(items), 3, start_worker)
    return total_counts
