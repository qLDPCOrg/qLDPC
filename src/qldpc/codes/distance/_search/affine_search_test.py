# SPDX-License-Identifier: Apache-2.0

"""Tests for the compiled scans of layers that distance and coset searches share."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import pack_bits

from . import affine_search, completion_bounds


@pytest.mark.usefixtures("compilation_mode")
def test_family_scan() -> None:
    """Chunked scans of a family enumerate each combination of basis rows once.

    The chunked scans preserve resumption state and multiword labels.
    """
    rng = np.random.default_rng(120)

    # The scan input uses full-rank, partial-rank, and rank-zero information sets.  The last two
    # information sets store labels only in their second packed word.
    for dimension in range(1, 7):
        generators = rng.integers(0, 2**64, (3, dimension, 5), dtype=np.uint64)
        labels = np.zeros((3, dimension, 2), dtype=np.uint64)
        labels[1:, :, 1] = rng.integers(0, 2, (2, dimension), dtype=np.uint64)
        ranks = np.array([dimension, dimension // 2, 0], dtype=np.int64)
        widths = np.array([5, 2, 1], dtype=np.int64)
        label_widths = np.array([0, 2, 2], dtype=np.int64)
        for weight in range(1, dimension + 1):
            support = np.zeros(weight, dtype=np.int64)
            prefixes = np.zeros((weight + 1, 3, 5), dtype=np.uint64)
            depth, best, expected = 0, 1000, 1000
            for indices in itertools.combinations(range(dimension), weight):
                # Update the expected running minimum from the current combination of basis rows.
                for index in range(3):
                    words = np.bitwise_xor.reduce(generators[index, indices, : widths[index]])
                    eligible = not label_widths[index] or np.any(
                        np.bitwise_xor.reduce(labels[index, indices], axis=0)
                    )
                    candidate = sum(int(word).bit_count() for word in words)
                    candidate += sum(row < ranks[index] for row in indices)
                    if eligible:
                        expected = min(expected, candidate)

                # Scan one basis-row combination and compare the running minimum and kernel state.
                best, depth, visited = affine_search.scan_family(
                    generators,
                    labels,
                    ranks,
                    widths,
                    label_widths,
                    support,
                    prefixes,
                    depth,
                    best,
                    0,
                    1,
                )
                assert visited == 1 and best == expected
                # The last selected row index has advanced in preparation for the next chunk.
                np.testing.assert_array_equal(support[:-1], indices[:-1])
                assert support[-1] == indices[-1] + 1

            # One more kernel call reports completion after all basis-row combinations are visited.
            best, depth, visited = affine_search.scan_family(
                generators,
                labels,
                ranks,
                widths,
                label_widths,
                support,
                prefixes,
                depth,
                best,
                0,
                1,
            )
            assert (best, depth, visited) == (expected, -1, 0)


@pytest.mark.usefixtures("compilation_mode")
def test_family_scans_weigh_every_set_of_the_support_that_reaches_the_cutoff() -> None:
    """A scan returns at the cutoff only after it weighs the support in every set.

    Two information sets of two rows each scan layer one.  At the first support, the word of the
    first set weighs one plus two, which reaches the cutoff three, and the word of the second set
    weighs one plus zero, so the scan returns one after one support.
    """
    generators = np.array([[[0b11], [0b111]], [[0], [0b1]]], dtype=np.uint64)
    best, _, visited = affine_search.scan_family(
        generators,
        np.zeros((2, 2, 0), dtype=np.uint64),
        np.full(2, 2, dtype=np.int64),
        np.ones(2, dtype=np.int64),
        np.zeros(2, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        np.zeros((2, 2, 1), dtype=np.uint64),
        0,
        10,
        3,
        100,
    )
    assert (best, visited) == (1, 1)


@pytest.mark.usefixtures("compilation_mode")
def test_scan_of_one_information_set(pytestconfig: pytest.Config) -> None:
    """A scan of one information set finds the lightest eligible word of a layer, in chunks.

    With or without tables, a layer that runs to completion weighs or skips every support exactly
    once, and each chunk weighs supports and looks up bounds at most ``batch_size`` times together.
    While the best weight stays the same, a scan with tables skips exactly the supports whose block
    bound reaches its limit.  A scan returns at the first support, in order, that brings the best
    weight to the cutoff.  The sets have multiword labels or no labels, which make every word
    eligible.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    no_tables = completion_bounds.NO_TABLES
    for dimension, num_label_words in itertools.product(range(2, 7), (0, 2)):
        tails = (rng.random((dimension, 70)) < 0.3).astype(np.uint8)
        generators = pack_bits(tails)[None]
        labels = np.zeros((1, dimension, num_label_words), dtype=np.uint64)
        if num_label_words:
            labels[0, :, 1] = rng.integers(0, 4, dimension, dtype=np.uint64)
        widths = np.array([generators.shape[2]], dtype=np.int64)
        label_widths = np.array([num_label_words], dtype=np.int64)
        all_tables = [no_tables] + [
            completion_bounds.CompletionTables.build(generators[0], bits) for bits in (8, 16)
        ]
        for weight in range(2, dimension + 1):
            # The weight of each support in order, or None if its word is ineligible.
            weights = [
                weight + int(np.sum(np.bitwise_xor.reduce(tails[list(rows)], axis=0)))
                if not num_label_words or np.any(np.bitwise_xor.reduce(labels[0, list(rows)]))
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
                    prefixes = np.zeros((weight + 1, 1, widths[0]), dtype=np.uint64)
                    depth, best, evaluated, skipped = 0, initial_best, 0, 0
                    while depth >= 0 and best > cutoff:
                        best, depth, num_evaluated, num_lookups, num_skipped = (
                            affine_search.scan_information_set(
                                generators,
                                labels,
                                widths,
                                label_widths,
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
                        # Without tables, the scan weighs every support up to the first that
                        # reaches the cutoff, and no more.
                        running = itertools.accumulate(
                            (value if value is not None else initial_best for value in weights),
                            min,
                            initial=initial_best,
                        )
                        assert evaluated == next(
                            index for index, value in enumerate(running) if value <= cutoff
                        )


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("css_weight", [False, True])
def test_coset_layer_scans_keep_the_first_lightest_word(
    css_weight: bool, pytestconfig: pytest.Config
) -> None:
    """A scan of a layer of a coset weighs the offset plus each sum of rows of the layer, in order.

    It keeps each word lighter than every word before it and than the initial best weight, and
    returns after the first word at which the best weight so far is at most the cutoff, having
    weighed exactly the supports up to it.  The words span two machine words, and the X parts of
    the CSS weight take bits of both.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    num_rows, num_bits = 6, 100
    row_bits = (rng.random((num_rows, num_bits)) < 0.3).astype(np.uint8)
    offset_bits = rng.integers(0, 2, num_bits, dtype=np.uint8)
    x_bits = rng.integers(0, 2, num_bits, dtype=np.uint8)
    rows = pack_bits(row_bits)[None]
    offset, x_mask = pack_bits(np.stack([offset_bits, x_bits]))
    width = rows.shape[2]
    widths = np.array([width], dtype=np.int64)

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
            prefixes = np.empty((layer, 1, width), dtype=np.uint64)
            if css_weight:
                best, weighed = affine_search.scan_coset_layer_by_css_weight(
                    offset,
                    rows,
                    widths,
                    support,
                    prefixes,
                    initial_best,
                    cutoff,
                    x_mask,
                    representative,
                )
            else:
                best, weighed = affine_search.scan_coset_layer(
                    offset, rows, widths, support, prefixes, initial_best, cutoff, representative
                )
            assert (best, weighed) == (expected_best, expected_weighed)
            if expected_word is None:
                assert np.all(representative == 7)
            else:
                np.testing.assert_array_equal(representative, expected_word)


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("css_weight", [False, True])
def test_coset_layer_scans_keep_the_first_of_equally_light_words(css_weight: bool) -> None:
    """A scan keeps the first of the equally light words ``001``, ``010``, and ``100`` of layer one.

    The X mask selects every bit, so the CSS weight of each word is its Hamming weight.
    """
    rows = np.array([[[0b001], [0b010], [0b100]]], dtype=np.uint64)
    arguments = (
        np.zeros(1, dtype=np.uint64),
        rows,
        np.ones(1, dtype=np.int64),
        np.empty(1, dtype=np.int64),
        np.empty((1, 1, 1), dtype=np.uint64),
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


@pytest.mark.usefixtures("compilation_mode")
def test_coset_layer_scans_reach_their_last_support() -> None:
    """A layer visits every support, including the last one.

    The rows are unit words and the offset is the sum of the last ``layer`` rows, so that only the
    last support of the layer gives the zero word.
    """
    for num_rows in range(1, 7):
        rows = (np.uint64(1) << np.arange(num_rows, dtype=np.uint64)).reshape(1, num_rows, 1)
        for layer in range(1, num_rows + 1):
            offset = np.bitwise_xor.reduce(rows[0, num_rows - layer :])
            representative = np.ones(1, dtype=np.uint64)
            best, weighed = affine_search.scan_coset_layer(
                offset,
                rows,
                np.ones(1, dtype=np.int64),
                np.empty(layer, dtype=np.int64),
                np.empty((layer, 1, 1), dtype=np.uint64),
                num_rows + 1,
                0,
                representative,
            )
            assert (best, weighed) == (0, math.comb(num_rows, layer))
            assert representative[0] == 0


def _get_completion_bound(
    tails: npt.NDArray[np.uint8], support: Sequence[int], block_bits: int
) -> int:
    """The block bound on the completions of a support from its first row of its last pair on.

    Add, over blocks of columns, the least weight on the block of the first rows of the support plus
    any pair of rows from the first row of its last pair on.
    """
    prefix = np.bitwise_xor.reduce(tails[list(support[:-2])], axis=0, initial=0)
    pairs = list(itertools.combinations(range(support[-2], len(tails)), 2))
    return sum(
        min(
            int(np.sum((prefix ^ tails[i] ^ tails[j])[start : start + block_bits]))
            for i, j in pairs
        )
        for start in range(0, tails.shape[1], block_bits)
    )
