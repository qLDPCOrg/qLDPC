# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the step from one support of a layer to the next."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from . import support_stepping


def test_layers_visit_each_support_once_with_its_sums(pytestconfig: pytest.Config) -> None:
    """A layer visits each combination of rows in order, with its sum on the first machine words."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    width = 2
    for dimension in range(1, 7):
        rows = rng.integers(2**64, size=(dimension, 3), dtype=np.uint64)
        start = rng.integers(2**64, size=3, dtype=np.uint64)
        for weight in range(1, dimension + 1):
            support = np.zeros(weight, dtype=np.int64)
            prefixes = np.zeros((weight, 3), dtype=np.uint64)
            prefixes[0] = start
            visited = []
            depth = support_stepping.advance_support(rows, width, support, prefixes, 0)
            while depth >= 0:
                sums = prefixes[depth] ^ rows[support[depth]]
                visited.append((support.tolist(), sums[:width].tolist()))
                support[depth] += 1
                depth = support_stepping.advance_support(rows, width, support, prefixes, depth)
            expected = []
            for indices in itertools.combinations(range(dimension), weight):
                sums = start ^ np.bitwise_xor.reduce(rows[list(indices)], axis=0)
                expected.append((list(indices), sums[:width].tolist()))
            assert visited == expected


def test_layers_resume_from_any_state_and_can_skip_the_other_last_rows(
    pytestconfig: pytest.Config,
) -> None:
    """Layers resume from any state, and skip the other last rows of a support on request.

    From the rows ``start[:depth]`` and the next row ``start[depth]``, a layer visits the supports
    whose first ``depth + 1`` rows are not before those of ``start`` in lexicographic order.
    Raising the last row of a support past the last row of the basis skips the supports that differ
    from it only in their last row.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for dimension in range(1, 7):
        rows = rng.integers(2**64, size=(dimension, 1), dtype=np.uint64)
        for weight in range(1, dimension + 1):
            supports = list(itertools.combinations(range(dimension), weight))
            start = supports[int(rng.integers(len(supports)))]
            first_depth = int(rng.integers(weight))
            skips = {indices for indices in supports if rng.random() < 0.3}

            # Resume from the state of ``start`` at ``first_depth``, and at each support in
            # ``skips``, raise its last row to the number of rows or one more.
            support = np.array(start, dtype=np.int64)
            prefixes = np.zeros((weight, 1), dtype=np.uint64)
            for level in range(first_depth):
                prefixes[level + 1] = prefixes[level] ^ rows[start[level]]
            visited = []
            depth = support_stepping.advance_support(rows, 1, support, prefixes, first_depth)
            while depth >= 0:
                indices = tuple(support.tolist())
                visited.append((indices, int(prefixes[depth, 0] ^ rows[support[depth], 0])))
                if indices in skips:
                    support[depth] = dimension + int(rng.integers(2))
                else:
                    support[depth] += 1
                depth = support_stepping.advance_support(rows, 1, support, prefixes, depth)

            expected = []
            skipped_head: tuple[int, ...] | None = None
            for indices in supports:
                if indices[: first_depth + 1] < start[: first_depth + 1]:
                    continue
                if indices[:-1] == skipped_head:
                    continue
                expected.append((indices, int(np.bitwise_xor.reduce(rows[list(indices), 0]))))
                if indices in skips:
                    skipped_head = indices[:-1]
            assert visited == expected
