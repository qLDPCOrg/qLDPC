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
