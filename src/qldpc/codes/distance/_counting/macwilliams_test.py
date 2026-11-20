# SPDX-License-Identifier: Apache-2.0

"""Unit tests for logical-operator counts by the MacWilliams method."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from .._tuning import DEFAULT_TUNING
from . import macwilliams as mw


@pytest.mark.parametrize("num_bits", [8, 70])
def test_logical_counts(monkeypatch: pytest.MonkeyPatch, num_bits: int) -> None:
    """Counts match a direct count at every weight, also where they exceed 64-bit integers."""
    add_span_weights = mw._add_span_weights
    chunk_sizes = []

    def record_chunk(*args: Any) -> None:
        chunk_sizes.append(int(args[3] - args[2]))
        add_span_weights(*args)

    monkeypatch.setattr(mw, "_add_span_weights", record_chunk)
    # Two four-qubit blocks, and any other qubits free: the checks span the words of weights 0, 4,
    # and 8 that are constant on each block, and a logical has even weight on each block.
    checks = np.zeros((2, num_bits), dtype=np.uint8)
    checks[0, :4] = checks[1, -4:] = 1
    enumerators = mw.CheckEnumerators(checks, checks if num_bits == 8 else checks.copy(), 1)
    for weight in range(num_bits + 1):
        kernel = sum(
            math.comb(4, block_0) * math.comb(4, block_1) * math.comb(num_bits - 8, rest)
            for block_0 in (0, 2, 4)
            for block_1 in (0, 2, 4)
            if (rest := weight - block_0 - block_1) >= 0
        )
        expected = kernel - {0: 1, 4: 2, 8: 1}.get(weight, 0)
        assert enumerators.count_logical_ops(weight) == (expected, expected)
    # With a chunk size of one, each call of the walk takes one step.
    assert chunk_sizes and set(chunk_sizes) == {1}


def test_logical_counts_of_unequal_check_spaces() -> None:
    """Each sector counts the kernel of the checks of the other type, less the span of its own."""
    # The X check is all ones, and the Z checks pair qubits 0 to 5, so ker(H_Z) holds the words that
    # repeat each pair, and ker(H_X) the words of even weight.  The excluded words are 0 and all
    # ones for X, and the unions of pairs for Z.
    num_bits, num_pairs = 8, 3
    checks_x = np.ones((1, num_bits), dtype=np.uint8)
    checks_z = np.zeros((num_pairs, num_bits), dtype=np.uint8)
    for pair in range(num_pairs):
        checks_z[pair, 2 * pair : 2 * pair + 2] = 1
    enumerators = mw.CheckEnumerators(checks_x, checks_z, DEFAULT_TUNING.macwilliams_chunk_size)
    for weight in range(num_bits + 1):
        count_x = sum(
            math.comb(num_pairs, pairs) * math.comb(num_bits - 2 * num_pairs, weight - 2 * pairs)
            for pairs in range(min(num_pairs, weight // 2) + 1)
        ) - (weight in (0, num_bits))
        count_z = (
            math.comb(num_bits, weight) - math.comb(num_pairs, weight // 2)
            if weight % 2 == 0
            else 0
        )
        assert enumerators.count_logical_ops(weight) == (count_x, count_z)


def test_check_rank_limit() -> None:
    """CheckEnumerators rejects a check space with more than 2**63 words."""
    checks = np.eye(mw._MAX_RANK + 1, mw._MAX_RANK + 2, dtype=np.uint8)
    with pytest.raises(ValueError, match="check ranks up to 63"):
        mw.CheckEnumerators(checks, checks[:0], DEFAULT_TUNING.macwilliams_chunk_size)
