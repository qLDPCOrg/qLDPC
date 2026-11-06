# SPDX-License-Identifier: Apache-2.0

"""Unit tests for logical-operator counts by the MacWilliams method."""

from __future__ import annotations

import math

import numpy as np
import pytest

from . import macwilliams as mw


@pytest.mark.parametrize("compiled", [True, False])
@pytest.mark.parametrize("num_bits", [8, 70])
def test_logical_counts(monkeypatch: pytest.MonkeyPatch, num_bits: int, compiled: bool) -> None:
    """Count two disjoint four-qubit blocks, with any remaining qubits free, at every weight.

    A logical has even weight on each block, and the checks are the words of weights 0, 4 and 8 that
    are constant on each block.  At length 70, middle weights exceed 64-bit integers.
    """
    if not compiled:
        monkeypatch.setattr(mw, "_add_span_weights", mw._add_span_weights.py_func)
        monkeypatch.setattr(mw, "_transform_weights", mw._transform_weights.py_func)
    monkeypatch.setattr(mw, "_CHUNK_SIZE", 1)
    checks = np.zeros((2, num_bits), dtype=np.uint8)
    checks[0, :4] = checks[1, -4:] = 1
    enumerators = mw.CheckEnumerators(checks, checks if num_bits == 8 else checks.copy())
    for weight in range(num_bits + 1):
        kernel = sum(
            math.comb(4, block_0) * math.comb(4, block_1) * math.comb(num_bits - 8, rest)
            for block_0 in (0, 2, 4)
            for block_1 in (0, 2, 4)
            if (rest := weight - block_0 - block_1) >= 0
        )
        expected = kernel - {0: 1, 4: 2, 8: 1}.get(weight, 0)
        assert enumerators.count_logical_ops(weight) == (expected, expected)


def test_check_rank_limit() -> None:
    """CheckEnumerators rejects a check space with more than 2**63 words."""
    checks = np.eye(mw._MAX_RANK + 1, mw._MAX_RANK + 2, dtype=np.uint8)
    with pytest.raises(ValueError, match="check ranks up to 63"):
        mw.CheckEnumerators(checks, checks[:0])
