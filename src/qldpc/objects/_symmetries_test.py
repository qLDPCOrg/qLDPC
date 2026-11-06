# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _symmetries.py."""

from __future__ import annotations

import pytest
import sympy.combinatorics as comb

from qldpc import abstract
from qldpc.objects import _symmetries


def test_get_block_permutations() -> None:
    """Permutations of consecutive blocks, given block by block or as one permutation."""
    sizes = (2, 3)
    perms = [[1, 0], [2, 0, 1]]
    assert _symmetries.get_block_permutations(perms, sizes, "perm") == perms
    member = abstract.GroupMember(perms[0]) @ abstract.GroupMember(perms[1])
    assert _symmetries.get_block_permutations(member, sizes, "perm") == perms

    # SymPy permutations fix the points beyond their own size
    assert _symmetries.get_block_permutations(comb.Permutation(0, 1), sizes, "perm") == [
        [1, 0],
        [0, 1, 2],
    ]
    assert _symmetries.get_block_permutations([comb.Permutation(1), [0, 2, 1]], sizes, "p") == [
        [0, 1],
        [0, 2, 1],
    ]

    with pytest.raises(ValueError, match="must map each block"):
        _symmetries.get_block_permutations(comb.Permutation(1, 2), sizes, "perm")
    with pytest.raises(ValueError, match="must have 2 permutations"):
        _symmetries.get_block_permutations([[1, 0]], sizes, "perm")

    # each permutation is an integer sequence that lists every image, or a SymPy permutation that
    # fits, and errors identify the permutation of the offending block
    for perm, error in [
        ("10", TypeError),
        ([1.0, 0.0], TypeError),
        ([1, 1], ValueError),
        ([1, 0, 2], ValueError),
        (comb.Permutation(2), ValueError),
    ]:
        with pytest.raises(error, match=r"perm\[0\]"):
            _symmetries.get_block_permutations([perm, [0, 1, 2]], sizes, "perm")
    with pytest.raises(ValueError, match="permutes 6 points"):
        _symmetries.get_block_permutations(comb.Permutation(5), sizes, "perm")
