# SPDX-License-Identifier: Apache-2.0

"""Validation of permutations that act separately on consecutive blocks of points.

A symmetry of a chain complex permutes the basis of each of its modules, and a symmetry of a
classical code permutes its checks and its bits.  Such a symmetry either has one permutation for
each block of points, or one permutation of all points that maps each block to itself.  Members of
the automorphism group of a parity check matrix take the second form, with bits following checks.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence

import numpy as np
import sympy.combinatorics as comb


def get_block_permutations(
    perm: comb.Permutation | Sequence[Iterable[int]], sizes: Sequence[int], name: str
) -> list[list[int]]:
    """Validate a permutation of consecutive blocks of points, and split it into its blocks.

    The blocks have the given sizes, and the permutation is either a sequence with one permutation
    of ``range(size)`` for each block, or one SymPy permutation of ``range(sum(sizes))`` that maps
    each block to itself.  Each permutation is an integer sequence that lists the image of every
    point, or a SymPy permutation, which fixes the points beyond its own size.  The returned lists
    hold, for each block, the image of every point of that block, counted from the start of the
    block.  Error messages identify the permutation by the given name.
    """
    if isinstance(perm, comb.Permutation):
        images = _get_permutation(perm, sum(sizes), name)
        bounds = list(itertools.accumulate(sizes, initial=0))
        perms = [
            [image - start for image in images[start:stop]]
            for start, stop in itertools.pairwise(bounds)
        ]
        if any(sorted(block) != list(range(size)) for block, size in zip(perms, sizes)):
            raise ValueError(f"{name} must map each block of points, of sizes {sizes}, to itself")
        return perms
    if len(perm) != len(sizes):
        raise ValueError(f"{name} must have {len(sizes)} permutations")
    return [
        _get_permutation(block, size, f"{name}[{index}]")
        for index, (block, size) in enumerate(zip(perm, sizes))
    ]


def _get_permutation(perm: Iterable[int], size: int, name: str) -> list[int]:
    """Validate a permutation of range(size), given as an integer sequence or SymPy permutation.

    An integer sequence lists the image of every point, so that it must have length ``size``.  A
    SymPy permutation fixes the points beyond its own size, which must not exceed ``size``.
    """
    if isinstance(perm, comb.Permutation):
        if perm.size > size:
            raise ValueError(f"{name} permutes {perm.size} points, but should permute {size}")
        return perm.array_form + list(range(perm.size, size))
    images = np.asarray(list(perm))
    if isinstance(perm, str) or (images.size and images.dtype.kind not in "iu"):
        raise TypeError(f"{name} must be a sequence of integers or a SymPy permutation")
    if images.shape != (size,) or not np.array_equal(np.sort(images), np.arange(size)):
        raise ValueError(f"{name} must be a permutation of range({size})")
    return [int(image) for image in images]
