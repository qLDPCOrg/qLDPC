# SPDX-License-Identifier: Apache-2.0

"""Exact packed enumeration of independent logical and excluded binary bases."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ._binary import (
    count_trailing_zeros,
    inplace_rowsum,
    pack_bits,
    riffle,
    symplectic_weight,
)

####################################################################################################
# exhaustive enumeration


def get_distance_quantum_brute_force(
    logical_ops: npt.ArrayLike,
    stabilizers: npt.ArrayLike,
    *,
    cutoff: int,
    block_size: int,
    homogeneous: bool,
) -> int:
    """Find the minimum weight of a nontrivial logical operator by exhaustive enumeration.

    Every combination of the rows of logical_ops that uses at least one row, multiplied by every
    combination of the rows of stabilizers, is a candidate.  The rows of logical_ops and stabilizers
    must together be linearly independent; callers in this package construct or verify such bases.

    Args:
        logical_ops: Binary rows that, together with the stabilizers, generate the logical space.
        stabilizers: Binary rows that generate the excluded (stabilizer) space.
        cutoff: Return early once the lightest operator seen has weight at most this cutoff.
        block_size: Batch size exponent: calculations are vectorized over ``2**block_size`` values.
        homogeneous: If True, rows are weighed by Hamming weight.  If False, rows are symplectic
            vectors, with X support in their first half and Z support in their second half, and
            are weighed by symplectic weight.

    Returns:
        The minimum weight of a nontrivial logical operator or, after an early exit, the weight (at
        most cutoff) of the lightest operator seen.
    """
    num_bits = np.shape(logical_ops)[-1]

    weight_func = np.bitwise_count if homogeneous else symplectic_weight
    if not homogeneous:
        logical_ops = riffle(logical_ops)
        stabilizers = riffle(stabilizers)

    int_logical_ops = pack_bits(np.asarray(logical_ops, dtype=np.uint8))
    int_stabilizers = pack_bits(np.asarray(stabilizers, dtype=np.uint8))
    num_stabilizers = len(int_stabilizers)

    # Number of generators to include in the operational array.  Most calculations will then be
    # vectorized over ``2**block_size`` values.  Clamp at 0: when the packed word-count per row
    # exceeds ``block_size + 1`` the first term goes negative, which would make a negative slice and
    # build an uncapped ``2**(S-k)`` array, defeating the block_size cap.
    num_vectorized_ops = max(
        0,
        min(
            block_size + 1 - int_logical_ops.shape[-1],
            len(int_logical_ops) + len(int_stabilizers),
        ),
    )

    # Vectorize all combinations of first `num_vectorized_ops` stabilizers
    array = np.zeros((1, int_logical_ops.shape[-1]), dtype=np.uint64)
    for op in int_stabilizers[:num_vectorized_ops]:
        array = np.vstack([array, array ^ op])

    if num_vectorized_ops > num_stabilizers:
        # fill out block with products of some logical ops
        for op in int_logical_ops[: num_vectorized_ops - num_stabilizers]:
            array = np.vstack([array, array ^ op])

        int_logical_ops = int_logical_ops[num_vectorized_ops - num_stabilizers :]

    int_stabilizers = int_stabilizers[num_vectorized_ops:]

    # Everything below will run much faster if we use Fortran-style ordering
    arrayf = np.asarray(array, order="F", dtype=np.uint64)

    # out is the uint64 buffer for the in-loop weight_func calls: passing out=out keeps
    # np.bitwise_count's uint8 result cast to uint64, avoiding the weight-reduction overflow for
    # codes with >= 256 columns (the pre-loop call below casts explicitly for the same reason).
    out = np.empty_like(arrayf)

    # Min weight of the block containing logical ops.  Cast to uint64 because np.bitwise_count
    # returns uint8 without ``out=``, which would overflow the reduction below for >= 256 columns.
    weights = np.asarray(weight_func(arrayf[2**num_stabilizers :]), dtype=np.uint64)
    min_weight = inplace_rowsum(weights).min(initial=num_bits)
    if min_weight <= cutoff:
        return int(min_weight)

    # Sweep over every remaining logical-op combination and, nested, every remaining stabilizer
    # combination, on top of the vectorized block above.  The sweep uses a reflected Gray code:
    # successive Gray-code words differ in one bit, whose position is the trailing-zero count of the
    # step index.  Each step therefore flips a single operator into/out of the running XOR
    # (``arrayf``) with one ^=, so the loop walks the 2**k - 1 nonzero combinations of the remaining
    # operators (the empty combination is the pre-loop state) without rebuilding any.
    # ``min_weight`` tracks the lightest operator seen; ``cutoff`` lets the sweep stop once that
    # bound is reached (see Returns).
    for li in range(1, 2 ** len(int_logical_ops)):
        arrayf ^= int_logical_ops[count_trailing_zeros(li)]
        weights = weight_func(arrayf, out=out)
        min_weight = inplace_rowsum(weights).min(initial=min_weight)
        if min_weight <= cutoff:
            return int(min_weight)

        for si in range(1, 2 ** len(int_stabilizers)):
            arrayf ^= int_stabilizers[count_trailing_zeros(si)]
            weights = weight_func(arrayf, out=out)
            min_weight = inplace_rowsum(weights).min(initial=min_weight)
            if min_weight <= cutoff:
                return int(min_weight)

    return int(min_weight)
