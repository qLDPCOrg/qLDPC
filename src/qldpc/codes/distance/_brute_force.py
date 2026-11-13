# SPDX-License-Identifier: Apache-2.0

"""Exact packed enumeration of independent logical and excluded binary bases."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from qldpc._binary import count_trailing_zeros, pack_bits

from ._binary import get_eligible_rows

# The low bit of each aligned pair of bits of a riffled word, one pair for each qubit.
_MASK55 = np.uint64(0x5555555555555555)

####################################################################################################
# exhaustive enumeration


def get_nested_distance_brute_force(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    *,
    cutoff: int,
    batch_exponent: int,
    homogeneous: bool,
) -> int:
    """Find the minimum weight of an eligible word of a nested-code basis by exhaustive enumeration.

    The rows of ``basis`` are independent, and ``labels`` are its quotient labels (see
    ``get_nested_code_basis``).  The rows marked by ``get_eligible_rows`` are the logical rows of
    ``_get_distance_brute_force``, and the others are its stabilizers.  The other arguments and the
    result are those of ``_get_distance_brute_force``.
    """
    eligible = get_eligible_rows(labels)
    return _get_distance_brute_force(
        basis[eligible],
        basis[~eligible],
        cutoff=cutoff,
        batch_exponent=batch_exponent,
        homogeneous=homogeneous,
    )


def _get_distance_brute_force(
    logical_ops: npt.ArrayLike,
    stabilizers: npt.ArrayLike,
    *,
    cutoff: int,
    batch_exponent: int,
    homogeneous: bool,
) -> int:
    """Find the minimum weight of a nontrivial logical operator by exhaustive enumeration.

    Every combination of the rows of logical_ops that uses at least one row, multiplied by every
    combination of the rows of stabilizers, is a candidate.  The rows of logical_ops and stabilizers
    must together be linearly independent; callers in this package construct such bases.

    Args:
        logical_ops: Binary rows that, together with the stabilizers, generate the logical space.
        stabilizers: Binary rows that generate the excluded (stabilizer) space.
        cutoff: Return early once the lightest operator seen has weight at most this cutoff.
        batch_exponent: Calculations are vectorized over at most ``2**batch_exponent`` words.
        homogeneous: If True, rows are weighed by Hamming weight.  If False, rows are symplectic
            vectors, with X support in their first half and Z support in their second half, and
            are weighed by symplectic weight.

    Returns:
        The minimum weight of a nontrivial logical operator or, after an early exit, the weight (at
        most cutoff) of the lightest operator seen.  Without logical_ops, the result is one more
        than the largest possible weight: the number of columns if homogeneous, and otherwise the
        number of qubits.
    """
    num_bits = np.shape(logical_ops)[-1]

    weight_func = np.bitwise_count if homogeneous else symplectic_weight
    if not homogeneous:
        logical_ops = riffle(logical_ops)
        stabilizers = riffle(stabilizers)

    int_logical_ops = pack_bits(np.asarray(logical_ops, dtype=np.uint8))
    int_stabilizers = pack_bits(np.asarray(stabilizers, dtype=np.uint8))
    num_stabilizers = len(int_stabilizers)

    # Choose how many generators to vectorize over, so that most calculations act on at most
    # ``2**batch_exponent`` words.  Rows of more packed words allow fewer generators, and never
    # fewer than zero.
    num_vectorized_ops = max(
        0,
        min(
            batch_exponent + 1 - int_logical_ops.shape[-1],
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

    # np.bitwise_count returns uint8, whose row sums would overflow at 256 or more columns.  The
    # loop below therefore writes the weights into this uint64 buffer.
    out = np.empty_like(arrayf)

    # Until an operator is weighed, the lightest weight seen is one more than any operator can have,
    # and an early exit must not return it.
    min_weight = (num_bits if homogeneous else num_bits // 2) + 1

    # Min weight of the block containing logical ops, if any, with weights cast to uint64 as above.
    logical_block = arrayf[2**num_stabilizers :]
    if len(logical_block):
        weights = np.asarray(weight_func(logical_block), dtype=np.uint64)
        min_weight = inplace_rowsum(weights).min()
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


####################################################################################################
# weights of riffled words, and array helpers


def symplectic_weight(
    words: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64] | None = None
) -> npt.NDArray[np.uint64]:
    """Count the qubits in the support of each riffled word (see ``riffle``).

    Each qubit occupies one aligned bit pair ``(2m, 2m + 1)`` of a word, so a qubit is in the
    support if either bit of its pair is set.  The result is uint64, and ``out`` doubles as scratch
    space.
    """
    out = np.right_shift(words, 1, out=out)
    out |= words
    out &= _MASK55
    return np.bitwise_count(out, out=out)


def inplace_rowsum(arr: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Destructively compute ``arr.sum(-1)``, placing the result in the first column of ``arr``.

    When complete, the returned sum will be stored in ``arr[..., 0]``, while other entries in
    ``arr[..., 1:]`` will be left in indeterminate states.  This permits a faster sum
    implementation.
    """
    width = arr.shape[-1]
    while width > 1:
        split = width // 2
        arr[..., :split] += arr[..., width - split : width]
        width -= split

    return arr[..., 0]


def riffle(array: npt.ArrayLike) -> npt.ArrayLike:
    """'Riffle' Pauli strings, putting X and Z support bits for each qubit next to each other."""
    arr = np.asarray(array)
    num_bits = arr.shape[-1]
    assert num_bits % 2 == 0
    if arr.size == 0:
        return np.empty((0, num_bits), dtype=arr.dtype)  # nothing to riffle (e.g. no stabilizers)
    return np.reshape(arr, (-1, 2, num_bits // 2)).transpose(0, 2, 1).reshape(-1, num_bits)
