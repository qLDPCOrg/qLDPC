# SPDX-License-Identifier: Apache-2.0

"""Exact distances by exhaustive enumeration of the combinations of packed binary rows."""

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
    symplectic: bool,
) -> int:
    """Find the minimum weight of an eligible word of a nested-code basis by exhaustive enumeration.

    The basis and its labels are those of ``get_nested_code_basis``.  The rows that
    ``get_eligible_rows`` marks are the logical operators of ``_get_distance_brute_force``, the
    others are its stabilizers, and the other arguments and the result are as there.
    """
    eligible = get_eligible_rows(labels)
    return _get_distance_brute_force(
        basis[eligible],
        basis[~eligible],
        cutoff=cutoff,
        batch_exponent=batch_exponent,
        symplectic=symplectic,
    )


def _get_distance_brute_force(
    logical_ops: npt.ArrayLike,
    stabilizers: npt.ArrayLike,
    *,
    cutoff: int,
    batch_exponent: int,
    symplectic: bool,
) -> int:
    """Find the minimum weight of a nontrivial logical operator by exhaustive enumeration.

    The candidates are the sums of a nonempty combination of the rows of ``logical_ops`` and any
    combination of the rows of ``stabilizers``.  The rows of both must together be independent.

    Args:
        logical_ops: Binary rows of logical operators.
        stabilizers: Binary rows of stabilizers.
        cutoff: Return early once the lightest operator seen has weight at most this cutoff.
        batch_exponent: Vectorize calculations over at most ``2**batch_exponent`` packed 64-bit
            integers.  That is fewer words if words are longer than 64 bits, and a single word if
            one word alone needs more.
        symplectic: If False, rows are weighed by Hamming weight.  If True, rows are symplectic
            vectors, with X support in their first half and Z support in their second half, and
            are weighed by symplectic weight.

    Returns:
        The minimum weight of a nontrivial logical operator.  After an early exit, it is instead the
        weight of the lightest operator seen, which is at most ``cutoff``.  Without ``logical_ops``,
        the result is one more than the number of qubits if symplectic, and otherwise than the
        number of columns.
    """
    num_bits = np.shape(logical_ops)[-1]

    weight_func = symplectic_weight if symplectic else np.bitwise_count
    if symplectic:
        logical_ops = riffle(logical_ops)
        stabilizers = riffle(stabilizers)

    int_logical_ops = pack_bits(np.asarray(logical_ops, dtype=np.uint8))
    int_stabilizers = pack_bits(np.asarray(stabilizers, dtype=np.uint8))
    num_stabilizers = len(int_stabilizers)

    # Vectorize over as many generators as keep a batch within ``2**batch_exponent`` packed
    # integers: each doubling of ``ints_per_row`` halves the words of a batch, down to one.
    ints_per_row = int_logical_ops.shape[-1]
    num_vectorized_ops = max(
        0,
        min(
            batch_exponent - (ints_per_row - 1).bit_length(),
            len(int_logical_ops) + len(int_stabilizers),
        ),
    )

    # Build a block of every combination of the first ``num_vectorized_ops`` generators, stabilizers
    # before logical operators, so that its rows from ``2**num_stabilizers`` on hold a logical
    # operator.  The sweep below covers the other generators.
    array = np.zeros((1, int_logical_ops.shape[-1]), dtype=np.uint64)
    for op in int_stabilizers[:num_vectorized_ops]:
        array = np.vstack([array, array ^ op])

    if num_vectorized_ops > num_stabilizers:
        # Add the first logical operators, as many as the stabilizers leave room for.
        for op in int_logical_ops[: num_vectorized_ops - num_stabilizers]:
            array = np.vstack([array, array ^ op])

        int_logical_ops = int_logical_ops[num_vectorized_ops - num_stabilizers :]

    int_stabilizers = int_stabilizers[num_vectorized_ops:]

    # Store the block in Fortran order, which makes the sweep below much faster.
    arrayf = np.asarray(array, order="F", dtype=np.uint64)

    # np.bitwise_count returns uint8, whose row sums would overflow at 256 columns, so the sweep
    # writes weights into this uint64 buffer.
    out = np.empty_like(arrayf)

    # Start the lightest weight seen at one more than any weight.  The early exits come only after
    # operators are weighed, so they never return this value.
    min_weight = (num_bits // 2 if symplectic else num_bits) + 1

    # Weigh the rows of the block that hold a logical operator, if any, in uint64 as above.
    logical_block = arrayf[2**num_stabilizers :]
    if len(logical_block):
        weights = np.asarray(weight_func(logical_block), dtype=np.uint64)
        min_weight = inplace_rowsum(weights).min()
        if min_weight <= cutoff:
            return int(min_weight)

    # Sweep the other generators over the block: each nonzero combination of the other logical
    # operators, with every combination of the other stabilizers.  Each loop steps through a
    # reflected Gray code, whose step ``i`` flips the operator indexed by the trailing zeros of
    # ``i``, so its ``2**k - 1`` steps over ``k`` operators reach each of the other combinations
    # once.
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
    """Count the qubits in the support of each riffled word, in which aligned bit pairs hold qubits.

    The result is uint64, and written into ``out`` if it is given.
    """
    out = np.right_shift(words, 1, out=out)
    out |= words
    out &= _MASK55
    return np.bitwise_count(out, out=out)


def inplace_rowsum(arr: npt.NDArray[np.uint64]) -> npt.NDArray[np.uint64]:
    """Compute ``arr.sum(-1)`` into ``arr[..., 0]``, and return that view.

    The other entries of ``arr`` are left indeterminate, which permits a faster sum.
    """
    width = arr.shape[-1]
    while width > 1:
        split = width // 2
        arr[..., :split] += arr[..., width - split : width]
        width -= split

    return arr[..., 0]


def riffle(array: npt.ArrayLike) -> npt.ArrayLike:
    """Riffle Pauli strings, putting the X and Z support bits of each qubit next to each other."""
    arr = np.asarray(array)
    num_bits = arr.shape[-1]
    assert num_bits % 2 == 0
    if arr.size == 0:
        # There is nothing to riffle, such as when there are no stabilizers.
        return np.empty((0, num_bits), dtype=arr.dtype)
    return np.reshape(arr, (-1, 2, num_bits // 2)).transpose(0, 2, 1).reshape(-1, num_bits)
