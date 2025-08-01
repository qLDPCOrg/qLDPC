"""Methods for computing the (exact) distance of error-correcting codes

Copyright 2023 The qLDPC Authors and Infleqtion Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

_mask55 = np.uint(0x5555555555555555)
_mask33 = np.uint(0x3333333333333333)
_mask0F = np.uint(0x0F0F0F0F0F0F0F0F)
_mask01 = np.uint(0x0101010101010101)


def _hamming_weight(
    arr: npt.NDArray[np.uint],
    buf: npt.NDArray[np.uint] | None = None,
    out: npt.NDArray[np.uint] | None = None,
) -> npt.NDArray[np.uint]:
    """Somewhat efficient (vectorized) Hamming weight calculation. Assumes 64-bit uints.

    For `numpy >= 2.0.0`, it's generally better to use `np.bitwise_count` (which uses processors'
    builtin `popcnt` instruction). Unfortunately this isn't available for numpy < 2.0.0.
    """
    out = np.right_shift(arr, 1, out=out)
    out &= _mask55
    out = np.subtract(arr, out, out=out)

    buf = np.right_shift(out, 2, out=buf)
    buf &= _mask33
    out &= _mask33
    out += buf

    buf = np.right_shift(out, 4, out=buf)
    out += buf
    out &= _mask0F

    # out *= _mask01
    out = np.multiply(out, _mask01, out=out)
    out >>= np.uint(56)
    return out


def _symplectic_weight(
    arr: npt.NDArray[np.uint],
    buf: npt.NDArray[np.uint] | None = None,
    out: npt.NDArray[np.uint] | None = None,
) -> npt.NDArray[np.uint]:
    """Somewhat efficient (vectorized) symplectic weight calculation. Assumes 64-bit uints.

    This function is equivalent to (but slightly more efficient than) the expression
    ``_hamming_weight((arr | (arr >> 1)) & 0x5555555555555555, buf=buf, out=out)``.
    """
    out = np.right_shift(arr, 1, out=out)
    out |= arr
    out &= _mask55

    buf = np.right_shift(out, 2, out=buf)
    buf &= _mask33
    out &= _mask33
    out += buf

    buf = np.right_shift(out, 4, out=buf)
    out += buf
    out &= _mask0F

    out *= _mask01
    out >>= np.uint(56)
    return out


def _hamming_weight_single(val: np.uint) -> np.uint:
    """Unbuffered version of `_hamming_weight`, useful for vectorization."""
    out = val >> np.uint(1)
    out &= _mask55
    out = val - out

    buf = out >> np.uint(2)
    buf &= _mask33
    out &= _mask33
    out += buf

    buf = out >> np.uint(4)
    out += buf
    out &= _mask0F

    out = np.multiply(out, _mask01)
    out >>= np.uint(56)
    return out


def _symplectic_weight_single(val: np.uint) -> np.uint:
    """Unbuffered version of `_symplectic_weight`, useful for vectorization."""
    out = val >> np.uint(1)
    out |= val
    out &= _mask55

    buf = out >> np.uint(2)
    buf &= _mask33
    out &= _mask33
    out += buf

    buf = out >> np.uint(4)
    out += buf
    out &= _mask0F

    out = np.multiply(out, _mask01)
    out >>= np.uint(56)
    return out


def _get_hamming_weight_fn(
    use_numba: bool = False,
) -> tuple[Callable[..., npt.NDArray[np.uint]], int]:
    if use_numba:
        import numba

        weight_fn = numba.vectorize([numba.uint(numba.uint)])(_hamming_weight_single)
        return weight_fn, 0

    if getattr(np, "bitwise_count", None) is not None:
        weight_fn = getattr(np, "bitwise_count")
        return weight_fn, 0

    return _hamming_weight, 1


def _get_symplectic_weight_fn(
    use_numba: bool = False,
) -> tuple[Callable[..., npt.NDArray[np.uint]], int]:
    if use_numba:
        import numba

        weight_fn = numba.vectorize([numba.uint(numba.uint)])(_symplectic_weight_single)
        return weight_fn, 0

    if getattr(np, "bitwise_count", None) is not None:
        np_bitwise_count = getattr(np, "bitwise_count")

        def weight_fn(
            arr: npt.NDArray[np.uint],
            buf: npt.NDArray[np.uint] | None = None,
            out: npt.NDArray[np.uint] | None = None,
        ) -> npt.NDArray[np.uint]:
            """Symplectic weight of an integer."""
            buf = np.right_shift(arr, 1, out=buf)
            buf |= arr
            buf &= _mask55
            return np_bitwise_count(buf, out=out)

        return weight_fn, 1

    return _symplectic_weight, 1


def _count_trailing_zeros(val: int) -> int:
    """Returns the position of the least significant 1 in the binary representation of `val`."""
    return (val & -val).bit_length() - 1


def _inplace_rowsum(arr: npt.NDArray[np.uint]) -> npt.NDArray[np.uint]:
    """Destructively compute ``arr.sum(-1)``, placing the result in the first column or `arr`.

    When complete, the returned sum will be stored in ``arr[..., 0]``, while other entries in
    ``arr[..., 1:]`` will be left in indeterminate states. This permits a faster sum implementation.
    """
    width = arr.shape[-1]
    while width > 1:
        split = width // 2
        arr[..., :split] += arr[..., width - split : width]
        width -= split

    return arr[..., 0]


def _rows_to_ints(
    array: npt.ArrayLike, dtype: npt.DTypeLike = np.uint, axis: int = -1
) -> npt.NDArray[np.uint]:
    """Pack rows of a binary array into rows of the given integral type."""
    array = np.asarray(array, dtype=dtype)
    tsize = array.itemsize * 8

    if array.size == 0:
        num_words = int(np.ceil(array.shape[-1] / tsize))
        return np.empty((*array.shape[:-1], num_words), dtype=dtype)

    def _to_int(bits: npt.NDArray[np.uint]) -> npt.NDArray[np.uint]:
        """Pack `bits` into a single integer (of type `dtype`)."""
        return (bits << np.arange(len(bits) - 1, -1, -1, dtype=dtype)).sum(dtype=dtype)

    def _to_ints(bits: npt.NDArray[np.uint]) -> list[npt.NDArray[np.uint]]:
        """Pack a single row of bits into a row of integers."""
        return [_to_int(bits[i : i + tsize]) for i in range(0, np.shape(bits)[-1], tsize)]

    return np.apply_along_axis(_to_ints, axis, array)


def _riffle(array: npt.ArrayLike) -> npt.ArrayLike:
    """'Riffle' Pauli strings, putting the X and Z support bits for each qubit next to each other."""
    num_bits = np.shape(array)[-1]
    assert num_bits % 2 == 0
    return np.reshape(array, (-1, 2, num_bits // 2)).transpose(0, 2, 1).reshape(-1, num_bits)


def get_distance_classical(
    generators: npt.ArrayLike, block_size: int = 15, use_numba: bool = False
) -> int:
    """Distance of a classical linear binary code."""

    # This calculation is exactly the same as in the quantum case, but with no stabilizers
    return get_distance_quantum(
        logical_ops=generators,
        stabilizers=[],
        block_size=block_size,
        homogeneous=True,
        use_numba=use_numba,
    )


def get_distance_quantum(
    logical_ops: npt.ArrayLike,
    stabilizers: npt.ArrayLike,
    block_size: int = 15,
    homogeneous: bool = False,
    use_numba: bool = False,
) -> int:
    """Distance of a binary quantum code.

    If homogeneous is True, all logical operators and stabilizers have the same homogeneous (X or Z)
    type, meaning:
    - Pauli strings are represented by bitstrings that indicate their support on each qubit, so
    - the weight of a Pauli string is the Hamming weight of the corresponding bitstring.

    If homogeneous is False, each Pauli string is represented by a bitstring with length equal to
    twice the qubit number.  The first and second half of this bitstring indicate, respectively, the
    X and Z support of the corresponding Pauli string.  The weight of a Pauli string is then the
    symplectic weight of the corresponding bitstring.
    """
    num_bits = np.shape(logical_ops)[-1]

    if homogeneous:
        weight_func, nbuf = _get_hamming_weight_fn(use_numba)
    else:
        weight_func, nbuf = _get_symplectic_weight_fn(use_numba)

        logical_ops = _riffle(logical_ops)
        stabilizers = _riffle(stabilizers)

    int_logical_ops = _rows_to_ints(logical_ops, dtype=np.uint)
    int_stabilizers = _rows_to_ints(stabilizers, dtype=np.uint)
    num_stabilizers = len(int_stabilizers)

    # Number of generators to include in the operational array. Most calculations will then be
    # vectorized over ``2**block_size`` values
    num_vectorized_ops = min(
        block_size + 1 - int_logical_ops.shape[-1],
        len(int_logical_ops) + len(int_stabilizers),
    )

    # Vectorize all combinations of first `num_vectorized_ops` stabilizers
    array = np.zeros((1, int_logical_ops.shape[-1]), dtype=np.uint)
    for op in int_stabilizers[:num_vectorized_ops]:
        array = np.vstack([array, array ^ op])

    if num_vectorized_ops > num_stabilizers:
        # fill out block with products of some logical ops
        for op in int_logical_ops[: num_vectorized_ops - num_stabilizers]:
            array = np.vstack([array, array ^ op])

        int_logical_ops = int_logical_ops[num_vectorized_ops - num_stabilizers :]

    int_stabilizers = int_stabilizers[num_vectorized_ops:]

    # Everything below will run much faster if we use Fortran-style ordering
    arrayf = np.asarray(array, order="F")

    out = np.empty_like(arrayf)
    bufs = [np.empty_like(arrayf) for _ in range(nbuf)]

    # Min weight of the part containing logical ops
    weights = weight_func(arrayf[2**num_stabilizers :])
    min_weight = _inplace_rowsum(weights).min(initial=num_bits)

    for li in range(1, 2 ** len(int_logical_ops)):
        arrayf ^= int_logical_ops[_count_trailing_zeros(li)]
        weights = weight_func(arrayf, *bufs, out=out)
        min_weight = _inplace_rowsum(weights).min(initial=min_weight)

        for si in range(1, 2 ** len(int_stabilizers)):
            arrayf ^= int_stabilizers[_count_trailing_zeros(si)]
            weights = weight_func(arrayf, *bufs, out=out)
            min_weight = _inplace_rowsum(weights).min(initial=min_weight)

    return int(min_weight)
