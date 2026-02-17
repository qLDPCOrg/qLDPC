"""Unit tests for distance.py

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

import itertools
from unittest import mock

import numba
import numpy as np
import numpy.typing as npt
import pytest

import qldpc


def _bitwise_count(
    val: npt.ArrayLike, out: npt.NDArray[np.uint] | None = None
) -> npt.NDArray[np.uint]:
    """Simplistic implementation of `bitwise_count` used to validate optimized variants."""
    val = np.asarray(val)
    nbits = 8 * val.itemsize

    if out is None:
        out = np.empty_like(val)

    for indices in np.ndindex(val.shape):
        ival = int(val[indices])
        out[indices] = sum(ival >> i & 1 for i in range(nbits))

    return out


def test_hamming_weight() -> None:
    vals = np.random.randint(0, 2**64, size=(7, 11), dtype=np.uint)
    expected_weights = _bitwise_count(vals)

    weights = qldpc.codes.distance._hamming_weight(vals)
    np.testing.assert_array_equal(weights, expected_weights)

    weight_fn = np.vectorize(qldpc.codes.distance._hamming_weight_single, signature="()->()")
    weights = weight_fn(vals)
    np.testing.assert_array_equal(weights, expected_weights)

    buf, out = np.random.randint(0, 2**64, size=(2, *vals.shape), dtype=vals.dtype)
    weights = qldpc.codes.distance._hamming_weight(vals, buf=buf, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert out is weights


def test_symplectic_weight() -> None:
    vals = np.random.randint(0, 2**64, size=(7, 11), dtype=np.uint)
    weights = qldpc.codes.distance._symplectic_weight(vals)
    expected_weights = _bitwise_count((vals | (vals >> np.uint(1))) & 0x5555555555555555)
    np.testing.assert_array_equal(weights, expected_weights)

    weight_fn = np.vectorize(qldpc.codes.distance._symplectic_weight_single, signature="()->()")
    weights = weight_fn(vals)
    np.testing.assert_array_equal(weights, expected_weights)

    buf, out = np.random.randint(0, 2**64, size=(2, *vals.shape), dtype=vals.dtype)
    weights = qldpc.codes.distance._symplectic_weight(vals, buf=buf, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert out is weights


def test_get_hamming_weight_fn() -> None:
    generators = np.random.randint(2, size=(4, 64), dtype=np.uint)
    weight_fn, nbuf = qldpc.codes.distance._get_hamming_weight_fn()
    weights_default = weight_fn(generators)

    # Tests should work with numpy < 2.0.0 so provide a backup `np.bitwise_count` implementation
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    with mock.patch.object(np, "bitwise_count", wraps=mock_weight, create=True) as patched:
        weight_fn, nbuf = qldpc.codes.distance._get_hamming_weight_fn(use_numba=False)
        assert weight_fn is not qldpc.codes.distance._hamming_weight
        assert nbuf == 0

        out = np.empty_like(generators)
        weights = weight_fn(generators, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out
        patched.assert_called_once()

    with mock.patch.object(np, "bitwise_count", None, create=True):
        weight_fn, nbuf = qldpc.codes.distance._get_hamming_weight_fn(use_numba=False)
        assert weight_fn is qldpc.codes.distance._hamming_weight
        assert nbuf == 1

        out = np.empty_like(generators)
        weights = weight_fn(generators, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out

        buf = np.empty_like(generators)
        weights = weight_fn(generators, buf, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out

    weight_fn, nbuf = qldpc.codes.distance._get_hamming_weight_fn(use_numba=True)
    assert isinstance(weight_fn, numba.np.ufunc.dufunc.DUFunc)
    assert nbuf == 0

    out = np.empty_like(generators)
    weights = weight_fn(generators, out=out)
    np.testing.assert_array_equal(weights, weights_default)
    assert weights is out


def test_get_symplectic_weight_fn() -> None:
    generators = np.random.randint(2, size=(4, 56), dtype=np.uint)
    weight_fn, nbuf = qldpc.codes.distance._get_symplectic_weight_fn()
    weights_default = weight_fn(generators)

    # Tests should work with numpy < 2.0.0 so provide a backup `np.bitwise_count` implementation
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # Using np.bitwise_count:
    with mock.patch.object(np, "bitwise_count", wraps=mock_weight, create=True) as patched:
        weight_fn, nbuf = qldpc.codes.distance._get_symplectic_weight_fn(use_numba=False)
        assert weight_fn is not qldpc.codes.distance._hamming_weight
        assert nbuf == 1

        out = np.empty_like(generators)
        weights = weight_fn(generators, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out
        patched.assert_called_once()

        buf = np.empty_like(generators)
        weights = weight_fn(generators, buf, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out

    # Using qldpc.codes.distance._symplectic_weight:
    with mock.patch.object(np, "bitwise_count", None, create=True):
        weight_fn, nbuf = qldpc.codes.distance._get_symplectic_weight_fn(use_numba=False)
        assert weight_fn is qldpc.codes.distance._symplectic_weight
        assert nbuf == 1

        out = np.empty_like(generators)
        weights = weight_fn(generators, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out

        buf = np.empty_like(generators)
        weights = weight_fn(generators, buf, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out

    # Using numba:
    weight_fn, nbuf = qldpc.codes.distance._get_symplectic_weight_fn(use_numba=True)
    assert isinstance(weight_fn, numba.np.ufunc.dufunc.DUFunc)
    assert nbuf == 0

    out = np.empty_like(generators)
    weights = weight_fn(generators, out=out)
    assert weights is out
    np.testing.assert_array_equal(weights, weights_default)


@pytest.mark.parametrize(
    "base_val",
    [1, 2**64 - 1, int(np.random.randint(2**64, dtype=np.uint)) | 1],
)
def test_count_trailing_zeros(base_val: int) -> None:
    for i in range(128):
        assert qldpc.codes.distance._count_trailing_zeros(base_val << i) == i


@pytest.mark.parametrize("width", range(1, 8))
def test_inplace_rowsum(width: int) -> None:
    arr = np.random.randint(2**31, size=(10, width), dtype=np.uint)
    expected = arr.sum(-1)
    actual = qldpc.codes.distance._inplace_rowsum(arr)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, arr[:, 0])


def test_rows_to_ints_endianness() -> None:
    # Compare bit order to that used by `np.packbits`
    bits = np.random.randint(2, size=(10, 120))
    ints = qldpc.codes.distance._rows_to_ints(bits, dtype=np.uint8)
    assert ints.shape == (10, 15)

    expected = np.packbits(bits).reshape(10, 15)
    np.testing.assert_array_equal(ints, expected)


@pytest.mark.parametrize("dtype", [int, np.uint, np.uint8, np.int16])
def test_rows_to_ints(dtype: npt.DTypeLike) -> None:
    bits = np.random.randint(2, size=(10, 93))
    ints = qldpc.codes.distance._rows_to_ints(bits, dtype=dtype)

    nbits = 8 * np.dtype(dtype).itemsize
    expected_words_per_row = int(np.ceil(93 / nbits))
    assert ints.shape == (10, expected_words_per_row)
    assert ints.dtype == np.dtype(dtype)

    np.testing.assert_array_equal(_bitwise_count(ints).sum(-1), bits.sum(-1))

    for indices in np.ndindex(ints.shape):
        i = indices[-1] * nbits
        packed_bits = bits[indices[:-1]][i : i + nbits]
        bitstr = "".join(map(str, packed_bits))
        assert np.binary_repr(ints[indices], len(bitstr)) == bitstr

    # Pack array with more dimensions
    bits = bits.reshape(5, 1, 2, -1)
    np.testing.assert_array_equal(
        qldpc.codes.distance._rows_to_ints(bits, dtype=dtype),
        ints.reshape(5, 1, 2, -1),
    )

    # Pack along a different axis
    bits = bits.swapaxes(0, 3)
    np.testing.assert_array_equal(
        qldpc.codes.distance._rows_to_ints(bits, dtype=dtype, axis=0),
        ints.reshape(5, 1, 2, -1).swapaxes(0, 3),
    )

    bits = np.zeros((11, 0), dtype=dtype)
    ints = qldpc.codes.distance._rows_to_ints(bits, dtype=dtype)
    np.testing.assert_array_equal(
        qldpc.codes.distance._rows_to_ints(bits, dtype=dtype, axis=0), bits
    )


@pytest.mark.parametrize("block_size", range(1, 14))
def test_get_distance_classical(block_size: int) -> None:
    generators = np.random.randint(2, size=(9, 137))

    # Intercept `hamming_weight` calls to check that every nontrivial combination of generators
    # is observed exactly once
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_hamming_weight(
        arr: npt.NDArray[np.uint],
        buf: npt.NDArray[np.uint] | None = None,
        out: npt.NDArray[np.uint] | None = None,
    ) -> npt.NDArray[np.uint]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._hamming_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance._get_hamming_weight_fn", return_value=(_mock_hamming_weight, 0)
    ):
        distance = qldpc.codes.distance.get_distance_classical(generators, block_size=block_size)

    int_generators = qldpc.codes.distance._rows_to_ints(generators)
    expected_bitstrings = [
        tuple(np.bitwise_xor.reduce(np.vstack(gens)).tolist())
        for n in range(1, len(generators) + 1)
        for gens in itertools.combinations(int_generators, n)
    ]

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(expected_bitstrings)

    observed_array = np.array(observed_bitstrings, dtype=np.uint)
    expected_distance = _bitwise_count(observed_array).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("block_size", range(1, 14))
def test_get_distance_quantum(block_size: int) -> None:
    stabilizers = np.random.randint(2, size=(8, 97))
    logical_ops = np.random.randint(2, size=(5, 97))

    # Intercept `hamming_weight` calls to check that every combination of stabilizers and at least
    # one logical op is observed exactly once
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_hamming_weight(
        arr: npt.NDArray[np.uint],
        buf: npt.NDArray[np.uint] | None = None,
        out: npt.NDArray[np.uint] | None = None,
    ) -> npt.NDArray[np.uint]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._hamming_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance._get_hamming_weight_fn", return_value=(_mock_hamming_weight, 0)
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, homogeneous=True, block_size=block_size
        )

    int_stabilizers = qldpc.codes.distance._rows_to_ints(stabilizers)
    int_logical_ops = qldpc.codes.distance._rows_to_ints(logical_ops)
    expected_bitstrings = [
        tuple(np.bitwise_xor.reduce(np.vstack(stabs + ops)).tolist())
        for ns in range(len(stabilizers) + 1)
        for stabs in itertools.combinations(int_stabilizers, ns)
        for nl in range(1, len(logical_ops) + 1)
        for ops in itertools.combinations(int_logical_ops, nl)
    ]

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(expected_bitstrings)

    observed_array = np.array(observed_bitstrings, dtype=np.uint)
    expected_distance = _bitwise_count(observed_array).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("block_size", range(1, 14))
def test_get_distance_quantum_symplectic(block_size: int) -> None:
    stabilizers = np.random.randint(2, size=(3, 98))
    logical_ops = np.random.randint(2, size=(5, 98))

    # Intercept `symplectic_weight` calls to check that every combination of stabilizers and at
    # least one logical op is observed exactly once
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_symplectic_weight(
        arr: npt.NDArray[np.uint],
        buf: npt.NDArray[np.uint] | None = None,
        out: npt.NDArray[np.uint] | None = None,
    ) -> npt.NDArray[np.uint]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._symplectic_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance._get_symplectic_weight_fn", return_value=(_mock_symplectic_weight, 0)
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, homogeneous=False, block_size=block_size
        )

    int_stabilizers = qldpc.codes.distance._rows_to_ints(qldpc.codes.distance._riffle(stabilizers))
    int_logical_ops = qldpc.codes.distance._rows_to_ints(qldpc.codes.distance._riffle(logical_ops))
    expected_bitstrings = np.array(
        [
            np.bitwise_xor.reduce(np.vstack(stabs + ops))
            for ns in range(len(stabilizers) + 1)
            for stabs in itertools.combinations(int_stabilizers, ns)
            for nl in range(1, len(logical_ops) + 1)
            for ops in itertools.combinations(int_logical_ops, nl)
        ]
    )

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(map(tuple, expected_bitstrings))

    vals = (expected_bitstrings | (expected_bitstrings >> 1)) & 0x5555555555555555
    expected_distance = _bitwise_count(vals).sum(-1).min()
    assert distance == expected_distance


def test_get_distance_classical_methods() -> None:
    generators = np.random.randint(2, size=(6, 56), dtype=np.uint)
    distance_default = qldpc.codes.distance.get_distance_classical(generators, block_size=3)

    # Tests should work with numpy < 2.0.0 so provide a backup `np.bitwise_count` implementation
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # Using np.bitwise_count:
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._hamming_weight",
            wraps=qldpc.codes.distance._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_classical(generators, block_size=3)
        bitcount.assert_called()
        fallback.assert_not_called()
        assert distance == distance_default

    # Using fallback (qldpc.codes.distance._hamming_weight):
    with (
        mock.patch("numpy.bitwise_count", None, create=True),
        mock.patch(
            "qldpc.codes.distance._hamming_weight",
            wraps=qldpc.codes.distance._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_classical(generators, block_size=3)
        fallback.assert_called()
        assert distance == distance_default

    # Using numba:
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._hamming_weight",
            wraps=qldpc.codes.distance._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_classical(
            generators, block_size=3, use_numba=True
        )
        bitcount.assert_not_called()
        fallback.assert_not_called()
        assert distance == distance_default


def test_get_distance_quantum_methods() -> None:
    stabilizers = np.random.randint(2, size=(4, 56), dtype=np.uint)
    logical_ops = np.random.randint(2, size=(3, 56), dtype=np.uint)
    distance_default = qldpc.codes.distance.get_distance_quantum(
        logical_ops, stabilizers, block_size=3, homogeneous=True
    )

    # Tests should work with numpy < 2.0.0 so provide a backup `np.bitwise_count` implementation
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # Using np.bitwise_count:
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._hamming_weight",
            wraps=qldpc.codes.distance._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, block_size=3, homogeneous=True, use_numba=False
        )
        bitcount.assert_called()
        fallback.assert_not_called()
        assert distance == distance_default

    # Using fallback (qldpc.codes.distance._hamming_weight):
    with (
        mock.patch("numpy.bitwise_count", None, create=True),
        mock.patch(
            "qldpc.codes.distance._hamming_weight",
            wraps=qldpc.codes.distance._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, block_size=3, homogeneous=True, use_numba=False
        )
        fallback.assert_called()
        assert distance == distance_default

    # Using numba:
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._hamming_weight",
            wraps=qldpc.codes.distance._hamming_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, block_size=3, homogeneous=True, use_numba=True
        )
        bitcount.assert_not_called()
        fallback.assert_not_called()
        assert distance == distance_default


def test_get_distance_quantum_methods_symplectic() -> None:
    stabilizers = np.random.randint(2, size=(4, 56), dtype=np.uint)
    logical_ops = np.random.randint(2, size=(3, 56), dtype=np.uint)
    distance_default = qldpc.codes.distance.get_distance_quantum(
        logical_ops, stabilizers, block_size=3, homogeneous=False
    )

    # Tests should work with numpy < 2.0.0 so provide a backup `np.bitwise_count` implementation
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # Using np.bitwise_count:
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._symplectic_weight",
            wraps=qldpc.codes.distance._symplectic_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, block_size=3, homogeneous=False, use_numba=False
        )
        bitcount.assert_called()
        fallback.assert_not_called()
        assert distance == distance_default

    # Using fallback (qldpc.codes.distance._symplectic_weight):
    with (
        mock.patch("numpy.bitwise_count", None, create=True),
        mock.patch(
            "qldpc.codes.distance._symplectic_weight",
            wraps=qldpc.codes.distance._symplectic_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, block_size=3, homogeneous=False, use_numba=False
        )
        fallback.assert_called()
        assert distance == distance_default

    # Using numba:
    with (
        mock.patch("numpy.bitwise_count", wraps=mock_weight, create=True) as bitcount,
        mock.patch(
            "qldpc.codes.distance._symplectic_weight",
            wraps=qldpc.codes.distance._symplectic_weight,
        ) as fallback,
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops, stabilizers, block_size=3, homogeneous=False, use_numba=True
        )
        bitcount.assert_not_called()
        fallback.assert_not_called()
        assert distance == distance_default


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.classical.HammingCode(4), 3),
        (qldpc.codes.classical.RepetitionCode(3), 3),
        (qldpc.codes.classical.RepetitionCode(8), 8),
        (qldpc.codes.classical.RingCode(8), 8),
    ],
)
def test_get_distance_classical_known_codes(
    code: qldpc.codes.ClassicalCode, expected_distance: int
) -> None:
    distance = qldpc.codes.distance.get_distance_classical(code.generator)
    assert distance == expected_distance

    code._distance = None
    assert code.get_distance_exact() == expected_distance


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.quantum.C4Code(), 2),
        (qldpc.codes.quantum.C6Code(), 2),
        (qldpc.codes.quantum.SteaneCode(), 3),
        (qldpc.codes.quantum.SurfaceCode(3), 3),
        (qldpc.codes.quantum.SurfaceCode(4), 4),
        (qldpc.codes.quantum.ToricCode(4), 4),
    ],
)
def test_get_distance_quantum_css_codes(code: qldpc.codes.CSSCode, expected_distance: int) -> None:
    distance_x = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(qldpc.objects.Pauli.X),
        code.get_stabilizer_ops(qldpc.objects.Pauli.X),
        homogeneous=True,
    )
    distance_z = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(qldpc.objects.Pauli.Z),
        code.get_stabilizer_ops(qldpc.objects.Pauli.Z),
        homogeneous=True,
    )
    distance_all = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        homogeneous=True,
    )
    assert min(distance_x, distance_z, distance_all) == expected_distance

    code._distance = None
    assert code.get_distance_exact() == expected_distance


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.quantum.FiveQubitCode(), 3),
    ],
)
def test_get_distance_quantum_noncss_codes(
    code: qldpc.codes.QuditCode, expected_distance: int
) -> None:
    distance = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        homogeneous=False,
    )
    assert distance == expected_distance

    code._distance = None
    assert code.get_distance_exact() == expected_distance
