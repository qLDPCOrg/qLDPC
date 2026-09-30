# SPDX-License-Identifier: Apache-2.0

"""Unit tests for distance.py."""

from __future__ import annotations

import itertools
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

import qldpc


def _bitwise_count(
    val: npt.ArrayLike, out: npt.NDArray[np.uint64] | None = None
) -> npt.NDArray[np.uint64]:
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
    """Validate _hamming_weight against a simple reference bit-counting implementation."""
    vals = np.random.randint(0, 2**64, size=(7, 11), dtype=np.uint64)
    expected_weights = _bitwise_count(vals)

    weights = qldpc.codes.distance._hamming_weight(vals)
    np.testing.assert_array_equal(weights, expected_weights)

    buf, out = np.random.randint(0, 2**64, size=(2, *vals.shape), dtype=vals.dtype)
    weights = qldpc.codes.distance._hamming_weight(vals, buf=buf, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert out is weights


def test_symplectic_weight() -> None:
    """Validate _symplectic_weight against the reference bit-counting implementation."""
    vals = np.random.randint(0, 2**64, size=(7, 11), dtype=np.uint64)
    weights = qldpc.codes.distance._symplectic_weight(vals)
    expected_weights = _bitwise_count((vals | (vals >> np.uint64(1))) & 0x5555555555555555)
    np.testing.assert_array_equal(weights, expected_weights)

    buf, out = np.random.randint(0, 2**64, size=(2, *vals.shape), dtype=vals.dtype)
    weights = qldpc.codes.distance._symplectic_weight(vals, buf=buf, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert out is weights


def test_get_hamming_weight_fn() -> None:
    """_get_hamming_weight_fn selects NumPy bitcount when available and the fallback otherwise."""
    generators = np.random.randint(2, size=(4, 64), dtype=np.uint64)
    weight_fn, nbuf = qldpc.codes.distance._get_hamming_weight_fn()
    weights_default = weight_fn(generators)

    # Tests should work with numpy < 2.0.0 so provide a backup `np.bitwise_count` implementation
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    with mock.patch.object(np, "bitwise_count", wraps=mock_weight, create=True) as patched:
        weight_fn, nbuf = qldpc.codes.distance._get_hamming_weight_fn()
        assert weight_fn is not qldpc.codes.distance._hamming_weight
        assert nbuf == 0

        out = np.empty_like(generators)
        weights = weight_fn(generators, out=out)
        np.testing.assert_array_equal(weights, weights_default)
        assert weights is out
        patched.assert_called_once()

    with mock.patch.object(np, "bitwise_count", None, create=True):
        weight_fn, nbuf = qldpc.codes.distance._get_hamming_weight_fn()
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


def test_get_symplectic_weight_fn() -> None:
    """_get_symplectic_weight_fn uses NumPy bitcount when available and the fallback otherwise."""
    generators = np.random.randint(2, size=(4, 56), dtype=np.uint64)
    weight_fn, nbuf = qldpc.codes.distance._get_symplectic_weight_fn()
    weights_default = weight_fn(generators)

    # Tests should work with numpy < 2.0.0 so provide a backup `np.bitwise_count` implementation
    mock_weight = getattr(np, "bitwise_count", _bitwise_count)

    # Using np.bitwise_count:
    with mock.patch.object(np, "bitwise_count", wraps=mock_weight, create=True) as patched:
        weight_fn, nbuf = qldpc.codes.distance._get_symplectic_weight_fn()
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
        weight_fn, nbuf = qldpc.codes.distance._get_symplectic_weight_fn()
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


@pytest.mark.parametrize(
    "base_val",
    [1, 2**64 - 1, int(np.random.randint(2**64, dtype=np.uint64)) | 1],
)
def test_count_trailing_zeros(base_val: int) -> None:
    """_count_trailing_zeros(val << i) == i for several base values across all 128 shift amounts."""
    for i in range(128):
        assert qldpc.codes.distance._count_trailing_zeros(base_val << i) == i


@pytest.mark.parametrize("width", range(1, 8))
def test_inplace_rowsum(width: int) -> None:
    """_inplace_rowsum reduces each row to its sum and stores the result in the first column."""
    arr = np.random.randint(2**31, size=(10, width), dtype=np.uint64)
    expected = arr.sum(-1)
    actual = qldpc.codes.distance._inplace_rowsum(arr)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, arr[:, 0])


def test_rows_to_ints_endianness() -> None:
    """_rows_to_ints produces the same bit order as np.packbits."""
    # Compare bit order to that used by `np.packbits`
    bits = np.random.randint(2, size=(10, 120))
    ints = qldpc.codes.distance._rows_to_ints(bits, dtype=np.uint8)
    assert ints.shape == (10, 15)

    expected = np.packbits(bits).reshape(10, 15)
    np.testing.assert_array_equal(ints, expected)


@pytest.mark.parametrize("dtype", [int, np.uint64, np.uint8, np.int16])
def test_rows_to_ints(dtype: npt.DTypeLike) -> None:
    """_rows_to_ints packs binary rows into integer words of the requested dtype."""
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


def _get_random_full_rank_matrix(
    rng: np.random.Generator, rows: int, cols: int
) -> npt.NDArray[np.uint8]:
    """Sample a full-row-rank binary matrix."""
    while True:
        matrix = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)
        if len(qldpc.codes.distance._get_independent_rows(matrix)) == rows:
            return matrix


def test_binary_row_reduction_and_information_sets() -> None:
    """Binary elimination transforms quotient labels and retains the residual rank."""
    matrix = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.uint8)
    labels = np.eye(2, dtype=np.uint8)
    reduced, reduced_labels, pivots = qldpc.codes.distance._row_reduce_binary(matrix, labels)
    np.testing.assert_array_equal(reduced, [[1, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(reduced_labels, [[1, 1], [1, 0]])
    np.testing.assert_array_equal(pivots, [0, 1])

    basis = np.hstack(
        [
            np.eye(3, dtype=np.uint8),
            np.eye(3, dtype=np.uint8),
            np.array([[1, 0], [0, 1], [0, 0]], dtype=np.uint8),
        ]
    )
    information_sets = qldpc.codes.distance._get_information_set_generators(
        basis, np.eye(3, dtype=np.uint8)
    )
    assert [rank for _, _, rank in information_sets] == [3, 3, 2]
    assert not qldpc.codes.distance._get_information_set_generators(
        np.zeros((1, 3), dtype=np.uint8), None
    )


def test_brouwer_zimmermann_excludes_stabilizers() -> None:
    """Only nonzero logical quotient labels may improve the BZ upper bound."""
    stabilizers = np.array([[1, 0, 0, 0]], dtype=np.uint8)
    logical_ops = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    assert (
        qldpc.codes.get_distance_quantum(logical_ops, stabilizers, homogeneous=True, cutoff=0) == 3
    )
    assert (
        qldpc.codes.get_distance_quantum(
            logical_ops,
            stabilizers,
            homogeneous=True,
            cutoff=0,
            block_size=0,
        )
        == 3
    )

    # Small-dimensional exhaustive mode skips combinations made only from stabilizers.
    assert (
        qldpc.codes.get_distance_quantum(
            [[0, 0, 1]],
            [[1, 0, 0], [0, 1, 0]],
            homogeneous=True,
            cutoff=0,
            block_size=0,
        )
        == 1
    )

    # The lightest logical can be a product of several chosen logical generators.
    logical_ops = np.array([[1, 1, 1, 0], [1, 1, 0, 1]], dtype=np.uint8)
    assert qldpc.codes.get_distance_quantum(logical_ops, [], homogeneous=True, cutoff=0) == 2

    # Redundant rows are reduced rather than creating a spurious zero-weight word.
    generators = np.array([[1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    assert qldpc.codes.get_distance_classical(generators, cutoff=0) == 3


def test_brouwer_zimmermann_random_cross_checks() -> None:
    """BZ agrees with exhaustive enumeration on small independent random bases."""
    rng = np.random.default_rng(118)
    for length in range(5, 10):
        dimension = min(5, length - 1)
        basis = _get_random_full_rank_matrix(rng, dimension, length)
        expected = qldpc.codes.get_distance_classical(basis, cutoff=0, method="brute_force")
        assert qldpc.codes.get_distance_classical(basis, cutoff=0) == expected
        assert qldpc.codes.get_distance_classical(basis) == expected

        num_stabilizers = dimension // 2
        stabilizers = basis[:num_stabilizers]
        logical_ops = basis[num_stabilizers:]
        expected = qldpc.codes.get_distance_quantum(
            logical_ops,
            stabilizers,
            cutoff=0,
            homogeneous=True,
            method="brute_force",
        )
        assert (
            qldpc.codes.get_distance_quantum(logical_ops, stabilizers, cutoff=0, homogeneous=True)
            == expected
        )
        assert (
            qldpc.codes.get_distance_quantum(logical_ops, stabilizers, homogeneous=True) == expected
        )

    for num_qubits in range(3, 7):
        basis = _get_random_full_rank_matrix(rng, min(6, 2 * num_qubits - 1), 2 * num_qubits)
        stabilizers = basis[:2]
        logical_ops = basis[2:]
        expected = qldpc.codes.get_distance_quantum(
            logical_ops, stabilizers, cutoff=0, method="brute_force"
        )
        assert qldpc.codes.get_distance_quantum(logical_ops, stabilizers, cutoff=0) == expected
        assert qldpc.codes.get_distance_quantum(logical_ops, stabilizers) == expected


@pytest.mark.parametrize("length", [63, 64, 65, 127, 128, 129, 255, 256, 257])
def test_brouwer_zimmermann_packed_widths(length: int) -> None:
    """BZ weights do not overflow at uint64 or uint8 reduction boundaries."""
    generators = np.ones((1, length), dtype=np.uint8)
    assert qldpc.codes.get_distance_classical(generators, cutoff=0) == length


@pytest.mark.parametrize("dimension", [1, 9, 16, 18])
def test_brouwer_zimmermann_low_dimension_fast_path(dimension: int) -> None:
    """Long, low-dimensional codes bypass quadratic information-set construction."""
    repeats = int(np.ceil(4000 / dimension))
    generators = np.tile(np.eye(dimension, dtype=np.uint8), (1, repeats))[:, :4000]
    with mock.patch("qldpc.codes.distance._get_information_set_generators") as information_sets:
        assert qldpc.codes.get_distance_classical(generators, cutoff=0) == 4000 // dimension
    information_sets.assert_not_called()


def test_brouwer_zimmermann_information_set_paths() -> None:
    """The information-set path filters labels and honors cutoffs for dimension above eight."""
    identity = np.eye(16, dtype=np.uint8)
    assert (
        qldpc.codes.get_distance_quantum(
            identity[-1:],
            identity[:-1],
            homogeneous=True,
            cutoff=0,
            block_size=0,
        )
        == 1
    )

    assert (
        qldpc.codes.get_distance_quantum(
            [np.ones(16, dtype=np.uint8)],
            identity[:-1],
            homogeneous=True,
            cutoff=1,
        )
        == 1
    )

    code = qldpc.codes.QuditCode.stack([qldpc.codes.FiveQubitCode()] * 3)
    assert (
        qldpc.codes.get_distance_quantum(
            code.get_logical_ops(),
            code.get_stabilizer_ops(),
            cutoff=0,
        )
        == 3
    )


def test_brouwer_zimmermann_dispatch_paths() -> None:
    """Default dispatch covers exhaustive and information-set search paths exactly."""
    code = qldpc.codes.QuditCode.stack([qldpc.codes.FiveQubitCode()] * 5)
    logical_ops = code.get_logical_ops()
    stabilizers = code.get_stabilizer_ops()
    expected = qldpc.codes.get_distance_quantum(
        logical_ops, stabilizers, cutoff=0, method="brute_force"
    )
    with mock.patch(
        "qldpc.codes.distance._brute_force_is_cheaper", return_value=False
    ) as brute_force_is_cheaper:
        assert qldpc.codes.get_distance_quantum(logical_ops, stabilizers, cutoff=0) == expected
        assert qldpc.codes.get_distance_quantum(logical_ops, stabilizers, cutoff=6) <= 3
    brute_force_is_cheaper.assert_called()

    with mock.patch("qldpc.codes.distance._brute_force_is_cheaper", return_value=True):
        assert qldpc.codes.get_distance_quantum(logical_ops, stabilizers, cutoff=0) == expected

    identity = np.eye(21, dtype=np.uint8)
    with mock.patch("qldpc.codes.distance._brute_force_is_cheaper", return_value=False):
        assert (
            qldpc.codes.get_distance_quantum(
                identity[-1:],
                identity[:-1],
                homogeneous=True,
                cutoff=0,
                block_size=0,
            )
            == 1
        )
        # Every input row has weight at least two; the weight-one witness appears only after
        # information-set row reduction.
        basis = np.zeros((21, 22), dtype=np.uint8)
        basis[:20, :20] = np.eye(20, dtype=np.uint8)
        basis[:, 20:] = 1
        assert (
            qldpc.codes.distance._get_distance_brouwer_zimmermann(
                basis, None, cutoff=1, block_size=0
            )
            == 1
        )

    assert not qldpc.codes.distance._brute_force_is_cheaper(
        dimension=20,
        num_logical_rows=20,
        ranks=[20, 20],
        upper_bound=3,
        weight_divisor=2,
    )


def test_brouwer_zimmermann_invariance() -> None:
    """Row operations and coordinate permutations preserve BZ distance."""
    rng = np.random.default_rng(811)
    generators = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    expected = qldpc.codes.get_distance_classical(generators, cutoff=0)

    transformed = generators.copy()
    transformed[0] ^= transformed[1]
    transformed = transformed[rng.permutation(len(transformed))]
    transformed = transformed[:, rng.permutation(transformed.shape[1])]
    assert qldpc.codes.get_distance_classical(transformed, cutoff=0) == expected


def test_brouwer_zimmermann_special_inputs() -> None:
    """BZ validates matrix shapes and preserves empty/invalid quotient behavior."""
    assert qldpc.codes.get_distance_classical(np.empty((0, 4), dtype=np.uint8), cutoff=0) == 4
    assert (
        qldpc.codes.get_distance_quantum(
            np.empty((0, 4), dtype=np.uint8),
            np.empty((0, 4), dtype=np.uint8),
            homogeneous=True,
        )
        == 4
    )
    assert qldpc.codes.get_distance_quantum([[1, 0]], [[1, 0]], homogeneous=True) == 0

    with pytest.raises(ValueError, match="incompatible widths"):
        qldpc.codes.get_distance_quantum([[1, 0, 0]], [[1, 0]], homogeneous=True)
    with pytest.raises(ValueError, match="even number of columns"):
        qldpc.codes.get_distance_quantum([[1, 0, 0]], [], homogeneous=False)
    with pytest.raises(ValueError, match="even number of columns"):
        qldpc.codes.get_distance_quantum(
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint8),
            homogeneous=False,
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        qldpc.codes.get_distance_classical(np.zeros((1, 1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="infer the width"):
        qldpc.codes.get_distance_classical([])


def test_brouwer_zimmermann_defensive_fallback() -> None:
    """BZ returns its witness if no information sets are supplied."""
    basis = np.eye(16, dtype=np.uint8)
    with mock.patch("qldpc.codes.distance._get_information_set_generators", return_value=[]):
        assert (
            qldpc.codes.distance._get_distance_brouwer_zimmermann(
                basis,
                None,
                cutoff=0,
                block_size=1,
            )
            == 1
        )


def test_brouwer_zimmermann_reduces_enumeration_work() -> None:
    """BZ evaluates fewer candidates than exhaustive enumeration on a Hamming code."""
    generators = qldpc.codes.HammingCode(5).generator
    counts: dict[str, int] = {}

    def run(method: qldpc.codes.DistanceMethod) -> int:
        count = 0

        def counting_weight(
            arr: npt.NDArray[np.uint64],
            buf: npt.NDArray[np.uint64] | None = None,
            out: npt.NDArray[np.uint64] | None = None,
        ) -> npt.NDArray[np.uint64]:
            nonlocal count
            count += len(arr)
            return qldpc.codes.distance._hamming_weight(arr, buf=buf, out=out)

        with mock.patch(
            "qldpc.codes.distance._get_hamming_weight_fn",
            return_value=(counting_weight, 1),
        ):
            distance = qldpc.codes.get_distance_classical(
                generators, cutoff=0, block_size=15, method=method
            )
        counts[method] = count
        return distance

    assert run("brouwer_zimmermann") == run("brute_force") == 3
    assert counts["brouwer_zimmermann"] < counts["brute_force"]


@pytest.mark.parametrize("block_size", range(1, 14))
def test_get_distance_classical(block_size: int) -> None:
    """get_distance_classical visits every nontrivial XOR combination of generators exactly once."""
    generators = np.random.randint(2, size=(9, 137))

    # Intercept `hamming_weight` calls to check that every nontrivial combination of generators
    # is observed exactly once
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_hamming_weight(
        arr: npt.NDArray[np.uint64],
        buf: npt.NDArray[np.uint64] | None = None,
        out: npt.NDArray[np.uint64] | None = None,
    ) -> npt.NDArray[np.uint64]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._hamming_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance._get_hamming_weight_fn", return_value=(_mock_hamming_weight, 0)
    ):
        distance = qldpc.codes.distance.get_distance_classical(
            generators, block_size=block_size, method="brute_force"
        )

    int_generators = qldpc.codes.distance._rows_to_ints(generators)
    expected_bitstrings = [
        tuple(np.bitwise_xor.reduce(np.vstack(gens)).tolist())
        for n in range(1, len(generators) + 1)
        for gens in itertools.combinations(int_generators, n)
    ]

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(expected_bitstrings)

    observed_array = np.array(observed_bitstrings, dtype=np.uint64)
    expected_distance = _bitwise_count(observed_array).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("block_size", range(1, 14))
def test_get_distance_quantum(block_size: int) -> None:
    """get_distance_quantum visits every stabilizer + logical-op combination exactly once.

    Uses Hamming (homogeneous) weight mode.
    """
    stabilizers = np.random.randint(2, size=(8, 97))
    logical_ops = np.random.randint(2, size=(5, 97))

    # Intercept `hamming_weight` calls to check that every combination of stabilizers and at least
    # one logical op is observed exactly once
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_hamming_weight(
        arr: npt.NDArray[np.uint64],
        buf: npt.NDArray[np.uint64] | None = None,
        out: npt.NDArray[np.uint64] | None = None,
    ) -> npt.NDArray[np.uint64]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._hamming_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance._get_hamming_weight_fn", return_value=(_mock_hamming_weight, 0)
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops,
            stabilizers,
            homogeneous=True,
            block_size=block_size,
            method="brute_force",
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

    observed_array = np.array(observed_bitstrings, dtype=np.uint64)
    expected_distance = _bitwise_count(observed_array).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("block_size", range(1, 14))
def test_get_distance_quantum_symplectic(block_size: int) -> None:
    """get_distance_quantum visits every stabilizer + logical-op combination exactly once.

    Uses symplectic weight mode.
    """
    stabilizers = np.random.randint(2, size=(3, 98))
    logical_ops = np.random.randint(2, size=(5, 98))

    # Intercept `symplectic_weight` calls to check that every combination of stabilizers and at
    # least one logical op is observed exactly once
    observed_bitstrings: list[tuple[int, ...]] = []

    def _mock_symplectic_weight(
        arr: npt.NDArray[np.uint64],
        buf: npt.NDArray[np.uint64] | None = None,
        out: npt.NDArray[np.uint64] | None = None,
    ) -> npt.NDArray[np.uint64]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return qldpc.codes.distance._symplectic_weight(arr, buf=buf, out=out)

    with mock.patch(
        "qldpc.codes.distance._get_symplectic_weight_fn", return_value=(_mock_symplectic_weight, 0)
    ):
        distance = qldpc.codes.distance.get_distance_quantum(
            logical_ops,
            stabilizers,
            homogeneous=False,
            block_size=block_size,
            method="brute_force",
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
    """get_distance_classical dispatches to NumPy bitcount or the fallback as available."""
    generators = np.random.randint(2, size=(6, 56), dtype=np.uint64)
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


def test_distance_backend_validation() -> None:
    """Distance backend and exact-method selectors are restricted to documented values."""
    for backend in ["auto", "decoder", "gap", "sqetch"]:
        qldpc.codes.validate_distance_backend(backend)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown distance backend"):
        qldpc.codes.validate_distance_backend("other")  # type: ignore[arg-type]

    for method in ["brouwer_zimmermann", "brute_force"]:
        qldpc.codes.validate_distance_method(method)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown distance method"):
        qldpc.codes.validate_distance_method("other")  # type: ignore[arg-type]


def test_get_distance_empty_stabilizers_symplectic() -> None:
    """A symplectic-weight distance with no stabilizers returns the min logical operator weight.

    With an empty ``stabilizers`` argument there is nothing to interleave, and the X/Z riffling
    still handles the empty input.
    """
    # symplectic vector [1, 0 | 0, 0] is an X on qubit 0, of symplectic weight 1
    distance = qldpc.codes.distance.get_distance_quantum(
        [[1, 0, 0, 0]], [], homogeneous=False, method="brute_force"
    )
    assert distance == 1


def test_get_distance_small_block_size() -> None:
    """A block_size smaller than the packed word count still yields the correct distance.

    When the columns span more than ``64 * (block_size + 1)`` bits, no operators are vectorized and
    the calculation proceeds by the sequential sweep alone.
    """
    generators = np.zeros((2, 200), dtype=np.uint64)
    generators[0] = 1  # weight 200
    generators[1, :100] = 1  # weight 100
    assert qldpc.codes.distance.get_distance_classical(generators, block_size=1) == 100


def test_cutoff_early_exit() -> None:
    """cutoff makes get_distance_* return early once an operator of weight <= cutoff is found.

    Exercises all three early-exit return paths: the pre-loop vectorized block, the logical-op
    sweep, and the nested stabilizer sweep.
    """
    # pre-loop block: a weight-1 codeword sits in the vectorized block (default block_size)
    assert (
        qldpc.codes.distance.get_distance_classical([[1, 0, 0, 0]], cutoff=1, method="brute_force")
        == 1
    )

    # logical-op sweep: block_size=1 spills logical ops into the sequential Gray-code loop, where a
    # combination of weight <= cutoff (here the weight-1 generator) is found.  These rows are
    # linearly independent, as the documented precondition requires.
    generators = [[1, 1, 1, 0], [0, 0, 0, 1], [1, 0, 0, 1]]
    assert (
        qldpc.codes.distance.get_distance_classical(
            generators, block_size=1, cutoff=1, method="brute_force"
        )
        == 1
    )

    # stabilizer sweep: the weight-1 operator only appears after XORing a swept stabilizer, so the
    # early exit fires in the inner (stabilizer) loop rather than the pre-loop or logical-op sweep
    logical_ops = [[0, 1, 1, 1, 1]]
    stabilizers = [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1]]
    distance = qldpc.codes.distance.get_distance_quantum(
        logical_ops,
        stabilizers,
        block_size=1,
        cutoff=1,
        homogeneous=True,
        method="brute_force",
    )
    assert distance == 1


def test_get_distance_quantum_methods() -> None:
    """Homogeneous get_distance_quantum dispatches to NumPy bitcount or the fallback."""
    stabilizers = np.random.randint(2, size=(4, 56), dtype=np.uint64)
    logical_ops = np.random.randint(2, size=(3, 56), dtype=np.uint64)
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
            logical_ops, stabilizers, block_size=3, homogeneous=True
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
            logical_ops, stabilizers, block_size=3, homogeneous=True
        )
        fallback.assert_called()
        assert distance == distance_default


def test_get_distance_quantum_methods_symplectic() -> None:
    """Symplectic get_distance_quantum dispatches to NumPy bitcount or the fallback."""
    stabilizers = np.random.randint(2, size=(4, 56), dtype=np.uint64)
    logical_ops = np.random.randint(2, size=(3, 56), dtype=np.uint64)
    distance_default = qldpc.codes.distance.get_distance_quantum(
        logical_ops,
        stabilizers,
        block_size=3,
        homogeneous=False,
        method="brute_force",
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
            logical_ops,
            stabilizers,
            block_size=3,
            homogeneous=False,
            method="brute_force",
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
            logical_ops,
            stabilizers,
            block_size=3,
            homogeneous=False,
            method="brute_force",
        )
        fallback.assert_called()
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
    """Classical distance matches known values for Hamming, repetition, and ring codes."""
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
    """Quantum distance matches known values for C4, C6, Steane, and small surface/toric codes."""
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

    # forget_distance clears the X and Z caches as well, which assigning to _distance would not:
    # get_distance_if_known(None) returns min(_distance_x, _distance_z) whenever both are known,
    # so the recomputation below would otherwise read back a value it never verified
    code.forget_distance()
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
    """Quantum distance matches known values for small non-CSS codes."""
    distance = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        homogeneous=False,
    )
    assert distance == expected_distance

    code._distance = None
    assert code.get_distance_exact() == expected_distance


def test_distance_requires_binary_input() -> None:
    """Distance calculations reject non-binary input rather than silently truncating it.

    The enumeration packs each row into the bits of uint64 words, so a non-binary entry would
    otherwise be reinterpreted and yield a wrong distance with no indication of a problem.
    """
    # a field of order > 2 is rejected on the strength of its type alone
    ternary_code = qldpc.codes.classical.HammingCode(3, 3)
    with pytest.raises(ValueError, match=r"only support binary codes.*GF\(3\)"):
        qldpc.codes.distance.get_distance_classical(ternary_code.generator)

    # an untyped array is rejected on the strength of its entries
    with pytest.raises(ValueError, match="entries other than 0 and 1"):
        qldpc.codes.distance.get_distance_classical(np.array([[1, 2, 0], [0, 1, 1]]))
    with pytest.raises(ValueError, match="entries other than 0 and 1"):
        qldpc.codes.distance.get_distance_quantum(
            np.array([[1, 1, 0, 0]]), np.array([[0, 0, 2, 0]]), homogeneous=True
        )

    # binary input still works, whether or not it carries a field type
    assert qldpc.codes.distance.get_distance_classical(np.array([[1, 1, 1]])) == 3
    assert qldpc.codes.distance.get_distance_classical(qldpc.codes.HammingCode(4).generator) == 3
