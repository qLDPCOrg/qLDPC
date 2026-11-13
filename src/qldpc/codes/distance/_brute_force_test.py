# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact exhaustive binary distance enumeration."""

from __future__ import annotations

import itertools
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest

import qldpc
from qldpc._binary import pack_bits

from ..conftest import Tune
from . import _brute_force
from .conftest import count_set_bits


@pytest.mark.parametrize("batch_exponent", [1, 3, 13])
def test_brute_force_visits_every_code_word_once(batch_exponent: int, tune: Tune) -> None:
    """Exhaustive enumeration visits every nontrivial XOR combination of generators exactly once."""
    tune(enumeration_batch_exponent=batch_exponent)
    generators = np.random.default_rng(137).integers(2, size=(9, 137))

    # The mocked popcount records every packed nonzero generator combination once.
    observed_bitstrings: list[tuple[int, ...]] = []

    bitwise_count = np.bitwise_count

    def _mock_bitwise_count(
        arr: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64] | None = None
    ) -> npt.NDArray[np.uint64]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return bitwise_count(arr, out=out)

    with mock.patch.object(np, "bitwise_count", _mock_bitwise_count):
        distance = qldpc.codes.distance.get_distance_from_generators(
            np.vstack([generators, generators[0], generators[0] ^ generators[1]]),
            method="brute_force",
        )

    int_generators = pack_bits(np.asarray(generators, dtype=np.uint8))
    expected_bitstrings = [
        tuple(np.bitwise_xor.reduce(np.vstack(gens)).tolist())
        for n in range(1, len(generators) + 1)
        for gens in itertools.combinations(int_generators, n)
    ]

    assert len(observed_bitstrings) == len(expected_bitstrings)
    assert set(observed_bitstrings) == set(expected_bitstrings)

    observed_array = np.array(observed_bitstrings, dtype=np.uint64)
    expected_distance = count_set_bits(observed_array).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("symplectic", [False, True], ids=["hamming", "symplectic"])
@pytest.mark.parametrize("batch_exponent", [1, 3, 7])
def test_brute_force_visits_every_nontrivial_word_once(
    symplectic: bool, batch_exponent: int, tune: Tune
) -> None:
    """Exhaustive enumeration visits every combination of stabilizers and logical operators once.

    The spy records each enumerated word through its weight function: ``np.bitwise_count`` for
    Hamming weights, and ``symplectic_weight`` for symplectic weights.

    A batch holds at most 2**batch_exponent 64-bit integers, and each word packs into two, so a
    batch vectorizes over batch_exponent - 1 generators.  For Hamming weights, the exponents 1, 3,
    and 7 vectorize over, respectively, no generators, two of the four stabilizers, and all four
    stabilizers and two logical operators.  For symplectic weights, they vectorize over,
    respectively, no generators, two of the three stabilizers, and all three stabilizers and three
    logical operators.
    """
    tune(enumeration_batch_exponent=batch_exponent)
    rng = np.random.default_rng(97)
    num_stabilizers, num_bits = (3, 98) if symplectic else (4, 97)
    stabilizers = rng.integers(2, size=(num_stabilizers, num_bits))
    logical_ops = rng.integers(2, size=(5, num_bits))

    # The weight spy records every nonzero logical combination combined with every stabilizer
    # combination, including the empty one.
    observed_bitstrings: list[tuple[int, ...]] = []
    weight_function = _brute_force.symplectic_weight if symplectic else np.bitwise_count

    def _spy_weight(
        arr: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64] | None = None
    ) -> npt.NDArray[np.uint64]:
        observed_bitstrings.extend(map(tuple, arr.tolist()))
        return weight_function(arr, out=out)

    spy = (
        mock.patch("qldpc.codes.distance._brute_force.symplectic_weight", _spy_weight)
        if symplectic
        else mock.patch.object(np, "bitwise_count", _spy_weight)
    )
    with spy:
        distance = qldpc.codes.distance.get_distance_from_generators(
            np.vstack([logical_ops, logical_ops[0] ^ stabilizers[0]]),
            np.vstack([stabilizers, stabilizers[0]]),
            symplectic=symplectic,
            method="brute_force",
        )

    # Symplectic words are packed with their X and Z bits interleaved.
    stabilizer_bits = _brute_force.riffle(stabilizers) if symplectic else stabilizers
    logical_bits = _brute_force.riffle(logical_ops) if symplectic else logical_ops
    int_stabilizers = pack_bits(np.asarray(stabilizer_bits, dtype=np.uint8))
    int_logical_ops = pack_bits(np.asarray(logical_bits, dtype=np.uint8))
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
    assert set(observed_bitstrings) == set(map(tuple, expected_bitstrings.tolist()))

    if symplectic:
        expected_bitstrings = (expected_bitstrings | expected_bitstrings >> 1) & 0x5555555555555555
    expected_distance = count_set_bits(expected_bitstrings).sum(-1).min()
    assert distance == expected_distance


@pytest.mark.parametrize("num_bits", [129, 300, 1000])
def test_brute_force_batches_fill_their_size(num_bits: int, tune: Tune) -> None:
    """A batch of exhaustive enumeration fills more than half of its packed integers, and no more.

    Each word packs into one 64-bit integer for every 64 bits, so longer words fit fewer to a batch
    of at most 2**batch_exponent integers.  Words of three and five integers, which are not powers
    of two, check that a batch rounds its number of words down, and words of 16 integers check that
    long words still share batches.
    """
    batch_exponent = 9
    tune(enumeration_batch_exponent=batch_exponent)
    rng = np.random.default_rng(num_bits)
    logical_ops = rng.integers(2, size=(4, num_bits))
    stabilizers = rng.integers(2, size=(8, num_bits))

    # The weight spy records the number of packed integers in each batch that it weighs.
    sizes: list[int] = []
    bitwise_count = np.bitwise_count

    def _spy_weight(
        arr: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64] | None = None
    ) -> npt.NDArray[np.uint64]:
        sizes.append(arr.size)
        return bitwise_count(arr, out=out)

    with mock.patch.object(np, "bitwise_count", _spy_weight):
        qldpc.codes.distance.get_distance_from_generators(
            logical_ops, stabilizers, method="brute_force"
        )
    assert 2 ** (batch_exponent - 1) < max(sizes) <= 2**batch_exponent


def test_get_distance_empty_stabilizers_symplectic() -> None:
    """A symplectic-weight distance with no stabilizers returns the min logical operator weight.

    With an empty ``stabilizers`` argument there is nothing to interleave, and the X/Z riffling
    still handles the empty input.
    """
    # The symplectic vector [1, 0 | 0, 0] is an X operator on qubit 0 with symplectic weight 1.
    distance = qldpc.codes.distance.get_distance_from_generators(
        [[1, 0, 0, 0]], [], symplectic=True, method="brute_force"
    )
    assert distance == 1


def test_cutoff_early_exit(tune: Tune) -> None:
    """cutoff makes get_distance_* return early once an operator of weight <= cutoff is found.

    Exercises all three early-exit return paths: the pre-loop vectorized block, the logical-op
    sweep, and the nested stabilizer sweep.
    """
    # The pre-loop vectorized block contains a codeword with Hamming weight 1.
    assert (
        qldpc.codes.distance.get_distance_from_generators(
            [[1, 0, 0, 0]], cutoff=1, method="brute_force"
        )
        == 1
    )

    # The logical-operator sweep with batches of two words finds a generator of Hamming weight 1.
    tune(enumeration_batch_exponent=1)
    generators = [[1, 0, 1], [0, 1, 0]]
    assert (
        qldpc.codes.distance.get_distance_from_generators(
            generators, cutoff=1, method="brute_force"
        )
        == 1
    )

    # The stabilizer sweep finds a Hamming-weight-1 operator only after XORing a stabilizer row.
    logical_ops = [[0, 0, 1, 1, 1]]
    stabilizers = [[1, 0, 0, 1, 0], [0, 1, 1, 1, 1]]
    distance = qldpc.codes.distance.get_distance_from_generators(
        logical_ops,
        stabilizers,
        cutoff=1,
        method="brute_force",
    )
    assert distance == 1


@pytest.mark.parametrize("homogeneous", [True, False], ids=["hamming", "symplectic"])
def test_early_exit_returns_only_weighed_operators(homogeneous: bool) -> None:
    """An early exit returns the weight of an operator that the search has weighed.

    With a batch of one word, the vectorized block holds no logical operator, so the search has
    weighed none before its sweep, and a cutoff at or above the number of columns must not end it
    there.
    """
    num_qubits = 3
    logical_op = [1, 0, 0] if homogeneous else [1, 0, 0, 0, 0, 0]
    stabilizer = [0, 1, 1] if homogeneous else [0, 1, 1, 0, 0, 0]
    distance = _brute_force._get_distance_brute_force(
        [logical_op],
        [stabilizer],
        cutoff=len(logical_op),
        batch_exponent=0,
        homogeneous=homogeneous,
    )
    assert distance == 1

    # Without logical operators, the result exceeds every weight, whatever the cutoff.
    distance = _brute_force._get_distance_brute_force(
        np.empty((0, len(logical_op)), dtype=np.uint8),
        [stabilizer],
        cutoff=100,
        batch_exponent=0,
        homogeneous=homogeneous,
    )
    assert distance == num_qubits + 1


def test_riffle() -> None:
    symplectic = np.array(list(itertools.product((0, 1), repeat=6)), dtype=np.uint8)
    expected = np.stack((symplectic[:, :3], symplectic[:, 3:]), axis=2).reshape(-1, 6)
    np.testing.assert_array_equal(_brute_force.riffle(symplectic), expected)
    assert np.shape(_brute_force.riffle(np.empty((0, 6)))) == (0, 6)
    with pytest.raises(AssertionError):
        _brute_force.riffle(np.zeros((1, 3)))


def test_symplectic_weight() -> None:
    """Validate symplectic_weight against the reference bit-counting implementation."""
    rng = np.random.default_rng(64)
    vals = rng.integers(0, 2**64, size=(7, 11), dtype=np.uint64)
    expected_weights = count_set_bits((vals | (vals >> np.uint64(1))) & 0x5555555555555555)
    weights = _brute_force.symplectic_weight(vals)
    assert weights.dtype == np.uint64
    np.testing.assert_array_equal(weights, expected_weights)

    out = rng.integers(0, 2**64, size=vals.shape, dtype=vals.dtype)
    weights = _brute_force.symplectic_weight(vals, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert weights is out


@pytest.mark.parametrize("width", range(1, 8))
def test_inplace_rowsum(width: int) -> None:
    """inplace_rowsum reduces each row to its sum and stores the result in the first column."""
    arr = np.random.default_rng(width).integers(2**31, size=(10, width), dtype=np.uint64)
    expected = arr.sum(-1)
    actual = _brute_force.inplace_rowsum(arr)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, arr[:, 0])
