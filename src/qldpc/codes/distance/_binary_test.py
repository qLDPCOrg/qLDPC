# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the binary input checks, nested bases, and weight helpers of distance searches."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import qldpc

from . import _binary as binary
from .conftest import count_set_bits

####################################################################################################
# binary matrices, weights, and array helpers


def test_binary_matrices_and_nested_bases() -> None:
    """Check matrix conversion and basis ownership independently of distance dispatch."""
    assert binary.as_binary_matrix(np.empty((0, 3))).shape == (0, 3)
    assert binary.as_binary_matrix([], num_cols=3).shape == (0, 3)
    with pytest.raises(ValueError, match="infer the width"):
        binary.as_binary_matrix([])
    with pytest.raises(ValueError, match="two-dimensional"):
        binary.as_binary_matrix(np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="incompatible widths"):
        binary.as_binary_matrix([[1, 0]], num_cols=3)

    matrix = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.uint8)
    basis, tags = binary.get_nested_code_basis(matrix, np.array([[1, 0, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(basis, [[1, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(tags, [[0], [1]])


def test_eligible_rows_have_nonzero_labels() -> None:
    """Rows with nonzero labels are eligible, and labels with no columns exclude no row."""
    labels = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    assert binary.get_eligible_rows(labels).tolist() == [False, True, True, True]
    assert binary.get_eligible_rows(np.empty((3, 0), dtype=np.uint8)).tolist() == [True] * 3


def test_riffle_and_symplectic_isometry() -> None:
    symplectic = np.array(list(itertools.product((0, 1), repeat=6)), dtype=np.uint8)
    expected = np.stack((symplectic[:, :3], symplectic[:, 3:]), axis=2).reshape(-1, 6)
    np.testing.assert_array_equal(binary.riffle(symplectic), expected)
    assert np.shape(binary.riffle(np.empty((0, 6)))) == (0, 6)
    with pytest.raises(AssertionError):
        binary.riffle(np.zeros((1, 3)))
    transformed = binary.symplectic_to_hamming(symplectic)
    np.testing.assert_array_equal(
        np.count_nonzero(transformed, axis=1),
        2 * np.count_nonzero(symplectic[:, :3] | symplectic[:, 3:], axis=1),
    )


def test_symplectic_weight() -> None:
    """Validate symplectic_weight against the reference bit-counting implementation."""
    rng = np.random.default_rng(64)
    vals = rng.integers(0, 2**64, size=(7, 11), dtype=np.uint64)
    expected_weights = count_set_bits((vals | (vals >> np.uint64(1))) & 0x5555555555555555)
    weights = binary.symplectic_weight(vals)
    assert weights.dtype == np.uint64
    np.testing.assert_array_equal(weights, expected_weights)

    out = rng.integers(0, 2**64, size=vals.shape, dtype=vals.dtype)
    weights = binary.symplectic_weight(vals, out=out)
    np.testing.assert_array_equal(weights, expected_weights)
    assert weights is out


@pytest.mark.parametrize("width", range(1, 8))
def test_inplace_rowsum(width: int) -> None:
    """inplace_rowsum reduces each row to its sum and stores the result in the first column."""
    arr = np.random.default_rng(width).integers(2**31, size=(10, width), dtype=np.uint64)
    expected = arr.sum(-1)
    actual = binary.inplace_rowsum(arr)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, arr[:, 0])


def test_distance_requires_binary_input() -> None:
    """Distance calculations reject non-binary input rather than silently truncating it.

    The enumeration packs each row into the bits of uint64 words, so a non-binary entry would
    otherwise be reinterpreted and yield a wrong distance with no indication of a problem.
    """
    # A GF(3) generator matrix is rejected because exact distance enumeration supports only GF(2).
    ternary_code = qldpc.codes.classical.HammingCode(3, 3)
    with pytest.raises(ValueError, match=r"only support binary codes.*GF\(3\)"):
        qldpc.codes.distance.get_distance_classical(ternary_code.generator)

    # A plain integer array with a nonbinary entry is rejected by value validation.
    with pytest.raises(ValueError, match="entries other than 0 and 1"):
        qldpc.codes.distance.get_distance_classical(np.array([[1, 2, 0], [0, 1, 1]]))
    with pytest.raises(ValueError, match="entries other than 0 and 1"):
        qldpc.codes.distance.get_distance_quantum(
            np.array([[1, 1, 0, 0]]), np.array([[0, 0, 2, 0]]), homogeneous=True
        )

    # A binary generator matrix is accepted with either a plain integer dtype or a GF(2) dtype.
    assert qldpc.codes.distance.get_distance_classical(np.array([[1, 1, 1]])) == 3
    assert qldpc.codes.distance.get_distance_classical(qldpc.codes.HammingCode(4).generator) == 3
