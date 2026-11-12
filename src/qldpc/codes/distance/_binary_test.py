# SPDX-License-Identifier: Apache-2.0

"""Tests of the input checks, nested bases, and symplectic-to-Hamming map of distance searches."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

import qldpc

from . import _binary as binary

####################################################################################################
# binary matrices


def test_binary_matrices_and_nested_bases() -> None:
    """Check matrix conversion and basis ownership independently of distance dispatch."""
    assert binary.as_binary_matrix(np.empty((0, 3))).shape == (0, 3)
    assert binary.as_binary_matrix([], num_cols=3).shape == (0, 3)
    with pytest.raises(ValueError, match="infer the width"):
        binary.as_binary_matrix([])
    with pytest.raises(ValueError, match="The rows must be a two-dimensional"):
        binary.as_binary_matrix(np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="The stabilizers have 2 columns"):
        binary.as_binary_matrix([[1, 0]], "stabilizers", num_cols=3)

    matrix = np.array([[0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 0, 0]], dtype=np.uint8)
    basis, tags = binary.get_nested_code_basis(matrix, np.array([[1, 0, 1]], dtype=np.uint8))
    np.testing.assert_array_equal(basis, [[1, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(tags, [[0], [1]])


def test_integer_arguments() -> None:
    """Integers of any type become Python integers, and bools and other values are rejected."""
    value = binary.as_integer(np.int64(3), "value")
    assert value == 3 and type(value) is int
    for invalid in (True, 2.5, "3"):
        not_an_integer: Any = invalid
        with pytest.raises(TypeError, match="value must be an integer"):
            binary.as_integer(not_an_integer, "value")


def test_eligible_rows_have_nonzero_labels() -> None:
    """Rows with nonzero labels are eligible, and labels with no columns exclude no row."""
    labels = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.uint8)
    assert binary.get_eligible_rows(labels).tolist() == [False, True, True, True]
    assert binary.get_eligible_rows(np.empty((3, 0), dtype=np.uint8)).tolist() == [True] * 3


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


def test_symplectic_to_hamming_doubles_weights() -> None:
    symplectic = np.array(list(itertools.product((0, 1), repeat=6)), dtype=np.uint8)
    transformed = binary.symplectic_to_hamming(symplectic)
    np.testing.assert_array_equal(
        np.count_nonzero(transformed, axis=1),
        2 * np.count_nonzero(symplectic[:, :3] | symplectic[:, 3:], axis=1),
    )
