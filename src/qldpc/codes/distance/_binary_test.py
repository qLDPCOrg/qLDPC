# SPDX-License-Identifier: Apache-2.0

"""Tests of the binary input checks, nested bases, stabilizers, and symplectic map of searches."""

from __future__ import annotations

import itertools
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse
import sympy

import qldpc
from qldpc._binary import get_independent_rows

from . import _binary as binary

####################################################################################################
# binary matrices


def test_binary_matrices_and_nested_bases() -> None:
    """Check matrix conversion and basis ownership independently of distance dispatch."""
    assert binary.as_binary_matrix(np.empty((0, 3))).shape == (0, 3)
    assert binary.as_binary_matrix([], num_cols=3).shape == (0, 3)
    with pytest.raises(ValueError, match="infer the width"):
        binary.as_binary_matrix([])
    with pytest.raises(ValueError, match="rows must be a matrix or a single row"):
        binary.as_binary_matrix(np.zeros((1, 1, 1)))
    with pytest.raises(ValueError, match="stabilizers has 2 columns"):
        binary.as_binary_matrix([[1, 0]], "stabilizers", num_cols=3)
    np.testing.assert_array_equal(binary.as_binary_matrix([1, 0, 1]), [[1, 0, 1]])

    # A matrix of another width, even one without rows, is rejected, but a matrix of shape (0, 0)
    # takes the given width.
    for wrong_width in (np.zeros((4, 0)), np.zeros((0, 2))):
        with pytest.raises(ValueError, match=r"rows has \d columns"):
            binary.as_binary_matrix(wrong_width, num_cols=3)
    assert binary.as_binary_matrix(np.zeros((0, 0)), num_cols=3).shape == (0, 3)

    # Arrays of objects, such as Python integers too large for NumPy, are checked entry by entry,
    # and so is a single number, but a single object that is neither an array nor a number is
    # rejected.
    np.testing.assert_array_equal(
        binary.as_binary_matrix(np.array([[1, 0]], dtype=object)), [[1, 0]]
    )
    for number in (sympy.Integer(1), np.array(np.True_, dtype=object)):
        np.testing.assert_array_equal(binary.as_binary_matrix(cast(Any, number)), [[1]])
    for not_binary in ([[2**70, 0]], [[None, 1]]):
        with pytest.raises(ValueError, match="entries other than 0 and 1"):
            binary.as_binary_matrix(cast(Any, not_binary))
    with pytest.raises(TypeError, match="rows must be an array of binary entries, not an object"):
        binary.as_binary_matrix(cast(Any, {}))

    # A matrix without a width takes the width of the other, in either order, but two empty lists
    # have no width.
    names = ("first", "second")
    first, second = binary.as_binary_matrix_pair([], [[1, 0, 1]], names)
    assert first.shape == (0, 3) and second.shape == (1, 3)
    first, second = binary.as_binary_matrix_pair([[1, 0, 1]], np.zeros((0, 0)), names)
    assert first.shape == (1, 3) and second.shape == (0, 3)
    first, second = binary.as_binary_matrix_pair(np.zeros((0, 0)), [], names)
    assert first.shape == second.shape == (0, 0)
    first, second = binary.as_binary_matrix_pair([], np.zeros((0, 0)), names)
    assert first.shape == second.shape == (0, 0)
    with pytest.raises(ValueError, match="infer the width of first"):
        binary.as_binary_matrix_pair([], [], names)

    sparse = scipy.sparse.csr_array([[1, 0, 1]])
    binary.assert_binary(sparse, "rows")
    np.testing.assert_array_equal(binary.as_binary_matrix(sparse), [[1, 0, 1]])
    with pytest.raises(ValueError, match="entries other than 0 and 1"):
        binary.assert_binary(scipy.sparse.csr_matrix([[2, 0]]), "rows")

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
    with pytest.raises(ValueError, match=r"only take binary input.*GF\(3\)"):
        qldpc.codes.distance.get_distance_from_generators(ternary_code.generator)

    # A plain integer array with a nonbinary entry is rejected by value validation.
    with pytest.raises(ValueError, match="entries other than 0 and 1"):
        qldpc.codes.distance.get_distance_from_generators(np.array([[1, 2, 0], [0, 1, 1]]))
    with pytest.raises(ValueError, match="entries other than 0 and 1"):
        qldpc.codes.distance.get_distance_from_generators(
            np.array([[1, 1, 0, 0]]), np.array([[0, 0, 2, 0]])
        )

    # A binary generator matrix is accepted with either a plain integer dtype or a GF(2) dtype.
    assert qldpc.codes.distance.get_distance_from_generators(np.array([[1, 1, 1]])) == 3
    assert (
        qldpc.codes.distance.get_distance_from_generators(qldpc.codes.HammingCode(4).generator) == 3
    )


def test_binary_products() -> None:
    """Binary products match products of integers, reduced modulo two."""
    rng = np.random.default_rng(0)
    left = rng.integers(2, size=(5, 70), dtype=np.uint8)
    right = rng.integers(2, size=(70, 4), dtype=np.uint8)
    product = binary.multiply_binary(left, right)
    assert product.dtype == np.uint8
    np.testing.assert_array_equal(product, left.astype(int) @ right.astype(int) % 2)


def test_symplectic_to_hamming_doubles_weights() -> None:
    symplectic = np.array(list(itertools.product((0, 1), repeat=6)), dtype=np.uint8)
    transformed = binary.symplectic_to_hamming(symplectic)
    np.testing.assert_array_equal(
        np.count_nonzero(transformed, axis=1),
        2 * np.count_nonzero(symplectic[:, :3] | symplectic[:, 3:], axis=1),
    )


####################################################################################################
# stabilizers of subsystem codes


def test_stabilizers_of_subsystem_codes() -> None:
    """The stabilizers are the gauge operators that commute with every gauge operator.

    For a Bacon-Shor code, the stabilizers of the CSS form and of the Pauli form span the
    combinations of gauge generators that commute with every gauge generator, which the test finds
    by enumerating them.  Commuting gauge generators come back unchanged, with their redundant rows.
    """
    code = qldpc.codes.BaconShorCode(3)
    gauge_x = code.matrix_x.view(np.ndarray).astype(np.uint8)
    gauge_z = code.matrix_z.view(np.ndarray).astype(np.uint8)

    def get_commuting(rows: npt.NDArray[np.uint8], others: npt.NDArray[np.uint8]) -> Any:
        combinations = np.array(list(itertools.product([0, 1], repeat=len(rows))), dtype=int)
        words = combinations @ rows % 2
        return get_independent_rows(words[~np.any(words @ others.T % 2, axis=1)].astype(np.uint8))

    stabilizers_x, stabilizers_z = binary.get_css_stabilizers(gauge_x, gauge_z)
    expected_x = get_commuting(gauge_x, gauge_z)
    expected_z = get_commuting(gauge_z, gauge_x)
    assert 0 < len(expected_x) < len(gauge_x) and 0 < len(expected_z) < len(gauge_z)
    assert len(stabilizers_x) == len(expected_x) and len(stabilizers_z) == len(expected_z)
    np.testing.assert_array_equal(get_independent_rows(stabilizers_x), expected_x)
    np.testing.assert_array_equal(get_independent_rows(stabilizers_z), expected_z)

    zeros_x = np.zeros_like(gauge_x)
    zeros_z = np.zeros_like(gauge_z)
    gauge = np.vstack([np.hstack([gauge_x, zeros_x]), np.hstack([zeros_z, gauge_z])])
    stabilizers = binary.get_pauli_stabilizers(gauge)
    expected = np.vstack(
        [
            np.hstack([expected_x, np.zeros_like(expected_x)]),
            np.hstack([np.zeros_like(expected_z), expected_z]),
        ]
    )
    assert len(stabilizers) == len(expected_x) + len(expected_z)
    np.testing.assert_array_equal(get_independent_rows(stabilizers), get_independent_rows(expected))

    checks = np.vstack([expected_x, expected_x[:1] ^ expected_x[1:2]])
    stabilizers_x, stabilizers_z = binary.get_css_stabilizers(checks, expected_z)
    assert stabilizers_x is checks and stabilizers_z is expected_z
    assert binary.get_pauli_stabilizers(stabilizers) is stabilizers
