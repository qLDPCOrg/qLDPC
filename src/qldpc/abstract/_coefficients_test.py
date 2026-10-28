# SPDX-License-Identifier: Apache-2.0

"""Unit tests for _coefficients.py."""

from __future__ import annotations

import functools

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import abstract
from qldpc.abstract import _coefficients


def assert_coefficient_matmul(
    matrix_a: abstract.RingArray | npt.NDArray[np.int_],
    matrix_b: abstract.RingArray | npt.NDArray[np.int_],
) -> None:
    """Assert that the coefficient-array kernel computes a product with RingMember arithmetic."""
    ring = next(mm.ring for mm in (matrix_a, matrix_b) if isinstance(mm, abstract.RingArray))
    product = _coefficients.coefficient_matmul(matrix_a, matrix_b, ring=ring)
    expected = np.matmul(np.asarray(matrix_a), np.asarray(matrix_b))  # RingMember arithmetic
    if isinstance(expected, abstract.RingMember):
        assert product == expected
    else:
        assert isinstance(product, abstract.RingArray) and np.array_equal(product, expected)


@pytest.fixture
def always_use_coefficients(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiply RingArrays with coefficient arrays, however small the product."""
    monkeypatch.setattr(_coefficients, "_MIN_MATMUL_COEFFICIENT_WORK", 0)


@pytest.mark.usefixtures("always_use_coefficients")
def test_coefficient_matmul(ring: abstract.GroupRing) -> None:
    """The coefficient-array kernel agrees with RingMember arithmetic."""
    rng = np.random.default_rng(0)

    def random_matrix(*shape: int) -> abstract.RingArray:
        coefficients = ring.field.Random((*shape, ring.group.order), seed=rng)
        return abstract.RingArray.from_field_array(coefficients, ring)

    # vector and batched products, following the shape conventions of numpy.matmul
    for shape_a, shape_b in [
        ((3,), (3,)),
        ((2, 3), (3,)),
        ((3,), (3, 2)),
        ((2, 1, 2, 3), (3, 3, 2)),
    ]:
        assert_coefficient_matmul(random_matrix(*shape_a), random_matrix(*shape_b))

    # the kernel loops over group members in whichever operand has fewer of them
    dense = random_matrix(3, 3)
    monomial = abstract.RingArray.build(np.full((3, 3), ring.generators[0], dtype=object), ring)
    assert_coefficient_matmul(dense, monomial)
    assert_coefficient_matmul(monomial, dense)

    # integers and elements of the base field are scalars in the ring
    integers = np.array([[0, 1, 2], [2, -1, 0], [1, 0, 1]])
    assert_coefficient_matmul(dense, integers)
    assert_coefficient_matmul(integers, dense)
    assert_coefficient_matmul(dense, ring.field.Random((3, 3), seed=rng))


def test_coefficient_matmul_group_orderings() -> None:
    """Equal rings may enumerate the members of their groups in different orders."""
    group = abstract.CyclicGroup(5)
    members = list(group.generate())
    reversed_group = abstract.Group(*group.generators, generate_func=lambda: iter(members[::-1]))
    ring = abstract.GroupRing(group)
    reversed_ring = abstract.GroupRing(reversed_group)
    assert ring == reversed_ring

    matrix = abstract.RingArray.build(np.resize(np.array(members, dtype=object), (4, 4)), ring)
    reversed_matrix = abstract.RingArray.build(matrix, reversed_ring)
    assert_coefficient_matmul(matrix, reversed_matrix)
    assert_coefficient_matmul(reversed_matrix, matrix)


def test_coefficient_matmul_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The coefficient-array kernel declines products that it does not handle."""
    ring = abstract.GroupRing(abstract.CyclicGroup(7), field=4)
    matrix = abstract.RingArray.build(np.eye(4, dtype=int), ring)
    kernel = functools.partial(_coefficients.coefficient_matmul, ring=ring)

    # operands without coefficients in the ring, or with shapes that numpy.matmul rejects
    assert kernel(matrix, np.eye(4)) is None
    assert kernel(matrix, galois.GF(2)(np.eye(4, dtype=int))) is None
    assert kernel(matrix[0, 0, ...], matrix) is None
    assert kernel(matrix, matrix[:3]) is None
    assert kernel(np.stack([matrix] * 2), np.stack([matrix] * 3)) is None

    # products that are too small, or too large
    work = 4 * 4 * 4 * ring.group.order  # rows * inner * columns * group order
    monkeypatch.setattr(_coefficients, "_MIN_MATMUL_COEFFICIENT_WORK", work + 1)
    assert kernel(matrix, matrix) is None
    monkeypatch.setattr(_coefficients, "_MIN_MATMUL_COEFFICIENT_WORK", work)
    assert kernel(matrix, matrix) is not None
    monkeypatch.setattr(_coefficients, "_MAX_MATMUL_COEFFICIENT_BYTES", 0)
    assert kernel(matrix, matrix) is None
