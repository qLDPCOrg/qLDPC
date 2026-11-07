# SPDX-License-Identifier: Apache-2.0

"""Unit tests for self_orthogonal.py."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc import codes

from ..conftest import assert_cached_parameters_are_genuine


def test_many_hypercube_code() -> None:
    """Goto's many-hypercube code requires a positive level."""
    with pytest.raises(ValueError, match="level of at least 1"):
        codes.ManyHypercubeCode(0)


def test_reed_muller_css_codes() -> None:
    """Self-orthogonal CSS(RM(r, m), RM(r, m)) codes."""
    for (order, size), params in [
        ((0, 2), (4, 2, 2)),
        ((1, 4), (16, 6, 4)),
        ((1, 5), (32, 20, 4)),
        ((2, 6), (64, 20, 8)),
        ((2, 7), (128, 70, 8)),
        ((3, 8), (256, 70, 16)),
        ((2, 10), (1024, 912, 8)),
    ]:
        code = codes.QuantumReedMullerCode(order, size)
        assert (code.order, code.size) == (order, size)
        assert code.get_code_params() == params
        assert not code.is_subsystem_code
        # self-orthogonality: H_x @ H_z.T == 0 on GF(2)
        hx = code.code_x.matrix
        hz = code.code_z.matrix
        assert np.array_equal(hx @ hz.T, np.zeros((hx.shape[0], hz.shape[0]), dtype=int))


def test_reed_muller_css_invalid_params() -> None:
    """Invalid parameters for self-orthogonal CSS(RM(r, m), RM(r, m)) codes."""
    for order, size in [(2, 5), (-1, 4), (5, 4)]:
        with pytest.raises(ValueError, match="self-orthogonal"):
            codes.QuantumReedMullerCode(order, size)


def test_reed_muller_css_brute_force_agreement() -> None:
    """Closed-form distances agree with brute-force enumeration on small codes."""
    for (order, size), dist in [((0, 2), 2), ((1, 4), 4)]:
        code = codes.QuantumReedMullerCode(order, size)
        assert code.get_distance() == dist
        assert codes.CSSCode(code.matrix_x, code.matrix_z).get_distance() == dist


def test_iceberg_code_requires_an_even_length() -> None:
    """Iceberg codes are only defined for even block lengths."""
    with pytest.raises(ValueError, match="even block lengths"):
        codes.IcebergCode(3)


def test_cached_parameters_are_genuine() -> None:
    """Codes that cache their parameters agree with a computation that ignores the cache."""
    assert_cached_parameters_are_genuine(
        [
            (codes.IcebergCode(4), (4, 2, 2)),
            (codes.IcebergCode(6), (6, 4, 2)),
            (codes.IcebergCode(8), (8, 6, 2)),
            (codes.QuantumHammingCode(3), (7, 1, 3)),
            (codes.QuantumHammingCode(4), (15, 7, 3)),
            (codes.ManyHypercubeCode(1), (6, 4, 2)),
            (codes.ManyHypercubeCode(2), (36, 16, 4)),
        ]
    )
