# SPDX-License-Identifier: Apache-2.0

"""Unit tests for surface.py."""

from __future__ import annotations

import unittest.mock

import numpy as np
import pytest

from qldpc import abstract, codes, decoders
from qldpc.objects import Pauli

from ..conftest import assert_cached_parameters_are_genuine, assert_valid_subgraphs


def test_surface_codes(rows: int = 3, cols: int = 2) -> None:
    """Ordinary and rotated surface codes."""

    # "ordinary"/original surface code
    code = codes.SurfaceCode(rows, cols, rotated=False)
    plain = codes.CSSCode(code.matrix_x, code.matrix_z)
    assert plain.dimension == 1
    assert code.num_qudits == rows * cols + (rows - 1) * (cols - 1)
    with unittest.mock.patch("qldpc.external.gap.is_callable", return_value=False):
        assert cols <= plain.get_distance(Pauli.X, bound=True) <= len(code)
        assert rows <= plain.get_distance(Pauli.Z, bound=True) <= len(code)
    assert_valid_subgraphs(code)

    # un-rotated SurfaceCode = HGPCode
    rep_codes = (codes.RepetitionCode(rows), codes.RepetitionCode(cols))
    assert code.is_equiv_to(codes.HGPCode(*rep_codes))

    # rotated surface code
    code = codes.SurfaceCode(rows, cols, rotated=True, field=3)
    plain = codes.CSSCode(code.matrix_x, code.matrix_z)
    assert plain.dimension == 1
    assert code.num_qudits == rows * cols
    with pytest.warns(UserWarning, match=r"may take a \(very\) long time"):
        distances = plain.get_distance(Pauli.X), plain.get_distance(Pauli.Z)
    assert distances == (cols, rows)
    assert_valid_subgraphs(code)

    # the bias-tailored rotated surface code is an XZZX code
    code = codes.SurfaceCode(max(rows, cols), rotated=True)
    for row in code.conjugated(code.bias_tailoring_qubits).matrix:
        row_x, row_z = row[: code.num_qudits], row[-code.num_qudits :]
        assert np.count_nonzero(row_x) == np.count_nonzero(row_z)


def test_toric_codes() -> None:
    """Ordinary and rotated toric codes."""

    # "ordinary"/original toric code
    distance = 3
    code = codes.ToricCode(distance, rotated=False)
    assert codes.CSSCode(code.matrix_x, code.matrix_z).dimension == 2
    assert code.num_qudits == 2 * distance**2
    assert_valid_subgraphs(code)

    # check minimal logical operator weights
    code.reduce_logical_ops(decoder=decoders.ilp())
    assert (
        {distance}
        == {sum(op) for op in code.get_logical_ops(Pauli.X).view(np.ndarray)}
        == {sum(op) for op in code.get_logical_ops(Pauli.Z).view(np.ndarray)}
    )

    # rotated toric code
    distance = 4
    code = codes.ToricCode(distance, rotated=True, field=3)
    plain = codes.CSSCode(code.matrix_x, code.matrix_z)
    assert plain.dimension == 2
    assert code.num_qudits == distance**2
    with pytest.warns(UserWarning, match=r"may take a \(very\) long time"):
        assert plain.get_distance() == distance
    assert_valid_subgraphs(code)

    # rotated toric code must have even side lengths
    with pytest.raises(ValueError, match="even side lengths"):
        codes.ToricCode(3, rotated=True)

    # the rotated 2x2 Toric code is special -- we remove redundant checks
    code = codes.ToricCode(2, rotated=True)
    assert len(code.matrix_x) == len(code.matrix_z) == 1

    # the bias-tailored rotated toric code is an XZZX code
    rows, cols = 6, 4
    code = codes.ToricCode(rows, cols, rotated=True)
    for row in code.conjugated(code.bias_tailoring_qubits).matrix:
        row_x, row_z = row[: code.num_qudits], row[-code.num_qudits :]
        assert np.count_nonzero(row_x) == np.count_nonzero(row_z)


def test_generalized_surface_codes(size: int = 3) -> None:
    """Multi-dimensional surface and toric codes."""

    # recover ordinary surface code in 2D
    assert np.array_equal(
        codes.GeneralizedSurfaceCode(size, dim=2, periodic=False).matrix,
        codes.SurfaceCode(size, rotated=False).matrix,
    )

    # recover ordinary toric code in 2D
    assert np.array_equal(
        codes.GeneralizedSurfaceCode(size, dim=2, periodic=True).matrix,
        codes.ToricCode(size, rotated=False).matrix,
    )

    for dim in [3, 4]:
        # surface code
        code = codes.GeneralizedSurfaceCode(size, dim, periodic=False)
        assert code.dimension == 1
        assert code.num_qudits == size**dim + (dim - 1) * size ** (dim - 2) * (size - 1) ** 2

        # toric code
        code = codes.GeneralizedSurfaceCode(size, dim, periodic=True)
        assert code.dimension == dim
        assert code.num_qudits == dim * size**dim

    with pytest.raises(ValueError, match=">= 2"):
        codes.GeneralizedSurfaceCode(size, dim=1)


def test_4d_toric_codes() -> None:
    """Reproduce Table 1 from arXiv:2506.15130v1."""
    t4_codes = {
        ((1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 0), (0, 0, 0, 2)): (12, 6, 2),
        ((1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (0, 0, 0, 3)): (18, 6, 3),
        ((1, 0, 0, 1), (0, 1, 0, 2), (0, 0, 1, 3), (0, 0, 0, 5)): (30, 6, 4),
    }
    for lattice, params in t4_codes.items():
        code = codes.T4Code(lattice)
        assert code.get_code_params() == params

        # check the Betti numbers of the chain complex
        d_1, d_2, d_3 = code.chain.ops
        assert not isinstance(d_1, abstract.RingArray)
        assert not isinstance(d_2, abstract.RingArray)
        assert not isinstance(d_3, abstract.RingArray)

        # the first Betti number of T^4 is 4
        dim_ker_d1 = d_1.shape[1] - np.linalg.matrix_rank(d_1)
        dim_im_d2 = np.linalg.matrix_rank(d_2)
        assert dim_ker_d1 - dim_im_d2 == 4

        # the second Betti number of T^4 is 6
        dim_ker_d2 = d_2.shape[1] - np.linalg.matrix_rank(d_2)
        dim_im_d3 = np.linalg.matrix_rank(d_3)
        assert dim_ker_d2 - dim_im_d3 == 6

    # the "Hadamard lattice code" of arXiv:2506.15130v1
    hadamard = [[1, 1], [1, -1]]
    code = codes.T4Code(np.kron(hadamard, hadamard))
    assert (len(code), code.dimension) == (96, 6)


def test_4d_toric_code_lattices() -> None:
    """A T4Code lattice must tile its torus with more than one cell.

    A unimodular basis leaves a single vertex, for which every boundary operator vanishes and the
    resulting code would have no parity checks whatsoever.
    """
    degenerate_lattices = [
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # determinant 1
        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # a shear
        [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # determinant -1
        [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],  # singular
    ]
    for lattice in degenerate_lattices:
        with pytest.raises(ValueError, match="abs\\(determinant\\) >= 2"):
            codes.T4Code(lattice)


def test_cached_parameters_are_genuine() -> None:
    """Codes that cache their parameters agree with a computation that ignores the cache."""
    assert_cached_parameters_are_genuine(
        [
            (codes.SurfaceCode(3, 5), (15, 1, 3)),
            (codes.ToricCode(4), (16, 2, 4)),
            (codes.GeneralizedSurfaceCode(2, 3), (12, 1, 2)),
        ]
    )
