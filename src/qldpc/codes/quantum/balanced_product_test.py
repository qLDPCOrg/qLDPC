# SPDX-License-Identifier: Apache-2.0

"""Unit tests for balanced_product.py."""

from __future__ import annotations

import unittest.mock

import numpy as np
import pytest

from qldpc import abstract, codes

from ..conftest import tanner_graphs_are_isomorphic


def test_balanced_product_code() -> None:
    """Balanced product codes."""
    # without symmetries, a balanced product code is a hypergraph product code
    code_a = codes.ClassicalCode([[1, 1, 0], [0, 1, 1]], field=3)
    code_b = codes.ClassicalCode([[1, 0], [1, 2], [0, 1]], field=3)
    code = codes.BPCode(code_a, code_b)
    hgp_code = codes.HGPCode(code_a, code_b)
    assert np.array_equal(code.matrix_x, hgp_code.matrix_x)
    assert np.array_equal(code.matrix_z, hgp_code.matrix_z)

    # with symmetries that generate a group acting freely, a balanced product code is a lifted
    # product code: pairing a cyclic shift within each block of one lifted matrix with the inverse
    # shift in the other gives the lifted product of the two matrices, up to the order of qudits and
    # checks; over GF(3), pairing equal shifts here would give an inequivalent code
    group = abstract.CyclicGroup(3)
    ring = abstract.GroupRing(group, field=3)
    one, shift, shift_squared = (abstract.RingMember(ring, member) for member in group.generate())
    matrix = abstract.RingArray.build([[one + 2 * shift, shift_squared]], ring)
    symmetry = ([1, 2, 0], [1, 2, 0, 4, 5, 3])
    # the inverse shift as one permutation of the 3 checks followed by the 6 bits
    inverse_symmetry = abstract.GroupMember([2, 0, 1, 5, 3, 4, 8, 6, 7])
    code = codes.BPCode(matrix.lift(), matrix.lift(), [symmetry], [inverse_symmetry])
    assert tanner_graphs_are_isomorphic(code, codes.LPCode(matrix, matrix))

    # over GF(2), from_seed constructs the code of Eq. (3) in Definition 1 of arXiv:2505.13679v2, up
    # to qudit order, even if the symmetry of the seed matrix has orbits of different lengths, which
    # Definition 1 excludes; the seed can also be given as a code, and its symmetry as one
    # permutation of rows followed by columns
    ring_seed = codes.RingCode(6).matrix.view(np.ndarray)
    seed = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
    for seed_matrix, (p_rows, p_cols), as_objects in [
        (ring_seed, ([2, 3, 4, 5, 0, 1], [2, 3, 4, 5, 0, 1]), False),
        (seed, ([1, 0, 2], [1, 0, 2, 3]), True),
    ]:
        num_checks, num_bits = seed_matrix.shape
        mat_rows = abstract.GroupMember(p_rows).to_matrix()
        mat_cols = abstract.GroupMember(p_cols).to_matrix().T
        if as_objects:
            member = abstract.GroupMember(p_rows) @ abstract.GroupMember(p_cols)
            code = codes.BPCode.from_seed(codes.ClassicalCode(seed_matrix), member)
        else:
            code = codes.BPCode.from_seed(seed_matrix, (p_rows, p_cols))
        matrix_x = np.hstack([seed_matrix.T, np.eye(num_bits, dtype=int) + mat_cols]) % 2
        matrix_z = np.hstack([np.eye(num_checks, dtype=int) + mat_rows, seed_matrix]) % 2
        qudit_order = [*range(num_bits, num_bits + num_checks), *p_cols]
        assert np.array_equal(
            code.matrix_x[:, qudit_order].row_space(), code.field(matrix_x).row_space()
        )
        assert np.array_equal(
            code.matrix_z[:, qudit_order].row_space(), code.field(matrix_z).row_space()
        )

    with pytest.raises(ValueError, match="not a symmetry of the seed matrix"):
        codes.BPCode.from_seed(seed, ([1, 0, 2], [0, 1, 2, 3]))


def test_balanced_product_seed_symmetries() -> None:
    """Find the symmetry of the [[12, 2, 3]] code in Example 1 of arXiv:2505.13679v2."""
    # mock the optional pybliss boundary with generators of the dihedral symmetries of a 6-cycle
    seed = codes.RingCode(6)
    generators = [[0, 5, 4, 3, 2, 1, 7, 6, 11, 10, 9, 8], [1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 6]]
    with unittest.mock.patch(
        "qldpc.external._bliss.get_matrix_automorphism_generators", return_value=generators
    ):
        [symmetry] = codes.BPCode.get_seed_symmetries(seed, 3)
        assert codes.BPCode.get_seed_symmetries(seed, 5) == []
    assert symmetry.order() == 3
    assert codes.BPCode.from_seed(seed, symmetry).get_code_params() == (12, 2, 3)
