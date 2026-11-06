# SPDX-License-Identifier: Apache-2.0

"""Unit tests for named.py."""

from __future__ import annotations

import numpy as np

from qldpc import codes
from qldpc.objects import Pauli

from ..conftest import assert_cached_parameters_are_genuine


def test_hamming_and_tetrahedral_codes() -> None:
    """Quantum Hamming and tetrahedral codes."""
    quantum_hamming_code = codes.QuantumHammingCode(4)
    assert quantum_hamming_code.get_code_params() == (15, 7, 3)

    tetrahedral_code = codes.TetrahedralCode(algebraic=False)
    assert tetrahedral_code.get_code_params() == (15, 1, 3)
    assert tetrahedral_code.is_equiv_to(codes.TetrahedralCode(algebraic=True))

    # the documented permutation maps the geometric checks to Eqs. 2-3 of arXiv:2409.13465v2, whose
    # X and Z conventions are reversed relative to this implementation
    qubit_map = [0, 10, 3, 14, 7, 13, 6, 8, 1, 9, 2, 12, 4, 11, 5]
    paper_faces = {
        frozenset(support)
        for support in [
            [0, 3, 6, 7],
            [3, 6, 10, 13],
            [6, 7, 13, 14],
            [8, 9, 11, 12],
            [1, 2, 4, 5],
            [4, 5, 6, 7],
            [2, 3, 5, 6],
            [4, 5, 11, 12],
            [2, 5, 9, 11],
            [5, 6, 11, 13],
        ]
    }
    paper_cells = {
        frozenset(support)
        for support in [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [2, 3, 5, 6, 9, 10, 11, 13],
            [4, 5, 6, 7, 11, 12, 13, 14],
            [1, 2, 4, 5, 8, 9, 11, 12],
        ]
    }

    def mapped_supports(matrix: np.ndarray) -> set[frozenset[int]]:
        """Supports after mapping this code's qubit order to the paper's."""
        return {frozenset(qubit_map[col] for col in np.flatnonzero(row)) for row in matrix}

    assert mapped_supports(tetrahedral_code.matrix_z) == paper_faces
    assert mapped_supports(tetrahedral_code.matrix_x) == paper_cells

    # The tetrahedral code (TC) can be constructed by concatenating the quantum Hamming code (QHC)
    # with a classical code on the logical X operators of the QHC, as we show below.  To this end,
    # we first decompose the logical X operator of the TC into a product of logical X operators of
    # the QHC, which can be found by checking (anti-)commutation with the logical Zs of the QHC.  We
    # then concatenate the QHC with a classical code that has only this combination of QHC logical
    # Xs as a nontrivial code word.
    decomposition = (
        tetrahedral_code.get_logical_ops(Pauli.X) @ quantum_hamming_code.get_logical_ops(Pauli.Z).T
    )
    code = codes.CSSCode.concatenate(
        quantum_hamming_code,
        codes.CSSCode.classical(codes.ClassicalCode.from_generator(decomposition), Pauli.Z),
        range(quantum_hamming_code.dimension),
    )
    assert code.is_equiv_to(tetrahedral_code)


def test_gross_codes() -> None:
    """Named bivariate bicycle codes."""
    assert codes.GrossCode().get_code_params() == (144, 12, 12)
    assert codes.TwoGrossCode().get_code_params() == (288, 12, 18)


def test_small_codes() -> None:
    """Small named codes."""
    assert codes.QuditCode(codes.FiveQubitCode().matrix).get_code_params() == (5, 1, 3)
    assert codes.SteaneCode().get_code_params() == (7, 1, 3)
    assert codes.CSSCode.equiv(codes.C4Code(), codes.IcebergCode(4))
    assert codes.C6Code().get_code_params() == (6, 2, 2)

    # the quantum Golay code has weight-8 stabilizers
    golay_code = codes.QuantumGolayCode()
    assert set(golay_code.matrix_x.view(np.ndarray).sum(axis=1)) == {8}


def test_cached_parameters_are_genuine() -> None:
    """Codes that cache their parameters agree with a computation that ignores the cache."""
    assert_cached_parameters_are_genuine([(codes.QuantumGolayCode(), (23, 1, 7))])
