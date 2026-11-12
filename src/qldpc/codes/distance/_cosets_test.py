# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the parts that the coset searches of ``CosetLeaderSearch`` share."""

from __future__ import annotations

import numpy as np
import pytest

from qldpc._binary import pack_bits

from . import _cosets


def test_inverses_of_checks_have_one_syndrome_bit_in_each_row(pytestconfig: pytest.Config) -> None:
    """Row ``j`` of the inverse of independent checks has a syndrome of only bit ``j``."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for num_checks, num_bits in [(1, 1), (3, 7), (5, 12)]:
        identity = np.eye(num_checks, dtype=np.uint8)
        extra = rng.integers(0, 2, (num_checks, num_bits - num_checks), dtype=np.uint8)
        checks = np.hstack([identity, extra])[:, rng.permutation(num_bits)]
        inverse = _cosets.get_unit_syndrome_words(checks)
        assert inverse.shape == checks.shape
        np.testing.assert_array_equal(checks @ inverse.T % 2, identity)


def test_rows_pack_into_integers_and_back(pytestconfig: pytest.Config) -> None:
    """Entry ``j`` of a row is bit ``j`` of its integer, which splits back into its packed words."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    matrix = rng.integers(0, 2, (5, 130), dtype=np.uint8)
    values = _cosets.row_ints(matrix)
    assert values == tuple(sum(int(bit) << jj for jj, bit in enumerate(row)) for row in matrix)
    packed = pack_bits(matrix)
    for row, value in zip(packed, values, strict=True):
        assert _cosets.words_to_int(row) == value
        np.testing.assert_array_equal(_cosets.int_to_words(value, packed.shape[1]), row)


def test_sums_of_selected_rows_follow_their_definition(pytestconfig: pytest.Config) -> None:
    """Both sums of the rows that the set bits of a support select add those rows modulo two.

    The 70 rows make the packed supports span two machine words.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    rows = rng.integers(0, 2, (70, 9), dtype=np.uint8)
    row_values, packed_rows = _cosets.row_ints(rows), pack_bits(rows)
    for _ in range(10):
        selected = rng.integers(0, 2, len(rows), dtype=np.uint8)
        expected = rows[selected.astype(bool)].sum(axis=0) % 2
        (support,) = _cosets.row_ints(selected[None])
        assert _cosets.xor_selected(row_values, support) == _cosets.row_ints(expected[None])[0]
        word = np.zeros(packed_rows.shape[1], dtype=np.uint64)
        _cosets.xor_selected_rows(pack_bits(selected[None])[0], packed_rows, word)
        np.testing.assert_array_equal(word, pack_bits(expected[None])[0])


def test_tables_of_single_errors_keep_the_first_word_of_each_syndrome() -> None:
    """Tables of the words of weight zero or one map each syndrome to its first word.

    For packed Paulis on two qubits, the words are tried qubit by qubit, X before Z before Y.  So
    the Z error on qubit zero, at bit two, takes the syndrome 0b10 before the X error on qubit one,
    and the Y error on qubit zero takes 0b11 before the Z error on qubit one.  For X or Z parts, the
    two bits with the syndrome 0b01 keep the first, and the tables of X and Z errors hold only the
    identity.
    """
    singles, x_singles, z_singles = _cosets.tabulate_singles(
        [0b01, 0b10, 0b10, 0b11], 2, is_pauli=True
    )
    assert singles == {0: 0, 0b01: 0b0001, 0b10: 0b0100, 0b11: 0b0101}
    assert x_singles == {0: 0, 0b01: 0b0001, 0b10: 0b0010}
    assert z_singles == {0: 0, 0b10: 0b0100, 0b11: 0b1000}
    singles, x_singles, z_singles = _cosets.tabulate_singles([0b01, 0b01, 0b10], 0, is_pauli=False)
    assert singles == {0: 0, 0b01: 0b001, 0b10: 0b100}
    assert x_singles == z_singles == {0: 0}
