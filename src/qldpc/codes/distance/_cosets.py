# SPDX-License-Identifier: Apache-2.0

"""Parts that the coset searches of ``CosetLeaderSearch`` share.

They are the words with unit syndromes (whose sums give a word of each syndrome), the packing of
binary rows into integers and back, the sums of the rows that a support selects, and the tables of
the errors of weight zero or one by syndrome.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits, row_reduce_binary, trailing_zeros_word
from qldpc._compilation import always_compiled


def get_unit_syndrome_words(checks: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Words with unit syndromes for independent checks: row ``j`` has a syndrome of only bit ``j``.

    As columns, these words form a right inverse of the checks.
    """
    _, labels, pivots = row_reduce_binary(checks, np.eye(len(checks), dtype=np.uint8))
    assert labels is not None
    inverse = np.zeros(checks.shape, dtype=np.uint8)
    inverse[:, pivots] = labels.T
    return inverse


def row_ints(matrix: npt.NDArray[np.uint8]) -> tuple[int, ...]:
    """Pack each row of a binary matrix into an integer, with entry ``j`` in bit ``j``."""
    return tuple(map(words_to_int, pack_bits(matrix)))


def words_to_int(row: npt.NDArray[np.uint64]) -> int:
    """Convert a row of 64-bit words into an integer, with word ``j`` in bits ``64 j`` and up."""
    return sum(int(word) << (64 * jj) for jj, word in enumerate(row))


def int_to_words(value: int, width: int) -> npt.NDArray[np.uint64]:
    """Split an integer into ``width`` 64-bit words, with word ``j`` in bits ``64 j`` and up."""
    return np.frombuffer(value.to_bytes(8 * width, "little"), dtype="<u8").astype(np.uint64)


def tabulate_singles(
    check_columns: Sequence[int], num_qubits: int, *, symplectic: bool
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Tabulate the words of weight zero or one by syndrome, and the single-qubit X and Z errors.

    The words are the identity and, for packed Paulis, X, Z, and Y on each qubit, and for X or Z
    parts, the single bits.  The syndrome of Y on a qubit is the sum of those of X and Z there.
    Each table maps each syndrome of its words to the first word with that syndrome.  The words
    come in this order: the identity, and then X, Z, and Y on each qubit in turn for packed Paulis,
    or each bit in turn for X or Z parts.

    Args:
        check_columns: Column ``c`` of the checks as an integer: the syndrome of the word with only
            bit ``c`` set.
        num_qubits: The number of qubits ``n`` of packed Paulis, which have ``2 n`` bits; unused for
            X or Z parts.
        symplectic: Whether the words are packed Paulis, rather than X or Z parts.

    Returns:
        The table of these words, and the tables of the identity and the single-qubit X errors, and
        of the identity and the single-qubit Z errors.  For X or Z parts, the last two hold only the
        identity.
    """
    singles, x_singles, z_singles = {0: 0}, {0: 0}, {0: 0}
    if not symplectic:
        for column, syndrome in enumerate(check_columns):
            singles.setdefault(syndrome, 1 << column)
        return singles, x_singles, z_singles
    for qubit in range(num_qubits):
        x, z = 1 << qubit, 1 << (qubit + num_qubits)
        sx, sz = check_columns[qubit], check_columns[qubit + num_qubits]
        x_singles.setdefault(sx, x)
        z_singles.setdefault(sz, z)
        singles.setdefault(sx, x)
        singles.setdefault(sz, z)
        singles.setdefault(sx ^ sz, x ^ z)
    return singles, x_singles, z_singles


def xor_selected(rows: Sequence[int], support: int) -> int:
    """XOR together the rows whose indices are the set bits of ``support``."""
    word = 0
    while support:
        # Add the row for the lowest set bit, and then clear that bit.
        low = support & -support
        word ^= rows[low.bit_length() - 1]
        support ^= low
    return word


@always_compiled(inline=True)
def xor_selected_rows(
    packed_bits: npt.NDArray[np.uint64], rows: npt.NDArray[np.uint64], word: npt.NDArray[np.uint64]
) -> None:
    """Add to ``word`` row ``b`` of ``rows`` for each set bit ``b`` of ``packed_bits``.

    ``packed_bits`` is split into 64-bit machine words, with machine word ``j`` holding bits
    ``64 j`` and up, and each set bit must be below the number of rows; this is not checked.
    """
    for machine_word in range(len(packed_bits)):
        bits = packed_bits[machine_word]
        while bits:
            bit = 64 * machine_word + trailing_zeros_word(bits)
            for index in range(len(word)):
                word[index] ^= rows[bit, index]
            bits &= bits - np.uint64(1)
