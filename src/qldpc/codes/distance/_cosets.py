# SPDX-License-Identifier: Apache-2.0

"""Parts that the coset searches of ``CosetLeaderSearch`` share.

They are the answers of coset queries (:class:`MinWeightBound`), the inverses of checks that turn
syndromes into words, the packing of binary rows into integers and back, the sums of the rows that
a support selects, and the tables of the errors of weight zero or one by syndrome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits, row_reduce_binary, trailing_zeros_word
from qldpc._compilation import always_compiled


@dataclass(frozen=True)
class MinWeightBound:
    """The answer to a query about the minimum weight of a coset.

    A query of ``CosetLeaderSearch`` asks whether a syndrome class of Pauli errors, which is a coset
    of the stabilizer group, has an error of weight at most ``max_weight``.  A query of one of its
    coset searches asks the same about a coset of the code that the search covers.  Which element
    and which lower bound a query returns can depend on the method and tuning of the search, and on
    the version of qLDPC, within the guarantees below.

    Attributes:
        within_max_weight: Whether the coset has an element of weight at most ``max_weight``.
        representative: An element of the coset: a packed Pauli (see ``CosetLeaderSearch``) for a
            query of the oracle, and a word of the search (a packed Pauli, or an X or Z part) for a
            query of a coset search.  If ``within_max_weight`` is True, its weight is at most
            ``max_weight``.
        lower_bound: A lower bound on the minimum weight of the coset.  If ``within_max_weight`` is
            False, it exceeds ``max_weight``.
    """

    within_max_weight: bool
    representative: int
    lower_bound: int


def _get_inverse(checks: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """A right inverse of independent checks: row ``j`` has a syndrome of only bit ``j``."""
    _, labels, pivots = row_reduce_binary(checks, np.eye(len(checks), dtype=np.uint8))
    assert labels is not None
    inverse = np.zeros(checks.shape, dtype=np.uint8)
    inverse[:, pivots] = labels.T
    return inverse


def _row_ints(matrix: npt.NDArray[np.uint8]) -> tuple[int, ...]:
    """Pack each row of a binary matrix into an integer, with entry ``j`` in bit ``j``."""
    return tuple(map(_packed_row_int, pack_bits(matrix)))


def _packed_row_int(row: npt.NDArray[np.uint64]) -> int:
    """Convert a row of 64-bit words into an integer, with word ``j`` in bits ``64 j`` and up."""
    return sum(int(word) << (64 * jj) for jj, word in enumerate(row))


def _int_to_words(value: int, width: int) -> npt.NDArray[np.uint64]:
    """Split an integer into ``width`` 64-bit words, with word ``j`` in bits ``64 j`` and up."""
    return np.frombuffer(value.to_bytes(8 * width, "little"), dtype="<u8").astype(np.uint64)


def _tabulate_singles(
    check_columns: Sequence[int], num_qubits: int, *, is_pauli: bool
) -> tuple[dict[int, int], dict[int, int], dict[int, int]]:
    """Tabulate the words of weight zero or one by syndrome, and the single-qubit X and Z errors.

    The words are the identity and, for packed Paulis, X, Z, and Y on each qubit, and for X or Z
    parts, the single bits.  The syndrome of Y on a qubit is the sum of those of X and Z there.
    Each table maps each syndrome of its words to the first of them, in the order of the identity,
    and then of the qubits, with X before Z before Y on each, or of the bits.

    Args:
        check_columns: Column ``c`` of the checks as an integer: the syndrome of the word with only
            bit ``c`` set.
        num_qubits: The number of qubits ``n`` of packed Paulis, which have ``2 n`` bits; unused for
            X or Z parts.
        is_pauli: Whether the words are packed Paulis, rather than X or Z parts.

    Returns:
        The table of these words, and the tables of the identity and the single-qubit X errors, and
        of the identity and the single-qubit Z errors.  For X or Z parts, the last two hold only the
        identity.
    """
    singles, x_singles, z_singles = {0: 0}, {0: 0}, {0: 0}
    if not is_pauli:
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


def _xor_selected(rows: Sequence[int], support: int) -> int:
    """XOR together the rows whose indices are the set bits of ``support``."""
    word = 0
    while support:
        # Add the row for the lowest set bit, and then clear that bit.
        low = support & -support
        word ^= rows[low.bit_length() - 1]
        support ^= low
    return word


@always_compiled(inline=True)
def _xor_selected_rows(
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
