# SPDX-License-Identifier: Apache-2.0

"""Exact supports of the minimum-weight logical operators of a binary CSS code.

The nontrivial X-only logical operators of a CSS stabilizer code with checks ``H_X`` and ``H_Z``
are the words of ``ker(H_Z)`` outside ``row(H_X)``, and likewise for Z with X and Z exchanged.
``get_min_weight_supports`` lists the supports of all such operators of weight equal to the code
distance, with no sampling.  It builds a basis of ``ker(H_Z)`` from the independent rows of ``H_X``
and one representative per logical qubit, and walks all ``2**dim`` combinations of that basis in
Gray-code order.  A combination is a nontrivial logical operator exactly when it involves a
representative, so the walk costs ``2**dim`` steps with ``dim = n - rank(H_Z)`` for X (and
``n - rank(H_X)`` for Z).

A support is an integer bit mask, in which bit ``q`` is set when qubit ``q`` is in the support.
``count_support_subsets`` counts the distinct subsets of a given size that lie in the supports.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from numba import njit

from qldpc.codes import CSSCode

# A support is a bit mask of one machine word.
_MAX_QUBITS = 64

# Steps of the walk per call of the compiled loop, and words that one call can report.  The first
# bounds the time between interruptions, and the second bounds the memory of the reported words.
_CHUNK = 1 << 24
_BUFFER = 1 << 16


@dataclass(frozen=True)
class MinWeightSupports:
    """The supports of the minimum-weight nontrivial X-only and Z-only logical operators."""

    distance: int
    """Distance of the code: the least weight of a nontrivial X-only or Z-only logical operator."""
    supports_x: tuple[int, ...]
    """Sorted supports (bit masks) of the X-only logical operators of weight ``distance``."""
    supports_z: tuple[int, ...]
    """Sorted supports (bit masks) of the Z-only logical operators of weight ``distance``."""


def get_min_weight_supports(code: CSSCode) -> MinWeightSupports:
    """The supports of all minimum-weight nontrivial X-only and Z-only logical operators.

    The weight is the distance of the code, so one sector has no supports when its own distance is
    larger.  Each support is the support of a distinct operator, so the numbers of supports are the
    counts of such operators.

    Raises:
        ValueError: If the code is not a binary stabilizer code with at most 64 qubits and at
            least one logical qubit.
    """
    if code.field.order != 2:
        raise ValueError(f"Supports require a binary code, not GF({code.field.order})")
    num_qubits = code.num_qubits
    if num_qubits > _MAX_QUBITS:
        raise ValueError(f"Supports require at most {_MAX_QUBITS} qubits, not {num_qubits}")
    rows_x = _get_masks(code.matrix_x)
    rows_z = _get_masks(code.matrix_z)
    if any((row_x & row_z).bit_count() % 2 for row_x in rows_x for row_z in rows_z):
        raise ValueError("Supports require a stabilizer code, whose X and Z checks commute")

    stabilizers_x, logicals_x = _split_normalizer(rows_x, rows_z, num_qubits)
    stabilizers_z, logicals_z = _split_normalizer(rows_z, rows_x, num_qubits)
    if not logicals_x:
        raise ValueError("Supports require at least one logical qubit")

    # The Z walk starts from the X distance: it reports nothing unless Z is at least as good.
    distance_x, words_x = _get_min_weight_words(stabilizers_x, logicals_x, num_qubits)
    distance, words_z = _get_min_weight_words(stabilizers_z, logicals_z, distance_x)
    if distance < distance_x:
        words_x = []
    return MinWeightSupports(distance, tuple(sorted(words_x)), tuple(sorted(words_z)))


def count_support_subsets(supports: Iterable[int], size: int) -> int:
    """The number of distinct subsets of ``size`` qubits that lie in at least one support."""
    subsets: set[int] = set()
    for support in supports:
        # The bits of the qubits are distinct powers of two, so their sum is their union.
        bits = [1 << qubit for qubit in _get_qubits(support)]
        subsets.update(map(sum, itertools.combinations(bits, size)))
    return len(subsets)


class _Span:
    """The span of the vectors that were added, as an echelon basis of bit masks."""

    def __init__(self) -> None:
        self._basis: dict[int, int] = {}

    def add(self, vector: int) -> bool:
        """Add a vector, and return whether it is outside the span of the earlier vectors."""
        while vector:
            top = vector.bit_length() - 1
            if top not in self._basis:
                self._basis[top] = vector
                return True
            vector ^= self._basis[top]
        return False


####################################################################################################
# helper methods


def _get_qubits(support: int) -> list[int]:
    """The qubits of a support."""
    return [qubit for qubit in range(support.bit_length()) if support >> qubit & 1]


def _get_masks(matrix: npt.ArrayLike) -> list[int]:
    """The rows of a binary matrix as bit masks."""
    return [sum(1 << int(qubit) for qubit in np.flatnonzero(row)) for row in np.asarray(matrix)]


def _split_normalizer(
    stabilizers: Sequence[int], checks: Sequence[int], num_qubits: int
) -> tuple[list[int], list[int]]:
    """A basis of the span of the stabilizers, and logical representatives that extend it.

    Together they form a basis of the words that commute with the checks, which are the kernel of
    the checks.  The stabilizers must lie in that kernel.
    """
    span = _Span()
    independent = [row for row in stabilizers if span.add(row)]
    logicals = [word for word in _get_kernel(checks, num_qubits) if span.add(word)]
    return independent, logicals


def _get_kernel(rows: Sequence[int], num_qubits: int) -> list[int]:
    """A basis of the words that have even overlap with every row.

    A word is a dependency among the columns of the rows, so it is found by reducing the columns
    one by one and keeping track of which columns were combined.
    """
    basis: dict[int, tuple[int, int]] = {}
    kernel = []
    for qubit in range(num_qubits):
        column = sum((row >> qubit & 1) << index for index, row in enumerate(rows))
        combination = 1 << qubit
        while column:
            top = column.bit_length() - 1
            if top not in basis:
                basis[top] = (column, combination)
                break
            column ^= basis[top][0]
            combination ^= basis[top][1]
        else:
            kernel.append(combination)
    return kernel


def _get_min_weight_words(
    stabilizers: Sequence[int], logicals: Sequence[int], bound: int
) -> tuple[int, list[int]]:
    """The least weight, at most ``bound``, of the nontrivial words of a normalizer, and the words.

    If every nontrivial word is heavier than the bound, this returns the bound and no words.
    """
    basis = np.array([*stabilizers, *logicals], dtype=np.uint64)
    buffer = np.empty(_BUFFER, dtype=np.uint64)
    total = 1 << len(basis)
    weight = bound
    words: list[int] = []
    start = 1  # the combination of no basis words is trivial
    while start < total:
        stop = min(start + _CHUNK, total)
        while start < stop:
            start, found, least = _walk(basis, len(stabilizers), start, stop, weight, buffer)
            if least < weight:
                weight, words = least, []
            words.extend(buffer[:found].tolist())
    return weight, words


@njit(cache=True)
def _walk(
    basis: npt.NDArray[np.uint64],
    num_stabilizers: int,
    start: int,
    stop: int,
    weight: int,
    out: npt.NDArray[np.uint64],
) -> tuple[int, int, int]:
    """Visit the Gray-code combinations ``start`` to ``stop - 1`` of the basis words.

    The first ``num_stabilizers`` words of the basis are stabilizers, so a combination is
    nontrivial if its Gray code has a bit set beyond them.  The nontrivial words of weight at most
    ``weight`` are stored in ``out``, starting over whenever a lighter word is found.  This returns
    the next combination to visit (which is ``stop`` unless ``out`` is full), the number of stored
    words, and the least weight.
    """
    gray = start ^ (start >> 1)
    word = np.uint64(0)
    for bit in range(len(basis)):
        if (gray >> bit) & 1:
            word ^= basis[bit]
    least = weight
    found = 0
    index = start
    while True:
        if gray >> num_stabilizers:
            word_weight = _popcount(word)
            if word_weight <= least:
                if word_weight < least:
                    least = int(word_weight)
                    found = 0
                out[found] = word
                found += 1
                if found == len(out):
                    return index + 1, found, least
        index += 1
        if index == stop:
            return stop, found, least
        bit = 0
        while not (index >> bit) & 1:
            bit += 1
        word ^= basis[bit]
        gray ^= 1 << bit


@njit(cache=True)
def _popcount(word: np.uint64) -> np.int64:
    """The number of set bits of a 64-bit word."""
    one, two, four = np.uint64(1), np.uint64(2), np.uint64(4)
    pairs = word - ((word >> one) & np.uint64(0x5555555555555555))
    nibbles = (pairs & np.uint64(0x3333333333333333)) + (
        (pairs >> two) & np.uint64(0x3333333333333333)
    )
    sums = (nibbles + (nibbles >> four)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return np.int64((sums * np.uint64(0x0101010101010101)) >> np.uint64(56))
