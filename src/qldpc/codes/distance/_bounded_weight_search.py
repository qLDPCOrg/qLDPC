# SPDX-License-Identifier: Apache-2.0

"""Exact decisions on whether a Pauli error class contains an error of weight at most one.

Used by ``circuits.certify_state_prep_fault_propagation``; not public API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from ._binary import as_binary_matrix, assert_binary, row_reduce_binary

####################################################################################################
# Pauli-coset weight decisions


@dataclass(frozen=True)
class PauliWeightDecision:
    """The answer to a weight query on a class of equivalent Pauli errors.

    ``within_bound`` says whether the class contains an error of weight at most the queried
    threshold, in the oracle's fixed metric.  ``representative`` is an error of the class, packed
    with its X part in the low ``n`` bits and its Z part in the next ``n``.  If ``within_bound`` is
    True, its weight is at most the threshold.  ``lower_bound`` is a proved lower bound on the
    weight of every error in the class, which exceeds the queried threshold when ``within_bound``
    is False.
    """

    within_bound: bool
    representative: int
    lower_bound: int


class PauliCosetSearch:
    """Exact weight oracle for the Pauli errors that leave a pure stabilizer state unchanged.

    The syndrome of an error lists whether it anticommutes with each of the ``n`` stabilizers.  Full
    syndromes identify classes of equivalent errors: a maximal isotropic space equals its
    symplectic orthogonal, so errors have the same syndrome exactly when they differ by an element
    of the stabilizer group.  Bit ``j`` of a packed syndrome refers to row ``j`` of ``stabilizers``,
    and a packed Pauli has its X part in the low ``n`` bits and its Z part in the next ``n``.

    Physical weight one means a single X, Y, or Z, so a table of the ``3n`` single-qubit syndromes
    decides it in linear space.  CSS weight one also includes ``X_i Z_j`` on distinct qubits:
    matching an X-single syndrome with a Z-single syndrome checks these in linear time without a
    quadratic table.  If no error of weight at most one has a given syndrome, every error with that
    syndrome has weight at least two.  The oracle decides only thresholds zero and one.

    Args:
        stabilizers: An independent commuting binary ``n x 2n`` matrix in ``[X|Z]`` order.
        metric: ``"pauli"`` counts physical support, and ``"css"`` counts ``max(|x|, |z|)``, where
            ``x`` and ``z`` are the X and Z parts of a whole error.
    """

    def __init__(
        self, stabilizers: npt.ArrayLike, *, metric: Literal["pauli", "css"] = "pauli"
    ) -> None:
        if metric not in ("pauli", "css"):
            raise ValueError("metric must be 'pauli' or 'css'")
        self.metric = metric
        assert_binary(stabilizers, "stabilizers")
        matrix = as_binary_matrix(stabilizers)
        self.num_qubits = len(matrix)
        n = self.num_qubits
        if not n or matrix.shape[1] != 2 * n:
            raise ValueError("Require a complete nonempty n by 2n stabilizer matrix")
        checks = np.roll(matrix, n, axis=1)
        _, labels, pivots = row_reduce_binary(checks, np.eye(n, dtype=np.uint8))
        if len(pivots) != n or np.any((checks @ matrix.T) % 2):
            raise ValueError("Stabilizers must be independent and commute")
        assert labels is not None

        # Row j of the inverse is an error whose syndrome has only bit j set.
        inverse = np.zeros((n, 2 * n), dtype=np.uint8)
        inverse[:, pivots] = labels.T
        self._inverse = tuple(_binary_row_int(row) for row in inverse)
        self._checks = tuple(_binary_row_int(row) for row in checks)

        # Map each syndrome of a single-qubit error, or of no error, to such an error.
        self._singles = {0: 0}
        self._x_singles = {0: 0}
        self._z_singles = {0: 0}
        for qubit in range(n):
            x, z = 1 << qubit, 1 << (qubit + n)
            sx, sz = self.syndrome(x), self.syndrome(z)
            self._x_singles.setdefault(sx, x)
            self._z_singles.setdefault(sz, z)
            self._singles.setdefault(sx, x)
            self._singles.setdefault(sz, z)
            self._singles.setdefault(sx ^ sz, x ^ z)

    def syndrome(self, pauli: int) -> int:
        """Compute the complete syndrome of a packed symplectic Pauli."""
        return sum(((row & pauli).bit_count() & 1) << jj for jj, row in enumerate(self._checks))

    def decide(self, syndrome: int, max_weight: int) -> PauliWeightDecision:
        """Decide whether some error with this syndrome has weight at most ``max_weight``.

        Args:
            syndrome: A packed syndrome, with bit ``j`` for stabilizer ``j``.
            max_weight: The weight threshold, which must be zero or one.

        Raises:
            ValueError: If the syndrome is out of range or the threshold is not zero or one.
        """
        if not 0 <= syndrome < 1 << self.num_qubits:
            raise ValueError("The syndrome is out of range")
        if max_weight not in (0, 1):
            raise ValueError("The weight threshold must be zero or one")
        pauli = self._weight_one_representative(syndrome)
        if pauli is None:
            return PauliWeightDecision(False, self._representative(syndrome), 2)
        return PauliWeightDecision(self._weight(pauli) <= max_weight, pauli, int(bool(syndrome)))

    def _representative(self, syndrome: int) -> int:
        return _xor_selected(self._inverse, syndrome)

    def _weight_one_representative(self, syndrome: int) -> int | None:
        """Find an error of weight zero or one with this syndrome, if there is one."""
        if syndrome in self._singles:
            return self._singles[syndrome]
        if self.metric == "css":
            for sx, x in self._x_singles.items():
                z = self._z_singles.get(syndrome ^ sx)
                if z is not None:
                    return x ^ z
        return None

    def _weight(self, pauli: int) -> int:
        mask = (1 << self.num_qubits) - 1
        x, z = pauli & mask, (pauli >> self.num_qubits) & mask
        return max(x.bit_count(), z.bit_count()) if self.metric == "css" else (x | z).bit_count()


####################################################################################################
# helper methods


def _binary_row_int(row: npt.NDArray[np.uint8]) -> int:
    return int.from_bytes(np.packbits(row, bitorder="little").tobytes(), "little")


def _xor_selected(rows: Sequence[int], support: int) -> int:
    word = 0
    while support:
        low = support & -support
        word ^= rows[low.bit_length() - 1]
        support ^= low
    return word
