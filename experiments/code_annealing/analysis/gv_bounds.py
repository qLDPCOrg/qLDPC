# SPDX-License-Identifier: Apache-2.0

"""The stabilizer and CSS Gilbert-Varshamov bounds, against which the paper compares its codes.

Each bound is the largest distance ``d`` for which a counting argument guarantees that an
``[[n, k, d]]`` code exists, so a code whose distance meets or exceeds it is at least as good as a
typical code of its size.
"""

from __future__ import annotations

import math


def get_qgv_distance(num_qubits: int, num_logicals: int) -> int:
    """The stabilizer Gilbert-Varshamov bound: the largest guaranteed distance of a code.

    This is the largest ``d`` with ``sum(3**w * comb(n, w) for w < d) <= 2**(n - k)``, so that an
    ``[[n, k, d]]`` stabilizer code exists.

    Args:
        num_qubits: The number ``n`` of physical qubits.
        num_logicals: The number ``k`` of logical qubits.

    Returns:
        The largest distance ``d`` that the bound guarantees.
    """
    distance, errors = 1, 1
    while distance < num_qubits:
        errors += 3**distance * math.comb(num_qubits, distance)
        if errors > 2 ** (num_qubits - num_logicals):
            break
        distance += 1
    return distance


def get_css_gv_distance(num_qubits: int, num_logicals: int) -> int:
    """The CSS Gilbert-Varshamov bound: the largest guaranteed distance of a CSS code.

    This is the largest ``d`` with ``2 * sum(comb(n, w) for 0 < w < d) < (2**n - 1) / D``, where
    ``D = 2**k_x - 2**(n - k_x) + 2**k_z - 2**(n - k_z)`` and ``k_x`` and ``k_z`` are the floor and
    ceiling of ``(n + k) / 2``, so that an ``[[n, k, d]]`` CSS code exists.

    Args:
        num_qubits: The number ``n`` of physical qubits.
        num_logicals: The number ``k`` of logical qubits.

    Returns:
        The largest distance ``d`` that the bound guarantees.
    """
    nn = num_qubits
    k_x, k_z = (nn + num_logicals) // 2, (nn + num_logicals + 1) // 2
    denominator = 2**k_x - 2 ** (nn - k_x) + 2**k_z - 2 ** (nn - k_z)
    distance, words = 1, 0
    while distance < nn:
        words += math.comb(nn, distance)
        if 2 * words * denominator >= 2**nn - 1:
            break
        distance += 1
    return distance
