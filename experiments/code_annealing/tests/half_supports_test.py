# SPDX-License-Identifier: Apache-2.0

"""The supports of minimum-weight logical operators, and the low-weight generators of code files.

These numbers go into the paper's figures and listings, and a wrong one would look plausible, so
they are checked against small codes whose supports are derived by hand.
"""

from __future__ import annotations

from pathlib import Path

import galois
import numpy as np
import pytest

from qldpc import codes

from ..analysis import half_supports
from ..analysis.half_supports import HalfSupportCounts, read_half_supports
from ..analysis.min_weight_supports import get_min_weight_supports
from ..core.code_files import format_code, reduce_weights, write_code_file
from ..core.paths import get_code_files, get_half_supports_file

# The [[4, 2, 2]] code: its minimum-weight X-only logicals are the 6 operators XX on a pair of
# qubits, which commute with ZZZZ, and likewise for Z.  Their half supports, of (2 + 1) // 2 = 1
# qubit, are the 4 qubits, and their majority supports, of 2 // 2 + 1 = 2 qubits, are the 6 pairs.
_CHECKS_422 = [[1, 1, 1, 1]]

# The Steane code: its minimum-weight X-only logicals are the 7 words of weight 3 of the Hamming
# code, the lines of the Fano plane, and likewise for Z.  Every pair of points of the Fano plane
# lies on one line, so the subsets of 2 qubits, which are both its half and its majority supports,
# are the 21 pairs of qubits.
_CHECKS_STEANE = [[1, 0, 0, 1, 0, 1, 1], [0, 1, 0, 1, 1, 0, 1], [0, 0, 1, 0, 1, 1, 1]]


def test_the_half_supports_file_has_the_counts_derived_by_hand(tmp_path: Path) -> None:
    blocks = [
        format_code(_CHECKS_STEANE, _CHECKS_STEANE, distance=3, count=14),
        format_code(_CHECKS_422, _CHECKS_422, distance=2, count=12),
    ]
    write_code_file(get_code_files(tmp_path)["css"], blocks)
    half_supports.main(["--results-dir", str(tmp_path)])
    counts = read_half_supports(get_half_supports_file(tmp_path))
    assert counts == {
        "annealed_codes.txt": {
            (7, 1, 3): HalfSupportCounts(7, 1, 3, 7, 7, 21, 21, 21, 21),
            (4, 2, 2): HalfSupportCounts(4, 2, 2, 6, 6, 4, 4, 6, 6),
        }
    }


@pytest.mark.parametrize("lighter", ["X", "Z"])
def test_only_the_sector_of_the_lighter_logicals_has_minimum_weight_supports(lighter: str) -> None:
    # Two checks of weight 2 on three qubits leave one logical qubit.  Its logical of the type of
    # the checks has weight 1 (any one qubit, since the checks make the three equivalent), and its
    # logical of the other type, which overlaps each check evenly, has weight 3 (all qubits).
    checks = np.array([[1, 1, 0], [0, 1, 1]])
    empty = np.zeros((0, 3), dtype=int)
    checks_x, checks_z = (checks, empty) if lighter == "X" else (empty, checks)
    supports = get_min_weight_supports(codes.CSSCode(checks_x, checks_z))
    single_qubits = (0b001, 0b010, 0b100)
    assert supports.distance == 1
    assert (supports.supports_x, supports.supports_z) == (
        (single_qubits, ()) if lighter == "X" else ((), single_qubits)
    )


def test_reducing_weights_keeps_the_span_of_the_generators() -> None:
    # Adding the rows 1111 and 1110 gives 0001, which is lighter than 1111, and no further sum is
    # lighter than a row.
    np.testing.assert_array_equal(
        reduce_weights([[1, 1, 1, 1], [1, 1, 1, 0]]), [[0, 0, 0, 1], [1, 1, 1, 0]]
    )
    rng = np.random.default_rng(0)
    for _ in range(20):
        matrix = galois.GF2.Random((5, 12), seed=rng)
        reduced = galois.GF2(reduce_weights(matrix))
        assert np.array_equal(reduced.row_reduce(), matrix.row_reduce())
        assert reduced.view(np.ndarray).sum() <= matrix.view(np.ndarray).sum()
