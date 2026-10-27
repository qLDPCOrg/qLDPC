# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for exact-distance, binary-algebra, BZ, and count tests."""

import itertools
from collections.abc import Iterator
from typing import Any
from unittest import mock

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes
from qldpc.objects import Pauli

from . import _binary as binary
from . import _information_sets
from . import logical_counts as lc
from ._binary import get_kernel
from .core import DistanceMethod


@pytest.fixture
def force_information_sets() -> Iterator[None]:
    """Route Brouwer-Zimmermann searches through information sets instead of exhaustive search."""
    with (
        mock.patch.multiple(
            "qldpc.codes.distance._brouwer_zimmermann",
            exhaustive_is_cheaper=mock.Mock(return_value=False),
            _brute_force_is_cheaper=mock.Mock(return_value=False),
        ),
        mock.patch("qldpc.codes.distance.core.exhaustive_is_cheaper", return_value=False),
    ):
        yield


@pytest.fixture(params=["python", "compiled"])
def compilation_mode(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the always-compiled functions of the basis family stages as Python, or compiled.

    Coverage can trace only the Python mode.
    """
    if request.param == "python":
        for name, function in list(vars(_information_sets).items()):
            if (
                hasattr(function, "py_func")
                and function.py_func.__module__ == _information_sets.__name__
            ):
                monkeypatch.setattr(_information_sets, name, function.py_func)


def get_random_full_rank_matrix(
    rng: np.random.Generator, rows: int, cols: int
) -> npt.NDArray[np.uint8]:
    """Return a random full-row-rank binary matrix with the requested row and column counts."""
    while True:
        matrix = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)
        if len(binary.get_independent_rows(matrix)) == rows:
            return matrix


def count_set_bits(val: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    """Return per-word bit counts, computed bit by bit as an independent reference for popcounts."""
    val = np.asarray(val)
    nbits = 8 * val.itemsize
    out = np.empty_like(val)
    for indices in np.ndindex(val.shape):
        ival = int(val[indices])
        out[indices] = sum(ival >> i & 1 for i in range(nbits))

    return out


####################################################################################################
# helpers for the tests of logical-operator counts

Counts = lc.CSSLogicalCounts
METHODS: tuple[DistanceMethod, ...] = ("brouwer_zimmermann", "brute_force")


def get_counter(checks_x: npt.ArrayLike, checks_z: npt.ArrayLike) -> lc.CSSLogicalCounter:
    """A counter for the CSS code with the given binary checks."""
    code = codes.CSSCode(np.asarray(checks_x, dtype=int), np.asarray(checks_z, dtype=int))
    return lc.CSSLogicalCounter(code)


def get_bits(values: list[int], width: int) -> npt.NDArray[np.uint8]:
    """Binary rows whose bit j is bit j of each integer."""
    return np.array([[value >> jj & 1 for jj in range(width)] for value in values], dtype=np.uint8)


def append_frozen_qubits(
    checks_x: npt.ArrayLike, checks_z: npt.ArrayLike, width: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Append qubits fixed in |0> by a Z check each; no logical of minimum weight touches them."""
    x, z = np.asarray(checks_x, dtype=np.uint8), np.asarray(checks_z, dtype=np.uint8)
    extra = width - x.shape[1]
    padded_x = np.pad(x, ((0, 0), (0, extra)))
    padded_z = np.vstack(
        [np.pad(z, ((0, 0), (0, extra))), np.hstack([np.zeros((extra, z.shape[1])), np.eye(extra)])]
    ).astype(np.uint8)
    return padded_x, padded_z


def get_random_css_code(
    rng: np.random.Generator, num_qubits: int, rank_x: int, rank_z: int
) -> codes.CSSCode:
    """A random binary CSS stabilizer code: Z checks are random combinations of ker(H_X)."""
    checks_x = rng.integers(0, 2, (rank_x, num_qubits), dtype=np.uint8)
    kernel = get_kernel(checks_x)
    checks_z = rng.integers(0, 2, (rank_z, len(kernel))) @ kernel % 2
    return codes.CSSCode(checks_x.astype(int), checks_z.astype(int))


def get_span(rows: npt.ArrayLike) -> set[int]:
    """All combinations of the given binary rows, as integers whose bit j is column j."""
    words = {0}
    for row in np.asarray(rows, dtype=int):
        value = sum(int(bit) << jj for jj, bit in enumerate(row))
        words |= {word ^ value for word in words}
    return words


def enumerate_logical_ops(code: codes.CSSCode) -> tuple[list[int], list[int], int, int]:
    """Count logical operators by enumerating the normalizers, using CSSCode's stabilizers.

    Returns the X-only and Z-only counts at every weight, the distance, and the number of nontrivial
    logical operators of any type at the distance.
    """
    num_qubits = len(code)
    normalizers = []
    for stabilizers in (code.get_stabilizer_ops(Pauli.Z), code.get_stabilizer_ops(Pauli.X)):
        field = galois.GF(2)
        kernel = field(stabilizers).null_space() if len(stabilizers) else field.Identity(num_qubits)
        normalizers.append(get_span(kernel))
    normal_x, normal_z = normalizers
    gauge_x, gauge_z = get_span(code.matrix_x), get_span(code.matrix_z)
    counts_x, counts_z = [0] * (num_qubits + 1), [0] * (num_qubits + 1)
    for word in normal_x - gauge_x:
        counts_x[word.bit_count()] += 1
    for word in normal_z - gauge_z:
        counts_z[word.bit_count()] += 1
    distance = next(
        weight for weight in range(1, num_qubits + 1) if counts_x[weight] or counts_z[weight]
    )
    full = sum(
        (word_x | word_z).bit_count() == distance
        for word_x, word_z in itertools.product(normal_x, normal_z)
        if word_x not in gauge_x or word_z not in gauge_z
    )
    return counts_x, counts_z, distance, full


def get_expected_counts(code: codes.CSSCode) -> lc.CSSLogicalCounts:
    """Counts at the distance, including full Pauli operators, from an independent enumeration."""
    counts_x, counts_z, distance, full = enumerate_logical_ops(code)
    return Counts(distance, counts_x[distance], counts_z[distance], full)


def assert_counts_match_enumeration(counter: lc.CSSLogicalCounter, **options: Any) -> None:
    """Check the counts at every weight, and the Pauli counts at the distance, by enumeration."""
    counts_x, counts_z, distance, full = enumerate_logical_ops(counter.code)
    for weight in range(counter.num_qubits + 1):
        result = counter.count_logical_ops(weight, **options)
        assert (result.count_x, result.count_z) == (counts_x[weight], counts_z[weight])
    expected = Counts(distance, counts_x[distance], counts_z[distance], full)
    assert counter.count_logical_ops(count_pauli=True, **options) == expected


# X checks {0,1,2,3,4} and {2,4}, and Z checks {1,2,3,4} and {0,1}, on five qubits.  By hand, the
# distance is two, the X-only logicals of weight two are {2,3} and {3,4}, and the Z-only ones are
# {2,4}, {0,3} and {1,3}.  The Z word {2,4} also supports the X stabilizer {2,4}, so in total there
# are 2 + 3 + 1 = 6 logical Pauli operators of weight two.
HAND_X, HAND_Z = get_bits([31, 20], 5), get_bits([30, 3], 5)
