# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the smallest weights that a basis of logical operators can have."""

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from ._sectors import LogicalOpsKind
from .evaluation import evaluate_logical_ops
from .spectrum import _get_min_weight, _has_light_word, get_logical_weight_spectrum


@pytest.mark.parametrize("pauli", [Pauli.X, Pauli.Z])
def test_spectrum(
    pauli: PauliXZ, stabilizer_code: codes.CSSCode, subsystem_code: codes.CSSCode
) -> None:
    """The spectrum matches a greedy search over every logical operator, and is achieved."""
    other: PauliXZ = Pauli.Z if pauli == Pauli.X else Pauli.X
    cases: list[tuple[codes.CSSCode, LogicalOpsKind, npt.ArrayLike, npt.ArrayLike]] = [
        (
            stabilizer_code,
            "dressed",
            stabilizer_code.get_matrix(other),
            stabilizer_code.get_matrix(pauli),
        ),
        (
            subsystem_code,
            "dressed",
            subsystem_code.get_stabilizer_ops(other),
            subsystem_code.get_matrix(pauli),
        ),
        (
            subsystem_code,
            "bare",
            subsystem_code.get_matrix(other),
            subsystem_code.get_stabilizer_ops(pauli),
        ),
    ]
    for code, logicals, checks, trivial in cases:
        spectrum = get_logical_weight_spectrum(code, pauli, logicals=logicals)
        assert list(spectrum.weights) == _get_brute_force_spectrum(checks, trivial)
        metrics = evaluate_logical_ops(code, spectrum.logical_ops, pauli=pauli, logicals=logicals)
        assert tuple(metrics.weights) == spectrum.weights

    # The dressed spectra of this subsystem code are lighter than its bare spectra.
    dressed = get_logical_weight_spectrum(subsystem_code, pauli)
    bare = get_logical_weight_spectrum(subsystem_code, pauli, logicals="bare")
    assert dressed.weights != bare.weights


def test_spectrum_properties() -> None:
    """Summary values follow from the weights, including for codes without logical qubits."""
    spectrum = get_logical_weight_spectrum(codes.SteaneCode(), Pauli.X)
    assert (spectrum.max_weight, spectrum.total_weight, spectrum.load_lower_bound) == (3, 3, 1)

    spectrum = get_logical_weight_spectrum(codes.CSSCode([[1, 1]], [[1, 1]]), Pauli.X)
    assert spectrum.weights == ()
    assert spectrum.logical_ops.shape == (0, 2)
    assert (spectrum.max_weight, spectrum.total_weight, spectrum.load_lower_bound) == (0, 0, 0)


def test_weight_queries() -> None:
    """Weight queries find words outside the excluded span, and none when there are none."""
    space = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    assert _get_min_weight(space, space[:1]) == 2
    assert _has_light_word(space, space[:1], 2)
    assert not _has_light_word(space, space[:1], 1)
    assert not _has_light_word(space, space, 3)


def _get_brute_force_spectrum(checks: npt.ArrayLike, trivial: npt.ArrayLike) -> list[int]:
    """Greedy weights over all words of ker(checks), sorted by weight, modulo row(trivial)."""
    space = [_to_int(row) for row in galois.GF2(np.asarray(checks)).null_space()]
    words = []
    for combination in range(1, 2 ** len(space)):
        word = 0
        for index, row in enumerate(space):
            if combination >> index & 1:
                word ^= row
        words.append(word)
    words.sort(key=lambda word: word.bit_count())

    pivots: dict[int, int] = {}
    for row in np.asarray(trivial):
        _insert(pivots, _to_int(row))
    return [word.bit_count() for word in words if _insert(pivots, word)]


def _to_int(row: npt.ArrayLike) -> int:
    """A binary row as the bits of an integer."""
    return int("".join(str(int(bit)) for bit in np.asarray(row)) or "0", 2)


def _insert(pivots: dict[int, int], word: int) -> bool:
    """Add a word to a basis in echelon form, keyed by leading bit, if it is independent."""
    while word:
        leading_bit = word.bit_length() - 1
        if leading_bit not in pivots:
            pivots[leading_bit] = word
            return True
        word ^= pivots[leading_bit]
    return False
