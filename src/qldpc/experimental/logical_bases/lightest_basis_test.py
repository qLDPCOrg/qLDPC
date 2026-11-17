# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the smallest weights that a basis of logical operators can have."""

from typing import Literal

import galois
import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes
from qldpc.objects import Pauli, PauliXZ

from . import lightest_basis
from .evaluation import evaluate_logical_ops
from .lightest_basis import (
    LightestLogicalBasis,
    _get_min_weight,
    _has_light_word,
    get_lightest_logical_basis,
)


@pytest.mark.parametrize("pauli", [Pauli.X, Pauli.Z])
def test_lightest_basis(
    pauli: PauliXZ, stabilizer_code: codes.CSSCode, subsystem_code: codes.CSSCode
) -> None:
    """The weights match a greedy search over every logical operator, and the basis has them."""
    other = pauli.swap_xz()
    cases: list[tuple[codes.CSSCode, Literal["dressed", "bare"], npt.ArrayLike, npt.ArrayLike]] = [
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
    weights = []
    for code, logicals, checks, trivial in cases:
        basis = get_lightest_logical_basis(code, pauli, logicals=logicals)
        assert list(basis.weights) == _get_brute_force_weights(checks, trivial)
        metrics = evaluate_logical_ops(code, basis.logical_ops, pauli=pauli, logicals=logicals)
        assert tuple(metrics.weights) == basis.weights
        weights.append(basis.weights)

    # the dressed operators of this subsystem code are lighter than its bare operators
    dressed, bare = weights[1:]
    assert dressed != bare
    assert all(dressed_weight <= bare_weight for dressed_weight, bare_weight in zip(dressed, bare))


def test_exact_searches_only_at_new_weights(
    stabilizer_code: codes.CSSCode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The greedy search runs an exact search only when the weight rises."""
    exact_weights = []
    get_min_weight = lightest_basis._get_min_weight

    def record_min_weight(space: npt.NDArray[np.uint8], excluded: npt.NDArray[np.uint8]) -> int:
        exact_weights.append(get_min_weight(space, excluded))
        return exact_weights[-1]

    monkeypatch.setattr(lightest_basis, "_get_min_weight", record_min_weight)
    basis = get_lightest_logical_basis(stabilizer_code, Pauli.X)
    assert len(set(basis.weights)) < len(basis.weights)
    assert exact_weights == sorted(set(basis.weights))


def test_lightest_basis_properties() -> None:
    """Summary values follow from the weights, including for codes without logical qubits."""
    basis = LightestLogicalBasis((3, 3, 3), np.zeros((3, 4), dtype=np.uint8))
    assert (basis.max_weight, basis.total_weight, basis.load_lower_bound) == (3, 9, 3)

    basis = get_lightest_logical_basis(codes.CSSCode([[1, 1]], [[1, 1]]), Pauli.X)
    assert basis.weights == ()
    assert basis.logical_ops.shape == (0, 2)
    assert (basis.max_weight, basis.total_weight, basis.load_lower_bound) == (0, 0, 0)


def test_weight_queries() -> None:
    """Weight queries find words outside the excluded span, and none when there are none."""
    space = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    assert _get_min_weight(space, space[:1]) == 2
    assert _has_light_word(space, space[:1], 2)
    assert not _has_light_word(space, space[:1], 1)
    assert not _has_light_word(space, space, 3)


def _get_brute_force_weights(checks: npt.ArrayLike, trivial: npt.ArrayLike) -> list[int]:
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
