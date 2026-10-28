# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact binary distance calculations and distance method validation."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

import qldpc

from . import core
from .conftest import get_random_full_rank_matrix, get_span


@pytest.mark.usefixtures("force_information_sets", "compilation_mode")
def test_brouwer_zimmermann_information_set_search() -> None:
    """Information-set search excludes stabilizers, finds logical products, and honors cutoffs."""
    # The Hamming-weight-one stabilizer is lighter than every logical operator.  With block_size=0,
    # some Brouwer-Zimmermann batches contain only that stabilizer.
    distance = qldpc.codes.get_distance_quantum(
        [[1, 1, 1, 1]], [[1, 0, 0, 0]], homogeneous=True, cutoff=0, block_size=0
    )
    assert distance == 3

    # The lightest logical operator is the product of both logical generators.
    distance = qldpc.codes.get_distance_quantum(
        [[1, 1, 1, 0], [1, 1, 0, 1]], [], homogeneous=True, cutoff=0
    )
    assert distance == 2

    # Every generator in every information set has Hamming weight at least three.  Certifying the
    # weight-two codeword requires searching two-generator combinations.
    generators = [
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    ]
    assert qldpc.codes.get_distance_classical(generators, cutoff=0) == 2

    # Both reduced generators have Hamming weight three.  Their weight-two sum reaches the distance
    # cutoff.
    assert qldpc.codes.get_distance_classical([[1, 0, 1, 1], [0, 1, 1, 1]], cutoff=2) == 2

    # Row reduction exposes a Hamming-weight-one generator before any search runs.
    assert qldpc.codes.get_distance_classical([[1, 1, 0], [0, 1, 1], [1, 1, 1]], cutoff=1) == 1


def test_brouwer_zimmermann_special_inputs() -> None:
    """Brouwer-Zimmermann validates matrix shapes and preserves quotient edge cases."""
    assert qldpc.codes.get_distance_classical(np.empty((0, 4), dtype=np.uint8), cutoff=0) == 4
    assert (
        qldpc.codes.get_distance_quantum(
            np.empty((0, 4), dtype=np.uint8),
            np.empty((0, 4), dtype=np.uint8),
            homogeneous=True,
        )
        == 4
    )
    assert qldpc.codes.get_distance_quantum([[1, 0]], [[1, 0]], homogeneous=True) == 0
    assert qldpc.codes.get_distance_quantum(np.empty((0, 4), dtype=np.uint8), [[1, 0, 0, 0]]) == 2

    with pytest.raises(ValueError, match="incompatible widths"):
        qldpc.codes.get_distance_quantum([[1, 0, 0]], [[1, 0]], homogeneous=True)
    with pytest.raises(ValueError, match="even number of columns"):
        qldpc.codes.get_distance_quantum([[1, 0, 0]], [], homogeneous=False)
    with pytest.raises(ValueError, match="even number of columns"):
        qldpc.codes.get_distance_quantum(
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint8),
            homogeneous=False,
        )
    with pytest.raises(ValueError, match="two-dimensional"):
        qldpc.codes.get_distance_classical(np.zeros((1, 1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="infer the width"):
        qldpc.codes.get_distance_classical([])


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.classical.HammingCode(4), 3),
        (qldpc.codes.classical.RepetitionCode(3), 3),
        (qldpc.codes.classical.RepetitionCode(8), 8),
        (qldpc.codes.classical.RingCode(8), 8),
    ],
)
def test_get_distance_classical_known_codes(
    code: qldpc.codes.ClassicalCode, expected_distance: int
) -> None:
    """Classical distance matches known values for Hamming, repetition, and ring codes."""
    distance = qldpc.codes.distance.get_distance_classical(code.generator)
    assert distance == expected_distance

    code._distance = None
    assert code.get_distance_exact() == expected_distance


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.quantum.C4Code(), 2),
        (qldpc.codes.quantum.C6Code(), 2),
        (qldpc.codes.quantum.SteaneCode(), 3),
        (qldpc.codes.quantum.SurfaceCode(3), 3),
        (qldpc.codes.quantum.SurfaceCode(4), 4),
        (qldpc.codes.quantum.ToricCode(4), 4),
    ],
)
def test_get_distance_quantum_css_codes(code: qldpc.codes.CSSCode, expected_distance: int) -> None:
    """Quantum distance matches known values for C4, C6, Steane, and small surface/toric codes."""
    distance_x = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(qldpc.objects.Pauli.X),
        code.get_stabilizer_ops(qldpc.objects.Pauli.X),
        homogeneous=True,
    )
    distance_z = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(qldpc.objects.Pauli.Z),
        code.get_stabilizer_ops(qldpc.objects.Pauli.Z),
        homogeneous=True,
    )
    distance_all = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        homogeneous=True,
    )
    assert min(distance_x, distance_z, distance_all) == expected_distance

    # CSSCode.forget_distance clears the X-sector and Z-sector distance caches before recomputation.
    # Assigning only CSSCode._distance would leave cached sector distances available to the method.
    code.forget_distance()
    assert code.get_distance_exact() == expected_distance


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.quantum.FiveQubitCode(), 3),
    ],
)
def test_get_distance_quantum_noncss_codes(
    code: qldpc.codes.QuditCode, expected_distance: int
) -> None:
    """Quantum distance matches known values for small non-CSS codes."""
    distance = qldpc.codes.distance.get_distance_quantum(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        homogeneous=False,
    )
    assert distance == expected_distance

    code._distance = None
    assert code.get_distance_exact() == expected_distance


@pytest.mark.usefixtures("force_information_sets", "compilation_mode")
def test_brouwer_zimmermann_random_cross_checks() -> None:
    """Information-set search agrees with exhaustive enumeration on random nested codes."""
    rng = np.random.default_rng(118)
    for length in range(5, 10):
        basis = get_random_full_rank_matrix(rng, min(5, length - 1), length)
        expected = qldpc.codes.get_distance_classical(basis, cutoff=0, method="brute_force")
        assert qldpc.codes.get_distance_classical(basis, cutoff=0) == expected

        stabilizers, logical_ops = basis[:2], basis[2:]
        expected = qldpc.codes.get_distance_quantum(
            logical_ops, stabilizers, cutoff=0, homogeneous=True, method="brute_force"
        )
        assert (
            qldpc.codes.get_distance_quantum(logical_ops, stabilizers, cutoff=0, homogeneous=True)
            == expected
        )

    for num_qubits in range(3, 7):
        basis = get_random_full_rank_matrix(rng, min(6, 2 * num_qubits - 1), 2 * num_qubits)
        stabilizers, logical_ops = basis[:2], basis[2:]
        expected = qldpc.codes.get_distance_quantum(
            logical_ops, stabilizers, cutoff=0, method="brute_force"
        )
        assert qldpc.codes.get_distance_quantum(logical_ops, stabilizers, cutoff=0) == expected


def test_max_weight(request: pytest.FixtureRequest) -> None:
    """A weight limit returns the smaller of the distance and max_weight + 1, on every route."""
    rng = np.random.default_rng(7)
    basis = get_random_full_rank_matrix(rng, 4, 7)
    classical = functools.partial(qldpc.codes.get_distance_classical, basis)
    classical_distance = min(word.bit_count() for word in get_span(basis) if word)
    css = functools.partial(
        qldpc.codes.get_distance_quantum, basis[2:], basis[:2], homogeneous=True
    )
    css_distance = min(word.bit_count() for word in get_span(basis) - get_span(basis[:2]))
    basis = get_random_full_rank_matrix(rng, 5, 8)
    pauli = functools.partial(qldpc.codes.get_distance_quantum, basis[2:], basis[:2])
    pauli_distance = min(
        ((word | word >> 4) & 0b1111).bit_count() for word in get_span(basis) - get_span(basis[:2])
    )

    def check(distance_with_limit: Callable[..., int], distance: int, method: str) -> None:
        assert distance >= 2
        for max_weight in [distance - 2, distance]:
            result = distance_with_limit(cutoff=0, max_weight=max_weight, method=method)
            assert result == min(distance, max_weight + 1)

    for distance_with_limit, distance in [(classical, classical_distance), (css, css_distance)]:
        for method in ["brouwer_zimmermann", "brute_force"]:
            check(distance_with_limit, distance, method)

    # Small problems take exhaustive routes by default, so also force the BZ search.  The Pauli case
    # also checks the doubled bound.
    request.getfixturevalue("force_information_sets")
    check(pauli, pauli_distance, "brouwer_zimmermann")


@pytest.mark.usefixtures("force_information_sets")
def test_max_weight_caps_cutoff() -> None:
    """A large cutoff cannot make the search give up on a word within the limit.

    Both basis rows have weight above the limit, but their sum has weight 3.
    """
    basis = [[1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]]
    assert qldpc.codes.get_distance_classical(basis, cutoff=6, max_weight=3) == 3
    distance = qldpc.codes.get_distance_quantum(basis, [], homogeneous=True, cutoff=6, max_weight=3)
    assert distance == 3


def test_max_weight_conventions_and_validation() -> None:
    """The limit also caps the results returned for inputs without nontrivial words."""
    assert qldpc.codes.get_distance_classical([[0, 0, 0]], max_weight=1) == 2
    no_logical_ops = np.zeros((0, 4), dtype=int)
    assert qldpc.codes.get_distance_quantum(no_logical_ops, [[1, 1, 0, 0]], max_weight=0) == 1
    with pytest.raises(ValueError, match="max_weight must be nonnegative"):
        qldpc.codes.get_distance_classical([[1, 1]], max_weight=-1)
    with pytest.raises(ValueError, match="max_weight must be nonnegative"):
        qldpc.codes.get_distance_quantum([[1, 1]], [], homogeneous=True, max_weight=-1)

    # The limit is an integer, and a NumPy integer gives a Python integer, even on a route that
    # clamps its result.
    for not_an_integer in (2.5, True):
        with pytest.raises(TypeError):
            qldpc.codes.get_distance_classical([[1, 1]], max_weight=cast(int, not_an_integer))
    distance = qldpc.codes.get_distance_classical(
        [[1, 1, 1, 1]], max_weight=cast(int, np.int64(1)), method="brute_force"
    )
    assert distance == 2 and type(distance) is int


def test_distance_backend_validation() -> None:
    """Distance backend and exact-method selectors are restricted to documented values."""
    for backend in ["auto", "decoder", "gap", "sqetch"]:
        core._validate_distance_backend(backend)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown distance backend"):
        core._validate_distance_backend("other")  # type: ignore[arg-type]

    for method in ["brouwer_zimmermann", "brute_force"]:
        assert core._validate_distance_method(method) == method  # type: ignore[arg-type]
    assert core._validate_distance_method("information_sets") == "brouwer_zimmermann"
    with pytest.raises(ValueError, match="Unknown distance method"):
        core._validate_distance_method("other")  # type: ignore[arg-type]
