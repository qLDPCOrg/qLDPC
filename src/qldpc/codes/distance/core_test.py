# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact binary distance calculations and distance method validation."""

from __future__ import annotations

import copy
import functools
import math
import pickle
from collections.abc import Callable
from typing import cast, get_args
from unittest import mock

import numpy as np
import pytest
from sympy.abc import x

import qldpc

from ..conftest import Tune
from . import core
from ._tuning import use_tuning
from .conftest import get_random_full_rank_matrix, get_span


def test_deprecated_block_size_argument() -> None:
    """The argument warns, and gives the batches that it gave before it was deprecated.

    Exhaustive enumeration took ``block_size`` as its exponent, and scans of information sets took
    at most it, so the argument sets both fields, clamped to their range.
    """
    with use_tuning(information_set_batch_exponent=10) as active:
        assert core._get_tuning(None) is active
        for block_size, batches in (
            (4, (4, 4)),
            (12, (12, 10)),
            (100, (62, 10)),
            (-3, (0, 0)),
            (True, (1, 1)),
        ):
            with pytest.warns(DeprecationWarning, match="use_tuning"):
                tuning = core._get_tuning(block_size)
            assert (tuning.enumeration_batch_exponent, tuning.information_set_batch_exponent) == (
                batches
            )

    # Both functions, by either method, warn at their caller and still find the distance.
    generator = np.asarray(qldpc.codes.HammingCode(3).generator)
    logical_ops, stabilizers = generator[:1], generator[1:]
    expected = qldpc.codes.get_distance_quantum(
        logical_ops, stabilizers, homogeneous=True, method="brute_force"
    )
    searches: list[Callable[[], float]] = [
        lambda: qldpc.codes.get_distance_classical(generator, block_size=2),
        lambda: qldpc.codes.get_distance_classical(generator, block_size=2, method="brute_force"),
        lambda: qldpc.codes.get_distance_quantum(
            logical_ops, stabilizers, homogeneous=True, block_size=2
        ),
        lambda: qldpc.codes.get_distance_quantum(
            logical_ops, stabilizers, homogeneous=True, block_size=2, method="brute_force"
        ),
    ]
    for search, distance in zip(searches, (3, 3, expected, expected), strict=True):
        with pytest.warns(DeprecationWarning) as records:
            assert search() == distance
        assert records[0].filename == __file__


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_information_set_search() -> None:
    """Information-set search excludes stabilizers, finds logical products, and honors cutoffs."""
    # The Hamming-weight-one stabilizer is lighter than every logical operator.  With batches of one
    # support, some Brouwer-Zimmermann batches contain only that stabilizer.
    with use_tuning(information_set_batch_exponent=0):
        distance = qldpc.codes.get_distance_quantum(
            [[1, 1, 1, 1]], [[1, 0, 0, 0]], homogeneous=True, cutoff=0
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


def test_css_brouwer_zimmermann_special_inputs() -> None:
    """The joint CSS helper handles empty, trivial, and nontrivial quotient sectors.

    A sector without nontrivial logical operators counts as one more than its width, so that it
    never lowers the distance of the other sectors.
    """
    empty = np.empty((0, 4), dtype=np.uint8)
    assert core._get_distance_css_brouwer_zimmermann([(empty, empty)], cutoff=0) == 5
    trivial = ([[1, 0]], [[1, 0]])
    assert core._get_distance_css_brouwer_zimmermann([trivial], cutoff=0) == 3
    assert core._get_distance_css_brouwer_zimmermann([trivial, ([[1, 1]], [])], cutoff=0) == 2
    # It is not the weight of an operator, so it does not end the search even at the cutoff.
    assert core._get_distance_css_brouwer_zimmermann([trivial, ([[1, 1]], [])], cutoff=3) == 2

    # The result is the minimum of the upper bound and the distances of the supplied sectors.
    assert core._get_distance_css_brouwer_zimmermann([([[1, 1]], [])], cutoff=0) == 2
    sectors: list[tuple[list[list[int]], list[list[int]]]] = [([[1, 1]], [])]
    assert core._get_distance_css_brouwer_zimmermann(sectors, cutoff=0, upper_bound=3) == 2
    assert core._get_distance_css_brouwer_zimmermann(sectors, cutoff=0, upper_bound=1) == 1

    # A known operator at the cutoff ends the search before it starts.
    assert core._get_distance_css_brouwer_zimmermann(sectors, cutoff=1, upper_bound=1) == 1
    assert core._get_distance_css_brouwer_zimmermann([], cutoff=0, upper_bound=4) == 4
    with pytest.raises(ValueError, match="at least one sector or an upper bound"):
        core._get_distance_css_brouwer_zimmermann([], cutoff=0)


def test_connected_support_problems_are_searched_with_the_active_tuning(tune: Tune) -> None:
    """The connected-support helper searches lists of checks and excluded rows, with the tuning.

    The words of the checks [[1, 1, 0], [0, 1, 1]] are 000 and 111, so the distance of a problem
    that excludes nothing is 3.  With ``automorphism_minimum_states=0``, the search asks for
    automorphisms before its first weight limit.
    """
    tune(automorphism_minimum_states=0)
    get_automorphisms = mock.Mock(return_value=None)
    problem = ([[1, 1, 0], [0, 1, 1]], [[0, 0, 0]])
    assert (
        core._get_distance_connected_support(
            [problem], cutoff=0, get_automorphisms=get_automorphisms
        )
        == 3
    )
    get_automorphisms.assert_called_once()


def test_connected_support_needs_parity_checks() -> None:
    """Generator matrices do not describe Tanner graphs, so these functions reject the method."""
    with pytest.raises(ValueError, match="get_distance_exact method of a code"):
        qldpc.codes.get_distance_classical([[1, 1, 0]], method="connected_support")
    with pytest.raises(ValueError, match="get_distance_exact method of a code"):
        qldpc.codes.get_distance_quantum([[1, 0]], [[1, 1]], method="connected_support")


def test_brouwer_zimmermann_special_inputs() -> None:
    """Brouwer-Zimmermann validates matrix shapes and preserves quotient edge cases.

    Inputs without nontrivial words have no distance and give NO_DISTANCE, whether they have no
    logical rows or only logical rows in the stabilizer span.
    """
    no_distance = qldpc.codes.NO_DISTANCE
    assert qldpc.codes.get_distance_classical(np.empty((0, 4), dtype=np.uint8), cutoff=0) is (
        no_distance
    )
    assert (
        qldpc.codes.get_distance_quantum(
            np.empty((0, 4), dtype=np.uint8),
            np.empty((0, 4), dtype=np.uint8),
            homogeneous=True,
        )
        is no_distance
    )
    for method in ("brouwer_zimmermann", "brute_force"):
        distance = qldpc.codes.get_distance_quantum(
            [[1, 0]], [[1, 0]], homogeneous=True, method=method
        )
        assert distance is no_distance
    no_logical_ops = np.empty((0, 4), dtype=np.uint8)
    assert qldpc.codes.get_distance_quantum(no_logical_ops, [[1, 0, 0, 0]]) is no_distance
    assert qldpc.codes.get_distance_quantum([[1, 0, 0, 0]], [[1, 0, 0, 0]]) is no_distance

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
    assert distance_x == distance_z == distance_all == expected_distance

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


@pytest.mark.usefixtures("force_information_sets")
def test_quantum_distances_use_qubit_automorphisms(tune: Tune) -> None:
    """Searches use qubit permutations for homogeneous operators, and lift them for Pauli operators.

    Hadamards on the second block of this [[14, 6, 3]] multivariate bicycle code make a code that is
    not CSS, which translations of both blocks still preserve.  A wrong permutation fails loudly.
    """
    tune(automorphism_minimum_layer_work=0)
    bicycle_code = qldpc.codes.MBCode([7], 1 + x + x**3, 1 + x**2 + x**3 + x**4)
    translations = bicycle_code._get_qudit_automorphisms()
    transposition = np.arange(len(bicycle_code))
    transposition[[0, 1]] = [1, 0]
    code = bicycle_code.conjugated(range(7, 14))
    problems = [
        (code.get_logical_ops(), code.get_stabilizer_ops(), False),
        (bicycle_code.get_logical_ops("X"), bicycle_code.get_stabilizer_ops("X"), True),
    ]
    for logical_ops, stabilizers, homogeneous in problems:
        search = functools.partial(
            core._get_distance_quantum, logical_ops, stabilizers, homogeneous=homogeneous
        )
        expected = search(method="brute_force")
        for automorphisms in (translations, None):
            get_automorphisms = mock.Mock(return_value=automorphisms)
            assert search(get_automorphisms=get_automorphisms) == expected == 3
            get_automorphisms.assert_called()
        with pytest.raises(ValueError, match="not an automorphism"):
            search(get_automorphisms=lambda: transposition[None])


@pytest.mark.usefixtures("force_information_sets")
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

    def check(distance_with_limit: Callable[..., float], distance: int, method: str) -> None:
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
    """The limit leaves NO_DISTANCE, for inputs without nontrivial words, unchanged."""
    no_distance = qldpc.codes.NO_DISTANCE
    assert qldpc.codes.get_distance_classical([[0, 0, 0]], max_weight=1) is no_distance
    no_logical_ops = np.zeros((0, 4), dtype=int)
    assert (
        qldpc.codes.get_distance_quantum(no_logical_ops, [[1, 1, 0, 0]], max_weight=0)
        is no_distance
    )
    assert (
        qldpc.codes.get_distance_quantum([[1, 1, 0, 0]], [[1, 1, 0, 0]], max_weight=1)
        is no_distance
    )
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
        [[1, 1, 1, 1]], max_weight=np.int64(1), method="brute_force"
    )
    assert distance == 2 and type(distance) is int


def test_distance_backend_validation() -> None:
    """Distance backend and exact-method selectors are restricted to documented values."""
    for backend in ["auto", "decoder", "gap", "sqetch"]:
        core._validate_distance_backend(backend)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown distance backend"):
        core._validate_distance_backend("other")  # type: ignore[arg-type]

    for method in ["brouwer_zimmermann", "brute_force", "connected_support"]:
        assert core._validate_distance_method(method) == method  # type: ignore[arg-type]
    assert core._validate_distance_method("information_sets") == "brouwer_zimmermann"
    with pytest.raises(ValueError, match="Unknown distance method"):
        core._validate_distance_method("other")  # type: ignore[arg-type]

    # The types of method names agree with the table that validates them.
    assert set(get_args(core.DistanceMethod)) == set(core._DISTANCE_METHODS)
    assert set(get_args(core._CanonicalDistanceMethod)) == set(core._DISTANCE_METHODS.values())


def test_no_distance_behaves_as_nan_and_survives_copies() -> None:
    """NO_DISTANCE is a float nan, and copies and pickles of it are the same member."""
    no_distance = core.NO_DISTANCE
    assert isinstance(no_distance, float) and math.isnan(no_distance)
    assert str(no_distance) == f"{no_distance}" == repr(no_distance) == "nan"
    assert qldpc.codes.NoDistance is qldpc.codes.distance.NoDistance is core.NoDistance
    assert qldpc.codes.NO_DISTANCE is qldpc.codes.distance.NO_DISTANCE is core.NoDistance.NAN
    assert repr((2, 0, no_distance)) == "(2, 0, nan)"
    assert not no_distance >= 3 and no_distance != core.NO_DISTANCE
    restored = pickle.loads(pickle.dumps(no_distance))  # noqa: S301 - trusted in-memory round trip
    assert restored is no_distance
    assert copy.copy(no_distance) is copy.deepcopy(no_distance) is no_distance
    assert core.NoDistance(math.nan) is no_distance
    with pytest.raises(ValueError, match="not a valid NoDistance"):
        core.NoDistance(1.0)
