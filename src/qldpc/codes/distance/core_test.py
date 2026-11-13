# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact binary distances of plain arrays and distance method validation."""

from __future__ import annotations

import copy
import functools
import math
import pickle
from collections.abc import Callable
from typing import Any, cast, get_args
from unittest import mock

import numpy as np
import pytest
import scipy.sparse
from sympy.abc import x

import qldpc
from qldpc.objects import Pauli

from ..conftest import Tune
from . import core
from ._tuning import use_tuning
from .conftest import get_random_full_rank_matrix, get_span


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_information_set_search() -> None:
    """Information-set search excludes stabilizers, finds logical products, and honors cutoffs."""
    # The Hamming-weight-one stabilizer is lighter than every logical operator.  With batches of one
    # support, some Brouwer-Zimmermann batches contain only that stabilizer.
    with use_tuning(information_set_batch_exponent=0):
        distance = core.get_distance_from_generators([[1, 1, 1, 1]], [[1, 0, 0, 0]], cutoff=0)
    assert distance == 3

    # The lightest logical operator is the product of both logical generators.
    distance = core.get_distance_from_generators([[1, 1, 1, 0], [1, 1, 0, 1]], [], cutoff=0)
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
    assert core.get_distance_from_generators(generators, cutoff=0) == 2

    # Both reduced generators have Hamming weight three.  Their weight-two sum reaches the distance
    # cutoff.
    assert core.get_distance_from_generators([[1, 0, 1, 1], [0, 1, 1, 1]], cutoff=2) == 2

    # Row reduction exposes a Hamming-weight-one generator before any search runs.
    assert core.get_distance_from_generators([[1, 1, 0], [0, 1, 1], [1, 1, 1]], cutoff=1) == 1


@pytest.mark.parametrize("method", ["brouwer_zimmermann", "connected_support", "brute_force"])
def test_css_distances(method: core.DistanceMethod) -> None:
    """CSS distances of sectors and of codes, of subsystem codes, and of codes that encode nothing.

    The X and Z sectors of the first code have distances 3 and 2.  The [[9, 1, 3]] Bacon-Shor code,
    given by its gauge generators, has dressed distance 3.  The checks [[1, 1]] and [[1, 1]] leave
    no logical qubit.
    """
    checks_x = [[1, 1, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 0, 0]]
    checks_z = [[1, 1, 0, 1, 0, 0, 1], [1, 1, 1, 0, 0, 1, 0], [1, 0, 1, 1, 1, 0, 0]]
    assert core.get_distance_css(checks_x, checks_z, "X", method=method) == 3
    assert core.get_distance_css(checks_x, checks_z, Pauli.Z, method=method) == 2
    assert core.get_distance_css(checks_x, checks_z, method=method) == 2
    assert core.get_distance_css(checks_x, checks_z, "x", method=method, max_weight=1) == 2
    assert core.get_distance_css(checks_x, checks_z, method=method, max_weight=0) == 1

    bacon_shor_code = qldpc.codes.BaconShorCode(3)
    distance = core.get_distance_css(
        bacon_shor_code.matrix_x, bacon_shor_code.matrix_z, method=method
    )
    assert distance == 3
    assert core.get_distance_css([[1, 1]], [[1, 1]], method=method) is core.NO_DISTANCE

    # These gauge generators leave no stabilizers, so X on one qubit is a dressed logical operator,
    # although the bare logical X operator, X on every qubit, has weight 3.
    gauge_x, gauge_z = [[0, 1, 1], [1, 1, 0]], [[1, 0, 1], [1, 1, 0]]
    assert core.get_distance_css(gauge_x, gauge_z, "X", method=method) == 1

    # Empty X checks take the width of the Z checks.  The X logical operators are then the words of
    # the Hamming code, and Z on any qubit is a logical operator.  Two empty matrices of no qubits
    # leave no logical qubit.
    hamming_checks = qldpc.codes.HammingCode(3).matrix
    assert core.get_distance_css([], hamming_checks, "X", method=method) == 3
    assert core.get_distance_css(np.zeros((0, 0)), hamming_checks, method=method) == 1
    assert core.get_distance_css(np.zeros((0, 0)), [], method=method) is core.NO_DISTANCE


@pytest.mark.parametrize("method", ["brouwer_zimmermann", "connected_support", "brute_force"])
def test_distances_from_checks(method: core.DistanceMethod) -> None:
    """Checks give the distances of classical codes, sectors of CSS codes, and stabilizer codes.

    The stabilizer code is given by Pauli checks.  Excluded rows must satisfy the checks.
    """
    hamming_code = qldpc.codes.HammingCode(3)
    steane_code = qldpc.codes.SteaneCode()
    five_qubit_code = qldpc.codes.FiveQubitCode()
    assert core.get_distance_from_checks(hamming_code.matrix, method=method) == 3
    assert core.get_distance_from_checks(hamming_code.matrix, method=method, max_weight=1) == 2
    distance = core.get_distance_from_checks(
        steane_code.matrix_z, steane_code.matrix_x, method=method
    )
    assert distance == 3
    distance = core.get_distance_from_checks(
        five_qubit_code.matrix, five_qubit_code.matrix, symplectic=True, method=method
    )
    assert distance == 3

    # Excluded rows that span every word that satisfies the checks leave no distance, even when they
    # are redundant, as the repeated row makes them here.
    generator = np.asarray(hamming_code.generator)
    distance = core.get_distance_from_checks(
        hamming_code.matrix, np.vstack([generator, generator[:1]]), method=method
    )
    assert distance is core.NO_DISTANCE
    with pytest.raises(ValueError, match="satisfy the checks"):
        core.get_distance_from_checks([[1, 1, 0]], [[1, 0, 0]], method=method)

    # Empty checks take the width of the excluded rows, and every word satisfies them.
    assert core.get_distance_from_checks([], [[1, 1, 0]], method=method) == 1


@pytest.mark.parametrize("method", ["brouwer_zimmermann", "connected_support", "brute_force"])
def test_distances_of_stabilizer_and_subsystem_codes(method: core.DistanceMethod) -> None:
    """Pauli checks give the distances of stabilizer codes, and dressed ones of subsystem codes.

    The checks of the stabilizer state include a redundant row.
    """
    five_qubit_code = qldpc.codes.FiveQubitCode()
    assert core.get_distance_pauli(five_qubit_code.matrix, method=method) == 3
    assert core.get_distance_pauli(five_qubit_code.matrix, method=method, max_weight=1) == 2
    bacon_shor_code = qldpc.codes.BaconShorCode(3)
    assert core.get_distance_pauli(bacon_shor_code.matrix, method=method) == 3
    stabilizer_state = [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]]
    assert core.get_distance_pauli(stabilizer_state, method=method) is core.NO_DISTANCE

    # The gauge generators of the three-qubit code [[0, 1, 1], [1, 1, 0]] and [[1, 0, 1], [1, 1, 0]]
    # leave no stabilizers.  An S gate on qubit 1 makes them non-CSS, and the last row is redundant.
    # A single-qubit Pauli is a dressed logical operator, so the dressed distance is 1, and the bare
    # recipe, with no stabilizers to exclude, gives 3.
    gauge = [
        [0, 1, 1, 0, 1, 0],
        [1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 1, 0],
        [1, 0, 1, 0, 0, 0],
    ]
    assert core.get_distance_pauli(gauge, method=method) == 1
    no_stabilizers = np.zeros((0, 6), dtype=np.uint8)
    assert core.get_distance_from_checks(gauge, no_stabilizers, symplectic=True, method=method) == 3


@pytest.mark.parametrize("method", ["brouwer_zimmermann", "connected_support", "brute_force"])
def test_bare_and_dressed_distances(method: core.DistanceMethod) -> None:
    """The recipes of get_distance_from_checks give the bare and dressed distances of a code.

    The only stabilizer of this subsystem code is Z on its last qubit, and X on one of its first
    three qubits is a dressed logical operator, but every bare logical operator has weight 3.  So
    excluding no stabilizers would give a bare distance of 1, and excluding those of the wrong type
    would give 1 or raise an error.  The dressed distance of the Bacon-Shor code is 3, whereas
    excluding only part of its gauge group, such as its stabilizers, would give 2.
    """
    code = qldpc.codes.CSSCode(
        [[0, 1, 1, 0], [1, 1, 0, 0]], [[1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]]
    )
    stabilizers = code.get_stabilizer_ops()
    assert (
        core.get_distance_from_checks(stabilizers, code.matrix, symplectic=True, method=method) == 1
    )
    assert (
        core.get_distance_from_checks(code.matrix, stabilizers, symplectic=True, method=method) == 3
    )
    stabilizers_x, stabilizers_z = (
        code.get_stabilizer_ops(Pauli.X),
        code.get_stabilizer_ops(Pauli.Z),
    )
    assert core.get_distance_from_checks(code.matrix_z, stabilizers_x, method=method) == 3
    assert core.get_distance_from_checks(code.matrix_x, stabilizers_z, method=method) == 3

    # The dressed recipe excludes all of the gauge generators of the Bacon-Shor code, in both forms.
    bacon_shor_code = qldpc.codes.BaconShorCode(3)
    distance = core.get_distance_from_checks(
        bacon_shor_code.get_stabilizer_ops(), bacon_shor_code.matrix, symplectic=True, method=method
    )
    assert distance == 3
    distance = core.get_distance_from_checks(
        bacon_shor_code.get_stabilizer_ops(Pauli.Z), bacon_shor_code.matrix_x, method=method
    )
    assert distance == 3

    # Bare logical operators as generators give the dressed distance with the gauge generators
    # excluded, and the bare distance with only the stabilizers excluded.
    if method != "connected_support":
        logical_ops = code.get_logical_ops()
        for excluded, expected in ((code.matrix, 1), (stabilizers, 3)):
            assert (
                core.get_distance_from_generators(
                    logical_ops, excluded, symplectic=True, method=method
                )
                == expected
            )


def test_distance_functions_check_their_input() -> None:
    """The distance functions take sparse matrices, and reject invalid matrices and methods."""
    hamming_code = qldpc.codes.HammingCode(3)
    sparse = scipy.sparse.csr_array(np.asarray(hamming_code.matrix, dtype=int))
    assert core.get_distance_from_checks(sparse) == 3

    checks = [[1, 1, 1, 1]]
    for search in (
        functools.partial(core.get_distance_from_generators, checks),
        functools.partial(core.get_distance_from_checks, checks),
        functools.partial(core.get_distance_css, checks, checks),
        functools.partial(core.get_distance_pauli, checks),
    ):
        for unknown in ("other", ["brouwer_zimmermann"]):
            with pytest.raises(ValueError, match="Unknown distance method"):
                search(method=cast(Any, unknown))
    with pytest.raises(TypeError, match="checks must be an array of binary entries, not a"):
        core.get_distance_from_checks(cast(Any, hamming_code))

    with pytest.raises(ValueError, match="satisfy the checks"):
        core.get_distance_from_checks([[1, 0, 0, 0]], [[0, 0, 1, 0]], symplectic=True)
    with pytest.raises(ValueError, match="need excluded rows"):
        core.get_distance_from_checks(qldpc.codes.FiveQubitCode().matrix, symplectic=True)
    with pytest.raises(ValueError, match="even number of columns"):
        core.get_distance_from_checks([[1, 0, 0]], [[0, 0, 0]], symplectic=True)
    with pytest.raises(ValueError, match="even number of columns"):
        core.get_distance_pauli([[1, 0, 1]])
    with pytest.raises(ValueError, match=r"Expected Pauli\.X or Pauli\.Z"):
        core.get_distance_css(checks, checks, "Y")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="checks_z has 6 columns"):
        core.get_distance_css(checks, [[1, 1, 1, 1, 1, 1]])
    with pytest.raises(ValueError, match="excluded has 0 columns"):
        core.get_distance_from_generators(checks, np.zeros((2, 0), dtype=int))
    with pytest.raises(ValueError, match="checks_z has 5 columns"):
        core.get_distance_css(np.zeros((0, 3)), np.zeros((0, 5)))


def test_automorphisms_are_checked_and_asked_for_once(tune: Tune) -> None:
    """Arrays of automorphisms must be permutations, and a function of them is called at most once.

    The code words of the checks [[1, 1, 0], [0, 1, 1]] are 000 and 111, so the distance is 3, and
    the cyclic shift [1, 2, 0] is an automorphism.  With ``automorphism_minimum_states=0``, a
    connected-support search asks for automorphisms before its first weight limit, so both sectors
    of the [[14, 6, 3]] bicycle code would ask for them.
    """
    tune(automorphism_minimum_states=0)
    checks = [[1, 1, 0], [0, 1, 1]]
    for method in ("brouwer_zimmermann", "connected_support"):
        assert core.get_distance_from_checks(checks, method=method, automorphisms=[[1, 2, 0]]) == 3
    get_automorphisms = mock.Mock(return_value=[[1, 2, 0]])
    distance = core.get_distance_from_checks(
        checks, method="connected_support", automorphisms=get_automorphisms
    )
    assert distance == 3
    get_automorphisms.assert_called_once()

    # The searches of both sectors of a CSS code share one call.
    code = qldpc.codes.MBCode([7], 1 + x + x**3, 1 + x**2 + x**3 + x**4)
    get_automorphisms = mock.Mock(return_value=code._get_qudit_automorphisms())
    distance = core.get_distance_css(
        code.matrix_x, code.matrix_z, method="connected_support", automorphisms=get_automorphisms
    )
    assert distance == 3
    get_automorphisms.assert_called_once()

    for automorphisms in ([[1, 0]], [[0, 0, 1]], [[[1, 2, 0]]]):
        with pytest.raises(ValueError, match="permutations of range"):
            core.get_distance_from_checks(checks, automorphisms=automorphisms)
    with pytest.raises(ValueError, match="permutations of range"):
        core.get_distance_from_checks(
            checks, method="connected_support", automorphisms=lambda: [[0, 0, 1]]
        )


def test_connected_support_needs_parity_checks() -> None:
    """Generator matrices do not describe Tanner graphs, so their function rejects the method."""
    with pytest.raises(ValueError, match="use get_distance_from_checks"):
        core.get_distance_from_generators([[1, 1, 0]], method="connected_support")


def test_special_inputs_of_generators() -> None:
    """The shapes of the matrices are checked, and inputs without nontrivial words have no distance.

    Inputs without nontrivial words give NO_DISTANCE, whether they have no logical rows or only
    logical rows in the stabilizer span.
    """
    no_distance = qldpc.codes.NO_DISTANCE
    assert (
        core.get_distance_from_generators(np.empty((0, 4), dtype=np.uint8), cutoff=0) is no_distance
    )
    # Generators in the row space of the excluded rows leave no distance, and so do no generators,
    # which take the width of the excluded rows.
    for method in ("brouwer_zimmermann", "brute_force"):
        distance = core.get_distance_from_generators([[1, 0]], [[1, 0]], method=method)
        assert distance is no_distance
    assert core.get_distance_from_generators([], [[1, 0]]) is no_distance
    no_logical_ops = np.empty((0, 4), dtype=np.uint8)
    assert (
        core.get_distance_from_generators(no_logical_ops, [[1, 0, 0, 0]], symplectic=True)
        is no_distance
    )
    assert (
        core.get_distance_from_generators([[1, 0, 0, 0]], [[1, 0, 0, 0]], symplectic=True)
        is no_distance
    )

    with pytest.raises(ValueError, match="excluded has 2 columns"):
        core.get_distance_from_generators([[1, 0, 0]], [[1, 0]])
    with pytest.raises(ValueError, match="even number of columns"):
        core.get_distance_from_generators([[1, 0, 0]], [], symplectic=True)
    with pytest.raises(ValueError, match="even number of columns"):
        core.get_distance_from_generators(
            np.empty((0, 3), dtype=np.uint8),
            np.empty((0, 3), dtype=np.uint8),
            symplectic=True,
        )
    with pytest.raises(ValueError, match="generators must be a matrix or a single row"):
        core.get_distance_from_generators(np.zeros((1, 1, 1), dtype=np.uint8))
    with pytest.raises(ValueError, match="infer the width of generators"):
        core.get_distance_from_generators([])
    with pytest.raises(ValueError, match="excluded must be a matrix or a single row"):
        core.get_distance_from_generators(
            [[1, 0]], np.zeros((1, 1, 2), dtype=np.uint8), symplectic=True
        )


@pytest.mark.parametrize(
    "code, expected_distance",
    [
        (qldpc.codes.classical.HammingCode(4), 3),
        (qldpc.codes.classical.RepetitionCode(3), 3),
        (qldpc.codes.classical.RepetitionCode(8), 8),
        (qldpc.codes.classical.RingCode(8), 8),
    ],
)
def test_distances_of_known_classical_codes(
    code: qldpc.codes.ClassicalCode, expected_distance: int
) -> None:
    """Classical distances match known values for Hamming, repetition, and ring codes."""
    assert core.get_distance_from_generators(code.generator) == expected_distance
    assert core.get_distance_from_checks(code.matrix) == expected_distance

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
def test_distances_of_known_css_codes(code: qldpc.codes.CSSCode, expected_distance: int) -> None:
    """Quantum distances match known values for C4, C6, Steane, and small surface/toric codes."""
    distance_x = core.get_distance_from_generators(
        code.get_logical_ops(qldpc.objects.Pauli.X),
        code.get_stabilizer_ops(qldpc.objects.Pauli.X),
    )
    distance_z = core.get_distance_from_generators(
        code.get_logical_ops(qldpc.objects.Pauli.Z),
        code.get_stabilizer_ops(qldpc.objects.Pauli.Z),
    )
    distance_all = core.get_distance_from_generators(
        code.get_logical_ops(), code.get_stabilizer_ops(), symplectic=True
    )
    distance_css = core.get_distance_css(code.matrix_x, code.matrix_z)
    assert distance_x == distance_z == distance_all == distance_css == expected_distance

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
def test_distances_of_known_noncss_codes(
    code: qldpc.codes.QuditCode, expected_distance: int
) -> None:
    """Quantum distances match known values for small non-CSS codes."""
    distance = core.get_distance_from_generators(
        code.get_logical_ops(),
        code.get_stabilizer_ops(),
        symplectic=True,
    )
    assert distance == core.get_distance_pauli(code.matrix) == expected_distance

    code._distance = None
    assert code.get_distance_exact() == expected_distance


@pytest.mark.usefixtures("force_information_sets")
def test_quantum_distances_use_qubit_automorphisms(tune: Tune) -> None:
    """Searches use qubit permutations for binary words, and lift them for Pauli strings.

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
        (code.get_logical_ops(), code.get_stabilizer_ops(), True),
        (bicycle_code.get_logical_ops("X"), bicycle_code.get_stabilizer_ops("X"), False),
    ]
    for logical_ops, stabilizers, symplectic in problems:
        search = functools.partial(
            core.get_distance_from_generators, logical_ops, stabilizers, symplectic=symplectic
        )
        expected = search(method="brute_force")
        for automorphisms in (translations, None):
            get_automorphisms = mock.Mock(return_value=automorphisms)
            assert search(automorphisms=get_automorphisms) == expected == 3
            get_automorphisms.assert_called_once()
        with pytest.raises(ValueError, match="not an automorphism"):
            search(automorphisms=transposition[None])


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_random_cross_checks() -> None:
    """Information-set search agrees with exhaustive enumeration on random nested codes."""
    rng = np.random.default_rng(118)
    for length in range(5, 10):
        basis = get_random_full_rank_matrix(rng, min(5, length - 1), length)
        expected = core.get_distance_from_generators(basis, cutoff=0, method="brute_force")
        assert core.get_distance_from_generators(basis, cutoff=0) == expected

        stabilizers, logical_ops = basis[:2], basis[2:]
        expected = core.get_distance_from_generators(
            logical_ops, stabilizers, cutoff=0, method="brute_force"
        )
        assert core.get_distance_from_generators(logical_ops, stabilizers, cutoff=0) == expected

    for num_qubits in range(3, 7):
        basis = get_random_full_rank_matrix(rng, min(6, 2 * num_qubits - 1), 2 * num_qubits)
        stabilizers, logical_ops = basis[:2], basis[2:]
        expected = core.get_distance_from_generators(
            logical_ops, stabilizers, symplectic=True, cutoff=0, method="brute_force"
        )
        assert (
            core.get_distance_from_generators(logical_ops, stabilizers, symplectic=True, cutoff=0)
            == expected
        )


def test_max_weight(request: pytest.FixtureRequest) -> None:
    """A weight limit returns the smaller of the distance and max_weight + 1, on every route."""
    rng = np.random.default_rng(7)
    basis = get_random_full_rank_matrix(rng, 4, 7)
    classical = functools.partial(core.get_distance_from_generators, basis)
    classical_distance = min(word.bit_count() for word in get_span(basis) if word)
    css = functools.partial(core.get_distance_from_generators, basis[2:], basis[:2])
    css_distance = min(word.bit_count() for word in get_span(basis) - get_span(basis[:2]))
    basis = get_random_full_rank_matrix(rng, 5, 8)
    pauli = functools.partial(
        core.get_distance_from_generators, basis[2:], basis[:2], symplectic=True
    )
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
    assert core.get_distance_from_generators(basis, cutoff=6, max_weight=3) == 3


def test_max_weight_conventions_and_validation() -> None:
    """The limit leaves NO_DISTANCE, for inputs without nontrivial words, unchanged."""
    no_distance = qldpc.codes.NO_DISTANCE
    assert core.get_distance_from_generators([[0, 0, 0]], max_weight=1) is no_distance
    no_logical_ops = np.zeros((0, 4), dtype=int)
    assert (
        core.get_distance_from_generators(
            no_logical_ops, [[1, 1, 0, 0]], symplectic=True, max_weight=0
        )
        is no_distance
    )
    assert (
        core.get_distance_from_generators(
            [[1, 1, 0, 0]], [[1, 1, 0, 0]], symplectic=True, max_weight=1
        )
        is no_distance
    )
    with pytest.raises(ValueError, match="max_weight must be nonnegative"):
        core.get_distance_from_generators([[1, 1]], max_weight=-1)

    # The limit is an integer, and a NumPy integer gives a Python integer, even on a route that
    # clamps its result.
    for not_an_integer in (2.5, True):
        with pytest.raises(TypeError):
            core.get_distance_from_generators([[1, 1]], max_weight=cast(int, not_an_integer))
    distance = core.get_distance_from_generators(
        [[1, 1, 1, 1]], max_weight=np.int64(1), method="brute_force"
    )
    assert distance == 2 and type(distance) is int


def test_distance_method_validation() -> None:
    """Exact-method selectors are restricted to documented values, and resolve aliases."""
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
    assert qldpc.codes.NoDistance is qldpc.codes.distance.NoDistance is core.NoDistance
    assert qldpc.codes.NO_DISTANCE is qldpc.codes.distance.NO_DISTANCE is core.NoDistance.NAN
    assert isinstance(no_distance, float) and math.isnan(no_distance)
    assert str(no_distance) == f"{no_distance}" == repr(no_distance) == "nan"
    assert repr((2, 0, no_distance)) == "(2, 0, nan)"
    assert not no_distance >= 3 and no_distance != core.NO_DISTANCE
    restored = pickle.loads(pickle.dumps(no_distance))  # noqa: S301 - trusted in-memory round trip
    assert restored is no_distance
    assert copy.copy(no_distance) is copy.deepcopy(no_distance) is no_distance
    assert core.NoDistance(math.nan) is no_distance
    with pytest.raises(ValueError, match="not a valid NoDistance"):
        core.NoDistance(1.0)
