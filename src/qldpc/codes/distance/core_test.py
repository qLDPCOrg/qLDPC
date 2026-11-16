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
import numpy.typing as npt
import pytest
import scipy.linalg
import scipy.sparse
from sympy.abc import x

import qldpc
from qldpc._binary import get_independent_rows
from qldpc.objects import Pauli

from ..conftest import Tune
from . import core
from ._binary import get_css_stabilizers
from ._tuning import use_tuning
from .conftest import get_random_full_rank_matrix, get_span


def test_brouwer_zimmermann_information_set_search() -> None:
    """Information-set search excludes stabilizers, finds logical products, and honors cutoffs."""
    # The stabilizer of weight one is lighter than every logical operator, and with one support per
    # batch, some batches hold only that stabilizer.
    with use_tuning(information_set_batch_exponent=0):
        distance = core.get_distance_from_generators([[1, 1, 1, 1]], [[1, 0, 0, 0]], cutoff=0)
    assert distance == 3

    # The lightest logical operator is the product of both logical generators.
    distance = core.get_distance_from_generators([[1, 1, 1, 0], [1, 1, 0, 1]], [], cutoff=0)
    assert distance == 2

    # Every row of every information set weighs at least three, so the codeword of weight two needs
    # combinations of two rows.
    generators = [
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
        [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
    ]
    assert core.get_distance_from_generators(generators, cutoff=0) == 2

    # Both reduced generators weigh three, and their sum of weight two reaches the cutoff.
    assert core.get_distance_from_generators([[1, 0, 1, 1], [0, 1, 1, 1]], cutoff=2) == 2

    # Row reduction exposes a Hamming-weight-one generator before any search runs.
    assert core.get_distance_from_generators([[1, 1, 0], [0, 1, 1], [1, 1, 1]], cutoff=1) == 1


@pytest.mark.parametrize("method", ["brouwer_zimmermann", "connected_cluster", "brute_force"])
def test_css_distances(method: core.DistanceMethod) -> None:
    """CSS distances of sectors, of codes, of subsystem codes, and of codes that encode nothing."""
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

    # Empty X checks take the width of the Z checks, so the X logical operators are the words of the
    # Hamming code, and Z on any qubit is a logical operator.
    hamming_checks = qldpc.codes.HammingCode(3).matrix
    assert core.get_distance_css([], hamming_checks, "X", method=method) == 3
    assert core.get_distance_css(np.zeros((0, 0)), hamming_checks, method=method) == 1
    assert core.get_distance_css(np.zeros((0, 0)), [], method=method) is core.NO_DISTANCE


@pytest.mark.parametrize("method", ["brouwer_zimmermann", "connected_cluster", "brute_force"])
def test_distances_from_checks(method: core.DistanceMethod) -> None:
    """Checks give classical, CSS, and stabilizer distances, and excluded rows must satisfy them."""
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


@pytest.mark.parametrize("method", ["brouwer_zimmermann", "connected_cluster", "brute_force"])
def test_distances_of_stabilizer_and_subsystem_codes(method: core.DistanceMethod) -> None:
    """Pauli checks give the distances of stabilizer codes, and dressed ones of subsystem codes."""
    five_qubit_code = qldpc.codes.FiveQubitCode()
    assert core.get_distance_pauli(np.vstack([five_qubit_code.matrix] * 2), method=method) == 3
    assert core.get_distance_pauli(five_qubit_code.matrix, method=method, max_weight=1) == 2
    bacon_shor_code = qldpc.codes.BaconShorCode(3)
    assert core.get_distance_pauli(bacon_shor_code.matrix, method=method) == 3
    assert core.get_distance_pauli(bacon_shor_code.matrix, method=method, max_weight=1) == 2
    stabilizer_state = [[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 1, 1]]
    assert core.get_distance_pauli(stabilizer_state, method=method) is core.NO_DISTANCE

    # The CSS gauge generators [[0, 1, 1], [1, 1, 0]] and [[1, 0, 1], [1, 1, 0]] leave no
    # stabilizers.  An S gate on qubit 1 makes them non-CSS, and the last row of gauge is redundant.
    # A single-qubit Pauli is a dressed logical operator, but every bare one weighs 3.
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


def test_pauli_checks_of_css_codes_are_searched_as_css_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pauli checks that generate a CSS group are searched as CSS, even if they mix X and Z."""
    # Adding the Z-type checks of the Steane code to its X-type checks mixes X and Z in the rows,
    # but keeps the group that they generate.
    steane_code = qldpc.codes.SteaneCode()
    checks_x = steane_code.matrix_x.view(np.ndarray)
    checks_z = steane_code.matrix_z.view(np.ndarray)
    checks = np.block([[checks_x, checks_z], [np.zeros_like(checks_z), checks_z]])

    searched: list[tuple[list[int], bool]] = []

    def record(sectors: Any, **kwargs: Any) -> Any:
        widths = [np.shape(sector_checks)[1] for sector_checks, _ in sectors]
        searched.append((widths, kwargs.get("symplectic", False)))
        return get_distance_connected_cluster(sectors, **kwargs)

    get_distance_connected_cluster = core.get_distance_connected_cluster
    monkeypatch.setattr(core, "get_distance_connected_cluster", record)
    assert core.get_distance_pauli(checks, method="connected_cluster") == 3
    assert searched == [([7, 7], False)]


@pytest.mark.parametrize("method", ["brouwer_zimmermann", "connected_cluster", "brute_force"])
def test_bare_and_dressed_distances(method: core.DistanceMethod) -> None:
    """The recipes of get_distance_from_checks give the bare and dressed distances of a code."""
    # The only stabilizer of this subsystem code is Z on its last qubit, and X on one of its first
    # three qubits is a dressed logical operator, but every bare one weighs 3.
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

    # The dressed recipe excludes all of the gauge generators of the Bacon-Shor code, as Pauli
    # strings and as X-type rows; excluding only its stabilizers would give 2.
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
    if method != "connected_cluster":
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
    """Automorphisms must be permutations, and a function returning them is called at most once."""
    # With automorphism_minimum_states=0, a connected-cluster search asks for automorphisms before
    # its first weight limit, so both sectors of the bicycle code below would ask.  The code words
    # of the checks below are 000 and 111, so the cyclic shift [1, 2, 0] is an automorphism.
    tune(automorphism_minimum_states=0)
    checks = [[1, 1, 0], [0, 1, 1]]
    for method in ("brouwer_zimmermann", "connected_cluster"):
        assert core.get_distance_from_checks(checks, method=method, automorphisms=[[1, 2, 0]]) == 3
    get_automorphisms = mock.Mock(return_value=[[1, 2, 0]])
    distance = core.get_distance_from_checks(
        checks, method="connected_cluster", automorphisms=get_automorphisms
    )
    assert distance == 3
    get_automorphisms.assert_called_once()

    # The searches of both sectors of a CSS code share one call.
    code = qldpc.codes.MBCode([7], 1 + x + x**3, 1 + x**2 + x**3 + x**4)
    get_automorphisms = mock.Mock(return_value=code._get_qudit_automorphisms())
    distance = core.get_distance_css(
        code.matrix_x, code.matrix_z, method="connected_cluster", automorphisms=get_automorphisms
    )
    assert distance == 3
    get_automorphisms.assert_called_once()

    for automorphisms in ([[1, 0]], [[0, 0, 1]], [[[1, 2, 0]]]):
        with pytest.raises(ValueError, match="permutations of range"):
            core.get_distance_from_checks(checks, automorphisms=automorphisms)
    with pytest.raises(ValueError, match="permutations of range"):
        core.get_distance_from_checks(
            checks, method="connected_cluster", automorphisms=lambda: [[0, 0, 1]]
        )


def test_connected_cluster_searches_sparse_stabilizers(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connected-cluster search of a subsystem code searches sparse checks of its stabilizers."""
    # With X and Z on a bulk qubit of the [[16, 1, 4]] surface code as gauge generators, the two
    # checks of each type on that qubit merge into one stabilizer of weight 6, the dressed distance
    # drops to 3, and the row-reduced stabilizers have rows of weight 8.  The other checks are
    # stabilizers, but they do not span the stabilizer group.
    code = qldpc.codes.SurfaceCode(4)
    qubit = 5
    gauge_qubit = np.zeros((1, len(code)), dtype=np.uint8)
    gauge_qubit[0, qubit] = 1
    checks_x = code.matrix_x.view(np.ndarray).astype(np.uint8)
    checks_z = code.matrix_z.view(np.ndarray).astype(np.uint8)
    gauge_x, gauge_z = np.vstack([checks_x, gauge_qubit]), np.vstack([checks_z, gauge_qubit])
    derived_x, derived_z = get_css_stabilizers(gauge_x, gauge_z)
    assert derived_x.sum(axis=1).max() == derived_z.sum(axis=1).max() == 8

    # The stabilizers are the checks off the gauge qubit, and the product of those on it.
    def get_stabilizers(checks: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        on_qubit = checks[:, qubit] == 1
        return np.vstack([checks[~on_qubit], checks[on_qubit].sum(axis=0) % 2])

    searched: list[npt.NDArray[np.uint8]] = []

    def record(sectors: Any, **kwargs: Any) -> Any:
        searched.extend(np.asarray(checks) for checks, _ in sectors)
        return get_distance_connected_cluster(sectors, **kwargs)

    get_distance_connected_cluster = core.get_distance_connected_cluster
    monkeypatch.setattr(core, "get_distance_connected_cluster", record)
    distance = core.get_distance_css(gauge_x, gauge_z, method="connected_cluster")
    assert distance == core.get_distance_css(gauge_x, gauge_z, method="brute_force") == 3
    for checks, stabilizers in zip(
        searched, (get_stabilizers(checks_z), get_stabilizers(checks_x)), strict=True
    ):
        assert checks.sum(axis=1).max() <= 6
        assert set(map(tuple, stabilizers[:-1])) <= set(map(tuple, checks))
        assert np.array_equal(get_independent_rows(checks), get_independent_rows(stabilizers))
    assert core.get_distance_css(gauge_x, gauge_z, Pauli.Z, method="connected_cluster") == 3

    searched.clear()
    assert core.get_distance_css(checks_x, checks_z, method="connected_cluster") == 4
    assert [checks.tolist() for checks in searched] == [checks_z.tolist(), checks_x.tolist()]

    # Hadamards on half of the qubits make a Pauli form that is not CSS.
    searched.clear()
    qubits = np.arange(0, len(code), 2)
    columns, swapped = np.r_[qubits, qubits + len(code)], np.r_[qubits + len(code), qubits]

    def apply_hadamards(rows: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        rows = rows.copy()
        rows[:, columns] = rows[:, swapped]
        return rows

    gauge = apply_hadamards(scipy.linalg.block_diag(gauge_x, gauge_z))
    assert core.get_distance_pauli(gauge, method="connected_cluster") == 3
    (checks,) = searched
    assert (checks[:, : len(code)] | checks[:, len(code) :]).sum(axis=1).max() <= 6
    stabilizers = scipy.linalg.block_diag(get_stabilizers(checks_x), get_stabilizers(checks_z))
    assert np.array_equal(
        get_independent_rows(checks), get_independent_rows(apply_hadamards(stabilizers))
    )
    kept = scipy.linalg.block_diag(get_stabilizers(checks_x)[:-1], get_stabilizers(checks_z)[:-1])
    assert set(map(tuple, apply_hadamards(kept))) <= set(map(tuple, checks))


def test_connected_cluster_needs_parity_checks() -> None:
    """get_distance_from_generators rejects the connected-cluster method, which needs checks."""
    with pytest.raises(ValueError, match="use get_distance_from_checks"):
        core.get_distance_from_generators([[1, 1, 0]], method="connected_cluster")


def test_special_inputs_of_generators() -> None:
    """Matrix shapes are checked, and inputs without nontrivial words have no distance."""
    no_distance = qldpc.codes.NO_DISTANCE
    assert (
        core.get_distance_from_generators(np.empty((0, 4), dtype=np.uint8), cutoff=0) is no_distance
    )
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

    # CSSCode.forget_distance also clears the cached distances of the X and Z sectors, which the
    # method would otherwise reuse.
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


def test_quantum_distances_use_qubit_automorphisms(tune: Tune) -> None:
    """Searches use qubit permutations, lifted for Pauli strings, and reject non-automorphisms."""
    # Hadamards on the second block of this bicycle code make a code that is not CSS, which the
    # translations of both blocks still preserve.
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


def test_max_weight() -> None:
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

    # The Pauli case checks that the BZ route, which searches binary words of twice the weight of
    # the Pauli strings, doubles the bound.
    check(pauli, pauli_distance, "brouwer_zimmermann")


def test_max_weight_caps_cutoff() -> None:
    """A large cutoff cannot make the search give up on a word within the limit."""
    # Both rows weigh more than the limit, but their sum weighs 3.
    basis = [[1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0]]
    assert core.get_distance_from_generators(basis, cutoff=6, max_weight=3) == 3


def test_max_weight_conventions_and_validation() -> None:
    """The limit leaves NO_DISTANCE unchanged, and must be a nonnegative integer."""
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

    for not_an_integer in (2.5, True):
        with pytest.raises(TypeError):
            core.get_distance_from_generators([[1, 1]], max_weight=cast(int, not_an_integer))
    # A NumPy limit gives a Python integer, even on a route that clamps its result.
    distance = core.get_distance_from_generators(
        [[1, 1, 1, 1]], max_weight=np.int64(1), method="brute_force"
    )
    assert distance == 2 and type(distance) is int


def test_distance_method_validation() -> None:
    """Exact-method selectors are restricted to documented values, and resolve aliases."""
    for method in ["brouwer_zimmermann", "brute_force", "connected_cluster"]:
        assert core._validate_distance_method(method) == method  # type: ignore[arg-type]
    assert core._validate_distance_method("information_sets") == "brouwer_zimmermann"
    with pytest.raises(ValueError, match="Unknown distance method"):
        core._validate_distance_method("other")  # type: ignore[arg-type]

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
