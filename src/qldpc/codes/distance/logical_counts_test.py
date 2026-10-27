# SPDX-License-Identifier: Apache-2.0

"""Tests for exact counts of the logical operators of a given weight in binary CSS codes."""

from __future__ import annotations

import functools
import itertools
import math
from dataclasses import replace
from typing import Any
from unittest import mock

import galois
import numba.core.event
import numpy as np
import numpy.typing as npt
import pytest

import qldpc
from qldpc import _compilation, codes

from . import _information_sets
from . import logical_counts as lc
from ._binary import get_independent_rows
from ._counting import macwilliams
from .conftest import (
    HAND_X,
    HAND_Z,
    METHODS,
    Counts,
    append_frozen_qubits,
    enumerate_logical_ops,
    get_counter,
    get_expected_counts,
    get_random_css_code,
)
from .core import DistanceMethod

STABILIZER_METHODS: tuple[lc.LogicalCountMethod, ...] = (
    "brouwer_zimmermann",
    "macwilliams",
    "brute_force",
)


def _random_subsystem_code(
    rng: np.random.Generator, num_qubits: int, *, self_dual: bool = False
) -> codes.CSSCode:
    """A random binary CSS subsystem code, optionally with identical X and Z checks."""
    while True:
        checks_x = rng.integers(0, 2, (3, num_qubits))
        checks_z = checks_x if self_dual else rng.integers(0, 2, (3, num_qubits))
        code = codes.CSSCode(checks_x, checks_z)
        if code.is_subsystem_code and code.dimension > 0:
            return code


####################################################################################################
# counts


@pytest.mark.parametrize("method", METHODS)
def test_known_codes(method: DistanceMethod) -> None:
    steane = codes.SteaneCode()
    examples = (
        (steane.matrix_x, steane.matrix_z, Counts(3, 7, 7, 21)),
        ([[1, 1, 1, 1]], [[1, 1, 1, 1]], Counts(2, 6, 6, 18)),
        (HAND_X, HAND_Z, Counts(2, 2, 3, 6)),
        # The Z sector is heavier, so it contributes nothing at the global minimum.
        ([[1, 1, 0]], [[0, 0, 1]], Counts(1, 2, 0, 2)),
        (np.zeros((0, 3)), np.zeros((0, 3)), Counts(1, 3, 3, 9)),
        (np.eye(4), np.zeros((0, 4)), Counts(4, 0, 0, 0)),
    )
    for checks_x, checks_z, expected in examples:
        counter = get_counter(checks_x, checks_z)
        assert counter.count_logical_ops(count_pauli=True, method=method) == expected
        assert counter.count_logical_ops(method=method).count_pauli is None
    assert lc.count_logical_ops_css(steane, method="information_sets") == Counts(3, 7, 7, None)


@pytest.mark.parametrize("method", METHODS)
def test_counts_match_enumeration(method: DistanceMethod) -> None:
    """Compare every weight, and full Pauli counts, with an enumeration of both normalizers."""
    rng = np.random.default_rng(17)
    examples = [
        codes.BaconShorCode(3),
        codes.BaconShorCode(2, 3),
        *(_random_subsystem_code(rng, num_qubits) for num_qubits in (7, 8, 9, 10)),
        _random_subsystem_code(rng, 9, self_dual=True),
        *(get_random_css_code(rng, 9, 3, 3) for _ in range(2)),
    ]
    for code in examples:
        counts_x, counts_z, distance, full = enumerate_logical_ops(code)
        counter = lc.CSSLogicalCounter(code)
        assert counter.dimension == code.dimension
        for weight in range(len(code) + 1):
            expected = Counts(weight, counts_x[weight], counts_z[weight], None)
            assert counter.count_logical_ops(weight, method=method) == expected
        expected_full = Counts(distance, counts_x[distance], counts_z[distance], full)
        assert counter.count_logical_ops(method=method, count_pauli=True) == expected_full
        assert counter.count_logical_ops(distance, method=method, count_pauli=True) == expected_full


def test_counting_continues_at_an_equal_lower_bound() -> None:
    """With no checks, every one of the three words of weight two is a logical of each type.

    After the first layer, the bound on unvisited words equals two, so the count must continue.
    """
    counter = get_counter(np.zeros((0, 3)), np.zeros((0, 3)))
    assert counter.count_logical_ops(2) == Counts(2, 3, 3, None)


@pytest.mark.parametrize("method", METHODS)
def test_pauli_counts_away_from_the_distance(method: DistanceMethod) -> None:
    counter = lc.CSSLogicalCounter(codes.SteaneCode())
    for weight in (0, 2):
        expected = Counts(weight, 0, 0, 0)
        assert counter.count_logical_ops(weight, method=method, count_pauli=True) == expected
    with pytest.raises(ValueError, match="less than weight 4"):
        counter.count_logical_ops(4, method=method, count_pauli=True)

    # XYZI is a logical of weight three in the [[4, 2, 2]] code, but no homogeneous logical is odd.
    # A frozen qubit adds a Z stabilizer of weight one, which turns each of the six Z logicals of
    # weight two into one of weight three.
    counter = get_counter(*append_frozen_qubits([[1, 1, 1, 1]], [[1, 1, 1, 1]], 5))
    assert counter.count_logical_ops(3, method=method) == Counts(3, 0, 6, None)
    with pytest.raises(ValueError, match="less than weight 3"):
        counter.count_logical_ops(3, method=method, count_pauli=True)


@pytest.mark.parametrize("method", STABILIZER_METHODS)
@pytest.mark.parametrize(
    "checks_x,checks_z,distance",
    [
        (codes.SteaneCode().matrix_x, codes.SteaneCode().matrix_z, 3),
        ([[1, 1, 0]], [[0, 0, 1]], 1),  # The lightest logical operator is X-only.
        ([[0, 0, 1]], [[1, 1, 0]], 1),  # The lightest logical operator is Z-only.
    ],
)
def test_within_distance_raises_exactly_above_the_distance(
    checks_x: npt.ArrayLike, checks_z: npt.ArrayLike, distance: int, method: lc.LogicalCountMethod
) -> None:
    counter = get_counter(checks_x, checks_z)
    for weight in range(counter.num_qubits + 1):
        if weight > distance:
            with pytest.raises(ValueError, match="distance is less than"):
                counter.count_logical_ops(weight, method=method, within_distance=True)
        else:
            counts = counter.count_logical_ops(weight, method=method)
            assert counter.count_logical_ops(weight, method=method, within_distance=True) == counts


def test_count_logical_ops_css_accepts_within_distance() -> None:
    with pytest.raises(ValueError, match="distance is less than"):
        lc.count_logical_ops_css(codes.SteaneCode(), 4, within_distance=True)


def test_surface_codes() -> None:
    # Unrotated surface codes have d minimum-weight logicals of each type: straight lines.
    for distance in (3, 5, 7):
        counter = lc.CSSLogicalCounter(codes.SurfaceCode(distance, rotated=False))
        expected = Counts(distance, distance, distance, 2 * distance)
        assert counter.count_logical_ops(count_pauli=True) == expected
    for distance in (3, 4):
        code = codes.SurfaceCode(distance)
        counter = lc.CSSLogicalCounter(code)
        assert counter.count_logical_ops(count_pauli=True) == get_expected_counts(code)


@pytest.mark.parametrize("width", [5, 64, 65, 129])
def test_frozen_qubits_preserve_counts(width: int) -> None:
    """Frozen qubits change the physical width, including across word boundaries, not the counts."""
    counter = get_counter(*append_frozen_qubits(HAND_X, HAND_Z, width))
    assert counter.count_logical_ops(count_pauli=True) == Counts(2, 2, 3, 6)


@pytest.mark.parametrize("num_qubits, rank_x, rank_z", [(16, 10, 2), (32, 14, 12), (40, 22, 18)])
def test_methods_agree_on_random_codes(num_qubits: int, rank_x: int, rank_z: int) -> None:
    """Brute force visits every word once, so it checks the information sets used by BZ."""
    rng = np.random.default_rng(num_qubits)
    for _ in range(3):
        counter = lc.CSSLogicalCounter(get_random_css_code(rng, num_qubits, rank_x, rank_z))
        assert counter.count_logical_ops(
            count_pauli=True, method="brouwer_zimmermann"
        ) == counter.count_logical_ops(count_pauli=True, method="brute_force")


def test_compile_search_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiling up front gives the same counts and distances, and nothing compiles afterwards.

    The test starts from functions that no other test has compiled.  Its counters are deformed after
    a count, so that they move their kept basis families through a named and a general gate.
    """
    fresh = [replace(item, compiled=None) for item in _compilation._ON_REQUEST_FUNCTIONS]
    monkeypatch.setattr(_compilation, "_ON_REQUEST_FUNCTIONS", fresh)
    code = qldpc.codes.SurfaceCode(5)

    def get_results() -> list[qldpc.codes.CSSLogicalCounts]:
        counter = qldpc.codes.CSSLogicalCounter(code)
        results = [
            counter.count_logical_ops(),
            counter.count_logical_ops(count_pauli=True, max_information_sets=4),
        ]
        for deformed in (counter.deformed_cnot(0, 1), counter.deformed([0, 1], [[1, 0], [1, 1]])):
            results.append(deformed.count_logical_ops())
            results.append(deformed.count_logical_ops(results[0].weight + 1))
        return results

    expected = get_results()
    try:
        qldpc.codes.distance.compile_search_setup()
        with numba.core.event.install_recorder("numba:compile") as recorder:
            assert get_results() == expected
    finally:
        _compilation.set_on_request_functions_compiled(False)
    late_compilations = [
        event.data["dispatcher"].py_func.__qualname__
        for _, event in recorder.buffer
        if event.is_start and event.data["dispatcher"].py_func.__module__.startswith("qldpc.")
    ]
    assert not late_compilations
    assert get_results() == expected


def test_a_failed_compilation_leaves_the_set_up_code_as_plain_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that could not compile is not left half compiled."""

    def refuse() -> None:
        raise RuntimeError("compilation refused")

    monkeypatch.setattr(lc, "_warm_up_compiled_functions", refuse)
    with pytest.raises(RuntimeError, match="refused"):
        qldpc.codes.distance.compile_search_setup()
    select = _information_sets.get_overlapping_family
    assert not isinstance(select, numba.core.registry.CPUDispatcher)


def test_max_information_sets_must_be_an_integer() -> None:
    """The limit is an integer, of any integer type, and a float is rejected."""
    counter = get_counter(HAND_X, HAND_Z)
    not_an_integer: Any = 2.5
    with pytest.raises(TypeError):
        counter.count_logical_ops(max_information_sets=not_an_integer)
    expected = counter.count_logical_ops()
    assert counter.count_logical_ops(max_information_sets=np.int32(4)) == expected


def test_later_counts_reuse_the_first_family_stage() -> None:
    """Counts at other weights, as when a guessed weight fails, and distance searches share it.

    The distance search of the unrotated distance-five surface code uses information sets.  Its
    lightest logical operators of each type are the five straight lines across the lattice.
    """
    counter = lc.CSSLogicalCounter(codes.SurfaceCode(5, rotated=False))
    with mock.patch.object(
        _information_sets, "get_disjoint_family", wraps=_information_sets.get_disjoint_family
    ) as first_stage:
        assert counter.count_logical_ops() == Counts(5, 5, 5, None)
        num_first_stages = first_stage.call_count
        assert counter.count_logical_ops(3) == Counts(3, 0, 0, None)
        assert counter.count_logical_ops() == Counts(5, 5, 5, None)
    assert num_first_stages > 0
    assert first_stage.call_count == num_first_stages


def test_deformed_counter_reuses_the_first_family_stage() -> None:
    """A deformed counter, which moves the kept basis families through its gate, counts correctly.

    Like any counter, it keeps the first stage for its later counts.
    """
    counter = get_counter(HAND_X, HAND_Z)
    counter.count_logical_ops()
    deformed = counter.deformed_cnot(0, 2)
    weights = [None, *range(counter.num_qubits + 1)]
    fresh = get_counter(deformed.checks_x, deformed.checks_z)
    expected = [fresh.count_logical_ops(weight) for weight in weights]
    with mock.patch.object(
        _information_sets, "get_disjoint_family", wraps=_information_sets.get_disjoint_family
    ) as first_stage:
        counts = [deformed.count_logical_ops(weights[0])]
        num_first_stages = first_stage.call_count
        counts += [deformed.count_logical_ops(weight) for weight in weights[1:]]
    assert counts == expected
    assert first_stage.call_count == num_first_stages


def test_dense_eighty_qubits_against_physical_enumeration() -> None:
    """Enumerate physical supports of weight at most four, independently of the counter."""
    n, k = 80, 32
    rng = np.random.default_rng(47)
    x = np.zeros((24, n), dtype=np.uint8)
    z = x.copy()
    x[:, k : k + 24] = np.eye(24, dtype=np.uint8)
    z[:, k + 24 :] = np.eye(24, dtype=np.uint8)
    for _ in range(30 * n):
        control, target = rng.choice(n, 2, replace=False)
        x[:, target] ^= x[:, control]
        z[:, control] ^= z[:, target]
    assert np.mean(np.vstack([x, z])) > 0.4
    counts = []
    for stabilizers, checks in ((x, z), (z, x)):
        pivots: dict[int, int] = {}
        for row in stabilizers:
            word = sum(int(bit) << jj for jj, bit in enumerate(row))
            while word:
                pivot = word.bit_length() - 1
                if pivot not in pivots:
                    pivots[pivot] = word
                    break
                word ^= pivots[pivot]
        columns = [sum(int(bit) << jj for jj, bit in enumerate(col)) for col in checks.T]
        found = 0
        for weight in range(1, 5):
            for positions in itertools.combinations(range(n), weight):
                if functools.reduce(int.__xor__, (columns[jj] for jj in positions)):
                    continue
                reduced = sum(1 << jj for jj in positions)
                for pivot in sorted(pivots, reverse=True):
                    if reduced >> pivot & 1:
                        reduced ^= pivots[pivot]
                if reduced:
                    assert weight == 4
                    found += 1
        counts.append(found)
    assert counts == [1, 0]
    assert get_counter(x, z).count_logical_ops(count_pauli=True) == Counts(4, 1, 0, 1)


####################################################################################################
# deformation


def test_deformation() -> None:
    counter = lc.CSSLogicalCounter(codes.SteaneCode())
    before = counter.checks_x
    changed = counter.deformed_cnot(1, 6)
    assert np.array_equal(counter.checks_x, before)
    expected = before.copy()
    expected[:, 6] ^= expected[:, 1]
    assert np.array_equal(changed.checks_x, get_independent_rows(expected))
    assert counter.is_self_dual and not changed.is_self_dual
    assert changed.deformed_cnot(1, 6).is_self_dual

    # General gates agree with the named ones.
    general = counter.deformed([1, 6], [[1, 0], [1, 1]])
    assert np.array_equal(general.checks_x, changed.checks_x)
    assert np.array_equal(general.checks_z, changed.checks_z)
    swel = [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]
    named = counter.deformed_swel(0, 2, 3, 5)
    assert np.array_equal(counter.deformed([0, 2, 3, 5], swel).checks_x, named.checks_x)
    assert named.is_self_dual


@pytest.mark.parametrize("size", [3, 64])
def test_deformation_transforms_checks(size: int) -> None:
    """X checks transform by the gate, and Z checks by its inverse transpose."""
    rng = np.random.default_rng(size)
    field = galois.GF(2)
    while True:
        matrix = rng.integers(0, 2, (size, size))
        if np.linalg.matrix_rank(field(matrix)) == size:
            break
    inverse_transpose = np.asarray(np.linalg.inv(field(matrix)).T, dtype=int)
    counter = lc.CSSLogicalCounter(get_random_css_code(rng, 70, 5, 5))
    targets = sorted(int(target) for target in rng.choice(70, size, replace=False))
    deformed = counter.deformed(targets, matrix)
    for checks, local, result in (
        (counter.checks_x, matrix, deformed.checks_x),
        (counter.checks_z, inverse_transpose, deformed.checks_z),
    ):
        expected = checks.astype(int)
        expected[:, targets] = expected[:, targets] @ local.T % 2
        assert np.array_equal(result, get_independent_rows(expected.astype(np.uint8)))


def test_deformed_counts_match_enumeration() -> None:
    """Counters that deform an already counted code give the counts of the new code."""
    rng = np.random.default_rng(29)
    swel = [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]
    for code in (
        codes.BaconShorCode(3),
        get_random_css_code(rng, 10, 3, 3),
        codes.CSSCode.stack([codes.SteaneCode()] * 2),
        _random_subsystem_code(rng, 9, self_dual=True),
    ):
        counter = lc.CSSLogicalCounter(code)
        counter.count_logical_ops()
        for step in range(4):
            targets = [int(target) for target in rng.choice(counter.num_qubits, 4, replace=False)]
            if step == 0:
                counter = counter.deformed_swel(*targets)
            elif step == 1:
                counter = counter.deformed(targets, swel)
            else:
                counter = counter.deformed_cnot(*targets[:2])
            assert counter.count_logical_ops(count_pauli=True) == get_expected_counts(counter.code)


####################################################################################################
# weight enumerators


def test_macwilliams_counts_match_enumeration() -> None:
    """Compare every weight, and self-dual full Pauli counts, with enumerated normalizers."""
    rng = np.random.default_rng(31)
    steane = codes.SteaneCode()
    examples = [
        codes.CSSCode(np.vstack([steane.matrix_x, steane.matrix_x]), steane.matrix_z),
        codes.CSSCode(HAND_X.astype(int), HAND_Z.astype(int)),
        codes.CSSCode(np.zeros((0, 3), dtype=int), np.zeros((0, 3), dtype=int)),
        get_random_css_code(rng, 10, 2, 5),
    ]
    for code in examples:
        counts_x, counts_z, distance, full = enumerate_logical_ops(code)
        counter = lc.CSSLogicalCounter(code)
        for weight in range(len(code) + 1):
            expected = Counts(weight, counts_x[weight], counts_z[weight], None)
            assert counter.count_logical_ops(weight, method="macwilliams") == expected
        expected = Counts(distance, counts_x[distance], counts_z[distance], None)
        assert counter.count_logical_ops(method="macwilliams") == expected
        if counter.is_self_dual:
            expected = Counts(distance, counts_x[distance], counts_z[distance], full)
            assert counter.count_logical_ops(method="macwilliams", count_pauli=True) == expected
    assert lc.count_logical_ops_css(steane, method="weight_enumerator") == Counts(3, 7, 7, None)


def test_macwilliams_counts_of_wide_codes() -> None:
    """At length 130, counts exceed 64-bit integers.

    The X check is all ones, and the Z checks pair qubits 0 to 5, so an X-only logical repeats each
    pair, and a Z-only logical has even weight.  The excluded words are 0 and all ones for X, and
    the three pairs combined for Z.
    """
    num_qubits = 130
    pairs = np.zeros((3, num_qubits), dtype=int)
    for pair in range(3):
        pairs[pair, 2 * pair : 2 * pair + 2] = 1
    counter = get_counter(np.ones((1, num_qubits)), pairs)
    for weight in (64, 65, 130):
        count_x = sum(
            math.comb(3, pair_count) * math.comb(num_qubits - 6, weight - 2 * pair_count)
            for pair_count in range(min(3, weight // 2) + 1)
        ) - (weight == num_qubits)
        count_z = (
            math.comb(num_qubits, weight) - math.comb(3, weight // 2) if weight % 2 == 0 else 0
        )
        expected = Counts(weight, count_x, count_z, None)
        assert counter.count_logical_ops(weight, method="macwilliams") == expected


@pytest.mark.parametrize("copies, count_pauli_first", [(1, False), (1, True), (5, False)])
def test_macwilliams_enumerators_are_computed_once_per_counter(
    copies: int, count_pauli_first: bool
) -> None:
    """Later weights and flags reuse the walk, including checks with different equal-space bases."""
    code = codes.CSSCode.stack([codes.SteaneCode()] * copies)
    counter = lc.CSSLogicalCounter(code).deformed_cnot(1, 6).deformed_cnot(1, 6)
    with mock.patch.object(
        macwilliams, "get_span_weights", wraps=macwilliams.get_span_weights
    ) as walk:
        expected = Counts(3, 7 * copies, 7 * copies, 21 * copies if count_pauli_first else None)
        assert (
            counter.count_logical_ops(method="macwilliams", count_pauli=count_pauli_first)
            == expected
        )
        walks = walk.call_count
        assert counter.count_logical_ops(2, method="macwilliams") == Counts(2, 0, 0, None)
        assert counter.count_logical_ops(3, method="macwilliams", count_pauli=True) == Counts(
            3, 7 * copies, 7 * copies, 21 * copies
        )
    assert walks > 0 and walk.call_count == walks


def test_deformed_counters_walk_their_own_checks() -> None:
    """A gate changes the weight distributions of the checks, so they are not reused."""
    counter = lc.CSSLogicalCounter(get_random_css_code(np.random.default_rng(41), 10, 3, 3))
    weights = range(counter.num_qubits + 1)
    before = [counter.count_logical_ops(weight, method="macwilliams") for weight in weights]
    deformed = counter.deformed_cnot(0, 1)
    fresh = get_counter(deformed.checks_x, deformed.checks_z)
    expected = [fresh.count_logical_ops(weight, method="macwilliams") for weight in weights]
    assert expected != before
    counts = [deformed.count_logical_ops(weight, method="macwilliams") for weight in weights]
    assert counts == expected


def test_macwilliams_capabilities() -> None:
    """Unsupported codes and requests raise errors, and still work with the other methods."""
    no_logicals = get_counter(np.eye(4)[:2], np.eye(4)[2:])
    assert no_logicals.count_logical_ops(method="macwilliams") == Counts(4, 0, 0, None)
    extended_hamming = codes.ExtendedHammingCode(3).matrix
    self_dual = get_counter(extended_hamming, extended_hamming)
    assert self_dual.dimension == 0
    expected = Counts(2, 0, 0, 0)
    assert self_dual.count_logical_ops(2, method="macwilliams", count_pauli=True) == expected

    bacon_shor = codes.BaconShorCode(3)
    subsystem = lc.CSSLogicalCounter(bacon_shor)
    with pytest.raises(ValueError, match="does not support subsystem codes"):
        subsystem.count_logical_ops(method="macwilliams")
    assert subsystem.count_logical_ops(count_pauli=True) == get_expected_counts(bacon_shor)

    asymmetric = get_counter(HAND_X, HAND_Z)
    with pytest.raises(ValueError, match="only for self-dual codes"):
        asymmetric.count_logical_ops(method="macwilliams", count_pauli=True)
    assert asymmetric.count_logical_ops(count_pauli=True) == Counts(2, 2, 3, 6)


####################################################################################################
# inputs and validation


def test_code_and_checks() -> None:
    steane = codes.SteaneCode()
    counter = lc.CSSLogicalCounter(steane)
    assert counter.code is steane

    derived = counter.deformed_cnot(0, 1)
    assert isinstance(derived.code, codes.CSSCode)
    derived_x = get_independent_rows(derived.code.matrix_x.view(np.ndarray).astype(np.uint8))
    assert np.array_equal(derived_x, derived.checks_x)

    # Redundant checks are reduced, and the returned checks are copies.
    redundant = np.vstack([steane.matrix_x, steane.matrix_x, np.zeros((1, 7), dtype=int)])
    counter = get_counter(redundant, steane.matrix_z)
    assert len(counter.checks_x) == 3 and counter.dimension == 1 and counter.is_self_dual
    counter.checks_x[:] = 0
    assert np.any(counter.checks_x)


def test_public_api() -> None:
    """Counts are exported by qldpc.codes, and default to the distance of the code."""
    code = codes.SurfaceCode(3)
    counter = codes.CSSLogicalCounter(code)
    assert counter.count_logical_ops() == codes.count_logical_ops_css(code)
    assert codes.count_logical_ops_css(code).weight == code.get_distance()
    with pytest.raises(AttributeError):
        counter.num_qubits = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="distinct qubit indices"):
        counter.deformed_cnot(1, 1)


def test_invalid_inputs() -> None:
    not_css: Any = codes.FiveQubitCode()
    with pytest.raises(TypeError, match=r"Expected a CSSCode.*to_css"):
        lc.CSSLogicalCounter(not_css)
    with pytest.raises(ValueError, match="binary"):
        lc.CSSLogicalCounter(codes.CSSCode([[1, 1]], [[1, 1]], field=3))

    counter = get_counter([[0, 0]], [[0, 0]])
    invalid_method: Any = "unknown"
    with pytest.raises(ValueError, match="Unknown logical-count method"):
        counter.count_logical_ops(method=invalid_method)
    for method in ("brouwer_zimmermann", "macwilliams", "brute_force"):
        assert lc._validate_logical_count_method(method) == method
    assert lc._validate_logical_count_method("information_sets") == "brouwer_zimmermann"
    assert lc._validate_logical_count_method("weight_enumerator") == "macwilliams"
    with pytest.raises(ValueError, match="Unknown logical-count method"):
        lc._validate_logical_count_method(invalid_method)
    for weight in (-1, 3, 1.5, True, "1"):
        invalid_weight: Any = weight
        with pytest.raises(ValueError, match="weight must be an integer"):
            counter.count_logical_ops(invalid_weight)
    numpy_weight: Any = np.int64(2)
    assert counter.count_logical_ops(numpy_weight) == Counts(2, 1, 1, None)

    steane = lc.CSSLogicalCounter(codes.SteaneCode())
    wide = get_counter(np.zeros((0, 65)), np.zeros((0, 65)))
    for targets in ([0, 0], [-1, 2], [0, 7], [0.5, 2], [True, 2], [], list(range(65))):
        invalid_targets: Any = targets
        with pytest.raises(ValueError, match="1 to 64 distinct"):
            (wide if len(targets) == 65 else steane).deformed(invalid_targets, [[1]])
    with pytest.raises(ValueError, match="1 to 64 distinct"):
        steane.deformed_swel(0, 1, 1, 2)
