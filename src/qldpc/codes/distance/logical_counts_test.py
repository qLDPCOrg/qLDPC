# SPDX-License-Identifier: Apache-2.0

"""Tests for exact counts of the logical operators of a given weight in binary CSS codes."""

from __future__ import annotations

import collections
import functools
import itertools
import math
from dataclasses import replace
from typing import Any, cast
from unittest import mock

import galois
import numba.core.event
import numpy as np
import numpy.typing as npt
import pytest
import scipy.sparse

import qldpc
from qldpc import _compilation, codes
from qldpc._binary import get_independent_rows

from ..conftest import Tune
from . import logical_counts as lc
from ._counting import macwilliams
from ._information_sets import basis_families
from ._tuning import MAX_VALUE, use_tuning
from .conftest import (
    HAND_X,
    HAND_Z,
    METHODS,
    SWEL_MATRIX,
    Counts,
    enumerate_logical_ops,
    get_random_css_code,
)

STABILIZER_METHODS: tuple[lc.LogicalCountMethod, ...] = (
    "brouwer_zimmermann",
    "macwilliams",
    "brute_force",
)


####################################################################################################
# counts


@pytest.mark.parametrize("method", METHODS)
def test_known_codes(method: lc.LogicalCountMethod) -> None:
    steane = codes.SteaneCode()
    examples = (
        (steane.matrix_x, steane.matrix_z, Counts(3, 7, 7, 21)),
        ([[1, 1, 1, 1]], [[1, 1, 1, 1]], Counts(2, 6, 6, 18)),
        (HAND_X, HAND_Z, Counts(2, 2, 3, 6)),
        # The Z sector is heavier, so it contributes nothing at the global minimum.
        ([[1, 1, 0]], [[0, 0, 1]], Counts(1, 2, 0, 2)),
        (np.zeros((0, 3)), np.zeros((0, 3)), Counts(1, 3, 3, 9)),
        (np.eye(4), np.zeros((0, 4)), Counts(codes.NO_DISTANCE, 0, 0, 0)),
    )
    for checks_x, checks_z, expected in examples:
        counter = lc.CSSLogicalCounter(checks_x, checks_z)
        assert counter.count_logical_ops(count_pauli=True, method=method) == expected
        assert counter.count_logical_ops(method=method).count_pauli is None


def test_counts_are_equal_exactly_when_their_classes_and_fields_are() -> None:
    """Counts compare and hash by their fields, in which NO_DISTANCE equals itself, and by class."""
    no_distance = Counts(codes.NO_DISTANCE, 0, 0, None)
    assert no_distance == Counts(codes.NO_DISTANCE, 0, 0, None)
    assert hash(no_distance) == hash(Counts(codes.NO_DISTANCE, 0, 0, None))
    assert no_distance != Counts(4, 0, 0, None) and no_distance != (codes.NO_DISTANCE, 0, 0, None)

    # As for dataclasses, counts of different classes differ.
    class SubclassCounts(Counts):
        pass

    assert no_distance != SubclassCounts(codes.NO_DISTANCE, 0, 0, None)


@pytest.mark.parametrize("method", METHODS)
def test_counts_match_enumeration(method: lc.LogicalCountMethod) -> None:
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
        counter = lc.CSSLogicalCounter(code.matrix_x, code.matrix_z)
        assert counter.dimension == code.dimension
        for weight in range(len(code) + 1):
            expected = Counts(weight, counts_x[weight], counts_z[weight], None)
            assert counter.count_logical_ops(weight, method=method) == expected
        expected_full = Counts(distance, counts_x[distance], counts_z[distance], full)
        assert counter.count_logical_ops(method=method, count_pauli=True) == expected_full
        assert counter.count_logical_ops(distance, method=method, count_pauli=True) == expected_full


def test_counting_continues_at_an_equal_lower_bound() -> None:
    """A count continues after a step whose bound equals the counted weight."""
    # With no checks, each of the three words of weight two is a logical of each type, and the
    # bound after the first layer is two.
    counter = lc.CSSLogicalCounter(np.zeros((0, 3)), np.zeros((0, 3)))
    assert counter.count_logical_ops(2) == Counts(2, 3, 3, None)


@pytest.mark.parametrize("method", METHODS)
def test_pauli_counts_away_from_the_distance(method: lc.LogicalCountMethod) -> None:
    counter = lc.CSSLogicalCounter.from_code(codes.SteaneCode())
    for weight in (0, 2):
        expected = Counts(weight, 0, 0, 0)
        assert counter.count_logical_ops(weight, method=method, count_pauli=True) == expected
    with pytest.raises(ValueError, match="less than weight 4"):
        counter.count_logical_ops(4, method=method, count_pauli=True)

    # XYZI is a logical of weight three in the [[4, 2, 2]] code, but no homogeneous logical is odd.
    # A frozen qubit adds a Z stabilizer of weight one, which turns each of the six Z logicals of
    # weight two into one of weight three.
    counter = lc.CSSLogicalCounter(*_append_frozen_qubits([[1, 1, 1, 1]], [[1, 1, 1, 1]], 5))
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
    counter = lc.CSSLogicalCounter(checks_x, checks_z)
    for weight in range(counter.num_qubits + 1):
        if weight > distance:
            with pytest.raises(ValueError, match="distance is less than"):
                counter.count_logical_ops(weight, method=method, within_distance=True)
        else:
            counts = counter.count_logical_ops(weight, method=method)
            assert counter.count_logical_ops(weight, method=method, within_distance=True) == counts


def test_count_logical_ops_css_accepts_within_distance() -> None:
    steane = codes.SteaneCode()
    with pytest.raises(ValueError, match="distance is less than"):
        lc.count_logical_ops_css(steane.matrix_x, steane.matrix_z, 4, within_distance=True)


def test_surface_codes() -> None:
    # Unrotated surface codes have d minimum-weight logicals of each type: straight lines.
    for distance in (3, 5):
        counter = lc.CSSLogicalCounter.from_code(codes.SurfaceCode(distance, rotated=False))
        expected = Counts(distance, distance, distance, 2 * distance)
        assert counter.count_logical_ops(count_pauli=True) == expected
    for distance in (3, 4):
        code = codes.SurfaceCode(distance)
        counter = lc.CSSLogicalCounter.from_code(code)
        assert counter.count_logical_ops(count_pauli=True) == _get_expected_counts(code)


@pytest.mark.parametrize("width", [5, 64, 65, 129])
def test_frozen_qubits_preserve_counts(width: int) -> None:
    """Frozen qubits change the physical width, including across word boundaries, not the counts."""
    counter = lc.CSSLogicalCounter(*_append_frozen_qubits(HAND_X, HAND_Z, width))
    assert counter.count_logical_ops(count_pauli=True) == Counts(2, 2, 3, 6)


@pytest.mark.parametrize("num_qubits, rank_x, rank_z, num_sets", [(16, 10, 2, 3), (24, 12, 8, 2)])
def test_methods_agree_on_random_codes(
    monkeypatch: pytest.MonkeyPatch, num_qubits: int, rank_x: int, rank_z: int, num_sets: int
) -> None:
    """Brute force visits every word once, so it checks which information set counts each word."""
    sizes: list[int] = []
    get_disjoint_family = basis_families.get_disjoint_family

    def record(*args: Any, **kwargs: Any) -> basis_families.BasisFamily:
        family = get_disjoint_family(*args, **kwargs)
        sizes.append(len(family.ranks))
        return family

    monkeypatch.setattr(basis_families, "get_disjoint_family", record)
    rng = np.random.default_rng(num_qubits)
    for _ in range(3):
        counter = lc.CSSLogicalCounter.from_code(
            get_random_css_code(rng, num_qubits, rank_x, rank_z)
        )
        assert counter.count_logical_ops(
            count_pauli=True, method="brouwer_zimmermann"
        ) == counter.count_logical_ops(count_pauli=True, method="brute_force")
    # Some disjoint family that a scan starts from has at least ``num_sets`` sets, so several sets
    # could count a word.
    assert max(sizes) >= num_sets


@pytest.mark.usefixtures("plain_search_setup")
def test_compile_search_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compiling the set-up code up front changes no count, and leaves nothing to compile later."""
    # Start from functions that no other test has compiled.
    fresh = [replace(item, compiled=None) for item in _compilation._ON_REQUEST_FUNCTIONS]
    monkeypatch.setattr(_compilation, "_ON_REQUEST_FUNCTIONS", fresh)
    code = qldpc.codes.SurfaceCode(5)

    def get_results(
        counter: lc.CSSLogicalCounter, *, fresh: bool = False
    ) -> list[lc.CSSLogicalCounts]:
        """Counts of a counter and of two deformations, or of fresh counters of their checks."""
        results = [counter.count_logical_ops()]
        assert results[0].weight is not qldpc.codes.NO_DISTANCE
        with use_tuning(max_information_sets=4):
            results.append(counter.count_logical_ops(count_pauli=True))
        # Deform after a count, so that the kept families move through a named and a general gate.
        for deformed in (counter.deformed_cnot(0, 1), counter.deformed([0, 1], [[1, 0], [1, 1]])):
            if fresh:
                deformed = lc.CSSLogicalCounter(deformed.checks_x, deformed.checks_z)
            results.append(deformed.count_logical_ops())
            results.append(deformed.count_logical_ops(results[0].weight + 1))
        return results

    expected = get_results(lc.CSSLogicalCounter.from_code(code), fresh=True)
    # A counter that counts before `compile_search_setup` keeps sectors and families that plain
    # Python built; afterwards it must still count with them and deform them.
    early = lc.CSSLogicalCounter.from_code(code)
    early.count_logical_ops()
    try:
        qldpc.codes.distance.compile_search_setup()
        with numba.core.event.install_recorder("numba:compile") as recorder:
            assert get_results(lc.CSSLogicalCounter.from_code(code)) == expected
            assert get_results(early) == expected
    finally:
        _compilation.set_on_request_functions_compiled(False)
    late_compilations = [
        event.data["dispatcher"].py_func.__qualname__
        for _, event in recorder.buffer
        if event.is_start and event.data["dispatcher"].py_func.__module__.startswith("qldpc.")
    ]
    assert not late_compilations
    assert get_results(lc.CSSLogicalCounter.from_code(code)) == expected


def test_a_failed_compilation_leaves_the_set_up_code_as_plain_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A process that could not compile is not left half compiled."""

    def refuse() -> None:
        raise RuntimeError("compilation refused")

    monkeypatch.setattr(lc, "_warm_up_compiled_functions", refuse)
    with pytest.raises(RuntimeError, match="refused"):
        qldpc.codes.distance.compile_search_setup()
    select = basis_families.get_overlapping_family
    assert not isinstance(select, numba.core.registry.CPUDispatcher)


@pytest.mark.usefixtures("plain_search_setup")
def test_the_information_set_limit_reaches_every_search_of_a_count(
    monkeypatch: pytest.MonkeyPatch, tune: Tune
) -> None:
    """The active limit bounds every family that a count builds, also for its distance search."""
    sizes: list[int] = []
    original = basis_families.get_overlapping_family

    def record(*args: Any, **kwargs: Any) -> basis_families.BasisFamily:
        family = original(*args, **kwargs)
        sizes.append(len(family.ranks))
        return family

    monkeypatch.setattr(basis_families, "get_overlapping_family", record)
    # Make every search try overlapping families.
    tune(overlapping_minimum_work=0)
    code = get_random_css_code(np.random.default_rng(2), 24, 8, 8)

    # A count of the same weight, without the distance search, builds the families of the scans.
    expected = lc.CSSLogicalCounter.from_code(code).count_logical_ops()
    assert expected.weight is not codes.NO_DISTANCE
    sizes_with_distance = collections.Counter(sizes)
    sizes.clear()
    lc.CSSLogicalCounter.from_code(code).count_logical_ops(expected.weight)
    # With the default limit, the distance search of this code builds a family of over three sets,
    # so a limit of three bounds it only if the limit reaches that search.
    distance_sizes = sizes_with_distance - collections.Counter(sizes)
    assert max(distance_sizes) > 3

    sizes.clear()
    tune(max_information_sets=3)
    assert lc.CSSLogicalCounter.from_code(code).count_logical_ops() == expected
    assert sizes and max(sizes) <= 3


def test_later_counts_reuse_the_first_family_stage() -> None:
    """Distance searches, and counts at other weights, share the disjoint basis families."""
    # The distance search of this code uses information sets, and its lightest logical operators of
    # each type are the five straight lines across the lattice.
    counter = lc.CSSLogicalCounter.from_code(codes.SurfaceCode(5, rotated=False))
    with mock.patch.object(
        basis_families, "get_disjoint_family", wraps=basis_families.get_disjoint_family
    ) as first_stage:
        assert counter.count_logical_ops() == Counts(5, 5, 5, None)
        num_first_stages = first_stage.call_count
        assert counter.count_logical_ops(3) == Counts(3, 0, 0, None)
        assert counter.count_logical_ops() == Counts(5, 5, 5, None)
    assert num_first_stages > 0
    assert first_stage.call_count == num_first_stages


def test_counters_remember_the_distance_of_each_method() -> None:
    """Default-weight counts search once per method, and a deformed counter searches again."""
    counter = lc.CSSLogicalCounter(HAND_X, HAND_Z)
    expected = counter.count_logical_ops(count_pauli=True)
    with mock.patch.object(lc, "_get_distance", wraps=lc._get_distance) as search:
        assert counter.count_logical_ops(count_pauli=True) == expected
        assert counter.count_logical_ops(method="brouwer_zimmermann") == replace(
            expected, count_pauli=None
        )
        assert search.call_count == 0
        assert counter.count_logical_ops(method="brute_force", count_pauli=True) == expected
        assert search.call_count == 1
        counter.deformed_cnot(0, 2).count_logical_ops()
        assert search.call_count == 2


def test_deformed_counter_reuses_the_first_family_stage() -> None:
    """A deformed counter counts exactly, and builds no family, since it receives every kept one."""
    counter = lc.CSSLogicalCounter(HAND_X, HAND_Z)
    counter.count_logical_ops()
    with use_tuning(disjoint_random_column_orders=3):
        counter.count_logical_ops(1)
    deformed = counter.deformed_cnot(0, 2)
    weights = [None, *range(counter.num_qubits + 1)]
    fresh = lc.CSSLogicalCounter(deformed.checks_x, deformed.checks_z)
    expected = [fresh.count_logical_ops(weight) for weight in weights]
    with mock.patch.object(
        basis_families, "get_disjoint_family", wraps=basis_families.get_disjoint_family
    ) as first_stage:
        counts = [deformed.count_logical_ops(weight) for weight in weights]
        with use_tuning(disjoint_random_column_orders=3):
            deformed.count_logical_ops(1)
    assert counts == expected
    assert not first_stage.call_count


def test_dense_eighty_qubits_against_physical_enumeration() -> None:
    """Enumerate physical supports of weight at most four, independently of the counter."""
    # Start from 24 X checks and 24 Z checks of weight one, which leave 32 logical qubits, and make
    # them dense by random CNOTs.
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

    # Count the nontrivial logicals of each type up to weight four.  The stabilizers of that type,
    # in echelon form keyed by their leading bits, tell whether a support is trivial.
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

        # A support has zero syndrome exactly if the column of its largest qubit equals the syndrome
        # of the other qubits.  Choosing the other qubits first, and then only qubits larger than
        # them whose column equals that syndrome, visits each support once.
        columns = [sum(int(bit) << jj for jj, bit in enumerate(col)) for col in checks.T]
        qubits_by_column: dict[int, list[int]] = {}
        for jj, column in enumerate(columns):
            qubits_by_column.setdefault(column, []).append(jj)
        found = 0
        for weight in range(1, 5):
            for head in itertools.combinations(range(n), weight - 1):
                syndrome = functools.reduce(int.__xor__, (columns[jj] for jj in head), 0)
                for last in qubits_by_column.get(syndrome, []):
                    if head and last <= head[-1]:
                        continue
                    reduced = sum(1 << jj for jj in (*head, last))
                    for pivot in sorted(pivots, reverse=True):
                        if reduced >> pivot & 1:
                            reduced ^= pivots[pivot]
                    if reduced:
                        assert weight == 4
                        found += 1
        counts.append(found)

    # The only nontrivial logical of weight at most four is an X logical of weight four.
    assert counts == [1, 0]
    assert lc.CSSLogicalCounter(x, z).count_logical_ops(count_pauli=True) == Counts(4, 1, 0, 1)


####################################################################################################
# deformation


def test_deformation() -> None:
    counter = lc.CSSLogicalCounter.from_code(codes.SteaneCode())
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
    named = counter.deformed_swel(0, 2, 3, 5)
    assert np.array_equal(counter.deformed([0, 2, 3, 5], SWEL_MATRIX).checks_x, named.checks_x)
    assert named.is_self_dual and named.code.is_swel

    # The gate changes the Z checks of the surface code, so the named gate must act on them too.
    surface = lc.CSSLogicalCounter.from_code(codes.SurfaceCode(3))
    named = surface.deformed_swel(0, 1, 2, 3)
    general = surface.deformed([0, 1, 2, 3], SWEL_MATRIX)
    assert not np.array_equal(general.checks_z, surface.checks_z)
    assert np.array_equal(general.checks_x, named.checks_x)
    assert np.array_equal(general.checks_z, named.checks_z)


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
    counter = lc.CSSLogicalCounter.from_code(get_random_css_code(rng, 70, 5, 5))
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
    for code in (
        codes.BaconShorCode(3),
        get_random_css_code(rng, 10, 3, 3),
        codes.CSSCode.stack([codes.SteaneCode()] * 2),
        _random_subsystem_code(rng, 9, self_dual=True),
    ):
        counter = lc.CSSLogicalCounter.from_code(code)
        counter.count_logical_ops()
        for step in range(4):
            targets = [int(target) for target in rng.choice(counter.num_qubits, 4, replace=False)]
            if step == 0:
                counter = counter.deformed_swel(*targets)
            elif step == 1:
                counter = counter.deformed(targets, SWEL_MATRIX)
            else:
                counter = counter.deformed_cnot(*targets[:2])
            assert counter.count_logical_ops(count_pauli=True) == _get_expected_counts(counter.code)


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
        counter = lc.CSSLogicalCounter.from_code(code)
        for weight in range(len(code) + 1):
            expected = Counts(weight, counts_x[weight], counts_z[weight], None)
            assert counter.count_logical_ops(weight, method="macwilliams") == expected
        expected = Counts(distance, counts_x[distance], counts_z[distance], None)
        assert counter.count_logical_ops(method="macwilliams") == expected
        if counter.is_self_dual:
            expected = Counts(distance, counts_x[distance], counts_z[distance], full)
            assert counter.count_logical_ops(method="macwilliams", count_pauli=True) == expected


def test_macwilliams_counts_of_wide_codes() -> None:
    """MacWilliams counts match a direct count where they exceed 64-bit integers."""
    # The X check is all ones, and the Z checks pair qubits 0 to 5, so an X-only logical repeats
    # each pair, and a Z-only logical has even weight.  The excluded words are 0 and all ones for
    # X, and the three pairs combined for Z.
    num_qubits = 130
    pairs = np.zeros((3, num_qubits), dtype=int)
    for pair in range(3):
        pairs[pair, 2 * pair : 2 * pair + 2] = 1
    counter = lc.CSSLogicalCounter(np.ones((1, num_qubits)), pairs)
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


def test_the_macwilliams_rank_limit_is_on_the_rank_of_the_checks() -> None:
    """Checks of rank 64 raise an error, but more checks of a lower rank do not."""
    # Each of these codes encodes two qubits, so that the method walks its checks.
    full_rank = np.eye(64, 66, dtype=np.uint8)
    no_checks = np.zeros((0, 66), dtype=np.uint8)
    for checks_x, checks_z in ((full_rank, no_checks), (no_checks, full_rank)):
        counter = lc.CSSLogicalCounter(checks_x, checks_z)
        assert counter.dimension == 2
        with pytest.raises(ValueError, match="check ranks up to 63, not 64"):
            counter.count_logical_ops(1, method="macwilliams")

    # Sixty-four copies of the check of the [[4, 2, 2]] code have rank one.
    redundant = np.ones((64, 4), dtype=np.uint8)
    counter = lc.CSSLogicalCounter(redundant, redundant)
    assert counter.count_logical_ops(method="macwilliams") == Counts(2, 6, 6, None)


@pytest.mark.parametrize(
    "shared_walk_rank, count_pauli_first, num_walks",
    [(MAX_VALUE, False, 2), (0, False, 1), (MAX_VALUE, True, 1)],
)
def test_macwilliams_enumerators_are_computed_once_per_counter(
    tune: Tune, shared_walk_rank: int, count_pauli_first: bool, num_walks: int
) -> None:
    """Later weights and flags reuse the walks, and checks of equal spaces may share one walk.

    Checks of equal spaces that the counter does not know to be self-dual share a walk from the
    tuned rank on, or for a full Pauli count.
    """
    tune(macwilliams_shared_walk_rank=shared_walk_rank)
    # The two CNOTs restore the Steane code, but the counter does not know that it is self-dual.
    steane = lc.CSSLogicalCounter.from_code(codes.SteaneCode())
    counter = steane.deformed_cnot(1, 6).deformed_cnot(1, 6)
    with mock.patch.object(
        macwilliams, "get_span_weights", wraps=macwilliams.get_span_weights
    ) as walk:
        expected = Counts(3, 7, 7, 21 if count_pauli_first else None)
        assert (
            counter.count_logical_ops(method="macwilliams", count_pauli=count_pauli_first)
            == expected
        )
        assert walk.call_count == num_walks
        assert counter.count_logical_ops(2, method="macwilliams") == Counts(2, 0, 0, None)
        assert counter.count_logical_ops(3, method="macwilliams", count_pauli=True) == Counts(
            3, 7, 7, 21
        )
    assert walk.call_count == num_walks


def test_deformed_counters_walk_their_own_checks() -> None:
    """A gate changes the weight distributions of the checks, so they are not reused."""
    counter = lc.CSSLogicalCounter.from_code(
        get_random_css_code(np.random.default_rng(41), 10, 3, 3)
    )
    weights = range(counter.num_qubits + 1)
    before = [counter.count_logical_ops(weight, method="macwilliams") for weight in weights]
    deformed = counter.deformed_cnot(0, 1)
    fresh = lc.CSSLogicalCounter(deformed.checks_x, deformed.checks_z)
    expected = [fresh.count_logical_ops(weight, method="macwilliams") for weight in weights]
    assert expected != before
    counts = [deformed.count_logical_ops(weight, method="macwilliams") for weight in weights]
    assert counts == expected


def test_macwilliams_capabilities() -> None:
    """Codes without logicals get zero counts, and unsupported codes and requests raise errors.

    Each unsupported code or request still works with the other methods.
    """
    no_logicals = lc.CSSLogicalCounter(np.eye(4)[:2], np.eye(4)[2:])
    no_distance = no_logicals.count_logical_ops(method="macwilliams")
    assert no_distance == Counts(codes.NO_DISTANCE, 0, 0, None)
    assert no_distance.weight is codes.NO_DISTANCE

    # The MacWilliams limit of rank 63 on the checks applies only to codes that encode qubits.
    rank_64 = lc.CSSLogicalCounter(np.eye(64, dtype=np.uint8), np.zeros((0, 64), dtype=np.uint8))
    assert rank_64.dimension == 0
    assert rank_64.count_logical_ops(method="macwilliams") == Counts(codes.NO_DISTANCE, 0, 0, None)

    extended_hamming = codes.ExtendedHammingCode(3).matrix
    self_dual = lc.CSSLogicalCounter(extended_hamming, extended_hamming)
    assert self_dual.dimension == 0
    expected = Counts(2, 0, 0, 0)
    assert self_dual.count_logical_ops(2, method="macwilliams", count_pauli=True) == expected

    bacon_shor = codes.BaconShorCode(3)
    subsystem = lc.CSSLogicalCounter.from_code(bacon_shor)
    with pytest.raises(ValueError, match="does not support subsystem codes"):
        subsystem.count_logical_ops(method="macwilliams")
    assert subsystem.count_logical_ops(count_pauli=True) == _get_expected_counts(bacon_shor)

    asymmetric = lc.CSSLogicalCounter(HAND_X, HAND_Z)
    with pytest.raises(ValueError, match="only for self-dual codes"):
        asymmetric.count_logical_ops(method="macwilliams", count_pauli=True)
    assert asymmetric.count_logical_ops(count_pauli=True) == Counts(2, 2, 3, 6)


####################################################################################################
# inputs and validation


def test_code_and_checks() -> None:
    """A counter of given checks reduces them, and builds its code from the reduced checks once."""
    steane = codes.SteaneCode()
    checks_x = steane.matrix_x.view(np.ndarray).astype(np.uint8)
    redundant = np.vstack([checks_x, checks_x, np.zeros((1, 7), dtype=np.uint8)])
    counter = lc.CSSLogicalCounter(redundant, steane.matrix_z)
    assert len(counter.checks_x) == 3 and counter.dimension == 1 and counter.is_self_dual
    _assert_reduced(counter.checks_x, redundant)
    _assert_reduced(counter.checks_z, steane.matrix_z)
    code = counter.code
    assert counter.code is code
    matrix_x = code.matrix_x.view(np.ndarray).astype(np.uint8)
    assert len(matrix_x) == 3
    assert np.array_equal(get_independent_rows(matrix_x), get_independent_rows(redundant))

    # The returned checks are copies.
    counter.checks_x[:] = 0
    assert np.any(counter.checks_x)

    derived = counter.deformed_cnot(0, 1)
    assert isinstance(derived.code, codes.CSSCode)
    _assert_reduced(derived.checks_x, derived.code.matrix_x)


def test_accepted_forms_of_checks() -> None:
    """Checks from a code, and as galois arrays, lists, or sparse matrices, give the same counts."""
    expected = Counts(2, 2, 3, 6)
    code = codes.CSSCode(HAND_X.astype(int), HAND_Z.astype(int))
    assert lc.CSSLogicalCounter.from_code(code).count_logical_ops(count_pauli=True) == expected
    for checks_x, checks_z in (
        (code.matrix_x, code.matrix_z),
        (HAND_X.tolist(), HAND_Z.tolist()),
        (scipy.sparse.csr_matrix(HAND_X), scipy.sparse.csr_array(HAND_Z)),
    ):
        counter = lc.CSSLogicalCounter(checks_x, checks_z)
        assert counter.count_logical_ops(count_pauli=True) == expected

    # Empty X checks take the width of the Z checks.  The X logical operators of weight 3 are then
    # the 7 words of weight 3 of the Hamming code.
    hamming_checks = codes.HammingCode(3).matrix
    assert lc.CSSLogicalCounter([], hamming_checks).count_logical_ops(3).count_x == 7


def test_invalid_checks() -> None:
    """Invalid checks raise errors that name them, and a code in place of its checks is rejected."""
    for checks_x, checks_z, message in (
        ([[0, 2]], [[1, 1]], "binary.*checks_x"),
        ([[1, 1]], galois.GF(3)([[1, 1]]), "binary.*checks_z"),
        ([[1, 1, 0]], [[1, 1]], "checks_z has 2 columns"),
        ([[1, 1]], np.zeros((2, 0)), "checks_z has 0 columns"),
        (np.ones((1, 1, 2)), [[1, 1]], "checks_x must be a matrix or a single row"),
    ):
        with pytest.raises(ValueError, match=message):
            lc.CSSLogicalCounter(checks_x, checks_z)

    steane: Any = codes.SteaneCode()
    with pytest.raises(TypeError, match="checks_x must be an array of binary entries, not a"):
        lc.CSSLogicalCounter(steane, steane.matrix_z)
    with pytest.raises(TypeError, match="checks_z must be an array of binary entries, not a"):
        lc.count_logical_ops_css(steane.matrix_x, steane, 3)


def test_counters_from_codes() -> None:
    steane = codes.SteaneCode()
    assert lc.CSSLogicalCounter.from_code(steane).code is steane
    not_css: Any = codes.FiveQubitCode()
    with pytest.raises(TypeError, match=r"Expected a CSSCode.*to_css"):
        lc.CSSLogicalCounter.from_code(not_css)
    with pytest.raises(ValueError, match="binary"):
        lc.CSSLogicalCounter.from_code(codes.CSSCode([[1, 1]], [[1, 1]], field=3))


def test_public_api() -> None:
    """Counts are exported by qldpc.codes.distance, and default to the distance of the code."""
    code = codes.SurfaceCode(3)
    counter = codes.distance.CSSLogicalCounter.from_code(code)
    counts = codes.distance.count_logical_ops_css(code.matrix_x, code.matrix_z)
    assert counter.count_logical_ops() == counts
    assert counts.weight == code.get_distance()
    with pytest.raises(AttributeError):
        # The property has no setter, which mypy reports.
        counter.num_qubits = 1  # type: ignore[misc]


def test_invalid_inputs() -> None:
    counter = lc.CSSLogicalCounter([[0, 0]], [[0, 0]])
    invalid_method: Any = "unknown"
    for unknown in (invalid_method, cast(Any, ["brouwer_zimmermann"])):
        with pytest.raises(ValueError, match="Unknown logical-count method"):
            counter.count_logical_ops(method=unknown)

    # The connected-cluster distance method does not count logical operators of every weight.
    distance_method: Any = "connected_cluster"
    with pytest.raises(ValueError, match="Unknown logical-count method"):
        counter.count_logical_ops(method=distance_method)
    for method in ("brouwer_zimmermann", "macwilliams", "brute_force"):
        assert lc._get_logical_count_method(method) == method
    assert lc._get_logical_count_method("auto") == "brouwer_zimmermann"
    for unknown in (invalid_method, "information_sets", "weight_enumerator"):
        with pytest.raises(ValueError, match="Unknown logical-count method"):
            lc._get_logical_count_method(cast(Any, unknown))
    for weight in (-1, 3):
        with pytest.raises(ValueError, match="weight must be between"):
            counter.count_logical_ops(weight)
    for value in (1.5, True, "1"):
        not_an_integer: Any = value
        with pytest.raises(TypeError, match="weight must be an integer"):
            counter.count_logical_ops(not_an_integer)
    assert counter.count_logical_ops(np.int64(2)) == Counts(2, 1, 1, None)

    steane = lc.CSSLogicalCounter.from_code(codes.SteaneCode())
    wide = lc.CSSLogicalCounter(np.zeros((0, 65)), np.zeros((0, 65)))
    for targets in ([0, 0], [-1, 2], [0, 7], [], list(range(65))):
        with pytest.raises(ValueError, match="1 to 64 distinct"):
            (wide if len(targets) == 65 else steane).deformed(targets, [[1]])
    for values in ([0.5, 2], [True, 2]):
        invalid_targets: Any = values
        with pytest.raises(TypeError, match="Clifford target must be an integer"):
            steane.deformed(invalid_targets, [[1]])
    with pytest.raises(ValueError, match="1 to 64 distinct"):
        steane.deformed_swel(0, 1, 1, 2)
    with pytest.raises(ValueError, match="1 to 64 distinct"):
        steane.deformed_cnot(1, 1)
    not_a_collection: Any = 3
    with pytest.raises(TypeError, match="collection of qubit indices"):
        steane.deformed(not_a_collection, [[1]])


####################################################################################################
# helpers


def _append_frozen_qubits(
    checks_x: npt.ArrayLike, checks_z: npt.ArrayLike, width: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Append qubits fixed in |0> by a Z check each.  No logical of minimum weight touches them."""
    x, z = np.asarray(checks_x, dtype=np.uint8), np.asarray(checks_z, dtype=np.uint8)
    extra = width - x.shape[1]
    padded_x = np.pad(x, ((0, 0), (0, extra)))
    padded_z = np.vstack(
        [np.pad(z, ((0, 0), (0, extra))), np.hstack([np.zeros((extra, z.shape[1])), np.eye(extra)])]
    ).astype(np.uint8)
    return padded_x, padded_z


def _get_expected_counts(code: codes.CSSCode) -> lc.CSSLogicalCounts:
    """Counts at the distance, including full Pauli operators, from an independent enumeration."""
    counts_x, counts_z, distance, full = enumerate_logical_ops(code)
    return Counts(distance, counts_x[distance], counts_z[distance], full)


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


def _assert_reduced(checks: npt.NDArray[np.uint8], matrix: npt.ArrayLike) -> None:
    """Assert that the checks are the reduced row echelon form of a matrix, without zero rows."""
    reduced = galois.GF(2)(np.asarray(matrix, dtype=np.uint8)).row_reduce().view(np.ndarray)
    np.testing.assert_array_equal(checks, reduced[np.any(reduced, axis=1)])
