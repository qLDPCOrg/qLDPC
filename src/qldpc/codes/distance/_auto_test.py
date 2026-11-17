# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the automatic choice of an exact distance search."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Iterator
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y, z

import qldpc
from qldpc._binary import get_kernel
from qldpc.objects import Pauli

from . import _auto, core
from ._brouwer_zimmermann.distance import BrouwerZimmermannSearch
from ._connected_cluster.search import ConnectedClusterSearch, SupportSearch
from ._information_sets import basis_families
from ._tuning import DEFAULT_TUNING, MAX_VALUE, SearchTuning, use_tuning
from .conftest import CostlyStages, get_costly_basis, get_golay_checks, get_sector

# Tunings that skip enumeration and the free phase, with a probe of unlimited work, or of none.
UNLIMITED = {"auto_brute_force_max_words": 0, "auto_free_work": 0, "auto_probe_work": MAX_VALUE}
NO_PROBE = {"auto_brute_force_max_words": 0, "auto_free_work": 0, "auto_probe_work": 0}


def get_nested_sector(
    checks: npt.NDArray[np.uint8], excluded: npt.NDArray[np.uint8]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """The nested basis and labels of the words that satisfy binary checks, outside the excluded."""
    return qldpc.codes.distance._binary.get_nested_code_basis(get_kernel(checks), excluded)


def get_shapes(rank: int, width: int) -> _auto.BrouwerZimmermannProblem:
    """A problem of one sector of the given rank and width, which is all that costs read."""
    return _auto.BrouwerZimmermannProblem(
        [(np.zeros((rank, width), np.uint8), np.zeros((rank, 1), np.uint8))]
    )


def test_brute_force_is_cheap_up_to_its_budget_of_packed_integers() -> None:
    """Enumeration is cheap if it examines at most the budget of integers over all of its spaces."""
    tuning = dataclasses.replace(DEFAULT_TUNING, auto_brute_force_max_words=1 << 10)
    assert _auto.is_cheap_to_enumerate([10], 64, tuning)
    assert not _auto.is_cheap_to_enumerate([10], 65, tuning)
    assert _auto.is_cheap_to_enumerate([9, 9], 64, tuning)
    assert not _auto.is_cheap_to_enumerate([9, 9, 0], 64, tuning)
    assert _auto.is_cheap_to_enumerate([8], 256, tuning)


def test_brouwer_zimmermann_costs_grow_with_the_level_and_add_over_sectors() -> None:
    """The estimated work grows with the level, adds over sectors, and scales with the weights.

    A basis of rank 4 and width 8 certifies 4, 6, and 8 after its layers 1 to 3, so no layer
    certifies 9, which then has no growth.  The level of a Pauli string is half that of its BZ word.
    """
    costs = _auto._BrouwerZimmermannCosts(get_shapes(4, 8))
    assert 0 < costs.get_work(4) < costs.get_work(6) < costs.get_work(8)
    assert costs.get_growth(0, 6) > 0 and costs.get_growth(0, 9) == 0
    pair = _auto._BrouwerZimmermannCosts(
        _auto.BrouwerZimmermannProblem(2 * list(get_shapes(4, 8).sectors))
    )
    assert pair.get_work(6) == 2 * costs.get_work(6)
    sectors = get_shapes(5, 12).sectors
    paulis = _auto._BrouwerZimmermannCosts(_auto.BrouwerZimmermannProblem(sectors, weight_scale=2))
    words = _auto._BrouwerZimmermannCosts(_auto.BrouwerZimmermannProblem(sectors))
    assert [paulis.get_work(level) for level in (3, 5)] == [words.get_work(6), words.get_work(10)]


@dataclasses.dataclass
class Routes:
    """What searches by cost did: their probes, switches, weight limits, and witnesses.

    Attributes:
        probes: Whether each probe finished, and its best weight.
        switch_bounds: The upper bound of each switch to a BZ search, in the weights of BZ words.
        switch_sectors: The sectors of each switch.
        switch_tunings: The tuning of each switch.
        limits: The weight limit of each call of ``SupportSearch.find_witness``, in order.
        witnesses: The weight of each witness that a connected-cluster search found.
    """

    probes: list[tuple[bool, int]] = dataclasses.field(default_factory=list)
    switch_bounds: list[int] = dataclasses.field(default_factory=list)
    switch_sectors: list[Any] = dataclasses.field(default_factory=list)
    switch_tunings: list[Any] = dataclasses.field(default_factory=list)
    limits: list[int] = dataclasses.field(default_factory=list)
    witnesses: list[int] = dataclasses.field(default_factory=list)


@pytest.fixture
def routes(monkeypatch: pytest.MonkeyPatch) -> Iterator[Routes]:
    """Record the routes of searches by cost, through pass-through wrappers."""
    recorded = Routes()
    probe = _auto._probe
    get_distance = _auto.get_distance_brouwer_zimmermann
    find_witness = SupportSearch.find_witness

    def recording_probe(*args: Any) -> BrouwerZimmermannSearch:
        search = probe(*args)
        recorded.probes.append((search.finished, search.best))
        return search

    def recording_switch(sectors: Any, **kwargs: Any) -> int:
        recorded.switch_bounds.append(kwargs["upper_bound"])
        recorded.switch_sectors.append(sectors)
        recorded.switch_tunings.append(kwargs["tuning"])
        return get_distance(sectors, **kwargs)

    def recording_find_witness(
        search: SupportSearch, max_weight: int
    ) -> npt.NDArray[np.int64] | None:
        recorded.limits.append(max_weight)
        witness = find_witness(search, max_weight)
        if witness is not None:
            recorded.witnesses.append(len(witness))
        return witness

    monkeypatch.setattr(_auto, "_probe", recording_probe)
    monkeypatch.setattr(_auto, "get_distance_brouwer_zimmermann", recording_switch)
    monkeypatch.setattr(SupportSearch, "find_witness", recording_find_witness)
    yield recorded


def test_a_search_by_cost_can_end_in_its_free_phase(routes: Routes) -> None:
    """A connected-cluster search that stays within its free work neither probes nor switches."""
    checks = get_golay_checks()
    surface_code = qldpc.codes.SurfaceCode(3).conjugated([0, 4])
    with use_tuning(auto_brute_force_max_words=0):
        assert core.get_distance_from_checks(checks, checks, method="auto") == 7
        assert core.get_distance_pauli(surface_code.matrix, method="auto") == 3
    assert routes.probes == routes.switch_bounds == []


def test_the_free_phase_ends_with_the_expected_work(routes: Routes) -> None:
    """The search probes once the expected work of a limit, at a state cost, exceeds the free work.

    The expected work of a limit is the work spent plus the predicted work of that limit.  A free
    work above the work spent before every limit of the Golay sector's search, but below the
    largest expected work, makes it probe only through the prediction.  The searches add no sums of
    pairs of checks, with which the work spent before the last limit is too close to its expected
    work for such a free work.
    """
    checks = get_golay_checks()
    tuning = SearchTuning(pair_sum_checks=False)
    clusters = ConnectedClusterSearch([(checks, checks)], cutoff=0, tuning=tuning)
    spent, expected = [], []
    while not clusters.finished:
        index, max_weight = clusters.next_limit
        spent.append(clusters.num_states)
        expected.append(clusters.num_states + clusters.predict_states(index, max_weight))
        clusters.advance()
    assert 2 * max(spent) < max(expected)

    state_cost = 3
    for free_work, probes in (
        (state_cost * (max(spent) + max(expected)) / 2, [(True, 7)]),
        (state_cost * 2 * max(expected), []),
    ):
        routes.probes.clear()
        with use_tuning(
            auto_brute_force_max_words=0,
            auto_free_work=int(free_work),
            connected_cluster_state_evaluations=state_cost,
            pair_sum_checks=False,
        ):
            assert core.get_distance_from_checks(checks, checks, method="auto") == 7
        assert routes.probes == probes


def test_a_cutoff_ends_the_free_phase_at_once(routes: Routes) -> None:
    """A first limit that a cutoff raises ends the free phase at once, whatever the free work.

    ``_predict_states`` predicts ``MAX_VALUE - 1`` states for that limit, so the search probes
    before it.
    """
    checks = get_golay_checks()
    with use_tuning(auto_brute_force_max_words=0, auto_free_work=10**12, auto_probe_work=0):
        assert core.get_distance_from_checks(checks, checks, method="auto", cutoff=4) == 7
    assert routes.probes == [(False, 7)] and routes.switch_bounds == []


def test_a_finished_probe_gives_the_distance(routes: Routes) -> None:
    """A probe that finishes within its work gives the result, also by its cutoff or upper bound.

    Pauli strings take a probe of their images, whose weights, cutoff, and upper bound are doubled.
    """
    checks = get_golay_checks()
    golay = qldpc.codes.CSSCode(checks.astype(int), checks.astype(int))
    pauli_checks = golay.conjugated(range(0, 23, 2)).matrix
    bacon_shor_code = qldpc.codes.BaconShorCode(3)
    hgp_code = qldpc.codes.HGPCode(qldpc.codes.RepetitionCode(4), qldpc.codes.RepetitionCode(5))
    hgp_checks = hgp_code.conjugated(range(0, len(hgp_code), 2)).matrix
    with use_tuning(**UNLIMITED):
        for search in (
            functools.partial(core.get_distance_from_checks, checks, checks, method="auto"),
            functools.partial(core.get_distance_pauli, pauli_checks, method="auto"),
        ):
            assert search() == 7
            assert search(max_weight=10) == 7
            assert search(max_weight=5) == 6
            assert search(cutoff=7) == 7
            assert 7 <= search(cutoff=9) <= 9
        assert core.get_distance_pauli(bacon_shor_code.matrix, method="auto") == 3
    assert [finished for finished, _ in routes.probes] == [True] * 11 and routes.switch_bounds == []

    # The lightest nontrivial row of the basis of the probe of hgp_checks weighs 14 as a BZ word, or
    # 7 as a Pauli string, more than the distance 4.  The setup of the probe's search finds a
    # witness of the distance, which joins the probe's bound only once that search advances.  So a
    # probe that advances no layer has the bound 14 in the weights of BZ words, or 7 in those of
    # Pauli strings.  A cutoff below 7 must not return 7, and a cutoff of 7, doubled for the BZ
    # words, returns 7 before the setup.
    with use_tuning(**NO_PROBE):
        core.get_distance_pauli(hgp_checks, method="auto")
    assert routes.probes[-1] == (False, 14)
    routes.probes.clear()
    with use_tuning(**UNLIMITED):
        assert core.get_distance_pauli(hgp_checks, method="auto", cutoff=4) == 4
        assert 4 <= core.get_distance_pauli(hgp_checks, method="auto", cutoff=6) <= 6
        assert core.get_distance_pauli(hgp_checks, method="auto", cutoff=7) == 7
    assert routes.probes == [(True, 8), (True, 8), (True, 14)]

    # A CSS search passes its cutoff on too: with the cutoff 7, a search of the Golay code probes
    # before its first weight limit, and a row of the probe's basis weighs 7, so the probe finishes
    # before its setup.
    routes.probes.clear()
    routes.limits.clear()
    with use_tuning(**NO_PROBE):
        assert core.get_distance_css(golay.matrix_x, golay.matrix_z, method="auto", cutoff=7) == 7
    assert routes.probes == [(True, 7)] and routes.limits == []


def test_an_unfinished_probe_caps_the_search_by_its_bound(routes: Routes) -> None:
    """The probe's bound ends the connected-cluster search.

    The search of the surface code finds no witness itself: it stops once its lower bounds reach the
    probe's bound.
    """
    surface_code = qldpc.codes.SurfaceCode(5)
    with use_tuning(**NO_PROBE):
        distance = core.get_distance_css(
            surface_code.matrix_x, surface_code.matrix_z, method="auto"
        )
    assert distance == 5
    assert routes.probes == [(False, 5)] and routes.switch_bounds == routes.witnesses == []


def test_a_switch_takes_the_probe_bound_and_finds_the_distance(routes: Routes) -> None:
    """A switch to BZ starts again with the full tuning and the bound of the probe.

    With a probe that does nothing, the Golay sector's search switches, as binary words and as
    Pauli strings, also with an upper bound or a cutoff below the distance.  The bound of a switch
    is in the weights of BZ words, which are twice those of Pauli strings.
    """
    checks = get_golay_checks()
    golay = qldpc.codes.CSSCode(checks.astype(int), checks.astype(int))
    pauli_checks = golay.conjugated(range(0, 23, 2)).matrix
    with use_tuning(**NO_PROBE) as tuning:
        for search in (
            functools.partial(core.get_distance_from_checks, checks, checks, method="auto"),
            functools.partial(core.get_distance_pauli, pauli_checks, method="auto"),
        ):
            assert search() == 7
            assert search(max_weight=5) == 6
            assert search(cutoff=3) == 7
    probe_bounds = [best for finished, best in routes.probes if not finished]
    assert len(routes.switch_bounds) == 6 and routes.switch_bounds == probe_bounds
    assert routes.switch_bounds[3:] == [14, 12, 14]
    assert all(switch_tuning == tuning for switch_tuning in routes.switch_tunings)


def test_a_switch_searches_every_sector(routes: Routes) -> None:
    """A search of several sectors switches to a BZ search of all of them, in the same order.

    The Golay sector has distance 7, and the sector of the costly basis, of a classical code, 6.
    """
    checks, basis = get_golay_checks(), get_costly_basis()
    cluster_sectors = [(checks, checks), (get_kernel(basis), np.zeros((0, 21), np.uint8))]
    problem = _auto.BrouwerZimmermannProblem(
        [get_nested_sector(checks, checks), (basis, np.empty((len(basis), 0), np.uint8))]
    )
    tuning = dataclasses.replace(
        DEFAULT_TUNING,
        auto_free_work=0,
        auto_probe_work=0,
        connected_cluster_state_evaluations=10**6,
    )
    distance = _auto.search_by_cost(
        cluster_sectors,
        problem,
        symplectic=False,
        cutoff=0,
        upper_bound=None,
        get_automorphisms=None,
        tuning=tuning,
    )
    assert distance == 6
    (switched,) = routes.switch_sectors
    assert len(switched) == len(problem.sectors) == 2
    for (switched_basis, labels), (expected_basis, expected_labels) in zip(
        switched, problem.sectors
    ):
        assert np.array_equal(switched_basis, expected_basis)
        assert np.array_equal(labels, expected_labels)


def test_searches_by_cost_use_lifted_automorphisms(routes: Routes) -> None:
    """The probe and the switched search verify and use automorphisms, lifted for Pauli strings.

    The shifts of the Golay code, and the translations of a bicycle code with Hadamards on its
    second block, which is not CSS, preserve the codes.  A transposition does not, and fails the
    verification.  The Golay code is searched by its two sectors and as Pauli strings.  Only the BZ
    searches ask for automorphisms here.
    """
    checks = get_golay_checks()
    golay = qldpc.codes.CSSCode(checks.astype(int), checks.astype(int))
    shifts = np.roll(np.arange(23), 1)[None]
    bicycle_code = qldpc.codes.MBCode([7], 1 + x + x**3, 1 + x**2 + x**3 + x**4)
    conjugated = bicycle_code.conjugated(range(7, 14))
    search_css = functools.partial(core.get_distance_css, golay.matrix_x, golay.matrix_z)
    search_golay = functools.partial(
        core.get_distance_from_checks, golay.matrix, golay.get_stabilizer_ops(), symplectic=True
    )
    search_bicycle = functools.partial(
        core.get_distance_from_checks,
        conjugated.matrix,
        conjugated.get_stabilizer_ops(),
        symplectic=True,
    )
    asks = {"automorphism_minimum_states": MAX_VALUE, "automorphism_minimum_layer_work": 0}
    for changes, search, automorphisms, distance in (
        (NO_PROBE, search_css, shifts, 7),
        (NO_PROBE, search_golay, shifts, 7),
        (UNLIMITED, search_bicycle, bicycle_code._get_qudit_automorphisms(), 3),
    ):
        num_switches = len(routes.switch_bounds)
        get_automorphisms = mock.Mock(return_value=automorphisms)
        with use_tuning(**changes, **asks):
            assert search(method="auto", automorphisms=get_automorphisms) == distance
        get_automorphisms.assert_called_once()
        if changes is NO_PROBE:
            assert len(routes.switch_bounds) == num_switches + 1
        else:
            assert routes.probes[-1][0]

        transposition = np.arange(len(automorphisms[0]))
        transposition[[0, 1]] = [1, 0]
        with use_tuning(**changes, **asks), pytest.raises(ValueError, match="not an automorphism"):
            search(method="auto", automorphisms=transposition[None])


def test_the_probe_skips_the_costly_stages_of_setup(
    monkeypatch: pytest.MonkeyPatch, costly_stages: CostlyStages
) -> None:
    """The probe takes no overlapping or partitioned family, and so no certificate from LP.

    The tuning asks for those stages.  A BZ search of a sector of this MB code then takes an
    overlapping family, and one of this basis a partitioned family with a certificate from LP, but
    their probes do not.
    """
    overlapping_families: list[basis_families.BasisFamily] = []
    get_overlapping_family = basis_families.get_overlapping_family

    def recording_get_overlapping_family(*args: Any, **kwargs: Any) -> basis_families.BasisFamily:
        family = get_overlapping_family(*args, **kwargs)
        overlapping_families.append(family)
        return family

    monkeypatch.setattr(basis_families, "get_overlapping_family", recording_get_overlapping_family)
    eager = dataclasses.replace(
        DEFAULT_TUNING,
        overlapping_minimum_work=0,
        partition_work_ratio=0,
        lp_minimum_work=0,
        max_information_sets=16,
        auto_probe_work=0,
    )
    mb_code = qldpc.codes.MBCode({x: 4, y: 4, z: 2}, 1 + x + y + x * y * z, 1 + y + x * z + x * y)
    basis = get_costly_basis()
    for sector, families, uses_lp in (
        (get_sector(mb_code, Pauli.X), overlapping_families, False),
        ((basis, np.empty((len(basis), 0), np.uint8)), costly_stages.partitioned_families, True),
    ):
        recordings = (overlapping_families, *costly_stages)
        for recording in recordings:
            recording.clear()
        problem = _auto.BrouwerZimmermannProblem([sector])
        BrouwerZimmermannSearch(problem.sectors, cutoff=0, tuning=eager)
        assert any(len(family.ranks) for family in families)
        certificates = costly_stages.search_certificates
        assert any(certificate.pivot_blocks for certificate in certificates) is uses_lp

        for recording in recordings:
            recording.clear()
        _auto._probe(problem, 0, None, eager)
        assert families and not any(len(family.ranks) for family in families)
        assert certificates and not any(certificate.pivot_blocks for certificate in certificates)


def test_a_switch_needs_faster_growth_and_more_work_to_a_level_or_to_the_probe_bound() -> None:
    """The search switches while it grows faster than BZ, once it would work more than BZ.

    The work to certify a level counts the states spent and projected for the limits of every
    sector below it, each weighing the state cost, and the projection to the probe's bound adds
    half the limit at the bound.  After its limits 1 to 4, the Golay sector's search grows faster
    than a BZ search of a basis of rank and width 16, and more slowly than one of rank and width
    23.  A state cost between the thresholds of level 6 and of the bound 8 switches only by the
    bound.
    """
    checks = get_golay_checks()
    clusters = ConnectedClusterSearch([(checks, checks)], cutoff=0)
    slow = _auto._BrouwerZimmermannCosts(get_shapes(16, 16))
    assert not _auto._should_switch(clusters, slow, 20, 10**6)
    for _ in range(4):
        clusters.advance()
    assert clusters.next_limit == (0, 5)

    # Two copies of the sector project twice the states.
    pair = ConnectedClusterSearch([(checks, checks), (checks, checks)], cutoff=0)
    for _ in range(8):
        pair.advance()
    assert _auto._project_states(pair, 8) == pytest.approx(2 * _auto._project_states(clusters, 8))

    spent = clusters.num_states
    predicted = {weight: clusters.predict_states(0, weight) for weight in range(5, 9)}
    level_threshold = slow.get_work(6) / (spent + predicted[5])
    bound_threshold = slow.get_work(8) / (
        spent + sum(predicted[w] for w in (5, 6, 7)) + predicted[8] / 2
    )
    assert 2 * bound_threshold < level_threshold
    state_cost = round(math.sqrt(level_threshold * bound_threshold))
    assert not _auto._should_switch(clusters, slow, 5, state_cost)
    assert _auto._should_switch(clusters, slow, 8, state_cost)
    assert _auto._should_switch(clusters, slow, 5, math.ceil(2 * level_threshold))

    fast = _auto._BrouwerZimmermannCosts(get_shapes(23, 23))
    assert not _auto._should_switch(clusters, fast, 20, 10**6)
