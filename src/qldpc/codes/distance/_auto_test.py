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
from ._tuning import DEFAULT_TUNING, MAX_VALUE, use_tuning
from .conftest import CostlyStages, get_costly_basis, get_golay_checks, get_sector

# Tunings that skip enumeration and the free phase, with a probe of unlimited work, or of none.
UNLIMITED = {"auto_brute_force_max_words": 0, "auto_free_work": 0, "auto_probe_work": MAX_VALUE}
NO_PROBE = {"auto_brute_force_max_words": 0, "auto_free_work": 0, "auto_probe_work": 0}


def get_golay_problem() -> _auto.BrouwerZimmermannProblem:
    """The sector of ``get_golay_checks``, as a BZ search takes it."""
    checks = get_golay_checks()
    return _auto.BrouwerZimmermannProblem([get_nested_sector(checks, checks)])


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


def test_brouwer_zimmermann_costs_follow_the_closed_form() -> None:
    """A basis of rank ``r`` and width ``n`` certifies ``ceil((t + 1) n / r)`` after layer ``t``.

    The weight is rounded up to the divisor, and work adds over sectors.
    """
    # One basis of rank 4 and width 8 certifies 2 (t + 1) after layer t.
    costs = _auto._BrouwerZimmermannCosts(get_shapes(4, 8))
    assert [costs.get_work(level) for level in (4, 5, 6, 8, 9)] == [4, 10, 10, 14, 15]
    assert costs.get_growth(0, 6) == pytest.approx(math.sqrt(2 / 3))
    assert costs.get_growth(0, 9) == 0

    # Two sectors add their work.  With weights scaled by two and a divisor of two, a basis of rank
    # 3 and width 8 certifies ceil(8 (t + 1) / 3) rounded up to even, 6 and then 8, and one of rank
    # 4 certifies 4, 6, and 8.  So certifying 3 on the scale of the result takes layers 1 and 2, and
    # certifying 4 takes layers 2 and 3.
    sectors = [(np.zeros((rank, 8), np.uint8), np.zeros((rank, 1), np.uint8)) for rank in (3, 4)]
    costs = _auto._BrouwerZimmermannCosts(
        _auto.BrouwerZimmermannProblem(sectors, weight_scale=2, weight_divisor=2)
    )
    assert costs.get_work(3) == 3 + (4 + 6)
    assert costs.get_work(4) == (3 + 3) + (4 + 6 + 4)
    assert costs.get_growth(0, 4) == pytest.approx((1 / 3) ** (3 * 2 / 8))

    # A basis of rank 3 and width 7 certifies ceil(7 (t + 1) / 3), 5 and then 7, which rounding up
    # to the divisor of two raises to 6 and 8, so layer 1 certifies 3 on the scale of the result.
    sectors = [(np.zeros((3, 7), np.uint8), np.zeros((3, 1), np.uint8))]
    costs = _auto._BrouwerZimmermannCosts(
        _auto.BrouwerZimmermannProblem(sectors, weight_scale=2, weight_divisor=2)
    )
    assert costs.get_work(3) == 3 and costs.get_work(4) == 3 + 3


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

    The expected work of a limit is the work spent plus the predicted work of that limit.  The
    expected work of the Golay sector's search is largest at its last limit, above the work spent
    before any limit, so only the prediction can make it probe when the free work is just below the
    largest expected work.
    """
    # Record the spent and the expected states before each limit of the search.
    checks = get_golay_checks()
    clusters = ConnectedClusterSearch([(checks, checks)], cutoff=0)
    spent_and_predicted = []
    while not clusters.finished:
        index, max_weight = clusters.next_limit
        predicted = clusters.predict_states(index, max_weight)
        spent_and_predicted.append((clusters.num_states, clusters.num_states + predicted))
        clusters.advance()
    most = max(total for _, total in spent_and_predicted)
    assert max(spent for spent, _ in spent_and_predicted) < most - 1

    # Probe with a free work just below the largest, and not with one just above it.
    state_cost = 3
    for free_work, probes in (
        (math.ceil(most * state_cost) - 1, [(True, 7)]),
        (math.ceil(most * state_cost), []),
    ):
        routes.probes.clear()
        with use_tuning(
            auto_brute_force_max_words=0,
            auto_free_work=free_work,
            connected_cluster_state_evaluations=state_cost,
        ):
            assert core.get_distance_from_checks(checks, checks, method="auto") == 7
        assert routes.probes == probes


def test_a_cutoff_ends_the_free_phase_at_once_but_switches_only_beyond_the_free_work(
    routes: Routes,
) -> None:
    """A first limit that a cutoff raises ends the free phase at once.

    ``_predict_states`` predicts ``MAX_VALUE - 1`` states for that limit.  The Golay sector's
    search considers a switch only before that limit, where it has no growth to compare, and its
    later limits stay within the free work, so it never switches.  With no free work, the same state
    cost switches.
    """
    checks = get_golay_checks()
    search = functools.partial(core.get_distance_from_checks, checks, checks, method="auto")
    changes = {"auto_probe_work": 0, "connected_cluster_state_evaluations": 10}
    with use_tuning(auto_brute_force_max_words=0, auto_free_work=10**12, **changes):
        assert search(cutoff=4) == 7
    assert routes.probes == [(False, 7)] and routes.switch_bounds == []
    routes.limits.clear()
    with use_tuning(auto_brute_force_max_words=0, auto_free_work=0, **changes):
        assert search(cutoff=4) == 7
    assert len(routes.switch_bounds) == 1 and routes.limits == [4, 5]


def test_the_first_switch_comes_before_the_limit_that_ended_the_free_phase(routes: Routes) -> None:
    """The search decides whether to switch before it runs the limit that ended its free phase.

    With a free work just above the expected work of limit 4 of the Golay sector's search, limit 5
    ends the free phase, and the search switches before running it.
    """
    # Record the expected states before each limit, which do not decrease.
    checks = get_golay_checks()
    clusters = ConnectedClusterSearch([(checks, checks)], cutoff=0)
    totals = {}
    for _ in range(5):
        index, max_weight = clusters.next_limit
        totals[max_weight] = clusters.num_states + clusters.predict_states(index, max_weight)
        clusters.advance()
    assert list(totals) == [1, 2, 3, 4, 5]

    # With a free work just above that of limit 4, limit 5 ends the free phase.
    state_cost = 1000
    free_work = math.floor(totals[4] * state_cost) + 1
    assert max(totals[w] for w in (1, 2, 3, 4)) * state_cost < free_work < totals[5] * state_cost
    changes = {"auto_probe_work": 0, "connected_cluster_state_evaluations": state_cost}
    routes.limits.clear()
    with use_tuning(auto_brute_force_max_words=0, auto_free_work=free_work, **changes):
        assert core.get_distance_from_checks(checks, checks, method="auto") == 7
    assert len(routes.switch_bounds) == 1 and routes.limits == [1, 2, 3, 4]


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
    # Pauli strings, a cutoff below 7 must not return 7, and a cutoff of 7, doubled for the BZ
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
    # before its first weight limit, and the setup of the probe finds a word of weight 7.
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


def test_the_probe_scans_whole_layers_within_its_work() -> None:
    """The probe stops before a layer that would take it over its work."""
    # Measure the work of each layer of a probe of no work, then stop at and around their sums.
    problem = get_golay_problem()
    works = []
    probe = _auto._probe(problem, 0, None, dataclasses.replace(DEFAULT_TUNING, auto_probe_work=0))
    while not probe.finished:
        works.append(probe.get_next_layer_work())
        probe.advance()
    assert len(works) >= 2 and probe.best == 7
    num_layers_in_all = len(works)
    works.append(0)  # a finished probe has no next layer
    first, second = works[0], works[0] + works[1]
    for probe_work, num_layers in ((first - 1, 0), (first, 1), (second - 1, 1), (second, 2)):
        tuning = dataclasses.replace(DEFAULT_TUNING, auto_probe_work=probe_work)
        probe = _auto._probe(problem, 0, None, tuning)
        assert probe.get_next_layer_work() == works[num_layers]
        assert probe.finished is (num_layers == num_layers_in_all)


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

    The work to certify a level counts the states spent and predicted for the limits below it, each
    weighing the state cost.  To reach the probe's bound, the projection adds half the limit at the
    bound, and a bound within the next limit leaves only the level to compare.  The Golay sector's
    search, after its limits 1 to 4, grows faster than a BZ search of a basis of rank and width 16,
    so the state cost decides; these thresholds lie between integers.
    """
    checks = get_golay_checks()
    clusters = ConnectedClusterSearch([(checks, checks)], cutoff=0)
    slow = _auto._BrouwerZimmermannCosts(get_shapes(16, 16))
    assert not _auto._should_switch(clusters, slow, 20, 10**6)
    for _ in range(4):
        clusters.advance()
    growth = clusters.get_growth(0)
    assert clusters.next_limit == (0, 5) and growth is not None
    assert slow.get_growth(0, 6) < growth

    # The level of the next limit, 5, is 6, and the bounds 6 and 8 add the limits up to them.
    spent = clusters.num_states
    predicted = {weight: clusters.predict_states(0, weight) for weight in range(5, 9)}
    level_cost = slow.get_work(6) / (spent + predicted[5])
    for bound, threshold in (
        (5, level_cost),
        (6, slow.get_work(6) / (spent + predicted[5] + predicted[6] / 2)),
        (8, slow.get_work(8) / (spent + sum(predicted[w] for w in (5, 6, 7)) + predicted[8] / 2)),
    ):
        assert 1 < threshold and math.floor(threshold) < threshold
        assert bound == 5 or threshold < level_cost - 1
        for state_cost in (math.floor(threshold), math.ceil(threshold)):
            switches = _auto._should_switch(clusters, slow, bound, state_cost)
            assert switches is (state_cost > threshold)

    # A BZ search of rank 15 and width 16 grows faster than the connected-cluster search at level
    # 5, but not at level 6, so it can lose the comparison.
    straddling = _auto._BrouwerZimmermannCosts(get_shapes(15, 16))
    assert straddling.get_growth(0, 5) > growth > straddling.get_growth(0, 6)
    assert _auto._should_switch(clusters, straddling, 20, 10**6)

    # A BZ search that grows faster never loses the comparison.
    fast = _auto._BrouwerZimmermannCosts(get_shapes(23, 23))
    assert fast.get_growth(0, 6) > growth
    assert not _auto._should_switch(clusters, fast, 20, 10**6)


def test_a_switch_compares_the_work_of_every_search_in_steps_of_its_divisor() -> None:
    """The projection adds every search's limits, each from its next one in steps of its divisor.

    Both sectors of the toric code of size 6 have the divisor 2, so the level of a limit is two
    above it.  A BZ search of two bases of rank and width 16 grows more slowly than the searches.
    """
    code = qldpc.codes.ToricCode(6)
    matrix_x, matrix_z = (
        np.asarray(matrix, dtype=np.uint8) for matrix in (code.matrix_x, code.matrix_z)
    )
    clusters = ConnectedClusterSearch([(matrix_z, matrix_x), (matrix_x, matrix_z)], cutoff=0)
    assert [search.weight_divisor for search in clusters.searches] == [2, 2]

    # After three limits, the next limits are 6 and 4, so the projection to level 8 adds the
    # predictions of limit 6 of the first search and limits 4 and 6 of the second.
    for _ in range(3):
        clusters.advance()
    assert [clusters.get_next_limit(index) for index in (0, 1)] == [6, 4]
    expected = sum(clusters.predict_states(index, w) for index, w in ((0, 6), (1, 4), (1, 6)))
    assert _auto._project_states(clusters, 8) == pytest.approx(expected)

    # After four limits, the next limit is 6, and the bound 6 within it leaves only level 8.
    clusters.advance()
    assert clusters.next_limit == (0, 6)
    slow = _auto._BrouwerZimmermannCosts(
        _auto.BrouwerZimmermannProblem(2 * list(get_shapes(16, 16).sectors))
    )
    growth = clusters.get_growth(0)
    assert growth is not None and slow.get_growth(0, 8) < growth
    predicted = clusters.predict_states(0, 6) + clusters.predict_states(1, 6)
    threshold = slow.get_work(8) / (clusters.num_states + predicted)
    assert 1 < threshold and math.floor(threshold) < threshold
    for state_cost in (math.floor(threshold), math.ceil(threshold)):
        assert _auto._should_switch(clusters, slow, 6, state_cost) is (state_cost > threshold)


def test_a_switch_compares_the_growth_of_the_sector_of_the_next_limit() -> None:
    """The search compares the growths of the sector of its next limit, its own and BZ's.

    Of two Golay sectors, the second has the next limit, after the first has run one more limit,
    and the growth of the first search is larger.  A BZ basis of rank and width 16 grows more
    slowly than either search at level 6, that of limit 5, and one of rank and width 24 between
    the two.  Comparing the growths of the second sector, and only those, gives a switch for the
    first order of the bases and none for the swapped order.  The state cost makes the work
    condition hold, so the growths decide.
    """
    checks = get_golay_checks()
    clusters = ConnectedClusterSearch([(checks, checks), (checks, checks)], cutoff=0)
    for _ in range(9):
        clusters.advance()
    assert clusters.next_limit == (1, 5)
    growths = (clusters.get_growth(0), clusters.get_growth(1))
    assert growths[0] is not None and growths[1] is not None
    small, large = get_shapes(16, 16).sectors[0], get_shapes(24, 24).sectors[0]
    for shapes, switches in (([large, small], True), ([small, large], False)):
        costs = _auto._BrouwerZimmermannCosts(_auto.BrouwerZimmermannProblem(shapes))
        if switches:
            assert costs.get_growth(1, 6) < growths[1] < costs.get_growth(0, 6) < growths[0]
        assert _auto._should_switch(clusters, costs, 20, 10**6) is switches


def test_projections_run_from_the_next_limit_to_the_level() -> None:
    """The projected states add the predictions of each limit from the next to the last below.

    A cutoff raises the next limit of a search that has not started.
    """
    checks = get_golay_checks()
    for cutoff, first in ((0, 5), (6, 6)):
        clusters = ConnectedClusterSearch([(checks, checks)], cutoff=cutoff)
        while clusters.next_limit[1] < 5 and not cutoff:
            clusters.advance()
        assert clusters.get_next_limit(0) == first
        predictions = [clusters.predict_states(0, weight) for weight in range(first, 9)]
        assert _auto._project_states(clusters, 9) == pytest.approx(sum(predictions))
        assert _auto._project_states(clusters, first) == 0
