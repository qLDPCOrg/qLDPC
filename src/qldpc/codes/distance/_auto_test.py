# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the automatic choice of an exact distance search."""

from __future__ import annotations

import dataclasses
import functools
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

from . import _auto, distances
from ._brouwer_zimmermann.distance import BrouwerZimmermannSearch
from ._code_matrices import get_nested_code_basis
from ._connected_cluster.search import ConnectedClusterSearch, SupportSearch
from ._information_sets import basis_families
from ._tuning import DEFAULT_TUNING, MAX_VALUE, use_tuning
from .conftest import CostlyStages, get_costly_basis, get_golay_checks, get_sector

# Tunings that skip enumeration and the free phase, with a probe of unlimited work, or of none.
UNLIMITED = {"auto_brute_force_max_words": 0, "auto_free_work": 0, "auto_probe_work": MAX_VALUE}
NO_PROBE = {"auto_brute_force_max_words": 0, "auto_free_work": 0, "auto_probe_work": 0}

# A tuning with a probe of no work, and a state cost far above the default, so that a search
# switches to BZ sooner.
SWITCH = {**NO_PROBE, "connected_cluster_state_evaluations": 10**6}


def get_nested_sector(
    checks: npt.NDArray[np.uint8], excluded: npt.NDArray[np.uint8]
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """The nested basis and labels of the words that satisfy binary checks, outside the excluded."""
    return get_nested_code_basis(get_kernel(checks), excluded)


def get_shapes(rank: int, width: int) -> _auto.BrouwerZimmermannProblem:
    """A problem of one sector of the given rank and width, which is all that costs read."""
    return _auto.BrouwerZimmermannProblem(
        [(np.zeros((rank, width), np.uint8), np.zeros((rank, 1), np.uint8))]
    )


def test_brute_force_is_cheap_within_its_budget_of_packed_integers() -> None:
    """Enumeration is cheap if it examines at most the budget of integers over all of its spaces.

    The budget is 1024 integers, and each case is a factor of 4 from it: 256 words of one integer
    fit, but 16 such spaces, 256 words of 16 integers, or 4096 words of one bit, which take one
    integer each, do not.
    """
    tuning = dataclasses.replace(DEFAULT_TUNING, auto_brute_force_max_words=1 << 10)
    assert _auto.is_cheap_to_enumerate([8], 64, tuning)
    assert not _auto.is_cheap_to_enumerate([8] * 16, 64, tuning)
    assert not _auto.is_cheap_to_enumerate([8], 16 * 64, tuning)
    assert not _auto.is_cheap_to_enumerate([12], 1, tuning)


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
        assert distances.get_distance_from_checks(checks, checks, method="auto") == 7
        assert distances.get_distance_pauli(surface_code.matrix, method="auto") == 3
    assert routes.probes == routes.switch_bounds == []


def test_the_free_phase_ends_with_the_expected_work(
    monkeypatch: pytest.MonkeyPatch, routes: Routes
) -> None:
    """The search probes once the expected work of a limit, at a state cost, exceeds the free work.

    The expected work of a limit is the work spent plus the predicted work of that limit, both at
    the state cost.  The Golay sector's search visits fewer than ``10**4`` states, so with a free
    work of ``3 * 10**8``, it ends in its free phase at a state cost of 1.  At a state cost of
    ``10**6``, it probes before a limit whose predicted work exceeds the rest of the free work,
    while the work spent is still within it.
    """
    spent: list[int] = []
    spent_at_probes: list[int] = []
    predict_states = ConnectedClusterSearch.predict_states
    probe = _auto._probe

    def record_spent(search: ConnectedClusterSearch, *args: Any) -> float:
        spent.append(search.num_states)
        return predict_states(search, *args)

    def record_probe(*args: Any) -> BrouwerZimmermannSearch:
        spent_at_probes.append(spent[-1])
        return probe(*args)

    monkeypatch.setattr(ConnectedClusterSearch, "predict_states", record_spent)
    monkeypatch.setattr(_auto, "_probe", record_probe)
    checks = get_golay_checks()
    free_work = 3 * 10**8
    for state_cost, num_probes in ((1, 0), (10**6, 1)):
        spent_at_probes.clear()
        with use_tuning(
            auto_brute_force_max_words=0,
            auto_free_work=free_work,
            connected_cluster_state_evaluations=state_cost,
        ):
            assert distances.get_distance_from_checks(checks, checks, method="auto") == 7
        assert len(spent_at_probes) == len(routes.probes) == num_probes
        assert all(states * state_cost <= free_work for states in spent_at_probes)
        routes.probes.clear()


def test_a_cutoff_ends_the_free_phase_at_once(routes: Routes) -> None:
    """A first limit that a cutoff raises ends the free phase at once, but cannot make a switch.

    ``_predict_states`` predicts ``MAX_VALUE - 1`` states for that limit, whatever the free work, so
    the search probes before it.  That prediction would exceed the work of BZ, but the search has
    no growth yet, so it does not switch, and searches on to the distance.
    """
    checks = get_golay_checks()
    with use_tuning(auto_brute_force_max_words=0, auto_free_work=10**12, auto_probe_work=0):
        assert distances.get_distance_from_checks(checks, checks, method="auto", cutoff=4) == 7
    assert len(routes.probes) == 1 and routes.switch_bounds == []


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
            functools.partial(distances.get_distance_from_checks, checks, checks, method="auto"),
            functools.partial(distances.get_distance_pauli, pauli_checks, method="auto"),
        ):
            assert search() == 7
            assert search(max_weight=10) == 7
            assert search(max_weight=5) == 6
            assert search(cutoff=7) == 7
            assert 7 <= search(cutoff=9) <= 9
        assert distances.get_distance_pauli(bacon_shor_code.matrix, method="auto") == 3
    assert [finished for finished, _ in routes.probes] == [True] * 11 and routes.switch_bounds == []

    # The lightest nontrivial row of the basis of the probe of hgp_checks weighs 14 as a BZ word, or
    # 7 as a Pauli string, more than the distance 4.  The setup of the probe's search finds a
    # witness of the distance, which joins the probe's bound only once that search advances.  So a
    # probe that advances no layer has the bound 14 in the weights of BZ words, or 7 in those of
    # Pauli strings.  A cutoff below 7 must not return 7, and a cutoff of 7, doubled for the BZ
    # words, returns 7 before the setup.
    with use_tuning(**NO_PROBE):
        distances.get_distance_pauli(hgp_checks, method="auto")
    assert routes.probes[-1] == (False, 14)

    # Each finished probe's bound, a weight of BZ words, is even and lies between twice the
    # distance and twice the cutoff.
    routes.probes.clear()
    with use_tuning(**UNLIMITED):
        assert distances.get_distance_pauli(hgp_checks, method="auto", cutoff=4) == 4
        assert 4 <= distances.get_distance_pauli(hgp_checks, method="auto", cutoff=6) <= 6
        assert distances.get_distance_pauli(hgp_checks, method="auto", cutoff=7) == 7
    for (finished, best), cutoff in zip(routes.probes, (4, 6, 7), strict=True):
        assert finished and 8 <= best <= 2 * cutoff and best % 2 == 0

    # A CSS search passes its cutoff on too: with the cutoff 7, a search of the Golay code probes
    # before its first weight limit, and a row of the probe's basis weighs 7, so the probe finishes
    # before its setup.
    routes.probes.clear()
    routes.limits.clear()
    with use_tuning(**NO_PROBE):
        assert (
            distances.get_distance_css(golay.matrix_x, golay.matrix_z, method="auto", cutoff=7) == 7
        )
    assert [finished for finished, _ in routes.probes] == [True] and routes.limits == []


def test_an_unfinished_probe_caps_the_search_by_its_bound(routes: Routes) -> None:
    """The probe's bound ends the connected-cluster search.

    The setup of the probe of the surface code, which advances no layer, finds a word of weight 6,
    the distance.  So the connected-cluster search finds no witness itself: it stops once its lower
    bounds reach the probe's bound.  The search of each sector, whose weight divisor is 2, reaches
    that bound after its limits 2 and 4, and has no growth before either, so it cannot switch.
    """
    surface_code = qldpc.codes.SurfaceCode(6)
    with use_tuning(**NO_PROBE):
        distance = distances.get_distance_css(
            surface_code.matrix_x, surface_code.matrix_z, method="auto"
        )
    assert distance == 6
    assert routes.probes == [(False, 6)]
    assert routes.switch_bounds == routes.witnesses == []


def test_a_switch_takes_the_probe_bound_and_finds_the_distance(routes: Routes) -> None:
    """A switch to BZ starts again with the full tuning and the bound of the probe.

    With a probe that does nothing, and a state cost far above the default, the Golay sector's
    search switches, as binary words and as Pauli strings, also with an upper bound or a cutoff
    below the distance.  The bound of a switch is in the weights of BZ words, which are twice those
    of Pauli strings.
    """
    checks = get_golay_checks()
    golay = qldpc.codes.CSSCode(checks.astype(int), checks.astype(int))
    pauli_checks = golay.conjugated(range(0, 23, 2)).matrix
    with use_tuning(**SWITCH) as tuning:
        for search in (
            functools.partial(distances.get_distance_from_checks, checks, checks, method="auto"),
            functools.partial(distances.get_distance_pauli, pauli_checks, method="auto"),
        ):
            assert search() == 7
            assert search(max_weight=5) == 6
            assert search(cutoff=3) == 7
    probe_bounds = [best for finished, best in routes.probes if not finished]
    assert len(routes.switch_bounds) == 6 and routes.switch_bounds == probe_bounds

    # The bounds of the Pauli strings are even and at least twice the distance.  With
    # ``max_weight=5``, the bound is instead twice ``max_weight + 1``.
    plain, with_max_weight, with_cutoff = routes.switch_bounds[3:]
    assert with_max_weight == 12
    assert all(bound >= 14 and bound % 2 == 0 for bound in (plain, with_cutoff))
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
    search_css = functools.partial(distances.get_distance_css, golay.matrix_x, golay.matrix_z)
    search_golay = functools.partial(
        distances.get_distance_from_checks,
        golay.matrix,
        golay.get_stabilizer_ops(),
        symplectic=True,
    )
    search_bicycle = functools.partial(
        distances.get_distance_from_checks,
        conjugated.matrix,
        conjugated.get_stabilizer_ops(),
        symplectic=True,
    )
    asks = {"automorphism_minimum_states": MAX_VALUE, "automorphism_minimum_layer_work": 0}
    for changes, search, automorphisms, distance in (
        (SWITCH, search_css, shifts, 7),
        (SWITCH, search_golay, shifts, 7),
        (UNLIMITED, search_bicycle, bicycle_code._get_permutation_automorphisms(), 3),
    ):
        num_switches = len(routes.switch_bounds)
        get_automorphisms = mock.Mock(return_value=automorphisms)
        with use_tuning(**changes, **asks):
            assert search(method="auto", automorphisms=get_automorphisms) == distance
        get_automorphisms.assert_called_once()
        if changes is SWITCH:
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


def test_a_switch_compares_work_to_a_level_or_with_faster_growth_to_the_probe_bound() -> None:
    """A search switches if it would work more than BZ to a level, or, growing faster, to a bound.

    It compares nothing before it has a growth: a first limit that a cutoff raises predicts a
    placeholder number of states, far above the BZ work here.

    After its limits 1 to 4, the Golay sector's search grows faster than a BZ search of a basis of
    rank and width 16, by a factor of more than 11 / 6 per unit of weight, and more slowly than one
    of rank and width 23.  Its states up to its next limit, even with half of those predicted for
    the limit after it, are fewer than the work of either BZ search to certify its level.  So a
    state cost of 1 switches neither for that level nor for the bound 6, which equals that level,
    but one of ``10**6`` does, whatever the growths.

    To reach the bound 23, the length of the code, its projection includes all of the states
    predicted for its limit 22 and half of those for 23.  The search visits at least one state at
    its limit 4, so this projection is more than ``(11 / 6)**18 + (11 / 6)**19 / 2``.  That exceeds
    the ``2**16 - 1`` supports of all of the layers of the BZ search of rank 16, so a state cost of
    1 switches with that bound.  The projection also exceeds the ``2**23 - 2`` supports of the BZ
    search of rank 23, which grows faster, so only the growth test keeps the search from switching.
    """
    checks = get_golay_checks()
    slow = _auto._BrouwerZimmermannCosts(get_shapes(16, 16))
    fast = _auto._BrouwerZimmermannCosts(get_shapes(23, 23))
    with_cutoff = ConnectedClusterSearch([(checks, checks)], cutoff=5)
    assert not _auto._should_switch(with_cutoff, slow, 20, 1)
    clusters = ConnectedClusterSearch([(checks, checks)], cutoff=0)
    for _ in range(4):
        clusters.advance()
    assert clusters.next_limit == (0, 5)
    growth = clusters.get_growth(0)
    assert growth is not None and 11 / 6 < growth < fast.get_growth(0, 6)

    # Two copies of the sector project twice the states.
    pair = ConnectedClusterSearch([(checks, checks), (checks, checks)], cutoff=0)
    for _ in range(8):
        pair.advance()
    assert _auto._project_states(pair, 8) == pytest.approx(2 * _auto._project_states(clusters, 8))

    # The level of its next limit, and the bound 6, which equals that level.
    for costs in (slow, fast):
        assert not _auto._should_switch(clusters, costs, 6, 1)
        assert _auto._should_switch(clusters, costs, 6, 10**6)

    # The bound 23.
    assert _auto._should_switch(clusters, slow, 23, 1)
    assert _auto._project_states(clusters, 23) > fast.get_work(23)
    assert not _auto._should_switch(clusters, fast, 23, 1)

    # A second BZ sector, of rank and width 200, keeps the growth of the first, but its work grows
    # by more than 30 times per unit of weight: about C(200, 5) to certify the level 6, and
    # C(200, 7) to reach the bound 8.  The search's states, about 600 to the level and 7000 to the
    # bound, grow 12 times.  At a state cost of 4 * 10**7, their work exceeds the BZ work to the
    # level by a factor of about 9, and falls short of the BZ work to the bound by a factor of about
    # 8.  So only the comparison at the level switches.
    steep = _auto._BrouwerZimmermannCosts(
        _auto.BrouwerZimmermannProblem([*get_shapes(16, 16).sectors, *get_shapes(200, 200).sectors])
    )
    assert _auto._should_switch(clusters, steep, 8, 4 * 10**7)
