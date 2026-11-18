# SPDX-License-Identifier: Apache-2.0

"""Tests for the routes of the BZ coset searches."""

from __future__ import annotations

import pickle
import threading
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import get_kernel

from .. import _cosets
from .._tuning import DEFAULT_TUNING
from ..conftest import (
    EVEN_ROUTES_CODE,
    OVERLAPPING_ROUTES_CODE,
    THREAD_TIMEOUT,
    TWO_FAMILIES_CODE,
    get_set_ranks,
)
from . import affine_instance, coset_routes, coset_tables

# Two thresholds of the route gate, and maximum weights of queries of OVERLAPPING_ROUTES_CODE.  At
# the middle weight, queries predict far fewer examinations than the high threshold with the greedy
# packing.  They predict more than the other threshold with the greedy packing, but fewer with the
# overlapping family, which they take.  At the heavy weights, queries predict far more examinations
# than the other threshold with the greedy packing and with the family that they take: the
# overlapping family at the first two heavy weights, and the greedy packing at the last.  The
# examinations of the families of small codes differ too little for the middle weight to be far
# from the other threshold.
_HIGH_THRESHOLD, _THRESHOLD = 64, 8
_MIDDLE_WEIGHT = 1
_OTHER_FAMILY_WEIGHTS = (3, 4)
_HEAVY_WEIGHTS = (*_OTHER_FAMILY_WEIGHTS, 5)

# A code of five rows whose greedy packing leaves a partial set of rank two.
_DEFICIENT_ROUTES_CODE = np.hstack(
    [np.eye(5, dtype=np.uint8), np.array([[1, 1], [1, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)]
)

# A code of two rows, so that its only layer of two rows is the last, and its sets need tables only
# at the maximum weights whose queries reach that layer.
_TWO_ROWS_CODE = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)


def test_the_gate_keeps_the_greedy_packing_far_below_its_threshold_and_not_far_above(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queries far below the threshold take the greedy packing, and build no other family.

    Queries far above it take another family at some maximum weights.
    """
    # Far below the high threshold, queries take the greedy packing and build no other family.
    families = _spy_on_families(monkeypatch)
    high_routes = _build_routes(OVERLAPPING_ROUTES_CODE, _HIGH_THRESHOLD)
    disjoint = (_get_instance(families, high_routes.disjoint_tables), high_routes.disjoint_tables)
    assert 4 * high_routes._get_examinations(*disjoint, _MIDDLE_WEIGHT) <= _HIGH_THRESHOLD
    assert high_routes.get_route(_MIDDLE_WEIGHT)[0] is high_routes.disjoint_tables
    assert len(families) == 1

    # At the other threshold, the same queries take another family, so only the gate kept the greedy
    # packing.  Far above that threshold, queries take the overlapping family, whose sets share
    # pivots, at the first two heavy weights, and the greedy packing at the last.
    routes = _build_routes(OVERLAPPING_ROUTES_CODE, _THRESHOLD)
    disjoint = (_get_instance(families, routes.disjoint_tables), routes.disjoint_tables)
    assert routes.get_route(_MIDDLE_WEIGHT)[0] is not routes.disjoint_tables
    for max_weight in _HEAVY_WEIGHTS:
        assert routes._get_examinations(*disjoint, max_weight) >= 4 * _THRESHOLD
        tables, _ = routes.get_route(max_weight)
        if max_weight in _OTHER_FAMILY_WEIGHTS:
            pivots = tables.pivots[tables.pivots >= 0]
            assert len(np.unique(pivots)) < len(pivots)
        else:
            assert tables is routes.disjoint_tables


@pytest.mark.parametrize(
    "code", [EVEN_ROUTES_CODE, TWO_FAMILIES_CODE], ids=["even", "two-families"]
)
def test_queries_take_each_family_at_some_maximum_weight(code: npt.NDArray[np.uint8]) -> None:
    """At the threshold zero, the queries of some maximum weight take each family."""
    # At the threshold zero, the routes build the other families at maximum weight two.
    routes = _build_routes(code, 0)
    families = [routes.disjoint_tables, *(tables for _, tables in routes._get_candidates(2))]
    taken = set()
    for max_weight in range(-1, code.shape[1] + 2):
        tables, _ = routes.get_route(max_weight)
        taken.add(
            next(
                index
                for index, family in enumerate(families)
                if np.array_equal(tables.pivots, family.pivots)
                and np.array_equal(tables.generators, family.generators)
            )
        )
    assert taken == set(range(len(families)))
    assert len(families) == (3 if code is TWO_FAMILIES_CODE else 2)


def test_routes_by_tails_take_the_latest_record_of_their_own_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queries far above the threshold with the family that they take go by tails.

    They take the latest record of that family.  Queries below the threshold with the family that
    they take, and queries of a search without routes by tails, do not go by tails.
    """
    families = _spy_on_families(monkeypatch)
    builds = _spy_on_tails(monkeypatch)
    routes = _build_routes(OVERLAPPING_ROUTES_CODE, _THRESHOLD, by_tails=True)
    plain_routes = _build_routes(OVERLAPPING_ROUTES_CODE, _THRESHOLD)
    disjoint = (_get_instance(families, routes.disjoint_tables), routes.disjoint_tables)
    high_routes = _build_routes(OVERLAPPING_ROUTES_CODE, _HIGH_THRESHOLD, by_tails=True)
    assert high_routes.get_route(_MIDDLE_WEIGHT)[1] is None

    # At the middle weight, the greedy packing reaches the threshold, but the family that the query
    # takes does not.
    tables, tails = routes.get_route(_MIDDLE_WEIGHT)
    examinations = routes._get_examinations(_get_instance(families, tables), tables, _MIDDLE_WEIGHT)
    assert routes._get_examinations(*disjoint, _MIDDLE_WEIGHT) >= _THRESHOLD > examinations
    assert tails is None

    # Far above the threshold, a route by tails gives a record of the family that it takes.
    for max_weight in _HEAVY_WEIGHTS:
        tables, tails = routes.get_route(max_weight)
        instance = _get_instance(families, tables)
        assert routes._get_examinations(instance, tables, max_weight) >= 4 * _THRESHOLD
        assert tails is not None
        assert next(build.instance for build in builds if build.record is tails) is instance
        assert plain_routes.get_route(max_weight)[1] is None

    # Each build gets the latest record of its family, and every route by tails gives it.
    latest: dict[int, coset_tables.CosetTails] = {}
    for build in builds:
        assert build.given is latest.get(id(build.instance))
        latest[id(build.instance)] = build.record
    assert len(latest) == 2
    for max_weight in _HEAVY_WEIGHTS:
        tables, tails = routes.get_route(max_weight)
        assert tails is latest[id(_get_instance(families, tables))]


@pytest.mark.parametrize(
    "code",
    [EVEN_ROUTES_CODE, _DEFICIENT_ROUTES_CODE, _TWO_ROWS_CODE],
    ids=["even", "deficient", "two-rows"],
)
def test_routes_by_tails_ask_for_the_tables_of_the_sets_that_queries_scan_with_two_rows(
    code: npt.NDArray[np.uint8], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routes by tails ask for tables of the sets that queries scan beyond layer one."""
    builds = _spy_on_tails(monkeypatch)
    num_rows = len(code)
    routes = _build_routes(code, 0, by_tails=True)
    for max_weight in range(2, code.shape[1] + 2):
        tables, _ = routes.get_route(max_weight)
        ranks = get_set_ranks(tables)
        steps = [(layer, j) for layer in range(num_rows + 1) for j in range(1, len(ranks) + 1)]
        expected = set()
        for table in tables.planning:
            stop_layer, stop_num_completed = next(
                (step for step in steps if table[step] > max_weight), steps[-1]
            )
            for index, rank in enumerate(ranks):
                last_layer = stop_layer if index < stop_num_completed else stop_layer - 1
                if last_layer >= max(2, num_rows - rank):
                    expected.add(index)
        assert builds[-1].sets == sorted(expected)


def test_threads_that_choose_routes_take_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Threads that choose routes take turns, so each record holds the tables of its routes.

    The threads start choosing at once, and the first build waits up to a second for another build
    to start, which only threads that do not take turns can do.
    """
    max_weights = range(-1, 9)
    turn = threading.Condition()
    building = 0
    overlaps = []
    chained = []
    latest: dict[int, tuple[affine_instance.AffineInstance, coset_tables.CosetTails]] = {}
    get_coset_tails = coset_tables.get_coset_tails

    def build_in_turn(
        instance: affine_instance.AffineInstance,
        sets: list[int],
        record: coset_tables.CosetTails | None,
        **kwargs: Any,
    ) -> coset_tables.CosetTails:
        """Build a record, and keep it as the latest record of its family."""
        nonlocal building
        with turn:
            building += 1
            overlaps.append(building > 1)
            turn.notify_all()
            previous = latest.get(id(instance))
            chained.append(record is (None if previous is None else previous[1]))
            if len(overlaps) == 1:
                # Threads that take turns start no other build, so this wait times out.
                turn.wait_for(lambda: building > 1, timeout=1)
        built = get_coset_tails(instance, sets, record, **kwargs)
        with turn:
            latest[id(instance)] = (instance, built)
            building -= 1
        return built

    monkeypatch.setattr(coset_tables, "get_coset_tails", build_in_turn)
    routes = _build_routes(EVEN_ROUTES_CODE, 0, by_tails=True)
    barrier = threading.Barrier(len(max_weights), timeout=THREAD_TIMEOUT)

    chosen: list[int] = []

    def get_route(max_weight: int) -> None:
        """Get the route of a maximum weight once every thread is ready to get its own."""
        barrier.wait()
        routes.get_route(max_weight)
        chosen.append(max_weight)

    # Daemon threads, joined with a timeout, let a thread that never returns fail the test rather
    # than hang it.
    threads = [
        threading.Thread(target=get_route, args=(max_weight,), daemon=True)
        for max_weight in max_weights
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=THREAD_TIMEOUT)
        assert not thread.is_alive()
    assert sorted(chosen) == list(max_weights)
    assert len(overlaps) == len(max_weights) and not any(overlaps)
    assert len(latest) == 2 and all(chained)
    for max_weight in max_weights:
        tables, tails = routes.get_route(max_weight)
        [record] = [
            record
            for instance, record in latest.values()
            if np.array_equal(instance.pivots, tables.pivots)
        ]
        assert tails is record


def test_routes_pickle() -> None:
    """Routes pickle, and the copy gives the same routes."""
    routes = _build_routes(EVEN_ROUTES_CODE, 0, by_tails=True)
    assert routes.get_route(4)[1] is not None
    copy = pickle.loads(pickle.dumps(routes))  # noqa: S301 - trusted in-memory round trip
    for max_weight in (4, 5):
        (tables, tails), (copied_tables, copied_tails) = (
            routes.get_route(max_weight),
            copy.get_route(max_weight),
        )
        assert np.array_equal(copied_tables.planning, tables.planning)
        assert tails is not None and copied_tails is not None
        assert all(
            np.array_equal(getattr(copied_tails, name), getattr(tails, name))
            for name in tails._fields
        )


####################################################################################################
# helpers


class _TailsBuild(NamedTuple):
    """A call of ``coset_tables.get_coset_tails``: the record that it was given, and its result."""

    instance: affine_instance.AffineInstance
    sets: list[int]
    given: coset_tables.CosetTails | None
    record: coset_tables.CosetTails


def _build_routes(
    code: npt.NDArray[np.uint8], minimum_work: int, *, by_tails: bool = False
) -> coset_routes.CosetRoutes:
    """The routes of the cosets of a code, from its greedy packing in natural column order.

    Every column counts as the X part.  If ``by_tails``, the routes include routes by tails, with
    the completion tables of the default tuning.
    """
    return coset_routes.CosetRoutes(
        code,
        _get_disjoint_instance(code),
        _cosets.get_unit_syndrome_words(get_kernel(code)),
        code.shape[1],
        minimum_work=minimum_work,
        max_sets=16,
        attempts_per_set=2,
        by_tails=by_tails,
        block_bits=DEFAULT_TUNING.coset_completion_block_bits,
        max_table_bytes=DEFAULT_TUNING.coset_completion_max_table_bytes,
    )


def _get_disjoint_instance(code: npt.NDArray[np.uint8]) -> affine_instance.AffineInstance:
    """The coset instance of the greedy packing of a code in natural column order."""
    return affine_instance.get_coset_instance(code, random_column_orders=0, weight_factor=1)


def _spy_on_tails(monkeypatch: pytest.MonkeyPatch) -> list[_TailsBuild]:
    """Record each call of ``coset_tables.get_coset_tails``, and pass it on to that function."""
    builds: list[_TailsBuild] = []
    get_coset_tails = coset_tables.get_coset_tails

    def spy(
        instance: affine_instance.AffineInstance,
        sets: list[int],
        record: coset_tables.CosetTails | None,
        **kwargs: Any,
    ) -> coset_tables.CosetTails:
        built = get_coset_tails(instance, sets, record, **kwargs)
        builds.append(_TailsBuild(instance, sets, record, built))
        return built

    monkeypatch.setattr(coset_tables, "get_coset_tails", spy)
    return builds


def _spy_on_families(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[affine_instance.AffineInstance, coset_tables.CosetSearchTables]]:
    """Record each family whose tables ``coset_tables.get_search_tables`` builds, and pass it on."""
    families: list[tuple[affine_instance.AffineInstance, coset_tables.CosetSearchTables]] = []
    get_search_tables = coset_tables.get_search_tables

    def spy(
        instance: affine_instance.AffineInstance, *args: Any, **kwargs: Any
    ) -> coset_tables.CosetSearchTables:
        tables = get_search_tables(instance, *args, **kwargs)
        families.append((instance, tables))
        return tables

    monkeypatch.setattr(coset_tables, "get_search_tables", spy)
    return families


def _get_instance(
    families: list[tuple[affine_instance.AffineInstance, coset_tables.CosetSearchTables]],
    tables: coset_tables.CosetSearchTables,
) -> affine_instance.AffineInstance:
    """The coset instance of the family of some tables, among the recorded families."""
    (instance,) = [instance for instance, built in families if built is tables]
    return instance
