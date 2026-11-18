# SPDX-License-Identifier: Apache-2.0

"""Tests for the routes of the BZ coset searches."""

from __future__ import annotations

import concurrent.futures
import pickle
import threading
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import get_kernel

from .. import _cosets
from .._tuning import DEFAULT_TUNING, MAX_VALUE
from ..conftest import (
    EVEN_ROUTES_CODE,
    OVERLAPPING_ROUTES_CODE,
    THREAD_TIMEOUT,
    TWO_FAMILIES_CODE,
)
from . import affine_instance, coset_routes, coset_tables

# A code of five rows whose greedy packing leaves a partial set of rank two.
_DEFICIENT_ROUTES_CODE = np.hstack(
    [np.eye(5, dtype=np.uint8), np.array([[1, 1], [1, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)]
)

# A code of two rows, so that its only layer of two rows is the last, and its sets need tables only
# at the maximum weights whose queries reach that layer.
_TWO_ROWS_CODE = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)


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


def _get_ranks(tables: coset_tables.CosetSearchTables) -> npt.NDArray[np.int64]:
    """The ranks of the sets of a family: their numbers of pivots that are not padding."""
    return np.count_nonzero(tables.pivots >= 0, axis=1)


@pytest.mark.parametrize(
    "code", [EVEN_ROUTES_CODE, OVERLAPPING_ROUTES_CODE], ids=["even", "overlapping"]
)
def test_the_gate_takes_other_families_only_far_above_its_threshold(
    code: npt.NDArray[np.uint8],
) -> None:
    """Far above the threshold, costly queries take another family, and far below it, none do."""
    routes = _build_routes(code, 0)
    assert len(routes.disjoint_tables.planning) == (2 if code is EVEN_ROUTES_CODE else 1)
    assert routes.get_route(2)[0] is not routes.disjoint_tables
    gated_routes = _build_routes(code, MAX_VALUE)
    for max_weight in range(-1, code.shape[1] + 2):
        assert gated_routes.get_route(max_weight)[0] is gated_routes.disjoint_tables


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


@pytest.mark.parametrize(
    "code", [EVEN_ROUTES_CODE, TWO_FAMILIES_CODE], ids=["even", "two-families"]
)
def test_routes_by_tails_take_the_latest_record_of_their_own_family(
    code: npt.NDArray[np.uint8], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Far above the threshold, every route is by tails, with the latest record of its family.

    Far below the threshold, and in a search without routes by tails, no route is.
    """
    max_weights = range(-1, code.shape[1] + 2)
    builds = _spy_on_tails(monkeypatch)
    routes = _build_routes(code, 0, by_tails=True)
    plain_routes = _build_routes(code, 0)
    for max_weight in max_weights:
        tables, tails = routes.get_route(max_weight)
        assert tails is not None
        instance = next(build.instance for build in builds if build.record is tails)
        assert np.array_equal(instance.pivots, tables.pivots)
        assert plain_routes.get_route(max_weight)[1] is None

    # Each build gets the latest record of its family, and every route by tails gives it.
    latest: dict[int, coset_tables.CosetTails] = {}
    for build in builds:
        assert build.given is latest.get(id(build.instance))
        latest[id(build.instance)] = build.record
    for max_weight in max_weights:
        tables, tails = routes.get_route(max_weight)
        records = [
            build.record for build in builds if np.array_equal(build.instance.pivots, tables.pivots)
        ]
        assert tails is records[-1]
    gated_routes = _build_routes(code, MAX_VALUE, by_tails=True)
    assert all(gated_routes.get_route(max_weight)[1] is None for max_weight in max_weights)


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
        ranks = _get_ranks(tables)
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
    counter_lock = threading.Lock()
    building = 0
    overlaps = []
    chained = []
    latest: dict[int, tuple[affine_instance.AffineInstance, coset_tables.CosetTails]] = {}
    overlapped = threading.Event()
    get_coset_tails = coset_tables.get_coset_tails

    def build_in_turn(
        instance: affine_instance.AffineInstance,
        sets: list[int],
        record: coset_tables.CosetTails | None,
        **kwargs: Any,
    ) -> coset_tables.CosetTails:
        """Build a record, and keep it as the latest record of its family."""
        nonlocal building
        with counter_lock:
            building += 1
            overlaps.append(building > 1)
            if building > 1:
                overlapped.set()
            first = len(overlaps) == 1
            previous = latest.get(id(instance))
            chained.append(record is (None if previous is None else previous[1]))
        if first:
            # Threads that take turns start no other build, so this wait times out.
            overlapped.wait(timeout=1)
        built = get_coset_tails(instance, sets, record, **kwargs)
        with counter_lock:
            latest[id(instance)] = (instance, built)
            building -= 1
        return built

    monkeypatch.setattr(coset_tables, "get_coset_tails", build_in_turn)
    routes = _build_routes(EVEN_ROUTES_CODE, 0, by_tails=True)
    barrier = threading.Barrier(len(max_weights), timeout=THREAD_TIMEOUT)

    def get_route(
        max_weight: int,
    ) -> tuple[coset_tables.CosetSearchTables, coset_tables.CosetTails | None]:
        """Get the route of a maximum weight once every thread is ready to get its own."""
        barrier.wait()
        return routes.get_route(max_weight)

    # Shut the pool down without waiting, so that a thread that never returns fails the test rather
    # than hangs it.
    executor = concurrent.futures.ThreadPoolExecutor(len(max_weights))
    try:
        futures = [executor.submit(get_route, max_weight) for max_weight in max_weights]
        for future in futures:
            future.result(timeout=60)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
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
