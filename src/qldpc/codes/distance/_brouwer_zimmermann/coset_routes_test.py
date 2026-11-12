# SPDX-License-Identifier: Apache-2.0

"""Tests for the routes of the BZ coset searches."""

from __future__ import annotations

import concurrent.futures
import pickle
import sys
import threading
import time
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import get_kernel

from .. import _cosets
from .._information_sets import certificates
from .._tuning import DEFAULT_TUNING
from ..conftest import EVEN_ROUTES_CODE, OVERLAPPING_ROUTES_CODE, TWO_FAMILIES_CODE
from . import affine_instance, coset_routes, coset_tables

# A code of five rows whose greedy packing leaves a partial set of rank two.
_DEFICIENT_ROUTES_CODE = np.hstack(
    [np.eye(5, dtype=np.uint8), np.array([[1, 1], [1, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)]
)

# A code of two rows, whose only layer of two rows is its last, so its sets need tables only at
# maximum weights whose queries reach it.
_TWO_ROWS_CODE = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)


class _TailsBuild(NamedTuple):
    """A call of ``coset_tables.get_coset_tails`` that builds the record of the tails of a family.

    It holds the coset instance of the family, the sets of the family that need completion tables,
    the record of the family that the call was given, and the record that it returned.
    """

    instance: affine_instance.AffineInstance
    sets: list[int]
    given: coset_tables.CosetTails | None
    record: coset_tables.CosetTails


def _build_routes(
    code: npt.NDArray[np.uint8], minimum_work: int, *, by_tails: bool = False
) -> coset_routes.CosetRoutes:
    """The routes of the cosets of a code, from its greedy packing in natural column order.

    Their tables hold the words with unit syndromes for the checks of the code, and every column
    counts as the X part.  So the planning buffers of an even code have a table of the bounds of the
    steps for each parity of its cosets, and those of other codes one table (see
    ``coset_tables.get_search_tables``).  The routes include routes by tails if ``by_tails`` is
    True, with completion tables under the default tuning.
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
        max_table_bytes=DEFAULT_TUNING.completion_bound_max_table_bytes,
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
    """The ranks of the sets of a family: their numbers of pivots, which are padded with -1."""
    return np.count_nonzero(tables.pivots >= 0, axis=1)


def _get_works(tables: coset_tables.CosetSearchTables, max_weight: int) -> list[int]:
    """The work of the plan of the steps of each table, without the sets that a query skips."""
    dimension = len(tables.planning[0]) - 2
    binomials = certificates.get_capped_binomials(dimension)
    ranks = _get_ranks(tables)
    return [
        certificates.plan_steps(table[:-1], max_weight, 0, dimension, ranks, binomials, True)[2]
        for table in tables.planning
    ]


def _get_examinations(tables: coset_tables.CosetSearchTables, max_weight: int) -> int:
    """The predicted examinations of a query: one more than the largest work of a table."""
    return 1 + max(_get_works(tables, max_weight))


@pytest.mark.parametrize(
    "code, max_weight",
    [(EVEN_ROUTES_CODE, 2), (OVERLAPPING_ROUTES_CODE, 2)],
    ids=["even", "overlapping"],
)
def test_the_gate_takes_other_families_from_its_threshold(
    code: npt.NDArray[np.uint8], max_weight: int
) -> None:
    """Queries take another family only once their predicted examinations reach the threshold.

    The predicted examinations of a query are the largest work of the disjoint family at its maximum
    weight over the tables of the search, one for each parity of the cosets of the even code, plus
    one for the word of a syndrome.  The work of a table leaves out the sets that a query skips,
    such as the partial set of the overlapping code at maximum weight two.  At the maximum weight of
    each case, a query does less work with another family.  It takes that family under a threshold
    at its predicted examinations, but not under a threshold just above them.  Every lighter query,
    which does less work, keeps the disjoint family.
    """
    disjoint_tables = _build_routes(code, 0).disjoint_tables
    assert len(disjoint_tables.planning) == (2 if code is EVEN_ROUTES_CODE else 1)
    examinations = _get_examinations(disjoint_tables, max_weight)
    for minimum_work in (examinations, examinations + 1):
        routes = _build_routes(code, minimum_work)
        assert (routes.get_route(max_weight)[0] is routes.disjoint_tables) == (
            minimum_work > examinations
        )
        for lighter in range(-1, max_weight):
            assert routes.get_route(lighter)[0] is routes.disjoint_tables


@pytest.mark.parametrize(
    "code", [EVEN_ROUTES_CODE, TWO_FAMILIES_CODE], ids=["even", "two-families"]
)
def test_routes_take_the_family_of_least_work(code: npt.NDArray[np.uint8]) -> None:
    """With the threshold at zero, each query takes the family of least work, the earliest on ties.

    With the threshold at zero, the routes build the other families at maximum weight two, so the
    test builds them there too, with ``_get_candidates``.  The work of a family sums the work of its
    tables.  At each maximum weight, the route is the family of least work among the disjoint family
    and the others, and the first of the disjoint, the maximum disjoint, and the overlapping family
    among those of equal work.  The tables of a route show its family by their generators and
    pivots.  ``_get_candidates`` returns the maximum disjoint family before the overlapping one, so
    the coverages of the families never decrease.  The two-families code builds both other families,
    and queries take each of them, and the disjoint family, at some maximum weights.  The even code
    has a table for each parity of its cosets.
    """
    routes = _build_routes(code, 0)
    families = [(_get_disjoint_instance(code), routes.disjoint_tables), *routes._get_candidates(2)]
    coverages = [instance.certificate.max_coverage for instance, _ in families]
    assert coverages == sorted(coverages)
    taken = set()
    for max_weight in range(-1, code.shape[1] + 2):
        works = [sum(_get_works(tables, max_weight)) for _, tables in families]
        index = works.index(min(works))
        tables, _ = routes.get_route(max_weight)
        assert np.array_equal(tables.pivots, families[index][1].pivots)
        assert np.array_equal(tables.generators, families[index][1].generators)
        taken.add(index)
    assert taken == set(range(len(families)))
    assert len(families) == (3 if code is TWO_FAMILIES_CODE else 2)


@pytest.mark.parametrize(
    "code", [EVEN_ROUTES_CODE, TWO_FAMILIES_CODE], ids=["even", "two-families"]
)
def test_routes_by_tails_reach_the_threshold_with_their_own_family(
    code: npt.NDArray[np.uint8], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A route is by tails exactly if its predicted examinations reach the threshold.

    The predicted examinations of a route are one more than the largest work of a table of its
    family.  Under thresholds at and just above the predicted examinations of each route at the
    threshold zero, the route of each maximum weight gives the record of the tails of its family
    exactly when they reach the threshold.  The routes build the record from the coset instance of
    the family, with ``coset_tables.get_coset_tails``, whose calls the test records.  They build it
    once for each family, and pass it to later builds for the family, which add completion tables.
    Once the routes of every maximum weight are chosen, every route by tails of a family gives the
    latest record of the family, so that no route keeps the tables of an earlier one.  Routes built
    with ``by_tails`` False give no records.  The two-families code has routes by tails on both of
    its other families, so each family must get its own record.
    """
    examinations = {
        _get_examinations(_build_routes(code, 0).get_route(max_weight)[0], max_weight)
        for max_weight in range(-1, code.shape[1] + 2)
    }
    builds = _spy_on_tails(monkeypatch)
    kinds = set()
    for minimum_work in sorted(examinations | {value + 1 for value in examinations}):
        builds.clear()
        routes = _build_routes(code, minimum_work, by_tails=True)
        plain_routes = _build_routes(code, minimum_work)
        for max_weight in range(-1, code.shape[1] + 2):
            tables, tails = routes.get_route(max_weight)
            by_tails = _get_examinations(tables, max_weight) >= minimum_work
            assert (tails is not None) == by_tails
            if tails is not None:
                instance = next(build.instance for build in builds if build.record is tails)
                assert np.array_equal(instance.pivots, tables.pivots)
            assert plain_routes.get_route(max_weight)[1] is None
            kinds.add(by_tails)

        # Each build gets the latest record of its family, which every route by tails gives.
        latest: dict[int, coset_tables.CosetTails] = {}
        for build in builds:
            assert build.given is latest.get(id(build.instance))
            latest[id(build.instance)] = build.record
        for max_weight in range(-1, code.shape[1] + 2):
            tables, tails = routes.get_route(max_weight)
            records = [
                build.record
                for build in builds
                if np.array_equal(build.instance.pivots, tables.pivots)
            ]
            assert tails is None or tails is records[-1]
    assert kinds == {False, True}


@pytest.mark.parametrize(
    "code",
    [EVEN_ROUTES_CODE, _DEFICIENT_ROUTES_CODE, _TWO_ROWS_CODE],
    ids=["even", "deficient", "two-rows"],
)
def test_routes_by_tails_ask_for_the_tables_of_the_sets_that_queries_scan_with_two_rows(
    code: npt.NDArray[np.uint8], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A route by tails asks for tables of the sets that queries scan in layers of two or more rows.

    A query stops at the first step whose bound in its table exceeds its maximum weight, found here
    by walking the steps in order, or at the last step.  Each set has then completed its last layer,
    and the query skips the sets whose rank deficiency exceeds that layer.  At the threshold zero,
    every route of a query that searches is by tails, and asks ``coset_tables.get_coset_tails`` for
    the tables of the sets that a query of some parity scans in a layer of at least two rows.  The
    even code has a table for each parity.  Some queries of the deficient code skip its partial set
    of rank deficiency three after it has completed layer two.  The sets of the code of two rows
    need tables only at the maximum weights whose queries scan its last layer.
    """
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
    """Threads that choose routes at once take turns, so each record holds the tables of its routes.

    At the threshold zero, every route of the even code is by tails, and the routes of maximum
    weights -1 to 8 share two families, whose records gain the tables of each route.  A thread for
    each maximum weight chooses its route, and the threads start at once and switch often.  The test
    wraps ``coset_tables.get_coset_tails`` to sleep before it builds, so that builds that ran at
    once would overlap, and to keep the record that each build of a family returns.  No two builds
    overlap, each build of a family is given the record that the previous build of the family
    returned, and every route gives the latest record of its family.
    """
    max_weights = range(-1, 9)
    counter_lock = threading.Lock()
    building = 0
    overlaps = []
    chained = []
    latest: dict[int, tuple[affine_instance.AffineInstance, coset_tables.CosetTails]] = {}
    get_coset_tails = coset_tables.get_coset_tails

    def build_slowly(
        instance: affine_instance.AffineInstance,
        sets: list[int],
        record: coset_tables.CosetTails | None,
        **kwargs: Any,
    ) -> coset_tables.CosetTails:
        """Build a record after a sleep, and keep it as the latest record of its family."""
        nonlocal building
        with counter_lock:
            building += 1
            overlaps.append(building > 1)
            previous = latest.get(id(instance))
            chained.append(record is (None if previous is None else previous[1]))
        time.sleep(0.02)
        built = get_coset_tails(instance, sets, record, **kwargs)
        with counter_lock:
            latest[id(instance)] = (instance, built)
            building -= 1
        return built

    monkeypatch.setattr(coset_tables, "get_coset_tails", build_slowly)
    routes = _build_routes(EVEN_ROUTES_CODE, 0, by_tails=True)
    barrier = threading.Barrier(len(max_weights))

    def get_route(
        max_weight: int,
    ) -> tuple[coset_tables.CosetSearchTables, coset_tables.CosetTails | None]:
        barrier.wait()
        return routes.get_route(max_weight)

    switch_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads often, so that an unguarded race is likely
    try:
        with concurrent.futures.ThreadPoolExecutor(len(max_weights)) as executor:
            list(executor.map(get_route, max_weights))
    finally:
        sys.setswitchinterval(switch_interval)
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
    """Routes pickle, since they hold no lock of their own, and the copy gives the same routes.

    At the threshold zero, the routes of the even code at maximum weights four and five are by
    tails.  The route of four is chosen before the routes are pickled, and that of five after.
    """
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
