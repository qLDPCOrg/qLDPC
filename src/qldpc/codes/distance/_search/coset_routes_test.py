# SPDX-License-Identifier: Apache-2.0

"""Tests for the routes of the BZ coset searches."""

from __future__ import annotations

import concurrent.futures
import functools
import pickle
import sys
import threading
import time
from collections.abc import Callable
from typing import NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt
import pytest

from ..conftest import EVEN_ROUTES_CODE, OVERLAPPING_ROUTES_CODE, TWO_FAMILIES_CODE
from . import affine_instance, coset_routes, information_sets


class _Tables(NamedTuple):
    """Tables of a coset search with only the planning buffer that routes read, and the instance."""

    planning: npt.NDArray[np.int64]
    instance: affine_instance.AffineInstance


# A code of five rows whose greedy packing leaves a partial set of rank two.
_DEFICIENT_ROUTES_CODE = np.hstack(
    [np.eye(5, dtype=np.uint8), np.array([[1, 1], [1, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)]
)

# A code of two rows, whose only layer of two rows is its last, so its sets need tables only at
# maximum weights whose queries reach it.
_TWO_ROWS_CODE = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)


# The record of the tails of a family that a test builds.
_Record = TypeVar("_Record")


def _build_routes(
    code: npt.NDArray[np.uint8],
    minimum_work: int,
    build_tails: Callable[[affine_instance.AffineInstance, list[int], _Record | None], _Record]
    | None = None,
) -> coset_routes.CosetRoutes[_Tables, _Record]:
    """The routes of the cosets of a code, from its greedy packing in natural column order.

    The planning buffers of the tables have a table of the bounds of the steps for each parity of
    the cosets of an even code, and one table otherwise, each with a row for a stopping step.  The
    routes build records of tails with ``build_tails``, if it is given, from the coset instance of a
    family, the sets of the family that need completion tables, and the record of the family, if
    any.
    """
    parities = (0, 1) if not np.any(code.sum(axis=1) % 2) else (None,)
    instance = affine_instance.get_coset_instance(code, random_column_orders=0, weight_factor=1)
    return coset_routes.CosetRoutes(
        code,
        instance,
        functools.partial(_build_tables, parities),
        build_tails,
        minimum_work=minimum_work,
        max_sets=16,
        attempts_per_set=2,
    )


def _build_tables(
    parities: tuple[int | None, ...], instance: affine_instance.AffineInstance
) -> _Tables:
    """The tables of a family, with a table of the bounds of its steps for each parity."""
    tables = [instance.get_weight_bounds(parity) for parity in parities]
    planning = np.stack([np.vstack([bounds, 0 * bounds[:1]]) for bounds in tables])
    return _Tables(planning, instance)


def _get_instance_record(
    instance: affine_instance.AffineInstance,
    sets: list[int],
    record: affine_instance.AffineInstance | None,
) -> affine_instance.AffineInstance:
    """Return the coset instance of a family as its record of tails."""
    return instance


def _get_work(tables: _Tables, max_weight: int) -> int:
    """The work of the plans of the steps of the tables, without the sets that a query skips."""
    dimension = len(tables.planning[0]) - 2
    binomials = information_sets.get_capped_binomials(dimension)
    return sum(
        information_sets.plan_steps(
            table[:-1], max_weight, 0, dimension, tables.instance.ranks, binomials, True
        )[2]
        for table in tables.planning
    )


def _get_examinations(tables: _Tables, max_weight: int) -> int:
    """The predicted examinations of a query: one more than the largest work of a table."""
    return 1 + max(
        _get_work(_Tables(table[None], tables.instance), max_weight) for table in tables.planning
    )


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
    such as the partial set of the random code at maximum weight two.  At the maximum weight of each
    case, a query does less work with another family.  It takes that family under a threshold at its
    predicted examinations, but not under a threshold just above them.  Every lighter query, which
    does less work, keeps the disjoint family.
    """
    examinations = _get_examinations(_build_routes(code, 0).disjoint_tables, max_weight)
    for minimum_work in (examinations, examinations + 1):
        routes: coset_routes.CosetRoutes[_Tables, None] = _build_routes(code, minimum_work)
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
    among those of equal work.  ``_get_candidates`` returns the maximum disjoint family before the
    overlapping one, so the coverages of the families never decrease.  The random code builds both
    other families, and queries take each of them, and the disjoint family, at some maximum weights.
    The even code has a table for each parity of its cosets.
    """
    routes: coset_routes.CosetRoutes[_Tables, None] = _build_routes(code, 0)
    families = [routes.disjoint_tables] + [tables for _, tables in routes._get_candidates(2)]
    coverages = [tables.instance.certificate.max_coverage for tables in families]
    assert coverages == sorted(coverages)
    taken = set()
    for max_weight in range(-1, code.shape[1] + 2):
        works = [_get_work(tables, max_weight) for tables in families]
        index = works.index(min(works))
        tables, _ = routes.get_route(max_weight)
        assert np.array_equal(tables.instance.pivots, families[index].instance.pivots)
        assert tables.instance.certificate == families[index].instance.certificate
        taken.add(index)
    assert taken == set(range(len(families)))
    assert len(families) == (3 if code is TWO_FAMILIES_CODE else 2)


@pytest.mark.parametrize(
    "code", [EVEN_ROUTES_CODE, TWO_FAMILIES_CODE], ids=["even", "two-families"]
)
def test_routes_by_tails_reach_the_threshold_with_their_own_family(
    code: npt.NDArray[np.uint8],
) -> None:
    """A route is by tails exactly if its predicted examinations reach the threshold.

    The predicted examinations of a route are one more than the largest work of a table of its
    family.  Under thresholds at and just above the predicted examinations of each route at the
    threshold zero, the route of each maximum weight gives the record of the tails of its family
    exactly when they reach the threshold.  The routes build the record from the coset instance of
    the family.  They build it once for each family, and pass it to later builds for the family,
    which add completion tables.  Once the routes of every maximum weight are chosen, every route by
    tails of a family gives the latest record of the family, so that no route keeps the tables of an
    earlier one.  A search that gives no builder has no routes by tails.  The random code has routes
    by tails on both of its other families, so each family must get its own record.
    """
    examinations = {
        _get_examinations(_build_routes(code, 0).get_route(max_weight)[0], max_weight)
        for max_weight in range(-1, code.shape[1] + 2)
    }
    records: dict[int, tuple[affine_instance.AffineInstance, int]] = {}

    def build_tails(
        instance: affine_instance.AffineInstance,
        sets: list[int],
        record: tuple[affine_instance.AffineInstance, int] | None,
    ) -> tuple[affine_instance.AffineInstance, int]:
        """Return the coset instance of a family and the number of its builds as its record."""
        assert record is records.get(id(instance))
        records[id(instance)] = (instance, 1 if record is None else records[id(instance)][1] + 1)
        return records[id(instance)]

    kinds = set()
    for minimum_work in sorted(examinations | {value + 1 for value in examinations}):
        records.clear()
        routes = _build_routes(code, minimum_work, build_tails)
        plain_routes: coset_routes.CosetRoutes[_Tables, None] = _build_routes(code, minimum_work)
        for max_weight in range(-1, code.shape[1] + 2):
            tables, tails = routes.get_route(max_weight)
            by_tails = _get_examinations(tables, max_weight) >= minimum_work
            assert (tails is not None) == by_tails
            if tails is not None:
                assert tails[0] is tables.instance
            assert plain_routes.get_route(max_weight)[1] is None
            kinds.add(by_tails)
        for max_weight in range(-1, code.shape[1] + 2):
            tables, tails = routes.get_route(max_weight)
            assert tails is None or tails is records[id(tables.instance)]
    assert kinds == {False, True}


@pytest.mark.parametrize(
    "code",
    [EVEN_ROUTES_CODE, _DEFICIENT_ROUTES_CODE, _TWO_ROWS_CODE],
    ids=["even", "deficient", "two-rows"],
)
def test_routes_by_tails_ask_for_the_tables_of_the_sets_that_queries_scan_with_two_rows(
    code: npt.NDArray[np.uint8],
) -> None:
    """A route by tails asks for tables of the sets that queries scan in layers of two or more rows.

    A query stops at the first step whose bound in its table exceeds its maximum weight, found here
    by walking the steps in order, or at the last step.  Each set has then completed its last layer,
    and the query skips the sets whose rank deficiency exceeds that layer.  At the threshold zero,
    every route of a query that searches is by tails, and asks for the tables of the sets that a
    query of some parity scans in a layer of at least two rows.  The even code has a table for each
    parity.  Some queries of the deficient code skip its partial set of rank deficiency three after
    it has completed layer two.  The sets of the code of two rows need tables only at the maximum
    weights whose queries scan its last layer.
    """
    asked: list[list[int]] = []

    def build_tails(
        instance: affine_instance.AffineInstance,
        sets: list[int],
        record: affine_instance.AffineInstance | None,
    ) -> affine_instance.AffineInstance:
        """Record the sets that need tables, and return the coset instance as the record."""
        asked.append(sets)
        return instance

    num_rows = len(code)
    routes = _build_routes(code, 0, build_tails)
    for max_weight in range(2, code.shape[1] + 2):
        tables, _ = routes.get_route(max_weight)
        ranks = tables.instance.ranks
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
        assert asked[-1] == sorted(expected)


def test_threads_that_choose_routes_take_turns() -> None:
    """Threads that choose routes at once take turns, so each record holds the tables of its routes.

    At the threshold zero, every route of the even code is by tails, and the routes of maximum
    weights -1 to 8 share two families, whose records gain the tables of each route.  A thread for
    each maximum weight chooses its route, and the threads start at once and switch often.  The
    builder of records sleeps, so that builds that ran at once would overlap.  The record that it
    returns holds the sets that its build asked for, and those of the record that it was given.  No
    two builds overlap, and the record of every route holds every set that the builds of its family
    asked for.
    """
    max_weights = range(-1, 9)
    counter_lock = threading.Lock()
    building = 0
    overlaps = []
    asked: dict[int, set[int]] = {}

    def build_tails(
        instance: affine_instance.AffineInstance,
        sets: list[int],
        record: tuple[affine_instance.AffineInstance, frozenset[int]] | None,
    ) -> tuple[affine_instance.AffineInstance, frozenset[int]]:
        """Record the sets of a build, and return them with those of the given record."""
        nonlocal building
        with counter_lock:
            building += 1
            overlaps.append(building > 1)
            asked.setdefault(id(instance), set()).update(sets)
        time.sleep(0.02)
        with counter_lock:
            building -= 1
        return instance, frozenset(sets).union(() if record is None else record[1])

    routes = _build_routes(EVEN_ROUTES_CODE, 0, build_tails)
    barrier = threading.Barrier(len(max_weights))

    def get_route(
        max_weight: int,
    ) -> tuple[_Tables, tuple[affine_instance.AffineInstance, frozenset[int]] | None]:
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
    assert len(asked) == 2 and any(asked.values())
    for max_weight in max_weights:
        _, tails = routes.get_route(max_weight)
        assert tails is not None and tails[1] == asked[id(tails[0])]


def test_routes_pickle() -> None:
    """Routes pickle, since they hold no lock of their own, and the copy gives the same routes.

    At the threshold zero, the routes of the even code at maximum weights four and five are by
    tails.  The route of four is chosen before the routes are pickled, and that of five after.
    """
    routes = _build_routes(EVEN_ROUTES_CODE, 0, _get_instance_record)
    assert routes.get_route(4)[1] is not None
    copy = pickle.loads(pickle.dumps(routes))  # noqa: S301 - trusted in-memory round trip
    for max_weight in (4, 5):
        (tables, tails), (copied_tables, copied_tails) = (
            routes.get_route(max_weight),
            copy.get_route(max_weight),
        )
        assert np.array_equal(copied_tables.planning, tables.planning)
        assert tails is not None and copied_tails is not None
        assert np.array_equal(copied_tails.pivots, tails.pivots)
