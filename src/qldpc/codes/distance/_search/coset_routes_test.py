# SPDX-License-Identifier: Apache-2.0

"""Tests for the routes of the BZ coset searches."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import pytest

from ..conftest import EVEN_ROUTES_CODE, OVERLAPPING_ROUTES_CODE, TWO_FAMILIES_CODE
from . import affine_instance, coset_routes, information_sets


class _Tables(NamedTuple):
    """Tables of a coset search with only the planning buffer that routes read, and the instance."""

    planning: npt.NDArray[np.int64]
    instance: affine_instance.AffineInstance


def _build_routes(
    code: npt.NDArray[np.uint8], minimum_work: int
) -> coset_routes.CosetRoutes[_Tables]:
    """The routes of the cosets of a code, from its greedy packing in natural column order.

    The planning buffers of the tables have a table of the bounds of the steps for each parity of
    the cosets of an even code, and one table otherwise, each with a row for a stopping step.
    """
    parities = (0, 1) if not np.any(code.sum(axis=1) % 2) else (None,)

    def build_tables(instance: affine_instance.AffineInstance) -> _Tables:
        tables = [instance.get_weight_bounds(parity) for parity in parities]
        planning = np.stack([np.vstack([bounds, 0 * bounds[:1]]) for bounds in tables])
        return _Tables(planning, instance)

    instance = affine_instance.get_coset_instance(code, random_column_orders=0, weight_factor=1)
    return coset_routes.CosetRoutes(
        code, instance, build_tables, minimum_work=minimum_work, max_sets=16, attempts_per_set=2
    )


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
    disjoint = _build_routes(code, 0).disjoint_tables
    examinations = 1 + max(
        _get_work(_Tables(table[None], disjoint.instance), max_weight)
        for table in disjoint.planning
    )
    for minimum_work in (examinations, examinations + 1):
        routes = _build_routes(code, minimum_work)
        assert (routes.get_tables(max_weight) is routes.disjoint_tables) == (
            minimum_work > examinations
        )
        for lighter in range(-1, max_weight):
            assert routes.get_tables(lighter) is routes.disjoint_tables


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
    routes = _build_routes(code, 0)
    families = [routes.disjoint_tables] + [tables for _, tables in routes._get_candidates(2)]
    coverages = [tables.instance.certificate.max_coverage for tables in families]
    assert coverages == sorted(coverages)
    taken = set()
    for max_weight in range(-1, code.shape[1] + 2):
        works = [_get_work(tables, max_weight) for tables in families]
        index = works.index(min(works))
        tables = routes.get_tables(max_weight)
        assert np.array_equal(tables.instance.pivots, families[index].instance.pivots)
        assert tables.instance.certificate == families[index].instance.certificate
        taken.add(index)
    assert taken == set(range(len(families)))
    assert len(families) == (3 if code is TWO_FAMILIES_CODE else 2)
