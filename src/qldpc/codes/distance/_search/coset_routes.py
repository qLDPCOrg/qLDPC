# SPDX-License-Identifier: Apache-2.0

"""Routes of the BZ coset searches: the family of information sets that answers a maximum weight.

A BZ coset search of ``CosetLeaderSearch`` answers whether the coset of a syndrome has a word of
weight at most ``max_weight`` by scanning the layers of a family of information sets of its code, up
to the stopping step of ``max_weight`` (see ``coset_leader_search.py``).  It is built with the
disjoint information sets of a greedy packing, which are cheap to find.  A code whose length is less
than twice its dimension has only one such information set, so queries at larger maximum weights
scan many layers.  A greedy packing can also miss information sets.  The columns ``e1``, ``e2``,
``e1 + e2``, and ``e1 + e2`` hold two disjoint information sets, but a greedy packing in natural
order takes ``e1`` and ``e2``, and leaves a partial set of rank one.  Families with more information
sets stop sooner, but take milliseconds to build, which only costly queries repay.

The routes choose the family that answers each maximum weight.  A search has a table of the bounds
of its steps for each parity of its cosets, or one table for every coset.  The work of a table at
``max_weight`` is the work of the plan of its steps (see ``information_sets.plan_steps``).  It is
the number of words that a query that takes the table weighs after the word of its syndrome if no
word of its coset weighs at most ``max_weight``, and at most that number otherwise.  The predicted
examinations of a query at ``max_weight`` are one more than the largest work of a table of the
disjoint family: the examinations of the costliest query that finds no light enough word.  The gate
compares them with its threshold, the field ``overlapping_minimum_work`` of the tuning.  Below the
threshold, the route of ``max_weight`` is the disjoint family.  Otherwise, the routes try to build
two other families once, at the target weight, the smallest maximum weight from two whose predicted
examinations reach the threshold.  They keep each that they find:

- the maximum disjoint family of ``information_sets.get_maximum_disjoint_family``, if it has more
  information sets or a larger partial set than the greedy packing; and
- a family of overlapping information sets from ``information_sets.get_overlapping_family``, if it
  finds one that does less work than the family that it starts from.  It is aimed at the Hamming
  weight ``η * target_weight + 1`` that a query at the target weight without a light word must
  certify, for the weight factor ``η`` of the search, and it is built with the fields
  ``max_information_sets`` and ``overlapping_attempts_per_set``, as in the distance searches.

The route of ``max_weight`` is then the family with the least work at ``max_weight``, summed over
its tables, which weighs every parity.  Among families of equal work, it is the first of the
disjoint, the maximum disjoint, and the overlapping family.  The certificate of every such family
bounds the words of each coset (see ``affine_instance.get_coset_family_instance``), so every route
answers exactly.  Routes can return different words and lower bounds, within the contract of
``MinWeightBound``.  Each route depends only on the code, ``max_weight``, and the tuning, so the
answers do not depend on which queries came before.  The routes keep a copy of the generators of
the code, which they read when they build the other families, so later changes to the array of the
caller do not reach them.

The gate tests the predicted examinations with the disjoint family against its threshold, to decide
whether a query may take another family.  A route is a route by tails if those with the route's own
family, which can do less work, also reach the threshold.  The queries of a route by tails weigh the
words of its family by their tails, which hold the columns other than the pivot columns of each set
(see ``coset_leader_search._CosetTails``).  So the routes also give the record of the tails of the
family of each route by tails, which they build once for each family, with a builder that the
search gives them.  A search that gives none, such as the search of Pauli errors by CSS weight, has
no routes by tails.  The tails change how a query weighs words, but not which words it visits, so
they change no answer.  A query by tails took about 50 ns more than a query of whole words in a
prototype, for the tails of its offsets and the rebuild of its lightest word, which made batches of
queries of under 100 examinations each up to about twice as slow.  So only queries that are costly
even with the family of their route weigh words by their tails.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Generic, Protocol, TypeVar

import numpy as np
import numpy.typing as npt

from qldpc._binary import unpack_bits

from .._tuning import SearchTuning
from . import affine_instance, information_sets


class _SearchTables(Protocol):
    """The tables of a coset search that its routes read: its planning buffer.

    The buffer holds a table of the bounds of the steps of the search for each parity of its
    cosets, or one table for every coset, each with one more row that caches a stopping step (see
    ``coset_leader_search._get_planning``).
    """

    @property
    def planning(self) -> npt.NDArray[np.int64]: ...


_Tables = TypeVar("_Tables", bound=_SearchTables)

# The record of the tails of a family of a coset search (see ``CosetRoutes``).
_Tails = TypeVar("_Tails")

####################################################################################################
# routes


class CosetRoutes(Generic[_Tables, _Tails]):
    """The tables, and records of tails, with which a BZ coset search answers each maximum weight.

    Args:
        basis: The independent embedded generators of the code of the search, which the routes
            copy.
        disjoint: The coset instance of the disjoint information sets of ``basis``.
        build_tables: Builds the tables of the search from a coset instance of ``basis``.
        build_tails: Builds the record of the tails of the family of a route by tails from its coset
            instance, or None for a search without routes by tails.
        minimum_work: The threshold of the gate and of the routes by tails, the field
            ``overlapping_minimum_work`` of the tuning.  A query whose predicted examinations with
            the disjoint family are below it keeps that family.  A route is a route by tails if the
            predicted examinations with its own family reach it.
        max_sets: The most sets of a family of overlapping information sets, the field
            ``max_information_sets`` of the tuning.
        attempts_per_set: The field ``overlapping_attempts_per_set`` of the tuning.

    Attributes:
        disjoint_tables: The tables of the disjoint family.
    """

    def __init__(
        self,
        basis: npt.NDArray[np.uint8],
        disjoint: affine_instance.AffineInstance,
        build_tables: Callable[[affine_instance.AffineInstance], _Tables],
        build_tails: Callable[[affine_instance.AffineInstance], _Tails] | None,
        *,
        minimum_work: int,
        max_sets: int,
        attempts_per_set: int,
    ) -> None:
        self._basis = basis.copy()
        self._disjoint = disjoint
        self._build_tables = build_tables
        self._build_tails = build_tails
        self._minimum_work = minimum_work
        # The tuning of the overlapping family, with the fixed seed of the distance searches.
        # ``get_overlapping_family_policy`` reads only these three fields, and the gate replaces the
        # minimum work of the stage, so the stage never skips for low work; it may still find no
        # family.
        self._overlapping_tuning = SearchTuning(
            max_information_sets=max_sets,
            overlapping_minimum_work=0,
            overlapping_attempts_per_set=attempts_per_set,
        )
        self._binomials = information_sets.get_capped_binomials(len(basis))
        self.disjoint_tables = build_tables(disjoint)
        self._routes_by_weight: dict[int, tuple[_Tables, _Tails | None]] = {}
        # The other families, with their tables, once a maximum weight reaches the threshold.
        self._candidates: list[tuple[affine_instance.AffineInstance, _Tables]] | None = None
        # The record of the tails of the family of each route by tails, by the position of the
        # family among the disjoint family and the others.
        self._tails: dict[int, _Tails] = {}

    def get_route(self, max_weight: int) -> tuple[_Tables, _Tails | None]:
        """The tables of the route of a maximum weight, and its record of tails if it is by tails.

        The routes keep both for later queries.
        """
        route = self._routes_by_weight.get(max_weight)
        if route is None:
            route = self._choose(max_weight)
            self._routes_by_weight[max_weight] = route
        return route

    def _get_works(
        self, instance: affine_instance.AffineInstance, tables: _Tables, max_weight: int
    ) -> list[int]:
        """The work of a family at a maximum weight, for each table of its planning buffer.

        This is the work of the plan of the steps of the table (see ``information_sets.plan_steps``)
        from layer zero, for the limit ``max_weight``, without the sets that a query skips.
        """
        return [
            information_sets.plan_steps(
                table[:-1],
                max_weight,
                0,
                len(table) - 2,
                instance.ranks,
                self._binomials,
                True,
            )[2]
            for table in tables.planning
        ]

    def _choose(self, max_weight: int) -> tuple[_Tables, _Tails | None]:
        """Choose the tables of the route of a maximum weight, and its record of tails, if any."""
        disjoint = (self._disjoint, self.disjoint_tables)
        if self._get_examinations(*disjoint, max_weight) < self._minimum_work:
            return self.disjoint_tables, None

        # Build the other families at the smallest weight from two at which the predicted
        # examinations reach the threshold.  The examinations never decrease with the weight, so
        # this weight is at most max_weight if max_weight is at least two.
        if self._candidates is None:
            target_weight = next(
                weight
                for weight in itertools.count(2)
                if self._get_examinations(*disjoint, weight) >= self._minimum_work
            )
            self._candidates = self._get_candidates(target_weight)

        # Take the family with the least work, and the earliest one of equal work.
        candidates = [disjoint, *self._candidates]
        works = [sum(self._get_works(*candidate, max_weight)) for candidate in candidates]
        index = works.index(min(works))
        instance, tables = candidates[index]

        # The route is by tails if its own predicted examinations reach the threshold.
        if (
            self._build_tails is None
            or self._get_examinations(instance, tables, max_weight) < self._minimum_work
        ):
            return tables, None
        if index not in self._tails:
            self._tails[index] = self._build_tails(instance)
        return tables, self._tails[index]

    def _get_examinations(
        self, instance: affine_instance.AffineInstance, tables: _Tables, max_weight: int
    ) -> int:
        """The predicted examinations of a query with a family that finds no word.

        They are the largest work of the family over its tables, plus one for the word of the
        syndrome.
        """
        return 1 + max(self._get_works(instance, tables, max_weight))

    def _get_candidates(
        self, target_weight: int
    ) -> list[tuple[affine_instance.AffineInstance, _Tables]]:
        """Try to build the maximum disjoint and the overlapping families, and their tables.

        The overlapping family starts from the information sets of the maximum disjoint family.  If
        the maximum disjoint family has no larger ranks than the disjoint one, it starts from those
        of the disjoint family.  Returns each family that the stages find, with its tables.
        """
        weight_factor = self._disjoint.weight_factor
        disjoint = _get_basis_family(self._disjoint, self._basis.shape[1])
        maximum = information_sets.get_maximum_disjoint_family(self._basis, disjoint)
        overlapping = information_sets.get_overlapping_family(
            self._basis,
            np.empty((len(self._basis), 0), dtype=np.uint8),
            maximum if len(maximum.ranks) else disjoint,
            weight_factor * target_weight + 1,
            *information_sets.get_overlapping_family_policy(
                self._overlapping_tuning, len(self._basis)
            ),
        )

        # Pack each family that has sets for the coset search, and build its tables.
        candidates = []
        for family in (maximum, overlapping):
            if len(family.ranks):
                instance = affine_instance.get_coset_family_instance(
                    family, weight_factor=weight_factor
                )
                candidates.append((instance, self._build_tables(instance)))
        return candidates


def _get_basis_family(
    instance: affine_instance.AffineInstance, num_columns: int
) -> information_sets.BasisFamily:
    """The basis family of the sets of a coset instance, without labels or a weight divisor.

    The routes keep the coset instance of the disjoint family, which every query may scan, and
    rebuild its basis family only for the stages that build other families, once for each search.
    """
    num_sets, dimension, width = instance.generators.shape
    generators = unpack_bits(instance.generators.reshape(-1, width), num_columns)
    return information_sets.BasisFamily(
        generators.reshape(num_sets, dimension, num_columns),
        np.empty((num_sets, dimension, 0), dtype=np.uint8),
        instance.pivots,
        instance.ranks,
        1,
        instance.certificate.max_coverage,
    )
