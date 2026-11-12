# SPDX-License-Identifier: Apache-2.0

"""Routes of the BZ coset searches: the family of information sets that answers a maximum weight.

A BZ coset search of ``CosetLeaderSearch`` answers whether the coset of a syndrome has a word of
weight at most ``max_weight`` by scanning the layers of a family of information sets of its code, up
to the stopping step of ``max_weight`` (see ``coset.py``).  It is built with the disjoint
information sets of a greedy packing, which are cheap to find.  A code whose length is less than
twice its dimension has only one such information set, so queries at larger maximum weights scan
many layers.  A greedy packing can also miss information sets.  The columns ``e1``, ``e2``,
``e1 + e2``, and ``e1 + e2`` hold two disjoint information sets, but a greedy packing in natural
order takes ``e1`` and ``e2``, and leaves a partial set of rank one.  Families with more information
sets stop sooner, but take milliseconds to build, which only costly queries repay.

The routes choose the family that answers each maximum weight.  A search has a table of the bounds
of its steps for each parity of its cosets, or one table for every coset.  The work of a table at
``max_weight`` is the work of the plan of its steps (see ``certificates.plan_steps``).  It is the
number of words that a query that takes the table weighs, or skips by completion tables (Section 3.9
of ``brouwer_zimmermann.md``), after the word of its syndrome if no word of its coset weighs at most
``max_weight``, and at most that number otherwise.  The predicted examinations of a query at
``max_weight`` are one more than the largest work of a table of the disjoint family: the
examinations of the costliest query that finds no light enough word, without completion tables, and
at most that number with them.  The gate compares them with its threshold, the field
``overlapping_minimum_work`` of the tuning.  Below the threshold, the route of ``max_weight`` is the
disjoint family.  Otherwise, the routes try to build two other families once, at the target weight,
the smallest maximum weight from two whose predicted examinations reach the threshold.  They keep
each that they find:

- the maximum disjoint family of ``basis_families.get_maximum_disjoint_family``, if it has more
  information sets or a larger partial set than the greedy packing; and
- a family of overlapping information sets from ``basis_families.get_overlapping_family``, if it
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
answers do not depend on which queries came before.  The routes keep copies of the generators of
the code and of the words with unit syndromes, which they read when they build the other families
and the tables and tails of the families, so later changes to the arrays of the caller do not reach
them.

The gate tests the predicted examinations with the disjoint family against its threshold, to decide
whether a query may take another family.  A route is a route by tails if those with the route's own
family, which can do less work, also reach the threshold.  The queries of a route by tails weigh the
words of its family by their tails, which hold the columns other than the pivot columns of each set
(see ``coset_tables.CosetTails``).  They also skip the completions that completion tables of the
tails show to lie beyond their maximum weight (Section 3.9 of ``brouwer_zimmermann.md``).  So the
routes also give the record of the tails and tables of the family of each route by tails, from
``coset_tables.get_coset_tails``.  A search without routes by tails, such as the search of Pauli
errors by CSS weight, gets no records.  The routes build the tails of a family once, and the tables
of a set of the family once, for the first route that needs them.  A route needs the tables of the
sets that its queries visit in layers of two or more rows, since tables bound the completions of a
prefix by two more rows (see ``completion_bounds.considers_tables``).  The gate of routes by tails
replaces the minimum work of a layer and the cost model of the tables of distance searches.  A set
gets tables only if they fit in the field ``completion_bound_max_table_bytes`` of the tuning.  Every
route by tails of a family gives the latest record of the family, which holds the tables of each set
once.  That record can hold tables that other routes of the family needed, but the queries of a
route read only the tables that it needs, so their answers do not depend on earlier queries.  A
search for minimum weights up to the maximum weight of a route takes the route too, and visits no
layer that its queries do not visit, since its stopping step comes no later than theirs (Section
3.10 of ``brouwer_zimmermann.md``).

The tails change how a query weighs words, and the tables which words it skips, but neither changes
which word of weight at most the maximum weight a query finds first, if any.  A query without such a
word can return another word, and a lower bound of the least weight beyond the maximum weight that a
word of its coset can have, within the contract of ``MinWeightBound``.  A query by tails took about
50 ns more than a query of whole words in a prototype, for the tails of its offsets and the rebuild
of its lightest word, which made batches of queries of under 100 examinations each up to about twice
as slow.  So only queries that are costly even with the family of their route weigh words by their
tails.
"""

from __future__ import annotations

import itertools
import threading

import numpy as np
import numpy.typing as npt

from qldpc._binary import unpack_bits

from .._information_sets import basis_families, certificates
from . import affine_instance, coset_tables
from .coset_tables import CosetSearchTables, CosetTails

# A family of information sets of a coset search: its coset instance and its tables.
_Family = tuple[affine_instance.AffineInstance, CosetSearchTables]

# Threads that choose routes at once take turns.  Choosing a route runs plain Python, which threads
# can interleave even with the global interpreter lock, and the route of a family by tails builds
# the record of the family from its latest record.  Two threads that built from the same record at
# once could each store a record without the tables of the other's route.  Routes are chosen
# rarely, typically in milliseconds, so one shared lock costs little.  A lock on each search would
# keep it from pickling.
_ROUTE_LOCK = threading.Lock()

####################################################################################################
# routes


class CosetRoutes:
    """The tables, and records of tails, with which a BZ coset search answers each maximum weight.

    Args:
        basis: The independent embedded generators of the code of the search, which the routes
            copy.
        disjoint: The coset instance of the disjoint information sets of ``basis``.
        inverse: The embedded words whose syndromes have one bit set, with every column of the
            embedded words, which the routes copy.  The tables and records of tails of every family
            hold them (see ``coset_tables.get_search_tables`` and ``coset_tables.get_coset_tails``).
        num_x_columns: The number of the first columns of the embedded words that count as their X
            part (see ``coset_tables.get_search_tables``).
        minimum_work: The threshold of the gate and of the routes by tails, the field
            ``overlapping_minimum_work`` of the tuning.  A query whose predicted examinations with
            the disjoint family are below it keeps that family.  A route is a route by tails if the
            predicted examinations with its own family reach it.
        max_sets: The most sets of a family of overlapping information sets, the field
            ``max_information_sets`` of the tuning.
        attempts_per_set: The field ``overlapping_attempts_per_set`` of the tuning.
        by_tails: Whether the search has routes by tails, which only searches that weigh Hamming
            weights have.
        block_bits: The number of columns of the blocks of the completion tables of the routes by
            tails, the field ``coset_completion_block_bits`` of the tuning.
        max_table_bytes: The most bytes of the completion tables of a set, the field
            ``completion_bound_max_table_bytes`` of the tuning.

    Attributes:
        disjoint_tables: The tables of the disjoint family.
    """

    def __init__(
        self,
        basis: npt.NDArray[np.uint8],
        disjoint: affine_instance.AffineInstance,
        inverse: npt.NDArray[np.uint8],
        num_x_columns: int,
        *,
        minimum_work: int,
        max_sets: int,
        attempts_per_set: int,
        by_tails: bool,
        block_bits: int,
        max_table_bytes: int,
    ) -> None:
        self._basis = basis.copy()
        self._disjoint = disjoint
        self._inverse = inverse.copy()
        self._num_x_columns = num_x_columns
        self._minimum_work = minimum_work
        self._max_sets = max_sets
        self._attempts_per_set = attempts_per_set
        self._by_tails = by_tails
        self._block_bits = block_bits
        self._max_table_bytes = max_table_bytes
        self._binomials = certificates.get_capped_binomials(len(basis))
        self.disjoint_tables = coset_tables.get_search_tables(
            disjoint, self._inverse, num_x_columns
        )
        # The tables of the route of each maximum weight, and the position of its family among the
        # disjoint family and the others if it is a route by tails.
        self._routes_by_weight: dict[int, tuple[CosetSearchTables, int | None]] = {}
        # The other families, with their tables, once a maximum weight reaches the threshold.
        self._candidates: list[_Family] | None = None
        # The latest record of the tails of each family of a route by tails, by its position.
        self._tails: dict[int, CosetTails] = {}

    def get_route(self, max_weight: int) -> tuple[CosetSearchTables, CosetTails | None]:
        """The tables of the route of a maximum weight, and its record of tails if it is by tails.

        The routes keep the tables and the family of each route for later queries.  A route by tails
        gives the latest record of its family, which holds the tables of the sets that the route
        needs, and maybe those of other routes of the family.
        """
        route = self._routes_by_weight.get(max_weight)
        if route is None:
            # Threads take turns to choose routes, so that the record of each family gains the
            # tables of the sets of every route (see ``_ROUTE_LOCK``).
            with _ROUTE_LOCK:
                route = self._routes_by_weight.get(max_weight)
                if route is None:
                    route = self._choose(max_weight)
                    self._routes_by_weight[max_weight] = route
        tables, family = route
        return tables, None if family is None else self._tails[family]

    def _get_works(
        self, instance: affine_instance.AffineInstance, tables: CosetSearchTables, max_weight: int
    ) -> list[int]:
        """The work of a family at a maximum weight, for each table of its planning buffer.

        This is the work of the plan of the steps of the table (see ``certificates.plan_steps``)
        from layer zero, for the limit ``max_weight``, without the unused sets that a query leaves
        out.  It is the number of words that a query that finds no word weighs, or skips by
        completion tables (Section 3.9 of ``brouwer_zimmermann.md``), after the word of its
        syndrome.
        """
        return [
            certificates.plan_steps(
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

    def _choose(self, max_weight: int) -> tuple[CosetSearchTables, int | None]:
        """Choose the tables of the route of a maximum weight, and its family if it is by tails.

        The record of the family of a route by tails gains the tables of the sets that the route
        needs.  A route by tails gives the position of its family among the disjoint family and the
        others.
        """
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

        # The route is by tails if its own predicted examinations reach the threshold.  Its record
        # of tails gains the tables of the sets that its queries need.
        if (
            not self._by_tails
            or self._get_examinations(instance, tables, max_weight) < self._minimum_work
        ):
            return tables, None
        sets = self._get_sets_needing_tables(instance, tables, max_weight)
        self._tails[index] = coset_tables.get_coset_tails(
            instance,
            sets,
            self._tails.get(index),
            inverse=self._inverse,
            block_bits=self._block_bits,
            max_table_bytes=self._max_table_bytes,
        )
        return tables, index

    def _get_examinations(
        self, instance: affine_instance.AffineInstance, tables: CosetSearchTables, max_weight: int
    ) -> int:
        """The predicted examinations of a query with a family that finds no word.

        They are the largest work of the family over its tables, plus one for the word of the
        syndrome.
        """
        return 1 + max(self._get_works(instance, tables, max_weight))

    def _get_sets_needing_tables(
        self, instance: affine_instance.AffineInstance, tables: CosetSearchTables, max_weight: int
    ) -> list[int]:
        """The sets of a family whose completion tables the queries at a maximum weight use.

        A query visits each set up to the last layer that the set completes by the stopping step of
        its table, unless the set is an unused set of that step (see ``certificates.plan_steps``),
        and then the query skips the set.  It uses the tables of a set in every layer of two or more
        rows, up to the number of rows of the basis (see ``completion_bounds.considers_tables``).
        So a set needs tables if its last layer is such a layer.
        """
        num_rows = len(self._basis)
        sets = set()
        for table in tables.planning:
            stop_layer, stop_num_completed = certificates.find_stopping_step(
                table[:-1], max_weight, 0, len(table) - 2
            )
            for set_index in range(len(instance.ranks)):
                last_layer = certificates.get_completed_layer(
                    set_index, stop_layer, stop_num_completed
                )
                # The check of unused sets requires a nonnegative layer, so it comes after the check
                # that the last layer has at least two rows.
                if 2 <= last_layer <= num_rows and not certificates.is_unused_set(
                    instance.pivots, set_index, stop_layer, stop_num_completed
                ):
                    sets.add(set_index)
        return sorted(sets)

    def _get_candidates(self, target_weight: int) -> list[_Family]:
        """Try to build the maximum disjoint and the overlapping families, and their tables.

        The overlapping family starts from the information sets of the maximum disjoint family.  If
        the maximum disjoint family has no larger ranks than the disjoint one, it starts from those
        of the disjoint family.  Returns each family that the stages find, with its tables.
        """
        weight_factor = self._disjoint.weight_factor
        disjoint = _get_basis_family(self._disjoint, self._basis.shape[1])
        maximum = basis_families.get_maximum_disjoint_family(self._basis, disjoint)
        # The gate replaces the minimum work of the overlapping stage, so the stage never skips for
        # low work; it may still find no family.
        overlapping = basis_families.get_overlapping_family(
            self._basis,
            np.empty((len(self._basis), 0), dtype=np.uint8),
            maximum if len(maximum.ranks) else disjoint,
            weight_factor * target_weight + 1,
            basis_families.build_overlapping_family_policy(
                self._max_sets, 0, self._attempts_per_set, len(self._basis)
            ),
        )

        # Pack each family that has sets for the coset search, and build its tables.
        candidates = []
        for family in (maximum, overlapping):
            if len(family.ranks):
                instance = affine_instance.get_coset_family_instance(
                    family, weight_factor=weight_factor
                )
                tables = coset_tables.get_search_tables(
                    instance, self._inverse, self._num_x_columns
                )
                candidates.append((instance, tables))
        return candidates


def _get_basis_family(
    instance: affine_instance.AffineInstance, num_columns: int
) -> basis_families.BasisFamily:
    """The basis family of the sets of a coset instance, without labels or a weight divisor.

    The routes keep the coset instance of the disjoint family, which every query may scan, and
    rebuild its basis family only for the stages that build other families, once for each search.
    """
    num_sets, dimension, width = instance.generators.shape
    generators = unpack_bits(instance.generators.reshape(-1, width), num_columns)
    return basis_families.BasisFamily(
        generators.reshape(num_sets, dimension, num_columns),
        np.empty((num_sets, dimension, 0), dtype=np.uint8),
        instance.pivots,
        instance.ranks,
        1,
        instance.certificate.max_coverage,
    )
