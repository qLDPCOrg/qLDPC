# SPDX-License-Identifier: Apache-2.0

"""Routes of the BZ coset searches: the family, tails, and tables that answer a maximum weight.

``CosetRoutes`` chooses the route of each maximum weight of a BZ coset search of
``CosetLeaderSearch`` once: its family of information sets, and whether its queries weigh words by
their tails and skip completions by completion tables.  Every route answers within the contract of
``MinWeightBound``, and depends only on the code, the maximum weight, and the tuning.  Section 4.4
of ``brouwer_zimmermann.md`` gives the rules of the routes and their reasons, Section 4.1 says how
``CosetRoutes`` keeps tails and tables, and Sections 3.1, 3.9, and 5 say why every route answers
exactly and reproducibly.
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
# can interleave despite the global interpreter lock, and the route of a family by tails builds the
# record of the family from its latest record.  Two threads building from one record at once could
# each store a record without the tables of the other's route.  A search chooses each route once, so
# threads rarely wait for one shared lock; a lock per search would keep the search from pickling.
_ROUTE_LOCK = threading.Lock()

####################################################################################################
# routes


class CosetRoutes:
    """The tables, and records of tails, with which a BZ coset search answers each maximum weight.

    Args:
        basis: The independent embedded generators of the code of the search.  The routes keep a
            copy, from which they build the other families, so later changes to the array of the
            caller do not reach them.
        disjoint: The coset instance of the disjoint information sets of ``basis``.
        inverse: The embedded words whose syndromes have one bit set, with every column of the
            embedded words.  The tables and records of tails of every family hold them (see
            ``coset_tables.get_search_tables`` and ``coset_tables.get_coset_tails``).
        num_x_columns: The number of the first columns of the embedded words that count as their X
            part (see ``coset_tables.get_search_tables``).
        minimum_work: The field ``coset_route_minimum_work`` of the tuning.  A query whose
            predicted examinations with the disjoint family are below it keeps that family.  A route
            is a route by tails if the predicted examinations with its own family reach it.
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
        self._inverse = inverse
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

        The work of a table is that of the plan of its steps from layer zero for the limit
        ``max_weight``, without the unused sets that a query skips (see
        ``certificates.plan_steps``).  It counts the words that a query without a light enough word
        weighs, or skips by completion tables, after the word of its syndrome (Section 3.9 of
        ``brouwer_zimmermann.md``).
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

        A route by tails gives the position of its family among the disjoint family and the others,
        and the record of that family gains the tables of the sets that the route needs.
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
        """The predicted examinations of a query with a family, if it finds no light enough word.

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
        # The minimum work of the policy is zero, since the gate replaces it, so the stage never
        # skips for low work; it may still find no family.
        overlapping = basis_families.get_overlapping_family(
            self._basis,
            np.empty((len(self._basis), 0), dtype=np.uint8),
            maximum if len(maximum.ranks) else disjoint,
            weight_factor * target_weight + 1,
            basis_families.build_overlapping_family_policy(
                max_sets=self._max_sets,
                minimum_work=0,
                attempts_per_set=self._attempts_per_set,
                dimension=len(self._basis),
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
