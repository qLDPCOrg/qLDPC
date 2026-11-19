# SPDX-License-Identifier: Apache-2.0

"""Routes of BZ coset searches: the family, tails, and tables for the queries at a maximum weight.

For a BZ coset search of ``CosetLeaderSearch``, ``CosetRoutes`` chooses the route of each maximum
weight once: its family of information sets, and whether its queries weigh words by their tails and
skip completions by completion tables.  Every route answers within the contract of
``MinWeightBound``, and depends only on the code, the maximum weight, and the tuning.  Sections
cited in this module are those of ``brouwer_zimmermann.md`` and ``brouwer_zimmermann_code.md``:
Section 4.4 gives the rules of the routes and their reasons, Section 4.1 says how ``CosetRoutes``
keeps tails and tables, and Sections 3.1, 3.9, and 5 say why every route answers correctly and
reproducibly.  Threads can ask for routes at once, and read the tables of a route while another
thread chooses one, so ``CosetRoutes`` chooses routes under ``_ROUTE_LOCK`` and puts new tables in a
new record of tails, as Section 5 requires under "Keep coset queries reproducible".
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

# Threads that choose routes take turns, since two threads that built records of tails from one
# record at once could each store a record without the tables of the other's route.  A search
# chooses each route once, so one shared lock rarely makes threads wait, and unlike a lock per
# search, it lets searches pickle.
_ROUTE_LOCK = threading.Lock()

####################################################################################################
# routes


class CosetRoutes:
    """The tables, and records of tails, with which a BZ coset search answers each maximum weight.

    Args:
        basis: The independent embedded generators of the code of the search, which the routes
            copy.
        disjoint: The coset instance of the disjoint information sets of ``basis``.
        inverse: The embedded words whose syndromes have one bit set, with every column, which the
            tables and records of tails of every family hold.
        num_x_columns: The number of leading columns of an embedded word that form its X part.
        minimum_work: The field ``coset_route_minimum_work`` of the tuning (Section 4.4).
        max_sets: The field ``max_information_sets`` of the tuning.
        attempts_per_set: The field ``overlapping_attempts_per_set`` of the tuning.
        by_tails: Whether the search has routes by tails, which only searches that weigh Hamming
            weights have.
        block_bits: The field ``coset_completion_block_bits`` of the tuning.
        max_table_bytes: The field ``coset_completion_max_table_bytes`` of the tuning.

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
        # The other families, with their tables, once the predicted examinations of a query with the
        # disjoint family reach the threshold.
        self._candidates: list[_Family] | None = None
        # The latest record of the tails of each family of a route by tails, by its position.
        self._tails: dict[int, CosetTails] = {}

    def get_route(self, max_weight: int) -> tuple[CosetSearchTables, CosetTails | None]:
        """The tables of the route of a maximum weight, and its record of tails if it is by tails.

        The routes keep the tables and the family of each route for later queries.  A route by tails
        gives the latest record of its family, which holds the tables of the sets that the route
        needs, and maybe those of other routes.
        """
        route = self._routes_by_weight.get(max_weight)
        if route is None:
            # Threads take turns to choose routes (see ``_ROUTE_LOCK``).
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

        It is the work of ``certificates.plan_steps`` from layer zero, without unused sets: the
        words that a query of a coset without a light enough word weighs or skips after the word of
        its syndrome (Section 3.6).
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

        # Build the other families once, at the least weight from two at which the predicted
        # examinations reach the threshold (Section 4.4).  They reach it at ``max_weight`` and never
        # decrease with the weight, so this weight is at most ``max(2, max_weight)``.
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

        A set needs tables if, at the stopping step of some table of the planning buffer, it is not
        an unused set, which a query skips (see ``certificates.plan_steps``), and the last layer
        that a query scans in it has at least two rows, and no more than the set has.
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

        The overlapping family starts from the maximum disjoint family, or from the disjoint one if
        the maximum disjoint family is not kept (Section 4.4).  Returns those of the two families
        that have sets, each with its tables.
        """
        weight_factor = self._disjoint.weight_factor
        disjoint = _get_basis_family(self._disjoint, self._basis.shape[1])
        maximum = basis_families.get_maximum_disjoint_family(self._basis, disjoint)
        # ``get_overlapping_family`` gets a minimum work of zero, since the route threshold gates it
        # instead (Section 4.4).
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

    The routes rebuild it from the coset instance only for the stages that build other families.
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
