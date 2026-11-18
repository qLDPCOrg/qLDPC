# SPDX-License-Identifier: Apache-2.0

"""Exact BZ searches with disjoint, overlapping, or automorphism families of information sets.

The search follows `Algorithm 994 <https://arxiv.org/abs/1603.06757>`_ and its
`adaptation to quantum codes <https://arxiv.org/abs/2408.10743>`_.  This module prepares each
search, takes its early exits, and scans its layers in resumable chunks with ``affine_search.py``.
Sections cited in this module are those of ``brouwer_zimmermann.md``, which derives the certificates
and invariants of the search.  A costly layer runs on several threads (Section 4.5), so a change
here must keep the requirements on threads in Section 5, such as "Decide only from what a layer
leaves the same on any number of threads".
"""

from __future__ import annotations

import dataclasses
import functools
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits

from .. import _threads
from .._binary import get_eligible_rows
from .._information_sets import (
    automorphism_families,
    basis_families,
    certificates,
    completion_bounds,
)
from .._tuning import MAX_VALUE, SearchTuning, get_tuning
from . import affine_instance, affine_search

# The automorphism family of a sector: a basis family of the one set that a search scans, the
# certificate of all of its images, and the orbit of each column under the verified group.
_AutomorphismFamily = tuple[
    basis_families.BasisFamily, certificates.Certificate, npt.NDArray[np.int64]
]

# A function that returns the automorphism family of a sector for a weight divisor, or None.
_GetAutomorphismFamily = Callable[[int], _AutomorphismFamily | None]

# The fewest work items for each thread that a layer split across threads aims for, so that the last
# ranges of items, which guided self-scheduling makes small, are small enough for threads to finish
# at about the same time (Section 4.5).
_ITEMS_PER_THREAD = 32

####################################################################################################
# search entry points


def get_distance_brouwer_zimmermann(
    sectors: Sequence[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]],
    *,
    cutoff: int,
    weight_divisor: int = 1,
    upper_bound: int | None = None,
    get_disjoint_basis_families: Sequence[Callable[[], basis_families.BasisFamily]] | None = None,
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
    tuning: SearchTuning | None = None,
) -> int:
    """Compute the minimum distance across nested-code sectors by a BZ search.

    The searches of the sectors advance round-robin, and share their best upper bound, the weight of
    the lightest witness that any of them has found, but not their certificates (Section 3.3).

    Args:
        sectors: Nested-code bases and their quotient labels (see ``get_nested_code_basis``).  The
            distance of a sector is the least weight of a word of its basis with nonzero labels, or
            of any nonzero word if its labels have no columns.  Each sector must have such a word.
        cutoff: Return once the best upper bound is at most this cutoff.
        weight_divisor: A proven divisor of the weights over which the sectors minimize, to whose
            multiples the certificates round up (Section 3.2).  Default: 1.
        upper_bound: A cap on the result, if any, such as the weight of a known witness outside
            ``sectors``, or ``max_weight + 1`` under a weight limit (Section 4.5).
        get_disjoint_basis_families: If given, a function for each sector that returns a disjoint
            basis family of it, with its labels and certified with ``weight_divisor``, such as that
            of ``basis_families.get_disjoint_family``.  A search calls it only if it needs the
            family.
        get_automorphisms: If given, returns None or an array whose rows ``g`` generate a group of
            column permutations, each mapping column ``j`` to column ``g[j]``.  Each generator must
            map the space and the excluded subspace of every sector onto themselves.  The search
            of each sector calls the function at most once, so a search of two sectors may call it
            twice, and verifies the generators (Section 3.7).
        tuning: The tuning that every search uses throughout.  Default: the active tuning.

    Returns:
        The minimum of the distances of the sectors and ``upper_bound``, or, after an early exit
        at ``cutoff``, the weight of a witness or ``upper_bound``, whichever is at most
        ``cutoff``.

    Raises:
        ValueError: If there are neither sectors nor an upper bound, or if a generator from
            ``get_automorphisms`` is not an automorphism of a sector whose search uses it.
    """
    search = BrouwerZimmermannSearch(
        sectors,
        cutoff=cutoff,
        weight_divisor=weight_divisor,
        upper_bound=upper_bound,
        get_disjoint_basis_families=get_disjoint_basis_families,
        get_automorphisms=get_automorphisms,
        tuning=tuning,
    )
    while not search.finished:
        search.advance()
    return search.best


class BrouwerZimmermannSearch:
    """A BZ search of sectors, which advances their searches one layer at a time, round-robin.

    The arguments are those of ``get_distance_brouwer_zimmermann``, which advances the search until
    it finishes.

    Attributes:
        best: The shared upper bound: the least of ``upper_bound``, the weights of the lightest
            eligible rows, the results of setup, and the weights of the witnesses that the searches
            of the sectors have found as they advanced.  The search of a sector can hold a lighter
            witness from its setup, which lowers ``best`` once that search advances.  Once the
            search finishes, ``best`` is the result of ``get_distance_brouwer_zimmermann``.

    Raises:
        ValueError: As ``get_distance_brouwer_zimmermann`` says, when the search is built.
    """

    best: int

    def __init__(
        self,
        sectors: Sequence[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]],
        *,
        cutoff: int,
        weight_divisor: int = 1,
        upper_bound: int | None = None,
        get_disjoint_basis_families: Sequence[Callable[[], basis_families.BasisFamily]]
        | None = None,
        get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
        tuning: SearchTuning | None = None,
    ) -> None:
        self._cutoff = cutoff
        self._searches: list[_SectorSearch] = []
        self._next = 0
        if not sectors:
            if upper_bound is None:
                raise ValueError("Require at least one sector or an upper bound")
            self.best = upper_bound
            return

        # Prepare a search of each sector, which setup may settle without searching.  Every sector
        # scans on the same number of threads.
        tuning = get_tuning() if tuning is None else tuning
        num_threads = _threads.get_num_threads(tuning.num_threads)
        initial_upper_bounds = [
            _get_lightest_eligible_row_weight(basis, labels) for basis, labels in sectors
        ]
        best = min(initial_upper_bounds)
        if upper_bound is not None:
            best = min(best, upper_bound)
        for index, ((basis, labels), initial_upper_bound) in enumerate(
            zip(sectors, initial_upper_bounds, strict=True)
        ):
            prepared = _prepare_sector_search(
                basis,
                labels,
                cutoff=cutoff,
                weight_divisor=weight_divisor,
                upper_bound=best,
                initial_upper_bound=initial_upper_bound,
                get_disjoint_basis_family=None
                if get_disjoint_basis_families is None
                else get_disjoint_basis_families[index],
                get_automorphisms=get_automorphisms,
                tuning=tuning,
                num_threads=num_threads,
            )
            if isinstance(prepared, int):
                best = min(best, prepared)
            else:
                self._searches.append(prepared)
        self.best = best
        self._share_best()

    @property
    def finished(self) -> bool:
        """Whether the best upper bound is at most the cutoff, or every sector's search finished."""
        return self.best <= self._cutoff or all(search.finished for search in self._searches)

    def get_next_layer_work(self) -> int:
        """The candidate/set evaluations of the layer that ``advance`` scans next, or zero if none.

        This plans the layer as ``advance`` does, but leaves out what ``advance`` may do first:
        attach the bound of the images of the blocks, which can shorten the layer, and switch to an
        automorphism family, after which the layer scans one set.
        """
        if self.finished:
            return 0
        return self._searches[self._get_next_index()].get_layer_work()

    def advance(self) -> None:
        """Advance the next unfinished sector's search by one layer, round-robin, unless finished.

        The search of every sector then takes the best upper bound, which shares witnesses between
        sectors.  If this raises, as at a keyboard interrupt, the search can be left inconsistent,
        so build a new one to search again.

        Raises:
            ValueError: If a generator from ``get_automorphisms`` is not an automorphism of the
                sector whose search uses it.
        """
        if self.finished:
            return
        self._next = self._get_next_index()
        search = self._searches[self._next]
        self._next = (self._next + 1) % len(self._searches)
        self.best = min(self.best, search.advance())
        self._share_best()

    def _get_next_index(self) -> int:
        """The index of the next unfinished search in round-robin order, while one is unfinished."""
        index = self._next
        while self._searches[index].finished:
            index = (index + 1) % len(self._searches)
        return index

    def _share_best(self) -> None:
        """Give each sector's search the best upper bound; one that finishes frees its tables."""
        for search in self._searches:
            search.update_upper_bound(self.best)


####################################################################################################
# resumable searches


@dataclass
class _SectorSearch:
    """State for the resumable Brouwer-Zimmermann search of one sector.

    ``instance`` holds the packed sets that the search scans, and the certificate of its family:
    those sets, or, if ``uses_automorphism_family``, every image of the one packed set under
    automorphisms.  Once the search gets automorphisms, and until it switches, the certificate of
    its family also covers the images of its pivot blocks, unless
    ``tuning.automorphism_orbit_bounds`` is off (Section 3.7).  No eligible word weighs less than
    ``min(lower_bound, best)``, and the search has finished once ``lower_bound`` reaches ``best``.
    ``lower_bound`` bounds only the eligible words lighter than ``best`` (Section 3.9, "Certificates
    and contracts").

    ``get_automorphism_family``, if given, returns an automorphism family of the sector, or None.
    The search calls it at most once, and then sets it to None.  A search that declines to switch
    keeps the family in ``automorphism_family``, and its best weight at that time in
    ``declined_best`` (Section 3.7, "When a search uses an automorphism family").
    ``completion_tables`` keeps the tables last chosen, for the set ``completion_tables_set``, for
    later layers of that set (Section 3.9, "When a search uses the tables").  ``num_threads`` is the
    number of threads on which the search scans a costly layer of a set (Section 4.5).
    """

    instance: affine_instance.AffineInstance
    cutoff: int
    batch_size: int
    tuning: SearchTuning
    best: int
    lower_bound: int
    next_weight: int = 1
    num_threads: int = 1
    get_automorphism_family: _GetAutomorphismFamily | None = field(default=None, repr=False)
    uses_automorphism_family: bool = field(default=False, init=False)
    automorphism_family: _AutomorphismFamily | None = field(default=None, init=False, repr=False)
    declined_best: int = field(default=0, init=False)
    completion_tables: completion_bounds.CompletionTables | None = field(
        default=None, init=False, repr=False
    )
    completion_tables_set: int = field(default=0, init=False, repr=False)

    @property
    def dimension(self) -> int:
        """Number of basis rows searched."""
        return self.instance.generators.shape[1]

    @property
    def finished(self) -> bool:
        """Whether this sector cannot improve the shared upper bound."""
        return (
            self.best <= self.cutoff
            or self.lower_bound >= self.best
            or self.next_weight > self.dimension
        )

    def get_layer_work(self) -> int:
        """The candidate/set evaluations of the sets that ``advance`` plans to scan next."""
        num_scanned, _ = self._plan_layer()
        return num_scanned * math.comb(self.dimension, self.next_weight)

    def update_upper_bound(self, upper_bound: int) -> None:
        """Import a witness found by another search, and free the tables of a finished search."""
        self.best = min(self.best, upper_bound)
        if self.finished:
            self.completion_tables = None

    def advance(self) -> int:
        """Search the next coefficient layer, and return the best shared upper bound.

        The search scans the sets in order, and stops after the first set whose step certifies the
        best weight, as lowered by any lighter word that the scan finds (Section 2.5).  A search of
        an automorphism family scans its one set, which completes the layer for every image.
        """
        if self.finished:
            return self.best

        # Plan to scan the sets up to the first whose step certifies the best weight.
        weight = self.next_weight
        certificate = self.instance.certificate
        num_sets = len(certificate.ranks)
        num_scanned, num_completed = self._plan_layer()

        # Before a costly layer, use automorphisms: bound with the images of the blocks, which may
        # finish the search, and consider switching to an automorphism family (Section 3.7).  The
        # layer work saturates at ``WORK_CAP``, as work estimates do, so that a threshold above it
        # forbids asking for automorphisms.
        layer_work = min(self.get_layer_work(), certificates.WORK_CAP)
        if (
            self._considers_automorphism_family
            and layer_work >= self.tuning.automorphism_minimum_layer_work
        ):
            self._switch_to_automorphism_family()
            return self.advance()

        # Scan the sets in order.  Each set steps through the supports of the layer on its own,
        # which keeps the scan simple but repeats the stepping.
        for index in range(num_scanned):
            if not self._scan_set(index, weight):
                return self.best
            if index + 1 < num_scanned and certificate.certifies(weight, index + 1, self.best):
                num_completed = index + 1
                break

        # Advance the certificate.  The bound of a family scanned before a switch still holds
        # (Section 3.7).
        self.lower_bound = max(
            self.lower_bound, certificate.step_lower_bound(weight, num_completed)
        )
        if num_completed == num_sets:
            self.next_weight += 1
        return self.best

    def _plan_layer(self) -> tuple[int, int]:
        """The numbers of sets that the next layer scans and completes, as ``advance`` plans.

        The layer scans the sets up to the first whose step certifies the best weight, or the one
        set of an automorphism family, which completes the layer for every image.
        """
        num_sets = len(self.instance.certificate.ranks)
        if self.uses_automorphism_family:
            return 1, num_sets
        num_completed = self.instance.certificate.get_num_sets_to_certify(
            self.next_weight, self.best
        )
        return num_completed, num_completed

    @property
    def _scan_cutoff(self) -> int:
        """The cutoff that the compiled scans take, which is ``cutoff`` clamped below at -1.

        Every negative cutoff disables early exits alike, and -1 is within int64 range.
        """
        return max(-1, self.cutoff)

    def _scan_set(self, index: int, weight: int) -> bool:
        """Scan a layer of the set ``index``, with the completion tables that it chooses, if any.

        The block of the set need not be an information set.  A costly layer of three or more rows
        is split across threads (Section 4.5).  Returns False if the scan stopped at the cutoff.
        """
        tables = self._select_completion_tables(index, weight)
        instance = self.instance
        if (
            self.num_threads > 1
            and weight >= 3
            and math.comb(self.dimension, weight) > self.num_threads * self.batch_size
        ):
            return self._scan_set_on_threads(index, weight, tables)
        support = np.zeros(weight, dtype=np.int64)
        prefixes = np.zeros((weight + 1, instance.generators.shape[2]), dtype=np.uint64)

        # Scan in chunks, returning early once the upper bound reaches the cutoff.
        depth = 0
        while depth >= 0:
            self.best, depth, _, _, _ = affine_search.scan_set(
                instance.generators[index],
                instance.labels[index],
                instance.ranks[index],
                instance.widths[index],
                instance.labels.shape[2],
                support,
                prefixes,
                depth,
                self.best,
                self._scan_cutoff,
                self.batch_size,
                tables.tables,
                tables.offsets,
                tables.block_bits,
            )
            if self.best <= self.cutoff:
                return False
        return True

    def _scan_set_on_threads(
        self, index: int, weight: int, tables: completion_bounds.CompletionTables
    ) -> bool:
        """Scan a layer of the set ``index`` on threads, as ``_scan_set`` scans it on one.

        The threads claim ranges of the work items of the layer in order, weighted by their numbers
        of supports.  Each thread scans in chunks of ``batch_size * num_threads`` supports and
        lookups, and lowers the best weight after each chunk.  It starts each chunk from the best
        weight so far, which skips more, but the best weight after the layer is the same as on one
        thread whatever the chunks start from (Section 4.5).  Returns False if the scan stopped at
        the cutoff.
        """
        instance = self.instance
        generators = instance.generators[index]
        set_arguments = (
            generators,
            instance.labels[index],
            instance.ranks[index],
            instance.widths[index],
            instance.labels.shape[2],
        )
        table_arguments = (tables.tables, tables.offsets, tables.block_bits)
        items = _WorkItems(self.dimension, weight, self.num_threads)
        lock = threading.Lock()
        # Each thread holds the global interpreter lock between its chunks, so chunks that grow with
        # the number of threads keep the share of time under the lock about constant.
        chunk_size = min(self.batch_size * self.num_threads, MAX_VALUE)

        def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
            """Give a thread its own support and prefixes, and return how it scans a range."""
            support = np.zeros(weight, dtype=np.int64)
            prefixes = np.zeros((weight + 1, generators.shape[1]), dtype=np.uint64)

            def scan_range(start: int, end: int) -> None:
                """Scan the items ``start, ..., end - 1`` in chunks, until done or stopped."""
                depth = affine_search.start_set_items(
                    generators, items.get_item(start), support, prefixes
                )
                last_item = items.get_item(end - 1)

                # Start each chunk from the best weight so far, which skips more, and lower it after
                # the chunk.
                while depth >= 0 and not stop.is_set():
                    best, depth, _, _, _ = affine_search.scan_set_items(
                        *set_arguments,
                        support,
                        prefixes,
                        depth,
                        self.best,
                        self._scan_cutoff,
                        chunk_size,
                        *table_arguments,
                        last_item,
                    )
                    if best < self.best:
                        with lock:
                            self.best = min(self.best, best)
                        if self.best <= self.cutoff:
                            stop.set()

            return scan_range

        # Compile the scan before the threads call it, so that the threads do not all wait behind
        # numba's compiler lock while one compiles it.  A scan from depth -1 returns at once.
        affine_search.scan_set_items(
            *set_arguments,
            np.zeros(weight, dtype=np.int64),
            np.zeros((weight + 1, generators.shape[1]), dtype=np.uint64),
            -1,
            self.best,
            self._scan_cutoff,
            chunk_size,
            *table_arguments,
            items.get_item(0),
        )

        # Scan the work items on the threads.
        _threads.run_work_items(
            items.num_items,
            self.num_threads,
            start_worker,
            cumulative_weights=items.cumulative_weights,
        )
        return self.best > self.cutoff

    def _select_completion_tables(
        self, index: int, weight: int
    ) -> completion_bounds.CompletionTables:
        """Choose the completion tables of a layer of the set ``index``.

        Only an information set can have tables.  The search frees the tables of any other set
        before choosing, so that it holds those of one set at a time, and keeps the chosen tables.

        Returns:
            The chosen tables, or ``completion_bounds.NO_TABLES``.
        """
        instance = self.instance
        if instance.ranks[index] < self.dimension:
            return completion_bounds.NO_TABLES
        if self.completion_tables_set != index:
            self.completion_tables = None
        tables = completion_bounds.select_completion_tables(
            instance.generators[index, :, : instance.widths[index]],
            weight,
            self.best,
            current=self.completion_tables,
            tuning=self.tuning,
        )
        if tables is None:
            return completion_bounds.NO_TABLES
        self.completion_tables, self.completion_tables_set = tables, index
        return tables

    @property
    def _considers_automorphism_family(self) -> bool:
        """Whether the search would consider switching to an automorphism family before a layer.

        It would until it has switched or found no family, and after declining, only once its best
        weight has dropped (Section 3.7).  The property is False right after every consideration,
        which ends the recursion of ``advance``.
        """
        if self.get_automorphism_family is not None:
            return True
        return self.automorphism_family is not None and self.best < self.declined_best

    def _switch_to_automorphism_family(self) -> None:
        """Bound with the images of the blocks, and switch to the images of one set if that pays.

        The first call gets the automorphism family, which the search keeps while it declines, and
        lets the current family certify with the images of its pivot blocks.  The search then
        switches to scan one set and certify with its images, if that predicts less work.  After a
        switch, the search scans the new set from layer two, and keeps its lower bound (Section 3.7,
        "When a search uses an automorphism family").
        """
        if self.get_automorphism_family is not None:
            weight_divisor = self.instance.certificate.weight_divisor
            self.automorphism_family = self.get_automorphism_family(weight_divisor)
            self.get_automorphism_family = None
        if self.automorphism_family is None:
            return

        # The rows of the scanned set weigh as much as the layer-one words of every image, and may
        # lower the best weight before the estimate.
        basis_family, certificate, orbits = self.automorphism_family
        self.best = _get_basis_family_upper_bound(basis_family, self.best)

        # From now on, the family also certifies with the images of its pivot blocks under the
        # automorphisms, which can finish the search without a switch (Section 3.7).
        if self.tuning.automorphism_orbit_bounds and not self.instance.certificate.column_orbits:
            self.instance = dataclasses.replace(
                self.instance,
                certificate=dataclasses.replace(
                    self.instance.certificate,
                    pivot_blocks=basis_families.get_pivot_blocks(
                        self.instance.pivots, self.instance.ranks
                    ),
                    lp_max_denominator=self.tuning.lp_dual_max_denominator,
                    column_orbits=tuple(int(orbit) for orbit in orbits),
                ),
            )
            self.lower_bound = max(
                self.lower_bound, self.instance.certificate.lower_bound(self.next_weight - 1)
            )
            if self.finished:
                return

        # Switch if the automorphism family predicts less work than the current family has left.
        work = automorphism_families.estimate_automorphism_family_work(certificate, self.best)
        if work >= self.instance.certificate.work(self.best, self.next_weight):
            self.declined_best = self.best
            return
        self.instance = affine_instance.get_distance_instance(basis_family, certificate)
        self.lower_bound = max(self.lower_bound, certificate.lower_bound(1))
        self.next_weight = 2
        self.uses_automorphism_family = True
        self.automorphism_family = None
        # Tables of the old set bound nothing in the new one.
        self.completion_tables = None


class _WorkItems:
    """The work items of a layer that threads scan: the supports whose first rows are fixed.

    An item fixes the first ``length`` rows of its supports: one row, or two if fixing one row
    leaves fewer than ``_ITEMS_PER_THREAD`` items for each thread and the layer has four or more
    rows.  So ``length <= weight - 2``, as ``affine_search.scan_set_items`` requires.  The items are
    numbered in lexicographic order.  Their numbers of supports vary widely, and earlier items tend
    to hold more.  ``cumulative_weights`` holds, for each item, the number of supports of the items
    up to it, as floats, which overflow only for layers far too large to scan.
    """

    def __init__(self, num_rows: int, weight: int, num_threads: int) -> None:
        few = num_rows - weight + 1 < _ITEMS_PER_THREAD * num_threads
        self.length = 2 if few and weight >= 4 else 1

        # The fixed rows of an item are combinations of the rows that leave room for the
        # ``weight - length`` later rows of its supports.  Count the items whose first row is less
        # than each row.
        num_values = num_rows - weight + self.length
        self.num_items = math.comb(num_values, self.length)
        self._starts = np.array(
            [self.num_items - math.comb(num_values - row, self.length) for row in range(num_values)]
        )

        # An item whose last fixed row is ``last`` completes its supports with ``weight - length``
        # of the ``num_rows - 1 - last`` later rows.
        last_rows = (
            np.arange(num_values)
            if self.length == 1
            else np.concatenate([np.arange(first + 1, num_values) for first in range(num_values)])
        )
        num_supports_by_last_row = np.array(
            [
                float(math.comb(num_rows - 1 - last, weight - self.length))
                for last in range(num_values)
            ]
        )
        self.cumulative_weights = np.cumsum(num_supports_by_last_row[last_rows])

    def get_item(self, index: int) -> npt.NDArray[np.int64]:
        """The fixed rows of the item ``index``."""
        first = int(np.searchsorted(self._starts, index, side="right")) - 1
        rows = [first, first + 1 + index - int(self._starts[first])]
        return np.array(rows[: self.length], dtype=np.int64)


def _prepare_sector_search(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    *,
    cutoff: int,
    weight_divisor: int,
    upper_bound: int,
    initial_upper_bound: int | None = None,
    get_disjoint_basis_family: Callable[[], basis_families.BasisFamily] | None = None,
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
    tuning: SearchTuning,
    num_threads: int = 1,
) -> int | _SectorSearch:
    """Prepare the BZ search of one sector, or return its result if setup already settles it.

    The stages of ``basis_families`` run in order, and between them setup returns if the rows of the
    family bring the best weight to the cutoff, or if its certificate proves the best weight
    (Section 4.2).  ``get_disjoint_basis_family``, if given, returns the result of the first stage,
    and ``get_automorphisms`` is as for ``get_distance_brouwer_zimmermann``.  The search scans on
    ``num_threads`` threads (Section 4.5).
    """
    dimension = len(basis)
    best = min(
        upper_bound,
        (
            _get_lightest_eligible_row_weight(basis, labels)
            if initial_upper_bound is None
            else initial_upper_bound
        ),
    )
    if best <= cutoff:
        return best

    # The certificate of the disjoint family also uses a weight divisor of two if the family infers
    # one (Section 3.2).
    basis_family = (
        basis_families.get_disjoint_family(
            basis,
            labels,
            random_column_orders=tuning.disjoint_random_column_orders,
            weight_divisor=weight_divisor,
        )
        if get_disjoint_basis_family is None
        else get_disjoint_basis_family()
    )
    best = _get_basis_family_upper_bound(basis_family, best)
    if best <= cutoff:
        return best

    basis_family = basis_families.prune_family(basis_family, upper_bound=best)
    # The rows of the sets, which the upper bound above weighed, complete layer one.
    completed_weight = 1
    lower_bound = basis_family.certificate.lower_bound(completed_weight)
    if lower_bound >= best:
        return best

    # Build the automorphism family at most once (Section 3.7).  Layer work saturates at
    # ``WORK_CAP``, so a threshold above it turns automorphisms off.
    get_automorphism_family = (
        None
        if get_automorphisms is None
        or tuning.automorphism_minimum_layer_work > certificates.WORK_CAP
        else functools.cache(
            functools.partial(_get_automorphism_family, basis, labels, get_automorphisms)
        )
    )

    # Overlapping sets, and then matroid partitioning, may each replace the family if they predict
    # less work (Section 4.4).  Partitioning yields to an automorphism family that would do no more
    # work than any partitioned family could (Section 3.7).
    for get_replacement in (
        lambda family, best: basis_families.get_overlapping_family(
            basis,
            labels,
            family,
            best,
            basis_families.get_overlapping_family_policy(tuning, dimension),
        ),
        lambda family, best: basis_families.get_partitioned_family(
            basis,
            labels,
            family,
            best,
            tuning,
            get_competing_work=None
            if get_automorphism_family is None
            else functools.partial(
                _get_automorphism_family_work,
                get_automorphism_family,
                family.weight_divisor,
                best,
            ),
        ),
    ):
        replacement = get_replacement(basis_family, best)
        if not len(replacement.ranks):
            continue
        basis_family = replacement

        # Check the replacement rows for lighter witnesses before scanning combinations.
        best = _get_basis_family_upper_bound(basis_family, best)
        if best <= cutoff:
            return best

        # The certificate of the replacement uses its own ranks and coverage.
        lower_bound = basis_family.certificate.lower_bound(completed_weight)
        if lower_bound >= best:
            return best

    search = _SectorSearch(
        affine_instance.get_distance_instance(
            basis_family,
            basis_family.get_search_certificate(
                best, tuning.lp_minimum_work, tuning.lp_dual_max_denominator
            ),
        ),
        cutoff=cutoff,
        batch_size=1 << tuning.information_set_batch_exponent,
        tuning=tuning,
        best=best,
        lower_bound=lower_bound,
        next_weight=completed_weight + 1,
        num_threads=num_threads,
        get_automorphism_family=get_automorphism_family,
    )

    # If setup has already asked for the automorphism family, use it now: its rows and certificate,
    # and the images of the blocks of the current family, can lower the best weight or finish the
    # search (Section 3.7).
    if get_automorphism_family is not None and get_automorphism_family.cache_info().currsize:
        search._switch_to_automorphism_family()
        if search.finished:
            return search.best
    return search


def _get_automorphism_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None],
    weight_divisor: int,
) -> _AutomorphismFamily | None:
    """Verify the automorphisms of a sector, and build the automorphism family of one set.

    ``basis`` is an independent basis of the searched space, with quotient labels ``labels``, and
    ``weight_divisor`` divides the Hamming weight of every eligible word.

    Returns:
        The family and certificate of ``automorphism_families.get_automorphism_family``, and the
        orbit of each column under the group, or None if ``get_automorphisms`` returns None.

    Raises:
        ValueError: If a generator is not an automorphism of the sector.
    """
    automorphisms = get_automorphisms()
    if automorphisms is None:
        return None
    automorphism_families.verify_automorphisms(basis, labels, automorphisms)
    return (
        *automorphism_families.get_automorphism_family(
            basis, labels, automorphisms, weight_divisor
        ),
        automorphism_families.get_orbits(automorphisms, basis.shape[1]),
    )


def _get_automorphism_family_work(
    get_automorphism_family: _GetAutomorphismFamily,
    weight_divisor: int,
    upper_bound: int,
) -> int:
    """The estimated work of the automorphism family that a search can switch to.

    Returns ``WORK_CAP`` if the sector has no automorphism family.
    """
    automorphism_family = get_automorphism_family(weight_divisor)
    if automorphism_family is None:
        return certificates.WORK_CAP
    _, certificate, _ = automorphism_family
    return automorphism_families.estimate_automorphism_family_work(certificate, upper_bound)


def _get_lightest_eligible_row_weight(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
) -> int:
    """Return the weight of the lightest eligible row of a nested-code basis."""
    packed_basis = pack_bits(basis[get_eligible_rows(labels)])
    # Sum the weights of the packed words of each row in uint64, so that wide rows cannot overflow.
    return int(np.bitwise_count(packed_basis).sum(axis=-1, dtype=np.uint64).min())


def _get_basis_family_upper_bound(
    basis_family: basis_families.BasisFamily,
    upper_bound: int,
) -> int:
    """Tighten an upper bound with the systematic rows of a basis family: its layer-one words."""
    for generators, set_labels in zip(basis_family.generators, basis_family.labels, strict=True):
        upper_bound = min(upper_bound, _get_lightest_eligible_row_weight(generators, set_labels))
    return upper_bound
