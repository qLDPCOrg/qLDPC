# SPDX-License-Identifier: Apache-2.0

"""Exact BZ searches with disjoint, overlapping, or automorphism families of information sets.

The search follows `Algorithm 994 <https://arxiv.org/abs/1603.06757>`_ and its
`adaptation to quantum codes <https://arxiv.org/abs/2408.10743>`_.  This module prepares each
search, takes its early exits, and scans its layers in resumable chunks with ``affine_search.py``.
``brouwer_zimmermann.md`` and ``brouwer_zimmermann_code.md`` derive the certificates and invariants
of the search.  A costly layer runs on several threads
(``brouwer_zimmermann_code.md#sec-cross-file-behavior``), so a change here must keep the
requirements on threads of ``brouwer_zimmermann_code.md#sec-requirements``, such as "Decide only
from what a layer leaves the same on any number of threads".

Here ``U`` is the best upper bound (``brouwer_zimmermann.md#sec-bounds-and-certificates``), and
``L(t)`` the lower certificate once layer ``t`` is complete.
``get_distance_brouwer_zimmermann`` builds a ``BrouwerZimmermannSearch``, which prepares a search of
each sector with ``_prepare_sector_search``.  Each call of its ``advance`` method advances the
search of one remaining sector by one layer, in round-robin order for a joint CSS search
(``brouwer_zimmermann.md#sec-shared-witnesses``).  Preparation returns a result, or a
``_SectorSearch`` whose ``advance`` method searches one layer ``t``, or only its first ``j`` sets if
step ``(t, j)`` certifies ``U`` (``brouwer_zimmermann.md#sec-stopping-certificate``).  A result of
preparation can include a witness from another sector or the bound of a weight limit, so it need not
be the distance of its sector.

``_prepare_sector_search`` considers the following exits in order.  The disjoint family, its
pruning, and any overlapping or partitioned replacement come from the stages of
``_information_sets/basis_families.py``; the exits between those stages belong to this module.

- Initial row witness: returns if ``U <= cutoff``, where ``U`` includes the lightest eligible row
  and any supplied bound (from another sector or a weight limit).
- Rows of the disjoint family: returns if checking the rows of every set gives ``U <= cutoff``.
- Certificate of the disjoint family: returns if ``L(1) >= U``.
- Rows of each replacement family, overlapping and then partitioned, if selected: returns if
  ``U <= cutoff``.
- Certificate of each replacement family: returns if ``L(1) >= U``.
- Switch at the end of setup: if setup asked for automorphisms before partitioning and got a family
  from them, the search weighs the rows of that family's one set and considers switching to the
  family (``brouwer_zimmermann.md#sec-automorphism-gate``); returns if the search is then finished
  (see ``_SectorSearch.finished``).
- Ready search: returns a ``_SectorSearch`` that scans from layer two.

Once prepared, the searches exit as follows:

- ``_SectorSearch.advance`` returns after one layer, or after the leading sets of a layer that
  certify ``U``, or earlier if ``U <= cutoff``.  Before a costly layer, a search may first get
  automorphisms and consider switching to an automorphism family, whose rows may lower ``U`` and
  whose certificate may finish the search.  A finished search does no further work.
- Once every sector's search has finished, or the shared ``U <= cutoff``,
  ``get_distance_brouwer_zimmermann`` returns ``U``.
"""

from __future__ import annotations

import functools
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits

from .. import _threads
from .._code_matrices import get_eligible_rows
from .._information_sets import (
    automorphism_families,
    basis_families,
    certificates,
    completion_bounds,
    support_stepping,
)
from .._tuning import MAX_VALUE, SearchTuning, get_tuning
from . import affine_instance, affine_search

# The automorphism family of a sector: a basis family of the one set that a search scans, and the
# certificate of all of its images.
_AutomorphismFamily = tuple[basis_families.BasisFamily, certificates.Certificate]

# A function that returns the automorphism family of a sector for a weight divisor, or None.
_GetAutomorphismFamily = Callable[[int], _AutomorphismFamily | None]


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
    the lightest witness that any of them has found, but not their certificates
    (``brouwer_zimmermann.md#sec-shared-witnesses``).

    Args:
        sectors: Nested-code bases and their quotient labels (see ``get_nested_code_basis``).  The
            distance of a sector is the least weight of a word of its basis with nonzero labels, or
            of any nonzero word if its labels have no columns.  Each sector must have such a word.
        cutoff: Return once the best upper bound is at most this cutoff.
        weight_divisor: A proven divisor of the weights over which the sectors minimize, to whose
            multiples the certificates round up (``brouwer_zimmermann.md#sec-weight-multiples``).
            Default: 1.
        upper_bound: A cap on the result, if any, such as the weight of a known witness outside
            ``sectors``, or ``max_weight + 1`` under a weight limit
            (``brouwer_zimmermann_code.md#sec-cross-file-behavior``, "Weight limit").
        get_disjoint_basis_families: If given, a function for each sector that returns a disjoint
            basis family of it, with its labels and certified with ``weight_divisor``, such as that
            of ``basis_families.get_disjoint_family``.  A search calls it only if it needs the
            family.
        get_automorphisms: If given, returns None or an array whose rows ``g`` generate a group of
            column permutations, each mapping column ``j`` to column ``g[j]``.  Each generator must
            map the space and the excluded subspace of every sector onto themselves.  The search
            of each sector calls the function at most once, so a search of two sectors may call it
            twice.  Each search verifies the generators
            (``brouwer_zimmermann.md#sec-automorphism-sources``).
        tuning: The tuning that every search uses throughout.  Default: the active tuning.

    Returns:
        The minimum of the distances of the sectors and ``upper_bound``.  After an early exit, it is
        a value at most ``cutoff``: the weight of a witness, or ``upper_bound``.

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
        best = min(_get_lightest_eligible_row_weight(basis, labels) for basis, labels in sectors)
        if upper_bound is not None:
            best = min(best, upper_bound)
        for index, (basis, labels) in enumerate(sectors):
            prepared = _prepare_sector_search(
                basis,
                labels,
                cutoff=cutoff,
                weight_divisor=weight_divisor,
                upper_bound=best,
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

        This plans the layer as ``advance`` does, but leaves out what ``advance`` may first do with
        an automorphism family.  Weighing the family's rows can lower the best weight, and so
        shorten the layer.  Unless switching to the family finishes the search, it makes ``advance``
        scan layer two of the family's one set rather than the layer planned here.
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
    automorphisms.  ``best`` and ``lower_bound`` are the upper bound ``U`` and the lower certificate
    ``L`` of ``brouwer_zimmermann.md#sec-bounds-and-certificates``, and ``next_weight`` is the layer
    after the last that every set has completed.  No eligible word weighs less than
    ``min(lower_bound, best)``, and the search has finished once ``lower_bound`` reaches ``best``.
    ``lower_bound`` bounds only the eligible words lighter than ``best``
    (``brouwer_zimmermann.md#sec-completion-contracts``).

    ``get_automorphism_family``, if given, returns an automorphism family of the sector, or None.
    The search calls it at most once, and then sets it to None.  A search that declines to switch
    keeps the family in ``automorphism_family``, and its best weight at that time in
    ``declined_best`` (``brouwer_zimmermann.md#sec-automorphism-gate``).  ``completion_tables``
    keeps the tables last chosen, for the set ``completion_tables_set``, for later layers of that
    set (``brouwer_zimmermann.md#sec-completion-gate``).  ``num_threads`` is the number of threads
    on which the search scans a costly layer of a set
    (``brouwer_zimmermann_code.md#sec-cross-file-behavior``, "Threads").
    """

    instance: affine_instance.AffineInstance
    cutoff: int
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
    def batch_size(self) -> int:
        """The most supports and lookups that one call of a compiled scan does on one thread.

        When the search scans on threads, each call does ``num_threads`` times as many, up to
        ``MAX_VALUE``, as the resumable layers of
        ``brouwer_zimmermann_code.md#sec-cross-file-behavior`` describe.
        """
        return 1 << self.tuning.search_chunk_exponent

    @property
    def finished(self) -> bool:
        """Whether this sector cannot improve the shared upper bound.

        It cannot once ``U <= cutoff``, once ``L >= U``, or once every layer has been enumerated.
        """
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
        best weight, as lowered by any lighter word that the scan finds
        (``brouwer_zimmermann.md#sec-stopping-certificate``).  A search of an automorphism family
        scans its one set, which completes the layer for every image.  A scan that stops at the
        cutoff or raises leaves ``lower_bound`` and ``next_weight`` unchanged, as the resumable
        layers of ``brouwer_zimmermann_code.md#sec-cross-file-behavior`` require.
        """
        if self.finished:
            return self.best

        # Plan to scan the sets up to the first whose step certifies the best weight.
        weight = self.next_weight
        certificate = self.instance.certificate
        num_sets = len(certificate.ranks)
        num_scanned, num_completed = self._plan_layer()

        # Before a costly layer, consider switching to an automorphism family, whose rows may lower
        # the best weight and so finish the search.
        if (
            self._considers_automorphism_family
            and self.get_layer_work() >= self.tuning.automorphism_minimum_layer_work
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
        # (``brouwer_zimmermann.md#sec-automorphism-gate``).
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
        is split across threads (``brouwer_zimmermann_code.md#sec-cross-file-behavior``, "Threads").
        Returns False if the scan stopped at the cutoff.
        """
        tables = self._select_completion_tables(index, weight)
        instance = self.instance
        if (
            self.num_threads > 1
            and weight >= 3
            and math.comb(self.dimension, weight) > self.num_threads * self.batch_size
        ):
            return self._scan_set_on_threads(index, weight, tables)
        basis = self._get_packed_basis(index)
        support = np.zeros(weight, dtype=np.int64)
        prefixes = np.zeros((weight + 1, instance.generators.shape[2]), dtype=np.uint64)

        # Scan in chunks, returning early once the upper bound reaches the cutoff.
        depth = 0
        while depth >= 0:
            self.best, depth, _, _, _ = affine_search.scan_set(
                basis,
                support,
                prefixes,
                depth,
                self.best,
                self._scan_cutoff,
                self.batch_size,
                tables,
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
        lookups.  It starts each chunk from the shared best weight so far, so that the tables skip
        more, and lowers that weight after the chunk.  Whatever weights the chunks start from, the
        best weight after the layer is that of a scan on one thread
        (``brouwer_zimmermann_code.md#sec-cross-file-behavior``, "Threads").  Returns False if the
        scan stopped at the cutoff.
        """
        basis = self._get_packed_basis(index)
        generators = basis.rows
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

                while depth >= 0 and not stop.is_set():
                    best, depth, _, _, _ = affine_search.scan_set_items(
                        basis,
                        support,
                        prefixes,
                        depth,
                        self.best,
                        self._scan_cutoff,
                        chunk_size,
                        tables,
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
            basis,
            np.zeros(weight, dtype=np.int64),
            np.zeros((weight + 1, generators.shape[1]), dtype=np.uint64),
            -1,
            self.best,
            self._scan_cutoff,
            chunk_size,
            tables,
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

    def _get_packed_basis(self, index: int) -> support_stepping.PackedBasis:
        """The set ``index``, with its rows, labels, rank, and width, as the scans read it."""
        instance = self.instance
        return support_stepping.PackedBasis(
            instance.generators[index],
            instance.labels[index],
            instance.ranks[index],
            instance.widths[index],
        )

    def _select_completion_tables(
        self, index: int, weight: int
    ) -> completion_bounds.CompletionTables:
        """Choose the completion tables of a layer of the set ``index``.

        Only a full set, one whose rank is the dimension, can have tables.  The search frees the
        tables of any other set before choosing, so that it holds those of one set at a time, and
        keeps the chosen tables.

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
        weight has dropped (``brouwer_zimmermann.md#sec-automorphism-gate``).  The property is False
        right after every consideration, which ends the recursion of ``advance``.
        """
        if self.get_automorphism_family is not None:
            return True
        return self.automorphism_family is not None and self.best < self.declined_best

    def _switch_to_automorphism_family(self) -> None:
        """Switch to scanning the images of one set under automorphisms, if that pays.

        The first call gets the automorphism family, which the search keeps while it declines.  The
        search switches to scan one set and certify with its images, if that predicts less work.
        After a switch, the search packs the new family in another instance, scans the new set from
        layer two, and keeps its lower bound (``brouwer_zimmermann.md#sec-automorphism-gate``).
        """
        if self.get_automorphism_family is not None:
            weight_divisor = self.instance.certificate.weight_divisor
            self.automorphism_family = self.get_automorphism_family(weight_divisor)
            self.get_automorphism_family = None
        if self.automorphism_family is None:
            return

        # The rows of the scanned set weigh as much as the layer-one words of every image, and may
        # lower the best weight before the estimate.
        basis_family, certificate = self.automorphism_family
        self.best = _get_basis_family_upper_bound(basis_family, self.best)

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
    leaves fewer than ``_threads.ITEMS_PER_THREAD`` items for each thread and the layer has four or
    more rows.  So ``length <= weight - 2``, as ``affine_search.scan_set_items`` requires.  The
    items are numbered in lexicographic order.  Their numbers of supports vary widely, and earlier
    items tend to hold more.  ``cumulative_weights`` holds, for each item, the number of supports of
    the items up to it, as floats, which overflow only for layers far too large to scan.
    ``brouwer_zimmermann_code.md#sec-cross-file-behavior`` ("Threads") says why the items partition
    the supports that a scan of the layer weighs or skips.
    """

    def __init__(self, num_rows: int, weight: int, num_threads: int) -> None:
        few = num_rows - weight + 1 < _threads.ITEMS_PER_THREAD * num_threads
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
    get_disjoint_basis_family: Callable[[], basis_families.BasisFamily] | None = None,
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
    tuning: SearchTuning,
    num_threads: int = 1,
) -> int | _SectorSearch:
    """Prepare the BZ search of one sector, or return its result if setup already settles it.

    ``upper_bound`` is the best weight so far, which must be at most the weight of the lightest
    eligible row of ``basis``.  The stages of ``basis_families`` run in order, and between them
    setup returns if the rows of the family bring the best weight to the cutoff, or if its
    certificate proves the best weight, at the exits that the module docstring lists.
    ``get_disjoint_basis_family``, if given, returns the result of the first stage, and
    ``get_automorphisms`` is as for ``get_distance_brouwer_zimmermann``.  The search scans on
    ``num_threads`` threads (``brouwer_zimmermann_code.md#sec-cross-file-behavior``, "Threads").
    """
    dimension = len(basis)
    best = upper_bound
    if best <= cutoff:
        return best

    # The certificate of the disjoint family also uses a weight divisor of two if the family infers
    # one (``brouwer_zimmermann.md#sec-weight-multiples``).
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

    # Build the automorphism family at most once (``brouwer_zimmermann.md#sec-automorphism-gate``).
    # A threshold above ``WORK_CAP`` turns automorphisms off (see ``_tuning.py``).  Setup asks
    # before partitioning, whatever the threshold, and ``advance`` asks once a layer's work, which
    # may exceed ``WORK_CAP``, reaches the threshold.  So only this check keeps a threshold above
    # ``WORK_CAP`` from building the family.
    get_automorphism_family = (
        None
        if get_automorphisms is None
        or tuning.automorphism_minimum_layer_work > certificates.WORK_CAP
        else functools.cache(
            functools.partial(_get_automorphism_family, basis, labels, get_automorphisms)
        )
    )

    # Overlapping sets, and then matroid partitioning, may each replace the family if they predict
    # less work (``brouwer_zimmermann_code.md#sec-heuristics``).  Partitioning yields to an
    # automorphism family that would do no more work than any partitioned family could.
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
        # An empty family means that this stage found no better family.
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
        affine_instance.get_distance_instance(basis_family, basis_family.certificate),
        cutoff=cutoff,
        tuning=tuning,
        best=best,
        lower_bound=lower_bound,
        next_weight=completed_weight + 1,
        num_threads=num_threads,
        get_automorphism_family=get_automorphism_family,
    )

    # If setup has already asked for the automorphism family, which the cache then holds, use it
    # now: its rows and certificate can lower the best weight or finish the search.
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
        The family and certificate of ``automorphism_families.get_automorphism_family``, or None if
        ``get_automorphisms`` returns None.

    Raises:
        ValueError: If a generator is not an automorphism of the sector.
    """
    automorphisms = get_automorphisms()
    if automorphisms is None:
        return None
    automorphism_families.verify_automorphisms(basis, labels, automorphisms)
    return automorphism_families.get_automorphism_family(
        basis, labels, automorphisms, weight_divisor
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
    _, certificate = automorphism_family
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
