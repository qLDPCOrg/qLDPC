# SPDX-License-Identifier: Apache-2.0

"""Exact BZ searches with disjoint, overlapping, or automorphism families of information sets.

The disjoint/rank-deficient certificates follow `Algorithm 994 <https://arxiv.org/abs/1603.06757>`_;
the symplectic reduction follows the `quantum-code adaptation <https://arxiv.org/abs/2408.10743>`_.
The overlapping-set certificate and implementation invariants are derived in
``brouwer_zimmermann.md`` in ``src/qldpc/codes/distance/``.

Information-set families and their certificates are built in ``_information_sets/``, which
fixed-weight logical counting shares.  Its ``basis_families.py`` builds the families, and
``certificates.py`` their certificates.  Its ``automorphism_families.py`` builds families of the
images of one information set under automorphisms, and ``completion_bounds.py`` the tables that let
a scan skip hopeless supports.  Beside this module, ``affine_search.py`` holds the compiled scans of
layers, and ``affine_instance.py`` packs the sets of a search with their certificate.  The coset
searches share both.  This module owns witness discovery, early exits, exhaustive dispatch, and the
resumable scans of layers in chunks.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits

from .._binary import get_eligible_rows
from .._brute_force import get_nested_distance_brute_force
from .._information_sets import (
    automorphism_families,
    basis_families,
    certificates,
    completion_bounds,
)
from .._tuning import SearchTuning, get_tuning
from . import affine_instance, affine_search

# A function that returns the automorphism family of a problem for a weight divisor, or None.
_GetAutomorphismFamily = Callable[
    [int], tuple[basis_families.BasisFamily, certificates.Certificate] | None
]

####################################################################################################
# search entry points


def get_distance_brouwer_zimmermann_many(
    problems: Sequence[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]],
    *,
    cutoff: int,
    weight_divisor: int = 1,
    upper_bound: int | None = None,
    get_disjoint_basis_families: Sequence[Callable[[], basis_families.BasisFamily]] | None = None,
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None,
    tuning: SearchTuning | None = None,
) -> int:
    """Compute the minimum distance across nested-code problems by BZ searches.

    The searches advance round-robin, and share their best upper bound: the weight of the lightest
    witness that any of them has found.

    Args:
        problems: Nested-code bases and their quotient labels (see ``get_nested_code_basis``).  The
            distance of a problem is the least weight of a word of its basis with nonzero labels, or
            of any nonzero word if its labels have no columns.  Each problem must have such a word.
        cutoff: Return once the best upper bound falls to or below this cutoff.
        weight_divisor: A proven divisor of the weight of every word whose distance a problem
            minimizes over, with which the certificates round their lower bounds up.  Default: 1.
        upper_bound: A cap on the result, if any, such as the weight of a known witness that is
            not in ``problems``, or ``max_weight + 1`` under a weight limit.  The searches stop
            once they certify that every word is at least this heavy.
        get_disjoint_basis_families: If given, holds a function for each problem that returns a
            disjoint basis family of it: sets with pairwise disjoint pivot blocks that span the
            problem's basis, with quotient labels of the problem, certified with this
            ``weight_divisor``.  The result of ``basis_families.get_disjoint_family`` is one, and a
            caller that keeps such families saves that first stage.  Each function is called only if
            its search needs it.
        get_automorphisms: If given, returns an array of generators of a group of column
            permutations, or None.  Row ``i`` of the array maps column ``j`` to column ``row[j]``.
            Each generator must map the searched space and the excluded subspace of every problem
            onto themselves.  A search calls the function at most once: before matroid partitioning
            if it would partition, or else before a costly layer.  It verifies the generators for
            its problem, and may then scan fewer information sets (see ``automorphism_families``).
        tuning: Sets the thresholds that choose the route of each search (see ``_tuning``), and
            defaults to the active tuning, which every search then uses throughout.

    Returns:
        The minimum of the distances of the problems and ``upper_bound``, or, after an early exit
        at ``cutoff``, the weight of a witness or ``upper_bound``, whichever is at most
        ``cutoff``.

    Raises:
        ValueError: If there are neither problems nor an upper bound, or if a generator from
            ``get_automorphisms`` is not an automorphism of a problem whose search uses it.
    """
    if not problems:
        if upper_bound is None:
            raise ValueError("Require at least one problem or an upper bound")
        return upper_bound

    tuning = get_tuning() if tuning is None else tuning
    initial_upper_bounds = [
        _get_lightest_eligible_row_weight(basis, labels) for basis, labels in problems
    ]
    best = min(initial_upper_bounds)
    if upper_bound is not None:
        best = min(best, upper_bound)
    searches: list[_BrouwerZimmermannSearch] = []
    for index, ((basis, labels), initial_upper_bound) in enumerate(
        zip(problems, initial_upper_bounds, strict=True)
    ):
        prepared = _prepare_brouwer_zimmermann_search(
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
        )
        if isinstance(prepared, int):
            best = min(best, prepared)
        else:
            searches.append(prepared)

    # Advance every unfinished search in turn, until all have finished.
    advanced = True
    while advanced:
        advanced = False
        for search in searches:
            # Share witnesses between sectors, but keep each lower certificate independent.
            search.update_upper_bound(best)
            if search.finished:
                continue
            best = min(best, search.advance())
            advanced = True
            if best <= cutoff:
                return best
    return best


####################################################################################################
# cost models


def enumeration_is_cheaper_than_setup(
    dimension: int, length: int, max_dimension: int, setup_cost_factor: int
) -> bool:
    """Estimate whether enumerating every codeword is cheaper than building information sets.

    Setup costs dominate the Brouwer-Zimmermann search for small dimensions, up to
    ``max_dimension``.  For long, low-dimensional codes, byte-wise elimination for each information
    set can cost more than packed ``uint64`` enumeration of all ``2**dimension`` codewords, which
    ``setup_cost_factor`` times the dimension and the length estimates.  Below 63 rows, the count of
    codewords fits a native integer.
    """
    return dimension <= max_dimension or (
        dimension < 63 and 1 << dimension <= setup_cost_factor * dimension * length
    )


def _enumeration_is_cheaper_than_search(
    *,
    num_logical_rows: int,
    certificate: certificates.Certificate,
    upper_bound: int,
) -> bool:
    """Estimate whether exhaustive nested-code enumeration will outperform BZ.

    The BZ estimate pessimistically assumes that the search must certify the current upper bound,
    whereas lighter witnesses usually end the search sooner.  Although the vectorized exhaustive
    kernel is a few times cheaper per candidate than the serial Numba BZ kernel, whole-search
    timings near this boundary favor BZ until the exhaustive count reaches roughly the estimated BZ
    count, so the two counts are compared directly.
    """
    dimension = certificate.dimension
    bz_candidates = certificate.work(upper_bound)
    exhaustive_candidates = ((1 << num_logical_rows) - 1) << (dimension - num_logical_rows)
    return exhaustive_candidates <= bz_candidates


####################################################################################################
# resumable searches


@dataclass
class _BrouwerZimmermannSearch:
    """State for one resumable Brouwer-Zimmermann search.

    ``instance`` holds the packed sets that the search scans, and the certificate of the family that
    certifies the search.  The certificate records the ranks, maximum coordinate coverage, and
    proven weight divisor of that family.  This family is the packed family, unless
    ``uses_automorphism_family`` is True.  Such a search packs one set, and its family holds the
    images of that set under automorphisms of the problem, which complete each layer when the set
    does (see ``automorphism_families``).

    ``lower_bound`` bounds the weights of the eligible words lighter than ``best``: the search
    certifies that no eligible word weighs less than ``min(lower_bound, best)``, and it has finished
    once ``lower_bound`` reaches ``best``.  An eligible word lighter than ``best`` cannot have been
    examined or skipped, so the bounds of every family that the search has scanned hold for it.
    ``lower_bound`` alone can exceed the weight of heavier words, such as the words of supports that
    the search skipped, so a bound on the distance is ``min(lower_bound, best)``.

    ``get_automorphism_family``, if given, returns an automorphism family of the problem, or None.
    If the search would partition, setup calls it first, and has the search consider the switch at
    the end of setup.  Otherwise the search calls it before the first layer whose scan would cost at
    least ``tuning.automorphism_minimum_layer_work`` candidate/set evaluations.  Either way, setup's
    cache builds the family at most once, and the search switches to it if it predicts less work
    (see ``_switch_to_automorphism_family``).

    A layer of two or more rows that scans one set whose block is an information set uses
    ``affine_search.scan_information_set``.  Such a layer may skip the supports whose last two rows
    cannot complete a word lighter than the best weight, with tables that the completion fields of
    ``tuning`` choose (see ``completion_bounds``).  ``completion_tables`` keeps the most recently
    chosen tables of such a layer for reuse, until the basis changes or the search finishes.  A
    layer of two or more rows that scans several sets, all of whose blocks are information sets,
    scans them one at a time in the same way if tables pay for some of them and for the whole layer
    (see ``_choose_tables_of_sets``), and otherwise scans them in lockstep.
    """

    instance: affine_instance.AffineInstance
    cutoff: int
    batch_size: int
    tuning: SearchTuning
    best: int
    lower_bound: int
    next_weight: int = 1
    get_automorphism_family: _GetAutomorphismFamily | None = field(default=None, repr=False)
    uses_automorphism_family: bool = field(default=False, init=False)
    completion_tables: completion_bounds.CompletionTables | None = field(
        default=None, init=False, repr=False
    )

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

    def update_upper_bound(self, upper_bound: int) -> None:
        """Import a witness found by another search, and free the tables of a finished search."""
        self.best = min(self.best, upper_bound)
        if self.finished:
            self.completion_tables = None

    def advance(self) -> int:
        """Search the next coefficient layer, and return the best shared upper bound.

        The search scans at most the fewest leading sets whose completion of the layer certifies the
        best weight at the start of the layer, or every set if no fewer do (see
        ``Certificate.get_num_sets_to_certify``).  A search that uses an automorphism family scans
        its one set, which completes the layer for the whole family.
        """
        if self.finished:
            return self.best

        # Scan only the first sets of this layer if their completion of it certifies the best
        # weight: an unvisited eligible word lighter than that would contradict the certificate, and
        # lighter witnesses found during the scan only lower the best weight further.
        weight = self.next_weight
        instance = self.instance
        certificate = instance.certificate
        num_sets = len(certificate.ranks)
        if self.uses_automorphism_family:
            num_scanned, num_completed = 1, num_sets
        else:
            num_completed = certificate.get_num_sets_to_certify(weight, self.best)
            num_scanned = num_completed

        # Before a costly layer, consider scanning one set and certifying with its images.  Like
        # the work estimates of certificates, this estimate saturates at their cap, so that the
        # largest threshold of a tuning forbids the switch.
        layer_work = min(num_scanned * math.comb(self.dimension, weight), certificates.WORK_CAP)
        if (
            self.get_automorphism_family is not None
            and layer_work >= self.tuning.automorphism_minimum_layer_work
        ):
            self._switch_to_automorphism_family()
            return self.advance()

        # Scan one information set alone, several one at a time if completion tables pay for some
        # of them and for the whole layer, or else every scanned set in lockstep.  A scan of several
        # sets one at a time may stop after fewer sets, once they certify the best weight.
        scans_information_sets = weight >= 2 and bool(
            np.all(instance.ranks[:num_scanned] == self.dimension)
        )
        block_bits = (
            self._choose_tables_of_sets(weight, num_scanned)
            if scans_information_sets and num_scanned > 1
            else None
        )
        if scans_information_sets and num_scanned == 1:
            tables = self._select_completion_tables(weight)
            if not self._scan_information_set(
                0, weight, completion_bounds.NO_TABLES if tables is None else tables
            ):
                return self.best
        elif block_bits is None:
            if not self._scan_sets_in_lockstep(weight, num_scanned):
                return self.best
        else:
            num_completed_in_turn = self._scan_sets_in_turn(weight, block_bits)
            if num_completed_in_turn is None:
                return self.best
            num_completed = num_completed_in_turn

        # The first ``num_completed`` sets have completed the layer, so the family certificate may
        # advance.  The bound of a family scanned before a switch to an automorphism family still
        # holds for the eligible words lighter than the best weight, which no layer has examined or
        # skipped.
        self.lower_bound = max(
            self.lower_bound, certificate.step_lower_bound(weight, num_completed)
        )
        if num_completed == num_sets:
            self.next_weight += 1
        return self.best

    @property
    def _scan_cutoff(self) -> int:
        """The cutoff that the compiled scans take.

        Any negative Python cutoff disables early exit, so it is passed as -1, within int64 range.
        """
        return max(-1, self.cutoff)

    def _scan_information_set(
        self, index: int, weight: int, tables: completion_bounds.CompletionTables
    ) -> bool:
        """Scan a layer of the set ``index``, whose block is an information set.

        ``tables`` are the completion tables of the rows of the set, or
        ``completion_bounds.NO_TABLES``.  Returns False if the scan stopped at the cutoff.
        """
        instance = self.instance
        selected = slice(index, index + 1)
        support = np.zeros(weight, dtype=np.int64)
        prefixes = np.zeros((weight + 1, 1, instance.generators.shape[2]), dtype=np.uint64)

        # Scan in chunks, returning early once the upper bound reaches the cutoff.
        depth = 0
        while depth >= 0:
            self.best, depth, _, _, _ = affine_search.scan_information_set(
                instance.generators[selected],
                instance.labels[selected],
                instance.widths[selected],
                instance.label_widths[selected],
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

    def _scan_sets_in_lockstep(self, weight: int, num_scanned: int) -> bool:
        """Scan a layer of the first ``num_scanned`` sets in lockstep.

        Returns False if the scan stopped at the cutoff.
        """
        instance = self.instance
        scanned = slice(0, num_scanned)
        support = np.zeros(weight, dtype=np.int64)
        prefixes = np.zeros(
            (weight + 1, num_scanned, instance.generators.shape[2]), dtype=np.uint64
        )

        # Scan in chunks, returning early once the upper bound reaches the cutoff.
        depth = 0
        while depth >= 0:
            self.best, depth, _ = affine_search.scan_family(
                instance.generators[scanned],
                instance.labels[scanned],
                instance.ranks[scanned],
                instance.widths[scanned],
                instance.label_widths[scanned],
                support,
                prefixes,
                depth,
                self.best,
                self._scan_cutoff,
                self.batch_size,
            )
            if self.best <= self.cutoff:
                return False
        return True

    def _choose_tables_of_sets(self, weight: int, num_scanned: int) -> list[int] | None:
        """Choose whether to scan a layer of several information sets one at a time, with tables.

        A set gets completion tables if their predicted cost is within the limit of the tuning (see
        ``completion_bounds.predict_completion_cost``).  The layer scans its sets one at a time if
        some set gets tables, and if the predicted cost of the layer is within that limit too,
        counting the full cost for each set without tables.  Scanning the sets one at a time
        without tables costs about as much as scanning them in lockstep.

        Returns:
            The width of the blocks of the tables of each set, or zero for a set without tables, or
            None if the layer scans its sets in lockstep.
        """
        costs, block_bits = [], []
        for index in range(num_scanned):
            cost, bits = completion_bounds.predict_completion_cost(
                self.instance.generators[index, :, : self.instance.widths[index]],
                weight,
                self.best,
                reuse=False,
                tuning=self.tuning,
            )
            if not completion_bounds.is_within_cost_limit(cost, self.tuning):
                cost, bits = 1.0, 0
            costs.append(cost)
            block_bits.append(bits)
        if any(block_bits) and completion_bounds.is_within_cost_limit(
            sum(costs) / num_scanned, self.tuning
        ):
            return block_bits
        return None

    def _scan_sets_in_turn(self, weight: int, block_bits: list[int]) -> int | None:
        """Scan a layer of information sets one at a time, in the order of the family.

        Each set is scanned by ``_scan_information_set``, with tables on blocks of its entry of
        ``block_bits`` columns, built just before its scan, or without tables for an entry of zero.
        So the search holds the tables of one set at a time.  Each set scanned completes a step of
        the layer, so the scan stops after the first set whose step certifies the best weight
        (Section 3.9 of ``brouwer_zimmermann.md``).

        Returns:
            The number of leading sets that have completed the layer: those up to the set whose step
            certifies the best weight, or every set.  None if the scan stopped at the cutoff.
        """
        instance = self.instance
        for index, bits in enumerate(block_bits):
            rows = instance.generators[index, :, : instance.widths[index]]
            tables = (
                completion_bounds.CompletionTables.build(rows, bits)
                if bits
                else completion_bounds.NO_TABLES
            )
            if not self._scan_information_set(index, weight, tables):
                return None
            del tables

            # Stop once the sets that have completed the layer certify the best weight.
            # ``certifies`` solves a linear program only if the program could certify that weight.
            if instance.certificate.certifies(weight, index + 1, self.best):
                return index + 1
        return len(block_bits)

    def _select_completion_tables(self, weight: int) -> completion_bounds.CompletionTables | None:
        """Choose the completion tables of a layer of one information set, or None.

        The search reuses its tables while it scans the same set, and keeps the chosen tables.
        """
        tables = completion_bounds.select_completion_tables(
            self.instance.generators[0, :, : self.instance.widths[0]],
            weight,
            self.best,
            current=self.completion_tables,
            tuning=self.tuning,
        )
        if tables is not None:
            self.completion_tables = tables
        return tables

    def _switch_to_automorphism_family(self) -> None:
        """Scan one set and certify with its images from now on, if that predicts less work.

        The search keeps its lower bound.  An eligible word lighter than the best weight is
        unvisited by both families, so the bounds of both hold for it.  The certificate of the
        automorphism family counts every layer of the scanned set, so the search scans that set
        again from layer two.
        """
        assert self.get_automorphism_family is not None
        get_automorphism_family, self.get_automorphism_family = self.get_automorphism_family, None
        automorphism_family = get_automorphism_family(self.instance.certificate.weight_divisor)
        if automorphism_family is None:
            return

        # The rows of the scanned set are the layer-one words of every image.
        basis_family, certificate = automorphism_family
        self.best = _get_basis_family_upper_bound(basis_family, self.best)
        work = automorphism_families.estimate_automorphism_family_work(certificate, self.best)
        if work >= self.instance.certificate.work(self.best):
            return
        self.instance = affine_instance.get_distance_instance(basis_family, certificate)
        self.lower_bound = max(self.lower_bound, certificate.lower_bound(1))
        self.next_weight = 2
        self.uses_automorphism_family = True
        # Tables of the old set bound nothing in the new one.
        self.completion_tables = None


def _prepare_brouwer_zimmermann_search(
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
) -> int | _BrouwerZimmermannSearch:
    """Prepare one BZ search, or return immediately when another route is cheaper.

    The basis family stages of ``basis_families`` run in order, with distance-specific witness,
    cutoff, certificate, and exhaustive exits between them.  ``tuning`` sets the thresholds of the
    stages and exits.  ``get_disjoint_basis_family``, if given, returns the result of the first
    stage.  ``get_automorphisms``, if given, lets the search switch to an automorphism family before
    a costly layer (see ``_BrouwerZimmermannSearch``).  If the search would partition, setup builds
    that family first, skips matroid partitioning if that family predicts no more work than any
    partitioned family could, and has the search consider switching to it at once.
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

    if tuning.allow_enumeration_by_size and enumeration_is_cheaper_than_setup(
        dimension,
        basis.shape[1],
        tuning.enumeration_max_dimension,
        tuning.enumeration_setup_cost_factor,
    ):
        distance = get_nested_distance_brute_force(
            basis,
            labels,
            cutoff=cutoff,
            batch_exponent=tuning.enumeration_batch_exponent,
            symplectic=False,
        )
        return min(best, distance)

    # The disjoint basis family's certificate also uses any inferred even ambient parity.
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
    # Each basis family of an independent basis has a set, whose systematic rows complete layer one.
    completed_weight = 1
    lower_bound = basis_family.certificate.lower_bound(completed_weight)
    if lower_bound >= best:
        return best

    # Build the automorphism family at most once: before matroid partitioning, to decide whether to
    # partition, or before the first costly layer of the search.  Layer work saturates at the cap of
    # work estimates, so a threshold above that cap turns automorphisms off, and needs no family.
    get_automorphism_family = (
        None
        if get_automorphisms is None
        or tuning.automorphism_minimum_layer_work > certificates.WORK_CAP
        else functools.cache(
            functools.partial(_get_automorphism_family, basis, labels, get_automorphisms)
        )
    )

    # Replace the basis family only when overlapping sets, built heuristically and then, for costly
    # searches, by matroid partitioning, predict less certification work.  Partitioning is skipped
    # if an automorphism family does no more work than any partitioned family could, since the
    # search then switches to that family at once.
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

        # The replacement certificate uses its own ranks and coordinate multiplicity.
        lower_bound = basis_family.certificate.lower_bound(completed_weight)
        if lower_bound >= best:
            return best

    if tuning.allow_enumeration_by_work and _enumeration_is_cheaper_than_search(
        num_logical_rows=int(np.count_nonzero(get_eligible_rows(labels))),
        certificate=basis_family.certificate,
        upper_bound=best,
    ):
        distance = get_nested_distance_brute_force(
            basis,
            labels,
            cutoff=cutoff,
            batch_exponent=tuning.enumeration_batch_exponent,
            symplectic=False,
        )
        return min(best, distance)

    search = _BrouwerZimmermannSearch(
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
        get_automorphism_family=get_automorphism_family,
    )

    # A search whose setup found automorphisms, to decide whether to partition, has paid for them,
    # so it considers switching to their family at once rather than before a costly layer.  The rows
    # of that family can lower the best weight, and its certificate can finish the search.
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
) -> tuple[basis_families.BasisFamily, certificates.Certificate] | None:
    """Verify the automorphisms of a problem, and build the automorphism family of one set.

    Args:
        basis: An independent basis of the searched space.
        labels: The quotient labels of ``basis``.
        get_automorphisms: A function that returns generators of a group of column permutations
            that preserve the problem, or None.
        weight_divisor: A proven divisor of the Hamming weight of every eligible word.

    Returns:
        The family and certificate of ``automorphism_families.get_automorphism_family``, or None if
        ``get_automorphisms`` returns None.

    Raises:
        ValueError: If a generator is not an automorphism of the problem.
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

    Returns ``WORK_CAP`` if the problem has no automorphism family.
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
    # get row weights, summing in uint64 to avoid overflow in wide rows
    return int(np.bitwise_count(packed_basis).sum(axis=-1, dtype=np.uint64).min())


def _get_basis_family_upper_bound(
    basis_family: basis_families.BasisFamily,
    upper_bound: int,
) -> int:
    """Tighten an upper bound with the systematic rows of a basis family: its layer-one words."""
    for generators, set_labels in zip(basis_family.generators, basis_family.labels, strict=True):
        upper_bound = min(upper_bound, _get_lightest_eligible_row_weight(generators, set_labels))
    return upper_bound
