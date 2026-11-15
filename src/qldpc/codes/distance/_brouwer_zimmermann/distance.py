# SPDX-License-Identifier: Apache-2.0

"""Exact BZ searches with disjoint, overlapping, or automorphism families of information sets.

The certificates of disjoint and rank-deficient information sets follow
`Algorithm 994 <https://arxiv.org/abs/1603.06757>`_, and the symplectic reduction follows an
`adaptation of the algorithm to quantum codes <https://arxiv.org/abs/2408.10743>`_.
``brouwer_zimmermann.md`` in ``src/qldpc/codes/distance/`` derives the certificate of overlapping
sets and the invariants of this implementation.

This module prepares each search, finds its witnesses, takes its early exits, and scans its layers
in resumable chunks.  The families and their certificates come from ``_information_sets/``, and
the compiled scans from ``affine_search.py``.  The package's ``README.md`` says where everything
else lives.
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
from .._information_sets import (
    automorphism_families,
    basis_families,
    certificates,
    completion_bounds,
)
from .._tuning import SearchTuning, get_tuning
from . import affine_instance, affine_search

# The automorphism family of a problem: a basis family of the one set that a search scans, and the
# certificate of all of its images.
_AutomorphismFamily = tuple[basis_families.BasisFamily, certificates.Certificate]

# A function that returns the automorphism family of a problem for a weight divisor, or None.
_GetAutomorphismFamily = Callable[[int], _AutomorphismFamily | None]

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
            disjoint basis family of it, such as the result of
            ``basis_families.get_disjoint_family``.  Such a family holds sets with pairwise disjoint
            pivot blocks that span the problem's basis, with the quotient labels of the problem,
            and is certified with this ``weight_divisor``.  A caller that keeps these families
            saves the searches that first stage.  Each function is called only if its search needs
            it.
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

    The search certifies that no eligible word weighs less than ``min(lower_bound, best)``, and it
    has finished once ``lower_bound`` reaches ``best``.  ``lower_bound`` bounds only the eligible
    words lighter than ``best``.  The search can neither have examined nor skipped such a word, so
    the bounds of every family that it has scanned hold for it.  Heavier words, such as those of
    supports that the search skipped, can weigh less than ``lower_bound``.

    ``get_automorphism_family``, if given, returns an automorphism family of the problem, or None.
    The search calls it at most once, the first time that it considers switching to that family,
    and then sets it to None.  If the search would partition, setup calls it first, and the search
    considers the switch at the end of setup.  Otherwise the search first considers the switch
    before the first layer whose scan would cost at least ``tuning.automorphism_minimum_layer_work``
    candidate/set evaluations.  Either way, setup's cache builds the family at most once.  The
    search switches to the family if it predicts less work (see ``_switch_to_automorphism_family``).
    A search that declines keeps the family in ``automorphism_family``, and its best weight at that
    time in ``declined_best``.  It considers the switch again before a later costly layer once its
    best weight has dropped below ``declined_best`` (see ``_considers_automorphism_family``).

    A layer scans its sets one at a time, in the order of the family, with
    ``affine_search.scan_set``.  The scan of a set whose block is an information set may use
    completion tables, which the completion fields of ``tuning`` choose for it (see
    ``completion_bounds``).  The tables let the scan skip supports whose words cannot be lighter
    than the best weight.  ``completion_tables`` keeps the most recently chosen tables, of the set
    ``completion_tables_set``, for reuse in later layers of that set.  The search keeps them until
    it chooses tables for another set, the basis changes, or the search finishes.
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

    def update_upper_bound(self, upper_bound: int) -> None:
        """Import a witness found by another search, and free the tables of a finished search."""
        self.best = min(self.best, upper_bound)
        if self.finished:
            self.completion_tables = None

    def advance(self) -> int:
        """Search the next coefficient layer, and return the best shared upper bound.

        It scans at most the fewest leading sets whose completion of the layer certifies the best
        weight at the start of the layer, or every set if no fewer do (see
        ``Certificate.get_num_sets_to_certify``).  If the layer finds a lighter word, the search
        stops once the sets that it has scanned certify the new best weight.  A search that uses an
        automorphism family scans its one set, which completes the layer for the whole family.
        """
        if self.finished:
            return self.best

        # Scan only as many leading sets as certify the best weight.  An unvisited eligible word
        # lighter than that weight would contradict the certificate, and the witnesses that the scan
        # finds only lower the best weight further.
        weight = self.next_weight
        instance = self.instance
        certificate = instance.certificate
        num_sets = len(certificate.ranks)
        if self.uses_automorphism_family:
            num_scanned, num_completed = 1, num_sets
        else:
            num_completed = certificate.get_num_sets_to_certify(weight, self.best)
            num_scanned = num_completed

        # Before a costly layer, consider scanning one set and certifying with its images.  This
        # estimate saturates at the cap of the work estimates of certificates, as they do, so that
        # the largest threshold of a tuning forbids the switch.
        layer_work = min(num_scanned * math.comb(self.dimension, weight), certificates.WORK_CAP)
        if (
            self._considers_automorphism_family
            and layer_work >= self.tuning.automorphism_minimum_layer_work
        ):
            self._switch_to_automorphism_family()
            return self.advance()

        # Scan the sets one at a time, in the order of the family.  Each set completes a step of the
        # layer, so the scan stops after the first set whose step certifies the best weight
        # (Sections 2.5 and 3.9 of ``brouwer_zimmermann.md``).  ``certifies`` solves a linear
        # program only if the program could certify that weight.  Each set steps through the
        # supports of the layer on its own, which keeps the scan simple but repeats the stepping, a
        # cost that shows on layers of many sets of short rows.
        for index in range(num_scanned):
            if not self._scan_set(index, weight):
                return self.best
            if index + 1 < num_scanned and certificate.certifies(weight, index + 1, self.best):
                num_completed = index + 1
                break

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
        """The cutoff that the compiled scans take: ``cutoff``, clamped below at -1.

        Every negative cutoff disables early exits alike, and -1 is within int64 range.
        """
        return max(-1, self.cutoff)

    def _scan_set(self, index: int, weight: int) -> bool:
        """Scan a layer of the set ``index``, with the completion tables that it chooses, if any.

        The block of the set need not be an information set.  Returns False if the scan stopped at
        the cutoff.
        """
        tables = self._select_completion_tables(index, weight)
        instance = self.instance
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
                instance.label_widths[index],
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

    def _select_completion_tables(
        self, index: int, weight: int
    ) -> completion_bounds.CompletionTables:
        """Choose the completion tables of a layer of the set ``index``.

        Only a set whose block is an information set can have tables.  It reuses the tables that the
        search keeps if they are its own.  Otherwise the search frees them before choosing, so that
        it holds the tables of one set at a time.  The search keeps the chosen tables.

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

        It would until it has switched or has found that the problem has no such family.  After it
        declines the switch, it would again only once its best weight has dropped.  At the same best
        weight it would decline again: the work of the automorphism family is the same, and the
        work left in the current family only falls as the search completes layers.  This property
        is False right after every consideration, which ends the recursion of ``advance`` after it
        considers the switch.
        """
        if self.get_automorphism_family is not None:
            return True
        return self.automorphism_family is not None and self.best < self.declined_best

    def _switch_to_automorphism_family(self) -> None:
        """Scan one set and certify with its images from now on, if that predicts less work.

        The first call gets the automorphism family, which the search keeps while it declines the
        switch.  The certificate of the automorphism family counts every layer of the scanned set,
        so after a switch the search scans that set again from layer two.  The switch abandons the
        layers that the current family has completed.  So it compares the work of the automorphism
        family with the work left in the current family, from its next layer.  The search keeps its
        lower bound: an eligible word lighter than the best weight is unvisited by both families, so
        the bounds of both hold for it.
        """
        if self.get_automorphism_family is not None:
            weight_divisor = self.instance.certificate.weight_divisor
            self.automorphism_family = self.get_automorphism_family(weight_divisor)
            self.get_automorphism_family = None
        if self.automorphism_family is None:
            return

        # The layer-one words of every image weigh as much as the rows of the scanned set.  The work
        # of the automorphism family counts them too, one evaluation each, so that it stays small
        # when they certify the best weight.
        basis_family, certificate = self.automorphism_family
        self.best = _get_basis_family_upper_bound(basis_family, self.best)
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
    """Prepare one BZ search, or return its result if setup already settles it.

    The basis family stages of ``basis_families`` run in order.  Between them, setup returns early
    if the rows of the family bring the best weight to the cutoff, or if its certificate already
    proves the best weight.  ``tuning`` sets the thresholds of the stages.
    ``get_disjoint_basis_family``, if given, returns the result of the first stage.
    ``get_automorphisms``, if given, lets the search switch to an automorphism family before a
    costly layer (see ``_BrouwerZimmermannSearch``).  If the search would partition, setup builds
    that family first.  It skips matroid partitioning if that family predicts no more work than any
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

    # The certificate of the disjoint basis family also uses any even parity of the words that the
    # family infers.
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
    # A family of an independent basis has at least one set, and the systematic rows of its sets,
    # which the upper bound above weighed, complete layer one.
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

    # Two stages may replace the basis family by overlapping sets: a heuristic, and then, for costly
    # searches, matroid partitioning.  Each replaces it only if its sets predict less certification
    # work.  Partitioning is skipped if an automorphism family does no more work than any
    # partitioned family could, since the search then switches to that family at once.
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

    # If setup found automorphisms to decide whether to partition, it has already paid for them.  So
    # the search considers switching to their family at once, rather than before a costly layer.
    # The rows of that family can lower the best weight, and its certificate can finish the search.
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
