# SPDX-License-Identifier: Apache-2.0

"""The parts of ``method="auto"`` that compare the costs of exact distance searches.

``DistanceMethod`` in ``distances.py`` states the rule by which ``"auto"`` chooses the method of
each search.  ``problems.md#sec-choosing-a-method`` explains the rule, why it keeps results exact,
and its estimates.  This module holds the test of whether brute force is cheap,
``is_cheap_to_enumerate``, and the search by cost, ``search_by_cost``: a connected-cluster search
that switches to a Brouwer-Zimmermann (BZ) search once that is predicted to cost less.  The rule
decides only from results that are the same on any number of threads, since both searches can run
on several threads (``brouwer_zimmermann_code.md#sec-requirements`` and
``connected_cluster.md#sec-requirements``).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_num_words

from ._brouwer_zimmermann.distance import (
    BrouwerZimmermannSearch,
    get_distance_brouwer_zimmermann,
)
from ._connected_cluster.search import ConnectedClusterSearch
from ._tuning import MAX_VALUE, SearchTuning

####################################################################################################
# brute force


def is_cheap_to_enumerate(dimensions: Sequence[int], width: int, tuning: SearchTuning) -> bool:
    """Whether brute force examines at most ``auto_brute_force_max_words`` packed 64-bit integers.

    Brute force enumerates the ``2**dimension`` words of each space of the given dimensions, and
    packs each word of ``width`` bits into ``ceil(width / 64)`` integers of 64 bits.
    """
    num_words = sum(1 << dimension for dimension in dimensions)
    return num_words * get_num_words(width) <= tuning.auto_brute_force_max_words


####################################################################################################
# searches by cost


@dataclasses.dataclass(frozen=True)
class BrouwerZimmermannProblem:
    """The sectors of a distance search, as a BZ search takes them.

    Attributes:
        sectors: Nested-code bases and their quotient labels, as
            ``get_distance_brouwer_zimmermann`` takes them, one for each sector.
        weight_scale: The factor that converts the weights of the searched words into those of
            their BZ words: two for Pauli strings, whose BZ words are their images under
            ``symplectic_to_hamming``, and one otherwise.
        weight_divisor: A proven divisor of the weights of BZ words.  Certificates round up to its
            multiples (``brouwer_zimmermann.md#sec-weight-multiples``).  It equals ``weight_scale``
            for the problems that ``distances.py`` builds.
        get_automorphisms: A function that returns automorphisms of the coordinates of the BZ
            words, or None, as ``get_distance_brouwer_zimmermann`` takes it.
    """

    sectors: Sequence[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]]
    weight_scale: int = 1
    weight_divisor: int = 1
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None = None


def search_by_cost(
    cluster_sectors: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
    brouwer_zimmermann_problem: BrouwerZimmermannProblem,
    *,
    symplectic: bool,
    cutoff: int,
    upper_bound: int | None,
    get_automorphisms: Callable[[], npt.NDArray[np.int_] | None] | None,
    tuning: SearchTuning,
) -> int:
    """The minimum distance across sectors, by a connected-cluster search that may switch to BZ.

    The function runs a connected-cluster search, probes with a BZ search, and may switch to one, as
    ``problems.md#sec-choosing-a-method`` describes.  The sectors and other arguments, and the
    result, are those of ``get_distance_connected_cluster``.  ``brouwer_zimmermann_problem`` holds
    the same sectors, in the same order, for a BZ search.

    Every sector must have a witness, a nontrivial word, so that the searches of
    ``ConnectedClusterSearch`` line up one to one with the sectors of
    ``brouwer_zimmermann_problem``.  ``ConnectedClusterSearch`` keeps a search only for a sector
    with witnesses.
    """
    clusters = ConnectedClusterSearch(
        cluster_sectors,
        cutoff=cutoff,
        upper_bound=upper_bound,
        symplectic=symplectic,
        get_automorphisms=get_automorphisms,
        tuning=tuning,
    )
    assert len(clusters.searches) == len(brouwer_zimmermann_problem.sectors)
    costs = _BrouwerZimmermannCosts(brouwer_zimmermann_problem)
    state_cost = tuning.connected_cluster_state_evaluations
    scale = brouwer_zimmermann_problem.weight_scale
    probe_bound: int | None = None
    while not clusters.finished:
        # Decide only once the expected work of the next limit exceeds the free work.
        index, max_weight = clusters.next_limit
        spent = clusters.num_states * state_cost
        if spent + clusters.predict_states(index, max_weight) * state_cost > tuning.auto_free_work:
            # Probe once, and let the probe's bound end the connected-cluster search once every
            # sector's lower bound reaches it.
            if probe_bound is None:
                probe = _probe(brouwer_zimmermann_problem, cutoff, upper_bound, tuning)
                if probe.finished:
                    return probe.best // scale
                probe_bound = probe.best // scale
                clusters.update_upper_bound(probe_bound)
                continue

            # Switch to a BZ search with the full tuning, which starts again, capped by the probe's
            # bound.
            if _should_switch(clusters, costs, probe_bound, state_cost):
                return (
                    get_distance_brouwer_zimmermann(
                        brouwer_zimmermann_problem.sectors,
                        cutoff=cutoff * scale,
                        weight_divisor=brouwer_zimmermann_problem.weight_divisor,
                        upper_bound=probe_bound * scale,
                        get_automorphisms=brouwer_zimmermann_problem.get_automorphisms,
                        tuning=tuning,
                    )
                    // scale
                )
        clusters.advance()
    return clusters.result


class _BrouwerZimmermannCosts:
    """Estimates of the work of a BZ search, from the shapes of its bases alone.

    A family of information sets of a basis of rank ``r`` and width ``n`` that covers the columns
    evenly certifies the weight ``ceil((t + 1) n / r)`` after layer ``t``, rounded up to the weight
    divisor (``brouwer_zimmermann.md#sec-automorphism-certificate`` and
    ``brouwer_zimmermann.md#sec-weight-multiples``), and each of its search bases scans ``C(r, t)``
    supports in layer ``t``.  The estimates count the supports of one search basis per layer, as for
    an automorphism family, and leave out completion tables.  ``problems.md#sec-choosing-a-method``
    discusses the errors of these estimates.
    """

    def __init__(self, problem: BrouwerZimmermannProblem) -> None:
        self._shapes = [basis.shape for basis, _ in problem.sectors]
        self._scale = problem.weight_scale
        self._divisor = problem.weight_divisor

    def get_work(self, level: int) -> int:
        """The evaluations, over every sector, that certify that no witness weighs below ``level``.

        ``level`` is a weight of the sectors' words, before the scale.
        """
        work = 0
        for rank, width in self._shapes:
            num_layers = self._get_num_layers(rank, width, level)
            work += sum(math.comb(rank, weight) for weight in range(1, num_layers + 1))
        return work

    def get_growth(self, index: int, level: int) -> float:
        """The factor by which the work of sector ``index`` grows per unit of weight near ``level``.

        The layer ``t`` that certifies ``level`` is followed by layer ``t + 1``, which scans
        ``(r - t) / (t + 1)`` times as many supports, and raises the certified weight by about
        ``n / r`` in the weights of BZ words, or ``n / (r * scale)`` in those of the sector.  The
        factor is zero if no layer certifies ``level``.
        """
        rank, width = self._shapes[index]
        num_layers = self._get_num_layers(rank, width, level)
        if num_layers >= rank:
            return 0.0
        return ((rank - num_layers) / (num_layers + 1)) ** (rank * self._scale / width)

    def _get_num_layers(self, rank: int, width: int, level: int) -> int:
        """The number of layers, from layer 1, that a search scans to certify ``level``.

        They run up to the first whose certified weight reaches ``level``, or to ``rank`` if none
        does.
        """
        target = level * self._scale
        for layer in range(1, rank):
            bound = -(-(layer + 1) * width // rank)
            if bound + -bound % self._divisor >= target:
                return layer
        return rank


def _probe(
    problem: BrouwerZimmermannProblem,
    cutoff: int,
    upper_bound: int | None,
    tuning: SearchTuning,
) -> BrouwerZimmermannSearch:
    """A BZ search of the sectors, advanced by whole layers within ``auto_probe_work``.

    The probe skips the costly stages of setup, which build families of overlapping and of
    partitioned information sets.
    """
    scale = problem.weight_scale
    search = BrouwerZimmermannSearch(
        problem.sectors,
        cutoff=cutoff * scale,
        weight_divisor=problem.weight_divisor,
        upper_bound=None if upper_bound is None else upper_bound * scale,
        get_automorphisms=problem.get_automorphisms,
        tuning=dataclasses.replace(
            tuning, overlapping_minimum_work=MAX_VALUE, partition_work_ratio=MAX_VALUE
        ),
    )
    work = 0
    while not search.finished:
        layer_work = search.get_next_layer_work()
        if work + layer_work > tuning.auto_probe_work:
            break
        work += layer_work
        search.advance()
    return search


def _should_switch(
    clusters: ConnectedClusterSearch,
    costs: _BrouwerZimmermannCosts,
    probe_bound: int,
    state_cost: int,
) -> bool:
    """Whether the connected-cluster search is predicted to cost more than a BZ search.

    ``problems.md#sec-choosing-a-method`` explains the rule.  ``probe_bound`` must exceed the weight
    limit of the next advance (``clusters.next_limit``).  ``search_by_cost`` ensures this: it lowers
    the best weight of ``clusters`` to at most ``probe_bound``, and ``clusters`` finishes before its
    next limit reaches its best weight.
    """
    index, max_weight = clusters.next_limit
    assert probe_bound > max_weight
    growth = clusters.get_growth(index)
    if growth is None:
        return False
    level = max_weight + clusters.searches[index].weight_divisor
    spent = clusters.num_states * state_cost
    if spent + _project_states(clusters, level) * state_cost > costs.get_work(level):
        return True
    if growth <= costs.get_growth(index, level):
        return False
    projected = _project_states(clusters, probe_bound)
    projected += clusters.predict_states(index, probe_bound) / 2
    return spent + projected * state_cost > costs.get_work(probe_bound)


def _project_states(clusters: ConnectedClusterSearch, level: int) -> float:
    """The states that the searches are predicted to visit until each lower bound reaches ``level``.

    Each search runs its weight limits from its next one to the last below ``level``.
    """
    states = 0.0
    for index, search in enumerate(clusters.searches):
        first = clusters.get_next_limit(index)
        for max_weight in range(first, level, search.weight_divisor):
            states += clusters.predict_states(index, max_weight)
    return states
