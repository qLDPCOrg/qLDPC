# SPDX-License-Identifier: Apache-2.0

"""Exact binary BZ searches using disjoint or overlapping information sets.

The disjoint/rank-deficient certificates follow `Algorithm 994 <https://arxiv.org/abs/1603.06757>`_;
the symplectic reduction follows the `quantum-code adaptation <https://arxiv.org/abs/2408.10743>`_.
The overlapping-set certificate and implementation invariants are derived in
``brouwer_zimmermann.md`` beside this module.  Information-set families and their certificates are
built in ``_information_sets.py``, which fixed-weight logical counting shares; this module owns
witness discovery, early exits, exhaustive dispatch, and the resumable native scan.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from qldpc._compilation import always_compiled

from . import _information_sets
from ._binary import (
    as_binary_matrix,
    assert_binary,
    get_nested_code_basis,
    pack_bits,
    popcount_word,
)
from ._brute_force import (
    get_distance_quantum_brute_force,
)

####################################################################################################
# search entry points


def get_distance_css_brouwer_zimmermann(
    sectors: Sequence[tuple[npt.ArrayLike, npt.ArrayLike]],
    *,
    cutoff: int,
    block_size: int = 15,
    upper_bound: int | None = None,
) -> int:
    """Compute the minimum distance across homogeneous binary CSS sectors.

    Args:
        sectors: Logical operators and stabilizers for every sector to search.  All operators use
            homogeneous binary support vectors rather than symplectic vectors.
        cutoff: Exit early and return once an upper bound on distance falls to or below this cutoff.
        block_size: Batch size exponent, as in get_distance_quantum.
        upper_bound: Weight of a known logical operator from an omitted sector, if any.  The result
            is the minimum of this bound and the distances of the supplied sectors.

    Returns:
        The minimum exact distance across the supplied sectors and optional upper bound.  As with
        get_distance_quantum, a result at or below ``cutoff`` is an observed upper bound.
    """
    problems: list[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]] = []
    best = upper_bound
    for logical_ops, stabilizers in sectors:
        assert_binary(logical_ops, "logical_ops")
        assert_binary(stabilizers, "stabilizers")
        logical_matrix = as_binary_matrix(logical_ops)
        stabilizer_matrix = as_binary_matrix(stabilizers, num_cols=logical_matrix.shape[1])
        basis, labels = get_nested_code_basis(logical_matrix, stabilizer_matrix)
        if labels.shape[1]:
            problems.append((basis, labels))
            continue

        # A supplied but trivial quotient has distance zero; an empty sector contributes its width.
        sector_distance = 0 if len(logical_matrix) else logical_matrix.shape[1]
        best = sector_distance if best is None else min(best, sector_distance)

    if best is not None and best <= cutoff:
        return best
    if not problems:
        assert best is not None
        return best
    return get_distance_brouwer_zimmermann_many(
        problems,
        cutoff=cutoff,
        block_size=block_size,
        upper_bound=best,
    )


def get_distance_brouwer_zimmermann_many(
    problems: Sequence[tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]],
    *,
    cutoff: int,
    block_size: int,
    weight_divisor: int = 1,
    upper_bound: int | None = None,
    get_disjoint_basis_families: Sequence[Callable[[], _information_sets.BasisFamily]]
    | None = None,
) -> int:
    """Advance nested-code searches round-robin while sharing their best upper bound.

    ``get_disjoint_basis_families``, if given, holds a function for each problem that returns a
    disjoint basis family of it: sets with pairwise disjoint pivot blocks that span the problem's
    basis, with quotient labels of the problem, certified with this ``weight_divisor``.  The result
    of ``_information_sets.get_disjoint_family`` is one, and a caller that keeps such families saves
    that first stage.  Each function is called only if its search needs it.
    """
    initial_upper_bounds = [
        _get_brouwer_zimmermann_initial_upper_bound(basis, labels) for basis, labels in problems
    ]
    best = min(
        *initial_upper_bounds,
        math.inf if upper_bound is None else upper_bound,
    )
    searches: list[_BrouwerZimmermannSearch] = []
    for index, ((basis, labels), initial_upper_bound) in enumerate(
        zip(problems, initial_upper_bounds, strict=True)
    ):
        prepared = _prepare_brouwer_zimmermann_search(
            basis,
            labels,
            cutoff=cutoff,
            block_size=block_size,
            weight_divisor=weight_divisor,
            upper_bound=int(best),
            initial_upper_bound=initial_upper_bound,
            get_disjoint_basis_family=None
            if get_disjoint_basis_families is None
            else get_disjoint_basis_families[index],
        )
        if isinstance(prepared, int):
            best = min(best, prepared)
        else:
            searches.append(prepared)

    while searches:
        advanced = False
        for search in searches:
            # Share witnesses between sectors, but keep each lower certificate independent.
            search.update_upper_bound(int(best))
            if search.finished:
                continue
            best = min(best, search.advance())
            advanced = True
            if best <= cutoff:
                return int(best)
        if not advanced:
            break

    return int(best)


def get_distance_brouwer_zimmermann(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    *,
    cutoff: int,
    block_size: int,
    weight_divisor: int = 1,
    upper_bound: int | None = None,
) -> int:
    """Compute an exact nested-code distance with the Brouwer-Zimmermann algorithm.

    The disjoint and rank-deficient bound follows Algorithm 1 of https://arxiv.org/abs/1603.06757.
    The overlapping-set extension divides pivot occupancy by the maximum coordinate coverage.
    Nonzero ``labels`` identify rows outside an excluded subcode; transforming them alongside the
    generators makes the upper-bound search exact for logical operators modulo stabilizers.

    An ``upper_bound`` starts the search as though a witness of that weight were known, so the
    search returns ``min(distance, upper_bound)`` and stops once it certifies that no word is
    lighter than ``upper_bound``.
    """
    return get_distance_brouwer_zimmermann_many(
        [(basis, labels)],
        cutoff=cutoff,
        block_size=block_size,
        weight_divisor=weight_divisor,
        upper_bound=upper_bound,
    )


####################################################################################################
# cost models


def exhaustive_is_cheaper(dimension: int, length: int) -> bool:
    """Estimate whether enumerating every codeword is cheaper than building information sets.

    Setup costs dominate the Brouwer-Zimmermann search for dimensions up to 20.  For long,
    low-dimensional codes, byte-wise elimination for each information set can cost more than packed
    ``uint64`` enumeration of all ``2**dimension`` codewords.
    """
    return dimension <= 20 or (dimension < 63 and 1 << dimension <= 128 * dimension * length)


def _brute_force_is_cheaper(
    *,
    num_logical_rows: int,
    certificate: _information_sets.Certificate,
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

    ``certificate`` records the ranks, maximum coordinate coverage, and proven weight divisor of the
    packed family.  ``ranks`` is its native copy for the compiled scan.
    """

    generators: npt.NDArray[np.uint64]
    labels: npt.NDArray[np.uint64]
    widths: npt.NDArray[np.int64]
    label_widths: npt.NDArray[np.int64]
    certificate: _information_sets.Certificate
    cutoff: int
    batch_size: int
    best: int
    lower_bound: int
    next_weight: int = 1
    ranks: npt.NDArray[np.int64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.ranks = np.array(self.certificate.ranks, dtype=np.int64)
        assert len(self.ranks) == len(self.generators)
        assert self.certificate.dimension == self.dimension

    @property
    def dimension(self) -> int:
        """Number of basis rows searched."""
        return self.generators.shape[1]

    @property
    def finished(self) -> bool:
        """Whether this sector cannot improve the shared upper bound."""
        return (
            self.best <= self.cutoff
            or self.lower_bound >= self.best
            or self.next_weight > self.dimension
        )

    def update_upper_bound(self, upper_bound: int) -> None:
        """Import a witness found by another search."""
        self.best = min(self.best, upper_bound)

    def advance(self) -> int:
        """Search one coefficient weight and return the best shared upper bound."""
        if self.finished:
            return self.best

        # Start a fresh enumeration of all supports with this coefficient weight.
        weight = self.next_weight
        support = np.zeros(weight, dtype=np.int64)
        prefixes = np.zeros(
            (weight + 1, len(self.ranks), self.generators.shape[2]), dtype=np.uint64
        )
        depth = 0
        # Any negative Python cutoff disables early exit; keep its native value in int64 range.
        cutoff = max(-1, self.cutoff)

        # Scan in chunks, returning early once the upper bound reaches the cutoff.
        while depth >= 0:
            self.best, depth, _ = _scan_brouwer_zimmermann(
                self.generators,
                self.labels,
                self.ranks,
                self.widths,
                self.label_widths,
                support,
                prefixes,
                depth,
                self.best,
                cutoff,
                self.batch_size,
            )
            if self.best <= self.cutoff:
                return self.best

        # The layer is complete in every search basis, so the family certificate may advance.
        self.lower_bound = self.certificate.lower_bound(weight)
        self.next_weight += 1
        return self.best


def _prepare_brouwer_zimmermann_search(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
    *,
    cutoff: int,
    block_size: int,
    weight_divisor: int,
    upper_bound: int,
    initial_upper_bound: int | None = None,
    get_disjoint_basis_family: Callable[[], _information_sets.BasisFamily] | None = None,
) -> int | _BrouwerZimmermannSearch:
    """Prepare one BZ search, or return immediately when another route is cheaper.

    The basis family stages of ``_information_sets`` run in order, with distance-specific witness,
    cutoff, certificate, and exhaustive exits between them.  ``get_disjoint_basis_family``, if
    given, returns the result of the first stage.
    """
    dimension = len(basis)
    eligible = np.ones(dimension, dtype=bool) if labels is None else np.any(labels, axis=1)

    def brute_force_distance() -> int:
        distance = get_distance_quantum_brute_force(
            basis[eligible],
            basis[~eligible],
            cutoff=cutoff,
            block_size=block_size,
            homogeneous=True,
        )
        return min(best, distance)

    best = min(
        upper_bound,
        (
            _get_brouwer_zimmermann_initial_upper_bound(basis, labels)
            if initial_upper_bound is None
            else initial_upper_bound
        ),
    )
    if best <= cutoff:
        return best

    if exhaustive_is_cheaper(dimension, basis.shape[1]):
        return brute_force_distance()

    # The disjoint basis family's certificate also uses any inferred even ambient parity.
    basis_family = (
        _information_sets.get_disjoint_family(basis, labels, weight_divisor=weight_divisor)
        if get_disjoint_basis_family is None
        else get_disjoint_basis_family()
    )
    best = _get_basis_family_upper_bound(basis_family, labels, best)
    if best <= cutoff:
        return best

    basis_family = _information_sets.prune_family(basis_family, upper_bound=best)
    # Each basis family of an independent basis has a set, whose systematic rows complete layer one.
    completed_weight = 1
    lower_bound = basis_family.certificate.lower_bound(completed_weight)
    if lower_bound >= best:
        return best

    # Replace disjoint blocks only when overlap predicts less certification work.
    overlapping = _information_sets.get_overlapping_family(
        basis,
        np.empty((dimension, 0), dtype=np.uint8) if labels is None else labels,
        basis_family,
        best,
        _information_sets.MAX_SETS,
        *_information_sets.get_overlapping_family_policy(dimension),
    )
    if len(overlapping.ranks):
        basis_family = overlapping

        # Check the replacement rows for lighter witnesses before scanning combinations.
        best = _get_basis_family_upper_bound(basis_family, labels, best)
        if best <= cutoff:
            return best

        # The replacement certificate uses its own ranks and coordinate multiplicity.
        lower_bound = basis_family.certificate.lower_bound(completed_weight)
        if lower_bound >= best:
            return best

    if _brute_force_is_cheaper(
        num_logical_rows=int(np.count_nonzero(eligible)),
        certificate=basis_family.certificate,
        upper_bound=best,
    ):
        return brute_force_distance()

    # Remove systematic pivots; advance() restores their known weight arithmetically.
    packed_rows = [
        pack_bits(np.delete(generators, pivots[:rank], axis=1))
        for generators, pivots, rank in zip(
            basis_family.generators, basis_family.pivots, basis_family.ranks, strict=True
        )
    ]

    # Zero-pad every set to a common word width, so the Numba kernel receives single arrays.  The
    # labels of every set already have the same width, which is zero for a basis without labels.
    widths = np.array([rows.shape[1] for rows in packed_rows], dtype=np.int64)
    packed_generators = np.zeros(
        (len(packed_rows), dimension, int(widths.max(initial=0))), dtype=np.uint64
    )
    for index, rows in enumerate(packed_rows):
        packed_generators[index, :, : widths[index]] = rows
    packed_labels = np.stack([pack_bits(set_labels) for set_labels in basis_family.labels])
    label_widths = np.full(len(packed_labels), packed_labels.shape[2], dtype=np.int64)

    return _BrouwerZimmermannSearch(
        packed_generators,
        packed_labels,
        widths,
        label_widths,
        basis_family.certificate,
        cutoff,
        1 << min(block_size, 15),
        best,
        lower_bound,
        completed_weight + 1,
    )


def _get_brouwer_zimmermann_initial_upper_bound(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8] | None,
) -> int:
    """Return the lightest eligible row in a nested-code basis."""
    eligible = np.ones(len(basis), dtype=bool) if labels is None else np.any(labels, axis=1)
    packed_basis = pack_bits(basis[eligible])
    # get row weights, summing in uint64 to avoid overflow in wide rows
    return int(np.bitwise_count(packed_basis).sum(axis=-1, dtype=np.uint64).min())


def _get_basis_family_upper_bound(
    basis_family: _information_sets.BasisFamily,
    labels: npt.NDArray[np.uint8] | None,
    upper_bound: int,
) -> int:
    """Tighten an upper bound with the systematic rows of a basis family, all layer-one candidates.

    ``labels`` are those of the searched basis, or None if every nonzero word is eligible.
    """
    for generators, set_labels in zip(basis_family.generators, basis_family.labels, strict=True):
        upper_bound = min(
            upper_bound,
            _get_brouwer_zimmermann_initial_upper_bound(
                generators, None if labels is None else set_labels
            ),
        )
    return upper_bound


####################################################################################################
# native scan


@always_compiled
def _scan_brouwer_zimmermann(
    generators: npt.NDArray[np.uint64],
    labels: npt.NDArray[np.uint64],
    ranks: npt.NDArray[np.int64],
    widths: npt.NDArray[np.int64],
    label_widths: npt.NDArray[np.int64],
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    depth: int,
    best: int,
    cutoff: int,
    batch_size: int,
) -> tuple[int, int, int]:
    """Scan one bounded chunk of coefficient supports for Brouwer-Zimmermann search.

    Args:
        generators: Packed information-set rows after pivot columns have been deleted.
        labels: Packed quotient labels whose nonzero XOR marks eligible candidates.
        ranks: Rank of each information-set block before pivot columns were deleted.
        widths: Number of packed generator words used by each information-set block.
        label_widths: Number of packed label words used by each information-set block.
        support: Mutable row indices for the current coefficient support.
        prefixes: Mutable prefix XORs for internal support depths and information-set blocks.
        depth: Current support-tree depth, or ``-1`` when the coefficient layer is complete.
        best: Current upper bound; only strictly lighter eligible candidates replace it.
        cutoff: Weight threshold that stops this chunk once ``best <= cutoff``.
        batch_size: Maximum number of complete supports visited by this chunk.

    Returns:
        Updated ``best``, updated ``depth``, and number of complete supports visited.
    """
    dimension = generators.shape[1]
    weight = len(support)
    visited = 0
    while depth >= 0:
        # Backtrack once the row at this depth leaves too few rows to complete the support.
        if support[depth] > dimension - weight + depth:
            depth -= 1
            if depth >= 0:
                support[depth] += 1
            continue

        row = support[depth]
        if depth + 1 < weight:
            for set_index in range(len(ranks)):
                for word in range(widths[set_index]):
                    prefixes[depth + 1, set_index, word] = (
                        prefixes[depth, set_index, word] ^ generators[set_index, row, word]
                    )
            depth += 1
            support[depth] = row + 1
            continue

        # Evaluate the completed support in each information set.
        for set_index in range(len(ranks)):
            # Each selected row whose pivot column was deleted contributes one to the Hamming
            # weight; packed words hold the remaining columns.
            candidate_weight = np.int64(weight) if ranks[set_index] == dimension else np.int64(0)
            if ranks[set_index] != dimension:
                for selected in range(weight):
                    if support[selected] < ranks[set_index]:
                        candidate_weight += 1
            for word in range(widths[set_index]):
                value = prefixes[depth, set_index, word] ^ generators[set_index, row, word]
                candidate_weight += popcount_word(value)
                if candidate_weight >= best:
                    break
            if candidate_weight >= best:
                continue

            # A candidate is eligible when no subcode is excluded or its quotient label is nonzero.
            eligible = label_widths[set_index] == 0
            for word in range(label_widths[set_index]):
                combined = labels[set_index, support[0], word]
                for selected in range(1, weight):
                    combined ^= labels[set_index, support[selected], word]
                if combined:
                    eligible = True
                    break
            if eligible:
                best = int(candidate_weight)

        # Move to the next support, returning at the cutoff or at the end of this chunk.
        visited += 1
        support[depth] += 1
        if best <= cutoff or visited >= batch_size:
            return best, depth, visited
    return best, depth, visited
