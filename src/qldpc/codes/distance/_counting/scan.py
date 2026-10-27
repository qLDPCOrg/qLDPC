# SPDX-License-Identifier: Apache-2.0

"""The scan of the words of one weight in a sector: its set-up, its loop, and its native chunks.

A sector is a basis whose quotient labels mark its nontrivial words.  ``count_words`` selects
information sets for the basis, with the stages of ``_information_sets.py``, packs them, and visits
each word of the given weight once, by enumerating combinations of the rows of every set, layer by
layer.  ``logical_counts.py`` calls it.  Section 3 of ``brouwer_zimmermann.md`` describes the
algorithm, and Section 5 states the counting requirement.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from .. import _information_sets
from .._binary import pack_bits, popcount_word, row_reduce_binary
from .._compilation import always_compiled
from ..core import DistanceMethod
from . import local_cliffords

_BinaryMatrix = npt.NDArray[np.uint8]
_PackedMatrix = npt.NDArray[np.uint64]

# The basis of a homogeneous sector, and its quotient labels.
_Sector = tuple[_BinaryMatrix, _BinaryMatrix]

# Coefficient combinations visited per native call, which bounds the time between interruptions.
_CHUNK_SIZE = 1 << 16


####################################################################################################
# counting words of one weight


def count_words(
    basis: _BinaryMatrix,
    labels: _BinaryMatrix,
    weight: int,
    method: DistanceMethod,
    rank_rows: _PackedMatrix,
    *,
    complement: bool,
    max_exponent: int,
    reject_lighter: bool,
    get_disjoint: Callable[[], _information_sets.BasisFamily],
    max_sets: int,
) -> npt.NDArray[np.int64] | None:
    """Visit each nontrivial word of one sector with the given weight exactly once.

    Returns a histogram over exponents ``max_exponent - rank(rank_rows[:, mask])``, where the mask
    is the support of the word, or its complement.  The histogram sums to the number of words, and
    the caller weighs each exponent by ``2**exponent``, which could overflow a native integer.  The
    scan also visits every lighter word, so with ``reject_lighter`` it returns None as soon as it
    finds a nontrivial one.
    """
    # Select the information sets, or scan the whole basis as one set for brute force.
    certificate: _information_sets.Certificate | None
    if method == "brute_force":
        sets, certificate = [(basis, labels)], None
    else:
        sets, certificate = _get_counting_information_sets(
            basis, labels, weight, get_disjoint(), max_sets
        )

    # Pack the sets for the native scan.
    rows = [pack_bits(generators) for generators, _ in sets]
    tags = [pack_bits(set_labels) for _, set_labels in sets]
    # Ownership compares sets, so a single set needs no coefficient extractor.
    extractors = (
        np.stack([_get_coefficient_extractor(generators) for generators, _ in sets])
        if len(sets) > 1
        else np.zeros((1, 0, rows[0].shape[1]), dtype=np.uint64)
    )

    # Scan the sets layer by layer, until the certificate rules out unvisited words of the weight.
    histogram = np.zeros(max_exponent + 1, dtype=np.int64)
    for layer in range(1, len(basis) + 1):
        for index in range(len(sets)):
            support = np.zeros(layer, dtype=np.int64)
            depth = 0
            while depth >= 0:
                depth, lighter = _count_chunk(
                    rows[index],
                    tags[index],
                    extractors,
                    index,
                    support,
                    depth,
                    weight,
                    rank_rows,
                    complement,
                    max_exponent,
                    histogram,
                    reject_lighter,
                    _CHUNK_SIZE,
                )
                if lighter:
                    return None
        # Equality would still allow unvisited words of the target weight.
        if certificate is not None and certificate.lower_bound(layer) > weight:
            break
    return histogram


def transport_basis_family(
    basis_family: _information_sets.BasisFamily,
    positions: npt.NDArray[np.int64],
    local: npt.NDArray[np.uint64],
    basis: _BinaryMatrix,
) -> _information_sets.BasisFamily | None:
    """The family of a sector after a local gate, or None if one of its sets stops being a set.

    Each set is mapped through the gate and reduced again on its own pivot columns, so the pivots
    stay disjoint and the labels move with their rows.  A set stops being one if its pivot columns
    are no longer independent.
    """
    sets = []
    for information_set in basis_family.information_sets:
        generators = local_cliffords.apply_local(information_set.generators, positions, local)
        reduced, labels, pivots = row_reduce_binary(
            generators, information_set.labels, columns=information_set.pivots
        )
        if len(pivots) != information_set.rank:
            return None
        sets.append(_information_sets.InformationSet(reduced, labels, len(pivots), pivots))
    return _information_sets.BasisFamily.from_information_sets(
        sets,
        dimension=basis_family.certificate.dimension,
        weight_divisor=_information_sets.infer_weight_divisor(basis),
    )


def _get_counting_information_sets(
    basis: _BinaryMatrix,
    labels: _BinaryMatrix,
    weight: int,
    disjoint: _information_sets.BasisFamily,
    max_sets: int,
) -> tuple[list[_Sector], _information_sets.Certificate]:
    """Select BZ information sets that certify every word of the given weight.

    Returns the sets and the certificate that bounds the weight of unvisited words after each layer.
    The family stages aim at the counting stop ``L(t) >= weight + 1``; any family gives an exact
    count.
    """
    basis_family = _information_sets.prepare_family(
        basis, labels, upper_bound=weight + 1, disjoint=disjoint, max_sets=max_sets
    )
    sets = []
    for information_set in basis_family.information_sets:
        assert information_set.labels is not None
        sets.append((information_set.generators, information_set.labels))
    return sets, basis_family.certificate


def _get_coefficient_extractor(generators: _BinaryMatrix) -> _PackedMatrix:
    """Packed rows ``E.T`` such that ``generators @ E`` is the identity over GF(2).

    Bit ``b`` of the coefficient vector of a word ``w`` is then the parity of ``w & E.T[b]``.
    """
    # Row reduction finds T with T @ generators = R, where R is the identity on its pivot columns P,
    # so that w[P] @ T recovers the coefficients of w.
    size = len(generators)
    _, transform, pivots = row_reduce_binary(generators, np.eye(size, dtype=np.uint8))
    assert transform is not None
    extractor = np.zeros((generators.shape[1], size), dtype=np.uint8)
    extractor[pivots] = transform
    return pack_bits(extractor.T)


####################################################################################################
# native kernels


@always_compiled
def _count_chunk(
    rows: _PackedMatrix,
    tags: _PackedMatrix,
    extractors: npt.NDArray[np.uint64],
    set_index: int,
    support: npt.NDArray[np.int64],
    depth: int,
    target: int,
    rank_rows: _PackedMatrix,
    complement: bool,
    max_exponent: int,
    histogram: npt.NDArray[np.int64],
    reject_lighter: bool,
    chunk_size: int,
) -> tuple[int, bool]:
    """Visit up to ``chunk_size`` combinations of ``len(support)`` rows, resuming at ``depth``.

    ``support`` holds the increasing row indices of the current combination; ``depth`` is the
    position being advanced, or -1 once every combination has been visited.  Each owned nontrivial
    word of weight ``target`` adds one to ``histogram`` (see ``count_words``).  Returns the depth
    from which to resume, and whether the scan stopped at a nontrivial word lighter than ``target``,
    which is checked only with ``reject_lighter``.
    """
    num_rows, width = rows.shape
    layer = len(support)
    # Prefixes are recomputed on entry, so only support and depth carry state between calls.
    prefixes = np.zeros((layer, width), dtype=np.uint64)
    tag_prefixes = np.zeros((layer, tags.shape[1]), dtype=np.uint64)
    for position in range(1, depth + 1):
        row = support[position - 1]
        for index in range(width):
            prefixes[position, index] = prefixes[position - 1, index] ^ rows[row, index]
        for index in range(tags.shape[1]):
            tag_prefixes[position, index] = tag_prefixes[position - 1, index] ^ tags[row, index]

    # Advance through the combinations until the chunk is used up.
    word = np.empty(width, dtype=np.uint64)
    pivots = np.empty((len(rank_rows), width), dtype=np.uint64)
    visited = 0
    while depth >= 0 and visited < chunk_size:
        # Backtrack when no room remains for the rows after this position.
        if support[depth] > num_rows - layer + depth:
            depth -= 1
            if depth >= 0:
                support[depth] += 1
            continue

        # Extend the prefix by the row at this position, and advance to the next position.
        if depth + 1 < layer:
            row = support[depth]
            for index in range(width):
                prefixes[depth + 1, index] = prefixes[depth, index] ^ rows[row, index]
            for index in range(tags.shape[1]):
                tag_prefixes[depth + 1, index] = tag_prefixes[depth, index] ^ tags[row, index]
            depth += 1
            support[depth] = row + 1
            continue

        # Visit the remaining combinations that differ only in their last row.
        stop = min(num_rows, support[depth] + chunk_size - visited)
        for row in range(support[depth], stop):
            weight = np.int64(0)
            for index in range(width):
                word[index] = prefixes[depth, index] ^ rows[row, index]
                weight += popcount_word(word[index])
                # A heavier word is rejected below, so its unwritten words are never read.
                if weight > target:
                    break
            if weight > target or (weight < target and not reject_lighter):
                continue

            # A nonzero quotient label marks a nontrivial logical rather than a stabilizer.
            nontrivial = False
            for index in range(tags.shape[1]):
                if tag_prefixes[depth, index] != tags[row, index]:
                    nontrivial = True
                    break
            if nontrivial and weight < target:
                return depth, True
            if nontrivial and _is_owned(word, extractors, set_index, layer):
                rank = _get_masked_rank(rank_rows, word, complement, pivots)
                histogram[max_exponent - rank] += 1

        # Record the visited rows, and resume after them.
        visited += stop - support[depth]
        support[depth] = stop
    return depth, False


@always_compiled
def _is_owned(
    word: npt.NDArray[np.uint64], extractors: npt.NDArray[np.uint64], set_index: int, layer: int
) -> bool:
    """Whether set ``set_index`` minimizes (layer, set index) among the sets that visit ``word``.

    Each set visits a word in the layer given by the weight of its coefficient vector in that set,
    so exactly one set owns each word, and it visits the word before any other set does.
    """
    for other in range(len(extractors)):
        if other == set_index:
            continue
        # A set that visits the word in an earlier layer, or in the same layer with a smaller index,
        # owns it.
        limit = layer if other < set_index else layer - 1
        size = 0
        for extractor_row in extractors[other]:
            parity = np.int64(0)
            for index in range(len(word)):
                parity += popcount_word(word[index] & extractor_row[index])
            size += int(parity & 1)
            if size > limit:
                break
        if size <= limit:
            return False
    return True


@always_compiled
def _get_masked_rank(
    rows: _PackedMatrix,
    mask: npt.NDArray[np.uint64],
    complement: bool,
    pivots: _PackedMatrix,
) -> int:
    """Rank of ``rows`` restricted to the support of ``mask``, or to its complement.

    ``pivots`` is scratch space with at least as many rows as ``rows``.
    """
    pivot_words = np.empty(len(rows), dtype=np.int64)
    pivot_bits = np.empty(len(rows), dtype=np.uint64)
    rank = 0
    for row in rows:
        current = row & ~mask if complement else row & mask
        # Pivot rows are reduced when stored, so eliminating them in order clears every pivot bit.
        for previous in range(rank):
            if current[pivot_words[previous]] & pivot_bits[previous]:
                current ^= pivots[previous]

        # A nonzero remainder becomes a new pivot row, whose pivot is its lowest set bit.
        for index in range(len(current)):
            if current[index]:
                pivots[rank] = current
                pivot_words[rank] = index
                pivot_bits[rank] = current[index] & (~current[index] + np.uint64(1))
                rank += 1
                break
    return rank
