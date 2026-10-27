# SPDX-License-Identifier: Apache-2.0

"""The scan of the words of one weight in a sector: its set-up, its loop, and its native chunks.

A sector is a basis whose quotient labels mark its nontrivial words.  ``count_words`` selects
information sets for the basis, with the basis family stages of ``_information_sets.py``, packs
them, and visits each word of the given weight once, by enumerating
combinations of the rows of every set, layer by layer.  ``logical_counts.py`` calls it.  Section 3
of ``brouwer_zimmermann.md`` describes the algorithm, and Section 5 states the counting requirement.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from qldpc._compilation import always_compiled, compiled_on_request

from .. import _information_sets
from .._binary import pack_bits, popcount_word, row_reduce_packed
from .._information_sets import advance_support
from ..core import DistanceMethod
from . import local_cliffords

_BinaryMatrix = npt.NDArray[np.uint8]
_PackedMatrix = npt.NDArray[np.uint64]

# Coefficient combinations that one call of ``_count_chunk`` visits.
_CHUNK_SIZE = 1 << 16

# Chunks that one call of ``_scan_coefficient_layers`` visits before it returns, unless the scan
# ends sooner.  Compiled code cannot be interrupted, so a compiled scan is interruptible only
# between its calls.
_CHUNKS_PER_SCAN_CALL = 64

# How a scan call ends: with every layer visited, after its allowance of chunks, or at a word that
# is lighter than the target and nontrivial.
_SCAN_FINISHED, _SCAN_UNFINISHED, _SCAN_FOUND_LIGHTER_WORD = 0, 1, 2

# The positions in the cursor of a scan, which marks where a later call resumes it: the layer, the
# set, and the depth within the combinations of that layer.
_CURSOR_LAYER, _CURSOR_SET, _CURSOR_DEPTH = 0, 1, 2


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
    get_disjoint_basis_family: Callable[[], _information_sets.BasisFamily],
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
    if method == "brute_force":
        rows, packed_labels = pack_bits(basis)[np.newaxis], pack_bits(labels)[np.newaxis]
        extractors = np.zeros((1, 0, rows.shape[2]), dtype=np.uint64)
        last_layer = len(basis)
    else:
        rows, packed_labels, extractors, last_layer = _prepare_scan(
            basis,
            labels,
            weight,
            get_disjoint_basis_family(),
            max_sets,
            _information_sets.MINIMUM_WORK,
            _information_sets.get_capped_binomials(len(basis)),
            np.random.default_rng(0),
        )

    # Scan in bounded calls, which compiled code cannot be interrupted within.
    histogram = np.zeros(max_exponent + 1, dtype=np.int64)
    cursor = np.zeros(3, dtype=np.int64)
    cursor[_CURSOR_LAYER] = 1
    support = np.zeros(len(basis), dtype=np.int64)
    # The words of a row are followed by those of its label, so that their sums are kept together.
    rows_and_labels = np.concatenate((rows, packed_labels), axis=2)
    prefixes = np.zeros((len(basis), 1, rows_and_labels.shape[2]), dtype=np.uint64)
    status = _SCAN_UNFINISHED
    while status == _SCAN_UNFINISHED:
        status = _scan_coefficient_layers(
            rows_and_labels,
            rows.shape[2],
            extractors,
            last_layer,
            weight,
            rank_rows,
            complement,
            max_exponent,
            reject_lighter,
            cursor,
            support,
            prefixes,
            histogram,
            _CHUNK_SIZE,
            _CHUNKS_PER_SCAN_CALL,
        )
    return None if status == _SCAN_FOUND_LIGHTER_WORD else histogram


def transport_basis_family(
    basis_family: _information_sets.BasisFamily,
    positions: npt.NDArray[np.int64],
    local: npt.NDArray[np.uint64],
    basis: _BinaryMatrix,
) -> _information_sets.BasisFamily | None:
    """The basis family of a sector after a local gate, or None if one of its sets stops being one.

    Each set is mapped through the gate and reduced again on its own pivot columns, so the pivots
    stay disjoint and the labels move with their rows.  A set stops being one if its pivot columns
    are no longer independent.
    """
    moved, generators, labels = _transport_sets(
        basis_family.generators,
        basis_family.labels,
        basis_family.pivots,
        basis_family.ranks,
        positions,
        local,
    )
    if not moved:
        return None
    return basis_family._replace(
        generators=generators,
        labels=labels,
        weight_divisor=_information_sets.infer_weight_divisor(basis),
    )


@compiled_on_request
def _prepare_scan(
    basis: _BinaryMatrix,
    labels: _BinaryMatrix,
    weight: int,
    basis_family: _information_sets.BasisFamily,
    max_sets: int,
    minimum_work: int,
    binomials: npt.NDArray[np.int64],
    rng: np.random.Generator,
) -> tuple[_PackedMatrix, _PackedMatrix, _PackedMatrix, int]:
    """Select BZ information sets that certify every word of the given weight, and pack them.

    The basis family stages aim at the counting stop ``L(t) >= weight + 1``; any basis family gives
    an exact count.  Returns the packed rows and labels of the sets, their coefficient extractors
    (which are empty for a single set, since ownership compares sets), and the last layer to scan,
    which is the first one after which the certificate rules out unvisited words of the weight.
    """
    dimension = basis.shape[0]
    upper_bound = weight + 1
    sets = _information_sets.prune_family(basis_family, upper_bound)
    replacement = _information_sets.get_overlapping_family(
        basis, labels, sets, upper_bound, max_sets, minimum_work, binomials, rng
    )
    if len(replacement.ranks) > 0:
        sets = replacement

    num_sets = len(sets.ranks)
    width = (basis.shape[1] + 63) // 64
    rows = np.empty((num_sets, dimension, width), dtype=np.uint64)
    packed_labels = np.empty((num_sets, dimension, (labels.shape[1] + 63) // 64), dtype=np.uint64)
    for index in range(num_sets):
        rows[index] = pack_bits(sets.generators[index])
        packed_labels[index] = pack_bits(sets.labels[index])

    bounds = _information_sets.get_certificate_lower_bounds(
        dimension, sets.ranks, sets.weight_divisor, sets.max_coverage
    )
    last_layer = dimension
    for layer in range(1, dimension + 1):
        # Equality would still allow unvisited words of the target weight.
        if bounds[layer] > weight:
            last_layer = layer
            break
    return rows, packed_labels, _get_coefficient_extractors(sets, width), last_layer


@compiled_on_request
def _get_coefficient_extractors(
    basis_family: _information_sets.BasisFamily, width: int
) -> _PackedMatrix:
    """Packed rows ``E.T`` for each set such that ``generators @ E`` is the identity over GF(2).

    Bit ``b`` of the coefficient vector of a word ``w`` is then the parity of ``w & E.T[b]``.
    """
    num_sets, dimension, num_columns = basis_family.generators.shape
    if num_sets == 1:
        return np.zeros((1, 0, width), dtype=np.uint64)

    extractors = np.empty((num_sets, dimension, width), dtype=np.uint64)
    for index in range(num_sets):
        if basis_family.ranks[index] == dimension:
            # A full set is the identity on its pivot columns, which hold its coefficients.
            one_hot = np.zeros(dimension * num_columns, dtype=np.uint8)
            one_hot[np.arange(dimension) * num_columns + basis_family.pivots[index]] = 1
            extractors[index] = pack_bits(one_hot.reshape(dimension, num_columns))
            continue

        # Row reduction finds T with T @ generators = R, where R is the identity on its pivot
        # columns P, so that w[P] @ T recovers the coefficients of w.
        identity = np.eye(dimension, dtype=np.uint8)
        _, transform, pivots = row_reduce_packed(
            basis_family.generators[index], identity, np.arange(num_columns)
        )
        extractor = np.zeros((dimension, num_columns), dtype=np.uint8)
        for row in range(len(pivots)):
            extractor[:, pivots[row]] = transform[row]
        extractors[index] = pack_bits(extractor)
    return extractors


@compiled_on_request
def _scan_coefficient_layers(
    rows_and_labels: _PackedMatrix,
    width: int,
    extractors: _PackedMatrix,
    last_layer: int,
    target: int,
    rank_rows: _PackedMatrix,
    complement: bool,
    max_exponent: int,
    reject_lighter: bool,
    cursor: npt.NDArray[np.int64],
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
    histogram: npt.NDArray[np.int64],
    chunk_size: int,
    max_chunks: int,
) -> int:
    """Visit the combinations of each layer in each set, for at most ``max_chunks`` chunks.

    The layers run from one to ``last_layer``, and each set visits all of its combinations of that
    many rows (see ``_count_chunk``).  This loop is a function of its own so that it can be
    compiled, which saves a call from Python for every chunk.

    The scan is resumable, since compiled code cannot be interrupted.  ``cursor`` holds the layer,
    the set, and the depth at which to resume, and ``support`` and ``prefixes`` hold the current
    combination and its prefix sums, so that a later call continues the scan.  The first call passes
    a cursor at layer one, set zero and depth zero, and zero prefixes, and every call passes on the
    same ``cursor``, ``support``, ``prefixes`` and ``histogram``.  ``support[:layer]`` is the
    combination of the current layer, which is how ``_count_chunk`` knows its length.

    Returns ``_SCAN_FINISHED``, ``_SCAN_UNFINISHED`` if chunks remain, or
    ``_SCAN_FOUND_LIGHTER_WORD``.
    """
    while cursor[_CURSOR_LAYER] <= last_layer:
        layer = cursor[_CURSOR_LAYER]
        while cursor[_CURSOR_SET] < len(rows_and_labels):
            while cursor[_CURSOR_DEPTH] >= 0:
                if max_chunks == 0:
                    return _SCAN_UNFINISHED
                max_chunks -= 1
                depth, lighter = _count_chunk(
                    rows_and_labels[cursor[_CURSOR_SET] : cursor[_CURSOR_SET] + 1],
                    width,
                    extractors,
                    cursor[_CURSOR_SET],
                    support[:layer],
                    prefixes,
                    cursor[_CURSOR_DEPTH],
                    target,
                    rank_rows,
                    complement,
                    max_exponent,
                    histogram,
                    reject_lighter,
                    chunk_size,
                )
                cursor[_CURSOR_DEPTH] = depth
                if lighter:
                    return _SCAN_FOUND_LIGHTER_WORD
            cursor[_CURSOR_SET] += 1
            cursor[_CURSOR_DEPTH] = 0
            support[:] = 0
        cursor[_CURSOR_LAYER] += 1
        cursor[_CURSOR_SET] = 0
    return _SCAN_FINISHED


@compiled_on_request
def _transport_sets(
    generators: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    pivots: npt.NDArray[np.int64],
    ranks: npt.NDArray[np.int64],
    positions: npt.NDArray[np.int64],
    local: npt.NDArray[np.uint64],
) -> tuple[bool, npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Map every set through a local gate, and reduce it again on its own pivot columns.

    Returns whether every set kept its rank, and the generators and labels of the sets.
    """
    moved_generators = np.empty_like(generators)
    moved_labels = np.empty_like(labels)
    for index in range(len(ranks)):
        moved = local_cliffords.apply_local(generators[index], positions, local)
        reduced, reduced_labels, new_pivots = row_reduce_packed(
            moved, labels[index], pivots[index, : ranks[index]]
        )
        if len(new_pivots) != ranks[index]:
            return False, moved_generators, moved_labels
        moved_generators[index] = reduced
        moved_labels[index] = reduced_labels
    return True, moved_generators, moved_labels


####################################################################################################
# native kernels


@always_compiled
def _count_chunk(
    rows_and_labels: _PackedMatrix,
    width: int,
    extractors: npt.NDArray[np.uint64],
    set_index: int,
    support: npt.NDArray[np.int64],
    prefixes: npt.NDArray[np.uint64],
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

    ``rows_and_labels`` holds the rows of one set, stacked as the only set that ``advance_support``
    enumerates, and each row has ``width`` words followed by those of its quotient label.
    ``support`` holds the increasing row indices of the current combination, and ``prefixes`` their
    prefix sums; ``depth`` is the position being advanced, or -1 once every combination has been
    visited.  Each owned nontrivial word of weight ``target`` adds one to ``histogram`` (see
    ``count_words``).  Returns the depth from which to resume, and whether the scan stopped at a
    nontrivial word lighter than ``target``, which is checked only with ``reject_lighter``.
    """
    rows = rows_and_labels[0]
    num_rows, total_width = rows.shape
    widths = np.full(1, total_width, dtype=np.int64)
    layer = len(support)
    word = np.empty(width, dtype=np.uint64)
    pivots = np.empty((len(rank_rows), width), dtype=np.uint64)
    visited = 0
    while visited < chunk_size:
        depth = advance_support(rows_and_labels, widths, support, prefixes, depth)
        if depth < 0:
            break

        # Visit the remaining combinations that differ only in their last row.
        stop = min(num_rows, support[depth] + chunk_size - visited)
        prefix = prefixes[depth, 0]
        for row in range(support[depth], stop):
            weight = np.int64(0)
            for index in range(width):
                word[index] = prefix[index] ^ rows[row, index]
                weight += popcount_word(word[index])
                # A heavier word is rejected below, so its unwritten words are never read.
                if weight > target:
                    break
            if weight > target or (weight < target and not reject_lighter):
                continue

            # A nonzero quotient label marks a nontrivial logical rather than a stabilizer.
            nontrivial = False
            for index in range(width, total_width):
                if prefix[index] != rows[row, index]:
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
