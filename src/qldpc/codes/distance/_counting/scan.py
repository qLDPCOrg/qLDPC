# SPDX-License-Identifier: Apache-2.0

"""The scan of the words of one weight in a sector: its set-up, its loop, and its native chunks.

A sector is a basis whose quotient labels mark its nontrivial words.  ``count_words`` selects
information sets for the basis, with the basis family stages of ``information_sets.py``, packs
them, and visits each word of the given weight once, by enumerating combinations of the rows of
every set, layer by layer.  A step of two or more rows of an information set may skip the supports
whose words all weigh more than the given weight, with the completion tables of
``_search/completion_bounds.py`` (Section 3.9 of ``brouwer_zimmermann.md``).
``transport_basis_family`` moves the disjoint basis family that a counter keeps for a sector, from
which both its counts and its distance searches start, through a local gate, for deformed counters.
``logical_counts.py`` calls both.  Section 3 of ``brouwer_zimmermann.md`` describes the algorithm,
and Section 5 states the counting requirement.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits, popcount_word, row_reduce_packed, unpack_bits
from qldpc._compilation import always_compiled, compiled_on_request

from .._search import completion_bounds, information_sets
from .._search.completion_bounds import get_completion_bound
from .._search.information_sets import advance_support
from .._tuning import SearchTuning
from . import local_cliffords

_PackedMatrix = npt.NDArray[np.uint64]

# How a scan call ends: with every layer visited, after its allowance of chunks, or at a word that
# is lighter than the target and nontrivial.
_SCAN_FINISHED, _SCAN_UNFINISHED, _SCAN_FOUND_LIGHTER_WORD = 0, 1, 2

# The positions in the cursor of a scan, which marks where a later call resumes it: the layer, the
# set, and the depth within the current support of that layer.
_CURSOR_LAYER, _CURSOR_SET, _CURSOR_DEPTH = 0, 1, 2


####################################################################################################
# counting words of one weight


def count_words(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    weight: int,
    brute_force: bool,
    rank_rows: _PackedMatrix,
    *,
    complement: bool,
    max_exponent: int,
    reject_lighter: bool,
    get_disjoint_basis_family: Callable[[], information_sets.BasisFamily],
    tuning: SearchTuning,
) -> npt.NDArray[np.int64] | None:
    """Visit each nontrivial word of one sector with the given weight exactly once.

    Returns a histogram over exponents ``max_exponent - rank(rank_rows[:, mask])``, where the mask
    is the support of the word, or its complement.  The histogram sums to the number of words, and
    the caller weighs each exponent by ``2**exponent``, which could overflow a native integer.  The
    scan also visits every lighter word, so with ``reject_lighter`` it returns None as soon as it
    finds a nontrivial one.  With ``brute_force``, it scans the whole basis as one set, rather than
    the information sets of the Brouwer-Zimmermann search.  ``tuning`` sets the thresholds that
    choose those information sets, the completion tables of its steps, and the sizes of the chunks
    of the scan (see ``_tuning``).
    """
    # Select the information sets, or scan the whole basis as one set for brute force.
    if brute_force:
        rows, packed_labels = pack_bits(basis)[np.newaxis], pack_bits(labels)[np.newaxis]
        extractors = np.zeros((1, 0, rows.shape[2]), dtype=np.uint64)
        last_layer, last_num_completed = len(basis), 1
    else:
        basis_family = _select_basis_family(
            basis,
            labels,
            weight,
            get_disjoint_basis_family(),
            *information_sets.get_overlapping_family_policy(tuning, len(basis)),
        )
        # Matroid partitioning runs outside of _select_basis_family, which can be compiled.
        partitioned = information_sets.get_partitioned_family(
            basis,
            labels,
            basis_family,
            weight + 1,
            tuning,
        )
        if len(partitioned.ranks):
            basis_family = partitioned
        rows, packed_labels, extractors = _pack_basis_family(
            basis_family, basis.shape[1], labels.shape[1]
        )
        last_layer, last_num_completed = _get_last_step(
            basis_family.get_search_certificate(
                weight + 1, tuning.lp_minimum_work, tuning.lp_dual_max_denominator
            ),
            weight,
        )

    # Each row's machine words are followed by those of its label, so that one prefix XOR carries
    # both.  A step that skips completions also appends the row's tail (see _StepTables).
    rows_and_labels = np.concatenate((rows, packed_labels), axis=2)
    step_tables = _StepTables(
        None if brute_force else basis_family, rows_and_labels, weight + 1, tuning
    )

    # Scan in bounded calls, since compiled code cannot be interrupted within a call.  Each call
    # ends at the end of a step, so that the tables of the next step can be chosen.
    histogram = np.zeros(max_exponent + 1, dtype=np.int64)
    cursor = np.zeros(3, dtype=np.int64)
    cursor[_CURSOR_LAYER] = 1
    support = np.zeros(len(basis), dtype=np.int64)
    prefixes = np.zeros((len(basis), 1, step_tables.max_row_width), dtype=np.uint64)
    status = _SCAN_UNFINISHED
    while status == _SCAN_UNFINISHED:
        tables, step_rows = step_tables.select(int(cursor[_CURSOR_LAYER]), int(cursor[_CURSOR_SET]))
        status = _scan_coefficient_layers(
            rows_and_labels,
            rows.shape[2],
            extractors,
            last_layer,
            last_num_completed,
            weight,
            rank_rows,
            complement,
            max_exponent,
            reject_lighter,
            cursor,
            support,
            prefixes,
            histogram,
            tuning.scan_chunk_size,
            tuning.scan_chunks_per_call,
            step_rows,
            tables.tables,
            tables.offsets,
            tables.block_bits,
        )
    return None if status == _SCAN_FOUND_LIGHTER_WORD else histogram


def transport_basis_family(
    basis_family: information_sets.BasisFamily,
    positions: npt.NDArray[np.int64],
    local: npt.NDArray[np.uint64],
    basis: npt.NDArray[np.uint8],
) -> information_sets.BasisFamily | None:
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
        weight_divisor=information_sets.infer_weight_divisor(basis),
    )


@compiled_on_request
def _select_basis_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    weight: int,
    basis_family: information_sets.BasisFamily,
    max_sets: int,
    minimum_work: int,
    attempts_per_set: int,
    binomials: npt.NDArray[np.int64],
    rng: np.random.Generator,
) -> information_sets.BasisFamily:
    """Run the basis family stages that can be compiled: pruning, then an overlapping replacement.

    The stages aim at the counting stop ``L(t) >= weight + 1``; any basis family gives an exact
    count.
    """
    upper_bound = weight + 1
    pruned = information_sets.prune_family(basis_family, upper_bound)
    replacement = information_sets.get_overlapping_family(
        basis, labels, pruned, upper_bound, max_sets, minimum_work, attempts_per_set, binomials, rng
    )
    return replacement if len(replacement.ranks) > 0 else pruned


@compiled_on_request
def _pack_basis_family(
    basis_family: information_sets.BasisFamily, num_columns: int, num_label_columns: int
) -> tuple[_PackedMatrix, _PackedMatrix, _PackedMatrix]:
    """Pack the rows and labels of the sets of a basis family, and their coefficient extractors.

    The extractors are empty for a single set, since ownership compares sets.
    """
    num_sets, dimension = basis_family.generators.shape[:2]
    width = (num_columns + 63) // 64
    rows = np.empty((num_sets, dimension, width), dtype=np.uint64)
    packed_labels = np.empty((num_sets, dimension, (num_label_columns + 63) // 64), dtype=np.uint64)
    for index in range(num_sets):
        rows[index] = pack_bits(basis_family.generators[index])
        packed_labels[index] = pack_bits(basis_family.labels[index])
    return rows, packed_labels, _get_coefficient_extractors(basis_family, width)


def _get_last_step(certificate: information_sets.Certificate, weight: int) -> tuple[int, int]:
    """The first step after which the certificate rules out unvisited words of the given weight.

    A step completes a layer of one set, and the scan visits the sets of each layer in family order.
    Returns the layer of that step, and the number of sets that have completed that layer.  Equality
    with the weight would still allow unvisited words of the weight, so the certificate must exceed
    it.
    """
    # The first set to complete the last layer visits every word, so some step certifies.
    steps = itertools.product(
        range(1, certificate.dimension + 1), range(1, len(certificate.ranks) + 1)
    )
    return next(step for step in steps if certificate.certifies(*step, weight + 1))


class _StepTables:
    """The completion tables, if any, with which each step of a count skips completions.

    A step visits one layer of one set of ``basis_family``, or of the whole basis for brute force,
    for which ``basis_family`` is None and no step skips anything.  A step of at least two rows of a
    set whose block is an information set may skip the supports whose words, by the bound of
    ``completion_bounds``, all weigh at least ``upper_bound``: the target weight plus one, so that
    no word of the target weight, and no lighter word, is skipped.  ``tuning`` chooses its tables as
    for a layer of a distance search (see ``completion_bounds.select_completion_tables``), and the
    tables of the sets that a count has used are kept within
    ``tuning.completion_bound_max_table_bytes`` in all, for later steps of the same sets.
    """

    def __init__(
        self,
        basis_family: information_sets.BasisFamily | None,
        rows_and_labels: _PackedMatrix,
        upper_bound: int,
        tuning: SearchTuning,
    ) -> None:
        self._basis_family = basis_family
        self._rows_and_labels = rows_and_labels
        self._upper_bound = upper_bound
        self._tuning = tuning
        self._tails: dict[int, _PackedMatrix] = {}
        self._kept: dict[int, completion_bounds.CompletionTables] = {}
        no_tables = completion_bounds.NO_TABLES
        self._step: tuple[int, int] | None = None
        self._selected = (no_tables, rows_and_labels[:1])

        # The tails of every set whose block is an information set have the same number of words.
        self.max_row_width = rows_and_labels.shape[2]
        if basis_family is not None and np.any(
            basis_family.ranks == basis_family.generators.shape[1]
        ):
            num_tail_columns = basis_family.generators.shape[2] - basis_family.generators.shape[1]
            self.max_row_width += (num_tail_columns + 63) // 64

    def select(
        self, layer: int, set_index: int
    ) -> tuple[completion_bounds.CompletionTables, _PackedMatrix]:
        """The tables of a step, and the rows and labels of its set followed by their tails.

        A step without tables gets ``completion_bounds.NO_TABLES``, and rows that its scan ignores.
        """
        if self._step == (layer, set_index):
            return self._selected
        self._step = (layer, set_index)
        family = self._basis_family
        tables = None
        if (
            family is not None
            and layer >= 2
            and set_index < len(family.ranks)
            and family.ranks[set_index] == family.generators.shape[1]
        ):
            tails = self._get_tails(set_index)
            tables = completion_bounds.select_completion_tables(
                tails, layer, self._upper_bound, self._kept.get(set_index), self._tuning
            )
        if tables is None:
            self._selected = (completion_bounds.NO_TABLES, self._rows_and_labels[:1])
            return self._selected

        # Keep these tables, dropping those of other sets if all would not fit.
        if set_index not in self._kept:
            budget = self._tuning.completion_bound_max_table_bytes - len(tables.tables)
            for other in list(self._kept):
                if sum(len(kept.tables) for kept in self._kept.values()) <= budget:
                    break
                del self._kept[other]
            self._kept[set_index] = tables
        rows = np.concatenate(
            (self._rows_and_labels[set_index], self._get_tails(set_index)), axis=1
        )
        self._selected = (tables, rows[np.newaxis])
        return self._selected

    def _get_tails(self, set_index: int) -> _PackedMatrix:
        """The packed rows of a set without its pivot columns."""
        if set_index not in self._tails:
            assert self._basis_family is not None
            generators = self._basis_family.generators[set_index]
            pivots = self._basis_family.pivots[set_index]
            self._tails[set_index] = pack_bits(np.delete(generators, pivots, axis=1))
        return self._tails[set_index]


@compiled_on_request
def _get_coefficient_extractors(
    basis_family: information_sets.BasisFamily, width: int
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
        identity = pack_bits(np.eye(dimension, dtype=np.uint8))
        _, packed_transform, pivots = row_reduce_packed(
            pack_bits(basis_family.generators[index]), identity, np.arange(num_columns)
        )
        transform = unpack_bits(packed_transform, dimension)
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
    last_num_completed: int,
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
    step_rows: _PackedMatrix,
    tables: npt.NDArray[np.uint8],
    offsets: npt.NDArray[np.int64],
    block_bits: int,
) -> int:
    """Visit the supports of the step at ``cursor``, for at most ``max_chunks`` chunks.

    A step visits all supports of one layer of one set (see ``_count_chunk``).  The layers run from
    one to ``last_layer``, each set has a step in each layer, except that only the first
    ``last_num_completed`` sets visit the last layer, and the steps run in lexicographic order of
    their layer and set, so the steps that the scan visits are a prefix of that order.  Each word
    of the target weight is owned by the first step that visits it (see ``_is_owned``), which
    therefore lies in the prefix, since the certificate rules out unvisited words of the target
    weight after its last step.  This loop is a function of its own so that it can be compiled,
    which saves a call from Python for every chunk.

    With ``tables`` of ``completion_bounds.CompletionTables``, the step skips the completions that
    cannot be lighter than ``target + 1`` (see ``_count_chunk_skipping``), and ``step_rows`` holds
    the rows and labels of its set followed by their tails.  With ``completion_bounds.NO_TABLES``,
    it visits every support, and ignores ``step_rows``.  Skipped words weigh more than ``target``,
    so they are neither counted nor lighter than ``target``, and the certificate still rules out
    the words of the target weight that the scan neither visits nor skips.

    The scan is resumable, since compiled code cannot be interrupted.  ``cursor`` holds the layer,
    the set, and the depth at which to resume, and ``support`` and ``prefixes`` hold the current
    support and its prefix XORs, so that a later call continues the scan.  The first call passes a
    cursor at layer one, set zero and depth zero, and zero prefixes, and every call passes on the
    same ``cursor``, ``support``, ``prefixes`` and ``histogram``, and the same tables for every
    call of a step.  ``support[:layer]`` is the support of the current layer, which is how the
    chunks know its length.  A call ends at the end of its step, so that the caller can choose the
    tables of the next step.

    The caller calls it only while a step remains.  Returns ``_SCAN_FINISHED`` once the last step
    is complete, ``_SCAN_UNFINISHED`` if steps or chunks remain, or ``_SCAN_FOUND_LIGHTER_WORD``.
    """
    layer = cursor[_CURSOR_LAYER]
    set_index = cursor[_CURSOR_SET]
    while cursor[_CURSOR_DEPTH] >= 0:
        if max_chunks == 0:
            return _SCAN_UNFINISHED
        max_chunks -= 1
        if len(tables):
            depth, lighter = _count_chunk_skipping(
                step_rows,
                width,
                rows_and_labels.shape[2],
                extractors,
                set_index,
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
                tables,
                offsets,
                block_bits,
            )
        else:
            depth, lighter = _count_chunk(
                rows_and_labels[set_index : set_index + 1],
                width,
                extractors,
                set_index,
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

    # Move to the next step.
    num_sets = last_num_completed if layer == last_layer else len(rows_and_labels)
    cursor[_CURSOR_SET] += 1
    cursor[_CURSOR_DEPTH] = 0
    support[:] = 0
    if cursor[_CURSOR_SET] >= num_sets:
        cursor[_CURSOR_LAYER] += 1
        cursor[_CURSOR_SET] = 0
    return _SCAN_FINISHED if cursor[_CURSOR_LAYER] > last_layer else _SCAN_UNFINISHED


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
            pack_bits(moved), pack_bits(labels[index]), pivots[index, : ranks[index]]
        )
        if len(new_pivots) != ranks[index]:
            return False, moved_generators, moved_labels
        moved_generators[index] = unpack_bits(reduced, generators.shape[2])
        moved_labels[index] = unpack_bits(reduced_labels, labels.shape[2])
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
    """Visit up to ``chunk_size`` supports of ``len(support)`` rows, resuming at ``depth``.

    ``rows_and_labels`` has shape ``(1, num_rows, total_width)``, a stack of one set, because
    ``advance_support`` takes a stack of sets.  Each row has ``width`` machine words of the word
    followed by those of its quotient label.  ``support`` holds the current support, the increasing
    indices of its rows, and ``prefixes`` its prefix XORs (see ``advance_support``); ``depth`` is
    the position being advanced, or -1 once every support has been visited.  Each owned nontrivial
    word of weight ``target`` adds one to ``histogram`` (see ``count_words``).  Returns the depth
    from which to resume, and whether the scan stopped at a nontrivial word lighter than ``target``,
    which is checked only with ``reject_lighter``.
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

        # Visit the remaining supports that differ from this one only in their last row.
        stop = min(num_rows, support[depth] + chunk_size - visited)
        if _visit_last_rows(
            rows,
            prefixes[depth, 0],
            support[depth],
            stop,
            width,
            total_width,
            extractors,
            set_index,
            layer,
            target,
            rank_rows,
            complement,
            max_exponent,
            histogram,
            reject_lighter,
            word,
            pivots,
        ):
            return depth, True

        # Record the visited rows, and resume after them.
        visited += stop - support[depth]
        support[depth] = stop
    return depth, False


@always_compiled
def _count_chunk_skipping(
    step_rows: _PackedMatrix,
    width: int,
    label_end: int,
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
    tables: npt.NDArray[np.uint8],
    offsets: npt.NDArray[np.int64],
    block_bits: int,
) -> tuple[int, bool]:
    """Visit a chunk of a layer of at least two rows of one set, skipping hopeless completions.

    The arguments are those of ``_count_chunk``, for a set whose block is an information set, with
    ``label_end`` words of word and label in each row of ``step_rows``, followed by the row's tail:
    the row without its pivot columns, which ``tables``, ``offsets`` and ``block_bits`` of
    ``completion_bounds.CompletionTables`` index.  Each word weighs the number of its rows plus the
    weight of its tail.  Before the scan tries row ``s`` as the first of the last two rows of a
    support, it bounds the weights of the words that complete the first ``len(support) - 2`` rows
    with two rows from row ``s`` on, and skips them all if none weighs at most ``target``.  Words of
    the target weight, and lighter ones, are therefore visited exactly as ``_count_chunk`` visits
    them, though the order of a chunk's lookups and weighed supports differs.

    ``depth`` is below ``len(support) - 1`` while the scan chooses the rows before the last one, and
    ``len(support) - 1`` while it visits last rows.  A chunk ends once the supports that it weighed
    and the bounds that it looked up reach ``chunk_size`` together.
    """
    rows = step_rows[0]
    num_rows, total_width = rows.shape
    widths = np.full(1, total_width, dtype=np.int64)
    layer = len(support)
    word = np.empty(width, dtype=np.uint64)
    pivots = np.empty((len(rank_rows), width), dtype=np.uint64)
    # The search steps through the supports of the first layer - 1 rows, whose last row is the
    # first of the last two rows of a support, and visits the last row of each in its own loop.
    # Skipping completes a word only if it weighs more than the target, so its limit is that of a
    # best weight of target + 1.
    head = support[: layer - 1]
    head_rows = step_rows[:, : num_rows - 1]
    pair_depth = layer - 2
    limit = target + 1 - layer
    visited = 0
    while visited < chunk_size:
        if depth < layer - 1:
            depth = advance_support(head_rows, widths, head, prefixes, depth)
            if depth < 0:
                break
            first = support[pair_depth]
            visited += 1
            prefix = prefixes[pair_depth, 0]
            bound = get_completion_bound(
                tables, offsets, block_bits, first, prefix[label_end:], limit
            )
            if bound >= limit:
                support[pair_depth] = num_rows
                continue

            # Otherwise add row first to the prefix, and try every later row as the last row.
            for index in range(total_width):
                prefixes[layer - 1, 0, index] = prefix[index] ^ rows[first, index]
            depth = layer - 1
            support[depth] = first + 1
            continue

        # Visit the remaining supports that differ from this one only in their last row.
        stop = min(num_rows, support[depth] + chunk_size - visited)
        if _visit_last_rows(
            rows,
            prefixes[depth, 0],
            support[depth],
            stop,
            width,
            label_end,
            extractors,
            set_index,
            layer,
            target,
            rank_rows,
            complement,
            max_exponent,
            histogram,
            reject_lighter,
            word,
            pivots,
        ):
            return depth, True
        visited += stop - support[depth]
        support[depth] = stop
        if stop == num_rows:
            depth = pair_depth
            support[depth] += 1
    return depth, False


@always_compiled(inline=True)
def _visit_last_rows(
    rows: _PackedMatrix,
    prefix: npt.NDArray[np.uint64],
    start: int,
    stop: int,
    width: int,
    label_end: int,
    extractors: npt.NDArray[np.uint64],
    set_index: int,
    layer: int,
    target: int,
    rank_rows: _PackedMatrix,
    complement: bool,
    max_exponent: int,
    histogram: npt.NDArray[np.int64],
    reject_lighter: bool,
    word: npt.NDArray[np.uint64],
    pivots: _PackedMatrix,
) -> bool:
    """Visit the words ``prefix ^ rows[row]`` for rows from ``start`` up to ``stop``.

    Each row has ``width`` machine words of the word, followed by those of its quotient label up to
    ``label_end``.  Each owned nontrivial word of weight ``target`` adds one to ``histogram`` (see
    ``count_words``).  ``word`` and ``pivots`` are scratch space.  Returns whether a nontrivial word
    is lighter than ``target``, which is checked only with ``reject_lighter``, and at which the
    visit stops.
    """
    for row in range(start, stop):
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
        for index in range(width, label_end):
            if prefix[index] != rows[row, index]:
                nontrivial = True
                break
        if nontrivial and weight < target:
            return True
        if nontrivial and _is_owned(word, extractors, set_index, layer):
            rank = _get_masked_rank(rank_rows, word, complement, pivots)
            histogram[max_exponent - rank] += 1
    return False


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
