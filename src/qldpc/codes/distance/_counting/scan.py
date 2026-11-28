# SPDX-License-Identifier: Apache-2.0

"""The scan of the words of one weight in a sector: its set-up, its loop, and its compiled chunks.

A sector is a basis whose quotient labels mark its nontrivial words.  ``count_words`` counts the
nontrivial words of one weight in a sector, and ``transport_basis_family`` moves a basis family of a
sector through a local gate.  ``brouwer_zimmermann.md#sec-counting`` explains counts.

``count_words`` builds its basis family from the result of ``get_disjoint_basis_family``.  It
applies the later stages of ``_information_sets/basis_families.py``, as distance searches do:
pruning and an overlapping replacement in ``_select_basis_family``, which can be compiled, and then
matroid partitioning.  It stops and skips only as the strict stop of counting of
``brouwer_zimmermann_code.md#sec-requirements`` allows.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_num_words, pack_bits, popcount_word, row_reduce_packed, unpack_bits
from qldpc._compilation import always_compiled, compiled_on_request

from .._information_sets import basis_families, certificates, completion_bounds
from .._information_sets.completion_bounds import get_completion_bound
from .._information_sets.support_stepping import advance_support
from .._tuning import SearchTuning
from . import local_cliffords

_PackedMatrix = npt.NDArray[np.uint64]

# How a scan call ends: with every layer visited, after its allowance of chunks or before its
# stopping step, or at a word that is lighter than the target and nontrivial.
_SCAN_FINISHED, _SCAN_UNFINISHED, _SCAN_FOUND_LIGHTER_WORD = 0, 1, 2

# The indices of the layer, the set, and the depth in the position of a ``_Cursor``.
_CURSOR_LAYER, _CURSOR_SET, _CURSOR_DEPTH = 0, 1, 2


class _Cursor(NamedTuple):
    """Where a scan resumes: its step and depth, its support, and the prefix XORs of the support.

    Attributes:
        position: The layer, the set, and the depth, at the indices ``_CURSOR_LAYER``,
            ``_CURSOR_SET``, and ``_CURSOR_DEPTH``.  The step is a pair of a layer and a set (see
            ``_Steps``).  The depth is that of ``advance_support`` in the step's support, or -1
            once the step has visited every support.
        support: The step's support in its first ``layer`` entries, with an entry for each row of
            the basis.
        prefixes: Row ``level`` holds the XOR of the rows ``support[:level]`` of the step's set, as
            ``advance_support`` describes, from a zero offset, in the machine words that the step
            reads from it.  Its rows are as wide as the widest rows of a step.
    """

    position: npt.NDArray[np.int64]
    support: npt.NDArray[np.int64]
    prefixes: npt.NDArray[np.uint64]


####################################################################################################
# counting words of one weight


def count_words(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    weight: int,
    rank_rows: _PackedMatrix,
    *,
    brute_force: bool,
    complement: bool,
    max_exponent: int,
    reject_lighter: bool,
    get_disjoint_basis_family: Callable[[], basis_families.BasisFamily],
    tuning: SearchTuning,
) -> npt.NDArray[np.int64] | None:
    """Count each nontrivial word of one sector with the given weight exactly once.

    Returns a histogram over the exponents ``max_exponent - rank(rank_rows[:, mask])``, where
    ``mask`` is the support of a counted word, or its complement with ``complement``.  The caller
    weighs each exponent by ``2**exponent``, which could overflow a 64-bit integer.  The scan also
    visits every lighter word, and with ``reject_lighter`` returns None at the first that is
    nontrivial.  With ``brute_force``, it scans the whole basis rather than a basis family.
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
            basis_families.get_overlapping_family_policy(tuning, len(basis)),
        )
        # Matroid partitioning is plain Python, so it runs after _select_basis_family, which can be
        # compiled.
        partitioned = basis_families.get_partitioned_family(
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
        last_layer, last_num_completed = _get_last_step(basis_family.certificate, weight)

    # Each row's machine words are followed by those of its label, so that one prefix XOR carries
    # both.  A step that skips completions also appends the row's tail (see _StepTables).
    rows_and_labels = np.concatenate((rows, packed_labels), axis=2)
    step_tables = _StepTables(
        None if brute_force else basis_family, rows_and_labels, weight + 1, tuning
    )
    steps = _Steps(len(rows_and_labels), last_layer, last_num_completed)

    # Scan in bounded calls, since compiled code cannot be interrupted within a call.  Tables are
    # chosen in Python, so a call that starts a step that may use tables ends with that step, and
    # any other call ends before the next such step.
    histogram = np.zeros(max_exponent + 1, dtype=np.int64)
    position = np.zeros(3, dtype=np.int64)
    position[_CURSOR_LAYER] = 1
    cursor = _Cursor(
        position,
        np.zeros(len(basis), dtype=np.int64),
        np.zeros((len(basis), step_tables.max_row_width), dtype=np.uint64),
    )
    status = _SCAN_UNFINISHED
    while status == _SCAN_UNFINISHED:
        step = int(cursor.position[_CURSOR_LAYER]), int(cursor.position[_CURSOR_SET])
        if step_tables.may_use_tables(*step):
            tables, step_rows = step_tables.select(*step)
            stop = steps.get_next(*step)
        else:
            tables, step_rows = completion_bounds.NO_TABLES, rows_and_labels[0]
            stop = steps.find(step, step_tables.may_use_tables) if step_tables.layers else (0, 0)
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
            histogram,
            tuning.scan_chunk_size,
            tuning.scan_chunks_per_call,
            stop[0],
            stop[1],
            step_rows,
            tables,
        )
    return None if status == _SCAN_FOUND_LIGHTER_WORD else histogram


def transport_basis_family(
    basis_family: basis_families.BasisFamily,
    positions: npt.NDArray[np.int64],
    local: npt.NDArray[np.uint64],
    basis: npt.NDArray[np.uint8],
) -> basis_families.BasisFamily | None:
    """The basis family of a sector after a local gate, or None if the gate breaks one of its sets.

    Each set is mapped through the gate and reduced again, with its labels, on its own pivot
    columns, and the gate breaks a set that loses rank there
    (``brouwer_zimmermann.md#sec-deformed-counters``).  ``basis`` is the basis of the sector after
    the gate, from which the weight divisor of the family is inferred again.
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
        weight_divisor=basis_families.infer_weight_divisor(basis),
    )


@compiled_on_request
def _select_basis_family(
    basis: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.uint8],
    weight: int,
    basis_family: basis_families.BasisFamily,
    policy: basis_families.OverlappingFamilyPolicy,
) -> basis_families.BasisFamily:
    """Run the basis family stages that can be compiled: pruning, then an overlapping replacement.

    The stages aim at the stop of a count, ``L(t) >= weight + 1``, though any basis family gives an
    exact count (``brouwer_zimmermann_code.md#sec-requirements``).  ``policy`` is that of the
    overlapping replacement.
    """
    upper_bound = weight + 1
    pruned = basis_families.prune_family(basis_family, upper_bound)
    replacement = basis_families.get_overlapping_family(basis, labels, pruned, upper_bound, policy)
    return replacement if len(replacement.ranks) > 0 else pruned


@compiled_on_request
def _pack_basis_family(
    basis_family: basis_families.BasisFamily, num_columns: int, num_label_columns: int
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


def _get_last_step(certificate: certificates.Certificate, weight: int) -> tuple[int, int]:
    """The first step after which the certificate rules out unvisited words of the given weight.

    Returns the step's layer, and how many sets have completed that layer (see ``_Steps``).  The
    certificate must exceed the weight, not equal it
    (``brouwer_zimmermann_code.md#sec-requirements``).
    """
    # The first set to complete the last layer visits every word, so some step certifies.
    steps = itertools.product(
        range(1, certificate.dimension + 1), range(1, len(certificate.ranks) + 1)
    )
    return next(step for step in steps if certificate.certifies(*step, weight + 1))


class _Steps:
    """The steps of a scan, the pairs of a layer and a set, in the order in which it visits them.

    Each of ``num_sets`` sets has a step in each layer from one to ``last_layer``, except that only
    the first ``last_num_completed`` sets have one in the last layer.  The step ``(0, 0)``, which
    no scan visits, stands for the end of the scan.
    """

    def __init__(self, num_sets: int, last_layer: int, last_num_completed: int) -> None:
        self._num_sets = num_sets
        self._last_layer = last_layer
        self._last_num_completed = last_num_completed

    def get_next(self, layer: int, set_index: int) -> tuple[int, int]:
        """The step after a step, or ``(0, 0)`` after the last step."""
        return _get_next_step(
            layer, set_index, self._num_sets, self._last_layer, self._last_num_completed
        )

    def find(self, step: tuple[int, int], accept: Callable[[int, int], bool]) -> tuple[int, int]:
        """The first step after ``step`` that ``accept`` accepts, or ``(0, 0)`` if none does."""
        step = self.get_next(*step)
        while step != (0, 0) and not accept(*step):
            step = self.get_next(*step)
        return step


@compiled_on_request
def _get_next_step(
    layer: int, set_index: int, num_sets: int, last_layer: int, last_num_completed: int
) -> tuple[int, int]:
    """The step after a step of a scan, or ``(0, 0)`` after its last step (see ``_Steps``)."""
    if set_index + 1 < (last_num_completed if layer == last_layer else num_sets):
        return layer, set_index + 1
    return (layer + 1, 0) if layer < last_layer else (0, 0)


class _StepTables:
    """The completion tables, if any, with which each step of a count skips completions.

    A step may skip completions if its layer has at least two rows, and the block of its set of
    ``basis_family`` is an information set.  It skips the supports whose words all weigh at least
    ``upper_bound``, the target weight plus one.  ``tuning`` chooses and keeps the tables, as
    ``brouwer_zimmermann.md#sec-completion-gate`` describes.  For brute force, ``basis_family`` is
    None, and no step skips.
    """

    def __init__(
        self,
        basis_family: basis_families.BasisFamily | None,
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
        # The step whose tables select chose last, and its choice, which a resumed step reuses.
        self._step: tuple[int, int] | None = None
        self._selected: tuple[completion_bounds.CompletionTables, _PackedMatrix] | None = None

        # The layers whose steps the tuning considers for tables.  They form a range, since
        # binomials rise and then fall.
        self.layers = range(0)
        num_rows = 0 if basis_family is None else basis_family.generators.shape[1]
        considered = [
            layer
            for layer in range(num_rows + 1)
            if completion_bounds.considers_tables(num_rows, layer, tuning)
        ]
        if considered:
            self.layers = range(considered[0], considered[-1] + 1)

        # Rows of steps with tables carry their tails, which have the same number of words in every
        # set whose block is an information set.
        self.max_row_width = rows_and_labels.shape[2]
        if self.layers:
            assert basis_family is not None
            num_tail_columns = basis_family.generators.shape[2] - num_rows
            self.max_row_width += get_num_words(num_tail_columns)

    def may_use_tables(self, layer: int, set_index: int) -> bool:
        """Whether a step might use tables: whether the tuning considers them for it.

        Such a step is in one of ``layers``, and its set's block is an information set.
        """
        family = self._basis_family
        return (
            layer in self.layers
            and family is not None
            and bool(family.ranks[set_index] == family.generators.shape[1])
        )

    def select(
        self, layer: int, set_index: int
    ) -> tuple[completion_bounds.CompletionTables, _PackedMatrix]:
        """The tables of a step that ``may_use_tables``, and the rows, labels, and tails of its set.

        A step for which ``completion_bounds.select_completion_tables`` declines tables gets
        ``completion_bounds.NO_TABLES``, with rows that its scan ignores.  Selecting the same step
        again, as a resumed scan does, returns the same choice.
        """
        if self._step == (layer, set_index) and self._selected is not None:
            return self._selected
        self._step = (layer, set_index)
        tables = completion_bounds.select_completion_tables(
            self._get_tails(set_index),
            layer,
            self._upper_bound,
            self._kept.get(set_index),
            self._tuning,
        )
        if tables is None:
            self._selected = (completion_bounds.NO_TABLES, self._rows_and_labels[0])
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
        self._selected = (tables, rows)
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
    basis_family: basis_families.BasisFamily, width: int
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
    cursor: _Cursor,
    histogram: npt.NDArray[np.int64],
    chunk_size: int,
    max_chunks: int,
    stop_layer: int,
    stop_set: int,
    step_rows: _PackedMatrix,
    tables: completion_bounds.CompletionTables,
) -> int:
    """Visit the supports of the steps from ``cursor`` on, for at most ``max_chunks`` chunks.

    The steps run in the order of ``_Steps``, and the call stops before ``(stop_layer, stop_set)``.
    ``tables`` are the ``CompletionTables`` of a step, or ``NO_TABLES``.  A call with tables covers
    one step, whose rows, labels, and tails ``step_rows`` holds; a call without ignores
    ``step_rows``.  The loop is a function of its own so that it can be compiled, which saves a call
    from Python for every chunk.

    Compiled code cannot be interrupted, so the scan resumes across calls from ``cursor``.  Every
    call passes the same ``cursor`` and ``histogram``, and every call of a step the same tables.
    The first call passes a cursor at layer one, set zero, and depth zero, and zero prefixes.
    Returns how the call ended, as a ``_SCAN_*`` code.
    """
    position, support, prefixes = cursor.position, cursor.support, cursor.prefixes
    with_tables = len(tables.tables) > 0
    while True:
        # Visit the chunks of the step at the cursor.
        layer = position[_CURSOR_LAYER]
        set_index = position[_CURSOR_SET]
        while position[_CURSOR_DEPTH] >= 0:
            if max_chunks == 0:
                return _SCAN_UNFINISHED
            max_chunks -= 1
            if with_tables:
                depth, lighter = _count_chunk_skipping(
                    step_rows,
                    width,
                    rows_and_labels.shape[2],
                    extractors,
                    set_index,
                    support[:layer],
                    prefixes,
                    position[_CURSOR_DEPTH],
                    target,
                    rank_rows,
                    complement,
                    max_exponent,
                    histogram,
                    reject_lighter,
                    chunk_size,
                    tables,
                )
            else:
                depth, lighter = _count_chunk(
                    rows_and_labels[set_index],
                    width,
                    extractors,
                    set_index,
                    support[:layer],
                    prefixes,
                    position[_CURSOR_DEPTH],
                    target,
                    rank_rows,
                    complement,
                    max_exponent,
                    histogram,
                    reject_lighter,
                    chunk_size,
                )
            position[_CURSOR_DEPTH] = depth
            if lighter:
                return _SCAN_FOUND_LIGHTER_WORD

        # Move to the next step, and stop at the end of the scan or before the stopping step.
        layer, set_index = _get_next_step(
            layer, set_index, len(rows_and_labels), last_layer, last_num_completed
        )
        if layer == 0:
            return _SCAN_FINISHED
        position[_CURSOR_LAYER], position[_CURSOR_SET] = layer, set_index
        position[_CURSOR_DEPTH] = 0
        support[:] = 0
        if layer == stop_layer and set_index == stop_set:
            return _SCAN_UNFINISHED


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
# compiled kernels


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

    ``rows_and_labels`` holds the rows of one set, each with ``width`` machine words of its word,
    then those of its quotient label.  ``prefixes`` holds the prefix XORs of ``support``, and
    ``depth`` is the position being advanced, or -1 once every support has been visited (see
    ``advance_support``).  Owned nontrivial words (see ``_is_owned``) of weight ``target`` enter
    ``histogram`` (see ``count_words``).  Returns the depth from which to resume, and whether the
    chunk stopped at a nontrivial word lighter than ``target``, which it checks for only with
    ``reject_lighter``.
    """
    num_rows, total_width = rows_and_labels.shape
    layer = len(support)
    word = np.empty(width, dtype=np.uint64)
    pivots = np.empty((len(rank_rows), width), dtype=np.uint64)
    visited = 0
    while visited < chunk_size:
        depth = advance_support(rows_and_labels, total_width, support, prefixes, depth)
        if depth < 0:
            break

        # Visit the remaining supports that differ from this one only in their last row.
        stop = min(num_rows, support[depth] + chunk_size - visited)
        if _visit_last_rows(
            rows_and_labels,
            prefixes[depth],
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
    tables: completion_bounds.CompletionTables,
) -> tuple[int, bool]:
    """Visit a chunk of a layer of at least two rows of one set, skipping hopeless completions.

    The set's block is an information set.  The arguments are those of ``_count_chunk``, except for
    ``step_rows``, ``label_end``, and ``tables``.  Each row of ``step_rows`` holds ``label_end``
    machine words of its word and label, then its tail, the row without its pivot columns.
    ``tables`` are the completion tables of the tails.  Completions that the tables prove heavier
    than ``target`` are skipped (``brouwer_zimmermann.md#sec-completion-contracts``).  So words of
    weight ``target`` and lighter are visited as ``_count_chunk`` visits them.  A chunk ends once
    its weighed supports and looked-up bounds reach ``chunk_size`` together.  ``depth`` is
    ``len(support) - 1`` while the scan visits last rows, and smaller while it chooses the rows
    before them.
    """
    table_bytes, offsets, block_bits = tables.tables, tables.offsets, tables.block_bits
    num_rows, total_width = step_rows.shape
    layer = len(support)
    word = np.empty(width, dtype=np.uint64)
    pivots = np.empty((len(rank_rows), width), dtype=np.uint64)
    # Step through the heads of the supports, their first ``layer - 1`` rows, and visit the rows
    # that complete each head in a loop of their own.  A head ends with the first of the last two
    # rows.  The pair steps of ``_scan`` in ``_brouwer_zimmermann/affine_search.py`` skip the same
    # way, with a limit that falls as they find lighter words.
    head = support[: layer - 1]
    head_rows = step_rows[: num_rows - 1]
    pair_depth = layer - 2
    limit = target + 1 - layer
    visited = 0
    while visited < chunk_size:
        # Choose the next first row of the last two, and skip it and every later one if the bound
        # rules out the target weight.  Each lookup counts toward the chunk.
        if depth < layer - 1:
            depth = advance_support(head_rows, total_width, head, prefixes, depth)
            if depth < 0:
                break
            first = support[pair_depth]
            visited += 1
            prefix = prefixes[pair_depth]
            bound = get_completion_bound(
                table_bytes, offsets, block_bits, first, prefix[label_end:], limit
            )
            if bound >= limit:
                support[pair_depth] = num_rows
                continue

            # Otherwise add row first to the prefix, and try every later row as the last row.  The
            # last rows read only the words and labels of the prefix.
            for index in range(label_end):
                prefixes[layer - 1, index] = prefix[index] ^ step_rows[first, index]
            depth = layer - 1
            support[depth] = first + 1
            continue

        # Visit the remaining supports that differ from this one only in their last row.
        stop = min(num_rows, support[depth] + chunk_size - visited)
        if _visit_last_rows(
            step_rows,
            prefixes[depth],
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

        # Record the visited rows, and after the last row, return to the next first row.
        visited += stop - support[depth]
        support[depth] = stop
        if stop == num_rows:
            depth = pair_depth
            support[depth] += 1
    return depth, False


# Inlined although it is not small: counts call it once for each prefix in their loop over words,
# with many arguments, several of them arrays.
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

    Each row holds ``width`` machine words of its word, then those of its quotient label up to
    ``label_end``.  The arguments are otherwise those of ``_count_chunk``, and ``word`` and
    ``pivots`` are scratch space.  Returns whether the visit stopped at a nontrivial word lighter
    than ``target``.
    """
    for row in range(start, stop):
        weight = np.int64(0)
        for index in range(width):
            word[index] = prefix[index] ^ rows[row, index]
            weight += popcount_word(word[index])
            # A heavier word is rejected below, so its unwritten machine words are never read.
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

    That set owns the word, and visits it first (``brouwer_zimmermann.md#sec-counting``).
    """
    for other in range(len(extractors)):
        if other == set_index:
            continue
        # A set that visits the word in an earlier layer, or in the same layer with a smaller index,
        # visits it before this set.
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
    # Counts compute this rank for every word that they count, so this does not reuse
    # ``qldpc._binary.get_rank_packed``.  That function copies its input, so the masked rows would
    # be copied twice, and it tests the columns one at a time for a pivot until the rank reaches the
    # number of rows.  Here each masked row meets only the pivot bits of the rows before it.
    pivot_words = np.empty(len(rows), dtype=np.int64)
    pivot_bits = np.empty(len(rows), dtype=np.uint64)
    rank = 0
    for row in rows:
        current = row & ~mask if complement else row & mask
        # Each pivot row was reduced by the earlier ones when it was stored, so eliminating them in
        # order clears every pivot bit.
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
