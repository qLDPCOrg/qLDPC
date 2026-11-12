# SPDX-License-Identifier: Apache-2.0

"""Tables of the Brouwer-Zimmermann coset searches: bounds of steps, stopping steps, and tails."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._binary import pack_bits, unpack_bits
from qldpc._compilation import always_compiled

from .._cosets import int_to_words
from .._information_sets import certificates, completion_bounds
from . import affine_instance

# The planning buffer of a coset search: its tables of the bounds of its steps, each with one more
# row that caches a stopping step (see ``_get_planning``).
PlanningBuffer = npt.NDArray[np.int64]

# The fields of a ``CosetTails`` record, which the compiled searches take as the record itself or
# as a plain tuple.
TailFields = tuple[
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint64],
    npt.NDArray[np.uint8],
    npt.NDArray[np.int64],
    int,
]


class CosetSearchTables(NamedTuple):
    """The tables of ``_search_coset`` for one coset search of a ``CosetLeaderSearch``.

    The search runs on embedded words.  For a search of Pauli errors, the embedded word of an error
    is its packed Pauli for the CSS weight, and its packed Pauli followed by ``x xor z`` for the
    physical weight (see ``symplectic_to_hamming``).  For a search of X or Z parts, the embedded
    word of a part is the part itself.

    Attributes:
        inverse: Row ``j`` is the packed embedded word of a word whose syndrome has only bit ``j``
            set.
        generators: The packed rows of each set of a family of information sets of the embedded
            generators of the code (see ``coset_routes``).  The last set of a disjoint family may
            be partial, of rank below the dimension of the code.
        pivots: The pivot columns of each set, padded with -1 beyond its rank.
        planning: The planning buffer of the search with this family: its tables of the bounds of
            its steps, each with one more row that caches a stopping step (see ``_get_planning``).
            For a nonnegative maximum weight, ``get_stopping_step`` reuses the cached step of a
            table exactly while it stays the first step whose bound exceeds that weight, and
            replaces it otherwise.  Queries write only the rows of cached steps, and oracles that
            share a cached search share them.
        x_mask: Selects the bits of a packed embedded word that count as its X part, with one
            machine word for each machine word of the embedded words.  Only searches of Pauli
            errors by CSS weight use the X part, and their layers scan with it, but every search
            reads the mask when it weighs the word of a syndrome, and a search of whole words also
            reads it when it weighs the offsets of layer zero.
    """

    inverse: npt.NDArray[np.uint64]
    generators: npt.NDArray[np.uint64]
    pivots: npt.NDArray[np.int64]
    planning: PlanningBuffer
    x_mask: npt.NDArray[np.uint64]


class CosetTails(NamedTuple):
    """The tails with which ``_search_coset`` weighs the words of a family by tails, and its tables.

    The tail of an embedded word on a set of the family holds every column of the word but the
    pivot columns of the set, including columns on which no row of the set has a one.  The tails are
    packed, and padded with zeros to one common width, that of the set of least rank.  A word of a
    coset is the offset of a set, which is zero on its pivot columns, plus rows of the set.  Each
    row below the rank of the set has a one on its own pivot column, where the other rows are zero.
    So the word weighs the number of its rows below the rank plus the weight of its tail (see
    Section 3.5 of ``brouwer_zimmermann.md``).

    Attributes:
        tails: The tails of the rows of each set, with shape ``(num_sets, num_rows, num_words)``.
        inverse_tails: The tails of the rows of ``inverse`` of ``CosetSearchTables`` on each set,
            with shape ``(num_sets, num_syndrome_bits, num_words)``.  The tail of the word of a
            syndrome on a set is the sum of the rows for the set bits of the syndrome.  These rows
            can be derived from ``inverse``, but a query that sums them directly finds the tails of
            its offsets faster than one that gathers the columns of each tail from the full word.
        completion_tables: The completion tables of the tails of the sets that have them, one after
            another (see ``completion_bounds.CompletionTables``).
        layout: Where the tables of each set lie in ``completion_tables``, with a row for each set:
            the start of its tables, their number of blocks ``b``, and their ``b + 1`` block
            offsets, padded with zeros.  A set without tables has zero blocks.  The compiled search
            takes the layout of every set in one array, since each argument of a compiled call adds
            to the time of every query.
        block_bits: The number of columns of the blocks of the tables.
    """

    tails: npt.NDArray[np.uint64]
    inverse_tails: npt.NDArray[np.uint64]
    completion_tables: npt.NDArray[np.uint8]
    layout: npt.NDArray[np.int64]
    block_bits: int


def get_search_tables(
    instance: affine_instance.AffineInstance, inverse: npt.NDArray[np.uint8], num_x_columns: int
) -> CosetSearchTables:
    """Build the tables of ``_search_coset`` for the cosets of the code of a coset instance.

    ``inverse`` holds embedded words whose syndromes have one bit set, and the first
    ``num_x_columns`` columns count as the X part of an embedded word.

    The search rounds its bounds to the parity of each coset if every word of the code has even
    Hamming weight and some coset is odd.  Then the words of each coset share a parity (see
    ``get_coset_instance``).  The word of a syndrome sums rows of ``inverse``, so some coset is odd
    exactly when some row is.  If the code and every row of ``inverse`` are even, every coset is
    even.  This happens only in the search of Pauli errors by physical weight, whose bounds are
    halved, so that rounding them up to even would change none.
    """
    width = instance.generators.shape[2]
    x_mask = int_to_words((1 << num_x_columns) - 1, width)

    # Round to the parity of each coset if the code is even and some coset is odd.  The rows of any
    # one information set span the code.
    even_code = not np.any(np.bitwise_count(instance.generators[0]).sum(axis=1) % 2)
    has_odd_coset = bool(np.any(np.count_nonzero(inverse, axis=1) % 2))
    parities = (0, 1) if even_code and has_odd_coset else (None,)
    return CosetSearchTables(
        pack_bits(inverse),
        instance.generators,
        instance.pivots,
        _get_planning([instance.get_weight_bounds(parity) for parity in parities]),
        x_mask,
    )


def get_coset_tails(
    instance: affine_instance.AffineInstance,
    sets: list[int],
    record: CosetTails | None,
    *,
    inverse: npt.NDArray[np.uint8],
    block_bits: int,
    max_table_bytes: int,
) -> CosetTails:
    """Build the record of the tails of the family of a coset instance, with tables of some sets.

    ``inverse`` holds embedded words whose syndromes have one bit set, with every column of the
    embedded words.  The tails of each set keep every column but its pivots, and are zero-padded to
    the width of the set of least rank, which keeps the most columns.  A new record has tables on
    blocks of ``block_bits`` columns.  Given the ``record`` of the family, the function keeps its
    tails, tables, and blocks rather than build them again.  The record that it returns also has the
    completion tables of each set of ``sets``, unless they would exceed ``max_table_bytes`` bytes or
    be empty (see ``completion_bounds.build_tables_within``).  It holds the tables of each set once,
    and is ``record`` itself if it adds no tables.
    """
    # Build the tails of the family once, with a layout of no tables.
    if record is None:
        num_sets, num_rows, width = instance.generators.shape
        num_columns = inverse.shape[1]
        rows = unpack_bits(instance.generators.reshape(-1, width), num_columns)
        rows = rows.reshape(num_sets, num_rows, num_columns)
        tail_width = (num_columns - int(instance.ranks.min()) + 63) // 64
        tails = np.zeros((num_sets, num_rows, tail_width), dtype=np.uint64)
        inverse_tails = np.zeros((num_sets, len(inverse), tail_width), dtype=np.uint64)
        for set_index, (pivots, rank) in enumerate(
            zip(instance.pivots, instance.ranks, strict=True)
        ):
            columns = np.delete(np.arange(num_columns), pivots[:rank])
            set_width = (len(columns) + 63) // 64
            tails[set_index, :, :set_width] = pack_bits(rows[set_index][:, columns])
            inverse_tails[set_index, :, :set_width] = pack_bits(inverse[:, columns])
        # The tables of a set have at most one block for each block_bits columns of its tails, and
        # block_bits divides the 64 bits of a word.
        max_blocks = 64 * tail_width // block_bits
        layout = np.zeros((num_sets, max_blocks + 3), dtype=np.int64)
        record = CosetTails(tails, inverse_tails, np.zeros(0, np.uint8), layout, block_bits)

    # Append the tables of each set that has none yet, if they fit, and lay them out.  The routes
    # keep only the latest record of a family, so the earlier tables are held once.
    layout = record.layout.copy()
    added = []
    start = len(record.completion_tables)
    for set_index in sets:
        if layout[set_index, 1]:
            continue
        tables = completion_bounds.build_tables_within(
            record.tails[set_index], record.block_bits, max_table_bytes
        )
        if tables is not None:
            num_blocks = len(tables.offsets) - 1
            layout[set_index, : num_blocks + 3] = [start, num_blocks, *tables.offsets]
            added.append(tables.tables)
            start += len(tables.tables)
    if not added:
        return record
    return record._replace(
        completion_tables=np.concatenate([record.completion_tables, *added]), layout=layout
    )


@always_compiled
def get_planning_table(planning: PlanningBuffer, parity: int) -> npt.NDArray[np.int64]:
    """The table of a planning buffer for a coset whose Hamming weights have parity ``parity``.

    The Hamming weights are those of the embedded words of the coset.  A buffer with a table for
    each parity belongs to an even code, whose cosets each have Hamming weights of one parity, and
    gives the table of the parity (see ``_get_planning``).  A buffer with one table gives it for
    every coset.
    """
    return planning[parity] if len(planning) == 2 else planning[0]


@always_compiled
def get_stopping_step(table: npt.NDArray[np.int64], max_weight: int) -> tuple[int, int, int]:
    """Find the step at which a coset query stops, and cache it in a table of the planning buffer.

    The stopping step is the first step whose bound in the table exceeds the maximum weight: the
    step at which ``stop_num_completed`` sets complete layer ``stop_layer``.  A query stops there,
    unless it finds a light enough word sooner.  The bounds of the steps do not decrease in their
    order.  Column zero of each row of bounds holds the bound before the first step of the row,
    which is zero in row zero.  So the bound of the step before step ``(w, j)`` is entry
    ``[w, j - 1]``, and for a nonnegative maximum weight, step ``(w, j)`` is the stopping step
    exactly if its bound exceeds the maximum weight and entry ``[w, j - 1]`` does not.  The step
    cached in the table (see ``_get_planning``) is reused if it passes this check.  Otherwise,
    ``certificates.find_stopping_step`` finds the stopping step by binary search, and it is cached
    in its place.  A negative maximum weight fails the check, and the binary search returns the
    first step.  If no bound exceeds the maximum weight, which happens only when the maximum weight
    reaches the largest native integer, the binary search returns the last step.

    Oracles that share a cached search share its planning buffer.  Compiled calls hold the global
    interpreter lock, since numba compiles them without ``nogil``, so calls interleave only when
    these functions run as plain Python in several threads, or in free-threaded Python.  The layer
    and the number of completed sets of a step are written as two entries, so a read may pair the
    layer of one step with the number of completed sets of another.  Any such pair indexes the
    bounds, which never change, and it is checked against them before it is used.

    Returns:
        ``stop_layer``, ``stop_num_completed``, and the bound of that step.
    """
    bounds = table[:-1]
    layer, num_completed = table[-1, 0], table[-1, 1]
    is_stopping_step = (
        max_weight < bounds[layer, num_completed] and bounds[layer, num_completed - 1] <= max_weight
    )
    if not is_stopping_step:
        layer, num_completed = certificates.find_stopping_step(
            bounds, max_weight, 0, len(bounds) - 1
        )
        table[-1, 0], table[-1, 1] = layer, num_completed
    return layer, num_completed, int(bounds[layer, num_completed])


@always_compiled(inline=True)
def get_first_step_reaching(bounds: npt.NDArray[np.int64], target: int) -> tuple[int, int]:
    """The first step whose bound in a table of bounds of steps reaches ``target``.

    It is the stopping step of ``certificates.find_stopping_step`` for the limit ``target - 1``.
    Unlike ``get_stopping_step``, this function caches no step, so the cached steps of the planning
    buffer serve only the queries of ``_search_coset``.
    """
    return certificates.find_stopping_step(bounds, target - 1, 0, len(bounds) - 1)


def _get_planning(bound_tables: Sequence[npt.NDArray[np.int64]]) -> PlanningBuffer:
    """Build the planning buffer of a coset search from tables of the bounds of its steps.

    The compiled search takes the bounds and its cached stopping steps in this one array, since each
    argument of a compiled call adds to the time of every query.  The buffer stacks one table for
    each given table of bounds.  A search that rounds its bounds to parity (see
    ``get_search_tables``) gives two tables: table ``p`` holds the bounds of the cosets whose words
    have Hamming weights of parity ``p`` (see ``get_planning_table``).  Other searches give one
    table, for every coset.  Each table of the buffer has one more row than the bounds:

    - Every row but the last holds the bounds of ``AffineInstance.get_weight_bounds``: entry
      ``[w, j]`` bounds the weight of every word that the search has not visited once the first
      ``j`` sets have completed layer ``w`` and the others layer ``w - 1``.  So column zero of each
      row holds the bound before the first step of that row.  These rows never change, and code
      reads them as the view ``table[:-1]``.
    - The last row holds no bounds.  Its first two entries are the layer and the number of completed
      sets of the stopping step that ``get_stopping_step`` returned last for this table, or of the
      first step, ``(0, 1)``, before its first call, and its other entries are zero.  Only
      ``get_stopping_step`` reads and writes this row.
    """
    planning = np.zeros(
        (len(bound_tables), len(bound_tables[0]) + 1, bound_tables[0].shape[1]), dtype=np.int64
    )
    planning[:, :-1] = bound_tables
    planning[:, -1, 1] = 1
    return planning
