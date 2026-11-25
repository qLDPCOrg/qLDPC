# SPDX-License-Identifier: Apache-2.0

"""Tables of the Brouwer-Zimmermann coset searches: bounds of steps, stopping steps, and tails.

Sections cited in this module are those of ``brouwer_zimmermann.md`` and
``brouwer_zimmermann_code.md``.  Section 4.1 says how the coset searches use these tables.  Threads
that answer queries at once read them, so a change here must keep the requirements on threads under
"Keep coset queries reproducible" in Section 5.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from qldpc._binary import get_num_words, int_to_words, pack_bits, unpack_bits
from qldpc._compilation import always_compiled

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
    npt.NDArray[np.int64],
    npt.NDArray[np.uint8],
    npt.NDArray[np.int64],
    int,
]


class CosetSearchTables(NamedTuple):
    """The tables of ``_search_coset`` in ``coset.py`` for one BZ coset search.

    The search runs on embedded words.  The embedded word of a Pauli error is its packed Pauli for
    the CSS weight, and its packed Pauli followed by ``x xor z`` for the physical weight (see
    ``symplectic_to_hamming`` in ``_code_matrices.py`` of this package).  An X or Z part is its own
    embedded word.

    Attributes:
        inverse: Row ``j`` is the packed embedded word of a word whose syndrome has only bit ``j``
            set.
        generators: For each set, or search basis, of a basis family of the code, its packed rows as
            embedded words (Section 4.4).  The last set of a disjoint family may be partial.
        pivots: The pivot columns of each set, padded with -1 beyond its rank.
        planning: The planning buffer of the search with this family (see ``_get_planning``).
            Section 4.1 says which queries write its cached stopping steps.
        x_mask: Selects the bits of a packed embedded word that form its X part, with one machine
            word for each machine word of the embedded words.  Only searches of Pauli errors by
            CSS weight scan with it, but every search reads it when it weighs the word of a
            syndrome and the offsets of layer zero, or their tails.
    """

    inverse: npt.NDArray[np.uint64]
    generators: npt.NDArray[np.uint64]
    pivots: npt.NDArray[np.int64]
    planning: PlanningBuffer
    x_mask: npt.NDArray[np.uint64]


class CosetTails(NamedTuple):
    """The tails with which a BZ coset search weighs the words of a family, and their tables.

    The tail of an embedded word on a set keeps every column but the pivots of the set, and a word
    of a coset weighs the number of its rows below the rank of the set plus the weight of its tail
    (Section 3.5).  The tails are packed, and zero-padded to the width of the set of least rank.

    Attributes:
        tails: The tails of the rows of each set, with shape ``(num_sets, num_rows, num_words)``.
        inverse_tails: The tails of the offsets of the rows of ``CosetSearchTables.inverse`` on
            each set, with shape ``(num_sets, num_syndrome_bits, num_words)``.  Reducing a word to
            the offset of a set is linear (see ``_reduce_to_offset`` in ``coset.py``), so a query
            sums these tails for the set bits of its syndrome to find the tail of its offset on the
            set, without gathering columns or reading pivots.
        ranks: The rank of each set.
        completion_tables: The completion tables of the tails of the sets that have them, one after
            another (see ``completion_bounds.CompletionTables``).
        layout: A row for each set: the start of its tables in ``completion_tables``, their number
            of blocks ``b``, zero for a set without tables, and their ``b + 1`` block offsets,
            padded with zeros.  One array holds every set, since each argument of a compiled call
            adds to the time of every query.
        block_bits: The number of columns of the blocks of the tables.
    """

    tails: npt.NDArray[np.uint64]
    inverse_tails: npt.NDArray[np.uint64]
    ranks: npt.NDArray[np.int64]
    completion_tables: npt.NDArray[np.uint8]
    layout: npt.NDArray[np.int64]
    block_bits: int


def get_search_tables(
    instance: affine_instance.AffineInstance, inverse: npt.NDArray[np.uint8], num_x_columns: int
) -> CosetSearchTables:
    """Build the tables of a BZ coset search for the cosets of the code of a coset instance.

    ``inverse`` holds embedded words whose syndromes have one bit set, and the first
    ``num_x_columns`` columns of an embedded word form its X part.  The tables round the bounds of
    each coset to its parity if the code is even and some row of ``inverse``, and hence some coset,
    is odd (Section 4.1).  If every coset is even, which happens only in the search of Pauli errors
    by physical weight, rounding would change none of its halved bounds (Section 3.2).
    """
    width = instance.generators.shape[2]
    x_mask = int_to_words((1 << num_x_columns) - 1, width)

    # The rows of any one set span the code, so the first set tells whether the code is even.
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

    ``inverse`` holds embedded words whose syndromes have one bit set, with every column.
    ``record`` is the earlier record of the family, or None.  The function adds the completion
    tables of each set of ``sets`` that has none, unless they would be empty or exceed
    ``max_table_bytes`` bytes (see ``completion_bounds.get_table_bytes_within``).  A record built
    anew has blocks of ``block_bits`` columns.

    Returns:
        ``record`` itself if it is given and the function adds no tables, and otherwise a new
        record, whose ``completion_tables`` copies the earlier tables before the new ones (Section
        4.1).
    """
    # Build the tails of the family once, with a layout of no tables.
    if record is None:
        num_sets, num_rows, width = instance.generators.shape
        num_columns = inverse.shape[1]
        rows = unpack_bits(instance.generators.reshape(-1, width), num_columns)
        rows = rows.reshape(num_sets, num_rows, num_columns)
        tail_width = get_num_words(num_columns - int(instance.ranks.min()))
        tails = np.zeros((num_sets, num_rows, tail_width), dtype=np.uint64)
        inverse_tails = np.zeros((num_sets, len(inverse), tail_width), dtype=np.uint64)
        for set_index, (pivots, rank) in enumerate(
            zip(instance.pivots, instance.ranks, strict=True)
        ):
            # Each row of the set is zero on the pivots of the others, so a word reduces to its
            # offset by adding the row of each pivot on which it has a one.
            set_rows = rows[set_index, :rank]
            selected = inverse[:, pivots[:rank]].astype(np.int64)
            offsets = ((inverse + selected @ set_rows) % 2).astype(np.uint8)
            columns = np.delete(np.arange(num_columns), pivots[:rank])
            set_width = get_num_words(len(columns))
            tails[set_index, :, :set_width] = pack_bits(rows[set_index][:, columns])
            inverse_tails[set_index, :, :set_width] = pack_bits(offsets[:, columns])
        # The tables of a set have at most one block for each block_bits columns of its tails, and
        # block_bits divides the 64 bits of a word.
        max_blocks = 64 * tail_width // block_bits
        layout = np.zeros((num_sets, max_blocks + 3), dtype=np.int64)
        record = CosetTails(
            tails,
            inverse_tails,
            instance.ranks,
            np.zeros(0, np.uint8),
            layout,
            block_bits,
        )

    # Size the tables of each set that has none yet, and keep those that fit.
    sizes = {
        set_index: completion_bounds.get_table_bytes_within(
            record.tails[set_index], record.block_bits, max_table_bytes
        )
        for set_index in sets
        if not record.layout[set_index, 1]
    }
    sizes = {set_index: size for set_index, size in sizes.items() if size}
    if not sizes:
        return record

    # Copy the earlier tables into an array with room for the new ones, and build the new tables in
    # place there, so that the build holds them once.
    start = len(record.completion_tables)
    completion_tables = np.empty(start + sum(sizes.values()), dtype=np.uint8)
    completion_tables[:start] = record.completion_tables
    layout = record.layout.copy()
    for set_index, size in sizes.items():
        tables = completion_bounds.CompletionTables.build(
            record.tails[set_index], record.block_bits, out=completion_tables[start : start + size]
        )
        num_blocks = len(tables.offsets) - 1
        layout[set_index, : num_blocks + 3] = [start, num_blocks, *tables.offsets]
        start += size
    return record._replace(completion_tables=completion_tables, layout=layout)


@always_compiled
def get_planning_table(planning: PlanningBuffer, parity: int) -> npt.NDArray[np.int64]:
    """The table of a planning buffer for a coset whose Hamming weights have parity ``parity``.

    A buffer with two tables, one for each parity, belongs to an even code, and a buffer with one
    table gives it for every coset (see ``_get_planning``).
    """
    return planning[parity] if len(planning) == 2 else planning[0]


@always_compiled
def get_stopping_step(table: npt.NDArray[np.int64], max_weight: int) -> tuple[int, int, int]:
    """Find the step at which a coset query stops, and cache it in a table of the planning buffer.

    The stopping step is the first step whose bound exceeds ``max_weight``: the step at which
    ``stop_num_completed`` sets complete layer ``stop_layer``.  The function reuses the cached step
    ``(w, j)`` if its bound exceeds ``max_weight`` and entry ``[w, j - 1]``, the bound of the step
    before it, does not.  Otherwise ``certificates.find_stopping_step`` finds the step, and the
    function caches it.  A negative ``max_weight`` gives the first step.  The largest native
    integer, the only ``max_weight`` that no bound exceeds, gives the last.  Section 4.1 says which
    queries share a buffer, and why their writes of cached steps are safe.

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
    Unlike ``get_stopping_step``, this function caches no step, which leaves the cached steps to the
    queries of ``_search_coset`` in ``coset.py``.
    """
    return certificates.find_stopping_step(bounds, target - 1, 0, len(bounds) - 1)


def _get_planning(bound_tables: Sequence[npt.NDArray[np.int64]]) -> PlanningBuffer:
    """Build the planning buffer of a coset search from tables of the bounds of its steps.

    The buffer stacks a table for each table of bounds: one for each parity ``p``, for the cosets
    whose Hamming weights have parity ``p``, if the search rounds to parity (see
    ``get_search_tables``), and otherwise one for every coset.  Each table has one more row than its
    bounds.  Entry ``[w, j]`` of the bounds is a bound on the words left unvisited once the first
    ``j`` sets have completed layer ``w`` and the others layer ``w - 1``, so column zero holds the
    bound before the first step of a row.  The bounds never change.  The last row holds the layer
    and the number of completed sets of the stopping step that ``get_stopping_step`` returned last,
    initially the first step ``(0, 1)``.  ``get_stopping_step`` reads and writes it, and
    ``coset._get_own_planning`` resets it in each copy for a worker.  One array holds the bounds and
    the cached steps, since each argument of a compiled call adds to the time of every query.
    """
    planning = np.zeros(
        (len(bound_tables), len(bound_tables[0]) + 1, bound_tables[0].shape[1]), dtype=np.int64
    )
    planning[:, :-1] = bound_tables
    planning[:, -1, 1] = 1
    return planning
