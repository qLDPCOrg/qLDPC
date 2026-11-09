# SPDX-License-Identifier: Apache-2.0

"""Settings and constants that tune the speed and memory of exact distance searches and counts.

Exact distances and logical-operator counts choose between routes, such as exhaustive enumeration
and searches of disjoint or overlapping information sets, and bound their batches, memory, and
caches.  :class:`SearchTuning` collects the values that tune a search, from the thresholds that
choose a route to the coefficients of cost models and the sizes of batches.  The page
:doc:`/exact_search_tuning` of the documentation maps routes and symptoms to these values.  Section
4.4 of ``brouwer_zimmermann.md`` says where each heuristic of a Brouwer-Zimmermann search uses them,
and Section 3.5 of ``connected_support.md`` where the connected-support search uses its field.

This module is private.  Its names, fields, and defaults can change in any release, without a
deprecation period or any other warning, so code that uses it should pin the exact version of
qLDPC that it was written for.  It is meant for contributors, for benchmarks, and for experienced
users of very costly searches who have measured that a different tuning helps them.  The defaults
suit typical searches.

What tuning can and cannot change:

- Tuning changes only how long a search takes and how much memory it uses.  It never changes an
  exact distance, a distance limited by ``max_weight``, or a count of logical operators.
- With an early exit (``cutoff`` of two or more), a distance search returns the weight of the first
  nontrivial word that it finds at or below ``cutoff``, and which word comes first depends on
  tuning.  The result is still an upper bound between the distance and ``cutoff``, as documented by
  :func:`~qldpc.codes.distance.get_distance_quantum`.
- Codes and counters remember their results, and some intermediate results are cached for the whole
  process (see ``BINOMIAL_CACHE_SIZE``, ``CSS_SEARCH_CACHE_SIZE``, ``KRAWTCHOUK_CACHE_SIZE``, and
  ``LOCAL_CLIFFORD_CACHE_SIZE``), so a later search with another tuning may not search again.  To
  time a tuning, run each search in a new process, after a first search that loads or compiles code.

To change the tuning, change the fields of the active tuning within a ``with`` block::

    from sympy.abc import x, y

    from qldpc import codes
    from qldpc.codes.distance import _tuning

    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    with _tuning.use_tuning(max_information_sets=8, lp_minimum_work=0):
        distance = code.get_distance_exact()

Each search reads the active tuning once, when it starts, and uses it throughout.  The active tuning
is a context variable (see the ``contextvars`` module), which ``use_tuning`` changes in the current
context until the block exits.  Copies of the context, such as asyncio tasks and the functions run
by ``asyncio.to_thread``, keep the tuning of the copy, and forked worker processes inherit it.  New
threads start with the default tuning, unless Python lets them inherit the context of the thread
that starts them (see ``sys.flags.thread_inherit_context``), and spawned worker processes start with
the default tuning.  To give workers a tuning, enter ``use_tuning`` in the function that each worker
runs.

To reproduce an experiment, save ``dataclasses.asdict(_tuning.get_tuning())`` within the thread or
process that runs the searches, and restore it with ``use_tuning(**saved)``.  Also save the version
of qLDPC or, for a checkout of its source, the commit and any differences of the source from it,
which fix the defaults and the constants.  Field values never make compiled code specialize again,
but a tuning that selects a route for the first time compiles the code of that route when it first
runs it, so warm every route that a comparison may select before timing it.

Which values are fields:

Every value that changes only how long a search takes or how much memory it uses, and that the
search can read when it starts and pass down to the code that uses it, is a field of
:class:`SearchTuning`, whether it expresses a trade-off, such as a memory limit, or a calibration,
such as a coefficient of a cost model.  So a field can differ between two searches in one process,
and a saved tuning records every field.  The arguments of the public functions that start searches
say instead what to compute and by which method, such as ``cutoff``, ``max_weight``, and ``method``.
The entry points that read the tuning are the distance searches, the counts of logical operators,
and the oracles of ``CosetLeaderSearch``, each when it starts or is built.  Only values that qLDPC
fixes before any search starts are constants: ``MAX_VALUE``, and the sizes of caches, which it fixes
when it is imported.  Change those by editing this module, and start a new process.

.. _tuning-retuning:

Retuning:

Compare candidate values on warmed searches in new processes, single-threaded, in paired runs that
alternate their order, and record the time and peak memory of each run; repeat runs to measure
their spread.  The comment of each field says what to measure to retune it.

The fields of completion tables (Section 3.9 of ``brouwer_zimmermann.md``) were chosen on an Apple
M3 Max, with layers of one information set of ``4 * 10**7`` to ``5 * 10**10`` supports from two
data sets:

- The width data set holds layers of weights 4 to 6 of both sectors of
  ``BBCode({x: 12, y: 12}, x**3 + y**2 + y**7, y**3 + x + x**2)``, the [[288, 12, 18]] code, and of
  ``QTCode.random(CyclicGroup(7), HammingCode(3), seed=1)``, and of two lifted-product codes of
  3 x 4 random monomials of ``CyclicGroup(13)``, drawn with NumPy's default generator from seeds 5
  and 8.
- The cost data set holds nine layers of the X sector of that bivariate bicycle code and of the
  [[232, 62, 12]], [[170, 8, 20]], and [[276, 98, 14]] pair-partition codes, which qLDPC does not
  construct; layers of similar size from codes that qLDPC constructs can stand in for them.  On
  five other layers, the cost model fitted to it predicted cost ratios at most 0.12 above, and 0.02
  below, the measured ones.

To refit the cost model of completion tables, which ``select_completion_tables`` and
``_estimate_build_cost`` in ``_search/completion_bounds.py`` compute:

1. Time layers of one information set with tables disabled
   (``completion_bound_max_table_bytes=0``), forced (``completion_bound_minimum_layer_work=0`` and
   ``completion_bound_max_cost_percent=MAX_VALUE``), and at the defaults.  For each layer, record
   the share of its supports that tables skipped and its number of lookups, which
   ``_scan_one_information_set`` in ``_search/brouwer_zimmermann.py`` returns for each call, and
   time ``CompletionTables.build`` separately.
2. Express times in scanned supports, the time per support of the arm without tables.
3. Fit the coefficients that the cost model reads, ``completion_bound_lookup_block_cost``,
   ``completion_bound_lookup_block_cost_per_mib``, ``completion_bound_kept_first_row_ratio``,
   ``completion_bound_table_byte_cost``, and ``completion_bound_build_step_cost``, to the measured
   lookups, scans, and builds, and round the coefficients of builds up.

Accept values if, at the defaults, no layer is slower than the faster of the other two arms by more
than the spread of repeated runs.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
import math
import numbers
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ._binary import as_integer

#: The largest value of an integer field.  Searches add and multiply some fields in compiled code,
#: whose integers have 64 bits, and a larger value would also compile that code again for another
#: integer type.
MAX_VALUE = 2**62

####################################################################################################
# tuning

# The widths of the blocks of completion tables that searches accept: the powers of two up to 16,
# which divide the 64 bits of a packed word.  Wider blocks would need tables of ``2**width`` bytes
# for each row of a basis.
_BLOCK_WIDTHS = {"choices": (1, 2, 4, 8, 16)}


@dataclass(frozen=True, kw_only=True)
class SearchTuning:
    """Settings of exact searches and counts: thresholds, limits, coefficients, and batch sizes.

    Work estimates count candidate/set evaluations.  One such evaluation visits one combination of
    the rows of one information set, and checks the word that the combination gives.  Section 3.6
    of ``brouwer_zimmermann.md`` defines the estimate, which assumes pessimistically that a search
    finds no lighter word before it certifies its best one.  Estimates are not runtime guarantees.
    Coefficient layer ``w`` of a search visits the combinations of ``w`` rows of each of its
    information sets, so later layers cost more.

    Every integer field is a Python integer between zero and ``MAX_VALUE``, unless its comment
    states narrower bounds, and every float field is a finite, non-negative float.

    Counts with the BZ method always scan information sets, rather than enumerate every word.  The
    distance searches that find their default weights consult both ``allow_enumeration`` fields.

    The comment above each field says what it decides and in what unit, what raising or lowering it
    does, how its default was chosen, and what to measure to retune it (see :ref:`Retuning
    <tuning-retuning>`, which also describes the width and cost data sets).
    """

    # Fields of exhaustive enumeration.

    #: Whether a distance search may enumerate every nontrivial word, instead of building
    #: information sets, when its basis is small (see ``enumeration_max_dimension`` and
    #: ``enumeration_setup_cost_factor``).  False makes small searches build information sets, which
    #: can be much slower; they may still enumerate later (see ``allow_enumeration_by_work``).  A
    #: switch for tests that force a route, not a calibration.  Defaults to True.
    allow_enumeration_by_size: bool = True

    #: Whether a distance search may enumerate every nontrivial word after choosing its information
    #: sets, when there are no more such words than the estimated work of searching those sets.
    #: False makes such searches scan information sets even when enumeration is cheaper.  A switch
    #: for tests that force a route, not a calibration.  Defaults to True.
    allow_enumeration_by_work: bool = True

    #: A distance search enumerates every nontrivial word, instead of building information sets, if
    #: its basis has at most this many rows.  Raising it enumerates larger bases, whose number of
    #: words doubles with each row; lowering it makes small searches pay for setup.  Defaults to 20;
    #: how it was measured is not recorded.  To retune it, time both routes on bases near the
    #: crossover.
    enumeration_max_dimension: int = 20

    #: A distance search of more than ``enumeration_max_dimension`` and fewer than 63 basis rows
    #: also enumerates every word if ``2**rows`` is at most this factor times its numbers of rows
    #: and columns, an estimate of the cost of setting up its information sets.  Raising it
    #: enumerates more long codes of low dimension; lowering it builds information sets for more of
    #: them.  Defaults to 128; how it was measured is not recorded.  To retune it, time both routes
    #: on long codes near the crossover.
    enumeration_setup_cost_factor: int = 128

    #: The exponent of the batches of exhaustive enumeration, which handles at most
    #: ``2**enumeration_batch_exponent`` words at a time in a few arrays of at most
    #: ``2**(enumeration_batch_exponent + 3)`` bytes each, and checks for a keyboard interrupt
    #: between batches.  Raising it can speed up enumeration, at the cost of memory; lowering it
    #: saves memory and makes interrupts prompter, but adds overhead.  At most 62.  Defaults to 15,
    #: chosen by hand.  To retune it, time enumerations of bases of 16 to 24 rows, and keep the
    #: smallest value beyond which they get no faster.
    enumeration_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    # Fields of information-set families.

    #: The most information sets in a family of overlapping sets, or in a family found by matroid
    #: partitioning.  More sets can skip costly layers of hard searches, but setup grows roughly
    #: with the square of this limit and dominates cheap searches; zero disables both kinds of
    #: family.  Defaults to 16; how it was chosen is not recorded.  To retune it, compare the total
    #: times of hard searches.
    max_information_sets: int = 16

    #: How many random column orders, in a fixed sequence, a search tries when the natural order
    #: packs fewer full disjoint information sets than the nonzero columns divided by the rows; it
    #: keeps the best packing.  More orders can find more full sets, which certify sooner, but each
    #: costs a packing of the basis; zero keeps the natural order.  Defaults to 8, chosen by hand.
    #: To retune it, record after how many orders the best packing stops improving, and weigh setup
    #: time against the search time saved.
    disjoint_random_column_orders: int = 8

    #: The estimated work of a search, in candidate/set evaluations, below which it does not try to
    #: replace its disjoint information sets by overlapping ones.  Lowering it tries this on cheaper
    #: searches, which may not recover the setup; zero always tries, and ``MAX_VALUE`` never does.
    #: Defaults to 100,000; how it was chosen is not recorded.  To retune it, time searches near the
    #: threshold with and without the replacement.
    overlapping_minimum_work: int = 100_000

    #: The column orders that building a family of overlapping information sets tries for each set
    #: that it may add; an order that repeats an existing set still counts.  More attempts can reach
    #: a cheaper family, but lengthen setup; zero builds no family.  Defaults to 2, chosen by hand.
    #: To retune it, record how the estimated work of the family falls with more attempts, and weigh
    #: setup time against the search time saved.
    overlapping_attempts_per_set: int = 2

    #: A search builds families by matroid partitioning only if its estimated work is at least this
    #: multiple of the estimated cost of partitioning (see ``partition_cost_factor``), so an attempt
    #: that does not pay off adds at most about ``1 / partition_work_ratio`` to its time.  Lowering
    #: it tries partitioning on cheaper searches; zero always tries, and ``MAX_VALUE`` never does.
    #: Defaults to 10, which bounds the wasted time by 10%.  To retune it, choose the share of time
    #: that an attempt may waste.
    partition_work_ratio: int = 10

    #: The largest coverage, the most information sets that contain any one column, that matroid
    #: partitioning tries; it tries each coverage from two to this value.  Larger values can find
    #: families with more sets, which certify sooner, but each coverage adds to the cost of
    #: partitioning; below two, it builds none.  Defaults to 4, chosen by hand.  To retune it,
    #: record which coverage gives the cheapest family, and weigh the time of each further coverage
    #: against the search time it saves.
    max_partition_coverage: int = 4

    #: The estimated cost of partitioning ``b`` copies of the ``n`` nonzero columns of a basis of
    #: ``k`` rows, in units of ``b * k**2 * n`` candidate/set evaluations.  An underestimate wastes
    #: time on partitioning that cannot repay itself; an overestimate skips families that would save
    #: time.  Defaults to 20, measured on bases of 85 to 156 rows and 127 to 432 columns, which are
    #: not recorded.  To refit it, time ``matroid_partition.get_information_sets_with_coverage`` in
    #: candidate/set evaluations.
    partition_cost_factor: int = 20

    #: The estimated work, in candidate/set evaluations, from which a search of overlapping
    #: information sets also bounds unvisited words by linear programming (Section 3.1 of
    #: ``brouwer_zimmermann.md``).  Each linear program takes milliseconds, which only costly
    #: searches recover.  Lowering it uses them on cheaper searches; zero on every search of
    #: overlapping sets, and ``MAX_VALUE`` on none.  Defaults to 10**9; how it was chosen is not
    #: recorded.  To retune it, time searches near the threshold with and without them.
    lp_minimum_work: int = 1_000_000_000

    #: The largest denominator of the rational dual weights that the certificate from linear
    #: programming takes from the floating-point solution of its solver.  Every positive value is
    #: sound; a smaller one can round away part of the bound, and a larger one lengthens exact
    #: arithmetic.  Defaults to 2**20, chosen by hand.  To retune it, keep the smallest denominator
    #: that loses no certified bound.
    lp_dual_max_denominator: int = dataclasses.field(default=1 << 20, metadata={"minimum": 1})

    #: A distance search that can find automorphisms of its code finds them before its first
    #: coefficient layer of at least this estimated work, in candidate/set evaluations, and then
    #: searches the images of one information set under them if that predicts less work (Section 3.7
    #: of ``brouwer_zimmermann.md``).  Finding them takes milliseconds; lowering it pays that on
    #: cheaper searches, zero before the first layer, and ``MAX_VALUE`` never.  Only the
    #: ``get_distance_exact`` methods of binary quantum codes find automorphisms.  Defaults to
    #: 10**8; how it was chosen is not recorded.  To retune it, time searches with and without
    #: automorphisms.
    automorphism_minimum_layer_work: int = 100_000_000

    # Fields of connected-support searches.

    #: A connected-support search of a quantum code asks for its automorphisms before the first
    #: weight limit at which it expects to have visited this many states, which are supports, over
    #: all limits of one problem (Section 3.5 of ``connected_support.md``).  It also asks before a
    #: first limit that a cutoff raises, unless this is ``MAX_VALUE``.  Lowering it pays the cost
    #: of finding them on cheaper searches; raising it searches more limits from every variable.
    #: Zero asks before the first limit, and ``MAX_VALUE`` never.  Defaults to 100,000, chosen by
    #: hand as about 0.05 s of searching at the rate of that section.  To retune it, time searches
    #: with and without automorphisms.
    automorphism_minimum_states: int = 100_000

    # Fields of completion tables.

    #: The number of supports, choices of rows of an information set, from which a layer that scans
    #: one information set considers completion tables, which skip supports that cannot complete a
    #: word lighter than its best one (Section 3.9 of ``brouwer_zimmermann.md``).  Considering them
    #: takes milliseconds of sampling.  Lowering it considers smaller layers; zero every layer, and
    #: ``MAX_VALUE`` none.  Defaults to 10**8, about 0.2 seconds of scanning on the calibration
    #: machine, so that sampling costs a few percent at most.  To retune it, time layers near the
    #: threshold with tables disabled and forced.
    completion_bound_minimum_layer_work: int = 100_000_000

    #: A layer uses the completion tables that it considers only if their predicted cost, including
    #: building them, is at most this percentage of the cost of scanning without them.  Lowering it
    #: demands larger predicted savings; zero never uses tables, and ``MAX_VALUE`` uses them
    #: wherever they are considered, even where they cost more.  Defaults to 90, a margin for the
    #: error of the cost model.  To retune it, compare predicted and measured costs on the cost data
    #: set.
    completion_bound_max_cost_percent: int = 90

    #: The most bytes of completion tables that each problem of a distance search keeps, the main
    #: memory limit of distance searches; so a joint search of both sectors of a CSS code can keep
    #: this much for each sector.  Tables use blocks of ``completion_bound_block_bits`` columns, or
    #: of ``completion_bound_fallback_block_bits`` if those do not fit, or none.  Zero disables
    #: tables, and ``MAX_VALUE`` removes the limit.  Defaults to 256 MiB, which fits blocks of 16
    #: columns for bases of up to about 500 rows and 140 tail columns.  To retune it, choose the
    #: memory that the tables of one problem may use.
    completion_bound_max_table_bytes: int = 256 * 2**20

    #: The number of tail columns in a block of completion tables, a power of two up to 16.  Wider
    #: blocks skip more supports, but the table of a block has ``2**width`` bytes for each row of
    #: the basis.  Defaults to 16, which on every layer of the width data set of at least ``5 *
    #: 10**8`` supports skipped more supports, in less time, than 8.  To retune it, force each width
    #: on that data set and keep the fastest.
    completion_bound_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

    #: The width of the blocks of completion tables when blocks of ``completion_bound_block_bits``
    #: columns would exceed ``completion_bound_max_table_bytes``.  Blocks of 8 columns need about
    #: 128 times less memory than blocks of 16, and skip fewer supports.  Defaults to 8, which still
    #: saved time on every layer of the width data set of at least ``10**9`` supports.  To retune
    #: it, lower the memory limit until that data set falls back, and compare narrower widths.
    completion_bound_fallback_block_bits: int = dataclasses.field(default=8, metadata=_BLOCK_WIDTHS)

    #: The number of random supports from which a search estimates the share of a layer that
    #: completion tables would skip, within the budget ``completion_bound_sampling_work``; drawing
    #: them takes about 16 bytes for each sample and each row of the basis.  More samples narrow the
    #: estimate, so that tables pay on more of the layers that accept them, but take longer.
    #: Between 1 and ``2**12``.  Defaults to 256, chosen by hand; sampling took 1 to 13 milliseconds
    #: on the cost data set.  To retune it, raise it while sampling stays under a tenth of the time
    #: of the smallest layer considered.
    completion_bound_samples: int = dataclasses.field(
        default=256, metadata={"minimum": 1, "maximum": 2**12}
    )

    #: The fewest samples with which a search trusts its estimate of the share that completion
    #: tables would skip; with fewer within the budget, it declines tables.  At least 1.  Defaults
    #: to 64, so that when every sample skips, the lower end of the confidence interval (see
    #: ``completion_bound_confidence_z``) is 0.94.  To retune it, keep the fewest samples that still
    #: accept tables on the layers of the cost data set that they speed up.
    completion_bound_min_samples: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most work that a search spends bounding sampled supports, in units of one pair of rows on
    #: one block of the tables; summing the first rows of a support costs one unit for each row of
    #: the basis.  Raising it allows more samples on large bases, at more cost; zero takes no
    #: samples, so the search declines tables.  Defaults to ``5 * 10**7``, so that sampling takes
    #: tens of milliseconds on large bases.  To retune it, time sampling on the largest bases of
    #: interest, and keep it under a tenth of the time of the smallest layer considered.
    completion_bound_sampling_work: int = 50_000_000

    #: How many standard errors below the sampled share of skipped supports a search sets its
    #: estimate, the lower end of a Wilson score interval; at most 10.  Raising it declines more
    #: layers where tables would pay; lowering it lets sampling error choose tables that cost more
    #: than they save.  Defaults to 2.0, a conventional confidence of about 98%.  To retune it,
    #: raise it if sampling error made choices slower on the cost data set.
    completion_bound_confidence_z: float = dataclasses.field(
        default=2.0, metadata={"maximum": 10.0}
    )

    #: The cost of one lookup of completion tables for each block of the tables, in scanned
    #: supports, the unit of their cost model: the time to scan one support without tables.  An
    #: underestimate makes searches use tables that cost more than they save; an overestimate makes
    #: them decline tables that would save time.  Defaults to 2.0, fitted to the cost data set, in
    #: which lookups cost 2 to 8 scanned supports per block.  To refit it, see :ref:`Retuning
    #: <tuning-retuning>`.
    completion_bound_lookup_block_cost: float = 2.0

    #: The additional cost of a lookup of completion tables, in scanned supports for each block and
    #: each MiB of tables, since larger tables miss the processor's caches more often.  Raising it
    #: makes searches decline more large tables; lowering it lets them use large tables that may
    #: cost more than they save.  Defaults to 0.04, fitted with
    #: ``completion_bound_lookup_block_cost`` to the cost data set.  To refit it, see :ref:`Retuning
    #: <tuning-retuning>`.
    completion_bound_lookup_block_cost_per_mib: float = 0.04

    #: A multiplier of the predicted number of lookups of completion tables, which is the share of
    #: pairs of a prefix and the first row of its last pair that a scan with tables keeps, divided
    #: by the share of supports that it keeps.  It is below one, since tables most often skip late
    #: first rows, which have few supports each.  Raising it makes tables look costlier, and
    #: lowering it cheaper.  Defaults to 0.35, fitted to the cost data set, whose ratios ranged from
    #: 0.31 to 0.50.  To refit it, see :ref:`Retuning <tuning-retuning>`.
    completion_bound_kept_first_row_ratio: float = 0.35

    #: The cost of building completion tables, in scanned supports for each byte of tables; with
    #: ``completion_bound_build_step_cost``, it predicts the cost of a build.  An underestimate
    #: builds tables that small layers do not repay; an overestimate declines tables that would save
    #: time.  Defaults to 0.1, chosen by hand with the cost of a step to predict builds about 1.5
    #: times slower than measured on the cost data set.  To refit it, see :ref:`Retuning
    #: <tuning-retuning>`.
    completion_bound_table_byte_cost: float = 0.1

    #: The cost of building completion tables, in scanned supports for each step of the
    #: breadth-first searches that fill them, as ``_estimate_build_cost`` in
    #: ``_search/completion_bounds.py`` counts steps.  Raising it makes searches decline tables that
    #: are costly to build; lowering it lets them build tables that small layers do not repay.
    #: Defaults to 0.2, chosen by hand with ``completion_bound_table_byte_cost``.  To refit it, see
    #: :ref:`Retuning <tuning-retuning>`.
    completion_bound_build_step_cost: float = 0.2

    # Fields of batches and memory limits.

    #: A scan of information sets by the BZ method visits at most
    #: ``2**information_set_batch_exponent`` supports and lookups of completion tables, counted
    #: together, in one call of compiled code, which cannot be interrupted.  Lowering it lets a
    #: keyboard interrupt take effect sooner, but returns to Python more often.  At most 62.
    #: Defaults to 15, chosen by hand.  To retune it, time large layers and the delay of an
    #: interrupt, and keep the smallest value whose overhead is negligible.
    information_set_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    #: The most supports that one compiled call of the scan of a count, by the BZ or brute-force
    #: method, visits; a call also ends at the end of each coefficient layer of each information
    #: set.  Lowering it makes interrupts prompter, at more overhead; ``MAX_VALUE`` lets each call
    #: finish its layer.  At least 1.  Defaults to 65,536; how it was chosen is not recorded.  To
    #: retune it, time long counts and the delay of an interrupt, and keep the smallest value whose
    #: overhead is negligible.
    scan_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The most calls of ``scan_chunk_size`` supports that the scan of a count makes before it
    #: returns to Python, once :func:`~qldpc.codes.distance.logical_counts.compile_search_setup` has
    #: compiled its loop over layers; until then, it returns after every call.  Lowering it makes
    #: interrupts prompter; ``MAX_VALUE`` lets a compiled scan run to its end.  At least 1.
    #: Defaults to 64; how it was chosen is not recorded.  To retune it, proceed as for
    #: ``scan_chunk_size``.
    scan_chunks_per_call: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most Gray-code steps that the walk of a check space by the MacWilliams method takes in
    #: one call of compiled code.  Lowering it makes interrupts prompter, but crosses into compiled
    #: code more often, which slows counts.  At least 1.  Defaults to 65,536, chosen by hand.  To
    #: retune it, time long walks and the delay of an interrupt, and keep the smallest value whose
    #: overhead is negligible.
    macwilliams_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The smallest rank of the checks at which the MacWilliams method tests whether the X and Z
    #: checks of a counter span the same space, so that one walk over its ``2**rank`` words serves
    #: both; below it, a second walk costs less than the test.  Lowering it tests on smaller codes;
    #: zero whenever the ranks are equal, and ``MAX_VALUE`` never.  Counters known to be self-dual,
    #: and counts of full Pauli operators, skip the test.  Defaults to 14; how it was chosen is not
    #: recorded.  To retune it, time the test and a walk at several ranks.
    macwilliams_shared_walk_rank: int = 14

    #: The most pairs of distinct syndromes of a single-qubit X error and a single-qubit Z error,
    #: each including the identity, that a ``CosetLeaderSearch`` by CSS weight tabulates to answer
    #: queries of weight one quickly; with more, it matches them at each query instead.  Lowering it
    #: saves memory but slows oracles that answer many queries, as in the certification of fault
    #: propagation; zero disables the table.  Defaults to ``2**16``, about ten megabytes, chosen by
    #: hand; 255 qubits with distinct syndromes reach it.  To retune it, time such certifications
    #: near the limit with and without the table.
    max_css_single_pairs: int = 2**16

    def __post_init__(self) -> None:
        """Validate the fields, and store numbers as Python integers and floats.

        Compiled code specializes on the types of its arguments, so storing every integer as a
        Python integer keeps a tuning from compiling code again.

        Raises:
            TypeError: If a field whose default is a bool is not a bool, or another field is not a
                number of the type of its default, or is a bool.
            ValueError: If an integer field is outside its bounds or choices, or a float field is
                negative, above its bound, or not finite.
        """
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(field.default, bool):
                if not isinstance(value, bool):
                    raise TypeError(f"{field.name} must be a bool, not {value!r}")
                continue
            if isinstance(field.default, float):
                if isinstance(value, bool) or not isinstance(value, numbers.Real):
                    raise TypeError(f"{field.name} must be a real number, not {value!r}")
                try:
                    value = float(value)
                except OverflowError as error:
                    raise ValueError(f"{field.name} must be finite, not {value!r}") from error
                maximum = field.metadata.get("maximum", math.inf)
                if not (math.isfinite(value) and 0 <= value <= maximum):
                    bound = f" at most {maximum}" if math.isfinite(maximum) else " finite"
                    raise ValueError(f"{field.name} must be non-negative and{bound}, not {value}")
                object.__setattr__(self, field.name, value)
                continue
            value = as_integer(value, field.name)
            choices = field.metadata.get("choices")
            if choices is not None and value not in choices:
                raise ValueError(f"{field.name} must be one of {choices}, not {value}")
            minimum = field.metadata.get("minimum", 0)
            maximum = field.metadata.get("maximum", MAX_VALUE)
            if not minimum <= value <= maximum:
                raise ValueError(
                    f"{field.name} must be between {minimum} and {maximum}, not {value}"
                )
            object.__setattr__(self, field.name, value)


# The tuning that searches use unless a use_tuning block changes it.
DEFAULT_TUNING = SearchTuning()


def get_tuning() -> SearchTuning:
    """The active tuning of the current context, which a search reads when it starts."""
    return _ACTIVE_TUNING.get()


@contextlib.contextmanager
def use_tuning(**changes: Any) -> Iterator[SearchTuning]:
    """Change fields of the active tuning until the ``with`` block exits.

    The changes apply to searches that start within the block, in the current context, and in
    copies of it made within the block (see the module docstring for threads, processes, and
    asyncio tasks).  Fields that are not given keep their values in the active tuning, so nested
    blocks combine their changes, and the inner block takes precedence.

    Args:
        **changes: New values of fields of :class:`SearchTuning`.

    Yields:
        The tuning that is active within the block.

    Raises:
        TypeError: If a name is not a field of :class:`SearchTuning`, or a value has the wrong type.
        ValueError: If a value is out of range.
    """
    token = _ACTIVE_TUNING.set(dataclasses.replace(get_tuning(), **changes))
    try:
        yield get_tuning()
    finally:
        _ACTIVE_TUNING.reset(token)


# The active tuning, which each context, such as a thread or asyncio task, can change.
_ACTIVE_TUNING: contextvars.ContextVar[SearchTuning] = contextvars.ContextVar(
    "qldpc_search_tuning", default=DEFAULT_TUNING
)


####################################################################################################
# constants of caches

#: The number of arrays of binomial coefficients, one for each number of rows of a basis, that the
#: estimates of the work of searches of information sets keep.  It takes effect when qLDPC is
#: imported.  Chosen by hand: each array of a basis of ``k`` rows takes ``8 * (k + 1)`` bytes, and
#: few workloads search bases of more than a thousand different numbers of rows.  A larger cache
#: uses more memory, and a smaller one recomputes arrays that a workload reuses.  Only speed and
#: memory depend on it.  To retune it, record the hit rate of the cache on representative workloads,
#: and keep the smallest size beyond which it stops growing.
BINOMIAL_CACHE_SIZE = 1024

#: The number of entries that ``coset_leader_search._get_css_searches`` keeps, one for each CSS
#: state and value of ``disjoint_random_column_orders``, each a pair of classical searches, of the X
#: parts and of the Z parts of errors.  It takes effect when qLDPC is imported.  Chosen by hand.  A
#: larger cache uses more memory, and a smaller one repeats the setup of searches that a workload
#: reuses.  Only speed and memory depend on it.  To retune it, record the hit rate of the cache, the
#: time to set up a missed pair of searches, and the memory of each pair on representative
#: workloads, such as the certification of fault propagation, and keep the smallest size beyond
#: which the hit rate stops growing.
CSS_SEARCH_CACHE_SIZE = 8

#: The number of results that each of two caches of the MacWilliams method keeps, one of single
#: Krawtchouk rows, of one weight, and one of arrays of the rows of light weights.  It takes effect
#: when qLDPC is imported.  Chosen by hand.  A larger cache uses more memory, and a smaller one
#: recomputes rows that a workload reuses.  Only speed and memory depend on it.  To retune it,
#: record the hit rates of both caches, the time to recompute a missed row, and the memory of the
#: rows on counts of codes of the lengths of interest, and keep the smallest size beyond which the
#: hit rates stop growing.
KRAWTCHOUK_CACHE_SIZE = 1024

#: The number of results that the cache of the local Clifford matrices of deformed counters keeps,
#: each a pair of packed matrices, of the X parts and of the Z parts.  It takes effect when qLDPC is
#: imported.  Chosen by hand.  A larger cache uses more memory, and a smaller one recomputes
#: matrices that a workload reuses.  Only speed and memory depend on it.  To retune it, record the
#: hit rate of the cache and the time to recompute a missed pair on counts of many deformed codes,
#: and keep the smallest size beyond which the hit rate stops growing.
LOCAL_CLIFFORD_CACHE_SIZE = 1024
