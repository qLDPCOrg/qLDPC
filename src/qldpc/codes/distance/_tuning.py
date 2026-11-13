# SPDX-License-Identifier: Apache-2.0

"""Settings and constants that tune the speed and memory of exact distance searches and counts.

Exact distances and logical-operator counts choose between routes, such as exhaustive enumeration
and searches of disjoint or overlapping information sets, and bound their batches, memory, and
caches.  :class:`SearchTuning` collects the values that tune a search, from the thresholds that
choose a route to the coefficients of cost models and the sizes of batches.  The tables of the page
:doc:`/exact_search_tuning` of the documentation, which renders this docstring, map routes and
symptoms to these values.  Section 4.4 of ``brouwer_zimmermann.md`` says where each heuristic of a
Brouwer-Zimmermann search uses them, and Sections 3.5, 3.8, and 3.9 of ``connected_support.md``
where the connected-support searches use theirs.

This module is private.  Its names, fields, defaults, and constants can change in any release,
without a deprecation period or any other warning, so code that uses it should pin the exact version
of qLDPC that it was written for.  It is meant for contributors, for benchmarks, and for experienced
users of very costly searches who have measured that a different tuning helps them.  The defaults
suit typical searches.

What tuning can and cannot change:

- Tuning changes only how long a search takes and how much memory it uses.  It never changes an
  exact distance, a distance limited by ``max_weight``, or a count of logical operators.
- With an early exit (``cutoff`` of two or more), a distance search returns the weight of a
  nontrivial word at or below ``cutoff`` that it finds, and which word that is depends on tuning.
  The result is still an upper bound between the distance and ``cutoff``, as documented by
  :func:`~qldpc.codes.distance.get_distance_from_generators`.
- Under every tuning, a query of :class:`~qldpc.codes.distance.CosetLeaderSearch` decides the same
  way whether a coset has a word of weight at most the query's maximum weight, and with the
  connected-support method it gives the same lower bound.  The word that it returns, and with the
  Brouwer-Zimmermann method its lower bound, can depend on tuning, within the contract of
  :class:`~qldpc.codes.distance.MinWeightBound`.
- Codes and counters remember their results, and some intermediate results are cached for the whole
  process (see ``BINOMIAL_CACHE_SIZE``, ``CSS_SEARCH_CACHE_SIZE``, ``KRAWTCHOUK_CACHE_SIZE``,
  ``LOCAL_CLIFFORD_CACHE_SIZE``, and ``PAULI_SEARCH_CACHE_SIZE``), so a later search with another
  tuning may not search again.  To time a tuning, run each search in a new process, after a first
  search that loads or compiles code.

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
and :class:`~qldpc.codes.distance.CosetLeaderSearch`, each when it starts or is built.  Only values
that qLDPC fixes before any search starts are constants: ``MAX_VALUE``, and the sizes of caches,
which it fixes when it is imported.  Change those by editing this module, and start a new process.

.. _tuning-retuning:

Retuning:

Compare candidate values on warmed searches in new processes, single-threaded, in paired runs that
alternate their order, and record the time and peak memory of each run; repeat runs to measure
their spread.  The comment of each field says what to measure to retune it.

The scripts of ``experiments/distance_tuning`` do this.  Its README walks through a whole
retuning, gives the commands of the sweeps behind the current values, and says what guards those
values against over-fitting and where the guards fall short.

Where those sweeps chose a value, they chose it among candidates, none of which had a special
standing.  A value rests on a verified mechanism or, for a coefficient, on a direct measurement of
what it stands for, where there is one.  Otherwise it rests on a large gain on long runs, those of a
minute or more one search at a time, with only small losses on quick runs, those of seconds.
Otherwise it is the simplest choice, such as a round value among candidates that made no clear
difference; some of those candidates made the same decisions on every search timed, so their timings
could differ only by noise.  A comment calls a value a judgment call where it trades one kind of
search against another.  Timings of the same work varied up to two- or threefold between sessions on
the one machine used, so comments state outcomes roughly.  The comment of each field other than the
two switches of enumeration says how its value was chosen: by these sweeps, by the measurements on
the data sets below, or by hand, or that this is not recorded.

The cost model and the block widths of the completion tables of distance searches and counts
(Section 3.9 of ``brouwer_zimmermann.md``) rest on measurements, made before these sweeps on an
Apple M3 Max, of layers of one information set of ``4 * 10**7`` to ``5 * 10**10`` supports from two
data sets:

- The width data set holds layers of weights 4 to 6 of both sectors of
  ``BBCode({x: 12, y: 12}, x**3 + y**2 + y**7, y**3 + x + x**2)``, the [[288, 12, 18]] code, and of
  ``QTCode.random(CyclicGroup(7), HammingCode(3), seed=1)``, and of two lifted-product codes of
  3 x 4 random monomials of ``CyclicGroup(13)``, drawn with NumPy's default generator from seeds 5
  and 8.
- The cost data set holds nine layers of the X sector of that bivariate bicycle code and of the
  [[232, 62, 12]], [[170, 8, 20]], and [[276, 98, 14]] pair-partition codes, ``pp232``, ``pp170``,
  and ``pp276`` of ``experiments/distance_tuning/workloads.py``.

To refit the cost model of completion tables, which ``predict_completion_cost`` and
``_estimate_build_cost`` in ``_information_sets/completion_bounds.py`` compute:

1. Time layers of one information set with tables disabled
   (``completion_bound_max_table_bytes=0``), forced (``completion_bound_minimum_layer_work=0`` and
   ``completion_bound_max_cost_percent=MAX_VALUE``), and at the defaults.  For each layer, record
   the share of its supports that tables skipped and its number of lookups, which
   ``scan_information_set`` in ``_brouwer_zimmermann/affine_search.py`` returns for each call, and
   time ``CompletionTables.build`` separately.
2. Express times in scanned supports, the time per support of the arm without tables.
3. Fit the coefficients that the cost model reads, ``completion_bound_lookup_block_cost``,
   ``completion_bound_lookup_block_cost_per_mib``, ``completion_bound_kept_first_row_ratio``,
   ``completion_bound_table_byte_cost``, and ``completion_bound_build_step_cost``, to the measured
   lookups, scans, and builds, and round the coefficients of builds up.

``fit_completion_costs.py`` of ``experiments/distance_tuning`` takes these steps.  Accept values
if, at the defaults, no layer is slower than the faster of the other two arms by more than the
spread of repeated runs.
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
    #: False makes such searches scan their information sets instead.  A switch for tests that force
    #: a route, not a calibration.  Defaults to True.
    allow_enumeration_by_work: bool = True

    #: A distance search enumerates every nontrivial word, instead of building information sets, if
    #: its basis has at most this many rows.  Raising it enumerates larger bases, whose number of
    #: words doubles with each row; lowering it makes small searches pay for setup.  Defaults to 20;
    #: how it was first measured is not recorded, and it predates larger enumeration batches for
    #: long rows, which made enumerating rows of several hundred bits or more much faster and can
    #: move the crossover on long codes.  To retune it, time both routes on bases near the
    #: crossover.
    enumeration_max_dimension: int = 20

    #: A distance search of more than ``enumeration_max_dimension`` and fewer than 63 basis rows
    #: also enumerates every word if ``2**rows`` is at most this factor times its numbers of rows
    #: and columns, an estimate of the cost of setting up its information sets.  Raising it
    #: enumerates more long codes of low dimension; lowering it builds information sets for more of
    #: them.  Defaults to 16, a judgment call: against 128, it made searches of most of the sparse
    #: codes of low dimension measured much faster, at a small cost to those of random ones.  Those
    #: measurements predate larger enumeration batches, which made rows of several hundred bits or
    #: more much faster to enumerate.  To retune it, time both routes on long codes near the
    #: crossover.
    enumeration_setup_cost_factor: int = 16

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
    #: partitioning with a coverage of two or more.  Distance searches and counts build both, and
    #: coset searches build overlapping ones for costly queries.  More sets can skip costly layers,
    #: but setup grows roughly with the square of this limit and dominates cheap searches.  Zero
    #: disables both, but not the maximum disjoint family that a coset search finds by matroid
    #: partitioning with coverage one.  Defaults to 16; how it was chosen is not recorded, and the
    #: quick searches timed build the same families under 32.  To retune it, compare the total times
    #: of hard searches.
    max_information_sets: int = 16

    #: How many random column orders, in a fixed sequence, a search tries when the natural order
    #: packs fewer full disjoint information sets than the nonzero columns divided by the rows; it
    #: keeps the best packing.  More orders can find more full sets, which certify sooner, but each
    #: costs a packing of the basis; zero keeps the natural order.  Defaults to 8, chosen by hand;
    #: on the quick distance searches timed, the natural order already packs the most sets, so no
    #: value changed them, while 64 slowed one coset search.  To retune it, record after how many
    #: orders the best packing stops improving, and weigh setup time against the search time saved.
    disjoint_random_column_orders: int = 8

    #: The estimated work of a search, in examinations (candidate/set evaluations), below which it
    #: keeps its disjoint information sets.  It also gates the coset searches of
    #: ``CosetLeaderSearch``, which build other families for the smallest weight whose queries reach
    #: it.  Lowering it tries other families sooner, and, in searches by Hamming weight, tails and
    #: completion tables, which may not repay setup; zero always tries, ``MAX_VALUE`` never.
    #: Defaults to ``2 * 10**4``, chosen with 8-column coset tables on CSS coset searches of a few
    #: codes, where values a few times lower or higher were slower; with 16-column tables, ``10**5``
    #: was a little faster on the one such search timed again.  To retune it, time searches near it
    #: with and without other families.
    overlapping_minimum_work: int = 20_000

    #: The column orders that building a family of overlapping information sets tries for each set
    #: that it may add; an order that repeats an existing set still counts.  Distance searches and
    #: counts build such families, and coset searches of ``CosetLeaderSearch`` build them for costly
    #: queries.  More attempts can reach a cheaper family, but lengthen setup; zero builds no
    #: family.  Defaults to 2, chosen by hand; the quick searches timed build the same families
    #: under 1 and 4.  To retune it, record how the estimated work of the family falls with more
    #: attempts, and weigh setup time against the search time saved.
    overlapping_attempts_per_set: int = 2

    #: A search builds families by matroid partitioning only if its estimated work is at least this
    #: multiple of the estimated cost of partitioning (see ``partition_cost_factor``).  Lowering it
    #: tries partitioning on cheaper searches; zero always tries, and ``MAX_VALUE`` never does
    #: unless ``partition_cost_factor`` is zero.  The estimated work can exceed the work done many
    #: times over, so the ratio does not bound the time that an attempt wastes.  Defaults to 500, a
    #: judgment call: no value separated the timed searches that partitioning slows, which switch to
    #: automorphism images, from those that it speeds up; 500 misclassified the fewest.  To retune
    #: it, time searches with and without partitioning, on codes with and without automorphisms.
    partition_work_ratio: int = 500

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
    #: time.  Defaults to 10, the higher end of what partitioning took on the two searches whose
    #: partitioning was timed.  To refit it, time ``get_information_sets_with_coverage`` of
    #: ``_information_sets/matroid_partition.py`` in candidate/set evaluations.
    partition_cost_factor: int = 10

    #: The estimated work, in candidate/set evaluations, from which a search of overlapping
    #: information sets also bounds unvisited words by linear programming (Section 3.1 of
    #: ``brouwer_zimmermann.md``).  Each linear program takes milliseconds, which only costly
    #: searches recover.  Lowering it uses them on cheaper searches; zero on every search of
    #: overlapping sets, and ``MAX_VALUE`` on none.  Defaults to 10**9, chosen by hand; few of the
    #: searches timed reach the range from 10**8 to 10**10.  To retune it, time searches near the
    #: threshold with and without them.
    lp_minimum_work: int = 1_000_000_000

    #: The largest denominator of the rational dual weights that the certificate from linear
    #: programming takes from the floating-point solution of its solver.  Every positive value is
    #: sound; a smaller one can round away part of the bound, and a larger one lengthens exact
    #: arithmetic.  Defaults to 2**20, chosen by hand.  To retune it, keep the smallest denominator
    #: that loses no certified bound.
    lp_dual_max_denominator: int = dataclasses.field(default=1 << 20, metadata={"minimum": 1})

    #: A distance search asks at most once for the automorphisms that its caller passes.  It asks
    #: before scanning its first coefficient layer of at least this estimated work, in candidate/set
    #: evaluations.  It then searches the images of one information set under them if that predicts
    #: less work (Section 3.7 of ``brouwer_zimmermann.md``).  Finding them, as the
    #: ``get_distance_exact`` methods of binary quantum codes do, takes milliseconds.  Lowering this
    #: threshold pays that cost on cheaper searches.  A threshold of zero asks before the first
    #: layer, and ``MAX_VALUE`` never asks.  Defaults to 10**8, between ``3 * 10**7`` and
    #: ``3 * 10**8``, over which the quick searches timed decide alike.  To retune it, time searches
    #: with and without automorphisms.
    automorphism_minimum_layer_work: int = 100_000_000

    # Fields of connected-support searches.

    #: A connected-support search asks for the automorphisms that its caller passes before the first
    #: weight limit at which it expects to have visited this many states, which are supports, over
    #: all limits of one problem (Section 3.5 of ``connected_support.md``).  It also asks before a
    #: first limit that a cutoff raises, unless this is ``MAX_VALUE``.  Lowering it pays the cost of
    #: finding them on cheaper searches; raising it searches more limits from every variable.  Zero
    #: asks before the first limit, and ``MAX_VALUE`` never.  Defaults to 100,000, chosen by hand.
    #: To retune it, time searches with and without automorphisms.
    automorphism_minimum_states: int = 100_000

    #: A ``CosetLeaderSearch`` with the connected-support method makes its checks sparser in up to
    #: this many rounds, each a row reduction in a random order of the columns (Section 3.8 of
    #: ``connected_support.md``).  Raising it can speed up queries but slows building the search;
    #: zero only adds checks to each other while that makes them lighter, and ``MAX_VALUE`` leaves
    #: the rounds to ``sparsifying_rounds_without_improvement``.  Defaults to 400, chosen by hand:
    #: on states ``|0⟩`` of the pair-partition codes of
    #: ``experiments/distance_tuning/workloads.py``, 1000 rounds make the checks less than 10%
    #: lighter on average than 400 do.  To retune it, time building the search and its queries.
    max_sparsifying_rounds: int = 400

    #: A connected-support distance search of a subsystem code makes its derived stabilizers sparser
    #: in up to this many rounds, each a row reduction in a random column order (Section 3.9 of
    #: ``connected_support.md``).  Raising it can speed up the search but delays its start.  Zero
    #: skips sparsifying, and ``MAX_VALUE`` leaves the rounds to
    #: ``sparsifying_rounds_without_improvement``.  Defaults to 50, chosen on gauged toric, surface,
    #: and bivariate bicycle codes and subsystem hypergraph product codes, where 400 rounds saved no
    #: time and started up to 3 s later.  To retune it, time searches and count their states.
    max_distance_sparsifying_rounds: int = 50

    #: Making checks sparser stops early after this many rounds in a row that leave the total
    #: weight of the checks unchanged (see ``max_sparsifying_rounds`` and
    #: ``max_distance_sparsifying_rounds``).  Raising it spends more rounds on checks that rarely
    #: get lighter; zero stops after the first round, and ``MAX_VALUE`` never stops early.  With
    #: both this field and the limit of the rounds at ``MAX_VALUE``, the rounds never end.  Defaults
    #: to 200, chosen by hand from the rounds between improvements on the pair-partition codes of
    #: ``experiments/distance_tuning/workloads.py``.  To retune it, record the rounds between
    #: improvements.
    sparsifying_rounds_without_improvement: int = 200

    # Fields of completion tables.

    #: The number of supports (choices of rows) of an information set from which a distance search
    #: or count considers completion tables for its layer, which skip supports that cannot complete
    #: a word lighter than the best, or, in a count, within the counted weight (Section 3.9 of
    #: ``brouwer_zimmermann.md``).  Considering them takes milliseconds of sampling, or tens on
    #: large bases.  Lowering it considers smaller layers; zero every layer, and ``MAX_VALUE`` none.
    #: Defaults to ``3 * 10**7``, so that a timed count of a code of moderate size uses tables,
    #: which made it several times faster.  To retune it, time layers near the threshold with tables
    #: disabled and forced.
    completion_bound_minimum_layer_work: int = 30_000_000

    #: A layer of a distance search or count uses the completion tables that it considers only if
    #: their predicted cost, including building them, is at most this percentage of the cost of
    #: scanning without them.  Lowering it demands larger predicted savings; zero never uses tables,
    #: and ``MAX_VALUE`` uses them wherever they are considered, even where they cost more.
    #: Defaults to 90, a margin for the error of the cost model; 100 slowed a timed search.  To
    #: retune it, compare predicted and measured costs on the cost data set.
    completion_bound_max_cost_percent: int = 90

    #: The most bytes of completion tables that a distance search keeps for each problem, a count
    #: for each sector besides the tables that it is building, and a coset search of
    #: ``CosetLeaderSearch`` for each set of a family.  Distance searches and counts use blocks of
    #: ``completion_bound_block_bits`` columns, or of ``completion_bound_fallback_block_bits`` if
    #: those do not fit, or none.  Zero disables tables, and ``MAX_VALUE`` removes the limit.
    #: Defaults to 256 MiB, chosen by hand: it fits blocks of 16 columns for bases of up to about
    #: 500 rows and 140 tail columns.  To retune it, choose the memory that tables may use.
    completion_bound_max_table_bytes: int = 256 * 2**20

    #: The number of tail columns in a block of the completion tables of distance searches and
    #: counts, a power of two up to 16.  Wider blocks skip more supports, but the table of a block
    #: has ``2**width`` bytes for each row of the basis: 1 builds fastest and skips least, and 16
    #: the reverse.  Defaults to 16, which on every layer of the width data set of at least
    #: ``5 * 10**8`` supports skipped more supports, in less time, than 8.  To retune it, force each
    #: width on that data set and keep the fastest.
    completion_bound_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

    #: The width of the blocks of completion tables when blocks of ``completion_bound_block_bits``
    #: columns would exceed ``completion_bound_max_table_bytes``.  Blocks of 8 columns need about
    #: 128 times less memory than blocks of 16, and skip fewer supports.  Defaults to 8, chosen by
    #: hand; forced on the largest layers of the width data set, it saved time on some and cost time
    #: on others, where the cost model declines it.  To retune it, lower the memory limit until that
    #: data set falls back, and compare narrower widths.
    completion_bound_fallback_block_bits: int = dataclasses.field(default=8, metadata=_BLOCK_WIDTHS)

    #: The number of tail columns in a block of the completion tables of ``CosetLeaderSearch``, a
    #: power of two up to 16.  Its searches by Hamming weight build tables for their costliest
    #: queries, within ``completion_bound_max_table_bytes`` for each set.  Wider blocks skip more
    #: completions but cost more memory and time (see ``completion_bound_block_bits``).  Defaults to
    #: 16, a judgment call: it made most CSS coset searches several times faster, but searches by
    #: physical weight slower on two of three codes, and it uses more memory.  On residual-weight
    #: estimates of state preparations, not in the repository, its tables took longer to build than
    #: most estimates spent scanning.  To retune it, time coset searches by CSS and by physical
    #: weight, and such estimates, with each width.
    coset_completion_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

    #: The number of random supports from which a search estimates the share of a layer that
    #: completion tables would skip, within the budget ``completion_bound_sampling_work``; drawing
    #: them takes about 16 bytes for each sample and each row of the basis.  More samples narrow the
    #: estimate, so that tables pay on more of the layers that accept them, but take longer: at
    #: 1024, up to about a third of the time of a layer near
    #: ``completion_bound_minimum_layer_work``.  Between 1 and ``2**12``.  Defaults to 1024, enough
    #: that the long-run layers traced near the limit of ``completion_bound_max_cost_percent``
    #: decided alike under every sampling draw tried.  To retune it, raise it until layers near the
    #: limit decide alike under different draws, while sampling stays a small part of quick runs.
    completion_bound_samples: int = dataclasses.field(
        default=1024, metadata={"minimum": 1, "maximum": 2**12}
    )

    #: The fewest samples with which a search trusts its estimate of the share that completion
    #: tables would skip; with fewer within the budget, it takes the skipped share to be zero, so
    #: that tables must pay without skipping.  At least 1.  Defaults to 64, chosen by hand so that
    #: when every sample skips, the lower end of the confidence interval (see
    #: ``completion_bound_confidence_z``) is 0.94.  To retune it, keep the fewest samples that still
    #: accept tables on the layers of the cost data set that they speed up.
    completion_bound_min_samples: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most work that a search spends bounding sampled supports, in units of one pair of rows on
    #: one block of the tables; summing the first rows of a support costs one unit for each row of
    #: the basis.  Raising it allows more samples on large bases, at more cost; zero takes no
    #: samples, so the search declines tables.  Defaults to ``5 * 10**7``, chosen by hand so that
    #: sampling takes tens of milliseconds on large bases.  To retune it, time sampling on the
    #: largest bases of interest, and keep it a small part of quick runs.
    completion_bound_sampling_work: int = 50_000_000

    #: How many standard errors below the sampled share of skipped supports a search sets its
    #: estimate, the lower end of a Wilson score interval; at most 10.  Raising it declines more
    #: layers where tables would pay; lowering it lets sampling error choose tables that cost more
    #: than they save.  Defaults to 2.0, a conventional confidence of about 98%, among values that
    #: made no clear difference.  To retune it, raise it if sampling error made choices slower on
    #: the cost data set.
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
    #: ``_information_sets/completion_bounds.py`` counts steps.  Raising it makes searches decline
    #: tables that are costly to build; lowering it lets them build tables that small layers do not
    #: repay.  Defaults to 0.2, chosen by hand with ``completion_bound_table_byte_cost``.  To refit
    #: it, see :ref:`Retuning <tuning-retuning>`.
    completion_bound_build_step_cost: float = 0.2

    # Fields of batches and memory limits.

    #: A scan of information sets by the BZ method visits at most
    #: ``2**information_set_batch_exponent`` supports and lookups of completion tables, counted
    #: together, in one call of compiled code, which cannot be interrupted.  Lowering it lets a
    #: keyboard interrupt take effect sooner, but returns to Python more often.  At most 62.
    #: Defaults to 15; timed one search at a time, smaller values were slower and larger ones not
    #: clearly faster.  To retune it, time large layers and the delay of an interrupt, and keep the
    #: smallest value whose overhead is negligible.
    information_set_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    #: The most supports, and lookups of completion tables, counted together, that one compiled call
    #: of the scan of a count, by the BZ or brute-force method, visits; a call also ends at the end
    #: of each coefficient layer of each information set.  Lowering it makes interrupts prompter, at
    #: more overhead; ``MAX_VALUE`` lets each call finish its layer.  At least 1.  Defaults to
    #: 65,536, twice the smallest value that cost nothing measurable on counts.  To retune it, time
    #: long counts and the delay of an interrupt, and keep the smallest value whose overhead is
    #: negligible.
    scan_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The most chunks of ``scan_chunk_size`` that the scan of a count visits before it returns to
    #: Python, once :func:`~qldpc.codes.distance.logical_counts.compile_search_setup` has compiled
    #: its loop over layers; until then, it returns after every chunk.  It also returns before and
    #: after each coefficient layer of an information set that may use completion tables.  Lowering
    #: it makes interrupts prompter; ``MAX_VALUE`` lets a compiled scan run to its end, or to such a
    #: layer.  At least 1.  Defaults to 64, inside a range of values that made no clear difference
    #: on compiled counts.  To retune it, proceed as for ``scan_chunk_size``.
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
    #: propagation; zero disables the table.  Defaults to ``2**18``, tens of megabytes, which 511
    #: qubits with distinct syndromes reach; on a timed state of 372 qubits, it made weight-one
    #: queries almost twice as fast.  To retune it, time such certifications near the limit with and
    #: without the table.
    max_css_single_pairs: int = 2**18

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

#: The number of arrays of binomial coefficients, one per number of rows of a basis, that the work
#: estimates of searches of information sets keep.  It takes effect when qLDPC is imported.  Chosen
#: by hand: the array of a basis of ``k`` rows takes ``8 * (k + 1)`` bytes, and few workloads search
#: bases of more than a thousand different numbers of rows.  A larger cache uses more memory, and a
#: smaller one recomputes arrays that a workload reuses.  To retune it, record the hit rate on
#: typical workloads, and keep the smallest size past which it stops rising.
BINOMIAL_CACHE_SIZE = 1024

#: The number of entries that the cache of ``get_css_searches`` in ``_brouwer_zimmermann/coset.py``
#: keeps, one for each CSS state and tuning of the setup of the searches, each a pair of classical
#: searches, for the X and Z parts of errors.  It takes effect when qLDPC is imported.  A larger
#: cache uses more memory, and a smaller one repeats the setup of searches that a workload reuses.
#: Chosen by hand.  To retune it, record the hit rate, the setup time of a missed pair, and the
#: memory of each pair on workloads such as certification, and keep the smallest size beyond which
#: the hit rate stops growing.
CSS_SEARCH_CACHE_SIZE = 8

#: The number of results that each of two caches of the MacWilliams method keeps, one of single
#: Krawtchouk rows, of one weight, and one of arrays of the rows of light weights.  It takes effect
#: when qLDPC is imported.  Chosen by hand.  A larger cache uses more memory, and a smaller one
#: recomputes rows that a workload reuses.  To retune it, record the hit rates of both caches, the
#: time to recompute a missed row, and the memory of the rows on counts of codes of interest, and
#: keep the smallest size beyond which the hit rates stop growing.
KRAWTCHOUK_CACHE_SIZE = 1024

#: The number of results that the cache of the local Clifford matrices of deformed counters keeps,
#: each a pair of packed matrices, of the X parts and of the Z parts.  It takes effect when qLDPC is
#: imported.  Chosen by hand.  A larger cache uses more memory, and a smaller one recomputes
#: matrices that a workload reuses.  Only speed and memory depend on it.  To retune it, record the
#: hit rate of the cache and the time to recompute a missed pair on counts of many deformed codes,
#: and keep the smallest size beyond which the hit rate stops growing.
LOCAL_CLIFFORD_CACHE_SIZE = 1024

#: The number of searches of Pauli errors that the cache of ``get_pauli_search`` in
#: ``_brouwer_zimmermann/coset.py`` keeps, one for each stabilizer matrix, measure of weight, and
#: tuning of the setup of the search.  It takes effect when qLDPC is imported.  A larger cache uses
#: more memory, a smaller one repeats setups, and zero disables the cache.  Only speed and memory
#: depend on it.  Chosen by hand.  To retune it, record the hit rate, the setup time of a missed
#: search, and the memory per search on workloads such as certification, and keep the smallest size
#: beyond which the hit rate stops growing.
PAULI_SEARCH_CACHE_SIZE = 8
