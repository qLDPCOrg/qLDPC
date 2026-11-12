# SPDX-License-Identifier: Apache-2.0

"""Settings and constants that tune the speed and memory of exact distance searches and counts.

Exact distances and logical-operator counts choose between routes, such as exhaustive enumeration
and searches of disjoint or overlapping information sets, and bound their batches, memory, and
caches.  :class:`SearchTuning` collects the values that tune a search, from the thresholds that
choose a route to the coefficients of cost models and the sizes of batches.  The page
:doc:`/exact_search_tuning` of the documentation maps routes and symptoms to these values.  Section
4.4 of ``brouwer_zimmermann.md`` says where each heuristic of a Brouwer-Zimmermann search uses them,
and Sections 3.5 and 3.8 of ``connected_support.md`` where the connected-support searches use
theirs.

This module is private.  Its names, fields, and defaults can change in any release, without a
deprecation period or any other warning, so code that uses it should pin the exact version of
qLDPC that it was written for.  It is meant for contributors, for benchmarks, and for experienced
users of very costly searches who have measured that a different tuning helps them.  The defaults
suit typical searches.

What tuning can and cannot change:

- Tuning changes only how long a search takes and how much memory it uses.  It never changes an
  exact distance, a distance limited by ``max_weight``, or a count of logical operators.
- With an early exit (``cutoff`` of two or more), a distance search returns the weight of a
  nontrivial word at or below ``cutoff`` that it finds, and which word that is depends on tuning.
  The result is still an upper bound between the distance and ``cutoff``, as documented by
  :func:`~qldpc.codes.distance.get_distance_quantum`.
- Under every tuning, a query of ``CosetLeaderSearch`` decides the same way whether a coset has a
  word of weight at most the query's maximum weight, and with the connected-support method it gives
  the same lower bound.  The word that it returns, and with the Brouwer-Zimmermann method its lower
  bound, can depend on tuning, within the contract of ``MinWeightBound``.
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
and the oracles of ``CosetLeaderSearch``, each when it starts or is built.  Only values that qLDPC
fixes before any search starts are constants: ``MAX_VALUE``, and the sizes of caches, which it fixes
when it is imported.  Change those by editing this module, and start a new process.

.. _tuning-retuning:

Retuning:

Compare candidate values on warmed searches in new processes, single-threaded, in paired runs that
alternate their order, and record the time and peak memory of each run; repeat runs to measure
their spread.  The comment of each field says what to measure to retune it.

The scripts of ``experiments/distance_tuning`` do this, and its README walks through a whole
retuning.  ``sweep.py`` times the named workloads of ``workloads.py``, such as ``distance/qt7/X``
or ``count/pp110/12``, under candidate tunings, and ``collect_results.py`` gives the median ratio
of the time of each candidate to that of the first in the same round, with a 95% confidence
interval.  ``trace.py`` records the decisions and the time of each stage of a distance search or
count, and ``cache_stats.py`` the hit rates of the caches.  The comments of the fields name these
workloads.

The comments of most fields of BZ searches give the measurements behind their values, taken with
these scripts on an Apple M3 Max at qLDPC commits from ``15897436`` to ``eba1dd2f``, in 3 to 10
rounds of one to six searches at a time.  The value of each such field was chosen among candidates,
none of which had a special standing, by the first of these that applies:

1. a verified mechanism, which a trace or the code shows and an intervention confirms, or, for a
   coefficient, a direct measurement of the quantity that it stands for;
2. a large gain on long runs, those that take a minute or more one search at a time, with any loss
   on quick runs, those of a few seconds or less, stated in absolute time;
3. otherwise, the candidate whose reason is simplest to state, with a note in its comment that the
   candidates measured alike.

Values thus favor long, large runs, whose speed should depend little on the precise values.

A comment that says a value was chosen by hand, or that how it was chosen is not recorded, marks a
field that these measurements did not choose.

Completion tables saved less when several searches ran at once: with tables wherever they are
considered, ``distance/classical:150:100:1/-`` took 0.75 of its time with
``completion_bound_max_cost_percent=90`` one search at a time, 0.90 six at a time, and 1.30 three
at a time on a machine with a load average of 16.  Comments report such effects of load, but they
do not decide values.

The fields of the completion tables of distance searches and counts (Section 3.9 of
``brouwer_zimmermann.md``) were first chosen on the same machine, with layers of one information
set of ``4 * 10**7`` to ``5 * 10**10`` supports from two data sets (the comment of
``coset_completion_block_bits`` names its own data):

- The width data set holds layers of weights 4 to 6 of both sectors of
  ``BBCode({x: 12, y: 12}, x**3 + y**2 + y**7, y**3 + x + x**2)``, the [[288, 12, 18]] code, and of
  ``QTCode.random(CyclicGroup(7), HammingCode(3), seed=1)``, and of two lifted-product codes of
  3 x 4 random monomials of ``CyclicGroup(13)``, drawn with NumPy's default generator from seeds 5
  and 8.
- The cost data set holds nine layers of the X sector of that bivariate bicycle code and of the
  [[232, 62, 12]], [[170, 8, 20]], and [[276, 98, 14]] pair-partition codes, ``pp232``, ``pp170``,
  and ``pp276`` of ``workloads.py``.  On five other layers, the cost model fitted to it predicted
  cost ratios at most 0.12 above, and 0.02 below, the measured ones.

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

``fit_completion_costs.py`` takes these steps, from traces of whole searches.  Accept values if, at
the defaults, no layer is slower than the faster of the other two arms by more than the spread of
repeated runs.
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
    #: words doubles with each row; lowering it makes small searches pay for setup.  Defaults to 20.
    #: Neither 18 nor 20 was faster on every code, and these searches take milliseconds.  With 18,
    #: ``distance/classical:40:20:2/-`` and ``distance/sparse:200:20:10:2/-``, of 20 rows, took 0.57
    #: and 0.35 of their time.  But ``distance/classical:200:180:2/-``, of 20 rows, took 1.31 times
    #: as long, since it then enumerates after building information sets: with
    #: ``allow_enumeration_by_work`` off too, it took 0.87.  And ``distance/sparse:200:19:10:2/-``,
    #: of 19 rows, took 1.41 times as long, since information sets are slower for it.  Every size
    #: test of ``experiments/distance_tuning`` stays the same only at 20, between
    #: ``distance/classical:1000:980:2/-`` of 20 rows and ``distance/classical:1000:979:2/-`` of 21,
    #: which at 21 enumerates by size instead of by work.  To retune it, time both routes on bases
    #: near the crossover, and then the defaults.
    enumeration_max_dimension: int = 20

    #: A distance search of more than ``enumeration_max_dimension`` and fewer than 63 basis rows
    #: also enumerates every word if ``2**rows`` is at most this factor times its numbers of rows
    #: and columns, an estimate of the cost of setting up its information sets.  Raising it
    #: enumerates more long codes of low dimension; lowering it builds information sets for more of
    #: them.  Defaults to 16, the nearer of 16 and 128 to a direct measurement of what it stands
    #: for: the time to build the disjoint information sets of a classical code, divided by its
    #: numbers of rows and columns and by the time to enumerate one word.  That ratio was 0.35 to
    #: 10.5 on random and sparse codes of 1000 and 4000 bits with 16 or 20 rows, and 57 to 97 on
    #: codes of 200 bits, for which no factor below 499 enumerates 21 rows.  It varies because rows
    #: of more than 960 bits enumerate one word at a time (see ``enumeration_batch_exponent``).  On
    #: codes of 1000 and 4000 bits with 21 to 23 rows, 16 instead of 128 made four of six sparse
    #: codes 2 to 25 times faster and the other two no slower, and random codes at most 1.02 times
    #: slower.  Every size test stays the same from 12.5 to 23.8, the ratios of ``2**rows`` to the
    #: product of rows and columns of ``distance/classical:8000:7979:2/-`` and
    #: ``distance/classical:8000:7978:2/-``, and since the second enumerates by work anyway, the
    #: first search whose route changes above 16 is ``distance/sparse:4000:21:200:2/-``, at 25.  To
    #: retune it, measure that ratio.
    enumeration_setup_cost_factor: int = 16

    #: The exponent of the batches of exhaustive enumeration, which handles at most
    #: ``2**enumeration_batch_exponent`` words at a time in a few arrays of at most
    #: ``2**(enumeration_batch_exponent + 3)`` bytes each, and checks for a keyboard interrupt
    #: between batches.  Rows of ``w`` packed 64-bit words are vectorized over
    #: ``2**(enumeration_batch_exponent + 1 - w)`` combinations of rows
    #: (``_brute_force.py``), so at 15, rows of more than 960 bits enumerate one word at a
    #: time: forced enumeration of ``distance/classical:1000:980:2/-`` took about 4 microseconds per
    #: word at 15, and 13 to 45 nanoseconds at 30, in two runs.  But at 30, rows of 4 words would
    #: take arrays of 4 GiB.
    #: Raising it can speed up enumeration, at the cost of memory; lowering it saves memory and
    #: makes interrupts prompter, but adds overhead.  At most 62.  Defaults to 15, chosen by hand.
    #: To retune it, time enumerations of bases of 16 to 24 rows, and keep the smallest value beyond
    #: which they get no faster.
    enumeration_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    # Fields of information-set families.

    #: The most information sets in a family of overlapping sets, or in a family found by matroid
    #: partitioning with a coverage of two or more.  Distance searches and counts build both, and
    #: coset searches build overlapping ones for costly queries.  More sets can skip costly layers,
    #: but setup grows roughly with the square of this limit and dominates cheap searches.  Zero
    #: disables both, but not the maximum disjoint family that a coset search finds by matroid
    #: partitioning with coverage one.  Defaults to 16: 16 and 32 measured alike, and 16 builds
    #: the smaller families, while 8 made ``distance/classical:140:90:1/-`` 1.46 times slower, and
    #: ``distance/lp13s5/X`` without automorphisms 1.22 times.  To retune it, compare the total
    #: times of hard searches.
    max_information_sets: int = 16

    #: How many random column orders, in a fixed sequence, a search tries when the natural order
    #: packs fewer full disjoint information sets than the nonzero columns divided by the rows; it
    #: keeps the best packing.  More orders can find more full sets, which certify sooner, but each
    #: costs a packing of the basis; zero keeps the natural order.  Defaults to 8: 8, 16, and 32
    #: measured alike, and 8 packs the fewest orders, while 0 made ``coset/gross/physical/6/6/500``
    #: 1.02 times slower, and 64 made ``hbound/pp232/2/2000`` 1.14 times slower.  To retune it,
    #: record after how many orders the best packing stops improving, and weigh setup time against
    #: the search time saved.
    disjoint_random_column_orders: int = 8

    #: The estimated work of a search, in examinations (candidate/set evaluations), below which it
    #: keeps its disjoint information sets.  It also gates the coset searches of
    #: ``CosetLeaderSearch``: a query with fewer predicted examinations keeps the disjoint sets and
    #: whole words.  Lowering it tries other families sooner, and, in searches by Hamming weight,
    #: tails and completion tables, which may not repay setup; zero always tries, ``MAX_VALUE``
    #: never.  Defaults to 100,000.  No candidate was faster on every coset search: 10,000 made
    #: ``coset/gross/css/10/10/3000``, of 3 seconds, 1.7 times faster, but
    #: ``coset/pp232/css/10/10/3000`` and ``coset/qt7/css/10/10/3000``, of 4 and 6 minutes, 3.2 and
    #: 2.1 times slower, so 100,000 saves minutes where 10,000 saves seconds.  Neither changed a
    #: distance search by more than 3%: the distance searches decide the same from ``8.4 * 10**4``
    #: to ``1.4 * 10**6``, and the long ones estimate at least ``5 * 10**11``.  To retune it, time
    #: searches near it with and without other families, including long coset searches.
    overlapping_minimum_work: int = 100_000

    #: The column orders that building a family of overlapping information sets tries for each set
    #: that it may add; an order that repeats an existing set still counts.  Distance searches and
    #: counts build such families, and coset searches of ``CosetLeaderSearch`` build them for costly
    #: queries.  More attempts can reach a cheaper family, but lengthen setup; zero builds no
    #: family.  Defaults to 2: 1, 2, and 4 measured alike, and 2 leaves a second order when the
    #: first repeats an existing set.  To retune it,
    #: record how the estimated work of the family falls with more attempts, and weigh setup time
    #: against the search time saved.
    overlapping_attempts_per_set: int = 2

    #: A search builds families by matroid partitioning only if its estimated work is at least this
    #: multiple of the estimated cost of partitioning (see ``partition_cost_factor``).  Lowering it
    #: tries partitioning on cheaper searches; zero always tries, and ``MAX_VALUE`` never does.
    #: Were the estimates accurate, an attempt that does not pay off would add at most about
    #: ``1 / partition_work_ratio`` to the work of a search.  But the estimate of the work of a
    #: search assumes that it finds no lighter word before it certifies its best one, and exceeded
    #: the work done 24 to 890 times on ``distance/lp13s5/X`` and ``hadamard/gross``.  So no ratio
    #: separates the attempts that waste time from those that help: partitioning took 1.4 of the 1.8
    #: seconds of ``distance/lp13s8/X`` and found no family, while on
    #: ``distance/classical:128:80:1/-`` it found one that made the search 1.19 times faster, and
    #: the gate admits both at every ratio up to about 1600.  Defaults to 100, which compensates for
    #: that bias as far as these workloads allow.  On long runs, the gate decides the same for every
    #: product of this ratio and ``partition_cost_factor`` from 33 to ``1.4 * 10**5``, and
    #: ``hadamard/gross`` and ``hadamard/pp110`` measured alike from 10 to never partitioning, so
    #: the value matters only for quick runs: 100 skips the attempt on ``distance/lp13s5/X``, which
    #: saves 1.3 of its 1.8 seconds.  To retune it, time searches that partition with and without
    #: it.
    partition_work_ratio: int = 100

    #: The largest coverage, the most information sets that contain any one column, that matroid
    #: partitioning tries; it tries each coverage from two to this value.  Larger values can find
    #: families with more sets, which certify sooner, but each coverage adds to the cost of
    #: partitioning; below two, it builds none.  Defaults to 4: on
    #: ``distance/classical:128:80:1/-``, whose search partitioning speeds up, 2 made it 1.21 times
    #: slower.  To retune it, record which coverage gives the cheapest family, and weigh the time of
    #: each further coverage against the search time it saves.
    max_partition_coverage: int = 4

    #: The estimated cost of partitioning ``b`` copies of the ``n`` nonzero columns of a basis of
    #: ``k`` rows, in units of ``b * k**2 * n`` candidate/set evaluations.  An underestimate wastes
    #: time on partitioning that cannot repay itself; an overestimate skips families that would save
    #: time.  Defaults to 10, a direct measurement: partitioning took 1.26 and 1.31 seconds on
    #: ``distance/lp13s5/X``, with and without automorphisms, and 0.85 seconds on
    #: ``hadamard/gross``, 10.2, 11.1, and 4.5 of these units at the 1.3 to 2.0 nanoseconds that the
    #: same searches took for each evaluation of layers without tables.  With 10 instead of 20, no
    #: search of ``experiments/distance_tuning`` partitions differently.  It acts only through its
    #: product with ``partition_work_ratio`` (see there).  To refit it, time
    #: ``get_information_sets_with_coverage`` of ``_information_sets/matroid_partition.py`` in
    #: candidate/set evaluations.
    partition_cost_factor: int = 10

    #: The estimated work, in candidate/set evaluations, from which a search of overlapping
    #: information sets also bounds unvisited words by linear programming (Section 3.1 of
    #: ``brouwer_zimmermann.md``).  Each linear program takes milliseconds, which only costly
    #: searches recover.  Lowering it uses them on cheaper searches; zero on every search of
    #: overlapping sets, and ``MAX_VALUE`` on none.  Defaults to 10**9, the middle of 10**8, 10**9,
    #: and 10**10, which measured alike: the long searches that can solve linear programs reach it
    #: by far, from ``1.2 * 10**13``, and solve 2 or 3 in under 10 milliseconds, while on all
    #: workloads, which solve at most 14 in 24 milliseconds, the decisions stay the same from
    #: ``3.7 * 10**8`` to ``2.0 * 10**9``.  To retune it, time searches near the threshold with and
    #: without them.
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
    #: of ``brouwer_zimmermann.md``).  It decides once.  Finding them takes milliseconds; lowering
    #: it pays that on cheaper searches, and can decide before the search finds its best word, when
    #: the images predict no less work; zero decides before the first layer, and ``MAX_VALUE``
    #: never.  Only the ``get_distance_exact`` methods of binary quantum codes find automorphisms.
    #: Defaults to 10**8, the middle of ``3 * 10**7`` to ``3 * 10**8``, which measured alike.  With
    #: 10**7, the search of ``distance/qt7/X`` decided before finding its best word, kept its
    #: family, and took 5 to 7 times as long, and with 10**9, it and the searches of
    #: ``distance/qt7/Z`` and ``distance/lp13s5/X`` were 1.3 to 1.4 times slower.  On long runs, the
    #: decisions stay the same from ``8.4 * 10**7``, layer 4 of ``hadamard/pp110``, to
    #: ``1.8 * 10**8``, layer 4 of ``distance/bb288/X``, and just past either edge, those searches
    #: switch one layer sooner or later, with the same later decisions, a difference of at most 0.22
    #: seconds.  To retune it, time searches with and without automorphisms.
    automorphism_minimum_layer_work: int = 100_000_000

    # Fields of connected-support searches.

    #: A connected-support search of a quantum code asks for its automorphisms before the first
    #: weight limit at which it expects to have visited this many states, which are supports, over
    #: all limits of one problem (Section 3.5 of ``connected_support.md``).  It also asks before a
    #: first limit that a cutoff raises, unless this is ``MAX_VALUE``.  Lowering it pays the cost
    #: of finding them on cheaper searches; raising it searches more limits from every variable.
    #: Zero asks before the first limit, and ``MAX_VALUE`` never.  Defaults to 100,000, chosen by
    #: hand.  To retune it, time searches with and without automorphisms.
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

    #: Making checks sparser stops early after this many rounds in a row that leave the total
    #: weight of the checks unchanged (see ``max_sparsifying_rounds``).  Raising it spends more
    #: rounds on checks that rarely get lighter; zero stops after the first round, and ``MAX_VALUE``
    #: never stops early, so that with both fields at ``MAX_VALUE`` the rounds never end.  Defaults
    #: to 200, chosen by hand from the rounds between improvements on the pair-partition codes of
    #: ``experiments/distance_tuning/workloads.py``.  To retune it, record the rounds between
    #: improvements.
    sparsifying_rounds_without_improvement: int = 200

    # Fields of completion tables.

    #: The number of supports (choices of rows) of an information set from which a distance search
    #: or count considers completion tables for its layer, which skip supports that cannot complete
    #: a word lighter than the best, or, in a count, within the counted weight (Section 3.9 of
    #: ``brouwer_zimmermann.md``).  It only bounds the cost of considering tables, a sampling of the
    #: layer, since the cost model then decides whether the layer uses them.  Lowering it considers
    #: smaller layers; zero every layer, and ``MAX_VALUE`` none.  Defaults to ``3 * 10**7``.  At
    #: 10**8, ``count/pp110/12``, whose steps have ``4.5 * 10**7`` supports, never considered
    #: tables; at ``3 * 10**7`` it uses them and was 2.75 times faster, and
    #: ``distance/classical:128:80:1/-``, with a layer of ``7.4 * 10**7`` supports, 1.49 times
    #: faster.  On the layers of at least ``3 * 10**7`` supports that then declined tables, sampling
    #: took 6 to 8 milliseconds for each set, at most 14% of the layer, and on larger layers under
    #: 3%.  10**7 saved no more.  On the long searches of ``experiments/distance_tuning``, every
    #: table decision stays the same from below 10**6 supports to ``5.9 * 10**8``, layer 5 of
    #: ``distance/bb288/X``, the smallest layer that uses tables there: the layers in between
    #: decline them, so the value matters only for quick runs.  To retune it, time layers near the
    #: threshold with tables disabled and forced, and the sampling of layers that decline them.
    completion_bound_minimum_layer_work: int = 30_000_000

    #: A layer of a distance search or count uses the completion tables that it considers only if
    #: their predicted cost, including building them, is at most this percentage of the cost of
    #: scanning without them.  Lowering it demands larger predicted savings; zero never uses tables,
    #: and ``MAX_VALUE`` uses them wherever they are considered, even where they cost more.
    #: Defaults to 90, a margin for the error of the cost model, which overestimates some layers
    #: and, with the machine under load, underestimates others.  It predicted 0.67 for layer 5 of
    #: ``distance/pp232/X``, which took 0.66 to 0.96 of its time without tables when it used them,
    #: in two sessions, but 1.41 for layer 9 of ``distance/classical:150:100:1/-``, of 50 rows,
    #: which would have taken 0.55 to 0.64, and 0.90 for layer 6 of ``hadamard/gross``, which took
    #: 0.45.  Its decisions stay the same from 89.84, the predicted cost of that layer in percent,
    #: to 103.5, that of layer 5 of ``hadamard/pp110``.  90 and 100 measured alike.  One search at a
    #: time, 200 made ``distance/classical:160:110:1/-`` and ``distance/classical:150:100:1/-`` 1.4
    #: and 1.3 times faster and slowed none of eight searches, and the long searches of
    #: ``hadamard/gross``, ``hadamard/pp110``, and ``distance/pp170/X`` build the same tables one or
    #: two layers sooner, a difference of at most 0.4 seconds; three at a time, it made
    #: ``distance/classical:140:90:1/-`` 1.3 times slower.  To retune it, compare predicted and
    #: measured costs with ``fit_completion_costs.py``, and time searches one and several at a time.
    completion_bound_max_cost_percent: int = 90

    #: The most bytes of completion tables that a distance search keeps for each problem, a count
    #: for each sector besides the tables that it is building, and a coset search of
    #: ``CosetLeaderSearch`` for each set of a family.  Distance searches and counts use blocks of
    #: ``completion_bound_block_bits`` columns, or of ``completion_bound_fallback_block_bits`` if
    #: those do not fit, or none.  Zero disables tables, and ``MAX_VALUE`` removes the limit.
    #: Defaults to 256 MiB, which fits blocks of 16 columns for bases of up to about 500 rows and
    #: 140 tail columns.  To retune it, choose the memory that tables may use.
    completion_bound_max_table_bytes: int = 256 * 2**20

    #: The number of tail columns in a block of the completion tables of distance searches and
    #: counts, a power of two up to 16.  Wider blocks skip more supports, but the table of a block
    #: has ``2**width`` bytes for each row of the basis: 1 builds fastest and skips least, and 16
    #: the reverse.  Defaults to 16, which on every layer of the width data set of at least
    #: ``5 * 10**8`` supports skipped more supports, in less time, than 8.  On whole searches, 8 and
    #: 4 left ``distance/qt7/X`` 13 times, and ``distance/pp232/X`` and ``count/gross/12`` 2 times,
    #: slower, as slow as without tables.  With 8, the cost model predicted 1.25 to 1.32 for the
    #: costliest layers of ``distance/qt7/X``, and declined tables, which forced there took 1.00 to
    #: 1.16 of the time without them.  To retune it, force each width on that data set and keep the
    #: fastest.
    completion_bound_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

    #: The width of the blocks of completion tables when blocks of ``completion_bound_block_bits``
    #: columns would exceed ``completion_bound_max_table_bytes``.  Blocks of 8 columns need about
    #: 128 times less memory than blocks of 16, and skip fewer supports.  Defaults to 8, which still
    #: saved time on every layer of the width data set of at least ``10**9`` supports.  To retune
    #: it, lower the memory limit until that data set falls back, and compare narrower widths.
    completion_bound_fallback_block_bits: int = dataclasses.field(default=8, metadata=_BLOCK_WIDTHS)

    #: The number of tail columns in a block of the completion tables of ``CosetLeaderSearch``, a
    #: power of two up to 16.  Its searches by Hamming weight build tables for their costliest
    #: queries, within ``completion_bound_max_table_bytes`` for each set.  Wider blocks skip more
    #: completions but cost more memory and time, as for ``completion_bound_block_bits``: 1 builds
    #: fastest and skips least, and 16 the reverse.  Defaults to 8.  Neither 8 nor 16 was faster on
    #: every search: 16 made the searches of X and Z parts of ``coset/qt7/css/10/10/3000`` and
    #: ``coset/pp232/css/10/10/3000`` 29 and 8 times faster, but the searches of Pauli errors of
    #: ``coset/lp13s5/physical/7/7/1000`` and ``coset/pp232/physical/6/6/500`` 4 and 2.8 times
    #: slower, and 4 made all of them 1.4 to 1.9 times slower.  On residual-weight estimates of
    #: state-preparation circuits of 37, 81, and 95 qubits, not in the repository, 16 columns took
    #: longer to build than most estimates spent scanning.  To retune it, time such estimates with
    #: each width.
    coset_completion_block_bits: int = dataclasses.field(default=8, metadata=_BLOCK_WIDTHS)

    #: The number of random supports from which a search estimates the share of a layer that
    #: completion tables would skip, within the budget ``completion_bound_sampling_work``; drawing
    #: them takes about 16 bytes for each sample and each row of the basis.  More samples narrow the
    #: estimate, so that tables pay on more of the layers that accept them, but take longer.
    #: Between 1 and ``2**12``.  Defaults to 256: 128 and 256 measured alike, and 256 narrows the
    #: estimate more, while 512 made ``count/pp110/12`` 1.07 times slower.  Sampling took 1 to 13
    #: milliseconds on the cost data set.
    #: To retune it, raise it while sampling stays under a tenth of the time of the smallest layer
    #: considered.
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
    #: than they save.  Defaults to 2.0: 1.0, 2.0, and 3.0 measured alike, and 2.0 is a conventional
    #: confidence of about 98%.  To retune it, raise it if sampling error made choices
    #: slower on the cost data set.
    completion_bound_confidence_z: float = dataclasses.field(
        default=2.0, metadata={"maximum": 10.0}
    )

    #: The cost of one lookup of completion tables for each block of the tables, in scanned
    #: supports, the unit of their cost model: the time to scan one support without tables.  An
    #: underestimate makes searches use tables that cost more than they save; an overestimate makes
    #: them decline tables that would save time.  Defaults to 2.0, fitted to the cost data set, in
    #: which lookups cost 2 to 8 scanned supports per block.  A refit by ``fit_completion_costs.py``
    #: on the layers of one set of 9 workloads of ``experiments/distance_tuning`` gave
    #: ``0.58 + 0.042`` scanned supports per block for each MiB of tables, from lookups of 0.46 to
    #: 0.85 in tables of under 1 MiB, and 1.2 to 7.6 in tables of 18 to 113 MiB.  Its lower costs
    #: would make searches use more tables, which no workload has timed.  To refit it, see
    #: :ref:`Retuning <tuning-retuning>`.
    completion_bound_lookup_block_cost: float = 2.0

    #: The additional cost of a lookup of completion tables, in scanned supports for each block and
    #: each MiB of tables, since lookups in larger tables cost more.  Raising it makes searches
    #: decline more large tables; lowering it lets them use large tables that may cost more than
    #: they save.  Defaults to 0.04, fitted with ``completion_bound_lookup_block_cost`` to the cost
    #: data set (see there for a refit).  To refit it, see :ref:`Retuning <tuning-retuning>`.
    completion_bound_lookup_block_cost_per_mib: float = 0.04

    #: A multiplier of the predicted number of lookups of completion tables, which is the share of
    #: pairs of a prefix and the first row of its last pair that a scan with tables keeps, divided
    #: by the share of supports that it keeps.  It is below one, since tables most often skip late
    #: first rows, which have few supports each.  Raising it makes tables look costlier, and
    #: lowering it cheaper.  Defaults to 0.35, a direct measurement: the median of the ratios on the
    #: layers of one set with blocks of 16 columns of the workloads of ``fit_completion_costs.py``,
    #: which ranged from 0.31 to 0.49 on quantum codes and were 0.04 on the last layer of
    #: ``distance/classical:150:100:1/-``.  The cost data set gave 0.31 to 0.50.  To refit it, see
    #: :ref:`Retuning <tuning-retuning>`.
    completion_bound_kept_first_row_ratio: float = 0.35

    #: The cost of building completion tables, in scanned supports for each byte of tables; with
    #: ``completion_bound_build_step_cost``, it predicts the cost of a build.  An underestimate
    #: builds tables that small layers do not repay; an overestimate declines tables that would save
    #: time.  Defaults to 0.1, with 0.2 for each step (see ``completion_bound_build_step_cost``),
    #: which a direct measurement supports: on 44 builds of the workloads of
    #: ``fit_completion_costs.py``, the two predicted a median of 0.96 of the measured cost, and
    #: 0.43 to 2.6 of it.  To refit it, see :ref:`Retuning <tuning-retuning>`.
    completion_bound_table_byte_cost: float = 0.1

    #: The cost of building completion tables, in scanned supports for each step of the
    #: breadth-first searches that fill them, as ``_estimate_build_cost`` in
    #: ``_information_sets/completion_bounds.py`` counts steps.  Raising it makes searches decline
    #: tables that are costly to build; lowering it lets them build tables that small layers do not
    #: repay.  Defaults to 0.2, which a direct measurement supports (see
    #: ``completion_bound_table_byte_cost``).  To refit it, see :ref:`Retuning <tuning-retuning>`.
    completion_bound_build_step_cost: float = 0.2

    # Fields of batches and memory limits.

    #: A scan of information sets by the BZ method visits at most
    #: ``2**information_set_batch_exponent`` supports and lookups of completion tables, counted
    #: together, in one call of compiled code, which cannot be interrupted.  Lowering it lets a
    #: keyboard interrupt take effect sooner, but returns to Python more often.  At most 62.
    #: Defaults to 15, the smallest value that measured alike with 18 and 21 at one and at six
    #: searches at a time, while 12 made ``distance/pp232/X`` 1.08 times slower; 13 and 14 measured
    #: alike with 15 within wider intervals.  To retune it, time large layers and
    #: the delay of an interrupt, and keep the smallest value whose overhead is negligible.
    information_set_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    #: The most supports, and lookups of completion tables, counted together, that one compiled call
    #: of the scan of a count, by the BZ or brute-force method, visits; a call also ends at the end
    #: of each coefficient layer of each information set.  Lowering it makes interrupts prompter, at
    #: more overhead; ``MAX_VALUE`` lets each call finish its layer.  At least 1.  Defaults to
    #: 65,536, a power of two among 32,768 to 2**20, which measured alike on ``count/gross/12`` and
    #: ``count/pp110/12``, while 4096 and 16,384 made them up to 1.09 and 1.02 times slower.  To
    #: retune it, time long counts and the delay of an interrupt, and keep the smallest value whose
    #: overhead is negligible.
    scan_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The most chunks of ``scan_chunk_size`` that the scan of a count visits before it returns to
    #: Python, once :func:`~qldpc.codes.distance.logical_counts.compile_search_setup` has compiled
    #: its loop over layers; until then, it returns after every chunk.  It also returns before and
    #: after each coefficient layer of an information set that may use completion tables.  Lowering
    #: it makes interrupts prompter; ``MAX_VALUE`` lets a compiled scan run to its end, or to such a
    #: layer.  At least 1.  Defaults to 64, a power of two among 4 to 1024, which measured alike on
    #: counts after ``compile_search_setup``.  To retune it, proceed as for
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
    #: propagation; zero disables the table.  Each search of Pauli errors keeps its table, a
    #: dictionary of Python integers, so a full table of ``2**18`` pairs, on a state of 511 qubits,
    #: takes 66 MiB, and one of ``2**16`` pairs, on 255 qubits, 11 MiB.  Defaults to ``2**18``: the
    #: table makes each query of weight one a lookup, instead of a match over the qubits, and on the
    #: 372-qubit state of ``hbound/pp372/2/2000``, which ``2**16`` does not fit, ``2**18`` made such
    #: queries 1.7 times faster, for a table of 28 MiB, and changed no other workload beyond the
    #: spread of runs.  To retune it, time such certifications near the limit with and without the
    #: table.
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
#: smaller one recomputes arrays that a workload reuses.  Seven distance, count, and coset
#: workloads of ``experiments/distance_tuning`` used 8 entries.  To retune it, record the hit rate
#: on typical workloads with ``cache_stats.py``, and keep the smallest size past which it stops
#: rising.
BINOMIAL_CACHE_SIZE = 1024

#: The number of entries that ``get_css_searches`` in ``_brouwer_zimmermann/coset.py`` keeps, one
#: for each CSS state and tuning of the setup of the searches, each a pair of classical searches,
#: for the X and Z parts of errors.  It takes effect when qLDPC is imported.  A larger cache uses
#: more memory, and a smaller one repeats the setup of searches that a workload reuses.  Chosen by
#: hand.  Certification and residual-weight estimates search one state at a time, and a missed pair
#: of the 144-qubit state of ``coset/gross/css/6/6/100`` took 0.015 seconds to set up.  To retune
#: it, record the hit rate, the setup time of a missed pair, and the memory of each pair on
#: workloads such as certification with ``cache_stats.py``, and keep the smallest size beyond which
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

#: The number of searches of Pauli errors that ``get_pauli_search`` in
#: ``_brouwer_zimmermann/coset.py`` keeps, one for each stabilizer matrix, measure of weight, and
#: tuning of the setup of the search.  It takes effect when qLDPC is imported.  A larger cache uses
#: more memory, a smaller one repeats setups, and zero disables the cache.  Only speed and memory
#: depend on it.  Chosen by hand; a missed search of the 343-qubit state of
#: ``coset/qt7/physical/5/5/50`` took 0.31 seconds to set up.  To retune it, record the hit rate,
#: the setup time of a missed search, and the memory per search on workloads such as certification
#: with ``cache_stats.py``, and keep the smallest size beyond which the hit rate stops growing.
PAULI_SEARCH_CACHE_SIZE = 8
