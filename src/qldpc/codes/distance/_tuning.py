# SPDX-License-Identifier: Apache-2.0

"""Settings and constants that tune the speed and memory of exact distance searches and counts.

Exact distances and logical-operator counts choose between routes, such as searches of disjoint or
overlapping information sets, and bound their batches, memory, and caches.  :class:`SearchTuning`
collects the values that tune a search, from the thresholds that choose a route to the coefficients
of cost models and the sizes of batches.  The tables of the page :doc:`/exact_search_tuning` of the
documentation, which renders this docstring, map routes and symptoms to these values.  Section 4.4
of ``brouwer_zimmermann.md`` explains the heuristics of Brouwer-Zimmermann searches that read them,
and Sections 3.5, 3.8, and 3.9 of ``connected_cluster.md`` those of the connected-cluster searches.

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
  connected-cluster method it gives the same lower bound.  The word that it returns, and with the
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

qLDPC chooses each default by a simple heuristic, as a round value, and never fits it to timings.
The comment of each field puts its default in one of these classes, or says that it is in none:

- A derived value is the gate of an optional step, such as building a family of information sets,
  solving linear programs, or asking for automorphisms.  The gate breaks even with the cost of the
  step: the step runs once the work that it acts on reaches that cost, which the gate assumes to an
  order of magnitude.  Work and cost count candidate/set evaluations in BZ searches and states in
  connected-cluster searches, and each gate is 1, 2, or 5 times a power of ten.  Most steps cost
  more on larger bases or codes, and their gates assume their cost on large ones, so on small
  searches a gate can skip a step that would pay.
- A limit at equality compares two estimates in one unit, and breaks even where they are equal.
- A rough estimate is a coefficient of the cost model of completion tables
  (``predict_completion_cost`` in ``_information_sets/completion_bounds.py``), such as the cost of a
  lookup of one block of the tables, in scanned supports.  The model decides at break-even with
  these rough estimates, so marginal decisions can go either way.
- A budget is a limit on memory, interrupt latency, or setup effort, chosen for safety rather than
  for speed.  A batch or chunk size keeps the overhead of each call of compiled code small next to
  its work, and keyboard interrupts prompt; a limit of memory or setup effort bounds the cost of the
  worst cases.
- A convention is a statistical or numerical setting, and its comment says what it guarantees.
- The widest allowed: the blocks of completion tables take the widest width that searches accept,
  since wider blocks never skip less where their tables fit, and a budget bounds their memory.

Two fields, ``coset_route_minimum_work`` and ``macwilliams_shared_walk_rank``, are in no class: they
were chosen by hand, and their comments say why.  The sizes of caches, which are constants, were
also chosen by hand.

Change a default only when its heuristic changes, or when a check shows that the assumption of its
heuristic, such as the cost of a step, is off by more than a factor of about 3.  Never change a
default to gain a factor of two on particular codes.  A change of heuristic applies to every field
of its class.

The scripts of ``experiments/distance_tuning`` serve these checks, and none of them sets or
justifies a value.  ``constants.py`` checks that the orders of magnitude that the heuristics assume
still hold on a machine, ``trace.py`` shows which decisions a search makes and what each of its
stages costs, and ``sweep.py`` catches gross regressions or bugs when the tuning or the search code
changes.  The README of that directory says how to run them.
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

    The comment above each field says what it decides and in what unit, and what raising or lowering
    it does.  It then names the class of its default and its rule, or, for a field in no class, how
    its default was chosen (see :ref:`Retuning <tuning-retuning>`), and points to the section of a
    guide that explains the heuristic that reads it, if any.
    """

    # Fields of exhaustive enumeration.

    #: The exponent of the batches of exhaustive enumeration, which finds distances by the method
    #: ``"brute_force"``, including the distance that a count of logical operators by that method
    #: finds first.  Enumeration handles at most ``2**enumeration_batch_exponent`` words at a time
    #: in a few arrays of at most ``2**(enumeration_batch_exponent + 3)`` bytes each, and checks for
    #: a keyboard interrupt between batches.  Raising it can speed up enumeration, at the cost of
    #: memory; lowering it saves memory and makes interrupts prompter, but adds overhead.  At most
    #: 62.  Defaults to 15, a budget of interrupt latency and memory (see :ref:`Retuning
    #: <tuning-retuning>`).
    enumeration_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    # Fields of information-set families.

    #: The most information sets in a family of overlapping sets, or in a family found by matroid
    #: partitioning, which tries each coverage from two up to the least at which this many full sets
    #: could fit, by counting their pivots (see ``get_partitioned_family`` in
    #: ``_information_sets/basis_families.py``).  Distance searches and counts build both, and coset
    #: searches build overlapping ones for costly queries.  More sets can skip costly layers, but
    #: setup grows roughly with the square of this limit, or its cube when partitioning, and
    #: dominates cheap searches.  Zero disables both, but not the maximum disjoint family that a
    #: coset search finds with coverage one.  Defaults to 16, a budget of setup effort (see
    #: :ref:`Retuning <tuning-retuning>`).
    max_information_sets: int = 16

    #: How many random column orders, in a fixed sequence, a search tries when the natural order
    #: packs fewer full disjoint information sets than the bound of Section 4.4 of
    #: ``brouwer_zimmermann.md``, which a basis may not reach; it keeps the best packing.  More
    #: orders can find more full sets, which certify sooner, but each costs a packing of the basis;
    #: zero keeps the natural order.  Defaults to 8, a budget of setup effort (see :ref:`Retuning
    #: <tuning-retuning>`).
    disjoint_random_column_orders: int = 8

    #: The estimated work of a distance search or count, in candidate/set evaluations, below which
    #: it keeps its disjoint information sets rather than try a family of overlapping ones.
    #: Lowering it tries that family on cheaper searches, where it may not repay its setup; zero
    #: always tries, and ``MAX_VALUE`` never does.  Defaults to ``2 * 10**6``, a derived value (see
    #: :ref:`Retuning <tuning-retuning>`): it breaks even with the cost of building an overlapping
    #: family on a large basis, a few million evaluations (Section 4.4 of
    #: ``brouwer_zimmermann.md``).
    overlapping_minimum_work: int = 2_000_000

    #: The predicted examinations of a query of a Brouwer-Zimmermann search of ``CosetLeaderSearch``
    #: with its disjoint information sets, from which the search also tries a maximum disjoint and
    #: an overlapping family; each query takes the family, the disjoint one included, with the least
    #: work.  In searches by Hamming weight, a query that reaches it even with that family also
    #: weighs words by their tails and skips completions by tables (Section 4.4 of
    #: ``brouwer_zimmermann.md``).  It also sets the weight that the overlapping family is aimed at,
    #: the least from two whose queries reach it.  Lowering it builds these for cheaper queries,
    #: with more memory, and aims lower; zero builds them at the first query, and ``MAX_VALUE``
    #: never.  Defaults to ``2 * 10**4``, chosen by hand, preferring speed to memory: it sets three
    #: decisions, and no break-even rule covers the aim.
    coset_route_minimum_work: int = 20_000

    #: The column orders that building a family of overlapping information sets tries for each set
    #: that it may add; an order that repeats an existing set still counts.  Distance searches and
    #: counts build such families, and coset searches of ``CosetLeaderSearch`` build them for costly
    #: queries.  More attempts can reach a cheaper family, but lengthen setup; zero builds no
    #: family.  Defaults to 2, a budget of setup effort (see :ref:`Retuning <tuning-retuning>`).
    overlapping_attempts_per_set: int = 2

    #: A search tries matroid partitioning only if its estimated work, in candidate/set evaluations,
    #: reaches this multiple of the sum of ``b**2 * k**2 * n`` over the coverages ``b`` that it
    #: tries, for ``k`` rows and ``n`` nonzero columns (see ``get_partitioned_family`` in
    #: ``_information_sets/basis_families.py``).  Lowering it tries partitioning on cheaper
    #: searches; zero always tries, and ``MAX_VALUE`` never does.  Searches given automorphisms
    #: switch to their images instead if those do no more work than any partitioned family could.
    #: Defaults to 5, a derived value (see :ref:`Retuning <tuning-retuning>`): it breaks even with
    #: the cost of partitioning, a few evaluations for each unit of that sum, to which the cost is
    #: roughly proportional (Section 4.4 of ``brouwer_zimmermann.md``).
    partition_work_ratio: int = 5

    #: The estimated work, in candidate/set evaluations, from which a search of overlapping
    #: information sets also bounds unvisited words by linear programming (Section 3.1 of
    #: ``brouwer_zimmermann.md``).  Lowering it uses them on cheaper searches; zero on every search
    #: of overlapping sets, and ``MAX_VALUE`` on none.  Defaults to ``2 * 10**8``, a derived value
    #: (see :ref:`Retuning <tuning-retuning>`): it breaks even with the cost of the linear programs
    #: of a search of a large basis, with the exact checks that plan the steps of their certificate,
    #: a few hundred million evaluations (Section 4.4 of ``brouwer_zimmermann.md``).
    lp_minimum_work: int = 200_000_000

    #: The largest denominator of the rational dual weights that the certificate from linear
    #: programming takes from the floating-point solution of its solver.  Every positive value is
    #: sound; a smaller one can round away part of the bound, and a larger one lengthens exact
    #: arithmetic.  Defaults to 2**20, a convention (see :ref:`Retuning <tuning-retuning>`):
    #: rounding moves each dual weight of the solver by at most 2**-21.
    lp_dual_max_denominator: int = dataclasses.field(default=1 << 20, metadata={"minimum": 1})

    #: The least work of a coefficient layer, in candidate/set evaluations, before which a distance
    #: search that would not partition considers switching to automorphism images (Section 3.7 of
    #: ``brouwer_zimmermann.md``); one that would partition considers it at the end of setup.  A
    #: search that declined reconsiders once its best weight drops.  Lowering it pays for
    #: automorphisms on cheaper searches; zero considers before the first layer, and ``MAX_VALUE``
    #: never.  Defaults to ``5 * 10**6``, a derived value (see :ref:`Retuning <tuning-retuning>`):
    #: it breaks even with the cost of finding automorphisms, verifying them, and building their
    #: images for a large code, several million evaluations (Section 4.4 of
    #: ``brouwer_zimmermann.md``).
    automorphism_minimum_layer_work: int = 5_000_000

    # Fields of connected-cluster searches.

    #: A connected-cluster search asks for the automorphisms that its caller passes before the first
    #: weight limit at which it expects to have visited this many states, which are supports, over
    #: all limits of one problem (Section 3.5 of ``connected_cluster.md``).  It also asks before a
    #: first limit that a cutoff raises, unless this is ``MAX_VALUE``.  Lowering it pays the cost of
    #: finding them on cheaper searches; raising it searches more limits from every variable.  Zero
    #: asks before the first limit, and ``MAX_VALUE`` never.  Defaults to ``5 * 10**4``, a derived
    #: value (see :ref:`Retuning <tuning-retuning>`): it breaks even with the cost of asking, which
    #: finds the automorphisms, verifies them, and orders their orbits, tens of thousands of states
    #: for a large code.
    automorphism_minimum_states: int = 50_000

    #: A ``CosetLeaderSearch`` with the connected-cluster method makes its checks sparser in up to
    #: this many rounds, each a row reduction in a random order of the columns (Section 3.8 of
    #: ``connected_cluster.md``).  Raising it can speed up queries but slows building the search;
    #: zero only adds checks to each other while that makes them lighter, and ``MAX_VALUE`` leaves
    #: the rounds to ``sparsifying_rounds_without_improvement``.  Defaults to 400, a budget of setup
    #: effort (see :ref:`Retuning <tuning-retuning>`), which a search spends once for all its
    #: queries.
    max_sparsifying_rounds: int = 400

    #: A connected-cluster distance search of a subsystem code makes its derived stabilizers sparser
    #: in up to this many rounds, each a row reduction in a random column order (Section 3.9 of
    #: ``connected_cluster.md``).  Raising it can speed up the search but delays its start.  Zero
    #: skips sparsifying, and ``MAX_VALUE`` leaves the rounds to
    #: ``sparsifying_rounds_without_improvement``.  Defaults to 50, a budget of setup effort (see
    #: :ref:`Retuning <tuning-retuning>`).
    max_distance_sparsifying_rounds: int = 50

    #: Making checks sparser stops early after this many rounds in a row that leave the total weight
    #: of the checks unchanged (see ``max_sparsifying_rounds`` and
    #: ``max_distance_sparsifying_rounds``).  Raising it spends more rounds on checks that rarely
    #: get lighter; zero stops after the first round, and ``MAX_VALUE`` never stops early.  With
    #: both this field and the limit of the rounds at ``MAX_VALUE``, the rounds never end.  Defaults
    #: to 200, a budget of setup effort (see :ref:`Retuning <tuning-retuning>`).
    sparsifying_rounds_without_improvement: int = 200

    # Fields of completion tables.

    #: The number of supports (choices of rows) of an information set from which a distance search
    #: or count considers completion tables for its layer, which skip supports that cannot complete
    #: a word lighter than the best, or, in a count, within the counted weight (Section 3.9 of
    #: ``brouwer_zimmermann.md``).  Considering them samples supports, to predict the cost of the
    #: layer with tables.  Lowering it considers smaller layers; zero every layer, and ``MAX_VALUE``
    #: none.  Defaults to ``2 * 10**7``, a derived value (see :ref:`Retuning <tuning-retuning>`): it
    #: breaks even with the cost of that prediction on a large basis, tens of millions of
    #: evaluations (Section 4.4 of ``brouwer_zimmermann.md``).
    completion_bound_minimum_layer_work: int = 20_000_000

    #: A layer of a distance search or count uses the completion tables that it considers only if
    #: their predicted cost, including building them, is at most this percentage of the cost of
    #: scanning without them.  Lowering it demands larger predicted savings; zero never uses tables,
    #: and ``MAX_VALUE`` uses them wherever they are considered, even where they cost more.
    #: Defaults to 100, a limit at equality (see :ref:`Retuning <tuning-retuning>`): the two costs
    #: are estimates in one unit, which break even at 100, so an error of the cost model calls for
    #: changing its rough estimates, not this limit.
    completion_bound_max_cost_percent: int = 100

    #: The most bytes of completion tables that a distance search keeps for each problem, a count
    #: for each sector besides the tables that it is building, and a coset search of
    #: ``CosetLeaderSearch`` for each set of a family.  Distance searches and counts use blocks of
    #: ``completion_bound_block_bits`` columns if their tables fit, and otherwise none, so a basis
    #: too large for them gets tables only if this limit is raised or the blocks narrowed.  Zero
    #: disables tables, and ``MAX_VALUE`` removes the limit.  Defaults to 256 MiB, a budget of
    #: memory (see :ref:`Retuning <tuning-retuning>`): it fits blocks of 16 columns for bases of up
    #: to about 500 rows and 140 tail columns.
    completion_bound_max_table_bytes: int = 256 * 2**20

    #: The number of tail columns in a block of the completion tables of distance searches and
    #: counts, a power of two up to 16.  Wider blocks skip more supports, but the table of a block
    #: has ``2**width`` bytes for each row of the basis: 1 builds fastest and skips least, and 16
    #: the reverse.  ``completion_bound_max_table_bytes`` says which bases get no tables.  Defaults
    #: to 16, the widest allowed (see :ref:`Retuning <tuning-retuning>`): wider blocks never skip
    #: fewer supports where their tables fit, and ``completion_bound_max_table_bytes`` bounds their
    #: memory.
    completion_bound_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

    #: The number of tail columns in a block of the completion tables of ``CosetLeaderSearch``, a
    #: power of two up to 16.  Its searches by Hamming weight build tables for their costliest
    #: queries, within ``completion_bound_max_table_bytes`` for each set.  Wider blocks skip more
    #: completions but cost more memory and time (see ``completion_bound_block_bits``).  Defaults to
    #: 16, the widest allowed (see :ref:`Retuning <tuning-retuning>`): wider blocks never skip fewer
    #: completions where their tables fit, and ``completion_bound_max_table_bytes`` bounds their
    #: memory.
    coset_completion_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

    #: The number of random supports from which a search estimates the share of a layer that
    #: completion tables would skip, within the budget ``completion_bound_sampling_work``; drawing
    #: them takes about 16 bytes for each sample and each row of the basis.  More samples narrow the
    #: estimate, so that tables pay on more of the layers that accept them, but take longer, and
    #: ``completion_bound_minimum_layer_work`` breaks even with the cost of sampling at the default.
    #: Between 1 and ``2**12``.  Defaults to 1024, a convention (see :ref:`Retuning
    #: <tuning-retuning>`): the standard error of the sampled share is at most 1/64 when the search
    #: keeps all 1024 within ``completion_bound_sampling_work``; on large bases that budget keeps
    #: fewer, and the error is at most ``1 / (2 * sqrt(n))`` for the ``n`` that it keeps.
    completion_bound_samples: int = dataclasses.field(
        default=1024, metadata={"minimum": 1, "maximum": 2**12}
    )

    #: The fewest samples with which a search trusts its estimate of the share that completion
    #: tables would skip; with fewer within the budget, it takes the skipped share to be zero, so
    #: that tables must pay without skipping.  At least 1.  Defaults to 64, a convention (see
    #: :ref:`Retuning <tuning-retuning>`): when every sample skips, the lower end of the confidence
    #: interval (see ``completion_bound_confidence_z``) is 0.94.
    completion_bound_min_samples: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most work that a search spends bounding sampled supports, in units of one pair of rows on
    #: one block of the tables; summing the first rows of a support costs one unit for each row of
    #: the basis.  Raising it allows more samples on large bases, at more cost; zero takes no
    #: samples, so the search declines tables.  Defaults to ``5 * 10**7``, a budget of setup effort
    #: (see :ref:`Retuning <tuning-retuning>`), which bounds the cost of sampling on large bases,
    #: where each sample costs more.
    completion_bound_sampling_work: int = 50_000_000

    #: How many standard errors below the sampled share of skipped supports a search sets its
    #: estimate, the lower end of a Wilson score interval; at most 10.  Raising it declines more
    #: layers where tables would pay; lowering it lets sampling error choose tables that cost more
    #: than they save.  Defaults to 2.0, a convention (see :ref:`Retuning <tuning-retuning>`): the
    #: share that tables skip exceeds the estimate with a one-sided confidence of about 98%.
    completion_bound_confidence_z: float = dataclasses.field(
        default=2.0, metadata={"maximum": 10.0}
    )

    #: The cost of one lookup of completion tables for each block of the tables, in scanned
    #: supports, the unit of their cost model: the time to scan one support without tables.  An
    #: underestimate makes searches use tables that cost more than they save; an overestimate makes
    #: them decline tables that would save time.  Defaults to 2.0, a rough estimate (see
    #: :ref:`Retuning <tuning-retuning>`) for the cost model of Section 3.9 of
    #: ``brouwer_zimmermann.md``.
    completion_bound_lookup_block_cost: float = 2.0

    #: The additional cost of a lookup of completion tables, in scanned supports for each block and
    #: each MiB of tables, since larger tables miss the processor's caches more often.  Raising it
    #: makes searches decline more large tables; lowering it lets them use large tables that may
    #: cost more than they save.  Defaults to 0.04, a rough estimate (see :ref:`Retuning
    #: <tuning-retuning>`).
    completion_bound_lookup_block_cost_per_mib: float = 0.04

    #: A multiplier of the predicted number of lookups of completion tables, which is the share of
    #: pairs of a prefix and the first row of its last pair that a scan with tables keeps, divided
    #: by the share of supports that it keeps.  It is below one, since tables most often skip late
    #: first rows, which have few supports each.  Raising it makes tables look costlier, and
    #: lowering it cheaper.  Defaults to 0.35, a rough estimate (see :ref:`Retuning
    #: <tuning-retuning>`).
    completion_bound_kept_first_row_ratio: float = 0.35

    #: The cost of building completion tables, in scanned supports for each byte of tables; with
    #: ``completion_bound_build_step_cost``, it predicts the cost of a build.  An underestimate
    #: builds tables that small layers do not repay; an overestimate declines tables that would save
    #: time.  Defaults to 0.1, a rough estimate (see :ref:`Retuning <tuning-retuning>`): half of
    #: ``completion_bound_build_step_cost``, a ratio set by hand.  Both err on the high side, so the
    #: model tends to overestimate the cost of builds.
    completion_bound_table_byte_cost: float = 0.1

    #: The cost of building completion tables, in scanned supports for each step of the
    #: breadth-first searches that fill them, as ``_estimate_build_cost`` in
    #: ``_information_sets/completion_bounds.py`` counts steps.  Raising it makes searches decline
    #: tables that are costly to build; lowering it lets them build tables that small layers do not
    #: repay.  Defaults to 0.2, a rough estimate (see :ref:`Retuning <tuning-retuning>`): twice
    #: ``completion_bound_table_byte_cost``, a ratio set by hand, and like it on the high side.
    completion_bound_build_step_cost: float = 0.2

    # Fields of batches and memory limits.

    #: A scan of information sets by the BZ method visits at most
    #: ``2**information_set_batch_exponent`` supports and lookups of completion tables, counted
    #: together, in one call of compiled code, which cannot be interrupted.  Lowering it lets a
    #: keyboard interrupt take effect sooner, but returns to Python more often.  At most 62.
    #: Defaults to 15, a budget of interrupt latency (see :ref:`Retuning <tuning-retuning>`).
    information_set_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    #: The most supports, and lookups of completion tables, counted together, that one compiled call
    #: of the scan of a count, by the BZ or brute-force method, visits; a call also ends at the end
    #: of each coefficient layer of each information set.  Lowering it makes interrupts prompter, at
    #: more overhead; ``MAX_VALUE`` lets each call finish its layer.  At least 1.  Defaults to
    #: 65,536, a budget of interrupt latency (see :ref:`Retuning <tuning-retuning>`).
    scan_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The most chunks of ``scan_chunk_size`` that the scan of a count visits before it returns to
    #: Python, once :func:`~qldpc.codes.distance.logical_counts.compile_search_setup` has compiled
    #: its loop over layers; until then, it returns after every chunk.  It also returns before and
    #: after each coefficient layer of an information set that may use completion tables.  Lowering
    #: it makes interrupts prompter; ``MAX_VALUE`` lets a compiled scan run to its end, or to such a
    #: layer.  At least 1.  Defaults to 64, a budget of interrupt latency (see :ref:`Retuning
    #: <tuning-retuning>`).
    scan_chunks_per_call: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most Gray-code steps that the walk of a check space by the MacWilliams method takes in
    #: one call of compiled code.  Lowering it makes interrupts prompter, but crosses into compiled
    #: code more often, which slows counts.  At least 1.  Defaults to 65,536, a budget of interrupt
    #: latency (see :ref:`Retuning <tuning-retuning>`).
    macwilliams_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The smallest rank of the checks at which the MacWilliams method tests whether the X and Z
    #: checks of a counter span the same space, so that one walk over its ``2**rank`` words serves
    #: both; below it, a second walk costs less than the test.  Lowering it tests on smaller codes;
    #: zero whenever the ranks are equal, and ``MAX_VALUE`` never.  Counters known to be self-dual,
    #: and counts of full Pauli operators, skip the test.  Defaults to 14, chosen by hand: the test
    #: costs about as much as a walk of between ``2**14`` and ``2**15`` steps, so breaking even does
    #: not settle between 14 and 15, and near break-even, testing and a second walk cost a count
    #: about the same.
    macwilliams_shared_walk_rank: int = 14

    #: The most pairs of distinct syndromes of a single-qubit X error and a single-qubit Z error,
    #: each including the identity, that a ``CosetLeaderSearch`` by CSS weight tabulates to answer
    #: queries of weight one quickly; with more, it matches them at each query instead.  Lowering it
    #: saves memory but slows oracles that answer many queries, as in the certification of fault
    #: propagation; zero disables the table.  Defaults to ``2**18``, a budget of tens of megabytes
    #: of memory (see :ref:`Retuning <tuning-retuning>`), which 511 qubits with distinct syndromes
    #: reach.
    max_css_single_pairs: int = 2**18

    def __post_init__(self) -> None:
        """Validate the fields, and store numbers as Python integers and floats.

        Compiled code specializes on the types of its arguments, so storing every integer as a
        Python integer keeps a tuning from compiling code again.

        Raises:
            TypeError: If a field is not a number of the type of its default, or is a bool.
            ValueError: If an integer field is outside its bounds or choices, or a float field is
                negative, above its bound, or not finite.
        """
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
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
