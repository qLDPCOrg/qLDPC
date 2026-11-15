# SPDX-License-Identifier: Apache-2.0

"""Settings and constants that tune the speed and memory of exact distance searches and counts.

A search for an exact distance, or a count of logical operators, chooses a route, such as a search
of disjoint or of overlapping information sets, and limits its batches, memory, and caches.
:class:`SearchTuning` collects the values that tune these choices and limits, from the thresholds
that choose a route to the coefficients of cost models and the sizes of batches.  The page
:doc:`/exact_search_tuning` of the documentation renders this docstring, after tables that lead from
routes and symptoms to these values.  Section 4.4 of ``brouwer_zimmermann.md`` explains the
heuristics that read these values in Brouwer-Zimmermann (BZ) searches, and Sections 3.5, 3.8, and
3.9 of ``connected_cluster.md`` explain those of connected-cluster searches.

This module is private.  Its names, fields, defaults, and constants can change in any release,
without a deprecation period or any other warning, so code that uses it should pin the exact version
of qLDPC that it was written for.  The module is meant for contributors, for benchmarks, and for
experienced users of very costly searches who have found that a different tuning helps them.  The
defaults suit typical searches.

What tuning can and cannot change:

- Tuning changes only how long a search takes and how much memory it uses.  It never changes an
  exact distance, a distance limited by ``max_weight``, or a count of logical operators.
- With an early exit (``cutoff`` of two or more), a distance search returns the weight of a
  nontrivial word at or below ``cutoff`` that it finds, and which word that is depends on tuning.
  The result is still an upper bound between the distance and ``cutoff``, as documented by
  :func:`~qldpc.codes.distance.get_distance_from_generators`.
- Under every tuning, a query of :class:`~qldpc.codes.distance.CosetLeaderSearch` decides the same
  way whether a coset has a word of weight at most the query's maximum weight, and with the
  connected-cluster method it gives the same lower bound.  The word that the query returns, and with
  the BZ method its lower bound, can depend on tuning, within the contract of
  :class:`~qldpc.codes.distance.MinWeightBound`.
- Codes and counters remember their results, and some intermediate results are cached for the whole
  process (see ``BINOMIAL_CACHE_SIZE``, ``CSS_SEARCH_CACHE_SIZE``, ``KRAWTCHOUK_CACHE_SIZE``,
  ``LOCAL_CLIFFORD_CACHE_SIZE``, and ``PAULI_SEARCH_CACHE_SIZE``), so a later search under another
  tuning may reuse them rather than search again.  To time a tuning, run each search in a new
  process, after a first search that loads or compiles code.

Searches use the active tuning, which is the default tuning unless ``use_tuning`` changes some of
its fields within a ``with`` block::

    from sympy.abc import x, y

    from qldpc import codes
    from qldpc.codes.distance import _tuning

    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    with _tuning.use_tuning(max_information_sets=8, lp_minimum_work=0):
        distance = code.get_distance_exact()

Each search reads the active tuning once, when it starts, and uses it throughout.  The active tuning
is a context variable (see the ``contextvars`` module), which ``use_tuning`` changes in the current
context until its block exits.  A copy of the context, such as that of an asyncio task or of a
function that ``asyncio.to_thread`` runs, keeps the tuning that was active when the copy was made,
and forked worker processes inherit the active tuning.  New threads start with the default tuning,
unless Python lets them inherit the context of the thread that starts them (see
``sys.flags.thread_inherit_context``), and spawned worker processes start with the default tuning.
To give workers a tuning, enter ``use_tuning`` in the function that each worker runs.

To reproduce an experiment, record ``saved = dataclasses.asdict(_tuning.get_tuning())`` in the
thread or process that runs its searches, and restore that tuning with ``use_tuning(**saved)``.
Also record the version of qLDPC or, for a checkout of its source, the commit and any differences of
the source from it, since they fix the defaults and the constants.  Changing the value of a field
never compiles code again, but the first search that takes a route compiles the code of that route.
So before timing searches, warm up every route that they may take.

Which values are fields:

Every value that changes only how long a search takes or how much memory it uses, and that a search
can read when it starts and pass down to the code that uses the value, is a field of
:class:`SearchTuning`, whether it sets a trade-off, such as a limit of memory, or is a rough
estimate, such as a coefficient of a cost model.  Since a search reads the fields when it starts, a
field can differ between two searches in one process, and a saved tuning records every field.  The
arguments of the public functions that start searches say instead what to compute and by which
method, such as ``cutoff``, ``max_weight``, and ``method``.  The entry points that read the active
tuning are the distance searches, the counts of logical operators, and
:class:`~qldpc.codes.distance.CosetLeaderSearch`, each when it starts or is built.  Only values that
qLDPC fixes before any search starts are constants: ``MAX_VALUE``, and the sizes of caches, which
qLDPC fixes when it is imported.  Change a constant by editing this module, and start a new process.

.. _tuning-retuning:

Retuning:

Each default is a round value that a simple heuristic gives.  The comment of each field names the
class of its default, among these classes, or says that it is in none:

- A derived value is the gate of an optional step, such as building a family of information sets,
  solving linear programs, or asking for automorphisms.  The step runs once the work that it acts on
  reaches the value.  The value is the cost of the step, known to an order of magnitude, and is 1,
  2, or 5 times a power of ten.  A step can save at most the work that it acts on, so the gate
  breaks even with the cost of the step (Section 4.4 of ``brouwer_zimmermann.md``).  Work and cost
  count candidate/set evaluations in BZ searches, and states in connected-cluster searches.  A step
  costs more on a larger basis or code, and each gate assumes the cost of its step on a large one,
  so a small search can skip a step that would have paid.
- A limit at equality compares two estimates in one unit, and breaks even where they are equal.
- A rough estimate is a coefficient of the cost model of completion tables
  (``predict_completion_cost`` in ``_information_sets/completion_bounds.py``), such as the cost of a
  lookup of one block of the tables, in units of the cost of scanning one support without tables.
  The model decides at break-even with these rough estimates, so its marginal decisions can go
  either way.
- A budget limits memory, interrupt latency, or setup effort, and is chosen for safety rather than
  for speed.  A batch or chunk size keeps the overhead of each call of compiled code small next to
  the work of the call, and keeps keyboard interrupts prompt.  A limit of memory or of setup effort
  bounds the cost of the worst cases.
- A convention is a statistical or numerical setting, and its comment says what it guarantees.
- The widest allowed: the blocks of completion tables take the widest width that searches accept,
  since wider blocks never skip fewer supports where their tables fit, and a budget of memory bounds
  the tables.

Two fields, ``coset_route_minimum_work`` and ``macwilliams_shared_walk_rank``, are in no class: they
were chosen by hand, and their comments say why.  The sizes of caches, which are constants, were
also chosen by hand.  To change the size of a cache, record its hit rate on workloads that reuse its
results, and keep the smallest size beyond which the hit rate stops growing.

Change a default only when its heuristic changes, or when a check shows that the assumption of its
heuristic, such as the cost of a step, is off by more than a factor of about 3.  Never change a
default to gain a factor of two on particular codes.  A new heuristic for a class applies to every
field of the class.

The scripts of ``experiments/distance_tuning`` make such checks, and none of them sets or justifies
a value.  ``constants.py`` checks that the orders of magnitude that the heuristics assume still hold
on a machine, ``trace.py`` shows which decisions a search makes and what each of its stages costs,
and ``sweep.py`` catches gross regressions or bugs when the tuning or the search code changes.  The
README of that directory says how to run them.
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

    The work estimates of Brouwer-Zimmermann (BZ) searches count candidate/set evaluations.  One
    evaluation visits one combination of the rows of one information set, and checks the word of
    that combination.  Section 3.6 of ``brouwer_zimmermann.md`` defines these estimates, which
    assume pessimistically that a search finds no lighter word before it certifies its best word.
    Estimates are not runtime guarantees.  Coefficient layer ``w`` of a search visits the
    combinations of ``w`` rows of each of its information sets, so later layers cost more.

    Every integer field is a Python integer between zero and ``MAX_VALUE``, unless its comment
    states narrower bounds, and every float field is a finite, non-negative float.

    The comment above each field says what the field decides and in what unit, and what raising or
    lowering it does.  The comment then names the class of the default and the rule of that class,
    or, for a field in no class, how its default was chosen (see :ref:`Retuning <tuning-retuning>`).
    Last, the comment points to the section of a guide that explains the heuristic that reads the
    field, if any.
    """

    # Fields of exhaustive enumeration.

    #: The exponent of the batches of exhaustive enumeration, which finds distances by the method
    #: ``"brute_force"``, including the distance that a count of logical operators by that method
    #: finds first (Section 3.6 of ``brouwer_zimmermann.md``).  Enumeration handles at most
    #: ``2**enumeration_batch_exponent`` words at a time in a few arrays of at most
    #: ``2**(enumeration_batch_exponent + 3)`` bytes each, and checks for a keyboard interrupt
    #: between batches.  Raising it can speed up enumeration, at the cost of memory; lowering it
    #: saves memory and makes interrupts prompter, but adds overhead.  At most 62.  Defaults to 15,
    #: a budget of interrupt latency and memory (see :ref:`Retuning <tuning-retuning>`).
    enumeration_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    # Fields of information-set families.

    #: The most information sets in a family of overlapping information sets, or in a family that
    #: matroid partitioning finds (Section 4.4 of ``brouwer_zimmermann.md``).  Distance searches and
    #: counts build both kinds of family, and coset searches build overlapping families for costly
    #: queries.  More sets can let a search stop before costly layers, but setup grows roughly with
    #: the square of this limit, or with its cube when partitioning, and dominates cheap searches.
    #: Zero disables both kinds, but not the maximum disjoint family of a coset search, which
    #: matroid partitioning finds with coverage one.  Defaults to 16, a budget of setup effort (see
    #: :ref:`Retuning <tuning-retuning>`).
    max_information_sets: int = 16

    #: How many random column orders, in a fixed sequence, a search tries when the natural order of
    #: the columns packs fewer full disjoint information sets than ``min(N // r, 2**r - 1)``, for
    #: ``r`` rows and ``N`` nonzero columns, a bound that a basis may not reach (Section 4.4 of
    #: ``brouwer_zimmermann.md``).  The search keeps the best packing.  More orders can find more
    #: full sets, which certify sooner, but each order costs another packing of the basis; zero
    #: keeps the natural order.  Defaults to 8, a budget of setup effort (see
    #: :ref:`Retuning <tuning-retuning>`).
    disjoint_random_column_orders: int = 8

    #: The estimated work of a distance search or count, in candidate/set evaluations, below which
    #: it keeps its disjoint information sets rather than try a family of overlapping ones.
    #: Lowering the field tries overlapping families in cheaper searches, which may not repay
    #: building them; zero always tries, and ``MAX_VALUE`` never does.  Defaults to ``2 * 10**6``, a
    #: derived value (see :ref:`Retuning <tuning-retuning>`): it breaks even with the cost of
    #: building an overlapping family of a large basis, a few million evaluations (Section 4.4 of
    #: ``brouwer_zimmermann.md``).
    overlapping_minimum_work: int = 2_000_000

    #: A query of a ``CosetLeaderSearch`` by the BZ method predicts how many words it examines if it
    #: finds no light enough word.  Once a query predicts this many with the disjoint family of
    #: information sets of its search, the search builds two other families once, a maximum disjoint
    #: family and an overlapping family, and each such query takes the family, of the three, that
    #: does the least work.  In searches by Hamming weight, a query that predicts this many even
    #: with that family also weighs words by their tails and skips completions by tables.  The field
    #: also sets the weight at which the overlapping family is aimed.  Lowering it builds families
    #: and tables for cheaper queries, at more memory, and aims lower; zero builds them at the first
    #: query, and ``MAX_VALUE`` never.  Defaults to ``2 * 10**4``, chosen by hand, preferring speed
    #: to memory, since the field sets three decisions and no break-even rule covers the aim
    #: (Section 4.4 of ``brouwer_zimmermann.md``).
    coset_route_minimum_work: int = 20_000

    #: How many column orders the construction of a family of overlapping information sets tries for
    #: each set that it may add; an order that gives a set already in the family still counts
    #: (Section 4.4 of ``brouwer_zimmermann.md``).  Distance searches and counts build such
    #: families, and coset searches of ``CosetLeaderSearch`` build them for costly queries.  More
    #: attempts can reach a cheaper family, but lengthen setup; zero builds no family.  Defaults to
    #: 2, a budget of setup effort (see :ref:`Retuning <tuning-retuning>`).
    overlapping_attempts_per_set: int = 2

    #: A search tries matroid partitioning only if its estimated work, in candidate/set evaluations,
    #: reaches this multiple of the sum of ``b**2 * k**2 * n`` over the coverages ``b`` that it
    #: tries, for ``k`` rows and ``n`` nonzero columns (see ``get_partitioned_family`` in
    #: ``_information_sets/basis_families.py``).  Lowering the field tries partitioning in cheaper
    #: searches; zero always tries, and ``MAX_VALUE`` never does.  A distance search given
    #: automorphisms skips partitioning if its family of automorphism images (Section 3.7 of
    #: ``brouwer_zimmermann.md``) does no more work than any partitioned family could, and switches
    #: to that family.  Defaults to 5, a derived value (see :ref:`Retuning <tuning-retuning>`): it
    #: breaks even with the cost of partitioning, which is roughly proportional to that sum, at a
    #: few evaluations for each unit (Section 4.4 of ``brouwer_zimmermann.md``).
    partition_work_ratio: int = 5

    #: The estimated work, in candidate/set evaluations, from which a search of overlapping
    #: information sets also bounds unvisited words by linear programming (Section 3.1 of
    #: ``brouwer_zimmermann.md``).  Lowering the field solves linear programs in cheaper searches;
    #: zero in every search of overlapping sets, and ``MAX_VALUE`` in none.  Defaults to
    #: ``2 * 10**8``, a derived value (see :ref:`Retuning <tuning-retuning>`): it breaks even with
    #: the cost of linear programming in a search of a large basis, a few hundred million
    #: evaluations, planning included (Section 4.4 of ``brouwer_zimmermann.md``).
    lp_minimum_work: int = 200_000_000

    #: The certificate from linear programming (Section 3.1 of ``brouwer_zimmermann.md``) rounds
    #: each dual weight that its solver returns to the nearest rational number whose denominator is
    #: at most this field.  Every positive value is sound; a smaller one can round away part of the
    #: bound, and a larger one lengthens exact arithmetic.  Defaults to ``2**20``, a convention (see
    #: :ref:`Retuning <tuning-retuning>`): rounding moves each dual weight by at most ``2**-21``.
    lp_dual_max_denominator: int = dataclasses.field(default=1 << 20, metadata={"minimum": 1})

    #: A distance search that would not partition considers switching to its family of automorphism
    #: images (Section 3.7 of ``brouwer_zimmermann.md``) before its first coefficient layer of at
    #: least this much work, in candidate/set evaluations; a search that would partition considers
    #: the switch at the end of its setup.  After declining, a search considers the switch again
    #: before a later layer of at least this much work, once its best weight drops.  Lowering the
    #: field pays for automorphisms in cheaper searches; zero considers the switch before the first
    #: layer, and ``MAX_VALUE`` never.  Defaults to ``5 * 10**6``, a derived value (see
    #: :ref:`Retuning <tuning-retuning>`): it breaks even with the cost of finding automorphisms,
    #: verifying them, and building the family of their images, several million evaluations for a
    #: large code (Section 4.4 of ``brouwer_zimmermann.md``).
    automorphism_minimum_layer_work: int = 5_000_000

    # Fields of connected-cluster searches.

    #: A connected-cluster search asks its caller for automorphisms before the first weight limit at
    #: which it expects to have visited this many states over all limits of one problem (Section 3.5
    #: of ``connected_cluster.md``).  A state is a support that the search visits.  A search whose
    #: first limit a cutoff raises also asks before that limit, unless the field is ``MAX_VALUE``.
    #: Lowering the field pays the cost of finding automorphisms in cheaper searches; raising it
    #: runs more limits from every variable as a root, rather than from one root for each orbit.
    #: Zero asks before the first limit, and ``MAX_VALUE`` never.  Defaults to ``5 * 10**4``, a
    #: derived value (see :ref:`Retuning <tuning-retuning>`): it breaks even with the cost of
    #: asking, tens of thousands of states for a large code.  Asking finds the automorphisms,
    #: verifies them, and orders their orbits.
    automorphism_minimum_states: int = 50_000

    #: A ``CosetLeaderSearch`` with the connected-cluster method makes its checks sparser in up to
    #: this many rounds, each a row reduction in a random order of the columns (Section 3.8 of
    #: ``connected_cluster.md``).  Raising the field can speed up queries but slows building the
    #: search; zero only adds checks to each other while that makes them lighter, and ``MAX_VALUE``
    #: leaves the number of rounds to ``sparsifying_rounds_without_improvement``.  Defaults to 400,
    #: a budget of setup effort (see :ref:`Retuning <tuning-retuning>`), which a search spends once
    #: for all its queries.
    max_sparsifying_rounds: int = 400

    #: A connected-cluster distance search of a subsystem code makes the stabilizers that it derives
    #: sparser in up to this many rounds, each a row reduction in a random order of the columns
    #: (Section 3.9 of ``connected_cluster.md``).  Raising the field can speed up the search but
    #: delays its start.  Zero skips sparsifying, and ``MAX_VALUE`` leaves the number of rounds to
    #: ``sparsifying_rounds_without_improvement``.  Defaults to 50, a budget of setup effort (see
    #: :ref:`Retuning <tuning-retuning>`).
    max_distance_sparsifying_rounds: int = 50

    #: Making checks sparser stops early after this many rounds in a row that leave the total weight
    #: of the checks unchanged (see ``max_sparsifying_rounds`` and
    #: ``max_distance_sparsifying_rounds``).  Raising the field spends more rounds on checks that
    #: rarely get lighter; zero stops after the first round, and ``MAX_VALUE`` never stops early.
    #: If both this field and the field that limits the rounds are ``MAX_VALUE``, the rounds never
    #: end.  Defaults to 200, a budget of setup effort (see :ref:`Retuning <tuning-retuning>`).
    sparsifying_rounds_without_improvement: int = 200

    # Fields of completion tables.

    #: Completion tables let a scan skip the supports, or choices of rows, that cannot complete a
    #: word lighter than the best one found, or, in a count, within the counted weight (Section 3.9
    #: of ``brouwer_zimmermann.md``).  A distance search or count considers tables for an
    #: information set in a layer only if the set has at least this many supports in the layer, and
    #: considering them samples supports to predict the cost of the layer with tables.  Lowering the
    #: field considers smaller layers; zero every layer, and ``MAX_VALUE`` none.  Defaults to
    #: ``2 * 10**7``, a derived value (see :ref:`Retuning <tuning-retuning>`): it breaks even with
    #: the cost of that prediction on a large basis, tens of millions of evaluations (Section 4.4 of
    #: ``brouwer_zimmermann.md``).
    completion_bound_minimum_layer_work: int = 20_000_000

    #: A layer of a distance search or count uses the completion tables that it considers only if
    #: their predicted cost, including building them if the search does not have them yet, is at
    #: most this percentage of the cost of scanning without them.  Lowering the field demands larger
    #: predicted savings; zero never uses tables, and ``MAX_VALUE`` uses them wherever they are
    #: considered, even where they cost more.  Defaults to 100, a limit at equality (see
    #: :ref:`Retuning <tuning-retuning>`): the two costs are estimates in one unit, which break even
    #: at 100, so an error of the cost model calls for changing its rough estimates, not this limit
    #: (Section 4.4 of ``brouwer_zimmermann.md``).
    completion_bound_max_cost_percent: int = 100

    #: The most bytes of completion tables that a distance search keeps for each problem, that a
    #: count keeps for each sector, besides the tables that it is building, and that a coset search
    #: of ``CosetLeaderSearch`` keeps for each set of a family.  Distance searches and counts use
    #: tables on blocks of ``completion_bound_block_bits`` columns if those tables fit, and
    #: otherwise none, so a basis whose tables do not fit gets tables only if this limit is raised
    #: or the blocks narrowed.  Zero disables tables, and ``MAX_VALUE`` removes the limit.  Defaults
    #: to 256 MiB, a budget of memory (see :ref:`Retuning <tuning-retuning>`): it fits blocks of 16
    #: columns for bases of up to about 500 rows and 140 tail columns.
    completion_bound_max_table_bytes: int = 256 * 2**20

    #: The number of tail columns in a block of the completion tables of distance searches and
    #: counts, a power of two up to 16 (Section 3.9 of ``brouwer_zimmermann.md``).  The table of a
    #: block has ``2**width`` bytes for each row of the basis: 1 builds fastest and skips least, and
    #: 16 the reverse.  ``completion_bound_max_table_bytes`` says which bases get no tables.
    #: Defaults to 16, the widest allowed (see :ref:`Retuning <tuning-retuning>`): wider blocks
    #: never skip fewer supports where their tables fit, and ``completion_bound_max_table_bytes``
    #: bounds their memory.
    completion_bound_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

    #: The number of tail columns in a block of the completion tables of ``CosetLeaderSearch``, a
    #: power of two up to 16.  The searches of ``CosetLeaderSearch`` by Hamming weight build tables
    #: for their costliest queries, within ``completion_bound_max_table_bytes`` for each set
    #: (Section 4.4 of ``brouwer_zimmermann.md``).  Wider blocks can skip more completions but cost
    #: more memory and time (see ``completion_bound_block_bits``).  Defaults to 16, the widest
    #: allowed (see :ref:`Retuning <tuning-retuning>`): wider blocks never skip fewer completions
    #: where their tables fit, and ``completion_bound_max_table_bytes`` bounds their memory.
    coset_completion_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

    #: The number of random supports from which a search estimates the share of a layer that
    #: completion tables would skip, within the budget ``completion_bound_sampling_work``; drawing
    #: them takes about 16 bytes for each sample and each row of the basis.  More samples narrow the
    #: estimate, so that searches use tables on more of the layers where they pay, but take longer,
    #: and ``completion_bound_minimum_layer_work`` assumes the cost of sampling at the default.
    #: Between 1 and ``2**12``.  Defaults to 1024, a convention (see
    #: :ref:`Retuning <tuning-retuning>`): the standard error of the sampled share is at most 1/64
    #: when ``completion_bound_sampling_work`` allows all 1024 samples, and at most
    #: ``1 / (2 * sqrt(n))`` when it allows only ``n``, as on large bases.
    completion_bound_samples: int = dataclasses.field(
        default=1024, metadata={"minimum": 1, "maximum": 2**12}
    )

    #: The fewest samples with which a search trusts its estimate of the share that completion
    #: tables would skip.  With fewer within the budget ``completion_bound_sampling_work``, the
    #: search takes the skipped share to be zero, so that tables must pay without skipping.  At
    #: least 1.  Defaults to 64, a convention (see :ref:`Retuning <tuning-retuning>`): when every
    #: sample skips, the lower end of the confidence interval (see
    #: ``completion_bound_confidence_z``) is 0.94.
    completion_bound_min_samples: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most work that a search spends bounding sampled supports, in units of one pair of rows on
    #: one block of completion tables; summing the rows of a support but its last two costs one unit
    #: for each row of the basis.  Raising the field allows more samples on large bases, at more
    #: cost; zero takes no samples, so the search declines tables.  Defaults to ``5 * 10**7``, a
    #: budget of setup effort (see :ref:`Retuning <tuning-retuning>`), which bounds the cost of
    #: sampling on large bases, where each sample costs more.
    completion_bound_sampling_work: int = 50_000_000

    #: A search estimates the share of supports that completion tables would skip by the lower end
    #: of a Wilson score interval of the sampled share, this many standard errors below it; at most
    #: 10.  Raising the field declines more layers where tables would pay; lowering it lets sampling
    #: error choose tables that cost more than they save.  Defaults to 2.0, a convention (see
    #: :ref:`Retuning <tuning-retuning>`): the share that tables skip exceeds the estimate with a
    #: one-sided confidence of about 98%.
    completion_bound_confidence_z: float = dataclasses.field(
        default=2.0, metadata={"maximum": 10.0}
    )

    #: The cost of one lookup of completion tables for each block of the tables, in scanned
    #: supports, the unit of their cost model: the cost of scanning one support without tables
    #: (Section 3.9 of ``brouwer_zimmermann.md``).  An underestimate makes searches use tables that
    #: cost more than they save; an overestimate makes them decline tables that would save time.
    #: Defaults to 2.0, a rough estimate (see :ref:`Retuning <tuning-retuning>`).
    completion_bound_lookup_block_cost: float = 2.0

    #: The additional cost of a lookup of completion tables, in scanned supports for each block and
    #: each MiB of tables, since larger tables miss the processor's caches more often.  Raising the
    #: field makes searches decline more large tables; lowering it lets them use large tables that
    #: may cost more than they save.  Defaults to 0.04, a rough estimate (see
    #: :ref:`Retuning <tuning-retuning>`).
    completion_bound_lookup_block_cost_per_mib: float = 0.04

    #: A multiplier of the predicted number of lookups of completion tables.  It stands for the
    #: share of heads of supports, their rows but the last, that a scan with tables keeps, divided
    #: by the share of supports that the scan keeps.  It is below one, since tables most often skip
    #: heads whose last row is late, which have few supports each.  Raising it makes tables look
    #: costlier, and lowering it cheaper.  Defaults to 0.35, a rough estimate (see
    #: :ref:`Retuning <tuning-retuning>`).
    completion_bound_kept_first_row_ratio: float = 0.35

    #: The cost of building completion tables, in scanned supports for each byte of tables; with
    #: ``completion_bound_build_step_cost``, it predicts the cost of a build.  An underestimate
    #: builds tables that small layers do not repay; an overestimate declines tables that would save
    #: time.  Defaults to 0.1, a rough estimate (see :ref:`Retuning <tuning-retuning>`): half of
    #: ``completion_bound_build_step_cost``, a ratio set by hand.  Both lean high, so the model
    #: tends to overestimate the cost of builds.
    completion_bound_table_byte_cost: float = 0.1

    #: The cost of building completion tables, in scanned supports for each step of the
    #: breadth-first searches that fill them, as ``_estimate_build_cost`` in
    #: ``_information_sets/completion_bounds.py`` counts steps.  Raising the field makes searches
    #: decline tables that are costly to build; lowering it lets them build tables that small layers
    #: do not repay.  Defaults to 0.2, a rough estimate (see :ref:`Retuning <tuning-retuning>`):
    #: twice ``completion_bound_table_byte_cost``, a ratio set by hand, and like it on the high
    #: side.
    completion_bound_build_step_cost: float = 0.2

    # Fields of batches and memory limits.

    #: A scan of information sets by the BZ method visits at most
    #: ``2**information_set_batch_exponent`` supports and lookups of completion tables, counted
    #: together, in one call of compiled code, which cannot be interrupted.  Lowering the field lets
    #: a keyboard interrupt take effect sooner, but returns to Python more often.  At most 62.
    #: Defaults to 15, a budget of interrupt latency (see :ref:`Retuning <tuning-retuning>`).
    information_set_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    #: One compiled call of the scan of a count, by the BZ or brute-force method, visits at most
    #: this many supports and lookups of completion tables, counted together, and ends at the end of
    #: each coefficient layer of each information set.  Lowering the field makes interrupts
    #: prompter, at more overhead; ``MAX_VALUE`` lets each call finish its layer.  At least 1.
    #: Defaults to 65,536, a budget of interrupt latency (see :ref:`Retuning <tuning-retuning>`).
    scan_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The most chunks of ``scan_chunk_size`` that the scan of a count visits before it returns to
    #: Python, once :func:`~qldpc.codes.distance.logical_counts.compile_search_setup` has compiled
    #: the loop of the scan over layers; until then, the scan returns after every chunk.  It also
    #: returns before and after each coefficient layer of an information set that may use completion
    #: tables.  Lowering the field makes interrupts prompter; ``MAX_VALUE`` lets a compiled scan run
    #: to its end, or to such a layer.  At least 1.  Defaults to 64, a budget of interrupt latency
    #: (see :ref:`Retuning <tuning-retuning>`).
    scan_chunks_per_call: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most Gray-code steps that the walk of a check space by the MacWilliams method takes in
    #: one call of compiled code.  Lowering the field makes interrupts prompter, but calls compiled
    #: code more often, which slows counts.  At least 1.  Defaults to 65,536, a budget of interrupt
    #: latency (see :ref:`Retuning <tuning-retuning>`).
    macwilliams_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The MacWilliams method tests whether the X and Z checks of a counter span the same space only
    #: if both have the same rank, at least this field; if they do span one space, one walk over its
    #: ``2**rank`` words serves both.  Below that rank, a second walk costs less than the test.
    #: Lowering the field tests on smaller codes; zero whenever the ranks are equal, and
    #: ``MAX_VALUE`` never.  Counters known to be self-dual, and counts of full Pauli operators,
    #: skip the test.  Defaults to 14, chosen by hand: the test costs about as much as a walk of
    #: ``2**14`` to ``2**15`` steps, so a break-even rule cannot choose between 14 and 15, and at
    #: either rank, testing and a second walk cost a count about the same.
    macwilliams_shared_walk_rank: int = 14

    #: A ``CosetLeaderSearch`` of Pauli errors by CSS weight answers queries of weight one from a
    #: table, by syndrome, of the products of a single-qubit X error and a single-qubit Z error,
    #: each possibly the identity, if the distinct syndromes of the two kinds of error make at most
    #: this many pairs.  With more pairs, the search matches the two kinds of error at each query
    #: instead.  Lowering the field saves memory but slows oracles that answer many queries, as in
    #: the certification of fault propagation; zero disables the table.  Defaults to ``2**18``, a
    #: budget of memory (see :ref:`Retuning <tuning-retuning>`): a table of tens of megabytes, which
    #: 511 qubits whose single-qubit errors have distinct syndromes reach.
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

#: The work estimates of searches of information sets cache arrays of binomial coefficients, one for
#: each number of rows of a basis, and keep at most this many arrays.  The size takes effect when
#: qLDPC is imported.  A larger cache uses more memory, and a smaller one recomputes arrays that a
#: workload reuses.  Chosen by hand (see :ref:`Retuning <tuning-retuning>`): the array of a basis of
#: ``k`` rows takes ``8 * (k + 1)`` bytes, and few workloads search bases of more than a thousand
#: different numbers of rows.
BINOMIAL_CACHE_SIZE = 1024

#: The most entries that the cache of ``get_css_searches`` in ``_brouwer_zimmermann/coset.py``
#: keeps.  Each entry is a pair of classical searches, of the X and Z parts of errors, for one CSS
#: state and one tuning of the setup of the searches.  The size takes effect when qLDPC is imported.
#: A larger cache uses more memory, and a smaller one repeats the setup of searches that a workload
#: reuses.  Chosen by hand (see :ref:`Retuning <tuning-retuning>`).
CSS_SEARCH_CACHE_SIZE = 8

#: The most results that each of two caches of the MacWilliams method keeps: one cache holds single
#: Krawtchouk rows, each of one weight, and the other holds arrays of the rows of the lightest
#: weights.  The size takes effect when qLDPC is imported.  A larger cache uses more memory, and a
#: smaller one recomputes rows that a workload reuses.  Chosen by hand (see
#: :ref:`Retuning <tuning-retuning>`).
KRAWTCHOUK_CACHE_SIZE = 1024

#: The most results that the cache of the local Clifford matrices of deformed counters keeps.  Each
#: result is a pair of packed matrices, of the X parts and of the Z parts.  The size takes effect
#: when qLDPC is imported.  A larger cache uses more memory, and a smaller one recomputes matrices
#: that a workload reuses.  Only speed and memory depend on it.  Chosen by hand (see
#: :ref:`Retuning <tuning-retuning>`).
LOCAL_CLIFFORD_CACHE_SIZE = 1024

#: The most searches of Pauli errors that the cache of ``get_pauli_search`` in
#: ``_brouwer_zimmermann/coset.py`` keeps, one for each stabilizer matrix, measure of weight, and
#: tuning of the setup of the search.  The size takes effect when qLDPC is imported.  A larger cache
#: uses more memory, a smaller one repeats setups, and zero disables the cache.  Only speed and
#: memory depend on it.  Chosen by hand (see :ref:`Retuning <tuning-retuning>`).
PAULI_SEARCH_CACHE_SIZE = 8
