# SPDX-License-Identifier: Apache-2.0

"""Settings and constants that tune the speed and memory of exact distance searches and counts.

:class:`SearchTuning` collects the values with which a search for an exact distance, a count of
logical operators, or a ``CosetLeaderSearch`` chooses its route and limits its batches and memory.
The page :doc:`/exact_search_tuning` renders this docstring after tables that lead from symptoms and
routes to these values.  Section 4.4 of ``brouwer_zimmermann_code.md`` and Sections 3.4, 3.5,
3.8, 3.9, and 3.11 of ``connected_cluster.md`` explain how searches use the fields, and the section
"Choosing a method" of ``README.md`` how ``method="auto"`` uses its fields.

This module is private: its names, fields, defaults, and constants can change in any release without
warning, so code that uses it should pin the exact version of qLDPC.  It is meant for contributors,
for benchmarks, and for users of very costly searches who have found that another tuning helps them.

What tuning can and cannot change:

- Tuning never changes an exact distance, a distance limited by ``max_weight``, or a count of
  logical operators.
- With an early exit (``cutoff`` of two or more), which nontrivial word at or below ``cutoff`` a
  distance search finds depends on tuning, and, for a Brouwer-Zimmermann search on more than one
  thread, on the timing of the threads.  Its weight still lies between the distance and ``cutoff``
  (see :func:`~qldpc.codes.distance.get_distance_from_generators`).
- Under every tuning, a query of :class:`~qldpc.codes.distance.CosetLeaderSearch` gives the same
  answer to whether a coset has a word of weight at most the query's maximum weight, and with the
  connected-cluster method the same lower bound.  The returned word, and with the Brouwer-Zimmermann
  (BZ) method the query's lower bound, can depend on tuning, within the contract of
  :class:`~qldpc.codes.distance.MinWeightBound`, but not on ``num_threads``.
- Codes and logical-operator counters remember their results, and some caches last for the whole
  process, so a later search under another tuning may reuse results rather than search again.

Searches use the active tuning, which is the default tuning unless ``use_tuning`` changes some of
its fields within a ``with`` block::

    from sympy.abc import x, y

    from qldpc import codes
    from qldpc.codes.distance import _tuning

    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    with _tuning.use_tuning(max_information_sets=8, lp_minimum_work=0):
        distance = code.get_distance_exact(method="brouwer_zimmermann")

A search reads the active tuning once, when it starts.  The active tuning is a context variable (see
the ``contextvars`` module).  So copies of the context, as in asyncio tasks and
``asyncio.to_thread``, and forked worker processes keep the tuning that was active when they were
made.  New threads and spawned worker processes start with the default tuning, unless Python lets
threads inherit the context of the thread that starts them (see
``sys.flags.thread_inherit_context``).  To give workers a tuning, enter ``use_tuning`` in the
function that each worker runs.

To time or reproduce a tuning, run each search in a new process, after a warm-up that compiles the
code of every route that the search may take.  In the thread or process that runs the searches,
record ``saved = dataclasses.asdict(_tuning.get_tuning())``, which ``use_tuning(**saved)`` restores.
A saved ``num_threads`` of ``None`` restores the rule that chooses the number of threads, not the
number, so record the environment variable ``QLDPC_NUM_THREADS`` and
``numba.config.NUMBA_NUM_THREADS`` too.  Also record the version of qLDPC, or its commit and any
changes to its source, which fix the defaults and the constants.

Fields can differ between two searches in one process, but the constants (``MAX_VALUE`` and the
sizes of caches) are fixed when qLDPC is imported: change them by editing this module, and start a
new process.

.. _tuning-retuning:

Retuning:

Each default is in one of these classes, or was chosen by hand:

- A derived value gates an optional step, such as building a family of information sets, solving
  linear programs, or asking for automorphisms.  The step runs when the estimated work that it acts
  on reaches the value, or, for ``partition_work_ratio``, the value times a formula; work counts
  candidate/set evaluations in BZ searches and states in connected-cluster searches.  The value is
  the cost of the step on a large basis or code, or its cost per unit of the formula, known to an
  order of magnitude and written as 1, 2, or 5 times a power of ten (Section 4.4 of
  ``brouwer_zimmermann_code.md``).  On a small search, where the step costs less, a gate can skip a
  step that would have paid.
- A limit at equality compares two estimates in one unit.
- The rough coefficients of the cost model of completion tables (``predict_completion_cost`` in
  ``_information_sets/completion_bounds.py``) are rough estimates, not round values.
- A budget, chosen by hand, limits memory, interrupt latency, or setup effort, which it trades
  against speed.  A batch or chunk size is meant to be the smallest whose overhead per call of
  compiled code is negligible, though not all current sizes meet this.
- A convention is a statistical or numerical setting, and its comment says what it guarantees.
- A switch turns a step of a search on or off, and defaults to on.
- Blocks of completion tables default to the widest allowed width, and a budget bounds their memory
  (Section 4.4 of ``brouwer_zimmermann_code.md``).
- A limit for experiments restricts what searches may build, and its default, ``MAX_VALUE``,
  restricts nothing.

The sizes of caches, ``auto_brute_force_max_words``, ``auto_free_work``,
``connected_cluster_state_evaluations``, ``coset_route_minimum_work``,
``macwilliams_shared_walk_rank``, and ``num_threads`` were chosen by hand, and the comments of those
fields say why.

Change a default only when its heuristic changes, or when a check shows that its assumption, such as
the cost of a step, is off by more than a factor of about 3; only a maintainer changes a budget or a
value chosen by hand.  Never change a default to gain a factor of two on particular codes.  The
scripts of ``experiments/distance_tuning`` check the tuning and set no value, apart from running
searches on one thread: ``constants.py`` checks the orders of magnitude that the heuristics assume,
``trace.py`` shows which decisions a search makes, and ``sweep.py`` catches gross regressions or
bugs.  The README of that directory says how to run them.  They do not check the fields of the
automatic choice of a method.
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

import numpy as np

from ._binary import as_integer

#: The largest value of an integer field.  Searches add and multiply some fields in compiled code
#: with 64-bit integers; a larger value could overflow them, or make numba compile again for another
#: integer type.
MAX_VALUE = 2**62

####################################################################################################
# tuning

# The widths of the blocks of completion tables that searches accept: the powers of two up to 16,
# which divide the 64 bits of a packed word; a block of 32 columns would need ``2**32`` bytes for
# each row of a basis.
_BLOCK_WIDTHS = {"choices": (1, 2, 4, 8, 16)}


@dataclass(frozen=True, kw_only=True)
class SearchTuning:
    """Settings of exact searches and counts, such as thresholds, limits, batch sizes, and switches.

    Work is counted in candidate/set evaluations: an evaluation checks the word of one support (a
    combination of rows of one information set); coefficient layer ``w`` evaluates the supports
    of ``w`` rows of each set (Sections 2.5 and 3.6 of ``brouwer_zimmermann.md``).  Integer fields
    lie between zero and ``MAX_VALUE``, and float fields are finite and non-negative, unless a
    comment narrows them.  Each comment says how its default was chosen
    (:ref:`Retuning <tuning-retuning>`).  The automorphisms that these comments mention are
    permutation automorphisms of a code, which permute qudits and apply no single-qudit gates.
    """

    # Fields of exhaustive enumeration.

    #: Exhaustive enumeration finds distances by the method ``"brute_force"``, also when a count by
    #: that method first finds the distance.  It handles batches of at most
    #: ``2**enumeration_batch_exponent`` packed 64-bit integers, or one word if a word needs more,
    #: and takes keyboard interrupts between batches.  At most 62.  Defaults to 15, a budget of
    #: interrupt latency and memory (Section 3.6 of ``brouwer_zimmermann.md``).
    enumeration_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    # Fields of information-set families.

    #: The most information sets in a family of overlapping information sets, or in a family of
    #: coverage two or more that matroid partitioning finds; zero disables both, but not maximum
    #: disjoint families.  More sets can stop a search before costly layers, but setup grows roughly
    #: with the square of this limit, or with its cube when partitioning.  Defaults to 16, a budget
    #: of setup effort (Section 4.4 of ``brouwer_zimmermann_code.md``).
    max_information_sets: int = 16

    #: How many random column orders, in a fixed sequence, a search tries when the natural order
    #: packs fewer full disjoint information sets than ``min(N // r, 2**r - 1)``, for ``r`` rows and
    #: ``N`` nonzero columns; the search keeps the best packing, and zero keeps the natural order.
    #: Defaults to 8, a budget of setup effort, since each order costs a packing of the basis
    #: (Section 4.4 of ``brouwer_zimmermann_code.md``).
    disjoint_random_column_orders: int = 8

    #: The estimated work of a distance search or count, in candidate/set evaluations, from which it
    #: tries a family of overlapping information sets; zero removes this gate, and ``MAX_VALUE``
    #: closes it.  Defaults to ``2 * 10**6``, a derived value: the cost of building such a family on
    #: a large basis, a few million evaluations (Section 4.4 of ``brouwer_zimmermann_code.md``).
    overlapping_minimum_work: int = 2_000_000

    #: A query of a ``CosetLeaderSearch`` by the BZ method that predicts at least this many
    #: examinations with the greedy disjoint family of its search makes the search try, once, to
    #: build a maximum disjoint and an overlapping family, and the query takes the family that does
    #: the least work.  In searches by Hamming weight, a query that predicts this many with the
    #: family that it takes also weighs words by their tails and skips completions by tables.  The
    #: field also sets the weight at which the overlapping family is aimed; zero tries to build the
    #: families at the first query, and ``MAX_VALUE`` never.  Defaults to ``2 * 10**4``, chosen by
    #: hand, preferring speed to memory, since the field sets three decisions and no break-even rule
    #: covers the aim (Section 4.4 of ``brouwer_zimmermann_code.md``).
    coset_route_minimum_work: int = 20_000

    #: How many column orders the construction of a family of overlapping information sets tries for
    #: each set that it may add, counting orders that give a set already in the family; zero builds
    #: no family.  Defaults to 2, a budget of setup effort (Section 4.4 of
    #: ``brouwer_zimmermann_code.md``).
    overlapping_attempts_per_set: int = 2

    #: A search tries matroid partitioning only if its estimated work, in candidate/set evaluations,
    #: reaches this multiple of the sum of ``b**2 * r**2 * N`` over the coverages ``b`` that it may
    #: try, for ``r`` rows and ``N`` nonzero columns; zero removes this gate, and ``MAX_VALUE``
    #: closes it.  Defaults to 5, a derived value: trying every such coverage costs roughly that sum
    #: times a few evaluations (Section 4.4 of ``brouwer_zimmermann_code.md``).
    partition_work_ratio: int = 5

    #: The largest coverage of the families that matroid partitioning builds in distance searches
    #: and counts.  At least 1.  At 1, partitioning builds the maximum disjoint family, which
    #: ``max_information_sets`` does not limit, instead of families of coverage two or more.  It
    #: skips even that family if the family that it would replace is disjoint and has the pruned
    #: disjoint rank bound.  Defaults to ``MAX_VALUE``, a limit for experiments (Section 4.4 of
    #: ``brouwer_zimmermann_code.md``).
    partition_max_coverage: int = dataclasses.field(default=MAX_VALUE, metadata={"minimum": 1})

    #: The estimated work, in candidate/set evaluations, from which a search of overlapping
    #: information sets also bounds unvisited words by the linear program of Section 3.1 of
    #: ``brouwer_zimmermann.md``; zero in every such search, and ``MAX_VALUE`` in none.  A search
    #: that gets automorphisms solves the program over orbits of Section 3.7 of
    #: ``brouwer_zimmermann.md`` in place of that program, for the family that it has then, disjoint
    #: or overlapping, whatever this value, unless ``automorphism_orbit_bounds`` is off.  Defaults
    #: to ``2 * 10**8``, a derived value: the cost of linear programming, planning included, in a
    #: search of a large basis, a few hundred million evaluations (Section 3.1 of
    #: ``brouwer_zimmermann.md`` and Section 4.4 of ``brouwer_zimmermann_code.md``).
    lp_minimum_work: int = 200_000_000

    #: The largest denominator of the fractions to which the certificate from linear programming
    #: rounds the dual weights of its solver.  Defaults to ``2**20``, a convention: rounding moves
    #: each dual weight by at most ``2**-21`` (Section 4.4 of ``brouwer_zimmermann_code.md``).
    lp_dual_max_denominator: int = dataclasses.field(default=1 << 20, metadata={"minimum": 1})

    #: A distance search that does not ask for automorphisms before matroid partitioning considers
    #: switching to its family of automorphism images before the first coefficient layer whose work,
    #: in candidate/set evaluations, reaches this value; zero considers the switch before the first
    #: layer.  After declining a switch, any search reconsiders before a later such layer once its
    #: upper bound has dropped.  Values above every work estimate, such as ``MAX_VALUE``, turn
    #: automorphisms off in every BZ distance search.  Defaults to ``5 * 10**6``, a derived value:
    #: the cost of finding automorphisms, verifying them, and building their family for a large
    #: code, several million evaluations (Section 3.7 of ``brouwer_zimmermann.md`` and Section 4.4
    #: of ``brouwer_zimmermann_code.md``).
    automorphism_minimum_layer_work: int = 5_000_000

    #: Whether a BZ distance search, once it has verified automorphisms, bounds the weight of
    #: unvisited words with the images of the pivot blocks of its family under them, by a linear
    #: program over their orbits (Section 3.7 of ``brouwer_zimmermann.md``).  ``False`` skips these
    #: bounds, and the search stays exact.  Defaults to ``True``, a switch; ``False`` is for
    #: comparisons, such as ablations.
    automorphism_orbit_bounds: bool = True

    # Fields of distance searches of CSS codes.

    #: Whether a distance search of both sectors of a CSS code, by the BZ or the connected-cluster
    #: method, or by ``"auto"`` unless it enumerates, looks for a qubit permutation that maps one
    #: sector onto the other, and then searches one sector (section "Searching one sector of a CSS
    #: code" of ``README.md``).  ``False`` skips the lookup, and the search stays exact.  Defaults
    #: to ``True``, a switch; ``False`` is for comparisons, and for many cheap searches of large
    #: codes, on which the lookup can cost more than it saves.
    sector_maps: bool = True

    # Fields of the automatic choice of a method.

    #: ``method="auto"`` enumerates every word, as ``"brute_force"`` does, if the enumeration
    #: examines at most this many packed 64-bit integers: ``2**dim * ceil(width / 64)`` for words of
    #: ``width`` bits in a space of dimension ``dim``, summed over the spaces of a search, such as
    #: the two sectors of a CSS code.  Raising it enumerates larger spaces.
    #: Defaults to ``2**30``, chosen by hand: an enumeration of about a second, whose cost, unlike
    #: that of the other methods, is known before it runs.
    auto_brute_force_max_words: int = 2**30

    #: The work, in candidate/set evaluations, below which a search by cost of ``method="auto"``
    #: runs its connected-cluster search without probing or switching: it runs on while its spent
    #: work plus the predicted work of its next weight limit stays within this.  Beyond it, the
    #: search probes with a Brouwer-Zimmermann search, and then considers switching to one.
    #: Connected-cluster work counts each state as ``connected_cluster_state_evaluations``.
    #: Defaults to ``10**8``, chosen by hand: a fraction of a second, below which fixed costs that
    #: the estimates leave out, such as setup, dominate the costs that a switch would compare
    #: (section "Choosing a method" of ``README.md``).
    auto_free_work: int = 100_000_000

    #: The most work, in candidate/set evaluations, that the Brouwer-Zimmermann probe of
    #: ``method="auto"`` may do.  The weight of the lightest word that it finds caps the
    #: connected-cluster search.  The probe skips the costlier stages of setup, and stops before a
    #: layer that would take it over this work.
    #: Defaults to ``10**8``, a budget for the cost of choosing: a fraction of a second (section
    #: "Choosing a method" of ``README.md``).
    auto_probe_work: int = 100_000_000

    #: The cost of a state of a connected-cluster search in candidate/set evaluations, which lets
    #: ``method="auto"`` compare the work of the two searches (section "Choosing a method" of
    #: ``README.md``).  Raising it switches to a Brouwer-Zimmermann search sooner.  Defaults to 500,
    #: chosen by hand: the cost of a state, hundreds of evaluations.
    connected_cluster_state_evaluations: int = 500

    # Fields of connected-cluster searches.

    #: A connected-cluster search asks its caller for automorphisms before the first weight limit at
    #: which it expects to have visited this many states over all limits of one sector, or before a
    #: first limit that ``cutoff`` sets; zero asks before the first limit, and ``MAX_VALUE`` never.
    #: Defaults to ``5 * 10**4``, a derived value: the cost of asking, for a large code, tens of
    #: thousands of states (Section 3.5 of ``connected_cluster.md``).
    automorphism_minimum_states: int = 50_000

    #: Whether a connected-cluster search, of distances or of vectors with a given syndrome,
    #: restricts the subtree of each state to viable symbols (Section 3.4 of
    #: ``connected_cluster.md``).  ``False`` skips the restriction, and the search stays exact.
    #: Defaults to ``True``, a switch; ``False`` is for comparisons, such as ablations.
    restrict_viable_symbols: bool = True

    #: Whether a connected-cluster search, of distances or of vectors with a given syndrome, forbids
    #: the symbols that would make its support meet a known word, such as a stabilizer or gauge
    #: generator, in more than about half that word (Section 3.4 of ``connected_cluster.md``).
    #: ``False`` skips these cuts, and the search stays exact.  Defaults to ``True``, a switch;
    #: ``False`` is for comparisons, such as ablations.
    stabilizer_overlap_cuts: bool = True

    #: Whether a connected-cluster distance search adds sums of pairs of its checks as redundant
    #: checks, and sums of pairs of its known words as known words (Section 3.4 of
    #: ``connected_cluster.md``).  It keeps a sum if it is no heavier than the heavier of its pair,
    #: lightest first while their total weight stays within that of the rows that it sums, and sums
    #: only the known words no heavier than its heaviest check.  ``False`` skips the sums, and the
    #: search stays exact.  The coset searches of ``CosetLeaderSearch`` never add them.  Defaults to
    #: ``True``, a switch; ``False`` is for comparisons, such as ablations.
    pair_sum_checks: bool = True

    #: The most rounds, each a row reduction in a random order of the columns, in which a
    #: ``CosetLeaderSearch`` with the connected-cluster method makes its checks sparser when it is
    #: built; zero only adds checks to each other while that makes them lighter, and ``MAX_VALUE``
    #: leaves the rounds to ``sparsifying_rounds_without_improvement``.  Defaults to 400, a budget
    #: of setup effort, which a search spends once for all its queries (Section 3.8 of
    #: ``connected_cluster.md``).
    max_sparsifying_rounds: int = 400

    #: The most rounds in which a connected-cluster distance search of a subsystem code makes the
    #: stabilizers that it derives sparser; zero skips sparsifying, and ``MAX_VALUE`` leaves the
    #: rounds to ``sparsifying_rounds_without_improvement``.  Defaults to 50, a budget of setup
    #: effort (Section 3.9 of ``connected_cluster.md``).
    max_distance_sparsifying_rounds: int = 50

    #: Making checks sparser stops after this many rounds in a row that leave the total weight of
    #: the checks unchanged; zero stops after the first round, and ``MAX_VALUE`` never stops early,
    #: so the rounds never end if the limit on rounds is ``MAX_VALUE`` too.  Defaults to 200, a
    #: budget of setup effort.
    sparsifying_rounds_without_improvement: int = 200

    # Fields of completion tables.

    #: A distance search or count considers completion tables for an information set in a layer,
    #: sampling supports to predict the cost of the tables, only if the set has at least this many
    #: supports in the layer; zero considers every layer, and ``MAX_VALUE`` none.  Defaults to
    #: ``2 * 10**7``, a derived value: the cost of that prediction on a large basis, tens of
    #: millions of evaluations (Section 3.9 of ``brouwer_zimmermann.md`` and Section 4.4 of
    #: ``brouwer_zimmermann_code.md``).
    completion_bound_minimum_layer_work: int = 20_000_000

    #: A layer of a distance search or count uses the completion tables that it considers only if
    #: their predicted cost, building included unless the search has them, is at most this
    #: percentage of the cost of scanning without them; zero never uses tables, and ``MAX_VALUE``
    #: uses them wherever they are considered.  Defaults to 100, a limit at equality (Section 4.4 of
    #: ``brouwer_zimmermann_code.md``).
    completion_bound_max_cost_percent: int = 100

    #: The most bytes of completion tables kept by a distance search for each sector, and by a count
    #: for each sector besides the tables that it is building; a basis whose tables do not fit gets
    #: none, and zero disables tables.  Defaults to 2 GiB, a budget of memory, which fits blocks of
    #: 16 columns for bases of up to about 500 rows and 1000 tail columns.  Distance searches and
    #: counts build tables only where their cost model accepts them (Section 3.9 of
    #: ``brouwer_zimmermann.md``).
    completion_bound_max_table_bytes: int = 2 * 2**30

    #: The number of tail columns in a block of the completion tables of distance searches and
    #: counts, a power of two up to 16.  The table of a block has ``2**width`` bytes for each row of
    #: the basis, so 1 builds fastest and skips least.  Defaults to 16, the widest allowed (Section
    #: 4.4 of ``brouwer_zimmermann_code.md``).
    completion_bound_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

    #: The number of tail columns in a block of the completion tables that the searches of
    #: ``CosetLeaderSearch`` by Hamming weight build for their costliest queries, a power of two up
    #: to 16.  Defaults to 16, the widest allowed (Section 4.4 of ``brouwer_zimmermann_code.md``).
    coset_completion_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

    #: The most bytes of completion tables, for each set of a family, that the searches of
    #: ``CosetLeaderSearch`` by Hamming weight build for their costliest queries; a set whose tables
    #: do not fit gets none, and zero disables these tables.  Defaults to 256 MiB, a budget of
    #: memory and of setup effort: these queries get the tables of every set that they scan in
    #: layers of two or more rows, if the tables fit, without predicting whether the tables pay
    #: (Section 4.4 of ``brouwer_zimmermann_code.md``).
    coset_completion_max_table_bytes: int = 256 * 2**20

    #: The number of random supports from which a search estimates the share of a layer that
    #: completion tables would skip, within ``completion_bound_sampling_work``; drawing them takes
    #: about 16 bytes for each sample and each row of the basis.  Between 1 and ``2**12``.  Defaults
    #: to 1024, a convention: the standard error of the sampled share is at most 1/64, or
    #: ``1 / (2 * sqrt(n))`` when the budget allows only ``n`` samples.
    completion_bound_samples: int = dataclasses.field(
        default=1024, metadata={"minimum": 1, "maximum": 2**12}
    )

    #: The fewest samples with which a search trusts its estimate of the share that completion
    #: tables would skip; with fewer, it takes the share to be zero.  At least 1.  Defaults to 64, a
    #: convention: when every sample skips, the lower end of the Wilson interval at the default
    #: ``completion_bound_confidence_z`` is 0.94.
    completion_bound_min_samples: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most work that a search spends bounding sampled supports, counting one unit for each pair
    #: of rows on each block of completion tables, and one for each row of the basis in each sample;
    #: zero takes no samples, so the search predicts that tables skip nothing.  Defaults to
    #: ``5 * 10**7``, a budget of setup effort, which bounds the cost of sampling on large bases.
    completion_bound_sampling_work: int = 50_000_000

    #: A search estimates the share of supports that completion tables would skip by the lower end
    #: of a Wilson score interval of the sampled share, this many standard errors below it; at most
    #: 10.  Defaults to 2.0, a convention: the true share exceeds the estimate with a one-sided
    #: confidence of about 98%.
    completion_bound_confidence_z: float = dataclasses.field(
        default=2.0, metadata={"maximum": 10.0}
    )

    #: The cost of one lookup of completion tables for each block of the tables, in scanned
    #: supports, the unit of their cost model: the cost of scanning one support without tables.
    #: Defaults to 2.0, a rough coefficient (Section 4.4 of ``brouwer_zimmermann_code.md``).
    completion_bound_lookup_block_cost: float = 2.0

    #: The additional cost of a lookup of completion tables, in scanned supports for each block and
    #: each MiB of tables, since larger tables miss the processor's caches more often.  Defaults to
    #: 0.04, a rough coefficient.
    completion_bound_lookup_block_cost_per_mib: float = 0.04

    #: A multiplier of the predicted number of lookups of completion tables, an estimate of the
    #: share of heads (supports less their last row) that a scan with tables keeps, divided by the
    #: share of supports that it keeps.  It is below one, since tables most often skip heads whose
    #: last row is late, which have few supports each.  Defaults to 0.35, a rough coefficient.
    completion_bound_kept_first_row_ratio: float = 0.35

    #: The cost of building completion tables, in scanned supports for each byte of tables; with
    #: ``completion_bound_build_step_cost``, it predicts the cost of a build.  Defaults to 0.1, a
    #: rough coefficient, half of ``completion_bound_build_step_cost`` by a ratio chosen by hand.
    completion_bound_table_byte_cost: float = 0.1

    #: The cost of building completion tables, in scanned supports for each step of the
    #: breadth-first searches that fill them, as ``_estimate_build_cost`` in
    #: ``_information_sets/completion_bounds.py`` counts steps.  Defaults to 0.2, a rough
    #: coefficient on the high side.
    completion_bound_build_step_cost: float = 0.2

    # Fields of batches and memory limits.

    #: A distance search by the BZ method scans at most ``2**information_set_batch_exponent``
    #: supports and lookups of completion tables, counted together, in one call of compiled code,
    #: which keyboard interrupts cannot stop; on several threads, each thread scans ``num_threads``
    #: times as many in one call.  A batch query of ``CosetLeaderSearch`` on several threads moves
    #: the rest of its batch to them once its queries on the calling thread have done
    #: ``num_threads * 2**information_set_batch_exponent`` units of work.  With the BZ method, a
    #: query does one unit, and one more for each word that it weighs after the word of its
    #: syndrome; with the connected-cluster method, it does one unit for each state that its
    #: searches visit.  A connected-cluster distance search keeps on one thread a weight limit that
    #: it predicts to visit at most a quarter of
    #: ``num_threads * 2**information_set_batch_exponent`` states, rounded down, since a state
    #: costs far more than a support, and counts at most ``2**information_set_batch_exponent``
    #: states in one call of compiled code, on each thread (Sections 3.6 and 3.11 of
    #: ``connected_cluster.md``).  At most 62.  Defaults to 15, a budget of interrupt latency on one
    #: thread, which grows with the number of threads for BZ scans.
    information_set_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    #: The number of threads on which a distance search by the BZ method scans a costly layer of a
    #: set, which they split into work items, sharing one copy of its completion tables; on which a
    #: connected-cluster distance search searches a costly weight limit; and on which the batch
    #: queries of ``CosetLeaderSearch`` search the blocks of a batch.  At least 1, and 1 runs on the
    #: calling thread alone.  ``None`` takes the value of the environment variable
    #: ``QLDPC_NUM_THREADS`` when qLDPC was imported, if it was set, and otherwise
    #: ``numba.config.NUMBA_NUM_THREADS``; the section "Threads" of the README of
    #: ``qldpc/codes/distance`` says how to set either, and when ``QLDPC_NUM_THREADS`` also sets
    #: numba's own count.  A distance search resolves ``None`` once, when it starts, and a
    #: ``CosetLeaderSearch`` when it is built.  A layer of a set runs on the calling thread alone
    #: unless it has at least three rows and more than
    #: ``num_threads * 2**information_set_batch_exponent`` supports, and a batch until its queries
    #: have done that many units of work, in the units that ``information_set_batch_exponent``
    #: defines for each method (Section 4.5 of ``brouwer_zimmermann_code.md`` and Section 3.11 of
    #: ``connected_cluster.md``).  A connected-cluster weight limit runs on the calling thread
    #: alone unless it is predicted to visit more than a quarter of
    #: ``num_threads * 2**information_set_batch_exponent`` states, rounded down (Section 3.11 of
    #: ``connected_cluster.md``).  Programs that run searches in several processes or threads at
    #: once should cap it, as with ``QLDPC_NUM_THREADS=1`` set before Python starts.
    #: ``use_tuning(num_threads=1)`` caps a worker only if entered in the function that it runs,
    #: since new threads and spawned processes normally start with the default tuning (see the
    #: module docstring).  Defaults to ``None``, chosen by hand: a search uses the cores that numba
    #: counts unless its caller caps them.
    num_threads: int | None = dataclasses.field(default=None, metadata={"minimum": 1})

    #: A chunk of the scan of a count, by the BZ or brute-force method, visits at most this many
    #: supports and lookups of completion tables, counted together, within one coefficient layer of
    #: one information set.  At least 1.  Defaults to 65,536, a budget of interrupt latency.
    scan_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The most chunks that the scan of a count visits before it returns to Python, once
    #: :func:`~qldpc.codes.distance.logical_counts.compile_search_setup` has compiled the loop of
    #: the scan over layers; until then, the scan returns after every chunk.  The scan also returns
    #: before and after each coefficient layer of an information set that may use completion tables.
    #: At least 1.  Defaults to 64, a budget of interrupt latency.
    scan_chunks_per_call: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most Gray-code steps that the walk of a check space by the MacWilliams method takes in
    #: one call of compiled code.  At least 1.  Defaults to 65,536, a budget of interrupt latency.
    macwilliams_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The least rank at which the MacWilliams method tests whether the X and Z checks of a counter,
    #: of equal rank, span the same space, so that one walk over its ``2**rank`` words serves both;
    #: zero tests whenever the ranks are equal, and ``MAX_VALUE`` never.  Counters known to be
    #: self-dual, and counts of full Pauli operators, skip the test.  Defaults to 14, chosen by
    #: hand, near where the test and a second walk break even.
    macwilliams_shared_walk_rank: int = 14

    #: Applies to a ``CosetLeaderSearch`` of Pauli errors by CSS weight, with the Brouwer-Zimmermann
    #: method, on a stabilizer state that is not CSS.  Such a search answers queries of weight one
    #: from a table, by syndrome, of the products of a single-qubit X error and a single-qubit Z
    #: error, each possibly the identity.  It builds the table if the distinct syndromes of the two
    #: kinds of error make at most this many pairs, and otherwise matches syndromes at each query.
    #: Zero disables the table.  Defaults to ``2**18``, a budget of memory: a table of tens of
    #: megabytes, reached at 511 qubits whose single-qubit errors have distinct syndromes.
    max_css_single_pairs: int = 2**18

    def __post_init__(self) -> None:
        """Validate the fields, and store them as Python bools, integers, and floats.

        Compiled code specializes on the types of its arguments, so storing every integer as a
        Python integer keeps a tuning from compiling code again.

        A field whose default is None also accepts None.

        Raises:
            TypeError: If a bool field is not a Python or NumPy bool, or a number field is a bool or
                is not a number of the type of its default.
            ValueError: If an integer field is outside its bounds or choices, or a float field is
                negative, above its bound, or not finite.
        """
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if value is None and field.default is None:
                continue
            if isinstance(field.default, bool):
                if not isinstance(value, bool | np.bool_):
                    raise TypeError(f"{field.name} must be a bool, not {value!r}")
                object.__setattr__(self, field.name, bool(value))
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

#: The most arrays of binomial coefficients that work estimates cache, one for each number ``r`` of
#: rows of a basis, of ``8 * (r + 1)`` bytes each; few workloads search bases of more than a
#: thousand different numbers of rows.
BINOMIAL_CACHE_SIZE = 1024

#: The most pairs of classical searches, of the X and Z parts of errors, that ``get_css_searches``
#: in ``_brouwer_zimmermann/coset.py`` caches, one for each CSS state and tuning of their setup.
CSS_SEARCH_CACHE_SIZE = 8

#: The most results that each of two caches of the MacWilliams method keeps, one for single
#: Krawtchouk rows and one for arrays of the Krawtchouk rows of the lightest weights.
KRAWTCHOUK_CACHE_SIZE = 1024

#: The most local Clifford matrices of deformed counters that their cache keeps, each a pair of
#: packed matrices, of X parts and of Z parts.
LOCAL_CLIFFORD_CACHE_SIZE = 1024

#: The most searches of Pauli errors that ``get_pauli_search`` in ``_brouwer_zimmermann/coset.py``
#: caches, one for each stabilizer matrix, measure of weight, and tuning of the setup; zero disables
#: the cache.
PAULI_SEARCH_CACHE_SIZE = 8
