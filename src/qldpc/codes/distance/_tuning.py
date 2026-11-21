# SPDX-License-Identifier: Apache-2.0

"""Settings and constants that tune the speed and memory of exact distance searches and counts.

:class:`SearchTuning` collects the values with which a search for an exact distance, a count of
logical operators, or a ``CosetLeaderSearch`` chooses its route and limits its batches and memory.
The page :doc:`/exact_search_tuning` renders this docstring after tables that lead from symptoms and
routes to these values.

This module is private: its names, fields, defaults, and constants can change in any release without
warning, so code that uses it should pin the exact version of qLDPC.  It is meant for contributors,
for benchmarks, and for users of very costly searches who have found that another tuning helps them.

What tuning can and cannot change:

- Tuning never changes an exact distance, a distance limited by ``max_weight``, or a count of
  logical operators.
- A distance search with an early exit (``cutoff`` of two or more) stops once it finds a nontrivial
  word at or below ``cutoff``.  Which word depends on tuning, and, for a Brouwer-Zimmermann (BZ)
  search on several threads, on the timing of the threads.  Its weight still lies between the
  distance and ``cutoff`` (see :func:`~qldpc.codes.distance.get_distance_from_generators`).
- Every tuning gives a query of :class:`~qldpc.codes.distance.CosetLeaderSearch` the same answer to
  whether a coset has a word within the query's maximum weight.  With the connected-cluster method,
  it also gives the same lower bound on the weight of every word of the coset.  The returned word,
  and with the BZ method the lower bound, can depend on tuning, within the contract of
  :class:`~qldpc.codes.distance.MinWeightBound`, but not on ``num_threads``.
- Codes and logical-operator counters remember their results, and some caches last for the whole
  process, so a later search under another tuning may reuse results rather than search again.

Searches use the active tuning, which is the default tuning unless ``use_tuning`` changes some of
its fields within a ``with`` block::

    from sympy.abc import x, y

    from qldpc import codes
    from qldpc.codes.distance import _tuning

    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    with _tuning.use_tuning(max_information_sets=8, overlapping_minimum_work=0):
        distance = code.get_distance_exact(method="brouwer_zimmermann")

A distance search or count reads the active tuning once, when it starts.  A ``CosetLeaderSearch``
reads it when it is built, and its queries keep that tuning.  The active tuning is a context
variable (see the ``contextvars`` module).  So copies of the context, as in asyncio tasks and
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

Each field description names one of the following kinds of default, or says why its default was
chosen by hand:

- A derived value gates an optional step, such as building a family of information sets or asking
  for automorphisms.  The step runs when the estimated work that it acts on reaches the value, or,
  for ``partition_work_ratio``, the value times the formula that its description gives.  The value
  is the cost of the step on a large basis or code, or its cost per unit of the formula.  It is
  known to an order of magnitude, and written as 1, 2, or 5 times a power of ten (Section 4.4 of
  ``brouwer_zimmermann_code.md``).  On a small search, where the step costs less, a gate can skip a
  step that would have paid.
- A limit at equality compares two estimates in one unit, and defaults to the value at which they
  are equal.
- A rough coefficient is a coefficient of the cost model of completion tables
  (``predict_completion_cost`` in ``_information_sets/completion_bounds.py``), known only roughly.
- A budget, chosen by hand, limits memory, interrupt latency, or setup effort, which it trades
  against speed.  A batch or chunk size is meant to be the smallest whose overhead per call of
  compiled code is negligible.  Not every current size meets this.
- A convention is a statistical or numerical setting, and its description says what it guarantees.
- A switch turns a step of a search on or off, and defaults to on.
- Blocks of completion tables default to the widest allowed width, and a budget bounds their memory
  (Section 4.4 of ``brouwer_zimmermann_code.md``).
- A limit for experiments restricts what searches may build, and its default, ``MAX_VALUE``,
  restricts nothing.

The sizes of caches, which are constants, were chosen by hand.  Each is the ``maxsize`` of a
``functools.lru_cache``, so zero disables its cache.

Change a default only when its heuristic changes, or when a check shows that its assumption, such as
the cost of a step, is off by more than a factor of about 3.  Only a maintainer changes a default
chosen by hand.  Never change a default to gain a factor of two on particular codes.  The scripts of
``experiments/distance_tuning`` check the tuning and set no default, and the README of that
directory lists them and says how to run them.  They do not check the fields that only
``method="auto"`` reads: ``auto_brute_force_max_words``, ``auto_free_work``, ``auto_probe_work``,
and ``connected_cluster_state_evaluations``.
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

from ._code_matrices import as_integer

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

    The descriptions of the fields use the following terms.  The section
    `Terms <https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/codes/distance/README.md#terms>`__
    of the README of the distance package defines sectors, search bases, and searches by Hamming
    weight more fully, and also defines other terms, such as information set and automorphism.

    - Sector: a space of words that a search or count examines separately for nontrivial words,
      such as the X sector of a CSS stabilizer code: the X-type operators that commute with the Z
      stabilizers.  A search of a classical code, of a qubit code that is not CSS, or of a circuit
      examines one sector.
    - Search basis, or set for short: a basis that a BZ search or count scans, row reduced on a
      block of its columns.  It spans a sector, or, in a coset search of ``CosetLeaderSearch``, the
      binary code whose cosets the search examines.  A full set is one whose pivot columns form an
      information set.
    - Tail columns: the columns of a search basis other than its pivot columns.  The tail of a word
      is its restriction to them (Section 3.5 of ``brouwer_zimmermann.md``).
    - Support: in a BZ search or count, a set of rows of a search basis.  The word of a support is
      the sum of its rows, plus, in a coset search of ``CosetLeaderSearch``, a fixed word of the
      coset (Section 3.5).
    - Coefficient layer ``w``, or layer: the supports of ``w`` rows of a search basis (Section 2.5).
    - Candidate/set evaluation: the examination of one support in one search basis (Section 2.5).
    - Symbol: in a connected-cluster search, a coordinate of a binary word, or, in a search of Pauli
      operators, a qubit with one of the values X, Y, and Z (Section 3.7 of
      ``connected_cluster.md``).
    - State: in a connected-cluster search, a set of symbols that the search has chosen, with the
      symbols that it forbids adding (Section 3.3 of ``connected_cluster.md``).
    - Weight limit: the largest weight of the words that a connected-cluster search of a sector
      looks for; the search raises it step by step (Section 3.6 of ``connected_cluster.md``).
    - Searches by Hamming weight: the BZ searches of a ``CosetLeaderSearch`` that weigh each binary
      word that they scan by its Hamming weight divided by a fixed number.  The number is two for
      the searches by physical weight, which scan the images of Pauli errors under the map of
      Section 2.7 of ``brouwer_zimmermann.md``.  It is one for the searches of X parts and of Z
      parts into which a ``CosetLeaderSearch`` splits a query by CSS weight on a CSS stabilizer
      state.

    Work is counted in candidate/set evaluations unless a description names another unit.  Integer
    fields lie between zero and ``MAX_VALUE``, and float fields are finite and non-negative.  A
    description can narrow these ranges.  Each description says how its default was chosen
    (:ref:`Retuning <tuning-retuning>`).
    """

    # Fields of exhaustive enumeration.

    #: The base-2 logarithm of the largest number of packed 64-bit integers in a batch of exhaustive
    #: enumeration.  A batch holds at least one word, even if that word alone needs more integers.
    #: Exhaustive enumeration finds distances by the method ``"brute_force"``, by ``"auto"`` when
    #: enumerating is cheap, and when a count by ``"brute_force"`` first finds the distance.  It
    #: takes keyboard interrupts between batches.  At most 62.  Defaults to 15, a budget of
    #: interrupt latency and memory (Section 3.6 of ``brouwer_zimmermann.md``).
    enumeration_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    # Fields of information-set families.

    #: The largest number of information sets in a family of overlapping information sets.  It also
    #: caps a family that matroid partitioning finds with a coverage ``b`` of two or more, in which
    #: a column can lie in up to ``b`` sets.  Zero disables both kinds of family, but not maximum
    #: disjoint families.  More sets can stop a search before costly layers, but setup grows roughly
    #: with the square of this limit, or with its cube when partitioning, so it is at most
    #: ``2**10``.  Defaults to 16, a budget of setup effort (Section 4.4 of
    #: ``brouwer_zimmermann_code.md``).
    max_information_sets: int = dataclasses.field(default=16, metadata={"maximum": 2**10})

    #: How many random column orders, in a fixed sequence, a search tries in order to pack more
    #: disjoint full sets.  Let a basis have ``r`` rows and ``N`` nonzero columns.  A search tries
    #: the orders only while its packing has fewer than ``min(N // r, 2**r - 1)`` full sets, and it
    #: keeps the best packing.  Zero keeps the natural order.  Defaults to 8, a budget of setup
    #: effort, since each order costs a packing of the basis (Section 4.4 of
    #: ``brouwer_zimmermann_code.md``).
    disjoint_random_column_orders: int = 8

    #: The estimated work of a distance search or count, in candidate/set evaluations, from which it
    #: tries a family of overlapping information sets; zero removes this gate, and ``MAX_VALUE``
    #: closes it.  Defaults to ``2 * 10**6``, a derived value: the cost of building such a family on
    #: a large basis, a few million evaluations (Section 4.4 of ``brouwer_zimmermann_code.md``).
    overlapping_minimum_work: int = 2_000_000

    #: A threshold on predicted examinations, the number of words that a query of a
    #: ``CosetLeaderSearch`` by the BZ method is predicted to examine or skip.  Each BZ search of a
    #: ``CosetLeaderSearch`` decides how to search once for each maximum weight, before it first
    #: searches a coset at that maximum weight:
    #:
    #: 1. It predicts the examinations with the disjoint family that it packed when the search was
    #:    built (see ``disjoint_random_column_orders``).  Below this value, it takes that family and
    #:    skips steps 2 and 3.
    #: 2. Otherwise, it tries once, for the whole search, to build a maximum disjoint family and an
    #:    overlapping family.  It then takes the family predicted to do the least work: the family
    #:    of step 1 or one that it built.
    #: 3. If it is a search by Hamming weight, it then predicts the examinations with the family
    #:    that it takes.  If they also reach this value, it computes the weight of each word from
    #:    the word's support and tail (Section 3.5 of ``brouwer_zimmermann.md``), and uses
    #:    completion tables to skip groups of words that are all too heavy (see
    #:    ``coset_completion_max_table_bytes``).
    #:
    #: Zero always goes on to step 2 and always passes the test of step 3; ``MAX_VALUE`` always
    #: skips steps 2 and 3.  Defaults to ``2 * 10**4``, chosen by hand (the ``CosetRoutes`` item of
    #: Section 4.4 of ``brouwer_zimmermann_code.md`` says why).
    coset_route_minimum_work: int = 20_000

    #: How many column orders the construction of a family of overlapping information sets tries for
    #: each set that it may add, counting orders that give a set already in the family; zero builds
    #: no family.  Defaults to 2, a budget of setup effort (Section 4.4 of
    #: ``brouwer_zimmermann_code.md``).
    overlapping_attempts_per_set: int = 2

    #: A distance search or count tries matroid partitioning only if its estimated work reaches
    #: ``partition_work_ratio * sum(b**2 * r**2 * N)``.  The sum runs over the coverages ``b`` that
    #: the search may try, for a basis of ``r`` rows and ``N`` nonzero columns.  A BZ distance
    #: search gets automorphisms before partitioning only if it passes this gate (Section 3.7 of
    #: ``brouwer_zimmermann.md``).  Zero removes the gate, and ``MAX_VALUE`` closes it.  Defaults to
    #: 2, a derived value: trying every such coverage costs roughly that sum times one or two
    #: evaluations when coverages of two or more are tried, and less at coverage one alone (Section
    #: 4.4 of ``brouwer_zimmermann_code.md``).
    partition_work_ratio: int = 2

    #: The largest coverage of the families that matroid partitioning builds in distance searches
    #: and counts.  At least 1.  At 1, partitioning builds the maximum disjoint family, which
    #: ``max_information_sets`` does not limit, instead of families of coverage two or more.  It
    #: skips the maximum disjoint family when a test shows that this family could not do less work
    #: than the family that the search has so far (Section 4.4 of ``brouwer_zimmermann_code.md``).
    #: Defaults to ``MAX_VALUE``, a limit for experiments.
    partition_max_coverage: int = dataclasses.field(default=MAX_VALUE, metadata={"minimum": 1})

    #: A BZ distance search can be given automorphisms, or a function that finds them.  The search
    #: of each sector gets them at most once, and builds from them a family of the images of one
    #: set.  The search considers switching to that family before the first coefficient layer whose
    #: work reaches this value, and gets the automorphisms then if it has not yet.  If it got them
    #: before matroid partitioning (see ``partition_work_ratio``), it considers the switch as soon
    #: as setup ends instead.  Zero considers the switch before the first layer.  After declining a
    #: switch, at either point, a search reconsiders before a later such layer once its upper bound
    #: has dropped.  Values above ``2**61``, the cap of work estimates (``WORK_CAP`` in
    #: ``_information_sets/certificates.py``), such as ``MAX_VALUE``, turn automorphisms off in
    #: every BZ distance search.  Defaults to ``5 * 10**6``, a derived value: the cost of finding
    #: automorphisms, verifying them, and building their family for a large code, several million
    #: evaluations (Section 3.7 of ``brouwer_zimmermann.md`` and Section 4.4 of
    #: ``brouwer_zimmermann_code.md``).
    automorphism_minimum_layer_work: int = 5_000_000

    # Fields of distance searches of CSS codes.

    #: Whether a distance search of both sectors of a CSS code looks for a qubit permutation that
    #: maps one sector onto the other, and then searches only one sector (section "Searching one
    #: sector of a CSS code" of ``problems.md``).  This applies to the BZ and connected-cluster
    #: methods, and to ``"auto"`` unless it enumerates.  ``False`` skips the lookup, and the search
    #: stays exact.  Defaults to ``True``, a switch; ``False`` is for comparisons, and for many
    #: cheap searches of large codes, on which the lookup can cost more than it saves.
    sector_maps: bool = True

    # Fields of the automatic choice of a method.

    #: ``method="auto"`` enumerates every word if the enumeration examines at most this many packed
    #: 64-bit integers.  A space of dimension ``dim``, with words of ``width`` bits, needs
    #: ``2**dim * ceil(width / 64)`` integers.  The enumeration sums this number over the spaces of
    #: the search, such as the two sectors of a CSS code (section "Choosing a method" of
    #: ``problems.md``).  Zero never enumerates.  Defaults to ``2**20``, chosen by hand: an
    #: enumeration of about a millisecond, the fixed cost of a Brouwer-Zimmermann search of a small
    #: code.  On larger spaces, a Brouwer-Zimmermann search is usually faster (section "Refining the
    #: choice" of ``problems.md`` names exceptions).
    auto_brute_force_max_words: int = 2**20

    #: The work, in candidate/set evaluations, that ``method="auto"`` lets its connected-cluster
    #: search do without probing or switching.  That search runs on while its spent work plus the
    #: predicted work of its next weight limit stays within this value, counting each state as
    #: ``connected_cluster_state_evaluations``.  Beyond it, the search probes with a
    #: Brouwer-Zimmermann search, and then considers switching to one.  Zero probes as soon as that
    #: sum is positive.  Defaults to ``10**8``, chosen by hand: a fraction of a second (section
    #: "Choosing a method" of ``problems.md`` says why).
    auto_free_work: int = 100_000_000

    #: The most work, in candidate/set evaluations, that the Brouwer-Zimmermann probe of
    #: ``method="auto"`` may do.  The probe's upper bound on the distance caps the connected-cluster
    #: search.  The probe skips the costlier stages of setup, the families of overlapping and
    #: partitioned information sets, whatever ``overlapping_minimum_work`` and
    #: ``partition_work_ratio`` say.  It stops before a layer that would take it over this work, so
    #: zero stops it after its setup.  Defaults to ``10**8``, a budget of setup effort, which the
    #: choice of a method spends: a fraction of a second (section "Choosing a method" of
    #: ``problems.md``).
    auto_probe_work: int = 100_000_000

    #: The cost of a state of a connected-cluster search in candidate/set evaluations, which lets
    #: ``method="auto"`` compare the work of the two searches (section "Choosing a method" of
    #: ``problems.md``).  Raising it switches to a Brouwer-Zimmermann search sooner.  Zero never
    #: probes or switches.  Defaults to 500, chosen by hand: the cost of a state, hundreds of
    #: evaluations.
    connected_cluster_state_evaluations: int = 500

    # Fields of connected-cluster searches.

    #: A connected-cluster distance search can be given automorphisms, or a function that finds
    #: them.  The search of each sector gets and verifies them at most once.  It does so before the
    #: first weight limit at which the number of states that it has visited in the sector, plus the
    #: number that it predicts for that limit, reaches this value.  If ``cutoff`` raises the first
    #: limit, the search gets the automorphisms before that limit, unless this field is
    #: ``MAX_VALUE``.  Zero gets them before the first limit, and ``MAX_VALUE`` never.  The coset
    #: searches of ``CosetLeaderSearch`` never use automorphisms.  Defaults to ``5 * 10**4``, a
    #: derived value: the cost of finding automorphisms, verifying them, and ordering their orbits
    #: for a large code, tens of thousands of states (Section 3.5 of ``connected_cluster.md``).
    automorphism_minimum_states: int = 50_000

    #: Whether a connected-cluster search, of distances or of vectors with a given syndrome,
    #: restricts the subtree of each state to viable symbols (Section 3.4 of
    #: ``connected_cluster.md``).  ``False`` skips the restriction, and the search stays exact.
    #: Defaults to ``True``, a switch; ``False`` is for comparisons, such as ablations.
    restrict_viable_symbols: bool = True

    #: Whether a connected-cluster search, of distances or of vectors with a given syndrome, applies
    #: overlap cuts from known words, such as stabilizers or gauge generators, to forbid symbols at
    #: each state (Sections 3.4, 3.7, and 3.8 of ``connected_cluster.md``).  ``False`` skips these
    #: cuts, and the search stays exact.  Defaults to ``True``, a switch; ``False`` is for
    #: comparisons, such as ablations.
    stabilizer_overlap_cuts: bool = True

    #: Whether a connected-cluster distance search adds sums of pairs of its checks as redundant
    #: checks, and sums of pairs of its known words as known words, keeping only light sums (Section
    #: 3.4 of ``connected_cluster.md``).  ``False`` skips the sums, and the search stays exact.  The
    #: coset searches of ``CosetLeaderSearch`` never add them.  Defaults to ``True``, a switch;
    #: ``False`` is for comparisons, such as ablations.
    pair_sum_checks: bool = True

    #: The most rounds in which a ``CosetLeaderSearch`` with the connected-cluster method makes its
    #: checks sparser when it is built.  Each round is a row reduction in a random order of the
    #: columns.  Zero only adds checks to each other while that makes them lighter, and
    #: ``MAX_VALUE`` leaves the rounds to ``sparsifying_rounds_without_improvement``.  Defaults to
    #: 400, a budget of setup effort, which a search spends once for all its queries (Section 3.8 of
    #: ``connected_cluster.md``).
    max_coset_sparsifying_rounds: int = 400

    #: The most rounds in which a connected-cluster distance search of a subsystem code makes the
    #: stabilizers that it derives sparser; zero skips sparsifying, and ``MAX_VALUE`` leaves the
    #: rounds to ``sparsifying_rounds_without_improvement``.  Defaults to 50, a budget of setup
    #: effort (Section 3.9 of ``connected_cluster.md``).
    max_distance_sparsifying_rounds: int = 50

    #: Making checks sparser stops after this many rounds in a row that leave the total weight of
    #: the checks unchanged.  Zero stops after the first round, and ``MAX_VALUE`` never stops early,
    #: so the rounds never end if ``max_coset_sparsifying_rounds`` or
    #: ``max_distance_sparsifying_rounds``, whichever applies, is ``MAX_VALUE`` too.  The field is
    #: shared: a connected-cluster ``CosetLeaderSearch`` reads it when it is built, and so does a
    #: connected-cluster distance search of a subsystem code, for the stabilizers that it derives
    #: (Sections 3.8 and 3.9 of ``connected_cluster.md``).  Defaults to 200, a budget of setup
    #: effort.
    sparsifying_rounds_without_improvement: int = 200

    # Fields of completion tables.

    #: A distance search or count considers completion tables for a search basis in a layer only if
    #: the search basis has at least this many supports in the layer.  Considering them means
    #: sampling supports to predict the cost of the tables.  Zero considers every layer, and
    #: ``MAX_VALUE`` none.  Defaults to ``2 * 10**7``, a derived value: the cost of that prediction
    #: on a large basis, tens of millions of evaluations (Section 3.9 of ``brouwer_zimmermann.md``
    #: and Section 4.4 of ``brouwer_zimmermann_code.md``).
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
    #: memory and of setup effort.  These queries get the tables of every set that they scan in
    #: layers of two or more rows, if the tables fit.  They do not predict whether the tables pay
    #: (Section 4.4 of ``brouwer_zimmermann_code.md``).
    coset_completion_max_table_bytes: int = 256 * 2**20

    #: The number of random supports from which a distance search or count estimates the share of a
    #: layer that completion tables would skip.  ``completion_bound_sampling_work`` can allow fewer.
    #: Drawing them takes about 16 bytes for each sample and each row of the basis.  Between 1 and
    #: ``2**12``.  Defaults to 1024, a convention: the standard error of the sampled share is at
    #: most 1/64, or ``1 / (2 * sqrt(n))`` when the budget allows only ``n`` samples (Section 3.9 of
    #: ``brouwer_zimmermann.md`` and Section 4.4 of ``brouwer_zimmermann_code.md``).
    completion_bound_samples: int = dataclasses.field(
        default=1024, metadata={"minimum": 1, "maximum": 2**12}
    )

    #: The fewest samples with which a distance search or count trusts its estimate of the share
    #: that completion tables would skip; with fewer, it takes the share to be zero.  At least 1.
    #: Defaults to 64, a convention: when every sample skips, the lower end of the Wilson interval
    #: at the default ``completion_bound_confidence_z`` is 0.94 (Section 4.4 of
    #: ``brouwer_zimmermann_code.md``).
    completion_bound_min_samples: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: Each time that a distance search or count considers completion tables for a layer of a search
    #: basis, it spends at most this much work, counted in steps, testing whether the tables would
    #: skip sampled supports of the layer.  A sample costs ``r + B * m * (m - 1) / 2`` steps.  Here
    #: the basis has ``r`` rows, the tables would have ``B`` blocks, and ``m = r - i``, where ``i``
    #: is the second-largest zero-based index of a basis row in the support.  The test tries each
    #: pair of the last ``m`` rows of the basis on each block.  Zero takes no samples, so the search
    #: predicts that tables skip nothing.  Defaults to ``5 * 10**7``, a budget of setup effort,
    #: which bounds the cost of sampling on large bases (Section 4.4 of
    #: ``brouwer_zimmermann_code.md``).
    completion_bound_sampling_work: int = 50_000_000

    #: A distance search or count estimates the share of supports that completion tables would skip
    #: by the lower end of a one-sided Wilson score interval of the sampled share.  This field is
    #: the z-score of that interval.  At most 10.  Defaults to 2.0, a convention: the true share
    #: exceeds the estimate with a one-sided confidence of about 98% (Section 4.4 of
    #: ``brouwer_zimmermann_code.md``).
    completion_bound_confidence_z: float = dataclasses.field(
        default=2.0, metadata={"maximum": 10.0}
    )

    #: The cost of one lookup of completion tables for each block of the tables, in scanned
    #: supports, the unit of their cost model: the cost of scanning one support without tables.
    #: Defaults to 2.0, a rough coefficient (Section 4.4 of ``brouwer_zimmermann_code.md``).
    completion_bound_lookup_block_cost: float = 2.0

    #: The additional cost of a lookup of completion tables, in scanned supports for each block and
    #: each MiB of tables, since larger tables miss the processor's caches more often.  Defaults to
    #: 0.04, a rough coefficient (Section 4.4 of ``brouwer_zimmermann_code.md``).
    completion_bound_lookup_block_cost_per_mib: float = 0.04

    #: A multiplier in the number of lookups of completion tables that the cost model predicts.  A
    #: larger value predicts more lookups, so distance searches and counts may use tables in fewer
    #: layers.  Defaults to 0.35, a rough coefficient (Section 4.4 of ``brouwer_zimmermann_code.md``
    #: gives the model).
    completion_bound_kept_first_row_ratio: float = 0.35

    #: The cost of building completion tables, in scanned supports for each byte of tables; with
    #: ``completion_bound_build_step_cost``, it predicts the cost of a build.  Defaults to 0.1, a
    #: rough coefficient, set by hand to half of ``completion_bound_build_step_cost`` (Section 4.4
    #: of ``brouwer_zimmermann_code.md``).
    completion_bound_table_byte_cost: float = 0.1

    #: The cost of building completion tables, in scanned supports for each step of the
    #: breadth-first searches that fill them, as ``_estimate_build_cost`` in
    #: ``_information_sets/completion_bounds.py`` counts steps.  Defaults to 0.2, a rough
    #: coefficient on the high side (Section 4.4 of ``brouwer_zimmermann_code.md``).
    completion_bound_build_step_cost: float = 0.2

    # Fields of batches and memory limits.

    #: The size of a chunk of work that a search does in one call of compiled code, which keyboard
    #: interrupts cannot stop: ``2**search_chunk_exponent`` units.  A BZ distance search scans at
    #: most that many supports and lookups of completion tables, counted together, in one call.  On
    #: several threads, each thread scans ``num_threads`` times as many (Section 4.5 of
    #: ``brouwer_zimmermann_code.md``).  A connected-cluster distance search visits at most a chunk
    #: of states in one call on each thread, except in the passes that choose its work items.  On
    #: threads, it splits a weight limit into work items of about a chunk of states each, within
    #: bounds on their number (Sections 3.6 and 3.11 of ``connected_cluster.md``).  The gates that
    #: decide when work moves to threads also count in chunks; their thresholds, chosen by hand,
    #: should pass only work that costs many times as much as starting threads.  On several threads,
    #: a layer of a set in a BZ search moves to threads if it has at least three rows and more than
    #: ``num_threads`` chunks of supports.  A connected-cluster weight limit moves under the gate of
    #: ``connected_cluster_split_shift``.  A batch query of ``CosetLeaderSearch`` runs on the
    #: calling thread alone until its queries have done ``num_threads`` chunks of work, and then
    #: moves the rest of the batch to threads; on one thread, the batch never moves.  With the BZ
    #: method, a query does one unit of work, and one more for each word of its coset that it weighs
    #: after the first.  With the connected-cluster method, it does one unit for each state that its
    #: searches visit.  At most 62.  Defaults to 15, a budget of interrupt latency on one thread,
    #: which grows with the number of threads for BZ scans.
    search_chunk_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    #: A connected-cluster distance search splits a weight limit across threads only if it predicts
    #: that the limit visits more than
    #: ``(num_threads << search_chunk_exponent) >> connected_cluster_split_shift`` states; each step
    #: of the shift halves that gate.  Unlike lowering ``search_chunk_exponent``, a larger shift
    #: lets smaller limits split without changing the size of chunks or the time before an interrupt
    #: takes effect (Section 3.11 of ``connected_cluster.md``).  At most 62.  Defaults to 2, chosen
    #: by hand.  At the default exponent and this shift, a limit that splits is predicted to cost at
    #: least about as much as the work of splitting it among threads.  A small search on eight
    #: threads ran about 1.6 times faster at a shift of 2 than at 0.
    connected_cluster_split_shift: int = dataclasses.field(default=2, metadata={"maximum": 62})

    #: The number of threads on which searches run costly work.  A BZ distance search scans a costly
    #: layer of a set on these threads, which share one copy of the set's completion tables.  A
    #: connected-cluster distance search searches a costly weight limit on them, and the batch
    #: queries of ``CosetLeaderSearch`` search the blocks of a batch.  The fault lister of the
    #: certifier of bounded fault propagation runs each costly listing on them too (section
    #: "Threads" of ``fault_propagation.md`` in ``qldpc/circuits/certification``).  At least 1, and
    #: 1 runs on the calling thread alone.  ``None`` takes the value of the environment variable
    #: ``QLDPC_NUM_THREADS`` when qLDPC was imported, if it was set, and otherwise
    #: ``numba.config.NUMBA_NUM_THREADS``.  The row on cores in "Start from a symptom" of
    #: :doc:`/exact_search_tuning` says how to set and cap it, and the section "Threads" of the
    #: README of ``qldpc/codes/distance`` says when ``QLDPC_NUM_THREADS`` also sets numba's own
    #: count.  The description of ``search_chunk_exponent`` says when work moves to threads (Section
    #: 4.5 of ``brouwer_zimmermann_code.md`` and Section 3.11 of ``connected_cluster.md``).
    #: Defaults to ``None``, chosen by hand: a search uses the cores that numba counts unless its
    #: caller caps them.
    num_threads: int | None = dataclasses.field(default=None, metadata={"minimum": 1})

    #: A count by the BZ or brute-force method scans in chunks.  Each chunk lies within one
    #: coefficient layer of one search basis, and visits at most this many supports and lookups of
    #: completion tables, counted together.  At least 1.  Defaults to 65,536, a budget of interrupt
    #: latency.
    scan_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The most chunks that the scan of a count visits in one call of its compiled loop, which
    #: keyboard interrupts cannot stop.  The scan also ends a call before and after each coefficient
    #: layer of a search basis that may use completion tables.  Until
    #: :func:`~qldpc.codes.distance.compile_search_setup` compiles the loop, the loop runs in Python
    #: and takes keyboard interrupts between chunks.  At least 1.  Defaults to 64, a budget of
    #: interrupt latency.
    scan_chunks_per_call: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most Gray-code steps that the walk of a check space by the MacWilliams method takes in
    #: one call of compiled code.  At least 1.  Defaults to 65,536, a budget of interrupt latency.
    macwilliams_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The least rank at which the MacWilliams method tests whether the X and Z checks of a counter
    #: span the same space, when their ranks are equal.  If they do, one walk over the ``2**rank``
    #: words of that space serves both.  Zero tests whenever the ranks are equal, and ``MAX_VALUE``
    #: never.  Counters known to be self-dual skip the test.  A count of full Pauli operators, which
    #: the method allows only on self-dual codes, runs the test at any rank to check that.  Defaults
    #: to 14, chosen by hand, near where the test and a second walk break even.
    macwilliams_shared_walk_rank: int = 14

    #: Applies to a ``CosetLeaderSearch`` of Pauli errors by CSS weight, with the Brouwer-Zimmermann
    #: method, on a stabilizer state that is not CSS.  Such a search finds errors of CSS weight one
    #: in a table, by syndrome, of the products of a single-qubit X error and a single-qubit Z
    #: error, each possibly the identity.  It builds the table if the distinct syndromes of the two
    #: kinds of error make at most this many pairs, and otherwise matches syndromes at each query
    #: (section "The coset-leader search" of ``problems.md``).  Zero disables the table.  Defaults
    #: to ``2**18``, a budget of memory: a table of tens of megabytes, reached at 511 qubits whose
    #: single-qubit errors have distinct syndromes.
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
    """The active tuning of the current context, which a search reads when it starts.

    A ``CosetLeaderSearch`` reads it when it is built.
    """
    return _ACTIVE_TUNING.get()


@contextlib.contextmanager
def use_tuning(**changes: Any) -> Iterator[SearchTuning]:
    """Change fields of the active tuning until the ``with`` block exits.

    The changes apply to the searches that start within the block, and to each ``CosetLeaderSearch``
    built within it.  They apply in the current context, and in copies of the context made within
    the block (see the module docstring for threads, processes, and asyncio tasks).  Fields that are
    not given keep their values in the active tuning, so nested blocks combine their changes, and
    the inner block takes precedence.

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

#: The most arrays of binomial coefficients that work estimates cache.  They keep one array for each
#: number ``r`` of rows of a basis, of ``8 * (r + 1)`` bytes.
BINOMIAL_CACHE_SIZE = 1024

#: The most pairs of classical searches, of the X and Z parts of errors, that ``get_css_searches``
#: in ``_brouwer_zimmermann/coset.py`` caches, one for each CSS state and tuning of their setup.
#: Each search can hold up to ``coset_completion_max_table_bytes`` of completion tables for each
#: set of each family.
CSS_SEARCH_CACHE_SIZE = 8

#: The most results that each of two caches of the MacWilliams method keeps, one for single
#: Krawtchouk rows and one for arrays of the Krawtchouk rows of the lightest weights.
KRAWTCHOUK_CACHE_SIZE = 1024

#: The most local Clifford matrices of deformed counters that their cache keeps, each a pair of
#: packed matrices, of X parts and of Z parts.
LOCAL_CLIFFORD_CACHE_SIZE = 1024

#: The most searches of Pauli errors that ``get_pauli_search`` in ``_brouwer_zimmermann/coset.py``
#: caches, one for each stabilizer matrix, measure of weight, and tuning of the setup.  A search by
#: Hamming weight can hold up to ``coset_completion_max_table_bytes`` of completion tables for each
#: set of each family.
PAULI_SEARCH_CACHE_SIZE = 8
