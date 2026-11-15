# SPDX-License-Identifier: Apache-2.0

"""Settings and constants that tune the speed and memory of exact distance searches and counts.

:class:`SearchTuning` collects the values with which a search for an exact distance, a count of
logical operators, or a ``CosetLeaderSearch`` chooses its route and limits its batches and memory,
and the constants of this module size caches.  The page :doc:`/exact_search_tuning` renders this
docstring, after tables that lead from symptoms and routes to these values.  Section 4.4 of
``brouwer_zimmermann.md`` and Sections 3.5, 3.8, and 3.9 of ``connected_cluster.md`` explain the
heuristics that read them.

This module is private: its names, fields, defaults, and constants can change in any release without
warning, so code that uses it should pin the exact version of qLDPC.  It is meant for contributors,
for benchmarks, and for users of very costly searches who have found that another tuning helps them.

What tuning can and cannot change:

- Tuning changes only how long a search takes and how much memory it uses, never an exact distance,
  a distance limited by ``max_weight``, or a count of logical operators.
- With an early exit (``cutoff`` of two or more), which nontrivial word at or below ``cutoff`` a
  distance search finds depends on tuning, and its weight is still an upper bound between the
  distance and ``cutoff`` (see :func:`~qldpc.codes.distance.get_distance_from_generators`).
- A query of :class:`~qldpc.codes.distance.CosetLeaderSearch` decides the same way under every
  tuning whether a coset has a word of weight at most the query's maximum weight.  The word that the
  query returns can depend on tuning, and so can its lower bound with the Brouwer-Zimmermann (BZ)
  method, within the contract of :class:`~qldpc.codes.distance.MinWeightBound`.
- Codes and counters remember their results, and some caches last for the whole process, so a later
  search under another tuning may reuse results rather than search again.

Searches use the active tuning, which is the default tuning unless ``use_tuning`` changes some of
its fields within a ``with`` block::

    from sympy.abc import x, y

    from qldpc import codes
    from qldpc.codes.distance import _tuning

    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    with _tuning.use_tuning(max_information_sets=8, lp_minimum_work=0):
        distance = code.get_distance_exact()

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
Also record the version of qLDPC, or its commit and any changes to its source, which fix the
defaults and the constants.

A search can read a field when it starts, so fields can differ between two searches in one process.
The constants, ``MAX_VALUE`` and the sizes of caches, are fixed when qLDPC is imported: change them
by editing this module, and start a new process.

.. _tuning-retuning:

Retuning:

Each default is a round value that a simple heuristic gives, and the comment of each field names the
class of its default:

- A derived value gates an optional step, such as building a family of information sets, solving
  linear programs, or asking for automorphisms: the step runs once the work that it acts on, in
  candidate/set evaluations in BZ searches or states in connected-cluster searches, reaches the
  value.  A step saves at most that work, so the value is the cost of the step on a large basis or
  code, to an order of magnitude, as 1, 2, or 5 times a power of ten, and on a small search a gate
  can skip a step that would have paid (Section 4.4 of ``brouwer_zimmermann.md``).
- A limit at equality compares two estimates in one unit, and breaks even where they are equal.
- A rough estimate is a coefficient of the cost model of completion tables
  (``predict_completion_cost`` in ``_information_sets/completion_bounds.py``).  The model decides at
  break-even with these estimates, so its marginal decisions can go either way.
- A budget limits memory, interrupt latency, or setup effort, for safety rather than for speed.  A
  batch size, for example, trades the overhead of each call of compiled code against the delay of
  keyboard interrupts.
- A convention is a statistical or numerical setting, and its comment says what it guarantees.
- The widest allowed: wider blocks of completion tables never skip fewer supports where their tables
  fit, and a budget of memory bounds the tables.

``coset_route_minimum_work`` and ``macwilliams_shared_walk_rank`` are in no class: they were chosen
by hand, and their comments say why.  The sizes of caches were also chosen by hand.  To change the
size of a cache, keep the smallest size beyond which its hit rate, on workloads that reuse it, stops
growing.

Change a default only when its heuristic changes, or when a check shows that the assumption of its
heuristic, such as the cost of a step, is off by more than a factor of about 3.  Never change a
default to gain a factor of two on particular codes.  A new heuristic for a class applies to every
field of the class.  The scripts of ``experiments/distance_tuning`` make such checks and set no
value: ``constants.py`` checks the orders of magnitude that the heuristics assume, ``trace.py``
shows which decisions a search makes, and ``sweep.py`` catches gross regressions or bugs.  The
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

    Work counts candidate/set evaluations: an evaluation checks the word of one combination of the
    rows of one information set, and coefficient layer ``w`` evaluates the combinations of ``w``
    rows of each set (Section 3.6 of ``brouwer_zimmermann.md``).  Integer fields lie between zero
    and ``MAX_VALUE``, and float fields are finite and non-negative, unless a comment narrows them.
    Each comment names the class of its default (see :ref:`Retuning <tuning-retuning>`).
    """

    # Fields of exhaustive enumeration.

    #: Exhaustive enumeration, which finds distances by the method ``"brute_force"``, also within
    #: counts, handles at most ``2**enumeration_batch_exponent`` words at a time, in a few arrays of
    #: at most ``2**(enumeration_batch_exponent + 3)`` bytes each, and takes keyboard interrupts
    #: between batches.  At most 62.  Defaults to 15, a budget of interrupt latency and memory
    #: (Section 3.6 of ``brouwer_zimmermann.md``).
    enumeration_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    # Fields of information-set families.

    #: The most information sets in a family of overlapping information sets, or in a family that
    #: matroid partitioning finds; zero disables both, but not the maximum disjoint family of a
    #: coset search.  More sets can stop a search before costly layers, but setup grows roughly with
    #: the square of this limit, or with its cube when partitioning.  Defaults to 16, a budget of
    #: setup effort (Section 4.4 of ``brouwer_zimmermann.md``).
    max_information_sets: int = 16

    #: How many random column orders, in a fixed sequence, a search tries when the natural order
    #: packs fewer full disjoint information sets than ``min(N // r, 2**r - 1)``, for ``r`` rows and
    #: ``N`` nonzero columns; the search keeps the best packing, and zero keeps the natural order.
    #: Defaults to 8, a budget of setup effort, since each order costs a packing of the basis
    #: (Section 4.4 of ``brouwer_zimmermann.md``).
    disjoint_random_column_orders: int = 8

    #: The estimated work of a distance search or count, in candidate/set evaluations, from which it
    #: tries a family of overlapping information sets; zero always tries, and ``MAX_VALUE`` never.
    #: Defaults to ``2 * 10**6``, a derived value: the cost of building such a family on a large
    #: basis, a few million evaluations (Section 4.4 of ``brouwer_zimmermann.md``).
    overlapping_minimum_work: int = 2_000_000

    #: A query of a ``CosetLeaderSearch`` by the BZ method that predicts at least this many
    #: examinations with the disjoint family makes its search build a maximum disjoint and an
    #: overlapping family, once, and takes the family that does the least work.  In searches by
    #: Hamming weight, a query that predicts this many even with that family also weighs words by
    #: their tails and skips completions by tables.  The field also sets the weight at which the
    #: overlapping family is aimed; zero builds the families at the first query, and ``MAX_VALUE``
    #: never.  Defaults to ``2 * 10**4``, chosen by hand, preferring speed to memory, since the
    #: field sets three decisions and no break-even rule covers the aim (Section 4.4 of
    #: ``brouwer_zimmermann.md``).
    coset_route_minimum_work: int = 20_000

    #: How many column orders the construction of a family of overlapping information sets tries for
    #: each set that it may add, counting orders that give a set already in the family; zero builds
    #: no family.  Defaults to 2, a budget of setup effort (Section 4.4 of
    #: ``brouwer_zimmermann.md``).
    overlapping_attempts_per_set: int = 2

    #: A search tries matroid partitioning only if its estimated work, in candidate/set evaluations,
    #: reaches this multiple of the sum of ``b**2 * k**2 * n`` over the coverages ``b`` that it
    #: tries, for ``k`` rows and ``n`` nonzero columns; zero always tries, and ``MAX_VALUE`` never.
    #: Defaults to 5, a derived value: partitioning costs roughly that sum times a few evaluations
    #: (Section 4.4 of ``brouwer_zimmermann.md``).
    partition_work_ratio: int = 5

    #: The estimated work, in candidate/set evaluations, from which a search of overlapping
    #: information sets also bounds unvisited words by linear programming; zero in every such
    #: search, and ``MAX_VALUE`` in none.  Defaults to ``2 * 10**8``, a derived value: the cost of
    #: linear programming, planning included, in a search of a large basis, a few hundred million
    #: evaluations (Sections 3.1 and 4.4 of ``brouwer_zimmermann.md``).
    lp_minimum_work: int = 200_000_000

    #: The certificate from linear programming rounds each dual weight that its solver returns to
    #: the nearest rational number whose denominator is at most this field.  Every positive value is
    #: sound; a smaller one can round away part of the bound, and a larger one lengthens exact
    #: arithmetic.  Defaults to ``2**20``, a convention: rounding moves each dual weight by at most
    #: ``2**-21`` (Section 3.1 of ``brouwer_zimmermann.md``).
    lp_dual_max_denominator: int = dataclasses.field(default=1 << 20, metadata={"minimum": 1})

    #: The least work of a coefficient layer, in candidate/set evaluations, before which a distance
    #: search that would not partition considers switching to its family of automorphism images;
    #: zero considers the switch before the first layer, and ``MAX_VALUE`` never.  Defaults to
    #: ``5 * 10**6``, a derived value: the cost of finding automorphisms, verifying them, and
    #: building their family for a large code, several million evaluations (Sections 3.7 and 4.4 of
    #: ``brouwer_zimmermann.md``).
    automorphism_minimum_layer_work: int = 5_000_000

    # Fields of connected-cluster searches.

    #: A connected-cluster search asks its caller for automorphisms before the first weight limit at
    #: which it expects to have visited this many states, or supports, over all limits of one
    #: problem, or before a first limit that a cutoff raises; zero asks before the first limit, and
    #: ``MAX_VALUE`` never.  Defaults to ``5 * 10**4``, a derived value: the cost of asking for a
    #: large code, tens of thousands of states (Section 3.5 of ``connected_cluster.md``).
    automorphism_minimum_states: int = 50_000

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
    #: so the rounds never end if the field that limits them is ``MAX_VALUE`` too.  Defaults to 200,
    #: a budget of setup effort.
    sparsifying_rounds_without_improvement: int = 200

    # Fields of completion tables.

    #: A distance search or count considers completion tables for an information set in a layer, by
    #: sampling supports to predict their cost, only if the set has at least this many supports in
    #: the layer; zero considers every layer, and ``MAX_VALUE`` none.  Defaults to ``2 * 10**7``, a
    #: derived value: the cost of that prediction on a large basis, tens of millions of evaluations
    #: (Sections 3.9 and 4.4 of ``brouwer_zimmermann.md``).
    completion_bound_minimum_layer_work: int = 20_000_000

    #: A layer of a distance search or count uses the completion tables that it considers only if
    #: their predicted cost, building included unless the search has them, is at most this
    #: percentage of the cost of scanning without them; zero never uses tables, and ``MAX_VALUE``
    #: uses them wherever they are considered.  Defaults to 100, a limit at equality: an error of
    #: the cost model calls for changing its rough estimates, not this limit (Section 4.4 of
    #: ``brouwer_zimmermann.md``).
    completion_bound_max_cost_percent: int = 100

    #: The most bytes of completion tables that a distance search keeps for each problem, a count
    #: for each sector besides the tables that it is building, and a ``CosetLeaderSearch`` for each
    #: set of a family; a basis whose tables do not fit gets none, and zero disables tables.
    #: Defaults to 256 MiB, a budget of memory, which fits blocks of 16 columns for bases of up to
    #: about 500 rows and 140 tail columns.
    completion_bound_max_table_bytes: int = 256 * 2**20

    #: The number of tail columns in a block of the completion tables of distance searches and
    #: counts, a power of two up to 16.  The table of a block has ``2**width`` bytes for each row of
    #: the basis, so 1 builds fastest and skips least.  Defaults to 16, the widest allowed (Section
    #: 3.9 of ``brouwer_zimmermann.md``).
    completion_bound_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

    #: The number of tail columns in a block of the completion tables that the searches of
    #: ``CosetLeaderSearch`` by Hamming weight build for their costliest queries, a power of two up
    #: to 16.  Defaults to 16, the widest allowed (Section 4.4 of ``brouwer_zimmermann.md``).
    coset_completion_block_bits: int = dataclasses.field(default=16, metadata=_BLOCK_WIDTHS)

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
    #: convention: when every sample skips, the lower end of the confidence interval is 0.94.
    completion_bound_min_samples: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most work that a search spends bounding sampled supports, in units of one pair of rows on
    #: one block of completion tables, plus one unit for each row of the basis for each sample; zero
    #: takes no samples, so the search declines tables.  Defaults to ``5 * 10**7``, a budget of
    #: setup effort, which bounds the cost of sampling on large bases.
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
    #: Defaults to 2.0, a rough estimate (Section 3.9 of ``brouwer_zimmermann.md``).
    completion_bound_lookup_block_cost: float = 2.0

    #: The additional cost of a lookup of completion tables, in scanned supports for each block and
    #: each MiB of tables, since larger tables miss the processor's caches more often.  Defaults to
    #: 0.04, a rough estimate.
    completion_bound_lookup_block_cost_per_mib: float = 0.04

    #: A multiplier of the predicted number of lookups of completion tables: the share of heads of
    #: supports, their rows but the last, that a scan with tables keeps, divided by the share of
    #: supports that it keeps.  It is below one, since tables most often skip heads whose last row
    #: is late, which have few supports each.  Defaults to 0.35, a rough estimate.
    completion_bound_kept_first_row_ratio: float = 0.35

    #: The cost of building completion tables, in scanned supports for each byte of tables; with
    #: ``completion_bound_build_step_cost``, it predicts the cost of a build.  Defaults to 0.1, a
    #: rough estimate on the high side, half of ``completion_bound_build_step_cost`` by a ratio set
    #: by hand.
    completion_bound_table_byte_cost: float = 0.1

    #: The cost of building completion tables, in scanned supports for each step of the
    #: breadth-first searches that fill them, as ``_estimate_build_cost`` in
    #: ``_information_sets/completion_bounds.py`` counts steps.  Defaults to 0.2, a rough estimate
    #: on the high side, twice ``completion_bound_table_byte_cost``.
    completion_bound_build_step_cost: float = 0.2

    # Fields of batches and memory limits.

    #: A scan of information sets by the BZ method visits at most
    #: ``2**information_set_batch_exponent`` supports and lookups of completion tables, counted
    #: together, in one call of compiled code, which keyboard interrupts cannot stop.  At most 62.
    #: Defaults to 15, a budget of interrupt latency.
    information_set_batch_exponent: int = dataclasses.field(default=15, metadata={"maximum": 62})

    #: One compiled call of the scan of a count, by the BZ or brute-force method, visits at most
    #: this many supports and lookups of completion tables, counted together, and ends at the end of
    #: each coefficient layer of each information set.  At least 1.  Defaults to 65,536, a budget of
    #: interrupt latency.
    scan_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The most chunks of ``scan_chunk_size`` that the scan of a count visits before it returns to
    #: Python, once :func:`~qldpc.codes.distance.logical_counts.compile_search_setup` has compiled
    #: the loop of the scan over layers; until then, the scan returns after every chunk.  The scan
    #: also returns before and after each layer that may use completion tables.  At least 1.
    #: Defaults to 64, a budget of interrupt latency.
    scan_chunks_per_call: int = dataclasses.field(default=64, metadata={"minimum": 1})

    #: The most Gray-code steps that the walk of a check space by the MacWilliams method takes in
    #: one call of compiled code.  At least 1.  Defaults to 65,536, a budget of interrupt latency.
    macwilliams_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})

    #: The least rank at which the MacWilliams method tests whether the X and Z checks of a counter,
    #: of equal rank, span the same space, so that one walk over its ``2**rank`` words serves both;
    #: zero tests whenever the ranks are equal, and ``MAX_VALUE`` never.  Below this rank, a second
    #: walk costs less than the test.  Defaults to 14, chosen by hand: the test costs about as much
    #: as a walk of ``2**14`` to ``2**15`` steps, so a break-even rule cannot choose between 14 and
    #: 15, and at either rank a count costs about the same.
    macwilliams_shared_walk_rank: int = 14

    #: A ``CosetLeaderSearch`` of Pauli errors by CSS weight answers queries of weight one from a
    #: table, by syndrome, of the products of a single-qubit X error and a single-qubit Z error,
    #: each possibly the identity, if the distinct syndromes of the two kinds of error make at most
    #: this many pairs; zero disables the table.  Defaults to ``2**18``, a budget of memory: a table
    #: of tens of megabytes, which 511 qubits whose single-qubit errors have distinct syndromes
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

#: The most arrays of binomial coefficients, one for each number of rows of a basis, that the work
#: estimates of searches of information sets cache; the array for ``k`` rows takes ``8 * (k + 1)``
#: bytes.  Fixed when qLDPC is imported, and chosen by hand, since few workloads search bases of
#: more than a thousand different numbers of rows.
BINOMIAL_CACHE_SIZE = 1024

#: The most pairs of classical searches, of the X and Z parts of errors, that the cache of
#: ``get_css_searches`` in ``_brouwer_zimmermann/coset.py`` keeps, one for each CSS state and tuning
#: of their setup.  Fixed when qLDPC is imported, and chosen by hand.
CSS_SEARCH_CACHE_SIZE = 8

#: The most results that each of two caches of the MacWilliams method keeps: single Krawtchouk rows,
#: and arrays of the rows of the lightest weights.  Fixed when qLDPC is imported, and chosen by
#: hand.
KRAWTCHOUK_CACHE_SIZE = 1024

#: The most pairs of packed matrices, of X parts and of Z parts, that the cache of the local
#: Clifford matrices of deformed counters keeps.  Fixed when qLDPC is imported, and chosen by hand.
LOCAL_CLIFFORD_CACHE_SIZE = 1024

#: The most searches of Pauli errors that the cache of ``get_pauli_search`` in
#: ``_brouwer_zimmermann/coset.py`` keeps, one for each stabilizer matrix, measure of weight, and
#: tuning of the setup; zero disables the cache.  Fixed when qLDPC is imported, and chosen by hand.
PAULI_SEARCH_CACHE_SIZE = 8
