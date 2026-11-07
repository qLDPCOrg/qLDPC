# SPDX-License-Identifier: Apache-2.0

"""Thresholds that tune the speed of exact distance searches and logical-operator counts.

Exact distances and logical-operator counts choose between several routes: exhaustive enumeration,
Brouwer-Zimmermann (BZ) searches of disjoint or overlapping information sets, families found by
matroid partitioning, a certificate from linear programming, and families of the images of an
information set under automorphisms of a code.  Each choice weighs the setup cost of a route against
the work that it saves, by cost models with thresholds.  The connected-support search has one
threshold too, for when it asks for automorphisms.  :class:`SearchTuning` collects those thresholds,
and the batch sizes of the scans of counts, in one record.  Section 4.4 of
``brouwer_zimmermann.md`` describes the heuristics that use the BZ thresholds, and Section 3.5 of
``connected_support.md`` the threshold of the connected-support search.  The batch size of a
distance search is not a field: it is the ``block_size`` argument of
:func:`~qldpc.codes.distance.get_distance_classical` and
:func:`~qldpc.codes.distance.get_distance_quantum`, which the methods of codes and counts fix at
15.

This module is private.  Its names, fields, and defaults can change in any release, without a
deprecation period or any other warning.  Code that uses this module should therefore pin the exact
version of qLDPC that it was written for.  The module is meant for contributors, for benchmarks, and
for experienced users of very costly searches who have measured that a different tuning helps them.
The defaults suit typical searches and need no changes.

What tuning can and cannot change:

- Tuning changes only how long a search takes and how much memory it uses.  It never changes an
  exact distance, a distance limited by ``max_weight``, or a count of logical operators.
- With an early exit (``cutoff`` of two or more), a distance search returns the weight of the first
  nontrivial word that it finds at or below ``cutoff``.  Which word is found first depends on the
  order of the search, and so on tuning.  The result is still an upper bound between the distance
  and ``cutoff``, as documented by :func:`~qldpc.codes.distance.get_distance_quantum`.
- Codes remember the exact distances that they compute, and counters remember the distances of
  their codes, so a later search with a different tuning can return a remembered value without
  searching again.  Likewise, a counter keeps the weight distributions of its first count with the
  MacWilliams method, so ``macwilliams_shared_walk_rank`` affects only that first count.  To time a
  tuning, build a new code, or a new counter, for each search.

To change the tuning, change the fields of the active tuning within a ``with`` block::

    from sympy.abc import x, y

    from qldpc import codes
    from qldpc.codes.distance import _tuning

    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    with _tuning.use_tuning(max_information_sets=8, lp_minimum_work=0):
        distance = code.get_distance_exact()

Each search reads the active tuning once, when it starts, and uses that tuning throughout.  The
active tuning is a context variable (see the ``contextvars`` module of the standard library), which
``use_tuning`` changes in the current context until the block exits.  So:

- Code that runs in a copy of the context uses the tuning that was active when the copy was made,
  even after the block exits.  This includes asyncio tasks created within the block, functions run
  by ``asyncio.to_thread``, and functions run by ``contextvars.copy_context().run``.
- Worker processes that are forked, which is the default start method of ``multiprocessing`` on
  Linux before Python 3.14, inherit the tuning that was active when they were forked.
- New threads, and worker processes that are spawned or started by a fork server, start with the
  default tuning.  Builds of Python in which new threads inherit the context of the thread that
  starts them (see ``sys.flags.thread_inherit_context``) are an exception for threads.

The executors of ``concurrent.futures`` start their workers when tasks are first submitted, not
when the executors are created, and reuse those workers for later tasks.  A worker keeps the tuning
with which it started, even for tasks submitted outside of the block in which it started.  To give
workers a particular tuning, or the default one, enter ``use_tuning`` in the function that each
worker runs, as in ``use_tuning(**dataclasses.asdict(_tuning.DEFAULT_TUNING))``.

To reproduce an experiment, save ``dataclasses.asdict(_tuning.get_tuning())``, which holds every
field, and restore its tuning with ``use_tuning(**saved)``.  Changing the tuning never compiles code
again, before or after :func:`~qldpc.codes.distance.compile_search_setup`.  The first search in a
process also loads or compiles code, so time searches after a first search that warms them up.
"""

from __future__ import annotations

import contextlib
import contextvars
import dataclasses
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from ._binary import as_integer

# The largest value of an integer field.  Searches add and multiply some fields in compiled code,
# whose integers have 64 bits, and a larger value would also compile that code again for another
# integer type.
MAX_VALUE = 2**62

####################################################################################################
# tuning


@dataclass(frozen=True, kw_only=True)
class SearchTuning:
    """Thresholds of the cost models that choose search routes, and batch sizes of counting scans.

    Work estimates count candidate/set evaluations.  One such evaluation visits one combination of
    the rows of one information set, and checks the word that the combination gives.  Section 3.6
    of ``brouwer_zimmermann.md`` defines the estimate, which assumes pessimistically that a search
    finds no lighter word before it certifies its best one.  Estimates are not runtime guarantees.
    Coefficient layer ``w`` of a search visits the combinations of ``w`` rows of each of its
    information sets, so later layers cost more.

    The thresholds ``overlapping_minimum_work``, ``partition_work_ratio``, and
    ``automorphism_minimum_layer_work`` decide whether a search considers a family of information
    sets, which it takes only if that family predicts less work than the family that it would
    otherwise use.  The threshold ``lp_minimum_work`` decides whether a search of overlapping sets
    uses linear programming.  A threshold of zero makes every search to which a heuristic applies
    consider its family, or use linear programming, and the largest value, ``MAX_VALUE``, makes no
    search do so.  The threshold ``automorphism_minimum_states`` of the connected-support search
    counts visited states rather than candidate/set evaluations (Section 3.5 of
    ``connected_support.md``).  Every integer field is a Python integer between zero and
    ``MAX_VALUE``, and the batch sizes are positive.

    Counts with the BZ method always scan information sets, rather than enumerate every word.  The
    distance searches that find their default weights consult both ``allow_enumeration`` fields.

    Attributes:
        allow_enumeration_by_size: Whether a distance search may enumerate every nontrivial word,
            before it builds information sets, if a rule of the size of the problem predicts that
            enumeration costs less than building information sets.  This suits small problems,
            such as those with at most 20 basis rows.  If False, such problems go on to build
            information sets, which can make them much slower.  They may then still be enumerated,
            unless ``allow_enumeration_by_work`` is False too.  Defaults to True.
        allow_enumeration_by_work: Whether a distance search may enumerate every nontrivial word,
            after it has chosen its family of information sets, if there are at most as many such
            words as the estimated work of searching that family.  Defaults to True.
        max_information_sets: The most information sets in a family of overlapping sets, or in a
            family found by matroid partitioning.  More sets can skip costly coefficient layers of
            a hard search, but each set costs setup work, which dominates a cheap search.  Setup
            estimates the work of the best conceivable family of each size up to this limit, so its
            work grows roughly with the square of the limit.  A limit far above the default slows
            every search that builds such families, and with ``MAX_VALUE``, the setup of such a
            search would never finish.  Zero disables both kinds of family.  The
            ``max_information_sets`` argument of
            :meth:`~qldpc.codes.distance.CSSLogicalCounter.count_logical_ops` changes this field for
            one count, including the distance search that finds its default weight.  Defaults to
            16.
        overlapping_minimum_work: The estimated work, in candidate/set evaluations, below which a
            search does not try to replace its disjoint information sets by a family of overlapping
            ones that it builds heuristically.  Building such a family costs time that a cheap
            search cannot recover.  Families found by matroid partitioning are tried or not
            according to ``partition_work_ratio`` instead.  Defaults to 100,000.
        partition_work_ratio: A search builds families by matroid partitioning only if its
            estimated work is at least this multiple of the estimated cost of partitioning, so that
            an attempt that does not pay off adds at most about ``1 / partition_work_ratio`` to the
            time of the search.  The cost of partitioning is estimated from measurements on bases of
            85 to 156 rows and 127 to 432 columns.  Defaults to 10.
        lp_minimum_work: The estimated work, in candidate/set evaluations, from which a search of
            overlapping information sets also bounds the weights of unvisited words by linear
            programming, which can end it sooner (Section 3.1 of ``brouwer_zimmermann.md``).  Each
            linear program takes milliseconds, which only costly searches recover.  Searches of
            disjoint information sets never use linear programming, since it gives them no stronger
            bound.  Defaults to 1,000,000,000.
        automorphism_minimum_layer_work: A distance search that can find automorphisms of its code
            finds them before the first coefficient layer whose estimated work, in candidate/set
            evaluations, is at least this value.  The search then switches to the family of the
            images of one information set under those automorphisms, if that predicts less work
            (Section 3.7 of ``brouwer_zimmermann.md``).  Finding automorphisms can take
            milliseconds, as long as many whole searches take.  Only the ``get_distance_exact``
            methods of binary quantum codes find automorphisms, with the optional ``pybliss``
            package or from the construction of the code.
            :func:`~qldpc.codes.distance.get_distance_classical`,
            :func:`~qldpc.codes.distance.get_distance_quantum`, and counts of logical operators do
            not.  Defaults to 100,000,000.
        automorphism_minimum_states: The number of visited states, counted over all weight
            limits of the search of one sector, from which a connected-support search
            (``method="connected_support"``) uses automorphisms of its code.  It asks for them
            before the first weight limit at which it expects to reach this number, from the growth
            in states between its last two limits, and then searches from one variable of each
            orbit of the automorphisms, rather than from every variable (Section 3.5 of
            ``connected_support.md``).  Finding automorphisms
            takes milliseconds for codes of hundreds of qubits, and about half a second with
            ``pybliss`` for a code of thousands of qubits.  Zero asks for them before the first
            weight limit, and ``MAX_VALUE`` in effect never does.  Defaults to 100,000.
        scan_chunk_size: The most supports, which are choices of rows of an information set, that
            one compiled call of the scan of a count with the BZ or brute-force method visits.  A
            scan proceeds in steps, each of which is one coefficient layer of one information set,
            and each call ends at the end of a step.  So ``MAX_VALUE`` lets each call finish its
            step, but not the whole scan.  Compiled code cannot be interrupted, so this bounds the
            time before a keyboard interrupt takes effect.  Defaults to 65,536.
        scan_chunks_per_call: The most calls of ``scan_chunk_size`` supports that the scan of a
            count makes before it returns to Python, once
            :func:`~qldpc.codes.distance.compile_search_setup` has compiled its loop over steps.
            Each call uses one of them, even if it ends early at the end of a step.  Until then,
            that loop runs as Python, so a keyboard interrupt takes effect between any two calls.
            Once that loop is compiled, ``MAX_VALUE`` lets a scan run to its end before it returns.
            Defaults to 64.
        macwilliams_shared_walk_rank: The smallest rank of the checks at which the MacWilliams
            method tests whether the X and Z checks of a counter span the same space.  If they do,
            one walk over the ``2**rank`` words of that space serves both.  Below this rank, a
            second walk costs less than the test.  Counters already known to be self-dual skip the
            test, as do counts of full Pauli operators, which only self-dual codes allow.  Zero
            tests whenever the ranks are equal, and ``MAX_VALUE`` never tests.
            Defaults to 14.
    """

    allow_enumeration_by_size: bool = True
    allow_enumeration_by_work: bool = True
    max_information_sets: int = 16
    overlapping_minimum_work: int = 100_000
    partition_work_ratio: int = 10
    lp_minimum_work: int = 1_000_000_000
    automorphism_minimum_layer_work: int = 100_000_000
    automorphism_minimum_states: int = 100_000
    scan_chunk_size: int = dataclasses.field(default=1 << 16, metadata={"minimum": 1})
    scan_chunks_per_call: int = dataclasses.field(default=64, metadata={"minimum": 1})
    macwilliams_shared_walk_rank: int = 14

    def __post_init__(self) -> None:
        """Validate the fields, and store integers as Python integers.

        Compiled code specializes on the types of its arguments, so storing every integer as a
        Python integer keeps a tuning from compiling code again.

        Raises:
            TypeError: If a field whose default is a bool is not a bool, or another field is not an
                integer, or is a bool.
            ValueError: If an integer field is negative or above ``MAX_VALUE``, or a batch size is
                not positive.
        """
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(field.default, bool):
                if not isinstance(value, bool):
                    raise TypeError(f"{field.name} must be a bool, not {value!r}")
                continue
            value = as_integer(value, field.name)
            minimum = field.metadata.get("minimum", 0)
            if not minimum <= value <= MAX_VALUE:
                raise ValueError(
                    f"{field.name} must be between {minimum} and {MAX_VALUE}, not {value}"
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
