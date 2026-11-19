# SPDX-License-Identifier: Apache-2.0

"""Run the work items of a search, or the blocks of a batch of queries, on several threads.

A worker runs compiled code that releases Python's global interpreter lock (see the docstring of
``qldpc/_compilation.py``), so that the threads run at once.  Threads share the read-only inputs of
a search, such as its completion tables, rather than copy them, as processes would.  A caller of
``run_work_items``, ``run_items_until_first``, or ``run_blocks`` must keep the requirements on
threads in Section 5 of the guide of its method, ``brouwer_zimmermann_code.md`` or
``connected_cluster.md``.  Its workers read the tuning only from the search, never through
``get_tuning``, since new threads start with the default tuning (see the module docstring of
``_tuning.py``).
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable
from typing import Protocol, cast

import numba
import numpy as np
import numpy.typing as npt

from qldpc import _num_threads


class _NumbaConfig(Protocol):
    """The setting of numba that this module reads."""

    NUMBA_NUM_THREADS: int


# Numba creates the attributes of its settings dynamically, which type checkers cannot see.
_numba_config = cast(_NumbaConfig, numba.config)


####################################################################################################
# threads

# The fewest work items for each thread that a search aims for when it splits work across threads: a
# layer of a BZ search (Section 4.5 of brouwer_zimmermann_code.md) or a weight limit of a
# connected-cluster search (Section 3.11 of connected_cluster.md).  More items balance the threads
# better at the end of the work; the number changes no result.
ITEMS_PER_THREAD = 32


def get_num_threads(num_threads: int | None) -> int:
    """The number of threads of a search, or the default number if ``num_threads`` is None.

    The default is the value of the environment variable ``QLDPC_NUM_THREADS`` when qLDPC was
    imported, if it was set (see ``qldpc/_num_threads.py``), and otherwise
    ``numba.config.NUMBA_NUM_THREADS``.  Numba reads that, normally when it is imported, from the
    environment variable ``NUMBA_NUM_THREADS``, or else from the CPUs available to the process.
    ``numba.set_num_threads`` does not change either.
    """
    if num_threads is not None:
        return num_threads
    if _num_threads.NUM_THREADS is not None:
        return _num_threads.NUM_THREADS
    return _numba_config.NUMBA_NUM_THREADS


def run_work_items(
    num_items: int,
    num_threads: int,
    start_worker: Callable[[threading.Event], Callable[[int, int], None]],
    *,
    cumulative_weights: npt.NDArray[np.float64] | None = None,
    single_items: bool = False,
) -> None:
    """Run the work items ``0, ..., num_items - 1`` on up to ``num_threads`` threads.

    The calling thread and ``min(num_threads, num_items) - 1`` new threads each call
    ``start_worker`` once with a shared stop event.  It returns the function that runs, on that
    thread, the items of a range ``start, ..., end - 1``, given as ``(start, end)``.  Each thread
    then claims the next range of items that no thread has claimed, in increasing order, and runs
    it, until every item is claimed or the event is set.  A function that runs a range should check
    the event between chunks of its work, and may set it to stop every thread.

    Ranges follow guided self-scheduling: a range holds the fewest next items whose weight reaches
    about ``1 / (2 * num_threads)`` of the weight that no thread has claimed, so a heavy item can
    make a range overshoot that share, and ranges shrink as the work ends.  The claims number at
    most about ``2 * num_threads * ln(W / w) + 1``, for the total weight ``W`` and the weight ``w``
    of the lightest item.  ``cumulative_weights``, if given, holds for each item the total weight of
    the items up to it, which estimates their work; without it, items count one each.  With
    ``single_items``, each range instead holds one item, and ``cumulative_weights`` is ignored.
    That suits work that may end early, as at the first witness of a search in DFS order: claims of
    one item share the items before the end among the threads, where a share of the remaining
    weight would leave them to one thread.

    The functions that run ranges must not call numba code compiled with ``parallel=True`` or
    ``prange``, such as some of galois's: two threads that call it at once abort the process under
    numba's default threading layer.

    If a thread raises an exception, the event is set, and once every thread has returned, the
    exception is raised again: that of the calling thread, as at a keyboard interrupt, or else the
    first that another thread raised.  An exception while the calling thread starts or waits for
    the others, such as a keyboard interrupt, also sets the event, and is raised again once every
    thread that started has returned, except in narrow windows in which a further interrupt can end
    this function while threads still run (see the comment in the code).  A thread whose start was
    interrupted before it counted as started may still run, but it finds the event set and returns
    without starting its worker.  No thread claims a range once the event is set.
    """
    stop = threading.Event()
    lock = threading.Lock()
    next_item = 0
    errors: list[BaseException] = []
    num_threads = max(1, min(num_threads, num_items))
    if cumulative_weights is None:
        cumulative_weights = np.arange(1, num_items + 1, dtype=np.float64)

    def claim() -> tuple[int, int]:
        """Claim the next range of items, which is empty once every item is claimed or stopped."""
        nonlocal next_item
        with lock:
            start = next_item
            if stop.is_set() or start == num_items:
                return start, start
            if single_items:
                next_item = start + 1
                return start, next_item
            claimed = cumulative_weights[start - 1] if start else 0.0
            share = (cumulative_weights[-1] - claimed) / (2 * num_threads)
            end = int(np.searchsorted(cumulative_weights, claimed + share)) + 1
            next_item = min(num_items, max(start + 1, end))
            return start, next_item

    def work() -> None:
        """Claim and run ranges until no item is left or the event is set."""
        try:
            if stop.is_set():
                return
            run_range = start_worker(stop)
            while not stop.is_set():
                start, end = claim()
                if start == end:
                    return
                run_range(start, end)
        except BaseException as error:
            errors.append(error)
            stop.set()
            raise

    def work_on_new_thread() -> None:
        """Work, leaving any exception, which ``work`` records, to the calling thread."""
        with contextlib.suppress(BaseException):
            work()

    # The calling thread works too, so that it takes keyboard interrupts while the others work.  A
    # thread is listed before it starts, so that the cleanup joins it if it started, even if its
    # start then raised.
    threads: list[threading.Thread] = []
    try:
        for _ in range(num_threads - 1):
            threads.append(threading.Thread(target=work_on_new_thread, name="qldpc-search"))
            threads[-1].start()
        work()
        _join(threads, stop)
    except BaseException:
        # A further interrupt can end this function before every thread that started has returned,
        # in narrow windows that only a signal handler could close, which a library should not
        # install.  One that arrives here while the event is not yet set leaves the threads running
        # until every item is claimed and run.  One that arrives as the wait below starts, or while
        # it handles an earlier exception, leaves them running only until their chunks end, since
        # the event is set by then.  An interrupt while the first wait, in the try, handles an
        # exception is safe, since this handler waits again.
        stop.set()
        _join(threads, stop)
        raise
    if errors:
        raise errors[0]


def run_items_until_first(
    num_items: int,
    num_threads: int,
    start_worker: Callable[[Callable[[int], bool]], Callable[[int], bool]],
    *,
    single_items: bool = False,
) -> int:
    """Run work items on threads until the first item that ends the work, and return that item.

    ``run_work_items`` runs the items ``0, ..., num_items - 1``, with ``single_items`` as there.
    Each thread calls ``start_worker`` once with a function ``needed``.  ``start_worker`` returns a
    function that runs an item, given its index, and returns whether the item ends the work, as the
    first item with a witness of a search does.  An item that ends the work lowers a limit to
    itself, and the threads skip every item after the limit, including the rest of a range that they
    have claimed.  ``needed(item)`` says whether an item is still before the limit and the stop
    event of ``run_work_items`` is not set, so that a function that runs an item in chunks can call
    it between them, and stop.  The limit only falls, and every item before it runs, so the item
    returned is the first that ends the work, as on one thread, or ``num_items`` if no item does.
    Items after the returned one may have run, in part or in full, so a caller reads only results up
    to it.  Exceptions propagate as in ``run_work_items``.
    """
    lock = threading.Lock()
    first = num_items

    def start_items(stop: threading.Event) -> Callable[[int, int], None]:
        """Return how a worker runs a range of items."""

        def needed(item: int) -> bool:
            """Whether an item comes before every item that has ended the work, and no stop."""
            return not stop.is_set() and item < first

        run_item = start_worker(needed)

        def run_range(start: int, end: int) -> None:
            """Run the items of a range in order, up to the limit or until the event is set."""
            nonlocal first
            for item in range(start, end):
                if not needed(item):
                    return
                if run_item(item):
                    with lock:
                        first = min(first, item)
                    return

        return run_range

    run_work_items(num_items, num_threads, start_items, single_items=single_items)
    return first


def _join(threads: list[threading.Thread], stop: threading.Event) -> None:
    """Wait until every thread in a list that has started returns.

    A thread that has not started is skipped: the stop event is set whenever its start was
    interrupted, so it returns without starting its worker if it runs later.  An exception while
    waiting, such as a keyboard interrupt, sets the stop event, and the wait goes on.  The last such
    exception is raised once the threads have returned, unless a further exception arrives while
    this handles one; the comment in ``run_work_items`` says when that leaves threads running.
    """
    interrupt: BaseException | None = None
    index = 0
    while True:
        try:
            while index < len(threads):
                if threads[index].is_alive():
                    threads[index].join()
                index += 1
            break
        except BaseException as error:  # noqa: BLE001 - raised once the threads return
            interrupt = error
            stop.set()
    if interrupt is not None:
        raise interrupt


####################################################################################################
# batches


# The number of blocks for each thread into which ``run_blocks`` splits the rows that it moves to
# threads.  More blocks balance the threads better at the end of a batch, and fewer blocks cost
# fewer calls; the number changes no result.
BLOCKS_PER_THREAD = 16

# The work limit that lets ``run_rows`` of ``run_blocks`` run every row it is given.
NO_WORK_LIMIT = (1 << 63) - 1

# A function that runs rows ``start, ..., end - 1`` of a batch in order, given ``(start, end,
# max_work)``, and stops after the first row that ends the batch, or after the row at which its
# work in this call reaches ``max_work``.  The caller defines the units of work (see
# ``run_blocks``).  It returns the index after the last row that it ran, and whether that row ends
# the batch.
RunRows = Callable[[int, int, int], tuple[int, bool]]


def run_blocks(
    num_rows: int, num_threads: int, split_work: int, start_worker: Callable[[], RunRows]
) -> int:
    """Run the rows ``0, ..., num_rows - 1`` of a batch, moving costly batches to threads.

    Each worker calls ``start_worker`` once, so that it can allocate its own buffers, and runs rows
    with the ``RunRows`` function that it returns.  The calling thread calls it once to start the
    batch, and once more, as one of the workers, if the rest of the batch moves to threads, unless
    another thread has already failed.  A row can end the batch, as the first row that fails a
    test does; then no row after it need run.

    The calling thread first runs the batch alone, in order, until its work reaches ``split_work``.
    So a batch of cheap rows, or any batch on one thread, runs on the calling thread in one call of
    ``run_rows``.  The work is the signal that the rest of the batch gains from threads, so each
    caller defines it, in its own units, as work that threads run in parallel: work done without
    Python's global interpreter lock, such as words that compiled code weighs, and none for rows
    too cheap to gain, such as rows that spend much of their time holding the lock.  ``split_work``
    should be enough such work to cost many times as much as starting the threads.  Elapsed time
    would not serve: a long batch of cheap rows that hold the lock takes long, yet runs more slowly
    on threads.  If rows remain after that, ``run_items_until_first`` runs them on ``num_threads``
    threads, in about ``BLOCKS_PER_THREAD`` contiguous blocks for each thread, which they claim in
    increasing order.  ``run_items_until_first`` returns the first block that ends the batch, and
    every row before that block runs, so the first row that ends the batch is the one that one
    thread would find.  Exceptions propagate as in ``run_work_items``, and the threads stop within
    their ranges, except in the narrow windows that its docstring describes.

    Returns:
        The first row that ends the batch, or ``num_rows`` if no row does.
    """
    if num_threads < 2:
        end, ends = start_worker()(0, num_rows, NO_WORK_LIMIT)
        return end - 1 if ends else num_rows
    first, ends = start_worker()(0, num_rows, split_work)
    if ends:
        return first - 1
    if first == num_rows:
        return num_rows
    block_size = -(-(num_rows - first) // (BLOCKS_PER_THREAD * num_threads))
    num_blocks = -(-(num_rows - first) // block_size)
    # The row that ends the batch in each block that ends it, which only its thread writes.
    ending_rows: dict[int, int] = {}

    def start_blocks(_needed: Callable[[int], bool]) -> Callable[[int], bool]:
        """Return how a worker runs a block, which runs in one call and so never asks ``needed``."""
        run_rows = start_worker()

        def run_block(block: int) -> bool:
            """Run the rows of a block, and return whether one of them ends the batch."""
            row = first + block * block_size
            end, ends = run_rows(row, min(num_rows, row + block_size), NO_WORK_LIMIT)
            if ends:
                ending_rows[block] = end - 1
            return ends

        return run_block

    block = run_items_until_first(num_blocks, num_threads, start_blocks)
    return ending_rows[block] if block < num_blocks else num_rows
