# SPDX-License-Identifier: Apache-2.0

"""Run the work items of a search, or the blocks of a batch of queries, on several threads.

A worker runs compiled code that releases Python's global interpreter lock (see the docstring of
``qldpc/_compilation.py``), so that the threads run at once.  Threads share the read-only inputs of
a search, such as its completion tables, rather than copy them, as processes would.  A caller of
``run_work_items``, ``run_items_until_first``, ``run_blocks``, or ``iterate_in_order`` must keep
the requirements on threads that the guide of its package states, as
``brouwer_zimmermann_code.md#sec-requirements`` does for the Brouwer-Zimmermann method.  Workers
read the tuning only from their search, never through ``get_tuning``, since new threads normally
start with the default tuning (see the module docstring of ``_tuning.py``).
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import threading
from collections.abc import Callable, Generator, Iterator
from typing import Generic, Protocol, TypeVar, cast

import numba
import numpy as np
import numpy.typing as npt

from qldpc import _num_threads


class _NumbaConfig(Protocol):
    """The setting of numba that this module reads."""

    NUMBA_NUM_THREADS: int


# Numba creates the attributes of its settings dynamically, which type checkers cannot see.
_numba_config = cast(_NumbaConfig, numba.config)

# The type of the outputs of the work items of ``iterate_in_order``.
_Output = TypeVar("_Output")


####################################################################################################
# threads

# The fewest work items for each thread that a search aims for when it splits work across threads: a
# layer of a BZ search (brouwer_zimmermann_code.md#sec-cross-file-behavior) or a weight limit of a
# connected-cluster search (connected_cluster.md#sec-threads).  More items balance the threads
# better at the end of the work; the number changes no result.
ITEMS_PER_THREAD = 32


def get_num_threads(num_threads: int | None) -> int:
    """The number of threads of a search, or the default number if ``num_threads`` is None.

    The default is the value of the environment variable ``QLDPC_NUM_THREADS`` when qLDPC was
    imported, if it was set (see ``qldpc/_num_threads.py``), and otherwise
    ``numba.config.NUMBA_NUM_THREADS`` (``README.md#sec-threads``).
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

    The calling thread and ``max(1, min(num_threads, num_items)) - 1`` new threads each call
    ``start_worker`` at most once with a shared stop event, and not at all if the event is already
    set when they begin.  ``start_worker`` returns the function that runs, on that thread, the items
    of a range ``start, ..., end - 1``, given as ``(start, end)``.  Each thread then claims the next
    range of items that no thread has claimed, in increasing order, and runs it, until every item is
    claimed or the event is set.  A function that runs a range should check the event between chunks
    of its work, and may set it to stop every thread.

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
    ``prange``, such as some functions of galois.  Two threads that call such code at once abort the
    process under numba's workqueue threading layer.

    If any thread raises an exception, the event is set.  This includes an exception, such as a
    keyboard interrupt, that the calling thread raises while it starts or waits for the others.
    Once every thread that started has returned, the calling thread's exception is raised again if
    it has one, and otherwise the first that another thread raised.  In narrow windows, which a
    comment in the code describes, a further interrupt can end this function while threads still
    run.  A thread whose start was interrupted may run after the wait has skipped it as not started,
    but it finds the event set and returns without calling ``start_worker``.
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
    Each thread calls ``start_worker`` at most once, as in ``run_work_items``, with a function
    ``needed``.  ``start_worker`` returns a function that runs an item, given its index, and returns
    whether the item ends the work, as the first item with a witness of a search does.  An item that
    ends the work lowers a limit to itself under a lock, as
    ``connected_cluster.md#sec-requirements`` requires of a result that threads share.  The threads
    skip every item after the limit, including the rest of a range that they have claimed.
    ``needed(item)`` says whether an item is still before the limit and the stop event of
    ``run_work_items`` is not set, so that a function that runs an item in chunks can call it
    between them, and stop.  The limit only falls, and every item before it runs, so the item
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
    ``run_rows``.  Reaching ``split_work`` signals that the rest of the batch would gain from
    threads.  Each caller defines the work in its own units, and counts only work that threads run
    in parallel, done without Python's global interpreter lock, such as words that compiled code
    weighs.  It counts none for rows too cheap to gain from threads, such as rows that spend much of
    their time holding the lock.  ``split_work`` should be enough such work to cost many times as
    much as starting the threads.  Elapsed time would not serve: a long batch of cheap rows that
    hold the lock takes long, yet runs more slowly on threads.

    If rows remain once the calling thread's work reaches ``split_work``, ``run_items_until_first``
    runs them on up to ``num_threads`` threads, in about ``BLOCKS_PER_THREAD`` contiguous blocks for
    each thread, which they claim in increasing order.  ``run_items_until_first`` returns the first
    block that ends the batch, and every row before that block runs, so the first row that ends the
    batch is the one that one thread would find.  Exceptions propagate as in ``run_work_items``, and
    the threads stop within their ranges, except in the narrow windows that its docstring describes.

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


####################################################################################################
# outputs in order


# The number of ranges for each of its threads into which ``iterate_in_order`` splits the weight of
# the items.  More ranges let the calling thread find the next range ready sooner, and balance the
# threads better where the weights estimate the work poorly, and fewer ranges cost fewer calls; the
# number changes no output.
RANGES_PER_THREAD = 256

# The threads of ``iterate_in_order`` hold at most this many outputs for each thread, and one more
# in total (see its docstring).  More let them run further ahead of the calling thread, at the cost
# of memory; the number changes no output.
OUTPUTS_PER_THREAD = 64

# The seconds after which a thread of ``iterate_in_order`` that waits to add an output checks again
# whether to stop.  A thread that cannot be started, or a second interrupt that comes before the
# threads are notified, sets an event without notifying the waiting threads.  So the timeout makes a
# waiting thread stop promptly after a failed start even if no other thread wakes it, and, after a
# second interrupt, it is the only way for a waiting thread to stop once no other thread runs.
_HALT_CHECK_SECONDS = 0.5


def iterate_in_order(
    num_items: int,
    num_threads: int,
    start_worker: Callable[[threading.Event], Callable[[int, int], Iterator[_Output]]],
    *,
    cumulative_weights: npt.NDArray[np.float64] | None = None,
) -> Generator[_Output, None, None]:
    """Yield on the calling thread, in item order, the outputs of work items that threads run.

    ``start_worker`` is called once on each thread that runs items, with a stop event, and returns a
    function that, given a range ``(start, end)`` of items, returns an iterator over the outputs of
    the items ``start, ..., end - 1``, in order.  On several threads, the stop event is set when the
    threads should stop, as when the iterator is closed, so that the iterator of a range can check
    it between chunks of its work and end early.  With fewer than two threads, the calling thread
    runs every item itself, as one range.  Otherwise a new thread calls ``run_work_items`` to start
    ``min(num_threads - 1, num_items)`` threads, including itself, while the calling thread yields
    the outputs of each range as they come, range by range in the order of the items.  So at most
    ``num_threads`` threads run, counting the calling thread, and the calling thread can work on
    each output while the threads run later items; work of its own that starts more threads, such
    as a batch of queries, adds to them.  The functions that the threads run follow the rules of
    ``run_work_items``.

    The threads claim ranges in increasing order, each with the fewest next items whose weight
    reaches ``1 / (RANGES_PER_THREAD * t)`` of the total, for ``t`` threads, so that the next range
    that the calling thread needs is soon ready.  ``run_work_items`` only starts the threads: it
    gives each of ``t`` threads one of ``t`` single items, whose function claims ranges until none
    is left.  ``cumulative_weights``, if given, holds for each item the total weight of the items up
    to it, which estimates their work; without it, items count one each.  A thread waits before it
    adds an output while ``OUTPUTS_PER_THREAD * t`` outputs are held, unless it runs the range that
    the calling thread reads and the calling thread has taken every output of that range, so that
    the calling thread can always go on.  So the threads hold at most ``OUTPUTS_PER_THREAD * t + 1``
    outputs.

    Closing the iterator, as ``contextlib.closing`` does when its block ends early or raises, stops
    the threads between outputs, or between chunks of work where the iterators of their ranges check
    the stop event, and waits until they have returned, as ``run_work_items`` does at an exception.
    A further interrupt while it waits can leave threads running, in the narrow windows that the
    comment in ``run_work_items`` describes; since the halt event and the stop event are set first,
    they stop before their next output or their iterator's next check of the stop event, or within
    ``_HALT_CHECK_SECONDS`` if they wait to add an output.

    An exception of a thread, or a thread that cannot be started, stops the others, and is raised on
    the calling thread once they have returned.  It can be raised before outputs that come before
    it in item order, so before the calling thread reaches what one thread would have reached
    first, such as an output that would have ended the iteration.
    """
    if num_threads < 2:
        yield from start_worker(threading.Event())(0, num_items)
        return
    num_workers = min(num_threads - 1, num_items)
    if not num_workers:
        return
    if cumulative_weights is None:
        cumulative_weights = np.arange(1, num_items + 1, dtype=np.float64)
    share = cumulative_weights[-1] / (RANGES_PER_THREAD * num_workers)
    ready = threading.Condition()
    halt = threading.Event()
    ranges: dict[int, _Range[_Output]] = {}
    errors: list[BaseException] = []
    max_held = OUTPUTS_PER_THREAD * num_workers
    num_held = 0
    next_item = 0
    position = 0

    def claim() -> tuple[int, int, _Range[_Output] | None]:
        """Claim and record the next range, or none once every item is claimed.

        The caller holds the lock of ``ready``.
        """
        nonlocal next_item
        start = next_item
        if start == num_items:
            return start, start, None
        claimed = cumulative_weights[start - 1] if start else 0.0
        end = int(np.searchsorted(cumulative_weights, claimed + share)) + 1
        next_item = min(num_items, max(start + 1, end))
        ranges[start] = _Range(next_item)
        return start, next_item, ranges[start]

    # The stop event of ``run_work_items``, which its threads share, once a thread has recorded it.
    shared_stop: threading.Event | None = None

    def stop_all() -> None:
        """Set the halt event and the threads' stop event, so that every thread stops."""
        with ready:
            halt.set()
            if shared_stop is not None:
                shared_stop.set()
            ready.notify_all()

    def start_ranges(stop: threading.Event) -> Callable[[int, int], None]:
        """Return how a thread claims ranges and adds their outputs to their records."""
        # Record the stop event, so that halting sets it.  A thread records it before it claims a
        # range, and claims none once the halt event is set, so no range runs with the event unset
        # after a halt.
        nonlocal shared_stop
        with ready:
            shared_stop = stop
        try:
            run_range = start_worker(stop)
        except BaseException:
            stop_all()
            raise

        def hold_outputs(_start: int, _end: int) -> None:
            """Claim ranges, and add their outputs to their records, until none is left.

            The thread stops once the halt event or the stop event of ``run_work_items`` is set.
            """
            nonlocal num_held
            while not (halt.is_set() or stop.is_set()):
                with ready:
                    start, end, record = claim()
                if record is None:
                    return
                try:
                    for output in run_range(start, end):
                        with ready:
                            # Wait while the cap is reached, unless this thread runs the range that
                            # the calling thread reads and the calling thread has taken its outputs,
                            # so that the calling thread can always go on.  Nothing notifies when an
                            # interrupt or a thread that cannot be started sets an event, so check
                            # the events again after a timeout too.
                            while (
                                num_held >= max_held
                                and (start != position or record.outputs)
                                and not (halt.is_set() or stop.is_set())
                            ):
                                ready.wait(_HALT_CHECK_SECONDS)
                            # Every thread checks the stop event, so setting it and notifying
                            # stops them all.
                            if halt.is_set() or stop.is_set():
                                stop.set()
                                ready.notify_all()
                                return
                            record.outputs.append(output)
                            num_held += 1
                            ready.notify_all()
                except BaseException:
                    # A range that raised is never done, so the calling thread yields none of the
                    # outputs after it, and waits for the exception.
                    stop_all()
                    raise
                # A range whose iterator ended after a stop may be incomplete, so it is never done.
                with ready:
                    if not (halt.is_set() or stop.is_set()):
                        record.done = True
                        ready.notify_all()

        return hold_outputs

    def run() -> None:
        """Run the threads, and record an exception for the calling thread."""
        try:
            run_work_items(num_workers, num_workers, start_ranges, single_items=True)
        except BaseException as error:  # noqa: BLE001 - raised again on the calling thread
            with ready:
                errors.append(error)
                ready.notify_all()

    # Yield the outputs of the range at the position, which starts at the first item and moves to
    # the end of each range once its thread is done and its outputs are taken.
    helper = threading.Thread(target=run, name="qldpc-search")
    try:
        helper.start()
        while position < num_items:
            with ready:
                while not errors and not (
                    position in ranges and (ranges[position].outputs or ranges[position].done)
                ):
                    ready.wait()
                if errors:
                    break
                record = ranges[position]
                if not record.outputs:
                    del ranges[position]
                    position = record.end
                    ready.notify_all()
                    continue
                output = record.outputs.popleft()
                num_held -= 1
                ready.notify_all()
            yield output
    finally:
        stop_all()
        _join([helper], halt)
    if errors:
        raise errors[0]


@dataclasses.dataclass
class _Range(Generic[_Output]):
    """The outputs of a range of items that ``iterate_in_order`` has not yet yielded.

    Attributes:
        end: The item after the last item of the range.
        outputs: The outputs that a thread has added and the calling thread has not yet taken.
        done: Whether the thread has added every output of the range.
    """

    end: int
    outputs: collections.deque[_Output] = dataclasses.field(default_factory=collections.deque)
    done: bool = False
