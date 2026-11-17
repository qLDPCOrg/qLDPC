# SPDX-License-Identifier: Apache-2.0

"""Run the work items of a search on several threads, which claim them in order.

A worker runs compiled code that releases Python's global interpreter lock (see the docstring of
``qldpc/_compilation.py``), so that the threads run at once.  Threads share the read-only inputs of
a search, such as its completion tables, rather than copy them, as processes would.
"""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable
from typing import Protocol, cast

import numba


class _NumbaConfig(Protocol):
    """The setting of numba that this module reads."""

    NUMBA_NUM_THREADS: int


# Numba creates the attributes of its settings dynamically, which type checkers cannot see.
_numba_config = cast(_NumbaConfig, numba.config)

####################################################################################################
# threads


def get_num_threads(num_threads: int | None) -> int:
    """The number of threads of a search, or numba's thread count if ``num_threads`` is None.

    Numba reads its thread count, ``numba.config.NUMBA_NUM_THREADS``, normally when it is imported,
    from the environment variable ``NUMBA_NUM_THREADS``, or else from the CPUs available to the
    process.
    """
    return _numba_config.NUMBA_NUM_THREADS if num_threads is None else num_threads


def run_work_items(
    num_items: int,
    num_threads: int,
    start_worker: Callable[[threading.Event], Callable[[int, int], None]],
) -> None:
    """Run the work items ``0, ..., num_items - 1`` on up to ``num_threads`` threads.

    The calling thread and ``min(num_threads, num_items) - 1`` other threads each call
    ``start_worker`` once with a shared stop event.  It returns the function that runs, on that
    thread, the items of a range ``start, ..., end - 1``, given as ``(start, end)``.  Each thread
    then claims the next range of items that no thread has claimed, in increasing order, and runs
    it, until every item is claimed or the event is set.  A range holds the ``max(1, remaining //
    (2 * num_threads))`` next items, for the ``remaining`` items that no thread has claimed, so
    ranges shrink as the work ends, and the number of claims grows only with the logarithm of the
    number of items.  A function that runs a range should check the event between chunks of its
    work, and may set it to stop every thread.

    If a thread raises an exception, the event is set, and once every thread has returned, the
    exception is raised again: that of the calling thread, as at a keyboard interrupt, or else the
    first that another thread raised.  An exception while the calling thread waits for the others,
    such as a keyboard interrupt, also sets the event, and is raised again once they have returned.
    """
    stop = threading.Event()
    lock = threading.Lock()
    next_item = 0
    errors: list[BaseException] = []
    num_threads = max(1, min(num_threads, num_items))

    def claim() -> tuple[int, int]:
        """Claim the next range of items, which is empty once every item is claimed."""
        nonlocal next_item
        with lock:
            start = next_item
            next_item = min(num_items, start + max(1, (num_items - start) // (2 * num_threads)))
            return start, next_item

    def work() -> None:
        """Claim and run ranges until no item is left or the event is set."""
        try:
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

    # The calling thread works too, so that it takes keyboard interrupts while the others work.  The
    # pool starts a thread for each submission, so it starts none for one thread.
    pool = concurrent.futures.ThreadPoolExecutor(max(1, num_threads - 1))
    try:
        for _ in range(num_threads - 1):
            pool.submit(work)
        work()
    except BaseException:
        stop.set()
        _wait_for_threads(pool, stop)
        raise
    _wait_for_threads(pool, stop)
    if errors:
        raise errors[0]


def _wait_for_threads(pool: concurrent.futures.ThreadPoolExecutor, stop: threading.Event) -> None:
    """Wait until the threads of a pool return.

    An exception while waiting, such as a keyboard interrupt, sets the stop event, and is raised
    again once the threads have returned.  The wait goes on through further exceptions, of which the
    last is raised.
    """
    try:
        pool.shutdown()
    except BaseException:
        stop.set()
        _wait_for_threads(pool, stop)
        raise
