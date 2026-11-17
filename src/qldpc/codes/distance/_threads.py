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

    Numba sets its thread count, ``numba.config.NUMBA_NUM_THREADS``, when it is imported, from the
    environment variable ``NUMBA_NUM_THREADS``, or else from the CPUs available to the process.
    """
    return _numba_config.NUMBA_NUM_THREADS if num_threads is None else num_threads


def run_work_items(
    num_items: int,
    num_threads: int,
    start_worker: Callable[[threading.Event], Callable[[int], None]],
) -> None:
    """Run the work items ``0, ..., num_items - 1`` on up to ``num_threads`` threads.

    The calling thread and ``min(num_threads, num_items) - 1`` other threads each call
    ``start_worker`` once with a shared stop event, which returns the function that runs an item on
    that thread.  Each thread then claims the next item that no thread has claimed, in increasing
    order, and runs it, until every item is claimed or the event is set.  A function that runs an
    item should check the event between chunks of its work, and may set it to stop every thread.

    If a thread raises an exception, as the calling thread does at a keyboard interrupt, the event
    is set.  Once every thread has returned, the first exception is raised again.
    """
    stop = threading.Event()
    lock = threading.Lock()
    next_item = 0
    errors: list[BaseException] = []

    def work() -> None:
        """Claim and run items until none is left or the event is set, and record any exception."""
        nonlocal next_item
        try:
            run_item = start_worker(stop)
            while not stop.is_set():
                with lock:
                    item, next_item = next_item, next_item + 1
                if item >= num_items:
                    return
                run_item(item)
        except BaseException as error:  # noqa: BLE001 - raised again once every thread returns
            errors.append(error)
            stop.set()

    # The calling thread works too, so that it takes keyboard interrupts while the others work.  The
    # pool starts a thread for each submission, so it starts none for one thread.
    num_threads = max(1, min(num_threads, num_items))
    pool = concurrent.futures.ThreadPoolExecutor(max(1, num_threads - 1))
    try:
        for _ in range(num_threads - 1):
            pool.submit(work)
        work()
    except BaseException as error:  # noqa: BLE001 - raised again once every thread returns
        errors.append(error)
        stop.set()
    finally:
        pool.shutdown()
    if errors:
        raise errors[0]
