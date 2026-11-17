# SPDX-License-Identifier: Apache-2.0

"""Run the work items of a search on several threads, which claim them in order.

A worker runs compiled code that releases Python's global interpreter lock (see the docstring of
``qldpc/_compilation.py``), so that the threads run at once.  Threads share the read-only inputs of
a search, such as its completion tables, rather than copy them, as processes would.
"""

from __future__ import annotations

import contextlib
import threading
import time
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

# The message of the error that joining a thread raises before the thread counts as started.
_NOT_STARTED = "cannot join thread before it is started"

####################################################################################################
# threads


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
    of the lightest item.  ``cumulative_weights``, if
    given, holds for each item the total weight of the items up to it, which estimates their work;
    without it, items count one each.

    If a thread raises an exception, the event is set, and once every thread has returned, the
    exception is raised again: that of the calling thread, as at a keyboard interrupt, or else the
    first that another thread raised.  An exception while the calling thread starts or waits for
    the others, such as a keyboard interrupt, also sets the event, and is raised again once every
    thread whose start set its ``ident`` has returned; no thread claims a range once the event is
    set.
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
    # thread is listed before it starts, so that the cleanup joins it if its start got far enough to
    # set its ident, even if the start then raised.  A thread whose start is interrupted earlier may
    # start late, but it finds the event set and returns without starting its worker.
    threads: list[threading.Thread] = []
    try:
        for _ in range(num_threads - 1):
            threads.append(threading.Thread(target=work_on_new_thread, name="qldpc-search"))
            threads[-1].start()
        work()
    except BaseException:
        # A second interrupt that arrives before the event is set ends this function without
        # stopping the other threads; only a signal handler could prevent that, which a library
        # should not install.
        stop.set()
        _join(threads, stop)
        raise
    _join(threads, stop)
    if errors:
        raise errors[0]


def _join(threads: list[threading.Thread], stop: threading.Event) -> None:
    """Wait until every thread in a list whose start set its ``ident`` returns.

    An exception while waiting, such as a keyboard interrupt, sets the stop event, and the wait goes
    on.  The last such exception is raised once the threads have returned.
    """
    interrupt: BaseException | None = None
    for thread in threads:
        delay = 0.0
        while True:
            try:
                time.sleep(delay)
                if thread.ident is not None:
                    thread.join()
                break
            except BaseException as error:  # noqa: BLE001 - raised once the threads return
                # A thread has an ident a moment before it counts as started, and joining it
                # before then raises this error, so try again.
                if isinstance(error, RuntimeError) and _NOT_STARTED in str(error):
                    delay = 0.001
                    continue
                interrupt = error
                stop.set()
    if interrupt is not None:
        raise interrupt
