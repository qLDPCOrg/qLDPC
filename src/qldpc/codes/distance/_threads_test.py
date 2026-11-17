# SPDX-License-Identifier: Apache-2.0

"""Unit tests for running the work items of a search on threads."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numba
import numpy as np
import pytest

from . import _threads

# How long a test waits for an event that a thread should set at once.
_TIMEOUT = 10


@pytest.fixture
def no_threads_left() -> Iterator[None]:
    """Check that a test leaves no thread running that it started."""
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before


def test_the_number_of_threads_defaults_to_that_of_numba(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 3)
    assert _threads.get_num_threads(None) == 3
    assert _threads.get_num_threads(2) == 2


@pytest.mark.usefixtures("no_threads_left")
@pytest.mark.parametrize("num_threads", [1, 2, 3, 8])
def test_every_item_runs_once(num_threads: int) -> None:
    """Each item runs once, in ranges that each worker claims in increasing order.

    Each of ``num_threads`` workers starts once, on the calling thread or on a new thread.  The
    number of ranges grows only with the logarithm of the number of items.
    """
    num_items = 500
    claims: list[list[tuple[int, int]]] = []
    workers: list[int] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        workers.append(threading.get_ident())
        claims.append([])
        ranges = claims[-1]
        return lambda start, end: ranges.append((start, end))

    _threads.run_work_items(num_items, num_threads, start_worker)
    assert len(workers) == num_threads and threading.get_ident() in workers
    assert all(ranges == sorted(ranges) for ranges in claims)
    ranges = sorted(claimed for worker_ranges in claims for claimed in worker_ranges)
    assert [item for start, end in ranges for item in range(start, end)] == list(range(num_items))
    assert len(ranges) <= 2 * num_threads * (1 + math.log(num_items))


@pytest.mark.parametrize("num_threads", [2, 4])
def test_ranges_hold_shares_of_the_weight_of_the_items(num_threads: int) -> None:
    """A range of several items holds at most half of the weight not yet claimed.

    Ranges never get heavier, and they cover the items.  The items weigh as the supports of the
    items of layer seven of a set of 150 rows, which fix one row each, so the first items weigh the
    most.
    """
    weights = np.array([math.comb(149 - row, 6) for row in range(144)], dtype=np.float64)
    ranges: list[tuple[int, int]] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        return lambda start, end: ranges.append((start, end))

    _threads.run_work_items(
        len(weights), num_threads, start_worker, cumulative_weights=np.cumsum(weights)
    )
    ranges.sort()
    assert [item for start, end in ranges for item in range(start, end)] == list(range(144))
    range_weights = [weights[start:end].sum() for start, end in ranges]
    assert range_weights == sorted(range_weights, reverse=True)
    for (start, end), weight in zip(ranges, range_weights, strict=True):
        assert end - start == 1 or weight <= weights[start:].sum() / 2


def test_no_items_run_on_the_calling_thread_alone() -> None:
    workers: list[int] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        workers.append(threading.get_ident())
        return print

    _threads.run_work_items(0, 4, start_worker)
    assert workers == [threading.get_ident()]


def test_a_range_can_stop_every_thread() -> None:
    """Once a range sets the stop event, no thread claims another range."""
    claimed: list[int] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        def run_range(start: int, end: int) -> None:
            claimed.extend(range(start, end))
            stop.set()

        return run_range

    _threads.run_work_items(100, 3, start_worker)
    assert 0 < len(claimed) < 100


@pytest.mark.usefixtures("no_threads_left")
@pytest.mark.parametrize("raises_on_calling_thread", [False, True])
def test_an_exception_stops_every_thread_and_is_raised_again(
    raises_on_calling_thread: bool,
) -> None:
    """An exception in any thread, such as an interrupt in the calling one, stops all threads.

    Every thread has returned when the exception is raised again.
    """
    calling_thread = threading.get_ident()
    stopped: list[bool] = []
    started = threading.Barrier(2, timeout=_TIMEOUT)

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        on_calling_thread = threading.get_ident() == calling_thread

        def run_range(start: int, end: int) -> None:
            started.wait()
            if on_calling_thread == raises_on_calling_thread:
                raise KeyboardInterrupt if on_calling_thread else ValueError("failed")
            # Wait for the stop event, as a thread that checks it between chunks would.
            stopped.append(stop.wait(timeout=_TIMEOUT))

        return run_range

    error = KeyboardInterrupt if raises_on_calling_thread else ValueError
    with pytest.raises(error):
        _threads.run_work_items(10, 2, start_worker)
    assert stopped == [True]


@pytest.mark.usefixtures("no_threads_left")
@pytest.mark.parametrize("joinable_late", [False, True])
def test_an_interrupt_while_a_thread_starts_stops_it(
    monkeypatch: pytest.MonkeyPatch, joinable_late: bool
) -> None:
    """A keyboard interrupt as a new thread starts stops that thread, which claims nothing more.

    The start of the thread raises the interrupt once the thread has claimed its first range, in
    place of a real one, which no test can time.  That range waits for the stop event, so the
    thread claims no other range, and the calling thread claims none.  A thread can have an ident
    a moment before it can be joined, as on Python 3.11 to 3.13, and the cleanup then tries again.
    """
    start = threading.Thread.start
    claimed = threading.Event()
    claims: list[tuple[int, int]] = []
    stopped: list[bool] = []

    def interrupted_start(thread: threading.Thread) -> None:
        start(thread)
        if thread.name == "qldpc-search":
            assert claimed.wait(timeout=_TIMEOUT)
            raise KeyboardInterrupt

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        def run_range(start: int, end: int) -> None:
            claims.append((start, end))
            claimed.set()
            stopped.append(stop.wait(timeout=_TIMEOUT))

        return run_range

    join = threading.Thread.join
    joins: list[threading.Thread] = []

    def late_join(thread: threading.Thread, *args: Any, **kwargs: Any) -> None:
        joins.append(thread)
        if len(joins) == 1:
            raise RuntimeError("cannot join thread before it is started")
        join(thread, *args, **kwargs)

    monkeypatch.setattr(threading.Thread, "start", interrupted_start)
    if joinable_late:
        monkeypatch.setattr(threading.Thread, "join", late_join)
    with pytest.raises(KeyboardInterrupt):
        _threads.run_work_items(10, 3, start_worker)
    assert len(claims) == 1 and stopped == [True]
    assert len(joins) == (2 if joinable_late else 0)


@pytest.mark.usefixtures("no_threads_left")
@pytest.mark.parametrize("num_interrupts", [1, 2])
def test_an_interrupt_while_the_calling_thread_waits_stops_the_other_threads(
    monkeypatch: pytest.MonkeyPatch, num_interrupts: int
) -> None:
    """A keyboard interrupt while the calling thread waits for a working thread stops that thread.

    The other thread stops after its current chunk, and the interrupt is raised once it has
    returned, also if another interrupt comes while the calling thread waits again.  The wait for
    the thread raises the interrupts as it starts, in place of real ones.
    """
    join = threading.Thread.join
    interrupts: list[KeyboardInterrupt] = []

    def interrupted_join(thread: threading.Thread, *args: Any, **kwargs: Any) -> None:
        if thread.name == "qldpc-search" and len(interrupts) < num_interrupts:
            interrupts.append(KeyboardInterrupt())
            raise interrupts[-1]
        join(thread, *args, **kwargs)

    calling_thread = threading.get_ident()
    working = threading.Event()
    stopped: list[bool] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        def run_range(start: int, end: int) -> None:
            # The calling thread runs out of work while the other thread works.
            if threading.get_ident() == calling_thread:
                assert working.wait(timeout=_TIMEOUT)
                return
            working.set()
            stopped.append(stop.wait(timeout=_TIMEOUT))

        return run_range

    monkeypatch.setattr(threading.Thread, "join", interrupted_join)
    with pytest.raises(KeyboardInterrupt) as raised:
        _threads.run_work_items(2, 2, start_worker)
    assert stopped == [True]
    assert raised.value is interrupts[-1]
