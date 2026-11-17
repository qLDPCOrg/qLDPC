# SPDX-License-Identifier: Apache-2.0

"""Unit tests for running the work items of a search on threads."""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numba
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

    Each of ``num_threads`` workers starts once, on the calling thread or on a thread of the pool.
    The first range holds the share of the items that guided self-scheduling gives it.
    """
    num_items = 50
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
    assert ranges[0] == (0, num_items // (2 * num_threads))


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
def test_an_interrupt_before_the_calling_thread_works_stops_the_other_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A keyboard interrupt while threads start, as the second starts, stops the first.

    The pool of threads raises the interrupt in place of a real one, which no test can time.  The
    first thread waits for the stop event before it claims a range, so it claims none.
    """
    submit = concurrent.futures.ThreadPoolExecutor.submit
    first_started = threading.Event()
    stopped: list[bool] = []
    claimed: list[tuple[int, int]] = []

    def interrupted_submit(pool: Any, *args: Any) -> Any:
        if first_started.is_set():
            raise KeyboardInterrupt
        future = submit(pool, *args)
        assert first_started.wait(timeout=_TIMEOUT)
        return future

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        first_started.set()
        stopped.append(stop.wait(timeout=_TIMEOUT))
        return lambda start, end: claimed.append((start, end))

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "submit", interrupted_submit)
    with pytest.raises(KeyboardInterrupt):
        _threads.run_work_items(10, 3, start_worker)
    assert stopped == [True] and not claimed


@pytest.mark.usefixtures("no_threads_left")
@pytest.mark.parametrize("num_interrupts", [1, 2])
def test_an_interrupt_while_the_calling_thread_waits_stops_the_other_threads(
    monkeypatch: pytest.MonkeyPatch, num_interrupts: int
) -> None:
    """A keyboard interrupt while the calling thread waits for a working thread stops that thread.

    The other thread stops after its current chunk, and the interrupt is raised once it has
    returned, also if another interrupt comes while the calling thread waits again.  The pool of
    threads raises the interrupts as the calling thread starts to wait, in place of real ones.
    """
    shutdown = concurrent.futures.ThreadPoolExecutor.shutdown
    interrupts: list[KeyboardInterrupt] = []

    def interrupted_shutdown(pool: Any, *args: Any, **kwargs: Any) -> None:
        if len(interrupts) < num_interrupts:
            interrupts.append(KeyboardInterrupt())
            raise interrupts[-1]
        shutdown(pool, *args, **kwargs)

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

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "shutdown", interrupted_shutdown)
    with pytest.raises(KeyboardInterrupt) as raised:
        _threads.run_work_items(2, 2, start_worker)
    assert stopped == [True]
    assert raised.value is interrupts[-1]
