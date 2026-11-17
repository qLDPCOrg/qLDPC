# SPDX-License-Identifier: Apache-2.0

"""Unit tests for running the work items of a search on threads."""

from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable
from typing import Any

import numba
import pytest

from . import _threads


def test_the_number_of_threads_defaults_to_that_of_numba() -> None:
    assert _threads.get_num_threads(None) == vars(numba.config)["NUMBA_NUM_THREADS"]
    assert _threads.get_num_threads(3) == 3


@pytest.mark.parametrize("num_threads", [1, 2, 3, 8])
def test_every_item_runs_once(num_threads: int) -> None:
    """Each item runs once, on at most one thread for each item, one of which is the calling one."""
    num_items = 5
    claims: list[tuple[int, int]] = []
    workers: list[int] = []

    def start_worker(stop: threading.Event) -> Callable[[int], None]:
        workers.append(threading.get_ident())
        return lambda item: claims.append((item, threading.get_ident()))

    before = threading.active_count()
    _threads.run_work_items(num_items, num_threads, start_worker)
    assert sorted(item for item, _ in claims) == list(range(num_items))
    assert len(workers) == min(num_threads, num_items)
    assert threading.get_ident() in workers
    assert threading.active_count() == before


def test_no_items_run_on_the_calling_thread_alone() -> None:
    workers: list[int] = []

    def start_worker(stop: threading.Event) -> Callable[[int], None]:
        workers.append(threading.get_ident())
        return print

    _threads.run_work_items(0, 4, start_worker)
    assert workers == [threading.get_ident()]


def test_an_item_can_stop_every_thread() -> None:
    """Once an item sets the stop event, no thread claims another item."""
    claimed: list[int] = []

    def start_worker(stop: threading.Event) -> Callable[[int], None]:
        def run_item(item: int) -> None:
            claimed.append(item)
            if item == 3:
                stop.set()

        return run_item

    _threads.run_work_items(100, 3, start_worker)
    assert 3 in claimed and len(claimed) < 100


@pytest.mark.parametrize("raises_on_calling_thread", [False, True])
def test_the_first_exception_stops_every_thread_and_is_raised_again(
    raises_on_calling_thread: bool,
) -> None:
    """An exception in any thread, such as an interrupt in the calling one, stops all threads.

    Every thread has returned when the exception is raised again.
    """
    calling_thread = threading.get_ident()
    stopped: list[bool] = []
    started = threading.Barrier(2)

    def start_worker(stop: threading.Event) -> Callable[[int], None]:
        on_calling_thread = threading.get_ident() == calling_thread

        def run_item(item: int) -> None:
            if on_calling_thread == raises_on_calling_thread:
                started.wait()
                raise KeyboardInterrupt if on_calling_thread else ValueError("failed")
            started.wait()
            # Work in chunks until the other thread's exception sets the stop event.
            stop.wait()
            stopped.append(True)

        return run_item

    before = threading.active_count()
    error = KeyboardInterrupt if raises_on_calling_thread else ValueError
    with pytest.raises(error):
        _threads.run_work_items(10, 2, start_worker)
    assert stopped == [True]
    assert threading.active_count() == before


def test_an_interrupt_before_the_calling_thread_works_stops_the_other_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A keyboard interrupt while threads start, as the second starts, stops the first.

    The pool of threads raises the interrupt in place of a real one, which no test can time.  The
    first thread waits for the stop event before it claims an item, so it claims none.
    """
    submit = concurrent.futures.ThreadPoolExecutor.submit
    first_started = threading.Event()
    stopped: list[bool] = []
    claimed: list[int] = []

    def interrupted_submit(pool: Any, *args: Any) -> Any:
        if first_started.is_set():
            raise KeyboardInterrupt
        future = submit(pool, *args)
        first_started.wait()
        return future

    def start_worker(stop: threading.Event) -> Callable[[int], None]:
        first_started.set()
        stopped.append(stop.wait())
        return claimed.append

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "submit", interrupted_submit)
    before = threading.active_count()
    with pytest.raises(KeyboardInterrupt):
        _threads.run_work_items(10, 3, start_worker)
    assert stopped == [True] and not claimed
    assert threading.active_count() == before
