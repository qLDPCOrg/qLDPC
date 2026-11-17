# SPDX-License-Identifier: Apache-2.0

"""Unit tests for running the work items of a search on threads."""

from __future__ import annotations

import functools
import itertools
import math
import threading
from collections.abc import Callable, Iterator
from typing import Any

import numba
import numpy as np
import pytest

from qldpc import _num_threads

from . import _threads

# How long a test waits for an event that a thread should set at once.
_TIMEOUT = 10


@pytest.fixture
def no_threads_left() -> Iterator[None]:
    """Check that a test leaves no thread running that it started."""
    before = set(threading.enumerate())
    yield
    assert set(threading.enumerate()) <= before


def test_the_number_of_threads_defaults_to_qldpc_num_threads_or_else_to_numba(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit number wins, then QLDPC_NUM_THREADS if it was set, then numba's count."""
    monkeypatch.setattr(numba.config, "NUMBA_NUM_THREADS", 3)
    monkeypatch.setattr(_num_threads, "NUM_THREADS", None)
    assert _threads.get_num_threads(None) == 3
    assert _threads.get_num_threads(2) == 2
    monkeypatch.setattr(_num_threads, "NUM_THREADS", 5)
    assert _threads.get_num_threads(None) == 5
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
    items of layer seven of a set of 150 rows, which fix one row each, so their weights decrease;
    these properties are for decreasing weights, since a heavy item can make a range overshoot.
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


def test_every_item_runs_when_a_share_of_the_weight_rounds_away() -> None:
    """A range holds at least one item, even when its share of the weight is lost to rounding.

    The threads are capped at the four items.  After the first item, an eighth of the three units of
    weight left falls below half the spacing of floats near ``8e15``, so the share adds nothing to
    the weight claimed so far.
    """
    claimed: list[int] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        return lambda start, end: claimed.extend(range(start, end))

    cumulative_weights = 8e15 + np.arange(4, dtype=np.float64)
    _threads.run_work_items(4, 4, start_worker, cumulative_weights=cumulative_weights)
    assert sorted(claimed) == [0, 1, 2, 3]


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
def test_an_interrupt_while_a_thread_starts_stops_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A keyboard interrupt as a new thread starts stops that thread, which claims nothing more.

    The start of the thread raises the interrupt once the thread has claimed its first range, in
    place of a real one, which no test can time.  That range waits for the stop event, so the
    thread claims no other range, and the calling thread claims none.
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

    monkeypatch.setattr(threading.Thread, "start", interrupted_start)
    with pytest.raises(KeyboardInterrupt):
        _threads.run_work_items(10, 3, start_worker)
    assert len(claims) == 1 and stopped == [True]


def test_a_thread_whose_start_is_interrupted_before_it_starts_never_works(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thread whose start is interrupted before the thread starts is skipped, and does no work.

    The start raises the interrupt in place of a real one, and keeps the thread, which the test
    then starts for real.  Its worker never starts, since the stop event is set by then.  The
    search runs on a helper thread, so that a cleanup that waited for the kept thread would time
    out rather than hang.
    """
    start = threading.Thread.start
    kept: list[threading.Thread] = []
    worker_starts = itertools.count()
    raised: list[BaseException] = []

    def interrupted_start(thread: threading.Thread) -> None:
        if thread.name != "qldpc-search":
            start(thread)
            return
        kept.append(thread)
        raise KeyboardInterrupt

    def search() -> None:
        try:
            # A worker counts its start, and would run no range.
            _threads.run_work_items(
                10, 2, lambda stop: functools.partial(print, next(worker_starts))
            )
        except BaseException as error:  # noqa: BLE001 - the test checks which error the search raised
            raised.append(error)

    monkeypatch.setattr(threading.Thread, "start", interrupted_start)
    helper = threading.Thread(target=search, daemon=True)
    helper.start()
    helper.join(timeout=_TIMEOUT)
    assert not helper.is_alive()
    (error,) = raised
    assert isinstance(error, KeyboardInterrupt)

    (thread,) = kept
    start(thread)
    thread.join(timeout=_TIMEOUT)
    assert not thread.is_alive() and next(worker_starts) == 0


@pytest.mark.usefixtures("no_threads_left")
def test_an_interrupt_as_the_calling_thread_starts_to_wait_stops_the_other_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A keyboard interrupt as the calling thread starts to wait for the others stops them too.

    The wait raises the interrupt as it is entered, in place of a real one, and then waits.
    """
    join = _threads._join
    interrupted: list[bool] = []

    def interrupted_join(*args: Any) -> None:
        if not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt
        join(*args)

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

    monkeypatch.setattr(_threads, "_join", interrupted_join)
    with pytest.raises(KeyboardInterrupt):
        _threads.run_work_items(2, 2, start_worker)
    assert stopped == [True]


@pytest.mark.usefixtures("no_threads_left")
def test_an_exception_of_the_calling_thread_wins_over_those_of_the_others() -> None:
    """The calling thread's exception, as at a keyboard interrupt, is raised, even if it comes last.

    The other thread fails first, and the calling thread raises once the stop event is set.
    """
    calling_thread = threading.get_ident()
    started = threading.Barrier(2, timeout=_TIMEOUT)

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        def run_range(start: int, end: int) -> None:
            started.wait()
            if threading.get_ident() != calling_thread:
                raise ValueError("first")
            assert stop.wait(timeout=_TIMEOUT)
            raise KeyboardInterrupt

        return run_range

    with pytest.raises(KeyboardInterrupt):
        _threads.run_work_items(2, 2, start_worker)


@pytest.mark.usefixtures("no_threads_left")
def test_the_first_error_of_the_other_threads_is_raised() -> None:
    """When several other threads fail, the error of the first to fail is raised again."""
    calling_thread = threading.get_ident()
    started = threading.Barrier(3, timeout=_TIMEOUT)
    order: list[int] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        def run_range(start: int, end: int) -> None:
            started.wait()
            if threading.get_ident() == calling_thread:
                assert stop.wait(timeout=_TIMEOUT)
                return
            # The thread that gets here first fails first, and the other fails after its error is
            # recorded.
            with lock:
                order.append(threading.get_ident())
                first = len(order) == 1
            if first:
                raise ValueError("first")
            # The stop event is set once the first error is recorded.
            assert stop.wait(timeout=_TIMEOUT)
            raise ValueError("second")

        return run_range

    lock = threading.Lock()
    with pytest.raises(ValueError, match="first"):
        _threads.run_work_items(3, 3, start_worker)


@pytest.mark.usefixtures("no_threads_left")
def test_another_error_while_the_calling_thread_waits_stops_the_other_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A runtime error while the calling thread waits, as from a signal handler, stops the others.

    The error is raised again once the other thread has returned.  The wait for the thread raises
    it, in place of a real one.
    """

    class HandlerError(RuntimeError):
        """An error that a signal handler raises."""

    join = threading.Thread.join
    errors: list[HandlerError] = []

    def failing_join(thread: threading.Thread, *args: Any, **kwargs: Any) -> None:
        if thread.name == "qldpc-search" and not errors:
            errors.append(HandlerError("from a signal handler"))
            raise errors[-1]
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

    monkeypatch.setattr(threading.Thread, "join", failing_join)
    with pytest.raises(HandlerError) as raised:
        _threads.run_work_items(2, 2, start_worker)
    assert stopped == [True]
    assert raised.value is errors[0]


@pytest.mark.usefixtures("no_threads_left")
@pytest.mark.parametrize(("method", "num_interrupts"), [("join", 1), ("join", 2), ("is_alive", 1)])
def test_an_interrupt_while_the_calling_thread_waits_stops_the_other_threads(
    monkeypatch: pytest.MonkeyPatch, method: str, num_interrupts: int
) -> None:
    """A keyboard interrupt while the calling thread waits for a working thread stops that thread.

    The other thread stops after its current chunk, and the interrupt is raised once it has
    returned, also if another interrupt comes while the calling thread waits again.  The wait for
    the thread, or its check of whether the thread is alive, raises the interrupts as it starts,
    in place of real ones.
    """
    original = getattr(threading.Thread, method)
    interrupts: list[KeyboardInterrupt] = []

    def interrupted(thread: threading.Thread, *args: Any, **kwargs: Any) -> Any:
        if thread.name == "qldpc-search" and len(interrupts) < num_interrupts:
            interrupts.append(KeyboardInterrupt())
            raise interrupts[-1]
        return original(thread, *args, **kwargs)

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

    monkeypatch.setattr(threading.Thread, method, interrupted)
    with pytest.raises(KeyboardInterrupt) as raised:
        _threads.run_work_items(2, 2, start_worker)
    assert stopped == [True]
    assert raised.value is interrupts[-1]
