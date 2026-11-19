# SPDX-License-Identifier: Apache-2.0

"""Unit tests for running the work items of a search on threads."""

from __future__ import annotations

import contextlib
import itertools
import math
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numba
import numpy as np
import pytest

from qldpc import _num_threads

from . import _threads
from .conftest import THREAD_TIMEOUT, run_on_a_thread

pytestmark = pytest.mark.usefixtures("no_threads_left")


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
        ranges: list[tuple[int, int]] = []
        claims.append(ranges)
        return lambda start, end: ranges.append((start, end))

    _threads.run_work_items(num_items, num_threads, start_worker)
    assert len(workers) == num_threads and threading.get_ident() in workers
    assert all(ranges == sorted(ranges) for ranges in claims)
    ranges = sorted(claimed for worker_ranges in claims for claimed in worker_ranges)
    assert [item for start, end in ranges for item in range(start, end)] == list(range(num_items))
    assert len(ranges) <= 2 * num_threads * (1 + math.log(num_items))


@pytest.mark.parametrize("num_threads", [1, 3])
def test_single_items_are_claimed_one_at_a_time(num_threads: int) -> None:
    """With ``single_items``, each range holds one item, and every item runs once."""
    claims: list[tuple[int, int]] = []
    lock = threading.Lock()

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        def run_range(start: int, end: int) -> None:
            with lock:
                claims.append((start, end))

        return run_range

    _threads.run_work_items(50, num_threads, start_worker, single_items=True)
    assert sorted(claims) == [(item, item + 1) for item in range(50)]


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

    Four threads share the four items.  After the first item, an eighth of the three units of
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
    """With no items, only the calling thread starts a worker."""
    workers: list[int] = []
    _threads.run_work_items(0, 4, _get_worker_that_records_starts(workers))
    assert workers == [threading.get_ident()]


def test_a_range_can_stop_every_thread() -> None:
    """Once a range sets the stop event, no thread claims another range.

    Each range sets the event, so each thread claims at most one.
    """
    claims: list[list[tuple[int, int]]] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        ranges: list[tuple[int, int]] = []
        claims.append(ranges)

        def run_range(start: int, end: int) -> None:
            ranges.append((start, end))
            stop.set()

        return run_range

    _threads.run_work_items(100, 3, start_worker)
    assert any(claims) and all(len(ranges) <= 1 for ranges in claims)


@pytest.mark.parametrize("raises_on_calling_thread", [False, True])
def test_an_exception_stops_every_thread_and_is_raised_again(
    raises_on_calling_thread: bool,
) -> None:
    """An exception in any thread, such as an interrupt in the calling one, stops all threads.

    Every thread has returned when the exception is raised again.
    """
    calling_thread = threading.get_ident()
    stopped: list[bool] = []
    started = threading.Barrier(2, timeout=THREAD_TIMEOUT)

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        on_calling_thread = threading.get_ident() == calling_thread

        def run_range(start: int, end: int) -> None:
            started.wait()
            if on_calling_thread == raises_on_calling_thread:
                raise KeyboardInterrupt if on_calling_thread else ValueError("failed")
            # Wait for the stop event, as a thread that checks it between chunks would.
            stopped.append(stop.wait(timeout=THREAD_TIMEOUT))

        return run_range

    error = KeyboardInterrupt if raises_on_calling_thread else ValueError
    with pytest.raises(error):
        _threads.run_work_items(10, 2, start_worker)
    assert stopped == [True]


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
            assert claimed.wait(timeout=THREAD_TIMEOUT)
            raise KeyboardInterrupt

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        def run_range(start: int, end: int) -> None:
            claims.append((start, end))
            claimed.set()
            stopped.append(stop.wait(timeout=THREAD_TIMEOUT))

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
    workers: list[int] = []
    raised: list[BaseException] = []

    def interrupted_start(thread: threading.Thread) -> None:
        if thread.name != "qldpc-search":
            start(thread)
            return
        kept.append(thread)
        raise KeyboardInterrupt

    def search() -> None:
        try:
            _threads.run_work_items(10, 2, _get_worker_that_records_starts(workers))
        except BaseException as error:  # noqa: BLE001 - the test checks which error the search raised
            raised.append(error)

    monkeypatch.setattr(threading.Thread, "start", interrupted_start)
    helper = threading.Thread(target=search, daemon=True)
    helper.start()
    helper.join(timeout=THREAD_TIMEOUT)
    assert not helper.is_alive()
    (error,) = raised
    assert isinstance(error, KeyboardInterrupt)

    (thread,) = kept
    start(thread)
    thread.join(timeout=THREAD_TIMEOUT)
    assert not thread.is_alive() and not workers


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

    start_worker, stopped = _get_worker_for_the_other_thread()

    monkeypatch.setattr(_threads, "_join", interrupted_join)
    with pytest.raises(KeyboardInterrupt):
        _threads.run_work_items(2, 2, start_worker)
    assert stopped == [True]


def test_an_exception_of_the_calling_thread_wins_over_those_of_the_others() -> None:
    """The calling thread's exception, as at a keyboard interrupt, is raised, even if it comes last.

    The other thread fails first, and the calling thread raises once the stop event is set.
    """
    calling_thread = threading.get_ident()
    started = threading.Barrier(2, timeout=THREAD_TIMEOUT)

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        def run_range(start: int, end: int) -> None:
            started.wait()
            if threading.get_ident() != calling_thread:
                raise ValueError("first")
            assert stop.wait(timeout=THREAD_TIMEOUT)
            raise KeyboardInterrupt

        return run_range

    with pytest.raises(KeyboardInterrupt):
        _threads.run_work_items(2, 2, start_worker)


def test_the_first_error_of_the_other_threads_is_raised() -> None:
    """When several other threads fail, the error of the first to fail is raised again."""
    calling_thread = threading.get_ident()
    started = threading.Barrier(3, timeout=THREAD_TIMEOUT)
    order: list[int] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        def run_range(start: int, end: int) -> None:
            started.wait()
            if threading.get_ident() == calling_thread:
                assert stop.wait(timeout=THREAD_TIMEOUT)
                return
            # The thread that gets here first fails first, and the other fails after its error is
            # recorded.
            with lock:
                order.append(threading.get_ident())
                first = len(order) == 1
            if first:
                raise ValueError("first")
            # The stop event is set once the first error is recorded.
            assert stop.wait(timeout=THREAD_TIMEOUT)
            raise ValueError("second")

        return run_range

    lock = threading.Lock()
    with pytest.raises(ValueError, match="first"):
        _threads.run_work_items(3, 3, start_worker)


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

    start_worker, stopped = _get_worker_for_the_other_thread()

    monkeypatch.setattr(threading.Thread, "join", failing_join)
    with pytest.raises(HandlerError) as raised:
        _threads.run_work_items(2, 2, start_worker)
    assert stopped == [True]
    assert raised.value is errors[0]


@pytest.mark.parametrize(("method", "num_interrupts"), [("join", 1), ("join", 2), ("is_alive", 2)])
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

    start_worker, stopped = _get_worker_for_the_other_thread()

    monkeypatch.setattr(threading.Thread, method, interrupted)
    with pytest.raises(KeyboardInterrupt) as raised:
        _threads.run_work_items(2, 2, start_worker)
    assert stopped == [True]
    assert raised.value is interrupts[-1]


@pytest.mark.parametrize("ending_items", [[3, 7], []])
@pytest.mark.parametrize("num_threads", [1, 3])
def test_items_run_until_the_first_that_ends_the_work(
    num_threads: int, ending_items: list[int]
) -> None:
    """The first item that ends the work is returned, and every item before it runs in full.

    Each item runs in three chunks, and an item that ends the work does so in its second chunk.  On
    one thread, no item after the first that ends the work runs.
    """
    num_items = 10
    chunks = [0] * num_items

    def start_worker(needed: Callable[[int], bool]) -> Callable[[int], bool]:
        def run_item(item: int) -> bool:
            for chunk in range(3):
                chunks[item] += 1
                if item in ending_items and chunk == 1:
                    return True
            return False

        return run_item

    first = _threads.run_items_until_first(num_items, num_threads, start_worker, single_items=True)
    assert first == (ending_items[0] if ending_items else num_items)
    assert chunks[:first] == [3] * first
    if ending_items:
        assert chunks[first] == 2
    if num_threads == 1:
        assert not any(chunks[first + 1 :])


@pytest.mark.parametrize("raises", [False, True])
def test_a_running_item_learns_once_an_earlier_item_ends_the_work(raises: bool) -> None:
    """An item that asks between its chunks learns that it is not needed once an earlier one ends.

    Item 0 ends the work, or raises an exception, which stops the work, only once item 1 has started
    on the other thread, and item 1 then asks until it is no longer needed.  Without the exception,
    item 1 then ends the work too, which leaves the limit at item 0, since the limit only falls.
    """
    started = threading.Event()
    stopped: list[int] = []

    def start_worker(needed: Callable[[int], bool]) -> Callable[[int], bool]:
        def run_item(item: int) -> bool:
            if item == 0:
                assert started.wait(THREAD_TIMEOUT)
                if raises:
                    raise ValueError("failed")
                return True
            started.set()
            deadline = time.monotonic() + THREAD_TIMEOUT
            while needed(item):
                assert time.monotonic() < deadline
                time.sleep(0.001)
            stopped.append(item)
            return True

        return run_item

    if raises:
        with pytest.raises(ValueError, match="failed"):
            _threads.run_items_until_first(2, 2, start_worker, single_items=True)
    else:
        assert _threads.run_items_until_first(2, 2, start_worker, single_items=True) == 0
    assert stopped == [1]


@pytest.mark.parametrize(("num_rows", "ending_rows"), [(0, []), (1000, [500])])
def test_batches_on_one_thread_run_in_one_call_on_the_calling_thread(
    num_rows: int, ending_rows: list[int]
) -> None:
    """On one thread, a batch, even an empty one, runs in one call on the calling thread.

    The call has no work limit, and the batch returns the row that ends it, if any.
    """
    rows = _Rows(ending_rows)
    index = _threads.run_blocks(num_rows, 1, 0, rows.start_worker)
    assert index == (ending_rows[0] if ending_rows else num_rows)
    assert rows.calls == [(0, num_rows, _threads.NO_WORK_LIMIT, threading.get_ident())]


@pytest.mark.parametrize("ending_rows", [[], [500]])
def test_batches_within_the_split_work_run_in_one_call_on_the_calling_thread(
    ending_rows: list[int],
) -> None:
    """A batch that ends, or whose work stays below the split work, starts no thread."""
    rows = _Rows(ending_rows)
    index = _threads.run_blocks(1000, 4, 10**6 if not ending_rows else 900, rows.start_worker)
    assert index == (ending_rows[0] if ending_rows else 1000)
    split_work = rows.calls[0][2]
    assert rows.calls == [(0, 1000, split_work, threading.get_ident())]
    assert rows.workers == [threading.get_ident()]


@pytest.mark.parametrize(
    ("num_rows", "num_threads", "split_work"), [(100, 2, 50), (1000, 3, 10), (333, 8, 0)]
)
def test_costly_batches_move_the_rest_to_threads_in_contiguous_blocks(
    num_rows: int, num_threads: int, split_work: int
) -> None:
    """The calling thread runs rows until their work reaches the split work, and threads the rest.

    The rest of the batch runs in contiguous blocks that partition it, each in one call.
    """
    rows = _Rows()
    assert _threads.run_blocks(num_rows, num_threads, split_work, rows.start_worker) == num_rows
    first = max(1, split_work)
    assert rows.calls[0] == (0, num_rows, split_work, threading.get_ident())
    blocks = sorted((start, end) for start, end, _, _ in rows.calls[1:])
    assert blocks[0][0] == first and blocks[-1][1] == num_rows
    assert all(end == start for (_, end), (start, _) in itertools.pairwise(blocks))
    assert all(start < end for start, end in blocks)
    assert {max_work for _, _, max_work, _ in rows.calls[1:]} == {_threads.NO_WORK_LIMIT}
    assert sorted(rows.rows) == list(range(num_rows))
    assert len(rows.workers) == 1 + num_threads


@pytest.mark.parametrize("ending_rows", [[1], [6], [6, 21], [32]])
def test_blocks_after_the_first_that_ends_a_batch_are_skipped_by_its_thread(
    ending_rows: list[int],
) -> None:
    """Every row up to the first that ends the batch runs, and its thread runs no later row.

    The batch returns that row.
    """
    rows = _Rows(ending_rows)
    first = ending_rows[0]
    assert _threads.run_blocks(33, 2, 0, rows.start_worker) == first
    assert set(range(first + 1)) <= set(rows.rows)
    (finder, start_of_first) = next(
        (thread, start) for start, end, _, thread in rows.calls[1:] if start <= first < end
    )
    assert max(start for start, _, _, thread in rows.calls[1:] if thread == finder) == (
        start_of_first
    )


def test_a_batch_returns_the_row_that_ends_it_within_a_block() -> None:
    """A batch in blocks of several rows returns the row that ends it, not the start of its block.

    The calling thread runs row 0, and two threads run the other rows in blocks of three.  Row 5
    ends the batch in the middle of its block.
    """
    rows = _Rows([5])
    assert _threads.run_blocks(1 + 6 * _threads.BLOCKS_PER_THREAD, 2, 0, rows.start_worker) == 5
    assert [(start, end) for start, end, _, _ in rows.calls[1:] if start <= 5 < end] == [(4, 7)]


def test_a_block_that_ends_a_batch_after_a_later_one_lowers_the_limit() -> None:
    """A block that ends a batch after a later one did still ends it, and later blocks stop.

    The block of row 6 waits until the block of row 21 has ended the batch.  Then every row up to 6
    runs, and no block between them starts after that.
    """
    later_ended = threading.Event()
    events: list[tuple[str, int]] = []

    def before_row(row: int) -> None:
        if row == 6:
            assert later_ended.wait(THREAD_TIMEOUT)

    class _LoggedRows(_Rows):
        def start_worker(self) -> _threads.RunRows:
            run_rows = super().start_worker()

            def logged(start: int, end: int, max_work: int) -> tuple[int, bool]:
                with self.lock:
                    events.append(("start", start))
                answer = run_rows(start, end, max_work)
                with self.lock:
                    events.append(("end", start))
                if start <= 21 < end and start > 0:
                    later_ended.set()
                return answer

            return logged

    rows = _LoggedRows([6, 21], before_row)
    assert _threads.run_blocks(33, 2, 0, rows.start_worker) == 6
    assert set(range(7)) <= set(rows.rows)
    (block_of_6,) = [start for start, end, _, _ in rows.calls[1:] if start <= 6 < end]
    after = events[events.index(("end", block_of_6)) + 1 :]
    assert not [start for kind, start in after if kind == "start" and start > 6]


@pytest.mark.parametrize("num_threads", [1, 2, 3, 8])
def test_outputs_come_in_the_order_of_the_items(num_threads: int) -> None:
    """The calling thread gets the outputs of every item once, in order, on any number of threads.

    The calling thread runs no item unless it runs them all.  With two or more threads, ranges hold
    the fewest next items whose weight reaches the share of a range: here a 32nd of the weight for
    each thread that runs items, which a single heavy item can exceed.
    """
    weights = np.array([1.0, 50.0] + [1.0] * 198)
    ranges: list[tuple[int, int, int]] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], Iterator[tuple[int, int]]]:
        def run_range(start: int, end: int) -> Iterator[tuple[int, int]]:
            ranges.append((start, end, threading.get_ident()))
            for item in range(start, end):
                yield from ((item, output) for output in range(item % 3))

        return run_range

    def iterate() -> tuple[list[tuple[int, int]], int]:
        outputs = list(
            _threads.iterate_in_order(
                len(weights), num_threads, start_worker, cumulative_weights=np.cumsum(weights)
            )
        )
        return outputs, threading.get_ident()

    outputs, calling_thread = run_on_a_thread(iterate)
    assert outputs == [(item, output) for item in range(200) for output in range(item % 3)]
    ranges.sort()
    assert [item for start, end, _ in ranges for item in range(start, end)] == list(range(200))
    on_calling_thread = [start for start, _, thread in ranges if thread == calling_thread]
    if num_threads == 1:
        assert ranges == [(0, 200, calling_thread)]
    else:
        share = weights.sum() / (_threads.RANGES_PER_THREAD * (num_threads - 1))
        assert not on_calling_thread
        for start, end, _ in ranges:
            assert weights[start : end - 1].sum() < share <= weights[start:end].sum() or end == 200


def test_threads_are_no_more_than_the_items() -> None:
    """With no items, no thread starts, and with fewer items than threads, one for each item."""
    workers: list[int] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], Iterator[int]]:
        workers.append(threading.get_ident())
        return lambda start, end: iter(range(start, end))

    for num_threads in (1, 3):
        assert not list(_threads.iterate_in_order(0, num_threads, start_worker))
    assert workers == [threading.get_ident()]
    workers.clear()
    assert list(_threads.iterate_in_order(2, 8, start_worker)) == [0, 1]
    assert len(set(workers)) == 2 and threading.get_ident() not in workers


@pytest.mark.parametrize("close_early", [False, True])
def test_outputs_keep_their_order_whichever_range_ends_first(close_early: bool) -> None:
    """A range that runs before an earlier one holds its outputs, at most the cap, for its turn.

    With one output held for each thread, two threads hold two outputs.  The thread of range 1
    adds two outputs, and waits with the third until the thread of range 0 adds its first, which
    it lists only once range 1 has made its third.  So range 1 has not made its fourth output when
    the calling thread gets the first.  Closing the iterator then stops both threads, the one of
    range 1 while it waits.
    """
    made_third = threading.Event()
    made: list[tuple[int, int]] = []
    outputs_by_item = {0: [(0, 0), (0, 1)], 1: [(1, 0), (1, 1), (1, 2), (1, 3)], 2: [(2, 0)]}

    def before_output(item: int, output: int) -> None:
        if (item, output) == (0, 0):
            assert made_third.wait(timeout=THREAD_TIMEOUT)
        if (item, output) == (1, 2):
            made_third.set()
        made.append((item, output))

    def iterate() -> list[tuple[int, int]]:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(_threads, "OUTPUTS_PER_THREAD", 1)
            monkeypatch.setattr(_threads, "RANGES_PER_THREAD", 2)
            listing = _start_listing(outputs_by_item, before_output)
            iterator = _threads.iterate_in_order(3, 3, listing)
            with contextlib.closing(iterator):
                outputs = [next(iterator)]
                assert (1, 3) not in made
                if not close_early:
                    outputs += list(iterator)
        return outputs

    outputs = run_on_a_thread(iterate)
    if close_early:
        assert outputs == [(0, 0)] and (1, 3) not in made
    else:
        assert outputs == [output for item in range(3) for output in outputs_by_item[item]]


def test_an_exception_of_a_thread_is_raised_with_no_output_after_its_own() -> None:
    """The calling thread gets outputs in order up to the one before an exception, then the error.

    It may get the exception before some of the outputs before it, but never gets an output after
    it, whichever range ends first.
    """
    outputs_by_item = {item: [(item, 0), (item, 1)] for item in range(6)}

    def before_output(item: int, output: int) -> None:
        if (item, output) == (3, 1):
            raise ValueError("failed")

    received: list[tuple[int, int]] = []

    def iterate() -> None:
        with (
            pytest.MonkeyPatch.context() as monkeypatch,
            pytest.raises(ValueError, match="failed"),
        ):
            monkeypatch.setattr(_threads, "RANGES_PER_THREAD", 3)
            listing = _start_listing(outputs_by_item, before_output)
            iterator = _threads.iterate_in_order(6, 3, listing)
            while True:
                received.append(next(iterator))

    run_on_a_thread(iterate)
    before = [output for item in range(4) for output in outputs_by_item[item]][:7]
    assert received == before[: len(received)]


def test_an_exception_of_a_thread_stops_a_thread_that_waits() -> None:
    """A thread that waits to add an output stops when another raises, so the exception comes.

    With one output held for each thread, the thread of range 1 adds two outputs and waits with the
    third, and then the thread of range 0 raises before its first output.
    """
    made_third = threading.Event()
    outputs_by_item = {0: [(0, 0)], 1: [(1, 0), (1, 1), (1, 2)], 2: [(2, 0)]}

    def before_output(item: int, output: int) -> None:
        if (item, output) == (0, 0):
            assert made_third.wait(timeout=THREAD_TIMEOUT)
            raise ValueError("failed")
        if (item, output) == (1, 2):
            made_third.set()

    def iterate() -> None:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(_threads, "OUTPUTS_PER_THREAD", 1)
            monkeypatch.setattr(_threads, "RANGES_PER_THREAD", 2)
            listing = _start_listing(outputs_by_item, before_output)
            with pytest.raises(ValueError, match="failed"):
                list(_threads.iterate_in_order(3, 3, listing))

    run_on_a_thread(iterate)


def test_a_thread_that_fails_to_start_stops_the_others() -> None:
    """If a thread raises as it starts, the others stop, also one that waits to add an output.

    Of three threads, the first claims range 0 and waits to list it, and the second, with one
    output held for each thread, adds three outputs of range 1 and waits with its fourth.  Then the
    third raises as it starts, which lets the first list.
    """
    starts: list[int] = []
    lock = threading.Lock()
    claimed, waiting, released = (threading.Event() for _ in range(3))

    def list_first(start: int, end: int) -> Iterator[int]:
        claimed.set()
        assert released.wait(timeout=THREAD_TIMEOUT)
        yield start

    def list_second(start: int, end: int) -> Iterator[int]:
        for output in range(10):
            if output == 3:
                waiting.set()
            yield output

    def start_worker(stop: threading.Event) -> Callable[[int, int], Iterator[int]]:
        with lock:
            starts.append(threading.get_ident())
            index = len(starts) - 1
        if index == 0:
            return list_first
        assert claimed.wait(timeout=THREAD_TIMEOUT)
        if index == 1:
            return list_second
        assert waiting.wait(timeout=THREAD_TIMEOUT)
        released.set()
        raise MemoryError("no room")

    def iterate() -> None:
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(_threads, "OUTPUTS_PER_THREAD", 1)
            monkeypatch.setattr(_threads, "RANGES_PER_THREAD", 1)
            with pytest.raises(MemoryError, match="no room"):
                list(_threads.iterate_in_order(6, 4, start_worker))

    run_on_a_thread(iterate)
    assert len(starts) == 3


####################################################################################################
# helpers


def _get_worker_that_records_starts(
    workers: list[int],
) -> Callable[[threading.Event], Callable[[int, int], None]]:
    """A worker that records the thread on which it starts, and does nothing with its ranges."""

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        workers.append(threading.get_ident())
        return _run_no_range

    return start_worker


def _run_no_range(start: int, end: int) -> None:
    """Run a range by doing nothing."""


def _start_listing(
    outputs_by_item: dict[int, list[tuple[int, int]]],
    before_output: Callable[[int, int], None] = lambda item, output: None,
) -> Callable[[threading.Event], Callable[[int, int], Iterator[tuple[int, int]]]]:
    """A worker of ``iterate_in_order`` that yields the outputs of each item of its ranges.

    ``before_output`` runs before each output, given its item and its index among the outputs of
    the item.
    """

    def start_worker(stop: threading.Event) -> Callable[[int, int], Iterator[tuple[int, int]]]:
        def run_range(start: int, end: int) -> Iterator[tuple[int, int]]:
            for item in range(start, end):
                for index, output in enumerate(outputs_by_item[item]):
                    before_output(item, index)
                    yield output

        return run_range

    return start_worker


def _get_worker_for_the_other_thread() -> tuple[
    Callable[[threading.Event], Callable[[int, int], None]], list[bool]
]:
    """A worker with which the calling thread runs out of work while the other thread works.

    The other thread works until the stop event is set, and the list records, for each of its
    ranges, whether the event was set before the timeout.  Call this on the calling thread.
    """
    calling_thread = threading.get_ident()
    working = threading.Event()
    stopped: list[bool] = []

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        def run_range(start: int, end: int) -> None:
            if threading.get_ident() == calling_thread:
                assert working.wait(timeout=THREAD_TIMEOUT)
                return
            working.set()
            stopped.append(stop.wait(timeout=THREAD_TIMEOUT))

        return run_range

    return start_worker, stopped


class _Rows:
    """Rows of a batch for ``run_blocks``, each of work one, which record how they run.

    ``calls`` lists each call of a worker as ``(start, end, max_work, thread)``, and ``rows`` each
    row that runs.  A row in ``ending_rows`` ends the batch, and ``before_row`` runs before a row.
    """

    def __init__(
        self, ending_rows: Sequence[int] = (), before_row: Callable[[int], None] | None = None
    ) -> None:
        self.ending_rows = set(ending_rows)
        self.before_row = before_row
        self.calls: list[tuple[int, int, int, int]] = []
        self.rows: list[int] = []
        self.workers: list[int] = []
        self.lock = threading.Lock()

    def start_worker(self) -> _threads.RunRows:
        """Record the worker, and return how it runs rows."""
        self.workers.append(threading.get_ident())

        def run_rows(start: int, end: int, max_work: int) -> tuple[int, bool]:
            with self.lock:
                self.calls.append((start, end, max_work, threading.get_ident()))
            for row in range(start, end):
                if self.before_row is not None:
                    self.before_row(row)
                with self.lock:
                    self.rows.append(row)
                if row in self.ending_rows:
                    return row + 1, True
                # Each row adds one to the work of the call.
                if row + 1 - start >= max_work:
                    return row + 1, False
            return end, False

        return run_rows
