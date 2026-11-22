# SPDX-License-Identifier: Apache-2.0

"""Tests of the shared code of the experiments, with a stand-in experiment.

Everything in ``state_prep_certification/`` is a prototype, subject to change: its code, interfaces,
inputs, outputs, and documentation.

    python experiments/state_prep_certification/common/selftest.py [--node] [--only NAME ...] \
        [--keep]

Each test writes into a fresh temporary directory and uses the stand-in experiment of
``selftest_plugin.py``.  By default, runs use local mode (``--local``), which keeps every run in
this session.  With ``--node`` (Linux only), the tests of the runner, of GNU parallel, and of
``--cleanup`` run in node mode instead: each run in a new session, killed by process group.  The
tests that run GNU parallel pin each run's child to its job slot's CPU, and only the test of GNU
parallel's pinning also runs the runners on other CPUs and checks both affinities; the other tests
run their runs on this process's CPUs.  ``run_node.py`` runs the node-mode tests before the node's
acceptance tests.  The tests that need git, of ``bundle.check_tree`` and of a dirty tree's label,
run only without ``--node``.  The tests of GNU parallel need ``parallel`` on the ``PATH``, and the
test of ``node.sh`` runs copies of it with ``bash``: through its install on Linux x86-64, with
stand-ins of uv, GNU parallel, and a bundle's interpreter, and up to its preflight elsewhere.

The script prints one line per test and exits with status 1 if any test failed.  An exception that
a test raises fails that test only, if it derives from ``Exception`` or is ``SystemExit``
(``acceptance.FAILURES``), and ``--only`` refuses an unknown test name.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import hashlib
import importlib
import io
import json
import multiprocessing
import os
import pathlib
import platform
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import types
from collections.abc import Callable, Iterator
from typing import Any
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.state_prep_certification import bundle, run_node
from experiments.state_prep_certification.common import (
    acceptance,
    local_runs,
    manifest,
    plugin,
    runner,
    selftest_plugin,
    store,
    tasks,
)

RUN_NODE = plugin.EXPERIMENTS_DIR / "run_node.py"
TESTS: dict[str, Callable[[pathlib.Path, bool], None]] = {}
WORKSTATION_ONLY: set[str] = set()


def main() -> None:
    """Run the selected tests."""
    args = parse_args()
    if args.node and not sys.platform.startswith("linux"):
        sys.exit("--node needs Linux")
    failed = []
    for name, test in TESTS.items():
        if args.only and name not in args.only:
            continue
        if args.node and name in WORKSTATION_ONLY:
            print(f"SKIP {name}: workstation only", flush=True)
            continue
        directory = pathlib.Path(tempfile.mkdtemp(prefix=f"spc-{name}-"))
        start = time.monotonic()
        try:
            test(directory, args.node)
            print(f"PASS {name} ({time.monotonic() - start:.1f} s)", flush=True)
        except acceptance.FAILURES:
            failed.append(name)
            print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
        finally:
            if not args.keep:
                shutil.rmtree(directory, ignore_errors=True)
    print(f"{len(failed)} failed: {failed}" if failed else "all passed")
    sys.exit(1 if failed else 0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--node", action="store_true", help="test node mode (Linux)")
    parser.add_argument("--only", nargs="*", choices=sorted(TESTS), help="run only these tests")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directories")
    return parser.parse_args()


def test(
    function: Callable[[pathlib.Path, bool], None],
) -> Callable[[pathlib.Path, bool], None]:
    """Register a test."""
    TESTS[function.__name__.removeprefix("test_")] = function
    return function


def workstation_only(
    function: Callable[[pathlib.Path, bool], None],
) -> Callable[[pathlib.Path, bool], None]:
    """Mark a registered test to run only without ``--node``."""
    WORKSTATION_ONLY.add(function.__name__.removeprefix("test_"))
    return function


####################################################################################################
# store


@test
def test_store_concurrent_appends(directory: pathlib.Path, node: bool) -> None:
    """Rows appended by concurrent processes are all complete and readable."""
    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(target=_append_many, args=(directory, worker, 50)) for worker in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
        assert process.exitcode == 0
    rows, problems = store.read_rows(directory)
    assert len(rows) == 200 and not problems, problems
    assert len({row["task_key"] for row in rows}) == 200


@test
def test_store_recovery(directory: pathlib.Path, node: bool) -> None:
    """An incomplete tail is ignored on reading, and cut before the next append."""
    store.append_row(directory, {"task_key": "a", "run_id": "1"})
    with open(directory / store.RUNS_FILE, "ab") as file:
        file.write(b'{"task_key": "b", "run_')
    rows, problems = store.read_rows(directory)
    assert [row["task_key"] for row in rows] == ["a"]
    assert len(problems) == 1 and "incomplete last line" in problems[0]

    store.append_row(directory, {"task_key": "c", "run_id": "2"})
    store.append_row(directory, {"task_key": "a", "run_id": "3"})
    rows, problems = store.read_rows(directory)
    assert [row["task_key"] for row in rows] == ["a", "c", "a"] and not problems
    assert b'"run_' in (directory / store.RECOVERED_FILE).read_bytes()

    final, problems = store.final_rows(directory)
    assert final["a"]["run_id"] == "1" and set(final) == {"a", "c"}
    assert len(problems) == 1 and "duplicate final row for a" in problems[0]

    # an interior line that is not a row is skipped and reported
    with open(directory / store.RUNS_FILE, "ab") as file:
        file.write(b"not json\n")
    store.append_row(directory, {"task_key": "d", "run_id": "4"})
    rows, problems = store.read_rows(directory)
    assert [row["task_key"] for row in rows] == ["a", "c", "a", "d"]
    assert len(problems) == 1 and "line 4" in problems[0]


@test
def test_store_archive(directory: pathlib.Path, node: bool) -> None:
    """The archive holds every file of the label with matching checksums, and its .sha256.

    It leaves out the temporary file of an interrupted ``write_atomic``, and the label's lock, but
    keeps a file named ``SHA256SUMS`` below the top level.  ``open_archive`` unpacks it, and refuses
    it once its files no longer match its inner list, or the archive its ``.sha256``, or once its
    ``.sha256`` or inner ``SHA256SUMS`` is missing.
    """
    label_dir = directory / "label"
    (label_dir / "logs").mkdir(parents=True)
    store.append_row(label_dir, {"task_key": "a"})
    (label_dir / "logs" / "a.txt").write_text("log")
    (label_dir / "logs" / "SHA256SUMS").write_text("sums")
    (label_dir / ".manifest.json.123.tmp").write_text("{")
    (label_dir / store.LOCK_FILE).write_text("")
    archive = store.make_archive(label_dir, "label-host-utc")
    outer = (archive.parent / (archive.name + ".sha256")).read_text()
    assert outer == f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n", outer
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
        sums_file = tar.extractfile("label-host-utc/SHA256SUMS")
        assert sums_file is not None
        sums = sums_file.read().decode()
    listed = ("logs/SHA256SUMS", "logs/a.txt", "runs.jsonl")
    expected = {f"label-host-utc/{name}" for name in (*listed, "SHA256SUMS")}
    assert names == expected, names
    assert sums == "".join(
        f"{hashlib.sha256((label_dir / name).read_bytes()).hexdigest()}  {name}\n"
        for name in listed
    ), sums
    store.make_archive(label_dir, "second")  # earlier archives never enter a later one
    with tarfile.open(label_dir / "archive" / "second.tar.gz") as tar:
        assert not any("archive/" in name for name in tar.getnames())

    # open_archive unpacks an archive that matches its checksums, and removes it after the block
    with store.open_archive(archive) as unpacked:
        assert (unpacked / "logs" / "a.txt").read_text() == "log"
    assert not unpacked.exists()

    # it refuses an archive with a file that its inner SHA256SUMS does not list, here a nested
    # SHA256SUMS, with a file that does not match its digest there, or without an inner SHA256SUMS,
    # even if the archive matches its .sha256; and an archive whose .sha256 does not match it or is
    # missing
    with tarfile.open(archive) as tar:
        contents = {}
        for member in tar.getmembers():
            file = tar.extractfile(member)
            assert file is not None
            contents[member.name] = file.read()
    outer_sums = archive.parent / (archive.name + ".sha256")
    inner_sums = "label-host-utc/SHA256SUMS"
    changes = (
        (contents | {"label-host-utc/extra/SHA256SUMS": b""}, "does not match its inner"),
        (contents | {"label-host-utc/logs/a.txt": b"changed"}, "does not match its inner"),
        ({name: contents[name] for name in contents if name != inner_sums}, "has no inner"),
    )
    for changed, message in changes:
        with tarfile.open(archive, "w:gz") as tar:
            for name, data in changed.items():
                info = tarfile.TarInfo(name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        outer_sums.write_text(f"{digest}  {archive.name}\n")
        error = acceptance.raises(SystemExit, lambda: store.open_archive(archive).__enter__())
        assert f"{message} SHA256SUMS" in str(error.code), (message, error.code)
    outer_sums.write_text(f"{'0' * 64}  {archive.name}\n")
    error = acceptance.raises(SystemExit, lambda: store.open_archive(archive).__enter__())
    assert f"does not match {outer_sums.name}" in str(error.code), error.code
    outer_sums.unlink()
    error = acceptance.raises(SystemExit, lambda: store.open_archive(archive).__enter__())
    assert f"{outer_sums.name} is missing" in str(error.code), error.code


####################################################################################################
# runner


@test
def test_runner_ok(directory: pathlib.Path, node: bool) -> None:
    """A run that ends within its limit is ok, with T, CPU time, memory, and its result."""
    row = _run(directory, _task("ok", [["sleep", 0.3], ["value", 7], ["partial", 3]]), node)
    assert row["status"] == "ok" and row["result"]["value"] == 7, row
    assert 0.3 <= row["T"] < 2 and row["T_cpu"] >= 0 and row["peak_rss_gib"] > 0
    assert row["partial"] == [{"index": 0}, {"index": 1}, {"index": 2}]
    assert row["post_status"] is None and row["error"] is None
    assert not row["compiled_in_timing"] and row["discarded_run"] is None, row
    assert (directory / "logs" / f"{row['run_id']}.txt").exists()

    # a task with a final row is not run again
    before = (directory / store.RUNS_FILE).read_bytes()
    subprocess.run(plugin.run_task_command(directory, "ok", local=not node), check=True)
    assert (directory / store.RUNS_FILE).read_bytes() == before


@test
def test_runner_timeout(directory: pathlib.Path, node: bool) -> None:
    """A run past its limit is killed with its grandchildren, and is a timeout with T >= L."""
    start = time.monotonic()
    row = _run(directory, _task("slow", [["spawn", 47.25]], limit_s=2.0), node)
    assert row["status"] == "timeout" and 2.0 <= row["T"] < 4 and row["result"] == {}, row
    assert time.monotonic() - start < 15
    assert not _processes_matching("time.sleep(47.25)")


@test
def test_runner_orphan(directory: pathlib.Path, node: bool) -> None:
    """A grandchild left behind by a run that timed out is killed too."""
    row = _run(directory, _task("orphan", [["orphan", 48.25], ["sleep", 30]], limit_s=2.0), node)
    assert row["status"] == "timeout", row
    assert not _processes_matching("time.sleep(48.25)")


@test
def test_runner_startup_cap(directory: pathlib.Path, node: bool) -> None:
    """A run that does not start timing within the start-up cap is an error."""
    row = _run(directory, _task("late", [["startup", 30]], startup_cap_s=2.0), node)
    assert row["status"] == "error" and "start-up cap" in row["error"] and row["T"] is None, row


@test
def test_runner_memout(directory: pathlib.Path, node: bool) -> None:
    """A run above its memory cap is killed and is a memout."""
    row = _run(directory, _task("big", [["alloc", 0.6]], mem_cap_gib=0.3), node)
    assert row["status"] == "memout" and row["peak_rss_gib"] > 0.3, row


@test
def test_runner_error(directory: pathlib.Path, node: bool) -> None:
    """A run whose measured call raises is an error with the exception's text."""
    row = _run(directory, _task("fail", [["fail"]]), node)
    assert row["status"] == "error" and "failed on purpose" in row["error"], row


@test
def test_runner_post_processing(directory: pathlib.Path, node: bool) -> None:
    """Post-processing has its own allowance, which marks only ``post_status``."""
    row = _run(directory, _task("post", [["value", 1], ["post", 0.2]], post_s=5.0), node)
    assert row["status"] == "ok" and row["post_status"] == "ok", row
    assert row["result"]["value"] == 1 and row["result"]["post"] == "done"
    (directory / "late").mkdir()
    task = _task("late", [["value", 2], ["post", 30]], post_s=2.0)
    row = _run(directory / "late", task, node)
    assert row["status"] == "ok" and row["post_status"] == "timeout", row
    assert row["result"]["value"] == 2 and "post" not in row["result"] and row["T"] < 2


@test
def test_runner_messages_at_exit(directory: pathlib.Path, node: bool) -> None:
    """Messages that the child writes just before it exits all reach the row."""
    for index in range(5):
        label_dir = directory / str(index)
        label_dir.mkdir()
        row = _run(label_dir, _task("burst", [["partial", 3000], ["value", index]]), node)
        assert row["status"] == "ok" and len(row["partial"]) == 3000, (index, row["status"])


@test
def test_runner_partial_before_kill(directory: pathlib.Path, node: bool) -> None:
    """A partial result and a compilation written just before a kill are kept; a result is not."""
    task = _task("killed", [], mem_cap_gib=0.1)
    watcher = runner._Watcher(
        directory, task, local=not node, job_slot=None, cpu=None, stop=runner._Stop()
    )
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    child = subprocess.Popen(command, start_new_session=node)
    watcher.root = child.pid
    watcher.group = os.getpgid(child.pid) if node else None
    read_fd, write_fd = os.pipe()

    # after the watcher's last read, the child writes its messages, and the run is a memout
    def poll() -> tuple[int, None]:
        for message in (
            {"type": "partial", "fields": {"index": 0}},
            {"type": "compiled", "function": "f"},
            {"type": "result", "fields": {}},
        ):
            os.write(write_fd, (json.dumps(message) + "\n").encode())
        return 2**30, None

    watcher._poll_resources = poll  # type: ignore[method-assign]
    try:
        reason, _ = watcher._watch(child, read_fd, time.monotonic())
        row = watcher._row(reason, None, "", "")
        assert row["status"] == "memout" and row["partial"] == [{"index": 0}], row
        assert row["compiled_in_timing"] and row["compiled_function"] == "f", row
        assert row["result"] == {}, row
    finally:
        child.kill()
        child.wait()
        os.close(read_fd)
        os.close(write_fd)


@test
def test_runner_timeout_before_memout(directory: pathlib.Path, node: bool) -> None:
    """A run whose T reached its limit is a timeout, even if a memory kill followed.

    So is a run that a memory poll killed during its measured call at or past the limit, as after
    a slow poll; before the limit, the kill is a memout.
    """

    def watcher() -> Any:
        task = _task("late", [], limit_s=1.0)
        return runner._Watcher(
            directory, task, local=not node, job_slot=None, cpu=None, stop=runner._Stop()
        )

    returned = watcher()
    returned._receive({"type": "measured", "T": 1.5, "T_cpu": 1.5})
    row = returned._row("memout", None, "", "")
    assert row["status"] == "timeout" and row["T"] == 1.5, row
    for killed_at, status, T in ((1.5, "timeout", 1.5), (0.5, "memout", None)):
        killed = watcher()
        killed.timing_started_at, killed.killed_at = 10.0, 10.0 + killed_at
        row = killed._row("memout", None, "", "")
        assert (row["status"], row["T"]) == (status, T), (killed_at, row)


@test
def test_runner_peak_after_run(directory: pathlib.Path, node: bool) -> None:
    """An ok run whose peak memory, found after the run, exceeds its cap is a memout.

    With post-processing, the peak that the child reports with its result decides the status, and a
    later peak marks only the post-processing.  The row's peak is the largest that the runner knows,
    and a memout that a poll found does not say that it was found after the run.  Without
    ``send_result``, a run killed after its final result, for not exiting within ``post_s``, is ok
    unless the peak that the child sends with that result exceeds the cap, and a run whose result
    did not arrive is an error.
    """

    def row(
        peak_gib: float, post: bool, result_peak_gib: float = 0.0, kill: str | None = None
    ) -> dict[str, Any]:
        task = _task("peak", [], mem_cap_gib=1.0)
        watcher = runner._Watcher(
            directory, task, local=not node, job_slot=None, cpu=None, stop=runner._Stop()
        )
        watcher._receive({"type": "measured", "T": 0.1, "T_cpu": 0.1})
        result_peak = int(result_peak_gib * 2**30)
        watcher._receive(
            {"type": "result", "fields": {}, "final": not post, "max_rss_bytes": result_peak}
        )
        if post and kill is None:
            watcher._receive({"type": "post", "fields": {}})
        watcher.peak_rss = int(peak_gib * 2**30)
        return watcher._row(kill, 0 if kill is None else -9, "", "")

    over = row(1.5, post=False)
    assert over["status"] == "memout" and "found after the run" in over["error"], over
    assert row(0.5, post=False)["status"] == "ok"
    late = row(1.5, post=True, result_peak_gib=0.5)
    assert late["status"] == "ok" and late["post_status"] == "memout", late
    assert "found after the run" in late["error"], late
    assert row(1.5, post=True, result_peak_gib=1.5)["status"] == "memout"

    # a peak that only the child reported, with post-processing killed: no rusage, and polls below
    killed = row(0.5, post=True, result_peak_gib=1.5, kill="post_timeout")
    assert killed["status"] == "memout" and killed["peak_rss_gib"] == 1.5, killed

    # without send_result: a run killed after its final result, for not exiting, is ok unless the
    # peak that the child sent with that result exceeds the cap, and one with no result is an error
    stalled = row(0.5, post=False, result_peak_gib=0.5, kill="post_timeout")
    assert stalled["status"] == "ok" and stalled["post_status"] is None, stalled
    stalled = row(0.5, post=False, result_peak_gib=1.5, kill="post_timeout")
    assert stalled["status"] == "memout" and stalled["peak_rss_gib"] == 1.5, stalled
    assert "found after the run" in stalled["error"], stalled
    silent = runner._Watcher(
        directory, _task("silent", []), local=not node, job_slot=None, cpu=None, stop=runner._Stop()
    )
    silent._receive({"type": "measured", "T": 0.1, "T_cpu": 0.1})
    silent_row = silent._row("post_timeout", -9, "", "")
    assert silent_row["status"] == "error" and "no result within" in silent_row["error"], silent_row
    sent: list[dict[str, Any]] = []
    channel = types.SimpleNamespace(timing_started=True, result_sent=False, send=sent.append)
    with mock.patch.object(runner, "_CHANNEL", channel):
        runner.child_main(_task("final", []), directory, lambda task, label_dir: {})
    assert sent[-1]["final"] and sent[-1]["max_rss_bytes"] > 0, sent

    # a poll that kills a run above its cap just after its final result
    polled = row(1.5, post=False, kill="memout")
    assert polled["status"] == "memout" and polled["error"] is None, polled


@test
def test_runner_missed_peak(directory: pathlib.Path, node: bool) -> None:
    """A measured call above its memory cap for a fraction of a second is a memout.

    The polls, a second apart, may miss the peak, but the peak that the child sends with its result,
    or that the runner reads when the child exits, does not.  The polls are blinded here, so that
    only these peaks can make the run a memout.
    """
    for steps in ([["flash", 0.6]], [["flash", 0.6], ["post", 0.1]]):
        label_dir = directory / str(len(steps))
        label_dir.mkdir()
        task = _task("flash", steps, mem_cap_gib=0.3, post_s=5.0)
        tasks.write_pass1(label_dir, [task])
        with mock.patch.object(runner._Watcher, "_poll_resources", lambda self: (0, None)):
            assert runner.run_task(label_dir, task, local=not node) == 0
        row = store.final_rows(label_dir)[0][task["key"]]
        assert row["status"] == "memout" and row["peak_rss_gib"] > 0.3, (steps, row)


@test
def test_runner_child_peak(directory: pathlib.Path, node: bool) -> None:
    """The peak of a process that the measured call waited for counts in the peak of its result.

    The polls are blinded, so that only the peak that the child sends with its result, which
    covers the processes that it waited for, can make the run a memout.
    """
    task = _task("child", [["child_flash", 0.6], ["post", 0.1]], mem_cap_gib=0.3, post_s=5.0)
    tasks.write_pass1(directory, [task])
    with mock.patch.object(runner._Watcher, "_poll_resources", lambda self: (0, None)):
        assert runner.run_task(directory, task, local=not node) == 0
    row = store.final_rows(directory)[0][task["key"]]
    assert row["status"] == "memout" and row["peak_rss_gib"] > 0.5, row


@test
def test_runner_compile_retry(directory: pathlib.Path, node: bool) -> None:
    """A run whose measured call compiles is discarded; its retry, which loads the code, is kept."""
    with mock.patch.dict(os.environ, {"NUMBA_CACHE_DIR": str(directory / "numba")}):
        row = _run(directory, _task("compiles", [["compile", True], ["value", 3]]), node)
    assert row["status"] == "ok" and row["result"]["value"] == 3, row
    assert not row["compiled_in_timing"] and row["compiled_function"] is None, row
    discarded = row["discarded_run"]
    assert discarded["status"] == "ok" and discarded["run_id"] != row["run_id"], row
    assert discarded["compiled_function"].endswith("selftest_plugin._add_one"), discarded
    assert len(store.read_rows(directory)[0]) == 1
    assert len(list((directory / "logs").iterdir())) == 2


@test
def test_runner_compile_every_time(directory: pathlib.Path, node: bool) -> None:
    """A task whose retry compiles too keeps the retry's row, flagged ``compiled_in_timing``."""
    with mock.patch.dict(os.environ, {"NUMBA_CACHE_DIR": str(directory / "numba")}):
        row = _run(directory, _task("uncached", [["compile", False], ["value", 4]]), node)
    assert row["status"] == "ok" and row["result"]["value"] == 4, row
    assert row["compiled_in_timing"] and row["compiled_function"].endswith("_add_one"), row
    assert row["discarded_run"]["compiled_function"] == row["compiled_function"], row
    assert len(store.read_rows(directory)[0]) == 1
    assert len(list((directory / "logs").iterdir())) == 2


@test
def test_runner_compile_on_thread(directory: pathlib.Path, node: bool) -> None:
    """A compilation on a thread other than the main one is reported too."""
    task = _task("threaded", [["compile_on_thread", True], ["value", 2]])
    with mock.patch.dict(os.environ, {"NUMBA_CACHE_DIR": str(directory / "numba")}):
        row = _run(directory, task, node)
    assert row["status"] == "ok" and not row["compiled_in_timing"], row
    assert row["discarded_run"]["compiled_function"].endswith("_add_one"), row


@test
def test_runner_compile_vectorize(directory: pathlib.Path, node: bool) -> None:
    """An uncached ``numba.vectorize`` ufunc is reported, and the retry recompiles it, flagged."""
    with mock.patch.dict(os.environ, {"NUMBA_CACHE_DIR": str(directory / "numba")}):
        row = _run(directory, _task("ufunc", [["vectorize"], ["value", 1]]), node)
    assert row["status"] == "ok" and row["compiled_in_timing"], row
    assert row["compiled_function"].endswith("_add_one"), row


@test
def test_runner_compile_untimed(directory: pathlib.Path, node: bool) -> None:
    """Compilations before and after the measured call, even uncached, are not reported."""
    task = _task("untimed", [["compile_untimed"], ["value", 6], ["post", 0.5]])
    with mock.patch.dict(os.environ, {"NUMBA_CACHE_DIR": str(directory / "numba")}):
        row = _run(directory, task, node)
    assert row["status"] == "ok" and row["post_status"] == "ok", row
    assert not row["compiled_in_timing"] and row["discarded_run"] is None, row


@test
def test_runner_compile_then_timeout(directory: pathlib.Path, node: bool) -> None:
    """A run killed at its limit after it compiled is discarded too, and the task retried."""
    task = _task("slow", [["compile", True], ["sleep", 30]], limit_s=10.0)
    with mock.patch.dict(os.environ, {"NUMBA_CACHE_DIR": str(directory / "numba")}):
        row = _run(directory, task, node)
    assert row["status"] == "timeout" and not row["compiled_in_timing"], row
    discarded = row["discarded_run"]
    assert discarded["status"] == "timeout" and discarded["T"] >= 10.0, row
    assert discarded["compiled_function"].endswith("_add_one"), discarded


@test
def test_runner_compile_then_killed(directory: pathlib.Path, node: bool) -> None:
    """A run killed by a signal after it compiled is discarded too, and the task retried."""
    with mock.patch.dict(os.environ, {"NUMBA_CACHE_DIR": str(directory / "numba")}):
        row = _run(directory, _task("dies", [["compile", True], ["die"]]), node)
    assert row["status"] == "error" and not row["compiled_in_timing"], row
    discarded = row["discarded_run"]
    assert discarded["status"] == "error" and discarded["error"], row
    assert discarded["compiled_function"].endswith("_add_one"), discarded


@test
def test_runner_compile_then_stopped(directory: pathlib.Path, node: bool) -> None:
    """A task stopped by SIGTERM after it compiled has no row, and resumes without compiling."""
    task = _task("stopped", [["compile", True], ["sleep", 3], ["value", 5]])
    tasks.write_pass1(directory, [task])
    with mock.patch.dict(os.environ, {"NUMBA_CACHE_DIR": str(directory / "numba")}):
        process = subprocess.Popen(plugin.run_task_command(directory, "stopped", local=not node))
        try:
            # numba writes a function's compiled code, after its index, once it has compiled it
            _wait_for(lambda: any((directory / "numba").rglob("*.nbc")))
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=40) == 2
        finally:
            _kill(process)
        assert store.read_rows(directory) == ([], [])
        row = _run(directory, task, node)
    assert row["status"] == "ok" and row["result"]["value"] == 5, row
    assert not row["compiled_in_timing"] and row["discarded_run"] is None, row
    assert len(store.read_rows(directory)[0]) == 1


@test
def test_runner_stop_during_retry(directory: pathlib.Path, node: bool) -> None:
    """On SIGTERM during the retry, the runner kills it, writes no row, and exits with status 2."""
    tasks.write_pass1(directory, [_task("retried", [["compile", False], ["sleep", 3]])])
    with mock.patch.dict(os.environ, {"NUMBA_CACHE_DIR": str(directory / "numba")}):
        process = subprocess.Popen(plugin.run_task_command(directory, "retried", local=not node))
        try:
            # the retry has started once it has a log of its own
            _wait_for(lambda: len(list((directory / "logs").glob("*"))) == 2)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=40) == 2
        finally:
            _kill(process)
    assert store.read_rows(directory) == ([], [])
    assert not (directory / "active" / "retried").exists()
    assert not _processes_matching(str(directory)), _processes_matching(str(directory))


@test
def test_runner_reparented(directory: pathlib.Path, node: bool) -> None:
    """Locally, a run's processes include a seen grandchild whose parent exited, and its children.

    A process that has reused the id of a seen process, the run's own included, is not one of them,
    nor are its children.
    """
    watcher = runner._Watcher(
        directory, _task("tree", []), local=True, job_slot=None, cpu=None, stop=runner._Stop()
    )
    watcher.root = 100
    watcher.seen = {100: "root", 101: "grandchild"}
    table = {
        100: runner.Process(1, 100, 1, None, "root"),
        101: runner.Process(1, 100, 2**30, None, "grandchild"),
        102: runner.Process(101, 100, 1, None, "its child"),
    }
    assert watcher._members(table) == [100, 101, 102], watcher._members(table)
    table = {
        100: runner.Process(1, 100, 1, None, "root"),
        101: runner.Process(1, 1, 1, None, "a later process"),
    }
    assert watcher._members(table) == [100], watcher._members(table)
    table = {
        100: runner.Process(1, 100, 1, None, "a later root"),
        103: runner.Process(100, 100, 1, None, "its child"),
    }
    assert watcher._members(table) == [], watcher._members(table)


@test
def test_runner_registration_failure(directory: pathlib.Path, node: bool) -> None:
    """A runner that fails to record its run after starting it kills the run before failing."""
    watcher = runner._Watcher(
        directory,
        _task("unrecorded", []),
        local=not node,
        job_slot=None,
        cpu=None,
        stop=runner._Stop(),
    )

    def failing_ps(pid: int) -> str | None:
        raise RuntimeError("ps failed")

    command = [sys.executable, "-c", "import time; time.sleep(56.25)"]
    with mock.patch.object(runner, "process_start", failing_ps):
        acceptance.raises(RuntimeError, lambda: watcher.run(command))
    survivors = _processes_matching("time.sleep(56.25)")
    if survivors:  # the run, still this process's unreaped child, is stopped before failing
        os.kill(watcher.root, signal.SIGKILL)
        os.waitpid(watcher.root, 0)
    assert not survivors, "the unrecorded run survived"


@test
def test_runner_survivors(directory: pathlib.Path, node: bool) -> None:
    """A process that survives the grace is listed, and --cleanup kills it after its runner ends.

    The kept record names the survivor, here a grandchild left after the run's own process was
    killed, with its start time, so that --cleanup finds it once that process and the runner are
    gone.
    """
    task = _task("survives", [], startup_cap_s=3.0, grace_s=1.0)
    watcher = runner._Watcher(
        directory, task, local=not node, job_slot=None, cpu=None, stop=runner._Stop()
    )

    def kill_root_only() -> None:  # as for a grandchild in D state
        with contextlib.suppress(ProcessLookupError):
            os.kill(watcher.root, signal.SIGKILL)

    watcher._kill = kill_root_only  # type: ignore[method-assign]
    grandchild = "import time; time.sleep(57.75)"
    script = (
        f"import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', {grandchild!r}]); "
        "time.sleep(57.25)"
    )
    survivors: list[list[Any]] = []
    try:
        row = watcher.run([sys.executable, "-c", script])
        assert row is not None and row["status"] == "error", row
        path = directory / "active" / row["run_id"]
        record = json.loads(path.read_text())
        survivors = record["survivors"]
        ((survivor, _),) = survivors
        assert survivor != watcher.root and f"{survivor} (" in row["error"], (record, row)
        assert "time.sleep(57.75)" in row["error"], row
        assert not (directory / "active" / task["key"]).exists()

        # once the runner has exited too, the survivor is still live, and --cleanup kills it
        record |= {"runner_pid": record["pid"], "runner_start": record["pid_start"]}
        path.write_text(json.dumps(record))
        assert run_node.live_runs(directory) == [row["run_id"]]
        with contextlib.redirect_stdout(io.StringIO()):
            run_node.cleanup(directory)
        time.sleep(1)
        assert not _processes_matching("time.sleep(57.75)"), "a survivor outlived --cleanup"
        assert run_node.live_runs(directory) == []
    finally:
        recorded = [(watcher.root, watcher.seen.get(watcher.root))] + [
            (pid, start) for pid, start in survivors
        ]
        for pid, start in recorded:
            if pid > 0 and start is not None and runner.process_start(pid) == start:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
        if watcher.root > 0:
            with contextlib.suppress(ChildProcessError):
                os.waitpid(watcher.root, os.WNOHANG)


@test
def test_runner_survivors_unrecorded(directory: pathlib.Path, node: bool) -> None:
    """A runner that cannot write the record of a run's survivors keeps the task's record.

    The write fails as on a full disk, and the runner fails, so that the task's record in
    ``active/`` still shows that the run may live.
    """
    task = _task("unrecorded", [["spawn", 56.75]], limit_s=3.0, grace_s=1.0)
    tasks.write_pass1(directory, [task])

    def kill_root_only(watcher: Any) -> None:  # as for a grandchild in D state
        with contextlib.suppress(ProcessLookupError):
            os.kill(watcher.root, signal.SIGKILL)

    write_atomic = store.write_atomic
    survivors: list[list[Any]] = []

    def failing_write(path: pathlib.Path, text: str, *args: Any) -> None:
        if path.name == task["key"]:
            write_atomic(path, text, *args)
            return
        survivors.extend(json.loads(text)["survivors"])
        raise OSError("no space left on the device")

    try:
        with (
            mock.patch.object(runner._Watcher, "_kill", kill_root_only),
            mock.patch.object(store, "write_atomic", failing_write),
        ):
            acceptance.raises(OSError, lambda: runner.run_task(directory, task, local=not node))
        assert survivors, "no process survived the grace"
        assert (directory / "active" / task["key"]).exists()
    finally:
        for pid, start in survivors:
            if start is not None and runner.process_start(pid) == start:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)


@test
def test_local_runs_reject_compiled(directory: pathlib.Path, node: bool) -> None:
    """``local_runs.run_locally`` fails if a row with status ok has ``compiled_in_timing`` set.

    A flagged row of another status, here a timeout, is accepted.
    """
    with mock.patch.dict(os.environ, {"NUMBA_CACHE_DIR": str(directory / "numba")}):
        error = acceptance.raises(
            AssertionError,
            lambda: local_runs.run_locally(
                directory / "label", [_task("uncached", [["compile", False]])]
            ),
        )
        assert "uncached" in str(error), error
        task = _task("slow", [["compile", False], ["sleep", 30]], limit_s=3.0)
        row = local_runs.run_locally(directory / "timeout", [task])["slow"]
    assert row["status"] == "timeout" and row["compiled_in_timing"], row


@test
def test_local_runs_interrupted(directory: pathlib.Path, node: bool) -> None:
    """``local_runs.run_locally`` stops a task's run if an exception interrupts it meanwhile."""

    class Interrupted(Exception):
        """The exception of a caller whose time limit ran out."""

    deadline = time.monotonic() + 30

    def interrupt(signum: int, frame: Any) -> None:
        # interrupt only once the run's sleep has started, so that there is a run to stop
        if not _processes_matching("time.sleep(53.25)"):
            assert time.monotonic() < deadline, "the run's sleep did not start"
            signal.setitimer(signal.ITIMER_REAL, 0.5)
            return
        raise Interrupted

    previous = signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    task = _task("stop", [["spawn", 53.25]], limit_s=60.0)
    try:
        acceptance.raises(Interrupted, lambda: local_runs.run_locally(directory, [task]))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert not _processes_matching("time.sleep(53.25)")


@test
def test_runner_sigterm(directory: pathlib.Path, node: bool) -> None:
    """On SIGTERM the runner kills its run, writes no row, and exits with status 2."""
    grandchild = directory / "grandchild"
    task = _task("stop", [["spawn", 49.25, str(grandchild)]], limit_s=60.0)
    tasks.write_pass1(directory, [task])
    process = subprocess.Popen(plugin.run_task_command(directory, "stop", local=not node))
    pid = _started(grandchild)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=40) == 2
    assert not (directory / "active" / "stop").exists()
    assert store.read_rows(directory) == ([], [])
    _wait_for(lambda: not _running(pid), timeout=10)


@test
def test_runner_late_signal(directory: pathlib.Path, node: bool) -> None:
    """A signal after a run's process has ended discards the run until the runner's last check.

    A SIGTERM after the runner has kept the tail of the run's log, but before its last check for a
    signal, makes ``run_task`` return 2 without a row.  One after that check, whether before the
    runner builds the last run's row or after, is ignored: the runner appends the row and returns 0.
    """
    late, unbuilt, built = (_task(key, [["value", 1]]) for key in ("late", "unbuilt", "built"))
    tasks.write_pass1(directory, [late, unbuilt, built])
    keep_tail, make_row, run = runner._keep_tail, runner._Watcher._row, runner._Watcher.run

    def signal_after_tail(path: pathlib.Path) -> None:
        keep_tail(path)
        signal.raise_signal(signal.SIGTERM)

    def signal_before_row(watcher: Any, *args: Any) -> dict[str, Any]:
        signal.raise_signal(signal.SIGTERM)
        return make_row(watcher, *args)

    def signal_after_row(watcher: Any, command: list[str]) -> dict[str, Any] | None:
        row = run(watcher, command)
        signal.raise_signal(signal.SIGTERM)
        return row

    with mock.patch.object(runner, "_keep_tail", signal_after_tail):
        assert runner.run_task(directory, late, local=not node) == 2
    with mock.patch.object(runner._Watcher, "_row", signal_before_row):
        assert runner.run_task(directory, unbuilt, local=not node) == 0
    with mock.patch.object(runner._Watcher, "run", signal_after_row):
        assert runner.run_task(directory, built, local=not node) == 0
    rows, problems = store.read_rows(directory)
    assert [row["task_key"] for row in rows] == ["unbuilt", "built"] and not problems, rows


@test
def test_runner_failure(directory: pathlib.Path, node: bool) -> None:
    """A runner that fails kills its run, removes its active file, and writes no row."""
    tasks.write_pass1(directory, [_task("broken", [["garbage", 52.25]], limit_s=60.0)])
    result = subprocess.run(
        plugin.run_task_command(directory, "broken", local=not node),
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert not (directory / "active" / "broken").exists()
    assert store.read_rows(directory) == ([], [])
    time.sleep(1)
    assert not _processes_matching("broken"), "the failed runner's run survived"


@test
def test_runner_failed_process_table(directory: pathlib.Path, node: bool) -> None:
    """A runner whose process table fails kills its run's own process and keeps the run's record.

    The runner cannot tell whether other processes of the run live, so the record stays in
    ``active/``: on the node, a start and --cleanup find members of the run's group that survive.
    """
    task = _task("blind", [["sleep", 54.75]], limit_s=60.0)
    tasks.write_pass1(directory, [task])
    process_table = runner.process_table
    calls: list[None] = []

    def failing_table() -> dict[int, Any]:  # works while the runner records the run
        calls.append(None)
        if len(calls) > 2:
            raise RuntimeError("ps failed")
        return process_table()

    def reaped(pid: int) -> bool:  # the run's process is this process's child
        try:
            return os.waitpid(pid, os.WNOHANG)[0] == pid
        except ChildProcessError:
            return True

    record = directory / "active" / task["key"]
    pid = None
    try:
        with mock.patch.object(runner, "process_table", failing_table):
            acceptance.raises(
                RuntimeError, lambda: runner.run_task(directory, task, local=not node)
            )
        pid = json.loads(record.read_text())["pid"]
        _wait_for(functools.partial(reaped, pid), timeout=10)
        pid = None
    finally:
        if pid is not None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
            reaped(pid)


@test
def test_runner_active_across_retry(directory: pathlib.Path, node: bool) -> None:
    """A task's active file stays from its first run, through its retry, until its row is in."""
    task = _task("twice", [["compile", False], ["value", 1]])
    tasks.write_pass1(directory, [task])
    active = directory / "active" / task["key"]
    after_runs: list[bool] = []
    run = runner._Watcher.run

    def watched_run(watcher: Any, command: list[str]) -> dict[str, Any] | None:
        row = run(watcher, command)
        after_runs.append(active.exists())
        return row

    with (
        mock.patch.dict(os.environ, {"NUMBA_CACHE_DIR": str(directory / "numba")}),
        mock.patch.object(runner._Watcher, "run", watched_run),
    ):
        assert runner.run_task(directory, task, local=not node) == 0
    assert after_runs == [True, True] and not active.exists(), after_runs
    assert len(store.read_rows(directory)[0]) == 1


@test
def test_runner_channel_json(directory: pathlib.Path, node: bool) -> None:
    """The child sends a set as a sorted list, and refuses an object that JSON lacks."""
    assert list({1, 8}) != [1, 8], "the set must not iterate in sorted order"
    read_fd, write_fd = os.pipe()
    channel = runner._Channel()
    channel.fd = write_fd
    try:
        channel.send({"value": {1, 8}})
        acceptance.raises(TypeError, lambda: channel.send({"value": object()}))
    finally:
        os.close(write_fd)
    with os.fdopen(read_fd) as file:
        assert file.read() == '{"value":[1,8]}\n'


####################################################################################################
# task lists, GNU parallel, run_node, and cleanup


@test
def test_tasks_write_once(directory: pathlib.Path, node: bool) -> None:
    """Task files are written once, in a fixed shuffled order, and never changed."""
    task_list = [_task(f"t{i}", [["value", i]]) for i in range(20)]
    order = tasks.write_pass1(directory, task_list)
    assert sorted(order) == sorted(task["key"] for task in task_list)
    assert order != [task["key"] for task in task_list]  # shuffled
    assert tasks.write_pass1(directory, task_list) == order  # the same again is fine
    acceptance.raises(FileExistsError, lambda: tasks.write_pass1(directory, task_list[:-1]))
    assert tasks.pass_keys(directory, "1") == order


@test
def test_tasks_ceiling_and_cores(directory: pathlib.Path, node: bool) -> None:
    """The ceiling, with repeats and retries, needs an approval; job slots get a CPU per core."""
    task_list = [_task(f"t{i}", [], limit_s=1800.0) for i in range(10)]
    task_list += [
        _task(f"u{i}", [], limit_s=1800.0, startup_cap_s=60.0, post_s=600.0) for i in range(5)
    ]
    ceiling = tasks.ceiling_hours(task_list, lambda task: 3 * (120 + task["limit_s"] + 30))
    # each task costs S + 2 W, with W its worst case of three runs, and S the sum of its start-up
    # cap, limit, post-processing allowance, and grace of 5 s, with the runner's defaults
    # STARTUP_CAP_S = 120 and FINISH_S = 30
    default_cost = (120 + 1800 + 30 + 5) + 2 * 3 * 1950
    own_cost = (60 + 1800 + 600 + 5) + 2 * 3 * 1950
    assert abs(ceiling - (10 * default_cost + 5 * own_cost) / 3600) < 1e-9
    for approved in (None, ceiling - 1):
        acceptance.raises(
            tasks.ApprovalError, functools.partial(tasks.check_approval, ceiling, approved)
        )
    tasks.check_approval(ceiling, ceiling)

    # 8 physical cores with SMT siblings: CPUs 0-7 on cores 0-7, CPUs 8-15 on the same cores
    lscpu = "# CPU,Core,Socket\n" + "".join(f"{cpu},{cpu % 8},0\n" for cpu in range(16))
    assert tasks.choose_cpus(lscpu, 128 * 2**30, 8) == ([2, 3, 4, 5, 6, 7], [0, 1, 8, 9])
    assert tasks.choose_cpus(lscpu, 48 * 2**30, 8)[0] == [2, 3, 4, 5]  # (48 - 16) / 8 = 4 jobs
    assert tasks.choose_cpus(lscpu, 128 * 2**30, 8, jobs=2)[0] == [2, 3]

    # a task whose memory cap exceeds a job slot's is refused
    tasks.check_memory_caps([_task("fits", [], mem_cap_gib=8.0)], 8.0)
    acceptance.raises(
        ValueError, lambda: tasks.check_memory_caps([_task("too_big", [], mem_cap_gib=9.0)], 8.0)
    )


@test
def test_stand_in_worst_case(directory: pathlib.Path, node: bool) -> None:
    """The stand-in's worst case is three runs of tasks.run_cost_s, post-processing included."""
    task = _task("long_post", [], post_s=600.0)
    assert selftest_plugin.worst_case_s(task) == 3 * tasks.run_cost_s(task) == 3 * 735.0


@test
def test_tasks_refused(directory: pathlib.Path, node: bool) -> None:
    """A task with a limit that cannot be enforced, or a reserved key, is refused when written.

    So is a non-finite approval.  Each limit must be finite and positive, and the grace short enough
    for GNU parallel's signal sequence.
    """
    nan, inf = float("nan"), float("inf")
    for bad in (
        {"limit_s": nan},
        {"mem_cap_gib": inf},
        {"startup_cap_s": 0.0},
        {"post_s": -1.0},
        {"grace_s": 40.0},
        {"key": "_parallel"},
    ):
        acceptance.raises(
            ValueError, functools.partial(tasks.write_pass1, directory, [_task("bad", []) | bad])
        )
    assert not (directory / tasks.TASKS_FILE).exists()
    for ceiling, approved in ((1.0, nan), (1.0, inf), (nan, 10.0)):
        acceptance.raises(
            tasks.ApprovalError, functools.partial(tasks.check_approval, ceiling, approved)
        )


@test
def test_tasks_pass2_keys(directory: pathlib.Path, node: bool) -> None:
    """Pass 2 refuses repeat keys that are not unique, or that are keys of pass 1."""
    tasks.write_pass1(directory, [_task("a", []), _task("b", [])])
    final = {key: {"status": "ok", "T": 0.1} for key in ("a", "b")}

    def same_keys(task: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
        return [dict(task, key="repeat")]

    def pass1_key(task: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
        return [dict(task, key="a")] if task["key"] == "b" else []

    for policy in (same_keys, pass1_key):
        acceptance.raises(
            ValueError, functools.partial(tasks.write_pass2, directory, final, policy)
        )
    assert not (directory / tasks.PASS2_TASKS_FILE).exists()


@test
def test_parallel_pinning_and_pass2(directory: pathlib.Path, node: bool) -> None:
    """Under GNU parallel, each run is pinned to its slot's CPU, and pass 2 has the repeats."""
    _require_parallel()
    runner_cpus, cpus = ([_node_cpus(3)[0]], _node_cpus(3)[1:]) if node else ([], [5, 9])
    task_list = [_task(f"t{i}", [["sleep", 0.3 if i % 2 else 1.2]]) for i in range(6)]
    tasks.write_pass1(directory, task_list)
    tasks.write_cores(directory, cpus)
    with _affinity(runner_cpus):
        assert _parallel(directory, "1", 2, node).wait(timeout=300) == 0

    # each run's CPU is its slot's, and no two concurrent runs share a CPU
    final, problems = store.final_rows(directory)
    assert len(final) == 6 and not problems
    for key, row in final.items():
        assert row["pinned_cpu"] == cpus[row["job_slot"] - 1], (key, row["job_slot"])
        if node:
            assert row["affinity"] == [row["pinned_cpu"]], (key, row["affinity"])
            assert row["runner_affinity"] == runner_cpus, (key, row["runner_affinity"])
    rows = list(final.values())
    for i, first in enumerate(rows):
        for second in rows[i + 1 :]:
            (a0, a1), (b0, b1) = first["result"]["wall"], second["result"]["wall"]
            overlap = a0 < b1 and b0 < a1
            assert not overlap or first["pinned_cpu"] != second["pinned_cpu"]

    # pass 2 holds exactly the repeats of the short runs, and a resume reruns nothing
    def bad_repeats(task: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
        return [dict(repeat, grace_s=40.0) for repeat in selftest_plugin.repeat_tasks(task, row)]

    acceptance.raises(ValueError, lambda: tasks.write_pass2(directory, final, bad_repeats))
    assert not (directory / tasks.PASS2_TASKS_FILE).exists()
    assert not tasks.pass_file(directory, "2").exists()
    keys = tasks.write_pass2(directory, final, selftest_plugin.repeat_tasks)
    short = [key for key in tasks.pass_keys(directory, "1") if final[key]["T"] < 1]
    assert keys == [f"{key}.r{i}" for key in short for i in (1, 2)], keys
    before = (directory / store.RUNS_FILE).read_bytes()
    assert _parallel(directory, "1", 2, node).wait(timeout=300) == 0
    assert (directory / store.RUNS_FILE).read_bytes() == before


@test
def test_parallel_stop_and_resume(directory: pathlib.Path, node: bool) -> None:
    """SIGTERM stops every run without rows; a resume runs only the tasks without a final row."""
    _require_parallel()
    task_list = [_task("quick", [["sleep", 0.1]])] + [
        _task(f"slow{i}", [["spawn", 50.25 + i], ["value", i]], limit_s=60.0) for i in range(3)
    ]
    tasks.write_pass1(directory, task_list)
    tasks.write_cores(directory, _node_cpus(2) if node else [0, 1])
    process = _parallel(directory, "1", 2, node)
    _wait_for(lambda: len(list((directory / "active").glob("slow*"))) >= 2)
    time.sleep(1)
    stopped = {path.name for path in (directory / "active").iterdir()}
    process.send_signal(signal.SIGTERM)
    process.wait(timeout=60)
    time.sleep(1)
    assert not any(_processes_matching(f"time.sleep({50.25 + i})") for i in range(3))
    finished, _ = store.final_rows(directory)
    assert len(stopped) == 2 and not stopped & set(finished), (stopped, sorted(finished))
    assert not list((directory / "active").iterdir())

    # resume with short sleeps in place: only the tasks without a final row run again
    for task in task_list[1:]:
        task["input"] = [["sleep", 0.1]]
    (directory / tasks.TASKS_FILE).write_text(json.dumps(task_list))
    before = set(finished)
    assert _parallel(directory, "1", 2, node).wait(timeout=300) == 0
    final, problems = store.final_rows(directory)
    assert set(final) == {task["key"] for task in task_list} and not problems
    assert all(final[key] == finished[key] for key in before)


@test
def test_run_node_local(directory: pathlib.Path, node: bool) -> None:
    """run_node.py refuses without approval, runs both passes, writes an archive, and reports.

    The refused start records itself in ``starts.jsonl`` too, as does a start whose acceptance tests
    fail, before it runs them.  ``--status`` and ``--cleanup`` name the label directory that they
    read, and ``--cleanup`` says when it has nothing to clean up.
    """
    command = [
        sys.executable,
        str(RUN_NODE),
        "selftest",
        "--local",
        "--results-dir",
        str(directory),
    ]
    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode != 0 and "No task file was written" in refused.stderr, refused
    assert not list(directory.rglob(tasks.TASKS_FILE))
    subprocess.run(command + ["--approved-cpu-hours", "10"], check=True, capture_output=True)
    (label_dir,) = [path.parent for path in directory.rglob(tasks.TASKS_FILE)]
    starts = [json.loads(line) for line in (label_dir / store.STARTS_FILE).read_text().splitlines()]
    assert ["jobs" in start for start in starts] == [False, True], starts
    final, problems = store.final_rows(label_dir)
    pass1, pass2 = tasks.pass_keys(label_dir, "1"), tasks.pass_keys(label_dir, "2")
    assert len(pass1) == 3 and len(pass2) == 6 and set(final) == set(pass1 + pass2), problems
    assert list((label_dir / "archive").glob("*.tar.gz"))
    status = subprocess.run(command + ["--status"], capture_output=True, text=True, check=True)
    assert f"label directory: {label_dir.resolve()}\npass 1: 3 tasks" in status.stdout, (
        status.stdout
    )
    cleanup = subprocess.run(command + ["--cleanup"], capture_output=True, text=True, check=True)
    assert f"label directory: {label_dir.resolve()}\n" in cleanup.stdout, cleanup.stdout
    assert "nothing to clean up" in cleanup.stdout, cleanup.stdout

    # a start records itself before its acceptance tests, so even one whose tests fail does
    module = plugin.load_experiment("selftest")
    results = directory / "failed"
    argv = ["run_node.py", "selftest", "--local", "--subset", "acceptance"]
    with (
        mock.patch.object(sys, "argv", argv + ["--results-dir", str(results)]),
        mock.patch.object(module, "acceptance", lambda node, results_dir: False),
        mock.patch.object(
            run_node, "lock_label", lambda label_dir: None
        ),  # no lock left in this process
        contextlib.redirect_stdout(io.StringIO()),
    ):
        error = acceptance.raises(SystemExit, run_node.main)
        assert "The acceptance tests failed" in str(error.code), error.code
    (starts_file,) = results.rglob(store.STARTS_FILE)
    (start,) = [json.loads(line) for line in starts_file.read_text().splitlines()]
    assert start["subset"] == "acceptance", start


@test
def test_cleanup(directory: pathlib.Path, node: bool) -> None:
    """After a runner is killed hard, its run is live until --cleanup kills it.

    ``--status`` says which records' runs are alive and which ended.
    """
    grandchild = directory / "grandchild"
    task = _task("hard", [["spawn", 51.25, str(grandchild)]], limit_s=60.0)
    tasks.write_pass1(directory, [task])
    process = subprocess.Popen(plugin.run_task_command(directory, "hard", local=not node))
    pid = _started(grandchild)
    _wait_for((directory / "active" / "hard").exists, timeout=120)  # written atomically
    record = json.loads((directory / "active" / "hard").read_text())
    assert record["host"] == socket.gethostname(), record
    process.kill()
    process.wait()
    assert _running(pid)
    assert run_node.live_runs(directory) == ["hard"]
    ended = record | {"pid_start": "another start", "runner_start": "another start"}
    (directory / "active" / "ended").write_text(json.dumps(ended))
    with contextlib.redirect_stdout(io.StringIO()) as output:
        run_node.print_status(directory)
    assert "\n  ended: ended (--cleanup" in output.getvalue(), output.getvalue()
    assert "\n  hard: alive\n" in output.getvalue(), output.getvalue()
    run_node.cleanup(directory)
    _wait_for(lambda: not _running(pid), timeout=10)
    assert not list((directory / "active").iterdir())
    assert run_node.live_runs(directory) == []


@test
def test_cleanup_acceptance(directory: pathlib.Path, node: bool) -> None:
    """A run of an acceptance attempt whose runner was killed hard is live until --cleanup kills it.

    Its record is in the active/ directory of a label directory under the attempt's directory.
    """
    results = directory / "results"
    label_dir = results / "selftest" / "label"
    label_dir.mkdir(parents=True)
    attempt = run_node.acceptance_dir(label_dir, results) / "attempt" / "AT1"
    attempt.mkdir(parents=True)
    grandchild = directory / "grandchild"
    tasks.write_pass1(attempt, [_task("hard", [["spawn", 50.75, str(grandchild)]], limit_s=60.0)])
    process = subprocess.Popen(plugin.run_task_command(attempt, "hard", local=True))
    record = attempt / "active" / "hard"
    pid = _started(grandchild)
    _wait_for(record.exists, timeout=120)
    process.kill()
    process.wait()
    assert _running(pid)
    assert run_node.live_runs(label_dir) == []
    assert run_node.live_runs(label_dir, results) == [str(record)]
    with contextlib.redirect_stdout(io.StringIO()):
        run_node.cleanup(label_dir, results)
    _wait_for(lambda: not _running(pid), timeout=10)
    assert not record.exists()
    assert run_node.live_runs(label_dir, results) == []


@test
def test_cleanup_parallel(directory: pathlib.Path, node: bool) -> None:
    """After a hard kill of run_node.py, its GNU parallel is live until --cleanup kills it."""
    stand_in = [sys.executable, "-c", "import time; time.sleep(53.25)"]  # for GNU parallel
    script = (
        f"import pathlib, sys; sys.path.insert(0, {str(RUN_NODE.parent)!r}); import run_node; "
        f"run_node.run_parallel(pathlib.Path({str(directory)!r}), {stand_in!r})"
    )
    record = directory / "active" / run_node.PARALLEL_RECORD
    process = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.DEVNULL)
    try:
        _wait_for(lambda: process.poll() is not None or record.exists())
        assert record.exists(), "no record of GNU parallel"
        process.kill()
        process.wait()
        assert _processes_matching("time.sleep(53.25)")
        assert run_node.live_runs(directory) == [run_node.PARALLEL_RECORD]
        with contextlib.redirect_stdout(io.StringIO()):
            run_node.cleanup(directory)
        time.sleep(1)
        assert not _processes_matching("time.sleep(53.25)"), "GNU parallel survived --cleanup"
        assert not list((directory / "active").iterdir())
    finally:
        _kill(process)


@test
def test_cleanup_reused_ids(directory: pathlib.Path, node: bool) -> None:
    """--cleanup never kills a process that has reused a recorded id, even as a group's id."""
    reused = os.getpid()
    table = {reused: runner.Process(1, reused, 0, None, "a later start")}
    killed: list[tuple[str, int]] = []
    record = {
        "pid": reused,
        "pid_start": "the recorded start",
        "pgid": reused,
        "runner_pid": reused,
        "runner_start": "the recorded start",
    }
    (directory / "active").mkdir()
    (directory / "active" / "task").write_text(json.dumps(record))
    with (
        mock.patch.object(runner, "process_table", lambda: table),
        mock.patch.object(run_node.os, "killpg", lambda pid, sig: killed.append(("group", pid))),
        mock.patch.object(run_node.os, "kill", lambda pid, sig: killed.append(("process", pid))),
    ):
        assert run_node.live_runs(directory) == []
        run_node.cleanup(directory)
        assert not killed, killed

        # a group whose leader is gone, with members left, may be a later group with the same id:
        # it stays live, and cleanup names its members, kills none, and keeps the record
        table = {reused + 1: runner.Process(1, reused, 0, None, "member")}
        (directory / "active" / "task").write_text(json.dumps(record))
        assert run_node.live_runs(directory) == ["task"]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            run_node.cleanup(directory)
        assert not killed, killed
        assert f"{reused + 1} (" in output.getvalue(), output.getvalue()
        assert run_node.live_runs(directory) == ["task"]


@test
def test_cleanup_unkillable(directory: pathlib.Path, node: bool) -> None:
    """A process left after a run's grace that outlives --cleanup's SIGKILL keeps its record.

    --cleanup says so, and the run stays live, so that a start still refuses.
    """
    survivor = os.getpid()
    table = {survivor: runner.Process(1, survivor, 0, None, "the recorded start")}
    record = {
        "pid": survivor + 1,
        "pid_start": "gone",
        "pgid": None,
        "runner_pid": survivor + 1,
        "runner_start": "gone",
        "survivors": [[survivor, "the recorded start"]],
    }
    (directory / "active").mkdir()
    path = directory / "active" / "task.0123abcd"
    path.write_text(json.dumps(record))
    killed: list[int] = []
    with (
        mock.patch.object(runner, "process_table", lambda: table),
        mock.patch.object(run_node.os, "kill", lambda pid, sig: killed.append(pid)),
        mock.patch.object(run_node, "CLEANUP_WAIT_S", 0.0),
    ):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            run_node.cleanup(directory)
        assert killed == [survivor], killed
        assert "survived SIGKILL" in output.getvalue(), output.getvalue()
        assert path.exists()
        assert run_node.live_runs(directory) == [path.name]


@test
def test_cleanup_unreadable_records(directory: pathlib.Path, node: bool) -> None:
    """An active record that vanished is skipped; an unreadable one is live until --cleanup.

    ``--status`` says that the record is unreadable, not that its run is alive.
    """
    active = directory / "active"
    active.mkdir()
    (active / "gone").symlink_to(active / "nowhere")  # a file removed after the listing
    (active / "torn").write_text('{"pid": 1')
    run_node.refuse_other_hosts(directory)
    assert run_node.live_runs(directory) == ["torn"]
    with contextlib.redirect_stdout(io.StringIO()) as output:
        run_node.print_status(directory)
    assert "\n  torn: unreadable (" in output.getvalue(), output.getvalue()
    with contextlib.redirect_stdout(io.StringIO()) as output:
        run_node.cleanup(directory)
    assert "torn: unreadable" in output.getvalue(), output.getvalue()
    assert not (active / "torn").exists()
    assert run_node.live_runs(directory) == []


@test
def test_run_node_refusals(directory: pathlib.Path, node: bool) -> None:
    """run_node.py refuses live runs and a local acceptance pass for node mode.

    It also recognizes a trial bundle, which a node start refuses unless given --allow-unfrozen.
    """
    # a start while a run of an earlier start lives
    command = [
        sys.executable,
        str(RUN_NODE),
        "selftest",
        "--local",
        "--results-dir",
        str(directory / "results"),
        "--approved-cpu-hours",
        "10",
    ]
    subprocess.run(command, check=True, capture_output=True)
    (label_dir,) = [path.parent for path in (directory / "results").rglob(tasks.TASKS_FILE)]
    (label_dir / "active").mkdir(exist_ok=True)
    (label_dir / "active" / "live").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "pid_start": runner.process_start(os.getpid()),
                "pgid": None,
                "runner_pid": os.getpid(),
                "runner_start": runner.process_start(os.getpid()),
            }
        )
    )
    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode != 0 and "still alive" in refused.stderr, refused.stderr

    # a local pass of the acceptance tests does not count for node mode
    calls = []

    class Module:
        @staticmethod
        def acceptance(node: bool, results_dir: pathlib.Path) -> bool:
            calls.append(node)
            return True

    record_dir = directory / "acceptance"
    record_dir.mkdir()
    module: Any = Module
    run = subprocess.run

    def run_selftest(command: list[str], *args: Any, **kwargs: Any) -> Any:
        # the node self-tests pass; every other command, such as the CPU model's, runs
        if command[:2] == [sys.executable, str(run_node.SELFTEST)]:
            return subprocess.CompletedProcess(command, 0)
        return run(command, *args, **kwargs)

    with mock.patch.object(run_node.subprocess, "run", run_selftest):
        assert run_node.run_acceptance(module, record_dir, directory, node=False)
        assert run_node.run_acceptance(module, record_dir, directory, node=False)
        assert run_node.run_acceptance(module, record_dir, directory, node=True)
        assert run_node.run_acceptance(module, record_dir, directory, node=False)
    assert calls == [False, True], calls

    # a trial bundle, without a pinned qLDPC commit
    record = directory / "bundle.json"
    record.write_text(json.dumps({"unfrozen": True}))
    with mock.patch.object(manifest, "BUNDLE_RECORD", record):
        assert run_node._unfrozen_bundle()
        record.write_text(json.dumps({"unfrozen": False}))
        assert not run_node._unfrozen_bundle()

    # node mode pins this process, and so the runners, to the reserved CPUs
    if node and sys.platform.startswith("linux"):
        cpus = _node_cpus(2)
        with _affinity(cpus):
            run_node.pin_runners(cpus[:1])
            assert os.sched_getaffinity(0) == {cpus[0]}


@test
def test_run_node_jobs(directory: pathlib.Path, node: bool) -> None:
    """--jobs is auto or a positive integer, and a machine without room for a slot is refused."""
    for jobs in ("0", "-1", "two"):
        with (
            mock.patch.object(sys, "argv", ["run_node.py", "selftest", "--jobs", jobs]),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            _expect_exit(run_node.parse_args)
    with mock.patch.object(sys, "argv", ["run_node.py", "selftest", "--jobs", "3"]):
        assert run_node.parse_args().jobs == "3"

    lscpu = "# CPU,Core,Socket\n0,0,0\n1,1,0\n"  # two physical cores, both reserved
    module = plugin.load_experiment("selftest")
    with (
        mock.patch.object(
            run_node.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, lscpu)
        ),
        mock.patch.object(manifest, "mem_total_bytes", lambda: 2**40),
    ):
        error = acceptance.raises(
            SystemExit, lambda: run_node.choose_slots(module, directory, "auto", local=False)
        )
        assert "No job slot fits on this machine: it has 2" in str(error.code), error.code


@test
def test_run_node_missing_row(directory: pathlib.Path, node: bool) -> None:
    """A pass after which a task has no final row stops, and lists the task in a file.

    GNU parallel never reruns a task whose runner exited with status 0, as for a row damaged after
    it was written.  The message says how to rerun the task through a start: on the node, after
    deleting its lines from the pass's job log.  ``--status`` names the file.
    """
    task_list = [_task("kept", []), _task("lost", [])]
    tasks.write_pass1(directory, task_list)
    for task in task_list:
        store.append_row(
            directory, {"task_key": task["key"], "run_id": task["key"], "status": "ok"}
        )
    runs = directory / store.RUNS_FILE
    runs.write_text(runs.read_text().replace('"task_key":"lost"', '"task_key":lost'))
    passes: list[str] = []
    messages = []
    with mock.patch.object(run_node, "_run_keys", lambda *args: passes.append(args[1])):
        for local in (True, False):
            error = acceptance.raises(
                SystemExit, functools.partial(run_node.run_pass, directory, "1", 1, local=local)
            )
            messages.append(str(error.code))
    assert passes == ["1", "1"], passes
    missing = tasks.missing_file(directory, "1")
    assert missing.read_text() == "lost\n"
    local_message, node_message = messages
    assert str(missing) in local_message and "repeat this start" in local_message, local_message
    assert "job log" not in local_message, local_message
    assert str(tasks.joblog_file(directory, "1")) in node_message, node_message
    with contextlib.redirect_stdout(io.StringIO()) as output:
        run_node.print_status(directory)
    assert f"1 of them ended the pass without a final row: see {missing}" in output.getvalue()


@test
def test_run_node_other_host(directory: pathlib.Path, node: bool) -> None:
    """A start and --cleanup refuse a label whose ``active/`` lists runs of other hosts.

    The refusal names every such record, and ``--status`` names each record's host, not whether its
    run is alive.
    """
    label_dir, command = _local_label(directory)
    (label_dir / "active").mkdir(parents=True)
    record = {"pid": 1, "pid_start": None, "pgid": None, "runner_pid": 1, "runner_start": None}
    (label_dir / "active" / "t").write_text(json.dumps(record | {"host": "elsewhere.example"}))
    (label_dir / "active" / "u").write_text(json.dumps(record | {"host": "far.example"}))
    for extra in ([], ["--cleanup"]):
        refused = subprocess.run(command + extra, capture_output=True, text=True, check=False)
        assert refused.returncode != 0, refused.stderr
        for name, host in (("t", "elsewhere.example"), ("u", "far.example")):
            assert f"active/{name} (host {host})" in refused.stderr, refused.stderr
    assert (label_dir / "active" / "t").exists()
    assert not (label_dir / tasks.TASKS_FILE).exists()
    with contextlib.redirect_stdout(io.StringIO()) as output:
        run_node.print_status(label_dir)
    assert "\n  t: on host elsewhere.example," in output.getvalue(), output.getvalue()


@test
def test_run_node_cpu_model(directory: pathlib.Path, node: bool) -> None:
    """A start refuses a CPU model other than the one in the label's manifest.

    A dry run writes no manifest, so that the manifest is that of a start that runs tasks; it
    records itself, with its CPU model, in ``starts.jsonl``.
    """
    label_dir, command = _local_label(directory)
    label_dir.mkdir(parents=True)
    (label_dir / "manifest.json").write_text(json.dumps({"cpu_model": "Another CPU 9000"}))
    refused = subprocess.run(command, capture_output=True, text=True, check=False)
    assert refused.returncode != 0 and "Another CPU 9000" in refused.stderr, refused.stderr
    assert not (label_dir / tasks.TASKS_FILE).exists()

    (label_dir / "manifest.json").unlink()
    subprocess.run(command, check=True, capture_output=True)
    assert not (label_dir / "manifest.json").exists()
    (start,) = [json.loads(line) for line in (label_dir / "starts.jsonl").read_text().splitlines()]
    assert start["dry_run"] and start["cpu_model"] == runner.cpu_model(), start


@test
def test_run_node_environment(directory: pathlib.Path, node: bool) -> None:
    """A node start refuses a misleading environment unless ``--allow-env`` accepts it.

    ``PYTHONOPTIMIZE`` is refused when Python, or a Python that it starts, runs with -O, as a Python
    that it starts reads the variable even when it does not (-E), and ``NUMBA_DISABLE_JIT`` when
    ``int`` reads a nonzero value, as numba does.  The manifest records the environment variables
    that can change a run.
    """
    environ = {
        "PYTHONOPTIMIZE": "1",
        "NUMBA_CACHE_DIR": "/cache",
        "NUMBA_DISABLE_JIT": "0",
        "QLDPC_DISABLE_COMPILATION": "0",
        "LD_PRELOAD": "/lib.so",
        "HOME": "/home",
    }
    error = acceptance.raises(
        SystemExit, lambda: run_node.check_environment(environ, [], optimize=1)
    )
    assert "unset PYTHONOPTIMIZE" in str(error.code), error.code
    run_node.check_environment(environ, ["PYTHONOPTIMIZE"], optimize=1)

    # PYTHONOPTIMIZE that this interpreter ignores, as under -E, still makes its children optimize
    assert manifest.optimize_level({**os.environ, "PYTHONOPTIMIZE": "1"}) == 1
    _expect_exit(functools.partial(run_node.check_environment, {"PYTHONOPTIMIZE": "1"}, []))
    assert manifest.optimize_level(
        {k: v for k, v in os.environ.items() if k != "PYTHONOPTIMIZE"}
    ) == (sys.flags.optimize)

    # PYTHONOPTIMIZE as Python reads it, in a fresh interpreter
    for value, on in (
        ("0 ", True),
        (" ", True),
        ("\u0660", True),
        (" 0", False),
        ("+0", False),
        ("-0", False),
        ("00", False),
    ):
        optimize = subprocess.run(
            [sys.executable, "-c", "import sys; print(sys.flags.optimize)"],
            env=dict(os.environ, PYTHONOPTIMIZE=value),
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        check = functools.partial(
            run_node.check_environment, {"PYTHONOPTIMIZE": value}, [], int(optimize)
        )
        if on:
            _expect_exit(check)
        else:
            check()
    for value, on in (("1", True), (" 1 ", True), ("00", False), ("yes", False)):
        check = functools.partial(run_node.check_environment, {"NUMBA_DISABLE_JIT": value}, [], 0)
        if on:
            _expect_exit(check)
        else:
            check()
    for refused in (
        {"QLDPC_DISABLE_COMPILATION": "1"},
        {"PYTHONPATH": ""},
        {"PYTHONHOME": "/python"},
    ):
        _expect_exit(functools.partial(run_node.check_environment, refused, [], 0))
    recorded = manifest.recorded_environment(environ)
    assert recorded == {name: environ[name] for name in environ if name != "HOME"}, recorded


@test
def test_run_node_lock(directory: pathlib.Path, node: bool) -> None:
    """A start, or --cleanup, refuses a label whose lock another process holds."""
    label_dir, command = _local_label(directory)
    label_dir.mkdir(parents=True)
    fd = os.open(label_dir / "lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        for extra in ([], ["--cleanup"]):
            refused = subprocess.run(command + extra, capture_output=True, text=True, check=False)
            assert refused.returncode != 0 and "Another start" in refused.stderr, refused.stderr
        assert not (label_dir / tasks.TASKS_FILE).exists()
    finally:
        os.close(fd)
    subprocess.run(command, check=True, capture_output=True)
    assert (label_dir / tasks.TASKS_FILE).exists()


@test
def test_records_atomic(directory: pathlib.Path, node: bool) -> None:
    """A killed write of ``acceptance.json``, ``starts.jsonl``, or the manifest leaves no part."""

    class Killed(Exception):
        """The stand-in of a kill during a write."""

    write_text = pathlib.Path.write_text

    def torn_write(path: pathlib.Path, text: str, *args: Any, **kwargs: Any) -> int:
        write_text(path, text[: len(text) // 2])
        raise Killed

    class Module:
        @staticmethod
        def acceptance(node: bool, results_dir: pathlib.Path) -> bool:
            return True

    module: Any = Module
    label_dir = directory / "label"
    label_dir.mkdir()
    record = label_dir / "acceptance.json"
    record.write_text(json.dumps({"passed": False, "node": False}))
    manifest.record_start(label_dir, directory, {"dry_run": False})
    starts = (label_dir / store.STARTS_FILE).read_text()
    with mock.patch.object(pathlib.Path, "write_text", torn_write):
        for write in (
            lambda: run_node.run_acceptance(module, label_dir, directory, node=False),
            lambda: manifest.write_manifest(label_dir, {"jobs": 1}),
        ):
            acceptance.raises(Killed, write)
    assert json.loads(record.read_text()) == {"passed": False, "node": False}
    assert (label_dir / store.STARTS_FILE).read_text() == starts
    assert not (label_dir / "manifest.json").exists()


####################################################################################################
# acceptance


@test
def test_acceptance_attempts(directory: pathlib.Path, node: bool) -> None:
    """A second acceptance attempt runs its tasks again, in a fresh directory, keeping the first."""
    task = _task("at", [["value", 1]])
    attempts: list[tuple[pathlib.Path, str]] = []

    class Module:
        @staticmethod
        def acceptance(node: bool, results_dir: pathlib.Path) -> bool:
            row = local_runs.run_locally(results_dir / "AT1", [task])[task["key"]]
            attempts.append((results_dir, row["run_id"]))
            return len(attempts) > 1  # the first attempt fails

    module: Any = Module
    label_dir = directory / "results" / "selftest" / "label"
    label_dir.mkdir(parents=True)
    assert not run_node.run_acceptance(module, label_dir, directory / "results", node=False)
    assert run_node.run_acceptance(module, label_dir, directory / "results", node=False)
    (first, first_run), (second, second_run) = attempts
    assert first_run != second_run, "the second attempt passed on the first attempt's row"
    assert first != second and first.exists()
    assert first.parent == second.parent == directory / "results" / "acceptance" / "label"
    record = json.loads((label_dir / "acceptance.json").read_text())
    assert record["directory"] == str(second), record

    # a pass recorded on another CPU model does not count: the tests run again
    record["cpu_model"] = "another CPU model"
    (label_dir / "acceptance.json").write_text(json.dumps(record) + "\n")
    assert run_node.run_acceptance(module, label_dir, directory / "results", node=False)
    assert len(attempts) == 3, "a pass on another CPU model was reused"


@test
def test_acceptance_suite(directory: pathlib.Path, node: bool) -> None:
    """A test that raises an ``Exception``, or ``SystemExit``, fails alone.

    An unknown test ID is refused.
    """

    class Unexpected(Exception):
        pass

    suite = acceptance.Suite("spc-suite-")
    ran = []

    @suite.test
    def at1_raises(test_dir: pathlib.Path, node: bool) -> None:
        ran.append(test_dir.name)
        raise Unexpected

    @suite.test
    def at2_exits(test_dir: pathlib.Path, node: bool) -> None:
        ran.append(test_dir.name)
        sys.exit("an analysis refused")

    @suite.test
    def at3_passes(test_dir: pathlib.Path, node: bool) -> None:
        ran.append(test_dir.name)

    with contextlib.redirect_stdout(io.StringIO()):
        assert not suite.run_all(node=node, results_dir=directory)
    assert ran == ["AT1", "AT2", "AT3"], ran
    assert suite.parse_args("", ["--only", "AT3"]).only == ["AT3"]
    with contextlib.redirect_stderr(io.StringIO()):
        _expect_exit(lambda: suite.parse_args("", ["--only", "AT4"]))

    # the self-tests refuse an unknown test name too
    command = [sys.executable, __file__, "--only", "no_such_test"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0 and "invalid choice" in result.stderr, result


@test
def test_acceptance_limit(directory: pathlib.Path, node: bool) -> None:
    """A test past its suite's limit fails, the next test still runs, and no timer stays armed."""
    suite = acceptance.Suite("spc-limit-", limit_s=0.5)
    ran = []

    @suite.test
    def at1_slow(test_dir: pathlib.Path, node: bool) -> None:
        time.sleep(10)
        ran.append(test_dir.name)

    @suite.test
    def at2_quick(test_dir: pathlib.Path, node: bool) -> None:
        ran.append(test_dir.name)

    previous = signal.getsignal(signal.SIGALRM)
    start = time.monotonic()
    try:
        with contextlib.redirect_stdout(io.StringIO()) as output:
            assert not suite.run_all(node=node, results_dir=directory)
        assert time.monotonic() - start < 5, "the slow test was not stopped"
        assert ran == ["AT2"], ran
        assert "FAIL AT1" in output.getvalue() and "PASS AT2" in output.getvalue(), output
        assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0), "a timer stayed armed"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous if previous is not None else signal.SIG_DFL)


@test
def test_plugin_name(directory: pathlib.Path, node: bool) -> None:
    """``load_experiment`` refuses a plug-in whose ``NAME`` is not the name that loads it."""
    with mock.patch.dict(plugin.PLUGINS, {"other": plugin.PLUGINS["selftest"]}):
        error = acceptance.raises(ValueError, lambda: plugin.load_experiment("other"))
    assert "is named 'selftest'" in str(error), error


@test
def test_acceptance_selection(directory: pathlib.Path, node: bool) -> None:
    """Each experiment's plug-in runs its suite's node tests on the node, and every test off it.

    The suite's tests are replaced by stand-ins that record their IDs; the IDs, the plug-in's
    ``acceptance``, and the shared runner's selection are the experiment's own.  ``--only`` runs
    only the tests that it names.
    """
    ran: list[str] = []
    for name in plugin.EXPERIMENTS:
        module = plugin.load_experiment(name)
        tests_module = importlib.import_module(module.__name__.rsplit(".", 1)[0] + ".acceptance")
        suite = tests_module.SUITE
        ids = list(suite.tests)
        node_ids = [test_id for test_id in ids if test_id in (suite.node_tests or ids)]
        assert node_ids and set(node_ids) == set(suite.node_tests or ids), (name, ids, node_ids)
        stand_ins = {
            test_id: lambda test_dir, node, test_id=test_id: ran.append(test_id) for test_id in ids
        }
        with (
            mock.patch.object(suite, "tests", stand_ins),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            for on_node, expected in ((True, node_ids), (False, ids)):
                ran.clear()
                assert module.acceptance(on_node, directory / name / str(on_node))
                assert ran == expected, (name, on_node, ran)
            ran.clear()
            assert suite.run_all(node=False, results_dir=directory / name, only=ids[-1:])
            assert ran == ids[-1:], (name, ran)


####################################################################################################
# manifest and labels


@test
def test_manifest_starts(directory: pathlib.Path, node: bool) -> None:
    """Each start records its environment and Python's optimization level in ``starts.jsonl``.

    So a resume under a changed environment is recorded.  ``manifest.json`` keeps the first start's
    line.  Each start's line gets its job slots.  The recorded level is that of the start's children
    when it is higher, as under ``PYTHONOPTIMIZE``, which this interpreter has already read.
    """
    for value in ("1", "2"):
        with mock.patch.dict(os.environ, {"NUMBA_OPT": value}):
            manifest.record_start(directory, directory, {"start": value, "dry_run": False})
            manifest.write_manifest(directory, {"jobs": int(value)})
    record = json.loads((directory / "manifest.json").read_text())
    assert (record["start"], record["environment"]["NUMBA_OPT"]) == ("1", "1"), record
    assert record["jobs"] == 1, record
    starts = [json.loads(line) for line in (directory / "starts.jsonl").read_text().splitlines()]
    assert [start["environment"]["NUMBA_OPT"] for start in starts] == ["1", "2"], starts
    assert [start["jobs"] for start in starts] == [1, 2], starts
    assert [start["python_optimize"] for start in starts] == [sys.flags.optimize] * 2, starts
    with mock.patch.dict(os.environ, {"PYTHONOPTIMIZE": "1"}):
        manifest.record_start(directory, directory, {"start": "optimized", "dry_run": False})
    last = json.loads((directory / "starts.jsonl").read_text().splitlines()[-1])
    assert last["python_optimize"] == max(sys.flags.optimize, 1), last


@test
def test_manifest_optimize_level(directory: pathlib.Path, node: bool) -> None:
    """``optimize_level`` reads the child's last line, and exits with the child's error if it fails.

    A ``sitecustomize`` that prints a line first leaves the level readable.  One that prints a level
    and then exits with an error, and one that silences the child's output, each end the start, the
    first with the child's error.
    """
    from experiments.state_prep_certification.common import manifest

    def environ(name: str, site: str, **extra: str) -> dict[str, str]:
        (directory / name).mkdir()
        (directory / name / "sitecustomize.py").write_text(site)
        return {**os.environ, "PYTHONPATH": str(directory / name), **extra}

    chatty = environ("chatty", 'print("site")\n', PYTHONOPTIMIZE="1")
    assert manifest.optimize_level(chatty) == max(sys.flags.optimize, 1)
    failing = environ("failing", 'import sys\nprint(0)\nsys.exit("child marker")\n')
    silent = environ("silent", "import os, sys\nsys.stdout = open(os.devnull, 'w')\n")
    for child, expected in ((failing, "child marker"), (silent, "fails")):
        try:
            manifest.optimize_level(child)
        except SystemExit as error:
            assert expected in str(error.code), error.code
        else:
            raise AssertionError("no error")


@test
def test_manifest_label(directory: pathlib.Path, node: bool) -> None:
    """A label covers the pins, frozen files, harness commit, and the requirements it is given.

    The label reads no bundle record: a ``bundle.json`` beside the code leaves it unchanged.
    """
    frozen = directory / "frozen.json"
    frozen.write_text("{}")
    pins = directory / "pins.json"
    pins.write_text("{}")
    record = directory / "bundle.json"
    labels = set()
    with (
        mock.patch.object(manifest, "BUNDLE_RECORD", record),
        mock.patch.object(manifest, "PINS_FILE", pins),
    ):
        assert manifest.label_files([frozen]) == [("pins.json", b"{}"), ("frozen.json", b"{}")]
        assert manifest.bundle_record() is None
        unbundled = manifest.compute_label("exp", manifest.label_files([frozen]), "c1", None)
        labels.add(unbundled)
        labels.add(manifest.compute_label("exp", manifest.label_files([frozen]), "c2", None))
        for digest in ("aa", "bb"):
            labels.add(manifest.compute_label("exp", manifest.label_files([frozen]), "c1", digest))
        record.write_text(json.dumps({"requirements_sha256": "aa"}))
        assert manifest.bundle_record() == {"requirements_sha256": "aa"}
        assert (
            manifest.compute_label("exp", manifest.label_files([frozen]), "c1", None) == unbundled
        )
        frozen.write_text('{"changed": 1}')
        labels.add(manifest.compute_label("exp", manifest.label_files([frozen]), "c1", None))
        pins.write_text('{"changed": 1}')
        labels.add(manifest.compute_label("exp", manifest.label_files([frozen]), "c1", None))
    assert len(labels) == 6, labels


@test
def test_label_suffixes(directory: pathlib.Path, node: bool) -> None:
    """A label directory's suffixes: -dirty-<hash>, then -local, then -<subset>.

    In a bundle, the label covers the commit and the requirements of ``bundle.json``.
    """
    module = plugin.load_experiment("selftest")
    names = {}
    for dirty in (False, True):
        commit = {
            "commit": "c",
            "dirty": dirty,
            "diff_sha256": "0123456789abcdef" if dirty else None,
        }
        with mock.patch.object(manifest, "harness_commit", lambda commit=commit: commit):
            for local in (False, True):
                for subset in ("main", "smoke"):
                    if local or not dirty:
                        label_dir = run_node.get_label_dir(module, directory, subset, local)
                        names[dirty, local, subset] = label_dir.name
    label = names[False, False, "main"]

    # in a bundle, the label covers the commit and the requirements that bundle.json records
    record = directory / "bundle.json"
    record.write_text(json.dumps({"harness_commit": "c" * 40, "requirements_sha256": "aa"}))
    with mock.patch.object(manifest, "BUNDLE_RECORD", record):
        bundled = run_node.get_label_dir(module, directory, "main", False).name
    files = manifest.label_files([])
    assert bundled == manifest.compute_label(module.NAME, files, "c" * 40, "aa"), bundled
    assert names == {
        (False, False, "main"): label,
        (False, False, "smoke"): f"{label}-smoke",
        (False, True, "main"): f"{label}-local",
        (False, True, "smoke"): f"{label}-local-smoke",
        (True, True, "main"): f"{label}-dirty-01234567-local",
        (True, True, "smoke"): f"{label}-dirty-01234567-local-smoke",
    }, names


@workstation_only
@test
def test_manifest_dirty_digest(directory: pathlib.Path, node: bool) -> None:
    """A dirty tree's label directory names a hash of its staged and unstaged tracked changes."""
    module = plugin.load_experiment("selftest")

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(directory), *args], capture_output=True, check=True)

    def digest() -> str | None:
        with (
            mock.patch.object(plugin, "ROOT", directory),
            mock.patch.object(manifest, "BUNDLE_RECORD", directory / "no-bundle.json"),
        ):
            name = run_node.get_label_dir(module, directory, "main", local=True).name
            commit = manifest.harness_commit()
        assert commit["dirty"] == (commit["diff_sha256"] is not None), commit
        if commit["dirty"]:
            assert f"-dirty-{commit['diff_sha256'][:8]}" in name, (name, commit)
        else:
            assert "-dirty" not in name, name
        return commit["diff_sha256"]

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (directory / "a.txt").write_text("1\n")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    assert digest() is None
    (directory / "a.txt").write_text("2\n")
    unstaged = digest()
    git("add", "a.txt")
    assert digest() == unstaged, "staging a change changed the digest"
    (directory / "a.txt").write_text("3\n")
    changed = digest()
    assert changed not in (None, unstaged)
    (directory / "untracked.txt").write_text("x\n")
    assert digest() == changed, "an untracked file changed the digest"


####################################################################################################
# bundle and node.sh


@test
def test_node_sh(directory: pathlib.Path, node: bool) -> None:
    """node.sh stops with a message on a missing pin, a missing bundle, or a misplaced option.

    On Linux x86-64, where its preflight passes, two bundles that share a results directory each
    install and record their own CPython and GNU parallel, also on a later start, and the preflight
    record is JSON where a value needs escaping.  Stand-ins of uv, GNU parallel, and the bundle's
    interpreter replace the real tools.
    """
    shutil.copy(plugin.EXPERIMENTS_DIR / "node.sh", directory)
    pins = {"python": {"version": "3.12"}, "platform": {}, "gnu_parallel": {"version": "20240122"}}
    (directory / "pins.json").write_text(json.dumps(pins))
    for arguments, message in (
        ([], "node.sh EXPERIMENT --approved-cpu-hours H"),
        (["--check"], 'pins.json has no string "min_glibc" in "platform"'),
        (["selftest", "--status"], "no bundle.json beside node.sh"),
        (["selftest", "--results-dir", "x"], "set FTSP_RESULTS"),
    ):
        result = subprocess.run(
            ["bash", str(directory / "node.sh"), *arguments],
            capture_output=True,
            text=True,
            check=False,
            env={key: value for key, value in os.environ.items() if key != "FTSP_RESULTS"},
        )
        output = result.stdout + result.stderr
        assert result.returncode == 1 and message in output, (arguments, result)
        # neither the system's python nor run_node.py ran
        assert "/bin/python:" not in output and "can't open file" not in output, (arguments, result)

    if not (sys.platform.startswith("linux") and platform.machine() == "x86_64"):
        return
    stubs, results = directory / "stubs", directory / "results"
    uv_version = 'uv 0.0 "quoted" \\back'
    _write_script(
        stubs / "uv",
        f"""case "$1" in
    --version) printf '%s\\n' {shlex.quote(uv_version)} ;;
    python) exit 1 ;;
    venv) mkdir -p "$5/bin" && printf '%s\\n' "$4" > "$5/built_with" &&
        printf '#!/bin/sh\\nexit 0\\n' > "$5/bin/python" && chmod +x "$5/bin/python" ;;
esac""",
    )
    _write_script(stubs / "parallel", "echo 'GNU parallel 20000101'")  # older than the pin
    # the stubs come first on the PATH, so that they stand in for uv and GNU parallel, and the
    # other tools of the preflight are found wherever this machine has them
    path = os.pathsep.join([str(stubs), os.environ.get("PATH", "/usr/bin:/bin")])
    environment = {"PATH": path, "FTSP_RESULTS": str(results), "HOME": "/"}
    for name in ("a", "b", "a"):
        script = _node_sh_bundle(directory / name, name)
        result = subprocess.run(
            ["bash", str(script), "selftest", "--approved-cpu-hours", "1"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        assert result.returncode == 0, (name, result)
        preflight = json.loads((results / f"preflight-{name}.json").read_text())
        assert preflight["uv"] == uv_version, preflight
        parallel = f"{results / f'ext-{name}' / 'bin' / 'parallel'} GNU parallel 20240122 {name}"
        assert preflight["parallel_used"] == parallel, preflight
        built_with = (results / f"venv-{name}" / "built_with").read_text().strip()
        assert built_with == str(results / f"python-{name}" / "bin" / "python3.12"), built_with


@test
def test_bundle_requirements(directory: pathlib.Path, node: bool) -> None:
    """Compiled requirements split into entries with their hashes, comments dropped."""
    text = (
        "diskcache==5.6.3 \\\n"
        "    --hash=sha256:aa \\\n"
        "    --hash=sha256:bb\n"
        "    # via qldpc\n"
        "sinter==1.16.0 \\\n"
        "    --hash=sha256:cc\n"
        "typing-extensions==4.16.0 ; python_version < '3.13' \\\n"
        "    --hash=sha256:dd\n"
    )
    entries = bundle._split_requirements(text)
    assert [(name, version) for name, version, _ in entries] == [
        ("diskcache", "5.6.3"),
        ("sinter", "1.16.0"),
        ("typing-extensions", "4.16.0"),
    ], entries
    assert entries[0][2].count("--hash") == 2 and "via" not in entries[0][2]


@test
def test_bundle_frozen_files(directory: pathlib.Path, node: bool) -> None:
    """A bundle refuses an experiment whose frozen files are missing, naming the one to add."""
    certification = types.SimpleNamespace(
        NAME="certification", FROZEN_FILES=["config.json", "data/circuits.json"]
    )
    reader = types.SimpleNamespace(
        NAME="reader", FROZEN_FILES=["config.json", "../certification/data/circuits.json"]
    )
    experiments_dir = directory / "experiments" / "state_prep_certification"
    for path in ("reader/config.json", "certification/config.json"):
        (experiments_dir / path).parent.mkdir(parents=True, exist_ok=True)
        (experiments_dir / path).write_text("{}")
    for modules, message in (
        ([reader], "reader needs certification's circuits.json: bundle certification too"),
        ([certification], "certification: its frozen file data/circuits.json is not in the bundle"),
    ):
        error = acceptance.raises(
            SystemExit, functools.partial(bundle.check_frozen_files, directory, modules)
        )
        assert str(error.code).startswith(message), error.code
    (experiments_dir / "certification" / "data").mkdir()
    (experiments_dir / "certification" / "data" / "circuits.json").write_text("{}")
    bundle.check_frozen_files(directory, [certification, reader])


@test
def test_bundle_pure_wheel(directory: pathlib.Path, node: bool) -> None:
    """A built wheel is matched by its normalized name and version, and may be py2.py3."""
    cache, wheels = directory / "cache", directory / "wheels"
    for path in ("sdists/foo.bar-1.1.tar.gz", "built/foo_bar-1.10-py3-none-any.whl"):
        (cache / path).parent.mkdir(parents=True, exist_ok=True)
        (cache / path).write_text("")
    wheels.mkdir()
    built = cache / "built"
    with mock.patch.object(bundle, "_run", lambda command: None):
        (built / "foo_bar-1.1-py2.py3-none-any.whl").write_text("")
        target = bundle.build_pure_python_wheel("Foo-Bar", "1.1", directory, wheels, cache)
        assert target == wheels / "foo_bar-1.1-py2.py3-none-any.whl", target
        (built / "foo_bar-1.1-py2.py3-none-any.whl").rename(
            built / "foo_bar-1.1-cp312-cp312-linux_x86_64.whl"
        )
        _expect_exit(
            lambda: bundle.build_pure_python_wheel("Foo-Bar", "1.1", directory, wheels, cache)
        )


@workstation_only
@test
def test_bundle_check_tree(directory: pathlib.Path, node: bool) -> None:
    """A bundle needs HEAD to descend from the pinned commit with src/ and pyproject unchanged."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(directory), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "test")
    (directory / "src").mkdir()
    (directory / "src" / "a.py").write_text("a = 1\n")
    (directory / "pyproject.toml").write_text("[project]\n")
    git("add", ".")
    git("commit", "-q", "-m", "base")
    base = git("rev-parse", "HEAD")
    (directory / "experiments.txt").write_text("harness\n")
    git("add", ".")
    git("commit", "-q", "-m", "harness")
    with mock.patch.object(plugin, "ROOT", directory):
        assert bundle.check_tree({"qldpc_commit": base}, unfrozen=False) == git("rev-parse", "HEAD")
        (directory / "src" / "a.py").write_text("a = 2\n")
        git("commit", "-qam", "change src")
        _expect_exit(lambda: bundle.check_tree({"qldpc_commit": base}, unfrozen=False))
        git("checkout", "-q", "--orphan", "other")
        git("commit", "-qm", "unrelated")
        _expect_exit(lambda: bundle.check_tree({"qldpc_commit": base}, unfrozen=False))
        _expect_exit(lambda: bundle.check_tree({"qldpc_commit": None}, unfrozen=False))
        bundle.check_tree({"qldpc_commit": None}, unfrozen=True)


####################################################################################################
# helpers


def _append_many(label_dir: pathlib.Path, worker: int, count: int) -> None:
    """Append rows from one process."""
    for index in range(count):
        store.append_row(label_dir, {"task_key": f"w{worker}.{index}", "pad": "x" * 5000})


def _task(key: str, steps: list[list[Any]], **fields: Any) -> dict[str, Any]:
    """A task of the stand-in experiment."""
    task = {
        "key": key,
        "experiment": "selftest",
        "arm": "stand_in",
        "mode": "timed",
        "repeat": 0,
        "input": steps,
        "limit_s": 10.0,
        "mem_cap_gib": 2.0,
        "grace_s": 5.0,
    }
    task.update(fields)
    return task


def _run(directory: pathlib.Path, task: dict[str, Any], node: bool) -> dict[str, Any]:
    """Run one task through ``run_task.py`` and return its final row."""
    tasks.write_pass1(directory, [task])
    status = subprocess.run(
        plugin.run_task_command(directory, task["key"], local=not node), check=False
    )
    assert status.returncode == 0, status.returncode
    final, problems = store.final_rows(directory)
    assert not problems, problems
    assert not (directory / "active" / task["key"]).exists()
    return final[task["key"]]


def _parallel(directory: pathlib.Path, name: str, jobs: int, node: bool) -> subprocess.Popen[bytes]:
    """Start GNU parallel on one pass, from the repository root."""
    command = tasks.parallel_command(
        directory, pathlib.Path(sys.executable), name, jobs, local=not node
    )
    return subprocess.Popen(command, cwd=plugin.ROOT, stdout=subprocess.DEVNULL)


def _processes_matching(text: str) -> list[str]:
    """The command lines of live processes that contain ``text``."""
    output = subprocess.run(
        ["ps", "-A", "-o", "command="], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in output.splitlines() if text in line and "ps -A" not in line]


def _started(path: pathlib.Path) -> int:
    """The process id that a stand-in's ``spawn`` step writes once its grandchild has started."""
    _wait_for(path.exists, timeout=120)
    return int(path.read_text())


def _running(pid: int) -> bool:
    """Whether a process exists and has not ended, as a zombie has."""
    state = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return bool(state) and not state.startswith("Z")


def _kill(process: subprocess.Popen[bytes]) -> None:
    """Kill a process that a test started, if it is still running, and reap it."""
    if process.poll() is None:
        process.kill()
        process.wait()


def _wait_for(condition: Callable[[], bool], timeout: float = 60) -> None:
    """Wait until a condition holds, or fail after the timeout."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.2)


def _node_cpus(count: int) -> list[int]:
    """CPUs that this process may use (Linux)."""
    if not sys.platform.startswith("linux"):
        raise RuntimeError("node mode needs Linux")
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < count:
        raise RuntimeError(f"node mode needs {count} CPUs")
    return cpus[:count]


@contextlib.contextmanager
def _affinity(cpus: list[int]) -> Iterator[None]:
    """Run the block, and the processes that it starts, on these CPUs (Linux; none: no change)."""
    if not cpus or not sys.platform.startswith("linux"):
        yield
        return
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    try:
        yield
    finally:
        os.sched_setaffinity(0, previous)


def _write_script(path: pathlib.Path, body: str) -> None:
    """Write an executable shell script."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/sh\n{body}\n")
    path.chmod(0o755)


def _node_sh_bundle(root: pathlib.Path, name: str) -> pathlib.Path:
    """Make an unpacked stand-in bundle, once, and return the path of its ``node.sh``.

    The bundle's CPython and GNU parallel are shell scripts, and its GNU parallel names the bundle.
    """
    experiments_dir = root / "qLDPC" / "experiments" / "state_prep_certification"
    if experiments_dir.exists():
        return experiments_dir / "node.sh"
    experiments_dir.mkdir(parents=True)
    shutil.copy(plugin.EXPERIMENTS_DIR / "node.sh", experiments_dir)
    pins = {
        "python": {"version": "3.12.7", "standalone_asset": "cpython.tar.gz"},
        "platform": {"min_glibc": "2.17"},
        "gnu_parallel": {"version": "20240122"},
    }
    (experiments_dir / "pins.json").write_text(json.dumps(pins))
    (experiments_dir / "bundle.json").write_text(json.dumps({"name": name}))
    (root / "wheels").mkdir()
    (root / "requirements-node.txt").write_text("")
    for path in ("python", "parallel"):
        (root / path).mkdir()
    with tarfile.open(root / "python" / "cpython.tar.gz", "w:gz") as tar:
        _add_text(tar, "python/bin/python3.12", "#!/bin/sh\n")
    with tarfile.open(root / "parallel" / "parallel-20240122.tar.bz2", "w:bz2") as tar:
        script = f"#!/bin/sh\necho 'GNU parallel 20240122 {name}'\n"
        _add_text(tar, "parallel-20240122/src/parallel", script)
    pins_path = "qLDPC/experiments/state_prep_certification/pins.json"
    (root / "SHA256SUMS").write_text(f"{store.sha256_file(root / pins_path)}  {pins_path}\n")
    return experiments_dir / "node.sh"


def _add_text(tar: tarfile.TarFile, name: str, text: str) -> None:
    """Add an executable file with the given text to an archive."""
    info = tarfile.TarInfo(name)
    info.size, info.mode = len(text.encode()), 0o755
    tar.addfile(info, io.BytesIO(text.encode()))


def _local_label(directory: pathlib.Path) -> tuple[pathlib.Path, list[str]]:
    """The stand-in's label directory for local runs in ``directory/results``, and a start of it.

    The start is a dry run, which stops after writing the task list and its line in
    ``starts.jsonl``.
    """
    results = directory / "results"
    module = plugin.load_experiment("selftest")
    label_dir = run_node.get_label_dir(module, results, "main", True)
    command = [sys.executable, str(RUN_NODE), "selftest", "--local", "--results-dir", str(results)]
    return label_dir, command + ["--approved-cpu-hours", "10", "--dry-run"]


def _require_parallel() -> None:
    """Fail if GNU parallel is not installed."""
    if shutil.which("parallel") is None:
        raise RuntimeError("GNU parallel is not installed")


def _expect_exit(function: Callable[[], Any]) -> None:
    """Check that a function exits with an error message."""
    assert acceptance.raises(SystemExit, function).code not in (None, 0)


if __name__ == "__main__":
    main()
