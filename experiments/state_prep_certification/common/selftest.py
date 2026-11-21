# SPDX-License-Identifier: Apache-2.0

"""Tests of the shared code of the experiments, with a stand-in experiment.

    python experiments/state_prep_certification/common/selftest.py [--node] [--only NAME ...] \
        [--keep]

Each test writes into a fresh temporary directory and uses the stand-in experiment of
``selftest_plugin.py``.  By default, runs use local mode (``--local``), which keeps every run in
this session.  With ``--node`` (Linux only), the tests of the runner, of GNU parallel, and of
``--cleanup`` run in node mode instead: each run in a new session, killed by process group, with its
child pinned to a CPU and its runner on other CPUs; ``run_node.py`` runs them before the node's
acceptance tests.  The test of ``bundle.check_tree``, which needs git, runs only without
``--node``.  The tests of GNU parallel need ``parallel`` on the ``PATH``.

The script prints one line per test and exits with status 1 if any test failed.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

from experiments.state_prep_certification.common import store, tasks

ROOT = pathlib.Path(__file__).resolve().parents[3]
RUN_TASK = ROOT / "experiments" / "state_prep_certification" / "common" / "run_task.py"
RUN_NODE = ROOT / "experiments" / "state_prep_certification" / "run_node.py"
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
        except (
            AssertionError,
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
            subprocess.SubprocessError,
        ):
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
    parser.add_argument("--only", nargs="*", help="run only these tests")
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
    """The archive holds every file of the label with matching checksums, and its .sha256."""
    label_dir = directory / "label"
    (label_dir / "logs").mkdir(parents=True)
    store.append_row(label_dir, {"task_key": "a"})
    (label_dir / "logs" / "a.txt").write_text("log")
    archive = store.make_archive(label_dir, "label-host-utc")
    sha256 = (archive.parent / (archive.name + ".sha256")).read_text().split()[0]
    assert sha256 == store.sha256_file(archive)
    with tarfile.open(archive) as tar:
        names = set(tar.getnames())
        sums_file = tar.extractfile("label-host-utc/SHA256SUMS")
        assert sums_file is not None
        sums = sums_file.read().decode()
    expected = {f"label-host-utc/{name}" for name in ("runs.jsonl", "logs/a.txt", "SHA256SUMS")}
    assert names == expected and "runs.jsonl" in sums and "logs/a.txt" in sums
    store.make_archive(label_dir, "second")  # earlier archives never enter a later one
    with tarfile.open(label_dir / "archive" / "second.tar.gz") as tar:
        assert not any("archive/" in name for name in tar.getnames())


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
    subprocess.run(_run_task_command(directory, "ok", node), check=True)
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
    from experiments.state_prep_certification.common import runner

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
def test_runner_compile_retry(directory: pathlib.Path, node: bool) -> None:
    """A run whose measured call compiles is discarded; its retry, which loads the code, is kept."""
    with _environment("NUMBA_CACHE_DIR", str(directory / "numba")):
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
    with _environment("NUMBA_CACHE_DIR", str(directory / "numba")):
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
    with _environment("NUMBA_CACHE_DIR", str(directory / "numba")):
        row = _run(directory, task, node)
    assert row["status"] == "ok" and not row["compiled_in_timing"], row
    assert row["discarded_run"]["compiled_function"].endswith("_add_one"), row


@test
def test_runner_compile_vectorize(directory: pathlib.Path, node: bool) -> None:
    """An uncached ``numba.vectorize`` ufunc is reported, and the retry recompiles it, flagged."""
    with _environment("NUMBA_CACHE_DIR", str(directory / "numba")):
        row = _run(directory, _task("ufunc", [["vectorize"], ["value", 1]]), node)
    assert row["status"] == "ok" and row["compiled_in_timing"], row
    assert row["compiled_function"].endswith("_add_one"), row


@test
def test_runner_compile_untimed(directory: pathlib.Path, node: bool) -> None:
    """Compilations before and after the measured call, even uncached, are not reported."""
    task = _task("untimed", [["compile_untimed"], ["value", 6], ["post", 0.5]])
    with _environment("NUMBA_CACHE_DIR", str(directory / "numba")):
        row = _run(directory, task, node)
    assert row["status"] == "ok" and row["post_status"] == "ok", row
    assert not row["compiled_in_timing"] and row["discarded_run"] is None, row


@test
def test_runner_compile_then_timeout(directory: pathlib.Path, node: bool) -> None:
    """A run killed at its limit after it compiled is discarded too, and the task retried."""
    task = _task("slow", [["compile", True], ["sleep", 30]], limit_s=10.0)
    with _environment("NUMBA_CACHE_DIR", str(directory / "numba")):
        row = _run(directory, task, node)
    assert row["status"] == "timeout" and not row["compiled_in_timing"], row
    discarded = row["discarded_run"]
    assert discarded["status"] == "timeout" and discarded["T"] >= 10.0, row
    assert discarded["compiled_function"].endswith("_add_one"), discarded


@test
def test_runner_compile_then_killed(directory: pathlib.Path, node: bool) -> None:
    """A run killed by a signal after it compiled is discarded too, and the task retried."""
    with _environment("NUMBA_CACHE_DIR", str(directory / "numba")):
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
    with _environment("NUMBA_CACHE_DIR", str(directory / "numba")):
        process = subprocess.Popen(_run_task_command(directory, "stopped", node))
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
    with _environment("NUMBA_CACHE_DIR", str(directory / "numba")):
        process = subprocess.Popen(_run_task_command(directory, "retried", node))
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
def test_local_runs_reject_compiled(directory: pathlib.Path, node: bool) -> None:
    """``local_runs.run_locally`` fails if a row with status ok has ``compiled_in_timing`` set.

    A flagged row of another status, here a timeout, is accepted.
    """
    from experiments.state_prep_certification.common import local_runs

    with _environment("NUMBA_CACHE_DIR", str(directory / "numba")):
        try:
            local_runs.run_locally(directory / "label", [_task("uncached", [["compile", False]])])
        except AssertionError as error:
            assert "uncached" in str(error), error
        else:
            raise AssertionError("a row with compiled_in_timing set was accepted")
        task = _task("slow", [["compile", False], ["sleep", 30]], limit_s=3.0)
        row = local_runs.run_locally(directory / "timeout", [task])["slow"]
    assert row["status"] == "timeout" and row["compiled_in_timing"], row


@test
def test_local_runs_interrupted(directory: pathlib.Path, node: bool) -> None:
    """``local_runs.run_locally`` stops a task's run if an exception interrupts it meanwhile."""
    from experiments.state_prep_certification.common import local_runs

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
    try:
        local_runs.run_locally(directory, [_task("stop", [["spawn", 53.25]], limit_s=60.0)])
    except Interrupted:
        pass
    else:
        raise AssertionError("the run was not interrupted")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert not _processes_matching("time.sleep(53.25)")


@test
def test_runner_sigterm(directory: pathlib.Path, node: bool) -> None:
    """On SIGTERM the runner kills its run, writes no row, and exits with status 2."""
    tasks.write_pass1(directory, [_task("stop", [["spawn", 49.25]], limit_s=60.0)])
    process = subprocess.Popen(_run_task_command(directory, "stop", node))
    _wait_for(lambda: (directory / "active" / "stop").exists())
    time.sleep(2)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=40) == 2
    assert not (directory / "active" / "stop").exists()
    assert store.read_rows(directory) == ([], [])
    assert not _processes_matching("time.sleep(49.25)")


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
    try:
        tasks.write_pass1(directory, task_list[:-1])
    except FileExistsError:
        pass
    else:
        raise AssertionError("a changed task list was written")
    assert tasks.pass_keys(directory, "1") == order


@test
def test_tasks_ceiling_and_cores(directory: pathlib.Path, node: bool) -> None:
    """The ceiling, with repeats and retries, needs an approval; job slots get a CPU per core."""
    task_list = [_task(f"t{i}", [], limit_s=1800.0) for i in range(10)]
    task_list += [
        _task(f"u{i}", [], limit_s=1800.0, startup_cap_s=60.0, post_s=600.0) for i in range(5)
    ]
    ceiling = tasks.ceiling_hours(task_list, lambda task: 3 * (120 + task["limit_s"] + 30))
    # each task costs S, the most that one run at its first run's limit can take, plus twice its
    # worst case of three runs.  S is the sum of its start-up cap, limit, post-processing allowance,
    # and grace of 5 s, where the runner's defaults are STARTUP_CAP_S = 120 and FINISH_S = 30.
    default_cost = (120 + 1800 + 30 + 5) + 2 * 3 * 1950
    own_cost = (60 + 1800 + 600 + 5) + 2 * 3 * 1950
    assert abs(ceiling - (10 * default_cost + 5 * own_cost) / 3600) < 1e-9
    for approved in (None, ceiling - 1):
        try:
            tasks.check_approval(ceiling, approved)
        except tasks.ApprovalError:
            pass
        else:
            raise AssertionError("an insufficient approval passed")
    tasks.check_approval(ceiling, ceiling)

    # 8 physical cores with SMT siblings: CPUs 0-7 on cores 0-7, CPUs 8-15 on the same cores
    lscpu = "# CPU,Core,Socket\n" + "".join(f"{cpu},{cpu % 8},0\n" for cpu in range(16))
    assert tasks.choose_cpus(lscpu, 128 * 2**30, 8) == ([2, 3, 4, 5, 6, 7], [0, 1, 8, 9])
    assert tasks.choose_cpus(lscpu, 48 * 2**30, 8)[0] == [2, 3, 4, 5]  # (48 - 16) / 8 = 4 jobs
    assert tasks.choose_cpus(lscpu, 128 * 2**30, 8, jobs=2)[0] == [2, 3]

    # a task whose limits cannot be enforced is rejected when written
    for bad in ({"grace_s": 40.0}, {"limit_s": 0.0}, {"mem_cap_gib": -1.0}):
        try:
            tasks.write_pass1(directory, [_task("bad", [], **bad)])
        except ValueError:
            pass
        else:
            raise AssertionError(f"a task with {bad} was written")
    assert not (directory / tasks.TASKS_FILE).exists()

    # a task whose memory cap exceeds a job slot's is refused
    tasks.check_memory_caps([_task("fits", [], mem_cap_gib=8.0)], 8.0)
    try:
        tasks.check_memory_caps([_task("too_big", [], mem_cap_gib=9.0)], 8.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a task with too large a memory cap passed")


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
    from experiments.state_prep_certification.common import selftest_plugin

    def bad_repeats(task: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
        return [dict(repeat, grace_s=40.0) for repeat in selftest_plugin.repeat_tasks(task, row)]

    try:
        tasks.write_pass2(directory, final, bad_repeats)
    except ValueError:
        pass
    else:
        raise AssertionError("a repeat with too long a grace was written")
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
    """SIGTERM stops every run without rows; a resume reruns only the stopped tasks."""
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
    """run_node.py refuses without approval, then runs both passes and writes an archive."""
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
    final, problems = store.final_rows(label_dir)
    pass1, pass2 = tasks.pass_keys(label_dir, "1"), tasks.pass_keys(label_dir, "2")
    assert len(pass1) == 3 and len(pass2) == 6 and set(final) == set(pass1 + pass2), problems
    assert list((label_dir / "archive").glob("*.tar.gz"))


@test
def test_cleanup(directory: pathlib.Path, node: bool) -> None:
    """After a runner is killed hard, its run is live until --cleanup kills it."""
    run_node = _run_node_module()

    tasks.write_pass1(directory, [_task("hard", [["spawn", 51.25]], limit_s=60.0)])
    process = subprocess.Popen(_run_task_command(directory, "hard", node))
    _wait_for(lambda: (directory / "active" / "hard").exists())
    time.sleep(2)
    process.kill()
    process.wait()
    assert _processes_matching("time.sleep(51.25)")
    assert run_node.live_runs(directory) == ["hard"]
    run_node.cleanup(directory)
    time.sleep(1)
    assert not _processes_matching("time.sleep(51.25)"), "a run survived --cleanup"
    assert not list((directory / "active").iterdir())
    assert run_node.live_runs(directory) == []


@test
def test_cleanup_reused_ids(directory: pathlib.Path, node: bool) -> None:
    """--cleanup never kills a process that has reused a recorded id, even as a group leader."""
    from experiments.state_prep_certification.common import runner

    run_node = _run_node_module()
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
        _patched(runner, "process_table", lambda: table),
        _patched(run_node.os, "killpg", lambda pid, sig: killed.append(("group", pid))),
        _patched(run_node.os, "kill", lambda pid, sig: killed.append(("process", pid))),
    ):
        assert run_node.live_runs(directory) == []
        run_node.cleanup(directory)
        assert not killed, killed

        # a group whose leader is gone, with members left, is killed
        table = {reused + 1: runner.Process(1, reused, 0, None, "member")}
        (directory / "active" / "task").write_text(json.dumps(record))
        assert run_node.live_runs(directory) == ["task"]
        run_node.cleanup(directory)
        assert ("group", reused) in killed, killed


@test
def test_runner_failure(directory: pathlib.Path, node: bool) -> None:
    """A runner that fails kills its run, removes its active file, and writes no row."""
    tasks.write_pass1(directory, [_task("broken", [["garbage", 52.25]], limit_s=60.0)])
    result = subprocess.run(
        _run_task_command(directory, "broken", node), capture_output=True, check=False
    )
    assert result.returncode != 0
    assert not (directory / "active" / "broken").exists()
    assert store.read_rows(directory) == ([], [])
    time.sleep(1)
    assert not _processes_matching("broken"), "the failed runner's run survived"


@test
def test_run_node_refusals(directory: pathlib.Path, node: bool) -> None:
    """run_node.py refuses live runs, a local acceptance pass for node mode, and trial bundles."""
    from experiments.state_prep_certification.common import manifest, runner

    run_node = _run_node_module()

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
    with _patched(run_node.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0)):
        assert run_node.run_acceptance(module, record_dir, directory, node=False)
        assert run_node.run_acceptance(module, record_dir, directory, node=False)
        assert run_node.run_acceptance(module, record_dir, directory, node=True)
        assert run_node.run_acceptance(module, record_dir, directory, node=False)
    assert calls == [False, True], calls

    # a trial bundle, without a pinned qLDPC commit
    record = directory / "bundle.json"
    record.write_text(json.dumps({"unfrozen": True}))
    with _patched(manifest, "BUNDLE_RECORD", record):
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
def test_acceptance_attempts(directory: pathlib.Path, node: bool) -> None:
    """A second acceptance attempt runs its tasks again, in a fresh directory, keeping the first."""
    from experiments.state_prep_certification.common import local_runs

    run_node = _run_node_module()
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


@test
def test_manifest_label(directory: pathlib.Path, node: bool) -> None:
    """A label covers the pins, frozen files, harness commit, and a bundle's requirements."""
    from experiments.state_prep_certification.common import manifest

    frozen = directory / "frozen.json"
    frozen.write_text("{}")
    pins = directory / "pins.json"
    pins.write_text("{}")
    record = directory / "bundle.json"
    labels = set()
    with _patched(manifest, "BUNDLE_RECORD", record), _patched(manifest, "PINS_FILE", pins):
        labels.add(manifest.compute_label("exp", [frozen], "c1"))
        labels.add(manifest.compute_label("exp", [frozen], "c2"))
        for digest in ("aa", "bb"):
            record.write_text(json.dumps({"requirements_sha256": digest}))
            labels.add(manifest.compute_label("exp", [frozen], "c1"))
        frozen.write_text('{"changed": 1}')
        labels.add(manifest.compute_label("exp", [frozen], "c1"))
        pins.write_text('{"changed": 1}')
        labels.add(manifest.compute_label("exp", [frozen], "c1"))
    assert len(labels) == 6, labels


####################################################################################################
# bundle


@test
def test_bundle_requirements(directory: pathlib.Path, node: bool) -> None:
    """Compiled requirements split into entries with their hashes, comments dropped."""
    from experiments.state_prep_certification import bundle

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


@workstation_only
@test
def test_bundle_check_tree(directory: pathlib.Path, node: bool) -> None:
    """A bundle needs HEAD to descend from the pinned commit with src/ and pyproject unchanged."""
    from experiments.state_prep_certification import bundle

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
    previous_root = bundle.ROOT
    bundle.ROOT = directory
    try:
        assert bundle.check_tree({"qldpc_commit": base}, [], unfrozen=False) == git(
            "rev-parse", "HEAD"
        )
        (directory / "src" / "a.py").write_text("a = 2\n")
        git("commit", "-qam", "change src")
        _expect_exit(lambda: bundle.check_tree({"qldpc_commit": base}, [], unfrozen=False))
        git("checkout", "-q", "--orphan", "other")
        git("commit", "-qm", "unrelated")
        _expect_exit(lambda: bundle.check_tree({"qldpc_commit": base}, [], unfrozen=False))
        _expect_exit(lambda: bundle.check_tree({"qldpc_commit": None}, [], unfrozen=False))
        bundle.check_tree({"qldpc_commit": None}, [], unfrozen=True)
    finally:
        bundle.ROOT = previous_root


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


def _run_task_command(directory: pathlib.Path, key: str, node: bool) -> list[str]:
    """The command that runs one task, in local or node mode."""
    command = [sys.executable, str(RUN_TASK), "--label-dir", str(directory), key]
    return command if node else command + ["--local"]


def _run(directory: pathlib.Path, task: dict[str, Any], node: bool) -> dict[str, Any]:
    """Run one task through ``run_task.py`` and return its final row."""
    tasks.write_pass1(directory, [task])
    status = subprocess.run(_run_task_command(directory, task["key"], node), check=False)
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
    return subprocess.Popen(command, cwd=ROOT, stdout=subprocess.DEVNULL)


def _processes_matching(text: str) -> list[str]:
    """The command lines of live processes that contain ``text``."""
    output = subprocess.run(
        ["ps", "-A", "-o", "command="], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in output.splitlines() if text in line and "ps -A" not in line]


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


@contextlib.contextmanager
def _environment(name: str, value: str) -> Iterator[None]:
    """Set an environment variable, which the runs that the block starts inherit, for the block."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            del os.environ[name]
        else:
            os.environ[name] = previous


@contextlib.contextmanager
def _patched(owner: Any, name: str, value: Any) -> Iterator[None]:
    """Replace an attribute for the block."""
    previous = getattr(owner, name)
    setattr(owner, name, value)
    try:
        yield
    finally:
        setattr(owner, name, previous)


def _run_node_module() -> Any:
    """The module of run_node.py."""
    if str(RUN_NODE.parent) not in sys.path:
        sys.path.insert(0, str(RUN_NODE.parent))
    import run_node

    return run_node


def _require_parallel() -> None:
    """Fail if GNU parallel is not installed."""
    if shutil.which("parallel") is None:
        raise RuntimeError("GNU parallel is not installed")


def _expect_exit(function: Callable[[], Any]) -> None:
    """Check that a function exits with an error message."""
    try:
        function()
    except SystemExit as error:
        assert error.code not in (None, 0)
    else:
        raise AssertionError("no error")


if __name__ == "__main__":
    main()
