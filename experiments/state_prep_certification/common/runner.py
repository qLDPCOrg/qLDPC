# SPDX-License-Identifier: Apache-2.0

"""The task runner: one task, in one or two fresh child processes, with its limits and its row.

The parent side, ``run_task``, starts the child with the thread variables of ``THREAD_VARIABLES``
set to 1, and, if it is given a CPU, has the child pin itself to that CPU.  On the node, the
runner itself inherits the CPUs of the reserved cores from ``run_node.py``, so that its polling
of processes every second runs off the measured cores; rows record both affinities.  The runner
records the child's process group (or, in local mode, its process id) and the start times of the
child and the runner in ``active/<task_key>``, and watches the child:

* **Start-up cap:** the child must report that timing started within ``startup_cap_s`` of its
  launch (process start, imports, and any untimed preparation); otherwise it is killed and the
  run is an ``error``.
* **Limit:** the measured call gets ``limit_s`` from the moment that the child reports its start.
  The parent kills the child as soon as the limit has passed, and the run is a ``timeout``
  whenever the measured time T is at least the limit, even if the call returned later.  The T of a
  killed timeout is the time from the start to the kill, a censoring time.
* **After the measured call:** once it has ended, the child has ``post_s`` (by default
  ``FINISH_S``) to send its result and exit.  An experiment with untimed post-processing, such as
  describing a counterexample, sends its measured result first with ``send_result`` and declares a
  longer ``post_s``; running out of it, or failing in it, then marks only the row's
  ``post_status``, and the measured result stands.  Without ``send_result``, running out of
  ``post_s`` makes the run an ``error``.
* **Memory:** every second the parent sums the resident memory of the child's process group (or
  process tree), records the peak, and kills a run above ``mem_cap_gib`` as a ``memout``.
* **Kills:** a kill sends SIGKILL to every process of the run at once; the parent then reads the
  partial results and the report of a compilation that the child wrote before the kill, waits up
  to ``grace_s`` for the processes to go, and accepts no other message.  A process is killed only
  if its start time is the one recorded when it was seen, so that a reused process id is never
  killed.
* **Compilation:** while the measured call runs, the child listens to numba's documented event
  ``"numba:compile"``, which numba sends when it compiles a ``numba.jit`` function, and reports
  the first compilation to the parent.  numba does not send this event when it compiles a
  ``numba.vectorize`` or ``numba.guvectorize`` ufunc or a ``numba.cfunc``, such as galois's ufuncs
  (see ``certification/counting.py``), so such a compilation in a measured call goes unreported.

  The parent discards a run whose measured call compiled, whatever its status.  The discarded run is
  not stopped early, since numba writes compiled code to its cache only after the compilation ends,
  and a kill could lose it.  The parent then runs the task once more, in a fresh process that loads
  the compiled code from numba's disk cache: the retry, which is never discarded.  A run's T
  includes loading from the cache any compiled code that its process first reaches in its measured
  call, and the other first-call work of a fresh process.  A task takes at most ``RUNS_PER_TASK``
  runs, and its kept row records the discarded run, if any, in ``discarded_run``: its run id,
  status, T, error, and the function that it compiled.

  A row has ``compiled_in_timing`` set when its run is a retry that compiled too; only then does its
  T include a compilation that numba reports.  That happens for code whose compilation was lost when
  the discarded run was killed at its limit or memory cap or ended in an error, for code that the
  discarded run never reached, and for code that numba does not cache on disk, such as galois's
  ``numba.jit`` functions when a task does not compile them first.  It can also happen when
  processes compile different versions of one function at once, since one can overwrite the cache
  entry that another wrote (numba/numba#10862).  Such a row is kept so that it is visible.
  ``compiled_in_timing`` is the flag that the analyses test, and ``compiled_function`` names the
  first function compiled, to show what to look into.

``run_task`` appends exactly one final row to the store and returns 0, or, on SIGTERM or SIGINT
during or between its runs, kills any current run, waits up to the grace for its processes to go,
and returns 2 without a row; ``run_task.py`` exits with that status.  If the parent itself fails, it
kills the run before it exits.

The child side is ``child_main``.  The experiment's ``run(task, label_dir)`` calls ``timed`` once
around the measured call, may call ``send_partial`` with partial results as they arrive and
``send_result`` with the measured result before it starts post-processing, and returns the
remaining fields.  On the node, the child runs in a new session, so that its process group holds
the whole run; locally (``local=True``), it stays in the runner's session, so that a supervisor of
the calling session can account for it and stop it, and kills walk its process tree instead.  In
local mode, a process that the run leaves behind after a normal exit is neither killed nor
reported.
"""

from __future__ import annotations

import json
import os
import pathlib
import platform
import resource
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable
from typing import Any, NamedTuple, TypeVar

from experiments.state_prep_certification.common import store

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "QLDPC_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

STARTUP_CAP_S = 120.0
# The most runs of one task: a run, and a retry if the measured call of that run compiled.  Raising
# it would need discarded_run to record every discarded run, and the docs that describe one retry,
# such as tasks.py's statement of the ceiling, to change.
RUNS_PER_TASK = 2
GRACE_S = 30.0
FINISH_S = 30.0
POLL_S = 1.0

# How long a runner that receives SIGTERM has before GNU parallel kills it (see ``tasks.TERMSEQ``).
# It must cover the grace and the runner's own clean-up after it, for which ``check_task`` leaves
# ``_CLEANUP_S``.
TERM_WAIT_S = 35.0
_CLEANUP_S = 2.0

_CHANNEL_VARIABLE = "FTSP_CHANNEL_FD"
_CPU_VARIABLE = "FTSP_CPU"
_LOG_TAIL_BYTES = 1 << 16
_ReturnType = TypeVar("_ReturnType")


def check_task(task: dict[str, Any]) -> None:
    """Raise ValueError if a task's limits cannot be enforced, such as a grace that is too long."""
    grace = float(task.get("grace_s", GRACE_S))
    if grace > TERM_WAIT_S - _CLEANUP_S:
        raise ValueError(f"{task['key']}: grace_s must be at most {TERM_WAIT_S - _CLEANUP_S:g} s")
    for field in ("limit_s", "mem_cap_gib"):
        if float(task[field]) <= 0:
            raise ValueError(f"{task['key']}: {field} must be positive")


class Process(NamedTuple):
    """A row of the process table."""

    ppid: int
    pgid: int
    rss_bytes: int
    threads: int | None
    start: str


####################################################################################################
# parent side


def run_task(
    label_dir: pathlib.Path,
    task: dict[str, Any],
    *,
    local: bool,
    job_slot: int | None = None,
    cpu: int | None = None,
) -> int:
    """Run one task, in one or two fresh child processes, and append its final row.

    Args:
        label_dir: The label's directory, which holds ``tasks.json`` and ``runs.jsonl``.
        task: The task, with at least ``key``, ``experiment``, ``arm``, ``repeat``, ``limit_s``,
            and ``mem_cap_gib``, and optionally ``post_s``, ``startup_cap_s``, and ``grace_s``.
        local: Whether to keep the child in this session and kill by process tree.
        job_slot: GNU parallel's job slot of this task, if any.
        cpu: The logical CPU to which the child pins itself, if any (Linux only).

    Returns:
        0 once it has appended the row, or 2, without a row, if SIGTERM or SIGINT stopped the task.
    """
    command = [
        sys.executable,
        str(pathlib.Path(__file__).with_name("run_task.py")),
        "--child",
        "--label-dir",
        str(label_dir),
        task["key"],
    ]
    # A stop signal stops the current run, or the task between its runs.
    stop = _Stop()

    def on_signal(signum: int, frame: Any) -> None:
        stop.signum = signum

    previous_handlers = {
        sig: signal.signal(sig, on_signal) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        row: dict[str, Any] | None = None
        for _ in range(RUNS_PER_TASK):
            if stop.signum is not None:
                return 2
            watcher = _Watcher(label_dir, task, local=local, job_slot=job_slot, cpu=cpu, stop=stop)
            new_row = watcher.run(command)
            if new_row is None:
                return 2
            if row is not None:
                new_row["discarded_run"] = {
                    name: row[name]
                    for name in ("run_id", "status", "T", "error", "compiled_function")
                }
            row = new_row
            if not row["compiled_in_timing"]:
                break
        assert row is not None
        store.append_row(label_dir, row)
        return 0
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


class _Stop:
    """The stop signal that the runner received, if any, which every run of the task reads."""

    signum: int | None = None


class _Watcher:
    """The parent's view of one run: the child, its messages, its limits, and its row."""

    def __init__(
        self,
        label_dir: pathlib.Path,
        task: dict[str, Any],
        *,
        local: bool,
        job_slot: int | None,
        cpu: int | None,
        stop: _Stop,
    ) -> None:
        self.label_dir = label_dir
        self.task = task
        self.local = local
        self.job_slot = job_slot
        self.cpu = cpu
        self.limit_s = float(task["limit_s"])
        self.post_s = float(task.get("post_s", FINISH_S))
        self.startup_cap_s = float(task.get("startup_cap_s", STARTUP_CAP_S))
        check_task(task)
        self.grace_s = float(task.get("grace_s", GRACE_S))
        self.mem_cap_bytes = float(task["mem_cap_gib"]) * 2**30
        self.run_id = f"{task['key']}.{uuid.uuid4().hex[:8]}"
        self.partials: list[dict[str, Any]] = []
        self.start_info: dict[str, Any] | None = None
        self.measured: dict[str, Any] | None = None
        self.result: dict[str, Any] | None = None
        self.post: dict[str, Any] | None = None
        self.post_phase = False
        self.child_error: str | None = None
        self.compiled_function: str | None = None
        self.peak_rss = 0
        self.threads_peak: int | None = None
        self.seen: dict[int, str] = {}
        self.stop = stop
        self.root = -1
        self.group: int | None = None
        self.buffer = b""
        self.timing_started_at: float | None = None
        self.measured_at: float | None = None
        self.killed_at: float | None = None

    def run(self, command: list[str]) -> dict[str, Any] | None:
        """Start the child, watch it, and return its row, or None if a signal stopped it."""
        for name in ("active", "logs"):
            (self.label_dir / name).mkdir(exist_ok=True)
        active_file = self.label_dir / "active" / self.task["key"]
        log_file = self.label_dir / "logs" / f"{self.run_id}.txt"
        started_at = _utc_now()

        # start the child with a private channel for its messages
        read_fd, write_fd = os.pipe()
        env = dict(os.environ, **{name: "1" for name in THREAD_VARIABLES})
        env[_CHANNEL_VARIABLE] = str(write_fd)
        if self.cpu is not None:
            env[_CPU_VARIABLE] = str(self.cpu)
        with open(log_file, "wb") as log:
            child = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                pass_fds=(write_fd,),
                start_new_session=not self.local,
            )
        os.close(write_fd)
        self.root = child.pid
        self.group = os.getpgid(child.pid) if not self.local else None
        active_file.write_text(
            json.dumps(
                {
                    "pid": child.pid,
                    "pid_start": process_start(child.pid),
                    "pgid": self.group,
                    "runner_pid": os.getpid(),
                    "runner_start": process_start(os.getpid()),
                }
            )
        )

        # watch the child until it ends, or until a limit or a signal ends it; if this runner
        # fails, kill the run before failing
        try:
            kill_reason, rusage = self._watch(child, read_fd, time.monotonic())
            finished_at = _utc_now()
            stray = self._wait_until_gone(child) if kill_reason or not self.local else []
        except BaseException:
            self._kill()
            self._wait_until_gone(child)
            active_file.unlink(missing_ok=True)
            raise
        finally:
            os.close(read_fd)
        _keep_tail(log_file)
        if self.stop.signum is not None:
            active_file.unlink(missing_ok=True)
            return None

        # the run's row
        if rusage is not None:
            scale = 1 if sys.platform == "darwin" else 1024
            self.peak_rss = max(self.peak_rss, rusage.ru_maxrss * scale)
        if child.returncode and self.child_error is None:
            self.child_error = log_file.read_text(errors="replace")[-2000:].strip()
        row = self._row(kill_reason, child.returncode, started_at, finished_at)
        if stray:
            left = f"processes left after the grace: {stray}"
            row["error"] = f"{row['error']}; {left}" if row["error"] else left
        active_file.unlink(missing_ok=True)
        return row

    def _watch(
        self, child: subprocess.Popen[bytes], read_fd: int, launch: float
    ) -> tuple[str | None, resource.struct_rusage | None]:
        """Follow the child's messages and resources; return why it was killed, if it was."""
        next_poll = launch
        open_channel = True
        while True:
            # read the messages that have arrived
            if open_channel:
                open_channel = self._read(read_fd, 0.2)
            else:
                time.sleep(0.2)

            # has the child ended?  Then read what it wrote before it ended.
            pid, status, rusage = os.wait4(child.pid, os.WNOHANG)
            if pid:
                child.returncode = os.waitstatus_to_exitcode(status)
                while open_channel and self._read(read_fd, 0.0, drain=True):
                    pass
                return None, rusage

            # limits and signals
            now = time.monotonic()
            reason = None
            if self.stop.signum is not None:
                reason = "signal"
            elif self.timing_started_at is None and now - launch > self.startup_cap_s:
                reason = "startup"
            elif (
                self.timing_started_at is not None
                and self.measured_at is None
                and now - self.timing_started_at >= self.limit_s
            ):
                reason = "timeout"
            elif self.measured_at is not None and now - self.measured_at >= self.post_s:
                reason = "post_timeout"
            if reason is None and now >= next_poll:
                next_poll = now + POLL_S
                rss, threads = self._poll_resources()
                self.peak_rss = max(self.peak_rss, rss)
                if threads is not None:
                    self.threads_peak = max(self.threads_peak or 0, threads)
                if rss > self.mem_cap_bytes:
                    reason = "memout"
            if reason is not None:
                self._kill()
                self.killed_at = time.monotonic()
                while open_channel and self._read(read_fd, 0.0, drain=True, after_kill=True):
                    pass
                return reason, None

    def _read(
        self, read_fd: int, timeout: float, drain: bool = False, after_kill: bool = False
    ) -> bool:
        """Read and handle available messages; return False once the channel is closed.

        With ``drain``, return False also when nothing is waiting, so that a loop ends.  With
        ``after_kill``, handle only the partial results and the report of a compilation.
        """
        ready, _, _ = select.select([read_fd], [], [], timeout)
        if not ready:
            return not drain
        data = os.read(read_fd, 1 << 16)
        self.buffer += data
        *lines, self.buffer = self.buffer.split(b"\n")
        for line in lines:
            message = json.loads(line)
            if after_kill and message["type"] not in ("partial", "compiled"):
                continue
            now = time.monotonic()
            if message["type"] == "start":
                self.timing_started_at = now
            elif message["type"] == "measured":
                self.measured_at = now
            self._receive(message)
        return bool(data)

    def _receive(self, message: dict[str, Any]) -> None:
        """Record one message of the child."""
        kind = message["type"]
        if kind == "start":
            self.start_info = message
        elif kind == "measured":
            self.measured = message
        elif kind == "partial":
            self.partials.append(message["fields"])
        elif kind == "result":
            self.result = message["fields"]
            self.post_phase = not message.get("final", False)
        elif kind == "post":
            self.post = message["fields"]
        elif kind == "error":
            self.child_error = message["message"]
        elif kind == "compiled" and self.compiled_function is None:
            self.compiled_function = message["function"]

    def _row(
        self, kill_reason: str | None, returncode: int | None, started_at: str, finished_at: str
    ) -> dict[str, Any]:
        """The row of the run."""
        error: str | None = None
        post_status: str | None = None
        T: float | None = None
        T_cpu: float | None = None
        if self.measured is not None:
            T, T_cpu = float(self.measured["T"]), float(self.measured["T_cpu"])

        # the status, from the measured call first
        if kill_reason == "memout" and self.result is None:
            status = "memout"
        elif kill_reason == "startup":
            status, error = "error", f"start-up cap of {self.startup_cap_s:g} s exceeded"
        elif self.measured is None and kill_reason == "timeout":
            status = "timeout"
            assert self.timing_started_at is not None and self.killed_at is not None
            T = max(self.limit_s, self.killed_at - self.timing_started_at)
        elif T is not None and T >= self.limit_s:
            status = "timeout"
        elif self.measured is not None and self.result is not None:
            status = "ok"
        elif kill_reason == "post_timeout":
            status, error = "error", f"no result within {self.post_s:g} s of the measured call"
        else:
            status = "error"
            error = self.child_error or f"the child exited with status {returncode}"

        # the post-processing, which never changes an ok status
        if status == "ok" and self.post_phase:
            if self.post is not None:
                post_status = "ok"
            elif kill_reason in ("post_timeout", "memout"):
                post_status = "timeout" if kill_reason == "post_timeout" else "memout"
            else:
                post_status = "error"
                error = self.child_error or f"the child exited with status {returncode}"

        result = dict(self.result or {})
        result.update(self.post or {})
        start_info = self.start_info or {}
        return {
            "experiment": self.task["experiment"],
            "label": self.label_dir.name,
            "run_id": self.run_id,
            "task_key": self.task["key"],
            "arm": self.task["arm"],
            "mode": self.task.get("mode"),
            "repeat": self.task["repeat"],
            "status": status,
            "post_status": post_status,
            "T": T,
            "T_cpu": T_cpu,
            "peak_rss_gib": self.peak_rss / 2**30,
            "limit_s": self.limit_s,
            "host": socket.gethostname(),
            "cpu": _cpu_model(),
            "pinned_cpu": self.cpu,
            "affinity": start_info.get("affinity"),
            "runner_affinity": _affinity(),
            "job_slot": self.job_slot,
            "threads_peak": self.threads_peak,
            "local": self.local,
            "error": error,
            "compiled_in_timing": self.compiled_function is not None,
            "compiled_function": self.compiled_function,
            "discarded_run": None,
            "started_at": started_at,
            "finished_at": finished_at,
            "partial": self.partials,
            "result": result,
        }

    def _members(self, table: dict[int, Process]) -> list[int]:
        """The live processes of the run: its process group, or locally its process tree."""
        if self.group is not None:
            members = [pid for pid, process in table.items() if process.pgid == self.group]
        else:
            members = descendants(table, self.root)
        for pid in members:
            self.seen.setdefault(pid, table[pid].start)
        return members

    def _poll_resources(self) -> tuple[int, int | None]:
        """The summed resident memory of the run in bytes, and its thread count, if known."""
        table = process_table()
        members = self._members(table)
        rss = sum(table[pid].rss_bytes for pid in members)
        threads = [table[pid].threads for pid in members]
        known = [count for count in threads if count is not None]
        return rss, (sum(known) if len(known) == len(threads) else None)

    def _kill(self) -> None:
        """Kill every process of the run that is still the process that was seen."""
        if self.group is not None:
            try:
                os.killpg(self.group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return
        table = process_table()
        self._members(table)
        for pid, start in self.seen.items():
            if pid in table and table[pid].start == start:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def _wait_until_gone(self, child: subprocess.Popen[bytes]) -> list[int]:
        """Reap the child and wait up to the grace for the run's processes; return any left."""
        deadline = time.monotonic() + self.grace_s
        while True:
            if child.returncode is None:
                pid, status = os.waitpid(child.pid, os.WNOHANG)
                if pid:
                    child.returncode = os.waitstatus_to_exitcode(status)
            table = process_table()
            if self.group is not None:
                alive = self._members(table)
            else:
                alive = sorted(
                    pid
                    for pid, start in self.seen.items()
                    if table.get(pid, None) is not None
                    and table[pid].start == start
                    and pid != self.root
                )
            if not alive and child.returncode is not None:
                return []
            if time.monotonic() >= deadline:
                return alive
            self._kill()
            time.sleep(0.2)


####################################################################################################
# child side


class _Channel:
    """The child's end of the message channel to its runner."""

    timing_started = False
    result_sent = False
    wall_start: float | None = None
    measured: float | None = None

    def __init__(self) -> None:
        fd = os.environ.get(_CHANNEL_VARIABLE)
        self.fd = int(fd) if fd is not None else None
        # numba may compile on a thread other than the one that sends the other messages
        self.lock = threading.Lock()

    def send(self, message: dict[str, Any]) -> None:
        """Send one message, as a line of JSON."""
        if self.fd is None:
            return
        data = (json.dumps(message, separators=(",", ":"), default=_jsonable) + "\n").encode()
        view = memoryview(data)
        with self.lock:
            while view:
                view = view[os.write(self.fd, view) :]


_CHANNEL = _Channel()


def timed(function: Callable[..., _ReturnType], *args: Any, **kwargs: Any) -> _ReturnType:
    """Call the measured function once, timing it, and tell the runner when it starts and ends.

    T is the wall time by ``time.perf_counter``, and the CPU time includes child processes.  While
    the function runs, the first compilation that numba reports in this process, on any thread, is
    reported to the runner (see the module docstring).
    """
    # numba is imported here, in the child, so that the runner's own process does not load it
    from numba.core import event

    if _CHANNEL.timing_started:
        raise RuntimeError("timed(...) may be called only once per run")
    _CHANNEL.timing_started = True
    _CHANNEL.send({"type": "start", "affinity": _affinity()})
    cpu_start = _cpu_seconds()
    reporter = _new_compile_reporter()
    with event.install_listener("numba:compile", reporter):
        _CHANNEL.wall_start = time.perf_counter()
        value = function(*args, **kwargs)
        wall = time.perf_counter() - _CHANNEL.wall_start
    cpu = _cpu_seconds() - cpu_start
    _CHANNEL.measured = wall
    _CHANNEL.send({"type": "measured", "T": wall, "T_cpu": cpu})
    return value


def elapsed() -> float | None:
    """The wall time since the measured call started, or None before it starts."""
    if _CHANNEL.wall_start is None:
        return None
    return time.perf_counter() - _CHANNEL.wall_start


def measured_seconds() -> float:
    """The wall time T of the finished measured call."""
    if _CHANNEL.measured is None:
        raise RuntimeError("No timed(...) call has finished")
    return _CHANNEL.measured


def send_partial(fields: dict[str, Any]) -> None:
    """Forward a partial result, such as the counts of a finished layer, to the runner."""
    _CHANNEL.send({"type": "partial", "fields": fields})


def send_result(fields: dict[str, Any]) -> None:
    """Send the measured result before untimed post-processing starts."""
    if _CHANNEL.measured is None:
        raise RuntimeError("send_result(...) needs a finished timed(...) call")
    _CHANNEL.result_sent = True
    _CHANNEL.send({"type": "result", "fields": fields})


def pin_child() -> None:
    """Pin this child process, and so its descendants, to the runner's CPU, if it gave one."""
    cpu = os.environ.get(_CPU_VARIABLE)
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {int(cpu)})


def child_main(
    task: dict[str, Any],
    label_dir: pathlib.Path,
    run: Callable[[dict[str, Any], pathlib.Path], dict[str, Any]],
) -> int:
    """Run an experiment's ``run`` in this child process, and send what it returns.

    An exception ends the child with its traceback on stderr, which the runner reports.
    """
    fields = run(task, label_dir)
    if not _CHANNEL.timing_started:
        _CHANNEL.send({"type": "error", "message": "the run made no timed(...) call"})
        return 1
    if _CHANNEL.result_sent:
        _CHANNEL.send({"type": "post", "fields": fields})
    else:
        _CHANNEL.send({"type": "result", "fields": fields, "final": True})
    return 0


def _new_compile_reporter() -> Any:
    """A listener of numba's event ``"numba:compile"`` that reports the first compilation.

    numba sends this event for the compilation of a ``numba.jit`` function (see the module
    docstring for what it does not report).  The listener's class is defined here, on call, since
    it subclasses numba's ``event.Listener``, and only the child imports numba.
    """
    from numba.core import event

    class CompileReporter(event.Listener):
        """Report the first compilation that starts, with the name of its function."""

        reported = False

        def on_start(self, started: event.Event) -> None:
            if self.reported:
                return
            self.reported = True
            function = started.data["dispatcher"].py_func
            name = f"{function.__module__}.{function.__qualname__}"
            _CHANNEL.send({"type": "compiled", "function": name})

        def on_end(self, ended: event.Event) -> None:
            pass

    return CompileReporter()


####################################################################################################
# processes


def process_table() -> dict[int, Process]:
    """Every process, by id, with its parent, group, resident bytes, threads, and start time.

    The thread count is known on Linux only.
    """
    linux = sys.platform.startswith("linux")
    fields = "pid=,ppid=,pgid=,rss=" + (",nlwp=" if linux else "") + ",lstart="
    output = subprocess.run(
        ["ps", "-A", "-o", fields], capture_output=True, text=True, check=True
    ).stdout
    table: dict[int, Process] = {}
    numbers = 5 if linux else 4
    for line in output.splitlines():
        parts = line.split(None, numbers)
        if len(parts) <= numbers:
            continue
        threads = int(parts[4]) if linux else None
        table[int(parts[0])] = Process(
            int(parts[1]), int(parts[2]), 1024 * int(parts[3]), threads, parts[numbers].strip()
        )
    return table


def process_start(pid: int) -> str | None:
    """The start time of a live process, as ``ps`` reports it, or None if it is gone."""
    process = process_table().get(pid)
    return None if process is None else process.start


def descendants(table: dict[int, Process], root: int) -> list[int]:
    """A process and its live descendants."""
    children: dict[int, list[int]] = {}
    for pid, process in table.items():
        children.setdefault(process.ppid, []).append(pid)
    members, stack = [], [root] if root in table else []
    while stack:
        pid = stack.pop()
        members.append(pid)
        stack.extend(children.get(pid, []))
    return members


####################################################################################################
# helpers


def _affinity() -> list[int] | None:
    """The CPUs that this process may run on, where the system reports them (Linux)."""
    if sys.platform.startswith("linux"):
        return sorted(os.sched_getaffinity(0))
    return None


def _cpu_seconds() -> float:
    """The CPU time of this process and its waited-for children, in seconds."""
    times = os.times()
    return times.user + times.system + times.children_user + times.children_system


def _cpu_model() -> str:
    """The model name of this machine's CPU."""
    try:
        if sys.platform.startswith("linux"):
            for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
    except OSError:
        pass
    return platform.processor()


def _keep_tail(path: pathlib.Path) -> None:
    """Keep only the last bytes of a log file."""
    if path.stat().st_size > _LOG_TAIL_BYTES:
        with open(path, "rb") as file:
            file.seek(-_LOG_TAIL_BYTES, os.SEEK_END)
            tail = file.read()
        path.write_bytes(b"[...]\n" + tail)


def _utc_now() -> str:
    """The current time in UTC, in ISO 8601 format."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _jsonable(value: Any) -> Any:
    """Convert NumPy scalars and arrays, and other simple objects, for JSON."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return str(value)
