# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin that records where each test process spends its time.

Temporary, for the CI timing probe on the branch ci/timing-probe, which is never merged.  Each
process writes a JSON file to the directory in the environment variable PROFILE_OUT, with absolute
times of its phases, the wall time of each test, and each outermost numba compilation with the test
that triggered it.
"""

import json
import os
import pathlib
import resource
import time

import pytest
from numba.core import event

_record: dict[str, object] = {"imported": time.time(), "tests": [], "compiles": []}
_current = ["<startup>"]


class _Listener(event.Listener):
    """Record the time of each outermost numba compilation, with the test that triggered it."""

    def __init__(self) -> None:
        self.depth = 0
        self.start = 0.0
        self.name = ""

    def on_start(self, ev: event.Event) -> None:
        if self.depth == 0:
            self.start = time.perf_counter()
            function = ev.data["dispatcher"].py_func
            self.name = f"{function.__module__}.{function.__qualname__}"
        self.depth += 1

    def on_end(self, ev: event.Event) -> None:
        self.depth -= 1
        if self.depth == 0:
            elapsed = time.perf_counter() - self.start
            _record["compiles"].append((_current[0], self.name, elapsed))  # type: ignore[union-attr]


def pytest_configure(config: pytest.Config) -> None:
    _record["configured"] = time.time()
    event.register("numba:compile", _Listener())


def pytest_collection(session: pytest.Session) -> None:
    _record["collection_start"] = time.time()
    _current[0] = "<collection>"


def pytest_collection_finish(session: pytest.Session) -> None:
    _record["collection_end"] = time.time()
    _current[0] = "<between>"


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None):  # noqa: ANN201
    _current[0] = item.nodeid
    start = time.time()
    yield
    _record["tests"].append((item.nodeid, start, time.time()))  # type: ignore[union-attr]
    _current[0] = "<between>"


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    out = pathlib.Path(os.environ["PROFILE_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    _record["end"] = time.time()
    _record["cpu"] = usage.ru_utime + usage.ru_stime
    _record["max_rss_mb"] = usage.ru_maxrss / 1024
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    (out / f"{worker}.json").write_text(json.dumps(_record))
