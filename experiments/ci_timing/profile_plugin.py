# SPDX-License-Identifier: Apache-2.0

"""Pytest plugin that records each test process's wall time, CPU time, and numba compile time.

Temporary, for the CI timing experiment on the branch ci/timing-experiment, which is never merged.
Each process writes a JSON file to the directory in the environment variable PROFILE_OUT.  If
QLDPC_EXPERIMENT_EAGER is "1", each process also compiles the set-up code of searches before its
first test, as the compiled distance run on main does.
"""

import json
import os
import pathlib
import resource
import time

import pytest
from numba.core import event

_T0 = time.time()
_state: dict[str, object] = {"test": "<startup>", "first_test": None, "eager_done": False}
_compiles: list[tuple[str, str, float]] = []


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
            _compiles.append((str(_state["test"]), self.name, time.perf_counter() - self.start))


def pytest_configure(config: pytest.Config) -> None:
    event.register("numba:compile", _Listener())


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    if _state["first_test"] is None:
        _state["first_test"] = time.time() - _T0
    if os.environ.get("QLDPC_EXPERIMENT_EAGER") == "1" and not _state["eager_done"]:
        from qldpc.codes.distance import compile_search_setup

        _state["eager_done"] = True
        compile_search_setup()
    _state["test"] = item.nodeid


def pytest_sessionfinish(session: pytest.Session) -> None:
    out = pathlib.Path(os.environ["PROFILE_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    usage = resource.getrusage(resource.RUSAGE_SELF)
    worker = os.environ.get("PYTEST_XDIST_WORKER", "controller")
    record = {
        "wall": time.time() - _T0,
        "first_test": _state["first_test"],
        "cpu": usage.ru_utime + usage.ru_stime,
        "compiles": _compiles,
    }
    (out / f"{worker}.json").write_text(json.dumps(record))
