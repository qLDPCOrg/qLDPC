# SPDX-License-Identifier: Apache-2.0

"""Tests of the environment variable QLDPC_NUM_THREADS, which sets the threads of searches."""

from __future__ import annotations

import json
import types
from typing import Any

import pytest

from qldpc import _num_threads
from qldpc.conftest import run_python

# Code for a child interpreter that imports qLDPC, recording its warnings, and prints the number of
# threads that searches use by default, numba's thread count, and the warnings.
_REPORT = """
import json, warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    {setup}
    import qldpc
    from qldpc.codes.distance import _threads
import numba
print(json.dumps([
    _threads.get_num_threads(None),
    numba.config.NUMBA_NUM_THREADS,
    [str(warning.message) for warning in caught if issubclass(warning.category, RuntimeWarning)],
]))
"""


def _fake_numba(num_threads: int) -> types.SimpleNamespace:
    """A stand-in for an imported numba module, with its thread count."""
    return types.SimpleNamespace(config=types.SimpleNamespace(NUMBA_NUM_THREADS=num_threads))


def test_an_unset_variable_changes_nothing() -> None:
    environ = {"NUMBA_NUM_THREADS": "2"}
    assert _num_threads.read_num_threads(environ, {}) is None
    assert environ == {"NUMBA_NUM_THREADS": "2"}


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "two", "", " 2", "²"])
def test_values_other_than_positive_integers_are_rejected(value: str) -> None:
    with pytest.raises(ValueError, match="QLDPC_NUM_THREADS must be a positive integer"):
        _num_threads.read_num_threads({"QLDPC_NUM_THREADS": value}, {})


def test_the_variable_sets_numba_thread_count_before_numba_is_imported() -> None:
    """The variable sets NUMBA_NUM_THREADS, and warns if it replaces another value."""
    environ = {"QLDPC_NUM_THREADS": "3"}
    assert _num_threads.read_num_threads(environ, {}) == 3
    assert environ["NUMBA_NUM_THREADS"] == "3"

    # The same value replaces nothing.
    assert _num_threads.read_num_threads(environ, {}) == 3

    environ = {"QLDPC_NUM_THREADS": "3", "NUMBA_NUM_THREADS": "5"}
    with pytest.warns(RuntimeWarning, match="QLDPC_NUM_THREADS=3 replaces NUMBA_NUM_THREADS=5"):
        assert _num_threads.read_num_threads(environ, {}) == 3
    assert environ["NUMBA_NUM_THREADS"] == "3"


def test_a_numba_imported_before_qldpc_keeps_its_count_with_a_warning() -> None:
    environ = {"QLDPC_NUM_THREADS": "3", "NUMBA_NUM_THREADS": "5"}
    with pytest.warns(RuntimeWarning, match="numba was imported before qLDPC, with 5 threads"):
        assert _num_threads.read_num_threads(environ, {"numba": _fake_numba(5)}) == 3
    assert environ["NUMBA_NUM_THREADS"] == "5"

    # Numba's own count agrees with the variable, so there is nothing to warn about.
    assert _num_threads.read_num_threads(environ, {"numba": _fake_numba(3)}) == 3


def test_imports_with_and_without_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """In new interpreters, searches use the variable if it is set, and NUMBA_NUM_THREADS if not.

    Each case reports the default number of threads of searches, numba's thread count, and the
    warnings of the import.
    """
    monkeypatch.delenv("QLDPC_NUM_THREADS", raising=False)
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)

    def report(setup: str = "pass", **env: str) -> list[Any]:
        return json.loads(run_python(_REPORT.format(setup=setup), env=env))

    assert report(NUMBA_NUM_THREADS="2") == [2, 2, []]
    assert report(QLDPC_NUM_THREADS="3") == [3, 3, []]
    assert report(QLDPC_NUM_THREADS="3", NUMBA_NUM_THREADS="3") == [3, 3, []]
    assert report(QLDPC_NUM_THREADS="3", NUMBA_NUM_THREADS="2") == [
        3,
        3,
        ["QLDPC_NUM_THREADS=3 replaces NUMBA_NUM_THREADS=2"],
    ]
    searches, numba_threads, (warning,) = report(
        "import numba", QLDPC_NUM_THREADS="3", NUMBA_NUM_THREADS="2"
    )
    assert (searches, numba_threads) == (3, 2)
    assert "numba was imported before qLDPC, with 2 threads" in str(warning)


def test_an_invalid_variable_stops_the_import(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QLDPC_NUM_THREADS", "0")
    code = "try:\n    import qldpc\nexcept ValueError as error:\n    print(error)"
    assert "QLDPC_NUM_THREADS must be a positive integer" in run_python(code)
