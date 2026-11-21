# SPDX-License-Identifier: Apache-2.0

"""Check the rules on code that threads run, which tests of results can miss.

Section 5 of ``brouwer_zimmermann_code.md`` and ``connected_cluster.md``, and the docstring of
``qldpc/_compilation.py``, state the rules.  An entry point compiled with ``nogil=True`` runs on
several threads at once, without Python's global interpreter lock, and so does every compiled
function that it calls.  A race between these threads would rarely change a result that tests
compare.  So a test here finds each such entry point of this package in the code, and checks that
Section 5 of its method's guide names it.  ``test_nothing_asks_numba_for_parallel_code`` of
``qldpc/_compiled_code_rules_test.py`` keeps qLDPC's own compiled code from being numba-parallel.

While every test of this package runs, its fixture ``check_code_on_threads`` of ``conftest.py``
checks that the Python code that threads run calls no numba-parallel code and reads the tuning only
from its search.  ``test_workers_that_call_numba_parallel_code_or_read_the_tuning_fail`` checks
that fixture, and ``test_read_only_arrays_are_restored_or_reported`` checks how
``shared_arrays_read_only`` of ``conftest.py`` restores the arrays that it marks.
"""

from __future__ import annotations

import inspect
import threading
import warnings
from collections.abc import Callable
from pathlib import Path

import numba
import numpy as np
import pytest

from qldpc import _compilation, conftest

from . import _threads, _tuning
from . import conftest as distance_conftest


def test_entry_points_without_the_lock_are_named_in_section_5_of_their_guides() -> None:
    """Section 5 of the guide of each entry point's method names it.

    The entry points are those of this package, whose guides state the rules on threads.  An entry
    point in ``_brouwer_zimmermann/`` or ``_connected_cluster/`` is named in its method's guide, and
    any other in either guide.
    """
    guides = {
        method: _get_section_5(Path(__file__).with_name(guide))
        for method, guide in (
            ("brouwer_zimmermann", "brouwer_zimmermann_code.md"),
            ("connected_cluster", "connected_cluster.md"),
        )
    }
    entry_points = conftest.get_entry_points_without_the_lock(Path(__file__).parent)
    assert entry_points
    for module, name in entry_points:
        methods = [method for method in guides if f"._{method}." in module] or list(guides)
        assert any(f"`{name}`" in guides[method] for method in methods), (
            f"Section 5 of no guide of {methods} names {module}.{name}"
        )


@pytest.mark.skipif(
    bool(_compilation._numba_config.DISABLE_JIT), reason="NUMBA_DISABLE_JIT makes no dispatchers"
)
def test_workers_that_call_numba_parallel_code_or_read_the_tuning_fail() -> None:
    """The check of ``conftest.py`` fails a worker of ``run_work_items`` that breaks these rules.

    Workers run on the calling thread and on a new one.  The calling thread reads the tuning before
    and after them, and the check of numba's dispatchers is removed once the workers return.
    """
    # If the check of dispatchers broke, numba would compile and run this function on the workers,
    # and the test would fail.
    parallel = numba.njit(parallel=True)(lambda array: array)
    own_call = vars(numba.core.dispatcher.Dispatcher).get("__call__")

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        with pytest.raises(AssertionError, match="get_tuning"):
            _tuning.get_tuning()

        def run_range(start: int, end: int) -> None:
            assert distance_conftest._CHECKED_RUNS[0] == 1
            with pytest.raises(AssertionError, match="numba-parallel"):
                parallel(np.zeros(1))
            with pytest.raises(AssertionError, match="get_tuning"):
                _tuning.get_tuning()

        return run_range

    tuning = _tuning.get_tuning()
    assert distance_conftest._CHECKED_RUNS[0] == 0
    _threads.run_work_items(4, 2, start_worker)
    assert distance_conftest._CHECKED_RUNS[0] == 0
    assert _tuning.get_tuning() is tuning
    assert vars(numba.core.dispatcher.Dispatcher).get("__call__") is own_call


def test_read_only_arrays_are_restored_or_reported() -> None:
    """``shared_arrays_read_only`` restores bases before views, and reports what it cannot restore.

    A view cannot be made writeable again once code makes its base, which the block does not reach,
    read-only.  If the block raises, that gives a warning at the with statement, and the error of
    the block; otherwise it fails the check.  An object of qLDPC without a ``__dict__`` is an error,
    since the arrays that it reaches cannot be found.
    """
    base = np.zeros(4)
    arrays = [base[1:], base, base[:2]]
    with distance_conftest.shared_arrays_read_only(arrays) as marked:
        assert len(marked) == 3
        assert not any(array.flags.writeable for array in arrays)
    assert all(array.flags.writeable for array in arrays)

    with_line = 0

    def hide_a_base(error: bool) -> None:
        """Make read-only the base of a marked view within the block, and raise if asked to."""
        nonlocal with_line
        hidden_base = np.zeros(4)
        frame = inspect.currentframe()
        assert frame is not None
        with_line = frame.f_lineno + 1
        with distance_conftest.shared_arrays_read_only([hidden_base[:2]]):
            hidden_base.flags.writeable = False
            if error:
                raise KeyError

    with pytest.warns(RuntimeWarning, match="left read-only") as record, pytest.raises(KeyError):
        hide_a_base(error=True)
    assert (record[0].filename, record[0].lineno) == (__file__, with_line)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(AssertionError, match="left read-only"):
            hide_a_base(error=False)

    slotted = type("Slotted", (), {"__slots__": (), "__module__": "qldpc"})()
    with pytest.raises(TypeError, match=r"roots\[0\] has no __dict__"):
        distance_conftest.shared_arrays_read_only(slotted).__enter__()


####################################################################################################
# helpers


def _get_section_5(guide: Path) -> str:
    """The text of Section 5 of a guide."""
    text = guide.read_text(encoding="utf-8")
    return text.split("\n## 5.", 1)[1].split("\n## ", 1)[0]
