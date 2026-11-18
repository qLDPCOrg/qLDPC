# SPDX-License-Identifier: Apache-2.0

"""Check the rules on code that threads run, which tests of results can miss.

Section 5 of ``brouwer_zimmermann.md`` and ``connected_cluster.md``, and the docstring of
``qldpc/_compilation.py``, state the rules.  An entry point compiled with ``nogil=True`` runs on
several threads at once, without Python's global interpreter lock, and so does every compiled
function that it calls.  A race between them would rarely change a result that tests compare, so
these tests read the code instead:

- Section 5 of the guide of its method names each such entry point of this package.  That section,
  with the docstring of ``_compilation.py``, states the rules of the entry point.
- No function of qLDPC is compiled with numba's ``parallel`` option.  ``always_compiled`` has no
  such option, and ``_compiled_code_rules_test.py`` keeps the functions of other packages, such as
  galois's, out of compiled code, so compiled code that threads run calls no numba-parallel code.

While every test of this package runs, its fixture ``check_code_on_threads`` of ``conftest.py``
checks that the Python code that threads run calls no numba-parallel code and reads the tuning only
from its search.  ``test_workers_that_call_numba_parallel_code_or_read_the_tuning_fail`` checks
that fixture, and ``test_read_only_arrays_are_restored_or_reported`` checks how
``shared_arrays_read_only`` of ``conftest.py`` restores the arrays that it marks.
"""

from __future__ import annotations

import ast
import inspect
import threading
import warnings
from collections.abc import Callable
from pathlib import Path

import numba
import numpy as np
import pytest

import qldpc
from qldpc import _compilation, conftest

from . import _threads, _tuning
from . import conftest as distance_conftest


def test_entry_points_without_the_lock_are_named_in_section_5_of_their_guides() -> None:
    """Section 5 of the guide of each entry point's method names it.

    The entry points are those of this package, whose guides state the rules on threads.  One of
    the Brouwer-Zimmermann or connected-cluster method is named in its method's guide, and any other
    in either guide.
    """
    guides = {
        name: _get_section_5(Path(__file__).with_name(f"{name}.md"))
        for name in ("brouwer_zimmermann", "connected_cluster")
    }
    entry_points = list(_get_entry_points())
    assert entry_points
    for module, name in entry_points:
        methods = [method for method in guides if f"._{method}." in module] or list(guides)
        assert any(f"`{name}`" in guides[method] for method in methods), (
            f"Section 5 of no guide of {methods} names {module}.{name}"
        )


def test_no_function_of_qldpc_is_compiled_numba_parallel() -> None:
    """qLDPC compiles no numba-parallel code, which threads must not call.

    Two threads that run such code at once abort the process under numba's workqueue layer.
    """
    for path, tree in _get_source_trees(Path(qldpc.__file__).parent):
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_njit(node.func):
                assert all(keyword.arg != "parallel" for keyword in node.keywords), (
                    f"{path.name} passes numba the parallel option"
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
    the block; otherwise it fails the check.
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
        with_line = inspect.currentframe().f_lineno + 1  # type: ignore[union-attr]
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


####################################################################################################
# helpers


def _get_source_trees(directory: Path) -> list[tuple[Path, ast.Module]]:
    """The path and syntax tree of every source file in a directory of qLDPC that is not a test."""
    return [
        (path, ast.parse(path.read_text(encoding="utf-8")))
        for path in _compilation._get_source_paths(directory)
    ]


def _get_entry_points() -> list[tuple[str, str]]:
    """The module and name of each function of this package compiled with ``nogil=True``."""
    package = Path(qldpc.__file__).parent
    entry_points = []
    for path, tree in _get_source_trees(Path(__file__).parent):
        for function in conftest.get_compiled_functions(tree)["always_compiled"]:
            if any(_releases_the_lock(decorator) for decorator in function.decorator_list):
                module = ".".join(path.relative_to(package.parent).with_suffix("").parts)
                entry_points.append((module, function.name))
    return entry_points


def _releases_the_lock(decorator: ast.expr) -> bool:
    """Whether a decorator compiles its function with ``nogil=True``."""
    return isinstance(decorator, ast.Call) and any(
        keyword.arg == "nogil" and getattr(keyword.value, "value", False) is True
        for keyword in decorator.keywords
    )


def _is_njit(function: ast.expr) -> bool:
    """Whether an expression is a name or an attribute called ``njit`` or ``jit``."""
    return getattr(function, "attr", getattr(function, "id", "")) in ("njit", "jit")


def _get_section_5(guide: Path) -> str:
    """The text of Section 5 of a guide."""
    text = guide.read_text(encoding="utf-8")
    return text.split("\n## 5.", 1)[1].split("\n## ", 1)[0]
