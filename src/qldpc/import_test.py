# SPDX-License-Identifier: Apache-2.0

"""Check that qLDPC imports cleanly in the fresh interpreters that run_python starts."""

import pytest

from qldpc.conftest import run_python


def test_import_in_fresh_interpreter() -> None:
    """In a fresh interpreter, qLDPC imports cleanly and loads deferred modules only on use.

    The package imports most subpackages when it is first imported, so importing any qLDPC module
    runs this same import sequence.  It defers its subpackages circuits and experimental to first
    access, and the scipy modules special and sparse.csgraph to the functions that use them.  If a
    dependency starts importing one of these scipy modules itself, deferring that module saves
    nothing, so drop it from ``deferred`` and import it at the top of the modules that use it.  Each
    decoder
    integration, including those in the deferred subpackages, imports its optional backend only when
    a decoder is built.
    """
    code = """
import sys
import qldpc
deferred = ("qldpc.circuits", "qldpc.experimental", "scipy.special", "scipy.sparse.csgraph")
backends = ("frontier", "ldpc", "pymatching", "relay_bp", "tesseract_decoder")
loaded = [name for name in (*deferred, *backends) if name in sys.modules]
assert not loaded, loaded
assert {"circuits", "experimental"} <= set(dir(qldpc))
assert not hasattr(qldpc, "not_a_subpackage")
assert qldpc.circuits is sys.modules["qldpc.circuits"]
assert qldpc.experimental is sys.modules["qldpc.experimental"]
from qldpc import *
assert circuits is sys.modules["qldpc.circuits"]
assert experimental is sys.modules["qldpc.experimental"]
loaded = [name for name in backends if name in sys.modules]
assert not loaded, loaded
"""
    run_python(code, "-W", "error")


def test_run_python_runs_assertions(monkeypatch: pytest.MonkeyPatch) -> None:
    """The interpreters that ``run_python`` starts run assertions even if PYTHONOPTIMIZE is set."""
    monkeypatch.setenv("PYTHONOPTIMIZE", "1")
    with pytest.raises(AssertionError, match="AssertionError"):
        run_python("assert False")
