# SPDX-License-Identifier: Apache-2.0

"""Check that the qLDPC package imports cleanly."""

from qldpc.conftest import run_python


def test_import_in_fresh_interpreter() -> None:
    """In a fresh interpreter, qLDPC imports cleanly and does not import modules that it defers.

    The package imports most subpackages when it is first imported, so importing any qLDPC module
    runs this same import sequence.  It defers its subpackages circuits and experimental to first
    access, and the scipy modules special and sparse.csgraph to the functions that use them.  If a
    dependency starts importing one of these scipy modules itself, drop it from the list, since
    deferring it then saves nothing.  Each decoder integration, including those in the deferred
    subpackages, imports its optional backend only when a decoder is built.
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
from qldpc import *
assert circuits is sys.modules["qldpc.circuits"]
assert experimental is sys.modules["qldpc.experimental"]
loaded = [name for name in backends if name in sys.modules]
assert not loaded, loaded
"""
    run_python(code, "-W", "error")
