# SPDX-License-Identifier: Apache-2.0

"""Check that the qLDPC package imports cleanly."""

from qldpc.conftest import run_python


def test_import_in_fresh_interpreter() -> None:
    """In a fresh interpreter, qLDPC imports cleanly and does not import optional decoder backends.

    The package imports every subpackage when it is first imported, so importing any qLDPC module
    runs this same import sequence.  That includes every decoder integration, each of which imports
    its optional backend only when a decoder is built.
    """
    code = """
import sys
import qldpc
backends = ("frontier", "ldpc", "pymatching", "relay_bp", "tesseract_decoder")
assert not [backend for backend in backends if backend in sys.modules]
"""
    run_python(code, "-W", "error")
