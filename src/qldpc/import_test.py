# SPDX-License-Identifier: Apache-2.0

"""Check that the qLDPC package imports cleanly."""

from qldpc.conftest import run_python


def test_import_in_fresh_interpreter() -> None:
    """qLDPC imports without errors or warnings in a fresh interpreter.

    The package imports every subpackage when it is first imported, so importing any qLDPC module
    runs this same import sequence.
    """
    run_python("import qldpc", "-W", "error")
