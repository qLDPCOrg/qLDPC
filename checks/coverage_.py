#!/usr/bin/env python3
import sys
from typing import Any
from unittest import mock

import checks_superstaq
import pytest_

# The slowest test files, slowest first.  These run first, so that they do not start late and hold
# up the end of the (parallel, modular) coverage check.  The order of all other files is unchanged.
# This list need not be complete or current -- missing files are ignored -- but it is worth
# refreshing when the check gets slow, using the "passed in ..." time printed for each test file.
SLOW_TESTS = (
    "src/qldpc/experimental/surgery/circuit_test.py",
    "src/qldpc/codes/quantum_test.py",
    "src/qldpc/codes/code_capacity_test.py",
    "src/qldpc/codes/classical_test.py",
    "src/qldpc/circuits/memory/alpha_syndrome_test.py",
    "src/qldpc/codes/common_test.py",
)


def _slowness_rank(file: str) -> int:
    """Position of a file, or of its test file, in SLOW_TESTS (or len(SLOW_TESTS) if absent)."""
    for name in (file, file.removesuffix(".py") + "_test.py"):
        if name in SLOW_TESTS:
            return SLOW_TESTS.index(name)
    return len(SLOW_TESTS)


def _extract_files_slow_first(*args: Any, **kwargs: Any) -> list[str]:
    """Collect files to check as checks_superstaq does, but with SLOW_TESTS and their sources first.

    checks_superstaq's modular coverage check queues its jobs in the order of these files (after a
    stable sort that moves test files behind other files), so this order sets the job order.
    """
    return sorted(_extract_files(*args, **kwargs), key=_slowness_rank)


_extract_files = checks_superstaq.check_utils.extract_files

if __name__ == "__main__":
    pytest_.limit_numba_threads()
    with mock.patch.object(
        checks_superstaq.check_utils, "extract_files", _extract_files_slow_first
    ):
        # The coverage check only runs *.py tests, not notebooks, so nbmake is unnecessary.
        returncode = checks_superstaq.coverage_.run(
            *sys.argv[1:], "--modular", "--sysmon", "-pno:nbmake", exclude=pytest_.EXCLUDE
        )
    sys.exit(returncode)
