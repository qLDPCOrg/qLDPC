#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the pytest suite, excluding non-test scripts and duplicate notebook links."""

import os
import sys

import checks_superstaq

EXCLUDE = (
    "*/__init__.py",
    "checks/*.py",
    "examples/*.py",
    "experiments/*.py",
    "docs/source/conf.py",
    # The notebooks under docs/source/examples are symlinks to their examples/ originals, so
    # collecting them would run every notebook twice -- and would fail besides, because the modules
    # that some notebooks import from their own directory have no counterpart in the docs tree.
    "docs/source/examples/*",
)


def limit_numba_threads() -> None:
    """Keep each test process to one thread unless the caller already chose a number.

    galois runs some kernels (such as matrix multiplication) in Numba's parallel mode, which by
    default starts one thread per CPU in every process.  Tests run in many concurrent processes and
    mostly multiply small matrices, so those threads only oversubscribe the CPU.  qLDPC's exact
    searches take their default number of threads from ``QLDPC_NUM_THREADS`` if it is set, so that
    defaults to the same number.
    """
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    os.environ.setdefault("QLDPC_NUM_THREADS", os.environ["NUMBA_NUM_THREADS"])


if __name__ == "__main__":
    limit_numba_threads()
    sys.exit(checks_superstaq.pytest_.run(*sys.argv[1:], exclude=EXCLUDE))
