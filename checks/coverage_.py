#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check that every module is fully covered by its own test file, in one parallel pytest run.

This runs the modular coverage check of checks-superstaq with ``--modular --single-run`` (see its
module ``coverage_``).  Each module ``x.py`` with a test file ``x_test.py`` must be fully covered by
``x_test.py`` alone, and each test file must be fully covered by itself, with the test files run
together in one pytest-xdist run.  The module ``_coverage_contexts`` of checks-superstaq says which
lines the check credits to each test file.  A module under ``src/`` without a test file of its own
also fails the check, unless it is a ``conftest.py`` file or matches ``EXCLUDE``, and every shard
reports such modules.  If everything else passed, the check then prints ``COVERAGE FAILURE!`` after
the success banner of checks-superstaq.

The check skips the tests marked ``compiled``, which run compiled code that coverage cannot trace.
It sets ``QLDPC_CLEAR_CACHES``, so that qLDPC's ``conftest.py`` empties qLDPC's caches after each
test file.

Unrecognized options are passed to pytest, and a ``-m`` option replaces the default marker
expression.  Write a pytest option and its value as one argument without spaces, as in ``-kname``,
since the check reads any other argument as a file.  Run ``python checks/coverage_.py -h`` for the
options of the check, which include:

* ``--no-single-run``, which runs each test file in its own process instead, as an audit of the
  coverage that the check credits;
* ``--shard K/N``, which checks only the K-th of N shards of the test files, balanced by the
  durations of test files in ``.test_durations``, so that N machines can share the check; and
* ``--store-durations --clean-durations``, which updates ``.test_durations``.  Use it only in a
  single run over the whole repository, without paths, ``-i``, ``-x``, ``-k``, ``--shard``, or
  ``--no-single-run``, and on a CI runner rather than locally, since the ratios of the durations
  differ between machines.

Until a release of checks-superstaq includes its single-run check, this check runs copies of its
modules in ``checks/_superstaq_coverage`` (see the README there).  To switch to such a release:

1. delete ``checks/_superstaq_coverage``;
2. in this file, replace ``from _superstaq_coverage import coverage_`` by ``from checks_superstaq
   import coverage_``, delete the line that sets ``coverage_._PLUGIN`` and the comment above it,
   and delete this paragraph and the list that it introduces; and
3. raise every minimum version of checks-superstaq in ``pyproject.toml`` to that release.
"""

import contextlib
import os
import pathlib
import sys
from collections.abc import Sequence

import checks_superstaq
import pytest_
from _superstaq_coverage import coverage_

check_utils = checks_superstaq.check_utils

# The pytest plugin of the copies, which the pytest processes import by this name.
coverage_._PLUGIN = "_superstaq_coverage._coverage_contexts"

# Globs of the files that are neither checked nor run as tests.
EXCLUDE = pytest_.EXCLUDE

# The environment variable that makes qLDPC's conftest.py empty qLDPC's functools caches after each
# test file, as a fresh process would start with them empty.
CLEAR_CACHES_VARIABLE = "QLDPC_CLEAR_CACHES"

# Options of the check, which come before the caller's arguments, so that the caller's own options
# replace them.  The check runs only *.py tests, not notebooks, so it does not need nbmake.
# --sysmon speeds up --no-single-run; a single run chooses its own coverage core.
OPTIONS = ("--modular", "--single-run", "--sysmon", "-pno:nbmake", "-mnot(compiled)")


def run(*args: str) -> int:
    """Run the check with command-line arguments, and return its exit code."""
    os.environ[CLEAR_CACHES_VARIABLE] = "1"
    # checks-superstaq parses the arguments first, so that it handles -h and usage errors.
    returncode = coverage_.run(*OPTIONS, *args, exclude=EXCLUDE)
    untested = get_untested_files(args)
    if not untested:
        return returncode
    for file in untested:
        print(f"\nNo test file for {file}: add {file.removesuffix('.py')}_test.py.")
    if not returncode:
        print(check_utils.failure("COVERAGE FAILURE!"))
    return returncode or 1


def get_untested_files(args: Sequence[str]) -> list[str]:
    """The selected modules under ``src/``, other than ``conftest.py`` files, without a test file.

    Files that match ``EXCLUDE``, which are selected only when they are given explicitly, need no
    test file.
    """
    parser = check_utils.get_check_parser()
    # The options of the coverage check that take a separate value, which is not a file.
    parser.add_argument("-j", "--jobs")
    parser.add_argument("--shard")
    parsed_args, _ = parser.parse_known_intermixed_args(args)
    if "coverage" in parsed_args.skip:
        return []
    # The files are found relative to the current directory, and their test files to the root.
    files = check_utils.extract_files(parsed_args, "*.py", EXCLUDE, silent=True)
    with contextlib.chdir(check_utils.root_dir):
        return [
            file
            for file in check_utils.exclude_files(files, EXCLUDE)
            if pathlib.PurePath(file).parts[0] == "src"
            and os.path.basename(file) != "conftest.py"
            and not check_utils.get_test_files([file], exclude=EXCLUDE, silent=True)
        ]


if __name__ == "__main__":
    pytest_.limit_numba_threads()
    sys.exit(run(*sys.argv[1:]))
