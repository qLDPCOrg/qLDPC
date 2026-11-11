#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the modular coverage check of checks-superstaq with ``--modular --single-run``.

This wrapper also skips the tests marked ``compiled``, sets ``QLDPC_CLEAR_CACHES`` (see
``src/qldpc/conftest.py``), and fails the check for each selected module under ``src/``, other than
a ``conftest.py`` file, that has no test file.

Until a release of checks-superstaq includes ``--single-run``, this runs copies of its coverage
modules in ``checks/_superstaq_coverage``.  To switch to such a release:

1. delete ``checks/_superstaq_coverage``;
2. in this file, replace ``from _superstaq_coverage import coverage_`` with ``from checks_superstaq
   import coverage_``;
3. in this file, delete the line that sets ``coverage_._PLUGIN``, with its comment;
4. in this file, delete the paragraph that begins "Until a release", together with this list; and
5. raise every minimum version of checks-superstaq in ``pyproject.toml`` to that release.
"""

import contextlib
import os
import pathlib
import sys
from collections.abc import Sequence

import checks_superstaq
import pytest_
from _superstaq_coverage import coverage_

from qldpc.conftest import CLEAR_CACHES_VARIABLE

check_utils = checks_superstaq.check_utils

# The pytest plugin of the copies, which the pytest processes import by this name.
coverage_._PLUGIN = "_superstaq_coverage._coverage_contexts"

# Globs of the files that are neither checked nor run as tests.
EXCLUDE = pytest_.EXCLUDE

# Options of the check, which come before the caller's arguments, so that the caller's own options
# replace them.  The check runs only *.py tests, not notebooks, so it does not need nbmake.
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
