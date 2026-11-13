#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run every repository check: formatting, lint, mypy, tests, coverage, and the docs build."""

import sys
from unittest import mock

import build_docs
import checks_superstaq
import coverage_
import pytest_


def run_coverage(*args: str, exit_on_failure: bool = False, silent: bool = False) -> int:
    """Run checks/coverage_.py, the coverage check of CI, as the combined check's coverage step.

    The combined check of checks_superstaq passes ``exit_on_failure`` and ``silent`` to its coverage
    step.  checks/coverage_.py takes neither, so this drops them: a coverage failure does not stop
    the combined check early, and a run on files or with -i prints the files that it checks.
    """
    return coverage_.run(*args)


if __name__ == "__main__":
    pytest_.limit_numba_threads()
    build_docs.ensure_sphinx_on_path()
    build_docs.ensure_pandoc_on_path()

    skip_files = ["--sysmon", "--skip=configs", "--skip=requirements"]
    with mock.patch.object(checks_superstaq.coverage_, "run", run_coverage):
        check_result = checks_superstaq.all_.run(*skip_files, *sys.argv[1:])

    # Only a full run builds the docs, so only a full run checks rendered API pages for docstrings
    # that Sphinx rendered wrongly (see build_docs.validate_rendered_docstrings).  After a run on
    # files or with -i, these pages would be missing or stale.
    full_run = not any(
        arg in ("-i", "--incremental") or not arg.startswith("-") for arg in sys.argv[1:]
    )
    sys.exit(check_result or (build_docs.validate_rendered_docstrings() if full_run else 0))
