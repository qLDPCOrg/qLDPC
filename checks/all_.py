#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run every repository check: formatting, lint, mypy, tests, coverage, and the docs build."""

import sys

import build_docs
import checks_superstaq
import pytest_

if __name__ == "__main__":
    pytest_.limit_numba_threads()
    build_docs.ensure_sphinx_on_path()
    build_docs.ensure_pandoc_on_path()

    skip_files = ["--sysmon", "--skip=configs", "--skip=requirements"]
    check_result = checks_superstaq.all_.run(*skip_files, *sys.argv[1:])

    # Only a full run builds the docs, so only a full run checks rendered API pages for docstrings
    # that Sphinx rendered wrongly (see build_docs.validate_rendered_docstrings).  After a run on
    # files or with -i, these pages would be missing or stale.
    full_run = not any(
        arg in ("-i", "--incremental") or not arg.startswith("-") for arg in sys.argv[1:]
    )
    sys.exit(check_result or (build_docs.validate_rendered_docstrings() if full_run else 0))
