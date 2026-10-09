#!/usr/bin/env python3
import sys

import build_docs
import checks_superstaq
import pytest_

if __name__ == "__main__":
    pytest_.limit_numba_threads()
    # The docs build runs here too, so the pandoc that nbsphinx needs has to be discoverable.
    build_docs.ensure_sphinx_on_path()
    build_docs.ensure_pandoc_on_path()
    skip_files = ["--sysmon", "--skip", "configs", "requirements"]
    check_result = checks_superstaq.all_.run(*skip_files, *sys.argv[1:])
    sys.exit(check_result or build_docs.validate_rendered_docstrings())
