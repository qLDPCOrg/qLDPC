#!/usr/bin/env python3
import sys

import _common
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

if __name__ == "__main__":
    _common.reject_unmatched_file_arguments(sys.argv[1:])
    sys.exit(checks_superstaq.pytest_.run(*sys.argv[1:], exclude=EXCLUDE))
