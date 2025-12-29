#!/usr/bin/env python3
import sys

import checks_superstaq

# TODO: remove
EXCLUDE = (
    "examples/scripts/find_bbcode_layouts.py",  # causes problems in github tests...
)

if __name__ == "__main__":
    exit(checks_superstaq.mypy_.run(*sys.argv[1:], exclude=EXCLUDE))
