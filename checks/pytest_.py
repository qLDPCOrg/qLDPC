#!/usr/bin/env python3
import sys

import checks_superstaq

EXCLUDE = (
    "*/__init__.py",
    "checks/*.py",
    "examples/*.py",
    "experiments/*.py",
    "docs/source/conf.py",
)

if __name__ == "__main__":
    exit(checks_superstaq.pytest_.run(*sys.argv[1:], exclude=EXCLUDE))
