#!/usr/bin/env python3
import sys

import checks_superstaq

EXCLUDE = ["experiments/run_search.py"]

if __name__ == "__main__":
    exit(checks_superstaq.mypy_.run(*sys.argv[1:], exclude=EXCLUDE))
