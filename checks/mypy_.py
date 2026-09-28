#!/usr/bin/env python3
import sys

import checks_superstaq

if __name__ == "__main__":
    sys.exit(checks_superstaq.mypy_.run(*sys.argv[1:]))
