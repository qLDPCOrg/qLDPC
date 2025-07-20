#!/usr/bin/env python3
import sys

import checks_superstaq

if __name__ == "__main__":
    exit(checks_superstaq.lint_.run(*sys.argv[1:]))
