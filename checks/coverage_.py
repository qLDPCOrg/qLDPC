#!/usr/bin/env python3
import sys

import checks_superstaq
import pytest_

if __name__ == "__main__":
    # The coverage check only runs *.py tests, never notebooks, so the nbmake plugin is pure
    # overhead here; disabling it saves ~0.7s of pytest startup in each modular-coverage subprocess.
    # "-pno:nbmake" is the attached spelling of "-p no:nbmake"; the separate "no:nbmake" token would
    # otherwise be parsed as a positional filename by checks_superstaq's argument parser.
    sys.exit(
        checks_superstaq.coverage_.run(
            *sys.argv[1:], "--modular", "--sysmon", "-pno:nbmake", exclude=pytest_.EXCLUDE
        )
    )
