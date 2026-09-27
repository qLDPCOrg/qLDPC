#!/usr/bin/env python3
import sys

import checks_superstaq
import pytest_

if __name__ == "__main__":
    pytest_.reject_unmatched_file_arguments(sys.argv[1:])
    # The coverage check only runs *.py tests, not notebooks, so the nbmake plugin is unnecessary.
    sys.exit(
        checks_superstaq.coverage_.run(
            *sys.argv[1:], "--modular", "--sysmon", "-pno:nbmake", exclude=pytest_.EXCLUDE
        )
    )
