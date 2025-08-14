#!/usr/bin/env python3
import sys

import checks_superstaq
import pytest_

if __name__ == "__main__":
    exit(
        checks_superstaq.coverage_.run(
            *sys.argv[1:], "--modular", "--sysmon", exclude=pytest_.EXCLUDE
        )
    )
