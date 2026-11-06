#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the strict mypy type check over source and tests."""

import sys

import checks_superstaq

if __name__ == "__main__":
    sys.exit(checks_superstaq.mypy_.run(*sys.argv[1:]))
