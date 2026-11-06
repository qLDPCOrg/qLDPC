#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Run the Ruff lint check over the code base."""

import sys

import checks_superstaq

if __name__ == "__main__":
    sys.exit(checks_superstaq.lint_.run(*sys.argv[1:]))
