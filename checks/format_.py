#!/usr/bin/env python3
"""Format the code base: ruff format + import sorting (checks-superstaq), then pyproject-fmt.

Pass --check to report without modifying files (used by CI); otherwise fixes are applied in place.
"""

import subprocess
import sys

import checks_superstaq

if __name__ == "__main__":
    args = sys.argv[1:]
    # "--check" reports formatting issues without modifying files (e.g. for CI).
    check = ["--check"] if "--check" in args else []

    returncode = checks_superstaq.format_.run(*args)
    returncode |= subprocess.run(
        [sys.executable, "-m", "pyproject_fmt", *check, "pyproject.toml"], check=False
    ).returncode

    sys.exit(returncode)
