#!/usr/bin/env python3
"""Format the code base.

Runs the standard checks-superstaq formatter (ruff format + import sorting), then pyproject-fmt
(which normalizes and sorts pyproject.toml).
"""

import subprocess
import sys

import checks_superstaq

if __name__ == "__main__":
    args = sys.argv[1:]
    # "--check" puts every formatter into report-only mode: each one reports what it would reformat
    # and exits non-zero if anything needs changing, but no files are modified.  This is the mode CI
    # uses to verify that the tree is already formatted.  Without "--check", the formatters apply
    # their fixes in place.
    check = "--check" in args
    returncode = checks_superstaq.format_.run(*args)

    # pyproject-fmt normalizes and sorts pyproject.toml.
    pyproject_fmt = subprocess.run(
        ["pyproject-fmt", *(["--check"] if check else []), "pyproject.toml"],
        check=False,
    )

    # pyproject-fmt exits non-zero when the file needs reformatting; surface that in "--check" mode.
    if check:
        returncode = returncode or pyproject_fmt.returncode

    sys.exit(returncode)
