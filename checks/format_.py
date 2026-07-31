#!/usr/bin/env python3
"""Format the code base.

Runs the standard checks-superstaq formatter (ruff format + import sorting), then docformatter
(which wraps docstring summaries and descriptions to the 100-column limit), then pyproject-fmt
(which normalizes and sorts pyproject.toml).
"""

import subprocess
import sys

import checks_superstaq

# docformatter targets and options (docstring formatting / wrapping).
DOCFORMATTER_PATHS = ["src/qldpc"]
DOCFORMATTER_OPTIONS = ["--wrap-summaries", "100", "--wrap-descriptions", "100"]

if __name__ == "__main__":
    args = sys.argv[1:]
    check = "--check" in args
    returncode = checks_superstaq.format_.run(*args)

    # In "--check" mode, report without modifying files; otherwise apply fixes in place.
    docformatter = subprocess.run(
        [
            sys.executable,
            # docformatter 1.5.0 imports the deprecated lib2to3; silence its DeprecationWarning.
            "-W",
            "ignore::DeprecationWarning",
            "-m",
            "docformatter",
            "--recursive",
            "--check" if check else "--in-place",
            *DOCFORMATTER_OPTIONS,
            *DOCFORMATTER_PATHS,
        ],
        check=False,
    )
    pyproject_fmt = subprocess.run(
        ["pyproject-fmt", *(["--check"] if check else []), "pyproject.toml"],
        check=False,
    )

    # Both tools exit non-zero when files need reformatting; surface that in "--check" mode.
    if check:
        returncode = returncode or docformatter.returncode or pyproject_fmt.returncode

    sys.exit(returncode)
