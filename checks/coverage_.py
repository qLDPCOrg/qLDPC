#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check that every module is fully covered by its own test file, in one parallel pytest run.

Each module ``x.py`` with a test file ``x_test.py`` must be fully covered by ``x_test.py`` alone,
and each test file must be fully covered itself.  Rather than run every test file in its own
process, which repeats the imports and other setup of every process for each test file, this check
runs the test files together in one pytest-xdist run, keeping the tests of each test file on one
worker.
The ``coverage_contexts`` plugin labels each line that runs with the test file responsible for it,
and resets shared state between test files (see that module).  This check credits a module only
with the lines that its own test file ran: in its tests, or while it and the ``conftest.py`` files
of its directories were collected.  It also credits every test file with the lines that ran while
pytest started, which loads the ``conftest.py`` files and imports the measured modules.

``--isolated`` instead runs each test file in its own process, as an audit of this check.
"""

import io
import os
import pathlib
import re
import subprocess
import sys
import textwrap
from collections.abc import Sequence

import checks_superstaq
import coverage
import coverage_contexts
import pytest_

# Globs of the files that are neither checked nor run as tests.
EXCLUDE = pytest_.EXCLUDE

# Options for pytest.  The check runs only *.py tests, not notebooks, so it does not need nbmake.
PYTEST_OPTIONS = ("-pno:nbmake",)

check_utils = checks_superstaq.check_utils


def run(*args: str) -> int:
    """Run the check with command-line arguments, and return its exit code."""
    parser = check_utils.get_check_parser()
    parser.description = textwrap.dedent(
        """
        Check that every module is fully covered by its own test file, in one parallel pytest run.
        Unrecognized arguments are passed to pytest.
        """
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="The number of pytest-xdist workers, or 0 to run in this process."
        "  Default: half of the CPUs, but no more than the number of test files.",
    )
    parser.add_argument(
        "--branch", action="store_true", help="Also require every branch to be covered."
    )
    parser.add_argument(
        "--isolated",
        action="store_true",
        help="Run each test file in its own process instead (slower), as an audit of this check.",
    )
    parsed_args, pytest_args = parser.parse_known_intermixed_args(args)
    if "coverage" in parsed_args.skip:
        return 0
    if parsed_args.isolated:
        modular_args = [arg for arg in args if arg != "--isolated"]
        return checks_superstaq.coverage_.run(
            *modular_args, "--modular", "--sysmon", *PYTEST_OPTIONS, exclude=EXCLUDE
        )

    files = check_utils.extract_files(parsed_args, "*.py", EXCLUDE)
    covered_files = get_covered_files(files)
    if not covered_files:
        print("No test files to check for pytest and coverage.")
        return 0

    jobs = parsed_args.jobs
    if jobs is None:
        jobs = min((os.cpu_count() or 2) // 2, len(covered_files))
    data_file = os.path.join(check_utils.root_dir, ".coverage")
    coverage.Coverage(data_file=data_file).erase()
    test_files = sorted(covered_files)
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(filter(None, [_CHECKS, os.environ.get("PYTHONPATH")])),
        coverage_contexts.DATA_FILE_VARIABLE: data_file,
        coverage_contexts.INCLUDE_VARIABLE: os.pathsep.join(
            os.path.join(check_utils.root_dir, file)
            for files_to_cover in covered_files.values()
            for file in files_to_cover
        ),
        coverage_contexts.TEST_FILES_VARIABLE: os.pathsep.join(
            os.path.join(check_utils.root_dir, file) for file in test_files
        ),
        coverage_contexts.BRANCH_VARIABLE: "1" if parsed_args.branch else "",
    }
    # Pytest loads the conftest.py files of the paths that it is given before it collects tests, so
    # it is given only the directory that contains every test file, whose conftest.py files every
    # test file needs.  The plugin restricts collection to the test files.
    root = os.path.commonpath([os.path.dirname(file) for file in test_files]) or os.curdir
    command = [sys.executable, "-m", "pytest", root, f"-n={jobs}"]
    command += ["--dist=loadfile"] if jobs else []
    command += ["-p", "coverage_contexts", *PYTEST_OPTIONS, *pytest_args]
    test_returncode = subprocess.call(command, cwd=check_utils.root_dir, env=environment)

    measurement = coverage.Coverage(data_file=data_file, data_suffix=False)
    measurement.combine(keep=False)
    measurement.save()
    uncovered = [
        report
        for test_file, files_to_cover in sorted(covered_files.items())
        if (report := get_missing_coverage(measurement, test_file, files_to_cover))
    ]
    for report in uncovered:
        print(report, end="")

    if test_returncode or uncovered:
        print(check_utils.failure("TEST AND COVERAGE FAILURE!"))
        return test_returncode or 1
    print(check_utils.success("TEST AND COVERAGE SUCCESS!"))
    return 0


def get_covered_files(files: Sequence[str]) -> dict[str, list[str]]:
    """Map each test file to the files that it must cover: its module, if any, and itself."""
    covered_files: dict[str, list[str]] = {}
    for file in files:
        test_files = check_utils.get_test_files([file], exclude=EXCLUDE, silent=True)
        for test_file in test_files:
            covered_files.setdefault(test_file, [])
            if file not in covered_files[test_file]:
                covered_files[test_file].append(file)
    return covered_files


def get_context_patterns(test_file: str) -> list[str]:
    """Regular expressions that match the coverage contexts credited to a test file."""
    # Pytest node IDs, which name the contexts, separate directories with "/" on every platform.
    path = pathlib.PurePath(test_file)
    node_id = path.as_posix()
    directories = [parent.as_posix() for parent in reversed(path.parents)]
    contexts = ["", *(f"{directory}|collect" for directory in [*directories[1:], node_id])]
    return [
        f"^({'|'.join(map(re.escape, contexts))})$",
        rf"^{re.escape(node_id)}::.*\|(setup|run|teardown)$",
    ]


def get_missing_coverage(
    measurement: coverage.Coverage,
    test_file: str,
    files_to_cover: Sequence[str],
) -> str:
    """A coverage report of files that a test file does not fully cover, or "" if it does."""
    report = io.StringIO()
    patterns = get_context_patterns(test_file)
    try:
        percentage = measurement.report(
            morfs=files_to_cover, contexts=patterns, show_missing=True, file=report
        )
    except coverage.exceptions.NoDataError:
        return f"\nNo coverage data for {', '.join(files_to_cover)}, run by {test_file}.\n"
    if percentage == 100:
        return ""
    return f"\nCoverage by {test_file}:\n{report.getvalue()}"


# The directory of the check scripts, from which pytest processes import coverage_contexts.
_CHECKS = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    pytest_.limit_numba_threads()
    sys.exit(run(*sys.argv[1:]))
