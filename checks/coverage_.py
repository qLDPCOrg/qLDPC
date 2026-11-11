#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Check that every module is fully covered by its own test file, in one parallel pytest run.

Each module ``x.py`` with a test file ``x_test.py`` must be fully covered by ``x_test.py`` alone,
and each test file must be fully covered itself.  A module under ``src/`` without a test file of its
own fails the check, unless it is a ``conftest.py`` file or matches ``EXCLUDE``.

Rather than run every test file in its own process, which repeats the imports and other setup of
every process for each test file, this check runs the test files together in one pytest-xdist run,
keeping the tests of each test file on one worker.  The ``coverage_contexts`` plugin labels each
line that runs with the test file responsible for it, and resets shared state between test files
(see that module).  This check credits a module only with the lines that its own test file ran: in
its tests, or while it and the ``conftest.py`` files of its directories were collected.  It also
credits every test file with the lines that ran while pytest started, or while a module of the
repository first ran, which every test file needs.  The check skips the tests marked ``compiled``,
which run compiled code that coverage cannot trace.

``--isolated`` instead runs each test file in its own process, as an audit of the coverage that this
check credits.  It runs only from the root directory of the repository.  If a selected module needs
a test file and has none, it fails without running the audit.

``--shard K/N`` checks only the K-th of N shards of the test files, so that N machines can share the
check.  Every shard reports every selected module without a test file, since such modules are in no
shard.
"""

import argparse
import contextlib
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
# It skips the tests marked ``compiled`` (see the module docstring).  The marker expression has no
# space, since --isolated passes these options through an argument parser that reads an argument
# containing a space as a file.  A caller's own -m option replaces it, if written without a space;
# -mcompiled selects only the tests marked compiled, which skips the tests that coverage needs.
PYTEST_OPTIONS = ("-pno:nbmake", "-mnot(compiled)")

check_utils = checks_superstaq.check_utils


def run(*args: str) -> int:
    """Run the check with command-line arguments, and return its exit code."""
    parser = check_utils.get_check_parser()
    parser.description = textwrap.dedent(
        """
        Check that every module is fully covered by its own test file, in one parallel pytest run.
        Unrecognized arguments are passed to pytest.  Tests marked compiled are skipped.  To replace
        this selection, pass a marker expression in one argument without spaces, as in
        -mcompiled, which also skips the tests that coverage needs.
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
        help="Run each test file in its own process instead (slower),"
        " as an audit of the coverage that this check credits."
        "  Runs only from the root directory of the repository.",
    )
    parser.add_argument(
        "--shard",
        type=parse_shard,
        default=None,
        metavar="K/N",
        help="Check only the K-th of N shards of the test files,"
        " so that N machines can share the check.",
    )
    parsed_args, pytest_args = parser.parse_known_intermixed_args(args)
    if "coverage" in parsed_args.skip:
        return 0
    if any("::" in file for file in parsed_args.files):
        parser.error(
            "the check takes paths, not test node IDs; to pass a pytest option whose value is"
            " a node ID, write it as --option=value"
        )
    if parsed_args.isolated:
        if parsed_args.shard is not None:
            parser.error("--shard does not apply to --isolated")
        # checks_superstaq looks for test files relative to the current directory, so from any other
        # directory it would find none, and pass.
        if not os.path.samefile(os.getcwd(), check_utils.root_dir):
            parser.error("--isolated runs only from the root directory of the repository")
        # checks_superstaq skips modules without a test file, so fail those that need one here,
        # without running the audit.
        if untested := get_untested_files(
            check_utils.extract_files(parsed_args, "*.py", EXCLUDE, silent=True)
        ):
            return report_results(0, [], untested)
        modular_args = [arg for arg in args if arg != "--isolated"]
        return checks_superstaq.coverage_.run(
            *PYTEST_OPTIONS, *modular_args, "--modular", "--sysmon", exclude=EXCLUDE
        )

    files = check_utils.extract_files(parsed_args, "*.py", EXCLUDE)

    # The selected files are relative to the root directory, where their test files are looked for.
    with contextlib.chdir(check_utils.root_dir):
        # Modules without a test file are found among all of the selected files, so that every
        # shard reports them.
        untested = get_untested_files(files)
        if parsed_args.shard is not None:
            files = get_shard(files, *parsed_args.shard)
        covered_files = get_covered_files(files)
        if not covered_files and not untested:
            print("No test files to check for pytest and coverage.")
            return 0
        test_returncode, coverage_reports = check_test_files(
            covered_files, parsed_args.jobs, parsed_args.branch, pytest_args
        )
    return report_results(test_returncode, coverage_reports, untested)


def get_untested_files(files: Sequence[str]) -> list[str]:
    """The selected modules under ``src/``, other than ``conftest.py`` files, without a test file.

    Files that match ``EXCLUDE``, which are selected only when they are given explicitly, need no
    test file.
    """
    return [
        file
        for file in check_utils.exclude_files(files, EXCLUDE)
        if pathlib.PurePath(file).parts[0] == "src"
        and os.path.basename(file) != "conftest.py"
        and not check_utils.get_test_files([file], exclude=EXCLUDE, silent=True)
    ]


def get_covered_files(files: Sequence[str]) -> dict[str, list[str]]:
    """Map each test file to the selected files that it must cover.

    These are its module, if any, and itself.
    """
    covered_files: dict[str, list[str]] = {}
    for file in files:
        test_files = check_utils.get_test_files([file], exclude=EXCLUDE, silent=True)
        for test_file in test_files:
            covered_files.setdefault(test_file, [])
            if file not in covered_files[test_file]:
                covered_files[test_file].append(file)
    return covered_files


def parse_shard(value: str) -> tuple[int, int]:
    """Parse a shard "K/N" of the test files into the pair (K, N), with 1 <= K <= N."""
    match = re.fullmatch(r"(\d+)/(\d+)", value)
    if match is None or not 1 <= int(match[1]) <= int(match[2]):
        raise argparse.ArgumentTypeError(f"expected K/N with 1 <= K <= N, not {value!r}")
    return int(match[1]), int(match[2])


def get_shard(files: Sequence[str], shard_number: int, num_shards: int) -> list[str]:
    """The selected files in shard ``shard_number`` of ``num_shards``, numbered from 1.

    A shard holds test files, each with the files that it must cover, so that the shards together
    check what the unsharded check does.  Files without a test file are in no shard.  The size of a
    test file stands in for the time that it takes: each test file, from largest to smallest, joins
    the shard with the smallest total size so far.  The shards depend only on the names and sizes of
    the test files of the selected files, so every machine with the same checkout computes the same
    shards.  Call it from the root directory of the repository, to which the paths of the files are
    relative.
    """
    covered_files = get_covered_files(files)
    file_sizes = {test_file: os.path.getsize(test_file) for test_file in covered_files}
    shard_sizes = [0] * num_shards
    shards: list[list[str]] = [[] for _ in range(num_shards)]
    for test_file in sorted(
        covered_files, key=lambda test_file: (-file_sizes[test_file], test_file)
    ):
        index = shard_sizes.index(min(shard_sizes))
        shard_sizes[index] += file_sizes[test_file]
        shards[index].append(test_file)
    if sorted(test_file for test_files in shards for test_file in test_files) != sorted(
        covered_files
    ):
        raise RuntimeError("The shards do not hold every test file exactly once.")
    shard_files = {
        file for test_file in shards[shard_number - 1] for file in covered_files[test_file]
    }
    return [file for file in files if file in shard_files]


def check_test_files(
    covered_files: dict[str, list[str]],
    jobs: int | None,
    branch: bool,
    pytest_args: Sequence[str],
) -> tuple[int, list[str]]:
    """Run test files in one pytest run, and check that each fully covers the files that it must.

    Return the exit code of pytest, and a coverage report of each test file that does not fully
    cover its files.
    """
    if not covered_files:
        return 0, []
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
        coverage_contexts.BRANCH_VARIABLE: "1" if branch else "",
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
    coverage_reports = [
        report
        for test_file, files_to_cover in sorted(covered_files.items())
        if (report := get_missing_coverage(measurement, test_file, files_to_cover))
    ]
    return test_returncode, coverage_reports


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


def report_results(
    test_returncode: int, coverage_reports: Sequence[str], untested: Sequence[str]
) -> int:
    """Print the coverage reports and the modules without a test file, and return the exit code."""
    for coverage_report in coverage_reports:
        print(coverage_report, end="")
    for file in untested:
        print(f"\nNo test file for {file}: add {file.removesuffix('.py')}_test.py.")
    if test_returncode or coverage_reports or untested:
        print(check_utils.failure("TEST AND COVERAGE FAILURE!"))
        return test_returncode or 1
    print(check_utils.success("TEST AND COVERAGE SUCCESS!"))
    return 0


# The directory of the check scripts, from which pytest processes import coverage_contexts.
_CHECKS = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    pytest_.limit_numba_threads()
    sys.exit(run(*sys.argv[1:]))
