#!/usr/bin/env python3
# Copyright 2026 Infleqtion, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import concurrent.futures
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap
from collections.abc import Iterable

import coverage

from checks_superstaq import check_utils

# The pytest plugin that labels coverage with test files for --single-run.
_PLUGIN = "checks_superstaq._coverage_contexts"

# Environment variables read by _PLUGIN (which duplicates these names).
_DATA_FILE_ENV = "CHECKS_SUPERSTAQ_COVERAGE_DATA_FILE"
_INCLUDE_ENV = "CHECKS_SUPERSTAQ_COVERAGE_INCLUDE"
_BRANCH_ENV = "CHECKS_SUPERSTAQ_COVERAGE_BRANCH"

# pytest-split's file of test durations (in the root directory), by which --shard balances shards.
_DURATIONS_FILE = ".test_durations"


@check_utils.enable_exit_on_failure
def run(
    *args: str,
    include: str | Iterable[str] = "*.py",
    exclude: str | Iterable[str] = "*_integration_test.py",
    silent: bool = False,
) -> int:
    """Checks to make sure that all code is covered by unit tests.

    Args:
        *args: Command line arguments.
        include: Glob(s) indicating which tracked files to consider (e.g. "*.py").
        exclude: Glob(s) indicating which tracked files to skip (e.g. "*integration_test.py").
        silent: If True, restrict printing to warning and error messages.

    Returns:
        Terminal exit code. 0 indicates success, while any other integer indicates a test failure.
    """
    parser = check_utils.get_check_parser()
    parser.add_argument(
        "--modular",
        action="store_true",
        help="Check that each file is covered by its own test file.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Parallelize modular coverage test across this many concurrent threads (with"
        " --single-run, the number of pytest-xdist workers, which overrides -n). This argument is"
        " ignored for non-modular coverage tests."
        " Default: min((os.cpu_count() or 2) // 2, len(files_being_tested))",
    )
    parser.add_argument(
        "--branch",
        action="store_true",
        help="Also require all branches to be covered (same as `coverage run --branch`).",
    )
    parser.add_argument(
        "--sysmon",
        action="store_true",
        help="Enable the `COVERAGE_CORE=sysmon` env variable for faster coverage (requires "
        "Python 3.12 or higher). Note: using the `--branch` option alongside `--sysmon` may require"
        " additional configuration for efficient execution.",
    )
    parser.add_argument(
        "--single-run",
        action=argparse.BooleanOptionalAction,
        help="With --modular, run all test files in one pytest-xdist run with -j workers, rather"
        " than one run per test file, and credit each file mainly with the lines its own test file"
        " runs (see checks_superstaq/_coverage_contexts.py).",
    )
    parser.add_argument(
        "--shard",
        type=_parse_shard,
        metavar="K/N",
        help=f"With --modular --single-run, check only shard K of N of the test files, balanced by"
        f" {_DURATIONS_FILE}. Update that file with a whole, unsharded run with --store-durations"
        " --clean-durations. Outdated durations only unbalance the shards.",
    )
    parser.description = textwrap.dedent(
        """
        Checks to make sure that all code is covered by unit tests.
        Fails if any pytest fails or if coverage is not 100%.
        Ignores integration tests and files in the [repo_root]/examples directory.
        Passes --disable-socket to coverage, unless running with --enable-socket.
        """
    )

    parsed_args, pytest_args = parser.parse_known_intermixed_args(args)
    if "coverage" in parsed_args.skip:
        return 0
    if parsed_args.single_run and not parsed_args.modular:
        parser.error("--single-run requires --modular")
    if parsed_args.shard and not parsed_args.single_run:
        parser.error("--shard requires --modular --single-run")

    # Enable threading setting before other args so -n can be overwritten
    default_threads = (
        parsed_args.jobs
        if not parsed_args.modular and parsed_args.jobs is not None
        else "auto"
        if not parsed_args.files
        and not parsed_args.modular
        and parsed_args.revisions is None
        and "-s" not in pytest_args
        else "0"
    )
    pytest_args = [f"-n={default_threads}", *pytest_args]

    coverage_args = []
    if parsed_args.sysmon and sys.version_info.minor >= 12:
        os.environ["COVERAGE_CORE"] = "sysmon"

    if parsed_args.branch:
        coverage_args.append("--branch")

    files = check_utils.extract_files(parsed_args, include, exclude, silent)
    silent = silent or not (parsed_args.files or parsed_args.revisions)

    test_files = check_utils.get_test_files(files, exclude=exclude, silent=silent)
    if not test_files:
        print("No test files to check for pytest and coverage.")  # noqa: T201
        return 0

    coverage_returncode = None
    try:
        if parsed_args.single_run:
            returncode, coverage_returncode = _run_single(
                files,
                pytest_args,
                exclude,
                parsed_args.jobs,
                branch=parsed_args.branch,
                shard=parsed_args.shard,
            )
        elif parsed_args.modular:
            returncode = _run_modular(files, coverage_args, pytest_args, exclude, parsed_args.jobs)
        else:
            result = _run_on_files(files, test_files, coverage_args, pytest_args)
            returncode = result.returncode

    except BaseException as e:
        # Clean up any temporary files if the check didn't complete
        subprocess.check_call([sys.executable, "-m", "coverage", "erase"], cwd=check_utils.root_dir)
        print(check_utils.failure(f"\nCoverage failed to complete ({e!r})."))  # noqa: T201
        return 1

    return _report(returncode, coverage_returncode)


def _get_test_pairs(
    files: list[str], exclude: str | Iterable[str]
) -> list[tuple[list[str], list[str]]]:
    """Pair the files requiring coverage with the test files that run them."""
    # Move test files to the end of the file list, so if both "x.py" and "x_test.py" are in `files`
    # both will be included in the coverage report
    files.sort(
        key=lambda file: file.endswith("_test.py") or os.path.basename(file).startswith("test_")
    )

    # Build (files_requiring_coverage, test_files) pairs
    test_pairs: list[tuple[list[str], list[str]]] = []
    while files:
        file = files.pop(0)
        test_files = check_utils.get_test_files([file], exclude=exclude, silent=True)

        if not test_files:
            # Arguably we should fail the modular coverage check if no test file exists, but for now
            # just skip
            continue

        # File(s) to include in coverage report. If both "x.py" and "x_test.py" are passed, require
        # full coverage on each; otherwise only require coverage on whichever was passed (even
        # though both are used to run the test)
        files_requiring_coverage = [file]
        for test_file in test_files:
            if test_file in files:
                files_requiring_coverage.append(test_file)
                files.remove(test_file)

        test_pairs.append((files_requiring_coverage, test_files))
    return test_pairs


def _run_modular(
    files: list[str],
    coverage_args: list[str],
    pytest_args: list[str],
    exclude: str | Iterable[str],
    num_jobs: int | None,
) -> int:
    """Run modular coverage checks.  If num_workers != 1, these checks are run concurrently."""
    subprocess.check_call([sys.executable, "-m", "coverage", "erase"], cwd=check_utils.root_dir)
    test_pairs = _get_test_pairs(files, exclude)

    test_returncode = 0
    if num_jobs == 0:
        # Run modular checks one at a time
        for files_requiring_coverage, test_files in test_pairs:
            result = _run_on_files(files_requiring_coverage, test_files, coverage_args, pytest_args)
            test_returncode |= result.returncode
        return test_returncode

    # Run modular checks concurrently
    max_workers = num_jobs or min((os.cpu_count() or 2) // 2, len(test_pairs))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        jobs = [
            executor.submit(
                _run_on_files,
                files_requiring_coverage,
                test_files,
                coverage_args,
                pytest_args,
                capture_output=True,
            )
            for files_requiring_coverage, test_files in test_pairs
        ]
        for future in concurrent.futures.as_completed(jobs):
            result = future.result()
            print(result.stdout, end="")  # noqa: T201
            test_returncode |= result.returncode
    return test_returncode


def _run_single(
    files: list[str],
    pytest_args: list[str],
    exclude: str | Iterable[str],
    num_jobs: int | None,
    *,
    branch: bool,
    shard: tuple[int, int] | None,
) -> tuple[int, int]:
    """Run all test files in one pytest run, and check each file against its test file's lines.

    Each file must reach fail_under on its own (plain --modular checks the total of all files).

    Returns:
        The exit code of pytest, and 2 if any file is not covered enough by its test file (else 0).
    """
    root_dir = check_utils.root_dir
    test_pairs = _get_test_pairs(files, exclude)
    if shard:
        test_pairs = _get_shard(test_pairs, *shard)
        if not test_pairs:
            print("No test files in this shard.")  # noqa: T201
            return 0, 0

    # Run the test files in one pytest-xdist run, with _PLUGIN labeling the lines by test file.
    data_file = os.path.join(root_dir, ".coverage")
    # Also erase parallel data files that an interrupted run left behind.
    coverage.CoverageData(data_file).erase(parallel=True)
    all_test_files = [test_file for _, test_files in test_pairs for test_file in test_files]
    if num_jobs is None:
        num_jobs = min((os.cpu_count() or 2) // 2, len(all_test_files))
    environment = {
        **os.environ,
        _DATA_FILE_ENV: data_file,
        _INCLUDE_ENV: os.pathsep.join(
            os.path.join(root_dir, file) for pair_files, _ in test_pairs for file in pair_files
        ),
        _BRANCH_ENV: "1" if branch else "",
    }
    # The node IDs of tests, which name coverage contexts, are relative to --rootdir.
    command = [sys.executable, "-m", "pytest", *all_test_files, *pytest_args, f"-n={num_jobs}"]
    command += ["--dist=loadfile", "-p", _PLUGIN, f"--rootdir={root_dir}"]
    sys.stdout.flush()
    test_returncode = subprocess.call(command, cwd=root_dir, env=environment)

    if "--store-durations" in pytest_args:
        # Store one duration per test file (the sum of its tests), which --shard reads.
        durations = _read_test_file_durations()
        with open(os.path.join(root_dir, _DURATIONS_FILE), "w", encoding="utf-8") as output:
            json.dump(durations, output, indent=4, sort_keys=True)
            output.write("\n")

    # Check each file against the lines credited to its own test files.
    measurement = coverage.Coverage(data_file=data_file)
    measurement.combine(keep=False)
    fail_under = float(measurement.get_option("report:fail_under"))  # type: ignore[arg-type]
    omit = [  # type: ignore[misc]
        *(measurement.get_option("run:omit") or []),
        *(measurement.get_option("report:omit") or []),
    ]
    coverage_returncode = 0
    for files_requiring_coverage, test_files in test_pairs:
        contexts = _get_credited_contexts(test_files)
        for file in files_requiring_coverage:
            report = io.StringIO()
            try:
                percentage = measurement.report(
                    [os.path.join(root_dir, file)],
                    contexts=contexts,
                    omit=omit,
                    precision=2,
                    file=report,
                )
            except coverage.exceptions.NoDataError:  # the coverage configuration omits the file
                continue
            if percentage < fail_under:
                print(f"\nCoverage of {file} by {', '.join(test_files)}:")  # noqa: T201
                print(report.getvalue(), end="")  # noqa: T201
                coverage_returncode = 2
    return test_returncode, coverage_returncode


def _parse_shard(value: str) -> tuple[int, int]:
    """Parse a shard "K/N" into (K, N)."""
    shard, _, num_shards = value.partition("/")
    if not (shard.isdigit() and num_shards.isdigit() and 1 <= int(shard) <= int(num_shards)):
        raise argparse.ArgumentTypeError(f"expected K/N with 1 <= K <= N, not {value!r}")
    return int(shard), int(num_shards)


def _get_shard(
    test_pairs: list[tuple[list[str], list[str]]], shard: int, num_shards: int
) -> list[tuple[list[str], list[str]]]:
    """Return the test pairs in shard `shard` (from 1) of `num_shards`.

    Each test pair, slowest first (ties by name), joins the shard with the smallest total time.
    """
    durations = _read_test_file_durations()
    sizes = {
        test_file: os.path.getsize(os.path.join(check_utils.root_dir, test_file))
        for _, test_files in test_pairs
        for test_file in test_files
    }
    # Estimate the times of test files not in _DURATIONS_FILE from their sizes.
    listed = [test_file for test_file in sizes if test_file in durations]
    listed_size = sum(sizes[test_file] for test_file in listed)
    listed_time = sum(durations[test_file] for test_file in listed)
    seconds_per_byte = listed_time / listed_size if listed_size else 1
    times = {
        test_file: durations.get(test_file, size * seconds_per_byte)
        for test_file, size in sizes.items()
    }

    def get_time(test_pair: tuple[list[str], list[str]]) -> float:
        return sum(times[test_file] for test_file in test_pair[1])

    shard_times = [0.0] * num_shards
    shards: list[list[tuple[list[str], list[str]]]] = [[] for _ in range(num_shards)]
    for test_pair in sorted(test_pairs, key=lambda pair: (-get_time(pair), pair[1])):
        index = shard_times.index(min(shard_times))
        shard_times[index] += get_time(test_pair)
        shards[index].append(test_pair)
    return shards[shard - 1]


def _read_test_file_durations() -> dict[str, float]:
    """Read the seconds per test file in _DURATIONS_FILE (or {} if there is none).

    A test file's time is the sum of its tests (keys with "::") if any are listed, else its own.
    """
    path = os.path.join(check_utils.root_dir, _DURATIONS_FILE)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as file:
        durations: dict[str, float] = json.load(file)
    file_durations: dict[str, float] = {}
    test_totals: dict[str, float] = {}
    for node_id, seconds in durations.items():
        test_file, separator, _ = node_id.partition("::")
        if separator:
            test_totals[test_file] = test_totals.get(test_file, 0.0) + seconds
        else:
            file_durations[test_file] = seconds
    return {**file_durations, **test_totals}


def _get_credited_contexts(test_files: list[str]) -> list[str]:
    """Return, in a list, a regular expression that matches the contexts credited to test files."""
    contexts = {""}
    phases = ("collect", "setup", "call", "teardown")
    for test_file in test_files:
        path = pathlib.PurePath(test_file)
        contexts.update(f"{directory.as_posix()}|collect" for directory in path.parents)
        contexts.update(f"{path.as_posix()}|{phase}" for phase in phases)
    return [f"^({'|'.join(map(re.escape, sorted(contexts)))})$"]


def _run_on_files(
    files_requiring_coverage: list[str],
    test_files: list[str],
    coverage_args: list[str],
    pytest_args: list[str],
    *,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Helper function to run coverage tests on the given files with the given pytest arguments."""
    coverage_args = ["--include=" + ",".join(files_requiring_coverage), *coverage_args]
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            *coverage_args,
            "-m",
            "pytest",
            *test_files,
            *pytest_args,
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.STDOUT if capture_output else None,
    )


def _report(test_returncode: int, coverage_returncode: int | None = None) -> int:
    """Print the result; if `coverage_returncode` is None, first combine and report coverage."""
    if coverage_returncode is None:
        subprocess.call(
            [sys.executable, "-m", "coverage", "combine"],
            cwd=check_utils.root_dir,
            stdout=subprocess.DEVNULL,
        )

        # Flush Python's stdout buffer before the subprocess writes directly to it, so captured
        # test output appears before the coverage report
        sys.stdout.flush()
        coverage_returncode = subprocess.call(
            [sys.executable, "-m", "coverage", "report", "--precision=2"],
            cwd=check_utils.root_dir,
        )

    if test_returncode:
        print(check_utils.failure("TEST FAILURE!"))  # noqa: T201
        return test_returncode

    if coverage_returncode:
        print(check_utils.failure("COVERAGE FAILURE!"))  # noqa: T201
        return coverage_returncode

    print(check_utils.success("TEST AND COVERAGE SUCCESS!"))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(run(*sys.argv[1:]))
