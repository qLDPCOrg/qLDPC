# SPDX-License-Identifier: Apache-2.0

"""The runner of each experiment's acceptance tests.

An experiment's ``acceptance.py`` makes one ``Suite``, registers its tests with ``Suite.test``, and
runs them with ``Suite.main`` from the command line, or with ``Suite.run_all`` from its plug-in's
``acceptance``.  A test takes its own directory, ``<results_dir>/<ID>``, and whether the tests run
on the node.  Each test prints PASS or FAIL.  An exception that a test raises fails that test, and
the other tests still run, if it derives from ``Exception`` or is ``SystemExit`` (``FAILURES``);
another, such as ``KeyboardInterrupt``, stops the suite.  ``--only`` refuses an ID that is not a
test of the suite.  ``check_marked`` checks the marks that an analysis puts on its outputs
(``store.mark_outputs``), and ``raises`` checks that a call raises an exception.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import signal
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Collection
from typing import Any, TypeVar

# The exceptions that fail a test, after which the other tests still run.
FAILURES = (Exception, SystemExit)

Test = Callable[[pathlib.Path, bool], None]
_Error = TypeVar("_Error", bound=BaseException)


class Suite:
    """The acceptance tests of one experiment, by ID, and their runner.

    Args:
        temp_prefix: The prefix of the temporary results directory of ``main`` without
            ``--results-dir``.
        node_tests: The IDs of the tests that run on the node, or None for every test.
        limit_s: The wall time after which a test fails, or None for no limit.
    """

    def __init__(
        self,
        temp_prefix: str,
        node_tests: Collection[str] | None = None,
        limit_s: float | None = None,
    ) -> None:
        self.temp_prefix = temp_prefix
        self.node_tests = node_tests
        self.limit_s = limit_s
        self.tests: dict[str, Test] = {}

    def test(self, function: Test) -> Test:
        """Register a test by the ID at the start of its name, as in ``at1_...`` for AT1."""
        self.tests[function.__name__.split("_")[0].upper()] = function
        return function

    def main(self, description: str) -> None:
        """Run the tests that the command line selects, and exit with status 1 if any failed."""
        args = self.parse_args(description)
        results_dir = args.results_dir or pathlib.Path(tempfile.mkdtemp(prefix=self.temp_prefix))
        passed = self.run_all(node=not args.local, results_dir=results_dir, only=args.only)
        sys.exit(0 if passed else 1)

    def parse_args(self, description: str, argv: list[str] | None = None) -> argparse.Namespace:
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(
            description=description, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        parser.add_argument(
            "--local", action="store_true", help="run off the node (see above for what differs)"
        )
        parser.add_argument(
            "--only", nargs="*", choices=sorted(self.tests), help="run only these tests"
        )
        parser.add_argument("--results-dir", type=pathlib.Path, help="where the tests' runs go")
        return parser.parse_args(argv)

    def run_all(
        self, node: bool, results_dir: pathlib.Path, only: Collection[str] | None = None
    ) -> bool:
        """Run the tests, print PASS or FAIL for each, and return whether all passed."""
        results_dir.mkdir(parents=True, exist_ok=True)
        failed = []
        for name, test in self.tests.items():
            if only and name not in only:
                continue
            if node and self.node_tests is not None and name not in self.node_tests:
                continue
            directory = results_dir / name
            directory.mkdir(parents=True, exist_ok=True)
            start = time.monotonic()
            if self.limit_s is not None:
                signal.signal(signal.SIGALRM, _stop)
                signal.setitimer(signal.ITIMER_REAL, self.limit_s)
            try:
                test(directory, node)
                print(f"PASS {name} ({time.monotonic() - start:.1f} s)", flush=True)
            except FAILURES:
                failed.append(name)
                print(f"FAIL {name}\n{traceback.format_exc()}", flush=True)
            finally:
                if self.limit_s is not None:
                    signal.setitimer(signal.ITIMER_REAL, 0)
        print(f"{len(failed)} failed: {failed}" if failed else "all passed", flush=True)
        return not failed


def check_marked(out: pathlib.Path, marks: dict[str, Any]) -> list[pathlib.Path]:
    """Check that every CSV and JSON output in ``out`` carries ``marks``; return the other files.

    A CSV file carries a mark as a column with its value in every row, and a JSON file as a key.
    ``out`` must hold at least one CSV or JSON file.
    """
    others, marked = [], 0
    for path in sorted(out.rglob("*")):
        if path.suffix == ".csv":
            with open(path, newline="") as file:
                reader = csv.DictReader(file)
                rows = list(reader)
            assert set(marks) <= set(reader.fieldnames or []), (path.name, reader.fieldnames)
            for name, value in marks.items():
                assert all(row[name] == str(value) for row in rows), (path.name, name)
            marked += 1
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
            assert all(data.get(name) == value for name, value in marks.items()), (path.name, data)
            marked += 1
        elif path.is_file():
            others.append(path)
    assert marked, f"no CSV or JSON output in {out}"
    return others


def raises(exception: type[_Error], function: Callable[[], Any]) -> _Error:
    """Call ``function``, check that it raises ``exception``, and return what it raised."""
    try:
        function()
    except exception as error:
        return error
    raise AssertionError(f"{exception.__name__} was not raised")


def _stop(signum: int, frame: Any) -> None:
    """Stop a test that exceeds its limit."""
    raise TimeoutError("the test exceeded its limit")
