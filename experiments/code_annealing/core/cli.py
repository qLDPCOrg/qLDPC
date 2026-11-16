# SPDX-License-Identifier: Apache-2.0

"""Command-line options that the scripts of the experiment share.

``--results-dir`` moves the results directory, and ``--test`` selects its ``test/`` subdirectory,
which holds the smoke-test campaign of ``write_jobs --test``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path

from . import paths


def make_parser(doc: str | None) -> argparse.ArgumentParser:
    """A parser whose help text is the docstring of a script, with its layout kept."""
    return argparse.ArgumentParser(
        description=doc, formatter_class=argparse.RawDescriptionHelpFormatter
    )


def run_step(name: str, main: Callable[[Sequence[str]], None], options: Sequence[str]) -> None:
    """Run the ``main`` function of a script as a step of a longer pipeline, under a heading."""
    print(f"\n==== {name} ====", flush=True)
    main(options)


def add_results_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Add the option that moves the directory of every file that the scripts read and write."""
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=paths.RESULTS_DIR,
        help="directory of results (default: %(default)s)",
    )


def add_test_argument(parser: argparse.ArgumentParser) -> None:
    """Add the option that selects the ``test/`` subdirectory of the results directory."""
    parser.add_argument(
        "--test",
        action="store_true",
        help="use the test subdirectory of the results directory, which holds a smoke-test campaign",
    )


def get_results_dir(args: argparse.Namespace) -> Path:
    """The results directory of the options: the test subdirectory of it with ``--test``."""
    return args.results_dir / paths.TEST_DIR_NAME if args.test else args.results_dir
