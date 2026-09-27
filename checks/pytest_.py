#!/usr/bin/env python3
import fnmatch
import os
import sys

import checks_superstaq

EXCLUDE = (
    "*/__init__.py",
    "checks/*.py",
    "examples/*.py",
    "experiments/*.py",
    "docs/source/conf.py",
    # The notebooks under docs/source/examples are symlinks to their examples/ originals, so
    # collecting them would run every notebook twice -- and would fail besides, because the modules
    # that some notebooks import from their own directory have no counterpart in the docs tree.
    "docs/source/examples/*",
)


def reject_unmatched_file_arguments(args: tuple[str, ...] | list[str]) -> None:
    """Exit with an error if file arguments were given but none of them match a file.

    A file argument matching nothing is dropped silently, so a check asked to run on files that all
    turn out to be unmatched instead runs nothing and exits 0 -- a mistyped path, or the detached
    spelling of a plugin flag such as "-p no:cacheprovider", reads as a passing check.  Arguments
    are only rejected when *every* one of them is unmatched, since a value belonging to an unknown
    option (as in "-k some_name") is also collected as a file argument and must not fail a run whose
    real paths do match.
    """
    parsed, _ = checks_superstaq.check_utils.get_check_parser().parse_known_intermixed_args(args)
    if not parsed.files:
        return
    tracked = checks_superstaq.check_utils.get_tracked_files([])
    if any(
        os.path.exists(arg.split("::")[0]) or fnmatch.filter(tracked, arg) for arg in parsed.files
    ):
        return
    text = "Matched no existing or tracked file: " + ", ".join(parsed.files)
    sys.exit(checks_superstaq.check_utils.failure(text))


if __name__ == "__main__":
    reject_unmatched_file_arguments(sys.argv[1:])
    sys.exit(checks_superstaq.pytest_.run(*sys.argv[1:], exclude=EXCLUDE))
