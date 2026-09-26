"""Helpers shared by more than one check script."""

import fnmatch
import os
import shutil
import sys

import checks_superstaq


def ensure_pandoc_on_path() -> None:
    """Make a bundled pandoc discoverable, if the system provides none.

    nbsphinx renders the example notebooks through pandoc, which it locates by searching PATH.  The
    pypandoc-binary distribution ships a pandoc executable inside its own package directory but
    installs no console script, so that copy is invisible to a PATH search until it is added here.
    A system pandoc, if present, takes precedence and this is a no-op.
    """
    if shutil.which("pandoc"):
        return
    try:
        import pypandoc
    except ImportError:
        return
    try:
        pandoc = pypandoc.get_pandoc_path()
    except OSError:
        return
    os.environ["PATH"] = os.path.dirname(pandoc) + os.pathsep + os.environ.get("PATH", "")


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
