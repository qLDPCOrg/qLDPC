#!/usr/bin/env python3
import os
import shutil
import sys

import checks_superstaq as checks


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


if __name__ == "__main__":
    ensure_pandoc_on_path()
    sys.exit(checks.build_docs.run(*sys.argv[1:]))
