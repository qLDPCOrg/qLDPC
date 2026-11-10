#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Verify the reproducibility of saved quantum Tanner codes.

This script regenerates every code saved by run_search.py from the specification and seed in its
file name, and checks that the regenerated code equals the saved code.  It reports every code that
fails this check, or cannot be loaded or regenerated, and exits with an error if there are any.

Example:
    python experiments/quantum_tanner/verify_codes.py
"""

from __future__ import annotations

import argparse
import pathlib
from typing import TYPE_CHECKING

from qldpc import codes

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.quantum_tanner.*, and would silently treat a bare "import saved_codes" as untyped.
if TYPE_CHECKING:
    from experiments.quantum_tanner import saved_codes
else:
    import saved_codes

####################################################################################################
# command-line interface


def main() -> None:
    """Check that all saved quantum Tanner codes can be regenerated from their seeds."""
    args = parse_args()
    paths = saved_codes.get_saved_code_paths(args.results_dir / saved_codes.CODES_NAME)

    failed_paths = []
    for index, path in enumerate(paths, start=1):
        print(f"{index}/{len(paths)}")
        if (problem := get_verification_problem(path)) is not None:
            print(f"  {problem}: {path}")
            failed_paths.append(path)

    if failed_paths:
        raise RuntimeError(f"{len(failed_paths)} of {len(paths)} saved codes fail verification")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    saved_codes.add_results_dir_argument(parser)
    return parser.parse_args()


####################################################################################################
# verification


def get_verification_problem(path: pathlib.Path) -> str | None:
    """Say why a saved quantum Tanner code fails verification, or return None if it passes.

    A saved code passes if it can be loaded, and if regenerating it from its specification and seed
    reproduces it.
    """
    # load the saved code, which fails if its file is truncated or malformed
    try:
        saved_code = codes.QTCode.load(str(path))
    except (IndexError, SyntaxError, ValueError) as error:
        return f"Saved code cannot be loaded ({error!r})"

    # regenerate the code from its seed, and compare
    try:
        spec, seed = saved_codes.parse_code_path(path)
        regenerated_code = spec.build(seed)
    except ValueError as error:
        return f"Saved code cannot be regenerated ({error!r})"
    if regenerated_code != saved_code:
        return "Saved code does not match its regenerated version"
    return None


if __name__ == "__main__":
    main()
