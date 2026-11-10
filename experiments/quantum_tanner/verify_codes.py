#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Verify the reproducibility of saved quantum Tanner codes.

This script regenerates every code saved by run_search.py from the group, base code, and seed in its
file name, and checks that the regenerated code equals the saved code.  It reports every code that
fails this check or cannot be loaded, and exits with an error if there are any.

Example:
    python experiments/quantum_tanner/verify_codes.py
"""

from __future__ import annotations

import argparse
import pathlib

import run_search as search

from qldpc import abstract, codes

####################################################################################################
# command-line interface


def main() -> None:
    """Check that all saved quantum Tanner codes can be regenerated from their seeds."""
    args = parse_args()
    paths = search.get_saved_code_paths(args.results_dir / search.CODES_NAME)

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
    search.add_results_dir_argument(parser)
    return parser.parse_args()


####################################################################################################
# verification


def get_verification_problem(path: pathlib.Path) -> str | None:
    """Say why a saved quantum Tanner code fails verification, or return None if it passes.

    A saved code passes if it can be loaded, and if regenerating it from its seed reproduces it.
    """
    # load the saved code, which fails if its file is truncated or malformed
    try:
        saved_code = codes.QTCode.load(str(path))
    except (IndexError, SyntaxError, ValueError) as error:
        return f"Saved code cannot be loaded ({error!r})"

    # regenerate the code from its seed, and compare
    group_order, group_index, base_code_id, seed = search.parse_code_path(path)
    group = abstract.SmallGroup(group_order, group_index)
    base_code = search.get_base_code(base_code_id)
    if codes.QTCode.random(group, base_code, seed=seed) != saved_code:
        return "Saved code does not match its regenerated version"
    return None


if __name__ == "__main__":
    main()
