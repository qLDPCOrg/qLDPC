#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Verify the reproducibility of saved quantum Tanner codes.

This script regenerates every code saved by run_search.py from the group, base code, and seed in its
file name, and checks that the regenerated code equals the saved code.  It reports every code that
fails this check, and exits with an error if there are any.

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

    mismatched_paths = []
    for index, path in enumerate(paths, start=1):
        print(f"{index}/{len(paths)}")
        if not is_reproducible(path):
            print(f"  Saved code does not match its regenerated version: {path}")
            mismatched_paths.append(path)

    if mismatched_paths:
        raise RuntimeError(
            f"{len(mismatched_paths)} of {len(paths)} saved codes are not reproducible"
        )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    search.add_results_dir_argument(parser)
    return parser.parse_args()


####################################################################################################
# verification


def is_reproducible(path: pathlib.Path) -> bool:
    """Does regenerating a saved quantum Tanner code from its seed reproduce the saved code?"""
    group_order, group_index, base_code_id, seed = search.parse_code_path(path)
    group = abstract.SmallGroup(group_order, group_index)
    base_code = search.get_base_code(base_code_id)
    new_code = codes.QTCode.random(group, base_code, seed=seed)
    return codes.QTCode.load(str(path)) == new_code


if __name__ == "__main__":
    main()
