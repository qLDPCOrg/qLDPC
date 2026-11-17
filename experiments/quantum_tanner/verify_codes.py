#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Verify the reproducibility of saved quantum Tanner codes.

This script regenerates every code saved by run_search.py from the specification and sample in its
file name, and checks that the regenerated code equals the saved code.  It reports every code that
fails this check, cannot be loaded or regenerated, or was saved before the search built
minimum-weight local checks, and exits with an error if there are any.

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
    """Check that all saved quantum Tanner codes can be regenerated from their file names."""
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

    A saved code passes if it can be loaded, if it records the minimum-weight local checks of the
    search, and if regenerating it from its specification and sample reproduces it.  Files saved
    before the search built minimum-weight local checks record no choice of local checks, and their
    headers record the weights of other checks.
    """
    # load the saved code, which fails if its file is truncated or malformed
    try:
        saved_code = codes.QTCode.load(str(path))
    except (IndexError, SyntaxError, ValueError) as error:
        return f"Saved code cannot be loaded ({error!r})"

    # the stabilizer weight in the header of the file belongs to the checks that the file records
    choice = saved_codes.read_sections(path).get("local checks")
    if choice is None:
        return (
            "Saved code records no choice of local checks, so an earlier search saved it, and the"
            " weights in its header belong to row-reduced parity checks"
        )
    if choice != ["min_weight"]:
        return (
            f"Saved code records the local checks {choice}, not ['min_weight'], so the weights in"
            " its header belong to other parity checks"
        )

    # regenerate the code from its file name, and compare
    try:
        spec, sample_id = saved_codes.parse_code_path(path)
        regenerated_code = spec.build(sample_id)
    except ValueError as error:
        return f"Saved code cannot be regenerated ({error!r})"
    if regenerated_code != saved_code:
        return "Saved code does not match its regenerated version"
    return None


if __name__ == "__main__":
    main()
