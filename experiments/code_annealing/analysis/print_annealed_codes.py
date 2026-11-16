# SPDX-License-Identifier: Apache-2.0

"""Print a summary of the codes that saved annealing runs found.

Run with ``python -m experiments.code_annealing.analysis.print_annealed_codes``.  Select codes
with ``--num-qubits``, ``--num-logicals``, and ``--code-type``.  Each line shows the code type, the
code size, the file that holds the code, the role of the code in its run (a record, or a code that
a chain checkpoint holds), and the exact (distance, number of minimum-weight X-only plus Z-only
logical operators, mean stabilizer weight).  The mean weight is that of the minimum-weight
generators of ``reduce_weights``.  Every code is rechecked, so this takes as long as counting the
logical operators of all of them.  Pass explicit checkpoints, which bring their records along, to
summarize runs saved elsewhere, and ``--csv`` to keep every exact count of every selected code in
``data/saved_codes.csv`` of the results directory.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..core.cli import add_results_dir_argument, make_parser
from ..core.code_files import get_reduced_weights
from ..core.paths import get_data_dir, get_logs_dir
from ..core.proposals import CODE_TYPES
from ..core.saved_runs import SavedCode, get_saved_run_paths, load_saved_codes, write_csv


def main(argv: Sequence[str] | None = None) -> None:
    """Print one line for every saved code, optionally writing a CSV of all of them."""
    parser = make_parser(__doc__)
    parser.add_argument("--num-qubits", type=int, help="only codes with this many qubits")
    parser.add_argument("--num-logicals", type=int, help="only codes with this many logical qubits")
    parser.add_argument("--code-type", choices=CODE_TYPES, help="only codes of this code type")
    parser.add_argument(
        "--csv", action="store_true", help="also write every exact count to a CSV file"
    )
    parser.add_argument("checkpoints", type=Path, nargs="*", help="explicit chain checkpoints")
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    saved = [
        code
        for code in load_saved_codes(
            get_saved_run_paths(get_logs_dir(args.results_dir), args.checkpoints)
        )
        if args.num_qubits in (None, code.num_qubits)
        and args.num_logicals in (None, code.num_logicals)
        and args.code_type in (None, code.code_type)
    ]
    if not saved:
        raise SystemExit(f"No saved codes found in {get_logs_dir(args.results_dir)}")
    print(
        "# code type (n, k) file role"
        " (distance, minimum-weight X-only plus Z-only logicals, mean weight)"
    )
    for code in sorted(
        saved, key=lambda code: (code.code_type, code.num_qubits, code.num_logicals)
    ):
        values = f"({code.score.distance}, {code.score.multiplicity}, {_get_mean_weight(code)})"
        print(
            f"{code.code_type} ({code.num_qubits}, {code.num_logicals}) {code.path.name}"
            f" {code.role} {values}"
        )
    if args.csv:
        path = get_data_dir(args.results_dir) / "saved_codes.csv"
        write_csv(saved, path)
        print(f"Wrote {len(saved)} codes to {path}")


####################################################################################################
# helper methods


def _get_mean_weight(code: SavedCode) -> float:
    """The mean weight of the stabilizer generators, after lowering their weights."""
    weights = get_reduced_weights(code.counter.code)
    return round(sum(weights) / len(weights), 2)


if __name__ == "__main__":
    main()
