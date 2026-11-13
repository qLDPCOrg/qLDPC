# SPDX-License-Identifier: Apache-2.0

"""Save the best code of each size that saved annealing runs found.

Run with ``python -m experiments.code_annealing.analysis.collect_annealed_codes``.  The best code
of each code type, number of qubits, and number of logical qubits has the greatest distance, and
among those the fewest minimum-weight logical operators.  The codes go to
``data/annealed_codes.txt`` (CSS) and ``data/annealed_codes_swel.txt`` (SWEL) of the results
directory, with generators of low weight.  Codes that are already in those files compete with those
of the runs, and every code is rechecked, so collecting only ever improves the files.  Pass
explicit checkpoints, which bring their records along, to collect runs saved elsewhere.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from qldpc.codes.distance import CSSLogicalCounter

from ..core.annealing import distance_rank
from ..core.cli import add_results_dir_argument, make_parser
from ..core.code_files import (
    format_code,
    read_code_file,
    recount,
    reduce_weights,
    write_code_file,
)
from ..core.paths import get_code_files, get_logs_dir
from ..core.proposals import CodeType
from ..core.saved_runs import get_saved_run_paths, load_saved_codes


def main(argv: Sequence[str] | None = None) -> None:
    """Rewrite the code files with the best code of each size."""
    parser = make_parser(__doc__)
    parser.add_argument("checkpoints", type=Path, nargs="*", help="explicit chain checkpoints")
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    logs_dir = get_logs_dir(args.results_dir)
    saved = load_saved_codes(get_saved_run_paths(logs_dir, args.checkpoints))
    if not saved:
        raise SystemExit(f"No saved runs in {logs_dir}: nothing to collect")
    files = get_code_files(args.results_dir)

    # Offer the codes of the files, rechecked, and then those of the runs; the best of a size wins.
    best: dict[tuple[CodeType, int, int], _Candidate] = {}

    def offer(code_type: CodeType, counter: CSSLogicalCounter, distance: int, count: int) -> None:
        key = (code_type, counter.num_qubits, counter.dimension)
        candidate = _Candidate(counter, distance, count)
        if key not in best or candidate.rank > best[key].rank:
            best[key] = candidate

    for code_type, path in files.items():
        for entry in read_code_file(path) if path.exists() else []:
            recount(entry)
            offer(
                code_type,
                CSSLogicalCounter.from_code(entry.code),
                entry.distance,
                entry.multiplicity,
            )
    for code in saved:
        offer(code.code_type, code.counter, code.score.distance, code.score.multiplicity)

    # Rewrite each file with the best code of each size, with generators of low weight.
    for code_type, path in files.items():
        blocks = []
        for key in sorted(key for key in best if key[0] == code_type):
            candidate = best[key]
            blocks.append(
                format_code(
                    reduce_weights(candidate.counter.checks_x),
                    reduce_weights(candidate.counter.checks_z),
                    candidate.distance,
                    candidate.multiplicity,
                )
            )
            print(
                f"{code_type} ({key[1]}, {key[2]}) -> distance {candidate.distance},"
                f" {candidate.multiplicity} minimum-weight logical operators"
            )
        if blocks:  # a code type without codes gets no file
            write_code_file(path, blocks)
            print(f"Wrote {len(blocks)} codes to {path}")


####################################################################################################
# helper methods


@dataclass(frozen=True)
class _Candidate:
    """A code with its exact distance and number of minimum-weight logical operators."""

    counter: CSSLogicalCounter
    distance: int
    multiplicity: int

    @property
    def rank(self) -> tuple[int, int]:
        return distance_rank(self.distance, self.multiplicity)


if __name__ == "__main__":
    main()
