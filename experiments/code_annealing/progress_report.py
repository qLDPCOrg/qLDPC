# SPDX-License-Identifier: Apache-2.0

"""Report how many annealing runs of each code size have finished.

Run with ``python -m experiments.code_annealing.progress_report [--test]``.  The script prints a
table with a row for each code size in the logs.  For each code type, it shows the finished runs out
of the runs with a checkpoint, and the best code that any of them has found: its distance and its
number of minimum-weight logical operators.  The script reads only the chain checkpoints, so it is
fast and rechecks no code.  A run appears once it has saved its first checkpoint: look for a missing
run's error in its ``.log`` file in the logs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .core.annealing import score_rank
from .core.cli import add_results_dir_argument, add_test_argument, get_results_dir, make_parser
from .core.paths import get_logs_dir
from .core.proposals import CODE_TYPES, CodeType
from .core.saved_runs import RunProgress, load_progress


def main(argv: Sequence[str] | None = None) -> None:
    """Print the progress table of the runs in the logs."""
    parser = make_parser(__doc__)
    add_results_dir_argument(parser)
    add_test_argument(parser)
    args = parser.parse_args(argv)

    runs: dict[tuple[int, int], dict[CodeType, list[RunProgress]]] = defaultdict(
        lambda: {code_type: [] for code_type in CODE_TYPES}
    )
    logs_dir = get_logs_dir(get_results_dir(args))
    for run in load_progress(logs_dir):
        runs[run.num_qubits, run.num_logicals][run.code_type].append(run)

    if not runs:
        raise SystemExit(f"No annealing runs in {logs_dir}")
    lines = [
        "       |" + "|".join(f"{code_type:^26}" for code_type in CODE_TYPES),
        " n   k |" + "|".join(f" {'finished  (d, count)':<25}" for _ in CODE_TYPES),
        "-------+" + "+".join("-" * 26 for _ in CODE_TYPES),
    ]
    for (num_qubits, num_logicals), by_code_type in sorted(runs.items()):
        cells = [_format_cell(by_code_type[code_type]) for code_type in CODE_TYPES]
        lines.append(f"{num_qubits:>2} {num_logicals:>3} | " + " | ".join(cells))
    print("\n".join(line.rstrip() for line in lines))
    print(
        "finished = finished runs / runs with a checkpoint; (d, count) = distance and minimum-weight logicals"
    )


####################################################################################################
# helper methods


def _format_cell(runs: Sequence[RunProgress]) -> str:
    """The cell of a code size and code type: the finished runs, and the best code of any run.

    The best code has the greatest distance, then the fewest minimum-weight logical operators.
    """
    if not runs:
        return f"{'-':>8}  {'':<14}"
    scores = [run.greatest_distance for run in runs]
    best = max(scores, key=score_rank)
    finished = f"{sum(run.finished for run in runs)}/{len(runs)}"
    best_text = f"({best.distance}, {best.multiplicity})"
    return f"{finished:>8}  {best_text:<14}"


if __name__ == "__main__":
    main()
