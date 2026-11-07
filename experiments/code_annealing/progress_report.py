# SPDX-License-Identifier: Apache-2.0

"""Report how many annealing runs of each code size have finished.

Run with ``python -m experiments.code_annealing.progress_report [--test]``.  The script prints a
table with a row for each code size that the logs hold, which shows for each code type the runs
that have finished all their epochs out of all runs that have started, and the best code of any of
them so far: the greatest distance with its number of minimum-weight logical operators.  A search
appears once it has saved a checkpoint, so searches that have not started, or that failed before
their first checkpoint, are missing: look for the reason in the ``.log`` files of the logs.  The
script reads the chain checkpoints only, so it is fast and does not recheck any code.  Pass
``--test`` to report on the smoke-test campaign of ``write_jobs.py --test``.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from .core.annealing import distance_rank
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
        "finished = finished runs / runs started; (d, count) = distance and minimum-weight logicals"
    )


####################################################################################################
# helper methods


def _format_cell(runs: Sequence[RunProgress]) -> str:
    """The cell of a code size and code type: the finished runs, and the best code of any run.

    The best code has the greatest distance, then the fewest minimum-weight logical operators.
    """
    if not runs:
        return f"{'-':>8}  {'':<14}"
    best = max(
        runs,
        key=lambda run: distance_rank(
            run.greatest_distance, run.greatest_distance_count_homogeneous
        ),
    )
    finished = f"{sum(run.finished for run in runs)}/{len(runs)}"
    return f"{finished:>8}  {f'({best.greatest_distance}, {best.greatest_distance_count_homogeneous})':<14}"


if __name__ == "__main__":
    main()
