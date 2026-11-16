# SPDX-License-Identifier: Apache-2.0

"""Check the paper's codes, and rebuild its table and figures from them.

Run with ``python -m experiments.code_annealing.reproduce_paper [options]``.  The script copies the
files of ``paper_data/`` to ``paper/data/`` of the results directory, and runs three steps on the
copies:

1. ``analysis/half_supports`` recounts every code exactly, and counts the supports that the figure
   of error estimates uses.
2. ``analysis/latex_tables`` writes the table and the listings of the paper's appendix, compared
   with the paper's snapshot of qecdb.org.
3. ``make_figures`` simulates the codes, and saves the figures to ``paper/figures/``.

The figures take most of the time, and ``--skip-figures`` skips them.  ``REPRODUCTION.md`` maps each
table and figure of the paper to its output files.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence

from .analysis import half_supports, latex_tables
from .core.cli import add_results_dir_argument, make_parser, run_step
from .core.paths import PAPER_DATA_DIR, PAPER_DIR_NAME, PAPER_SNAPSHOT_FILE, get_data_dir


def main(argv: Sequence[str] | None = None) -> None:
    """Copy the data of the paper, check its codes, and rebuild its table and figures."""
    parser = make_parser(__doc__)
    parser.add_argument("--skip-figures", action="store_true", help="do not simulate the codes")
    parser.add_argument(
        "--max-qubits",
        type=int,
        help="plot only codes with at most this many qubits, for a quick look",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="errors to sample for each code (default: that of make_figures); a small number gives"
        " a quick look",
    )
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    results_dir = args.results_dir / PAPER_DIR_NAME
    data_dir = get_data_dir(results_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(PAPER_DATA_DIR.glob("*.txt")):
        shutil.copyfile(path, data_dir / path.name)
    print(f"Copied the data of the paper from {PAPER_DATA_DIR} to {data_dir}")

    options = ["--results-dir", str(results_dir)]
    snapshot = data_dir / PAPER_SNAPSHOT_FILE.name
    run_step("half_supports", half_supports.main, options)
    run_step("latex_tables", latex_tables.main, [*options, "--data-file", str(snapshot)])
    if not args.skip_figures:
        figure_options = [*options]
        for name in ("max_qubits", "num_samples"):
            if getattr(args, name) is not None:
                figure_options += [f"--{name.replace('_', '-')}", str(getattr(args, name))]
        # only the figures need matplotlib, so the table alone runs without it
        from . import make_figures

        run_step("make_figures", make_figures.main, figure_options)


if __name__ == "__main__":
    main()
