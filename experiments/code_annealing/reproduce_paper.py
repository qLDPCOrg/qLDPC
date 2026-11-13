# SPDX-License-Identifier: Apache-2.0

"""Check the codes that the paper reports, and rebuild its table and figures from them.

Run with ``python -m experiments.code_annealing.reproduce_paper [--revised] [--skip-figures]``.  The
paper's best codes and its snapshot of qecdb.org, taken before the paper, are kept in
``paper_data/``, which no script writes.  The script copies them to the ``paper/data/``
subdirectory of the results directory, and runs three scripts on that subdirectory, in order:

1. ``analysis/half_supports`` recounts every code exactly, which checks the distance and the number
   of minimum-weight logical operators that the paper reports, and counts the half supports that
   the figure of error estimates needs.
2. ``analysis/latex_tables`` writes the table of the codes, compared with the paper's snapshot of
   qecdb.org even if the directory holds a newer scrape, to ``paper/data/table_css.tex`` and
   ``paper/data/table_swel.tex``, which a LaTeX document includes with ``\\input``, and the
   listings of the codes of the paper's appendix to ``paper/data/listing_css.txt`` and
   ``paper/data/listing_swel.txt``.
3. ``make_figures`` simulates the codes, and saves the paper's figures to ``paper/figures/``,
   together with ``error_rates.pdf``, which shows both code types in one figure.

With ``--revised``, the script does the same for the revised version of the paper: its codes, the
best of each size of the campaign of ``write_jobs.py``, and its scrape of qecdb.org are kept in
``paper_data/revised/``, and the script works in the ``paper_revised/`` subdirectory.

The simulations take most of the time.  Pass ``--skip-figures`` for the first two steps alone, or a
small ``--num-samples`` and ``--max-qubits`` for a quick look at the figures.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence

from .analysis import half_supports, latex_tables
from .core.cli import add_results_dir_argument, make_parser, run_step
from .core.paths import (
    PAPER_DATA_DIR,
    PAPER_DIR_NAME,
    PAPER_SNAPSHOT_FILE,
    REVISED_DATA_DIR,
    REVISED_DIR_NAME,
    REVISED_SNAPSHOT_FILE,
    get_data_dir,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Copy the data of the paper, check its codes, and rebuild its table and figures."""
    parser = make_parser(__doc__)
    parser.add_argument(
        "--revised", action="store_true", help="use the data of the revised version of the paper"
    )
    parser.add_argument("--skip-figures", action="store_true", help="do not simulate the codes")
    parser.add_argument(
        "--max-qubits", type=int, help="plot only the codes with at most this many qubits"
    )
    parser.add_argument("--num-samples", type=int, help="errors to sample for each code")
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    # The paper labels its columns of best known neighbors "prior", and the revised paper "qecdb".
    if args.revised:
        source_dir, snapshot_file, dir_name = (
            REVISED_DATA_DIR,
            REVISED_SNAPSHOT_FILE,
            REVISED_DIR_NAME,
        )
        best_label = "qecdb"
    else:
        source_dir, snapshot_file, dir_name = PAPER_DATA_DIR, PAPER_SNAPSHOT_FILE, PAPER_DIR_NAME
        best_label = "prior"
    results_dir = args.results_dir / dir_name
    data_dir = get_data_dir(results_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(source_dir.glob("*.txt")):
        shutil.copyfile(path, data_dir / path.name)
    print(f"Copied the data of the paper from {source_dir} to {data_dir}")

    options = ["--results-dir", str(results_dir)]
    snapshot = data_dir / snapshot_file.name
    run_step("half_supports", half_supports.main, options)
    table_options = [*options, "--data-file", str(snapshot), "--best-label", best_label]
    run_step("latex_tables", latex_tables.main, table_options)
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
