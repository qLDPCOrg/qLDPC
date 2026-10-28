# SPDX-License-Identifier: Apache-2.0

"""Plot how many minimum-weight logical operators random codes have.

Run with ``python -m experiments.code_annealing.figures.plot_random_histogram N K``.  The random
codes are those that ``baseline/run_random.py`` saved in the logs of the results directory, in
files named like ``random_css_n30_k04_d5_s0-99.txt`` with a line ``seed distance count`` for each
code.  The figure ``random_counts_css_n30_k04.pdf`` is a histogram of ``count``, the number of
minimum-weight X-only plus Z-only logical operators, over the random codes of the code type with
N qubits and K logical qubits.  The bars are stacked by the distance of the code.  A code that
appears in several files with overlapping seeds is counted once.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib
import numpy as np
from matplotlib.figure import Figure

from ..baseline.run_random import read_random_counts
from ..core.cli import add_results_dir_argument, make_parser
from ..core.paths import get_figures_dir, get_logs_dir
from ..core.proposals import CODE_TYPES, CodeType
from .common import (
    CODE_TYPE_TITLES,
    PLOT_STYLE,
    save_figure,
)


def main(argv: Sequence[str] | None = None) -> None:
    """Save a histogram of the numbers of minimum-weight logical operators of random codes."""
    parser = make_parser(__doc__)
    parser.add_argument("num_qubits", type=int)
    parser.add_argument("num_logicals", type=int)
    parser.add_argument("--code-type", choices=CODE_TYPES, default="css")
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    counts = read_random_counts(
        get_logs_dir(args.results_dir), args.code_type, args.num_qubits, args.num_logicals
    )
    if not counts:
        parser.error(
            f"no files of {args.code_type} codes with {args.num_qubits} qubits and"
            f" {args.num_logicals} logical qubits in {get_logs_dir(args.results_dir)}; run"
            " baseline.run_random first"
        )
    figure = _make_histogram_figure(counts, args.code_type, args.num_qubits, args.num_logicals)
    name = f"random_counts_{args.code_type}_n{args.num_qubits}_k{args.num_logicals:02d}.pdf"
    save_figure(figure, get_figures_dir(args.results_dir) / name)


####################################################################################################
# helper methods


def _make_histogram_figure(
    counts_by_distance: Mapping[int, Sequence[int]],
    code_type: CodeType,
    num_qubits: int,
    num_logicals: int,
) -> Figure:
    """Plot a histogram of the counts of random codes, with bars stacked by code distance."""
    all_counts = [count for counts in counts_by_distance.values() for count in counts]
    with matplotlib.rc_context(PLOT_STYLE):
        figure = Figure(figsize=(3.25, 2.5))
        axis = figure.subplots()
        axis.hist(
            list(counts_by_distance.values()),
            bins=np.histogram_bin_edges(all_counts, bins="auto").tolist(),
            stacked=True,
            label=[f"d = {distance}" for distance in counts_by_distance],
        )
        axis.set_xlabel("minimum-weight logical operators")
        axis.set_ylabel("random codes")
        axis.set_title(f"{CODE_TYPE_TITLES[code_type]} [[{num_qubits},{num_logicals}]]")
        axis.legend(framealpha=0)
        figure.tight_layout(pad=0.3)
    return figure


if __name__ == "__main__":
    main()
