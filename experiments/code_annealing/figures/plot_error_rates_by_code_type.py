# SPDX-License-Identifier: Apache-2.0

"""Plot the logical error rates and discard rates of the annealed codes, one figure per code type.

Run with ``python -m experiments.code_annealing.figures.plot_error_rates_by_code_type``.  Each code
type (CSS and SWEL) gets its own figure, ``error_rates_css.pdf`` or ``error_rates_swel.pdf``, with
the logical error rate per logical qubit in the top row and the discard rate per logical qubit in
the bottom row.  The codes are spread over columns of as many codes as the color cycle has colors,
so that no color repeats within a column.

The simulations, their cache, and the options are those of ``plot_error_rates``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from ..core.cli import add_results_dir_argument, make_parser
from ..core.code_files import CodeEntry
from ..core.paths import get_cache_dir, get_figures_dir
from .common import (
    PLOT_STYLE,
    add_code_arguments,
    add_error_rate_range_arguments,
    add_legend,
    add_reference_line,
    format_error_rate_axes,
    get_error_rate_func,
    get_error_rates_from_arguments,
    get_label,
    load_selected_codes,
    plot_code,
    save_figure,
)

_LEGEND_NUM_COLS = 2


def main(argv: Sequence[str] | None = None) -> None:
    """Simulate the annealed codes and save a figure of their rates for each code type."""
    parser = make_parser(__doc__)
    add_code_arguments(parser)
    add_error_rate_range_arguments(parser)
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    error_rates = get_error_rates_from_arguments(parser, args)
    for code_type, entries in load_selected_codes(parser, args).items():
        figure = _make_code_type_figure(
            entries,
            error_rates,
            args.num_samples,
            cache_dir=get_cache_dir(args.results_dir),
        )
        save_figure(figure, get_figures_dir(args.results_dir) / f"error_rates_{code_type}.pdf")


####################################################################################################
# helper methods


def _make_code_type_figure(
    entries: Sequence[CodeEntry],
    error_rates: npt.NDArray[np.float64],
    num_samples: int,
    *,
    cache_dir: Path,
) -> Figure:
    """Plot logical error rates (top) and discard rates (bottom) of the codes of one code type."""
    max_error_rate = float(error_rates[-1])
    with matplotlib.rc_context(PLOT_STYLE):
        num_colors = len(matplotlib.rcParams["axes.prop_cycle"].by_key()["color"])
        num_cols = max(1, math.ceil(len(entries) / num_colors))
        figure = Figure(figsize=(6.5, 3.5))
        axes = figure.subplots(
            2,
            num_cols,
            sharex=True,
            sharey="row",
            squeeze=False,
            gridspec_kw={"hspace": 0.1, "wspace": 0.15},
        )

        # a full column holds num_colors codes and the reference line, and every legend has as many
        # rows as a full one so that the legends line up
        legend_rows = math.ceil((num_colors + 1) / _LEGEND_NUM_COLS)
        for column, (axis_logical, axis_discard) in enumerate(axes.T):
            for entry in entries[column * num_colors : (column + 1) * num_colors]:
                print(get_label(entry), flush=True)
                func = get_error_rate_func(entry, num_samples, max_error_rate, cache_dir=cache_dir)
                plot_code(axis_logical, axis_discard, entry, error_rates, func)
            add_reference_line(axis_logical)
            add_legend(axis_logical, nrow=legend_rows, ncol=_LEGEND_NUM_COLS)

        format_error_rate_axes(axes, error_rates, max_rate=1e-1, min_discard_rate=1e-8)
        axes[0, 0].set_ylabel("logical error rate\nper logical qubit")
        axes[1, 0].set_ylabel("discard rate\nper logical qubit")
        figure.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.11)
    return figure


if __name__ == "__main__":
    main()
