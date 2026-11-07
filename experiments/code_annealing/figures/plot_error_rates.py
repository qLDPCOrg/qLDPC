# SPDX-License-Identifier: Apache-2.0

"""Plot the logical error rates of the annealed codes in a code-capacity depolarizing noise model.

Run with ``python -m experiments.code_annealing.figures.plot_error_rates``.  The figure
``error_rates.pdf`` has a column for each code type: the logical error rate per logical qubit of
every code in the top row, and its discard rate per logical qubit in the bottom row, as functions
of the physical error rate.  Each qubit suffers a depolarizing error with probability equal to the
physical error rate, and each CSS sector is decoded by a lookup table of the errors that the code
distance guarantees to correct, which flags the other syndromes as discarded.

The simulations are the slow part, and are cached (see ``common.py``).  Use ``--max-qubits`` and
``--code-types`` to select codes, and ``--num-samples`` for a quick run.  The figure goes to
``figures/`` of the results directory.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from ..core.cli import add_results_dir_argument, make_parser
from ..core.code_files import CodeEntry
from ..core.paths import get_cache_dir, get_figures_dir
from ..core.proposals import CodeType
from .common import (
    CODE_TYPE_TITLES,
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


def main(argv: Sequence[str] | None = None) -> None:
    """Simulate the annealed codes and save their logical error rates and discard rates."""
    parser = make_parser(__doc__)
    add_code_arguments(parser)
    add_error_rate_range_arguments(parser)
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)

    error_rates = get_error_rates_from_arguments(parser, args)
    entries_by_code_type = load_selected_codes(parser, args)
    figure = _make_error_rate_figure(
        entries_by_code_type,
        error_rates,
        args.num_samples,
        cache_dir=get_cache_dir(args.results_dir),
    )
    save_figure(figure, get_figures_dir(args.results_dir) / "error_rates.pdf")


####################################################################################################
# helper methods


def _make_error_rate_figure(
    entries_by_code_type: Mapping[CodeType, Sequence[CodeEntry]],
    error_rates: npt.NDArray[np.float64],
    num_samples: int,
    *,
    cache_dir: Path,
) -> Figure:
    """Plot logical and discard rates (top and bottom rows), with a column for each code type."""
    max_error_rate = float(error_rates[-1])
    with matplotlib.rc_context(PLOT_STYLE):
        figure = Figure(figsize=(6.5, 5.5))
        axes = figure.subplots(
            2, len(entries_by_code_type), sharex=True, sharey="row", squeeze=False
        )
        num_colors = len(matplotlib.rcParams["axes.prop_cycle"].by_key()["color"])

        for (code_type, entries), (axis_logical, axis_discard) in zip(
            entries_by_code_type.items(), axes.T
        ):
            for entry in entries:
                print(get_label(entry), flush=True)
                func = get_error_rate_func(entry, num_samples, max_error_rate, cache_dir=cache_dir)
                plot_code(axis_logical, axis_discard, entry, error_rates, func)
            add_reference_line(axis_logical)
            add_legend(axis_logical, nrow=num_colors, offset=1.08)
            axis_logical.set_title(CODE_TYPE_TITLES[code_type])

        format_error_rate_axes(axes, error_rates)
        axes[0, 0].set_ylabel("logical error rate per logical qubit")
        axes[1, 0].set_ylabel("discard rate per logical qubit")
        figure.tight_layout(pad=0.2, h_pad=0.1, w_pad=0.1)
    return figure


if __name__ == "__main__":
    main()
