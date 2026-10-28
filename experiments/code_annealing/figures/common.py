# SPDX-License-Identifier: Apache-2.0

"""What the plotting scripts share: the annealed codes, their simulations, and figure helpers.

``load_annealed_codes`` reads the annealed codes of the results directory.  ``get_error_rate_func``
simulates a code in a code-capacity depolarizing noise model, and caches the result on disk in
``cache/`` of the results directory, so that every figure script reuses the simulations of the
others.  Helpers turn a simulation into rates and uncertainties to plot, and the rest are helpers of
the figures and the command-line options that the scripts have in common.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib
import matplotlib.axes
import matplotlib.patches as mpatches
import numpy as np
import numpy.typing as npt
from matplotlib import ticker
from matplotlib.artist import Artist
from matplotlib.figure import Figure

from qldpc import cache, codes, decoders

from ..core.code_files import CodeEntry, read_code_file, recount
from ..core.paths import find_code_files
from ..core.proposals import CODE_TYPES, CodeType

if TYPE_CHECKING:  # matplotlib.typing has RcKeyType from version 3.11
    from matplotlib.typing import RcKeyType

DEFAULT_NUM_SAMPLES = 10**7
_DEFAULT_MIN_ERROR_RATE = 1e-3
_DEFAULT_MAX_ERROR_RATE = 1e-1
_DEFAULT_NUM_ERROR_RATES = 100

CODE_TYPE_TITLES: dict[CodeType, str] = {"css": "CSS", "swel": "SWEL"}

PLOT_STYLE: dict[RcKeyType, Any] = {
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
    "figure.titlesize": 8,
}


####################################################################################################
# loading codes and simulating them


def load_annealed_codes(
    results_dir: Path,
    code_types: Sequence[CodeType] = CODE_TYPES,
    *,
    max_qubits: int | None = None,
) -> dict[CodeType, list[CodeEntry]]:
    """Load the annealed codes of each code type, in file order, with at most ``max_qubits`` qubits.

    A code type whose code file does not exist has no codes.  The simulations rely on the distance
    in the header of a code, so each code is counted again, which is quick next to a simulation.

    Raises:
        SystemExit: If no code file exists, because ``collect_data`` has not been run.
        ValueError: If the distance or count of a code differs from its header.
    """
    files = find_code_files(results_dir)
    entries_by_code_type = {
        code_type: [
            entry
            for entry in (read_code_file(files[code_type]) if code_type in files else [])
            if max_qubits is None or entry.num_qubits <= max_qubits
        ]
        for code_type in dict.fromkeys(code_types)
    }
    for entries in entries_by_code_type.values():
        for entry in entries:
            recount(entry)
    return entries_by_code_type


def get_error_rate_func(
    entry: CodeEntry, num_samples: int, max_error_rate: float, *, cache_dir: Path
) -> codes.ErrorRateFunc:
    """The logical error and discard rates of a code, simulated once and cached in ``cache_dir``.

    The cache key is the code, its distance, the number of samples, and the maximum error rate.
    """
    cached = cache.use_disk_cache(
        "error_rate_funcs",
        cache_dir=cache_dir,
        key_func=lambda entry, num_samples, max_error_rate: (
            _get_code_key(entry.code),
            entry.distance,
            int(num_samples),
            float(max_error_rate),
        ),
    )(_simulate_error_rate_func)
    return cached(entry, num_samples, max_error_rate)


####################################################################################################
# rates of a code and their uncertainties


@dataclass(frozen=True)
class _RateBand:
    """A rate at each physical error rate, with the range that its uncertainty allows."""

    rates: npt.NDArray[np.float64]
    lower: npt.NDArray[np.float64]
    upper: npt.NDArray[np.float64]


####################################################################################################
# plotting


def plot_code(
    axis_logical: matplotlib.axes.Axes,
    axis_discard: matplotlib.axes.Axes,
    entry: CodeEntry,
    error_rates: npt.NDArray[np.float64],
    func: codes.ErrorRateFunc,
) -> None:
    """Plot the logical error rate and discard rate per logical qubit of a code, with bands."""
    logical, discard = _get_rate_bands(func, error_rates, entry.num_logicals)
    (line,) = axis_logical.plot(error_rates, logical.rates, label=get_label(entry))
    axis_discard.plot(error_rates, discard.rates, color=line.get_color())
    for axis, band in [(axis_logical, logical), (axis_discard, discard)]:
        axis.fill_between(error_rates, band.lower, band.upper, color=line.get_color(), alpha=0.2)


def add_reference_line(axis: matplotlib.axes.Axes) -> None:
    """Add a line where the logical error rate equals the physical error rate."""
    axis.axline(
        (0, 0), slope=1, color="k", linestyle=":", label=r"$p_{\mathrm{log}}=p_{\mathrm{phys}}$"
    )


def add_legend(
    axis: matplotlib.axes.Axes, *, nrow: int, ncol: int | None = None, offset: float = 1.02
) -> None:
    """Add a legend above an axis, with columns of ``nrow`` entries that fill left to right.

    Each column takes entries in order, so empty entries pad the last one.

    Args:
        axis: The axis whose lines the legend lists.
        nrow: The number of entries in each column.
        ncol: The least number of columns; the legend has more if its entries need them.
        offset: The height of the bottom of the legend above the axis, in axis heights.
    """
    handles, labels = _get_legend_entries_padded(axis, nrow, ncol)
    axis.legend(
        handles,
        labels,
        bbox_to_anchor=(0.5, offset),
        bbox_transform=axis.transAxes,
        loc="lower center",
        fontsize="small",
        ncol=len(handles) // nrow,
        framealpha=0,
    )


def format_error_rate_axes(
    axes: npt.NDArray[Any],
    error_rates: npt.NDArray[np.float64],
    *,
    max_rate: float = 1,
    min_logical_rate: float = 1e-12,
    min_discard_rate: float = 1e-10,
) -> None:
    """Format a grid of logical error rate axes (top row) and discard rate axes (bottom row).

    Args:
        axes: A grid of two rows of axes, which share their x-axes, and their y-axes within each
            row.
        error_rates: The physical error rates, whose first and last set the range of the x-axes.
        max_rate: The top of every y-axis.
        min_logical_rate: The bottom of the y-axes of logical error rates.
        min_discard_rate: The bottom of the y-axes of discard rates.
    """
    for axis in axes.ravel():
        axis.loglog()
        axis.grid(which="both")
        axis.set_ylim(top=max_rate)
        axis.yaxis.set_major_locator(ticker.LogLocator(base=100))
    axes[0, 0].set_xlim(left=error_rates[0], right=error_rates[-1])
    axes[0, 0].set_ylim(bottom=min_logical_rate)
    axes[1, 0].set_ylim(bottom=min_discard_rate)
    for axis in axes[1, :]:
        axis.set_xlabel("physical error rate")


def get_label(entry: CodeEntry) -> str:
    """The label of a code in a legend or on an axis: its code parameters."""
    return f"[[{entry.num_qubits},{entry.num_logicals},{entry.distance}]]"


def save_figure(figure: Figure, path: Path) -> None:
    """Save a figure, creating the directory of its file if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context(PLOT_STYLE):
        figure.savefig(path, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved {path}")


####################################################################################################
# command-line options that the figure scripts share


def add_code_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options that select the annealed codes and the size of their simulations."""
    parser.add_argument(
        "--code-types",
        nargs="+",
        choices=CODE_TYPES,
        default=list(CODE_TYPES),
        help="code types to plot (default: all)",
    )
    parser.add_argument(
        "--max-qubits", type=int, help="skip codes with more qubits, for a quick run"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="total errors to sample for each code (default: %(default)s)",
    )
    parser.add_argument(
        "--max-error-rate",
        type=float,
        default=_DEFAULT_MAX_ERROR_RATE,
        help="largest physical error rate that the simulations cover (default: %(default)s)",
    )


def add_error_rate_range_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options of the range of physical error rates, spaced logarithmically."""
    parser.add_argument(
        "--min-error-rate",
        type=float,
        default=_DEFAULT_MIN_ERROR_RATE,
        help="smallest physical error rate to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-error-rates",
        type=int,
        default=_DEFAULT_NUM_ERROR_RATES,
        help="number of physical error rates to plot (default: %(default)s)",
    )


def load_selected_codes(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> dict[CodeType, list[CodeEntry]]:
    """Load the codes that the options select, leaving out the code types that have none."""
    entries_by_code_type = {
        code_type: entries
        for code_type, entries in load_annealed_codes(
            args.results_dir, args.code_types, max_qubits=args.max_qubits
        ).items()
        if entries
    }
    if not entries_by_code_type:
        parser.error("the options select no codes")
    return entries_by_code_type


def get_error_rates_from_arguments(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> npt.NDArray[np.float64]:
    """Logarithmically spaced physical error rates from the options, which are checked."""
    if not 0 < args.min_error_rate < args.max_error_rate <= 1:
        parser.error("error rates must satisfy 0 < --min-error-rate < --max-error-rate <= 1")
    if args.num_error_rates < 2 or args.num_samples < 1:
        parser.error("--num-error-rates must be at least 2 and --num-samples at least 1")
    return np.geomspace(args.min_error_rate, args.max_error_rate, args.num_error_rates)


####################################################################################################
# helper methods


def _get_code_key(code: codes.CSSCode) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """A hashable identifier of a code that does not depend on the basis of its generators."""
    return tuple(
        tuple(tuple(int(bit) for bit in row) for row in sector.canonicalized.matrix)
        for sector in (code.code_x, code.code_z)
    )


def _simulate_error_rate_func(
    entry: CodeEntry, num_samples: int, max_error_rate: float
) -> codes.ErrorRateFunc:
    """Simulate a code with depolarizing noise, and return its logical error and discard rates.

    The decoder is a lookup table of the X and Z errors of weight up to ``(d - 1) // 2``, which
    flags every other syndrome as a discard.  Such errors are always corrected, so sampling starts
    at weight ``(d + 1) // 2``.  This relies on the distance that the file header claims, and the
    result is a function of the physical error rate up to ``max_error_rate``.
    """
    distance = entry.distance
    return entry.code.get_logical_error_rate_func(
        num_samples,
        max_error_rate,
        min_error_weight=(distance + 1) // 2,
        decoder=decoders.lookup(max_weight=(distance - 1) // 2, add_erasure_bit=True),
    )


def _get_rate_bands(
    func: codes.ErrorRateFunc, error_rates: npt.ArrayLike, num_logicals: int
) -> tuple[_RateBand, _RateBand]:
    """The logical error rate and the discard rate per logical qubit, with their uncertainties.

    The logical error rate is the failure rate among the shots that the decoder keeps.  Its band
    runs from ``rate - error - truncation`` (not below zero) to ``rate + error``, where the
    truncation is how much the errors too heavy for the simulation to have sampled, which count as
    failures, can raise the reported rate.  The band of the discard rate runs from
    ``rate - error`` to ``rate + error``.
    """
    rates_list = np.asarray(error_rates, dtype=float).tolist()
    logical, logical_error = (np.asarray(array, dtype=float) for array in func(rates_list))
    discard, discard_error = (
        np.asarray(array, dtype=float) for array in func(rates_list, discard_rate=True)
    )
    truncation = np.asarray(func.truncation_error_bound(rates_list), dtype=float)
    return (
        _RateBand(
            logical / num_logicals,
            np.maximum(logical - logical_error - truncation, 0) / num_logicals,
            (logical + logical_error) / num_logicals,
        ),
        _RateBand(
            discard / num_logicals,
            np.maximum(discard - discard_error, 0) / num_logicals,
            (discard + discard_error) / num_logicals,
        ),
    )


def _get_legend_entries_padded(
    axis: matplotlib.axes.Axes, nrow: int, ncol: int | None = None
) -> tuple[list[Artist], list[str]]:
    """The legend handles and labels of an axis, padded with empty entries to fill all columns."""
    handles, labels = axis.get_legend_handles_labels()
    num_columns = max(ncol or 0, math.ceil(len(handles) / nrow))
    num_empty = nrow * num_columns - len(handles)
    return (
        [*handles, *(mpatches.Patch(visible=False) for _ in range(num_empty))],
        [*labels, *([""] * num_empty)],
    )
