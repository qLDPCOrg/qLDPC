# SPDX-License-Identifier: Apache-2.0

"""Compare simulated logical error rates with estimates from minimum-weight logical operators.

The annealing ranks codes of one distance by their numbers of minimum-weight logical operators, on
the premise that fewer such operators mean fewer logical errors at low physical error rates.  This
figure checks that premise against simulations of the annealed codes.

Run with ``python -m experiments.code_annealing.figures.plot_error_estimates``.  The figure
``error_estimates.pdf`` has a panel for each code type, with the logical error rate per logical
qubit of every code at one physical error rate: the simulated rate, an estimate from the numbers of
minimum-weight logical operators, and a refined estimate from the numbers of distinct subsets of
their supports, which also leaves out the errors that the decoder discards.  The codes are grouped
by distance, and sorted by decreasing simulated rate per logical qubit within each group.  Their
colors identify the code sizes ``(n, k)``, in order of size.

The counts of operators and subsets come from ``data/half_supports.txt`` of the results directory,
which ``analysis/half_supports.py`` writes and reads.  Codes without counts are skipped.  The
simulations and their cache are those of ``plot_error_rates`` (see ``common.py``), and
``--error-rate`` sets the physical error rate of the comparison.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import lines
from matplotlib.figure import Figure

from ..analysis.half_supports import HalfSupportCounts, read_half_supports
from ..core.cli import add_results_dir_argument, make_parser
from ..core.code_files import CodeEntry, CodeParams
from ..core.paths import CODE_FILE_NAMES, get_cache_dir, get_figures_dir, get_half_supports_file
from ..core.proposals import CodeType
from .common import (
    CODE_TYPE_TITLES,
    PLOT_STYLE,
    add_code_arguments,
    get_error_rate_func,
    get_label,
    load_annealed_codes,
    save_figure,
)

_DEFAULT_ERROR_RATE = 1e-3


def main(argv: Sequence[str] | None = None) -> None:
    """Simulate the annealed codes with counts, and save their simulated and estimated rates."""
    parser = make_parser(__doc__)
    add_code_arguments(parser)
    parser.add_argument(
        "--error-rate",
        type=float,
        default=_DEFAULT_ERROR_RATE,
        help="physical error rate to compare at (default: %(default)s)",
    )
    add_results_dir_argument(parser)
    args = parser.parse_args(argv)
    if not 0 < args.error_rate <= args.max_error_rate <= 1 or args.num_samples < 1:
        parser.error("require 0 < --error-rate <= --max-error-rate <= 1 and --num-samples >= 1")

    # Read the counts, then simulate and estimate the codes of each code type that have counts.
    half_supports_file = get_half_supports_file(args.results_dir)
    try:
        counts_by_file = read_half_supports(half_supports_file)
    except (OSError, ValueError) as error:
        parser.error(f"cannot read {half_supports_file} (run collect_data first): {error}")
    estimates_by_code_type: dict[CodeType, list[_ErrorEstimate]] = {}
    for code_type, entries in load_annealed_codes(
        args.results_dir, args.code_types, max_qubits=args.max_qubits
    ).items():
        counts = counts_by_file.get(CODE_FILE_NAMES[code_type], {})
        estimates = _get_error_estimates(
            entries,
            counts,
            args.error_rate,
            args.num_samples,
            args.max_error_rate,
            cache_dir=get_cache_dir(args.results_dir),
        )
        if estimates:
            estimates_by_code_type[code_type] = estimates
    if not estimates_by_code_type:
        parser.error(f"{half_supports_file} has no counts for the selected codes")
    save_figure(
        _make_error_estimates_figure(estimates_by_code_type),
        get_figures_dir(args.results_dir) / "error_estimates.pdf",
    )


####################################################################################################
# helper methods


@dataclass(frozen=True)
class _ErrorEstimate:
    """The simulated and estimated logical error rates of a code at a physical error rate."""

    num_qubits: int
    num_logicals: int
    distance: int
    simulated: float
    estimate: float
    refined_estimate: float


def _get_error_estimates(
    entries: Sequence[CodeEntry],
    counts: Mapping[CodeParams, HalfSupportCounts],
    error_rate: float,
    num_samples: int,
    max_error_rate: float,
    *,
    cache_dir: Path,
) -> list[_ErrorEstimate]:
    """Simulate the codes that have counts, and estimate their logical error rates.

    The result is sorted by distance and then by decreasing simulated rate per logical qubit.  A
    code that has no counts is skipped, and counts that disagree with the code are an error.
    """
    estimates = []
    for entry in entries:
        code_counts = counts.get(entry.params)
        if code_counts is None:
            print(f"{get_label(entry)} has no counts, so it is skipped")
            continue
        if code_counts.count_x + code_counts.count_z != entry.multiplicity:
            raise ValueError(
                f"The code {get_label(entry)} has {entry.multiplicity} minimum-weight logical"
                " operators, but the counts add up to"
                f" {code_counts.count_x} + {code_counts.count_z}"
            )
        print(get_label(entry), flush=True)
        func = get_error_rate_func(entry, num_samples, max_error_rate, cache_dir=cache_dir)
        simulated, _ = func(error_rate)
        estimates.append(
            _ErrorEstimate(
                entry.num_qubits,
                entry.num_logicals,
                entry.distance,
                simulated,
                _estimate_error_rate(code_counts, error_rate),
                _refine_error_rate_estimate(code_counts, error_rate),
            )
        )
    return sorted(estimates, key=lambda row: (row.distance, -row.simulated / row.num_logicals))


def _estimate_error_rate(counts: HalfSupportCounts, error_rate: float) -> float:
    """Estimate the logical error rate of a code under depolarizing noise from its counts.

    Each qubit has an X-type error with probability ``r = 2 p / 3`` for a physical error rate ``p``,
    and the X and Z sectors are decoded separately, as in the simulations: by a lookup table of the
    errors of weight up to ``(d - 1) // 2``, which discards all other syndromes.  This decoder
    applies a minimum-weight logical operator of weight ``d`` when ``w = d // 2 + 1`` qubits of its
    support err, because the rest of the support is then in the table.  For even ``d``, errors on
    exactly half of a support are discarded, so they add to the discard rate and not to this one.

    The estimate multiplies the number of minimum-weight operators by the ``comb(d, w)`` subsets of
    each support and by ``r**w``, the probability that every qubit in a subset errs.
    """
    distance = counts.distance
    weight = distance // 2 + 1
    rate = 2 * error_rate / 3
    return (counts.count_x + counts.count_z) * math.comb(distance, weight) * rate**weight


def _refine_error_rate_estimate(counts: HalfSupportCounts, error_rate: float) -> float:
    """Estimate the logical error rate of a code from its numbers of distinct majority supports.

    The noise, the decoder, ``r``, and ``w`` are those of ``_estimate_error_rate``.  This estimate
    counts the distinct subsets of ``w`` qubits of all supports (the majority supports), which are
    fewer than ``comb(d, w)`` per operator if supports overlap, and gives each the probability
    ``r**w * (1 - r)**(n - w)`` that exactly its qubits err.

    It also leaves out the patterns that the decoder discards rather than fails on.  Each X-type
    error is a Y error with probability 1/2, which adds a Z-type error on its qubit, and a pattern
    with more than ``t = (d - 1) // 2`` Y errors leaves a Z-type error that the lookup table of the
    other sector does not hold, so the shot is discarded.  A pattern thus fails with probability
    ``sum(comb(w, j) for j <= t) / 2**w``, and likewise with X and Z swapped.  The sectors are then
    combined as if they were otherwise independent, ``1 - (1 - P_x) * (1 - P_z)``.
    """
    num_qubits, distance = counts.num_qubits, counts.distance
    weight = distance // 2 + 1
    rate = 2 * error_rate / 3
    kept = sum(math.comb(weight, num_y) for num_y in range((distance - 1) // 2 + 1)) / 2**weight
    pattern_probability = rate**weight * (1 - rate) ** (num_qubits - weight) * kept
    failure_x = counts.num_majority_supports_x * pattern_probability
    failure_z = counts.num_majority_supports_z * pattern_probability
    return 1 - (1 - failure_x) * (1 - failure_z)


def _make_error_estimates_figure(
    estimates_by_code_type: Mapping[CodeType, Sequence[_ErrorEstimate]],
) -> Figure:
    """Plot simulated and estimated logical error rates per logical qubit, a panel per code type."""
    sizes = sorted(
        {
            (row.num_qubits, row.num_logicals)
            for rows in estimates_by_code_type.values()
            for row in rows
        }
    )
    with matplotlib.rc_context(PLOT_STYLE):
        colors = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        size_colors = {size: colors[index % len(colors)] for index, size in enumerate(sizes)}
        figure = Figure(figsize=(6.5, 2.4 * len(estimates_by_code_type)))
        axes = figure.subplots(len(estimates_by_code_type), 1, sharey=True, squeeze=False)[:, 0]

        for axis, (code_type, rows) in zip(axes, estimates_by_code_type.items()):
            # A column of markers for each code, colored by its size, with lines between distances.
            for index, row in enumerate(rows):
                color = size_colors[(row.num_qubits, row.num_logicals)]
                for rate, marker in [
                    (row.simulated, "o"),
                    (row.estimate, "x"),
                    (row.refined_estimate, "+"),
                ]:
                    axis.semilogy(index + 0.5, rate / row.num_logicals, marker, color=color)
            for index in range(1, len(rows)):  # separate the distances
                if rows[index].distance != rows[index - 1].distance:
                    axis.axvline(index, color="k", linewidth=0.8, linestyle="--")

            # Label each column by its code, between the major ticks that bound the columns.
            axis.set_xticks(np.arange(len(rows) + 1))
            axis.set_xticklabels([])
            axis.set_xticks(np.arange(len(rows)) + 0.9, minor=True)
            axis.set_xticklabels(
                [f"({row.num_qubits},{row.num_logicals},{row.distance})" for row in rows],
                minor=True,
                rotation=60,
                ha="right",
                fontsize=5,
            )
            axis.tick_params(axis="x", which="minor", length=0)
            axis.set_xlim(0, len(rows))
            axis.grid(axis="y", which="both")
            axis.set_title(CODE_TYPE_TITLES[code_type])

        # One legend for the markers, which mean the same in every panel.
        axes[0].legend(
            handles=[
                lines.Line2D([0], [0], marker=marker, color="k", linestyle="none", label=label)
                for marker, label in [
                    ("o", "simulation"),
                    ("x", "estimate"),
                    ("+", "refined estimate"),
                ]
            ],
            loc="upper right",
            framealpha=1,
        )
        for axis in axes:
            axis.set_ylabel("logical error rate\nper logical qubit")
        figure.tight_layout(pad=0.3, h_pad=1.5, w_pad=0.1)
    return figure


if __name__ == "__main__":
    main()
