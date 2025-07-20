from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

import qldpc


def get_label(
    code: qldpc.codes.ClassicalCode | qldpc.codes.QuditCode,
    distance_estimation_trials: bool | int = False,
) -> str:
    """Get a label for a code in a figure."""
    known_distance = code.get_distance_if_known()
    if isinstance(known_distance, int):
        return f"$d={known_distance}$"
    if not distance_estimation_trials:
        return f"[{len(code)}, {code.dimension}]"
    distance_estimate = code.get_distance_bound(num_trials=int(distance_estimation_trials))
    return f"[{len(code)}, {code.dimension}, <= {distance_estimate}]"


def make_error_rate_figure(
    codes: Sequence[qldpc.codes.ClassicalCode | qldpc.codes.CSSCode],
    num_samples: int = 10**4,
    physical_rates: Sequence[float] = list(np.logspace(-2, -0.1, 100)),
    figsize: tuple[int, int] = (5, 4),
    distance_estimation_trials: bool | int = False,
    **decoding_args: object,
) -> tuple[plt.Figure, plt.Axes]:
    """Make a figure comparing physical vs. logical error rates in a code family."""
    figure, axis = plt.subplots(figsize=figsize)

    for code in codes:
        get_logical_error_rate = code.get_logical_error_rate_func(
            num_samples,
            max(physical_rates),
            **decoding_args,  # type:ignore[arg-type]
        )
        logical_rates, stderrs = get_logical_error_rate(physical_rates)
        label = get_label(code, distance_estimation_trials)
        line, *_ = axis.plot(physical_rates, logical_rates, label=label)
        axis.fill_between(
            physical_rates,
            logical_rates - stderrs,
            logical_rates + stderrs,
            color=line.get_color(),
            alpha=0.2,
        )

    axis.axline(
        (0, 0),
        slope=1,
        color="k",
        linestyle=":",
        label=r"$p_{\mathrm{log}}=p_{\mathrm{phys}}$",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(right=1)
    axis.set_ylim(bottom=max(min(physical_rates) ** 2, axis.get_ylim()[0]), top=1)
    axis.set_xlabel(r"physical error rate $p_{\mathrm{phys}}$")
    axis.set_ylabel(r"logical error rate $p_{\mathrm{log}}$")
    axis.legend(loc="best")
    figure.tight_layout()

    return figure, axis
