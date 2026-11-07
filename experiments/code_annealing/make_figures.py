# SPDX-License-Identifier: Apache-2.0

"""Plot the figures of the experiment from the data of ``collect_data.py``.

Run with
``python -m experiments.code_annealing.make_figures [--max-qubits N] [--num-samples N] [--test]``.
It runs the plotting scripts of ``figures/``, in order, on the same results directory, and saves
their PDF files to ``figures/`` of that directory:

1. ``plot_error_rates`` saves ``error_rates.pdf``: the logical error rates and discard rates of the
   annealed codes of both code types.
2. ``plot_error_rates_by_code_type`` saves ``error_rates_css.pdf`` and ``error_rates_swel.pdf``: the
   same curves, with one figure for each code type.
3. ``plot_error_estimates`` saves ``error_estimates.pdf``: the simulated logical error rates at one
   physical error rate, next to estimates from the counts in ``data/half_supports.txt``.
4. ``plot_random_histogram`` saves ``random_counts_{code_type}_n{N}_k{KK}.pdf`` for every set of
   random codes that ``baseline/run_random.py`` saved to the logs: how many minimum-weight logical
   operators random codes have.

The simulations of the first three scripts take most of the running time, and are cached in
``cache/``.  With ``--workers N``, the codes are first simulated in N parallel processes, and the
plotting scripts then read their simulations from the cache.  Pass a small ``--max-qubits`` and
``--num-samples`` for a quick look.  ``--test`` plots the data of the smoke-test campaign of
``write_jobs.py --test``, with 1000 samples for each code.
Each plotting script can be run alone, with more options, as
``python -m experiments.code_annealing.figures.NAME``.
"""

from __future__ import annotations

from collections.abc import Sequence

from .baseline.run_random import find_random_code_sizes
from .core.cli import (
    add_results_dir_argument,
    add_test_argument,
    get_results_dir,
    make_parser,
    run_step,
)
from .core.paths import find_code_files, get_cache_dir, get_half_supports_file, get_logs_dir
from .figures import (
    plot_error_estimates,
    plot_error_rates,
    plot_error_rates_by_code_type,
    plot_random_histogram,
)
from .figures.common import DEFAULT_NUM_SAMPLES, load_annealed_codes, simulate_codes

_TEST_NUM_SAMPLES = 1000


def main(argv: Sequence[str] | None = None) -> None:
    """Save every figure of the experiment."""
    parser = make_parser(__doc__)
    parser.add_argument("--max-qubits", type=int, help="skip codes with more qubits")
    parser.add_argument(
        "--num-samples",
        type=int,
        help=f"errors to sample for each code (default: {DEFAULT_NUM_SAMPLES}, or"
        f" {_TEST_NUM_SAMPLES} with --test)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="processes that simulate the codes in parallel (default: %(default)s)",
    )
    add_results_dir_argument(parser)
    add_test_argument(parser)
    args = parser.parse_args(argv)

    results_dir = get_results_dir(args)
    find_code_files(results_dir)
    half_supports = get_half_supports_file(results_dir)
    if not half_supports.exists():
        parser.error(f"{half_supports} does not exist: run collect_data first")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    options = ["--results-dir", str(results_dir)]
    simulation_options = [*options]
    if args.max_qubits is not None:
        simulation_options += ["--max-qubits", str(args.max_qubits)]
    num_samples = args.num_samples
    if num_samples is None and args.test:
        num_samples = _TEST_NUM_SAMPLES
    if num_samples is not None:
        simulation_options += ["--num-samples", str(num_samples)]
    if args.workers > 1:
        entries_by_code_type = load_annealed_codes(results_dir, max_qubits=args.max_qubits)
        simulate_codes(
            [entry for entries in entries_by_code_type.values() for entry in entries],
            DEFAULT_NUM_SAMPLES if num_samples is None else num_samples,
            cache_dir=get_cache_dir(results_dir),
            workers=args.workers,
        )
    run_step("plot_error_rates", plot_error_rates.main, simulation_options)
    run_step(
        "plot_error_rates_by_code_type", plot_error_rates_by_code_type.main, simulation_options
    )
    run_step("plot_error_estimates", plot_error_estimates.main, simulation_options)
    for code_type, num_qubits, num_logicals in find_random_code_sizes(get_logs_dir(results_dir)):
        histogram_options = [str(num_qubits), str(num_logicals), "--code-type", code_type, *options]
        run_step("plot_random_histogram", plot_random_histogram.main, histogram_options)


if __name__ == "__main__":
    main()
