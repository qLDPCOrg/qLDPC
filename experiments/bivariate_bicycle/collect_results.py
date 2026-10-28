#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Export the results of a search for bivariate bicycle codes with toric layouts to a data file.

This script reads the disk cache written by run_search.py, and saves every code whose figure of
merit r = k d^2 / n exceeds 1 (the bound satisfied by 2D topological codes such as the toric code)
to codes.csv in --results-dir (by default, results/codes.csv in this directory).  Only results
computed with --num-trials distance trials are exported.

Example:
    python experiments/bivariate_bicycle/collect_results.py
"""

from __future__ import annotations

import argparse

import diskcache
import numpy as np
from run_search import CACHE_NAME, NUM_TRIALS, add_results_dir_argument

import qldpc.cache

OUTPUT_NAME = "codes.csv"  # name of the exported data file in the results directory

# format of each row in the exported data file
ROW_FORMAT = "%d, %d, %d, %d, %d, %d, %d, %d, %d, %.3f"

####################################################################################################
# command-line interface


def main() -> None:
    """Export bivariate bicycle codes that outperform the toric code to a data file."""
    args = parse_args()
    cache = qldpc.cache.get_disk_cache(CACHE_NAME, cache_dir=args.results_dir)
    data = get_code_data(cache, args.num_trials)
    if not data:
        print(f"No cached codes with r > 1 were found for --num-trials {args.num_trials}")

    # reshape the data to have one column per field, even if there are no rows
    rows = np.array(data, dtype=float).reshape(-1, ROW_FORMAT.count("%"))
    output = args.results_dir / OUTPUT_NAME
    np.savetxt(output, rows, header=get_header(args.num_trials), fmt=ROW_FORMAT)
    print(f"Wrote {len(data)} codes to {output}")


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=NUM_TRIALS,
        help="number of distance trials used by the search results to export"
        " (default: %(default)s)",
    )
    add_results_dir_argument(parser)
    args = parser.parse_args()
    if args.num_trials < 1:
        parser.error("--num-trials must be positive")
    if not (args.results_dir / CACHE_NAME).is_dir():
        parser.error(f"no search-results cache found in {args.results_dir}")
    return args


####################################################################################################
# data collection


def get_code_data(cache: diskcache.Cache, num_trials: int) -> list[tuple[int | float, ...]]:
    """Collect the parameters of cached bivariate bicycle codes with figure of merit r > 1.

    Only codes whose distance was bounded with num_trials trials are collected.  Returns one row
    (Rx, Ry, ax, ay, bx, by, n, k, d, r) of the data file per code.
    """
    data: list[tuple[int | float, ...]] = []
    last_print_val = -1

    for key in cache.iterkeys():
        # identify cyclic group orders and polynomial exponents
        dims, exponents, key_num_trials = key
        if key_num_trials != num_trials:
            continue

        # report progress whenever the first polynomial exponent changes
        if (print_val := exponents[0]) != last_print_val:
            last_print_val = print_val
            print(dims, exponents)

        # retrieve code parameters, skipping codes whose distance was not computed
        nn, kk, dd = cache[key]
        if dd is None:
            continue

        # only report codes that outperform the surface code
        merit = kk * dd**2 / nn
        if merit > 1:
            data.append((*dims, *exponents, nn, kk, dd, merit))

    return data


def get_header(num_trials: int) -> str:
    """Get the header of the exported data file."""
    return "\n".join(
        [
            "AUTHOR: Michael A. Perlin, 2024",
            "bivariate bicycle codes of arXiv:2308.07915, with generating polynomials",
            "    A = 1 + x + x**ax * y**ay",
            "    B = 1 + y + x**bx * y**by",
            "here x and y are generators of cyclic groups with orders Rx and Ry",
            "code parameters [[n, k, d]] indicate",
            "    n = number of physical qubits",
            "    k = number of logical qubits",
            "    d = upper bound on the code distance"
            + " (minimal weight of a nontrivial logical operator)",
            "code distance is estimated by the method of arXiv:2308.07915,"
            + f" minimizing over {num_trials} trials",
            "the last column reports r = k d^2 / n",
            "topological 2D codes such as the toric code strictly satisfy r <= 1",
            "we only keep track of codes with r > 1",
            "",
            "Rx, Ry, ax, ay, bx, by, n, k, d, r",
        ]
    )


if __name__ == "__main__":
    main()
