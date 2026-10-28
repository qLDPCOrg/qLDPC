#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Summarize the results of a search for quantum Tanner codes in a CSV file.

This script reads the files of codes saved by run_search.py, and writes their parameters to
codes.csv in --results-dir (by default, results/codes.csv in this directory), with one row per
saved code that has at least one logical qubit.  The columns of the CSV file are CSV_FIELDS:
- group_order, group_index: the group that defines the code, identified by its order and its index
  among groups of the same order in the GAP SmallGroups library,
- base_code: the identifier of the classical base code (see run_search.get_base_code),
- seed: the random seed used to construct the code,
- n, k, d: the number of physical qubits, the number of logical qubits, and an upper bound on the
  code distance,
- num_trials: the number of randomized trials used to bound the code distance, and
- stabilizer_weight: the maximum weight of a stabilizer of the code.

Example:
    python experiments/quantum_tanner/collect_results.py
"""

from __future__ import annotations

import argparse
import csv
import pathlib

import run_search as search

OUTPUT_NAME = "codes.csv"  # name of the summary file in the results directory

# columns of the summary file
CSV_FIELDS = (
    "group_order",
    "group_index",
    "base_code",
    "seed",
    "n",
    "k",
    "d",
    "num_trials",
    "stabilizer_weight",
)

####################################################################################################
# command-line interface


def main() -> None:
    """Summarize the parameters of saved quantum Tanner codes in a CSV file."""
    args = parse_args()
    rows = get_code_data(args.results_dir / search.CODES_NAME)
    output = args.results_dir / OUTPUT_NAME
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_FIELDS)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} codes to {output}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    search.add_results_dir_argument(parser)
    return parser.parse_args()


####################################################################################################
# data collection


def get_code_data(codes_dir: pathlib.Path) -> list[tuple[int | str, ...]]:
    """Collect the parameters of saved quantum Tanner codes with k > 0.

    Returns one row with the fields in CSV_FIELDS per code, sorted by group, base code, and seed.
    """
    rows: list[tuple[int | str, ...]] = []
    for path in search.get_saved_code_paths(codes_dir):
        group_order, group_index, base_code_id, seed = search.parse_code_path(path)
        num_trials, (num_qudits, dimension, distance), weight = search.read_code_header(path)

        # this code has no logical qubits, so we can ignore it
        if dimension == 0:
            continue

        # codes with k > 0 have an integer distance bound
        rows.append(
            (
                group_order,
                group_index,
                base_code_id,
                seed,
                num_qudits,
                dimension,
                int(distance),
                num_trials,
                weight,
            )
        )
    return sorted(rows)


if __name__ == "__main__":
    main()
