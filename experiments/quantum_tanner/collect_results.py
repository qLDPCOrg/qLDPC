#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Summarize the results of a search for quantum Tanner codes in a CSV file.

This script reads the files of codes saved by run_search.py, and the exact distances computed by
compute_distances.py, if any.  It writes the parameters of the codes to codes.csv in --results-dir
(by default, results/codes.csv in this directory), with one row per distinct code that has at least
one logical qubit.  Saved codes count as distinct if their generating data (subsets, local codes,
and complex; see saved_codes.get_code_key) differ, and the saved files with the same generating
data are copies of one code.  Each row summarizes the copies of one code.  Different generating data
can still give the same code, which then has several rows.  The group, local codes, complex, and
sample of a row are those of its first copy, where copies are ordered by group (order first), local
codes, complex, and sample (enumerated before random, and then by number).  The columns of the CSV
file are CSV_FIELDS:
- group, group_order: the identifier and order of the group that defines the code (see
  saved_codes.get_group),
- code_a, code_b: the identifiers of the local codes on the two subsets of the group (see
  saved_codes.get_local_code),
- complex: "quadripartite" or "bipartite", the Cayley complex on which the code lives,
- sample: the sample identifier of the first copy, which is "s{seed}" for subsets drawn with a
  random seed, or "e{index}" for subsets at an index in an enumeration (see saved_codes),
- num_copies: the number of saved copies of the code, which may have different groups, local
  codes, or samples,
- n, k: the number of physical qubits and the number of logical qubits,
- d, d_kind: the code distance if d_kind is "exact", or else the smallest upper bound on the code
  distance among saved copies of the code, in which case d_kind is "upper_bound",
- num_trials: the total number of randomized trials used to bound the code distance, over all
  saved copies of the code, and
- stabilizer_weight: the maximum weight of a parity check (stabilizer generator) of the code.

Example:
    python experiments/quantum_tanner/collect_results.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import pathlib
from typing import TYPE_CHECKING

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.quantum_tanner.*, and would silently treat a bare "import saved_codes" as untyped.
if TYPE_CHECKING:
    from experiments.quantum_tanner import saved_codes
else:
    import saved_codes

OUTPUT_NAME = "codes.csv"  # name of the summary file in the results directory

# columns of the summary file
CSV_FIELDS = (
    "group",
    "group_order",
    "code_a",
    "code_b",
    "complex",
    "sample",
    "num_copies",
    "n",
    "k",
    "d",
    "d_kind",
    "num_trials",
    "stabilizer_weight",
)

####################################################################################################
# command-line interface


def main() -> None:
    """Summarize the parameters of saved quantum Tanner codes in a CSV file."""
    args = parse_args()
    rows = get_code_data(args.results_dir)
    output = args.results_dir / OUTPUT_NAME
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_FIELDS)
        writer.writerows(rows)
    print(f"Wrote {len(rows)} distinct codes to {output}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    saved_codes.add_results_dir_argument(parser)
    return parser.parse_args()


####################################################################################################
# data collection


def get_code_data(results_dir: pathlib.Path) -> list[tuple[int | str, ...]]:
    """Collect the parameters of the distinct saved quantum Tanner codes with k > 0.

    Returns one row with the fields in CSV_FIELDS per distinct code, sorted by group, local codes,
    complex, and sample.  Raises a ValueError if saved results contradict each other: if copies of a
    code have different parameters (n, k) or stabilizer weights, if computations disagree on its
    exact distance, or if its exact distance exceeds an upper bound.
    """
    # group the saved files by the code that they save
    paths_by_key = collections.defaultdict(list)
    for path in saved_codes.get_saved_code_paths(results_dir / saved_codes.CODES_NAME):
        paths_by_key[saved_codes.get_code_key(path)].append(path)
    exact_distances = get_exact_distances(results_dir)

    keyed_rows: list[tuple[tuple[int | str | bool, ...], tuple[int | str, ...]]] = []
    for key, paths in paths_by_key.items():
        # read the header of every copy of this code, and check that their parameters agree
        headers = [saved_codes.read_code_header(path) for path in paths]
        params_and_weights = {(nn, kk, weight) for _, (nn, kk, _), weight in headers}
        if len(params_and_weights) > 1:
            raise ValueError(
                f"Copies of a code have different parameters or stabilizer weights: {paths}"
            )
        num_qudits, dimension, weight = params_and_weights.pop()

        # this code has no logical qubits, so we can ignore it
        if dimension == 0:
            continue

        # identify the best known distance of this code; codes with k > 0 have integer bounds
        upper_bound = min(int(distance) for _, (_, _, distance), _ in headers)
        num_trials = sum(trials for trials, _, _ in headers)
        if key in exact_distances:
            distance, distance_kind = exact_distances[key], "exact"
            if distance > upper_bound:
                raise ValueError(
                    f"Exact distance {distance} exceeds upper bound {upper_bound}: {paths}"
                )
        else:
            distance, distance_kind = upper_bound, "upper_bound"

        # identify this code by its first copy
        spec, sample_id = min(
            (saved_codes.parse_code_path(path) for path in paths), key=get_spec_sort_key
        )
        row = (
            spec.group_id,
            saved_codes.get_group_order(spec.group_id),
            spec.code_a_id,
            spec.code_b_id,
            "bipartite" if spec.bipartite else "quadripartite",
            sample_id,
            len(paths),
            num_qudits,
            dimension,
            distance,
            distance_kind,
            num_trials,
            weight,
        )
        keyed_rows.append((get_spec_sort_key((spec, sample_id)), row))
    return [row for _, row in sorted(keyed_rows)]


def get_exact_distances(results_dir: pathlib.Path) -> dict[str, int]:
    """Get the exact distances computed by compute_distances.py, keyed by code.

    Raises a ValueError if two computations disagree on the exact distance of a code.
    """
    exact_distances: dict[str, int] = {}
    for record in saved_codes.read_exact_distance_records(results_dir):
        if record["status"] != "exact":
            continue
        key, distance = record["code_key"], int(record["distance"])
        if exact_distances.setdefault(key, distance) != distance:
            raise ValueError(
                f"Computations disagree on the exact distance of {record['code_file']}"
            )
    return exact_distances


def get_spec_sort_key(
    spec_and_sample_id: tuple[saved_codes.CodeSpec, str],
) -> tuple[int | str | bool, ...]:
    """Get a key that sorts specifications and samples by group, local codes, complex, and sample.

    Groups are sorted by their orders first; see saved_codes.get_group_sort_key.
    """
    spec, sample_id = spec_and_sample_id
    group_key = saved_codes.get_group_sort_key(spec.group_id)
    sample_key = saved_codes.get_sample_sort_key(sample_id)
    return (*group_key, spec.code_a_id, spec.code_b_id, spec.bipartite, *sample_key)


if __name__ == "__main__":
    main()
