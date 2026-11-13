# SPDX-License-Identifier: Apache-2.0

"""Recompute the predicted costs of completion tables of a search's layers under other samplings.

A layer decides whether to use completion tables by comparing a predicted cost with
``completion_bound_max_cost_percent`` (see ``predict_completion_cost`` in
``qldpc.codes.distance._information_sets.completion_bounds``).  The prediction samples random
supports of the layer with a fixed seed, so a layer whose prediction lies near the limit may decide
by the draw.  The script runs a ``distance``, ``hadamard``, or ``count`` workload in this process,
with the active defaults, until each of the given weights has made a finite prediction, and stops
it there.  It then recomputes the first finite prediction of each such weight under sampling seeds
0 to ``--seeds`` - 1, and under each set of changes to the tuning in ``--changes``, and prints how
many seeds keep it within the limit.  It scans no layer of the given weights.

Example:

    python experiments/distance_tuning/resample.py hadamard/gross --weights 6
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from typing import TYPE_CHECKING, Any

import numpy as np

from qldpc.codes.distance._information_sets import completion_bounds

# Scripts in this directory import each other as top-level modules.  mypy knows these files only as
# experiments.distance_tuning.*, and would silently treat a bare "import workloads" as untyped.
if TYPE_CHECKING:
    from experiments.distance_tuning import workloads
else:
    import workloads

DEFAULT_CHANGES = [
    {"completion_bound_confidence_z": 1.0},
    {"completion_bound_confidence_z": 3.0},
    {"completion_bound_samples": 128},
    {"completion_bound_samples": 512},
    {"completion_bound_samples": 4096},
    {"completion_bound_kept_first_row_ratio": 0.31},
    {"completion_bound_kept_first_row_ratio": 0.38},
]


class _Captured(Exception):
    """Stops the workload once every weight has made a prediction."""


def main() -> None:
    """Capture the predictions of the given weights, and recompute them."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workload", help="a distance, hadamard, or count workload")
    parser.add_argument("--weights", nargs="+", type=int, required=True, help="layer weights")
    parser.add_argument("--seeds", type=int, default=40, help="sampling seeds (default: 40)")
    parser.add_argument(
        "--changes",
        default=json.dumps(DEFAULT_CHANGES),
        help="a JSON list of objects of changes to the tuning (default: a few sampling fields)",
    )
    args = parser.parse_args()
    captured = capture(args.workload, set(args.weights))
    for arguments in captured.values():
        rows, weight, _, _, tuning, cost = arguments
        limit = tuning.completion_bound_max_cost_percent / 100
        costs = [predict(arguments, seed=seed) for seed in range(args.seeds)]
        within = sum(cost <= limit for cost in costs)
        print(
            f"weight {weight}, {len(rows)} rows: {cost:.4f} with seed 0; {min(costs):.3f} to"
            f" {max(costs):.3f} (median {statistics.median(costs):.3f}) with {args.seeds} seeds,"
            f" {within} of them within {limit:.2f}"
        )
        for changes in json.loads(args.changes):
            print(f"    {changes}: {predict(arguments, **changes):.4f}")


def capture(name: str, weights: set[int]) -> dict[int, tuple[Any, ...]]:
    """Run a workload until each weight has made a finite prediction, and return their arguments."""
    original = completion_bounds.predict_completion_cost
    captured: dict[int, tuple[Any, ...]] = {}

    def wrapper(rows: Any, weight: int, upper_bound: int, *, reuse: bool, tuning: Any) -> Any:
        result = original(rows, weight, upper_bound, reuse=reuse, tuning=tuning)
        if weight in weights and weight not in captured and np.isfinite(result[0]):
            captured[weight] = (rows.copy(), weight, upper_bound, reuse, tuning, result[0])
            if set(captured) == weights:
                raise _Captured
        return result

    completion_bounds.predict_completion_cost = wrapper
    try:
        workloads.run_workload(name)
    except _Captured:
        pass
    finally:
        completion_bounds.predict_completion_cost = original
    return captured


def predict(arguments: tuple[Any, ...], seed: int = 0, **changes: Any) -> float:
    """Recompute a captured prediction with another sampling seed or changes to its tuning."""
    rows, weight, upper_bound, reuse, tuning, _ = arguments
    default_rng = np.random.default_rng

    def seeded_rng(_: object) -> np.random.Generator:
        return default_rng(seed)

    completion_bounds.np.random.default_rng = seeded_rng  # type: ignore[assignment]
    try:
        cost, _ = completion_bounds.predict_completion_cost(
            rows, weight, upper_bound, reuse=reuse, tuning=dataclasses.replace(tuning, **changes)
        )
    finally:
        completion_bounds.np.random.default_rng = default_rng
    return cost


if __name__ == "__main__":
    main()
