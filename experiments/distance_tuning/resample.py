# SPDX-License-Identifier: Apache-2.0

"""Recompute the predicted costs of completion tables of a search's layers under other samplings.

A layer of one scanned set decides whether to use completion tables by comparing a predicted cost
with ``completion_bound_max_cost_percent`` (see ``predict_completion_cost`` and
``is_within_cost_limit`` in ``qldpc.codes.distance._information_sets.completion_bounds``); a layer
of several sets also weighs the mean cost over its sets.  The prediction samples random supports of
the layer with a fixed seed, so a layer whose prediction lies near the limit may decide by the draw.
The script runs a ``distance``, ``hadamard``, or ``count`` workload in this process, with the
active defaults, until each of the given weights has made a finite prediction, and stops it before
scanning the layer of the last of them; it scans the layers of the others.  It then recomputes the
first finite prediction of each weight, usually that of its first set, under sampling seeds 0 to
``--num-seeds`` - 1, and with seed 0 under each set of changes to the tuning in ``--changes``, and
prints how many seeds keep it within the limit.  It fails, after running the whole workload, if a
weight makes no finite prediction, as a layer that the tuning does not consider for tables does not.

Example:

    python experiments/distance_tuning/resample.py hadamard/gross --weights 6
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
from typing import TYPE_CHECKING, Any, NamedTuple
from unittest import mock

import numpy as np
import numpy.typing as npt

from qldpc.codes.distance._information_sets import completion_bounds
from qldpc.codes.distance._tuning import SearchTuning

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


class Prediction(NamedTuple):
    """The arguments of a prediction of the cost of completion tables, and the cost predicted."""

    rows: npt.NDArray[np.uint64]
    weight: int
    upper_bound: int
    reuse: bool
    tuning: SearchTuning
    cost: float


def main() -> None:
    """Capture the predictions of the given weights, and recompute them."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workload", help="a distance, hadamard, or count workload")
    parser.add_argument("--weights", nargs="+", type=int, required=True, help="layer weights")
    parser.add_argument(
        "--num-seeds", type=int, default=40, help="number of sampling seeds (default: 40)"
    )
    parser.add_argument(
        "--changes",
        default=json.dumps(DEFAULT_CHANGES),
        help="a JSON list of objects of changes to the tuning (default: a few sampling fields)",
    )
    args = parser.parse_args()
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    if args.workload.split("/")[0] not in ("distance", "hadamard", "count"):
        raise ValueError("Only distance, hadamard, and count workloads predict costs of tables")
    captured = capture(args.workload, set(args.weights))
    missing = sorted(set(args.weights) - set(captured))
    if missing:
        raise ValueError(f"Weights {missing} made no finite prediction of {args.workload}")
    for weight in sorted(captured):
        prediction = captured[weight]
        costs = [predict(prediction, seed=seed) for seed in range(args.num_seeds)]
        if costs and costs[0] != prediction.cost:
            raise RuntimeError("Seed 0 does not reproduce the captured prediction")
        within = sum(completion_bounds.is_within_cost_limit(c, prediction.tuning) for c in costs)
        limit = prediction.tuning.completion_bound_max_cost_percent / 100
        print(
            f"weight {weight}, {len(prediction.rows)} rows: {prediction.cost:.4f} with seed 0;"
            f" {min(costs):.3f} to {max(costs):.3f} (median {statistics.median(costs):.3f}) with"
            f" {args.num_seeds} seeds, {within} of them within {limit:.2f}"
        )
        for changes in json.loads(args.changes):
            print(f"    {changes}: {predict(prediction, **changes):.4f}")


def capture(name: str, weights: set[int]) -> dict[int, Prediction]:
    """Run a workload until each weight has made a finite prediction, and return them."""
    original = completion_bounds.predict_completion_cost
    captured: dict[int, Prediction] = {}

    def wrapper(
        rows: npt.NDArray[np.uint64],
        weight: int,
        upper_bound: int,
        *,
        reuse: bool,
        tuning: SearchTuning,
    ) -> tuple[float, int]:
        result = original(rows, weight, upper_bound, reuse=reuse, tuning=tuning)
        if weight in weights and weight not in captured and np.isfinite(result[0]):
            captured[weight] = Prediction(
                rows.copy(), weight, upper_bound, reuse, tuning, result[0]
            )
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


def predict(prediction: Prediction, seed: int = 0, **changes: Any) -> float:
    """Recompute a captured prediction with another sampling seed or changes to its tuning.

    The seed reaches the prediction through ``np.random.default_rng``, which this patches for the
    whole process while it predicts.
    """
    default_rng = np.random.default_rng
    draws = []

    def seeded_rng(_: object) -> np.random.Generator:
        draws.append(seed)
        return default_rng(seed)

    with mock.patch.object(np.random, "default_rng", seeded_rng):
        cost, _ = completion_bounds.predict_completion_cost(
            prediction.rows,
            prediction.weight,
            prediction.upper_bound,
            reuse=prediction.reuse,
            tuning=dataclasses.replace(prediction.tuning, **changes),
        )
    # Changes to the tuning may stop the prediction before it samples, but another seed must sample.
    if not draws and not changes:
        raise RuntimeError("The prediction drew no samples, so the seed had no effect")
    return cost


if __name__ == "__main__":
    main()
