#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Collect publication-scale lattice-surgery logical error rates.

Each preset compares a lattice-surgery experiment against a quantum memory baseline, sampling both
with Sinter under circuit-level depolarizing noise (without idling errors) at several physical error
rates p:
- bb72: measure a logical X operator of the [[72, 12, 6]] bivariate bicycle code (9 rounds), versus
  memory of the same code (9 rounds).
- bb18: measure a weight-20 logical operator of the [[248, 10]] bivariate bicycle code bb_18 of Cain
  et al. (arXiv:2603.28627) with a boosted gadget (15 rounds), versus memory of the same code
  (9 rounds).
- steane-joint: jointly measure logical Z operators of two Steane codes (9 rounds), versus memory of
  one Steane code (9 rounds).

Every preset decodes with belief propagation plus localized statistics decoding (BP+LSD; see
DECODER_OPTIONS).  Every circuit may track several logical observables, but each task keeps only the
first one, so a logical error is a flip of that observable.

Raw Sinter progress is saved to a resumable file, and aggregate results are written to a CSV file
together with their provenance: the Git state of this repository, the decoder, and the collection
request.  By default, both files are written to --results-dir (results/ in this directory) as
{preset}_{run_id}_progress.csv and {preset}.csv.  See README.md in this directory for example
commands and details on resuming.

Example:
    python experiments/lattice_surgery/collect_ler.py --preset bb72
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import sinter
import stim
import sympy

from qldpc import circuits, codes, decoders
from qldpc.experimental.surgery import (
    boost_gadget,
    build_bridge,
    build_gadget,
    build_joint_ppm_circuit,
    build_single_ppm_circuit,
    keep_only_observable,
)
from qldpc.objects import Pauli

Preset = Literal["bb72", "bb18", "steane-joint"]

RESULTS_DIR = Path(__file__).parent / "results"  # default directory of all results

# default physical error rates to sample for each preset
DEFAULT_P_VALUES: dict[Preset, tuple[float, ...]] = {
    "bb72": (0.0008, 0.0011, 0.0014, 0.0019, 0.0025),
    "bb18": (0.006, 0.007, 0.008),
    "steane-joint": (0.0002, 0.0004, 0.0008),
}

# options of the BP+LSD decoder (decoders.bp_lsd) for each preset
DECODER_OPTIONS: dict[Preset, dict[str, Any]] = {
    "bb72": {"max_iter": 20, "bp_method": "ms", "lsd_method": "LSD_CS", "lsd_order": 5},
    "bb18": {
        "max_iter": 100,
        "bp_method": "ms",
        "ms_scaling_factor": 0.0,
        "schedule": "serial",
        "lsd_method": "LSD_E",
        "lsd_order": 5,
    },
    "steane-joint": {"max_iter": 20, "bp_method": "ms", "lsd_method": "LSD_CS", "lsd_order": 5},
}

# Support of the weight-20 logical operator measured by the bb18 preset: a logical Z operator of
# bb_18, which becomes a logical X operator after exchanging the X and Z checks of bb_18.  This
# representative and the boost seed in get_bb18_experiments were found by an offline stochastic
# BP+OSD search; see examples/experimental/lattice_surgery.ipynb.
BB18_LOGICAL_SUPPORT = (
    8,
    9,
    14,
    18,
    24,
    34,
    40,
    56,
    75,
    76,
    97,
    111,
    122,
    171,
    202,
    208,
    213,
    218,
    228,
    238,
)

# columns of the aggregate CSV file
CSV_FIELDS = (
    "preset",
    "kind",
    "p",
    "rounds",
    "shots",
    "errors",
    "discards",
    "seconds",
    "git_state",
    "decoder_settings",
    "collection_request",
)

####################################################################################################
# command-line interface


def main() -> None:
    """Collect logical error rates for one preset, and write them to an aggregate CSV file."""
    args = parse_args()
    preset: Preset = args.preset
    p_values = tuple(args.p_values or DEFAULT_P_VALUES[preset])
    decoder_settings = {"decoder": "bp_lsd", **DECODER_OPTIONS[preset]}

    # Identify tasks by everything that determines their circuits and decoder, but not by collection
    # limits, which may therefore change when resuming.
    task_definition = {"preset": preset, "git_state": get_git_state(), "decoder": decoder_settings}
    run_id = hashlib.sha256(json.dumps(task_definition, sort_keys=True).encode()).hexdigest()[:16]
    provenance = {**task_definition, "run_id": run_id}
    tasks = get_tasks(get_experiments(preset), p_values, provenance)

    resume = args.resume or args.results_dir / f"{preset}_{run_id}_progress.csv"
    output = args.output or args.results_dir / f"{preset}.csv"
    resume.parent.mkdir(parents=True, exist_ok=True)
    stats = sinter.collect(
        tasks=tasks,
        decoders=["custom"],
        custom_decoders={"custom": get_sinter_decoder(preset)},
        num_workers=args.workers,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        print_progress=True,
        save_resume_filepath=resume,
    )
    requested_tasks = {get_task_key(task.json_metadata) for task in tasks}
    stats = get_stats_for_run(stats, run_id, requested_tasks)

    collection_request = {
        "p_values": p_values,
        "max_shots": args.max_shots,
        "max_errors": args.max_errors,
        "workers": args.workers,
    }
    write_aggregate_csv(output, preset, stats, collection_request)
    print(f"Wrote {len(stats)} aggregate rows to {output}")
    print(f"Raw resumable progress remains in {resume}")


def parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--preset",
        choices=get_args(Preset),
        required=True,
        help="the experiment to run",
    )
    parser.add_argument(
        "--p-values",
        type=float,
        nargs="+",
        help="physical error rates to sample (default: preset-specific; see DEFAULT_P_VALUES)",
    )
    parser.add_argument(
        "--max-shots",
        type=int,
        default=1_000_000,
        help="maximum number of shots per task (default: %(default)s)",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=100,
        help="stop sampling a task after this many logical errors (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="number of parallel Sinter workers (default: %(default)s)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="directory of progress and aggregate CSV files"
        " (default: the 'results' directory beside this script)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Sinter progress file to resume from and append to"
        " (default: {preset}_{run_id}_progress.csv in the results directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="path of the aggregate CSV file to write"
        " (default: {preset}.csv in the results directory)",
    )
    args = parser.parse_args()
    if args.p_values is not None and not all(0 <= p <= 1 for p in args.p_values):
        parser.error("--p-values must be probabilities between 0 and 1")
    for name in ["max_shots", "max_errors", "workers"]:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


####################################################################################################
# experiments


@dataclasses.dataclass(frozen=True)
class Experiment:
    """A family of circuits, labeled by kind, that is sampled at every physical error rate.

    get_circuit builds a circuit with a given number of syndrome measurement rounds and noise model.
    """

    kind: str
    rounds: int
    get_circuit: Callable[[int, circuits.NoiseModel], stim.Circuit]


def get_experiments(preset: Preset) -> list[Experiment]:
    """Get the lattice-surgery experiment and memory baseline of a preset."""
    builders: dict[Preset, Callable[[], list[Experiment]]] = {
        "bb72": get_bb72_experiments,
        "bb18": get_bb18_experiments,
        "steane-joint": get_steane_joint_experiments,
    }
    return builders[preset]()


def get_bb72_experiments() -> list[Experiment]:
    """Measure a logical X operator of the [[72, 12, 6]] bivariate bicycle code, versus memory."""
    x, y = sympy.symbols("x y")
    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    logical = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    gadget = build_gadget(code, logical, basis=Pauli.X)
    return [
        Experiment(
            "surgery",
            rounds=9,
            get_circuit=lambda num_rounds, noise_model: build_single_ppm_circuit(
                gadget, rounds=num_rounds, noise_model=noise_model
            ),
        ),
        get_memory_experiment("memory", code, Pauli.X, rounds=9),
    ]


def get_bb18_experiments() -> list[Experiment]:
    """Measure a weight-20 logical operator of the bb_18 code of Cain et al., versus memory."""
    x, y = sympy.symbols("x y")
    source = codes.BBCode((31, 4), 1 + x**6 * y + x**27, y**2 + x**15 * y**3 + x**24)
    # exchange X and Z checks, so that the logical Z operator of bb_18 becomes a logical X operator
    code = codes.CSSCode(source.matrix_z, source.matrix_x, is_subsystem_code=False)
    logical = np.zeros(code.num_qudits, dtype=np.uint8)
    logical[list(BB18_LOGICAL_SUPPORT)] = 1
    gadget = boost_gadget(
        build_gadget(code, logical, basis=Pauli.X),
        method="combinatorial",
        target=1.0,
        max_extra_qubits=20,
        seed=2,
    )
    return [
        Experiment(
            "surgery",
            rounds=15,
            get_circuit=lambda num_rounds, noise_model: build_single_ppm_circuit(
                gadget, rounds=num_rounds, noise_model=noise_model
            ),
        ),
        get_memory_experiment("memory", code, Pauli.X, rounds=9),
    ]


def get_steane_joint_experiments() -> list[Experiment]:
    """Jointly measure logical Z operators of two Steane codes, versus memory of one Steane code."""
    code_l, code_r = codes.SteaneCode(), codes.SteaneCode()
    logical_l = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    logical_r = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    gadget_l = build_gadget(code_l, logical_l, basis=Pauli.Z)
    gadget_r = build_gadget(code_r, logical_r, basis=Pauli.Z)
    bridge = build_bridge(gadget_l, gadget_r)
    return [
        Experiment(
            "joint_surgery",
            rounds=9,
            get_circuit=lambda num_rounds, noise_model: build_joint_ppm_circuit(
                gadget_l, gadget_r, bridge, rounds=num_rounds, noise_model=noise_model
            )[0],
        ),
        get_memory_experiment("single_memory", codes.SteaneCode(), Pauli.Z, rounds=9),
    ]


def get_memory_experiment(
    kind: str, code: codes.CSSCode, basis: Literal[Pauli.X, Pauli.Z], *, rounds: int
) -> Experiment:
    """Get a quantum memory experiment for a code in a given basis."""
    return Experiment(
        kind,
        rounds=rounds,
        get_circuit=lambda num_rounds, noise_model: circuits.get_memory_experiment(
            code, basis=basis, num_rounds=num_rounds, noise_model=noise_model
        ),
    )


def get_tasks(
    experiments: list[Experiment],
    p_values: tuple[float, ...],
    provenance: Mapping[str, object],
) -> list[sinter.Task]:
    """Get Sinter tasks for every experiment at every physical error rate.

    Args:
        experiments: The experiments to sample.
        p_values: The physical error rates of circuit-level depolarizing noise.
        provenance: Metadata that identifies this run, which is added to every task.

    Returns:
        One task per physical error rate and experiment, which keeps only the first observable of
        the experiment's circuit.
    """
    tasks = []
    for p in p_values:
        noise_model = circuits.DepolarizingNoiseModel(p, include_idling_error=False)
        for experiment in experiments:
            circuit = experiment.get_circuit(experiment.rounds, noise_model)
            metadata = {**provenance, "kind": experiment.kind, "p": p, "rounds": experiment.rounds}
            task = sinter.Task(
                circuit=keep_only_observable(circuit, keep_idx=0), json_metadata=metadata
            )
            tasks.append(task)
    return tasks


def get_sinter_decoder(preset: Preset) -> decoders.SinterDecoder:
    """Get the Sinter decoder of a preset: BP+LSD with options DECODER_OPTIONS[preset]."""
    return decoders.SinterDecoder(decoder=decoders.bp_lsd(**DECODER_OPTIONS[preset]))


####################################################################################################
# outputs


def get_stats_for_run(
    stats: list[sinter.TaskStats],
    run_id: str,
    requested_tasks: set[tuple[object, object, object]],
) -> list[sinter.TaskStats]:
    """Keep the statistics of requested tasks of this run, ignoring other rows of a resume file.

    Args:
        stats: The statistics collected by Sinter, which include all rows of the resume file.
        run_id: The identifier of this run's task definition.
        requested_tasks: The keys (see get_task_key) of the tasks requested by this run.

    Returns:
        The statistics of the requested tasks of this run.
    """
    return [
        stat
        for stat in stats
        if isinstance(stat.json_metadata, dict)
        and stat.json_metadata.get("run_id") == run_id
        and get_task_key(stat.json_metadata) in requested_tasks
    ]


def write_aggregate_csv(
    path: Path,
    preset: Preset,
    stats: list[sinter.TaskStats],
    collection_request: Mapping[str, object],
) -> None:
    """Write collected statistics and their provenance to a CSV file with columns CSV_FIELDS.

    Args:
        path: The path of the CSV file.
        preset: The preset that produced the statistics.
        stats: The statistics to write, with one row per task.
        collection_request: The settings with which statistics were collected.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for stat in sorted(stats, key=lambda stat: get_task_key(stat.json_metadata)):
            metadata = stat.json_metadata
            writer.writerow(
                {
                    "preset": preset,
                    "kind": metadata["kind"],
                    "p": metadata["p"],
                    "rounds": metadata["rounds"],
                    "shots": stat.shots,
                    "errors": stat.errors,
                    "discards": stat.discards,
                    "seconds": stat.seconds,
                    "git_state": metadata["git_state"],
                    "decoder_settings": json.dumps(metadata["decoder"], sort_keys=True),
                    "collection_request": json.dumps(collection_request, sort_keys=True),
                }
            )


####################################################################################################
# helper functions


def get_task_key(metadata: Mapping[str, object]) -> tuple[object, object, object]:
    """Identify a task within a run by its kind, physical error rate, and number of rounds."""
    return metadata.get("kind"), metadata.get("p"), metadata.get("rounds")


def get_git_state() -> str:
    """Identify the state of the source code that produces data.

    Returns:
        The current Git commit if there are no uncommitted changes to tracked files.  Otherwise,
        the commit followed by "-dirty-" and a digest of those changes and of this script, or
        "unknown-" followed by a digest of this script if Git is unavailable.  Untracked files are
        ignored.
    """
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if not status:
            return commit
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]
        return f"unknown-{digest}"
    digest = hashlib.sha256(status.encode() + diff + Path(__file__).read_bytes()).hexdigest()[:12]
    return f"{commit}-dirty-{digest}"


if __name__ == "__main__":
    main()
