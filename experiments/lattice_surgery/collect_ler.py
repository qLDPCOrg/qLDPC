#!/usr/bin/env python3
"""Collect publication-scale lattice-surgery logical-error-rate data.

Copyright 2026 The qLDPC Authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Literal

import numpy as np
import sinter
import stim
import sympy

from qldpc import circuits, codes, decoders
from qldpc.circuits.noise_model import DepolarizingNoiseModel
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
DEFAULT_P_VALUES: dict[Preset, tuple[float, ...]] = {
    "bb72": (0.0008, 0.0011, 0.0014, 0.0019, 0.0025),
    "bb18": (0.006, 0.007, 0.008),
    "steane-joint": (0.0002, 0.0004, 0.0008),
}
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
DECODER_SETTINGS: dict[Preset, dict[str, object]] = {
    "bb72": {
        "with_BP_LSD": True,
        "max_iter": 20,
        "bp_method": "ms",
        "lsd_method": "lsd_cs",
        "lsd_order": 5,
    },
    "bb18": {
        "with_BP_LSD": True,
        "max_iter": 100,
        "bp_method": "ms",
        "ms_scaling_factor": 0.0,
        "schedule": "serial",
        "lsd_method": "lsd_e",
        "lsd_order": 5,
    },
    "steane-joint": {
        "with_BP_LSD": True,
        "max_iter": 20,
        "bp_method": "ms",
        "lsd_method": "lsd_cs",
        "lsd_order": 5,
    },
}


def _task(
    circuit: stim.Circuit,
    *,
    kind: str,
    p: float,
    rounds: int,
    provenance: dict[str, object],
) -> sinter.Task:
    return sinter.Task(
        circuit=circuit,
        json_metadata={**provenance, "kind": kind, "p": p, "rounds": rounds},
    )


def _bb72_tasks(
    p_values: tuple[float, ...],
    provenance: dict[str, object],
) -> tuple[list[sinter.Task], decoders.SinterDecoder]:
    x, y = sympy.symbols("x y")
    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    logical = np.asarray(code.get_logical_ops(Pauli.X)[0]).astype(np.uint8)
    gadget = build_gadget(code, logical, basis=Pauli.X)
    tasks: list[sinter.Task] = []
    for p in p_values:
        noise = DepolarizingNoiseModel(p, include_idling_error=False)
        surgery = build_single_ppm_circuit(gadget, rounds=9, noise_model=noise)
        memory = circuits.get_memory_experiment(
            code,
            basis=Pauli.X,
            num_rounds=9,
            noise_model=noise,
        )
        tasks.extend(
            [
                _task(
                    keep_only_observable(surgery, keep_idx=0),
                    kind="surgery",
                    p=p,
                    rounds=9,
                    provenance=provenance,
                ),
                _task(
                    keep_only_observable(memory, keep_idx=0),
                    kind="memory",
                    p=p,
                    rounds=9,
                    provenance=provenance,
                ),
            ]
        )
    decoder = decoders.SinterDecoder(
        with_BP_LSD=True,
        max_iter=20,
        bp_method="ms",
        lsd_method="lsd_cs",
        lsd_order=5,
    )
    return tasks, decoder


def _bb18_tasks(
    p_values: tuple[float, ...],
    provenance: dict[str, object],
) -> tuple[list[sinter.Task], decoders.SinterDecoder]:
    x, y = sympy.symbols("x y")
    source = codes.BBCode(
        (31, 4),
        1 + x**6 * y + x**27,
        y**2 + x**15 * y**3 + x**24,
    )
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
    tasks: list[sinter.Task] = []
    for p in p_values:
        noise = DepolarizingNoiseModel(p, include_idling_error=False)
        surgery = build_single_ppm_circuit(gadget, rounds=15, noise_model=noise)
        memory = circuits.get_memory_experiment(
            code,
            basis=Pauli.X,
            num_rounds=9,
            noise_model=noise,
        )
        tasks.extend(
            [
                _task(
                    keep_only_observable(surgery, keep_idx=0),
                    kind="surgery",
                    p=p,
                    rounds=15,
                    provenance=provenance,
                ),
                _task(
                    keep_only_observable(memory, keep_idx=0),
                    kind="memory",
                    p=p,
                    rounds=9,
                    provenance=provenance,
                ),
            ]
        )
    decoder = decoders.SinterDecoder(
        with_BP_LSD=True,
        max_iter=100,
        bp_method="ms",
        ms_scaling_factor=0.0,
        schedule="serial",
        lsd_method="lsd_e",
        lsd_order=5,
    )
    return tasks, decoder


def _steane_joint_tasks(
    p_values: tuple[float, ...],
    provenance: dict[str, object],
) -> tuple[list[sinter.Task], decoders.SinterDecoder]:
    code_l, code_r = codes.SteaneCode(), codes.SteaneCode()
    logical_l = np.asarray(code_l.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    logical_r = np.asarray(code_r.get_logical_ops(Pauli.Z)[0]).astype(np.uint8)
    gadget_l = build_gadget(code_l, logical_l, basis=Pauli.Z)
    gadget_r = build_gadget(code_r, logical_r, basis=Pauli.Z)
    bridge = build_bridge(gadget_l, gadget_r)
    tasks: list[sinter.Task] = []
    for p in p_values:
        noise = DepolarizingNoiseModel(p, include_idling_error=False)
        surgery = build_joint_ppm_circuit(
            gadget_l,
            gadget_r,
            bridge,
            rounds=9,
            noise_model=noise,
        )[0]
        memory = circuits.get_memory_experiment(
            codes.SteaneCode(),
            basis=Pauli.Z,
            num_rounds=9,
            noise_model=noise,
        )
        tasks.extend(
            [
                _task(
                    keep_only_observable(surgery, keep_idx=0),
                    kind="joint_surgery",
                    p=p,
                    rounds=9,
                    provenance=provenance,
                ),
                _task(
                    keep_only_observable(memory, keep_idx=0),
                    kind="single_memory",
                    p=p,
                    rounds=9,
                    provenance=provenance,
                ),
            ]
        )
    decoder = decoders.SinterDecoder(
        with_BP_LSD=True,
        max_iter=20,
        bp_method="ms",
        lsd_method="lsd_cs",
        lsd_order=5,
    )
    return tasks, decoder


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect publication-scale lattice-surgery logical-error-rate data."
    )
    parser.add_argument("--preset", choices=tuple(DEFAULT_P_VALUES), required=True)
    parser.add_argument("--p-values", type=float, nargs="+")
    parser.add_argument("--max-shots", type=int, default=1_000_000)
    parser.add_argument("--max-errors", type=int, default=100)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git_state() -> str:
    """Commit plus a digest when tracked or untracked source state is dirty."""
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
            ["git", "status", "--porcelain"],
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


def _stats_for_run(stats: list[sinter.TaskStats], run_id: str) -> list[sinter.TaskStats]:
    """Keep only rows produced for this run, tolerating unrelated legacy metadata."""
    return [
        stat
        for stat in stats
        if isinstance(stat.json_metadata, dict) and stat.json_metadata.get("run_id") == run_id
    ]


def main() -> None:
    args = _parse_args()
    preset: Preset = args.preset
    p_values = tuple(args.p_values or DEFAULT_P_VALUES[preset])
    run_settings = {
        "preset": preset,
        "p_values": p_values,
        "max_shots": args.max_shots,
        "max_errors": args.max_errors,
        "workers": args.workers,
        "decoder": DECODER_SETTINGS[preset],
    }
    git_state = _git_state()
    run_id = hashlib.sha256(
        json.dumps(
            {"git_state": git_state, "run_settings": run_settings},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    provenance: dict[str, object] = {
        "run_id": run_id,
        "git_state": git_state,
        "run_settings": run_settings,
    }
    builders = {
        "bb72": _bb72_tasks,
        "bb18": _bb18_tasks,
        "steane-joint": _steane_joint_tasks,
    }
    tasks, decoder = builders[preset](p_values, provenance)
    resume = args.resume or Path(__file__).with_name(f"{preset}_{run_id}_progress.csv")
    stats = sinter.collect(
        tasks=tasks,
        decoders=["custom"],
        custom_decoders={"custom": decoder},
        num_workers=args.workers,
        max_shots=args.max_shots,
        max_errors=args.max_errors,
        print_progress=True,
        save_resume_filepath=resume,
    )
    stats = _stats_for_run(stats, run_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "preset",
        "kind",
        "p",
        "rounds",
        "shots",
        "errors",
        "discards",
        "seconds",
        "git_state",
        "settings",
    ]
    with args.output.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for stat in sorted(
            stats,
            key=lambda item: (item.json_metadata["kind"], item.json_metadata["p"]),
        ):
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
                    "settings": json.dumps(metadata["run_settings"], sort_keys=True),
                }
            )
    print(f"Wrote {len(stats)} aggregate rows to {args.output}")
    print(f"Raw resumable progress remains in {resume}")


if __name__ == "__main__":
    main()
