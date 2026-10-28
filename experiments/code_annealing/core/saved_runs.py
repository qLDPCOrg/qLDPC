# SPDX-License-Identifier: Apache-2.0

"""Saved runs: finding their files, and checking the codes in them against exact scores.

A run saves a chain checkpoint (``kind="chain"``), which holds its lowest-energy and
greatest-distance codes, and one record for each of those codes as soon as it improves
(``lowest-energy`` and ``greatest-distance``), which survives an interrupted epoch.  Loading
recomputes the exact score of every code, so a saved score is never trusted.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal

from qldpc.codes import CSSLogicalCounter

from .annealing import Score, SearchConfig, energy
from .atomic import write_atomic
from .checkpoints import (
    RECORD_KINDS,
    RecordKind,
    decode_code,
    decode_scored_code,
    get_record_path,
    read_saved_file,
    read_seed,
    report_malformed,
)
from .proposals import CodeType

# The role of a saved code: a record, or one of the two optima that a chain checkpoint holds.
SavedRole = RecordKind | Literal["chain-lowest-energy", "chain-greatest-distance"]

CSV_FIELDS = (
    "path",
    "role",
    "code_type",
    "seed",
    "n",
    "k",
    "d",
    "count_x",
    "count_z",
    "count_pauli",
    "objective",
    "energy",
    "method",
    "epochs",
    "steps_per_epoch",
    "completed_epochs",
)


@dataclass(frozen=True)
class SavedCode:
    """A code from a saved run, with its exactly recomputed score."""

    path: Path
    role: SavedRole
    """A record kind, or ``chain-`` and a kind for a code that a chain checkpoint holds."""

    config: SearchConfig
    seed: int
    counter: CSSLogicalCounter
    score: Score
    completed_epochs: int | None
    """Complete epochs of the chain of a code from a checkpoint, or None for a record, which
    does not store them."""

    @property
    def code_type(self) -> CodeType:
        return self.config.code_type

    @property
    def num_qubits(self) -> int:
        return self.counter.num_qubits

    @property
    def num_logicals(self) -> int:
        return self.counter.dimension

    @property
    def count_homogeneous(self) -> int:
        """The number of minimum-weight X-only plus Z-only logical operators, for any objective."""
        return self.score.count_homogeneous

    @property
    def energy(self) -> float:
        return energy(self.score, self.config)


@dataclass(frozen=True)
class RunProgress:
    """How far a chain has run, read from its checkpoint without rechecking any code."""

    path: Path
    code_type: CodeType
    num_qubits: int
    num_logicals: int
    seed: int
    completed_epochs: int
    epochs: int
    greatest_distance: int
    """Greatest distance saved so far, which this module does not recheck."""

    greatest_distance_count_homogeneous: int
    """Its number of minimum-weight X-only plus Z-only logical operators."""

    @property
    def finished(self) -> bool:
        return self.completed_epochs >= self.epochs


####################################################################################################
# finding and loading saved runs


def get_saved_run_paths(logs_dir: Path, explicit: Sequence[Path] = ()) -> list[Path]:
    """Explicit checkpoints with their adjacent records, else every JSON file in the logs."""
    if not explicit:
        return sorted(logs_dir.glob("*.json"))
    paths = []
    for path in explicit:
        paths.append(path)
        paths.extend(
            record for kind in RECORD_KINDS if (record := get_record_path(path, kind)).exists()
        )
    return paths


def load_saved_codes(paths: Iterable[Path]) -> list[SavedCode]:
    """Validate each saved optimum against its actual code, keeping its exact count columns.

    Only chain and immediate records of the current schema are accepted.  Resource, malformed-data
    and exact-score failures propagate instead of silently omitting codes.  The runtime of the run
    need not match the current one: loading verifies scores, not seeded restart equivalence.
    """
    saved = []
    for path in sorted(set(paths)):
        kind, data = read_saved_file(path)
        with report_malformed(path):
            config = SearchConfig(**data["config"])
            scored_codes: list[tuple[SavedRole, dict[str, Any]]]
            if kind == "chain":
                scored_codes = [
                    ("chain-lowest-energy", data["lowest_energy"]),
                    ("chain-greatest-distance", data["greatest_distance"]),
                ]
            else:
                scored_codes = [(kind, data)]
            for role, scored_code in scored_codes:
                code = decode_scored_code(scored_code, config, label=f"Saved {role}")
                completed_epochs = data.get("completed_epochs")
                saved.append(
                    SavedCode(
                        path,
                        role,
                        config,
                        read_seed(data),
                        code.counter,
                        code.score,
                        completed_epochs,
                    )
                )
    return saved


def load_run_progress(path: Path) -> RunProgress | None:
    """The progress of the chain in a checkpoint file, or None if the file holds a record."""
    kind, data = read_saved_file(path)
    if kind != "chain":
        return None
    with report_malformed(path):
        config = SearchConfig(**data["config"])
        counter = decode_code(data["current"]["code"])
        greatest = Score(**data["greatest_distance"]["score"])
        return RunProgress(
            path,
            config.code_type,
            counter.num_qubits,
            counter.dimension,
            read_seed(data),
            int(data["completed_epochs"]),
            config.epochs,
            greatest.distance,
            greatest.count_homogeneous,
        )


def load_progress(logs_dir: Path) -> list[RunProgress]:
    """The progress of every chain in the logs, from the checkpoints alone."""
    runs = (load_run_progress(path) for path in sorted(logs_dir.glob("*.json")))
    return [run for run in runs if run is not None]


####################################################################################################
# CSV of every saved code


def write_csv(saved: Sequence[SavedCode], destination: Path) -> None:
    """Replace the CSV of saved codes only after the complete file has been flushed."""

    def write(stream: IO[str]) -> None:
        writer = csv.writer(stream)
        writer.writerow(CSV_FIELDS)
        for code in saved:
            writer.writerow(
                (
                    str(code.path),
                    code.role,
                    code.code_type,
                    code.seed,
                    code.num_qubits,
                    code.num_logicals,
                    code.score.distance,
                    code.score.count_x,
                    code.score.count_z,
                    code.score.count_pauli,
                    code.config.objective,
                    code.energy,
                    code.config.method,
                    code.config.epochs,
                    code.config.steps_per_epoch,
                    code.completed_epochs,
                )
            )

    write_atomic(destination, write)
