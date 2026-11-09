# SPDX-License-Identifier: Apache-2.0

"""Versioned atomic checkpoints and immediate records of the notable codes of a chain.

A chain checkpoint (``kind="chain"``) holds everything needed for an exact restart.  A
``lowest-energy`` or ``greatest-distance`` record preserves one code, as soon as it is found, and
cannot restart a chain.  All of them carry the current ``SCHEMA_VERSION`` and store every code
as hexadecimal strings, so codes of more than 64 qubits never pass through imprecise JSON numbers.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import platform
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any, Literal, cast, get_args

import numba
import numpy as np
import scipy

from qldpc.codes import CSSCode, CSSLogicalCounter

from .annealing import (
    LOG_TEMPERATURE_BOUNDS,
    EpochStatistics,
    Score,
    ScoredCode,
    SearchConfig,
    SearchState,
    energy,
    score_code,
    score_rank,
)
from .atomic import write_atomic

# Bump the version when a change would make a resumed chain differ from an uninterrupted one, in
# the proposals, the energy, the temperature fit, or the scores, or would stop old checkpoints from
# loading as the same chain, so that old checkpoints are refused.  Fields that loading ignores, such
# as those that older checkpoints still have, need no bump.
SCHEMA_VERSION = 1

# The kinds of record that a chain preserves beside its checkpoint, each in a file of its own.
RecordKind = Literal["lowest-energy", "greatest-distance"]
RECORD_KINDS: tuple[RecordKind, ...] = get_args(RecordKind)

# The kinds of saved file: a chain checkpoint, or a record.
SavedKind = Literal["chain", RecordKind]

# The ``SearchState`` attributes that hold the codes of a chain checkpoint, which are also their
# keys in its JSON data.
_CHAIN_CODE_ROLES = ("current", "lowest_energy", "greatest_distance")

####################################################################################################
# reading saved files, and encoding codes


def read_saved_file(
    path: str | Path, *, any_schema: bool = False
) -> tuple[SavedKind, dict[str, Any]]:
    """Read a saved chain or record, as its kind (``"chain"`` or a record kind) and its data.

    Args:
        path: The JSON file of a chain checkpoint or of a record.
        any_schema: Accept any schema version, for a reader that needs only a code, whose encoding
            every schema version shares.

    Raises:
        ValueError: If the file is not a JSON object of a known kind and, unless ``any_schema``,
            of the current schema version.
    """
    try:
        data = json.loads(Path(path).read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    kind = data.get("kind") if isinstance(data, dict) else None
    if kind not in ("chain", *RECORD_KINDS):
        raise ValueError(f"{path} is not a chain checkpoint or a record of a code")
    if not any_schema and data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema version {data.get('schema_version')!r}, not {SCHEMA_VERSION}"
        )
    return cast(SavedKind, kind), data


def read_seed(data: dict[str, Any]) -> int:
    """The seed of the data of a saved chain or record, which must be a nonnegative integer."""
    seed = data["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError(f"The seed must be a nonnegative integer, not {seed!r}")
    return seed


def read_config(data: dict[str, Any]) -> SearchConfig:
    """The search policy of the data of a saved chain or record.

    Earlier versions also saved an objective, which the campaign always set to ``homogeneous``, the
    X-only and Z-only operators that the search counts.  Another objective is refused.
    """
    config = dict(data["config"])
    if config.pop("objective", "homogeneous") != "homogeneous":
        raise ValueError("The search no longer counts Pauli operators, as this file does")
    return SearchConfig(**config)


def read_score(data: dict[str, Any]) -> Score:
    """A saved score, which earlier versions saved with an empty count of Pauli operators."""
    score = dict(data)
    if score.pop("count_pauli", None) is not None:
        raise ValueError("The search no longer counts Pauli operators, as this file does")
    return Score(**score)


@contextlib.contextmanager
def report_malformed(path: str | Path) -> Iterator[None]:
    """Report any error in the data of a saved file as a ValueError that names the file."""
    try:
        yield
    except (KeyError, TypeError) as error:
        raise ValueError(f"{path} is malformed: {error!r}") from error
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error


def get_record_path(chain_path: str | Path, kind: RecordKind) -> Path:
    """The file of a kind of record of a chain, such as ``css_n20_k04_s003.lowest-energy.json``."""
    return Path(chain_path).with_suffix(f".{kind}.json")


def encode_code(counter: CSSLogicalCounter) -> dict[str, Any]:
    """Encode stabilizer supports as hexadecimal strings, with one bit per physical qubit."""
    return {
        "size": counter.num_qubits,
        "x": [hex(sum(int(bit) << jj for jj, bit in enumerate(row))) for row in counter.checks_x],
        "z": [hex(sum(int(bit) << jj for jj, bit in enumerate(row))) for row in counter.checks_z],
    }


def decode_code(data: dict[str, Any]) -> CSSLogicalCounter:
    """Validate and decode a code written by ``encode_code``."""
    size = data["size"]
    if not isinstance(size, int) or isinstance(size, bool) or size < 1:
        raise ValueError("Code size must be a positive integer")

    def matrix(values: list[str]) -> np.ndarray:
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise TypeError("Packed checks must be a list of hexadecimal strings")
        words = [int(value, 16) for value in values]
        if any(not 0 <= word < 1 << size for word in words):
            raise ValueError("Packed check exceeds its physical width")
        rows = [[(word >> jj) & 1 for jj in range(size)] for word in words]
        return np.array(rows, dtype=np.uint8).reshape(len(rows), size)

    return CSSLogicalCounter(CSSCode(matrix(data["x"]), matrix(data["z"])))


def load_code(path: str | Path) -> CSSLogicalCounter:
    """Import the lowest-energy code of a chain checkpoint, or the code of a record."""
    kind, data = read_saved_file(path, any_schema=True)
    with report_malformed(path):
        return decode_code((data["lowest_energy"] if kind == "chain" else data)["code"])


def decode_scored_code(data: dict[str, Any], config: SearchConfig, *, label: str) -> ScoredCode:
    """Decode a saved code and independently recompute its score, rejecting any mismatch."""
    counter = decode_code(data["code"])
    config.validate_code(counter)
    score = score_code(counter, config)
    if read_score(data["score"]) != score:
        raise ValueError(f"{label} score does not match its code")
    return ScoredCode(counter, score)


####################################################################################################
# atomic epoch-boundary checkpoints


def save_checkpoint(state: SearchState, path: str | Path) -> None:
    """Atomically save a complete chain at a completed epoch boundary, not just its best codes."""
    data = {
        "schema_version": SCHEMA_VERSION,
        "kind": "chain",
        "runtime": _get_runtime_versions(),
        "config": dataclasses.asdict(state.config),
        "seed": state.seed,
        "completed_epochs": state.completed_epochs,
        "log_temperature": state.log_temperature,
        "energy_jumps": state.energy_jumps.tolist(),
        "rng": state.rng.bit_generator.state,
        **{role: _encode_scored_code(getattr(state, role)) for role in _CHAIN_CODE_ROLES},
        "history": [dataclasses.asdict(epoch) for epoch in state.history],
    }
    write_atomic(Path(path), lambda stream: _dump_json(data, stream))


def save_record(state: SearchState, chain_path: str | Path, kind: RecordKind) -> None:
    """Preserve the lowest-energy or greatest-distance code of a chain immediately, as a record.

    The record is not a restartable chain.  A previous unfinished epoch can have saved a better code
    than the last chain checkpoint, so an existing record is replaced only by a strictly better
    code, and a replayed epoch never overwrites it with a worse one.  The greatest-distance code is
    the one of the lowest energy at its distance.

    Args:
        state: The chain whose code to preserve.
        chain_path: The checkpoint of the chain, beside which the record is saved.
        kind: Which code of the chain to preserve.

    Raises:
        ValueError: If an existing record has another kind, policy, schema version, or code size, is
            malformed, or has a score that does not match its code.
    """
    by_distance = kind == "greatest-distance"
    candidate = state.greatest_distance if by_distance else state.lowest_energy
    destination = get_record_path(chain_path, kind)

    # Keep an existing record unless the new candidate is strictly better.
    if destination.exists():
        previous_kind, previous = read_saved_file(destination)
        with report_malformed(destination):
            if previous_kind != kind or read_config(previous) != state.config:
                raise ValueError("the record has an incompatible kind or policy")
            saved = decode_scored_code(previous, state.config, label="The record")
        if (saved.counter.num_qubits, saved.counter.dimension) != (
            candidate.counter.num_qubits,
            candidate.counter.dimension,
        ):
            raise ValueError(f"{destination} has different n/k parameters")
        keep_previous = (
            score_rank(saved.score) >= score_rank(candidate.score)
            if by_distance
            else energy(saved.score, state.config) <= energy(candidate.score, state.config)
        )
        if keep_previous:
            return

    data = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "config": dataclasses.asdict(state.config),
        "seed": state.seed,
        **_encode_scored_code(candidate),
    }
    write_atomic(destination, lambda stream: _dump_json(data, stream))


def load_checkpoint(path: str | Path) -> SearchState:
    """Restore a chain from its checkpoint, recomputing every score and checking its state.

    Stabilizer generators may be put in a canonical form on import; the codes and decisions of the
    chain are preserved, not the order of its generators.

    Raises:
        ValueError: If the file is a record, was written by another numerical runtime or schema
            version, or is malformed or inconsistent.
    """
    # Reject a file that cannot restart a chain, before spending any time on exact scoring.
    kind, data = read_saved_file(path)
    if kind != "chain":
        raise ValueError(f"{path} is a record of a code, not a chain checkpoint; use --refine")
    if data.get("runtime") != _get_runtime_versions():
        raise ValueError("Checkpoint numerical runtime differs; use --refine for a new chain")
    with report_malformed(path):
        return _restore_chain(data)


def _restore_chain(data: dict[str, Any]) -> SearchState:
    """Rebuild and check the chain of the data of a checkpoint."""
    config = read_config(data)

    # Re-score the three saved codes, then check that the chain state is self-consistent.
    codes = {
        role: decode_scored_code(data[role], config, label=f"Checkpoint {role}")
        for role in _CHAIN_CODE_ROLES
    }
    _check_progress(data, config)
    _check_codes(codes, config)
    history = _decode_history(data["history"])

    rng = np.random.default_rng()
    rng.bit_generator.state = data["rng"]
    return SearchState(
        config=config,
        seed=read_seed(data),
        current=codes["current"],
        lowest_energy=codes["lowest_energy"],
        greatest_distance=codes["greatest_distance"],
        rng=rng,
        log_temperature=data["log_temperature"],
        energy_jumps=np.asarray(data["energy_jumps"], dtype=float),
        completed_epochs=data["completed_epochs"],
        history=history,
    )


def _encode_scored_code(code: ScoredCode) -> dict[str, Any]:
    return {"code": encode_code(code.counter), "score": dataclasses.asdict(code.score)}


def _dump_json(data: dict[str, Any], stream: IO[str]) -> None:
    json.dump(data, stream, indent=2, allow_nan=False)
    stream.write("\n")


def _check_progress(data: dict[str, Any], config: SearchConfig) -> None:
    """Check that epoch and temperature bookkeeping agree with the policy."""
    completed = data["completed_epochs"]
    jumps = np.asarray(data["energy_jumps"], dtype=float)
    log_temperature = data["log_temperature"]
    if (
        not isinstance(completed, int)
        or not 0 <= completed <= config.epochs
        or jumps.shape != (config.steps_per_epoch,)
        or not np.all(np.isfinite(jumps))
        or not math.isfinite(log_temperature)
        or not LOG_TEMPERATURE_BOUNDS[0] <= log_temperature <= LOG_TEMPERATURE_BOUNDS[1]
        or len(data["history"]) != completed
    ):
        raise ValueError("Inconsistent epoch or temperature state")


def _check_codes(codes: dict[str, ScoredCode], config: SearchConfig) -> None:
    """Check that the current, lowest-energy, and greatest-distance codes are consistent."""
    current, lowest, greatest = (codes[role].score for role in _CHAIN_CODE_ROLES)
    first = codes["current"].counter
    if any(code.counter.dimension != first.dimension for code in codes.values()):
        raise ValueError("Checkpoint codes must encode the same number of logical qubits")
    if any(code.counter.num_qubits != first.num_qubits for code in codes.values()):
        raise ValueError("Checkpoint code lengths differ")
    if energy(lowest, config) > min(energy(current, config), energy(greatest, config)):
        raise ValueError(
            "Checkpoint lowest-energy code is worse than its current or greatest-distance code"
        )
    if score_rank(greatest) < max(score_rank(current), score_rank(lowest)):
        raise ValueError("Checkpoint greatest-distance code is inconsistent")


def _decode_history(records: list[dict[str, Any]]) -> list[EpochStatistics]:
    history = [EpochStatistics(**epoch) for epoch in records]
    for epoch, statistics in enumerate(history, start=1):
        if (
            statistics.epoch != epoch
            or not 0 <= statistics.acceptance_rate <= 1
            or not 0 <= statistics.uphill_acceptance_rate <= 1
            or not math.exp(LOG_TEMPERATURE_BOUNDS[0])
            <= statistics.temperature
            <= math.exp(LOG_TEMPERATURE_BOUNDS[1])
            or not math.isfinite(statistics.current_energy)
            or not math.isfinite(statistics.lowest_energy)
        ):
            raise ValueError("Invalid checkpoint epoch history")
    return history


def _get_runtime_versions() -> dict[str, str]:
    """The library and platform versions that a resumed chain must match.

    A restored random state alone cannot reproduce a chain whose samplers or temperature solver
    changed.
    """
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "scipy": scipy.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
    }
