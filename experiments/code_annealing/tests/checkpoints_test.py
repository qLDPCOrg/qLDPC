# SPDX-License-Identifier: Apache-2.0

"""Chain checkpoints restore exactly and are verified, and records of codes only improve."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qldpc.codes import CSSLogicalCounter

from ..core import checkpoints
from ..core.annealing import (
    ScoredCode,
    SearchConfig,
    SearchState,
    run_epochs,
    score_code,
    start_search,
)
from .search_helpers import canonical_checks, counter_from_checks, steane_counter

CONFIG = SearchConfig(epochs=1, steps_per_epoch=2)
ROLES = ("current", "lowest_energy", "greatest_distance")


def _weak_code(num_qubits: int = 7) -> CSSLogicalCounter:
    """A code of one logical qubit and distance 1, which is worse than the Steane code."""
    return counter_from_checks(np.zeros((0, num_qubits)), np.eye(num_qubits)[1:])


def _set_codes(
    state: SearchState, *, current: CSSLogicalCounter, best: CSSLogicalCounter
) -> SearchState:
    """Set a chain's current code, and its lowest-energy and greatest-distance code to ``best``."""
    for role, code in zip(ROLES, (current, best, best)):
        setattr(state, role, ScoredCode(code, score_code(code, state.config)))
    return state


def _state_holding(code: CSSLogicalCounter) -> SearchState:
    """A chain whose current code and optima are all the given code."""
    return _set_codes(start_search(code, CONFIG), current=code, best=code)


def _edit(path: Path, edit: Callable[[dict[str, Any]], object]) -> None:
    """Edit the data of a JSON file."""
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


@pytest.fixture(name="chain")
def _chain_fixture(tmp_path: Path) -> tuple[Path, SearchState]:
    """A saved chain of one real epoch, whose current code is weak and whose optima are Steane."""
    state = run_epochs(start_search(steane_counter(), CONFIG))
    _set_codes(state, current=_weak_code(), best=steane_counter())
    path = tmp_path / "chain.json"
    checkpoints.save_checkpoint(state, path)
    return path, state


def test_a_saved_chain_loads_as_the_same_state(chain: tuple[Path, SearchState]) -> None:
    path, state = chain
    loaded = checkpoints.load_checkpoint(path)
    for name in ("config", "seed", "completed_epochs", "proposed", "accepted", "history"):
        assert getattr(loaded, name) == getattr(state, name)
    assert loaded.log_temperature == state.log_temperature
    assert loaded.rng.bit_generator.state == state.rng.bit_generator.state
    np.testing.assert_array_equal(loaded.energy_jumps, state.energy_jumps)
    for role in ROLES:
        assert getattr(loaded, role).score == getattr(state, role).score
        assert canonical_checks(getattr(loaded, role).counter) == canonical_checks(
            getattr(state, role).counter
        )


def test_a_failed_save_keeps_the_previous_checkpoint_and_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    weak, steane = _state_holding(_weak_code()), _state_holding(steane_counter())
    chain = tmp_path / "chain.json"
    record = checkpoints.get_record_path(chain, "lowest-energy")
    checkpoints.save_checkpoint(weak, chain)
    checkpoints.save_record(weak, chain, "lowest-energy")
    before = chain.read_bytes(), record.read_bytes()

    def fail(*args: object) -> None:
        raise OSError("disk error")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        checkpoints.save_checkpoint(steane, chain)
    with pytest.raises(OSError):
        checkpoints.save_record(steane, chain, "lowest-energy")  # a better code to replace it
    assert (chain.read_bytes(), record.read_bytes()) == before


# Changes to the chain of the fixture, each of which breaks one check of loading.
CORRUPTIONS = {
    "a tampered score": lambda data: data["current"]["score"].update(count_x=-1),
    "proposals that the epochs did not make": lambda data: data.update(proposed=3),
    "an acceptance rate above one": lambda data: data["history"][0].update(acceptance_rate=1.5),
    "a lowest-energy code that is worse than another": lambda data: data.update(
        lowest_energy=data["current"]
    ),
    "a greatest-distance code that is worse than another": lambda data: data.update(
        greatest_distance=data["current"]
    ),
    "another schema version": lambda data: data.update(schema_version=2),
    "another numerical runtime": lambda data: data["runtime"].update(numpy="0"),
    "a record in place of a chain": lambda data: data.update(kind="lowest-energy"),
    "a missing field": lambda data: data.pop("history"),
}


@pytest.mark.parametrize("corrupt", CORRUPTIONS.values(), ids=CORRUPTIONS)
def test_a_corrupted_chain_is_rejected(
    chain: tuple[Path, SearchState], corrupt: Callable[[dict[str, Any]], object]
) -> None:
    path, _ = chain
    _edit(path, corrupt)
    with pytest.raises(ValueError):
        checkpoints.load_checkpoint(path)


@pytest.mark.parametrize("kind", checkpoints.RECORD_KINDS)
def test_a_record_is_replaced_by_a_better_code_but_not_by_a_worse_or_equal_one(
    tmp_path: Path, kind: checkpoints.RecordKind
) -> None:
    better, worse = _state_holding(steane_counter()), _state_holding(_weak_code())
    chain = tmp_path / "chain.json"
    path = checkpoints.get_record_path(chain, kind)
    assert checkpoints.save_record(worse, chain, kind)
    assert checkpoints.save_record(better, chain, kind)
    assert canonical_checks(checkpoints.load_code(path)) == canonical_checks(steane_counter())
    before = path.read_bytes()
    assert not checkpoints.save_record(worse, chain, kind)  # a replayed epoch is behind
    assert not checkpoints.save_record(better, chain, kind)
    assert path.read_bytes() == before


def test_each_record_keeps_the_better_code_by_its_own_order(tmp_path: Path) -> None:
    # At p = 1/4, a code of distance 2 has a lower energy than the Steane code of distance 3, so the
    # two records disagree on which of them is better.
    config = SearchConfig(epochs=1, steps_per_epoch=2, physical_error_rate=0.25)
    steane, lower = steane_counter(), steane_counter().deformed_cnot(0, 1)

    def holding(code: CSSLogicalCounter) -> SearchState:
        return _set_codes(start_search(code, config), current=code, best=code)

    kinds: tuple[tuple[checkpoints.RecordKind, CSSLogicalCounter], ...] = (
        ("greatest-distance", steane),
        ("lowest-energy", lower),
    )
    for kind, kept in kinds:
        chain = tmp_path / f"{kind}.json"
        checkpoints.save_record(holding(lower), chain, kind)
        checkpoints.save_record(holding(steane), chain, kind)
        record = checkpoints.get_record_path(chain, kind)
        assert canonical_checks(checkpoints.load_code(record)) == canonical_checks(kept)


def test_a_record_does_not_overwrite_a_file_that_is_not_a_verified_record_of_its_size(
    tmp_path: Path, chain: tuple[Path, SearchState]
) -> None:
    chains = [tmp_path / f"{name}.json" for name in ("not_a_record", "other_size", "tampered")]
    records = [checkpoints.get_record_path(path, "lowest-energy") for path in chains]
    records[0].write_bytes(chain[0].read_bytes())
    checkpoints.save_record(_state_holding(_weak_code(8)), chains[1], "lowest-energy")
    checkpoints.save_record(_state_holding(steane_counter()), chains[2], "lowest-energy")
    _edit(records[2], lambda data: data["score"].update(count_x=-1))
    for path, record in zip(chains, records):
        before = record.read_bytes()
        with pytest.raises(ValueError):
            checkpoints.save_record(_state_holding(steane_counter()), path, "lowest-energy")
        assert record.read_bytes() == before


def test_a_code_of_more_than_64_qubits_survives_encoding() -> None:
    checks = np.zeros((1, 70), dtype=int)
    checks[0, [0, 69]] = 1
    counter = counter_from_checks(checks, checks)
    decoded = checkpoints.decode_code(json.loads(json.dumps(checkpoints.encode_code(counter))))
    assert canonical_checks(decoded) == canonical_checks(counter)


@pytest.mark.parametrize(
    "data,error",
    [
        ({"size": 0, "x": [], "z": []}, ValueError),
        ({"size": 2, "x": ["0x4"], "z": []}, ValueError),  # beyond the physical width
        ({"size": 4, "x": "0x3", "z": []}, TypeError),  # not a list of hexadecimal strings
    ],
)
def test_a_malformed_code_is_rejected(data: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        checkpoints.decode_code(data)
