# SPDX-License-Identifier: Apache-2.0

"""Chain checkpoints restore exactly and are verified, and records of codes only improve."""

from __future__ import annotations

import json
import os
import tempfile
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
from .search_helpers import canonical_checks, steane_counter, weak_counter

CONFIG = SearchConfig(epochs=1, steps_per_epoch=2)
ROLES = ("current", "lowest_energy", "greatest_distance")


def test_a_saved_chain_loads_as_the_same_state(chain: tuple[Path, SearchState]) -> None:
    path, state = chain
    loaded = checkpoints.load_checkpoint(path)
    for name in ("config", "seed", "completed_epochs", "history"):
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
    weak, steane = _state_holding(weak_counter()), _state_holding(steane_counter())
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
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted([chain.name, record.name])


def test_a_chain_saved_with_the_fields_of_older_versions_loads_as_the_same_state(
    chain: tuple[Path, SearchState],
) -> None:
    # Earlier versions also saved the numbers of proposed and accepted moves, and the objective of
    # the search, which was always the X-only and Z-only operators, with no count of Pauli
    # operators.  Loading accepts them, so that the chains of a campaign that started with them
    # still resume.
    path, _ = chain
    expected = _get_state_data(checkpoints.load_checkpoint(path))

    def add_old_fields(data: dict[str, Any]) -> None:
        data.update(proposed=2, accepted=1)
        data["config"]["objective"] = "homogeneous"
        for role in ROLES:
            data[role]["score"]["count_pauli"] = None

    _edit(path, add_old_fields)
    assert _get_state_data(checkpoints.load_checkpoint(path)) == expected


# Changes to the chain of the fixture, each of which breaks one check of loading.
CORRUPTIONS = {
    "a tampered score": lambda data: data["current"]["score"].update(count_x=-1),
    "a lowest-energy code that is worse than another": lambda data: data.update(
        lowest_energy=data["current"]
    ),
    "a greatest-distance code that is worse than another": lambda data: data.update(
        greatest_distance=data["current"]
    ),
    "another schema version": lambda data: data.update(schema_version=2),
    "another numerical runtime": lambda data: data["runtime"].update(numpy="0"),
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
    better, worse = _state_holding(steane_counter()), _state_holding(weak_counter())
    chain = tmp_path / "chain.json"
    path = checkpoints.get_record_path(chain, kind)
    checkpoints.save_record(worse, chain, kind)
    checkpoints.save_record(better, chain, kind)
    assert canonical_checks(checkpoints.load_code(path)) == canonical_checks(steane_counter())
    before = path.read_bytes()
    checkpoints.save_record(worse, chain, kind)  # a replayed epoch is behind
    checkpoints.save_record(better, chain, kind)
    assert path.read_bytes() == before


def test_each_record_keeps_the_better_code_by_its_own_order(tmp_path: Path) -> None:
    # At p = 1/4, a code of distance 2 has a lower energy than the Steane code of distance 3, so the
    # two records disagree on which of them is better.
    config = SearchConfig(epochs=1, steps_per_epoch=2, physical_error_rate=0.25)
    steane, lower = steane_counter(), steane_counter().deformed_cnot(0, 1)

    kinds: tuple[tuple[checkpoints.RecordKind, CSSLogicalCounter], ...] = (
        ("greatest-distance", steane),
        ("lowest-energy", lower),
    )
    for kind, kept in kinds:
        chain = tmp_path / f"{kind}.json"
        checkpoints.save_record(_state_holding(lower, config), chain, kind)
        checkpoints.save_record(_state_holding(steane, config), chain, kind)
        record = checkpoints.get_record_path(chain, kind)
        assert canonical_checks(checkpoints.load_code(record)) == canonical_checks(kept)


def test_a_record_does_not_overwrite_a_file_that_is_not_a_verified_record_of_its_size(
    tmp_path: Path, chain: tuple[Path, SearchState]
) -> None:
    chains = [tmp_path / f"{name}.json" for name in ("not_a_record", "other_size", "tampered")]
    records = [checkpoints.get_record_path(path, "lowest-energy") for path in chains]
    records[0].write_bytes(chain[0].read_bytes())
    checkpoints.save_record(_state_holding(weak_counter(8)), chains[1], "lowest-energy")
    checkpoints.save_record(_state_holding(steane_counter()), chains[2], "lowest-energy")
    _edit(records[2], lambda data: data["score"].update(count_x=-1))
    for path, record in zip(chains, records):
        before = record.read_bytes()
        with pytest.raises(ValueError):
            checkpoints.save_record(_state_holding(steane_counter()), path, "lowest-energy")
        assert record.read_bytes() == before


@pytest.mark.parametrize(
    "data",
    [
        {"size": 0, "x": [], "z": []},
        {"size": 2, "x": ["0x4"], "z": []},  # beyond the physical width
    ],
)
def test_a_malformed_code_is_rejected(data: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        checkpoints.decode_code(data)


####################################################################################################
# helper methods


def _set_codes(
    state: SearchState, *, current: CSSLogicalCounter, best: CSSLogicalCounter
) -> SearchState:
    """Set a chain's current code, and its lowest-energy and greatest-distance code to ``best``."""
    for role, code in zip(ROLES, (current, best, best)):
        setattr(state, role, ScoredCode(code, score_code(code, state.config)))
    return state


def _state_holding(code: CSSLogicalCounter, config: SearchConfig = CONFIG) -> SearchState:
    """A chain whose current code and optima are all the given code."""
    return _set_codes(start_search(code, config), current=code, best=code)


def _edit(path: Path, edit: Callable[[dict[str, Any]], object]) -> None:
    """Edit the data of a JSON file."""
    data = json.loads(path.read_text())
    edit(data)
    path.write_text(json.dumps(data))


@pytest.fixture(name="chain")
def _chain_fixture(tmp_path: Path) -> tuple[Path, SearchState]:
    """A saved chain of one real epoch, whose current code is weak and whose optima are Steane."""
    state = run_epochs(start_search(steane_counter(), CONFIG))
    _set_codes(state, current=weak_counter(), best=steane_counter())
    path = tmp_path / "chain.json"
    checkpoints.save_checkpoint(state, path)
    return path, state


def _get_state_data(state: SearchState) -> dict[str, Any]:
    """The data of a chain's checkpoint, without the versions of its numerical runtime."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "chain.json"
        checkpoints.save_checkpoint(state, path)
        data = json.loads(path.read_text())
    data.pop("runtime")
    return data
