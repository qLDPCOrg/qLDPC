# SPDX-License-Identifier: Apache-2.0

"""The annealing runner: its files, exact restart, starting codes, and rejected command lines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ..core import checkpoints, paths
from ..core.annealing import ScoredCode, SearchConfig, energy, score_code, score_rank, start_search
from ..core.atomic import hold_lock
from ..core.run_annealing import main
from .search_helpers import canonical_checks, steane_counter, weak_counter


@pytest.mark.parametrize("suffix", [".json", ".lowest-energy.json"])
def test_a_new_search_refuses_to_overwrite_the_chain_or_a_record_of_a_run(
    tmp_path: Path, suffix: str
) -> None:
    occupied = paths.get_logs_dir(tmp_path) / f"css_n7_k01_s007{suffix}"
    occupied.parent.mkdir()
    occupied.write_text("owned")
    _fail_with_usage_error([*_get_args(), "--results-dir", str(tmp_path)])
    assert list(occupied.parent.iterdir()) == [occupied]
    assert occupied.read_text() == "owned"


def test_a_resumed_search_ends_as_an_uninterrupted_one(tmp_path: Path, chain: Path) -> None:
    uninterrupted = tmp_path / "uninterrupted.json"
    _run([*_get_args(), "--output", str(uninterrupted)])
    assert checkpoints.load_checkpoint(chain).completed_epochs == 1
    _run(["--resume", str(chain)])
    assert checkpoints.load_checkpoint(chain).completed_epochs == 2
    assert _read(chain) == _read(uninterrupted)


def test_the_records_follow_the_chain_as_it_improves(tmp_path: Path) -> None:
    chain = tmp_path / "chain.json"
    _run([*_get_args(_get_improving_seed(tmp_path)), "--output", str(chain)])
    state = checkpoints.load_checkpoint(chain)
    best = (state.lowest_energy.score, state.greatest_distance.score)
    for kind, score in zip(checkpoints.RECORD_KINDS, best, strict=True):
        recorded = checkpoints.load_code(checkpoints.get_record_path(chain, kind))
        assert score_code(recorded, state.config) == score


def test_replaying_an_interrupted_epoch_never_worsens_a_record_that_is_ahead_of_the_chain(
    tmp_path: Path,
) -> None:
    chain = _save_run(tmp_path)  # a chain at its pilot, with records of the Steane code
    config = checkpoints.load_checkpoint(chain).config
    steane = score_code(steane_counter(), config)
    _run(["--resume", str(chain)])
    lowest, greatest = (
        score_code(checkpoints.load_code(checkpoints.get_record_path(chain, kind)), config)
        for kind in ("lowest-energy", "greatest-distance")
    )
    assert energy(lowest, config) <= energy(steane, config)
    assert score_rank(greatest) >= score_rank(steane)


def test_a_refined_search_starts_a_new_chain_from_the_code_of_a_record(
    tmp_path: Path, record: Path
) -> None:
    output = tmp_path / "refined.json"
    options = ["--epochs", "1", "--steps-per-epoch", "1", "--stop-after-epochs", "0"]
    _run(["--refine", str(record), *options, "--output", str(output)])
    state = checkpoints.load_checkpoint(output)
    assert state.completed_epochs == 0
    assert canonical_checks(state.current.counter) == canonical_checks(steane_counter())


def test_a_chain_that_another_holder_has_locked_is_not_continued(
    tmp_path: Path, chain: Path
) -> None:
    # Without the lock, this resume would continue the chain.
    files = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    with hold_lock(chain.with_suffix(".lock")):
        _fail_with_usage_error(["--resume", str(chain)])
    assert {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == files


def test_a_resumed_chain_cannot_continue_in_another_file(tmp_path: Path, chain: Path) -> None:
    _fail_with_usage_error(["--resume", str(chain), "--output", str(tmp_path / "fork.json")])


def test_a_dry_run_writes_nothing(tmp_path: Path) -> None:
    _run([*_get_args(), "--dry-run", "--results-dir", str(tmp_path)])
    assert not any(tmp_path.iterdir())


####################################################################################################
# helper methods


def _get_args(seed: int = 7) -> list[str]:
    """A search of the Steane code size with two epochs of two proposals."""
    return ["7", "1", "--epochs", "2", "--steps-per-epoch", "2", "--seed", str(seed)]


def _get_improving_seed(directory: Path) -> int:
    """A seed whose search ends with better optima than those of its pilot."""
    for seed in range(40):
        chain = directory / f"probe{seed}.json"
        _run([*_get_args(seed), "--stop-after-epochs", "0", "--output", str(chain)])
        pilot = checkpoints.load_checkpoint(chain)
        _run(["--resume", str(chain)])
        state = checkpoints.load_checkpoint(chain)
        if (
            state.lowest_energy.score != pilot.lowest_energy.score
            and state.greatest_distance.score != pilot.greatest_distance.score
        ):
            return seed
    raise AssertionError("No seed improved on its pilot")


def _save_run(directory: Path) -> Path:
    """Save a chain at its pilot, of a code of distance 1, with records of the Steane code.

    A chain whose records are ahead of its checkpoint is what an interrupted epoch leaves.  The
    codes are set directly, so that they do not depend on the proposals of a random search.
    """
    config = SearchConfig(code_type="css", gate_size=2, epochs=2, steps_per_epoch=1)
    code = weak_counter()
    state = start_search(code, config, seed=0)
    state.current = state.lowest_energy = state.greatest_distance = ScoredCode(
        code, score_code(code, config)
    )
    chain = directory / "css_n7_k01_s000.json"
    checkpoints.save_checkpoint(state, chain)
    steane = steane_counter()
    state.lowest_energy = state.greatest_distance = ScoredCode(steane, score_code(steane, config))
    for kind in checkpoints.RECORD_KINDS:
        checkpoints.save_record(state, chain, kind)
    return chain


def _run(args: list[str]) -> None:
    """Run a search without compiling the Python set-up of its exact counts.

    Compiling changes no result, but would stay on for the rest of the test process.
    """
    main([*args, "--skip-compile"])


def _fail_with_usage_error(args: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main([*args, "--skip-compile"])
    assert error.value.code == 2


def _read(path: Path) -> Any:
    """The data of a chain, re-saved so that it does not depend on the order of generators."""
    resaved = path.with_name("resaved_" + path.name)
    checkpoints.save_checkpoint(checkpoints.load_checkpoint(path), resaved)
    return json.loads(resaved.read_text())


@pytest.fixture(name="chain")
def _chain_fixture(tmp_path: Path) -> Path:
    """A chain of two epochs of two proposals, which stopped after its first epoch."""
    path = tmp_path / "chain.json"
    _run([*_get_args(), "--stop-after-epochs", "1", "--output", str(path)])
    return path


@pytest.fixture(name="record")
def _record_fixture(tmp_path: Path) -> Path:
    """The record of a lowest-energy code, which is the Steane code of distance 3."""
    return checkpoints.get_record_path(_save_run(tmp_path), "lowest-energy")
