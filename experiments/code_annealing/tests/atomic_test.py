# SPDX-License-Identifier: Apache-2.0

"""A file is replaced whole or not at all, keeps its permissions, and has one lock holder."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ..core.atomic import hold_lock, write_text_atomic


def test_a_file_is_replaced_by_complete_text_and_its_directory_is_created(tmp_path: Path) -> None:
    path = tmp_path / "new directory" / "file.txt"
    write_text_atomic(path, "first\n")
    path.chmod(0o640)
    write_text_atomic(path, "second\n")
    assert path.read_text() == "second\n"
    assert path.stat().st_mode & 0o777 == 0o640
    assert [child.name for child in path.parent.iterdir()] == ["file.txt"]


def test_a_failed_replacement_keeps_the_previous_file_and_leaves_no_debris(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "file.txt"
    write_text_atomic(path, "previous\n")

    def fail(*args: object) -> None:
        raise OSError("disk error")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        write_text_atomic(path, "next\n")
    assert path.read_text() == "previous\n"
    assert [child.name for child in tmp_path.iterdir()] == ["file.txt"]


def test_a_lock_excludes_a_second_holder_until_it_is_released(tmp_path: Path) -> None:
    path = tmp_path / "locks" / "run.lock"
    with hold_lock(path), pytest.raises(BlockingIOError), hold_lock(path):
        pass
    with hold_lock(path):
        pass
