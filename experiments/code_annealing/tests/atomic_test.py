# SPDX-License-Identifier: Apache-2.0

"""A file is replaced atomically, also when a network file system repeats a rename."""

from __future__ import annotations

import os
from pathlib import Path
from typing import IO

import pytest

from ..core.atomic import write_atomic, write_text_atomic


def test_a_rename_that_took_effect_but_reports_a_missing_file_still_replaces_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An NFS client that sends a rename again after it succeeded reports the temporary file missing.
    path = tmp_path / "file.txt"
    write_text_atomic(path, "previous\n")
    replace, calls = os.replace, []

    def replace_then_fail_once(source: str | Path, destination: str | Path) -> None:
        replace(source, destination)
        calls.append(source)
        if len(calls) == 1:
            raise FileNotFoundError(source)

    # The contents are produced once, so that a writer that also prints does not print twice.
    writes = []

    def write(stream: IO[str]) -> None:
        writes.append(stream)
        stream.write("next\n")

    monkeypatch.setattr(os, "replace", replace_then_fail_once)
    write_atomic(path, write)
    assert (path.read_text(), len(writes), len(calls)) == ("next\n", 1, 2)
    assert [child.name for child in tmp_path.iterdir()] == ["file.txt"]
