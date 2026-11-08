# SPDX-License-Identifier: Apache-2.0

"""A lock has one holder at a time."""

from __future__ import annotations

from pathlib import Path

import pytest

from ..core.atomic import hold_lock


def test_a_lock_excludes_a_second_holder_until_it_is_released(tmp_path: Path) -> None:
    path = tmp_path / "locks" / "run.lock"
    with hold_lock(path), pytest.raises(BlockingIOError), hold_lock(path):
        pass
    with hold_lock(path):
        pass
