# SPDX-License-Identifier: Apache-2.0

"""Tests that the guides of this package name only files that the package has."""

import re
from pathlib import Path


def test_the_guides_name_no_missing_file() -> None:
    """Neither the README nor problems.md names a file of this package that is missing.

    The README's maps of the modules are selective, so nothing checks that they name every module.
    """
    package = Path(__file__).parent
    files = {path.relative_to(package).as_posix() for path in package.rglob("*.py")}
    named = _get_named_files(
        (package / "README.md").read_text(encoding="utf-8")
    ) | _get_named_files((package / "problems.md").read_text(encoding="utf-8"))
    assert not named - files, f"The guides name missing files {sorted(named - files)}"


def _get_named_files(text: str) -> set[str]:
    """The Python files of this package that a text names in code spans.

    The README names files outside this package by their paths from the root of the repository,
    which start with src/.
    """
    names = re.findall(r"`((?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py)`", text)
    return {name for name in names if not name.startswith("src/")}
