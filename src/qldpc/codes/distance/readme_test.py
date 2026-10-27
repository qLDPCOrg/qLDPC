# SPDX-License-Identifier: Apache-2.0

"""Tests that the README of this package names the modules that the package has."""

import re
from pathlib import Path


def test_the_readme_names_every_module_and_no_other_file() -> None:
    """A module that the README does not name, or a file that it names and is missing, is stale."""
    package = Path(__file__).parent
    files = {path.relative_to(package).as_posix() for path in package.rglob("*.py")}
    modules = {
        name
        for name in files
        if not name.endswith("_test.py") and Path(name).name not in ("conftest.py", "__init__.py")
    }
    # The README names files of this package by their paths in it, and other files by their paths
    # from the root of the repository, which start with src/.
    text = (package / "README.md").read_text(encoding="utf-8")
    named = {
        name
        for name in re.findall(r"`((?:[a-z_]+/)*[a-z_]+\.py)`", text)
        if not name.startswith("src/")
    }
    assert not modules - named, f"The README does not name {sorted(modules - named)}"
    assert not named - files, f"The README names missing files {sorted(named - files)}"
