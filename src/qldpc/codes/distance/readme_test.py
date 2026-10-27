# SPDX-License-Identifier: Apache-2.0

"""Tests that the README of this package names the modules that the package has."""

import re
from pathlib import Path


def test_the_readme_names_every_module_and_no_other_file() -> None:
    """A module that the README does not name, or a file that it names and is missing, is stale."""
    package = Path(__file__).parent
    files = {path.name for path in package.rglob("*.py")}
    modules = {
        path.name
        for path in package.rglob("*.py")
        if not path.name.endswith("_test.py") and path.name not in ("conftest.py", "__init__.py")
    }
    # Names in this package are bare file names, or have the prefix of its one subpackage.
    pattern = r"`((?:_counting/)?[a-z_]+\.py)`"
    named = {Path(name).name for name in re.findall(pattern, (package / "README.md").read_text())}
    assert not modules - named, f"The README does not name {sorted(modules - named)}"
    assert not named - files, f"The README names missing files {sorted(named - files)}"
