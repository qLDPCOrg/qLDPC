# SPDX-License-Identifier: Apache-2.0

"""Tests that the README of this package names the modules that the package has."""

import re
from pathlib import Path


def test_the_readme_names_every_module_and_no_other_file() -> None:
    """The README maps every module, and names no file that is missing.

    "Where to look" names every module, and "Layout" every top-level module or subpackage.  A module
    that they omit, or a file that the README names and is missing, is stale.
    """
    package = Path(__file__).parent
    files = {path.relative_to(package).as_posix() for path in package.rglob("*.py")}
    modules = {
        name
        for name in files
        if not name.endswith("_test.py") and Path(name).name not in ("conftest.py", "__init__.py")
    }
    text = (package / "README.md").read_text(encoding="utf-8")

    in_table = _get_named_files(_get_section(text, "Where to look"))
    assert not modules - in_table, f"'Where to look' has no row for {sorted(modules - in_table)}"
    layout = _get_section(text, "Layout")
    units = {name.partition("/")[0] + "/" if "/" in name else name for name in modules}
    missing = sorted(unit for unit in units if f"`{unit}`" not in layout)
    assert not missing, f"'Layout' does not name {missing}"

    named = _get_named_files(text)
    assert not named - files, f"The README names missing files {sorted(named - files)}"


def _get_section(text: str, title: str) -> str:
    """The text of a section of the README, up to the next section of the same level."""
    return text.split(f"\n## {title}\n", 1)[1].split("\n## ", 1)[0]


def _get_named_files(text: str) -> set[str]:
    """The Python files of this package that a text names in code spans.

    The README names files outside this package by their paths from the root of the repository,
    which start with src/.
    """
    names = re.findall(r"`((?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py)`", text)
    return {name for name in names if not name.startswith("src/")}
