# SPDX-License-Identifier: Apache-2.0

"""Check that the repository paths and qLDPC names that the contributor docs cite still exist."""

import os
import pathlib
import pkgutil
import re

import pytest

_ROOT = pathlib.Path(__file__).parents[2]
_DOCS = [*sorted((_ROOT / "docs/source/contributing").rglob("*.rst")), _ROOT / "AGENTS.md"]

# a literal is a path if it ends with a slash or a file extension, as in ``codes/common.py``
_PATH = re.compile(r"[\w./-]*[\w-](/|\.(py|rst|md|toml|yaml|yml|ipynb))")
_NAME = re.compile(r"qldpc(\.\w+)+")
_LINK = re.compile(r"\]\((?![a-z]+:)([^)#\s]+)")


def _cited(pattern: re.Pattern[str]) -> list[tuple[str, str]]:
    """Find the literals in the contributor docs that match a pattern, with the doc citing each."""
    cited: set[tuple[str, str]] = set()
    for doc in _DOCS:
        for literal in re.findall(r"``([^`]+)``|`([^`]+)`", doc.read_text(encoding="utf-8")):
            text = literal[0] or literal[1]
            if pattern.fullmatch(text):
                cited.add((doc.relative_to(_ROOT).as_posix(), text))
    return sorted(cited)


def _repository_paths() -> set[str]:
    """List the files and directories of the repository, relative to its root."""
    skipped = {"__pycache__", "autoapi", "build"}
    paths: set[str] = set()
    for directory, subdirectories, files in os.walk(_ROOT):
        subdirectories[:] = [
            name
            for name in subdirectories
            if name not in skipped and (name == ".github" or not name.startswith("."))
        ]
        relative = pathlib.Path(directory).relative_to(_ROOT).as_posix()
        paths.update(f"{relative}/{name}".removeprefix("./") for name in subdirectories + files)
    return paths


_PATHS = _repository_paths()


@pytest.mark.parametrize(("doc", "path"), _cited(_PATH), ids=lambda value: value)
def test_cited_path_exists(doc: str, path: str) -> None:
    """A cited path names a file or directory of the repository, or the end of one's path."""
    path = path.rstrip("/")
    assert any(known == path or known.endswith(f"/{path}") for known in _PATHS), doc


@pytest.mark.parametrize(("doc", "name"), _cited(_NAME), ids=lambda value: value)
def test_cited_name_exists(doc: str, name: str) -> None:
    """A cited qLDPC module or object can be imported."""
    assert pkgutil.resolve_name(name) is not None, doc


@pytest.mark.parametrize(
    "link", _LINK.findall((_ROOT / "AGENTS.md").read_text(encoding="utf-8")), ids=str
)
def test_agents_md_link_exists(link: str) -> None:
    """Every relative link in AGENTS.md points to a file or directory of the repository."""
    assert (_ROOT / link).exists()
