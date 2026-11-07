# SPDX-License-Identifier: Apache-2.0

"""Check that the repository paths and qLDPC names that the contributor docs cite still exist."""

import os
import pathlib
import pkgutil
import re

_ROOT = pathlib.Path(__file__).parents[2]
_DOCS = [*sorted((_ROOT / "docs/source/contributing").rglob("*.rst")), _ROOT / "AGENTS.md"]

# a path ends with a slash or with one of these file extensions, as in ``codes/common.py``
_PATH = re.compile(r"[\w./-]*[\w-](/|\.(py|rst|md|toml|yaml|yml|ipynb))")
# a dotted qLDPC name, but not part of a path or URL such as https://qldpc.readthedocs.io
_NAME = re.compile(r"(?<![\w/.:])qldpc(\.\w+)+")
# the target of a relative Markdown link, without its anchor
_LINK = re.compile(r"\]\((?![a-z]+:)([^)#\s]+)")


def test_cited_paths_exist() -> None:
    """Each cited path names a file or directory of the repository, or ends the path of one."""
    cited = _cited_paths()
    assert cited
    known = _repository_paths()
    missing = {
        (doc, path)
        for doc, path in cited
        if not any(known_path == path or known_path.endswith(f"/{path}") for known_path in known)
    }
    assert not missing, sorted(missing)


def test_cited_names_exist() -> None:
    """Each cited qLDPC module or object can be imported."""
    assert not _resolves("qldpc.no_such_name")
    cited = _cited_names()
    assert cited
    missing = {(doc, name) for doc, name in cited if not _resolves(name)}
    assert not missing, sorted(missing)


def test_agents_md_links_exist() -> None:
    """Each relative link in AGENTS.md points to a file or directory of the repository."""
    links = set(_LINK.findall((_ROOT / "AGENTS.md").read_text(encoding="utf-8")))
    missing = [link for link in sorted(links) if not (_ROOT / link).exists()]
    assert not missing, missing


def _literals() -> list[tuple[str, str]]:
    """List the inline literals of the contributor docs, with the doc that contains each."""
    literals: list[tuple[str, str]] = []
    for doc in _DOCS:
        doc_path = doc.relative_to(_ROOT).as_posix()
        for double, single in re.findall(r"``([^`]+)``|`([^`]+)`", doc.read_text(encoding="utf-8")):
            literals.append((doc_path, double or single))
    return literals


def _cited_paths() -> set[tuple[str, str]]:
    """Find the paths in the literals, including in commands, and strip trailing slashes."""
    return {
        (doc, token.rstrip("/"))
        for doc, literal in _literals()
        for token in re.split(r"\s+|::", literal)
        if _PATH.fullmatch(token)
    }


def _cited_names() -> set[tuple[str, str]]:
    """Find the dotted qLDPC names in the literals, including role targets and calls."""
    return {
        (doc, match.group()) for doc, literal in _literals() for match in _NAME.finditer(literal)
    }


def _resolves(name: str) -> bool:
    """Report whether a dotted qLDPC name can be imported."""
    try:
        pkgutil.resolve_name(name)
    except (ImportError, AttributeError):
        return False
    return True


def _repository_paths() -> set[str]:
    """List the files and directories of the repository, relative to its root."""
    # skip caches, build outputs, and hidden directories other than .github, such as .git and .venv
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
