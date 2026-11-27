# SPDX-License-Identifier: Apache-2.0

"""Check that each citation of a labeled guide section names a label of that guide.

A guide labels a section with an anchor on the line above the section's heading, and code and
documents cite the section by that label, as the writing standard describes.  A guide is labeled
once it has a label, and only citations of labeled guides are checked.
"""

from __future__ import annotations

import functools
import os
import pathlib
import re
import sys

import pytest

_ROOT = pathlib.Path(__file__).parents[2]
_SOURCE = _ROOT / "src" / "qldpc"
_SUFFIXES = {".ipynb", ".md", ".py", ".rst"}
_SKIPPED = {"__pycache__", "autoapi", "build"}

_ANCHOR = re.compile(r'<a name="([^"]*)"></a>')
_LABEL = re.compile(r"sec-[a-z0-9]+(?:-[a-z0-9]+)*")
# a link to a file on GitHub's main branch, with its path from the root of the repository
_URL = re.compile(r"github\.com/qLDPCOrg/qLDPC/blob/main/([\w./-]+\.md)#([\w-]+)")
# a :guide: role, with or without an explicit title, with its path from src/qldpc/
_ROLE = re.compile(r":guide:`(?:[^`<]*<)?([\w./-]+\.md)#([\w-]+)>?`")
# a relative path or a bare file name, as in a Markdown link or a comment
_PATH = re.compile(r"(?<![\w./:-])([\w./-]*\.md)#([\w-]+)")
# a link from a guide to one of its own sections
_LOCAL = re.compile(r"\]\(#([\w-]+)\)")


def test_labels_sit_above_headings() -> None:
    """Each label names one section of its guide, from the line above the section's heading."""
    for path in _files():
        if path.suffix != ".md":
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        labels = []
        for number, line in enumerate(lines):
            for name in _ANCHOR.findall(line):
                where = f"{path.relative_to(_ROOT)}, {name}"
                assert _LABEL.fullmatch(name), f"not a label: {where}"
                assert line == f'<a name="{name}"></a>', f"not alone on its line: {where}"
                following = lines[number + 1] if number + 1 < len(lines) else ""
                assert re.match(r"#+ ", following), f"not above a heading: {where}"
                labels.append(name)
        assert len(labels) == len(set(labels)), f"repeated labels in {path.relative_to(_ROOT)}"


def test_citations_of_labeled_guides_name_their_labels() -> None:
    """Each citation of a section of a labeled guide names a label of that guide."""
    assert _labels(), "no guide has labels"
    assert not _get_broken_citations()


def test_broken_citations_are_found(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Citations of missing labels or guides, and of labeled guides by their headings, are broken.

    A role or URL names its guide by a path from a fixed directory, so a bare file name in one names
    no guide.  A label in a document other than a guide is not checked.
    """
    source = tmp_path / "src" / "qldpc"
    (source / "codes").mkdir(parents=True)
    (source / "codes" / "guide.md").write_text(
        '<a name="sec-kept"></a>\n## Kept\n\nSee [Kept](#sec-kept) and [Gone](#sec-gone).\n'
    )
    (source / "other.md").write_text("# Other\n\n## A heading\n")
    (source / "codes" / "module.py").write_text(
        "# guide.md#sec-kept, guide.md#kept, codes/guide.md#sec-gone, other.md#a-heading\n"
        "# :guide:`Kept <codes/guide.md#sec-gone>`, missing.md#sec-lost, :guide:`<guide.md#sec-kept>`\n"
    )
    (tmp_path / "labels.rst").write_text('Labels look like ``<a name="sec-…"></a>``.\n')
    (tmp_path / "notes.rst").write_text(
        "guide.md#sec-kept and https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/"
        "codes/guide.md#kept, and https://github.com/qLDPCOrg/qLDPC/blob/main/guide.md#sec-kept\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "_SOURCE", source)
    _files.cache_clear()
    _labels.cache_clear()
    try:
        test_labels_sit_above_headings()
        assert _get_broken_citations() == [
            ("notes.rst", "?#sec-kept"),
            ("notes.rst", "src/qldpc/codes/guide.md#kept"),
            ("src/qldpc/codes/guide.md", "src/qldpc/codes/guide.md#sec-gone"),
            ("src/qldpc/codes/module.py", "?#sec-kept"),
            ("src/qldpc/codes/module.py", "?#sec-lost"),
            ("src/qldpc/codes/module.py", "src/qldpc/codes/guide.md#kept"),
            ("src/qldpc/codes/module.py", "src/qldpc/codes/guide.md#sec-gone"),
        ]
    finally:
        _files.cache_clear()
        _labels.cache_clear()


def _get_broken_citations() -> list[tuple[str, str]]:
    """List each citation of a labeled guide that names no label of it, with the citing file."""
    broken = set()
    for path in _files():
        for guide, fragment in _citations(path):
            if guide in _labels() and fragment not in _labels()[guide]:
                broken.add((path.relative_to(_ROOT).as_posix(), f"{guide}#{fragment}"))
            elif guide is None and _LABEL.fullmatch(fragment):
                broken.add((path.relative_to(_ROOT).as_posix(), f"?#{fragment}"))
    return sorted(broken)


def _citations(path: pathlib.Path) -> list[tuple[str | None, str]]:
    """List the guide sections that a file cites, each as the guide's path and a fragment.

    The path is relative to the root of the repository, or None if no guide resolves it.
    """
    text = path.read_text(encoding="utf-8")
    citations: list[tuple[str | None, str]] = []
    for target, fragment in _URL.findall(text):
        citations.append((_get_file(_ROOT / target), fragment))
    for target, fragment in _ROLE.findall(text):
        citations.append((_get_file(_SOURCE / target), fragment))
    # paths in URLs and roles start from fixed directories, so they are not read again as paths
    text = _ROLE.sub("", _URL.sub("", text))
    for target, fragment in _PATH.findall(text):
        citations.append((_resolve(path, target), fragment))
    own = path.relative_to(_ROOT).as_posix()
    if own in _labels():
        citations.extend((own, fragment) for fragment in _LOCAL.findall(text))
    return citations


def _resolve(path: pathlib.Path, target: str) -> str | None:
    """Find the guide that a file means by a path or a bare file name.

    A path is relative to the citing file, to src/qldpc/, to src/, or to the root of the repository.
    A bare file name means the nearest guide of that name in the directories that contain the citing
    file, or else the only labeled guide of that name.
    """
    candidates = [path.parent / target, _SOURCE / target, _SOURCE.parent / target, _ROOT / target]
    if "/" not in target:
        candidates = [
            directory / target
            for directory in path.parents
            if directory == _ROOT or _ROOT in directory.parents
        ]
    for candidate in candidates:
        if (guide := _get_file(candidate)) is not None:
            return guide
    if "/" not in target:
        named = [guide for guide in _labels() if guide.rsplit("/", 1)[-1] == target]
        if len(named) == 1:
            return named[0]
    return None


def _get_file(path: pathlib.Path) -> str | None:
    """The path of a file from the root of the repository, or None if there is no such file."""
    if not path.is_file():
        return None
    return pathlib.Path(os.path.normpath(path)).relative_to(_ROOT).as_posix()


@functools.cache
def _labels() -> dict[str, set[str]]:
    """Map each labeled guide, by its path from the root of the repository, to its labels."""
    labels = {}
    for path in _files():
        if path.suffix == ".md":
            names = set(_ANCHOR.findall(path.read_text(encoding="utf-8")))
            if names:
                labels[path.relative_to(_ROOT).as_posix()] = names
    return labels


@functools.cache
def _files() -> list[pathlib.Path]:
    """List the files of the repository that can cite a guide, apart from this test."""
    files = []
    for directory, subdirectories, names in os.walk(_ROOT):
        subdirectories[:] = sorted(
            name for name in subdirectories if name not in _SKIPPED and not name.startswith(".")
        )
        for name in sorted(names):
            path = pathlib.Path(directory) / name
            if path.suffix in _SUFFIXES and path != pathlib.Path(__file__):
                files.append(path)
    return files
