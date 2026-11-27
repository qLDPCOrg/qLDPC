# SPDX-License-Identifier: Apache-2.0

"""Check that each citation of a labeled guide section names a label of that guide.

A guide labels a section with an anchor on the line above the section's heading, and code and
documents cite the section by that label, as the writing standard describes.  A guide is labeled
once it has a label, and only citations of labeled guides are checked, apart from two kinds: a
link by the ``:guide:`` or ``:repo:`` role of the documentation or by GitHub URL must name a file
that exists, and a path whose fragment is a label must name a guide.
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
# a link to a file on GitHub's main branch, with its path from the root of the repository, or to the
# repository itself, whose page shows README.md
_URL = re.compile(r"github\.com/qLDPCOrg/qLDPC(?:/blob/main/([\w./-]+\.md))?(?:#([\w-]+))?")
# a :guide: role, with or without an explicit title, with its path from src/qldpc/
_ROLE = re.compile(r":guide:`(?:[^`<]*<)?([\w./-]+\.md)(?:#([\w-]+))?>?`")
# a :repo: role, with or without an explicit title, with its path from the root of the repository
_REPO_ROLE = re.compile(r":repo:`(?:[^`<]*<)?([\w./-]+\.md)(?:#([\w-]+))?>?`")
# a relative path or a bare file name, as in a Markdown link or a comment
_PATH = re.compile(r"(?<![\w./:-])([\w./-]*\.md)#([\w-]+)")
# a link from a guide to one of its own sections
_LOCAL = re.compile(r"\]\(#([\w-]+)\)")
# a Markdown link to a section of a reStructuredText page, by a relative path or by the page's
# URL on Read the Docs, and a Sphinx label in such a page; pages that Read the Docs builds from
# notebooks or from the code have no such source, and are not checked
_RST_LINK = re.compile(r"\]\(([^)\s#]*\.rst)#([\w-]+)\)")
_RST_URL = re.compile(r"qldpc\.readthedocs\.io/(?:en/[\w.-]+/)?([\w/-]+)\.html#([\w.-]+)")
_SPHINX_LABEL = re.compile(r"^\.\. _([\w-]+):$", re.MULTILINE)


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
    """Each citation of a section of a labeled guide names a label of that guide.

    Each link to a guide by role or URL names a file that exists.
    """
    assert _labels(), "no guide has labels"
    assert not _get_broken_citations()


def test_markdown_links_to_rst_sections_name_their_sphinx_labels() -> None:
    """Each Markdown link to a section of a reStructuredText page names a Sphinx label of it."""
    assert not _get_broken_rst_links()


def test_broken_rst_links_are_found(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Links to a missing page, or to a section of a page by other than its label, are broken.

    Links to pages that Read the Docs builds from notebooks or from the code are not checked.
    """
    pages = tmp_path / "docs" / "source"
    pages.mkdir(parents=True)
    (pages / "page.rst").write_text(".. _page-rounds:\n\nRounds\n------\n")
    (tmp_path / "notes.md").write_text(
        "[ok](docs/source/page.rst#page-rounds), [slug](docs/source/page.rst#rounds), "
        "[gone](docs/source/gone.rst#page-rounds),\n"
        "https://qldpc.readthedocs.io/page.html#page-rounds and "
        "https://qldpc.readthedocs.io/en/latest/page.html#rounds, and the pages built from a notebook "
        "and from the code, https://qldpc.readthedocs.io/examples/basics.html#Setup and "
        "https://qldpc.readthedocs.io/autoapi/qldpc/codes/index.html#qldpc.codes.CSSCode\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", tmp_path)
    _files.cache_clear()
    try:
        assert _get_broken_rst_links() == [
            ("notes.md", "docs/source/gone.rst#page-rounds"),
            ("notes.md", "docs/source/page.rst#rounds"),
            ("notes.md", "docs/source/page.rst#rounds"),
        ]
    finally:
        _files.cache_clear()


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
        "# :guide:`Moved <codes/moved.md>`, :guide:`Moved <codes/moved.md#a-heading>`\n"
        "# :repo:`src/qldpc/codes/guide.md#sec-kept`, :repo:`codes/guide.md#sec-kept`\n"
    )
    (tmp_path / "labels.rst").write_text('Labels look like ``<a name="sec-…"></a>``.\n')
    (tmp_path / "README.md").write_text('<a name="sec-goals"></a>\n## 🎯 Goals\n')
    (tmp_path / "notes.rst").write_text(
        "guide.md#sec-kept and https://github.com/qLDPCOrg/qLDPC/blob/main/src/qldpc/"
        "codes/guide.md#kept, and https://github.com/qLDPCOrg/qLDPC/blob/main/guide.md#sec-kept\n"
        "https://github.com/qLDPCOrg/qLDPC#-goals and https://github.com/qLDPCOrg/qLDPC#sec-goals\n"
    )
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", tmp_path)
    monkeypatch.setattr(sys.modules[__name__], "_SOURCE", source)
    _files.cache_clear()
    _labels.cache_clear()
    try:
        test_labels_sit_above_headings()
        assert _get_broken_citations() == [
            ("notes.rst", "README.md#-goals"),
            ("notes.rst", "github.com/qLDPCOrg/qLDPC/blob/main/guide.md#sec-kept"),
            ("notes.rst", "src/qldpc/codes/guide.md#kept"),
            ("src/qldpc/codes/guide.md", "src/qldpc/codes/guide.md#sec-gone"),
            ("src/qldpc/codes/module.py", ":guide:`<guide.md#sec-kept>`"),
            ("src/qldpc/codes/module.py", ":guide:`Moved <codes/moved.md#a-heading>`"),
            ("src/qldpc/codes/module.py", ":guide:`Moved <codes/moved.md>`"),
            ("src/qldpc/codes/module.py", ":repo:`codes/guide.md#sec-kept`"),
            ("src/qldpc/codes/module.py", "missing.md#sec-lost"),
            ("src/qldpc/codes/module.py", "src/qldpc/codes/guide.md#kept"),
            ("src/qldpc/codes/module.py", "src/qldpc/codes/guide.md#sec-gone"),
        ]
    finally:
        _files.cache_clear()
        _labels.cache_clear()


def _get_broken_citations() -> list[tuple[str, str]]:
    """List the broken citations of guide sections, each with the citing file.

    A citation is broken if it names a labeled guide but no label of it, if it is a role or URL
    whose path names no file, or if it names a label but no guide.  A broken citation that names a
    guide is given by the guide's path from the root of the repository and the fragment, and any
    other as written.
    """
    broken = set()
    for path in _files():
        for guide, fragment, cited in _citations(path):
            if guide is None or (guide in _labels() and fragment not in _labels()[guide]):
                broken.add(
                    (
                        path.relative_to(_ROOT).as_posix(),
                        cited if guide is None else f"{guide}#{fragment}",
                    )
                )
    return sorted(broken)


def _get_broken_rst_links() -> list[tuple[str, str]]:
    """List each Markdown link to a section of a reStructuredText page that names no label of it."""
    broken = []
    for path in _files():
        if path.suffix not in (".md", ".ipynb"):
            continue
        text = path.read_text(encoding="utf-8")
        targets = [(path.parent / page, fragment) for page, fragment in _RST_LINK.findall(text)]
        pages = _ROOT / "docs" / "source"
        targets += [
            (pages / f"{page}.rst", fragment)
            for page, fragment in _RST_URL.findall(text)
            if (pages / f"{page}.rst").is_file()
        ]
        for page, fragment in targets:
            page = pathlib.Path(os.path.normpath(page))
            labels = (
                _SPHINX_LABEL.findall(page.read_text(encoding="utf-8")) if page.is_file() else []
            )
            if fragment not in labels:
                where = page.relative_to(_ROOT).as_posix() if _ROOT in page.parents else str(page)
                broken.append((path.relative_to(_ROOT).as_posix(), f"{where}#{fragment}"))
    return sorted(broken)


def _citations(path: pathlib.Path) -> list[tuple[str | None, str, str]]:
    """List the guide sections that a file cites, each as a guide, a fragment, and the text cited.

    The guide is a path from the root of the repository, or None if the citation names no file: a
    role or URL whose path names no file, or a path that names no file but has a label for its
    fragment.  Other paths that name no file, and roles and URLs without a fragment, are not
    citations of sections and are left out.
    """
    text = path.read_text(encoding="utf-8")
    citations: list[tuple[str | None, str, str]] = []
    for directory, pattern in ((_ROOT, _URL), (_SOURCE, _ROLE), (_ROOT, _REPO_ROLE)):
        for match in pattern.finditer(text):
            target, fragment = match.groups()
            guide = _get_file(directory / (target or "README.md"))
            if guide is None or fragment is not None:
                citations.append((guide, fragment or "", match.group()))
    # paths in URLs and roles start from fixed directories, so they are not read again as paths
    text = _REPO_ROLE.sub("", _ROLE.sub("", _URL.sub("", text)))
    for match in _PATH.finditer(text):
        target, fragment = match.groups()
        guide = _resolve(path, target)
        if guide is not None or _LABEL.fullmatch(fragment):
            citations.append((guide, fragment, match.group()))
    own = path.relative_to(_ROOT).as_posix()
    if own in _labels():
        citations.extend((own, fragment, f"#{fragment}") for fragment in _LOCAL.findall(text))
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
