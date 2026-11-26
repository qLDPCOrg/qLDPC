# SPDX-License-Identifier: Apache-2.0

"""Tests that the README and the guides of this package match the files and sections they cite."""

import bisect
import re
from pathlib import Path

# The guides whose numbered sections the package cites: the two files of the Brouwer-Zimmermann
# guide number their sections as one guide.
_GUIDES = {
    "brouwer_zimmermann": ("brouwer_zimmermann.md", "brouwer_zimmermann_code.md"),
    "connected_cluster": ("connected_cluster.md",),
}


def test_the_readme_names_every_module_and_no_other_file() -> None:
    """The README maps every module, and neither it nor problems.md names a missing file.

    "Where to look" names every module, and "Layout" every top-level module or subpackage.  A module
    that they omit, or a file that the README or problems.md names and is missing, is stale.
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
    assert not modules - in_table, f"'Where to look' does not name {sorted(modules - in_table)}"
    layout = _get_section(text, "Layout")
    units = {name.partition("/")[0] + "/" if "/" in name else name for name in modules}
    missing = sorted(unit for unit in units if f"`{unit}`" not in layout)
    assert not missing, f"'Layout' does not name {missing}"

    named = _get_named_files(text) | _get_named_files(
        (package / "problems.md").read_text(encoding="utf-8")
    )
    assert not named - files, f"The guides name missing files {sorted(named - files)}"


def test_cited_sections_and_anchors_exist() -> None:
    """Each section that the package cites by number, and each anchor of a guide, exists.

    A citation such as "Section 3.9" or "Sections 3.6 and 3.11" refers to the guide that follows it,
    as in "of ``connected_cluster.md``"; otherwise to the guides that a module says it cites, the
    connected-cluster guide in ``_connected_cluster/``, or else the Brouwer-Zimmermann guide.
    A citation may break across lines of a docstring, comment, or paragraph.  Citations of sections
    of other works, such as papers in the references of a guide, are skipped.  Links into the guides
    from Markdown files of the repository, outside hidden directories, must name existing headings.
    """
    package = Path(__file__).parent
    headings = {
        guide: {
            number
            for name in names
            for number in re.findall(
                r"^#{2,4} (\d+(?:\.\d+)*)\.? ", _read(package / name), re.MULTILINE
            )
        }
        for guide, names in _GUIDES.items()
    }
    # This file quotes citations as examples, so it is skipped.
    files = [*package.rglob("*.py"), *package.rglob("*.md")]
    missing = [
        f"{path.relative_to(package)}:{line} cites Section {number}"
        for path in sorted(path for path in files if path.name != Path(__file__).name)
        for number, guides, line in _get_citations(path, package)
        if not any(number in headings[guide] for guide in guides)
    ]
    assert not missing, "\n".join(missing)

    # Links from Markdown files of the repository, outside hidden directories, into the guides of
    # this package, or within them.
    root = Path(__file__).parents[4]
    links = [
        (path, match, (path.parent / (match.group(1) or path.name)).resolve())
        for path in sorted(root.rglob("*.md"))
        if not any(part.startswith(".") for part in path.relative_to(root).parts)
        for match in re.finditer(r"\]\(([^)\s#:]*\.md)?#([^)\s]+)\)", _read(path))
    ]
    broken = [
        f"{path.relative_to(root)} links to {match[0]}"
        for path, match, target in links
        if target.is_relative_to(package.resolve())
        and not (target.exists() and match.group(2) in _get_anchors(target))
    ]
    assert not broken, "\n".join(broken)


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


def _read(path: Path) -> str:
    """The text of a file."""
    return path.read_text(encoding="utf-8")


def _get_citations(path: Path, package: Path) -> list[tuple[str, set[str], int]]:
    """The section numbers that a file of this package cites, each with its guides and line.

    The references of a guide, which cite sections of other works, are skipped.
    """
    text = _read(path).split("\n## References\n", 1)[0]
    defaults = _get_default_guides(path.relative_to(package).as_posix(), text)
    # Join the lines, without their indentation and comment marks, so that a citation that breaks
    # across lines matches, and keep where each line starts in the joined text.
    lines = [re.sub(r"\A\s*(?:#:?\s+)?", "", line) for line in text.split("\n")]
    starts = [0]
    for joined in lines[:-1]:
        starts.append(starts[-1] + len(joined) + 1)
    text = " ".join(lines)
    citations = []
    for match in re.finditer(
        r"Sections? (\d+(?:\.\d+)?)((?:(?:,| and| or| to|, and) \d+(?:\.\d+)?)*)", text
    ):
        after = text[match.end() : match.end() + 80]
        before = text[max(0, match.start() - 20) : match.start()]
        # Skip sections of other works, as in "Section 4.3.2", "its Section 5.1", and "Section 6 of
        # Tillmann's preprint".
        if re.match(r"\A(?:\.\d|[\s,]*of (?:\[?arXiv|the guide of|its |their |[A-Z]))", after):
            continue
        if re.search(r"\b(?:its|their|[A-Z]\w*'s)\s*\Z", before):
            continue
        # A citation followed by the name of a guide, as in "of ``connected_cluster.md``", refers to
        # that guide.
        named = re.match(r"\A[\s,]*(?:of|in) \[?`{0,2}(?:[\w./]*/)?(\w+\.md)", after)
        guide = _get_guide(named.group(1)) if named else None
        guides = {guide} if guide else defaults
        line = bisect.bisect_right(starts, match.start())
        for number in [match.group(1), *re.findall(r"\d+(?:\.\d+)?", match.group(2))]:
            citations.append((number, guides, line))
    return citations


def _get_default_guides(name: str, text: str) -> set[str]:
    """The guides whose sections a file of this package cites when it names no guide."""
    if name.startswith("_connected_cluster/") or name == "connected_cluster.md":
        return {"connected_cluster"}
    declared = re.search(r"Sections cited in this module are those of(.{0,200})", text, re.DOTALL)
    if declared:
        names = re.findall(r"\w+\.md", declared.group(1))
        return {guide for name in names if (guide := _get_guide(name))}
    return {"brouwer_zimmermann"}


def _get_guide(name: str) -> str | None:
    """The guide of which a Markdown file of this package is part, if any."""
    return next((guide for guide, names in _GUIDES.items() if name in names), None)


def _get_anchors(path: Path) -> set[str]:
    """The anchors of the headings of a Markdown file, as GitHub makes them."""
    titles = re.findall(r"^#{1,6} (.+?)\s*$", _read(path), re.MULTILINE)
    return {
        re.sub(r"[^\w\- ]", "", title.replace("`", "").lower()).replace(" ", "-")
        for title in titles
    }
