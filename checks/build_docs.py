#!/usr/bin/env python3
import html
import os
import pathlib
import re
import shutil
import sys
import sysconfig

import checks_superstaq as checks

PARAGRAPH_PATTERN = re.compile(r"<p>(.*?)</p>", re.DOTALL)
HEADING_PATTERN = re.compile(r"<h([2-6])[^>]*>(.*?)</h\1>", re.DOTALL)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
EDITOR_WARNING_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:!!![^!\n]*!!!|(?:warnings?|notes?|cautions?|important|tip|todo):\s+\S)",
    re.IGNORECASE,
)
COLLAPSED_LIST_PATTERN = re.compile(r"(?:^|\n)\s*(?:[-+*]|[0-9]+[.)])\s+\S")
MALFORMED_CONTINUATION_PATTERN = re.compile(r"<li>\s*<dl", re.DOTALL)


def ensure_sphinx_on_path() -> None:
    """Prefer the sphinx-build installed alongside the active Python interpreter."""
    scripts_dir = sysconfig.get_path("scripts")
    if shutil.which("sphinx-build", path=scripts_dir):
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")


def ensure_pandoc_on_path() -> None:
    """Make a bundled pandoc discoverable, if the system provides none.

    nbsphinx renders the example notebooks through pandoc, which it locates by searching PATH.  The
    pypandoc-binary distribution ships a pandoc executable inside its own package directory but
    installs no console script, so that copy is invisible to a PATH search until it is added here.
    A system pandoc, if present, takes precedence and this is a no-op.
    """
    if shutil.which("pandoc"):
        return
    try:
        import pypandoc
    except ImportError:
        return
    try:
        pandoc = pypandoc.get_pandoc_path()
    except OSError:
        return
    os.environ["PATH"] = os.path.dirname(pandoc) + os.pathsep + os.environ.get("PATH", "")


def validate_rendered_docstrings() -> int:
    """Reject editor-oriented docstring structures that Sphinx silently renders as plain prose."""
    api_docs = pathlib.Path(__file__).parents[1] / "docs" / "build" / "html" / "autoapi"
    problems: list[str] = []
    pages = sorted(api_docs.rglob("*.html"))
    if not pages:
        print(f"Rendered API documentation not found under {api_docs}", file=sys.stderr)
        return 1

    # inspect rendered paragraphs for structures that should have become admonitions or lists
    for path in pages:
        rendered_page = path.read_text(encoding="utf-8")
        for paragraph in PARAGRAPH_PATTERN.finditer(rendered_page):
            text = html.unescape(HTML_TAG_PATTERN.sub("", paragraph.group(1)))
            line_number = rendered_page.count("\n", 0, paragraph.start()) + 1
            location = f"{path.relative_to(api_docs)}:{line_number}"
            excerpt = " ".join(text.split())[:80]
            if EDITOR_WARNING_PATTERN.search(text):
                problems.append(f"{location}: editor-style warning or note: {excerpt}")
            if COLLAPSED_LIST_PATTERN.search(text):
                problems.append(f"{location}: list rendered as plain paragraph text: {excerpt}")

        for continuation in MALFORMED_CONTINUATION_PATTERN.finditer(rendered_page):
            line_number = rendered_page.count("\n", 0, continuation.start()) + 1
            location = f"{path.relative_to(api_docs)}:{line_number}"
            problems.append(f"{location}: list continuation rendered as a definition list")

        for heading in HEADING_PATTERN.finditer(rendered_page):
            text = (
                html.unescape(HTML_TAG_PATTERN.sub("", heading.group(2))).replace("", "").strip()
            )
            if text.endswith(":"):
                line_number = rendered_page.count("\n", 0, heading.start()) + 1
                location = f"{path.relative_to(api_docs)}:{line_number}"
                problems.append(f"{location}: editor-style heading: {text}")

    if not problems:
        return 0

    print("Rendered API documentation contains malformed docstrings:", file=sys.stderr)
    for problem in problems:
        print(f"- {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    ensure_sphinx_on_path()
    ensure_pandoc_on_path()
    build_result = checks.build_docs.run(*sys.argv[1:])
    sys.exit(build_result or validate_rendered_docstrings())
