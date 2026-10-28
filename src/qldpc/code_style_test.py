# SPDX-License-Identifier: Apache-2.0

"""Check the module layout rules in AGENTS.md: license headers, section headers, and name order."""

import ast
import pathlib

import pytest

_SOURCE = pathlib.Path(__file__).parent
_MODULES = [
    path
    for path in sorted(_SOURCE.rglob("*.py"))
    if "experimental" not in path.parts and not path.name.endswith(("_test.py", "conftest.py"))
]


@pytest.mark.parametrize("path", _MODULES, ids=lambda path: str(path.relative_to(_SOURCE)))
def test_module_layout(path: pathlib.Path) -> None:
    """Modules have license and section headers, and plain names precede underscore names."""
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# SPDX-License-Identifier: Apache-2.0", "missing license header"
    sections = [0]
    for index, line in enumerate(lines):
        if line.startswith("#" * 20) and set(line) == {"#"}:
            assert line == "#" * 100 and lines[index + 1].startswith("# "), f"line {index + 1}"
            sections.append(index + 1)

    # names that code reads while the module loads must be defined first, whatever their names
    tree = ast.parse("\n".join(lines))
    import_time_names = {node.id for node in _walk_import_time(tree) if isinstance(node, ast.Name)}
    first_private: dict[int, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            section = sum(start < node.lineno for start in sections)
            if not node.name.startswith("_"):
                assert section not in first_private, f"{node.name} follows {first_private[section]}"
            elif node.name not in import_time_names:
                first_private.setdefault(section, node.name)


def _walk_import_time(node: ast.AST) -> list[ast.AST]:
    """Nodes that run when a module is imported, skipping the bodies of functions."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        children = [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]
        return [node, *(child for c in children if c for child in _walk_import_time(c))]
    return [node, *(child for c in ast.iter_child_nodes(node) for child in _walk_import_time(c))]
