# SPDX-License-Identifier: Apache-2.0

"""Check that the modules and subpackages of qldpc.circuits import each other in layers."""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Collection

import pytest

# The top-level modules and subpackages of qldpc.circuits, in layers from shared foundations
# (first) to complete workflows (last).  Each imports only from itself and from lower layers, which
# rules out import cycles between them.  Add each new top-level module or subpackage to a layer.
_LAYERS = (
    ("common", "bookkeeping"),
    ("transforms", "_post_selection"),
    ("encoding", "noise_model"),
    ("benchmarking", "certification", "memory", "transversal"),
)

_PACKAGE = "qldpc.circuits"
_ROOT = pathlib.Path(__file__).parent
_LAYER_INDEX = {unit: index for index, layer in enumerate(_LAYERS) for unit in layer}


def _is_source(path: pathlib.Path) -> bool:
    return path.name != "conftest.py" and not path.name.endswith("_test.py")


# The top-level modules and subpackages of qldpc.circuits, found on disk.
_UNITS = {
    path.stem
    for path in _ROOT.iterdir()
    if (path.suffix == ".py" and path.stem != "__init__" and _is_source(path))
    or (path / "__init__.py").is_file()
}

# Source files below the root of qldpc.circuits, which re-exports from every layer.
_SOURCES = sorted(
    path for path in _ROOT.rglob("*.py") if _is_source(path) and path != _ROOT / "__init__.py"
)


def test_layers_list_every_unit_once() -> None:
    """_LAYERS lists every top-level module and subpackage of qldpc.circuits exactly once."""
    listed = sorted(unit for layer in _LAYERS for unit in layer)
    assert listed == sorted(_UNITS), "list each top-level module and subpackage in _LAYERS once"


@pytest.mark.parametrize("path", _SOURCES, ids=lambda path: str(path.relative_to(_ROOT)))
def test_imports_from_lower_layers(path: pathlib.Path) -> None:
    """Each module of qldpc.circuits imports only from its own unit or from lower layers."""
    relative_path = path.relative_to(_ROOT)
    unit = relative_path.parts[0].removesuffix(".py")
    package = ".".join((_PACKAGE, *relative_path.parent.parts))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for imported in _get_imported_units(tree, package, _PACKAGE, _UNITS) - {unit}:
        assert imported, f"{relative_path} imports from the root of {_PACKAGE}"
        assert _LAYER_INDEX[imported] < _LAYER_INDEX[unit], (
            f"{relative_path} imports {imported}, which is not in a lower layer than {unit}"
        )


def test_import_parsing() -> None:
    """Imports anywhere in a module resolve to the units of the package that they load."""
    source = """
import pkg.sub.a
from pkg.sub.b import name
from pkg.sub import c, name
from .. import d
from ..e.module import name
from . import module
from pkg import other

def function() -> None:
    from pkg.sub.g import name
"""
    units = {"a", "b", "c", "d", "e", "f", "g", "unused"}
    imported = _get_imported_units(ast.parse(source), "pkg.sub.f", "pkg.sub", units)
    assert imported == {"a", "b", "c", "d", "e", "f", "g", ""}


def _get_imported_units(tree: ast.AST, package: str, root: str, units: Collection[str]) -> set[str]:
    """The units of a root package that a syntax tree imports, from a module in the given package.

    Each unit is a top-level module or subpackage of the root package, and the empty string stands
    for the root package itself.  Imports count wherever they appear, including in functions.
    """
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.rsplit(".", node.level - 1)[0]
                base = f"{anchor}.{base}" if base else anchor
            names = [f"{base}.{alias.name}" for alias in node.names]
        else:
            continue
        for name in names:
            if name == root or name.startswith(f"{root}."):
                unit = name.removeprefix(root).removeprefix(".").partition(".")[0]
                imported.add(unit if unit in units else "")
    return imported
