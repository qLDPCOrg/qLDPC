# SPDX-License-Identifier: Apache-2.0

"""Check that the modules and subpackages of qldpc.circuits import each other in layers."""

from __future__ import annotations

import re

import grimp
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

# An import statement whose line ends with this marker and a reason may break the layering rule.
# For an import statement that spans several lines, the marker goes on its first line.
_EXCEPTION_MARKER = re.compile(r"#\s*layering-exception:\s*\S")

_PACKAGE = "qldpc.circuits"
_LAYER_INDEX = {unit: index for index, layer in enumerate(_LAYERS) for unit in layer}


@pytest.fixture(scope="module")
def graph() -> grimp.ImportGraph:
    """The import graph of qLDPC, which includes imports in functions."""
    return grimp.build_graph("qldpc", cache_dir=None)


def test_layers_list_every_unit_once(graph: grimp.ImportGraph) -> None:
    """_LAYERS lists every top-level module and subpackage of qldpc.circuits exactly once."""
    units = {_get_unit(module) for module in graph.find_children(_PACKAGE) if _is_source(module)}
    listed = sorted(unit for layer in _LAYERS for unit in layer)
    assert listed == sorted(units), "list each top-level module and subpackage in _LAYERS once"


def test_imports_from_lower_layers(graph: grimp.ImportGraph) -> None:
    """Modules of qldpc.circuits import only from their own unit or lower layers, or say why not."""
    violations = [
        f"{module}:{line} imports {imported}"
        for module, imported, statements in _get_imports(graph)
        if not _may_import(module, imported)
        for line, marked in statements
        if not marked
    ]
    message = (
        "These imports come from the root of qldpc.circuits or from layers that are not lower.  Move"
        " them, regroup the layers in _LAYERS, or end each import line with"
        " '# layering-exception: <reason>':\n" + "\n".join(violations)
    )
    assert not violations, message


def test_layering_exceptions_are_needed(graph: grimp.ImportGraph) -> None:
    """Imports that follow the layering rule are not marked as exceptions to it."""
    unneeded = [
        f"{module}:{line} imports {imported}"
        for module, imported, statements in _get_imports(graph)
        if _may_import(module, imported)
        for line, marked in statements
        if marked
    ]
    message = "Remove the layering-exception markers of these imports:\n" + "\n".join(unneeded)
    assert not unneeded, message


def _get_imports(graph: grimp.ImportGraph) -> list[tuple[str, str, list[tuple[int, bool]]]]:
    """Imports between modules of qldpc.circuits, other than test modules and test configuration.

    Each import is a tuple of the importing module, the imported module, and the line number of each
    statement that makes the import, with whether that statement is marked as a layering exception.
    """
    return [
        (module, imported, _get_statements(graph, module, imported))
        for module in sorted(graph.find_descendants(_PACKAGE))
        if _is_source(module)
        for imported in sorted(graph.find_modules_directly_imported_by(module))
        if imported == _PACKAGE or imported.startswith(f"{_PACKAGE}.")
    ]


def _get_statements(graph: grimp.ImportGraph, module: str, imported: str) -> list[tuple[int, bool]]:
    """The line number of each statement of an import, with whether it is a layering exception."""
    details = graph.get_import_details(importer=module, imported=imported)
    return [
        (detail["line_number"], bool(_EXCEPTION_MARKER.search(detail["line_contents"])))
        for detail in details
    ]


def _is_source(module: str) -> bool:
    """Whether a module is source code, rather than a test module or test configuration."""
    name = module.rpartition(".")[2]
    return name != "conftest" and not name.endswith("_test")


def _get_unit(module: str) -> str:
    """The top-level module or subpackage of qldpc.circuits that contains a module of the package.

    The unit of the root of qldpc.circuits, which re-exports from every layer, is the empty string.
    """
    return module.removeprefix(_PACKAGE).removeprefix(".").partition(".")[0]


def _may_import(module: str, imported: str) -> bool:
    """Whether the layering rule lets a module of qldpc.circuits import another module of it."""
    unit, imported_unit = _get_unit(module), _get_unit(imported)
    return imported_unit == unit or (
        imported_unit != "" and _LAYER_INDEX[imported_unit] < _LAYER_INDEX[unit]
    )
