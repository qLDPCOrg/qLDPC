# SPDX-License-Identifier: Apache-2.0

"""Check that the modules and subpackages of qldpc.circuits import each other in layers."""

from __future__ import annotations

import ast
import pathlib

import pytest

_PACKAGE = "qldpc.circuits"
_ROOT = pathlib.Path(__file__).parent

# The layer of each top-level module or subpackage of qldpc.circuits (see its docstring).
_LAYERS = {
    "common": 1,
    "bookkeeping": 1,
    "transforms": 2,
    "_post_selection": 2,
    "encoding": 3,
    "noise_model": 3,
    "benchmarking": 4,
    "certification": 4,
    "memory": 4,
    "transversal": 4,
}

_SOURCES = sorted(
    path
    for path in _ROOT.rglob("*.py")
    if path.name != "conftest.py" and not path.name.endswith("_test.py")
)


def test_every_unit_has_a_layer() -> None:
    """Every module and subpackage of qldpc.circuits is assigned a layer."""
    units = {_get_unit(_get_module_name(path)) for path in _SOURCES} - {None}
    assert units == _LAYERS.keys()


@pytest.mark.parametrize("path", _SOURCES, ids=lambda path: str(path.relative_to(_ROOT)))
def test_imports_from_lower_layers(path: pathlib.Path) -> None:
    """Each module of qldpc.circuits imports only from its own unit or from lower layers."""
    module = _get_module_name(path)
    unit = _get_unit(module)
    if unit is None:
        return  # the package root re-exports from every layer
    for imported in _get_imported_modules(path, module):
        imported_unit = _get_unit(imported)
        assert imported_unit is not None, f"{module} imports the root of {_PACKAGE}"
        assert imported_unit == unit or _LAYERS[imported_unit] < _LAYERS[unit], (
            f"{module} (layer {_LAYERS[unit]}) imports {imported} (layer {_LAYERS[imported_unit]})"
        )


def test_layering_checks_imports() -> None:
    """The import parser finds relative, absolute, and function-level imports of modules."""
    source = """
from . import common
from .bookkeeping import QubitIDs
from qldpc.circuits.memory import memory
from qldpc import circuits, codes
import qldpc.circuits.transversal

def function() -> None:
    from .noise_model import _moments
"""
    assert _iter_imported_modules(ast.parse(source), "qldpc.circuits.encoding", False) == [
        "qldpc.circuits.common",
        "qldpc.circuits.bookkeeping",
        "qldpc.circuits.memory.memory",
        "qldpc.circuits",
        "qldpc.circuits.transversal",
        "qldpc.circuits.noise_model._moments",
    ]
    source = "from ..common import StimCircuitLike\nfrom .channels import PauliChannel"
    assert _iter_imported_modules(ast.parse(source), "qldpc.circuits.noise_model.rules", False) == [
        "qldpc.circuits.common",
        "qldpc.circuits.noise_model.channels",
    ]
    source = "from .qubit_remapping import with_remapped_qubits"
    assert _iter_imported_modules(ast.parse(source), "qldpc.circuits.transforms", True) == [
        "qldpc.circuits.transforms.qubit_remapping"
    ]


def _get_module_name(path: pathlib.Path) -> str:
    """The dotted name of the module in a source file of qldpc.circuits."""
    parts = path.relative_to(_ROOT).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join((_PACKAGE, *parts))


def _get_unit(module: str) -> str | None:
    """The top-level module or subpackage of qldpc.circuits that contains a module, if any."""
    parts = module.split(".")
    return parts[2] if len(parts) > 2 else None


def _get_imported_modules(path: pathlib.Path, module: str) -> set[str]:
    """The modules of qldpc.circuits that a source file imports, anywhere in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return set(_iter_imported_modules(tree, module, path.name == "__init__.py"))


def _iter_imported_modules(tree: ast.AST, module: str, is_package: bool) -> list[str]:
    """The modules of qldpc.circuits imported in a syntax tree, by a module or package.

    For ``from X import Y``, the imported module is ``X.Y`` if ``Y`` may be a submodule of the
    package ``qldpc.circuits``, and ``X`` otherwise.
    """
    package = module if is_package else module.rpartition(".")[0]
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                anchor = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
                base = f"{anchor}.{base}" if base else anchor
            for alias in node.names:
                name = f"{base}.{alias.name}"
                is_circuits = name == _PACKAGE or name.startswith(f"{_PACKAGE}.")
                imported.append(name if is_circuits and not _defines(base, alias.name) else base)
    return [name for name in imported if name == _PACKAGE or name.startswith(f"{_PACKAGE}.")]


def _defines(module: str, name: str) -> bool:
    """Whether a module of qldpc.circuits, rather than a submodule, provides a name it exports."""
    if not module.startswith(_PACKAGE):
        return False
    path = _ROOT.joinpath(*module.split(".")[2:])
    return not (path / f"{name}.py").exists() and not (path / name / "__init__.py").exists()
