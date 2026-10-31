# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures and test helpers for qLDPC package tests."""

from __future__ import annotations

import functools
import importlib
import inspect
import os
import pathlib
import re
import subprocess
import sys
from collections.abc import Iterator, Mapping, Sequence

import grimp
import pytest

from qldpc import abstract, external

# A comment that lets an import break the layering rule of check_import_layering.
_LAYERING_EXCEPTION = re.compile(r"#\s*layering-exception:\s*\S")


def run_python(
    code: str, *options: str, env: Mapping[str, str] | None = None, timeout: float = 600
) -> str:
    """Run Python code in a fresh interpreter that imports qLDPC from this checkout.

    The child interpreter receives this interpreter's import path as its PYTHONPATH, so that it
    imports the same qLDPC sources rather than whatever qldpc its environment happens to resolve.

    Args:
        code: The code to run, passed to the interpreter with -c.
        *options: Interpreter options that precede -c, such as "-W", "error".
        env: Environment variables to set in the child, in addition to this process's environment.
        timeout: The number of seconds after which to abort the child.  Default: 600.

    Returns:
        The standard output of the child.

    Raises:
        AssertionError: If the child exits with a nonzero return code, showing its standard error.
    """
    import_path = os.pathsep.join(path for path in sys.path if path)
    result = subprocess.run(
        [sys.executable, *options, "-c", code],
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": import_path, **(env or {})},
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


@pytest.fixture(autouse=True)
def isolate_gap_probe_state() -> Iterator[None]:
    """Reset cached GAP availability probes and backend registries around every test.

    These caches and registries are process-wide, so without a reset, one test's mocked (or real)
    GAP availability would leak into later tests that run in the same worker process.
    """
    external.gap.clear_cache()
    yield
    external.gap.clear_cache()


@pytest.fixture(autouse=True)
def block_clipboard(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests of the manual GAP and MAGMA workflows from overwriting the user's clipboard."""
    monkeypatch.setattr("pyperclip.copy", lambda text: None)


@pytest.fixture(scope="session")
def ring_cyclic3_gf2(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a small ring with a pre-built Wedderburn-Artin transformer."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="session")
def ring_cyclic3_gf4(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a ring over a non-prime field with a pre-built Wedderburn-Artin transformer."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=4)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="session")
def ring_dihedral3_gf5(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a non-commutative ring with a pre-built Wedderburn-Artin transformer."""
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="session")
def ring_cyclic5_gf2(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a ring with a degree-4 field-extension component (over a prime field)."""
    ring = abstract.GroupRing(abstract.CyclicGroup(5), field=2)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="session")
def ring_alternating4_gf5(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a non-commutative ring with a size-3 matrix component."""
    ring = abstract.GroupRing(abstract.AlternatingGroup(4), field=5)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(scope="session")
def ring_dihedral5_gf3(pytestconfig: pytest.Config) -> abstract.GroupRing:
    """Construct a ring with a matrix component over a field extension (size 2, degree 2)."""
    ring = abstract.GroupRing(abstract.DihedralGroup(5), field=3)
    ring.get_transformer(seed=pytestconfig.getoption("randomly_seed"))
    return ring


@pytest.fixture(name="ring", scope="session", params=["cyclic3_gf4", "dihedral3_gf5"])
def rings_to_test(
    request: pytest.FixtureRequest,
    ring_cyclic3_gf4: abstract.GroupRing,
    ring_dihedral3_gf5: abstract.GroupRing,
) -> abstract.GroupRing:
    """Retrieve a ring for which we have pre-built a Wedderburn-Artin transformer."""
    match request.param:
        case "cyclic3_gf4":
            return ring_cyclic3_gf4
        case "dihedral3_gf5":
            return ring_dihedral3_gf5
    raise ValueError(f"Invalid fixture name: {request.param}")  # pragma: no cover


@pytest.fixture(
    name="wa_ring",
    scope="session",
    params=["cyclic3_gf4", "dihedral3_gf5", "cyclic5_gf2", "alternating4_gf5", "dihedral5_gf3"],
)
def wedderburn_artin_rings_to_test(
    request: pytest.FixtureRequest,
    ring_cyclic3_gf4: abstract.GroupRing,
    ring_dihedral3_gf5: abstract.GroupRing,
    ring_cyclic5_gf2: abstract.GroupRing,
    ring_alternating4_gf5: abstract.GroupRing,
    ring_dihedral5_gf3: abstract.GroupRing,
) -> abstract.GroupRing:
    """Retrieve a ring for exercising the Wedderburn-Artin machinery.

    These rings jointly cover a commutative field extension (cyclic3_gf4), matrix components of size
    2 and 3 (dihedral3_gf5, alternating4_gf5), a degree-4 field-extension component (cyclic5_gf2),
    and a matrix component that is itself over a field extension -- size 2 and degree 2, i.e. 2x2
    matrices over GF(9) (dihedral5_gf3).
    """
    match request.param:
        case "cyclic3_gf4":
            return ring_cyclic3_gf4
        case "dihedral3_gf5":
            return ring_dihedral3_gf5
        case "cyclic5_gf2":
            return ring_cyclic5_gf2
        case "alternating4_gf5":
            return ring_alternating4_gf5
        case "dihedral5_gf3":
            return ring_dihedral5_gf3
    raise ValueError(f"Invalid fixture name: {request.param}")  # pragma: no cover


####################################################################################################
# import layering


def check_import_layering(package: str, layers: Sequence[Sequence[str]]) -> None:
    """Check that the top-level modules and subpackages of a package import each other in layers.

    Each top-level module or subpackage of the package (a unit) belongs to one of the given layers,
    listed from lowest to highest, and may import only from itself and from lower layers.  No module
    may import the root of the package, which re-exports from every layer.  Together, these rules
    prevent import cycles between units.  Imports count wherever they appear, including in
    functions and under TYPE_CHECKING.  Test modules and test configuration are not checked.

    An import whose line ends with "# layering-exception: <reason>" may break these rules; for an
    import that spans several lines, the comment goes on its first line.  Such a comment that no
    import needs is an error, so that exceptions do not outlive the imports that they excuse.

    Args:
        package: The dotted name of the package to check, such as "qldpc.circuits".
        layers: The units of the package, in layers from lowest to highest.  Every unit must appear
            in exactly one layer.

    Raises:
        AssertionError: If the layers or imports break these rules, listing each problem.
    """
    graph = _get_import_graph()
    sources = _get_source_files(package)
    config = f"_LAYERS in {_get_display_path(sources[package].with_name('layering_test.py'))}"
    listed = [unit for layer in layers for unit in layer]
    layer_index = {unit: index for index, layer in enumerate(layers) for unit in layer}
    units = {_get_unit(package, module) for module in graph.find_children(package)}
    units = {unit for unit in units if not _is_test_name(unit)}

    # lines of imports that break the layering rules, and lines with layering-exception comments
    breaking_imports = {
        (module, detail["line_number"], imported)
        for module in graph.find_descendants(package)
        if module in sources
        for imported in graph.find_modules_directly_imported_by(module)
        if _is_in_package(package, imported)
        and not _may_import(_get_unit(package, module), _get_unit(package, imported), layer_index)
        for detail in graph.get_import_details(importer=module, imported=imported)
    }
    breaking_lines = {(module, line) for module, line, _ in breaking_imports}
    marked_lines = {
        (module, number)
        for module, path in sources.items()
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if _LAYERING_EXCEPTION.search(line)
    }

    problems = [
        *(f"Add {unit} to a layer in {config}." for unit in sorted(units - set(listed))),
        *(f"Remove {unit} from {config}." for unit in sorted(set(listed) - units)),
        *(
            f"List {unit} once in {config}."
            for unit in sorted(set(listed))
            if listed.count(unit) > 1
        ),
        *(
            f"{_get_display_path(sources[module])}:{line} imports {imported}, from the root of"
            f" {package} or from a layer that is not lower.  Move the import, regroup {config}, or"
            " end the line with '# layering-exception: <reason>'."
            for module, line, imported in sorted(breaking_imports)
            if (module, line) not in marked_lines
        ),
        *(
            f"{_get_display_path(sources[module])}:{line} has a layering-exception comment that no"
            " import needs.  Remove it."
            for module, line in sorted(marked_lines - breaking_lines)
        ),
    ]
    assert not problems, "\n".join(problems)


@functools.cache
def _get_import_graph() -> grimp.ImportGraph:
    """The import graph of qLDPC, built once per test process and not cached on disk."""
    return grimp.build_graph("qldpc", cache_dir=None)


def _get_source_files(package: str) -> dict[str, pathlib.Path]:
    """The source files of the modules of a package, other than tests and test configuration."""
    root = pathlib.Path(inspect.getfile(importlib.import_module(package))).parent
    modules = {
        path: ".".join((package, *path.relative_to(root).with_suffix("").parts))
        for path in sorted(root.rglob("*.py"))
        if not _is_test_name(path.stem)
    }
    return {module.removesuffix(".__init__"): path for path, module in modules.items()}


def _get_display_path(path: pathlib.Path) -> str:
    """A path relative to the root of the repository, which contains src/qldpc/conftest.py."""
    return str(path.relative_to(pathlib.Path(__file__).parents[2]))


def _is_test_name(name: str) -> bool:
    """Whether a module name is that of a test module or of test configuration."""
    return name == "conftest" or name.endswith("_test")


def _is_in_package(package: str, module: str) -> bool:
    """Whether a module is a package or one of its descendants."""
    return module == package or module.startswith(f"{package}.")


def _get_unit(package: str, module: str) -> str:
    """The top-level module or subpackage of a package that contains a module of the package.

    The unit of the root of the package is the empty string.
    """
    return module.removeprefix(package).removeprefix(".").partition(".")[0]


def _may_import(unit: str, imported_unit: str, layer_index: Mapping[str, int]) -> bool:
    """Whether the layering rules let a unit import from another unit or from the root ("").

    Imports by or of units that are not in a layer are allowed here, since those units are reported
    separately.
    """
    return imported_unit == unit or (
        imported_unit != ""
        and layer_index.get(imported_unit, -1) < layer_index.get(unit, len(layer_index))
    )
