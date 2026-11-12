# SPDX-License-Identifier: Apache-2.0

"""Shared pytest fixtures, hooks, and test helpers for qLDPC package tests."""

from __future__ import annotations

import ast
import builtins
import functools
import importlib
import inspect
import os
import pathlib
import re
import subprocess
import sys
import types
from collections.abc import Iterator, Mapping, Sequence

import galois
import grimp
import pytest

from qldpc import _compilation, abstract, codes, external

# A comment that lets an import break the layering rule of check_import_layering.
_LAYERING_EXCEPTION = re.compile(r"#\s*layering-exception:\s*\S")


def run_python(
    code: str, *options: str, env: Mapping[str, str] | None = None, timeout: float = 600
) -> str:
    """Run Python code in a fresh interpreter that imports qLDPC from this checkout.

    The child interpreter receives this interpreter's import path as its PYTHONPATH, so that it
    imports the same qLDPC sources rather than whatever qldpc its environment happens to resolve.
    The child does not inherit ``QLDPC_DISABLE_COMPILATION``, so that it compiles qLDPC's functions
    as a user's interpreter does.  ``env`` can still set that variable, and the child inherits
    ``NUMBA_DISABLE_JIT``.

    Args:
        code: The code to run, passed to the interpreter with -c.
        *options: Interpreter options that precede -c, such as "-W", "error".
        env: Environment variables to set in the child, in addition to the ones that it inherits.
        timeout: The number of seconds after which to abort the child.  Default: 600.

    Returns:
        The standard output of the child.

    Raises:
        AssertionError: If the child exits with a nonzero return code, showing its standard error.
    """
    import_path = os.pathsep.join(path for path in sys.path if path)
    environment = {**os.environ, "PYTHONPATH": import_path}
    environment.pop(_compilation.DISABLE_COMPILATION_VARIABLE, None)
    result = subprocess.run(
        [sys.executable, *options, "-c", code],
        capture_output=True,
        check=False,
        env={**environment, **(env or {})},
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def fail_import(monkeypatch: pytest.MonkeyPatch, module: str, error: ImportError) -> None:
    """Make importing a module raise an error, while every other import works as usual.

    Other imports must keep working: other threads, such as those of pytest-xdist workers, can
    import modules while the error is in place.
    """
    original_import = builtins.__import__

    def import_or_fail(
        name: str,
        globals_: dict[str, object] | None = None,
        locals_: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> types.ModuleType:
        if name == module:
            raise error
        return original_import(name, globals_, locals_, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_or_fail)


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


@pytest.fixture(autouse=True, scope="session")
def vary_default_transformer_seed(pytestconfig: pytest.Config) -> Iterator[None]:
    """Build seedless Wedderburn-Artin decompositions from the seed of the test run.

    GroupRing.get_transformer uses a fixed seed by default, so without this fixture, tests would
    exercise only one decomposition of each ring.
    """
    with pytest.MonkeyPatch.context() as monkeypatch:
        seed = pytestconfig.getoption("randomly_seed", default=0)
        monkeypatch.setattr(abstract.rings, "_DEFAULT_TRANSFORMER_SEED", seed)
        yield


@pytest.fixture(scope="session")
def ring_cyclic3_gf2() -> abstract.GroupRing:
    """Construct a small ring with a pre-built Wedderburn-Artin transformer."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=2)
    ring.get_transformer()
    return ring


@pytest.fixture(scope="session")
def ring_cyclic3_gf4() -> abstract.GroupRing:
    """Construct a ring over a non-prime field with a pre-built Wedderburn-Artin transformer."""
    ring = abstract.GroupRing(abstract.CyclicGroup(3), field=4)
    ring.get_transformer()
    return ring


@pytest.fixture(scope="session")
def ring_dihedral3_gf5() -> abstract.GroupRing:
    """Construct a non-commutative ring with a pre-built Wedderburn-Artin transformer."""
    ring = abstract.GroupRing(abstract.DihedralGroup(3), field=5)
    ring.get_transformer()
    return ring


@pytest.fixture(scope="session")
def ring_cyclic5_gf2() -> abstract.GroupRing:
    """Construct a ring with a degree-4 field-extension component (over a prime field)."""
    ring = abstract.GroupRing(abstract.CyclicGroup(5), field=2)
    ring.get_transformer()
    return ring


@pytest.fixture(scope="session")
def ring_alternating4_gf5() -> abstract.GroupRing:
    """Construct a non-commutative ring with a size-3 matrix component."""
    ring = abstract.GroupRing(abstract.AlternatingGroup(4), field=5)
    ring.get_transformer()
    return ring


@pytest.fixture(scope="session")
def ring_dihedral5_gf3() -> abstract.GroupRing:
    """Construct a ring with a matrix component over a field extension (size 2, degree 2)."""
    ring = abstract.GroupRing(abstract.DihedralGroup(5), field=3)
    ring.get_transformer()
    return ring


@pytest.fixture(scope="session")
def ring_dihedral7_gf9() -> abstract.GroupRing:
    """Construct a ring over a field whose irreducible polynomial is not the default one.

    In GF(9) with the irreducible polynomial x^2 + 1, the element x is not primitive.  This ring has
    a matrix component over a field extension (size 2, degree 3).
    """
    field = galois.GF(9, irreducible_poly="x^2 + 1")
    ring = abstract.GroupRing(abstract.DihedralGroup(7), field=field)
    ring.get_transformer()
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
    params=[
        "cyclic3_gf4",
        "dihedral3_gf5",
        "cyclic5_gf2",
        "alternating4_gf5",
        "dihedral5_gf3",
        "dihedral7_gf9",
    ],
)
def wedderburn_artin_rings_to_test(
    request: pytest.FixtureRequest,
    ring_cyclic3_gf4: abstract.GroupRing,
    ring_dihedral3_gf5: abstract.GroupRing,
    ring_cyclic5_gf2: abstract.GroupRing,
    ring_alternating4_gf5: abstract.GroupRing,
    ring_dihedral5_gf3: abstract.GroupRing,
    ring_dihedral7_gf9: abstract.GroupRing,
) -> abstract.GroupRing:
    """Retrieve a ring for exercising the Wedderburn-Artin machinery.

    These rings jointly cover a commutative field extension (cyclic3_gf4), matrix components of size
    2 and 3 (dihedral3_gf5, alternating4_gf5), a degree-4 field-extension component (cyclic5_gf2),
    a matrix component that is itself over a field extension -- size 2 and degree 2, i.e. 2x2
    matrices over GF(9) (dihedral5_gf3), and a field whose irreducible polynomial is not the default
    one (dihedral7_gf9).
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
        case "dihedral7_gf9":
            return ring_dihedral7_gf9
    raise ValueError(f"Invalid fixture name: {request.param}")  # pragma: no cover


####################################################################################################
# runs with compiled code

# The decorators of qldpc._compilation that make numba compile a function.
_COMPILING_DECORATORS = frozenset({"always_compiled", "compiled_on_request"})


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the options of test runs with compiled code."""
    # Explicit defaults let pytest set these options when it loads this file after reading the
    # command line, as in a run of ``pytest src``.
    parser.addoption(
        "--compiled-module-tests",
        action="store_true",
        default=False,
        help="Collect, from the directories given, only the test files of the modules that define"
        " compiled functions.",
    )
    parser.addoption(
        "--compile-search-setup",
        action="store_true",
        default=False,
        help="Switch the set-up code of distance and count searches to compiled code before every"
        " test, as qldpc.codes.distance.compile_search_setup() does.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Refuse the options of runs with compiled code while compilation is disabled."""
    options = [
        option
        for option in ("--compiled-module-tests", "--compile-search-setup")
        if config.getoption(option)
    ]
    if options and _compilation.is_compilation_disabled():
        raise pytest.UsageError(
            f"Compilation is disabled, but it is needed by {' and '.join(options)}.  Unset"
            " QLDPC_DISABLE_COMPILATION and NUMBA_DISABLE_JIT."
        )


def pytest_ignore_collect(collection_path: pathlib.Path, config: pytest.Config) -> bool | None:
    """With ``--compiled-module-tests``, skip each test file whose module has no compiled function.

    The module of a test file ``x_test.py`` is ``x.py`` beside it.  Whatever this returns, pytest
    collects a test file named on the command line if no directory named there contains it.
    """
    if config.getoption("--compiled-module-tests") and collection_path.name.endswith("_test.py"):
        module = collection_path.with_name(collection_path.name.removesuffix("_test.py") + ".py")
        if not _defines_compiled_functions(module):
            return True
    return None


def get_compiled_functions(tree: ast.Module) -> dict[str, list[ast.FunctionDef]]:
    """The functions of a module that numba compiles, by the name of the decorator that marks them.

    The decorators are ``always_compiled`` and ``compiled_on_request``, however they are imported,
    as in ``@always_compiled``, ``@_compilation.always_compiled(inline=True)``, or a name that
    ``from ... import ... as`` gives them.
    """
    # Map each name under which the module imports a decorator to the decorator's name.
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            aliases |= {
                alias.asname or alias.name: alias.name
                for alias in node.names
                if alias.name in _COMPILING_DECORATORS
            }

    # Find the functions with each decorator.
    functions: dict[str, list[ast.FunctionDef]] = {name: [] for name in _COMPILING_DECORATORS}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):  # a decorator with options
                    decorator = decorator.func
                name = (
                    decorator.attr
                    if isinstance(decorator, ast.Attribute)
                    else aliases.get(getattr(decorator, "id", ""))
                )
                if name in functions:
                    functions[name].append(node)
    return functions


def _defines_compiled_functions(path: pathlib.Path) -> bool:
    """Whether a module exists and has a function with a decorator of compiled code.

    The decorators are ``always_compiled`` and ``compiled_on_request`` (see get_compiled_functions).
    """
    if not path.is_file():
        return False
    return any(get_compiled_functions(ast.parse(path.read_text(encoding="utf-8"))).values())


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """With ``--compile-search-setup``, switch the set-up code of searches to compiled code.

    The code is compiled by ``compile_search_setup`` before the first test of each process.
    Fixtures such as ``plain_search_setup`` of the distance tests may switch it back to plain Python
    for one test, so this hook switches it to compiled code before every test, ahead of the test's
    fixtures.
    """
    if item.config.getoption("--compile-search-setup"):
        compile_search_setup_once()
        _compilation.set_on_request_functions_compiled(True)


@functools.cache
def compile_search_setup_once() -> None:
    """Compile the set-up code of searches, the first time that this is called in a process."""
    codes.distance.compile_search_setup()


####################################################################################################
# cache clearing between test files

# If this environment variable is "1", qLDPC's functools caches are emptied after each test file
# (see pytest_runtest_teardown).  The coverage check sets it.
CLEAR_CACHES_VARIABLE = "QLDPC_CLEAR_CACHES"


@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown(item: pytest.Item, nextitem: pytest.Item | None) -> Iterator[None]:
    """Empty qLDPC's functools caches after each test file if QLDPC_CLEAR_CACHES is "1".

    The coverage check runs several test files in each process.  A cache that one test file fills
    could keep lines from running in a later test file, which would then fail coverage falsely.  As
    the outermost wrapper of the hook, this empties the caches after every other implementation of
    the hook has run, including those that tear down fixtures.  It applies to the test files under
    the directory of this conftest.py file.
    """
    result = yield
    if os.environ.get(CLEAR_CACHES_VARIABLE) == "1" and (
        nextitem is None or nextitem.path != item.path
    ):
        _clear_caches()
    return result


def _clear_caches() -> None:
    """Empty every functools cache of a function or method of a module of qLDPC."""
    for name, module in list(sys.modules.items()):
        if module is None or name.partition(".")[0] != "qldpc":
            continue
        for value in list(vars(module).values()):
            # Modules hold no cache.  Looking up a missing attribute of one could run its
            # __getattr__, such as one that warns of a deprecated name or finishes a lazy import.
            if isinstance(value, types.ModuleType):
                continue
            candidates = [value]
            if isinstance(value, type) and value.__module__ == name:
                candidates += [
                    getattr(member, "__func__", member) for member in vars(value).values()
                ]
            for candidate in candidates:
                cache_clear = getattr(candidate, "cache_clear", None)
                if callable(cache_clear):
                    cache_clear()


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
