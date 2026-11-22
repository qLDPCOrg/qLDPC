# SPDX-License-Identifier: Apache-2.0

"""Check the rules of the code standard for compiled functions.

The code standard and the docstring of ``_compilation`` state these rules, and the name of each test
of a rule says which rule it checks.  ``_compilation_test.py`` tests what ``_compilation`` does.
There, ``test_compiled_module_tests_are_those_of_modules_with_compiled_functions`` also fails for a
module whose only compiled functions have neither decorator of ``_compilation``, since
``--compiled-module-tests`` would skip the test file of that module.
"""

from __future__ import annotations

import ast
import functools
from dataclasses import dataclass
from pathlib import Path

import qldpc
from qldpc import conftest
from qldpc.conftest import run_python

from . import _compilation


def test_only_compilation_asks_numba_for_a_disk_cache() -> None:
    """Only ``_compilation`` may ask numba for a disk cache, since it makes that cache safe."""
    for file in _get_source_files():
        for node in file.nodes:
            if not (isinstance(node, ast.Call) and _is_njit(node.func)):
                continue
            if any(keyword.arg == "cache" for keyword in node.keywords):
                assert file.path.name == "_compilation.py", f"{file.path.name} caches without it"


def test_only_compilation_calls_numba_jit() -> None:
    """Numba code uses the decorators of ``_compilation``, which the tests of the rules read."""
    for file in _get_source_files():
        if file.path.name == "_compilation.py":
            continue
        for node in file.nodes:
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("numba"):
                names = {alias.name for alias in node.names}
            elif isinstance(node, ast.Attribute):
                names = {node.attr}
            else:
                continue
            assert not names & {"njit", "jit"}, (
                f"{file.path.name}:{node.lineno} uses numba's njit or jit"
            )


def test_compiling_decorators_are_used_only_as_decorators() -> None:
    """``conftest.get_compiled_functions`` sees the decorators only where ``@`` applies them."""
    for file in _get_source_files():
        if file.path.name == "_compilation.py":
            continue

        # The names under which the module imports the decorators.
        names = {"always_compiled", "compiled_on_request"}
        for node in file.nodes:
            if isinstance(node, ast.ImportFrom):
                names |= {
                    alias.asname for alias in node.names if alias.name in names and alias.asname
                }

        # Each use of those names applies a decorator to a function.
        decorators = {
            decorator.func if isinstance(decorator, ast.Call) else decorator
            for node in file.nodes
            if isinstance(node, ast.FunctionDef)
            for decorator in node.decorator_list
        }
        for node in file.nodes:
            if isinstance(node, ast.Name | ast.Attribute):
                name = node.id if isinstance(node, ast.Name) else node.attr
                assert name not in names or node in decorators, (
                    f"{file.path.name}:{node.lineno} uses {name} other than as a decorator"
                )


def test_nothing_asks_numba_for_parallel_code() -> None:
    """Threads run compiled code, and two that run numba-parallel code at once abort the process.

    ``always_compiled`` has no ``parallel`` option, and compiled code uses nothing from outside
    qLDPC, so this keeps numba-parallel code out of the compiled code that threads run.
    """
    for file in _get_source_files():
        for node in file.nodes:
            if isinstance(node, ast.Call) and _is_njit(node.func):
                names = {keyword.arg for keyword in node.keywords}
                assert "parallel" not in names, f"{file.path.name} asks numba for parallel code"


def test_always_compiled_functions_do_not_call_compiled_on_request_functions() -> None:
    """An always-compiled function is compiled and cached by itself, in every mode."""
    registered = _get_names_decorated_by("compiled_on_request")
    assert registered
    for file in _get_source_files():
        for function in file.compiled_functions["always_compiled"]:
            used = {
                node.id if isinstance(node, ast.Name) else node.attr
                for statement in function.body
                for node in ast.walk(statement)
                if isinstance(node, ast.Name | ast.Attribute)
            }
            assert not used & registered, (
                f"{file.path.name}: {function.name} calls {used & registered}"
            )


def test_compiled_on_request_functions_are_not_imported_by_name() -> None:
    """A name imported by ``from`` keeps the plain function, since only its module is updated."""
    registered = _get_names_decorated_by("compiled_on_request")
    assert registered
    for file in _get_source_files():
        for node in file.nodes:
            if isinstance(node, ast.ImportFrom):
                imported = {alias.name for alias in node.names} & registered
                assert not imported, f"{file.path.name} imports {imported} by name"


def test_compiled_on_request_functions_are_defined_in_the_distance_package() -> None:
    """CI runs only the distance tests with compiled-on-request functions compiled."""
    modules = run_python(_FIND_ON_REQUEST_MODULES).split()
    assert modules
    outside = [
        name
        for name in modules
        if name != "qldpc.codes.distance" and not name.startswith("qldpc.codes.distance.")
    ]
    assert not outside, outside


def test_nogil_is_passed_as_a_literal() -> None:
    """``get_entry_points_without_the_lock`` of ``conftest`` sees ``nogil`` only as a literal.

    It reads every decorator of an always-compiled function, and cannot see ``nogil`` in ``**``.
    """
    num_literals = 0
    for file in _get_source_files():
        for function in file.compiled_functions["always_compiled"]:
            for decorator in function.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                for keyword in decorator.keywords:
                    assert keyword.arg is not None, (
                        f"{file.path.name}: {function.name} passes options by **"
                    )
                    if keyword.arg == "nogil":
                        value = getattr(keyword.value, "value", None)
                        assert isinstance(value, bool), (
                            f"{file.path.name}: {function.name} passes nogil other than as True or"
                            " False"
                        )
                        num_literals += 1
    assert num_literals


def test_run_work_items_is_not_imported_by_name() -> None:
    """A ``run_work_items`` imported by ``from`` keeps the original, which escapes the fixture.

    ``check_code_on_threads`` replaces only ``_threads.run_work_items``, so helpers imported by
    name, which call it there, stay checked.
    """
    for file in _get_source_files():
        for node in file.nodes:
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names}
                assert "run_work_items" not in names, (
                    f"{file.path.name} imports run_work_items by name"
                )


def test_entry_points_without_the_lock_are_in_packages_that_state_rules_on_threads() -> None:
    """The docstring of ``_compilation`` says when a package may compile with ``nogil=True``."""
    entry_points = conftest.get_entry_points_without_the_lock(Path(qldpc.__file__).parent)
    assert entry_points
    outside = [
        f"{module}.{name}"
        for module, name in entry_points
        if not module.startswith(_PACKAGES_WITH_RULES_ON_THREADS)
    ]
    assert not outside, (
        f"{outside} release the lock outside _PACKAGES_WITH_RULES_ON_THREADS.  Compile them without"
        " nogil=True, or meet the conditions in the docstring of _compilation and list the package."
    )


def test_always_compiled_functions_use_nothing_from_outside_qldpc() -> None:
    """Only qLDPC's source is hashed, but numba checks its version and implements NumPy itself."""
    for file in _get_source_files():
        functions = file.compiled_functions["always_compiled"]
        outside = _get_names_imported_from_outside_qldpc(file.tree) if functions else set()
        for function in functions:
            names = {
                node.id
                for statement in function.body
                for node in ast.walk(statement)
                if isinstance(node, ast.Name)
            }
            assert not names & outside, f"{file.path.name}: {function.name} uses {names & outside}"


def test_names_imported_from_outside_qldpc_are_found() -> None:
    """The names that always-compiled code may not use, imported from other packages, are found."""
    tree = ast.parse(
        "import numpy as np\n"
        "import numba.types\n"
        "import scipy.special\n"
        "from other import first as renamed\n"
        "from qldpc.math import second\n"
        "from .. import third\n"
    )
    assert _get_names_imported_from_outside_qldpc(tree) == {"scipy", "renamed"}


# The packages whose functions always-compiled code may use.
_COMPILABLE_PACKAGES = {"qldpc", "numpy", "numba"}

# The packages that meet the conditions of ``_compilation`` for functions with ``nogil=True``.
_PACKAGES_WITH_RULES_ON_THREADS = ("qldpc.codes.distance.", "qldpc.circuits.certification.")

# A program that prints the modules that register compiled-on-request functions.  It first imports
# every module of qLDPC other than tests, which registers them all.
_FIND_ON_REQUEST_MODULES = """
import importlib, pkgutil
import qldpc
from qldpc import _compilation

for info in pkgutil.walk_packages(qldpc.__path__, "qldpc."):
    if not info.name.endswith(("_test", "conftest")):
        importlib.import_module(info.name)
print(*sorted({item.namespace["__name__"] for item in _compilation._ON_REQUEST_FUNCTIONS}))
"""


@dataclass(frozen=True)
class _SourceFile:
    """A source file of the library that is not a test, parsed for the tests of the rules.

    ``nodes`` holds every node of ``tree`` in the order of ``ast.walk``, and ``compiled_functions``
    holds the compiled functions of the module by the name of their decorator.
    """

    path: Path
    tree: ast.Module
    nodes: tuple[ast.AST, ...]
    compiled_functions: dict[str, list[ast.FunctionDef]]


@functools.cache
def _get_source_files() -> tuple[_SourceFile, ...]:
    """Every source file of the library that is not a test.

    The tests of the rules only read the syntax trees, so they share one parse of each file.
    """
    files = []
    for path in _compilation._get_source_paths(Path(qldpc.__file__).parent):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = tuple(ast.walk(tree))
        files.append(_SourceFile(path, tree, nodes, conftest.get_compiled_functions(tree)))
    return tuple(files)


def _is_njit(function: ast.expr) -> bool:
    """Whether an expression is a name or an attribute called ``njit`` or ``jit``."""
    return getattr(function, "attr", getattr(function, "id", "")) in ("njit", "jit")


def _get_names_decorated_by(name: str) -> set[str]:
    """The names of the library's functions with the decorator ``name``, one of ``_compilation``."""
    return {
        function.name for file in _get_source_files() for function in file.compiled_functions[name]
    }


def _get_names_imported_from_outside_qldpc(tree: ast.Module) -> set[str]:
    """The names that a module of qLDPC imports from packages other than qLDPC, NumPy, and numba."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            if (node.module or "").split(".")[0] not in _COMPILABLE_PACKAGES:
                names |= {alias.asname or alias.name for alias in node.names}
        elif isinstance(node, ast.Import):
            names |= {
                alias.asname or alias.name.split(".")[0]
                for alias in node.names
                if alias.name.split(".")[0] not in _COMPILABLE_PACKAGES
            }
    return names
