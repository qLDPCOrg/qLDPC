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
from collections.abc import Iterator
from pathlib import Path

import qldpc
from qldpc import conftest
from qldpc.conftest import run_python

from . import _compilation


def test_only_always_compiled_asks_numba_for_a_disk_cache() -> None:
    """Only ``always_compiled`` may ask numba for a disk cache, since it makes that cache safe."""
    for path, tree in _get_source_trees():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _is_njit(node.func)):
                continue
            if any(keyword.arg == "cache" for keyword in node.keywords):
                assert path.name == "_compilation.py", f"{path.name} caches without it"


def test_nothing_asks_numba_for_parallel_code() -> None:
    """Threads run compiled code, and two that run numba-parallel code at once abort the process.

    ``always_compiled`` has no ``parallel`` option, and compiled code uses nothing from outside
    qLDPC, so this keeps numba-parallel code out of the compiled code that threads run.
    """
    for path, tree in _get_source_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_njit(node.func):
                names = {keyword.arg for keyword in node.keywords}
                assert "parallel" not in names, f"{path.name} asks numba for parallel code"


def test_always_compiled_functions_do_not_call_compiled_on_request_functions() -> None:
    """An always-compiled function is compiled and cached by itself, in every mode."""
    trees = list(_get_source_trees())
    registered = _get_names_decorated_by(trees, "compiled_on_request")
    assert registered
    for path, tree in trees:
        for function in _get_functions_decorated_by(tree, "always_compiled"):
            used = {
                node.id if isinstance(node, ast.Name) else node.attr
                for statement in function.body
                for node in ast.walk(statement)
                if isinstance(node, ast.Name | ast.Attribute)
            }
            assert not used & registered, f"{path.name}: {function.name} calls {used & registered}"


def test_compiled_on_request_functions_are_not_imported_by_name() -> None:
    """A name imported by ``from`` keeps the plain function, since only its module is updated."""
    trees = list(_get_source_trees())
    registered = _get_names_decorated_by(trees, "compiled_on_request")
    assert registered
    for path, tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                imported = {alias.name for alias in node.names} & registered
                assert not imported, f"{path.name} imports {imported} by name"


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
    for path, tree in _get_source_trees():
        for function in _get_functions_decorated_by(tree, "always_compiled"):
            for decorator in function.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                for keyword in decorator.keywords:
                    assert keyword.arg is not None, (
                        f"{path.name}: {function.name} passes options by **"
                    )
                    if keyword.arg == "nogil":
                        value = getattr(keyword.value, "value", None)
                        assert isinstance(value, bool), (
                            f"{path.name}: {function.name} passes nogil other than as True or False"
                        )
                        num_literals += 1
    assert num_literals


def test_run_work_items_is_not_imported_by_name() -> None:
    """A ``run_work_items`` imported by ``from`` keeps the original, which escapes the fixture.

    ``check_code_on_threads`` replaces only ``_threads.run_work_items``, so helpers imported by
    name, which call it there, stay checked.
    """
    for path, tree in _get_source_trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names}
                assert "run_work_items" not in names, f"{path.name} imports run_work_items by name"


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
    for path, tree in _get_source_trees():
        outside = _get_names_imported_from_outside_qldpc(tree)
        for function in _get_functions_decorated_by(tree, "always_compiled"):
            names = {
                node.id
                for statement in function.body
                for node in ast.walk(statement)
                if isinstance(node, ast.Name)
            }
            assert not names & outside, f"{path.name}: {function.name} uses {names & outside}"


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


def _get_source_trees() -> Iterator[tuple[Path, ast.Module]]:
    """The syntax tree of every source file of the library that is not a test."""
    for path in _compilation._get_source_paths(Path(qldpc.__file__).parent):
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def _is_njit(function: ast.expr) -> bool:
    """Whether an expression is a name or an attribute called ``njit`` or ``jit``."""
    return getattr(function, "attr", getattr(function, "id", "")) in ("njit", "jit")


def _get_functions_decorated_by(tree: ast.Module, name: str) -> list[ast.FunctionDef]:
    """The functions of a module with the decorator ``name``: one of the two of ``_compilation``."""
    return conftest.get_compiled_functions(tree)[name]


def _get_names_decorated_by(trees: list[tuple[Path, ast.Module]], name: str) -> set[str]:
    """The names of the functions in the given syntax trees that have the decorator ``name``."""
    return {
        function.name for _, tree in trees for function in _get_functions_decorated_by(tree, name)
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
