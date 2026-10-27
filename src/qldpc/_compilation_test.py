# SPDX-License-Identifier: Apache-2.0

"""Tests for compiled functions: the disk cache, the registry, and the rules that make them safe."""

from __future__ import annotations

import ast
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numba
import numpy as np
import numpy.typing as npt
import platformdirs
import pytest

import qldpc
from qldpc.conftest import run_python

from . import _compilation


def test_registered_functions_switch_between_plain_and_compiled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A registered function is plain by default, compiled on request, and restorable."""
    monkeypatch.setattr(_compilation, "_ON_REQUEST_FUNCTIONS", [])
    plain = _scaled_sum
    assert _compilation.compiled_on_request(plain) is plain
    values = np.arange(5, dtype=np.int64)

    _compilation.set_on_request_functions_compiled(True)
    compiled = globals()["_scaled_sum"]
    try:
        assert isinstance(compiled, numba.core.registry.CPUDispatcher)
        assert compiled(values, 3) == plain(values, 3) == 30

        # Switching back, and on again, reuses the function that was already compiled.
        _compilation.set_on_request_functions_compiled(False)
        assert globals()["_scaled_sum"] is plain
        _compilation.set_on_request_functions_compiled(True)
        assert globals()["_scaled_sum"] is compiled
    finally:
        _compilation.set_on_request_functions_compiled(False)
    assert globals()["_scaled_sum"] is plain

    # Nothing is cached on disk, so that no outdated function can ever be loaded.
    assert compiled.stats.cache_path is None


def test_always_compiled_functions_are_cached_in_the_directory_of_their_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_compilation, "_get_cache_directory", lambda package: tmp_path / "cache")
    previous = _compilation._numba_config.CACHE_DIR
    compiled = _compilation.always_compiled(_add_one)
    assert _compilation._numba_config.CACHE_DIR == previous
    cache_path = compiled.stats.cache_path
    assert cache_path is not None and Path(cache_path).parent == tmp_path / "cache"
    assert compiled(1) == _add_one(1) == 2


def test_the_source_hash_covers_every_source_file_but_no_tests(tmp_path: Path) -> None:
    assert _compilation._get_source_hash(tmp_path) is None
    (tmp_path / "source.py").write_text("one")
    first = _compilation._get_source_hash(tmp_path)
    assert first is not None and _compilation._get_source_hash(tmp_path) == first

    (tmp_path / "source_test.py").write_text("tests do not change the compiled code")
    (tmp_path / "conftest.py").write_text("neither do test fixtures")
    assert _compilation._get_source_hash(tmp_path) == first

    (tmp_path / "source.py").write_text("two")
    changed = _compilation._get_source_hash(tmp_path)
    (tmp_path / "package").mkdir()
    (tmp_path / "package" / "other.py").write_text("a file of a subpackage")
    assert len({first, changed, _compilation._get_source_hash(tmp_path)}) == 3


def test_the_cache_directory_is_named_by_the_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is in ``NUMBA_CACHE_DIR`` if that is set, and otherwise in the user cache directory."""
    packages = [tmp_path / "first", tmp_path / "second"]
    for package in packages:
        package.mkdir()
        (package / "source.py").write_text("one")
    name = f"qldpc-{_compilation._get_source_hash(packages[0])}"

    monkeypatch.setattr(_compilation._numba_config, "CACHE_DIR", str(tmp_path / "numba"))
    assert _compilation._get_cache_directory(packages[0]) == tmp_path / "numba" / name

    monkeypatch.setattr(_compilation._numba_config, "CACHE_DIR", "")
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda: str(tmp_path / "user"))
    assert _compilation._get_cache_directory(packages[1]) == tmp_path / "user/qldpc/numba" / name


def test_there_is_no_cache_directory_without_source_files(tmp_path: Path) -> None:
    with pytest.warns(RuntimeWarning, match="no source files"):
        assert _compilation._get_cache_directory(tmp_path) is None


def test_an_unwritable_cache_directory_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "source.py").write_text("one")
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("a file, so that no directory can be made inside it")
    monkeypatch.setattr(_compilation._numba_config, "CACHE_DIR", str(not_a_directory))
    with pytest.warns(RuntimeWarning, match="cannot be written"):
        assert _compilation._get_cache_directory(tmp_path) is None


def test_a_read_only_cache_directory_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Numba would move on to other cache locations, or fail if there were none."""
    (tmp_path / "source.py").write_text("one")
    monkeypatch.setattr(_compilation._numba_config, "CACHE_DIR", str(tmp_path / "numba"))

    # The directory exists, but the system refuses to create files in it.
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr(_compilation.tempfile, "TemporaryFile", refuse)
    with pytest.warns(RuntimeWarning, match="cannot be written"):
        assert _compilation._get_cache_directory(tmp_path) is None


def test_functions_are_not_cached_without_a_cache_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_compilation, "_get_cache_directory", lambda package: None)
    compiled = _compilation.always_compiled(_add_one)
    assert compiled.stats.cache_path is None
    assert compiled(1) == 2


def test_a_function_is_not_cached_if_numba_does_not_use_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Numba moves on to its other cache locations, where outdated code could be loaded."""
    monkeypatch.setattr(_compilation, "_get_cache_directory", lambda package: tmp_path / "cache")
    njit = numba.njit

    def cache_elsewhere(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("cache"):
            elsewhere = types.SimpleNamespace(cache_path=str(tmp_path / "elsewhere"))
            return lambda function: types.SimpleNamespace(stats=elsewhere)
        return njit(*args, **kwargs)

    monkeypatch.setattr(_compilation.numba, "njit", cache_elsewhere)
    with pytest.warns(RuntimeWarning, match="numba does not use"):
        compiled = _compilation.always_compiled(_add_one)
    assert compiled.stats.cache_path is None
    assert compiled(1) == 2


def test_always_compiled_functions_are_plain_python_if_numba_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_compilation._numba_config, "DISABLE_JIT", True)
    compiled = _compilation.always_compiled(_add_one)
    assert compiled(1) == 2
    assert not hasattr(compiled, "stats")


def test_editing_a_callee_in_another_subpackage_recompiles_its_caller(tmp_path: Path) -> None:
    """Numba alone would serve the outdated caller, since it checks only the caller's own file."""
    package = tmp_path / "package"
    (package / "other").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "other" / "__init__.py").write_text("")
    (package / "caller.py").write_text(_CALLER)
    code = f"""
import pathlib, sys
from qldpc import _compilation
_compilation._PACKAGE_DIRECTORY = pathlib.Path({str(package)!r})
sys.path.insert(0, {str(tmp_path)!r})
from package import caller
print(caller.add_to(1))
"""
    results = []
    for increment in (1, 10):
        (package / "other" / "callee.py").write_text(_CALLEE.format(increment=increment))
        results.append(int(run_python(code, env={"NUMBA_CACHE_DIR": str(tmp_path / "cache")})))
    assert results == [2, 11]


def test_only_always_compiled_asks_numba_for_a_disk_cache() -> None:
    """Only ``always_compiled`` may ask numba for a disk cache, since it makes that cache safe."""
    for path, tree in _get_source_trees():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and _is_njit(node.func)):
                continue
            if any(keyword.arg == "cache" for keyword in node.keywords):
                assert path.name == "_compilation.py", f"{path.name} caches without it"


def test_always_compiled_functions_do_not_call_compiled_on_request_functions() -> None:
    """An always-compiled function is compiled and cached by itself, in every mode."""
    trees = list(_get_source_trees())
    registered = _get_names_decorated_by(trees, "compiled_on_request")
    assert registered
    for path, tree in trees:
        for function in ast.walk(tree):
            if isinstance(function, ast.FunctionDef) and _is_decorated_by(
                function, "always_compiled"
            ):
                used = {
                    node.id if isinstance(node, ast.Name) else node.attr
                    for statement in function.body
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Name | ast.Attribute)
                }
                assert not used & registered, (
                    f"{path.name}: {function.name} calls {used & registered}"
                )


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


def test_always_compiled_functions_use_nothing_from_outside_qldpc() -> None:
    """Only qLDPC's source is hashed, but numba checks its version and implements NumPy itself."""
    for path, tree in _get_source_trees():
        outside = _get_names_imported_from_outside_qldpc(tree)
        for function in ast.walk(tree):
            if isinstance(function, ast.FunctionDef) and _is_decorated_by(
                function, "always_compiled"
            ):
                names = {
                    node.id
                    for statement in function.body
                    for node in ast.walk(statement)
                    if isinstance(node, ast.Name)
                }
                assert not names & outside, f"{path.name}: {function.name} uses {names & outside}"


def test_names_imported_from_outside_qldpc_are_found() -> None:
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

_CALLEE = """
import numba

@numba.njit
def increment(value):
    return value + {increment}
"""

_CALLER = """
from qldpc._compilation import always_compiled
from .other.callee import increment

@always_compiled
def add_to(value):
    return increment(value)
"""


def _scaled_sum(values: npt.NDArray[np.int64], scale: int) -> int:
    total = 0
    for value in values:
        total += scale * value
    return total


def _add_one(value: int) -> int:
    return value + 1


def _get_source_trees() -> Iterator[tuple[Path, ast.Module]]:
    """The syntax tree of every source file of the library that is not a test."""
    for path in sorted(Path(qldpc.__file__).parent.rglob("*.py")):
        if not path.name.endswith("_test.py") and path.name != "conftest.py":
            yield path, ast.parse(path.read_text())


def _is_njit(function: ast.expr) -> bool:
    """Whether an expression is a name or an attribute called ``njit`` or ``jit``."""
    return getattr(function, "attr", getattr(function, "id", "")) in ("njit", "jit")


def _is_decorated_by(function: ast.FunctionDef, name: str) -> bool:
    """Whether a function has the decorator of the given name, however it was imported."""
    for decorator in function.decorator_list:
        decorator_name = (
            decorator.attr if isinstance(decorator, ast.Attribute) else getattr(decorator, "id", "")
        )
        if decorator_name == name:
            return True
    return False


def _get_names_decorated_by(trees: list[tuple[Path, ast.Module]], name: str) -> set[str]:
    """The names of the functions in the given syntax trees that have the decorator ``name``."""
    return {
        function.name
        for _, tree in trees
        for function in ast.walk(tree)
        if isinstance(function, ast.FunctionDef) and _is_decorated_by(function, name)
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
