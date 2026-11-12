# SPDX-License-Identifier: Apache-2.0

"""Tests for compiled functions, and for running their tests compiled or as plain Python.

They cover the disk cache, the registry of compiled-on-request functions, the switch to plain
Python, the options of the compiled runs, and the rules for compiled functions.
"""

from __future__ import annotations

import ast
import os
import pickle
import time
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import numba
import numpy as np
import numpy.typing as npt
import platformdirs
import pytest

import qldpc
from qldpc import conftest
from qldpc.conftest import run_python

from . import _compilation


@pytest.fixture(autouse=True)
def enable_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Compile by default in each test here, even in a run with ``QLDPC_DISABLE_COMPILATION``."""
    monkeypatch.setattr(_compilation, "_DISABLED_BY_VARIABLE", False)


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


def test_always_compiled_functions_are_cached_in_the_chosen_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_compilation, "_get_cache_directory", lambda package: tmp_path / "cache")
    previous = _compilation._numba_config.CACHE_DIR
    compiled = _compilation.always_compiled(_add_one)
    assert _compilation._numba_config.CACHE_DIR == previous
    cache_path = compiled.stats.cache_path
    assert cache_path is not None and Path(cache_path).is_relative_to(tmp_path / "cache")
    assert compiled(1) == _add_one(1) == 2


def test_inlined_functions_are_cached_and_their_compiled_callers_include_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_compilation, "_get_cache_directory", lambda package: tmp_path / "cache")
    inlined = _compilation.always_compiled(inline=True)(_add_one)
    cache_path = inlined.stats.cache_path
    assert cache_path is not None and Path(cache_path).is_relative_to(tmp_path / "cache")
    caller = numba.njit(lambda value: inlined(value) * 2)
    assert caller(1) == 4
    assert not any("_add_one" in code for code in caller.inspect_llvm().values())


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


def test_the_source_hash_skips_entries_that_are_not_imported(tmp_path: Path) -> None:
    """Such as the lock files that editors make beside modified files, as in ``.#source.py``."""
    (tmp_path / "source.py").write_text("one")
    expected = _compilation._get_source_hash(tmp_path)
    (tmp_path / ".#source.py").write_text("a hidden file")
    (tmp_path / "directory.py").mkdir()
    assert _compilation._get_source_hash(tmp_path) == expected


def test_there_is_no_cache_directory_if_the_source_files_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "source.py").write_text("one")

    def refuse(path: Path) -> bytes:
        raise PermissionError("unreadable")

    monkeypatch.setattr(Path, "read_bytes", refuse)
    with pytest.warns(RuntimeWarning, match="cannot be read.*reinstall qLDPC"):
        assert _compilation._get_cache_directory(tmp_path) is None


def test_the_cache_directory_is_named_by_the_source_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is in ``NUMBA_CACHE_DIR`` if that is set, and otherwise in the user cache directory."""
    package = tmp_path / "package"
    package.mkdir()
    (package / "source.py").write_text("one")
    name = f"qldpc-{_compilation._get_source_hash(package)}"

    monkeypatch.setattr(_compilation._numba_config, "CACHE_DIR", str(tmp_path / "numba"))
    assert _compilation._get_cache_directory(package) == tmp_path / "numba" / name

    monkeypatch.setattr(_compilation._numba_config, "CACHE_DIR", "")
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda: str(tmp_path / "user"))
    assert _compilation._get_cache_directory(package) == tmp_path / "user/qldpc/numba" / name


def test_outdated_cache_directories_unused_for_a_week_are_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only directories of earlier source files of the same package directory are outdated.

    The directory in use is marked as used, so that its age restarts.
    """
    package = tmp_path / "package"
    package.mkdir()
    (package / "source.py").write_text("one")
    source_hash = _compilation._get_source_hash(package)
    assert source_hash is not None
    base = tmp_path / "numba"
    monkeypatch.setattr(_compilation._numba_config, "CACHE_DIR", str(base))

    def name(digit: str) -> str:
        return f"qldpc-{digit * len(source_hash)}"

    in_use = base / f"qldpc-{source_hash}"
    outdated, recent, elsewhere, unknown = (base / name(digit) for digit in "0123")
    other = base / f"qldpc-tools_{'0' * 40}"  # where numba caches code from a folder "qldpc-tools"
    day = 24 * 60 * 60
    now = time.time()
    for directory, source, age in [
        (in_use, package, 8),
        (outdated, package, 8),
        (recent, package, 6),
        (elsewhere, tmp_path / "other_checkout", 8),
        (unknown, None, 8),
        (other, package, 8),
    ]:
        (directory / "contents").mkdir(parents=True)
        if source is not None:
            (directory / _compilation._SOURCE_MARKER).write_text(str(source.resolve()))
        os.utime(directory, (now - age * day, now - age * day))

    assert _compilation._get_cache_directory(package) == in_use
    kept = {in_use, recent, elsewhere, unknown, other}
    assert {path.name for path in base.iterdir()} == {path.name for path in kept}
    assert in_use.stat().st_mtime > now - day


def test_a_new_cache_directory_names_its_package_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "source.py").write_text("one")
    monkeypatch.setattr(_compilation._numba_config, "CACHE_DIR", str(tmp_path / "numba"))
    directory = _compilation._get_cache_directory(tmp_path)
    assert directory is not None
    assert _compilation._read_source_marker(directory) == str(tmp_path.resolve())


def test_errors_while_deleting_outdated_cache_directories_are_ignored(tmp_path: Path) -> None:
    """For example, another process may have deleted the directory meanwhile."""
    _compilation._delete_outdated_cache_directories(tmp_path / "missing" / "qldpc-0", tmp_path)


def test_there_is_no_cache_directory_without_source_files(tmp_path: Path) -> None:
    with pytest.warns(RuntimeWarning, match="no source files.*with its .py source files"):
        assert _compilation._get_cache_directory(tmp_path) is None


def test_an_unwritable_cache_directory_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "source.py").write_text("one")
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("a file, so that no directory can be made inside it")
    monkeypatch.setattr(_compilation._numba_config, "CACHE_DIR", str(not_a_directory))
    with pytest.warns(RuntimeWarning, match="cannot be written.*NUMBA_CACHE_DIR"):
        assert _compilation._get_cache_directory(tmp_path) is None


def test_a_read_only_cache_directory_is_not_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Numba would move on to other cache locations, or fail if there were none."""
    (tmp_path / "source.py").write_text("one")
    monkeypatch.setattr(_compilation._numba_config, "CACHE_DIR", str(tmp_path / "numba"))

    # The directory exists, but the system refuses to create files in it.
    def refuse(*args: object, **kwargs: object) -> None:
        raise PermissionError("read-only")

    monkeypatch.setattr(_compilation.tempfile, "TemporaryFile", refuse)
    with pytest.warns(RuntimeWarning, match="cannot be written.*NUMBA_CACHE_DIR"):
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

    def cache_elsewhere(*, cache: bool = False, inline: Literal["always", "never"]) -> object:
        if cache:
            elsewhere = types.SimpleNamespace(cache_path=str(tmp_path / "elsewhere"))
            return lambda function: types.SimpleNamespace(stats=elsewhere)
        return njit(inline=inline)

    monkeypatch.setattr(_compilation.numba, "njit", cache_elsewhere)
    with pytest.warns(RuntimeWarning, match="numba does not use.*NUMBA_CACHE_DIR"):
        compiled = _compilation.always_compiled(_add_one)
    assert compiled.stats.cache_path is None
    assert compiled(1) == 2


def test_versions_compiled_at_the_same_time_are_loaded_only_for_their_argument_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A version compiled for some argument types is never loaded for others.

    The docstring of ``_compilation`` explains how processes that compile a function at once can
    leave numba's index pointing at code compiled for other argument types.  Without the check of
    keys, a later process here would run code compiled for F-ordered arrays on a C-ordered array and
    return a wrong entry.  Each new dispatcher stands for a process, and replacing the index file
    stages the order of the writes.
    """
    monkeypatch.setattr(_compilation, "_get_cache_directory", lambda package: tmp_path / "cache")
    matrix = np.array([[0, 1], [2, 3]], dtype=np.uint8)

    # Two processes read the empty index before either writes it.  The first compiles for C-ordered
    # arrays, and the second for F-ordered arrays.  The second writes the data file last, and the
    # first writes the index last.
    assert _compilation.always_compiled(_get_top_right_entry)(matrix) == 1
    (index,) = (tmp_path / "cache").rglob("*.nbi")
    first_index = index.read_bytes()
    index.unlink()
    assert _compilation.always_compiled(_get_top_right_entry)(np.asfortranarray(matrix)) == 1
    index.write_bytes(first_index)

    # A later process compiles for C-ordered arrays again, and saves that code in place of the
    # other, which the process after it loads.
    later_process = _compilation.always_compiled(_get_top_right_entry)
    assert later_process(matrix) == _get_top_right_entry(matrix) == 1
    assert later_process.stats.cache_misses
    next_process = _compilation.always_compiled(_get_top_right_entry)
    assert next_process(matrix) == 1
    assert next_process.stats.cache_hits


@pytest.mark.parametrize("missing_module", [False, True], ids=["loadable", "missing_module"])
def test_code_saved_by_another_version_of_numba_is_compiled_again(
    missing_module: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Code that another version of numba saved for the same key is compiled again, not loaded.

    ``_KeyCheckedCacheFile`` explains how the index of this version can name a data file that
    another version of numba wrote, and why its code is not unpickled.  In one case, this version
    can unpickle that code.  In the other, the code refers to a module that this version lacks, as
    code that numba 0.61.2 saved does for numba 0.67.0.
    """
    monkeypatch.setattr(_compilation, "_get_cache_directory", lambda package: tmp_path / "cache")

    # A process with another version of numba writes the data file last, and a process with this
    # version writes the index last.
    assert _compilation.always_compiled(_add_one)(1) == 2
    (index,) = (tmp_path / "cache").rglob("*.nbi")
    first_index = index.read_bytes()
    with monkeypatch.context() as other_numba:
        other_numba.setattr(numba, "__version__", "other")
        assert _compilation.always_compiled(_add_one)(1) == 2
    if missing_module:
        # A pickle that refers to the global ``name`` of a module that does not exist.
        (data_file,) = (tmp_path / "cache").rglob("*.nbc")
        data_file.write_bytes(pickle.dumps(("other", b"cno_such_module\nname\n.")))
    index.write_bytes(first_index)

    # A later process with this version compiles again.
    later_process = _compilation.always_compiled(_add_one)
    assert later_process(1) == 2
    assert later_process.stats.cache_misses


def test_a_function_is_not_cached_if_numba_reads_its_files_through_another_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """qLDPC caches only through a cache-file object exactly of the class that qLDPC extends.

    Here ``object`` stands for the class that qLDPC extends, so numba's cache-file object is of a
    subclass of it.
    """
    monkeypatch.setattr(_compilation, "_get_cache_directory", lambda package: tmp_path / "cache")
    monkeypatch.setattr(_compilation, "caching", types.SimpleNamespace(IndexDataCacheFile=object))
    with pytest.warns(RuntimeWarning, match="class that qLDPC extends.*earlier version"):
        compiled = _compilation.always_compiled(_add_one)
    assert compiled.stats.cache_path is None
    assert compiled(1) == 2


@pytest.mark.parametrize("switch", ["qldpc", "numba"])
def test_functions_run_as_plain_python_if_compilation_is_disabled(
    switch: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Either qLDPC's switch or numba's leaves functions of both kinds as they are.

    qLDPC never looks up its directory of compiled code, so it neither uses nor cleans up its
    directories.
    """
    monkeypatch.setattr(
        _compilation,
        "_get_cache_directory",
        lambda package: pytest.fail("looked up the cache directory"),
    )
    if switch == "qldpc":
        monkeypatch.setattr(_compilation, "_DISABLED_BY_VARIABLE", True)
    else:
        monkeypatch.setattr(_compilation._numba_config, "DISABLE_JIT", 1)
    assert _compilation.always_compiled(_add_one) is _add_one
    assert _compilation.always_compiled(inline=True)(_add_one) is _add_one

    monkeypatch.setattr(_compilation, "_ON_REQUEST_FUNCTIONS", [])
    plain = _compilation.compiled_on_request(_scaled_sum)
    try:
        _compilation.set_on_request_functions_compiled(True)
        assert globals()["_scaled_sum"] is plain
    finally:
        _compilation.set_on_request_functions_compiled(False)


def test_the_environment_variable_disables_compilation(monkeypatch: pytest.MonkeyPatch) -> None:
    """``QLDPC_DISABLE_COMPILATION`` disables compilation if it is 1 when qLDPC is imported.

    The interpreters that ``run_python`` starts do not inherit it, so they compile as a user's does.
    """
    code = "from qldpc import _binary\nprint(type(_binary.pack_bits).__name__)"
    monkeypatch.setenv("QLDPC_DISABLE_COMPILATION", "1")
    assert run_python(code).strip() == "CPUDispatcher"
    assert run_python(code, env={"QLDPC_DISABLE_COMPILATION": "0"}).strip() == "CPUDispatcher"
    assert run_python(code, env={"QLDPC_DISABLE_COMPILATION": "1"}).strip() == "function"


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
    tree = ast.parse(
        "import numpy as np\n"
        "import numba.types\n"
        "import scipy.special\n"
        "from other import first as renamed\n"
        "from qldpc.math import second\n"
        "from .. import third\n"
    )
    assert _get_names_imported_from_outside_qldpc(tree) == {"scipy", "renamed"}


def test_compiled_module_tests_are_those_of_modules_with_compiled_functions(
    pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--compiled-module-tests`` collects just the test files of modules with compiled functions.

    An independent program imports every module of qLDPC, and lists those that hold compiled
    functions of their own or register compiled-on-request functions.  The selection counts only
    functions with ``always_compiled`` or ``compiled_on_request``.  So this fails if a module's only
    compiled functions have neither, such as functions compiled by a module-level ``numba.njit``
    call, since the selection would then skip its test file.
    """
    # Every test file of qLDPC is collected if and only if the program lists its module.
    compiled_modules = run_python(_FIND_COMPILED_MODULES).split()
    assert "qldpc._binary" in compiled_modules
    package = Path(qldpc.__file__).parent
    monkeypatch.setattr(pytestconfig.option, "compiled_module_tests", True)
    for test_file in package.rglob("*_test.py"):
        module = ".".join(("qldpc", *test_file.relative_to(package).with_suffix("").parts))
        ignored = module.removesuffix("_test") not in compiled_modules
        assert conftest.pytest_ignore_collect(test_file, pytestconfig) is (
            True if ignored else None
        )

    # Directories are never skipped, and nothing is skipped without the option.
    assert conftest.pytest_ignore_collect(package / "codes", pytestconfig) is None
    monkeypatch.setattr(pytestconfig.option, "compiled_module_tests", False)
    assert conftest.pytest_ignore_collect(package / "codes/common_test.py", pytestconfig) is None

    # The decorators count however they are imported, but a function with their name does not.
    functions = conftest.get_compiled_functions(ast.parse(_DECORATED_SOURCE))
    assert {name: [function.name for function in found] for name, found in functions.items()} == {
        "always_compiled": ["bare", "qualified"],
        "compiled_on_request": ["aliased"],
    }


def test_the_compile_search_setup_option_compiles_once_and_switches_on_before_each_test(
    request: pytest.FixtureRequest, pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``--compile-search-setup``, each process compiles the set-up code of searches once.

    Before each test, the compiled-on-request functions are switched to their compiled versions.
    Without the option, nothing is compiled.
    """
    calls: list[None] = []
    monkeypatch.setattr(qldpc.codes.distance, "compile_search_setup", lambda: calls.append(None))
    # Start from an empty cache, as a new process does.
    conftest.compile_search_setup_once.cache_clear()
    monkeypatch.setattr(_compilation, "_ON_REQUEST_FUNCTIONS", [])
    plain = _compilation.compiled_on_request(_scaled_sum)

    # Without the option, nothing is compiled.
    monkeypatch.setattr(pytestconfig.option, "compile_search_setup", False)
    conftest.pytest_runtest_setup(request.node)
    assert globals()["_scaled_sum"] is plain and not calls

    # Two tests in a row, each of which switches the functions back to plain Python.
    monkeypatch.setattr(pytestconfig.option, "compile_search_setup", True)
    try:
        for _ in range(2):
            conftest.pytest_runtest_setup(request.node)
            assert isinstance(globals()["_scaled_sum"], numba.core.registry.CPUDispatcher)
            _compilation.set_on_request_functions_compiled(False)
            assert globals()["_scaled_sum"] is plain
    finally:
        _compilation.set_on_request_functions_compiled(False)
        conftest.compile_search_setup_once.cache_clear()
    assert calls == [None]


@pytest.mark.parametrize(
    "switch, option", [("qldpc", "compiled_module_tests"), ("numba", "compile_search_setup")]
)
def test_options_of_compiled_runs_are_refused_while_compilation_is_disabled(
    switch: str, option: str, pytestconfig: pytest.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each option is refused while either switch disables compilation, and accepted otherwise.

    Each case pairs one option with one switch, so that dropping the check of any one option or
    switch fails a case.
    """
    monkeypatch.setattr(pytestconfig.option, "compiled_module_tests", False)
    monkeypatch.setattr(pytestconfig.option, "compile_search_setup", False)
    monkeypatch.setattr(pytestconfig.option, option, True)
    conftest.pytest_configure(pytestconfig)
    if switch == "qldpc":
        monkeypatch.setattr(_compilation, "_DISABLED_BY_VARIABLE", True)
    else:
        monkeypatch.setattr(_compilation._numba_config, "DISABLE_JIT", True)
    with pytest.raises(pytest.UsageError) as error:
        conftest.pytest_configure(pytestconfig)
    names = ("compiled_module_tests", "compile_search_setup")
    named = [f"--{name.replace('_', '-')}" in str(error.value) for name in names]
    assert named == [name == option for name in names]

    # Without the options, disabled compilation is accepted.
    monkeypatch.setattr(pytestconfig.option, option, False)
    conftest.pytest_configure(pytestconfig)


# The packages whose functions always-compiled code may use.
_COMPILABLE_PACKAGES = {"qldpc", "numpy", "numba"}

# Code that imports every module of qLDPC other than tests, which registers every compiled function.
_IMPORT_EVERY_MODULE = """
import importlib, pkgutil, sys
import numba
import qldpc
from qldpc import _compilation

for info in pkgutil.walk_packages(qldpc.__path__, "qldpc."):
    if not info.name.endswith(("_test", "conftest")):
        importlib.import_module(info.name)
"""

# A program that prints the modules that register compiled-on-request functions.
_FIND_ON_REQUEST_MODULES = (
    _IMPORT_EVERY_MODULE
    + """
print(*sorted({item.namespace["__name__"] for item in _compilation._ON_REQUEST_FUNCTIONS}))
"""
)

# A program that prints the modules that hold compiled functions of their own or register
# compiled-on-request functions.
_FIND_COMPILED_MODULES = (
    _IMPORT_EVERY_MODULE
    + """
modules = {item.namespace["__name__"] for item in _compilation._ON_REQUEST_FUNCTIONS}
for name, module in list(sys.modules.items()):
    if module is None or not name.startswith("qldpc."):
        continue
    for value in vars(module).values():
        is_dispatcher = isinstance(value, numba.core.registry.CPUDispatcher)
        if is_dispatcher and value.py_func.__module__ == name:
            modules.add(name)
print(*sorted(modules))
"""
)

# A module whose functions ``bare`` and ``qualified`` are always compiled, and ``aliased`` is
# compiled on request, beside two functions that are neither.  Each name says how its decorator is
# written.
_DECORATED_SOURCE = """
import functools
from qldpc import _compilation
from qldpc._compilation import always_compiled, compiled_on_request as on_request

@always_compiled
def bare(): ...

@_compilation.always_compiled(inline=True)
def qualified(): ...

@on_request
def aliased(): ...

@functools.cache
def always_compiled(): ...

def undecorated(): ...
"""

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


def _get_top_right_entry(matrix: npt.NDArray[np.uint8]) -> int:
    return matrix[0, -1]


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
