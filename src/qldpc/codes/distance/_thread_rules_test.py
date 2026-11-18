# SPDX-License-Identifier: Apache-2.0

"""Check the rules on compiled code that threads run, which tests of results can miss.

Section 5 of ``brouwer_zimmermann.md`` and ``connected_cluster.md``, and the docstring of
``qldpc/_compilation.py``, state the rules.  An entry point compiled with ``nogil=True`` runs on
several threads at once, without Python's global interpreter lock, and so does every compiled
function that it calls.  A race between them would rarely change a result that tests compare, so
these tests read the code instead:

- each such entry point is named in Section 5 of the guide of its method, which states its rules;
- it, with the compiled functions that it calls, writes only the arrays that ``_WRITTEN_ARGUMENTS``
  lists for it, which its callers give to one thread or one call at a time; and
- it calls no numba code compiled with ``parallel=True``.

The tests find what a function writes from its syntax tree.  A write is an assignment, or an
augmented assignment, to an element, a slice, or an attribute of an array; an ``out`` argument, or a
positional output, of a NumPy ufunc; a call of a method that changes an array in place, such as
``fill``; or a write by a function of qLDPC that it calls.  A local name aliases the arguments of
the expressions that it is bound to, through subscripts, attributes, views, loops, and the results
of functions of qLDPC that return them.  The analysis follows every function of qLDPC that the
entry point calls, and assumes that other functions, such as those of NumPy, write nothing else.

A name bound only to elements of arrays, such as ``labels[row, word]``, may hold a scalar, which an
augmented assignment rebinds rather than writes.

While every test of this package runs, its fixture ``check_code_on_threads`` of ``conftest.py``
checks that the Python code that threads run calls no numba-parallel code and reads the tuning only
from its search.  ``test_workers_that_call_numba_parallel_code_or_read_the_tuning_fail`` checks
that fixture.
"""

from __future__ import annotations

import ast
import builtins
import functools
import importlib
import inspect
import linecache
import re
import textwrap
import threading
from collections.abc import Callable, Iterator
from pathlib import Path

import numba
import numpy as np
import pytest

import qldpc
from qldpc import _compilation, conftest

from . import _threads, _tuning

# The arguments that each entry point compiled with ``nogil=True`` may write, with the compiled
# functions that it calls: those that its callers give to one thread, or to one call, at a time.
_WRITTEN_ARGUMENTS = {
    "qldpc.codes.distance._brouwer_zimmermann.affine_search.scan_set_items": {
        "support",
        "prefixes",
    },
    "qldpc.codes.distance._brouwer_zimmermann.coset._find_first_not_within_max_weight": {
        "best",
        "planning",
    },
    "qldpc.codes.distance._brouwer_zimmermann.coset._find_min_weights": {"weights"},
    "qldpc.codes.distance._connected_cluster.search._search_supports": set(),
}

# Methods of NumPy arrays that change the array in place.
_MUTATING_METHODS = {"fill", "itemset", "partition", "put", "resize", "setfield", "sort"}

# Methods of NumPy arrays, and NumPy functions, that return a view of their first argument.
_VIEW_METHODS = {"ravel", "reshape", "squeeze", "swapaxes", "transpose", "view"}
_VIEW_FUNCTIONS = {np.ascontiguousarray, np.asarray, np.ravel, np.reshape, np.squeeze}

# NumPy functions that write their first argument.
_WRITING_FUNCTIONS = {np.copyto, np.fill_diagonal, np.place, np.put, np.putmask}

# The names of the scalar types of arguments, which a function cannot change for its caller.
_SCALAR_TYPES = re.compile(r"(bool|int|float|np\.u?int\d*|np\.float\d*)")


def test_entry_points_without_the_lock_are_named_in_section_5_of_their_guides() -> None:
    """Section 5 of the guide of each entry point's method names it, and states its rules.

    An entry point of the Brouwer-Zimmermann or connected-cluster method is named in its method's
    guide, and any other entry point in either guide.
    """
    guides = {
        name: _get_section_5(Path(__file__).with_name(f"{name}.md"))
        for name in ("brouwer_zimmermann", "connected_cluster")
    }
    for path in _get_entry_points():
        module, name = path.rsplit(".", 1)
        methods = [method for method in guides if f"._{method}." in module] or list(guides)
        assert any(f"`{name}`" in guides[method] for method in methods), (
            f"Section 5 of no guide of {methods} names {path}"
        )


def test_entry_points_without_the_lock_write_only_the_arguments_of_one_thread() -> None:
    """Each entry point writes only arguments that its callers give to one thread or call at once.

    Add a new entry point to ``_WRITTEN_ARGUMENTS``, with the arguments that it writes, once Section
    5 of its guide says why no other thread reads or writes them while it runs.
    """
    entry_points = _get_entry_points()
    assert set(entry_points) == set(_WRITTEN_ARGUMENTS)
    for path, function in entry_points.items():
        assert _get_effects(function).written == _WRITTEN_ARGUMENTS[path], path


def test_entry_points_without_the_lock_call_no_numba_parallel_code() -> None:
    """Two threads that run numba-parallel code at once abort the process under numba's workqueue.

    Compiled code may call only compiled functions of qLDPC, but a function of qLDPC that numba
    compiles itself, rather than through ``_compilation``, could be numba-parallel.
    """
    for path, function in _get_entry_points().items():
        assert not _get_effects(function).parallel_calls, path


def test_writes_are_found_through_aliases_calls_and_outputs() -> None:
    """The analysis finds the writes that the module docstring lists, and only those."""
    namespace = _define_functions(
        """
        def write_first(target, source):
            target[0] = source[0]
            return source

        def entry(a, b, c, d, e, f, g, h, i, j, k, read, element_source, count: int):
            view = a[1:].reshape(-1)
            view[0] = 1
            flat = np.ravel(j)
            flat[0] = 1
            first, *rest = read, k
            rest[0][0] = 1
            for row in b:
                row[0] = 0
            np.add(read, 1, out=c)
            np.bitwise_xor(read, read, d)
            e.fill(0)
            write_first(f, read)
            returned = write_first(read[:0], g)
            returned[0] = 0
            h.field[0] = 1
            i += 1
            element = element_source[0]
            element ^= 1
            np.copyto(np.zeros(1), read)
            fresh = np.zeros(3)
            fresh[0] = count
            count += 1
            parallel(read)

        @numba.njit(parallel=True)
        def parallel(array):
            return array
        """
    )
    effects = _get_effects(namespace["entry"])
    assert effects.written == {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "read"}
    assert effects.parallel_calls == {"parallel"}

    # A read through a view writes nothing, and a call through unpacked arguments may write each.
    namespace = _define_functions(
        """
        def entry(source, target):
            total = np.sum(source[1:])
            write_first(*(target, source))
            return total

        def write_first(target, source):
            target[0] = source[0]
        """
    )
    assert _get_effects(namespace["entry"]).written == {"source", "target"}


def test_workers_that_call_numba_parallel_code_or_read_the_tuning_fail() -> None:
    """The check of ``conftest.py`` fails a worker of ``run_work_items`` that breaks these rules.

    Workers run on the calling thread and on a new one.  The calling thread reads the tuning before
    and after them.
    """
    parallel = _define_functions(
        """
        @numba.njit(parallel=True)
        def parallel(array):
            return array
        """
    )["parallel"]
    assert callable(parallel)

    def start_worker(stop: threading.Event) -> Callable[[int, int], None]:
        with pytest.raises(AssertionError, match="get_tuning"):
            _tuning.get_tuning()

        def run_range(start: int, end: int) -> None:
            with pytest.raises(AssertionError, match="numba-parallel"):
                parallel(np.zeros(1))
            with pytest.raises(AssertionError, match="get_tuning"):
                _tuning.get_tuning()

        return run_range

    tuning = _tuning.get_tuning()
    _threads.run_work_items(4, 2, start_worker)
    assert _tuning.get_tuning() is tuning


####################################################################################################
# helpers


class _Effects:
    """What a function, with the functions of qLDPC that it calls, may do to its arguments.

    ``written`` holds the arguments that it may write, ``returned`` those that its result may
    alias, and ``parallel_calls`` the names of the numba-parallel functions that it calls.
    """

    def __init__(self) -> None:
        self.written: set[str] = set()
        self.returned: set[str] = set()
        self.parallel_calls: set[str] = set()


@functools.cache
def _get_entry_points() -> dict[str, Callable[..., object]]:
    """The plain Python functions of the entry points compiled with ``nogil=True``, by full name."""
    entry_points = {}
    package = Path(qldpc.__file__).parent
    for path in _compilation._get_source_paths(package):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for function in conftest.get_compiled_functions(tree)["always_compiled"]:
            if any(_releases_the_lock(decorator) for decorator in function.decorator_list):
                parts = path.relative_to(package.parent).with_suffix("").parts
                module = importlib.import_module(".".join(parts))
                entry_points[f"{module.__name__}.{function.name}"] = _get_python_function(
                    getattr(module, function.name)
                )
    return entry_points


def _releases_the_lock(decorator: ast.expr) -> bool:
    """Whether a decorator compiles its function with ``nogil=True``."""
    return isinstance(decorator, ast.Call) and any(
        keyword.arg == "nogil" and getattr(keyword.value, "value", False) is True
        for keyword in decorator.keywords
    )


def _get_section_5(guide: Path) -> str:
    """The text of Section 5 of a guide."""
    text = guide.read_text(encoding="utf-8")
    return text.split("\n## 5.", 1)[1].split("\n## ", 1)[0]


def _define_functions(source: str) -> dict[str, object]:
    """The namespace of a module of qLDPC with the given source, in which ``inspect`` finds it."""
    source = textwrap.dedent(source)
    filename = f"<qldpc.thread_rules_test.{abs(hash(source))}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace: dict[str, object] = {"__name__": "qldpc.example", "np": np, "numba": numba}
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - the source of this test
    return namespace


def _get_python_function(function: object) -> Callable[..., object]:
    """The plain Python function of a numba dispatcher, or the function itself."""
    python_function = getattr(function, "py_func", function)
    assert callable(python_function)
    return python_function


def _is_qldpc_function(function: object) -> bool:
    """Whether an object is a function, or a numba dispatcher of a function, defined in qLDPC."""
    function = getattr(function, "py_func", function)
    return inspect.isfunction(function) and function.__module__.split(".")[0] == "qldpc"


@functools.cache
def _get_effects(function: Callable[..., object]) -> _Effects:
    """What a function of qLDPC, with the functions of qLDPC that it calls, does to its arguments.

    A function that calls itself, directly or not, sees its own effects as those found so far.
    """
    effects = _Effects()
    _EFFECTS_IN_PROGRESS[function] = effects
    try:
        _find_effects(function, effects)
    finally:
        del _EFFECTS_IN_PROGRESS[function]
    return effects


# The effects of the functions whose analysis is in progress, which recursive calls see.
_EFFECTS_IN_PROGRESS: dict[Callable[..., object], _Effects] = {}


def _find_effects(function: Callable[..., object], effects: _Effects) -> None:
    """Fill in the effects of a function of qLDPC (see ``_get_effects``)."""
    node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
    assert isinstance(node, ast.FunctionDef)
    arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
    aliases: dict[str, set[str]] = {
        argument.arg: set()
        if argument.annotation is not None
        and _SCALAR_TYPES.fullmatch(ast.unparse(argument.annotation))
        else {argument.arg}
        for argument in arguments
    }
    resolve = functools.partial(_resolve, function.__globals__)

    def get_aliases(expression: ast.expr | None) -> set[str]:
        """The arguments that the value of an expression may alias."""
        if isinstance(expression, ast.Name):
            return aliases.get(expression.id, set())
        if isinstance(expression, ast.Subscript | ast.Attribute | ast.Starred):
            return get_aliases(expression.value)
        if isinstance(expression, ast.Tuple | ast.List):
            return set().union(*map(get_aliases, expression.elts))
        if isinstance(expression, ast.IfExp):
            return get_aliases(expression.body) | get_aliases(expression.orelse)
        if isinstance(expression, ast.Call):
            callee = resolve(expression.func)
            # A method of an array, which its module does not resolve, may return a view of it.
            function = expression.func
            if callee is None and isinstance(function, ast.Attribute):
                return get_aliases(function.value) if function.attr in _VIEW_METHODS else set()
            if callee in _VIEW_FUNCTIONS:
                return get_aliases(expression.args[0])
            if _is_qldpc_function(callee):
                returned = _get_callee_effects(callee).returned
                return _get_bound_aliases(callee, expression, returned, get_aliases)
        return set()

    def bind(target: ast.expr, value: ast.expr | None) -> bool:
        """Let the names of a target alias what a value aliases, and say whether that grew."""
        if isinstance(target, ast.Tuple | ast.List):
            if isinstance(value, ast.Tuple | ast.List) and len(value.elts) == len(target.elts):
                grew = [bind(each, part) for each, part in zip(target.elts, value.elts)]
            else:
                grew = [bind(each, value) for each in target.elts]
            return any(grew)
        if isinstance(target, ast.Starred):
            return bind(target.value, value)
        if isinstance(target, ast.Name):
            new = get_aliases(value) - aliases.setdefault(target.id, set())
            aliases[target.id] |= new
            return bool(new)
        return False

    # Bind names until their aliases stop growing, since a loop can bind a name after its use.
    grew = True
    while grew:
        grew = False
        for target, value in _get_bindings(node):
            grew = target is not None and bind(target, value) or grew

    array_names = _get_array_names(node, aliases)
    for statement in ast.walk(node):
        if isinstance(statement, ast.Assign | ast.AugAssign | ast.AnnAssign):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            for target in targets:
                effects.written |= _get_written(target, get_aliases)
            if (
                isinstance(statement, ast.AugAssign)
                and isinstance(statement.target, ast.Name)
                and statement.target.id in array_names
            ):
                effects.written |= get_aliases(statement.target)
        elif isinstance(statement, ast.Call):
            callee = resolve(statement.func)
            effects.written |= _get_call_writes(statement, callee, get_aliases)
            if isinstance(callee, numba.core.dispatcher.Dispatcher) and callee.targetoptions.get(
                "parallel"
            ):
                effects.parallel_calls.add(ast.unparse(statement.func))
            if _is_qldpc_function(callee):
                effects.parallel_calls |= _get_callee_effects(callee).parallel_calls
        elif isinstance(statement, ast.Return):
            effects.returned |= get_aliases(statement.value)


def _get_callee_effects(callee: object) -> _Effects:
    """The effects of a function of qLDPC that a function calls."""
    function = _get_python_function(callee)
    return _EFFECTS_IN_PROGRESS.get(function) or _get_effects(function)


def _get_bindings(node: ast.FunctionDef) -> Iterator[tuple[ast.expr | None, ast.expr | None]]:
    """The targets of the bindings of names in a function, with the values bound to them.

    A loop binds its target to the elements of what it iterates over, which alias that.
    """
    for statement in ast.walk(node):
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                yield target, statement.value
        elif isinstance(statement, ast.AnnAssign | ast.NamedExpr):
            yield statement.target, statement.value
        elif isinstance(statement, ast.For):
            yield statement.target, statement.iter


def _get_array_names(node: ast.FunctionDef, aliases: dict[str, set[str]]) -> set[str]:
    """The names in a function that hold arrays, for all that its syntax shows.

    An argument that is not a scalar, a name that the function subscripts, and a name bound to a
    name, an attribute, a slice, or a view hold arrays.  A name bound only to elements, such as
    ``labels[row, word]``, may hold a scalar, which an augmented assignment rebinds.
    """
    names = {name for name, argument_aliases in aliases.items() if name in argument_aliases}
    for each in ast.walk(node):
        if isinstance(each, ast.Subscript) and isinstance(each.value, ast.Name):
            names.add(each.value.id)
    for target, value in _get_bindings(node):
        if isinstance(target, ast.Name) and _is_array_expression(value):
            names.add(target.id)
    return names


def _is_array_expression(expression: ast.expr | None) -> bool:
    """Whether an expression is a name, an attribute, a slice, or a view of an array."""
    if isinstance(expression, ast.Name | ast.Attribute):
        return True
    if isinstance(expression, ast.Subscript):
        index = expression.slice
        parts = index.elts if isinstance(index, ast.Tuple) else [index]
        return any(isinstance(part, ast.Slice) for part in parts)
    return (
        isinstance(expression, ast.Call)
        and isinstance(expression.func, ast.Attribute)
        and expression.func.attr in _VIEW_METHODS
    )


def _get_written(target: ast.expr, get_aliases: Callable[[ast.expr], set[str]]) -> set[str]:
    """The arguments that an assignment to a target writes: those of its elements or attributes."""
    if isinstance(target, ast.Tuple | ast.List):
        return set().union(*(_get_written(each, get_aliases) for each in target.elts))
    if isinstance(target, ast.Subscript | ast.Attribute):
        return get_aliases(target.value)
    return set()


def _get_call_writes(
    call: ast.Call, callee: object, get_aliases: Callable[[ast.expr], set[str]]
) -> set[str]:
    """The arguments that a call writes, by an output, an in-place method, or a callee's write."""
    written = set().union(
        *(get_aliases(keyword.value) for keyword in call.keywords if keyword.arg == "out")
    )
    if isinstance(call.func, ast.Attribute) and call.func.attr in _MUTATING_METHODS:
        written |= get_aliases(call.func.value)
    if isinstance(callee, np.ufunc):
        written |= set().union(*map(get_aliases, call.args[callee.nin :]))
    if callee in _WRITING_FUNCTIONS:
        written |= get_aliases(call.args[0])
    if _is_qldpc_function(callee):
        callee_written = _get_callee_effects(callee).written
        written |= _get_bound_aliases(callee, call, callee_written, get_aliases)
    return written


def _get_bound_aliases(
    callee: object,
    call: ast.Call,
    parameters: set[str],
    get_aliases: Callable[[ast.expr], set[str]],
) -> set[str]:
    """What the arguments of a call that bind to the given parameters of its callee alias.

    If the call unpacks arguments, any of them may bind to any parameter.
    """
    if not parameters:
        return set()
    arguments: list[ast.expr] = [*call.args, *(keyword.value for keyword in call.keywords)]
    if any(isinstance(argument, ast.Starred) for argument in call.args) or any(
        keyword.arg is None for keyword in call.keywords
    ):
        return set().union(*map(get_aliases, arguments))
    names = list(inspect.signature(_get_python_function(callee)).parameters)
    bound = dict(zip(names, call.args)) | {
        keyword.arg: keyword.value for keyword in call.keywords if keyword.arg is not None
    }
    return set().union(*(get_aliases(bound[name]) for name in parameters if name in bound))


def _resolve(namespace: dict[str, object], expression: ast.expr) -> object:
    """The object that a name or attribute in a module's namespace refers to, or None."""
    if isinstance(expression, ast.Name):
        return namespace.get(expression.id, getattr(builtins, expression.id, None))
    if isinstance(expression, ast.Attribute):
        owner = _resolve(namespace, expression.value)
        return None if owner is None else getattr(owner, expression.attr, None)
    return None
