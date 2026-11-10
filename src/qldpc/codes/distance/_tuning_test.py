# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tuning of exact searches."""

from __future__ import annotations

import ast
import collections
import concurrent.futures
import contextvars
import dataclasses
import inspect
import math
import pathlib
import re
import sys
from fractions import Fraction
from typing import Any

import numpy as np
import pytest
from sympy.abc import x, y

from qldpc import codes
from qldpc.objects import Pauli

from . import _tuning, core
from . import logical_counts as lc
from ._counting import scan
from ._search import brouwer_zimmermann as bz
from ._search import connected_support, information_sets
from .conftest import CostlyStages, get_random_css_code, get_sector_problem


def test_tuning_changes_apply_within_their_block() -> None:
    """Changes apply until their block exits, nest with the inner one first, and survive errors."""
    assert _tuning.get_tuning() == _tuning.DEFAULT_TUNING
    with _tuning.use_tuning(max_information_sets=4, lp_minimum_work=0) as outer:
        assert _tuning.get_tuning() is outer
        assert (outer.max_information_sets, outer.lp_minimum_work) == (4, 0)
        with _tuning.use_tuning(max_information_sets=2) as inner:
            assert (inner.max_information_sets, inner.lp_minimum_work) == (2, 0)
        assert _tuning.get_tuning() is outer
        with pytest.raises(RuntimeError), _tuning.use_tuning(allow_enumeration_by_size=False):
            raise RuntimeError
        assert _tuning.get_tuning() is outer
    assert _tuning.get_tuning() == _tuning.DEFAULT_TUNING


def test_copies_of_the_context_keep_the_tuning_of_their_copy() -> None:
    """Code run in a copy of the context uses the tuning that was active when the copy was made."""
    with _tuning.use_tuning(max_information_sets=4):
        context = contextvars.copy_context()
    assert _tuning.get_tuning() == _tuning.DEFAULT_TUNING
    assert context.run(_tuning.get_tuning).max_information_sets == 4


def test_new_threads_start_with_the_default_tuning() -> None:
    """A change applies to the thread that makes it, but not to new threads that it starts.

    Builds of Python whose new threads inherit the context of the thread that starts them are the
    exception that the module docstring states.
    """
    inherits = getattr(sys.flags, "thread_inherit_context", False)
    with (
        _tuning.use_tuning(max_information_sets=4),
        concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor,
    ):
        worker_tuning = executor.submit(_tuning.get_tuning).result()
        assert worker_tuning == (_tuning.get_tuning() if inherits else _tuning.DEFAULT_TUNING)
        assert _tuning.get_tuning().max_information_sets == 4


def test_tunings_are_validated() -> None:
    """Fields hold Python numbers, and a tuning rejects wrong types and out-of-range values."""
    numpy_fields: dict[str, Any] = {"max_information_sets": np.int64(4)}
    assert type(_tuning.SearchTuning(**numpy_fields).max_information_sets) is int
    assert _tuning.SearchTuning(lp_minimum_work=_tuning.MAX_VALUE).lp_minimum_work == 2**62
    # Float fields hold Python floats, from any real number.
    floats: dict[str, Any] = {"completion_bound_confidence_z": 3, "completion_bound_samples": 8}
    assert type(_tuning.SearchTuning(**floats).completion_bound_confidence_z) is float

    invalid: list[tuple[type[Exception], str, dict[str, Any]]] = [
        (TypeError, "allow_enumeration_by_work must be a bool", {"allow_enumeration_by_work": 1}),
        (TypeError, "must be an integer", {"max_information_sets": True}),
        (TypeError, "must be an integer", {"lp_minimum_work": 1.5}),
        (ValueError, "between 0 and", {"overlapping_minimum_work": -1}),
        (ValueError, "between 0 and", {"lp_minimum_work": _tuning.MAX_VALUE + 1}),
        (ValueError, "between 1 and", {"scan_chunk_size": 0}),
        (ValueError, "between 0 and 62", {"information_set_batch_exponent": 63}),
        (ValueError, "must be one of", {"completion_bound_block_bits": 12}),
        (TypeError, "must be a real number", {"completion_bound_confidence_z": True}),
        (TypeError, "must be a real number", {"completion_bound_confidence_z": "2"}),
        (ValueError, "non-negative and finite", {"completion_bound_lookup_block_cost": -0.5}),
        (ValueError, "non-negative and finite", {"completion_bound_table_byte_cost": math.inf}),
        (ValueError, "at most 10.0", {"completion_bound_confidence_z": 10.5}),
        (ValueError, "must be finite", {"completion_bound_lookup_block_cost": Fraction(10**400)}),
    ]
    for error, message, fields in invalid:
        with pytest.raises(error, match=message):
            _tuning.SearchTuning(**fields)

    # The fields with bounds other than zero and MAX_VALUE, or with choices, accept their edges and
    # reject values past them.
    widths = {"choices": (1, 2, 4, 8, 16)}
    narrowed: dict[str, dict[str, Any]] = {
        "lp_dual_max_denominator": {"minimum": 1},
        "completion_bound_block_bits": widths,
        "completion_bound_fallback_block_bits": widths,
        "completion_bound_samples": {"minimum": 1, "maximum": 2**12},
        "completion_bound_min_samples": {"minimum": 1},
        "completion_bound_confidence_z": {"maximum": 10.0},
        "enumeration_batch_exponent": {"maximum": 62},
        "information_set_batch_exponent": {"maximum": 62},
        "scan_chunk_size": {"minimum": 1},
        "scan_chunks_per_call": {"minimum": 1},
        "macwilliams_chunk_size": {"minimum": 1},
    }
    record = dataclasses.fields(_tuning.SearchTuning)
    assert {field.name: dict(field.metadata) for field in record if field.metadata} == narrowed
    for name, bounds in narrowed.items():
        edges = [
            *bounds.get("choices", ()),
            *(bounds[key] for key in ("minimum", "maximum") if key in bounds),
        ]
        rejected = [3] if "choices" in bounds else []
        rejected += [bounds["minimum"] - 1] if "minimum" in bounds else []
        rejected += [bounds["maximum"] + 1] if "maximum" in bounds else []
        for value in edges:
            accepted: dict[str, Any] = {name: value}
            assert getattr(_tuning.SearchTuning(**accepted), name) == value
        for value in rejected:
            change: dict[str, Any] = {name: value}
            with pytest.raises(ValueError, match=name):
                _tuning.SearchTuning(**change)
    pytest.raises(TypeError, _tuning.use_tuning(no_such_field=1).__enter__)


def test_tunings_change_routes_but_not_results(costly_stages: CostlyStages) -> None:
    """Exact distances and counts are the same under tunings that take different routes.

    By default, and with every threshold below at ``MAX_VALUE``, the distance searches of this code
    enumerate every word.  With every threshold below at zero, the searches take a family found by
    matroid partitioning instead.  These searches never scan one information set at a time, so the
    tests of ``_search/brouwer_zimmermann_test.py`` force completion tables instead.
    """
    code = get_random_css_code(np.random.default_rng(2), 24, 8, 8)
    eager = {
        "allow_enumeration_by_size": False,
        "allow_enumeration_by_work": False,
        "overlapping_minimum_work": 0,
        "partition_work_ratio": 0,
        "lp_minimum_work": 0,
        "automorphism_minimum_layer_work": 0,
        "scan_chunk_size": 1,
        "scan_chunks_per_call": 1,
        "macwilliams_shared_walk_rank": 0,
    }
    reluctant = {
        name: _tuning.MAX_VALUE
        for name in eager
        if name not in ("allow_enumeration_by_size", "allow_enumeration_by_work")
    }
    results, partitioned = [], []
    for changes in ({}, eager, reluctant):
        costly_stages.partitioned_families.clear()
        with _tuning.use_tuning(**changes):
            # New codes and counters, since codes and counters remember their distances.
            fresh = codes.CSSCode(code.matrix_x, code.matrix_z)
            distances = (fresh.get_distance_exact(Pauli.X), fresh.get_distance_exact(Pauli.Z))
            counter_code = codes.CSSCode(code.matrix_x, code.matrix_z)
            counts = lc.CSSLogicalCounter(counter_code).count_logical_ops(count_pauli=True)
            macwilliams = lc.CSSLogicalCounter(counter_code).count_logical_ops(
                counts.weight, method="macwilliams"
            )
        results.append((distances, counts, macwilliams))
        partitioned.append(
            any(
                len(family.ranks) and family.max_coverage > 1
                for family in costly_stages.partitioned_families
            )
        )
    assert results[0] == results[1] == results[2]
    assert partitioned == [False, True, False]


def test_the_tuning_page_documents_every_field_and_constant() -> None:
    """The tuning page names every field and constant, and renders each constant.

    Each field and constant has its own ``#:`` comment, since Sphinx gives a comment above several
    names only to the first of them, and renders the docstring of ``int`` or ``float`` for the
    others.
    """
    root = pathlib.Path(__file__).parents[4]
    page = (root / "docs" / "source" / "exact_search_tuning.rst").read_text()
    route_map = page.split("Routes and the names that control them")[1].split("\nSettings\n")[0]
    fields = {field.name for field in dataclasses.fields(_tuning.SearchTuning)}
    constants = {name for name in vars(_tuning) if re.fullmatch(r"[A-Z][A-Z0-9_]*", name)}
    constants -= {"DEFAULT_TUNING"}

    assert set(re.findall(r"^\.\. autodata:: (\w+)$", page, flags=re.MULTILINE)) == constants
    assert set(re.findall(r":attr:`SearchTuning\.(\w+)`", route_map)) == fields
    assert set(re.findall(r":data:`(\w+)`", route_map)) == constants - {"MAX_VALUE"}

    source = pathlib.Path(_tuning.__file__).read_text()
    lines = source.splitlines()
    # The statements of the module, outside functions and classes, including nested ones.
    tree = ast.parse(source)
    scoped = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    inner = {
        id(node)
        for scope in ast.walk(tree)
        if isinstance(scope, scoped)
        for node in ast.walk(scope)
    }
    statements = [
        node for node in ast.walk(tree) if isinstance(node, ast.stmt) and id(node) not in inner
    ]
    lines_of_statements = collections.Counter(node.lineno for node in statements)

    # Each constant has one assignment, a statement of the module itself on a line of its own,
    # directly below its own comment.
    documented = []
    for node in statements:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            targets = [node.target]
        else:
            continue
        names = [
            name.id for target in targets for name in ast.walk(target) if isinstance(name, ast.Name)
        ]
        if constants.intersection(names):
            assert node in tree.body and len(names) == 1, names
            assert lines_of_statements[node.lineno] == 1, names
            assert lines[node.lineno - 2].startswith("#: "), names
            documented.append(names[0])
    assert sorted(documented) == sorted(constants)

    # Each field has one declaration in the record, directly below its own comment.
    (record,) = [
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SearchTuning"
    ]
    declared = []
    for node in record.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assert lines[node.lineno - 2].lstrip().startswith("#: "), node.target.id
            declared.append(node.target.id)
    assert sorted(declared) == sorted(fields)


def test_every_field_reaches_the_code_that_reads_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Distance searches and counts pass the values of their tuning to the code that uses them.

    Tuning keeps results, so recording the arguments of each reader is what shows that a value
    arrives.  The tuning takes every route, with values that differ from the defaults.  Fields that
    this test does not follow are read in one of two ways.  Some are read within one function from
    the tuning that it receives, and their tests in ``_search/completion_bounds_test.py`` and
    ``_search/information_sets_test.py`` change them there.  Others are read once by
    ``CosetLeaderSearch``, which passes their values down to its coset searches:
    ``max_css_single_pairs``, ``max_sparsifying_rounds``, and
    ``sparsifying_rounds_without_improvement``.  Tests in ``_search/coset_leader_search_test.py``
    change or record these, and the fields that the oracle passes down to the routes of its BZ
    searches.
    """
    seen: dict[str, list[Any]] = collections.defaultdict(list)

    def record(owner: Any, name: str, *parameters: str, prefix: str = "") -> None:
        original = getattr(owner, name)
        signature = inspect.signature(original)

        def recording(*args: Any, **kwargs: Any) -> Any:
            arguments = signature.bind(*args, **kwargs).arguments
            for parameter in parameters:
                seen[prefix + parameter].append(arguments[parameter])
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, recording)

    record(bz, "enumeration_is_cheaper_than_setup", "max_dimension", "setup_cost_factor")
    record(information_sets, "_get_disjoint_sets", "random_column_orders")
    record(information_sets, "get_overlapping_family", "attempts_per_set")
    record(information_sets, "get_partitioned_family", "tuning")
    record(information_sets.BasisFamily, "get_search_certificate", "lp_max_denominator")
    record(bz, "get_nested_distance_brute_force", "batch_exponent", prefix="bz ")
    record(core, "get_distance_quantum_brute_force", "batch_exponent", prefix="classical ")
    record(core, "get_nested_distance_brute_force", "batch_exponent", prefix="quantum ")
    record(lc, "get_nested_distance_brute_force", "batch_exponent", prefix="count ")
    record(lc.CSSLogicalCounter, "_count_with_macwilliams", "shared_walk_rank", "chunk_size")
    record(scan, "_scan_coefficient_layers", "chunk_size", "max_chunks")
    record(connected_support.SupportSearch, "__init__", "automorphism_minimum_states")
    original_prepare = bz._prepare_brouwer_zimmermann_search

    def prepare(*args: Any, **kwargs: Any) -> Any:
        result = original_prepare(*args, **kwargs)
        if isinstance(result, bz._BrouwerZimmermannSearch):
            seen["search"].append((result.batch_size, result.tuning))
        return result

    monkeypatch.setattr(bz, "_prepare_brouwer_zimmermann_search", prepare)

    code = get_random_css_code(np.random.default_rng(2), 24, 8, 8)
    with _tuning.use_tuning(
        allow_enumeration_by_work=False,
        enumeration_max_dimension=3,
        enumeration_setup_cost_factor=1,
        disjoint_random_column_orders=5,
        overlapping_minimum_work=0,
        overlapping_attempts_per_set=3,
        partition_work_ratio=0,
        max_partition_coverage=3,
        partition_cost_factor=7,
        lp_minimum_work=0,
        lp_dual_max_denominator=97,
        enumeration_batch_exponent=6,
        scan_chunk_size=1001,
        scan_chunks_per_call=3,
        macwilliams_chunk_size=1000,
        macwilliams_shared_walk_rank=5,
        automorphism_minimum_states=7,
    ) as tuning:
        codes.CSSCode(code.matrix_x, code.matrix_z).get_distance_exact(Pauli.X)
        codes.CSSCode(code.matrix_x, code.matrix_z).get_distance_exact(method="connected_support")
        counter = lc.CSSLogicalCounter(codes.CSSCode(code.matrix_x, code.matrix_z))
        counter.count_logical_ops()
        counter.count_logical_ops(method="brute_force")
        lc.CSSLogicalCounter(code).count_logical_ops(method="macwilliams")
        # A basis of one row is small enough to enumerate under either method.
        core.get_distance_classical([[1, 1, 1, 0]])
        core.get_distance_classical([[1, 1, 1, 0]], method="brute_force")
        core.get_distance_quantum([[1, 1, 0]], [[0, 1, 1]], homogeneous=True, method="brute_force")

    expected = {
        "max_dimension": {3},
        "setup_cost_factor": {1},
        "random_column_orders": {5},
        "attempts_per_set": {3},
        "tuning": {tuning},
        "lp_max_denominator": {97},
        "bz batch_exponent": {6},
        "classical batch_exponent": {6},
        "quantum batch_exponent": {6},
        "count batch_exponent": {6},
        "shared_walk_rank": {5},
        "chunk_size": {1000, 1001},
        "max_chunks": {3},
        "automorphism_minimum_states": {7},
    }
    assert {key: set(values) for key, values in seen.items()} == expected

    # A search that scans layers, rather than finish while it prepares, scans with batches of
    # 2**information_set_batch_exponent supports, and passes its tuning to its choices of completion
    # tables.
    bb_code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    with _tuning.use_tuning(information_set_batch_exponent=5) as tuning:
        bz.get_distance_brouwer_zimmermann_many([get_sector_problem(bb_code, Pauli.X)], cutoff=0)
    assert seen["search"] and all(search == (1 << 5, tuning) for search in seen["search"])
