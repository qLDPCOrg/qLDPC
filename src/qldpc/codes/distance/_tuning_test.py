# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tuning of exact searches."""

from __future__ import annotations

import concurrent.futures
import contextvars
import sys
from typing import Any

import numpy as np
import pytest

from qldpc import codes
from qldpc.objects import Pauli

from . import _tuning
from . import logical_counts as lc
from .conftest import CostlyStages, get_random_css_code


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
    """Fields hold Python integers, and a tuning rejects wrong types and out-of-range values."""
    numpy_fields: dict[str, Any] = {"max_information_sets": np.int64(4)}
    assert type(_tuning.SearchTuning(**numpy_fields).max_information_sets) is int
    assert _tuning.SearchTuning(lp_minimum_work=_tuning.MAX_VALUE).lp_minimum_work == 2**62

    invalid: list[tuple[type[Exception], str, dict[str, Any]]] = [
        (TypeError, "allow_enumeration_by_work must be a bool", {"allow_enumeration_by_work": 1}),
        (TypeError, "must be an integer", {"max_information_sets": True}),
        (TypeError, "must be an integer", {"lp_minimum_work": 1.5}),
        (ValueError, "between 0 and", {"overlapping_minimum_work": -1}),
        (ValueError, "between 0 and", {"lp_minimum_work": _tuning.MAX_VALUE + 1}),
        (ValueError, "between 1 and", {"scan_chunk_size": 0}),
    ]
    for error, message, fields in invalid:
        with pytest.raises(error, match=message):
            _tuning.SearchTuning(**fields)
    pytest.raises(TypeError, _tuning.use_tuning(no_such_field=1).__enter__)


def test_tunings_change_routes_but_not_results(costly_stages: CostlyStages) -> None:
    """Exact distances and counts are the same under tunings that take different routes.

    By default, and with every threshold at ``MAX_VALUE``, the distance searches of this code
    enumerate every word.  With every threshold at zero, the searches take a family found by matroid
    partitioning instead.
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
