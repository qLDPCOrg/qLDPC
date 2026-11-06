# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tuning of exact searches."""

from __future__ import annotations

import concurrent.futures
import contextlib
import contextvars
import sys
from typing import Any

import numpy as np
import pytest

from . import _tuning


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
    with pytest.raises(TypeError), contextlib.ExitStack() as stack:
        stack.enter_context(_tuning.use_tuning(no_such_field=1))
