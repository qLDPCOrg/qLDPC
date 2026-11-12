# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tables, tails, and stopping steps of Brouwer-Zimmermann coset searches."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pytest

from qldpc._binary import get_kernel

from .. import _cosets
from .._information_sets import certificates as certificates_module
from .._information_sets import completion_bounds
from ..conftest import (
    _RANK_DEFICIENT_CODE,
)
from . import affine_instance, coset_tables


def test_cached_stopping_steps_are_reused_exactly_while_they_stay_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """For nonnegative weights, a cached stopping step is reused exactly while it stays first.

    The stopping step of a maximum weight is the first step whose bound exceeds it.
    ``_get_stopping_step`` reuses the step that a planning buffer caches if that step is the
    stopping step.  Otherwise, it plans the stopping step, which a binary search finds, and caches
    it.  A reuse is observable only as a skipped plan, so the test records the plans.

    Each step of the bounds is planted as the cached step.  A planted step may be stale, or pair the
    layer of one step with the number of completed sets of another, as interleaved queries might
    leave it.  Each maximum weight from zero to past the largest finite bound is then queried twice.
    The first query plans exactly if the planted step is not its stopping step.  The second query
    does not plan, since the first cached its step.  Both queries return the stopping step, which a
    linear search finds, and its bound.  The bounds in the buffer never change.  A negative maximum
    weight always plans, since every bound is at least zero, so it is left out.

    The bounds are those of the empty code, of a rank-one code with 64 sets, and of sets of ranks
    zero, two, and four in a code of dimension four, for both weight factors.
    """
    plans: list[int] = []
    find_stopping_step = certificates_module.find_stopping_step

    def record(*args: Any) -> tuple[int, int]:
        plans.append(args[1])
        return find_stopping_step(*args)

    monkeypatch.setattr(certificates_module, "find_stopping_step", record)
    # Take the Python version even when the function is compiled, since only it calls the recording
    # function.
    function = coset_tables._get_stopping_step
    get_stopping_step = getattr(function, "py_func", function)
    certificates = [
        certificates_module.Certificate(0, (0,)),
        certificates_module.Certificate(1, (1,) * 64),
        certificates_module.Certificate(4, (0, 2, 4, 0)),
    ]
    for certificate, factor in itertools.product(certificates, (1, 2)):
        bounds = certificate.get_weight_bounds(factor)
        num_rows, num_sets = bounds.shape[0] - 1, bounds.shape[1] - 1
        steps = list(itertools.product(range(num_rows + 1), range(1, num_sets + 1)))
        largest = int(bounds[bounds < certificates_module.NO_WORD_BOUND].max())
        (table,) = coset_tables._get_planning([bounds])
        for planted, max_weight in itertools.product(steps, range(largest + 2)):
            stopping_step = next(step for step in steps if bounds[step] > max_weight)
            expected = (*stopping_step, int(bounds[stopping_step]))
            # Plant the cached step in the last row of the table (see ``_get_planning``).
            table[-1, :2] = planted
            plans.clear()
            assert get_stopping_step(table, max_weight) == expected
            assert len(plans) == (planted != stopping_step)
            assert get_stopping_step(table, max_weight) == expected
            assert len(plans) == (planted != stopping_step)
        assert np.array_equal(table[:-1], bounds)


def test_records_of_tails_add_the_tables_of_the_sets_that_fit() -> None:
    """A record of tails gains the tables of the given sets that fit its limit, once for each set.

    The rank-deficient code has disjoint information sets of ranks five and four, whose tails of
    five and six columns have tables of 128 and 256 bytes on blocks of eight columns.  Under a limit
    of 128 bytes, only the first set gets tables, in the layout of the record, and a second request
    for both sets keeps the record.  Under a larger limit, the second set gets its tables too in a
    new record, and the first set keeps its own, while the earlier record stays as it was.  A record
    built under that limit gets the tables of both sets at once.  Each record holds the tables of
    each set once.
    """
    checks = get_kernel(_RANK_DEFICIENT_CODE)
    instance = affine_instance.get_coset_instance(
        _RANK_DEFICIENT_CODE, random_column_orders=0, weight_factor=1
    )
    assert instance.ranks.tolist() == [5, 4]
    inverse = _cosets._get_inverse(checks)

    def get_record(
        sets: list[int], record: coset_tables._CosetTails | None, max_table_bytes: int
    ) -> coset_tables._CosetTails:
        """The record of the tails of the family, with the tables of the sets that fit."""
        return coset_tables._get_coset_tails(
            instance, sets, record, inverse=inverse, block_bits=8, max_table_bytes=max_table_bytes
        )

    def get_set_tables(record: coset_tables._CosetTails, set_index: int) -> list[int]:
        """The tables of a set, as its layout places them in the record."""
        start, num_blocks = record.layout[set_index, :2]
        offsets = record.layout[set_index, 2 : num_blocks + 3]
        size = (len(_RANK_DEFICIENT_CODE) - 1) * offsets[-1] if num_blocks else 0
        assert start + size <= len(record.completion_tables)
        return record.completion_tables[start : start + size].tolist()

    expected = [
        completion_bounds.CompletionTables.build(tails, 8)
        for tails in get_record([], None, 0).tails
    ]
    first = get_record([0, 1], None, 128)
    assert get_set_tables(first, 0) == expected[0].tables.tolist()
    assert get_set_tables(first, 1) == []
    assert get_record([0, 1], first, 128) is first
    for both in (get_record([1], first, 256), get_record([0, 1], None, 256)):
        assert [get_set_tables(both, index) for index in range(2)] == [
            tables.tables.tolist() for tables in expected
        ]
        assert len(both.completion_tables) == sum(len(tables.tables) for tables in expected)
    assert get_set_tables(first, 1) == []
