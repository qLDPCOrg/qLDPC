# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the tables, tails, and stopping steps of Brouwer-Zimmermann coset searches."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from qldpc._binary import get_kernel, unpack_bits

from .. import _cosets
from .._information_sets import certificates, completion_bounds
from ..conftest import EVEN_ROUTES_CODE, RANK_DEFICIENT_CODE, TWO_FAMILIES_CODE
from . import affine_instance, coset_tables


def test_cached_stopping_steps_are_reused_exactly_while_they_stay_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cached step serves a nonnegative maximum weight exactly if it is its stopping step.

    The extreme maximum weights, a negative one and the largest native integer, give the first and
    the last steps.
    """
    plans: list[int] = []
    find_stopping_step = certificates.find_stopping_step

    def record(*args: Any) -> tuple[int, int]:
        plans.append(args[1])
        return find_stopping_step(*args)

    monkeypatch.setattr(certificates, "find_stopping_step", record)
    # Take the plain-Python ``get_stopping_step``, since only its plans reach the test's recorder; a
    # reuse then shows as a skipped plan.
    function = coset_tables.get_stopping_step
    get_stopping_step = getattr(function, "py_func", function)
    certificate_cases = [
        certificates.Certificate(0, (0,)),
        certificates.Certificate(1, (1,) * 64),
        certificates.Certificate(4, (0, 2, 4, 0)),
    ]
    for certificate, factor in itertools.product(certificate_cases, (1, 2)):
        bounds = certificate.get_weight_bounds(factor)
        num_rows, num_sets = bounds.shape[0] - 1, bounds.shape[1] - 1
        steps = list(itertools.product(range(num_rows + 1), range(1, num_sets + 1)))
        largest = int(bounds[bounds < certificates.NO_WORD_BOUND].max())
        (table,) = coset_tables._get_planning([bounds])
        for planted, max_weight in itertools.product(steps, range(largest + 2)):
            stopping_step = next(step for step in steps if bounds[step] > max_weight)
            expected = (*stopping_step, int(bounds[stopping_step]))
            # Plant each step as the cached one, including stale steps, and pairs that mix the layer
            # of one step and the number of completed sets of another, as interleaved queries can
            # leave (Section 4.1 of ``brouwer_zimmermann_code.md``).
            table[-1, :2] = planted
            plans.clear()
            assert get_stopping_step(table, max_weight) == expected
            assert len(plans) == (planted != stopping_step)
            assert get_stopping_step(table, max_weight) == expected
            assert len(plans) == (planted != stopping_step)
        # A negative maximum weight gives the first step, and the largest native integer, which no
        # bound exceeds, gives the last, whatever step is cached.
        extremes = [(-1, steps[0]), (int(np.iinfo(np.int64).max), steps[-1])]
        for planted, (max_weight, step) in itertools.product(steps, extremes):
            table[-1, :2] = planted
            assert get_stopping_step(table, max_weight) == (*step, int(bounds[step]))
        assert np.array_equal(table[:-1], bounds)


def test_records_of_tails_add_the_tables_of_the_sets_that_fit() -> None:
    """A record of tails gains the tables of the given sets that fit its limit, each once."""
    checks = get_kernel(RANK_DEFICIENT_CODE)
    instance = affine_instance.get_coset_instance(
        RANK_DEFICIENT_CODE, random_column_orders=0, weight_factor=1
    )
    assert instance.ranks.tolist() == [5, 4]
    inverse = _cosets.get_unit_syndrome_words(checks)

    def get_record(
        sets: list[int], record: coset_tables.CosetTails | None, max_table_bytes: int
    ) -> coset_tables.CosetTails:
        """The record of the tails of the family, with the tables of the sets that fit."""
        return coset_tables.get_coset_tails(
            instance, sets, record, inverse=inverse, block_bits=8, max_table_bytes=max_table_bytes
        )

    def get_set_tables(record: coset_tables.CosetTails, set_index: int) -> list[int]:
        """The tables of a set, as its layout places them in the record."""
        start, num_blocks = record.layout[set_index, :2]
        offsets = record.layout[set_index, 2 : num_blocks + 3]
        size = (len(RANK_DEFICIENT_CODE) - 1) * offsets[-1] if num_blocks else 0
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
    assert [get_set_tables(first, index) for index in range(2)] == [expected[0].tables.tolist(), []]


@pytest.mark.parametrize("code", [RANK_DEFICIENT_CODE, TWO_FAMILIES_CODE])
def test_records_of_tails_hold_the_offset_tails_of_unit_syndromes(
    code: npt.NDArray[np.uint8],
) -> None:
    """A record of tails holds the rank of each set, and the offset tails of unit syndrome words."""
    num_columns = code.shape[1]
    inverse = _cosets.get_unit_syndrome_words(get_kernel(code))
    instance = affine_instance.get_coset_instance(code, random_column_orders=0, weight_factor=1)
    record = coset_tables.get_coset_tails(
        instance, [], None, inverse=inverse, block_bits=8, max_table_bytes=0
    )
    tail_bits = 64 * record.inverse_tails.shape[2]
    for set_index, (packed_rows, pivots) in enumerate(
        zip(instance.generators, instance.pivots, strict=True)
    ):
        # The rows of a set with a pivot come first, and reduce a word to its offset.
        rank = int(np.count_nonzero(pivots >= 0))
        assert record.ranks[set_index] == rank
        rows = unpack_bits(packed_rows[:rank], num_columns)
        span = [np.array(bits) @ rows % 2 for bits in itertools.product((0, 1), repeat=rank)]
        tail_columns = np.delete(np.arange(num_columns), pivots[:rank])
        for word, tail in zip(inverse, record.inverse_tails[set_index], strict=True):
            # The offset is the one word of ``word`` plus the span of the set that is zero on the
            # pivots of the set.
            offsets = [(word + other) % 2 for other in span]
            (offset,) = [offset for offset in offsets if not offset[pivots[:rank]].any()]
            expected = np.zeros(tail_bits, dtype=np.uint8)
            expected[: len(tail_columns)] = offset[tail_columns]
            np.testing.assert_array_equal(unpack_bits(tail[None], tail_bits)[0], expected)


def test_search_tables_plan_by_parity_exactly_when_cosets_of_an_even_code_differ() -> None:
    """A search plans with bounds by parity exactly if its code is even and some coset is odd."""
    even_inverse = _cosets.get_unit_syndrome_words(get_kernel(EVEN_ROUTES_CODE))
    assert np.any(even_inverse.sum(axis=1) % 2)
    rank_deficient_inverse = _cosets.get_unit_syndrome_words(get_kernel(RANK_DEFICIENT_CODE))
    assert np.any(RANK_DEFICIENT_CODE.sum(axis=1) % 2)
    for code, inverse, parities in [
        (EVEN_ROUTES_CODE, even_inverse, (0, 1)),
        (EVEN_ROUTES_CODE, np.zeros_like(even_inverse), (None,)),
        (RANK_DEFICIENT_CODE, rank_deficient_inverse, (None,)),
    ]:
        instance = affine_instance.get_coset_instance(code, random_column_orders=0, weight_factor=1)
        tables = coset_tables.get_search_tables(instance, inverse, 0)
        expected = [instance.get_weight_bounds(parity).tolist() for parity in parities]
        assert [table[:-1].tolist() for table in tables.planning] == expected
        for parity in (0, 1):
            table = coset_tables.get_planning_table(tables.planning, parity)
            assert table[:-1].tolist() == expected[parity if len(expected) == 2 else 0]


def test_first_steps_reaching_a_target_follow_their_definition() -> None:
    """The first step whose bound reaches a target is the first step whose bound is at least it."""
    certificate_cases = [
        certificates.Certificate(1, (1,) * 64),
        certificates.Certificate(4, (0, 2, 4, 0)),
    ]
    for certificate, factor in itertools.product(certificate_cases, (1, 2)):
        bounds = certificate.get_weight_bounds(factor)
        steps = list(itertools.product(range(len(bounds)), range(1, bounds.shape[1])))
        largest = int(bounds[bounds < certificates.NO_WORD_BOUND].max())
        for target in range(largest + 2):
            expected = next(step for step in steps if bounds[step] >= target)
            assert coset_tables.get_first_step_reaching(bounds, target) == expected
