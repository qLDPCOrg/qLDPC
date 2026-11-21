# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Brouwer-Zimmermann coset searches of ``CosetLeaderSearch``."""

from __future__ import annotations

import inspect
import itertools
import threading
from collections.abc import Sequence
from math import comb
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from qldpc import math
from qldpc._binary import get_kernel, pack_bits, row_reduce_binary, unpack_bits

from ...conftest import Tune
from .. import _cosets, _threads
from .._code_matrices import symplectic_to_hamming
from .._information_sets import basis_families, certificates, completion_bounds
from .._tuning import DEFAULT_TUNING, MAX_VALUE, get_tuning, use_tuning
from ..conftest import (
    EVEN_ROUTES_CODE,
    OVERLAPPING_ROUTES_CODE,
    RANK_DEFICIENT_CODE,
    ROUTED_CSS_WEIGHT_STATE,
    ROUTED_PHYSICAL_STATES,
    THREAD_TIMEOUT,
    TWO_FAMILIES_CODE,
    assert_exact_answers,
    get_binary_matrix,
    get_binary_minima,
    get_binary_syndrome,
    get_exhaustive_minima,
    get_graph_state,
    get_random_full_rank_matrix,
    get_set_ranks,
    get_stabilizer_matrix,
    get_syndrome,
    scramble,
    shared_arrays_read_only,
    split_syndromes_into_words,
)
from ..coset_leaders import CosetLeaderSearch
from ..min_weight_bounds import MinWeightBound
from . import affine_instance, coset, coset_tables

# A code whose searches for minimum weights by tails skip completions by tables at the threshold
# zero, as do those of its even version.
_SKIPPING_CODE = np.array(
    [
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
    ],
    dtype=np.uint8,
)

# A longer code with cosets of minimum weight six.  At maximum weight four and the threshold zero,
# its searches of these cosets for minimum weights skip completions whose bound reaches the skip
# bound but not the weight of the lightest word weighed.  Its searches by tails of some other
# cosets, such as those of syndromes 1126, 4578, and 6129, stop early within steps.
_HEAVY_COSETS_CODE = get_binary_matrix(
    [
        "11100011110000110001",
        "00001011001010001000",
        "10010110101010101100",
        "10001100101100100101",
        "01010100110001001100",
        "00010000001110010101",
        "00101100011111000001",
    ]
)


def test_css_weight_searches_of_even_stabilizers_stop_at_bounds_of_each_class_parity() -> None:
    """Searches by CSS weight stop at bounds of each class, rounded to the parity of |x| + |z|."""
    generators = ("Y______", "_XZ____", "_ZX__ZZ", "___Y___", "____Y__", "__Z__YZ", "__Z__ZY")
    matrix = get_stabilizer_matrix(generators)
    assert_exact_answers(CosetLeaderSearch(matrix, css_weight=True), matrix, True)

    # Read the parity of |x| + |z| in each syndrome class from every error, and check that the
    # class shares it.
    num_qubits = len(matrix)
    checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
    errors = (np.arange(1 << 2 * num_qubits)[:, None] >> np.arange(2 * num_qubits)) & 1
    syndromes, parities = get_binary_syndrome(checks, errors), errors.sum(axis=1) % 2
    class_parities = np.zeros(1 << num_qubits, dtype=int)
    class_parities[syndromes] = parities
    assert np.array_equal(class_parities[syndromes], parities)

    # Answers alone cannot show rounding, which mainly moves where an odd class stops, so find the
    # stopping step of each query as ``_search_coset`` does.  Steps compare as tuples in the order
    # of the search.
    instance = affine_instance.get_coset_instance(matrix, random_column_orders=0, weight_factor=2)
    inverse = _cosets.get_unit_syndrome_words(checks)
    tables = coset_tables.get_search_tables(instance, inverse, num_qubits)
    unrounded = instance.get_weight_bounds()
    steps = list(itertools.product(range(len(unrounded)), range(1, unrounded.shape[1])))
    best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
    num_sooner = 0
    for syndrome in range(1 << num_qubits):
        _, parity = coset._get_syndrome_word(
            split_syndromes_into_words([syndrome], num_qubits)[0],
            best,
            tables.inverse,
            tables.x_mask,
            0,
        )
        table = coset_tables.get_planning_table(tables.planning, parity)
        bounds = instance.get_weight_bounds(class_parities[syndrome])
        for max_weight in range(2, num_qubits):
            stopping_step = next(step for step in steps if bounds[step] > max_weight)
            assert coset_tables.get_stopping_step(table, max_weight)[:2] == stopping_step
            num_sooner += stopping_step < next(
                step for step in steps if unrounded[step] > max_weight
            )
    assert num_sooner


def test_cached_searches_answer_only_for_their_stabilizer_group(
    pytestconfig: pytest.Config,
) -> None:
    """Searches built in turn for |000+++>, another generating set of it, and |+++000> are exact."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    eye, zeros = np.eye(3, dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)
    zero_plus = np.block([[zeros, eye, zeros, zeros], [zeros, zeros, eye, zeros]])
    plus_zero = np.block([[eye, zeros, zeros, zeros], [zeros, zeros, zeros, eye]])
    matrices = [scramble(matrix, rng) for matrix in (zero_plus, zero_plus, plus_zero)]
    coset._get_css_searches.cache_clear()
    searches = [CosetLeaderSearch(matrix, css_weight=True) for matrix in matrices]
    for search, matrix in zip(searches, matrices, strict=True):
        minima = get_exhaustive_minima(matrix, css_weight=True)
        for syndrome, max_weight in itertools.product(range(1 << 6), range(4)):
            bound = search.get_min_weight_bound(syndrome, max_weight)
            assert bound.within_max_weight == (minima[syndrome] <= max_weight)
            assert get_syndrome(matrix, bound.representative) == syndrome


def test_searches_use_the_tuning_with_which_they_are_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """A coset-leader search builds its coset searches with the tuning in force when it is built."""
    orders: list[int] = []
    get_disjoint_family = basis_families.get_disjoint_family

    def record(*args: Any, random_column_orders: int) -> basis_families.BasisFamily:
        orders.append(random_column_orders)
        return get_disjoint_family(*args, random_column_orders=random_column_orders)

    monkeypatch.setattr(basis_families, "get_disjoint_family", record)
    eye, zeros = np.eye(3, dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)
    zero_plus = np.block([[zeros, eye, zeros, zeros], [zeros, zeros, eye, zeros]])
    phased = get_graph_state(2, [(0, 1)], phase=True)
    coset._get_css_searches.cache_clear()
    coset._get_pauli_search.cache_clear()
    default = DEFAULT_TUNING.disjoint_random_column_orders
    for column_orders, matrix in itertools.product((default, 3), (zero_plus, phased)):
        with use_tuning(disjoint_random_column_orders=column_orders):
            first = CosetLeaderSearch(matrix, css_weight=True)
            assert orders and set(orders) == {column_orders}
            orders.clear()
            second = CosetLeaderSearch(matrix, css_weight=True)
            assert not orders
        for search in (first, second):
            assert_exact_answers(search, matrix, True)
    for matrix, css_weight in ((zero_plus, True), (phased, False), (phased, True)):
        CosetLeaderSearch(matrix, css_weight=css_weight)
        paulis_by_css_weight = matrix is phased and css_weight
        changes: list[tuple[dict[str, Any], bool]] = [
            ({"max_css_single_pairs": 0}, paulis_by_css_weight),
            ({"coset_completion_block_bits": 4}, not paulis_by_css_weight),
            ({"coset_completion_max_table_bytes": 4}, not paulis_by_css_weight),
            ({"coset_route_minimum_work": 0}, True),
            ({"overlapping_minimum_work": 0}, False),
            ({"completion_bound_max_table_bytes": 4}, False),
        ]
        # Only the fields that a search reads key its cache.
        for change, reads in changes:
            orders.clear()
            with use_tuning(**change):
                CosetLeaderSearch(matrix, css_weight=css_weight)
            assert bool(orders) == reads


def test_searches_build_other_families_with_the_tuning_they_are_built_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Costly queries build their families and tables with the tuning of the search that answers."""
    calls: list[tuple[int, int]] = []
    table_calls: set[tuple[int, int]] = set()
    get_overlapping_family = basis_families.get_overlapping_family
    get_table_bytes_within = completion_bounds.get_table_bytes_within
    signature = inspect.signature(get_overlapping_family)

    def record(*args: Any, **kwargs: Any) -> basis_families.BasisFamily:
        policy = signature.bind(*args, **kwargs).arguments["policy"]
        calls.append((policy.max_sets, policy.attempts_per_set))
        return get_overlapping_family(*args, **kwargs)

    def record_tables(rows: npt.NDArray[np.uint64], block_bits: int, max_bytes: int) -> int:
        table_calls.add((block_bits, max_bytes))
        return get_table_bytes_within(rows, block_bits, max_bytes)

    monkeypatch.setattr(basis_families, "get_overlapping_family", record)
    monkeypatch.setattr(completion_bounds, "get_table_bytes_within", record_tables)
    coset._get_pauli_search.cache_clear()
    with use_tuning(
        coset_route_minimum_work=0,
        max_information_sets=5,
        overlapping_attempts_per_set=3,
        coset_completion_block_bits=4,
        coset_completion_max_table_bytes=12345,
        completion_bound_max_table_bytes=54321,
    ):
        search = CosetLeaderSearch(ROUTED_PHYSICAL_STATES[1])
    for syndrome, max_weight in itertools.product(range(1 << 5), range(2, 5)):
        search.get_min_weight_bound(syndrome, max_weight)
    assert calls == [(5, 3)]
    assert table_calls == {(4, 12345)}


def test_searches_under_other_gates_do_not_share_searches(tune: Tune) -> None:
    """A search answers as a fresh search with its gate threshold would, whatever is cached."""
    checks = get_kernel(OVERLAPPING_ROUTES_CODE)
    css_state = np.block(
        [
            [OVERLAPPING_ROUTES_CODE, np.zeros_like(OVERLAPPING_ROUTES_CODE)],
            [np.zeros_like(checks), checks],
        ]
    )
    for matrix in (ROUTED_PHYSICAL_STATES[1], css_state):
        css_weight = matrix is css_state
        syndromes = range(0, 1 << len(matrix), max(1, (1 << len(matrix)) // 64))
        queries = list(itertools.product(syndromes, range(6)))
        answers = {}
        for minimum_work in (0, MAX_VALUE):
            coset._get_css_searches.cache_clear()
            coset._get_pauli_search.cache_clear()
            with use_tuning(coset_route_minimum_work=minimum_work):
                search = CosetLeaderSearch(matrix, css_weight=css_weight)
                answers[minimum_work] = [search.get_min_weight_bound(*query) for query in queries]
        assert answers[0] != answers[MAX_VALUE]
        for minimum_work in (0, MAX_VALUE, 0):
            tune(coset_route_minimum_work=minimum_work)
            search = CosetLeaderSearch(matrix, css_weight=css_weight)
            answers_now = [search.get_min_weight_bound(*query) for query in queries]
            assert answers_now == answers[minimum_work]


@pytest.mark.parametrize(
    ("matrix", "css_weight", "more_sets"),
    [
        (ROUTED_PHYSICAL_STATES[0], False, True),
        (ROUTED_PHYSICAL_STATES[1], False, False),
        (ROUTED_CSS_WEIGHT_STATE, True, False),
    ],
    ids=["physical-0", "physical-1", "css-weight"],
)
def test_routed_states_take_families_that_beat_the_greedy_packing(
    matrix: npt.NDArray[np.uint8], css_weight: bool, more_sets: bool
) -> None:
    """Costly queries of the routed states take disjoint families that beat the greedy packing.

    At the threshold zero, some route has pairwise disjoint sets, and more full information sets
    than the greedy packing, or as many sets and a larger partial set, as ``conftest.py`` says of
    these states.  The routes are by tails exactly when the search counts physical weight.  Tests
    that need other families rely on these states.
    """
    # Get the greedy packing, which a closed gate keeps, and a search whose gate is open.
    with use_tuning(coset_route_minimum_work=MAX_VALUE) as tuning:
        greedy, _ = coset.get_pauli_search(matrix, css_weight, tuning)._get_route(2)
    with use_tuning(coset_route_minimum_work=0) as tuning:
        search = coset.get_pauli_search(matrix, css_weight, tuning)

    # Compare the family of the route of each maximum weight with the greedy packing.
    dimension = greedy.pivots.shape[1]
    greedy_ranks = get_set_ranks(greedy)
    beaten = False
    for max_weight in range(2, len(matrix) + 1):
        tables, tails = search._get_route(max_weight)
        assert (tails is not None) == (not css_weight)
        ranks = get_set_ranks(tables)
        pivots = tables.pivots[tables.pivots >= 0]
        disjoint = len(np.unique(pivots)) == len(pivots)
        if more_sets:
            beaten |= disjoint and ranks.count(dimension) > greedy_ranks.count(dimension)
        else:
            beaten |= (
                disjoint and len(ranks) == len(greedy_ranks) and min(ranks) > min(greedy_ranks)
            )
    assert beaten


def test_cached_pauli_searches_answer_for_their_own_stabilizers_and_weight() -> None:
    """Searches built in turn answer exactly for their own stabilizers and weight."""
    states = [
        get_stabilizer_matrix(generators)
        for generators in (("Y__", "_Z_", "__X"), ("Y__", "YZ_", "__X"), ("X__", "_Y_", "__Z"))
    ]
    coset._get_pauli_search.cache_clear()
    searches = []
    for changes in ({}, {"max_css_single_pairs": 0}, {"disjoint_random_column_orders": 3}):
        with use_tuning(**changes):
            for matrix, css_weight in itertools.product(states, (False, True)):
                searches.append(
                    (matrix, css_weight, CosetLeaderSearch(matrix, css_weight=css_weight))
                )
    physical_bound = searches[0][2].get_min_weight_bound(0b011, 1)
    assert not physical_bound.within_max_weight and physical_bound.lower_bound == 2
    css_bound = searches[1][2].get_min_weight_bound(0b011, 1)
    assert css_bound.within_max_weight and css_bound.representative == 0b001010
    for matrix, css_weight, search in searches:
        assert_exact_answers(search, matrix, css_weight)


def test_cached_pauli_searches_ignore_later_changes_to_the_stabilizer_array(tune: Tune) -> None:
    """Searches answer for the stabilizers they were built from, when an array changes later."""
    # At the threshold zero, the searches build their other families at their first queries, after
    # the array changes.
    tune(coset_route_minimum_work=0)
    first, second = ROUTED_CSS_WEIGHT_STATE, ROUTED_PHYSICAL_STATES[1]
    coset._get_pauli_search.cache_clear()
    array = first.copy()
    searches = [
        (first, css_weight, CosetLeaderSearch(array, css_weight=css_weight))
        for css_weight in (False, True)
    ]
    array[:] = second
    searches += [
        (second, css_weight, CosetLeaderSearch(array, css_weight=css_weight))
        for css_weight in (False, True)
    ]
    searches += [
        (first, css_weight, CosetLeaderSearch(first.copy(), css_weight=css_weight))
        for css_weight in (False, True)
    ]
    for matrix, css_weight, search in searches:
        assert_exact_answers(search, matrix, css_weight)


def test_cosets_of_a_rank_deficient_code_match_brute_force() -> None:
    """The coset searches of a rank-deficient code match the minima of all words."""
    _assert_classical_cosets_match_brute_force(RANK_DEFICIENT_CODE)


def test_cosets_of_an_even_code_have_bounds_of_their_parity() -> None:
    """Searches of an even code give each coset with no light enough word a bound of its parity."""
    parities = (RANK_DEFICIENT_CODE.sum(axis=1, keepdims=True) % 2).astype(np.uint8)
    _assert_classical_cosets_match_brute_force(np.hstack([RANK_DEFICIENT_CODE, parities]))


@pytest.mark.parametrize(
    "code", [EVEN_ROUTES_CODE, TWO_FAMILIES_CODE], ids=["even", "two-families"]
)
def test_cosets_match_brute_force_with_other_families(
    code: npt.NDArray[np.uint8], tune: Tune
) -> None:
    """Searches of X or Z parts stay exact when costly queries take other families."""
    tune(coset_route_minimum_work=0)
    _assert_classical_cosets_match_brute_force(code)


def test_answers_do_not_depend_on_earlier_queries(pytestconfig: pytest.Config, tune: Tune) -> None:
    """The answers of a search depend only on its code, the syndrome, and the maximum weight."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    checks = get_kernel(OVERLAPPING_ROUTES_CODE)

    def get_answers(order: str) -> dict[tuple[int, int], MinWeightBound]:
        """Every answer of a fresh search, with the maximum weights in the given order."""
        tuning = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
        search = coset._get_classical_search(
            OVERLAPPING_ROUTES_CODE, checks, tuning._replace(disjoint_random_column_orders=0)
        )
        weights = list(range(15))
        if order != "up":
            weights = weights[::-1] if order == "down" else rng.permutation(weights).tolist()
        return {
            (syndrome, max_weight): search.get_min_weight_bound(syndrome, max_weight)
            for max_weight in weights
            for syndrome in range(1 << len(checks))
        }

    tune(coset_route_minimum_work=0)
    answers = [get_answers(order) for order in ("up", "down", "random")]
    assert answers[0] == answers[1] == answers[2]
    with use_tuning(coset_route_minimum_work=MAX_VALUE):
        assert get_answers("up") != answers[0]


def test_searches_visit_only_the_sets_that_add_to_their_bounds(pytestconfig: pytest.Config) -> None:
    """A search skips exactly the sets that add nothing to the bound of its stopping step."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(12):
        num_columns = int(rng.integers(2, 9))
        num_rows = int(rng.integers(1, num_columns))
        generators = get_random_full_rank_matrix(rng, num_rows, num_columns)
        checks = get_kernel(generators)
        family = basis_families.get_disjoint_family(generators, None, random_column_orders=0)
        ranks = (*family.ranks.tolist(), 0)
        num_sets = len(ranks)
        rows = np.concatenate([family.generators, generators[None]])
        instance = affine_instance.AffineInstance(
            generators=pack_bits(rows.reshape(-1, num_columns)).reshape(num_sets, num_rows, -1),
            labels=np.zeros((num_sets, num_rows, 0), dtype=np.uint64),
            widths=np.ones(num_sets, dtype=np.int64),
            pivots=np.vstack([family.pivots, np.full(num_rows, -1)]),
            certificate=certificates.Certificate(num_rows, ranks),
        )
        inverse = _cosets.get_unit_syndrome_words(checks)
        tables = coset_tables.get_search_tables(instance, inverse, num_columns)
        minima = get_binary_minima(checks)

        for syndrome, max_weight in itertools.product(
            range(1 << len(checks)), range(num_columns + 1)
        ):
            best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
            packed_syndrome = split_syndromes_into_words([syndrome], len(checks))[0]
            best_weight, parity = coset._get_syndrome_word(
                packed_syndrome, best, tables.inverse, tables.x_mask, 1
            )
            if best_weight <= max_weight:
                continue
            table = coset_tables.get_planning_table(tables.planning, parity)
            stop_layer, stop_num_completed, stop_bound = coset_tables.get_stopping_step(
                table, max_weight
            )
            lower_bound, weighed, _ = coset._search_from_syndrome_word(
                packed_syndrome,
                max_weight,
                stop_layer,
                stop_num_completed,
                stop_bound,
                best,
                best_weight,
                tables.generators,
                tables.pivots,
                tables.x_mask,
                None,
                1,
                coset._get_rounding_parity(tables.planning, 1, parity),
            )
            representative = unpack_bits(best[None], num_columns)[0].astype(int)
            assert get_binary_syndrome(checks, representative) == syndrome
            assert (lower_bound <= max_weight) == (minima[syndrome] <= max_weight)
            if lower_bound > max_weight:
                assert lower_bound <= minima[syndrome]
                # A search whose answer is no weighs every word of the layers of the sets that it
                # visits, and checking that number exactly guards the rule for skipping sets.
                last_layers = [
                    stop_layer if index < stop_num_completed else stop_layer - 1
                    for index in range(num_sets)
                ]
                assert weighed == sum(
                    comb(num_rows, layer)
                    for rank, last_layer in zip(ranks, last_layers, strict=True)
                    if last_layer >= num_rows - rank
                    for layer in range(last_layer + 1)
                )
                binomials = certificates.get_capped_binomials(num_rows)
                plan = certificates.plan_steps(
                    table[:-1], max_weight, 0, num_rows, instance.ranks, binomials, True
                )
                assert plan == (stop_layer, stop_num_completed, weighed)


@pytest.mark.parametrize("block_bits", [None, 4])
def test_searches_by_tails_answer_as_searches_of_whole_words(
    block_bits: int | None, pytestconfig: pytest.Config
) -> None:
    """Coset searches by tails decide as searches of whole words, and match them without tables."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    # The arguments of each check, and whether its searches with tables must skip words.
    searches: list[tuple[tuple[Any, ...], bool]] = []
    for _ in range(3):
        num_rows, num_columns = int(rng.integers(3, 5)), int(rng.integers(6, 10))
        generators = get_random_full_rank_matrix(rng, num_rows, num_columns)
        code = np.insert(generators, int(rng.integers(num_columns + 1)), 0, axis=1)
        codes = [code, np.hstack([code, code.sum(axis=1, keepdims=True) % 2]).astype(np.uint8)]
        for code in codes if block_bits else codes[:1]:
            checks = get_kernel(code)
            searches.append(
                (
                    (
                        code,
                        _cosets.get_unit_syndrome_words(checks),
                        1,
                        (num_rows, num_rows - 1),
                        range(-1, code.shape[1] + 2),
                        range(1 << len(checks)),
                        get_binary_minima(checks),
                    ),
                    False,
                )
            )
    if block_bits:
        parities = (RANK_DEFICIENT_CODE.sum(axis=1, keepdims=True) % 2).astype(np.uint8)
        even = np.hstack([RANK_DEFICIENT_CODE, parities])
        checks = get_kernel(even)
        searches.append(
            (
                (
                    even,
                    _cosets.get_unit_syndrome_words(checks),
                    1,
                    (5, 4),
                    range(-1, even.shape[1] + 2),
                    range(1 << len(checks)),
                    get_binary_minima(checks),
                ),
                True,
            )
        )

    wide = np.hstack([np.eye(4, dtype=np.uint8), rng.integers(0, 2, (4, 64), dtype=np.uint8)])
    checks = get_kernel(wide)
    syndromes = [int.from_bytes(rng.bytes(9), "little") % (1 << len(checks)) for _ in range(16)]
    searches.append(
        (
            (
                wide,
                _cosets.get_unit_syndrome_words(checks),
                1,
                (4, 3),
                range(-1, 40, 4),
                syndromes,
            ),
            True,
        )
    )

    matrix = get_graph_state(4, [(0, 1), (1, 2)], phase=True)
    checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
    searches.append(
        (
            (
                symplectic_to_hamming(matrix),
                symplectic_to_hamming(_cosets.get_unit_syndrome_words(checks)),
                2,
                (2, 1) if block_bits else (4, 3),
                range(-1, 6),
                range(16),
                get_exhaustive_minima(matrix, css_weight=False),
            ),
            True,
        )
    )
    for arguments, must_skip in searches:
        skipped = _assert_searches_by_tails_match_searches_of_words(
            *arguments, block_bits=block_bits
        )
        assert skipped or not (block_bits and must_skip)


def test_queries_of_routes_by_tails_weigh_the_tails_of_their_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single queries and batches of routes by tails weigh words by the tails of their record."""
    get_coset_tails = coset_tables.get_coset_tails

    def get_zero_tails(*args: Any, **kwargs: Any) -> coset_tables.CosetTails:
        """The record of the tails of a family, with every tail zero."""
        record = get_coset_tails(*args, **kwargs)
        return record._replace(
            tails=np.zeros_like(record.tails), inverse_tails=np.zeros_like(record.inverse_tails)
        )

    monkeypatch.setattr(coset_tables, "get_coset_tails", get_zero_tails)
    generators = np.ones((1, 7), dtype=np.uint8)
    checks = get_kernel(generators)
    tuning = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    search = coset._get_classical_search(
        generators, checks, tuning._replace(coset_route_minimum_work=0)
    )
    syndromes = np.flatnonzero(get_binary_minima(checks) > 2).tolist()
    assert syndromes
    for syndrome in syndromes:
        bound = search.get_min_weight_bound(syndrome, 2)
        assert (bound.within_max_weight, bound.lower_bound) == (True, 2)
    packed_syndromes = split_syndromes_into_words(syndromes, len(checks))
    assert search.find_first_not_within_max_weight(packed_syndromes, 2) == (len(syndromes), None)


def test_queries_of_routes_by_tails_skip_completions_by_the_tables_of_their_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queries of routes by tails skip completions by the tables of their record."""
    get_coset_tails = coset_tables.get_coset_tails

    def get_skipping_tails(*args: Any, **kwargs: Any) -> coset_tables.CosetTails:
        """The record of the tails of a family, with tables that skip every completion."""
        record = get_coset_tails(*args, **kwargs)
        return record._replace(completion_tables=np.full_like(record.completion_tables, 255))

    monkeypatch.setattr(coset_tables, "get_coset_tails", get_skipping_tails)
    checks = get_kernel(OVERLAPPING_ROUTES_CODE)
    tuning = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    search = coset._get_classical_search(
        OVERLAPPING_ROUTES_CODE, checks, tuning._replace(coset_route_minimum_work=0)
    )
    minima = get_binary_minima(checks)
    missed = [
        syndrome
        for syndrome in np.flatnonzero(minima <= 3).tolist()
        if not search.get_min_weight_bound(syndrome, 3).within_max_weight
    ]
    assert missed
    assert {search.get_min_weight_bound(syndrome, 3).lower_bound for syndrome in missed} == {4}
    _, bound = search.find_first_not_within_max_weight(
        split_syndromes_into_words(missed, len(checks)), 3
    )
    assert bound is not None and bound.lower_bound == 4


@pytest.mark.parametrize(
    "tuning",
    [
        {},
        {"coset_route_minimum_work": 0},
        {"coset_route_minimum_work": 0, "coset_completion_max_table_bytes": 0},
    ],
    ids=["disjoint", "tables", "no-tables"],
)
def test_min_weights_of_cosets_match_brute_force(
    tuning: dict[str, int],
    pytestconfig: pytest.Config,
    tune: Tune,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Searches for the minimum weights of cosets agree with the least weights of their words.

    Routes by tails build completion tables, unless their budget is zero.
    """
    tune(**tuning)

    # Record the sizes of the tables that routes by tails ask for.
    table_bytes: list[int] = []
    get_table_bytes_within = completion_bounds.get_table_bytes_within

    def record(rows: npt.NDArray[np.uint64], block_bits: int, max_bytes: int) -> int:
        table_bytes.append(get_table_bytes_within(rows, block_bits, max_bytes))
        return table_bytes[-1]

    monkeypatch.setattr(completion_bounds, "get_table_bytes_within", record)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    tuning_fields = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    num_rows, num_columns = int(rng.integers(3, 6)), int(rng.integers(8, 11))
    random_code = get_random_full_rank_matrix(rng, num_rows, num_columns)
    for code in (RANK_DEFICIENT_CODE, _SKIPPING_CODE, random_code):
        even_code = np.hstack([code, code.sum(axis=1, keepdims=True) % 2]).astype(np.uint8)
        for generators in (code, even_code):
            checks = get_kernel(generators)
            search = coset._get_classical_search(generators, checks, tuning_fields)
            syndromes = range(1 << len(checks))
            minima = _coset_minima(generators, checks, syndromes)
            _assert_min_weights_match_brute_force(search, syndromes, minima, len(checks), rng)
    checks = get_kernel(_HEAVY_COSETS_CODE)
    search = coset._get_classical_search(_HEAVY_COSETS_CODE, checks, tuning_fields)
    early_stop_syndromes = [1126, 4578, 6129]
    minima = _coset_minima(_HEAVY_COSETS_CODE, checks, early_stop_syndromes)
    _assert_min_weights_match_brute_force(search, early_stop_syndromes, minima, len(checks), rng)
    # Routes by tails, which the costly tunings take, build tables unless their budget is zero.
    if "coset_route_minimum_work" in tuning:
        assert any(table_bytes) == ("coset_completion_max_table_bytes" not in tuning)
    if "coset_completion_max_table_bytes" in tuning:
        return
    matrix = get_graph_state(4, [(0, 1), (1, 2), (2, 3)], phase=True)
    search = coset.get_pauli_search(matrix, True, get_tuning())
    syndromes = range(1 << len(matrix))
    _assert_min_weights_match_brute_force(
        search, syndromes, get_exhaustive_minima(matrix, css_weight=True), len(matrix), rng
    )


@pytest.mark.parametrize("tuning", [{}, {"coset_route_minimum_work": 0}], ids=["words", "tails"])
def test_min_weight_searches_above_max_weight_examine_as_queries(
    tuning: dict[str, int], tune: Tune
) -> None:
    """A search for a minimum weight above ``max_weight`` weighs and skips as a query does."""
    tune(**tuning)
    tuning_fields = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    cases = []
    parities = (_SKIPPING_CODE.sum(axis=1, keepdims=True) % 2).astype(np.uint8)
    for generators, chosen_syndromes in [
        (_SKIPPING_CODE, None),
        (np.hstack([_SKIPPING_CODE, parities]), None),
        (_HEAVY_COSETS_CODE, [3571, 3722, 7870]),
    ]:
        checks = get_kernel(generators)
        syndromes = chosen_syndromes or list(range(1 << len(checks)))
        search = coset._get_classical_search(generators, checks, tuning_fields)
        minima = _coset_minima(generators, checks, syndromes)
        cases.append((search, len(checks), syndromes, minima, True))
    matrix = ROUTED_PHYSICAL_STATES[1]
    search = coset.get_pauli_search(matrix, False, get_tuning())
    syndromes = list(range(1 << len(matrix)))
    cases.append(
        (search, len(matrix), syndromes, get_exhaustive_minima(matrix, css_weight=False), False)
    )

    for search, num_syndrome_bits, syndromes, minima, must_skip in cases:
        skipped = 0
        for max_weight in range(2, int(max(minima))):
            # The fields of the tails are a plain tuple, as the searches pass them, so that numba
            # compiles the searches for the type of argument that they use.
            tables, tails = search._get_route(max_weight)
            for syndrome, minimum in zip(syndromes, minima, strict=True):
                if minimum <= max_weight:
                    continue
                packed_syndrome = split_syndromes_into_words([syndrome], num_syndrome_bits)[0]
                best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
                best_weight, parity = coset._get_syndrome_word(
                    packed_syndrome, best, tables.inverse, tables.x_mask, search.scale
                )
                stop_layer, stop_num_completed, stop_bound = coset_tables.get_stopping_step(
                    coset_tables.get_planning_table(tables.planning, parity),
                    max_weight,
                )
                _, *expected = coset._search_from_syndrome_word(
                    packed_syndrome,
                    max_weight,
                    stop_layer,
                    stop_num_completed,
                    stop_bound,
                    best,
                    best_weight,
                    tables.generators,
                    tables.pivots,
                    tables.x_mask,
                    tails,
                    search.scale,
                    coset._get_rounding_parity(tables.planning, search.scale, parity),
                )
                _, *found = coset._search_coset_for_min_weight(
                    packed_syndrome, max_weight, 0, best, *tables, tails, search.scale
                )
                skipped += expected[1]
                assert found == expected
        assert skipped or not (tuning and must_skip)


def test_tables_of_pairs_of_single_qubit_errors_respect_their_limit() -> None:
    """A table of products of X and Z errors keeps the first product of each syndrome."""
    x_singles = {0: 0, 0b01: 0b0001, 0b11: 0b0010}
    z_singles = {0: 0, 0b10: 0b0100}
    table = coset._tabulate_css_singles(x_singles, z_singles, 100)
    assert table == {0: 0, 0b10: 0b0100, 0b01: 0b0001, 0b11: 0b0101}
    assert coset._tabulate_css_singles(x_singles, z_singles, 1) is None


@pytest.mark.parametrize("tuning", [{}, {"coset_route_minimum_work": 0}], ids=["words", "tails"])
def test_batches_on_threads_answer_as_on_one_thread(
    tuning: dict[str, int], pytestconfig: pytest.Config, tune: Tune
) -> None:
    """Batch queries on threads return the index, bound, word, and weights of one thread.

    The calling thread searches the first rows of a batch alone, until their work reaches
    ``split_work``, and the threads search the rest in blocks, which hold several syndromes of a
    long batch on three threads, and one syndrome on eight threads.
    """
    tune(**tuning)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    tuning_fields = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    thread_options = [
        {"num_threads": num_threads, "split_work": split_work}
        for num_threads, split_work in itertools.product((3, 8), (0, 50))
    ]
    for search, minima in _get_searches_with_minima(tuning_fields):
        num_bits = search._num_syndrome_bits
        for max_weight in range(int(minima.max()) + 1):
            for syndromes in _get_batches(minima, max_weight, rng):
                batch = split_syndromes_into_words(syndromes.tolist(), num_bits)
                beyond = np.flatnonzero(minima[syndromes] > max_weight)
                expected = search.find_first_not_within_max_weight(batch, max_weight)
                assert expected[0] == (beyond[0] if len(beyond) else len(syndromes))
                floors = rng.integers(max_weight + 1, size=len(syndromes))
                weights = search.get_min_weights(batch, max_weight, floors)
                for options in thread_options:
                    found = search.find_first_not_within_max_weight(batch, max_weight, **options)
                    assert found == expected
                    assert np.array_equal(
                        search.get_min_weights(batch, max_weight, floors, **options), weights
                    )


@pytest.mark.parametrize(
    ("query", "compiled_name"),
    [
        ("find_first_not_within_max_weight", "_find_first_not_within_max_weight"),
        ("get_min_weights", "_find_min_weights"),
    ],
)
def test_batches_within_the_split_work_run_on_the_calling_thread(
    query: str, compiled_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch whose queries do less work than ``split_work`` runs in one call, on no thread."""
    search, _ = _get_rank_deficient_search()
    compiled = getattr(coset, compiled_name)
    calls: list[int] = []

    def record(syndromes: npt.NDArray[np.uint64], *args: Any) -> Any:
        calls.append(threading.get_ident())
        return compiled(syndromes, *args)

    monkeypatch.setattr(coset, compiled_name, record)
    batch = split_syndromes_into_words(list(range(32)) * 2, search._num_syndrome_bits)
    floors = np.zeros(len(batch), dtype=np.int64)
    arguments = (batch, 5) if query == "find_first_not_within_max_weight" else (batch, 5, floors)
    getattr(search, query)(*arguments, num_threads=4, split_work=10**9)
    assert calls == [threading.get_ident()]


@pytest.mark.parametrize(
    ("query", "compiled_name"),
    [
        ("find_first_not_within_max_weight", "_find_first_not_within_max_weight"),
        ("get_min_weights", "_find_min_weights"),
    ],
)
def test_errors_on_threads_propagate(
    query: str, compiled_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exception that a thread raises on a block reaches the caller once every thread ends."""
    search, _ = _get_rank_deficient_search()
    compiled = getattr(coset, compiled_name)
    sentinel = 7

    def fail_on_sentinel(syndromes: npt.NDArray[np.uint64], *args: Any) -> Any:
        """Raise an exception on a block that holds the sentinel syndrome, at row 40.

        The calling thread first searches row 0 alone, in a call that starts at row 0.
        """
        start = _get_row(syndromes, batch)
        if 0 < start <= 40 < start + len(syndromes):
            raise ValueError("sentinel")
        return compiled(syndromes, *args)

    monkeypatch.setattr(coset, compiled_name, fail_on_sentinel)
    batch = split_syndromes_into_words([1] * 40 + [sentinel] + [1] * 40, search._num_syndrome_bits)
    floors = np.zeros(len(batch), dtype=np.int64)
    arguments = (batch, 3) if query == "find_first_not_within_max_weight" else (batch, 3, floors)
    num_threads = threading.active_count()
    with pytest.raises(ValueError, match="sentinel"):
        getattr(search, query)(*arguments, num_threads=3, split_work=0)
    assert threading.active_count() == num_threads


def test_an_earlier_heavy_coset_that_ends_last_is_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first heavy coset of a batch is the answer even when a later one is found first.

    On two threads, the block of the earlier heavy coset waits until a thread has searched the
    block of the later one, and the answer is still that of one thread.
    """
    search, batch = _get_two_heavy_cosets()
    expected = search.find_first_not_within_max_weight(batch, 1)
    assert expected[0] == 2
    later_done = threading.Event()
    compiled = coset._find_first_not_within_max_weight

    def search_rows(syndromes: npt.NDArray[np.uint64], *args: Any) -> Any:
        """Hold the block of row 2 until the block of row 40 has returned."""
        start = _get_row(syndromes, batch)
        # The calling thread first searches row 0 alone, in a call that starts at row 0.
        if 0 < start <= 2 < start + len(syndromes):
            assert later_done.wait(THREAD_TIMEOUT)
        answer = compiled(syndromes, *args)
        if 0 < start <= 40 < start + len(syndromes):
            later_done.set()
        return answer

    monkeypatch.setattr(coset, "_find_first_not_within_max_weight", search_rows)
    assert search.find_first_not_within_max_weight(batch, 1, num_threads=2, split_work=0) == (
        expected
    )
    assert later_done.is_set()


def test_an_earlier_heavy_coset_recorded_before_a_later_one_is_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first heavy coset is the answer when it is recorded before a later one is found.

    On two threads, the block of row 40 starts before the block of row 2 is searched, and searches
    only once the range of the thread that found row 2 has returned, after it recorded row 2.
    """
    search, batch = _get_two_heavy_cosets()
    expected = search.find_first_not_within_max_weight(batch, 1)
    started_40, recorded_2 = threading.Event(), threading.Event()
    finders: list[int] = []
    run_work_items = coset._threads.run_work_items
    compiled = coset._find_first_not_within_max_weight

    def run_and_watch_ranges(num_items: int, num_threads: int, start: Any, **kwargs: Any) -> None:
        """Run the work items, and note when the thread that searched row 2 ends its range."""

        def start_worker(stop: threading.Event) -> Any:
            """Start a worker, and watch the ends of its ranges."""
            run_range = start(stop)

            def watched_range(first: int, end: int) -> None:
                """Run a range, and signal its end if its thread searched row 2."""
                run_range(first, end)
                if finders == [threading.get_ident()]:
                    recorded_2.set()

            return watched_range

        run_work_items(num_items, num_threads, start_worker, **kwargs)

    def search_rows(syndromes: npt.NDArray[np.uint64], *args: Any) -> Any:
        """Hold row 2 until row 40 starts, and row 40 until row 2 is recorded."""
        start = _get_row(syndromes, batch)
        if 0 < start <= 2 < start + len(syndromes):
            finders.append(threading.get_ident())
            assert started_40.wait(THREAD_TIMEOUT)
        if 0 < start <= 40 < start + len(syndromes):
            started_40.set()
            assert recorded_2.wait(THREAD_TIMEOUT)
        return compiled(syndromes, *args)

    monkeypatch.setattr(coset._threads, "run_work_items", run_and_watch_ranges)
    monkeypatch.setattr(coset, "_find_first_not_within_max_weight", search_rows)
    assert search.find_first_not_within_max_weight(batch, 1, num_threads=2, split_work=0) == (
        expected
    )


def test_the_word_of_an_earlier_heavy_coset_survives_a_later_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The word of the first heavy coset is its own, though another thread searches after it.

    On two threads, the search of the block of row 2 returns, and its thread waits until the search
    of the block of row 40 has returned before it records row 2.
    """
    search, batch = _get_two_heavy_cosets()
    expected = search.find_first_not_within_max_weight(batch, 1)
    returned_2, returned_40 = threading.Event(), threading.Event()
    compiled = coset._find_first_not_within_max_weight

    def search_rows(syndromes: npt.NDArray[np.uint64], *args: Any) -> Any:
        """Hold the answer for row 2 until the search of row 40 has returned."""
        start = _get_row(syndromes, batch)
        if 0 < start <= 2 < start + len(syndromes):
            answer = compiled(syndromes, *args)
            returned_2.set()
            assert returned_40.wait(THREAD_TIMEOUT)
            return answer
        if 0 < start <= 40 < start + len(syndromes):
            assert returned_2.wait(THREAD_TIMEOUT)
            answer = compiled(syndromes, *args)
            returned_40.set()
            return answer
        return compiled(syndromes, *args)

    monkeypatch.setattr(coset, "_find_first_not_within_max_weight", search_rows)
    assert search.find_first_not_within_max_weight(batch, 1, num_threads=2, split_work=0) == (
        expected
    )


@pytest.mark.parametrize(
    ("query", "compiled_name"),
    [
        ("find_first_not_within_max_weight", "_find_first_not_within_max_weight"),
        ("get_min_weights", "_find_min_weights"),
    ],
)
def test_costly_batches_move_to_threads_once_their_work_reaches_the_split_work(
    query: str, compiled_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch of costly queries moves to threads once their work reaches ``split_work``.

    The 16 queries alone count 16 units, one less than ``split_work``; each also weighs words, so
    the work reaches it before the batch ends.
    """
    search, minima = _get_rank_deficient_search()
    max_weight = int(minima.max())
    heaviest = int(np.flatnonzero(minima == max_weight)[0])
    batch = split_syndromes_into_words([heaviest] * 16, search._num_syndrome_bits)
    compiled = getattr(coset, compiled_name)
    starts: list[int] = []

    def record(syndromes: npt.NDArray[np.uint64], *args: Any) -> Any:
        """Record the row at which each call starts."""
        starts.append(_get_row(syndromes, batch))
        return compiled(syndromes, *args)

    monkeypatch.setattr(coset, compiled_name, record)
    floors = np.zeros(len(batch), dtype=np.int64)
    arguments = (
        (batch, max_weight)
        if query == "find_first_not_within_max_weight"
        else (batch, max_weight, floors)
    )
    getattr(search, query)(*arguments, num_threads=2, split_work=len(batch) + 1)
    assert starts[0] == 0
    assert any(start > 0 for start in starts[1:])


def test_copies_of_planning_buffers_share_nothing_and_start_from_the_first_step() -> None:
    """A copy of a planning buffer has its bounds, shares no memory, and caches the first step."""
    search, _ = _get_rank_deficient_search()
    planning = search._get_route(3)[0].planning.copy()
    planning[:, -1, :2] = (1, 2)
    own_planning = coset._get_own_planning(planning)
    assert not np.shares_memory(own_planning, planning)
    assert (own_planning[:, -1, :2] == (0, 1)).all()
    assert np.array_equal(own_planning[:, :-1], planning[:, :-1])
    assert np.array_equal(own_planning[:, -1, 2:], planning[:, -1, 2:])


def test_threads_skip_the_blocks_of_their_range_after_a_heavy_coset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The thread that finds a coset without a light word in a block searches no later block."""
    search, minima = _get_rank_deficient_search()
    light, heavy = int(np.flatnonzero(minima <= 1)[1]), int(np.flatnonzero(minima > 1)[0])
    batch = split_syndromes_into_words([light, heavy] + [light] * 62, search._num_syndrome_bits)
    calls: list[tuple[int, int]] = []
    compiled = coset._find_first_not_within_max_weight

    def record(syndromes: npt.NDArray[np.uint64], *args: Any) -> Any:
        """Record the first row of each block that a thread searches, and the thread."""
        calls.append((_get_row(syndromes, batch), threading.get_ident()))
        return compiled(syndromes, *args)

    monkeypatch.setattr(coset, "_find_first_not_within_max_weight", record)
    index, bound = search.find_first_not_within_max_weight(batch, 1, num_threads=2, split_work=0)
    assert index == 1 and bound is not None
    # The calling thread searches row 0 alone first, and the threads take the rest in blocks.
    assert calls[0] == (0, threading.get_ident())
    (finder,) = [thread for row, thread in calls[1:] if row == 1]
    assert [row for row, thread in calls[1:] if thread == finder] == [1]


def test_each_worker_searches_with_its_own_copy_of_the_planning_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The threads of a batch search with copies of the planning buffer, one for each worker.

    On two threads, the block at row 1 waits until the other thread has started a block, so both
    threads search.
    """
    search, _ = _get_rank_deficient_search()
    batch = split_syndromes_into_words([1] * 64, search._num_syndrome_bits)
    route_planning = search._get_route(1)[0].planning
    calls: list[tuple[int, npt.NDArray[np.int64]]] = []
    other_searched = threading.Event()
    compiled = coset._find_first_not_within_max_weight

    def search_rows(syndromes: npt.NDArray[np.uint64], *args: Any) -> Any:
        """Record the thread and the planning buffer of each call, and hold the block at row 1."""
        # The arguments after the syndromes start with max_weight, max_work, best, inverse,
        # generators, pivots, and planning.
        calls.append((threading.get_ident(), args[6]))
        start = _get_row(syndromes, batch)
        if start == 1:
            assert other_searched.wait(THREAD_TIMEOUT)
        elif start > 1:
            other_searched.set()
        return compiled(syndromes, *args)

    monkeypatch.setattr(coset, "_find_first_not_within_max_weight", search_rows)
    search.find_first_not_within_max_weight(batch, 1, num_threads=2, split_work=0)
    assert len({thread for thread, _ in calls}) == 2
    assert not any(np.shares_memory(planning, route_planning) for _, planning in calls)
    for (thread, planning), (other, other_planning) in itertools.combinations(calls, 2):
        assert thread == other or not np.shares_memory(planning, other_planning)


def test_threads_stop_within_their_ranges_after_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A thread searches no more blocks of its range once another thread raises an exception.

    On two threads, the first block waits until the threads stop, which happens once the other
    thread has raised an exception on its first block.
    """
    search, _ = _get_rank_deficient_search()
    batch = split_syndromes_into_words([1] * 64, search._num_syndrome_bits)
    stops: list[threading.Event] = []
    calls: list[tuple[int, int]] = []
    run_work_items = coset._threads.run_work_items
    compiled = coset._find_first_not_within_max_weight

    def run_and_keep_the_stop_event(
        num_items: int, num_threads: int, start: Any, **kwargs: Any
    ) -> None:
        """Run the work items, and keep the stop event that the threads share."""

        def start_worker(stop: threading.Event) -> Any:
            """Keep the stop event of the threads, and start a worker."""
            stops.append(stop)
            return start(stop)

        run_work_items(num_items, num_threads, start_worker, **kwargs)

    def search_rows(syndromes: npt.NDArray[np.uint64], *args: Any) -> Any:
        """Wait at the first block until the threads stop, and raise at any later block."""
        row = _get_row(syndromes, batch)
        calls.append((row, threading.get_ident()))
        if row == 1:
            assert stops[0].wait(THREAD_TIMEOUT)
        elif row > 1:
            raise ValueError("a later block")
        return compiled(syndromes, *args)

    monkeypatch.setattr(coset._threads, "run_work_items", run_and_keep_the_stop_event)
    monkeypatch.setattr(coset, "_find_first_not_within_max_weight", search_rows)
    with pytest.raises(ValueError, match="a later block"):
        search.find_first_not_within_max_weight(batch, 1, num_threads=2, split_work=0)
    (waiter,) = [thread for row, thread in calls if row == 1]
    assert [row for row, thread in calls if thread == waiter and row > 0] == [1]


def test_concurrent_batches_share_a_search(pytestconfig: pytest.Config, tune: Tune) -> None:
    """Batches that run at once on threads of their own share a search, and answer as alone.

    Each batch has its own maximum weight, and so its own stopping steps, which it caches in its
    own copies of the planning buffer of its route.  The batches run with every array of the
    search, such as the shared buffer, read-only.
    """
    tune(coset_route_minimum_work=0)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    tuning_fields = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    even_code = np.hstack([_SKIPPING_CODE, _SKIPPING_CODE.sum(axis=1, keepdims=True) % 2])
    checks = get_kernel(even_code.astype(np.uint8))
    search = coset._get_classical_search(even_code.astype(np.uint8), checks, tuning_fields)
    max_weights = rng.integers(even_code.shape[1] + 1, size=64).tolist()
    batches = [
        split_syndromes_into_words(rng.integers(1 << len(checks), size=50).tolist(), len(checks))
        for _ in max_weights
    ]
    floors = np.zeros(50, dtype=np.int64)
    plannings = [search._get_route(max_weight)[0].planning for max_weight in set(max_weights)]
    # Plant a cached step other than the first, so that a copy that shared memory with its buffer
    # would change the buffer when it resets its own step to the first.
    for planning in plannings:
        planning[:, -1, :2] = (len(planning[0]) - 2, planning.shape[2] - 1)

    def query(index: int, num_threads: int) -> tuple[Any, ...]:
        """Answer both batch queries of a batch."""
        batch, max_weight = batches[index], max_weights[index]
        options = {"num_threads": num_threads, "split_work": 0}
        found = search.find_first_not_within_max_weight(batch, max_weight, **options)
        weights = search.get_min_weights(batch, max_weight, floors, **options)
        return found, weights.tolist()

    # The batch queries cache stopping steps only in copies of the shared planning buffers, and
    # write no other array of the search or of its batches.  Daemon threads, joined with a timeout,
    # let a query that never returns fail the test rather than hang it.
    with shared_arrays_read_only(search, batches, floors) as arrays:
        expected = [query(index, 1) for index in range(len(batches))]
        answers: list[tuple[Any, ...] | None] = [None] * len(batches)

        def answer(first: int) -> None:
            """Answer every fourth batch from ``first``, with two threads per batch."""
            for index in range(first, len(batches), 4):
                answers[index] = query(index, 2)

        threads = [
            threading.Thread(target=answer, args=(first,), daemon=True) for first in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=THREAD_TIMEOUT)
            assert not thread.is_alive()
        assert answers == expected
    assert any(each.path.startswith("roots[0].") for each in arrays)
    marked = {id(each.array) for each in arrays}
    assert all(id(planning) in marked for planning in plannings)


def _assert_classical_cosets_match_brute_force(generators: npt.NDArray[np.uint8]) -> None:
    """Check single queries and batches of a code's classical coset search against brute force.

    A coset without a word within the maximum weight must get a bound of at least the least weight
    above the maximum weight that a word of the coset can have.  Below maximum weight two, a single
    query must give a coset without a word of weight at most one exactly the least weight above one
    that a word of the coset can have, and from two on, single queries and batches must agree.
    """
    num_columns = generators.shape[1]
    checks = get_kernel(generators)
    tuning = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    search = coset._get_classical_search(
        generators, checks, tuning._replace(disjoint_random_column_orders=0)
    )
    minima = get_binary_minima(checks)
    even = not np.any(generators.sum(axis=1) % 2)

    def get_least_bound(syndrome: int, max_weight: int) -> int:
        """The least weight above the maximum weight that a word of the coset can have."""
        return max_weight + 1 + int(even and (max_weight + 1 - minima[syndrome]) % 2)

    syndromes = list(range(1 << len(checks)))
    packed_syndromes = split_syndromes_into_words(syndromes, len(checks))
    for max_weight in range(-1, num_columns + 2):
        bounds = {}
        for syndrome in syndromes:
            bound = search.get_min_weight_bound(syndrome, max_weight)
            representative = (bound.representative >> np.arange(num_columns)) & 1
            weight = int(representative.sum())
            assert bound.within_max_weight == (minima[syndrome] <= max_weight)
            assert get_binary_syndrome(checks, representative) == syndrome
            assert bound.lower_bound <= minima[syndrome] <= weight
            if bound.within_max_weight:
                assert weight <= max_weight
            else:
                assert bound.lower_bound >= get_least_bound(syndrome, max_weight)
            if max_weight < 2 and minima[syndrome] >= 2:
                assert bound.lower_bound == get_least_bound(syndrome, 1)
            bounds[syndrome] = bound

        # With batches that each start after the last syndrome found, find each syndrome whose
        # minimum exceeds the maximum weight.
        found, start = [], 0
        while True:
            index, batch_bound = search.find_first_not_within_max_weight(
                packed_syndromes[start:], max_weight
            )
            if batch_bound is None:
                assert start + index == len(syndromes)
                break
            syndrome = syndromes[start + index]
            representative = (batch_bound.representative >> np.arange(num_columns)) & 1
            assert get_binary_syndrome(checks, representative) == syndrome
            assert not batch_bound.within_max_weight
            assert get_least_bound(syndrome, max_weight) <= batch_bound.lower_bound
            assert batch_bound.lower_bound <= minima[syndrome]
            if max_weight >= 2:
                assert bounds[syndrome] == batch_bound
            found.append(syndrome)
            start += index + 1
        assert found == [syndrome for syndrome in syndromes if minima[syndrome] > max_weight]


def _assert_searches_by_tails_match_searches_of_words(
    generators: npt.NDArray[np.uint8],
    inverse: npt.NDArray[np.uint8],
    scale: int,
    block_sizes: Sequence[int],
    max_weights: Sequence[int],
    syndromes: Sequence[int],
    minima: npt.NDArray[np.int_] | None = None,
    *,
    block_bits: int | None = None,
) -> int:
    """Check searches by tails against searches of whole words, without or with completion tables.

    The family reduces the embedded generators on consecutive blocks, whose sizes cycle through
    ``block_sizes``, and adds a set of rank zero.  Without ``block_bits``, both searches give the
    same bounds, words, and counts.  With tables on blocks of ``block_bits`` columns, they differ
    only as Section 3.9 of ``brouwer_zimmermann.md`` ("Coset searches") allows.  If ``minima`` is
    given, the searches also decide as these least weights of the cosets.

    Returns:
        The number of supports that the searches by tails skipped.
    """
    num_rows, num_columns = generators.shape
    sets, pivots, start = [], [], 0
    for size in itertools.cycle(block_sizes):
        if start >= num_columns:
            break
        block = np.arange(start, min(start + size, num_columns))
        reduced, _, block_pivots = row_reduce_binary(generators, columns=block)
        sets.append(reduced)
        pivots.append(np.pad(block_pivots, (0, num_rows - len(block_pivots)), constant_values=-1))
        start += size
    sets.append(generators)
    pivots.append(np.full(num_rows, -1))
    instance = affine_instance.AffineInstance(
        generators=np.stack([pack_bits(rows) for rows in sets]),
        labels=np.zeros((len(sets), num_rows, 0), dtype=np.uint64),
        widths=np.ones(len(sets), dtype=np.int64),
        pivots=np.array(pivots, dtype=np.int64),
        certificate=certificates.Certificate(
            num_rows, tuple(int(np.sum(row >= 0)) for row in pivots)
        ),
        weight_factor=2 if scale == 2 else 1,
    )
    tables = coset_tables.get_search_tables(instance, inverse, num_columns)
    # Pass the fields of the tails as a plain tuple, as the searches do, so that numba compiles the
    # searches for the type of argument that they use.
    tails = coset_tables.get_coset_tails(
        instance,
        list(range(len(sets))) if block_bits else [],
        None,
        inverse=inverse,
        block_bits=block_bits or 8,
        max_table_bytes=MAX_VALUE,
    )[:]
    even = scale == 1 and not np.any(generators.sum(axis=1) % 2)
    packed_syndromes = split_syndromes_into_words(syndromes, len(inverse))
    total_skipped = 0
    for max_weight in max_weights:
        answers = {}
        for syndrome, packed_syndrome in zip(syndromes, packed_syndromes, strict=True):
            words = np.empty((2, tables.inverse.shape[1]), dtype=np.uint64)
            best_weight, parity = coset._get_syndrome_word(
                packed_syndrome, words[0], tables.inverse, tables.x_mask, scale
            )
            if best_weight <= max_weight:
                continue
            words[1] = words[0]
            word = unpack_bits(words[:1], num_columns)[0]
            coset_parity = int(word.sum()) % 2
            stop_layer, stop_num_completed, stop_bound = coset_tables.get_stopping_step(
                coset_tables.get_planning_table(tables.planning, parity), max_weight
            )
            *answer, _ = coset._search_from_syndrome_word(
                packed_syndrome,
                max_weight,
                stop_layer,
                stop_num_completed,
                stop_bound,
                words[0],
                best_weight,
                tables.generators,
                tables.pivots,
                tables.x_mask,
                None,
                scale,
                coset._get_rounding_parity(tables.planning, scale, parity),
            )
            *answer_by_tails, skipped = coset._search_from_syndrome_word(
                packed_syndrome,
                max_weight,
                stop_layer,
                stop_num_completed,
                stop_bound,
                words[1],
                best_weight,
                tables.generators,
                tables.pivots,
                tables.x_mask,
                tails,
                scale,
                coset._get_rounding_parity(tables.planning, scale, parity),
            )
            if minima is not None:
                assert (answer[0] <= max_weight) == (minima[syndrome] <= max_weight)
                assert answer[0] <= minima[syndrome]
            answers[syndrome] = answer_by_tails[0], words[1]
            total_skipped += skipped
            if not block_bits:
                assert answer_by_tails == list(answer) and not skipped
                np.testing.assert_array_equal(words[0], words[1])
            elif answer[0] <= max_weight:
                assert answer_by_tails[0] == 0
                np.testing.assert_array_equal(words[0], words[1])
            else:
                least_beyond = max_weight + 1
                if even and (least_beyond + coset_parity) % 2:
                    least_beyond += 1
                assert answer_by_tails[0] == (least_beyond if skipped else answer[0])
                assert int(np.bitwise_count(words[1]).sum()) >= scale * answer_by_tails[0]
                # The skip bound ``h(ω)`` of Section 3.9 of ``brouwer_zimmermann.md``: one more than
                # the greatest Hamming weight at most ``scale * max_weight`` that a word of the
                # coset can have.
                skip_bound = scale * max_weight + 1
                if even and (skip_bound - coset_parity) % 2 == 0:
                    skip_bound -= 1
                assert answer_by_tails[1] + skipped == answer[1]
                assert skipped == sum(
                    _count_skipped_supports(rows, set_pivots, word, skip_bound, block_bits, last)
                    for index, (rows, set_pivots) in enumerate(zip(sets, pivots, strict=True))
                    if (last := stop_layer - (index >= stop_num_completed))
                    >= num_rows - np.sum(set_pivots >= 0)
                )

        # With both batch loops, find in turn each syndrome without a light enough word.
        start = 0
        while True:
            words = np.empty((2, tables.inverse.shape[1]), dtype=np.uint64)
            found = coset._find_first_not_within_max_weight(
                packed_syndromes[start:],
                max_weight,
                _threads.NO_WORK_LIMIT,
                words[0],
                *tables,
                None,
                scale,
            )
            found_by_tails = coset._find_first_not_within_max_weight(
                packed_syndromes[start:],
                max_weight,
                _threads.NO_WORK_LIMIT,
                words[1],
                *tables,
                tails,
                scale,
            )
            assert found_by_tails[0] == found[0]
            assert found_by_tails[2] == found[2] == (start + found[0] < len(packed_syndromes))
            if start + found[0] == len(packed_syndromes):
                break
            if block_bits:
                bound, word = answers[syndromes[start + found[0]]]
                assert found_by_tails[1] == bound
                np.testing.assert_array_equal(words[1], word)
            else:
                assert found_by_tails == found
                np.testing.assert_array_equal(words[0], words[1])
            start += found[0] + 1
    return total_skipped


def _count_skipped_supports(
    rows: npt.NDArray[np.uint8],
    pivots: npt.NDArray[np.int_],
    word: npt.NDArray[np.uint8],
    skip_bound: int,
    block_bits: int,
    last_layer: int,
) -> int:
    """Count the supports that tables skip in the layers from two to ``last_layer`` of a set.

    ``word`` is the word of a syndrome.  The bound of a support follows Section 3.9 of
    ``brouwer_zimmermann.md`` ("Coset searches"), on blocks that cover the tails up to their last
    column with a one.
    """
    rank = int(np.sum(pivots >= 0))
    offset = (word + rows[[row for row in range(rank) if word[pivots[row]]]].sum(axis=0)) % 2
    columns = np.delete(np.arange(rows.shape[1]), pivots[:rank])
    tails, offset_tail = rows[:, columns], offset[columns]
    num_bits = int(np.flatnonzero(tails.any(axis=0)).max(initial=-1)) + 1
    tails, offset_tail = tails[:, :num_bits], offset_tail[:num_bits]
    skipped = 0
    for layer in range(2, last_layer + 1):
        for support in itertools.combinations(range(len(rows)), layer):
            prefix = (offset_tail + tails[list(support[:-2])].sum(axis=0)) % 2
            pairs = list(itertools.combinations(range(support[-2], len(rows)), 2))
            bound = sum(row < rank for row in support[:-2])
            bound += min(sum(row < rank for row in pair) for pair in pairs)
            bound += sum(
                min(
                    int(((prefix + tails[i] + tails[j]) % 2)[start : start + block_bits].sum())
                    for i, j in pairs
                )
                for start in range(0, num_bits, block_bits)
            )
            if num_bits and bound >= skip_bound:
                skipped += 1
    return skipped


def _coset_minima(
    generators: npt.NDArray[np.uint8], checks: npt.NDArray[np.uint8], syndromes: Sequence[int]
) -> npt.NDArray[np.int_]:
    """The least weight of the words of the coset of each syndrome, from every word of the coset."""
    code = np.array(list(itertools.product([0, 1], repeat=len(generators)))) @ generators % 2
    bits = (np.array(syndromes)[:, None] >> np.arange(len(checks))) & 1
    words = bits @ _cosets.get_unit_syndrome_words(checks) % 2
    assert np.array_equal(get_binary_syndrome(checks, words), syndromes)
    return np.array([((word + code) % 2).sum(axis=1).min() for word in words])


def _assert_min_weights_match_brute_force(
    search: coset.BrouwerZimmermannCosetSearch,
    syndromes: Sequence[int],
    minima: npt.NDArray[np.int_],
    num_syndrome_bits: int,
    rng: np.random.Generator,
) -> None:
    """Check the minimum weights that a search finds for cosets against their least weights.

    One batch holds the syndromes, in random order.  At each maximum weight, the floors of the batch
    are zero, then random, then the maximum weight.  A coset whose least weight is at most its floor
    may get any weight from its least weight to its floor.  Any other coset gets its least weight,
    or one more than the maximum weight if that is smaller.
    """
    order = rng.permutation(len(syndromes))
    packed_syndromes = split_syndromes_into_words(
        [syndromes[index] for index in order], num_syndrome_bits
    )
    least = minima[order]
    for max_weight in range(int(least.max()) + 2):
        random_floors = rng.integers(0, max_weight + 1, len(least))
        for floors in (np.zeros_like(least), random_floors, np.full_like(least, max_weight)):
            weights = search.get_min_weights(packed_syndromes, max_weight, floors.astype(np.int64))
            exact = least > floors
            assert np.array_equal(weights[exact], np.minimum(least, max_weight + 1)[exact])
            assert np.all(exact | ((least <= weights) & (weights <= floors)))


def _get_searches_with_minima(
    tuning: coset._CosetTuning,
) -> list[tuple[coset.BrouwerZimmermannCosetSearch, npt.NDArray[np.int_]]]:
    """Searches of each kind, by physical and CSS weight and of X or Z parts, with their minima.

    The minima are the least weights of the words with each packed syndrome of a search.
    """
    states = [(ROUTED_PHYSICAL_STATES[0], False), (ROUTED_CSS_WEIGHT_STATE, True)]
    searches = [
        (
            coset._get_pauli_search(coset._MatrixKey(matrix), css_weight, tuning),
            get_exhaustive_minima(matrix, css_weight),
        )
        for matrix, css_weight in states
    ]
    even_code = np.hstack([_SKIPPING_CODE, _SKIPPING_CODE.sum(axis=1, keepdims=True) % 2])
    for code in (RANK_DEFICIENT_CODE, even_code.astype(np.uint8)):
        checks = get_kernel(code)
        searches.append(
            (coset._get_classical_search(code, checks, tuning), get_binary_minima(checks))
        )
    return searches


def _get_batches(
    minima: npt.NDArray[np.int_], max_weight: int, rng: np.random.Generator
) -> list[npt.NDArray[np.int_]]:
    """Batches of syndromes whose first coset without a word within ``max_weight`` varies.

    The batches hold such a coset nowhere, first, last, or at several places, and one batch is
    short.
    """
    within = np.flatnonzero(minima <= max_weight)
    beyond = np.flatnonzero(minima > max_weight)
    base = rng.choice(within, 100)
    batches = [base, base[:3]]
    if len(beyond):
        for places in ([0], [-1], [33, 50, -1]):
            batch = base.copy()
            batch[places] = rng.choice(beyond, len(places))
            batches.append(batch)
    return batches


def _get_rank_deficient_search() -> tuple[coset.BrouwerZimmermannCosetSearch, npt.NDArray[np.int_]]:
    """The classical search of ``RANK_DEFICIENT_CODE``, and the minima of its syndromes."""
    tuning_fields = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    checks = get_kernel(RANK_DEFICIENT_CODE)
    search = coset._get_classical_search(RANK_DEFICIENT_CODE, checks, tuning_fields)
    return search, get_binary_minima(checks)


def _get_row(syndromes: npt.NDArray[np.uint64], batch: npt.NDArray[np.uint64]) -> int:
    """The index in ``batch`` of the first row of ``syndromes``, a view of rows of ``batch``."""
    return (syndromes.ctypes.data - batch.ctypes.data) // batch.strides[0]


def _get_two_heavy_cosets() -> tuple[coset.BrouwerZimmermannCosetSearch, npt.NDArray[np.uint64]]:
    """A search, and a batch of 64 syndromes whose only heavy cosets are at rows 2 and 40.

    A coset is heavy if it has no word within weight one.
    """
    search, minima = _get_rank_deficient_search()
    light, heavy = int(np.flatnonzero(minima <= 1)[1]), np.flatnonzero(minima > 1)[:2].tolist()
    rows = [light] * 64
    rows[2], rows[40] = heavy
    return search, split_syndromes_into_words(rows, search._num_syndrome_bits)
