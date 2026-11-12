# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Brouwer-Zimmermann coset searches of ``CosetLeaderSearch``."""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Sequence
from math import comb
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import math
from qldpc._binary import get_kernel, pack_bits, row_reduce_binary, unpack_bits

from ...conftest import Tune
from .. import _cosets
from .._binary import symplectic_to_hamming
from .._coset_leader_search import CosetLeaderSearch
from .._cosets import MinWeightBound
from .._information_sets import basis_families, completion_bounds
from .._information_sets import certificates as certificates_module
from .._tuning import DEFAULT_TUNING, MAX_VALUE, get_tuning, use_tuning
from ..conftest import (
    _RANK_DEFICIENT_CODE,
    _ROUTED_CSS_WEIGHT_STATE,
    _ROUTED_PHYSICAL_STATES,
    EVEN_ROUTES_CODE,
    OVERLAPPING_ROUTES_CODE,
    TWO_FAMILIES_CODE,
    _assert_exact_answers,
    _css_minima,
    _get_syndrome,
    _graph_state,
    _pack_syndromes,
    _product_state,
    _scramble,
    _stabilizer_matrix,
    get_binary_matrix,
    get_random_full_rank_matrix,
)
from . import affine_instance, coset, coset_tables


def _classical_syndrome(checks: npt.NDArray[np.uint8], words: npt.NDArray[np.int_]) -> Any:
    """The packed syndrome of a binary word for binary checks, or of each row of an array of words.

    Bit ``j`` of a syndrome is the parity of the word with row ``j`` of ``checks``.
    """
    return (words @ checks.T % 2) @ (1 << np.arange(len(checks)))


def _classical_minima(checks: npt.NDArray[np.uint8]) -> npt.NDArray[np.int_]:
    """The least Hamming weight of the words with each packed syndrome for binary checks.

    The minima come from all ``2**n`` words of ``n`` bits.
    """
    num_columns = checks.shape[1]
    words = (np.arange(1 << num_columns)[:, None] >> np.arange(num_columns)) & 1
    minima = np.full(1 << len(checks), num_columns + 1)
    np.minimum.at(minima, _classical_syndrome(checks, words), words.sum(axis=1))
    return minima


def test_css_weight_searches_of_even_stabilizers_stop_at_bounds_of_each_class_parity() -> None:
    """Searches of Pauli errors by CSS weight stop at the bounds of each class, rounded to parity.

    The state is the graph state of a triangle on qubits 2, 5, and 6, with qubit 1 joined to qubit 2
    and qubits 0, 3, and 4 isolated, after S on each qubit of even degree.  Its stabilizers have
    even weight as binary words ``(x|z)``, and it is not CSS, so the oracle searches its Pauli
    errors.  The errors of a syndrome class share the parity of ``|x| + |z|``, which the test reads
    from every error.  Every query agrees with exhaustive minima.  A search that took the parity of
    the CSS weight instead would not: for syndrome 57, whose class has minimum two, the word of the
    syndrome has Hamming weight four but CSS weight three, and the bounds of odd classes stop the
    search at maximum weight two before it finds an error of weight two.

    Answers alone cannot show whether these searches round, since rounding mainly moves where an
    odd class stops (see Section 3.2 of ``brouwer_zimmermann.md``).  So the test also finds the
    stopping step of each query as ``_search_coset`` does, and checks it against the first step
    whose bound in ``AffineInstance.get_weight_bounds`` for the parity of the class exceeds the
    maximum weight.  ``information_sets_test.py`` and ``affine_instance_test.py`` check those bounds
    against their definition and against brute force.  Some odd classes stop sooner than they would
    without parity.
    """
    generators = ("Y______", "_XZ____", "_ZX__ZZ", "___Y___", "____Y__", "__Z__YZ", "__Z__ZY")
    matrix = _stabilizer_matrix([stim.PauliString(pauli) for pauli in generators])
    _assert_exact_answers(CosetLeaderSearch(matrix, css_weight=True), matrix, True)

    # Read the parity of each syndrome class from every error, and check that the class shares it.
    num_qubits = len(matrix)
    checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
    errors = (np.arange(1 << 2 * num_qubits)[:, None] >> np.arange(2 * num_qubits)) & 1
    syndromes, parities = _classical_syndrome(checks, errors), errors.sum(axis=1) % 2
    class_parities = np.zeros(1 << num_qubits, dtype=int)
    class_parities[syndromes] = parities
    assert np.array_equal(class_parities[syndromes], parities)

    # Find the stopping step of each query as ``_search_coset`` does.  Steps ``(layer,
    # num_completed)`` compare as tuples in the order of the search.
    instance = affine_instance.get_coset_instance(matrix, random_column_orders=0, weight_factor=2)
    inverse = _cosets._get_inverse(checks)
    tables = coset_tables._get_search_tables(instance, inverse, num_qubits)
    unrounded = instance.get_weight_bounds()
    steps = list(itertools.product(range(len(unrounded)), range(1, unrounded.shape[1])))
    best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
    num_sooner = 0
    for syndrome in range(1 << num_qubits):
        _, parity = coset._get_syndrome_word(
            _pack_syndromes([syndrome], num_qubits)[0], best, tables.inverse, tables.x_mask, 0
        )
        table = coset_tables._get_planning_table(tables.planning, parity)
        bounds = instance.get_weight_bounds(class_parities[syndrome])
        for max_weight in range(2, num_qubits):
            stopping_step = next(step for step in steps if bounds[step] > max_weight)
            assert coset_tables._get_stopping_step(table, max_weight)[:2] == stopping_step
            num_sooner += stopping_step < next(
                step for step in steps if unrounded[step] > max_weight
            )
    assert num_sooner


def test_cached_searches_answer_only_for_their_stabilizer_group(
    pytestconfig: pytest.Config,
) -> None:
    """Oracles built in turn for |000+++>, another generating set of it, and |+++000> are exact.

    The searches of a CSS state are cached by stabilizer group, so the second oracle reuses those of
    the first.  |+++000> has the same numbers of qubits and of pure-X stabilizers as |000+++>, but a
    different stabilizer group, so its oracle must not reuse them.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    eye, zeros = np.eye(3, dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)
    zero_plus = np.block([[zeros, eye, zeros, zeros], [zeros, zeros, eye, zeros]])
    plus_zero = np.block([[eye, zeros, zeros, zeros], [zeros, zeros, zeros, eye]])
    matrices = [_scramble(matrix, rng) for matrix in (zero_plus, zero_plus, plus_zero)]
    coset._get_css_searches.cache_clear()
    oracles = [CosetLeaderSearch(matrix, css_weight=True) for matrix in matrices]
    for oracle, matrix in zip(oracles, matrices, strict=True):
        minima = _css_minima(matrix)
        for syndrome, max_weight in itertools.product(range(1 << 6), range(4)):
            bound = oracle.get_min_weight_bound(syndrome, max_weight)
            assert bound.within_max_weight == (minima[syndrome] <= max_weight)
            assert _get_syndrome(matrix, bound.representative) == syndrome


def test_oracles_use_the_tuning_with_which_they_are_built(monkeypatch: pytest.MonkeyPatch) -> None:
    """An oracle builds its searches with the tuning in force, keyed by the fields that they read.

    The searches are cached, and the column orders key the cache.  So the first oracle built under
    some orders builds searches of its own with them, and a second oracle built under the same
    orders shares those searches.  The orders reach a search through
    ``information_sets.get_disjoint_family``, which the test wraps to record them.  The states are
    |000+++>, whose two classical searches are cached by stabilizer group, and a two-qubit graph
    state with phases, which is not CSS, so its search of Pauli errors is cached by stabilizers.
    Every oracle answers exactly.  The caches are cleared first, since earlier tests may have filled
    them.  Only a search of Pauli errors by CSS weight reads ``max_css_single_pairs``, so the
    classical searches, and a search of Pauli errors by physical weight, are shared by oracles built
    under different values of it, and a search of Pauli errors by CSS weight is not.  That search
    alone has no completion tables, so it alone is shared by oracles built under different values of
    ``coset_completion_block_bits``.
    """
    orders: list[int] = []
    get_disjoint_family = basis_families.get_disjoint_family

    def record(*args: Any, random_column_orders: int) -> basis_families.BasisFamily:
        orders.append(random_column_orders)
        return get_disjoint_family(*args, random_column_orders=random_column_orders)

    monkeypatch.setattr(basis_families, "get_disjoint_family", record)
    eye, zeros = np.eye(3, dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)
    zero_plus = np.block([[zeros, eye, zeros, zeros], [zeros, zeros, eye, zeros]])
    phased = _graph_state(2, [(0, 1)], phase=True)
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
        for oracle in (first, second):
            _assert_exact_answers(oracle, matrix, True)
    for matrix, css_weight in ((zero_plus, True), (phased, False), (phased, True)):
        CosetLeaderSearch(matrix, css_weight=css_weight)
        paulis_by_css_weight = matrix is phased and css_weight
        changes: list[tuple[dict[str, Any], bool]] = [
            ({"max_css_single_pairs": 0}, paulis_by_css_weight),
            ({"coset_completion_block_bits": 4}, not paulis_by_css_weight),
        ]
        for change, reads in changes:
            orders.clear()
            with use_tuning(**change):
                CosetLeaderSearch(matrix, css_weight=css_weight)
            assert bool(orders) == reads


def test_oracles_build_other_families_with_the_tuning_they_are_built_with(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Costly queries build their families and tables with the tuning of the oracle that answers.

    With the threshold of the gate at zero, the first query that searches builds the other families
    of the search (see ``coset_routes``).  The most sets of the overlapping family and the column
    orders that it tries for each set reach ``information_sets.get_overlapping_family``, and the
    width of the blocks of completion tables and their limit of bytes reach
    ``completion_bounds.build_tables_within``, which the test wraps to record them.  They are those
    of the tuning in force when the oracle was built, not when it answers.
    """
    calls: list[tuple[int, int]] = []
    table_calls: set[tuple[int, int]] = set()
    get_overlapping_family = basis_families.get_overlapping_family
    build_tables_within = completion_bounds.build_tables_within
    signature = inspect.signature(get_overlapping_family)

    def record(*args: Any, **kwargs: Any) -> basis_families.BasisFamily:
        policy = signature.bind(*args, **kwargs).arguments["policy"]
        calls.append((policy.max_sets, policy.attempts_per_set))
        return get_overlapping_family(*args, **kwargs)

    def record_tables(
        rows: npt.NDArray[np.uint64], block_bits: int, max_bytes: int
    ) -> completion_bounds.CompletionTables | None:
        table_calls.add((block_bits, max_bytes))
        return build_tables_within(rows, block_bits, max_bytes)

    monkeypatch.setattr(basis_families, "get_overlapping_family", record)
    monkeypatch.setattr(completion_bounds, "build_tables_within", record_tables)
    coset._get_pauli_search.cache_clear()
    with use_tuning(
        overlapping_minimum_work=0,
        max_information_sets=5,
        overlapping_attempts_per_set=3,
        coset_completion_block_bits=4,
        completion_bound_max_table_bytes=12345,
    ):
        oracle = CosetLeaderSearch(_ROUTED_PHYSICAL_STATES[1])
    for syndrome, max_weight in itertools.product(range(1 << 5), range(2, 5)):
        oracle.get_min_weight_bound(syndrome, max_weight)
    assert calls == [(5, 3)]
    assert table_calls == {(4, 12345)}


def test_oracles_under_other_gates_do_not_share_searches(tune: Tune) -> None:
    """An oracle answers as a fresh oracle under the threshold of its gate, whatever is cached.

    Oracles are built for a graph state, whose search of Pauli errors is cached by stabilizers, and
    for a CSS state whose pure-X stabilizers span a random code, whose two classical searches are
    cached by stabilizer group.  They are built under the threshold zero, then the largest value,
    then zero again.  Some answers differ between the thresholds, so an oracle that shared the
    search of another threshold would answer differently.
    """
    checks = get_kernel(OVERLAPPING_ROUTES_CODE)
    css_state = np.block(
        [
            [OVERLAPPING_ROUTES_CODE, np.zeros_like(OVERLAPPING_ROUTES_CODE)],
            [np.zeros_like(checks), checks],
        ]
    )
    for matrix in (_ROUTED_PHYSICAL_STATES[1], css_state):
        css_weight = matrix is css_state
        syndromes = range(0, 1 << len(matrix), max(1, (1 << len(matrix)) // 64))
        queries = list(itertools.product(syndromes, range(6)))
        answers = {}
        for minimum_work in (0, MAX_VALUE):
            coset._get_css_searches.cache_clear()
            coset._get_pauli_search.cache_clear()
            with use_tuning(overlapping_minimum_work=minimum_work):
                oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
                answers[minimum_work] = [oracle.get_min_weight_bound(*query) for query in queries]
        assert answers[0] != answers[MAX_VALUE]
        for minimum_work in (0, MAX_VALUE, 0):
            tune(overlapping_minimum_work=minimum_work)
            oracle = CosetLeaderSearch(matrix, css_weight=css_weight)
            answers_now = [oracle.get_min_weight_bound(*query) for query in queries]
            assert answers_now == answers[minimum_work]


def test_cached_pauli_searches_answer_for_their_own_stabilizers_and_weight() -> None:
    """Oracles built in turn answer exactly for their own stabilizers and weight.

    The product states with stabilizers ``Y__``, ``_Z_``, ``__X`` and ``X__``, ``_Y_``, ``__Z`` are
    not CSS, so their oracles search Pauli errors by either weight, and both have three qubits.  The
    generators ``Y__``, ``YZ_``, ``__X`` of the first state give other syndromes to the same errors.
    In the first state, ``Z0 X1`` has the syndrome ``0b011`` and CSS weight one, but no error with
    that syndrome has physical weight below two.  Oracles are built for every state and weight under
    three tunings of the setup of their searches, and queried only once all are built, so that each
    answers after every search is cached.
    """
    states = [
        _product_state(generators)
        for generators in (("Y__", "_Z_", "__X"), ("Y__", "YZ_", "__X"), ("X__", "_Y_", "__Z"))
    ]
    coset._get_pauli_search.cache_clear()
    oracles = []
    for changes in ({}, {"max_css_single_pairs": 0}, {"disjoint_random_column_orders": 3}):
        with use_tuning(**changes):
            for matrix, css_weight in itertools.product(states, (False, True)):
                oracles.append(
                    (matrix, css_weight, CosetLeaderSearch(matrix, css_weight=css_weight))
                )
    physical_bound = oracles[0][2].get_min_weight_bound(0b011, 1)
    assert not physical_bound.within_max_weight and physical_bound.lower_bound == 2
    css_bound = oracles[1][2].get_min_weight_bound(0b011, 1)
    assert css_bound.within_max_weight and css_bound.representative == 0b001010
    for matrix, css_weight, oracle in oracles:
        _assert_exact_answers(oracle, matrix, css_weight)


def test_cached_pauli_searches_ignore_later_changes_to_the_stabilizer_array(tune: Tune) -> None:
    """Oracles answer for the stabilizers that they were built from, when an array changes later.

    Oracles of a graph state are built from an array, which is then overwritten with another graph
    state, from which further oracles are built from the same array.  Oracles of the first state
    built again from a copy of it, and the oracles built before the change, still answer for the
    first state.  With the threshold of the gate at zero, the searches build their other families
    at their first queries, after the change (see ``coset_routes``).  The first state has a larger
    disjoint family by CSS weight, and its search by CSS weight reads its generators from the
    stabilizers themselves.
    """
    tune(overlapping_minimum_work=0)
    first, second = _ROUTED_CSS_WEIGHT_STATE, _ROUTED_PHYSICAL_STATES[1]
    coset._get_pauli_search.cache_clear()
    array = first.copy()
    oracles = [
        (first, css_weight, CosetLeaderSearch(array, css_weight=css_weight))
        for css_weight in (False, True)
    ]
    array[:] = second
    oracles += [
        (second, css_weight, CosetLeaderSearch(array, css_weight=css_weight))
        for css_weight in (False, True)
    ]
    oracles += [
        (first, css_weight, CosetLeaderSearch(first.copy(), css_weight=css_weight))
        for css_weight in (False, True)
    ]
    for matrix, css_weight, oracle in oracles:
        _assert_exact_answers(oracle, matrix, css_weight)


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
# its searches of them for minimum weights skip some completions by tables whose bound reaches the
# skip bound of that weight, but not the weight of the lightest word that the searches weigh.  The
# searches by tails of some of its cosets, such as those of syndromes 1126, 4578, and 6129, stop
# early within steps.
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


def _assert_classical_cosets_match_brute_force(generators: npt.NDArray[np.uint8]) -> None:
    """Check the classical search of the cosets of a code against the least weight of their words.

    Single queries and batches, at maximum weights from below zero to past the number of columns,
    agree with the least weight of the words of each syndrome.  From maximum weight two, they also
    agree with each other.  A coset without a word of weight at most the maximum weight gets a lower
    bound of at least the least weight above the maximum weight that a word of the coset can have.
    That weight has the parity of the coset's least weight if every row of ``generators`` has even
    weight.  Below maximum weight two, a single query of a coset without a word of weight at most
    one gets exactly that weight for maximum weight one, which it finds without a search.
    """
    num_columns = generators.shape[1]
    checks = get_kernel(generators)
    tuning = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    search = coset._get_classical_search(
        generators, checks, tuning._replace(disjoint_random_column_orders=0)
    )
    minima = _classical_minima(checks)
    even = not np.any(generators.sum(axis=1) % 2)

    def get_least_bound(syndrome: int, max_weight: int) -> int:
        """The least weight above the maximum weight that a word of the coset can have."""
        return max_weight + 1 + int(even and (max_weight + 1 - minima[syndrome]) % 2)

    syndromes = list(range(1 << len(checks)))
    packed_syndromes = _pack_syndromes(syndromes, len(checks))
    for max_weight in range(-1, num_columns + 2):
        bounds = {}
        for syndrome in syndromes:
            bound = search.get_min_weight_bound(syndrome, max_weight)
            representative = (bound.representative >> np.arange(num_columns)) & 1
            weight = int(representative.sum())
            assert bound.within_max_weight == (minima[syndrome] <= max_weight)
            assert _classical_syndrome(checks, representative) == syndrome
            assert bound.lower_bound <= minima[syndrome] <= weight
            if bound.within_max_weight:
                assert weight <= max_weight
            else:
                assert bound.lower_bound >= get_least_bound(syndrome, max_weight)
            if max_weight < 2 and minima[syndrome] >= 2:
                assert bound.lower_bound == get_least_bound(syndrome, 1)
            bounds[syndrome] = bound

        # Find each syndrome whose minimum exceeds the maximum weight with batches, each of which
        # starts after the last syndrome found.
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
            assert _classical_syndrome(checks, representative) == syndrome
            assert not batch_bound.within_max_weight
            assert get_least_bound(syndrome, max_weight) <= batch_bound.lower_bound
            assert batch_bound.lower_bound <= minima[syndrome]
            if max_weight >= 2:
                assert bounds[syndrome] == batch_bound
            found.append(syndrome)
            start += index + 1
        assert found == [syndrome for syndrome in syndromes if minima[syndrome] > max_weight]


def test_cosets_of_a_rank_deficient_code_match_brute_force() -> None:
    """The coset searches of a rank-deficient code match the minima of all words.

    The code has disjoint information sets of ranks five and four, so its searches skip the second
    set while it adds nothing to the bounds of their stopping steps.  Syndrome 19 has a word of
    weight two that the first set reaches only above layer one.  At maximum weight two, the stopping
    step is the one at which the second set completes layer one, which overcomes its deficiency of
    one, so the search visits that set.  It finds the word in layer zero of that set, its offset.
    Single queries and batches agree with the least weight of the words of each syndrome (see
    ``_assert_classical_cosets_match_brute_force``).  Below two, a single query of a coset without a
    word of weight at most one returns a bound of two without searching.
    """
    _assert_classical_cosets_match_brute_force(_RANK_DEFICIENT_CODE)


def test_cosets_of_an_even_code_have_bounds_of_their_parity() -> None:
    """Coset searches of an even code give each coset with no light word a bound of its parity.

    The code is ``_RANK_DEFICIENT_CODE`` with a last column that holds the parity of each row, so
    every word of the code has even weight, and the weights of the words of each coset share a
    parity.  A coset without words of weight at most ``m`` then has no word of weight ``m + 1`` if
    that integer has the other parity, so its lower bound is at least the least integer above ``m``
    of its parity (see ``_assert_classical_cosets_match_brute_force``).  The bounds of the searches,
    which round up to the parity of each coset, reach it.  So does the bound three of an odd coset
    of least weight three in a single query below maximum weight two, which does not search.  Bounds
    without parity would fall short of both.  The cosets have both parities, and least weights up to
    four.
    """
    parities = (_RANK_DEFICIENT_CODE.sum(axis=1, keepdims=True) % 2).astype(np.uint8)
    _assert_classical_cosets_match_brute_force(np.hstack([_RANK_DEFICIENT_CODE, parities]))


@pytest.mark.parametrize(
    "code", [EVEN_ROUTES_CODE, TWO_FAMILIES_CODE], ids=["even", "two-families"]
)
def test_cosets_match_brute_force_with_other_families(
    code: npt.NDArray[np.uint8], tune: Tune
) -> None:
    """Searches of X or Z parts answer exactly when costly queries take other families.

    With the threshold of the gate at zero, queries that the disjoint family would answer with more
    work take another family (see ``coset_routes``).  The even code takes a disjoint family with
    two information sets, for cosets of both parities.  The random code takes a disjoint family with
    a larger partial set, and a family of overlapping information sets.  Single queries and
    batches, at maximum weights from below zero to past the number of columns, and so just below
    and at the least weight of each coset, agree with the least weight of the words of each coset
    (see ``_assert_classical_cosets_match_brute_force``).  At the threshold zero, the queries also
    run the searches by tails.
    """
    tune(overlapping_minimum_work=0)
    _assert_classical_cosets_match_brute_force(code)


def test_answers_do_not_depend_on_earlier_queries(pytestconfig: pytest.Config, tune: Tune) -> None:
    """The answers of a search depend only on its code, the syndrome, and the maximum weight.

    With the threshold of the gate at zero, fresh searches of the cosets of a random code answer
    every query in increasing, decreasing, and random order of the maximum weight.  Each order
    reaches the threshold at a different query, but the overlapping family of the code, which
    depends on the weight at which it is built, and so the answers, are the same.  With the largest
    threshold, some answers differ, so the other families answer some of the queries.
    """
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

    tune(overlapping_minimum_work=0)
    answers = [get_answers(order) for order in ("up", "down", "random")]
    assert answers[0] == answers[1] == answers[2]
    with use_tuning(overlapping_minimum_work=MAX_VALUE):
        assert get_answers("up") != answers[0]


def test_searches_visit_only_the_sets_that_add_to_their_bounds(pytestconfig: pytest.Config) -> None:
    """A search skips exactly the sets that add nothing to the bound of its stopping step.

    A set of rank ``r`` in a code of dimension ``k`` adds to the bound of a step only once it has
    completed layer ``k - r`` (see ``information_sets.get_step_lower_bounds``).  A search that stops
    at step ``(w, j)`` completes layer ``w`` on the first ``j`` sets and layer ``w - 1`` on the
    others, so it visits a set only if that layer is at least ``k - r``.  The sets are the disjoint
    information sets of random codes, whose last one is often rank deficient, and a set of rank zero
    with every row of the code.  A search whose answer is no weighs every word of the layers of the
    sets that it visits, and every answer is exact.  The exact number of words weighed guards the
    documented rule for skipping sets, so a legitimate future pruning of the coset scan would
    change it.  The work of the plan of the steps of the search, from layer zero and without the
    sets that it skips (see ``information_sets.plan_steps``), predicts this number.
    """
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
            certificate=certificates_module.Certificate(num_rows, ranks),
        )
        inverse = _cosets._get_inverse(checks)
        tables = coset_tables._get_search_tables(instance, inverse, num_columns)
        minima = _classical_minima(checks)

        for syndrome, max_weight in itertools.product(
            range(1 << len(checks)), range(num_columns + 1)
        ):
            best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
            packed_syndrome = _pack_syndromes([syndrome], len(checks))[0]
            best_weight, parity = coset._get_syndrome_word(
                packed_syndrome, best, tables.inverse, tables.x_mask, 1
            )
            if best_weight <= max_weight:
                continue
            table = coset_tables._get_planning_table(tables.planning, parity)
            stop_layer, stop_num_completed, stop_bound = coset_tables._get_stopping_step(
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
                parity if len(tables.planning) == 2 else -1,
            )
            representative = unpack_bits(best[None], num_columns)[0].astype(int)
            assert _classical_syndrome(checks, representative) == syndrome
            assert (lower_bound <= max_weight) == (minima[syndrome] <= max_weight)
            if lower_bound > max_weight:
                assert lower_bound <= minima[syndrome]
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
                binomials = certificates_module.get_capped_binomials(num_rows)
                plan = certificates_module.plan_steps(
                    table[:-1], max_weight, 0, num_rows, instance.ranks, binomials, True
                )
                assert plan == (stop_layer, stop_num_completed, weighed)


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

    The family holds a set of the embedded generators reduced on each block of consecutive columns,
    whose sizes cycle through ``block_sizes``, and a set of rank zero with every generator.  The
    blocks are disjoint, so the certificate of the ranks of the sets bounds the words of each coset.
    Every column counts as the X part.  The lower bound of the search of whole words agrees with the
    minimum of the coset in ``minima``, if given.  Without ``block_bits``, for each syndrome and
    maximum weight, the two searches from the word of the syndrome give the same lower bound, word,
    and number of words weighed.  For each maximum weight, the two batch loops find the same
    syndromes, in turn, with the same bounds and words.

    With ``block_bits``, every set of at least two rows has completion tables on blocks of that many
    columns, and the searches by tails decide as the searches of whole words, and give the same word
    to a coset with a word within the maximum weight.  To another coset, they give the same lower
    bound, or the least weight beyond the maximum weight that a word of the coset can have if they
    skipped any word, and a word of at least that weight.  This least weight is one more than the
    maximum weight, or two more if the generators are even and the words of the coset have the
    parity of the maximum weight.  They weigh every word that the search of whole words weighs, but
    the words that they skip, which ``_count_skipped_supports`` counts.  The batch loops with tables
    find the same syndromes, with the bounds and words of single queries with tables.

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
        certificate=certificates_module.Certificate(
            num_rows, tuple(int(np.sum(row >= 0)) for row in pivots)
        ),
        weight_factor=2 if scale == 2 else 1,
    )
    tables = coset_tables._get_search_tables(instance, inverse, num_columns)
    tails = coset_tables._get_coset_tails(
        instance,
        list(range(len(sets))) if block_bits else [],
        None,
        inverse=inverse,
        block_bits=block_bits or 8,
        max_table_bytes=MAX_VALUE,
    )
    even = scale == 1 and not np.any(generators.sum(axis=1) % 2)
    packed_syndromes = _pack_syndromes(syndromes, len(inverse))
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
            stop_layer, stop_num_completed, stop_bound = coset_tables._get_stopping_step(
                coset_tables._get_planning_table(tables.planning, parity), max_weight
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
                parity if len(tables.planning) == 2 else -1,
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
                parity if len(tables.planning) == 2 else -1,
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
                # The skip bound is one more than the greatest Hamming weight at most
                # ``scale * max_weight`` that a word of the coset can have.
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

        # Find each syndrome without a light enough word with both batch loops, in turn.
        start = 0
        while True:
            words = np.empty((2, tables.inverse.shape[1]), dtype=np.uint64)
            found = coset._find_first_not_within_max_weight(
                packed_syndromes[start:], max_weight, words[0], *tables, None, scale
            )
            found_by_tails = coset._find_first_not_within_max_weight(
                packed_syndromes[start:], max_weight, words[1], *tables, tails, scale
            )
            assert found_by_tails[0] == found[0]
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

    ``rows`` are the rows of the set, ``pivots`` their pivot columns, padded with -1 beyond the
    rank, and ``word`` the word of a syndrome, which the rows of its pivot ones reduce to the
    offset.  A support of a layer is skipped if the pivot ones of the rows of its prefix below the
    rank, the fewest pivot ones of a pair of rows from its row ``s`` on, and the least weight of the
    tail of such a completion on each block of columns add up to at least ``skip_bound``.  The tails
    hold the columns other than the pivots, and the blocks cover them up to the last on which a row
    has a one.  A set whose tails have no ones has no tables, and skips nothing.
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


@pytest.mark.parametrize("block_bits", [None, 4])
def test_searches_by_tails_answer_as_searches_of_whole_words(
    block_bits: int | None, pytestconfig: pytest.Config
) -> None:
    """Coset searches by tails decide as searches of whole words, and match them without tables.

    The searches by tails weigh a word by its rows below the rank of its set and its tail, which
    holds every column but the pivot columns of the set.  Without completion tables, they visit the
    same words, and keep the same lightest word, as the searches of whole words (see
    ``_assert_searches_by_tails_match_searches_of_words``).  With tables on blocks of four columns,
    they skip words beyond the maximum weight, which changes only the answers to cosets without a
    word within it.  Blocks of as many columns as the generators have rows alternate with blocks of
    one column fewer, whose sets are rank deficient, so that the tails of the sets differ in their
    numbers of columns.  The random codes have a column of zeros, which a coset can still have ones
    on.  With tables, each is also made even by a column of the parities of its rows, so that the
    searches round their bounds to the parity of each coset.  A rank-deficient code made even in the
    same way is searched too.  The searches of every syndrome of these codes, at every maximum
    weight up to past their length, agree with the minima of the cosets.  A wide code begins with an
    information set, whose tails fit in one machine word, while the tails of its rank-deficient sets
    take two, and a sample of its syndromes is searched.  The words of the Pauli errors of a graph
    state that is not CSS are weighed by their physical weight, and with tables, its family has sets
    of ranks two and one.  With tables, the searches of the even rank-deficient code, the wide code,
    and the graph state skip words; those of small random codes need not.
    """
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
                        _cosets._get_inverse(checks),
                        1,
                        (num_rows, num_rows - 1),
                        range(-1, code.shape[1] + 2),
                        range(1 << len(checks)),
                        _classical_minima(checks),
                    ),
                    False,
                )
            )
    if block_bits:
        parities = (_RANK_DEFICIENT_CODE.sum(axis=1, keepdims=True) % 2).astype(np.uint8)
        even = np.hstack([_RANK_DEFICIENT_CODE, parities])
        checks = get_kernel(even)
        searches.append(
            (
                (
                    even,
                    _cosets._get_inverse(checks),
                    1,
                    (5, 4),
                    range(-1, even.shape[1] + 2),
                    range(1 << len(checks)),
                    _classical_minima(checks),
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
                _cosets._get_inverse(checks),
                1,
                (4, 3),
                range(-1, 40, 4),
                syndromes,
            ),
            True,
        )
    )

    matrix = _graph_state(4, [(0, 1), (1, 2)], phase=True)
    checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
    searches.append(
        (
            (
                symplectic_to_hamming(matrix),
                symplectic_to_hamming(_cosets._get_inverse(checks)),
                2,
                (2, 1) if block_bits else (4, 3),
                range(-1, 6),
                range(16),
                _css_minima(matrix, css_weight=False),
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
    """Single queries and batches of routes by tails weigh words by the tails of their record.

    At the threshold zero, every route of the classical search of the repetition code of length
    seven is by tails.  A record whose tails are all zero makes every offset weigh nothing, so the
    queries at maximum weight two find a light enough word in every coset, even in the cosets whose
    words weigh at least three, which a search of whole words would reject.
    """
    get_coset_tails = coset_tables._get_coset_tails

    def get_zero_tails(*args: Any, **kwargs: Any) -> coset_tables._CosetTails:
        """The record of the tails of a family, with every tail zero."""
        record = get_coset_tails(*args, **kwargs)
        return record._replace(
            tails=np.zeros_like(record.tails), inverse_tails=np.zeros_like(record.inverse_tails)
        )

    monkeypatch.setattr(coset, "_get_coset_tails", get_zero_tails)
    generators = np.ones((1, 7), dtype=np.uint8)
    checks = get_kernel(generators)
    tuning = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    search = coset._get_classical_search(
        generators, checks, tuning._replace(overlapping_minimum_work=0)
    )
    syndromes = np.flatnonzero(_classical_minima(checks) > 2).tolist()
    assert syndromes
    for syndrome in syndromes:
        bound = search.get_min_weight_bound(syndrome, 2)
        assert (bound.within_max_weight, bound.lower_bound) == (True, 2)
    packed_syndromes = _pack_syndromes(syndromes, len(checks))
    assert search.find_first_not_within_max_weight(packed_syndromes, 2) == (len(syndromes), None)


def test_queries_of_routes_by_tails_skip_completions_by_the_tables_of_their_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single queries and batches of routes by tails skip completions by the tables of their record.

    At the threshold zero, every route of the classical search of a random code is by tails.  A
    record whose tables give every completion the bound 255 makes each query skip every completion
    in its layers of two or more rows.  At maximum weight three, some coset then gets no word within
    that weight, although it has one.  The searches skipped words, so single queries and batches
    bound such cosets by four.  The code is not even, so four is the least weight beyond the maximum
    weight that a word of such a coset can have.
    """
    get_coset_tails = coset_tables._get_coset_tails

    def get_skipping_tails(*args: Any, **kwargs: Any) -> coset_tables._CosetTails:
        """The record of the tails of a family, with tables that skip every completion."""
        record = get_coset_tails(*args, **kwargs)
        return record._replace(completion_tables=np.full_like(record.completion_tables, 255))

    monkeypatch.setattr(coset, "_get_coset_tails", get_skipping_tails)
    checks = get_kernel(OVERLAPPING_ROUTES_CODE)
    tuning = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    search = coset._get_classical_search(
        OVERLAPPING_ROUTES_CODE, checks, tuning._replace(overlapping_minimum_work=0)
    )
    minima = _classical_minima(checks)
    missed = [
        syndrome
        for syndrome in np.flatnonzero(minima <= 3).tolist()
        if not search.get_min_weight_bound(syndrome, 3).within_max_weight
    ]
    assert missed
    assert {search.get_min_weight_bound(syndrome, 3).lower_bound for syndrome in missed} == {4}
    _, bound = search.find_first_not_within_max_weight(_pack_syndromes(missed, len(checks)), 3)
    assert bound is not None and bound.lower_bound == 4


def _coset_minima(
    generators: npt.NDArray[np.uint8], checks: npt.NDArray[np.uint8], syndromes: Sequence[int]
) -> npt.NDArray[np.int_]:
    """The least weight of the words of the coset of each syndrome, from every word of the coset.

    A word with the syndrome sums rows of a right inverse of the checks, from the module under test,
    so the helper checks the syndromes of these words.  The coset adds to such a word every word of
    the code of ``generators``.
    """
    code = np.array(list(itertools.product([0, 1], repeat=len(generators)))) @ generators % 2
    bits = (np.array(syndromes)[:, None] >> np.arange(len(checks))) & 1
    words = bits @ _cosets._get_inverse(checks) % 2
    assert np.array_equal(_classical_syndrome(checks, words), syndromes)
    return np.array([((word + code) % 2).sum(axis=1).min() for word in words])


def _assert_min_weights_match_brute_force(
    search: coset._CosetSearch,
    syndromes: Sequence[int],
    minima: npt.NDArray[np.int_],
    num_syndrome_bits: int,
    rng: np.random.Generator,
) -> None:
    """Check the minimum weights that a search finds for cosets against their least weights.

    One batch holds the syndromes, in random order, and ``minima`` holds the least weight of the
    coset of each.  At each maximum weight from zero to one past the largest least weight, the
    floors of the batch are zero, then random up to the maximum weight, and then that weight.  A
    coset whose least weight is at most its floor gets a weight from its least weight to its floor.
    Any other coset gets its least weight, or one more than the maximum weight if that is smaller.
    """
    order = rng.permutation(len(syndromes))
    packed_syndromes = _pack_syndromes([syndromes[index] for index in order], num_syndrome_bits)
    least = minima[order]
    for max_weight in range(int(least.max()) + 2):
        random_floors = rng.integers(0, max_weight + 1, len(least))
        for floors in (np.zeros_like(least), random_floors, np.full_like(least, max_weight)):
            weights = search.get_min_weights(packed_syndromes, max_weight, floors.astype(np.int64))
            exact = least > floors
            assert np.array_equal(weights[exact], np.minimum(least, max_weight + 1)[exact])
            assert np.all(exact | ((least <= weights) & (weights <= floors)))


@pytest.mark.parametrize(
    "tuning",
    [
        {},
        {"overlapping_minimum_work": 0},
        {"overlapping_minimum_work": 0, "completion_bound_max_table_bytes": 0},
    ],
    ids=["disjoint", "tables", "no-tables"],
)
def test_min_weights_of_cosets_match_brute_force(
    tuning: dict[str, int], pytestconfig: pytest.Config, tune: Tune
) -> None:
    """Searches for the minimum weights of cosets agree with the least weights of their words.

    The codes are the rank-deficient code, a code whose searches skip words by tables, and a random
    code, each also made even by a column of the parities of its rows.  The searches of the even
    codes round to the parity of each coset, and their incumbents are one below the Hamming weight
    of their best words.  The cosets have both parities.  Three cosets of a longer code are
    searched too.  Their searches by tails stop early within steps.  The searches take the disjoint
    family of each code, and at the threshold zero other families by tails, with completion tables
    and without them (see ``_assert_min_weights_match_brute_force``).  The least weights come from
    every word of each coset.  The Pauli errors of a graph state that is not CSS are searched by
    CSS weight too, with the disjoint family and at the threshold zero, and their least weights come
    from every Pauli.  These searches never weigh words by tails, so they have no completion tables.
    """
    tune(**tuning)
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    tuning_fields = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=False)
    num_rows, num_columns = int(rng.integers(3, 6)), int(rng.integers(8, 11))
    random_code = get_random_full_rank_matrix(rng, num_rows, num_columns)
    for code in (_RANK_DEFICIENT_CODE, _SKIPPING_CODE, random_code):
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
    if "completion_bound_max_table_bytes" in tuning:
        return
    matrix = _graph_state(4, [(0, 1), (1, 2), (2, 3)], phase=True)
    checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
    css_fields = coset._CosetTuning.from_tuning(get_tuning(), paulis_by_css_weight=True)
    search = coset._get_pauli_search(coset._StabilizerKey(matrix, checks), True, css_fields)
    syndromes = range(1 << len(matrix))
    _assert_min_weights_match_brute_force(search, syndromes, _css_minima(matrix), len(matrix), rng)


@pytest.mark.parametrize("tuning", [{}, {"overlapping_minimum_work": 0}], ids=["words", "tails"])
def test_min_weight_searches_above_max_weight_examine_as_queries(
    tuning: dict[str, int], tune: Tune
) -> None:
    """A search for the minimum weight of a coset without a word within a weight works as a query.

    Until it finds a word within ``max_weight``, the search keeps the skip bound of ``max_weight``
    as its incumbent, at which a query at ``max_weight`` skips completions too.  It stops at the
    stopping step of that query, so it weighs and skips the same words.  A code and its even version
    are searched at every syndrome, a longer code at three syndromes of minimum weight six, and the
    Pauli errors of a graph state by physical weight at every syndrome.  They are searched by whole
    words, and at the threshold zero by tails, with tables, which skip words of each of the three
    codes but need not skip any of the graph state.  The minimum weights come from every word of
    each coset.
    """
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
    matrix = _ROUTED_PHYSICAL_STATES[1]
    checks = (math.symplectic_conjugate(matrix.astype(int)) % 2).astype(np.uint8)
    search = coset._get_pauli_search(coset._StabilizerKey(matrix, checks), False, tuning_fields)
    syndromes = list(range(1 << len(matrix)))
    cases.append((search, len(matrix), syndromes, _css_minima(matrix, css_weight=False), False))

    for search, num_syndrome_bits, syndromes, minima, must_skip in cases:
        skipped = 0
        for max_weight in range(2, int(max(minima))):
            tables, tails = search._routes.get_route(max_weight)
            for syndrome, minimum in zip(syndromes, minima, strict=True):
                if minimum <= max_weight:
                    continue
                packed_syndrome = _pack_syndromes([syndrome], num_syndrome_bits)[0]
                best = np.empty(tables.inverse.shape[1], dtype=np.uint64)
                best_weight, parity = coset._get_syndrome_word(
                    packed_syndrome, best, tables.inverse, tables.x_mask, search.scale
                )
                stop_layer, stop_num_completed, stop_bound = coset_tables._get_stopping_step(
                    coset_tables._get_planning_table(tables.planning, parity),
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
                    parity if len(tables.planning) == 2 else -1,
                )
                _, *found = coset._search_coset_for_min_weight(
                    packed_syndrome, max_weight, 0, best, *tables, tails, search.scale
                )
                skipped += expected[1]
                assert found == expected
        assert skipped or not (tuning and must_skip)


def test_tables_of_pairs_of_single_qubit_errors_respect_their_limit() -> None:
    """A table of products of X and Z errors keeps the first product of each syndrome.

    Three entries of X errors and two of Z errors, each counting the identity, make six pairs, which
    fit a limit of six but not of five.
    """
    x_singles = {0: 0, 0b01: 0b0001, 0b11: 0b0010}
    z_singles = {0: 0, 0b10: 0b0100}
    table = coset._tabulate_css_singles(x_singles, z_singles, 6)
    assert table == {0: 0, 0b10: 0b0100, 0b01: 0b0001, 0b11: 0b0101}
    assert coset._tabulate_css_singles(x_singles, z_singles, 5) is None
