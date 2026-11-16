# SPDX-License-Identifier: Apache-2.0

"""Tests of the search of the supports that are connected in a Tanner graph, and its queries."""

from __future__ import annotations

import inspect
import itertools
import unittest.mock
from collections.abc import Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y

from qldpc import codes
from qldpc._binary import get_independent_rows, get_kernel
from qldpc.objects import PAULIS_XZ, Pauli

from .. import _tuning
from ..conftest import get_span
from ..core import get_distance_css, get_distance_pauli
from . import search as cs

Problem = tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]

# The [[276, 98, 14]] pair-partition code qc_276_98_14 of arXiv:2607.14091, with its exponents
# normalized so that their first row and column are zero.
EXPONENTS_X_276 = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 10, 16, 12, 20, 9, 6, 11, 2, 5, 21, 22],
    [0, 7, 20, 11, 3, 12, 2, 22, 14, 19, 13, 6],
    [0, 19, 17, 4, 2, 21, 12, 10, 18, 3, 11, 9],
]
EXPONENTS_Z_276 = [
    [0, 11, 18, 6, 13, 1, 0, 11, 18, 6, 13, 1],
    [0, 12, 7, 22, 19, 10, 7, 10, 12, 19, 0, 22],
    [0, 5, 6, 1, 4, 11, 1, 0, 4, 5, 11, 6],
    [0, 8, 22, 21, 12, 20, 12, 22, 0, 20, 21, 8],
]


def get_sectors(code: codes.CSSCode) -> list[Problem]:
    """The X and Z sectors of a code, built independently of ``get_distance_css``."""
    return [
        (
            np.asarray(code.get_stabilizer_ops(Pauli.Z if pauli is Pauli.X else Pauli.X), np.uint8),
            np.asarray(code.get_matrix(pauli), dtype=np.uint8),
        )
        for pauli in PAULIS_XZ
    ]


def get_golay_problem() -> Problem:
    """A sector of the CSS code of the cyclic [23, 12, 7] Golay code, which contains its dual.

    Its checks and excluded rows are both a basis of the dual, so cyclic shifts are automorphisms.
    """
    generator = np.zeros(23, dtype=np.uint8)
    generator[[0, 1, 5, 6, 7, 9, 11]] = 1
    checks = get_kernel(np.array([np.roll(generator, shift) for shift in range(12)]))
    return checks, checks


def get_brute_force_distance(problem: Problem) -> int:
    """The least weight of a vector in the kernel of the checks and outside the excluded rows."""
    checks, excluded = problem
    return min(word.bit_count() for word in get_span(get_kernel(checks)) - get_span(excluded))


def is_witness(problem: Problem, support: npt.NDArray[np.int64]) -> bool:
    """Whether a sorted support is that of a vector in the kernel, outside the excluded rows."""
    checks, excluded = problem
    vector = np.zeros(checks.shape[1], dtype=np.uint8)
    vector[support] = 1
    return (
        list(support) == sorted(set(support))
        and not np.any(checks @ vector % 2)
        and len(get_independent_rows(np.vstack([excluded, vector])))
        > len(get_independent_rows(excluded))
    )


def get_search(*problem: npt.ArrayLike, **options: Any) -> cs.SupportSearch:
    """A search that asks for any automorphisms before its first weight limit, by default.

    ``restrict_viable_symbols`` defaults to its value in the default tuning.
    """
    options.setdefault("automorphism_minimum_states", 0)
    options.setdefault("restrict_viable_symbols", _tuning.DEFAULT_TUNING.restrict_viable_symbols)
    return cs.SupportSearch(*problem, **options)


def get_symplectic_witness_supports(problem: Problem) -> list[int]:
    """The supports of the witnesses of a symplectic problem, with bit ``j`` for qubit ``j``."""
    checks, excluded = problem
    num_qubits = checks.shape[1] // 2
    words = get_span(get_kernel(np.roll(checks, num_qubits, axis=1))) - get_span(excluded)
    return [(word | word >> num_qubits) % 2**num_qubits for word in words]


def get_symplectic_vector(symbols: npt.NDArray[np.int64], num_qubits: int) -> npt.NDArray[np.uint8]:
    """The symplectic vector of the symbols of a witness, which give each qubit X, Z, or Y."""
    vector = np.zeros(2 * num_qubits, dtype=np.uint8)
    for symbol in symbols:
        value, qubit = divmod(int(symbol), num_qubits)
        vector[qubit] ^= value in (0, 2)
        vector[num_qubits + qubit] ^= value in (1, 2)
    return vector


def is_symplectic_witness(problem: Problem, symbols: npt.NDArray[np.int64]) -> bool:
    """Whether sorted symbols on distinct qubits give a Pauli operator that is a witness."""
    checks, excluded = problem
    num_qubits = checks.shape[1] // 2
    vector = get_symplectic_vector(symbols, num_qubits)
    return (
        list(symbols) == sorted(symbols)
        and len({int(symbol) % num_qubits for symbol in symbols}) == len(symbols)
        and not np.any(np.roll(checks, num_qubits, axis=1) @ vector % 2)
        and len(get_independent_rows(np.vstack([excluded, vector])))
        > len(get_independent_rows(excluded))
    )


# Sectors of small codes: the Steane code has words of odd weight, the toric code several lightest
# logical operators, the Z sector of the surface code has checks lighter than its distance, and the
# Reed-Muller code only words of even weight.
PROBLEMS: dict[str, Problem] = {
    "steane-X": get_sectors(codes.SteaneCode())[0],
    "toric-X": get_sectors(codes.ToricCode(4, rotated=True))[0],
    "surface-Z": get_sectors(codes.SurfaceCode(3))[1],
    "reed-muller-X": get_sectors(codes.QuantumReedMullerCode(1, 4))[0],
}

# The checks have the words {0, 1, 2} and {3, 4}, and their sum, so the lightest word, {3, 4}, has
# the greater least variable.
PROBLEMS["components"] = (
    np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=np.uint8),
    np.zeros((0, 5), dtype=np.uint8),
)

# Variables 0 and 1 are in all 43 checks, and every other variable in one, so the distance is 2, and
# the scale lcm(1, ..., 43) of the weights of checks exceeds 2**63, which the search caps.
PROBLEMS["capped-scale"] = (
    np.hstack([np.ones((43, 2), dtype=np.uint8), np.eye(43, dtype=np.uint8)]),
    np.zeros((0, 45), dtype=np.uint8),
)

# Nested states restrict their subtrees, and could allow more variables in total than there are
# variables, so the search makes more room for them than it starts with.
PROBLEMS["many-allowances"] = (
    np.array(
        [
            [1, 1, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 1, 1, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=np.uint8,
    ),
    np.zeros((0, 9), dtype=np.uint8),
)


# In searches of "empty-restriction", states restrict their subtrees to no variable, and in those of
# "nested-restrictions", the restrictions of states nest.
PROBLEMS["empty-restriction"] = (
    np.array(
        [
            [0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 0, 1, 1],
            [1, 1, 1, 0, 1, 0, 0, 0, 1, 0],
        ],
        dtype=np.uint8,
    ),
    np.zeros((0, 10), dtype=np.uint8),
)
PROBLEMS["nested-restrictions"] = (
    np.array(
        [
            [1, 0, 0, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 1, 1, 1, 0, 1, 0],
            [1, 1, 1, 1, 0, 0, 0, 1, 1],
            [0, 1, 1, 1, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [0, 1, 1, 1, 1, 0, 1, 1, 1],
            [0, 1, 0, 1, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    ),
    np.zeros((0, 9), dtype=np.uint8),
)

# The problems that tests compare with an enumeration of their words.
SEARCHED_PROBLEMS = [name for name in PROBLEMS if name not in ("steane-X", "toric-X")]


def get_qubit_problem(code: codes.QuditCode) -> Problem:
    """The stabilizers and gauge generators of a qubit code, as symplectic matrices."""
    checks = code.get_stabilizer_ops() if code.is_subsystem_code else code.matrix
    return np.asarray(checks, dtype=np.uint8), np.asarray(code.matrix, dtype=np.uint8)


# The [[8, 3, 3]] code of Gottesman (quant-ph/9604038), every word of which has an even number of
# nonzero bits, though its distance is odd.
EIGHT_QUBIT_CODE = codes.QuditCode.from_strings(
    ["XXXXXXXX", "ZZZZZZZZ", "IXIXYZYZ", "IXZYIXZY", "IYXZXZIY"]
)

# Small codes, whose qubits a test mixes with single-qubit Cliffords.  The Bacon-Shor code is a
# subsystem code.
SMALL_QUBIT_CODES = [
    codes.FiveQubitCode(),
    codes.QuditCode(codes.SteaneCode().matrix),
    codes.QuditCode(codes.SurfaceCode(3).matrix),
    codes.QuditCode(codes.BaconShorCode(3).matrix, is_subsystem_code=True),
    EIGHT_QUBIT_CODE,
]

# Codes that are not CSS: the five-qubit code, the Bacon-Shor code with Hadamards on some of its
# qubits, which is a subsystem code, and the [[8, 3, 3]] code.
QUBIT_PROBLEMS = {
    "five-qubit": get_qubit_problem(codes.FiveQubitCode()),
    "bacon-shor-hadamards": get_qubit_problem(
        codes.QuditCode(
            codes.BaconShorCode(3).conjugated(range(0, 9, 2)).matrix, is_subsystem_code=True
        )
    ),
    "eight-qubit": get_qubit_problem(EIGHT_QUBIT_CODE),
}


@pytest.mark.parametrize("restrict_viable_symbols", [True, False])
@pytest.mark.parametrize("name", SEARCHED_PROBLEMS)
def test_the_distance_matches_enumeration(name: str, restrict_viable_symbols: bool) -> None:
    """The distance search, and a search through each weight limit, agree with an enumeration."""
    problem = PROBLEMS[name]
    distance = get_brute_force_distance(problem)
    tuning = _tuning.SearchTuning(restrict_viable_symbols=restrict_viable_symbols)
    assert cs.get_distance_connected_cluster([problem], cutoff=0, tuning=tuning) == distance
    for max_weight in range(1, distance + 1):
        search = get_search(*problem, restrict_viable_symbols=restrict_viable_symbols)
        witness = search.find_witness(max_weight)
        if max_weight < distance:
            assert witness is None
        else:
            assert (
                witness is not None and len(witness) <= max_weight and is_witness(problem, witness)
            )


@pytest.mark.parametrize("name", SEARCHED_PROBLEMS)
def test_each_root_finds_the_lightest_witnesses_that_it_roots(name: str) -> None:
    """A root finds a witness at the distance exactly if it is the least variable of a lightest one.

    Section 3.3 of ``connected_cluster.md`` proves this completeness.
    """
    problem = PROBLEMS[name]
    checks, excluded = problem
    words = get_span(get_kernel(checks)) - get_span(excluded)
    distance = min(word.bit_count() for word in words)
    least_variables = {
        (word & -word).bit_length() - 1 for word in words if word.bit_count() == distance
    }
    for root in range(checks.shape[1]):
        search = get_search(*problem)
        search._roots = np.array([root])
        witness = search.find_witness(distance)
        assert (witness is not None) == (root in least_variables)
        assert witness is None or (min(witness) == root and is_witness(problem, witness))


def test_each_support_is_visited_once_above_its_least_variable() -> None:
    """With every word excluded, a search visits the supports that grow from each root once."""
    # The checks {0, 1, 2} and {1, 2, 3} have the words {0, 1, 3}, {0, 2, 3}, and {1, 2}.  At limit
    # 4, root 0 visits {0}, {0, 1}, {0, 1, 2}, {0, 1, 3}, {0, 2}, and {0, 2, 3}, since its branch on
    # 1 forbids 1 for its sibling {0, 2}.  Root 1 visits {1} and {1, 2}, and roots 2 and 3 visit
    # only themselves.  At limit 2, {0, 1} and {0, 2} leave a check unsatisfied, so they would need
    # a third variable.
    checks = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=np.uint8)
    num_states = {}
    for max_weight in (2, 4):
        search = get_search(checks, get_kernel(checks))
        assert search.find_witness(max_weight) is None
        num_states[max_weight] = search.num_states
    assert num_states == {2: 5, 4: 6 + 2 + 1 + 1}


def test_automorphisms_reduce_the_roots_and_keep_the_distance() -> None:
    """Cyclic shifts of the Golay code leave one root and fewer states, and keep the distance."""
    problem = get_golay_problem()
    distance = get_brute_force_distance(problem)
    # The quantum Golay code has distance 7 (https://errorcorrectionzoo.org/c/qubit_golay).
    assert distance == 7
    shift = (np.arange(23) + 1) % 23
    plain = get_search(*problem)
    symmetric = get_search(*problem, get_automorphisms=lambda: shift[None])
    for max_weight in range(1, distance + 1):
        assert (plain.find_witness(max_weight) is None) == (max_weight < distance)
        witness = symmetric.find_witness(max_weight)
        assert (witness is None) == (max_weight < distance)
    assert witness is not None and is_witness(problem, witness)
    assert symmetric.num_states < plain.num_states


def test_each_root_excludes_the_orbits_before_its_own() -> None:
    """The search from the root of an orbit uses no variable of an earlier orbit."""
    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    order = np.random.default_rng(0).permutation(len(code))
    automorphisms = np.argsort(order)[code._get_qudit_automorphisms()[:, order]]
    problem = (get_sectors(code)[0][0][:, order], get_sectors(code)[0][1][:, order])

    # The orbits under translations are the two blocks of 36 qubits, which the shuffle interleaves.
    # Orbit 0 is the block of the first shuffled qubit.
    orbits = ((order >= 36) != (order[0] >= 36)).astype(int)
    roots = np.unique(orbits, return_index=True)[1]
    assert roots[1] < len(code) // 2

    for orbit, root in enumerate(roots):
        search = get_search(*problem)
        search._use_automorphisms(automorphisms)
        search._roots = np.array([root])
        witness = search.find_witness(8)
        assert witness is not None and is_witness(problem, witness)
        assert np.all(orbits[witness] >= orbit)


def test_each_root_ranks_lowest_in_its_orbit() -> None:
    """No variable of an orbit ranks below its root, as a word that needs them all shows."""
    # The checks join consecutive variables of two interleaved cycles, of 60 and 30 variables.  Each
    # cycle is a word, and an orbit of the shift along both cycles.
    labels = np.random.default_rng(5).permutation([0] * 59 + [1] * 30)
    cycles = [np.flatnonzero(np.append(0, labels) == label) for label in (0, 1)]
    checks = np.zeros((90, 90), dtype=np.uint8)
    shift = np.arange(90)
    for cycle in cycles:
        following = np.roll(cycle, -1)
        checks[cycle, cycle] = checks[cycle, following] = 1
        shift[cycle] = following
    search = get_search(
        checks, np.zeros((0, 90), dtype=np.uint8), get_automorphisms=lambda: shift[None]
    )
    witness = search.find_witness(30)
    assert witness is not None and list(witness) == list(cycles[1])


@pytest.mark.parametrize("minimum_states", [0, 30, _tuning.MAX_VALUE])
def test_automorphisms_are_found_once_by_the_threshold(minimum_states: int) -> None:
    """A search asks for automorphisms at most once, and no later than ``minimum_states`` states."""
    checks = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=np.uint8)
    found = []
    search = get_search(
        checks,
        get_kernel(checks),
        get_automorphisms=lambda: found.append(True),
        automorphism_minimum_states=minimum_states,
    )
    for max_weight in [1, 2, 3, 4, 4, 4]:
        reached = search.num_states >= minimum_states
        search.find_witness(max_weight)
        assert len(found) <= 1
        assert found or not reached
    assert bool(found) == (minimum_states < _tuning.MAX_VALUE)


def test_a_limit_without_states_does_not_ask_for_automorphisms() -> None:
    """After a first limit without states, the search predicts no states for the next limit."""
    found = []
    search = get_search(
        *QUBIT_PROBLEMS["five-qubit"],
        symplectic=True,
        get_automorphisms=lambda: found.append(True),
        automorphism_minimum_states=10**6,
    )
    search.find_witness(1)
    assert search.num_states == 0
    search.find_witness(2)
    assert not found


def test_a_cutoff_raises_the_first_limit_to_a_multiple_of_the_divisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A search asks for automorphisms before a first limit that a cutoff above the divisor sets."""
    # The words of this Reed-Muller code have even weights, and its distance is 4.
    problem = PROBLEMS["reed-muller-X"]
    limits = []
    find_witness = cs.SupportSearch.find_witness

    def record_limit(search: cs.SupportSearch, max_weight: int) -> npt.NDArray[np.int64] | None:
        limits.append(max_weight)
        return find_witness(search, max_weight)

    monkeypatch.setattr(cs.SupportSearch, "find_witness", record_limit)
    for cutoff, first_limit, num_calls in [(0, 2, 0), (7, 6, 1), (8, 8, 1)]:
        limits.clear()
        get_automorphisms = unittest.mock.Mock(return_value=None)
        with _tuning.use_tuning(automorphism_minimum_states=10**5):
            distance = cs.get_distance_connected_cluster(
                [problem], cutoff=cutoff, get_automorphisms=get_automorphisms
            )
        assert 4 <= distance <= max(4, cutoff)
        assert limits[0] == first_limit
        assert get_automorphisms.call_count == num_calls


def test_pruning_keeps_the_search_of_a_high_rate_code_small() -> None:
    """Pruning rules out witnesses of weight up to 8 in a [[276, 98, 14]] code in few states."""
    code = codes.PairPartitionCode(EXPONENTS_X_276, EXPONENTS_Z_276, lift_size=23)
    search = get_search(*get_sectors(code)[0], get_automorphisms=code._get_qudit_automorphisms)
    assert search.find_witness(8) is None

    # The bound on the states fails if states stop forbidding variables for their subtrees (Section
    # 3.4 of connected_cluster.md).  A heuristic that moves the count across it needs a new bound.
    assert search.num_states < 5000


def test_the_restriction_to_viable_symbols_can_be_switched_off() -> None:
    """Without restrict_viable_symbols, a search visits exactly one more state.

    The bound Y of Section 3.4 of connected_cluster.md still prunes that state.
    """
    # Variable 0 is in all four checks, whose only nonzero word, {1, 2, 3, 4}, has weight 4.  At
    # limit W = 3, the state S = {0} leaves every check unsatisfied, with weights 1/3, 1/2, 1/3, and
    # 1/3.  Their total Y = 3/2 exceeds W - |S| - 1 = 1, so the state allows only the variables that
    # repair at least Y - 1 = 1/2 (Section 3.4 of connected_cluster.md).  Variable 2, only in check
    # 0, repairs 1/3.  Check 0 then allows only variable 1, the fewest, so the search with the
    # restriction branches only to {0, 1}.  Without the restriction, each check has two available
    # variables, so the search branches on check 0, the first, whose available variables are 1 and
    # 2.  So only the search without the restriction visits {0, 2}.  That state leaves checks 1, 2,
    # and 3 unsatisfied with weight 1/2 each, and Y = 3/2 > W - 2 prunes it.  Both searches visit
    # {0, 1}, whose children the bound b1 of Section 3.4 skips.  From each later root, both visit
    # only the root, since the variables of lower rank are blocked.  {1} leaves checks 0, 2, and 3
    # unsatisfied with weight 1 each, and Y = 3 > W - 1 prunes it.  {2}, {3}, and {4} each leave a
    # check with no available variable.  If the search without the restriction stopped pruning by Y,
    # it would also visit a child of {1}, and the equality would fail.
    checks = np.array(
        [[1, 1, 1, 0, 0], [1, 0, 0, 1, 1], [1, 1, 0, 1, 0], [1, 1, 0, 0, 1]], dtype=np.uint8
    )
    excluded = np.zeros((0, 5), dtype=np.uint8)
    num_states = {}
    for restrict_viable_symbols in (True, False):
        search = get_search(checks, excluded, restrict_viable_symbols=restrict_viable_symbols)
        assert search.find_witness(3) is None
        num_states[restrict_viable_symbols] = search.num_states
    assert num_states[False] == num_states[True] + 1


def test_the_restriction_can_be_switched_off_at_the_start_of_a_syndrome_query() -> None:
    """Switching the restriction off also turns it off at the starting state of a syndrome query."""
    # The query of syndrome 1101 at limit W = 2 starts with checks 0, 1, and 3 unsatisfied.
    # Variable 2, in checks 0 and 3, has overlap 2, and variables 0 and 1 have overlap 1, so the
    # checks weigh 1/2, 1, and 1/2, and Y = 2.  Since 1 + Y > W, the restriction allows only the
    # variables that repair at least Y - 1 = 1: variable 1, of check 1, and variable 2, which
    # repairs checks 0 and 3 (Section 3.4 of connected_cluster.md).  Each check then has one allowed
    # variable, so the search branches on check 0, the first, and enters {2}.  Its child {1, 2}
    # leaves check 2 unsatisfied with no room left, so b1 skips it.  Without the restriction, check
    # 1 is the first check with the fewest available variables, one, so the search branches only to
    # {1}.  That state leaves three checks unsatisfied, and the one more variable that the limit
    # allows flips at most two, so b1 skips it.  The counts rely on ties going to the first check,
    # so a change of the tie-break needs a new instance.
    checks = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=np.uint8)
    search = get_search(checks, np.zeros((0, 4), dtype=np.uint8))
    packed = cs.pack_bits(np.array([[1, 1, 0, 1]], dtype=np.uint8))[0]
    num_states = {}
    for restrict_viable_symbols in (True, False):
        support, num_states[restrict_viable_symbols] = cs._search_supports(
            search._check_starts,
            search._check_symbols,
            search._symbol_masks,
            search._test_columns,
            np.array([-1], dtype=np.int64),
            search._ranks,
            search._inverse_overlaps,
            2,
            packed,
            True,
            restrict_viable_symbols,
        )
        assert not len(support)
    assert num_states == {True: 1, False: 0}


@pytest.mark.parametrize(
    ("changes", "expected"), [({}, True), ({"restrict_viable_symbols": False}, False)]
)
def test_connected_cluster_distance_searches_read_the_restriction_of_their_tuning(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, bool], expected: bool
) -> None:
    """Distance searches pass ``restrict_viable_symbols`` of their tuning to ``SupportSearch``."""
    settings: list[tuple[bool, bool]] = []
    initialize = cs.SupportSearch.__init__
    signature = inspect.signature(initialize)

    def record(*args: Any, **kwargs: Any) -> None:
        arguments = signature.bind(*args, **kwargs).arguments
        settings.append((arguments["symplectic"], arguments["restrict_viable_symbols"]))
        initialize(*args, **kwargs)

    # get_distance_css searches through core._search_css, and get_distance_pauli, whose searches are
    # symplectic, through core._search_checks.
    monkeypatch.setattr(cs.SupportSearch, "__init__", record)
    steane = codes.SteaneCode()
    with _tuning.use_tuning(**changes):
        get_distance_css(steane.matrix_x, steane.matrix_z, method="connected_cluster")
        get_distance_pauli(codes.FiveQubitCode().matrix, method="connected_cluster")
    assert set(settings) == {(False, expected), (True, expected)}


def test_wrong_automorphisms_are_rejected() -> None:
    """A permutation that does not map the kernel of the checks to itself raises an error."""
    swap = np.arange(7)[None]
    swap[0, :2] = [1, 0]
    search = get_search(*PROBLEMS["steane-X"], get_automorphisms=lambda: swap)
    with pytest.raises(ValueError, match="automorphism"):
        search.find_witness(3)


def test_sectors_are_searched_together(monkeypatch: pytest.MonkeyPatch) -> None:
    """A witness of weight 3 in one sector stops the search of the other, of distance 7, at 3."""
    sectors = get_sectors(codes.TetrahedralCode())
    assert [get_brute_force_distance(sector) for sector in sectors] == [7, 3]

    limits = []
    find_witness = cs.SupportSearch.find_witness

    def record_limit(search: cs.SupportSearch, max_weight: int) -> npt.NDArray[np.int64] | None:
        limits.append(max_weight)
        return find_witness(search, max_weight)

    monkeypatch.setattr(cs.SupportSearch, "find_witness", record_limit)
    assert cs.get_distance_connected_cluster(sectors, cutoff=0) == 3
    assert max(limits) == 3


def test_stepped_searches_advance_the_problem_with_the_lowest_lower_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each advance searches one limit of the problem with the lowest lower bound, until the end."""
    searches_made: list[cs.SupportSearch] = []
    limits: list[tuple[int, int]] = []
    init, find_witness = cs.SupportSearch.__init__, cs.SupportSearch.find_witness

    def recording_init(search: cs.SupportSearch, *args: Any, **kwargs: Any) -> None:
        init(search, *args, **kwargs)
        searches_made.append(search)

    def recording_find_witness(
        search: cs.SupportSearch, max_weight: int
    ) -> npt.NDArray[np.int64] | None:
        limits.append((searches_made.index(search), max_weight))
        return find_witness(search, max_weight)

    monkeypatch.setattr(cs.SupportSearch, "__init__", recording_init)
    monkeypatch.setattr(cs.SupportSearch, "find_witness", recording_find_witness)
    # The problems have the divisors 2 and 1, so choosing the lowest lower bound differs from turns.
    problems = [PROBLEMS["toric-X"], PROBLEMS["steane-X"]]
    searches = cs.ConnectedClusterSearches(problems, cutoff=0)
    lower_bounds = [search.weight_divisor for search in searches_made]
    assert lower_bounds == [2, 1]
    while not searches.finished:
        num_limits = len(limits)
        searches.advance()
        assert len(limits) == num_limits + 1
        index, max_weight = limits[-1]
        assert max_weight == lower_bounds[index] == min(lower_bounds)
        lower_bounds[index] += searches_made[index].weight_divisor
    assert searches.result == 3 and [max_weight for _, max_weight in limits] == [1, 2, 2, 3]

    # Once finished, the searches do nothing.
    num_limits = len(limits)
    searches.advance()
    assert len(limits) == num_limits

    # Problems without witnesses, and a known bound within the cutoff, leave nothing to search.
    trivial = (np.array([[1, 1]], dtype=np.uint8), np.array([[1, 1]], dtype=np.uint8))
    for searches in (
        cs.ConnectedClusterSearches([trivial], cutoff=0),
        cs.ConnectedClusterSearches(problems, cutoff=3, upper_bound=3),
    ):
        assert searches.finished and searches.result == 3
        searches.advance()
    assert len(limits) == num_limits


def test_the_logical_test_reads_every_block() -> None:
    """The logical test finds a witness that fails only tests past the first block of 64."""
    problem = PROBLEMS["steane-X"]
    search = get_search(*problem)
    zeros = np.zeros((search.num_variables, 1), dtype=np.uint64)
    search._test_columns = np.hstack([zeros, search._test_columns])
    witness = search.find_witness(3)
    assert witness is not None and is_witness(problem, witness)


def get_random_qubit_problems(rng: np.random.Generator, num_problems: int) -> Iterator[Problem]:
    """Small codes with permuted qubits and a random single-qubit Clifford on each qubit.

    The Cliffords make the checks act as X, Z, and Y.
    """
    cliffords = [
        np.array(clifford) for clifford in itertools.permutations([[1, 0], [0, 1], [1, 1]], 2)
    ]
    for index in range(num_problems):
        code = SMALL_QUBIT_CODES[index % len(SMALL_QUBIT_CODES)]
        matrix = np.array(code.matrix, dtype=int)
        num_qubits = len(code)
        for qubit in range(num_qubits):
            clifford = cliffords[rng.integers(len(cliffords))]
            columns = np.stack([matrix[:, qubit], matrix[:, num_qubits + qubit]], axis=1)
            matrix[:, [qubit, num_qubits + qubit]] = columns @ clifford % 2
        order = rng.permutation(num_qubits)
        matrix = np.hstack([matrix[:, order], matrix[:, num_qubits + order]])
        yield get_qubit_problem(codes.QuditCode(matrix, is_subsystem_code=code.is_subsystem_code))


@pytest.mark.parametrize("restrict_viable_symbols", [True, False])
def test_random_qubit_problems_match_enumeration(
    pytestconfig: pytest.Config, restrict_viable_symbols: bool
) -> None:
    """Qubit problems agree with an enumeration at every weight limit, and from every root."""
    # The five-qubit code in two frames: in the first, a state needs Y on a qubit after an earlier
    # sibling of the state tried X there, and in the second, a state needs the other values of the
    # qubit of a branch once the search has left that branch.
    mixed_five_qubit_codes = [
        [
            [1, 1, 1, 0, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 1, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1, 1],
        ],
        [
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        ],
    ]
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    problems = [
        *QUBIT_PROBLEMS.values(),
        *(get_qubit_problem(codes.QuditCode(matrix)) for matrix in mixed_five_qubit_codes),
        *get_random_qubit_problems(rng, 10),
    ]
    tuning = _tuning.SearchTuning(restrict_viable_symbols=restrict_viable_symbols)
    for problem in problems:
        num_qubits = problem[0].shape[1] // 2
        supports = get_symplectic_witness_supports(problem)
        distance = min(support.bit_count() for support in supports)
        assert distance >= 2
        assert (
            cs.get_distance_connected_cluster([problem], cutoff=0, symplectic=True, tuning=tuning)
            == distance
        )

        # Searching at every limit catches a witness with two values on a qubit, and searching from
        # every root catches a value that the search misses.
        for max_weight in range(1, num_qubits + 1):
            search = get_search(
                *problem, symplectic=True, restrict_viable_symbols=restrict_viable_symbols
            )
            witness = search.find_witness(max_weight)
            assert (witness is None) == (max_weight < distance)
            assert witness is None or (
                len(witness) <= max_weight and is_symplectic_witness(problem, witness)
            )

        least_qubits = {
            (support & -support).bit_length() - 1
            for support in supports
            if support.bit_count() == distance
        }
        for root in range(num_qubits):
            search = get_search(
                *problem, symplectic=True, restrict_viable_symbols=restrict_viable_symbols
            )
            search._roots = np.array([root])
            witness = search.find_witness(distance)
            assert (witness is not None) == (root in least_qubits)
            assert witness is None or is_symplectic_witness(problem, witness)


def test_qubit_automorphisms_reduce_the_roots_and_keep_the_distance() -> None:
    """The cyclic shift of the five-qubit code leaves one root, and so fewer states."""
    problem = QUBIT_PROBLEMS["five-qubit"]
    shift = (np.arange(5) + 1) % 5
    plain = get_search(*problem, symplectic=True)
    symmetric = get_search(*problem, symplectic=True, get_automorphisms=lambda: shift[None])
    for search in (plain, symmetric):
        assert search.find_witness(2) is None
        witness = search.find_witness(3)
        assert witness is not None and is_symplectic_witness(problem, witness)
    assert symmetric.num_states < plain.num_states


def test_the_scale_of_weights_leaves_room_for_every_weight_limit() -> None:
    """The capped scale, times a weight limit as large as the width, fits in 64 bits."""
    problem = PROBLEMS["capped-scale"]
    witness = get_search(*problem).find_witness(45)
    assert witness is not None and is_witness(problem, witness)


@pytest.mark.parametrize("returns_heavy_witnesses", [False, True])
def test_a_broken_search_fails_instead_of_looping(
    monkeypatch: pytest.MonkeyPatch, returns_heavy_witnesses: bool
) -> None:
    """A search that finds no witness at any limit, or one above the limit, raises an error."""
    limits = []

    def find_witness(search: cs.SupportSearch, max_weight: int) -> npt.NDArray[np.int64] | None:
        limits.append(max_weight)
        assert len(limits) < 1000, "the search loops"
        return np.arange(max_weight + 1) if returns_heavy_witnesses else None

    monkeypatch.setattr(cs.SupportSearch, "find_witness", find_witness)
    with pytest.raises(RuntimeError, match="only a bug"):
        cs.get_distance_connected_cluster([PROBLEMS["toric-X"], PROBLEMS["toric-X"]], cutoff=0)

    # Each of the two problems has 16 variables and weight divisor 2, so a correct search needs at
    # most 8 limits for each.
    assert len(limits) == 2 * 16 // 2


def test_cutoffs_and_upper_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search can stop early at a witness within its cutoff, or at a known upper bound."""
    problem = PROBLEMS["toric-X"]
    limits = []
    find_witness = cs.SupportSearch.find_witness

    def record_limit(search: cs.SupportSearch, max_weight: int) -> npt.NDArray[np.int64] | None:
        limits.append(max_weight)
        return find_witness(search, max_weight)

    monkeypatch.setattr(cs.SupportSearch, "find_witness", record_limit)
    assert get_brute_force_distance(problem) == 4
    assert 4 <= cs.get_distance_connected_cluster([problem], cutoff=16) <= 16

    # A known upper bound caps the result.
    assert cs.get_distance_connected_cluster([problem], cutoff=0, upper_bound=3) == 3
    assert cs.get_distance_connected_cluster([], cutoff=0, upper_bound=5) == 5

    # A known upper bound within the cutoff ends the search before it starts.
    limits.clear()
    assert cs.get_distance_connected_cluster([problem], cutoff=3, upper_bound=3) == 3
    assert not limits

    # The first weight limit is the cutoff, so a search can exit at a word that is not the lightest.
    components = PROBLEMS["components"]
    assert cs.get_distance_connected_cluster([components], cutoff=0) == 2
    assert cs.get_distance_connected_cluster([components], cutoff=3) == 3


def test_a_problem_without_logical_operators_has_no_distance() -> None:
    """Without witnesses, a problem counts as one more than its number of variables."""
    trivial = (np.array([[1, 1]], dtype=np.uint8), np.array([[1, 1]], dtype=np.uint8))
    assert not get_search(*trivial).has_witnesses
    assert cs.get_distance_connected_cluster([trivial], cutoff=0) == 3
    assert cs.get_distance_connected_cluster([PROBLEMS["toric-X"], trivial], cutoff=0) == 3

    # On one qubit, X is excluded, and only the identity and X commute with it.
    one_qubit = (np.array([[1, 0]], dtype=np.uint8), np.array([[1, 0]], dtype=np.uint8))
    assert not get_search(*one_qubit, symplectic=True).has_witnesses
    assert cs.get_distance_connected_cluster([one_qubit], cutoff=0, symplectic=True) == 2


def test_degenerate_checks() -> None:
    """Checks may be missing, zero, or repeated, and may be arrays over GF(2)."""
    no_checks = np.zeros((0, 3), dtype=np.uint8)
    witness = get_search(no_checks, no_checks).find_witness(1)
    assert witness is not None and list(witness) == [0]
    checks, excluded = PROBLEMS["steane-X"]
    padded = np.vstack([checks, checks[0], np.zeros_like(checks[0])])
    assert cs.get_distance_connected_cluster([(padded, excluded)], cutoff=0) == 3

    # The matrices of codes are arrays over GF(2).
    steane_code = codes.SteaneCode()
    witness = get_search(steane_code.matrix_z, steane_code.matrix_x).find_witness(3)
    assert witness is not None and is_witness((checks, excluded), witness)


def get_vector(
    symbols: npt.NDArray[np.int64], width: int, symplectic: bool
) -> npt.NDArray[np.uint8]:
    """The binary or symplectic vector of the symbols of a search."""
    if symplectic:
        return get_symplectic_vector(symbols, width // 2)
    vector = np.zeros(width, dtype=np.uint8)
    vector[symbols] = 1
    return vector


# Problems for syndrome queries.
SYNDROME_PROBLEMS = {
    "nested-restrictions": (PROBLEMS["nested-restrictions"][0], False),
    # A zero check, and a check before its copy, so some syndromes have no vector.
    "copied-and-zero-checks": (
        np.vstack([np.zeros((1, 7)), PROBLEMS["steane-X"][0][2:], PROBLEMS["steane-X"][0]]),
        False,
    ),
    # The stabilizers of a pure state: those of the five-qubit code, and its logical X.
    "five-qubit-state": (
        np.vstack([QUBIT_PROBLEMS["five-qubit"][0], [[1] * 5 + [0] * 5]]),
        True,
    ),
    "zero-checks": (np.zeros((2, 3)), False),
    # The syndrome 11 needs every variable, so a query reaches the limit of their number.
    "identity": (np.eye(2), False),
}


@pytest.mark.parametrize("name", SYNDROME_PROBLEMS)
def test_syndrome_queries_match_enumeration(name: str) -> None:
    """A query finds a vector with a syndrome exactly if the limit reaches the least weight."""
    checks, symplectic = SYNDROME_PROBLEMS[name]
    checks = np.asarray(checks, dtype=np.uint8)
    width = checks.shape[1]
    num_variables = width // 2 if symplectic else width
    products = np.roll(checks, num_variables, axis=1) if symplectic else checks
    least_weights: dict[tuple[int, ...], int] = {}
    for entries in itertools.product([0, 1], repeat=width):
        vector = np.array(entries, dtype=np.uint8)
        support = (vector[:num_variables] | vector[num_variables:]) if symplectic else vector
        syndrome = tuple(products @ vector % 2)
        least_weights[syndrome] = min(least_weights.get(syndrome, width), int(np.sum(support)))

    search = get_search(checks, np.zeros((0, width)), symplectic=symplectic)
    for syndrome in itertools.product([0, 1], repeat=len(checks)):
        least_weight = least_weights.get(syndrome, num_variables + 1)
        for max_weight in range(num_variables + 1):
            symbols = search.find_vector_with_syndrome(syndrome, max_weight)
            assert (symbols is None) == (max_weight < least_weight)
            if symbols is not None:
                assert len(symbols) <= max_weight and list(symbols) == sorted(symbols)
                assert len(set(symbols % num_variables)) == len(symbols)
                vector = get_vector(symbols, width, symplectic)
                assert tuple(products @ vector % 2) == syndrome

    # Syndrome queries count no states.
    assert search.num_states == 0


def test_a_syndrome_query_leaves_no_state_for_later_roots() -> None:
    """A failed syndrome query leaves no syndrome, restriction, or block to later roots."""
    checks = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0, 0],
            [1, 1, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    search = get_search(checks, np.zeros((0, checks.shape[1])))

    def search_from(roots: list[int], syndrome: tuple[int, ...]) -> list[int]:
        support, _ = cs._search_supports(
            search._check_starts,
            search._check_symbols,
            search._symbol_masks,
            search._test_columns,
            np.array(roots, dtype=np.int64),
            search._ranks,
            search._inverse_overlaps,
            2,
            cs.pack_bits(np.array([syndrome], dtype=np.uint8))[0],
            True,
            True,
        )
        return sorted(support.tolist())

    # The syndrome query of 01001 restricts its empty state, and finds nothing within the limit.
    syndrome = (0, 1, 0, 0, 1)
    assert search.find_vector_with_syndrome(syndrome, 2) is None
    no_syndrome = (0,) * len(checks)
    roots = list(range(checks.shape[1]))
    expected = search_from(roots, no_syndrome)
    assert expected
    assert search_from([-1, *roots], syndrome) == expected
    for root in roots:
        assert search_from([-1, root], syndrome) == search_from([root], no_syndrome)


@pytest.mark.parametrize("name", ["nested-restrictions", "five-qubit-state"])
def test_memoizing_changes_no_result_or_count_of_states(name: str) -> None:
    """Memoizing changes neither the support that a query returns nor its number of states."""
    checks, symplectic = SYNDROME_PROBLEMS[name]
    search = get_search(checks, np.zeros((0, np.shape(checks)[1])), symplectic=symplectic)
    for bits in itertools.product([0, 1], repeat=len(search._checks)):
        if not any(bits):
            continue
        packed = cs.pack_bits(np.array([bits], dtype=np.uint8))[0]
        for roots, max_weight in itertools.product([[-1], [0, -1]], range(1, 4)):
            results = [
                cs._search_supports(
                    search._check_starts,
                    search._check_symbols,
                    search._symbol_masks,
                    search._test_columns,
                    np.array(roots, dtype=np.int64),
                    search._ranks,
                    search._inverse_overlaps,
                    max_weight,
                    packed,
                    memoize,
                    True,
                )
                for memoize in (False, True)
            ]
            # Stale memos, such as those of a search from root 0, would likely change the count.
            assert results[0][0].tolist() == results[1][0].tolist()
            assert results[0][1] == results[1][1]


def test_invalid_syndrome_queries_are_rejected() -> None:
    """A syndrome needs a binary entry for each check, and a weight limit is at least zero."""
    search = get_search(*PROBLEMS["steane-X"])
    with pytest.raises(ValueError, match="one entry for each of 3 checks"):
        search.find_vector_with_syndrome([1, 0], 1)
    with pytest.raises(ValueError, match="binary"):
        search.find_vector_with_syndrome([2, 0, 0], 1)
    with pytest.raises(ValueError, match="at least zero"):
        search.find_vector_with_syndrome([1, 0, 0], -1)


def test_invalid_problems_are_rejected() -> None:
    """Invalid problems and weight limits, and distance searches of nothing, are rejected."""
    with pytest.raises(ValueError, match="equal widths"):
        get_search(np.zeros((1, 3), dtype=np.uint8), np.zeros((1, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="satisfy the checks"):
        get_search(np.ones((1, 3), dtype=np.uint8), np.array([[1, 0, 0]], dtype=np.uint8))
    with pytest.raises(ValueError, match="binary"):
        get_search(np.array([[1, 2, 0]]), np.zeros((0, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="binary"):
        get_search(np.zeros((0, 3), dtype=np.uint8), np.array([[1, 0.5, 0]]))
    with pytest.raises(ValueError, match="at least one"):
        get_search(*PROBLEMS["steane-X"]).find_witness(0)
    with pytest.raises(ValueError, match="at least one problem or an upper bound"):
        cs.get_distance_connected_cluster([], cutoff=0)
    with pytest.raises(ValueError, match="even number of columns"):
        get_search(
            np.ones((1, 3), dtype=np.uint8), np.ones((0, 3), dtype=np.uint8), symplectic=True
        )
