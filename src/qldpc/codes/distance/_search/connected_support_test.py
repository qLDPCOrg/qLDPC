# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact binary distance search over supports that are connected in a Tanner graph."""

from __future__ import annotations

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
from . import connected_support as cs

Problem = tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]

# The [[276, 98, 14]] pair-partition code of arXiv:2607.14091: normalized exponents, whose first
# row and column are zero, of the instance qc_276_98_14 of the data of its authors.
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
    """The X and Z sectors of a code: its stabilizers of the opposite type, and its gauge matrix.

    This builds the sectors independently of ``CSSCode._get_parity_check_sector``, as an oracle.
    """
    return [
        (
            np.asarray(code.get_stabilizer_ops(Pauli.Z if pauli is Pauli.X else Pauli.X), np.uint8),
            np.asarray(code.get_matrix(pauli), dtype=np.uint8),
        )
        for pauli in PAULIS_XZ
    ]


def get_golay_problem() -> Problem:
    """The CSS sector of the [23, 12, 7] Golay code, with its dual as the excluded rows.

    The Golay code is a cyclic code that contains its dual, and its checks are a basis of its dual,
    so cyclic shifts map its kernel and its excluded rows onto themselves.
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
    """A search of a problem, which asks for any automorphisms before its first weight limit.

    A test can pass another ``automorphism_minimum_states``, and the other options of the search.
    """
    options.setdefault("automorphism_minimum_states", 0)
    return cs.SupportSearch(*problem, **options)


def get_symplectic_witness_supports(problem: Problem) -> list[int]:
    """The supports of the Pauli operators that commute with the checks, outside the excluded.

    A word is an integer whose bits ``j`` and ``n + j`` are its X and Z parts on qubit ``j``, and a
    support is an integer whose bit ``j`` is set if the word acts on qubit ``j``.
    """
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
    """Whether sorted symbols on distinct qubits give a Pauli operator that is a witness.

    A witness commutes with every check, and is outside the row space of the excluded rows.
    """
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


# Sectors of small codes.  The Steane code has words of odd weight, and the toric code several
# lightest logical operators.  The Z sector of the rotated surface code has stabilizers lighter than
# its distance, and the Reed-Muller code has only words of even weight.
PROBLEMS: dict[str, Problem] = {
    "steane-X": get_sectors(codes.SteaneCode())[0],
    "toric-X": get_sectors(codes.ToricCode(4, rotated=True))[0],
    "surface-Z": get_sectors(codes.SurfaceCode(3))[1],
    "reed-muller-X": get_sectors(codes.QuantumReedMullerCode(1, 4))[0],
}

# Words of weight 3 on the first three variables, and of weight 2 on the last two.  The distance is
# the weight of the second, whose least variable is greater than that of the first.
PROBLEMS["components"] = (
    np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=np.uint8),
    np.zeros((0, 5), dtype=np.uint8),
)

# Variables 0 and 1 are in all 43 checks, and each other variable in one, so the distance is 2.  The
# scale lcm(1, ..., 43) of the weights of checks exceeds 2**63, so the search caps it at 2**40 and
# rounds the weights down.
PROBLEMS["capped-scale"] = (
    np.hstack([np.ones((43, 2), dtype=np.uint8), np.eye(43, dtype=np.uint8)]),
    np.zeros((0, 45), dtype=np.uint8),
)

# Nested states of a search of these checks restrict their subtrees, and could allow more variables
# in total than there are variables, so the search makes more room for them than it starts with.
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


# Searches of these checks need the variables that a state allows to be recorded, and released, with
# the state: the first has states that restrict their subtrees to no variable, and the second has
# nested restrictions.
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

# The problems that the tests of the distance, and of the searches from each root, compare with an
# enumeration.  The others serve the tests that use them.
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


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("name", SEARCHED_PROBLEMS)
def test_the_distance_matches_enumeration(name: str) -> None:
    """The distance search, and a search through each weight limit, agree with an enumeration."""
    problem = PROBLEMS[name]
    distance = get_brute_force_distance(problem)
    assert cs.get_distance_connected_support([problem], cutoff=0) == distance

    # A search finds a witness exactly if its weight limit reaches the distance.
    for max_weight in range(1, distance + 1):
        witness = get_search(*problem).find_witness(max_weight)
        if max_weight < distance:
            assert witness is None
        else:
            assert (
                witness is not None and len(witness) <= max_weight and is_witness(problem, witness)
            )


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("name", SEARCHED_PROBLEMS)
def test_each_root_finds_the_lightest_witnesses_that_it_roots(name: str) -> None:
    """The search from a variable alone, through the distance, finds the witnesses that it roots.

    Every lightest witness has a support that is connected, and the search from its least variable
    finds it (Section 3.3 of ``connected_support.md``).  So the search from a variable finds a
    witness exactly if some lightest witness has that least variable.
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


@pytest.mark.usefixtures("compilation_mode")
def test_each_support_is_visited_once_above_its_least_variable() -> None:
    """With every word excluded, a search visits the supports that grow from each root once.

    Take the checks {0, 1, 2} and {1, 2, 3}, whose words are {0, 1, 3}, {0, 2, 3}, and {1, 2}.  With
    a weight limit of 4, a search from variable 0 branches on variable 1 or 2.  After {0, 1} it
    branches on variable 2, which leaves a check without available variables, or on variable 3,
    which reaches a word.  After {0, 2} it can only branch on variable 3, since the branch on 1
    blocks variable 1 for its sibling.  That gives 6 states.  From variable 1, the checks force
    variable 2, which reaches a word, so there are 2 states.  Variables 2 and 3 are each left with a
    check that only variables below them can satisfy, which gives 1 state each.

    With a weight limit of 2, a support with a check left unsatisfied needs one more variable, as
    every variable is in at most two checks, so the supports {0, 1} and {0, 2} are not visited.
    That leaves the states {0}, {1}, {1, 2}, {2}, and {3}.
    """
    checks = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=np.uint8)
    num_states = {}
    for max_weight in (2, 4):
        search = get_search(checks, get_kernel(checks))
        assert search.find_witness(max_weight) is None
        num_states[max_weight] = search.num_states
    assert num_states == {2: 5, 4: 6 + 2 + 1 + 1}


@pytest.mark.usefixtures("compilation_mode")
def test_automorphisms_reduce_the_roots_and_keep_the_distance() -> None:
    """A cyclic code has a symmetry that leaves one root, and so fewer states.

    The Golay code is a cyclic code that contains its dual, so cyclic shifts of its qubits map its
    kernel and its excluded rows onto themselves.  The quantum Golay code has distance 7 (see
    https://errorcorrectionzoo.org/c/qubit_golay).
    """
    problem = get_golay_problem()
    distance = get_brute_force_distance(problem)
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


@pytest.mark.usefixtures("compilation_mode")
def test_each_root_excludes_the_orbits_before_its_own() -> None:
    """The search from the root of an orbit uses no variable of an earlier orbit.

    The [[72, 12, 6]] bivariate bicycle code has two orbits of qubits under the translations of its
    group, one for each block of 36 qubits.  Its qubits are shuffled here, so that the orbits
    interleave.  Every lightest witness is the image of one whose least orbit is that of its root,
    so the search from each root can exclude the earlier orbits entirely.
    """
    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    order = np.random.default_rng(0).permutation(len(code))
    automorphisms = np.argsort(order)[code._get_qudit_automorphisms()[:, order]]
    problem = (get_sectors(code)[0][0][:, order], get_sectors(code)[0][1][:, order])

    # Orbit 0 is the block of the first shuffled qubit, and orbit 1 the other block.
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


@pytest.mark.usefixtures("compilation_mode")
def test_each_root_ranks_lowest_in_its_orbit() -> None:
    """The ranks keep the order of the variables within each orbit, so each root ranks lowest.

    The checks join the variables of two cycles, of 60 and 30 variables, which interleave.  Each
    cycle is a word, so the distance is 30, and the shift along both cycles is an automorphism with
    the cycles as its orbits.  The word of the second cycle needs every variable of its orbit, so
    the search from its root finds it only if no variable of that orbit ranks below the root.
    """
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


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("minimum_states", [0, 30, _tuning.MAX_VALUE])
def test_automorphisms_are_found_once_by_the_threshold(minimum_states: int) -> None:
    """A search asks for automorphisms at most once, and by the time it reaches the threshold.

    A search may ask sooner, if it expects to reach the threshold, which is a heuristic that this
    test leaves open.  The checks of ``test_each_support_is_visited_once_above_its_least_variable``
    have words of weights 2 and 3, so their weight divisor is 1, and no limit here is raised.  The
    searches of weight limits 1, 2, 3, and 4 visit 0, 5, 9, and 10 states.
    """
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


@pytest.mark.usefixtures("compilation_mode")
def test_a_limit_without_states_does_not_ask_for_automorphisms() -> None:
    """A search without an estimate of its states asks for automorphisms only before a raised limit.

    The first limit of a symplectic search of the five-qubit code, a weight of one, visits no
    states, which makes a growth of zero the estimate for the next limit, and not a missing one.
    """
    found = []
    search = get_search(
        *QUBIT_PROBLEMS["five-qubit"],
        homogeneous=False,
        get_automorphisms=lambda: found.append(True),
        automorphism_minimum_states=10**6,
    )
    search.find_witness(1)
    assert search.num_states == 0
    search.find_witness(2)
    assert not found


@pytest.mark.usefixtures("compilation_mode")
def test_a_cutoff_raises_the_first_limit_to_a_multiple_of_the_divisor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cutoff above the weight divisor sets the first limit, rounded down to a multiple of it.

    Such a first limit skips the cheap limits below it, so it has no estimate of its states, and
    the search asks for automorphisms before it.  Every word of the Reed-Muller code has even
    weight, and its distance is 4.
    """
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
            distance = cs.get_distance_connected_support(
                [problem], cutoff=cutoff, get_automorphisms=get_automorphisms
            )
        assert 4 <= distance <= max(4, cutoff)
        assert limits[0] == first_limit
        assert get_automorphisms.call_count == num_calls


def test_pruning_keeps_the_search_of_a_high_rate_code_small() -> None:
    """The pair-partition code [[276, 98, 14]] has no logical operator of weight 8 or less.

    The distance of this code is 14 (https://arxiv.org/abs/2607.14091).  There are more than 10**14
    supports of 8 qubits, and the bounds on the qubits that a support needs rule out all but a few
    thousand of them.  The code translates its blocks of qubits cyclically, which leaves 12 roots.

    The limit of 5000 states guards the variables that a state forbids for its subtree: the search
    visits about 4300 states with them, and about 9300 without them.  A change of heuristic that
    moves the count across the limit needs a new limit, with its reasons.
    """
    code = codes.PairPartitionCode(EXPONENTS_X_276, EXPONENTS_Z_276, lift_size=23)
    search = get_search(*get_sectors(code)[0], get_automorphisms=code._get_qudit_automorphisms)
    assert search.find_witness(8) is None
    assert search.num_states < 5000


def test_wrong_automorphisms_are_rejected() -> None:
    """A permutation that does not map the kernel of the checks to itself raises an error."""
    swap = np.arange(7)[None]
    swap[0, :2] = [1, 0]
    search = get_search(*PROBLEMS["steane-X"], get_automorphisms=lambda: swap)
    with pytest.raises(ValueError, match="automorphism"):
        search.find_witness(3)


@pytest.mark.usefixtures("compilation_mode")
def test_sectors_are_searched_together(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tetrahedral code has X-distance 7 and Z-distance 3: the Z sector bounds the X search.

    The search of the X sector stops once the Z sector has a witness of weight 3, rather than going
    on to weight 7.
    """
    sectors = get_sectors(codes.TetrahedralCode())
    assert [get_brute_force_distance(sector) for sector in sectors] == [7, 3]

    limits = []
    find_witness = cs.SupportSearch.find_witness

    def record_limit(search: cs.SupportSearch, max_weight: int) -> npt.NDArray[np.int64] | None:
        limits.append(max_weight)
        return find_witness(search, max_weight)

    monkeypatch.setattr(cs.SupportSearch, "find_witness", record_limit)
    assert cs.get_distance_connected_support(sectors, cutoff=0) == 3
    assert max(limits) == 3


@pytest.mark.usefixtures("compilation_mode")
def test_the_logical_test_reads_every_block() -> None:
    """A witness may fail only tests past the first 64, in a later block of each packed column.

    Prepending a block of zeros to the tests of the Steane code moves every test it has there.
    """
    problem = PROBLEMS["steane-X"]
    search = get_search(*problem)
    zeros = np.zeros((search.num_variables, 1), dtype=np.uint64)
    search._test_columns = np.hstack([zeros, search._test_columns])
    witness = search.find_witness(3)
    assert witness is not None and is_witness(problem, witness)


def get_random_qubit_problems(rng: np.random.Generator, num_problems: int) -> Iterator[Problem]:
    """Small codes with a random single-qubit Clifford on each qubit, and their qubits permuted.

    A Clifford permutes X, Z, and Y on its qubit, so the checks have all three.
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


@pytest.mark.usefixtures("compilation_mode")
def test_random_qubit_problems_match_enumeration(pytestconfig: pytest.Config) -> None:
    """Qubit problems agree with an enumeration at every weight limit, and from every root.

    A witness gives each qubit of its support one of X, Z, and Y, so its weight is its number of
    symbols.  The search of every weight limit, up to the number of qubits, returns such witnesses,
    which fails if a qubit can take two values.

    Every lightest witness has a support that is connected, and the search from its least qubit
    finds it (Section 3.3 of ``connected_support.md``).  So the search from a qubit alone, through
    the distance, finds a witness exactly if some lightest witness has that least qubit, which
    fails if the search misses a value of a qubit.
    """
    # The five-qubit code with single-qubit Cliffords, for which a state needs the value Y of a
    # qubit after an earlier sibling tried X there, and in another frame, for which a state needs
    # the other values of the qubit of its last branch after the search leaves it.
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
        *get_random_qubit_problems(rng, 30),
    ]
    for problem in problems:
        num_qubits = problem[0].shape[1] // 2
        supports = get_symplectic_witness_supports(problem)
        distance = min(support.bit_count() for support in supports)
        assert distance >= 2
        assert cs.get_distance_connected_support([problem], cutoff=0, homogeneous=False) == distance
        for max_weight in range(1, num_qubits + 1):
            witness = get_search(*problem, homogeneous=False).find_witness(max_weight)
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
            search = get_search(*problem, homogeneous=False)
            search._roots = np.array([root])
            witness = search.find_witness(distance)
            assert (witness is not None) == (root in least_qubits)
            assert witness is None or is_symplectic_witness(problem, witness)


@pytest.mark.usefixtures("compilation_mode")
def test_qubit_automorphisms_reduce_the_roots_and_keep_the_distance() -> None:
    """The cyclic shift of the five-qubit code leaves one root, and so fewer states."""
    problem = QUBIT_PROBLEMS["five-qubit"]
    shift = (np.arange(5) + 1) % 5
    plain = get_search(*problem, homogeneous=False)
    symmetric = get_search(*problem, homogeneous=False, get_automorphisms=lambda: shift[None])
    for search in (plain, symmetric):
        assert search.find_witness(2) is None
        witness = search.find_witness(3)
        assert witness is not None and is_symplectic_witness(problem, witness)
    assert symmetric.num_states < plain.num_states


@pytest.mark.usefixtures("compilation_mode")
def test_the_scale_of_weights_leaves_room_for_every_weight_limit() -> None:
    """Weights in units of the capped scale, times a weight limit as large as the width, fit.

    A cap near ``2**63`` would overflow the products of a weight limit of 45 with the scale.
    """
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
        cs.get_distance_connected_support([PROBLEMS["toric-X"], PROBLEMS["toric-X"]], cutoff=0)

    # Each of the two problems has 16 variables and weight divisor 2, so a correct search needs at
    # most 8 limits for each.
    assert len(limits) == 2 * 16 // 2


@pytest.mark.usefixtures("compilation_mode")
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
    assert 4 <= cs.get_distance_connected_support([problem], cutoff=16) <= 16

    # A known upper bound is a candidate for the distance, and the search can only lower it.
    assert cs.get_distance_connected_support([problem], cutoff=0, upper_bound=3) == 3
    assert cs.get_distance_connected_support([], cutoff=0, upper_bound=5) == 5

    # A known upper bound within the cutoff ends the search before it starts.
    limits.clear()
    assert cs.get_distance_connected_support([problem], cutoff=3, upper_bound=3) == 3
    assert not limits

    # The first weight limit is the cutoff, a multiple of the weight divisor of 1, so a search can
    # exit at a word that is not the lightest.
    components = PROBLEMS["components"]
    assert cs.get_distance_connected_support([components], cutoff=0) == 2
    assert cs.get_distance_connected_support([components], cutoff=3) == 3


@pytest.mark.usefixtures("compilation_mode")
def test_a_problem_without_logical_operators_has_no_distance() -> None:
    """Without a vector outside the excluded rows, the distance is one more than the width."""
    trivial = (np.array([[1, 1]], dtype=np.uint8), np.array([[1, 1]], dtype=np.uint8))
    assert not get_search(*trivial).has_witnesses
    assert cs.get_distance_connected_support([trivial], cutoff=0) == 3
    assert cs.get_distance_connected_support([PROBLEMS["toric-X"], trivial], cutoff=0) == 3

    # The Pauli X on one qubit is excluded, and the identity and X are the operators that commute
    # with it, so a symplectic problem on one qubit has no distance either.
    one_qubit = (np.array([[1, 0]], dtype=np.uint8), np.array([[1, 0]], dtype=np.uint8))
    assert not get_search(*one_qubit, homogeneous=False).has_witnesses
    assert cs.get_distance_connected_support([one_qubit], cutoff=0, homogeneous=False) == 2


@pytest.mark.usefixtures("compilation_mode")
def test_degenerate_checks() -> None:
    """Checks may be missing, zero, or repeated, and may be arrays over GF(2)."""
    no_checks = np.zeros((0, 3), dtype=np.uint8)
    witness = get_search(no_checks, no_checks).find_witness(1)
    assert witness is not None and list(witness) == [0]
    checks, excluded = PROBLEMS["steane-X"]
    padded = np.vstack([checks, checks[0], np.zeros_like(checks[0])])
    assert cs.get_distance_connected_support([(padded, excluded)], cutoff=0) == 3

    # The matrices of codes are arrays over GF(2).
    steane_code = codes.SteaneCode()
    witness = get_search(steane_code.matrix_z, steane_code.matrix_x).find_witness(3)
    assert witness is not None and is_witness((checks, excluded), witness)


def get_vector(
    symbols: npt.NDArray[np.int64], width: int, homogeneous: bool
) -> npt.NDArray[np.uint8]:
    """The binary or symplectic vector of the symbols of a search."""
    if not homogeneous:
        return get_symplectic_vector(symbols, width // 2)
    vector = np.zeros(width, dtype=np.uint8)
    vector[symbols] = 1
    return vector


# Problems for syndrome queries.  The states of nested-restrictions restrict their subtrees.
# copied-and-zero-checks has a zero check and a check before its copy, so a query maps its syndrome
# onto fewer checks, and some syndromes have no vector.  five-qubit-state is the stabilizers of a
# pure state: the five-qubit code, with its logical X.  zero-checks has only zero checks, and the
# syndrome 11 of identity needs every variable, so a query reaches the limit of their number.
SYNDROME_PROBLEMS = {
    "nested-restrictions": (PROBLEMS["nested-restrictions"][0], True),
    "copied-and-zero-checks": (
        np.vstack([np.zeros((1, 7)), PROBLEMS["steane-X"][0][2:], PROBLEMS["steane-X"][0]]),
        True,
    ),
    "five-qubit-state": (
        np.vstack([QUBIT_PROBLEMS["five-qubit"][0], [[1] * 5 + [0] * 5]]),
        False,
    ),
    "zero-checks": (np.zeros((2, 3)), True),
    "identity": (np.eye(2), True),
}


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("name", SYNDROME_PROBLEMS)
def test_syndrome_queries_match_enumeration(name: str) -> None:
    """A query finds a vector with a syndrome exactly if the limit reaches the least weight.

    An enumeration of every vector gives the least weight of each syndrome, and a syndrome that no
    vector has, of any weight, has no vector within any limit.
    """
    checks, homogeneous = SYNDROME_PROBLEMS[name]
    checks = np.asarray(checks, dtype=np.uint8)
    width = checks.shape[1]
    num_variables = width if homogeneous else width // 2
    products = checks if homogeneous else np.roll(checks, num_variables, axis=1)
    least_weights: dict[tuple[int, ...], int] = {}
    for entries in itertools.product([0, 1], repeat=width):
        vector = np.array(entries, dtype=np.uint8)
        support = vector if homogeneous else vector[:num_variables] | vector[num_variables:]
        syndrome = tuple(products @ vector % 2)
        least_weights[syndrome] = min(least_weights.get(syndrome, width), int(np.sum(support)))

    search = get_search(checks, np.zeros((0, width)), homogeneous=homogeneous)
    for syndrome in itertools.product([0, 1], repeat=len(checks)):
        least_weight = least_weights.get(syndrome, num_variables + 1)
        for max_weight in range(num_variables + 1):
            symbols = search.find_vector_with_syndrome(syndrome, max_weight)
            assert (symbols is None) == (max_weight < least_weight)
            if symbols is not None:
                assert len(symbols) <= max_weight and list(symbols) == sorted(symbols)
                assert len(set(symbols % num_variables)) == len(symbols)
                vector = get_vector(symbols, width, homogeneous)
                assert tuple(products @ vector % 2) == syndrome

    # Syndrome queries are not states of witness searches.
    assert search.num_states == 0


@pytest.mark.usefixtures("compilation_mode")
def test_a_syndrome_query_leaves_no_state_for_later_roots() -> None:
    """A search from a root of -1 that finds nothing leaves the searches from variables unchanged.

    They must start from the empty syndrome, and without the restriction of the empty state of the
    syndrome query, which these checks have for the syndrome 01001.  Only a root that is a variable
    blocks the variables of lower rank than it.
    """
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


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("name", ["nested-restrictions", "five-qubit-state"])
def test_memoizing_changes_no_result_or_count_of_states(name: str) -> None:
    """A syndrome query that memoizes overlaps and repairs finds what one without it finds.

    Each state uses the same overlaps and repairs either way, so the searches return the same
    support and visit as many states.  They do so also after a search from root 0, which leaves its
    own stamps in the memos.  A memo that kept stale values could still find correct vectors, with
    different pruning, but would likely change the number of states.
    """
    checks, homogeneous = SYNDROME_PROBLEMS[name]
    search = get_search(checks, np.zeros((0, np.shape(checks)[1])), homogeneous=homogeneous)
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
                )
                for memoize in (False, True)
            ]
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
    """Problems need binary matrices of equal widths, whose excluded rows are in the kernel.

    A distance search also needs a problem or an upper bound.
    """
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
        cs.get_distance_connected_support([], cutoff=0)
    with pytest.raises(ValueError, match="even number of columns"):
        get_search(
            np.ones((1, 3), dtype=np.uint8), np.ones((0, 3), dtype=np.uint8), homogeneous=False
        )
