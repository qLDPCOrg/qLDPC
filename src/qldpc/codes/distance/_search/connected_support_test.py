# SPDX-License-Identifier: Apache-2.0

"""Tests for the exact binary distance search over supports that are connected in a Tanner graph."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y

from qldpc import codes
from qldpc._binary import get_independent_rows, get_kernel
from qldpc.codes.quantum.pair_partition_test import EXPONENTS_X_276, EXPONENTS_Z_276
from qldpc.objects import PAULIS_XZ, Pauli

from .. import _tuning
from ..conftest import get_random_css_code, get_span
from . import automorphism_families
from . import connected_support as cs

Problem = tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]


@pytest.fixture
def automorphisms_at_once() -> Iterator[None]:
    """Let searches find automorphisms before their first state."""
    with _tuning.use_tuning(automorphism_minimum_states=0):
        yield


def get_sectors(code: codes.CSSCode) -> list[Problem]:
    """The X and Z sectors of a code: its stabilizers of the opposite type, and its gauge matrix."""
    return [
        (
            np.asarray(code.get_stabilizer_ops(Pauli.Z if pauli is Pauli.X else Pauli.X), np.uint8),
            np.asarray(code.get_matrix(pauli), dtype=np.uint8),
        )
        for pauli in PAULIS_XZ
    ]


def get_cyclic_problem(num_variables: int, exponents: list[int], dimension: int) -> Problem:
    """The CSS sector of a cyclic code that contains its dual, with its dual as the excluded rows.

    The code has the generator polynomial with the given exponents, and its checks are a basis of
    its dual, so cyclic shifts map its kernel and its excluded rows onto themselves.
    """
    generator = np.zeros(num_variables, dtype=np.uint8)
    generator[exponents] = 1
    checks = get_kernel(np.array([np.roll(generator, shift) for shift in range(dimension)]))
    return checks, checks


def get_brute_force_distance(problem: Problem) -> int:
    """The least weight of a vector in the kernel of the checks and outside the excluded rows."""
    checks, excluded = problem
    return min(word.bit_count() for word in get_span(get_kernel(checks)) - get_span(excluded))


def is_witness(problem: Problem, support: npt.NDArray[np.int64]) -> bool:
    """Whether a support is that of a vector in the kernel of the checks, outside the excluded."""
    checks, excluded = problem
    vector = np.zeros(checks.shape[1], dtype=np.uint8)
    vector[support] = 1
    return (
        len(set(support)) == len(support)
        and not np.any(checks @ vector % 2)
        and len(get_independent_rows(np.vstack([excluded, vector])))
        > len(get_independent_rows(excluded))
    )


# The odd-weight Steane code, and the toric code, whose lightest logical operators are not unique.
# The rotated surface code and the Bacon-Shor code, a subsystem code, have stabilizers or gauge
# operators that are lighter than their distances.  The Reed-Muller code has only even weights.
CODES = {
    "steane": codes.SteaneCode(),
    "toric": codes.ToricCode(4, rotated=True),
    "surface": codes.SurfaceCode(3),
    "bacon-shor": codes.CSSCode(
        codes.BaconShorCode(3).matrix_x, codes.BaconShorCode(3).matrix_z, is_subsystem_code=True
    ),
    "reed-muller": codes.QuantumReedMullerCode(1, 4),
}
PROBLEMS = {
    f"{name}-{pauli.name}": problem
    for name, code in CODES.items()
    for pauli, problem in zip(PAULIS_XZ, get_sectors(code))
}

# Words of weight 3 on the first three variables, and of weight 2 on the last two.  The distance is
# the weight of the second, whose least variable is greater than that of the first.
PROBLEMS["components"] = (
    np.array([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]], dtype=np.uint8),
    np.zeros((0, 5), dtype=np.uint8),
)


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("problem", PROBLEMS.values(), ids=PROBLEMS.keys())
def test_the_distance_matches_enumeration(problem: Problem) -> None:
    """Both searches of a sector agree with an enumeration of its vectors."""
    distance = get_brute_force_distance(problem)
    assert cs.get_distance_connected_support([problem], cutoff=0) == distance

    # A search finds a witness exactly if its weight limit reaches the distance.
    for max_weight in range(1, distance + 1):
        witness = cs.SupportSearch(*problem).find_witness(max_weight)
        if max_weight < distance:
            assert witness is None
        else:
            assert (
                witness is not None and len(witness) <= max_weight and is_witness(problem, witness)
            )


@pytest.mark.usefixtures("compilation_mode")
def test_random_codes_match_enumeration(pytestconfig: pytest.Config) -> None:
    """Sectors of random CSS codes, which have checks with and without overlaps, agree."""
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(10):
        code = get_random_css_code(rng, int(rng.integers(6, 11)), 2, 2)
        for problem in get_sectors(code):
            if code.dimension:
                assert cs.get_distance_connected_support([problem], cutoff=0) == (
                    get_brute_force_distance(problem)
                )


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
        search = cs.SupportSearch(checks, get_kernel(checks))
        assert search.find_witness(max_weight) is None
        num_states[max_weight] = search.num_states
    assert num_states == {2: 5, 4: 6 + 2 + 1 + 1}


@pytest.mark.usefixtures("compilation_mode")
def test_roots_can_come_in_any_order() -> None:
    """The search from the last variable leaves every variable available for the search from 0."""
    problem = PROBLEMS["toric-X"]
    search = cs.SupportSearch(*problem)
    search._roots = np.array([problem[0].shape[1] - 1, 0])
    witness = search.find_witness(get_brute_force_distance(problem))
    assert witness is not None and is_witness(problem, witness)


@pytest.mark.usefixtures("compilation_mode")
def test_a_search_from_one_root_finds_words_with_that_least_variable() -> None:
    """A search from a root only uses the variables above it.

    A word of weight w with least variable r has w - 1 variables above r, so no word of the distance
    has a root among the last variables.
    """
    problem = PROBLEMS["toric-X"]
    num_variables = problem[0].shape[1]
    distance = get_brute_force_distance(problem)
    roots_with_witnesses = []
    for root in range(num_variables):
        search = cs.SupportSearch(*problem)
        search._roots = np.array([root])
        witness = search.find_witness(distance)
        if root > num_variables - distance:
            assert witness is None
        elif witness is not None:
            assert min(witness) == root and is_witness(problem, witness)
            roots_with_witnesses.append(root)
    assert roots_with_witnesses


@pytest.mark.usefixtures("compilation_mode", "automorphisms_at_once")
def test_automorphisms_reduce_the_roots_and_keep_the_distance() -> None:
    """A cyclic code has a symmetry that leaves one root, and so fewer states.

    The Golay code is a cyclic code that contains its dual, so cyclic shifts of its qubits map its
    kernel and its excluded rows onto themselves.
    """
    problem = get_cyclic_problem(23, [0, 1, 5, 6, 7, 9, 11], 12)
    distance = get_brute_force_distance(problem)
    shift = (np.arange(23) + 1) % 23
    plain = cs.SupportSearch(*problem)
    symmetric = cs.SupportSearch(*problem, get_automorphisms=lambda: shift[None])
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
    group, one for each block.  Its qubits are shuffled here, so that the orbits interleave.  Every
    lightest witness is the image of one whose least orbit is that of its root, so the search from
    each root can exclude the earlier orbits entirely.
    """
    code = codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    order = np.random.default_rng(0).permutation(len(code))
    automorphisms = np.argsort(order)[code._get_qudit_automorphisms()[:, order]]
    problem = (get_sectors(code)[0][0][:, order], get_sectors(code)[0][1][:, order])
    orbits = automorphism_families._get_orbits(automorphisms, len(code))
    roots = np.unique(orbits, return_index=True)[1]
    assert roots[1] < len(code) // 2

    for max_weight in (6, 8):
        for orbit, root in enumerate(roots):
            search = cs.SupportSearch(*problem)
            search._use_automorphisms(automorphisms)
            search._roots = np.array([root])
            witness = search.find_witness(max_weight)
            assert witness is not None and is_witness(problem, witness)
            assert np.all(orbits[witness] >= orbit)


@pytest.mark.usefixtures("compilation_mode", "automorphisms_at_once")
def test_automorphisms_may_be_missing() -> None:
    """A function that finds no automorphisms leaves the search as it is."""
    assert cs.get_distance_connected_support(
        [PROBLEMS["steane-X"]], cutoff=0, get_automorphisms=lambda: None
    ) == get_brute_force_distance(PROBLEMS["steane-X"])


@pytest.mark.usefixtures("compilation_mode")
@pytest.mark.parametrize("minimum_states, found_before", [(10, 2), (30, 3)])
def test_automorphisms_are_found_before_a_search_that_is_expected_to_reach_the_threshold(
    minimum_states: int, found_before: int
) -> None:
    """Each search is expected to grow by the ratio between the last two, or not to grow after one.

    The checks of ``test_each_support_is_visited_once_above_its_least_variable`` visit 5 states with
    a weight limit of 2, and 10 with a limit of 4.  After the first search, the second is expected
    to visit 5 more states, for 10 in total.  After the second, the third is expected to visit 20,
    for 35 in total, which reaches a threshold of 30 even though only 15 have been visited.
    """
    checks = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=np.uint8)
    found = []
    with _tuning.use_tuning(automorphism_minimum_states=minimum_states):
        search = cs.SupportSearch(
            checks, get_kernel(checks), get_automorphisms=lambda: found.append(True)
        )
    for num_searches, max_weight in enumerate([2, 4, 4], start=1):
        search.find_witness(max_weight)
        assert len(found) == (num_searches >= found_before)


@pytest.mark.usefixtures("compilation_mode", "automorphisms_at_once")
def test_pruning_keeps_the_search_of_a_high_rate_code_small() -> None:
    """The pair-partition code [[276, 98, 14]] has no logical operator of weight 8 or less.

    The distance of this code is 14 (https://arxiv.org/abs/2607.14091).  There are more than 10**14
    supports of 8 qubits, and the search leaves all but thousands of them, which the bounds on the
    qubits that a support needs rule out.  The code translates its blocks of qubits cyclically,
    which leaves 12 roots.
    """
    code = codes.PairPartitionCode(EXPONENTS_X_276, EXPONENTS_Z_276, lift_size=23)
    search = cs.SupportSearch(
        *get_sectors(code)[0], get_automorphisms=code._get_qudit_automorphisms
    )
    assert search.find_witness(8) is None
    assert search.num_states < 5000


@pytest.mark.usefixtures("automorphisms_at_once")
def test_wrong_automorphisms_are_rejected() -> None:
    """A permutation that does not map the kernel of the checks to itself is not used."""
    swap = np.arange(7)[None]
    swap[0, :2] = [1, 0]
    search = cs.SupportSearch(*PROBLEMS["steane-X"], get_automorphisms=lambda: swap)
    with pytest.raises(ValueError, match="automorphism"):
        search.find_witness(3)


@pytest.mark.usefixtures("compilation_mode")
def test_the_quantum_golay_code_has_distance_seven() -> None:
    """The quantum Golay code encodes one qubit with distance 7 (https://errorcorrectionzoo.org/c/qubit_golay)."""
    sectors = get_sectors(codes.QuantumGolayCode())
    assert cs.get_distance_connected_support(sectors, cutoff=0) == 7


@pytest.mark.usefixtures("compilation_mode")
def test_sectors_are_searched_together() -> None:
    """The tetrahedral code has X-distance 7 and Z-distance 3: the Z sector bounds the X search."""
    sectors = get_sectors(codes.TetrahedralCode())
    assert [get_brute_force_distance(sector) for sector in sectors] == [7, 3]
    assert cs.get_distance_connected_support(sectors, cutoff=0) == 3


@pytest.mark.usefixtures("compilation_mode")
def test_cutoffs_and_upper_bounds() -> None:
    """A search can stop early at a witness within its cutoff, or at a known upper bound."""
    problem = PROBLEMS["toric-X"]
    assert get_brute_force_distance(problem) == 4
    assert cs.get_distance_connected_support([problem], cutoff=2) == 4
    assert 4 <= cs.get_distance_connected_support([problem], cutoff=16) <= 16

    # A known upper bound is a candidate for the distance, and the search can only lower it.
    assert cs.get_distance_connected_support([problem], cutoff=0, upper_bound=9) == 4
    assert cs.get_distance_connected_support([problem], cutoff=0, upper_bound=3) == 3
    assert cs.get_distance_connected_support([problem], cutoff=3, upper_bound=3) == 3
    assert cs.get_distance_connected_support([], cutoff=0, upper_bound=5) == 5

    # The first weight limit is the cutoff, so a search can exit at a word that is not the lightest.
    components = PROBLEMS["components"]
    assert cs.get_distance_connected_support([components], cutoff=0) == 2
    assert cs.get_distance_connected_support([components], cutoff=3) == 3


@pytest.mark.usefixtures("compilation_mode")
def test_a_problem_without_logical_operators_has_no_distance() -> None:
    """Without a vector outside the excluded rows, the distance is one more than the width."""
    trivial = (np.array([[1, 1]], dtype=np.uint8), np.array([[1, 1]], dtype=np.uint8))
    assert not cs.SupportSearch(*trivial).has_witnesses
    assert cs.get_distance_connected_support([trivial], cutoff=0) == 3
    assert cs.get_distance_connected_support([trivial, PROBLEMS["steane-X"]], cutoff=0) == 3
    assert cs.get_distance_connected_support([PROBLEMS["toric-X"], trivial], cutoff=0) == 3


@pytest.mark.usefixtures("compilation_mode")
def test_degenerate_checks() -> None:
    """Checks may be missing, zero, or repeated."""
    no_checks = np.zeros((0, 3), dtype=np.uint8)
    assert list(cs.SupportSearch(no_checks, no_checks).find_witness(1)) == [0]  # type: ignore[arg-type]
    checks, excluded = PROBLEMS["steane-X"]
    padded = np.vstack([checks, checks[0], np.zeros_like(checks[0])])
    assert cs.get_distance_connected_support([(padded, excluded)], cutoff=0) == 3


def test_the_weight_divisor_is_inferred_from_the_kernel() -> None:
    """Every vector of the kernel of the checks of the Reed-Muller code has even weight."""
    for name, divisor in [("reed-muller-X", 2), ("steane-X", 1)]:
        checks, _ = PROBLEMS[name]
        weights = {word.bit_count() for word in get_span(get_kernel(checks))}
        assert cs.SupportSearch(*PROBLEMS[name]).weight_divisor == divisor
        assert (divisor == 2) == all(weight % 2 == 0 for weight in weights)


def test_invalid_problems_are_rejected() -> None:
    """The checks and excluded rows need equal widths, and the excluded rows are in the kernel."""
    with pytest.raises(ValueError, match="equal widths"):
        cs.SupportSearch(np.zeros((1, 3), dtype=np.uint8), np.zeros((1, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="kernel"):
        cs.SupportSearch(np.ones((1, 3), dtype=np.uint8), np.array([[1, 0, 0]], dtype=np.uint8))
