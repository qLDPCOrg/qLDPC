# SPDX-License-Identifier: Apache-2.0

"""Tests of the search of the supports that are connected in a Tanner graph, and its queries."""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import math
import threading
import unittest.mock
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y

from qldpc import codes
from qldpc._binary import get_independent_rows, get_kernel, pack_bits
from qldpc.objects import PAULIS_XZ, Pauli

from ...conftest import Tune
from .. import _threads, _tuning
from ..conftest import (
    THREAD_TIMEOUT,
    check_shared_arrays_on_threads,
    get_golay_checks,
    get_span,
)
from ..distances import get_distance_css
from . import search as cs
from .sparse_checks import get_light_pair_sums, get_row_weights

####################################################################################################
# problems and helpers


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
    """A sector of the CSS code of the cyclic [23, 12, 7] Golay code (see ``get_golay_checks``)."""
    checks = get_golay_checks()
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

    ``restrict_viable_symbols``, ``stabilizer_overlap_cuts``, and ``pair_sum_checks`` default to
    their values in the default tuning.
    """
    options.setdefault("automorphism_minimum_states", 0)
    options.setdefault("restrict_viable_symbols", _tuning.DEFAULT_TUNING.restrict_viable_symbols)
    options.setdefault("stabilizer_overlap_cuts", _tuning.DEFAULT_TUNING.stabilizer_overlap_cuts)
    options.setdefault("pair_sum_checks", _tuning.DEFAULT_TUNING.pair_sum_checks)
    return cs.SupportSearch(*problem, **options)


# The signature of the compiled search, which tests that replace it read before they do.
KERNEL_SIGNATURE = inspect.signature(cs._search_supports)


def get_kernel_arguments(args: tuple[Any, ...]) -> dict[str, Any]:
    """The arguments of a call of ``_search_supports``, by name."""
    return dict(KERNEL_SIGNATURE.bind(*args).arguments)


def get_known_words(problem: Problem, symplectic: bool, pair_sums: bool) -> npt.NDArray[np.uint8]:
    """The known words of a distance search: the excluded rows, with light pair sums of some.

    With ``pair_sums``, they include the light sums of pairs of those excluded rows that are no
    heavier than the heaviest check (Section 3.4 of ``connected_cluster.md``).
    """
    checks, excluded = problem
    if not pair_sums:
        return excluded
    max_check_weight = max(get_row_weights(checks, symplectic=symplectic), default=0)
    light = excluded[get_row_weights(excluded, symplectic=symplectic) <= max_check_weight]
    return np.vstack([excluded, get_light_pair_sums(light, symplectic=symplectic)[0]])


def passes_overlap_cuts(
    words: npt.NDArray[np.uint8],
    vector: int,
    symplectic: bool,
    ranks: npt.NDArray[np.int64] | None = None,
    *,
    odd_rule: bool = True,
) -> bool:
    """Whether a vector passes the overlap cut of each word, with variables ranked by index or rank.

    Bit ``j`` of ``vector`` is entry ``j`` of the vector, as in ``get_span``.  The cut of a word
    ``g`` lets each variable of ``g`` add 2 to the load if the vector equals ``g`` there, and 1 if
    it is nonzero and differs, except the variable of ``g`` that ranks lowest, unless the problem is
    binary, ``|g|`` is odd, and ``odd_rule`` is true.  A vector passes the cut if its load is at
    most ``|g| - 1`` (Section 3.4 of ``connected_cluster.md``).
    """
    width = words.shape[1]
    num_variables = width // 2 if symplectic else width
    bits = np.array([vector >> index & 1 for index in range(width)], dtype=np.uint8)
    for word in words:
        pairs: list[tuple[int, ...]]
        values: list[tuple[int, ...]]
        if symplectic:
            pairs = list(zip(word[:num_variables], word[num_variables:]))
            values = list(zip(bits[:num_variables], bits[num_variables:]))
        else:
            pairs, values = [(bit,) for bit in word], [(bit,) for bit in bits]
        support = [index for index, pair in enumerate(pairs) if any(pair)]
        if ranks is not None:
            support.sort(key=lambda index: ranks[index])
        loaded = support if odd_rule and not symplectic and len(support) % 2 else support[1:]
        load = sum(
            2 if values[index] == pairs[index] else 1 for index in loaded if any(values[index])
        )
        if len(support) >= 2 and load > len(support) - 1:
            return False
    return True


def get_symplectic_witness_words(problem: Problem) -> set[int]:
    """The witnesses of a symplectic problem, with bit ``j`` for entry ``j`` of their vectors."""
    checks, excluded = problem
    num_qubits = checks.shape[1] // 2
    return get_span(get_kernel(np.roll(checks, num_qubits, axis=1))) - get_span(excluded)


def get_symplectic_witness_supports(problem: Problem) -> list[int]:
    """The supports of the witnesses of a symplectic problem, with bit ``j`` for qubit ``j``."""
    num_qubits = problem[0].shape[1] // 2
    words = get_symplectic_witness_words(problem)
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


####################################################################################################
# search entry points


@pytest.mark.parametrize("restrict_viable_symbols", [True, False])
@pytest.mark.parametrize("name", SEARCHED_PROBLEMS)
def test_the_distance_matches_enumeration(name: str, restrict_viable_symbols: bool) -> None:
    """The distance search, and a search through each weight limit, agree with an enumeration."""
    problem = PROBLEMS[name]
    distance = get_brute_force_distance(problem)
    for cuts, pair_sums in itertools.product((True, False), repeat=2):
        tuning = _tuning.SearchTuning(
            restrict_viable_symbols=restrict_viable_symbols,
            stabilizer_overlap_cuts=cuts,
            pair_sum_checks=pair_sums,
        )
        assert cs.get_distance_connected_cluster([problem], cutoff=0, tuning=tuning) == distance
    for max_weight, pair_sums in itertools.product(range(1, distance + 1), (True, False)):
        search = get_search(
            *problem, restrict_viable_symbols=restrict_viable_symbols, pair_sum_checks=pair_sums
        )
        witness = search.find_witness(max_weight)
        if max_weight < distance:
            assert witness is None
        else:
            assert (
                witness is not None and len(witness) <= max_weight and is_witness(problem, witness)
            )


@pytest.mark.parametrize("pair_sums", [True, False])
@pytest.mark.parametrize("cuts", [True, False])
@pytest.mark.parametrize("name", [*SEARCHED_PROBLEMS, "steane-X"])
def test_each_root_finds_the_lightest_witnesses_that_it_roots(
    name: str, cuts: bool, pair_sums: bool
) -> None:
    """A root finds a witness at the distance exactly if it is the least variable of a lightest one.

    With the overlap cuts, the lightest witness must also pass the cut of each known word: each
    excluded row, and with ``pair_sums``, the light sums of pairs of them.  Sections 3.3 and 3.4 of
    ``connected_cluster.md`` prove this completeness, with or without the sums of pairs of checks,
    and some lightest witness passes every cut.  In "steane-X", the sums of the excluded rows cut
    the lightest witnesses whose least variable is 1, but not those whose least variable is 0.
    """
    problem = PROBLEMS[name]
    checks, excluded = problem
    words = get_span(get_kernel(checks)) - get_span(excluded)
    distance = min(word.bit_count() for word in words)
    lightest = [word for word in words if word.bit_count() == distance]
    known_words = get_known_words(problem, False, pair_sums)
    admissible = [word for word in lightest if passes_overlap_cuts(known_words, word, False)]
    assert admissible
    least_variables = {
        (word & -word).bit_length() - 1 for word in (admissible if cuts else lightest)
    }
    for root in range(checks.shape[1]):
        search = get_search(*problem, stabilizer_overlap_cuts=cuts, pair_sum_checks=pair_sums)
        search._roots = np.array([root])
        witness = search.find_witness(distance)
        assert (witness is not None) == (root in least_variables)
        assert witness is None or (min(witness) == root and is_witness(problem, witness))


def get_ranked_cut_problem() -> tuple[
    npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.int_]
]:
    """Checks, excluded rows, and an automorphism whose ranks change which roots the cuts let in.

    See ``test_overlap_cuts_follow_the_ranks_of_symmetries``.
    """
    checks = np.array([[0, 0, 0, 0, 1, 0, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1]], dtype=np.uint8)
    excluded = np.array(
        [[0, 0, 1, 0, 0, 0, 0, 1], [1, 0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 1, 0, 0]],
        dtype=np.uint8,
    )
    return checks, excluded, np.array([5, 2, 3, 1, 4, 7, 6, 0])


def test_overlap_cuts_follow_the_ranks_of_symmetries() -> None:
    """With symmetries, the cuts leave out the variable of each word that ranks lowest.

    The ranks order the orbits under the permutation by their lowest indices, and the variables
    within each orbit by index.  A root finds a witness at the distance exactly if it ranks lowest
    in a lightest witness that passes every cut under these ranks (Sections 3.4 and 3.5 of
    ``connected_cluster.md``).  Under cuts by index, another root would find one.
    """
    checks, excluded, automorphism = get_ranked_cut_problem()
    num_variables = len(automorphism)
    orbits = np.arange(num_variables)
    for variable in range(num_variables):
        image = variable
        while (image := int(automorphism[image])) != variable:
            orbits[variable] = min(orbits[variable], image)
    ranks = np.argsort(np.lexsort((np.arange(num_variables), orbits)))
    roots = sorted(set(orbits.tolist()))
    assert not np.array_equal(ranks, np.arange(num_variables))

    words = get_span(get_kernel(checks)) - get_span(excluded)
    distance = min(word.bit_count() for word in words)
    expected = {}
    for name, cut_ranks in [("ranks", ranks), ("indices", None)]:
        expected[name] = {
            min(
                (index for index in range(num_variables) if word >> index & 1),
                key=ranks.__getitem__,
            )
            for word in words
            if word.bit_count() == distance
            and passes_overlap_cuts(excluded, word, False, cut_ranks)
        } & set(roots)
    assert expected["ranks"] and expected["ranks"] != expected["indices"]

    # The cuts follow the ranks also in a search that cut by index at an earlier weight limit.
    for root, earlier_limit in itertools.product(roots, [False, True]):
        search = get_search(checks, excluded)
        if earlier_limit:
            assert search.find_witness(distance - 1) is None
        search._use_automorphisms(automorphism[None])
        search._roots = np.array([root])
        witness = search.find_witness(distance)
        assert (witness is not None) == (root in expected["ranks"])
        assert witness is None or is_witness((checks, excluded), witness)
    for max_weight in range(1, distance + 1):
        search = get_search(checks, excluded, get_automorphisms=lambda: automorphism[None])
        assert (search.find_witness(max_weight) is None) == (max_weight < distance)


def get_tree_code(rng: np.random.Generator, num_bits: int) -> codes.ClassicalCode:
    """A repetition code whose checks, of two bits each, form a random tree on the bits."""
    order = rng.permutation(num_bits)
    matrix = np.zeros((num_bits - 1, num_bits), dtype=np.uint8)
    for index in range(1, num_bits):
        matrix[index - 1, [order[index], order[rng.integers(index)]]] = 1
    return codes.ClassicalCode(matrix)


def test_overlap_cuts_keep_the_distances_of_random_codes(pytestconfig: pytest.Config) -> None:
    """Random surface-like codes have the same distances with and without the cuts and pair sums.

    The hypergraph product of two repetition codes of 3 or 4 bits, with random trees of checks, has
    distance 3 or 4 and stabilizers of weight at most 5, some of them odd, whose cuts are tight on
    lightest witnesses.  BZ gives an independent answer.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for _ in range(2):
        first, second = [get_tree_code(rng, int(rng.integers(3, 5))) for _ in range(2)]
        code = codes.HGPCode(first, second)
        expected = get_distance_css(code.matrix_x, code.matrix_z, method="brouwer_zimmermann")
        assert expected >= 3
        assert 2 * expected > max(np.sum(code.matrix_x.view(np.ndarray), axis=1, dtype=int))
        assert 2 * expected > max(np.sum(code.matrix_z.view(np.ndarray), axis=1, dtype=int))
        for cuts, pair_sums in itertools.product((True, False), repeat=2):
            tuning = _tuning.SearchTuning(stabilizer_overlap_cuts=cuts, pair_sum_checks=pair_sums)
            distance = cs.get_distance_connected_cluster(get_sectors(code), cutoff=0, tuning=tuning)
            assert distance == expected


def test_odd_excluded_words_cut_their_lowest_ranked_variable_too() -> None:
    """The cut of an excluded binary word of odd weight counts all of its variables.

    The word ``g = {2, 3, 4}`` is excluded, and the lightest witnesses, such as ``{4, 5}``, weigh 2.
    The witnesses of weight at most 3 whose least variable is 2, such as ``{2, 3, 5}``, hold two
    variables of ``g``, so they fail its cut, though they would pass it without variable 2.  So the
    search from root 2 finds no witness at limit 3 (Section 3.4 of ``connected_cluster.md``).
    """
    checks = np.array([[1, 0, 1, 1, 0, 0], [0, 1, 0, 1, 1, 1]], dtype=np.uint8)
    excluded = np.array([[0, 0, 1, 1, 1, 0]], dtype=np.uint8)
    words = get_span(get_kernel(checks)) - get_span(excluded)
    rooted = [word for word in words if word.bit_count() <= 3 and (word & -word) == 1 << 2]
    assert rooted
    assert not any(passes_overlap_cuts(excluded, word, False) for word in rooted)
    assert any(passes_overlap_cuts(excluded, word, False, odd_rule=False) for word in rooted)

    search = get_search(checks, excluded)
    search._roots = np.array([2])
    assert search.find_witness(3) is None
    assert get_brute_force_distance((checks, excluded)) == 2
    witness = get_search(checks, excluded).find_witness(2)
    assert witness is not None and is_witness((checks, excluded), witness)


def test_cosets_of_classical_codes_are_cut_by_their_generators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distance of a coset of a classical code agrees with brute force, with or without cuts.

    The pair sums of Section 3.4 of ``connected_cluster.md`` keep the distance too.

    For a code with checks ``H`` and generator matrix ``G``, and a vector ``v``, the words of the
    checks ``[H | Hv]`` are the code words with a zero appended, and the vectors of the coset of
    ``v`` with a one appended.  The search excludes ``[G | 0]``, so its witnesses are the vectors of
    the coset.  The rows of ``G`` are known words, here of weights 2 and 3, so the cuts act exactly
    when the switch is on (Section 3.4 of ``connected_cluster.md``).
    """
    num_cuts_per_search: list[int] = []
    search_supports = cs._search_supports
    signature = inspect.signature(search_supports)

    def record(*args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.int64], ...]:
        num_cuts_per_search.append(len(signature.bind(*args, **kwargs).arguments["cut_capacities"]))
        return search_supports(*args, **kwargs)

    monkeypatch.setattr(cs, "_search_supports", record)
    weights = set()
    for seed in (2, 7):
        code = codes.ClassicalCode.random(6, 3, seed=seed)
        matrix = code.matrix.view(np.ndarray).astype(np.uint8)
        generator = code.generator.view(np.ndarray).astype(np.uint8)
        weights |= set(np.sum(generator, axis=1).tolist())
        excluded = np.hstack([generator, np.zeros((len(generator), 1), dtype=np.uint8)])
        for vector in range(2**6):
            bits = np.array([vector >> index & 1 for index in range(6)], dtype=np.uint8)
            checks = np.hstack([matrix, (matrix @ bits % 2)[:, None]]).astype(np.uint8)
            expected = get_brute_force_distance((checks, excluded))
            for cuts, pair_sums in itertools.product((True, False), repeat=2):
                num_cuts_per_search.clear()
                tuning = _tuning.SearchTuning(
                    stabilizer_overlap_cuts=cuts, pair_sum_checks=pair_sums
                )
                distance = cs.get_distance_connected_cluster(
                    [(checks, excluded)], cutoff=0, tuning=tuning
                )
                assert distance == expected
                assert num_cuts_per_search
                assert all(bool(num_cuts) == cuts for num_cuts in num_cuts_per_search)
    assert {2, 3} <= weights


def query_symbols(search: cs.SupportSearch) -> list[int] | None:
    """The symbols that a query for the syndrome 01 at limit 2 returns, or None."""
    vector = search.find_vector_with_syndrome([0, 1], 2)
    return None if vector is None else vector.tolist()


def test_query_overlap_words_cut_queries_but_not_witnesses() -> None:
    """Words outside the excluded rows cut syndrome queries, but not the search for witnesses.

    The word 111 of these checks is not excluded, so it is a witness, and its cut would forbid it.
    Its cut lets a query at limit 2 return only the lighter vector of its syndrome.  The search adds
    no sum of the two checks, with which the query would return the lighter vector even without the
    cut.
    """
    checks = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    search = get_search(checks, np.zeros((0, 3), dtype=np.uint8), pair_sum_checks=False)
    assert query_symbols(search) == [0, 1]
    search.use_query_overlap_words([[1, 1, 1]])
    witness = search.find_witness(3)
    assert witness is not None and witness.tolist() == [0, 1, 2]
    assert query_symbols(search) == [2]


def test_queries_cut_under_the_current_ranks_and_only_read_the_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queries cut under the ranks of symmetries learned late, and tabulate no cuts themselves.

    The checks are the Z-type checks of the rotated surface code of distance 3, whose X-type checks
    are the excluded rows and the query words, and the symmetry is its turn by half a circle.  A
    search that tabulated the cuts of its queries before it learns the symmetry answers every query
    as a search that learns it first, with a vector within the limit exactly if there is one.  The
    ranks of the symmetry move the variables that some cuts leave out, so that if the first search
    kept its earlier cuts, it would return other vectors for some syndromes, such as the support
    {3} rather than {2, 4} for the syndrome 0010.  After ``use_query_overlap_words``, and after the
    symmetry, queries tabulate no cuts.
    """
    tabulated: list[object] = []
    get_overlap_cuts = cs._get_overlap_cuts

    def record(*args: Any) -> Any:
        tabulated.append(args)
        return get_overlap_cuts(*args)

    monkeypatch.setattr(cs, "_get_overlap_cuts", record)
    code = codes.SurfaceCode(3)
    checks = code.matrix_z.view(np.ndarray).astype(np.uint8)
    words = code.matrix_x.view(np.ndarray).astype(np.uint8)
    turn = np.arange(9)[::-1].copy()
    early = get_search(checks, words)
    early.use_query_overlap_words(words)
    early._use_automorphisms(turn[None])
    late = get_search(checks, words)
    late._use_automorphisms(turn[None])
    late.use_query_overlap_words(words)
    num_tabulated = len(tabulated)

    vectors = np.array(list(itertools.product([0, 1], repeat=9)), dtype=np.uint8)
    for syndrome in itertools.product([0, 1], repeat=len(checks)):
        weights = vectors[np.all(vectors @ checks.T % 2 == syndrome, axis=1)].sum(axis=1)
        answers = [search.find_vector_with_syndrome(syndrome, 2) for search in (early, late)]
        symbols = [None if answer is None else answer.tolist() for answer in answers]
        assert symbols[0] == symbols[1]
        assert (answers[0] is None) == (weights.min() > 2)
        if answers[0] is not None:
            vector = get_vector(answers[0], 9, False)
            assert np.array_equal(checks @ vector % 2, syndrome) and vector.sum() <= 2
    assert len(tabulated) == num_tabulated


def test_query_overlap_words_must_satisfy_the_checks_and_need_the_switch() -> None:
    """Query words must be a binary matrix as wide as the checks, and satisfy them.

    They cut only with the switch on.
    """
    checks = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
    search = get_search(
        checks,
        np.zeros((0, 3), dtype=np.uint8),
        stabilizer_overlap_cuts=False,
        pair_sum_checks=False,
    )
    with pytest.raises(ValueError, match="satisfy the checks"):
        search.use_query_overlap_words([[1, 1, 0]])
    with pytest.raises(ValueError, match="as wide"):
        search.use_query_overlap_words([[1, 1]])
    with pytest.raises(ValueError, match="as wide"):
        search.use_query_overlap_words([1, 1, 1])
    with pytest.raises(ValueError, match="binary"):
        search.use_query_overlap_words([[2, 0, 0]])
    search.use_query_overlap_words([[1, 1, 1]])
    assert query_symbols(search) == [0, 1]


@pytest.mark.parametrize("num_threads", [1, 8])
def test_each_support_is_visited_once_above_its_least_variable(
    num_threads: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With every word excluded, a search visits the supports that grow from each root once.

    The overlap cuts are off, since the excluded words would cut some of these supports, and so are
    the sums of pairs of checks, which would add the check {0, 3}.  On threads, every limit splits.
    """
    # The checks {0, 1, 2} and {1, 2, 3} have the words {0, 1, 3}, {0, 2, 3}, and {1, 2}.  At limit
    # 4, root 0 visits {0}, {0, 1}, {0, 1, 2}, {0, 1, 3}, {0, 2}, and {0, 2, 3}, since its branch on
    # 1 forbids 1 for its sibling {0, 2}.  Root 1 visits {1} and {1, 2}, and roots 2 and 3 visit
    # only themselves.  At limit 2, {0, 1} and {0, 2} leave a check unsatisfied, so they would need
    # a third variable.
    checks = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=np.uint8)
    calls = record_thread_counts(monkeypatch)
    num_states = {}
    for max_weight in (2, 4):
        search = get_search(
            checks,
            get_kernel(checks),
            stabilizer_overlap_cuts=False,
            pair_sum_checks=False,
            num_threads=num_threads,
            split_minimum_states=-1,
        )
        assert search.find_witness(max_weight) is None
        num_states[max_weight] = search.num_states
    assert num_states == {2: 5, 4: 6 + 2 + 1 + 1}
    assert bool(calls) == (num_threads > 1)


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
    automorphisms = np.argsort(order)[code._get_permutation_automorphisms()[:, order]]
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
        stabilizer_overlap_cuts=False,
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
    search = get_search(
        *get_sectors(code)[0], get_automorphisms=code._get_permutation_automorphisms
    )
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
    # it would also visit a child of {1}, and the equality would fail.  The counts rely on ties
    # going to the first check, so a change of the tie-break needs a new instance.
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


@pytest.mark.parametrize(
    "switch", ["restrict_viable_symbols", "stabilizer_overlap_cuts", "pair_sum_checks"]
)
@pytest.mark.parametrize("expected", [True, False])
def test_connected_cluster_distance_searches_read_the_switches_of_their_tuning(
    monkeypatch: pytest.MonkeyPatch, switch: str, expected: bool
) -> None:
    """Distance searches pass the switches of their tuning to ``SupportSearch``.

    Without a tuning, they read the active one.
    """
    changes = {switch: expected}
    settings: list[bool] = []
    initialize = cs.SupportSearch.__init__
    signature = inspect.signature(initialize)

    def record(*args: Any, **kwargs: Any) -> None:
        settings.append(signature.bind(*args, **kwargs).arguments[switch])
        initialize(*args, **kwargs)

    monkeypatch.setattr(cs.SupportSearch, "__init__", record)
    sectors = get_sectors(codes.SteaneCode())
    with _tuning.use_tuning(**changes):
        cs.get_distance_connected_cluster(sectors, cutoff=0)
    assert settings and set(settings) == {expected}

    # Searches built with a tuning use it, whatever the active tuning, and the function that builds
    # them passes its tuning on.
    settings.clear()
    tuning = dataclasses.replace(_tuning.DEFAULT_TUNING, **changes)
    with _tuning.use_tuning(**{switch: not expected}):
        cs.get_distance_connected_cluster(sectors, cutoff=0, tuning=tuning)
    assert settings and set(settings) == {expected}


def test_pair_sums_of_checks_save_states() -> None:
    """Light sums of pairs of checks leave fewer states to visit before a lightest witness.

    These problems have such sums (Section 3.4 of ``connected_cluster.md``).
    """
    problems = [
        (PROBLEMS["nested-restrictions"], False),
        (PROBLEMS["reed-muller-X"], False),
        (QUBIT_PROBLEMS["eight-qubit"], True),
    ]
    for problem, symplectic in problems:
        distance = (
            min(support.bit_count() for support in get_symplectic_witness_supports(problem))
            if symplectic
            else get_brute_force_distance(problem)
        )
        num_states = []
        for pair_sums in (False, True):
            search = get_search(*problem, symplectic=symplectic, pair_sum_checks=pair_sums)
            for max_weight in range(1, distance):
                assert search.find_witness(max_weight) is None
            assert search.find_witness(distance) is not None
            num_states.append(search.num_states)
        assert num_states[1] < num_states[0]


def test_only_known_words_no_heavier_than_the_heaviest_check_are_summed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A distance search sums no pair of known words with one heavier than the heaviest check.

    The sums of heavy words, such as the rows of the generator matrix that the distance of a vector
    excludes, would cost much to find (Section 3.4 of ``connected_cluster.md``).  Here the checks
    weigh 2, 3, and 2, and the excluded rows 2, 3, and 4, so the search sums only the first two.
    """
    summed: list[list[list[int]]] = []

    def record(rows: npt.NDArray[np.uint8], **kwargs: Any) -> Any:
        summed.append(rows.tolist())
        return get_light_pair_sums(rows, **kwargs)

    monkeypatch.setattr(cs, "get_light_pair_sums", record)
    checks = np.array(
        [[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 0]],
        dtype=np.uint8,
    )
    excluded = np.array(
        [[1, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 1], [1, 1, 0, 0, 0, 1, 1, 0]],
        dtype=np.uint8,
    )
    get_search(checks, excluded, pair_sum_checks=True)
    assert summed[-1] == excluded[:2].tolist()


def test_syndrome_queries_cut_by_no_sums_of_known_words() -> None:
    """Syndrome queries cut by the excluded rows, but not by the sums of pairs of them.

    Only distance searches cut by these sums.  The two excluded rows sum to the light word {4, 5}.
    Its cut would forbid variable 5 from the start, and the query at limit 3 would return
    {0, 2, 3} rather than {0, 4} (Sections 3.4 and 3.8 of ``connected_cluster.md``).
    """
    checks = np.array([[0, 1, 1, 1, 0, 0], [0, 0, 1, 0, 1, 1], [1, 1, 1, 0, 1, 1]], dtype=np.uint8)
    excluded = np.array([[1, 1, 1, 0, 1, 0], [1, 1, 1, 0, 0, 1]], dtype=np.uint8)
    assert get_known_words((checks, excluded), False, True).tolist()[2:] == [[0, 0, 0, 0, 1, 1]]
    search = get_search(checks, excluded, pair_sum_checks=True, stabilizer_overlap_cuts=True)
    vector = search.find_vector_with_syndrome([0, 1, 0], 3)
    assert vector is not None and vector.tolist() == [0, 4]


def test_wrong_automorphisms_are_rejected() -> None:
    """A permutation that does not map the kernel of the checks to itself raises an error."""
    swap = np.arange(7)[None]
    swap[0, :2] = [1, 0]
    search = get_search(*PROBLEMS["steane-X"], get_automorphisms=lambda: swap)
    with pytest.raises(ValueError, match="automorphism"):
        search.find_witness(3)


@pytest.mark.parametrize("symplectic", [False, True])
def test_automorphisms_must_keep_the_row_space_of_the_excluded_rows(symplectic: bool) -> None:
    """A permutation that keeps the words but not the row space of the excluded rows is rejected.

    Every permutation of the variables keeps the words of the check 111, or of the Pauli XXX.  The
    swap of variables 0 and 1 keeps the excluded row 110, or Z0 Z1, but the swap of 1 and 2 does
    not.  Accepting the second swap would let the search skip variable 2 as a root, which is sound
    only for permutations that keep the excluded rows too, so it could give a wrong distance
    (Section 3.5 of ``connected_cluster.md``).  A symplectic problem moves the X and Z parts of each
    qubit together.
    """
    if symplectic:
        problem = (
            np.array([[1, 1, 1, 0, 0, 0]], dtype=np.uint8),
            np.array([[0, 0, 0, 1, 1, 0]], dtype=np.uint8),
        )
    else:
        problem = (np.array([[1, 1, 1]], dtype=np.uint8), np.array([[1, 1, 0]], dtype=np.uint8))
    keeps, moves = np.array([[1, 0, 2]]), np.array([[0, 2, 1]])
    witness = get_search(
        *problem, symplectic=symplectic, get_automorphisms=lambda: keeps
    ).find_witness(2)
    assert witness is not None
    assert (is_symplectic_witness if symplectic else is_witness)(problem, witness)
    search = get_search(*problem, symplectic=symplectic, get_automorphisms=lambda: moves)
    with pytest.raises(ValueError, match="not an automorphism"):
        search.find_witness(2)


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


def test_stepped_searches_advance_the_sector_with_the_lowest_lower_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each advance searches the sector with the lowest lower bound until the search finishes."""
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
    # The sectors have the divisors 2 and 1, so advancing the sector with the lowest lower bound
    # gives a different order from round-robin.
    sectors = [PROBLEMS["toric-X"], PROBLEMS["steane-X"]]
    search = cs.ConnectedClusterSearch(sectors, cutoff=0)
    lower_bounds = [sector_search.weight_divisor for sector_search in searches_made]
    assert lower_bounds == [2, 1]
    while not search.finished:
        num_limits = len(limits)
        search.advance()
        assert len(limits) == num_limits + 1
        index, max_weight = limits[-1]
        assert max_weight == lower_bounds[index] == min(lower_bounds)
        lower_bounds[index] += searches_made[index].weight_divisor
    assert search.result == 3 and [max_weight for _, max_weight in limits] == [1, 2, 2, 3]

    # Once finished, the search does nothing.
    num_limits = len(limits)
    search.advance()
    assert len(limits) == num_limits


def test_the_logical_test_reads_every_block() -> None:
    """The logical test finds a witness that fails only tests past the first block of 64."""
    problem = PROBLEMS["steane-X"]
    search = get_search(*problem)
    zeros = np.zeros((search.num_variables, 1), dtype=np.uint64)
    search._check_tables = search._check_tables._replace(
        test_columns=np.hstack([zeros, search._check_tables.test_columns])
    )
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


@pytest.mark.parametrize("pair_sums", [True, False])
@pytest.mark.parametrize("cuts", [True, False])
@pytest.mark.parametrize("restrict_viable_symbols", [True, False])
def test_random_qubit_problems_match_enumeration(
    pytestconfig: pytest.Config, restrict_viable_symbols: bool, cuts: bool, pair_sums: bool
) -> None:
    """Qubit problems agree with an enumeration at every weight limit, and from every root.

    With the overlap cuts, a root finds a lightest witness only if one passes the cut of every known
    word, which with ``pair_sums`` include the light sums of pairs of gauge generators.
    """
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
    options = {
        "restrict_viable_symbols": restrict_viable_symbols,
        "stabilizer_overlap_cuts": cuts,
        "pair_sum_checks": pair_sums,
    }
    tuning = _tuning.SearchTuning(**options)
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
            search = get_search(*problem, symplectic=True, **options)
            witness = search.find_witness(max_weight)
            assert (witness is None) == (max_weight < distance)
            assert witness is None or (
                len(witness) <= max_weight and is_symplectic_witness(problem, witness)
            )

        lightest = [
            word
            for word in get_symplectic_witness_words(problem)
            if ((word | word >> num_qubits) % 2**num_qubits).bit_count() == distance
        ]
        if cuts:
            known_words = get_known_words(problem, True, pair_sums)
            lightest = [word for word in lightest if passes_overlap_cuts(known_words, word, True)]
            assert lightest
        least_qubits = {
            (support & -support).bit_length() - 1
            for support in ((word | word >> num_qubits) % 2**num_qubits for word in lightest)
        }
        for root in range(num_qubits):
            search = get_search(*problem, symplectic=True, **options)
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

    # Each of the two sectors has 16 variables and weight divisor 2, so a correct search needs at
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


def test_stepped_searches_report_their_progress_and_take_caps() -> None:
    """Stepped searches report their states and the growth of their states, and stop at a cap."""
    sector = get_golay_problem()
    assert get_brute_force_distance(sector) == 7
    search = cs.ConnectedClusterSearch([sector], cutoff=0)
    (sector_search,) = search.searches
    assert search.next_limit == (0, 1)
    assert search.num_states == 0 and search.get_growth(0) is None
    assert search.predict_states(0, 1) == sector_search._predict_states(1)

    # Each limit without a witness raises the lower bound past it, and records its states.  The
    # growth is the ratio of states per unit of weight while there is only one, and then the
    # geometric mean of the last two ratios; the first limit visits no states, so it gives no ratio.
    states = []
    for max_weight in range(1, 5):
        assert search.next_limit == (0, max_weight)
        num_states = search.num_states
        search.advance()
        states.append(search.num_states - num_states)
        assert search.next_limit == (0, max_weight + 1) and not search.finished
        if max_weight == 3:
            assert search.get_growth(0) == pytest.approx(states[2] / states[1])
    assert states[0] == 0 and all(states[1:])
    growth = search.get_growth(0)
    assert growth is not None and growth == pytest.approx(np.sqrt(states[3] / states[1]))
    assert search.predict_states(0, 7) == pytest.approx(states[3] * growth**3)
    assert search.predict_states(0, 10**4) == math.inf

    # A weaker cap changes nothing, and a cap at the next lower bound ends the search there.
    search.update_upper_bound(30)
    search.update_upper_bound(6)
    search.update_upper_bound(8)
    assert not search.finished
    search.advance()
    assert search.finished and search.result == 6


def test_the_growth_of_states_skips_a_change_of_roots() -> None:
    """The growth compares only limits from the same number of roots, which automorphisms reduce.

    The search asks for the cyclic shifts once it has visited as many states as a search without
    them visits through limit 3.  Since the states grow from limit to limit, it asks before limit 4,
    which then searches from fewer roots.
    """
    plain = cs.ConnectedClusterSearch(
        [get_golay_problem()], cutoff=0, tuning=_tuning.DEFAULT_TUNING
    )
    for _ in range(3):
        plain.advance()
    shifts = np.roll(np.arange(23), 1)[None]
    states: list[int] = []
    growths: list[float | None] = []
    asked: list[int] = []

    def get_automorphisms() -> npt.NDArray[np.int_]:
        asked.append(len(states))
        return shifts

    tuning = dataclasses.replace(
        _tuning.DEFAULT_TUNING, automorphism_minimum_states=plain.num_states
    )
    search = cs.ConnectedClusterSearch(
        [get_golay_problem()], cutoff=0, get_automorphisms=get_automorphisms, tuning=tuning
    )
    for _ in range(5):
        num_states = search.num_states
        search.advance()
        states.append(search.num_states - num_states)
        growths.append(search.get_growth(0))
    assert asked == [3]
    assert growths[2] == growths[3] == pytest.approx(states[2] / states[1])
    assert growths[4] == pytest.approx(np.sqrt(states[2] / states[1] * states[4] / states[3]))


def test_stepped_searches_add_the_states_of_their_sectors_and_grow_per_unit_of_weight() -> None:
    """The states add over sectors, and the growth is per unit of weight, not per weight limit.

    Both sectors of the toric code of size 6 have only words of even weight, so the limits of each
    rise in steps of two, and the growth is the square root of the ratio of their states.
    """
    search = cs.ConnectedClusterSearch(get_sectors(codes.ToricCode(6)), cutoff=0)
    assert [sector_search.weight_divisor for sector_search in search.searches] == [2, 2]
    records = []
    for _ in range(4):
        index, max_weight = search.next_limit
        num_states = search.searches[index].num_states
        search.advance()
        records.append((index, max_weight, search.searches[index].num_states - num_states))
    assert [record[:2] for record in records] == [(0, 2), (1, 2), (0, 4), (1, 4)]
    assert all(states for *_, states in records) and not search.finished
    assert search.num_states == sum(states for *_, states in records)
    growth = (records[2][2] / records[0][2]) ** (1 / 2)
    assert search.get_growth(0) == pytest.approx(growth)
    assert search.predict_states(0, 8) == pytest.approx(records[2][2] * growth**4)


def test_a_problem_without_logical_operators_has_no_distance() -> None:
    """Without witnesses, a sector counts as one more than its number of variables."""
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


def test_distinct_rows_are_found_as_numpy_finds_them(pytestconfig: pytest.Config) -> None:
    """``_get_unique_rows`` gives the first copies and positions of ``np.unique``, in its order.

    Widths around a multiple of 8 test the packing of rows into bytes.
    """
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    for width in [1, 7, 8, 9, 70]:
        matrix = rng.integers(2, size=(12, width), dtype=np.uint8)
        matrix[5] = matrix[0]
        _, first_rows, positions = np.unique(matrix, axis=0, return_index=True, return_inverse=True)
        assert [array.tolist() for array in cs._get_unique_rows(matrix)] == [
            first_rows.tolist(),
            positions.ravel().tolist(),
        ]


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


@pytest.mark.parametrize("pair_sums", [True, False])
@pytest.mark.parametrize("name", SYNDROME_PROBLEMS)
def test_syndrome_queries_match_enumeration(name: str, pair_sums: bool) -> None:
    """A query finds a vector with a syndrome exactly if the limit reaches the least weight.

    With ``pair_sums``, the search maps the syndrome to the light sums of pairs of its checks.
    """
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

    search = get_search(
        checks, np.zeros((0, width)), symplectic=symplectic, pair_sum_checks=pair_sums
    )
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
    with pytest.raises(ValueError, match="at least one sector or an upper bound"):
        cs.get_distance_connected_cluster([], cutoff=0)
    with pytest.raises(ValueError, match="even number of columns"):
        get_search(
            np.ones((1, 3), dtype=np.uint8), np.ones((0, 3), dtype=np.uint8), symplectic=True
        )


####################################################################################################
# weight limits on threads


def record_thread_counts(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the number of threads of each call of ``run_work_items``, which still runs."""
    calls: list[int] = []
    run_work_items = _threads.run_work_items

    def record(num_items: int, num_threads: int, *args: Any, **kwargs: Any) -> None:
        calls.append(num_threads)
        run_work_items(num_items, num_threads, *args, **kwargs)

    monkeypatch.setattr(_threads, "run_work_items", record)
    return calls


def record_witness_positions(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record where the witness of each limit on threads comes from, in wrappers that still run.

    The witness is in the first work item, a middle one, or the last, or above the frontier, after
    every item.  A limit without a witness records nothing.  The wrappers number the item of each
    claim, which holds one item.
    """
    # The positions, the items of the current limit whose searches found a witness, and whether a
    # search above the frontier found one.
    positions: list[str] = []
    found_items: list[int] = []
    above: list[bool] = []
    next_item = threading.local()
    run_work_items = _threads.run_work_items
    search_supports = cs._search_supports
    signature = inspect.signature(search_supports)

    def run_numbered(
        num_items: int,
        num_threads: int,
        start_worker: Callable[[threading.Event], Callable[[int, int], None]],
        **kwargs: Any,
    ) -> None:
        # Each claim numbers its item, on the thread that searches it.
        def start_numbered(stop: threading.Event) -> Callable[[int, int], None]:
            run_range = start_worker(stop)

            def run_numbered_range(start: int, end: int) -> None:
                next_item.index = start
                try:
                    run_range(start, end)
                finally:
                    del next_item.index

            return run_numbered_range

        # Run the limit, and record the position of the first item that found a witness, or else
        # whether the witness came from above the frontier.
        found_items.clear()
        run_work_items(num_items, num_threads, start_numbered, **kwargs)
        if found_items:
            first = min(found_items)
            positions.append(
                "first" if first == 0 else "last" if first == num_items - 1 else "middle"
            )
        elif above:
            positions.append("above")
        above.clear()

    # A search of an item records whether it found a witness, and so does a search above the
    # frontier, which has no item.
    def search_numbered(*args: Any, **kwargs: Any) -> Any:
        result = search_supports(*args, **kwargs)
        item = getattr(next_item, "index", None)
        if item is not None:
            if len(result[0]):
                found_items.append(item)
        elif signature.bind(*args, **kwargs).arguments["frontier_depth"] and len(result[0]):
            above.append(True)
        return result

    monkeypatch.setattr(_threads, "run_work_items", run_numbered)
    monkeypatch.setattr(cs, "_search_supports", search_numbered)
    return positions


def get_threaded_cases() -> Iterator[
    tuple[Problem, bool, Callable[[], npt.NDArray[np.int_]] | None]
]:
    """Problems, whether each is symplectic, and how to get automorphisms, for threaded searches.

    The searches of these problems split some weight limits into items so that the first witness
    is in the first item, a middle one, or the last, or comes after every item, above the frontier
    at which the items start.  Other limits have no witness.
    """
    for problem in PROBLEMS.values():
        yield problem, False, None
    for problem in QUBIT_PROBLEMS.values():
        yield problem, True, None
    # The Golay code has one root under its cyclic shifts, and 23 without them.
    yield get_golay_problem(), False, None
    yield get_golay_problem(), False, lambda: np.roll(np.arange(23), -1)[None]
    # Only the last root is a witness, and on up to three threads, its 100 roots are the items.
    checks = np.hstack([np.eye(99, dtype=np.uint8), np.zeros((99, 1), dtype=np.uint8)])
    yield (checks, np.zeros((0, 100), dtype=np.uint8)), False, None


def test_limits_on_threads_give_the_witnesses_and_states_of_one_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In chunks and on threads, each weight limit finds the witness and states of one call.

    The limits run past the distance, with the restriction to viable symbols and the overlap cuts
    both on or both off (Section 3.11 of ``connected_cluster.md``).  Chunks of one state resume the
    search after every state, on one thread and on threads.  The threads run with every array of
    the search read-only.  The first witness of some limit is in each of the positions that
    ``get_threaded_cases`` names, in a case with at least as many roots as the 8 threads.  So a
    witness above the frontier shows that the frontier deepens to more items than one per thread.
    """
    calls = record_thread_counts(monkeypatch)
    positions = record_witness_positions(monkeypatch)
    # The roots and paths of the work items, which _search_on_threads makes for the threads, are not
    # reached from the search, so they stay writeable (Section 5 of connected_cluster.md).
    checked = check_shared_arrays_on_threads(monkeypatch, lambda *_, **__: [search])
    many_root_positions: set[str] = set()
    for (problem, symplectic, get_automorphisms), switch in itertools.product(
        get_threaded_cases(), [True, False]
    ):
        num_positions = len(positions)
        results = []
        for num_threads, chunk_states in [
            (1, cs.MAX_VALUE),
            (3, cs.MAX_VALUE),
            (8, cs.MAX_VALUE),
            (1, 1),
            (8, 1),
        ]:
            search = get_search(
                *problem,
                symplectic=symplectic,
                get_automorphisms=get_automorphisms,
                restrict_viable_symbols=switch,
                stabilizer_overlap_cuts=switch,
                num_threads=num_threads,
                split_minimum_states=-1,
                chunk_states=chunk_states,
            )
            results.append(
                [
                    (None if witness is None else witness.tolist(), search.num_states)
                    for witness in map(search.find_witness, range(1, 8))
                ]
            )
        assert all(result == results[0] for result in results)
        # Without automorphisms, every variable is a root.
        num_variables = problem[0].shape[1] // 2 if symplectic else problem[0].shape[1]
        if get_automorphisms is None and num_variables >= 8:
            many_root_positions.update(positions[num_positions:])
    assert set(calls) == {3, 8}
    assert len(checked) == len(calls) and min(checked) > 0
    assert many_root_positions == {"first", "middle", "last", "above"}


def test_limits_cut_with_the_latest_tables_tabulated_before_their_threads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each limit cuts with the tables of its current ranks, which it tabulates before its threads.

    A wrapper of ``_get_overlap_cuts`` records the ranks and the tables of each tabulation, and a
    wrapper of ``_search_supports`` checks that each call gets the latest of them, the same objects,
    and so never tables of ranks from before a symmetry.  The searches of two problems learn their
    symmetries at a later limit, and tabulate their cuts again at the start of that limit:

    - The problem of ``test_overlap_cuts_follow_the_ranks_of_symmetries`` learns its permutation at
      its third limit, at which the new tables give the witness of a search that learns it first,
      [0, 5], where the tables of its ranks by index would give [0, 1].  With a threshold of one
      state, this relies on the search predicting no states for its first two limits.
    - The Golay code learns its cyclic shifts once it has visited as many states as a search without
      them visits through its third limit, a number of states that its first limit does not reach.
      It runs on four threads, with a split gate that splits every limit, and with
      ``ITEMS_PER_THREAD`` set to one.  Its first limit then runs on threads under the tables of its
      ranks by index, since its 23 roots are enough items.  Its last limit, which has a witness,
      runs on threads under the tables that it tabulated again with the shifts.  It finds the
      witnesses, and visits the states, of one thread.

    A limit tabulates its cuts before it calls ``_search_on_threads``, so no tabulation can happen
    while threads run.  The test checks this as well.
    """
    events: list[str] = []
    tabulations: list[tuple[npt.NDArray[np.int64], tuple[npt.NDArray[np.int64], ...]]] = []
    get_overlap_cuts = cs._get_overlap_cuts
    search_supports = cs._search_supports
    run_work_items = _threads.run_work_items

    def record_cuts(words: Any, ranks: npt.NDArray[np.int64], symplectic: bool) -> Any:
        events.append("cuts")
        tables = get_overlap_cuts(words, ranks, symplectic)
        tabulations.append((ranks, tables))
        return tables

    def check_cuts(*args: Any) -> Any:
        ranks, tables = tabulations[-1]
        # Argument 7 is the ranks, and the last six are the tables of the cuts.
        assert args[7] is ranks
        assert all(passed is table for passed, table in zip(args[-6:], tables, strict=True))
        return search_supports(*args)

    def record_threads(*args: Any, **kwargs: Any) -> None:
        events.append("start")
        run_work_items(*args, **kwargs)
        events.append("end")

    monkeypatch.setattr(cs, "_get_overlap_cuts", record_cuts)
    monkeypatch.setattr(cs, "_search_supports", check_cuts)
    monkeypatch.setattr(_threads, "run_work_items", record_threads)

    # The 8-variable problem learns its permutation at its first limit, or at its third.
    checks, excluded, automorphism = get_ranked_cut_problem()
    witnesses = []
    for minimum_states in (0, 1):
        search = get_search(
            checks,
            excluded,
            get_automorphisms=lambda: automorphism[None],
            automorphism_minimum_states=minimum_states,
        )
        witnesses.append([search.find_witness(max_weight) for max_weight in (1, 2, 3)])
    assert len(tabulations) == 3
    symbols = [[None if w is None else w.tolist() for w in limits] for limits in witnesses]
    assert symbols[1] == [None, [0, 1], [0, 5]] and symbols[0][2] == [0, 5]

    # The Golay code learns its shifts at a limit after its first, on one thread and on four, which
    # split its limits into items of one per thread.
    plain = get_search(*get_golay_problem())
    for max_weight in (1, 2, 3):
        plain.find_witness(max_weight)
    monkeypatch.setattr(_threads, "ITEMS_PER_THREAD", 1)
    results = []
    for num_threads in (1, 4):
        events.clear()
        search = get_search(
            *get_golay_problem(),
            get_automorphisms=lambda: np.roll(np.arange(23), -1)[None],
            automorphism_minimum_states=plain.num_states,
            num_threads=num_threads,
            split_minimum_states=-1,
        )
        golay_witnesses = [search.find_witness(max_weight) for max_weight in range(1, 8)]
        results.append(
            ([None if w is None else w.tolist() for w in golay_witnesses], search.num_states)
        )
        assert events.count("cuts") == 2
    assert results[0] == results[1] and results[0][0][-1] is not None
    starts = [index for index, event in enumerate(events) if event == "start"]
    cuts = [index for index, event in enumerate(events) if event == "cuts"]
    assert cuts[0] < starts[0] < cuts[1] < starts[-1]
    assert all(events[start + 1] == "end" for start in starts)


@pytest.mark.parametrize("cutoff", [0, 6])
def test_distance_searches_on_threads_take_the_steps_of_one_thread(
    cutoff: int, tune: Tune, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A distance search on threads takes the limits, and counts the states, of one thread.

    At exponent 0, every limit on eight threads that is predicted to visit more than two states
    splits, into many items.  A cutoff makes the search exit at its first limit, which visits
    states; on eight threads, the frontier passes of that limit find the witness.  Except at that
    limit on eight threads, the searches of the limits run in chunks of
    ``2**search_chunk_exponent`` states, here one, and some stop early (Section 3.6 of
    ``connected_cluster.md``).
    """
    tune(search_chunk_exponent=0)
    calls = record_thread_counts(monkeypatch)
    search_supports = cs._search_supports
    # The budget of each call of a limit, and whether it stopped early.
    chunks: list[tuple[int, bool]] = []

    def record_chunks(*args: Any) -> tuple[npt.NDArray[np.int64], ...]:
        result = search_supports(*args)
        arguments = get_kernel_arguments(args)
        if len(arguments["roots"]) and not arguments["frontier_depth"]:
            chunks.append((arguments["max_states"], bool(len(result[3]))))
        return result

    monkeypatch.setattr(cs, "_search_supports", record_chunks)
    sectors = get_sectors(codes.ToricCode(4))
    steps: list[list[Any]] = []
    for num_threads in (1, 8):
        tune(num_threads=num_threads)
        chunks.clear()
        search = cs.ConnectedClusterSearch(sectors, cutoff=cutoff)
        steps.append([])
        while not search.finished:
            limit = search.next_limit
            search.advance()
            steps[-1].append((limit, search.num_states))
        steps[-1].append(search.result)
        if num_threads == 1 or not cutoff:
            assert {budget for budget, _ in chunks} == {1}
            assert any(stopped for _, stopped in chunks)
    assert steps[0] == steps[1]
    assert calls and set(calls) == {8}
    assert (steps[0][-1] == 4) if cutoff == 0 else (4 <= steps[0][-1] <= cutoff)


def test_the_frontier_deepens_to_about_a_chunk_of_states_for_each_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The frontier of a limit on threads deepens to the predicted number of chunks of states.

    The Golay code has one root under its cyclic shifts.  With chunks of one state, its last three
    limits on two threads predict more than 40 chunks, so their frontiers deepen to at least
    ``_MAX_ITEMS`` items, here 40.  With chunks without a limit, every frontier stops once it has
    ``ITEMS_PER_THREAD`` items for each thread, here one, which takes fewer than 40 items (Section
    3.11 of ``connected_cluster.md``).
    """
    calls = record_thread_counts(monkeypatch)
    num_items: list[int] = []
    run_work_items = _threads.run_work_items

    def record_items(count: int, *args: Any, **kwargs: Any) -> None:
        num_items.append(count)
        run_work_items(count, *args, **kwargs)

    monkeypatch.setattr(_threads, "run_work_items", record_items)
    monkeypatch.setattr(_threads, "ITEMS_PER_THREAD", 1)
    monkeypatch.setattr(cs, "_MAX_ITEMS", 40)
    items = []
    for chunk_states in (1, cs.MAX_VALUE):
        num_items.clear()
        search = get_search(
            *get_golay_problem(),
            get_automorphisms=lambda: np.roll(np.arange(23), -1)[None],
            num_threads=2,
            split_minimum_states=-1,
            chunk_states=chunk_states,
        )
        for max_weight in range(1, 8):
            if max_weight >= 5:
                assert search._predict_states(max_weight) > 40
            search.find_witness(max_weight)
        items.append(list(num_items))
    assert len(items[0]) >= 3 and min(items[0][-3:]) >= 40
    assert items[1] and max(items[1]) < 40
    assert calls


def test_limits_split_across_threads_only_if_costly(
    monkeypatch: pytest.MonkeyPatch, tune: Tune
) -> None:
    """A limit runs on threads only if it is predicted to visit more states than its split gate.

    The gate is ``(num_threads << search_chunk_exponent) >> connected_cluster_split_shift`` states.
    The second and last limit of this search is predicted to visit far fewer states than the gate of
    two threads at the default exponent and shift, and more than the gate at exponent 0, which is
    zero, or at the default exponent and the largest shift, which is zero too.
    """
    calls = record_thread_counts(monkeypatch)
    problem = PROBLEMS["toric-X"]
    tune(num_threads=2)
    for exponent, shift, splits in [(15, 2, False), (0, 2, True), (15, 62, True)]:
        calls.clear()
        with _tuning.use_tuning(
            search_chunk_exponent=exponent, connected_cluster_split_shift=shift
        ):
            assert cs.get_distance_connected_cluster([problem], cutoff=0) == 4
        assert (bool(calls) and set(calls) == {2}) if splits else not calls
    calls.clear()
    tune(num_threads=1, search_chunk_exponent=0)
    assert cs.get_distance_connected_cluster([problem], cutoff=0) == 4
    assert not calls


def test_an_error_in_a_thread_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """An error in the search of an item reaches the caller once every thread has returned."""
    search_supports = cs._search_supports
    signature = inspect.signature(search_supports)

    def fail_items(*args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.int64], ...]:
        arguments = signature.bind(*args, **kwargs).arguments
        if arguments["frontier_depth"] == 0 and len(arguments["roots"]):
            raise RuntimeError("failed item")
        return search_supports(*args, **kwargs)

    # With 100 roots and two threads, the roots are the items, whatever the limit.
    monkeypatch.setattr(cs, "_search_supports", fail_items)
    problem = (np.zeros((0, 100), dtype=np.uint8), np.zeros((0, 100), dtype=np.uint8))
    search = get_search(*problem, num_threads=2, split_minimum_states=-1)
    num_threads = threading.active_count()
    with pytest.raises(RuntimeError, match="failed item"):
        search.find_witness(1)
    assert threading.active_count() == num_threads


def get_ring_problem(num_ring: int) -> Problem:
    """A problem whose variable 0 is a witness alone, followed by a ring of variables.

    The checks of the ring are its edges, so its only nonzero word is all of it, and a search from
    one of its variables visits many arcs of the ring below that weight.
    """
    checks = np.zeros((num_ring, num_ring + 1), dtype=np.uint8)
    for index in range(num_ring):
        checks[index, 1 + index] = checks[index, 1 + (index + 1) % num_ring] = 1
    return checks, np.zeros((0, num_ring + 1), dtype=np.uint8)


@pytest.mark.parametrize("later_first", [True, False])
def test_limits_on_threads_keep_the_first_witness_in_dfs_order(
    later_first: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A witness of a later item never displaces that of an earlier item, whichever ends first.

    Every symbol of this problem is a witness, so each of its 100 roots is an item with a witness.
    On two threads, item 0 runs either after a later item has run, or while a later item that has
    started searching waits for it.
    """
    later_ran, later_started, first_ran = (threading.Event() for _ in range(3))
    run_work_items = _threads.run_work_items
    search_supports = cs._search_supports
    signature = inspect.signature(search_supports)

    def gated_run(
        num_items: int,
        num_threads: int,
        start_worker: Callable[[threading.Event], Callable[[int, int], None]],
        **kwargs: Any,
    ) -> None:
        def gated_start(stop: threading.Event) -> Callable[[int, int], None]:
            run_range = start_worker(stop)

            def gated_range(start: int, end: int) -> None:
                if start == 0:
                    assert (later_ran if later_first else later_started).wait(
                        timeout=THREAD_TIMEOUT
                    )
                run_range(start, end)
                (later_ran if start else first_ran).set()

            return gated_range

        run_work_items(num_items, num_threads, gated_start, **kwargs)

    def gated_search(*args: Any, **kwargs: Any) -> Any:
        roots = signature.bind(*args, **kwargs).arguments["roots"]
        if not later_first and len(roots) and roots[0] != 0:
            later_started.set()
            assert first_ran.wait(timeout=THREAD_TIMEOUT)
        return search_supports(*args, **kwargs)

    problem = (np.zeros((0, 100), dtype=np.uint8), np.zeros((0, 100), dtype=np.uint8))
    serial = get_search(*problem)
    expected = serial.find_witness(1)
    monkeypatch.setattr(_threads, "run_work_items", gated_run)
    monkeypatch.setattr(cs, "_search_supports", gated_search)
    search = get_search(*problem, num_threads=2, split_minimum_states=-1)
    witness = search.find_witness(1)
    assert later_ran.is_set() and first_ran.is_set()
    assert witness is not None and expected is not None
    assert witness.tolist() == expected.tolist() and search.num_states == serial.num_states


@pytest.mark.parametrize("ends_by", ["witness", "error"])
def test_a_range_on_threads_stops_within_a_chunk_once_an_earlier_item_ends_the_limit(
    ends_by: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A thread searches no further chunk of an item once an earlier item ends the limit.

    Variable 0 of this problem is a witness alone, and the search from a variable of its ring is
    long (see ``get_ring_problem``).  Under the rotations of the ring, the roots are 0 and 1, which
    are the items on two threads, and the limit runs in chunks of one state.  The search of item 0
    finds that witness, or raises an error, while the other thread searches the first chunk of item
    1.  After the witness, that thread searches no further chunk of item 1, though it had more to
    search; after the error, the stop event ends it.
    """
    later_started, first_ran = threading.Event(), threading.Event()
    stops: list[threading.Event] = []
    ranges: list[tuple[int, int]] = []
    later_calls: list[tuple[int, bool]] = []
    run_work_items = _threads.run_work_items
    search_supports = cs._search_supports

    def gated_run(
        num_items: int,
        num_threads: int,
        start_worker: Callable[[threading.Event], Callable[[int, int], None]],
        **kwargs: Any,
    ) -> None:
        def gated_start(stop: threading.Event) -> Callable[[int, int], None]:
            stops.append(stop)
            run_range = start_worker(stop)

            def gated_range(start: int, end: int) -> None:
                ranges.append((start, end))
                if start == 0:
                    assert later_started.wait(timeout=THREAD_TIMEOUT)
                try:
                    run_range(start, end)
                finally:
                    if start == 0:
                        first_ran.set()

            return gated_range

        run_work_items(num_items, num_threads, gated_start, **kwargs)

    def gated_search(*args: Any) -> Any:
        arguments = get_kernel_arguments(args)
        roots, frontier_depth = arguments["roots"], arguments["frontier_depth"]
        if frontier_depth == 0 and len(roots):
            if roots[0] == 0 and ends_by == "error":
                raise RuntimeError("failed item 0")
            if roots[0] != 0:
                later_started.set()
                assert (first_ran if ends_by == "witness" else stops[0]).wait(
                    timeout=THREAD_TIMEOUT
                )
                result = search_supports(*args)
                later_calls.append((int(roots[0]), bool(len(result[3]))))
                return result
        return search_supports(*args)

    monkeypatch.setattr(_threads, "run_work_items", gated_run)
    monkeypatch.setattr(cs, "_search_supports", gated_search)
    monkeypatch.setattr(_threads, "ITEMS_PER_THREAD", 0)
    monkeypatch.setattr(cs, "_MAX_ITEMS", 0)
    rotation = np.array([[0, *range(2, 31), 1]])
    search = get_search(
        *get_ring_problem(30),
        get_automorphisms=lambda: rotation,
        num_threads=2,
        split_minimum_states=-1,
        chunk_states=1,
    )
    if ends_by == "witness":
        witness = search.find_witness(8)
        assert witness is not None and witness.tolist() == [0]
    else:
        with pytest.raises(RuntimeError, match="failed item 0"):
            search.find_witness(8)

    # The other thread searched one chunk of item 1, which had more to search.
    assert search._roots.tolist() == [0, 1] and ranges and first_ran.is_set()
    assert later_calls == [(1, True)]


@pytest.mark.parametrize("num_threads", [1, 2])
def test_an_interrupt_during_a_long_limit_takes_effect_within_a_chunk(
    num_threads: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A keyboard interrupt on the calling thread ends a limit within a chunk on each thread.

    Python raises an interrupt that arrives during a compiled call once the call returns, as the
    wrapper of ``_search_supports`` here raises one after a call of the calling thread that stops
    with more to search.  On two threads, the other thread starts only once the calling thread has,
    and the interrupt waits until the other thread has searched a chunk.  Most roots of the toric
    code are items whose searches at this limit, which has no witness, span several chunks of one
    state.  So the search ends before the limit does, and the other thread starts at most one chunk
    once the interrupt has set the stop event of the threads.
    """
    problem = get_sectors(codes.ToricCode(6))[0]
    serial = get_search(*problem)
    assert serial.find_witness(5) is None
    stops: list[threading.Event] = []
    main_started, other_started = threading.Event(), threading.Event()
    run_work_items = _threads.run_work_items
    search_supports = cs._search_supports
    # The states that each call counted, and whether it started after the stop event was set.
    calls: list[tuple[int, bool]] = []

    def record_stops(
        num_items: int,
        num_threads: int,
        start_worker: Callable[[threading.Event], Callable[[int, int], None]],
        **kwargs: Any,
    ) -> None:
        def start(stop: threading.Event) -> Callable[[int, int], None]:
            stops.append(stop)
            return start_worker(stop)

        run_work_items(num_items, num_threads, start, **kwargs)

    def interrupt(*args: Any) -> Any:
        on_main = threading.current_thread() is threading.main_thread()
        searches_items = len(get_kernel_arguments(args)["roots"]) > 0
        if searches_items:
            if on_main:
                main_started.set()
            else:
                assert main_started.wait(timeout=THREAD_TIMEOUT)
        after = any(stop.is_set() for stop in stops)
        result = search_supports(*args)
        if searches_items:
            calls.append((int(np.sum(result[1])), after))
            if not on_main:
                other_started.set()
            elif len(result[3]) and not after:
                assert num_threads == 1 or other_started.wait(timeout=THREAD_TIMEOUT)
                raise KeyboardInterrupt
        return result

    monkeypatch.setattr(_threads, "ITEMS_PER_THREAD", 0)
    monkeypatch.setattr(cs, "_MAX_ITEMS", 0)
    monkeypatch.setattr(_threads, "run_work_items", record_stops)
    monkeypatch.setattr(cs, "_search_supports", interrupt)
    search = get_search(*problem, num_threads=num_threads, split_minimum_states=-1, chunk_states=1)
    with pytest.raises(KeyboardInterrupt):
        search.find_witness(5)
    assert max(states for states, _ in calls) <= 1
    assert sum(after for _, after in calls) <= num_threads - 1
    assert sum(states for states, _ in calls) < serial.num_states


####################################################################################################
# compiled search


def run_search_supports(
    search: cs.SupportSearch,
    roots: npt.ArrayLike,
    max_weight: int,
    start_syndrome: npt.NDArray[np.uint64] | None = None,
    *,
    resume: npt.NDArray[np.int64] = cs._NO_PATH,
    ends: npt.NDArray[np.int64] = cs._NO_PATH,
    max_states: int = cs.MAX_VALUE,
    fixed_words: tuple[int, ...] | None = None,
    memoize: bool = False,
    restrict_viable_symbols: bool = True,
) -> tuple[npt.NDArray[np.int64], ...]:
    """Run ``_search_supports`` with the tables of a search, from whole roots.

    The overlap cuts are those of the rows of ``excluded`` under the current ranks.  The starting
    syndrome is zero unless given, and ``fixed_words`` selects the version for its number of words
    unless given.
    """
    roots = np.asarray(roots, dtype=np.int64)
    if start_syndrome is None:
        start_syndrome = np.zeros(search._check_tables.symbol_masks.shape[1], dtype=np.uint64)
    if fixed_words is None:
        fixed_words = cs._get_fixed_words(len(start_syndrome))
    return cs._search_supports(
        *search._check_tables,
        roots,
        np.empty((len(roots), 0), dtype=np.int64),
        0,
        search._ranks,
        search._inverse_overlaps,
        max_weight,
        resume,
        ends,
        max_states,
        start_syndrome,
        fixed_words,
        memoize,
        restrict_viable_symbols,
        *cs._get_overlap_cuts(search._excluded, search._ranks, search._symplectic),
    )


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
    # so a change of the tie-break needs a new instance.  The search adds no sums of pairs of
    # checks, which would change the counts.
    checks = np.array([[1, 0, 1, 0], [0, 1, 0, 0], [1, 1, 0, 1], [0, 0, 1, 0]], dtype=np.uint8)
    search = get_search(checks, np.zeros((0, 4), dtype=np.uint8), pair_sum_checks=False)
    packed = pack_bits(np.array([[1, 1, 0, 1]], dtype=np.uint8))[0]
    num_states = {}
    for restrict_viable_symbols in (True, False):
        support, item_states, *_ = run_search_supports(
            search, [-1], 2, packed, memoize=True, restrict_viable_symbols=restrict_viable_symbols
        )
        assert not len(support)
        num_states[restrict_viable_symbols] = int(np.sum(item_states))
    assert num_states == {True: 1, False: 0}


@pytest.mark.parametrize("symplectic", [False, True])
def test_the_versions_for_fixed_and_any_numbers_of_words_search_alike(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config, symplectic: bool
) -> None:
    """The version of the search for two-word syndromes and the general version search alike.

    They find the same witnesses and vectors with a syndrome, and visit the same states.  The
    problems have 70 checks, and the searches add no sums of pairs of checks, so the syndromes have
    two words.  With ``_MAX_FIXED_WORDS`` set to one, these syndromes are wider than it, so they
    take the general version.  The limits at and below the distance run on one thread and on three.
    On three threads, they also run the frontier passes and the items of each version.  The queries
    run on one thread.  On one thread, every call of the compiled search returns the same with
    either version.
    """
    # A small number of words gets a version of its own, and a large one the general version.
    assert cs._get_fixed_words(1) == (0,)
    assert cs._get_fixed_words(4 * cs._MAX_FIXED_WORDS) == ()

    # Variables 0 and 1 are in every check of a block, and every other variable in one, so the
    # lightest nonzero words act on variables 0 and 1 alone, and weigh 2.  The binary problem has a
    # block of 70 checks, and the symplectic problem a block of 35 X-type checks and one of 35
    # Z-type checks.
    def get_block(num_checks: int) -> npt.NDArray[np.uint8]:
        """Checks that each hold variables 0 and 1, and one variable of their own."""
        ones = np.ones((num_checks, 2), dtype=np.uint8)
        return np.hstack([ones, np.eye(num_checks, dtype=np.uint8)])

    distance = 2
    if symplectic:
        block = get_block(35)
        checks = np.block([[block, np.zeros_like(block)], [np.zeros_like(block), block]])
    else:
        checks = get_block(70)
    problem = (checks, np.zeros((0, checks.shape[1]), dtype=np.uint8))
    width = checks.shape[1]
    num_variables = width // 2 if symplectic else width
    products = np.roll(checks, num_variables, axis=1) if symplectic else checks
    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    syndromes = [
        products @ get_vector(rng.choice(width, 2, replace=False), width, False) % 2
        for _ in range(3)
    ]

    # Record what each call of the compiled search returns, and the length of its fixed_words, which
    # is zero for the general version.
    calls: list[tuple[int, list[int], list[int]]] = []
    search_supports = cs._search_supports

    def record(*args: Any) -> tuple[npt.NDArray[np.int64], ...]:
        result = search_supports(*args)
        fixed_words = get_kernel_arguments(args)["fixed_words"]
        calls.append((len(fixed_words), result[0].tolist(), result[1].tolist()))
        return result

    monkeypatch.setattr(cs, "_search_supports", record)
    results = []
    for max_fixed_words, expected_length in [(cs._MAX_FIXED_WORDS, 2), (1, 0)]:
        monkeypatch.setattr(cs, "_MAX_FIXED_WORDS", max_fixed_words)
        for num_threads in (1, 3):
            calls.clear()
            search = get_search(
                *problem,
                symplectic=symplectic,
                pair_sum_checks=False,
                num_threads=num_threads,
                split_minimum_states=-1,
            )
            no_witness, witness = map(search.find_witness, (distance - 1, distance))
            assert no_witness is None and witness is not None and len(witness) == distance
            vectors = []
            for syndrome in syndromes:
                symbols = search.find_vector_with_syndrome(syndrome, 2)
                assert symbols is not None and len(symbols) <= 2
                vector = get_vector(symbols, width, symplectic)
                assert np.array_equal(products @ vector % 2, syndrome)
                vectors.append(symbols.tolist())
            assert {length for length, _, _ in calls} == {expected_length}
            # Threads may or may not run the items after a witness, in any order.
            returned = [call[1:] for call in calls] if num_threads == 1 else None
            results.append((returned, witness.tolist(), search.num_states, vectors))
    assert results[0] == results[2] and results[1] == results[3]
    assert results[0][1:] == results[1][1:]


def test_a_wrong_number_of_fixed_words_is_rejected() -> None:
    """A version for a number of words of a syndrome is never run on syndromes of other widths.

    Steane's checks have syndromes of one word, and the X sector of the distance-13 surface code
    syndromes of two, which are given too many and too few fixed words.
    """
    surface = codes.SurfaceCode(13)
    surface_x = (np.asarray(surface.matrix_z, np.uint8), np.asarray(surface.matrix_x, np.uint8))
    for problem, fixed_words in [(PROBLEMS["steane-X"], (0, 0)), (surface_x, (0,))]:
        search = get_search(*problem)
        assert len(fixed_words) != search._check_tables.symbol_masks.shape[1]
        with pytest.raises(ValueError, match="fixed_words"):
            run_search_supports(search, search._roots, 3, fixed_words=fixed_words)


@pytest.mark.parametrize(("max_weight", "root"), [(5, 0), (6, 1)])
def test_a_split_call_finds_the_witness_and_counts_the_states_of_one(
    max_weight: int, root: int
) -> None:
    """Two calls of a split, in DFS order, find the witness and count the states of the call.

    Each call searches one item, as a split requires.  A search from a root of the X sector of the
    toric code of distance 6 runs in chunks of one state, at a limit below the distance, or at the
    distance, from a root whose splits give the witness to either call.  At each stop, for each
    state of the path that it returns, and each branch above its entry in the path and below its
    end, the call that goes on is split there, as ``_search_supports`` describes.
    """
    search = get_search(*get_sectors(codes.ToricCode(6))[0])
    roots = search._roots[root : root + 1]

    def call(resume: npt.NDArray[np.int64], ends: npt.NDArray[np.int64], max_states: int) -> Any:
        """The witness and count of a search from the root, and the path and ends it returns."""
        support, item_states, _, path, path_ends = run_search_supports(
            search, roots, max_weight, resume=resume, ends=ends, max_states=max_states
        )
        return support.tolist(), int(item_states[0]), path, path_ends

    # The number of splits in which the first call has the witness, in which only the second does,
    # and in which neither does.
    witnesses_found = [0, 0, 0]
    _, _, resume, ends = call(cs._NO_PATH, cs._NO_PATH, 1)
    while len(resume):
        witness, states, _, _ = call(resume, ends, cs.MAX_VALUE)
        for depth in range(len(resume)):
            for branch in range(resume[depth] + 1, ends[depth]):
                first_ends = np.concatenate([resume[:depth] + 1, [branch], ends[depth + 1 :]])
                first_witness, first_states, _, _ = call(resume, first_ends, cs.MAX_VALUE)
                second_witness, second_states, _, _ = call(
                    np.append(resume[:depth], branch), ends[: depth + 1], cs.MAX_VALUE
                )
                witnesses_found[0 if first_witness else 1 if second_witness else 2] += 1
                if first_witness:
                    assert (first_witness, first_states) == (witness, states)
                else:
                    assert (second_witness, first_states + second_states) == (witness, states)
        _, _, resume, ends = call(resume, ends, 1)
    assert witnesses_found[2] if max_weight < 6 else witnesses_found[0] and witnesses_found[1]


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
    search = get_search(checks, np.zeros((0, checks.shape[1])), pair_sum_checks=False)

    def search_from(roots: list[int], syndrome: tuple[int, ...]) -> list[int]:
        packed = pack_bits(np.array([syndrome], dtype=np.uint8))[0]
        support, *_ = run_search_supports(search, roots, 2, packed, memoize=True)
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
    search = get_search(
        checks, np.zeros((0, np.shape(checks)[1])), symplectic=symplectic, pair_sum_checks=False
    )
    for bits in itertools.product([0, 1], repeat=len(search._checks)):
        if not any(bits):
            continue
        packed = pack_bits(np.array([bits], dtype=np.uint8))[0]
        for roots, max_weight in itertools.product([[-1], [0, -1]], range(1, 4)):
            results = [
                run_search_supports(search, roots, max_weight, packed, memoize=memoize)
                for memoize in (False, True)
            ]
            # Stale memos, such as those of a search from root 0, would likely change the count.
            assert results[0][0].tolist() == results[1][0].tolist()
            assert np.array_equal(results[0][1], results[1][1])
