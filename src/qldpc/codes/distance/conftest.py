# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for exact-distance, binary-algebra, BZ, and count tests."""

import itertools
from collections.abc import Iterable, Sequence
from typing import Any, NamedTuple

import galois
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import _compilation, codes, math
from qldpc._binary import get_independent_rows, get_kernel, row_reduce_binary
from qldpc.objects import Pauli, PauliXZ

from ..conftest import Tune
from . import logical_counts as lc
from ._binary import get_nested_code_basis
from ._coset_leader_search import CosetLeaderSearch
from ._information_sets import basis_families, certificates


@pytest.fixture
def force_information_sets(tune: Tune) -> None:
    """Route Brouwer-Zimmermann searches through information sets instead of exhaustive search."""
    tune(allow_enumeration_by_size=False, allow_enumeration_by_work=False)


@pytest.fixture
def plain_search_setup() -> None:
    """Run the set-up code of searches as plain Python, even with ``--compile-search-setup``.

    A test needs this if it relies on a compiled-on-request function calling a function that the
    test replaces, such as to record calls to it, since compiled callers would not call the
    replacement.  A test that compares the plain Python set-up code with its compiled version needs
    it too.  This fixture sets every compiled-on-request function back to plain Python, which also
    undoes an earlier replacement of one.  So a fixture that replaces one must request this fixture,
    which then runs first.
    """
    _compilation.set_on_request_functions_compiled(False)


class CostlyStages(NamedTuple):
    """The results of the costly stages of searches, which the costly_stages fixture records."""

    partitioned_families: list[basis_families.BasisFamily]
    search_certificates: list[certificates.Certificate]


@pytest.fixture
def costly_stages(monkeypatch: pytest.MonkeyPatch) -> CostlyStages:
    """Record the families of matroid partitioning and the certificates that searches choose.

    A family with sets and a maximum coverage above one shows that a search took a partitioned
    family, and a certificate with pivot blocks shows that a search used linear programming.
    """
    stages = CostlyStages([], [])
    get_partitioned_family = basis_families.get_partitioned_family
    get_search_certificate = basis_families.BasisFamily.get_search_certificate

    def record_family(*args: Any, **kwargs: Any) -> basis_families.BasisFamily:
        family = get_partitioned_family(*args, **kwargs)
        stages.partitioned_families.append(family)
        return family

    def record_certificate(
        basis_family: basis_families.BasisFamily, *args: Any, **kwargs: Any
    ) -> certificates.Certificate:
        certificate = get_search_certificate(basis_family, *args, **kwargs)
        stages.search_certificates.append(certificate)
        return certificate

    monkeypatch.setattr(basis_families, "get_partitioned_family", record_family)
    monkeypatch.setattr(basis_families.BasisFamily, "get_search_certificate", record_certificate)
    return stages


def get_binary_matrix(rows: Iterable[str]) -> npt.NDArray[np.uint8]:
    """The binary matrix with the given rows of digits."""
    return np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)


# Codes whose greedy disjoint packings, in natural column order, leave room for the other families
# of the routes of coset searches (see ``_brouwer_zimmermann/coset_routes.py``).  The even code has
# the columns e1, e2, e3, e1 + e2, e1 + e2, and e1 + e2 + e3, so every row has even weight.  Its
# greedy packing has ranks three and two, and its columns hold two disjoint information sets.  In
# the random [10, 6] code, a greedy packing of ranks six and three leaves room for one of ranks six
# and four, and for a family of three overlapping information sets, and queries take each of these
# at some maximum weights.  In the random [14, 8] code, a greedy packing of ranks eight and six uses
# every column, and three overlapping information sets do less work for some queries, but not for
# others.  So its overlapping family depends on the weight at which it is built, and that family
# changes some answers.
EVEN_ROUTES_CODE = get_binary_matrix(["100111", "010111", "001001"])
TWO_FAMILIES_CODE = get_binary_matrix(
    ["1001010111", "0110001011", "0101111010", "0010111100", "0010100000", "0000100110"]
)
OVERLAPPING_ROUTES_CODE = get_binary_matrix(
    [
        "01111100111010",
        "00010010000010",
        "00010001001101",
        "11000101110011",
        "10111001110101",
        "01000001111111",
        "00101110110101",
        "11001001110111",
    ]
)


def get_random_full_rank_matrix(
    rng: np.random.Generator, rows: int, cols: int
) -> npt.NDArray[np.uint8]:
    """Return a random full-row-rank binary matrix with the requested row and column counts."""
    while True:
        matrix = rng.integers(0, 2, size=(rows, cols), dtype=np.uint8)
        if len(get_independent_rows(matrix)) == rows:
            return matrix


def count_set_bits(val: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    """Return per-word bit counts, computed bit by bit as an independent reference for popcounts."""
    val = np.asarray(val)
    nbits = 8 * val.itemsize
    out = np.empty_like(val)
    for indices in np.ndindex(val.shape):
        ival = int(val[indices])
        out[indices] = sum(ival >> i & 1 for i in range(nbits))

    return out


####################################################################################################
# helpers for the tests of logical-operator counts

Counts = lc.CSSLogicalCounts
METHODS: tuple[lc.LogicalCountMethod, ...] = ("brouwer_zimmermann", "brute_force")


def get_counter(checks_x: npt.ArrayLike, checks_z: npt.ArrayLike) -> lc.CSSLogicalCounter:
    """A counter for the CSS code with the given binary checks."""
    code = codes.CSSCode(np.asarray(checks_x, dtype=int), np.asarray(checks_z, dtype=int))
    return lc.CSSLogicalCounter(code)


def get_bits(values: list[int], width: int) -> npt.NDArray[np.uint8]:
    """Binary rows whose bit j is bit j of each integer."""
    return np.array([[value >> jj & 1 for jj in range(width)] for value in values], dtype=np.uint8)


def append_frozen_qubits(
    checks_x: npt.ArrayLike, checks_z: npt.ArrayLike, width: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Append qubits fixed in |0> by a Z check each; no logical of minimum weight touches them."""
    x, z = np.asarray(checks_x, dtype=np.uint8), np.asarray(checks_z, dtype=np.uint8)
    extra = width - x.shape[1]
    padded_x = np.pad(x, ((0, 0), (0, extra)))
    padded_z = np.vstack(
        [np.pad(z, ((0, 0), (0, extra))), np.hstack([np.zeros((extra, z.shape[1])), np.eye(extra)])]
    ).astype(np.uint8)
    return padded_x, padded_z


def get_random_css_code(
    rng: np.random.Generator, num_qubits: int, rank_x: int, rank_z: int
) -> codes.CSSCode:
    """A random binary CSS stabilizer code: Z checks are random combinations of ker(H_X)."""
    checks_x = rng.integers(0, 2, (rank_x, num_qubits), dtype=np.uint8)
    kernel = get_kernel(checks_x)
    checks_z = rng.integers(0, 2, (rank_z, len(kernel))) @ kernel % 2
    return codes.CSSCode(checks_x.astype(int), checks_z.astype(int))


def get_sector_problem(
    code: codes.CSSCode, pauli: PauliXZ
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """The adapted basis and quotient labels of one sector of a binary CSS code."""
    return get_nested_code_basis(
        np.asarray(code.get_logical_ops(pauli), dtype=np.uint8),
        np.asarray(code.get_stabilizer_ops(pauli, canonicalized=True), dtype=np.uint8),
    )


def get_span(rows: npt.ArrayLike) -> set[int]:
    """All combinations of the given binary rows, as integers whose bit j is column j."""
    words = {0}
    for row in np.asarray(rows, dtype=int):
        value = sum(int(bit) << jj for jj, bit in enumerate(row))
        words |= {word ^ value for word in words}
    return words


def enumerate_logical_ops(code: codes.CSSCode) -> tuple[list[int], list[int], int, int]:
    """Count logical operators by enumerating the normalizers, using CSSCode's stabilizers.

    Returns the X-only and Z-only counts at every weight, the distance, and the number of nontrivial
    logical operators of any type at the distance.
    """
    num_qubits = len(code)
    normalizers = []
    for stabilizers in (code.get_stabilizer_ops(Pauli.Z), code.get_stabilizer_ops(Pauli.X)):
        field = galois.GF(2)
        kernel = field(stabilizers).null_space() if len(stabilizers) else field.Identity(num_qubits)
        normalizers.append(get_span(kernel))
    normal_x, normal_z = normalizers
    gauge_x, gauge_z = get_span(code.matrix_x), get_span(code.matrix_z)
    counts_x, counts_z = [0] * (num_qubits + 1), [0] * (num_qubits + 1)
    for word in normal_x - gauge_x:
        counts_x[word.bit_count()] += 1
    for word in normal_z - gauge_z:
        counts_z[word.bit_count()] += 1
    distance = next(
        weight for weight in range(1, num_qubits + 1) if counts_x[weight] or counts_z[weight]
    )
    full = sum(
        (word_x | word_z).bit_count() == distance
        for word_x, word_z in itertools.product(normal_x, normal_z)
        if word_x not in gauge_x or word_z not in gauge_z
    )
    return counts_x, counts_z, distance, full


def get_expected_counts(code: codes.CSSCode) -> lc.CSSLogicalCounts:
    """Counts at the distance, including full Pauli operators, from an independent enumeration."""
    counts_x, counts_z, distance, full = enumerate_logical_ops(code)
    return Counts(distance, counts_x[distance], counts_z[distance], full)


def assert_counts_match_enumeration(counter: lc.CSSLogicalCounter, **options: Any) -> None:
    """Check the counts at every weight, and the Pauli counts at the distance, by enumeration."""
    counts_x, counts_z, distance, full = enumerate_logical_ops(counter.code)
    for weight in range(counter.num_qubits + 1):
        result = counter.count_logical_ops(weight, **options)
        assert (result.count_x, result.count_z) == (counts_x[weight], counts_z[weight])
    expected = Counts(distance, counts_x[distance], counts_z[distance], full)
    assert counter.count_logical_ops(count_pauli=True, **options) == expected


# X checks {0,1,2,3,4} and {2,4}, and Z checks {1,2,3,4} and {0,1}, on five qubits.  By hand, the
# distance is two, the X-only logicals of weight two are {2,3} and {3,4}, and the Z-only ones are
# {2,4}, {0,3} and {1,3}.  The Z word {2,4} also supports the X stabilizer {2,4}, so in total there
# are 2 + 3 + 1 = 6 logical Pauli operators of weight two.
HAND_X, HAND_Z = get_bits([31, 20], 5), get_bits([30, 3], 5)


def _stabilizer_matrix(generators: Sequence[stim.PauliString]) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] with the given Pauli strings as rows."""
    return np.array([np.concatenate(gg.to_numpy()) for gg in generators], dtype=np.uint8)


def _graph_state(
    num_qubits: int, edges: Sequence[tuple[int, int]], phase: bool = False
) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] of a graph state, with S gates on every qubit if ``phase``.

    Generator j acts as X (or Y) on qubit j, so the X parts of the generators form the identity.
    """
    circuit = stim.Circuit()
    circuit.append("H", range(num_qubits))
    for edge in edges:
        circuit.append("CZ", edge)
    if phase:
        circuit.append("S", range(num_qubits))
    return _stabilizer_matrix(stim.Tableau.from_circuit(circuit).to_stabilizers())


def _pauli_weight(pauli: int, num_qubits: int, css_weight: bool) -> int:
    """Weight of a packed Pauli, with X in the low bits and Z in the next block."""
    x, z = pauli & ((1 << num_qubits) - 1), pauli >> num_qubits
    return (x | z).bit_count() if not css_weight else max(x.bit_count(), z.bit_count())


def _get_syndrome(matrix: npt.NDArray[np.uint8], pauli: int) -> int:
    """Syndrome of a packed Pauli: its symplectic product with each row of the stabilizer matrix."""
    bits = np.array([(pauli >> jj) & 1 for jj in range(matrix.shape[1])])
    products = matrix @ math.symplectic_conjugate(bits) % 2
    return sum(int(bit) << jj for jj, bit in enumerate(products))


def _pack_syndromes(syndromes: Sequence[int], num_qubits: int) -> npt.NDArray[np.uint64]:
    """Split packed syndromes into rows of 64-bit words, with word ``j`` in bits ``64 j`` and up."""
    width = (num_qubits + 63) // 64
    words = [[(syndrome >> (64 * jj)) % 2**64 for jj in range(width)] for syndrome in syndromes]
    return np.array(words, dtype=np.uint64).reshape(len(syndromes), width)


# Graph states whose searches of Pauli errors, by physical and by CSS weight, have a maximum
# disjoint family with more information sets, or a larger partial set, than their greedy packing,
# which their costly queries take (see ``coset_routes``).  The family of the first state for the
# physical weight has more information sets, and those of the other two states have a larger partial
# set.  The two states for the physical weight differ in the answers that this family changes.  On
# the second, it returns other errors for some classes without a light enough error.
_ROUTED_PHYSICAL_STATES = (
    _graph_state(5, [(0, 3), (1, 2), (1, 3), (1, 4)], phase=True),
    _graph_state(5, [(0, 4), (2, 3)]),
)

_ROUTED_CSS_WEIGHT_STATE = _graph_state(5, [(0, 1), (0, 4), (1, 4)])


def _random_css_state(
    num_qubits: int, num_x_stabilizers: int, rng: np.random.Generator
) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] of a random CSS state, with generators that mix pure-X and pure-Z.

    The X parts of the pure-X stabilizers are random independent rows, and the Z parts of the pure-Z
    stabilizers span their null space.  The generators are random invertible combinations of these
    stabilizers.
    """
    while True:
        x_rows = rng.integers(2, size=(num_x_stabilizers, num_qubits), dtype=np.uint8)
        if len(row_reduce_binary(x_rows)[2]) == num_x_stabilizers:
            break
    pure = np.zeros((num_qubits, 2 * num_qubits), dtype=np.uint8)
    pure[:num_x_stabilizers, :num_qubits] = x_rows
    pure[num_x_stabilizers:, num_qubits:] = get_kernel(x_rows)
    return _scramble(pure, rng)


def _scramble(matrix: npt.NDArray[np.uint8], rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """Random invertible combinations of the rows of a stabilizer matrix, which generate its group.

    The combinations are a random permutation of an upper triangular matrix with ones on its
    diagonal and random bits above it.
    """
    size = len(matrix)
    upper = np.triu(rng.integers(2, size=(size, size), dtype=np.uint8), 1) + np.eye(
        size, dtype=np.uint8
    )
    return (upper[rng.permutation(size)] @ matrix) % 2


def _css_minima(matrix: npt.NDArray[np.uint8], css_weight: bool = True) -> npt.NDArray[np.int_]:
    """The smallest CSS (or physical) weight of an error with each packed syndrome.

    The minima come from all ``4**n`` Paulis.
    """
    n = len(matrix)
    errors = (np.arange(1 << (2 * n))[:, None] >> np.arange(2 * n)) & 1
    checks = math.symplectic_conjugate(matrix.astype(int))
    syndromes = (errors @ checks.T % 2) @ (1 << np.arange(n))
    x, z = errors[:, :n], errors[:, n:]
    weights = np.maximum(x.sum(axis=1), z.sum(axis=1)) if css_weight else (x | z).sum(axis=1)
    minima = np.full(1 << n, 2 * n + 1)
    np.minimum.at(minima, syndromes, weights)
    return minima


# A code whose disjoint information sets have ranks five and four.
_RANK_DEFICIENT_CODE = np.array(
    [
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    ],
    dtype=np.uint8,
)


def _product_state(generators: Sequence[str]) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] of the given Pauli strings."""
    return _stabilizer_matrix([stim.PauliString(pauli) for pauli in generators])


def _assert_exact_answers(
    oracle: CosetLeaderSearch, matrix: npt.NDArray[np.uint8], css_weight: bool
) -> None:
    """Check every query of an oracle against exhaustive minima of the state of ``matrix``."""
    num_qubits = len(matrix)
    minima = _css_minima(matrix, css_weight)
    for syndrome, max_weight in itertools.product(range(1 << num_qubits), range(num_qubits + 1)):
        bound = oracle.get_min_weight_bound(syndrome, max_weight)
        weight = _pauli_weight(bound.representative, num_qubits, css_weight)
        assert bound.within_max_weight == (minima[syndrome] <= max_weight)
        assert _get_syndrome(matrix, bound.representative) == syndrome
        assert bound.lower_bound <= minima[syndrome] <= weight
        assert weight <= max_weight if bound.within_max_weight else bound.lower_bound > max_weight
