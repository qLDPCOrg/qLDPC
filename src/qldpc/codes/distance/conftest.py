# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures and helpers for tests of exact distances, counts, and coset-leader searches."""

import contextlib
import dataclasses
import itertools
import threading
import warnings
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, NamedTuple, TypeVar

import numba
import numpy as np
import numpy.typing as npt
import pytest
import stim

from qldpc import _compilation, codes, math
from qldpc._binary import get_independent_rows, get_kernel, row_reduce_binary
from qldpc.objects import PauliXZ

from . import _threads, _tuning
from . import logical_counts as lc
from ._binary import get_nested_code_basis
from ._brouwer_zimmermann import coset_tables
from ._information_sets import basis_families, certificates
from .coset_leaders import CosetLeaderSearch

# How long, in seconds, a test waits for another thread, so that a deadlock fails the test rather
# than hangs it.
THREAD_TIMEOUT = 10

# The result of a function that ``run_on_a_thread`` runs.
_Result = TypeVar("_Result")


@pytest.fixture
def plain_search_setup() -> None:
    """Run the set-up code of searches as plain Python, even with ``--compile-search-setup``.

    A test needs this if it replaces a function that compiled-on-request code calls, since compiled
    callers would not call the replacement, or if it compares the plain set-up code with its
    compiled version.  This fixture sets every compiled-on-request function back to plain Python,
    which undoes an earlier replacement of one, so a fixture that replaces one must request this
    fixture, which then runs first.
    """
    _compilation.set_on_request_functions_compiled(False)


class CostlyStages(NamedTuple):
    """The results of the costly stages of searches, which the costly_stages fixture records."""

    partitioned_families: list[basis_families.BasisFamily]
    search_certificates: list[certificates.Certificate]


@pytest.fixture
def costly_stages(monkeypatch: pytest.MonkeyPatch) -> CostlyStages:
    """Record the families of matroid partitioning and the certificates that searches choose.

    A family with sets shows that a search took a partitioned family, and a certificate with pivot
    blocks shows that a search used linear programming.
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
# that coset searches can take (Section 4.4 of ``brouwer_zimmermann_code.md``):
#
# - EVEN_ROUTES_CODE has the columns e1, e2, e3, e1 + e2, e1 + e2, and e1 + e2 + e3, so its rows
#   are even.  Its greedy packing has ranks three and two, but its columns hold two disjoint
#   information sets.
# - TWO_FAMILIES_CODE, a random [10, 6] code, has a greedy packing of ranks six and three, a
#   disjoint family of ranks six and four, and a family of three overlapping information sets, and
#   queries take each family at some maximum weights.
# - OVERLAPPING_ROUTES_CODE, a random [14, 8] code, has a greedy packing of ranks eight and six that
#   uses every column.  Three overlapping information sets do less work than the greedy packing for
#   some queries only, so its overlapping family depends on the weight at which it is built, and
#   changes some answers.
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

# A [15, 6, 4] code whose last nine columns have even weight on the first four rows.  A greedy
# packing of its columns in natural order takes the first six, and leaves a partial set of rank
# five, but the columns hold two disjoint information sets, which matroid partitioning with coverage
# one finds (Section 4.4 of ``brouwer_zimmermann_code.md``).
MISSED_SET_CODE = get_binary_matrix(
    [
        "100000111011011",
        "010000110000010",
        "001000101001100",
        "000100100010101",
        "000010010110101",
        "000001111110101",
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


def assert_certificate_bounds_unvisited_words(
    basis: npt.NDArray[np.uint8], family: basis_families.BasisFamily
) -> None:
    """Check that each step's certificate bounds every word that no search basis has visited.

    A step is a pair of a layer ``t`` and the number of search bases that have finished it, while
    the others have finished layer ``t - 1``.  A search basis visits a word after layer ``t`` if the
    word combines at most ``t`` of its rows.  The check covers the family's certificate, with and
    without the bounds of linear programming.  Also check the weight divisor, and that
    ``certifies`` agrees with each step's bound.  The family's labels must make every nonzero word
    eligible.
    """

    def get_coefficient_weights(generators: npt.NDArray[np.uint8]) -> dict[bytes, int]:
        return {
            (np.asarray(bits) @ generators % 2).astype(np.uint8).tobytes(): sum(bits)
            for bits in itertools.product((0, 1), repeat=len(generators))
        }

    # Map each word of the code to the number of rows of each search basis that it combines.
    words = get_coefficient_weights(basis)
    coefficient_weights = [get_coefficient_weights(generators) for generators in family.generators]
    assert all(set_weights.keys() == words.keys() for set_weights in coefficient_weights)

    # Check the family's certificate both without and with the bounds of linear programming.
    certificate = family.certificate
    pivot_blocks = tuple(
        tuple(int(column) for column in pivots[:rank])
        for pivots, rank in zip(family.pivots, family.ranks, strict=True)
    )
    denominator = _tuning.DEFAULT_TUNING.lp_dual_max_denominator
    checked = [
        certificate,
        dataclasses.replace(certificate, pivot_blocks=pivot_blocks, lp_max_denominator=denominator),
    ]

    # Each word's weight is a multiple of the divisor, and, at each step at which no search basis
    # has visited the word, at least the bound of each certificate.
    hamming_weights = {
        word: np.count_nonzero(np.frombuffer(word, dtype=np.uint8)) for word in words
    }
    steps = [
        (layer, num_completed)
        for layer in range(1, len(basis))
        for num_completed in range(len(family.ranks) + 1)
    ]
    for word, weight in hamming_weights.items():
        assert weight % certificate.weight_divisor == 0
        for layer, num_completed in steps:
            completed = [
                layer if index < num_completed else layer - 1 for index in range(len(family.ranks))
            ]
            if all(item[word] > done for item, done in zip(coefficient_weights, completed)):
                for each in checked:
                    assert weight >= each.step_lower_bound(layer, num_completed)

    # At each step, ``certifies`` agrees with the bound.
    for each, (layer, num_completed) in itertools.product(checked, steps):
        bound = each.step_lower_bound(layer, num_completed)
        for upper_bound in range(1, basis.shape[1] + 2):
            assert each.certifies(layer, num_completed, upper_bound) == (bound >= upper_bound)


####################################################################################################
# helpers for the tests of logical-operator counts

Counts = lc.CSSLogicalCounts
METHODS: tuple[lc.LogicalCountMethod, ...] = ("brouwer_zimmermann", "brute_force")


def get_random_css_code(
    rng: np.random.Generator, num_qubits: int, rank_x: int, rank_z: int
) -> codes.CSSCode:
    """A random binary CSS stabilizer code: Z checks are random combinations of ker(H_X)."""
    checks_x = rng.integers(0, 2, (rank_x, num_qubits), dtype=np.uint8)
    kernel = get_kernel(checks_x)
    checks_z = rng.integers(0, 2, (rank_z, len(kernel))) @ kernel % 2
    return codes.CSSCode(checks_x.astype(int), checks_z.astype(int))


def get_costly_basis() -> npt.NDArray[np.uint8]:
    """A basis of a [21, 8, 6] code whose search reaches matroid partitioning at low thresholds."""
    return get_binary_matrix(
        [
            "100000010110010010011",
            "010000010100101111100",
            "001000000111011111100",
            "000100010101000101010",
            "000010010110000011000",
            "000001010011000111011",
            "000000110001110101011",
            "000000001110001010110",
        ]
    )


def get_golay_checks() -> npt.NDArray[np.uint8]:
    """A basis of the dual of the cyclic [23, 12, 7] Golay code, which contains its dual.

    As checks and as excluded rows, it gives a sector of the [[23, 1, 7]] CSS code of the Golay
    code, which cyclic shifts preserve.
    """
    generator = np.zeros(23, dtype=np.uint8)
    generator[[0, 1, 5, 6, 7, 9, 11]] = 1
    return get_kernel(np.array([np.roll(generator, shift) for shift in range(12)]))


def get_sector(
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
    """Count logical operators by enumerating the stabilizers and normalizers by brute force.

    Returns the X-only and Z-only counts at every weight, the distance, and the number of nontrivial
    logical operators of any type at the distance.
    """

    def commute(word: int, others: set[int]) -> bool:
        return all(not (word & other).bit_count() & 1 for other in others)

    # The stabilizers are the gauge operators that commute with every gauge operator of the other
    # type, and a normalizer holds the words that commute with every stabilizer of the other type.
    gauge_x, gauge_z = get_span(code.matrix_x), get_span(code.matrix_z)
    stabilizers_x = {word for word in gauge_x if commute(word, gauge_z)}
    stabilizers_z = {word for word in gauge_z if commute(word, gauge_x)}
    num_qubits = len(code)
    normal_x = {word for word in range(1 << num_qubits) if commute(word, stabilizers_z)}
    normal_z = {word for word in range(1 << num_qubits) if commute(word, stabilizers_x)}
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


# X checks {0,1,2,3,4} and {2,4}, and Z checks {1,2,3,4} and {0,1}, on five qubits.  By hand, the
# distance is two, the X-only logicals of weight two are {2,3} and {3,4}, and the Z-only ones are
# {2,4}, {0,3} and {1,3}.  The Z word {2,4} also supports the X stabilizer {2,4}, so in total there
# are 2 + 3 + 1 = 6 logical Pauli operators of weight two.
HAND_X = get_binary_matrix(["11111", "00101"])
HAND_Z = get_binary_matrix(["01111", "11000"])

# The matrix by which the Clifford of ``CSSLogicalCounter.deformed_swel`` acts on X and on Z
# supports.
SWEL_MATRIX = [[1, 1, 1, 0], [1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 1]]


####################################################################################################
# helpers for the tests of coset-leader searches


def get_stabilizer_matrix(generators: Sequence[stim.PauliString | str]) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] with the given Pauli strings as rows."""
    return np.array(
        [np.concatenate(stim.PauliString(gg).to_numpy()) for gg in generators], dtype=np.uint8
    )


def get_graph_state(
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
    return get_stabilizer_matrix(stim.Tableau.from_circuit(circuit).to_stabilizers())


def get_pauli_weight(pauli: int, num_qubits: int, css_weight: bool) -> int:
    """Weight of a packed Pauli, with X in the low bits and Z in the next block."""
    x, z = pauli & ((1 << num_qubits) - 1), pauli >> num_qubits
    return (x | z).bit_count() if not css_weight else max(x.bit_count(), z.bit_count())


def get_syndrome(matrix: npt.NDArray[np.uint8], pauli: int) -> int:
    """Syndrome of a packed Pauli: its symplectic product with each row of the stabilizer matrix."""
    bits = np.array([(pauli >> jj) & 1 for jj in range(matrix.shape[1])])
    products = matrix @ math.symplectic_conjugate(bits) % 2
    return sum(int(bit) << jj for jj, bit in enumerate(products))


def split_syndromes_into_words(syndromes: Sequence[int], num_bits: int) -> npt.NDArray[np.uint64]:
    """Split packed syndromes of ``num_bits`` bits into rows of 64-bit words.

    Bit ``k`` of word ``j`` of a row is bit ``64 j + k`` of its syndrome.
    """
    width = (num_bits + 63) // 64
    words = [[(syndrome >> (64 * jj)) % 2**64 for jj in range(width)] for syndrome in syndromes]
    return np.array(words, dtype=np.uint64).reshape(len(syndromes), width)


# Graph states whose costly queries of Pauli errors take a maximum disjoint family that beats the
# greedy packing (Section 4.4 of ``brouwer_zimmermann_code.md``).  By physical weight, the family
# has more information sets on the first of ROUTED_PHYSICAL_STATES, and a larger partial set on the
# second, where it returns other errors than the greedy packing for some classes without an error
# within the maximum weight.  By CSS weight, it has a larger partial set on ROUTED_CSS_WEIGHT_STATE.
ROUTED_PHYSICAL_STATES = (
    get_graph_state(5, [(0, 3), (1, 2), (1, 3), (1, 4)], phase=True),
    get_graph_state(5, [(0, 4), (2, 3)]),
)

ROUTED_CSS_WEIGHT_STATE = get_graph_state(5, [(0, 1), (0, 4), (1, 4)])


def get_random_css_state(
    num_qubits: int, num_x_stabilizers: int, rng: np.random.Generator
) -> npt.NDArray[np.uint8]:
    """Stabilizer matrix [X|Z] of a random CSS state, with generators that mix pure-X and pure-Z.

    The X parts of the pure-X stabilizers are random independent rows, and the Z parts of the pure-Z
    stabilizers span their null space.
    """
    while True:
        x_rows = rng.integers(2, size=(num_x_stabilizers, num_qubits), dtype=np.uint8)
        if len(row_reduce_binary(x_rows)[2]) == num_x_stabilizers:
            break
    pure = np.zeros((num_qubits, 2 * num_qubits), dtype=np.uint8)
    pure[:num_x_stabilizers, :num_qubits] = x_rows
    pure[num_x_stabilizers:, num_qubits:] = get_kernel(x_rows)
    return scramble(pure, rng)


def scramble(matrix: npt.NDArray[np.uint8], rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """Random invertible combinations of the rows of a stabilizer matrix, which keep its group."""
    size = len(matrix)
    upper = np.triu(rng.integers(2, size=(size, size), dtype=np.uint8), 1) + np.eye(
        size, dtype=np.uint8
    )
    return (upper[rng.permutation(size)] @ matrix) % 2


def get_exhaustive_minima(matrix: npt.NDArray[np.uint8], css_weight: bool) -> npt.NDArray[np.int_]:
    """The smallest CSS (or physical) weight of a Pauli error with each packed syndrome."""
    n = len(matrix)
    errors = (np.arange(1 << (2 * n))[:, None] >> np.arange(2 * n)) & 1
    checks = math.symplectic_conjugate(matrix.astype(int))
    syndromes = (errors @ checks.T % 2) @ (1 << np.arange(n))
    x, z = errors[:, :n], errors[:, n:]
    weights = np.maximum(x.sum(axis=1), z.sum(axis=1)) if css_weight else (x | z).sum(axis=1)
    minima = np.full(1 << n, 2 * n + 1)
    np.minimum.at(minima, syndromes, weights)
    return minima


def get_binary_syndrome(checks: npt.NDArray[np.uint8], words: npt.NDArray[np.int_]) -> Any:
    """The packed syndrome of a binary word, or of each row of an array, for binary checks."""
    return (words @ checks.T % 2) @ (1 << np.arange(len(checks)))


def get_binary_minima(checks: npt.NDArray[np.uint8]) -> npt.NDArray[np.int_]:
    """The least Hamming weight of the words with each packed syndrome for binary checks."""
    num_columns = checks.shape[1]
    words = (np.arange(1 << num_columns)[:, None] >> np.arange(num_columns)) & 1
    minima = np.full(1 << len(checks), num_columns + 1)
    np.minimum.at(minima, get_binary_syndrome(checks, words), words.sum(axis=1))
    return minima


def get_set_ranks(tables: coset_tables.CosetSearchTables) -> list[int]:
    """The ranks of the sets of a family: their numbers of pivots that are not padding."""
    return np.count_nonzero(tables.pivots >= 0, axis=1).tolist()


# A code whose disjoint information sets have ranks five and four.
RANK_DEFICIENT_CODE = np.array(
    [
        [0, 0, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 1, 1],
        [1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    ],
    dtype=np.uint8,
)


def assert_exact_answers(
    search: CosetLeaderSearch, matrix: npt.NDArray[np.uint8], css_weight: bool
) -> None:
    """Check every query of a search against exhaustive minima of the state of ``matrix``."""
    num_qubits = len(matrix)
    minima = get_exhaustive_minima(matrix, css_weight)
    for syndrome, max_weight in itertools.product(range(1 << num_qubits), range(num_qubits + 1)):
        bound = search.get_min_weight_bound(syndrome, max_weight)
        weight = get_pauli_weight(bound.representative, num_qubits, css_weight)
        assert bound.within_max_weight == (minima[syndrome] <= max_weight)
        assert get_syndrome(matrix, bound.representative) == syndrome
        assert bound.lower_bound <= minima[syndrome] <= weight
        assert weight <= max_weight if bound.within_max_weight else bound.lower_bound > max_weight


####################################################################################################
# checks of the rules on threads


@pytest.fixture(autouse=True)
def check_code_on_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the workers of ``run_work_items`` fail if they break a rule on threads, in every test.

    Section 5 of each method's guide, ``brouwer_zimmermann_code.md`` or ``connected_cluster.md``,
    states the rules, and "Threads" in ``qldpc/circuits/certification/fault_propagation.md`` those
    of the certifier's fault lister, whose tests use this fixture too.  While a worker runs, on a
    new thread or on the thread that called ``run_work_items``, a call from Python of numba code
    compiled with ``parallel=True``, such as some of galois's, or of ``_tuning.get_tuning``, raises
    an ``AssertionError``, which ``run_work_items`` raises again.  The workers include the threads
    of ``iterate_in_order``, which start through ``run_work_items``.  Two threads that run
    numba-parallel code at once abort the process under numba's workqueue layer, and a new thread
    reads the default tuning.  The fixture sees only the calls that tests make on workers.  In
    compiled code, two tests of ``qldpc/_compiled_code_rules_test.py`` check the rest:
    ``test_nothing_asks_numba_for_parallel_code`` keeps the functions of qLDPC from being
    numba-parallel, and ``test_always_compiled_functions_use_nothing_from_outside_qldpc`` keeps out
    those of galois.
    """
    run_work_items = _threads.run_work_items

    def run_checked(
        num_items: int,
        num_threads: int,
        start_worker: Callable[[threading.Event], Callable[[int, int], None]],
        **options: Any,
    ) -> None:
        def start_checked(stop: threading.Event) -> Callable[[int, int], None]:
            with _running_worker():
                run_range = start_worker(stop)

            def run_range_checked(start: int, end: int) -> None:
                with _running_worker():
                    run_range(start, end)

            return run_range_checked

        with _checking_numba_calls():
            run_work_items(num_items, num_threads, start_checked, **options)

    monkeypatch.setattr(_threads, "run_work_items", run_checked)
    monkeypatch.setattr(_tuning, "_ACTIVE_TUNING", _CheckedTuning(_tuning._ACTIVE_TUNING))


def run_on_a_thread(function: Callable[[], _Result], timeout: float = THREAD_TIMEOUT) -> _Result:
    """Run a function on a thread of its own, and fail if it does not return within the timeout.

    So code on threads that deadlocks fails a test rather than hangs it.  The thread is a daemon, as
    are the threads that it starts, so that a stuck thread does not keep the test run from ending.
    An exception of the function is raised again.
    """
    results: list[_Result] = []
    errors: list[BaseException] = []

    def run() -> None:
        try:
            results.append(function())
        except BaseException as error:  # noqa: BLE001 - raised again on the calling thread
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "The function did not return within the timeout"
    if errors:
        raise errors[0]
    return results[0]


@contextlib.contextmanager
def shared_arrays_read_only(*roots: object) -> Iterator["list[_SharedArray]"]:
    """Make the arrays that some objects reach read-only, and check that they do not change.

    Compiled code that numba types with a read-only array fails to compile if it writes that array
    in any branch, also through views, loops, and the compiled functions that it calls.  Some writes
    compile anyway, such as those through ``.flat``, ``np.ascontiguousarray``,
    ``np.asfortranarray``, ``np.nditer``, or ``np.fill_diagonal``.  Plain Python fails only on the
    writes that it runs.

    An array taken from these arrays before this starts, such as a slice that a local name holds,
    stays writeable, so a test starts this before such arrays are taken.  A view taken within the
    block stays read-only after it, if code keeps it.  The flags are restored when the block exits,
    also on an error.  If the block raised, an array that cannot be made writeable again gives a
    warning, and the error of the block propagates.  Otherwise such an array fails the check; if
    none does, the arrays are then compared with their copies.  Yields the records of the arrays
    (see ``_get_shared_arrays``).
    """
    arrays = _get_shared_arrays(*roots)
    try:
        for each in arrays:
            each.array.flags.writeable = False
        yield arrays
    except BaseException:
        unrestored = _restore_flags(arrays)
        if unrestored:
            # Warn rather than raise, so that the error of the block is not hidden.  The warning
            # names the line of the with statement, past this generator and contextlib's __exit__.
            warnings.warn(f"Arrays left read-only: {unrestored}", RuntimeWarning, stacklevel=3)
        raise
    unrestored = _restore_flags(arrays)
    assert not unrestored, f"Arrays left read-only: {unrestored}"
    _assert_unchanged(arrays)


def check_shared_arrays_on_threads(
    monkeypatch: pytest.MonkeyPatch,
    get_roots: Callable[..., Iterable[object]],
    owner: object = _threads,
    name: str = "run_work_items",
) -> list[int]:
    """Run each call of a function with the arrays that some objects reach read-only and unchanged.

    By default, the function is ``run_work_items``, which runs the threads of a search.  To mark
    the slices that a method takes from the arrays of its search before it calls
    ``run_work_items``, wrap that method instead, as the test of BZ layers on threads does.
    ``get_roots``, which takes the arguments of the call, returns the objects whose arrays threads
    share, such as a search (see ``shared_arrays_read_only``).  Returns the list of the numbers of
    arrays that each call marked, so that a test can check that threads ran on marked arrays.
    """
    function = getattr(owner, name)
    checked = []

    def run_checked(*args: Any, **kwargs: Any) -> Any:
        with shared_arrays_read_only(*get_roots(*args, **kwargs)) as arrays:
            result = function(*args, **kwargs)
        checked.append(len(arrays))
        return result

    monkeypatch.setattr(owner, name, run_checked)
    return checked


# The number of workers of ``run_work_items`` that run on each thread, which is at most one unless a
# worker calls ``run_work_items`` itself.
_RUNNING_WORKERS = threading.local()

# The number of calls of ``run_work_items`` that run, while ``_checking_numba_calls`` checks calls
# of numba code, and a lock on it.
_CHECKED_RUNS = [0]
_CHECKED_RUNS_LOCK = threading.Lock()

# The ``__call__`` that numba's dispatchers define themselves, or None, while the check replaces it.
_OWN_CALL: list[Any] = [None]


@contextlib.contextmanager
def _running_worker() -> Iterator[None]:
    """Mark the current thread as running a worker of ``run_work_items``."""
    _RUNNING_WORKERS.count = _get_running_workers() + 1
    try:
        yield
    finally:
        _RUNNING_WORKERS.count -= 1


def _get_running_workers() -> int:
    """The number of workers of ``run_work_items`` that the current thread runs."""
    return getattr(_RUNNING_WORKERS, "count", 0)


@contextlib.contextmanager
def _checking_numba_calls() -> Iterator[None]:
    """Make numba dispatchers check, while any call of ``run_work_items`` runs, who calls them.

    Calls from Python of numba-parallel code raise on the threads that run workers.  The check is
    installed only while threads run, so that other calls of compiled code from Python cost nothing
    more.  Once no call runs, the dispatchers get back their own ``__call__``, if they define one.
    """
    dispatcher = numba.core.dispatcher.Dispatcher
    with _CHECKED_RUNS_LOCK:
        if not _CHECKED_RUNS[0]:
            own_call = vars(dispatcher).get("__call__")
            call = dispatcher.__call__

            def checked_call(self: Any, *args: Any, **kwargs: Any) -> object:
                if _get_running_workers() and self.targetoptions.get("parallel"):
                    raise AssertionError(
                        f"A worker of run_work_items called numba-parallel code: {self.py_func}"
                    )
                return call(self, *args, **kwargs)

            _OWN_CALL[0] = own_call
            dispatcher.__call__ = checked_call  # type: ignore[method-assign,assignment]
        _CHECKED_RUNS[0] += 1
    try:
        yield
    finally:
        with _CHECKED_RUNS_LOCK:
            _CHECKED_RUNS[0] -= 1
            if not _CHECKED_RUNS[0]:
                # Numba's Dispatcher inherits __call__ from a type of numba's C extension, so no
                # test reaches the branch that restores a __call__ of its own, which a later numba
                # could define.
                if _OWN_CALL[0] is None:
                    del dispatcher.__call__
                else:
                    dispatcher.__call__ = _OWN_CALL[0]  # type: ignore[method-assign]


class _CheckedTuning:
    """The context variable of the active tuning, which workers of ``run_work_items`` may not read.

    Code that threads run reads the tuning only from its search, since new threads start with the
    default tuning (see the module docstring of ``_tuning.py``).
    """

    def __init__(self, variable: Any) -> None:
        self.variable = variable

    def get(self) -> _tuning.SearchTuning:
        """The active tuning, which raises an ``AssertionError`` on a worker of a search."""
        if _get_running_workers():
            raise AssertionError("A worker of run_work_items read the tuning through get_tuning")
        return self.variable.get()

    def set(self, value: _tuning.SearchTuning) -> Any:
        """Set the active tuning, and return the token that resets it."""
        return self.variable.set(value)

    def reset(self, token: Any) -> None:
        """Reset the active tuning to its value before ``set`` returned a token."""
        self.variable.reset(token)


class _SharedArray(NamedTuple):
    """An array that threads share, with the path from the object that reaches it, and a copy."""

    path: str
    array: npt.NDArray[Any]
    copy: npt.NDArray[Any]
    writeable: bool


def _get_shared_arrays(*roots: object) -> list[_SharedArray]:
    """A record of each array that some objects reach: its path, a copy, and its writeable flag.

    The objects reach arrays through the items of dictionaries, lists, and tuples, and through the
    attributes of objects of qLDPC, which must have a ``__dict__``.  A path on which threads write
    shared data under a lock, such as a cache that they fill, would need that data left out of the
    records, for example by a parameter that names the attributes that this function skips, and a
    test of its own for the lock.  A path ``roots[i]...`` starts at the object ``roots[i]``.
    """
    arrays = []
    seen = set()
    pending: list[tuple[str, object]] = [
        (f"roots[{index}]", root) for index, root in enumerate(roots)
    ]
    while pending:
        path, value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            arrays.append(_SharedArray(path, value, value.copy(), bool(value.flags.writeable)))
        elif isinstance(value, dict):
            pending += [(f"{path}[{key!r}]", item) for key, item in value.items()]
        elif isinstance(value, list | tuple):
            pending += [(f"{path}[{index}]", item) for index, item in enumerate(value)]
        elif type(value).__module__.split(".")[0] == "qldpc":
            try:
                attributes = vars(value)
            except TypeError as error:
                raise TypeError(f"{path} has no __dict__") from error
            pending += [(f"{path}.{name}", item) for name, item in attributes.items()]
    return arrays


def _assert_unchanged(arrays: list[_SharedArray]) -> None:
    """Check that no shared array has changed since ``_get_shared_arrays`` copied it."""
    changed = [
        each.path
        for each in arrays
        if each.array.shape != each.copy.shape or each.array.tobytes() != each.copy.tobytes()
    ]
    assert not changed, f"Shared arrays changed: {changed}"


def _restore_flags(arrays: list[_SharedArray]) -> list[str]:
    """Make the arrays that were writeable writeable again, and return the paths of any that fail.

    NumPy lets a view become writeable only once its base is, so the arrays are restored in order of
    the length of their chains of bases.
    """
    unrestored = []
    for each in sorted(arrays, key=lambda each: _count_bases(each.array)):
        if each.writeable:
            try:
                each.array.flags.writeable = True
            except ValueError:
                unrestored.append(each.path)
    return unrestored


def _count_bases(array: npt.NDArray[Any]) -> int:
    """The number of arrays in the chain of bases of an array."""
    count = 0
    while isinstance(array.base, np.ndarray):
        array = array.base
        count += 1
    return count
