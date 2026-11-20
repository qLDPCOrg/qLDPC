# SPDX-License-Identifier: Apache-2.0

"""Tests of the grid of problems, methods, and modes in "The grid" of this package's README.

The first table of that section says which methods each problem takes, and how they run.  These
tests check the table against the methods that each problem's entry point accepts, and run every
supported cell, in each of its modes, on small random instances against brute force.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from qldpc import codes, math
from qldpc.codes.conftest import Tune
from qldpc.codes.distance import (
    CosetLeaderSearch,
    count_logical_ops_css,
    get_distance_css,
    get_distance_from_checks,
    get_distance_from_generators,
    get_distance_pauli,
)

from . import _threads, _tuning, core, logical_counts
from .conftest import (
    assert_exact_answers,
    enumerate_logical_ops,
    get_exhaustive_minima,
    get_golay_checks,
    get_graph_state,
    get_random_css_code,
    get_random_css_state,
    get_span,
    scramble,
    split_syndromes_into_words,
)

# The modes of each kind of cell of the table.
_MODES = {"threads": ("one thread", "threads"), "one thread": ("one thread",)}


def _read_grid() -> dict[str, dict[str, str]]:
    """The first table of "The grid" in the README: the cell of each problem and method."""
    text = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
    section = text.split("\n## The grid\n", 1)[1].split("\n## ", 1)[0]
    table = section.split("\n| ", 1)[1].split("\n\n", 1)[0]
    rows = [
        [cell.strip(" `") for cell in line.strip("| ").split("|")] for line in table.split("\n")
    ]
    methods = rows[0][1:]
    return {row[0]: dict(zip(methods, row[1:], strict=True)) for row in rows[2:]}


_GRID = _read_grid()
_CELLS = [
    (problem, method, mode)
    for problem, row in _GRID.items()
    for method, cell in row.items()
    for mode in _MODES.get(cell, ())
]


####################################################################################################
# the grid


def test_the_grid_lists_every_problem_and_the_methods_that_it_takes() -> None:
    """The table has a column for each method, and each problem takes the methods of its cells."""
    assert set(_GRID) == set(_ACCEPTS) == set(_CHECKS)
    methods = {*core._DISTANCE_METHODS.values(), *logical_counts._LOGICAL_COUNT_METHODS.values()}
    for problem, row in _GRID.items():
        assert set(row) == methods
        assert set(row.values()) <= {*_MODES, "not yet", "not by design"}
        for method, cell in row.items():
            assert _ACCEPTS[problem](method) == (cell in _MODES), (problem, method, cell)


@pytest.mark.parametrize(("problem", "method", "mode"), _CELLS)
def test_every_cell_answers_as_brute_force(
    problem: str,
    method: str,
    mode: str,
    tune: Tune,
    monkeypatch: pytest.MonkeyPatch,
    pytestconfig: pytest.Config,
) -> None:
    """Every supported cell gives brute force's answers, using threads exactly in "threads" mode.

    ``"auto"`` never enumerates here, so that it takes the routes of the other methods.  In the
    "threads" mode, and in every "one thread" cell, the test allows two threads and search chunks
    of one unit, so that the searches of small instances split wherever they can.
    """
    threads = mode == "threads"
    if threads or _GRID[problem][method] == "one thread":
        tune(num_threads=2, search_chunk_exponent=0, auto_brute_force_max_words=0)
    else:
        tune(num_threads=1, auto_brute_force_max_words=0)

    # Record the number of threads of each run of work items, which every threaded path starts.
    thread_counts: list[int] = []
    run_work_items = _threads.run_work_items

    def record(num_items: int, num_threads: int, *args: Any, **kwargs: Any) -> None:
        thread_counts.append(min(num_items, num_threads))
        run_work_items(num_items, num_threads, *args, **kwargs)

    monkeypatch.setattr(_threads, "run_work_items", record)

    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    _CHECKS[problem](method, rng, threads)
    assert any(count > 1 for count in thread_counts) == threads


####################################################################################################
# the problems


def _accepts_distance(method: Any) -> bool:
    """Whether the distance functions accept a method."""
    return _accepts(lambda: get_distance_css([[1, 1, 1, 1]], [[1, 1, 1, 1]], method=method))


def _accepts_vector(method: Any) -> bool:
    """Whether the coset weight of a vector accepts a method."""
    code = codes.ClassicalCode([[1, 1, 1]])
    return _accepts(lambda: code.get_distance_exact(vector=[1, 1, 1], method=method))


def _accepts_coset_leaders(method: Any) -> bool:
    """Whether ``CosetLeaderSearch`` accepts a method."""
    return _accepts(lambda: CosetLeaderSearch([[0, 0, 1, 0], [0, 0, 0, 1]], method=method))


def _accepts_counts(method: Any) -> bool:
    """Whether logical counts accept a method."""
    return _accepts(lambda: count_logical_ops_css([[1, 1, 1, 1]], [[1, 1, 1, 1]], method=method))


def _check_distances(method: Any, rng: np.random.Generator, threads: bool) -> None:
    """Distances of classical, CSS, and other stabilizer codes, against brute force."""
    # A classical code, from its checks and, except by the connected-cluster method, which needs
    # checks, from its generators.
    checks = rng.integers(2, size=(4, 9), dtype=np.uint8)
    words = _get_words_with_syndrome(checks, 0)
    distance = min(int(word.sum()) for word in words if word.any())
    assert get_distance_from_checks(checks, method=method) == distance
    if method != "connected_cluster":
        assert get_distance_from_generators(words, method=method) == distance

    # A CSS code.
    code = get_random_css_code(rng, 8, 2, 3)
    expected = enumerate_logical_ops(code)[2]
    assert get_distance_css(code.matrix_x, code.matrix_z, method=method) == expected

    # A stabilizer code that is not CSS: a CSS code with S on random qubits, which turns some X
    # checks into checks with Y.
    code = get_random_css_code(rng, 6, 2, 2)
    num_qubits = len(code)
    pauli_checks = np.zeros((len(code.matrix), 2 * num_qubits), dtype=np.uint8)
    pauli_checks[: len(code.matrix_x), :num_qubits] = code.matrix_x
    pauli_checks[len(code.matrix_x) :, num_qubits:] = code.matrix_z
    phased = rng.integers(2, size=num_qubits, dtype=np.uint8)
    pauli_checks[:, num_qubits:] ^= pauli_checks[:, :num_qubits] & phased
    expected = _get_pauli_distance(pauli_checks)
    assert get_distance_pauli(pauli_checks, method=method) == expected

    # On threads, codes whose searches split: the [23, 12, 7] Golay code from its checks, whose BZ
    # search scans layers of three rows (Golay, Proc. IRE 37, 657 (1949)), and the toric code of
    # distance four, whose connected-cluster search, which "auto" runs, splits its weight limits
    # (Kitaev, Ann. Phys. 303, 2 (2003)).
    if threads and method == "brouwer_zimmermann":
        assert get_distance_from_checks(get_golay_checks(), method=method) == 7
    if threads and method in ("connected_cluster", "auto"):
        toric = codes.ToricCode(4)
        assert get_distance_css(toric.matrix_x, toric.matrix_z, method=method) == 4


def _check_vector_cosets(method: Any, rng: np.random.Generator, threads: bool) -> None:
    """The coset weights of vectors of a classical code, against brute force."""
    # The coset of each vector holds the words with its syndrome.
    checks = rng.integers(2, size=(4, 10), dtype=np.uint8)
    code = codes.ClassicalCode(checks)
    for vector in rng.integers(2, size=(4, len(code)), dtype=np.uint8):
        syndrome = int(checks @ vector % 2 @ (1 << np.arange(len(checks))))
        expected = int(_get_words_with_syndrome(checks, syndrome).sum(axis=1).min())
        assert code.get_distance_exact(vector=vector, method=method) == expected

    # On threads, a fixed code and vector whose coset weighs five, so that its BZ search reaches a
    # layer of three rows, which splits onto threads; a small random case may stop earlier.  Its
    # expected weight comes from the span of the code's generator, which the library computes; the
    # random cases above use an oracle independent of the library.
    if threads:
        fixed_rng = np.random.default_rng(0)
        code = codes.ClassicalCode(fixed_rng.integers(2, size=(14, 30), dtype=np.uint8))
        vector = fixed_rng.integers(2, size=(1, 30), dtype=np.uint8)[0]
        value = sum(int(bit) << jj for jj, bit in enumerate(vector))
        expected = min((word ^ value).bit_count() for word in get_span(code.generator))
        assert code.get_distance_exact(vector=vector, method=method) == expected


def _check_coset_leaders(method: Any, rng: np.random.Generator, threads: bool) -> None:
    """Single and batch queries of coset leaders, against the exhaustive minima of each state."""
    cases = [
        (get_random_css_state(5, 2, rng), True),
        (get_random_css_state(5, 2, rng), False),
        (scramble(get_graph_state(5, [(0, 1), (1, 2), (2, 3), (3, 4)], phase=True), rng), False),
    ]
    for matrix, css_weight in cases:
        search = CosetLeaderSearch(matrix, css_weight=css_weight, method=method)
        assert_exact_answers(search, matrix, css_weight)

        num_qubits = len(matrix)
        minima = get_exhaustive_minima(matrix, css_weight)
        syndromes = rng.permutation(1 << num_qubits)
        batch = split_syndromes_into_words(syndromes.tolist(), num_qubits)
        for max_weight in range(num_qubits + 1):
            weights = search.get_min_weights(batch, max_weight)
            assert np.array_equal(weights, np.minimum(minima[syndromes], max_weight + 1))
            outside = np.flatnonzero(minima[syndromes] > max_weight)
            index, _ = search.find_first_not_within_max_weight(batch, max_weight)
            assert index == (outside[0] if len(outside) else len(syndromes))


def _check_counts(method: Any, rng: np.random.Generator, threads: bool) -> None:
    """Counts of logical operators at the distance and above it, against enumeration."""
    code = get_random_css_code(rng, 7, 2, 2)
    counts_x, counts_z, distance, full = enumerate_logical_ops(code)

    # A count by brouwer_zimmermann at its default weight runs a distance search, which can use
    # threads (see "The grid" in the README), so the default-weight count runs on one thread here.
    count_pauli = method != "macwilliams" or code.is_self_dual
    with _tuning.use_tuning(num_threads=1):
        counts = count_logical_ops_css(
            code.matrix_x, code.matrix_z, method=method, count_pauli=count_pauli
        )
    assert counts.weight == distance
    assert (counts.count_x, counts.count_z) == (counts_x[distance], counts_z[distance])
    assert counts.count_pauli == (full if count_pauli else None)
    for weight in (distance, distance + 1):
        counts = count_logical_ops_css(code.matrix_x, code.matrix_z, weight, method=method)
        assert (counts.count_x, counts.count_z) == (counts_x[weight], counts_z[weight])


_ACCEPTS: dict[str, Callable[[Any], bool]] = {
    "Distance": _accepts_distance,
    "Coset weight of a vector": _accepts_vector,
    "Coset leaders": _accepts_coset_leaders,
    "Logical counts": _accepts_counts,
}
_CHECKS: dict[str, Callable[[Any, np.random.Generator, bool], None]] = {
    "Distance": _check_distances,
    "Coset weight of a vector": _check_vector_cosets,
    "Coset leaders": _check_coset_leaders,
    "Logical counts": _check_counts,
}


####################################################################################################
# helpers


def _accepts(call: Callable[[], object]) -> bool:
    """Whether a call runs, rather than rejecting its method with a ``ValueError``."""
    try:
        call()
    except ValueError as error:
        assert "method" in str(error), error
        return False
    return True


def _get_paulis(num_qubits: int) -> npt.NDArray[np.int_]:
    """Every Pauli string on ``num_qubits`` qubits, as rows ``[x|z]``."""
    return np.array(list(itertools.product([0, 1], repeat=2 * num_qubits)), dtype=int)


def _get_words_with_syndrome(checks: npt.NDArray[np.uint8], syndrome: int) -> npt.NDArray[np.int_]:
    """Every binary word with a packed syndrome for the checks, by enumeration, one per row."""
    num_bits = checks.shape[1]
    words = (np.arange(1 << num_bits)[:, None] >> np.arange(num_bits)) & 1
    syndromes = (words @ checks.T % 2) @ (1 << np.arange(len(checks)))
    return words[syndromes == syndrome]


def _get_pauli_distance(checks: npt.NDArray[np.uint8]) -> int:
    """The least weight of a Pauli string that commutes with the checks but is not a stabilizer."""
    num_qubits = checks.shape[1] // 2
    paulis = _get_paulis(num_qubits)
    commutes = ~np.any(paulis @ math.symplectic_conjugate(checks.astype(int)).T % 2, axis=1)
    stabilizers = get_span(checks)
    weights = np.count_nonzero(paulis[:, :num_qubits] | paulis[:, num_qubits:], axis=1)
    return min(
        int(weight)
        for pauli, weight, commuting in zip(paulis, weights, commutes, strict=True)
        if commuting and sum(int(bit) << jj for jj, bit in enumerate(pauli)) not in stabilizers
    )
