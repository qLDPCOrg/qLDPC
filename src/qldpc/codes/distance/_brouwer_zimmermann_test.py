# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact binary Brouwer-Zimmermann distance searches."""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y, z

import qldpc
from qldpc._binary import get_independent_rows
from qldpc.objects import Pauli, PauliXZ

from . import _automorphisms, _information_sets
from . import _brouwer_zimmermann as bz
from ._binary import get_nested_code_basis
from .conftest import get_random_full_rank_matrix


@pytest.mark.parametrize("interpreted", [False, True])
def test_brouwer_zimmermann_scan(interpreted: bool) -> None:
    """Chunked kernel calls enumerate each combination of basis rows once.

    The chunked kernel calls preserve resumption state and multiword labels.
    """
    scan = bz._scan_brouwer_zimmermann
    if interpreted:
        scan = getattr(scan, "py_func", scan)
    rng = np.random.default_rng(120)

    # The scan input uses full-rank, partial-rank, and rank-zero information sets.  The last two
    # information sets store labels only in their second packed word.
    for dimension in range(1, 7):
        generators = rng.integers(0, 2**64, (3, dimension, 5), dtype=np.uint64)
        labels = np.zeros((3, dimension, 2), dtype=np.uint64)
        labels[1:, :, 1] = rng.integers(0, 2, (2, dimension), dtype=np.uint64)
        ranks = np.array([dimension, dimension // 2, 0], dtype=np.int64)
        widths = np.array([5, 2, 1], dtype=np.int64)
        label_widths = np.array([0, 2, 2], dtype=np.int64)
        for weight in range(1, dimension + 1):
            support = np.zeros(weight, dtype=np.int64)
            prefixes = np.zeros((weight + 1, 3, 5), dtype=np.uint64)
            depth, best, expected = 0, 1000, 1000
            for indices in itertools.combinations(range(dimension), weight):
                # Update the expected running minimum from the current combination of basis rows.
                for index in range(3):
                    words = np.bitwise_xor.reduce(generators[index, indices, : widths[index]])
                    eligible = not label_widths[index] or np.any(
                        np.bitwise_xor.reduce(labels[index, indices], axis=0)
                    )
                    candidate = sum(int(word).bit_count() for word in words)
                    candidate += sum(row < ranks[index] for row in indices)
                    if eligible:
                        expected = min(expected, candidate)

                # Scan one basis-row combination and compare the running minimum and kernel state.
                best, depth, visited = scan(
                    generators,
                    labels,
                    ranks,
                    widths,
                    label_widths,
                    support,
                    prefixes,
                    depth,
                    best,
                    0,
                    1,
                )
                assert visited == 1 and best == expected
                # The last selected row index has advanced in preparation for the next chunk.
                np.testing.assert_array_equal(support[:-1], indices[:-1])
                assert support[-1] == indices[-1] + 1

            # One more kernel call reports completion after all basis-row combinations are visited.
            best, depth, visited = scan(
                generators,
                labels,
                ranks,
                widths,
                label_widths,
                support,
                prefixes,
                depth,
                best,
                0,
                1,
            )
            assert (best, depth, visited) == (expected, -1, 0)


def test_brouwer_zimmermann_search_state() -> None:
    """A search state handles partial-rank information sets.

    The search state handles quotient labels, cutoffs, and lower-bound updates.
    """
    # The input is one rank-zero information set with two basis rows.  The two basis rows have
    # Hamming weights 2 and 1, both outside the excluded subcode.
    search = bz._BrouwerZimmermannSearch(
        np.array([[[3], [2]]], dtype=np.uint64),
        np.ones((1, 2, 1), dtype=np.uint64),
        np.array([1], dtype=np.int64),
        np.array([1], dtype=np.int64),
        _information_sets.Certificate(2, (0,)),
        cutoff=1,
        batch_size=1,
        best=3,
        lower_bound=0,
    )

    # An interrupted coefficient-weight layer leaves the lower-bound certificate unchanged.
    with (
        mock.patch(
            "qldpc.codes.distance._brouwer_zimmermann._scan_brouwer_zimmermann",
            side_effect=KeyboardInterrupt,
        ),
        pytest.raises(KeyboardInterrupt),
    ):
        search.advance()
    assert (search.lower_bound, search.next_weight) == (0, 1)

    # Reaching the distance cutoff returns before the layer updates the certificate.
    assert search.advance() == 1
    assert (search.lower_bound, search.next_weight) == (0, 1)
    assert search.advance() == 1

    # Without an effective distance cutoff, each call finalizes one coefficient-weight layer.
    search.cutoff = -(2**100)
    assert search.advance() == 1
    assert (search.lower_bound, search.next_weight) == (0, 2)
    # After the last layer, every word has been visited, so the certificate rules out all of them.
    assert search.advance() == 1
    assert (search.lower_bound, search.next_weight) == (np.iinfo(np.int64).max, 3)


# Each case reaches one early exit from search setup or a shared search with real inputs.  These
# exits depend on cost heuristics, so retuning a heuristic can move a case to another exit.  If the
# coverage gate then fails, find a new natural input here rather than stubbing internals.
SETUP_EXIT_CASES = [
    # A shared search reaches the cutoff inside a coefficient-weight scan.
    pytest.param(
        [
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
            [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
        ],
        [],
        2,
        False,
        id="scan-cutoff",
    ),
    # Brute force is cheaper than the information-set search prepared for this quotient.
    pytest.param(
        [[1, 0, 0, 0, 1, 0, 1, 1, 0]],
        [
            [0, 1, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 0, 1],
        ],
        0,
        False,
        id="brute-force",
    ),
    # Rows of an overlapping replacement family reach the cutoff.
    pytest.param(
        [[1, 0, 0, 0, 0, 0, 0, 1, 1, 1]],
        [
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        ],
        3,
        True,
        id="overlap-cutoff",
    ),
    # The certificate of an overlapping replacement family closes the gap to the best witness.
    pytest.param(
        [[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0]],
        [
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        ],
        0,
        True,
        id="overlap-certificate",
    ),
]


@pytest.mark.parametrize("logicals, stabilizers, cutoff, overlap", SETUP_EXIT_CASES)
def test_brouwer_zimmermann_setup_exits(
    logicals: list[list[int]],
    stabilizers: list[list[int]],
    cutoff: int,
    overlap: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Early exits from search setup or a shared search agree with a brute-force oracle."""
    expected = qldpc.codes.get_distance_quantum(
        logicals, stabilizers, homogeneous=True, cutoff=0, method="brute_force"
    )

    # Lower only the cost knobs, so that these small inputs reach the information-set search.
    monkeypatch.setattr(bz, "enumeration_is_cheaper_than_setup", lambda *_: False)
    if overlap:
        monkeypatch.setattr(_information_sets, "MINIMUM_WORK", 0)
    distance = qldpc.codes.get_distance_quantum(
        logicals, stabilizers, homogeneous=True, cutoff=cutoff
    )
    assert expected <= distance <= max(cutoff, expected)


def test_brouwer_zimmermann_setup_cutoffs() -> None:
    """Preparation can reach the cutoff from initial rows or a newly systematic row."""
    basis = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    assert (
        bz._prepare_brouwer_zimmermann_search(
            basis, _no_labels(basis), cutoff=3, block_size=2, weight_divisor=1, upper_bound=4
        )
        == 3
    )
    with mock.patch.object(bz, "enumeration_is_cheaper_than_setup", return_value=False):
        assert (
            bz._prepare_brouwer_zimmermann_search(
                basis, _no_labels(basis), cutoff=2, block_size=2, weight_divisor=1, upper_bound=4
            )
            == 2
        )


def test_an_upper_bound_acts_as_a_known_witness() -> None:
    """A search returns the smaller of its upper bound and the distance."""
    basis = get_independent_rows(np.asarray(qldpc.codes.HammingCode(3).generator, dtype=np.uint8))
    for upper_bound, expected in ((2, 2), (3, 3), (5, 3)):
        distance = bz.get_distance_brouwer_zimmermann(
            basis, _no_labels(basis), cutoff=0, block_size=15, upper_bound=upper_bound
        )
        assert distance == expected


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_shared_search() -> None:
    """A shared Brouwer-Zimmermann search returns the minimum distance across nested codes."""
    rng = np.random.default_rng(119)
    basis_a = get_random_full_rank_matrix(rng, 5, 8)
    basis_b = get_random_full_rank_matrix(rng, 5, 9)
    labels_b = np.vstack(
        [
            np.zeros((2, 3), dtype=np.uint8),
            np.eye(3, dtype=np.uint8),
        ]
    )

    expected_a = qldpc.codes.get_distance_classical(
        basis_a,
        cutoff=0,
        method="brute_force",
    )
    expected_b = qldpc.codes.get_distance_quantum(
        basis_b[2:],
        basis_b[:2],
        cutoff=0,
        homogeneous=True,
        method="brute_force",
    )
    actual = bz.get_distance_brouwer_zimmermann_many(
        [(basis_a, _no_labels(basis_a)), (basis_b, labels_b)],
        cutoff=0,
        block_size=2,
    )
    assert actual == min(expected_a, expected_b)


@pytest.mark.usefixtures("force_information_sets")
def test_a_given_disjoint_basis_family_replaces_the_first_stage() -> None:
    """A search that reaches the first stage takes its result from the given function instead.

    A search that exits before the first stage, here at the weight of a row, does not call it.
    """
    rng = np.random.default_rng(121)
    basis = get_random_full_rank_matrix(rng, 5, 9)
    labels = np.vstack([np.zeros((2, 3), dtype=np.uint8), np.eye(3, dtype=np.uint8)])
    expected = bz.get_distance_brouwer_zimmermann_many([(basis, labels)], cutoff=0, block_size=2)
    get_basis_family = mock.Mock(return_value=_information_sets.get_disjoint_family(basis, labels))
    search = functools.partial(
        bz.get_distance_brouwer_zimmermann_many,
        [(basis, labels)],
        block_size=2,
        get_disjoint_basis_families=[get_basis_family],
    )
    with mock.patch.object(
        _information_sets, "get_disjoint_family", wraps=_information_sets.get_disjoint_family
    ) as first_stage:
        assert search(cutoff=0) == expected
        get_basis_family.assert_called_once_with()
        search(cutoff=basis.shape[1])
        get_basis_family.assert_called_once_with()
    first_stage.assert_not_called()


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_sector_certificates() -> None:
    """A completed sector's certificate must not finish a different sector's search."""
    generators = np.asarray(qldpc.codes.HammingCode(4).generator, dtype=np.uint8)
    basis_a = get_independent_rows(
        np.column_stack([generators, np.bitwise_xor.reduce(generators, axis=1)])
    )
    basis_b = get_independent_rows(
        np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0, 1, 1],
                [1, 0, 1, 1, 0, 0, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
                [0, 1, 1, 0, 0, 0, 1, 1, 1, 0],
                [0, 1, 0, 0, 1, 0, 0, 1, 1, 1],
            ],
            dtype=np.uint8,
        )
    )
    expected = [
        qldpc.codes.get_distance_classical(basis, cutoff=0, method="brute_force")
        for basis in (basis_a, basis_b)
    ]
    assert expected == [4, 2]

    searches: list[bz._BrouwerZimmermannSearch] = []
    advanced: set[int] = set()
    original_prepare = bz._prepare_brouwer_zimmermann_search
    original_advance = bz._BrouwerZimmermannSearch.advance

    def prepare(*args: Any, **kwargs: Any) -> int | bz._BrouwerZimmermannSearch:
        result = original_prepare(*args, **kwargs)
        if isinstance(result, bz._BrouwerZimmermannSearch):
            searches.append(result)
        return result

    def advance(search: bz._BrouwerZimmermannSearch) -> int:
        advanced.add(id(search))
        return original_advance(search)

    # Exercise real layer advancement, rather than two results certified during preparation.
    with (
        mock.patch.object(bz, "_prepare_brouwer_zimmermann_search", side_effect=prepare),
        mock.patch.object(bz._BrouwerZimmermannSearch, "advance", advance),
    ):
        actual = bz.get_distance_brouwer_zimmermann_many(
            [(basis_a, _no_labels(basis_a)), (basis_b, _no_labels(basis_b))], cutoff=0, block_size=2
        )
    assert actual == min(expected)
    assert len(searches) == 2
    assert {id(search) for search in searches} <= advanced


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_subsystem_code() -> None:
    """Information-set search treats the gauge operators of a subsystem code as trivial.

    The 3x3 Bacon-Shor code is a [[9,1,3]] subsystem code; see
    https://arxiv.org/abs/quant-ph/0506023.
    """
    code = qldpc.codes.QuditCode(qldpc.codes.BaconShorCode(3).matrix, is_subsystem_code=True)
    assert code.get_distance_exact() == 3


@pytest.mark.usefixtures("force_information_sets")
@pytest.mark.parametrize("length", [64, 65, 257])
def test_brouwer_zimmermann_packed_widths(length: int) -> None:
    """Information-set weights are exact across uint64 words and for weights above 255."""
    generators = np.ones((1, length), dtype=np.uint8)
    assert qldpc.codes.get_distance_classical(generators, cutoff=0) == length


def test_dependent_logical_rows_give_the_distance_of_their_span() -> None:
    """Repeated logical rows are reduced to a basis of the quotient before any search."""
    distance = qldpc.codes.get_distance_quantum(
        [[1, 1, 1], [1, 1, 1]], [], homogeneous=True, cutoff=0
    )
    assert distance == 3


@pytest.mark.parametrize("extended", [False, True])
def test_brouwer_zimmermann_reduces_enumeration_work(
    extended: bool, get_candidate_count: Callable[[], int]
) -> None:
    """Hamming(5) and its even-weight extension need far less work than exhaustive search."""
    generators = np.asarray(qldpc.codes.HammingCode(5).generator, dtype=np.uint8)
    if extended:
        generators = np.column_stack([generators, np.bitwise_xor.reduce(generators, axis=1)])
    assert qldpc.codes.get_distance_classical(generators, cutoff=0) == (4 if extended else 3)
    assert 0 < 1000 * get_candidate_count() < 2 ** len(generators)


def test_the_lightest_eligible_row_excludes_stabilizers() -> None:
    """Upper bounds from rows come from eligible rows, not lighter stabilizers."""
    basis = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    labels = np.array([[0], [1]], dtype=np.uint8)
    assert bz._get_lightest_eligible_row_weight(basis, labels) == 2


def test_brouwer_zimmermann_overlap_qc(get_candidate_count: Callable[[], int]) -> None:
    """The default QC64 X-sector search stays within a broad candidate-work budget."""
    code = qldpc.codes.QCCode(
        {x: 4, y: 4, z: 2},
        1 + x + y + x * y * z,
        1 + x * y + x**2 * y**3 * z + x**3 * y**2,
    )
    assert code.forget_distance().get_distance_exact("X", cutoff=0) == 8
    assert 0 < get_candidate_count() < 1_000_000


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_overlap_improves_quotient_witness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare witness-improving overlap enumeration with an exhaustive quotient oracle."""

    # Shorten a systematic BCH(31,21) basis to [26,16] and isolate a one-bit quotient.
    basis = get_independent_rows(np.asarray(qldpc.codes.BCHCode(31, 21).generator, dtype=np.uint8))[
        5:, 5:
    ]
    logicals = basis[13:14]
    stabilizers = np.delete(basis, 13, axis=0)
    expected = qldpc.codes.get_distance_quantum(
        logicals, stabilizers, homogeneous=True, cutoff=0, method="brute_force"
    )

    # Lower only the work threshold so this small oracle exercises production basis family
    # construction, and check that the overlapping family has no witness as light as the distance.
    monkeypatch.setattr(_information_sets, "MINIMUM_WORK", 0)
    original = _information_sets.get_overlapping_family

    def select(*args: Any, **kwargs: Any) -> _information_sets.BasisFamily:
        selected = original(*args, **kwargs)
        assert len(selected.ranks) and selected.max_coverage > 1
        assert (
            min(
                bz._get_lightest_eligible_row_weight(generators, labels)
                for generators, labels in zip(selected.generators, selected.labels, strict=True)
            )
            > expected
        )
        return selected

    with mock.patch.object(
        _information_sets, "get_overlapping_family", side_effect=select
    ) as select_mock:
        assert (
            qldpc.codes.get_distance_quantum(logicals, stabilizers, homogeneous=True, cutoff=0)
            == expected
            == 5
        )
    select_mock.assert_called()


@pytest.fixture
def get_candidate_count(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Count row-weight evaluations and scanned supports across BZ and exhaustive routes."""
    num_candidates = 0
    original_scan = bz._scan_brouwer_zimmermann

    def counting_scan(*args: Any) -> tuple[int, int, int]:
        nonlocal num_candidates
        best, depth, visited = original_scan(*args)
        num_candidates += visited * len(args[2])
        return best, depth, visited

    bitwise_count = np.bitwise_count

    def counting_bitwise_count(
        arr: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64] | None = None
    ) -> npt.NDArray[np.uint64]:
        nonlocal num_candidates
        num_candidates += len(arr)
        return bitwise_count(arr, out=out)

    # Row weights and exhaustive enumeration count every packed word with np.bitwise_count.
    monkeypatch.setattr(np, "bitwise_count", counting_bitwise_count)
    monkeypatch.setattr(bz, "_scan_brouwer_zimmermann", counting_scan)
    return lambda: num_candidates


@pytest.mark.usefixtures("force_information_sets")
def test_a_search_scans_every_set_that_its_certificate_counts() -> None:
    """A [22, 7, 5] code whose search finds a word of weight five only in its second set.

    Its two disjoint information sets reach the certificate six once both complete layer two, and
    the first word of weight five is in layer two of the second set.  A search that counted the
    second set's layer without scanning it would return six.
    """
    rows = [
        "1000100010001101001110",
        "0100100010000111110000",
        "0010000010000100101010",
        "0001100010000000011111",
        "0000010110000001011001",
        "0000001100000010001011",
        "0000000000101110101110",
    ]
    basis = np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)
    assert qldpc.codes.get_distance_classical(basis, cutoff=0, method="brute_force") == 5
    assert qldpc.codes.get_distance_classical(basis, cutoff=0) == 5


@pytest.mark.usefixtures("force_information_sets")
def test_a_search_through_the_costly_stages_finds_the_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A [21, 8, 6] code, searched with a partitioned family and a certificate of linear programs.

    With the thresholds of the costly stages lowered, the search of this code replaces its family by
    information sets from matroid partitioning, whose columns lie in up to two of them, and bounds
    unvisited words by linear programming.
    """
    monkeypatch.setattr(_information_sets, "MINIMUM_WORK", 0)
    monkeypatch.setattr(_information_sets, "LP_MINIMUM_WORK", 0)
    monkeypatch.setattr(_information_sets, "PARTITION_WORK_RATIO", 0)
    rows = [
        "100000010110010010011",
        "010000010100101111100",
        "001000000111011111100",
        "000100010101000101010",
        "000010010110000011000",
        "000001010011000111011",
        "000000110001110101011",
        "000000001110001010110",
    ]
    basis = np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)
    assert qldpc.codes.get_distance_classical(basis, cutoff=0, method="brute_force") == 6
    assert qldpc.codes.get_distance_classical(basis, cutoff=0) == 6


@pytest.fixture
def switch_before_any_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let searches consider translates before their first layer, however cheap it is."""
    monkeypatch.setattr(_automorphisms, "MINIMUM_LAYER_WORK", 0)


@pytest.mark.usefixtures("force_information_sets", "switch_before_any_layer")
@pytest.mark.parametrize(
    "orders, poly_a, poly_b, pauli",
    [
        ({x: 6, y: 3}, x**3 * y**2 + x**3 * y + x**3 + x * y, x**2 * y + x * y, Pauli.Z),
        ({x: 2, y: 3, z: 4}, x * y * z**2 + x * z**3, x * y**2 * z**3 + x * y * z**2, Pauli.X),
        ({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x, Pauli.X),
    ],
)
def test_translated_searches_scan_every_layer_they_certify(
    orders: dict[Any, int], poly_a: Any, poly_b: Any, pauli: PauliXZ
) -> None:
    """Sectors whose translated searches find a lightest word only in their last layer.

    A search that stopped a layer early, or certified layers without scanning them, would
    overestimate these distances.
    """
    code = qldpc.codes.QCCode(orders, poly_a, poly_b)
    logical_ops = np.asarray(code.get_logical_ops(pauli), dtype=np.uint8)
    stabilizers = np.asarray(code.get_stabilizer_ops(pauli, canonicalized=True), dtype=np.uint8)
    expected = qldpc.codes.get_distance_quantum(
        logical_ops, stabilizers, homogeneous=True, cutoff=0, method="brute_force"
    )
    distance = bz.get_distance_brouwer_zimmermann_many(
        [get_nested_code_basis(logical_ops, stabilizers)],
        cutoff=0,
        block_size=15,
        get_automorphisms=code._get_qudit_automorphisms,
    )
    assert distance == expected


@pytest.mark.usefixtures("force_information_sets", "switch_before_any_layer")
def test_translated_searches_find_exact_distances() -> None:
    """Searches that can switch to translates find exact distances.

    The translations of random quasi-cyclic codes preserve both sectors.  The searches, of each
    sector alone and of both sectors together, agree with brute force, as do searches given no
    automorphisms, the identity alone, which does not help, or a function that returns None.
    """
    rng = np.random.default_rng(2)
    groups = [((9,), (x,)), ((14,), (x,)), ((5, 3), (x, y)), ((4, 2, 2), (x, y, z))]
    num_codes = 0
    while num_codes < 16:
        orders, symbols = groups[num_codes % len(groups)]
        polys = [
            sum(
                np.prod(
                    [symbol ** int(rng.integers(order)) for symbol, order in zip(symbols, orders)]
                )
                for _ in range(int(rng.integers(2, 5)))
            )
            for _ in range(2)
        ]
        code = qldpc.codes.QCCode(dict(zip(symbols, orders)), *polys)
        if not code.dimension:
            continue
        num_codes += 1
        identity = np.arange(len(code))[None]
        problems = []
        for pauli in (Pauli.X, Pauli.Z):
            logical_ops = np.asarray(code.get_logical_ops(pauli), dtype=np.uint8)
            stabilizers = np.asarray(code.get_stabilizer_ops(pauli, canonicalized=True), np.uint8)
            expected = qldpc.codes.get_distance_quantum(
                logical_ops, stabilizers, homogeneous=True, cutoff=0, method="brute_force"
            )
            problems.append(get_nested_code_basis(logical_ops, stabilizers))
            for get_automorphisms in (
                None,
                code._get_qudit_automorphisms,
                lambda identity=identity: identity,
                lambda: None,
            ):
                distance = bz.get_distance_brouwer_zimmermann_many(
                    problems[-1:], cutoff=0, block_size=15, get_automorphisms=get_automorphisms
                )
                assert distance == expected
        assert bz.get_distance_brouwer_zimmermann_many(
            problems, cutoff=0, block_size=15, get_automorphisms=code._get_qudit_automorphisms
        ) == code.get_distance(method="brute_force")


@pytest.mark.usefixtures("force_information_sets")
def test_searches_ask_for_automorphisms_only_before_costly_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A search asks for automorphisms once, and only before a layer that costs enough to repay.

    Finding automorphisms can cost more than an easy search, so a search that finishes before any
    costly layer does not ask for them.
    """
    code = qldpc.codes.QCCode({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    problem = get_nested_code_basis(
        np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8),
        np.asarray(code.get_stabilizer_ops(Pauli.X, canonicalized=True), dtype=np.uint8),
    )
    for minimum_layer_work, expected_calls in ((10**9, 0), (0, 1)):
        monkeypatch.setattr(_automorphisms, "MINIMUM_LAYER_WORK", minimum_layer_work)
        get_automorphisms = mock.Mock(wraps=code._get_qudit_automorphisms)
        bz.get_distance_brouwer_zimmermann_many(
            [problem], cutoff=0, block_size=15, get_automorphisms=get_automorphisms
        )
        assert get_automorphisms.call_count == expected_calls


@pytest.mark.usefixtures("force_information_sets")
def test_translated_searches_restart_from_layer_two(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search that switches to translates before a later layer scans its new set from layer two.

    The certificate of translates counts every layer of the scanned set, including the layers that
    the search scanned only with its previous family.  The [[90, 8, 10]] bivariate bicycle code of
    arXiv:2308.07915 takes more than one layer to search.
    """
    code = qldpc.codes.BBCode({x: 15, y: 3}, x**9 + y + y**2, 1 + x**2 + x**7)
    basis, labels = get_nested_code_basis(
        np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8),
        np.asarray(code.get_stabilizer_ops(Pauli.X, canonicalized=True), dtype=np.uint8),
    )
    search = bz._prepare_brouwer_zimmermann_search(
        basis,
        labels,
        cutoff=0,
        block_size=15,
        weight_divisor=1,
        upper_bound=basis.shape[1] + 1,
        get_automorphisms=code._get_qudit_automorphisms,
    )
    assert isinstance(search, bz._BrouwerZimmermannSearch)

    # Scan layer two with the disjoint family, and then switch before layer three.
    monkeypatch.setattr(_automorphisms, "MINIMUM_LAYER_WORK", 10**12)
    search.advance()
    assert not search.finished
    monkeypatch.setattr(_automorphisms, "MINIMUM_LAYER_WORK", 0)
    scanned_weights = []
    original_scan = bz._scan_brouwer_zimmermann

    def scan(*args: Any) -> tuple[int, int, int]:
        scanned_weights.append(len(args[5]))
        return original_scan(*args)

    monkeypatch.setattr(bz, "_scan_brouwer_zimmermann", scan)
    while not search.finished:
        search.advance()
    assert scanned_weights[0] == 2


@pytest.mark.usefixtures("force_information_sets", "switch_before_any_layer")
def test_translated_rows_can_reach_the_cutoff() -> None:
    """The rows of a scanned set with translates can end a search at the cutoff.

    In this sector, the rows of the disjoint basis family have weight five or more, and a row of the
    scanned set has weight four.
    """
    code = qldpc.codes.QCCode({x: 5, y: 3}, x**4 * y + x + y**2, x**4 * y + x**3 * y)
    logical_ops = np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8)
    stabilizers = np.asarray(code.get_stabilizer_ops(Pauli.X, canonicalized=True), dtype=np.uint8)
    expected = qldpc.codes.get_distance_quantum(
        logical_ops, stabilizers, homogeneous=True, cutoff=0, method="brute_force"
    )
    cutoff = 4
    distance = bz.get_distance_brouwer_zimmermann_many(
        [get_nested_code_basis(logical_ops, stabilizers)],
        cutoff=cutoff,
        block_size=15,
        get_automorphisms=code._get_qudit_automorphisms,
    )
    assert expected <= distance <= max(cutoff, expected)


@pytest.mark.usefixtures("force_information_sets", "switch_before_any_layer")
def test_searches_reject_wrong_automorphisms() -> None:
    """A permutation that does not preserve a problem fails loudly when a search would use it."""
    code = qldpc.codes.QCCode({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    problem = get_nested_code_basis(
        np.asarray(code.get_logical_ops(Pauli.X), dtype=np.uint8),
        np.asarray(code.get_stabilizer_ops(Pauli.X, canonicalized=True), dtype=np.uint8),
    )
    transposition = np.arange(len(code))
    transposition[[0, 1]] = [1, 0]
    with pytest.raises(ValueError, match="not an automorphism"):
        bz.get_distance_brouwer_zimmermann_many(
            [problem], cutoff=0, block_size=15, get_automorphisms=lambda: transposition[None]
        )


def _no_labels(basis: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Quotient labels with no columns, which exclude no subcode."""
    return np.empty((len(basis), 0), dtype=np.uint8)
