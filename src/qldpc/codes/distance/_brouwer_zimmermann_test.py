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

from . import _binary as binary
from . import _brouwer_zimmermann as bz
from . import _information_sets, core
from .conftest import get_random_full_rank_matrix


def test_css_brouwer_zimmermann_special_inputs() -> None:
    """The joint CSS helper handles empty, trivial, and nontrivial quotient sectors."""
    empty = np.empty((0, 4), dtype=np.uint8)
    assert (
        bz.get_distance_css_brouwer_zimmermann(
            [(empty, empty)],
            cutoff=0,
        )
        == 4
    )
    assert (
        bz.get_distance_css_brouwer_zimmermann(
            [([[1, 0]], [[1, 0]])],
            cutoff=0,
        )
        == 0
    )

    # The result is the minimum of the upper bound and the distances of the supplied sectors.
    sectors: list[tuple[npt.ArrayLike, npt.ArrayLike]] = [([[1, 1]], [])]
    assert bz.get_distance_css_brouwer_zimmermann(sectors, cutoff=0, upper_bound=3) == 2
    assert bz.get_distance_css_brouwer_zimmermann(sectors, cutoff=0, upper_bound=1) == 1


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
    assert search.advance() == 1
    assert (search.lower_bound, search.next_weight) == (1, 3)


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
    monkeypatch.setattr(bz, "exhaustive_is_cheaper", lambda *_: False)
    monkeypatch.setattr(core, "exhaustive_is_cheaper", lambda *_: False)
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
            basis, None, cutoff=3, block_size=2, weight_divisor=1, upper_bound=4
        )
        == 3
    )
    with mock.patch.object(bz, "exhaustive_is_cheaper", return_value=False):
        assert (
            bz._prepare_brouwer_zimmermann_search(
                basis, None, cutoff=2, block_size=2, weight_divisor=1, upper_bound=4
            )
            == 2
        )


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
        [(basis_a, None), (basis_b, labels_b)],
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
    basis_a = binary.get_independent_rows(
        np.column_stack([generators, np.bitwise_xor.reduce(generators, axis=1)])
    )
    basis_b = binary.get_independent_rows(
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
            [(basis_a, None), (basis_b, None)], cutoff=0, block_size=2
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


def test_brouwer_zimmermann_exhaustive_dispatch() -> None:
    """Dependent logical rows skip the top-level independent-row exhaustive path.

    The exhaustive dispatcher uses the reduced quotient basis instead.
    """
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
    assert 1000 * get_candidate_count() < 2 ** len(generators)


def test_brouwer_zimmermann_initial_upper_bound_excludes_stabilizers() -> None:
    """The initial upper bound comes from eligible rows, not lighter stabilizers."""
    basis = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
    labels = np.array([[0], [1]], dtype=np.uint8)
    assert bz._get_brouwer_zimmermann_initial_upper_bound(basis, labels) == 2


def test_brouwer_zimmermann_overlap_qc(get_candidate_count: Callable[[], int]) -> None:
    """The default QC64 X-sector search stays within a broad candidate-work budget."""
    code = qldpc.codes.QCCode(
        {x: 4, y: 4, z: 2},
        1 + x + y + x * y * z,
        1 + x * y + x**2 * y**3 * z + x**3 * y**2,
    )
    assert code.forget_distance().get_distance_exact("X", cutoff=0) == 8
    assert get_candidate_count() < 1_000_000


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_overlap_improves_quotient_witness() -> None:
    """Compare witness-improving overlap enumeration with an exhaustive quotient oracle."""

    # Shorten a systematic BCH(31,21) basis to [26,16] and isolate a one-bit quotient.
    basis = binary.get_independent_rows(
        np.asarray(qldpc.codes.BCHCode(31, 21).generator, dtype=np.uint8)
    )[5:, 5:]
    logicals = basis[13:14]
    stabilizers = np.delete(basis, 13, axis=0)
    expected = qldpc.codes.get_distance_quantum(
        logicals, stabilizers, homogeneous=True, cutoff=0, method="brute_force"
    )

    # Lower only the work threshold so this small oracle exercises production basis family
    # construction.
    original = _information_sets.get_overlapping_family

    def select(*args: Any) -> _information_sets.BasisFamily:
        basis, labels, basis_family, upper_bound, max_sets, _, binomials, rng = args
        selected = original(basis, labels, basis_family, upper_bound, max_sets, 0, binomials, rng)
        assert len(selected.ranks) and selected.max_coverage > 1
        assert (
            min(
                bz._get_brouwer_zimmermann_initial_upper_bound(generators, labels)
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
