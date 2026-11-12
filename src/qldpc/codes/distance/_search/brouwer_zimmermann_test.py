# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact binary Brouwer-Zimmermann distance searches."""

from __future__ import annotations

import dataclasses
import functools
import math
from collections.abc import Callable, Iterator, Sequence
from typing import Any
from unittest import mock

import numpy as np
import numpy.typing as npt
import pytest
from sympy.abc import x, y, z

import qldpc
from qldpc._binary import get_independent_rows, row_reduce_binary
from qldpc.objects import PAULIS_XZ, Pauli

from ...conftest import Tune
from .._tuning import DEFAULT_TUNING, MAX_VALUE, SearchTuning, use_tuning
from ..conftest import CostlyStages, get_random_full_rank_matrix, get_sector_problem
from . import affine_instance, affine_search, completion_bounds, information_sets
from . import brouwer_zimmermann as bz

# A tuning that never uses completion tables, for searches built directly.
TABLES_DISABLED = dataclasses.replace(
    DEFAULT_TUNING,
    completion_bound_minimum_layer_work=MAX_VALUE,
    completion_bound_max_cost_percent=0,
    completion_bound_max_table_bytes=0,
)


def test_brouwer_zimmermann_search_state() -> None:
    """A search state handles partial-rank information sets.

    The search state handles quotient labels, cutoffs, and lower-bound updates.
    """
    # The input is one rank-zero information set with two basis rows.  The two basis rows have
    # Hamming weights 2 and 1, both outside the excluded subcode.
    search = bz._BrouwerZimmermannSearch(
        affine_instance.AffineInstance(
            generators=np.array([[[3], [2]]], dtype=np.uint64),
            labels=np.ones((1, 2, 1), dtype=np.uint64),
            widths=np.array([1], dtype=np.int64),
            pivots=np.full((1, 2), -1, dtype=np.int64),
            certificate=information_sets.Certificate(2, (0,)),
        ),
        cutoff=1,
        batch_size=1,
        tuning=TABLES_DISABLED,
        best=3,
        lower_bound=0,
    )

    # An interrupted coefficient-weight layer leaves the lower-bound certificate unchanged.
    with (
        mock.patch.object(affine_search, "scan_family", side_effect=KeyboardInterrupt),
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


def test_the_largest_layer_threshold_never_asks_for_automorphisms() -> None:
    """The work of a layer saturates at the cap of work estimates, below the largest threshold.

    The next layer of this search chooses 35 of the 70 rows of its information set, which costs
    more than ``MAX_VALUE`` candidate/set evaluations, so only a smaller threshold asks for
    automorphisms.
    """
    dimension, weight = 70, 35
    assert math.comb(dimension, weight) > MAX_VALUE
    for minimum_layer_work, expected_calls in ((MAX_VALUE, 0), (2**61, 1)):
        get_automorphism_family = mock.Mock(return_value=None)
        search = bz._BrouwerZimmermannSearch(
            affine_instance.AffineInstance(
                generators=np.zeros((1, dimension, 2), dtype=np.uint64),
                labels=np.zeros((1, dimension, 1), dtype=np.uint64),
                widths=np.array([2], dtype=np.int64),
                pivots=np.arange(dimension, dtype=np.int64)[None],
                certificate=information_sets.Certificate(dimension, (dimension,)),
            ),
            cutoff=0,
            batch_size=1,
            tuning=dataclasses.replace(
                TABLES_DISABLED, automorphism_minimum_layer_work=minimum_layer_work
            ),
            best=dimension,
            lower_bound=0,
            next_weight=weight,
            get_automorphism_family=get_automorphism_family,
        )
        # Skip the scan of the layer itself, which no search could finish.
        with (
            mock.patch.object(affine_search, "scan_family", return_value=(dimension, -1, 0)),
            mock.patch.object(
                affine_search, "scan_information_set", return_value=(dimension, -1, 0, 0, 0)
            ),
        ):
            search.advance()
        assert get_automorphism_family.call_count == expected_calls


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
    tune: Tune,
) -> None:
    """Early exits from search setup or a shared search agree with a brute-force oracle."""
    expected = qldpc.codes.get_distance_quantum(
        logicals, stabilizers, homogeneous=True, cutoff=0, method="brute_force"
    )

    # Forbid only enumeration by size, so that these small inputs reach the information-set search,
    # and may still enumerate if that costs less than the search of their information sets.
    tune(allow_enumeration_by_size=False)
    if overlap:
        tune(overlapping_minimum_work=0)
    distance = qldpc.codes.get_distance_quantum(
        logicals, stabilizers, homogeneous=True, cutoff=cutoff
    )
    assert expected <= distance <= max(cutoff, expected)


def test_enumeration_by_work_can_be_forbidden(monkeypatch: pytest.MonkeyPatch, tune: Tune) -> None:
    """The "brute-force" case enumerates its quotient after setup only if the tuning allows it."""
    case = next(case for case in SETUP_EXIT_CASES if case.id == "brute-force")
    values: Sequence[Any] = case.values
    logicals, stabilizers = values[:2]
    expected = qldpc.codes.get_distance_quantum(
        logicals, stabilizers, homogeneous=True, method="brute_force"
    )
    enumerations = []
    original = bz.get_nested_distance_brute_force

    def enumerate_words(*args: Any, **kwargs: Any) -> int:
        enumerations.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(bz, "get_nested_distance_brute_force", enumerate_words)
    tune(allow_enumeration_by_size=False)
    for allowed in (True, False):
        enumerations.clear()
        with use_tuning(allow_enumeration_by_work=allowed):
            distance = qldpc.codes.get_distance_quantum(logicals, stabilizers, homogeneous=True)
        assert distance == expected
        assert bool(enumerations) == allowed


def test_brouwer_zimmermann_setup_cutoffs() -> None:
    """Preparation can reach the cutoff from initial rows or a newly systematic row."""
    basis = np.array([[1, 0, 1, 1], [0, 1, 1, 1]], dtype=np.uint8)
    assert (
        bz._prepare_brouwer_zimmermann_search(
            basis,
            _no_labels(basis),
            cutoff=3,
            weight_divisor=1,
            upper_bound=4,
            tuning=DEFAULT_TUNING,
        )
        == 3
    )
    assert (
        bz._prepare_brouwer_zimmermann_search(
            basis,
            _no_labels(basis),
            cutoff=2,
            weight_divisor=1,
            upper_bound=4,
            tuning=SearchTuning(allow_enumeration_by_size=False),
        )
        == 2
    )


def test_an_upper_bound_acts_as_a_known_witness() -> None:
    """A search returns the smaller of its upper bound and the distance."""
    basis = get_independent_rows(np.asarray(qldpc.codes.HammingCode(3).generator, dtype=np.uint8))
    for upper_bound, expected in ((2, 2), (3, 3), (5, 3)):
        distance = bz.get_distance_brouwer_zimmermann(
            basis, _no_labels(basis), cutoff=0, upper_bound=upper_bound
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
    expected = bz.get_distance_brouwer_zimmermann_many([(basis, labels)], cutoff=0)
    get_basis_family = mock.Mock(
        return_value=information_sets.get_disjoint_family(
            basis, labels, random_column_orders=DEFAULT_TUNING.disjoint_random_column_orders
        )
    )
    search = functools.partial(
        bz.get_distance_brouwer_zimmermann_many,
        [(basis, labels)],
        get_disjoint_basis_families=[get_basis_family],
    )
    with mock.patch.object(
        information_sets, "get_disjoint_family", wraps=information_sets.get_disjoint_family
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
            [(basis_a, _no_labels(basis_a)), (basis_b, _no_labels(basis_b))], cutoff=0
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


def test_brouwer_zimmermann_overlap_mb(get_candidate_count: Callable[[], int]) -> None:
    """The default MB64 X-sector search stays within a broad candidate-work budget."""
    code = qldpc.codes.MBCode(
        {x: 4, y: 4, z: 2},
        1 + x + y + x * y * z,
        1 + x * y + x**2 * y**3 * z + x**3 * y**2,
    )
    assert code.forget_distance().get_distance_exact("X", cutoff=0) == 8
    assert 0 < get_candidate_count() < 1_000_000


@pytest.mark.usefixtures("force_information_sets")
def test_brouwer_zimmermann_overlap_improves_quotient_witness(tune: Tune) -> None:
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
    tune(overlapping_minimum_work=0)
    original = information_sets.get_overlapping_family

    def select(*args: Any, **kwargs: Any) -> information_sets.BasisFamily:
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
        information_sets, "get_overlapping_family", side_effect=select
    ) as select_mock:
        assert (
            qldpc.codes.get_distance_quantum(logicals, stabilizers, homogeneous=True, cutoff=0)
            == expected
            == 5
        )
    select_mock.assert_called()


@pytest.fixture
def get_candidate_count(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Count row-weight evaluations, scanned supports, and lookups across BZ and exhaustive routes.

    A scan of one information set counts each support that it weighs and each bound that it looks
    up, but not the supports that it skips.
    """
    num_candidates = 0
    original_scan = affine_search.scan_family
    original_one_set_scan = affine_search.scan_information_set

    def counting_scan(*args: Any) -> tuple[int, int, int]:
        nonlocal num_candidates
        best, depth, visited = original_scan(*args)
        num_candidates += visited * len(args[2])
        return best, depth, visited

    def counting_one_set_scan(*args: Any) -> tuple[int, int, int, int, int]:
        nonlocal num_candidates
        result = original_one_set_scan(*args)
        num_candidates += result[2] + result[3]
        return result

    bitwise_count = np.bitwise_count

    def counting_bitwise_count(
        arr: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64] | None = None
    ) -> npt.NDArray[np.uint64]:
        nonlocal num_candidates
        num_candidates += len(arr)
        return bitwise_count(arr, out=out)

    # Row weights and exhaustive enumeration count every packed word with np.bitwise_count.
    monkeypatch.setattr(np, "bitwise_count", counting_bitwise_count)
    monkeypatch.setattr(affine_search, "scan_family", counting_scan)
    monkeypatch.setattr(affine_search, "scan_information_set", counting_one_set_scan)
    return lambda: num_candidates


@pytest.mark.usefixtures("force_information_sets", "completion_tables")
def test_a_search_scans_every_set_that_its_certificate_counts() -> None:
    """A [22, 7, 5] code whose search finds a word of weight five only in its second set.

    Its two disjoint information sets reach the certificate six once both complete layer two, and
    the first word of weight five is in layer two of the second set.  A search that counted the
    second set's layer without scanning it would return six.  With tables in every layer, the
    search scans the sets of layer two one at a time.
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
    tune: Tune, costly_stages: CostlyStages
) -> None:
    """A [21, 8, 6] code, searched with a partitioned family and a certificate of linear programs.

    With the thresholds of the costly stages lowered, the search of this code replaces its family by
    information sets from matroid partitioning, whose columns lie in up to two of them, and bounds
    unvisited words by linear programming.
    """
    tune(overlapping_minimum_work=0, lp_minimum_work=0, partition_work_ratio=0)
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
    assert any(
        len(family.ranks) and family.max_coverage > 1
        for family in costly_stages.partitioned_families
    )
    assert any(certificate.pivot_blocks for certificate in costly_stages.search_certificates)


@pytest.fixture
def switch_before_any_layer(tune: Tune) -> None:
    """Let searches consider automorphism families before their first layer, however cheap it is."""
    tune(automorphism_minimum_layer_work=0)


@dataclasses.dataclass
class CompletionRecord:
    """Whether searches had to use completion tables, the tables that they chose, and their work.

    Each entry of ``choices`` holds the tables that a search held before a layer and the tables that
    it chose for the layer, or None.
    """

    forced: bool
    choices: list[tuple[completion_bounds.CompletionTables | None, ...]]
    lookups: int = 0
    skipped: int = 0


@pytest.fixture(params=["default", "forced"])
def completion_tables(
    request: pytest.FixtureRequest, tune: Tune, monkeypatch: pytest.MonkeyPatch
) -> Iterator[CompletionRecord]:
    """Run a test with the default choice of completion tables, or with tables in every layer.

    The tables of every layer have eight-column blocks.

    Record the tables that searches choose and the bounds that their scans look up, and check that a
    forced search used tables.
    """
    record = CompletionRecord(request.param == "forced", [])
    if record.forced:
        # Eight-column blocks, rather than the default sixteen, keep the tables quick to build as
        # plain Python.
        tune(
            completion_bound_minimum_layer_work=0,
            completion_bound_max_cost_percent=MAX_VALUE,
            completion_bound_block_bits=8,
        )
    select = completion_bounds.select_completion_tables
    scan = affine_search.scan_information_set

    def recording_select(*args: Any, **kwargs: Any) -> completion_bounds.CompletionTables | None:
        tables = select(*args, **kwargs)
        record.choices.append((kwargs["current"], tables))
        return tables

    def counting_scan(*args: Any) -> tuple[int, int, int, int, int]:
        result = scan(*args)
        record.lookups += result[3]
        record.skipped += result[4]
        return result

    monkeypatch.setattr(completion_bounds, "select_completion_tables", recording_select)
    monkeypatch.setattr(affine_search, "scan_information_set", counting_scan)
    yield record
    if record.forced:
        assert record.lookups


@pytest.mark.usefixtures("force_information_sets", "switch_before_any_layer")
def test_automorphism_family_searches_scan_every_layer_they_certify(
    completion_tables: CompletionRecord,
) -> None:
    """A search with an automorphism family that reaches the distance only in layer four, its last.

    This [[72, 4, 8]] bivariate bicycle code comes from a search for codes like those of
    arXiv:2308.07915.  In its X sector, the lightest rows of the scanned set weigh ten, and the
    first word of weight eight that the search finds is a sum of four of them.  Completing layer
    four also certifies that no unexamined word weighs less than ten, so a search that certified
    layer four before scanning it would return ten.
    """
    code = qldpc.codes.BBCode({x: 12, y: 3}, 1 + x + x**4 * y, 1 + y + x**2)
    problem = get_sector_problem(code, Pauli.X)
    expected = bz.get_distance_brouwer_zimmermann_many([problem], cutoff=0)
    assert expected == 8
    distance = bz.get_distance_brouwer_zimmermann_many(
        [problem], cutoff=0, get_automorphisms=code._get_qudit_automorphisms
    )
    assert distance == expected

    # With tables in every layer, layers after the first that uses them reuse the same tables.
    if completion_tables.forced:
        assert any(
            held is not None and tables is held for held, tables in completion_tables.choices
        )


@pytest.mark.usefixtures("force_information_sets", "switch_before_any_layer", "completion_tables")
def test_automorphism_family_searches_find_exact_distances() -> None:
    """Searches given automorphisms agree with brute force, alone or sharing witnesses.

    The searches use the translations of a multivariate bicycle code, the identity alone, which does
    not reduce work, or a function that returns no automorphisms.
    """
    code = qldpc.codes.MBCode({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    identity = np.arange(len(code))[None]
    problems = [get_sector_problem(code, pauli) for pauli in PAULIS_XZ]
    for problem, pauli in zip(problems, PAULIS_XZ):
        expected = qldpc.codes.get_distance_quantum(
            code.get_logical_ops(pauli),
            code.get_stabilizer_ops(pauli),
            homogeneous=True,
            cutoff=0,
            method="brute_force",
        )
        for get_automorphisms in (code._get_qudit_automorphisms, lambda: identity, lambda: None):
            distance = bz.get_distance_brouwer_zimmermann_many(
                [problem], cutoff=0, get_automorphisms=get_automorphisms
            )
            assert distance == expected
    assert bz.get_distance_brouwer_zimmermann_many(
        problems, cutoff=0, get_automorphisms=code._get_qudit_automorphisms
    ) == code.get_distance(method="brute_force")


@pytest.mark.usefixtures("force_information_sets")
def test_automorphism_family_searches_scan_their_new_set_from_layer_two(
    tune: Tune, completion_tables: CompletionRecord
) -> None:
    """A search that switches to an automorphism family before layer three rescans from layer two.

    This [16, 8, 4] quasi-cyclic code has the generator matrix ``[I | circ(1 + x + x^3 + x^4)]``,
    and shifting both blocks at once preserves it.  The search starts with one information set,
    the first block, which holds four ones of each word of weight four, so its first two layers
    miss them.  Before layer three it switches to a set with half of its columns in each block,
    which holds two ones of every shift of these words.  A search that continued the new set
    from layer three would certify that no word is lighter than its best witness of weight five.
    """
    size, word = 8, [1, 1, 0, 1, 1, 0, 0, 0]
    basis = np.hstack(
        [np.eye(size, dtype=np.uint8), np.array([np.roll(word, i) for i in range(size)], np.uint8)]
    )
    labels = _no_labels(basis)
    shift = np.roll(np.arange(size), -1)
    automorphisms = np.concatenate([shift, shift + size])[None]
    first_family = information_sets.BasisFamily(
        basis[None],
        labels[None],
        np.arange(size)[None],
        np.array([size]),
        information_sets.infer_weight_divisor(basis),
        1,
    )
    tune(automorphism_minimum_layer_work=math.comb(size, 3))
    distance = bz.get_distance_brouwer_zimmermann_many(
        [(basis, labels)],
        cutoff=0,
        get_disjoint_basis_families=[lambda: first_family],
        get_automorphisms=lambda: automorphisms,
    )
    assert distance == qldpc.codes.get_distance_classical(basis, method="brute_force") == 4

    # With tables in every layer, the search builds tables for its first set, and again for the
    # set of its automorphism family, after it drops the tables of the first set.
    if completion_tables.forced:
        built = [(held, tables) for held, tables in completion_tables.choices if tables is not held]
        assert len(built) == 2 and built[1][0] is None


@pytest.mark.usefixtures("force_information_sets")
def test_searches_ask_for_automorphisms_only_before_costly_layers() -> None:
    """A search asks for automorphisms once, and only before a layer that costs enough to repay.

    Finding automorphisms can cost more than an easy search, so a search that finishes before any
    costly layer does not ask for them.
    """
    code = qldpc.codes.MBCode({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    problem = get_sector_problem(code, Pauli.X)
    for minimum_layer_work, expected_calls in ((10**9, 0), (0, 1)):
        get_automorphisms = mock.Mock(wraps=code._get_qudit_automorphisms)
        with use_tuning(automorphism_minimum_layer_work=minimum_layer_work):
            bz.get_distance_brouwer_zimmermann_many(
                [problem], cutoff=0, get_automorphisms=get_automorphisms
            )
        assert get_automorphisms.call_count == expected_calls


@pytest.mark.usefixtures("force_information_sets", "switch_before_any_layer")
def test_rows_of_an_automorphism_family_can_reach_the_cutoff() -> None:
    """The rows of a scanned set with images can end a search at the cutoff.

    In the Z sector of this [[36, 4, 4]] bivariate bicycle code, the rows of the first family weigh
    more than six, and a row of the scanned set weighs six.  With a cutoff of six, the search
    returns that weight, an upper bound on the distance, rather than scanning on to a word of
    weight four.
    """
    code = qldpc.codes.BBCode({x: 6, y: 3}, 1 + x + x**3 * y, 1 + y + x)
    problem = get_sector_problem(code, Pauli.Z)
    assert bz.get_distance_brouwer_zimmermann_many([problem], cutoff=0) == 4
    distance = bz.get_distance_brouwer_zimmermann_many(
        [problem], cutoff=6, get_automorphisms=code._get_qudit_automorphisms
    )
    assert distance == 6


@pytest.mark.usefixtures("force_information_sets", "switch_before_any_layer")
def test_searches_reject_wrong_automorphisms() -> None:
    """A permutation that does not preserve a problem fails loudly when a search would use it."""
    code = qldpc.codes.MBCode({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    transposition = np.arange(len(code))
    transposition[[0, 1]] = [1, 0]
    with pytest.raises(ValueError, match="not an automorphism"):
        bz.get_distance_brouwer_zimmermann_many(
            [get_sector_problem(code, Pauli.X)],
            cutoff=0,
            get_automorphisms=lambda: transposition[None],
        )


def _no_labels(basis: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Quotient labels with no columns, which exclude no subcode."""
    return np.empty((len(basis), 0), dtype=np.uint8)


def _get_family(
    generators: npt.NDArray[np.uint8], column_sets: list[list[int]]
) -> tuple[information_sets.BasisFamily, information_sets.Certificate]:
    """A basis family without labels, of generators row reduced on each set of columns.

    Returns the family and its certificate, with the ranks that the row reduction finds.
    """
    num_rows = len(generators)
    reduced, pivots = [], np.full((len(column_sets), num_rows), -1, dtype=np.int64)
    for index, columns in enumerate(column_sets):
        rows, _, found = row_reduce_binary(generators, None, columns=np.array(columns))
        reduced.append(rows)
        pivots[index, : len(found)] = found
    ranks = np.count_nonzero(pivots >= 0, axis=1)
    family = information_sets.BasisFamily(
        np.array(reduced),
        np.empty((len(column_sets), num_rows, 0), dtype=np.uint8),
        pivots,
        ranks,
        1,
        1,
    )
    return family, information_sets.Certificate(num_rows, tuple(ranks.tolist()))


@pytest.mark.usefixtures("force_information_sets", "switch_before_any_layer")
@pytest.mark.parametrize("completion_tables", ["forced"], indirect=True)
def test_completion_tables_keep_distances_exact(
    pytestconfig: pytest.Config, completion_tables: CompletionRecord
) -> None:
    """Searches that skip hopeless completions agree with independent oracles.

    Every layer uses tables, and every search below skips supports.  In each sector of this
    [[36, 4, 4]] bivariate bicycle code, the rows of the scanned set weigh at least six, so a scan
    must find a word of weight four, with or without an upper bound above it.  In the X sector of a
    [[72, 4, 8]] code, with its coordinates relabelled at random and its automorphisms relabelled
    alike, the search must scan to layer four.  A general Pauli problem checks that searched words
    weigh twice their number of nontrivial qubits.
    """

    def assert_skips(distance: int) -> int:
        assert completion_tables.skipped > skipped
        return distance

    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    code = qldpc.codes.BBCode({x: 6, y: 3}, 1 + x + x**3 * y, 1 + y + x)
    for pauli in PAULIS_XZ:
        expected = qldpc.codes.get_distance_quantum(
            code.get_logical_ops(pauli),
            code.get_stabilizer_ops(pauli),
            homogeneous=True,
            method="brute_force",
        )
        for upper_bound in (None, expected + 1):
            skipped = completion_tables.skipped
            # One sixteen-column block covers the tails of this code, which makes its bounds exact,
            # so a bound that is too high changes the distance found.
            with use_tuning(completion_bound_block_bits=16):
                distance = bz.get_distance_brouwer_zimmermann_many(
                    [get_sector_problem(code, pauli)],
                    cutoff=0,
                    upper_bound=upper_bound,
                    get_automorphisms=code._get_qudit_automorphisms,
                )
            assert assert_skips(distance) == expected

    # Column j of the relabelled problem is column columns[j] of the given one, so an automorphism
    # that maps column c to g[c] maps column j to inverse[g[columns[j]]].  The search of the given
    # problem without automorphisms, which uses no tables, is the oracle.
    code = qldpc.codes.BBCode({x: 12, y: 3}, 1 + x + x**4 * y, 1 + y + x**2)
    basis, labels = get_sector_problem(code, Pauli.X)
    with use_tuning(completion_bound_max_table_bytes=0):
        expected = bz.get_distance_brouwer_zimmermann_many([(basis, labels)], cutoff=0)
    columns = rng.permutation(len(code))
    automorphisms = np.asarray(code._get_qudit_automorphisms())
    skipped = completion_tables.skipped
    distance = bz.get_distance_brouwer_zimmermann_many(
        [(basis[:, columns], labels)],
        cutoff=0,
        get_automorphisms=functools.partial(
            np.array, np.argsort(columns)[automorphisms[:, columns]]
        ),
    )
    assert assert_skips(distance) == expected == 8

    five_qubit_code = qldpc.codes.FiveQubitCode()
    logicals, stabilizers = five_qubit_code.get_logical_ops(), five_qubit_code.get_stabilizer_ops()
    skipped = completion_tables.skipped
    distance = qldpc.codes.get_distance_quantum(
        logicals, stabilizers, homogeneous=False, cutoff=0, method="brouwer_zimmermann"
    )
    assert assert_skips(distance) == qldpc.codes.get_distance_quantum(
        logicals, stabilizers, homogeneous=False, method="brute_force"
    )


def test_sets_that_are_not_information_sets_use_the_general_scan() -> None:
    """A set of rank zero has no pivot ones, so its words weigh only their packed rows.

    Its two rows weigh three and two, and their sum weighs one.  A scan that counted a pivot one for
    each row, as the scan of one information set does, would find no word lighter than two, even
    with completion tables in every layer.
    """
    search = bz._BrouwerZimmermannSearch(
        affine_instance.AffineInstance(
            generators=np.array([[[7], [6]]], dtype=np.uint64),
            labels=np.array([[[1], [2]]], dtype=np.uint64),
            widths=np.array([1], dtype=np.int64),
            pivots=np.full((1, 2), -1, dtype=np.int64),
            certificate=information_sets.Certificate(2, (0,)),
        ),
        cutoff=-1,
        batch_size=1,
        tuning=dataclasses.replace(
            DEFAULT_TUNING,
            automorphism_minimum_layer_work=MAX_VALUE,
            completion_bound_minimum_layer_work=0,
            completion_bound_max_cost_percent=MAX_VALUE,
            completion_bound_max_table_bytes=MAX_VALUE,
        ),
        best=10,
        lower_bound=0,
    )
    assert search.advance() == 2
    assert search.advance() == 1


@pytest.mark.usefixtures("force_information_sets", "switch_before_any_layer")
@pytest.mark.parametrize("completion_tables", ["forced"], indirect=True)
def test_joint_searches_with_completion_tables(
    completion_tables: CompletionRecord, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Joint searches that skip supports in every layer agree with brute force, and free tables.

    The searches of both sectors of this [[30, 2, 6]] code share their best weight, with or without
    an upper bound of five, and each skips supports.  The tables of a search bound the memory that
    it keeps, so each search frees them once it has finished.
    """
    searches: list[bz._BrouwerZimmermannSearch] = []
    skipped: list[int] = []
    advance = bz._BrouwerZimmermannSearch.advance

    def counting_advance(search: bz._BrouwerZimmermannSearch) -> int:
        if not any(known is search for known in searches):
            searches.append(search)
            skipped.append(0)
        skipped_before = completion_tables.skipped
        best = advance(search)
        index = [id(known) for known in searches].index(id(search))
        skipped[index] += completion_tables.skipped - skipped_before
        return best

    monkeypatch.setattr(bz._BrouwerZimmermannSearch, "advance", counting_advance)
    code = qldpc.codes.MBCode({x: 15}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    problems = [get_sector_problem(code, pauli) for pauli in PAULIS_XZ]
    for upper_bound, expected in ((None, code.get_distance(method="brute_force")), (5, 5)):
        searches.clear()
        skipped.clear()
        distance = bz.get_distance_brouwer_zimmermann_many(
            problems,
            cutoff=0,
            upper_bound=upper_bound,
            get_automorphisms=code._get_qudit_automorphisms,
        )
        assert distance == expected
        assert len(searches) == 2 and all(skipped)
        assert all(search.completion_tables is None for search in searches)


@pytest.mark.usefixtures("force_information_sets")
@pytest.mark.parametrize("completion_tables", ["forced"], indirect=True)
@pytest.mark.parametrize("mixed", [False, True])
def test_layers_of_several_sets_scan_each_set_with_its_tables(
    completion_tables: CompletionRecord, monkeypatch: pytest.MonkeyPatch, mixed: bool
) -> None:
    """Searches that scan the sets of a layer one at a time, each with tables or none, are exact.

    This random code with 10 rows and 40 columns has four disjoint information sets, so a search
    must scan several of them in some layers.  The oracle enumerates every nonzero word.  The
    tables of each scan must be those of the rows that it scans, since tables of other rows could
    skip lighter words.  The forced tuning gives every set tables, unless the test takes the tables
    of the first set of each such layer away, which every such layer scans.
    """
    choose = bz._BrouwerZimmermannSearch._choose_tables_of_sets
    scan_in_turn = bz._BrouwerZimmermannSearch._scan_sets_in_turn
    scan = affine_search.scan_information_set
    scans_in_turn = {"with tables": 0, "without tables": 0}
    in_turn = False

    def mixing_choose(search: bz._BrouwerZimmermannSearch, *args: Any) -> list[int] | None:
        block_bits = choose(search, *args)
        return [0, *block_bits[1:]] if mixed and block_bits is not None else block_bits

    def recording_scan_in_turn(search: bz._BrouwerZimmermannSearch, *args: Any) -> int | None:
        nonlocal in_turn
        in_turn = True
        try:
            return scan_in_turn(search, *args)
        finally:
            in_turn = False

    def checking_scan(*args: Any) -> tuple[int, int, int, int, int]:
        rows, widths, tables, block_bits = args[0], args[2], args[10], args[12]
        if in_turn:
            scans_in_turn["with tables" if len(tables) else "without tables"] += 1
        if len(tables):
            own = completion_bounds.CompletionTables.build(rows[0, :, : widths[0]], block_bits)
            assert np.array_equal(tables, own.tables)
        return scan(*args)

    monkeypatch.setattr(bz._BrouwerZimmermannSearch, "_choose_tables_of_sets", mixing_choose)
    monkeypatch.setattr(bz._BrouwerZimmermannSearch, "_scan_sets_in_turn", recording_scan_in_turn)
    monkeypatch.setattr(affine_search, "scan_information_set", checking_scan)
    generators = get_random_full_rank_matrix(np.random.default_rng(0), 10, 40)
    distance = bz.get_distance_brouwer_zimmermann_many(
        [(generators, _no_labels(generators))], cutoff=0
    )
    assert distance == qldpc.codes.get_distance_classical(
        generators, cutoff=0, method="brute_force"
    )
    assert completion_tables.skipped
    assert scans_in_turn["with tables"]
    assert bool(scans_in_turn["without tables"]) == mixed


@pytest.mark.parametrize(("forced", "cutoff"), [(True, 0), (True, 5), (False, 0)])
def test_a_layer_of_several_sets_stops_once_its_sets_certify_the_best_weight(
    forced: bool, cutoff: int
) -> None:
    """A search that scans sets one at a time stops after the set whose completion certifies.

    The two sets of this code are disjoint information sets, the first three columns and the next
    three, and their rows weigh six, seven, and six, and six, nine, and six.  After both sets
    complete layer one, the search certifies weight four.  In layer two, it needs both sets to
    certify its best weight of six, but the first set finds a word of weight five, the sum of the
    first two rows.  Its completion of the layer then certifies weight five, so a search that scans
    the sets one at a time with tables scans no other set.  Without tables, it scans both sets in
    lockstep, and certifies six.  With a cutoff of five, a scan of the sets one at a time stops at
    the word of weight five, before it certifies more than weight four.
    """
    generators = np.array(
        [
            [1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1],
            [0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1],
            [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    family, certificate = _get_family(generators, [[0, 1, 2], [3, 4, 5]])
    tuning = dataclasses.replace(
        DEFAULT_TUNING,
        completion_bound_minimum_layer_work=0 if forced else MAX_VALUE,
        completion_bound_max_cost_percent=MAX_VALUE,
    )
    search = bz._BrouwerZimmermannSearch(
        affine_instance.get_distance_instance(family, certificate),
        cutoff=cutoff,
        batch_size=1 << 20,
        tuning=tuning,
        best=6,
        lower_bound=certificate.lower_bound(1),
        next_weight=2,
    )
    scanned: list[int] = []
    scan_one, scan_family = affine_search.scan_information_set, affine_search.scan_family

    def recording_scan_one(*args: Any) -> tuple[int, int, int, int, int]:
        scanned.append(len(args[0]))
        return scan_one(*args)

    def recording_scan_family(*args: Any) -> tuple[int, int, int]:
        scanned.append(len(args[0]))
        return scan_family(*args)

    with (
        mock.patch.object(affine_search, "scan_information_set", recording_scan_one),
        mock.patch.object(affine_search, "scan_family", recording_scan_family),
    ):
        assert search.advance() == 5
    assert search.finished and search.lower_bound == (4 if cutoff else 5 if forced else 6)
    assert scanned == ([1] if forced else [2])


def test_layers_scan_several_sets_one_at_a_time_only_if_that_pays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A layer scans several sets one at a time if tables pay for some, and for the whole layer.

    The predicted costs include building the tables, which a layer of several sets builds anew.  A
    layer of two sets averages their predicted costs, at the full cost for a set without tables,
    and compares the average with the default limit of 90 percent.  A set whose tables cost half
    as much as a scan without them takes them, and the average of 0.75 is within the limit.  A set
    whose tables would cost 95 percent scans without them.  With tables that cost 85 percent for
    one set and none for the other, the average is 0.925, so the layer scans in lockstep.
    """
    costs: list[float] = []

    def predict(*args: Any, **kwargs: Any) -> tuple[float, int]:
        assert not kwargs["reuse"]
        cost = costs.pop(0)
        return cost, 0 if cost == math.inf else 16

    monkeypatch.setattr(completion_bounds, "predict_completion_cost", predict)
    search = bz._BrouwerZimmermannSearch(
        affine_instance.AffineInstance(
            generators=np.zeros((2, 3, 1), dtype=np.uint64),
            labels=np.zeros((2, 3, 0), dtype=np.uint64),
            widths=np.ones(2, dtype=np.int64),
            pivots=np.zeros((2, 3), dtype=np.int64),
            certificate=information_sets.Certificate(3, (3, 3)),
        ),
        cutoff=0,
        batch_size=1,
        tuning=DEFAULT_TUNING,
        best=6,
        lower_bound=0,
    )
    for set_costs, expected in (
        ([0.5, math.inf], [16, 0]),
        ([0.5, 0.95], [16, 0]),
        ([0.85, math.inf], None),
    ):
        costs[:] = set_costs
        assert search._choose_tables_of_sets(2, 2) == expected


def test_a_layer_with_a_set_that_is_not_an_information_set_scans_in_lockstep() -> None:
    """A layer of several sets scans them one at a time only if each is an information set.

    The second set of this code, on columns three to five, has rank two, so its third row has no
    pivot.  The lightest rows of both sets weigh five, and the lightest word weighs four.  A scan of
    one information set counts a pivot one for each row, so it would overstate the weights of the
    words of the second set, and the search, with tables in every layer, would certify weight five.
    """
    generators = np.array(
        [
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1],
            [0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    family, certificate = _get_family(generators, [[0, 1, 2], [3, 4, 5]])
    search = bz._BrouwerZimmermannSearch(
        affine_instance.get_distance_instance(family, certificate),
        cutoff=0,
        batch_size=1 << 20,
        tuning=dataclasses.replace(
            DEFAULT_TUNING,
            completion_bound_minimum_layer_work=0,
            completion_bound_max_cost_percent=MAX_VALUE,
        ),
        best=5,
        lower_bound=certificate.lower_bound(1),
        next_weight=2,
    )
    while not search.finished:
        search.advance()
    assert (
        search.best
        == qldpc.codes.get_distance_classical(generators, cutoff=0, method="brute_force")
        == 4
    )


@pytest.mark.usefixtures("force_information_sets", "switch_before_any_layer")
def test_searches_take_completion_limits_from_their_tuning(
    tune: Tune, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Searches use completion tables within the limits of their tuning.

    The scanned sets of both sectors of this [[36, 4, 4]] code have 20 rows and 16 tail columns, so
    tables of one sixteen-column block take 19 * 2**16 bytes, and tables of two eight-column blocks
    take 19 * 2 * 2**8 bytes.  A tuning that forces tables uses sixteen-column blocks, a smaller
    memory limit eight-column blocks, and a limit of zero bytes or of zero cost, or the largest
    threshold of layer work, no tables.
    """
    block_bits: set[int] = set()
    select = completion_bounds.select_completion_tables

    def recording_select(*args: Any, **kwargs: Any) -> completion_bounds.CompletionTables | None:
        tables = select(*args, **kwargs)
        if tables is not None:
            block_bits.add(tables.block_bits)
        return tables

    monkeypatch.setattr(completion_bounds, "select_completion_tables", recording_select)
    code = qldpc.codes.BBCode({x: 6, y: 3}, 1 + x + x**3 * y, 1 + y + x)
    problems = [get_sector_problem(code, pauli) for pauli in PAULIS_XZ]
    tune(completion_bound_minimum_layer_work=0, completion_bound_max_cost_percent=MAX_VALUE)
    for limits, expected_block_bits in (
        ({}, {16}),
        ({"completion_bound_max_table_bytes": 19 * 2 * 2**8}, {8}),
        ({"completion_bound_max_table_bytes": 0}, set()),
        ({"completion_bound_max_cost_percent": 0}, set()),
        ({"completion_bound_minimum_layer_work": MAX_VALUE}, set()),
    ):
        block_bits.clear()
        with use_tuning(**limits):
            assert bz.get_distance_brouwer_zimmermann_many(
                problems,
                cutoff=0,
                get_automorphisms=code._get_qudit_automorphisms,
            ) == code.get_distance(method="brute_force")
        assert block_bits == expected_block_bits
