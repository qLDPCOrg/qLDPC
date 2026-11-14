# SPDX-License-Identifier: Apache-2.0

"""Unit tests for exact binary Brouwer-Zimmermann distance searches."""

from __future__ import annotations

import dataclasses
import functools
import gc
import math
import weakref
from collections.abc import Callable, Iterator
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
from .. import _brute_force
from .._information_sets import basis_families, certificates, completion_bounds, matroid_partition
from .._tuning import DEFAULT_TUNING, MAX_VALUE, use_tuning
from ..conftest import CostlyStages, get_random_full_rank_matrix, get_sector_problem
from . import affine_instance, affine_search
from . import distance as bz

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
            certificate=certificates.Certificate(2, (0,)),
        ),
        cutoff=1,
        batch_size=1,
        tuning=TABLES_DISABLED,
        best=3,
        lower_bound=0,
    )

    # An interrupted coefficient-weight layer leaves the lower-bound certificate unchanged.
    with (
        mock.patch.object(affine_search, "scan_information_set", side_effect=KeyboardInterrupt),
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
                certificate=certificates.Certificate(dimension, (dimension,)),
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
        with mock.patch.object(
            affine_search, "scan_information_set", return_value=(dimension, -1, 0, 0, 0)
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
    expected = qldpc.codes.distance.get_distance_from_generators(
        logicals, stabilizers, cutoff=0, method="brute_force"
    )

    if overlap:
        tune(overlapping_minimum_work=0)
    distance = qldpc.codes.distance.get_distance_from_generators(
        logicals, stabilizers, cutoff=cutoff
    )
    assert expected <= distance <= max(cutoff, expected)


def test_searches_never_enumerate_every_word() -> None:
    """A search scans information sets, even where enumerating every word would cost less.

    The quotient of this logical row by four excluded rows has 16 eligible words, fewer than the
    estimated work of the search of its information sets.
    """
    logicals = [[1, 0, 0, 0, 1, 0, 1, 1, 0]]
    stabilizers = [
        [0, 1, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 1, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 0, 1],
    ]
    expected = qldpc.codes.distance.get_distance_from_generators(
        logicals, stabilizers, method="brute_force"
    )
    with mock.patch.object(
        _brute_force, "_get_distance_brute_force", wraps=_brute_force._get_distance_brute_force
    ) as enumerate_words:
        distance = qldpc.codes.distance.get_distance_from_generators(logicals, stabilizers)
    assert distance == expected
    enumerate_words.assert_not_called()


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
            tuning=DEFAULT_TUNING,
        )
        == 2
    )


def test_an_upper_bound_acts_as_a_known_witness() -> None:
    """A search returns the smaller of its upper bound and the distance."""
    basis = get_independent_rows(np.asarray(qldpc.codes.HammingCode(3).generator, dtype=np.uint8))
    for upper_bound, expected in ((2, 2), (3, 3), (5, 3)):
        distance = bz.get_distance_brouwer_zimmermann_many(
            [(basis, _no_labels(basis))], cutoff=0, upper_bound=upper_bound
        )
        assert distance == expected


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

    expected_a = qldpc.codes.distance.get_distance_from_generators(
        basis_a,
        cutoff=0,
        method="brute_force",
    )
    expected_b = qldpc.codes.distance.get_distance_from_generators(
        basis_b[2:],
        basis_b[:2],
        cutoff=0,
        method="brute_force",
    )
    actual = bz.get_distance_brouwer_zimmermann_many(
        [(basis_a, _no_labels(basis_a)), (basis_b, labels_b)],
        cutoff=0,
    )
    assert actual == min(expected_a, expected_b)


def test_a_given_disjoint_basis_family_replaces_the_first_stage() -> None:
    """A search that reaches the first stage takes its result from the given function instead.

    A search that exits before the first stage, here at the weight of a row, does not call it.
    """
    rng = np.random.default_rng(121)
    basis = get_random_full_rank_matrix(rng, 5, 9)
    labels = np.vstack([np.zeros((2, 3), dtype=np.uint8), np.eye(3, dtype=np.uint8)])
    expected = bz.get_distance_brouwer_zimmermann_many([(basis, labels)], cutoff=0)
    get_basis_family = mock.Mock(
        return_value=basis_families.get_disjoint_family(
            basis, labels, random_column_orders=DEFAULT_TUNING.disjoint_random_column_orders
        )
    )
    search = functools.partial(
        bz.get_distance_brouwer_zimmermann_many,
        [(basis, labels)],
        get_disjoint_basis_families=[get_basis_family],
    )
    with mock.patch.object(
        basis_families, "get_disjoint_family", wraps=basis_families.get_disjoint_family
    ) as first_stage:
        assert search(cutoff=0) == expected
        get_basis_family.assert_called_once_with()
        search(cutoff=basis.shape[1])
        get_basis_family.assert_called_once_with()
    first_stage.assert_not_called()


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
        qldpc.codes.distance.get_distance_from_generators(basis, cutoff=0, method="brute_force")
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


def test_brouwer_zimmermann_subsystem_code() -> None:
    """Information-set search treats the gauge operators of a subsystem code as trivial.

    The 3x3 Bacon-Shor code is a [[9,1,3]] subsystem code; see
    https://arxiv.org/abs/quant-ph/0506023.
    """
    code = qldpc.codes.QuditCode(qldpc.codes.BaconShorCode(3).matrix, is_subsystem_code=True)
    assert code.get_distance_exact() == 3


@pytest.mark.parametrize("length", [64, 65, 257])
def test_brouwer_zimmermann_packed_widths(length: int) -> None:
    """Information-set weights are exact across uint64 words and for weights above 255."""
    generators = np.ones((1, length), dtype=np.uint8)
    assert qldpc.codes.distance.get_distance_from_generators(generators, cutoff=0) == length


def test_dependent_logical_rows_give_the_distance_of_their_span() -> None:
    """Repeated logical rows are reduced to a basis of the quotient before any search."""
    distance = qldpc.codes.distance.get_distance_from_generators(
        [[1, 1, 1], [1, 1, 1]], [], cutoff=0
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
    assert qldpc.codes.distance.get_distance_from_generators(generators, cutoff=0) == (
        4 if extended else 3
    )
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


def test_brouwer_zimmermann_overlap_improves_quotient_witness(tune: Tune) -> None:
    """Compare witness-improving overlap enumeration with an exhaustive quotient oracle."""

    # Shorten a systematic BCH(31,21) basis to [26,16] and isolate a one-bit quotient.
    basis = get_independent_rows(np.asarray(qldpc.codes.BCHCode(31, 21).generator, dtype=np.uint8))[
        5:, 5:
    ]
    logicals = basis[13:14]
    stabilizers = np.delete(basis, 13, axis=0)
    expected = qldpc.codes.distance.get_distance_from_generators(
        logicals, stabilizers, cutoff=0, method="brute_force"
    )

    # Lower only the work threshold so this small oracle exercises production basis family
    # construction, and check that the overlapping family has no witness as light as the distance.
    tune(overlapping_minimum_work=0)
    original = basis_families.get_overlapping_family

    def select(*args: Any, **kwargs: Any) -> basis_families.BasisFamily:
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
        basis_families, "get_overlapping_family", side_effect=select
    ) as select_mock:
        assert (
            qldpc.codes.distance.get_distance_from_generators(logicals, stabilizers, cutoff=0)
            == expected
            == 5
        )
    select_mock.assert_called()


@pytest.fixture
def get_candidate_count(monkeypatch: pytest.MonkeyPatch) -> Callable[[], int]:
    """Count the row-weight evaluations, scanned supports, and lookups of BZ searches.

    A scan of a set counts each support that it weighs and each bound that it looks up, but not the
    supports that it skips.
    """
    num_candidates = 0
    original_scan = affine_search.scan_information_set

    def counting_scan(*args: Any) -> tuple[int, int, int, int, int]:
        nonlocal num_candidates
        result = original_scan(*args)
        num_candidates += result[2] + result[3]
        return result

    bitwise_count = np.bitwise_count

    def counting_bitwise_count(
        arr: npt.NDArray[np.uint64], out: npt.NDArray[np.uint64] | None = None
    ) -> npt.NDArray[np.uint64]:
        nonlocal num_candidates
        num_candidates += len(arr)
        return bitwise_count(arr, out=out)

    # Row weights count every packed word with np.bitwise_count.
    monkeypatch.setattr(np, "bitwise_count", counting_bitwise_count)
    monkeypatch.setattr(affine_search, "scan_information_set", counting_scan)
    return lambda: num_candidates


@pytest.mark.usefixtures("completion_tables")
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
    assert (
        qldpc.codes.distance.get_distance_from_generators(basis, cutoff=0, method="brute_force")
        == 5
    )
    assert qldpc.codes.distance.get_distance_from_generators(basis, cutoff=0) == 5


def test_a_search_through_the_costly_stages_finds_the_distance(
    tune: Tune, costly_stages: CostlyStages
) -> None:
    """A [21, 8, 6] code, searched with a partitioned family and a certificate of linear programs.

    With the thresholds of the costly stages lowered, the search of this code replaces its family by
    information sets from matroid partitioning, whose columns lie in up to two of them, and bounds
    unvisited words by linear programming.
    """
    tune(overlapping_minimum_work=0, lp_minimum_work=0, partition_work_ratio=0)
    basis = _get_costly_basis()
    assert (
        qldpc.codes.distance.get_distance_from_generators(basis, cutoff=0, method="brute_force")
        == 6
    )
    assert qldpc.codes.distance.get_distance_from_generators(basis, cutoff=0) == 6
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
        # plain Python, and the fewest samples that the cost model accepts keep its predictions
        # quick.
        tune(
            completion_bound_minimum_layer_work=0,
            completion_bound_max_cost_percent=MAX_VALUE,
            completion_bound_block_bits=8,
            completion_bound_samples=DEFAULT_TUNING.completion_bound_min_samples,
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


@pytest.mark.usefixtures("switch_before_any_layer")
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


@pytest.mark.usefixtures("switch_before_any_layer", "completion_tables")
def test_automorphism_family_searches_find_exact_distances() -> None:
    """Searches given automorphisms agree with brute force, alone or sharing witnesses.

    The searches use the translations of a multivariate bicycle code, the identity alone, which does
    not reduce work, or a function that returns no automorphisms.
    """
    code = qldpc.codes.MBCode({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    identity = np.arange(len(code))[None]
    problems = [get_sector_problem(code, pauli) for pauli in PAULIS_XZ]
    for problem, pauli in zip(problems, PAULIS_XZ):
        expected = qldpc.codes.distance.get_distance_from_generators(
            code.get_logical_ops(pauli),
            code.get_stabilizer_ops(pauli),
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
    first_family = basis_families.BasisFamily(
        basis[None],
        labels[None],
        np.arange(size)[None],
        np.array([size]),
        basis_families.infer_weight_divisor(basis),
        1,
    )
    tune(automorphism_minimum_layer_work=math.comb(size, 3))
    distance = bz.get_distance_brouwer_zimmermann_many(
        [(basis, labels)],
        cutoff=0,
        get_disjoint_basis_families=[lambda: first_family],
        get_automorphisms=lambda: automorphisms,
    )
    assert (
        distance
        == qldpc.codes.distance.get_distance_from_generators(basis, method="brute_force")
        == 4
    )

    # With tables in every layer, the search builds tables for its first set, and again for the
    # set of its automorphism family, after it drops the tables of the first set.
    if completion_tables.forced:
        built = [(held, tables) for held, tables in completion_tables.choices if tables is not held]
        assert len(built) == 2 and built[1][0] is None


def test_searches_ask_for_automorphisms_only_before_costly_layers() -> None:
    """A search asks for automorphisms once, and only before a costly layer or matroid partitioning.

    Finding automorphisms can cost more than an easy search, so a search that finishes before any
    costly layer, without partitioning, does not ask for them.
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


@pytest.fixture
def switch_decisions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, bool]]:
    """Record each consideration of a switch to automorphism images: the best weight and outcome."""
    decisions: list[tuple[int, bool]] = []
    switch = bz._BrouwerZimmermannSearch._switch_to_automorphism_family

    def recording_switch(search: bz._BrouwerZimmermannSearch) -> None:
        switch(search)
        decisions.append((search.best, search.uses_automorphism_family))

    monkeypatch.setattr(
        bz._BrouwerZimmermannSearch, "_switch_to_automorphism_family", recording_switch
    )
    return decisions


def test_a_declined_switch_is_reconsidered_once_the_best_weight_drops(
    switch_decisions: list[tuple[int, bool]],
) -> None:
    """A search reconsiders a declined switch to automorphism images only at a lower best weight.

    Each search may consider the switch before every layer, and the shift of the columns of its
    binary cyclic code by ``step`` positions generates its automorphisms.  The rows that setup finds
    in the [45, 21, 8] code weigh ten.  Before layer two, the images of one set predict more work
    than the two disjoint sets of the search, so it declines to switch.  Layer two finds a word of
    weight eight, and before layer three the images predict less work, so it switches.  In the
    [21, 9, 8] code, setup finds a word of weight eight, so the search, which declines before layer
    two, does not consider the switch again before layer three.  Each search asks for automorphisms
    once, and gets their family from setup once.
    """
    tuning = dataclasses.replace(
        TABLES_DISABLED,
        automorphism_minimum_layer_work=0,
        overlapping_minimum_work=MAX_VALUE,
        partition_work_ratio=MAX_VALUE,
    )
    for coefficients, num_columns, step, expected in (
        ("1001111001000111111110111", 45, 5, [(10, False), (8, True)]),
        ("1111000101011", 21, 7, [(8, False)]),
    ):
        basis = _get_cyclic_basis(coefficients, num_columns)
        labels = _no_labels(basis)
        get_automorphisms = mock.Mock(return_value=np.roll(np.arange(num_columns), -step)[None])
        switch_decisions.clear()
        search = bz._prepare_brouwer_zimmermann_search(
            basis,
            labels,
            cutoff=0,
            weight_divisor=1,
            upper_bound=num_columns + 1,
            get_automorphisms=get_automorphisms,
            tuning=tuning,
        )
        assert isinstance(search, bz._BrouwerZimmermannSearch)
        get_family = search.get_automorphism_family = mock.Mock(
            wraps=search.get_automorphism_family
        )
        while not search.finished:
            search.advance()
        assert (
            search.best == bz.get_distance_brouwer_zimmermann_many([(basis, labels)], cutoff=0) == 8
        )
        assert switch_decisions == expected
        assert get_automorphisms.call_count == get_family.call_count == 1


def test_a_decline_at_the_end_of_setup_is_reconsidered_before_a_costly_layer(
    switch_decisions: list[tuple[int, bool]],
) -> None:
    """A search that declines at the end of setup reconsiders the switch once its best weight drops.

    Setup of this random [45, 16, 9] basis tries matroid partitioning, so it first finds the
    automorphisms, the identity alone, which give no cheaper family.  It partitions, and at the end
    of setup, with rows of weight ten, declines the switch.  Layer two finds a word of weight nine.
    Under a threshold of zero, the search reconsiders before layer three, and declines again.  Under
    a threshold that no layer reaches, it does not reconsider.
    """
    rows = [
        "110100010001010110111011100110110110011100111",
        "011011101011000011001011111000101110001010101",
        "100101000101010111001110101000010011111011111",
        "111101001100110000100000001100010011011111000",
        "110010001101111111000101101001000100101100000",
        "100010011000101010110100010011101000011101011",
        "111111011100010010010001000110001000101010110",
        "000100000000111111010000111010011111011010000",
        "101011110111000101110101000110111101011000111",
        "110100101100111101110010110000001000010110100",
        "111011000100110101100001001000000001011011101",
        "011101111001000011000000010010011001001011010",
        "101011100001101010101101111000111010101101101",
        "010011010011011001010101011111101000010000001",
        "110101000111000111001100101110000000010101011",
        "101111100100100000001101000000010111010110000",
    ]
    basis = np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)
    labels = _no_labels(basis)
    identity = np.arange(basis.shape[1])[None]
    expected_distance = qldpc.codes.distance.get_distance_from_generators(
        basis, method="brute_force"
    )
    get_information_sets = matroid_partition.get_information_sets_with_coverage
    for minimum_layer_work, expected in ((0, [(10, False), (9, False)]), (10**9, [(10, False)])):
        switch_decisions.clear()
        with (
            use_tuning(partition_work_ratio=0, automorphism_minimum_layer_work=minimum_layer_work),
            mock.patch.object(
                matroid_partition, "get_information_sets_with_coverage", wraps=get_information_sets
            ) as partition,
        ):
            distance = bz.get_distance_brouwer_zimmermann_many(
                [(basis, labels)], cutoff=0, get_automorphisms=lambda: identity
            )
        assert distance == expected_distance == 9
        assert partition.called
        assert switch_decisions == expected


def test_a_late_switch_weighs_the_layers_left_in_the_current_family(
    tune: Tune, switch_decisions: list[tuple[int, bool]]
) -> None:
    """A search declines a switch that would cost more than the layers that its family has left.

    In this [35, 15, 8] binary cyclic code, the shift of the columns by seven positions generates
    the automorphisms.  With the threshold of the switch at the work of layer three, the search
    first considers the switch before that layer, after its two disjoint sets have completed layers
    one and two and found a word of weight eight.  Layer three of one of these sets then certifies
    that weight, at 455 candidate/set evaluations, while the images of one set would scan layers one
    to three again, at 575.  So the search declines, though the whole search of the disjoint sets,
    at 695 evaluations, predicts more work than the images.
    """
    tune(
        automorphism_minimum_layer_work=math.comb(15, 3),
        overlapping_minimum_work=MAX_VALUE,
        partition_work_ratio=MAX_VALUE,
    )
    basis = _get_cyclic_basis("111100101101100010011", 35)
    labels = _no_labels(basis)
    automorphisms = np.roll(np.arange(35), -7)[None]
    distance = bz.get_distance_brouwer_zimmermann_many(
        [(basis, labels)], cutoff=0, get_automorphisms=lambda: automorphisms
    )
    assert distance == bz.get_distance_brouwer_zimmermann_many([(basis, labels)], cutoff=0) == 8
    assert switch_decisions == [(8, False)]


def test_searches_switch_to_automorphism_images_instead_of_partitioning(tune: Tune) -> None:
    """A search skips partitioning, and switches at once, if automorphism images do no more work.

    The translations of this multivariate bicycle code give an automorphism family that does no more
    work than any partitioned family could, so the search builds that family once, and switches to
    it at once, under a threshold that none of its layers reaches.  With the identity alone, which
    gives no cheaper family, or if none are found, the same search partitions, and does not switch.
    Under a threshold above the cap of work estimates, it asks for no automorphisms.  A search that
    partitions after finding automorphisms still considers the switch at the end of setup.
    ``test_a_search_through_the_costly_stages_finds_the_distance`` partitions without automorphisms.
    """
    tune(partition_work_ratio=0)
    code = qldpc.codes.MBCode({x: 12}, x**10 + x**4 + x**2 + x, x**9 + x**8 + x**3 + x)
    problem = get_sector_problem(code, Pauli.X)
    identity = np.arange(len(code))[None]
    expected = qldpc.codes.distance.get_distance_from_generators(
        code.get_logical_ops(Pauli.X), code.get_stabilizer_ops(Pauli.X), method="brute_force"
    )
    get_information_sets = matroid_partition.get_information_sets_with_coverage
    switch = bz._BrouwerZimmermannSearch._switch_to_automorphism_family

    # The switch spy records whether each search that considers a switch makes it.  A search that
    # partitions in setup may finish there, and consider none.
    switched: list[bool] = []

    def _spy_switch(search: bz._BrouwerZimmermannSearch) -> None:
        switch(search)
        switched.append(search.uses_automorphism_family)

    for get_automorphisms, minimum_layer_work, partitions in (
        (code._get_qudit_automorphisms, 10**9, False),
        (code._get_qudit_automorphisms, MAX_VALUE, True),
        (lambda: identity, 0, True),
        (lambda: None, 0, True),
    ):
        recorder = mock.Mock(wraps=get_automorphisms)
        switched.clear()
        with (
            use_tuning(automorphism_minimum_layer_work=minimum_layer_work),
            mock.patch.object(
                matroid_partition, "get_information_sets_with_coverage", wraps=get_information_sets
            ) as partition,
            mock.patch.object(
                bz._BrouwerZimmermannSearch, "_switch_to_automorphism_family", _spy_switch
            ),
        ):
            distance = bz.get_distance_brouwer_zimmermann_many(
                [problem], cutoff=0, get_automorphisms=recorder
            )
        assert distance == expected
        assert partition.called == partitions
        assert recorder.call_count == (minimum_layer_work != MAX_VALUE)
        assert any(switched) != partitions

    # The identity gives this [21, 8, 6] basis no cheaper family, so its search partitions, and then
    # considers and declines the switch at the end of setup.
    switched.clear()
    with (
        use_tuning(automorphism_minimum_layer_work=10**9),
        mock.patch.object(
            matroid_partition, "get_information_sets_with_coverage", wraps=get_information_sets
        ) as partition,
        mock.patch.object(
            bz._BrouwerZimmermannSearch, "_switch_to_automorphism_family", _spy_switch
        ),
    ):
        assert (
            qldpc.codes.distance.get_distance_from_generators(
                _get_costly_basis(), cutoff=0, automorphisms=np.arange(21)[None]
            )
            == 6
        )
    assert partition.called
    assert switched == [False]


def test_a_switch_at_the_end_of_setup_can_certify_the_distance(tune: Tune) -> None:
    """A search whose switch at the end of setup certifies a lighter word returns its weight.

    In the X sector of this bivariate bicycle code, the rows of the automorphism family include a
    word lighter than any that setup had found, and its certificate then proves that weight, so the
    search finishes in setup.
    """
    tune(partition_work_ratio=0)
    code = qldpc.codes.BBCode({x: 6, y: 2}, x**3 * y + x**2 * y + x, x**5 + x**3 * y + x)
    expected = qldpc.codes.distance.get_distance_from_generators(
        code.get_logical_ops(Pauli.X), code.get_stabilizer_ops(Pauli.X), method="brute_force"
    )
    distance = bz.get_distance_brouwer_zimmermann_many(
        [get_sector_problem(code, Pauli.X)],
        cutoff=0,
        get_automorphisms=code._get_qudit_automorphisms,
    )
    assert distance == expected


@pytest.mark.parametrize("partition_work_ratio", [MAX_VALUE, 0])
@pytest.mark.usefixtures("switch_before_any_layer")
def test_rows_of_an_automorphism_family_can_reach_the_cutoff(
    tune: Tune, partition_work_ratio: int
) -> None:
    """The rows of a scanned set with images can end a search at the cutoff.

    In the Z sector of this [[36, 4, 4]] bivariate bicycle code, the rows of the disjoint family,
    which the search keeps, weigh more than six, and a row of the scanned set weighs six.  With a
    cutoff of six, the search returns that weight, an upper bound on the distance, rather than
    scanning on to a word of weight four.  It does so whether it switches before its first layer,
    or at the end of setup, after asking for automorphisms to decide whether to partition.
    """
    tune(overlapping_minimum_work=MAX_VALUE, partition_work_ratio=partition_work_ratio)
    code = qldpc.codes.BBCode({x: 6, y: 3}, 1 + x + x**3 * y, 1 + y + x)
    problem = get_sector_problem(code, Pauli.Z)
    assert bz.get_distance_brouwer_zimmermann_many([problem], cutoff=0) == 4
    distance = bz.get_distance_brouwer_zimmermann_many(
        [problem], cutoff=6, get_automorphisms=code._get_qudit_automorphisms
    )
    assert distance == 6


@pytest.mark.usefixtures("switch_before_any_layer")
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


def _get_costly_basis() -> npt.NDArray[np.uint8]:
    """A basis of a [21, 8, 6] code whose search reaches matroid partitioning when tuned to."""
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
    return np.array([[int(bit) for bit in row] for row in rows], dtype=np.uint8)


def _no_labels(basis: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Quotient labels with no columns, which exclude no subcode."""
    return np.empty((len(basis), 0), dtype=np.uint8)


def _get_cyclic_basis(coefficients: str, num_columns: int) -> npt.NDArray[np.uint8]:
    """A basis of a binary cyclic code: the shifts of the coefficients of its generator polynomial.

    ``coefficients`` lists them from the constant term up, and the polynomial must divide
    ``x**num_columns - 1``, so that shifting the columns maps the code onto itself.
    """
    row = np.zeros(num_columns, dtype=np.uint8)
    row[: len(coefficients)] = [int(bit) for bit in coefficients]
    return np.array([np.roll(row, shift) for shift in range(num_columns - len(coefficients) + 1)])


def _get_family(
    generators: npt.NDArray[np.uint8], column_sets: list[list[int]]
) -> tuple[basis_families.BasisFamily, certificates.Certificate]:
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
    family = basis_families.BasisFamily(
        np.array(reduced),
        np.empty((len(column_sets), num_rows, 0), dtype=np.uint8),
        pivots,
        ranks,
        1,
        1,
    )
    return family, certificates.Certificate(num_rows, tuple(ranks.tolist()))


@pytest.mark.usefixtures("switch_before_any_layer")
@pytest.mark.parametrize("completion_tables", ["forced"], indirect=True)
def test_completion_tables_keep_distances_exact(
    pytestconfig: pytest.Config, completion_tables: CompletionRecord
) -> None:
    """Searches that skip hopeless completions agree with independent oracles.

    Every layer uses tables, and every search below skips supports.  The tuning keeps the searches
    of this [[36, 4, 4]] bivariate bicycle code from building an overlapping family, which would
    hold a word of weight four and end them before any layer.  Before layer two, they switch from
    the disjoint family to an automorphism family, the rows of whose scanned set weigh at least six
    in each sector, so a scan must find a word of weight four, with or without an upper bound above
    it.  In the X
    sector of a [[72, 4, 8]] code, with its coordinates relabelled at random and its automorphisms
    relabelled alike, the search must scan to layer four.  A general Pauli problem checks that
    searched words weigh twice their number of nontrivial qubits.
    """

    def assert_skips(distance: int) -> int:
        assert completion_tables.skipped > skipped
        return distance

    rng = np.random.default_rng(pytestconfig.getoption("randomly_seed"))
    code = qldpc.codes.BBCode({x: 6, y: 3}, 1 + x + x**3 * y, 1 + y + x)
    for pauli in PAULIS_XZ:
        expected = qldpc.codes.distance.get_distance_from_generators(
            code.get_logical_ops(pauli),
            code.get_stabilizer_ops(pauli),
            method="brute_force",
        )
        assert expected is not qldpc.codes.NO_DISTANCE
        for upper_bound in (None, expected + 1):
            skipped = completion_tables.skipped
            # One sixteen-column block covers the tails of this code, which makes its bounds exact,
            # so a bound that is too high changes the distance found.
            with use_tuning(completion_bound_block_bits=16, overlapping_minimum_work=MAX_VALUE):
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
    quantum_distance = qldpc.codes.distance.get_distance_from_generators(
        logicals, stabilizers, symplectic=True, cutoff=0, method="brouwer_zimmermann"
    )
    assert quantum_distance is not qldpc.codes.NO_DISTANCE
    assert assert_skips(quantum_distance) == qldpc.codes.distance.get_distance_from_generators(
        logicals, stabilizers, symplectic=True, method="brute_force"
    )


def test_sets_that_are_not_information_sets_count_pivot_ones_only_below_their_rank() -> None:
    """A set of rank zero has no pivot ones, so its words weigh only their packed rows.

    Its two rows weigh three and two, and their sum weighs one.  A scan that counted a pivot one for
    each row, as for an information set, would find no word lighter than two.
    """
    search = bz._BrouwerZimmermannSearch(
        affine_instance.AffineInstance(
            generators=np.array([[[7], [6]]], dtype=np.uint64),
            labels=np.array([[[1], [2]]], dtype=np.uint64),
            widths=np.array([1], dtype=np.int64),
            pivots=np.full((1, 2), -1, dtype=np.int64),
            certificate=certificates.Certificate(2, (0,)),
        ),
        cutoff=-1,
        batch_size=1,
        tuning=TABLES_DISABLED,
        best=10,
        lower_bound=0,
    )
    assert search.advance() == 2
    assert search.advance() == 1


@pytest.mark.usefixtures("switch_before_any_layer")
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


@pytest.mark.parametrize("completion_tables", ["forced"], indirect=True)
@pytest.mark.parametrize("mixed", [False, True])
def test_layers_of_several_sets_scan_each_set_with_its_tables(
    completion_tables: CompletionRecord, monkeypatch: pytest.MonkeyPatch, mixed: bool
) -> None:
    """Searches that scan the sets of a layer one at a time, each with tables or none, are exact.

    This random code with 10 rows and 36 columns has three disjoint information sets and a partial
    set, so a search scans several sets in layer two, and the first set again in layer three.  The
    oracle enumerates every nonzero word.  The tables of each scan must be those of the rows that it
    scans, since tables of other rows could skip lighter words, also when the first set returns
    after the others.  The forced tuning gives every information set tables, unless the test
    declines every other choice of tables, so that a set that declines them must scan without the
    tables of the set before it.
    """
    select = completion_bounds.select_completion_tables
    scan_set = bz._BrouwerZimmermannSearch._scan_information_set
    scan = affine_search.scan_information_set
    num_choices, scanned_index, returned_to_first_set = 0, 0, False
    later_scans = {"with tables": 0, "without tables": 0}
    returns_with_tables = 0

    def mixing_select(*args: Any, **kwargs: Any) -> completion_bounds.CompletionTables | None:
        nonlocal num_choices
        num_choices += 1
        tables = select(*args, **kwargs)
        return None if mixed and num_choices % 2 else tables

    def recording_scan_set(search: bz._BrouwerZimmermannSearch, index: int, weight: int) -> bool:
        nonlocal scanned_index, returned_to_first_set
        returned_to_first_set = index == 0 and scanned_index > 0
        scanned_index = index
        return scan_set(search, index, weight)

    def checking_scan(*args: Any) -> tuple[int, int, int, int, int]:
        nonlocal returns_with_tables
        rows, widths, tables, block_bits = args[0], args[3], args[11], args[13]
        if scanned_index:
            later_scans["with tables" if len(tables) else "without tables"] += 1
        if len(tables):
            returns_with_tables += returned_to_first_set
            own = completion_bounds.CompletionTables.build(rows[0, :, : widths[0]], block_bits)
            assert np.array_equal(tables, own.tables)
        return scan(*args)

    monkeypatch.setattr(completion_bounds, "select_completion_tables", mixing_select)
    monkeypatch.setattr(bz._BrouwerZimmermannSearch, "_scan_information_set", recording_scan_set)
    monkeypatch.setattr(affine_search, "scan_information_set", checking_scan)
    generators = get_random_full_rank_matrix(np.random.default_rng(4), 10, 36)
    distance = bz.get_distance_brouwer_zimmermann_many(
        [(generators, _no_labels(generators))], cutoff=0
    )
    assert distance == qldpc.codes.distance.get_distance_from_generators(
        generators, cutoff=0, method="brute_force"
    )
    assert completion_tables.skipped
    assert later_scans["with tables"]
    assert bool(later_scans["without tables"]) == mixed
    assert returns_with_tables


@pytest.mark.parametrize("cutoff", [0, 5])
def test_a_layer_of_several_sets_stops_once_its_sets_certify_the_best_weight(cutoff: int) -> None:
    """A search stops a layer after the set whose completion of it certifies the best weight.

    The two sets of this code are disjoint information sets, the first three columns and the next
    three, and their rows weigh six, seven, and six, and six, nine, and six.  After both sets
    complete layer one, the search certifies weight four.  In layer two, it needs both sets to
    certify its best weight of six, but the first set finds a word of weight five, the sum of the
    first two rows.  Its completion of the layer then certifies weight five, so the search scans no
    other set.  With a cutoff of five, the scan stops at the word of weight five, before it
    certifies more than weight four.
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
    search = bz._BrouwerZimmermannSearch(
        affine_instance.get_distance_instance(family, certificate),
        cutoff=cutoff,
        batch_size=1 << 20,
        tuning=TABLES_DISABLED,
        best=6,
        lower_bound=certificate.lower_bound(1),
        next_weight=2,
    )
    scanned: list[int] = []
    scan_set = bz._BrouwerZimmermannSearch._scan_information_set

    def recording_scan_set(search: bz._BrouwerZimmermannSearch, index: int, weight: int) -> bool:
        scanned.append(index)
        return scan_set(search, index, weight)

    with mock.patch.object(
        bz._BrouwerZimmermannSearch, "_scan_information_set", recording_scan_set
    ):
        assert search.advance() == 5
    assert search.finished and search.lower_bound == (4 if cutoff else 5)
    assert scanned == [0]


def test_a_layer_with_a_set_that_is_not_an_information_set_is_exact() -> None:
    """A layer that scans an information set and a set of deficient rank finds the lightest word.

    The second set of this code, on columns three to five, has rank two, so its third row has no
    pivot.  The lightest rows of both sets weigh five, and the lightest word, which the second set
    finds in layer two, weighs four.  A scan that counted a pivot one for each row would overstate
    the weights of the words of the second set, and the search would certify weight five.  The
    tuning forces tables, so the search holds the tables of the first set when it scans the second,
    which must scan without them.
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
        == qldpc.codes.distance.get_distance_from_generators(
            generators, cutoff=0, method="brute_force"
        )
        == 4
    )


def test_searches_hold_the_tables_of_one_set_at_a_time(
    tune: Tune, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A search frees the tables of one set before it builds the tables of another.

    With tables forced, the search of this random code with 10 rows and 36 columns scans several
    sets in layer two, and the first set again in layer three.  When it builds tables, no tables
    that it built before are alive, so it never holds the tables of more than one set.
    """
    tune(
        completion_bound_minimum_layer_work=0,
        completion_bound_max_cost_percent=MAX_VALUE,
        completion_bound_block_bits=8,
        completion_bound_samples=DEFAULT_TUNING.completion_bound_min_samples,
    )
    build = completion_bounds.CompletionTables.build
    built: list[weakref.ref[completion_bounds.CompletionTables]] = []

    def checking_build(
        rows: npt.NDArray[np.uint64], block_bits: int
    ) -> completion_bounds.CompletionTables:
        gc.collect()
        assert all(ref() is None for ref in built)
        tables = build(rows, block_bits)
        built.append(weakref.ref(tables))
        return tables

    monkeypatch.setattr(completion_bounds.CompletionTables, "build", staticmethod(checking_build))
    generators = get_random_full_rank_matrix(np.random.default_rng(4), 10, 36)
    distance = bz.get_distance_brouwer_zimmermann_many(
        [(generators, _no_labels(generators))], cutoff=0
    )
    assert distance == qldpc.codes.distance.get_distance_from_generators(
        generators, cutoff=0, method="brute_force"
    )
    assert len(built) > 1


@pytest.mark.usefixtures("switch_before_any_layer")
def test_searches_take_completion_limits_from_their_tuning(
    tune: Tune, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Searches use completion tables within the limits of their tuning.

    The tuning keeps the searches from building an overlapping family, which would hold a word of
    weight four and end them before any layer.  Every information set of either sector of this
    [[36, 4, 4]] code has 20 rows and 16 tail columns, so tables of one sixteen-column block take
    19 * 2**16 bytes.  A tuning that forces tables uses sixteen-column blocks.  With one byte less
    memory, a limit of zero cost, or the largest threshold of layer work, it uses none.
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
    tune(
        overlapping_minimum_work=MAX_VALUE,
        completion_bound_minimum_layer_work=0,
        completion_bound_max_cost_percent=MAX_VALUE,
    )
    for limits, expected_block_bits in (
        ({}, {16}),
        ({"completion_bound_max_table_bytes": 19 * 2**16 - 1}, set()),
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


def test_many_problems_need_a_problem_or_an_upper_bound() -> None:
    """Without problems, the result is the known upper bound, which must then be given."""
    with pytest.raises(ValueError, match="at least one problem or an upper bound"):
        bz.get_distance_brouwer_zimmermann_many([], cutoff=0)
    assert bz.get_distance_brouwer_zimmermann_many([], cutoff=0, upper_bound=3) == 3
